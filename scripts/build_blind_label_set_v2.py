#!/usr/bin/env python3
"""Export a fresh prediction-blind label set and a self-contained annotator.

The original 24 calibration and 12 held-out labels per camera have all been
inspected repeatedly, so they are development data now and cannot score a new
method. This builds a second, larger, genuinely unseen set: 40 Run A query
frames per camera, sampled deterministically from ordinals the original 36
anchors never touched.

Three sampling rules follow directly from the data audit:

* Idle frames are excluded. Run A is parked for its first 173 frames and its
  last 57; those frames are all the same place, so a one-to-one frame
  correspondence there is meaningless even where one exists, and including them
  would inflate or deflate coverage depending on which way you guessed.
* Queries are stratified across the route so the sample cannot concentrate in
  easy corridors.
* Each query carries a *linear-progress* start position only. That prior is
  method-independent - it is arithmetic on the route boundaries, not a model
  output - so it cannot leak a prediction. It is deliberately weak: the measured
  Run A to Run B offset drifts across a 124-frame span, so the annotator is
  expected to scrub well away from it, and the annotator can reach any Run B
  route frame.

The emitted page shows the Run A query frame beside a scrubbable Run B
filmstrip and nothing else. There is no prediction, no score, no baseline path,
and no calibration row anywhere in the payload.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

import frame_service


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_blind_v2"
ANCHOR_FILE = ROOT / "outputs" / "task2_keyframes" / "anchor_candidates.csv"
BOUNDARY_FILE = ROOT / "outputs" / "task2_keyframes" / "route_phase_boundaries.csv"

CAMERAS = ("cam0", "cam5")
QUERIES_PER_CAMERA = 40
RANDOM_SEED = 20260901

# Keep new queries clear of the original anchors so the two sets stay
# independent rather than sampling the same places twice.
MINIMUM_DISTANCE_FROM_OLD_ANCHOR = 10
MINIMUM_DISTANCE_BETWEEN_QUERIES = 18

# Run A idle segments measured from per-frame image motion. The existing route
# detector already places the departure boundary at 180; the trailing boundary
# is not recorded anywhere, so it is stated here.
RUN_A_ROUTE_TAIL_EXCLUSIVE = frame_service.RUN_A_ROUTE_TAIL_EXCLUSIVE

QUERY_JPEG_QUALITY = 3      # ffmpeg -q:v, lower is better
FILMSTRIP_JPEG_QUALITY = 4
FILMSTRIP_WIDTH = 960
FILMSTRIP_HEIGHT = 720

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise RuntimeError("ffmpeg is required but was not found on PATH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing output.")
    parser.add_argument(
        "--queries-per-camera", type=int, default=QUERIES_PER_CAMERA,
    )
    parser.add_argument(
        "--skip-filmstrip",
        action="store_true",
        help="Only rewrite the manifest and page, reusing extracted frames.",
    )
    return parser.parse_args()


def route_boundaries() -> dict[str, dict[str, int]]:
    with BOUNDARY_FILE.open(newline="") as handle:
        rows = {row["run"]: row for row in csv.DictReader(handle)}
    return {
        run: {
            "route_start": int(rows[run]["route_start"]),
            "stationary_end": int(rows[run]["stationary_end"]),
        }
        for run in ("runA", "runB")
    }


def existing_anchor_frames(camera: str) -> set[int]:
    with ANCHOR_FILE.open(newline="") as handle:
        return {
            int(row["runA_frame"])
            for row in csv.DictReader(handle)
            if row["camera_id"] == camera
        }


def frame_counts() -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for run in ("runA", "runB"):
        for camera in CAMERAS:
            stream_scan = frame_service.hevc_bitstream.scan(
                frame_service.source_path(run, camera)
            )
            counts[(run, camera)] = stream_scan.picture_count
    return counts


def sample_queries(
    camera: str,
    count: int,
    run_a_route_start: int,
    run_a_end_exclusive: int,
    blocked: set[int],
) -> list[int]:
    """Draw `count` stratified Run A ordinals, deterministic for a fixed seed."""

    rng = np.random.default_rng(RANDOM_SEED + (0 if camera == "cam0" else 1))
    edges = np.linspace(run_a_route_start, run_a_end_exclusive, count + 1).astype(int)
    chosen: list[int] = []
    for index in range(count):
        low, high = int(edges[index]), int(edges[index + 1])
        candidates = [
            ordinal
            for ordinal in range(low, max(low + 1, high))
            if all(abs(ordinal - other) >= MINIMUM_DISTANCE_FROM_OLD_ANCHOR for other in blocked)
            and all(abs(ordinal - other) >= MINIMUM_DISTANCE_BETWEEN_QUERIES for other in chosen)
        ]
        if not candidates:
            # Fall back to the stratum midpoint rather than silently dropping a
            # query and returning fewer than the requested count.
            candidates = [(low + high) // 2]
        chosen.append(int(rng.choice(candidates)))
    return sorted(chosen)


def linear_progress_prior(
    run_a_frame: int,
    run_a_start: int,
    run_a_end: int,
    run_b_start: int,
    run_b_end: int,
) -> int:
    """Map route progress linearly. Method-independent, and known to be weak."""

    span = max(run_a_end - run_a_start, 1)
    offset = (run_a_frame - run_a_start) / span
    return int(round(run_b_start + offset * (run_b_end - run_b_start)))


def select_expression(ordinals: list[int]) -> str:
    """Build a compact FFmpeg select expression for a set of ordinals.

    Contiguous stretches collapse into `between(n,a,b)`. A filmstrip of 2,500
    consecutive frames would otherwise produce an expression long enough for
    FFmpeg's parser to reject outright.
    """

    terms: list[str] = []
    start = previous = ordinals[0]
    for ordinal in ordinals[1:] + [None]:
        if ordinal is not None and ordinal == previous + 1:
            previous = ordinal
            continue
        if start == previous:
            terms.append(f"eq(n\\,{start})")
        else:
            terms.append(f"between(n\\,{start}\\,{previous})")
        if ordinal is not None:
            start = previous = ordinal
    return "+".join(terms)


def extract_frames(
    run: str,
    camera: str,
    ordinals: list[int],
    target_dir: Path,
    width: int | None,
    height: int | None,
    quality: int,
) -> None:
    """Write the requested display ordinals as JPEGs named by ordinal."""

    target_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted(set(ordinals))
    if not wanted:
        return
    select = select_expression(wanted)
    scale = ""
    if width and height:
        scale = (
            f",scale={width}:{height}:flags=lanczos"
            f":in_color_matrix={frame_service.COLOUR_MATRIX}"
            f":in_range={frame_service.COLOUR_INPUT_RANGE}"
            f":out_range={frame_service.COLOUR_OUTPUT_RANGE}"
        )
    with tempfile.TemporaryDirectory(dir=target_dir.parent) as staging:
        command = [
            FFMPEG, "-v", "error", "-nostdin", "-hwaccel", "none",
            "-i", str(frame_service.source_path(run, camera)),
            "-map", "0:v:0",
            "-vf", f"select='{select}'{scale}",
            "-fps_mode", "passthrough",
            "-q:v", str(quality),
            str(Path(staging) / "sel_%06d.jpg"),
        ]
        subprocess.run(command, check=True)
        produced = sorted(Path(staging).glob("sel_*.jpg"))
        if len(produced) != len(wanted):
            raise ValueError(
                f"{run}/{camera}: expected {len(wanted)} extracted frames, got {len(produced)}"
            )
        for ordinal, source in zip(wanted, produced):
            os.replace(source, target_dir / f"{ordinal:06d}.jpg")


# The alignment convention starts at zero and is set by the annotator, not
# asserted here. CAM0's camera pose is measurably unchanged between runs (ego
# band correlates at zero shift, sharply). CAM5's could not be measured at all:
# three independent attempts failed - raw ego-band correlation slid to the
# search boundary, the same correlation with rain masked out had no peak, and
# the fisheye circle is wider than the frame so its left and right edges are not
# visible. So the tool offers the control and records what was chosen, rather
# than pretending to know the number.
POSE_OFFSET_PX_AT_FILMSTRIP = {"cam0": 0, "cam5": 0}


def query_set_sha256(cameras: dict[str, dict[str, object]]) -> str:
    """A hash of exactly which Run A frames were asked, and under what seed.

    The seal cites this, so it is checkable that the sealed labels answer the
    queries that were exported - not a regenerated or edited set.
    """

    import hashlib

    canonical = {
        camera: sorted(int(q["runAFrame"]) for q in entry["queries"])
        for camera, entry in cameras.items()
    }
    payload = json.dumps({"seed": RANDOM_SEED, "queries": canonical}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_page(camera: str, queries: list[dict[str, object]], run_b_frames: list[int]) -> str:
    payload = {
        "camera": camera,
        "queries": queries,
        "runBFrames": run_b_frames,
        "poseOffsetPx": POSE_OFFSET_PX_AT_FILMSTRIP.get(camera, 0),
        "builtAtUtc": datetime.now(timezone.utc).isoformat(),
    }
    return _PAGE_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))


def main() -> None:
    arguments = parse_args()
    if OUTPUT_DIR.exists() and not arguments.force and not arguments.skip_filmstrip:
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists; pass --force to rebuild. Rebuilding "
            "changes which frames are queried and invalidates any labelling in progress."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    boundaries = route_boundaries()
    counts = frame_counts()
    run_a_start = boundaries["runA"]["route_start"]
    run_b_start = boundaries["runB"]["route_start"]

    manifest_cameras = {}
    for camera in CAMERAS:
        run_a_end = RUN_A_ROUTE_TAIL_EXCLUSIVE
        run_b_end = counts[("runB", camera)] - 1
        blocked = existing_anchor_frames(camera)
        query_frames = sample_queries(
            camera, arguments.queries_per_camera, run_a_start, run_a_end, blocked
        )

        queries = []
        for index, run_a_frame in enumerate(query_frames):
            queries.append(
                {
                    "queryId": f"{camera}_v2_{index:02d}",
                    "runAFrame": run_a_frame,
                    "linearPriorRunBFrame": linear_progress_prior(
                        run_a_frame, run_a_start, run_a_end, run_b_start, run_b_end
                    ),
                }
            )

        run_b_frames = list(range(run_b_start, run_b_end + 1))
        camera_dir = OUTPUT_DIR / camera
        if not arguments.skip_filmstrip:
            print(f"[{camera}] extracting {len(query_frames)} Run A query frames at native resolution")
            extract_frames(
                "runA", camera, query_frames, camera_dir / "runA",
                None, None, QUERY_JPEG_QUALITY,
            )
            print(f"[{camera}] extracting {len(run_b_frames)} Run B filmstrip frames at {FILMSTRIP_WIDTH}x{FILMSTRIP_HEIGHT}")
            extract_frames(
                "runB", camera, run_b_frames, camera_dir / "runB",
                FILMSTRIP_WIDTH, FILMSTRIP_HEIGHT, FILMSTRIP_JPEG_QUALITY,
            )

        page = build_page(camera, queries, run_b_frames)
        (OUTPUT_DIR / f"annotate_{camera}.html").write_text(page)
        manifest_cameras[camera] = {
            "queries": queries,
            "runA_route_start": run_a_start,
            "runA_route_end_exclusive": run_a_end,
            "runB_route_start": run_b_start,
            "runB_route_end": run_b_end,
            "runB_filmstrip_frames": len(run_b_frames),
        }

    # A page-only rebuild must not rewrite the export time: that timestamp is
    # part of the evidence that the queries predate the method freeze. Keep the
    # original and record the regeneration separately.
    manifest_path = OUTPUT_DIR / "blind_v2_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    now = datetime.now(timezone.utc).isoformat()
    built_at = previous.get("built_at_utc", now) if arguments.skip_filmstrip else now

    query_set_digest = query_set_sha256(manifest_cameras)

    manifest = {
        "schema_version": 2,
        "label_set": "blind_v2",
        "built_at_utc": built_at,
        "pages_regenerated_at_utc": now if arguments.skip_filmstrip else None,
        "query_set_sha256": query_set_digest,
        "random_seed": RANDOM_SEED,
        "queries_per_camera": arguments.queries_per_camera,
        "sampling_rules": {
            "stratified": "one query per equal-width route stratum",
            "excluded_idle": (
                f"Run A frames outside [{run_a_start}, {RUN_A_ROUTE_TAIL_EXCLUSIVE}) are "
                "parked and carry no travelling correspondence"
            ),
            "minimum_distance_from_original_anchor": MINIMUM_DISTANCE_FROM_OLD_ANCHOR,
            "minimum_distance_between_queries": MINIMUM_DISTANCE_BETWEEN_QUERIES,
        },
        "prior_policy": (
            "linear route-progress arithmetic only; contains no model output, "
            "no baseline path, and no previous prediction"
        ),
        "blindness": {
            "model_columns_present": False,
            "predictions_present": False,
            "previous_labels_present": False,
        },
        "cameras": manifest_cameras,
    }
    (OUTPUT_DIR / "blind_v2_manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"Wrote {OUTPUT_DIR.relative_to(ROOT)}")
    for camera in CAMERAS:
        print(f"  annotate_{camera}.html   {arguments.queries_per_camera} queries")
    print()
    print("Open each page in a browser, label every query, then use its Download")
    print("button and import the two CSVs with scripts/import_blind_labels_v2.py.")


_PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind labelling</title>
<style>
:root{
  --bg:#11161a; --panel:#1a2228; --line:#2c383f; --ink:#e8eef1; --muted:#8fa3ad;
  --accent:#5ac8d8; --ok:#5fc98f; --warn:#e0a94b; --crit:#f0796c;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;gap:16px;align-items:center;flex-wrap:wrap;
  padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);
  position:sticky;top:0;z-index:10}
h1{font-size:15px;margin:0;font-weight:650;letter-spacing:.01em}
.chip{font:600 11px/1 ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;
  padding:5px 9px;border:1px solid var(--line);border-radius:3px;color:var(--muted)}
.chip b{color:var(--ink);font-weight:700}
button{font:600 13px/1 inherit;padding:8px 13px;border-radius:4px;cursor:pointer;
  border:1px solid var(--line);background:#243038;color:var(--ink)}
button:hover{border-color:var(--accent)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.primary{background:var(--accent);color:#06222a;border-color:var(--accent)}
button.danger{background:#3a1f1c;border-color:#5c2f29;color:var(--crit)}
main{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:5px;overflow:hidden;
  display:flex;flex-direction:column}
.pane h2{margin:0;padding:9px 13px;font-size:11px;letter-spacing:.11em;text-transform:uppercase;
  color:var(--muted);border-bottom:1px solid var(--line);font-weight:700;
  display:flex;justify-content:space-between;align-items:center}
.pane h2 span{color:var(--ink);font:700 13px/1 ui-monospace,monospace;letter-spacing:0}
.imgwrap{background:#000;display:flex;align-items:center;justify-content:center;min-height:340px;
  position:relative;overflow:hidden}
#overlayImage{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain;
  opacity:0;pointer-events:none}
#guides{position:absolute;inset:0;pointer-events:none;display:none}
#guides.on{display:block}
#guides i{position:absolute;top:0;bottom:0;width:1px;background:rgba(90,200,216,.55)}
#guides i.mid{background:rgba(224,169,75,.8);width:2px}
select{background:#0d1418;color:var(--ink);border:1px solid var(--line);border-radius:3px;
  padding:5px 7px;font:600 12px/1 inherit}
.checkline{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted)}
img{max-width:100%;max-height:62vh;display:block}
.controls{padding:11px 13px;border-top:1px solid var(--line);display:flex;flex-direction:column;gap:9px}
.row{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
.brightness-row label{min-width:108px;color:var(--muted);font-size:12px;font-weight:650}
.brightness-row output{min-width:44px;color:var(--ink);font:700 12px/1 ui-monospace,monospace}
input[type=range]{flex:1;min-width:180px;accent-color:var(--accent)}
input[type=number]{width:84px;background:#0d1418;color:var(--ink);
  border:1px solid var(--line);border-radius:3px;padding:5px 7px;font:600 13px/1 ui-monospace,monospace}
.hint{color:var(--muted);font-size:12px}
kbd{background:#0d1418;border:1px solid var(--line);border-bottom-width:2px;border-radius:3px;
  padding:1px 5px;font:600 11px/1.5 ui-monospace,monospace;color:var(--ink)}
.status{padding:9px 16px;border-top:1px solid var(--line);background:var(--panel);
  display:flex;gap:14px;flex-wrap:wrap;align-items:center;position:sticky;bottom:0}
.pill{font:600 11px/1 ui-monospace,monospace;padding:5px 9px;border-radius:3px}
.pill.done{background:#12301f;color:var(--ok)}
.pill.todo{background:#372a12;color:var(--warn)}
.pill.nomatch{background:#3a1c19;color:var(--crit)}
.grid{display:flex;gap:3px;flex-wrap:wrap;padding:9px 16px;background:var(--panel);
  border-top:1px solid var(--line)}
.cell{width:22px;height:22px;border-radius:2px;border:1px solid var(--line);cursor:pointer;
  font:600 9px/20px ui-monospace,monospace;text-align:center;color:var(--muted);background:#0d1418}
.cell.done{background:#12301f;color:var(--ok);border-color:#1e5236}
.cell.nomatch{background:#3a1c19;color:var(--crit);border-color:#5c2f29}
.cell.active{outline:2px solid var(--accent);outline-offset:1px}
@media (max-width:720px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Blind labelling — <span id="cameraName"></span></h1>
  <span class="chip">query <b id="queryPos"></b></span>
  <span class="chip">run A frame <b id="runAFrame"></b></span>
  <span class="chip">labelled <b id="doneCount"></b></span>
  <button id="prevBtn">← Prev</button>
  <button id="nextBtn">Next →</button>
  <button id="saveBtn" class="primary">Download CSV</button>
</header>

<main>
  <div class="pane">
    <h2>Run A · query <span id="runALabel"></span></h2>
    <div class="imgwrap"><img id="runAImage" alt="Run A query frame"></div>
    <div class="controls"><div class="hint">This is the place to find. The Run B panel starts at a linear route-progress guess, which is deliberately weak — the true offset drifts by up to 124 frames across the route, so scrub freely.</div></div>
  </div>

  <div class="pane">
    <h2>Run B · candidate <span id="runBLabel"></span></h2>
    <div class="imgwrap" id="runBWrap">
      <img id="runBImage" alt="Run B candidate frame">
      <img id="overlayImage" alt="Run A overlaid for alignment">
      <div id="guides"></div>
    </div>
    <div class="controls">
      <div class="row">
        <input type="range" id="scrub" min="0" max="0" step="1">
        <input type="number" id="frameInput">
      </div>
      <div class="row brightness-row">
        <label for="runBBrightness">Run B brightness</label>
        <input type="range" id="runBBrightness" min="50" max="250" step="5" value="100">
        <output id="runBBrightnessValue" for="runBBrightness">100%</output>
        <button id="brightnessResetBtn" type="button">Reset brightness</button>
      </div>
      <div class="row">
        <label for="compareMode">Compare</label>
        <select id="compareMode">
          <option value="side">Side by side</option>
          <option value="blend">Blend 50/50</option>
          <option value="diff">Difference</option>
          <option value="blink">Blink A/B</option>
        </select>
        <label for="poseOffset">Pose fix</label>
        <input type="range" id="poseOffset" min="-160" max="160" step="1" value="0">
        <output id="poseOffsetValue" for="poseOffset">0 px</output>
        <label class="checkline"><input type="checkbox" id="showGuides"> guides</label>
      </div>
      <div class="row">
        <button data-jump="-100">−100</button>
        <button data-jump="-10">−10</button>
        <button data-jump="-1">−1</button>
        <button data-jump="1">+1</button>
        <button data-jump="10">+10</button>
        <button data-jump="100">+100</button>
        <button id="priorBtn">Reset to prior</button>
      </div>
      <div class="row">
        <button id="acceptBtn" class="primary">Accept this frame</button>
        <button id="widenBtn">Mark ± tolerance</button>
        <button id="noMatchBtn" class="danger">No usable match</button>
        <button id="clearBtn">Clear</button>
      </div>
      <div class="hint">
        <kbd>←</kbd><kbd>→</kbd> step 1 · <kbd>Shift</kbd>+arrows step 10 · <kbd>Enter</kbd> accept ·
        <kbd>N</kbd> no match · <kbd>[</kbd><kbd>]</kbd> prev/next query
      </div>
      <div class="hint">
        <b>Pose fix</b> exists because the two runs may not point the camera the
        same way. Set it once per camera: switch to <i>Difference</i>, then slide
        until the wing mirror and the car body go dark. After that, aligning the
        scene means the vehicle is in the same place, and both ways of judging
        agree. The value you choose is saved with every label, so the convention
        is recorded rather than guessed at later.
      </div>
      <div class="hint" id="currentAnswer"></div>
    </div>
  </div>
</main>

<div class="status">
  <span class="pill done" id="pillDone"></span>
  <span class="pill nomatch" id="pillNoMatch"></span>
  <span class="pill todo" id="pillTodo"></span>
  <span class="hint">Answers persist in this browser. Download the CSV when every query is labelled.</span>
</div>
<div class="grid" id="queryGrid"></div>

<script>
const DATA = __PAYLOAD__;
const STORAGE_KEY = "ra-blind-v2-" + DATA.camera;
const POSE_STORAGE_KEY = "ra-blind-v2-pose-" + DATA.camera;
let compareMode = "side";
let showGuides = false;
let blinkTimer = null;
let poseOffset = DATA.poseOffsetPx || 0;
try {
  const stored = localStorage.getItem(POSE_STORAGE_KEY);
  if (stored !== null) poseOffset = Number(stored);
} catch (error) {}
const BRIGHTNESS_STORAGE_KEY = "ra-blind-v2-brightness-" + DATA.camera;

let answers = {};
try {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) answers = JSON.parse(stored);
} catch (error) { answers = {}; }

let index = 0;
let runBFrame = DATA.queries[0].linearPriorRunBFrame;
let runBBrightness = 100;
try {
  const storedBrightness = Number(localStorage.getItem(BRIGHTNESS_STORAGE_KEY));
  if (storedBrightness >= 50 && storedBrightness <= 250) runBBrightness = storedBrightness;
} catch (error) {}

const runBMin = DATA.runBFrames[0];
const runBMax = DATA.runBFrames[DATA.runBFrames.length - 1];

const pad = (value) => String(value).padStart(6, "0");
const el = (id) => document.getElementById(id);

function persist() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(answers)); } catch (error) {}
}

function currentQuery() { return DATA.queries[index]; }

function setRunBBrightness(value, persistSetting = true) {
  runBBrightness = Math.min(250, Math.max(50, Math.round(Number(value) || 100)));
  el("runBImage").style.filter = `brightness(${runBBrightness}%)`;
  el("runBBrightness").value = runBBrightness;
  el("runBBrightnessValue").textContent = `${runBBrightness}%`;
  if (persistSetting) {
    try { localStorage.setItem(BRIGHTNESS_STORAGE_KEY, String(runBBrightness)); } catch (error) {}
  }
}

function applyPoseOffset() {
  // Shift Run B by the measured camera pose difference so that "scene in the
  // same image position" and "vehicle at the same place" mean the same thing
  // again. Without this the two rules disagree and the label silently records
  // whichever one the annotator happened to use.
  el("runBImage").style.transform = `translateX(${poseOffset}px)`;
  el("poseOffset").value = poseOffset;
  el("poseOffsetValue").textContent = `${poseOffset} px`;
  try { localStorage.setItem(POSE_STORAGE_KEY, String(poseOffset)); } catch (error) {}
}

function applyCompareMode() {
  const overlay = el("overlayImage");
  if (blinkTimer) { clearInterval(blinkTimer); blinkTimer = null; }
  overlay.style.mixBlendMode = compareMode === "diff" ? "difference" : "normal";
  if (compareMode === "side") {
    overlay.style.opacity = 0;
  } else if (compareMode === "blend") {
    overlay.style.opacity = 0.5;
  } else if (compareMode === "diff") {
    overlay.style.opacity = 1;
  } else {
    let on = false;
    blinkTimer = setInterval(() => {
      on = !on;
      overlay.style.opacity = on ? 1 : 0;
    }, 450);
  }
}

function renderGuides() {
  const box = el("guides");
  box.classList.toggle("on", showGuides);
  if (!showGuides || box.childElementCount) return;
  for (let pct = 10; pct <= 90; pct += 10) {
    const line = document.createElement("i");
    line.style.left = `${pct}%`;
    if (pct === 50) line.className = "mid";
    box.appendChild(line);
  }
}

function setRunBFrame(value) {
  runBFrame = Math.min(runBMax, Math.max(runBMin, value));
  el("runBImage").src = `${DATA.camera}/runB/${pad(runBFrame)}.jpg`;
  el("runBLabel").textContent = runBFrame;
  el("scrub").value = runBFrame;
  el("frameInput").value = runBFrame;
}

function render() {
  const query = currentQuery();
  el("cameraName").textContent = DATA.camera;
  el("queryPos").textContent = `${index + 1} / ${DATA.queries.length}`;
  el("runAFrame").textContent = query.runAFrame;
  el("runALabel").textContent = query.runAFrame;
  el("runAImage").src = `${DATA.camera}/runA/${pad(query.runAFrame)}.jpg`;
  el("overlayImage").src = `${DATA.camera}/runA/${pad(query.runAFrame)}.jpg`;
  el("scrub").min = runBMin;
  el("scrub").max = runBMax;

  const answer = answers[query.queryId];
  if (answer && answer.label === "match") setRunBFrame(answer.runBFrame);
  else setRunBFrame(query.linearPriorRunBFrame);

  let text = "Not yet labelled.";
  if (answer && answer.label === "match") {
    text = `Recorded: match at B${answer.runBFrame}, acceptable ${answer.runBMin}–${answer.runBMax}.`;
  } else if (answer && answer.label === "no_match") {
    text = "Recorded: no usable correspondence.";
  }
  el("currentAnswer").textContent = text;

  const done = DATA.queries.filter((q) => answers[q.queryId]).length;
  const noMatch = DATA.queries.filter((q) => answers[q.queryId]?.label === "no_match").length;
  el("doneCount").textContent = `${done} / ${DATA.queries.length}`;
  el("pillDone").textContent = `${done - noMatch} matches`;
  el("pillNoMatch").textContent = `${noMatch} no-match`;
  el("pillTodo").textContent = `${DATA.queries.length - done} remaining`;
  renderGrid();
}

function renderGrid() {
  const grid = el("queryGrid");
  grid.innerHTML = "";
  DATA.queries.forEach((query, position) => {
    const cell = document.createElement("button");
    cell.className = "cell";
    cell.textContent = position + 1;
    const answer = answers[query.queryId];
    if (answer?.label === "match") cell.classList.add("done");
    if (answer?.label === "no_match") cell.classList.add("nomatch");
    if (position === index) cell.classList.add("active");
    cell.addEventListener("click", () => { index = position; render(); });
    grid.appendChild(cell);
  });
}

function accept(tolerance) {
  const query = currentQuery();
  answers[query.queryId] = {
    label: "match",
    runBFrame: runBFrame,
    runBMin: Math.max(runBMin, runBFrame - tolerance),
    runBMax: Math.min(runBMax, runBFrame + tolerance),
    poseOffset: poseOffset,
    compareMode: compareMode,
  };
  persist();
  render();
}

function markNoMatch() {
  answers[currentQuery().queryId] = {
    label: "no_match", poseOffset: poseOffset, compareMode: compareMode,
  };
  persist();
  render();
}

function move(delta) {
  index = Math.min(DATA.queries.length - 1, Math.max(0, index + delta));
  render();
}

el("compareMode").addEventListener("change", (event) => {
  compareMode = event.target.value; applyCompareMode();
});
el("poseOffset").addEventListener("input", (event) => {
  poseOffset = Number(event.target.value); applyPoseOffset();
});
el("showGuides").addEventListener("change", (event) => {
  showGuides = event.target.checked; renderGuides();
});
el("prevBtn").addEventListener("click", () => move(-1));
el("nextBtn").addEventListener("click", () => move(1));
el("acceptBtn").addEventListener("click", () => accept(0));
el("widenBtn").addEventListener("click", () => accept(1));
el("noMatchBtn").addEventListener("click", markNoMatch);
el("clearBtn").addEventListener("click", () => {
  delete answers[currentQuery().queryId];
  persist();
  render();
});
el("priorBtn").addEventListener("click", () => setRunBFrame(currentQuery().linearPriorRunBFrame));
el("scrub").addEventListener("input", (event) => setRunBFrame(Number(event.target.value)));
el("frameInput").addEventListener("change", (event) => setRunBFrame(Number(event.target.value)));
el("runBBrightness").addEventListener("input", (event) => setRunBBrightness(event.target.value));
el("brightnessResetBtn").addEventListener("click", () => setRunBBrightness(100));
document.querySelectorAll("[data-jump]").forEach((button) => {
  button.addEventListener("click", () => setRunBFrame(runBFrame + Number(button.dataset.jump)));
});

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT") return;
  const step = event.shiftKey ? 10 : 1;
  if (event.key === "ArrowLeft") { setRunBFrame(runBFrame - step); event.preventDefault(); }
  else if (event.key === "ArrowRight") { setRunBFrame(runBFrame + step); event.preventDefault(); }
  else if (event.key === "Enter") { accept(0); event.preventDefault(); }
  else if (event.key.toLowerCase() === "n") { markNoMatch(); event.preventDefault(); }
  else if (event.key === "[") { move(-1); event.preventDefault(); }
  else if (event.key === "]") { move(1); event.preventDefault(); }
});

el("saveBtn").addEventListener("click", () => {
  const header = "query_id,camera_id,runA_frame,label,runB_frame,runB_min,runB_max,pose_offset_px,compare_mode,source";
  const lines = DATA.queries.map((query) => {
    const answer = answers[query.queryId];
    const convention = `${answer ? answer.poseOffset ?? 0 : 0},${answer ? answer.compareMode ?? "side" : "side"}`;
    if (!answer) return `${query.queryId},${DATA.camera},${query.runAFrame},unlabelled,,,,${convention},manual_viewer_blind`;
    if (answer.label === "no_match") {
      return `${query.queryId},${DATA.camera},${query.runAFrame},no_match,,,,${convention},manual_viewer_blind`;
    }
    return `${query.queryId},${DATA.camera},${query.runAFrame},match,` +
      `${answer.runBFrame},${answer.runBMin},${answer.runBMax},${convention},manual_viewer_blind`;
  });
  const blob = new Blob([[header].concat(lines).join("\n") + "\n"], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `blind_v2_${DATA.camera}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
});

setRunBBrightness(runBBrightness, false);
applyPoseOffset();
applyCompareMode();
renderGuides();
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
