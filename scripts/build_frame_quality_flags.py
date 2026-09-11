#!/usr/bin/env python3
"""Flag frames that are too dark to carry recoverable scene content.

A visual sweep of all 10,585 decoded frames found a contiguous 294-frame stretch
of Run B CAM0 (roughly ordinals 1351-1644, a canopied unlit road at dusk in
rain) that is visually black: a 2.4x brightness boost recovers only a tree trunk
and a kerb line. The submitted mapping labels every row whose partner falls in
that stretch `accepted_strong`. Both sides of such a pair are dark, so a
dark-against-dark descriptor match is close to guaranteed - a mechanism for
*systematic* high-confidence error rather than random error.

Nothing in the pipeline measured this. The reliability mask asks which *pixels*
are attached to the camera; it never asks whether a whole *frame* has any signal
left. Aggregate luma statistics (median 62 in Run A CAM0 against 28 in Run B)
average the stretch away.

So this module measures one thing per frame and declares its thresholds up
front:

    mean_luma  - Rec. 709 luma of the decoded RGB, averaged over the frame
    p99_luma   - its 99th percentile, i.e. how bright the brightest content is

    too_dark   <=>  mean_luma < 12  or  p99_luma < 40   (of 255)

Two thresholds because they fail differently. A frame can have a respectable
mean and still be featureless (uniform grey fog); it can have a low mean and
still be usable if a lit shopfront occupies a corner. Requiring *both* to be
healthy is the conservative reading, and it is the one a downstream consumer
wants: the flag demotes a pair out of the top tier, it does not delete a frame.

The values are chosen from the measured distribution rather than from taste:
mean 12 is the level the visual audit found unrecoverable at 720x540, and p99 40
is roughly where the brightest 1% of a frame stops separating from sensor noise.
Both are recorded in the JSON summary so a consumer can re-derive the flag at
another threshold from the same CSV.

Analysis raster is 192x144. Exposure is a low-frequency property, the flag is
per-frame rather than per-pixel, and a small raster keeps a whole-stream decode
pass cheap. The decode itself is the canonical one, with pinned colour.

The script also cross-references the flag against the two things it is supposed
to inform: the submitted mapping (how many accepted rows have a dark partner on
either side) and the prediction-blind labels (what the annotator's answers say
in those regions). Neither cross-reference changes any mapping; both are
reporting.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import frame_service


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task3"
FLAGS_CSV = OUTPUT_DIR / "frame_quality_flags.csv"
SUMMARY_JSON = OUTPUT_DIR / "frame_quality_flags.json"

MAPPING = ROOT / "outputs" / "task2_unified" / "{camera}" / "frame_mapping_unified.csv"
LABELS = ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "blind_v2_labels.csv"

RUNS = ("runA", "runB")
CAMERAS = ("cam0", "cam5")

ANALYSIS_WIDTH, ANALYSIS_HEIGHT = 192, 144

# Declared thresholds. Changing either invalidates the flag column, not the
# measurements beside it, which is why both are written to the CSV's summary.
MEAN_LUMA_FLOOR = 12.0
P99_LUMA_FLOOR = 40.0

FLAG_TOO_DARK = "too_dark"
FLAG_OK = "ok"

# The slope gate. `outputs/task2_kinks/` shows that the submitted mapping
# asserts one-step Run B increments its own motion model forbids, and that the
# blind labels miss more often near them. The gate is a Task 3 *policy* over the
# frozen mapping - it demotes rows, it never rewrites one - so it belongs beside
# the other per-frame flags rather than inside the pipeline. The column carries
# one value per Run A ordinal per camera; Run B rows are empty, because a kink
# is a property of a Run A frame's neighbourhood in that camera's mapping.
KINKS_JSON = ROOT / "outputs" / "task2_kinks" / "mapping_kinks.json"
KINK_MAPPING = "submitted_unified"
SLOPE_GATE_COLUMN = "slope_gate"
FLAG_NEAR_KINK = "near_kink"

CSV_FIELDNAMES = ["run", "camera", "ordinal", "mean_luma", "p99_luma", "flag", SLOPE_GATE_COLUMN]

# A stretch shorter than this is a single dark frame or two, which a matcher
# rides through; the reportable object is a run of frames long enough that a
# partner has nowhere brighter to go.
MIN_REPORTED_STRETCH = 5

# Rec. 709 luma from the pinned BT.709 RGB the canonical decoder emits.
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

ACCEPTED_PREFIX = "accepted"


def frame_luma(image: np.ndarray) -> tuple[float, float]:
    """Mean and 99th-percentile Rec. 709 luma of one HxWx3 uint8 RGB frame."""

    luma = image.astype(np.float64) @ LUMA_WEIGHTS
    return float(luma.mean()), float(np.percentile(luma, 99.0))


def classify(mean_luma: float, p99_luma: float) -> str:
    if mean_luma < MEAN_LUMA_FLOOR or p99_luma < P99_LUMA_FLOOR:
        return FLAG_TOO_DARK
    return FLAG_OK


def contiguous_stretches(
    ordinals: list[int], minimum: int = MIN_REPORTED_STRETCH
) -> list[dict[str, int]]:
    """Group sorted ordinals into maximal runs of consecutive integers."""

    stretches: list[dict[str, int]] = []
    start = previous = None
    for ordinal in sorted(ordinals):
        if start is None:
            start = previous = ordinal
            continue
        if ordinal == previous + 1:
            previous = ordinal
            continue
        stretches.append({"start": start, "end": previous, "length": previous - start + 1})
        start = previous = ordinal
    if start is not None:
        stretches.append({"start": start, "end": previous, "length": previous - start + 1})
    return [stretch for stretch in stretches if stretch["length"] >= minimum]


def measure_stream(run: str, camera: str) -> list[dict[str, object]]:
    """One decode pass over a stream, one row per display-order ordinal."""

    rows: list[dict[str, object]] = []
    path = frame_service.source_path(run, camera)
    for ordinal, image in frame_service.iter_frames(
        path, ANALYSIS_WIDTH, ANALYSIS_HEIGHT
    ):
        mean_luma, p99_luma = frame_luma(image)
        rows.append(
            {
                "run": run,
                "camera": camera,
                "ordinal": ordinal,
                "mean_luma": round(mean_luma, 3),
                "p99_luma": round(p99_luma, 3),
                "flag": classify(mean_luma, p99_luma),
            }
        )
    return rows


def stream_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    dark = [int(row["ordinal"]) for row in rows if row["flag"] == FLAG_TOO_DARK]
    means = np.array([row["mean_luma"] for row in rows], dtype=np.float64)
    return {
        "frames": len(rows),
        "too_dark": len(dark),
        "too_dark_fraction": round(len(dark) / len(rows), 4) if rows else 0.0,
        "mean_luma_median": round(float(np.median(means)), 3) if rows else None,
        "mean_luma_min": round(float(means.min()), 3) if rows else None,
        "dark_stretches": contiguous_stretches(dark),
        "longest_dark_stretch": max(
            (stretch["length"] for stretch in contiguous_stretches(dark)), default=0
        ),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mapping_cross_reference(
    flags: dict[tuple[str, str], dict[int, str]]
) -> dict[str, object]:
    """How many accepted rows of the submitted mapping touch a dark frame.

    Reporting only. The submitted mapping is frozen and this module never
    rewrites it; the point is to put a number on an exposure hazard the mapping
    has no column for.
    """

    report: dict[str, object] = {}
    for camera in CAMERAS:
        rows = _read_csv(Path(str(MAPPING).format(camera=camera)))
        run_a = flags[("runA", camera)]
        run_b = flags[("runB", camera)]
        accepted = 0
        dark_a = dark_b = dark_either = dark_both = 0
        touched: list[int] = []
        for row in rows:
            if not row["status"].startswith(ACCEPTED_PREFIX):
                continue
            partner = row["runB_frame"].strip()
            if not partner:
                continue
            accepted += 1
            a_dark = run_a.get(int(row["runA_frame"])) == FLAG_TOO_DARK
            b_dark = run_b.get(int(partner)) == FLAG_TOO_DARK
            dark_a += int(a_dark)
            dark_b += int(b_dark)
            dark_either += int(a_dark or b_dark)
            dark_both += int(a_dark and b_dark)
            if a_dark or b_dark:
                touched.append(int(row["runA_frame"]))
        report[camera] = {
            "accepted_rows": accepted,
            "runA_frame_too_dark": dark_a,
            "runB_partner_too_dark": dark_b,
            "either_side_too_dark": dark_either,
            "both_sides_too_dark": dark_both,
            "fraction_of_accepted": round(dark_either / accepted, 4) if accepted else 0.0,
            "runA_stretches_touched": contiguous_stretches(touched, minimum=1),
        }
    return report


def label_cross_reference(
    flags: dict[tuple[str, str], dict[int, str]]
) -> dict[str, object]:
    """What the prediction-blind labels say where the flag fires.

    A label counts as being in a dark region when the annotator's own preferred
    Run B frame, or the frame the submitted mapping predicted, is flagged. The
    interval test is the evaluation's: correct iff the prediction lies in
    [runB_min, runB_max]. Distance is to the nearer interval edge.
    """

    if not LABELS.exists():
        return {"available": False}
    labels = _read_csv(LABELS)
    report: dict[str, object] = {"available": True}
    for camera in CAMERAS:
        predictions = {
            int(row["runA_frame"]): row
            for row in _read_csv(Path(str(MAPPING).format(camera=camera)))
        }
        run_b = flags[("runB", camera)]
        cases: list[dict[str, object]] = []
        for label in labels:
            if label["camera_id"] != camera:
                continue
            frame_a = int(label["runA_frame"])
            row = predictions.get(frame_a, {})
            partner = (row.get("runB_frame") or "").strip()
            predicted = int(partner) if partner else None
            preferred = label["runB_frame"].strip()
            marks = [value for value in (predicted, int(preferred) if preferred else None)
                     if value is not None]
            if not any(run_b.get(value) == FLAG_TOO_DARK for value in marks):
                continue
            case: dict[str, object] = {
                "query_id": label["query_id"],
                "runA_frame": frame_a,
                "label": label["label"],
                "labelled_runB_frame": int(preferred) if preferred else None,
                "predicted_runB_frame": predicted,
                "status": row.get("status", "missing"),
            }
            if label["label"] == "match" and predicted is not None:
                low, high = int(label["runB_min"]), int(label["runB_max"])
                case["correct"] = low <= predicted <= high
                case["distance_frames"] = 0 if case["correct"] else min(
                    abs(predicted - low), abs(predicted - high)
                )
            cases.append(case)
        determinable = [case for case in cases if case["label"] == "match"]
        report[camera] = {
            "labels_in_dark_regions": len(cases),
            "determinable": len(determinable),
            "undetermined": len(cases) - len(determinable),
            "correct": sum(1 for case in determinable if case.get("correct")),
            "miss_distances_frames": sorted(
                case["distance_frames"]
                for case in determinable
                if not case.get("correct") and "distance_frames" in case
            ),
            "cases": cases,
        }
    return report


def apply_slope_gate(
    by_stream: dict[tuple[str, str], list[dict[str, object]]]
) -> dict[str, object]:
    """Write the `slope_gate` column and report what it would demote.

    The kink list and the demotion radius are read from
    `outputs/task2_kinks/mapping_kinks.json` and from the module that produced
    it, so the flag and the costing in that artifact cannot disagree. Every row
    gets a value: `near_kink` for a Run A ordinal within the radius of a kink in
    that camera's submitted mapping, empty otherwise.
    """

    for rows in by_stream.values():
        for row in rows:
            row.setdefault(SLOPE_GATE_COLUMN, "")

    if not KINKS_JSON.is_file():
        return {"available": False, "reason": f"{KINKS_JSON.name} absent"}

    import analyze_mapping_kinks as kinks

    payload = json.loads(KINKS_JSON.read_text())
    submitted = payload["mappings"][KINK_MAPPING]["per_camera"]
    policy = payload.get("slope_gate_policy", {})
    threshold = payload["physical_bound"]["headline_threshold"]

    report: dict[str, object] = {
        "available": True,
        "source": "outputs/task2_kinks/mapping_kinks.json",
        "mapping": KINK_MAPPING,
        "kink_threshold": threshold,
        "demote_radius_runA_frames": kinks.DEMOTE_RADIUS,
        "rule": (
            "near_kink iff the Run A ordinal is within the radius of a one-step "
            f"Run B increment >= {threshold} in that camera's submitted mapping"
        ),
        "evidence_status": (
            "Label-free flag. The precision figures carried over from the kink "
            "audit are diagnostics on SPENT labels, not held-out estimates."
        ),
        "cameras": {},
    }
    for camera in CAMERAS:
        window = kinks.demoted_frames(
            [k for k in submitted[camera]["kinks"] if k["increment"] >= threshold]
        )
        flagged = 0
        for row in by_stream.get(("runA", camera), []):
            if int(row["ordinal"]) in window:
                row[SLOPE_GATE_COLUMN] = FLAG_NEAR_KINK
                flagged += 1
        costed = policy.get("per_camera", {}).get(camera, {})
        report["cameras"][camera] = {
            "kinks": submitted[camera]["kink_counts"][f"K>={threshold}"]["kinks"],
            "runA_frames_flagged": flagged,
            "top_tier_rows_demoted": costed.get("demoted_rows"),
            "top_tier_rows": costed.get("top_tier_rows"),
            "spent_labels_before_gate": costed.get("spent_labels_before_gate"),
            "spent_labels_after_gate": costed.get("spent_labels_after_gate"),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-reference-only",
        action="store_true",
        help="Re-read an existing flags CSV and rebuild only the summary JSON.",
    )
    parser.add_argument(
        "--slope-gate",
        action="store_true",
        help=(
            "Re-read the existing flags CSV, rewrite it with the slope_gate "
            "column from outputs/task2_kinks/, and rebuild the summary. No decode."
        ),
    )
    arguments = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if arguments.cross_reference_only or arguments.slope_gate:
        rows = _read_csv(FLAGS_CSV)
        for row in rows:
            row["ordinal"] = int(row["ordinal"])
            row["mean_luma"] = float(row["mean_luma"])
            row["p99_luma"] = float(row["p99_luma"])
        by_stream: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in rows:
            by_stream.setdefault((row["run"], row["camera"]), []).append(row)
    else:
        by_stream = {}
        for run in RUNS:
            for camera in CAMERAS:
                print(f"  measuring {run} {camera} ...", flush=True)
                by_stream[(run, camera)] = measure_stream(run, camera)

    slope_gate = apply_slope_gate(by_stream)
    if not arguments.cross_reference_only or arguments.slope_gate:
        with FLAGS_CSV.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for run in RUNS:
                for camera in CAMERAS:
                    writer.writerows(by_stream[(run, camera)])
        print(f"  {FLAGS_CSV.relative_to(ROOT)}")

    flags = {
        key: {int(row["ordinal"]): row["flag"] for row in rows}
        for key, rows in by_stream.items()
    }

    summary = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_raster": [ANALYSIS_WIDTH, ANALYSIS_HEIGHT],
        "luma": "Rec. 709 luma of the canonical pinned-colour RGB decode, 0-255",
        "thresholds": {
            "mean_luma_floor": MEAN_LUMA_FLOOR,
            "p99_luma_floor": P99_LUMA_FLOOR,
            "rule": "too_dark iff mean_luma < 12 or p99_luma < 40",
            "min_reported_stretch": MIN_REPORTED_STRETCH,
        },
        "evidence_status": (
            "Measured on every decoded frame. The mapping and label "
            "cross-references are reporting only: no mapping row was rewritten."
        ),
        "streams": {
            f"{run}_{camera}": stream_summary(by_stream[(run, camera)])
            for run in RUNS
            for camera in CAMERAS
        },
        "submitted_mapping": mapping_cross_reference(flags),
        "blind_labels": label_cross_reference(flags),
        "slope_gate": slope_gate,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  {SUMMARY_JSON.relative_to(ROOT)}")

    for name, stream in summary["streams"].items():
        print(
            f"    {name}: {stream['too_dark']}/{stream['frames']} too_dark, "
            f"longest stretch {stream['longest_dark_stretch']}"
        )


if __name__ == "__main__":
    main()
