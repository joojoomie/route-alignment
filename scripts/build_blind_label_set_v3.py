#!/usr/bin/env python3
"""Export a third, TARGETED prediction-blind label set with a better annotator.

v2 sampled uniformly along the route and asked 40 questions per camera. It
answered "how often is the method right on an average frame". It could not
answer "is the method right where it is most likely to be wrong", because a
uniform sample puts almost no queries in the three places the audit says are
dangerous: mapping kinks, dark Run B stretches, and the one confirmed
occlusion. It also had a protocol defect - "no usable match" conflated *I
cannot tell* with *there is no such place* - so its no-match rows cannot be
scored either way.

v3 fixes both. 24 queries per camera, drawn from four strata:

* ``kink``     - Run A frames next to a large jump in the submitted mapping.
* ``dark``     - Run A frames whose submitted partner lands in a Run B stretch
                 the quality flags call too dark.
* ``occlusion``- Run A frames inside the confirmed Run B occlusion (CAM0 only).
* ``corridor`` - uniformly spaced elsewhere, the v2-style control.

The strata are chosen *using* the submitted mapping's structure. That is a real
and disclosed dependency, and it is bounded: it decides **which frames are
asked about**, never **what the annotator is shown**. The page carries the same
linear route-progress start position v2 used - arithmetic on the route
boundaries - and no stratum tag, no mapping partner, no score and no
prediction. A test greps the exported HTML for the submitted partner of every
query and fails if any of them appears.

The annotator itself is the other half of the fix. Three aids are new:

1. A two-axis pose compensation for CAM5, whose camera pointed differently in
   Run B (measured -37.5, -11 px at 896x672). Without it the annotator either
   aligns the scene and mislabels the vehicle position, or aligns the vehicle
   and sees a scene that never matches. The chosen shift is saved per label.
2. A far line and a near line the annotator drags onto A. One common shift can
   only align both depths at the correct frame; parallax breaks the near one
   first. This is the single most useful cue for frame-level precision and the
   annotator had no way to apply it before.
3. Band-restricted blink - top 45% only, or bottom third only - so the two
   depths can be judged one at a time instead of competing for attention.

No automatic best-frame hint is offered anywhere. A suggestion would leak a
method into the labels, which is the whole thing this set exists to avoid.

**Sign convention for the pose compensation.** The diagnostic measures a scene
*displacement*: on CAM5 the scene in Run B sits 37.5 px to the LEFT of where it
sits in Run A (dx = -37.5 at 896x672). The page applies `translate(dx, dy)` to
the Run B image, so the compensation it needs is the NEGATIVE of that
displacement - Run B must move RIGHT by +40 px at the 960-wide filmstrip. The
first v3 export shipped the raw measurement as the compensation and doubled the
misalignment; see `outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`.
`measuredDxPx`/`measuredDyPx` keep the raw measurement, `dxPx`/`dyPx` carry the
compensation, and positive means "move Run B right/down".

`--camera`, `--set-name` and `--seed` make this builder reusable for a fresh
query set. `--camera cam5 --set-name blind_v3b --seed 20260903` writes the CAM5
redo set to `outputs/task2_blind_v3b/`, excluding every ordinal already labelled
in v1, v2 and v3.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

import build_task2_unified as unified
import frame_service


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET_NAME = "blind_v3"


def output_dir(set_name: str = DEFAULT_SET_NAME) -> Path:
    return ROOT / "outputs" / f"task2_{set_name}"


def pages_dir(set_name: str = DEFAULT_SET_NAME) -> Path:
    # The annotator's web server is pointed at the pages directory, not at the
    # set directory. The manifest records the stratum of every query and, for
    # the dark stratum, the Run B window its partner falls in - which is close
    # enough to an answer to matter. Keeping it one level above the served root
    # means a directory listing at localhost cannot hand it over.
    return output_dir(set_name) / "pages"


def manifest_path(set_name: str = DEFAULT_SET_NAME) -> Path:
    return output_dir(set_name) / f"{set_name}_manifest.json"


def version_tag(set_name: str = DEFAULT_SET_NAME) -> str:
    """`blind_v3` -> `v3`, `blind_v3b` -> `v3b`. Used in query ids and filenames."""

    return set_name.split("blind_", 1)[-1] if set_name.startswith("blind_") else set_name


OUTPUT_DIR = output_dir()
PAGES_DIR = pages_dir()

ANCHOR_FILE = ROOT / "outputs" / "task2_keyframes" / "anchor_candidates.csv"
MANUAL_LABEL_FILE = ROOT / "outputs" / "task2_keyframes" / "manual_annotations.csv"
BLIND_V2_LABEL_FILE = (
    ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "blind_v2_labels.csv"
)
BLIND_V3_LABEL_FILE = (
    ROOT / "outputs" / "task2_evaluation" / "blind_v3" / "blind_v3_labels.csv"
)
KINK_FILE_TEMPLATE = ROOT / "outputs" / "task2_kinks" / "kinks_{camera}.csv"
QUALITY_FLAG_FILE = ROOT / "outputs" / "task3" / "frame_quality_flags.csv"
OCCLUSION_FILE = (
    ROOT / "outputs" / "task2_abstention" / "confirmed_occlusion_runB_cam0.json"
)
MAPPING_FILE_TEMPLATE = (
    ROOT / "outputs" / "task2_unified" / "{camera}" / "frame_mapping_unified.csv"
)
POSE_FILE = ROOT / "outputs" / "task2_pose_offset" / "pose_offset_diagnostic.json"

CAMERAS = ("cam0", "cam5")
QUERIES_PER_CAMERA = 24
RANDOM_SEED = 20260902

# CAM0 keeps three occlusion queries. CAM5 has no confirmed occlusion - the one
# instance in the audit was seen on CAM0's side of the vehicle only - so its
# three slots go to the two strata that do exist there: two more kinks and one
# more dark frame. Inventing a CAM5 occlusion stratum would mean asking about
# frames chosen for a phenomenon nobody has evidence of on that camera.
STRATA_PLAN: dict[str, dict[str, int]] = {
    "cam0": {"kink": 8, "dark": 4, "occlusion": 3, "corridor": 9},
    "cam5": {"kink": 10, "dark": 5, "occlusion": 0, "corridor": 9},
}
STRATUM_ORDER = ("occlusion", "dark", "kink", "corridor")

MINIMUM_DISTANCE_FROM_LABELLED = 10
MINIMUM_DISTANCE_BETWEEN_QUERIES = 10
# The confirmed occlusion is 30 Run A frames long and two of those frames are
# already blocked by earlier labels. Three queries cannot be 10 apart inside
# what remains, so inside this stratum only, queries may be 5 apart. The
# distance to previously labelled ordinals is *not* relaxed - that is the
# constraint that keeps the set unseen.
MINIMUM_DISTANCE_WITHIN_OCCLUSION = 5

KINK_INCREMENT_THRESHOLD = 6
KINK_QUERY_RADIUS = 12
MINIMUM_DARK_STRETCH = 4

QUERY_JPEG_QUALITY = 3      # ffmpeg -q:v, lower is better
FILMSTRIP_JPEG_QUALITY = 4
FILMSTRIP_WIDTH = 960
FILMSTRIP_HEIGHT = 720

# The pose diagnostic measured CAM5's shift on 896x672 rasters. The annotator
# works on 960x720 JPEGs, so the number is scaled; it is stored as a fraction of
# frame width so the page can apply it at whatever size the browser renders.
POSE_MEASUREMENT_RASTER = frame_service.GEOMETRY_SIZE
POSE_COMPENSATION_DEFAULT_ON = {"cam0": False, "cam5": True}
# The sliders may move +-25 px around the exported default and no further. A
# wider slider is not a better tool: v3 showed that a free 400 px sweep lets the
# annotator zero both residuals at the wrong frame, so the shift silently
# absorbs frame error. 25 px is about two and a half frames of scene sweep at
# CAM5's measured 10.3 px/frame - enough for a genuine residual pose error,
# too little to stand in for a frame.
POSE_SLIDER_RANGE_PX = 25
# The far guide must sit in the top half of the frame and the near guide in the
# bottom third. A "far" reference picked off the road surface is a near
# reference wearing a hat, and the far + near rule collapses when both guides
# are at the same depth.
FAR_GUIDE_BAND = (0.0, 0.5)
NEAR_GUIDE_BAND = (2 / 3, 1.0)

# The page writes this header and the importer requires exactly it; a test pins
# the two together so a column can never be added on one side only.
CSV_COLUMNS = (
    "query_id",
    "camera_id",
    "runA_frame",
    "label",
    "runB_frame",
    "runB_min",
    "runB_max",
    "pose_dx_px",
    "pose_dy_px",
    "pose_compensated",
    "far_line_x_frac",
    "far_line_y_frac",
    "near_line_x_frac",
    "near_line_y_frac",
    "compare_mode",
    "source",
)

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise RuntimeError("ffmpeg is required but was not found on PATH")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing output.")
    parser.add_argument("--queries-per-camera", type=int, default=QUERIES_PER_CAMERA)
    parser.add_argument(
        "--skip-filmstrip",
        action="store_true",
        help="Only rewrite the manifest and pages, reusing extracted frames.",
    )
    parser.add_argument(
        "--camera",
        choices=CAMERAS,
        action="append",
        help="Build only this camera. Repeatable; default is both.",
    )
    parser.add_argument(
        "--set-name",
        default=DEFAULT_SET_NAME,
        help="Label set name; decides outputs/task2_<set-name>/ and the query ids.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def _camera_column(path: Path, camera: str, column: str = "runA_frame") -> set[int]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {
            int(row[column])
            for row in csv.DictReader(handle)
            if row["camera_id"] == camera and row[column].strip()
        }


# Every file a human has already been shown Run A frames from. `blind_v3` is
# absent from its own set's list because it did not exist when v3 was drawn;
# every later set excludes it, which is what keeps the CAM5 redo free of any
# memory of the answers given under the defective tool.
LABEL_SOURCE_FILES = {
    "blind_v2_labels": BLIND_V2_LABEL_FILE,
    "blind_v3_labels": BLIND_V3_LABEL_FILE,
    "manual_annotations": MANUAL_LABEL_FILE,
    "anchor_candidates": ANCHOR_FILE,
}
EXCLUSION_SOURCES = {
    "blind_v3": ("blind_v2_labels", "manual_annotations", "anchor_candidates"),
}
DEFAULT_EXCLUSION_SOURCES = tuple(LABEL_SOURCE_FILES)


def exclusion_sources(set_name: str = DEFAULT_SET_NAME) -> tuple[str, ...]:
    return EXCLUSION_SOURCES.get(set_name, DEFAULT_EXCLUSION_SOURCES)


def previously_labelled(
    camera: str, set_name: str = DEFAULT_SET_NAME
) -> dict[str, set[int]]:
    """Every Run A ordinal a human has already been shown, by source."""

    return {
        name: _camera_column(LABEL_SOURCE_FILES[name], camera)
        for name in exclusion_sources(set_name)
    }


def submitted_partners(camera: str) -> dict[int, int]:
    """Run A -> Run B from the submitted unified mapping.

    Used only to *locate* the strata. It never reaches the page.
    """

    path = Path(str(MAPPING_FILE_TEMPLATE).format(camera=camera))
    with path.open(newline="") as handle:
        return {
            int(row["runA_frame"]): int(row["runB_frame"])
            for row in csv.DictReader(handle)
            if row["runB_frame"].strip()
        }


def submitted_kinks(camera: str) -> list[dict[str, object]]:
    path = Path(str(KINK_FILE_TEMPLATE).format(camera=camera))
    with path.open(newline="") as handle:
        rows = [
            {
                "runA_frame": int(row["runA_frame"]),
                "increment": int(row["increment"]),
                "dark": row["dark"].strip().lower() == "true",
            }
            for row in csv.DictReader(handle)
            if row["mapping"] == "submitted_unified"
            and abs(int(row["increment"])) >= KINK_INCREMENT_THRESHOLD
        ]
    rows.sort(key=lambda row: row["runA_frame"])
    return rows


def dark_run_b_stretches(camera: str) -> list[tuple[int, int]]:
    with QUALITY_FLAG_FILE.open(newline="") as handle:
        ordinals = sorted(
            int(row["ordinal"])
            for row in csv.DictReader(handle)
            if row["run"] == "runB" and row["camera"] == camera and row["flag"] == "too_dark"
        )
    stretches: list[tuple[int, int]] = []
    for _, group in itertools.groupby(enumerate(ordinals), lambda pair: pair[1] - pair[0]):
        block = [value for _, value in group]
        if len(block) >= MINIMUM_DARK_STRETCH:
            stretches.append((block[0], block[-1]))
    return stretches


def occlusion_windows() -> dict[str, dict[str, int]]:
    """The one confirmed no-correspondence stretch, read from the audit file."""

    payload = json.loads(OCCLUSION_FILE.read_text())
    run_a = payload["unified_mapping"]["runA_frames"]
    return {
        payload["camera"]: {
            "runA_first": int(run_a["first"]),
            "runA_last": int(run_a["last"]),
            "runB_start": int(payload["interval"]["start"]),
            "runB_end": int(payload["interval"]["end"]),
        }
    }


def measured_pose_shift(camera: str) -> tuple[float, float]:
    payload = json.loads(POSE_FILE.read_text())
    convention = payload["cameras"][camera]["matcher_convention"]
    return float(convention["dx_px"]["median"]), float(convention["dy_px"]["median"])


def frame_counts() -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for run in ("runA", "runB"):
        for camera in CAMERAS:
            counts[(run, camera)] = frame_service.hevc_bitstream.scan(
                frame_service.source_path(run, camera)
            ).picture_count
    return counts


# --------------------------------------------------------------------------
# Query selection
# --------------------------------------------------------------------------


class Selector:
    """Deterministic stratified selection under the spacing constraints."""

    def __init__(
        self,
        camera: str,
        route_start: int,
        route_end_exclusive: int,
        blocked: set[int],
        rng: np.random.Generator,
    ) -> None:
        self.camera = camera
        self.route_start = route_start
        self.route_end_exclusive = route_end_exclusive
        self.blocked = blocked
        self.rng = rng
        self.chosen: list[dict[str, object]] = []

    def in_route(self, ordinal: int) -> bool:
        return self.route_start <= ordinal < self.route_end_exclusive

    def is_valid(self, ordinal: int, stratum: str) -> bool:
        if not self.in_route(ordinal):
            return False
        if any(abs(ordinal - other) < MINIMUM_DISTANCE_FROM_LABELLED for other in self.blocked):
            return False
        for previous in self.chosen:
            gap = (
                MINIMUM_DISTANCE_WITHIN_OCCLUSION
                if stratum == "occlusion" and previous["stratum"] == "occlusion"
                else MINIMUM_DISTANCE_BETWEEN_QUERIES
            )
            if abs(ordinal - int(previous["runA_frame"])) < gap:
                return False
        return True

    def candidates(self, low: int, high: int, stratum: str) -> list[int]:
        return [
            ordinal
            for ordinal in range(low, high + 1)
            if self.is_valid(ordinal, stratum)
        ]

    def take_nearest(
        self, target: float, low: int, high: int, stratum: str, detail: dict[str, object]
    ) -> int | None:
        pool = self.candidates(low, high, stratum)
        if not pool:
            return None
        pick = min(pool, key=lambda ordinal: (abs(ordinal - target), ordinal))
        self.record(pick, stratum, detail)
        return pick

    def take_random(
        self, low: int, high: int, stratum: str, detail: dict[str, object]
    ) -> int | None:
        pool = self.candidates(low, high, stratum)
        if not pool:
            return None
        pick = int(self.rng.choice(pool))
        self.record(pick, stratum, detail)
        return pick

    def record(self, ordinal: int, stratum: str, detail: dict[str, object]) -> None:
        entry = {"runA_frame": int(ordinal), "stratum": stratum}
        entry.update(detail)
        self.chosen.append(entry)


def farthest_point_order(values: list[int]) -> list[int]:
    """Order values so each successive one is as far as possible from the rest.

    Kinks cluster - CAM0 has eleven of them inside 120 frames around 1600 - so
    taking the first N in route order would put most of the stratum in one
    place. This spreads the picks without needing equal-width bins that half the
    time contain no kink at all.
    """

    remaining = sorted(values)
    if not remaining:
        return []
    midpoint = (remaining[0] + remaining[-1]) / 2
    ordered = [min(remaining, key=lambda value: (abs(value - midpoint), value))]
    remaining.remove(ordered[0])
    while remaining:
        nxt = max(
            remaining,
            key=lambda value: (min(abs(value - taken) for taken in ordered), -value),
        )
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def select_occlusion(selector: Selector, count: int, window: dict[str, int]) -> int:
    """Place `count` occlusion queries; returns how many were actually placed."""

    low, high = window["runA_first"], window["runA_last"]
    # Spread the targets over the part of the window that is actually free
    # rather than over its nominal bounds. Earlier labels sit at both ends of
    # this occlusion, so evenly spacing over 327-356 would aim two of three
    # queries at ordinals no query may occupy.
    free = [
        ordinal
        for ordinal in range(low, high + 1)
        if selector.in_route(ordinal)
        and all(
            abs(ordinal - other) >= MINIMUM_DISTANCE_FROM_LABELLED
            for other in selector.blocked
        )
    ]
    if not free:
        return 0
    low, high = free[0], free[-1]
    placed = 0
    for slot in range(count):
        target = low + (high - low) * (slot + 0.5) / count
        detail = {
            "occlusion_runB_interval": [window["runB_start"], window["runB_end"]],
        }
        if selector.take_nearest(target, low, high, "occlusion", detail) is not None:
            placed += 1
    return placed


def select_dark(
    selector: Selector,
    count: int,
    stretches: list[tuple[int, int]],
    partners: dict[int, int],
) -> int:
    """Place `count` dark queries; returns how many were actually placed."""

    if count == 0:
        return 0
    if not stretches:
        return 0

    by_stretch: list[tuple[tuple[int, int], list[int]]] = []
    for stretch in stretches:
        low, high = stretch
        pool = sorted(
            run_a for run_a, run_b in partners.items() if low <= run_b <= high
        )
        if pool:
            by_stretch.append((stretch, pool))
    if not by_stretch:
        return 0

    # Longest stretch first: a 177-frame blackout is a stronger test than a
    # 10-frame one, and if there are fewer stretches than queries the extra
    # queries should land in the long ones.
    by_stretch.sort(key=lambda item: (-(item[0][1] - item[0][0]), item[0][0]))

    assignments: list[tuple[tuple[int, int], list[int], int, int]] = []
    per_stretch = [0] * len(by_stretch)
    for index in range(count):
        per_stretch[index % len(by_stretch)] += 1
    for (stretch, pool), wanted in zip(by_stretch, per_stretch):
        for slot in range(wanted):
            assignments.append((stretch, pool, slot, wanted))

    placed = 0
    for stretch, pool, slot, wanted in assignments:
        target = pool[0] + (pool[-1] - pool[0]) * (slot + 0.5) / wanted
        detail = {
            "dark_runB_stretch": [stretch[0], stretch[1]],
            "dark_runB_stretch_length": stretch[1] - stretch[0] + 1,
        }
        if selector.take_nearest(target, pool[0], pool[-1], "dark", detail) is not None:
            placed += 1
    return placed


def select_kink(selector: Selector, count: int, kinks: list[dict[str, object]]) -> int:
    """Place `count` kink queries; returns how many were actually placed."""

    if count == 0:
        return 0
    lit = [kink for kink in kinks if not kink["dark"]]
    dark = [kink for kink in kinks if kink["dark"]]
    by_frame = {int(kink["runA_frame"]): kink for kink in kinks}

    # Lit kinks first: a kink inside a blackout is already covered by the dark
    # stratum, and asking about a frame nobody can see tests the annotator's
    # patience rather than the mapping.
    order = farthest_point_order([int(kink["runA_frame"]) for kink in lit])
    order += farthest_point_order([int(kink["runA_frame"]) for kink in dark])

    taken = 0
    for kink_frame in order:
        if taken == count:
            break
        kink = by_frame[kink_frame]
        detail = {
            "kink_runA_frame": kink_frame,
            "kink_increment": int(kink["increment"]),
            "kink_dark": bool(kink["dark"]),
        }
        picked = selector.take_random(
            kink_frame - KINK_QUERY_RADIUS,
            kink_frame + KINK_QUERY_RADIUS,
            "kink",
            detail,
        )
        if picked is not None:
            taken += 1
    return taken


def select_corridor(selector: Selector, count: int) -> int:
    if count == 0:
        return 0
    edges = np.linspace(selector.route_start, selector.route_end_exclusive, count + 1)
    for index in range(count):
        low, high = int(edges[index]), int(edges[index + 1]) - 1
        jitter = float(selector.rng.uniform(0.2, 0.8))
        target = low + jitter * (high - low)
        detail = {"corridor_stratum": [low, high]}
        if selector.take_nearest(target, low, high, "corridor", detail) is None:
            # Widen once: a corridor bin can be fully blocked when v2 and the
            # original anchors both landed in it.
            if selector.take_nearest(
                target, low - 30, high + 30, "corridor", detail
            ) is None:
                raise ValueError(
                    f"{selector.camera}: corridor bin {low}-{high} has no free ordinal"
                )
    return count


def select_queries(
    camera: str,
    count: int,
    route_start: int,
    route_end_exclusive: int,
    blocked: set[int],
    seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Returns (queries, shortfalls).

    A targeted stratum can run out of room - the free ordinals near the kinks of
    an already-labelled camera are finite, and each new set excludes ten frames
    around every ordinal any earlier set used. When that happens the unfilled
    slots move to the corridor stratum, which can always be placed, and the move
    is recorded rather than silently absorbed: a set with 7 kink and 12 corridor
    queries is a different instrument from one with 10 and 9.
    """

    plan = dict(STRATA_PLAN[camera])
    if sum(plan.values()) != count:
        # Scale the corridor stratum so --queries-per-camera stays usable.
        plan["corridor"] += count - sum(plan.values())
        if plan["corridor"] < 0:
            raise ValueError(f"{camera}: {count} queries cannot hold the targeted strata")

    rng = np.random.default_rng(seed + CAMERAS.index(camera))
    selector = Selector(camera, route_start, route_end_exclusive, blocked, rng)

    windows = occlusion_windows()
    shortfalls: list[dict[str, object]] = []
    corridor_wanted = plan.get("corridor", 0)
    for stratum in STRATUM_ORDER:
        wanted = plan.get(stratum, 0)
        if stratum == "corridor":
            continue
        if stratum == "occlusion":
            placed = (
                select_occlusion(selector, wanted, windows[camera])
                if wanted and camera in windows
                else 0
            )
        elif stratum == "dark":
            placed = select_dark(
                selector,
                wanted,
                dark_stretches_for_selection(camera),
                submitted_partners(camera),
            )
        else:
            placed = select_kink(selector, wanted, submitted_kinks(camera))
        if placed < wanted:
            shortfalls.append(
                {
                    "stratum": stratum,
                    "planned": wanted,
                    "placed": placed,
                    "moved_to_corridor": wanted - placed,
                    "why": (
                        "no free Run A ordinal remained in this stratum at least "
                        f"{MINIMUM_DISTANCE_FROM_LABELLED} frames from every "
                        "previously labelled ordinal and "
                        f"{MINIMUM_DISTANCE_BETWEEN_QUERIES} from every other query"
                    ),
                }
            )
            corridor_wanted += wanted - placed
    select_corridor(selector, corridor_wanted)

    return (
        sorted(selector.chosen, key=lambda entry: int(entry["runA_frame"])),
        shortfalls,
    )


def dark_stretches_for_selection(camera: str) -> list[tuple[int, int]]:
    """Dark stretches with the confirmed occlusion removed.

    On CAM0 the darkest Run B stretches sit at 236-275, which is inside the
    occlusion the occlusion stratum already covers. Sampling both from there
    would spend seven of twenty-four queries on thirty frames of route and
    would not tell darkness and occlusion apart afterwards.
    """

    stretches = dark_run_b_stretches(camera)
    windows = occlusion_windows()
    if camera not in windows:
        return stretches
    low, high = windows[camera]["runB_start"], windows[camera]["runB_end"]
    return [
        stretch for stretch in stretches if stretch[1] < low or stretch[0] > high
    ]


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


# --------------------------------------------------------------------------
# Frame export
# --------------------------------------------------------------------------


def select_expression(ordinals: list[int]) -> str:
    """Compact FFmpeg select expression; contiguous runs collapse to between()."""

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


# --------------------------------------------------------------------------
# Page payload
# --------------------------------------------------------------------------


POSE_CONVENTION = (
    "compensation = -displacement; positive moves Run B right/down. "
    "measuredDxPx/measuredDyPx are the raw measured scene displacement of Run B "
    "relative to Run A on the measurement raster; dxPx/dyPx are the translate() "
    "the page applies to the Run B image to undo it, on the filmstrip raster."
)


def pose_defaults(camera: str) -> dict[str, object]:
    """The translate() the page must apply to Run B, in filmstrip pixels.

    The diagnostic reports a *displacement*: CAM5's scene sits 37.5 px left and
    11 px up in Run B relative to Run A. Undoing that means moving Run B the
    other way, so the exported compensation is the negative of the measurement:
    +40, +12 at 960x720. Shipping the raw measurement here is the v3 defect
    recorded in outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md, which
    doubled the misalignment instead of removing it.
    """

    dx_measured, dy_measured = measured_pose_shift(camera)
    raster_w, raster_h = POSE_MEASUREMENT_RASTER
    dx_fraction = dx_measured / raster_w
    dy_fraction = dy_measured / raster_h
    return {
        "measuredDxPx": dx_measured,
        "measuredDyPx": dy_measured,
        "measuredRaster": [raster_w, raster_h],
        "dxPx": -round(dx_fraction * FILMSTRIP_WIDTH),
        "dyPx": -round(dy_fraction * FILMSTRIP_HEIGHT),
        "compensated": bool(POSE_COMPENSATION_DEFAULT_ON[camera]),
        "sliderRangePx": POSE_SLIDER_RANGE_PX,
        "convention": POSE_CONVENTION,
    }


def build_page(
    camera: str,
    queries: list[dict[str, object]],
    run_b_min: int,
    run_b_max: int,
    built_at: str,
    set_name: str = DEFAULT_SET_NAME,
) -> str:
    payload = {
        "camera": camera,
        "labelSet": set_name,
        # Deliberately only three numbers per query. No stratum, no partner, no
        # prediction: everything else lives in the manifest.
        "queries": [
            {
                "queryId": query["queryId"],
                "runAFrame": int(query["runAFrame"]),
                "linearPriorRunBFrame": int(query["linearPriorRunBFrame"]),
            }
            for query in queries
        ],
        "runBMin": run_b_min,
        "runBMax": run_b_max,
        "filmstrip": {"width": FILMSTRIP_WIDTH, "height": FILMSTRIP_HEIGHT},
        "pose": pose_defaults(camera),
        "guideBands": {"far": list(FAR_GUIDE_BAND), "near": list(NEAR_GUIDE_BAND)},
        "builtAtUtc": built_at,
    }
    return (
        _PAGE_TEMPLATE
        .replace("__PAYLOAD__", json.dumps(payload))
        .replace("__CSV_HEADER__", ",".join(CSV_COLUMNS))
    )


def query_set_sha256(
    cameras: dict[str, dict[str, object]],
    seed: int = RANDOM_SEED,
    set_name: str = DEFAULT_SET_NAME,
) -> str:
    """Hash of exactly which Run A frames were asked, in which stratum, and how.

    The seal cites this, so it stays checkable that the sealed labels answer the
    queries that were exported rather than a regenerated or edited set.
    """

    canonical = {
        camera: sorted(
            [int(query["runAFrame"]), str(query["stratum"])]
            for query in entry["queries"]
        )
        for camera, entry in cameras.items()
    }
    payload = json.dumps(
        {"seed": seed, "label_set": set_name, "queries": canonical},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def build_camera(
    camera: str,
    count: int,
    bounds: dict[str, int],
    counts: dict,
    seed: int = RANDOM_SEED,
    set_name: str = DEFAULT_SET_NAME,
) -> dict:
    run_a_start = bounds["runA_start"]
    run_b_start = bounds["runB_start"]
    run_a_end = unified.RUN_A_ROUTE_TAIL_EXCLUSIVE
    run_b_end = counts[("runB", camera)] - 1

    sources = previously_labelled(camera, set_name)
    blocked: set[int] = set()
    for ordinals in sources.values():
        blocked |= ordinals

    selected, shortfalls = select_queries(
        camera, count, run_a_start, run_a_end, blocked, seed
    )

    tag = version_tag(set_name)
    queries: list[dict[str, object]] = []
    for index, entry in enumerate(selected):
        run_a_frame = int(entry["runA_frame"])
        record = {
            "queryId": f"{camera}_{tag}_{index:02d}",
            "runAFrame": run_a_frame,
            "linearPriorRunBFrame": linear_progress_prior(
                run_a_frame, run_a_start, run_a_end, run_b_start, run_b_end
            ),
        }
        record.update({key: value for key, value in entry.items() if key != "runA_frame"})
        queries.append(record)

    return {
        "queries": queries,
        "runA_route_start": run_a_start,
        "runA_route_end_exclusive": run_a_end,
        "runB_route_start": run_b_start,
        "runB_route_end": run_b_end,
        "strata_counts": {
            stratum: sum(1 for query in queries if query["stratum"] == stratum)
            for stratum in STRATUM_ORDER
        },
        "strata_shortfalls": shortfalls,
        "previously_labelled_ordinals_blocked": {
            source: len(ordinals) for source, ordinals in sources.items()
        },
        "pose_defaults": pose_defaults(camera),
    }


def selection_only(
    count: int = QUERIES_PER_CAMERA,
    cameras: tuple[str, ...] = CAMERAS,
    seed: int = RANDOM_SEED,
    set_name: str = DEFAULT_SET_NAME,
) -> dict[str, dict[str, object]]:
    """Selection without touching ffmpeg. Used by the tests."""

    bounds = unified.route_bounds()
    counts = frame_counts()
    return {
        camera: build_camera(camera, count, bounds, counts, seed, set_name)
        for camera in cameras
    }


def main() -> None:
    arguments = parse_args()
    set_name = arguments.set_name
    cameras = tuple(dict.fromkeys(arguments.camera)) if arguments.camera else CAMERAS
    seed = arguments.seed
    out_dir = output_dir(set_name)
    served_dir = pages_dir(set_name)

    if out_dir.exists() and not arguments.force and not arguments.skip_filmstrip:
        raise FileExistsError(
            f"{out_dir} already exists; pass --force to rebuild. Rebuilding "
            "changes which frames are queried and invalidates any labelling in progress."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    served_dir.mkdir(parents=True, exist_ok=True)

    bounds = unified.route_bounds()
    counts = frame_counts()

    target_manifest = manifest_path(set_name)
    previous = (
        json.loads(target_manifest.read_text()) if target_manifest.is_file() else {}
    )
    now = datetime.now(timezone.utc).isoformat()
    built_at = previous.get("built_at_utc", now) if arguments.skip_filmstrip else now

    manifest_cameras: dict[str, dict[str, object]] = {}
    for camera in cameras:
        entry = build_camera(
            camera, arguments.queries_per_camera, bounds, counts, seed, set_name
        )
        manifest_cameras[camera] = entry

        query_frames = [int(query["runAFrame"]) for query in entry["queries"]]
        run_b_frames = list(range(entry["runB_route_start"], entry["runB_route_end"] + 1))
        camera_dir = served_dir / camera
        if not arguments.skip_filmstrip:
            print(
                f"[{camera}] extracting {len(query_frames)} Run A query frames "
                "at native resolution"
            )
            extract_frames(
                "runA", camera, query_frames, camera_dir / "runA",
                None, None, QUERY_JPEG_QUALITY,
            )
            print(
                f"[{camera}] extracting {len(run_b_frames)} Run B filmstrip frames "
                f"at {FILMSTRIP_WIDTH}x{FILMSTRIP_HEIGHT}"
            )
            extract_frames(
                "runB", camera, run_b_frames, camera_dir / "runB",
                FILMSTRIP_WIDTH, FILMSTRIP_HEIGHT, FILMSTRIP_JPEG_QUALITY,
            )

        page = build_page(
            camera,
            entry["queries"],
            entry["runB_route_start"],
            entry["runB_route_end"],
            built_at,
            set_name,
        )
        (served_dir / f"annotate_{camera}.html").write_text(page)
        entry["runB_filmstrip_frames"] = len(run_b_frames)

    manifest = {
        "schema_version": 3,
        "label_set": set_name,
        "cameras_exported": list(cameras),
        "built_at_utc": built_at,
        "pages_regenerated_at_utc": now if arguments.skip_filmstrip else None,
        "query_set_sha256": query_set_sha256(manifest_cameras, seed, set_name),
        "random_seed": seed,
        "queries_per_camera": arguments.queries_per_camera,
        "strata_plan": STRATA_PLAN,
        "sampling_rules": {
            "strata": {
                "kink": (
                    f"Run A frames within +-{KINK_QUERY_RADIUS} of a submitted-mapping "
                    f"kink of at least {KINK_INCREMENT_THRESHOLD} frames; lit kinks "
                    "preferred; spread by farthest-point ordering; at most one query "
                    "per kink"
                ),
                "dark": (
                    "Run A frames whose submitted partner falls inside a contiguous "
                    f"Run B too_dark stretch of at least {MINIMUM_DARK_STRETCH} frames, "
                    "longest stretches first"
                ),
                "occlusion": (
                    "Run A frames inside the confirmed Run B occlusion. CAM0 only: the "
                    "occlusion was observed on CAM0's side of the vehicle, so there is "
                    "no evidence for a matching CAM5 stratum. CAM5's three slots were "
                    "reallocated to two extra kink queries and one extra dark query."
                ),
                "corridor": (
                    "uniformly spaced bins over the whole route with a seeded jitter "
                    "inside each bin; the untargeted control stratum"
                ),
            },
            "excluded_idle": (
                f"Run A frames outside [{bounds['runA_start']}, "
                f"{unified.RUN_A_ROUTE_TAIL_EXCLUSIVE}) are parked and carry no "
                "travelling correspondence"
            ),
            "minimum_distance_from_previously_labelled": MINIMUM_DISTANCE_FROM_LABELLED,
            "previously_labelled_sources": [
                str(LABEL_SOURCE_FILES[name].relative_to(ROOT))
                for name in exclusion_sources(set_name)
            ],
            "stratum_shortfall_policy": (
                "A targeted stratum with no free ordinal left moves its unfilled "
                "slots to the corridor stratum. Every move is recorded per camera "
                "under strata_shortfalls, because a set with fewer kink queries "
                "than planned is a weaker test of kinks and the reported "
                "per-stratum counts must be the ones actually exported."
            ),
            "minimum_distance_between_queries": MINIMUM_DISTANCE_BETWEEN_QUERIES,
            "minimum_distance_between_queries_within_occlusion": (
                MINIMUM_DISTANCE_WITHIN_OCCLUSION
            ),
            "occlusion_spacing_note": (
                "The confirmed occlusion spans 30 Run A frames and two nearby ordinals "
                "are already labelled, so three queries cannot be 10 apart inside it. "
                "Only the query-to-query spacing is relaxed there; the 10-frame "
                "distance to every previously labelled ordinal is not."
            ),
        },
        "stratum_dependency_disclosure": (
            "Strata were located using the structure of the submitted unified mapping: "
            "its kink positions and, for the dark stratum, its Run A -> Run B partner "
            "for each frame. This affects WHICH frames are asked about. It does not "
            "affect WHAT the annotator sees: the page carries the Run A frame, a linear "
            "route-progress start position computed from the route boundaries alone, and "
            "the Run B filmstrip. No stratum tag, no mapping partner, no confidence and "
            "no prediction is present in the exported HTML, and a test greps the pages "
            "for the submitted partner of every query and fails if one appears. The "
            "consequence to state when reporting: per-stratum rates are conditional on "
            "the submitted mapping having put its kinks where it did, so they measure "
            "'is this method right where it says something surprising', not an unbiased "
            "route-wide rate. The corridor stratum is the unbiased comparison."
        ),
        "prior_policy": (
            "linear route-progress arithmetic only; contains no model output, no "
            "baseline path, and no previous prediction"
        ),
        "served_root": {
            "path": str(served_dir.relative_to(ROOT)),
            "why": (
                "The annotator's http.server is pointed one level below this "
                "manifest. The manifest names each query's stratum and the dark "
                "stratum's Run B window; a directory listing at the served root must "
                "not be able to reach it."
            ),
        },
        "blindness": {
            "model_columns_present": False,
            "predictions_present": False,
            "previous_labels_present": False,
            "stratum_shown_to_annotator": False,
            "automatic_best_frame_hint": False,
        },
        "annotator_aids": {
            "pose_compensation": (
                "Two-axis translation of the Run B image with sliders. "
                f"{POSE_CONVENTION} The measured CAM5 displacement (-37.5, -11) px at "
                f"896x672 therefore exports as a compensation of "
                f"({pose_defaults('cam5')['dxPx']}, {pose_defaults('cam5')['dyPx']}) px "
                f"at {FILMSTRIP_WIDTH}x{FILMSTRIP_HEIGHT}; CAM0 is (0, 0). ON by default "
                "for CAM5, OFF for CAM0. The chosen value is written into every label."
            ),
            "pose_compensation_sign_history": (
                "The first v3 export shipped the raw measurement as the compensation, "
                "so the page moved Run B a further 40 px left and doubled the CAM5 "
                "misalignment. The annotator then used the slider to recover, which "
                "let frame error and shift trade off against each other. See "
                "outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md."
            ),
            "pose_slider_clamp": (
                f"Both sliders are clamped to the exported default +-"
                f"{POSE_SLIDER_RANGE_PX} px and the default is drawn as a tick. A "
                "persistent warning under them says the slider may not be used to make "
                "a frame fit: change the frame until the far and near residuals are "
                "equal, and only then take a small slider tweak to zero both."
            ),
            "guide_band_rule": (
                f"The far guide must be placed in the top {FAR_GUIDE_BAND[1]:.0%} of the "
                f"frame and the near guide in the bottom "
                f"{1 - NEAR_GUIDE_BAND[0]:.0%}; both bands are drawn faintly and the "
                "page refuses to save a label while either guide is outside its band, "
                "with a tooltip saying which one and why. Two guides at the same depth "
                "cannot break the far + near rule apart."
            ),
            "depth_guides": (
                "A far line and a near line the annotator drags onto Run A. Both lines "
                "are drawn in Run A image coordinates in both panes; the compensation "
                "moves the Run B raster onto those coordinates rather than duplicating "
                "the shift on the lines, which would cancel it. At the correct Run B "
                "frame one common shift aligns both depths; parallax breaks the near "
                "line first. Each guide carries an x and a y: the vertical line at x is "
                "the alignment cue, the marker at y says which depth band the feature "
                "was taken from. Both are saved per query as fractions of frame width "
                "and height."
            ),
            "band_blink": (
                "Blink restricted to the far band (top 45%) or the near band (bottom "
                "third), alongside the full-frame blend, difference and blink modes."
            ),
            "no_suggestion": (
                "No best-frame hint, ghost-direction arrow or ranked candidate is "
                "computed anywhere on the page. Any such aid would leak a matching "
                "method into labels whose only value is being independent of one."
            ),
        },
        "protocol_fix_vs_v2": (
            "v2 offered a single 'no usable match' button, so its no-match rows mix "
            "'I cannot tell' with 'there is no such place' and can be scored as neither "
            "an abstention target nor an error. v3 splits them: label is one of match, "
            "undetermined, or no_correspondence, and the reason is a required choice "
            "rather than a free-text afterthought."
        ),
        "pose_convention": POSE_CONVENTION,
        "cameras": manifest_cameras,
    }
    target_manifest.write_text(json.dumps(manifest, indent=2))
    (out_dir / "README.md").write_text(build_readme(manifest, set_name, cameras))

    print()
    print(f"Wrote {out_dir.relative_to(ROOT)}")
    for camera in cameras:
        counts_by_stratum = manifest_cameras[camera]["strata_counts"]
        print(f"  annotate_{camera}.html   {counts_by_stratum}")
        for shortfall in manifest_cameras[camera]["strata_shortfalls"]:
            print(
                f"    shortfall: {shortfall['stratum']} placed "
                f"{shortfall['placed']}/{shortfall['planned']}, "
                f"{shortfall['moved_to_corridor']} moved to corridor"
            )
    print(f"  query_set_sha256 {manifest['query_set_sha256']}")
    print(f"  built_at_utc     {built_at}")
    print(
        f"  pose default     cam5 "
        f"({pose_defaults('cam5')['dxPx']}, {pose_defaults('cam5')['dyPx']}) px "
        f"(compensation = -displacement)"
    )
    print()
    print(f"cd {served_dir.relative_to(ROOT)} && python3 -m http.server 8732")
    for camera in cameras:
        print(f"  http://localhost:8732/annotate_{camera}.html")


def build_readme(
    manifest: dict[str, object], set_name: str, cameras: tuple[str, ...]
) -> str:
    """The set's own README. Generated so it cannot drift from the manifest."""

    tag = version_tag(set_name)
    rows = []
    for stratum in ("kink", "dark", "occlusion", "corridor"):
        counts = " | ".join(
            str(manifest["cameras"][camera]["strata_counts"][stratum])
            for camera in cameras
        )
        rows.append(f"| `{stratum}` | {counts} |")
    shortfall_lines = []
    for camera in cameras:
        for shortfall in manifest["cameras"][camera]["strata_shortfalls"]:
            shortfall_lines.append(
                f"- **{camera} `{shortfall['stratum']}`**: {shortfall['placed']} of "
                f"{shortfall['planned']} placed, {shortfall['moved_to_corridor']} "
                f"moved to `corridor` — {shortfall['why']}."
            )
    shortfall_block = (
        "\n".join(shortfall_lines)
        if shortfall_lines
        else "None: every stratum was filled as planned."
    )
    camera_urls = "\n".join(
        f"- http://localhost:8732/annotate_{camera}.html" for camera in cameras
    )
    importer_flags = " \\\n  ".join(
        f"--{camera} ~/Downloads/{set_name}_{camera}.csv" for camera in cameras
    )
    pose = pose_defaults(cameras[0])
    cam5_pose = pose_defaults("cam5")
    per_camera = manifest["queries_per_camera"]
    return f"""# Blind label set {tag} — {", ".join(cameras)}

{per_camera} Run A query frames per camera. Every ordinal is at least
{MINIMUM_DISTANCE_FROM_LABELLED} frames from anything a human has already been
shown and at least {MINIMUM_DISTANCE_BETWEEN_QUERIES} frames from every other
query in this set, and the parked prefix and tail of Run A are excluded, so
every query is a frame the vehicle was actually moving through.

Excluded sources:

{chr(10).join("- `" + source + "`" for source in manifest["sampling_rules"]["previously_labelled_sources"])}

This file is **not** inside the served directory. Neither is
`{set_name}_manifest.json`. Both name each query's stratum, and the manifest also
records the Run B window the dark stratum was drawn from — close enough to an
answer that a directory listing at the annotator's `localhost` must not reach
it. If you are the annotator, stop reading here and use the on-page
instructions.

- `query_set_sha256`: `{manifest["query_set_sha256"]}`
- `built_at_utc`: `{manifest["built_at_utc"]}`
- `random_seed`: `{manifest["random_seed"]}`

## Start the annotator

```bash
cd {pages_dir(set_name).relative_to(ROOT)} && python3 -m http.server 8732
```

{camera_urls}

A local server is required: browsers treat `file://` as an opaque origin, so
answers would stop persisting between reloads.

## Strata

| stratum | {" | ".join(cameras)} |
|---|{"---|" * len(cameras)}
{chr(10).join(rows)}

Shortfalls (a targeted stratum with no free ordinal left moves its slots to
`corridor`):

{shortfall_block}

The strata were located using the submitted mapping — its kink positions, and
its Run A → Run B partner for each frame. That decides *which frames are asked
about*, never *what the annotator sees*: the pages carry the Run A frame, a
linear route-progress start position computed from the route boundaries alone,
and the Run B filmstrip. No stratum tag, no partner, no confidence, no
prediction. `scripts/test_blind_label_set_v3.py` greps each exported page for
the submitted partner of every query and fails if one appears.

## Pose compensation — the sign

{POSE_CONVENTION}

CAM5's measured displacement is `(-37.5, -11)` px at 896×672, so the exported
compensation is `({cam5_pose["dxPx"]}, {cam5_pose["dyPx"]})` px at
{FILMSTRIP_WIDTH}×{FILMSTRIP_HEIGHT}: Run B moves right and down. The first v3
export shipped the raw measurement instead and doubled the CAM5 misalignment;
see `outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`.

## What the page enforces

1. **The sliders are clamped** to the exported default ±{POSE_SLIDER_RANGE_PX} px,
   with the default drawn as a tick, and a warning line sits under them
   permanently: *do not use the slider to make a frame fit — change the frame
   until FAR and NEAR residuals are equal; only then may a small slider tweak
   zero both*.
2. **The guides must straddle a real depth difference.** The far guide has to be
   in the top {FAR_GUIDE_BAND[1]:.0%} of the frame and the near guide in the
   bottom {1 - NEAR_GUIDE_BAND[0]:.0%}. Both bands are drawn faintly; while
   either guide is outside its band every label button is disabled and its
   tooltip says which guide and why.
3. **Every label records the slider.** `pose_dx_px` and `pose_dy_px` are written
   into the CSV per label, in filmstrip pixels, alongside the guide positions
   and the compare mode in use.
4. **No automatic best-frame hint of any kind**, anywhere on the page.

Interval controls are exact, ±1, ±2 and ±3. Widen honestly: a defensible ±3 is
better evidence than a ±0 nobody can stand behind.

| Key | Action |
|---|---|
| `←` `→` | step one frame |
| `Shift` + `←` `→` | step ten frames |
| `Enter` | accept exact |
| `1` `2` `3` | accept ±1 / ±2 / ±3 |
| `U` | cannot determine |
| `X` | no corresponding place |
| `[` `]` | previous / next query |
| `G` | cycle compare mode |
| `P` | toggle pose compensation |

## Labels

| label | meaning |
|---|---|
| `match` | a Run B frame shows the same world position, within the recorded interval |
| `undetermined` | a counterpart may exist; the frame could not be identified |
| `no_correspondence` | no Run B frame shows this place — occluded, or not traversed |

## When the annotation is finished

Click **Download CSV** on each page, then:

```bash
PYTHONPATH=scripts python3 scripts/import_blind_labels_v3.py \\
  --set-name {set_name} \\
  {importer_flags}
```

The importer validates blindness, rejects any populated model or stratum column,
rejects an unanswered query, rejects an abstention that still names a Run B
frame, and seals the file with a SHA-256 that cites the query set hash and the
export timestamp. Do not run it before the labelling is complete.

CSV columns:

```
{", ".join(CSV_COLUMNS)}
```

`pose_dx_px` and `pose_dy_px` are in filmstrip pixels
({FILMSTRIP_WIDTH}×{FILMSTRIP_HEIGHT} basis), positive = Run B moved right/down.
The guide columns are fractions of frame width and height.

## Reproducing the export

```bash
PYTHONPATH=scripts python3 scripts/build_blind_label_set_v3.py \\
  {" ".join("--camera " + camera for camera in cameras)} \\
  --set-name {set_name} --seed {manifest["random_seed"]} --force
PYTHONPATH=scripts python3 scripts/test_blind_label_set_v3.py
```

The selection is seeded and deterministic; `query_set_sha256` covers the seed,
the set name, and every query's Run A frame and stratum. Rebuilding resets any
labelling in progress, which is why `--force` is required.

Pose default in this set: compensation
`({pose["dxPx"]}, {pose["dyPx"]})` px for `{cameras[0]}`, slider range
`±{POSE_SLIDER_RANGE_PX}` px.
"""


_PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind labelling v3</title>
<style>
:root{
  --bg:#11161a; --panel:#1a2228; --line:#2c383f; --ink:#e8eef1; --muted:#8fa3ad;
  --accent:#5ac8d8; --ok:#5fc98f; --warn:#e0a94b; --crit:#f0796c; --far:#8fd3ff; --near:#ffc46b;
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
button[disabled]{opacity:.45;cursor:not-allowed}
details.guide{margin:12px 14px 0;background:var(--panel);border:1px solid var(--line);
  border-radius:5px;padding:10px 14px}
details.guide summary{cursor:pointer;font-weight:700;font-size:12px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted)}
details.guide ol{margin:10px 0 4px;padding-left:20px}
details.guide li{margin-bottom:7px}
details.guide .warn{color:var(--warn)}
main{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:5px;overflow:hidden;
  display:flex;flex-direction:column}
.pane h2{margin:0;padding:9px 13px;font-size:11px;letter-spacing:.11em;text-transform:uppercase;
  color:var(--muted);border-bottom:1px solid var(--line);font-weight:700;
  display:flex;justify-content:space-between;align-items:center}
.pane h2 span{color:var(--ink);font:700 13px/1 ui-monospace,monospace;letter-spacing:0}
.stage{position:relative;width:100%;aspect-ratio:4 / 3;background:#000;overflow:hidden;
  touch-action:none}
.stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block}
#overlayImage{opacity:0;pointer-events:none}
.lines{position:absolute;inset:0;pointer-events:none}
.lines i{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px}
.lines i.far{background:var(--far);color:var(--far);box-shadow:0 0 0 1px rgba(0,0,0,.55)}
.lines i.near{background:var(--near);color:var(--near);box-shadow:0 0 0 1px rgba(0,0,0,.55)}
.lines i.out{background:var(--crit);color:var(--crit)}
.lines i s{position:absolute;left:-9px;width:20px;height:2px;margin-top:-1px;
  background:currentColor;text-decoration:none;box-shadow:0 0 0 1px rgba(0,0,0,.55)}
.lines i b{position:absolute;left:7px;font:700 10px/1.4 ui-monospace,monospace;
  background:rgba(0,0,0,.6);padding:1px 4px;border-radius:2px;color:currentColor;
  transform:translateY(-50%);white-space:nowrap}
/* The two depth bands, drawn faintly and always: a guide outside its band is
   not a depth reference and the far + near rule cannot separate them. */
.guidebands{position:absolute;inset:0;pointer-events:none}
.guidebands u{position:absolute;left:0;right:0;display:block;text-decoration:none}
.guidebands u.far{top:0;background:rgba(143,211,255,.07);
  border-bottom:1px dashed rgba(143,211,255,.30)}
.guidebands u.near{bottom:0;background:rgba(255,196,107,.07);
  border-top:1px dashed rgba(255,196,107,.30)}
.warnline{color:var(--warn);font-size:12px;border-left:2px solid var(--warn);
  padding:2px 0 2px 8px;margin:1px 0}
.blockline{color:var(--crit);font-size:12px;font-weight:650;min-height:1px}
.bandmask{position:absolute;inset:0;pointer-events:none;display:none}
.bandmask.on{display:block}
.bandmask u{position:absolute;left:0;right:0;background:rgba(17,22,26,.88);display:block}
select{background:#0d1418;color:var(--ink);border:1px solid var(--line);border-radius:3px;
  padding:5px 7px;font:600 12px/1 inherit}
.checkline{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted)}
.controls{padding:11px 13px;border-top:1px solid var(--line);display:flex;flex-direction:column;gap:9px}
.row{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
.row label.name{min-width:96px;color:var(--muted);font-size:12px;font-weight:650}
.row output{min-width:56px;color:var(--ink);font:700 12px/1 ui-monospace,monospace}
input[type=range]{flex:1;min-width:140px;accent-color:var(--accent)}
input[type=number]{width:84px;background:#0d1418;color:var(--ink);
  border:1px solid var(--line);border-radius:3px;padding:5px 7px;font:600 13px/1 ui-monospace,monospace}
.hint{color:var(--muted);font-size:12px}
.readout{font:700 12px/1.4 ui-monospace,monospace}
.readout .f{color:var(--far)} .readout .n{color:var(--near)}
kbd{background:#0d1418;border:1px solid var(--line);border-bottom-width:2px;border-radius:3px;
  padding:1px 5px;font:600 11px/1.5 ui-monospace,monospace;color:var(--ink)}
fieldset{border:1px solid var(--line);border-radius:4px;padding:7px 10px;margin:0}
fieldset.flash{border-color:var(--crit);background:#2b1a18}
legend{font:700 10px/1 ui-monospace,monospace;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);padding:0 5px}
fieldset label{display:block;font-size:12px;margin:3px 0;cursor:pointer}
.status{padding:9px 16px;border-top:1px solid var(--line);background:var(--panel);
  display:flex;gap:14px;flex-wrap:wrap;align-items:center;position:sticky;bottom:0}
.pill{font:600 11px/1 ui-monospace,monospace;padding:5px 9px;border-radius:3px}
.pill.done{background:#12301f;color:var(--ok)}
.pill.todo{background:#372a12;color:var(--warn)}
.pill.nomatch{background:#3a1c19;color:var(--crit)}
.pill.undet{background:#33291c;color:var(--warn)}
.grid{display:flex;gap:3px;flex-wrap:wrap;padding:9px 16px;background:var(--panel);
  border-top:1px solid var(--line)}
.cell{width:24px;height:24px;border-radius:2px;border:1px solid var(--line);cursor:pointer;
  font:600 9px/22px ui-monospace,monospace;text-align:center;color:var(--muted);background:#0d1418}
.cell.done{background:#12301f;color:var(--ok);border-color:#1e5236}
.cell.undet{background:#33291c;color:var(--warn);border-color:#5c4a26}
.cell.nomatch{background:#3a1c19;color:var(--crit);border-color:#5c2f29}
.cell.active{outline:2px solid var(--accent);outline-offset:1px}
@media (max-width:820px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Blind labelling v3 — <span id="cameraName"></span></h1>
  <span class="chip">query <b id="queryPos"></b></span>
  <span class="chip">run A frame <b id="runAFrame"></b></span>
  <span class="chip">labelled <b id="doneCount"></b></span>
  <button id="prevBtn">← Prev</button>
  <button id="nextBtn">Next →</button>
  <button id="saveBtn" class="primary">Download CSV</button>
</header>

<details class="guide" open>
  <summary>How to label — read once</summary>
  <ol>
    <li><b>What you are deciding.</b> For the Run A frame on the left, find the Run B
      frame taken from <b>the same world position</b> — the point on the road the
      vehicle had reached, not the point where the picture looks prettiest. If the
      vehicle is one metre further on, that is a different frame even if the scene
      looks nearly identical.</li>
    <li><b>Start position.</b> Run B opens at a linear route-progress guess. It is
      arithmetic on the route boundaries — no model, no previous answer — and it is
      deliberately weak: the true offset drifts across roughly 124 frames over the
      route. Expect to scrub well away from it. You can reach any Run B route frame.</li>
    <li><b>Place the two guides.</b> Drag the blue <span class="readout f">far</span>
      guide onto something distant in Run A — a building edge, a pole top, a roofline
      — inside the faint blue band across the top of the frame. Drag the amber
      <span class="readout n">near</span> guide onto something close — a kerb feature,
      a driveway edge, a road marking — inside the faint amber band along the bottom.
      Both guides are drawn at the same place in both panes. <b>You cannot record a
      label while a guide is outside its band</b>; the buttons go dead and say why.
      That is deliberate: a "far" reference picked off the near road surface is a
      near reference, and then nothing on the page can pin the frame.</li>
    <li><b>The far + near rule.</b> At the correct B frame one common shift aligns
      BOTH the far and the near feature. If only one aligns, scrub. Near things sweep
      past far faster than far things, so the near guide is what pins the exact frame;
      the far guide is what tells you that you are in the right place at all.</li>
    <li><b>Pose compensation, and what it is not.</b> The <i>Compensate B</i> control
      slides the whole Run B image by one common (dx, dy) to remove the constant
      camera-pointing difference between the runs. It starts at the measured value and
      the sliders move only a little either side of it, because a shift and a frame
      change look the same on one depth and only differ across two.
      <span class="warn">Do not use the slider to make a frame fit. Change the frame
      until FAR and NEAR residuals are equal; only then may a small slider tweak zero
      both.</span> <span class="warn" id="cam5Note"></span></li>
    <li><b>Compare modes.</b> <i>Blend</i> and <i>Difference</i> show A over B.
      <i>Blink</i> alternates them. <i>Blink far band</i> shows only the top 45% and
      <i>Blink near band</i> only the bottom third, so you can settle one depth at a
      time instead of letting the two argue.</li>
    <li><b>Recording an answer.</b> <i>Accept exact</i> when you can name the frame.
      <i>±1</i>, <i>±2</i>, <i>±3</i> when you are sure of the place but not the
      frame — widen honestly, a defensible ±3 is better evidence than a ±0 you
      cannot stand behind.</li>
    <li><b>No usable match</b> needs a reason, and the two reasons mean different
      things. <i>Cannot determine</i> = a counterpart may well exist, you cannot tell
      which frame it is. <i>No corresponding place</i> = there is no such Run B frame,
      because something blocked the view or the route was not driven there. Do not use
      the first when you mean the second.</li>
    <li><b>Keys.</b> <kbd>←</kbd><kbd>→</kbd> step 1 · <kbd>Shift</kbd>+arrows step 10 ·
      <kbd>Enter</kbd> accept exact · <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> accept ±n ·
      <kbd>U</kbd> cannot determine · <kbd>X</kbd> no corresponding place ·
      <kbd>[</kbd><kbd>]</kbd> prev/next query · <kbd>G</kbd> cycle compare mode ·
      <kbd>P</kbd> toggle compensation.</li>
  </ol>
</details>

<main>
  <div class="pane">
    <h2>Run A · query <span id="runALabel"></span></h2>
    <div class="stage" id="stageA">
      <img id="runAImage" alt="Run A query frame">
      <div class="guidebands" id="bandsA"><u class="far"></u><u class="near"></u></div>
      <div class="lines" id="linesA"></div>
      <div class="bandmask" id="bandA"><u></u><u></u></div>
    </div>
    <div class="controls">
      <div class="row">
        <span class="readout"><span class="f">far guide</span> <b id="farReadout"></b></span>
        <span class="readout"><span class="n">near guide</span> <b id="nearReadout"></b></span>
        <button id="resetLinesBtn" type="button">Reset guides</button>
      </div>
      <div class="hint">Drag on this panel to move the nearer guide to your pointer.
        Each guide has an x (the vertical line you align on) and a y (the feature you
        picked it from). Positions are saved per query.</div>
      <div class="hint">The blue band is the far band and the amber band is the near
        band. The far guide must sit inside the far band and the near guide inside the
        near band: two guides at the same depth cannot tell you anything, because the
        whole test is that near sweeps faster than far.</div>
      <div class="blockline" id="guideWarning"></div>
      <div class="hint">This is the place to find. One common shift must align the far
        guide's feature and the near guide's feature at the same B frame.</div>
    </div>
  </div>

  <div class="pane">
    <h2>Run B · candidate <span id="runBLabel"></span></h2>
    <div class="stage" id="stageB">
      <img id="runBImage" alt="Run B candidate frame">
      <img id="overlayImage" alt="Run A overlaid for alignment">
      <div class="guidebands" id="bandsB"><u class="far"></u><u class="near"></u></div>
      <div class="lines" id="linesB"></div>
      <div class="bandmask" id="bandB"><u></u><u></u></div>
    </div>
    <div class="controls">
      <div class="row">
        <input type="range" id="scrub" min="0" max="0" step="1">
        <input type="number" id="frameInput">
      </div>
      <div class="row">
        <label class="name" for="runBBrightness">B brightness</label>
        <input type="range" id="runBBrightness" min="50" max="250" step="5" value="100">
        <output id="runBBrightnessValue" for="runBBrightness">100%</output>
        <button id="brightnessResetBtn" type="button">Reset</button>
      </div>
      <div class="row">
        <label class="name" for="compareMode">Compare</label>
        <select id="compareMode">
          <option value="side">Side by side</option>
          <option value="blend">Blend 50/50</option>
          <option value="diff">Difference</option>
          <option value="blink">Blink A/B</option>
          <option value="blink_far">Blink far band (top 45%)</option>
          <option value="blink_near">Blink near band (bottom third)</option>
        </select>
        <label class="checkline"><input type="checkbox" id="poseOn"> Compensate B</label>
        <button id="poseResetBtn" type="button">Reset shift to default</button>
      </div>
      <div class="row">
        <label class="name" for="poseDx">shift dx</label>
        <input type="range" id="poseDx" list="poseDxTicks" step="1" value="0">
        <datalist id="poseDxTicks"></datalist>
        <output id="poseDxValue" for="poseDx">0 px</output>
      </div>
      <div class="row">
        <label class="name" for="poseDy">shift dy</label>
        <input type="range" id="poseDy" list="poseDyTicks" step="1" value="0">
        <datalist id="poseDyTicks"></datalist>
        <output id="poseDyValue" for="poseDy">0 px</output>
      </div>
      <div class="warnline" id="sliderWarning">Do not use the slider to make a frame fit. Change the frame until FAR and NEAR residuals are equal; only then may a small slider tweak zero both.</div>
      <div class="hint" id="sliderRangeNote"></div>
      <div class="row">
        <button data-jump="-100">−100</button>
        <button data-jump="-10">−10</button>
        <button data-jump="-1">−1</button>
        <button data-jump="1">+1</button>
        <button data-jump="10">+10</button>
        <button data-jump="100">+100</button>
        <button id="priorBtn">Reset to start position</button>
      </div>
      <div class="row">
        <button id="acceptBtn" class="primary">Accept exact</button>
        <button data-tolerance="1">± 1</button>
        <button data-tolerance="2">± 2</button>
        <button data-tolerance="3">± 3</button>
        <button id="clearBtn">Clear</button>
      </div>
      <div class="blockline" id="labelBlocked"></div>
      <div class="row">
        <fieldset id="reasonSet">
          <legend>No usable match — reason required</legend>
          <label><input type="radio" name="reason" value="undetermined">
            Cannot determine — a counterpart may exist, I cannot tell which frame</label>
          <label><input type="radio" name="reason" value="no_correspondence">
            No corresponding place — occluded, or that stretch was not traversed</label>
        </fieldset>
        <button id="noMatchBtn" class="danger" disabled>Record no usable match</button>
      </div>
      <div class="hint" id="currentAnswer"></div>
    </div>
  </div>
</main>

<div class="status">
  <span class="pill done" id="pillDone"></span>
  <span class="pill undet" id="pillUndetermined"></span>
  <span class="pill nomatch" id="pillNoCorrespondence"></span>
  <span class="pill todo" id="pillTodo"></span>
  <span class="hint">Answers persist in this browser. Download the CSV when every query is labelled.</span>
</div>
<div class="grid" id="queryGrid"></div>

<script>
const DATA = __PAYLOAD__;
const KEY = (name) => `ra-${DATA.labelSet}-${name}-${DATA.camera}`;
const RASTER_W = DATA.filmstrip.width;
const RASTER_H = DATA.filmstrip.height;
const COMPARE_MODES = ["side", "blend", "diff", "blink", "blink_far", "blink_near"];
const FAR_BAND = 0.45;      // top 45% of the frame
const NEAR_BAND = 1 / 3;    // bottom third
// Where each depth guide is allowed to sit, as [top, bottom] fractions of frame
// height. Enforced, not advisory: a label cannot be recorded while a guide is
// outside its band.
const GUIDE_BANDS = DATA.guideBands;
const GUIDE_DEFAULTS = {
  far: { x: 0.35, y: 0.30 },
  near: { x: 0.65, y: 0.80 },
};
// The slider may move this far either side of the exported default and no
// further. A wide slider lets a frame error be absorbed as a shift.
const POSE_RANGE = DATA.pose.sliderRangePx;
const POSE_LIMITS = {
  dx: [DATA.pose.dxPx - POSE_RANGE, DATA.pose.dxPx + POSE_RANGE],
  dy: [DATA.pose.dyPx - POSE_RANGE, DATA.pose.dyPx + POSE_RANGE],
};
const clampPose = (axis, value) => {
  const [low, high] = POSE_LIMITS[axis];
  const number = Math.round(Number(value));
  if (!Number.isFinite(number)) return DATA.pose[axis === "dx" ? "dxPx" : "dyPx"];
  return Math.min(high, Math.max(low, number));
};

const el = (id) => document.getElementById(id);
const pad = (value) => String(value).padStart(6, "0");
const readStore = (name, fallback) => {
  try {
    const raw = localStorage.getItem(KEY(name));
    return raw === null ? fallback : JSON.parse(raw);
  } catch (error) { return fallback; }
};
const writeStore = (name, value) => {
  try { localStorage.setItem(KEY(name), JSON.stringify(value)); } catch (error) {}
};

let answers = readStore("answers", {}) || {};
let lines = readStore("lines", {}) || {};
let poseDx = clampPose("dx", readStore("poseDx", DATA.pose.dxPx));
let poseDy = clampPose("dy", readStore("poseDy", DATA.pose.dyPx));
let poseOn = Boolean(readStore("poseOn", DATA.pose.compensated));
let runBBrightness = Number(readStore("brightness", 100));
let compareMode = "side";
let blinkTimer = null;
let index = 0;
let runBFrame = DATA.queries[0].linearPriorRunBFrame;
let dragging = null;

const runBMin = DATA.runBMin;
const runBMax = DATA.runBMax;

function currentQuery() { return DATA.queries[index]; }

const clamp01 = (value) => Math.min(0.999, Math.max(0.001, Number(value)));

function normaliseGuide(value, fallback) {
  // Tolerates the older shape, where a guide was a bare x fraction.
  if (typeof value === "number") return { x: clamp01(value), y: fallback.y };
  if (value && typeof value === "object" && "x" in value) {
    return { x: clamp01(value.x), y: clamp01("y" in value ? value.y : fallback.y) };
  }
  return { x: fallback.x, y: fallback.y };
}

function currentLines() {
  const stored = lines[currentQuery().queryId] || {};
  return {
    far: normaliseGuide(stored.far, GUIDE_DEFAULTS.far),
    near: normaliseGuide(stored.near, GUIDE_DEFAULTS.near),
  };
}

function setLines(next) {
  lines[currentQuery().queryId] = {
    far: normaliseGuide(next.far, GUIDE_DEFAULTS.far),
    near: normaliseGuide(next.near, GUIDE_DEFAULTS.near),
  };
  writeStore("lines", lines);
  renderLines();
}

function inBand(name, guide) {
  const [top, bottom] = GUIDE_BANDS[name];
  return guide.y >= top && guide.y <= bottom;
}

// Empty when both guides sit in their band; otherwise one sentence per offence,
// used as the tooltip on every label button and shown under the guides.
function guideViolations() {
  const guides = currentLines();
  const said = [];
  for (const name of ["far", "near"]) {
    if (inBand(name, guides[name])) continue;
    const [top, bottom] = GUIDE_BANDS[name];
    const where = name === "far"
      ? `the top ${(bottom * 100).toFixed(0)}% of the frame`
      : `the bottom ${((1 - top) * 100).toFixed(0)}% of the frame`;
    said.push(
      `The ${name.toUpperCase()} guide is at ${(guides[name].y * 100).toFixed(0)}% ` +
      `height; it must be dragged onto a ${name} feature inside the ${name} band, ` +
      `${where}. Two guides at the same depth cannot separate a shift from a frame.`
    );
  }
  return said;
}

function renderBands() {
  for (const box of [el("bandsA"), el("bandsB")]) {
    box.children[0].style.height = `${GUIDE_BANDS.far[1] * 100}%`;
    box.children[1].style.height = `${(1 - GUIDE_BANDS.near[0]) * 100}%`;
  }
}

// The guides live in Run A image coordinates and are drawn at the same place in
// both panes. The compensation moves the Run B raster onto those coordinates;
// adding the shift to the guide as well would cancel it out and the far + near
// test would stop meaning anything.
function renderLines() {
  const guides = currentLines();
  for (const box of [el("linesA"), el("linesB")]) {
    box.innerHTML = "";
    for (const name of ["far", "near"]) {
      const guide = guides[name];
      const line = document.createElement("i");
      line.className = inBand(name, guide) ? name : `${name} out`;
      line.style.left = `${guide.x * 100}%`;
      const mark = document.createElement("s");
      mark.style.top = `${guide.y * 100}%`;
      const tag = document.createElement("b");
      tag.textContent = name;
      tag.style.top = `${guide.y * 100}%`;
      line.appendChild(mark);
      line.appendChild(tag);
      box.appendChild(line);
    }
  }
  const show = (guide) =>
    `x ${(guide.x * 100).toFixed(1)}% · y ${(guide.y * 100).toFixed(1)}%`;
  el("farReadout").textContent = show(guides.far);
  el("nearReadout").textContent = show(guides.near);
  syncLabelButtons();
}

const LABEL_BUTTONS = () => [
  el("acceptBtn"),
  ...document.querySelectorAll("[data-tolerance]"),
  el("noMatchBtn"),
];

function syncLabelButtons() {
  const blocked = guideViolations();
  const reason = blocked.join(" ");
  for (const button of LABEL_BUTTONS()) {
    if (!button) continue;
    if (button.id === "noMatchBtn") {
      button.disabled = blocked.length > 0 || selectedReason() === null;
      button.title = blocked.length
        ? reason
        : (selectedReason() === null ? "Choose a reason first." : "");
      continue;
    }
    button.disabled = blocked.length > 0;
    button.title = reason;
  }
  el("guideWarning").textContent = reason;
  el("labelBlocked").textContent = blocked.length
    ? "No label can be recorded until both guides sit in their bands."
    : "";
}

// The sliders are bounded at build time: default +- sliderRangePx, with the
// default itself marked as a tick. The range is a guard rail, not a preference -
// see the warning line under the sliders.
function initPoseSliders() {
  for (const [axis, id, ticks] of [["dx", "poseDx", "poseDxTicks"],
                                   ["dy", "poseDy", "poseDyTicks"]]) {
    const [low, high] = POSE_LIMITS[axis];
    const slider = el(id);
    slider.min = low;
    slider.max = high;
    const tick = document.createElement("option");
    tick.value = DATA.pose[axis === "dx" ? "dxPx" : "dyPx"];
    tick.label = "default";
    el(ticks).appendChild(tick);
  }
  el("sliderRangeNote").textContent =
    `Range is the measured default ±${POSE_RANGE} px only ` +
    `(dx ${POSE_LIMITS.dx[0]} to ${POSE_LIMITS.dx[1]}, ` +
    `dy ${POSE_LIMITS.dy[0]} to ${POSE_LIMITS.dy[1]}); ` +
    `the tick marks the default ${DATA.pose.dxPx}, ${DATA.pose.dyPx} px. ` +
    `Positive moves Run B right and down.`;
}

function applyPose() {
  const stage = el("stageB");
  const scale = (stage.clientWidth || RASTER_W) / RASTER_W;
  // The stage has the same 4:3 aspect as the raster, so one scale serves both
  // axes: the shift is stored in filmstrip pixels and drawn in display pixels.
  poseDx = clampPose("dx", poseDx);
  poseDy = clampPose("dy", poseDy);
  const dx = poseOn ? poseDx * scale : 0;
  const dy = poseOn ? poseDy * scale : 0;
  el("runBImage").style.transform = `translate(${dx}px, ${dy}px)`;
  el("poseDx").value = poseDx;
  el("poseDy").value = poseDy;
  const offset = (value, base) => {
    const delta = value - base;
    return delta === 0 ? " (default)" : ` (${delta > 0 ? "+" : ""}${delta} vs default)`;
  };
  el("poseDxValue").textContent = `${poseDx} px${offset(poseDx, DATA.pose.dxPx)}`;
  el("poseDyValue").textContent = `${poseDy} px${offset(poseDy, DATA.pose.dyPx)}`;
  el("poseOn").checked = poseOn;
  writeStore("poseDx", poseDx);
  writeStore("poseDy", poseDy);
  writeStore("poseOn", poseOn);
}

function applyBrightness(value, persist = true) {
  runBBrightness = Math.min(250, Math.max(50, Math.round(Number(value) || 100)));
  el("runBImage").style.filter = `brightness(${runBBrightness}%)`;
  el("runBBrightness").value = runBBrightness;
  el("runBBrightnessValue").textContent = `${runBBrightness}%`;
  if (persist) writeStore("brightness", runBBrightness);
}

function bandMask(mode) {
  // Returns [topFraction, bottomFraction] of the frame to hide, or null.
  if (mode === "blink_far") return [0, 1 - FAR_BAND];
  if (mode === "blink_near") return [1 - NEAR_BAND, 0];
  return null;
}

function applyCompareMode() {
  const overlay = el("overlayImage");
  if (blinkTimer) { clearInterval(blinkTimer); blinkTimer = null; }
  overlay.style.mixBlendMode = compareMode === "diff" ? "difference" : "normal";

  const mask = bandMask(compareMode);
  for (const [box, band] of [[el("bandA"), mask], [el("bandB"), mask]]) {
    box.classList.toggle("on", Boolean(band));
    if (band) {
      const [top, bottom] = band;
      box.children[0].style.top = "0";
      box.children[0].style.height = `${top * 100}%`;
      box.children[1].style.bottom = "0";
      box.children[1].style.height = `${bottom * 100}%`;
    }
  }

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
  el("compareMode").value = compareMode;
}

function setRunBFrame(value) {
  runBFrame = Math.min(runBMax, Math.max(runBMin, Math.round(Number(value) || 0)));
  el("runBImage").src = `${DATA.camera}/runB/${pad(runBFrame)}.jpg`;
  el("runBLabel").textContent = runBFrame;
  el("scrub").value = runBFrame;
  el("frameInput").value = runBFrame;
}

function selectedReason() {
  const picked = document.querySelector('input[name="reason"]:checked');
  return picked ? picked.value : null;
}

function syncReasonButton() {
  // A no-match still needs a reason; the guide bands gate it as well.
  syncLabelButtons();
}

function describe(answer) {
  if (!answer) return "Not yet labelled.";
  if (answer.label === "match") {
    return `Recorded: match at B${answer.runBFrame}, acceptable ${answer.runBMin}–${answer.runBMax}.`;
  }
  if (answer.label === "undetermined") {
    return "Recorded: undetermined — a counterpart may exist, the frame could not be identified.";
  }
  return "Recorded: no corresponding place — occluded, or that stretch was not traversed.";
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
  setRunBFrame(answer && answer.label === "match"
    ? answer.runBFrame : query.linearPriorRunBFrame);

  for (const radio of document.querySelectorAll('input[name="reason"]')) {
    radio.checked = Boolean(answer) && answer.label === radio.value;
  }
  syncReasonButton();
  el("reasonSet").classList.remove("flash");
  el("currentAnswer").textContent = describe(answer);

  const total = DATA.queries.length;
  const tally = (label) =>
    DATA.queries.filter((q) => answers[q.queryId] && answers[q.queryId].label === label).length;
  const done = DATA.queries.filter((q) => answers[q.queryId]).length;
  el("doneCount").textContent = `${done} / ${total}`;
  el("pillDone").textContent = `${tally("match")} match`;
  el("pillUndetermined").textContent = `${tally("undetermined")} undetermined`;
  el("pillNoCorrespondence").textContent = `${tally("no_correspondence")} no correspondence`;
  el("pillTodo").textContent = `${total - done} remaining`;

  renderLines();
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
    if (answer && answer.label === "match") cell.classList.add("done");
    if (answer && answer.label === "undetermined") cell.classList.add("undet");
    if (answer && answer.label === "no_correspondence") cell.classList.add("nomatch");
    if (position === index) cell.classList.add("active");
    cell.addEventListener("click", () => { index = position; render(); });
    grid.appendChild(cell);
  });
}

const round4 = (value) => Number(Number(value).toFixed(4));

function conventionSnapshot() {
  const { far, near } = currentLines();
  return {
    poseDx: poseDx,
    poseDy: poseDy,
    poseOn: poseOn,
    compareMode: compareMode,
    farLine: { x: round4(far.x), y: round4(far.y) },
    nearLine: { x: round4(near.x), y: round4(near.y) },
  };
}

// One gate for every answer: a label may not be recorded while a guide is
// outside its band, whatever route the annotator took to get here.
function guardGuides() {
  const blocked = guideViolations();
  if (!blocked.length) return true;
  el("guideWarning").textContent = blocked.join(" ");
  el("labelBlocked").textContent =
    "No label can be recorded until both guides sit in their bands.";
  return false;
}

function accept(tolerance) {
  if (!guardGuides()) return;
  const query = currentQuery();
  answers[query.queryId] = Object.assign({
    label: "match",
    runBFrame: runBFrame,
    runBMin: Math.max(runBMin, runBFrame - tolerance),
    runBMax: Math.min(runBMax, runBFrame + tolerance),
  }, conventionSnapshot());
  writeStore("answers", answers);
  render();
}

function recordNoMatch(reason) {
  if (!guardGuides()) return;
  if (!reason) {
    el("reasonSet").classList.add("flash");
    return;
  }
  answers[currentQuery().queryId] = Object.assign({ label: reason }, conventionSnapshot());
  writeStore("answers", answers);
  render();
}

function move(delta) {
  index = Math.min(DATA.queries.length - 1, Math.max(0, index + delta));
  render();
}

function pointerFraction(event) {
  const rect = el("stageA").getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
  };
}

function beginDrag(event) {
  const stage = el("stageA");
  const at = pointerFraction(event);
  const { far, near } = currentLines();
  const distance = (guide) => (at.x - guide.x) ** 2 + (at.y - guide.y) ** 2;
  dragging = distance(far) <= distance(near) ? "far" : "near";
  stage.setPointerCapture(event.pointerId);
  moveDrag(event);
}

function moveDrag(event) {
  if (!dragging) return;
  const next = currentLines();
  next[dragging] = pointerFraction(event);
  setLines(next);
  event.preventDefault();
}

el("stageA").addEventListener("pointerdown", beginDrag);
el("stageA").addEventListener("pointermove", moveDrag);
el("stageA").addEventListener("pointerup", () => { dragging = null; });
el("stageA").addEventListener("pointercancel", () => { dragging = null; });
el("resetLinesBtn").addEventListener("click", () => setLines(GUIDE_DEFAULTS));

el("compareMode").addEventListener("change", (event) => {
  compareMode = event.target.value; applyCompareMode();
});
el("poseOn").addEventListener("change", (event) => { poseOn = event.target.checked; applyPose(); });
el("poseDx").addEventListener("input", (event) => { poseDx = clampPose("dx", event.target.value); applyPose(); });
el("poseDy").addEventListener("input", (event) => { poseDy = clampPose("dy", event.target.value); applyPose(); });
el("poseResetBtn").addEventListener("click", () => {
  poseDx = DATA.pose.dxPx; poseDy = DATA.pose.dyPx; applyPose();
});
el("prevBtn").addEventListener("click", () => move(-1));
el("nextBtn").addEventListener("click", () => move(1));
el("acceptBtn").addEventListener("click", () => accept(0));
document.querySelectorAll("[data-tolerance]").forEach((button) => {
  button.addEventListener("click", () => accept(Number(button.dataset.tolerance)));
});
el("noMatchBtn").addEventListener("click", () => recordNoMatch(selectedReason()));
document.querySelectorAll('input[name="reason"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    el("reasonSet").classList.remove("flash");
    syncReasonButton();
  });
});
el("clearBtn").addEventListener("click", () => {
  delete answers[currentQuery().queryId];
  writeStore("answers", answers);
  render();
});
el("priorBtn").addEventListener("click", () => setRunBFrame(currentQuery().linearPriorRunBFrame));
el("scrub").addEventListener("input", (event) => setRunBFrame(event.target.value));
el("frameInput").addEventListener("change", (event) => setRunBFrame(event.target.value));
el("runBBrightness").addEventListener("input", (event) => applyBrightness(event.target.value));
el("brightnessResetBtn").addEventListener("click", () => applyBrightness(100));
document.querySelectorAll("[data-jump]").forEach((button) => {
  button.addEventListener("click", () => setRunBFrame(runBFrame + Number(button.dataset.jump)));
});
window.addEventListener("resize", applyPose);

document.addEventListener("keydown", (event) => {
  const tag = event.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  const step = event.shiftKey ? 10 : 1;
  const key = event.key.toLowerCase();
  if (event.key === "ArrowLeft") { setRunBFrame(runBFrame - step); event.preventDefault(); }
  else if (event.key === "ArrowRight") { setRunBFrame(runBFrame + step); event.preventDefault(); }
  else if (event.key === "Enter") { accept(0); event.preventDefault(); }
  else if (["1", "2", "3"].includes(event.key)) { accept(Number(event.key)); event.preventDefault(); }
  else if (key === "u") { recordNoMatch("undetermined"); event.preventDefault(); }
  else if (key === "x") { recordNoMatch("no_correspondence"); event.preventDefault(); }
  else if (key === "n") { recordNoMatch(selectedReason()); event.preventDefault(); }
  else if (event.key === "[") { move(-1); event.preventDefault(); }
  else if (event.key === "]") { move(1); event.preventDefault(); }
  else if (key === "g") {
    compareMode = COMPARE_MODES[(COMPARE_MODES.indexOf(compareMode) + 1) % COMPARE_MODES.length];
    applyCompareMode(); event.preventDefault();
  }
  else if (key === "p") { poseOn = !poseOn; applyPose(); event.preventDefault(); }
});

const CSV_HEADER = "__CSV_HEADER__";

el("saveBtn").addEventListener("click", () => {
  const rows = DATA.queries.map((query) => {
    const answer = answers[query.queryId];
    const stored = lines[query.queryId] || {};
    // The pose slider is recorded per label, as set when the label was made:
    // the convention a label was judged under is part of the label.
    const guide = (name) => {
      const source = (answer && answer[`${name}Line`]) || stored[name];
      const point = source === undefined ? null : normaliseGuide(source, GUIDE_DEFAULTS[name]);
      return point === null ? "," : `${round4(point.x)},${round4(point.y)}`;
    };
    const convention = [
      answer ? answer.poseDx : poseDx,
      answer ? answer.poseDy : poseDy,
      (answer ? answer.poseOn : poseOn) ? "true" : "false",
      guide("far"),
      guide("near"),
      answer ? answer.compareMode : compareMode,
    ].join(",");
    const label = answer ? answer.label : "unlabelled";
    const interval = answer && answer.label === "match"
      ? `${answer.runBFrame},${answer.runBMin},${answer.runBMax}` : ",,";
    return `${query.queryId},${DATA.camera},${query.runAFrame},${label},${interval},` +
      `${convention},manual_viewer_blind`;
  });
  const blob = new Blob([[CSV_HEADER].concat(rows).join("\n") + "\n"], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${DATA.labelSet}_${DATA.camera}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
});

if (DATA.camera === "cam5") {
  el("cam5Note").textContent =
    " CAM5 pointed differently in Run B, so the whole scene sits shifted — a mailbox " +
    "that was centred in Run A will not be centred in Run B at the correct frame. " +
    "The default compensation already undoes the measured difference (it moves Run B " +
    "right and down). Judge alignment by the far + near test after compensation, " +
    "never by where an object sits in the frame.";
}

initPoseSliders();
renderBands();
applyBrightness(runBBrightness, false);
applyPose();
applyCompareMode();
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
