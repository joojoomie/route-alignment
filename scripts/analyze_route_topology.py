#!/usr/bin/env python3
"""Measure the route's topology from the descriptor cache, so §1.3 has an artifact.

Two claims in the report were previously typed rather than produced:

* the route is a **loop** --- the tail of each run looks like its head;
* CAM5 **aliases** internally far more than CAM0 --- many pairs of frames far
  apart along the route look alike.

Both are readings over the same frozen DINOv2-S descriptors the matcher uses
(`build_task2_unified.descriptor_cache_path`), so they describe the evidence the
method actually sees rather than a separate feature. Nothing here changes a
mapping, a confidence tier or an evaluation artifact: it is a read-only
measurement whose only output is `outputs/task1/route_topology.json`.

Definitions, fixed here and stated in the output so the numbers are reproducible
rather than merely quoted:

* **Route frames** start at the detected route onset for that run
  (`outputs/task2_keyframes/route_phase_boundaries.csv`); the stationary and
  transition prefixes are excluded, because a parked camera trivially matches
  itself.
* **Tail-to-head revisit** is the largest cosine between the *last 100* route
  frames and the *first 300* route frames of one run, with the frame pair that
  achieved it. A loop should produce a high value at a plausible pair; an open
  route should not.
* **Non-local aliasing** counts ordered pairs $(i,j)$, $i<j$, of Run A route
  frames with $|i-j| > 150$ and cosine $> 0.85$. The separation floor is what
  makes it *non-local*: adjacent frames are similar because they are adjacent,
  which says nothing about place recognition.

The thresholds are conventions, not tuned values, and the counts scale with
them; what the report uses is the CAM5/CAM0 **ratio**, which is far more stable
than either count. The script reports both, and the ratio's sensitivity to the
threshold, so a reader can see that.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import build_task2_unified


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "outputs" / "task1" / "route_topology.json"

CAMERAS = ("cam0", "cam5")
RUNS = ("runA", "runB")

TAIL_FRAMES = 100
HEAD_FRAMES = 300
NONLOCAL_SEPARATION = 150
ALIAS_COSINE = 0.85
ALIAS_SENSITIVITY = (0.80, 0.85, 0.90)


def load_descriptors(run: str, camera: str) -> tuple[np.ndarray, Path]:
    """The frozen descriptor cache for one stream, L2-normalised for cosines."""

    path = build_task2_unified.descriptor_cache_path(run, camera)
    if not path.is_file():
        raise SystemExit(
            f"No descriptor cache at {path.relative_to(ROOT)}. This reading needs "
            "the cached DINOv2-S descriptors, which are excluded from the "
            "submission bundle; rebuild them with "
            f"`build_task2_unified.py --camera {camera}` from the raw HEVC, or "
            "read the shipped outputs/task1/route_topology.json instead."
        )
    descriptors = np.load(path).astype(np.float32)
    norms = np.clip(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8, None)
    return descriptors / norms, path


def revisit(descriptors: np.ndarray, start: int) -> dict[str, object]:
    """Best cosine between the run's last 100 route frames and its first 300."""

    total = descriptors.shape[0]
    head_stop = min(start + HEAD_FRAMES, total)
    tail_start = max(total - TAIL_FRAMES, head_stop)
    head = descriptors[start:head_stop]
    tail = descriptors[tail_start:total]
    if head.size == 0 or tail.size == 0:
        return {"available": False}
    similarity = tail @ head.T
    flat = int(np.argmax(similarity))
    row, column = divmod(flat, similarity.shape[1])
    return {
        "available": True,
        "tail_window": [tail_start, total - 1],
        "head_window": [start, head_stop - 1],
        "best_cosine": round(float(similarity[row, column]), 4),
        "tail_frame": tail_start + row,
        "head_frame": start + column,
        "median_cosine": round(float(np.median(similarity)), 4),
    }


def aliasing(descriptors: np.ndarray, start: int) -> dict[str, object]:
    """Non-local look-alike pairs among one run's route frames.

    Computed in row blocks so the full similarity matrix is never materialised;
    only the upper triangle beyond the separation floor is counted, so each
    unordered pair is counted once.
    """

    route = descriptors[start:]
    count = route.shape[0]
    tallies = {threshold: 0 for threshold in ALIAS_SENSITIVITY}
    eligible = 0
    block = 256
    for begin in range(0, count, block):
        end = min(begin + block, count)
        similarity = route[begin:end] @ route.T
        rows = np.arange(begin, end)[:, None]
        columns = np.arange(count)[None, :]
        mask = (columns - rows) > NONLOCAL_SEPARATION
        eligible += int(mask.sum())
        values = similarity[mask]
        for threshold in ALIAS_SENSITIVITY:
            tallies[threshold] += int((values > threshold).sum())
    return {
        "route_frames": count,
        "separation_frames": NONLOCAL_SEPARATION,
        "eligible_pairs": eligible,
        "cosine_threshold": ALIAS_COSINE,
        "non_local_pairs": tallies[ALIAS_COSINE],
        "non_local_fraction": round(tallies[ALIAS_COSINE] / eligible, 6) if eligible else None,
        "by_threshold": {str(key): value for key, value in tallies.items()},
    }


def build() -> dict[str, object]:
    bounds = build_task2_unified.route_bounds()
    cameras: dict[str, object] = {}
    sources: dict[str, str] = {}
    for camera in CAMERAS:
        per_run: dict[str, object] = {}
        alias = None
        for run in RUNS:
            descriptors, path = load_descriptors(run, camera)
            sources[f"{run}_{camera}"] = str(path.relative_to(ROOT))
            start = bounds[f"{run}_start"]
            per_run[run] = {
                "frames": int(descriptors.shape[0]),
                "route_start": start,
                "revisit": revisit(descriptors, start),
            }
            if run == "runA":
                alias = aliasing(descriptors, start)
        cameras[camera] = {"runs": per_run, "runA_aliasing": alias}

    zero = cameras["cam0"]["runA_aliasing"]
    five = cameras["cam5"]["runA_aliasing"]
    ratio = {
        "definition": "CAM5 non-local look-alike pairs divided by CAM0's, Run A",
        "cam0_pairs": zero["non_local_pairs"],
        "cam5_pairs": five["non_local_pairs"],
        "ratio_cam5_over_cam0": (
            round(five["non_local_pairs"] / zero["non_local_pairs"], 2)
            if zero["non_local_pairs"]
            else None
        ),
        "by_threshold": {
            key: (
                round(five["by_threshold"][key] / zero["by_threshold"][key], 2)
                if zero["by_threshold"][key]
                else None
            )
            for key in five["by_threshold"]
        },
        "note": (
            "The counts move with the cosine threshold; the ratio is what the "
            "report cites and it is reported at three thresholds so its "
            "stability is visible rather than assumed."
        ),
    }

    return {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "what_this_measures": (
            "Loop structure and internal aliasing of the route, read from the "
            "frozen DINOv2-S descriptor cache the matcher itself uses."
        ),
        "what_it_does_not_do": (
            "It does not prove metric loop closure. Without GPS or odometry, a "
            "high tail-to-head cosine supports an approximate revisit "
            "hypothesis only, and the alignment still treats departure and "
            "arrival as distinct occurrences."
        ),
        "definitions": {
            "tail_frames": TAIL_FRAMES,
            "head_frames": HEAD_FRAMES,
            "non_local_separation_frames": NONLOCAL_SEPARATION,
            "alias_cosine_threshold": ALIAS_COSINE,
            "route_start_source": "outputs/task2_keyframes/route_phase_boundaries.csv",
        },
        "route_bounds": bounds,
        "descriptor_caches": sources,
        "cameras": cameras,
        "aliasing_ratio": ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing route_topology.json (the report quotes it)",
    )
    arguments = parser.parse_args()
    if OUTPUT_FILE.exists() and not arguments.force:
        raise FileExistsError(
            f"{OUTPUT_FILE.relative_to(ROOT)} already exists; pass --force to "
            "replace the reading the report quotes."
        )
    payload = build()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2))
    for camera, block in payload["cameras"].items():
        for run, entry in block["runs"].items():
            best = entry["revisit"]
            print(
                f"{camera} {run}: best tail-to-head cosine {best['best_cosine']:.3f} "
                f"at {run[-1]}{best['tail_frame']} -> {best['head_frame']}"
            )
        alias = block["runA_aliasing"]
        print(
            f"{camera} runA: {alias['non_local_pairs']:,} non-local pairs above "
            f"cosine {alias['cosine_threshold']} of {alias['eligible_pairs']:,} eligible"
        )
    print(f"ratio CAM5/CAM0 = {payload['aliasing_ratio']['ratio_cam5_over_cam0']}")
    print(f"\nWrote {OUTPUT_FILE.relative_to(ROOT)}")
    print(f"sha256 {hashlib.sha256(OUTPUT_FILE.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
