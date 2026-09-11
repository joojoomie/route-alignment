#!/usr/bin/env python3
"""Dense agreement signals that need no labels at all.

A blind label set of forty queries per camera gives a Wilson half-width around
0.15. That is a real improvement on twelve, but it is still a thin basis for
saying where a mapping fails. Two signals in this dataset are dense over all
2,674 frames and cost nothing to compute.

**Cross-camera agreement.** cam0 faces left and cam5 faces right, so the two
alignments are estimated from disjoint scenes. Their failure modes are close to
independent: cam5 carries roughly five times cam0's density of look-alike
places, on entirely different content. Where the two independently agree, that
is meaningful corroboration; where they diverge by tens of frames, at least one
is wrong.

Two honest limits, both of which belong in any report of this number:

* It measures *consistency*, not correctness. Both cameras can fail together
  where the cause is the vehicle's own trajectory - a shared stop, a shared
  turn - rather than the scenery.
* The two cameras are not proved synchronised. Run B's cam0 and cam5 decode to
  2,622 and 2,615 frames while sharing one timestamp log, and their motion
  signals correlate best at a lag of -20 frames. So the absolute difference
  between the two predictions is not an error; only its *variation* along the
  route is. This module therefore reports the spread about the running median,
  not the raw difference.

**Cycle consistency** is the natural companion - align A to B and B to A, then
measure the round trip - but it needs a second full alignment per camera and is
reported separately when that has been run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_DIR = ROOT / "outputs" / "task2_unified"
OUTPUT_DIR = ROOT / "outputs" / "task2_consistency"

# Window used to separate a slowly varying camera offset from real disagreement.
LAG_WINDOW = 201
LARGE_DISAGREEMENT_FRAMES = 10


def load_mapping(camera: str) -> dict[int, int]:
    path = UNIFIED_DIR / camera / "frame_mapping_unified.csv"
    with path.open(newline="") as handle:
        return {
            int(row["runA_frame"]): int(row["runB_frame"])
            for row in csv.DictReader(handle)
            if row["runB_frame"]
        }


def load_status(camera: str) -> dict[int, str]:
    path = UNIFIED_DIR / camera / "frame_mapping_unified.csv"
    with path.open(newline="") as handle:
        return {int(row["runA_frame"]): row["status"] for row in csv.DictReader(handle)}


def running_median(values: np.ndarray, window: int) -> np.ndarray:
    """Median filter with edge padding, used as the slowly varying camera lag."""

    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array(
        [np.median(padded[index : index + window]) for index in range(len(values))]
    )


def cross_camera_agreement() -> dict[str, object]:
    cam0 = load_mapping("cam0")
    cam5 = load_mapping("cam5")
    shared = sorted(set(cam0) & set(cam5))
    if len(shared) < LAG_WINDOW:
        raise ValueError(
            f"Only {len(shared)} frames are mapped by both cameras; too few to "
            "separate a camera lag from real disagreement"
        )

    frames = np.array(shared)
    difference = np.array([cam0[frame] - cam5[frame] for frame in shared], dtype=float)
    baseline = running_median(difference, LAG_WINDOW)
    residual = difference - baseline

    absolute = np.abs(residual)
    large = absolute > LARGE_DISAGREEMENT_FRAMES
    status0 = load_status("cam0")
    status5 = load_status("cam5")

    runs: list[dict[str, int]] = []
    index = 0
    while index < len(frames):
        if not large[index]:
            index += 1
            continue
        start = index
        while index < len(frames) and large[index]:
            index += 1
        runs.append(
            {
                "runA_start": int(frames[start]),
                "runA_end": int(frames[index - 1]),
                "frames": int(index - start),
                "peak_disagreement": int(absolute[start:index].max()),
            }
        )
    runs.sort(key=lambda entry: -entry["peak_disagreement"])

    return {
        "frames_mapped_by_both_cameras": len(shared),
        "apparent_camera_offset_frames": {
            "median": float(np.median(difference)),
            "p5": float(np.percentile(difference, 5)),
            "p95": float(np.percentile(difference, 95)),
            "note": (
                "Not an error. The two cameras are not proved synchronised, so a "
                "constant component of this difference is unattributable."
            ),
        },
        "disagreement_about_the_running_offset": {
            "median_absolute": float(np.median(absolute)),
            "p90_absolute": float(np.percentile(absolute, 90)),
            "p95_absolute": float(np.percentile(absolute, 95)),
            "max_absolute": float(absolute.max()),
            "frames_over_threshold": int(large.sum()),
            "fraction_over_threshold": float(large.mean()),
            "threshold_frames": LARGE_DISAGREEMENT_FRAMES,
        },
        "largest_disagreement_regions": runs[:10],
        "status_of_disagreeing_frames": {
            "cam0": _status_histogram(frames[large], status0),
            "cam5": _status_histogram(frames[large], status5),
        },
        "interpretation": (
            "Agreement is corroboration, not proof: the cameras can fail together "
            "where the vehicle's own trajectory is the cause. Disagreement is the "
            "stronger direction - it proves at least one camera is wrong there."
        ),
    }


def _status_histogram(frames: np.ndarray, status: dict[int, str]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for frame in frames:
        key = status.get(int(frame), "missing")
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: -item[1]))


def coverage_summary() -> dict[str, object]:
    summary: dict[str, object] = {}
    for camera in ("cam0", "cam5"):
        status = load_status(camera)
        histogram: dict[str, int] = {}
        for value in status.values():
            histogram[value] = histogram.get(value, 0) + 1
        mapped = len(load_mapping(camera))
        summary[camera] = {
            "rows": len(status),
            "mapped": mapped,
            "coverage": round(mapped / max(len(status), 1), 6),
            "status_counts": dict(sorted(histogram.items(), key=lambda item: -item[1])),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / "consistency_summary.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_class": "label-free dense consistency; never an accuracy claim",
        "coverage": coverage_summary(),
        "cross_camera_agreement": cross_camera_agreement(),
    }
    target.write_text(json.dumps(payload, indent=2))

    agreement = payload["cross_camera_agreement"]
    spread = agreement["disagreement_about_the_running_offset"]
    print(f"Frames mapped by both cameras: {agreement['frames_mapped_by_both_cameras']}")
    print(
        f"Apparent camera offset (unattributable constant): "
        f"median {agreement['apparent_camera_offset_frames']['median']:.0f} frames"
    )
    print(
        f"Disagreement about that offset: median {spread['median_absolute']:.1f}, "
        f"p90 {spread['p90_absolute']:.1f}, p95 {spread['p95_absolute']:.1f}, "
        f"max {spread['max_absolute']:.0f} frames"
    )
    print(
        f"Frames disagreeing by more than {LARGE_DISAGREEMENT_FRAMES}: "
        f"{spread['frames_over_threshold']} ({spread['fraction_over_threshold']:.2%})"
    )
    for region in agreement["largest_disagreement_regions"][:5]:
        print(
            f"   A{region['runA_start']}-{region['runA_end']} "
            f"({region['frames']} frames, peak {region['peak_disagreement']})"
        )
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
