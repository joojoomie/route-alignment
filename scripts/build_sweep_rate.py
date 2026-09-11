#!/usr/bin/env python3
"""Per-frame sweep rate: how many pixels the scene moves between consecutive frames.

Measured with the same RootSIFT detector, raster and mask that measure the
scene displacement dx at matched pairs, because on a side-facing fisheye the
apparent speed depends on depth and different detectors pick different depths.
Whole-frame and tiled phase correlation were tried first; both failed
validation against this quantity (Spearman 0.05-0.48), which is why this
script exists and why it is slow (about 0.3 s per frame).

Output: outputs/task2_motion_bayes/sweep_{run}_{camera}.npy, index i giving the
median |dx| between frame i and i+1 (last entry repeated). Frames with too few
matches are linearly interpolated from their neighbours and counted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import build_motion_events
import build_reliability_masks
import build_task2_unified as unified
import frame_service


ROOT = Path(__file__).resolve().parents[1]
RUNS = ("runA", "runB")
CAMERAS = ("cam0", "cam5")
MIN_MATCHES = 15
SIFT_RATIO = 0.8


def sweep_rate(run: str, camera: str) -> tuple[np.ndarray, int]:
    from skimage.feature import match_descriptors

    exclusion = build_reliability_masks.load_mask(run, camera)["exclusion_224"]
    source = frame_service.source_path(run, camera)
    rates: list[float] = []
    previous = None
    interpolated = 0
    for ordinal, image in frame_service.iter_frames(source, *unified.GEOMETRY_SIZE):
        points, descriptors = unified.rootsift_features(image, exclusion)
        if previous is not None:
            value = np.nan
            if len(descriptors) >= MIN_MATCHES and len(previous[1]) >= MIN_MATCHES:
                matches = match_descriptors(
                    previous[1], descriptors, metric="euclidean", cross_check=True, max_ratio=SIFT_RATIO
                )
                if len(matches) >= MIN_MATCHES:
                    value = abs(float(np.median(points[matches[:, 1], 0] - previous[0][matches[:, 0], 0])))
            rates.append(value)
        previous = (points, descriptors)
        if ordinal % 400 == 0:
            print(f"   {run}/{camera}: {ordinal}", flush=True)
    array = np.array(rates, dtype=np.float64)
    missing = np.isnan(array)
    interpolated = int(missing.sum())
    if missing.any() and not missing.all():
        index = np.arange(array.size)
        array[missing] = np.interp(index[missing], index[~missing], array[~missing])
    return np.concatenate([array, array[-1:]]), interpolated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run", choices=RUNS)
    parser.add_argument("--camera", choices=CAMERAS)
    arguments = parser.parse_args()

    build_motion_events.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for run in RUNS:
        for camera in CAMERAS:
            if (arguments.run and run != arguments.run) or (arguments.camera and camera != arguments.camera):
                continue
            target = build_motion_events.sweep_path(run, camera)
            if target.is_file() and not arguments.force:
                print(f"{run}/{camera}: sweep rate cached")
                continue
            print(f"{run}/{camera}: RootSIFT between consecutive frames ...", flush=True)
            rates, interpolated = sweep_rate(run, camera)
            np.save(target, rates)
            summary[f"{run}_{camera}"] = {
                "frames": int(rates.size),
                "interpolated": interpolated,
                "median_px_per_frame": round(float(np.median(rates)), 2),
                "p10": round(float(np.percentile(rates, 10)), 2),
                "p90": round(float(np.percentile(rates, 90)), 2),
                "fraction_below_2px": round(float((rates < 2.0).mean()), 4),
            }
            print(f"   median {summary[f'{run}_{camera}']['median_px_per_frame']} px/frame, "
                  f"{interpolated} interpolated")
    record = build_motion_events.OUTPUT_DIR / "sweep_rate_summary.json"
    existing = json.loads(record.read_text()) if record.is_file() else {"streams": {}}
    existing["built_at_utc"] = datetime.now(timezone.utc).isoformat()
    existing["method"] = "RootSIFT median |dx| between consecutive frames, geometry raster, reliability mask"
    existing["streams"].update(summary)
    record.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
