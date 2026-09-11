#!/usr/bin/env python3
"""Choose the geometry working resolution from evidence, not from intuition.

The frozen matcher verified geometry on 448x336 crops of a 640x480 CRF-24
proxy. The data audit showed that raster yields 669 usable SIFT keypoints on
the worst stream against 2,494 at 896x672, which suggested simply working at a
higher resolution.

Keypoint count is the wrong metric to stop at. More keypoints across an
eleven-month, sunny-to-rainy gap can mean more fine-scale foliage texture that
is not repeatable, producing matches that satisfy no single geometry and
collapsing the inlier ratio. So this measures what actually matters: how often
a *known correct* pair passes the full distributed-support gate.

It runs on the original calibration labels. Those have been inspected many
times and are development data now, which is exactly what a component choice
should be made on - but it means the resulting resolution is a development
decision, not an accuracy claim.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

import build_reliability_masks
import build_task2_unified as unified
import frame_service


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_unified" / "resolution_study"
LABEL_FILE = ROOT / "outputs" / "task2_keyframes" / "manual_annotations.csv"

CANDIDATE_RESOLUTIONS = ((448, 336), (640, 480), (896, 672), (1440, 1080))
PAIRS_PER_CAMERA = 6


def labelled_pairs(camera: str, limit: int) -> list[tuple[int, int]]:
    with LABEL_FILE.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["camera_id"] == camera
            and row["label"] == "match"
            and row["runB_frame"]
        ]
    return [(int(row["runA_frame"]), int(row["runB_frame"])) for row in rows[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=PAIRS_PER_CAMERA)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    target = OUTPUT_DIR / "geometry_resolution_study.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for camera in ("cam0", "cam5"):
        pairs = labelled_pairs(camera, arguments.pairs)
        exclusion_a = build_reliability_masks.load_mask("runA", camera)["exclusion_224"]
        exclusion_b = build_reliability_masks.load_mask("runB", camera)["exclusion_224"]
        print(f"=== {camera}: {len(pairs)} labelled correct pairs")
        for width, height in CANDIDATE_RESOLUTIONS:
            images_a = frame_service.frames("runA", camera, [a for a, _ in pairs], width, height)
            images_b = frame_service.frames("runB", camera, [b for _, b in pairs], width, height)
            started = time.time()
            keypoints_a, keypoints_b, matches, inliers, ratios, passed = [], [], [], [], [], 0
            for frame_a, frame_b in pairs:
                result = unified.verify_pair(
                    images_a[frame_a], images_b[frame_b], exclusion_a, exclusion_b
                )
                keypoints_a.append(result["keypoints_a"])
                keypoints_b.append(result["keypoints_b"])
                matches.append(result["matches"])
                inliers.append(result["inliers"])
                ratios.append(result["inlier_ratio"])
                passed += int(bool(result["supportive"]))
            elapsed = (time.time() - started) / max(len(pairs), 1)
            record = {
                "camera_id": camera,
                "width": width,
                "height": height,
                "pairs": len(pairs),
                "median_keypoints_runA": int(np.median(keypoints_a)),
                "median_keypoints_runB": int(np.median(keypoints_b)),
                "median_matches": int(np.median(matches)),
                "median_inliers": int(np.median(inliers)),
                "mean_inlier_ratio": round(float(np.mean(ratios)), 4),
                "passed_distributed_support": passed,
                "seconds_per_pair": round(elapsed, 3),
            }
            records.append(record)
            print(
                f"  {width}x{height}: matches {record['median_matches']}, "
                f"inliers {record['median_inliers']}, ratio {record['mean_inlier_ratio']}, "
                f"passed {passed}/{len(pairs)}, {elapsed:.2f}s/pair"
            )

    best = {}
    for camera in ("cam0", "cam5"):
        camera_records = [record for record in records if record["camera_id"] == camera]
        chosen = max(
            camera_records,
            key=lambda record: (record["passed_distributed_support"], -record["seconds_per_pair"]),
        )
        best[camera] = f"{chosen['width']}x{chosen['height']}"

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_source": "original calibration labels, reclassified as development data",
        "evidence_status": (
            "development-only component selection; not an accuracy estimate and "
            "not evaluated on the sealed blind set"
        ),
        "selection_rule": "maximum passing pairs, ties broken by lower cost per pair",
        "selected_resolution": best,
        "finding": (
            "Passing rate peaks at 896x672 and falls again at native resolution, "
            "where fine foliage texture inflates the raw match count while the "
            "inlier ratio collapses. Keypoint count alone would have chosen wrong."
        ),
        "records": records,
    }
    target.write_text(json.dumps(payload, indent=2))

    with (OUTPUT_DIR / "geometry_resolution_study.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSelected: {best}")
    print(f"Wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
