#!/usr/bin/env python3
"""Freeze the unified method, and verify a frozen method has not drifted.

The evaluation contract this project runs on is simple: select every component
and threshold on development evidence, write down exactly what was selected,
and only then look at labels that have never been seen. The value of the
resulting number depends entirely on that ordering being real.

So the freeze is written *before* the blind labels are imported, not after. The
seal produced by `import_blind_labels_v2.py` then cites this file's hash, which
makes the order checkable by anyone reading the artifacts: the method was fully
determined at a recorded time, and the labels were sealed later against it.

`--verify` re-reads the live module constants and compares them field by field
against the frozen record. The frozen v1 check omitted the mask entirely, which
is precisely how a mask swap could have slipped past it unnoticed; here the mask
version, descriptor version, colour identity and both working rasters are all
part of the comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import build_reliability_masks
import build_task2_unified as unified
import frame_service


ROOT = Path(__file__).resolve().parents[1]
FREEZE_FILE = ROOT / "outputs" / "task2_evaluation" / "method_freeze_v2.json"


def frozen_parameters() -> dict[str, object]:
    """Every knob that can change a prediction, read from the live modules."""

    return {
        "pipeline_version": unified.PIPELINE_VERSION,
        "descriptor_version": unified.DESCRIPTOR_VERSION,
        "mask_version": build_reliability_masks.mask_version(),
        "colour_id": frame_service.COLOUR_ID,
        "descriptor_raster": list(frame_service.DESCRIPTOR_SIZE),
        "geometry_raster": list(unified.GEOMETRY_SIZE),
        "coarse_stride": unified.COARSE_STRIDE,
        "dtw_steps": [list(step) for step in unified.DTW_STEPS],
        "slope_penalty_weight": unified.SLOPE_PENALTY_WEIGHT,
        "skip_penalty_weight": unified.SKIP_PENALTY_WEIGHT,
        "min_slope": unified.MIN_SLOPE,
        "max_slope": unified.MAX_SLOPE,
        "refine_radius": unified.REFINE_RADIUS,
        "structure_grid": list(unified.STRUCTURE_GRID),
        "window_radii": list(unified.WINDOW_RADII),
        "alternative_separation": unified.ALTERNATIVE_SEPARATION,
        "strong_sequence_scales": unified.STRONG_SEQUENCE_SCALES,
        "supported_sequence_scales": unified.SUPPORTED_SEQUENCE_SCALES,
        "max_bridged_gap": unified.MAX_BRIDGED_GAP,
        "geometry_sample_stride": unified.GEOMETRY_SAMPLE_STRIDE,
        "geometry_gates": {
            "min_matches": unified.GEOMETRY_MIN_MATCHES,
            "min_inliers": unified.GEOMETRY_MIN_INLIERS,
            "min_inlier_ratio": unified.GEOMETRY_MIN_INLIER_RATIO,
            "grid": list(unified.GEOMETRY_GRID),
            "min_cells": unified.GEOMETRY_MIN_CELLS,
            "min_rows": unified.GEOMETRY_MIN_ROWS,
            "min_columns": unified.GEOMETRY_MIN_COLUMNS,
            "min_horizontal_spread": unified.GEOMETRY_MIN_HORIZONTAL_SPREAD,
            "min_vertical_spread": unified.GEOMETRY_MIN_VERTICAL_SPREAD,
        },
        "sift": {
            "ratio_threshold": unified.SIFT_RATIO_THRESHOLD,
            "min_matches_for_ransac": unified.SIFT_MIN_MATCHES_FOR_RANSAC,
            "bottom_exclusion_fraction": unified.BOTTOM_EXCLUSION_FRACTION,
        },
        "run_a_route_tail_exclusive": unified.RUN_A_ROUTE_TAIL_EXCLUSIVE,
    }


def development_evidence() -> dict[str, object]:
    """What each non-obvious choice was made on. All of it pre-label."""

    study = ROOT / "outputs" / "task2_unified" / "resolution_study" / "geometry_resolution_study.json"
    resolution = json.loads(study.read_text()) if study.is_file() else {}
    consistency = ROOT / "outputs" / "task2_consistency" / "consistency_summary.json"
    agreement = json.loads(consistency.read_text()) if consistency.is_file() else {}
    return {
        "geometry_raster": {
            "decided_by": "controlled resolution study on original calibration labels",
            "artifact": "outputs/task2_unified/resolution_study/geometry_resolution_study.json",
            "result": resolution.get("selected_resolution"),
            "finding": resolution.get("finding"),
        },
        "reliability_mask": {
            "decided_by": (
                "measured temporal statistics; no learned model and no label was "
                "consulted to build it"
            ),
            "artifact": "outputs/task2_masks/reliability_mask_summary.json",
        },
        "global_search": {
            "decided_by": (
                "measured interior aliases sit 300-1300 frames from their "
                "look-alike, beyond any usable prior band"
            ),
            "artifact": "the data audit; verified by a synthetic decoy unit test",
        },
        "warp_constraints": {
            "decided_by": (
                "measured Run A to Run B offset drifts across a 124-frame span "
                "but at most about 0.5 frames per Run A frame"
            ),
        },
        "label_free_consistency_at_freeze_time": agreement.get(
            "cross_camera_agreement", {}
        ).get("disagreement_about_the_running_offset"),
    }


def verify() -> None:
    if not FREEZE_FILE.is_file():
        raise SystemExit(f"No freeze at {FREEZE_FILE}; nothing to verify")
    record = json.loads(FREEZE_FILE.read_text())
    live = frozen_parameters()
    frozen = record["frozen_parameters"]
    differences = [
        (key, frozen.get(key), live.get(key))
        for key in sorted(set(frozen) | set(live))
        if frozen.get(key) != live.get(key)
    ]
    if differences:
        rendered = "\n".join(
            f"  {key}: frozen {frozen!r} but code now says {current!r}"
            for key, frozen, current in differences
        )
        raise SystemExit(
            "Code-level parameters changed after the method freeze:\n"
            f"{rendered}\n"
            "Either revert the change or collect a new blind label set. The "
            "existing blind result cannot describe this code."
        )
    print(f"Method freeze {record['freeze_sha256'][:12]} verified against live code")
    print(f"  frozen at {record['frozen_at_utc']}")
    print(f"  {len(live)} parameter groups match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check the code has not drifted.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing freeze.")
    arguments = parser.parse_args()

    if arguments.verify:
        verify()
        return

    if FREEZE_FILE.is_file() and not arguments.force:
        raise FileExistsError(
            f"{FREEZE_FILE} already exists. Re-freezing after labels have been "
            "read invalidates the evaluation. Pass --force only if no blind "
            "evaluation has been run against it."
        )

    FREEZE_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "status": "frozen_before_blind_v2_labels_were_imported",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "cameras": ["cam0", "cam5"],
        "supervision": "automatic; no manual anchor participates in prediction",
        "training_performed": (
            "none. DINOv2 is frozen pretrained; the mask, path, sequence gate "
            "and geometry verifier are deterministic algorithms. Component "
            "selection on development data is still model selection."
        ),
        "post_freeze_tuning_allowed": False,
        "blind_label_set": "blind_v2",
        "blind_labels_read_at_freeze_time": False,
        "frozen_parameters": frozen_parameters(),
        "development_evidence": development_evidence(),
        "claim_boundary": (
            "Two traversals of one route, one weather pair, one season pair. "
            "Any resulting number is a within-route estimate and says nothing "
            "about a new route, a new city, or a third condition."
        ),
    }
    payload = json.dumps(record, indent=2, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    record["freeze_sha256"] = digest
    FREEZE_FILE.write_text(json.dumps(record, indent=2))

    print(f"Froze the unified method at {record['frozen_at_utc']}")
    print(f"  {FREEZE_FILE.relative_to(ROOT)}")
    print(f"  sha256 {digest}")
    print()
    print("The blind labels have not been read. Import them next; the seal will")
    print("cite this hash, which records that the method predates the labels.")


if __name__ == "__main__":
    main()
