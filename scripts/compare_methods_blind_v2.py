#!/usr/bin/env python3
"""Score both frozen methods head to head on the same sealed blind labels.

The submitted mapping is the frozen v1 method, whose original held-out estimate
rested on 6 and 4 accepted cases - a Wilson half-width around 0.3, wide enough
that it could not distinguish a good method from a mediocre one. The blind v2
label set is 40 queries per camera and was built long after v1 was frozen, so
scoring v1 against it is a legitimate blind evaluation and not a re-test: no
v1 threshold was ever chosen with any knowledge of these frames.

Running both methods against one label set is also the only way to compare them
honestly. The two were previously quoted on different label sets of different
sizes, which is not a comparison at all.

Neither method is re-tuned here. Both mappings are read from disk exactly as
they were frozen.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


from evaluate_task2_blind_v2 import (
    _distribution,
    distance_to_interval,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]
BLIND_DIR = ROOT / "outputs" / "task2_evaluation" / "blind_v2"
OUTPUT_DIR = BLIND_DIR

METHODS = {
    "v1_frozen_submitted": ROOT / "outputs" / "task2_final" / "{camera}_frame_mapping.csv",
    "v2_unified": ROOT / "outputs" / "task2_unified" / "{camera}" / "frame_mapping_unified.csv",
}


def load_mapping(pattern: Path, camera: str) -> dict[int, dict[str, str]]:
    path = Path(str(pattern).format(camera=camera))
    with path.open(newline="") as handle:
        return {int(row["runA_frame"]): row for row in csv.DictReader(handle)}


def score(camera: str, labels: list[dict[str, str]], mapping: dict[int, dict[str, str]]) -> dict:
    accepted = correct = no_match_false = 0
    match_total = no_match_total = 0
    preferred: list[int] = []
    intervals: list[int] = []
    abstention_statuses: dict[str, int] = {}

    for label in labels:
        frame = int(label["runA_frame"])
        row = mapping.get(frame)
        if row is None:
            raise ValueError(f"{camera}: Run A frame {frame} absent from mapping")
        predicted = row["runB_frame"].strip()
        is_match = label["label"] == "match"
        if is_match:
            match_total += 1
        else:
            no_match_total += 1

        if not predicted:
            status = row.get("status", "") or "unmapped"
            abstention_statuses[status] = abstention_statuses.get(status, 0) + 1
            continue

        accepted += 1
        prediction = int(predicted)
        if is_match:
            low, high = int(label["runB_min"]), int(label["runB_max"])
            if low <= prediction <= high:
                correct += 1
            preferred.append(abs(prediction - int(label["runB_frame"])))
            intervals.append(distance_to_interval(prediction, low, high))
        else:
            no_match_false += 1

    precision = correct / accepted if accepted else None
    return {
        "camera_id": camera,
        "evaluable_labels": len(labels),
        "ground_truth_matches": match_total,
        "ground_truth_no_matches": no_match_total,
        "accepted": accepted,
        "abstained": len(labels) - accepted,
        "correct_accepted": correct,
        "strict_accepted_precision": round(precision, 4) if precision is not None else None,
        "strict_accepted_precision_wilson_95": wilson_interval(correct, accepted),
        "selective_risk": round(1 - precision, 4) if precision is not None else None,
        "coverage_all_labels": round(accepted / len(labels), 4),
        "no_match_false_accepts": no_match_false,
        "no_match_correctly_abstained": no_match_total - no_match_false,
        "preferred_frame_error": _distribution(preferred),
        "interval_distance": _distribution(intervals),
        "abstention_statuses": dict(
            sorted(abstention_statuses.items(), key=lambda item: -item[1])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    target = OUTPUT_DIR / "method_comparison_blind_v2.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    seal = json.loads((BLIND_DIR / "label_seal.json").read_text())
    label_file = ROOT / seal["label_file"]
    digest = hashlib.sha256(label_file.read_bytes()).hexdigest()
    if digest != seal["label_file_sha256"]:
        raise SystemExit("Sealed labels have changed; refusing to score")
    with label_file.open(newline="") as handle:
        labels = list(csv.DictReader(handle))

    results: dict[str, dict[str, dict]] = {}
    for method, pattern in METHODS.items():
        results[method] = {}
        for camera in ("cam0", "cam5"):
            camera_labels = [row for row in labels if row["camera_id"] == camera]
            results[method][camera] = score(
                camera, camera_labels, load_mapping(pattern, camera)
            )

    payload = {
        "schema_version": 1,
        "compared_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_set": "blind_v2",
        "label_seal_sha256": digest,
        "labels_per_camera": 40,
        "legitimacy": (
            "Both mappings were frozen before this label set existed and neither "
            "was tuned with any knowledge of these frames. Reading the two "
            "against one label set is the only honest comparison; the previously "
            "quoted numbers came from different sets of different sizes."
        ),
        "methods": results,
    }
    target.write_text(json.dumps(payload, indent=2))

    header = (
        f"{'method':<22} {'cam':<5} {'acc':>6} {'corr':>5} {'prec':>6} "
        f"{'Wilson 95%':>16} {'cov':>6} {'noM abst':>9} {'med err':>8}"
    )
    print(header)
    print("-" * len(header))
    for method, cameras in results.items():
        for camera, result in cameras.items():
            interval = result["strict_accepted_precision_wilson_95"]
            rendered = f"{interval[0]:.2f}-{interval[1]:.2f}" if interval else "n/a"
            error = result["preferred_frame_error"]
            median = f"{error['median']:.1f}" if error["n"] else "n/a"
            print(
                f"{method:<22} {camera:<5} "
                f"{result['accepted']:>3}/{result['evaluable_labels']:<2} "
                f"{result['correct_accepted']:>5} "
                f"{str(result['strict_accepted_precision']):>6} "
                f"{rendered:>16} "
                f"{result['coverage_all_labels']:>6} "
                f"{result['no_match_correctly_abstained']:>4}/"
                f"{result['ground_truth_no_matches']:<4} {median:>8}"
            )
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
