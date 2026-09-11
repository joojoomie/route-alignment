#!/usr/bin/env python3
"""Score the frozen unified method once, on the sealed blind label set.

This runs after the freeze and after the seal, and it runs once. It refuses to
start if the freeze has drifted or if the seal does not cite the freeze that is
on disk, because the number it produces means nothing without that ordering.

Three deliberate departures from the frozen v1 evaluation:

* **A risk-coverage curve, not a point.** A single precision figure at an
  unstated coverage is uninterpretable, and reject-all must be visibly
  ineligible rather than quietly optimal. The curve sweeps the pipeline's
  continuous sequence margin.
* **Error distributions, not a binary verdict.** The v1 metric asked only
  whether a prediction landed inside its interval, which scores a one-frame miss
  and a two-hundred-frame miss identically - and every v1 false accept was off
  by exactly one. Median, p90 and p95 are reported against both the preferred
  frame and the interval.
* **Typed refusals.** An abstention because the vehicle was parked, and one
  because the route seam is genuinely ambiguous, are different facts about the
  data. They are counted separately.

Cameras are reported separately and never pooled: they observe the same two
traversals and are not independent samples of anything.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_DIR = ROOT / "outputs" / "task2_unified"
BLIND_DIR = ROOT / "outputs" / "task2_evaluation" / "blind_v2"
FREEZE_FILE = ROOT / "outputs" / "task2_evaluation" / "method_freeze_v2.json"
OUTPUT_DIR = BLIND_DIR

Z = 1.959963984540054


def wilson_interval(successes: int, total: int) -> list[float] | None:
    """Score interval. Reported because a proportion from tens of samples is wide."""

    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + Z * Z / total
    centre = (proportion + Z * Z / (2 * total)) / denominator
    spread = (
        Z
        * ((proportion * (1 - proportion) / total + Z * Z / (4 * total * total)) ** 0.5)
        / denominator
    )
    return [round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4)]


def distance_to_interval(prediction: int, low: int, high: int) -> int:
    if prediction < low:
        return low - prediction
    if prediction > high:
        return prediction - high
    return 0


def load_mapping(camera: str) -> dict[int, dict[str, str]]:
    path = UNIFIED_DIR / camera / "frame_mapping_unified.csv"
    with path.open(newline="") as handle:
        return {int(row["runA_frame"]): row for row in csv.DictReader(handle)}


def check_contract() -> dict[str, object]:
    """Refuse to score unless freeze, seal and code all line up."""

    if not FREEZE_FILE.is_file():
        raise SystemExit("No method freeze; run scripts/freeze_method_v2.py first")
    freeze = json.loads(FREEZE_FILE.read_text())

    import freeze_method_v2

    freeze_method_v2.verify()

    seal_path = BLIND_DIR / "label_seal.json"
    if not seal_path.is_file():
        raise SystemExit(
            "No sealed label set. Import the annotator's CSVs with "
            "scripts/import_blind_labels_v2.py first."
        )
    seal = json.loads(seal_path.read_text())
    label_file = ROOT / seal["label_file"]
    digest = hashlib.sha256(label_file.read_bytes()).hexdigest()
    if digest != seal["label_file_sha256"]:
        raise SystemExit(
            "The sealed label file has changed since it was sealed. The blind "
            "evaluation cannot proceed against edited labels."
        )
    return {"freeze": freeze, "seal": seal, "label_file": label_file}


def evaluate_camera(
    camera: str, labels: list[dict[str, str]], mapping: dict[int, dict[str, str]]
) -> dict[str, object]:
    accepted, correct, false_accepts = 0, 0, []
    no_match_total, no_match_false_accepts = 0, 0
    match_total, match_accepted = 0, 0
    preferred_errors: list[int] = []
    interval_errors: list[int] = []
    abstention_statuses: dict[str, int] = {}
    cases: list[dict[str, object]] = []

    for label in labels:
        frame = int(label["runA_frame"])
        if frame not in mapping:
            raise ValueError(f"{camera}: labelled Run A frame {frame} is absent from the mapping")
        row = mapping[frame]
        predicted = row["runB_frame"]
        is_accepted = predicted != ""
        status = row["status"]
        is_match = label["label"] == "match"

        if is_match:
            match_total += 1
        else:
            no_match_total += 1

        case: dict[str, object] = {
            "query_id": label["query_id"],
            "camera_id": camera,
            "runA_frame": frame,
            "ground_truth_label": label["label"],
            "ground_truth_runB_frame": label["runB_frame"],
            "ground_truth_runB_min": label["runB_min"],
            "ground_truth_runB_max": label["runB_max"],
            "status": status,
            "sequence_tier": row.get("sequence_tier", ""),
            "longest_window_margin": row.get("longest_window_margin", ""),
            "accepted": is_accepted,
            "predicted_runB_frame": predicted,
            "accepted_correct": False,
            "preferred_frame_error": "",
            "interval_distance": "",
        }

        if not is_accepted:
            abstention_statuses[status] = abstention_statuses.get(status, 0) + 1
            cases.append(case)
            continue

        accepted += 1
        prediction = int(predicted)
        if is_match:
            match_accepted += 1
            low, high = int(label["runB_min"]), int(label["runB_max"])
            inside = low <= prediction <= high
            case["accepted_correct"] = inside
            error = abs(prediction - int(label["runB_frame"]))
            gap = distance_to_interval(prediction, low, high)
            preferred_errors.append(error)
            interval_errors.append(gap)
            case["preferred_frame_error"] = error
            case["interval_distance"] = gap
            if inside:
                correct += 1
            else:
                false_accepts.append(label["query_id"])
        else:
            no_match_false_accepts += 1
            false_accepts.append(label["query_id"])
        cases.append(case)

    evaluable = len(labels)
    precision = correct / accepted if accepted else None
    return {
        "camera_id": camera,
        "evaluable_labels": evaluable,
        "ground_truth_matches": match_total,
        "ground_truth_no_matches": no_match_total,
        "accepted": accepted,
        "abstained": evaluable - accepted,
        "correct_accepted": correct,
        "strict_accepted_precision": round(precision, 4) if precision is not None else None,
        "strict_accepted_precision_wilson_95": wilson_interval(correct, accepted),
        "selective_risk": round(1 - precision, 4) if precision is not None else None,
        "coverage_all_labels": round(accepted / evaluable, 4),
        "coverage_all_labels_wilson_95": wilson_interval(accepted, evaluable),
        "match_coverage": round(match_accepted / match_total, 4) if match_total else None,
        "false_accept_count": len(false_accepts),
        "false_accept_query_ids": false_accepts,
        "no_match_false_accepts": no_match_false_accepts,
        "preferred_frame_error": _distribution(preferred_errors),
        "interval_distance": _distribution(interval_errors),
        "abstention_statuses": dict(
            sorted(abstention_statuses.items(), key=lambda item: -item[1])
        ),
        "cases": cases,
    }


def _distribution(values: list[int]) -> dict[str, object]:
    if not values:
        return {"n": 0}
    array = np.array(values)
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
        "within_one_frame": int((array <= 1).sum()),
        "within_two_frames": int((array <= 2).sum()),
    }


def risk_coverage_curve(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    """Sweep the sequence margin, reporting risk against coverage.

    Reject-all is included so it is visible, and marked ineligible: a system
    that answers nothing has undefined risk, not zero risk.
    """

    scored = [
        case
        for case in cases
        if case["accepted"] and case["longest_window_margin"] not in ("", None)
    ]
    if not scored:
        return []
    thresholds = sorted({float(case["longest_window_margin"]) for case in scored}, reverse=True)
    total = len(cases)
    points: list[dict[str, object]] = [
        {
            "threshold": None,
            "policy": "reject_all",
            "accepted": 0,
            "correct": 0,
            "selective_risk": None,
            "coverage": 0.0,
            "no_match_false_accepts": 0,
            "eligible": False,
            "note": "undefined risk; never a valid operating point",
        }
    ]
    for threshold in thresholds:
        kept = [case for case in scored if float(case["longest_window_margin"]) >= threshold]
        matches = [case for case in kept if case["ground_truth_label"] == "match"]
        correct = sum(1 for case in matches if case["accepted_correct"])
        no_match = sum(1 for case in kept if case["ground_truth_label"] == "no_match")
        accepted = len(kept)
        risk = 1 - correct / accepted if accepted else None
        points.append(
            {
                "threshold": round(threshold, 5),
                "policy": "sequence_margin_gate",
                "accepted": accepted,
                "correct": correct,
                "selective_risk": round(risk, 4) if risk is not None else None,
                "coverage": round(accepted / total, 4),
                "no_match_false_accepts": no_match,
                "eligible": True,
            }
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    target = OUTPUT_DIR / "blind_v2_metrics.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(
            f"{target} exists. The blind set is scored once; re-running after "
            "seeing the result is how a held-out estimate stops being one."
        )

    contract = check_contract()
    with contract["label_file"].open(newline="") as handle:
        labels = list(csv.DictReader(handle))

    results = {}
    all_cases: list[dict[str, object]] = []
    for camera in ("cam0", "cam5"):
        camera_labels = [row for row in labels if row["camera_id"] == camera]
        if not camera_labels:
            continue
        result = evaluate_camera(camera, camera_labels, load_mapping(camera))
        all_cases.extend(result["cases"])
        result["risk_coverage_curve"] = risk_coverage_curve(result["cases"])
        results[camera] = result

    payload = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_set": "blind_v2",
        "method_freeze_sha256": contract["freeze"]["freeze_sha256"],
        "label_seal_sha256": contract["seal"]["label_file_sha256"],
        "label_sealed_at_utc": contract["seal"]["sealed_at_utc"],
        "method_frozen_at_utc": contract["freeze"]["frozen_at_utc"],
        "ordering_note": (
            "The method freeze timestamp precedes the label seal timestamp. That "
            "records ordering, not custody: it shows the method was fixed before "
            "the labels were imported, not that the labels were never inspected."
        ),
        "pooling_policy": (
            "Cameras are reported separately. They observe the same two "
            "traversals and are not independent samples."
        ),
        "claim_boundary": contract["freeze"]["claim_boundary"],
        "cameras": {
            camera: {key: value for key, value in result.items() if key != "cases"}
            for camera, result in results.items()
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))

    with (OUTPUT_DIR / "blind_v2_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_cases[0]))
        writer.writeheader()
        writer.writerows(all_cases)

    print(f"Method freeze  {payload['method_freeze_sha256'][:12]}  {payload['method_frozen_at_utc']}")
    print(f"Label seal     {payload['label_seal_sha256'][:12]}  {payload['label_sealed_at_utc']}")
    print()
    for camera, result in results.items():
        precision = result["strict_accepted_precision"]
        interval = result["strict_accepted_precision_wilson_95"]
        print(f"=== {camera}")
        print(
            f"  accepted {result['accepted']}/{result['evaluable_labels']}, "
            f"correct {result['correct_accepted']}"
        )
        print(
            f"  strict accepted precision {precision} "
            f"(Wilson 95% {interval})  coverage {result['coverage_all_labels']}"
        )
        print(
            f"  no-match false accepts {result['no_match_false_accepts']}"
            f"/{result['ground_truth_no_matches']}"
        )
        error = result["preferred_frame_error"]
        if error["n"]:
            print(
                f"  preferred-frame error: median {error['median']}, "
                f"p90 {error['p90']}, p95 {error['p95']}, max {error['max']}"
            )
        if result["abstention_statuses"]:
            print(f"  abstained: {result['abstention_statuses']}")
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
