#!/usr/bin/env python3
"""Rescore the blind set after a label-semantics correction, showing both readings.

The annotation page offered two answers: a match, or "no usable match". That was
a protocol defect. It merged two different facts:

  (a) run B has no counterpart for this place, and
  (b) the annotator could not determine the counterpart.

The annotator has since confirmed that all 11 such labels meant (b). The claim is
structurally checkable and does not depend on any result: the query set was
sampled entirely inside the stretch both runs traversed (Run A 180-2616 mapping
into Run B 114-2621), so (a) was impossible by construction. There was never a
true no-correspondence case in this label set.

That has two consequences, and only one of them flatters the methods.

**Undetermined labels cannot score precision.** A prediction on a query whose
truth is unknown is neither correct nor incorrect, so those rows leave the
denominator. This raises both methods' numbers, and it raises the denser one
more, because the denser one answered all of them.

**The set cannot test abstention at all.** With zero true negatives, "no-match
false accepts" measures nothing. The earlier reading of this evaluation - that
the dense method's abstention mechanism had collapsed - is not supported by
these labels. Neither is the opposite. The question is simply unanswered, and
that is a limitation of how the queries were sampled.

Both readings are printed. The correction was made after the results were seen,
which is stated rather than hidden; it is applied identically to both methods.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


from evaluate_task2_blind_v2 import _distribution, distance_to_interval, wilson_interval


ROOT = Path(__file__).resolve().parents[1]
BLIND_DIR = ROOT / "outputs" / "task2_evaluation" / "blind_v2"

METHODS = {
    "v1_frozen_submitted": ROOT / "outputs" / "task2_final" / "{camera}_frame_mapping.csv",
    "v2_unified": ROOT / "outputs" / "task2_unified" / "{camera}" / "frame_mapping_unified.csv",
}


def load_mapping(pattern: Path, camera: str) -> dict[int, dict[str, str]]:
    path = Path(str(pattern).format(camera=camera))
    with path.open(newline="") as handle:
        return {int(row["runA_frame"]): row for row in csv.DictReader(handle)}


def score(labels: list[dict[str, str]], mapping: dict[int, dict[str, str]]) -> dict:
    """Score one method on one camera under both label readings."""

    determinable = [row for row in labels if row["label"] == "match"]
    undetermined = [row for row in labels if row["label"] != "match"]

    accepted_det = correct = 0
    accepted_undet = 0
    preferred: list[int] = []
    intervals: list[int] = []

    for row in determinable:
        predicted = mapping[int(row["runA_frame"])]["runB_frame"].strip()
        if not predicted:
            continue
        accepted_det += 1
        prediction = int(predicted)
        low, high = int(row["runB_min"]), int(row["runB_max"])
        if low <= prediction <= high:
            correct += 1
        preferred.append(abs(prediction - int(row["runB_frame"])))
        intervals.append(distance_to_interval(prediction, low, high))

    for row in undetermined:
        if mapping[int(row["runA_frame"])]["runB_frame"].strip():
            accepted_undet += 1

    # As-labelled reading: undetermined rows were counted as true negatives, so
    # every acceptance on one of them was scored as a false accept.
    as_labelled_accepted = accepted_det + accepted_undet
    as_labelled_precision = correct / as_labelled_accepted if as_labelled_accepted else None

    corrected_precision = correct / accepted_det if accepted_det else None

    return {
        "labels_total": len(labels),
        "labels_determinable": len(determinable),
        "labels_undetermined": len(undetermined),
        "as_labelled": {
            "accepted": as_labelled_accepted,
            "correct": correct,
            "precision": round(as_labelled_precision, 4) if as_labelled_precision else None,
            "precision_wilson_95": wilson_interval(correct, as_labelled_accepted),
            "coverage": round(as_labelled_accepted / len(labels), 4),
        },
        "corrected": {
            "accepted": accepted_det,
            "correct": correct,
            "precision": round(corrected_precision, 4) if corrected_precision else None,
            "precision_wilson_95": wilson_interval(correct, accepted_det),
            "coverage_of_determinable": round(accepted_det / len(determinable), 4),
            "answered_undetermined": accepted_undet,
        },
        "preferred_frame_error": _distribution(preferred),
        "interval_distance": _distribution(intervals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    target = BLIND_DIR / "blind_v2_undetermined_rescore.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    with (BLIND_DIR / "blind_v2_labels.csv").open(newline="") as handle:
        labels = list(csv.DictReader(handle))

    results: dict[str, dict[str, dict]] = {}
    for method, pattern in METHODS.items():
        results[method] = {}
        for camera in ("cam0", "cam5"):
            camera_labels = [row for row in labels if row["camera_id"] == camera]
            results[method][camera] = score(camera_labels, load_mapping(pattern, camera))

    payload = {
        "schema_version": 1,
        "rescored_at_utc": datetime.now(timezone.utc).isoformat(),
        "correction": (
            "All 11 'no usable match' labels meant the annotator could not "
            "determine the counterpart, not that no counterpart exists. The "
            "query set was sampled entirely inside the mutually traversed "
            "route, so a genuine absence was impossible by construction."
        ),
        "correction_timing": (
            "Made after the blind results were seen. Applied identically to both "
            "methods. It raises both, and raises the denser method more."
        ),
        "abstention_is_untestable_here": (
            "With zero true no-correspondence cases, this label set cannot "
            "evaluate abstention in either direction. The earlier reading, that "
            "the dense method's abstention had collapsed, is withdrawn as "
            "unsupported. Testing abstention needs queries sampled where "
            "correspondence genuinely fails."
        ),
        "methods": results,
    }
    target.write_text(json.dumps(payload, indent=2))

    header = f"{'method':<22} {'cam':<5} {'as-labelled':>22} {'corrected':>22} {'med err':>8}"
    print(header)
    print("-" * len(header))
    for method, cameras in results.items():
        for camera, result in cameras.items():
            a, c = result["as_labelled"], result["corrected"]
            error = result["preferred_frame_error"]
            median = f"{error['median']:.1f}" if error["n"] else "n/a"
            left = f"{a['correct']}/{a['accepted']} = {a['precision']:.3f}"
            right = f"{c['correct']}/{c['accepted']} = {c['precision']:.3f}"
            print(f"{method:<22} {camera:<5} {left:>22} {right:>22} {median:>8}")
    print()
    for method, cameras in results.items():
        for camera, result in cameras.items():
            c = result["corrected"]
            print(
                f"  {method} {camera}: corrected precision {c['precision']:.3f} "
                f"(Wilson {c['precision_wilson_95']}) at "
                f"{c['coverage_of_determinable']:.1%} coverage of determinable labels"
            )
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
