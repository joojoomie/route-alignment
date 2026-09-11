#!/usr/bin/env python3
"""Does the unpinned suspension channel change the correspondence it feeds?

run_static_removal_study.py answers a question about the *sensor*: whether the
event channel is measuring the vehicle or the lens. This answers the separate
question about the *consumer*: whether the posterior model's output moves when
it is given the better sensor.

The comparison reuses the existing label-free instruments rather than inventing
new ones, so the numbers sit on the same scale as every other variant in this
repository:

* cross-camera agreement (evaluate_variants.cross_camera) - two cameras on one
  vehicle must land on Run B frames a fixed lag apart, and the residual after
  removing the running median lag is a dense error signal that needs no labels;
* endpoint closure (evaluate_loop_consistency.endpoint_closure) - two Run A
  views of one place must not be sent to two different Run B places;
* coverage of the route.

The spent blind labels are also read and reported, clearly marked, because the
number exists and hiding it would be worse than labelling it. It is a
diagnostic: that set was read during development, so it can no longer support a
claim of accuracy, and it does not decide anything here.

Expect small differences. The motion channel is one multiplier of 2.5 on a few
dozen frame pairs inside a model whose emissions come from dense descriptors;
a large swing would be evidence of instability, not of a better cue.

The baseline here is the withdrawn "none" construction and the variant is
whatever `--variant` names, which is why the file keeps that name after
"mean-divide" became the default: the comparison being recorded is still
old-construction against new, not default against ablation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import build_motion_events as motion
import build_task2_bayes as bayes
import evaluate_loop_consistency as loop
import evaluate_variants as variants


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_motion_bayes"
BAYES_DIR = ROOT / "outputs" / "task2_bayes"
CAMERAS = variants.CAMERAS


def pattern_for(mode: str) -> Path:
    """Where the posterior mapping built on one event construction lives.

    The naming follows build_task2_bayes.mapping_suffix, so the current default
    construction owns the plain frame_mapping_bayes.csv and every other mode -
    including the withdrawn "none" - is suffixed.
    """

    return BAYES_DIR / "{camera}" / f"frame_mapping_bayes{bayes.mapping_suffix(True, True, False, mode)}.csv"


def disagreement(baseline: Path, variant: Path, camera: str) -> dict[str, object]:
    """How far the two mappings differ frame by frame, on frames both accept."""

    left = variants.load(baseline, camera)
    right = variants.load(variant, camera)
    shared = sorted(set(left) & set(right))
    both = [
        (int(left[f]["runB_frame"]), int(right[f]["runB_frame"]))
        for f in shared
        if left[f]["runB_frame"] and right[f]["runB_frame"]
    ]
    identical = sum(1 for a, b in both if a == b)
    within_two = sum(1 for a, b in both if abs(a - b) <= 2)
    only_left = sum(1 for f in shared if left[f]["runB_frame"] and not right[f]["runB_frame"])
    only_right = sum(1 for f in shared if right[f]["runB_frame"] and not left[f]["runB_frame"])
    return {
        "frames_accepted_by_both": len(both),
        "identical_prediction": identical,
        "identical_fraction": round(identical / len(both), 4) if both else None,
        "within_2_frames": round(within_two / len(both), 4) if both else None,
        "max_disagreement_frames": max((abs(a - b) for a, b in both), default=0),
        "accepted_only_by_baseline": only_left,
        "accepted_only_by_variant": only_right,
    }


def evaluate(pattern: Path, labels: list[dict[str, str]], descriptors) -> dict[str, object]:
    entry: dict[str, object] = {"cross_camera": variants.cross_camera(pattern), "cameras": {}}
    for camera in CAMERAS:
        rows = variants.load(pattern, camera)
        mapped = {f: int(r["runB_frame"]) for f, r in rows.items() if r["runB_frame"]}
        closure = loop.endpoint_closure(
            camera, mapped, descriptors[("runA", camera)], descriptors[("runB", camera)]
        )
        entry["cameras"][camera] = {
            "coverage": round(len(mapped) / len(rows), 4),
            "endpoint_closure": f"{closure['consistent']}/{closure['pairs_with_both_ends_mapped']}",
            "endpoint_consistent": closure["consistent"],
            "endpoint_checked": closure["pairs_with_both_ends_mapped"],
            "spent_labels_diagnostic": variants.spent_labels(pattern, camera, labels),
        }
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=motion.STATIC_REMOVAL_MODES, default="mean-divide"
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    target = OUTPUT_DIR / f"motion_variant_mapping_comparison_{arguments.variant}.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    baseline = pattern_for("none")
    variant = pattern_for(arguments.variant)
    for pattern in (baseline, variant):
        for camera in CAMERAS:
            path = Path(str(pattern).format(camera=camera))
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path.relative_to(ROOT)} is missing; run "
                    f"scripts/build_task2_bayes.py --camera {camera} "
                    f"--motion-variant {arguments.variant}"
                )

    with variants.LABELS.open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    descriptors = {
        (run, camera): loop.descriptors(run, camera)
        for run in ("runA", "runB")
        for camera in CAMERAS
    }

    results = {}
    for name, pattern in (("motion none (withdrawn)", baseline), (f"motion {arguments.variant}", variant)):
        print(f"[{name}]", flush=True)
        results[name] = evaluate(pattern, labels, descriptors)

    changes = {camera: disagreement(baseline, variant, camera) for camera in CAMERAS}

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": arguments.variant,
        "evidence_status": (
            "Cross-camera agreement, endpoint closure and coverage are label-free. "
            "The spent-label column reuses the blind set after it was read and is a "
            "diagnostic only; neither mapping here has a held-out number."
        ),
        "mappings": results,
        "prediction_changes": changes,
    }
    motion.atomic_write(target, lambda path: path.write_text(json.dumps(payload, indent=2)))

    print("\nLabel-free comparison")
    print(f"  {'mapping':<26}{'x-cam median':<14}{'x-cam p95':<12}{'over 10':<10}"
          f"{'coverage c0/c5':<18}{'closure c0':<13}{'closure c5'}")
    for name, entry in results.items():
        cross = entry["cross_camera"]
        cameras = entry["cameras"]
        print(
            f"  {name:<26}{cross['median']:<14}{cross['p95']:<12}{cross['over_10']:<10}"
            f"{cameras['cam0']['coverage']:.4f}/{cameras['cam5']['coverage']:<10.4f}"
            f"{cameras['cam0']['endpoint_closure']:<13}{cameras['cam5']['endpoint_closure']}"
        )

    print("\nSpent blind labels (diagnostic only, not an accuracy claim)")
    for name, entry in results.items():
        for camera in CAMERAS:
            spent = entry["cameras"][camera]["spent_labels_diagnostic"]
            print(f"  {name:<26}{camera}  precision {spent['precision']} "
                  f"within-2 {spent['within_2']} ({spent['accepted']}/{spent['of']})")

    print("\nHow much the prediction actually moved")
    for camera, change in changes.items():
        print(
            f"  {camera}: {change['identical_fraction']:.4f} identical, "
            f"{change['within_2_frames']:.4f} within 2 frames, "
            f"max disagreement {change['max_disagreement_frames']} frames, "
            f"accepted only by baseline {change['accepted_only_by_baseline']}, "
            f"only by variant {change['accepted_only_by_variant']}"
        )
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
