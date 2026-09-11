#!/usr/bin/env python3
"""Diagnostics for the posterior model, none of them a held-out accuracy claim.

The blind label set is spent: it selected nothing for this model, but it has been
read, so no number computed against it is a blind estimate. Everything here is a
development diagnostic and is labelled as one. What makes that tolerable is that
the interesting question about a posterior is not its accuracy but its
**calibration** - whether a stated probability of 0.8 corresponds to being right
about 80% of the time - and calibration is a property this model has and its
predecessors structurally could not, since their confidence was categorical by
its own manifest.

Three things are reported:

* a reliability curve, binning frames by stated posterior and measuring how often
  the answer is actually within tolerance;
* an ablation of the suspension-shake channel, which shows what that cue is worth
  once its measured likelihood ratio is taken seriously rather than assumed; and
* cross-camera agreement, which needs no labels at all and is therefore the only
  number here comparable to the frozen evaluation of the earlier methods.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BAYES_DIR = ROOT / "outputs" / "task2_bayes"
BLIND_LABELS = ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "blind_v2_labels.csv"
OUTPUT_DIR = BAYES_DIR

TOLERANCE_FRAMES = 2
RELIABILITY_BINS = (0.0, 0.4, 0.55, 0.7, 0.8, 0.9, 0.95, 1.01)
LAG_WINDOW = 201


def load_mapping(camera: str, motion: bool = True) -> dict[int, dict[str, str]]:
    name = "frame_mapping_bayes.csv" if motion else "frame_mapping_bayes_no_motion.csv"
    with (BAYES_DIR / camera / name).open(newline="") as handle:
        return {int(row["runA_frame"]): row for row in csv.DictReader(handle)}


def determinable_labels(camera: str) -> list[dict[str, str]]:
    """Labels the annotator could actually resolve. Development data now."""

    with BLIND_LABELS.open(newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["camera_id"] == camera and row["label"] == "match"
        ]


def score(camera: str, motion: bool) -> dict[str, object]:
    mapping = load_mapping(camera, motion)
    labels = determinable_labels(camera)
    accepted = strict = within = 0
    errors: list[int] = []
    scored: list[tuple[float, bool]] = []

    for label in labels:
        row = mapping[int(label["runA_frame"])]
        if not row["runB_frame"]:
            continue
        accepted += 1
        prediction = int(row["runB_frame"])
        low, high = int(label["runB_min"]), int(label["runB_max"])
        error = abs(prediction - int(label["runB_frame"]))
        errors.append(error)
        inside = low <= prediction <= high
        close = error <= TOLERANCE_FRAMES
        strict += inside
        within += close
        scored.append((float(row["posterior_probability"]), close))

    return {
        "camera_id": camera,
        "motion_channel": motion,
        "determinable_labels": len(labels),
        "accepted": accepted,
        "strict_correct": strict,
        "within_tolerance": within,
        "strict_precision": round(strict / accepted, 4) if accepted else None,
        "within_tolerance_precision": round(within / accepted, 4) if accepted else None,
        "median_error": float(np.median(errors)) if errors else None,
        "p90_error": float(np.percentile(errors, 90)) if errors else None,
        "reliability": reliability_curve(scored),
        "expected_calibration_error": expected_calibration_error(scored),
    }


def reliability_curve(scored: list[tuple[float, bool]]) -> list[dict[str, object]]:
    """Does a stated posterior of p correspond to being right p of the time?"""

    curve = []
    for low, high in zip(RELIABILITY_BINS, RELIABILITY_BINS[1:]):
        bucket = [(p, ok) for p, ok in scored if low <= p < high]
        if not bucket:
            continue
        curve.append(
            {
                "bin": f"{low:.2f}-{min(high, 1.0):.2f}",
                "n": len(bucket),
                "mean_stated_posterior": round(float(np.mean([p for p, _ in bucket])), 4),
                "observed_within_tolerance": round(
                    float(np.mean([ok for _, ok in bucket])), 4
                ),
            }
        )
    return curve


def expected_calibration_error(scored: list[tuple[float, bool]]) -> float | None:
    if not scored:
        return None
    total = 0.0
    for low, high in zip(RELIABILITY_BINS, RELIABILITY_BINS[1:]):
        bucket = [(p, ok) for p, ok in scored if low <= p < high]
        if not bucket:
            continue
        stated = float(np.mean([p for p, _ in bucket]))
        observed = float(np.mean([ok for _, ok in bucket]))
        total += len(bucket) / len(scored) * abs(stated - observed)
    return round(total, 4)


def running_median(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(values))])


def cross_camera_agreement() -> dict[str, object]:
    """Label-free, dense, and comparable with the earlier methods' figure."""

    cam0 = {
        frame: int(row["runB_frame"])
        for frame, row in load_mapping("cam0").items()
        if row["runB_frame"]
    }
    cam5 = {
        frame: int(row["runB_frame"])
        for frame, row in load_mapping("cam5").items()
        if row["runB_frame"]
    }
    shared = sorted(set(cam0) & set(cam5))
    difference = np.array([cam0[f] - cam5[f] for f in shared], dtype=float)
    residual = np.abs(difference - running_median(difference, LAG_WINDOW))
    return {
        "frames_mapped_by_both": len(shared),
        "median_disagreement": round(float(np.median(residual)), 3),
        "p90_disagreement": round(float(np.percentile(residual, 90)), 3),
        "p95_disagreement": round(float(np.percentile(residual, 95)), 3),
        "fraction_over_ten_frames": round(float((residual > 10).mean()), 4),
        "caveat": (
            "Consistency, not correctness. Both cameras can fail together where "
            "the vehicle's own trajectory is the cause, and the unproven camera "
            "synchronisation means only the variation is interpretable."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    target = OUTPUT_DIR / "bayes_diagnostics.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": (
            "DEVELOPMENT ONLY. The blind label set is spent. These labels selected "
            "nothing for this model, but they have been read, so nothing here is a "
            "held-out estimate. A fresh prediction-blind set is required before any "
            "accuracy claim."
        ),
        "tolerance_frames": TOLERANCE_FRAMES,
        "with_motion": {c: score(c, True) for c in ("cam0", "cam5")},
        "without_motion": {c: score(c, False) for c in ("cam0", "cam5")},
        "cross_camera_agreement": cross_camera_agreement(),
    }
    target.write_text(json.dumps(payload, indent=2))

    for camera in ("cam0", "cam5"):
        with_motion = payload["with_motion"][camera]
        without = payload["without_motion"][camera]
        print(f"=== {camera}  (development diagnostic, not a blind result)")
        print(
            f"  accepted {with_motion['accepted']}/{with_motion['determinable_labels']}, "
            f"strict {with_motion['strict_precision']}, "
            f"within {TOLERANCE_FRAMES} frames {with_motion['within_tolerance_precision']}"
        )
        print(
            f"  motion ablation: within-tolerance "
            f"{with_motion['within_tolerance_precision']} with, "
            f"{without['within_tolerance_precision']} without"
        )
        print(f"  expected calibration error {with_motion['expected_calibration_error']}")
        for row in with_motion["reliability"]:
            print(
                f"     stated {row['mean_stated_posterior']:.2f}  "
                f"observed {row['observed_within_tolerance']:.2f}  (n={row['n']})"
            )
    agreement = payload["cross_camera_agreement"]
    print(
        f"\nlabel-free cross-camera agreement over {agreement['frames_mapped_by_both']} frames: "
        f"median {agreement['median_disagreement']}, p95 {agreement['p95_disagreement']}"
    )
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
