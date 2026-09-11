#!/usr/bin/env python3
"""Test abstention by removing stretches of Run B, where the truth is known.

The blind label set cannot evaluate refusal behaviour, because its queries were
sampled entirely inside the stretch both runs traversed and so contain no case
where a correspondence genuinely fails to exist. The obvious remedy - collect
labels where correspondence fails - turns out to be hard here for the same
reason: both runs complete the loop, so almost every place does have a
counterpart. Asking an annotator to find true negatives on this route would
mostly produce "I cannot tell", which is what happened last time.

Removing a stretch of Run B from the candidate pool sidesteps the whole problem.
The frames whose partner was in the removed stretch now genuinely have none, and
that is known by construction rather than by anyone's judgement. Everything
outside the hole is unchanged and serves as the control, so the same run yields
both a hit rate and a false-alarm rate.

This measures the mechanism, not the route: it says whether the method can
notice an absence, not how often absences occur in the wild. That is the right
scope, because the mechanism is what the earlier evaluation could not test at
all.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import build_task2_bayes as bayes
import build_task2_unified as unified
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_abstention"

# Holes are placed away from the route ends, where the loop seam already makes
# the correspondence ambiguous for reasons that have nothing to do with this.
HOLE_WIDTHS = (60, 150, 400)
HOLES_PER_WIDTH = 3
EDGE_MARGIN = 250

# A frame is scored only if it sits well inside the hole. Right at the boundary
# the neighbouring evidence legitimately still supports a match, so counting
# those would penalise correct behaviour.
BOUNDARY_GUARD = 10



def baseline_alignment(
    emission: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    """Where each Run A frame maps with the full candidate pool available.

    This is the model under test's own view, and defining "orphaned" from it
    is circular: it then measures whether the model notices when *its own
    preferred* evidence vanishes. Kept for comparison; the primary definition
    comes from `independent_alignment`.
    """

    posterior, _ = bayes.forward_backward(emission, emission.mean(axis=1), kernel)
    return posterior.argmax(axis=1)


def independent_alignment(camera: str, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Where each Run A frame maps according to a *different* method.

    The unified matcher shares descriptors with the posterior model but not its
    inference, so its alignment is an outside opinion on where the true partner
    sits. Frames it leaves unmapped get -1 and are excluded from scoring.
    """

    import csv

    path = ROOT / "outputs" / "task2_unified" / camera / "frame_mapping_unified.csv"
    mapping = {
        int(r["runA_frame"]): int(r["runB_frame"])
        for r in csv.DictReader(path.open(newline=""))
        if r["runB_frame"]
    }
    column_index = {int(c): i for i, c in enumerate(columns)}
    out = np.full(rows.size, -1, dtype=int)
    for i, frame in enumerate(rows):
        partner = mapping.get(int(frame))
        if partner is not None and partner in column_index:
            out[i] = column_index[partner]
    return out


def run_with_hole(
    emission: np.ndarray, kernel: np.ndarray, low: int, high: int
) -> tuple[np.ndarray, np.ndarray]:
    """Re-run inference with a stretch of Run B made unavailable."""

    punctured = emission.copy()
    # Neutral, not zero: the removed frames must carry no evidence either way,
    # which is what the background level means.
    punctured[:, low:high] = emission.mean(axis=1, keepdims=True)
    posterior, null_posterior = bayes.forward_backward(
        punctured, punctured.mean(axis=1), kernel
    )
    return posterior, null_posterior


def evaluate_camera(camera: str, rng: np.random.Generator) -> dict[str, object]:
    bounds = unified.route_bounds()
    descriptors_a = bayes.descriptors("runA", camera)
    descriptors_b = bayes.descriptors("runB", camera)
    run_a_end = min(unified.RUN_A_ROUTE_TAIL_EXCLUSIVE, descriptors_a.shape[0])

    rows = np.arange(bounds["runA_start"], run_a_end)
    columns = np.arange(bounds["runB_start"], descriptors_b.shape[0])
    emission = np.exp(
        bayes.EVIDENCE_SCALE * bayes.evidence_matrix(descriptors_a, descriptors_b, rows, columns)
    )
    kernel = np.array(
        bayes.step_kernel_from_path(
            ROOT / "outputs" / "task2_unified" / camera / "frame_mapping_unified.csv"
        )
    )

    baseline_self = baseline_alignment(emission, kernel)
    baseline_indep = independent_alignment(camera, rows, columns)
    states = emission.shape[1]

    trials: list[dict[str, object]] = []
    for width in HOLE_WIDTHS:
        for _ in range(HOLES_PER_WIDTH):
            low = int(rng.integers(EDGE_MARGIN, states - EDGE_MARGIN - width))
            high = low + width
            posterior, null_posterior = run_with_hole(emission, kernel, low, high)
            peak = posterior.argmax(axis=1)
            window = bayes.window_posterior(posterior, peak, bayes.POSTERIOR_WINDOW_FRAMES)
            abstains = (null_posterior > bayes.MAX_NULL_POSTERIOR) | (
                window < bayes.MIN_WINDOW_POSTERIOR
            )

            # Ground truth by construction: frames whose partner was removed.
            # Primary definition uses the independent alignment; the model's own
            # is kept alongside so the circularity is visible, not hidden.
            def classify(baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                valid = baseline >= 0
                orphaned = valid & (baseline >= low + BOUNDARY_GUARD) & (baseline < high - BOUNDARY_GUARD)
                unaffected = valid & ((baseline < low - BOUNDARY_GUARD) | (baseline >= high + BOUNDARY_GUARD))
                return orphaned, unaffected

            orph_i, unaf_i = classify(baseline_indep)
            orph_s, unaf_s = classify(baseline_self)
            if orph_i.sum() < 5:
                continue

            trials.append(
                {
                    "hole_width": width,
                    "hole_runB_range": [int(columns[low]), int(columns[high - 1])],
                    "orphaned_frames": int(orph_i.sum()),
                    "unaffected_frames": int(unaf_i.sum()),
                    "abstained_when_orphaned": round(float(abstains[orph_i].mean()), 4),
                    "abstained_when_unaffected": round(float(abstains[unaf_i].mean()), 4),
                    "self_defined_abstained_when_orphaned": round(float(abstains[orph_s].mean()), 4) if orph_s.sum() else None,
                    "median_null_posterior_orphaned": round(float(np.median(null_posterior[orph_i])), 5),
                    "median_null_posterior_unaffected": round(float(np.median(null_posterior[unaf_i])), 5),
                }
            )

    by_width: dict[str, object] = {}
    for width in HOLE_WIDTHS:
        subset = [t for t in trials if t["hole_width"] == width]
        if not subset:
            continue
        by_width[str(width)] = {
            "trials": len(subset),
            "detection_rate": round(
                float(np.mean([t["abstained_when_orphaned"] for t in subset])), 4
            ),
            "detection_rate_self_defined": round(
                float(np.mean([t["self_defined_abstained_when_orphaned"] for t in subset
                               if t["self_defined_abstained_when_orphaned"] is not None])), 4
            ),
            "false_alarm_rate": round(
                float(np.mean([t["abstained_when_unaffected"] for t in subset])), 4
            ),
        }

    return {
        "camera_id": camera,
        "route_frames": int(rows.size),
        "by_hole_width": by_width,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=20260902)
    arguments = parser.parse_args()

    target = OUTPUT_DIR / "abstention_holdout.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    rng = np.random.default_rng(arguments.seed)
    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "posterior model with an explicit null state",
        "ground_truth": (
            "by construction: a stretch of Run B is withheld, so frames whose "
            "partner lay inside it genuinely have none. Which frames those are "
            "is decided by an INDEPENDENT alignment (the unified matcher), not "
            "by the model under test; the model's own definition is reported "
            "beside it so the difference - the circularity - is measurable."
        ),
        "scope": (
            "Measures whether the mechanism can notice an absence. It does not "
            "estimate how often absences occur on a real route, and it is not a "
            "substitute for a held-out label set."
        ),
        "decision_thresholds": {
            "max_null_posterior": bayes.MAX_NULL_POSTERIOR,
            "min_window_posterior": bayes.MIN_WINDOW_POSTERIOR,
        },
        "cameras": {},
    }
    for camera in ("cam0", "cam5"):
        print(f"[{camera}] puncturing Run B and re-running inference ...", flush=True)
        payload["cameras"][camera] = evaluate_camera(camera, rng)

    atomic_write(target, lambda path: path.write_text(json.dumps(payload, indent=2)))

    print(f"\n{'camera':<8} {'hole':>6} {'trials':>7} {'detected':>10} {'(self-def)':>11} {'false alarm':>12}")
    for camera, entry in payload["cameras"].items():
        for width, stats in entry["by_hole_width"].items():
            print(
                f"{camera:<8} {width:>6} {stats['trials']:>7} "
                f"{stats['detection_rate']:>9.1%} {stats['detection_rate_self_defined']:>10.1%} "
                f"{stats['false_alarm_rate']:>11.1%}"
            )
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
