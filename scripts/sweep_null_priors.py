#!/usr/bin/env python3
"""Give the null-state priors a provenance, on the same synthetic harness as the threshold.

`P_LEAVE` and `P_RETURN` shape how readily the posterior model enters and leaves
the no-correspondence state. The acceptance threshold was chosen on synthetic
sequences with a known deleted stretch and says so; these two were typed in. The
abstention numbers depend on them, so they deserve the same treatment.

The harness is the one the unit tests use: a clean diagonal correspondence with
a stretch removed. For each (leave, return) pair it reports how much of the gap
is rejected and how much of the rest is falsely rejected. The shipped values are
then either confirmed or moved, and either way the reason is on disk.

This is model selection on simulated data, which is the right place for it. It
touches no label, blind or otherwise.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np

import build_task2_bayes as bayes


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_bayes"

LEAVE_GRID = (0.005, 0.01, 0.02, 0.05, 0.10)
RETURN_GRID = (0.02, 0.05, 0.10, 0.20, 0.40)
GAP = (150, 250)


def synthetic(seed: int, steps: int = 400, states: int = 500):
    rng = np.random.default_rng(seed)
    truth = np.round(np.linspace(20, states - 40, steps)).astype(int)
    evidence = rng.normal(0.0, 1.0, (steps, states))
    for index, position in enumerate(truth):
        evidence[index, max(0, position - 1) : position + 2] += 4.0
    low, high = GAP
    evidence[low:high, :] = rng.normal(0.0, 1.0, (high - low, states))
    return np.exp(np.clip(evidence, -bayes.MAX_ABS_Z, bayes.MAX_ABS_Z))


def score(leave: float, return_: float, seeds: tuple[int, ...]) -> dict[str, float]:
    saved = (bayes.P_LEAVE, bayes.P_RETURN)
    bayes.P_LEAVE, bayes.P_RETURN = leave, return_
    try:
        inside, outside = [], []
        for seed in seeds:
            emission = synthetic(seed)
            _, null_posterior = bayes.forward_backward(
                emission, emission.mean(axis=1), np.array(bayes.DEFAULT_STEP_KERNEL)
            )
            low, high = GAP
            mask = np.zeros(emission.shape[0], dtype=bool)
            mask[low + 10 : high - 10] = True
            reject = null_posterior > bayes.MAX_NULL_POSTERIOR
            inside.append(reject[mask].mean())
            outside.append(reject[~mask].mean())
    finally:
        bayes.P_LEAVE, bayes.P_RETURN = saved
    return {
        "leave": leave,
        "return": return_,
        "gap_rejected": round(float(np.mean(inside)), 4),
        "outside_rejected": round(float(np.mean(outside)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    target = OUTPUT_DIR / "null_prior_sweep.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    seeds = (0, 1, 2)
    results = [
        score(leave, return_, seeds)
        for leave, return_ in itertools.product(LEAVE_GRID, RETURN_GRID)
    ]

    # Selection rule, declared before looking: maximise gap rejection subject to
    # outside rejection at most 8%. Ties go to the more conservative (smaller
    # leave), which is the direction that errs toward answering.
    feasible = [r for r in results if r["outside_rejected"] <= 0.08]
    chosen = max(feasible, key=lambda r: (r["gap_rejected"], -r["leave"])) if feasible else None
    shipped = next(r for r in results if r["leave"] == bayes.P_LEAVE and r["return"] == bayes.P_RETURN)

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "harness": "synthetic diagonal with a 100-frame deleted stretch, 3 seeds",
        "selection_rule": "max gap rejection s.t. outside rejection <= 8%; ties to smaller leave",
        "shipped": shipped,
        "selected_by_rule": chosen,
        "shipped_is_selected": bool(chosen and chosen["leave"] == shipped["leave"] and chosen["return"] == shipped["return"]),
        "grid": results,
        "label_contact": "none",
    }
    target.write_text(json.dumps(payload, indent=2))

    print(f"{'leave':>7} {'return':>7} {'gap rejected':>13} {'outside rejected':>17}")
    for r in results:
        mark = "  <- shipped" if r is shipped else ("  <- rule" if r is chosen else "")
        print(f"{r['leave']:>7.3f} {r['return']:>7.2f} {r['gap_rejected']:>12.1%} {r['outside_rejected']:>16.1%}{mark}")
    print(f"\nshipped values {'ARE' if payload['shipped_is_selected'] else 'are NOT'} what the rule selects")
    print(f"Wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
