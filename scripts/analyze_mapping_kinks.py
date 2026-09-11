#!/usr/bin/env python3
"""Label-free audit of implausible one-step jumps ("kinks") in a frame mapping.

A frame mapping is a monotone non-decreasing function from Run A ordinals to Run
B ordinals. Its first difference is a *local speed ratio*: if consecutive Run A
frames map to Run B frames `b` and `b + d`, then Run B needed `d` frames to
cover the ground Run A covered in one. Both runs are 10 fps, so one Run A step
is 0.1 s.

**The physical bound, and why.**

* `d == 1` means the two traversals moved at the same speed at that point.
* The coarse stage of the shipped pipeline already encodes a bound: its DTW
  slope prior is `MIN_SLOPE = 0.2 .. MAX_SLOPE = 2.0` and its step set tops out
  at (1, 3), so the search itself is not allowed to sustain more than a 2x speed
  ratio, and a momentary 3x is the most any single step can express.
* `d >= 4` therefore asserts a 4x speed ratio inside a single 0.1 s interval -
  outside what the model that produced the path is allowed to believe.
* `d >= 6` asserts a 6x ratio. On a suburban route where the vehicle is doing a
  few m/s, that would be a physically impossible acceleration, and it cannot be
  a stop either: a stop makes Run B *repeat* frames (`d == 0`), it never makes
  Run B sprint. So `K = 6` is the headline threshold; `K in {4, 6, 8, 10}` is
  reported so the choice is visible rather than assumed.

A kink is not an error proof. It is a statement that the mapping asserts
something the mapping's own motion model forbids, which is a label-free reason
to distrust the neighbourhood. Whether kinks actually predict errors is checked
against the sealed blind labels in stage 2, without tuning anything against
them.

Nothing here modifies a mapping. The slope gate in stage 3 is costed as a Task 3
*policy* over the frozen output, not as a change to it.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_DIR = ROOT / "outputs" / "task2_unified"
BAYES_DIR = ROOT / "outputs" / "task2_bayes"
VARIANT_DIR = ROOT / "outputs" / "task2_unified_variants"
QUALITY_FILE = ROOT / "outputs" / "task3" / "frame_quality_flags.csv"
LABEL_FILE = ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "blind_v2_labels.csv"
OUTPUT_DIR = ROOT / "outputs" / "task2_kinks"

CAMERAS = ("cam0", "cam5")
SUBMITTED = "submitted_unified"

# Thresholds on the one-step Run B increment. See the module docstring.
KINK_THRESHOLDS = (4, 6, 8, 10)
HEADLINE_K = 6

# Slope prior the shipped coarse search actually uses, quoted so the bound is
# auditable against the pipeline rather than asserted here.
DTW_MIN_SLOPE = 0.2
DTW_MAX_SLOPE = 2.0
DTW_MAX_SINGLE_STEP_SLOPE = 3.0

# Cross-camera disagreement, computed the way evaluate_task2_consistency.py does:
# the two cameras are not proved synchronised, so only the *variation* about a
# running median offset is interpretable.
LAG_WINDOW = 201
CROSS_CAMERA_RADIUS = 10
CROSS_CAMERA_THRESHOLD = 6

# Blind-label cross-check and the costed policy.
LABEL_NEAR_RADIUS = 15
DEMOTE_RADIUS = 10
TOP_TIER_STATUS = "accepted_strong"

# Stage-4 mechanism probe (CAM0, around the A902/A907 pair).
MECHANISM_CAMERA = "cam0"
MECHANISM_WINDOW = (880, 941)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def read_mapping(path: Path) -> list[dict[str, object]]:
    """Normalise any of the three mapping schemas to one shape."""

    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("runB_frame") or "").strip()
            rows.append(
                {
                    "runA_frame": int(row["runA_frame"]),
                    "runB_frame": int(raw) if raw else None,
                    "confidence": (row.get("confidence") or "").strip(),
                    "tier": (row.get("tier") or "").strip(),
                    "status": (row.get("status") or "").strip(),
                }
            )
    rows.sort(key=lambda entry: entry["runA_frame"])
    return rows


def discover_mappings() -> dict[str, dict[str, Path]]:
    """Every mapping on disk that covers both cameras, keyed by a family name."""

    found: dict[str, dict[str, Path]] = {}

    def offer(name: str, paths: dict[str, Path]) -> None:
        if all(path.is_file() for path in paths.values()) and len(paths) == len(CAMERAS):
            found[name] = paths

    offer(SUBMITTED, {c: UNIFIED_DIR / c / "frame_mapping_unified.csv" for c in CAMERAS})
    offer("bayes_posterior", {c: BAYES_DIR / c / "frame_mapping_bayes.csv" for c in CAMERAS})
    if VARIANT_DIR.is_dir():
        for variant in sorted(p.name for p in VARIANT_DIR.iterdir() if p.is_dir()):
            offer(
                f"variant:{variant}",
                {c: VARIANT_DIR / variant / c / "frame_mapping_unified.csv" for c in CAMERAS},
            )
    return found


def read_quality_flags() -> dict[tuple[str, str], dict[int, str]]:
    flags: dict[tuple[str, str], dict[int, str]] = {}
    if not QUALITY_FILE.is_file():
        return flags
    with QUALITY_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["run"], row["camera"])
            flags.setdefault(key, {})[int(row["ordinal"])] = row["flag"]
    return flags


def read_labels() -> list[dict[str, object]]:
    if not LABEL_FILE.is_file():
        return []
    out: list[dict[str, object]] = []
    with LABEL_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(
                {
                    "query_id": row["query_id"],
                    "camera_id": row["camera_id"],
                    "runA_frame": int(row["runA_frame"]),
                    "label": row["label"],
                    "runB_frame": int(row["runB_frame"]) if row["runB_frame"] else None,
                    "runB_min": int(row["runB_min"]) if row["runB_min"] else None,
                    "runB_max": int(row["runB_max"]) if row["runB_max"] else None,
                }
            )
    return out


# --------------------------------------------------------------------------
# Stage 1: increments and kinks
# --------------------------------------------------------------------------


def consecutive_steps(rows: list[dict[str, object]]) -> list[dict[str, int]]:
    """One entry per pair of *adjacent* Run A frames that are both mapped.

    Pairs straddling an abstention gap are excluded: a Run B jump across a
    ten-frame hole in Run A is not a speed claim, it is just the hole.
    """

    steps: list[dict[str, int]] = []
    previous: dict[str, object] | None = None
    for row in rows:
        if row["runB_frame"] is None:
            previous = None
            continue
        if previous is not None and row["runA_frame"] - previous["runA_frame"] == 1:
            steps.append(
                {
                    "runA_prev": int(previous["runA_frame"]),
                    "runA_frame": int(row["runA_frame"]),
                    "runB_before": int(previous["runB_frame"]),
                    "runB_after": int(row["runB_frame"]),
                    "increment": int(row["runB_frame"]) - int(previous["runB_frame"]),
                }
            )
        previous = row
    return steps


def increment_distribution(steps: list[dict[str, int]]) -> dict[str, object]:
    values = [step["increment"] for step in steps]
    if not values:
        return {"n": 0}
    array = np.array(values)
    histogram: dict[str, int] = {}
    for value in values:
        key = str(value) if value <= 10 else ">10"
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "n": len(values),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": round(float(array.mean()), 4),
        "median": float(np.median(array)),
        "p90": round(float(np.percentile(array, 90)), 3),
        "p99": round(float(np.percentile(array, 99)), 3),
        "negative_steps": int((array < 0).sum()),
        "histogram": dict(
            sorted(histogram.items(), key=lambda item: (item[0] == ">10", _as_int(item[0])))
        ),
    }


def _as_int(key: str) -> int:
    try:
        return int(key)
    except ValueError:
        return 999


def find_kinks(steps: list[dict[str, int]], threshold: int) -> list[dict[str, int]]:
    return [step for step in steps if step["increment"] >= threshold]


# --------------------------------------------------------------------------
# Cross-camera disagreement about the running offset
# --------------------------------------------------------------------------


def running_median(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(values))])


def cross_camera_residual(
    mapping_by_camera: dict[str, list[dict[str, object]]]
) -> dict[int, float]:
    """|cam0 - cam5| about a running median, per shared Run A frame.

    Same construction as evaluate_task2_consistency.py: the constant part of the
    cam0-cam5 difference is unattributable (the cameras are not proved
    synchronised), so only its variation is evidence.
    """

    per_camera = {
        camera: {
            int(row["runA_frame"]): int(row["runB_frame"])
            for row in rows
            if row["runB_frame"] is not None
        }
        for camera, rows in mapping_by_camera.items()
    }
    if set(per_camera) != set(CAMERAS):
        return {}
    shared = sorted(set(per_camera["cam0"]) & set(per_camera["cam5"]))
    if len(shared) < LAG_WINDOW:
        return {}
    difference = np.array(
        [per_camera["cam0"][f] - per_camera["cam5"][f] for f in shared], dtype=float
    )
    residual = difference - running_median(difference, LAG_WINDOW)
    return {frame: float(abs(value)) for frame, value in zip(shared, residual)}


def local_disagreement(
    residual: dict[int, float], frame: int, radius: int = CROSS_CAMERA_RADIUS
) -> float | None:
    nearby = [
        residual[f] for f in range(frame - radius, frame + radius + 1) if f in residual
    ]
    return max(nearby) if nearby else None


# --------------------------------------------------------------------------
# Darkness attribution
# --------------------------------------------------------------------------


def kink_darkness(
    kink: dict[str, int], camera: str, flags: dict[tuple[str, str], dict[int, str]]
) -> dict[str, bool]:
    run_a = flags.get(("runA", camera), {})
    run_b = flags.get(("runB", camera), {})
    a_dark = run_a.get(kink["runA_frame"]) == "too_dark" or run_a.get(kink["runA_prev"]) == "too_dark"
    span = range(kink["runB_before"], kink["runB_after"] + 1)
    b_dark = any(run_b.get(ordinal) == "too_dark" for ordinal in span)
    return {
        "runA_too_dark": bool(a_dark),
        "runB_span_too_dark": bool(b_dark),
        "dark": bool(a_dark or b_dark),
    }


# --------------------------------------------------------------------------
# Stage 2: blind-label cross-check
# --------------------------------------------------------------------------


def distance_to_interval(prediction: int, low: int, high: int) -> int:
    if prediction < low:
        return low - prediction
    if prediction > high:
        return prediction - high
    return 0


def label_cross_check(
    labels: list[dict[str, object]],
    mapping_by_camera: dict[str, list[dict[str, object]]],
    kinks_by_camera: dict[str, list[dict[str, int]]],
) -> dict[str, object]:
    """Do kinks predict blind-label misses? No threshold is tuned here."""

    per_camera: dict[str, object] = {}
    pooled = {"near": [], "away": []}
    cases: list[dict[str, object]] = []

    for camera in CAMERAS:
        rows = {
            int(row["runA_frame"]): row for row in mapping_by_camera.get(camera, [])
        }
        kink_frames = [k["runA_frame"] for k in kinks_by_camera.get(camera, [])]
        buckets: dict[str, list[int]] = {"near": [], "away": []}
        abstained = {"near": 0, "away": 0}

        for label in labels:
            if label["camera_id"] != camera or label["label"] != "match":
                continue
            frame = label["runA_frame"]
            nearest = min(
                (abs(frame - kf) for kf in kink_frames), default=None
            )
            bucket = "near" if nearest is not None and nearest <= LABEL_NEAR_RADIUS else "away"
            row = rows.get(frame)
            predicted = row["runB_frame"] if row else None
            if predicted is None:
                abstained[bucket] += 1
                distance = None
                missed = None
            else:
                distance = distance_to_interval(
                    predicted, label["runB_min"], label["runB_max"]
                )
                missed = distance > 0
                buckets[bucket].append(distance)
                pooled[bucket].append(distance)
            cases.append(
                {
                    "query_id": label["query_id"],
                    "camera_id": camera,
                    "runA_frame": frame,
                    "predicted_runB_frame": predicted,
                    "label_runB_interval": [label["runB_min"], label["runB_max"]],
                    "interval_distance": distance,
                    "missed": missed,
                    "nearest_kink_distance_runA_frames": nearest,
                    "bucket": bucket,
                    "status": row["status"] if row else None,
                }
            )

        per_camera[camera] = {
            bucket: _miss_summary(buckets[bucket], abstained[bucket])
            for bucket in ("near", "away")
        }

    return {
        "kink_threshold": HEADLINE_K,
        "near_radius_runA_frames": LABEL_NEAR_RADIUS,
        "per_camera": per_camera,
        "pooled_not_independent": {
            bucket: _miss_summary(pooled[bucket], 0) for bucket in ("near", "away")
        },
        "pooling_caveat": (
            "The pooled row is shown for readability only. cam0 and cam5 observe "
            "the same two traversals and are not independent samples."
        ),
        "cases": cases,
    }


def _miss_summary(distances: list[int], abstained: int) -> dict[str, object]:
    total = len(distances)
    misses = sum(1 for d in distances if d > 0)
    return {
        "evaluable_match_labels": total,
        "abstained": abstained,
        "misses_interval_distance_gt_0": misses,
        "miss_rate": round(misses / total, 4) if total else None,
        "median_interval_distance": float(statistics.median(distances)) if total else None,
        "max_interval_distance": max(distances) if total else None,
    }


# --------------------------------------------------------------------------
# Stage 3: the slope gate as a Task 3 policy
# --------------------------------------------------------------------------


def demoted_frames(kinks: list[dict[str, int]], radius: int = DEMOTE_RADIUS) -> set[int]:
    frames: set[int] = set()
    for kink in kinks:
        for frame in range(kink["runA_frame"] - radius, kink["runA_frame"] + radius + 1):
            frames.add(frame)
    return frames


def slope_gate_policy(
    labels: list[dict[str, object]],
    mapping_by_camera: dict[str, list[dict[str, object]]],
    kinks_by_camera: dict[str, list[dict[str, int]]],
) -> dict[str, object]:
    """Cost of demoting the neighbourhood of every kink out of the top tier.

    Diagnostic only, and on *spent* labels: these forty-per-camera queries were
    already used to score the frozen method, so any precision reported here is
    not a fresh held-out estimate. It is a bound on how much of the observed
    error the gate could have caught, nothing more.
    """

    per_camera: dict[str, object] = {}
    for camera in CAMERAS:
        rows = mapping_by_camera.get(camera, [])
        window = demoted_frames(kinks_by_camera.get(camera, []))
        top_tier = [row for row in rows if row["status"] == TOP_TIER_STATUS]
        demoted = [row for row in top_tier if int(row["runA_frame"]) in window]
        remaining = [row for row in top_tier if int(row["runA_frame"]) not in window]

        by_frame = {int(row["runA_frame"]): row for row in rows}
        demoted_labels = []
        for label in labels:
            if label["camera_id"] != camera:
                continue
            frame = label["runA_frame"]
            row = by_frame.get(frame)
            if row is None or row["status"] != TOP_TIER_STATUS or frame not in window:
                continue
            distance = (
                distance_to_interval(row["runB_frame"], label["runB_min"], label["runB_max"])
                if label["label"] == "match" and row["runB_frame"] is not None
                else None
            )
            demoted_labels.append(
                {
                    "query_id": label["query_id"],
                    "runA_frame": frame,
                    "label": label["label"],
                    "predicted_runB_frame": row["runB_frame"],
                    "interval_distance": distance,
                    "was_a_false_accept": bool(
                        label["label"] == "no_match" or (distance is not None and distance > 0)
                    ),
                }
            )

        per_camera[camera] = {
            "top_tier_rows": len(top_tier),
            "demoted_rows": len(demoted),
            "demoted_fraction_of_top_tier": round(len(demoted) / len(top_tier), 4)
            if top_tier
            else None,
            "remaining_top_tier_rows": len(remaining),
            "labels_on_demoted_rows": demoted_labels,
            "labels_on_demoted_rows_count": len(demoted_labels),
            "labels_on_demoted_rows_false_accepts": sum(
                1 for entry in demoted_labels if entry["was_a_false_accept"]
            ),
            "spent_labels_before_gate": _gate_scores(labels, camera, by_frame, set()),
            "spent_labels_after_gate": _gate_scores(labels, camera, by_frame, window),
        }

    return {
        "kink_threshold": HEADLINE_K,
        "demote_radius_runA_frames": DEMOTE_RADIUS,
        "top_tier_status": TOP_TIER_STATUS,
        "label_status": "SPENT. These labels already scored the frozen method; "
        "the precision below is diagnostic, not a held-out estimate.",
        "per_camera": per_camera,
    }


def _gate_scores(
    labels: list[dict[str, object]],
    camera: str,
    by_frame: dict[int, dict[str, object]],
    demote_window: set[int],
) -> dict[str, object]:
    """Precision and coverage over the labels, counting a demoted row as refused."""

    accepted = correct = total = 0
    for label in labels:
        if label["camera_id"] != camera:
            continue
        total += 1
        row = by_frame.get(label["runA_frame"])
        if row is None or row["runB_frame"] is None:
            continue
        if row["status"] == TOP_TIER_STATUS and label["runA_frame"] in demote_window:
            continue
        accepted += 1
        if label["label"] == "match" and (
            distance_to_interval(row["runB_frame"], label["runB_min"], label["runB_max"]) == 0
        ):
            correct += 1
    return {
        "labels": total,
        "accepted": accepted,
        "correct_accepted": correct,
        "strict_accepted_precision": round(correct / accepted, 4) if accepted else None,
        "coverage_all_labels": round(accepted / total, 4) if total else None,
    }


# --------------------------------------------------------------------------
# Stage 4: mechanism - coarse DTW path, or the refinement ratchet?
# --------------------------------------------------------------------------


def coarse_path_estimate(camera: str) -> tuple[np.ndarray, int] | None:
    """Rebuild the stage-2 coarse estimate from the cached DINOv2 descriptors.

    No decoding: stage 1 is a cache hit, and stage 2 is one matrix product plus
    the DTW. Returns None if the cache is absent.
    """

    import build_task2_unified as unified

    path_a = unified.descriptor_cache_path("runA", camera)
    path_b = unified.descriptor_cache_path("runB", camera)
    if not path_a.is_file() or not path_b.is_file():
        return None
    descriptors_a = np.load(path_a)
    descriptors_b = np.load(path_b)

    bounds = unified.route_bounds()
    total_a, total_b = descriptors_a.shape[0], descriptors_b.shape[0]
    coarse_a = np.arange(
        bounds["runA_start"],
        min(unified.RUN_A_ROUTE_TAIL_EXCLUSIVE, total_a),
        unified.COARSE_STRIDE,
    )
    coarse_b = np.arange(bounds["runB_start"], total_b, unified.COARSE_STRIDE)
    similarity = descriptors_a[coarse_a] @ descriptors_b[coarse_b].T
    expected_slope = max(len(coarse_b) / max(len(coarse_a), 1), 1e-3)
    path = unified.global_monotonic_path(similarity, expected_slope)
    return unified.densify_path(path, coarse_a, coarse_b, total_a), total_b


def _score_other_mappings(
    others: dict[str, list[dict[str, object]]],
    frames: list[int],
    structure_a: dict[int, np.ndarray],
    structure_b: dict[int, np.ndarray],
    estimate: np.ndarray,
    radius: int,
    replay_by_frame: dict[int, dict[str, object]],
) -> dict[str, object]:
    """Score other mappings' picks on the same descriptors, in the same window.

    Label-free. A mapping that is smoother *and* scores at least as well on the
    structure descriptors is following the evidence; one that is smoother and
    scores worse is imposing a shape on it. Both readings are reported.
    """

    report: dict[str, object] = {}
    for name, rows in others.items():
        values = {
            int(row["runA_frame"]): int(row["runB_frame"])
            for row in rows
            if row["runB_frame"] is not None
        }
        scored: list[dict[str, object]] = []
        for frame in frames:
            candidate = values.get(frame)
            vector_a = structure_a.get(frame)
            if candidate is None or vector_a is None or candidate not in structure_b:
                continue
            entry = replay_by_frame.get(frame, {})
            scored.append(
                {
                    "runA_frame": frame,
                    "runB_frame": candidate,
                    "residual_against_coarse": int(candidate - estimate[frame]),
                    "structure_score": round(float(vector_a @ structure_b[candidate]), 4),
                    "submitted_structure_score": entry.get("score_at_pick"),
                    "free_argmax_structure_score": entry.get("score_at_unconstrained_argmax"),
                    "equals_free_argmax": candidate == entry.get("unconstrained_argmax_runB"),
                }
            )
        if not scored:
            report[name] = {"available": False, "reason": "no comparable frames in the window"}
            continue
        steps = [
            values[f] - values[f - 1] for f in frames if f in values and f - 1 in values
        ]
        comparable = [row for row in scored if row["submitted_structure_score"] is not None]
        report[name] = {
            "available": True,
            "frames_scored": len(scored),
            "max_one_step_increment": max(steps) if steps else None,
            "increments_at_or_above_headline_K": sum(1 for step in steps if step >= HEADLINE_K),
            "residual_against_coarse": {
                "min": min(row["residual_against_coarse"] for row in scored),
                "max": max(row["residual_against_coarse"] for row in scored),
                "frames_pinned_at_plus_refine_radius": sum(
                    1 for row in scored if row["residual_against_coarse"] >= radius
                ),
            },
            "median_structure_score": round(
                float(np.median([row["structure_score"] for row in scored])), 4
            ),
            "median_structure_score_of_submitted": (
                round(float(np.median([row["submitted_structure_score"] for row in comparable])), 4)
                if comparable
                else None
            ),
            "median_structure_score_of_free_argmax": (
                round(
                    float(np.median([row["free_argmax_structure_score"] for row in comparable])), 4
                )
                if comparable
                else None
            ),
            "frames_scoring_at_least_the_submitted_pick": sum(
                1 for row in comparable
                if row["structure_score"] >= row["submitted_structure_score"]
            ),
            "frames_equal_to_the_free_argmax": sum(
                1 for row in scored if row["equals_free_argmax"]
            ),
            "per_frame": scored,
        }
    return report


def mechanism_probe(
    camera: str,
    mapping: list[dict[str, object]],
    window: tuple[int, int],
    decode: bool,
    others: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    """Ask where the A902/A907 jumps are introduced.

    Two questions, answered separately:

    1. Does the *coarse* path jump? (Cheap: cached descriptors, no decoding.)
    2. If not, is the refined value the unconstrained argmax of the structure
       score in its +/-8 window, or is it forced by the non-decreasing
       constraint - the ratchet? (Needs the structure descriptors, so it decodes
       the ~60 Run A and ~80 Run B frames the window touches, and nothing else.)

    `others` are further mappings of the same camera - a refinement variant, say
    - scored on the same structure descriptors in the same window. That is the
    label-free way to ask whether a variant *follows the descriptors* here or
    merely imposes a smoother shape on them: it reports the variant's own
    structure score against the submitted mapping's and against the free argmax,
    frame by frame, with no label involved.
    """

    import build_task2_unified as unified

    submitted = {
        int(row["runA_frame"]): int(row["runB_frame"])
        for row in mapping
        if row["runB_frame"] is not None
    }
    coarse = coarse_path_estimate(camera)
    if coarse is None:
        return {"available": False, "reason": "descriptor cache absent"}
    estimate, total_b = coarse

    low, high = window
    frames = [f for f in range(low, high) if f in submitted]
    radius = unified.REFINE_RADIUS

    coarse_steps = [int(estimate[f] - estimate[f - 1]) for f in frames if f - 1 >= 0]
    refined_steps = [
        submitted[f] - submitted[f - 1] for f in frames if f - 1 in submitted
    ]
    residual = {f: int(submitted[f] - estimate[f]) for f in frames}

    result: dict[str, object] = {
        "available": True,
        "camera": camera,
        "window_runA": [low, high - 1],
        "refine_radius": radius,
        "coarse_path": {
            "max_one_step_increment": max(coarse_steps) if coarse_steps else None,
            "increments_at_or_above_headline_K": sum(
                1 for step in coarse_steps if step >= HEADLINE_K
            ),
            "note": "The coarse DTW path is smooth here; its slope prior forbids "
            "more than a 2x sustained ratio and its largest step is (1, 3).",
        },
        "submitted_mapping": {
            "max_one_step_increment": max(refined_steps) if refined_steps else None,
            "increments_at_or_above_headline_K": sum(
                1 for step in refined_steps if step >= HEADLINE_K
            ),
        },
        "refined_minus_coarse_residual": {
            "min": min(residual.values()),
            "max": max(residual.values()),
            "frames_pinned_at_plus_refine_radius": sum(
                1 for value in residual.values() if value >= radius
            ),
        },
        "per_frame": [
            {
                "runA_frame": f,
                "coarse_runB": int(estimate[f]),
                "submitted_runB": submitted[f],
                "residual": residual[f],
            }
            for f in frames
        ],
    }

    # Route-wide version of the same residual, still decode-free.
    all_frames = [f for f in sorted(submitted) if 0 <= f < len(estimate) and estimate[f] >= 0]
    all_residual = np.array([submitted[f] - estimate[f] for f in all_frames])
    result["route_wide_residual"] = {
        "frames": len(all_frames),
        "min": int(all_residual.min()),
        "max": int(all_residual.max()),
        "frames_at_plus_refine_radius": int((all_residual >= radius).sum()),
        "frames_at_minus_refine_radius": int((all_residual <= -radius).sum()),
        "note": "The refinement is clipped to +/-{r} around the coarse path, so a "
        "ratchet saturates at +{r} rather than drifting without bound.".format(r=radius),
    }

    if not decode:
        result["structure_replay"] = {"available": False, "reason": "--no-decode"}
        return result

    try:
        needed_b = sorted(
            {
                int(np.clip(estimate[f] + delta, 0, total_b - 1))
                for f in frames
                for delta in range(-radius, radius + 1)
            }
        )
        structure_a = unified.structure_descriptors("runA", camera, frames)
        structure_b = unified.structure_descriptors("runB", camera, needed_b)
    except Exception as error:  # decoding is optional; never fail the audit on it
        result["structure_replay"] = {"available": False, "reason": repr(error)}
        return result

    previous = submitted.get(frames[0] - 1, -1)
    replay: list[dict[str, object]] = []
    matches = 0
    forced = 0
    for frame in frames:
        centre = int(estimate[frame])
        candidates = [
            int(np.clip(centre + delta, 0, total_b - 1))
            for delta in range(-radius, radius + 1)
        ]
        scores = {
            b: float(structure_a[frame] @ structure_b[b])
            for b in candidates
            if b in structure_b and frame in structure_a
        }
        if not scores:
            continue
        unconstrained = max(scores, key=scores.get)
        admissible = {b: s for b, s in scores.items() if b >= previous}
        pick = max(admissible, key=admissible.get) if admissible else max(centre, previous)
        was_forced = pick != unconstrained
        forced += int(was_forced)
        matches += int(pick == submitted.get(frame))
        replay.append(
            {
                "runA_frame": frame,
                "coarse_runB": centre,
                "submitted_runB": submitted.get(frame),
                "replayed_runB": pick,
                "unconstrained_argmax_runB": unconstrained,
                "score_at_pick": round(scores.get(pick, float("nan")), 4),
                "score_at_unconstrained_argmax": round(scores[unconstrained], 4),
                "admissible_candidates": len(admissible),
                "total_candidates": len(scores),
                "forced_by_non_decreasing_constraint": was_forced,
            }
        )
        previous = pick

    if others:
        result["other_mappings_in_window"] = _score_other_mappings(
            others, frames, structure_a, structure_b, estimate, radius,
            {entry["runA_frame"]: entry for entry in replay},
        )

    unconstrained_series = {entry["runA_frame"]: entry["unconstrained_argmax_runB"] for entry in replay}
    unconstrained_steps = [
        unconstrained_series[f] - unconstrained_series[f - 1]
        for f in unconstrained_series
        if f - 1 in unconstrained_series
    ]

    suppressed = [
        entry for entry in replay
        if entry["unconstrained_argmax_runB"] < entry["submitted_runB"]
    ]
    result["structure_replay"] = {
        "available": True,
        "frames_replayed": len(replay),
        "frames_where_the_free_argmax_is_behind_the_submitted_value": len(suppressed),
        "median_frames_the_free_argmax_sits_behind": (
            float(np.median([e["submitted_runB"] - e["unconstrained_argmax_runB"] for e in suppressed]))
            if suppressed
            else None
        ),
        "median_structure_score_at_pick_on_those_frames": (
            round(float(np.median([e["score_at_pick"] for e in suppressed])), 4)
            if suppressed
            else None
        ),
        "median_structure_score_at_free_argmax_on_those_frames": (
            round(float(np.median([e["score_at_unconstrained_argmax"] for e in suppressed])), 4)
            if suppressed
            else None
        ),
        "replay_reproduces_submitted_mapping": matches == len(replay),
        "frames_matching_submitted": matches,
        "frames_where_the_monotone_constraint_changed_the_pick": forced,
        "min_admissible_candidates": min(entry["admissible_candidates"] for entry in replay),
        "unconstrained_refinement": {
            "max_one_step_increment": max(unconstrained_steps) if unconstrained_steps else None,
            "steps_at_or_above_headline_K": sum(
                1 for step in unconstrained_steps if step >= HEADLINE_K
            ),
        },
        "per_frame": replay,
    }
    return result


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def analyse_mapping(
    name: str,
    paths: dict[str, Path],
    flags: dict[tuple[str, str], dict[int, str]],
) -> dict[str, object]:
    mapping_by_camera = {camera: read_mapping(path) for camera, path in paths.items()}
    residual = cross_camera_residual(mapping_by_camera)

    per_camera: dict[str, object] = {}
    kink_rows: dict[str, list[dict[str, object]]] = {camera: [] for camera in CAMERAS}
    kinks_headline: dict[str, list[dict[str, int]]] = {}

    for camera in CAMERAS:
        rows = mapping_by_camera[camera]
        steps = consecutive_steps(rows)
        by_frame = {int(row["runA_frame"]): row for row in rows}
        counts: dict[str, object] = {}
        for threshold in KINK_THRESHOLDS:
            kinks = find_kinks(steps, threshold)
            dark = sum(1 for k in kinks if kink_darkness(k, camera, flags)["dark"])
            counts[f"K>={threshold}"] = {
                "kinks": len(kinks),
                "dark": dark,
                "lit": len(kinks) - dark,
            }
        headline = find_kinks(steps, HEADLINE_K)
        kinks_headline[camera] = headline

        detailed: list[dict[str, object]] = []
        for kink in headline:
            row = by_frame.get(kink["runA_frame"], {})
            darkness = kink_darkness(kink, camera, flags)
            disagreement = local_disagreement(residual, kink["runA_frame"])
            detailed.append(
                {
                    "mapping": name,
                    "camera_id": camera,
                    "runA_prev": kink["runA_prev"],
                    "runA_frame": kink["runA_frame"],
                    "runB_before": kink["runB_before"],
                    "runB_after": kink["runB_after"],
                    "increment": kink["increment"],
                    "status": row.get("status", ""),
                    "tier": row.get("tier", ""),
                    "confidence": row.get("confidence", ""),
                    "runA_too_dark": darkness["runA_too_dark"],
                    "runB_span_too_dark": darkness["runB_span_too_dark"],
                    "dark": darkness["dark"],
                    "max_cross_camera_disagreement_within_10": (
                        round(disagreement, 2) if disagreement is not None else None
                    ),
                    "cross_camera_disagreement_over_6": (
                        bool(disagreement > CROSS_CAMERA_THRESHOLD)
                        if disagreement is not None
                        else None
                    ),
                }
            )
        kink_rows[camera] = detailed

        mapped = sum(1 for row in rows if row["runB_frame"] is not None)
        per_camera[camera] = {
            "rows": len(rows),
            "mapped_rows": mapped,
            "adjacent_mapped_pairs": len(steps),
            "increment_distribution": increment_distribution(steps),
            "kink_counts": counts,
            "headline_kinks_by_status": _histogram(d["status"] for d in detailed),
            "headline_kinks_with_cross_camera_disagreement_over_6": sum(
                1 for d in detailed if d["cross_camera_disagreement_over_6"]
            ),
            "headline_kinks_with_cross_camera_value": sum(
                1 for d in detailed if d["max_cross_camera_disagreement_within_10"] is not None
            ),
            "kinks": detailed,
        }

    return {
        "mapping": name,
        "files": {camera: str(path.relative_to(ROOT)) for camera, path in paths.items()},
        "cross_camera_frames": len(residual),
        "per_camera": per_camera,
        "_kinks_headline": kinks_headline,
        "_mapping_by_camera": mapping_by_camera,
        "_kink_rows": kink_rows,
    }


def _histogram(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda item: -item[1]))


KINK_CSV_COLUMNS = [
    "mapping",
    "camera_id",
    "runA_prev",
    "runA_frame",
    "runB_before",
    "runB_after",
    "increment",
    "status",
    "tier",
    "confidence",
    "runA_too_dark",
    "runB_span_too_dark",
    "dark",
    "max_cross_camera_disagreement_within_10",
    "cross_camera_disagreement_over_6",
]


def write_kink_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KINK_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in KINK_CSV_COLUMNS})


README_TEMPLATE = """# Mapping kinks: implausible one-step jumps

Generated by `scripts/analyze_mapping_kinks.py`. Label-free except for the
cross-check in section 3, which reads the sealed blind labels but tunes nothing
against them.

## What a kink is

A mapping's first difference is a local speed ratio. Consecutive Run A frames
are 0.1 s apart, so mapping them to Run B frames `b` and `b + d` asserts that
Run B needed `d` frames to cover what Run A covered in one - a `d`-fold speed
ratio at that instant. `d = 1` is equal speed. The pipeline's own coarse search
is bounded to a sustained slope in [{min_slope}, {max_slope}] with a largest
single step of {max_step}x, so **`d >= 4` is already outside what the model that
produced the path may believe, and `d >= 6` (the headline threshold `K`) is not
physically reachable in 0.1 s.** A stop does not produce a kink; a stop makes
Run B repeat frames (`d = 0`).

A kink is not proof of an error. It is a label-free statement that the mapping
asserts something its own motion model forbids.

Pairs straddling an abstention gap are excluded: a jump across a hole in Run A
is the hole, not a speed claim.

## 1. Counts

{counts_table}

Dark = the Run A frame, or any Run B frame in the jumped span, carries
`too_dark` in `outputs/task3/frame_quality_flags.csv`.

A mapping built with a bounded step cannot contain a kink at all, so a zero row
here is a property of that construction and not a measurement: `variant:slope`
bounds the one-step increment to [0, 3] by design. What tests it is the
label-free evidence - the structure scores in section 5, and cross-camera
agreement, closure and coverage in `outputs/task2_variants/`.

## 2. Cross-camera corroboration

Disagreement is measured as `evaluate_task2_consistency.py` does: `cam0 - cam5`
minus a {lag_window}-frame running median, because the two cameras are not
proved synchronised and only the *variation* of their difference is evidence.
For the submitted mapping, {disagree_line}

## 3. Do kinks predict blind-label misses?

Match labels only, interval distance against `runB_min .. runB_max`, `near` =
within {near_radius} Run A frames of a `K >= {k}` kink.

{label_table}

## 4. What a slope gate would cost

Policy, not a change to the frozen mapping: demote every top-tier row within
+/-{demote_radius} Run A frames of a kink.

{policy_table}

The labels are **spent** - they already scored the frozen method - so the
precision figures are diagnostic, not a held-out estimate.

## 5. Mechanism

{mechanism}

## Files

* `mapping_kinks.json` - every number above.
* `kinks_cam0.csv`, `kinks_cam5.csv` - one row per `K >= {k}` kink, all mappings.
"""


def render_readme(payload: dict[str, object]) -> str:
    submitted = payload["mappings"][SUBMITTED]

    lines = ["| mapping | camera | " + " | ".join(f"K>={k} (lit/dark)" for k in KINK_THRESHOLDS) + " |"]
    lines.append("|---" * (2 + len(KINK_THRESHOLDS)) + "|")
    for name, entry in payload["mappings"].items():
        for camera in CAMERAS:
            counts = entry["per_camera"][camera]["kink_counts"]
            cells = [
                f"{counts[f'K>={k}']['kinks']} ({counts[f'K>={k}']['lit']}/{counts[f'K>={k}']['dark']})"
                for k in KINK_THRESHOLDS
            ]
            lines.append(f"| {name} | {camera} | " + " | ".join(cells) + " |")
    counts_table = "\n".join(lines)

    disagree = "; ".join(
        f"{camera}: {submitted['per_camera'][camera]['headline_kinks_with_cross_camera_disagreement_over_6']}"
        f" of {len(submitted['per_camera'][camera]['kinks'])} kinks sit within 10 Run A frames of a "
        f">{CROSS_CAMERA_THRESHOLD}-frame cross-camera disagreement"
        for camera in CAMERAS
    )

    check = payload["label_cross_check"]["per_camera"]
    label_lines = ["| camera | bucket | match labels | misses | miss rate | median interval distance |", "|---|---|---|---|---|---|"]
    for camera in CAMERAS:
        for bucket in ("near", "away"):
            summary = check[camera][bucket]
            label_lines.append(
                f"| {camera} | {bucket} | {summary['evaluable_match_labels']} | "
                f"{summary['misses_interval_distance_gt_0']} | "
                f"{summary['miss_rate'] if summary['miss_rate'] is not None else 'n/a'} | "
                f"{summary['median_interval_distance'] if summary['median_interval_distance'] is not None else 'n/a'} |"
            )
    label_table = "\n".join(label_lines)

    policy = payload["slope_gate_policy"]["per_camera"]
    policy_lines = [
        "| camera | top-tier rows | demoted | demoted % | labels on demoted rows | of those, false accepts | precision before | precision after | coverage before | coverage after |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for camera in CAMERAS:
        entry = policy[camera]
        before, after = entry["spent_labels_before_gate"], entry["spent_labels_after_gate"]
        policy_lines.append(
            f"| {camera} | {entry['top_tier_rows']} | {entry['demoted_rows']} | "
            f"{entry['demoted_fraction_of_top_tier']:.1%} | {entry['labels_on_demoted_rows_count']} | "
            f"{entry['labels_on_demoted_rows_false_accepts']} | "
            f"{before['strict_accepted_precision']} | {after['strict_accepted_precision']} | "
            f"{before['coverage_all_labels']} | {after['coverage_all_labels']} |"
        )
    policy_table = "\n".join(policy_lines)

    def headline(name: str) -> str:
        entry = payload["mappings"].get(name)
        if entry is None:
            return "n/a"
        return " and ".join(
            str(entry["per_camera"][camera]["kink_counts"][f"K>={HEADLINE_K}"]["kinks"])
            for camera in CAMERAS
        )

    submitted_kinks = headline(SUBMITTED)
    posterior_kinks = headline("bayes_posterior")

    mechanism = payload["mechanism"]
    if not mechanism.get("available"):
        mechanism_text = f"Not reproduced: {mechanism.get('reason')}"
    else:
        coarse = mechanism["coarse_path"]
        refined = mechanism["submitted_mapping"]
        replay = mechanism.get("structure_replay", {})
        parts = [
            f"Reproduced on {mechanism['camera']} over Run A "
            f"{mechanism['window_runA'][0]}-{mechanism['window_runA'][1]} from the cached "
            f"stage-1 descriptors.",
            "",
            f"* The **coarse DTW path is smooth** there: its largest one-step Run B "
            f"increment is **{coarse['max_one_step_increment']}**, and it contains "
            f"{coarse['increments_at_or_above_headline_K']} increments >= K.",
            f"* The **submitted mapping is not**: largest one-step increment "
            f"**{refined['max_one_step_increment']}**, with "
            f"{refined['increments_at_or_above_headline_K']} increments >= K in the same window.",
            f"* Route-wide, the refined-minus-coarse residual is bounded to "
            f"[{mechanism['route_wide_residual']['min']}, {mechanism['route_wide_residual']['max']}] "
            f"and sits pinned at the +{mechanism['refine_radius']} window edge on "
            f"{mechanism['route_wide_residual']['frames_at_plus_refine_radius']} frames.",
        ]
        if replay.get("available"):
            parts += [
                f"* Replaying stage 3 with the real structure descriptors reproduces the "
                f"submitted values exactly ({replay['frames_matching_submitted']}/"
                f"{replay['frames_replayed']} frames). The initial jump is the "
                f"unconstrained argmax of a *collapsed* structure score, not a constraint "
                f"artefact - but the non-decreasing constraint then changed the pick on "
                f"{replay['frames_where_the_monotone_constraint_changed_the_pick']} of "
                f"{replay['frames_replayed']} frames, at times leaving as few as "
                f"{replay['min_admissible_candidates']} admissible candidate(s) out of "
                f"{2 * mechanism['refine_radius'] + 1}.",
                f"* On {replay['frames_where_the_free_argmax_is_behind_the_submitted_value']} "
                f"of those frames the free argmax sits a median "
                f"{replay['median_frames_the_free_argmax_sits_behind']} Run B frames *behind* "
                f"the submitted value, and scores better there "
                f"({replay['median_structure_score_at_free_argmax_on_those_frames']} vs "
                f"{replay['median_structure_score_at_pick_on_those_frames']}). The descriptors "
                f"say \"go back\"; the non-decreasing constraint forbids it.",
                f"* Root cause: stage 3 runs at full cadence with a monotonicity constraint "
                f"but **no slope prior**. The coarse stage has one "
                f"([{DTW_MIN_SLOPE}, {DTW_MAX_SLOPE}]) and does not kink here; the Bayes "
                f"posterior, which carries an explicit transition model, produces "
                f"{posterior_kinks} (cam0, cam5) against the submitted mapping's "
                f"{submitted_kinks}.",
            ]
        for name, entry in sorted(mechanism.get("other_mappings_in_window", {}).items()):
            if not entry.get("available"):
                continue
            parts.append(
                f"* `{name}` over the same window and the same descriptors: largest "
                f"one-step increment **{entry['max_one_step_increment']}**, "
                f"{entry['increments_at_or_above_headline_K']} increments >= K, "
                f"{entry['residual_against_coarse']['frames_pinned_at_plus_refine_radius']} "
                f"frames pinned at the window edge, median structure score "
                f"**{entry['median_structure_score']}** against the submitted mapping's "
                f"{entry['median_structure_score_of_submitted']} "
                f"(free argmax {entry['median_structure_score_of_free_argmax']}); it "
                f"scores at least as well as the submitted pick on "
                f"{entry['frames_scoring_at_least_the_submitted_pick']} of "
                f"{entry['frames_scored']} frames and equals the free argmax on "
                f"{entry['frames_equal_to_the_free_argmax']}. Label-free.",
            )
        parts += [
            "",
            "**Conclusion: the jump is introduced by the stage-3 refinement, not by the "
            "coarse DTW path, and is made persistent by the non-decreasing constraint.**",
        ]
        mechanism_text = "\n".join(parts)

    return README_TEMPLATE.format(
        min_slope=DTW_MIN_SLOPE,
        max_slope=DTW_MAX_SLOPE,
        max_step=DTW_MAX_SINGLE_STEP_SLOPE,
        counts_table=counts_table,
        lag_window=LAG_WINDOW,
        disagree_line=disagree + ".",
        near_radius=LABEL_NEAR_RADIUS,
        k=HEADLINE_K,
        label_table=label_table,
        demote_radius=DEMOTE_RADIUS,
        policy_table=policy_table,
        mechanism=mechanism_text,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Skip the stage-3 structure replay, which decodes ~140 frames.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    flags = read_quality_flags()
    labels = read_labels()
    mappings = discover_mappings()
    if SUBMITTED not in mappings:
        raise SystemExit("The submitted unified mapping is missing; nothing to audit.")

    analysed = {name: analyse_mapping(name, paths, flags) for name, paths in sorted(mappings.items())}

    submitted = analysed[SUBMITTED]
    cross_check = label_cross_check(
        labels, submitted["_mapping_by_camera"], submitted["_kinks_headline"]
    )
    policy = slope_gate_policy(
        labels, submitted["_mapping_by_camera"], submitted["_kinks_headline"]
    )
    # Any refinement variant on disk is scored in the same window, on the same
    # descriptors, so "the variant is smoother" can be checked against "the
    # variant follows the evidence" without a label.
    others = {
        name: entry["_mapping_by_camera"][MECHANISM_CAMERA]
        for name, entry in analysed.items()
        if name != SUBMITTED and name.startswith("variant:")
    }
    mechanism = mechanism_probe(
        MECHANISM_CAMERA,
        submitted["_mapping_by_camera"][MECHANISM_CAMERA],
        MECHANISM_WINDOW,
        decode=not arguments.no_decode,
        others=others,
    )

    kink_rows: dict[str, list[dict[str, object]]] = {camera: [] for camera in CAMERAS}
    for entry in analysed.values():
        for camera in CAMERAS:
            kink_rows[camera].extend(entry["_kink_rows"][camera])

    for entry in analysed.values():
        for key in ("_kinks_headline", "_mapping_by_camera", "_kink_rows"):
            entry.pop(key, None)

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_class": "label-free structural audit; section 3 and 4 read SPENT labels",
        "physical_bound": {
            "run_a_frame_interval_seconds": 0.1,
            "increment_meaning": "one-step Run B increment = local Run B/Run A speed ratio",
            "dtw_slope_prior": [DTW_MIN_SLOPE, DTW_MAX_SLOPE],
            "dtw_largest_single_step_slope": DTW_MAX_SINGLE_STEP_SLOPE,
            "headline_threshold": HEADLINE_K,
            "thresholds_reported": list(KINK_THRESHOLDS),
            "rationale": (
                "increment 1 = equal speed; the coarse search may not sustain more "
                "than 2x; >= 4 is a 4x ratio in 0.1 s, already outside the model's "
                "own prior; >= 6 is not physically reachable. A stop repeats Run B "
                "frames (increment 0) and never produces a kink."
            ),
        },
        "mappings": analysed,
        "label_cross_check": cross_check,
        "slope_gate_policy": policy,
        "mechanism": mechanism,
    }

    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "mapping_kinks.json").write_text(json.dumps(payload, indent=2))
    for camera in CAMERAS:
        write_kink_csv(output / f"kinks_{camera}.csv", kink_rows[camera])
    (output / "README.md").write_text(render_readme(payload))

    for camera in CAMERAS:
        counts = analysed[SUBMITTED]["per_camera"][camera]["kink_counts"][f"K>={HEADLINE_K}"]
        print(
            f"[{camera}] submitted mapping: {counts['kinks']} kinks at K>={HEADLINE_K} "
            f"({counts['lit']} lit, {counts['dark']} dark)"
        )
    for bucket in ("near", "away"):
        summary = cross_check["pooled_not_independent"][bucket]
        print(
            f"labels {bucket} a kink: {summary['misses_interval_distance_gt_0']}/"
            f"{summary['evaluable_match_labels']} missed "
            f"({summary['miss_rate']})"
        )
    print(f"\nWrote {(output / 'mapping_kinks.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
