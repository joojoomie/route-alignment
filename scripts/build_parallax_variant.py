#!/usr/bin/env python3
"""Re-choose each Run B partner by how uniform its displacement field is with depth.

Development variant. It never touches the frozen mapping; it writes to
`outputs/task2_unified_variants/parallax/` where `evaluate_variants.py` finds it.

The idea. Both cameras face sideways. Between the runs CAM5's camera is yawed
and pitched a little: at the matcher's chosen pairs the scene sits at
dx = -37.5 px, at the human labels at -56 px, dy = -11 px; on CAM0 both are 0.
A camera rotation moves near and far scene content by the same number of pixels.
Driving further along the road does not: near content (kerb, road edge) sweeps
faster than far content (buildings, trees). So among the Run B frames around the
baseline partner, the one where the vehicle stood at the same road position is
the one whose Run A -> Run B displacement field is most *uniform with depth*,
whatever its overall shift. Depth is not measured, so image row stands in for
it: the bottom band of the usable rows is near, the top band is far.

What is measured, per candidate: RootSIFT correspondences under the reliability
mask (which already removes the ego wing and Run B's static droplets), the
horizontal displacement dx of each match and the row it sits on in Run A, then

    uniformity score = |median dx(near band) - median dx(far band)|

with a Theil-Sen slope of dx against row as a second, independent reading, plus
the overall median dx and dy. The candidate with the smallest score wins, ties
break toward the baseline partner.

Two honest caveats, both visible in the numbers this script writes.

* The lens is a fisheye. A pure yaw does *not* shift a fisheye image uniformly,
  so the criterion's premise ("rotation is uniform, translation is not") holds
  only to first order near the image centre. Any residual row-dependence of the
  rotation term is absorbed into the score and biases the choice.
* The per-sample score is noisy. Its aggregate over ~190 samples is the reading
  worth quoting, not any single pick; the manifest reports both.

The matching ratio is the one parameter that differs from the shipped geometry
stage, and it was set on label-free grounds: at the shipped 0.80, only 12% of
CAM5 and 24% of CAM0 candidates reach the required 12 matches in *both* bands,
so the criterion is simply not measurable. At 0.90, with cross-checking kept,
it is 100%, and dx against candidate offset is at least as clean a straight line
(Spearman |0.84| vs |0.82| on CAM5, |0.74| vs |0.52| on CAM0; median residual
7.4 px vs 11.0 px on CAM0). The measurability figures are recorded in the
manifest so the choice can be re-checked without rerunning the study.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

import build_reliability_masks
import build_task2_unified as unified
import frame_service
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "outputs" / "task2_unified"
OUTPUT_DIR = unified.VARIANT_OUTPUT_DIR / "parallax"
CAMERAS = ("cam0", "cam5")

VARIANT_VERSION = "parallax-v1"

# Candidate window around the baseline partner, in Run B frames.
SEARCH_RADIUS = 5

# Row bands, as a fraction of the rows RootSIFT is allowed to use (the detector
# already drops the bottom BOTTOM_EXCLUSION_FRACTION, which is ego bodywork).
BAND_FRACTION = 1.0 / 3.0
MIN_BAND_MATCHES = 12
MIN_MEASURABLE_CANDIDATES = 3

# See the module docstring: chosen on measurability and on the straightness of
# dx against candidate offset, both label-free. The shipped geometry stage keeps
# unified.SIFT_RATIO_THRESHOLD; nothing here changes that constant.
SIFT_RATIO = 0.90

# Theil-Sen is O(n^2) in the number of matches; above this many the pairs are
# subsampled with a fixed seed so the reading stays reproducible and bounded.
MAX_SLOPE_POINTS = 600
SLOPE_SEED = 20260902



# --------------------------------------------------------------------------
# The same sample set stage 5 of the unified pipeline verifies
# --------------------------------------------------------------------------


def load_baseline(camera: str) -> list[dict[str, str]]:
    path = BASELINE_DIR / camera / "frame_mapping_unified.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


_FRAME_COUNTS: dict[tuple[str, str], int] = {}


def frame_count(run: str, camera: str) -> int:
    """`unified.frame_count`, memoised; each call rescans the Annex-B stream."""

    key = (run, camera)
    if key not in _FRAME_COUNTS:
        _FRAME_COUNTS[key] = unified.frame_count(run, camera)
    return _FRAME_COUNTS[key]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def geometry_samples(rows: list[dict[str, str]]) -> list[tuple[int, int]]:
    """Reproduce stage 5's sample set from the written mapping.

    Stage 5 takes the route frames, strides by GEOMETRY_SAMPLE_STRIDE and keeps
    those whose sequence tier is not `failed`. A route frame is any row the
    pipeline did not type as parked or off-route, and the partner it verified is
    the row's `candidate_runB_frame`. Reading it back from the CSV rather than
    recomputing it keeps this variant on exactly the frames the shipped method
    checked; the count is asserted against `geometry_checked` by the caller.
    """

    route = [
        row
        for row in rows
        if row["status"] not in (unified.STATUS_IDLE, unified.STATUS_OUT_OF_ROUTE)
    ]
    return [
        (int(row["runA_frame"]), int(row["candidate_runB_frame"]))
        for row in route[:: unified.GEOMETRY_SAMPLE_STRIDE]
        if row["sequence_tier"] != "failed" and row["candidate_runB_frame"]
    ]


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


def band_bounds(height: int) -> tuple[float, float]:
    """(far band upper row, near band lower row) over the rows SIFT may use."""

    usable = height * (1.0 - unified.BOTTOM_EXCLUSION_FRACTION)
    return usable * BAND_FRACTION, usable * (1.0 - BAND_FRACTION)


def theil_sen_slope(rows: np.ndarray, values: np.ndarray) -> float:
    """Median of pairwise slopes of `values` against `rows`, in px per row."""

    if rows.size < 2:
        return float("nan")
    if rows.size > MAX_SLOPE_POINTS:
        generator = np.random.default_rng(SLOPE_SEED)
        keep = generator.choice(rows.size, MAX_SLOPE_POINTS, replace=False)
        rows, values = rows[keep], values[keep]
    left, right = np.triu_indices(rows.size, k=1)
    run = rows[right] - rows[left]
    usable = np.abs(run) > 1e-9
    if not usable.any():
        return float("nan")
    return float(np.median((values[right] - values[left])[usable] / run[usable]))


def measure_matches(
    dx: np.ndarray, dy: np.ndarray, rows: np.ndarray, height: int
) -> dict[str, float] | None:
    """Uniformity of a displacement field with depth, or None if unmeasurable.

    `rows` are the Run A image rows of the matches, standing in for depth.
    """

    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    rows = np.asarray(rows, dtype=float)
    far_top, near_bottom = band_bounds(height)
    far = rows < far_top
    near = rows >= near_bottom
    near_count, far_count = int(near.sum()), int(far.sum())
    if near_count < MIN_BAND_MATCHES or far_count < MIN_BAND_MATCHES:
        return None
    near_dx = float(np.median(dx[near]))
    far_dx = float(np.median(dx[far]))
    return {
        "matches": int(dx.size),
        "near_matches": near_count,
        "far_matches": far_count,
        "near_dx_px": near_dx,
        "far_dx_px": far_dx,
        # Signed, so the manifest can show that the difference sweeps through
        # zero with the candidate offset; the decision uses its magnitude.
        "band_difference_px": near_dx - far_dx,
        "score_px": abs(near_dx - far_dx),
        "slope_px_per_row": theil_sen_slope(rows, dx),
        "dx_px": float(np.median(dx)),
        "dy_px": float(np.median(dy)),
    }


def match_displacements(
    points_a: np.ndarray,
    descriptors_a: np.ndarray,
    points_b: np.ndarray,
    descriptors_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-checked ratio matches, as (dx, dy, Run A row) per match.

    The same correspondence rule `estimate_pose_offset.median_displacement`
    uses, at this variant's ratio.
    """

    from skimage.feature import match_descriptors

    empty = np.zeros(0)
    if (
        len(descriptors_a) < unified.SIFT_MIN_MATCHES_FOR_RANSAC
        or len(descriptors_b) < unified.SIFT_MIN_MATCHES_FOR_RANSAC
    ):
        return empty, empty, empty
    matches = match_descriptors(
        descriptors_a,
        descriptors_b,
        metric="euclidean",
        cross_check=True,
        max_ratio=SIFT_RATIO,
    )
    if len(matches) == 0:
        return empty, empty, empty
    source = points_a[matches[:, 0]]
    target = points_b[matches[:, 1]]
    return target[:, 0] - source[:, 0], target[:, 1] - source[:, 1], source[:, 1]


def _measure_sample(job: tuple) -> dict[str, object]:
    """One Run A sample against every candidate partner. Runs in a worker."""

    frame, partner, image_a, images_b, exclusion_a, exclusion_b = job
    height = image_a.shape[0]
    points_a, descriptors_a = unified.rootsift_features(image_a, exclusion_a)
    candidates: dict[int, dict[str, float] | None] = {}
    for delta in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        image_b = images_b.get(partner + delta)
        if image_b is None:
            candidates[delta] = None
            continue
        points_b, descriptors_b = unified.rootsift_features(image_b, exclusion_b)
        dx, dy, rows = match_displacements(
            points_a, descriptors_a, points_b, descriptors_b
        )
        candidates[delta] = measure_matches(dx, dy, rows, height)
    return {"runA_frame": frame, "baseline_runB_frame": partner, "candidates": candidates}


def choose_candidate(
    candidates: dict[int, dict[str, float] | None]
) -> tuple[int, str]:
    """Smallest uniformity score; ties break toward the baseline partner."""

    measurable = [(delta, m) for delta, m in candidates.items() if m is not None]
    if len(measurable) < MIN_MEASURABLE_CANDIDATES:
        return 0, "kept_baseline_unmeasurable"
    delta, _ = min(measurable, key=lambda item: (item[1]["score_px"], abs(item[0]), item[0]))
    return delta, "chosen"


MIN_CURVE_POINTS = 6
MIN_CURVE_SLOPE_PX_PER_FRAME = 1.0


def resolving_power(spread_px: float, slope_px_per_frame: float, samples: int) -> dict:
    """How many frames of travel the criterion can actually tell apart.

    The scatter of the band difference divided by how much one frame of travel
    moves it. If the per-sample figure is wider than the search window, no
    individual pick means anything and only the aggregate can be read.
    """

    if not np.isfinite(spread_px) or not np.isfinite(slope_px_per_frame) or slope_px_per_frame == 0:
        return {"per_sample": None, "aggregate": None, "search_window": SEARCH_RADIUS}
    per_sample = abs(spread_px / slope_px_per_frame)
    return {
        "per_sample": round(float(per_sample), 2),
        "aggregate": round(float(1.2533 * per_sample / np.sqrt(max(samples, 1))), 2),
        "search_window": SEARCH_RADIUS,
        "note": (
            "scatter of the band difference over its sensitivity to one frame "
            "of travel; a per-sample figure wider than the search window means "
            "the argmin over the window is noise"
        ),
    }


def implied_offset(band_px: float, standard_error_px: float, slope_px_per_frame: float) -> dict:
    """Where the baseline partner sits relative to the depth-uniform position.

    band_difference(d) ~ slope * (d - d_true), so the offset implied by the
    reading at the baseline partner is -band_difference(0) / slope.
    """

    if not np.isfinite(band_px) or not np.isfinite(slope_px_per_frame) or slope_px_per_frame == 0:
        return {"estimate": None, "standard_error": None}
    return {
        "estimate": round(float(-band_px / slope_px_per_frame), 2),
        "standard_error": round(float(abs(standard_error_px / slope_px_per_frame)), 2),
        "note": (
            "negative means the baseline partner is later along Run B than the "
            "depth-uniform position; free of the argmin's selection bias"
        ),
    }


def fit_band_curve(
    candidates: dict[int, dict[str, float] | None]
) -> tuple[float | None, float | None]:
    """(slope, zero crossing) of the *signed* band difference against offset.

    A diagnostic, not the decision. Picking the single smallest score throws
    away ten of the eleven readings, and each one is noisy; the signed
    difference should instead fall through zero as the candidate walks past the
    matching road position, at a rate set by how much faster near content
    sweeps than far. Fitting the whole curve uses every reading, so where the
    per-sample argmin is noise this can still carry a usable aggregate.
    """

    points = sorted((delta, m["band_difference_px"]) for delta, m in candidates.items() if m)
    if len(points) < MIN_CURVE_POINTS:
        return None, None
    offsets = np.array([delta for delta, _ in points], dtype=float)
    values = np.array([value for _, value in points], dtype=float)
    slope, intercept = np.polyfit(offsets, values, 1)
    if abs(slope) < MIN_CURVE_SLOPE_PX_PER_FRAME:
        return float(slope), None
    return float(slope), float(-intercept / slope)


# --------------------------------------------------------------------------
# The dense variant
# --------------------------------------------------------------------------


def interpolate_delta(frames: list[int], deltas: list[int], total: int) -> np.ndarray:
    """Piecewise-linear delta along Run A; zero outside the sampled span."""

    result = np.zeros(total, dtype=float)
    if not frames:
        return result
    return np.interp(
        np.arange(total, dtype=float),
        np.asarray(frames, dtype=float),
        np.asarray(deltas, dtype=float),
        left=0.0,
        right=0.0,
    )


def apply_delta(
    rows: list[dict[str, str]], delta: np.ndarray, total_b: int
) -> list[dict[str, object]]:
    """Shift every mapped partner, then re-impose a non-decreasing Run B.

    The delta varies along the route, so monotonicity has to be restored the
    same way `estimate_pose_offset.corrected_mapping` restores it: by a running
    maximum over Run A order. `parallax_delta_frames` records what was actually
    applied after that clamp, not what was asked for.
    """

    output: list[dict[str, object]] = []
    previous = -1
    for row in rows:
        record: dict[str, object] = dict(row)
        if row["runB_frame"]:
            frame = int(row["runA_frame"])
            base = int(row["runB_frame"])
            shifted = int(round(base + float(delta[frame])))
            shifted = int(min(max(shifted, 0), total_b - 1))
            shifted = max(shifted, previous)
            previous = shifted
            record["runB_frame"] = str(shifted)
            record["parallax_delta_frames"] = shifted - base
        else:
            record["parallax_delta_frames"] = ""
        output.append(record)
    return output


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def measure_camera(camera: str, workers: int) -> list[dict[str, object]]:
    rows = load_baseline(camera)
    samples = geometry_samples(rows)
    checked = sum(1 for row in rows if row["geometry_checked"] == "true")
    if len(samples) != checked:
        raise ValueError(
            f"{camera}: reconstructed {len(samples)} geometry samples but the "
            f"mapping records {checked}; the sampling rule has drifted"
        )
    print(f"[{camera}] {len(samples)} samples, {2 * SEARCH_RADIUS + 1} candidates each")

    exclusion_a = build_reliability_masks.load_mask("runA", camera)["exclusion_224"]
    exclusion_b = build_reliability_masks.load_mask("runB", camera)["exclusion_224"]
    total_b = frame_count("runB", camera)

    wanted_b = sorted(
        {
            min(max(partner + delta, 0), total_b - 1)
            for _, partner in samples
            for delta in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
        }
    )
    print(f"[{camera}] decoding {len(samples)} Run A and {len(wanted_b)} Run B frames")
    images_a = frame_service.frames(
        "runA", camera, [frame for frame, _ in samples], *unified.GEOMETRY_SIZE
    )
    images_b = frame_service.frames("runB", camera, wanted_b, *unified.GEOMETRY_SIZE)

    jobs = [
        (
            frame,
            partner,
            images_a[frame],
            {
                partner + delta: images_b[partner + delta]
                for delta in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
                if partner + delta in images_b
            },
            exclusion_a,
            exclusion_b,
        )
        for frame, partner in samples
    ]

    measured: list[dict[str, object]] = []
    if workers > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for index, result in enumerate(pool.map(_measure_sample, jobs, chunksize=1)):
                measured.append(result)
                if index % 20 == 0:
                    print(f"   {index + 1}/{len(jobs)}", flush=True)
    else:
        for index, job in enumerate(jobs):
            measured.append(_measure_sample(job))
            if index % 20 == 0:
                print(f"   {index + 1}/{len(jobs)}", flush=True)
    return measured


def summarise(measured: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Per-sample records and the camera's aggregate readings."""

    records: list[dict[str, object]] = []
    for entry in measured:
        candidates: dict[int, dict[str, float] | None] = entry["candidates"]
        delta, flag = choose_candidate(candidates)
        chosen = candidates.get(delta)
        baseline = candidates.get(0)
        measurable = sum(1 for m in candidates.values() if m is not None)
        curve_slope, crossing = fit_band_curve(candidates)
        records.append(
            {
                "runA_frame": entry["runA_frame"],
                "baseline_runB_frame": entry["baseline_runB_frame"],
                "chosen_runB_frame": entry["baseline_runB_frame"] + delta,
                "delta_frames": delta,
                "flag": flag,
                "measurable_candidates": measurable,
                "score_at_baseline_px": None if baseline is None else round(baseline["score_px"], 2),
                "score_at_chosen_px": None if chosen is None else round(chosen["score_px"], 2),
                "band_difference_at_baseline_px": (
                    None if baseline is None else round(baseline["band_difference_px"], 2)
                ),
                "slope_at_baseline_px_per_row": (
                    None if baseline is None else round(baseline["slope_px_per_row"], 5)
                ),
                "slope_at_chosen_px_per_row": (
                    None if chosen is None else round(chosen["slope_px_per_row"], 5)
                ),
                "dx_at_baseline_px": None if baseline is None else round(baseline["dx_px"], 2),
                "dx_at_chosen_px": None if chosen is None else round(chosen["dx_px"], 2),
                "dy_at_chosen_px": None if chosen is None else round(chosen["dy_px"], 2),
                "matches_at_chosen": None if chosen is None else chosen["matches"],
                "near_matches_at_chosen": None if chosen is None else chosen["near_matches"],
                "far_matches_at_chosen": None if chosen is None else chosen["far_matches"],
                "curve_slope_px_per_frame": None if curve_slope is None else round(curve_slope, 3),
                "curve_zero_crossing_frames": None if crossing is None else round(crossing, 3),
            }
        )

    def column(key: str) -> np.ndarray:
        return np.array(
            [record[key] for record in records if record[key] is not None], dtype=float
        )

    deltas = np.array([record["delta_frames"] for record in records], dtype=float)
    candidate_total = sum(len(entry["candidates"]) for entry in measured)
    candidate_measurable = sum(
        1 for entry in measured for m in entry["candidates"].values() if m is not None
    )
    dx_chosen, dx_baseline = column("dx_at_chosen_px"), column("dx_at_baseline_px")
    score_chosen, score_baseline = column("score_at_chosen_px"), column("score_at_baseline_px")
    band_baseline = column("band_difference_at_baseline_px")
    slope_baseline = column("slope_at_baseline_px_per_row")
    slope_chosen = column("slope_at_chosen_px_per_row")
    curve_slope = column("curve_slope_px_per_frame")
    crossing = column("curve_zero_crossing_frames")

    def mad(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        return float(1.4826 * np.median(np.abs(values - np.median(values))))

    summary = {
        "samples": len(records),
        "samples_measured": int(sum(1 for r in records if r["flag"] == "chosen")),
        "samples_unmeasurable": int(
            sum(1 for r in records if r["flag"] == "kept_baseline_unmeasurable")
        ),
        "candidates_total": candidate_total,
        "candidates_measurable": candidate_measurable,
        "candidate_measurable_fraction": round(candidate_measurable / max(candidate_total, 1), 4),
        "delta_frames": {
            "median": float(np.median(deltas)),
            "p5": float(np.percentile(deltas, 5)),
            "p95": float(np.percentile(deltas, 95)),
            "mad": round(mad(deltas), 2),
            "fraction_zero": round(float((deltas == 0).mean()), 4),
            "histogram": {
                str(offset): int((deltas == offset).sum())
                for offset in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
            },
        },
        # If the per-sample choice were pure noise the deltas would be spread
        # uniformly over the window and this is the share that would land on
        # the baseline by chance. Read the two together: a delta distribution
        # that matches the uniform share carries no information, however
        # convincingly the score fell.
        "delta_uniform_null": {
            "fraction_zero_if_uniform": round(1.0 / (2 * SEARCH_RADIUS + 1), 4),
            "note": (
                "the score at the chosen candidate is the minimum of up to "
                f"{2 * SEARCH_RADIUS + 1} noisy readings, so it falls whether or "
                "not the criterion carries signal; the delta distribution and "
                "the aggregate band difference below are the tests that do not "
                "share that selection bias"
            ),
        },
        # Not subject to the argmin's selection bias: if the baseline partner
        # sat at the wrong road position, the *signed* near-minus-far difference
        # measured there would be systematically off zero.
        "band_difference_at_baseline_px": {
            "median": round(float(np.median(band_baseline)), 2) if band_baseline.size else None,
            "mad": round(mad(band_baseline), 2) if band_baseline.size else None,
            "standard_error_of_median": (
                round(float(1.2533 * mad(band_baseline) / np.sqrt(band_baseline.size)), 2)
                if band_baseline.size
                else None
            ),
            "samples": int(band_baseline.size),
        },
        "band_difference_curve": {
            "samples_fitted": int(crossing.size),
            "slope_px_per_frame_median": (
                round(float(np.median(curve_slope)), 2) if curve_slope.size else None
            ),
            "zero_crossing_frames_median": (
                round(float(np.median(crossing)), 2) if crossing.size else None
            ),
            "zero_crossing_frames_mad": round(mad(crossing), 2) if crossing.size else None,
            "note": (
                "least squares on the whole signed curve rather than its argmin; "
                "the crossing is where the near and far bands agree, in frames "
                "from the baseline partner"
            ),
        },
        # The band difference is the measurement and the curve slope is its
        # sensitivity to a frame of travel, so their ratio is what the criterion
        # can actually resolve. Compare the per-sample figure with the +/-5
        # search window before believing any individual pick.
        "resolving_power_frames": resolving_power(
            mad(band_baseline) if band_baseline.size else float("nan"),
            float(np.median(curve_slope)) if curve_slope.size else float("nan"),
            band_baseline.size,
        ),
        "implied_position_offset_frames": implied_offset(
            float(np.median(band_baseline)) if band_baseline.size else float("nan"),
            float(1.2533 * mad(band_baseline) / np.sqrt(band_baseline.size))
            if band_baseline.size
            else float("nan"),
            float(np.median(curve_slope)) if curve_slope.size else float("nan"),
        ),
        "slope_px_per_row": {
            "at_baseline_median": (
                round(float(np.median(slope_baseline)), 5) if slope_baseline.size else None
            ),
            "at_chosen_median": (
                round(float(np.median(slope_chosen)), 5) if slope_chosen.size else None
            ),
        },
        "dx_px": {
            "at_baseline_partners": {
                "median": round(float(np.median(dx_baseline)), 2) if dx_baseline.size else None,
                "mad": round(mad(dx_baseline), 2) if dx_baseline.size else None,
            },
            "at_chosen_partners": {
                "median": round(float(np.median(dx_chosen)), 2) if dx_chosen.size else None,
                "mad": round(mad(dx_chosen), 2) if dx_chosen.size else None,
            },
        },
        "dy_px_at_chosen_median": (
            round(float(np.median(column("dy_at_chosen_px"))), 2)
            if column("dy_at_chosen_px").size
            else None
        ),
        "uniformity_score_px": {
            "mean_at_baseline": round(float(score_baseline.mean()), 2) if score_baseline.size else None,
            "mean_at_chosen": round(float(score_chosen.mean()), 2) if score_chosen.size else None,
            "median_at_baseline": round(float(np.median(score_baseline)), 2) if score_baseline.size else None,
            "median_at_chosen": round(float(np.median(score_chosen)), 2) if score_chosen.size else None,
        },
        "matches_at_chosen_median": (
            int(np.median(column("matches_at_chosen"))) if column("matches_at_chosen").size else None
        ),
    }
    return records, summary


SAMPLE_FIELDS = (
    "runA_frame",
    "baseline_runB_frame",
    "chosen_runB_frame",
    "delta_frames",
    "flag",
    "measurable_candidates",
    "score_at_baseline_px",
    "score_at_chosen_px",
    "band_difference_at_baseline_px",
    "slope_at_baseline_px_per_row",
    "slope_at_chosen_px_per_row",
    "dx_at_baseline_px",
    "dx_at_chosen_px",
    "dy_at_chosen_px",
    "matches_at_chosen",
    "near_matches_at_chosen",
    "far_matches_at_chosen",
    "curve_slope_px_per_frame",
    "curve_zero_crossing_frames",
)


def write_samples(camera: str, records: list[dict[str, object]]) -> Path:
    target = OUTPUT_DIR / camera / "parallax_samples.csv"

    def writer(path: Path) -> None:
        with path.open("w", newline="") as handle:
            output = csv.DictWriter(handle, fieldnames=list(SAMPLE_FIELDS))
            output.writeheader()
            for record in records:
                output.writerow({key: record[key] for key in SAMPLE_FIELDS})

    atomic_write(target, writer)
    return target


def build_camera(camera: str, workers: int) -> dict[str, object]:
    measured = measure_camera(camera, workers)
    records, summary = summarise(measured)
    write_samples(camera, records)

    rows = load_baseline(camera)
    total_a = frame_count("runA", camera)
    total_b = frame_count("runB", camera)
    delta = interpolate_delta(
        [int(record["runA_frame"]) for record in records],
        [int(record["delta_frames"]) for record in records],
        total_a,
    )
    shifted = apply_delta(rows, delta, total_b)
    applied = np.array(
        [record["parallax_delta_frames"] for record in shifted if record["parallax_delta_frames"] != ""],
        dtype=float,
    )
    summary["applied_delta_frames"] = {
        "rows": int(applied.size),
        "median": float(np.median(applied)) if applied.size else None,
        "p5": float(np.percentile(applied, 5)) if applied.size else None,
        "p95": float(np.percentile(applied, 95)) if applied.size else None,
        "fraction_zero": round(float((applied == 0).mean()), 4) if applied.size else None,
        "changed_rows": int((applied != 0).sum()),
    }

    target = OUTPUT_DIR / camera / "frame_mapping_unified.csv"
    fieldnames = list(shifted[0])
    atomic_write(target, lambda path: write_csv(path, fieldnames, shifted))
    unified.validate_mapping(target, total_a)
    summary["mapping"] = str(target.relative_to(ROOT))
    return summary


def load_manifest() -> dict[str, object]:
    target = OUTPUT_DIR / "parallax_manifest.json"
    if target.is_file():
        return json.loads(target.read_text())
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=CAMERAS, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="Sample-level processes; 1 runs in this process.",
    )
    arguments = parser.parse_args()
    cameras = tuple(arguments.camera) if arguments.camera else CAMERAS

    for camera in cameras:
        target = OUTPUT_DIR / camera / "frame_mapping_unified.csv"
        if target.is_file() and not arguments.force:
            raise FileExistsError(f"{target} exists; pass --force to rebuild")

    manifest = load_manifest()
    summaries: dict[str, object] = dict(manifest.get("cameras", {}))
    for camera in cameras:
        summaries[camera] = build_camera(camera, max(1, arguments.workers))

    payload = {
        "schema_version": 1,
        "variant": "parallax",
        "variant_version": VARIANT_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "outputs/task2_unified/{camera}/frame_mapping_unified.csv",
        "idea": (
            "A camera rotation shifts near and far scene content by the same "
            "number of pixels; driving further shifts near content more. The "
            "partner whose displacement field is most uniform with image row "
            "(the depth proxy) is the one at the same road position."
        ),
        "parameters": {
            "search_radius_frames": SEARCH_RADIUS,
            "band_fraction_of_usable_rows": round(BAND_FRACTION, 4),
            "min_matches_per_band": MIN_BAND_MATCHES,
            "min_measurable_candidates": MIN_MEASURABLE_CANDIDATES,
            "sift_ratio": SIFT_RATIO,
            "shipped_sift_ratio": unified.SIFT_RATIO_THRESHOLD,
            "sift_ratio_reason": (
                "at the shipped 0.80 only 12% (cam5) and 24% (cam0) of candidates "
                "reach 12 matches in both bands, so the criterion is unmeasurable; "
                "at 0.90 with cross-checking kept it is 100%, and dx against "
                "candidate offset is at least as straight a line (Spearman |0.84| "
                "vs |0.82| cam5, |0.74| vs |0.52| cam0). No label was consulted."
            ),
            "geometry_raster": list(unified.GEOMETRY_SIZE),
            "geometry_sample_stride": unified.GEOMETRY_SAMPLE_STRIDE,
            "bottom_exclusion_fraction": unified.BOTTOM_EXCLUSION_FRACTION,
            "mask_version": build_reliability_masks.mask_version(),
            "colour_id": frame_service.COLOUR_ID,
        },
        "reference_conventions_px": {
            "cam5_matcher_dx": -37.5,
            "cam5_label_dx": -56.0,
            "cam5_matcher_dy": -11.0,
            "cam0_matcher_dx": 0.0,
            "cam0_label_dx": 0.0,
            "source": "outputs/task2_pose_offset/pose_offset_diagnostic.json",
        },
        "caveats": [
            "The lens is a fisheye, so a pure yaw is not a uniform pixel shift; "
            "the criterion's premise holds only to first order and any residual "
            "row-dependence of the rotation term biases the choice.",
            "The per-sample score is noisy; the aggregate over the samples is the "
            "reading worth quoting, not any individual pick.",
            "cam0 is the control: its two conventions already agree at dx = 0, so "
            "its deltas should sit near zero and its dx near zero.",
        ],
        "evidence_status": (
            "Development variant built after the blind labels were spent. It has "
            "no held-out number; score it with evaluate_variants.py, whose "
            "label-free columns are the ones that mean anything here."
        ),
        "cameras": summaries,
    }
    atomic_write(
        OUTPUT_DIR / "parallax_manifest.json",
        lambda path: path.write_text(json.dumps(payload, indent=2)),
    )
    for camera, summary in summaries.items():
        if camera not in cameras:
            continue
        delta = summary["delta_frames"]
        dx = summary["dx_px"]
        score = summary["uniformity_score_px"]
        print(
            f"[{camera}] delta median {delta['median']:+.1f} "
            f"(p5 {delta['p5']:+.1f}, p95 {delta['p95']:+.1f}, "
            f"zero {delta['fraction_zero']:.1%})"
        )
        print(
            f"[{camera}] dx at baseline {dx['at_baseline_partners']['median']} px "
            f"(mad {dx['at_baseline_partners']['mad']}) -> at chosen "
            f"{dx['at_chosen_partners']['median']} px (mad {dx['at_chosen_partners']['mad']})"
        )
        print(
            f"[{camera}] mean uniformity score {score['mean_at_baseline']} -> "
            f"{score['mean_at_chosen']} px"
        )
        band = summary["band_difference_at_baseline_px"]
        curve = summary["band_difference_curve"]
        print(
            f"[{camera}] signed band difference at the baseline partners "
            f"{band['median']} +/- {band['standard_error_of_median']} px "
            f"(uniform-null share of zero deltas "
            f"{summary['delta_uniform_null']['fraction_zero_if_uniform']:.3f})"
        )
        print(
            f"[{camera}] whole-curve zero crossing {curve['zero_crossing_frames_median']} "
            f"frames (mad {curve['zero_crossing_frames_mad']}) at "
            f"{curve['slope_px_per_frame_median']} px per frame"
        )
        power = summary["resolving_power_frames"]
        offset = summary["implied_position_offset_frames"]
        print(
            f"[{camera}] resolving power {power['per_sample']} frames per sample "
            f"(window +/-{power['search_window']}), {power['aggregate']} frames in "
            f"aggregate; implied baseline offset {offset['estimate']} "
            f"+/- {offset['standard_error']} frames"
        )
    print(f"Wrote {(OUTPUT_DIR / 'parallax_manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
