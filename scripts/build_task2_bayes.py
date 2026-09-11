#!/usr/bin/env python3
"""Frame correspondence as posterior inference, with an explicit null state.

The dense matcher this replaces produced a single best path and a categorical
confidence tier that was, by its own manifest, not a probability. That has three
consequences the blind evaluation made concrete. A global monotonic path with
pinned endpoints cannot represent "no counterpart here", so it always answers.
Its confidence measured whether the chosen corridor beat translated copies of
itself, never whether the corridor should exist. And a hand-tuned tier cannot be
checked for calibration, so there is no way to ask whether "high confidence"
means anything.

A hidden Markov model over Run B position fixes all three at once, because each
is a symptom of not having a posterior.

**The null state.** The state space is every Run B route frame in one of two
flavours: *matched*, asserting a correspondence, and *null*, asserting only that
the vehicle has progressed this far. Route order is carried by the position;
whether a correspondence is being claimed is carried by the flavour. "No
counterpart for this stretch" is then an ordinary state sequence rather than
something the architecture forbids.

**Likelihood ratios, not similarities.** Emissions are the evidence a frame
provides *relative to the local background*, estimated per query from the
descriptor's own similarity row by a median and MAD - no labels involved. A
descriptor cosine of 0.85 in a uniform foliage corridor, where a random wrong
frame also scores 0.85, produces a ratio near one and moves the posterior
almost not at all. That is precisely the case the earlier method scored as its
highest confidence tier.

**Weak channels weaken themselves.** The suspension-shake channel enters as a
second likelihood whose strength is measured, not assumed. On this data the
measured ratio is about 2.5 on a handful of usable pairs, so it nudges and no
more - which is the correct treatment of a cue that 10 FPS samples too coarsely
to trust, and it needs no decision from anyone about whether to believe it.

Forward-backward then gives the exact posterior marginal at every frame, which
is strictly more than a best path: it distinguishes one confident answer from
two equally good answers three hundred frames apart, and it makes abstention a
decision about probability mass rather than a threshold on a score.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import build_motion_events
import build_task2_unified as unified
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_bayes"

PIPELINE_VERSION = "bayes-hmm-v1"

# Evidence scale: how many nats of log-likelihood one robust standard deviation
# of descriptor similarity is worth. Higher makes the posterior sharper and the
# model more willing to commit.
EVIDENCE_SCALE = 1.0
MAX_ABS_Z = 8.0                 # clip, so one outlier row cannot dominate

# Null-state dynamics. LEAVE is the per-frame prior odds of entering a stretch
# with no correspondence; RETURN is the prior odds of coming back out.
P_LEAVE = 0.02
P_RETURN = 0.10

# Progress kernel over Run B advance per Run A frame, measured from the route.
DEFAULT_STEP_KERNEL = (0.12, 0.56, 0.24, 0.06, 0.02)

# Motion channel. Any measured cross-run ratio here rests on a handful of
# development pairs - 2 of 3 under the current construction - which is far too
# thin to trust at face value, so the multiplier is capped well below it. The
# ratio of 2.83 that the earlier raw-frame construction reported is withdrawn
# (see build_motion_events.py); the cap is unchanged, because it was never
# derived from that figure, and the channel moves the mapping either way in
# under 3% of frames.
MOTION_LIKELIHOOD_RATIO = 2.5
MOTION_TOLERANCE_FRAMES = 5

# Speed-profile channel, opt-in. Image flow speed is odometry without GPS: a
# stretch driven fast cannot correspond to one driven at walking pace. Bins are
# quantiles of the pooled speeds so "stopped" means the same in both runs. The
# likelihood-ratio table is measured on the unified mapping's accepted pairs
# (label-free) and tempered, because the driver does not repeat speeds exactly:
# traffic lights and other cars decide part of the profile.
SPEED_BINS = 6
SPEED_EVIDENCE_SCALE = 0.5
SPEED_LR_MIN, SPEED_LR_MAX = 0.25, 4.0
SPEED_SMOOTHING = 1.0          # Laplace smoothing count per cell

# Sub-frame channel. DINOv2 locates the place but is too smooth to separate
# adjacent frames, which shows up as good place accuracy and poor exact-frame
# accuracy. A pooled gradient-structure descriptor supplies the last couple of
# frames, applied only in a narrow band around the coarse peak because that is
# the only place it is informative.
STRUCTURE_BAND_FRAMES = 8
STRUCTURE_EVIDENCE_SCALE = 0.8

# Decision thresholds on the posterior. These are probabilities, so they mean
# the same thing on both cameras and can be checked for calibration.
# Chosen on synthetic sequences with a known absent stretch, where this value
# rejects 100% of the gap against 6% outside it. Set on simulated ground truth,
# never on the spent blind labels.
MAX_NULL_POSTERIOR = 0.05
# Adjacent Run B frames are genuinely near-indistinguishable, so the meaningful
# quantity is the mass within a couple of frames of the peak, not on one exact
# frame. This also matches how the acceptable intervals were annotated.
POSTERIOR_WINDOW_FRAMES = 2
MIN_WINDOW_POSTERIOR = 0.30

# Loop-closure coupling. Two Run A frames that are the same place on the way out
# and the way back must map to two Run B frames that are also the same place.
# That is a constraint between distant time steps, which an HMM cannot express
# in its transitions, so it is applied as extra evidence and the chain re-run -
# iterated conditional modes on the loop it would otherwise ignore.
LOOP_CLOSURE_WEIGHT = 1.5
LOOP_CLOSURE_ITERATIONS = 2
LOOP_REVISIT_RANK = 5
LOOP_REVISIT_SIMILARITY = 0.88
LOOP_MIN_SEPARATION = 150
LOOP_SAME_PLACE_TOP_K = 15

STATUS_IDLE = "idle_segment"
STATUS_NO_CORRESPONDENCE = "posterior_no_correspondence"
STATUS_DIFFUSE = "posterior_too_diffuse"
STATUS_ACCEPTED = "accepted_posterior"



def descriptors(run: str, camera: str) -> np.ndarray:
    path = unified.descriptor_cache_path(run, camera)
    if not path.is_file():
        raise FileNotFoundError(
            f"No descriptor cache for {run}/{camera}. Run "
            "scripts/build_task2_unified.py --stage descriptors first."
        )
    return np.load(path)


def evidence_matrix(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Per-query evidence in robust standard deviations above the local background.

    The background is the query's own similarity row: a place that resembles the
    entire route equally provides no evidence about any particular frame, and
    this is what makes that fall out automatically. Median and MAD are used
    because the handful of true matches in a row must not shift the reference.
    """

    similarity = descriptors_a[rows] @ descriptors_b[columns].T
    median = np.median(similarity, axis=1, keepdims=True)
    deviation = np.median(np.abs(similarity - median), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * deviation, 1e-4)
    return np.clip((similarity - median) / scale, -MAX_ABS_Z, MAX_ABS_Z)


def motion_multiplier(
    camera: str,
    rows: np.ndarray,
    columns: np.ndarray,
    ratio: float,
    variant: str = build_motion_events.DEFAULT_STATIC_REMOVAL,
) -> np.ndarray | None:
    """Raise the odds where a shake event fires on both sides of a candidate pair.

    `variant` selects which static-removal build of the event channel to read.
    The default is build_motion_events.DEFAULT_STATIC_REMOVAL - the construction
    whose events satisfy the same-vehicle co-firing constraint - and "none" is
    the withdrawn construction, kept selectable. See run_static_removal_study.py
    for what each one is worth.
    """

    try:
        run_a = build_motion_events.load_signal("runA", camera, variant)
        run_b = build_motion_events.load_signal("runB", camera, variant)
    except FileNotFoundError:
        return None
    if not run_a.events or not run_b.events:
        return None

    def near(frames: np.ndarray, events: list[int]) -> np.ndarray:
        marks = np.zeros(frames.size, dtype=bool)
        for event in events:
            marks |= np.abs(frames - event) <= MOTION_TOLERANCE_FRAMES
        return marks

    row_marks = near(rows, run_a.events)
    column_marks = near(columns, run_b.events)
    multiplier = np.ones((rows.size, columns.size), dtype=np.float64)
    multiplier[np.ix_(row_marks, column_marks)] = ratio
    return multiplier


def speed_multiplier(
    camera: str, rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Odds from how fast the scene sweeps past in each run.

    Returns the multiplier and a record of the table it used, so the manifest
    can show what the channel believed rather than only that it was on.
    """

    try:
        run_a = build_motion_events.load_signal("runA", camera)
        run_b = build_motion_events.load_signal("runB", camera)
    except FileNotFoundError:
        return None
    if run_a.speed is None or run_b.speed is None:
        return None

    def at(speed: np.ndarray, frames: np.ndarray) -> np.ndarray:
        index = np.clip(frames, 0, speed.size - 1)
        return speed[index]

    speed_a = at(run_a.speed, rows)
    speed_b = at(run_b.speed, columns)
    pooled = np.concatenate([speed_a, speed_b])
    edges = np.quantile(pooled, np.linspace(0, 1, SPEED_BINS + 1)[1:-1])
    bins_a = np.searchsorted(edges, speed_a)
    bins_b = np.searchsorted(edges, speed_b)

    # Joint distribution of bins at accepted pairs of the unified mapping.
    mapping_path = ROOT / "outputs" / "task2_unified" / camera / "frame_mapping_unified.csv"
    row_index = {int(frame): position for position, frame in enumerate(rows)}
    column_index = {int(frame): position for position, frame in enumerate(columns)}
    joint = np.full((SPEED_BINS, SPEED_BINS), SPEED_SMOOTHING, dtype=np.float64)
    pairs = 0
    with mapping_path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if not record["runB_frame"] or record["status"] not in unified.ACCEPTED_STATUSES:
                continue
            a = row_index.get(int(record["runA_frame"]))
            b = column_index.get(int(record["runB_frame"]))
            if a is None or b is None:
                continue
            joint[bins_a[a], bins_b[b]] += 1
            pairs += 1
    joint /= joint.sum()
    marginal_a = np.bincount(bins_a, minlength=SPEED_BINS) / bins_a.size
    marginal_b = np.bincount(bins_b, minlength=SPEED_BINS) / bins_b.size
    ratio = joint / np.outer(marginal_a, marginal_b)
    table = np.clip(ratio ** SPEED_EVIDENCE_SCALE, SPEED_LR_MIN, SPEED_LR_MAX)

    multiplier = table[bins_a][:, bins_b]
    record = {
        "bins": SPEED_BINS,
        "bin_edges_px_per_frame": [round(float(e), 3) for e in edges],
        "pairs_used": pairs,
        "likelihood_ratio_table": [[round(float(v), 3) for v in row] for row in table],
        "diagonal_mean": round(float(np.mean(np.diag(table))), 3),
        "off_diagonal_mean": round(float((table.sum() - np.trace(table)) / (SPEED_BINS**2 - SPEED_BINS)), 3),
    }
    return multiplier, record


def structure_multiplier(
    camera: str, rows: np.ndarray, columns: np.ndarray, peak_columns: np.ndarray
) -> np.ndarray:
    """Sharpen the posterior within a narrow band using frame-level structure.

    Computed only inside the band: outside it the descriptor is uninformative
    and including it would add noise rather than evidence.
    """

    wanted_a = [int(f) for f in rows]
    wanted_b = sorted(
        {
            int(np.clip(columns[peak] + delta, columns[0], columns[-1]))
            for peak, delta in (
                (p, d)
                for p in peak_columns
                for d in range(-STRUCTURE_BAND_FRAMES, STRUCTURE_BAND_FRAMES + 1)
            )
        }
    )
    structure_a = unified.structure_descriptors("runA", camera, wanted_a)
    structure_b = unified.structure_descriptors("runB", camera, wanted_b)

    multiplier = np.ones((rows.size, columns.size), dtype=np.float64)
    column_lookup = {int(value): index for index, value in enumerate(columns)}
    for position, frame in enumerate(rows):
        vector_a = structure_a.get(int(frame))
        if vector_a is None:
            continue
        centre = int(columns[peak_columns[position]])
        scores, targets = [], []
        for delta in range(-STRUCTURE_BAND_FRAMES, STRUCTURE_BAND_FRAMES + 1):
            candidate = centre + delta
            index = column_lookup.get(candidate)
            vector_b = structure_b.get(candidate)
            if index is None or vector_b is None:
                continue
            scores.append(float(vector_a @ vector_b))
            targets.append(index)
        if len(scores) < 3:
            continue
        values = np.array(scores)
        standardised = (values - values.mean()) / max(values.std(), 1e-6)
        multiplier[position, targets] = np.exp(STRUCTURE_EVIDENCE_SCALE * standardised)
    return multiplier


def step_kernel_from_path(mapping_path: Path) -> tuple[float, ...]:
    """Measure the Run B advance per Run A frame from an existing alignment."""

    if not mapping_path.is_file():
        return DEFAULT_STEP_KERNEL
    with mapping_path.open(newline="") as handle:
        mapped = [
            (int(row["runA_frame"]), int(row["runB_frame"]))
            for row in csv.DictReader(handle)
            if row["runB_frame"]
        ]
    steps = [
        b_next - b
        for (a, b), (a_next, b_next) in zip(mapped, mapped[1:])
        if a_next == a + 1 and 0 <= b_next - b <= len(DEFAULT_STEP_KERNEL) - 1
    ]
    if len(steps) < 200:
        return DEFAULT_STEP_KERNEL
    counts = np.bincount(steps, minlength=len(DEFAULT_STEP_KERNEL)).astype(float)
    counts += 1.0  # Laplace, so no advance is ever impossible
    return tuple(counts / counts.sum())


def forward_backward(
    emission: np.ndarray, null_emission: np.ndarray, kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward-backward over (matched, null) x Run B position.

    `null_emission` is the likelihood of the observation when no correspondence
    is being claimed. It must be the background expectation for that query, not
    a constant: a matched state only deserves to win where its evidence beats
    what an arbitrary position on the route would have produced anyway.

    Returns the posterior over matched states and the posterior of being in the
    null state, per Run A frame. Scaling each step keeps this in linear space
    without underflow, which matters over 2,600 steps.
    """

    steps, states = emission.shape

    def advance(vector: np.ndarray) -> np.ndarray:
        return np.convolve(vector, kernel)[:states]

    def advance_reverse(vector: np.ndarray) -> np.ndarray:
        return np.convolve(vector[::-1], kernel)[:states][::-1]

    alpha_m = np.zeros((steps, states))
    alpha_n = np.zeros((steps, states))
    alpha_m[0] = emission[0] / max(emission[0].sum(), 1e-300)
    alpha_n[0] = np.full(states, null_emission[0] / states) * P_LEAVE
    total = alpha_m[0].sum() + alpha_n[0].sum()
    alpha_m[0] /= total
    alpha_n[0] /= total

    for index in range(1, steps):
        moved_m = advance(alpha_m[index - 1])
        moved_n = advance(alpha_n[index - 1])
        alpha_m[index] = (moved_m * (1.0 - P_LEAVE) + moved_n * P_RETURN) * emission[index]
        alpha_n[index] = (moved_m * P_LEAVE + moved_n * (1.0 - P_RETURN)) * null_emission[index]
        total = alpha_m[index].sum() + alpha_n[index].sum()
        if total <= 0:
            alpha_m[index] = emission[index] / max(emission[index].sum(), 1e-300)
            alpha_n[index] = 0.0
            total = 1.0
        alpha_m[index] /= total
        alpha_n[index] /= total

    beta_m = np.zeros((steps, states))
    beta_n = np.zeros((steps, states))
    beta_m[-1] = 1.0
    beta_n[-1] = 1.0
    for index in range(steps - 2, -1, -1):
        weighted_m = beta_m[index + 1] * emission[index + 1]
        weighted_n = beta_n[index + 1] * null_emission[index + 1]
        beta_m[index] = advance_reverse(weighted_m * (1.0 - P_LEAVE) + weighted_n * P_LEAVE)
        beta_n[index] = advance_reverse(weighted_m * P_RETURN + weighted_n * (1.0 - P_RETURN))
        total = beta_m[index].max() + beta_n[index].max()
        if total > 0:
            beta_m[index] /= total
            beta_n[index] /= total

    gamma_m = alpha_m * beta_m
    gamma_n = alpha_n * beta_n
    normaliser = gamma_m.sum(axis=1) + gamma_n.sum(axis=1)
    normaliser = np.maximum(normaliser, 1e-300)
    return gamma_m / normaliser[:, None], gamma_n.sum(axis=1) / normaliser


def _mutual_pairs(descriptors_a: np.ndarray, rows: np.ndarray) -> set[tuple[int, int]]:
    """Mutual top-N distant matches within one camera's descriptors."""

    subset = descriptors_a[rows]
    similarity = subset @ subset.T
    separation = np.abs(rows[:, None] - rows[None, :])
    similarity = np.where(separation >= LOOP_MIN_SEPARATION, similarity, -np.inf)

    order = np.argsort(-similarity, axis=1)
    rank = np.empty_like(order)
    grid = np.arange(order.shape[0])[:, None]
    rank[grid, order] = np.arange(order.shape[1])[None, :] + 1

    pairs = np.argwhere(
        (rank <= LOOP_REVISIT_RANK)
        & (rank.T <= LOOP_REVISIT_RANK)
        & (similarity >= LOOP_REVISIT_SIMILARITY)
    )
    return {(int(a), int(b)) for a, b in pairs if a < b}


# A genuine return to the same place shows in both cameras. A foliage corridor
# that merely resembles another one 250 frames on does not - the two cameras
# see disjoint scenes, so the coincidence would have to happen twice. Requiring
# agreement within this many frames removes the false revisits that made up
# most of the single-camera set (64% under 400 frames apart).
LOOP_CROSS_CAMERA_TOLERANCE = 6


def run_a_revisits(
    descriptors_a: np.ndarray, rows: np.ndarray, descriptors_other: np.ndarray | None = None
) -> list[tuple[int, int]]:
    """Row pairs that are the same place on two separate occasions.

    Mutual nearest neighbours in this camera, and - when the other camera's
    descriptors are supplied - a mutual pair in that camera too, within a few
    frames. Only the intersection survives.
    """

    own = _mutual_pairs(descriptors_a, rows)
    if descriptors_other is None:
        return sorted(own)
    other = _mutual_pairs(descriptors_other, rows)
    if not other:
        return []
    # Index the other camera's pairs by first element for tolerant lookup.
    by_first: dict[int, list[int]] = {}
    for a, b in other:
        by_first.setdefault(a, []).append(b)
    confirmed: list[tuple[int, int]] = []
    tol = LOOP_CROSS_CAMERA_TOLERANCE
    for a, b in own:
        hit = any(
            abs(b - b2) <= tol
            for a2 in range(a - tol, a + tol + 1)
            for b2 in by_first.get(a2, ())
        )
        if hit:
            confirmed.append((a, b))
    return sorted(confirmed)


def run_b_same_place(descriptors_b: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Sparse kernel: for each Run B state, which distant states are the same place."""

    subset = descriptors_b[columns]
    similarity = subset @ subset.T
    separation = np.abs(columns[:, None] - columns[None, :])
    similarity = np.where(separation >= LOOP_MIN_SEPARATION, similarity, -np.inf)

    kernel = np.zeros_like(similarity)
    top = np.argsort(-similarity, axis=1)[:, :LOOP_SAME_PLACE_TOP_K]
    grid = np.arange(similarity.shape[0])[:, None]
    scores = similarity[grid, top]
    kernel[grid, top] = np.where(scores >= LOOP_REVISIT_SIMILARITY, scores, 0.0)
    total = kernel.sum(axis=1, keepdims=True)
    return kernel / np.maximum(total, 1e-9)


def apply_loop_closure(
    emission: np.ndarray,
    posterior: np.ndarray,
    pairs: list[tuple[int, int]],
    same_place: np.ndarray,
) -> np.ndarray:
    """Push each occurrence towards places its twin's partner also revisits."""

    updated = emission.copy()
    for first, second in pairs:
        # Where the first occurrence believes it is, carried through Run B's own
        # revisit structure, is where the second occurrence should be looking.
        expectation_second = posterior[first] @ same_place
        expectation_first = posterior[second] @ same_place
        updated[second] *= 1.0 + LOOP_CLOSURE_WEIGHT * expectation_second
        updated[first] *= 1.0 + LOOP_CLOSURE_WEIGHT * expectation_first
    return updated


def window_posterior(posterior: np.ndarray, peak: np.ndarray, radius: int) -> np.ndarray:
    """Posterior mass within `radius` frames of the peak, per Run A frame."""

    steps, states = posterior.shape
    offsets = np.arange(-radius, radius + 1)
    indices = np.clip(peak[:, None] + offsets[None, :], 0, states - 1)
    return np.take_along_axis(posterior, indices, axis=1).sum(axis=1)


@dataclass
class BayesResult:
    camera: str
    rows: np.ndarray
    columns: np.ndarray
    posterior: np.ndarray
    null_posterior: np.ndarray
    speed_table: dict[str, object] | None = None


def run_camera(
    camera: str,
    use_motion: bool,
    force: bool,
    use_structure: bool = True,
    use_loop_closure: bool = True,
    use_speed: bool = False,
    motion_variant: str = build_motion_events.DEFAULT_STATIC_REMOVAL,
) -> BayesResult:
    bounds = unified.route_bounds()
    descriptors_a = descriptors("runA", camera)
    descriptors_b = descriptors("runB", camera)
    run_a_end = min(unified.RUN_A_ROUTE_TAIL_EXCLUSIVE, descriptors_a.shape[0])

    rows = np.arange(bounds["runA_start"], run_a_end)
    columns = np.arange(bounds["runB_start"], descriptors_b.shape[0])

    z_scores = evidence_matrix(descriptors_a, descriptors_b, rows, columns)
    emission = np.exp(EVIDENCE_SCALE * z_scores)

    if use_motion:
        multiplier = motion_multiplier(
            camera, rows, columns, MOTION_LIKELIHOOD_RATIO, motion_variant
        )
        if multiplier is None and motion_variant != build_motion_events.DEFAULT_STATIC_REMOVAL:
            raise FileNotFoundError(
                f"{camera}: motion variant {motion_variant!r} requested but its event "
                "files are missing; run scripts/build_motion_events.py "
                f"--static-removal {motion_variant}"
            )
        if multiplier is not None:
            emission = emission * multiplier

    speed_table = None
    if use_speed:
        speed = speed_multiplier(camera, rows, columns)
        if speed is None:
            raise FileNotFoundError(
                f"{camera}: speed channel requested but no speed profile is cached; "
                "rerun scripts/build_motion_events.py --force"
            )
        multiplier, speed_table = speed
        emission = emission * multiplier
        print(f"   speed channel: {speed_table['pairs_used']} pairs, "
              f"diagonal LR {speed_table['diagonal_mean']}, off-diagonal {speed_table['off_diagonal_mean']}")

    kernel = np.array(
        step_kernel_from_path(
            ROOT / "outputs" / "task2_unified" / camera / "frame_mapping_unified.csv"
        )
    )

    # The background expectation for each query: what an arbitrary route
    # position would have produced. A place that resembles the whole route
    # equally raises this in step with its best candidate, so nothing wins.
    posterior, null_posterior = forward_backward(emission, emission.mean(axis=1), kernel)

    if use_loop_closure:
        other_camera = "cam5" if camera == "cam0" else "cam0"
        try:
            descriptors_other = descriptors("runA", other_camera)
        except FileNotFoundError:
            descriptors_other = None
        pairs = run_a_revisits(descriptors_a, rows, descriptors_other)
        if pairs:
            same_place = run_b_same_place(descriptors_b, columns)
            for _ in range(LOOP_CLOSURE_ITERATIONS):
                emission = apply_loop_closure(emission, posterior, pairs, same_place)
                posterior, null_posterior = forward_backward(
                    emission, emission.mean(axis=1), kernel
                )
            print(f"   loop closure: {len(pairs)} revisit pairs coupled")

    if use_structure:
        # Second pass: the coarse posterior says where to look, then a
        # frame-level descriptor decides which frame inside that band.
        peak = posterior.argmax(axis=1)
        emission = emission * structure_multiplier(camera, rows, columns, peak)
        posterior, null_posterior = forward_backward(
            emission, emission.mean(axis=1), kernel
        )

    result = BayesResult(camera, rows, columns, posterior, null_posterior)
    result.speed_table = speed_table
    return result


def mapping_suffix(
    use_motion: bool,
    use_loop_closure: bool,
    use_speed: bool = False,
    motion_variant: str = build_motion_events.DEFAULT_STATIC_REMOVAL,
) -> str:
    suffix = "" if use_motion else "_no_motion"
    if not use_loop_closure:
        suffix += "_no_loop"
    if use_speed:
        suffix += "_speed"
    if use_motion and motion_variant != build_motion_events.DEFAULT_STATIC_REMOVAL:
        suffix += f"_motion-{motion_variant}"
    return suffix


def write_mapping(
    result: BayesResult, total_a: int, use_motion: bool, use_loop_closure: bool = True,
    use_speed: bool = False,
    motion_variant: str = build_motion_events.DEFAULT_STATIC_REMOVAL,
) -> Path:
    camera_dir = OUTPUT_DIR / result.camera
    best_index = result.posterior.argmax(axis=1)
    window_mass = window_posterior(result.posterior, best_index, POSTERIOR_WINDOW_FRAMES)

    by_frame = {
        int(frame): (int(result.columns[best_index[position]]), float(window_mass[position]),
                     float(result.null_posterior[position]))
        for position, frame in enumerate(result.rows)
    }

    rows: list[dict[str, object]] = []
    for frame in range(total_a):
        record = {
            "camera_id": result.camera,
            "runA_frame": frame,
            "runB_frame": "",
            "confidence": "",
            "posterior_probability": "",
            "posterior_no_correspondence": "",
            "confidence_is_calibrated_probability": "True",
            "status": "",
            "method": PIPELINE_VERSION,
            "motion_channel": str(use_motion).lower(),
        }
        if frame not in by_frame:
            record["status"] = STATUS_IDLE
            record["confidence"] = "abstain"
            rows.append(record)
            continue
        candidate, probability, null_probability = by_frame[frame]
        record["posterior_probability"] = round(probability, 6)
        record["posterior_no_correspondence"] = round(null_probability, 6)
        if null_probability > MAX_NULL_POSTERIOR:
            record["status"] = STATUS_NO_CORRESPONDENCE
            record["confidence"] = "abstain"
        elif probability < MIN_WINDOW_POSTERIOR:
            record["status"] = STATUS_DIFFUSE
            record["confidence"] = "abstain"
        else:
            record["runB_frame"] = candidate
            record["status"] = STATUS_ACCEPTED
            record["confidence"] = round(probability, 6)
        rows.append(record)

    suffix = mapping_suffix(use_motion, use_loop_closure, use_speed, motion_variant)
    target = camera_dir / f"frame_mapping_bayes{suffix}.csv"
    fieldnames = list(rows[0])

    def writer(path: Path) -> None:
        with path.open("w", newline="") as handle:
            output = csv.DictWriter(handle, fieldnames=fieldnames)
            output.writeheader()
            output.writerows(rows)

    atomic_write(target, writer)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("cam0", "cam5"), required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-loop-closure",
        action="store_true",
        help="Ablate the loop-closure coupling.",
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Ablate the sub-frame structure channel.",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Ablate the suspension-shake channel, to show what it is worth.",
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="Add the speed-profile channel (development; separate output file).",
    )
    parser.add_argument(
        "--motion-variant",
        choices=build_motion_events.STATIC_REMOVAL_MODES,
        default=build_motion_events.DEFAULT_STATIC_REMOVAL,
        help=(
            "Which build of the suspension-event channel to read. The default "
            f"({build_motion_events.DEFAULT_STATIC_REMOVAL!r}) writes the plain "
            "frame_mapping_bayes.csv; every other mode, including the withdrawn "
            "'none', requires scripts/build_motion_events.py --static-removal "
            "<mode> and writes to a separate _motion-<mode> output."
        ),
    )
    arguments = parser.parse_args()

    camera = arguments.camera
    total_a = unified.frame_count("runA", camera)
    use_motion = not arguments.no_motion
    motion_variant = arguments.motion_variant

    print(f"[{camera}] posterior inference over Run B position plus a null state")
    if use_motion and motion_variant != build_motion_events.DEFAULT_STATIC_REMOVAL:
        print(f"   motion channel variant: {motion_variant}")
    result = run_camera(
        camera, use_motion, arguments.force,
        not arguments.no_structure, not arguments.no_loop_closure, arguments.speed,
        motion_variant,
    )
    target = write_mapping(
        result, total_a, use_motion, not arguments.no_loop_closure, arguments.speed,
        motion_variant,
    )

    best_index = result.posterior.argmax(axis=1)
    window_mass = window_posterior(result.posterior, best_index, POSTERIOR_WINDOW_FRAMES)
    accepted = int(
        (
            (result.null_posterior <= MAX_NULL_POSTERIOR)
            & (window_mass >= MIN_WINDOW_POSTERIOR)
        ).sum()
    )
    suffix_for_manifest = mapping_suffix(
        use_motion, not arguments.no_loop_closure, arguments.speed, motion_variant
    )
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera,
        "motion_channel_used": use_motion,
        "motion_variant": motion_variant,
        "structure_channel_used": not arguments.no_structure,
        "loop_closure_used": not arguments.no_loop_closure,
        "speed_channel_used": bool(arguments.speed),
        "speed_channel": result.speed_table,
        "loop_closure_weight": LOOP_CLOSURE_WEIGHT,
        "motion_likelihood_ratio": MOTION_LIKELIHOOD_RATIO if use_motion else None,
        "evidence_scale": EVIDENCE_SCALE,
        "p_leave": P_LEAVE,
        "p_return": P_RETURN,
        "step_kernel": list(
            step_kernel_from_path(
                ROOT / "outputs" / "task2_unified" / camera / "frame_mapping_unified.csv"
            )
        ),
        "decision": {
            "max_null_posterior": MAX_NULL_POSTERIOR,
            "min_window_posterior": MIN_WINDOW_POSTERIOR,
            "posterior_window_frames": POSTERIOR_WINDOW_FRAMES,
            "note": "thresholds are on probabilities, so they are comparable across cameras",
        },
        "route_frames": int(result.rows.size),
        "accepted": accepted,
        "coverage_of_route": round(accepted / max(result.rows.size, 1), 6),
        "median_window_posterior": round(float(np.median(window_mass)), 6),
        "median_null_posterior": round(float(np.median(result.null_posterior)), 6),
        "evidence_status": (
            "Development artifact. The blind label set is spent; this method has "
            "not been scored against a held-out set and no accuracy is claimed."
        ),
    }
    atomic_write(
        OUTPUT_DIR / camera / f"bayes_manifest{suffix_for_manifest}.json",
        lambda path: path.write_text(json.dumps(manifest, indent=2)),
    )

    print(f"   route frames {result.rows.size}, accepted {accepted} ({accepted / result.rows.size:.2%})")
    print(f"   median posterior mass within +/-{POSTERIOR_WINDOW_FRAMES} frames "
          f"{manifest['median_window_posterior']:.4f}")
    print(f"   median P(no correspondence) {manifest['median_null_posterior']:.4f}")
    print(f"   wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
