#!/usr/bin/env python3
"""One dense frame-correspondence pipeline, replacing the v1-to-v5 chain.

The earlier work grew as five layers that were copied forward rather than
composed: v5 duplicates about 140 lines of the v1 matcher and silently drops the
reliability mask from its RootSIFT stage, and the dense branches hardcode the
mask mode as a string literal in two places. This module carries the same ideas
in one pass, and fixes the three defects the data audit found.

What changed, and why:

* Frames come from `frame_service`, which decodes the HEVC with a pinned colour
  conversion. Every earlier stage read 640x480 CRF-24 proxies. Geometry now runs
  at 896x672 instead of 448x336, which on the worst stream (run B cam5) raises
  usable SIFT keypoints from 669 to 2,494.
* The reliability mask is measured rather than predicted. It excludes the
  vignette, the ego wing mirror, and run B's static rain droplets, none of which
  an ADE20K segmenter has a class for.
* The sequence search is global. Dense v2 banded the search to +/-128 frames
  around a prior, but the interior aliases measured on this route sit 300 to
  1,300 frames away, so a local band cannot reject them.

Decisions are per frame and every refusal is typed. `runB_frame` may be empty;
`status` may not. In particular `loop_ambiguous` is kept distinct from
`low_evidence`, because the route returns to its start and in the closure region
the ground truth itself is ambiguous - that is a property of the data, not a
failure of the matcher.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import build_reliability_masks
import frame_service
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_unified"
BOUNDARY_FILE = ROOT / "outputs" / "task2_keyframes" / "route_phase_boundaries.csv"

DESCRIPTOR_VERSION = "dinov2s-masked-v1"
PIPELINE_VERSION = "unified-v1"

# Development variants. Both default to the shipped behaviour and are set only
# from the command line. The freeze verifier reads the module constants, so a
# variant cannot masquerade as the submitted method, and variant outputs go to
# their own directory so the frozen mapping is never overwritten.
# Stage-3 refinement rule. "ratchet" is the shipped one: a hard non-decreasing
# floor and no slope prior. "slope" is the development fix; see refine_sequence.
REFINEMENT_DEFAULT = "ratchet"
REFINEMENT_MODES = ("ratchet", "slope")
# Descriptor aggregation. "mean" is the shipped masked mean of patch tokens;
# "vlad" is the development variant in vlad_aggregation.py.
AGGREGATION_MODES = ("mean", "vlad")


@dataclass(frozen=True)
class VariantConfig:
    """Which development variant, if any, this process is building.

    The defaults are the shipped method. A variant is selected once, from the
    command line, through `configure_variant`; every stage reads this object,
    so there is exactly one place where "which variant" is decided.
    """

    mask_variant: str = ""
    greyscale: bool = False
    refinement: str = REFINEMENT_DEFAULT
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if self.mask_variant not in build_reliability_masks.KNOWN_VARIANTS:
            raise ValueError(f"unknown mask variant {self.mask_variant!r}")
        if self.refinement not in REFINEMENT_MODES:
            raise ValueError(f"unknown refinement {self.refinement!r}")
        if self.aggregation not in AGGREGATION_MODES:
            raise ValueError(f"unknown aggregation {self.aggregation!r}")


VARIANT = VariantConfig()


def configure_variant(**overrides: object) -> VariantConfig:
    """Replace the active variant; returns the previous one so callers can restore it."""

    global VARIANT
    previous = VARIANT
    VARIANT = VariantConfig(**overrides)
    return previous


def set_variant(config: VariantConfig) -> None:
    global VARIANT
    VARIANT = config
VARIANT_OUTPUT_DIR = ROOT / "outputs" / "task2_unified_variants"


def variant_name() -> str:
    refinement = VARIANT.refinement if VARIANT.refinement != REFINEMENT_DEFAULT else ""
    parts = [
        part
        for part in (VARIANT.mask_variant, "grey" if VARIANT.greyscale else "",
                     VARIANT.aggregation if VARIANT.aggregation != "mean" else "", refinement)
        if part
    ]
    return "+".join(parts)


def descriptor_version() -> str:
    version = DESCRIPTOR_VERSION + ("-grey" if VARIANT.greyscale else "")
    if VARIANT.aggregation == "vlad":
        import vlad_aggregation

        version += f"-vlad{vlad_aggregation.VOCABULARY_SIZE}"
    return version


def output_dir(camera: str) -> Path:
    name = variant_name()
    return (VARIANT_OUTPUT_DIR / name / camera) if name else (OUTPUT_DIR / camera)

DINO_DIR = ROOT / "outputs" / "task2_models" / "dinov2-small"
MODEL_SIZE = 224
PATCH_GRID = 16
DESCRIPTOR_BATCH = 16

# Run A is parked for its first 173 frames and its last 57. Those frames are all
# the same place, so a one-to-one correspondence there is meaningless.
RUN_A_ROUTE_TAIL_EXCLUSIVE = frame_service.RUN_A_ROUTE_TAIL_EXCLUSIVE

COARSE_STRIDE = 4

# Measured warp constraints. The implied Run A to Run B offset drifts across a
# 124-frame span but does so smoothly, at most about 0.5 frames per Run A frame.
MIN_SLOPE = 0.2
MAX_SLOPE = 2.0
DTW_STEPS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2))
SLOPE_PENALTY_WEIGHT = 0.035
SKIP_PENALTY_WEIGHT = 0.008

# Full-cadence refinement around the coarse path.
REFINE_RADIUS = 8
STRUCTURE_GRID = (12, 16)

# Parameters of the "slope" refinement only. They are inert under the shipped
# default and are not part of the method freeze, which describes `ratchet`.
# REFINE_SLOPE_WEIGHT converts one frame of departure from the coarse path's
# local slope into structure-score units; the structure score is a cosine in
# [-1, 1], so 0.05 makes a one-frame surprise worth about a 0.05 cosine drop -
# enough to carry the sequence through a stretch where the score is flat, not
# enough to override a descriptor that actually discriminates.
REFINE_SLOPE_WEIGHT = 0.05
# Hard cap on the Run B increment per Run A frame, in the same units as the
# coarse stage's largest single step (1, 3). Monotonicity is the lower bound.
REFINE_MAX_STEP = 3
REFINE_MIN_STEP = 0

# Multi-scale sequence uniqueness, in coarse nodes. At stride 4 and 10 fps these
# are roughly 1.6, 4, 8 and 16 second windows.
WINDOW_RADII = (2, 5, 10, 20)
ALTERNATIVE_SEPARATION = 48

# Sparse geometry verification.
GEOMETRY_SIZE = frame_service.GEOMETRY_SIZE
GEOMETRY_SAMPLE_STRIDE = 12
SIFT_RATIO_THRESHOLD = 0.80
SIFT_MIN_MATCHES_FOR_RANSAC = 8
GEOMETRY_MIN_MATCHES = 20
GEOMETRY_MIN_INLIERS = 12
GEOMETRY_MIN_INLIER_RATIO = 0.25
GEOMETRY_GRID = (3, 4)
GEOMETRY_MIN_CELLS = 3
GEOMETRY_MIN_ROWS = 2
GEOMETRY_MIN_COLUMNS = 2
GEOMETRY_MIN_HORIZONTAL_SPREAD = 0.20
GEOMETRY_MIN_VERTICAL_SPREAD = 0.18
BOTTOM_EXCLUSION_FRACTION = 0.125

# Acceptance tiers.
STRONG_SEQUENCE_SCALES = 4
SUPPORTED_SEQUENCE_SCALES = 3
MAX_BRIDGED_GAP = 10

STATUS_IDLE = "idle_segment"
STATUS_OUT_OF_ROUTE = "out_of_route"
STATUS_LOOP_AMBIGUOUS = "loop_ambiguous"
STATUS_LOW_EVIDENCE = "low_evidence"
STATUS_SEQUENCE_FAILED = "sequence_failed"
STATUS_GEOMETRY_FAILED = "geometry_failed"
STATUS_ACCEPTED_STRONG = "accepted_strong"
STATUS_ACCEPTED_SUPPORTED = "accepted_sequence_supported"
STATUS_ACCEPTED_BRIDGED = "accepted_bridged_short_gap"
# Geometry samples one frame in twelve and marks the span between two passing
# checks as supported, so a frame can reach the top tier having never been
# verified. Under --strict-geometry only the frame actually checked keeps that
# tier and the span it vouches for is demoted, which is what the tier was
# supposed to mean.
STATUS_ACCEPTED_PROPAGATED = "accepted_geometry_propagated"

ACCEPTED_STATUSES = frozenset(
    (
        STATUS_ACCEPTED_STRONG,
        STATUS_ACCEPTED_SUPPORTED,
        STATUS_ACCEPTED_BRIDGED,
        STATUS_ACCEPTED_PROPAGATED,
    )
)


def route_bounds() -> dict[str, int]:
    with BOUNDARY_FILE.open(newline="") as handle:
        rows = {row["run"]: row for row in csv.DictReader(handle)}
    return {
        "runA_start": int(rows["runA"]["route_start"]),
        "runB_start": int(rows["runB"]["route_start"]),
    }


def frame_count(run: str, camera: str) -> int:
    return frame_service.hevc_bitstream.scan(
        frame_service.source_path(run, camera)
    ).picture_count



# --------------------------------------------------------------------------
# Stage 1: masked DINOv2 descriptors
# --------------------------------------------------------------------------


def descriptor_cache_path(run: str, camera: str) -> Path:
    digest = frame_service.file_sha256(frame_service.source_path(run, camera))
    key = frame_service.cache_key(
        run, camera, *frame_service.DESCRIPTOR_SIZE, digest
    )
    version = build_reliability_masks.mask_version(VARIANT.mask_variant)
    return OUTPUT_DIR / camera / f"descriptors_{key}_mask{version}_{descriptor_version()}.npy"


def compute_descriptors(run: str, camera: str, force: bool = False) -> np.ndarray:
    """Mask-pooled DINOv2-S descriptors for every frame, cached on disk.

    The cache key carries the source hash, resolution, colour identity, mask
    version and descriptor version, so a change to any of them produces a
    different file rather than silently reusing stale arrays.
    """

    target = descriptor_cache_path(run, camera)
    if target.is_file() and not force:
        return np.load(target)
    if VARIANT.aggregation == "vlad":
        import vlad_aggregation

        return vlad_aggregation.descriptors(run, camera, target, force)

    import torch
    from transformers import AutoImageProcessor, Dinov2Model

    torch.set_num_threads(1)
    processor = AutoImageProcessor.from_pretrained(str(DINO_DIR), local_files_only=True)
    model = (
        Dinov2Model.from_pretrained(
            str(DINO_DIR), local_files_only=True, use_safetensors=True
        )
        .eval()
    )

    mask = build_reliability_masks.load_mask(run, camera, VARIANT.mask_variant)
    patch_valid = torch.tensor(mask["patch_valid"].reshape(-1).copy(), dtype=torch.bool)
    if not bool(patch_valid.any()):
        raise ValueError(f"{run}/{camera}: reliability mask leaves no valid patch")

    total = frame_count(run, camera)
    descriptors = np.zeros((total, model.config.hidden_size), dtype=np.float32)

    batch_images: list[Image.Image] = []
    batch_ordinals: list[int] = []

    def flush() -> None:
        if not batch_images:
            return
        with torch.inference_mode():
            inputs = processor(
                images=batch_images,
                return_tensors="pt",
                do_resize=True,
                size={"height": MODEL_SIZE, "width": MODEL_SIZE},
                do_center_crop=False,
            )
            hidden = model(**inputs).last_hidden_state[:, 1:, :]
            tokens = torch.nn.functional.normalize(hidden, dim=-1)
            pooled = tokens[:, patch_valid, :].mean(dim=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
        for index, ordinal in enumerate(batch_ordinals):
            descriptors[ordinal] = pooled[index].numpy()
        batch_images.clear()
        batch_ordinals.clear()

    source = frame_service.source_path(run, camera)
    for ordinal, image in frame_service.iter_frames(
        source, *frame_service.DESCRIPTOR_SIZE
    ):
        picture = Image.fromarray(image)
        if VARIANT.greyscale:
            # Luminance only, replicated to three channels so the model sees
            # its expected input shape. Season and exposure differ between the
            # runs; colour is the part of the appearance that differs most.
            picture = picture.convert("L").convert("RGB")
        batch_images.append(picture)
        batch_ordinals.append(ordinal)
        if len(batch_images) == DESCRIPTOR_BATCH:
            flush()
            if ordinal % 400 < DESCRIPTOR_BATCH:
                print(f"   {run}/{camera}: {ordinal + 1}/{total}", flush=True)
    flush()

    atomic_write(target, lambda path: np.save(path, descriptors))
    return descriptors


# --------------------------------------------------------------------------
# Stage 2: global monotonic path
# --------------------------------------------------------------------------


def global_monotonic_path(
    similarity: np.ndarray, expected_slope: float
) -> list[tuple[int, int]]:
    """Dynamic time warp over the whole route, with a slope prior.

    No prior band. The interior aliases on this route sit 300 to 1,300 frames
    away from their look-alike, so only a globally consistent path can reject
    them; a window around a prior cannot see far enough.
    """

    rows, columns = similarity.shape
    cost = 1.0 - similarity
    accumulated = np.full((rows, columns), np.inf, dtype=np.float64)
    back_row = np.full((rows, columns), -1, dtype=np.int32)
    back_column = np.full((rows, columns), -1, dtype=np.int32)

    step_penalty = []
    for row_step, column_step in DTW_STEPS:
        slope = column_step / row_step
        penalty = SLOPE_PENALTY_WEIGHT * abs(math.log(slope / expected_slope))
        penalty += SKIP_PENALTY_WEIGHT * (row_step + column_step - 2)
        step_penalty.append(penalty)

    accumulated[0, 0] = cost[0, 0]
    column_index = np.arange(columns)
    for row in range(1, rows):
        for step_index, (row_step, column_step) in enumerate(DTW_STEPS):
            source_row = row - row_step
            if source_row < 0:
                continue
            previous = accumulated[source_row]
            if not np.isfinite(previous).any():
                continue
            candidate = np.full(columns, np.inf)
            candidate[column_step:] = previous[:-column_step]
            # Charge the unary cost once per Run A row the step consumes.
            # Without this a step that skips rows pays fewer unary terms than
            # one that does not, and the optimiser buys a cheaper path by
            # stepping over half the route.
            candidate = candidate + step_penalty[step_index] + row_step * cost[row]
            improved = candidate < accumulated[row]
            if improved.any():
                accumulated[row] = np.where(improved, candidate, accumulated[row])
                back_row[row] = np.where(improved, source_row, back_row[row])
                back_column[row] = np.where(
                    improved, column_index - column_step, back_column[row]
                )

    row, column = rows - 1, int(np.argmin(accumulated[rows - 1]))
    path: list[tuple[int, int]] = []
    while row >= 0 and column >= 0:
        path.append((row, column))
        previous_row = int(back_row[row, column])
        previous_column = int(back_column[row, column])
        if previous_row < 0 or previous_column < 0:
            break
        row, column = previous_row, previous_column
    return path[::-1]


def densify_path(
    path: list[tuple[int, int]], run_a_frames: np.ndarray, run_b_frames: np.ndarray, total_a: int
) -> np.ndarray:
    """Interpolate the coarse path to a Run B estimate for every Run A frame."""

    node_a = np.array([run_a_frames[row] for row, _ in path], dtype=np.float64)
    node_b = np.array([run_b_frames[column] for _, column in path], dtype=np.float64)
    estimate = np.full(total_a, -1, dtype=np.int64)
    covered = np.arange(int(node_a[0]), int(node_a[-1]) + 1)
    estimate[covered] = np.round(np.interp(covered, node_a, node_b)).astype(np.int64)
    return estimate


# --------------------------------------------------------------------------
# Stage 3: full-cadence structure refinement
# --------------------------------------------------------------------------


def structure_descriptors(run: str, camera: str, ordinals: list[int]) -> dict[int, np.ndarray]:
    """Brightness-standardised pooled gradient structure, for local phase.

    DINOv2 locates the place; it is too smooth to separate adjacent frames. This
    cheap descriptor does the last few frames of alignment and is normalised per
    frame so run B's 2.2x darker exposure does not dominate.
    """

    rows, columns = STRUCTURE_GRID
    output: dict[int, np.ndarray] = {}
    wanted = set(ordinals)
    for ordinal, image in frame_service.iter_frames(
        frame_service.source_path(run, camera), *frame_service.DESCRIPTOR_SIZE
    ):
        if ordinal not in wanted:
            continue
        grey = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32)
        gradient_y, gradient_x = np.gradient(grey)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        height, width = magnitude.shape
        row_edges = np.linspace(0, height, rows + 1).astype(int)
        column_edges = np.linspace(0, width, columns + 1).astype(int)
        cells = []
        for r in range(rows):
            for c in range(columns):
                block = magnitude[row_edges[r] : row_edges[r + 1], column_edges[c] : column_edges[c + 1]]
                cells.append(block.mean())
        vector = np.array(cells, dtype=np.float32)
        vector -= vector.mean()
        norm = float(np.linalg.norm(vector))
        output[ordinal] = vector / norm if norm > 1e-6 else vector
        if len(output) == len(wanted):
            break
    return output


def refine_sequence(
    route_frames: list[int],
    estimate: np.ndarray,
    structure_a: dict[int, np.ndarray],
    structure_b: dict[int, np.ndarray],
    total_b: int,
    mode: str | None = None,
) -> dict[int, int]:
    """Pick a Run B frame for every Run A route frame from the structure score.

    Two rules, selected by `mode`; the shipped one is `ratchet`.

    `ratchet` scans the +/-REFINE_RADIUS window per frame and takes the best
    admissible candidate, where admissible means "not behind the previous
    pick". It is greedy and has no slope prior, so where the structure score
    collapses - a dark or featureless stretch - its argmax can land at the far
    edge of the window, and the floor then makes that jump permanent: every
    later frame inherits it. That is the mechanism behind the mapping kinks
    documented in outputs/task2_kinks/.

    `slope` replaces the greedy scan with a Viterbi over the same candidate
    windows: emission is the same structure score, the transition cost is a
    penalty on departing from the coarse path's own local slope, and
    monotonicity is a hard transition constraint (Run B increment in
    [REFINE_MIN_STEP, REFINE_MAX_STEP]) rather than a floor carried forward.
    Because the whole sequence is optimised at once, a frame may sit behind
    where a greedy pass would have put it without any need for a backtrack
    allowance, and no single collapsed frame can ratchet the rest of the route.
    The output is non-decreasing by construction.
    """

    mode = VARIANT.refinement if mode is None else mode
    if mode == "ratchet":
        return _refine_ratchet(route_frames, estimate, structure_a, structure_b, total_b)
    if mode == "slope":
        return _refine_slope(route_frames, estimate, structure_a, structure_b, total_b)
    raise ValueError(f"unknown refinement mode {mode!r}; expected one of {REFINEMENT_MODES}")


def _refine_ratchet(
    route_frames: list[int],
    estimate: np.ndarray,
    structure_a: dict[int, np.ndarray],
    structure_b: dict[int, np.ndarray],
    total_b: int,
) -> dict[int, int]:
    """The shipped rule, unchanged: greedy argmax behind a non-decreasing floor."""

    picked: dict[int, int] = {}
    previous = -1
    for frame in route_frames:
        centre = int(estimate[frame])
        # Non-decreasing, not strictly increasing: when Run A moves slower than
        # Run B, several Run A frames legitimately share one Run B frame.
        best_score, best_frame = -2.0, max(centre, previous)
        for delta in range(-REFINE_RADIUS, REFINE_RADIUS + 1):
            candidate = int(np.clip(centre + delta, 0, total_b - 1))
            if candidate < previous:
                continue
            vector_a = structure_a.get(frame)
            vector_b = structure_b.get(candidate)
            if vector_a is None or vector_b is None:
                continue
            score = float(vector_a @ vector_b)
            if score > best_score:
                best_score, best_frame = score, candidate
        picked[frame] = best_frame
        previous = best_frame
    return picked


def _refine_slope(
    route_frames: list[int],
    estimate: np.ndarray,
    structure_a: dict[int, np.ndarray],
    structure_b: dict[int, np.ndarray],
    total_b: int,
) -> dict[int, int]:
    """Viterbi over the refinement windows: structure score against a slope prior.

    States at Run A frame `f` are the Run B candidates `estimate[f] + d` for
    `d` in [-REFINE_RADIUS, REFINE_RADIUS], clipped to the stream. Emission is
    the structure cosine. The transition from candidate `p` to candidate `c` is
    admissible when `c - p` lies in [REFINE_MIN_STEP, REFINE_MAX_STEP] times the
    Run A gap, and costs REFINE_SLOPE_WEIGHT per frame of departure from the
    coarse path's own increment over that gap. A frame whose descriptors are
    missing contributes a flat emission, so the prior alone carries it, which is
    the behaviour the ratchet lacks.
    """

    if not route_frames:
        return {}

    deltas = np.arange(-REFINE_RADIUS, REFINE_RADIUS + 1)

    def window(frame: int) -> np.ndarray:
        return np.clip(int(estimate[frame]) + deltas, 0, total_b - 1)

    def emissions(frame: int, states: np.ndarray) -> np.ndarray:
        vector_a = structure_a.get(frame)
        if vector_a is None:
            return np.zeros(states.size)
        values = np.zeros(states.size)
        for index, candidate in enumerate(states.tolist()):
            vector_b = structure_b.get(int(candidate))
            values[index] = 0.0 if vector_b is None else float(vector_a @ vector_b)
        return values

    first = route_frames[0]
    previous_frame = first
    previous_states = window(first)
    total = emissions(first, previous_states)
    trellis: list[tuple[np.ndarray, np.ndarray]] = []  # per step: states, backpointers

    for frame in route_frames[1:]:
        states = window(frame)
        gap = frame - previous_frame
        expected = float(estimate[frame] - estimate[previous_frame])
        step = states[None, :] - previous_states[:, None]
        admissible = (step >= REFINE_MIN_STEP * gap) & (step <= REFINE_MAX_STEP * gap)
        transition = np.where(
            admissible,
            total[:, None] - REFINE_SLOPE_WEIGHT * np.abs(step - expected),
            -np.inf,
        )
        if not np.isfinite(transition).any():
            # The coarse path moved further between two Run A frames than any
            # admissible step can cover. Restart the chain from the best state
            # so far, keeping only candidates that leave the mapping
            # non-decreasing; if there are none, take the furthest candidate
            # there is. This does not fire on either camera.
            best_previous = int(np.argmax(total))
            floor = int(previous_states[best_previous])
            reachable = states >= floor
            if not reachable.any():
                reachable = states == states.max()
            transition = np.full(step.shape, -np.inf)
            transition[best_previous, reachable] = total[best_previous]
        backpointers = np.argmax(transition, axis=0)
        total = transition[backpointers, np.arange(states.size)] + emissions(frame, states)
        trellis.append((states, backpointers))
        previous_frame, previous_states = frame, states

    index = int(np.argmax(total))
    picked: dict[int, int] = {}
    for frame, (states, backpointers) in zip(reversed(route_frames[1:]), reversed(trellis)):
        picked[frame] = int(states[index])
        index = int(backpointers[index])
    picked[first] = int(window(first)[index])
    ordered = [picked[frame] for frame in route_frames]
    if any(later < earlier for earlier, later in zip(ordered, ordered[1:])):
        raise ValueError("slope refinement produced a non-monotonic sequence")
    return dict(zip(route_frames, ordered))


# --------------------------------------------------------------------------
# Stage 4: multi-scale sequence uniqueness
# --------------------------------------------------------------------------


def sequence_margins(
    similarity: np.ndarray, path: list[tuple[int, int]]
) -> dict[int, list[float]]:
    """Ask whether the chosen corridor beats translated copies of itself.

    A single frame in dense foliage is not unique; a stretch of route usually is.
    Each scale slides the selected path shape to every distant Run B centre and
    reports how much the chosen position wins by. cam5 carries five times cam0's
    density of look-alike places, which is why this matters more there.
    """

    rows = np.array([row for row, _ in path])
    columns = np.array([column for _, column in path])
    total_columns = similarity.shape[1]
    margins: dict[int, list[float]] = {}

    for index in range(len(path)):
        per_scale: list[float] = []
        for radius in WINDOW_RADII:
            low = max(0, index - radius)
            high = min(len(path) - 1, index + radius)
            window_rows = rows[low : high + 1]
            window_columns = columns[low : high + 1]
            offsets = window_columns - columns[index]
            chosen = float(similarity[window_rows, window_columns].mean())

            centres = np.arange(total_columns)
            distant = centres[np.abs(centres - columns[index]) >= ALTERNATIVE_SEPARATION]
            if distant.size == 0:
                per_scale.append(float("inf"))
                continue
            candidate_columns = distant[:, None] + offsets[None, :]
            valid = (candidate_columns >= 0) & (candidate_columns < total_columns)
            candidate_columns = np.clip(candidate_columns, 0, total_columns - 1)
            scores = similarity[window_rows[None, :], candidate_columns]
            scores = np.where(valid, scores, -1.0)
            best_alternative = float(scores.mean(axis=1).max())
            per_scale.append(chosen - best_alternative)
        margins[index] = per_scale
    return margins


# --------------------------------------------------------------------------
# Stage 5: sparse geometry at the larger raster
# --------------------------------------------------------------------------


def rootsift_features(image: np.ndarray, exclusion: np.ndarray | None):
    from skimage.exposure import equalize_adapthist
    from skimage.feature import SIFT

    grey = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float64) / 255.0
    grey = equalize_adapthist(grey, clip_limit=0.02)
    detector = SIFT(upsampling=1, n_octaves=5, n_scales=3, c_dog=0.004, c_edge=15)
    try:
        detector.detect_and_extract(grey)
    except RuntimeError:
        return np.zeros((0, 2)), np.zeros((0, 128))
    points = detector.keypoints[:, ::-1].astype(np.float64)
    descriptors = detector.descriptors.astype(np.float64)

    height, width = grey.shape
    keep = points[:, 1] < height * (1.0 - BOTTOM_EXCLUSION_FRACTION)
    if exclusion is not None and exclusion.size:
        mask_y = np.clip(
            (points[:, 1] * exclusion.shape[0] / height).astype(int), 0, exclusion.shape[0] - 1
        )
        mask_x = np.clip(
            (points[:, 0] * exclusion.shape[1] / width).astype(int), 0, exclusion.shape[1] - 1
        )
        keep &= ~exclusion[mask_y, mask_x]
    points, descriptors = points[keep], descriptors[keep]
    if descriptors.size:
        descriptors /= np.clip(descriptors.sum(axis=1, keepdims=True), 1e-8, None)
        descriptors = np.sqrt(descriptors)
        descriptors /= np.clip(
            np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8, None
        )
    return points, descriptors


def spatial_support(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    if points.size == 0:
        return {"cells": 0, "rows": 0, "columns": 0, "horizontal_spread": 0.0, "vertical_spread": 0.0}
    grid_rows, grid_columns = GEOMETRY_GRID
    row_index = np.clip((points[:, 1] / height * grid_rows).astype(int), 0, grid_rows - 1)
    column_index = np.clip((points[:, 0] / width * grid_columns).astype(int), 0, grid_columns - 1)
    occupied = set(zip(row_index.tolist(), column_index.tolist()))
    horizontal = float(
        (np.percentile(points[:, 0], 90) - np.percentile(points[:, 0], 10)) / width
    )
    vertical = float(
        (np.percentile(points[:, 1], 90) - np.percentile(points[:, 1], 10)) / height
    )
    return {
        "cells": len(occupied),
        "rows": len({row for row, _ in occupied}),
        "columns": len({column for _, column in occupied}),
        "horizontal_spread": horizontal,
        "vertical_spread": vertical,
    }


def verify_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    exclusion_a: np.ndarray,
    exclusion_b: np.ndarray,
) -> dict[str, object]:
    """Fit epipolar and homography models and require distributed support.

    Requiring the inliers to spread across the frame is what rejects a repeated
    awning edge or a run of identical railings, which can otherwise produce many
    locally consistent matches at completely the wrong place.
    """

    from skimage.feature import match_descriptors
    from skimage.measure import ransac
    from skimage.transform import FundamentalMatrixTransform, ProjectiveTransform

    points_a, descriptors_a = rootsift_features(image_a, exclusion_a)
    points_b, descriptors_b = rootsift_features(image_b, exclusion_b)
    result: dict[str, object] = {
        "keypoints_a": int(len(points_a)),
        "keypoints_b": int(len(points_b)),
        "matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "supportive": False,
    }
    if len(descriptors_a) < SIFT_MIN_MATCHES_FOR_RANSAC or len(descriptors_b) < SIFT_MIN_MATCHES_FOR_RANSAC:
        return result

    matches = match_descriptors(
        descriptors_a, descriptors_b, metric="euclidean",
        cross_check=True, max_ratio=SIFT_RATIO_THRESHOLD,
    )
    result["matches"] = int(len(matches))
    if len(matches) < SIFT_MIN_MATCHES_FOR_RANSAC:
        return result

    source = points_a[matches[:, 0]]
    target = points_b[matches[:, 1]]
    best: tuple[int, np.ndarray] | None = None
    for model_class, minimum, residual in (
        (FundamentalMatrixTransform, 8, 1.5),
        (ProjectiveTransform, 4, 3.0),
    ):
        if len(matches) < minimum:
            continue
        try:
            _, inliers = ransac(
                (source, target), model_class, min_samples=minimum,
                residual_threshold=residual, max_trials=750, random_state=42,
            )
        except Exception:
            continue
        if inliers is None:
            continue
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, inliers)
    if best is None:
        return result

    count, inliers = best
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    support_a = spatial_support(source[inliers], width_a, height_a)
    support_b = spatial_support(target[inliers], width_b, height_b)
    ratio = count / max(len(matches), 1)

    supportive = (
        len(matches) >= GEOMETRY_MIN_MATCHES
        and count >= GEOMETRY_MIN_INLIERS
        and ratio >= GEOMETRY_MIN_INLIER_RATIO
        and min(support_a["cells"], support_b["cells"]) >= GEOMETRY_MIN_CELLS
        and min(support_a["rows"], support_b["rows"]) >= GEOMETRY_MIN_ROWS
        and min(support_a["columns"], support_b["columns"]) >= GEOMETRY_MIN_COLUMNS
        and min(support_a["horizontal_spread"], support_b["horizontal_spread"])
        >= GEOMETRY_MIN_HORIZONTAL_SPREAD
        and min(support_a["vertical_spread"], support_b["vertical_spread"])
        >= GEOMETRY_MIN_VERTICAL_SPREAD
    )
    result.update(
        {
            "inliers": count,
            "inlier_ratio": round(ratio, 4),
            "support_a": support_a,
            "support_b": support_b,
            "supportive": bool(supportive),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("cam0", "cam5"), required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("descriptors", "path", "geometry", "all"),
        default="all",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--mask-variant", choices=build_reliability_masks.KNOWN_VARIANTS, default="",
        help="Development mask variant; output goes to outputs/task2_unified_variants/.",
    )
    parser.add_argument(
        "--greyscale", action="store_true",
        help="Luminance-only DINOv2 input; development variant, separate output.",
    )
    parser.add_argument(
        "--aggregation", choices=AGGREGATION_MODES, default="mean",
        help=(
            "Descriptor aggregation. 'mean' is the shipped masked mean; 'vlad' is "
            "an unsupervised VLAD over the same tokens (development variant, "
            "output under outputs/task2_unified_variants/vlad/)."
        ),
    )
    parser.add_argument(
        "--refinement", choices=REFINEMENT_MODES, default=REFINEMENT_DEFAULT,
        help=(
            "Stage-3 rule. 'ratchet' is the shipped greedy scan behind a "
            "non-decreasing floor. 'slope' is a development variant: a Viterbi "
            "over the same windows with a slope prior and a bounded step, which "
            "removes the forward ratchet. Output goes to "
            "outputs/task2_unified_variants/slope/."
        ),
    )
    parser.add_argument(
        "--strict-geometry",
        action="store_true",
        help=(
            "Give the top tier only to frames geometry actually verified; demote "
            "propagated spans. Writes to a separate file so the frozen mapping, "
            "which carries a blind number, is left alone."
        ),
    )
    arguments = parser.parse_args()

    configure_variant(
        mask_variant=arguments.mask_variant,
        greyscale=bool(arguments.greyscale),
        refinement=arguments.refinement,
        aggregation=arguments.aggregation,
    )

    camera = arguments.camera
    camera_dir = output_dir(camera)
    camera_dir.mkdir(parents=True, exist_ok=True)
    if variant_name():
        print(f"[{camera}] development variant {variant_name()!r} -> {camera_dir.relative_to(ROOT)}")

    if arguments.validate_only:
        mapping = camera_dir / "frame_mapping_unified.csv"
        validate_mapping(mapping, expected_row_count(camera, mapping))
        return

    print(f"[{camera}] stage 1: masked DINOv2 descriptors")
    descriptors_a = compute_descriptors("runA", camera, arguments.force)
    descriptors_b = compute_descriptors("runB", camera, arguments.force)
    if arguments.stage == "descriptors":
        return

    run_pipeline(camera, descriptors_a, descriptors_b, arguments)


def classify_sequence(margins: list[float]) -> str:
    """Turn the per-scale margins into a tier.

    The longest window is required in both tiers: short windows are exactly what
    a repeated row of awnings or a continuous wall can satisfy by accident.
    """

    positive = sum(1 for margin in margins if margin > 0)
    longest_positive = margins[-1] > 0
    if positive == STRONG_SEQUENCE_SCALES and longest_positive:
        return "strong"
    if positive >= SUPPORTED_SEQUENCE_SCALES and longest_positive:
        return "supported"
    return "failed"


def run_pipeline(
    camera: str,
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    arguments: argparse.Namespace,
) -> None:
    camera_dir = output_dir(camera)
    bounds = route_bounds()
    total_a = descriptors_a.shape[0]
    total_b = descriptors_b.shape[0]
    run_a_start = bounds["runA_start"]
    run_a_end = min(RUN_A_ROUTE_TAIL_EXCLUSIVE, total_a)
    run_b_start = bounds["runB_start"]
    run_b_end = total_b

    print(f"[{camera}] stage 2: global monotonic path over the whole route")
    coarse_a = np.arange(run_a_start, run_a_end, COARSE_STRIDE)
    coarse_b = np.arange(run_b_start, run_b_end, COARSE_STRIDE)
    similarity = descriptors_a[coarse_a] @ descriptors_b[coarse_b].T
    expected_slope = max(len(coarse_b) / max(len(coarse_a), 1), 1e-3)
    path = global_monotonic_path(similarity, expected_slope)
    print(f"   {len(coarse_a)}x{len(coarse_b)} coarse nodes, path length {len(path)}")

    estimate = densify_path(path, coarse_a, coarse_b, total_a)

    print(
        f"[{camera}] stage 3: full-cadence structure refinement "
        f"(+/-{REFINE_RADIUS}, rule {VARIANT.refinement})"
    )
    route_frames = [int(f) for f in range(run_a_start, run_a_end) if estimate[f] >= 0]
    needed_b = sorted(
        {
            int(np.clip(estimate[f] + delta, 0, total_b - 1))
            for f in route_frames
            for delta in range(-REFINE_RADIUS, REFINE_RADIUS + 1)
        }
    )
    structure_a = structure_descriptors("runA", camera, route_frames)
    structure_b = structure_descriptors("runB", camera, needed_b)
    refined = estimate.copy()
    for frame, candidate in refine_sequence(
        route_frames, estimate, structure_a, structure_b, total_b
    ).items():
        refined[frame] = candidate

    print(f"[{camera}] stage 4: multi-scale sequence uniqueness")
    margins = sequence_margins(similarity, path)
    node_frames = np.array([coarse_a[row] for row, _ in path])
    tier_by_frame: dict[int, str] = {}
    margin_by_frame: dict[int, list[float]] = {}
    for frame in route_frames:
        node = int(np.argmin(np.abs(node_frames - frame)))
        margin_by_frame[frame] = margins[node]
        tier_by_frame[frame] = classify_sequence(margins[node])
    strong = sum(1 for tier in tier_by_frame.values() if tier == "strong")
    supported = sum(1 for tier in tier_by_frame.values() if tier == "supported")
    print(
        f"   strong {strong}, sequence-supported {supported}, "
        f"failed {len(route_frames) - strong - supported}"
    )

    print(f"[{camera}] stage 5: sparse geometry at {GEOMETRY_SIZE[0]}x{GEOMETRY_SIZE[1]}")
    mask_a = build_reliability_masks.load_mask("runA", camera, VARIANT.mask_variant)["exclusion_224"]
    mask_b = build_reliability_masks.load_mask("runB", camera, VARIANT.mask_variant)["exclusion_224"]
    sample_frames = [
        frame for frame in route_frames[:: GEOMETRY_SAMPLE_STRIDE] if tier_by_frame[frame] != "failed"
    ]
    images_a = frame_service.frames("runA", camera, sample_frames, *GEOMETRY_SIZE)
    partner = {frame: int(refined[frame]) for frame in sample_frames}
    images_b = frame_service.frames("runB", camera, sorted(set(partner.values())), *GEOMETRY_SIZE)
    geometry: dict[int, dict[str, object]] = {}
    for index, frame in enumerate(sample_frames):
        geometry[frame] = verify_pair(
            images_a[frame], images_b[partner[frame]], mask_a, mask_b
        )
        if index % 25 == 0:
            print(f"   geometry {index + 1}/{len(sample_frames)}", flush=True)
    passing = [frame for frame in sample_frames if geometry[frame]["supportive"]]
    match_counts = [int(geometry[frame]["matches"]) for frame in sample_frames]
    print(
        f"   {len(passing)}/{len(sample_frames)} checks passed; "
        f"median sparse matches {int(np.median(match_counts)) if match_counts else 0}"
    )

    print(f"[{camera}] stage 6: typed per-frame decision")
    geometry_supported = np.zeros(total_a, dtype=bool)
    for left, right in zip(passing, passing[1:]):
        geometry_supported[left : right + 1] = True
    for frame in passing:
        geometry_supported[frame] = True

    rows: list[dict[str, object]] = []
    for frame in range(total_a):
        record = {
            "camera_id": camera,
            "runA_frame": frame,
            "runB_frame": "",
            "confidence": "0.00",
            "tier": "abstain",
            "status": "",
            "sequence_tier": "",
            "longest_window_margin": "",
            "candidate_runB_frame": "",
            "geometry_checked": "false",
            "geometry_supportive": "false",
            "confidence_is_calibrated_probability": "False",
            "method": PIPELINE_VERSION,
        }
        if frame < run_a_start or frame >= run_a_end:
            record["status"] = STATUS_IDLE
            rows.append(record)
            continue
        if estimate[frame] < 0:
            record["status"] = STATUS_OUT_OF_ROUTE
            rows.append(record)
            continue

        candidate = int(refined[frame])
        record["candidate_runB_frame"] = candidate
        tier = tier_by_frame.get(frame, "failed")
        record["sequence_tier"] = tier
        record["longest_window_margin"] = round(margin_by_frame[frame][-1], 5)
        if frame in geometry:
            record["geometry_checked"] = "true"
            record["geometry_supportive"] = str(bool(geometry[frame]["supportive"])).lower()

        if tier == "failed":
            near_start = candidate < run_b_start + int(0.12 * (run_b_end - run_b_start))
            near_end = candidate > run_b_end - int(0.12 * (run_b_end - run_b_start))
            record["status"] = (
                STATUS_LOOP_AMBIGUOUS if (near_start or near_end) else STATUS_SEQUENCE_FAILED
            )
            rows.append(record)
            continue
        if not geometry_supported[frame]:
            record["status"] = STATUS_GEOMETRY_FAILED
            rows.append(record)
            continue

        record["runB_frame"] = candidate
        directly_checked = frame in geometry and bool(geometry[frame]["supportive"])
        if arguments.strict_geometry and not directly_checked:
            record["status"] = STATUS_ACCEPTED_PROPAGATED
            record["tier"] = "uncalibrated_geometry_propagated"
            record["confidence"] = "0.50"
        else:
            record["status"] = (
                STATUS_ACCEPTED_STRONG if tier == "strong" else STATUS_ACCEPTED_SUPPORTED
            )
            record["tier"] = (
                "uncalibrated_strong" if tier == "strong" else "uncalibrated_sequence_supported"
            )
            record["confidence"] = "1.00" if tier == "strong" else "0.67"
        rows.append(record)

    bridge_short_gaps(rows)

    target = camera_dir / (
        "frame_mapping_unified_strict_geometry.csv"
        if arguments.strict_geometry
        else "frame_mapping_unified.csv"
    )
    fieldnames = list(rows[0])
    atomic_write(
        target,
        lambda path: _write_csv(path, fieldnames, rows),
    )

    mapped = sum(1 for row in rows if row["runB_frame"] != "")
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera,
        "variant": variant_name() or None,
        "refinement": {
            "mode": VARIANT.refinement,
            "refine_radius": REFINE_RADIUS,
            "structure_grid": list(STRUCTURE_GRID),
            "slope_weight": REFINE_SLOPE_WEIGHT if VARIANT.refinement == "slope" else None,
            "step_bounds": (
                [REFINE_MIN_STEP, REFINE_MAX_STEP] if VARIANT.refinement == "slope" else None
            ),
            "expected_step_source": (
                "coarse densified estimate's local increment"
                if VARIANT.refinement == "slope"
                else None
            ),
            "note": (
                "Viterbi over the +/-radius windows: emission is the structure "
                "cosine, transition cost is slope_weight per frame of departure "
                "from the coarse increment, monotonicity is a hard step bound."
                if VARIANT.refinement == "slope"
                else "Shipped rule: greedy argmax behind a non-decreasing floor."
            ),
        },
        "descriptor_version": descriptor_version(),
        "mask_version": build_reliability_masks.mask_version(VARIANT.mask_variant),
        "colour_id": frame_service.COLOUR_ID,
        "descriptor_raster": list(frame_service.DESCRIPTOR_SIZE),
        "geometry_raster": list(GEOMETRY_SIZE),
        "route": {
            "runA_start": run_a_start,
            "runA_end_exclusive": run_a_end,
            "runB_start": run_b_start,
            "runB_end_exclusive": run_b_end,
        },
        "rows": len(rows),
        "mapped": mapped,
        "coverage": round(mapped / len(rows), 6),
        "status_counts": status_counts,
        "geometry_checks": len(sample_frames),
        "geometry_passed": len(passing),
        "geometry_median_matches": int(np.median(match_counts)) if match_counts else 0,
        "search_policy": "global monotonic path; no prior band",
        "strict_geometry": bool(arguments.strict_geometry),
        "confidence_is_calibrated_probability": False,
        "confidence_semantics": (
            "ordinal, not a probability: 1.00 strong, 0.67 sequence-supported, "
            "0.50 geometry-propagated, 0.33 bridged, 0.00 abstain. The categorical "
            "label is carried in `tier`. A calibrated probability exists only in "
            "the posterior model, which has no held-out number and is not submitted."
        ),
    }
    atomic_write(
        camera_dir / (
            "unified_manifest_strict_geometry.json"
            if arguments.strict_geometry
            else "unified_manifest.json"
        ),
        lambda path: path.write_text(json.dumps(manifest, indent=2)),
    )
    print(f"   {mapped}/{len(rows)} mapped ({mapped / len(rows):.2%})")
    print(f"   statuses: {status_counts}")
    validate_mapping(target, total_a)


def bridge_short_gaps(rows: list[dict[str, object]]) -> None:
    """Fill runs of at most MAX_BRIDGED_GAP frames bounded by accepted rows.

    Only where the candidate already sits between its accepted neighbours, so
    bridging cannot invent a correspondence that breaks route order.
    """

    index = 0
    while index < len(rows):
        if rows[index]["runB_frame"] != "":
            index += 1
            continue
        start = index
        while index < len(rows) and rows[index]["runB_frame"] == "":
            index += 1
        end = index
        if start == 0 or end >= len(rows):
            continue
        if end - start > MAX_BRIDGED_GAP:
            continue
        left = rows[start - 1]
        right = rows[end]
        if left["runB_frame"] == "" or right["runB_frame"] == "":
            continue
        if str(rows[start]["status"]) in (STATUS_IDLE, STATUS_OUT_OF_ROUTE, STATUS_LOOP_AMBIGUOUS):
            continue
        low, high = int(left["runB_frame"]), int(right["runB_frame"])
        for position in range(start, end):
            candidate = rows[position]["candidate_runB_frame"]
            if candidate == "" or not (low <= int(candidate) <= high):
                break
        else:
            for position in range(start, end):
                rows[position]["runB_frame"] = rows[position]["candidate_runB_frame"]
                rows[position]["status"] = STATUS_ACCEPTED_BRIDGED
                rows[position]["tier"] = "uncalibrated_bridged"
                rows[position]["confidence"] = "0.33"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expected_row_count(camera: str, mapping: Path) -> int:
    """How many Run A rows the mapping must have, without decoding the video.

    Decoding is the authority when the raw HEVC is present: the row count is a
    claim about the recording, and scanning the bitstream is what makes the
    check independent of the file being checked. But the submission bundle
    excludes the raw video by policy, and a validation that can only run where
    the confidential source lives is not a validation a reviewer can use.

    So the count falls back to the manifest the build wrote beside the mapping -
    still a second file, still written by the pipeline rather than read out of
    the CSV, so a truncated or padded mapping is still caught. What is lost is
    only the cross-check against the bitstream itself, and the fallback says so.
    """

    try:
        return frame_count("runA", camera)
    except FileNotFoundError:
        pass
    manifest = mapping.parent / "unified_manifest.json"
    if not manifest.is_file():
        raise SystemExit(
            f"Neither the raw Run A {camera} recording nor {manifest.name} is "
            "present, so the expected row count cannot be established. "
            "Validation needs one of the two."
        )
    rows = int(json.loads(manifest.read_text())["rows"])
    print(
        f"[{camera}] raw video absent; expecting {rows} rows from "
        f"{manifest.name} rather than from a bitstream scan"
    )
    return rows


def validate_mapping(path: Path, expected_rows: int) -> None:
    """Check the output contract: every ordinal present, typed, and monotonic."""

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(f"{path.name}: {len(rows)} rows, expected {expected_rows}")
    ordinals = [int(row["runA_frame"]) for row in rows]
    if ordinals != list(range(expected_rows)):
        raise ValueError(f"{path.name}: Run A ordinals are not 0..{expected_rows - 1}")
    untyped = [row for row in rows if not row["status"]]
    if untyped:
        raise ValueError(f"{path.name}: {len(untyped)} rows have an empty status")
    mapped = [(int(row["runA_frame"]), int(row["runB_frame"])) for row in rows if row["runB_frame"]]
    for (_, previous), (_, current) in zip(mapped, mapped[1:]):
        if current < previous:
            raise ValueError(f"{path.name}: mapping is not monotonic at Run B {current}")
    print(
        f"{path.name}: {len(rows)} rows, {len(mapped)} mapped "
        f"({len(mapped) / len(rows):.2%}), monotonic, all rows typed"
    )


if __name__ == "__main__":
    main()
