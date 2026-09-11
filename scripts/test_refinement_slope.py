#!/usr/bin/env python3
"""Tests for the stage-3 refinement rules: the shipped ratchet and the fix.

The mapping audit in `outputs/task2_kinks/` traced the implausible one-step Run
B jumps ("kinks") to stage 3: it is a greedy argmax behind a non-decreasing
floor, with no slope prior, so where the structure score collapses the argmax
can land at the far edge of its window and the floor then makes that jump
permanent. These tests pin both halves of that claim on synthetic scores, and
guard the refactor that made the rule swappable: `refine_sequence` under
`ratchet` must still be the exact inline loop the frozen method shipped with.

The scores are built rather than measured. `structure_b[b]` is the b-th basis
vector and `structure_a[f]` is the row of scores for Run A frame f, so the dot
product the pipeline takes returns exactly the intended number.
"""

from __future__ import annotations

import unittest

import numpy as np

import build_task2_unified as unified
import frame_service


TOTAL_B = 80
CENTRE_OFFSET = 20


def score_field(rows: int, builder) -> tuple[list[int], np.ndarray, dict, dict]:
    """Build (route_frames, coarse estimate, structure_a, structure_b).

    `builder(frame, candidate)` returns the structure score for that pair. The
    coarse estimate is the identity warp shifted by CENTRE_OFFSET, so the
    truth-following mapping is `frame + CENTRE_OFFSET`.
    """

    route_frames = list(range(rows))
    estimate = np.array([f + CENTRE_OFFSET for f in route_frames], dtype=np.int64)
    structure_b = {b: np.eye(TOTAL_B)[b] for b in range(TOTAL_B)}
    structure_a = {
        frame: np.array([builder(frame, b) for b in range(TOTAL_B)], dtype=np.float64)
        for frame in route_frames
    }
    return route_frames, estimate, structure_a, structure_b


def legacy_refine(route_frames, estimate, structure_a, structure_b, total_b):
    """The stage-3 loop exactly as it stood inline in `run_pipeline`.

    Copied, not imported: it is the regression oracle for the refactor.
    """

    refined = {}
    previous = -1
    for frame in route_frames:
        centre = int(estimate[frame])
        best_score, best_frame = -2.0, max(centre, previous)
        for delta in range(-unified.REFINE_RADIUS, unified.REFINE_RADIUS + 1):
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
        refined[frame] = best_frame
        previous = best_frame
    return refined


def increments(picked: dict[int, int], route_frames: list[int]) -> list[int]:
    values = [picked[frame] for frame in route_frames]
    return [later - earlier for earlier, later in zip(values, values[1:])]


class RatchetRegressionTests(unittest.TestCase):
    """The shipped rule must be untouched by the refactor."""

    def test_default_mode_is_the_shipped_rule(self) -> None:
        self.assertEqual(unified.VARIANT.refinement, "ratchet")
        self.assertEqual(unified.REFINEMENT_DEFAULT, "ratchet")
        self.assertEqual(unified.variant_name(), "")

    def test_matches_the_old_inline_loop_on_random_scores(self) -> None:
        rng = np.random.default_rng(7)
        frames, estimate, structure_a, structure_b = score_field(
            40, lambda frame, b: float(rng.random())
        )
        self.assertEqual(
            unified.refine_sequence(frames, estimate, structure_a, structure_b, TOTAL_B),
            legacy_refine(frames, estimate, structure_a, structure_b, TOTAL_B),
        )

    def test_matches_the_old_inline_loop_on_a_collapsed_stretch(self) -> None:
        frames, estimate, structure_a, structure_b = score_field(45, collapsed_builder)
        self.assertEqual(
            unified.refine_sequence(
                frames, estimate, structure_a, structure_b, TOTAL_B, mode="ratchet"
            ),
            legacy_refine(frames, estimate, structure_a, structure_b, TOTAL_B),
        )

    def test_matches_the_old_inline_loop_when_descriptors_are_missing(self) -> None:
        frames, estimate, structure_a, structure_b = score_field(20, collapsed_builder)
        for frame in (5, 6):
            structure_a.pop(frame)
        for candidate in list(structure_b)[:CENTRE_OFFSET]:
            structure_b.pop(candidate)
        self.assertEqual(
            unified.refine_sequence(frames, estimate, structure_a, structure_b, TOTAL_B),
            legacy_refine(frames, estimate, structure_a, structure_b, TOTAL_B),
        )

    def test_an_unknown_mode_is_refused(self) -> None:
        frames, estimate, structure_a, structure_b = score_field(5, collapsed_builder)
        with self.assertRaises(ValueError):
            unified.refine_sequence(
                frames, estimate, structure_a, structure_b, TOTAL_B, mode="viterbi"
            )


def collapsed_builder(frame: int, candidate: int) -> float:
    """Truth at `frame + CENTRE_OFFSET`, with evidence collapsing for ten frames.

    Outside the collapse the truth wins by a wide margin. Inside it, every
    candidate scores about the same and a marginally better value sits at the
    far forward edge of the +/-8 window - which is what a dark or featureless
    stretch looks like to a gradient-structure descriptor.
    """

    truth = frame + CENTRE_OFFSET
    if 15 <= frame < 25:
        if candidate == truth + unified.REFINE_RADIUS:
            return 0.32
        return 0.30
    return 0.90 if candidate == truth else 0.10


class CollapsedEvidenceTests(unittest.TestCase):
    """The kink mechanism, and that the fix removes it."""

    def setUp(self) -> None:
        self.frames, self.estimate, self.structure_a, self.structure_b = score_field(
            45, collapsed_builder
        )

    def refine(self, mode: str) -> dict[int, int]:
        return unified.refine_sequence(
            self.frames, self.estimate, self.structure_a, self.structure_b, TOTAL_B, mode=mode
        )

    def test_the_ratchet_jumps_and_never_comes_back(self) -> None:
        picked = self.refine("ratchet")
        steps = increments(picked, self.frames)
        # One implausible step at the onset of the collapse: the argmax leaves
        # the truth for the far window edge.
        self.assertGreaterEqual(max(steps), unified.REFINE_RADIUS)
        self.assertGreaterEqual(sum(1 for step in steps if step >= 6), 1)
        # And the floor keeps it there after the evidence returns at frame 25:
        # the truth is not admissible again until Run B has caught up with the
        # eight frames the jump spent in advance.
        stuck = [frame for frame in range(25, 33) if picked[frame] != frame + CENTRE_OFFSET]
        self.assertEqual(stuck, list(range(25, 32)))

    def test_the_viterbi_does_not_jump(self) -> None:
        picked = self.refine("slope")
        steps = increments(picked, self.frames)
        self.assertLessEqual(max(steps), unified.REFINE_MAX_STEP)
        self.assertGreaterEqual(min(steps), unified.REFINE_MIN_STEP)
        self.assertEqual(sum(1 for step in steps if step >= 6), 0)

    def test_the_viterbi_stays_on_the_truth_through_the_collapse(self) -> None:
        picked = self.refine("slope")
        for frame in self.frames:
            self.assertEqual(picked[frame], frame + CENTRE_OFFSET)

    def test_the_viterbi_recovers_after_a_single_frame_spike(self) -> None:
        # One frame's argmax is wrong and forward. The ratchet inherits it for
        # the rest of the route; the sequence model treats it as one outlier.
        def builder(frame: int, candidate: int) -> float:
            truth = frame + CENTRE_OFFSET
            if frame == 20 and candidate == truth + unified.REFINE_RADIUS:
                return 0.95
            return 0.90 if candidate == truth else 0.10

        frames, estimate, structure_a, structure_b = score_field(40, builder)
        ratchet = unified.refine_sequence(
            frames, estimate, structure_a, structure_b, TOTAL_B, mode="ratchet"
        )
        slope = unified.refine_sequence(
            frames, estimate, structure_a, structure_b, TOTAL_B, mode="slope"
        )
        # The floor carries the spike forward; the truth at 41 is behind it.
        self.assertEqual(ratchet[20], 20 + CENTRE_OFFSET + unified.REFINE_RADIUS)
        self.assertEqual(ratchet[21], ratchet[20])
        self.assertEqual(slope[20], 20 + CENTRE_OFFSET)
        self.assertEqual(slope[21], 21 + CENTRE_OFFSET)


class SlopeContractTests(unittest.TestCase):
    """Whatever the scores are, the output must still be a legal mapping."""

    def test_monotonic_and_inside_the_window_on_random_scores(self) -> None:
        rng = np.random.default_rng(3)
        frames, estimate, structure_a, structure_b = score_field(
            60, lambda frame, b: float(rng.random())
        )
        picked = unified.refine_sequence(
            frames, estimate, structure_a, structure_b, TOTAL_B, mode="slope"
        )
        values = [picked[frame] for frame in frames]
        self.assertEqual(values, sorted(values))
        for frame in frames:
            self.assertLessEqual(
                abs(picked[frame] - int(estimate[frame])), unified.REFINE_RADIUS
            )
        for step in increments(picked, frames):
            self.assertGreaterEqual(step, unified.REFINE_MIN_STEP)
            self.assertLessEqual(step, unified.REFINE_MAX_STEP)

    def test_a_stall_is_representable(self) -> None:
        # Run B parked: several Run A frames legitimately share one Run B frame.
        def builder(frame: int, candidate: int) -> float:
            truth = min(frame, 10) + CENTRE_OFFSET
            return 0.90 if candidate == truth else 0.10

        frames, estimate, structure_a, structure_b = score_field(20, builder)
        picked = unified.refine_sequence(
            frames, estimate, structure_a, structure_b, TOTAL_B, mode="slope"
        )
        self.assertEqual(picked[12], 10 + CENTRE_OFFSET)
        self.assertEqual(picked[14], 10 + CENTRE_OFFSET)

    def test_clipping_at_the_stream_edge_is_handled(self) -> None:
        frames = list(range(10))
        estimate = np.array([TOTAL_B - 3 + f for f in frames], dtype=np.int64)
        structure_b = {b: np.eye(TOTAL_B)[b] for b in range(TOTAL_B)}
        structure_a = {
            frame: np.full(TOTAL_B, 0.1) for frame in frames
        }
        picked = unified.refine_sequence(
            frames, estimate, structure_a, structure_b, TOTAL_B, mode="slope"
        )
        values = [picked[frame] for frame in frames]
        self.assertEqual(values, sorted(values))
        self.assertLessEqual(max(values), TOTAL_B - 1)

    def test_empty_route_is_empty(self) -> None:
        self.assertEqual(
            unified.refine_sequence([], np.zeros(0), {}, {}, TOTAL_B, mode="slope"), {}
        )


class VariantPlumbingTests(unittest.TestCase):
    """The variant must be a flag, and must not be able to overwrite the frozen run."""

    def test_slope_routes_to_its_own_directory(self) -> None:
        saved = unified.configure_variant(refinement="slope")
        try:
            self.assertEqual(unified.variant_name(), "slope")
            self.assertEqual(
                unified.output_dir("cam0"),
                unified.VARIANT_OUTPUT_DIR / "slope" / "cam0",
            )
            # Descriptors are unaffected by the refinement rule, so the cache is
            # shared with the shipped run rather than recomputed.
            self.assertEqual(unified.descriptor_version(), unified.DESCRIPTOR_VERSION)
            if frame_service.source_available("runA", "cam0"):
                self.assertEqual(
                    unified.descriptor_cache_path("runA", "cam0").parent,
                    unified.OUTPUT_DIR / "cam0",
                )
        finally:
            unified.set_variant(saved)

    def test_slope_composes_with_the_other_variants(self) -> None:
        saved = unified.configure_variant(mask_variant="sky", greyscale=True, refinement="slope")
        try:
            self.assertEqual(unified.variant_name(), "sky+grey+slope")
        finally:
            unified.set_variant(saved)


if __name__ == "__main__":
    unittest.main()
