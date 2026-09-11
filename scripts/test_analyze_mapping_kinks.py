#!/usr/bin/env python3
"""Unit checks for the kink audit, on synthetic mappings only.

Nothing here reads video, model weights, the frozen mapping, or the label file.
Every fixture is hand-built so the expected answer is known by construction.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import analyze_mapping_kinks as kinks


def mapping_rows(pairs, statuses=None, tier="uncalibrated_strong"):
    """Build normalised mapping rows from (runA, runB_or_None) pairs."""

    rows = []
    for index, (frame_a, frame_b) in enumerate(pairs):
        rows.append(
            {
                "runA_frame": frame_a,
                "runB_frame": frame_b,
                "confidence": "1.00" if frame_b is not None else "0.00",
                "tier": tier if frame_b is not None else "abstain",
                "status": (statuses[index] if statuses else ("accepted_strong" if frame_b is not None else "idle_segment")),
            }
        )
    return rows


class PhysicalBoundTests(unittest.TestCase):
    """The declared bound must stay tied to the pipeline it is derived from."""

    def test_headline_threshold_is_above_the_coarse_slope_prior(self) -> None:
        # K must exceed what the coarse DTW is allowed to sustain, otherwise the
        # audit would flag paths the search was entitled to produce.
        self.assertGreater(kinks.HEADLINE_K, kinks.DTW_MAX_SLOPE)
        self.assertGreater(kinks.HEADLINE_K, kinks.DTW_MAX_SINGLE_STEP_SLOPE)

    def test_headline_threshold_is_one_of_the_reported_thresholds(self) -> None:
        self.assertIn(kinks.HEADLINE_K, kinks.KINK_THRESHOLDS)

    def test_thresholds_are_increasing_so_counts_are_nested(self) -> None:
        self.assertEqual(list(kinks.KINK_THRESHOLDS), sorted(kinks.KINK_THRESHOLDS))

    def test_bound_matches_the_shipped_pipeline_constants(self) -> None:
        import build_task2_unified as unified

        self.assertEqual(kinks.DTW_MIN_SLOPE, unified.MIN_SLOPE)
        self.assertEqual(kinks.DTW_MAX_SLOPE, unified.MAX_SLOPE)
        self.assertEqual(
            kinks.DTW_MAX_SINGLE_STEP_SLOPE,
            max(column / row for row, column in unified.DTW_STEPS),
        )


class StepExtractionTests(unittest.TestCase):
    def test_increments_are_first_differences_of_run_b(self) -> None:
        rows = mapping_rows([(0, 10), (1, 11), (2, 13), (3, 13)])
        steps = kinks.consecutive_steps(rows)
        self.assertEqual([s["increment"] for s in steps], [1, 2, 0])

    def test_a_gap_in_run_a_is_not_a_step(self) -> None:
        # A jump across a hole in Run A is the hole, not a speed claim.
        rows = mapping_rows([(0, 10), (1, 11), (2, None), (3, 40), (4, 41)])
        steps = kinks.consecutive_steps(rows)
        self.assertEqual([(s["runA_prev"], s["runA_frame"]) for s in steps], [(0, 1), (3, 4)])
        self.assertEqual([s["increment"] for s in steps], [1, 1])

    def test_a_missing_run_a_row_entirely_also_breaks_the_pair(self) -> None:
        rows = mapping_rows([(0, 10), (5, 20)])
        self.assertEqual(kinks.consecutive_steps(rows), [])

    def test_step_records_the_run_b_endpoints(self) -> None:
        rows = mapping_rows([(7, 100), (8, 111)])
        step = kinks.consecutive_steps(rows)[0]
        self.assertEqual(
            (step["runB_before"], step["runB_after"], step["increment"]), (100, 111, 11)
        )


class KinkDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        # A clean unit-speed mapping with one +7 jump planted at Run A 4.
        pairs = [(index, index) for index in range(4)]
        pairs += [(4, 10)] + [(index, index + 6) for index in range(5, 10)]
        self.steps = kinks.consecutive_steps(mapping_rows(pairs))

    def test_finds_the_planted_jump_at_the_headline_threshold(self) -> None:
        found = kinks.find_kinks(self.steps, kinks.HEADLINE_K)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["runA_frame"], 4)
        self.assertEqual(found[0]["increment"], 7)

    def test_counts_are_nested_in_the_threshold(self) -> None:
        counts = [len(kinks.find_kinks(self.steps, k)) for k in (4, 6, 8, 10)]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts, [1, 1, 0, 0])

    def test_a_clean_unit_speed_mapping_has_no_kinks(self) -> None:
        steps = kinks.consecutive_steps(mapping_rows([(i, i) for i in range(50)]))
        self.assertEqual(kinks.find_kinks(steps, 4), [])

    def test_a_stop_repeats_run_b_and_is_never_a_kink(self) -> None:
        # A stop is increment 0. Only Run B *sprinting* makes a kink.
        pairs = [(i, 5) for i in range(20)]
        steps = kinks.consecutive_steps(mapping_rows(pairs))
        self.assertEqual(kinks.find_kinks(steps, 4), [])
        self.assertEqual(max(s["increment"] for s in steps), 0)

    def test_a_sustained_two_times_speed_ratio_is_not_a_kink(self) -> None:
        # 2x is inside the coarse slope prior, so the audit must not flag it.
        steps = kinks.consecutive_steps(mapping_rows([(i, 2 * i) for i in range(30)]))
        self.assertEqual(kinks.find_kinks(steps, kinks.HEADLINE_K), [])


class DistributionTests(unittest.TestCase):
    def test_empty_input_reports_no_steps(self) -> None:
        self.assertEqual(kinks.increment_distribution([]), {"n": 0})

    def test_histogram_buckets_large_increments_together(self) -> None:
        pairs = [(0, 0), (1, 1), (2, 2), (3, 20)]
        summary = kinks.increment_distribution(kinks.consecutive_steps(mapping_rows(pairs)))
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["max"], 18)
        self.assertEqual(summary["histogram"][">10"], 1)
        self.assertEqual(summary["histogram"]["1"], 2)

    def test_a_non_monotone_mapping_is_reported_not_hidden(self) -> None:
        summary = kinks.increment_distribution(
            kinks.consecutive_steps(mapping_rows([(0, 10), (1, 4)]))
        )
        self.assertEqual(summary["negative_steps"], 1)
        self.assertEqual(summary["min"], -6)


class DemotionWindowTests(unittest.TestCase):
    def test_window_is_symmetric_and_inclusive(self) -> None:
        window = kinks.demoted_frames([{"runA_frame": 100}], radius=10)
        self.assertEqual(min(window), 90)
        self.assertEqual(max(window), 110)
        self.assertEqual(len(window), 21)

    def test_overlapping_windows_are_not_double_counted(self) -> None:
        window = kinks.demoted_frames(
            [{"runA_frame": 100}, {"runA_frame": 105}], radius=10
        )
        self.assertEqual(len(window), len(set(range(90, 116))))

    def test_no_kinks_demotes_nothing(self) -> None:
        self.assertEqual(kinks.demoted_frames([], radius=10), set())

    def test_default_radius_is_the_declared_policy(self) -> None:
        self.assertEqual(
            kinks.demoted_frames([{"runA_frame": 0}]),
            set(range(-kinks.DEMOTE_RADIUS, kinks.DEMOTE_RADIUS + 1)),
        )


class IntervalDistanceTests(unittest.TestCase):
    def test_inside_the_interval_is_zero(self) -> None:
        self.assertEqual(kinks.distance_to_interval(50, 49, 51), 0)

    def test_below_and_above_are_symmetric(self) -> None:
        self.assertEqual(kinks.distance_to_interval(45, 49, 51), 4)
        self.assertEqual(kinks.distance_to_interval(55, 49, 51), 4)


class DarknessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.flags = {
            ("runA", "cam0"): {10: "ok", 11: "ok", 20: "too_dark"},
            ("runB", "cam0"): {102: "ok", 103: "too_dark", 108: "ok"},
        }

    def test_a_dark_run_b_frame_inside_the_jumped_span_marks_the_kink(self) -> None:
        kink = {"runA_prev": 10, "runA_frame": 11, "runB_before": 100, "runB_after": 108}
        result = kinks.kink_darkness(kink, "cam0", self.flags)
        self.assertTrue(result["runB_span_too_dark"])
        self.assertFalse(result["runA_too_dark"])
        self.assertTrue(result["dark"])

    def test_a_lit_kink_is_lit(self) -> None:
        kink = {"runA_prev": 10, "runA_frame": 11, "runB_before": 104, "runB_after": 112}
        self.assertFalse(kinks.kink_darkness(kink, "cam0", self.flags)["dark"])

    def test_a_dark_run_a_frame_alone_marks_the_kink(self) -> None:
        kink = {"runA_prev": 19, "runA_frame": 20, "runB_before": 104, "runB_after": 112}
        result = kinks.kink_darkness(kink, "cam0", self.flags)
        self.assertTrue(result["runA_too_dark"])
        self.assertTrue(result["dark"])


class CrossCameraTests(unittest.TestCase):
    def test_a_constant_offset_is_removed_and_is_not_disagreement(self) -> None:
        # cam5 lags cam0 by a fixed 20 frames. That is unattributable, not error.
        cam0 = mapping_rows([(i, i) for i in range(400)])
        cam5 = mapping_rows([(i, i - 20) for i in range(400)])
        residual = kinks.cross_camera_residual({"cam0": cam0, "cam5": cam5})
        self.assertEqual(len(residual), 400)
        self.assertLess(max(residual.values()), 1e-9)

    def test_a_bounded_excursion_in_one_camera_shows_up_as_disagreement(self) -> None:
        # cam0 runs 12 frames ahead for 40 frames and then rejoins: exactly the
        # shape a ratcheted kink leaves behind.
        cam0 = mapping_rows([(i, i + (12 if 200 <= i < 240 else 0)) for i in range(600)])
        cam5 = mapping_rows([(i, i) for i in range(600)])
        residual = kinks.cross_camera_residual({"cam0": cam0, "cam5": cam5})
        self.assertGreater(kinks.local_disagreement(residual, 205), 6)
        self.assertLess(kinks.local_disagreement(residual, 20), 6)

    def test_a_permanent_step_is_absorbed_by_the_running_median(self) -> None:
        # An honest limit of this signal, asserted so it is not mistaken for a
        # guarantee: an offset that never comes back looks like a camera lag,
        # not a disagreement, once the running median has slid past it.
        cam0 = mapping_rows([(i, i + (12 if i >= 200 else 0)) for i in range(600)])
        cam5 = mapping_rows([(i, i) for i in range(600)])
        residual = kinks.cross_camera_residual({"cam0": cam0, "cam5": cam5})
        self.assertLess(kinks.local_disagreement(residual, 400), 1)

    def test_too_few_shared_frames_yields_no_signal_rather_than_a_guess(self) -> None:
        cam0 = mapping_rows([(i, i) for i in range(10)])
        cam5 = mapping_rows([(i, i) for i in range(10)])
        self.assertEqual(kinks.cross_camera_residual({"cam0": cam0, "cam5": cam5}), {})

    def test_local_disagreement_is_none_where_nothing_is_mapped(self) -> None:
        self.assertIsNone(kinks.local_disagreement({}, 500))


class LabelCrossCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        # Run A 0..99 mapped to Run B 0..99, with a +9 kink planted at Run A 50.
        pairs = [(i, i) for i in range(50)] + [(i, i + 9) for i in range(50, 100)]
        self.mapping = {"cam0": mapping_rows(pairs), "cam5": mapping_rows(pairs)}
        self.kinks = {
            camera: kinks.find_kinks(
                kinks.consecutive_steps(self.mapping[camera]), kinks.HEADLINE_K
            )
            for camera in ("cam0", "cam5")
        }

    def label(self, query, frame, low, high, camera="cam0", kind="match"):
        return {
            "query_id": query,
            "camera_id": camera,
            "runA_frame": frame,
            "label": kind,
            "runB_frame": (low + high) // 2,
            "runB_min": low,
            "runB_max": high,
        }

    def test_the_planted_kink_is_found_at_the_expected_frame(self) -> None:
        self.assertEqual([k["runA_frame"] for k in self.kinks["cam0"]], [50])

    def test_a_label_next_to_the_kink_is_bucketed_near_and_scored_a_miss(self) -> None:
        labels = [self.label("near", 55, 55, 55)]
        result = kinks.label_cross_check(labels, self.mapping, self.kinks)
        near = result["per_camera"]["cam0"]["near"]
        self.assertEqual(near["evaluable_match_labels"], 1)
        self.assertEqual(near["misses_interval_distance_gt_0"], 1)
        self.assertEqual(near["median_interval_distance"], 9.0)

    def test_a_label_far_from_the_kink_is_bucketed_away_and_scored_correct(self) -> None:
        labels = [self.label("away", 10, 10, 10)]
        result = kinks.label_cross_check(labels, self.mapping, self.kinks)
        away = result["per_camera"]["cam0"]["away"]
        self.assertEqual(away["evaluable_match_labels"], 1)
        self.assertEqual(away["misses_interval_distance_gt_0"], 0)
        self.assertEqual(away["miss_rate"], 0.0)

    def test_the_near_radius_boundary_is_inclusive(self) -> None:
        edge = 50 + kinks.LABEL_NEAR_RADIUS
        result = kinks.label_cross_check([self.label("edge", edge, 0, 0)], self.mapping, self.kinks)
        self.assertEqual(result["cases"][0]["bucket"], "near")
        beyond = kinks.label_cross_check(
            [self.label("beyond", edge + 1, 0, 0)], self.mapping, self.kinks
        )
        self.assertEqual(beyond["cases"][0]["bucket"], "away")

    def test_no_match_labels_are_excluded_from_the_miss_rate(self) -> None:
        labels = [self.label("nm", 55, 0, 0, kind="no_match")]
        result = kinks.label_cross_check(labels, self.mapping, self.kinks)
        self.assertEqual(result["per_camera"]["cam0"]["near"]["evaluable_match_labels"], 0)

    def test_an_abstention_is_counted_separately_from_a_miss(self) -> None:
        mapping = {
            "cam0": mapping_rows([(0, None), (1, 1)]),
            "cam5": mapping_rows([(0, None), (1, 1)]),
        }
        result = kinks.label_cross_check(
            [self.label("abs", 0, 5, 5)], mapping, {"cam0": [], "cam5": []}
        )
        away = result["per_camera"]["cam0"]["away"]
        self.assertEqual(away["abstained"], 1)
        self.assertEqual(away["evaluable_match_labels"], 0)


class SlopeGatePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        pairs = [(i, i) for i in range(50)] + [(i, i + 9) for i in range(50, 100)]
        self.mapping = {"cam0": mapping_rows(pairs), "cam5": mapping_rows(pairs)}
        self.kinks = {
            camera: kinks.find_kinks(
                kinks.consecutive_steps(self.mapping[camera]), kinks.HEADLINE_K
            )
            for camera in ("cam0", "cam5")
        }

    def test_demotes_exactly_the_window_around_the_kink(self) -> None:
        result = kinks.slope_gate_policy([], self.mapping, self.kinks)
        entry = result["per_camera"]["cam0"]
        self.assertEqual(entry["top_tier_rows"], 100)
        self.assertEqual(entry["demoted_rows"], 2 * kinks.DEMOTE_RADIUS + 1)
        self.assertEqual(entry["remaining_top_tier_rows"], 100 - (2 * kinks.DEMOTE_RADIUS + 1))

    def test_only_top_tier_rows_can_be_demoted(self) -> None:
        pairs = [(i, i) for i in range(50)] + [(i, i + 9) for i in range(50, 100)]
        statuses = ["accepted_sequence_supported"] * 100
        mapping = {c: mapping_rows(pairs, statuses) for c in ("cam0", "cam5")}
        result = kinks.slope_gate_policy([], mapping, self.kinks)
        self.assertEqual(result["per_camera"]["cam0"]["top_tier_rows"], 0)
        self.assertEqual(result["per_camera"]["cam0"]["demoted_rows"], 0)

    def test_a_demoted_false_accept_raises_precision_and_lowers_coverage(self) -> None:
        labels = [
            {
                "query_id": "bad",
                "camera_id": "cam0",
                "runA_frame": 55,
                "label": "match",
                "runB_frame": 55,
                "runB_min": 55,
                "runB_max": 55,
            },
            {
                "query_id": "good",
                "camera_id": "cam0",
                "runA_frame": 10,
                "label": "match",
                "runB_frame": 10,
                "runB_min": 10,
                "runB_max": 10,
            },
        ]
        entry = kinks.slope_gate_policy(labels, self.mapping, self.kinks)["per_camera"]["cam0"]
        self.assertEqual(entry["labels_on_demoted_rows_count"], 1)
        self.assertEqual(entry["labels_on_demoted_rows"][0]["query_id"], "bad")
        self.assertTrue(entry["labels_on_demoted_rows"][0]["was_a_false_accept"])
        self.assertEqual(entry["spent_labels_before_gate"]["strict_accepted_precision"], 0.5)
        self.assertEqual(entry["spent_labels_after_gate"]["strict_accepted_precision"], 1.0)
        self.assertEqual(entry["spent_labels_before_gate"]["coverage_all_labels"], 1.0)
        self.assertEqual(entry["spent_labels_after_gate"]["coverage_all_labels"], 0.5)

    def test_the_report_says_out_loud_that_the_labels_are_spent(self) -> None:
        result = kinks.slope_gate_policy([], self.mapping, self.kinks)
        self.assertIn("SPENT", result["label_status"])


class MappingIoTests(unittest.TestCase):
    def test_reads_a_schema_without_a_tier_column(self) -> None:
        # The Bayes posterior has no `tier`; the reader must not require one.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["camera_id", "runA_frame", "runB_frame", "confidence", "status"])
                writer.writerow(["cam0", "0", "", "abstain", "idle_segment"])
                writer.writerow(["cam0", "1", "7", "0.9", "accepted"])
            rows = kinks.read_mapping(path)
        self.assertEqual(rows[0]["runB_frame"], None)
        self.assertEqual(rows[1]["runB_frame"], 7)
        self.assertEqual(rows[1]["tier"], "")

    def test_rows_are_returned_in_run_a_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["runA_frame", "runB_frame"])
                writer.writerow(["5", "5"])
                writer.writerow(["1", "1"])
            rows = kinks.read_mapping(path)
        self.assertEqual([row["runA_frame"] for row in rows], [1, 5])


if __name__ == "__main__":
    unittest.main()
