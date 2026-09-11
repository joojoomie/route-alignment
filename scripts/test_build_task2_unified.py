#!/usr/bin/env python3
"""Focused unit checks for the unified correspondence pipeline.

These exercise the decision logic on synthetic inputs. Nothing here touches
video, model weights, or the label files.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

import build_task2_unified as unified


class SequenceTierTests(unittest.TestCase):
    def test_all_scales_positive_is_strong(self) -> None:
        self.assertEqual(unified.classify_sequence([0.1, 0.1, 0.1, 0.1]), "strong")

    def test_three_scales_with_longest_positive_is_supported(self) -> None:
        self.assertEqual(unified.classify_sequence([-0.01, 0.1, 0.1, 0.1]), "supported")

    def test_longest_window_negative_always_fails(self) -> None:
        # A short window can be satisfied by a repeated awning or a continuous
        # wall; the long window is the one that says the corridor is unique.
        self.assertEqual(unified.classify_sequence([0.2, 0.2, 0.2, -0.01]), "failed")

    def test_two_scales_is_not_enough(self) -> None:
        self.assertEqual(unified.classify_sequence([-0.1, -0.1, 0.1, 0.1]), "failed")


class MonotonicPathTests(unittest.TestCase):
    def test_recovers_a_known_linear_warp(self) -> None:
        rows, columns = 40, 60
        truth = np.round(np.linspace(0, columns - 1, rows)).astype(int)
        similarity = np.full((rows, columns), 0.1)
        similarity[np.arange(rows), truth] = 0.95
        path = unified.global_monotonic_path(similarity, columns / rows)
        recovered = {row: column for row, column in path}
        errors = [abs(recovered[row] - truth[row]) for row in range(rows) if row in recovered]
        self.assertLessEqual(float(np.median(errors)), 1.0)

    def test_path_is_strictly_non_decreasing_in_run_b(self) -> None:
        rng = np.random.default_rng(0)
        similarity = rng.random((30, 45))
        path = unified.global_monotonic_path(similarity, 1.5)
        columns = [column for _, column in path]
        self.assertEqual(columns, sorted(columns))

    def test_global_search_beats_a_distant_decoy(self) -> None:
        # A local band around a linear prior would be captured by the decoy; a
        # global path should not be, because the decoy is not route-consistent.
        rows, columns = 30, 120
        truth = np.round(np.linspace(0, 59, rows)).astype(int)
        similarity = np.full((rows, columns), 0.05)
        similarity[np.arange(rows), truth] = 0.9
        similarity[10:20, 100:110] = 0.95  # bright but order-inconsistent block
        path = unified.global_monotonic_path(similarity, 60 / rows)
        recovered = {row: column for row, column in path}
        self.assertLess(recovered[25], 90)


class SpatialSupportTests(unittest.TestCase):
    def test_distributed_points_occupy_several_cells(self) -> None:
        points = np.array([[100.0, 100.0], [700.0, 150.0], [400.0, 500.0], [800.0, 600.0]])
        support = unified.spatial_support(points, 896, 672)
        self.assertGreaterEqual(support["cells"], 3)
        self.assertGreaterEqual(support["rows"], 2)
        self.assertGreaterEqual(support["columns"], 2)

    def test_points_on_one_horizontal_edge_fail_the_row_requirement(self) -> None:
        # This is the repeated-awning failure the distributed-support gate exists
        # to reject: many inliers, all on one horizontal band.
        points = np.array([[x, 300.0] for x in np.linspace(50, 850, 40)])
        support = unified.spatial_support(points, 896, 672)
        self.assertEqual(support["rows"], 1)
        self.assertLess(support["vertical_spread"], unified.GEOMETRY_MIN_VERTICAL_SPREAD)

    def test_empty_input_is_zeroed_not_an_error(self) -> None:
        support = unified.spatial_support(np.zeros((0, 2)), 896, 672)
        self.assertEqual(support["cells"], 0)


def _row(frame: int, run_b: object, status: str, candidate: object = "") -> dict[str, object]:
    return {
        "camera_id": "cam0",
        "runA_frame": frame,
        "runB_frame": run_b,
        "confidence": "abstain" if run_b == "" else "uncalibrated_strong",
        "status": status,
        "candidate_runB_frame": candidate,
    }


class BridgeShortGapTests(unittest.TestCase):
    def test_bridges_a_short_bounded_gap(self) -> None:
        rows = [
            _row(0, 10, unified.STATUS_ACCEPTED_STRONG, 10),
            _row(1, "", unified.STATUS_GEOMETRY_FAILED, 11),
            _row(2, "", unified.STATUS_GEOMETRY_FAILED, 12),
            _row(3, 13, unified.STATUS_ACCEPTED_STRONG, 13),
        ]
        unified.bridge_short_gaps(rows)
        self.assertEqual(rows[1]["runB_frame"], 11)
        self.assertEqual(rows[1]["status"], unified.STATUS_ACCEPTED_BRIDGED)

    def test_refuses_when_a_candidate_leaves_the_bounding_interval(self) -> None:
        rows = [
            _row(0, 10, unified.STATUS_ACCEPTED_STRONG, 10),
            _row(1, "", unified.STATUS_GEOMETRY_FAILED, 400),
            _row(2, 13, unified.STATUS_ACCEPTED_STRONG, 13),
        ]
        unified.bridge_short_gaps(rows)
        self.assertEqual(rows[1]["runB_frame"], "")

    def test_refuses_to_bridge_a_loop_ambiguous_gap(self) -> None:
        # Near the route seam the ground truth itself is ambiguous, so filling
        # the gap would manufacture confidence the data does not support.
        rows = [
            _row(0, 10, unified.STATUS_ACCEPTED_STRONG, 10),
            _row(1, "", unified.STATUS_LOOP_AMBIGUOUS, 11),
            _row(2, 12, unified.STATUS_ACCEPTED_STRONG, 12),
        ]
        unified.bridge_short_gaps(rows)
        self.assertEqual(rows[1]["runB_frame"], "")

    def test_refuses_a_gap_longer_than_the_limit(self) -> None:
        rows = [_row(0, 10, unified.STATUS_ACCEPTED_STRONG, 10)]
        for offset in range(1, unified.MAX_BRIDGED_GAP + 2):
            rows.append(_row(offset, "", unified.STATUS_GEOMETRY_FAILED, 10 + offset))
        rows.append(_row(len(rows), 100, unified.STATUS_ACCEPTED_STRONG, 100))
        unified.bridge_short_gaps(rows)
        self.assertTrue(all(row["runB_frame"] == "" for row in rows[1:-1]))

    def test_never_fills_an_idle_segment(self) -> None:
        rows = [
            _row(0, 10, unified.STATUS_ACCEPTED_STRONG, 10),
            _row(1, "", unified.STATUS_IDLE, ""),
            _row(2, 12, unified.STATUS_ACCEPTED_STRONG, 12),
        ]
        unified.bridge_short_gaps(rows)
        self.assertEqual(rows[1]["runB_frame"], "")


class ValidateMappingTests(unittest.TestCase):
    def _write(self, rows: list[dict[str, object]]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.DictWriter(handle, fieldnames=["runA_frame", "runB_frame", "status"])
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        return Path(handle.name)

    def test_accepts_a_well_formed_mapping(self) -> None:
        path = self._write(
            [
                {"runA_frame": 0, "runB_frame": "", "status": unified.STATUS_IDLE},
                {"runA_frame": 1, "runB_frame": 5, "status": unified.STATUS_ACCEPTED_STRONG},
                {"runA_frame": 2, "runB_frame": 7, "status": unified.STATUS_ACCEPTED_STRONG},
            ]
        )
        unified.validate_mapping(path, 3)
        path.unlink()

    def test_rejects_a_non_monotonic_mapping(self) -> None:
        path = self._write(
            [
                {"runA_frame": 0, "runB_frame": 9, "status": unified.STATUS_ACCEPTED_STRONG},
                {"runA_frame": 1, "runB_frame": 4, "status": unified.STATUS_ACCEPTED_STRONG},
            ]
        )
        with self.assertRaises(ValueError):
            unified.validate_mapping(path, 2)
        path.unlink()

    def test_rejects_an_untyped_refusal(self) -> None:
        # An empty runB_frame is allowed; an empty status is not. A bare blank
        # invites a consumer to read it as "no such place".
        path = self._write([{"runA_frame": 0, "runB_frame": "", "status": ""}])
        with self.assertRaises(ValueError):
            unified.validate_mapping(path, 1)
        path.unlink()

    def test_rejects_a_missing_ordinal(self) -> None:
        path = self._write(
            [
                {"runA_frame": 0, "runB_frame": "", "status": unified.STATUS_IDLE},
                {"runA_frame": 2, "runB_frame": "", "status": unified.STATUS_IDLE},
            ]
        )
        with self.assertRaises(ValueError):
            unified.validate_mapping(path, 2)
        path.unlink()


class DensifyPathTests(unittest.TestCase):
    def test_interpolates_between_coarse_nodes(self) -> None:
        path = [(0, 0), (1, 1), (2, 2)]
        coarse_a = np.array([100, 104, 108])
        coarse_b = np.array([200, 208, 216])
        estimate = unified.densify_path(path, coarse_a, coarse_b, 120)
        self.assertEqual(estimate[100], 200)
        self.assertEqual(estimate[104], 208)
        self.assertEqual(estimate[102], 204)
        self.assertEqual(estimate[99], -1)
        self.assertEqual(estimate[109], -1)


if __name__ == "__main__":
    unittest.main()
