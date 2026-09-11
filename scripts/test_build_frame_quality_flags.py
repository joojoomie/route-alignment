#!/usr/bin/env python3
"""Tests for the per-frame exposure flag, on synthetic frames.

The measurement is small enough that the tests can state the arithmetic
exactly: a constant grey frame has a known Rec. 709 luma, and a mostly-black
frame with one bright patch has a known 99th percentile. That is the whole
point of the two-threshold rule - the second case is the one a mean-only floor
would wave through.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import build_frame_quality_flags as quality


def constant_frame(value: int, height: int = 16, width: int = 24) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


class LumaTests(unittest.TestCase):
    def test_grey_frame_luma_is_its_own_level(self):
        # The Rec. 709 weights sum to one, so R = G = B = v gives luma v.
        mean, p99 = quality.frame_luma(constant_frame(90))
        self.assertAlmostEqual(mean, 90.0, places=6)
        self.assertAlmostEqual(p99, 90.0, places=6)

    def test_channels_are_weighted_not_averaged(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[..., 1] = 200  # green only
        mean, _ = quality.frame_luma(frame)
        self.assertAlmostEqual(mean, 0.7152 * 200, places=6)

    def test_black_frame_is_zero(self):
        self.assertEqual(quality.frame_luma(constant_frame(0)), (0.0, 0.0))


class ClassificationTests(unittest.TestCase):
    def test_a_normally_exposed_frame_passes(self):
        mean, p99 = quality.frame_luma(constant_frame(80))
        self.assertEqual(quality.classify(mean, p99), quality.FLAG_OK)

    def test_the_measured_dark_stretch_level_is_flagged(self):
        # Run B CAM0 1460-1636 sits near mean luma 5 at this raster.
        mean, p99 = quality.frame_luma(constant_frame(5))
        self.assertEqual(quality.classify(mean, p99), quality.FLAG_TOO_DARK)

    def test_a_bright_mean_with_no_bright_content_is_still_flagged(self):
        # Uniform dim fog: mean clears the floor, nothing in the frame is
        # bright enough to carry structure. A mean-only rule would pass it.
        mean, p99 = quality.frame_luma(constant_frame(30))
        self.assertGreater(mean, quality.MEAN_LUMA_FLOOR)
        self.assertLess(p99, quality.P99_LUMA_FLOOR)
        self.assertEqual(quality.classify(mean, p99), quality.FLAG_TOO_DARK)

    def test_a_dark_frame_with_one_lit_corner_fails_on_the_mean(self):
        frame = constant_frame(2, height=100, width=100)
        frame[:8, :8, :] = 250  # 0.64% of the frame, below the p99 cut
        mean, p99 = quality.frame_luma(frame)
        self.assertLess(mean, quality.MEAN_LUMA_FLOOR)
        self.assertEqual(quality.classify(mean, p99), quality.FLAG_TOO_DARK)

    def test_the_thresholds_are_exclusive_bounds(self):
        self.assertEqual(
            quality.classify(quality.MEAN_LUMA_FLOOR, quality.P99_LUMA_FLOOR),
            quality.FLAG_OK,
        )
        self.assertEqual(
            quality.classify(quality.MEAN_LUMA_FLOOR - 1e-9, quality.P99_LUMA_FLOOR),
            quality.FLAG_TOO_DARK,
        )
        self.assertEqual(
            quality.classify(quality.MEAN_LUMA_FLOOR, quality.P99_LUMA_FLOOR - 1e-9),
            quality.FLAG_TOO_DARK,
        )


class StretchTests(unittest.TestCase):
    def test_consecutive_ordinals_become_one_stretch(self):
        self.assertEqual(
            quality.contiguous_stretches(list(range(10, 20))),
            [{"start": 10, "end": 19, "length": 10}],
        )

    def test_a_single_gap_splits_the_stretch(self):
        stretches = quality.contiguous_stretches([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
        self.assertEqual(
            stretches,
            [
                {"start": 0, "end": 4, "length": 5},
                {"start": 6, "end": 10, "length": 5},
            ],
        )

    def test_short_stretches_are_dropped_at_the_reporting_threshold(self):
        ordinals = [0, 1, 2] + list(range(100, 110))
        self.assertEqual(
            quality.contiguous_stretches(ordinals),
            [{"start": 100, "end": 109, "length": 10}],
        )
        self.assertEqual(len(quality.contiguous_stretches(ordinals, minimum=1)), 2)

    def test_unsorted_input_is_sorted_first(self):
        self.assertEqual(
            quality.contiguous_stretches([4, 1, 3, 0, 2]),
            [{"start": 0, "end": 4, "length": 5}],
        )

    def test_no_dark_frames_gives_no_stretches(self):
        self.assertEqual(quality.contiguous_stretches([]), [])


class SlopeGateTests(unittest.TestCase):
    """The slope-gate column, against a synthetic kink artifact."""

    def payload(self) -> dict:
        kinks = [{"runA_frame": 100, "increment": 9}, {"runA_frame": 400, "increment": 4}]
        return {
            "physical_bound": {"headline_threshold": 6},
            "mappings": {
                quality.KINK_MAPPING: {
                    "per_camera": {
                        "cam0": {"kinks": kinks, "kink_counts": {"K>=6": {"kinks": 1}}},
                        "cam5": {"kinks": [], "kink_counts": {"K>=6": {"kinks": 0}}},
                    }
                }
            },
            "slope_gate_policy": {"per_camera": {}},
        }

    def streams(self) -> dict:
        return {
            (run, camera): [{"ordinal": ordinal, "flag": quality.FLAG_OK} for ordinal in range(500)]
            for run in quality.RUNS
            for camera in quality.CAMERAS
        }

    def apply(self, tmp: Path) -> tuple[dict, dict]:
        import json

        target = tmp / "mapping_kinks.json"
        target.write_text(json.dumps(self.payload()))
        by_stream = self.streams()
        saved = quality.KINKS_JSON
        try:
            quality.KINKS_JSON = target
            report = quality.apply_slope_gate(by_stream)
        finally:
            quality.KINKS_JSON = saved
        return report, by_stream

    def test_flags_the_radius_around_a_headline_kink_only(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            report, by_stream = self.apply(Path(directory))
        flagged = {
            row["ordinal"]
            for row in by_stream[("runA", "cam0")]
            if row[quality.SLOPE_GATE_COLUMN] == quality.FLAG_NEAR_KINK
        }
        # +/-10 around frame 100, and nothing around the sub-threshold kink at 400.
        self.assertEqual(flagged, set(range(90, 111)))
        self.assertEqual(report["cameras"]["cam0"]["runA_frames_flagged"], 21)
        self.assertEqual(report["cameras"]["cam5"]["runA_frames_flagged"], 0)

    def test_run_b_rows_and_the_other_camera_are_left_empty(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            _, by_stream = self.apply(Path(directory))
        for key in (("runB", "cam0"), ("runA", "cam5"), ("runB", "cam5")):
            self.assertEqual(
                {row[quality.SLOPE_GATE_COLUMN] for row in by_stream[key]}, {""}
            )

    def test_a_missing_audit_still_writes_an_empty_column(self):
        import tempfile

        by_stream = self.streams()
        saved = quality.KINKS_JSON
        try:
            with tempfile.TemporaryDirectory() as directory:
                quality.KINKS_JSON = Path(directory) / "absent.json"
                report = quality.apply_slope_gate(by_stream)
        finally:
            quality.KINKS_JSON = saved
        self.assertFalse(report["available"])
        self.assertEqual(
            {row[quality.SLOPE_GATE_COLUMN] for row in by_stream[("runA", "cam0")]}, {""}
        )

    def test_the_column_is_part_of_the_csv_schema(self):
        self.assertEqual(quality.CSV_FIELDNAMES[-1], quality.SLOPE_GATE_COLUMN)
        self.assertEqual(quality.CSV_FIELDNAMES[:6],
                         ["run", "camera", "ordinal", "mean_luma", "p99_luma", "flag"])


if __name__ == "__main__":
    unittest.main()
