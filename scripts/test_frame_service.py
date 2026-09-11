#!/usr/bin/env python3
"""Focused unit checks for the canonical frame service and bitstream parser.

The bit-level tests use hand-built payloads rather than the recordings, so they
run in milliseconds and would still catch a regression if the source files were
unavailable. A few cheap checks do touch the real files, because resolving run A
against run B depends on the supplied " (1)" naming convention and that is worth
pinning down.
"""

from __future__ import annotations

import unittest

import hevc_bitstream as bitstream
import frame_service as frames


NEEDS_RAW_VIDEO = (
    "the raw HEVC recordings are absent (they are excluded from the submission bundle by policy); this check needs them"
)


class BitReaderTests(unittest.TestCase):
    def test_unsigned_exp_golomb(self) -> None:
        # H.265 section 9.2: "1" is 0, "010" is 1, "011" is 2, "00100" is 3.
        for payload, expected in ((0x80, 0), (0x40, 1), (0x60, 2), (0x20, 3), (0x28, 4)):
            with self.subTest(payload=payload):
                self.assertEqual(bitstream.BitReader(bytes([payload])).ue(), expected)

    def test_signed_exp_golomb_alternates_sign(self) -> None:
        for payload, expected in ((0x80, 0), (0x40, 1), (0x60, -1), (0x20, 2), (0x28, -2)):
            with self.subTest(payload=payload):
                self.assertEqual(bitstream.BitReader(bytes([payload])).se(), expected)

    def test_fixed_width_read(self) -> None:
        reader = bitstream.BitReader(bytes([0b10110010]))
        self.assertEqual(reader.u(3), 0b101)
        self.assertEqual(reader.u(5), 0b10010)

    def test_reading_past_the_end_raises(self) -> None:
        with self.assertRaises(ValueError):
            bitstream.BitReader(bytes([0xFF])).u(16)


class EmulationPreventionTests(unittest.TestCase):
    def test_strips_the_escape_byte(self) -> None:
        self.assertEqual(
            bitstream.remove_emulation_prevention(b"\x00\x00\x03\x01"), b"\x00\x00\x01"
        )

    def test_leaves_ordinary_payloads_untouched(self) -> None:
        payload = b"\x12\x34\x00\x56\x00\x00\x78"
        self.assertEqual(bitstream.remove_emulation_prevention(payload), payload)

    def test_strips_several_escapes(self) -> None:
        self.assertEqual(
            bitstream.remove_emulation_prevention(b"\x00\x00\x03\x00\x00\x00\x03\x02"),
            b"\x00\x00\x00\x00\x00\x02",
        )


def _sps(**overrides) -> bitstream.SequenceParameterSet:
    defaults = dict(
        chroma_format_idc=1,
        coded_width=1440,
        coded_height=1088,
        conformance_window=(0, 0, 0, 4),
        bit_depth_luma=8,
        bit_depth_chroma=8,
        log2_max_pic_order_cnt_lsb=8,
        profile_idc=1,
        level_idc=120,
    )
    defaults.update(overrides)
    return bitstream.SequenceParameterSet(**defaults)


class ConformanceWindowTests(unittest.TestCase):
    def test_crops_eight_luma_rows_for_420(self) -> None:
        # The supplied files code 1440x1088 and display 1440x1080. A pipeline
        # that reads the coded raster gets eight rows of encoder padding.
        sps = _sps()
        self.assertEqual((sps.display_width, sps.display_height), (1440, 1080))
        self.assertEqual(sps.cropped_luma_rows, 8)

    def test_no_window_means_display_equals_coded(self) -> None:
        sps = _sps(conformance_window=(0, 0, 0, 0))
        self.assertEqual((sps.display_width, sps.display_height), (1440, 1088))


def _scan(entries: list[tuple[int, int]]) -> bitstream.StreamScan:
    """Build a scan from (nal_type, picture_order_count) pairs."""

    pictures = [
        bitstream.Picture(
            decode_index=index,
            byte_offset=index * 100,
            byte_size=100,
            nal_type=nal_type,
            slice_type=2 if nal_type in bitstream.IDR_NAL_TYPES else 0,
            picture_order_count=poc,
        )
        for index, (nal_type, poc) in enumerate(entries)
    ]
    return bitstream.StreamScan(
        path=frames.Path("synthetic.hevc"), file_size=1024, sps=_sps(), pictures=pictures
    )


class DisplayOrderTests(unittest.TestCase):
    def test_recovers_display_order_from_a_b_pyramid(self) -> None:
        # The real streams decode as 0, 4, 2, 1, 3, which is the pattern that
        # scrambles 74% of frames if you trust decode order.
        scan = _scan([(19, 0), (1, 4), (0, 2), (0, 1), (0, 3)])
        self.assertEqual(scan.display_order(), [0, 3, 2, 4, 1])
        # Display positions 1, 3 and 4 hold a different picture than the decode
        # index of the same number; positions 0 and 2 happen to coincide.
        self.assertEqual(scan.reordered_picture_count, 3)

    def test_sequential_stream_reports_no_reordering(self) -> None:
        scan = _scan([(19, 0), (1, 1), (1, 2), (1, 3)])
        self.assertEqual(scan.display_order(), [0, 1, 2, 3])
        self.assertEqual(scan.reordered_picture_count, 0)

    def test_order_restarts_at_each_idr(self) -> None:
        scan = _scan([(19, 0), (1, 2), (0, 1), (19, 0), (1, 2), (0, 1)])
        self.assertEqual(scan.display_order(), [0, 2, 1, 3, 5, 4])

    def test_permutation_check_accepts_a_valid_stream(self) -> None:
        _scan([(19, 0), (1, 4), (0, 2), (0, 1), (0, 3)]).verify_display_order_is_permutation()

    def test_permutation_check_rejects_duplicate_order_counts(self) -> None:
        # Two pictures claiming the same output position means frame identity
        # cannot be trusted, and that must fail loudly rather than pick one.
        scan = _scan([(19, 0), (1, 2), (0, 2)])
        with self.assertRaises(ValueError):
            scan.verify_display_order_is_permutation()

    def test_gop_lengths_follow_the_idr_boundaries(self) -> None:
        scan = _scan([(19, 0), (1, 1), (19, 0), (1, 1), (1, 2)])
        self.assertEqual(scan.gop_lengths, [2, 3])


class CacheKeyTests(unittest.TestCase):
    def test_key_changes_with_resolution(self) -> None:
        first = frames.cache_key("runA", "cam0", 448, 336, "abcdef123456")
        second = frames.cache_key("runA", "cam0", 896, 672, "abcdef123456")
        self.assertNotEqual(first, second)

    def test_key_changes_with_source_digest(self) -> None:
        # The old descriptor caches were keyed by frame ordinal alone, so a
        # decode change silently reused arrays built from different pixels.
        first = frames.cache_key("runA", "cam0", 448, 336, "abcdef123456")
        second = frames.cache_key("runA", "cam0", 448, 336, "0000ff999999")
        self.assertNotEqual(first, second)

    def test_key_carries_the_colour_identity(self) -> None:
        self.assertIn(frames.COLOUR_ID, frames.cache_key("runA", "cam0", 448, 336, "abc"))

    def test_key_is_stable_for_identical_inputs(self) -> None:
        arguments = ("runB", "cam5", 448, 336, "abcdef123456")
        self.assertEqual(frames.cache_key(*arguments), frames.cache_key(*arguments))


class GateTests(unittest.TestCase):
    def test_a_failing_gate_without_a_waiver_is_reported_as_failed(self) -> None:
        results: list[frames.GateResult] = []
        frames._gate(results, {}, "some_gate", "runA", False, "detail")
        self.assertEqual(results[0].status, "FAIL")

    def test_a_failing_gate_with_a_waiver_is_reported_as_waived(self) -> None:
        results: list[frames.GateResult] = []
        frames._gate(results, {"some_gate": "known defect"}, "some_gate", "runA", False, "detail")
        self.assertEqual(results[0].status, "waived")
        self.assertEqual(results[0].waiver_reason, "known defect")

    def test_a_waiver_never_turns_a_pass_into_a_waive(self) -> None:
        results: list[frames.GateResult] = []
        frames._gate(results, {"some_gate": "known defect"}, "some_gate", "runA", True, "detail")
        self.assertEqual(results[0].status, "pass")


@unittest.skipUnless(frames.source_available(), NEEDS_RAW_VIDEO)
class SourceResolutionTests(unittest.TestCase):
    def test_runs_resolve_to_different_files(self) -> None:
        self.assertNotEqual(
            frames.source_path("runA", "cam0"), frames.source_path("runB", "cam0")
        )

    def test_cameras_resolve_to_different_files(self) -> None:
        self.assertNotEqual(
            frames.source_path("runA", "cam0"), frames.source_path("runA", "cam5")
        )

    def test_unknown_run_or_camera_raises(self) -> None:
        with self.assertRaises(ValueError):
            frames.source_path("runC", "cam0")
        with self.assertRaises(ValueError):
            frames.source_path("runA", "cam9")

    def test_both_cameras_share_one_sidecar_in_each_run(self) -> None:
        # This is the supplied data's defect, pinned as a test so that a future
        # drop which fixes it shows up as a change rather than passing silently.
        for run in ("runA", "runB"):
            with self.subTest(run=run):
                self.assertEqual(
                    frames.file_sha256(frames.sidecar_path(run, "cam0")),
                    frames.file_sha256(frames.sidecar_path(run, "cam5")),
                )


if __name__ == "__main__":
    unittest.main()
