#!/usr/bin/env python3
"""Parse the supplied HEVC Annex-B elementary streams without decoding them.

The Task 2 matcher addresses frames by zero-based *display-order* ordinal. That
convention is only safe if something proves the decoder actually reordered the
stream, because these recordings use a four-frame B-pyramid: the bitstream order
of the first pictures is 0, 4, 2, 1, 3, 8, 6, 5, 7, 12, and 1,989 of run A
cam0's 2,674 pictures sit at a different index in decode order than in display
order. Any tool that enumerates NAL units or packets and calls the n-th one
"frame n" therefore produces a locally scrambled sequence, which is invisible to
the eye and the same magnitude as the alignment error being measured.

This module reads the container-free stream directly so that the claim can be
checked rather than assumed. It reports the picture count independently of
libavcodec (a cheap, strong cross-check on the decoder), reconstructs the
picture order count so the reordering can be verified, and exposes the SPS
raster and conformance window so the 1440x1088 coded raster is never confused
with the 1440x1080 display raster.

Nothing here decodes pixels; it is pure byte and bit parsing over the NAL
structure, so it needs no codec library and runs in a second per file.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass, field
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# HEVC NAL unit type numbers (ITU-T H.265 Table 7-1). Only the ones this module
# reasons about are named; anything else is reported by number.
NAL_TYPE_NAMES = {
    0: "TRAIL_N",
    1: "TRAIL_R",
    2: "TSA_N",
    3: "TSA_R",
    4: "STSA_N",
    5: "STSA_R",
    6: "RADL_N",
    7: "RADL_R",
    8: "RASL_N",
    9: "RASL_R",
    16: "BLA_W_LP",
    17: "BLA_W_RADL",
    18: "BLA_N_LP",
    19: "IDR_W_RADL",
    20: "IDR_N_LP",
    21: "CRA_NUT",
    32: "VPS",
    33: "SPS",
    34: "PPS",
    35: "AUD",
    36: "EOS",
    37: "EOB",
    38: "FD",
    39: "PREFIX_SEI",
    40: "SUFFIX_SEI",
}

VCL_NAL_TYPES = frozenset(range(0, 32))
IRAP_NAL_TYPES = frozenset(range(16, 24))
IDR_NAL_TYPES = frozenset((19, 20))

SLICE_TYPE_NAMES = {0: "B", 1: "P", 2: "I"}

# Chroma subsampling multipliers for the conformance window, indexed by
# chroma_format_idc. The supplied files are 4:2:0 (idc 1), so a cropped unit is
# two luma samples tall.
SUBHEIGHT_BY_CHROMA_FORMAT = {0: 1, 1: 2, 2: 1, 3: 1}
SUBWIDTH_BY_CHROMA_FORMAT = {0: 1, 1: 2, 2: 2, 3: 1}


class BitReader:
    """Minimal MSB-first bit reader over an unescaped RBSP payload."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0

    def u(self, bit_count: int) -> int:
        value = 0
        for _ in range(bit_count):
            byte_index = self._position >> 3
            if byte_index >= len(self._data):
                raise ValueError("Bit reader ran past the end of the RBSP payload")
            bit = (self._data[byte_index] >> (7 - (self._position & 7))) & 1
            value = (value << 1) | bit
            self._position += 1
        return value

    def ue(self) -> int:
        """Unsigned Exp-Golomb code, H.265 section 9.2."""

        leading_zeros = 0
        while self.u(1) == 0:
            leading_zeros += 1
            if leading_zeros > 32:
                raise ValueError("Malformed Exp-Golomb code")
        if leading_zeros == 0:
            return 0
        return (1 << leading_zeros) - 1 + self.u(leading_zeros)

    def se(self) -> int:
        """Signed Exp-Golomb code, H.265 section 9.2."""

        value = self.ue()
        return (value + 1) // 2 if value % 2 else -(value // 2)


def remove_emulation_prevention(payload: bytes) -> bytes:
    """Strip 0x03 emulation-prevention bytes to recover the raw RBSP."""

    output = bytearray()
    index = 0
    length = len(payload)
    while index < length:
        if (
            index + 2 < length
            and payload[index] == 0
            and payload[index + 1] == 0
            and payload[index + 2] == 3
        ):
            output += payload[index : index + 2]
            index += 3
        else:
            output.append(payload[index])
            index += 1
    return bytes(output)


@dataclass(frozen=True)
class SequenceParameterSet:
    chroma_format_idc: int
    coded_width: int
    coded_height: int
    conformance_window: tuple[int, int, int, int]
    bit_depth_luma: int
    bit_depth_chroma: int
    log2_max_pic_order_cnt_lsb: int
    profile_idc: int
    level_idc: int

    @property
    def display_width(self) -> int:
        left, right, _, _ = self.conformance_window
        subwidth = SUBWIDTH_BY_CHROMA_FORMAT.get(self.chroma_format_idc, 1)
        return self.coded_width - subwidth * (left + right)

    @property
    def display_height(self) -> int:
        _, _, top, bottom = self.conformance_window
        subheight = SUBHEIGHT_BY_CHROMA_FORMAT.get(self.chroma_format_idc, 1)
        return self.coded_height - subheight * (top + bottom)

    @property
    def cropped_luma_rows(self) -> int:
        return self.coded_height - self.display_height


@dataclass(frozen=True)
class Picture:
    decode_index: int
    byte_offset: int
    byte_size: int
    nal_type: int
    slice_type: int
    picture_order_count: int

    @property
    def is_idr(self) -> bool:
        return self.nal_type in IDR_NAL_TYPES

    @property
    def slice_type_name(self) -> str:
        return SLICE_TYPE_NAMES.get(self.slice_type, str(self.slice_type))


@dataclass
class StreamScan:
    path: Path
    file_size: int
    sps: SequenceParameterSet
    pictures: list[Picture]
    nal_type_counts: dict[str, int] = field(default_factory=dict)
    sei_payload_counts: dict[str, int] = field(default_factory=dict)

    @property
    def picture_count(self) -> int:
        return len(self.pictures)

    @property
    def idr_indices(self) -> list[int]:
        return [picture.decode_index for picture in self.pictures if picture.is_idr]

    @property
    def gop_lengths(self) -> list[int]:
        boundaries = self.idr_indices
        if not boundaries:
            return []
        ends = boundaries[1:] + [self.picture_count]
        return [end - start for start, end in zip(boundaries, ends)]

    @property
    def slice_type_counts(self) -> dict[str, int]:
        counter = collections.Counter(
            picture.slice_type_name for picture in self.pictures
        )
        return dict(sorted(counter.items()))

    def display_order(self) -> list[int]:
        """Decode indices sorted into display order, GOP by GOP.

        Picture order count restarts at each IDR, so reordering is resolved
        within a GOP and the GOPs are concatenated in decode order.
        """

        boundaries = self.idr_indices
        if not boundaries:
            raise ValueError(f"{self.path.name} contains no IDR picture")
        if boundaries[0] != 0:
            raise ValueError(f"{self.path.name} does not begin with an IDR picture")
        ends = boundaries[1:] + [self.picture_count]
        order: list[int] = []
        for start, end in zip(boundaries, ends):
            segment = sorted(
                range(start, end),
                key=lambda index: self.pictures[index].picture_order_count,
            )
            order.extend(segment)
        return order

    @property
    def reordered_picture_count(self) -> int:
        """How many pictures sit at a different index in the two orders."""

        return sum(
            1
            for display_index, decode_index in enumerate(self.display_order())
            if display_index != decode_index
        )

    def verify_display_order_is_permutation(self) -> None:
        """Fail if display order is not a clean permutation of 0..N-1.

        This is the gate that makes the "zero-based display-order ordinal"
        frame identity checkable rather than assumed.
        """

        order = self.display_order()
        if sorted(order) != list(range(self.picture_count)):
            raise ValueError(
                f"{self.path.name}: display order is not a permutation of "
                f"0..{self.picture_count - 1}; frame identity cannot be trusted"
            )
        boundaries = self.idr_indices
        ends = boundaries[1:] + [self.picture_count]
        for start, end in zip(boundaries, ends):
            counts = [self.pictures[index].picture_order_count for index in range(start, end)]
            if sorted(counts) != sorted(set(counts)):
                raise ValueError(
                    f"{self.path.name}: duplicate picture order counts in the GOP "
                    f"starting at decode index {start}"
                )

    def summary(self) -> dict[str, object]:
        return {
            "file": self.path.name,
            "file_size_bytes": self.file_size,
            "file_size_is_4096_block_aligned": self.file_size % 4096 == 0,
            "picture_count": self.picture_count,
            "coded_raster": f"{self.sps.coded_width}x{self.sps.coded_height}",
            "display_raster": f"{self.sps.display_width}x{self.sps.display_height}",
            "conformance_window_cropped_luma_rows": self.sps.cropped_luma_rows,
            "chroma_format_idc": self.sps.chroma_format_idc,
            "bit_depth_luma": self.sps.bit_depth_luma,
            "slice_type_counts": self.slice_type_counts,
            "idr_count": len(self.idr_indices),
            "gop_lengths": self.gop_lengths,
            "reordered_picture_count": self.reordered_picture_count,
            "reordered_picture_fraction": (
                self.reordered_picture_count / self.picture_count
                if self.picture_count
                else 0.0
            ),
            "nal_type_counts": self.nal_type_counts,
            "sei_payload_counts": self.sei_payload_counts,
        }


def parse_sequence_parameter_set(rbsp: bytes) -> SequenceParameterSet:
    """Parse seq_parameter_set_rbsp up to the fields this project needs."""

    reader = BitReader(rbsp)
    reader.u(4)  # sps_video_parameter_set_id
    max_sub_layers_minus1 = reader.u(3)
    reader.u(1)  # sps_temporal_id_nesting_flag

    # profile_tier_level(1, sps_max_sub_layers_minus1)
    reader.u(2)  # general_profile_space
    reader.u(1)  # general_tier_flag
    profile_idc = reader.u(5)
    reader.u(32)  # general_profile_compatibility_flag[32]
    reader.u(4)  # progressive/interlaced/non-packed/frame-only constraint flags
    reader.u(43)  # general_reserved_zero_43bits
    reader.u(1)  # general_inbld_flag or reserved
    level_idc = reader.u(8)
    sub_layer_profile_present: list[int] = []
    sub_layer_level_present: list[int] = []
    for _ in range(max_sub_layers_minus1):
        sub_layer_profile_present.append(reader.u(1))
        sub_layer_level_present.append(reader.u(1))
    if max_sub_layers_minus1 > 0:
        for _ in range(max_sub_layers_minus1, 8):
            reader.u(2)  # reserved_zero_2bits
    for index in range(max_sub_layers_minus1):
        if sub_layer_profile_present[index]:
            reader.u(2)
            reader.u(1)
            reader.u(5)
            reader.u(32)
            reader.u(4)
            reader.u(43)
            reader.u(1)
        if sub_layer_level_present[index]:
            reader.u(8)

    reader.ue()  # sps_seq_parameter_set_id
    chroma_format_idc = reader.ue()
    if chroma_format_idc == 3:
        reader.u(1)  # separate_colour_plane_flag
    coded_width = reader.ue()
    coded_height = reader.ue()
    conformance_window = (0, 0, 0, 0)
    if reader.u(1):  # conformance_window_flag
        conformance_window = (reader.ue(), reader.ue(), reader.ue(), reader.ue())
    bit_depth_luma = reader.ue() + 8
    bit_depth_chroma = reader.ue() + 8
    log2_max_pic_order_cnt_lsb = reader.ue() + 4

    return SequenceParameterSet(
        chroma_format_idc=chroma_format_idc,
        coded_width=coded_width,
        coded_height=coded_height,
        conformance_window=conformance_window,
        bit_depth_luma=bit_depth_luma,
        bit_depth_chroma=bit_depth_chroma,
        log2_max_pic_order_cnt_lsb=log2_max_pic_order_cnt_lsb,
        profile_idc=profile_idc,
        level_idc=level_idc,
    )


def parse_picture_parameter_set(rbsp: bytes) -> dict[str, int]:
    """Parse the few PPS fields needed to locate slice_type in a slice header."""

    reader = BitReader(rbsp)
    reader.ue()  # pps_pic_parameter_set_id
    reader.ue()  # pps_seq_parameter_set_id
    dependent_slice_segments_enabled_flag = reader.u(1)
    output_flag_present_flag = reader.u(1)
    num_extra_slice_header_bits = reader.u(3)
    return {
        "dependent_slice_segments_enabled_flag": dependent_slice_segments_enabled_flag,
        "output_flag_present_flag": output_flag_present_flag,
        "num_extra_slice_header_bits": num_extra_slice_header_bits,
    }


def _find_nal_units(buffer: bytes) -> list[tuple[int, int, int]]:
    """Locate Annex-B NAL units as (start_offset, payload_offset, end_offset)."""

    starts: list[tuple[int, int]] = []
    index = buffer.find(b"\x00\x00\x01", 0)
    while index != -1:
        has_four_byte_prefix = index >= 1 and buffer[index - 1] == 0
        start = index - 1 if has_four_byte_prefix else index
        starts.append((start, index + 3))
        index = buffer.find(b"\x00\x00\x01", index + 3)
    units: list[tuple[int, int, int]] = []
    for position, (start, payload) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(buffer)
        units.append((start, payload, end))
    return units


def _reconstruct_picture_order_count(
    nal_type: int,
    pic_order_cnt_lsb: int | None,
    previous_msb: int,
    previous_lsb: int,
    max_lsb: int,
) -> tuple[int, int, int]:
    """Apply H.265 section 8.3.1 to recover a full POC from its LSB."""

    if nal_type in IDR_NAL_TYPES:
        return 0, 0, 0
    assert pic_order_cnt_lsb is not None
    half = max_lsb // 2
    if pic_order_cnt_lsb < previous_lsb and (previous_lsb - pic_order_cnt_lsb) >= half:
        msb = previous_msb + max_lsb
    elif pic_order_cnt_lsb > previous_lsb and (pic_order_cnt_lsb - previous_lsb) > half:
        msb = previous_msb - max_lsb
    else:
        msb = previous_msb
    return msb + pic_order_cnt_lsb, msb, pic_order_cnt_lsb


def scan(path: Path) -> StreamScan:
    """Parse one Annex-B stream into its pictures, SPS, and NAL statistics."""

    buffer = path.read_bytes()
    units = _find_nal_units(buffer)
    if not units:
        raise ValueError(f"{path.name} contains no Annex-B start code")

    nal_counter: collections.Counter[str] = collections.Counter()
    sei_counter: collections.Counter[str] = collections.Counter()
    sps: SequenceParameterSet | None = None
    pps: dict[str, int] | None = None
    raw_pictures: list[tuple[int, int, int, int, int | None]] = []

    for start, payload, end in units:
        if end - payload < 2:
            continue
        nal_type = (buffer[payload] >> 1) & 0x3F
        nal_counter[NAL_TYPE_NAMES.get(nal_type, str(nal_type))] += 1

        if nal_type == 33 and sps is None:
            sps = parse_sequence_parameter_set(
                remove_emulation_prevention(buffer[payload + 2 : end])
            )
        elif nal_type == 34 and pps is None:
            pps = parse_picture_parameter_set(
                remove_emulation_prevention(buffer[payload + 2 : end])
            )
        elif nal_type == 39:
            sei_counter[_first_sei_payload_type(buffer[payload + 2 : end])] += 1
        elif nal_type in VCL_NAL_TYPES:
            if sps is None or pps is None:
                raise ValueError(
                    f"{path.name}: slice NAL encountered before SPS/PPS were parsed"
                )
            header = remove_emulation_prevention(buffer[payload + 2 : min(end, payload + 64)])
            reader = BitReader(header)
            if reader.u(1) != 1:  # first_slice_segment_in_pic_flag
                continue  # a dependent or later slice segment of the same picture
            if nal_type in IRAP_NAL_TYPES:
                reader.u(1)  # no_output_of_prior_pics_flag
            reader.ue()  # slice_pic_parameter_set_id
            for _ in range(pps["num_extra_slice_header_bits"]):
                reader.u(1)
            slice_type = reader.ue()
            if pps["output_flag_present_flag"]:
                reader.u(1)  # pic_output_flag
            pic_order_cnt_lsb = (
                None
                if nal_type in IDR_NAL_TYPES
                else reader.u(sps.log2_max_pic_order_cnt_lsb)
            )
            raw_pictures.append((start, end - start, nal_type, slice_type, pic_order_cnt_lsb))

    if sps is None:
        raise ValueError(f"{path.name} contains no sequence parameter set")
    if not raw_pictures:
        raise ValueError(f"{path.name} contains no coded picture")

    max_lsb = 1 << sps.log2_max_pic_order_cnt_lsb
    previous_msb = 0
    previous_lsb = 0
    pictures: list[Picture] = []
    for decode_index, (offset, size, nal_type, slice_type, lsb) in enumerate(raw_pictures):
        poc, previous_msb, previous_lsb = _reconstruct_picture_order_count(
            nal_type, lsb, previous_msb, previous_lsb, max_lsb
        )
        pictures.append(
            Picture(
                decode_index=decode_index,
                byte_offset=offset,
                byte_size=size,
                nal_type=nal_type,
                slice_type=slice_type,
                picture_order_count=poc,
            )
        )

    return StreamScan(
        path=path,
        file_size=len(buffer),
        sps=sps,
        pictures=pictures,
        nal_type_counts=dict(sorted(nal_counter.items())),
        sei_payload_counts=dict(sorted(sei_counter.items())),
    )


def _first_sei_payload_type(payload: bytes) -> str:
    """Name the first SEI message in a prefix SEI NAL, for inventory purposes."""

    names = {0: "buffering_period", 1: "pic_timing", 4: "user_data_registered", 5: "user_data_unregistered", 136: "time_code"}
    rbsp = remove_emulation_prevention(payload)
    index = 0
    payload_type = 0
    try:
        while index < len(rbsp) and rbsp[index] == 0xFF:
            payload_type += 255
            index += 1
        if index >= len(rbsp):
            return "unknown"
        payload_type += rbsp[index]
    except IndexError:
        return "unknown"
    return names.get(payload_type, f"type_{payload_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="HEVC Annex-B files to scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    arguments = parser.parse_args()

    summaries = []
    for path in arguments.paths:
        stream = scan(path)
        stream.verify_display_order_is_permutation()
        summaries.append(stream.summary())

    if arguments.json:
        print(json.dumps(summaries, indent=2))
        return
    for summary in summaries:
        print(f"== {summary['file']}")
        for key, value in summary.items():
            if key == "file":
                continue
            print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
