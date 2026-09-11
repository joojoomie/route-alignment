#!/usr/bin/env python3
"""The single canonical decode path for the supplied recordings.

Every earlier decoder in this project read the 640x480 CRF-24 annotation
proxies rather than the source video, so the matcher operated on a double-lossy
chain: 1440x1080 HEVC -> lanczos -> 640x480 x264 CRF 24 -> lanczos -> 448x336.
The compression cost is modest at 448x336 (90-94% of gradient energy survives),
but the *downscale* is not: run B cam5 yields 669 usable SIFT keypoints at
448x336 against 2,494 at 896x672. That is the measured cause of cam5's thin
sparse-match counts and its failing distributed-support geometry gate.

This module therefore decodes from the HEVC, and it pins the things the stream
leaves undeclared. The colour matrix, primaries, and transfer function are all
absent from these files; only `color_range=tv` is signalled. swscale then picks
a default from the frame height, which means the old two-stage chain converted
the 1080-tall source with BT.709 coefficients and the 480-tall proxy with BT.601
ones. Converting the same frame under the two matrices differs by up to 11/255,
so the RGB a descriptor sees depended on the pipeline stage it came through.
Here the conversion is stated explicitly and recorded in the manifest.

Two other conventions matter downstream:

* frame identity is the zero-based *display-order* ordinal, verified against the
  bitstream by `hevc_bitstream` rather than assumed; and
* cache filenames carry the source hash, resolution, and colour identity, so a
  pixel-pipeline change can never silently mix old and new arrays. The previous
  descriptor caches were keyed by frame ordinal alone.

Sequential decode is the only reliable access order for a container-free Annex-B
stream, so `frames()` decodes in a single pass and callers should batch their
ordinals. `cached_array()` exists for whole-stream work and pays the pass once.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable

import numpy as np

import hevc_bitstream


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "outputs" / "frame_cache"
TASK1_DIR = ROOT / "outputs" / "task1"
CHECKSUM_FILE = ROOT / "SHA256SUMS.txt"
WAIVER_FILE = TASK1_DIR / "integrity_waivers.json"

RUNS = ("runA", "runB")
CAMERAS = ("cam0", "cam5")

# Run A is parked for its last 57 frames: the vehicle has stopped, every frame
# shows the same place, and a one-to-one correspondence there is meaningless.
# This is the ONE place that boundary lives. It is checked against the image
# motion signal by verify(), so if a future export parks somewhere else the
# gate fails instead of every downstream stage quietly using a wrong number.
RUN_A_ROUTE_TAIL_EXCLUSIVE = 2616
TAIL_MOTION_RASTER = (192, 144)
TAIL_STILL_FRACTION_OF_MEDIAN = 0.12   # motion below this share of the route median = parked
TAIL_GATE_TOLERANCE_FRAMES = 8

# Display raster after the SPS conformance window crops 8 luma rows from the
# 1440x1088 coded raster.
DISPLAY_WIDTH = 1440
DISPLAY_HEIGHT = 1080

# Working resolutions, both served from the source. DINOv2 resizes its input to
# 224x224 regardless, so 448x336 loses it nothing; RootSIFT is resolution-bound
# and gets the larger raster.
DESCRIPTOR_SIZE = (448, 336)
GEOMETRY_SIZE = (896, 672)

# The colour conversion this project commits to, since the stream declares none.
# BT.709 matches the source raster height, which is what swscale would have
# chosen for the first stage of the old chain; limited-range luma is expanded to
# full-range RGB, which is the normal convention for RGB output.
COLOUR_MATRIX = "bt709"
COLOUR_INPUT_RANGE = "tv"
COLOUR_OUTPUT_RANGE = "pc"
COLOUR_ID = f"{COLOUR_MATRIX}-{COLOUR_INPUT_RANGE}2{COLOUR_OUTPUT_RANGE}"

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise RuntimeError("ffmpeg is required but was not found on PATH")


def ffmpeg_version() -> str:
    result = subprocess.run(
        [FFMPEG, "-version"], capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()[0].strip()


def first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"None of these expected files exists:\n{rendered}")


def source_path(run: str, camera: str) -> Path:
    """Resolve one recording, accepting either the run/ layout or flat names.

    The supplied download names both runs identically and distinguishes run B
    only by a " (1)" suffix, so this is the one place that convention lives.
    """

    if run not in RUNS:
        raise ValueError(f"Unknown run {run!r}; expected one of {RUNS}")
    if camera not in CAMERAS:
        raise ValueError(f"Unknown camera {camera!r}; expected one of {CAMERAS}")
    name = f"{camera}_20_yuv420p_output.hevc"
    suffix = " (1)" if run == "runB" else ""
    flat_name = name.replace(".hevc", f"{suffix}.hevc")
    return first_existing(ROOT / run / name, ROOT / flat_name)


def source_available(run: str = "runA", camera: str = "cam0") -> bool:
    """Whether the raw recording for one stream is present on this machine.

    The submission bundle excludes the confidential HEVC by policy, so anything
    that needs it must **skip** there rather than error: a check that cannot run
    is not a check that failed. This is the single place that asks, so a test
    and a tool cannot disagree about what "present" means.
    """

    try:
        source_path(run, camera)
    except FileNotFoundError:
        return False
    return True


def sidecar_path(run: str, camera: str) -> Path:
    name = f"{camera}_20_yuv420p_output.hevc.timestamps.txt"
    suffix = " (1)" if run == "runB" else ""
    flat_name = name.replace(".txt", f"{suffix}.txt")
    return first_existing(ROOT / run / name, ROOT / flat_name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def published_checksums() -> dict[str, str]:
    """Read the supplied SHA256SUMS.txt as {relative_path: digest}."""

    if not CHECKSUM_FILE.is_file():
        return {}
    checksums: dict[str, str] = {}
    for line in CHECKSUM_FILE.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            checksums[parts[1].strip()] = parts[0].strip()
    return checksums


def load_waivers() -> dict[str, str]:
    """Integrity gates that are known to fail and have been acknowledged.

    A gate with no waiver raises. A gate with a waiver is reported as waived
    together with its recorded reason, so an acknowledged defect stays visible
    instead of quietly passing.
    """

    if not WAIVER_FILE.is_file():
        return {}
    payload = json.loads(WAIVER_FILE.read_text())
    return {entry["gate"]: entry["reason"] for entry in payload.get("waivers", [])}


@dataclass(frozen=True)
class DecodeManifest:
    run: str
    camera: str
    source_name: str
    source_sha256: str
    checksum_verified: bool
    ffmpeg_version: str
    colour_matrix: str
    colour_input_range: str
    colour_output_range: str
    coded_raster: str
    display_raster: str
    cropped_luma_rows: int
    bitstream_picture_count: int
    decoded_frame_count: int
    reordered_picture_count: int
    frame_identity: str = (
        "zero-based display-order decoded-image ordinal, verified against "
        "reconstructed picture order counts"
    )

    def as_dict(self) -> dict[str, object]:
        payload = {key: getattr(self, key) for key in self.__dataclass_fields__}
        return payload


def cache_key(run: str, camera: str, width: int, height: int, source_digest: str) -> str:
    """Compose a cache identity that changes whenever the pixels would.

    The previous descriptor caches were keyed by frame ordinal only, so editing
    a decode flag reused arrays computed under the old pipeline without error.
    """

    return f"{run}_{camera}_{source_digest[:12]}_{width}x{height}_{COLOUR_ID}"


def _scale_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:flags=lanczos"
        f":in_color_matrix={COLOUR_MATRIX}"
        f":in_range={COLOUR_INPUT_RANGE}"
        f":out_range={COLOUR_OUTPUT_RANGE}"
    )


def _decode_command(path: Path, width: int, height: int) -> list[str]:
    return [
        FFMPEG,
        "-v",
        "error",
        "-nostdin",
        "-hwaccel",
        "none",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-fps_mode",
        "passthrough",
        "-vf",
        _scale_filter(width, height),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]


def iter_frames(
    path: Path, width: int, height: int, stop_after: int | None = None
) -> Iterable[tuple[int, np.ndarray]]:
    """Yield (display_ordinal, HxWx3 uint8) in display order, one decode pass."""

    frame_bytes = width * height * 3
    process = subprocess.Popen(
        _decode_command(path, width, height),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError(f"Could not decode {path}")
    ordinal = 0
    try:
        while True:
            payload = process.stdout.read(frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise ValueError(
                    f"Incomplete decoded frame {ordinal} from {path.name}: "
                    f"{len(payload)} of {frame_bytes} bytes"
                )
            yield ordinal, np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
            ordinal += 1
            if stop_after is not None and ordinal > stop_after:
                break
    finally:
        if process.poll() is None:
            process.kill()
        stderr = process.stderr.read().decode("utf-8", "replace").strip()
        process.stdout.close()
        process.stderr.close()
        process.wait()
        if stderr and stop_after is None:
            raise RuntimeError(f"FFmpeg reported errors decoding {path.name}: {stderr}")


def frames(
    run: str,
    camera: str,
    ordinals: Iterable[int],
    width: int = DESCRIPTOR_SIZE[0],
    height: int = DESCRIPTOR_SIZE[1],
) -> dict[int, np.ndarray]:
    """Decode the requested display ordinals in a single sequential pass.

    Annex-B streams carry no index, so cost scales with the largest ordinal
    requested. Batch your ordinals rather than calling this per frame.
    """

    wanted = set(int(ordinal) for ordinal in ordinals)
    if not wanted:
        return {}
    if min(wanted) < 0:
        raise ValueError("Frame ordinals must be non-negative")
    path = source_path(run, camera)
    highest = max(wanted)
    collected: dict[int, np.ndarray] = {}
    for ordinal, image in iter_frames(path, width, height, stop_after=highest):
        if ordinal in wanted:
            collected[ordinal] = image.copy()
            if len(collected) == len(wanted):
                break
    missing = sorted(wanted - set(collected))
    if missing:
        raise ValueError(
            f"{run}/{camera}: requested ordinals not present in the stream: {missing[:8]}"
        )
    return collected


def cached_array(
    run: str,
    camera: str,
    width: int = DESCRIPTOR_SIZE[0],
    height: int = DESCRIPTOR_SIZE[1],
    force: bool = False,
) -> np.ndarray:
    """Return the whole stream as a memory-mapped (N, H, W, 3) uint8 array.

    Builds the cache on first use. Worth it whenever more than a few hundred
    frames are needed, since random access then costs nothing.
    """

    digest = file_sha256(source_path(run, camera))
    key = cache_key(run, camera, width, height, digest)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / f"{key}.npy"
    if target.is_file() and not force:
        return np.load(target, mmap_mode="r")

    path = source_path(run, camera)
    with tempfile.NamedTemporaryFile(
        dir=CACHE_DIR, prefix=f".{key}-", suffix=".npy", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        stack = [image.copy() for _, image in iter_frames(path, width, height)]
        np.save(temporary_path, np.stack(stack))
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return np.load(target, mmap_mode="r")


def decoded_frame_count(run: str, camera: str) -> int:
    """Count decoded frames with FFmpeg, independently of the bitstream scan."""

    path = source_path(run, camera)
    result = subprocess.run(
        [FFMPEG, "-v", "error", "-stats", "-nostdin", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in result.stderr.replace("\r", "\n").splitlines() if line.startswith("frame=")]
    if not lines:
        raise RuntimeError(f"FFmpeg produced no frame statistics for {path.name}")
    return int(lines[-1].split("=", 1)[1].split()[0])


def decode_manifest(run: str, camera: str) -> DecodeManifest:
    path = source_path(run, camera)
    digest = file_sha256(path)
    published = published_checksums()
    expected = published.get(f"{run}/{camera}_20_yuv420p_output.hevc")
    stream = hevc_bitstream.scan(path)
    return DecodeManifest(
        run=run,
        camera=camera,
        source_name=path.name,
        source_sha256=digest,
        checksum_verified=expected is not None and expected == digest,
        ffmpeg_version=ffmpeg_version(),
        colour_matrix=COLOUR_MATRIX,
        colour_input_range=COLOUR_INPUT_RANGE,
        colour_output_range=COLOUR_OUTPUT_RANGE,
        coded_raster=f"{stream.sps.coded_width}x{stream.sps.coded_height}",
        display_raster=f"{stream.sps.display_width}x{stream.sps.display_height}",
        cropped_luma_rows=stream.sps.cropped_luma_rows,
        bitstream_picture_count=stream.picture_count,
        decoded_frame_count=decoded_frame_count(run, camera),
        reordered_picture_count=stream.reordered_picture_count,
    )


def load_timestamps(path: Path) -> np.ndarray:
    values = np.array(
        [int(line.strip()) for line in path.read_text().splitlines() if line.strip()],
        dtype=np.int64,
    )
    if values.size < 2:
        raise ValueError(f"Expected at least two timestamps in {path}")
    return values


@dataclass
class GateResult:
    gate: str
    subject: str
    passed: bool
    detail: str
    waived: bool = False
    waiver_reason: str = ""

    @property
    def status(self) -> str:
        if self.passed:
            return "pass"
        return "waived" if self.waived else "FAIL"


def _gate(
    results: list[GateResult],
    waivers: dict[str, str],
    gate: str,
    subject: str,
    passed: bool,
    detail: str,
) -> None:
    waived = (not passed) and gate in waivers
    results.append(
        GateResult(
            gate=gate,
            subject=subject,
            passed=passed,
            detail=detail,
            waived=waived,
            waiver_reason=waivers.get(gate, "") if waived else "",
        )
    )


def measured_parked_tail_onset(camera: str) -> int:
    """First frame of the final parked stretch in Run A, from image motion alone."""

    from PIL import Image

    width, height = TAIL_MOTION_RASTER
    previous = None
    motion: list[float] = []
    for _, image in iter_frames(source_path("runA", camera), width, height):
        grey = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32)
        if previous is not None:
            motion.append(float(np.abs(grey - previous).mean()))
        previous = grey
    series = np.array(motion)
    threshold = TAIL_STILL_FRACTION_OF_MEDIAN * float(np.median(series))
    still = series < threshold
    # Walk back from the end while the vehicle is still; the onset is where it stops being.
    onset = len(still)
    while onset > 0 and still[onset - 1]:
        onset -= 1
    return onset + 1  # motion[i] is between frame i and i+1


def verify() -> list[GateResult]:
    """Run every integrity gate over all four recordings and both sidecars."""

    waivers = load_waivers()
    results: list[GateResult] = []
    published = published_checksums()

    for camera in CAMERAS:
        onset = measured_parked_tail_onset(camera)
        _gate(
            results,
            waivers,
            "run_a_parked_tail_matches_route_bound",
            f"runA/{camera}",
            abs(onset - RUN_A_ROUTE_TAIL_EXCLUSIVE) <= TAIL_GATE_TOLERANCE_FRAMES,
            f"measured onset {onset} vs bound {RUN_A_ROUTE_TAIL_EXCLUSIVE} "
            f"(tolerance {TAIL_GATE_TOLERANCE_FRAMES})",
        )

    for run in RUNS:
        for camera in CAMERAS:
            subject = f"{run}/{camera}"
            path = source_path(run, camera)
            digest = file_sha256(path)
            expected = published.get(f"{run}/{camera}_20_yuv420p_output.hevc")
            _gate(
                results,
                waivers,
                "source_checksum_matches_manifest",
                subject,
                expected is not None and expected == digest,
                f"sha256 {digest[:12]} vs published {str(expected)[:12]}",
            )

            stream = hevc_bitstream.scan(path)
            decoded = decoded_frame_count(run, camera)
            _gate(
                results,
                waivers,
                "decoded_count_equals_bitstream_picture_count",
                subject,
                decoded == stream.picture_count,
                f"decoder {decoded} vs bitstream {stream.picture_count}",
            )

            try:
                stream.verify_display_order_is_permutation()
                permutation_ok, permutation_detail = True, (
                    f"{stream.reordered_picture_count} of {stream.picture_count} pictures "
                    f"reordered ({stream.reordered_picture_count / stream.picture_count:.1%})"
                )
            except ValueError as error:
                permutation_ok, permutation_detail = False, str(error)
            _gate(
                results,
                waivers,
                "display_order_is_permutation",
                subject,
                permutation_ok,
                permutation_detail,
            )

            _gate(
                results,
                waivers,
                "display_raster_is_1440x1080",
                subject,
                (stream.sps.display_width, stream.sps.display_height)
                == (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                f"coded {stream.sps.coded_width}x{stream.sps.coded_height}, "
                f"display {stream.sps.display_width}x{stream.sps.display_height}, "
                f"{stream.sps.cropped_luma_rows} luma rows cropped",
            )

    for run in RUNS:
        timestamps = load_timestamps(sidecar_path(run, "cam0"))
        deltas = np.diff(timestamps)
        _gate(
            results,
            waivers,
            "sidecar_strictly_increasing",
            run,
            bool((deltas > 0).all()),
            f"{timestamps.size} rows, min delta {deltas.min() / 1e6:.3f} ms",
        )
        _gate(
            results,
            waivers,
            "sidecar_has_no_duplicates",
            run,
            timestamps.size == len(set(timestamps.tolist())),
            f"{timestamps.size - len(set(timestamps.tolist()))} duplicates",
        )
        cam0_digest = file_sha256(sidecar_path(run, "cam0"))
        cam5_digest = file_sha256(sidecar_path(run, "cam5"))
        _gate(
            results,
            waivers,
            "cameras_do_not_share_a_sidecar",
            run,
            cam0_digest != cam5_digest,
            f"cam0 sha256 {cam0_digest[:12]}, cam5 sha256 {cam5_digest[:12]}",
        )

    return results


def write_bitstream_summary() -> Path:
    """Emit the Task 1 bitstream inventory as CSV."""

    import csv

    TASK1_DIR.mkdir(parents=True, exist_ok=True)
    target = TASK1_DIR / "task1_bitstream_summary.csv"
    rows = []
    for run in RUNS:
        for camera in CAMERAS:
            stream = hevc_bitstream.scan(source_path(run, camera))
            summary = stream.summary()
            rows.append(
                {
                    "run": run,
                    "camera": camera,
                    "file": summary["file"],
                    "file_size_bytes": summary["file_size_bytes"],
                    "file_size_is_4096_block_aligned": summary["file_size_is_4096_block_aligned"],
                    "bitstream_picture_count": summary["picture_count"],
                    "coded_raster": summary["coded_raster"],
                    "display_raster": summary["display_raster"],
                    "cropped_luma_rows": summary["conformance_window_cropped_luma_rows"],
                    "slice_type_I": summary["slice_type_counts"].get("I", 0),
                    "slice_type_P": summary["slice_type_counts"].get("P", 0),
                    "slice_type_B": summary["slice_type_counts"].get("B", 0),
                    "idr_count": summary["idr_count"],
                    "idr_interval_frames": summary["gop_lengths"][0] if summary["gop_lengths"] else "",
                    "final_gop_length": summary["gop_lengths"][-1] if summary["gop_lengths"] else "",
                    "reordered_picture_count": summary["reordered_picture_count"],
                    "reordered_picture_fraction": round(summary["reordered_picture_fraction"], 6),
                    "per_picture_sei": ";".join(
                        f"{name}={count}" for name, count in summary["sei_payload_counts"].items()
                    ),
                }
            )
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-all", action="store_true", help="Run every integrity gate.")
    parser.add_argument("--manifest", nargs=2, metavar=("RUN", "CAMERA"))
    parser.add_argument("--checksum", nargs=2, metavar=("RUN", "CAMERA"))
    parser.add_argument("--ordinals", type=str, default="0,100,2000")
    parser.add_argument(
        "--write-bitstream-summary",
        action="store_true",
        help="Write outputs/task1/task1_bitstream_summary.csv.",
    )
    arguments = parser.parse_args()

    if arguments.verify_all:
        results = verify()
        width = max(len(item.gate) for item in results)
        failures = 0
        for item in results:
            if item.status == "FAIL":
                failures += 1
            marker = {"pass": "ok  ", "waived": "WAIV", "FAIL": "FAIL"}[item.status]
            print(f"[{marker}] {item.gate:<{width}}  {item.subject:<11} {item.detail}")
            if item.waived:
                print(f"         waived: {item.waiver_reason}")
        print()
        waived = sum(1 for item in results if item.waived)
        print(
            f"{len(results)} gates: {len(results) - failures - waived} passed, "
            f"{waived} waived, {failures} failed"
        )
        if failures:
            raise SystemExit(1)
        return

    if arguments.manifest:
        run, camera = arguments.manifest
        print(json.dumps(decode_manifest(run, camera).as_dict(), indent=2))
        return

    if arguments.checksum:
        run, camera = arguments.checksum
        ordinals = [int(part) for part in arguments.ordinals.split(",") if part.strip()]
        images = frames(run, camera, ordinals)
        for ordinal in sorted(images):
            digest = hashlib.sha256(images[ordinal].tobytes()).hexdigest()
            print(f"{run}/{camera} ordinal {ordinal}: {digest[:32]}")
        return

    if arguments.write_bitstream_summary:
        target = write_bitstream_summary()
        print(f"Wrote {target.relative_to(ROOT)}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
