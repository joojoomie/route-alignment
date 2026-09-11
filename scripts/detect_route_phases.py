#!/usr/bin/env python3
"""Detect the initial stationary, departure, and route-in-motion phases.

The detector is deliberately independent of timestamp sidecar row ordinals. It
works on low-resolution decoded images, preserves the original decoded-image
ordinals, and uses two complementary traditional-CV signals:

* intensity-compensated consecutive-frame difference; and
* phase correlation between edge images.

Each camera produces an apparent-motion onset. The later onset is used as a
conservative run-level departure boundary so that a moving vehicle visible in
only one camera cannot start route matching. This is a run-level gate only; it
does not assert that CAM0 and CAM5 frame ordinals are synchronized.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterator

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "route-alignment-matplotlib"))

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_keyframes"
TASK1_SUMMARY = ROOT / "outputs" / "task1" / "task1_recording_summary.csv"

ANALYSIS_WIDTH = 192
ANALYSIS_HEIGHT = 144
ANALYSIS_FRAMES = 360
DISPLAY_FPS = 10
PIXEL_DIFFERENCE_THRESHOLD = 10.0
TILE_ROWS = 3
TILE_COLUMNS = 4
TILE_CHANGED_FRACTION = 0.035
PHASE_SHIFT_THRESHOLD_PX = 1.0
PHASE_MEDIAN_WINDOW = 3
DIFFERENCE_MEDIAN_WINDOW = 7
PERSISTENCE_WINDOW = 10
PERSISTENCE_REQUIRED = 6
SEARCH_START_FRAME = 30
ROUTE_PERSISTENCE_WINDOW = 12
ROUTE_PERSISTENCE_REQUIRED = 10

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise RuntimeError("ffmpeg is required but was not found on PATH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-frames",
        type=int,
        default=ANALYSIS_FRAMES,
        help="Maximum number of decoded images inspected from each stream.",
    )
    return parser.parse_args()


def first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"None of these expected files exists:\n{rendered}")


def resolve_video_files() -> dict[tuple[str, str], Path]:
    names = {
        "cam0": "cam0_20_yuv420p_output.hevc",
        "cam5": "cam5_20_yuv420p_output.hevc",
    }
    files: dict[tuple[str, str], Path] = {}
    for run in ("runA", "runB"):
        for camera, name in names.items():
            suffix = " (1)" if run == "runB" else ""
            flat_name = name.replace(".hevc", f"{suffix}.hevc")
            files[(run, camera)] = first_existing(ROOT / run / name, ROOT / flat_name)
    return files


def load_task1_counts() -> dict[tuple[str, str], int]:
    with TASK1_SUMMARY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {
        (row["run"], row["camera"]): int(row["canonical_decoded_images"])
        for row in rows
    }
    expected = {(run, camera) for run in ("runA", "runB") for camera in ("cam0", "cam5")}
    if set(counts) != expected:
        raise ValueError(f"Unexpected Task 1 stream set: {sorted(counts)}")
    return counts


def decoded_gray_frames(path: Path, limit: int) -> Iterator[np.ndarray]:
    command = [
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
        f"scale={ANALYSIS_WIDTH}:{ANALYSIS_HEIGHT}:flags=area",
        "-frames:v",
        str(limit),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        raise RuntimeError(f"Could not decode {path}")
    frame_bytes = ANALYSIS_WIDTH * ANALYSIS_HEIGHT
    normal_completion = False
    try:
        while True:
            payload = process.stdout.read(frame_bytes)
            if not payload:
                break
            while len(payload) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(payload))
                if not chunk:
                    break
                payload += chunk
            if len(payload) != frame_bytes:
                raise ValueError(f"Incomplete analysis frame from {path.name}")
            yield np.frombuffer(payload, dtype=np.uint8).reshape(
                ANALYSIS_HEIGHT, ANALYSIS_WIDTH
            ).astype(np.float32)
        stderr_text = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0 or stderr_text:
            raise RuntimeError(
                f"Decoder failed for {path.name} (exit {return_code}): {stderr_text}"
            )
        normal_completion = True
    finally:
        if not normal_completion and process.poll() is None:
            process.kill()
            process.wait()


def edge_image(frame: np.ndarray) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(frame, sigma=0.8)
    horizontal = ndimage.sobel(smoothed, axis=1)
    vertical = ndimage.sobel(smoothed, axis=0)
    return np.hypot(horizontal, vertical)


def tile_motion_coverage(changed: np.ndarray) -> float:
    active = 0
    total = TILE_ROWS * TILE_COLUMNS
    for row in range(TILE_ROWS):
        top = row * ANALYSIS_HEIGHT // TILE_ROWS
        bottom = (row + 1) * ANALYSIS_HEIGHT // TILE_ROWS
        for column in range(TILE_COLUMNS):
            left = column * ANALYSIS_WIDTH // TILE_COLUMNS
            right = (column + 1) * ANALYSIS_WIDTH // TILE_COLUMNS
            if float(changed[top:bottom, left:right].mean()) >= TILE_CHANGED_FRACTION:
                active += 1
    return active / total


def analyse_stream(path: Path, limit: int) -> list[dict[str, float | int]]:
    frames = list(decoded_gray_frames(path, limit))
    if len(frames) < PERSISTENCE_WINDOW + SEARCH_START_FRAME:
        raise ValueError(f"Too few decoded analysis images from {path.name}: {len(frames)}")

    rows: list[dict[str, float | int]] = []
    previous = frames[0]
    previous_edge = edge_image(previous)
    for frame_index, current in enumerate(frames[1:], start=1):
        intensity_delta = current - previous
        intensity_delta -= np.median(intensity_delta)
        changed = np.abs(intensity_delta) > PIXEL_DIFFERENCE_THRESHOLD
        current_edge = edge_image(current)
        shift, error, _ = phase_cross_correlation(
            previous_edge,
            current_edge,
            upsample_factor=1,
            normalization=None,
        )
        rows.append(
            {
                "frame": frame_index,
                "changed_fraction": float(changed.mean()),
                "tile_motion_coverage": tile_motion_coverage(changed),
                "phase_shift_y": float(shift[0]),
                "phase_shift_x": float(shift[1]),
                "phase_shift_magnitude": float(np.linalg.norm(shift)),
                "phase_error": float(error),
            }
        )
        previous = current
        previous_edge = current_edge

    changed_values = np.asarray([float(row["changed_fraction"]) for row in rows])
    phase_values = np.asarray([float(row["phase_shift_magnitude"]) for row in rows])
    smoothed_changed = ndimage.median_filter(
        changed_values, size=DIFFERENCE_MEDIAN_WINDOW, mode="nearest"
    )
    smoothed_phase = ndimage.median_filter(
        phase_values, size=PHASE_MEDIAN_WINDOW, mode="nearest"
    )
    for index, row in enumerate(rows):
        row["smoothed_changed_fraction"] = float(smoothed_changed[index])
        row["smoothed_phase_magnitude"] = float(smoothed_phase[index])
    return rows


def first_sustained_phase_motion(rows: list[dict[str, float | int]]) -> int:
    values = np.asarray([float(row["smoothed_phase_magnitude"]) for row in rows])
    active = values >= PHASE_SHIFT_THRESHOLD_PX
    first_index = max(0, SEARCH_START_FRAME - 1)
    for start in range(first_index, len(active) - PERSISTENCE_WINDOW + 1):
        if int(active[start : start + PERSISTENCE_WINDOW].sum()) >= PERSISTENCE_REQUIRED:
            return int(rows[start]["frame"])
    raise RuntimeError("No sustained departure motion found inside the analysis window")


def first_sustained_route_motion(
    camera_signals: dict[str, list[dict[str, float | int]]],
    transition_start: int,
) -> tuple[int, float]:
    length = min(len(camera_signals["cam0"]), len(camera_signals["cam5"]))
    changed = {
        camera: np.asarray(
            [
                float(row["smoothed_changed_fraction"])
                for row in camera_signals[camera][:length]
            ]
        )
        for camera in ("cam0", "cam5")
    }
    # The minimum is a conservative two-camera gate: a localized moving object
    # in one view cannot by itself declare that the recording vehicle is moving.
    consensus = np.minimum(changed["cam0"], changed["cam5"])
    threshold = float(threshold_otsu(consensus))
    active = consensus >= threshold
    first_index = max(0, transition_start - 1)
    for start in range(first_index, len(active) - ROUTE_PERSISTENCE_WINDOW + 1):
        if (
            int(active[start : start + ROUTE_PERSISTENCE_WINDOW].sum())
            >= ROUTE_PERSISTENCE_REQUIRED
        ):
            return int(camera_signals["cam0"][start]["frame"]), threshold
    raise RuntimeError("No sustained two-camera route motion found inside the analysis window")


def phase_name(frame: int, transition_start: int, route_start: int) -> str:
    if frame < transition_start:
        return "pre_route_stationary"
    if frame < route_start:
        return "departure_transition"
    return "route_in_motion"


def confidence_for_run(
    onsets: dict[str, int],
    signals: dict[str, list[dict[str, float | int]]],
    route_start: int,
) -> tuple[str, str]:
    onset_gap = abs(onsets["cam0"] - onsets["cam5"])
    confirmations: list[bool] = []
    details: list[str] = []
    for camera in ("cam0", "cam5"):
        values = np.asarray(
            [float(row["smoothed_changed_fraction"]) for row in signals[camera]]
        )
        baseline = float(np.median(values[: min(40, len(values))]))
        start_index = max(0, route_start - 1)
        after = float(np.median(values[start_index : start_index + 20]))
        confirmed = after >= max(0.05, baseline * 4.0)
        confirmations.append(confirmed)
        details.append(f"{camera} diff {baseline:.4f}->{after:.4f}")
    confidence = "high" if onset_gap <= 5 and all(confirmations) else "medium"
    note = (
        f"Independent apparent-motion onsets: CAM0 {onsets['cam0']}, "
        f"CAM5 {onsets['cam5']} (gap {onset_gap}); " + ", ".join(details) + "."
    )
    if onset_gap > 5:
        note += " The later camera gate is retained to reduce single-view dynamic-object bias."
    return confidence, note


def save_diagnostic_plot(
    signals: dict[tuple[str, str], list[dict[str, float | int]]],
    boundaries: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex="col")
    colors = {"cam0": "#2563eb", "cam5": "#d97706"}
    for column, run in enumerate(("runA", "runB")):
        boundary = boundaries[run]
        for camera in ("cam0", "cam5"):
            rows = signals[(run, camera)]
            frames = [int(row["frame"]) for row in rows]
            axes[0, column].plot(
                frames,
                [float(row["smoothed_changed_fraction"]) for row in rows],
                color=colors[camera],
                linewidth=1.4,
                label=camera.upper(),
            )
            axes[1, column].plot(
                frames,
                [float(row["smoothed_phase_magnitude"]) for row in rows],
                color=colors[camera],
                linewidth=1.25,
                label=camera.upper(),
            )
        for axis in axes[:, column]:
            axis.axvspan(
                boundary["transition_start"],
                boundary["route_start"],
                color="#facc15",
                alpha=0.22,
                label="departure transition",
            )
            axis.axvline(boundary["route_start"], color="#16a34a", linewidth=1.5)
            axis.grid(alpha=0.25)
        axes[0, column].set_title(f"{run}: intensity-compensated frame difference")
        axes[1, column].set_title(f"{run}: edge phase-correlation shift")
        axes[1, column].set_xlabel("Zero-based decoded-image ordinal")
    axes[0, 0].set_ylabel("Changed-pixel fraction")
    axes[1, 0].set_ylabel("Global shift magnitude (analysis pixels)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    deduplicated = dict(zip(labels, handles))
    figure.legend(
        deduplicated.values(),
        deduplicated.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Automatic route-phase detection (phase boundaries are gates, not cross-camera sync)",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(analysis_frames: int) -> None:
    if analysis_frames <= SEARCH_START_FRAME + PERSISTENCE_WINDOW:
        raise ValueError("The analysis window is too short for persistent-motion detection")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = resolve_video_files()
    counts = load_task1_counts()
    signals: dict[tuple[str, str], list[dict[str, float | int]]] = {}
    onsets: dict[tuple[str, str], int] = {}
    for run in ("runA", "runB"):
        for camera in ("cam0", "cam5"):
            limit = min(analysis_frames, counts[(run, camera)])
            print(f"Analysing {run}/{camera}: first {limit} decoded images", flush=True)
            rows = analyse_stream(files[(run, camera)], limit)
            signals[(run, camera)] = rows
            onsets[(run, camera)] = first_sustained_phase_motion(rows)
            print(
                f"  apparent sustained global motion at frame {onsets[(run, camera)]}",
                flush=True,
            )

    boundaries: dict[str, dict[str, int]] = {}
    route_thresholds: dict[str, float] = {}
    boundary_rows: list[dict[str, object]] = []
    for run in ("runA", "runB"):
        camera_onsets = {camera: onsets[(run, camera)] for camera in ("cam0", "cam5")}
        transition_start = max(camera_onsets.values())
        route_start, route_threshold = first_sustained_route_motion(
            {camera: signals[(run, camera)] for camera in ("cam0", "cam5")},
            transition_start,
        )
        route_thresholds[run] = route_threshold
        confidence, note = confidence_for_run(
            camera_onsets,
            {camera: signals[(run, camera)] for camera in ("cam0", "cam5")},
            route_start,
        )
        boundaries[run] = {
            "stationary_end": transition_start - 1,
            "transition_start": transition_start,
            "transition_end": route_start - 1,
            "route_start": route_start,
        }
        boundary_rows.append(
            {
                "run": run,
                "camera_scope": "run_level_later_of_independent_cam0_cam5_detections",
                "stationary_start": 0,
                "stationary_end": transition_start - 1,
                "transition_start": transition_start,
                "transition_end": route_start - 1,
                "route_start": route_start,
                "cam0_apparent_motion_onset": camera_onsets["cam0"],
                "cam5_apparent_motion_onset": camera_onsets["cam5"],
                "two_camera_changed_fraction_threshold": round(route_threshold, 8),
                "confidence": confidence,
                "method": "edge_phase_correlation_plus_frame_difference_median_filter_and_persistence",
                "notes": note,
            }
        )

    signal_rows: list[dict[str, object]] = []
    for run in ("runA", "runB"):
        boundary = boundaries[run]
        for camera in ("cam0", "cam5"):
            for row in signals[(run, camera)]:
                frame = int(row["frame"])
                signal_rows.append(
                    {
                        "run": run,
                        "camera_id": camera,
                        **row,
                        "route_phase": phase_name(
                            frame, boundary["transition_start"], boundary["route_start"]
                        ),
                    }
                )

    write_csv(
        OUTPUT_DIR / "route_phase_boundaries.csv",
        boundary_rows,
        list(boundary_rows[0]),
    )
    write_csv(
        OUTPUT_DIR / "route_motion_signal.csv",
        signal_rows,
        list(signal_rows[0]),
    )
    save_diagnostic_plot(
        signals,
        boundaries,
        OUTPUT_DIR / "route_phase_diagnostics.png",
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_identity": "zero-based decoded-image ordinal",
        "timestamp_rows_assigned_to_frames": False,
        "source_frames_deleted_or_renumbered": False,
        "cross_camera_sync_claimed": False,
        "analysis_resolution": [ANALYSIS_WIDTH, ANALYSIS_HEIGHT],
        "analysis_frames_requested": analysis_frames,
        "parameters": {
            "pixel_difference_threshold": PIXEL_DIFFERENCE_THRESHOLD,
            "phase_shift_threshold_pixels": PHASE_SHIFT_THRESHOLD_PX,
            "phase_median_window": PHASE_MEDIAN_WINDOW,
            "difference_median_window": DIFFERENCE_MEDIAN_WINDOW,
            "persistence_window": PERSISTENCE_WINDOW,
            "persistence_required": PERSISTENCE_REQUIRED,
            "route_persistence_window": ROUTE_PERSISTENCE_WINDOW,
            "route_persistence_required": ROUTE_PERSISTENCE_REQUIRED,
            "two_camera_changed_fraction_thresholds": route_thresholds,
        },
        "boundaries": boundaries,
    }
    (OUTPUT_DIR / "route_phase_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote automatic route-phase artifacts under outputs/task2_keyframes", flush=True)


if __name__ == "__main__":
    main(parse_args().analysis_frames)
