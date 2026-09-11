#!/usr/bin/env python3
"""Render side-by-side comparison videos of the aligned runs.

One video per camera. Each output frame is a 1280x1060 composite:

    +---------------------------+---------------------------+
    | runA frame (640x480)      | runB mapped frame         |   guide overlays
    +---------------------------+---------------------------+
    | 50/50 blend               | absolute difference       |   pose-compensated
    +---------------------------+---------------------------+
    | info bar (camera, ordinals, status, tier, confidence)  |   60 px
    +--------------------------------------------------------+
    | alignment strip: far-field NCC across the whole route  |   40 px
    +--------------------------------------------------------+

The point of the guides is that a human can *see* whether the mapping holds:
identical grids on both top panels mean structure that sits on a line in A
should sit on the same line in B, and the alignment strip shows at a glance
where along the route that agreement is strong.

Usage
-----
    PYTHONPATH=scripts MPLCONFIGDIR=/tmp/route-alignment-mpl python3.11 \
        scripts/render_aligned_video.py --camera cam0 --stage all

Stages: ``stats`` (NCC pre-pass only), ``preview`` (three sample PNGs),
``full``, ``highlights``, ``all``.

Writes only to outputs/videos/ and to the scratch preview directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import frame_service

ROOT = Path(__file__).resolve().parents[1]
MAPPING_DIR = ROOT / "outputs" / "task2_unified"
POSE_JSON = ROOT / "outputs" / "task2_pose_offset" / "pose_offset_diagnostic.json"
VIDEO_DIR = ROOT / "outputs" / "videos"
# Intermediate arrays and preview stills for the `stats` and `preview` stages.
# These are working files, not deliverables, so they default inside the repo's
# own outputs tree rather than to a machine-specific temporary directory; set
# RA_VIDEO_SCRATCH to send them elsewhere (a RAM disk, a scratch volume).
SCRATCH = Path(os.environ.get("RA_VIDEO_SCRATCH", str(VIDEO_DIR / "_scratch")))

# ---------------------------------------------------------------- geometry --

PANEL_W, PANEL_H = 640, 480
CANVAS_W = PANEL_W * 2
INFO_H = 60
STRIP_H = 40
CANVAS_H = PANEL_H * 2 + INFO_H + STRIP_H  # 1060
INFO_Y = PANEL_H * 2
STRIP_Y = INFO_Y + INFO_H

FPS = 10
RUN_A_FRAMES = 2674  # display ordinals 0..2673
GRID_STEP = 80
TICK_STEP = 40

# The pose offset was measured on the 896x672 geometry raster.
POSE_RASTER = (896, 672)
PANEL_SCALE = PANEL_W / POSE_RASTER[0]  # 0.714286; identical for height

# NCC pre-pass raster.
SMALL_W, SMALL_H = 160, 120
BAND_LO, BAND_HI = 0.12, 0.62  # far-field band, fraction of frame height

# Alignment-strip colour scale (documented on the strip itself).
NCC_RED, NCC_GREEN = 0.05, 0.35
NCC_FULL_BAR = 0.50  # far-field NCC that fills the strip bar to full height

RUN_LABEL = {"runA": "runA 2025-02-28", "runB": "runB 2024-04-13"}
CAM_LABEL = {"cam0": "cam0 (left-facing)", "cam5": "cam5 (right-facing)"}

# Regions called out in the brief, used to pick highlight segments.
HARD_REGION = {"cam0": (1050, 1450), "cam5": (1000, 1450)}  # runA ordinals
DARK_STRETCH_B = {"cam0": (1460, 1636), "cam5": (345, 493)}  # runB ordinals
OCCLUSION_A = (327, 356)  # runA partners of the confirmed runB cam0 occlusion
LOOP_CLOSURE_A = (2560, 2616)
SEGMENT_FRAMES = 100  # 10 s at 10 fps
TITLE_FRAMES = 10  # 1 s


# ------------------------------------------------------------------ fonts --


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_mono(size: int) -> ImageFont.FreeTypeFont:
    for candidate in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return _load_font(size)


FONT_SMALL = _load_font(13)
FONT_PANEL = _load_font(16)
FONT_INFO = _load_mono(16)
FONT_TINY = _load_font(11)
FONT_TITLE = _load_font(42)
FONT_SUB = _load_font(22)


# ------------------------------------------------------------------- data --


@dataclass
class Row:
    a: int
    b: int | None
    confidence: float
    tier: str
    status: str


def load_mapping(camera: str) -> list[Row]:
    path = MAPPING_DIR / camera / "frame_mapping_unified.csv"
    by_a: dict[int, Row] = {}
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record.get("camera_id") and record["camera_id"] != camera:
                continue
            a = int(record["runA_frame"])
            raw_b = (record.get("runB_frame") or "").strip()
            try:
                confidence = float(record.get("confidence") or "nan")
            except ValueError:
                confidence = float("nan")
            by_a[a] = Row(
                a=a,
                b=int(raw_b) if raw_b else None,
                confidence=confidence,
                tier=(record.get("tier") or "").strip(),
                status=(record.get("status") or "").strip(),
            )
    # The brief asks for the whole runA range even where the CSV stops.
    return [
        by_a.get(a, Row(a=a, b=None, confidence=float("nan"), tier="", status="not_in_mapping"))
        for a in range(RUN_A_FRAMES)
    ]


def pose_offset(camera: str) -> tuple[float, float]:
    """Measured (dx, dy) in panel pixels: runB scene sits at A + (dx, dy)."""

    diagnostic = json.loads(POSE_JSON.read_text())
    convention = diagnostic["cameras"][camera]["matcher_convention"]
    dx = float(convention["dx_px"]["median"]) * PANEL_SCALE
    dy = float(convention["dy_px"]["median"]) * PANEL_SCALE
    return dx, dy


# ------------------------------------------------------------- decode/NCC --


def decode_stream(run: str, camera: str, capacity: int) -> np.ndarray:
    path = frame_service.source_path(run, camera)
    buffer = np.empty((capacity, PANEL_H, PANEL_W, 3), dtype=np.uint8)
    count = 0
    for ordinal, image in frame_service.iter_frames(path, PANEL_W, PANEL_H):
        if ordinal >= capacity:
            raise RuntimeError(f"{run}/{camera}: more frames than the {capacity} reserved")
        buffer[ordinal] = image
        count = ordinal + 1
    print(f"  decoded {run}/{camera}: {count} frames at {PANEL_W}x{PANEL_H}", flush=True)
    return buffer[:count]


def gradient_stack(frames: np.ndarray) -> np.ndarray:
    """Greyscale gradient magnitude at 160x120 for every frame (float32)."""

    n = frames.shape[0]
    fy, fx = PANEL_H // SMALL_H, PANEL_W // SMALL_W  # 4, 4
    out = np.empty((n, SMALL_H, SMALL_W), dtype=np.float32)
    chunk = 128
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = frames[start:stop].astype(np.float32)
        grey = block[..., 0] * 0.299 + block[..., 1] * 0.587 + block[..., 2] * 0.114
        small = grey.reshape(stop - start, SMALL_H, fy, SMALL_W, fx).mean(axis=(2, 4))
        gx = np.zeros_like(small)
        gy = np.zeros_like(small)
        gx[:, :, 1:-1] = small[:, :, 2:] - small[:, :, :-2]
        gy[:, 1:-1, :] = small[:, 2:, :] - small[:, :-2, :]
        out[start:stop] = np.sqrt(gx * gx + gy * gy)
    return out


def shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate by (dx, dy) pixels, filling the vacated border with black."""

    if dx == 0 and dy == 0:
        return image
    out = np.zeros_like(image)
    h, w = image.shape[:2]
    src_y0, dst_y0 = (0, dy) if dy >= 0 else (-dy, 0)
    src_x0, dst_x0 = (0, dx) if dx >= 0 else (-dx, 0)
    height = h - abs(dy)
    width = w - abs(dx)
    out[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = image[
        src_y0 : src_y0 + height, src_x0 : src_x0 + width
    ]
    return out


def band_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of two far-field bands."""

    lo, hi = int(round(BAND_LO * SMALL_H)), int(round(BAND_HI * SMALL_H))
    x = a[lo:hi].ravel().astype(np.float64)
    y = b[lo:hi].ravel().astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator <= 1e-9:
        return 0.0
    return float(x @ y) / denominator


def compute_ncc(
    rows: list[Row],
    grad_a: np.ndarray,
    grad_b: np.ndarray,
    shift_small: tuple[int, int],
) -> np.ndarray:
    values = np.full(len(rows), np.nan, dtype=np.float32)
    sx, sy = shift_small
    for index, row in enumerate(rows):
        if row.b is None or row.a >= grad_a.shape[0] or row.b >= grad_b.shape[0]:
            continue
        b_small = shift_image(grad_b[row.b], sx, sy)
        values[index] = band_ncc(grad_a[row.a], b_small)
    return values


def weak_accepted_runs(
    rows: list[Row], ncc: np.ndarray, threshold: float = NCC_RED, minimum: int = 5
) -> list[dict]:
    """Stretches the mapping accepted where the picture disagrees with it.

    A run of at least ``minimum`` consecutive accepted frames whose far-field
    NCC sits at or below ``threshold`` is where a viewer should look hardest:
    the mapping claims a partner and the imagery does not obviously support it.
    """

    found: list[dict] = []
    start: int | None = None
    for index, row in enumerate(rows):
        weak = (
            row.b is not None
            and math.isfinite(float(ncc[index]))
            and float(ncc[index]) <= threshold
        )
        if weak and start is None:
            start = index
        elif not weak and start is not None:
            if index - start >= minimum:
                window = ncc[start:index]
                found.append(
                    {
                        "runA": [rows[start].a, rows[index - 1].a],
                        "runB": [rows[start].b, rows[index - 1].b],
                        "frames": index - start,
                        "median_ncc": round(float(np.median(window)), 3),
                        "status": rows[start].status,
                    }
                )
            start = None
    return found


# --------------------------------------------------------------- overlays --


def _overlay_layers(draw_fn) -> tuple[np.ndarray, np.ndarray]:
    layer = Image.new("RGBA", (PANEL_W, PANEL_H), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    array = np.asarray(layer).astype(np.float32)
    rgb = array[..., :3]
    alpha = (array[..., 3:4]) / 255.0
    return rgb, alpha


def _draw_guides(draw: ImageDraw.ImageDraw) -> None:
    grid = (255, 255, 255, 56)  # thin, semi-transparent
    for x in range(0, PANEL_W + 1, GRID_STEP):
        draw.line([(min(x, PANEL_W - 1), 0), (min(x, PANEL_W - 1), PANEL_H)], fill=grid)
    for y in range(0, PANEL_H + 1, GRID_STEP):
        draw.line([(0, min(y, PANEL_H - 1)), (PANEL_W, min(y, PANEL_H - 1))], fill=grid)
    # Row boundaries used by the near/far analysis: top third and bottom third.
    boundary = (255, 214, 92, 200)
    for y in (PANEL_H // 3, 2 * PANEL_H // 3):
        draw.line([(0, y), (PANEL_W, y)], fill=boundary, width=1)
    # Edge ticks.
    tick = (255, 255, 255, 130)
    for x in range(0, PANEL_W + 1, TICK_STEP):
        xx = min(x, PANEL_W - 1)
        draw.line([(xx, 0), (xx, 7)], fill=tick)
        draw.line([(xx, PANEL_H - 8), (xx, PANEL_H - 1)], fill=tick)
    for y in range(0, PANEL_H + 1, TICK_STEP):
        yy = min(y, PANEL_H - 1)
        draw.line([(0, yy), (7, yy)], fill=tick)
        draw.line([(PANEL_W - 8, yy), (PANEL_W - 1, yy)], fill=tick)


def _dashed_line(draw, start, end, fill, dash=9, gap=7) -> None:
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    position = 0.0
    while position < length:
        end_position = min(position + dash, length)
        draw.line(
            [
                (x0 + ux * position, y0 + uy * position),
                (x0 + ux * end_position, y0 + uy * end_position),
            ],
            fill=fill,
        )
        position += dash + gap


def _draw_pose_grid(draw: ImageDraw.ImageDraw, dx: float, dy: float) -> None:
    colour = (90, 226, 255, 215)  # cyan, clearly distinct from the white grid
    for k in range(-2, PANEL_W // GRID_STEP + 3):
        x = k * GRID_STEP + dx
        if -2 <= x <= PANEL_W + 2:
            _dashed_line(draw, (x, 0), (x, PANEL_H), colour)
    for k in range(-2, PANEL_H // GRID_STEP + 3):
        y = k * GRID_STEP + dy
        if -2 <= y <= PANEL_H + 2:
            _dashed_line(draw, (0, y), (PANEL_W, y), colour)


def build_overlays(camera: str, dx: float, dy: float):
    guides = _overlay_layers(_draw_guides)
    if camera == "cam5":
        pose = _overlay_layers(lambda d: _draw_pose_grid(d, dx, dy))
    else:
        pose = None
    return guides, pose


def apply_overlay(panel: np.ndarray, layer, strength: float = 1.0) -> np.ndarray:
    rgb, alpha = layer
    if strength != 1.0:
        alpha = alpha * strength
    return (panel.astype(np.float32) * (1.0 - alpha) + rgb * alpha).astype(np.uint8)


# ----------------------------------------------------------- strip / info --


def ncc_colour(value: float) -> tuple[int, int, int]:
    t = (value - NCC_RED) / (NCC_GREEN - NCC_RED)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return (int(238 * (1 - t) + 40 * t), int(60 * (1 - t) + 214 * t), int(60 * (1 - t) + 90 * t))


def build_strip(ncc: np.ndarray) -> np.ndarray:
    strip = np.zeros((STRIP_H, CANVAS_W, 3), dtype=np.uint8)
    strip[:, :] = (16, 17, 22)
    total = len(ncc)
    bar_top, bar_bottom = 14, STRIP_H - 3
    height = bar_bottom - bar_top
    for x in range(CANVAS_W):
        lo = int(x * total / CANVAS_W)
        hi = max(lo + 1, int((x + 1) * total / CANVAS_W))
        window = ncc[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            strip[bar_bottom - 2 : bar_bottom, x] = (70, 70, 78)  # no match here
            continue
        value = float(finite.mean())
        filled = int(round(max(0.0, min(1.0, value / NCC_FULL_BAR)) * height))
        if filled > 0:
            strip[bar_bottom - filled : bar_bottom, x] = ncc_colour(value)
    image = Image.fromarray(strip)
    draw = ImageDraw.Draw(image)
    draw.text(
        (6, 1),
        "far-field NCC over the whole route (rows 12-62%, gradient magnitude)"
        f"   red <= {NCC_RED:.2f}   green >= {NCC_GREEN:.2f}"
        f"   full bar = {NCC_FULL_BAR:.2f}   grey = no match",
        font=FONT_TINY,
        fill=(196, 198, 208),
    )
    return np.asarray(image)


def draw_info_bar(
    draw: ImageDraw.ImageDraw, camera: str, row: Row, ncc_value: float, dx: float, dy: float
) -> None:
    y = INFO_Y
    confidence = "n/a" if not math.isfinite(row.confidence) else f"{row.confidence:.2f}"
    b_text = str(row.b) if row.b is not None else "--"
    ncc_text = "  --  " if not math.isfinite(ncc_value) else f"{ncc_value:+.3f}"
    line = (
        f"{camera}  runA {row.a:4d} -> runB {b_text:>4}  "
        f"status {row.status or 'n/a':<24} tier {row.tier or 'n/a':<32} "
        f"conf {confidence:>4}"
    )
    draw.text((8, y + 5), line, font=FONT_INFO, fill=(232, 234, 242))

    bar_x0, bar_x1, bar_y = 8, 700, y + 38
    draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + 12], outline=(96, 100, 114), fill=(30, 32, 40))
    fraction = row.a / max(1, RUN_A_FRAMES - 1)
    draw.rectangle(
        [bar_x0 + 1, bar_y + 1, bar_x0 + 1 + int((bar_x1 - bar_x0 - 2) * fraction), bar_y + 11],
        fill=(96, 168, 232),
    )
    draw.text(
        (bar_x1 + 12, bar_y - 1),
        f"route position {row.a}/{RUN_A_FRAMES - 1}  t={row.a / FPS:6.1f}s",
        font=FONT_SMALL,
        fill=(196, 198, 208),
    )
    draw.text(
        (bar_x1 + 232, bar_y - 1),
        f"far-field NCC {ncc_text}",
        font=FONT_SMALL,
        fill=ncc_colour(ncc_value) if math.isfinite(ncc_value) else (150, 152, 162),
    )
    if dx or dy:
        draw.text(
            (bar_x1 + 372, bar_y - 1),
            f"pose offset dx={dx:+.1f} dy={dy:+.1f} px",
            font=FONT_SMALL,
            fill=(140, 220, 250),
        )


# ---------------------------------------------------------------- render ---


class Renderer:
    def __init__(self, camera: str, rows, frames_a, frames_b, ncc):
        self.camera = camera
        self.rows = rows
        self.frames_a = frames_a
        self.frames_b = frames_b
        self.ncc = ncc
        self.dx, self.dy = pose_offset(camera)
        self.shift = (int(round(-self.dx)), int(round(-self.dy)))
        self.guides, self.pose_grid = build_overlays(camera, self.dx, self.dy)
        self.strip = build_strip(ncc)
        compensated = "pose-compensated " if self.pose_grid is not None else ""
        self.label_b = f"B: {RUN_LABEL['runB']} (mapped)"
        self.label_blend = f"50/50 blend of A and {compensated}B"
        self.label_diff = f"|A - {compensated}B|  absolute difference"
        if self.pose_grid is not None:
            note = f"B translated by ({self.shift[0]:+d}, {self.shift[1]:+d}) px"
            self.label_blend += f"   [{note}]"
            self.label_diff += f"   [{note}]"

    def _panel_header(self, canvas: np.ndarray, x: int, y: int, height: int = 22) -> None:
        region = canvas[y : y + height, x : x + PANEL_W]
        region[:] = (region.astype(np.float32) * 0.32).astype(np.uint8)

    def compose(self, index: int) -> Image.Image:
        row = self.rows[index]
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[INFO_Y:STRIP_Y] = (22, 23, 30)

        frame_a = (
            self.frames_a[row.a]
            if row.a < self.frames_a.shape[0]
            else np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
        )
        panel_a = apply_overlay(frame_a, self.guides)
        canvas[0:PANEL_H, 0:PANEL_W] = panel_a

        matched = row.b is not None and row.b < self.frames_b.shape[0]
        if matched:
            frame_b = self.frames_b[row.b]
            panel_b = apply_overlay(frame_b, self.guides)
            if self.pose_grid is not None:
                panel_b = apply_overlay(panel_b, self.pose_grid)
            compensated_b = shift_image(frame_b, *self.shift)
            blend = ((frame_a.astype(np.uint16) + compensated_b) // 2).astype(np.uint8)
            diff = np.abs(frame_a.astype(np.int16) - compensated_b.astype(np.int16)).astype(
                np.uint8
            )
            blend = apply_overlay(blend, self.guides)
            diff = apply_overlay(diff, self.guides)
        else:
            dim = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
            panel_b = apply_overlay(dim, self.guides, strength=0.30)
            blend = panel_b.copy()
            diff = panel_b.copy()

        canvas[0:PANEL_H, PANEL_W:CANVAS_W] = panel_b
        canvas[PANEL_H : 2 * PANEL_H, 0:PANEL_W] = blend
        canvas[PANEL_H : 2 * PANEL_H, PANEL_W:CANVAS_W] = diff
        canvas[STRIP_Y:CANVAS_H] = self.strip

        for x, y in ((0, 0), (PANEL_W, 0), (0, PANEL_H), (PANEL_W, PANEL_H)):
            self._panel_header(canvas, x, y, 44 if (x, y) == (PANEL_W, 0) and
                               self.pose_grid is not None else 22)
        canvas[PANEL_H - 1 : PANEL_H + 1, :] = (70, 74, 88)
        canvas[:, PANEL_W - 1 : PANEL_W + 1] = (70, 74, 88)
        canvas[2 * PANEL_H - 1 : 2 * PANEL_H + 1, :] = (70, 74, 88)

        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text((8, 3), f"A: {RUN_LABEL['runA']}   frame {row.a}", font=FONT_PANEL,
                  fill=(236, 238, 246))
        b_caption = f"{self.label_b}   frame {row.b}" if matched else self.label_b
        draw.text((PANEL_W + 8, 3), b_caption, font=FONT_PANEL,
                  fill=(236, 238, 246) if matched else (150, 152, 162))
        draw.text((8, PANEL_H + 3), self.label_blend, font=FONT_PANEL, fill=(236, 238, 246))
        draw.text((PANEL_W + 8, PANEL_H + 3), self.label_diff, font=FONT_PANEL,
                  fill=(236, 238, 246))
        if self.pose_grid is not None:
            draw.text(
                (PANEL_W + 8, 26),
                f"dashed cyan = pose-compensated grid (dx={self.dx:+.1f}, dy={self.dy:+.1f} px)",
                font=FONT_SMALL,
                fill=(120, 226, 255),
            )
        if not matched:
            message = f"no match: {row.status or 'unmapped'}"
            for panel_x, panel_y in ((PANEL_W, 0), (0, PANEL_H), (PANEL_W, PANEL_H)):
                draw.text(
                    (panel_x + PANEL_W // 2, panel_y + PANEL_H // 2),
                    message,
                    font=FONT_PANEL,
                    fill=(214, 176, 96),
                    anchor="mm",
                )

        draw_info_bar(draw, self.camera, row, float(self.ncc[index]), self.dx, self.dy)
        playhead = int(round(row.a / max(1, RUN_A_FRAMES - 1) * (CANVAS_W - 1)))
        draw.line([(playhead, STRIP_Y + 10), (playhead, CANVAS_H - 1)], fill=(255, 255, 255))
        return image

    def title_card(self, title: str, subtitle: str, detail: str) -> Image.Image:
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (14, 15, 20)
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text((CANVAS_W // 2, 430), title, font=FONT_TITLE, fill=(240, 242, 250), anchor="mm")
        draw.text((CANVAS_W // 2, 500), subtitle, font=FONT_SUB, fill=(150, 200, 240), anchor="mm")
        draw.text((CANVAS_W // 2, 545), detail, font=FONT_SUB, fill=(180, 182, 196), anchor="mm")
        draw.text(
            (CANVAS_W // 2, 620),
            f"{CAM_LABEL[self.camera]}   alignment highlights",
            font=FONT_SMALL,
            fill=(130, 132, 146),
            anchor="mm",
        )
        return image


# ---------------------------------------------------------------- encoder --


def open_encoder(path: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def close_encoder(process: subprocess.Popen) -> None:
    assert process.stdin is not None
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")


def ffprobe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt",
        "-show_entries", "format=duration,size", "-of", "json", str(path),
    ]
    return json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)


# --------------------------------------------------------------- segments --


def pick_segments(camera: str, rows: list[Row], ncc: np.ndarray) -> list[dict]:
    matched = [row.a for row in rows if row.b is not None]
    start_a = matched[0] if matched else 0

    def window(centre_lo: int, centre_hi: int) -> tuple[int, int]:
        centre = (centre_lo + centre_hi) // 2
        lo = max(0, min(RUN_A_FRAMES - SEGMENT_FRAMES, centre - SEGMENT_FRAMES // 2))
        return lo, lo + SEGMENT_FRAMES - 1

    def mean_ncc(lo: int) -> float:
        window_values = ncc[lo : lo + SEGMENT_FRAMES]
        finite = window_values[np.isfinite(window_values)]
        return float(finite.mean()) if finite.size else float("nan")

    hard_lo, hard_hi = HARD_REGION[camera]

    # Strong corridor: an all-strong window with the best far-field agreement,
    # away from the route start and the hard region.
    best_lo, best_score = None, -2.0
    for lo in range(start_a + SEGMENT_FRAMES, RUN_A_FRAMES - SEGMENT_FRAMES):
        if hard_lo - SEGMENT_FRAMES <= lo <= hard_hi:
            continue
        block = rows[lo : lo + SEGMENT_FRAMES]
        if any(row.status != "accepted_strong" for row in block):
            continue
        score = mean_ncc(lo)
        if math.isfinite(score) and score > best_score:
            best_lo, best_score = lo, score
    if best_lo is None:
        best_lo = start_a + SEGMENT_FRAMES

    # Hard region: the window inside it with the weakest far-field agreement.
    worst_lo, worst_score = hard_lo, 2.0
    for lo in range(hard_lo, max(hard_lo + 1, hard_hi - SEGMENT_FRAMES)):
        score = mean_ncc(lo)
        if math.isfinite(score) and score < worst_score:
            worst_lo, worst_score = lo, score

    # Dark stretch: the runA frames whose partners fall in the named runB range.
    b_lo, b_hi = DARK_STRETCH_B[camera]
    partners = [row.a for row in rows if row.b is not None and b_lo <= row.b <= b_hi]
    dark = window(partners[0], partners[-1]) if partners else window(b_lo, b_hi)

    return [
        {
            "title": "1. Route start",
            "subtitle": f"runA {start_a}-{start_a + SEGMENT_FRAMES - 1}",
            "detail": "first matched frames after the idle prefix",
            "range": (start_a, start_a + SEGMENT_FRAMES - 1),
        },
        {
            "title": "2. Strong-tier corridor",
            "subtitle": f"runA {best_lo}-{best_lo + SEGMENT_FRAMES - 1}",
            "detail": f"all accepted_strong; mean far-field NCC {best_score:+.2f}",
            "range": (best_lo, best_lo + SEGMENT_FRAMES - 1),
        },
        {
            "title": "3. Hardest region",
            "subtitle": f"runA {worst_lo}-{worst_lo + SEGMENT_FRAMES - 1}",
            "detail": f"weakest agreement inside runA {hard_lo}-{hard_hi} "
                      f"(mean NCC {worst_score:+.2f})",
            "range": (worst_lo, worst_lo + SEGMENT_FRAMES - 1),
        },
        {
            "title": "4. Dark stretch",
            "subtitle": f"runA {dark[0]}-{dark[1]}",
            "detail": f"partners of runB {b_lo}-{b_hi}",
            "range": dark,
        },
        {
            "title": "5. Confirmed occlusion",
            "subtitle": f"runA {window(*OCCLUSION_A)[0]}-{window(*OCCLUSION_A)[1]}",
            "detail": f"runB cam0 240-270 blocked; runA partners {OCCLUSION_A[0]}-{OCCLUSION_A[1]}",
            "range": window(*OCCLUSION_A),
        },
        {
            "title": "6. Loop closure",
            "subtitle": f"runA {LOOP_CLOSURE_A[1] - SEGMENT_FRAMES + 1}-{LOOP_CLOSURE_A[1]}",
            "detail": "end of the loop, back at the start of the route",
            "range": (LOOP_CLOSURE_A[1] - SEGMENT_FRAMES + 1, LOOP_CLOSURE_A[1]),
        },
    ]


# ------------------------------------------------------------------- main --


def render_camera(camera: str, stage: str) -> dict:
    print(f"[{camera}] loading mapping and streams", flush=True)
    rows = load_mapping(camera)
    frames_a = decode_stream("runA", camera, 2700)
    frames_b = decode_stream("runB", camera, 2700)

    dx, dy = pose_offset(camera)
    shift_small = (int(round(-dx * SMALL_W / PANEL_W)), int(round(-dy * SMALL_H / PANEL_H)))
    print(f"[{camera}] pose offset dx={dx:+.2f} dy={dy:+.2f} px at {PANEL_W}x{PANEL_H}; "
          f"NCC compensation {shift_small}", flush=True)

    print(f"[{camera}] far-field NCC pre-pass", flush=True)
    grad_a = gradient_stack(frames_a)
    grad_b = gradient_stack(frames_b)
    ncc = compute_ncc(rows, grad_a, grad_b, shift_small)
    finite = ncc[np.isfinite(ncc)]
    stats = {
        "matched": int(finite.size),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "below_0.15": int((finite < NCC_RED).sum()),
    }
    print(f"[{camera}] NCC {stats}", flush=True)
    del grad_a, grad_b

    if stage == "stats":
        SCRATCH.mkdir(parents=True, exist_ok=True)
        np.save(SCRATCH / f"ncc_{camera}.npy", ncc)
        stats["weak_accepted_runs"] = weak_accepted_runs(rows, ncc)
        for run in stats["weak_accepted_runs"]:
            print(f"[{camera}] weak accepted run {run}", flush=True)

    renderer = Renderer(camera, rows, frames_a, frames_b, ncc)
    result = {"camera": camera, "ncc": stats, "segments": pick_segments(camera, rows, ncc)}

    if stage == "stats":
        return result

    if stage == "preview":
        SCRATCH.mkdir(parents=True, exist_ok=True)
        matched_indices = [row.a for row in rows if row.b is not None]
        hard_matched = [
            a for a in matched_indices if HARD_REGION[camera][0] <= a <= HARD_REGION[camera][1]
        ]
        picks = [
            ("start", matched_indices[0] + 8),
            ("middle", matched_indices[len(matched_indices) // 2]),
            ("hard", hard_matched[len(hard_matched) // 2] if hard_matched else matched_indices[-1]),
            ("nomatch", 60),
        ]
        for name, index in picks:
            renderer.compose(index).save(SCRATCH / f"preview_{camera}_{name}_{index}.png")
            print(f"  wrote preview_{camera}_{name}_{index}.png", flush=True)
        return result

    if stage in ("full", "all"):
        target = VIDEO_DIR / f"aligned_full_{camera}.mp4"
        print(f"[{camera}] rendering {RUN_A_FRAMES} frames -> {target.name}", flush=True)
        encoder = open_encoder(target)
        assert encoder.stdin is not None
        for index in range(RUN_A_FRAMES):
            encoder.stdin.write(np.asarray(renderer.compose(index)).tobytes())
            if index % 250 == 0:
                print(f"    {index}/{RUN_A_FRAMES}", flush=True)
        close_encoder(encoder)
        result["full"] = str(target)

    if stage in ("highlights", "all"):
        target = VIDEO_DIR / f"aligned_highlights_{camera}.mp4"
        print(f"[{camera}] rendering highlights -> {target.name}", flush=True)
        encoder = open_encoder(target)
        assert encoder.stdin is not None
        for segment in result["segments"]:
            card = np.asarray(
                renderer.title_card(segment["title"], segment["subtitle"], segment["detail"])
            ).tobytes()
            for _ in range(TITLE_FRAMES):
                encoder.stdin.write(card)
            lo, hi = segment["range"]
            for index in range(lo, hi + 1):
                encoder.stdin.write(np.asarray(renderer.compose(index)).tobytes())
        close_encoder(encoder)
        result["highlights"] = str(target)

    for key in ("full", "highlights"):
        if key in result:
            path = Path(result[key])
            result[f"{key}_probe"] = ffprobe(path)
            result[f"{key}_bytes"] = path.stat().st_size
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="both", choices=["cam0", "cam5", "both"])
    parser.add_argument(
        "--stage", default="all", choices=["stats", "preview", "full", "highlights", "all"]
    )
    arguments = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not on PATH")

    cameras = ["cam0", "cam5"] if arguments.camera == "both" else [arguments.camera]
    summary = [render_camera(camera, arguments.stage) for camera in cameras]
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
