#!/usr/bin/env python3
"""Measure which pixels are attached to the camera rather than to the world.

The v1 matcher masked with SegFormer, excluding the ADE20K dynamic and sky
classes; the submitted v2 matcher uses this module instead. That model has no class for the two contaminants that actually
dominate these recordings: the ego vehicle's wing mirror, which sits in the
bottom of every frame, and run B's rain droplets, which are static on the lens
for the entire recording. Both are high-contrast and perfectly repeatable, which
makes them the most attractive features in the image and the most useless.

They are also directly measurable, with no learned model, because they share one
property: content attached to the camera does not move when the vehicle does.
The discriminating statistic is

    staticness = |grad(temporal median)| / (temporal std + 1)

A world feature sweeping past a side-facing camera produces high temporal
variance wherever it appears, so its staticness is low however sharp it is. A
droplet rim or a mirror edge is sharp in the median *and* quiet in time, so its
staticness is high. The ratio is close to scale-free, which matters because run
B is 2.2x darker and half as sharp as run A - a fixed quantile would mask the
same fraction of both and tell us nothing.

Sharp rims alone are not enough: a droplet's interior is a smeared, low-gradient
version of the scene behind it, and that is what corrupts a descriptor. So the
rims seed a morphological closing and hole fill that recovers the droplet body.

Separately, pixels that are never bright in any frame are optically dead
(vignette, or the fisheye circle's edge) and are excluded on their own evidence.

Output is one mask per (run, camera), because the droplets exist only in run B.
Masks are cached with a version derived from these parameters, so changing a
threshold invalidates every descriptor computed under the old mask instead of
silently mixing them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import frame_service
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_masks"

RUNS = ("runA", "runB")
CAMERAS = ("cam0", "cam5")

# Statistics are gathered at the descriptor decode raster and the mask is then
# resampled to the 224x224 square the backbones consume.
ANALYSIS_WIDTH, ANALYSIS_HEIGHT = 448, 336
MASK_SIZE = 224
PATCH_GRID = 16

SAMPLE_STRIDE = 8

# A pixel that never reaches 18% of the sequence's own bright reference carries
# no scene information in any frame.
NEVER_BRIGHT_FRACTION = 0.18
BRIGHT_REFERENCE_PERCENTILE = 99.0

# Seed threshold on the staticness ratio. Measured separation at this value:
# run A cam5 0.41% of pixels against run B cam5 3.51%, an 8.5x increase that
# tracks the appearance of rain on the lens.
STATICNESS_SEED = 0.30
CLOSING_SIZE = 9
DILATION_SIZE = 5
MINIMUM_COMPONENT_PIXELS = 60

# A patch is dropped when this much of it is masked, matching the existing
# descriptor pooling rule.
PATCH_EXCLUSION_FRACTION = 0.25

# Optional sky component, variant "sky". Sky is bright in the temporal median,
# smooth, and reaches the top edge. It is never excluded by default: the shipped
# mask version predates it and the method freeze cites that version, so the
# variant has to be asked for and produces a different version string.
SKY_BRIGHT_FRACTION = 0.80
SKY_MAX_TEXTURE = 1.5
SKY_MAX_ROW_FRACTION = 0.55
SKY_SMOOTHING_SIGMA = 3.0
# Brighter than the image median by this fraction of the median-to-p99 range.
# Relative, because run A's temporal median is dark (median 49, p99 78 on
# CAM0) and an absolute margin would cut real sky; a flat image has zero range
# and therefore no sky.
SKY_MIN_CONTRAST_FRACTION = 0.25
KNOWN_VARIANTS = ("", "sky")


def mask_version(variant: str = "") -> str:
    """Version derived from the parameters, so a threshold change invalidates caches.

    The baseline payload is unchanged by the variant machinery: with no variant
    the hash is the one the freeze cites.
    """

    if variant not in KNOWN_VARIANTS:
        raise ValueError(f"unknown mask variant {variant!r}; known: {KNOWN_VARIANTS}")
    parameters: dict[str, object] = {
            "analysis": [ANALYSIS_WIDTH, ANALYSIS_HEIGHT],
            "mask_size": MASK_SIZE,
            "sample_stride": SAMPLE_STRIDE,
            "never_bright_fraction": NEVER_BRIGHT_FRACTION,
            "bright_reference_percentile": BRIGHT_REFERENCE_PERCENTILE,
            "staticness_seed": STATICNESS_SEED,
            "closing": CLOSING_SIZE,
            "dilation": DILATION_SIZE,
            "minimum_component": MINIMUM_COMPONENT_PIXELS,
            "colour": frame_service.COLOUR_ID,
    }
    if variant:
        parameters["variant"] = variant
    if "sky" in variant:
        parameters["sky"] = {
            "bright_fraction": SKY_BRIGHT_FRACTION,
            "max_texture": SKY_MAX_TEXTURE,
            "max_row_fraction": SKY_MAX_ROW_FRACTION,
            "smoothing_sigma": SKY_SMOOTHING_SIGMA,
            "min_contrast_fraction": SKY_MIN_CONTRAST_FRACTION,
        }
    payload = json.dumps(parameters, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass
class MaskResult:
    run: str
    camera: str
    frames_used: int
    never_bright: np.ndarray
    camera_attached: np.ndarray
    exclusion_224: np.ndarray
    patch_valid: np.ndarray
    median_image: np.ndarray
    sky: np.ndarray | None = None

    @property
    def statistics(self) -> dict[str, float]:
        stats = {
            "never_bright_fraction": float(self.never_bright.mean()),
            "camera_attached_fraction": float(self.camera_attached.mean()),
            "exclusion_fraction": float(self.exclusion_224.mean()),
            "valid_patch_fraction": float(self.patch_valid.mean()),
        }
        if self.sky is not None:
            stats["sky_fraction"] = float(self.sky.mean())
        return stats


def temporal_statistics(run: str, camera: str) -> tuple[np.ndarray, ...]:
    """Collect temporal median, standard deviation, and maximum luma."""

    source = frame_service.source_path(run, camera)
    collected: list[np.ndarray] = []
    for ordinal, image in frame_service.iter_frames(
        source, ANALYSIS_WIDTH, ANALYSIS_HEIGHT
    ):
        if ordinal % SAMPLE_STRIDE:
            continue
        collected.append(
            np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32)
        )
    if len(collected) < 16:
        raise ValueError(f"{run}/{camera}: too few sampled frames for temporal statistics")
    stack = np.stack(collected)
    return (
        np.median(stack, axis=0),
        stack.std(axis=0),
        stack.max(axis=0),
        len(collected),
    )


def resize_bool(mask: np.ndarray, size: int) -> np.ndarray:
    """Area-average a boolean mask to a square raster and re-threshold at half."""

    image = Image.fromarray((mask.astype(np.float32) * 255).astype(np.uint8))
    resized = np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32)
    return resized > 127.5


def sky_component(median_image: np.ndarray) -> np.ndarray:
    """Bright, smooth, and connected to the top edge in the temporal median.

    Three tests because each alone admits something else: bright alone takes
    white walls, smooth alone takes asphalt, top-connected alone takes trees.
    """

    smoothed = ndimage.gaussian_filter(median_image, SKY_SMOOTHING_SIGMA)
    gradient_y, gradient_x = np.gradient(smoothed)
    texture = np.sqrt(gradient_x**2 + gradient_y**2)
    p99 = float(np.percentile(smoothed, 99))
    median = float(np.median(smoothed))
    bright = (smoothed > SKY_BRIGHT_FRACTION * p99) & (
        smoothed > median + SKY_MIN_CONTRAST_FRACTION * (p99 - median)
    )
    height = median_image.shape[0]
    rows = np.arange(height)[:, None] < SKY_MAX_ROW_FRACTION * height
    candidate = bright & (texture < SKY_MAX_TEXTURE) & rows
    candidate = ndimage.binary_opening(candidate, structure=np.ones((5, 5)))
    labelled, count = ndimage.label(candidate)
    if not count:
        return np.zeros_like(candidate)
    touching_top = set(np.unique(labelled[0, :])) - {0}
    return np.isin(labelled, sorted(touching_top))


def build_mask(run: str, camera: str, variant: str = "") -> MaskResult:
    if variant not in KNOWN_VARIANTS:
        raise ValueError(f"unknown mask variant {variant!r}")
    median_image, temporal_std, temporal_max, frames_used = temporal_statistics(run, camera)

    bright_reference = float(np.percentile(temporal_max, BRIGHT_REFERENCE_PERCENTILE))
    never_bright = temporal_max < NEVER_BRIGHT_FRACTION * bright_reference

    gradient_y, gradient_x = np.gradient(median_image)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    staticness = gradient / (temporal_std + 1.0)

    seed = staticness > STATICNESS_SEED
    grown = ndimage.binary_closing(seed, structure=np.ones((CLOSING_SIZE, CLOSING_SIZE)))
    grown = ndimage.binary_fill_holes(grown)
    grown = ndimage.binary_dilation(grown, structure=np.ones((DILATION_SIZE, DILATION_SIZE)))

    labelled, count = ndimage.label(grown)
    if count:
        sizes = ndimage.sum(grown, labelled, range(1, count + 1))
        keep_labels = [index + 1 for index, size in enumerate(sizes) if size >= MINIMUM_COMPONENT_PIXELS]
        grown = np.isin(labelled, keep_labels)
    camera_attached = grown & ~never_bright

    exclusion = camera_attached | never_bright
    sky = None
    if "sky" in variant:
        sky = sky_component(median_image) & ~exclusion
        exclusion = exclusion | sky
    exclusion_224 = resize_bool(exclusion, MASK_SIZE)

    fraction = (
        exclusion_224.astype(np.float32)
        .reshape(PATCH_GRID, MASK_SIZE // PATCH_GRID, PATCH_GRID, MASK_SIZE // PATCH_GRID)
        .mean(axis=(1, 3))
    )
    patch_valid = fraction < PATCH_EXCLUSION_FRACTION

    return MaskResult(
        run=run,
        camera=camera,
        frames_used=frames_used,
        never_bright=never_bright,
        camera_attached=camera_attached,
        exclusion_224=exclusion_224,
        patch_valid=patch_valid,
        median_image=median_image,
        sky=sky,
    )


def write_overlay(result: MaskResult, target: Path) -> None:
    """Render the mask over the temporal median so it can be judged by eye."""

    base = np.stack([result.median_image] * 3, axis=-1).astype(np.uint8)
    base[result.camera_attached] = (
        base[result.camera_attached] * 0.25 + np.array([200, 40, 30]) * 0.75
    ).astype(np.uint8)
    base[result.never_bright] = (
        base[result.never_bright] * 0.30 + np.array([40, 90, 220]) * 0.70
    ).astype(np.uint8)
    if result.sky is not None:
        base[result.sky] = (
            base[result.sky] * 0.40 + np.array([240, 200, 40]) * 0.60
        ).astype(np.uint8)
    Image.fromarray(base).save(target)



def mask_path(run: str, camera: str, variant: str = "") -> Path:
    return OUTPUT_DIR / f"{run}_{camera}_mask_{mask_version(variant)}.npz"


def load_mask(run: str, camera: str, variant: str = "") -> dict[str, np.ndarray]:
    """Load a prebuilt mask. Raises if it was never built for these parameters."""

    target = mask_path(run, camera, variant)
    if not target.is_file():
        raise FileNotFoundError(
            f"No reliability mask for {run}/{camera} at mask version "
            f"{mask_version(variant)} (variant {variant!r}). "
            "Run scripts/build_reliability_masks.py first."
        )
    with np.load(target) as payload:
        return {key: payload[key] for key in payload.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing masks.")
    parser.add_argument(
        "--variant", choices=KNOWN_VARIANTS, default="",
        help="Development variant; the empty default is the shipped mask.",
    )
    arguments = parser.parse_args()
    variant = arguments.variant

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version = mask_version(variant)
    summary = {
        "schema_version": 1,
        "mask_version": version,
        "variant": variant or None,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "staticness = |grad(temporal median)| / (temporal std + 1); seeded "
            "threshold, morphological closing and hole fill to recover droplet "
            "bodies, unioned with never-bright optical dead zone"
        ),
        "parameters": {
            "analysis_raster": f"{ANALYSIS_WIDTH}x{ANALYSIS_HEIGHT}",
            "sample_stride": SAMPLE_STRIDE,
            "staticness_seed": STATICNESS_SEED,
            "never_bright_fraction": NEVER_BRIGHT_FRACTION,
            "closing_size": CLOSING_SIZE,
            "dilation_size": DILATION_SIZE,
            "minimum_component_pixels": MINIMUM_COMPONENT_PIXELS,
            "patch_exclusion_fraction": PATCH_EXCLUSION_FRACTION,
        },
        "learned_components": "none; this mask uses no trained model",
        "streams": {},
    }

    for run in RUNS:
        for camera in CAMERAS:
            target = mask_path(run, camera, variant)
            if target.is_file() and not arguments.force:
                print(f"{run}/{camera}: mask {version} already present, skipping")
                with np.load(target) as payload:
                    summary["streams"][f"{run}_{camera}"] = json.loads(
                        str(payload["statistics_json"])
                    )
                continue
            print(f"{run}/{camera}: sampling every {SAMPLE_STRIDE}th frame ...", flush=True)
            result = build_mask(run, camera, variant)
            statistics = result.statistics | {"frames_used": result.frames_used}
            summary["streams"][f"{run}_{camera}"] = statistics

            atomic_write(
                target,
                lambda path, result=result, statistics=statistics: np.savez_compressed(
                    path,
                    exclusion_224=result.exclusion_224,
                    patch_valid=result.patch_valid,
                    never_bright_analysis=result.never_bright,
                    camera_attached_analysis=result.camera_attached,
                    statistics_json=json.dumps(statistics),
                    **({"sky_analysis": result.sky} if result.sky is not None else {}),
                ),
            )
            overlay_suffix = f"_{variant}" if variant else ""
            write_overlay(result, OUTPUT_DIR / f"{run}_{camera}_mask_overlay{overlay_suffix}.png")
            print(
                f"   never-bright {statistics['never_bright_fraction']:.2%}, "
                f"camera-attached {statistics['camera_attached_fraction']:.2%}, "
                f"excluded {statistics['exclusion_fraction']:.2%}, "
                f"valid patches {statistics['valid_patch_fraction']:.2%}"
            )

    summary_name = f"reliability_mask_summary_{variant}.json" if variant else "reliability_mask_summary.json"
    atomic_write(
        OUTPUT_DIR / summary_name,
        lambda path: path.write_text(json.dumps(summary, indent=2)),
    )
    print(f"\nWrote {OUTPUT_DIR.relative_to(ROOT)} at mask version {version}")


if __name__ == "__main__":
    main()
