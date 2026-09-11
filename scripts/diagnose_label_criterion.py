#!/usr/bin/env python3
"""Detect and correct annotation-criterion drift caused by a camera pose change.

"Same place" has two definitions that people slide between without noticing, and
they only diverge when the camera pointing has changed between the two runs:

* **world-centric** - the vehicle is at the same position along the route;
* **camera-centric** - the scene sits at the same place in the frame.

If the camera yawed, no single frame satisfies both. An annotator with an
oriented road marker in view naturally uses the first; with only a mailbox to go
on, the second. The result is labels that are individually reasonable and
collectively inconsistent, and nothing in the label file records which rule was
used.

This module measures the rule from the pixels. For each labelled pair it takes
the median horizontal displacement of RootSIFT correspondences, `dx`. A
camera-centric label sits at `dx` near zero; a world-centric one sits at the
yaw. Where a label actually falls says which rule produced it, and the spread
across labels is the inconsistency.

**On correcting it.** Two calibration-free attempts to recover the vehicle
position per pair were tried and both failed on this data: regressing `dx`
against image row (a fisheye scrambles row-as-depth badly enough that the slope
is pure noise) and using the slowest-moving matches as a far-field yaw estimate
(dispersion across pairs is as large as the label noise itself). So this does
*not* pretend to move each label to the right frame.

What it does instead is honest and sufficient for evaluation: since the rule
behind any one label is unrecoverable, the label's acceptable interval is
widened to span both rules. The bound is geometric rather than arbitrary - the
answer lies between the camera-centric frame and the world-centric one - and it
is self-limiting, because on a camera whose pose did not change the two coincide
and the interval is left alone.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image

import build_reliability_masks
import build_task2_unified as unified
import frame_service
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_label_criterion"
DEFAULT_LABELS = ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "blind_v2_labels.csv"

CAMERAS = ("cam0", "cam5")

# Rate is measured across this many frames either side of the label; wide enough
# to beat the noise in a single median, narrow enough that the rate is locally
# constant.
RATE_HALF_SPAN = 3
MIN_MATCHES = 15
SIFT_RATIO = 0.8

# Converting a pixel ambiguity into frames divides by the local sweep rate, so a
# nearly stationary stretch produces a meaningless interval. Below this rate the
# criterion is simply unresolvable and the label is flagged rather than widened
# to something absurd.
MIN_RATE_PX_PER_FRAME = 2.0
MAX_WIDENING_FRAMES = 10

# The ego vehicle is rigidly attached and is the same car in both runs, so any
# shift of this band is camera pose and nothing else. It is the one always
# available calibration target in the frame.
EGO_BAND_TOP_FRACTION = 0.75
ANALYSIS_WIDTH, ANALYSIS_HEIGHT = 448, 336



def temporal_median(run: str, camera: str, stride: int = 8) -> np.ndarray:
    collected = []
    for ordinal, image in frame_service.iter_frames(
        frame_service.source_path(run, camera), ANALYSIS_WIDTH, ANALYSIS_HEIGHT
    ):
        if ordinal % stride:
            continue
        collected.append(np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32))
    return np.median(np.stack(collected), axis=0)


# A phase-correlation peak this weak is not a measurement. Run B's ego band is
# largely rain, so the two runs may share too little rigid structure to register
# at all, and the estimate must say so rather than return a confident number.
MIN_POSE_PEAK_SHARPNESS = 1.5


def _shared_ego_structure(camera: str) -> np.ndarray:
    """Pixels the reliability mask calls camera-attached in both runs."""

    def resize(mask: np.ndarray) -> np.ndarray:
        image = Image.fromarray((mask * 255).astype(np.uint8))
        return np.asarray(image.resize((ANALYSIS_WIDTH, ANALYSIS_HEIGHT), Image.NEAREST)) > 127

    run_a = build_reliability_masks.load_mask("runA", camera)["camera_attached_analysis"]
    run_b = build_reliability_masks.load_mask("runB", camera)["camera_attached_analysis"]
    return resize(run_a) & resize(run_b)


def ego_band_yaw(camera: str) -> dict[str, object]:
    """Camera pose shift between runs, read off the rigidly attached ego vehicle.

    Reported with the sharpness of its own correlation peak, because on a camera
    whose ego band is buried in rain the peak vanishes and the shift it returns
    is meaningless. Three attempts to measure CAM5's pose this way all failed;
    the estimate is kept, flagged, and not relied upon.
    """

    top = int(ANALYSIS_HEIGHT * EGO_BAND_TOP_FRACTION)
    band_a = temporal_median("runA", camera)[top:]
    band_b = temporal_median("runB", camera)[top:]

    # Restrict to structure that is camera-attached in BOTH runs. The raw band
    # also holds road surface and, in Run B, a great deal of rain, and those
    # correlate spuriously: on the raw band CAM5 returns a confident-looking
    # quarter-frame shift that the shared-structure version shows to be noise.
    shared = _shared_ego_structure(camera)[top:]
    if shared.sum() >= 200:
        band_a = np.where(shared, band_a, 0.0)
        band_b = np.where(shared, band_b, 0.0)
    band_a = (band_a - band_a.mean()) / max(band_a.std(), 1e-6)
    band_b = (band_b - band_b.mean()) / max(band_b.std(), 1e-6)

    margin = 110
    offsets = np.arange(-140, 141)
    scores = np.array(
        [
            float((band_a[:, margin:-margin] * np.roll(band_b, shift, axis=1)[:, margin:-margin]).mean())
            for shift in offsets
        ]
    )
    peak = int(scores.argmax())
    sharpness = float((scores[peak] - np.median(scores)) / max(scores.std(), 1e-9))
    at_boundary = peak in (0, len(offsets) - 1)
    reliable = sharpness >= MIN_POSE_PEAK_SHARPNESS and not at_boundary
    scale = unified.GEOMETRY_SIZE[0] / ANALYSIS_WIDTH
    return {
        "dx_analysis_px": float(offsets[peak]),
        "dx_geometry_px": round(float(offsets[peak]) * scale, 3),
        "peak_correlation": round(float(scores[peak]), 4),
        "peak_sharpness": round(sharpness, 3),
        "peak_at_search_boundary": at_boundary,
        "reliable": reliable,
        "note": (
            "Rigid reference, so a sharp peak means camera pose. A flat one means "
            "the two runs share too little ego structure to register - which is "
            "what happens when rain fills the band - and the shift is then not a "
            "measurement of anything."
        ),
    }


def median_dx(
    image_a: np.ndarray,
    image_b: np.ndarray,
    exclusion_a: np.ndarray,
    exclusion_b: np.ndarray,
) -> tuple[float | None, int]:
    """Median horizontal displacement of correspondences: where the scene sits."""

    from skimage.feature import match_descriptors

    points_a, descriptors_a = unified.rootsift_features(image_a, exclusion_a)
    points_b, descriptors_b = unified.rootsift_features(image_b, exclusion_b)
    if len(descriptors_a) < MIN_MATCHES or len(descriptors_b) < MIN_MATCHES:
        return None, 0
    matches = match_descriptors(
        descriptors_a, descriptors_b, metric="euclidean",
        cross_check=True, max_ratio=SIFT_RATIO,
    )
    if len(matches) < MIN_MATCHES:
        return None, len(matches)
    displacement = points_b[matches[:, 1], 0] - points_a[matches[:, 0], 0]
    return float(np.median(displacement)), len(matches)


def analyse_camera(camera: str, labels: list[dict[str, str]]) -> dict[str, object]:
    exclusion_a = build_reliability_masks.load_mask("runA", camera)["exclusion_224"]
    exclusion_b = build_reliability_masks.load_mask("runB", camera)["exclusion_224"]
    yaw = ego_band_yaw(camera)

    wanted_a = [int(row["runA_frame"]) for row in labels]
    wanted_b = sorted(
        {
            int(row["runB_frame"]) + offset
            for row in labels
            for offset in (-RATE_HALF_SPAN, 0, RATE_HALF_SPAN)
        }
    )
    images_a = frame_service.frames("runA", camera, wanted_a, *unified.GEOMETRY_SIZE)
    images_b = frame_service.frames("runB", camera, wanted_b, *unified.GEOMETRY_SIZE)

    records: list[dict[str, object]] = []
    for row in labels:
        frame_a = int(row["runA_frame"])
        frame_b = int(row["runB_frame"])
        centre, matches = median_dx(
            images_a[frame_a], images_b[frame_b], exclusion_a, exclusion_b
        )
        low, _ = median_dx(
            images_a[frame_a], images_b[frame_b - RATE_HALF_SPAN], exclusion_a, exclusion_b
        )
        high, _ = median_dx(
            images_a[frame_a], images_b[frame_b + RATE_HALF_SPAN], exclusion_a, exclusion_b
        )
        rate = (
            (high - low) / (2 * RATE_HALF_SPAN)
            if low is not None and high is not None
            else None
        )
        records.append(
            {
                "query_id": row["query_id"],
                "camera_id": camera,
                "runA_frame": frame_a,
                "runB_frame": frame_b,
                "runB_min": int(row["runB_min"]),
                "runB_max": int(row["runB_max"]),
                "matches": matches,
                "dx_at_label": None if centre is None else round(centre, 2),
                "dx_per_frame": None if rate is None else round(rate, 3),
            }
        )

    measured = [r["dx_at_label"] for r in records if r["dx_at_label"] is not None]
    array = np.array(measured, dtype=float)
    return {
        "camera_id": camera,
        "labels": len(labels),
        "measured": len(measured),
        "ego_band_pose_shift": yaw,
        "dx_at_label": {
            "median": round(float(np.median(array)), 2) if len(array) else None,
            "mean": round(float(array.mean()), 2) if len(array) else None,
            "sd": round(float(array.std()), 2) if len(array) else None,
            "min": round(float(array.min()), 2) if len(array) else None,
            "max": round(float(array.max()), 2) if len(array) else None,
        },
        "records": records,
    }


# A camera-centric label sits at dx near zero. A systematic offset of more than
# this many standard errors from zero means the labels were placed by some other
# rule, whatever that rule was.
OFFSET_ALARM_STANDARD_ERRORS = 4.0


def _verdict(analysis: dict[str, object], pose: dict[str, object]) -> str:
    """Judge from the systematic offset, not the dispersion or the pose estimate.

    Dispersion turned out to be the weaker signal - CAM5 scatters only 1.35x
    more than CAM0 across the full label set, not the 2.5x a 16-label subset
    suggested. The offset is what separates the two cameras cleanly, and it is
    also what a pose change would produce.
    """

    stats = analysis["dx_at_label"]
    if stats["sd"] is None or not analysis["measured"]:
        return "not enough matched pairs to judge"
    standard_error = stats["sd"] / max(analysis["measured"] ** 0.5, 1.0)
    if abs(stats["median"]) > OFFSET_ALARM_STANDARD_ERRORS * standard_error:
        cause = (
            "camera pose changed between runs"
            if pose["reliable"]
            else "cause not established; the pose estimate did not converge"
        )
        return (
            f"labels sit {stats['median']:+.0f} px from camera-centric "
            f"({cause})"
        )
    return "labels are camera-centric and consistent"


def widen_intervals(
    analysis: dict[str, object], world_centric_dx: float
) -> list[dict[str, object]]:
    """Widen each interval to span the camera-centric and world-centric answers.

    The rule behind any single label is unrecoverable, so the interval has to
    admit both. Where the two coincide - a camera whose pose did not change -
    this leaves the interval as it was.
    """

    corrected: list[dict[str, object]] = []
    for record in analysis["records"]:
        low = int(record["runB_min"])
        high = int(record["runB_max"])
        widened_low, widened_high = low, high
        shift_camera = shift_world = None

        rate = record["dx_per_frame"]
        centre = record["dx_at_label"]
        resolvable = (
            rate is not None
            and centre is not None
            and abs(rate) >= MIN_RATE_PX_PER_FRAME
        )
        if resolvable:
            # Frames from the labelled position to each of the two rules,
            # clipped so a slow stretch cannot manufacture a huge interval.
            shift_camera = float(np.clip((0.0 - centre) / rate, -MAX_WIDENING_FRAMES, MAX_WIDENING_FRAMES))
            shift_world = float(np.clip((world_centric_dx - centre) / rate, -MAX_WIDENING_FRAMES, MAX_WIDENING_FRAMES))
            candidates = [
                record["runB_frame"] + shift_camera,
                record["runB_frame"] + shift_world,
            ]
            widened_low = int(np.floor(min(low, *candidates)))
            widened_high = int(np.ceil(max(high, *candidates)))

        corrected.append(
            {
                **record,
                "criterion_shift_to_camera_centric": (
                    None if shift_camera is None else round(shift_camera, 2)
                ),
                "criterion_shift_to_world_centric": (
                    None if shift_world is None else round(shift_world, 2)
                ),
                "criterion_resolvable": str(bool(resolvable)).lower(),
                "runB_min_widened": widened_low,
                "runB_max_widened": widened_high,
                "widening_frames": (widened_high - widened_low) - (high - low),
            }
        )
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    target = OUTPUT_DIR / "label_criterion_diagnostic.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    with arguments.labels.open(newline="") as handle:
        labels = [row for row in csv.DictReader(handle) if row["label"] == "match"]

    analyses = {}
    for camera in CAMERAS:
        camera_labels = [row for row in labels if row["camera_id"] == camera]
        if not camera_labels:
            continue
        print(f"[{camera}] measuring where the scene sits at {len(camera_labels)} labels ...", flush=True)
        analyses[camera] = analyse_camera(camera, camera_labels)

    dispersions = [
        a["dx_at_label"]["sd"] for a in analyses.values() if a["dx_at_label"]["sd"] is not None
    ]
    reference = min(dispersions) if dispersions else None
    for analysis in analyses.values():
        analysis["reference_dispersion"] = reference

    corrected_rows: list[dict[str, object]] = []
    summary = {}
    for camera, analysis in analyses.items():
        # World-centric target: the scene displacement expected when the vehicle
        # is at the same position, i.e. the camera pose shift alone.
        pose = analysis["ego_band_pose_shift"]
        if pose["reliable"]:
            world_dx = float(pose["dx_geometry_px"])
        else:
            # The pose could not be measured, so the alternative rule is taken
            # from where the labels themselves sit. Not knowing widens the
            # interval; it must never collapse it.
            world_dx = float(analysis["dx_at_label"]["median"] or 0.0)
        corrected = widen_intervals(analysis, world_dx)
        corrected_rows.extend(corrected)
        widening = [r["widening_frames"] for r in corrected]
        unresolvable = sum(1 for r in corrected if r["criterion_resolvable"] == "false")
        summary[camera] = {
            "labels": analysis["labels"],
            "ego_band_dx_geometry_px": world_dx,
            "dx_at_label": analysis["dx_at_label"],
            "median_widening_frames": float(np.median(widening)),
            "max_widening_frames": int(max(widening)),
            "criterion_unresolvable_labels": unresolvable,
            "unresolvable_reason": (
                f"local sweep rate below {MIN_RATE_PX_PER_FRAME} px/frame, so a "
                "pixel ambiguity cannot be converted into frames"
            ),
            "pose_estimate_reliable": bool(pose["reliable"]),
            "verdict": _verdict(analysis, pose),
        }

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_file": str(arguments.labels.relative_to(ROOT)),
        "what_this_measures": (
            "Which definition of 'same place' each label used, read from the "
            "median horizontal displacement of correspondences at the labelled "
            "frame. Near zero means camera-centric; near the camera pose shift "
            "means world-centric."
        ),
        "what_it_does_not_do": (
            "It does not move labels to the correct frame. Two calibration-free "
            "per-pair decompositions were tried and failed on this data: row-as-"
            "depth regression, and a far-field yaw estimate. The interval is "
            "widened to span both rules instead."
        ),
        "cameras": summary,
    }
    atomic_write(target, lambda path: path.write_text(json.dumps(payload, indent=2)))

    fieldnames = list(corrected_rows[0])
    atomic_write(
        OUTPUT_DIR / "labels_with_criterion_intervals.csv",
        lambda path: _write_csv(path, fieldnames, corrected_rows),
    )

    for camera, entry in summary.items():
        stats = entry["dx_at_label"]
        print(f"\n=== {camera}")
        flag = "" if entry["pose_estimate_reliable"] else "   [not reliable: peak did not converge]"
        print(f"  ego-band camera pose shift : {entry['ego_band_dx_geometry_px']:+.1f} px{flag}")
        print(
            f"  scene dx at labels         : median {stats['median']:+.1f}, "
            f"sd {stats['sd']:.1f}, range [{stats['min']:+.0f}, {stats['max']:+.0f}]"
        )
        print(f"  interval widening          : median {entry['median_widening_frames']:.1f} frames, max {entry['max_widening_frames']}")
        print(f"  criterion unresolvable     : {entry['criterion_unresolvable_labels']}/{entry['labels']} labels (too slow to convert)")
        print(f"  verdict                    : {entry['verdict']}")
    print(f"\nWrote {OUTPUT_DIR.relative_to(ROOT)}")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
