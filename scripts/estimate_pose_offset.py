#!/usr/bin/env python3
"""Where does the matcher put "same place", and can a camera pose shift be undone?

CAM5's camera pointed slightly differently in the two runs. For a side-facing
camera that is a degenerate nuisance: a yaw offset shifts the scene sideways in
the image, and so does driving a few frames further. The two are separable
only with an external reference to world position. This script measures what
can be measured and refuses the rest, in four parts.

1. Matcher convention. At accepted pairs of the submitted mapping, the median
   horizontal (dx) and vertical (dy) displacement of RootSIFT correspondences,
   sampled along the whole route. dx near zero means the matcher answers
   camera-centrically (the scene sits in the same image position). dy is not
   confounded by driving and reads a pitch change directly.

2. Label convention. From the per-label dx already measured by
   `diagnose_label_criterion.py`: a global offset with a bootstrap interval,
   and whether it is constant along the route, which a pose shift must be and
   a lane-position difference need not be.

3. World anchors. Suspension events are world events: a speed bump fires at
   the same road position whatever the camera pointing. Where a Run A event's
   mapped partner sits within reach of a Run B event, the frame offset between
   them is a world-centric correction the appearance matcher cannot see.

4. What follows. The blind labels are rescored (they are spent, so this is a
   diagnostic) under the camera-centric convention, by shifting each label to
   dx = 0 using its own measured sweep rate. If the anchors in part 3 agree, a
   world-centric mapping is also written, with the per-frame correction
   s / v(t): the pose shift in pixels over the local sweep speed.

Nothing here touches the submitted mapping file.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import build_motion_events
import build_reliability_masks
import build_task2_unified as unified
import diagnose_label_criterion as criterion
from evaluate_task2_blind_v2 import distance_to_interval, wilson_interval
import frame_service
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_pose_offset"
CRITERION_CSV = ROOT / "outputs" / "task2_label_criterion" / "labels_with_criterion_intervals.csv"
LABELS = ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "blind_v2_labels.csv"
CAMERAS = ("cam0", "cam5")

PAIR_STRIDE = 24                 # accepted pairs sampled along the route
RATE_HALF_SPAN = criterion.RATE_HALF_SPAN
ANCHOR_TOLERANCE_FRAMES = 15     # a Run B event this close to the mapped partner anchors it
MIN_ANCHORS = 3
MAX_ANCHOR_MAD_FRAMES = 3.0      # anchors must agree this well to define a correction
MAX_CORRECTION_FRAMES = 10
MIN_CONVENTION_GAP_PX = 5.0      # below this the two conventions agree and nothing is done
MIN_RATE_PX_PER_FRAME = criterion.MIN_RATE_PX_PER_FRAME
BOOTSTRAP_SAMPLES = 2000
# The sweep rate is measured by build_sweep_rate.py with the same detector and
# raster as dx, so no rescaling is needed. Earlier phase-correlation estimates
# needed a factor of two and still failed validation.
SPEED_TO_GEOMETRY_SCALE = 1.0



def load_mapping(camera: str) -> dict[int, dict[str, str]]:
    path = ROOT / "outputs" / "task2_unified" / camera / "frame_mapping_unified.csv"
    with path.open(newline="") as handle:
        return {int(row["runA_frame"]): row for row in csv.DictReader(handle)}


def bootstrap_median(values: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    medians = [
        float(np.median(rng.choice(values, values.size, replace=True)))
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return [round(float(np.percentile(medians, 2.5)), 2), round(float(np.percentile(medians, 97.5)), 2)]


# --------------------------------------------------------------------------
# Part 1: matcher convention
# --------------------------------------------------------------------------


def median_displacement(image_a, image_b, exclusion_a, exclusion_b) -> tuple[float, float, int] | None:
    from skimage.feature import match_descriptors

    points_a, descriptors_a = unified.rootsift_features(image_a, exclusion_a)
    points_b, descriptors_b = unified.rootsift_features(image_b, exclusion_b)
    if len(descriptors_a) < criterion.MIN_MATCHES or len(descriptors_b) < criterion.MIN_MATCHES:
        return None
    matches = match_descriptors(
        descriptors_a, descriptors_b, metric="euclidean", cross_check=True, max_ratio=criterion.SIFT_RATIO
    )
    if len(matches) < criterion.MIN_MATCHES:
        return None
    delta = points_b[matches[:, 1]] - points_a[matches[:, 0]]
    return float(np.median(delta[:, 0])), float(np.median(delta[:, 1])), int(len(matches))


def matcher_convention(camera: str, mapping: dict[int, dict[str, str]]) -> dict[str, object]:
    exclusion_a = build_reliability_masks.load_mask("runA", camera)["exclusion_224"]
    exclusion_b = build_reliability_masks.load_mask("runB", camera)["exclusion_224"]
    accepted = [
        frame for frame, row in sorted(mapping.items())
        if row["status"] == unified.STATUS_ACCEPTED_STRONG and row["runB_frame"]
    ]
    sampled = accepted[::PAIR_STRIDE]
    partners = {frame: int(mapping[frame]["runB_frame"]) for frame in sampled}
    wanted_b = sorted({p + d for p in partners.values() for d in (-RATE_HALF_SPAN, 0, RATE_HALF_SPAN)})
    images_a = frame_service.frames("runA", camera, sampled, *unified.GEOMETRY_SIZE)
    images_b = frame_service.frames("runB", camera, wanted_b, *unified.GEOMETRY_SIZE)

    records = []
    for frame in sampled:
        partner = partners[frame]
        centre = median_displacement(images_a[frame], images_b[partner], exclusion_a, exclusion_b)
        if centre is None:
            continue
        low = median_displacement(images_a[frame], images_b[partner - RATE_HALF_SPAN], exclusion_a, exclusion_b)
        high = median_displacement(images_a[frame], images_b[partner + RATE_HALF_SPAN], exclusion_a, exclusion_b)
        rate = (high[0] - low[0]) / (2 * RATE_HALF_SPAN) if low and high else None
        records.append(
            {
                "runA_frame": frame,
                "runB_frame": partner,
                "matches": centre[2],
                "dx_px": round(centre[0], 2),
                "dy_px": round(centre[1], 2),
                "dx_per_frame": None if rate is None else round(rate, 3),
            }
        )

    dx = np.array([r["dx_px"] for r in records])
    dy = np.array([r["dy_px"] for r in records])
    frames = np.array([r["runA_frame"] for r in records], dtype=float)
    slope_dx = float(np.polyfit(frames, dx, 1)[0]) * 1000 if len(records) > 3 else None
    return {
        "pairs_sampled": len(sampled),
        "pairs_measured": len(records),
        "dx_px": {
            "median": round(float(np.median(dx)), 2),
            "bootstrap_95": bootstrap_median(dx),
            "mad": round(float(1.4826 * np.median(np.abs(dx - np.median(dx)))), 2),
        },
        "dy_px": {
            "median": round(float(np.median(dy)), 2),
            "bootstrap_95": bootstrap_median(dy, seed=1),
            "mad": round(float(1.4826 * np.median(np.abs(dy - np.median(dy)))), 2),
        },
        "dx_trend_px_per_1000_frames": None if slope_dx is None else round(slope_dx, 2),
        "sweep_rate_px_per_frame_median": round(
            float(np.median([r["dx_per_frame"] for r in records if r["dx_per_frame"] is not None])), 3
        ),
        "records": records,
    }


# --------------------------------------------------------------------------
# Part 2: label convention
# --------------------------------------------------------------------------


def label_convention(camera: str) -> dict[str, object]:
    with CRITERION_CSV.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["camera_id"] == camera and r["dx_at_label"]]
    dx = np.array([float(r["dx_at_label"]) for r in rows])
    frames = np.array([int(r["runA_frame"]) for r in rows], dtype=float)
    order = np.argsort(frames)
    halves = np.array_split(dx[order], 2)
    slope = float(np.polyfit(frames, dx, 1)[0]) * 1000 if len(rows) > 3 else None
    # Spearman rank correlation between position and dx: near zero if constant.
    rank_f = np.argsort(np.argsort(frames))
    rank_d = np.argsort(np.argsort(dx))
    rho = float(np.corrcoef(rank_f, rank_d)[0, 1]) if len(rows) > 3 else None
    return {
        "labels_measured": len(rows),
        "dx_px": {
            "median": round(float(np.median(dx)), 2),
            "bootstrap_95": bootstrap_median(dx, seed=2),
            "mad": round(float(1.4826 * np.median(np.abs(dx - np.median(dx)))), 2),
        },
        "first_half_median": round(float(np.median(halves[0])), 2),
        "second_half_median": round(float(np.median(halves[1])), 2),
        "trend_px_per_1000_frames": None if slope is None else round(slope, 2),
        "spearman_position_vs_dx": None if rho is None else round(rho, 3),
        "constant_along_route": bool(rho is not None and abs(rho) < 0.3),
    }


# --------------------------------------------------------------------------
# Part 3: world anchors from suspension events
# --------------------------------------------------------------------------


def world_anchors(camera: str, mapping: dict[int, dict[str, str]]) -> dict[str, object]:
    try:
        run_a = build_motion_events.load_signal("runA", camera)
        run_b = build_motion_events.load_signal("runB", camera)
    except FileNotFoundError:
        return {"usable": False, "reason": "no motion signals cached"}
    events_b = np.array(run_b.events)
    anchors = []
    for event in run_a.events:
        row = mapping.get(event)
        if not row or not row["runB_frame"] or row["status"] not in unified.ACCEPTED_STATUSES:
            continue
        partner = int(row["runB_frame"])
        if not events_b.size:
            break
        nearest = int(events_b[np.argmin(np.abs(events_b - partner))])
        if abs(nearest - partner) > ANCHOR_TOLERANCE_FRAMES:
            continue
        speed = None
        if run_b.speed is not None:
            speed = float(run_b.speed[min(partner, run_b.speed.size - 1)]) * SPEED_TO_GEOMETRY_SCALE
        anchors.append(
            {
                "runA_event": int(event),
                "mapped_runB": partner,
                "runB_event": nearest,
                "offset_frames": nearest - partner,
                "sweep_px_per_frame_at_partner": None if speed is None else round(speed, 2),
            }
        )
    if not anchors:
        return {"usable": False, "reason": "no Run A event has a Run B event near its mapped partner", "anchors": []}
    offsets = np.array([a["offset_frames"] for a in anchors], dtype=float)
    mad = float(1.4826 * np.median(np.abs(offsets - np.median(offsets))))
    usable = len(anchors) >= MIN_ANCHORS and mad <= MAX_ANCHOR_MAD_FRAMES
    implied_px = [
        a["offset_frames"] * a["sweep_px_per_frame_at_partner"]
        for a in anchors if a["sweep_px_per_frame_at_partner"]
    ]
    return {
        "usable": bool(usable),
        "anchors": anchors,
        "count": len(anchors),
        "offset_frames_median": round(float(np.median(offsets)), 2),
        "offset_frames_mad": round(mad, 2),
        "implied_pose_shift_px_median": round(float(np.median(implied_px)), 1) if implied_px else None,
        "reason": None if usable else (
            f"{len(anchors)} anchors (need {MIN_ANCHORS}) with MAD {mad:.1f} frames "
            f"(need <= {MAX_ANCHOR_MAD_FRAMES}); the anchors do not define one correction"
        ),
        "caveat": (
            "Both event sets are sparse and their co-firing rate was measured at "
            "2.8x chance on four pairs; an anchor here is suggestive, not a fix."
        ),
    }


# --------------------------------------------------------------------------
# Part 4: rescoring and the world-centric mapping
# --------------------------------------------------------------------------


def shifted_labels(camera: str) -> list[dict[str, object]]:
    """Each label moved to the camera-centric answer using its own sweep rate."""

    with CRITERION_CSV.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["camera_id"] == camera]
    output = []
    for row in rows:
        shift = row["criterion_shift_to_camera_centric"]
        resolvable = row["criterion_resolvable"] == "true" and shift not in ("", "None")
        delta = int(round(float(shift))) if resolvable else 0
        output.append(
            {
                "runA_frame": int(row["runA_frame"]),
                "preferred": int(row["runB_frame"]) + delta,
                "low": int(row["runB_min"]) + delta,
                "high": int(row["runB_max"]) + delta,
                "widened_low": int(row["runB_min_widened"]),
                "widened_high": int(row["runB_max_widened"]),
                "as_labelled_low": int(row["runB_min"]),
                "as_labelled_high": int(row["runB_max"]),
                "resolvable": resolvable,
                "shift": delta,
            }
        )
    return output


def score(labels: list[dict[str, object]], mapping: dict[int, dict[str, str]], low_key: str, high_key: str) -> dict:
    accepted = correct = 0
    errors = []
    for label in labels:
        predicted = mapping[label["runA_frame"]]["runB_frame"].strip()
        if not predicted:
            continue
        accepted += 1
        prediction = int(predicted)
        low, high = label[low_key], label[high_key]
        distance = distance_to_interval(prediction, low, high)
        errors.append(distance)
        correct += distance == 0
    return {
        "accepted": accepted,
        "correct": correct,
        "precision": round(correct / accepted, 4) if accepted else None,
        "wilson_95": wilson_interval(correct, accepted),
        "within_2": round(sum(1 for e in errors if e <= 2) / accepted, 4) if accepted else None,
        "median_interval_distance": float(np.median(errors)) if errors else None,
    }


def corrected_mapping(
    camera: str,
    mapping: dict[int, dict[str, str]],
    shift_px: float,
    rate_sign: float,
    filename: str,
) -> tuple[Path, dict]:
    """Move every partner by shift_px of scene displacement, at the local sweep speed.

    The sweep speed comes from the motion signal (unsigned); its sign is the one
    RootSIFT measured at the matched pairs, so the correction moves the partner
    in the direction that actually changes dx the right way.
    """

    run_b = build_motion_events.load_signal("runB", camera)
    speed = run_b.speed * SPEED_TO_GEOMETRY_SCALE
    rows = []
    corrections = []
    for frame in sorted(mapping):
        record = dict(mapping[frame])
        if record["runB_frame"]:
            partner = int(record["runB_frame"])
            rate = float(speed[min(partner, speed.size - 1)])
            if rate >= MIN_RATE_PX_PER_FRAME:
                delta = float(np.clip(shift_px / (rate_sign * rate), -MAX_CORRECTION_FRAMES, MAX_CORRECTION_FRAMES))
            else:
                delta = 0.0
            corrections.append(delta)
            record["runB_frame"] = str(int(round(partner + delta)))
            record["convention_correction_frames"] = round(delta, 2)
        else:
            record["convention_correction_frames"] = ""
        rows.append(record)
    # Corrections vary with speed, so monotonicity must be re-imposed.
    previous = -1
    for record in rows:
        if record["runB_frame"]:
            value = max(int(record["runB_frame"]), previous)
            record["runB_frame"] = str(value)
            previous = value
    target = OUTPUT_DIR / camera / filename

    def writer(path: Path) -> None:
        with path.open("w", newline="") as handle:
            output = csv.DictWriter(handle, fieldnames=list(rows[0]))
            output.writeheader()
            output.writerows(rows)

    atomic_write(target, writer)
    array = np.array(corrections)
    return target, {
        "shift_px": shift_px,
        "rate_sign": rate_sign,
        "correction_frames": {
            "median": round(float(np.median(array)), 2),
            "p5": round(float(np.percentile(array, 5)), 2),
            "p95": round(float(np.percentile(array, 95)), 2),
            "clipped_fraction": round(float((np.abs(array) >= MAX_CORRECTION_FRAMES).mean()), 4),
        },
    }


def speed_versus_rootsift(camera: str, records: list[dict[str, object]]) -> dict[str, object]:
    """Is the dense speed profile the same quantity RootSIFT measures at pairs?

    The correction divides pixels by this speed, so a biased speed is a biased
    correction. The RootSIFT rate at the sampled pairs is the reference.
    """

    try:
        run_b = build_motion_events.load_signal("runB", camera)
    except FileNotFoundError:
        return {"pairs": 0}
    if run_b.speed is None:
        return {"pairs": 0}
    dense, sparse = [], []
    for record in records:
        rate = record["dx_per_frame"]
        if rate is None or abs(rate) < MIN_RATE_PX_PER_FRAME:
            continue
        partner = int(record["runB_frame"])
        dense.append(float(run_b.speed[min(partner, run_b.speed.size - 1)]) * SPEED_TO_GEOMETRY_SCALE)
        sparse.append(abs(float(rate)))
    if len(dense) < 5:
        return {"pairs": len(dense)}
    dense_a, sparse_a = np.array(dense), np.array(sparse)
    rank_d = np.argsort(np.argsort(dense_a))
    rank_s = np.argsort(np.argsort(sparse_a))
    return {
        "pairs": len(dense),
        "median_ratio_speed_over_rootsift": round(float(np.median(dense_a / sparse_a)), 3),
        "spearman": round(float(np.corrcoef(rank_d, rank_s)[0, 1]), 3),
        "dense_median_px_per_frame": round(float(np.median(dense_a)), 2),
        "rootsift_median_px_per_frame": round(float(np.median(sparse_a)), 2),
    }


def convention_check(
    camera: str, corrected: dict[int, dict[str, str]], sampled: list[int]
) -> dict[str, object]:
    """Re-measure dx at the corrected partners of the same sampled pairs."""

    exclusion_a = build_reliability_masks.load_mask("runA", camera)["exclusion_224"]
    exclusion_b = build_reliability_masks.load_mask("runB", camera)["exclusion_224"]
    partners = {f: int(corrected[f]["runB_frame"]) for f in sampled if corrected[f]["runB_frame"]}
    images_a = frame_service.frames("runA", camera, sorted(partners), *unified.GEOMETRY_SIZE)
    images_b = frame_service.frames("runB", camera, sorted(set(partners.values())), *unified.GEOMETRY_SIZE)
    dx = []
    for frame, partner in partners.items():
        measured = median_displacement(images_a[frame], images_b[partner], exclusion_a, exclusion_b)
        if measured is not None:
            dx.append(measured[0])
    array = np.array(dx)
    return {
        "pairs_measured": len(dx),
        "dx_px_median": round(float(np.median(array)), 2) if len(dx) else None,
        "dx_px_bootstrap_95": bootstrap_median(array, seed=3) if len(dx) > 5 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-matcher-convention", action="store_true", help="Reuse part 1 from disk.")
    arguments = parser.parse_args()

    target = OUTPUT_DIR / "pose_offset_diagnostic.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")
    previous = json.loads(target.read_text()) if target.is_file() else {}

    cameras: dict[str, dict[str, object]] = {}
    for camera in CAMERAS:
        mapping = load_mapping(camera)
        print(f"[{camera}] part 1: where the matcher puts the scene ...", flush=True)
        if arguments.skip_matcher_convention and camera in previous.get("cameras", {}):
            matcher = previous["cameras"][camera]["matcher_convention"]
        else:
            matcher = matcher_convention(camera, mapping)
        print(f"   dx {matcher['dx_px']['median']:+.1f} px, dy {matcher['dy_px']['median']:+.1f} px "
              f"over {matcher['pairs_measured']} pairs")

        labels = label_convention(camera)
        print(f"   labels: dx {labels['dx_px']['median']:+.1f} px "
              f"[{labels['dx_px']['bootstrap_95'][0]:+.1f}, {labels['dx_px']['bootstrap_95'][1]:+.1f}], "
              f"constant along route: {labels['constant_along_route']}")

        anchors = world_anchors(camera, mapping)
        if anchors.get("anchors"):
            print(f"   world anchors: {anchors['count']}, offset median {anchors['offset_frames_median']:+.1f} "
                  f"frames, MAD {anchors['offset_frames_mad']:.1f}, usable: {anchors['usable']}")
        else:
            print(f"   world anchors: none ({anchors.get('reason')})")

        shifted = shifted_labels(camera)
        rescoring = {
            "as_labelled": score(shifted, mapping, "as_labelled_low", "as_labelled_high"),
            "widened_to_both_conventions": score(shifted, mapping, "widened_low", "widened_high"),
            "shifted_to_camera_centric": score(shifted, mapping, "low", "high"),
            "labels_shifted": sum(1 for s in shifted if s["resolvable"]),
            "median_shift_frames": float(np.median([abs(s["shift"]) for s in shifted])),
            "evidence_status": "the blind labels are spent; this is a diagnostic rescoring, not a held-out number",
        }
        for key in ("as_labelled", "widened_to_both_conventions", "shifted_to_camera_centric"):
            entry = rescoring[key]
            print(f"   rescoring {key:<30}: precision {entry['precision']}, within-2 {entry['within_2']}")

        rates = [r["dx_per_frame"] for r in matcher["records"] if r["dx_per_frame"] is not None]
        rate_sign = float(np.sign(np.median(rates))) if rates else -1.0
        sampled = [r["runA_frame"] for r in matcher["records"]]
        speed_check = speed_versus_rootsift(camera, matcher["records"])
        print(f"   dense speed vs RootSIFT rate at {speed_check['pairs']} pairs: "
              f"median ratio {speed_check['median_ratio_speed_over_rootsift']}, "
              f"Spearman {speed_check['spearman']}")

        world = None
        if anchors.get("usable") and anchors.get("implied_pose_shift_px_median") is not None:
            path, world = corrected_mapping(
                camera, mapping, float(anchors["implied_pose_shift_px_median"]), rate_sign,
                "frame_mapping_unified_world_centric.csv",
            )
            world_mapping = {int(r["runA_frame"]): r for r in csv.DictReader(path.open(newline=""))}
            world["rescoring_as_labelled"] = score(shifted, world_mapping, "as_labelled_low", "as_labelled_high")
            world["rescoring_widened"] = score(shifted, world_mapping, "widened_low", "widened_high")
            world["file"] = str(path.relative_to(ROOT))
            print(f"   world-centric mapping written; as-labelled precision "
                  f"{world['rescoring_as_labelled']['precision']}")
        else:
            print("   world-centric mapping: not written (anchors do not define a correction)")

        # The convention gap: how far the matcher's "same place" sits from the
        # labels' along the sweep axis. One constant per camera. Converting the
        # mapping to the label convention is a change of definition, not a fit
        # to 32 labels, but the constant was measured on those labels, so the
        # rescoring below is in-sample for that one degree of freedom and the
        # self-consistency check (no labels) is the independent evidence.
        gap = float(labels["dx_px"]["median"]) - float(matcher["dx_px"]["median"])
        convention = {
            "gap_px": round(gap, 2),
            "gap_frames_at_median_sweep": round(
                gap / max(abs(matcher["sweep_rate_px_per_frame_median"]), MIN_RATE_PX_PER_FRAME), 2
            ),
            "applied": False,
        }
        if abs(gap) >= MIN_CONVENTION_GAP_PX and labels["constant_along_route"]:
            path, applied = corrected_mapping(
                camera, mapping, gap, rate_sign, "frame_mapping_unified_label_convention.csv"
            )
            converted = {int(r["runA_frame"]): r for r in csv.DictReader(path.open(newline=""))}
            applied["file"] = str(path.relative_to(ROOT))
            applied["rescoring_as_labelled_in_sample_constant"] = score(
                shifted, converted, "as_labelled_low", "as_labelled_high"
            )
            applied["rescoring_widened"] = score(shifted, converted, "widened_low", "widened_high")
            print(f"   label-convention mapping: shift {gap:+.1f} px, "
                  f"median correction {applied['correction_frames']['median']:+.2f} frames; "
                  f"as-labelled precision {applied['rescoring_as_labelled_in_sample_constant']['precision']}")
            print("   self-consistency: re-measuring dx at the corrected pairs ...", flush=True)
            applied["self_consistency"] = convention_check(camera, converted, sampled)
            applied["self_consistency"]["target_dx_px"] = labels["dx_px"]["median"]
            print(f"   dx at corrected pairs {applied['self_consistency']['dx_px_median']} px "
                  f"(target {labels['dx_px']['median']})")
            convention.update(applied)
            convention["applied"] = True
        else:
            print(f"   convention gap {gap:+.1f} px: below threshold or not constant; no conversion")

        cameras[camera] = {
            "matcher_convention": matcher,
            "label_convention": labels,
            "world_anchors": anchors,
            "rescoring": rescoring,
            "world_centric_mapping": world,
            "label_convention_conversion": convention,
            "dense_speed_check": speed_check,
        }

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "degeneracy": (
            "For a side-facing camera a yaw offset and a few frames of travel "
            "produce the same horizontal scene shift. dx at matched pairs therefore "
            "reports the matcher's convention, not the pose; dy is unconfounded. "
            "Only an external reference (suspension events, GPS, or calibration) "
            "separates pose from position."
        ),
        "cameras": cameras,
    }
    atomic_write(target, lambda path: path.write_text(json.dumps(payload, indent=2)))
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
