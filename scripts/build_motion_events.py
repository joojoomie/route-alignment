#!/usr/bin/env python3
"""Detect suspension-shake events and measure how much they are worth as evidence.

A speed bump is a physical feature of the road surface. Both runs must cross it,
and the vertical jolt it produces is independent of weather, exposure and
foliage - exactly the kind of evidence that is missing in the uniform green
corridors where appearance matching fails. That makes it attractive as a sparse,
sharp anchor even though it can never provide coverage.

The honest question is not whether the idea is appealing but how much a detected
event is actually worth, and this module measures that rather than assuming it.
The answer is a likelihood ratio: how much more often does a Run B event appear
at the true partner than at an arbitrary route position. A ratio near 1 means the
channel carries no information and a Bayesian consumer will correctly ignore it;
that is the point of expressing it this way rather than as a hand-set weight.

Two physical limits are worth stating up front, because they bound what any
processing of these frames can recover. Vehicle body bounce sits at 1-2 Hz, so at
10 FPS a jolt is described by five to ten samples. Wheel hop at 10-15 Hz is above
the 5 Hz Nyquist limit and is aliased outright. This signal wants an IMU or a
30+ FPS camera; what follows is the most that 10 FPS video can support.

**What the default construction is, and why it changed.** Each frame is divided
by its own stream's temporal mean image before the correlation is taken
(`--static-removal mean-divide`, now the default). The reason is a falsifiable
physical test rather than a preference: cam0 and cam5 are bolted to one chassis,
so a genuine suspension event must appear in both streams of the *same run* at
the same frame. Under the earlier construction, which correlated raw frames, the
Run B channel had **zero** such co-firings - 0 of 28 events within +-2 frames,
at any inter-camera lag from -5 to +5 - so whatever it was detecting, it was not
the vehicle. Dividing the static component out first gives 4/33 on Run B
(3.85x the chance rate, p ~ 0.019) and 9/45 on Run A (4.75x, p ~ 9e-5). The
event counts also stop being lopsided: Run A 24/21 and Run B 16/17 across
cam0/cam5, against 22/29 and 7/21 before, so the old "7 events in Run B against
22 in Run A" asymmetry was an artefact of the estimator, not a road with fewer
bumps. `median-subtract` was measured too and destroys Run B; `none` is kept
selectable so the withdrawn construction stays reproducible. See
run_static_removal_study.py for the full scoring.

Two consequences for what may be claimed. The cross-run likelihood ratio the old
construction reported (2.83, on four usable calibration pairs) is **withdrawn**,
not reinterpreted; the corrected channel's ratio rests on three development pairs
and is worth a direction, not a number. And the correction does not rescue the
channel's usefulness: the posterior mapping it feeds is 97%+ identical either way
(outputs/task2_motion_bayes/motion_variant_mapping_comparison_mean-divide.json).
What changed is that the channel is now measuring the right thing while
contributing nothing much, rather than measuring the wrong thing.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image

import frame_service
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_motion_bayes"
CALIBRATION_LABELS = ROOT / "outputs" / "task2_keyframes" / "manual_annotations.csv"

RUNS = ("runA", "runB")
CAMERAS = ("cam0", "cam5")

ANALYSIS_WIDTH, ANALYSIS_HEIGHT = 448, 336
# Crop away the sky band and the ego vehicle so the estimate tracks the scene.
CROP_TOP, CROP_BOTTOM, CROP_SIDE = 40, 70, 60

# Run B was recorded in rain and carries a field of lens droplets that is
# identical in every frame, and unnormalised cross correlation is driven by
# whatever carries the most energy - a fixed pattern has all of its energy at
# zero displacement. These modes divide or subtract the stream's own static
# component out before the estimate is taken. "mean-divide" is the default
# because it is the only mode whose events satisfy the same-vehicle co-firing
# constraint on both runs; "none" is the withdrawn construction, kept selectable.
STATIC_REMOVAL_MODES = ("none", "mean-divide", "median-subtract")
DEFAULT_STATIC_REMOVAL = "mean-divide"
MEDIAN_SAMPLE_STRIDE = 8      # every 8th frame is enough to estimate a static image
MEAN_DIVIDE_EPSILON = 1.0     # grey levels, keeps dark pixels from exploding

BASELINE_WINDOW = 15          # frames, removes the slow driving trend
EVENT_Z_THRESHOLD = 3.0
EVENT_MERGE_GAP = 3
MATCH_TOLERANCE_FRAMES = 5    # how close two events must be to count as the same
# The sweep rate (horizontal scene speed) is NOT estimated here. Whole-frame and
# tiled phase correlation were both tried and both failed validation against
# consecutive-frame RootSIFT (Spearman 0.05-0.48): on a side-facing fisheye the
# apparent speed depends on depth, so only the detector that measures dx can
# measure the rate dx is converted with. See build_sweep_rate.py.


@dataclass
class MotionSignal:
    run: str
    camera: str
    residual: np.ndarray       # per-frame vertical residual, index i is frame i->i+1
    events: list[int]
    speed: np.ndarray | None = None   # sweep rate, px/frame at the geometry raster (build_sweep_rate.py)
    static_removal: str = DEFAULT_STATIC_REMOVAL
    shifts: np.ndarray | None = None  # raw vertical shift before the baseline is removed

    def as_dict(self) -> dict[str, object]:
        payload = {
            "run": self.run,
            "camera": self.camera,
            "frames": int(self.residual.size),
            "residual_sd_px": round(float(self.residual.std()), 4),
            "residual_p99_px": round(float(np.percentile(np.abs(self.residual), 99)), 4),
            "event_count": len(self.events),
            "events": self.events,
        }
        payload["static_removal"] = self.static_removal
        if self.shifts is not None:
            payload["raw_zero_fraction"] = round(float((self.shifts == 0.0).mean()), 4)
        if self.speed is not None:
            payload["speed_px_per_frame"] = {
                "median": round(float(np.median(self.speed)), 3),
                "p10": round(float(np.percentile(self.speed, 10)), 3),
                "p90": round(float(np.percentile(self.speed, 90)), 3),
                "fraction_below_1px": round(float((self.speed < 1.0).mean()), 4),
            }
        return payload


def grey_stack(run: str, camera: str) -> np.ndarray:
    """Every frame of one stream as 8-bit greyscale, uncropped."""

    frames = []
    for _, image in frame_service.iter_frames(
        frame_service.source_path(run, camera), ANALYSIS_WIDTH, ANALYSIS_HEIGHT
    ):
        frames.append(np.asarray(Image.fromarray(image).convert("L"), dtype=np.uint8))
    return np.stack(frames)


def static_reference(stack: np.ndarray, mode: str) -> np.ndarray | None:
    """The stream's own time-invariant image, or None when nothing is removed.

    Anything that never moves - lens droplets, a dirty windscreen, a fixed
    vignette - survives a temporal average intact while the scene averages away,
    so the average *is* the contaminant. The median over a subsample is the same
    idea with less sensitivity to bright transients, and cheaper.
    """

    if mode not in STATIC_REMOVAL_MODES:
        raise ValueError(f"Unknown static removal mode {mode!r}; expected one of {STATIC_REMOVAL_MODES}")
    if mode == "none":
        return None
    if mode == "mean-divide":
        return stack.mean(axis=0, dtype=np.float64).astype(np.float32)
    return np.median(
        np.asarray(stack[::MEDIAN_SAMPLE_STRIDE], dtype=np.float32), axis=0
    ).astype(np.float32)


def remove_static(frame: np.ndarray, reference: np.ndarray | None, mode: str) -> np.ndarray:
    """Take the static component out of one frame, before any cropping."""

    frame = np.asarray(frame, dtype=np.float32)
    if reference is None or mode == "none":
        return frame
    if mode == "mean-divide":
        return frame / (reference + MEAN_DIVIDE_EPSILON)
    return frame - reference


def analysis_crop(frame: np.ndarray) -> np.ndarray:
    return frame[
        CROP_TOP : ANALYSIS_HEIGHT - CROP_BOTTOM, CROP_SIDE : ANALYSIS_WIDTH - CROP_SIDE
    ]


def shifts_from_stack(
    stack: np.ndarray, static_removal: str = DEFAULT_STATIC_REMOVAL
) -> np.ndarray:
    """Raw per-pair vertical shift, static component optionally removed first.

    Static removal happens on the whole frame and before the crop, so the mode
    changes what the correlator is shown and nothing else: the crop, the
    upsampling and the unnormalised correlation are identical in every mode, and
    the comparison between modes is therefore a comparison of one variable.
    """

    from skimage.registration import phase_cross_correlation

    reference = static_reference(stack, static_removal)
    shifts = np.zeros(len(stack) - 1, dtype=np.float64)
    previous = analysis_crop(remove_static(stack[0], reference, static_removal))
    for index in range(len(stack) - 1):
        current = analysis_crop(remove_static(stack[index + 1], reference, static_removal))
        offset, _, _ = phase_cross_correlation(
            previous, current, upsample_factor=20, normalization=None
        )
        shifts[index] = offset[0]
        previous = current
    return shifts


def remove_baseline(shifts: np.ndarray) -> np.ndarray:
    """Strip the slow driving trend, leaving the jolt."""

    kernel = np.ones(BASELINE_WINDOW) / BASELINE_WINDOW
    return shifts - np.convolve(shifts, kernel, mode="same")


def raw_vertical_shifts(
    run: str, camera: str, static_removal: str = DEFAULT_STATIC_REMOVAL
) -> np.ndarray:
    return shifts_from_stack(grey_stack(run, camera), static_removal)


def vertical_residual(
    run: str, camera: str, static_removal: str = DEFAULT_STATIC_REMOVAL
) -> np.ndarray:
    """Sub-pixel vertical image motion with the driving baseline removed.

    Pixel-quantised estimates are useless here: at a 192-wide analysis raster one
    pixel is nearly eight native pixels, which is larger than the whole signal.
    Phase correlation with upsampling is what makes the residual visible at all.
    """

    return remove_baseline(raw_vertical_shifts(run, camera, static_removal))


def detect_events(residual: np.ndarray) -> list[int]:
    """Frames whose residual is an outlier for this stream.

    The threshold is stream-relative, which matters: Run B's residual standard
    deviation is more than twice Run A's because rain makes the estimate noisy.
    A shared absolute threshold would silently detect far fewer events in Run B
    and be mistaken for Run B having fewer bumps.
    """

    scale = residual.std()
    if scale <= 0:
        return []
    outliers = np.where(np.abs(residual) / scale > EVENT_Z_THRESHOLD)[0]
    events: list[int] = []
    start = None
    for position, index in enumerate(outliers):
        if start is None:
            start = index
        last = position + 1 == len(outliers) or outliers[position + 1] - index > EVENT_MERGE_GAP
        if last:
            events.append(int(round((start + index) / 2)))
            start = None
    return events


def measure_likelihood_ratio(
    signals: dict[tuple[str, str], MotionSignal], camera: str, pairs: list[tuple[int, int]]
) -> dict[str, object]:
    """How much more often does a Run B event sit at the true partner than anywhere.

    `pairs` are known-correct correspondences. The result is the factor by which
    observing a Run A event should raise the odds of a Run B frame being the
    partner - and if it comes out near 1, the channel is worthless and should be
    seen to be worthless.
    """

    run_a = signals[("runA", camera)]
    run_b = signals[("runB", camera)]
    if not pairs or not run_b.events:
        return {"camera": camera, "usable": False, "reason": "no pairs or no Run B events"}

    a_events = set(run_a.events)
    hits = considered = 0
    for frame_a, frame_b in pairs:
        # Only pairs where Run A actually fired can test the channel.
        if not any(abs(frame_a - event) <= MATCH_TOLERANCE_FRAMES for event in a_events):
            continue
        considered += 1
        if any(abs(frame_b - event) <= MATCH_TOLERANCE_FRAMES for event in run_b.events):
            hits += 1

    span = max(run_b.residual.size, 1)
    window = 2 * MATCH_TOLERANCE_FRAMES + 1
    chance = min(1.0, len(run_b.events) * window / span)
    rate = hits / considered if considered else None
    ratio = (rate / chance) if (rate is not None and chance > 0) else None
    return {
        "camera": camera,
        "usable": considered >= 3,
        "run_a_events": len(run_a.events),
        "run_b_events": len(run_b.events),
        "pairs_with_a_run_a_event": considered,
        "of_those_matched_in_run_b": hits,
        "observed_rate": round(rate, 4) if rate is not None else None,
        "chance_rate": round(chance, 4),
        "likelihood_ratio": round(ratio, 3) if ratio is not None else None,
        "interpretation": (
            "Multiply the odds of a Run B frame being the partner by this factor "
            "when both sides carry an event. A value near 1 means no information."
        ),
    }


def calibration_pairs(camera: str) -> list[tuple[int, int]]:
    """Known-correct pairs from the original calibration labels.

    Those labels have been inspected many times and are development data. Using
    them to measure a sensor's reliability is exactly what development data is
    for; no blind label is touched.
    """

    if not CALIBRATION_LABELS.is_file():
        return []
    with CALIBRATION_LABELS.open(newline="") as handle:
        return [
            (int(row["runA_frame"]), int(row["runB_frame"]))
            for row in csv.DictReader(handle)
            if row["camera_id"] == camera and row["label"] == "match" and row["runB_frame"]
        ]



def mode_suffix(mode: str) -> str:
    """The default construction owns the plain names; every other mode is suffixed.

    The default moved from "none" to "mean-divide", so the plain files -
    motion_{run}_{camera}.npz and motion_event_summary.json - now carry the
    temporal-mean-normalised construction, and the withdrawn one is written to
    the _none names. Consumers that ask for no mode get the current default,
    which is the point: a downstream script should not have to know which
    construction won.
    """

    if mode not in STATIC_REMOVAL_MODES:
        raise ValueError(f"Unknown static removal mode {mode!r}; expected one of {STATIC_REMOVAL_MODES}")
    return "" if mode == DEFAULT_STATIC_REMOVAL else f"_{mode}"


def signal_path(run: str, camera: str, mode: str = DEFAULT_STATIC_REMOVAL) -> Path:
    return OUTPUT_DIR / f"motion_{run}_{camera}{mode_suffix(mode)}.npz"


def summary_path(mode: str = DEFAULT_STATIC_REMOVAL) -> Path:
    return OUTPUT_DIR / f"motion_event_summary{mode_suffix(mode)}.json"


def sweep_path(run: str, camera: str) -> Path:
    return OUTPUT_DIR / f"sweep_{run}_{camera}.npy"


def load_signal(run: str, camera: str, mode: str = DEFAULT_STATIC_REMOVAL) -> MotionSignal:
    with np.load(signal_path(run, camera, mode)) as payload:
        return MotionSignal(
            run=run,
            camera=camera,
            residual=payload["residual"],
            events=[int(value) for value in payload["events"]],
            speed=np.load(sweep_path(run, camera)) if sweep_path(run, camera).is_file() else None,
            static_removal=mode,
            # The shipped files predate this field; a variant file always has it.
            shifts=payload["shifts"] if "shifts" in payload.files else None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--static-removal",
        choices=STATIC_REMOVAL_MODES,
        default=DEFAULT_STATIC_REMOVAL,
        help=(
            "Remove the stream's time-invariant image before the estimate. "
            f"{DEFAULT_STATIC_REMOVAL!r} is the default and writes the plain "
            "motion_{run}_{camera}.npz and motion_event_summary.json; every "
            "other mode, including the withdrawn 'none', writes its own "
            "_{mode}-suffixed files and leaves the default ones alone."
        ),
    )
    arguments = parser.parse_args()
    mode = arguments.static_removal

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signals: dict[tuple[str, str], MotionSignal] = {}
    for run in RUNS:
        for camera in CAMERAS:
            target = signal_path(run, camera, mode)
            if target.is_file() and not arguments.force:
                signals[(run, camera)] = load_signal(run, camera, mode)
                print(f"{run}/{camera}: reusing cached signal")
                continue
            print(
                f"{run}/{camera}: estimating sub-pixel vertical motion "
                f"(static removal: {mode}) ...",
                flush=True,
            )
            shifts = raw_vertical_shifts(run, camera, mode)
            residual = remove_baseline(shifts)
            events = detect_events(residual)
            signal = MotionSignal(
                run, camera, residual, events, static_removal=mode, shifts=shifts
            )
            signals[(run, camera)] = signal
            atomic_write(
                target,
                lambda path, s=signal: np.savez_compressed(
                    path,
                    residual=s.residual,
                    events=np.array(s.events, dtype=np.int64),
                    shifts=s.shifts,
                ),
            )
            print(
                f"   sd {residual.std():.3f}px, {len(events)} events, "
                f"raw exact-zero fraction {float((shifts == 0.0).mean()):.3f}"
            )

    ratios = {
        camera: measure_likelihood_ratio(signals, camera, calibration_pairs(camera))
        for camera in CAMERAS
    }

    summary = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "sub-pixel phase correlation of the vertical image shift, rolling "
            "baseline removed, outliers at |z| > 3 merged into events"
        ),
        "static_removal": mode,
        "construction": {
            "default_mode": DEFAULT_STATIC_REMOVAL,
            "is_default": mode == DEFAULT_STATIC_REMOVAL,
            "what": (
                "each frame is divided by its own stream's temporal mean image "
                "before the phase correlation, so the camera-attached static "
                "component (rain droplets on the lens, a fixed vignette) is "
                "removed from what the correlator is shown"
            ),
            "modes_tested": list(STATIC_REMOVAL_MODES),
            "why": (
                "cam0 and cam5 are bolted to one chassis, so a real suspension event "
                "must fire in both streams of the SAME run at the same frame. That la"
                "bel-free test is what SELECTS among the three constructions, and all"
                " three are reported because two of them fail it. Run B, co-firing wi"
                "thin +/-2 frames at zero lag: 'none' (raw frames) 0 of 28; 'mean-div"
                "ide' 4 of 33 (3.85x chance, p ~ 0.019); 'median-subtract' 0 of 8. Ru"
                "n A: 4 of 51, 9 of 45 (4.75x, p ~ 9e-5), 10 of 45. Only mean-divide "
                "passes on BOTH runs, so it is the default. median-subtract is not me"
                "rely worse: it suppresses the Run B channel almost entirely (8 event"
                "s against mean-divide's 33), so its zero is a channel with nothing i"
                "n it rather than a channel that disagrees, and reporting only the tw"
                "o extremes would hide that."
            ),
            "what_the_test_is_and_is_not": (
                "The same-vehicle co-firing constraint is a selection criterion over "
                "constructions, not a validation of the winner. It says which estimat"
                "or is looking at the vehicle; it does not say the surviving events a"
                "re speed bumps, and no blind label was read at any point."
            ),
            "artifact": "outputs/task2_motion_bayes/static_removal_study.json",
            "withdrawn": (
                "The cross-run likelihood ratio of 2.83 measured under the "
                "'none' construction on four usable calibration pairs is "
                "withdrawn, not reinterpreted: the channel that produced it has "
                "been shown not to be measuring the vehicle. The earlier "
                "'7 Run B events against 22 in Run A' asymmetry was likewise an "
                "estimator artefact and is superseded by 16 against 24."
            ),
            "effect_on_the_consumer": (
                "nil either way: the posterior mapping is 97%+ identical frame "
                "by frame under both constructions, see "
                "outputs/task2_motion_bayes/motion_variant_mapping_comparison_"
                "mean-divide.json. The correction makes the channel honest, not "
                "useful."
            ),
        },
        "evidence_status": (
            "The likelihood ratio is measured on the original calibration labels, "
            "which are development data, and on a handful of usable pairs; it is "
            "a direction, not a number. The construction itself was selected on "
            "the label-free same-vehicle co-firing test instead. No blind label "
            "was read."
        ),
        "physical_limits": {
            "body_bounce_hz": "1-2, giving 5-10 samples per cycle at 10 FPS",
            "wheel_hop_hz": "10-15, aliased outright below the 5 Hz Nyquist limit",
            "implication": "an IMU or 30+ FPS video is the right instrument for this cue",
        },
        "signals": {f"{run}_{camera}": signals[(run, camera)].as_dict() for run in RUNS for camera in CAMERAS},
        "likelihood_ratios": ratios,
    }
    atomic_write(
        summary_path(mode),
        lambda path: path.write_text(json.dumps(summary, indent=2)),
    )

    print("\nMeasured evidence value of the motion channel:")
    for camera, ratio in ratios.items():
        if not ratio.get("usable"):
            print(f"  {camera}: not measurable ({ratio.get('reason', 'too few usable pairs')})")
            continue
        print(
            f"  {camera}: {ratio['of_those_matched_in_run_b']}/{ratio['pairs_with_a_run_a_event']} "
            f"true pairs co-fire, chance {ratio['chance_rate']:.3f} "
            f"-> likelihood ratio {ratio['likelihood_ratio']}"
        )
    print(f"\nWrote {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
