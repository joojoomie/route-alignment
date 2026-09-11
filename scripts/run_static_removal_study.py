#!/usr/bin/env python3
"""Is the suspension-event channel measuring the road, or the lens?

Run B was recorded in rain, and its lenses carry a field of droplets that is
identical in every frame. Unnormalised whole-frame cross correlation is driven
by whatever has the most energy, and a fixed pattern has all of its energy at
zero displacement: the correlator locks onto the droplets and reports that
nothing moved. The audit that motivated this study reported exactly that -
Run B cam0 returning a shift of *exactly* zero for 91% of consecutive pairs
against 13% in Run A - and the shipped event counts (Run A 22/29, Run B 7/21)
looked like the same story.

That audit figure does not survive measurement, and `pinning_premise` below
records the attempt: under the shipped estimator every stream sits at an
exact-zero fraction of 0.10-0.16, with Run A cam0 the *highest* of the four, and
the 76-89% readings appear only when the sub-pixel upsampling is dropped, which
quantises all four streams alike and has nothing to do with rain. The premise is
therefore withdrawn. The study survives it, because the case for static removal
never rested on that figure: it rests on check (d), which is a physical
constraint rather than an indicator.

This module scores three static-removal modes on evidence that needs no labels,
because the blind label set is spent and the calibration labels are development
data that has been read many times.

The checks, in the order they deserve to be believed:

a. **Exact-zero fraction of the raw shift.** A continuous physical quantity
   estimated to 1/20 pixel does not land on precisely zero a tenth of the time.
   This is a pinning indicator, not a motion statistic, and as `pinning_premise`
   shows it does not separate the runs, so it is reported and not ranked on.

b. **Residual dispersion**, and its Run B / Run A ratio. A pinned stream has an
   artificially small residual, and because the event threshold is
   stream-relative, that inflates or deflates event counts in ways that have
   nothing to do with bumps.

c. **Event counts**, reported but never ranked on: more events is not better.

d. **Same-run cross-camera co-firing.** The decisive one. cam0 and cam5 are
   bolted to one vehicle. When that vehicle crosses a bump, both cameras are
   shaken by the same chassis at the same instant, so a real suspension event
   must appear in both streams of the *same run* at the same frame, up to a
   fixed unproved inter-camera lag which is searched over -5..+5. This is a
   physical constraint, it uses no labels at all, and it is available in both
   runs. Scored against the chance rate implied by the event density, using the
   same construction as `measure_likelihood_ratio`, so a mode that simply fires
   more often gains nothing.

e. **Cross-run likelihood ratio** on the calibration pairs, exactly as the
   shipped script measures it. Reported for continuity, ranked below (d): it
   rests on development labels and on a handful of usable pairs, so it is a
   noisy statistic that can move a long way on one event.

Nothing here is an accuracy estimate. A mode that wins on (a) and (d) has been
shown to be measuring the vehicle rather than the lens; whether that improves
the final correspondence is a separate question, answered in
compare_motion_variant_mappings.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

import build_motion_events as motion


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = motion.OUTPUT_DIR
SHIFT_CACHE = OUTPUT_DIR / "static_removal_shifts"
TARGET = OUTPUT_DIR / "static_removal_study.json"

MODES = motion.STATIC_REMOVAL_MODES
RUNS = motion.RUNS
CAMERAS = motion.CAMERAS

# Both cameras are rigid to one chassis, so a genuine jolt is simultaneous to
# within the frame period. Two frames of slack covers the merge ambiguity in
# `detect_events`, which reports the centre of a run of outliers.
COFIRE_TOLERANCE = 2
LAG_SEARCH = range(-5, 6)


def signal_for(run: str, camera: str, mode: str) -> motion.MotionSignal:
    """The stream's signal, with raw shifts guaranteed present.

    The original `none` files predate the raw-shift field, and they are kept
    byte-for-byte as the record of the withdrawn construction, so the raw shifts
    for that mode are recomputed and cached beside them. The recomputation is
    checked against the stored residual, which is also a regression test on the
    refactor that made the mode selectable.
    """

    signal = motion.load_signal(run, camera, mode)
    if signal.shifts is not None:
        return signal
    SHIFT_CACHE.mkdir(parents=True, exist_ok=True)
    cached = SHIFT_CACHE / f"{run}_{camera}_{mode}.npy"
    if cached.is_file():
        signal.shifts = np.load(cached)
    else:
        print(f"   recomputing raw shifts for {run}/{camera} [{mode}] ...", flush=True)
        signal.shifts = motion.raw_vertical_shifts(run, camera, mode)
        np.save(cached, signal.shifts)
    return signal


def reproduction_error(signal: motion.MotionSignal) -> float:
    """How far the recomputed shifts are from the stored residual."""

    return float(np.abs(motion.remove_baseline(signal.shifts) - signal.residual).max())


def cofire(
    events_here: list[int], events_there: list[int], span_there: int, lag: int
) -> dict[str, object]:
    """Fraction of this camera's events with a partner in the other camera.

    `lag` shifts the other camera's events onto this one's frame index. The
    chance rate is the fraction of the route the other camera's events cover
    within the tolerance window - the same denominator `measure_likelihood_ratio`
    uses, so that firing more often cannot by itself raise the score.
    """

    window = 2 * COFIRE_TOLERANCE + 1
    chance = min(1.0, len(events_there) * window / max(span_there, 1))
    if not events_here or not events_there:
        return {"events": len(events_here), "hits": 0, "rate": None,
                "chance": round(chance, 4), "ratio": None}
    shifted = np.array(events_there, dtype=float) + lag
    hits = sum(
        1 for event in events_here if np.min(np.abs(shifted - event)) <= COFIRE_TOLERANCE
    )
    rate = hits / len(events_here)
    return {
        "events": len(events_here),
        "hits": hits,
        "rate": round(rate, 4),
        "chance": round(chance, 4),
        "ratio": round(rate / chance, 3) if chance > 0 else None,
    }


def binomial_tail(hits: int, trials: int, chance: float) -> float | None:
    """P(at least this many co-firings by luck), one sided.

    Approximate on purpose, and worth saying why. The two directions pooled here
    share their hits, so the trials are not independent, and the event positions
    within a stream are not either. It is reported to keep a handful of hits from
    reading as a result; it is not a test anyone should quote.
    """

    if trials <= 0 or not 0.0 < chance < 1.0:
        return None
    tail = sum(
        math.comb(trials, k) * chance**k * (1 - chance) ** (trials - k)
        for k in range(hits, trials + 1)
    )
    return round(float(min(1.0, tail)), 5)


def cross_camera_cofiring(
    signals: dict[tuple[str, str], motion.MotionSignal], run: str
) -> dict[str, object]:
    """Do the two cameras of one vehicle feel the same bumps at the same frame?"""

    cam0 = signals[(run, "cam0")]
    cam5 = signals[(run, "cam5")]

    def at(lag: int) -> dict[str, object]:
        forward = cofire(cam0.events, cam5.events, cam5.residual.size, lag)
        backward = cofire(cam5.events, cam0.events, cam0.residual.size, -lag)
        pooled_hits = forward["hits"] + backward["hits"]
        pooled_events = forward["events"] + backward["events"]
        pooled_chance = (forward["chance"] + backward["chance"]) / 2
        rate = pooled_hits / pooled_events if pooled_events else None
        return {
            "lag": lag,
            "cam0_with_cam5": forward,
            "cam5_with_cam0": backward,
            "pooled_rate": round(rate, 4) if rate is not None else None,
            "pooled_hits": pooled_hits,
            "pooled_events": pooled_events,
            "pooled_chance": round(pooled_chance, 4),
            "pooled_ratio": round(rate / pooled_chance, 3) if rate and pooled_chance else 0.0,
            "approximate_p_value": binomial_tail(pooled_hits, pooled_events, pooled_chance),
        }

    by_lag = [at(lag) for lag in LAG_SEARCH]
    zero = next(entry for entry in by_lag if entry["lag"] == 0)
    best = max(by_lag, key=lambda entry: (entry["pooled_hits"], -abs(entry["lag"])))
    return {
        "at_zero_lag": zero,
        "best_lag": best,
        "pooled_hits_by_lag": {str(entry["lag"]): entry["pooled_hits"] for entry in by_lag},
    }


def build(mode: str) -> dict[str, object]:
    signals = {
        (run, camera): signal_for(run, camera, mode) for run in RUNS for camera in CAMERAS
    }
    streams = {}
    for (run, camera), signal in signals.items():
        streams[f"{run}_{camera}"] = {
            "frames": int(signal.residual.size),
            "raw_zero_fraction": round(float((signal.shifts == 0.0).mean()), 4),
            "residual_sd_px": round(float(signal.residual.std()), 4),
            "event_count": len(signal.events),
            "events": signal.events,
            "shift_reproduction_error_px": round(reproduction_error(signal), 9),
        }
    sd_ratio = {
        camera: round(
            streams[f"runB_{camera}"]["residual_sd_px"]
            / max(streams[f"runA_{camera}"]["residual_sd_px"], 1e-9),
            3,
        )
        for camera in CAMERAS
    }
    return {
        "streams": streams,
        "runB_over_runA_sd_ratio": sd_ratio,
        "cross_camera_cofiring": {run: cross_camera_cofiring(signals, run) for run in RUNS},
        "cross_run_likelihood_ratio": {
            camera: motion.measure_likelihood_ratio(
                signals, camera, motion.calibration_pairs(camera)
            )
            for camera in CAMERAS
        },
    }


def pinning_premise() -> dict[str, object]:
    """Try to reproduce the audit figure that motivated this study.

    The audit reported an exact-zero shift for 91% of Run B cam0 pairs against
    13% in Run A, described as "whole-frame phase correlation". The shipped
    estimator crops and upsamples 20x, so this walks the obvious readings of
    that description - with and without the crop, with and without upsampling -
    and records what each one actually gives. A premise this study rests on is
    worth measuring rather than repeating.
    """

    from skimage.registration import phase_cross_correlation

    configurations = {
        "cropped_upsample20_shipped": (True, 20),
        "uncropped_upsample20": (False, 20),
        "cropped_upsample1": (True, 1),
        "uncropped_upsample1": (False, 1),
    }
    measured: dict[str, dict[str, float]] = {name: {} for name in configurations}
    for run in RUNS:
        for camera in CAMERAS:
            print(f"   pinning premise: {run}/{camera} ...", flush=True)
            stack = motion.grey_stack(run, camera).astype(np.float32)
            for name, (cropped, upsample) in configurations.items():
                frames = np.stack([motion.analysis_crop(f) for f in stack]) if cropped else stack
                shifts = np.array(
                    [
                        phase_cross_correlation(
                            frames[i], frames[i + 1], upsample_factor=upsample, normalization=None
                        )[0][0]
                        for i in range(len(frames) - 1)
                    ]
                )
                measured[name][f"{run}_{camera}"] = round(float((shifts == 0.0).mean()), 4)
    return {
        "claim_under_test": (
            "exact-zero shift for 91% of Run B cam0 and 72% of Run B cam5 "
            "consecutive pairs, against 13% Run A cam0 and 1% Run A cam5"
        ),
        "measured_zero_fraction": measured,
        "note": (
            "None of these readings reproduces the claimed asymmetry. At the "
            "shipped setting every stream sits near 0.10-0.16 with Run A cam0 "
            "the highest of the four, and dropping the upsampling pins all four "
            "streams alike because the shift is then quantised to whole pixels. "
            "The static-removal case therefore does not rest on this figure; it "
            "rests on check (d)."
        ),
    }


def print_table(results: dict[str, dict[str, object]]) -> None:
    print("\n(a) exact-zero fraction of the raw vertical shift - the pinning indicator")
    print(f"    {'mode':<16}" + "".join(f"{run}/{cam:<8}" for run in RUNS for cam in CAMERAS))
    for mode, entry in results.items():
        cells = "".join(
            f"{entry['streams'][f'{run}_{cam}']['raw_zero_fraction']:<14.3f}"
            for run in RUNS
            for cam in CAMERAS
        )
        print(f"    {mode:<16}{cells}")

    print("\n(b) residual SD, px, and the Run B / Run A ratio")
    print(f"    {'mode':<16}" + "".join(f"{run}/{cam:<8}" for run in RUNS for cam in CAMERAS)
          + "  B/A cam0  B/A cam5")
    for mode, entry in results.items():
        cells = "".join(
            f"{entry['streams'][f'{run}_{cam}']['residual_sd_px']:<14.3f}"
            for run in RUNS
            for cam in CAMERAS
        )
        ratio = entry["runB_over_runA_sd_ratio"]
        print(f"    {mode:<16}{cells}  {ratio['cam0']:<9} {ratio['cam5']}")

    print("\n(c) event counts")
    print(f"    {'mode':<16}" + "".join(f"{run}/{cam:<8}" for run in RUNS for cam in CAMERAS))
    for mode, entry in results.items():
        cells = "".join(
            f"{entry['streams'][f'{run}_{cam}']['event_count']:<14d}"
            for run in RUNS
            for cam in CAMERAS
        )
        print(f"    {mode:<16}{cells}")

    print("\n(d) SAME-RUN cross-camera co-firing within +/-2 frames (the decisive check)")
    print(f"    {'mode':<16}{'run':<7}{'hits/events':<14}{'rate':<8}{'chance':<9}"
          f"{'ratio':<8}{'p~':<10}{'best lag':<10}{'hits at best':<14}{'ratio at best'}")
    for mode, entry in results.items():
        for run in RUNS:
            block = entry["cross_camera_cofiring"][run]
            zero, best = block["at_zero_lag"], block["best_lag"]
            print(
                f"    {mode:<16}{run:<7}"
                f"{str(zero['pooled_hits']) + '/' + str(zero['pooled_events']):<14}"
                f"{(zero['pooled_rate'] or 0.0):<8.3f}{zero['pooled_chance']:<9.3f}"
                f"{zero['pooled_ratio']:<8.2f}{str(zero['approximate_p_value']):<10}"
                f"{best['lag']:<10d}"
                f"{str(best['pooled_hits']) + '/' + str(best['pooled_events']):<14}"
                f"{best['pooled_ratio']:.2f}"
            )

    print("\n(e) cross-run likelihood ratio on the calibration pairs (development labels)")
    print(f"    {'mode':<16}{'camera':<8}{'matched/usable pairs':<24}{'chance':<9}{'ratio'}")
    for mode, entry in results.items():
        for camera in CAMERAS:
            ratio = entry["cross_run_likelihood_ratio"][camera]
            pairs = f"{ratio.get('of_those_matched_in_run_b')}/{ratio.get('pairs_with_a_run_a_event')}"
            usable = "" if ratio.get("usable") else "  (too few pairs)"
            print(
                f"    {mode:<16}{camera:<8}{pairs:<24}"
                f"{ratio.get('chance_rate', float('nan')):<9}"
                f"{ratio.get('likelihood_ratio')}{usable}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--check-pinning-premise",
        action="store_true",
        help=(
            "Also re-measure the exact-zero fraction under four readings of "
            "'whole-frame phase correlation', to test the audit figure this "
            "study was launched on. Costs a second decode pass per stream."
        ),
    )
    arguments = parser.parse_args()
    if TARGET.is_file() and not arguments.force:
        raise FileExistsError(f"{TARGET} exists; pass --force to rebuild")

    missing = [
        motion.signal_path(run, camera, mode)
        for mode in MODES
        for run in RUNS
        for camera in CAMERAS
        if not motion.signal_path(run, camera, mode).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Build the variants first, for example "
            "`python3 scripts/build_motion_events.py --static-removal mean-divide`. "
            f"Missing: {[str(path.relative_to(ROOT)) for path in missing]}"
        )

    results = {}
    for mode in MODES:
        print(f"[{mode}]", flush=True)
        results[mode] = build(mode)

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Does removing each stream's camera-attached static image before "
            "phase correlation unpin the vertical shift estimate, and does the "
            "resulting event channel obey the physical constraint that two "
            "cameras on one chassis feel the same bump at the same frame?"
        ),
        "evidence_status": (
            "Checks (a)-(d) are label-free. Check (e) reuses the original "
            "calibration labels, which are development data read many times; it "
            "is reported for continuity and ranked below (d). No blind label was "
            "read and nothing here is an accuracy estimate."
        ),
        "definitions": {
            "raw_zero_fraction": "fraction of consecutive pairs whose estimated vertical shift is exactly 0.0",
            "cofire_tolerance_frames": COFIRE_TOLERANCE,
            "lag_search": [min(LAG_SEARCH), max(LAG_SEARCH)],
            "chance_rate": "events * (2*tolerance+1) / stream length, capped at 1",
            "unchanged": (
                "crop, upsampling factor, unnormalised correlation, baseline "
                "window, |z| > 3 threshold and event merging are identical in "
                "every mode; only what the correlator is shown differs"
            ),
        },
        "modes": results,
    }
    if arguments.check_pinning_premise:
        payload["pinning_premise"] = pinning_premise()
    motion.atomic_write(TARGET, lambda path: path.write_text(json.dumps(payload, indent=2)))
    print_table(results)
    print(f"\nWrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
