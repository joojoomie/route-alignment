#!/usr/bin/env python3
"""Tests for the label-free lateral-offset measurement.

The estimator has one job: separate a rescaling of the near field from a
translation of the whole image. Everything below is a synthetic point set where
the true answer is known by construction, so a regression in the estimator shows
up as a wrong number rather than as a plausible-looking one.

    PYTHONPATH=scripts python3 -m pytest scripts/test_measure_lateral_offset.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import measure_lateral_offset as lateral


WIDTH, HEIGHT = 896, 672


def synthetic_points(count: int = 200, seed: int = 7) -> np.ndarray:
    """Keypoints spread over the whole usable frame."""

    rng = np.random.default_rng(seed)
    usable = HEIGHT * lateral.USABLE_ROW_FRACTION
    return np.column_stack(
        [rng.uniform(0, WIDTH, count), rng.uniform(0, usable, count)]
    )


def apply_similarity(points: np.ndarray, scale: float, shift=(0.0, 0.0), centre=None) -> np.ndarray:
    centre = np.array([WIDTH / 2, HEIGHT / 2] if centre is None else centre, dtype=float)
    return (points - centre) * scale + centre + np.asarray(shift, dtype=float)


# --------------------------------------------------------------------------
# Band splitting
# --------------------------------------------------------------------------


def test_bands_are_disjoint_and_inside_the_rows_rootsift_uses():
    near_low, near_high = lateral.band_rows(HEIGHT, lateral.NEAR_BAND)
    far_low, far_high = lateral.band_rows(HEIGHT, lateral.FAR_BAND)
    usable = HEIGHT * lateral.USABLE_ROW_FRACTION
    assert far_low == 0.0
    assert far_high <= near_low                      # disjoint
    assert near_high == pytest.approx(usable)        # stops where RootSIFT stops
    assert near_high < HEIGHT                        # the wing mirror rows are excluded


def test_select_band_keeps_only_rows_of_that_band():
    points = synthetic_points()
    near_a, near_b = lateral.select_band(points, points, HEIGHT, lateral.NEAR_BAND)
    far_a, _ = lateral.select_band(points, points, HEIGHT, lateral.FAR_BAND)
    low, high = lateral.band_rows(HEIGHT, lateral.NEAR_BAND)
    assert near_a.shape == near_b.shape
    assert np.all(near_a[:, 1] >= low) and np.all(near_a[:, 1] < high)
    assert np.all(far_a[:, 1] < lateral.band_rows(HEIGHT, lateral.FAR_BAND)[1])
    assert len(near_a) and len(far_a)
    # No keypoint can be in both bands.
    assert not set(map(tuple, near_a.tolist())) & set(map(tuple, far_a.tolist()))


# --------------------------------------------------------------------------
# The estimator: pure shift must not look like a lane change
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [(0.0, 0.0), (37.5, 11.0), (-120.0, 0.0), (0.0, -25.0)])
def test_pure_translation_gives_unit_scale(shift):
    """Driving further along the road, and a camera yaw offset, are translations.

    Neither may produce a scale, or the measurement would report a lane change
    every time the frame pairing is a few frames off.
    """

    points_a = synthetic_points()
    points_b = points_a + np.asarray(shift)
    assert lateral.distance_ratio_scale(points_a, points_b) == pytest.approx(1.0, abs=1e-9)
    fit = lateral.band_scale(points_a, points_b)
    assert fit is not None
    assert fit["scale"] == pytest.approx(1.0, abs=1e-6)
    assert fit["log_scale"] == pytest.approx(0.0, abs=1e-6)
    assert fit["dx_px"] == pytest.approx(shift[0], abs=1e-6)
    assert fit["dy_px"] == pytest.approx(shift[1], abs=1e-6)


@pytest.mark.parametrize("scale", [0.90, 0.97, 1.03, 1.12])
def test_lateral_rescale_is_recovered(scale):
    """A lateral offset rescales the near field; both estimators must recover it."""

    points_a = synthetic_points()
    points_b = apply_similarity(points_a, scale, shift=(-40.0, 6.0))
    assert lateral.distance_ratio_scale(points_a, points_b) == pytest.approx(scale, rel=1e-9)
    fit = lateral.band_scale(points_a, points_b)
    assert fit["scale"] == pytest.approx(scale, rel=1e-5)
    assert fit["scale_ransac"] == pytest.approx(scale, rel=1e-5)


def test_scale_survives_gross_mismatches():
    """A quarter of the correspondences wrong, which real matching produces."""

    points_a = synthetic_points(count=160, seed=11)
    points_b = apply_similarity(points_a, 1.05)
    rng = np.random.default_rng(3)
    corrupt = rng.choice(len(points_b), 40, replace=False)
    points_b[corrupt] = rng.uniform(0, WIDTH, (40, 2))
    fit = lateral.band_scale(points_a, points_b)
    assert fit is not None
    assert fit["scale"] == pytest.approx(1.05, rel=1e-3)
    assert fit["inliers"] < len(points_a)      # the wrong matches were gated out
    assert fit["inliers"] >= 110               # and the right ones were not


def test_sparse_band_still_yields_a_scale():
    """The near band routinely carries a dozen matches; that must be enough.

    RANSAC is not stable at that support, which is why the distance-ratio
    estimator is the primary one. Here it must still land on the true scale
    with only MIN_BAND_MATCHES points and one of them wrong.
    """

    points_a = synthetic_points(count=lateral.MIN_BAND_MATCHES + 2, seed=17)
    points_b = apply_similarity(points_a, 1.04, shift=(20.0, -5.0))
    points_b[0] = np.array([10.0, 600.0])
    fit = lateral.band_scale(points_a, points_b)
    assert fit is not None
    assert fit["scale"] == pytest.approx(1.04, rel=2e-3)
    assert fit["inliers"] == len(points_a) - 1


def test_displacement_filter_is_model_free():
    """It must keep a genuine rescaling and drop only wild correspondences.

    A gate seeded from a fitted scale can lock onto a bad seed and reject a
    whole band - that is what happened on cam5 before this filter replaced one -
    and a gate that assumed scale 1 would suppress the very signal being
    measured. So: a 5% rescale must survive intact, and a wild match must not.
    """

    points_a = synthetic_points(count=60, seed=19)
    points_b = apply_similarity(points_a, 1.05, shift=(-15.0, 4.0))
    assert lateral.displacement_inliers(points_a, points_b).all()

    points_b[5] = np.array([10.0, 640.0])
    keep = lateral.displacement_inliers(points_a, points_b)
    assert not keep[5]
    assert keep.sum() == len(points_a) - 1


def test_displacement_filter_keeps_a_pure_translation_whole():
    points_a = synthetic_points(count=40, seed=21)
    for shift in ((0.0, 0.0), (-37.5, -11.0), (200.0, 0.0)):
        assert lateral.displacement_inliers(points_a, points_a + np.asarray(shift)).all()


def test_distance_ratio_ignores_pairs_that_are_too_close_together():
    """Ratios from near-coincident points are noise, not evidence."""

    rng = np.random.default_rng(5)
    points_a = rng.uniform(0, 20, (40, 2))          # every pair below the separation floor
    points_b = points_a * 1.5
    assert lateral.distance_ratio_scale(points_a, points_b) is None


def test_too_few_matches_returns_none():
    points = synthetic_points(count=lateral.MIN_BAND_MATCHES - 1)
    assert lateral.band_scale(points, points.copy()) is None


def test_near_band_rescales_while_far_band_does_not():
    """The end-to-end geometric claim, on a two-band synthetic frame.

    A lateral offset scales the near band and leaves the far band alone; the
    pipeline's split must see exactly that.
    """

    points = synthetic_points(count=400, seed=13)
    moved = points.copy()
    near_low, near_high = lateral.band_rows(HEIGHT, lateral.NEAR_BAND)
    in_near = (points[:, 1] >= near_low) & (points[:, 1] < near_high)
    moved[in_near] = apply_similarity(points[in_near], 1.06, shift=(-30.0, 0.0))
    moved[~in_near] = points[~in_near] + np.array([-30.0, 0.0])

    near = lateral.band_scale(*lateral.select_band(points, moved, HEIGHT, lateral.NEAR_BAND))
    far = lateral.band_scale(*lateral.select_band(points, moved, HEIGHT, lateral.FAR_BAND))
    assert near["scale"] == pytest.approx(1.06, rel=1e-4)
    assert far["scale"] == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------
# Calibration and sign logic
# --------------------------------------------------------------------------


def test_threshold_comes_from_the_far_band_and_respects_its_floor():
    rng = np.random.default_rng(1)
    quiet = rng.normal(0.0, 0.0005, 400)
    calibration = lateral.calibrate_threshold(quiet)
    # 3 x MAD of this is far below the floor, so the floor must win.
    assert calibration["threshold_abs_log_scale"] == lateral.MIN_ABS_LOG_SCALE
    assert calibration["threshold_is_floor"]

    noisy = rng.normal(0.0, 0.02, 400)
    calibration = lateral.calibrate_threshold(noisy)
    assert calibration["threshold_abs_log_scale"] > lateral.MIN_ABS_LOG_SCALE
    assert not calibration["threshold_is_floor"]
    assert calibration["threshold_abs_log_scale"] == pytest.approx(
        lateral.MAD_MULTIPLIER * lateral.mad(noisy)
    )


def test_threshold_centres_on_the_far_band_median():
    """A global scale bias common to both bands must not create flags."""

    offset = 0.03
    calibration = lateral.calibrate_threshold(np.full(50, offset))
    assert calibration["far_log_scale_median"] == pytest.approx(offset)
    # A near band with the same bias is centred to zero and cannot be flagged.
    centred = offset - calibration["far_log_scale_median"]
    assert abs(centred) < calibration["threshold_abs_log_scale"]


def test_direction_and_lane_side_are_opposite_on_the_two_cameras():
    """The physical cross-check, stated as code.

    Near content larger in run B means it was closer. cam0 looks left and cam5
    looks right, so the same manoeuvre must give them opposite signs.
    """

    assert lateral.direction_label(+0.05) == "closer_to_roadside_in_B"
    assert lateral.direction_label(-0.05) == "farther_from_roadside_in_B"
    assert lateral.direction_label(0.0) == "unchanged"

    # A shift toward the left kerb: cam0 closer (+), cam5 farther (-).
    assert lateral.lane_shift_direction("cam0", +0.05) == "left"
    assert lateral.lane_shift_direction("cam5", -0.05) == "left"
    # And toward the right kerb.
    assert lateral.lane_shift_direction("cam0", -0.05) == "right"
    assert lateral.lane_shift_direction("cam5", +0.05) == "right"


def test_sign_consistency_detects_the_physical_case_and_rejects_the_artefact():
    def records(camera, logs, frames):
        return [
            {
                "camera": camera,
                "runA_frame": frame,
                "near_log_scale_centred": value,
                "flagged": abs(value) > 0.02,
            }
            for frame, value in zip(frames, logs)
        ]

    frames = list(range(0, 1000, 50))
    signal = [0.05 * math.sin(index + 1) for index in range(len(frames))]  # never exactly zero
    physical = lateral.sign_consistency(
        records("cam0", signal, frames), records("cam5", [-v for v in signal], frames)
    )
    assert physical["opposite_sign_fraction_all"] == 1.0
    assert physical["pearson_log_scale"] == pytest.approx(-1.0, abs=1e-6)

    artefact = lateral.sign_consistency(
        records("cam0", signal, frames), records("cam5", signal, frames)
    )
    assert artefact["opposite_sign_fraction_all"] == 0.0
    assert artefact["pearson_log_scale"] == pytest.approx(1.0, abs=1e-6)


def test_sign_consistency_ignores_samples_with_no_partner_in_range():
    a = [{"camera": "cam0", "runA_frame": 0, "near_log_scale_centred": 0.05, "flagged": True}]
    b = [{"camera": "cam5", "runA_frame": 900, "near_log_scale_centred": -0.05, "flagged": True}]
    assert lateral.sign_consistency(a, b)["paired_samples"] == 0


# --------------------------------------------------------------------------
# Stretches and the chance model
# --------------------------------------------------------------------------


def record(frame, log_scale, flagged=True, camera="cam0"):
    return {
        "camera": camera,
        "runA_frame": frame,
        "runB_frame": frame + 5,
        "near_log_scale_centred": log_scale,
        "near_dy_px": 0.0,
        "flagged": flagged,
    }


def test_merge_stretches_joins_neighbours_and_splits_on_a_gap():
    records = [
        record(100, 0.04), record(124, 0.05), record(148, 0.05),
        record(600, -0.04),                       # far away: its own stretch
        record(1000, 0.03, flagged=False),        # unflagged: not a stretch at all
    ]
    stretches = lateral.merge_stretches(records)
    assert len(stretches) == 2
    assert (stretches[0]["runA_first"], stretches[0]["runA_last"]) == (100, 148)
    assert stretches[0]["samples"] == 3
    assert stretches[0]["sign"] == "+"
    assert stretches[0]["meaning"] == "closer_to_roadside_in_B"
    assert stretches[0]["lane_shift_toward"] == "left"      # cam0 faces left
    assert stretches[1]["runA_first"] == stretches[1]["runA_last"] == 600
    assert stretches[1]["sign"] == "-"


def test_merge_stretches_marks_a_sign_disagreement_as_mixed():
    stretches = lateral.merge_stretches([record(10, 0.05), record(40, -0.05)])
    assert len(stretches) == 1
    assert stretches[0]["sign"] == "mixed"
    assert stretches[0]["meaning"] == "mixed"


def test_merge_stretches_boundary_is_the_declared_gap():
    gap = lateral.STRETCH_MERGE_FRAMES
    assert len(lateral.merge_stretches([record(0, 0.05), record(gap, 0.05)])) == 1
    assert len(lateral.merge_stretches([record(0, 0.05), record(gap + 1, 0.05)])) == 2


def test_covered_fraction_unions_overlapping_stretches():
    pad = lateral.STRETCH_PAD_FRAMES
    one = [{"runA_first": 100, "runA_last": 200}]
    assert lateral.covered_fraction(one, 0, 1000) == pytest.approx((100 + 2 * pad) / 1000)
    # Two stretches whose padded intervals overlap must not be counted twice.
    two = [{"runA_first": 100, "runA_last": 200}, {"runA_first": 205, "runA_last": 300}]
    assert lateral.covered_fraction(two, 0, 1000) == pytest.approx((200 + 2 * pad) / 1000)
    assert lateral.covered_fraction([], 0, 1000) == 0.0


def test_binomial_tail_is_a_probability_and_moves_the_right_way():
    assert lateral.binomial_tail(0, 10, 0.3) == pytest.approx(1.0)
    assert lateral.binomial_tail(10, 10, 0.5) == pytest.approx(0.5**10)
    assert lateral.binomial_tail(8, 10, 0.1) < lateral.binomial_tail(4, 10, 0.1)
    assert lateral.binomial_tail(3, 0, 0.5) == 1.0


def test_mad_is_zero_for_a_constant_and_scaled_for_a_gaussian():
    assert lateral.mad(np.full(20, 4.2)) == 0.0
    rng = np.random.default_rng(0)
    assert lateral.mad(rng.normal(0, 1, 20000)) == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------
# Magnitude, pooling and support matching
# --------------------------------------------------------------------------


def test_implied_distance_change_inverts_the_scale():
    """Image size goes as 1/distance, so a band that grew was closer."""

    assert lateral.implied_distance_change(0.0) == pytest.approx(0.0)
    # Near content 10% larger in B means it was about 9% closer.
    assert lateral.implied_distance_change(math.log(1.10)) == pytest.approx(1 / 1.10 - 1)
    assert lateral.implied_distance_change(math.log(1.10)) < 0
    assert lateral.implied_distance_change(math.log(0.90)) > 0


def test_rolling_median_smooths_spikes_but_keeps_a_sustained_shift():
    """The pooled rule must suppress a lone outlier and keep a real stretch."""

    spike = np.zeros(9)
    spike[4] = 1.0
    assert lateral.rolling_median(spike).max() == 0.0

    sustained = np.array([0, 0, 0, 1.0, 1.0, 1.0, 0, 0, 0])
    pooled = lateral.rolling_median(sustained)
    assert pooled[4] == 1.0
    assert pooled.max() == 1.0
    # The ends are shortened, not padded, so nothing is invented there.
    assert lateral.rolling_median(np.array([5.0])) == pytest.approx([5.0])


def test_matched_support_uses_the_near_band_count():
    """The far band must be estimated with the near band's support, not its own.

    Otherwise the threshold is set by a quieter estimator than the one it
    thresholds, and pairs get flagged for having fewer near features rather than
    for having moved.
    """

    far_a = synthetic_points(count=300, seed=23)
    far_b = apply_similarity(far_a, 1.0, shift=(4.0, -2.0))
    fit = lateral.matched_support_scale(far_a, far_b, target=20, seed=1)
    assert fit["matches"] == 20
    # A far band already smaller than the target is used whole.
    small_a = synthetic_points(count=15, seed=29)
    small_b = small_a + np.array([3.0, 1.0])
    assert lateral.matched_support_scale(small_a, small_b, target=40, seed=1)["matches"] == 15


def test_band_scale_reports_its_own_support():
    points_a = synthetic_points(count=50, seed=31)
    points_b = apply_similarity(points_a, 1.02)
    fit = lateral.band_scale(points_a, points_b)
    assert fit["matches"] == 50 and fit["inliers"] == 50
    assert fit["row_median"] == pytest.approx(float(np.median(points_a[:, 1])))


# --------------------------------------------------------------------------
# The persistence gate: is the scatter a manoeuvre, or is it noise?
# --------------------------------------------------------------------------


def test_lag_one_autocorrelation_separates_persistent_from_white():
    rng = np.random.default_rng(2)
    white = rng.normal(0, 1, 400)
    assert abs(lateral.lag_one_autocorrelation(white)) < 0.2

    # A step that lasts: exactly the shape a lane change makes.
    step = np.concatenate([np.zeros(200), np.ones(200)])
    assert lateral.lag_one_autocorrelation(step) > 0.9

    # Alternating: the opposite of persistent.
    assert lateral.lag_one_autocorrelation(np.tile([1.0, -1.0], 100)) < -0.9

    assert lateral.lag_one_autocorrelation(np.zeros(3)) is None      # too short
    assert lateral.lag_one_autocorrelation(np.zeros(50)) is None     # no variation


def test_persistence_test_accepts_a_sustained_offset():
    """A lane change that lasts a fifth of the route must be called persistent."""

    rng = np.random.default_rng(3)
    series = rng.normal(0, 0.005, 60)
    series[20:32] += 0.06                       # a sustained near-field rescale
    far = rng.normal(0, 0.005, 60)
    result = lateral.persistence_test(series, far, trials=400)
    assert result["testable"]
    assert result["near_lag1_autocorrelation"] > 0
    assert result["permutation_p_lag1"] < lateral.PERSISTENCE_ALPHA
    assert result["permutation_p_pooled_excursion"] < lateral.PERSISTENCE_ALPHA
    assert result["persistent"]


def test_persistence_test_rejects_white_noise_however_large():
    """Large excursions in a random order are not a manoeuvre.

    This is the case that matters: the far-band threshold happily flags big
    per-pair excursions, and only this test can say they are noise.
    """

    rng = np.random.default_rng(4)
    loud = rng.normal(0, 0.08, 60)              # much larger than the case above
    far = rng.normal(0, 0.005, 60)
    result = lateral.persistence_test(loud, far, trials=400)
    assert result["testable"]
    assert not result["persistent"]
    assert result["permutation_p_lag1"] > lateral.PERSISTENCE_ALPHA


def test_persistence_null_holds_the_values_fixed():
    """The permutation null must reorder the data, never resample it.

    Otherwise a significant result could come from the values being large rather
    than from their being ordered, which is the whole point of the test.
    """

    rng = np.random.default_rng(5)
    series = rng.normal(0, 0.05, 40)
    result = lateral.persistence_test(series, series, trials=200)
    # The null brackets what the same values give in a random order, so a series
    # with no time structure lands inside the null interval.
    low, high = result["permutation_null_pooled_excursion_90pct"]
    assert low <= result["permutation_null_pooled_excursion_median"] <= high
    assert 0.0 <= result["permutation_p_pooled_excursion"] <= 1.0


def test_persistence_test_reports_untestable_short_series():
    assert lateral.persistence_test([0.1, 0.2], [0.1, 0.2])["testable"] is False


def test_detection_limits_shrink_as_more_samples_are_pooled():
    """The value of a negative result is its limit, and pooling improves it."""

    rng = np.random.default_rng(6)
    far = rng.normal(0, 0.02, 80)
    limits = lateral.detection_limits(far)
    changes = [d["min_detectable_lateral_distance_change"] for d in limits]
    assert changes == sorted(changes, reverse=True)     # longer manoeuvres are easier
    assert all(d["approx_runA_frames"] == d["window_samples"] * lateral.PAIR_STRIDE
               for d in limits)
    # The floor still applies: a silent far band cannot claim unlimited reach.
    quiet = lateral.detection_limits(np.zeros(80))
    assert all(
        d["threshold_abs_log_scale"] == lateral.MIN_ABS_LOG_SCALE for d in quiet
    )


def test_build_conclusion_reports_a_detection_and_a_null_differently():
    """The headline must not read the same whether or not anything was found."""

    limits = [{"window_samples": 3, "min_detectable_lateral_distance_change": 0.03}]
    null = lateral.build_conclusion(
        {
            "cam0": {"usable": True, "lateral_offset_detected": False,
                     "detection_limits": limits},
            "cam5": {"usable": True, "lateral_offset_detected": False,
                     "detection_limits": limits},
        },
        {"paired_samples": 30},
    )
    assert null["lateral_offset_detected"] is False
    assert null["cameras_with_detection"] == []
    assert "detection_limits" in null["answer"]
    assert null["detection_limits_by_pooling_window"]["cam0"]["3"] == 0.03

    found = lateral.build_conclusion(
        {
            "cam0": {"usable": True, "lateral_offset_detected": True,
                     "detection_limits": limits},
            "cam5": {"usable": True, "lateral_offset_detected": False,
                     "detection_limits": limits},
        },
        {"paired_samples": 30},
    )
    assert found["lateral_offset_detected"] is True
    assert found["cameras_with_detection"] == ["cam0"]
    assert found["answer"] != null["answer"]


if __name__ == "__main__":  # `python scripts/test_measure_lateral_offset.py` must run these, not exit 0
    # These are pytest-style functions (two of them parametrised), so the file
    # delegates to pytest rather than pretending to be a unittest module. The
    # exit status is pytest's, so a failure fails the shell command.
    raise SystemExit(pytest.main([__file__, "-q"]))
