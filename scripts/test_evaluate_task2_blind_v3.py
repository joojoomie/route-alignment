#!/usr/bin/env python3
"""Unit tests for the v3 blind scoring, on synthetic labels only.

These pin the parts of the evaluation that are easy to get quietly wrong: which
labels enter the precision denominator, which direction a `no_correspondence`
row is scored in, and whether a gate turns an acceptance into an abstention. All
inputs are constructed here; nothing reads or rescores the sealed label set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import evaluate_task2_blind_v3 as ev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "task2_evaluation" / "blind_v3" / "blind_v3_results.json"
RESULTS_V3B = (
    ROOT / "outputs" / "task2_evaluation" / "blind_v3b" / "blind_v3b_results.json"
)
SEAL_V3B = ROOT / "outputs" / "task2_evaluation" / "blind_v3b" / "label_seal.json"
FREEZE_V3B = ROOT / "outputs" / "task2_evaluation" / "method_freeze_v3b.json"


def case(
    label,
    accepted=True,
    distance=0,
    stratum="corridor",
    camera="cam0",
    signed=None,
):
    scored = accepted and label == "match"
    if signed is None:
        signed = distance
    return {
        "method": "synthetic",
        "query_id": f"{camera}_{label}_{distance}_{signed}",
        "camera_id": camera,
        "runA_frame": 1000,
        "stratum": stratum,
        "ground_truth_label": label,
        "accepted": accepted,
        "gated_by": "" if accepted else "near_kink",
        "correct": (distance == 0) if scored else "",
        "interval_distance": distance if scored else "",
        "preferred_frame_error": distance if scored else "",
        "signed_interval_error": signed if scored else "",
        "signed_preferred_error": signed if scored else "",
    }


class ScoreTest(unittest.TestCase):
    def test_undetermined_leaves_the_denominator(self):
        """An undetermined label is not a negative and not a positive."""

        with_undetermined = ev.score(
            [case("match", distance=0), case("match", distance=5), case("undetermined")]
        )
        without = ev.score([case("match", distance=0), case("match", distance=5)])
        self.assertEqual(with_undetermined["accepted"], without["accepted"])
        self.assertEqual(with_undetermined["precision"], without["precision"])
        self.assertEqual(with_undetermined["precision"], 0.5)
        self.assertEqual(with_undetermined["undetermined"]["accepted"], 1)

    def test_no_correspondence_acceptance_is_a_false_accept(self):
        accepted = ev.score([case("no_correspondence")])
        self.assertEqual(accepted["no_correspondence"]["false_accepts"], 1)
        self.assertEqual(accepted["no_correspondence"]["true_rejects"], 0)
        # ... and it never enters the match-label precision.
        self.assertIsNone(accepted["precision"])

        abstained = ev.score([case("no_correspondence", accepted=False)])
        self.assertEqual(abstained["no_correspondence"]["false_accepts"], 0)
        self.assertEqual(abstained["no_correspondence"]["true_rejects"], 1)

    def test_within_two_uses_interval_distance_not_correctness(self):
        result = ev.score([case("match", distance=2), case("match", distance=3)])
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["within_two"], 1)

    def test_abstaining_removes_the_label_from_the_denominator(self):
        result = ev.score([case("match", distance=9, accepted=False), case("match")])
        self.assertEqual(result["match_labels"], 2)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["coverage_of_match_labels"], 0.5)
        self.assertEqual(result["gated_match_labels"], 1)

    def test_miss_rate_counts_only_accepted_match_labels(self):
        result = ev.miss_rate(
            [
                case("match", distance=0),
                case("match", distance=4),
                case("match", distance=7, accepted=False),
                case("undetermined"),
            ]
        )
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["misses_interval_distance_gt_0"], 1)
        self.assertEqual(result["miss_rate"], 0.5)
        self.assertEqual(result["abstained"], 1)


class SignedErrorTest(unittest.TestCase):
    """The reading that separated the v3 annotator-tool defect from real error."""

    def test_sign_says_which_side_of_the_interval_the_prediction_falls(self):
        self.assertEqual(ev.signed_distance_to_interval(105, 100, 102), 3)
        self.assertEqual(ev.signed_distance_to_interval(97, 100, 102), -3)
        for inside in (100, 101, 102):
            self.assertEqual(ev.signed_distance_to_interval(inside, 100, 102), 0)

    def test_magnitude_matches_the_unsigned_distance(self):
        for prediction in range(90, 115):
            self.assertEqual(
                abs(ev.signed_distance_to_interval(prediction, 100, 102)),
                ev.distance_to_interval(prediction, 100, 102),
            )

    def test_summary_counts_late_and_early_separately(self):
        result = ev.signed_error_summary(
            [
                case("match", distance=3, signed=3),
                case("match", distance=4, signed=4),
                case("match", distance=2, signed=-2),
                case("match", distance=0, signed=0),
            ]
        )
        signed = result["signed_interval_error"]
        self.assertEqual(result["n"], 4)
        self.assertEqual(signed["later_than_interval"], 2)
        self.assertEqual(signed["earlier_than_interval"], 1)
        self.assertEqual(signed["inside_interval"], 1)
        self.assertEqual(signed["min"], -2)
        self.assertEqual(signed["max"], 4)

    def test_a_symmetric_set_has_a_zero_mean_an_unsigned_read_would_hide(self):
        """Two sets with identical unsigned error, one biased and one not."""

        symmetric = ev.signed_error_summary(
            [case("match", distance=4, signed=4), case("match", distance=4, signed=-4)]
        )
        biased = ev.signed_error_summary(
            [case("match", distance=4, signed=4), case("match", distance=4, signed=4)]
        )
        self.assertEqual(
            ev.distribution([4, 4])["median"], ev.distribution([4, 4])["median"]
        )
        self.assertEqual(symmetric["signed_interval_error"]["mean"], 0.0)
        self.assertEqual(biased["signed_interval_error"]["mean"], 4.0)

    def test_abstained_and_non_match_rows_are_excluded(self):
        result = ev.signed_error_summary(
            [
                case("match", distance=3, signed=3),
                case("match", distance=9, signed=9, accepted=False),
                case("no_correspondence"),
                case("undetermined"),
            ]
        )
        self.assertEqual(result["n"], 1)


class GateTest(unittest.TestCase):
    def setUp(self):
        self.exposure = {
            ("runA", "cam0", 10): "ok",
            ("runA", "cam0", 20): "too_dark",
            ("runB", "cam0", 500): "too_dark",
            ("runB", "cam0", 600): "ok",
        }
        self.slope = {("runA", "cam0", 10): "near_kink", ("runA", "cam0", 20): ""}

    def test_slope_gate_reads_the_frozen_column(self):
        self.assertEqual(
            ev.gate_blocks(("slope",), "cam0", 10, "600", self.exposure, self.slope),
            "near_kink",
        )
        self.assertEqual(
            ev.gate_blocks(("slope",), "cam0", 20, "600", self.exposure, self.slope), ""
        )

    def test_dark_gate_fires_on_either_side_of_the_pair(self):
        self.assertEqual(
            ev.gate_blocks(("dark",), "cam0", 20, "600", self.exposure, self.slope),
            "runA_too_dark",
        )
        self.assertEqual(
            ev.gate_blocks(("dark",), "cam0", 10, "500", self.exposure, self.slope),
            "runB_too_dark",
        )
        self.assertEqual(
            ev.gate_blocks(("dark",), "cam0", 10, "600", self.exposure, self.slope), ""
        )

    def test_gates_are_independent(self):
        """A near_kink row is untouched by the dark gate and vice versa."""

        self.assertEqual(
            ev.gate_blocks(("dark",), "cam0", 10, "600", self.exposure, self.slope), ""
        )
        self.assertEqual(
            ev.gate_blocks(("slope", "dark"), "cam0", 10, "600", self.exposure, self.slope),
            "near_kink",
        )


class ScoredArtifactTest(unittest.TestCase):
    """Consistency checks on the single scoring pass, if it has been run."""

    def setUp(self):
        if not RESULTS.is_file():
            self.skipTest("blind v3 has not been scored in this checkout")
        self.payload = json.loads(RESULTS.read_text())

    def test_counts_are_internally_consistent(self):
        for method, block in self.payload["methods"].items():
            for camera, result in block["cameras"].items():
                self.assertLessEqual(
                    result["correct"], result["accepted"], f"{method}/{camera}"
                )
                self.assertLessEqual(
                    result["accepted"], result["match_labels"], f"{method}/{camera}"
                )
                self.assertEqual(
                    result["labels"],
                    result["match_labels"]
                    + result["undetermined_labels"]
                    + result["no_correspondence_labels"],
                    f"{method}/{camera}",
                )

    def test_gates_only_ever_remove_acceptances(self):
        """A gate is a policy over the frozen mapping; it cannot add answers."""

        base = self.payload["methods"]["submitted_unified"]["cameras"]
        for method in (
            "submitted_plus_slope_gate",
            "submitted_plus_dark_gate",
            "submitted_plus_both_gates",
        ):
            for camera, result in self.payload["methods"][method]["cameras"].items():
                self.assertLessEqual(
                    result["accepted"], base[camera]["accepted"], f"{method}/{camera}"
                )

    def test_the_disclosure_is_carried_into_the_result(self):
        self.assertIn(
            "corridor stratum is the unbiased comparison",
            self.payload["stratum_dependency_disclosure"],
        )


class RedoSetTest(unittest.TestCase):
    """The v3b CAM5 redo: same methods, same freeze contents, one camera.

    The point of the redo is that only the annotator tool changed, so these
    assertions are about what must NOT have moved between v3 and v3b, plus the
    two facts that make the redo readable at all: the seal still matches the
    label file byte for byte, and the viewer's seeded pose default now carries
    the compensation sign rather than the raw measurement's.
    """

    def setUp(self):
        if not RESULTS_V3B.is_file():
            self.skipTest("blind v3b has not been scored in this checkout")
        self.payload = json.loads(RESULTS_V3B.read_text())

    def test_it_scores_cam5_alone(self):
        self.assertEqual(self.payload["cameras_scored"], ["cam5"])
        self.assertEqual(self.payload["label_set"], "blind_v3b")
        for block in self.payload["methods"].values():
            self.assertEqual(list(block["cameras"]), ["cam5"])

    def test_the_seal_still_covers_the_label_file(self):
        seal = json.loads(SEAL_V3B.read_text())
        digest = hashlib.sha256((ROOT / seal["label_file"]).read_bytes()).hexdigest()
        self.assertEqual(digest, seal["label_file_sha256"])
        self.assertEqual(self.payload["label_seal_sha256"], seal["label_file_sha256"])

    def test_the_result_cites_the_freeze_on_disk_and_the_seal_cites_it_too(self):
        freeze = json.loads(FREEZE_V3B.read_text())
        self.assertEqual(self.payload["method_freeze_v3_sha256"], freeze["freeze_sha256"])
        seal = json.loads(SEAL_V3B.read_text())
        self.assertEqual(
            seal["method_freeze_sha256_at_seal_time"], freeze["freeze_sha256"]
        )

    def test_no_scored_method_changed_between_v3_and_v3b(self):
        """The redo freeze must hash the same artifacts as the v3 freeze."""

        v3b = json.loads(FREEZE_V3B.read_text())
        v3_path = ROOT / "outputs" / "task2_evaluation" / "method_freeze_v3.json"
        if not v3_path.is_file():
            self.skipTest("no v3 freeze in this checkout")
        v3 = json.loads(v3_path.read_text())
        self.assertEqual(v3b["artifact_sha256"], v3["artifact_sha256"])
        self.assertEqual(v3b["unified_parameters"], v3["unified_parameters"])
        self.assertEqual(v3b["slope_gate_rule"], v3["slope_gate_rule"])
        self.assertNotEqual(v3b["query_set_sha256"], v3["query_set_sha256"])

    def test_the_pose_default_now_carries_the_compensation_sign(self):
        cam5 = self.payload["pose_convention_check"]["cameras"]["cam5"]
        self.assertEqual(cam5["seeded_default"], [40, 12])
        self.assertEqual(cam5["seeded_default_convention"], "compensation")
        self.assertTrue(cam5["seeded_default_sign_is_correct"])
        # The defect worked by tempting the annotator off the default; nobody
        # moved a slider on this set, so it could not have worked here.
        self.assertEqual(
            cam5["rows_left_at_the_seeded_default"], self.payload["labels"]
        )

    def test_the_signed_error_is_reported_against_the_withdrawn_v3_set(self):
        check = self.payload["signed_error_check"]
        self.assertTrue(check["v3_cam5_reference"]["available"])
        self.assertIn("systematic_offset_gone_on_cam5", check)
        self.assertIsInstance(check["systematic_offset_gone_on_cam5"], bool)
        signed = self.payload["methods"]["submitted_unified"]["signed_error"]["cam5"]
        counts = signed["signed_interval_error"]
        self.assertEqual(
            counts["later_than_interval"]
            + counts["earlier_than_interval"]
            + counts["inside_interval"],
            signed["n"],
        )
        self.assertEqual(
            signed["n"],
            self.payload["methods"]["submitted_unified"]["cameras"]["cam5"]["accepted"],
        )

    def test_the_v2_comparison_recomputes_the_published_v2_pair(self):
        comparison = self.payload["v2_comparison"]
        self.assertTrue(comparison["available"])
        self.assertEqual(comparison["v2"]["precision"], 0.375)
        self.assertEqual(comparison["v2"]["accepted"], 32)
        self.assertEqual(comparison["v2"]["within_two_by_preferred_frame_error"], 21)
        corridor = comparison["this_set_corridor_only"]
        overall = comparison["this_set_overall"]
        submitted = self.payload["methods"]["submitted_unified"]
        self.assertEqual(corridor, self._brief(submitted["strata"]["cam5"]["corridor"]))
        self.assertEqual(overall, self._brief(submitted["cameras"]["cam5"]))

    @staticmethod
    def _brief(result: dict) -> dict:
        return {
            "match_labels": result["match_labels"],
            "accepted": result["accepted"],
            "correct": result["correct"],
            "precision": result["precision"],
            "precision_wilson_95": result["precision_wilson_95"],
            "within_two_by_interval_distance": result["within_two"],
            "coverage_of_match_labels": result["coverage_of_match_labels"],
        }

    def test_counts_are_internally_consistent(self):
        for method, block in self.payload["methods"].items():
            result = block["cameras"]["cam5"]
            self.assertLessEqual(result["correct"], result["accepted"], method)
            self.assertLessEqual(result["accepted"], result["match_labels"], method)
            strata = block["strata"]["cam5"]
            self.assertEqual(
                sum(strata[stratum]["labels"] for stratum in ev.STRATA),
                result["labels"],
                method,
            )

    def test_the_exported_strata_are_the_ones_the_manifest_recorded(self):
        counts = self.payload["strata_counts_exported"]["cam5"]
        self.assertEqual(counts, {"occlusion": 0, "dark": 1, "kink": 5, "corridor": 18})
        shortfalls = self.payload["strata_shortfalls"]["cam5"]
        self.assertEqual(
            {row["stratum"] for row in shortfalls}, {"dark", "kink"}
        )

    def test_the_gates_still_only_remove_acceptances(self):
        base = self.payload["methods"]["submitted_unified"]["cameras"]["cam5"]
        for method in (
            "submitted_plus_slope_gate",
            "submitted_plus_dark_gate",
            "submitted_plus_both_gates",
        ):
            result = self.payload["methods"][method]["cameras"]["cam5"]
            self.assertLessEqual(result["accepted"], base["accepted"], method)


SENSITIVITY_V3B = (
    ROOT
    / "outputs"
    / "task2_evaluation"
    / "blind_v3b"
    / "convention_shift_sensitivity.json"
)
POSE_DIAGNOSTIC = ROOT / "outputs" / "task2_pose_offset" / "pose_offset_diagnostic.json"


def shift_case(
    prediction,
    low,
    high,
    preferred=None,
    camera="cam5",
    label="match",
    accepted=True,
    method="submitted_unified",
    query_id=None,
):
    """A scored row in the shape `blind_*_cases.csv` writes, for shift tests."""

    distance = ev.distance_to_interval(prediction, low, high) if accepted and label == "match" else ""
    signed = (
        ev.signed_distance_to_interval(prediction, low, high)
        if accepted and label == "match"
        else ""
    )
    if preferred is None:
        preferred = (low + high) // 2
    return {
        "method": method,
        "query_id": query_id or f"{camera}_{prediction}_{low}_{high}",
        "camera_id": camera,
        "runA_frame": 1000 + prediction,
        "stratum": "corridor",
        "ground_truth_label": label,
        "runB_min": str(low),
        "runB_max": str(high),
        "raw_predicted_runB_frame": str(prediction),
        "gated_by": "" if accepted else "near_kink",
        "accepted": accepted,
        "predicted_runB_frame": str(prediction) if accepted else "",
        "correct": (distance == 0) if distance != "" else "",
        "interval_distance": distance,
        "preferred_frame_error": abs(prediction - preferred) if distance != "" else "",
        "signed_interval_error": signed,
        "signed_preferred_error": prediction - preferred if distance != "" else "",
    }, preferred


class ConventionShiftTest(unittest.TestCase):
    """A constant shift, applied to already-scored rows, on synthetic labels.

    The reading this pins is narrow and easy to overstate, so the tests fix the
    two things that make it meaningful: a known constant offset must be exactly
    what a shift of its negation removes, and the shift must not be able to
    manufacture coverage, move a different camera or touch an abstention.
    """

    def _rows(self, offsets, camera="cam5", method="submitted_unified"):
        cases = []
        preferred = {}
        for index, offset in enumerate(offsets):
            low = 500 + 10 * index
            case, target = shift_case(
                low + offset,
                low,
                low,
                preferred=low,
                camera=camera,
                method=method,
                query_id=f"{camera}_q{index}",
            )
            cases.append(case)
            preferred[case["query_id"]] = target
        return cases, preferred

    def test_a_known_constant_offset_is_removed_by_shifting_its_negation(self):
        """Every prediction is exactly +3 late; -3 makes every one of them a hit."""

        cases, preferred = self._rows([3, 3, 3, 3, 3])
        self.assertEqual(ev.score(cases)["precision"], 0.0)
        restored = ev.score(ev.shift_cases(cases, "cam5", -3, preferred))
        self.assertEqual(restored["correct"], 5)
        self.assertEqual(restored["precision"], 1.0)
        self.assertEqual(restored["accepted"], 5)
        # ... and the wrong shift does not.
        for wrong in (-1, -2, -4, 0):
            self.assertLess(
                ev.score(ev.shift_cases(cases, "cam5", wrong, preferred))["correct"], 5
            )

    def test_the_peak_sits_at_the_offset_even_when_it_is_not_exact(self):
        """A scattered late bias still peaks at the shift nearest its centre."""

        cases, preferred = self._rows([1, 2, 2, 2, 3, 2, 2])
        scores = {
            shift: ev.score(ev.shift_cases(cases, "cam5", shift, preferred))["correct"]
            for shift in (0, -1, -2, -3)
        }
        self.assertEqual(max(scores, key=lambda key: scores[key]), -2)
        self.assertEqual(scores[0], 0)

    def test_the_shift_cannot_create_coverage_or_move_a_rejection(self):
        cases, preferred = self._rows([3, 3])
        abstained, _ = shift_case(999, 500, 500, accepted=False)
        absent, _ = shift_case(700, 0, 0, label="no_correspondence")
        rows = cases + [abstained, absent]
        before = ev.score(rows)
        after = ev.score(ev.shift_cases(rows, "cam5", -3, preferred))
        self.assertEqual(after["accepted"], before["accepted"])
        self.assertEqual(after["match_labels"], before["match_labels"])
        self.assertEqual(
            after["no_correspondence"]["false_accepts"],
            before["no_correspondence"]["false_accepts"],
        )
        shifted_abstention = [
            case for case in ev.shift_cases(rows, "cam5", -3, preferred)
            if not case["accepted"] and case["ground_truth_label"] == "match"
        ][0]
        self.assertEqual(shifted_abstention["predicted_runB_frame"], "")

    def test_only_the_named_camera_moves(self):
        cam5, preferred5 = self._rows([3, 3], camera="cam5")
        cam0, preferred0 = self._rows([3, 3], camera="cam0")
        preferred = {**preferred5, **preferred0}
        moved = ev.shift_cases(cam5 + cam0, "cam5", -3, preferred)
        self.assertEqual(
            ev.score([case for case in moved if case["camera_id"] == "cam5"])["correct"], 2
        )
        self.assertEqual(
            ev.score([case for case in moved if case["camera_id"] == "cam0"])["correct"], 0
        )

    def test_the_signed_error_collapses_to_zero_at_the_right_shift(self):
        cases, preferred = self._rows([2, 2, 2, 2])
        summary = ev.signed_error_summary(ev.shift_cases(cases, "cam5", -2, preferred))
        self.assertEqual(summary["signed_interval_error"]["median"], 0.0)
        self.assertEqual(summary["signed_interval_error"]["later_than_interval"], 0)
        self.assertEqual(summary["signed_interval_error"]["inside_interval"], 4)


class ConventionConstantTest(unittest.TestCase):
    """The constant is read from the pose diagnostic; it is never typed."""

    def test_it_is_read_from_the_file_and_not_hard_coded(self):
        original = ev.POSE_DIAGNOSTIC
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "pose.json"
            fake.write_text(
                json.dumps(
                    {
                        "built_at_utc": "1999-01-01T00:00:00+00:00",
                        "degeneracy": "synthetic",
                        "cameras": {
                            "cam5": {
                                "label_convention_conversion": {
                                    "gap_px": -44.0,
                                    "gap_frames_at_median_sweep": -4.4,
                                }
                            }
                        },
                    }
                )
            )
            ev.POSE_DIAGNOSTIC = fake
            try:
                constant = ev.convention_constant("cam5")
            finally:
                ev.POSE_DIAGNOSTIC = original
        self.assertEqual(constant["gap_frames_at_median_sweep"], -4.4)
        self.assertEqual(constant["rounded_frames"], -4)
        self.assertEqual(constant["source_built_at_utc"], "1999-01-01T00:00:00+00:00")

    def test_the_real_diagnostic_gives_minus_two_on_cam5_and_zero_on_cam0(self):
        if not POSE_DIAGNOSTIC.is_file():
            self.skipTest("no pose diagnostic in this checkout")
        payload = json.loads(POSE_DIAGNOSTIC.read_text())["cameras"]
        for camera in ("cam0", "cam5"):
            constant = ev.convention_constant(camera)
            self.assertEqual(
                constant["gap_frames_at_median_sweep"],
                payload[camera]["label_convention_conversion"][
                    "gap_frames_at_median_sweep"
                ],
            )
            self.assertEqual(constant["rounded_frames"], round(constant["gap_frames_at_median_sweep"]))
        # CAM0's pose did not change between the runs, so it needs no shift.
        self.assertEqual(ev.convention_constant("cam0")["rounded_frames"], 0)
        self.assertEqual(ev.convention_constant("cam5")["rounded_frames"], -2)


class ConventionSensitivityArtifactTest(unittest.TestCase):
    """The written reading, if it has been computed in this checkout."""

    def setUp(self):
        if not SENSITIVITY_V3B.is_file():
            self.skipTest("the convention sensitivity has not been computed here")
        self.payload = json.loads(SENSITIVITY_V3B.read_text())

    def test_it_did_not_rewrite_the_single_scoring_pass(self):
        recorded = self.payload["scored_pass"]
        digest = hashlib.sha256(RESULTS_V3B.read_bytes()).hexdigest()
        self.assertEqual(digest, recorded["sha256"])
        self.assertEqual(
            json.loads(RESULTS_V3B.read_text())["evaluated_at_utc"],
            recorded["evaluated_at_utc"],
        )

    def test_it_is_labelled_a_reading_and_not_a_held_out_number(self):
        self.assertIn("NOT A HELD-OUT NUMBER", self.payload["evidence_status"])
        self.assertIn("submitted mapping files are unchanged", self.payload["the_submission_is_not_shifted"])

    def test_the_constant_carries_its_provenance(self):
        constant = self.payload["convention_shift_sensitivity"]["cam5"]["constant"]
        self.assertEqual(
            constant["source_file"], "outputs/task2_pose_offset/pose_offset_diagnostic.json"
        )
        self.assertTrue(constant["source_built_at_utc"])
        self.assertEqual(constant["gap_frames_at_median_sweep"], -1.79)
        self.assertEqual(constant["rounded_frames"], -2)

    def test_the_unshifted_row_reproduces_the_scored_precision(self):
        """Shift 0 must be exactly what the single pass reported, or the
        rescoring is measuring something other than the scored set."""

        scored = json.loads(RESULTS_V3B.read_text())["methods"]
        block = self.payload["convention_shift_sensitivity"]["cam5"]["methods"]
        for method, rows in block.items():
            unshifted = rows["by_shift"]["0"]
            reference = scored[method]["cameras"]["cam5"]
            self.assertEqual(unshifted["correct"], reference["correct"], method)
            self.assertEqual(unshifted["accepted"], reference["accepted"], method)
            self.assertEqual(unshifted["precision"], reference["precision"], method)
            self.assertEqual(
                unshifted["precision_wilson_95"], reference["precision_wilson_95"], method
            )

    def test_every_method_peaks_at_the_measured_constant(self):
        camera = self.payload["convention_shift_sensitivity"]["cam5"]
        self.assertEqual(camera["measured_shift_frames"], -2)
        self.assertEqual(camera["shifts_frames"], [0, -1, -2, -3])
        for method, rows in camera["methods"].items():
            self.assertEqual(rows["peak_shift_frames"], -2, method)
            self.assertTrue(rows["peak_is_the_measured_constant"], method)

    def test_the_three_scored_mappings_are_all_present(self):
        methods = self.payload["convention_shift_sensitivity"]["cam5"]["methods"]
        self.assertEqual(
            set(methods),
            {"submitted_unified", "slope_refinement_variant", "posterior_model"},
        )


class RedoContractTest(unittest.TestCase):
    """The redo is held to the same export -> freeze -> label -> seal ordering."""

    def test_the_contract_verifies_for_the_redo_set(self):
        if not SEAL_V3B.is_file():
            self.skipTest("no sealed v3b set in this checkout")
        contract = ev.check_contract("blind_v3b")
        seal = contract["seal"]
        self.assertEqual(seal["label_set"], "blind_v3b")
        self.assertTrue(seal["chain"]["ordering_holds"])
        chain = seal["chain"]
        self.assertLess(
            chain["query_set_exported_at_utc"], chain["method_frozen_at_utc"]
        )
        self.assertLess(chain["method_frozen_at_utc"], chain["labels_sealed_at_utc"])
        self.assertEqual(ev.cameras_of(contract["manifest"]), ("cam5",))

    def test_the_two_sets_do_not_share_a_query(self):
        """The redo must not re-ask a frame the defective run already asked."""

        if not ev.manifest_file("blind_v3b").is_file():
            self.skipTest("no v3b manifest in this checkout")
        v3b = json.loads(ev.manifest_file("blind_v3b").read_text())
        v3 = json.loads(ev.manifest_file("blind_v3").read_text())
        redo = {query["runAFrame"] for query in v3b["cameras"]["cam5"]["queries"]}
        original = {query["runAFrame"] for query in v3["cameras"]["cam5"]["queries"]}
        self.assertEqual(redo & original, set())

    def test_load_strata_reads_the_redo_manifest(self):
        if not ev.manifest_file("blind_v3b").is_file():
            self.skipTest("no v3b manifest in this checkout")
        strata = ev.load_strata("blind_v3b")
        self.assertEqual(len(strata), 24)
        self.assertTrue(all(key.startswith("cam5_v3b_") for key in strata))
        self.assertEqual(sum(1 for value in strata.values() if value == "kink"), 5)


if __name__ == "__main__":
    unittest.main()
