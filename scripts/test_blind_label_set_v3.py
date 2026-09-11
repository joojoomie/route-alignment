#!/usr/bin/env python3
"""Tests for the targeted blind label set v3.

The load-bearing test is `test_pages_carry_no_submitted_partner`. Everything
else in this pipeline is a convenience; that one is the reason the labels are
worth anything. v3 chooses *which* frames to ask about using the submitted
mapping's kinks and its Run A -> Run B partners, so the obvious failure mode is
letting one of those partners reach the page and become an answer key. The test
parses the payload out of each exported page and fails if any query's submitted
partner appears in it, and separately fails if a query carries any field beyond
the three it is allowed.

Run with:
    PYTHONPATH=scripts python3 -m unittest scripts.test_blind_label_set_v3 -v
or:
    PYTHONPATH=scripts python3 scripts/test_blind_label_set_v3.py
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
import re
import unittest

import build_blind_label_set_v3 as builder
import import_blind_labels_v3 as importer
import frame_service


# Re-running the *selection* needs the raw HEVC: it counts frames from the
# bitstream. The submission bundle excludes the recordings by policy, so the
# tests that regenerate a query set must skip there rather than error. The
# tests that read the exported manifest, the page template or the importer
# contract still run, because those artifacts ship.
NEEDS_RAW_VIDEO = (
    "the raw HEVC recordings are absent (excluded from the submission "
    "bundle by policy), so the query selection cannot be regenerated here"
)


def skip_without_raw_video():
    return unittest.skipUnless(frame_service.source_available(), NEEDS_RAW_VIDEO)


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_QUERY_FIELDS = {"queryId", "runAFrame", "linearPriorRunBFrame"}

# The CAM5-only redo. v3's CAM5 labels were taken with the pose compensation
# applied with the wrong sign, so CAM5 is asked again on a fresh query set that
# excludes everything v1, v2 and v3 showed a human.
V3B_SET = "blind_v3b"
V3B_SEED = 20260903


@functools.lru_cache(maxsize=4)
def selection(count: int = builder.QUERIES_PER_CAMERA) -> dict:
    return builder.selection_only(count)


@functools.lru_cache(maxsize=4)
def partners(camera: str) -> dict:
    return builder.submitted_partners(camera)


@functools.lru_cache(maxsize=8)
def exported_manifest(set_name: str = builder.DEFAULT_SET_NAME) -> dict | None:
    path = builder.manifest_path(set_name)
    return json.loads(path.read_text()) if path.is_file() else None


def pages_exist(set_name: str) -> bool:
    """Whether the exported annotation pages are on disk.

    The submission bundle ships each set's manifest, README and sealed labels
    but **not** its `pages/` directory: those are dozens of exported HTML pages
    plus a full Run B filmstrip, and the manifest already records everything
    that was asked. The tests that read a page are therefore a repository-only
    check, and must SKIP from the bundle rather than error - a test that cannot
    run is not a test that failed.
    """

    directory = builder.pages_dir(set_name)
    return directory.is_dir() and any(directory.glob("annotate_*.html"))


def skip_without_pages(set_name: str):
    return unittest.skipUnless(
        pages_exist(set_name),
        f"exported {set_name} pages are not present (they are excluded from the "
        f"submission bundle by policy); rebuild them with "
        f"scripts/build_blind_label_set_v3.py --set-name {set_name} to run this",
    )


def payload_of(html: str) -> dict:
    match = re.search(r"^const DATA = (\{.*\});$", html, re.MULTILINE)
    if match is None:
        raise AssertionError("the exported page has no `const DATA = {...};` payload")
    return json.loads(match.group(1))


@skip_without_raw_video()
class SelectionTests(unittest.TestCase):
    def test_selection_is_deterministic(self) -> None:
        first = builder.selection_only()
        second = builder.selection_only()
        for camera in builder.CAMERAS:
            self.assertEqual(
                [query["runAFrame"] for query in first[camera]["queries"]],
                [query["runAFrame"] for query in second[camera]["queries"]],
                f"{camera}: the seeded selection is not reproducible",
            )
            self.assertEqual(
                [query["stratum"] for query in first[camera]["queries"]],
                [query["stratum"] for query in second[camera]["queries"]],
            )
        self.assertEqual(
            builder.query_set_sha256(first), builder.query_set_sha256(second)
        )

    def test_strata_counts_match_the_plan(self) -> None:
        chosen = selection()
        for camera in builder.CAMERAS:
            counts = chosen[camera]["strata_counts"]
            for stratum, wanted in builder.STRATA_PLAN[camera].items():
                self.assertEqual(
                    counts[stratum], wanted, f"{camera}/{stratum}: expected {wanted}"
                )
            self.assertEqual(sum(counts.values()), builder.QUERIES_PER_CAMERA)

    def test_cam5_has_no_occlusion_stratum(self) -> None:
        # The confirmed occlusion was observed on CAM0's side of the vehicle
        # only, so a CAM5 occlusion stratum would be asking about a phenomenon
        # nobody has evidence of on that camera.
        self.assertEqual(selection()["cam5"]["strata_counts"]["occlusion"], 0)
        self.assertEqual(selection()["cam0"]["strata_counts"]["occlusion"], 3)

    def test_queries_lie_inside_the_route(self) -> None:
        chosen = selection()
        for camera in builder.CAMERAS:
            entry = chosen[camera]
            for query in entry["queries"]:
                self.assertGreaterEqual(query["runAFrame"], entry["runA_route_start"])
                self.assertLess(query["runAFrame"], entry["runA_route_end_exclusive"])

    def test_minimum_distance_from_previously_labelled(self) -> None:
        chosen = selection()
        for camera in builder.CAMERAS:
            blocked: set[int] = set()
            for ordinals in builder.previously_labelled(camera).values():
                blocked |= ordinals
            self.assertTrue(blocked, f"{camera}: no previously labelled ordinals loaded")
            for query in chosen[camera]["queries"]:
                nearest = min(abs(query["runAFrame"] - other) for other in blocked)
                self.assertGreaterEqual(
                    nearest,
                    builder.MINIMUM_DISTANCE_FROM_LABELLED,
                    f"{camera}/{query['queryId']}: A{query['runAFrame']} is only "
                    f"{nearest} frames from an already-labelled ordinal",
                )

    def test_minimum_distance_between_queries(self) -> None:
        chosen = selection()
        for camera in builder.CAMERAS:
            queries = chosen[camera]["queries"]
            for left, right in zip(queries, queries[1:]):
                gap = right["runAFrame"] - left["runAFrame"]
                both_occlusion = (
                    left["stratum"] == "occlusion" and right["stratum"] == "occlusion"
                )
                floor = (
                    builder.MINIMUM_DISTANCE_WITHIN_OCCLUSION
                    if both_occlusion
                    else builder.MINIMUM_DISTANCE_BETWEEN_QUERIES
                )
                self.assertGreaterEqual(
                    gap, floor, f"{camera}: {left['queryId']} and {right['queryId']} are "
                    f"only {gap} frames apart"
                )

    def test_kink_queries_sit_next_to_a_distinct_kink(self) -> None:
        chosen = selection()
        for camera in builder.CAMERAS:
            kink_queries = [
                query for query in chosen[camera]["queries"] if query["stratum"] == "kink"
            ]
            used = [query["kink_runA_frame"] for query in kink_queries]
            self.assertEqual(len(set(used)), len(used), f"{camera}: two queries share a kink")
            for query in kink_queries:
                self.assertLessEqual(
                    abs(query["runAFrame"] - query["kink_runA_frame"]),
                    builder.KINK_QUERY_RADIUS,
                )
                self.assertGreaterEqual(
                    abs(query["kink_increment"]), builder.KINK_INCREMENT_THRESHOLD
                )

    def test_dark_queries_have_a_dark_partner(self) -> None:
        chosen = selection()
        for camera in builder.CAMERAS:
            for query in chosen[camera]["queries"]:
                if query["stratum"] != "dark":
                    continue
                low, high = query["dark_runB_stretch"]
                partner = partners(camera)[query["runAFrame"]]
                self.assertTrue(
                    low <= partner <= high,
                    f"{camera}/{query['queryId']}: partner {partner} is outside the "
                    f"dark stretch {low}-{high} it was sampled from",
                )

    def test_occlusion_queries_sit_inside_the_confirmed_window(self) -> None:
        window = builder.occlusion_windows()["cam0"]
        for query in selection()["cam0"]["queries"]:
            if query["stratum"] != "occlusion":
                continue
            self.assertGreaterEqual(query["runAFrame"], window["runA_first"])
            self.assertLessEqual(query["runAFrame"], window["runA_last"])

    def test_linear_prior_is_arithmetic_on_the_route_bounds(self) -> None:
        chosen = selection()
        for camera in builder.CAMERAS:
            entry = chosen[camera]
            for query in entry["queries"]:
                expected = builder.linear_progress_prior(
                    query["runAFrame"],
                    entry["runA_route_start"],
                    entry["runA_route_end_exclusive"],
                    entry["runB_route_start"],
                    entry["runB_route_end"],
                )
                self.assertEqual(query["linearPriorRunBFrame"], expected)


class PoseDefaultTests(unittest.TestCase):
    """The v3 defect, pinned.

    The diagnostic measures a scene *displacement*: CAM5's scene sits 37.5 px
    left and 11 px up in Run B. The compensation the page applies must be the
    negative of it - Run B moves RIGHT and DOWN. v3 exported the raw measurement
    and doubled the misalignment, which is what the annotator then fought with
    the slider. See outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md.
    """

    def test_cam5_default_is_plus_40_plus_12(self) -> None:
        cam5 = builder.pose_defaults("cam5")
        self.assertEqual(
            (cam5["dxPx"], cam5["dyPx"]),
            (40, 12),
            "CAM5's exported compensation must be +40/+12: the scene in Run B sits "
            "left and high, so Run B must be moved right and down",
        )

    def test_cam0_default_stays_zero(self) -> None:
        cam0 = builder.pose_defaults("cam0")
        self.assertEqual((cam0["dxPx"], cam0["dyPx"]), (0, 0))
        self.assertEqual((cam0["measuredDxPx"], cam0["measuredDyPx"]), (0.0, 0.0))

    def test_the_raw_measurement_is_kept_unnegated(self) -> None:
        cam5 = builder.pose_defaults("cam5")
        self.assertEqual(cam5["measuredDxPx"], -37.5)
        self.assertEqual(cam5["measuredDyPx"], -11.0)
        self.assertEqual(cam5["measuredRaster"], [896, 672])

    def test_compensation_is_the_negated_measurement_scaled_to_the_filmstrip(self) -> None:
        cam5 = builder.pose_defaults("cam5")
        self.assertEqual(cam5["dxPx"], -round(-37.5 * builder.FILMSTRIP_WIDTH / 896))
        self.assertEqual(cam5["dyPx"], -round(-11.0 * builder.FILMSTRIP_HEIGHT / 672))
        for axis, measured in (("dxPx", "measuredDxPx"), ("dyPx", "measuredDyPx")):
            self.assertLess(
                cam5[axis] * cam5[measured],
                0,
                f"{axis} must have the opposite sign to {measured}",
            )

    def test_cam5_is_compensated_by_default_and_cam0_is_not(self) -> None:
        self.assertFalse(builder.pose_defaults("cam0")["compensated"])
        self.assertTrue(builder.pose_defaults("cam5")["compensated"])

    def test_the_convention_is_stated_where_it_is_exported(self) -> None:
        convention = builder.POSE_CONVENTION.lower()
        self.assertIn("compensation = -displacement", convention)
        self.assertIn("positive moves run b right/down", convention)
        self.assertEqual(builder.pose_defaults("cam5")["convention"], builder.POSE_CONVENTION)

    def test_filmstrip_raster_is_not_reduced_below_720_wide(self) -> None:
        self.assertGreaterEqual(builder.FILMSTRIP_WIDTH, 720)


class SliderHardeningTests(unittest.TestCase):
    """The slider must not be able to absorb frame error again.

    On 22 of 23 CAM5 v3 match labels the slider was dragged from the (wrongly
    signed) default out to +10..+200 px, and the frame error correlated with how
    far it was dragged. A 400 px sweep is enough to make almost any frame look
    aligned on one depth; +-25 px is not.
    """

    def test_the_slider_is_clamped_to_the_default_plus_minus_25(self) -> None:
        self.assertEqual(builder.POSE_SLIDER_RANGE_PX, 25)
        self.assertIn("const POSE_RANGE = DATA.pose.sliderRangePx;", builder._PAGE_TEMPLATE)
        self.assertIn("dx: [DATA.pose.dxPx - POSE_RANGE, DATA.pose.dxPx + POSE_RANGE]",
                      builder._PAGE_TEMPLATE)
        self.assertIn("dy: [DATA.pose.dyPx - POSE_RANGE, DATA.pose.dyPx + POSE_RANGE]",
                      builder._PAGE_TEMPLATE)
        self.assertEqual(builder.pose_defaults("cam5")["sliderRangePx"], 25)
        # The bounds are set from the payload, never as literals in the markup.
        self.assertIn("slider.min = low;", builder._PAGE_TEMPLATE)
        self.assertIn("slider.max = high;", builder._PAGE_TEMPLATE)

    def test_every_slider_read_is_clamped(self) -> None:
        for source in (
            'poseDx = clampPose("dx", readStore("poseDx", DATA.pose.dxPx))',
            'poseDy = clampPose("dy", readStore("poseDy", DATA.pose.dyPx))',
            'poseDx = clampPose("dx", event.target.value)',
            'poseDy = clampPose("dy", event.target.value)',
        ):
            self.assertIn(source, builder._PAGE_TEMPLATE)

    def test_the_default_is_drawn_as_a_tick(self) -> None:
        self.assertIn('list="poseDxTicks"', builder._PAGE_TEMPLATE)
        self.assertIn('list="poseDyTicks"', builder._PAGE_TEMPLATE)
        self.assertIn('<datalist id="poseDxTicks"></datalist>', builder._PAGE_TEMPLATE)
        self.assertIn('<datalist id="poseDyTicks"></datalist>', builder._PAGE_TEMPLATE)
        self.assertIn('tick.value = DATA.pose[axis === "dx" ? "dxPx" : "dyPx"];',
                      builder._PAGE_TEMPLATE)

    def test_the_flip_sign_escape_hatch_is_gone(self) -> None:
        # It existed to rescue a wrongly signed default. The default is right
        # now, and a sign flip would leave the slider outside its own clamp.
        self.assertNotIn("poseFlipBtn", builder._PAGE_TEMPLATE)
        self.assertNotIn("Flip sign", builder._PAGE_TEMPLATE)

    def test_the_warning_line_is_present_and_permanent(self) -> None:
        warning = (
            "Do not use the slider to make a frame fit. Change the frame until FAR "
            "and NEAR residuals are equal; only then may a small slider tweak zero both."
        )
        self.assertIn(warning, builder._PAGE_TEMPLATE)
        # Not inside a <details>, not toggled by anything: it is always visible.
        self.assertIn('<div class="warnline" id="sliderWarning">', builder._PAGE_TEMPLATE)
        self.assertNotIn('el("sliderWarning")', builder._PAGE_TEMPLATE)


class GuideBandTests(unittest.TestCase):
    """The two guides must come from two different depths, or nothing is pinned."""

    def test_the_bands_are_the_top_half_and_the_bottom_third(self) -> None:
        self.assertEqual(builder.FAR_GUIDE_BAND, (0.0, 0.5))
        self.assertEqual(builder.NEAR_GUIDE_BAND[1], 1.0)
        self.assertAlmostEqual(builder.NEAR_GUIDE_BAND[0], 2 / 3)

    def test_the_bands_reach_the_page_and_are_drawn(self) -> None:
        self.assertIn("const GUIDE_BANDS = DATA.guideBands;", builder._PAGE_TEMPLATE)
        self.assertIn('<div class="guidebands" id="bandsA">', builder._PAGE_TEMPLATE)
        self.assertIn('<div class="guidebands" id="bandsB">', builder._PAGE_TEMPLATE)
        self.assertIn("function renderBands()", builder._PAGE_TEMPLATE)

    def test_default_guides_start_inside_their_bands(self) -> None:
        far_low, far_high = builder.FAR_GUIDE_BAND
        near_low, near_high = builder.NEAR_GUIDE_BAND
        self.assertTrue(far_low <= 0.30 <= far_high)
        self.assertTrue(near_low <= 0.80 <= near_high)
        self.assertIn("far: { x: 0.35, y: 0.30 }", builder._PAGE_TEMPLATE)
        self.assertIn("near: { x: 0.65, y: 0.80 }", builder._PAGE_TEMPLATE)

    def test_a_label_cannot_be_saved_while_a_guide_is_out_of_band(self) -> None:
        self.assertIn("function guideViolations()", builder._PAGE_TEMPLATE)
        self.assertIn("function guardGuides()", builder._PAGE_TEMPLATE)
        # Every path that writes an answer is gated.
        self.assertIn("function accept(tolerance) {\n  if (!guardGuides()) return;",
                      builder._PAGE_TEMPLATE)
        self.assertIn("function recordNoMatch(reason) {\n  if (!guardGuides()) return;",
                      builder._PAGE_TEMPLATE)
        # ... and the buttons go dead with a tooltip rather than failing silently.
        self.assertIn("button.disabled = blocked.length > 0;", builder._PAGE_TEMPLATE)
        self.assertIn("button.title = reason;", builder._PAGE_TEMPLATE)

    def test_the_guides_carry_a_depth_coordinate(self) -> None:
        for column in ("far_line_y_frac", "near_line_y_frac"):
            self.assertIn(column, builder.CSV_COLUMNS)
        self.assertIn("farLine: { x: round4(far.x), y: round4(far.y) }",
                      builder._PAGE_TEMPLATE)
        self.assertIn("nearLine: { x: round4(near.x), y: round4(near.y) }",
                      builder._PAGE_TEMPLATE)


class CsvSchemaTests(unittest.TestCase):
    def test_page_csv_header_matches_the_importer(self) -> None:
        self.assertEqual(builder.CSV_COLUMNS, importer.REQUIRED_COLUMNS)
        self.assertIn('const CSV_HEADER = "__CSV_HEADER__";', builder._PAGE_TEMPLATE)
        page = builder.build_page("cam5", [], 0, 1, "now", V3B_SET)
        header = re.search(r'const CSV_HEADER = "(.*?)";', page)
        self.assertIsNotNone(header, "the built page no longer declares CSV_HEADER")
        self.assertEqual(tuple(header.group(1).split(",")), importer.REQUIRED_COLUMNS)

    def test_the_slider_value_is_written_into_every_label(self) -> None:
        for column in ("pose_dx_px", "pose_dy_px", "pose_compensated"):
            self.assertIn(column, importer.REQUIRED_COLUMNS)
        # The snapshot is taken when the answer is recorded, so the CSV carries
        # the shift each label was judged under rather than the last one set.
        self.assertIn("poseDx: poseDx,", builder._PAGE_TEMPLATE)
        self.assertIn("poseDy: poseDy,", builder._PAGE_TEMPLATE)
        self.assertIn("answer ? answer.poseDx : poseDx,", builder._PAGE_TEMPLATE)
        self.assertIn("answer ? answer.poseDy : poseDy,", builder._PAGE_TEMPLATE)

    def test_label_vocabulary_is_shared(self) -> None:
        self.assertEqual(
            importer.ACTIVE_LABELS, ("match", "undetermined", "no_correspondence")
        )
        self.assertEqual(importer.MATCH_LABEL, "match")
        self.assertEqual(
            importer.ABSTENTION_LABELS, ("undetermined", "no_correspondence")
        )
        # The page writes "match" literally and takes the other two straight
        # from the radio value, so the radio values are the vocabulary.
        self.assertIn('label: "match"', builder._PAGE_TEMPLATE)
        self.assertIn("Object.assign({ label: reason }", builder._PAGE_TEMPLATE)
        radio_values = set(
            re.findall(r'name="reason" value="([a-z_]+)"', builder._PAGE_TEMPLATE)
        )
        self.assertEqual(radio_values, set(importer.ABSTENTION_LABELS))

    def test_the_two_no_match_reasons_are_separate_values(self) -> None:
        # The v2 defect: one button meaning both "I cannot tell" and "there is
        # no such place". They must be distinct radio values and distinct labels.
        for value in ("undetermined", "no_correspondence"):
            self.assertIn(
                f'name="reason" value="{value}"',
                builder._PAGE_TEMPLATE,
                f"the page has no radio for {value}",
            )
        self.assertIn("noMatchBtn", builder._PAGE_TEMPLATE)
        # The button stays disabled until a reason is chosen (and, since the
        # hardening, until both guides sit in their bands).
        self.assertIn(
            "button.disabled = blocked.length > 0 || selectedReason() === null;",
            builder._PAGE_TEMPLATE,
        )

    def test_tolerance_controls_reach_three(self) -> None:
        for tolerance in (1, 2, 3):
            self.assertIn(f'data-tolerance="{tolerance}"', builder._PAGE_TEMPLATE)
        self.assertIn('el("acceptBtn").addEventListener("click", () => accept(0));',
                      builder._PAGE_TEMPLATE)

    def test_band_restricted_blink_modes_exist(self) -> None:
        for mode in ("blink_far", "blink_near"):
            self.assertIn(f'value="{mode}"', builder._PAGE_TEMPLATE)
        self.assertIn("const FAR_BAND = 0.45;", builder._PAGE_TEMPLATE)
        self.assertIn("const NEAR_BAND = 1 / 3;", builder._PAGE_TEMPLATE)


@skip_without_raw_video()
class ImporterContractTests(unittest.TestCase):
    """The CSV the page writes must be the CSV the importer accepts."""

    def rows(self, camera: str = "cam0") -> list[dict[str, str]]:
        entry = selection()[camera]
        made: list[dict[str, str]] = []
        for position, query in enumerate(entry["queries"]):
            label = ("match", "undetermined", "no_correspondence")[position % 3]
            frame = query["linearPriorRunBFrame"]
            made.append(
                {
                    "query_id": query["queryId"],
                    "camera_id": camera,
                    "runA_frame": str(query["runAFrame"]),
                    "label": label,
                    "runB_frame": str(frame) if label == "match" else "",
                    "runB_min": str(frame - 1) if label == "match" else "",
                    "runB_max": str(frame + 1) if label == "match" else "",
                    "pose_dx_px": "0",
                    "pose_dy_px": "0",
                    "pose_compensated": "false",
                    "far_line_x_frac": "0.35",
                    "far_line_y_frac": "0.30",
                    "near_line_x_frac": "0.65",
                    "near_line_y_frac": "0.80",
                    "compare_mode": "blink_near",
                    "source": "manual_viewer_blind",
                }
            )
        return made

    def validate(self, rows: list[dict[str, str]], camera: str = "cam0"):
        entry = selection()[camera]
        expected = {
            query["queryId"]: query["runAFrame"] for query in entry["queries"]
        }
        return importer.validate_rows(
            rows, camera, expected, entry["runB_route_start"], entry["runB_route_end"]
        )

    def test_a_complete_file_validates(self) -> None:
        validated = self.validate(self.rows())
        self.assertEqual(len(validated), builder.QUERIES_PER_CAMERA)
        labels = {record["label"] for record in validated}
        self.assertEqual(labels, set(importer.ACTIVE_LABELS))

    def test_unlabelled_rows_are_rejected(self) -> None:
        rows = self.rows()
        rows[0]["label"] = "unlabelled"
        with self.assertRaisesRegex(ValueError, "still unlabelled"):
            self.validate(rows)

    def test_the_v2_label_is_no_longer_accepted(self) -> None:
        # v2's single "no_match" is exactly the ambiguity v3 exists to remove,
        # so a file still using it must fail rather than be silently coerced.
        rows = self.rows()
        rows[1]["label"] = "no_match"
        with self.assertRaisesRegex(ValueError, "unknown label"):
            self.validate(rows)

    def test_an_abstention_may_not_name_a_run_b_frame(self) -> None:
        rows = self.rows()
        rows[1]["runB_frame"] = "500"
        with self.assertRaisesRegex(ValueError, "still names a Run B frame"):
            self.validate(rows)

    def test_a_match_outside_the_exported_range_is_rejected(self) -> None:
        rows = self.rows()
        rows[0]["runB_frame"] = "99999"
        rows[0]["runB_max"] = "99999"
        with self.assertRaisesRegex(ValueError, "outside the exported"):
            self.validate(rows)

    def test_model_derived_columns_are_rejected(self) -> None:
        rows = self.rows()
        for row in rows:
            row["model_prediction"] = "123"
        with self.assertRaisesRegex(ValueError, "not prediction-blind"):
            self.validate(rows)

    def test_a_stratum_column_is_rejected(self) -> None:
        rows = self.rows()
        for row in rows:
            row["stratum"] = "kink"
        with self.assertRaisesRegex(ValueError, "not prediction-blind"):
            self.validate(rows)

    def test_a_foreign_run_a_frame_is_rejected(self) -> None:
        rows = self.rows()
        rows[0]["runA_frame"] = str(int(rows[0]["runA_frame"]) + 1)
        with self.assertRaisesRegex(ValueError, "does not match the exported"):
            self.validate(rows)

    def test_a_missing_answer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "never answered"):
            self.validate(self.rows()[:-1])


class ExportedPageTests(unittest.TestCase):
    """Checks against the pages actually written to disk."""

    SET_NAME = builder.DEFAULT_SET_NAME

    def setUp(self) -> None:
        self.manifest = exported_manifest(self.SET_NAME)
        if self.manifest is None:
            self.skipTest(
                f"no exported {self.SET_NAME} set; run "
                f"scripts/build_blind_label_set_v3.py --set-name {self.SET_NAME} first"
            )
        if not pages_exist(self.SET_NAME):
            self.skipTest(
                f"the exported {self.SET_NAME} annotation pages are not present "
                "(excluded from the submission bundle by policy); rebuild with "
                f"scripts/build_blind_label_set_v3.py --set-name {self.SET_NAME}"
            )
        self.cameras = tuple(self.manifest["cameras"])

    def page(self, camera: str) -> str:
        return (builder.pages_dir(self.SET_NAME) / f"annotate_{camera}.html").read_text()

    def test_pages_carry_no_submitted_partner(self) -> None:
        """No query's submitted Run B partner may appear in the page payload.

        Greps the payload as text with word boundaries. A number is only
        forgiven when it is a value the page is *allowed* to contain - a Run A
        query frame or a linear prior - and even then only when it does not
        belong to the query whose partner it is.
        """

        for camera in self.cameras:
            payload_text = re.search(
                r"^const DATA = (\{.*\});$", self.page(camera), re.MULTILINE
            ).group(1)
            payload = json.loads(payload_text)
            allowed: set[int] = {payload["runBMin"], payload["runBMax"]}
            for query in payload["queries"]:
                allowed.add(query["runAFrame"])
                allowed.add(query["linearPriorRunBFrame"])

            mapping = partners(camera)
            for query in payload["queries"]:
                partner = mapping.get(query["runAFrame"])
                if partner is None:
                    continue
                if re.search(rf"(?<![\d.]){partner}(?![\d.])", payload_text):
                    self.assertIn(
                        partner,
                        allowed,
                        f"{camera}/{query['queryId']}: the submitted partner "
                        f"B{partner} appears in the exported payload",
                    )
                    # Forgiven as a coincidence only if it is not this query's
                    # own prior, which would make it an answer key.
                    self.assertNotEqual(
                        partner,
                        query["linearPriorRunBFrame"],
                        f"{camera}/{query['queryId']}: the linear prior equals the "
                        f"submitted partner B{partner}",
                    )

    def test_query_records_carry_only_the_three_allowed_fields(self) -> None:
        for camera in self.cameras:
            payload = payload_of(self.page(camera))
            for query in payload["queries"]:
                self.assertEqual(
                    set(query),
                    ALLOWED_QUERY_FIELDS,
                    f"{camera}/{query['queryId']} carries unexpected fields",
                )

    def test_pages_name_no_stratum_and_no_method(self) -> None:
        forbidden = (
            "stratum", "strata", "kink", "corridor", "too_dark", "confidence",
            "prediction", "predicted", "unified", "frame_mapping", "posterior",
            "dinov", "baseline",
        )
        for camera in self.cameras:
            lowered = self.page(camera).lower()
            for token in forbidden:
                self.assertNotIn(
                    token, lowered, f"annotate_{camera}.html mentions {token!r}"
                )

    def test_pages_offer_no_automatic_best_frame_hint(self) -> None:
        for camera in self.cameras:
            lowered = self.page(camera).lower()
            for token in ("suggest", "best frame", "recommend", "auto-align", "argmin", "argmax"):
                self.assertNotIn(token, lowered)

    def test_payload_priors_match_the_manifest(self) -> None:
        for camera in self.cameras:
            payload = payload_of(self.page(camera))
            entry = self.manifest["cameras"][camera]
            expected = {
                query["queryId"]: query["linearPriorRunBFrame"] for query in entry["queries"]
            }
            self.assertEqual(
                {query["queryId"]: query["linearPriorRunBFrame"] for query in payload["queries"]},
                expected,
            )
            self.assertEqual(payload["runBMin"], entry["runB_route_start"])
            self.assertEqual(payload["runBMax"], entry["runB_route_end"])

    def test_manifest_records_the_stratum_dependency(self) -> None:
        disclosure = self.manifest["stratum_dependency_disclosure"].lower()
        self.assertIn("which frames are asked", disclosure)
        self.assertIn("not affect what the annotator sees", disclosure)
        self.assertFalse(self.manifest["blindness"]["stratum_shown_to_annotator"])
        self.assertFalse(self.manifest["blindness"]["automatic_best_frame_hint"])
        for camera in self.cameras:
            for query in self.manifest["cameras"][camera]["queries"]:
                self.assertIn("stratum", query)

    def test_manifest_is_outside_the_served_root(self) -> None:
        manifest_file = builder.manifest_path(self.SET_NAME)
        self.assertTrue(manifest_file.is_file())
        self.assertNotIn(builder.pages_dir(self.SET_NAME), manifest_file.parents)

    def test_every_referenced_frame_was_exported(self) -> None:
        for camera in self.cameras:
            payload = payload_of(self.page(camera))
            for query in payload["queries"]:
                path = builder.pages_dir(self.SET_NAME) / camera / "runA" / f"{query['runAFrame']:06d}.jpg"
                self.assertTrue(path.is_file(), f"missing {path}")
            run_b = builder.pages_dir(self.SET_NAME) / camera / "runB"
            for frame in (payload["runBMin"], payload["runBMax"]):
                self.assertTrue((run_b / f"{frame:06d}.jpg").is_file())
            self.assertEqual(
                len(list(run_b.glob("*.jpg"))), payload["runBMax"] - payload["runBMin"] + 1
            )

    def test_filmstrip_covers_160_frames_either_side_of_every_prior(self) -> None:
        for camera in self.cameras:
            payload = payload_of(self.page(camera))
            for query in payload["queries"]:
                prior = query["linearPriorRunBFrame"]
                low = max(payload["runBMin"], prior - 160)
                high = min(payload["runBMax"], prior + 160)
                self.assertGreaterEqual(high - low, 160)


class ExportedV3bPageTests(ExportedPageTests):
    """Every exported-page check again, against the CAM5 redo set."""

    SET_NAME = V3B_SET


class V3bSelectionTests(unittest.TestCase):
    """The redo set: CAM5 only, fresh frames, and honest about its strata."""

    def setUp(self) -> None:
        self.manifest = exported_manifest(V3B_SET)
        if self.manifest is None:
            self.skipTest("no exported blind_v3b set")

    def test_it_is_cam5_only(self) -> None:
        self.assertEqual(list(self.manifest["cameras"]), ["cam5"])
        self.assertEqual(self.manifest["cameras_exported"], ["cam5"])
        self.assertEqual(self.manifest["label_set"], V3B_SET)
        self.assertEqual(self.manifest["random_seed"], V3B_SEED)

    @skip_without_raw_video()
    def test_the_query_set_hash_is_reproducible(self) -> None:
        chosen = v3b_selection()
        self.assertEqual(
            builder.query_set_sha256(chosen, V3B_SEED, V3B_SET),
            self.manifest["query_set_sha256"],
            "the exported blind_v3b query set cannot be regenerated from its seed",
        )
        self.assertNotEqual(
            self.manifest["query_set_sha256"],
            exported_manifest()["query_set_sha256"],
            "the redo must not be the same query set as v3",
        )

    def test_no_ordinal_was_ever_shown_to_a_human(self) -> None:
        sources = builder.previously_labelled("cam5", V3B_SET)
        self.assertIn("blind_v3_labels", sources)
        self.assertTrue(sources["blind_v3_labels"], "no v3 CAM5 ordinals loaded")
        blocked: set[int] = set()
        for ordinals in sources.values():
            blocked |= ordinals
        for query in self.manifest["cameras"]["cam5"]["queries"]:
            nearest = min(abs(query["runAFrame"] - other) for other in blocked)
            self.assertGreaterEqual(
                nearest,
                builder.MINIMUM_DISTANCE_FROM_LABELLED,
                f"{query['queryId']}: A{query['runAFrame']} is only {nearest} frames "
                "from an ordinal v1, v2 or v3 already showed a human",
            )

    def test_the_exclusion_names_v1_v2_and_v3(self) -> None:
        listed = self.manifest["sampling_rules"]["previously_labelled_sources"]
        for source in (
            "outputs/task2_keyframes/manual_annotations.csv",
            "outputs/task2_evaluation/blind_v2/blind_v2_labels.csv",
            "outputs/task2_evaluation/blind_v3/blind_v3_labels.csv",
        ):
            self.assertIn(source, listed)

    def test_queries_are_ten_apart_from_each_other(self) -> None:
        queries = self.manifest["cameras"]["cam5"]["queries"]
        self.assertEqual(len(queries), builder.QUERIES_PER_CAMERA)
        for left, right in zip(queries, queries[1:]):
            self.assertGreaterEqual(
                right["runAFrame"] - left["runAFrame"],
                builder.MINIMUM_DISTANCE_BETWEEN_QUERIES,
            )

    def test_unfilled_strata_went_to_corridor_and_were_recorded(self) -> None:
        entry = self.manifest["cameras"]["cam5"]
        counts = entry["strata_counts"]
        self.assertEqual(sum(counts.values()), builder.QUERIES_PER_CAMERA)
        planned = builder.STRATA_PLAN["cam5"]
        moved = 0
        for shortfall in entry["strata_shortfalls"]:
            stratum = shortfall["stratum"]
            self.assertEqual(counts[stratum], shortfall["placed"])
            self.assertEqual(shortfall["planned"], planned[stratum])
            self.assertEqual(
                shortfall["moved_to_corridor"],
                shortfall["planned"] - shortfall["placed"],
            )
            self.assertTrue(shortfall["why"])
            moved += shortfall["moved_to_corridor"]
        self.assertEqual(counts["corridor"], planned["corridor"] + moved)
        for stratum, wanted in planned.items():
            self.assertLessEqual(
                counts[stratum] if stratum != "corridor" else wanted, wanted
            )

    @skip_without_pages(V3B_SET)
    def test_the_page_carries_the_corrected_compensation(self) -> None:
        page = (builder.pages_dir(V3B_SET) / "annotate_cam5.html").read_text()
        payload = payload_of(page)
        self.assertEqual((payload["pose"]["dxPx"], payload["pose"]["dyPx"]), (40, 12))
        self.assertEqual(payload["pose"]["measuredDxPx"], -37.5)
        self.assertEqual(payload["pose"]["sliderRangePx"], 25)
        self.assertEqual(payload["labelSet"], V3B_SET)
        self.assertEqual(payload["guideBands"]["far"], list(builder.FAR_GUIDE_BAND))
        self.assertEqual(payload["guideBands"]["near"], list(builder.NEAR_GUIDE_BAND))
        self.assertIn(
            "Do not use the slider to make a frame fit.",
            page,
            "the exported page has no slider warning",
        )
        self.assertEqual(
            self.manifest["cameras"]["cam5"]["pose_defaults"]["dxPx"], 40
        )

    def test_the_manifest_states_the_sign_convention(self) -> None:
        self.assertEqual(self.manifest["pose_convention"], builder.POSE_CONVENTION)
        self.assertIn(
            "compensation = -displacement",
            self.manifest["annotator_aids"]["pose_compensation"],
        )
        history = self.manifest["annotator_aids"]["pose_compensation_sign_history"]
        self.assertIn("shipped the raw measurement as the compensation", history)
        self.assertIn("TOOL_DEFECT_cam5.md", history)
        self.assertIn(
            f"±{builder.POSE_SLIDER_RANGE_PX} px",
            self.manifest["annotator_aids"]["pose_slider_clamp"].replace("+-", "±"),
        )
        self.assertIn(
            "refuses to save a label",
            self.manifest["annotator_aids"]["guide_band_rule"],
        )

    def test_the_readme_sits_next_to_the_manifest_and_outside_the_served_root(self) -> None:
        readme = builder.output_dir(V3B_SET) / "README.md"
        self.assertTrue(readme.is_file())
        self.assertNotIn(builder.pages_dir(V3B_SET), readme.parents)
        text = readme.read_text()
        self.assertIn(self.manifest["query_set_sha256"], text)
        self.assertIn("compensation = -displacement", text)


@functools.lru_cache(maxsize=2)
def v3b_selection() -> dict:
    return builder.selection_only(
        builder.QUERIES_PER_CAMERA, ("cam5",), V3B_SEED, V3B_SET
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
