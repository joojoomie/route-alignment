#!/usr/bin/env python3
"""Generate the report's result tables from the artifacts.

Every number in the previous report was typed by hand, including tables that
already existed as generated files elsewhere in the repo. That is how a PDF
quietly stops matching the evidence it describes. These fragments are written
from the JSON the pipelines emit and are pulled into the LaTeX with \\input, so
a metric can no longer change in one place and not the other.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "report_tables"

COMPARISON = ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "method_comparison_blind_v2.json"
RESCORE = ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "blind_v2_undetermined_rescore.json"
BITSTREAM = ROOT / "outputs" / "task1" / "task1_bitstream_summary.csv"
MASKS = ROOT / "outputs" / "task2_masks" / "reliability_mask_summary.json"
RESOLUTION = ROOT / "outputs" / "task2_unified" / "resolution_study" / "geometry_resolution_study.json"
CONSISTENCY = ROOT / "outputs" / "task2_consistency" / "consistency_summary.json"
HOLDOUT = ROOT / "outputs" / "task2_abstention" / "abstention_holdout.json"
QUALITY = ROOT / "outputs" / "task3" / "frame_quality_flags.json"
OCCLUSION = ROOT / "outputs" / "task2_abstention" / "confirmed_occlusion_runB_cam0.json"
POSE = ROOT / "outputs" / "task2_pose_offset" / "pose_offset_diagnostic.json"
PARALLAX = ROOT / "outputs" / "task2_unified_variants" / "parallax" / "parallax_manifest.json"
PAIR_EVIDENCE = ROOT / "outputs" / "task2_evaluation" / "visual_audit" / "pair_evidence.json"
VARIANTS = ROOT / "outputs" / "task2_variants" / "variant_comparison.json"
KINKS = ROOT / "outputs" / "task2_kinks" / "mapping_kinks.json"
BLIND_V3 = ROOT / "outputs" / "task2_evaluation" / "blind_v3" / "blind_v3_results.json"
BLIND_V3B = (
    ROOT / "outputs" / "task2_evaluation" / "blind_v3b" / "blind_v3b_results.json"
)
CONVENTION = (
    ROOT
    / "outputs"
    / "task2_evaluation"
    / "blind_v3b"
    / "convention_shift_sensitivity.json"
)
LATERAL = ROOT / "outputs" / "task1_lateral" / "lateral_offset.json"
SLOPE_MANIFEST = (
    ROOT / "outputs" / "task2_unified_variants" / "slope" / "{camera}" / "unified_manifest.json"
)
SPEED = ROOT / "outputs" / "task2_bayes" / "{camera}" / "bayes_manifest_speed.json"
STATIC_REMOVAL = ROOT / "outputs" / "task2_motion_bayes" / "static_removal_study.json"
BAYES_DIAGNOSTICS = ROOT / "outputs" / "task2_bayes" / "bayes_diagnostics.json"
LOOP_POSTERIOR = (
    ROOT / "outputs" / "task2_loop_consistency" / "loop_consistency_bayes.json"
)
POLICY_SUMMARY = ROOT / "outputs" / "task3" / "policy_summary.csv"
MAPPING_CSV = ROOT / "outputs" / "task2_unified" / "{camera}" / "frame_mapping_unified.csv"
QUALITY_FLAGS_CSV = ROOT / "outputs" / "task3" / "frame_quality_flags.csv"
STRICT_GEOMETRY = (
    ROOT / "outputs" / "task2_unified" / "cam0" / "unified_manifest_strict_geometry.json"
)
UNIFIED_MANIFEST = ROOT / "outputs" / "task2_unified" / "{camera}" / "unified_manifest.json"
BOUNDARIES = ROOT / "outputs" / "task2_keyframes" / "route_phase_boundaries.csv"
TOPOLOGY = ROOT / "outputs" / "task1" / "route_topology.json"
SWEEP_RATE = ROOT / "outputs" / "task2_motion_bayes" / "sweep_rate_summary.json"
GEOMETRY_RASTER = (896, 672)
NATIVE_WIDTH = 1440
MOTION_MAPPINGS = (
    ROOT / "outputs" / "task2_motion_bayes" / "motion_variant_mapping_comparison_mean-divide.json"
)

METHOD_LABELS = {
    "v2_unified": r"v2 unified \textbf{(submitted)}",
    "v1_frozen_submitted": "v1 sparse-anchor",
}


def write(name: str, body: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / name).write_text(body)
    print(f"  {name}")


def blind_comparison_table() -> None:
    """Primary table uses the corrected label reading; both are reported."""

    payload = json.loads(RESCORE.read_text())
    lines = []
    for method in ("v2_unified", "v1_frozen_submitted"):
        for camera in ("cam0", "cam5"):
            result = payload["methods"][method][camera]
            corrected = result["corrected"]
            as_labelled = result["as_labelled"]
            interval = corrected["precision_wilson_95"]
            rendered = f"{interval[0]:.2f}--{interval[1]:.2f}" if interval else "n/a"
            error = result["preferred_frame_error"]
            median = f"{error['median']:.1f}" if error["n"] else "n/a"
            within = f"{error['within_two_frames']}/{error['n']}" if error["n"] else "n/a"
            coverage = _percent(corrected["coverage_of_determinable"])
            lines.append(
                f"{METHOD_LABELS[method]} & {camera.upper()} & "
                f"{corrected['accepted']}/{result['labels_determinable']} & "
                f"{corrected['correct']} & "
                f"{corrected['precision']:.3f} & {rendered} & {coverage} & "
                f"{median} & {within} & "
                f"{as_labelled['precision']:.3f} " + r"\\"
            )
    body = "\n".join(
        [
            r"\begin{tabularx}{\textwidth}{@{}llrrrlrrrr@{}}",
            r"\toprule",
            r"Method & Cam & Accepted & Correct & Precision & Wilson 95\% & Coverage & "
            r"Median err & Within 2 & As-labelled \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabularx}",
        ]
    )
    write("blind_comparison.tex", body)


TOLERANCE = ROOT / "outputs" / "task2_evaluation" / "tolerance_curve.json"
TOLERANCE_ROWS = (
    ("blind_v2", "cam0", "submitted", "v2 unified (submitted)"),
    ("blind_v2", "cam5", "submitted", "v2 unified (submitted)"),
    ("blind_v2", "cam0", "v1_sparse_anchor", "v1 sparse-anchor"),
    ("blind_v2", "cam5", "v1_sparse_anchor", "v1 sparse-anchor"),
    ("blind_v3", "cam0", "submitted", "v2 unified, targeted set"),
    ("blind_v3b", "cam5", "submitted", "v2 unified, targeted redo"),
)


def tolerance_table() -> None:
    """Headline table: precision at frame tolerances, strict reading in its own column."""

    payload = json.loads(TOLERANCE.read_text())
    lines = []
    for set_name, camera, method, label in TOLERANCE_ROWS:
        result = payload["sets"][set_name]["cameras"][camera][method]
        within = result["within"]
        two = within["2"]
        interval = two["wilson_95"]
        rendered = f"{interval[0]:.2f}--{interval[1]:.2f}" if interval else "n/a"
        cells = " & ".join(_percent(within[str(t)]["fraction"]) for t in (1, 2, 5, 10))
        lines.append(
            f"{label} & {camera.upper()} & {result['accepted']}/{result['match_labels']} & "
            f"{_percent(result['coverage'])} & {cells} & {rendered} & "
            f"{_percent(within['0']['fraction'])} " + r"\\"
        )
    body = "\n".join(
        [
            r"\begin{tabularx}{\textwidth}{@{}llrrrrrrlr@{}}",
            r"\toprule",
            r"Method & Cam & Accepted & Coverage & $\pm1$ & $\pm2$ & $\pm5$ & $\pm10$ & "
            r"Wilson 95\% ($\pm2$) & strict $\pm0$ \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabularx}",
        ]
    )
    write("tolerance_curve.tex", body)


def tolerance_macros() -> dict[str, str]:
    payload = json.loads(TOLERANCE.read_text())
    def at(set_name, camera, method, tolerance):
        return _percent(payload["sets"][set_name]["cameras"][camera][method]["within"][str(tolerance)]["fraction"])
    def wilson(set_name, camera, method, tolerance):
        low, high = payload["sets"][set_name]["cameras"][camera][method]["within"][str(tolerance)]["wilson_95"]
        return f"{low:.2f}--{high:.2f}"
    return {
        "TolHeadline": str(payload["headline_tolerance_frames"]),
        "TolVtwoCamZeroTwo": at("blind_v2", "cam0", "submitted", 2),
        "TolVtwoCamZeroTwoWilson": wilson("blind_v2", "cam0", "submitted", 2),
        "TolVtwoCamFiveTwo": at("blind_v2", "cam5", "submitted", 2),
        "TolVtwoCamFiveTwoWilson": wilson("blind_v2", "cam5", "submitted", 2),
        "TolVtwoCamZeroFive": at("blind_v2", "cam0", "submitted", 5),
        "TolVtwoCamFiveFive": at("blind_v2", "cam5", "submitted", 5),
        "TolVtwoCamZeroTen": at("blind_v2", "cam0", "submitted", 10),
        "TolVtwoCamFiveTen": at("blind_v2", "cam5", "submitted", 10),
        "TolVoneCamZeroTwo": at("blind_v2", "cam0", "v1_sparse_anchor", 2),
        "TolVthreeCamZeroTwo": at("blind_v3", "cam0", "submitted", 2),
        "TolVthreebCamFiveTwo": at("blind_v3b", "cam5", "submitted", 2),
        "TolVthreebCamFiveTwoWilson": wilson("blind_v3b", "cam5", "submitted", 2),
        "TolVthreebCamFiveFive": at("blind_v3b", "cam5", "submitted", 5),
        "TolVthreebVariantCamFiveTwo": at("blind_v3b", "cam5", "slope_variant", 2),
        "TolVthreebPosteriorCamFiveTwo": at("blind_v3b", "cam5", "posterior", 2),
    }


def resolution_table() -> None:
    payload = json.loads(RESOLUTION.read_text())
    by_resolution: dict[str, dict[str, dict]] = {}
    for record in payload["records"]:
        key = f"{record['width']}$\\times${record['height']}"
        by_resolution.setdefault(key, {})[record["camera_id"]] = record
    lines = []
    for key, cameras in by_resolution.items():
        cam0, cam5 = cameras.get("cam0"), cameras.get("cam5")
        lines.append(
            f"{key} & {cam0['median_matches']} & {cam0['passed_distributed_support']}/{cam0['pairs']} & "
            f"{cam5['median_matches']} & {cam5['passed_distributed_support']}/{cam5['pairs']} & "
            f"{cam5['mean_inlier_ratio']:.2f} " + r"\\"
        )
    body = "\n".join(
        [
            r"\begin{tabularx}{\textwidth}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Geometry raster & CAM0 matches & CAM0 passed & CAM5 matches & CAM5 passed & "
            r"CAM5 inlier ratio \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabularx}",
        ]
    )
    write("resolution_study.tex", body)


def _raster(value: str) -> str:
    return value.replace("x", r"$\times$")


def _percent(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}" + r"\%"


def bitstream_table() -> None:
    with BITSTREAM.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lines = []
    for row in rows:
        coded = _raster(row["coded_raster"])
        display = _raster(row["display_raster"])
        reordered = _percent(float(row["reordered_picture_fraction"]))
        pictures = f"{int(row['bitstream_picture_count']):,}"
        size = f"{int(row['file_size_bytes']):,}"
        lines.append(
            f"{row['run'][-1]} & {row['camera'].upper()} & {pictures} & {size} & "
            f"{coded} & {display} & {reordered} " + r"\\"
        )
    body = "\n".join(
        [
            r"\begin{tabularx}{\textwidth}{@{}llrrllr@{}}",
            r"\toprule",
            r"Run & Camera & Pictures & File size (B) & Coded & Display & Reordered \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabularx}",
        ]
    )
    write("bitstream_inventory.tex", body)


def mask_table() -> None:
    payload = json.loads(MASKS.read_text())
    order = ["runA_cam0", "runB_cam0", "runA_cam5", "runB_cam5"]
    lines = []
    for key in order:
        stream = payload["streams"][key]
        run, camera = key.split("_")
        attached = _percent(stream["camera_attached_fraction"], 2)
        dark = _percent(stream["never_bright_fraction"], 2)
        excluded = _percent(stream["exclusion_fraction"], 2)
        lines.append(
            f"{run[-1]} & {camera.upper()} & {attached} & {dark} & {excluded} " + r"\\"
        )
    body = "\n".join(
        [
            r"\begin{tabularx}{\textwidth}{@{}llrrr@{}}",
            r"\toprule",
            r"Run & Camera & Camera-attached & Never bright & Excluded \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabularx}",
        ]
    )
    write("reliability_mask.tex", body)


def _thousands(value: int) -> str:
    """LaTeX thin-space thousands separator, matching the rest of the report."""

    return f"{value:,}".replace(",", "{,}")


def _listed(values: list[int]) -> str:
    """Render a short list as prose: "3, 8 and 16"."""

    rendered = [str(value) for value in values]
    if len(rendered) < 2:
        return "".join(rendered)
    return ", ".join(rendered[:-1]) + " and " + rendered[-1]


def _signed(value: float, places: int = 1) -> str:
    """A signed number, so LaTeX sets a real minus rather than a hyphen.

    `\ensuremath` rather than `$...$` because these macros are used both inside
    and outside maths in the report, and nested dollars are a compile error.
    """

    return rf"\ensuremath{{{value:.{places}f}}}"


def _interval(camera: dict) -> str:
    """The parallax study's implied position offset, as estimate +/- one s.e."""

    offset = camera["implied_position_offset_frames"]
    return (
        rf"\ensuremath{{{offset['estimate']:.1f}\pm{offset['standard_error']:.1f}}}"
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def motion_macros() -> dict[str, str]:
    """The suspension channel, read from the study that corrected it.

    Every figure the report gives for this channel is a comparison between two
    constructions of the same estimator, and getting one of the two wrong is
    exactly the failure that made the correction necessary. So both sides are
    read from the artifact rather than typed.
    """

    study = json.loads(STATIC_REMOVAL.read_text())["modes"]
    old, new = study["none"], study["mean-divide"]
    median = study["median-subtract"]
    changes = json.loads(MOTION_MAPPINGS.read_text())["prediction_changes"]

    def cofire(mode: dict, run: str) -> dict:
        return mode["cross_camera_cofiring"][run]["at_zero_lag"]

    def events(mode: dict, run: str, camera: str) -> int:
        return mode["streams"][f"{run}_{camera}"]["event_count"]

    old_b, new_a, new_b = cofire(old, "runB"), cofire(new, "runA"), cofire(new, "runB")
    # The cross-run ratio is only measurable on cam5; cam0 has too few pairs.
    corrected = new["cross_run_likelihood_ratio"]["cam5"]
    identical = [change["identical_fraction"] for change in changes.values()]

    return {
        "MotionCofireOldRunB": f"{old_b['pooled_hits']} of {old_b['pooled_events']}",
        "MotionCofireRunB": f"{new_b['pooled_hits']} of {new_b['pooled_events']}",
        "MotionCofireRunBRatio": f"{new_b['pooled_ratio']:.2f}",
        "MotionCofireRunA": f"{new_a['pooled_hits']} of {new_a['pooled_events']}",
        "MotionCofireRunARatio": f"{new_a['pooled_ratio']:.2f}",
        "MotionEventsOld": f"{events(old, 'runB', 'cam0')} against {events(old, 'runA', 'cam0')}",
        "MotionEventsNew": f"{events(new, 'runB', 'cam0')} against {events(new, 'runA', 'cam0')}",
        "MotionWithdrawnRatio": (
            f"{old['cross_run_likelihood_ratio']['cam5']['likelihood_ratio']:.2f}"
        ),
        "MotionCorrectedPairs": (
            f"{corrected['of_those_matched_in_run_b']} of "
            f"{corrected['pairs_with_a_run_a_event']}"
        ),
        # Both cameras: quoting only the worse one reads as a bound the reader
        # cannot check, and quoting only the better one is selective.
        "MotionMappingIdentical": f"{min(identical) * 100:.1f}",
        "MotionMappingIdenticalCamZero": f"{changes['cam0']['identical_fraction'] * 100:.1f}",
        "MotionMappingIdenticalCamFive": f"{changes['cam5']['identical_fraction'] * 100:.1f}",
        # The third construction, which the report used to leave out entirely.
        "MotionCofireMedianRunB": (
            f"{cofire(median, 'runB')['pooled_hits']} of "
            f"{cofire(median, 'runB')['pooled_events']}"
        ),
        "MotionCofireMedianRunA": (
            f"{cofire(median, 'runA')['pooled_hits']} of "
            f"{cofire(median, 'runA')['pooled_events']}"
        ),
        "MotionEventsMedianRunB": str(
            median["streams"]["runB_cam0"]["event_count"]
            + median["streams"]["runB_cam5"]["event_count"]
        ),
    }


def evidence_macros() -> dict[str, str]:
    """Macros for the findings added after the blind evaluation was spent.

    Each one is read from the artifact that produced it, for the same reason
    the tables are: a number that lives only in the prose is a number that can
    drift away from the file it claims to describe.
    """

    quality = json.loads(QUALITY.read_text())
    occlusion = json.loads(OCCLUSION.read_text())
    pose = json.loads(POSE.read_text())["cameras"]
    parallax = json.loads(PARALLAX.read_text())["cameras"]
    evidence = json.loads(PAIR_EVIDENCE.read_text())
    variants = json.loads(VARIANTS.read_text())["variants"]

    run_b_cam0 = quality["streams"]["runB_cam0"]
    run_b_cam5 = quality["streams"]["runB_cam5"]
    mapping = quality["submitted_mapping"]
    labels = quality["blind_labels"]

    cam5_pose = pose["cam5"]
    conversion = cam5_pose["label_convention_conversion"]

    # Only the descriptor-side variants: the posterior rows are a different
    # model, reported separately, and would flatter this comparison.
    # Appearance-side variants only. The posterior rows are a different model,
    # and the slope refinement is a fix to the search rather than to the
    # descriptor; both are reported on their own terms instead.
    percentiles = [
        variant["cross_camera"]["p95"]
        for name, variant in variants.items()
        if "posterior" not in name and name not in ("baseline (submitted)", "unified slope", "unified vlad")
    ]

    speed_diagonal = {
        camera: json.loads(Path(str(SPEED).format(camera=camera)).read_text())[
            "speed_channel"
        ]["diagonal_mean"]
        for camera in ("cam0", "cam5")
    }

    return {
        # Exposure.
        "DarkRunBCamZero": f"{run_b_cam0['too_dark']} of {_thousands(run_b_cam0['frames'])}",
        "DarkRunBCamFive": f"{run_b_cam5['too_dark']} of {_thousands(run_b_cam5['frames'])}",
        "DarkLongestStretch": str(run_b_cam0["longest_dark_stretch"]),
        "DarkAcceptedCamZero": (
            f"{mapping['cam0']['either_side_too_dark']} of "
            f"{_thousands(mapping['cam0']['accepted_rows'])}"
        ),
        "DarkAcceptedCamFive": (
            f"{mapping['cam5']['either_side_too_dark']} of "
            f"{_thousands(mapping['cam5']['accepted_rows'])}"
        ),
        "DarkLabelMissesCamFive": _listed(labels["cam5"]["miss_distances_frames"]),
        # The confirmed occlusion.
        "OcclusionInterval": (
            f"B{occlusion['interval']['start']}--{occlusion['interval']['end']}"
        ),
        "OcclusionRows": str(occlusion["unified_mapping"]["rows_mapped_into_interval"]),
        "OcclusionNullMax": (
            f"{occlusion['posterior_mapping']['posterior_no_correspondence']['max']:.5f}"
        ),
        # CAM5 pose and the two conventions of "same place".
        "CamFiveMatcherDx": _signed(cam5_pose["matcher_convention"]["dx_px"]["median"]),
        "CamFiveMatcherDy": _signed(cam5_pose["matcher_convention"]["dy_px"]["median"], 0),
        "CamFiveLabelDx": _signed(cam5_pose["label_convention"]["dx_px"]["median"], 0),
        "CamFiveConventionFrames": f"{abs(conversion['gap_frames_at_median_sweep']):.1f}",
        "CamFiveConvertedPrecision": (
            f"{conversion['rescoring_as_labelled_in_sample_constant']['precision']:.3f}"
        ),
        "CamFiveWidenedPrecision": (
            f"{cam5_pose['rescoring']['widened_to_both_conventions']['precision']:.3f}"
        ),
        "ParallaxCamZeroOffset": _interval(parallax["cam0"]),
        **resolution_macros(),
        "ParallaxCamFiveOffset": _interval(parallax["cam5"]),
        # Label-free cross-run evidence per camera.
        "EvidenceZCamZero": f"{_median([row[3] for row in evidence['cam0']]):.1f}",
        "EvidenceZCamFive": f"{_median([row[3] for row in evidence['cam5']]):.1f}",
        # The variant study.
        "VariantPninetyfiveRange": f"{min(percentiles):.0f}--{max(percentiles):.0f}",
        **vlad_macros(),
        **tolerance_macros(),
        "SpeedDiagonalCamZero": f"{speed_diagonal['cam0']:.2f}",
        "SpeedDiagonalCamFive": f"{speed_diagonal['cam5']:.2f}",
        **motion_macros(),
        **kink_macros(),
        **blind_v3_macros(),
        **blind_v3b_macros(),
        **convention_macros(),
        **lateral_macros(),
        **sweep_rate_macros(),
        **topology_macros(),
        **task3_macros(),
    }


def resolution_macros() -> dict[str, str]:
    """Inlier ratios at the SELECTED raster and at native, both cameras.

    The report used to compare native against the *smallest* raster tried
    (448x336, CAM5 0.71), which makes the collapse look twice as large as it is
    against the raster actually chosen. Both ends now come from the study.
    """

    records = json.loads(RESOLUTION.read_text())["records"]
    indexed = {(row["camera_id"], row["width"]): row for row in records}
    macros: dict[str, str] = {}
    for camera, name in (("cam0", "CamZero"), ("cam5", "CamFive")):
        for width, label in ((896, "Selected"), (1440, "Native")):
            row = indexed.get((camera, width))
            if row:
                macros[f"Inlier{label}{name}"] = f"{row['mean_inlier_ratio']:.2f}"
    return macros


def vlad_macros() -> dict[str, str]:
    """The unsupervised VLAD aggregation variant, the one descriptor-side change that moved the label-free signals."""

    variants = json.loads(VARIANTS.read_text())["variants"]
    base = variants["baseline (submitted)"]
    vlad = variants["unified vlad"]
    kinks_path = ROOT / "outputs" / "task2_kinks_variants" / "mapping_kinks.json"
    kinks = json.loads(kinks_path.read_text())["mappings"] if kinks_path.is_file() else {}

    def kink_count(name: str, camera: str) -> str:
        entry = kinks.get(name, {}).get("per_camera", {}).get(camera, {}).get("kink_counts", {}).get("K>=6")
        if isinstance(entry, dict):
            entry = entry.get("kinks")
        return "n/a" if entry is None else str(entry)

    return {
        "VladPninetyfive": f"{vlad['cross_camera']['p95']:.0f}",
        "VladFrames": f"{vlad['cross_camera']['frames']:,}",
        "BaselineFrames": f"{base['cross_camera']['frames']:,}",
        "VladCoverageCamFive": f"{vlad['cameras']['cam5']['coverage']:.3f}",
        "BaselineCoverageCamFive": f"{base['cameras']['cam5']['coverage']:.3f}",
        "VladSpentCamZero": f"{vlad['cameras']['cam0']['spent_labels_diagnostic']['precision']:.3f}",
        "VladSpentCamFive": f"{vlad['cameras']['cam5']['spent_labels_diagnostic']['precision']:.3f}",
        "VladWithinTwoCamZero": f"{vlad['cameras']['cam0']['spent_labels_diagnostic']['within_2']:.2f}",
        "VladClosureCamZero": vlad["cameras"]["cam0"]["endpoint_closure"],
        "VladClosureCamFive": vlad["cameras"]["cam5"]["endpoint_closure"],
        "VladKinkCamZero": kink_count("variant:vlad", "cam0"),
        "VladKinkCamFive": kink_count("variant:vlad", "cam5"),
    }


def kink_macros() -> dict[str, str]:
    """The implausible-jump audit, its mechanism, and the refinement variant.

    Read from `outputs/task2_kinks/mapping_kinks.json` and, for the variant,
    from the label-free variant comparison, so the paragraph in the report and
    the artifacts it cites cannot drift apart.
    """

    payload = json.loads(KINKS.read_text())
    threshold = payload["physical_bound"]["headline_threshold"]
    key = f"K>={threshold}"

    def counts(mapping: str, camera: str) -> dict:
        return payload["mappings"][mapping]["per_camera"][camera]["kink_counts"][key]

    low_key = f"K>={min(payload['physical_bound']['thresholds_reported'])}"

    def low(mapping: str, camera: str) -> dict:
        return payload["mappings"][mapping]["per_camera"][camera]["kink_counts"][low_key]

    cross = payload["label_cross_check"]["per_camera"]["cam0"]
    near, away = cross["near"], cross["away"]
    policy = payload["slope_gate_policy"]["per_camera"]
    mechanism = payload["mechanism"]
    replay = mechanism.get("structure_replay", {})
    variants = json.loads(VARIANTS.read_text())["variants"]
    slope = variants.get("unified slope", {})
    baseline = variants["baseline (submitted)"]

    macros = {
        "KinkThreshold": str(threshold),
        "KinkCamZero": str(counts("submitted_unified", "cam0")["kinks"]),
        "KinkCamFive": str(counts("submitted_unified", "cam5")["kinks"]),
        "KinkDarkCamZero": str(counts("submitted_unified", "cam0")["dark"]),
        "KinkDarkCamFive": str(counts("submitted_unified", "cam5")["dark"]),
        "KinkPosteriorCamZero": str(counts("bayes_posterior", "cam0")["kinks"]),
        "KinkPosteriorCamFive": str(counts("bayes_posterior", "cam5")["kinks"]),
        "KinkMissNear": f"{near['miss_rate']:.2f}",
        "KinkMissAway": f"{away['miss_rate']:.2f}",
        "KinkMissRatio": f"{near['miss_rate'] / away['miss_rate']:.1f}",
        "KinkNearLabels": (
            f"{near['misses_interval_distance_gt_0']} of {near['evaluable_match_labels']}"
        ),
        "KinkAwayLabels": (
            f"{away['misses_interval_distance_gt_0']} of {away['evaluable_match_labels']}"
        ),
        "KinkCoarseMaxStep": str(mechanism["coarse_path"]["max_one_step_increment"]),
        "KinkRefinedMaxStep": str(mechanism["submitted_mapping"]["max_one_step_increment"]),
        "KinkPinnedFrames": str(
            mechanism["route_wide_residual"]["frames_at_plus_refine_radius"]
        ),
        "KinkPinnedFramesMinus": str(
            mechanism["route_wide_residual"]["frames_at_minus_refine_radius"]
        ),
        "KinkRefineRadius": str(
            abs(int(mechanism["route_wide_residual"]["max"]))
        ),
        # All four thresholds were computed; quoting only the strictest is the
        # flattering half of the audit's own output.
        "KinkThresholds": "/".join(
            str(value) for value in payload["physical_bound"]["thresholds_reported"]
        ),
        "KinkLowThreshold": str(min(payload["physical_bound"]["thresholds_reported"])),
        "KinkLowCamZero": str(low("submitted_unified", "cam0")["kinks"]),
        "KinkLowCamFive": str(low("submitted_unified", "cam5")["kinks"]),
        "KinkLowPosteriorCamZero": str(low("bayes_posterior", "cam0")["kinks"]),
        "KinkLowPosteriorCamFive": str(low("bayes_posterior", "cam5")["kinks"]),
        "KinkFreeArgmaxBehind": (
            f"{replay.get('frames_where_the_free_argmax_is_behind_the_submitted_value')} of "
            f"{replay.get('frames_replayed')}"
        ),
        "KinkFreeArgmaxMedian": (
            f"{replay['median_frames_the_free_argmax_sits_behind']:.0f}"
            if replay.get("median_frames_the_free_argmax_sits_behind") is not None
            else "n/a"
        ),
        "SlopeGateDemotedCamZero": (
            f"{policy['cam0']['demoted_rows']} of "
            f"{_thousands(policy['cam0']['top_tier_rows'])}"
        ),
        "SlopeGateDemotedCamFive": (
            f"{policy['cam5']['demoted_rows']} of "
            f"{_thousands(policy['cam5']['top_tier_rows'])}"
        ),
        "SlopeGatePrecisionCamZero": (
            f"{policy['cam0']['spent_labels_before_gate']['strict_accepted_precision']:.3f} to "
            f"{policy['cam0']['spent_labels_after_gate']['strict_accepted_precision']:.3f}"
        ),
        "SlopeGateCoverageCamZero": (
            f"{policy['cam0']['spent_labels_before_gate']['coverage_all_labels']:.2f} to "
            f"{policy['cam0']['spent_labels_after_gate']['coverage_all_labels']:.2f}"
        ),
        "SlopeGatePrecisionCamFive": (
            f"{policy['cam5']['spent_labels_before_gate']['strict_accepted_precision']:.3f} to "
            f"{policy['cam5']['spent_labels_after_gate']['strict_accepted_precision']:.3f}"
        ),
        "BaselinePninetyfive": f"{baseline['cross_camera']['p95']:.0f}",
    }

    window = mechanism.get("other_mappings_in_window", {}).get("variant:slope", {})
    if window.get("available"):
        macros.update(
            {
                "SlopeWindowScore": f"{window['median_structure_score']:.2f}",
                "SubmittedWindowScore": f"{window['median_structure_score_of_submitted']:.2f}",
                "SlopeWindowBetter": (
                    f"{window['frames_scoring_at_least_the_submitted_pick']} of "
                    f"{window['frames_scored']}"
                ),
                "SlopeWindowFreeArgmax": str(window["frames_equal_to_the_free_argmax"]),
            }
        )

    if "variant:slope" in payload["mappings"]:
        macros["SlopeKinkCamZero"] = str(counts("variant:slope", "cam0")["kinks"])
        macros["SlopeKinkCamFive"] = str(counts("variant:slope", "cam5")["kinks"])
    if slope:
        macros.update(
            {
                "SlopePninetyfive": f"{slope['cross_camera']['p95']:.0f}",
                "SlopeMedian": f"{slope['cross_camera']['median']:.0f}",
                "SlopeCoverageCamZero": _percent(slope["cameras"]["cam0"]["coverage"]),
                "SlopeCoverageCamFive": _percent(slope["cameras"]["cam5"]["coverage"]),
                "SlopeClosureCamZero": slope["cameras"]["cam0"]["endpoint_closure"],
                "SlopeClosureCamFive": slope["cameras"]["cam5"]["endpoint_closure"],
                "BaselineClosureCamZero": baseline["cameras"]["cam0"]["endpoint_closure"],
                "BaselineClosureCamFive": baseline["cameras"]["cam5"]["endpoint_closure"],
                "SlopeSpentCamZero": (
                    f"{slope['cameras']['cam0']['spent_labels_diagnostic']['precision']:.3f}"
                ),
                "SlopeSpentCamFive": (
                    f"{slope['cameras']['cam5']['spent_labels_diagnostic']['precision']:.3f}"
                ),
                "SubmittedSpentCamZero": (
                    f"{baseline['cameras']['cam0']['spent_labels_diagnostic']['precision']:.3f}"
                ),
                "SubmittedSpentCamFive": (
                    f"{baseline['cameras']['cam5']['spent_labels_diagnostic']['precision']:.3f}"
                ),
            }
        )
    return macros


def blind_v3_macros() -> dict[str, str]:
    """The held-out v3 numbers, straight from the single scoring pass.

    Every figure the report states about v3 is read from
    `blind_v3_results.json`, including the ones that go the wrong way for the
    shipped method, so the paragraph cannot drift from the evaluation.
    """

    if not BLIND_V3.is_file():
        return {}
    payload = json.loads(BLIND_V3.read_text())
    methods = payload["methods"]

    def interval(result: dict) -> str:
        bounds = result["precision_wilson_95"]
        return f"{bounds[0]:.2f}--{bounds[1]:.2f}" if bounds else "n/a"

    def fraction(result: dict) -> str:
        return f"{result['correct']}/{result['accepted']}"

    submitted = methods["submitted_unified"]
    variant = methods["slope_refinement_variant"]
    posterior = methods["posterior_model"]
    gated = methods["submitted_plus_slope_gate"]
    dark = methods["submitted_plus_dark_gate"]
    claim = payload["kink_claim_test"]["per_method"]["submitted_unified"]
    absent = submitted["pooled"]["no_correspondence"]

    macros = {
        "VthreeLabels": str(payload["labels"]),
        "VthreeCamZeroPrecision": f"{submitted['cameras']['cam0']['precision']:.3f}",
        "VthreeCamZeroWilson": interval(submitted["cameras"]["cam0"]),
        "VthreeCamFivePrecision": f"{submitted['cameras']['cam5']['precision']:.3f}",
        "VthreeCamFiveWilson": interval(submitted["cameras"]["cam5"]),
        "VthreeCamZeroWithinTwo": (
            f"{submitted['cameras']['cam0']['within_two']}/"
            f"{submitted['cameras']['cam0']['accepted']}"
        ),
        "VthreeCamFiveWithinTwo": (
            f"{submitted['cameras']['cam5']['within_two']}/"
            f"{submitted['cameras']['cam5']['accepted']}"
        ),
        "VthreeCorridorCamZero": (
            f"{submitted['strata']['cam0']['corridor']['precision']:.3f}"
        ),
        "VthreeCorridorCamZeroCount": fraction(submitted["strata"]["cam0"]["corridor"]),
        "VthreeCorridorCamZeroWilson": interval(submitted["strata"]["cam0"]["corridor"]),
        "VthreeCorridorCamFive": (
            f"{submitted['strata']['cam5']['corridor']['precision']:.3f}"
        ),
        "VthreeCorridorCamFiveCount": fraction(submitted["strata"]["cam5"]["corridor"]),
        "VthreeKinkCamZero": f"{submitted['strata']['cam0']['kink']['precision']:.3f}",
        "VthreeKinkCamZeroCount": fraction(submitted["strata"]["cam0"]["kink"]),
        "VthreeKinkCamFiveCount": fraction(submitted["strata"]["cam5"]["kink"]),
        "VthreeKinkMissCamZero": f"{claim['cam0']['kink']['miss_rate']:.2f}",
        "VthreeCorridorMissCamZero": f"{claim['cam0']['corridor']['miss_rate']:.2f}",
        "VthreeKinkMissCamFive": f"{claim['cam5']['kink']['miss_rate']:.2f}",
        "VthreeCorridorMissCamFive": f"{claim['cam5']['corridor']['miss_rate']:.2f}",
        "VthreeDarkCamZeroCount": fraction(submitted["strata"]["cam0"]["dark"]),
        "VthreeDarkCamFiveCount": fraction(submitted["strata"]["cam5"]["dark"]),
        "VthreeSlopeGateCamZero": (
            f"{submitted['cameras']['cam0']['precision']:.3f}$\\to$"
            f"{gated['cameras']['cam0']['precision']:.3f}"
        ),
        "VthreeSlopeGateCamZeroCoverage": (
            f"{submitted['cameras']['cam0']['coverage_of_match_labels']:.2f}$\\to$"
            f"{gated['cameras']['cam0']['coverage_of_match_labels']:.2f}"
        ),
        "VthreeSlopeGateCamFive": (
            f"{submitted['cameras']['cam5']['precision']:.3f}$\\to$"
            f"{gated['cameras']['cam5']['precision']:.3f}"
        ),
        "VthreeSlopeGateCamFiveCoverage": (
            f"{submitted['cameras']['cam5']['coverage_of_match_labels']:.2f}$\\to$"
            f"{gated['cameras']['cam5']['coverage_of_match_labels']:.2f}"
        ),
        "VthreeDarkGateCamZero": f"{dark['cameras']['cam0']['precision']:.3f}",
        "VthreeDarkGateCamFive": f"{dark['cameras']['cam5']['precision']:.3f}",
        "VthreeVariantCamZero": f"{variant['cameras']['cam0']['precision']:.3f}",
        "VthreeVariantCamFive": f"{variant['cameras']['cam5']['precision']:.3f}",
        "VthreePosteriorCamZero": f"{posterior['cameras']['cam0']['precision']:.3f}",
        "VthreePosteriorCamFive": f"{posterior['cameras']['cam5']['precision']:.3f}",
        "VthreeNoCorrespondenceLabels": str(absent["labels"]),
        "VthreeNoCorrespondenceFalseAccepts": str(absent["false_accepts"]),
        "VthreeUndetermined": str(submitted["pooled"]["undetermined"]["labels"]),
    }
    return macros


def blind_v3b_macros() -> dict[str, str]:
    """The CAM5 redo's held-out numbers, straight from its single scoring pass.

    The v3 CAM5 figures are withdrawn: that half was annotated with a viewer
    that applied the pose compensation with the wrong sign, so its precision is
    a property of the tool. Every CAM5 number the report now states comes from
    `blind_v3b_results.json` instead, and the report guards these with
    \\providecommand so a checkout without the redo still compiles.
    """

    if not BLIND_V3B.is_file():
        return {}
    payload = json.loads(BLIND_V3B.read_text())
    methods = payload["methods"]

    def interval(result: dict) -> str:
        bounds = result["precision_wilson_95"]
        return f"{bounds[0]:.2f}--{bounds[1]:.2f}" if bounds else "n/a"

    def fraction(result: dict) -> str:
        return f"{result['correct']}/{result['accepted']}"

    submitted = methods["submitted_unified"]["cameras"]["cam5"]
    strata = methods["submitted_unified"]["strata"]["cam5"]
    variant = methods["slope_refinement_variant"]["cameras"]["cam5"]
    posterior = methods["posterior_model"]["cameras"]["cam5"]
    gated = methods["submitted_plus_slope_gate"]["cameras"]["cam5"]
    dark = methods["submitted_plus_dark_gate"]["cameras"]["cam5"]
    claim = payload["kink_claim_test"]["per_method"]["submitted_unified"]["cam5"]
    absent = methods["submitted_unified"]["cameras"]["cam5"]["no_correspondence"]
    signed = methods["submitted_unified"]["signed_error"]["cam5"]["signed_interval_error"]
    reference = payload["signed_error_check"]["v3_cam5_reference"]
    rejecting = sum(
        1
        for block in methods.values()
        if block["cameras"]["cam5"]["no_correspondence"]["true_rejects"]
    )

    return {
        "VthreebLabels": str(payload["labels"]),
        "VthreebCamFivePrecision": f"{submitted['precision']:.3f}",
        "VthreebCamFiveWilson": interval(submitted),
        "VthreebCamFiveCount": fraction(submitted),
        "VthreebCamFiveWithinTwo": f"{submitted['within_two']}/{submitted['accepted']}",
        "VthreebCamFiveCoverage": _percent(submitted["coverage_of_match_labels"]),
        "VthreebCorridorCamFive": f"{strata['corridor']['precision']:.3f}",
        "VthreebCorridorCamFiveCount": fraction(strata["corridor"]),
        "VthreebCorridorCamFiveWilson": interval(strata["corridor"]),
        "VthreebCorridorCamFiveWithinTwo": (
            f"{strata['corridor']['within_two']}/{strata['corridor']['accepted']}"
        ),
        "VthreebKinkCamFive": f"{strata['kink']['precision']:.3f}",
        "VthreebKinkCamFiveCount": fraction(strata["kink"]),
        "VthreebDarkCamFiveCount": fraction(strata["dark"]),
        "VthreebKinkMissCamFive": f"{claim['kink']['miss_rate']:.2f}",
        "VthreebCorridorMissCamFive": f"{claim['corridor']['miss_rate']:.2f}",
        "VthreebSlopeGateCamFive": (
            f"{submitted['precision']:.3f}$\\to${gated['precision']:.3f}"
        ),
        "VthreebSlopeGateCamFiveCoverage": (
            f"{submitted['coverage_of_match_labels']:.2f}$\\to$"
            f"{gated['coverage_of_match_labels']:.2f}"
        ),
        "VthreebDarkGateCamFive": f"{dark['precision']:.3f}",
        "VthreebVariantCamFive": f"{variant['precision']:.3f}",
        "VthreebPosteriorCamFive": f"{posterior['precision']:.3f}",
        "VthreebNoCorrespondenceLabels": str(absent["labels"]),
        "VthreebNoCorrespondenceRejectingMethods": f"{rejecting} of {len(methods)}",
        "VthreebSignedMedian": f"{signed['median']:+.0f}",
        "VthreebSignedLater": str(signed["later_than_interval"]),
        "VthreebSignedEarlier": str(signed["earlier_than_interval"]),
        "VthreebSignedInside": str(signed["inside_interval"]),
        "VthreeCamFiveSignedMedian": (
            f"{reference['signed_interval_error_median']:+.0f}"
            if reference.get("available")
            else "n/a"
        ),
        "VthreebPoseDefault": (
            f"{payload['pose_convention_check']['cameras']['cam5']['seeded_default'][0]}/"
            f"{payload['pose_convention_check']['cameras']['cam5']['seeded_default'][1]}"
        ),
        "VthreebVtwoPrecision": f"{payload['v2_comparison']['v2']['precision']:.3f}",
        "VthreebVtwoCount": (
            f"{payload['v2_comparison']['v2']['correct']}/"
            f"{payload['v2_comparison']['v2']['accepted']}"
        ),
    }


def convention_macros() -> dict[str, str]:
    """The CAM5 convention-shift sensitivity, from its own artifact.

    Every number here comes from `convention_shift_sensitivity.json`, including
    the constant itself - which that file in turn reads from the pose
    diagnostic. Nothing in the chain from the measurement to the report sentence
    is typed by hand, which is the only thing that keeps "the peak coincides
    with the independent measurement" checkable rather than asserted.
    """

    if not CONVENTION.is_file():
        return {}
    payload = json.loads(CONVENTION.read_text())
    camera = payload["convention_shift_sensitivity"]["cam5"]
    constant = camera["constant"]
    applied = camera["measured_shift_frames"]
    methods = camera["methods"]

    def at(method: str, shift: int) -> dict:
        return methods[method]["by_shift"][str(shift)]

    def precision(method: str, shift: int) -> str:
        return f"{at(method, shift)['precision']:.2f}"

    shifted = at("submitted_unified", applied)
    bounds = shifted["precision_wilson_95"]
    cam_zero = json.loads(POSE.read_text())["cameras"]["cam0"]
    cam_zero_frames = round(
        cam_zero["label_convention_conversion"]["gap_frames_at_median_sweep"]
    )
    return {
        "ConventionConstantFrames": _signed(
            constant["gap_frames_at_median_sweep"], 2
        ),
        "ConventionShiftFrames": _signed(applied, 0),
        "ConventionCamZeroShift": _signed(cam_zero_frames, 0),
        "ConventionShiftedCount": f"{shifted['correct']}/{shifted['accepted']}",
        "ConventionShiftedPrecision": precision("submitted_unified", applied),
        "ConventionShiftedWilson": f"{bounds[0]:.2f}--{bounds[1]:.2f}",
        "ConventionUnshiftedPrecision": precision("submitted_unified", 0),
        "ConventionShiftedAtOne": precision("submitted_unified", -1),
        "ConventionShiftedAtThree": precision("submitted_unified", -3),
        "ConventionVariantShifted": precision("slope_refinement_variant", applied),
        "ConventionPosteriorShifted": precision("posterior_model", applied),
        "ConventionSourceBuilt": constant["source_built_at_utc"][:10],
        **control_macros(payload),
    }


def control_macros(payload: dict) -> dict[str, str]:
    """The same shift on the two sets that decide what the constant means.

    Read from the `control_sweeps` block rather than restated: the whole point
    of the controls is that they were not chosen to agree, so the report must
    quote them from the file that computed them.
    """

    sets = payload.get("control_sweeps", {}).get("sets", {})
    macros: dict[str, str] = {}
    prefixes = {"blind_v2:cam5": "ConventionVtwoCamFive", "blind_v3:cam0": "ConventionVthreeCamZero"}
    for key, prefix in prefixes.items():
        block = sets.get(key)
        if not block:
            continue
        for shift, suffix in ((0, "AtZero"), (-1, "AtOne"), (-2, "AtTwo"), (-3, "AtThree")):
            row = block["by_shift"].get(str(shift))
            if not row:
                continue
            macros[f"{prefix}{suffix}"] = f"{row['correct']}/{row['accepted']}"
            macros[f"{prefix}{suffix}Precision"] = f"{row['precision']:.3f}"
            macros[f"{prefix}{suffix}Signed"] = row["signed_late_early_inside"]
        macros[f"{prefix}Peak"] = _signed(block["peak_shift_frames"], 0)
        macros[f"{prefix}Rows"] = str(block["accepted_match_rows"])
    return macros


def task3_macros() -> dict[str, str]:
    """Numbers the Task 3 policy table used to state without an artifact.

    Two of them: the strict-geometry collapse (top tier reserved for frames a
    RootSIFT/RANSAC check actually visited) and the pre-route frames the phase
    detector excludes. Both now come from files that ship.
    """

    macros: dict[str, str] = {}
    if STRICT_GEOMETRY.is_file():
        strict = json.loads(STRICT_GEOMETRY.read_text())["status_counts"]
        shipped = json.loads(
            Path(str(UNIFIED_MANIFEST).format(camera="cam0")).read_text()
        )["status_counts"]
        loose = shipped["accepted_strong"]
        tight = strict["accepted_strong"]
        macros["StrictGeometryBefore"] = f"{loose:,}".replace(",", "{,}")
        macros["StrictGeometryAfter"] = f"{tight:,}".replace(",", "{,}")
        macros["StrictGeometryUnverified"] = _percent((loose - tight) / loose, 0)
        macros["StrictGeometryPropagated"] = (
            f"{strict['accepted_geometry_propagated']:,}".replace(",", "{,}")
        )
    for camera, name in (("cam0", "CamZero"), ("cam5", "CamFive")):
        manifest = Path(str(UNIFIED_MANIFEST).format(camera=camera))
        if not manifest.is_file():
            continue
        structure = json.loads(manifest.read_text()).get("many_to_one_structure")
        if not structure:
            continue
        macros[f"ManyToOne{name}"] = _percent(
            structure["fraction_of_mapped_rows_sharing_a_partner"], 0
        )
        macros[f"ManyToOneMax{name}"] = str(structure["max_multiplicity"])
    if BAYES_DIAGNOSTICS.is_file():
        diagnostics = json.loads(BAYES_DIAGNOSTICS.read_text())["with_motion"]
        for camera, name in (("cam0", "CamZero"), ("cam5", "CamFive")):
            macros[f"Ece{name}"] = (
                f"{diagnostics[camera]['expected_calibration_error']:.3f}"
            )
    if LOOP_POSTERIOR.is_file():
        closure = json.loads(LOOP_POSTERIOR.read_text())["cameras"]
        for camera, name in (("cam0", "CamZero"), ("cam5", "CamFive")):
            block = closure[camera]["endpoint_closure"]
            macros[f"PosteriorClosure{name}"] = (
                f"{block['consistent']}/{block['pairs_with_both_ends_mapped']}"
            )
            macros[f"RevisitPairs{name}"] = str(block["revisit_pairs_found"])
    if POLICY_SUMMARY.is_file():
        with POLICY_SUMMARY.open(newline="") as handle:
            total = sum(
                int(row["canonical_decoded_images"]) for row in csv.DictReader(handle)
            )
        macros["DecodedImages"] = f"{total:,}".replace(",", "{,}")
    if QUALITY_FLAGS_CSV.is_file():
        flags = {}
        with QUALITY_FLAGS_CSV.open(newline="") as handle:
            for row in csv.DictReader(handle):
                flags[(row["run"], row["camera"], int(row["ordinal"]))] = row["flag"]
        for camera, name in (("cam0", "CamZero"), ("cam5", "CamFive")):
            mapping = Path(str(MAPPING_CSV).format(camera=camera))
            if not mapping.is_file():
                continue
            tally: dict[str, int] = {}
            with mapping.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if not row["runB_frame"]:
                        continue
                    a = flags.get(("runA", camera, int(row["runA_frame"])), "")
                    b = flags.get(("runB", camera, int(row["runB_frame"])), "")
                    if "too_dark" in a or "too_dark" in b:
                        tally[row["status"]] = tally.get(row["status"], 0) + 1
            macros[f"DarkStrong{name}"] = str(tally.get("accepted_strong", 0))
            macros[f"DarkSequence{name}"] = str(
                tally.get("accepted_sequence_supported", 0)
            )
    if BOUNDARIES.is_file():
        with BOUNDARIES.open(newline="") as handle:
            rows = {row["run"]: row for row in csv.DictReader(handle)}
        for run, name in (("runA", "RunA"), ("runB", "RunB")):
            macros[f"Parked{name}"] = rows[run]["route_start"]
            macros[f"ParkedStationary{name}"] = (
                f"{rows[run]['stationary_start']}--{rows[run]['stationary_end']}"
            )
    return macros


def topology_macros() -> dict[str, str]:
    """Loop structure and internal aliasing, from `analyze_route_topology.py`.

    These replace four numbers the report used to state with no artifact behind
    them. The recomputed values differ from the old ones, which is the point of
    producing them: the aliasing ratio in particular is about 2.4, not the five
    that was previously asserted.
    """

    if not TOPOLOGY.is_file():
        return {}
    payload = json.loads(TOPOLOGY.read_text())
    macros: dict[str, str] = {}
    for camera, name in (("cam0", "CamZero"), ("cam5", "CamFive")):
        block = payload["cameras"][camera]
        for run, suffix in (("runA", ""), ("runB", "RunB")):
            revisit = block["runs"][run]["revisit"]
            letter = "A" if run == "runA" else "B"
            macros[f"Topology{name}{suffix}Pair"] = (
                f"{letter}{revisit['tail_frame']}$\\to${revisit['head_frame']}"
            )
            macros[f"Topology{name}{suffix}Cosine"] = f"{revisit['best_cosine']:.3f}"
        alias = block["runA_aliasing"]
        macros[f"TopologyAlias{name}"] = f"{alias['non_local_pairs']:,}".replace(
            ",", "{,}"
        )
        macros[f"TopologyAliasFraction{name}"] = _percent(alias["non_local_fraction"])
    ratio = payload["aliasing_ratio"]
    values = [value for value in ratio["by_threshold"].values() if value is not None]
    macros["TopologyRatio"] = f"{ratio['ratio_cam5_over_cam0']:.1f}"
    macros["TopologyRatioRange"] = f"{min(values):.1f}--{max(values):.1f}"
    definitions = payload["definitions"]
    macros["TopologySeparation"] = str(definitions["non_local_separation_frames"])
    macros["TopologyCosine"] = f"{definitions['alias_cosine_threshold']:.2f}"
    macros["TopologyEligiblePairs"] = (
        f"{payload['cameras']['cam0']['runA_aliasing']['eligible_pairs']:,}".replace(
            ",", "{,}"
        )
    )
    return macros


def sweep_rate_macros() -> dict[str, str]:
    """How fast the scene sweeps across each camera, measured not estimated.

    The report used to say CAM5's mid-band "translates about 104 px per frame at
    native resolution". That number was typed, and nothing produced it. The
    measurement that exists is RootSIFT median |dx| between consecutive frames
    at the geometry raster (`build_sweep_rate.py`), which is also the rate the
    pose diagnostic divides by to turn its 18.5 px convention gap into frames.
    It is reported at the raster it was measured on and scaled to native width,
    with the scaling shown rather than folded in.
    """

    if not SWEEP_RATE.is_file():
        return {}
    streams = json.loads(SWEEP_RATE.read_text())["streams"]
    scale = NATIVE_WIDTH / GEOMETRY_RASTER[0]
    macros = {
        "SweepRaster": f"{GEOMETRY_RASTER[0]}$\\times${GEOMETRY_RASTER[1]}",
        "SweepNativeWidth": f"{NATIVE_WIDTH:,}",
    }
    for camera, name in (("cam0", "CamZero"), ("cam5", "CamFive")):
        rate = streams[f"runA_{camera}"]["median_px_per_frame"]
        macros[f"SweepRate{name}"] = f"{rate:.0f}"
        macros[f"SweepRate{name}Native"] = f"{rate * scale:.0f}"
        macros[f"SweepPninety{name}"] = f"{streams[f'runA_{camera}']['p90']:.0f}"
    return macros


def lateral_macros() -> dict[str, str]:
    """Task 1 lateral-offset headline, if that study has been built.

    The report guards these with \\providecommand, so a build without the study
    still compiles; nothing here invents a number when the file is absent.
    """

    if not LATERAL.is_file():
        return {}
    payload = json.loads(LATERAL.read_text())
    consistency = payload["sign_consistency"]
    return {
        "LateralFlaggedCamZero": _percent(payload["cameras"]["cam0"]["flagged_fraction"]),
        "LateralFlaggedCamFive": _percent(payload["cameras"]["cam5"]["flagged_fraction"]),
        "LateralPairs": str(consistency["paired_samples"]),
        "LateralOppositeSign": _percent(consistency["opposite_sign_fraction_all"]),
        "LateralSignP": f"{consistency['binomial_p_opposite_all']:.3f}",
        "LateralDetected": (
            "yes" if payload["conclusion"]["lateral_offset_detected"] else "no"
        ),
    }


def facts_macros() -> None:
    """Single-value facts, as macros, so prose numbers also come from artifacts."""

    payload = json.loads(RESCORE.read_text())
    consistency = json.loads(CONSISTENCY.read_text())
    spread = consistency["cross_camera_agreement"]["disagreement_about_the_running_offset"]
    v1 = payload["methods"]["v1_frozen_submitted"]
    v2 = payload["methods"]["v2_unified"]
    macros = {
        "BlindLabelsPerCamera": "40",
        "UndeterminedLabels": str(
            v2["cam0"]["labels_undetermined"] + v2["cam5"]["labels_undetermined"]
        ),
        "SubCamZeroPrecision": f"{v2['cam0']['corrected']['precision']:.3f}",
        "SubCamZeroCoverage": _percent(v2["cam0"]["corrected"]["coverage_of_determinable"]),
        "SubCamFivePrecision": f"{v2['cam5']['corrected']['precision']:.3f}",
        "SubCamFiveCoverage": _percent(v2["cam5"]["corrected"]["coverage_of_determinable"]),
        "SubCamFiveWithinTwo": (
            f"{v2['cam5']['preferred_frame_error']['within_two_frames']}/"
            f"{v2['cam5']['preferred_frame_error']['n']}"
        ),
        "SparseCamZeroPrecision": f"{v1['cam0']['corrected']['precision']:.3f}",
        "SparseCamZeroCoverage": _percent(v1["cam0"]["corrected"]["coverage_of_determinable"]),
        "SparseCamFiveCoverage": _percent(v1["cam5"]["corrected"]["coverage_of_determinable"]),
        "SparseCamFivePrecision": f"{v1['cam5']['corrected']['precision']:.3f}",
        "SparseCamFiveAccepted": str(v1["cam5"]["corrected"]["accepted"]),
        "AsLabelledSubCamZero": f"{v2['cam0']['as_labelled']['precision']:.3f}",
        "ConsistencyMedian": f"{spread['median_absolute']:.0f}",
        "ConsistencyPninetyfive": f"{spread['p95_absolute']:.0f}",
        "AbstentionSmallHole": _percent(
            json.loads(HOLDOUT.read_text())["cameras"]["cam0"]["by_hole_width"]["60"]["detection_rate"]
        ),
        "AbstentionLargeHole": _percent(
            json.loads(HOLDOUT.read_text())["cameras"]["cam0"]["by_hole_width"]["400"]["detection_rate"]
        ),
        "AbstentionFalseAlarm": _percent(
            json.loads(HOLDOUT.read_text())["cameras"]["cam0"]["by_hole_width"]["400"]["false_alarm_rate"]
        ),
        "AbstentionCamFive": _percent(
            json.loads(HOLDOUT.read_text())["cameras"]["cam5"]["by_hole_width"]["150"]["detection_rate"]
        ),
        # CAM5 falls with hole width where CAM0 rises, and buys that with a
        # false-alarm rate about three times CAM0's. Quoting only its best cell
        # was the selective reading this replaces.
        "AbstentionCamFiveByWidth": "/".join(
            f"{json.loads(HOLDOUT.read_text())['cameras']['cam5']['by_hole_width'][width]['detection_rate']:.2f}"
            for width in ("60", "150", "400")
        ),
        "AbstentionCamZeroByWidth": "/".join(
            f"{json.loads(HOLDOUT.read_text())['cameras']['cam0']['by_hole_width'][width]['detection_rate']:.2f}"
            for width in ("60", "150", "400")
        ),
        "AbstentionFalseAlarmCamFive": _percent(
            json.loads(HOLDOUT.read_text())["cameras"]["cam5"]["by_hole_width"]["400"]["false_alarm_rate"], 1
        ),
        "AbstentionTrials": str(
            json.loads(HOLDOUT.read_text())["cameras"]["cam0"]["by_hole_width"]["400"]["trials"]
        ),
        "ConsistencyFrames": _thousands(
            consistency["cross_camera_agreement"]["frames_mapped_by_both_cameras"]
        ),
    }
    macros.update(evidence_macros())
    body = "\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()
    )
    write("facts.tex", body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print("Generated report fragments:")
    bitstream_table()
    mask_table()
    resolution_table()
    blind_comparison_table()
    tolerance_table()
    facts_macros()
    print(f"\nInto {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
