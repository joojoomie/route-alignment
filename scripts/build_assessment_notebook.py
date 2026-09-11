#!/usr/bin/env python3
"""Compose (and optionally execute) assessment.ipynb, the working behind the report.

The notebook follows the report section by section. Every number is either read
from the evidence artifact the report cites or recomputed live by calling the
module in scripts/ that produced it; where both exist the notebook checks that
they agree. Cells that need the raw recordings say so and skip in a bundle that
does not carry them.

    python scripts/build_assessment_notebook.py            # write the notebook
    python scripts/build_assessment_notebook.py --execute  # write and execute it
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "assessment.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# --------------------------------------------------------------------------- #
md(r"""
# Route alignment — the working behind the report

Two dashcam traversals of one loop route, two side-facing fisheye cameras, 10 FPS.
The report (`report/route_alignment_report.pdf`) states results; this
notebook shows where each of them comes from, in the report's own order.

**How to read it.** Each section calls the module in `scripts/` that produced the
number, or reads the evidence artifact under `outputs/` that the report cites, and
where both exist it checks that they agree. The code stays in the scripts: the
notebook imports and calls it rather than re-typing it, so what runs here is what
ran for the report.

**What it never does.** No sealed label set is scored again with different
parameters. Every held-out figure is read from its single scoring pass; the live
computations are re-derivations from the sealed labels and the frozen mapping
files, and they must reproduce the shipped figure.

**What needs the raw recordings.** The HEVC files are not in the submission
bundle. The cells that parse or decode them say so at the top and skip cleanly
when the files are absent; their outputs are kept as executed.

| Section | Report | What runs |
|---|---|---|
| 1 | Characterising the recordings | bitstream scan, integrity gates, sidecar timing, cameras, reliability mask, route topology |
| 2 | Method | frozen parameters verified against live code, resolution study, mapping contract, evaluation ordering |
| 3 | Results and two protocol defects | tolerance curve recomputed, label semantics, targeted sets, convention constant, abstention, CAM5 mechanism, label-free signals, exposure |
| 4 | Downstream policy | policy table, quality flags, slope gate, loop closure, posterior, the strict single-frame table |
""")

code(r"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image as ShowImage, display

ROOT = Path.cwd() if (Path.cwd() / "scripts").is_dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "scripts"))
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

RUNS, CAMERAS = ("runA", "runB"), ("cam0", "cam5")


def load(relative):
    return json.loads((ROOT / relative).read_text())


def available(*relative):
    missing = [r for r in relative if not (ROOT / r).exists()]
    if missing:
        print("skipped here: not in this bundle ->", ", ".join(missing))
    return not missing


def wilson(interval):
    return f"{interval[0]:.2f}-{interval[1]:.2f}"


import frame_service

RAW = all(frame_service.source_available(run, camera) for run in RUNS for camera in CAMERAS)
print("repository root:", ROOT.name)
print("raw recordings present:", RAW)
""")

# ----------------------------------------------------------------- section 1 #
md(r"""
## 1. Characterising the recordings

### 1.1 Picture counts, rasters and picture order, read from the bitstream

The report's inventory table comes from a purpose-written HEVC Annex-B scanner
(`scripts/hevc_bitstream.py`) that parses NAL units, the sequence parameter
set and every slice header, and reconstructs picture order count independently of
libavcodec. The table below is the shipped artifact; when the recordings are
present the scanner runs again and must agree with it exactly.
""")

code(r"""
inventory = pd.read_csv(ROOT / "outputs/task1/task1_bitstream_summary.csv")
display(inventory[["run", "camera", "bitstream_picture_count", "coded_raster", "display_raster",
                   "slice_type_I", "slice_type_P", "slice_type_B", "reordered_picture_fraction"]])

# needs the raw recordings
if RAW:
    import hevc_bitstream

    expected = inventory.set_index(["run", "camera"])
    for run in RUNS:
        for camera in CAMERAS:
            scan = hevc_bitstream.scan(frame_service.source_path(run, camera))
            scan.verify_display_order_is_permutation()
            pictures = len(scan.pictures)
            reordered = scan.reordered_picture_count
            assert pictures == expected.loc[(run, camera), "bitstream_picture_count"]
            assert abs(reordered / pictures - expected.loc[(run, camera), "reordered_picture_fraction"]) < 1e-6
            first = [p.picture_order_count for p in scan.pictures[:9]]
            print(f"{run}/{camera}: {pictures} pictures, {reordered} reordered ({reordered / pictures:.1%}), "
                  f"decode-order POC {first}, SEI {scan.sei_payload_counts}, "
                  f"file size mod 4096 = {scan.file_size % 4096}")
    print("live scan agrees with the shipped inventory on all four files")
""")

md(r"""
The decode-order picture-order-count sequence 0, 4, 2, 1, 3, 8, 6, 5, 7 is the
four-frame B-pyramid: a tool that enumerates packets and calls the n-th one
"frame n" scrambles the sequence locally, by the same magnitude as the alignment
error being measured. The only SEI present is `pic_timing`; there is no GPS, no
odometry and no calibration anywhere in the four files or the two sidecars.

### 1.2 Integrity gates

`frame_service.verify()` runs every gate over all four recordings: published
checksum, decoded count against the bitstream count, display order a clean
permutation, raster, sidecar monotonicity, parked-tail bound, and the shared
sidecar. Two gates are **waived, not passed**; the waiver is a recorded exception,
not a silent pass, because a finding that cannot fail a build is a finding that
gets forgotten.
""")

code(r"""
# needs the raw recordings (decodes all four files; a few minutes)
if RAW:
    results = frame_service.verify()
    width = max(len(item.gate) for item in results)
    for item in results:
        marker = {"pass": "ok  ", "waived": "WAIV", "FAIL": "FAIL"}[item.status]
        print(f"[{marker}] {item.gate:<{width}}  {item.subject:<11} {item.detail}")
    waived = sum(1 for item in results if item.waived)
    failed = sum(1 for item in results if item.status == "FAIL")
    print(f"\n{len(results)} gates: {len(results) - waived - failed} passed, {waived} waived, {failed} failed")
else:
    waivers = load("outputs/task1/integrity_waivers.json")
    print(json.dumps(waivers, indent=2)[:1500])
""")

md(r"""
### 1.3 Timing: what the sidecars can and cannot say

Each run ships one integer-nanosecond timestamp log. Within a run the two
cameras' logs are byte-identical, yet the two cameras decode to different frame
counts (Run B: 2,622 and 2,615 against one 2,663-row log), so a sidecar row is a
capture-log event, not a decoded frame, and no per-frame capture time is
published. Cadence is measured from the log, not read from the stream's
declaration (which contradicts itself: `r_frame_rate` 10/1, `avg_frame_rate` 25/1).
""")

code(r"""
timing = pd.read_csv(ROOT / "outputs/task1/task1_timing_summary.csv")
display(timing[["run", "timestamp_records", "strictly_increasing", "sidecar_schedules_identical", "start_utc",
                "timestamp_log_span_s", "mean_capture_event_cadence_hz", "interval_median_ms", "interval_max_ms"]])

# needs the raw sidecars
if RAW:
    counts = inventory.set_index(["run", "camera"])["bitstream_picture_count"]
    for run in RUNS:
        stamps = {camera: frame_service.load_timestamps(frame_service.sidecar_path(run, camera)) for camera in CAMERAS}
        t = stamps["cam0"]
        span = (t[-1] - t[0]) / 1e9
        start = datetime.fromtimestamp(t[0] / 1e9, tz=timezone.utc)
        print(f"{run}: {len(t)} rows from {start:%Y-%m-%d %H:%M:%S} UTC, span {span:.3f} s, "
              f"cadence {(len(t) - 1) / span:.4f} Hz, strictly increasing {bool(np.all(np.diff(t) > 0))}, "
              f"cam0 and cam5 logs identical {np.array_equal(stamps['cam0'], stamps['cam5'])}, "
              f"decoded frames cam0 {counts[(run, 'cam0')]} / cam5 {counts[(run, 'cam5')]}")
""")

md(r"""
### 1.4 Declared against measured

Every figure in the Task 1 inventory carries its provenance class. The colour
finding belongs here: only `color_range=tv` is signalled, so the decoder's matrix
choice depends on frame height, and pinning it changes 3.17% of decoded bytes.
The canonical decode therefore pins BT.709 and every cache key carries source
hash, raster and colour id.
""")

code(r"""
provenance = pd.read_csv(ROOT / "outputs/task1/task1_provenance.csv")
with pd.option_context("display.max_colwidth", 110):
    display(provenance)
""")

md(r"""
### 1.5 What the cameras actually are

CAM0 faces left and CAM5 right; both are wide fisheye with heavy vignetting. The
scene sweeps laterally, so appearance depends on lane position far more than on a
forward camera. Run B is darker and rain sits on its lens, static for the whole
recording. Two measurements set the difficulty: the sweep rate (RootSIFT median
|dx| between consecutive frames at the geometry raster) and the exposure statistics.
""")

code(r"""
if available("outputs/task2_motion_bayes/sweep_rate_summary.json"):
    sweep = load("outputs/task2_motion_bayes/sweep_rate_summary.json")
    display(pd.DataFrame(sweep["streams"]).T[["frames", "median_px_per_frame", "p10", "p90", "fraction_below_2px"]])

quality = load("outputs/task3/frame_quality_flags.json")
print(quality["thresholds"]["rule"])
display(pd.DataFrame(quality["streams"]).T[["frames", "too_dark", "too_dark_fraction", "mean_luma_median",
                                            "mean_luma_min", "longest_dark_stretch"]])
""")

md(r"""
**The reliability mask, measured rather than predicted.** Content attached to the
camera does not move when the vehicle does. The statistic

$$\text{staticness} = \frac{|\nabla(\text{temporal median})|}{\text{temporal std} + 1}$$

is close to scale-free, which matters because Run B is 2.2× darker and half as
sharp, so a fixed quantile would mask the same fraction of both runs and say
nothing. The same rule finds the wing mirror and the vignette in the dry run and
grows the droplets in the wet one; no trained model is involved
(`scripts/build_reliability_masks.py`).
""")

code(r"""
masks = load("outputs/task2_masks/reliability_mask_summary.json")
print(f"mask version {masks['mask_version']}; learned components: {masks['learned_components']}")
rows = []
for key, stats in masks["streams"].items():
    run, camera = key.split("_")
    rows.append({"run": run, "camera": camera,
                 "camera_attached": f"{stats['camera_attached_fraction']:.2%}",
                 "never_bright": f"{stats['never_bright_fraction']:.2%}",
                 "excluded": f"{stats['exclusion_fraction']:.2%}"})
display(pd.DataFrame(rows))

if available("outputs/task2_masks/runA_cam0_mask_overlay.png", "outputs/task2_masks/runB_cam0_mask_overlay.png"):
    for run in RUNS:
        print(f"{run} CAM0: red = camera-attached, blue = optically dead")
        display(ShowImage(filename=str(ROOT / f"outputs/task2_masks/{run}_cam0_mask_overlay.png"), width=520))
""")

md(r"""
### 1.6 Route topology: a loop that also aliases internally

Retrieval over the matcher's own frozen descriptors finds a strong tail-to-head
revisit in all four sequences, so the route is a loop and its endpoints alias.
Interior look-alike pairs far apart along the route are counted too; CAM5 has
about 2.4× CAM0's density of them. Both facts shape the method: no wrap and no
interpolation across the seam, and a global rather than a banded search.
""")

code(r"""
topology = load("outputs/task1/route_topology.json")
print(topology["what_this_measures"], "\n")
for camera in CAMERAS:
    entry = topology["cameras"][camera]
    for run, stats in entry["runs"].items():
        revisit = stats["revisit"]
        print(f"{camera} {run}: tail frame {revisit['tail_frame']} revisits head frame {revisit['head_frame']} "
              f"at cosine {revisit['best_cosine']} (route start {stats['route_start']}; median tail-head cosine {revisit['median_cosine']})")
    alias = entry["runA_aliasing"]
    print(f"{camera} Run A interior aliasing: {alias['non_local_pairs']:,} of {alias['eligible_pairs']:,} pairs at least "
          f"{alias['separation_frames']} frames apart exceed cosine {alias['cosine_threshold']} ({alias['non_local_fraction']:.1%})\n")
ratio = topology["aliasing_ratio"]
print(f"CAM5 / CAM0 non-local look-alike pairs: {ratio['cam5_pairs']:,} / {ratio['cam0_pairs']:,} = "
      f"{ratio['ratio_cam5_over_cam0']}x  (by cosine threshold: {ratio['by_threshold']})")
display(ShowImage(filename=str(ROOT / "outputs/task1/cam0_loop_closure_comparison.png"), width=760))
""")

# ----------------------------------------------------------------- section 2 #
md(r"""
## 2. Method

### 2.1 The stages, and the frozen parameters verified against live code

Canonical decoded image → reliability mask → frozen DINOv2-S masked descriptor →
full-route coarse similarity → monotonic path with a slope prior → full-cadence
refinement → local RootSIFT/RANSAC → accept or abstain
(`scripts/build_task2_unified.py`). Each stage answers an observed failure of the
simpler one: linear progress drifts 12.4 s, nearest-neighbour retrieval confuses
the repeated awnings and corridors, and forced dense output cannot be right where
nothing corresponds.

The method was frozen before any blind label was imported. The freeze file hashes
every parameter, and the verifier below refuses if any live constant has drifted.
""")

code(r"""
freeze = load("outputs/task2_evaluation/method_freeze_v2.json")
print(freeze["status"], "at", freeze["frozen_at_utc"])
print("training performed:", freeze["training_performed"][:120])
frozen = {key: value for key, value in freeze["frozen_parameters"].items() if not isinstance(value, dict)}
display(pd.Series(frozen, name="frozen value").to_frame())

import freeze_method_v2
import freeze_method_v3

freeze_method_v2.verify()
freeze_method_v3.verify(quiet=True)
freeze_method_v3.verify(quiet=True, set_name="blind_v3b")
print("all three freezes verify against the live code and the judged artifacts")
""")

md(r"""
### 2.2 The geometry raster was chosen by measurement

RootSIFT/RANSAC runs at 896×672, not native. The controlled study
(`scripts/run_geometry_resolution_study.py`, on the original calibration labels
reclassified as development data) shows passing peaks at 896×672 and *falls* at
native: fine foliage texture inflates the raw match count while the inlier ratio
drops. Keypoint count alone would have chosen wrong.
""")

code(r"""
resolution = pd.read_csv(ROOT / "outputs/task2_unified/resolution_study/geometry_resolution_study.csv")
display(resolution)
if available("outputs/task2_unified/resolution_study/geometry_resolution_study.json"):
    study = load("outputs/task2_unified/resolution_study/geometry_resolution_study.json")
    print("selected:", study["selected_resolution"], "|", study["selection_rule"])
    print(study["finding"])
""")

md(r"""
### 2.3 The mapping contract, and what the shipped files contain

Every Run A ordinal gets one row; `runB_frame` may be empty but `status` may not;
the mapping is monotonic. `confidence` is an ordinal rank, not a probability, and
its 0.50 tier never occurs. The mapping is many-to-one where Run B is slower,
which is expected and fatal to a consumer keying on `runB_frame`.
""")

code(r"""
import build_task2_unified

for camera in CAMERAS:
    build_task2_unified.validate_mapping(ROOT / f"outputs/task2_unified/{camera}/frame_mapping_unified.csv", 2674)

rows = []
for camera in CAMERAS:
    manifest = load(f"outputs/task2_unified/{camera}/unified_manifest.json")
    mapping = pd.read_csv(ROOT / f"outputs/task2_unified/{camera}/frame_mapping_unified.csv")
    mapped = mapping.dropna(subset=["runB_frame"])
    multiplicity = mapped["runB_frame"].value_counts()
    rows.append({"camera": camera, "rows": manifest["rows"], "mapped": manifest["mapped"],
                 "coverage": f"{manifest['coverage']:.3f}", **manifest["status_counts"],
                 "confidence tiers": dict(mapping["confidence"].round(2).value_counts().sort_index()),
                 "rows sharing a partner": f"{(multiplicity[mapped['runB_frame']].values > 1).mean():.0%}",
                 "max multiplicity": int(multiplicity.max())})
with pd.option_context("display.max_colwidth", 80):
    display(pd.DataFrame(rows).set_index("camera").T)
""")

md(r"""
### 2.4 Evaluation design: the ordering is on record

Prediction-blind labels: the page showed the query frame and a scrubbable Run B
filmstrip, nothing else. For every set the query set was exported, the method
frozen, the labels sealed, and the set scored once, and each file cites the hash
of the one before it. That records *ordering*, not custody; the report says so.
""")

code(r"""
seal_v2 = load("outputs/task2_evaluation/blind_v2/label_seal.json")
metrics_v2 = load("outputs/task2_evaluation/blind_v2/blind_v2_metrics.json")
v3 = load("outputs/task2_evaluation/blind_v3/blind_v3_results.json")
v3b = load("outputs/task2_evaluation/blind_v3b/blind_v3b_results.json")

timeline = pd.DataFrame([
    {"set": "blind_v2", "labels": seal_v2["rows"],
     "query set exported": seal_v2["query_set_exported_at_utc_recovered_from_frame_mtimes"],
     "method frozen": metrics_v2["method_frozen_at_utc"], "labels sealed": seal_v2["sealed_at_utc"],
     "scored": metrics_v2["evaluated_at_utc"]},
    {"set": "blind_v3", "labels": v3["labels"], "query set exported": v3["query_set_exported_at_utc"],
     "method frozen": v3["method_frozen_at_utc"], "labels sealed": v3["label_sealed_at_utc"],
     "scored": v3["evaluated_at_utc"]},
    {"set": "blind_v3b", "labels": v3b["labels"], "query set exported": v3b["query_set_exported_at_utc"],
     "method frozen": v3b["method_frozen_at_utc"], "labels sealed": v3b["label_sealed_at_utc"],
     "scored": v3b["evaluated_at_utc"]},
]).set_index("set")
for column in timeline.columns[1:]:
    timeline[column] = pd.to_datetime(timeline[column]).dt.strftime("%m-%d %H:%M:%S")
display(timeline)

stages = ["query set exported", "method frozen", "labels sealed", "scored"]
for name, row in timeline.iterrows():
    assert list(row[stages]) == sorted(row[stages]), name
print("every set: exported < frozen < sealed < scored")
print("\nblindness checks performed on v2:")
for check in seal_v2["blindness_checks_performed"]:
    print(" -", check)
print("\nseal trust model:", seal_v2["trust_model"])
""")

# ----------------------------------------------------------------- section 3 #
md(r"""
## 3. Results, and two protocol defects

### 3.1 Precision at frame tolerances, recomputed from the sealed labels

The headline reading: the share of accepted match labels whose prediction lies
within ±k frames of the annotator's acceptable interval (distance 0 inside it).
One frame is 0.1 s, about a metre of travel, so ±2 is roughly 2 m, an order of
magnitude tighter than the 25 m or ±10-frames-at-1 FPS tolerances published
place-recognition protocols use. Cameras are never pooled. The cell calls
`evaluate_tolerance_curve.curve()` on the sealed label files and the frozen
mapping files and checks the result against the shipped `tolerance_curve.json`.
""")

code(r"""
import evaluate_tolerance_curve as tol

shipped = load("outputs/task2_evaluation/tolerance_curve.json")
rows, agree = [], True
for set_name, spec in tol.LABEL_SETS.items():
    with (ROOT / spec["labels"]).open(newline="") as handle:
        labels = [r for r in csv.DictReader(handle) if r["label"] == "match"]
    for camera in spec["cameras"]:
        camera_labels = [r for r in labels if r["camera_id"] == camera]
        for method, pattern in spec["methods"].items():
            if not (ROOT / pattern.format(camera=camera)).is_file():
                continue
            result = tol.curve(camera_labels, tol.load_mapping(pattern, camera))
            reference = shipped["sets"][set_name]["cameras"][camera][method]
            agree &= all(result["within"][str(t)]["fraction"] == reference["within"][str(t)]["fraction"]
                         for t in tol.TOLERANCES)
            rows.append({"set": set_name, "camera": camera, "method": method,
                         "match labels": result["match_labels"], "accepted": result["accepted"],
                         "coverage": f"{result['coverage']:.0%}",
                         **{f"±{t}": (f"{result['within'][str(t)]['fraction']:.1%}"
                                      if result["within"][str(t)]["fraction"] is not None else "-")
                            for t in tol.TOLERANCES}})
display(pd.DataFrame(rows))
print(f"headline tolerance ±{tol.HEADLINE_TOLERANCE}; recomputed curves equal the shipped tolerance_curve.json: {agree}")
print("\n" + shipped["literature_context"])
""")

md(r"""
### 3.2 A label-semantics correction, made after the fact

The v2 page offered a match or "no usable match", merging two facts: that Run B
has no counterpart here, and that the annotator could not tell. Every such label
meant the second, and that is checkable without any result: queries were drawn
inside the stretch both runs traversed, so a genuine absence was impossible by
construction. Undetermined queries leave the denominator; the correction was made
*after* the results were seen, applied identically to both methods, and both
readings are kept (the strict table in §4.3 carries the uncorrected column).
""")

code(r"""
labels_v2 = pd.read_csv(ROOT / "outputs/task2_evaluation/blind_v2/blind_v2_labels.csv")
route = load("outputs/task2_unified/cam0/unified_manifest.json")["route"]
undetermined = labels_v2[labels_v2["label"] != "match"]
inside = undetermined["runA_frame"].between(route["runA_start"], route["runA_end_exclusive"] - 1)
print(f"{len(undetermined)} of {len(labels_v2)} v2 labels were 'no usable match' "
      f"(cam0 {int((undetermined['camera_id'] == 'cam0').sum())}, cam5 {int((undetermined['camera_id'] == 'cam5').sum())})")
print(f"all of them lie inside the traversed stretch Run A {route['runA_start']}-{route['runA_end_exclusive'] - 1}: {bool(inside.all())}")
print("so none can be a genuine absence; they are undetermined and leave the precision denominator")
""")

md(r"""
### 3.3 The targeted sets: v3, and the CAM5 redo v3b

With the v2 labels spent, 48 fresh queries were sealed against a freeze naming
the submitted mapping, the slope variant, the posterior and the gates, and scored
once. Half target kinks and dark stretches, half are a uniform *corridor* control;
targeting used the submitted mapping's own structure, so **only the corridor
stratum is an unbiased route-wide rate**.

**The CAM5 half of v3 was withdrawn** for a defect found in the sealed labels
themselves: the viewer displaced Run B by the measured pose offset instead of its
negation, and the annotator's slider corrections correlate with the mapping's
errors. CAM0's offset is zero, so CAM0 stands. CAM5 was re-asked on 24 fresh
queries under its own freeze with the sign fixed and the slider clamped (v3b).
""")

code(r"""
def method_rows(results, set_name):
    rows = []
    for method, entry in results["methods"].items():
        for camera, stats in entry["cameras"].items():
            rows.append({"set": set_name, "method": method, "camera": camera, "held out": entry["held_out"],
                         "match labels": stats["match_labels"], "accepted": stats["accepted"],
                         "correct": stats["correct"], "strict precision": stats["precision"],
                         "wilson 95%": wilson(stats["precision_wilson_95"]),
                         "within ±2": stats["within_two_rate"], "coverage": stats["coverage_of_match_labels"]})
    return rows

display(pd.DataFrame(method_rows(v3, "blind_v3") + method_rows(v3b, "blind_v3b")))

strata = []
for set_name, results, camera in (("blind_v3", v3, "cam0"), ("blind_v3b", v3b, "cam5")):
    for stratum, stats in results["methods"]["submitted_unified"]["strata"][camera].items():
        strata.append({"set": set_name, "camera": camera, "stratum": stratum, "labels": stats["labels"],
                       "match labels": stats["match_labels"], "accepted": stats["accepted"],
                       "correct": stats["correct"], "precision": stats["precision"],
                       "within ±2": stats["within_two_rate"]})
print("submitted mapping by stratum (only 'corridor' is an unbiased route-wide rate):")
display(pd.DataFrame(strata))
""")

code(r"""
# The tool defect, re-derived from the sealed v3 labels and cases.
cases_v3 = pd.read_csv(ROOT / "outputs/task2_evaluation/blind_v3/blind_v3_cases.csv")
labels_v3 = pd.read_csv(ROOT / "outputs/task2_evaluation/blind_v3/blind_v3_labels.csv")
joined = cases_v3[(cases_v3["method"] == "submitted_unified") & cases_v3["accepted"]
                  & (cases_v3["ground_truth_label"] == "match")].merge(
    labels_v3[["query_id", "pose_dx_px", "pose_dy_px"]], on="query_id")
for camera in CAMERAS:
    subset = joined[joined["camera_id"] == camera]
    if subset["pose_dx_px"].std() == 0:
        print(f"{camera}: slider dx constant at {subset['pose_dx_px'].iloc[0]:+.0f} px on {len(subset)} labels (no offset to correct)")
        continue
    rho = subset["pose_dx_px"].corr(subset["preferred_frame_error"], method="spearman")
    print(f"{camera}: slider dx used on {len(subset)} accepted match labels, median {subset['pose_dx_px'].median():+.0f} px, "
          f"range {subset['pose_dx_px'].min():+.0f}..{subset['pose_dx_px'].max():+.0f}; "
          f"Spearman(slider dx, signed frame error) = {rho:.2f}")
print("\nCAM5's slider absorbed part of the frame error, so its v3 half cannot judge the mapping; see TOOL_DEFECT_cam5.md")
print(f"v3b: sign fixed at source, every row at the exported default, slider clamped; {v3b['labels']} fresh CAM5 queries")
""")

md(r"""
### 3.4 One pre-measured constant explains part of CAM5's residual

The matcher's "same place" is image content; the labels' is a world position.
For a side-facing camera whose pose changed between the runs the two differ by a
constant: the 18.5 px gap at CAM5's sweep rate is −1.79 frames, measured on the
v2 labels **before** v3b was drawn. Applying the rounded −2 to every CAM5
prediction (the submitted file unchanged) is a sensitivity reading on a spent
set, not a held-out number. Two controls bound it: on the mixed-convention v2
labels the same shift is neutral, and on CAM0, whose offset is zero, the sweep
peaks at 0.
""")

code(r"""
sensitivity = load("outputs/task2_evaluation/blind_v3b/convention_shift_sensitivity.json")
block = sensitivity["convention_shift_sensitivity"]["cam5"]
constant = block["constant"]
print(f"constant: {constant['gap_px']} px = {constant['gap_frames_at_median_sweep']} frames, rounded {constant['rounded_frames']}; "
      f"measured on: {constant['measured_on'][:60]}")
print(sensitivity["evidence_status"][:160], "\n")
def shift_table(by_shift):
    frame = pd.DataFrame(by_shift).T.drop(columns=["shift_frames", "signed_interval_error", "signed_interval_counts"], errors="ignore")
    frame["precision_wilson_95"] = frame["precision_wilson_95"].map(wilson)
    return frame

for method, entry in block["methods"].items():
    frame = shift_table(entry["by_shift"])
    frame.index.name = f"{method}: shift (frames)"
    display(frame)
    print(f"  peak at {entry['peak_shift_frames']} frames; peak is the pre-measured constant: {entry['peak_is_the_measured_constant']}\n")

print("controls: the same sweep on two sets the reading was not meant for")
for name, control in sensitivity["control_sweeps"]["sets"].items():
    frame = shift_table(control["by_shift"])
    frame.index.name = f"{name} ({control['role']}): shift"
    display(frame)
    print(f"  peak at {control['peak_shift_frames']}\n")
""")

md(r"""
### 3.5 Abstention: what the labels cannot say, and what can

The v2 set held no true no-correspondence case, so it cannot evaluate abstention.
Withholding a stretch of Run B supplies cases by construction: frames whose
partner lay inside the hole have none. The route also supplies one *real*
absence found by looking at frames: an oncoming lorry fills Run B CAM0 for about
30 frames, and both mappings drive straight through it at confidence 1.00. The
strict-geometry re-read shows why: geometry is propagated between sampled checks,
and reserving the top tier for frames actually checked collapses it.
""")

code(r"""
holdout = load("outputs/task2_abstention/abstention_holdout.json")
print(holdout["scope"])
rows = []
for camera, entry in holdout["cameras"].items():
    for width, stats in entry["by_hole_width"].items():
        rows.append({"camera": camera, "hole (frames)": int(width), "trials": stats["trials"],
                     "detected": f"{stats['detection_rate']:.1%}", "false alarm": f"{stats['false_alarm_rate']:.1%}"})
display(pd.DataFrame(rows))

occlusion = load("outputs/task2_abstention/confirmed_occlusion_runB_cam0.json")
unified = occlusion["unified_mapping"]
print(f"\nconfirmed occlusion: Run B CAM0 {occlusion['interval']['start']}-{occlusion['interval']['end']} "
      f"({occlusion['detail'][0]['occluder']}); source: {occlusion['source']}")
print(f"submitted mapping: {unified['rows_mapped_into_interval']} rows (Run A {unified['runA_frames']['first']}-"
      f"{unified['runA_frames']['last']}) mapped into it, status {unified['status_counts']}, confidence min {unified['confidence']['min']}, "
      f"geometry actually checked on {unified['geometry_checked'].get('true', 0)} of them")
posterior = occlusion["posterior_mapping"]
print(f"posterior model: {posterior['rows_mapped_into_interval']} rows, no-correspondence probability at most "
      f"{posterior['posterior_no_correspondence']['max']}")

strict = load("outputs/task2_unified/cam0/unified_manifest_strict_geometry.json")
loose = load("outputs/task2_unified/cam0/unified_manifest.json")
before, after = loose["status_counts"]["accepted_strong"], strict["status_counts"]["accepted_strong"]
print(f"\nCAM0 top tier when geometry must be verified on the frame itself: {before} -> {after} "
      f"({1 - after / before:.0%} of top-tier acceptances were never verified)")
""")

md(r"""
### 3.6 Why CAM5 is weaker: the camera moved, and it is aimed low

At the matcher's own accepted pairs the CAM5 scene sits displaced by a constant
dx/dy along the whole route, and at the annotator's pairs by a larger dx; CAM0 is
0/0 on both. For a side-facing camera there is no vertical parallax, so dy is
pitch, and a yaw offset and a few frames of travel produce the same horizontal
shift (the degeneracy the diagnostic states). CAM5's failure is a localisation
bias, not place confusion: §3.1 shows it inside ±5 frames on 90% of queries.
""")

code(r"""
pose = load("outputs/task2_pose_offset/pose_offset_diagnostic.json")
print("degeneracy:", pose["degeneracy"][:200], "\n")


def px(stat):
    return f"median {stat['median']:+.1f}, 95% {stat.get('bootstrap_95', stat.get('interval_95', '-'))}, MAD {stat.get('mad', '-')}"


rows = []
for camera in CAMERAS:
    entry = pose["cameras"][camera]
    matcher, label = entry["matcher_convention"], entry["label_convention"]
    rows.append({"camera": camera,
                 "matcher pairs": matcher["pairs_measured"],
                 "matcher dx (px)": px(matcher["dx_px"]), "matcher dy (px)": px(matcher["dy_px"]),
                 "dx trend / 1000 frames": matcher["dx_trend_px_per_1000_frames"],
                 "label pairs": label["labels_measured"], "label dx (px)": px(label["dx_px"]),
                 "label dx constant along route": label["constant_along_route"],
                 "convention gap": f"{entry['label_convention_conversion']['gap_px']} px = "
                                   f"{entry['label_convention_conversion']['gap_frames_at_median_sweep']} frames"})
with pd.option_context("display.max_colwidth", 120):
    display(pd.DataFrame(rows).set_index("camera").T)
speed = pose["cameras"]["cam5"]["dense_speed_check"]
print(f"dense-flow speed against RootSIFT sweep on CAM5: ratio {speed['median_ratio_speed_over_rootsift']}, "
      f"Spearman {speed['spearman']} over {speed['pairs']} pairs")
""")

md(r"""
### 3.7 A label-free signal, dense over the whole route

CAM0 and CAM5 face opposite sides, so their alignments come from disjoint scenes
and fail near-independently. Comparing them is dense over every frame both map
(`scripts/evaluate_task2_consistency.py`). It measures consistency, not
correctness, and it ran *before any label existed*. It also judged the variants
built once the labels were spent: five input-side changes move nothing; one
change to the pooling (an unsupervised VLAD over the same masked tokens) does.
Kinks, one-step jumps the motion model forbids, belong to the refinement stage and
no descriptor change touches them (`scripts/analyze_mapping_kinks.py`).
""")

code(r"""
consistency = load("outputs/task2_consistency/consistency_summary.json")
agreement = consistency["cross_camera_agreement"]
print(consistency["signal_class"])
print(f"frames mapped by both cameras: {agreement['frames_mapped_by_both_cameras']}; "
      f"apparent constant camera offset median {agreement['apparent_camera_offset_frames']['median']} frames (not an error)")
display(pd.Series(agreement["disagreement_about_the_running_offset"], name="disagreement about the running offset").to_frame())
display(pd.DataFrame(agreement["largest_disagreement_regions"]).head(5))

if available("outputs/task2_variants/variant_comparison.json"):
    variants = load("outputs/task2_variants/variant_comparison.json")["variants"]
    rows = []
    for name, entry in variants.items():
        cross = entry["cross_camera"]
        rows.append({"variant": name, "frames both mapped": cross["frames"], "median": cross["median"], "p95": cross["p95"],
                     "over 10": cross["over_10"],
                     **{f"{camera} coverage": entry["cameras"][camera]["coverage"] for camera in CAMERAS},
                     **{f"{camera} closure": entry["cameras"][camera]["endpoint_closure"] for camera in CAMERAS}})
    print("\nvariants judged by the label-free signals (none submitted; the freeze came first):")
    display(pd.DataFrame(rows).set_index("variant"))
""")

code(r"""
kinks = load("outputs/task2_kinks/mapping_kinks.json")
bound = kinks["physical_bound"]
print(f"physical bound: {bound['rationale'][:140]}...; headline threshold K>={bound['headline_threshold']}")
rows = []
for name in ("submitted_unified", "variant:slope", "bayes_posterior", "variant:grey", "variant:sky"):
    for camera in CAMERAS:
        entry = kinks["mappings"][name]["per_camera"][camera]
        rows.append({"mapping": name, "camera": camera,
                     **{key: value["kinks"] for key, value in entry["kink_counts"].items()},
                     "max one-step increment": entry["increment_distribution"]["max"],
                     "headline kinks by status": entry.get("headline_kinks_by_status", {})})
display(pd.DataFrame(rows))

mechanism = kinks["mechanism"]
print(f"\nmechanism, CAM0 window Run A {mechanism['window_runA']}: coarse path max step "
      f"{mechanism['coarse_path']['max_one_step_increment']}, submitted mapping max step "
      f"{mechanism['submitted_mapping']['max_one_step_increment']}; route-wide refinement residual pinned at "
      f"+{mechanism['refine_radius']} on {mechanism['route_wide_residual']['frames_at_plus_refine_radius']} frames and at "
      f"-{mechanism['refine_radius']} on {mechanism['route_wide_residual']['frames_at_minus_refine_radius']}")
replay = mechanism["structure_replay"]
print(f"structure replay reproduces the submitted mapping on {replay['frames_matching_submitted']}/{replay['frames_replayed']} frames; "
      f"the monotone floor changed the pick on {replay['frames_where_the_monotone_constraint_changed_the_pick']} of them")
""")

md(r"""
### 3.8 Under-exposed frames: a systematic, not a random, error

Dark matches dark almost regardless of place. The exposure flag is measured on
every decoded frame (`scripts/build_frame_quality_flags.py`); the cross-reference
with the submitted mapping and with the blind labels shows where that bites.
""")

code(r"""
display(pd.DataFrame(quality["submitted_mapping"]).T[["accepted_rows", "runB_partner_too_dark", "either_side_too_dark",
                                                       "fraction_of_accepted"]])
for camera in CAMERAS:
    dark = quality["blind_labels"][camera]
    print(f"{camera}: {dark['labels_in_dark_regions']} blind labels in dark regions, {dark['determinable']} determinable, "
          f"{dark['correct']} correct, miss distances {dark['miss_distances_frames']}")
""")

# ----------------------------------------------------------------- section 4 #
md(r"""
## 4. Downstream policy for training or testing

### 4.1 What ships against the per-frame schema

`DATA_POLICY.md` gives the rules (split before anything else, group by route,
date, rig and both cameras, never randomise adjacent frames). What ships against
the closed schema is the quality-flag table plus the mapping CSVs' own columns.
The slope gate demotes any top-tier row within 10 Run A frames of a forbidden
one-step jump; it is advisory on CAM5, where it cost precision on the blind set.
""")

code(r"""
display(pd.read_csv(ROOT / "outputs/task3/policy_summary.csv"))
display(pd.read_csv(ROOT / "outputs/task2_keyframes/route_phase_boundaries.csv")[
    ["run", "stationary_start", "stationary_end", "transition_start", "transition_end", "route_start", "confidence"]])

flags = pd.read_csv(ROOT / "outputs/task3/frame_quality_flags.csv")
display(flags.groupby(["run", "camera"])["flag"].value_counts().unstack(fill_value=0))
print("slope-gate rows demoted:", {camera: quality["slope_gate"]["cameras"][camera]["top_tier_rows_demoted"] for camera in CAMERAS})

schema = load("outputs/task3/frame_manifest_schema.json")
fields = schema.get("properties") or schema.get("fields") or schema
print("\nschema field groups:", [key for key in fields][:40])
""")

md(r"""
### 4.2 What the loop constrains, without any labels; and the posterior model

Two Run A frames that are the same place on the way out and the way back must map
to two Run B frames that are also the same place (endpoint closure), and
cumulative image motion is a monotone route coordinate a correct mapping should
carry across (path closure). The posterior formulation (a hidden Markov model over
Run B position with an explicit null state) was built to make confidence
checkable; **none of its numbers is a held-out result** except the ones the v3
and v3b freezes named.
""")

code(r"""
if available("outputs/task2_loop_consistency/loop_consistency_v2_unified.json"):
    loop = load("outputs/task2_loop_consistency/loop_consistency_v2_unified.json")
    for camera, entry in loop["cameras"].items():
        endpoint, path = entry["endpoint_closure"], entry["path_length_closure"]
        print(f"{camera}: endpoint closure {endpoint['consistent']}/{endpoint['pairs_with_both_ends_mapped']} revisit pairs; "
              f"path closure median progress error {path['median_progress_error']:.4f}, p90 {path['p90_progress_error']:.4f}")
    controls = loop["discrimination_controls"]
    print("controls:", controls["real_mapping"], "real vs", controls["fully_random_partners"], "random\n")

if available("outputs/task2_bayes/bayes_diagnostics.json"):
    diag = load("outputs/task2_bayes/bayes_diagnostics.json")
    print(diag["evidence_status"], "\n")
    for camera in CAMERAS:
        entry = diag["with_motion"][camera]
        print(f"{camera}: strict {entry['strict_precision']}, within {diag['tolerance_frames']} frames "
              f"{entry['within_tolerance_precision']}, expected calibration error {entry['expected_calibration_error']}")

print("\nheld-out posterior readings (named in the freezes):")
for set_name, results, camera in (("blind_v3", v3, "cam0"), ("blind_v3b", v3b, "cam5")):
    stats = results["methods"]["posterior_model"]["cameras"][camera]
    print(f"  {set_name} {camera}: precision {stats['precision']} ({wilson(stats['precision_wilson_95'])}), within ±2 {stats['within_two_rate']}")
""")

md(r"""
### 4.3 The strict single-frame reading (v2 labels, tolerance zero)

The last table, as in the report: a hit only inside the annotator's interval.
*As-labelled* keeps the uncorrected denominator of §3.2. Read this after §3.1,
not instead of it.
""")

code(r"""
rescore = load("outputs/task2_evaluation/blind_v2/blind_v2_undetermined_rescore.json")
rows = []
for method, cameras in rescore["methods"].items():
    for camera, result in cameras.items():
        corrected, as_labelled = result["corrected"], result["as_labelled"]
        rows.append({"method": method, "camera": camera,
                     "accepted": f"{corrected['accepted']}/{result['labels_determinable']}",
                     "strict precision": corrected["precision"],
                     "wilson 95%": wilson(corrected["precision_wilson_95"]),
                     "coverage": f"{corrected['coverage_of_determinable']:.1%}",
                     "median frame error": result["preferred_frame_error"]["median"],
                     "as-labelled": as_labelled["precision"]})
display(pd.DataFrame(rows))
""")

md(r"""
### 4.4 The report reads these artifacts, not typed numbers

Every figure in the PDF is a macro generated by `scripts/build_report_tables.py`
from the artifacts above (`outputs/report_tables/facts.tex` and the table
fragments), so a metric change cannot silently desynchronise the PDF from the
evidence. The cell recomputes the tolerance macros and finds them in the shipped
fragment.
""")

code(r"""
import build_report_tables

facts = (ROOT / "outputs/report_tables/facts.tex").read_text()
macros = build_report_tables.tolerance_macros()
missing = [name for name, value in macros.items()
           if f"\\newcommand{{\\{name}}}{{{value}}}" not in facts]
for name in list(macros)[:8]:
    print(f"\\{name:<26} = {macros[name]}")
print(f"\n{len(macros)} tolerance macros recomputed; absent from the shipped facts.tex: {missing or 'none'}")
""")


# --------------------------------------------------------------------------- #
def build_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    for kind, text in CELLS:
        cell = nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
        notebook.cells.append(cell)
    return notebook


def execute(notebook: nbformat.NotebookNode, kernel_name: str, timeout: int) -> None:
    from nbclient import NotebookClient

    client = NotebookClient(notebook, kernel_name=kernel_name, timeout=timeout,
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute the notebook after composing it")
    parser.add_argument("--kernel", default=os.environ.get("RA_KERNEL", "python3"))
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    arguments = parser.parse_args()

    notebook = build_notebook()
    if arguments.execute:
        execute(notebook, arguments.kernel, arguments.timeout)
    nbformat.write(notebook, TARGET)
    code_cells = sum(1 for cell in notebook.cells if cell.cell_type == "code")
    print(f"{TARGET.relative_to(ROOT)}: {len(notebook.cells)} cells, {code_cells} code cells"
          + (", executed" if arguments.execute else ""))


if __name__ == "__main__":
    main()
