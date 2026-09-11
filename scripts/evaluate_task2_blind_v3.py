#!/usr/bin/env python3
"""Score the v3 targeted blind labels once, against the artifacts the v3 freeze names.

The v2 labels answered one question: how good is the submitted mapping on a
route-uniform sample. They were then read, and three claims were built on top of
that reading - that kinks predict errors, that a slope gate buys precision, that
the slope refinement is better. None of those could be tested on the labels that
suggested them. The v3 set was drawn to test them, sealed before it was opened,
and this script is its single scoring pass.

What is different from v2, and why:

* **Three label types, not two.** v2's single "no usable match" button merged
  "I cannot tell" with "there is no such place". v3 splits them, so
  `undetermined` can leave the denominator (a prediction whose truth is unknown
  is neither right nor wrong) while `no_correspondence` becomes the one thing v2
  could not supply: a real abstention target, on which an acceptance is a false
  accept and an abstention is a true reject.
* **Six methods on the same labels.** The submitted mapping, the slope
  refinement variant and the posterior model are all named by the v3 freeze, so
  all three earn held-out numbers. The three gate readings - slope gate, dark
  gate, both - are policies over the frozen submitted mapping and change no
  mapping row; they are scored to measure what the gates would have bought,
  which DATA_POLICY.md could previously only cost against spent labels.
* **A stratified read, with its dependency stated.** The strata were located
  using the submitted mapping's own structure, so per-stratum rates are
  conditional on where that mapping put its kinks. The corridor stratum is the
  untargeted control and the only stratum whose rate is unbiased route-wide.
  The manifest's disclosure is copied verbatim into the output.

Cameras are reported separately and never pooled for a conclusion: they observe
the same two traversals and are not independent samples. Pooled rows appear only
where readability demands one, carrying that caveat.

`--set-name blind_v3b` scores the CAM5 redo. The CAM5 half of v3 was annotated
with a viewer that applied the pose compensation with the wrong sign (see
`outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`), so CAM5 was asked again
on a fresh query set under `method_freeze_v3b.json`, which hashes exactly the
same artifacts. Nothing about the methods or the metrics changes; the set name
decides which manifest, freeze, seal and label file are read and where the
results are written. Two extra readings are computed for the redo and are
reported for any set: a **signed** error (is the prediction later or earlier than
the label interval, not just how far), which is how the v3 tool defect showed
itself as a systematic +3 to +4 on CAM5, and a comparison against the CAM5 v2
numbers so the redo can be read next to the route-uniform set it succeeds.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import freeze_method_v2
import freeze_method_v3
from evaluate_task2_blind_v2 import distance_to_interval, wilson_interval


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET_NAME = "blind_v3"
BLIND_DIR = ROOT / "outputs" / "task2_evaluation" / "blind_v3"
MANIFEST_FILE = ROOT / "outputs" / "task2_blind_v3" / "blind_v3_manifest.json"
FREEZE_FILE = ROOT / "outputs" / "task2_evaluation" / "method_freeze_v3.json"
QUALITY_FLAGS = ROOT / "outputs" / "task3" / "frame_quality_flags.csv"
KINK_AUDIT = ROOT / "outputs" / "task2_kinks" / "mapping_kinks.json"
POSE_DIAGNOSTIC = ROOT / "outputs" / "task2_pose_offset" / "pose_offset_diagnostic.json"
V2_CASES = ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "blind_v2_cases.csv"
V3_LABELS = ROOT / "outputs" / "task2_evaluation" / "blind_v3" / "blind_v3_labels.csv"

CAMERAS = ("cam0", "cam5")
STRATA = ("kink", "dark", "occlusion", "corridor")


def blind_dir(set_name: str = DEFAULT_SET_NAME) -> Path:
    return ROOT / "outputs" / "task2_evaluation" / set_name


def manifest_file(set_name: str = DEFAULT_SET_NAME) -> Path:
    return ROOT / "outputs" / f"task2_{set_name}" / f"{set_name}_manifest.json"


def cameras_of(manifest: dict) -> tuple[str, ...]:
    """The cameras this query set actually exported, in the canonical order.

    v3 asked both cameras; the v3b redo asks CAM5 only, because CAM0's v3 labels
    are unaffected by the tool defect and stand. Reading the camera list from the
    manifest rather than from a constant is what keeps a single-camera set from
    silently scoring an absent one.
    """

    return tuple(camera for camera in CAMERAS if camera in manifest["cameras"])

# The annotator's viewer draws 960x720 JPEGs; the pose diagnostic measured on
# 896x672. The constant is asserted against the manifest's own scaled default
# below rather than trusted, so a raster change cannot pass silently.
FILMSTRIP_RASTER = (960, 720)

MAPPINGS = {
    "submitted_unified": "outputs/task2_unified/{camera}/frame_mapping_unified.csv",
    "slope_refinement_variant": (
        "outputs/task2_unified_variants/slope/{camera}/frame_mapping_unified.csv"
    ),
    "posterior_model": "outputs/task2_bayes/{camera}/frame_mapping_bayes.csv",
}

METHODS: dict[str, dict[str, object]] = {
    "submitted_unified": {
        "mapping": "submitted_unified",
        "gates": (),
        "description": "the submitted unified mapping, unmodified",
        "held_out": True,
    },
    "slope_refinement_variant": {
        "mapping": "slope_refinement_variant",
        "gates": (),
        "description": (
            "build_task2_unified.py --refinement slope: Viterbi refinement with a "
            "slope prior, built before the v3 freeze and named by it"
        ),
        "held_out": True,
    },
    "posterior_model": {
        "mapping": "posterior_model",
        "gates": (),
        "description": "the Bayes HMM posterior mapping, with an explicit null state",
        "held_out": True,
    },
    "submitted_plus_slope_gate": {
        "mapping": "submitted_unified",
        "gates": ("slope",),
        "description": (
            "submitted mapping with the frozen slope gate: a Run A row flagged "
            "near_kink in frame_quality_flags.csv (increment >= 6 within +-10 Run "
            "A frames) is turned into an abstention"
        ),
        "held_out": True,
    },
    "submitted_plus_dark_gate": {
        "mapping": "submitted_unified",
        "gates": ("dark",),
        "description": (
            "Task 3 reading: submitted mapping with DATA_POLICY.md's exposure "
            "rule as an abstention - a row whose Run A frame or whose predicted "
            "Run B frame is flagged too_dark is turned into an abstention"
        ),
        "held_out": True,
    },
    "submitted_plus_both_gates": {
        "mapping": "submitted_unified",
        "gates": ("slope", "dark"),
        "description": "submitted mapping with both gates applied (union of the two)",
        "held_out": True,
    },
}

MATCH = "match"
UNDETERMINED = "undetermined"
NO_CORRESPONDENCE = "no_correspondence"


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


def check_contract(set_name: str = DEFAULT_SET_NAME) -> dict[str, object]:
    """Refuse to score unless both freezes verify and the seal matches the labels.

    The v3 number means nothing without the ordering export -> freeze -> label ->
    seal -> score, so every link is rechecked here rather than assumed. The redo
    set is held to the identical contract against its own freeze; a seal that
    cites a different freeze than the one on disk stops the run.
    """

    tag = freeze_method_v3.version_tag(set_name)
    freeze_method_v2.verify()
    frozen_v3 = freeze_method_v3.verify(set_name=set_name)

    seal_path = blind_dir(set_name) / "label_seal.json"
    if not seal_path.is_file():
        raise SystemExit(
            f"No sealed {tag} label set. Import the annotator's CSV with "
            "scripts/import_blind_labels_v3.py first."
        )
    seal = json.loads(seal_path.read_text())
    if seal.get("label_set") != set_name:
        raise SystemExit(
            f"The seal at {seal_path} is for {seal.get('label_set')!r}, not {set_name!r}"
        )
    label_file = ROOT / seal["label_file"]
    digest = hashlib.sha256(label_file.read_bytes()).hexdigest()
    if digest != seal["label_file_sha256"]:
        raise SystemExit(
            f"The sealed {tag} label file has changed since it was sealed. The "
            "blind evaluation cannot proceed against edited labels."
        )
    # v3's seal names the freeze under `method_freeze_v3_sha256`; the redo seal
    # names it under `method_freeze_sha256_at_seal_time`. Either spelling is
    # accepted, but one of them must be present and must match.
    cited = seal.get("method_freeze_v3_sha256") or seal.get(
        "method_freeze_sha256_at_seal_time"
    )
    if cited != frozen_v3["freeze_sha256"]:
        raise SystemExit(
            f"The seal does not cite the {tag} freeze on disk; the ordering that "
            "makes this a held-out number is not established."
        )
    manifest = json.loads(manifest_file(set_name).read_text())
    if manifest["query_set_sha256"] != frozen_v3["query_set_sha256"]:
        raise SystemExit(f"The query manifest is not the one the {tag} freeze hashed")
    return {"freeze": frozen_v3, "seal": seal, "label_file": label_file, "manifest": manifest}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def load_mapping(name: str, camera: str) -> dict[int, str]:
    """Run A frame -> predicted Run B frame as a string ('' means abstained)."""

    path = ROOT / MAPPINGS[name].format(camera=camera)
    with path.open(newline="") as handle:
        return {
            int(row["runA_frame"]): row["runB_frame"].strip()
            for row in csv.DictReader(handle)
        }


def load_quality_flags() -> tuple[dict[tuple[str, str, int], str], dict[tuple[str, str, int], str]]:
    """(run, camera, ordinal) -> exposure flag, and the same key -> slope gate."""

    exposure: dict[tuple[str, str, int], str] = {}
    slope: dict[tuple[str, str, int], str] = {}
    with QUALITY_FLAGS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["run"], row["camera"], int(row["ordinal"]))
            exposure[key] = row["flag"]
            slope[key] = row.get("slope_gate", "")
    return exposure, slope


def load_strata(set_name: str = DEFAULT_SET_NAME) -> dict[str, str]:
    """query_id -> stratum, from the manifest that the freeze hashed."""

    manifest = json.loads(manifest_file(set_name).read_text())
    return {
        query["queryId"]: query["stratum"]
        for camera in cameras_of(manifest)
        for query in manifest["cameras"][camera]["queries"]
    }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def gate_blocks(
    gates: tuple[str, ...],
    camera: str,
    runa_frame: int,
    predicted: str,
    exposure: dict[tuple[str, str, int], str],
    slope: dict[tuple[str, str, int], str],
) -> str:
    """Return the gate that demotes this row to an abstention, or ''.

    The slope gate is the frozen rule read straight out of the `slope_gate`
    column. The dark gate is DATA_POLICY.md's exposure rule applied as an
    abstention rather than as a tier cap: either side of the pair being
    `too_dark` is enough, because the failure mechanism is dark matching dark.
    """

    if "slope" in gates and slope.get(("runA", camera, runa_frame), "") == "near_kink":
        return "near_kink"
    if "dark" in gates:
        if exposure.get(("runA", camera, runa_frame), "") == "too_dark":
            return "runA_too_dark"
        if predicted and exposure.get(("runB", camera, int(predicted)), "") == "too_dark":
            return "runB_too_dark"
    return ""


def distribution(values: list[int]) -> dict[str, object]:
    if not values:
        return {"n": 0}
    array = np.array(values)
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": int(array.max()),
        "within_two_frames": int((array <= 2).sum()),
    }


def signed_distance_to_interval(prediction: int, low: int, high: int) -> int:
    """How far outside the label interval the prediction is, keeping the sign.

    Positive means the prediction names a Run B frame LATER than the annotator
    would accept, negative EARLIER, zero inside. `distance_to_interval` throws
    the sign away, which is exactly the information that exposed the v3 CAM5
    annotator-tool defect: a random matcher error is symmetric, a tool that
    displaces the Run B raster in one direction is not.
    """

    if prediction > high:
        return prediction - high
    if prediction < low:
        return prediction - low
    return 0


def build_cases(
    method: str,
    labels: list[dict[str, str]],
    strata: dict[str, str],
    exposure: dict[tuple[str, str, int], str],
    slope: dict[tuple[str, str, int], str],
    cameras: tuple[str, ...] = CAMERAS,
) -> list[dict[str, object]]:
    """One row per label, carrying the method's decision and its scoring."""

    spec = METHODS[method]
    mappings = {camera: load_mapping(str(spec["mapping"]), camera) for camera in cameras}
    cases: list[dict[str, object]] = []
    for label in labels:
        camera = label["camera_id"]
        frame = int(label["runA_frame"])
        mapping = mappings[camera]
        if frame not in mapping:
            raise ValueError(f"{camera}: labelled Run A frame {frame} absent from {method}")
        raw = mapping[frame]
        gate = gate_blocks(
            tuple(spec["gates"]), camera, frame, raw, exposure, slope  # type: ignore[arg-type]
        )
        predicted = "" if gate else raw
        case: dict[str, object] = {
            "method": method,
            "query_id": label["query_id"],
            "camera_id": camera,
            "runA_frame": frame,
            "stratum": strata[label["query_id"]],
            "ground_truth_label": label["label"],
            "runB_min": label["runB_min"],
            "runB_max": label["runB_max"],
            "raw_predicted_runB_frame": raw,
            "gated_by": gate,
            "accepted": bool(predicted),
            "predicted_runB_frame": predicted,
            "correct": "",
            "interval_distance": "",
            "preferred_frame_error": "",
            "signed_interval_error": "",
            "signed_preferred_error": "",
        }
        if predicted and label["label"] == MATCH:
            prediction = int(predicted)
            low, high = int(label["runB_min"]), int(label["runB_max"])
            gap = distance_to_interval(prediction, low, high)
            case["correct"] = gap == 0
            case["interval_distance"] = gap
            case["preferred_frame_error"] = abs(prediction - int(label["runB_frame"]))
            case["signed_interval_error"] = signed_distance_to_interval(
                prediction, low, high
            )
            case["signed_preferred_error"] = prediction - int(label["runB_frame"])
        cases.append(case)
    return cases


def score(cases: list[dict[str, object]]) -> dict[str, object]:
    """Metrics for one method over one slice of labels."""

    matches = [case for case in cases if case["ground_truth_label"] == MATCH]
    undetermined = [case for case in cases if case["ground_truth_label"] == UNDETERMINED]
    absent = [case for case in cases if case["ground_truth_label"] == NO_CORRESPONDENCE]

    accepted = [case for case in matches if case["accepted"]]
    correct = sum(1 for case in accepted if case["correct"])
    gaps = [int(case["interval_distance"]) for case in accepted]
    errors = [int(case["preferred_frame_error"]) for case in accepted]
    precision = correct / len(accepted) if accepted else None
    within_two = sum(1 for gap in gaps if gap <= 2)

    undetermined_accepted = sum(1 for case in undetermined if case["accepted"])
    false_accepts = [case["query_id"] for case in absent if case["accepted"]]
    true_rejects = [case["query_id"] for case in absent if not case["accepted"]]

    # Secondary, deliberately pessimistic reading: every acceptance on a label
    # that is not a confirmed match counted as an error. The seal's semantics say
    # undetermined rows are NOT negatives, so this is a lower bound and not the
    # headline. It is reported because v2 reported both readings.
    lower_denominator = len(accepted) + undetermined_accepted + len(false_accepts)

    return {
        "labels": len(cases),
        "match_labels": len(matches),
        "undetermined_labels": len(undetermined),
        "no_correspondence_labels": len(absent),
        "accepted": len(accepted),
        "correct": correct,
        "precision": round(precision, 4) if precision is not None else None,
        "precision_wilson_95": wilson_interval(correct, len(accepted)),
        "within_two": within_two,
        "within_two_rate": (
            round(within_two / len(accepted), 4) if accepted else None
        ),
        "coverage_of_match_labels": (
            round(len(accepted) / len(matches), 4) if matches else None
        ),
        "abstained_on_match_labels": len(matches) - len(accepted),
        "gated_match_labels": sum(1 for case in matches if case["gated_by"]),
        "interval_distance": distribution(gaps),
        "preferred_frame_error": distribution(errors),
        "undetermined": {
            "labels": len(undetermined),
            "accepted": undetermined_accepted,
            "abstained": len(undetermined) - undetermined_accepted,
            "note": (
                "reported as counts only; an undetermined label is not a negative "
                "and cannot make an acceptance right or wrong"
            ),
        },
        "no_correspondence": {
            "labels": len(absent),
            "false_accepts": len(false_accepts),
            "false_accept_query_ids": false_accepts,
            "true_rejects": len(true_rejects),
            "true_reject_query_ids": true_rejects,
        },
        "lower_bound_reading": {
            "accepted": lower_denominator,
            "correct": correct,
            "precision": (
                round(correct / lower_denominator, 4) if lower_denominator else None
            ),
            "note": (
                "every acceptance on a non-match label charged as an error. The "
                "seal says undetermined rows are not negatives, so this is a "
                "lower bound, not the reported precision."
            ),
        },
    }


def miss_rate(cases: list[dict[str, object]]) -> dict[str, object]:
    """Miss rate on accepted match labels, the v2 kink claim's own statistic."""

    matches = [case for case in cases if case["ground_truth_label"] == MATCH]
    accepted = [case for case in matches if case["accepted"]]
    misses = [case for case in accepted if int(case["interval_distance"]) > 0]
    gaps = [int(case["interval_distance"]) for case in accepted]
    return {
        "match_labels": len(matches),
        "accepted": len(accepted),
        "abstained": len(matches) - len(accepted),
        "misses_interval_distance_gt_0": len(misses),
        "miss_rate": round(len(misses) / len(accepted), 4) if accepted else None,
        "miss_rate_wilson_95": wilson_interval(len(misses), len(accepted)),
        "median_interval_distance": float(np.median(gaps)) if gaps else None,
        "max_interval_distance": int(max(gaps)) if gaps else None,
    }


def signed_error_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    """Direction of the error on accepted match labels, not just its size.

    Reported because an unsigned median cannot tell a matcher that is imprecise
    from a viewer that shows the wrong picture. The v3 CAM5 labels were taken
    against a raster the tool had displaced, and the submitted mapping's error
    against them came out systematically positive; a set annotated with the
    corrected tool should be centred near zero, with later and earlier errors in
    roughly equal number.
    """

    accepted = [
        case
        for case in cases
        if case["ground_truth_label"] == MATCH
        and case["accepted"]
        and case.get("signed_interval_error") != ""
        and case.get("signed_interval_error") is not None
    ]
    if not accepted:
        return {"n": 0}
    interval = np.array([int(case["signed_interval_error"]) for case in accepted])
    preferred = np.array([int(case["signed_preferred_error"]) for case in accepted])
    outside = interval[interval != 0]
    return {
        "n": int(interval.size),
        "definition": (
            "signed_interval_error: prediction minus the nearer end of the "
            "annotator's acceptable interval, 0 when inside. Positive means the "
            "prediction names a LATER Run B frame than the label allows, "
            "negative an EARLIER one."
        ),
        "signed_interval_error": {
            "median": float(np.median(interval)),
            "mean": round(float(interval.mean()), 4),
            "later_than_interval": int((interval > 0).sum()),
            "earlier_than_interval": int((interval < 0).sum()),
            "inside_interval": int((interval == 0).sum()),
            "min": int(interval.min()),
            "max": int(interval.max()),
            "median_of_the_ones_outside": (
                float(np.median(outside)) if outside.size else None
            ),
        },
        "signed_preferred_frame_error": {
            "median": float(np.median(preferred)),
            "mean": round(float(preferred.mean()), 4),
            "later_than_preferred": int((preferred > 0).sum()),
            "earlier_than_preferred": int((preferred < 0).sum()),
            "equal_to_preferred": int((preferred == 0).sum()),
            "min": int(preferred.min()),
            "max": int(preferred.max()),
        },
    }


def v3_cam5_signed_reference() -> dict[str, object]:
    """The same signed reading on the withdrawn v3 CAM5 labels, for comparison.

    Read-only: the v3 label file is opened, never written, and no v3 result is
    rescored. Its purpose is to give the redo something to be measured against,
    because "the systematic offset is gone" is only a claim if the number it was
    is on the page next to the number it now is.
    """

    if not V3_LABELS.is_file():
        return {"available": False}
    mapping = load_mapping("submitted_unified", "cam5")
    signed: list[int] = []
    preferred: list[int] = []
    with V3_LABELS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["camera_id"] != "cam5" or row["label"] != MATCH:
                continue
            raw = mapping[int(row["runA_frame"])].strip()
            if not raw:
                continue
            prediction = int(raw)
            signed.append(
                signed_distance_to_interval(
                    prediction, int(row["runB_min"]), int(row["runB_max"])
                )
            )
            preferred.append(prediction - int(row["runB_frame"]))
    array = np.array(signed)
    pref = np.array(preferred)
    return {
        "available": True,
        "source": "outputs/task2_evaluation/blind_v3/blind_v3_labels.csv, CAM5 match labels",
        "method": "submitted_unified",
        "n": int(array.size),
        "signed_interval_error_median": float(np.median(array)),
        "signed_interval_error_mean": round(float(array.mean()), 4),
        "later_than_interval": int((array > 0).sum()),
        "earlier_than_interval": int((array < 0).sum()),
        "inside_interval": int((array == 0).sum()),
        "signed_preferred_frame_error_median": float(np.median(pref)),
        "later_than_preferred": int((pref > 0).sum()),
        "earlier_than_preferred": int((pref < 0).sum()),
        "why_it_is_here": (
            "The v3 CAM5 viewer moved the Run B raster the wrong way, and the "
            "annotator recovered with the slider; the resulting labels sat "
            "systematically earlier than the mapping's predictions. This is the "
            "defect's signature, quoted so the redo can be checked against it. "
            "See outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md."
        ),
    }


def v2_cam5_comparison(cases: list[dict[str, object]]) -> dict[str, object]:
    """CAM5 on the route-uniform v2 set against CAM5 on this set.

    v2 is recomputed here from its own sealed case file rather than copied from
    a summary, so the comparison cannot drift from the artifact. The two sets are
    not interchangeable and the block says so: v2 is route-uniform over 37 match
    labels, this one is stratified and its corridor stratum is the part that is
    comparable.
    """

    if not V2_CASES.is_file():
        return {"available": False}
    with V2_CASES.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["camera_id"] == "cam5"]
    matches = [row for row in rows if row["ground_truth_label"] == MATCH]
    accepted = [row for row in matches if row["accepted"] == "True"]
    correct = sum(1 for row in accepted if row["accepted_correct"] == "True")
    within_two_interval = sum(1 for row in accepted if int(row["interval_distance"]) <= 2)
    within_two_preferred = sum(
        1 for row in accepted if int(row["preferred_frame_error"]) <= 2
    )
    v2 = {
        "source": "outputs/task2_evaluation/blind_v2/blind_v2_cases.csv",
        "sampling": "route-uniform; no stratum targeting",
        "match_labels": len(matches),
        "accepted": len(accepted),
        "correct": correct,
        "precision": round(correct / len(accepted), 4) if accepted else None,
        "precision_wilson_95": wilson_interval(correct, len(accepted)),
        "within_two_by_interval_distance": within_two_interval,
        "within_two_by_preferred_frame_error": within_two_preferred,
        "note": (
            "The headline v2 CAM5 pair quoted elsewhere is 0.375 strict on the "
            "32 accepted match labels with 21/32 within two frames of the "
            "preferred frame. The interval-distance reading of within-2 is "
            f"{within_two_interval}/{len(accepted)}; this evaluation's "
            "`within_two` column is the interval-distance one, so both are given."
        ),
    }
    cam5 = [case for case in cases if case["camera_id"] == "cam5"]
    corridor = score([case for case in cam5 if case["stratum"] == "corridor"])
    overall = score(cam5)

    def brief(result: dict[str, object]) -> dict[str, object]:
        return {
            "match_labels": result["match_labels"],
            "accepted": result["accepted"],
            "correct": result["correct"],
            "precision": result["precision"],
            "precision_wilson_95": result["precision_wilson_95"],
            "within_two_by_interval_distance": result["within_two"],
            "coverage_of_match_labels": result["coverage_of_match_labels"],
        }

    def delta(result: dict[str, object]) -> float | None:
        if result["precision"] is None or v2["precision"] is None:
            return None
        return round(float(result["precision"]) - float(v2["precision"]), 4)

    return {
        "available": True,
        "method": "submitted_unified",
        "v2": v2,
        "this_set_corridor_only": brief(corridor),
        "this_set_overall": brief(overall),
        "precision_minus_v2": {
            "corridor_only": delta(corridor),
            "overall": delta(overall),
        },
        "comparability": (
            "The corridor stratum is the like-for-like comparison: it is the "
            "untargeted control, sampled uniformly over the route the way the "
            "whole v2 set was. The overall figure mixes in kink and dark "
            "queries deliberately drawn where the method is least plausible, so "
            "it is expected to sit below the corridor number and is not a "
            "route-wide rate. Both Wilson intervals are wide at these counts; a "
            "difference that does not clear them is not a difference."
        ),
    }


# --------------------------------------------------------------------------- #
# convention-shift sensitivity (a reading over an already-scored set)
# --------------------------------------------------------------------------- #


SENSITIVITY_FILENAME = "convention_shift_sensitivity.json"
SENSITIVITY_METHODS = (
    "submitted_unified",
    "slope_refinement_variant",
    "posterior_model",
)


def convention_constant(camera: str) -> dict[str, object]:
    """The camera-to-world convention offset, READ from the pose diagnostic.

    The constant is never typed here. It is the gap between the matcher's
    "same place" (image content aligns) and the label convention (the world
    position a far+near rule names), measured on the v2 labels and written to
    `pose_offset_diagnostic.json` before the v3b labels existed. Reading it
    rather than restating it is what keeps this a sensitivity check against a
    pre-existing number instead of a constant fitted to the set it is tested on.
    """

    payload = json.loads(POSE_DIAGNOSTIC.read_text())
    block = payload["cameras"][camera]["label_convention_conversion"]
    frames = float(block["gap_frames_at_median_sweep"])
    try:
        source = str(POSE_DIAGNOSTIC.relative_to(ROOT))
    except ValueError:  # a test pointing the constant at a fixture
        source = str(POSE_DIAGNOSTIC)
    return {
        "camera": camera,
        "source_file": source,
        "source_built_at_utc": payload["built_at_utc"],
        "source_field": (
            f"cameras.{camera}.label_convention_conversion.gap_frames_at_median_sweep"
        ),
        "gap_px": block.get("gap_px"),
        "gap_frames_at_median_sweep": frames,
        "rounded_frames": int(round(frames)),
        "measured_on": (
            "the v2 blind labels, before this label set was drawn; see the "
            "diagnostic's own degeneracy note"
        ),
        "degeneracy_note": payload["degeneracy"],
    }


def load_scored_cases(set_name: str) -> list[dict[str, object]]:
    """Re-read the per-label rows the single scoring pass already wrote.

    Nothing is rescored and no mapping is re-read: the shift is applied to the
    predictions that pass recorded, so this cannot become a second scoring of
    the sealed set by accident.
    """

    path = blind_dir(set_name) / f"{set_name}_cases.csv"
    if not path.is_file():
        raise SystemExit(
            f"No scored cases at {path}. Score the set once first with "
            f"scripts/evaluate_task2_blind_v3.py --set-name {set_name}."
        )
    cases: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            case: dict[str, object] = dict(row)
            case["runA_frame"] = int(row["runA_frame"])
            case["accepted"] = row["accepted"] == "True"
            case["correct"] = (row["correct"] == "True") if row["correct"] else ""
            for key in (
                "interval_distance",
                "preferred_frame_error",
                "signed_interval_error",
                "signed_preferred_error",
            ):
                case[key] = int(row[key]) if row[key] != "" else ""
            cases.append(case)
    return cases


def shift_cases(
    cases: list[dict[str, object]],
    camera: str,
    shift: int,
    preferred: dict[str, int],
) -> list[dict[str, object]]:
    """Add `shift` frames to every accepted match prediction on one camera.

    Only accepted match rows move: an abstention has no frame to shift, and a
    constant offset cannot turn a rejection into an acceptance. That is what
    makes this readable as a convention conversion rather than a new method -
    coverage, the gates and the no-correspondence rows are untouched.
    """

    shifted: list[dict[str, object]] = []
    for case in cases:
        moved = dict(case)
        if (
            case["camera_id"] == camera
            and case["accepted"]
            and case["ground_truth_label"] == MATCH
        ):
            prediction = int(str(case["predicted_runB_frame"])) + shift
            low, high = int(str(case["runB_min"])), int(str(case["runB_max"]))
            moved["predicted_runB_frame"] = str(prediction)
            moved["interval_distance"] = distance_to_interval(prediction, low, high)
            moved["correct"] = moved["interval_distance"] == 0
            moved["signed_interval_error"] = signed_distance_to_interval(
                prediction, low, high
            )
            if case["query_id"] in preferred:
                target = preferred[str(case["query_id"])]
                moved["preferred_frame_error"] = abs(prediction - target)
                moved["signed_preferred_error"] = prediction - target
        shifted.append(moved)
    return shifted


CONTROL_SETS = (
    {
        "label_set": "blind_v2",
        "camera": "cam5",
        "method": "submitted_unified",
        "role": "mixed_convention_set",
        "why": (
            "The v2 CAM5 labels are the set the constant was measured ON, and "
            "they are also the set whose labelling criterion was later found to "
            "be mixed: the annotator used the marker's own direction for some "
            "queries and the image layout for others (FAILURE_ANALYSIS.md; "
            "scripts/diagnose_label_criterion.py). If the deficit were one "
            "convention constant everywhere, the same shift would pay here too. "
            "A mixture predicts that it does not."
        ),
    },
    {
        "label_set": "blind_v3",
        "camera": "cam0",
        "method": "submitted_unified",
        "role": "zero_offset_control",
        "why": (
            "CAM0's measured pose offset is 0/0 px, so its convention gap is "
            "zero and its sweep must peak at shift 0. It is the negative "
            "control: a sweep that peaked away from 0 here would say the shift "
            "is buying something other than a convention conversion."
        ),
    },
)


def _control_case_rows(
    label_set: str, camera: str, method: str
) -> tuple[list[dict[str, int]], Path]:
    """Accepted match rows of one camera from an already-scored set's cases file.

    Reads the two blind cases schemas without rescoring anything. v2 wrote one
    method per file and named the interval columns `ground_truth_runB_*`; v3 and
    v3b write one row per method and name them `runB_*`. Undetermined and
    no-correspondence rows are excluded by the `match` filter: a shift cannot
    make a prediction whose truth is unknown right or wrong.
    """

    path = blind_dir(label_set) / f"{label_set}_cases.csv"
    if not path.is_file():
        raise SystemExit(f"No scored cases at {path}; score {label_set} once first.")
    rows: list[dict[str, int]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "method" in row and row["method"] != method:
                continue
            if row["camera_id"] != camera:
                continue
            if row["ground_truth_label"] != MATCH or row["accepted"] != "True":
                continue
            low_key = "runB_min" if "runB_min" in row else "ground_truth_runB_min"
            high_key = "runB_max" if "runB_max" in row else "ground_truth_runB_max"
            rows.append(
                {
                    "prediction": int(row["predicted_runB_frame"]),
                    "low": int(row[low_key]),
                    "high": int(row[high_key]),
                }
            )
    return rows, path


def control_sweep(
    label_set: str, camera: str, method: str, shifts: tuple[int, ...]
) -> dict[str, object]:
    """The same constant-shift sweep, on a set this sensitivity did not come from."""

    rows, path = _control_case_rows(label_set, camera, method)
    by_shift: dict[str, object] = {}
    for shift in shifts:
        correct = later = earlier = inside = 0
        for row in rows:
            prediction = row["prediction"] + shift
            signed = signed_distance_to_interval(prediction, row["low"], row["high"])
            if signed == 0:
                inside += 1
                correct += 1
            elif signed > 0:
                later += 1
            else:
                earlier += 1
        by_shift[str(shift)] = {
            "shift_frames": shift,
            "accepted": len(rows),
            "correct": correct,
            "precision": round(correct / len(rows), 4) if rows else None,
            "precision_wilson_95": wilson_interval(correct, len(rows)),
            "signed_interval_counts": {
                "later_than_interval": later,
                "earlier_than_interval": earlier,
                "inside_interval": inside,
            },
            "signed_late_early_inside": f"{later}:{earlier}:{inside}",
        }
    scored = {
        shift: by_shift[str(shift)]["precision"]
        for shift in shifts
        if by_shift[str(shift)]["precision"] is not None
    }
    peak = max(scored, key=lambda key: scored[key]) if scored else None
    return {
        "label_set": label_set,
        "camera": camera,
        "method": method,
        "cases_file": str(path.relative_to(ROOT)),
        "cases_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "accepted_match_rows": len(rows),
        "shifts_frames": list(shifts),
        "by_shift": by_shift,
        "peak_shift_frames": peak,
    }


def control_sweeps(shifts: tuple[int, ...]) -> dict[str, object]:
    """Run the sweep on the two sets that decide what the constant means.

    The v3b sweep alone cannot distinguish "one convention constant" from "a
    shift that happens to help this set". Two other already-scored sets can:
    the set the constant was measured on, and a camera whose offset is zero.
    """

    sets: dict[str, object] = {}
    for spec in CONTROL_SETS:
        block = control_sweep(
            str(spec["label_set"]), str(spec["camera"]), str(spec["method"]), shifts
        )
        block["role"] = spec["role"]
        block["why"] = spec["why"]
        sets[f"{spec['label_set']}:{spec['camera']}"] = block
    return {
        "purpose": (
            "The same constant frame shift, applied to two sets this reading was "
            "not derived from: the mixed-convention set the constant was measured "
            "on, and the zero-offset camera."
        ),
        "how_to_read": (
            "A convention conversion should pay on a consistently-labelled "
            "non-zero-offset camera, be NEUTRAL on a set whose labels mix two "
            "conventions (the two halves move in opposite directions and cancel), "
            "and peak at 0 on a camera whose measured offset is 0."
        ),
        "sets": sets,
    }


def convention_shift_sensitivity(set_name: str) -> dict[str, object]:
    """Rescore the already-scored set under a constant per-camera frame shift.

    This is a **sensitivity reading, not a held-out number**. The set was scored
    once, against the submitted mapping as it stands; this asks a different and
    narrower question of the same rows - how much of the CAM5 error is a
    constant that was already measured elsewhere, before these labels existed -
    and it answers it by reading that constant from its own artifact.

    It writes its own file. `<set>_results.json` is not rewritten, because the
    headline numbers must stay exactly what the single pass produced.
    """

    contract = check_contract(set_name)
    manifest = contract["manifest"]
    cameras = cameras_of(manifest)
    with Path(str(contract["label_file"])).open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    preferred = {
        row["query_id"]: int(row["runB_frame"])
        for row in labels
        if row["label"] == MATCH and row["runB_frame"] != ""
    }
    cases = load_scored_cases(set_name)

    results_file = blind_dir(set_name) / f"{set_name}_results.json"
    scored = json.loads(results_file.read_text())

    per_camera: dict[str, object] = {}
    control_shifts: tuple[int, ...] = (0, -1, -2, -3)
    for camera in cameras:
        constant = convention_constant(camera)
        measured = int(constant["rounded_frames"])
        shifts = sorted({0, -1, measured, -3}, reverse=True)
        control_shifts = tuple(shifts)
        methods: dict[str, object] = {}
        for method in SENSITIVITY_METHODS:
            selected = [case for case in cases if case["method"] == method]
            if not selected:
                continue
            by_shift: dict[str, object] = {}
            for shift in shifts:
                moved = shift_cases(selected, camera, shift, preferred)
                camera_rows = [
                    case for case in moved if case["camera_id"] == camera
                ]
                result = score(camera_rows)
                by_shift[str(shift)] = {
                    "shift_frames": shift,
                    "accepted": result["accepted"],
                    "correct": result["correct"],
                    "precision": result["precision"],
                    "precision_wilson_95": result["precision_wilson_95"],
                    "within_two": result["within_two"],
                    "within_two_rate": result["within_two_rate"],
                    "median_interval_distance": result["interval_distance"].get("median"),
                    "signed_interval_error": signed_error_summary(camera_rows).get(
                        "signed_interval_error"
                    ),
                }
            scored_precisions = {
                shift: by_shift[str(shift)]["precision"]
                for shift in shifts
                if by_shift[str(shift)]["precision"] is not None
            }
            peak = (
                max(scored_precisions, key=lambda key: scored_precisions[key])
                if scored_precisions
                else None
            )
            methods[method] = {
                "by_shift": by_shift,
                "peak_shift_frames": peak,
                "peak_is_the_measured_constant": peak == measured,
            }
        per_camera[camera] = {
            "constant": constant,
            "shifts_frames": shifts,
            "measured_shift_frames": measured,
            "methods": methods,
        }

    return {
        "schema_version": 1,
        "block_name": "convention_shift_sensitivity",
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_set": set_name,
        "version_tag": freeze_method_v3.version_tag(set_name),
        "cameras": list(cameras),
        "question": (
            "How much of the measured error is one constant that converts the "
            "matcher's 'same place' to the world position the labels name?"
        ),
        "evidence_status": (
            "A SENSITIVITY READING, NOT A HELD-OUT NUMBER. The set was scored "
            "once and those numbers stand unchanged in "
            f"{set_name}_results.json. This rescores the same rows under a "
            "constant frame shift whose value was measured on the v2 labels and "
            "written to the pose diagnostic before this set was drawn, so the "
            "constant is not fitted here - but the set has now been read, and "
            "no further claim may be built on it."
        ),
        "why_a_separate_file": (
            f"{set_name}_results.json records a single scoring pass and is not "
            "rewritten. A reading taken after that pass belongs beside it, not "
            "inside it."
        ),
        "the_submission_is_not_shifted": (
            "The submitted mapping files are unchanged. The evaluation contract "
            "froze them before the labels were sealed, and a constant chosen "
            "after reading a score is not a frozen method. The constant is "
            "attached as per-frame metadata instead (DATA_POLICY.md), so a "
            "consumer that wants world position can apply it and one that wants "
            "the matcher's convention can ignore it."
        ),
        "shift_semantics": (
            "A shift of -2 means every accepted prediction names a Run B frame "
            "two earlier. Only accepted match rows move; coverage, the gates and "
            "the no-correspondence rows are untouched."
        ),
        "scored_pass": {
            "file": str(results_file.relative_to(ROOT)),
            "sha256": hashlib.sha256(results_file.read_bytes()).hexdigest(),
            "evaluated_at_utc": scored["evaluated_at_utc"],
            "label_seal_sha256": scored["label_seal_sha256"],
        },
        "convention_shift_sensitivity": per_camera,
        "control_sweeps": control_sweeps(control_shifts),
    }


def pose_check(
    labels: list[dict[str, str]], manifest: dict, cameras: tuple[str, ...] = CAMERAS
) -> dict[str, object]:
    """Compare the compensation the annotator settled on against the measurement.

    The viewer seeded CAM5 with the matcher-convention shift scaled to the
    filmstrip raster, and wrote whatever the annotator left in the sliders into
    every label. Those recorded values are therefore a check on the convention:
    if the seeded default were right, the annotator would have had no reason to
    move it far, and certainly not across zero.

    The convention is `compensation = -displacement`. v3 shipped the raw measured
    displacement as the compensation, so its seeded default carried the wrong
    sign; v3b ships the negation, which is correct. Both are recognised here and
    the answer is recorded per camera rather than asserted, because which of the
    two the viewer used is precisely the fact that decides whether a set's labels
    can be read as the mapping's precision.
    """

    result: dict[str, object] = {
        "filmstrip_raster": list(FILMSTRIP_RASTER),
        "method": (
            "pose_dx_px / pose_dy_px as written into each label, in filmstrip "
            "pixels; compared against the pose diagnostic's matcher-convention "
            "median scaled from its own raster to the filmstrip raster"
        ),
        "cameras": {},
    }
    diagnostic = json.loads(POSE_DIAGNOSTIC.read_text())["cameras"]
    for camera in cameras:
        defaults = manifest["cameras"][camera]["pose_defaults"]
        raster_w, raster_h = defaults["measuredRaster"]
        scaled_dx = defaults["measuredDxPx"] * FILMSTRIP_RASTER[0] / raster_w
        scaled_dy = defaults["measuredDyPx"] * FILMSTRIP_RASTER[1] / raster_h
        seeded = (defaults["dxPx"], defaults["dyPx"])
        as_displacement = (round(scaled_dx), round(scaled_dy))
        as_compensation = (round(-scaled_dx), round(-scaled_dy))
        if seeded == as_compensation:
            seeded_convention = "compensation"
        elif seeded == as_displacement:
            seeded_convention = "raw_displacement"
        else:
            raise SystemExit(
                f"{camera}: the filmstrip raster in this script reproduces neither "
                f"the measurement {as_displacement} nor its negation "
                f"{as_compensation} as the manifest's seeded default {seeded}"
            )
        # Zero is its own negation, so CAM0's 0/0 default is consistent with the
        # correct convention and is recorded as such rather than as ambiguous.
        sign_correct = seeded_convention == "compensation" or seeded == (0, 0)
        rows = [row for row in labels if row["camera_id"] == camera]
        used = [row for row in rows if row["label"] == MATCH]
        dx = np.array([float(row["pose_dx_px"]) for row in used])
        dy = np.array([float(row["pose_dy_px"]) for row in used])
        at_default = sum(
            1
            for row in rows
            if (float(row["pose_dx_px"]), float(row["pose_dy_px"]))
            == (float(defaults["dxPx"]), float(defaults["dyPx"]))
        )
        label_convention = diagnostic[camera].get("label_convention", {}).get("dx_px", {})
        result["cameras"][camera] = {
            "match_labels_measured": len(used),
            "measured_dx_px_at_measurement_raster": defaults["measuredDxPx"],
            "measured_dy_px_at_measurement_raster": defaults["measuredDyPx"],
            "measured_dx_px_scaled_to_filmstrip": round(float(scaled_dx), 2),
            "measured_dy_px_scaled_to_filmstrip": round(float(scaled_dy), 2),
            "seeded_default": [defaults["dxPx"], defaults["dyPx"]],
            "annotator_dx_px": {
                "median": float(np.median(dx)) if dx.size else None,
                "mad": float(np.median(np.abs(dx - np.median(dx)))) if dx.size else None,
                "iqr": (
                    [float(np.percentile(dx, 25)), float(np.percentile(dx, 75))]
                    if dx.size
                    else None
                ),
                "min": float(dx.min()) if dx.size else None,
                "max": float(dx.max()) if dx.size else None,
            },
            "annotator_dy_px": {
                "median": float(np.median(dy)) if dy.size else None,
                "mad": float(np.median(np.abs(dy - np.median(dy)))) if dy.size else None,
                "min": float(dy.min()) if dy.size else None,
                "max": float(dy.max()) if dy.size else None,
            },
            "rows": len(rows),
            "rows_left_at_the_seeded_default": at_default,
            "seeded_default_convention": seeded_convention,
            "seeded_default_sign_is_correct": sign_correct,
            "dx_median_minus_scaled_measurement": (
                round(float(np.median(dx) - scaled_dx), 2) if dx.size else None
            ),
            "dx_median_minus_seeded_default": (
                round(float(np.median(dx) - defaults["dxPx"]), 2) if dx.size else None
            ),
            "dx_sign_agrees_with_measurement": (
                bool(np.sign(np.median(dx)) == np.sign(scaled_dx)) if dx.size else None
            ),
            "dx_agrees_with_the_seeded_default": (
                bool(np.median(dx) == defaults["dxPx"]) if dx.size else None
            ),
            "label_convention_dx_px_scaled_to_filmstrip": (
                round(label_convention["median"] * FILMSTRIP_RASTER[0] / raster_w, 2)
                if label_convention
                else None
            ),
        }
    return result


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _precision(result: dict[str, object]) -> str:
    if result["precision"] is None:
        return "n/a"
    interval = result["precision_wilson_95"]
    return f"{result['precision']:.3f} [{interval[0]:.2f}-{interval[1]:.2f}]"


def markdown(payload: dict) -> str:
    lines: list[str] = []
    add = lines.append
    cameras = tuple(payload["cameras_scored"])
    tag = payload["version_tag"]
    add(f"# Blind {tag}: {payload['set_title']}")
    add("")
    if payload.get("redo_of"):
        add(payload["redo_of"])
        add("")
    add(
        f"{payload['labels']} labels ({', '.join(camera.upper() for camera in cameras)}), "
        f"sealed {payload['label_sealed_at_utc']} against method freeze "
        f"`{payload['method_freeze_v3_sha256'][:12]}` frozen "
        f"{payload['method_frozen_at_utc']}. The query set was exported "
        f"{payload['query_set_exported_at_utc']}, before the freeze; the labels "
        "were sealed after it. Scored once."
    )
    add("")
    add("## Headline: match labels, per camera")
    add("")
    add(
        "| Method | Cam | Accepted | Correct | Precision | Wilson 95% | Within 2 | "
        "Coverage | Median dist | p90 |"
    )
    add("|---|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for method, block in payload["methods"].items():
        for camera in cameras:
            result = block["cameras"][camera]
            interval = result["precision_wilson_95"]
            rendered = f"{interval[0]:.2f}-{interval[1]:.2f}" if interval else "n/a"
            gaps = result["interval_distance"]
            add(
                f"| `{method}` | {camera.upper()} | "
                f"{result['accepted']}/{result['match_labels']} | {result['correct']} | "
                f"{result['precision'] if result['precision'] is not None else 'n/a'} | "
                f"{rendered} | {result['within_two']}/{result['accepted']} | "
                f"{_percent(result['coverage_of_match_labels'])} | "
                f"{gaps.get('median', 'n/a')} | {gaps.get('p90', 'n/a')} |"
            )
    add("")
    add("## Abstention")
    add("")
    add(
        "`undetermined` labels are counts only - they are not negatives, and they "
        "leave the precision denominator. `no_correspondence` is the real "
        "abstention target: an acceptance is a false accept, an abstention a true "
        "reject."
    )
    add("")
    add("| Method | Undetermined answered | no_correspondence false accepts | true rejects |")
    add("|---|---:|---:|---:|")
    for method, block in payload["methods"].items():
        pooled = block["pooled"]
        add(
            f"| `{method}` | {pooled['undetermined']['accepted']}/"
            f"{pooled['undetermined']['labels']} | "
            f"{pooled['no_correspondence']['false_accepts']}/"
            f"{pooled['no_correspondence']['labels']} | "
            f"{pooled['no_correspondence']['true_rejects']}/"
            f"{pooled['no_correspondence']['labels']} |"
        )
    add("")
    add("## Per stratum")
    add("")
    add(payload["stratum_dependency_disclosure"])
    add("")
    for method, block in payload["methods"].items():
        add(f"### `{method}`")
        add("")
        add("| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |")
        add("|---|---|---:|---:|---:|---|---:|")
        for camera in cameras:
            for stratum in STRATA:
                result = block["strata"][camera].get(stratum)
                if not result or not result["match_labels"]:
                    continue
                interval = result["precision_wilson_95"]
                rendered = f"{interval[0]:.2f}-{interval[1]:.2f}" if interval else "n/a"
                add(
                    f"| {camera.upper()} | {stratum} | "
                    f"{result['accepted']}/{result['match_labels']} | "
                    f"{result['correct']} | "
                    f"{result['precision'] if result['precision'] is not None else 'n/a'} | "
                    f"{rendered} | {result['within_two']}/{result['accepted']} |"
                )
        add("")
    add("## The kink claim, tested on labels it did not select")
    add("")
    claim = payload["kink_claim_test"]
    add(claim["claim"])
    add("")
    add(claim["v2_reference"])
    add("")
    add("| Method | Cam | Kink miss rate | Corridor miss rate | Difference |")
    add("|---|---|---:|---:|---:|")
    for method, block in claim["per_method"].items():
        for camera in cameras:
            row = block[camera]
            kink, corridor = row["kink"], row["corridor"]
            difference = row["kink_minus_corridor_miss_rate"]
            add(
                f"| `{method}` | {camera.upper()} | "
                f"{kink['misses_interval_distance_gt_0']}/{kink['accepted']} = "
                f"{kink['miss_rate'] if kink['miss_rate'] is not None else 'n/a'} | "
                f"{corridor['misses_interval_distance_gt_0']}/{corridor['accepted']} = "
                f"{corridor['miss_rate'] if corridor['miss_rate'] is not None else 'n/a'} | "
                f"{difference if difference is not None else 'n/a'} |"
            )
    add("")
    add(claim["verdict"])
    add("")
    add("## Signed error: is the prediction late or early?")
    add("")
    add(payload["signed_error_check"]["why"])
    add("")
    add(
        "| Method | Cam | n | Median signed | Mean | Later | Earlier | Inside | "
        "Median signed vs preferred |"
    )
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for method, block in payload["methods"].items():
        for camera in cameras:
            result = block["signed_error"][camera]
            if not result.get("n"):
                continue
            signed = result["signed_interval_error"]
            preferred = result["signed_preferred_frame_error"]
            add(
                f"| `{method}` | {camera.upper()} | {result['n']} | "
                f"{signed['median']} | {signed['mean']} | "
                f"{signed['later_than_interval']} | "
                f"{signed['earlier_than_interval']} | "
                f"{signed['inside_interval']} | {preferred['median']} |"
            )
    add("")
    reference = payload["signed_error_check"]["v3_cam5_reference"]
    if reference.get("available"):
        add(
            f"The withdrawn v3 CAM5 set, for reference: n {reference['n']}, median "
            f"signed interval error {reference['signed_interval_error_median']}, "
            f"{reference['later_than_interval']} later against "
            f"{reference['earlier_than_interval']} earlier, median signed error "
            f"against the preferred frame "
            f"{reference['signed_preferred_frame_error_median']}."
        )
        add("")
    add(payload["signed_error_check"]["verdict"])
    add("")
    comparison = payload.get("v2_comparison", {})
    if comparison.get("available"):
        add("## CAM5 against the v2 set")
        add("")
        v2 = comparison["v2"]
        add(
            "| Set | Slice | Accepted | Correct | Precision | Wilson 95% | Within 2 |"
        )
        add("|---|---|---:|---:|---:|---|---:|")
        add(
            f"| v2 | route-uniform | {v2['accepted']}/{v2['match_labels']} | "
            f"{v2['correct']} | {v2['precision']} | "
            f"{v2['precision_wilson_95'][0]:.2f}-{v2['precision_wilson_95'][1]:.2f} | "
            f"{v2['within_two_by_interval_distance']}/{v2['accepted']} |"
        )
        for key, name in (
            ("this_set_corridor_only", f"{tag} corridor only"),
            ("this_set_overall", f"{tag} overall"),
        ):
            row = comparison[key]
            bounds = row["precision_wilson_95"]
            rendered = f"{bounds[0]:.2f}-{bounds[1]:.2f}" if bounds else "n/a"
            add(
                f"| {tag} | {name.split(' ', 1)[1]} | "
                f"{row['accepted']}/{row['match_labels']} | {row['correct']} | "
                f"{row['precision']} | {rendered} | "
                f"{row['within_two_by_interval_distance']}/{row['accepted']} |"
            )
        add("")
        add(v2["note"])
        add("")
        add(comparison["comparability"])
        add("")
    add("## Pose convention check")
    add("")
    for camera, result in payload["pose_convention_check"]["cameras"].items():
        add(
            f"- **{camera.upper()}**: annotator median "
            f"dx {result['annotator_dx_px']['median']} px "
            f"(MAD {result['annotator_dx_px']['mad']}, range "
            f"{result['annotator_dx_px']['min']} to {result['annotator_dx_px']['max']}), "
            f"dy {result['annotator_dy_px']['median']} px "
            f"(MAD {result['annotator_dy_px']['mad']}), over "
            f"{result['match_labels_measured']} match labels. Measured "
            f"{result['measured_dx_px_at_measurement_raster']}/"
            f"{result['measured_dy_px_at_measurement_raster']} px at "
            f"{'x'.join(str(v) for v in [960, 720])} scale = "
            f"{result['measured_dx_px_scaled_to_filmstrip']}/"
            f"{result['measured_dy_px_scaled_to_filmstrip']} px; seeded default "
            f"({result['seeded_default'][0]}, {result['seeded_default'][1]}) px is "
            f"the {result['seeded_default_convention']} and its sign is "
            f"{'correct' if result['seeded_default_sign_is_correct'] else 'WRONG'}; "
            f"rows left at the seeded default: "
            f"{result['rows_left_at_the_seeded_default']}/{result['rows']}."
        )
    add("")
    add(payload["pose_convention_check"]["verdict"])
    add("")
    add("## Annotator remarks, recorded before the seal")
    add("")
    for remark in payload["annotator_remarks_recorded_before_seal"]:
        add(
            f"- **{remark['camera_id']} A{remark['runA_frame']}** "
            f"({remark['kind']}): {remark['remark']}"
        )
    add("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rescore an already-scored blind set (this destroys its held-out "
        "status); with --convention-sensitivity it only permits replacing the "
        "sensitivity file, and rescores nothing",
    )
    parser.add_argument(
        "--set-name",
        default=DEFAULT_SET_NAME,
        help="which sealed label set to score; decides the manifest, the freeze, "
        "the seal and where the results are written",
    )
    parser.add_argument(
        "--convention-sensitivity",
        action="store_true",
        help="do not score: re-read an already-scored set and report how its "
        "precision moves under a constant per-camera frame shift, the shift "
        "being read from the pose diagnostic. Writes "
        f"{SENSITIVITY_FILENAME} and never rewrites the results file.",
    )
    arguments = parser.parse_args()

    set_name = arguments.set_name
    tag = freeze_method_v3.version_tag(set_name)
    directory = blind_dir(set_name)
    target = directory / f"{set_name}_results.json"

    if arguments.convention_sensitivity:
        destination = directory / SENSITIVITY_FILENAME
        if destination.exists() and not arguments.force:
            raise FileExistsError(
                f"{destination.relative_to(ROOT)} already exists. This reading is "
                "cited by the report and by README.md; overwriting it silently "
                "would let a changed sweep replace the one those numbers were "
                "read from. Pass --force to replace it deliberately."
            )
        payload = convention_shift_sensitivity(set_name)
        destination.write_text(json.dumps(payload, indent=2))
        print(f"{tag}: convention-shift sensitivity (a reading, not a held-out number)")
        print(f"scored pass unchanged: {payload['scored_pass']['sha256'][:12]}")
        for camera, block in payload["convention_shift_sensitivity"].items():
            constant = block["constant"]
            print(
                f"\n{camera}: constant "
                f"{constant['gap_frames_at_median_sweep']:+.2f} frames "
                f"(rounded {constant['rounded_frames']:+d}) read from "
                f"{constant['source_file']} built {constant['source_built_at_utc']}"
            )
            header = (
                f"{'method':<28} " + " ".join(
                    f"{shift:>+14d}" for shift in block["shifts_frames"]
                )
            )
            print(header)
            print("-" * len(header))
            for method, rows in block["methods"].items():
                cells = []
                for shift in block["shifts_frames"]:
                    row = rows["by_shift"][str(shift)]
                    precision = (
                        f"{row['correct']}/{row['accepted']}={row['precision']:.2f}"
                        if row["precision"] is not None
                        else "n/a"
                    )
                    cells.append(f"{precision:>14}")
                print(f"{method:<28} " + " ".join(cells))
            for method, rows in block["methods"].items():
                print(
                    f"peak {method:<28} {rows['peak_shift_frames']:+d} frames"
                    f"  (the measured constant: "
                    f"{'yes' if rows['peak_is_the_measured_constant'] else 'no'})"
                )
        controls = payload["control_sweeps"]["sets"]
        print("\ncontrol sweeps (sets this reading was not derived from)")
        header = (
            f"{'set':<20} {'role':<24} "
            + " ".join(f"{shift:>+16d}" for shift in payload["control_sweeps"]["sets"][
                next(iter(controls))
            ]["shifts_frames"])
        )
        print(header)
        print("-" * len(header))
        for key, block in controls.items():
            cells = []
            for shift in block["shifts_frames"]:
                row = block["by_shift"][str(shift)]
                cells.append(
                    f"{row['correct']}/{row['accepted']}"
                    f" {row['signed_late_early_inside']}".rjust(16)
                )
            print(f"{key:<20} {str(block['role']):<24} " + " ".join(cells))
        print(f"\nWrote {destination.relative_to(ROOT)}")
        return
    if target.is_file() and not arguments.force:
        raise FileExistsError(
            f"{target} exists. The blind set is scored once; re-running after "
            "seeing the result is how a held-out estimate stops being one."
        )

    contract = check_contract(set_name)
    manifest = contract["manifest"]
    cameras = cameras_of(manifest)
    with Path(contract["label_file"]).open(newline="") as handle:  # type: ignore[arg-type]
        labels = list(csv.DictReader(handle))

    strata = load_strata(set_name)
    exposure, slope = load_quality_flags()

    all_cases: list[dict[str, object]] = []
    methods: dict[str, object] = {}
    for method in METHODS:
        cases = build_cases(method, labels, strata, exposure, slope, cameras)
        all_cases.extend(cases)
        per_camera = {
            camera: score([case for case in cases if case["camera_id"] == camera])
            for camera in cameras
        }
        per_stratum = {
            camera: {
                stratum: score(
                    [
                        case
                        for case in cases
                        if case["camera_id"] == camera and case["stratum"] == stratum
                    ]
                )
                for stratum in STRATA
            }
            for camera in cameras
        }
        methods[method] = {
            "description": METHODS[method]["description"],
            "held_out": METHODS[method]["held_out"],
            "cameras": per_camera,
            "pooled": score(cases),
            "strata": per_stratum,
            "strata_pooled": {
                stratum: score([case for case in cases if case["stratum"] == stratum])
                for stratum in STRATA
            },
            "signed_error": {
                camera: signed_error_summary(
                    [case for case in cases if case["camera_id"] == camera]
                )
                for camera in cameras
            },
        }

    v2_cross = json.loads(KINK_AUDIT.read_text())["label_cross_check"]["per_camera"]
    kink_test: dict[str, object] = {
        "claim": (
            "The claim under test, made on the v2 labels: match labels near a "
            f"mapping kink miss more often than labels away from one. On {tag} the "
            "comparison is the kink stratum against the corridor stratum, which "
            "is the untargeted control; the v2 labels play no part in it."
        ),
        "v2_reference": (
            "For reference only, the v2 numbers that produced the claim: CAM0 "
            f"{v2_cross['cam0']['near']['miss_rate']} near "
            f"({v2_cross['cam0']['near']['misses_interval_distance_gt_0']} of "
            f"{v2_cross['cam0']['near']['evaluable_match_labels']}) against "
            f"{v2_cross['cam0']['away']['miss_rate']} away "
            f"({v2_cross['cam0']['away']['misses_interval_distance_gt_0']} of "
            f"{v2_cross['cam0']['away']['evaluable_match_labels']}); CAM5 "
            f"{v2_cross['cam5']['near']['miss_rate']} against "
            f"{v2_cross['cam5']['away']['miss_rate']}."
        ),
        "miss_definition": "an accepted match label whose interval distance exceeds 0",
        "per_method": {},
    }
    for method in METHODS:
        cases = build_cases(method, labels, strata, exposure, slope, cameras)
        block: dict[str, object] = {}
        for camera in cameras + ("pooled",):
            selected = [
                case
                for case in cases
                if camera == "pooled" or case["camera_id"] == camera
            ]
            kink = miss_rate([case for case in selected if case["stratum"] == "kink"])
            corridor = miss_rate(
                [case for case in selected if case["stratum"] == "corridor"]
            )
            difference = (
                round(kink["miss_rate"] - corridor["miss_rate"], 4)
                if kink["miss_rate"] is not None and corridor["miss_rate"] is not None
                else None
            )
            block[camera] = {
                "kink": kink,
                "corridor": corridor,
                "kink_minus_corridor_miss_rate": difference,
            }
        kink_test["per_method"][method] = block

    submitted = kink_test["per_method"]["submitted_unified"]
    replicated = [
        camera
        for camera in cameras
        if (submitted[camera]["kink_minus_corridor_miss_rate"] or 0) > 0
    ]
    kink_test[f"replicated_on_{tag}"] = {
        "cameras_where_the_kink_stratum_misses_more": replicated,
        "verdict_basis": (
            "direction only; at these per-stratum counts the Wilson intervals "
            "overlap heavily and no rate is established"
        ),
    }
    kink_test["verdict"] = (
        "Replication verdict: the kink stratum misses more often than the "
        f"corridor stratum on {', '.join(camera.upper() for camera in replicated) or 'neither camera'}"
        f"{'' if len(replicated) == len(cameras) else ' only'}. "
        "The intervals overlap at these counts, so this is a direction, not a rate."
    )

    pose = pose_check(labels, manifest, cameras)
    cam5 = pose["cameras"].get("cam5")
    if cam5 is None:
        pose["verdict"] = "CAM5 was not part of this query set; no CAM5 pose check."
    elif not cam5["seeded_default_sign_is_correct"]:
        pose["verdict"] = (
            "The CAM5 sliders the annotator settled on have the opposite sign to the "
            "seeded default and a larger magnitude: median "
            f"{cam5['annotator_dx_px']['median']} px against the measured "
            f"{cam5['measured_dx_px_scaled_to_filmstrip']} px, with dy at "
            f"{cam5['annotator_dy_px']['median']} px against "
            f"{cam5['measured_dy_px_scaled_to_filmstrip']}. The one CAM5 row left at "
            "the seeded default is the undetermined query, so the default was "
            "overridden on every row the annotator could actually align. That is a "
            "sign error in how the measured shift was applied to the viewer, not a "
            "defect in the labels: the compensation moves the displayed Run B raster "
            "and the annotator corrected it by eye per query."
        )
    else:
        pose["verdict"] = (
            "The CAM5 viewer seeded the compensation with the correct sign this "
            f"time: default ({cam5['seeded_default'][0]}, "
            f"{cam5['seeded_default'][1]}) px, the negation of the measured "
            f"displacement ({cam5['measured_dx_px_scaled_to_filmstrip']}, "
            f"{cam5['measured_dy_px_scaled_to_filmstrip']}) px on the filmstrip "
            f"raster. The annotator left {cam5['rows_left_at_the_seeded_default']} "
            f"of {cam5['rows']} rows exactly at that default (median dx "
            f"{cam5['annotator_dx_px']['median']}, dy "
            f"{cam5['annotator_dy_px']['median']}). The defect that withdrew the "
            "v3 CAM5 half does not appear here: there was no reason to drag a "
            "slider that was already compensating the right way, so frame choice "
            "and shift could not trade off against each other."
        )

    signed_check = {
        "why": (
            "An unsigned error cannot separate a mapping that is imprecise from a "
            "viewer that shows the wrong picture. On the withdrawn v3 CAM5 set the "
            "submitted mapping's error against the labels was systematically "
            "positive - the annotator, recovering a wrongly-displaced raster with "
            "the slider, settled on Run B frames earlier than the mapping's - so "
            "the direction of the error is reported here, not only its size."
        ),
        "v3_cam5_reference": v3_cam5_signed_reference(),
        "this_set": {
            camera: methods["submitted_unified"]["signed_error"][camera]
            for camera in cameras
        },
    }
    reference = signed_check["v3_cam5_reference"]
    if "cam5" in cameras and methods["submitted_unified"]["signed_error"]["cam5"].get("n"):
        here = methods["submitted_unified"]["signed_error"]["cam5"]["signed_interval_error"]
        balanced = abs(here["later_than_interval"] - here["earlier_than_interval"]) <= max(
            2, round(0.25 * (here["later_than_interval"] + here["earlier_than_interval"]))
        )
        # The criterion is stated before the number is read: an offset counts as
        # gone only if the median sits within a frame of zero AND late and early
        # errors are in comparable number. Either alone can be met by accident.
        gone = bool(abs(here["median"]) <= 1 and balanced)
        signed_check["criterion"] = (
            "The offset counts as gone if the median signed interval error is "
            "within +-1 frame of zero and the later and earlier counts differ by "
            "no more than the larger of 2 and a quarter of the errors that fall "
            "outside the interval."
        )
        signed_check["systematic_offset_gone_on_cam5"] = gone
        signed_check["median_shift_against_v3"] = (
            round(here["median"] - reference["signed_interval_error_median"], 4)
            if reference.get("available")
            else None
        )
        signed_check["what_the_residual_means"] = (
            "The v3b viewer applied the compensation with the correct sign and "
            "every v3b row was left at the seeded default, so no slider excess "
            "could trade against frame choice on this set. Whatever signed offset "
            "survives here is therefore a property of the submitted mapping on "
            "CAM5 - it predicts Run B frames late - and not of the annotation "
            "tool. The v3 figure is not a clean measurement of either, which is "
            "why it was withdrawn rather than reinterpreted."
        )
        signed_check["verdict"] = (
            f"CAM5, submitted mapping: median signed interval error {here['median']} "
            f"(mean {here['mean']}), {here['later_than_interval']} predictions later "
            f"than the label interval against {here['earlier_than_interval']} earlier "
            f"and {here['inside_interval']} inside"
            + (
                f", against the withdrawn v3 set's median "
                f"{reference['signed_interval_error_median']} with "
                f"{reference['later_than_interval']} later against "
                f"{reference['earlier_than_interval']} earlier. "
                if reference.get("available")
                else ". "
            )
            + (
                "By the stated criterion the systematic offset is GONE."
                if gone
                else "By the stated criterion the systematic offset is NOT gone: "
                "it is smaller than on v3 but still one-directional. With the "
                "corrected tool and every slider at its default, the residual is "
                "the mapping predicting late on CAM5, not the viewer."
            )
        )
    else:
        signed_check["verdict"] = "No CAM5 acceptances in this set; nothing to check."

    comparison = v2_cam5_comparison(
        build_cases("submitted_unified", labels, strata, exposure, slope, cameras)
    ) if "cam5" in cameras else {"available": False}

    payload = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_set": set_name,
        "version_tag": tag,
        "cameras_scored": list(cameras),
        "set_title": (
            "the CAM5 redo, scored once"
            if set_name != DEFAULT_SET_NAME
            else "the targeted label set, scored once"
        ),
        "redo_of": (
            "This set re-asks CAM5 alone. The CAM5 half of blind v3 was annotated "
            "with a viewer that applied the pose compensation with the wrong sign, "
            "found after the seal from the sealed labels themselves and recorded in "
            "`outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`; its CAM5 "
            "precision is withdrawn. The CAM0 half of v3 is unaffected and stands. "
            "The redo runs on a fresh query set under `method_freeze_v3b.json`, "
            "which hashes exactly the same artifacts as the v3 freeze, so no "
            "parameter of any scored method changed between the two."
            if set_name != DEFAULT_SET_NAME
            else ""
        ),
        "method_freeze_v3_sha256": contract["freeze"]["freeze_sha256"],
        "method_frozen_at_utc": contract["freeze"]["frozen_at_utc"],
        "query_set_sha256": contract["freeze"]["query_set_sha256"],
        "query_set_exported_at_utc": contract["freeze"]["query_set_exported_at_utc"],
        "label_seal_sha256": contract["seal"]["label_file_sha256"],
        "label_sealed_at_utc": contract["seal"]["sealed_at_utc"],
        "labels": len(labels),
        "ordering_note": (
            "Query set exported, then the method frozen, then the labels sealed, "
            "then this single scoring pass. That records ordering, not custody."
        ),
        "pooling_policy": (
            "Cameras are reported separately. Pooled rows exist for readability; "
            "the two cameras observe the same two traversals and are not "
            "independent samples."
        ),
        "label_semantics": contract["seal"]["label_semantics"],
        "stratum_dependency_disclosure": manifest["stratum_dependency_disclosure"],
        "strata_counts_exported": {
            camera: manifest["cameras"][camera].get("strata_counts")
            for camera in cameras
        },
        "strata_shortfalls": {
            camera: manifest["cameras"][camera].get("strata_shortfalls", [])
            for camera in cameras
        },
        "what_the_v3_labels_may_judge": contract["freeze"]["what_the_labels_may_judge"],
        "gate_definitions": {
            "slope_gate": (
                "frozen rule, read from the slope_gate column of "
                "outputs/task3/frame_quality_flags.csv: near_kink on a Run A "
                "ordinal within 10 frames of a one-step Run B increment >= 6"
            ),
            "dark_gate": (
                "DATA_POLICY.md's exposure rule read as an abstention: the Run A "
                "frame or the predicted Run B frame flagged too_dark (mean luma "
                "< 12 or p99 < 40 of 255)"
            ),
            "both_gates": "the union of the two",
        },
        "annotator_remarks_recorded_before_seal": contract["seal"][
            "annotator_remarks_recorded_before_seal"
        ],
        "methods": methods,
        "kink_claim_test": kink_test,
        "signed_error_check": signed_check,
        "v2_comparison": comparison,
        "pose_convention_check": pose,
    }

    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))
    (directory / f"{set_name}_results.md").write_text(markdown(payload))
    with (directory / f"{set_name}_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_cases[0]))
        writer.writeheader()
        writer.writerows(all_cases)

    print(f"{tag} freeze  {payload['method_freeze_v3_sha256'][:12]}  {payload['method_frozen_at_utc']}")
    print(f"label seal {payload['label_seal_sha256'][:12]}  {payload['label_sealed_at_utc']}")
    print()
    header = (
        f"{'method':<28} {'cam':<5} {'accepted':>9} {'precision':>10} "
        f"{'wilson 95%':>15} {'within2':>8} {'coverage':>9}"
    )
    print(header)
    print("-" * len(header))
    for method, block in methods.items():
        for camera in cameras:
            result = block["cameras"][camera]
            interval = result["precision_wilson_95"]
            rendered = (
                f"{interval[0]:.2f}-{interval[1]:.2f}" if interval else "n/a"
            )
            precision = (
                f"{result['precision']:.3f}" if result["precision"] is not None else "n/a"
            )
            print(
                f"{method:<28} {camera:<5} "
                f"{result['accepted']:>4}/{result['match_labels']:<4} "
                f"{precision:>10} {rendered:>15} "
                f"{result['within_two']:>3}/{result['accepted']:<4} "
                f"{_percent(result['coverage_of_match_labels']):>9}"
            )
    print()
    for method, block in methods.items():
        corridor = {
            camera: block["strata"][camera]["corridor"] for camera in cameras
        }
        rendered = "  ".join(
            f"{camera.upper()} {corridor[camera]['correct']}/{corridor[camera]['accepted']}"
            f" = {corridor[camera]['precision']}"
            for camera in cameras
        )
        print(f"corridor-only precision  {method:<28} {rendered}")
    print()
    for camera in cameras:
        row = kink_test["per_method"]["submitted_unified"][camera]
        print(
            f"kink vs corridor miss rate  {camera}: "
            f"kink {row['kink']['misses_interval_distance_gt_0']}/{row['kink']['accepted']}"
            f" = {row['kink']['miss_rate']}   corridor "
            f"{row['corridor']['misses_interval_distance_gt_0']}/{row['corridor']['accepted']}"
            f" = {row['corridor']['miss_rate']}"
        )
    print()
    pooled = methods["submitted_unified"]["pooled"]
    print(
        f"no_correspondence: {pooled['no_correspondence']['false_accepts']} false accepts, "
        f"{pooled['no_correspondence']['true_rejects']} true rejects of "
        f"{pooled['no_correspondence']['labels']} (submitted mapping)"
    )
    print()
    print(signed_check["verdict"])
    if comparison.get("available"):
        corridor_block = comparison["this_set_corridor_only"]
        overall_block = comparison["this_set_overall"]
        print(
            f"CAM5 v2 {comparison['v2']['correct']}/{comparison['v2']['accepted']} = "
            f"{comparison['v2']['precision']}  ->  {tag} corridor "
            f"{corridor_block['correct']}/{corridor_block['accepted']} = "
            f"{corridor_block['precision']}, {tag} overall "
            f"{overall_block['correct']}/{overall_block['accepted']} = "
            f"{overall_block['precision']}"
        )
    print(
        f"\nWrote {target.relative_to(ROOT)} and {set_name}_results.md, "
        f"{set_name}_cases.csv"
    )


if __name__ == "__main__":
    main()
