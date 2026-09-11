#!/usr/bin/env python3
"""Score development variants on the evidence that is still available.

The blind labels are spent, so a variant cannot earn a held-out number. What it
can earn is a label-free comparison on the same terms as the shipped method:

* cross-camera agreement, the dense signal that flagged the failure region
  before any label existed;
* endpoint closure against cross-camera-confirmed revisits;
* coverage of the route;

and, clearly marked as a diagnostic, the spent blind labels under the corrected
reading. A variant that wins on the label-free signals and loses on the labels
is reported that way; the two do not get averaged.

Variants are any directory under outputs/task2_unified_variants/ plus the
posterior model's opt-in channels.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import evaluate_loop_consistency as loop
import evaluate_task2_consistency as consistency
from evaluate_task2_blind_v2 import distance_to_interval, wilson_interval
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_variants"
LABELS = ROOT / "outputs" / "task2_evaluation" / "blind_v2" / "blind_v2_labels.csv"
CAMERAS = ("cam0", "cam5")



def candidates() -> dict[str, Path]:
    """name -> pattern with {camera}."""

    found = {"baseline (submitted)": ROOT / "outputs" / "task2_unified" / "{camera}" / "frame_mapping_unified.csv"}
    variants = ROOT / "outputs" / "task2_unified_variants"
    if variants.is_dir():
        for directory in sorted(variants.iterdir()):
            if all((directory / camera / "frame_mapping_unified.csv").is_file() for camera in CAMERAS):
                found[f"unified {directory.name}"] = directory / "{camera}" / "frame_mapping_unified.csv"
    bayes = ROOT / "outputs" / "task2_bayes"
    for suffix, name in (("", "posterior"), ("_speed", "posterior + speed")):
        if all((bayes / camera / f"frame_mapping_bayes{suffix}.csv").is_file() for camera in CAMERAS):
            found[name] = bayes / "{camera}" / f"frame_mapping_bayes{suffix}.csv"
    return found


def load(pattern: Path, camera: str) -> dict[int, dict[str, str]]:
    with Path(str(pattern).format(camera=camera)).open(newline="") as handle:
        return {int(row["runA_frame"]): row for row in csv.DictReader(handle)}


def cross_camera(pattern: Path) -> dict[str, float]:
    mappings = {
        camera: {frame: int(row["runB_frame"]) for frame, row in load(pattern, camera).items() if row["runB_frame"]}
        for camera in CAMERAS
    }
    shared = sorted(set(mappings["cam0"]) & set(mappings["cam5"]))
    if len(shared) < consistency.LAG_WINDOW:
        return {"frames": len(shared), "median": None, "p95": None}
    difference = np.array([mappings["cam0"][f] - mappings["cam5"][f] for f in shared], dtype=float)
    residual = np.abs(difference - consistency.running_median(difference, consistency.LAG_WINDOW))
    return {
        "frames": len(shared),
        "median": float(np.median(residual)),
        "p95": float(np.percentile(residual, 95)),
        "over_10": round(float((residual > consistency.LARGE_DISAGREEMENT_FRAMES).mean()), 4),
    }


def spent_labels(pattern: Path, camera: str, labels: list[dict[str, str]]) -> dict[str, object]:
    mapping = load(pattern, camera)
    determinable = [r for r in labels if r["camera_id"] == camera and r["label"] == "match"]
    accepted = correct = within = 0
    for row in determinable:
        predicted = mapping[int(row["runA_frame"])]["runB_frame"].strip()
        if not predicted:
            continue
        accepted += 1
        distance = distance_to_interval(int(predicted), int(row["runB_min"]), int(row["runB_max"]))
        correct += distance == 0
        within += distance <= 2
    return {
        "accepted": accepted,
        "of": len(determinable),
        "precision": round(correct / accepted, 4) if accepted else None,
        "wilson_95": wilson_interval(correct, accepted),
        "within_2": round(within / accepted, 4) if accepted else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    target = OUTPUT_DIR / "variant_comparison.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    with LABELS.open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    descriptors = {
        (run, camera): loop.descriptors(run, camera) for run in ("runA", "runB") for camera in CAMERAS
    }

    results: dict[str, dict[str, object]] = {}
    for name, pattern in candidates().items():
        print(f"[{name}]", flush=True)
        entry: dict[str, object] = {"cross_camera": cross_camera(pattern), "cameras": {}}
        for camera in CAMERAS:
            rows = load(pattern, camera)
            mapped = {f: int(r["runB_frame"]) for f, r in rows.items() if r["runB_frame"]}
            closure = loop.endpoint_closure(
                camera, mapped, descriptors[("runA", camera)], descriptors[("runB", camera)]
            )
            entry["cameras"][camera] = {
                "coverage": round(len(mapped) / len(rows), 4),
                "endpoint_closure": f"{closure['consistent']}/{closure['pairs_with_both_ends_mapped']}",
                "spent_labels_diagnostic": spent_labels(pattern, camera, labels),
            }
        results[name] = entry
        cc = entry["cross_camera"]
        print(f"   cross-camera median {cc['median']} p95 {cc['p95']} over {cc['frames']} frames")
        for camera in CAMERAS:
            c = entry["cameras"][camera]
            s = c["spent_labels_diagnostic"]
            print(f"   {camera}: coverage {c['coverage']:.3f}, closure {c['endpoint_closure']}, "
                  f"spent labels precision {s['precision']} within-2 {s['within_2']} ({s['accepted']}/{s['of']})")

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": (
            "Cross-camera agreement, closure and coverage are label-free. The spent-label "
            "column reuses the blind set after it was read and is a diagnostic only; no "
            "variant here has a held-out number."
        ),
        "variants": results,
    }
    atomic_write(target, lambda path: path.write_text(json.dumps(payload, indent=2)))
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
