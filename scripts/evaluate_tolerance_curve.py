#!/usr/bin/env python3
"""Precision of every scored mapping as a function of frame tolerance.

The field reports place recognition at 25 m or ±10 frames; this data is 10 FPS,
so one frame is about a metre of travel and a strict single-frame hit is one to
two orders of magnitude tighter than any published protocol. The headline
reading here is therefore *within ±2 frames of the annotator's acceptable
interval* (0.2 s, about 2 m) --- still five times tighter than the tightest
published Nordland protocol --- with ±1, ±5, ±10 and the strict ±0 reading
beside it. Nothing is re-scored: the same sealed labels and the same frozen
mappings, read at several tolerances. The strict reading stays the last table
in the report.

Distance is `distance_to_interval`, the same function the blind evaluators use:
zero when the prediction lies inside [runB_min, runB_max], otherwise the gap to
the nearer end. Undetermined and no_correspondence labels are excluded, as in
the corrected reading; abstentions count against coverage, not precision.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from evaluate_task2_blind_v2 import distance_to_interval, wilson_interval
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "task2_evaluation" / "tolerance_curve.json"
TOLERANCES = (0, 1, 2, 5, 10)
HEADLINE_TOLERANCE = 2

LABEL_SETS = {
    "blind_v2": {
        "labels": "outputs/task2_evaluation/blind_v2/blind_v2_labels.csv",
        "cameras": ("cam0", "cam5"),
        "methods": {
            "submitted": "outputs/task2_unified/{camera}/frame_mapping_unified.csv",
            "v1_sparse_anchor": "outputs/task2_final/{camera}_frame_mapping.csv",
        },
        "status": "held-out for both methods (frozen before the labels were read)",
    },
    "blind_v3": {
        "labels": "outputs/task2_evaluation/blind_v3/blind_v3_labels.csv",
        "cameras": ("cam0",),
        "methods": {
            "submitted": "outputs/task2_unified/{camera}/frame_mapping_unified.csv",
            "slope_variant": "outputs/task2_unified_variants/slope/{camera}/frame_mapping_unified.csv",
            "posterior": "outputs/task2_bayes/{camera}/frame_mapping_bayes.csv",
        },
        "status": "held-out (v3 freeze); CAM5 half withdrawn for a tool defect, see TOOL_DEFECT_cam5.md",
    },
    "blind_v3b": {
        "labels": "outputs/task2_evaluation/blind_v3b/blind_v3b_labels.csv",
        "cameras": ("cam5",),
        "methods": {
            "submitted": "outputs/task2_unified/{camera}/frame_mapping_unified.csv",
            "slope_variant": "outputs/task2_unified_variants/slope/{camera}/frame_mapping_unified.csv",
            "posterior": "outputs/task2_bayes/{camera}/frame_mapping_bayes.csv",
        },
        "status": "held-out (v3b freeze); the CAM5 redo",
    },
}


def load_mapping(pattern: str, camera: str) -> dict[int, str]:
    path = ROOT / pattern.format(camera=camera)
    with path.open(newline="") as handle:
        return {int(row["runA_frame"]): row["runB_frame"].strip() for row in csv.DictReader(handle)}


def curve(labels: list[dict[str, str]], mapping: dict[int, str]) -> dict[str, object]:
    """Fractions of accepted match labels within each tolerance."""

    distances: list[int] = []
    for row in labels:
        predicted = mapping[int(row["runA_frame"])]
        if not predicted:
            continue
        distances.append(distance_to_interval(int(predicted), int(row["runB_min"]), int(row["runB_max"])))
    accepted = len(distances)
    result: dict[str, object] = {
        "match_labels": len(labels),
        "accepted": accepted,
        "coverage": round(accepted / len(labels), 4) if labels else None,
        "within": {},
    }
    for tolerance in TOLERANCES:
        hits = sum(1 for d in distances if d <= tolerance)
        result["within"][str(tolerance)] = {
            "hits": hits,
            "fraction": round(hits / accepted, 4) if accepted else None,
            "wilson_95": wilson_interval(hits, accepted) if accepted else None,
        }
    result["median_distance"] = float(sorted(distances)[len(distances) // 2]) if distances else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if TARGET.is_file() and not arguments.force:
        raise FileExistsError(f"{TARGET} exists; pass --force to rebuild")

    sets: dict[str, object] = {}
    for set_name, spec in LABEL_SETS.items():
        with (ROOT / spec["labels"]).open(newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["label"] == "match"]
        entry: dict[str, object] = {"status": spec["status"], "cameras": {}}
        for camera in spec["cameras"]:
            camera_labels = [r for r in rows if r["camera_id"] == camera]
            entry["cameras"][camera] = {
                method: curve(camera_labels, load_mapping(pattern, camera))
                for method, pattern in spec["methods"].items()
            }
        sets[set_name] = entry

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "tolerances_frames": list(TOLERANCES),
        "headline_tolerance_frames": HEADLINE_TOLERANCE,
        "distance": "distance_to_interval: 0 inside [runB_min, runB_max], else the gap to the nearer end",
        "frame_is_about": "0.1 s at 10 FPS; roughly one metre of travel at the route's median sweep",
        "literature_context": (
            "Published place-recognition protocols count a hit at 25 m (Pitts30k, Tokyo24/7, MSLS, "
            "RobotCar) or at +-10 frames at 1 FPS on Nordland; the tightest published Nordland "
            "readings use +-2 or +-1 frames at 1 FPS, i.e. +-10-40 m. The +-2-frame headline here "
            "is about 2 m. Not the same data; context for the scale of the tolerance only."
        ),
        "sets": sets,
    }
    atomic_write(TARGET, lambda path: path.write_text(json.dumps(payload, indent=2)))

    for set_name, entry in sets.items():
        print(f"[{set_name}] {entry['status']}")
        for camera, methods in entry["cameras"].items():
            for method, result in methods.items():
                within = result["within"]
                print(
                    f"   {camera} {method:18} accepted {result['accepted']}/{result['match_labels']}  "
                    + "  ".join(f"±{t}: {within[str(t)]['fraction']}" for t in TOLERANCES)
                )
    print(f"\nWrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
