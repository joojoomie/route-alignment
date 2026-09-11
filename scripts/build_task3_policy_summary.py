#!/usr/bin/env python3
"""Regenerate the Task 3 dataset-disposition summary from the shipped artifacts.

The previous `outputs/task3/policy_summary.csv` was written before the
audit-driven rebuild and its `weak_alignment_rows` column still carried the
**v1 sparse-anchor** counts (1,071 CAM0 / 287 CAM5). Those are the alternative
mapping's numbers, not the submitted one's, so a reader taking the disposition
table at face value would size the weak-supervision problem at roughly half of
what is actually shipped. Nothing regenerated the file, which is how it went
stale without anything noticing.

Every column now comes from a named artifact:

| column | source |
|---|---|
| `canonical_decoded_images` | row count per stream in `outputs/task3/frame_quality_flags.csv` |
| `pre_route_or_transition_excluded` | `route_start` in `outputs/task2_keyframes/route_phase_boundaries.csv` |
| `route_region_images` | the difference of the two above |
| `development_representatives` | `selected_route_keyframes` in `keyframe_selection_summary.csv` |
| `weak_alignment_rows` | mapped Run A rows of the **submitted** mapping |
| `sparse_anchor_alternative_rows` | mapped rows of the v1 alternative, kept so the change is visible rather than silent |

Run A carries the mapping columns because the mapping is indexed by Run A
ordinal; the Run B rows leave them empty, as before.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALITY_FLAGS = ROOT / "outputs" / "task3" / "frame_quality_flags.csv"
BOUNDARIES = ROOT / "outputs" / "task2_keyframes" / "route_phase_boundaries.csv"
KEYFRAMES = ROOT / "outputs" / "task2_keyframes" / "keyframe_selection_summary.csv"
UNIFIED = ROOT / "outputs" / "task2_unified" / "{camera}" / "frame_mapping_unified.csv"
SPARSE = ROOT / "outputs" / "task2_final" / "{camera}_frame_mapping.csv"
TARGET = ROOT / "outputs" / "task3" / "policy_summary.csv"
SIDECAR = ROOT / "outputs" / "task3" / "policy_summary_provenance.json"

FIELDS = (
    "run",
    "camera_id",
    "canonical_decoded_images",
    "pre_route_or_transition_excluded",
    "route_region_images",
    "development_representatives",
    "weak_alignment_rows",
    "sparse_anchor_alternative_rows",
    "raw_archive_disposition",
    "default_route_alignment_disposition",
)

STREAMS = (("runA", "cam0"), ("runA", "cam5"), ("runB", "cam0"), ("runB", "cam5"))


def mapped_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for row in csv.DictReader(handle) if row["runB_frame"])


def build() -> tuple[list[dict[str, object]], dict[str, object]]:
    with QUALITY_FLAGS.open(newline="") as handle:
        decoded = Counter((row["run"], row["camera"]) for row in csv.DictReader(handle))
    with BOUNDARIES.open(newline="") as handle:
        boundaries = {row["run"]: row for row in csv.DictReader(handle)}
    with KEYFRAMES.open(newline="") as handle:
        keyframes = {
            (row["run"], row["camera_id"]): row for row in csv.DictReader(handle)
        }

    rows: list[dict[str, object]] = []
    for run, camera in STREAMS:
        images = decoded[(run, camera)]
        excluded = int(boundaries[run]["route_start"])
        row: dict[str, object] = {
            "run": run,
            "camera_id": camera,
            "canonical_decoded_images": images,
            "pre_route_or_transition_excluded": excluded,
            "route_region_images": images - excluded,
            "development_representatives": int(
                float(keyframes[(run, camera)]["selected_route_keyframes"])
            ),
            "weak_alignment_rows": "",
            "sparse_anchor_alternative_rows": "",
            "raw_archive_disposition": "retain",
            "default_route_alignment_disposition": "eligible_after_qc",
        }
        if run == "runA":
            row["weak_alignment_rows"] = mapped_rows(
                Path(str(UNIFIED).format(camera=camera))
            )
            row["sparse_anchor_alternative_rows"] = mapped_rows(
                Path(str(SPARSE).format(camera=camera))
            )
        rows.append(row)

    provenance = {
        "schema_version": 2,
        "regenerated_because": (
            "weak_alignment_rows carried the v1 sparse-anchor counts (1071 CAM0, "
            "287 CAM5) long after the submitted mapping became the unified dense "
            "one. The column now describes the submitted mapping and the v1 "
            "counts move to their own column so the change is visible."
        ),
        "columns": {
            "canonical_decoded_images": str(QUALITY_FLAGS.relative_to(ROOT)),
            "pre_route_or_transition_excluded": str(BOUNDARIES.relative_to(ROOT)),
            "route_region_images": "canonical_decoded_images - pre_route_or_transition_excluded",
            "development_representatives": str(KEYFRAMES.relative_to(ROOT)),
            "weak_alignment_rows": "outputs/task2_unified/<camera>/frame_mapping_unified.csv, rows with a runB_frame",
            "sparse_anchor_alternative_rows": "outputs/task2_final/<camera>_frame_mapping.csv, rows with a runB_frame",
        },
        "weak_supervision_note": (
            "Every one of these rows is a weak positive, not a label. Coverage is "
            "not accuracy: see the blind evaluations for what a populated row is "
            "actually worth, per camera."
        ),
        "totals": {
            "canonical_decoded_images": sum(
                int(row["canonical_decoded_images"]) for row in rows
            ),
            "pre_route_or_transition_excluded": sum(
                int(row["pre_route_or_transition_excluded"]) for row in rows
            ),
            "route_region_images": sum(int(row["route_region_images"]) for row in rows),
            "development_representatives": sum(
                int(row["development_representatives"]) for row in rows
            ),
            "weak_alignment_rows": sum(
                int(row["weak_alignment_rows"]) for row in rows if row["weak_alignment_rows"] != ""
            ),
            "sparse_anchor_alternative_rows": sum(
                int(row["sparse_anchor_alternative_rows"])
                for row in rows
                if row["sparse_anchor_alternative_rows"] != ""
            ),
        },
    }
    return rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows, provenance = build()
    with TARGET.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    SIDECAR.write_text(json.dumps(provenance, indent=2))
    for row in rows:
        print(
            f"{row['run']} {row['camera_id']}: {row['canonical_decoded_images']} decoded, "
            f"{row['route_region_images']} in route, weak rows "
            f"{row['weak_alignment_rows'] or '-'} "
            f"(v1 alternative {row['sparse_anchor_alternative_rows'] or '-'})"
        )
    print(f"\nWrote {TARGET.relative_to(ROOT)} and {SIDECAR.name}")
    print(json.dumps(provenance["totals"], indent=2))


if __name__ == "__main__":
    main()
