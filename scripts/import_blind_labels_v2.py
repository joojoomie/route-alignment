#!/usr/bin/env python3
"""Validate and seal the annotator's blind label CSVs.

Blindness is a property of the process, not of the file, so this checks what it
can and records what it cannot. It rejects any row carrying a model-derived
column, any row whose Run A frame was not in the exported query set, and any
no-match row that still names a Run B frame.

The seal is the part that matters for the evaluation contract. It records the
label file's SHA-256 and the time it was sealed. The method freeze then embeds
that hash, which makes the ordering checkable after the fact: the labels
demonstrably existed, unchanged, before the method was frozen.

What the seal does not do is prove nobody looked. That would need the labels
withheld by a third party. The honest claim is ordering, not custody, and the
report should say so rather than implying more.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLIND_DIR = ROOT / "outputs" / "task2_blind_v2"
MANIFEST_FILE = BLIND_DIR / "blind_v2_manifest.json"
OUTPUT_DIR = ROOT / "outputs" / "task2_evaluation" / "blind_v2"

REQUIRED_COLUMNS = (
    "query_id",
    "camera_id",
    "runA_frame",
    "label",
    "runB_frame",
    "runB_min",
    "runB_max",
    "source",
)
ACTIVE_LABELS = ("match", "no_match")
REQUIRED_SOURCE = "manual_viewer_blind"
FORBIDDEN_COLUMN_PREFIXES = ("model_", "prediction", "suggested", "score", "confidence")


def validate_rows(
    rows: list[dict[str, str]], camera: str, expected_queries: dict[str, int]
) -> list[dict[str, object]]:
    if not rows:
        raise ValueError(f"{camera}: no rows supplied")

    present = set(rows[0])
    missing = [column for column in REQUIRED_COLUMNS if column not in present]
    if missing:
        raise ValueError(f"{camera}: label file is missing columns {missing}")
    leaking = [
        column
        for column in present
        if any(column.lower().startswith(prefix) for prefix in FORBIDDEN_COLUMN_PREFIXES)
    ]
    if leaking:
        raise ValueError(
            f"{camera}: label file carries model-derived columns {leaking}; "
            "these labels are not prediction-blind"
        )

    seen: set[str] = set()
    validated: list[dict[str, object]] = []
    for row in rows:
        query_id = row["query_id"].strip()
        if query_id in seen:
            raise ValueError(f"{camera}: duplicate query id {query_id}")
        seen.add(query_id)
        if query_id not in expected_queries:
            raise ValueError(f"{camera}: {query_id} was not in the exported query set")
        if row["camera_id"].strip() != camera:
            raise ValueError(f"{camera}: {query_id} carries camera {row['camera_id']!r}")
        if int(row["runA_frame"]) != expected_queries[query_id]:
            raise ValueError(
                f"{camera}: {query_id} Run A frame {row['runA_frame']} does not match "
                f"the exported {expected_queries[query_id]}"
            )
        if row["source"].strip() != REQUIRED_SOURCE:
            raise ValueError(
                f"{camera}: {query_id} source is {row['source']!r}, expected {REQUIRED_SOURCE!r}"
            )

        label = row["label"].strip()
        if label == "unlabelled":
            raise ValueError(
                f"{camera}: {query_id} is still unlabelled. Every query must be "
                "answered, including with 'no usable match'."
            )
        if label not in ACTIVE_LABELS:
            raise ValueError(f"{camera}: {query_id} has unknown label {label!r}")

        record: dict[str, object] = {
            "query_id": query_id,
            "camera_id": camera,
            "runA_frame": int(row["runA_frame"]),
            "label": label,
            "runB_frame": "",
            "runB_min": "",
            "runB_max": "",
            "source": REQUIRED_SOURCE,
        }
        if label == "no_match":
            if any(row[column].strip() for column in ("runB_frame", "runB_min", "runB_max")):
                raise ValueError(
                    f"{camera}: {query_id} is labelled no_match but still names a Run B frame"
                )
        else:
            frame = int(row["runB_frame"])
            low = int(row["runB_min"])
            high = int(row["runB_max"])
            if not low <= frame <= high:
                raise ValueError(
                    f"{camera}: {query_id} preferred frame {frame} is outside its "
                    f"interval {low}-{high}"
                )
            record.update({"runB_frame": frame, "runB_min": low, "runB_max": high})
        validated.append(record)

    unanswered = set(expected_queries) - seen
    if unanswered:
        raise ValueError(
            f"{camera}: {len(unanswered)} queries were never answered, "
            f"for example {sorted(unanswered)[:3]}"
        )
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam0", type=Path, required=True, help="blind_v2_cam0.csv")
    parser.add_argument("--cam5", type=Path, required=True, help="blind_v2_cam5.csv")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    manifest = json.loads(MANIFEST_FILE.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / "blind_v2_labels.csv"
    seal_target = OUTPUT_DIR / "label_seal.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(
            f"{target} already exists. These labels are sealed; rebuilding them "
            "after a method has been frozen would break the evaluation contract. "
            "Pass --force only if the freeze has not happened yet."
        )

    combined: list[dict[str, object]] = []
    per_camera: dict[str, dict[str, int]] = {}
    for camera, path in (("cam0", arguments.cam0), ("cam5", arguments.cam5)):
        expected = {
            query["queryId"]: int(query["runAFrame"])
            for query in manifest["cameras"][camera]["queries"]
        }
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        validated = validate_rows(rows, camera, expected)
        combined.extend(validated)
        matches = sum(1 for row in validated if row["label"] == "match")
        per_camera[camera] = {
            "labels": len(validated),
            "matches": matches,
            "no_matches": len(validated) - matches,
        }
        print(
            f"{camera}: {len(validated)} labels validated "
            f"({matches} match, {len(validated) - matches} no_match)"
        )

    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    # Cite the query set and, if it already exists, the method freeze. The
    # chain a reader can then verify is freeze -> queries -> labels.
    freeze_path = ROOT / "outputs" / "task2_evaluation" / "method_freeze_v2.json"
    freeze_digest = None
    if freeze_path.is_file():
        freeze_digest = json.loads(freeze_path.read_text()).get("freeze_sha256")
    seal = {
        "schema_version": 2,
        "query_set_sha256": manifest.get("query_set_sha256"),
        "query_set_exported_at_utc": manifest.get("built_at_utc"),
        "method_freeze_sha256_at_seal_time": freeze_digest,
        "label_set": "blind_v2",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_file": str(target.relative_to(ROOT)),
        "label_file_sha256": digest,
        "rows": len(combined),
        "per_camera": per_camera,
        "blindness_checks_performed": [
            "no model-derived columns present",
            "every query answered, none left unlabelled",
            "Run A frames match the exported query set exactly",
            "no_match rows name no Run B frame",
            "preferred frame lies inside its acceptable interval",
        ],
        "trust_model": (
            "The seal proves these labels existed unchanged before the method "
            "freeze that cites this hash. It does not prove the labels were "
            "never inspected; that would require third-party custody."
        ),
    }
    seal_target.write_text(json.dumps(seal, indent=2))

    print(f"\nSealed {len(combined)} labels")
    print(f"  {target.relative_to(ROOT)}")
    print(f"  sha256 {digest}")
    print("\nRecord this hash in the method freeze before running any evaluation.")


if __name__ == "__main__":
    main()
