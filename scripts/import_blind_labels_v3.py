#!/usr/bin/env python3
"""Validate and seal the annotator's blind v3 label CSVs.

Same contract as v2, with one substantive change. v2 had a single "no usable
match" answer, so its no-match rows mix two incompatible claims: *I could not
tell* and *there is no such place*. Only the second is a correct abstention
target; the first is a missing label. v3 records them as separate labels and
this importer keeps them separate all the way into the sealed file, so the
evaluation can score abstention against no_correspondence and exclude
undetermined rows from the denominator instead of guessing which is which.

Blindness is a property of the process, not of the file, so this checks what it
can and records what it cannot. It rejects any row carrying a model-derived
column, any row whose Run A frame was not in the exported query set, any row
left unanswered, and any non-match row that still names a Run B frame.

The seal records the label file's SHA-256, the query set hash, the export
timestamp and the time of sealing. The method freeze then embeds that hash,
which makes the ordering checkable after the fact. What the seal does not do is
prove nobody looked; that would need third-party custody. The honest claim is
ordering, not custody.

Do not run this until the annotation is finished. Sealing an incomplete file and
re-sealing later destroys the ordering guarantee the seal exists to provide.

`--set-name` selects the label set. It decides which manifest is read, which
cameras must be supplied, and where the sealed file lands: `blind_v3b` reads
`outputs/task2_blind_v3b/blind_v3b_manifest.json` and writes
`outputs/task2_evaluation/blind_v3b/`. Only the cameras that set exported may be
passed, and all of them must be passed at once.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET_NAME = "blind_v3"


def blind_dir(set_name: str = DEFAULT_SET_NAME) -> Path:
    return ROOT / "outputs" / f"task2_{set_name}"


def manifest_file(set_name: str = DEFAULT_SET_NAME) -> Path:
    return blind_dir(set_name) / f"{set_name}_manifest.json"


def output_dir(set_name: str = DEFAULT_SET_NAME) -> Path:
    return ROOT / "outputs" / "task2_evaluation" / set_name


def version_tag(set_name: str = DEFAULT_SET_NAME) -> str:
    return set_name.split("blind_", 1)[-1] if set_name.startswith("blind_") else set_name


# Which method freeze was in force when a set was sealed. v3 was sealed while
# the v2 freeze was the standing one and that record is history, not a default;
# every later set cites its own freeze.
FREEZE_AT_SEAL_TIME = {"blind_v3": "method_freeze_v2.json"}


def freeze_file(set_name: str = DEFAULT_SET_NAME) -> Path:
    name = FREEZE_AT_SEAL_TIME.get(set_name, f"method_freeze_{version_tag(set_name)}.json")
    return ROOT / "outputs" / "task2_evaluation" / name


BLIND_DIR = blind_dir()
MANIFEST_FILE = manifest_file()
OUTPUT_DIR = output_dir()

# The page writes exactly these columns, in this order; a test pins the two
# together. far/near carry an x and a y because the annotator must pick the far
# reference from the far band and the near one from the near band.
REQUIRED_COLUMNS = (
    "query_id",
    "camera_id",
    "runA_frame",
    "label",
    "runB_frame",
    "runB_min",
    "runB_max",
    "pose_dx_px",
    "pose_dy_px",
    "pose_compensated",
    "far_line_x_frac",
    "far_line_y_frac",
    "near_line_x_frac",
    "near_line_y_frac",
    "compare_mode",
    "source",
)
MATCH_LABEL = "match"
ABSTENTION_LABELS = ("undetermined", "no_correspondence")
ACTIVE_LABELS = (MATCH_LABEL,) + ABSTENTION_LABELS
REQUIRED_SOURCE = "manual_viewer_blind"
FORBIDDEN_COLUMN_PREFIXES = (
    "model_", "prediction", "predicted", "suggested", "score", "confidence", "stratum",
)


def _optional_float(value: str) -> float | str:
    value = value.strip()
    if not value:
        return ""
    return float(value)


def validate_rows(
    rows: list[dict[str, str]],
    camera: str,
    expected_queries: dict[str, int],
    run_b_min: int,
    run_b_max: int,
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
                "answered - as a match, as 'cannot determine', or as 'no "
                "corresponding place'."
            )
        if label not in ACTIVE_LABELS:
            raise ValueError(
                f"{camera}: {query_id} has unknown label {label!r}; expected one of "
                f"{ACTIVE_LABELS}"
            )

        compensated = row["pose_compensated"].strip().lower()
        if compensated not in ("true", "false"):
            raise ValueError(
                f"{camera}: {query_id} pose_compensated is {row['pose_compensated']!r}"
            )

        record: dict[str, object] = {
            "query_id": query_id,
            "camera_id": camera,
            "runA_frame": int(row["runA_frame"]),
            "label": label,
            "runB_frame": "",
            "runB_min": "",
            "runB_max": "",
            "pose_dx_px": float(row["pose_dx_px"]) if row["pose_dx_px"].strip() else 0.0,
            "pose_dy_px": float(row["pose_dy_px"]) if row["pose_dy_px"].strip() else 0.0,
            "pose_compensated": compensated == "true",
            "far_line_x_frac": _optional_float(row["far_line_x_frac"]),
            "far_line_y_frac": _optional_float(row["far_line_y_frac"]),
            "near_line_x_frac": _optional_float(row["near_line_x_frac"]),
            "near_line_y_frac": _optional_float(row["near_line_y_frac"]),
            "compare_mode": row["compare_mode"].strip(),
            "source": REQUIRED_SOURCE,
        }

        if label in ABSTENTION_LABELS:
            if any(row[column].strip() for column in ("runB_frame", "runB_min", "runB_max")):
                raise ValueError(
                    f"{camera}: {query_id} is labelled {label} but still names a Run B frame"
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
            for name, value in (("runB_frame", frame), ("runB_min", low), ("runB_max", high)):
                if not run_b_min <= value <= run_b_max:
                    raise ValueError(
                        f"{camera}: {query_id} {name} {value} is outside the exported "
                        f"Run B range {run_b_min}-{run_b_max}"
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
    parser.add_argument("--cam0", type=Path, help="<set-name>_cam0.csv")
    parser.add_argument("--cam5", type=Path, help="<set-name>_cam5.csv")
    parser.add_argument(
        "--set-name",
        default=DEFAULT_SET_NAME,
        help="Label set to seal; decides which manifest is read and where the "
        "sealed file is written (outputs/task2_evaluation/<set-name>/).",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    set_name = arguments.set_name
    manifest = json.loads(manifest_file(set_name).read_text())
    supplied = {
        camera: path
        for camera, path in (("cam0", arguments.cam0), ("cam5", arguments.cam5))
        if path is not None
    }
    exported = list(manifest["cameras"])
    if not supplied:
        raise SystemExit(f"{set_name}: pass a CSV for each exported camera {exported}")
    missing = [camera for camera in exported if camera not in supplied]
    if missing:
        raise SystemExit(
            f"{set_name}: no CSV given for {missing}. Every camera in the exported "
            "set must be sealed at once; sealing half a set and re-sealing later "
            "destroys the ordering guarantee."
        )
    extra = [camera for camera in supplied if camera not in exported]
    if extra:
        raise SystemExit(f"{set_name}: {extra} were not exported in this set")

    out_dir = output_dir(set_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{set_name}_labels.csv"
    seal_target = out_dir / "label_seal.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(
            f"{target} already exists. These labels are sealed; rebuilding them "
            "after a method has been frozen would break the evaluation contract. "
            "Pass --force only if the freeze has not happened yet."
        )

    combined: list[dict[str, object]] = []
    per_camera: dict[str, dict[str, object]] = {}
    for camera in exported:
        path = supplied[camera]
        entry = manifest["cameras"][camera]
        expected = {
            query["queryId"]: int(query["runAFrame"]) for query in entry["queries"]
        }
        strata = {
            query["queryId"]: query["stratum"] for query in entry["queries"]
        }
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        validated = validate_rows(
            rows, camera, expected, entry["runB_route_start"], entry["runB_route_end"]
        )
        combined.extend(validated)

        counts = {label: 0 for label in ACTIVE_LABELS}
        by_stratum: dict[str, dict[str, int]] = {}
        for record in validated:
            counts[str(record["label"])] += 1
            stratum = strata[str(record["query_id"])]
            bucket = by_stratum.setdefault(stratum, {label: 0 for label in ACTIVE_LABELS})
            bucket[str(record["label"])] += 1
        per_camera[camera] = {"labels": len(validated), "by_label": counts, "by_stratum": by_stratum}
        print(
            f"{camera}: {len(validated)} labels validated ("
            + ", ".join(f"{count} {label}" for label, count in counts.items())
            + ")"
        )

    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    freeze_path = freeze_file(set_name)
    freeze_digest = None
    if freeze_path.is_file():
        freeze_digest = json.loads(freeze_path.read_text()).get("freeze_sha256")

    seal = {
        "schema_version": 3,
        "label_set": set_name,
        "cameras": exported,
        "method_freeze_file": str(freeze_path.relative_to(ROOT)),
        "query_set_sha256": manifest.get("query_set_sha256"),
        "query_set_exported_at_utc": manifest.get("built_at_utc"),
        "method_freeze_sha256_at_seal_time": freeze_digest,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_file": str(target.relative_to(ROOT)),
        "label_file_sha256": digest,
        "rows": len(combined),
        "per_camera": per_camera,
        "label_semantics": {
            "match": "a Run B frame shows the same world position, within the recorded interval",
            "undetermined": (
                "a counterpart may exist; the annotator could not identify which frame. "
                "Not evidence for or against abstention - exclude from the denominator "
                "and report how many were excluded."
            ),
            "no_correspondence": (
                "no Run B frame shows this place, because the view was blocked or the "
                "route was not driven there. This is the correct abstention target."
            ),
        },
        "blindness_checks_performed": [
            "no model-derived or stratum columns present",
            "every query answered, none left unlabelled",
            "Run A frames match the exported query set exactly",
            "undetermined and no_correspondence rows name no Run B frame",
            "preferred frame lies inside its acceptable interval",
            "every Run B frame lies inside the exported Run B route range",
        ],
        "trust_model": (
            "The seal proves these labels existed unchanged before the method freeze "
            "that cites this hash. It does not prove the labels were never inspected; "
            "that would require third-party custody."
        ),
        "stratum_dependency_disclosure": manifest.get("stratum_dependency_disclosure"),
    }
    seal_target.write_text(json.dumps(seal, indent=2))

    print(f"\nSealed {len(combined)} labels")
    print(f"  {target.relative_to(ROOT)}")
    print(f"  sha256 {digest}")
    print("\nRecord this hash in the method freeze before running any evaluation.")


if __name__ == "__main__":
    main()
