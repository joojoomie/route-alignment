#!/usr/bin/env python3
"""Freeze everything the v3 targeted blind labels will be allowed to judge.

The v3 label set was drawn to test claims made after the v2 labels were read:
that kinks predict errors, that the slope gate buys precision, that the slope
refinement is better. Each of those is a statement about a specific artifact.
This freeze hashes those artifacts and the query set BEFORE the annotator opens
the pages, so the order export -> freeze -> labels -> seal is checkable and no
artifact judged by the v3 labels can be changed after the labels exist.

`--verify` recomputes every hash and refuses on any drift.

`--set-name` freezes a later query set against exactly the same artifacts.
`--set-name blind_v3b` writes `method_freeze_v3b.json`, hashing the same
mappings, variants, posteriors, kinks and quality flags as v3 plus the v3b query
set. That is the freeze the CAM5 redo runs under: the v3 CAM5 labels were taken
with a tool that applied the pose compensation with the wrong sign (see
`outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`), so CAM5 is asked again
on a fresh query set, and this freeze must precede the annotation for the redo
to be held out at all.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import freeze_method_v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET_NAME = "blind_v3"


def version_tag(set_name: str = DEFAULT_SET_NAME) -> str:
    return set_name.split("blind_", 1)[-1] if set_name.startswith("blind_") else set_name


def target_path(set_name: str = DEFAULT_SET_NAME) -> Path:
    return (
        ROOT / "outputs" / "task2_evaluation" / f"method_freeze_{version_tag(set_name)}.json"
    )


def query_manifest_path(set_name: str = DEFAULT_SET_NAME) -> Path:
    return ROOT / "outputs" / f"task2_{set_name}" / f"{set_name}_manifest.json"


TARGET = target_path()
QUERY_MANIFEST = query_manifest_path()

JUDGED_ARTIFACTS = {
    "submitted_unified": "outputs/task2_unified/{camera}/frame_mapping_unified.csv",
    "slope_refinement_variant": "outputs/task2_unified_variants/slope/{camera}/frame_mapping_unified.csv",
    "posterior_model": "outputs/task2_bayes/{camera}/frame_mapping_bayes.csv",
    "kinks": "outputs/task2_kinks/kinks_{camera}.csv",
}
SHARED_ARTIFACTS = {
    "frame_quality_flags": "outputs/task3/frame_quality_flags.csv",
    "mapping_kinks": "outputs/task2_kinks/mapping_kinks.json",
}
SLOPE_GATE_RULE = {"kink_min_increment": 6, "demote_within_runA_frames": 10}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(set_name: str = DEFAULT_SET_NAME) -> dict[str, object]:
    manifest = json.loads(query_manifest_path(set_name).read_text())
    hashes: dict[str, str] = {}
    for name, pattern in JUDGED_ARTIFACTS.items():
        for camera in ("cam0", "cam5"):
            path = ROOT / pattern.format(camera=camera)
            hashes[f"{name}/{camera}"] = sha256(path)
    for name, relative in SHARED_ARTIFACTS.items():
        hashes[name] = sha256(ROOT / relative)
    slope_manifest = json.loads(
        (ROOT / "outputs/task2_unified_variants/slope/cam0/unified_manifest.json").read_text()
    )
    tag = version_tag(set_name)
    lineage: dict[str, object] = {}
    if set_name != DEFAULT_SET_NAME and target_path(DEFAULT_SET_NAME).is_file():
        # A redo freeze names the freeze it repeats, so the chain from the
        # defective CAM5 run to this one is readable without the git log.
        lineage["v3_freeze_sha256"] = json.loads(
            target_path(DEFAULT_SET_NAME).read_text()
        )["freeze_sha256"]
        lineage["repeats"] = (
            "Same artifacts as method_freeze_v3.json, against a fresh query set. "
            "The CAM5 half of v3 was annotated with the pose compensation applied "
            "with the wrong sign; see "
            "outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md."
        )
    return {
        "schema_version": 3,
        "label_set": set_name,
        "query_set_cameras": list(manifest["cameras"]),
        "query_set_sha256": manifest["query_set_sha256"],
        "query_set_exported_at_utc": manifest["built_at_utc"],
        "artifact_sha256": hashes,
        "slope_gate_rule": SLOPE_GATE_RULE,
        "unified_parameters": freeze_method_v2.frozen_parameters(),
        "slope_variant_parameters": {
            key: slope_manifest[key] for key in slope_manifest if "refine" in key.lower() or key == "variant"
        },
        "v2_freeze_sha256": json.loads(
            (ROOT / "outputs/task2_evaluation/method_freeze_v2.json").read_text()
        )["freeze_sha256"],
        "what_the_labels_may_judge": sorted(JUDGED_ARTIFACTS),
        "rule": (
            f"No artifact hashed here may change after this freeze until the {tag} "
            f"labels are sealed and scored. Anything else built afterwards has no "
            f"held-out number from {tag}."
        ),
        **lineage,
    }


VERIFIED_KEYS = (
    "query_set_sha256",
    "artifact_sha256",
    "slope_gate_rule",
    "unified_parameters",
    "slope_variant_parameters",
)


def verify(quiet: bool = False, set_name: str = DEFAULT_SET_NAME) -> dict[str, object]:
    """Recompute every hash and refuse on any drift. Returns the frozen record.

    Exposed as a function so the v3 evaluation can refuse to score a drifted
    state without shelling out to this script.
    """

    tag = version_tag(set_name)
    target = target_path(set_name)
    if not target.is_file():
        raise SystemExit(f"No {tag} freeze at {target}; nothing to verify")
    frozen = json.loads(target.read_text())
    current = payload(set_name)
    drift = [key for key in VERIFIED_KEYS if frozen[key] != current[key]]
    if drift:
        raise SystemExit(
            f"{tag} freeze DRIFTED in {drift}; the labels cannot score this state"
        )
    if not quiet:
        print(
            f"{tag} freeze {frozen['freeze_sha256'][:12]} verified: "
            f"{len(frozen['artifact_sha256'])} artifacts unchanged"
        )
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--set-name",
        default=DEFAULT_SET_NAME,
        help="Query set to freeze against; decides which manifest is read and "
        "which method_freeze_<tag>.json is written.",
    )
    arguments = parser.parse_args()

    set_name = arguments.set_name
    target = target_path(set_name)

    if arguments.verify:
        verify(set_name=set_name)
        return

    current = payload(set_name)
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; a freeze is not rewritten")
    body = json.dumps(current, sort_keys=True).encode()
    current["freeze_sha256"] = hashlib.sha256(body).hexdigest()
    current["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(current, indent=2))
    print(f"Wrote {target.relative_to(ROOT)}")
    print(f"Froze {len(current['artifact_sha256'])} artifacts + query set {current['query_set_sha256'][:12]}")
    print(f"  freeze sha256 {current['freeze_sha256']}")
    print(f"  frozen at    {current['frozen_at_utc']}")
    print(f"  query set exported at {current['query_set_exported_at_utc']}")


if __name__ == "__main__":
    main()
