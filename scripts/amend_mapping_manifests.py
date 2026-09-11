#!/usr/bin/env python3
"""Document what `confidence` means, in the manifests, without touching a CSV.

Two things a consumer of the shipped mappings could previously get wrong, and
neither was written down beside the files:

1. **`confidence` is ordinal, not a probability.** The column already carries a
   `confidence_is_calibrated_probability: false` flag, which says what it is
   *not*. It never said what the five values mean, nor which of them actually
   occur. In particular **0.50, the geometry-propagated tier, does not occur in
   the submitted unified files at all** - every row that could have carried it
   is either a verified strong acceptance or an abstention. A reader who sees
   the scheme documented in the code and assumes the shipped file exercises all
   of it will mis-weight the middle of the range.

2. **The mapping is many-to-one.** Run B moves more slowly through parts of the
   route, so several Run A frames legitimately name the same Run B partner. On
   CAM0 a third of the mapped rows share their partner with at least one other
   row; on CAM5 it is over half. That is the correct answer to the question the
   mapping is asked, but a consumer that treats `runB_frame` as a key, or that
   samples training pairs without weighting, will silently over-count those
   places.

Both are recorded here as an **amendment**: the manifests keep every field the
pipeline wrote, gain a `confidence_semantics` block, a
`confidence_tier_values_present` census and a many-to-one census, and carry
`manifest_amended_at_utc` with the reason. Nothing recomputes a metric and no
mapping CSV is read for anything but its own census, because the mapping files
are frozen by three method freezes and must stay byte-identical.

Run: `PYTHONPATH=scripts python3 scripts/amend_mapping_manifests.py`
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

UNIFIED = tuple(
    (
        ROOT / "outputs" / "task2_unified" / camera / "frame_mapping_unified.csv",
        ROOT / "outputs" / "task2_unified" / camera / "unified_manifest.json",
    )
    for camera in ("cam0", "cam5")
)
SPARSE = tuple(
    (
        ROOT / "outputs" / "task2_final" / f"{camera}_frame_mapping.csv",
        ROOT / "outputs" / "task2_final" / f"{camera}_mapping_manifest.json",
    )
    for camera in ("cam0", "cam5")
)

# The full ordinal scheme the unified writer can emit. Values absent from a
# shipped file are listed as absent rather than left to be inferred.
UNIFIED_SCALE = {
    "1.00": "uncalibrated_strong: local geometry was verified at this row",
    "0.67": "uncalibrated_sequence_supported: the sequence gate carried it, geometry did not verify it here",
    "0.50": "uncalibrated_geometry_propagated: a nearby sampled check passed and was propagated",
    "0.33": "uncalibrated_bridged: a short gap bridged between two accepted neighbours",
    "0.00": "abstain: no partner is asserted; this is not a negative",
}

# The sparse-anchor alternative's own scale. Its CSV carries the CATEGORY in
# `confidence`, which is why the bundle rewrites that column to the number and
# moves the category to `tier` (see build_submission.py).
SPARSE_SCALE = {
    "1.00": "uncalibrated_geometry_anchor: one of the 36 fixed anchors, geometry-supported",
    "0.50": "uncalibrated_bounded_interpolation: interpolated between two adjacent accepted anchors",
    "0.00": "abstain: no partner is asserted; this is not a negative",
}
SPARSE_CATEGORY_TO_VALUE = {
    "uncalibrated_geometry_anchor": "1.00",
    "uncalibrated_bounded_interpolation": "0.50",
    "abstain": "0.00",
}

AMENDMENT_REASON = (
    "The manifest recorded that confidence is not a calibrated probability but "
    "never said what its values mean, which of them the shipped file actually "
    "contains, or that the mapping is many-to-one. An adversarial read of the "
    "bundle found all three undocumented. Added here as an amendment rather "
    "than by rebuilding, because the mapping CSVs are hashed by three method "
    "freezes and must stay byte-identical."
)


def census(mapping: Path, sparse: bool) -> dict[str, object]:
    """Value counts and many-to-one structure, read from the CSV as it stands."""

    with mapping.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if sparse:
        values = Counter(
            SPARSE_CATEGORY_TO_VALUE.get(row["confidence"], row["confidence"])
            for row in rows
        )
        tiers = Counter(row["confidence"] for row in rows)
    else:
        values = Counter(row["confidence"] for row in rows)
        tiers = Counter(row["tier"] for row in rows)

    partners = Counter(row["runB_frame"] for row in rows if row["runB_frame"])
    mapped = sum(partners.values())
    shared = sum(count for count in partners.values() if count > 1)
    return {
        "rows": len(rows),
        "mapped_rows": mapped,
        "confidence_value_counts": dict(sorted(values.items(), reverse=True)),
        "tier_counts": dict(sorted(tiers.items())),
        "many_to_one": {
            "distinct_runB_partners": len(partners),
            "rows_sharing_a_partner": shared,
            "fraction_of_mapped_rows_sharing_a_partner": (
                round(shared / mapped, 4) if mapped else None
            ),
            "max_multiplicity": max(partners.values()) if partners else 0,
            "why": (
                "Run B traverses parts of the route more slowly than Run A, so "
                "several Run A frames correctly name the same Run B frame. This "
                "is the expected shape of the answer, not a defect."
            ),
            "what_a_consumer_must_do": (
                "Do not treat runB_frame as a key. When sampling training pairs, "
                "weight by 1/multiplicity or de-duplicate on the Run B side, or "
                "the slow stretches dominate."
            ),
        },
    }


def amend(mapping: Path, manifest: Path, sparse: bool) -> dict[str, object]:
    payload = json.loads(manifest.read_text())
    counts = census(mapping, sparse)
    scale = SPARSE_SCALE if sparse else UNIFIED_SCALE
    present = set(counts["confidence_value_counts"])
    payload["manifest_amended_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["manifest_amendment_reason"] = AMENDMENT_REASON
    payload["confidence_semantics"] = {
        "kind": "ordinal rank, not a probability",
        "is_calibrated_probability": False,
        "scale": scale,
        "categorical_label_column": "tier" if not sparse else (
            "tier in the bundled copy; the repository CSV carries the category "
            "in the confidence column itself and build_submission.py rewrites it"
        ),
        "ordering_only": (
            "The numbers order the tiers and nothing more. 1.00 does not mean "
            "certain and 0.67 is not two thirds of a probability; the blind "
            "evaluations measure what a tier is actually worth."
        ),
        "abstention": (
            "0.00 with an empty runB_frame is an abstention, never a negative. "
            "It says this method asserts no partner here, not that no partner "
            "exists."
        ),
        "values_absent_from_this_file": {
            value: f"{meaning} - does not occur in this file"
            for value, meaning in scale.items()
            if value not in present
        },
        "note_on_0_50": (
            "0.50, the geometry-propagated tier, DOES NOT OCCUR in the submitted "
            "unified mappings: no shipped row carries it."
            if not sparse
            else "0.50 here is bounded interpolation between two accepted anchors."
        ),
    }
    payload["confidence_tier_values_present"] = counts["confidence_value_counts"]
    payload["tier_counts_present"] = counts["tier_counts"]
    payload["many_to_one_structure"] = counts["many_to_one"]
    manifest.write_text(json.dumps(payload, indent=2))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for group, sparse in ((UNIFIED, False), (SPARSE, True)):
        for mapping, manifest in group:
            if not mapping.is_file() or not manifest.is_file():
                print(f"  skipped {manifest.name} (mapping or manifest absent)")
                continue
            counts = amend(mapping, manifest, sparse)
            structure = counts["many_to_one"]
            print(
                f"{manifest.relative_to(ROOT)}: values "
                f"{sorted(counts['confidence_value_counts'], reverse=True)}, "
                f"{structure['rows_sharing_a_partner']}/{counts['mapped_rows']} rows "
                f"share a partner (max {structure['max_multiplicity']})"
            )


if __name__ == "__main__":
    main()
