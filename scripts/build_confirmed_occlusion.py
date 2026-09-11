#!/usr/bin/env python3
"""Record the one confirmed no-correspondence case in this data, and what the mappings do with it.

`FAILURE_ANALYSIS.md` says the blind label set cannot evaluate abstention in
either direction, because it contains no true no-correspondence case: every
query was sampled inside the stretch both runs traversed, so a genuine absence
was impossible by construction. That is still true of the labels. It is not true
of the data.

A frame-by-frame visual sweep of all four streams found one: an oncoming box
lorry passes Run B CAM0 at one to two metres, fills the entire frame for roughly
six frames, and is followed by a silver car that occludes about two thirds of
the frame for a further dozen. For that stretch the Run B descriptors describe a
vehicle's side panel, not a place. No Run A frame has a counterpart there.

The source is visual inspection, not a label. It was found by looking at
frames, it has no annotation protocol behind it, and its bounds are read off
contact sheets rather than measured by a detector - all of which is recorded in
the JSON so nobody quotes it as ground truth it is not. What it can do is act as
a single confirmed positive for the abstention question the labels cannot ask:
does either mapping notice?

This script only reads. It writes one JSON file and never touches a mapping.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "task2_abstention" / "confirmed_occlusion_runB_cam0.json"

UNIFIED = ROOT / "outputs" / "task2_unified" / "cam0" / "frame_mapping_unified.csv"
POSTERIOR = ROOT / "outputs" / "task2_bayes" / "cam0" / "frame_mapping_bayes.csv"
QUALITY = ROOT / "outputs" / "task3" / "frame_quality_flags.json"

CAMERA = "cam0"
RUN = "runB"

# Read from the contact sheets, in Run B CAM0 display ordinals.
INTERVAL = {"start": 240, "end": 270}
FULLY_OCCLUDED = {"start": 247, "end": 252, "occluder": "oncoming box lorry, whole frame"}
PARTLY_OCCLUDED = {"start": 258, "end": 270, "occluder": "passing silver car, about two thirds of frame"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rows_into_interval(path: Path, low: int, high: int) -> list[dict[str, str]]:
    return [
        row
        for row in read_rows(path)
        if row["runB_frame"].strip() and low <= int(row["runB_frame"]) <= high
    ]


def _floats(rows: list[dict[str, str]], column: str) -> list[float] | None:
    """Sorted numeric column values, or None if the column is categorical."""

    values = []
    for row in rows:
        text = row.get(column, "").strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            return None
    return sorted(values)


def describe(
    path: Path,
    columns: list[str],
    low: int = INTERVAL["start"],
    high: int = INTERVAL["end"],
) -> dict[str, object]:
    rows = rows_into_interval(path, low, high)
    if not rows:
        return {"rows_mapped_into_interval": 0, "abstained": True}
    frames = [int(row["runA_frame"]) for row in rows]
    report: dict[str, object] = {
        "mapping": str(path.relative_to(ROOT)),
        "rows_mapped_into_interval": len(rows),
        "runA_frames": {"first": min(frames), "last": max(frames)},
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "abstained": False,
    }
    for column in columns:
        if column not in rows[0]:
            continue
        values = _floats(rows, column)
        if values:
            report[column] = {"min": values[0], "median": values[len(values) // 2], "max": values[-1]}
        else:
            report[column] = dict(Counter(row[column] for row in rows))
    return report


def exposure_overlap() -> dict[str, object]:
    """Whether the independent exposure flag also fires here.

    It does, and by a different mechanism: the lorry's shadowed side panel fills
    the frame, so the frames are dark as well as uninformative. Two independent
    measurements landing on the same interval is worth recording; neither was
    built with the other in view.
    """

    if not QUALITY.exists():
        return {"available": False}
    payload = json.loads(QUALITY.read_text())
    stream = payload["streams"][f"{RUN}_{CAMERA}"]
    overlapping = [
        stretch
        for stretch in stream["dark_stretches"]
        if stretch["start"] <= INTERVAL["end"] and stretch["end"] >= INTERVAL["start"]
    ]
    return {
        "available": True,
        "source": str(QUALITY.relative_to(ROOT)),
        "overlapping_dark_stretches": overlapping,
        "note": (
            "The exposure flag was built to find under-exposed road, not "
            "occluders. It fires here because the lorry panel is dark, which "
            "means one flag partially covers the other failure mode."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": RUN,
        "camera": CAMERA,
        "interval": INTERVAL,
        "detail": [FULLY_OCCLUDED, PARTLY_OCCLUDED],
        "source": "visual inspection of decoded frames; not a label",
        "evidence_status": (
            "One instance, found by eye, with bounds read off a contact sheet. "
            "It is a confirmed positive for 'no correspondence exists here', "
            "which the blind label set contains none of. It is not an "
            "annotated ground-truth set and no rate may be computed from it."
        ),
        "why_it_matters": (
            "For roughly 30 consecutive Run B frames the descriptors describe a "
            "vehicle's side panel rather than a place. The claim is strongest "
            "over the fully occluded core, where no Run A frame can have a "
            "counterpart at all; over the partly occluded tail a third of the "
            "scene is still visible, and one blind label (cam0_v2_02, A355) does "
            "sit there and was answered. A method that can express 'no "
            "correspondence' should abstain over the core."
        ),
        "unified_mapping": describe(
            UNIFIED, ["confidence", "tier", "geometry_checked", "geometry_supportive"]
        ),
        "posterior_mapping": describe(
            POSTERIOR, ["posterior_probability", "posterior_no_correspondence"]
        ),
        "fully_occluded_core": {
            "interval": [FULLY_OCCLUDED["start"], FULLY_OCCLUDED["end"]],
            "unified": describe(
                UNIFIED, ["confidence", "tier"], FULLY_OCCLUDED["start"], FULLY_OCCLUDED["end"]
            ),
            "posterior": describe(
                POSTERIOR,
                ["posterior_probability", "posterior_no_correspondence"],
                FULLY_OCCLUDED["start"],
                FULLY_OCCLUDED["end"],
            ),
        },
        "exposure_flag": exposure_overlap(),
    }

    unified = payload["unified_mapping"]
    posterior = payload["posterior_mapping"]
    payload["finding"] = (
        f"Both mappings send Run A {unified['runA_frames']['first']}-"
        f"{unified['runA_frames']['last']} into the occluded interval and neither "
        f"abstains: {unified['rows_mapped_into_interval']} unified rows, all "
        f"{sorted(unified['status_counts'])[0]}, and "
        f"{posterior['rows_mapped_into_interval']} posterior rows whose "
        f"no-correspondence probability peaks at "
        f"{posterior['posterior_no_correspondence']['max']:.2g}. The posterior "
        "model has an explicit null state and still does not use it here, so its "
        "null state is demonstrated to be under-used on the one confirmed case "
        "available, independently of the held-out-stretch test it passes."
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  {OUTPUT.relative_to(ROOT)}")
    print(f"    {payload['finding']}")


if __name__ == "__main__":
    main()
