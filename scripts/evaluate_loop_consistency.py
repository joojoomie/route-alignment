#!/usr/bin/env python3
"""Check a mapping against the route's own closure, using no labels at all.

The vehicle returns to where it started, and that fact constrains any correct
mapping in two ways that cost nothing to test.

**Endpoint closure.** If two Run A frames are the same place seen on the way out
and on the way back, then whatever Run B frames they map to must also be the
same place. The mapping is free to be wrong about *where* they are; it is not
free to send two views of one place to two different places. This is the only
independent check available on the closure region, where the correspondence is
genuinely ambiguous and every method here abstains or guesses.

**Path-length closure.** Both runs traverse the same loop, so cumulative image
motion is a monotone coordinate along the route with the same shape in both -
the depth structure that turns motion into distance is a property of the road,
not of the run. A correct mapping should therefore carry Run A's normalised
progress onto Run B's. Deviations localise where the mapping drifts.

Angle closure - the heading must also come back around by one full turn - is the
third constraint the loop provides and is *not* implemented, because it could not
be recovered from this data. Four attempts failed: regressing displacement on
image row, a far-field yaw estimate, ego-band pose correlation, and separating
rotation from translation using the two opposed cameras, whose flow turns out to
be uncorrelated (-0.09 on run A). All four founder on the same missing thing, a
camera calibration. That is worth stating plainly: the blocker here is not
algorithmic, and no amount of further processing of these frames replaces a
checkerboard session.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image

import build_task2_unified as unified
import frame_service
from atomic_io import atomic_write


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task2_loop_consistency"

MAPPINGS = {
    "v2_unified": ROOT / "outputs" / "task2_unified" / "{camera}" / "frame_mapping_unified.csv",
    "bayes": ROOT / "outputs" / "task2_bayes" / "{camera}" / "frame_mapping_bayes.csv",
    "bayes_no_loop": ROOT / "outputs" / "task2_bayes" / "{camera}" / "frame_mapping_bayes_no_loop.csv",
}

# A revisit must be far enough apart in time to be a second occurrence rather
# than the next frame, but not so far that genuine nearby revisits are excluded:
# this route passes some places twice within 25 seconds. Demanding a large
# separation forces the search onto weaker, coincidental pairs, which then look
# like mapping failures when they are really detection failures.
MIN_REVISIT_SEPARATION = 150
REVISIT_SIMILARITY = 0.88
MAX_REVISIT_PAIRS = 40
# A pair qualifies only if each frame is among the other's strongest distant
# matches. Mutual nearest neighbours, not merely a high score, which is what
# separates a real second occurrence from two places that happen to resemble
# each other.
REVISIT_MUTUAL_RANK = 5

# Two Run B frames count as the same place if they are close in time, or if each
# is among the other's strongest non-local matches. A rank test rather than a
# similarity threshold, because Run B is darker and rainier and its self-
# similarity sits systematically lower: judging both runs by one absolute cut
# would call every genuine Run B revisit a failure.
SAME_PLACE_FRAMES = 12
SAME_PLACE_RANK = 25

MOTION_WIDTH, MOTION_HEIGHT = 192, 144



def load_mapping(pattern: Path, camera: str) -> dict[int, int]:
    path = Path(str(pattern).format(camera=camera))
    with path.open(newline="") as handle:
        return {
            int(row["runA_frame"]): int(row["runB_frame"])
            for row in csv.DictReader(handle)
            if row["runB_frame"]
        }


def descriptors(run: str, camera: str) -> np.ndarray:
    return np.load(unified.descriptor_cache_path(run, camera))


def find_revisits(descriptors_a: np.ndarray, route: range) -> list[tuple[int, int, float]]:
    """Run A frame pairs that are the same place on two separate occasions."""

    index = np.array(route)
    subset = descriptors_a[index]
    similarity = subset @ subset.T
    separation = np.abs(index[:, None] - index[None, :])
    similarity = np.where(separation >= MIN_REVISIT_SEPARATION, similarity, -np.inf)

    # Rank every frame's distant matches once, then keep only mutual top-N pairs.
    order = np.argsort(-similarity, axis=1)
    rank = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    rank[rows, order] = np.arange(order.shape[1])[None, :] + 1

    chosen: list[tuple[int, int, float]] = []
    candidates = np.argwhere(
        (rank <= REVISIT_MUTUAL_RANK)
        & (rank.T <= REVISIT_MUTUAL_RANK)
        & (similarity >= REVISIT_SIMILARITY)
    )
    scored = sorted(
        ((int(a), int(b), float(similarity[a, b])) for a, b in candidates if a < b),
        key=lambda entry: -entry[2],
    )
    for first_index, second_index, score in scored:
        first = int(index[first_index])
        second = int(index[second_index])
        if any(abs(first - a) < 60 and abs(second - b) < 60 for a, b, _ in chosen):
            continue
        chosen.append((first, second, score))
        if len(chosen) >= MAX_REVISIT_PAIRS:
            break
    return chosen


def nonlocal_rank(descriptors: np.ndarray, anchor: int, candidate: int) -> int | None:
    """Where `candidate` ranks among `anchor`'s distant matches, 1 being best.

    Rank rather than similarity, so the test adapts to each run's own scale
    instead of importing a threshold calibrated on the other one.
    """

    if anchor >= len(descriptors) or candidate >= len(descriptors):
        return None
    scores = descriptors @ descriptors[anchor]
    distant = np.abs(np.arange(len(descriptors)) - anchor) >= SAME_PLACE_FRAMES
    if not distant[candidate]:
        return 1
    ordered = np.argsort(-np.where(distant, scores, -np.inf))
    position = int(np.where(ordered == candidate)[0][0]) + 1
    return position


def endpoint_closure(
    camera: str, mapping: dict[int, int], descriptors_a: np.ndarray, descriptors_b: np.ndarray
) -> dict[str, object]:
    """Two Run A views of one place must not map to two different places.

    Revisit pairs are those confirmed by BOTH cameras, the same definition the
    posterior model's constraint uses. Scoring against single-camera pairs
    would judge the mapping on resemblances that may not be revisits at all.
    """

    import build_task2_bayes as bayes

    bounds = unified.route_bounds()
    route = range(bounds["runA_start"], min(unified.RUN_A_ROUTE_TAIL_EXCLUSIVE, len(descriptors_a)))
    rows = np.array(route)
    other = "cam5" if camera == "cam0" else "cam0"
    try:
        descriptors_other = descriptors("runA", other)
    except FileNotFoundError:
        descriptors_other = None
    confirmed = bayes.run_a_revisits(descriptors_a, rows, descriptors_other)
    revisits = [
        (int(rows[a]), int(rows[b]), float(descriptors_a[rows[a]] @ descriptors_a[rows[b]]))
        for a, b in confirmed
    ]
    # Thin to one pair per neighbourhood so a long revisited stretch does not
    # count dozens of times.
    thinned: list[tuple[int, int, float]] = []
    for first, second, score in sorted(revisits, key=lambda e: -e[2]):
        if any(abs(first - a) < 60 and abs(second - b) < 60 for a, b, _ in thinned):
            continue
        thinned.append((first, second, score))
        if len(thinned) >= MAX_REVISIT_PAIRS:
            break
    revisits = thinned

    cases: list[dict[str, object]] = []
    for first, second, similarity in revisits:
        partner_first = mapping.get(first)
        partner_second = mapping.get(second)
        if partner_first is None or partner_second is None:
            cases.append(
                {
                    "runA_first": first,
                    "runA_second": second,
                    "runA_similarity": round(similarity, 4),
                    "verdict": "not_both_mapped",
                }
            )
            continue
        gap = abs(partner_first - partner_second)
        partner_similarity = float(descriptors_b[partner_first] @ descriptors_b[partner_second])
        rank = nonlocal_rank(descriptors_b, partner_first, partner_second)
        consistent = (
            gap <= SAME_PLACE_FRAMES
            or (rank is not None and rank <= SAME_PLACE_RANK)
        )
        cases.append(
            {
                "runA_first": first,
                "runA_second": second,
                "runA_similarity": round(similarity, 4),
                "runB_first": partner_first,
                "runB_second": partner_second,
                "runB_frame_gap": gap,
                "runB_similarity": round(partner_similarity, 4),
                "runB_nonlocal_rank": rank,
                "verdict": "consistent" if consistent else "INCONSISTENT",
            }
        )

    checked = [c for c in cases if c["verdict"] in ("consistent", "INCONSISTENT")]
    inconsistent = [c for c in checked if c["verdict"] == "INCONSISTENT"]
    return {
        "revisit_pairs_found": len(revisits),
        "pairs_with_both_ends_mapped": len(checked),
        "consistent": len(checked) - len(inconsistent),
        "inconsistent": len(inconsistent),
        "worst": sorted(inconsistent, key=lambda c: -c["runB_frame_gap"])[:5],
        "cases": cases,
    }


def cumulative_motion(run: str, camera: str) -> np.ndarray:
    """Cumulative frame-to-frame image motion: a monotone route coordinate."""

    frames = []
    for _, image in frame_service.iter_frames(
        frame_service.source_path(run, camera), MOTION_WIDTH, MOTION_HEIGHT
    ):
        frames.append(np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32))
    stack = np.stack(frames)
    motion = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], np.cumsum(motion)])


def path_length_closure(
    camera: str, mapping: dict[int, int], progress_a: np.ndarray, progress_b: np.ndarray
) -> dict[str, object]:
    """Normalised progress must transfer along the mapping."""

    bounds = unified.route_bounds()
    a_start = bounds["runA_start"]
    a_end = min(unified.RUN_A_ROUTE_TAIL_EXCLUSIVE, len(progress_a) - 1)
    b_start = bounds["runB_start"]
    b_end = len(progress_b) - 1

    def normalise(values: np.ndarray, low: int, high: int) -> np.ndarray:
        span = values[high] - values[low]
        return (values - values[low]) / max(span, 1e-9)

    normal_a = normalise(progress_a, a_start, a_end)
    normal_b = normalise(progress_b, b_start, b_end)

    frames = sorted(f for f in mapping if a_start <= f <= a_end)
    residual = np.array(
        [normal_a[f] - normal_b[min(mapping[f], b_end)] for f in frames], dtype=float
    )
    absolute = np.abs(residual)

    worst: list[dict[str, object]] = []
    flagged = absolute > 0.05
    index = 0
    frames_array = np.array(frames)
    while index < len(flagged):
        if not flagged[index]:
            index += 1
            continue
        start = index
        while index < len(flagged) and flagged[index]:
            index += 1
        worst.append(
            {
                "runA_start": int(frames_array[start]),
                "runA_end": int(frames_array[index - 1]),
                "frames": int(index - start),
                "peak_progress_error": round(float(absolute[start:index].max()), 4),
            }
        )
    worst.sort(key=lambda entry: -entry["peak_progress_error"])

    return {
        "mapped_frames_checked": len(frames),
        "median_progress_error": round(float(np.median(absolute)), 5),
        "p90_progress_error": round(float(np.percentile(absolute, 90)), 5),
        "max_progress_error": round(float(absolute.max()), 5),
        "fraction_over_five_percent": round(float((absolute > 0.05).mean()), 4),
        "worst_regions": worst[:8],
        "caveat": (
            "Image motion is not distance; it is a monotone proxy whose relation "
            "to arc length depends on scene depth. That relation is shared by "
            "both runs because it is a property of the road, which is what makes "
            "the comparison meaningful, but a large residual localises a problem "
            "rather than measuring one in metres."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", choices=tuple(MAPPINGS), default="v2_unified")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    target = OUTPUT_DIR / f"loop_consistency_{arguments.mapping}.json"
    if target.is_file() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild")

    payload: dict[str, object] = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping": arguments.mapping,
        "signal_class": "label-free structural consistency; not an accuracy estimate",
        "angle_closure": (
            "Not implemented. Heading closure is a real constraint but could not "
            "be recovered without camera calibration; see the module docstring "
            "for the four attempts that failed."
        ),
        "cameras": {},
    }

    for camera in ("cam0", "cam5"):
        print(f"[{camera}] loading mapping and descriptors ...", flush=True)
        mapping = load_mapping(MAPPINGS[arguments.mapping], camera)
        descriptors_a = descriptors("runA", camera)
        descriptors_b = descriptors("runB", camera)

        print(f"[{camera}] endpoint closure: do two views of one place agree?", flush=True)
        endpoint = endpoint_closure(camera, mapping, descriptors_a, descriptors_b)

        print(f"[{camera}] path-length closure: does progress transfer?", flush=True)
        progress_a = cumulative_motion("runA", camera)
        progress_b = cumulative_motion("runB", camera)
        path = path_length_closure(camera, mapping, progress_a, progress_b)

        payload["cameras"][camera] = {
            "endpoint_closure": {k: v for k, v in endpoint.items() if k != "cases"},
            "path_length_closure": path,
        }
        atomic_write(
            OUTPUT_DIR / f"revisit_cases_{arguments.mapping}_{camera}.csv",
            lambda p, cases=endpoint["cases"]: _write_csv(p, cases),
        )

    atomic_write(target, lambda path: path.write_text(json.dumps(payload, indent=2)))

    for camera, entry in payload["cameras"].items():
        end = entry["endpoint_closure"]
        path = entry["path_length_closure"]
        print(f"\n=== {camera}")
        print(
            f"  endpoint closure : {end['consistent']}/{end['pairs_with_both_ends_mapped']} "
            f"revisit pairs consistent ({end['revisit_pairs_found']} pairs found)"
        )
        for case in end["worst"]:
            print(
                f"     A{case['runA_first']} and A{case['runA_second']} -> "
                f"B{case['runB_first']} and B{case['runB_second']} "
                f"(gap {case['runB_frame_gap']}, Run B rank {case.get('runB_nonlocal_rank')})"
            )
        print(
            f"  path closure     : median progress error {path['median_progress_error']:.4f}, "
            f"p90 {path['p90_progress_error']:.4f}, "
            f"{path['fraction_over_five_percent']:.1%} of frames over 5%"
        )
        for region in path["worst_regions"][:3]:
            print(
                f"     A{region['runA_start']}-{region['runA_end']} "
                f"({region['frames']} frames, peak {region['peak_progress_error']:.3f})"
            )
    print(f"\nWrote {target.relative_to(ROOT)}")


def _write_csv(path: Path, cases: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for case in cases for key in case})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)


if __name__ == "__main__":
    main()
