# Route alignment: matching two traversals of one loop route, frame by frame

Two dashcam drives of the same loop route, recorded eleven months apart (a dry
afternoon and a rainy dusk), from two side-facing fisheye cameras at 10 FPS,
with no GPS, no odometry and no camera calibration. For every frame of the
first drive: which frame of the second drive shows the same place, and how sure?

This repository is the complete working of a take-home assessment I did for a
robotics company in September 2026, cleaned for publication. The recordings
themselves were provided for the assessment and are **not** redistributed; the
pipeline runs on any pair of side-camera traversals in the same format.

## What is here

| | |
|---|---|
| `report/route_alignment_report.pdf` | the 4-page report: findings, method, results, failure account, policy |
| `report/slides/interview_slides.pdf` | the interview deck |
| `assessment.ipynb` | the report's working, section by section: every number recomputed from the shipped artifacts by calling the scripts, with executed outputs |
| `mappings/` | the submitted frame mappings, one CSV per camera: `runA_frame, runB_frame, confidence, tier, status` |
| `scripts/` | the pipeline, evaluation, and unit tests (Python; stdlib `unittest`); `scripts/README.md` indexes them by role and explains the `v2`/`v3`/`v3b` suffixes (blind label rounds, not code versions) |
| `outputs/` | evidence: bitstream inventory, integrity gates, masks, three sealed blind label sets with freezes and seals, tolerance curves, kink audit, quality flags |
| `DATA_POLICY.md` | what a downstream consumer may do with the mapping, and what is attached to every frame |

## The approach in one paragraph

Frame identity first: the streams are raw H.265 in which 74% of pictures are
stored out of display order and the colour matrix is undeclared, so a
purpose-written Annex-B parser rebuilds the display order and every decode is
pinned and gated. Camera-attached artefacts (rain fixed on the lens, the wing
mirror, the vignette) are masked from temporal statistics, with no learned
model. Frames are described with frozen DINOv2 features pooled over the
trusted patches, compared across the whole other drive, and aligned with a
global monotonic dynamic-time-warp whose slope bounds come from the measured
drift. The path is refined to full frame rate, tested for sequence uniqueness
at four window scales, verified geometrically with RootSIFT/RANSAC, and every
frame gets a typed decision: a match with an ordinal confidence, or *no answer
under this method* with the reason.

## How it was evaluated

There is no position reference in the data, so the instrument is a person:
three prediction-blind label sets (40, 24 and 24 queries per camera), each
sealed after the method was frozen and scored once. Two defects in the
instrument itself were found and are reported with their timing. Precision at
frame tolerances on the first set (one frame = 0.1 s, about one metre):

| Camera | Coverage | ±1 | ±2 | ±5 | ±10 | exact frame (Wilson 95%) |
|---|---:|---:|---:|---:|---:|---|
| left (cam0) | 100% | 81.2% | **84.4%** | 90.6% | 96.9% | 71.9% (0.55–0.84) |
| right (cam5) | 86.5% | 56.2% | **68.8%** | 90.6% | 96.9% | 37.5% (0.23–0.55) |

Published place-recognition protocols count a hit at 25 m or ±10 frames at
1 FPS; ±2 here is about 2 m. The right camera's residual has a measured
mechanism (a change of mount angle between the drives), and the failure cases
(dark stretches, an occluding lorry, refinement jumps) are documented with their
causes rather than smoothed over.

## Verify

```bash
source env.sh                                  # RA_PY: a Python with torch, transformers, scikit-image
$RA_PY scripts/freeze_method_v2.py --verify
$RA_PY scripts/freeze_method_v3.py --verify
$RA_PY scripts/freeze_method_v3.py --set-name blind_v3b --verify
$RA_PY scripts/build_task2_unified.py --camera cam0 --validate-only
for t in scripts/test_*.py; do $RA_PY "$t"; done
```

Rebuilding the mapping needs the raw recordings and the DINOv2-S weights;
`REPRODUCTION_AND_EXPERIMENTS.md` lists every command in order.

## Data and privacy

The recordings are the assessment provider's property and are not included.
The handful of frames shown in the slides and figures have number plates and
faces blurred. `DATA_POLICY.md` records the privacy handling required before
any frame leaves an archive.
