# Reproduction guide

This is the order in which the submitted results are produced, with the exact
command for each step. Part A is the current pipeline and is what the report
describes. Part B points to the archived v1 guide, kept because the v1 result is
still quoted in the report alongside the submitted one.

Everything here runs from the repository root after:

```bash
source env.sh
```

`env.sh` defines `RA_PY` (torch, transformers, scikit-image). Edit that
line for your machine; nothing else in the repository names an interpreter path. See
`requirements.txt` for versions and why there are two environments.

External tools: `ffmpeg`/`ffprobe` 8.x on `PATH`; `latexmk` with `pdflatex`
for the report.

## Reproduction boundary: what runs from the bundle, and what does not

The submission bundle excludes the raw HEVC recordings and their sidecars
(confidential), the pretrained model weights (not ours to redistribute), the
descriptor caches and similarity matrices, the extracted frame sets and
annotation filmstrips, and the rendered video. That is a deliberate policy, but
it splits this guide in two, and the split is stated here rather than
discovered:

**Runs from the unzipped bundle, with nothing else installed beyond
`requirements.txt`:**

* every `scripts/test_*.py` (`python scripts/test_x.py`); the tests that read
  exported annotation pages **skip** with a reason, because the pages are not
  shipped;
* `scripts/build_task2_unified.py --camera cam0 --validate-only` — the row
  count falls back to `unified_manifest.json` when the video is absent, and
  says so;
* `scripts/freeze_method_v2.py --verify`, `scripts/freeze_method_v3.py
  --verify`, `scripts/freeze_method_v3.py --set-name blind_v3b --verify` — all
  three hash shipped artifacts only;
* `scripts/build_report_tables.py` and the `latexmk` compile of the report;
* `scripts/evaluate_task2_blind_v3.py --set-name blind_v3b
  --convention-sensitivity --force` — it re-reads scored cases, not frames;
* `scripts/analyze_mapping_kinks.py`, `scripts/evaluate_loop_consistency.py`,
  `scripts/rescore_blind_v2_undetermined.py`,
  `scripts/compare_motion_variant_mappings.py` — all read shipped CSV/JSON.

**Needs the raw video** (`cam0_20_yuv420p_output.hevc` and the three siblings,
plus their timestamp sidecars) **under `runA/` and `runB/`** (the flat root
layout with ` (1)` suffixes for Run B is also accepted by `source_path()`):
`frame_service.py --verify-all`, `hevc_bitstream.py`,
`build_reliability_masks.py`, `detect_route_phases.py`,
`build_frame_quality_flags.py`, `build_confirmed_occlusion.py`,
`estimate_pose_offset.py`, `build_sweep_rate.py`,
`measure_lateral_offset.py`, `render_aligned_video.py`,
`build_blind_label_set_v3.py`, and any `build_task2_unified.py` run other than
`--validate-only`.

**Needs the model weights** (DINOv2-S and SegFormer, downloaded into
`outputs/task2_models/`): the descriptor stage of `build_task2_unified.py`,
`build_task2_bayes.py`, `run_dino_backbone_benchmark.py` and the SegFormer
ablation. `verify_dinov3_meta_checkpoint.py` checks the checkpoint identity.

**Needs the descriptor caches** (regenerable from video + weights, several GB):
`analyze_route_topology.py`, `evaluate_task2_consistency.py`,
`run_geometry_resolution_study.py`, `build_parallax_variant.py`, and the
variant sweeps in A6b. Their *outputs* are shipped, so the numbers they produce
can be read without rerunning them.


Every command below that falls outside the first group names what it needs at
the point of use.

## What must not be rerun, and why

Several artifacts are **frozen** and the numbers in the report depend on their
staying byte-identical:

| Artifact | Why it is frozen |
|---|---|
| `outputs/task2_evaluation/method_freeze_v2.json` | Hashed at 15:41 UTC before any blind label was read. Re-freezing after the labels have been seen is how a held-out number stops being one. |
| `outputs/task2_evaluation/blind_v2/blind_v2_labels.csv` and `label_seal.json` | The seal cites the freeze and the query set; the chain is checkable only while these are untouched. |
| `outputs/task2_unified/{cam0,cam5}/frame_mapping_unified.csv` | The submitted mapping. Its blind score describes this file, not a regenerated one. `runB_frame` is verified byte-identical whenever the writer changes. |
| `outputs/task2_evaluation/method_freeze_v3.json` | Hashed at 07:53 UTC, after the v3 queries were exported and before they were annotated. It hashes file contents, so every artifact it names is frozen too. |
| `outputs/task2_evaluation/blind_v3/blind_v3_labels.csv` and `label_seal.json` | Sealed at 09:14 UTC citing the v3 freeze. Editing either breaks the only chain that makes the v3 numbers held out. |
| `outputs/task2_unified_variants/slope/{cam0,cam5}/frame_mapping_unified.csv`, `outputs/task2_bayes/{cam0,cam5}/frame_mapping_bayes.csv`, `outputs/task2_kinks/**`, `outputs/task3/frame_quality_flags.csv` | Hashed by the v3 freeze, so the v3 labels judge these exact files. |
| `outputs/task2_evaluation/method_freeze_v3b.json` | Hashed at 10:03 UTC for the CAM5 redo, after its queries were exported and before they were annotated. It hashes the **same ten artifacts** as the v3 freeze, so a drift in any scored method fails both verifies. |
| `outputs/task2_evaluation/blind_v3b/blind_v3b_labels.csv` and `label_seal.json` | Sealed at 10:53 UTC citing the v3b freeze. The withdrawn v3 CAM5 labels are frozen too and are **not** deleted: a retracted number stays in the record. |

`scripts/freeze_method_v2.py --verify` refuses to proceed if any parameter in
the live code differs from the frozen record, and
`scripts/freeze_method_v3.py --verify` (add `--set-name blind_v3b` for the redo)
recomputes all ten artifact hashes. Run all three before anything else.

---

# Part A — the current pipeline

## A0. Integrity gates (run first, run always)

```bash
$RA_PY scripts/frame_service.py --verify-all
```

24 gates. Twenty-two must pass. Two are **waived, not passed**, with the reason
recorded in `outputs/task1/integrity_waivers.json`: within each run both
cameras share one byte-identical timestamp sidecar, which cannot describe both.
A gate that fails without a waiver aborts. The gates include:

* decoded frame count equals the independently parsed NAL picture count;
* display order is a verified permutation (74.4% of pictures are reordered by
  the B-pyramid — see `scripts/hevc_bitstream.py`);
* the parked-tail route boundary matches what image motion measures.

Also emits the Task 1 bitstream inventory:

```bash
$RA_PY scripts/frame_service.py --write-bitstream-summary
```

## A1. Reliability masks

Per (run, camera), from temporal statistics, no learned model:

```bash
$RA_PY scripts/build_reliability_masks.py
```

Writes `outputs/task2_masks/{run}_{camera}_mask_<version>.npz` and an overlay
PNG for inspection. The version is a hash of the parameters, and it is part of
every downstream cache key.

## A2. Descriptors and the unified matcher

```bash
$RA_PY scripts/build_task2_unified.py --camera cam0 --stage descriptors
$RA_PY scripts/build_task2_unified.py --camera cam5 --stage descriptors
```

About six minutes for all four streams. Caches are named by source hash,
raster, colour identity, mask version and descriptor version, so a change to any
of them produces a new file rather than reusing a stale one.

```bash
$RA_PY scripts/build_task2_unified.py --camera cam0
$RA_PY scripts/build_task2_unified.py --camera cam5
```

**Do not pass `--force` to these two**: they would regenerate the frozen
submitted mapping. To inspect the effect of verifying geometry only where it was
actually checked, use the separate output:

```bash
$RA_PY scripts/build_task2_unified.py --camera cam0 --strict-geometry
```

Validate any mapping's contract (every ordinal present, typed, monotonic):

```bash
$RA_PY scripts/build_task2_unified.py --camera cam0 --validate-only
```

The geometry working raster was chosen by measurement on the old calibration
labels (development data), and native resolution was found to be *worse*:

```bash
$RA_PY scripts/run_geometry_resolution_study.py
```

## A3. The blind label set

The chain is **export → freeze → seal**, and each link cites the previous.

```bash
$RA_PY scripts/build_blind_label_set_v2.py          # export queries + annotator pages
$RA_PY scripts/freeze_method_v2.py                  # freeze the method, hash it
# ... annotate in the browser (see outputs/task2_blind_v2/README.md) ...
$RA_PY scripts/import_blind_labels_v2.py --cam0 <csv> --cam5 <csv>   # seal
```

The seal records the label-file hash, the query-set hash and the freeze hash.
Page-only rebuilds (`--skip-filmstrip`) preserve the export timestamp. The
annotator carries a difference overlay and a pose-compensation control and
records the alignment convention with every label, because "same place" has two
definitions that diverge when the camera pointing differs between runs.

## A4. Blind evaluation — run once

```bash
$RA_PY scripts/freeze_method_v2.py --verify
$RA_PY scripts/evaluate_task2_blind_v2.py
$RA_PY scripts/compare_methods_blind_v2.py
$RA_PY scripts/rescore_blind_v2_undetermined.py
```

The evaluator refuses to run if the freeze has drifted or the seal does not
match. `compare_methods_blind_v2.py` scores both methods on the same labels.
`rescore_blind_v2_undetermined.py` applies the label-semantics correction
(eleven "no usable match" labels meant "could not determine") and reports both
readings — the correction was made after the results were seen, which the
report states.

Label-criterion drift, measured from the pixels:

```bash
$RA_PY scripts/diagnose_label_criterion.py
```

## A4b. The targeted blind set v3 — build, freeze, seal, score once

The v2 labels are spent, so every claim built after reading them needed a second
set. The chain is the same and one level stricter: **export → freeze → annotate
→ seal → score**, where the freeze hashes the *file contents* of every artifact
the new labels are allowed to judge, not just the code constants.

```bash
# 1. export the queries and the annotator pages (48 queries, 24 per camera)
$RA_PY scripts/build_blind_label_set_v3.py

# 2. freeze what the labels may judge, AFTER the export and BEFORE annotation
$RA_PY scripts/freeze_method_v3.py

# 3. ... annotate in the browser (see outputs/task2_blind_v3/README.md) ...
#    served from outputs/task2_blind_v3/pages, one level below the manifest, so
#    a directory listing cannot reach the strata

# 4. import and seal; the seal cites the v3 freeze hash
$RA_PY scripts/import_blind_labels_v3.py --cam0 <csv> --cam5 <csv>

# 5. score once
$RA_PY scripts/freeze_method_v2.py --verify
$RA_PY scripts/freeze_method_v3.py --verify
$RA_PY scripts/evaluate_task2_blind_v3.py
```

**Order matters and is checkable.** `query_set_exported_at_utc`
(07:43:09Z) precedes `frozen_at_utc` (07:53:32Z) precedes `sealed_at_utc`
(09:14:33Z); the seal carries `method_freeze_v3_sha256` and the label file's
own SHA-256. `evaluate_task2_blind_v3.py` refuses to start unless **both**
freezes verify, the seal's label hash still matches the file on disk, the seal
cites the freeze on disk, and the query manifest is the one the freeze hashed.
It also refuses to overwrite an existing result without `--force`, because
re-running after seeing the number is how a held-out estimate stops being one.

What the v3 freeze hashes (ten files): the submitted unified mapping, the slope
refinement variant, the posterior model and the kink CSVs for both cameras, plus
`outputs/task3/frame_quality_flags.csv` and
`outputs/task2_kinks/mapping_kinks.json`. Only these have held-out v3 numbers.
Anything built after 07:53:32Z does not, and `--verify` fails if any of them
changed.

Outputs: `outputs/task2_evaluation/blind_v3/blind_v3_results.json`,
`blind_v3_results.md` (headline table, per-stratum breakdown, the kink-claim
test and the pose-convention check) and `blind_v3_cases.csv` (one row per label
per method, carrying the raw prediction, which gate demoted it, and the scoring).

Six methods are scored on the same labels: the submitted mapping, the slope
refinement variant, the posterior model, and three policies over the frozen
submitted mapping — `slope gate` (rows flagged `near_kink` in
`frame_quality_flags.csv` become abstentions), `dark gate` (DATA_POLICY.md's
exposure rule read as an abstention) and both. The gate policies change no
mapping row.

## A4c. The CAM5 redo (blind v3b) — same artifacts, corrected tool, scored once

The CAM5 half of v3 was annotated with a viewer that applied the pose
compensation with the wrong sign: the measured displacement was passed through
as the compensation instead of its negation, so the page moved Run B a further
40 px in the wrong direction and the annotator recovered with the slider, which
let frame error and shift trade off. The defect was found *after* the seal, from
the sealed labels themselves — the slider value is written into every row — and
is recorded in `outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`. CAM0's
measured offset is 0/0, so CAM0's v3 result is unaffected and stands.

The redo keeps every method parameter fixed and changes only the tool and the
queries. `freeze_method_v3.py --set-name blind_v3b` hashes **the same ten
artifacts** as the v3 freeze against a fresh query set, so a drift in any scored
method would fail the verify.

```bash
# 1. export 24 fresh CAM5 queries (none shares a Run A ordinal with v3, and each
#    is >= 10 frames from every previously labelled ordinal in v2, v3, the manual
#    annotations and the anchor candidates)
$RA_PY scripts/build_blind_label_set_v3.py --set-name blind_v3b --camera cam5

# 2. freeze the same artifacts against the new query set, before annotation
$RA_PY scripts/freeze_method_v3.py --set-name blind_v3b

# 3. ... annotate in the browser (see outputs/task2_blind_v3b/README.md) ...
#    the pose slider now starts at the compensation (+40, +12) px, not the raw
#    measurement, and all 24 rows were left at that default

# 4. import and seal; the seal cites the v3b freeze hash
$RA_PY scripts/import_blind_labels_v3.py --set-name blind_v3b --cam5 <csv>

# 5. score once
$RA_PY scripts/freeze_method_v2.py --verify
$RA_PY scripts/freeze_method_v3.py --verify
$RA_PY scripts/freeze_method_v3.py --set-name blind_v3b --verify
$RA_PY scripts/evaluate_task2_blind_v3.py --set-name blind_v3b
```

**Order:** `query_set_exported_at_utc` (09:57:53Z) precedes `frozen_at_utc`
(10:03:08Z) precedes `sealed_at_utc` (10:53:30Z); the seal carries
`method_freeze_sha256_at_seal_time` (`b98045635d7a`) and the label file's own
SHA-256. `--set-name` decides which manifest, freeze, seal and label file are
read and where the results are written; the contract check refuses a seal whose
`label_set` is not the one asked for.

Outputs: `outputs/task2_evaluation/blind_v3b/blind_v3b_results.json`,
`blind_v3b_results.md` and `blind_v3b_cases.csv`. Two readings are added for
this run and computed for any set: a **signed** interval error (is the
prediction later or earlier than the label interval — the direction is what
exposed the tool defect) and a comparison of CAM5 against the v2 set,
recomputed from `blind_v2_cases.csv` rather than copied from a summary.

**Convention-shift sensitivity (a reading, not a second scoring).** The CAM5
error on v3b is one-directional, and its size was measured before the set
existed: `pose_offset_diagnostic.json` puts the gap between the matcher's "same
place" and the label's world position at **-1.79 frames** on CAM5 and **0.0** on
CAM0. `--convention-sensitivity` re-reads the already-written
`blind_v3b_cases.csv`, applies shifts of 0, -1, round(constant) and -3 to the
accepted CAM5 predictions of the three scored mappings, and writes its own file.
It **never** re-runs the scoring and **never** rewrites
`blind_v3b_results.json`; it records that file's SHA-256 so the reader can see
which pass it read. The constant is read from the diagnostic, never typed - a
unit test points the module at a fixture with a different value and asserts the
rounding follows it.

```bash
$RA_PY scripts/evaluate_task2_blind_v3.py --set-name blind_v3b \
    --convention-sensitivity
```

Output: `outputs/task2_evaluation/blind_v3b/convention_shift_sensitivity.json`.
All three mappings peak at -2, the rounded pre-measured constant (shipped
mapping 0.143 -> 0.667, Wilson 0.45-0.83). This is a **sensitivity reading on a
spent set**, not a held-out number, and the submitted mapping is not shifted;
the constant is attached as per-frame metadata instead (`DATA_POLICY.md`).

```bash
# both freezes and both seals, at any time
$RA_PY scripts/freeze_method_v3.py --verify
$RA_PY scripts/freeze_method_v3.py --set-name blind_v3b --verify
$RA_PY -m pytest scripts/test_evaluate_task2_blind_v3.py -q
```

## A4d. Task 1: is there a lateral offset between the runs?

A near/far scale test on RootSIFT pairs: a lateral shift changes the apparent
scale of near-band content much more than far-band content, and the two
opposite-facing cameras must disagree in sign at the same place if the vehicle
sat in a different lane.

```bash
$RA_PY scripts/measure_lateral_offset.py
$RA_PY -m pytest scripts/test_measure_lateral_offset.py -q
```

Outputs: `outputs/task1_lateral/lateral_offset.json` (per-camera stretches,
per-window detection limits, the cross-camera sign test), `lateral_offset_pairs.csv`
and `outputs/task1_lateral/README.md`. The answer is an **upper bound, not a
list of places**: the flagged excursions fail a permutation test for persistence
and the two cameras agree in sign on 64.7% of 34 paired samples
(p = 0.061), which is not beyond chance. The report quotes it that way.

## A4e. The tolerance reading of every sealed set

```bash
$RA_PY scripts/evaluate_tolerance_curve.py
```

Reads the three sealed label files and the frozen mappings once and reports
precision at ±0/±1/±2/±5/±10 frames (`outputs/task2_evaluation/tolerance_curve.json`).
The report's headline is the ±2-frame column (0.2 s, about 2 m, still an
order of magnitude tighter than published place-recognition tolerances); the
strict ±0 column is the report's last table. Nothing is re-scored.

## A5. Label-free evidence

None of these touches a label. They are the only numbers comparable across
methods without spending a held-out set.

```bash
$RA_PY scripts/evaluate_task2_consistency.py      # cross-camera agreement
$RA_PY scripts/evaluate_loop_consistency.py --mapping v2_unified
$RA_PY scripts/run_abstention_holdout.py          # withhold Run B stretches
```

`evaluate_loop_consistency.py` scores endpoint closure against revisit pairs
confirmed by both cameras, and path-length closure against cumulative image
motion. `run_abstention_holdout.py` defines orphaned frames from the unified
matcher's alignment, not the model under test's, and reports the model's own
definition alongside so the circularity is visible.

## A5b. Per-frame quality flags and the confirmed occlusion

Two failure modes found by looking at frames rather than at summaries. Both are
label-free, and both only report: neither rewrites a mapping row.

```bash
$RA_PY scripts/build_frame_quality_flags.py   # one decode pass per stream
$RA_PY scripts/build_confirmed_occlusion.py   # reads the flags and both mappings
```

`build_frame_quality_flags.py` writes `outputs/task3/frame_quality_flags.csv`
(mean and p99 Rec. 709 luma per frame at 192x144, plus the `too_dark` verdict)
and a JSON summary that also cross-references the submitted mapping and the
blind labels. Re-derive the flag at a different threshold from the CSV rather
than editing the constants; `--cross-reference-only` rebuilds the summary from
an existing CSV without decoding again.

`build_confirmed_occlusion.py` records the lorry occlusion in Run B CAM0 and
what each mapping does inside it. Its interval is read off a contact sheet by
eye, so it is a single confirmed positive, not an annotated set: no rate may be
computed from it.

## A6. The posterior model (built, not submitted)

```bash
$RA_PY scripts/build_motion_events.py             # shake channel, measured LR
$RA_PY scripts/build_task2_bayes.py --camera cam0
$RA_PY scripts/build_task2_bayes.py --camera cam5
$RA_PY scripts/evaluate_task2_bayes.py            # calibration, ablations
$RA_PY scripts/sweep_null_priors.py               # priors on the synthetic harness
$RA_PY scripts/evaluate_loop_consistency.py --mapping bayes
```

Ablations: `--no-motion`, `--no-structure`, `--no-loop-closure`. Every number
from this model is a development diagnostic; the blind labels are spent.

`build_motion_events.py` defaults to `--static-removal mean-divide`, so the plain
`motion_*.npz`, `motion_event_summary.json` and `frame_mapping_bayes.csv` carry
that construction. The withdrawn raw-frame construction is still buildable with
`--static-removal none` and `build_task2_bayes.py --motion-variant none`, which
write `_none` and `_motion-none` files beside the defaults.

**The static-removal study.** The suspension channel was corrected after a
label-free physical test showed the shipped construction was not measuring the
vehicle: CAM0 and CAM5 are on one chassis, yet 0 of 28 Run B events co-fired
across the two cameras at any inter-camera lag. Dividing by each stream's
temporal mean gives 4/33 on Run B and 9/45 on Run A, 3.85x and 4.75x chance. The
study scores all three modes; the second script asks the separate question of
whether the corrected channel moves the mapping it feeds (it does not: 95-98%
of predictions are identical).

```bash
$RA_PY scripts/build_motion_events.py --static-removal none --force
$RA_PY scripts/build_motion_events.py --static-removal median-subtract --force
$RA_PY scripts/run_static_removal_study.py --force --check-pinning-premise
$RA_PY scripts/build_task2_bayes.py --camera cam0 --motion-variant none --force
$RA_PY scripts/build_task2_bayes.py --camera cam5 --motion-variant none --force
$RA_PY scripts/compare_motion_variant_mappings.py --variant mean-divide --force
```

`--check-pinning-premise` costs a second decode pass per stream and re-measures
the audit figure the study was launched on (91% of Run B CAM0 pairs pinned at
`dx = 0`); it does not reproduce, and `pinning_premise` in
`outputs/task2_motion_bayes/static_removal_study.json` records what four
readings of "whole-frame phase correlation" actually give.

## A6b. Development variants (after the blind run; none is submitted)

Each writes to its own directory and its own descriptor cache; the frozen
mapping and the freeze verifier are unaffected. `evaluate_variants.py` then
scores them on the label-free signals, with the spent labels as a marked
diagnostic.

```bash
$RA_PY scripts/build_reliability_masks.py --variant sky          # sky excluded by temporal statistics
$RA_PY scripts/build_task2_unified.py --camera cam0 --greyscale   # luminance-only DINOv2
$RA_PY scripts/build_task2_unified.py --camera cam0 --mask-variant sky
$RA_PY scripts/build_task2_unified.py --camera cam0 --mask-variant sky --greyscale
$RA_PY scripts/build_sweep_rate.py                                # per-frame sweep rate, ~15 min per stream
$RA_PY scripts/build_task2_bayes.py --camera cam0 --speed          # speed-profile channel
$RA_PY scripts/build_task2_unified.py --camera cam0 --aggregation vlad   # unsupervised VLAD pooling, ~10 min per camera
$RA_PY scripts/evaluate_variants.py
$RA_PY scripts/analyze_mapping_kinks.py --no-decode --output-dir outputs/task2_kinks_variants   # kinks per variant, away from the frozen kink files
```

`--aggregation vlad` caches the masked DINOv2 patch tokens once per stream
(float16, under `outputs/task2_unified/<camera>/`), fits a 32-word k-means++
vocabulary per camera from both runs, and aggregates with intra-normalised
residuals (`scripts/vlad_aggregation.py`). It is the one descriptor-side
variant that moved the label-free signals (cross-camera p95 11 → 8, CAM5
coverage 0.816 → 0.898); see README.

`build_parallax_variant.py` re-chooses each sampled Run B partner by a
depth-uniformity criterion instead of by appearance alone. Both cameras face
sideways, so a camera rotation between the runs shifts near and far scene
content by the same number of pixels while driving further along the road
shifts near content more. Among the Run B frames within +/-5 of the baseline
partner it therefore keeps the one whose RootSIFT displacement field is flattest
against image row (the depth proxy: bottom third near, top third far, at least
12 matches in each band), scores it as
`|median dx(near) - median dx(far)|`, interpolates the chosen per-sample offset
along Run A and re-imposes a non-decreasing Run B.

```bash
$RA_PY scripts/build_parallax_variant.py --workers 6   # ~20 min for both cameras
$RA_PY scripts/evaluate_variants.py --force
```

It matches at ratio 0.90 rather than the shipped 0.80. That is the one
parameter that differs and it was set label-free: at 0.80 only 12% of CAM5 and
24% of CAM0 candidates reach 12 matches in both bands, so the criterion cannot
be measured at all, while at 0.90 (cross-checking kept) it is 100% and dx
against candidate offset is at least as straight a line. The shipped constant is
untouched; `outputs/task2_unified_variants/parallax/parallax_manifest.json`
records the measurability figures, the delta distribution, dx at the baseline
and chosen partners, and the caveat that a fisheye yaw is not in fact a uniform
pixel shift, so the criterion's premise holds only to first order.

The result is negative and the manifest says why. The gradient itself is real
and consistent: the signed near-minus-far difference moves +2.8 px per frame on
CAM0 and -4.7 px on CAM5, opposite signs for cameras facing opposite ways. But
its per-sample scatter is 23 px (CAM0) and 78 px (CAM5), so one sample resolves
only 8 and 17 frames of travel against a +/-5 search window: the argmin over the
window is noise, and on CAM5 the chosen offsets are distributed almost exactly
as the uniform null (8.0% at the baseline against the 9.1% that chance alone
gives). The aggregate, which does not share that selection bias, is the usable
reading and says the shipped partners already sit at the depth-uniform position
to within -1.18 +/- 0.73 frames on CAM0 and -0.64 +/- 1.52 on CAM5. So the
CAM5 residual dx of -37.5 px at the matcher's pairs looks like camera pose, not
travel, which is what `estimate_pose_offset.py` could not separate. Scored as a
mapping the variant is worse than the baseline on every label-free signal
(cross-camera residual 1.0 -> 3.0 frames, CAM0 closure 4/4 -> 1/4), as it should
be: it adds a several-frame random walk to a mapping that was already right.

### The slope refinement, and the kink audit that motivated it

`analyze_mapping_kinks.py` audits every mapping on disk for one-step Run B
increments the mapping's own motion model forbids, cross-checks them against
cross-camera disagreement and against the spent blind labels, costs the slope
gate as a Task 3 policy, and reproduces the mechanism from the cached
descriptors. It is label-free except for sections 3 and 4 of its README, which
read the spent labels and tune nothing.

```bash
$RA_PY scripts/analyze_mapping_kinks.py                  # ~2 min; --no-decode skips the stage-3 replay
$RA_PY scripts/build_frame_quality_flags.py --slope-gate # writes the slope_gate column, no decode
```

`build_task2_unified.py --refinement slope` is the fix, as a development
variant: stage 3's greedy argmax behind a non-decreasing floor becomes a Viterbi
over the same +/-8 windows and the same structure descriptors, with a transition
cost of `REFINE_SLOPE_WEIGHT = 0.05` per frame of departure from the coarse
path's own local increment and monotonicity enforced as a hard step bound of
`[REFINE_MIN_STEP, REFINE_MAX_STEP] = [0, 3]`. The default is `ratchet`, which is
the shipped behaviour byte for byte, so the freeze verifier and the frozen
mapping are unaffected; the DINOv2 descriptor cache is shared, because the
refinement rule does not touch the descriptors. Everything downstream - stage 4
margins, stage 5 geometry, tiers, bridging, the writer - is identical.

```bash
$RA_PY scripts/build_task2_unified.py --camera cam0 --refinement slope  # ~12 min per camera
$RA_PY scripts/build_task2_unified.py --camera cam5 --refinement slope
$RA_PY scripts/analyze_mapping_kinks.py                                  # picks the variant up automatically
$RA_PY scripts/evaluate_variants.py --force
```

Its kink count is zero by construction, so read the label-free columns instead:
cross-camera p95 falls 11 -> 9 frames at unchanged coverage, and over Run A
880-940 the variant's picks score a median structure cosine of 0.60 against the
submitted mapping's 0.52 on the same descriptors (at least as good on 50 of 61
frames, equal to the unconstrained argmax on 36). CAM0 endpoint closure falls
4/4 -> 2/4 and CAM5's strict spent-label precision falls; both are in
`outputs/task2_variants/variant_comparison.json`. The variant is evidence about
the defect, not a submission: the blind labels are spent, so no method fixed
after reading them can carry a held-out number.

`build_sweep_rate.py` measures how many pixels the scene moves per frame with
the same RootSIFT detector and raster that measure scene displacement at
matched pairs. Whole-frame and tiled phase correlation were tried first and
failed validation against it (Spearman 0.05 to 0.48): on a side-facing fisheye
the apparent speed depends on depth, so only the detector that measures the
displacement can measure the rate it is divided by. The failure is recorded in
`outputs/task2_pose_offset/pose_offset_diagnostic.json` under
`dense_speed_check` for the phase-correlation runs that preceded it.

The camera pose question for CAM5 (its camera pointed slightly differently in
the two runs) is measured, not assumed:

```bash
$RA_PY scripts/estimate_pose_offset.py
```

It reports where the matcher puts "same place", where the labels put it, and
whether suspension events supply a world anchor. A yaw offset and a few frames
of travel are indistinguishable for a side camera, which the script says. Where
the two conventions differ by a constant it writes
`frame_mapping_unified_label_convention.csv`, the submitted mapping moved to
the labels' convention by that constant over the local sweep rate, and checks
by re-measuring dx at the moved pairs that the conversion did what it claims.
The constant was measured on the spent labels, so its rescoring is in-sample
for one degree of freedom; the re-measurement is the label-free evidence.

## A7. Report and bundle

```bash
$RA_PY scripts/build_report_tables.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/pdf report/route_alignment_report.tex
$RA_PY scripts/build_submission.py --force
```

Report tables are generated from the JSON artifacts and pulled in with
`\input`, so a number cannot change in one place and not the other. The
bundler refuses to zip a PDF older than its tables, and refuses to ship any
script whose local imports are absent from the bundle.

## A8. Tests

```bash
for t in scripts/test_*.py; do $RA_PY "$t"; done
```

Eighteen files, stdlib `unittest`, synthetic inputs only. The posterior model's
tests are where the null-state acceptance threshold was chosen; the quality-flag
tests fix the arithmetic of the exposure rule on constructed frames, including
the uniform dim frame that a mean-only floor would pass.

---

# Part B — the legacy v1 pipeline

The sparse-anchor method whose frozen result the report quotes as the
comparison point. Its guide is preserved unchanged in
`archive/REPRODUCTION_AND_EXPERIMENTS_v1_archive.md`. In this repository its scripts sit
beside the current ones in `scripts/`; the bundle moves them to `scripts/legacy/`.
They read 640×480 proxies and ordinal-keyed caches; do not
mix their outputs with Part A's.
