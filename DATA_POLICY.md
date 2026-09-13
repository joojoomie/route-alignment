# Downstream data policy

## Purpose and claim boundary

This policy answers Task 3 for place recognition and cross-traversal frame
alignment. It does not prescribe what should be discarded for unrelated
perception tasks such as vehicle detection. “Discard” below means exclude from
a derived training or test view; the supplied raw files remain immutable audit
evidence.

The current data cannot support an unseen-route or production-safety test. It
contains only two dates on one route, and the timestamp rows cannot be linked
uniquely to decoded images. The existing calibration/held-out split is therefore
an interview-scale within-route evaluation only.

## Concrete disposition for the supplied recordings

| Run / camera | Decoded images retained in raw archive | Exclude from route-alignment view | Route-region images eligible for QC/sampling | Non-redundant development representatives | Frozen automatic mapping rows |
|---|---:|---:|---:|---:|---:|
| Run A / CAM0 | 2,674 | 180 before route start | 2,494 | 295 | 2,401 mapped by the submitted v2 method (v1 sparse: 1,071) |
| Run A / CAM5 | 2,674 | 180 before route start | 2,494 | 304 | 2,183 mapped by the submitted v2 method (v1 sparse: 287) |
| Run B / CAM0 | 2,622 | 114 before route start | 2,508 | 297 | gallery axis, not a Run A query output |
| Run B / CAM5 | 2,615 | 114 before route start | 2,501 | 301 | gallery axis, not a Run A query output |
| **Total** | **10,585** | **588** | **9,997** | **1,197** | report cameras separately |

The route starts are measured heuristic boundaries, not exact physical
departure times. They are suitable for excluding the long stationary prefix
from route alignment, while the original 588 images remain available for
audit or stationary-scene tests.

### Keep

- Keep all 10,585 canonical decoded images in the immutable source archive.
- In the route-alignment view, keep the 9,997 images at or after the
  camera-pair route gate, subject to quality flags rather than silent deletion.
- Keep the sunny/rainy appearance difference. It is valuable domain variation,
  not a quality defect.
- Keep human-reviewed static correspondences with their acceptable frame
  interval, reviewer status, and split. Held-out labels are evaluation-only.
- Keep changed, occluded, and unsupported frames as explicit challenge or
  unlabelled cases. Do not silently turn them into negative pairs.
- Keep route `location_group` separate from temporal `occurrence`. For the CAM0
  endpoint-area hypothesis, departure and arrival may share a location group
  but must remain different ordered occurrences.

### Exclude from the default route-alignment training/test view

- Exclude the 588 pre-route and departure-transition images. Their physical
  location changes little and they would over-represent a stationary scene.
- Exclude any future decode failure or mid-stream resolution change. None was
  observed in these four streams.
- Exclude vehicles, people, cyclists, and temporary objects from the matching
  representation when the semantic mask is used. Do not delete the whole image
  solely because the mask is imperfect.
- Exclude held-out route blocks and their labels from component selection,
  threshold selection, confidence calibration, and training.
- Exclude every automatic mapping row from benchmark ground truth unless it is
  independently reviewed. Bounded interpolation is weak supervision, not a
  measured label.

### Downweight or quarantine

- Collapse near-duplicate temporal neighbours only inside one already-assigned
  split. Preserve the representative ordinal, temporal-cluster identifier,
  cluster size, and a default sampling weight of 1/cluster_size. The existing
  development selector retains at most a 10-frame gap and is an annotation
  aid, not evidence that discarded neighbours are corrupt.
- Treat geometry-supported automatic anchors as unverified weak positives.
  Treat bounded interpolation rows as lower-authority weak positives.
- Quarantine all supported segments that contain a known blind false accept
  from supervised-pair training until independent frame-level review:

  | Camera | Segment | Known frozen failure |
  |---|---|---|
  | CAM0 | A611–677 → B515–574 | cam0_A07 |
  | CAM0 | A890–1035 → B779–935 | cam0_A12 |
  | CAM0 | A1748–2534 → B1688–2545 | cam0_A24 |
  | CAM5 | A1747–1888 → B1688–1838 | cam5_A23 |

- The remaining supported segments are still unverified weak labels; absence of
  an observed failure is not proof of correctness.
- Use an empty Run B frame as “abstain / no usable correspondence under this
  method,” not as a hard negative or proof that the place is absent. Only an
  independently source-reviewed no-match may become a negative label.

## Per-frame quality flags that gate the top tier

Three failure modes were invisible to every aggregate statistic in this project.
Two are visible in the frames; the third is visible in the mapping itself. All
three are measured per row or per frame, and all three demote a pair out of the
highest-authority tier rather than deleting it.

### Exposure: `too_dark`

`scripts/build_frame_quality_flags.py` measures mean and 99th-percentile Rec. 709
luma of every decoded frame at 192x144 and flags

    too_dark  <=>  mean_luma < 12  or  p99_luma < 40   (of 255)

Two thresholds, because a frame can hold a respectable mean and still be
featureless, or a low mean and still be usable if a lit shopfront occupies a
corner. Measured counts (`outputs/task3/frame_quality_flags.csv`, summarised in
`frame_quality_flags.json`):

| Stream | `too_dark` frames | Fraction | Longest contiguous stretch |
|---|---:|---:|---|
| Run A / CAM0 | 48 / 2,674 | 1.8% | 1574-1594 (21) |
| Run A / CAM5 | 0 / 2,674 | 0.0% | none |
| Run B / CAM0 | 304 / 2,622 | 11.6% | **1460-1636 (177)** |
| Run B / CAM5 | 136 / 2,615 | 5.2% | 350-421 (72), 424-487 (64) |

Run B CAM0 1361-1636 is a canopied unlit road at dusk in rain; at 720x540 a
2.4x brightness boost recovers only a tree trunk and a kerb line. Both sides of
a pair there are dark, so a dark-against-dark descriptor match is close to
guaranteed. That is a mechanism for *systematic* high-confidence error, not
random error, and the submitted mapping has no column for it: **273 of 2,401
accepted CAM0 rows (11.4%) and 147 of 2,183 accepted CAM5 rows (6.7%) have a
Run A or Run B partner flagged `too_dark`**, all of them `accepted_strong`.

The blind labels reach these regions and behave accordingly. On CAM5 all three
labels inside the flagged stretch B350-487 are wrong, missed by 3, 8 and 16
frames; on CAM0 five labels fall in dark regions, of which two were undetermined,
two correct and one wrong by 14 frames. Eight labels cannot measure a rate, but
they are consistent with the mechanism rather than against it.

**Rule.** Any correspondence whose Run A or Run B frame carries `too_dark` is
capped below the top tier and enters the review queue. Do not delete the frames:
they are legitimate night-driving domain variation and the flag is recomputable
at any threshold from the CSV, which keeps both raw measurements beside the
verdict.

### Occlusion: `occluded`

`outputs/task2_abstention/confirmed_occlusion_runB_cam0.json` records the one
confirmed no-correspondence case in this data, found by visual inspection and
not by a label: an oncoming box lorry fills the whole Run B CAM0 frame at
ordinals 247-252 and a passing car covers about two thirds of it at 258-270. For
that stretch the descriptors describe a vehicle's side panel, so no Run A frame
has a counterpart at all over the core.

Both mappings map straight through it. The submitted mapping sends Run A
327-356 into B240-270 as 30 `accepted_strong` rows at confidence 1.00; the
posterior model, which has an explicit null state, sends the same 30 rows with a
no-correspondence probability of at most 0.0002. The exposure flag independently
fires over most of the interval, because the lorry panel is dark — one flag
partially covers the other failure mode, which is luck, not design.

**Rule.** Carry an `occluded` flag with the same demoting effect as `too_dark`,
sourced from review rather than from a detector until one exists. This instance
is a single confirmed positive, adequate as a seed for the three-way annotation
protocol the failure analysis asks for and inadequate for any rate.

### Slope: `slope_gate`

A mapping's first difference is a local speed ratio: consecutive Run A frames are
0.1 s apart, so mapping them to Run B `b` and `b + d` asserts that Run B covered
in `d` frames what Run A covered in one. The pipeline's own coarse search is
bounded to a sustained slope of 2.0 with a largest single step of 3.0, so
`d >= 6` is not reachable by the model that produced the path, and it cannot be
a stop either — a stop makes Run B *repeat* frames (`d = 0`).

The submitted mapping contains 27 such jumps on CAM0 (17 lit, 10 dark) and 25 on
CAM5 (19 lit, 6 dark), **every one `accepted_strong` at confidence 1.00**. The
Bayes posterior, which carries an explicit transition model, contains one per
camera. On CAM0, 20 of the 27 sit within 10 Run A frames of a cross-camera
disagreement over 6 frames — an independent, label-free corroboration — and the
sealed blind labels miss more often near them: 0.50 within 15 frames of a jump
against 0.21 away (4 of 8 against 5 of 24 labels; suggestive, not a rate).
A kink is not proof of an error. It is a label-free statement that the mapping
asserts something its own motion model forbids, which is reason enough to
distrust the neighbourhood.

**Rule.** Demote out of the top tier any row within **10 Run A frames** of a
one-step Run B increment `>= 6`. `scripts/build_frame_quality_flags.py
--slope-gate` writes this as the `slope_gate` column of
`outputs/task3/frame_quality_flags.csv` (`near_kink` on the Run A ordinals of the
camera whose mapping kinks), and `outputs/task2_kinks/` carries the audit that
sets the threshold and the radius.

Cost, measured: 454 of 2,366 top-tier CAM0 rows (19.2%) and 320 of 1,881 on CAM5
(17.0%). On the **spent** blind labels — they already scored the frozen method,
so this is a diagnostic and not a held-out estimate — the gate moves CAM0 from
0.575 precision at 1.00 coverage to 0.625 at 0.80, and CAM5 from 0.343 to 0.367
at 0.875 → 0.75. Five of the eight CAM0 labels it demotes were false accepts.

The gate is a policy over the frozen mapping and changes no mapping row. The
underlying defect is in stage 3 of the pipeline and is fixed in a development
variant (`build_task2_unified.py --refinement slope`), which is *not* what is
submitted: see `outputs/task2_evaluation/blind_v2/FAILURE_ANALYSIS.md`.

## Personal data

The recordings contain identifiable personal data. Earlier versions of this
policy answered Task 3 purely as "which frames help matching" and did not
mention it; that was an omission, not a judgement that there is nothing to
handle.

**What is present**, established at native resolution from the decoded frames:

- **Legible licence plates.** In the stationary car-park segments the rear
  plates of parked vehicles resolve to readable characters. Run A is parked for
  its first 173 frames and last 57, so the same ten to fifteen residents' cars
  and their plates are recorded for roughly 23 seconds in a well-lit static
  view. This is the highest-density PII in the set, and it sits inside the 588
  pre-route images this policy keeps in the archive.
- **Identifiable faces, including children.** Run A CAM0 around ordinal 910
  shows two school-age children in uniform at three to five metres, the nearer
  face about 70 px and clearly recognisable; Run A CAM5 340 a man crossing at
  close range; Run A CAM0 600 a group of five pedestrians. People appear in
  roughly 2-6% of frames, of order 60-160 frames per stream, with close-range
  appearances in about 1%.
- **Private frontages throughout.** Both cameras face sideways, so the whole
  route looks into driveways, gates, porches and balconies with occupancy cues.
  A side-facing rig captures fewer plates than a forward camera and
  substantially more residential and pedestrian detail.

**Jurisdiction.** Malaysia's Personal Data Protection Act 2010 governs the
recordings (Malay signage, a road sign reading "Kawasan SD 13" in Bandar Sri
Damansara). GDPR/UK GDPR applies additionally if any downstream processing,
storage or model training happens in the EU or UK.

**Disposition.**

1. The immutable source archive keeps the unmodified frames as audit evidence,
   with access restricted and logged.
2. Every view that leaves that archive — training sets, annotation pages,
   figures, this repository's contact sheets — carries plate and face blurring
   applied before export. Blurring is a derived-view operation and is recorded
   in the manifest, so a consumer can never mistake a blurred frame for the
   raw one.
3. The stationary car-park segments (Run A 0-172 and 2611-2673, Run B 0-42 and
   its tail) are flagged as the highest PII density in the set and are already
   excluded from the route-alignment view; they must not be reintroduced as
   "free stationary training data" without redaction.
4. Face and plate detectors are themselves imperfect. Their recall is a
   documented residual risk, not a completion criterion, and a manual review of
   the flagged stationary segments is cheaper than a route-wide one.
5. Publication of individual frames outside the assessment context requires
   redaction of plates and faces regardless of the detector's verdict.

## Split policy and leakage prevention

### Current assessment

No task-specific model was trained. Frozen pretrained DINOv2 and SegFormer
weights were used. Per camera, six contiguous Run A route blocks define 24
calibration queries in blocks 0, 2, 4, and 5 and 12 prediction-blind held-out
queries in blocks 1 and 3. Labels selected components and thresholds only on
calibration; each camera's held-out run occurred once after method freeze.

This split prevents direct label leakage, but it is not a strict inductive
test: both splits come from the same route and dates, unsupervised candidate
statistics saw the full route, and Run B is a shared retrieval gallery.

### Production data

1. Assign split before frame sampling, descriptor extraction, normalization,
   threshold selection, or augmentation.
2. Group by route identity, geography, traversal date, vehicle/rig, and all
   paired cameras. Group loop-equivalent departure/arrival locations together
   before assigning a split, even though their temporal occurrences remain
   distinct. Every member of a group belongs to one split.
3. Hold out complete routes and dates for validation and test. Never randomize
   adjacent frames across splits and never count CAM0/CAM5 views from one
   traversal as independent examples.
4. If a within-route development split is unavoidable, use contiguous spatial
   blocks and remove a boundary buffer. Without GPS, use at least 50 decoded
   ordinals on each side as a documented nominal five-second heuristic; do not
   describe it as a measured distance or capture duration.
5. Fit de-duplication thresholds, confidence calibration, and any learned
   preprocessing on training only. Keep test labels hidden until the method is
   frozen.
6. Report accepted-match precision, false accepts, no-match false accepts, and
   coverage with confidence intervals. High precision is the safety objective;
   coverage is required so reject-all cannot appear successful.

## Required per-frame manifest

Every surviving frame must have a machine-readable row. The normative field
definitions and allowed values are in
[frame_manifest_schema.json](outputs/task3/frame_manifest_schema.json).

The minimum information is:

- **Identity and lineage:** dataset/policy version, run, camera, source filename,
  zero-based canonical decoded-image ordinal, decoder version, dimensions, and
  whether the image is raw or derived.
- **Time semantics:** run-level recording start/end, nullable capture timestamp,
  timestamp-linkage status, and nominal versus timestamp-derived cadence kept
  as separate fields. For this data, per-image capture time is null because the
  sidecar rows cannot be assigned uniquely.
- **Quality and sampling:** route phase and boundary method, decode/resolution
  status, route location-group ID, temporal occurrence, quality flags —
  including `too_dark` with its `mean_luma` and `p99_luma` measurements and the
  threshold version that produced it, `slope_gate` with the kink threshold and
  radius that produced it, and `occluded` with its review source —
  semantic-mask provenance, temporal cluster, cluster size, representative
  flag, and sample weight.
- **Split governance:** route/geography/traversal group, split, split-assignment
  version, leakage-buffer status, and paired-camera group.
- **Correspondence semantics:** partner run/camera/frame or null, acceptable
  interval when human labelled, mapping status, label source, review status,
  calibrated-probability flag, support score, method-freeze identifier, the
  camera-to-world convention offset (below), and known-failure flag.
- **Usage control:** allowed use, prohibited use, and a reason for any exclusion
  or quarantine.

### The mapping is many-to-one, and a consumer must know

`runB_frame` is **not a key**. Run B traverses parts of the route more slowly
than Run A, so several Run A frames correctly name the same Run B partner:

| Camera | Mapped rows | Rows sharing a partner | Max multiplicity |
|---|---:|---:|---:|
| CAM0 | 2,401 | **825 (34%)** | 12 |
| CAM5 | 2,183 | **1,128 (52%)** | 15 |

This is the expected shape of the answer where Run B is slower, not a defect,
and the counts are recorded in each `unified_manifest.json` under
`many_to_one_structure` (added by `scripts/amend_mapping_manifests.py`). But a
consumer that de-duplicates on `runB_frame`, joins on it as a key, or samples
training pairs uniformly will over-weight the slow stretches by up to 15x.
Weight by 1/multiplicity, or de-duplicate on the Run B side deliberately.

### What `confidence` means, and which values actually occur

`confidence` is an **ordinal rank, not a probability** — the manifests carry
`confidence_is_calibrated_probability: false` and now also a
`confidence_semantics` block and a `confidence_tier_values_present` census. The
writer's full scale is 1.00 strong, 0.67 sequence-supported, 0.50
geometry-propagated, 0.33 bridged, 0.00 abstain, with the category in `tier`.
**0.50 (geometry-propagated) does not occur in the submitted files**: CAM0 ships
1.00/0.67/0.00 only and CAM5 adds 0.33. A reader who sees the scale in the code
and assumes the shipped file exercises all of it will mis-weight the middle of
the range.

The sparse-anchor alternative shipped in `mappings_sparse_anchor_alternative/`
uses its own scale (1.00 geometry anchor, 0.50 bounded interpolation, 0.00
abstain). Its repository CSV puts that *category* in the `confidence` column;
The copy in `mappings_sparse_anchor_alternative/` carries the number in
`confidence` and the category in a `tier` column; `runB_frame` and every other
field are byte-identical to the repository CSV.

### Camera-to-world convention offset

`partner_frame` (`runB_frame`) is written in the **matcher's** convention: the
Run B frame whose image content aligns with the Run A frame. Human labels for
this route were taken in a **world-position** convention instead - the frame at
which a fixed far object and a fixed near object satisfy the far+near rule. On
CAM5 the two differ by a constant, measured independently of any label set in
`outputs/task2_pose_offset/pose_offset_diagnostic.json`
(`cameras.<camera>.label_convention_conversion.gap_frames_at_median_sweep`):

| Camera | Convention offset | Why |
|---|---:|---|
| CAM0 | **0 frames** | the camera did not move between the runs (dx = dy = 0 px at both the matcher's and the annotator's pairs) |
| CAM5 | **-2 frames** (measured -1.79) | an 18.5 px residual after pose compensation, at CAM5's median sweep rate |

**Attachment rule.** Ship the offset as a per-frame field beside the partner:
`convention_offset_frames`, with the value above and a pointer to the artifact
that measured it. A consumer that wants **world position** adds it to
`runB_frame`; a consumer that wants the matcher's convention ignores it. The
mapping CSVs themselves are **not** shifted - they were frozen before the blind
labels were sealed, and altering them after reading a score would void every
held-out number in this repository. What the offset buys, measured on the v3b
CAM5 rows as a sensitivity reading rather than a held-out number: strict
precision 0.143 -> 0.667 (Wilson 0.45-0.83) for the shipped mapping, with the
peak falling on the pre-measured value rather than on a fitted one
(`outputs/task2_evaluation/blind_v3b/convention_shift_sensitivity.json`).

**And what it does not buy.** The same file records two controls. On the **v2**
CAM5 labels, the set the constant was measured on, the shift is **neutral**:
0.375 at 0 and 0.375 at -2, with the signed late:early:inside counts moving
15:5:12 -> 7:13:12. Those labels mix two criteria (marker direction on some
queries, image layout on others), and a mixture is exactly what a neutral sweep
looks like. On **CAM0**, whose offset is 0, the sweep peaks at 0 (13/21 falling
to 5/21 at -2), as a zero-offset camera should. So the offset is
**-2 frames for world-position use: measured on v2, validated on v3b, neutral
on v2 itself.** Apply it to convert a convention, not to improve a number.

- Never apply the offset twice, and never apply CAM0's camera's value to CAM5.
- Never treat the shifted frame as a new ground truth: it converts a convention,
  it does not improve localisation.

## Consumer rules

* **CAM5 is a second channel, not a second-class one.** Report the cameras
  separately and never pool them. CAM0 rows may be used frame-exact (±0/±1)
  subject to the flags above; CAM5 rows are quarantined from frame-exact use
  (strict blind precision 0.375 on v2, 0.143 on the v3b redo) but stay usable
  at ±2 frames (69–71% on both blind sets, 97–100% at ±10) and as
  corroboration for CAM0. The CAM5 deficit is a localisation bias with a
  measured convention constant (−2 frames), not place confusion.

- Never infer that timestamp line i belongs to decoded image i.
- Never infer that CAM0 frame i and CAM5 frame i are synchronized.
- Never interpret confidence or support_score as a correctness probability;
  it is an ordinal rank, and 0.50 never occurs in the submitted files.
- Never treat `runB_frame` as a key or de-duplicate on it blindly: 34% of mapped
  CAM0 rows and 52% of CAM5 rows share a partner, up to 15 rows deep.
- Never use mapping coverage as accuracy.
- Never use an empty mapping as an automatic negative.
- Never compare a partner frame against a world-position label without first
  applying the camera's convention offset (CAM0 0, CAM5 -2 frames).
- Never read a populated row as top-tier supervision when either side is
  flagged `too_dark` or `occluded`; the mapping was confident on both and
  demonstrably wrong on the second.
- Never export a frame outside the immutable archive without plate and face
  redaction.
- Never wrap an arrival occurrence to the departure occurrence or interpolate
  across the route seam, even when endpoint imagery is visually similar.
- Never use calibration-selected 7/7 or the small 3/4 and 3/6 held-out results
  as a production-safety claim.
- Preserve abstentions and known failures; they are part of the deliverable,
  not rows to clean away after evaluation.
