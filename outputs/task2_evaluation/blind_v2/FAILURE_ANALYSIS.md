# Blind v2, v3 and the v3b CAM5 redo: results, a protocol defect, a tool defect, and what is withdrawn

Sections 1-10 are the v2 set. "Blind v3: the targeted set, and what
survived it" near the end is a second, independently sealed set drawn to
test the claims v2 suggested; it is the only held-out evidence for the slope
gate, the slope variant and the posterior model. **Its CAM5 half is
withdrawn** - the annotation tool applied the pose compensation with the wrong
sign - and is replaced by "Blind v3b: the CAM5 redo", a third sealed set. CAM0's
v3 numbers are unaffected and stand.

Method freeze `e3b115533f5d` (2026-09-01T15:41:34Z) scored once against label
seal `5a058323518b` (2026-09-01T15:52:15Z). The freeze predates the seal. No
threshold was changed after the result.

## The correction that changed the conclusion

The annotation page offered two answers: a match, or "no usable match". That was
a protocol defect of mine. It merged two different facts:

* Run B has no counterpart for this place, and
* the annotator could not determine the counterpart.

All 11 such labels meant the second. The claim is checkable independently of any
result: queries were sampled entirely inside the stretch both runs traversed
(Run A 180-2616 mapping into Run B 114-2621), so a genuine absence was
impossible by construction. A geometric audit agrees - six of the eight CAM0
cases admit a distributed-support epipolar fit at the predicted partner, with
13-22 inliers spread across five to seven grid cells. The two that fail do so on
inlier count in the rainiest frames; one of them, A1083 to B1001, is visually an
unambiguous same-place pair.

Undetermined queries therefore leave the precision denominator. **The correction
was made after the results were seen.** It is applied identically to both
methods, and it raises the denser method more, because that method answered
those queries while the sparse one abstained. Both readings are kept in
`blind_v2_undetermined_rescore.json`.

## Result

| Method | Cam | Accepted | Correct | Precision | Wilson 95% | Coverage | Median err | Within 2 | As-labelled |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| v2 unified (submitted) | CAM0 | 32/32 | 23 | **0.719** | 0.55-0.84 | 100% | 1.0 | 26/32 | 0.575 |
| v2 unified (submitted) | CAM5 | 32/37 | 12 | 0.375 | 0.23-0.55 | 86.5% | 2.0 | 21/32 | 0.343 |
| v1 sparse-anchor | CAM0 | 17/32 | 11 | 0.647 | 0.41-0.83 | 53.1% | 1.0 | 14/17 | 0.647 |
| v1 sparse-anchor | CAM5 | 4/37 | 1 | 0.250 | 0.05-0.70 | 10.8% | 1.5 | 3/4 | 0.250 |

The dense method dominates on both axes and both cameras. Its extra coverage is
not cheap volume: on the fifteen CAM0 queries only it answered, it was right on
eleven, matching its overall rate.

## Why CAM5 is weak: a mechanism, not a mystery

CAM5 accepts 32 of 37 determinable queries and gets 12 strictly right, but 21 of
those 32 land within two frames. That shape - many near misses, few gross
errors - is a localisation bias, and it decomposes into three measured causes.

**1. The camera moved between the runs, and the two "same place" conventions
disagree.** `outputs/task2_pose_offset/pose_offset_diagnostic.json` measures the
horizontal scene shift at the matcher's own chosen pairs and, separately, at the
annotator's labelled pairs. On CAM0 both are 0 px. On CAM5 the matcher sits at
dx = **-37.5 px** (bootstrap 95% -44.5 to -29.0) with dy = **-11 px** (-12 to
-9), and the labels at dx = **-56 px** (-63 to -44). The offset is constant along
the route - first-half median -59, second-half -54.5, Spearman against route
position -0.17 - so it is a fixed property of the rig, not drift.

For a side-facing camera a yaw offset and a few frames of travel produce the
same horizontal shift, so dx alone cannot separate them; dy can, and dy is
non-zero only on CAM5. The parallax study settles it from the other direction:
a rotation moves near and far content equally, travel does not. Aggregated over
187 CAM5 and 202 CAM0 samples
(`outputs/task2_unified_variants/parallax/parallax_manifest.json`), the shipped
partners already sit at the depth-uniform position, at -1.18 +/- 0.73 frames on
CAM0 and -0.64 +/- 1.52 on CAM5. So the CAM5 residual is **camera rotation
between the runs, not a travel offset**.

The 18.5 px gap between the two conventions is then a disagreement about what
"the same place" means. At CAM5's median sweep rate that is about **1.8 frames**
- which is the size of the error being reported. The annotator used the
world-centric convention: they aligned on the direction to a marker, which is
what a person does, and the matcher aligned the image content, which is what
RootSIFT does - **on the queries where the annotator used it**. Converting the
mapping to the label convention moves precision from 0.375 to **0.406** and
within-two from 0.688 to 0.781; scoring against an interval widened to admit
*either* convention gives 0.906. Those are in-sample diagnostics on spent labels
and are not claimed as results.

That last figure is the tell, and it is why the convention story has to be
stated carefully. A widened interval that admits *either* convention scores 0.906
while *committing* to the world-centric one scores only 0.406: the labels are
not consistently in one convention, they are **split between two**. The
criterion diagnostic measures the same thing directly - CAM5's labelled dx has a
standard deviation of 29.8 px about a -56 px median and spans -118 to +22, which
is two criteria averaged rather than one criterion with noise. So on v2 the
convention locates *part* of the error and cannot be applied as a constant; the
sweep in the blind v3b section below confirms that a constant -2 shift on these
same rows is exactly neutral (0.375 -> 0.375).

**2. CAM5 is aimed low, so it sees less.** An independent, label-free evidence
sweep (gradient NCC of the far-field band at mapped pairs, against a background
of Run B frames +/-70 to +/-140 away, 219 CAM5 and 240 CAM0 samples) gives a
median mapped-pair NCC of **0.196 on CAM5 against 0.419 on CAM0**, and a median
z against background of **3.37 against 6.81**. CAM5's cross-run image evidence is
about half CAM0's *everywhere*, not only at the failure points. The cause is
framing: roughly 70% of every CAM5 frame is bare asphalt within a few metres of
the lens, plus lane markings and chevrons that alias along the route, while
CAM0's mirror-and-kerb framing puts far more of the frame at 10-30 m.

**3. Where CAM5 fails, it is dark.** See the next section. All three CAM5 blind
labels inside the under-exposed stretch B350-487 are wrong.

Note that cause 1 is correctable by a calibration or a stated convention, cause
2 by cropping CAM5 to its informative upper band, and cause 3 by an exposure
flag. None of the three is a place-recognition failure.

## Two failure modes the labels never saw, found by looking at frames

**Under-exposure.** `scripts/build_frame_quality_flags.py` measures mean and p99
Rec. 709 luma per frame and flags `too_dark` at mean < 12 or p99 < 40 of 255.
Run B CAM0 has 304 of 2,622 frames flagged, including a contiguous 177-frame
stretch at 1460-1636; Run B CAM5 has 136 of 2,615, in two stretches covering
350-487; Run A CAM5 has none. **273 of 2,401 accepted CAM0 rows and 147 of 2,183
accepted CAM5 rows have a partner flagged `too_dark`, every one of them
`accepted_strong`.** Both sides of such a pair are dark, so the descriptor match
is close to guaranteed regardless of place: a mechanism for systematic
high-confidence error, not random error. The labels reach these regions eight
times: on CAM5, all three are wrong, missed by 3, 8 and 16 frames; on CAM0, two
were undetermined, two correct, one wrong by 14. Eight labels cannot measure a
rate; they are consistent with the mechanism.

**A confirmed occlusion, and the one no-correspondence case in this data.** An
oncoming box lorry fills the entire Run B CAM0 frame at ordinals 247-252, and a
passing car covers about two thirds of it at 258-270
(`outputs/task2_abstention/confirmed_occlusion_runB_cam0.json`; source is visual
inspection, not a label). Over the fully occluded core no Run A frame can have a
counterpart. The submitted mapping sends Run A 327-356 into B240-270 as 30
`accepted_strong` rows at confidence 1.00. The posterior model - the one built
specifically to be able to say "no correspondence" - sends the same 30 rows with
a no-correspondence probability of at most 0.0002. **Its null state is
demonstrably under-used on the only confirmed case available**, which the
held-out-stretch test in the report could not have revealed, because a withheld
stretch removes the evidence while an occluder replaces it with confident
nonsense.

## A third failure mode, found in the mapping itself: implausible one-step jumps

A mapping's first difference is a local speed ratio. Run A frames are 0.1 s
apart, so mapping consecutive frames to Run B `b` and `b + d` asserts that Run B
covered in `d` frames what Run A covered in one. The pipeline's own coarse search
is bounded to a sustained slope of 2.0 with a largest single step of 3.0, so
`d >= 6` is outside what the model that produced the path may believe, and it
cannot be a stop: a stop makes Run B *repeat* frames (`d = 0`).

**Counts** (`outputs/task2_kinks/`, label-free): the submitted mapping contains
**27 such jumps on CAM0 (17 lit, 10 dark) and 25 on CAM5 (19 lit, 6 dark), every
one `accepted_strong` at confidence 1.00**; at `d >= 10` it is still 11 and 7.
The Bayes posterior, which carries an explicit transition model, contains **one
per camera**. On CAM0, 20 of the 27 sit within 10 Run A frames of a cross-camera
disagreement over 6 frames, which is independent of any label.

**Label cross-check** (spent labels, tuning nothing): match labels within 15 Run
A frames of a jump miss at **0.50 against 0.21 away** on CAM0 — 4 of 8 against 5
of 24, too few for a rate, and consistent with the mechanism rather than against
it. CAM5 is weak everywhere (0.67 near, 0.61 away), so the signal there is
swamped.

**Mechanism, reproduced on CAM0 over A880–940 from the cached descriptors.** The
coarse DTW path is *smooth* there: largest one-step increment 3, no increment at
or above 6. The submitted mapping's largest is 11. Route-wide the refined-minus-
coarse residual is bounded to ±8 and sits **pinned at the +8 window edge on 74
frames**. Replaying stage 3 with the real structure descriptors reproduces the
submitted values exactly (61/61), so this is not a reporting artefact: the jump
is the unconstrained argmax of a *collapsed* structure score, and the
non-decreasing floor then changed the pick on 27 of 61 frames, at times leaving
one admissible candidate out of 17. On those 27 frames the free argmax sits a
median 9 frames *behind* the submitted value and scores better there (0.65
against 0.42). The descriptors say "go back"; the floor forbids it.

**Root cause.** Stage 3 refines at full cadence with a monotonicity constraint
but **no slope prior**. The coarse stage has one ([0.2, 2.0]) and does not kink
here. Greedy argmax plus a hard floor is a ratchet: one collapsed frame moves the
sequence forward and every later frame inherits it.

**The fix, built as a development variant.**
`build_task2_unified.py --refinement slope` replaces the greedy scan with a
Viterbi over the same ±8 windows and the same descriptors: emission is the
structure cosine, the transition cost is 0.05 per frame of departure from the
coarse path's own local increment, and monotonicity is a hard step bound of
[0, 3] rather than a floor carried forward. Nothing else changes — stage 4
margins, stage 5 geometry, the tiers, bridging and the writer are identical, and
the DINOv2 descriptor cache is shared with the shipped run. Output goes to
`outputs/task2_unified_variants/slope/`; the parameters are in that directory's
`unified_manifest.json`; `scripts/test_refinement_slope.py` pins the ratchet's
behaviour against a copy of the old inline loop and shows the Viterbi does not
jump where the ratchet does.

Its kink count is **zero by construction** — the step bound forbids one — so the
count is not evidence about the variant. The label-free comparison is
(`outputs/task2_variants/variant_comparison.json`,
`outputs/task2_kinks/mapping_kinks.json`):

| | submitted | `--refinement slope` |
|---|---|---|
| kinks at `d >= 6`, CAM0 / CAM5 | 27 / 25 | 0 / 0 (by construction) |
| cross-camera median / p95 | 1 / 11 frames | 1 / **9** frames |
| coverage CAM0 / CAM5 | 89.8% / 81.6% | 89.8% / **82.5%** |
| endpoint closure CAM0 / CAM5 | **4/4** / 0/3 | 2/4 / **1/3** |
| structure score at its own picks, A880–940 | 0.52 | **0.60** (free argmax 0.65) |
| spent-label precision CAM0 / CAM5 | 0.719 / **0.375** | **0.813** / 0.303 |
| spent-label within-2 CAM0 / CAM5 | 0.844 / 0.688 | **0.906** / **0.788** |

The window check is the strongest label-free item: over the same 61 frames and
the same descriptors the variant's picks score at least as well as the submitted
mapping's on 50 of 61 frames and coincide with the unconstrained argmax on 36, so
it is following the evidence rather than merely imposing a smoother shape on it.
The result is not uniform: **CAM0 endpoint closure drops from 4/4 to 2/4 and
CAM5's strict spent-label precision falls**, and both are reported here for that
reason.

**Why the submission was not switched.** The blind labels are spent. They were
read once, against a frozen method, and that single reading is what gives the
submitted number its meaning. A method fixed after reading them cannot have a
held-out number, and the spent-label column above is a diagnostic in exactly the
way the CAM5 convention rescoring is — an in-sample reading of labels that have
already selected. Swapping the submission for the better-looking variant would
buy an improvement that no one can check and spend the one thing in this project
that is not recoverable: the ordering of freeze, seal and score. The evaluation
contract is worth more than the improvement. What ships instead is the finding,
the mechanism, the variant as evidence, and a policy gate over the frozen
mapping (`slope_gate` in `outputs/task3/frame_quality_flags.csv`, costed in
DATA_POLICY.md: 454 of 2,366 top-tier CAM0 rows and 320 of 1,881 on CAM5). The
right next step is a fresh prediction-blind set, scored once against the fixed
method.

## The hard region is 28% of the route, not 40 frames

An earlier reading named A1068-1110 as the route's hard region, on the strength
of the cross-camera disagreement peak. Frame inspection makes it much larger:
Run A CAM0 1050-1450 is one continuous corridor of green chain-link fence,
concrete kerb, grass verge and a single white edge line - 40 seconds with no
distinguishing structure - and 1500-1750 is a dark tree tunnel. That is about
700 Run A frames, **28% of the route**, and 5 of the 7 lowest-evidence CAM0
samples in the independent sweep fall inside or beside it. Run A CAM5 1000-1450
is the analogous stretch, an empty dual carriageway with a guardrail and a tree
line. Coverage figures should be read against that difficulty profile, not
against a single 40-frame region.

## What the descriptor stage is not

Five variants were built and evaluated label-free against cross-camera
agreement, endpoint closure and coverage
(`outputs/task2_variants/variant_comparison.json`): greyscale descriptors, a
temporal-statistics sky mask, sky+greyscale together, parallax-based partner
re-selection, and the posterior model with an added speed channel. **None of
these input-side changes moves the label-free signals materially.** Cross-camera p95 disagreement is 11 frames
at baseline and 11-12 for every appearance variant; coverage moves by about a
point; endpoint closure is unchanged at 4/4 CAM0 and 0/3 CAM5 except under
parallax, which makes cross-camera agreement worse (median 1 to 3 frames). The
speed channel's own likelihood-ratio table has a diagonal mean of only 1.11
(CAM0) and 1.28 (CAM5) against off-diagonal 0.73 and 0.71, so it barely
discriminates, and adding it changes nothing. The conclusion is negative and
useful: **the input side of the descriptor stage is not the bottleneck.** The
pooling is: an unsupervised VLAD over the same masked tokens
(`--aggregation vlad`, 32 words learned from the route, `vlad_aggregation.py`)
takes cross-camera p95 from 11 to 8 frames over 2,389 frames mapped by both
cameras (2,183 before), lifts CAM5 coverage 0.816 to 0.898, and on the spent
CAM0 labels reads 0.781 (0.91 within two) against 0.719; CAM5 0.361 on 36
accepted against 0.375 on 32; closure 3/4 and 0/4; kinks 24/33 against 27/25
(`outputs/task2_kinks_variants/mapping_kinks.json`). A mean is owned by area
and averages away the small structures that separate look-alike places; VLAD
keeps a block per word. The kinks do not move because they are stage 3's. No
held-out number; not submitted. Beyond that, what is missing is
geometry that gates the frame it checked, an existence term, and a stated
convention for "same place" - the three things listed under *What a next
iteration needs*.

## What is withdrawn

An earlier reading of this evaluation held that the dense method's abstention
mechanism had collapsed, because it accepted all 11 labelled no-match cases.
**That conclusion is withdrawn.** There were no true no-correspondence cases in
the set, so nothing was falsely accepted.

The stronger statement is that **this label set cannot evaluate abstention in
either direction**. It contains zero true negatives by construction. Neither the
claim that abstention works nor that it fails is supported here.

### What the held-out-stretch test does and does not establish

The withheld-stretch experiment (`outputs/task2_abstention/abstention_holdout.json`)
supplies negatives by construction where the labels could not. It should be read
with three limits stated up front, none of which the headline number carries:

| Hole width | CAM0 detection | CAM0 false alarm | CAM5 detection | CAM5 false alarm |
|---:|---:|---:|---:|---:|
| 60 | 0.40 | 0.007 | **0.93** | 0.019 |
| 150 | 0.64 | 0.006 | 0.86 | 0.023 |
| 400 | 0.83 | 0.007 | **0.79** | 0.020 |

1. **CAM5's detection rate falls as the hole gets wider** — 0.93 → 0.86 → 0.79 —
   which is the wrong direction: a larger absence should be easier to notice, as
   it is on CAM0. That is the signature of a detector firing on something other
   than the absence.
2. **CAM5 buys its higher numbers with about three times CAM0's false-alarm
   rate** (~0.02 against ~0.007). On a camera whose null state fires twice as
   readily, a higher detection rate is not evidence of a better mechanism.
3. **Each cell is 3 trials and carries no interval.** Read the direction, not
   the digits.

And the one thing the route itself supplied: **the posterior's null state missed
both real `no_correspondence` queries** — v3's CAM0 A2590 (accepted by all six
methods, the posterior at a no-correspondence probability of 6.9e-05) and v3b's
CAM5 A2587 (where five of six methods correctly abstained and the posterior was
the only one that did not). Two labels cannot measure a rate, but the direction
is consistent with the confirmed occlusion, and none of the three is in the null
state's favour. **The abstention mechanism is not claimed to work.**

## What still stands

Three code-level weaknesses, verified by inspection rather than by these labels.
Their consequence is now unmeasured, not disproved.

1. **The architecture cannot express "no correspondence."** A global monotonic
   path with pinned endpoints returns a complete path, so every route frame gets
   a partner by construction. There is no null state.
2. **The sequence gate measures relative, not absolute, evidence.** It asks
   whether the chosen corridor beats translated copies of itself - a real
   aliasing test, and CAM5's margins are duly lower than CAM0's - but never
   whether the corridor should exist. Its longest window was positive for 100%
   of frames on both cameras, so it never rejected anything.
3. **Geometry was propagated, not verified.** The stage samples one frame in
   twelve and marks the span between adjacent passing checks as supported. None
   of the 11 undetermined queries was ever directly checked.

## Why an absolute similarity gate will not fix it

Post-hoc, normalising the descriptor cosine against a local null distribution
(sampled from distant Run B frames for the same query) separates the two label
classes on CAM5 but not on CAM0:

| | match median z | undetermined median z | zero-FP threshold keeps |
|---|---:|---:|---|
| CAM0 | 2.61 | **2.68** | 5/32 true matches |
| CAM5 | 1.94 | 1.64 | 20/32 true matches |

On CAM0 the undetermined queries score *higher* than confirmed matches even after
normalisation. Appearance alone cannot separate "same place" from "same road,
different lane" on a side-facing camera. That needs geometry, or an
appearance-independent cue.

## The appearance-independent cue that this data cannot supply

Speed bumps produce a vertical suspension signature that is fixed in world
coordinates and independent of weather - exactly the orthogonal evidence the
foliage corridors need. Measured on this data with subpixel phase correlation
after removing the driving baseline:

* residual vertical shift sd 1.11 px (Run A CAM0) to 2.58 px (Run B CAM0, rain);
* 22 events detected in Run A CAM0 at |z| > 3, but only 7 in Run B, read at the
  time as an incomparable run-relative threshold;
* a cross-run likelihood ratio near 2.8 on four usable development pairs.

**Every number in that list came from a channel that was not measuring the
vehicle, and all three are withdrawn.** The decisive test is physical and needs
no labels: CAM0 and CAM5 are bolted to one chassis, so a real jolt must fire in
both streams of the **same run** at the same frame. The shipped construction,
which correlates raw frames with `normalization=None`, produced **0 of 28** such
co-firings on Run B within +/-2 frames, at every inter-camera lag from -5 to +5.
Zero is not a weak signal; it is a signal about something else. Run A scored
4 of 51 at zero lag, 1.64x chance, which is not much better.

Dividing each frame by its own stream's temporal mean before the correlation -
the static component of a stream *is* its temporal mean, which is what makes
this the right removal for a fixed droplet field - passes the test on both runs:
**4 of 33 on Run B** (3.85x chance, p ~ 0.019) and **9 of 45 on Run A** (4.75x,
p ~ 9e-5). That construction is now the default in `build_motion_events.py` and
the events *were* re-run, so the corrected figures replace the withdrawn ones:

| | Run A CAM0 | Run A CAM5 | Run B CAM0 | Run B CAM5 |
|---|---:|---:|---:|---:|
| events, withdrawn construction | 22 | 29 | 7 | 21 |
| events, corrected construction | 24 | 21 | **16** | 17 |

So the 7-versus-22 asymmetry was an **artefact of the estimator**, not a road
with fewer bumps and not a rain-degraded measurement of the same road. The
corrected channel's own cross-run ratio rests on 3 development pairs, 2 of which
co-fire, and is reported as a direction rather than a number. `median-subtract`
was measured as a third mode and destroys Run B; the scoring for all three is in
[`static_removal_study.json`](../../task2_motion_bayes/static_removal_study.json).

**The correction changes nothing downstream**, which is the other half of the
honest statement: the posterior mapping this channel feeds is 95-98% identical
frame by frame and over 98% within two frames under either construction
([`motion_variant_mapping_comparison_mean-divide.json`](../../task2_motion_bayes/motion_variant_mapping_comparison_mean-divide.json)).
The channel is now measuring the right thing while contributing almost nothing,
rather than measuring the wrong thing.

**One claim in the original audit does not reproduce and is withdrawn.** It
reported that whole-frame phase correlation returns exactly `dx = 0` for 91% of
Run B CAM0 pairs and 72% of Run B CAM5 pairs against 13% and 1% in Run A. Under
the shipped estimator the exact-zero fraction is 0.10-0.16 in **all four**
streams, with Run A CAM0 the highest of the four; the 76-89% readings appear
only with the sub-pixel upsampling switched off, which quantises every stream
alike and has nothing to do with rain (`pinning_premise` in the same JSON). The
case for the correction rests on the co-firing test, not on that figure.

The physics still bounds what any processing can recover, and it is independent
of the droplets. Vehicle body bounce is 1-2 Hz, giving 5-10 samples per cycle at
10 FPS; wheel hop at 10-15 Hz is fully aliased below the 5 Hz Nyquist limit. The
earlier reading that cross-camera correlation of the vertical residual is only
-0.21 was computed on the withdrawn construction and goes with it; the co-firing
rates above are its replacement, and at 4x chance they say the common-mode
component is real but sparse. Exploiting the cue properly needs an IMU or
30+ FPS video, not more processing of these frames. `detect_route_phases.py`
still correlates raw frames and was not re-run.

## Blind v3: the targeted set, and what survived it

Method freeze `014782a969c1` (2026-09-02T07:53:32Z) scored once against label
seal `e747d9775e2b` (2026-09-02T09:14:33Z). The query set was exported at
07:43:09Z, *before* the freeze; the labels were sealed after it. Both freezes
verify. `scripts/evaluate_task2_blind_v3.py` refuses to run otherwise, and
refuses to run twice.

### Design

The v2 labels are spent: they scored a frozen method once, and the three claims
built on top of that reading - kinks predict errors, a slope gate buys
precision, the slope refinement is better - could not be tested on the labels
that suggested them. v3 is 48 queries, 24 per camera, drawn to test exactly
those claims and nothing else, under a **three-way protocol**: `match`,
`undetermined`, `no_correspondence`. That fixes the v2 defect at source, so
"I cannot tell" is no longer recorded as "there is no such place".

Four strata: **kink** (within +-12 Run A frames of a submitted-mapping kink of
at least 6), **dark** (partner inside a contiguous Run B `too_dark` stretch of
at least 4), **occlusion** (inside the confirmed Run B CAM0 occlusion; CAM0
only, because the occlusion was on CAM0's side and CAM5 has no evidence for a
matching stratum), and **corridor** (uniform bins over the whole route with a
seeded jitter - the untargeted control).

**The dependency that has to be stated.** Strata were located using the
submitted mapping's own structure: its kink positions, and its Run A -> Run B
partner for the dark stratum. That affects *which* frames are asked about. It
does not affect *what* the annotator sees - the page carries the Run A frame, a
linear route-progress start position computed from the route boundaries alone,
and the Run B filmstrip; no stratum tag, no partner, no prediction, and a test
greps the exported pages for the submitted partner of every query and fails if
one appears. The consequence: **per-stratum rates are conditional on the
submitted mapping having put its kinks where it did.** They measure "is this
method right where it says something surprising", not a route-wide rate. **The
corridor stratum is the unbiased comparison** and is the number to quote.

### Result

Match labels only. `undetermined` rows leave the denominator by construction
(a prediction whose truth is unknown is neither right nor wrong) and are
reported as accept counts; the `no_correspondence` row is scored separately.

**The CAM5 rows in every table below are withdrawn.** They are kept rather than
deleted, because a number that was computed and then retracted is part of the
record; the CAM5 numbers to quote are in "Blind v3b: the CAM5 redo". CAM0's rows
stand.

| Method | Cam | Accepted | Correct | Precision | Wilson 95% | Within 2 | Coverage | Median dist |
|---|---|---:|---:|---:|---|---:|---:|---:|
| submitted unified | CAM0 | 21/21 | 13 | **0.619** | 0.41-0.79 | 17/21 | 100% | 0 |
| submitted unified | CAM5 | 23/23 | 2 | **0.087** | 0.02-0.27 | 5/23 | 100% | 4 |
| slope refinement variant | CAM0 | 21/21 | 16 | **0.762** | 0.55-0.89 | 17/21 | 100% | 0 |
| slope refinement variant | CAM5 | 23/23 | 3 | **0.130** | 0.05-0.32 | 6/23 | 100% | 4 |
| posterior model | CAM0 | 21/21 | 15 | 0.714 | 0.50-0.86 | 18/21 | 100% | 0 |
| posterior model | CAM5 | 22/23 | 2 | 0.091 | 0.03-0.28 | 10/22 | 95.7% | 3 |
| submitted + slope gate | CAM0 | 14/21 | 10 | 0.714 | 0.45-0.88 | 12/14 | 66.7% | 0 |
| submitted + slope gate | CAM5 | 12/23 | 1 | 0.083 | 0.01-0.35 | 4/12 | 52.2% | 4 |
| submitted + dark gate | CAM0 | 16/21 | 8 | 0.500 | 0.28-0.72 | 12/16 | 76.2% | 0.5 |
| submitted + dark gate | CAM5 | 19/23 | 2 | 0.105 | 0.03-0.31 | 5/19 | 82.6% | 4 |
| submitted + both gates | CAM0 | 9/21 | 5 | 0.556 | 0.27-0.81 | 7/9 | 42.9% | 0 |
| submitted + both gates | CAM5 | 9/23 | 1 | 0.111 | 0.02-0.43 | 4/9 | 39.1% | 3 |

Cameras are never pooled for a conclusion: they observe the same two traversals.

**Per stratum, submitted mapping** (correct / accepted):

| Cam | kink | dark | occlusion | corridor |
|---|---:|---:|---:|---:|
| CAM0 | 3/7 = 0.429 | 3/3 = 1.000 | 2/2 = 1.000 | **5/9 = 0.556** |
| CAM5 | 0/10 = 0.000 | 0/4 = 0.000 | n/a | **2/9 = 0.222** |

The bolded corridor column is the only unbiased route-wide reading. CAM5's
all-labels figure of 0.087 is *lower* than its corridor figure because half its
queries were deliberately drawn where the method is least plausible; the
comparable v2 number (0.375, on a route-uniform sample) is not the same
quantity and should not be read as a regression of the method.

### Did the kink claim replicate?

The v2 claim was "match labels within 15 Run A frames of a kink miss more often"
(CAM0 0.50 near against 0.21 away, 4 of 8 against 5 of 24). On v3 the test is
kink stratum against corridor stratum, and the v2 labels play no part in it.
Miss = an accepted match label whose interval distance exceeds 0.

| Cam | kink miss rate | corridor miss rate | difference |
|---|---:|---:|---:|
| CAM0 | 4/7 = 0.571 (0.25-0.84) | 4/9 = 0.444 (0.19-0.73) | +0.127 |
| CAM5 | 10/10 = 1.000 (0.72-1.00) | 7/9 = 0.778 (0.45-0.94) | +0.222 |

**The direction replicates on both cameras; the rate does not.** With 7-10 kink
and 9 corridor labels a side the Wilson intervals overlap heavily, and the CAM0
gap is a third of the size the v2 reading suggested (0.57 vs 0.44, against 0.50
vs 0.21). The honest statement is that a kink marks a neighbourhood the mapping
cannot justify - which was always a label-free argument - and that the labels
are consistent with it without establishing a rate.

### What the gates would actually have bought

Held-out, on labels the gates could not have been tuned on:

* **Slope gate.** CAM0 0.619 -> **0.714** at coverage 1.00 -> 0.67; it demotes
  7 of 21 CAM0 match labels, 6 of them in the kink stratum, and the one kink
  label it leaves is correct. CAM5 0.087 -> **0.083** at coverage 1.00 -> 0.52:
  it buys nothing there and costs half the answers. The spent-label diagnostic
  in DATA_POLICY.md predicted 0.575 -> 0.625 on CAM0; the held-out gain is
  larger, on a stratum built to contain kinks.
* **Exposure (`too_dark`) gate.** CAM0 0.619 -> **0.500**: it *costs* precision,
  because all 3 CAM0 dark-stratum labels were correct and the gate removes them
  along with two more. CAM5 0.087 -> 0.105, within noise. **The Task 3 dark rule
  is not supported as a precision gate by these labels.** Its case remains the
  mechanism (dark matches dark almost regardless of place) and the 273 CAM0 /
  147 CAM5 accepted rows with a flagged partner, not a measured gain.
* **Both gates.** CAM0 0.556 at 42.9% coverage, CAM5 0.111 at 39.1%. The union
  is worse than the slope gate alone on CAM0 and buys nothing on CAM5.

Reported as measured, including where the gates look worse. Nothing here was
tuned after the labels were opened; no parameter of any scored method changed.

### What the slope variant did

`build_task2_unified.py --refinement slope` was frozen by name before the labels
existed, so this is a held-out number and not the in-sample diagnostic of the
previous section: **CAM0 0.762 against the submitted 0.619, CAM5 0.130 against
0.087**, with p90 interval distance falling 6 -> 5 on CAM0 and 12.4 -> 9.6 on
CAM5 and the maximum error falling 10 -> 8 and 14 -> 13. In the kink stratum it
scores 5/7 against 3/7 on CAM0 and 1/10 against 0/10 on CAM5, which is where the
mechanism predicted the gain. It is still **not submitted**: the submission was
frozen before either blind set was opened, and the value of that ordering is
worth more than the improvement. The finding is that the fix works, measured
where it was supposed to work.

The posterior model, also named by the freeze, scores 0.714 / 0.091 and abstains
once on CAM5 - the only abstention any method produced on a match label.

### Abstention, on the one query that could test it

The set contains one `no_correspondence` label: CAM0 A2590, in the kink stratum,
where the annotator judged that no Run B frame shows the place.

**Every one of the six scored methods accepted it.** The submitted mapping sends
A2590 to B2608 as `accepted_strong` at confidence 1.00. The posterior model -
the one built with an explicit null state - sends it to B2606 with a
no-correspondence probability of **6.9e-05**. Neither gate catches it: the
nearest kink is at A2579 with increment 7, and the slope gate's radius is 10
Run A frames, so the gate covers A2569-2589 and stops **one frame short**.

One label cannot measure a false-accept rate. It can falsify a claim, and it
does: the null state is not merely under-used on the confirmed occlusion, it is
under-used on an independently labelled absence that no one chose for it. This
is the second confirmed no-correspondence case in the project and the first one
found by a blind annotator rather than by inspection.

### Two things the labels say about the evidence itself

**The occlusion stratum.** Of the three CAM0 queries inside the confirmed Run B
occlusion (B240-270), the annotator marked one `undetermined` and **two as
matches**, at B250-252 and B254-256 - inside the interval the visual audit
called fully occluded (B247-252) - and the submitted mapping is right on both.
Either the fully-occluded core is narrower than the audit recorded, or the
annotator aligned on periphery that survives the occluder. The audit's interval
is a visual estimate and this is evidence against its edges, not against the
occlusion; the confirmed-occlusion record should be read as approximate.

**A pose-convention error, caught by the sliders.** The viewer seeded CAM5's
compensation with the measured matcher-convention shift, -37.5/-11 px at
896x672, scaled to the 960x720 filmstrip as (-40, -12), and wrote whatever the
annotator left in the sliders into every label. The annotator moved it on
**every CAM5 row they could align**: median dx **+81 px** (MAD 17, IQR 68-102,
range 10-200), dy **0** on all 23. The one CAM5 row still at (-40, -12) is the
`undetermined` query. CAM0's measured offset is 0/0 and all 24 CAM0 rows are
0/0, which is the control. The sign is **opposite** to the seeded default, so
the shift was applied to the viewer in the wrong direction; the magnitude is
about twice the sign-flipped matcher value (+40) and closer to the sign-flipped
*label*-convention value (+60). This does not invalidate the labels - the
compensation only translates the displayed Run B raster and the annotator
corrected it by eye per query - but it is a real defect in the annotation tool
and it is one more instance of the CAM5 convention problem in section
"Why CAM5 is weak".

> **Correction, added after this section was written.** The clause "this does not
> invalidate the labels" is wrong, and the evidence that it is wrong is in the
> sealed labels themselves. Sign-flipping the compensation is not a neutral
> translation the annotator can undo for free: the slider it forced them onto
> also lets frame error and shift trade off against each other, and the recorded
> slider values correlate with the mapping's error (Spearman 0.45, least-squares
> slope 0.053 frames/px against the 0.09 a full trade-off would give). The CAM5
> half of v3 is therefore withdrawn in full - see
> `outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md` - and re-run as v3b.
> CAM0's measured offset is 0/0, its default was 0/0 and every CAM0 row is 0/0,
> so CAM0 is untouched by this.

### Remarks the annotator recorded before the seal

Both were recorded during labelling, timestamped before the seal, and neither
changed a label:

* **CAM0 A646** (`bracketed_by_occlusion`): "The house is visible in Run B a few
  frames before and after, but a truck hides it at the frames where the position
  matches. Labelled as match with an interval spanning the occluded frames; the
  preferred frame is the interval midpoint and could not be verified visually.
  This is a second occlusion instance, distinct from the confirmed runB cam0
  240-270 one." That makes **three** occluders now identified in Run B CAM0, of
  which only one is in `confirmed_occlusion_runB_cam0.json`. The submitted
  mapping is correct on this query; the label's interval is wider than usual for
  a reason that is recorded rather than hidden.
* **CAM0 A2399** (`turn_and_lateral_offset`): "Run B is mid-turn for many
  consecutive frames here and the distance to the pavement differs slightly from
  Run A, so neither the far+near test nor the centre-line test pins one frame.
  Labelled as match with +-3 frames." A lateral-offset difference between the
  runs is exactly what the Task 1 lateral study measures, and a mid-turn segment
  is where a monotonic path is least constrained.

### Evidence status

The four artifacts hashed by the v3 freeze - the submitted mapping, the slope
variant, the posterior model and the kink CSVs, plus `frame_quality_flags.csv`
and `mapping_kinks.json` - **have held-out numbers from this set.** Anything
built after 2026-09-02T07:53:32Z does not. The strata are conditional on the
submitted mapping's structure; the corridor stratum is not. The seal proves
ordering, not custody: it shows the method was fixed before the labels were
imported, not that nobody ever looked at them.

Full output: `outputs/task2_evaluation/blind_v3/blind_v3_results.json`,
`blind_v3_results.md`, `blind_v3_cases.csv`.

## Blind v3b: the CAM5 redo

Method freeze `b98045635d7a` (2026-09-02T10:03:08Z) scored once against label
seal `d82913b90c92` (2026-09-02T10:53:30Z). The query set was exported at
09:57:53Z, *before* the freeze; the labels were sealed after it. That freeze
hashes **exactly the same ten artifacts as the v3 freeze** - byte-identical
mappings, variants, posteriors, kink CSVs and quality flags - so no parameter of
any scored method changed between the two runs, and a test asserts it. All three
freezes (`freeze_method_v2.py --verify`, `freeze_method_v3.py --verify`,
`freeze_method_v3.py --set-name blind_v3b --verify`) verify.

### Why there is a redo at all: the defect, and how it was caught

1. The measured CAM5 scene displacement is dx = -37.5 px at 896x672: content in
   Run B sits to the **left** of where it sits in Run A. The convention is
   `compensation = -displacement`, so the viewer must move Run B **right** by
   +40 px on the 960-wide filmstrip. The v3 builder shipped the raw measurement
   as the compensation, so the page moved Run B a further 40 px left and
   **doubled** the misalignment. CAM0's measurement is 0/0, so CAM0 was
   unaffected - the control was built in by accident.
2. The annotator recovered with the pose slider on 22 of 23 CAM5 match labels,
   dragging it from -40 to between +10 and +200 px (median +81). That is what
   made it visible: the sliders are written into every label, so the sealed file
   records the tool's state, not only the answer.
3. Slider and error are correlated (Spearman 0.45, Pearson 0.46; least-squares
   slope 0.053 frames/px, zero-error at dx = 29 px against the correct +40).
   A full trade-off would give 1/sweep = 0.09, so roughly half the slider excess
   became frame error. Labels set with dx >= 120 px have errors 0, 3, 14, 13, 10;
   labels with dx <= 40 have 0, 1, 4, -4.
4. Signed error against the labels was systematically **+3 to +4** on CAM5
   (median +4, 18 late against 3 early) where CAM0's median was 0.

Recorded in `outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`. Nothing in
the v3 label file was altered. CAM5 v3 precision (2/23 = 0.087) is a property of
the tool and is withdrawn from the headline; CAM0 v3 stands.

### Design

24 fresh CAM5 queries, no Run A ordinal shared with the v3 set (a test asserts
the intersection is empty), and at least 10 frames from every previously
labelled ordinal in v2, v3, the manual annotations and the anchor candidates.
Same three-way protocol (`match` / `undetermined` / `no_correspondence`), same
six scored methods, same metrics, same blindness checks.

The strata plan asked for 10 kink and 5 dark; the spacing rule left almost no
free ordinals, so **the exported set is kink 5, dark 1, corridor 18** with the
moves recorded per camera under `strata_shortfalls` in the manifest and copied
into the results JSON. That is a weaker test of kinks than v3 was and a
**stronger** unbiased set: 17 of the 23 match labels are corridor, so the
corridor number here rests on nearly the whole set rather than on nine queries.

The tool fix is verifiable from the labels: the seeded default is (40, 12) px -
the negation of the measurement - and **all 24 rows are at that default**. The
scorer no longer asserts one convention; it detects which of the two the viewer
used and records it, because that is the fact deciding whether a set can be read
as the mapping's precision at all.

### Result

Match labels only; 23 `match`, 0 `undetermined`, 1 `no_correspondence`.

| Method | Accepted | Correct | Precision | Wilson 95% | Within 2 | Coverage |
|---|---:|---:|---:|---|---:|---:|
| submitted unified | 21/23 | 3 | **0.143** | 0.05-0.35 | 15/21 | 91.3% |
| slope refinement variant | 21/23 | 4 | **0.191** | 0.08-0.40 | 17/21 | 91.3% |
| posterior model | 23/23 | 4 | 0.174 | 0.07-0.37 | 19/23 | 100% |
| submitted + slope gate | 20/23 | 2 | 0.100 | 0.03-0.30 | 14/20 | 87.0% |
| submitted + dark gate | 20/23 | 3 | 0.150 | 0.05-0.36 | 14/20 | 87.0% |
| submitted + both gates | 19/23 | 2 | 0.105 | 0.03-0.31 | 13/19 | 82.6% |

**Per stratum, submitted mapping** (correct / accepted):

| kink | dark | corridor |
|---:|---:|---:|
| 1/5 = 0.200 | 0/1 = 0.000 | **2/15 = 0.133** (0.04-0.38) |

The corridor column is the unbiased route-wide reading and it is now most of the
set. Two conclusions the v3 CAM5 half appeared to support do **not** survive:
the kink stratum misses *less* than the corridor stratum (0.80 against 0.87, so
the kink claim does not replicate on CAM5), and the slope gate *costs*
precision rather than buying nothing (0.143 -> 0.100 at coverage 0.91 -> 0.87).
The exposure gate is 0.143 -> 0.150, within noise, on a single dark label. The
slope variant still beats the shipped mapping (0.191 against 0.143), which is
the one v3 CAM5 direction that does survive the redo.

### Against v2

| Set | Slice | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| v2 | route-uniform, 37 match labels | 32 | 12 | 0.375 | 0.23-0.55 | 22/32 |
| v3b | corridor only | 15/17 | 2 | 0.133 | 0.04-0.38 | 12/15 |
| v3b | overall | 21/23 | 3 | 0.143 | 0.05-0.35 | 15/21 |

The v2 pair quoted elsewhere is **0.375 strict with 21/32 within two frames of
the preferred frame**; 22/32 is the same set read by interval distance, which is
the column this evaluation uses, and both are recorded in the results JSON. The
corridor row is the like-for-like comparison. The two Wilson intervals overlap
(0.23-0.55 against 0.04-0.38), so **this is not a measured regression of the
method**: it is 15 labels against 32, on a different sample, and the honest
statement is that CAM5 is weak on both and the redo does not resolve how weak.
v2 was also scored under the two-way protocol, whose `no_match` button merged
"cannot tell" with "no such place".

### Signed error: is the +3 to +4 gone?

Reported for the first time here, because an unsigned median cannot separate a
mapping that is imprecise from a viewer showing the wrong picture.

| Set (submitted mapping) | n | Median signed | Later | Earlier | Inside |
|---|---:|---:|---:|---:|---:|
| v3 CAM5, withdrawn | 23 | +4 | 18 | 3 | 2 |
| v3b CAM5 | 21 | **+2** | 17 | 1 | 3 |

Criterion, stated in the scorer before the run: the offset counts as gone if the
median is within +-1 frame of zero **and** the late/early counts differ by no
more than the larger of 2 and a quarter of the errors outside the interval.
**It is not gone.** The median halves, from +4 to +2, but 17 predictions against
1 still fall late.

That is the informative outcome. On v3 the offset could not be attributed: tool
and mapping were confounded. On v3b the compensation carries the correct sign
and every slider sits at its default, so nothing in the viewer can displace the
answer - the residual is **the submitted mapping predicting Run B frames late on
CAM5**, which is the localisation bias described in "Why CAM5 is weak", now
measured with a direction instead of an absolute value. The tool defect
inflated it; it did not create it.

### The CAM5 deficit is mostly a convention constant

The late bias above has a size that was measured **before this label set was
drawn**, on a different set, for a different purpose. `estimate_pose_offset.py`
compared two definitions of "the same place" on CAM5: the matcher's (the image
content aligns) and the annotator's (the world position a far+near rule names).
They differ by 18.5 px, which at CAM5's median sweep rate is
**-1.79 frames** -
`outputs/task2_pose_offset/pose_offset_diagnostic.json`, field
`cameras.cam5.label_convention_conversion.gap_frames_at_median_sweep`, built
`2026-09-02T06:08:40Z`, from the **v2** labels. CAM0's same field is `0.0`: that
camera did not move between the runs and needs no shift.

Applying that constant - rounded to **-2 frames** - to every accepted CAM5
prediction and rescoring the v3b rows gives:

| Mapping | 0 (as scored) | -1 | **-2 (measured)** | -3 |
|---|---:|---:|---:|---:|
| submitted unified **(shipped)** | 3/21 = 0.143 | 7/21 = 0.333 | **14/21 = 0.667** | 12/21 = 0.571 |
| slope refinement variant | 4/21 = 0.191 | 9/21 = 0.429 | **15/21 = 0.714** | 14/21 = 0.667 |
| posterior model | 4/23 = 0.174 | 10/23 = 0.435 | **18/23 = 0.783** | 15/23 = 0.652 |

At -2 the shipped mapping's Wilson 95% interval is **0.45-0.83** (within two
frames 15/21); the variant's is 0.50-0.86 and the posterior's 0.58-0.90. The
interval at -2 overlaps the one at -3 and, at these counts, is not cleanly
separated from the one at 0 either, so what the sweep establishes is that the
peak *sits on* the pre-measured value, not that the value is resolved. And -2 is
not chosen here - it is what the diagnostic already said.

#### The two controls, and what they take away

The same command sweeps two sets this reading was not derived from
(`control_sweeps` in the same file), because a peak on one set cannot
distinguish "one convention constant" from "a shift that happens to help these
rows":

| Control | 0 | -1 | -2 | -3 | signed late:early:inside at 0 -> -2 |
|---|---:|---:|---:|---:|---|
| **v2 CAM5** (32 accepted match rows, route-uniform) | 12/32 = **0.375** | 11/32 | 12/32 = **0.375** | 8/32 | 15:5:12 -> 7:13:12 |
| **v3 CAM0** (21 rows, measured offset 0) | **13/21** | 9/21 | 5/21 | 2/21 | 4:4:13 -> 1:15:5 |

**On v2 the same shift is neutral.** Precision at -2 is identical to precision at
0; only the *direction* of the errors flips, from 15 late / 5 early to 7 late /
13 early, with the same 12 inside. That is exactly what a **mixture** predicts:
if some labels were taken in the world-position convention and some in the
matcher's, a constant shift moves one group into the interval and the other
group out, and the count does not change. And the v2 CAM5 labels are documented
as mixed - `scripts/diagnose_label_criterion.py` measures the labelled dx at a
**standard deviation of 29.8 px about a -56 px median, spanning -118 to +22**,
which is not one criterion with noise but two criteria averaged
(`outputs/task2_label_criterion/label_criterion_diagnostic.json`, whose own
verdict for CAM5 is "cause not established"). The annotator used the marker's
own direction on some queries and the image layout on others. That mixture is
also the simplest account of why v2 CAM5 (0.375) sat *between* the two readings
the v3b set produced.

**On CAM0 the sweep peaks at 0 and falls monotonically**, 13/21 down to 2/21 at
-3, as a camera with a measured offset of 0.0 must. That is the negative control:
had it peaked away from zero, the shift would be buying something other than a
convention conversion.

So the honest statement is three-part, and replaces the earlier "largely a
convention constant, not place recognition":

> **A convention constant on a consistently-labelled set; neutral on the
> mixed-convention set; and the two blind CAM5 readings (0.375 on v2 at n=32,
> 0.133 on the v3b corridor at n=15) are not distinguishable at those counts.**

The constant is worth attaching - it converts a convention a consumer may need -
but it does not license the claim that CAM5's place recognition is as good as
CAM0's. On the labels that measure the convention consistently it is; on the
labels that do not, the shift buys nothing, and n=32 and n=15 cannot tell those
two worlds apart.

Two things this is not:

* **It is not a held-out number.** The v3b set was scored once, and those
  numbers stand unchanged in `blind_v3b_results.json`. This rescores the same
  rows under a shift, so it is a *sensitivity reading*. What keeps it from
  being circular is only that the constant is read from an artifact that
  predates the set; nothing further may be built on these labels.
* **It is not a reason to shift the submission.** The submitted mapping files
  are unchanged, and stay unchanged. The evaluation contract froze them before
  the labels were sealed; a constant applied after reading a score is not a
  frozen method, and shipping one would make every held-out number in this
  repository unreadable. The constant is attached as **per-frame metadata**
  instead (`DATA_POLICY.md`): CAM0 0 frames, CAM5 **-2 frames for
  world-position use - measured on v2, validated on v3b, neutral on v2 itself**,
  applied to `runB_frame` by a consumer that wants world position and ignored by
  one that wants the matcher's convention.

It also does not rescue the `within_two` reading, which barely moves (15/21 at
0, 15/21 at -2): the mapping was already close, and the constant decides
whether "close" lands inside a +-1 or +-2 label interval or just outside it.
That is exactly what a convention offset should look like, and exactly why the
strict number alone was misleading about the mechanism.

Reproduce: `scripts/evaluate_task2_blind_v3.py --set-name blind_v3b
--convention-sensitivity`. Output:
`outputs/task2_evaluation/blind_v3b/convention_shift_sensitivity.json`, which
records the constant's provenance, the shifts, the Wilson intervals and the
SHA-256 of the scoring pass it reads (and does not rewrite).

### The `no_correspondence` query

One label: **CAM5 A2587**, corridor stratum, at the very end of the route
(Run A ends at 2616). Five of the six methods **correctly abstained** - the
submitted mapping, the slope variant and all three gate readings leave that row
empty. The posterior model accepted it, and is the only method that did.

This is the opposite outcome to v3, where all six methods accepted the single
`no_correspondence` query, and it is worth stating precisely what it does and
does not show. One label cannot measure a false-accept rate in either direction.
It does show that the submitted mapping's abstention is not vacuous - it fires
on at least one independently labelled absence that nobody chose for it - and it
adds a second instance of the posterior's null state being under-used, after the
confirmed occlusion and the v3 CAM0 query. Three instances, three methods, one
consistent direction: the null state is not doing the job it was added for.

### The two lane-difference remarks

Both were recorded during labelling, timestamped before the seal, and neither
changed a label:

* **CAM5 A1618** -> B1538 (`lane_difference_observed`): "Run B drives at a
  different lateral distance from the roadside here. Far+near rule replaced by
  the centre-line rule (fixed object nearest the compensated centre column);
  labelled as match with +-2."
* **CAM5 A1704** -> B1639 (`lane_difference_observed`): the same observation and
  the same substitution, also +-2.

Both sit in **A1600-1710**, inside the A1500-1750 dark tree tunnel, and both are
in the stretch where the RootSIFT lateral-offset measurement of Task 1 finds no
usable pairs. So the one place the annotator reports seeing a lane difference by
eye is exactly the place the automated measurement cannot look, which is a
limitation of that measurement rather than a contradiction of it: an
independent human observation of the effect Task 1 could only bound from above.
Neither remark is evidence of a systematic lane change - two frames of a route
are two frames - but both are recorded because the alternative is a label whose
interval is wider than its neighbours for no reason a reader can see.

### Evidence status

The ten artifacts hashed by the v3b freeze have held-out CAM5 numbers from this
set. They are the same artifacts the v3 freeze hashed, so nothing built after
2026-09-02T07:53:32Z has a held-out number from either. The strata are
conditional on the submitted mapping's structure; the corridor stratum, which is
18 of the 24 queries here, is not. The seal proves ordering, not custody.

Full output: `outputs/task2_evaluation/blind_v3b/blind_v3b_results.json`,
`blind_v3b_results.md`, `blind_v3b_cases.csv`.

## What a next iteration needs

1. ~~**A label set that can test abstention**~~ - **done**: the v3 set uses the
   three-way protocol and contains one true `no_correspondence` query. Every
   method accepted it. What is still needed is *enough* of them: one label
   falsifies a claim but cannot measure a false-accept rate, and the sampling
   that produced exactly one has to be widened to parked segments, occluded
   stretches and route tails one run did not cover.
2. **A null state** in the path, so "no counterpart for this stretch" is
   representable, and **a slope prior in the refinement** — the Viterbi variant
   above, which v3 scores at 0.762 against 0.619 on CAM0 and v3b at 0.191 against
   0.143 on CAM5, held out. It should become the default at the next freeze; it
   is not swapped in retrospectively, because the submission was frozen first.
3. **Geometry that gates the frame it checked**, with propagation permitted only
   at a weaker tier.
4. **A confidence signal with an existence term.** A state-space formulation
   gives this naturally: forward-backward posteriors over a state space that
   includes a null symbol yield a calibrated per-frame probability, and
   abstention becomes a decision-theoretic threshold rather than a hand-tuned
   tier. With 40 labels per camera the emission model must come from physics and
   label-free null distributions; the labels can only check calibration.

Each would change predictions, so none may be quoted against these labels.

## Reading the same labels at frame tolerances

`scripts/evaluate_tolerance_curve.py` reads every sealed set at ±0/±1/±2/±5/±10
frames (`outputs/task2_evaluation/tolerance_curve.json`). At ±2 frames (0.2 s,
about 2 m) the submitted mapping reads 84.4% (CAM0) and 68.8% (CAM5) on v2,
81.0% on the v3 CAM0 set and 71.4% on the v3b CAM5 redo; at ±10 frames, the
scale of published place-recognition tolerances, 96.9%/96.9% and 100%/100%. The
strict ±0 numbers quoted throughout this document are unchanged and remain the
last table in the report; the tolerance reading is a re-reading, not a
re-scoring.
