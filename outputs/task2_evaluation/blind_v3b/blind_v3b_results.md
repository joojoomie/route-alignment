# Blind v3b: the CAM5 redo, scored once

This set re-asks CAM5 alone. The CAM5 half of blind v3 was annotated with a viewer that applied the pose compensation with the wrong sign, found after the seal from the sealed labels themselves and recorded in `outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`; its CAM5 precision is withdrawn. The CAM0 half of v3 is unaffected and stands. The redo runs on a fresh query set under `method_freeze_v3b.json`, which hashes exactly the same artifacts as the v3 freeze, so no parameter of any scored method changed between the two.

24 labels (CAM5), sealed 2026-09-02T10:53:30.879079+00:00 against method freeze `b98045635d7a` frozen 2026-09-02T10:03:08.989372+00:00. The query set was exported 2026-09-02T09:57:53.124514+00:00, before the freeze; the labels were sealed after it. Scored once.

## Headline: match labels, per camera

| Method | Cam | Accepted | Correct | Precision | Wilson 95% | Within 2 | Coverage | Median dist | p90 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `submitted_unified` | CAM5 | 21/23 | 3 | 0.1429 | 0.05-0.35 | 15/21 | 91.3% | 2.0 | 6.0 |
| `slope_refinement_variant` | CAM5 | 21/23 | 4 | 0.1905 | 0.08-0.40 | 17/21 | 91.3% | 1.0 | 4.0 |
| `posterior_model` | CAM5 | 23/23 | 4 | 0.1739 | 0.07-0.37 | 19/23 | 100.0% | 2.0 | 4.0 |
| `submitted_plus_slope_gate` | CAM5 | 20/23 | 2 | 0.1 | 0.03-0.30 | 14/20 | 87.0% | 2.0 | 6.200000000000003 |
| `submitted_plus_dark_gate` | CAM5 | 20/23 | 3 | 0.15 | 0.05-0.36 | 14/20 | 87.0% | 2.0 | 6.200000000000003 |
| `submitted_plus_both_gates` | CAM5 | 19/23 | 2 | 0.1053 | 0.03-0.31 | 13/19 | 82.6% | 2.0 | 6.399999999999999 |

## Abstention

`undetermined` labels are counts only - they are not negatives, and they leave the precision denominator. `no_correspondence` is the real abstention target: an acceptance is a false accept, an abstention a true reject.

| Method | Undetermined answered | no_correspondence false accepts | true rejects |
|---|---:|---:|---:|
| `submitted_unified` | 0/0 | 0/1 | 1/1 |
| `slope_refinement_variant` | 0/0 | 0/1 | 1/1 |
| `posterior_model` | 0/0 | 1/1 | 0/1 |
| `submitted_plus_slope_gate` | 0/0 | 0/1 | 1/1 |
| `submitted_plus_dark_gate` | 0/0 | 0/1 | 1/1 |
| `submitted_plus_both_gates` | 0/0 | 0/1 | 1/1 |

## Per stratum

Strata were located using the structure of the submitted unified mapping: its kink positions and, for the dark stratum, its Run A -> Run B partner for each frame. This affects WHICH frames are asked about. It does not affect WHAT the annotator sees: the page carries the Run A frame, a linear route-progress start position computed from the route boundaries alone, and the Run B filmstrip. No stratum tag, no mapping partner, no confidence and no prediction is present in the exported HTML, and a test greps the pages for the submitted partner of every query and fails if one appears. The consequence to state when reporting: per-stratum rates are conditional on the submitted mapping having put its kinks where it did, so they measure 'is this method right where it says something surprising', not an unbiased route-wide rate. The corridor stratum is the unbiased comparison.

### `submitted_unified`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM5 | kink | 5/5 | 1 | 0.2 | 0.04-0.62 | 2/5 |
| CAM5 | dark | 1/1 | 0 | 0.0 | 0.00-0.79 | 1/1 |
| CAM5 | corridor | 15/17 | 2 | 0.1333 | 0.04-0.38 | 12/15 |

### `slope_refinement_variant`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM5 | kink | 5/5 | 2 | 0.4 | 0.12-0.77 | 3/5 |
| CAM5 | dark | 1/1 | 0 | 0.0 | 0.00-0.79 | 1/1 |
| CAM5 | corridor | 15/17 | 2 | 0.1333 | 0.04-0.38 | 13/15 |

### `posterior_model`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM5 | kink | 5/5 | 2 | 0.4 | 0.12-0.77 | 3/5 |
| CAM5 | dark | 1/1 | 0 | 0.0 | 0.00-0.79 | 1/1 |
| CAM5 | corridor | 17/17 | 2 | 0.1176 | 0.03-0.34 | 15/17 |

### `submitted_plus_slope_gate`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM5 | kink | 4/5 | 0 | 0.0 | 0.00-0.49 | 1/4 |
| CAM5 | dark | 1/1 | 0 | 0.0 | 0.00-0.79 | 1/1 |
| CAM5 | corridor | 15/17 | 2 | 0.1333 | 0.04-0.38 | 12/15 |

### `submitted_plus_dark_gate`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM5 | kink | 5/5 | 1 | 0.2 | 0.04-0.62 | 2/5 |
| CAM5 | dark | 0/1 | 0 | n/a | n/a | 0/0 |
| CAM5 | corridor | 15/17 | 2 | 0.1333 | 0.04-0.38 | 12/15 |

### `submitted_plus_both_gates`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM5 | kink | 4/5 | 0 | 0.0 | 0.00-0.49 | 1/4 |
| CAM5 | dark | 0/1 | 0 | n/a | n/a | 0/0 |
| CAM5 | corridor | 15/17 | 2 | 0.1333 | 0.04-0.38 | 12/15 |

## The kink claim, tested on labels it did not select

The claim under test, made on the v2 labels: match labels near a mapping kink miss more often than labels away from one. On v3b the comparison is the kink stratum against the corridor stratum, which is the untargeted control; the v2 labels play no part in it.

For reference only, the v2 numbers that produced the claim: CAM0 0.5 near (4 of 8) against 0.2083 away (5 of 24); CAM5 0.6667 against 0.6087.

| Method | Cam | Kink miss rate | Corridor miss rate | Difference |
|---|---|---:|---:|---:|
| `submitted_unified` | CAM5 | 4/5 = 0.8 | 13/15 = 0.8667 | -0.0667 |
| `slope_refinement_variant` | CAM5 | 3/5 = 0.6 | 13/15 = 0.8667 | -0.2667 |
| `posterior_model` | CAM5 | 3/5 = 0.6 | 15/17 = 0.8824 | -0.2824 |
| `submitted_plus_slope_gate` | CAM5 | 4/4 = 1.0 | 13/15 = 0.8667 | 0.1333 |
| `submitted_plus_dark_gate` | CAM5 | 4/5 = 0.8 | 13/15 = 0.8667 | -0.0667 |
| `submitted_plus_both_gates` | CAM5 | 4/4 = 1.0 | 13/15 = 0.8667 | 0.1333 |

Replication verdict: the kink stratum misses more often than the corridor stratum on neither camera only. The intervals overlap at these counts, so this is a direction, not a rate.

## Signed error: is the prediction late or early?

An unsigned error cannot separate a mapping that is imprecise from a viewer that shows the wrong picture. On the withdrawn v3 CAM5 set the submitted mapping's error against the labels was systematically positive - the annotator, recovering a wrongly-displaced raster with the slider, settled on Run B frames earlier than the mapping's - so the direction of the error is reported here, not only its size.

| Method | Cam | n | Median signed | Mean | Later | Earlier | Inside | Median signed vs preferred |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `submitted_unified` | CAM5 | 21 | 2.0 | 2.5238 | 17 | 1 | 3 | 3.0 |
| `slope_refinement_variant` | CAM5 | 21 | 1.0 | 1.619 | 16 | 1 | 4 | 3.0 |
| `posterior_model` | CAM5 | 23 | 1.0 | 0.0435 | 17 | 2 | 4 | 2.0 |
| `submitted_plus_slope_gate` | CAM5 | 20 | 2.0 | 2.65 | 17 | 1 | 2 | 3.0 |
| `submitted_plus_dark_gate` | CAM5 | 20 | 2.0 | 2.55 | 16 | 1 | 3 | 3.0 |
| `submitted_plus_both_gates` | CAM5 | 19 | 2.0 | 2.6842 | 16 | 1 | 2 | 3.0 |

The withdrawn v3 CAM5 set, for reference: n 23, median signed interval error 4.0, 18 later against 3 earlier, median signed error against the preferred frame 4.0.

CAM5, submitted mapping: median signed interval error 2.0 (mean 2.5238), 17 predictions later than the label interval against 1 earlier and 3 inside, against the withdrawn v3 set's median 4.0 with 18 later against 3 earlier. By the stated criterion the systematic offset is NOT gone: it is smaller than on v3 but still one-directional. With the corrected tool and every slider at its default, the residual is the mapping predicting late on CAM5, not the viewer.

## CAM5 against the v2 set

| Set | Slice | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| v2 | route-uniform | 32/37 | 12 | 0.375 | 0.23-0.55 | 22/32 |
| v3b | corridor only | 15/17 | 2 | 0.1333 | 0.04-0.38 | 12/15 |
| v3b | overall | 21/23 | 3 | 0.1429 | 0.05-0.35 | 15/21 |

The headline v2 CAM5 pair quoted elsewhere is 0.375 strict on the 32 accepted match labels with 21/32 within two frames of the preferred frame. The interval-distance reading of within-2 is 22/32; this evaluation's `within_two` column is the interval-distance one, so both are given.

The corridor stratum is the like-for-like comparison: it is the untargeted control, sampled uniformly over the route the way the whole v2 set was. The overall figure mixes in kink and dark queries deliberately drawn where the method is least plausible, so it is expected to sit below the corridor number and is not a route-wide rate. Both Wilson intervals are wide at these counts; a difference that does not clear them is not a difference.

## Pose convention check

- **CAM5**: annotator median dx 40.0 px (MAD 0.0, range 40.0 to 40.0), dy 12.0 px (MAD 0.0), over 23 match labels. Measured -37.5/-11.0 px at 960x720 scale = -40.18/-11.79 px; seeded default (40, 12) px is the compensation and its sign is correct; rows left at the seeded default: 24/24.

The CAM5 viewer seeded the compensation with the correct sign this time: default (40, 12) px, the negation of the measured displacement (-40.18, -11.79) px on the filmstrip raster. The annotator left 24 of 24 rows exactly at that default (median dx 40.0, dy 12.0). The defect that withdrew the v3 CAM5 half does not appear here: there was no reason to drag a slider that was already compensating the right way, so frame choice and shift could not trade off against each other.

## Annotator remarks, recorded before the seal

- **cam5 A1618** (lane_difference_observed): Annotator observed that Run B drives at a different lateral distance from the roadside here (lane difference). Far+near rule replaced by the centre-line rule (fixed object nearest the compensated centre column); labelled as match with +-2. Both frames lie in the A1600-1710 tree-tunnel stretch where the RootSIFT lateral-offset measurement has no usable pairs.
- **cam5 A1704** (lane_difference_observed): Annotator observed that Run B drives at a different lateral distance from the roadside here (lane difference). Far+near rule replaced by the centre-line rule (fixed object nearest the compensated centre column); labelled as match with +-2. Both frames lie in the A1600-1710 tree-tunnel stretch where the RootSIFT lateral-offset measurement has no usable pairs.
