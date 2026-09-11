# Blind v3: the targeted label set, scored once

48 labels, sealed 2026-09-02T09:14:33.436283+00:00 against method freeze `014782a969c1` frozen 2026-09-02T07:53:32.023239+00:00. The query set was exported 2026-09-02T07:43:09.034629+00:00, before the freeze; the labels were sealed after it. Scored once.

## Headline: match labels, per camera

| Method | Cam | Accepted | Correct | Precision | Wilson 95% | Within 2 | Coverage | Median dist | p90 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `submitted_unified` | CAM0 | 21/21 | 13 | 0.619 | 0.41-0.79 | 17/21 | 100.0% | 0.0 | 6.0 |
| `submitted_unified` | CAM5 | 23/23 | 2 | 0.087 | 0.02-0.27 | 5/23 | 100.0% | 4.0 | 12.400000000000002 |
| `slope_refinement_variant` | CAM0 | 21/21 | 16 | 0.7619 | 0.55-0.89 | 17/21 | 100.0% | 0.0 | 5.0 |
| `slope_refinement_variant` | CAM5 | 23/23 | 3 | 0.1304 | 0.05-0.32 | 6/23 | 100.0% | 4.0 | 9.600000000000001 |
| `posterior_model` | CAM0 | 21/21 | 15 | 0.7143 | 0.50-0.86 | 18/21 | 100.0% | 0.0 | 5.0 |
| `posterior_model` | CAM5 | 22/23 | 2 | 0.0909 | 0.03-0.28 | 10/22 | 95.7% | 3.0 | 5.900000000000002 |
| `submitted_plus_slope_gate` | CAM0 | 14/21 | 10 | 0.7143 | 0.45-0.88 | 12/14 | 66.7% | 0.0 | 4.100000000000003 |
| `submitted_plus_slope_gate` | CAM5 | 12/23 | 1 | 0.0833 | 0.01-0.35 | 4/12 | 52.2% | 4.0 | 5.0 |
| `submitted_plus_dark_gate` | CAM0 | 16/21 | 8 | 0.5 | 0.28-0.72 | 12/16 | 76.2% | 0.5 | 6.0 |
| `submitted_plus_dark_gate` | CAM5 | 19/23 | 2 | 0.1053 | 0.03-0.31 | 5/19 | 82.6% | 4.0 | 10.599999999999998 |
| `submitted_plus_both_gates` | CAM0 | 9/21 | 5 | 0.5556 | 0.27-0.81 | 7/9 | 42.9% | 0.0 | 5.2 |
| `submitted_plus_both_gates` | CAM5 | 9/23 | 1 | 0.1111 | 0.02-0.43 | 4/9 | 39.1% | 3.0 | 5.0 |

## Abstention

`undetermined` labels are counts only - they are not negatives, and they leave the precision denominator. `no_correspondence` is the real abstention target: an acceptance is a false accept, an abstention a true reject.

| Method | Undetermined answered | no_correspondence false accepts | true rejects |
|---|---:|---:|---:|
| `submitted_unified` | 3/3 | 1/1 | 0/1 |
| `slope_refinement_variant` | 3/3 | 1/1 | 0/1 |
| `posterior_model` | 3/3 | 1/1 | 0/1 |
| `submitted_plus_slope_gate` | 2/3 | 1/1 | 0/1 |
| `submitted_plus_dark_gate` | 0/3 | 1/1 | 0/1 |
| `submitted_plus_both_gates` | 0/3 | 1/1 | 0/1 |

## Per stratum

Strata were located using the structure of the submitted unified mapping: its kink positions and, for the dark stratum, its Run A -> Run B partner for each frame. This affects WHICH frames are asked about. It does not affect WHAT the annotator sees: the page carries the Run A frame, a linear route-progress start position computed from the route boundaries alone, and the Run B filmstrip. No stratum tag, no mapping partner, no confidence and no prediction is present in the exported HTML, and a test greps the pages for the submitted partner of every query and fails if one appears. The consequence to state when reporting: per-stratum rates are conditional on the submitted mapping having put its kinks where it did, so they measure 'is this method right where it says something surprising', not an unbiased route-wide rate. The corridor stratum is the unbiased comparison.

### `submitted_unified`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM0 | kink | 7/7 | 3 | 0.4286 | 0.16-0.75 | 5/7 |
| CAM0 | dark | 3/3 | 3 | 1.0 | 0.44-1.00 | 3/3 |
| CAM0 | occlusion | 2/2 | 2 | 1.0 | 0.34-1.00 | 2/2 |
| CAM0 | corridor | 9/9 | 5 | 0.5556 | 0.27-0.81 | 7/9 |
| CAM5 | kink | 10/10 | 0 | 0.0 | 0.00-0.28 | 1/10 |
| CAM5 | dark | 4/4 | 0 | 0.0 | 0.00-0.49 | 0/4 |
| CAM5 | corridor | 9/9 | 2 | 0.2222 | 0.06-0.55 | 4/9 |

### `slope_refinement_variant`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM0 | kink | 7/7 | 5 | 0.7143 | 0.36-0.92 | 5/7 |
| CAM0 | dark | 3/3 | 3 | 1.0 | 0.44-1.00 | 3/3 |
| CAM0 | occlusion | 2/2 | 2 | 1.0 | 0.34-1.00 | 2/2 |
| CAM0 | corridor | 9/9 | 6 | 0.6667 | 0.35-0.88 | 7/9 |
| CAM5 | kink | 10/10 | 1 | 0.1 | 0.02-0.40 | 2/10 |
| CAM5 | dark | 4/4 | 0 | 0.0 | 0.00-0.49 | 0/4 |
| CAM5 | corridor | 9/9 | 2 | 0.2222 | 0.06-0.55 | 4/9 |

### `posterior_model`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM0 | kink | 7/7 | 5 | 0.7143 | 0.36-0.92 | 6/7 |
| CAM0 | dark | 3/3 | 3 | 1.0 | 0.44-1.00 | 3/3 |
| CAM0 | occlusion | 2/2 | 2 | 1.0 | 0.34-1.00 | 2/2 |
| CAM0 | corridor | 9/9 | 5 | 0.5556 | 0.27-0.81 | 7/9 |
| CAM5 | kink | 9/10 | 0 | 0.0 | 0.00-0.30 | 5/9 |
| CAM5 | dark | 4/4 | 0 | 0.0 | 0.00-0.49 | 1/4 |
| CAM5 | corridor | 9/9 | 2 | 0.2222 | 0.06-0.55 | 4/9 |

### `submitted_plus_slope_gate`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM0 | kink | 1/7 | 1 | 1.0 | 0.21-1.00 | 1/1 |
| CAM0 | dark | 3/3 | 3 | 1.0 | 0.44-1.00 | 3/3 |
| CAM0 | occlusion | 2/2 | 2 | 1.0 | 0.34-1.00 | 2/2 |
| CAM0 | corridor | 8/9 | 4 | 0.5 | 0.22-0.78 | 6/8 |
| CAM5 | kink | 1/10 | 0 | 0.0 | 0.00-0.79 | 1/1 |
| CAM5 | dark | 3/4 | 0 | 0.0 | 0.00-0.56 | 0/3 |
| CAM5 | corridor | 8/9 | 1 | 0.125 | 0.02-0.47 | 3/8 |

### `submitted_plus_dark_gate`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM0 | kink | 7/7 | 3 | 0.4286 | 0.16-0.75 | 5/7 |
| CAM0 | dark | 0/3 | 0 | n/a | n/a | 0/0 |
| CAM0 | occlusion | 1/2 | 1 | 1.0 | 0.21-1.00 | 1/1 |
| CAM0 | corridor | 8/9 | 4 | 0.5 | 0.22-0.78 | 6/8 |
| CAM5 | kink | 10/10 | 0 | 0.0 | 0.00-0.28 | 1/10 |
| CAM5 | dark | 0/4 | 0 | n/a | n/a | 0/0 |
| CAM5 | corridor | 9/9 | 2 | 0.2222 | 0.06-0.55 | 4/9 |

### `submitted_plus_both_gates`

| Cam | Stratum | Accepted | Correct | Precision | Wilson 95% | Within 2 |
|---|---|---:|---:|---:|---|---:|
| CAM0 | kink | 1/7 | 1 | 1.0 | 0.21-1.00 | 1/1 |
| CAM0 | dark | 0/3 | 0 | n/a | n/a | 0/0 |
| CAM0 | occlusion | 1/2 | 1 | 1.0 | 0.21-1.00 | 1/1 |
| CAM0 | corridor | 7/9 | 3 | 0.4286 | 0.16-0.75 | 5/7 |
| CAM5 | kink | 1/10 | 0 | 0.0 | 0.00-0.79 | 1/1 |
| CAM5 | dark | 0/4 | 0 | n/a | n/a | 0/0 |
| CAM5 | corridor | 8/9 | 1 | 0.125 | 0.02-0.47 | 3/8 |

## The kink claim, tested on labels it did not select

The claim under test, made on the v2 labels: match labels near a mapping kink miss more often than labels away from one. On v3 the comparison is the kink stratum against the corridor stratum, which is the untargeted control; the v2 labels play no part in it.

For reference only, the v2 numbers that produced the claim: CAM0 0.5 near (4 of 8) against 0.2083 away (5 of 24); CAM5 0.6667 against 0.6087.

| Method | Cam | Kink miss rate | Corridor miss rate | Difference |
|---|---|---:|---:|---:|
| `submitted_unified` | CAM0 | 4/7 = 0.5714 | 4/9 = 0.4444 | 0.127 |
| `submitted_unified` | CAM5 | 10/10 = 1.0 | 7/9 = 0.7778 | 0.2222 |
| `slope_refinement_variant` | CAM0 | 2/7 = 0.2857 | 3/9 = 0.3333 | -0.0476 |
| `slope_refinement_variant` | CAM5 | 9/10 = 0.9 | 7/9 = 0.7778 | 0.1222 |
| `posterior_model` | CAM0 | 2/7 = 0.2857 | 4/9 = 0.4444 | -0.1587 |
| `posterior_model` | CAM5 | 9/9 = 1.0 | 7/9 = 0.7778 | 0.2222 |
| `submitted_plus_slope_gate` | CAM0 | 0/1 = 0.0 | 4/8 = 0.5 | -0.5 |
| `submitted_plus_slope_gate` | CAM5 | 1/1 = 1.0 | 7/8 = 0.875 | 0.125 |
| `submitted_plus_dark_gate` | CAM0 | 4/7 = 0.5714 | 4/8 = 0.5 | 0.0714 |
| `submitted_plus_dark_gate` | CAM5 | 10/10 = 1.0 | 7/9 = 0.7778 | 0.2222 |
| `submitted_plus_both_gates` | CAM0 | 0/1 = 0.0 | 4/7 = 0.5714 | -0.5714 |
| `submitted_plus_both_gates` | CAM5 | 1/1 = 1.0 | 7/8 = 0.875 | 0.125 |

Replication verdict: the kink stratum misses more often than the corridor stratum on CAM0, CAM5. The intervals overlap at these counts, so this is a direction, not a rate.

> **Correction, 2026-09-02 (hand-edited into this Markdown only; `blind_v3_results.json` is untouched).**
> The verdict line above was generated before the CAM5 half of this set was
> withdrawn, and it still names CAM5. **It should read "on CAM0" alone.** The
> CAM5 labels in this set were annotated through a viewer that applied the pose
> compensation with the wrong sign
> (`outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`), so every CAM5 row in
> the table above — including both CAM5 kink and corridor miss rates — is
> withdrawn and must not be read as evidence for or against the kink claim.
> CAM0's offset is measured at 0/0 px and CAM0 stands. The CAM5 replication was
> re-asked on a fresh sealed set, `blind_v3b`, where it **does not** replicate:
> kink miss 0.80 against corridor 0.87, i.e. the kink stratum misses *less*.
> This Markdown is a rendering of the single scoring pass; the pass itself is
> recorded in `blind_v3_results.json` and is deliberately not rewritten.


## Pose convention check

- **CAM0**: annotator median dx 0.0 px (MAD 0.0, range 0.0 to 0.0), dy 0.0 px (MAD 0.0), over 21 match labels. Measured 0.0/0.0 px at 960x720 scale = 0.0/0.0 px; sign agrees: True; rows left at the seeded default: 24.
- **CAM5**: annotator median dx 81.0 px (MAD 17.0, range 10.0 to 200.0), dy 0.0 px (MAD 0.0), over 23 match labels. Measured -37.5/-11.0 px at 960x720 scale = -40.18/-11.79 px; sign agrees: False; rows left at the seeded default: 1.

The CAM5 sliders the annotator settled on have the opposite sign to the seeded default and a larger magnitude: median 81.0 px against the measured -40.18 px, with dy at 0.0 px against -11.79. The one CAM5 row left at the seeded default is the undetermined query, so the default was overridden on every row the annotator could actually align. That is a sign error in how the measured shift was applied to the viewer, not a defect in the labels: the compensation moves the displayed Run B raster and the annotator corrected it by eye per query. CAM0's measured offset is 0/0 and every CAM0 row is 0/0, which is the control.

## Annotator remarks, recorded before the seal

- **cam0 A646** (bracketed_by_occlusion): The house is visible in Run B a few frames before and after, but a truck hides it at the frames where the position matches. Labelled as match with an interval spanning the occluded frames; the preferred frame is the interval midpoint and could not be verified visually. This is a second occlusion instance, distinct from the confirmed runB cam0 240-270 one.
- **cam0 A2399** (turn_and_lateral_offset): Run B is mid-turn for many consecutive frames here and the distance to the pavement differs slightly from Run A, so neither the far+near test nor the centre-line test pins one frame. Labelled as match with +-3 frames.
