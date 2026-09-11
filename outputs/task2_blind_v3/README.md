# Blind label set v3 — targeted, with a better annotator

24 Run A query frames per camera, 48 in total. Every ordinal is at least 10
frames from anything a human has already been shown (the v2 blind set, the
original manual annotations, and the full anchor-candidate list), and the parked
prefix and tail of Run A are excluded, so every query is a frame the vehicle was
actually moving through.

This file is **not** inside the served directory. Neither is
`blind_v3_manifest.json`. Both name each query's stratum, and the manifest also
records the Run B window the dark stratum was drawn from — close enough to an
answer that a directory listing at the annotator's `localhost` must not reach
it. If you are the annotator, stop reading here and use the on-page
instructions.

## Start the annotator

```bash
cd outputs/task2_blind_v3/pages && python3 -m http.server 8732
```

- http://localhost:8732/annotate_cam0.html
- http://localhost:8732/annotate_cam5.html

A local server is required: browsers treat `file://` as an opaque origin, so
answers would stop persisting between reloads.

## What changed from v2, and why

**v2 answered the wrong question.** A uniform sample along the route measures
"how often is the method right on an average frame". It puts almost no queries
in the three places the audit says the method is most likely to be wrong. v3
spends two thirds of its budget there and keeps a uniform `corridor` stratum as
the control:

| stratum | cam0 | cam5 | what it tests |
|---|---|---|---|
| `kink` | 8 | 10 | a large jump in the submitted mapping — is the jump real? |
| `dark` | 4 | 5 | the partner frame lands in a too-dark Run B stretch |
| `occlusion` | 3 | 0 | the confirmed Run B occlusion, where no counterpart exists |
| `corridor` | 9 | 9 | uniformly spaced control |

CAM5 has no `occlusion` row because the confirmed occlusion was seen on CAM0's
side of the vehicle. Its three slots went to two extra kinks and one extra dark
frame rather than to an invented stratum.

**The strata were located using the submitted mapping** — its kink positions,
and its Run A → Run B partner for each frame. That is a real dependency and it
is disclosed in the manifest. It is also bounded: it decides *which frames are
asked about*, never *what the annotator sees*. The exported pages carry the Run
A frame, a linear route-progress start position computed from the route
boundaries alone, and the Run B filmstrip — no stratum tag, no partner, no
confidence, no prediction. `scripts/test_blind_label_set_v3.py` greps each
exported page for the submitted partner of every query and fails if one appears.

When reporting per-stratum rates, say what they are: conditional on the
submitted mapping having put its kinks where it did. They measure "is the method
right where it says something surprising", not an unbiased route-wide rate. The
`corridor` stratum is the unbiased comparison.

**v2's "no usable match" meant two incompatible things.** *I cannot tell which
frame* and *there is no such place* are not the same claim, and only the second
is a correct abstention target. v2 recorded both under one value, so its
no-match rows can be scored as neither. v3 splits them into two labels and makes
the reason a required choice:

| label | meaning |
|---|---|
| `match` | a Run B frame shows the same world position, within the recorded interval |
| `undetermined` | a counterpart may exist; the frame could not be identified |
| `no_correspondence` | no Run B frame shows this place — occluded, or not traversed |

## What is new in the annotator

1. **Pose compensation.** A two-axis rigid shift of the Run B image, with
   sliders, a sign flip and a reset. Default is the measured matcher-convention
   shift scaled to the filmstrip raster: CAM5 `(-37.5, -11)` px at 896×672
   becomes `(-40, -12)` px at 960×720, ON by default; CAM0 is `(0, 0)`, OFF.
   Without it a CAM5 annotator either aligns the scene and mislabels the vehicle
   position, or aligns the vehicle and sees a scene that never matches. The
   chosen shift is written into every label, so the convention is recorded
   rather than reconstructed afterwards.
2. **A far line and a near line**, dragged onto Run A and drawn at the same
   place in both panes. At the correct B frame one common shift aligns both
   depths; parallax breaks the near one first. Positions are saved per query.
   The lines stay in Run A coordinates in both panes — the compensation moves
   the Run B raster onto those coordinates instead of duplicating the shift on
   the lines, which would cancel it out.
3. **Band-restricted blink** — top 45% only, or bottom third only — so the far
   and near depths can be settled one at a time, alongside the full-frame blend,
   difference and blink modes.
4. **No automatic best-frame hint of any kind.** A suggestion would leak a
   matching method into labels whose entire value is being independent of one.

Interval controls are exact, ±1, ±2 and ±3. Widen honestly: a defensible ±3 is
better evidence than a ±0 nobody can stand behind.

| Key | Action |
|---|---|
| `←` `→` | step one frame |
| `Shift` + `←` `→` | step ten frames |
| `Enter` | accept exact |
| `1` `2` `3` | accept ±1 / ±2 / ±3 |
| `U` | cannot determine |
| `X` | no corresponding place |
| `[` `]` | previous / next query |
| `G` | cycle compare mode |
| `P` | toggle pose compensation |

## When the annotation is finished

Click **Download CSV** on each page, then:

```bash
PYTHONPATH=scripts python3 scripts/import_blind_labels_v3.py \
  --cam0 ~/Downloads/blind_v3_cam0.csv \
  --cam5 ~/Downloads/blind_v3_cam5.csv
```

The importer validates blindness, rejects any populated model or stratum column,
rejects an unanswered query, rejects an abstention that still names a Run B
frame, and seals the file with a SHA-256 under schema 3 that cites the query set
hash and the export timestamp. Do not run it before the labelling is complete:
sealing an incomplete file and re-sealing later destroys the ordering guarantee
the seal exists to provide.

CSV columns:

```
query_id, camera_id, runA_frame, label, runB_frame, runB_min, runB_max,
pose_dx_px, pose_dy_px, pose_compensated, far_line_x_frac, near_line_x_frac,
compare_mode, source
```

`pose_dx_px` and `pose_dy_px` are in filmstrip pixels (960×720 basis).
`far_line_x_frac` and `near_line_x_frac` are fractions of frame width.

## Reproducing the export

```bash
PYTHONPATH=scripts python3 scripts/build_blind_label_set_v3.py --force
PYTHONPATH=scripts python3 scripts/test_blind_label_set_v3.py
```

The selection is seeded and deterministic; `query_set_sha256` in the manifest
covers the seed and every query's Run A frame and stratum. Rebuilding changes
nothing unless the inputs change — but it does reset any labelling in progress,
which is why `--force` is required.
