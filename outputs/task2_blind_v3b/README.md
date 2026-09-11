# Blind label set v3b — cam5

24 Run A query frames per camera. Every ordinal is at least
10 frames from anything a human has already been
shown and at least 10 frames from every other
query in this set, and the parked prefix and tail of Run A are excluded, so
every query is a frame the vehicle was actually moving through.

Excluded sources:

- `outputs/task2_evaluation/blind_v2/blind_v2_labels.csv`
- `outputs/task2_evaluation/blind_v3/blind_v3_labels.csv`
- `outputs/task2_keyframes/manual_annotations.csv`
- `outputs/task2_keyframes/anchor_candidates.csv`

This file is **not** inside the served directory. Neither is
`blind_v3b_manifest.json`. Both name each query's stratum, and the manifest also
records the Run B window the dark stratum was drawn from — close enough to an
answer that a directory listing at the annotator's `localhost` must not reach
it. If you are the annotator, stop reading here and use the on-page
instructions.

- `query_set_sha256`: `b7df8d342d4664f9ccb367bb73d37f6c7aee873bb2679e56e0eda5168ad715e8`
- `built_at_utc`: `2026-09-02T09:57:53.124514+00:00`
- `random_seed`: `20260903`

## Start the annotator

```bash
cd outputs/task2_blind_v3b/pages && python3 -m http.server 8732
```

- http://localhost:8732/annotate_cam5.html

A local server is required: browsers treat `file://` as an opaque origin, so
answers would stop persisting between reloads.

## Strata

| stratum | cam5 |
|---|---|
| `kink` | 5 |
| `dark` | 1 |
| `occlusion` | 0 |
| `corridor` | 18 |

Shortfalls (a targeted stratum with no free ordinal left moves its slots to
`corridor`):

- **cam5 `dark`**: 1 of 5 placed, 4 moved to `corridor` — no free Run A ordinal remained in this stratum at least 10 frames from every previously labelled ordinal and 10 from every other query.
- **cam5 `kink`**: 5 of 10 placed, 5 moved to `corridor` — no free Run A ordinal remained in this stratum at least 10 frames from every previously labelled ordinal and 10 from every other query.

The strata were located using the submitted mapping — its kink positions, and
its Run A → Run B partner for each frame. That decides *which frames are asked
about*, never *what the annotator sees*: the pages carry the Run A frame, a
linear route-progress start position computed from the route boundaries alone,
and the Run B filmstrip. No stratum tag, no partner, no confidence, no
prediction. `scripts/test_blind_label_set_v3.py` greps each exported page for
the submitted partner of every query and fails if one appears.

## Pose compensation — the sign

compensation = -displacement; positive moves Run B right/down. measuredDxPx/measuredDyPx are the raw measured scene displacement of Run B relative to Run A on the measurement raster; dxPx/dyPx are the translate() the page applies to the Run B image to undo it, on the filmstrip raster.

CAM5's measured displacement is `(-37.5, -11)` px at 896×672, so the exported
compensation is `(40, 12)` px at
960×720: Run B moves right and down. The first v3
export shipped the raw measurement instead and doubled the CAM5 misalignment;
see `outputs/task2_evaluation/blind_v3/TOOL_DEFECT_cam5.md`.

## What the page enforces

1. **The sliders are clamped** to the exported default ±25 px,
   with the default drawn as a tick, and a warning line sits under them
   permanently: *do not use the slider to make a frame fit — change the frame
   until FAR and NEAR residuals are equal; only then may a small slider tweak
   zero both*.
2. **The guides must straddle a real depth difference.** The far guide has to be
   in the top 50% of the frame and the near guide in the
   bottom 33%. Both bands are drawn faintly; while
   either guide is outside its band every label button is disabled and its
   tooltip says which guide and why.
3. **Every label records the slider.** `pose_dx_px` and `pose_dy_px` are written
   into the CSV per label, in filmstrip pixels, alongside the guide positions
   and the compare mode in use.
4. **No automatic best-frame hint of any kind**, anywhere on the page.

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

## Labels

| label | meaning |
|---|---|
| `match` | a Run B frame shows the same world position, within the recorded interval |
| `undetermined` | a counterpart may exist; the frame could not be identified |
| `no_correspondence` | no Run B frame shows this place — occluded, or not traversed |

## When the annotation is finished

Click **Download CSV** on each page, then:

```bash
PYTHONPATH=scripts python3 scripts/import_blind_labels_v3.py \
  --set-name blind_v3b \
  --cam5 ~/Downloads/blind_v3b_cam5.csv
```

The importer validates blindness, rejects any populated model or stratum column,
rejects an unanswered query, rejects an abstention that still names a Run B
frame, and seals the file with a SHA-256 that cites the query set hash and the
export timestamp. Do not run it before the labelling is complete.

CSV columns:

```
query_id, camera_id, runA_frame, label, runB_frame, runB_min, runB_max, pose_dx_px, pose_dy_px, pose_compensated, far_line_x_frac, far_line_y_frac, near_line_x_frac, near_line_y_frac, compare_mode, source
```

`pose_dx_px` and `pose_dy_px` are in filmstrip pixels
(960×720 basis), positive = Run B moved right/down.
The guide columns are fractions of frame width and height.

## Reproducing the export

```bash
PYTHONPATH=scripts python3 scripts/build_blind_label_set_v3.py \
  --camera cam5 \
  --set-name blind_v3b --seed 20260903 --force
PYTHONPATH=scripts python3 scripts/test_blind_label_set_v3.py
```

The selection is seeded and deterministic; `query_set_sha256` covers the seed,
the set name, and every query's Run A frame and stratum. Rebuilding resets any
labelling in progress, which is why `--force` is required.

Pose default in this set: compensation
`(40, 12)` px for `cam5`, slider range
`±25` px.
