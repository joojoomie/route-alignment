# Blind label set v2 — annotation instructions

40 Run A query frames per camera, 80 in total. These ordinals were never touched
by the original 36 anchors (minimum distance 10 frames), and the parked
prefix/tail of Run A is excluded, so every query is a frame the vehicle was
actually moving through.

## Start the annotator

The pages need a local web server — opening them straight from the filesystem
works for the images but browsers treat `file://` as an opaque origin, so your
answers would stop persisting between reloads.

```bash
cd outputs/task2_blind_v2 && python3 -m http.server 8731
```

Then open:

- http://localhost:8731/annotate_cam0.html
- http://localhost:8731/annotate_cam5.html

## What you are deciding

For each Run A frame, find the Run B frame showing **the same place**.

The Run B panel opens at a linear route-progress guess. That guess is
arithmetic on the route boundaries — no model, no previous prediction — and it
is deliberately weak: the true offset drifts across a 124-frame span over the
route, so expect to scrub well away from it. You can reach any Run B route
frame.

| Key | Action |
|---|---|
| `←` `→` | step one frame |
| `Shift` + `←` `→` | step ten frames |
| `Enter` | accept the current frame as an exact match |
| `N` | no usable correspondence |
| `[` `]` | previous / next query |

Buttons cover the same actions plus ±1/±10/±100 jumps and *Reset to prior*.

**Accept** records an exact match (interval width 0). **Mark ± tolerance**
records the same frame with a ±1 acceptable interval — use it when you are
confident about the place but not the exact frame. Being honest here matters
more than being precise: a widened interval that reflects real uncertainty is
better evidence than a narrow one you cannot defend.

**No usable match** is a real answer, not a failure. Parts of the route may have
no counterpart, and the run ends near where it began, so a frame in the closure
region can be genuinely ambiguous. Say so rather than forcing a match.

## When you are done

Click **Download CSV** on each page, then import both:

```bash
PYTHONPATH=scripts python3 scripts/import_blind_labels_v2.py \
  --cam0 ~/Downloads/blind_v2_cam0.csv \
  --cam5 ~/Downloads/blind_v2_cam5.csv
```

The importer validates blindness, rejects any populated model column, and seals
the file with a SHA-256 recorded before the method freeze reads it.

## Why this is worth two hours

The current submission's blind estimate rests on 6 and 4 accepted cases, giving
a Wilson interval of roughly ±0.3 — wide enough that it cannot distinguish a
good method from a mediocre one. Forty labels per camera bring that to about
±0.15 and make a risk–coverage curve meaningful.

Answers are saved in your browser as you go, so you can stop and resume. Do not
clear site data for localhost:8731 before downloading.
