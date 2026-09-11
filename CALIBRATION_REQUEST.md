# Request: camera calibration for the recording rig

## Why this is being asked for

Four separate attempts to recover route geometry from these recordings failed,
and all four failed for the same reason. Recording it here because the pattern is
more useful than any of the attempts individually.

| Attempt | What it tried to recover | How it failed |
|---|---|---|
| Displacement regressed on image row | Separate camera yaw from along-track motion | Slope is noise; a fisheye scrambles row-as-depth |
| Far-field yaw estimate | The same, using slow-moving matches as distant points | Dispersion across pairs is 31 px, as large as the label noise |
| Ego-band pose correlation | CAM5's camera pose change, on a rigid reference | Peak sharpness 0.00; rain fills the band in run B |
| Opposed-camera decomposition | Heading rate, by cancelling translation across CAM0/CAM5 | Flow correlation −0.09; there is nothing to cancel |

Every route-geometry quantity — camera pose, heading, distance travelled,
trajectory — needs a camera model, and none was supplied. The blocker is not
algorithmic. No further processing of these frames replaces it.

## What is blocked right now

* **A pose-invariant definition of "same place."** With intrinsics, the correct
  Run B frame is the one where the relative translation has no along-track
  component. That is computable and needs no annotator judgement. Without them,
  the definition falls back to what a human sees, which is why CAM5's labels sit
  56 px from camera-centric with no way to say whether the camera moved or the
  annotator's rule did.
* **Angle closure.** The route returns to its start, so heading must come back
  around by one full turn. That is a strong global constraint and it is
  currently unusable.
* **Metric statements of any kind.** Nothing can currently be expressed in
  metres, only in frames and pixels.

## What is being asked for

A standard checkerboard or ChArUco session with the rig as mounted, for each
camera:

1. 20–30 images of the target at varied distances, angles and image positions,
   with the target reaching the frame corners — fisheye distortion is estimated
   from the periphery, and a set that only covers the centre will fit the centre
   well and the edges badly.
2. A fisheye or omnidirectional model (Kannala–Brandt or equivalent), not a
   pinhole model with radial terms. The measured image circle is wider than the
   sensor, so this is a genuine fisheye.
3. The rig extrinsics if they can be measured: the transform from each camera to
   the vehicle frame. CAM0 and CAM5 face opposite ways with no overlap, so they
   cannot be calibrated against each other from imagery alone.

## What would change if it arrives

The four failures above become one calculation. "Same place" stops depending on
which cue the annotator happened to have in view. The loop's angle closure
becomes usable alongside its position and path-length closure, both of which are
already implemented and working. And the CAM5 result, currently the weakest half
of the deliverable, gets a diagnosis rather than a description.

## What is worth doing regardless

A higher capture rate would help more than any processing change. At 10 FPS the
scene sweeps laterally across CAM5 at a **median 18 px per frame** at the
896x672 geometry raster — about 29 px at native 1440 width, with CAM5's p90 at
35 and CAM0's median 23 (p90 46), measured by `scripts/build_sweep_rate.py` and
recorded in `outputs/task2_motion_bayes/sweep_rate_summary.json` — and features
survive 2–4 frames; at 30 FPS they would survive long enough to triangulate. An IMU or wheel
odometry would make the trajectory problem straightforward. These are recording
decisions, and they bound what any downstream method can achieve.
