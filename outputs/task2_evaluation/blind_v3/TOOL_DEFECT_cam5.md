# v3 annotator defect: pose compensation applied with the wrong sign on CAM5

Found after the seal, from the sealed labels themselves. Nothing in the label
file was altered; this note records why the CAM5 half of v3 cannot be read as
the mapping's precision, and how that was established.

**The defect.** The measured scene displacement on CAM5 is dx = -37.5 px at
896x672 (scene in Run B sits to the LEFT of where it sits in Run A). To
compensate, the Run B image must be moved RIGHT by +40 px at the 960-wide
filmstrip. The page applied `translate(dx)` with the measured sign, i.e. it
moved Run B a further 40 px LEFT, doubling the misalignment (default -40).
CAM0 has zero measured shift, so its default was 0 and it is unaffected.

**What the annotator did.** On 22 of 23 CAM5 match labels the slider was
dragged from -40 to between +10 and +200 px (median +81, IQR 68-102), dy reset
to 0. The intended rule ("accept the frame where far and near residuals are
equal, whatever the slider says") is slider-invariant only when the two
references sit at clearly different depths; in practice the slider was used to
zero both residuals, which makes frame and shift trade off against each other.

**Evidence that the slider absorbed frame error.** Signed error of the
submitted mapping against the label (positive = prediction later than the
label's interval), CAM5, n = 23: median +3, 17 positive / 3 negative, against
CAM0 median 0. Error against the slider value used: Spearman 0.45, Pearson
0.46; least-squares slope 0.053 frames/px with zero error at dx = 29 px (the
correct compensation is about +40). Labels set with dx >= 120 px have errors
0, 3, 14, 13, 10; labels with dx <= 40 have 0, 1, 4, -4. If the slider fully
absorbed the frame error the slope would be 1/sweep = 0.09 frames/px; the
observed 0.05 says roughly half of the slider excess became frame error.

**Consequence.** CAM5 v3 precision (2/23, 0.087) is a property of the tool,
not of the mapping, and is withdrawn from the headline. CAM0 v3 stands. The
sign is fixed in the builder; CAM5 is re-annotated on a fresh query set under
a new freeze (v3b) so the redo is not contaminated by memory of these answers.
