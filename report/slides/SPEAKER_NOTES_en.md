# Speaker notes (English) — "Reasoning first"

Twenty-five main slides plus three backups; about 90 seconds each, 25 minutes with the backups untouched. The spine of every slide is the same sentence: *this is what I found, this is what I did about it, this is what it settled.*

One rule for the whole talk: never claim more than the numbers carry. The strict figures are in the last table and the first backup; say them before anyone asks.

---

## 1 · Title — *Reasoning first*

Two dashcam traversals of the same loop, two side-facing cameras, ten frames a second. The task was to say, for every frame of the first drive, which frame of the second drive shows the same place, and to say how sure.

The subtitle is the method. I did not start from a model; I started from the files, let what I found decide the design, decided how it would be tested before I tuned anything, and reported the failures with their mechanisms. Everything in the next twenty minutes follows that order.

## 2 · The task, in one sentence

Open with the framing sentence, slowly: *we are aligning two side-camera videos recorded on the same loop route, eleven months apart, without GPS or calibration* — for every frame of the first drive, which frame of the second shows the same place, and how sure. That one sentence carries the scene and every constraint.

Then the findings from the first look at the data, before any model — three difficulties first. First, appearance: a dry afternoon against a rainy dusk, Run B more than twice as dark with rain sitting on the lens, and a side-facing fisheye whose picture changes with every lane shift. Second, there is no sensor to lean on: no GPS, no IMU, no odometry, no calibration anywhere in the files — I parsed the bitstream to be sure. Only the pixels. Third, and hardest: long repetitive stretches. Fence corridors and tree tunnels where one frame cannot be told from its neighbours, or from a look-alike a thousand frames away; CAM5 has two and a half times CAM0's density of such pairs.

Add the fourth finding, briefly, and promise to come back to it: some behaviours looked like noise at first — CAM5 points differently in Run B, the car changes lanes, speed bumps jolt the picture. Each got measured and each got its own treatment; that is slide 17.

Then lift it one level. Nobody recognises a single fence frame. A person solves this by scrubbing and watching what comes before and after: the order of events disambiguates what one image cannot. So the algorithm should use the temporal context too, and decide per sequence rather than per frame.

Land the transition, and pause on it: *so I framed it as a sequence alignment problem instead of independent image matching.* Then hand over to stage 1 with: *first, I make the videos reliable before any matching.*

## 3–17 · The pipeline: three stages, one to three slides per step

Nine slides in three stages: *Stage 1 · Prepare the frames* (1.1–1.3), *Stage 2 · Sequence matching* (2.1–2.3), *Stage 3 · Geometric verification and decision* (3.1–3.3). Each slide shows the whole strip of its stage with the current step outlined in amber, and a line under it says which of the three stages we are in. Below: a two-column table, one row per problem — left the problem, right the method chosen for it with its parameters and a one-sentence reason. Read row by row: problem, then method, then the reason; the slide carries the parameters so you do not have to. Steps with a picture or a table take two slides: the first states the problem with the picture, the second gives the method and the rationale. 1.2 the temporal-variation map and the two masks; 2.2 the loop's two ends and a sketch of the forward-only path; 2.3 the jump failure; 3.2 the resolution-study table; 3.3 the lorry. Old note kept for the pictures: 1.2 the temporal-variation map and the two masks (drops appear only on the rainy drive), 2.2 the loop's two ends (why the path may not wrap), 2.3 the jump failure (two different houses at top confidence), 3.3 the lorry (a place with no usable view). Point at the picture before reading the method. Read the bridge line on the first slide of each stage: stage 1, sequence alignment needs trustworthy frames, one path over the whole sequence, and checks on what the path cannot see; stage 2, sequence matching, coarse to fine, finds roughly where each frame belongs; stage 3, geometric verification and decision — check the match is unique, confirm the exact frame on the pixels, decide whether to answer. Transition sentence: Stage 2 tells me roughly where I am by sequence matching; Stage 3 confirms the exact frame by geometric matching, and decides whether to answer. On each slide: left, the problem I saw; right, why this choice, written the way I would say it; bottom, what it settles. Say the problem, then read the quote, then the one-line settle; do not go back to earlier stages.

1.1 **Decode one fixed way** — this slide now carries the five file findings (stored order, two frame rates, missing colour standard, one log for two cameras, the drives' dates). Walk them top to bottom, then the quote: a small parser rebuilds the order, decode from the original with colour fixed, 24 checks, 22 pass, 2 waived with the reason on record.
1.2 **Mask the camera's own artefacts** — open with the line on the slide: before extracting descriptors, I mask fixed artefacts like raindrops and mirrors using temporal statistics. Third slide is the SegFormer story in one sentence — I first tried SegFormer, but it doesn't cover raindrops, mirrors or dark vignetting, so I switched to a temporal-statistics mask; it masked people, cars and sky, had no class for drops, mirror or vignette, and on the 8-pair development set removed one false accept while cutting retrieval recall@5 from 0.71 to 0.57; the switch to the temporal statistic found the rain exactly where it is (right camera 1.2 → 9.9%); say plainly that the mask's effect is not isolated in a held-out test. — drops, mirror, dark corners; what never changes over the drive belongs to the camera; one rule, no model.
1.3 **Describe each frame** — say: because the runs are months apart, I start with frozen DINOv2 patch features and cosine similarity to get robust coarse matches under lighting change. Pause, then the pooling over mask-trusted patches.
2.1 **Compare with the whole other drive** — this page answers *where do I search?* Say: at full resolution, matching every frame is too expensive — pause — so I start with a coarse pass, sampling every fourth frame. Then: a local band is risky because the offset drifts by around 12 seconds, so I compute a full similarity matrix but make it cheap by subsampling. Look-alikes are "sometimes hundreds of frames apart"; do not read the 300–1,300 figure.
2.2 **One forward-only path** — this page answers *how do I find the path?* Dynamic time warping, forward only, because order is what a person uses; then: because the drives were recorded at different speeds, I allow the slope to vary within measured limits; then the loop seam. Parameters are in the grey line at the bottom — only if asked.
2.3 **Settle every single frame** — from every-fourth to frame-exact; then three points, no more: this stage has no slope prior, so where the score is flat the optimiser runs to the window edge and produces jumps at high confidence; I reported that as it is; after the freeze I built a Viterbi version with a slope prior that avoids the jumps by construction, kept as reference evidence because it came after the freeze.
3.1 **Uniqueness check** — some corridors and tree stretches look almost identical, so one frame is easily misjudged; I compare the best match against the second-best at several window scales to make sure the stretch is genuinely distinctive; confidence reflects uniqueness, not strength — the size of the gap, not the height of the score.
3.2 **Check the actual pixels** — geometric verification to filter false matches: RootSIFT plus ratio test and RANSAC, a pixel-level check on the masked images; support required across several grid cells so one textured patch cannot mislead it; 896 by 672 chosen by a resolution comparison as the best balance of reliability and cost.
3.3 **Answer, or say "no answer"** — the decision logic: after uniqueness and geometry the system decides whether to give a match at all; some frames should have no answer (blocked, very dark, no corresponding segment), so the output carries an explicit type with a clear reason for each refusal. The key point: a wrong answer becomes ground truth downstream, not answering only costs coverage, so no answer beats a confident wrong one. HMM in one sentence: confidence is a ranking, not a calibrated probability; I separately built an HMM with a null state for calibrated posteriors, reported after the freeze, not part of the submitted method.

Close on step 3.3 with: the test was decided before any of this was tuned.

## 18–19 · After the pipeline — Three behaviours that looked like noise

Two slides, one table each: behaviour on the left, handling on the right. Slide 18: the right camera is mounted at a slightly different angle in the second drive; say the consequence first — same picture and same position of the car are no longer the same frame, so the label has to be based on the real position — then the size, a constant offset worth about minus 1.8 frames, an angle estimated because there is no calibration. Handling: matching survives because the pixel check tolerates a shift; the labelling page pre-shifts the second drive; the leftover constant is measured, confirmed on a fresh set, and shipped as a field. Slide 19: lane changes — the path is decided over a stretch of video so a few frames of changed view cannot move it; speed bumps — both cameras share one chassis, so a real jolt must appear in both at the same frame, which became a label-free test; the submitted mapping is unchanged either way.

## 20 · Evaluation — How would anyone know it works?

Three points. The field checks against GPS; this data has none. So the only reference for "same place" is a person looking at two frames — the instrument is a person, which means: define same place precisely, show the person nothing the method thinks, freeze first, score once, and test the instrument itself. And automatic checks only test consistency; things can be consistently wrong — the lorry proved it.

## 21 · Evaluation — The instrument: a blind labelling page

Point at the screenshot: query on the left, candidate on the right, opened at a plain guess, three answers, nothing from the method on the page. Then the three rules: far and near guide lines — the near one pins the frame; lane changes — use the object nearest the centre; right camera — the page pre-shifts it, the slider is not there to make a frame fit.

## 22 · Evaluation — The contract around the instrument

Read the timeline: export, freeze, annotate, seal, score once. Then two sentences: this proves order, not custody — it does not prove nobody peeked, and the report says so; and it was done three times, round 2 evenly, round 3 targeted, round 3b a redo after the instrument failed a test.

## 23 · Results — Precision at frame tolerances

Scale first: a frame is a tenth of a second, about a metre; I report at ±2 and show ±10, the field's tolerance. Then two sentences: errors are small, not gross — 97 to 100 percent within ten frames, and the right camera's residual has a known cause; coverage is not accuracy — the earlier method answered half and a tenth of the queries.

## 24 · Results — How strict this is, compared with published work

Not the same data, context for the scale only: benchmarks count a hit at 25 metres or ten frames, here a frame is a metre, so ±2 is about ten times tighter, and at the field's own tolerance the result is 97 to 100 percent.

## 25 · Future work, in order

Five one-liners: ship the Viterbi refinement after a fresh blind round; the learned-vocabulary pooling; a learned point matcher for the rain; camera calibration, because the mount angle is estimated today; a "no such place" label set, the one thing no blind round could test. Close on the rule, and name it: no model training, no data leakage. No model was trained on this data and no label ever entered the pipeline; the method was frozen before each label set was opened, and nothing tuned after a set was read gets that set's number.

## 26 · Lessons learned

Three points, then stop. Test the instrument, not just the method: the labelling tool failed twice and I found both by treating the labels as data; both are in the report with their timing. Consistency is not correctness: everything agreed on the lorry and everything was wrong — a held-out number is worth exactly the process that produced it. Freeze first, then improve, and keep the two apart: the Viterbi refinement and the learned pooling are better than what I submitted and stay evidence, because they came after the labels were opened.

## Backups

- **Strict single-frame table**: open when asked "what is the accuracy, really". 0.72 and 0.38, with Wilson intervals, and the uncorrected column beside them.
- **The right camera's mechanism** now lives only here: the camera moved, so "same picture" and "same car position" differ by a constant of about −1.8 frames, measured before round 3b and confirmed on it (0.14 → 0.67 at −2). Open when asked why the right camera is weaker.
- **CAM5 constant sweep**: open when asked whether minus two was fitted. Point at v2 neutral and CAM0 peaking at zero.
- **Label-free evidence and variants**: open when asked what else supports the result without labels, or why VLAD is not submitted.
- **Downstream policy** (Task 3): open when asked what a consumer may do with the mapping — keep / exclude / quarantine, per-frame fields, the two stated facts.
- **Canonical decode**: open when asked what the frame-identity work actually was. Raw Annex-B streams; order, colour, rate and time are all things a decoder gets silently wrong; one pinned path, display-order ordinals, composite cache keys, gates that fail.

---

## Questions I expect, with the answer in one breath

**Why so strict — the field uses 25 metres?** Because the output is training data, not a retrieval demo. A confident wrong frame becomes ground truth downstream; an abstention just costs coverage. So I score at the frame, report at plus-minus two because that is usable, and show plus-minus ten to place it against the field.

**So is there zero false acceptance?** Gross ones, zero on the two targeted sets — every accepted match within ten frames. On the uniform set, one per camera out of 32 was beyond ten frames. Strictly, a quarter of CAM0's accepted matches are a frame or two off. That is the honest reading and it is why the policy quarantines.

**Why manual labels and not GPS or a map?** There is no GPS in the files — I parsed the SEI; the only payload is pic_timing. Without a position reference, "same place" exists only as a human judgement, so the instrument is a person, blinded, and the method is frozen before they start.

**Why not submit the better variant?** The contract: the submission is what was frozen before the labels were opened. Swap it and there is no held-out number. The variants are reported as evidence with the label status attached.

**Was withdrawing v3 CAM5 hiding a bad result?** The redo reads 0.14 strict — not better. What changed is the explanation of the bias, and the evidence for withdrawal was the sealed labels' own statistics plus CAM0's zero-offset control.

**Why is minus two not curve-fitted?** It was measured on v2 before v3b existed, minus one and minus three are worse, v2 is neutral as a mixed set predicts, and CAM0 peaks at zero.

**Can you evaluate abstention?** Not on v2 — a protocol defect. The holdout measures the mechanism; the two real absences were missed by every method including the posterior's null state, and I say so.

**Why so many scripts?** Each corresponds to one number's provenance in the report; the notebook calls them in the report's order and checks they agree with the shipped artifacts.
