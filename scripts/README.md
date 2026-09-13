# Scripts, by role

Every script here produced something that ships in `outputs/`, is part of the
submitted pipeline, or is a test. `PYTHONPATH=scripts` is assumed (see `env.sh`).

The `v2`, `v3` and `v3b` suffixes are **blind label rounds**, not code versions:
round 2 was the uniform 40-per-camera set, round 3 the targeted 24-per-camera set,
and round 3b the right-camera redo after the annotation tool's sign defect. Each
round has its own export, import, freeze and scoring script because the sealed
artifacts must never be re-scored by newer code.

## Frame identity and integrity

| Script | What it does |
|---|---|
| `atomic_io.py` | Small shared I/O helpers: atomic writes and CSV rows. |
| `detect_route_phases.py` | Detect the initial stationary, departure, and route-in-motion phases. |
| `frame_service.py` | The single canonical decode path for the supplied recordings. |
| `hevc_bitstream.py` | Parse the supplied HEVC Annex-B elementary streams without decoding them. |

## Submitted pipeline

| Script | What it does |
|---|---|
| `build_reliability_masks.py` | Measure which pixels are attached to the camera rather than to the world. |
| `build_task2_unified.py` | One dense frame-correspondence pipeline, replacing the v1-to-v5 chain. |
| `run_geometry_resolution_study.py` | Choose the geometry working resolution from evidence, not from intuition. |

## Development variants

| Script | What it does |
|---|---|
| `vlad_aggregation.py` | Unsupervised VLAD aggregation of DINOv2 patch tokens (development variant). |

## Blind evaluation contract: export, freeze, seal, score once

| Script | What it does |
|---|---|
| `build_blind_label_set_v2.py` | Export a fresh prediction-blind label set and a self-contained annotator. |
| `build_blind_label_set_v3.py` | Export a third, TARGETED prediction-blind label set with a better annotator. |
| `compare_methods_blind_v2.py` | Score both frozen methods head to head on the same sealed blind labels. |
| `evaluate_task2_blind_v2.py` | Score the frozen unified method once, on the sealed blind label set. |
| `evaluate_task2_blind_v3.py` | Score the v3 targeted blind labels once, against the artifacts the v3 freeze names. |
| `freeze_method_v2.py` | Freeze the unified method, and verify a frozen method has not drifted. |
| `freeze_method_v3.py` | Freeze everything the v3 targeted blind labels will be allowed to judge. |
| `import_blind_labels_v2.py` | Validate and seal the annotator's blind label CSVs. |
| `import_blind_labels_v3.py` | Validate and seal the annotator's blind v3 label CSVs. |
| `rescore_blind_v2_undetermined.py` | Rescore the blind set after a label-semantics correction, showing both readings. |

## Label-free evidence

| Script | What it does |
|---|---|
| `analyze_mapping_kinks.py` | Label-free audit of implausible one-step jumps ("kinks") in a frame mapping. |
| `analyze_route_topology.py` | Measure the route's topology from the descriptor cache, so §1.3 has an artifact. |
| `evaluate_loop_consistency.py` | Check a mapping against the route's own closure, using no labels at all. |
| `evaluate_task2_consistency.py` | Dense agreement signals that need no labels at all. |
| `evaluate_tolerance_curve.py` | Precision of every scored mapping as a function of frame tolerance. |
| `evaluate_variants.py` | Score development variants on the evidence that is still available. |
| `run_abstention_holdout.py` | Test abstention by removing stretches of Run B, where the truth is known. |

## Diagnostics behind report numbers

| Script | What it does |
|---|---|
| `build_confirmed_occlusion.py` | Record the one confirmed no-correspondence case in this data, and what the mappings do with it. |
| `build_frame_quality_flags.py` | Flag frames that are too dark to carry recoverable scene content. |
| `diagnose_label_criterion.py` | Detect and correct annotation-criterion drift caused by a camera pose change. |
| `estimate_pose_offset.py` | Where does the matcher put "same place", and can a camera pose shift be undone? |
| `render_aligned_video.py` | Render side-by-side comparison videos of the aligned runs. |

## Posterior model (built, not submitted)

| Script | What it does |
|---|---|
| `build_motion_events.py` | Detect suspension-shake events and measure how much they are worth as evidence. |
| `build_sweep_rate.py` | Per-frame sweep rate: how many pixels the scene moves between consecutive frames. |
| `build_task2_bayes.py` | Frame correspondence as posterior inference, with an explicit null state. |
| `evaluate_task2_bayes.py` | Diagnostics for the posterior model, none of them a held-out accuracy claim. |

## Report, notebook and Task 3

| Script | What it does |
|---|---|
| `build_assessment_notebook.py` | Compose (and optionally execute) assessment.ipynb, the working behind the report. |
| `build_report_tables.py` | Generate the report's result tables from the artifacts. |
| `build_task3_policy_summary.py` | Regenerate the Task 3 dataset-disposition summary from the shipped artifacts. |

## Tests (stdlib unittest, synthetic inputs)

| Script | What it does |
|---|---|
| `test_analyze_mapping_kinks.py` | Unit checks for the kink audit, on synthetic mappings only. |
| `test_blind_label_set_v3.py` | Tests for the targeted blind label set v3. |
| `test_build_frame_quality_flags.py` | Tests for the per-frame exposure flag, on synthetic frames. |
| `test_build_task2_unified.py` | Focused unit checks for the unified correspondence pipeline. |
| `test_evaluate_task2_blind_v3.py` | Unit tests for the v3 blind scoring, on synthetic labels only. |
| `test_evaluate_tolerance_curve.py` | Tests for the tolerance-curve reading of the sealed blind sets. |
| `test_frame_service.py` | Focused unit checks for the canonical frame service and bitstream parser. |
| `test_refinement_slope.py` | Tests for the stage-3 refinement rules: the shipped ratchet and the fix. |
| `test_vlad_aggregation.py` | Tests for the unsupervised VLAD aggregation variant. |
