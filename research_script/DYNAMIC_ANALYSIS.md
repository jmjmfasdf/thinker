# Dynamic Thinker subject analyses

## Extended native dynamic cache (2026-09-23)

`00_cache.py --capture-index ...` (or `--capture-dir ...`) now uses native
adapter version **3**. Existing analysis-table formulas/columns are unchanged.
The additions are separate, joinable files in the same `subXXX_gameY` cache:

| File | Content |
| --- | --- |
| `native_decisions.csv.gz` | Every decision, including B=0: VOC online/EMA Q(CONTINUE)/Q(STOP), Q difference, initial/terminal/at-STOP values, control-index slopes, continue probability, state value, cost, control logits, STOP/censoring, incoming carry, initial/terminal tree size and components, native seconds, recorded action probabilities. |
| `native_events.csv.gz` | All native rows, including final STOP, initial root and real transitions. Explicit validity, accepted controls/actions, tree pre/post components, size/delta, VOC, reward and timing. No event-level VOC trajectory is discarded. |
| `imagination_paths.npz` | Full variable-length action paths for every computation; CSR `path_offsets`, `path_actions`, `path_lengths`, plus `window_index`, `event_index`, `native_event`. Empty root RESET paths remain present. No pickle/object arrays. |
| `forecast_outcomes.csv.gz` | Origin/path/depth, aligned future action/state targets, prefix agreement, reward prediction error and model-value discrepancy. Optional image/latent distances are added here. |
| `representation_summaries.csv.gz` | Only with image/latent options: 24D static image descriptor and/or latent per-channel mean/std. No raw images or vectors are copied. |
| `native_features_schema.json`, `native_features_qa.json` | Semantics/options, per-field validity, control/STOP/forced counts, decision/path coverage and missing tree components. |

Example decision columns: `voc_online_voc_initial`,
`voc_online_q_continue_at_stop`, `voc_online_voc_slope`,
`voc_soft_continue_probability_initial`, `root_carried`, `useful_carry`,
`tree_reachable_node_count_pre`, `tree_reachable_node_count_post`,
`tree_reachable_node_count_delta`, `decision_time`,
`actor_action_probability_0`. Online and EMA summaries remain distinct.
The slope is per control choice, **not per human second**; one-choice B=0
decisions have valid endpoint VOC but undefined slope.

### Joins and interpretation

- Join decision rows to legacy window tables on `real_step_id` (also keep
  `run_label`/`real_step`). `window_index` joins paths to decisions. Join native
  events/representation summaries on `(episode_id, native_event)`; event indices
  restart in each episode. CSR paths use computation order for `event_index`,
  whereas native `event_order` includes STOP and other native rows.
- Carry comes from the **incoming root**, not the root after the current real
  action. Tree initial/terminal summaries end before the real action executes.
  Native visit counts are decoded counts, without the legacy budget denominator.
- `actor_action_probability_*` is the recorded probability vector, not a softmax
  of it and not an alias for the legacy `actor_policy` probabilities. Search and
  real heads are not interpreted as a same-head policy revision.
- Missing/invalid is NaN with validity/count columns; observed zero stays zero.
  A branch Q with zero visits is not evidence of having evaluated that branch.
  Masked logits may legitimately be `-inf`.
- Voluntary STOP and forced termination are separate. `stop_choice_censored=1`
  means no voluntary final STOP was observed. Q(CONTINUE) at STOP is a **model
  estimate**, not an observed causal "what would have happened" outcome.
- Depth d predicts action `a[t+d-1]` and state `x[t+d]`, bounded by the same
  episode. `prefix_before_match` excludes the target action; `prefix_full_match`
  includes it. All forecasts, including divergent paths, are retained.
- `forecast_available_time` is the origin time; `outcome_available_time` is
  when the target observation becomes available. **Mismatch must never be used
  as an origin-time predictor.** For matched-path state-error analyses require
  `state_comparison_primary=1` and the relevant `*_comparison_valid=1`.
  Final captured `x_T` has no invented next-decision timestamp: its outcome
  time is NaN but its native event reference is retained.
- Native real-transition rows have pre-action and post-state availability
  separately: their displayed image is `x[t+1]`, not `x[t]`.
- Reward error uses the native observed reward's first component at the target
  transition. Value/latent discrepancies compare model representations and are
  not ground-truth future value or human neural prediction error.
- Image distances use RGB in [0,1] and a fixed static descriptor, with the
  origin frame as a persistence baseline. `predicted_xs` uses the latest three
  channels of the recorded stack, following recorder display conversion.
  Latent comparisons require a target node marked as containing real state.
  PCA is deliberately not fitted on the whole dataset: downstream analyses
  must fit dimensionality reduction within training folds.

### Build options

Scalar/tree/control fields and full paths are enabled by default. Add
`--include-prediction-summaries` for predicted/imagined-vs-real images and
`--include-latent-summaries` for latent summaries/distances. Heavy reads are
bounded by `--native-summary-batch-size` (default 128 rows).

For example, run inside the existing Python 3.10 tarball/Singularity environment:

```bash
python /home/jeongmin/thinker-dynamic-imitation/research_script/00_cache.py \
  --subject 1 --game 0 \
  --capture-index /home/jeongmin/thinker-dynamic-imitation/logs/visual_behav/heldout_5m_20260921/subjects/sub001_game0.json \
  --cache-root /home/jeongmin/thinker-dynamic-imitation/research_script/outputs/00_dynamic_cache_extended \
  --include-prediction-summaries --include-latent-summaries
```

Use game 1 and its matching index for Pong. `--max-files 1` selects one entire
source run, retaining all of its episodes. Reusing a cache requires matching
input signature, source hashes, options and completed artifacts. Old caches are
not silently treated as extended: choose a new root, or explicitly use
`--overwrite` to move the old subject/game directory to a timestamped backup.
These code changes do **not** rebuild existing caches or submit jobs.

Regression tests (temporary directories, not `outputs/`):

```bash
python -B -m unittest discover \
  -s /home/jeongmin/thinker-dynamic-imitation/research_script \
  -p 'test_dynamic*.py' -v
```

## Full held-out campaign (2026-09-21)

`00_cache.py --capture-index logs/visual_behav/heldout_5m_20260921/subjects/sub001_game0.json`
reads the four **real** completed capture manifests for one subject/game. Each
leave-session checkpoint contributes only its held-out session. Source hashes,
checkpoint identity, session coverage and native completion proofs are checked;
split episodes remain one source/LORO run and never create cross-episode windows.
The analysis formulas are unchanged. The old sub001 numeric acceptance fixture
is now opt-in (`verify_dynamic_analysis.py --legacy-sub001-fixture`); new caches
are checked against each capture's actual replay edge counts instead.

Run `python3 research_script/submit_heldout_analysis.py --submit` once to freeze
the relevant source files and submit 12 subject jobs plus two game-group jobs
with `afterok` dependencies on all six subject jobs. This entry point refuses
to overwrite an existing campaign directory. The pre-relocated, checksum-checked
tarball runtime is mounted read-only in Singularity; no environment unpacking
is performed. Jobs request 8 CPUs, 24 GiB and 168 hours. Per the user's updated
placement policy, all future jobs explicitly request **slave5-8 only**, with
three subject jobs per node. Direct PBS-template submissions default to slave5.

New results are isolated under `outputs/heldout_5m_20260921/`:

- `00_dynamic_cache/sub{001..006}_game{0,1}`
- `02_evaluation_revision_geometry/sub{001..006}_game{0,1}` and `group_game{0,1}`
- `03_actor_integration_mechanisms/sub{001..006}_game{0,1}` and `group_game{0,1}`
- `04_search_evidence_geometry_pathway_v2/sub{001..006}_game{0,1}` and `group_game{0,1}`
- `dynamic_jobs/submission.json`, live job logs, stage timings, and independent
  subject/group `.completed.json` verification reports
- `source/`: frozen analysis sources and checksums

### 04-only rerun, Actor solver maxiter=200 (2026-09-21)

In this preserved run, the Actor soft-target L-BFGS-B cap was raised to 200
(previously 180). Its frozen sources remain under that campaign's `source/`.
Objectives, features, regularization, convergence tolerances, fold inclusion,
and statistical models are unchanged. Nonconverged fits are still rejected;
logs now identify the failed model/run and the optimizer's termination reason.
The cap is recorded in manifests and the Actor derivative implementation hash.

```bash
python3 research_script/submit_heldout_analysis.py --only-04 \
  --shared-cache-root "$PWD/research_script/outputs/heldout_5m_20260921/00_dynamic_cache" \
  --output-root "$PWD/research_script/outputs/heldout_5m_20260921_04_maxiter200" \
  --submit
```

This refits all twelve subject 04 analyses from the existing read-only numeric
caches, followed by two `afterok` group jobs. It does not rerun or change 00/02/03,
old 04 outputs, or old frozen snapshots. All new subject/group analyses use the
same new source checksum. Results are under
`outputs/heldout_5m_20260921_04_maxiter200/04_search_evidence_geometry_pathway_v2/`.
The verifier's `--only-04 --shared-cache-root PATH` mode checks cache alignment,
the 200-iteration provenance, all 11 Actor folds, declared table checksums,
nonempty figures, and explicit reasons for unestimable Stage 1 effects.

The 200-iteration rerun completed Pong's 04 group, but Enduro sub004 still hit
the cap in four A3 fits, so its dependent 04 group was deleted by PBS. Missing
Enduro 02/03 groups are independently submitted as job `1478835.bml2-cluster`
using `heldout_group_0203.pbs` on slave5. All 04 subject jobs had ended before
this submission; the earlier dependency submission `1478834` was rejected
because its prerequisite had just finished. The job uses the original frozen
02/03 sources and completed subject tables. Group outputs go into the original
`heldout_5m_20260921` campaign; a separate
`dynamic_jobs/group0203_game0.completed.json` verifies only those two analyses.
Pong's existing 02/03 groups are preserved, not recomputed. Enduro 04 remains
blocked on the failed subject in the 200-iteration run; this supplementary job
does not repair or bypass it. The supplementary 02/03 group job subsequently
completed and passed verification, so both games now have complete 02/03 groups.

### 04 rerun: maxiter=1000 for both games (2026-09-21)

The user requested a 1,000-iteration Actor soft-target L-BFGS-B cap for **both**
Enduro and Pong. Only that numerical limit changes; objectives, regularization,
tolerances, preprocessing, fold coverage requirements and statistical models
are unchanged. The verifier and solver regression test now expect 1,000.

```bash
python3 research_script/submit_heldout_analysis.py --only-04 \
  --shared-cache-root "$PWD/research_script/outputs/heldout_5m_20260921/00_dynamic_cache" \
  --output-root "$PWD/research_script/outputs/heldout_5m_20260921_04_maxiter1000" \
  --submit
```

This submits twelve fresh subject 04 jobs and two dependent group jobs using
one consistent frozen source version, on slave5-8 only (8 CPUs, 24 GiB, 168h).
It refits Actor derivatives from the original read-only numeric caches and
does not reuse the 200-iteration Actor predictions. Original 00/02/03 results,
all completed 02/03 groups, previous 04 results, and earlier snapshots remain
unchanged. New results and verification reports are isolated under
`outputs/heldout_5m_20260921_04_maxiter1000/`.

### Current 04 rerun: Stage 3 minimum training rows=400

After diagnosing Enduro sub004's Stage 3 failure, the user explicitly lowered
the minimum training rows from 500 to 400. The old rule rejected two
NOOP-to-active folds with 424 and 494 training rows even though both classes
were present. Both the outer fold check and `_fit_binary_l2` now use
`STAGE3_MIN_TRAIN_ROWS=400`; the threshold is recorded in subject/group manifests
and Stage 3 QA. The Actor soft-target fit retains its separate minimum of 500
and its 1,000-iteration cap. No feature, outcome, complete-case, regularization,
convergence or LORO definition changes. This user-directed threshold revision
does not establish statistical stability for small held-out runs.

```bash
python3 research_script/submit_heldout_analysis.py --only-04 \
  --shared-cache-root "$PWD/research_script/outputs/heldout_5m_20260921/00_dynamic_cache" \
  --output-root "$PWD/research_script/outputs/heldout_5m_20260921_04_maxiter1000_mintrain400" \
  --submit
```

Twelve subject jobs and two `afterok` group jobs run on slave5-8 using the same
frozen revision. Earlier outputs and completed 02/03 groups remain untouched.
Boundary tests cover 399 (reject), 400/424/494 (accept with both classes), and
the complete 11-fold Stage 3 pipeline with exactly 400 training rows per fold.

The sections below document the earlier single-checkpoint sub001 dataset and
runner, which are preserved and are **not** mixed into the full held-out results.

The native input path is `00_cache.py --capture-dir CAPTURE`. It reads the
completed capture manifest and episode Zarr stores, not behavioral source NPZs
or historical `video_stat_*.npy`. Source NPZ names are provenance identifiers
only. Numeric CSV/NPZ derivatives remain the interface to analyses 02, 03 and 04.

## Current dataset

| Capture | Subject/game | Actions | Source runs | Episodes | Complete real steps | Mean budget | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Enduro step 5,000,000 | 1/0 | 9 | 11 | 12 | 79,136 | 0.787593509907 | 30 |
| Pong step 5,000,000 | 1/1 | 6 | 11 | 11 | 79,137 | 0.936350885174 | 15 |

Both captures are under `logs/visual_behav/full/`, named
`{enduro,pong}-sub-001-allsessions-step5000000-images-all-checkpoint`.

## Alignment and compatibility

- Adjacent real anchors within one Zarr episode define one complete window.
  All windows, including budgets 0 and 1, remain in the cache.
- Episodes from the same source share a LORO run label. Real-step numbering has
  a gap between episodes, preventing cross-episode bouts/windows.
- Native visit arrays encode `log1p(count)`. The adapter decodes them and stores
  `count / (n_status2 + 1)` for the historical table contract. Analysis 03's
  existing count conversion is unchanged.
- Native policies are logits, converted to probabilities. The initial actor
  policy is the first search-source policy, or the terminal policy for budget
  zero. The terminal policy comes from the same window's real transition.
- Search paths are reconstructed from accepted actions and checked against
  native pre/post depths. A RESET-to-root computation without a branch still
  contributes to budget, but not action-level geometry comparisons.
- The 40-dimensional visual formula is unchanged, applied to native real
  observations after CHW-to-HWC conversion. Current and preceding real anchors
  stay within the episode. The first window has no past descriptor.
- Analysis 04 uses strict future suffixes (`MIN_FUTURE=1`), an `L//2` early/late
  split, and the original slope/complete-case rules. Stage 1 requires 40
  informative windows and an estimable design. Non-estimable effects are NaN
  with reasons; the prespecified inference family is not reduced.
- Temporal action controls use the full real-step spine, including short-budget
  windows, and do not carry history across episodes.
- Only subject analysis is run. No group analysis is generated.

The reconstructed shared modules support this dynamic numeric-cache pipeline;
the old NPY extraction pipeline is not restored or validated by this change.

## Run and verify

From the repository root, the two full CPU jobs can be submitted with:

```bash
qsub -N dyn-analysis-enduro -v ANALYSIS_INDEX=0 \
  -o "$PWD/research_script/outputs/dynamic_jobs/enduro.pbs.log" \
  research_script/dynamic_analysis.pbs
qsub -N dyn-analysis-pong -v ANALYSIS_INDEX=1 \
  -o "$PWD/research_script/outputs/dynamic_jobs/pong.pbs.log" \
  research_script/dynamic_analysis.pbs
```

Each job requests 8 CPUs, 24 GB RAM and seven days, unpacks the existing
SHA-verified Python tarball into job-local storage, and runs inside the existing
Ubuntu Singularity image. It executes cache → 02 → 03 → 04 → verification.
Completed analysis directories must be deliberately backed up or rerun with
the scripts' `--overwrite` option; the job does not silently replace a completed
04 manifest. Do not edit analysis sources while these jobs run.

Final directories under `research_script/outputs/`:

```text
00_dynamic_cache/sub001_game{0,1}/
02_evaluation_revision_geometry/sub001_game{0,1}/
03_actor_integration_mechanisms/sub001_game{0,1}/
04_search_evidence_geometry_pathway_v2/sub001_game{0,1}/
```

`dynamic_jobs/` contains stage timing reports, delivered PBS logs, and successful
`sub001_game{0,1}.completed.json` verification reports. Analysis 04's
`qa/stage1_not_estimable.csv` enumerates missing effect estimates.

Regression tests and independent artifact verification:

```bash
/home/jeongmin/vbenv-25d766c04fe7/bin/python -B -m unittest discover \
  -s research_script -p test_dynamic_analysis.py
/home/jeongmin/vbenv-25d766c04fe7/bin/python -B \
  research_script/04_search_evidence_geometry_pathway_v2.py --self-test
/home/jeongmin/vbenv-25d766c04fe7/bin/python -B \
  research_script/verify_dynamic_analysis.py --game 0
/home/jeongmin/vbenv-25d766c04fe7/bin/python -B \
  research_script/verify_dynamic_analysis.py --game 1
```
