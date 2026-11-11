# Continuous Dynamic Planning V2

This is an additive, post-hoc reanalysis. V1 output, its frozen source, and the
original research scripts are read-only inputs. Human thinking is not directly
observed; model computations are behavioral comparison variables.

## Interface and sequence

Use the shared tarball environment under Singularity (see `campaign.pbs`).

```
python run.py --subject 1 --game 0 --v1-root ../outputs/continuous_planning_v1 \
  --output-root ../outputs/continuous_planning_v2
python run.py --subject 1 --game 0 --render-only
/usr/bin/python3 submit.py --phase pilot --submit
```

`run.py --stage` supports `derive`, `fit`, `tables`, `render`, `verify`, and
`group`. `--render-only` reads existing numeric panel bundles; it cannot extract,
fit, retune, or recompute significance. `--capture-index` is checked against
the V1 native cache identity. `--config` applies only to new fits. The default
new-fit tuning grid, training-fold PCA and eight-second inner embargo are the
frozen V1 settings. No cross-checkpoint second-stage training is introduced.
Submission runs on the host (standard-library Python 3.6 is sufficient), not
inside Singularity, because PBS clients belong to the host. Interrupted
submissions can resume with `--resume` without repeating recorded job IDs.

## Source modules

- `core.py`: verified read-only V1 bridge, provenance and primary lock.
- `derive.py`: native rollout returns, first versus repeated Q evidence,
  terminal action summaries, accepted-control lagging, image examples.
- `fitting.py`: process additions, R0/R1/R2 soft-target Actor readouts, locked
  human evaluation, and evidence / margin-interaction STOP comparisons.
- `panels.py`: numeric source tables for every panel, inclusion counts and units.
- `render.py`: dedicated legacy-style layouts, 300-dpi PNG and vector PDF.
- `group_report.py`: equal-run subject aggregation, separate new test families,
  six-subject synchronized 64-pattern max-|t|, paired game contrasts.
- `test_v2.py`: fast edge-case tests without creating smoke output directories.

## Statistical separation

Primary H1–H3 are never refit or re-tested. Original effect and run-effect CSVs
are locked byte-for-byte. The canonical V1 primary group table is copied with
identical SHA-256 when available; pending V1 group inference is explicitly
marked pending. New process (4), Actor (12), and control (4) comparisons are
separate families across both games. Destination-history additions are
descriptive and do not create a new primary hypothesis.

Actor readouts have action-specific global coefficients and shared coefficients
for each alternative's own prior, Q, visits, rollout gain and process evidence.
Their soft target and regularization score are exclusively the terminal
REAL-head distribution. Human outcome columns are removed from the fitting
data. Human exact-action evaluation is computed only after held-out Actor
probabilities have been locked. R0 uses pre-search covariates, R1 adds quantity,
and R2 adds search content/process. No nonexistent pre-search REAL head or
SEARCH-to-REAL "revision" is constructed.

STOP predictors use the preceding accepted computation, never final budget or
normalized progress. Forced final CONTINUE is recorded as CONTINUE followed by
censoring; no artificial STOP is added. j=0 is shown descriptively, while the
continuation model concerns j>0. RESET, first evaluation and repeated updates
remain distinct. B=0 retains unavailable evidence rather than fabricated zero.

## Figure inventory

- 02-F1–F7: M × B maps, real-time pre-switch trajectories, individual OOF gains,
  terminal Actor/readout comparisons, competing transitions, initial-state
  computation curves, elapsed-age histories and censored bout survival.
- 03-F1–F3: computation control, Actor readouts, locked human validation.
- 03-S1: arithmetic update QC (not behavioral mechanism evidence).
- 04-F1/F1b: pre-control evidence and initial-margin moderation of STOP.
- 04-F2: long-search early/late comparisons, supplement only.
- 04-F3: existing paired human-model performance; Actor and history are parallel
  extensions and are not connected into one sequential nested model.
- N1: lag-band removal and real-time alignment schematic.
- N2: native image/path examples, alignment/coverage/error, arrived mismatch and
  structural carry. Target-action matching is never a filter for action agreement.
- N3: independent 200-frame-per-game annotation gate. Until passed, C/D remain
  visibly pending and no urgency estimates are fabricated.
- 02-S1: re-evaluation-subset M × endpoint-evidence map.

Each `subNNN_gameG/panels/FIGURE_PANEL.csv` has matching metadata with units,
N, exclusions, availability and test IDs. Empty support is masked, not smoothed.
Run/subject variability is distinguished from event-level sample size. Group
confidence intervals and tests require all six subjects; unsupported cells keep
individual observations and N only. All figure curves are descriptive or
explicit OOF residuals, never causal response functions.

## Execution and review gate

PBS jobs name only slave5–8 and request 168 hours. Pilot submission creates an
immutable source snapshot and two full-data sub001 jobs. After all 19 figures
for each game have been opened and inspected, record actual image hashes in
`visual_review.json` (`status: APPROVED`, `reviewed: {sub001_game0/02-F1.png:
sha256, ...}`). The full submission refuses to run without all 38 matching
reviewed hashes. Do not produce this approval automatically from file existence.
Full submission then adds the other ten subject/game jobs and an afterok group
job. Completed snapshots and outputs are never overwritten for changed code.

The reviewed cohort submission freezes live code into `source_full`, leaving
the pilot `source` snapshot untouched. Review records also contain
`reviewed_sources` hashes for `render.py`, `panels.py`, and `group_report.py`.
Pilot panel/figure revisions are archived under each subject's
`review_archive/` before rebuilding descriptive tables; fitted predictions and
locked primary results are not regenerated. The September 22 review fixes
float-tie counting in descriptive Kaplan–Meier curves, while leaving all model
estimates and tests unchanged. Repeated panel denominators are labeled as
per-bin/per-estimate support, not a fictitious unique sample total.

The output tree is `research_script/outputs/continuous_planning_v2/`, with
`subNNN_gameG`, `group_game0`, `group_game1`, `group_report`, `jobs` and `source`.
