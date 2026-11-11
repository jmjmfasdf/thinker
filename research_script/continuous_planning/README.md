# Continuous Dynamic Planning v1

This is an additive, observational reanalysis of the completed held-out
Enduro/Pong captures. Existing research scripts and historical outputs are
untouched. Model search is a process-resolved proxy, not observed human thought.

## Entry points

| Script | Question / artifact |
| --- | --- |
| `00_native_cache.py` | Native decisions, controls including STOP, forecasts, arrived comparisons, numeric image cache |
| `01_information_timescales.py` | Current/past visual information and past model evaluations; lag-band ablations |
| `02_maintenance_switch_destination.py` | All-risk switching, conditional destinations, factorized action likelihood, competing transitions |
| `03_evidence_revision_accumulation.py` | Incumbent/alternative evidence and earlier-decision history |
| `04_dynamic_computation.py` | Initiation and voluntary stopping with administrative cap censoring |
| `05_plan_continuity_update.py` | Past paths, prefix-matched arrived image errors, carry diagnostics |
| `06_urgency_modulation.py` | Causal object candidates; annotation-gated urgency inference |
| `07_neural_correspondence.py` | Native-time regressor export only; no fMRI fitting |
| `08_group_report.py` | Cohort acceptance, paired game comparisons, synchronized six-subject inference |
| `09_supplementary_checks.py` | Direct incumbent/alternative comparison, formal horizon contrasts, expanded secondary correction |

Run subject scripts with `--subject`, `--game`, `--capture-index`,
`--output-root`, and optionally a JSON `--config`. `--cache-root` can point to
an existing read-only numeric cache. Never use `--max-episodes` for production.
`integration_smoke.py` creates explicitly marked derived-data subsamples that
the cohort verifier refuses. The smoke JSON has only one C candidate and four
visual PCs; it is NOT used by the production submission script.

## Analysis contracts

- Inputs are manifest/index and numeric native Zarr. The checkpoint YAML is
  read only to verify its capture hash and decoding/control configuration.
  No behavioral NPZ or pickled NPY is loaded. Numeric NPZ in this package is
  exclusively a derived visual/PCA cache.
- Images paired with real-transition rows are post-action. Predictors use
  the preceding image anchor, explicitly matched to `pre_observation_source_index`.
- Human timestamps are the causal 15-Hz decision grid, not measured neural or
  wall-clock computation times. Imagination order has no invented millisecond axis.
- Budget includes additional search computations; root evaluation is separate.
  Expansion/reset/revisit/depth are distinct. A forced terminal computation
  remains a CONTINUE choice; no fictitious final STOP trial is appended.
- A forecast path of length d predicts action t+d-1 and state t+d. State
  discrepancy is eligible for prediction only after the target observation
  arrives, and only with a matching actual action prefix in the primary analysis.
- Historical evidence is re-expressed relative to the CURRENT incumbent,
  including the onset decision where the previous incumbent differed.
- No SEARCH-head versus REAL-head subtraction is called a same-head update.
  First-time branch evaluation and revision of an already visited branch have
  distinct numeric features and missingness/availability indicators.
- Past time bands are fixed in advance: (0,.125], (.125,.25], (.25,.5],
  (.5,1], (1,2], (2,4] seconds. All three EWMA scales (.25, 1, 4 s) are jointly
  regularized, not selected by an in-sample peak. The 8-s delayed-history
  sensitivity is causal and does not wrap future rows around an episode.

## Prediction and statistics

Outer folds are leave-one-source-block-out WITHIN a held-out session. The
feature-generating checkpoint never saw any blocks in that session. Different
leave-session-out feature generators are not mixed in a second-stage fit.
This estimates new-block performance within a held-out session, NOT completely
new-session generalization. Split Zarr episodes from a source remain one fold.

Inner selection holds out source blocks when possible; with one training
block it uses forward temporal validation and an 8-s exclusion before the
validation start. Imputation, scaling, splines and visual PCA fit training
rows only. C is selected from [.001,.01,.1,1,10,100]. The solver is capped at
1000 iterations; failures are recorded rather than accepted as convergence.
Absent destination classes are kept in the fixed 6/9-action space with weakly
penalized intercepts and recorded in fold QA. The incumbent is masked out of
the conditional destination choice set. No active-to-other-active cases are dropped.

Held-out log probabilities and Brier scores are averaged equally over source
runs, then summarized over six independent subjects. Partial fold results are
descriptive only and cannot become a finite population effect. The primary
family is H1 current computation -> switch, H2 current computation -> destination,
H3 past computation -> switch, jointly corrected across both games. All 64
subject sign patterns are used with max-|t| correction, under sign-exchangeability.
Secondary families are corrected within question across games and contrasts.
Two-sided exact p has a minimum of .03125 for six subjects. CI bars are pointwise
95% t intervals, not multiplicity-corrected intervals. A non-significant gain
means additional explanatory power is not established, not evidence of no effect.

Figure risk surfaces display observed-minus-baseline OOF switch probability
in supported quantile bins (at least 20 observations), explicitly labeled as
baseline-adjusted residuals. They are not causal counterfactual surfaces.
Horizon agreement and carry plots are descriptive unless a corresponding
subject-level contrast is tested. Empty/degenerate cells are masked.

09 tests action agreement against persistence/root-policy baselines on ALL
forecast paths, never a subset selected for matching the full action prefix.
State error gains use the prefix-matched subset. Horizon effects are conditional
on forecast availability; a subject must contribute at least two sessions.
09 adds these horizon contrasts and the direct incumbent-minus-alternative
contrast to the complete 03/05 secondary families. For these two families,
`09_supplementary_checks/enlarged_secondary_family_max_t.csv` supersedes the
narrower secondary correction in 08. The six primary tests in 08 are unchanged.
Horizon units are agreement probability or normalized pixel-MSE gain, NOT
log likelihood, even though the shared table uses the `delta_ll` field.

## Urgency and neural gates

The image detectors are candidate implementations, not validated Atari-state
measurements. Exactly 200 frames per game (34 each for subjects 1-2, 33 each
for 3-6) are sampled across sessions with a fixed seed. Annotation templates
require independent visible/absent judgments and coordinates, including missed
objects and false positives. Every required object in a frame must match within
2 pixels; >=90% of frames must pass. Until that gate passes, the urgency result
is explicitly NOT_ESTIMABLE and analyses 01-05 retain their visual/motion controls.
No fake annotation or automated prediction is treated as ground truth.
Each selected frame also has an unannotated PNG for independent inspection.
Pass a completed annotation CSV to 06 with `--annotations`; this writes a
separate hash-named validated variant and preserves the original gated outputs.
The group command can select that variant with `--urgency-annotations`.
For a joint two-game validation report, combine the completed game templates
into one CSV (200 distinct frames per game) and use that same file for all
twelve validated subject runs and the group command.

The neural entry point exports original decision times and leaves scanner
onsets undefined. BOLD, actual scanner alignment, anatomical ROIs and motion
confounds must be supplied before a separate encoding/FIR analysis can run.
It does not generate a neural significance result or infer propagation from BOLD lag.

## Execution and provenance

`submit_campaign.py` defaults to a read-only submission preview. `--submit`
refuses existing campaign directories, freezes this package and the native IO
reader, and creates 12 subject jobs plus one group job dependent on all twelve,
then a supplementary report job dependent on the joint group (14 jobs total).
Each job requests 8 CPU / 32 GiB / 168 hours and explicitly names slave5-8.
The group job validates all 132 source runs / 136 episodes, complete counts
474846/474853, and score-mask counts 474776/474787. It never silently drops a
failed subject. Completed artifacts carry source/config/input signatures and
file hashes. Reanalysis with different sources/configuration requires a fresh
output root. `jobs/submission.json` is the job ledger.
The read-only supplement has its own immutable `supplementary_source` and
`jobs/supplementary_submission.json`; it does not modify the original frozen
snapshot or the completed 08 report. `submit_supplementary.py --submit` can
attach it to a campaign submitted before the supplement was added.

Inspect unit tests with `python -B test_continuous_planning.py`; run them in the
existing Python 3.10 tarball/Singularity runtime, not the host Python 2.
