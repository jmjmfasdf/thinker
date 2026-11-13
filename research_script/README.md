# Dynamic Thinker: research themes

The maintained entry points below separate the scientific questions. New
results always live directly below `research_script/outputs/NN/`. The imported
`outputs/00_dynamic_cache_10m_20261110` is a read-only source; its 72 checkpoint
caches, previous results, frozen training sources and `logs/` are preserved.

| Entry point | Question | Output |
| --- | --- | --- |
| `00_cache.py`, `00_brain_cache.py` | Native cache and verified behavioral/BOLD alignment | `outputs/00/` |
| `00_training_diagnostics.py` | Actual checkpoint training condition | `outputs/00/training/` |
| `01_behavioral_analysis.py` | Human action durations and transitions | `outputs/01/panels/` |
| `01_model_fidelity.py` | Recorded REAL-head human action likelihood | `outputs/01/model_fidelity/` |
| `02_action_boundary_mechanisms.py` | M/U/K and human persistence or switching | `outputs/02/` |
| `03_dynamic_computation_control.py` | Pre-choice VOC, actual continuation, start/stop and budget | `outputs/03/` |
| `04_search_to_action.py` | Completed search evidence, subsequent control and final REAL policy | `outputs/04/` |
| `05_human_prediction.py` | Incremental held-out exact action and bidirectional transition prediction | `outputs/05/` |
| `06_neural_decoding.py` | Incremental multivoxel decoding of computational variables | `outputs/06/` |
| `07_neural_rsa.py` | Controlled computational–neural representational correspondence | `outputs/07/` |
| `08_neural_encoding.py` | Separate localization and held-out BOLD encoding | `outputs/08/` |
| `09_task_state_complexity.py` | Visual/RAM state change and allocation; observational Gramian | `outputs/09/` |
| `10_computation_utility.py` | Existing paired environment STOP/CONTINUE benefit | `outputs/10/` |
| `12_continuous_dynamics_analysis.py` | Prior evaluation, accumulated evidence, recorded carry and changing context | `outputs/12/` |
| `13_behavioral_modes.py` | Exploratory phase-controlled HSMM, from raw human RAM data | `outputs/13/` |

`11_planning_budget_histogram.py` is an alias for 03. Older numbered names are
compatibility launchers, not additional primary analyses. Their original
implementations are archived with SHA-256 in `legacy/pre_theme_20261111/`;
legacy results are isolated within `outputs/NN/legacy/`. The `.v9` and `.orig`
backups are archived too. Use `script_migration.csv` to find the replacements.
Compatibility launchers execute the Git-visible copies in
`analyses/compat_impl/`; they do not depend on ignored local archive files.

## Implementation

- `common/`: native loading, M/U/K/history, timing, fit QA, inference and outputs.
- `analyses/`: behavior/control/readouts, neural analyses, task state and utility.
- `orchestration/run_themes.py`: bounded CPU campaign, phase ordering, progress and resume.
- `preprocessing/`: atlas and RAM preprocessing entry points.
- `voc_diagnostics/`: existing training and intervention tools retained for
  campaign compatibility; current utility analysis is exposed through 10.
- `continuous_planning/`, `continuous_planning_v2/`: preserved earlier post-hoc
  protocols. They are not treated as results of this 10M campaign.

## Run

Use the dynamic analysis environment (Python 3.10, numpy, pandas, scipy,
scikit-learn, matplotlib, PyYAML; numba additionally for HSMM).

```bash
python research_script/orchestration/run_themes.py --workers 3
python research_script/02_action_boundary_mechanisms.py --subject 1 --game 1
python research_script/07_neural_rsa.py --aggregate
python research_script/10_computation_utility.py
```

`--cache-root` selects the read-only input, and `--outputs-root` selects the
parent containing numbered directories. Each theme stores `subXXX_gameY/`
tables, figures, `qa.json`, and a source/input manifest, followed by `group/`.
Campaign progress is under `outputs/00/campaign/`. Existing completed results
are reused only with matching inputs, code and output checksums; a different
completed analysis is not silently overwritten. Partial work is rerunnable.

The default campaign runs behavior/control/prediction for all 18 subject-game
units first, then neural analyses, then HSMM. It launches no RL training or
game-score evaluation. It is local CPU execution; any later PBS submission
must explicitly use the hosts allowed in the repository's `AGENTS.md`.

## Scientific contracts

- Each human session contributes only its leave-session checkpoint to primary
  associations. The four replays are not four independent human observations.
- Held-out readouts are fit separately within each checkpoint: sessions other
  than that checkpoint's omitted session train the readout; the omitted session
  is its only test set. Imputation/scaling/PCA use training data only. Fixed
  regularization is declared in code, never selected from final results.
- `M` is initial model-policy log probability of the incumbent minus its
  strongest competitor. `U_sum` sums valid absolute sampled-branch Q updates;
  `U_all` divides by all native computations, including invalid-update events
  as zero. `U_valid` divides by valid-update count. At B=0, total/all-event
  evidence is zero, the intensive valid-update mean is undefined, and `searched`
  is explicit. `K = 4 U_all sigmoid(M) sigmoid(-M)`; K is not learned VOC.
- Native event order, validity masks and variable budgets are retained. No
  `concat29`, padding interpreted as evidence, invented pre-search REAL head,
  or SEARCH-to-REAL within-head revision is used.
- Gate analyses use pre-choice values. A legal CONTINUE that reaches the cap
  remains a CONTINUE; its absent next choice is censored. Forced STOPs are not
  treated as voluntary STOPs. Completed evidence may predict the next choice,
  not an earlier choice.
- Bidirectional NOOP/active risk sets depend on the preceding state. An active
  action changing to a different active action remains a non-termination in
  the active-to-NOOP analysis; it is not excluded using its future destination.
- Association models use participant-specific session fixed effects and
  decision-clustered temporal covariance at lags 64/128/256 (128 primary).
  Complete separation, deficient rank, convergence failures and coefficients
  above 50 on standardized predictors are reported as unestimable, not as
  significant effects. Prediction scores average equally over runs within
  participants, then equally over participants. Inference uses six-subject
  exact synchronized sign flips and explicit families; missing tests do not
  silently shrink a family. Within-run matching is descriptive, with
  replacement, on incumbent, age, time/reward strata and visual distance.
- Neural regressors are recomputed from the dynamic checkpoint. Only
  independently cached BOLD is reused after matching the raw behavioral SHA.
  `time[current_frame]` from the original NPZ supplies real timestamps, rather
  than assuming that replay `decision_time` equals human time. The inherited
  assumption is behavior zero at fMRI volume 60, 480 volumes, TR 1 s; scanner
  triggers are not available. HRF convolution uses 50-Hz causal holds for
  persistent states and decision-locked impulses for U/K/budget;
  individual imagined events do not receive invented timestamps. Current
  BOLD caches cover Pong/Space Invaders; Enduro neural inputs are absent.
- RSA uses a minimum eight-TR pair separation, with visual, preceding action,
  behavior/history, missingness and time nuisances; the second stage adds
  initial support and quantity. It is representational association. Neural
  decoding uses training-session PCA of all cached ROI voxels (up to 32
  components) and fixed ridge alpha 100. Encoding uses fixed ridge alpha 100
  and whole-session holdout. ROI-mean partial correlation is a separate
  localization table, not predictive encoding.
- Theme 09 uses factual RAM/visual descriptors; RAM entropy is not cognitive
  uncertainty, and action-conditioned `B B^T` is not causal controllability.
- Theme 10 verifies existing paired-pilot source hashes and reports its
  sub001/protocol/training-step coverage. It cannot turn human replay into
  causal reward utility or generalize that pilot to all 72 models.
- Theme 12 distinguishes lagged evaluation from recorded incoming tree carry;
  neither implies actual reuse of an imagined action sequence. Matched-path
  forecast errors enter only once both their outcome timestamp and native
  availability event precede the current decision, never at forecast origin;
  availability resets across episodes. Theme 13 is a
  model-independent behavioral follow-up, not evidence of latent human planning
  states; its existing feature contract covers Pong and Space Invaders only.

Tests: `python -B -m unittest discover -s research_script/tests -p 'test_theme_contracts.py' -v`.

## COSYNE panels, split by theme and game

The panel implementation in `panels/` restores the scientific panels rather
than replacing them with regression-coefficient forest plots. The exact map
is `panels/panel_mapping.csv`: Figure 2A–C belong to 01; Figure 2D–E and
Figure 3C to 02; Figure 3A–B to 04. Figure 3D–E are neural and are excluded
from this non-neural rerun. The static framework diagram remains a reference.

```bash
python research_script/orchestration/run_panels.py --workers 3
python research_script/orchestration/run_panels.py --render-only
python research_script/02_action_boundary_mechanisms.py --panels --aggregate --game 1
python research_script/01_behavioral_analysis.py --subject 1 --game 1
```

The complete previous behavioral workflow remains available through
`01_behavioral_analysis.py --legacy`, with its original arguments and virtual
source location; its source is retained in `analyses/compat_impl/`.

Each exported panel contains **one game only**, with PNG, vector PDF, caption,
source tables and provenance under `outputs/NN/panels/game0/` (Enduro),
`game1/` (Pong), or `game2/` (Space Invaders). New subject analyses are in
`outputs/NN/panels/subXXX_gameY/`. The original validated regressions and
held-out readouts are preserved; their manifest hashes identify reused results.
New persistence, matching levels, M/U surfaces, K curves, search-index update
contrasts and same-head endpoint sensitivities are recomputed from the 10M
held-out replay. Campaign status is in `outputs/00/panel_campaign/`.

Same-head endpoint change uses native `policy_type=1` SEARCH probabilities at
root, never the final REAL head or the legacy `actor_initial_p` fallback at
B=0. Two distinct observed root choices are required. The original Gaussian
local regression and natural cubic K-curve estimators are preserved, with
explicit support checks. Unsupported ranges remain blank, with coverage QA;
no significance stars are copied from the old model. In variable-length
search heatmaps, both matched members must have the queried index: late
effects are conditional on search length, and missing computations are not
zero-filled. All six participants are required for group curves. The descriptive
M–U heatmap uses the original COSYNE rule of at least four supported participants.

### Individual and group M–U heatmaps

`python research_script/02_action_boundary_mechanisms.py --heatmaps` generates
18 individual and three group figures, with a gallery at
`outputs/02/heatmaps/index.html`. The normal theme02 panel renderer also builds
and uses these maps. Existing 40×40 estimates remain archived/available as the
previous analysis and are not silently relabeled as the new surface.

Each person pools decisions across their held-out runs, using the original
COSYNE Gaussian histogram-convolution estimator (180×180, bandwidth 4% of axis
spans, >=12 observations within two bandwidths). The group is an equal-person
average of supported probabilities, not a pooled-count estimate; at least 4/6
people are required, exactly as in the original Fig2E. This **differs** from the
equal-run aggregation in the matched tests and other theme02 estimates. Within
each game all panels share axes; each transition has one probability color
scale across requested games and subjects. The colorbar marks 97th-percentile
saturation. U remains the sum of valid absolute revisions divided by actual
computation count, with U=0 at B=0. Zero-mean-U runs retain zeros. A separate
support sheet reports both total local observations and observations with U>0:
kernel smoothing of the zero mass does not establish positive-update support.

Human-only Figure 2A–C uses **all original human observations**, with source
files discovered through the cache and verified by SHA256. It does not inherit
the replay sampling grid. Model-linked panels use verified original
`time[current_frame]` on the delivered replay grid (including repeated source
frames), with no additional downsampling. Task-state curves and transition-centered trajectories are
descriptive complements to the adjusted primary analyses. Existing sub001
paired utility is plotted by game and protocol; no new score evaluation is
launched. The separate existing campaign owns HSMM execution to avoid two
writers to the same output; its status is explicitly reported.

Panel tests: `python -B -m unittest discover -s research_script/tests -p 'test_cosyne_panels.py' -v`.

## Exploratory value-update candidates

`experiments/value_update_candidates.py --workers 3` compares 16 signed,
absolute, branch-specific update and Q-level definitions on the held-out 10M
cache, without changing the primary theme02 estimates. The analysis protocol
is saved before extraction under `outputs/02/value_update_candidates_20261111/`.
Existing context-matched pairs are reused; positive equal-budget and
valid-update-pair subsets are reported separately. Raw-SD and rank contrasts
are jointly corrected across the complete 864-test family with synchronized
subject sign flips. Missing six-subject coverage remains untestable. This is
exploration of already examined data, not independent confirmation.

`experiments/value_update_branch_coverage.py` and
`experiments/value_update_branch_choices.py` add descriptive support audits:
which root actions were queried/revised, whether multiple root branches were
queried, and concentration of the first queried branch. They introduce no
additional hypothesis tests. Sampled/terminal Q **levels** include first
evaluations, while repeated ΔQ metrics require observed prior visits; an
unvisited encoded zero is never an observed value estimate.
