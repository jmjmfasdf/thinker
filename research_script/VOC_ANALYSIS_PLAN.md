# VoC: expectation → computation → revision → stopping → behavior

## Scope and interpretation

Implement additive analyses in canonical 03, 05–10. Preserve existing analyses,
04 paired utility evaluation, 11–13 neural code, completed outputs and frozen
capture runtimes. No training, rollout, STOP/CONTINUE intervention or full cohort
job submission is part of this change. Planned cohort: sub001–004 × Pong,
Enduro and Space Invaders; CLI defaults remain unchanged.

Inputs are completed visualization captures, original behavior when needed,
and their hash-bound original checkpoint/config. No external intermediate
analysis is required. Internal numeric caches are reproducible from these inputs.
Canonical numbered scripts remain the entrypoints; render-only reads published
tables, never source captures or checkpoints.

The environmental clock t and imagination index k are different. Task V is not
meta-value. Q(C) − Q(S) is predicted net VoC; its cost must not be subtracted
again. The current learning target includes future environmental reward but
only remaining computation cost in the current planning episode. Gate agreement
is execution validation, internal revision is not realized reward improvement,
and human-action fit is not environmental utility. Dueling Q shares a baseline;
algebraic relationships are not independent empirical discoveries.

## Common numerical contract

Keys: subject, game, run, episode, decision; choice/transition rows also retain
native event and k. Tables distinguish native pre-choice Q/VoC and actual control
from common SEARCH task-value/VoC readouts and common REAL policy readouts.
The latter are frozen linear readouts, not policies from counterfactual STOPs.

- Choices: online/EMA Q(C), Q(S), VoC, soft gate probability, actual control,
  free/forced status, depth, past computation count, state controls.
- Transitions: only computations that actually occurred; signed/absolute task
  V changes, visited-to-visited root Q changes and coverage, policy JS, endpoint
  VoC, next actual free choice if one exists.
- Decisions: actual K and summed effective costs, initial/last-choice/endpoint
  VoC separately, A_V=sum|ΔV|, U_V=A_V/K, D_V=V_end−V_start, policy change,
  human log-probability change (outcome only), and termination reason.

Policy JS uses the existing helper's **square root of Jensen–Shannon divergence**
(natural logarithms), i.e. JS distance. Native and common-phase readouts remain
separate. 05 reports forced-minimum versus free-choice computations separately.

First and forced-minimum computations count toward revisions/cost. Forced choices
do not enter free-choice fits. A cap-reaching CONTINUE remains CONTINUE; there is
no invented STOP or next choice at the cap. Unsupported Q changes are missing,
with explicit coverage. K=0 revisions/cost are zero. Temporal joins never cross
an episode, missing decision, nonpositive clock increment, or >0.25-second gap.

## Analysis additions

| Script | Addition |
|---|---|
|03|Separate Q levels/contrast; online–EMA disagreement; gate identity and choice calibration; initial VoC vs budget/cost.|
|05|Pre-computation VoC vs subsequent signed/absolute V revision, supported Q revision and policy JS, conditional on observed CONTINUE.|
|06|Revision vs subsequent VoC and next actual voluntary choice; distinguish last-choice from endpoint VoC; deterministic trajectory panels and human-fit change.|
|07|Same held-session full vs no-VoC/no-revision/no-policy-change readouts, plus a pre-search/quantity reference. Never use human-fit change as a predictor.|
|08|Lagged and trailing-window VoC/revision summaries with raw-behavior boundary checks and current-action prediction. Optional forecast evaluation remains optional.|
|09|Use existing prospective 0.5s input/0.5s outcome windows; add VoC level/trend and within-planning change to magnitude/direction comparisons.|
|10|Fold-specific candidate-pattern differences in VoC–revision–next-choice relationships; retain existing leakage-safe initial/first-computation budget prediction.|

EMA is primary, online is a sensitivity diagnostic. Local fits control only
pre-choice quantities, never eventual K. Session fixed effects and decision-level
HAC lags 64/128/256 are retained. New effects are exploratory, with coefficients,
confidence intervals and support, not significance stars. Prediction transforms
fit training sessions only and comparisons use identical held-session rows.
Thinker itself trained on these sessions: this is readout holdout, not unseen-
subject/session validation of Thinker. No causal mediation or optimality claim.

New curves average runs within participant, then participants; all games remain
separate. Late-k curves report survivor support. Candidate IDs and ACE bases
remain participant/fold-specific. Figures use large labels/legends, no titles,
300-dpi PNG and vector PDF. Exact plotted quantities remain in CSV tables.

## Verification and execution

Test first/single/multiple computations, RESET, voluntary STOP, capped CONTINUE,
zero budget, unsupported Q, absent next choices, duplicate keys, gaps, invalid
hashes, nonfinite input, constant/rare outcomes, future leakage, cache invalidation,
table-only rendering and run/participant weighting. Check actual three-game
capture samples for frozen readout and legacy scalar reproduction. Preserve
01/02 regression tests. Full 12-combination execution is a separate step.

Example after implementation (each selected number independently prepares its
inputs; 04 is intentionally absent):

```bash
python -B research_script/03_computation_allocation.py --subjects 1 2 3 4 --games pong enduro spaceinvaders --resume
python -B research_script/05_evaluation_revision.py --subjects 1 2 3 4 --games pong enduro spaceinvaders --resume
# Likewise canonical 06, 07, 08, 09, 10; aggregate after participant runs.
```

Use the configured dynamic-thinker environment and cluster placement policy for
heavy production runs. No implicit scheduler submission is added.

Implementation verification (2026-11-26): the dedicated VoC tests, existing 01/02
regressions and explicit three-game capture checks pass. The checked 38 native
decisions contain 111 actual computations; maximum readout reproduction error
was 1.89e-6. Existing temporal/bout self-checks and 29 control-pattern contracts
also pass (two obsolete checkpoint fixtures were updated to include file hashes).
Actual Enduro voluntary-stop and cap trajectories were rendered in a temporary
review directory. These checks are not a full participant/game analysis run.

## Deferred neural extension (documentation only)

11 encoding: compare full versus minus-VoC and minus-revision feature groups,
holding visual/behavior/initial state/quantity controls fixed. 12 decoding:
conditional out-of-session prediction of each group. 13 RSA: distinguish shared
and conditional representational correspondence. Fit nuisance transforms inside
training folds; HRF uses physical behavior timestamps, not invented k timestamps.
Validate original BOLD provenance/alignment before use; unsupported games remain
explicitly unavailable. Existing TR=1s / volume-60 alignment is an assumption,
not scanner-trigger validation. No neural implementation is included here.
