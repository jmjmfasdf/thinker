# Enduro VoC-v10 powered 300k preregistered acceptance

This protocol is frozen after the single preregistered VoC-v9 fresh-200k run
failed and before any qualifying VoC-v10 training run.  The v9 result remains
a permanent failure: in the frozen windows it observed only 230 deep-negative
events in the full window (required 256) and 55 in the late window (required
64).  The v9 result may not be reclassified, rescued by extension, or pooled
with this experiment.

V10 is a separately named power and observation-support experiment.  It does
not lower any VoC-v7 event-count or behavioural acceptance threshold and does
not change the learning mechanism.  It gives the already-defined rare
behavioural slices more prospective observation time.  Because the
populations are wider, v10 must not be described as a verbatim rerun or
retrospective pass of the v7/v9 200k protocol.

The wider populations do change the event-density implied by the unchanged
absolute count floors.  Full grows from 100k to 200k real steps and late from
25k to 50k, so the nominal deep-negative density floor changes from 2.56 to
1.28 events per 1k steps in both populations.  The useful next-positive and
next-negative floor changes from 1.28 to 0.64 events per 1k steps.  The
unchanged counts remain prospective precision/support floors, not event-rate
requirements.  Meeting them in v10 cannot revise the v9 failure.

The pre-run planning estimate of approximately 99% is only a
bootstrap/negative-binomial heuristic for clearing the widened deep-negative
support count, conditional on stationarity and exchangeability of future
support occurrence.  It is not the probability of overall acceptance,
correct stability direction or strength, generalization to another seed, or
fixed-checkpoint success.  Those outcomes retain their independent gates
below.

## Immutable experiment identity

- The one primary run is a fresh Enduro run with base seed exactly `2`, the
  next unused seed selected before this protocol was frozen.  No prior seed-2
  training, evaluation, reward, support, or behavioural result was inspected
  when selecting it.  There is exactly one primary attempt: no replacement
  seed, retry, fallback, restart-selection, or repeat-selection.  A crash,
  corrupt artifact, failed integrity check, or failed acceptance criterion
  makes the primary v10 experiment fail.
- Use `total_steps=300000` and keep
  `schedule_total_steps=100000000`, production model warm-up 10000, actor
  unroll 201, W&B enabled, and all other v9 production settings unchanged.
- Keep the complete v9 mechanism unchanged, including
  `dynamic_voc_mode=control`, `voc_gate_confidence_weighted=false`,
  `voc_gate_adam_beta1=0`, `voc_gate_learning_rate=0.001`,
  `voc_gate_target_tau=0.1`, `voc_gate_q_temperature=0.05`,
  `voc_gate_temperature=1.0`, `voc_train_epsilon=0.02`, stochastic gate
  sampling, Adam gate optimization, `voc_gate_param_align=true`, and
  `voc_gate_param_align_coef=1.0`.
- Keep gate-policy schema 3 and the v9 batch-start, detached FP32 parameter
  target and unnormalized parameter energy exactly unchanged.  Runtime action
  selection remains the learned scalar stochastic gate with epsilon mixing;
  Q, depth, control token, Q sign, and acceptance labels may not hard-code or
  override an action.
- The run must be fresh: `ckp=false`, empty actor and model preloads, empty VoC
  parent checkpoint, and zero-initialized online-Q, EMA-Q, and gate heads.
  Promotion, resume, or continuation from the failed v9 run is forbidden.
- Preserve the immutable v9 snapshot byte-for-byte.  Create a separately named
  immutable v10 snapshot as that exact v9 base plus an enumerated overlay
  containing only this protocol, the hardened fixed-evaluator/profile work
  needed for `v10-300k`, and its corresponding tests.  The behaviour-bearing
  learner, model, feature code, configuration, optimizer, loss, `cenv` and
  environment code, telemetry, and staged Enduro data must be byte-identical
  to the v9 base.  Freeze the base identity, exact overlay diff, complete v10
  tree, tests, and data in checksum manifests before any v10 execution.
- Seeds `3` and `4` are not backups.  They may be used only by a future,
  separately frozen generalization experiment.  Their outcomes cannot replace,
  pool with, majority-vote, or otherwise rescue the primary seed-2 result.

## Nonqualifying integrity wire

Before the primary run, execute one 1.2k integrity wire with the frozen v10
snapshot, seed `1` as a sentinel, `total_steps=1200`, the established v9
short-run settings (`model_warm_up_n=512`, `actor_unroll_len=41`, W&B
disabled), and otherwise the unchanged mechanism and scheduler.  Its scope is
only launch/exit, manifest, protocol/schema-3 flag, provenance, checkpoint,
CSV-header/width, finiteness, counter-wiring, and process-cleanup integrity.
Do not inspect or use its reward, support, conditional behaviour, stability,
or alignment-convergence values.

The seed-1 wire is nonqualifying and cannot initialize, replace, select,
modify, or contribute observations to the seed-2 experiment.  It cannot
justify a mechanism/configuration change.  A failed integrity wire blocks the
primary under this frozen protocol; a successful wire authorizes only the one
fresh seed-2 primary attempt described above.

## Canonical rows and windows

The gating training populations are fixed before the run:

- Full window: `100000 < real_step <= 300000`.
- Late window: `250000 < real_step <= 300000`.
- Stability W1: `270000 < real_step <= 280000`.
- Stability W2: `280000 < real_step <= 290000`.
- Stability W3: `290000 < real_step <= 300000`.

Rows with `real_step > 300000` are overshoot and are excluded from every
training-telemetry calculation.  Do not interpolate, truncate a row, replace a
boundary with a nearest row, or use a terminal checkpoint value as a missing
CSV row.  A canonical row is a complete CSV row in file order with a finite
integer `real_step` and all fields required by the calculation finite.
`real_step` must be unique and strictly increasing across the file.  A
malformed row, duplicate/nonmonotone step, absent required column, or non-finite
required cell fails closed; it is not silently dropped or reordered.

For historical continuity, calculate and report the old v9 100k diagnostic
aggregate `(70000,100000]` and its three windows `(70000,80000]`,
`(80000,90000]`, and `(90000,100000]`.  These rows and every metric derived
from them are non-gating diagnostics.  They cannot pass, fail, replace, or
modify any v10 criterion.

## Frozen metric algebra

- The authoritative label is all-valid EMA
  `delta_q = Q_continue_ema - Q_stop_ema`.  Positive means strictly
  `delta_q > 1e-6`, negative means strictly `delta_q < -1e-6`, and all other
  finite values are ties.  Ties belong to neither sign.
- Deep means pre-decision depth at least 8.  WAIT, forced-only, and invalid
  control rows are excluded unless a criterion explicitly audits them.
- Pool event counts and sufficient statistics; never take an unweighted mean
  of row-level rates or means.  For metric `x_i` with event support `n_i`,
  use `sum(n_i*x_i)/sum(n_i)`.  A pooled conditional with zero total support
  is undefined and makes every criterion requiring it fail.  A finite zero
  emitted by a helper for zero support is not evidence of a defined value.
- Conditional teacher and student means use the direct
  `actor/voc_gate_acceptance_*` all-valid EMA sufficient statistics, weighted
  by their matching positive or negative counts.  Historical train-only BCE
  teacher/confidence fields are not substitutes.  Teacher and student gaps
  are their pooled positive conditional minus pooled negative conditional.
- Signed margins use the corresponding non-tie count.  Online-versus-EMA sign
  agreement uses its exposed non-tie agreement count.  Held-out EMA
  selected-action RMSE is
  `sqrt(sum(row_count*row_rmse^2)/sum(row_count))`.
- A pooled event success rate is the sum of integer successes divided by the
  sum of integer eligible events.  If CSV exposes only an integer support
  `n_i` and its float32 rate `r_i`, reconstruct the row success count as the
  unique nearest integer `k_i = round(n_i*r_i)`.  Require
  `abs(n_i*r_i-k_i) <= 1e-4`; otherwise fail closed.  Sum `k_i` and `n_i`
  before division.  This rule prevents float32 roundoff from turning an exact
  0.5 event tie into a directional result.
- Train/holdout support fractions pool their CONTINUE and STOP counts.
  Wrong-side saturation pools wrong-side counts over their corresponding
  positive/negative teacher counts.  Forced-stop rate pools forced stage ends
  over all stage ends.  Every required denominator must be positive.
- Strict useful-compute coverage is
  `sum(prior_useful_count)/sum(prior_useful_candidate_count)` on the combined
  compute slice.  PROCEED and RESET support use their respective pooled
  `prior_useful_count`; next-sign conditionals use the combined compute
  slice's matching sign counts; PROCEED/RESET margins use each slice's
  non-tie count.  All of these denominators must be positive.
- Counts must be finite, nonnegative integers and must reconcile: positive +
  negative + tie equals slice count, non-tie equals positive + negative,
  PROCEED + RESET eligible pairs equals combined eligible pairs, and the
  depth-bin partition equals the all-valid acceptance count.  Failure to
  reconcile fails closed.

### Trailing-five definition

Order canonical full-window rows exactly as written in the CSV.  Every row at
index 5 or later (counting the first full-window row as index 1) defines one
trailing-five endpoint using itself and its four immediately preceding
canonical full-window rows.  Pool positive and negative support and student
continue-probability numerators separately over those five rows, then subtract
the negative conditional from the positive conditional.

Both pooled sign supports must be positive at every endpoint.  A zero-sign
denominator fails the run; it is not converted to finite zero, omitted, or
allowed to break/compress a streak.  A gap is negative only when it is
strictly below zero; exact zero is non-negative.  Consecutive means adjacent
eligible endpoints in canonical CSV order.  Report every negative endpoint
and the maximum consecutive negative run.

## Mechanism acceptance

Both the full and late windows must independently satisfy every item:

- teacher conditional gap at least `0.075`;
- student conditional gap at least `0.05`;
- student-gap / teacher-gap retention at least `0.50`, with a positive finite
  teacher-gap denominator;
- pooled signed margin strictly greater than zero;
- online-versus-EMA non-tie sign agreement at least `0.60`;
- train CONTINUE, train STOP, holdout CONTINUE, and holdout STOP fractions each
  strictly greater than `0.05`;
- wrong-continue and wrong-stop saturation rates each strictly below `0.01`;
- held-out EMA selected-action TD RMSE at most `0.5`;
- actor, online-Q, and gate AMP skips, consecutive-skip terminal events, and
  non-finite events all exactly zero.

At least two of W1, W2, and W3 must each have both a strictly positive student
gap and a strictly positive signed margin.  The full-window maximum
consecutive negative trailing-five run must be at most 3.

## Four required learned behaviours

Every numerical threshold below is unchanged from the v7 200k protocol.  The
absolute support floors remain 256/64 and 128/32; they are not divided,
duration-scaled, estimated, or replaced by expected counts.  Each behaviour
must pass independently in both the full and late windows.

### B1. Easy negative-Q states stop

- Positive and negative Q signs each have at least `0.05` of non-tie support.
- `E[p_continue | delta_q < -1e-6] <= 0.475`.
- Sampled `STOP`, conditional on `delta_q < -1e-6`, has rate at least
  `0.525`.
- Conditional argmax STOP accuracy is at least `0.60`.

### B2. Hard positive-Q states continue searching

- `E[p_continue | delta_q > 1e-6] >= 0.525`.
- Sampled `CONTINUE`, conditional on `delta_q > 1e-6`, has rate at least
  `0.525`.
- Conditional argmax CONTINUE accuracy is at least `0.60`.

### B3. Deep negative-Q states stop

For `depth >= 8 and delta_q < -1e-6`:

- support is at least 256 in the full window and at least 64 in the late
  window;
- mean continue probability is at most `0.475`;
- sampled STOP rate is at least `0.525`;
- conditional argmax STOP accuracy is at least `0.60`.

Forced stops never count as sampled or argmax successes.  Pooled forced-stop
rate must be strictly below `0.01` in both the full and late windows.

### B4. Useful computation is accepted and re-evaluated

An eligible transition is an immediate adjacent pair in the same environment
stream for which both decisions are valid, prior EMA `delta_q > 1e-6`, the
prior sampled control is PROCEED or RESET, current
`predecision_last_control` equals that prior control, and current pre-decision
depth equals prior depth plus one.

- observable eligible-pair coverage is at least `0.95`;
- PROCEED and RESET support are each at least 256 full and 64 late;
- next-positive and next-negative support are each at least 128 full and 32
  late;
- next mean `p_continue`, conditional on a positive sign, is at least
  `0.525`, and, conditional on a negative sign, is at most `0.475`;
- next sampled sign-correct rate is at least `0.525` for each sign;
- next conditional argmax sign-correct rate is at least `0.60` for each sign;
- pooled PROCEED and RESET transition-slice signed margins are each strictly
  positive.

## Stability behaviour semantics

Every stability window must have positive defining support for every listed
conditional and must retain all four behaviours in the strict correct
direction:

- B1: negative mean `p_continue < 0.5`, sampled STOP rate `> 0.5`, and argmax
  STOP rate `> 0.5`;
- B2: positive mean `p_continue > 0.5`, sampled CONTINUE rate `> 0.5`, and
  argmax CONTINUE rate `> 0.5`;
- B3: at least one deep-negative event, deep-negative mean
  `p_continue < 0.5`, sampled STOP rate `> 0.5`, and argmax STOP rate `> 0.5`;
- B4: at least one next-positive and next-negative event, positive mean
  `p_continue > 0.5`, negative mean `p_continue < 0.5`, both sampled and both
  argmax sign-correct rates `> 0.5`, and PROCEED and RESET each have positive
  non-tie support and strictly positive pooled signed margin.

Every equality at 0.5 is neutral and fails strict direction, including a
sampled directional rate reconstructed as exactly one half from integer event
counts.

A stability window meets the `.525` strength conditions only when all of the
following hold in that same window:

- B1 negative mean `p_continue <= 0.475` and sampled STOP `>= 0.525`;
- B2 positive mean `p_continue >= 0.525` and sampled CONTINUE `>= 0.525`;
- B3 deep-negative mean `p_continue <= 0.475` and sampled STOP `>= 0.525`;
- B4 next-positive mean `p_continue >= 0.525`, next-negative mean
  `p_continue <= 0.475`, and both next sampled sign-correct rates
  `>= 0.525`.

All three windows must pass strict direction.  At least two of the three must
meet every `.525` strength condition above.  The full/late support, coverage,
argmax-0.60, calibration, saturation, and safety gates remain independently
required in their stated populations; they are not silently added to or
substituted for the stability-strength definition.

## Alignment diagnostics

Report the v9 alignment BCE, parameter energy, scaled and total gate
objective, target/gate/error norms, bias error, relative-error and cosine
values with their defined counts, raw/post-clip gradient norms, clipping
count, learning rate, and gate/Q/EMA update and skip counters over the
historical, full, late, and stability populations.  Every applicable value
must be finite and schema/config values must remain exactly enabled with
coefficient 1.0.  These metrics are diagnostic: no post-hoc numerical
alignment threshold or slice-conditioned loss is permitted.

## Decision and fixed-checkpoint confirmation

The primary training run passes only if artifact/provenance integrity,
mechanism acceptance, all four behaviours in full and late, both stability
requirements, trailing-five, calibration, forced-stop safety, AMP safety, and
non-finite safety all pass.  There is no partial, mechanism-only, support-only,
or sensitivity pass.  Data above 300000, optional seeds, historical 70--100k
diagnostics, and a longer continuation cannot change the decision.

Only after the primary seed-2 training telemetry passes may its terminal
checkpoint be evaluated by the hardened fixed-checkpoint evaluator under the
exact profile name `v10-300k`.  That profile must be implemented, tested, and
included in the pre-run source manifest.  It uses training disabled, epsilon
zero, the learned stochastic gate, seeds `20260827..20260842`, exactly 16
streams by 6250 real steps per stream (100000 total), and calibration V-trace
unroll 201.  Diagnostic or shortened evaluations are ineligible.  Apply the
same four behavioural definitions, do not count forced stops as successes,
and report selected-action heldout calibration.  Evaluator integrity or
behaviour failure makes fixed confirmation fail; another evaluator run,
checkpoint, seed set, or profile cannot replace it.

The Enduro v10 claim requires both the primary training-telemetry pass and its
eligible `v10-300k` fixed-checkpoint pass.  Pong and Space Invaders remain out
of scope and may not start before both Enduro stages pass.  A failure at either
stage is permanent for this preregistered experiment and cannot be rescued by
changing a population, denominator convention, threshold, horizon, seed,
checkpoint, or evaluator interpretation.
