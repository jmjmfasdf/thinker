# Enduro VoC-v11 exact-projection preregistered acceptance

This protocol is frozen after the single VoC-v10 seed-2 300k result failed,
after the v11 implementation and contract audit, and before any v11 wiring,
qualification, primary training, or fixed-checkpoint evaluation.  V11 is a
separately named experiment.  It changes exactly one behavior-bearing
mechanism relative to v10: the learned Adam update of the scalar gate is
replaced by the deterministic post-EMA projection specified below.  No result
from a v11 stage may be used to change that mechanism, a seed, a population, a
threshold, or an aggregation rule.

The immutable v10 preregistration and result remain permanent.  In particular,
v10 failed the frozen acceptance rather than producing a partial pass: W1
failed the B4 next-positive sampled-strength threshold (`736/1423 =
0.517217... < 0.525`); W3 failed strict direction in the deep-negative and
useful-next-negative slices; full B3 failed probability, sampled, and argmax
requirements; full B4 next-negative sampled correctness was
`15320/29188 = 0.524873... < 0.525`; and late B3 failed.  V11 must not
reclassify, extend, pool with, or rescue v10.  The v10 snapshot, artifacts,
telemetry, and the following two normative source documents are preserved
byte-for-byte:

- `VOC_V9_PARAM_ALIGN_ACCEPTANCE.md`, SHA-256
  `f832ae998332c5ff0ce03cb9334447a0e456646f80ccfdb044271cfb19681972`;
- `VOC_V10_POWERED_300K_ACCEPTANCE.md`, SHA-256
  `9cf52993ce4dcf1044b867028db9ae4a7d91b4e69516f1823e6abefd3059b6e4`.

Whenever this document incorporates a frozen v9 or v10 criterion, the exact
hashed text controls.  A summary below cannot weaken, strengthen, round, or
reinterpret it.

## The one mechanism change

- Keep the v10 production configuration except for the atomic replacement
  `voc_gate_param_align=true` -> `voc_gate_param_align=false` and
  `voc_gate_exact_projection=false` -> `voc_gate_exact_projection=true`.
  This pair names one replacement update rule, not two tunable interventions.
  Keep `voc_gate_param_align_coef=1.0` as exact protocol metadata.
- Keep `dynamic_voc_mode=control`, fresh control from step zero,
  `voc_gate_confidence_weighted=false`, `voc_gate_adam_beta1=0`,
  `voc_gate_learning_rate=0.001`, `voc_gate_target_tau=0.1`,
  `voc_gate_q_temperature=0.05`, `voc_gate_temperature=1.0`,
  `voc_train_epsilon=0.02`, stochastic gate sampling, and the
  `100000000`-step scheduler horizon.  The actor, online Q loss, EMA rule,
  features, environment, reward, support definitions, and action-selection
  distribution are otherwise unchanged.
- Use gate-policy schema 4.  Schema 4 requires explicit
  `voc_gate_exact_projection=true`, explicit
  `voc_gate_param_align=false`, and exact coefficient `1.0` in configuration,
  actor metadata, ModelNet metadata, resume/smoke/evaluation surfaces, and the
  resolved validation record.  Schema 1--3 retain their historical
  `voc_gate_exact_projection=false` interpretation and cannot silently become
  v11.
- After a successful online-Q optimizer step, first perform the existing FP32
  EMA Polyak update.  Then, under `no_grad` and before actor publication or a
  checkpoint, copy
  `W_gate=(T_policy/T_Q)*(W_ema,C-W_ema,S)` and
  `b_gate=(T_policy/T_Q)*(b_ema,C-b_ema,S)` into the FP32 scalar gate.  The
  stored gate weight and bias must be `torch.equal` to that raw EMA affine
  target.
- If the online-Q step is skipped, non-finite, or absent because there is no
  training support, the EMA and gate must both remain unchanged.  An actor AMP
  skip is independent: a successful Q step still advances EMA and projection.
- The gate head is frozen and receives no backward pass.  Its Adam step and LR
  scheduler step are bypassed, its optimizer state remains empty, and its
  scheduler and GradScaler remain pristine.  `voc_gate_update_count` means
  successful projection count in schema 4 and must equal the successful
  online-Q and EMA update counts; the parent EMA count is zero.
- Bit equality applies only to stored gate W/b versus the raw EMA head-
  difference parameter target.  It does not apply to per-state reconstructed
  dueling Q, probability, or delta values: common-value and centering FP32
  cancellation can differ by operation order.  Those residuals are finite
  diagnostics with a frozen implementation tolerance, not an additional
  behavioral gate.  The existing `1e-6` Q-sign tie tolerance is unchanged.
- Runtime control remains the scalar stochastic gate with epsilon mixing.  Q,
  depth, control token, Q sign, acceptance slice, and stored behavior
  probability may not override an action or enter a new loss.

The only new behavior telemetry is the detached continue probability recovered
from the behavior logits stored with each rollout.  Report its positive and
negative conditional means for the all-valid, depth-8-plus, and strict useful-
compute slices under the same temporal alignment as the existing target-gate
telemetry.  These fields are instrumentation only: they have no barrier,
threshold, gradient, loss, optimizer, selection, or acceptance role.

## Immutable identities and no-retry rule

Every v11 run is fresh: `ckp=false`, `preload=''`, `preload_actor=''`,
`voc_parent_checkpoint=''`, zero parent update count, and exactly zero online-Q,
EMA-Q, and scalar-gate heads before training.  Wire, qualification, and primary
runs are separate processes and separate output directories; no checkpoint or
observation passes between them.

- The 1.2k integrity wire uses base seed exactly 1.  It is nonqualifying.
- The fresh 100k qualification also uses base seed exactly 1.  There is exactly
  one qualification attempt.
- The one primary 300k run uses base seed exactly 3, prospectively selected
  before v11 execution.  There is exactly one primary attempt.  Seed 1 is not a
  backup for seed 3, and no other seed may replace, pool with, majority-vote,
  or rescue either frozen stage.

A crash, corruption, provenance failure, qualification failure, primary
training failure, or fixed-evaluator failure is the failure of that stage.  No
same-run extension, resume, replacement seed, retry-selection, threshold edit,
or result-conditioned mechanism edit is permitted.

Create v11 from the immutable v10 snapshot as an independently copied,
separately named snapshot plus an exact enumerated overlay containing only the
v11 mechanism/schema/public validation, its tests, fixed evaluator/profile,
this document, and the report-only stored-behavior telemetry.  The v10 result,
all unrelated behavior-bearing files, `cenv` binaries, selected Enduro data,
and historical documents must remain byte-identical.  Freeze source and data
manifests outside `src`, seal modes, exclude historical run outputs, prohibit
links/special/writable source nodes, and prove inode independence before any
launch.

## Sequential release gates

The stages are strictly ordered:

1. Freeze and independently audit the v11 snapshot, exact v10-to-v11 overlay,
   source/data manifests, runtime binding, binaries, tests, and this document.
2. Run one seed-1 1.2k integrity wire with `total_steps=1200`,
   `model_warm_up_n=512`, `actor_unroll_len=41`, W&B disabled, and otherwise
   the production mechanism and `schedule_total_steps=100000000`.
3. Only after the wire passes integrity, run one separate fresh seed-1 100k
   qualification with `total_steps=100000`, production model warm-up 10000,
   actor unroll 201, W&B enabled, and the qualification rules below.
4. A failed 100k qualification permanently ends v11 and forbids the 300k run.
   A pass authorizes exactly one separate fresh seed-3 primary with
   `total_steps=300000`, model warm-up 10000, actor unroll 201, W&B enabled,
   and otherwise identical production configuration.
5. Only if the seed-3 training telemetry and artifact gates pass may its one
   terminal checkpoint be evaluated with exact profile `v11-300k`.

Pong and Space Invaders remain out of scope until every Enduro v11 stage
passes.

## Integrity-wire acceptance

The wire may be inspected only for launch/exit, snapshot and data binding,
schema/config/provenance, exact projection, first/later update mechanics,
checkpoint completeness, CSV header/width, finiteness, counters, W&B-disabled
identity, and process/GPU/Ray cleanup.  Its reward, support, conditional
behavior, stability, and acceptance values are forbidden inputs to any
decision or edit.

Before any supported update, online Q, EMA Q, and gate W/b are exact zero; the
batch-start teacher and student probabilities are 0.5, confidence is zero,
and unweighted BCE is `ln(2)`.  On a successful first Q step the Q and EMA
advance, the post-EMA affine target is projected exactly (and may become
nonzero), and Q/EMA/projection counters each advance once while gate optimizer,
scheduler, and scaler state remain pristine.  A later supported non-tie must
exercise a finite nonzero EMA target and finite pre-projection error followed
by exact zero parameter residual.  The Q-skip/no-support and actor-skip
branches are exercised by the frozen unit/contract tests; the wire does not
require any of those branches to occur.  If a no-support branch occurs, its
state-transition semantics must still agree, but any actor or Q AMP skip makes
the qualifying wire fail because the required AMP-skip counts are zero.

Any wire failure blocks qualification.  Wire success authorizes only the one
100k qualification and supplies no qualifying observations.

## Frozen 100k qualification

Canonical rows, population, aggregation, malformed-input rules, Q-sign tie
definition, sufficient-statistic pooling, conditional-gap sources, trailing-
five construction, and heldout RMSE algebra are incorporated verbatim from
the section **Frozen 100k population and aggregation** of the hashed v9
document.  Thus the only gating rows are `(70000,100000]`, with fixed windows
`(70000,80000]`, `(80000,90000]`, and `(90000,100000]`; overshoot above 100000
is excluded.

The 100k qualification passes only if every v9 numeric mechanism/safety gate
below holds under that exact algebra:

- teacher gap at least `0.075`, student gap at least `0.05`, retention at
  least `0.50`, and signed margin strictly positive;
- at least two of three windows have both student gap and signed margin
  strictly positive;
- maximum consecutive negative trailing-five pooled gaps at most 3, with zero
  non-negative;
- train and holdout CONTINUE and STOP fractions each strictly greater than
  `0.05`, with positive denominators;
- wrong-continue saturation strictly below `0.01`; wrong-stop remains the v9
  report-only 100k safety diagnostic and is not promoted to a new threshold;
- online-versus-EMA non-tie sign agreement at least `0.60`;
- held-out EMA selected-action TD RMSE at most `0.5`;
- actor, online-Q, and gate AMP skips and every non-finite counter exactly
  zero.

In addition, schema 4 and exact projection are hard qualification gates:
every successful Q/EMA/projection count must reconcile; every applicable
stored gate W/b must equal its raw EMA affine target bit-for-bit; Q skip and
no-support must not update EMA or gate; and gate optimizer state must be empty
with pristine scheduler/scaler state.  All projection and stored-behavior
telemetry must be present, finite, and internally consistent, but no numerical
projection-convergence or behavior-probability threshold may be added.

The v9 alignment-objective diagnostics are not a v11 objective and are not
silently retained as gates.  This paragraph is the only mechanism-specific
substitution; every frozen v9 numeric population, threshold, and safety
interpretation listed above is unchanged.

## Frozen 300k primary acceptance

The hashed v10 sections **Canonical rows and windows**, **Frozen metric
algebra**, **Trailing-five definition**, **Mechanism acceptance**, **Four
required learned behaviours**, and **Stability behaviour semantics** are
incorporated verbatim.  V11 changes neither their labels nor any numeric
criterion.  In particular:

- full is `100000 < real_step <= 300000`;
- late is `250000 < real_step <= 300000`;
- W1/W2/W3 are `(270000,280000]`, `(280000,290000]`, and
  `(290000,300000]`;
- support floors remain deep-negative 256 full/64 late; PROCEED and RESET 256
  full/64 late each; next-positive and next-negative 128 full/32 late each;
- probability thresholds remain `0.475/0.525`, conditional argmax threshold
  `0.60`, useful-pair coverage `0.95`, sign agreement `0.60`, support fractions
  strictly above `0.05`, both wrong-side saturation rates strictly below
  `0.01`, forced-stop rate strictly below `0.01`, and heldout RMSE at most
  `0.5`;
- both full and late must pass mechanism acceptance and all four behaviors;
  all three stability windows must retain every strict direction, at least two
  must meet every `.525` strength condition, and equality at 0.5 fails strict
  direction;
- the full-window maximum consecutive negative trailing-five run remains at
  most 3, and every AMP/non-finite safety event remains exactly zero.

All event-count reconstruction, denominator, support, pairing, depth,
forced-stop, overshoot, malformed-row, and fail-closed rules are exactly those
of v10.  The old v9 `(70000,100000]` aggregate remains a report-only historical
diagnostic during the 300k run.  Stored behavior-probability slices and
projection residuals are also report-only and cannot create a pass, failure,
barrier, or post-hoc threshold.

Artifact/provenance integrity, mechanism acceptance, all four full/late
behaviors, stability direction and strength, trailing-five, calibration,
support, saturation, forced-stop safety, AMP safety, and non-finite safety must
all pass.  There is no partial, mechanism-only, support-only, or diagnostic
pass.

## Fixed-checkpoint confirmation and final claim

Only after the seed-3 300k training decision passes may the exact terminal
checkpoint be evaluated once with closed profile `v11-300k`.  The evaluator
uses training disabled, epsilon zero, the learned stochastic gate, seeds
`20260827..20260842`, exactly 16 streams by 6250 real steps (100000 total), and
calibration V-trace unroll 201.  Diagnostic or shortened evaluations are
confirmation-ineligible.

The profile must fail closed unless configuration, actor, and ModelNet all
bind `base_seed=3`, `total_steps=300000`,
`schedule_total_steps=100000000`, control mode, schema 4,
`voc_gate_param_align=false`, exact coefficient `1.0`,
`voc_gate_exact_projection=true`, dedicated soft-Q gate enabled, and fresh
empty preload/parent identity.  It must validate the terminal FP32 stored gate
W/b against the raw scaled EMA affine target with bit equality.  Per-state
dueling-Q probability residuals remain diagnostic/tolerance checks only.

Apply the same four behavioral definitions and support semantics incorporated
from v10, never count forced stops as sampled or argmax successes, and report
selected-action heldout calibration.  As in v10, fixed-checkpoint calibration
is required reporting and finite audit, not a newly invented post-hoc RMSE
threshold unless the hashed v10 fixed-confirmation text explicitly makes it
one.

The Enduro v11 claim requires the immutable artifact/integrity pass, the one
seed-1 100k qualification pass, the one seed-3 300k training-telemetry pass,
and its eligible `v11-300k` fixed-checkpoint pass.  Failure at any stage is
permanent for this preregistered experiment.
