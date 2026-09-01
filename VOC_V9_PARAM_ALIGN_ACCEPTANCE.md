# Enduro VoC-v9 parameter-alignment preregistration

This protocol is frozen after the v9 implementation and unit/contract audit,
but before the v9 1.2k wiring job or any training execution.  V9 changes one
learning mechanism relative to the accepted v7 100k configuration: it adds the
gate-only parameter-alignment objective described below.  Results must not be
used to change its coefficient, seeds, windows, thresholds, or aggregation.  Pong and
Space Invaders remain out of scope until every required Enduro stage passes.

## Frozen mechanism and configuration

- Keep the v7 control configuration, including fresh control from step zero,
  `voc_gate_confidence_weighted=false`, `voc_gate_adam_beta1=0`,
  `voc_gate_learning_rate=0.001`, `voc_gate_target_tau=0.1`,
  `voc_gate_q_temperature=0.05`, `voc_gate_temperature=1.0`,
  `voc_train_epsilon=0.02`, stochastic gate sampling, Adam gate optimizer,
  and a `100000000`-step scheduler horizon.
- Add exactly `voc_gate_param_align=true` and
  `voc_gate_param_align_coef=1.0`.  The coefficient is not tunable.
- Preserve the existing unweighted soft-Q BCE mean.  At learner update `t`,
  derive a detached FP32 target from the batch-start EMA Q head:
  `W*=(T_policy/T_Q)(W_ema,C-W_ema,S)` and
  `b*=(T_policy/T_Q)(b_ema,C-b_ema,S)`.  Add
  `0.5*(||W_gate-W*||_F^2+||b_gate-b*||_2^2)` to the gate loss.  Do not divide
  this Euclidean energy by the parameter count.
- Apply the alignment only when the existing gate training population has
  positive support.  It may update only the dedicated gate.  Online Q, actor,
  features, and EMA target remain detached from this term.  The EMA advances
  only after a successful online-Q step, as before.
- Runtime action selection remains the learned scalar stochastic gate with
  epsilon mixing.  Q must not directly override an action, and no depth,
  control-token, Q-sign, or acceptance-slice label may enter the loss.
- Use gate-policy schema 3.  Schema 3 requires the two alignment fields
  explicitly and binds them exactly on resume, promotion, smoke, and
  evaluation paths.  Schema 1/2 checkpoints retain their historical
  `false/1.0` interpretation and cannot silently become v9.

All qualifying runs are fresh: `ckp=false`, empty actor/model preload, empty
VoC parent, seed 1, and zero-initialized online-Q, EMA-Q, and gate heads.  The
same immutable source and staged Enduro data snapshot must be used throughout.

## Release sequence

1. Freeze source, tests, this document, runtime extension, and selected Enduro
   data in a checksum manifest.  Run the full repository test suite and strict
   manifest/mode audit.
2. Run a fresh 1.2k wiring job with the production mechanism but the previously
   established short-run settings (`model_warm_up_n=512`,
   `actor_unroll_len=41`, W&B disabled).  It must exit cleanly and pass all
   checkpoint/provenance validators.
3. Run one fresh Enduro 100k confirmation with production warm-up 10k,
   actor unroll 201, W&B enabled, and the frozen 100k rules below.  Do not tune
   or repeat-select based on its result.
4. Only if 100k passes, run a separate fresh Enduro 200k job from step zero
   under the identical immutable source, seed, and mechanism.  Apply
   `VOC_V7_200K_ACCEPTANCE.md` verbatim.
5. Only if training telemetry passes the 200k specification, run its exact
   fixed-checkpoint confirmation.  Only an Enduro pass at every required stage
   permits a later, separately preregistered Pong or Space Invaders run.

## Wiring acceptance

- The first supported update must reproduce the exact neutral tie: online and
  EMA Q values equal zero, teacher and student probability equal `0.5`, true
  teacher confidence zero, unweighted BCE equal `ln(2)`, parameter-alignment
  loss/error zero, and directed/BCE/alignment gate gradients zero.  The gate
  parameters and Adam moments remain exactly zero while the valid gate, Q, and
  EMA counters advance once.
- A later non-tie update must exercise a nonzero EMA target and alignment
  error/loss, finite gate gradient, nonzero gate parameters and moments, and
  independent Q/gate/EMA counters.  No actor or Q gradient may be introduced
  by the alignment term.
- All new alignment metrics, optimizer/scaler/scheduler state, protocol fields,
  and checkpoint tensors must be present where applicable and finite.  A zero
  target norm yields relative-error-defined `0`; cosine-defined is `1` only
  when both target and gate parameter norms are positive.  Undefined
  diagnostics are finite zero rather than epsilon-derived ratios, and their
  authoritative support is the sum of the corresponding defined flags over
  applied updates.
- Source/data manifests, recorded hashes, runtime paths, checkpoint schemas,
  first/final artifacts, CSV width, and process cleanup must pass the existing
  strict audit.  AMP skips and non-finite events must be zero.

## Frozen 100k population and aggregation

- Canonical rows are complete CSV rows satisfying `lo < real_step <= hi`.
  The fixed windows are `(70000,80000]`, `(80000,90000]`, and
  `(90000,100000]`; aggregate over `(70000,100000]`.
- Exclude overshoot above 100000.  Do not interpolate or substitute a nearest
  row.  Required metric cells must be finite, `real_step` must be unique and
  strictly increasing in file order, and malformed, duplicate, or nonmonotone
  input fails rather than being dropped or reordered.
- Define EMA `delta_q=Q_continue-Q_stop`; positive and negative mean strictly
  above and below `1e-6`.  Ties are neither sign.
- Pool counts and sufficient statistics over their event support.  A mean or
  rate is `sum(row_support*row_metric)/sum(row_support)`.  Held-out RMSE is
  `sqrt(sum(row_count*row_rmse^2)/sum(row_count))`.
- Teacher and student conditional gaps use the EMA-Q sign population exposed
  by the `actor/voc_gate_acceptance_*` sufficient statistics.  For each sign,
  pool `teacher_continue_probability_delta_{positive,negative}` and
  `continue_probability_delta_{positive,negative}` with the corresponding
  `delta_q_{positive,negative}_count`; subtract the pooled negative conditional
  from the pooled positive conditional.  These direct all-valid conditionals
  are authoritative; the historical BCE teacher/confidence fields use the
  train-only population and therefore must not be substituted into this gap.
- The signed margin and online-versus-EMA sign agreement are pooled by their
  corresponding non-tie counts.  A trailing-five endpoint is eligible only
  when it and its previous four canonical rows all lie in `(70000,100000]`.
  Pool those five rows' positive and negative supports separately before
  taking the student conditional difference; a value below zero is negative
  and exactly zero is not.

The 100k run passes only if all of the following hold:

- aggregate teacher gap is at least `0.075`;
- aggregate student gap is at least `0.05`;
- aggregate student-gap/teacher-gap retention is at least `0.50`;
- aggregate signed margin is greater than zero;
- at least two of the three fixed windows have both student gap greater than
  zero and signed margin greater than zero;
- the maximum consecutive run of negative trailing-five pooled gaps is at most
  three (zero is not negative);
- train CONTINUE and STOP counts divided by their train total, and held-out
  CONTINUE and STOP counts divided by their held-out total, are each strictly
  greater than `5%`; every denominator must be positive;
- the historical 100k `voc_gate_wrong_continue_saturation_rate` is below
  `1%`; report the symmetric wrong-stop rate as a safety diagnostic, while the
  later 200k specification still requires both wrong-side rates below `1%`;
- pooled online-versus-EMA non-tie sign agreement is at least `0.60`;
- pooled held-out EMA selected-action TD RMSE is at most `0.5`;
- actor, online-Q, and gate AMP skips and all non-finite counters are zero.

Alignment telemetry is diagnostic at 100k: report its BCE, parameter energy,
target/gate/error norms, bias error, relative-error and cosine defined counts,
raw/post-clip gradient norms, clipping count, and update/skip counters.  These
metrics must be finite and internally consistent, but no post-hoc numerical
alignment threshold may be introduced for this run.

## Frozen 200k and fixed-checkpoint confirmation

The fresh 200k run, if authorized by the 100k pass, is judged without edits by
`VOC_V7_200K_ACCEPTANCE.md`: full `100000 < real_step <= 200000`, late
`175000 < real_step <= 200000`, its three stability windows, exact event
pooling, support, coverage, four learned behaviours, calibration, and safety
requirements.  Overshoot is excluded.

The final fixed-checkpoint confirmation uses training disabled, epsilon zero,
the learned stochastic gate, seeds `20260827..20260842`, exactly 16 streams by
6250 real steps (100k total), and calibration V-trace unroll 201.  It must use
the hardened fail-closed evaluator and satisfy the four-behaviour confirmation
in the 200k specification.  Diagnostic or shortened evaluator runs cannot qualify.

No failed stage may be rescued by extending the same run, changing a threshold
or population, selecting another seed, or interpreting a diagnostic evaluator
as the preregistered confirmation.
