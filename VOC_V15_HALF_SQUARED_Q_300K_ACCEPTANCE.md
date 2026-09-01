# Enduro VoC-v15 half-squared-Q preregistered acceptance

This protocol is frozen after the sole VoC-v14 seed-1 100k qualification
failed and before any v15 implementation edit, immutable snapshot, wire,
qualification, primary, or fixed evaluation.  V14 remains a permanent
failure.  V15 is one separately named, prospective, no-retry experiment that
changes exactly one algorithmic rule:

1. for gate-policy schema 8 only, replace each selected-action VoC-Q training
   loss row from the schema-7 beta-1 Smooth-L1 value to exact half-squared TD
   error, `0.5 * (selected_q.float() - target.float()).square()`, and sum those
   rows on the unchanged `q_train_valid` mask.

Nothing else changes.  In particular, v15 does not add a loss coefficient,
head, parameter, tensor, optimizer, training example, held-out gradient,
target, action mapping, value reconstruction, EMA rule, projected-gate rule,
execution distribution, barrier, ModelBuffer seal, acceptance population, or
threshold.  A v15 observation may not be used to tune this loss, select a
second seed, relax a gate, or create a retry.

## Permanent v14 qualification failure

The only v14 qualification was
`enduro-voc-v14-sealed-eps25-seed1-qual-fresh-100k`, run once from immutable
snapshot `/tmp/di-voc-v14-sealed-eps25-final-eKgdrk`.  Its driver exited zero
after `1918.067587` seconds.  Schema-7 actor/version, Q-to-EMA-to-projection,
sealed ModelNet input/drain, W&B two-phase completion, public finish, source,
data, process, Ray, and GPU validation all passed.  This successful mechanics
closure cannot rescue the numeric qualification failure.

The frozen canonical population contains exactly 63 complete CSV rows with
`70000 < real_step <= 100000`, from 70032 through 99376.  The terminal row at
100112 is overshoot and is excluded.  The sole failed gate is the pooled
held-out EMA selected-action TD RMSE:

```text
sqrt(sum(row_holdout_count * row_ema_selected_action_td_rmse^2) / 7703)
    = 0.5247954789453232 > 0.5
```

The online-Q companion is `0.5248123566562355`.  Every other inherited 100k
gate passed:

- teacher gap `0.27034248188280874`, student gap
  `0.26493562794609854`, retention `0.9799999840977489`, and signed margin
  `0.26223210510810246`;
- all three fixed windows had positive student gap and positive signed
  margin;
- all 59 trailing-five endpoints had both sign denominators, none was
  negative, and the maximum negative run was zero;
- train CONTINUE/STOP support was `28219/25794`, held-out CONTINUE/STOP
  support was `4016/3687`, and online-versus-EMA non-tie delta-sign agreement
  was `46420/61714 = 0.7521794082380011`;
- the separate EMA-Q versus exact-projected-gate sign diagnostic was
  `61715/61715`;
- wrong-CONTINUE saturation was `0/28368`, wrong-STOP saturation was
  `0/25644`, and the report-only forced-stop diagnostic was
  `121/29602 = 0.004087561651239781`; and
- actor, online-Q, gate, and ModelNet AMP-skip and non-finite counters were
  all zero.

V14 qualification failure permanently forbids a v14 primary, v14 fixed
evaluation, resume, extension, replacement seed, or retry.  V15 is not a
second v14 attempt and may not load any v14 checkpoint or runtime state.

## Frozen v14 failure artifacts

The v14 run directory is
`/tmp/di-voc-v14-sealed-eps25-final-eKgdrk/runs/enduro-voc-v14-sealed-eps25-seed1-qual-fresh-100k`.
Its exact regular-file SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `ckp_actor.tar` | `a86094cc3131bbe55bd5ac5995203cf8f3830ab36fbaf60e630306b0fe1d107c` |
| `ckp_actor.tar_step_480` | `95db486d6ff0f978e235bb762174bf63f0ef37a0db7c90c409bdec4500747173` |
| `ckp_actor.tar_step_100112` | `76ea5211bf0a6b5ba0070f29b751241769ebc07f3234069fa02a58c36fab3a7a` |
| `ckp_model.tar` | `6d6c2ff3ba5335337476998eb13be6ea80e9962d5ffabc67ade3020cb362af7a` |
| `ckp_model.tar_step_10000` | `f0cbcb5b5ae0a0fab2c4769637d3c32af37d4a481e054c10e32ffe4009152471` |
| `ckp_model.tar_step_100112` | `6d6c2ff3ba5335337476998eb13be6ea80e9962d5ffabc67ade3020cb362af7a` |
| `config_c.yaml` | `97d6a78d78d0d49dbd3e038fa77f268c29d4b21244c98d345060505f70c40350` |
| `finish` | `5ea6592e5f57cafb908902eed9decd7535e78d4e9b9fa33c23f7bf00b16f3b4e` |
| `logs.csv` | `6d72052279a59983d102bed3db767b3792b115d221f155cdd10d4dc29bbe85a8` |
| `logs_model.csv` | `b38dfd254fab250394ca463c4371c573b8c37f332e5f5f0b43c6d9b5fbfc3f64` |
| `meta.json` | `1799dc6bc2fb9be4acddffaea4bbd7c1b09d690ed2d04b92be7315088e0c796f` |
| `meta_model.json` | `04bcd2c1eb1738fd4a513b2aa9ef7a76943e84ea70e645dd1bf73bcf5eb08b47` |
| `out.log` | `67015c9a0896551cdbf19aa49e5d673837d947e50df415bbecdd835a1a8c4006` |
| `out_model.log` | `8e4a649f2b936ee549633001cc49641e01365b0199085d424cac63f5c66313d6` |

The one launch runtime `/tmp/v14qual100-hZGppv` has provenance SHA-256
`0e9f4ad97ea5774ccc579da43118546b1c837d6e21ceaa3b9be8d7d330af5b54`,
driver-log SHA-256
`04b18b4110c247425cd6234ebc097eb6efabd99a140ae3865f156615db072833`,
and exit-file SHA-256
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
The exit file contains exact `0` plus one trailing LF.  The immutable snapshot
source manifest is
`4fa507706250f349acfb6034453f1e8c70681c54e9e52718d848cd81755dbfea`
with 1063 entries; the data manifest is
`23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343`
with 11 entries.

Terminal config/actor/model identity was exact 229-key schema 7 with
complete-surface SHA-256
`ad87841c8c5dd18e9b5291b35eae3e473bfc4f22e0a5532872c5cf726d2c31b9`
and unchanged 209-key v12 projection SHA-256
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.
Actor policy version and publication count were 204 with a 205-event history,
terminal ack `1/1`, state SHA-256
`330fa1dee9ad167c46acc1c81eada7852daf3677fd1e0909f72d9a6eff2425fc`,
and history SHA-256
`a7d1e4f6d515d71cdfa9b936d0fece6fd2f300e71fc077666d8070aca757cde0`.
Model input sealed once at processed/real step 100112 and took one terminal
drain from pre-real 100096 and pre m/p counts 744 to final counts 745, with
zero late writes and aborts.

## Prospective rationale, not a causal promise

The exact change was selected prospectively after freezing the v14 failure.
On the v14 canonical qualification training population, the already-logged
targets and predictions give full pooled beta-1 Huber loss
`0.12421774185643177`, corresponding half-MSE `0.1721628948929676`, and
`E[error^2] = 0.34432578978593525`.  Define the diagnostic tail-energy
fraction exactly as

```text
E[(abs(error) - 1)_+^2] / E[error^2]
    = (E[error^2] - 2 * E[Huber_beta=1(error)]) / E[error^2].
```

Its full value is `0.2784871447842638` (`27.85%`); W1, W2, and W3 values are
`0.05739221686026769`, `0.24244326953644876`, and
`0.31423144488463206` (`5.74%`, `24.24%`, and `31.42%`).  The analogous full
values in the prior v11 and v12 evidence are `0.045495154619112216` and
`0.02894378200595532` (`4.55%` and `2.89%`).  This is the fraction of train
online squared-error energy equal to the excess beyond beta-1 Huber's
quadratic-to-linear transition.  It is not the fraction of samples satisfying
`abs(error) > 1`, a held-out-tail fraction, a gradient fraction, or a causal
fraction of the v14 qualification failure.

V14 W3 held-out EMA selected-action TD RMSE is
`0.8753410807478021`.  Its observed SSE is `1203.7347740098712` of the full
canonical `2121.4855002393383`, or approximately `56.74%`.  With support fixed
at 7703, reaching RMSE `0.5` would arithmetically require an observed pooled
SSE/MSE reduction of
`195.7355002393383 / 2121.4855002393383 = 0.09226341646796843`
(`9.2263%`).  These facts motivate an objective whose tail gradient does not
saturate.  They do not establish causality, guarantee a held-out improvement,
authorize a coefficient search, or imply that any window, sign, margin,
support, safety, or primary behavior will remain in the same direction after
learning.

Two easier post-hoc explanations are insufficient.  A single global bias
correction has an optimistic held-out RMSE lower bound
`0.5245088518192085`, still above `0.5`; and the online-Q companion RMSE is
slightly worse than the EMA value, so EMA-lag-only tuning is not supported.
V15 changes neither bias handling nor EMA tau.

The unchanged dueling reconstruction is also an explicit limitation.  It
uses the detached state value plus a probability-centered VoC advantage.  A
common shift in the raw two-action VoC head cancels under that centering, so
the new loss cannot create or train an independent common-value correction.
V15 adds no state-value or residual-calibration head.  It changes only the
tail value and gradient applied to the existing centered selected-action
error.

The unsaturated tail gradient can increase raw VoC-Q gradient norms.  Across
the complete v14 run the observed raw maximum was approximately `1551.6`
against the unchanged effective nominal clipping boundary `1608`.  V15 does
not change clipping, precision, scaler, or skip semantics.  Any actor,
online-Q, gate, or model AMP skip or non-finite event remains a hard failure,
and clipping telemetry remains required and finite.

## Normative inheritance from v14

The hashed v14 protocol
`VOC_V14_SEALED_MODEL_INPUT_300K_ACCEPTANCE.md`, SHA-256
`f422089ec4df7479f89477a3e6e63744dc74d06d1ceab9d5ea10397b34500972`,
is incorporated verbatim except for these closed substitutions:

- experiment names and the fixed profile change from v14 to v15 identities;
- gate-policy schema 7 changes to schema 8; and
- the one selected-action Q training loss-row mapping changes exactly as
  specified below.

All v13 Changes A-C and v14 Change D remain unchanged.  Soft training epsilon
is exactly `voc_train_epsilon=0.02`; executed gate epsilon is exactly
`voc_gate_execution_epsilon=0.25`; main actor AMP initial scale is exactly 32;
the strict actor-policy version barrier, terminal publication history, W&B
two-phase completion, no-Ray-retry topology, exact EMA-to-gate projection,
and schema-7 model-input-seal mechanism continue as their schema-8 successors.

The v14 ModelBuffer last-write acknowledgement, model-update claim
linearization, exact 13-key runtime status, one terminal input seal,
zero-or-one fresh terminal drain, durable ModelNet save, exact ten-field seal
evidence, successful completion ordering, and no-post-terminal-action contract
are unchanged.  Model-input-seal schema remains exact integer 1.

Every v14 network architecture, tensor shape, initialization, optimizer and
scheduler topology, parameter ownership, actor/model precision rule, target,
action mapping, trajectory, replay rule, learning-rate schedule, gradient
clip, scalar coefficient, telemetry definition, artifact rule, population,
window, sufficient-statistic pooling rule, support floor, and acceptance
threshold remains unchanged unless the single loss-row substitution below
requires an explicit schema-8 interpretation.

Schemas 1 through 7 retain exact historical behavior, keysets, return shapes,
public records, fixed records, and checkpoint semantics.  In particular,
schema 7 continues to use beta-1 Smooth-L1 and remains byte-compatible with
the immutable v14 implementation and artifacts.  No v15 change may be
backported, defaulted, or normalized into a legacy schema.

## The one schema-8 loss change

Let the existing schema-7 computation produce the same strict tensors:

- `selected_q`, obtained by gathering the reconstructed two-action Q with the
  existing gate action;
- `target`, with all existing detaches, reward/value construction, V-trace,
  think-cost, and validity rules; and
- `q_train_valid`, the existing valid non-holdout mask.

For gate-policy schemas at most 7, the loss remains the existing
`F.smooth_l1_loss(selected_q.float(), target.float(), reduction="none")`,
with default `beta=1`, summed on `q_train_valid`.

For gate-policy schema 8 only, define exactly:

```python
q_error = selected_q.float() - target.float()
q_loss_rows = 0.5 * q_error.square()
q_loss = torch.sum(q_loss_rows * q_train_valid.float())
```

The factor `0.5`, FP32 casts, sign convention, elementwise square, unchanged
mask, multiplication, and sum reduction are normative.  There is no mean at
this internal reduction and no epsilon, clamp, Huber transition, importance
weight, clipping, auxiliary term, or new coefficient.  The existing outer
`voc_loss_cost` remains exactly 1.0.  The existing telemetry
`actor/voc_q_loss` under schema 8 is the mean half-squared training TD error:
the summed `q_loss` divided by the unchanged positive `q_train_valid` count.
All-held-out or zero-train-support behavior remains the existing no-Q-update
path and may not divide by zero or synthesize an update.

For every finite error with `abs(error) <= 1`, beta-1 Smooth-L1 and the v15
half-squared loss have identical value `0.5 * error^2` and identical gradient
`error`, including the boundary.  Only finite tail rows change: the old loss
is `abs(error) - 0.5` with derivative `sign(error)`, while the v15 loss is
`0.5 * error^2` with derivative `error`.  This local mathematical identity
does not imply equality of future parameter trajectories or invariance of Q
differences, behavior, calibration, or acceptance direction after a tail row
changes an update.

The target, selected gate action, dueling reconstruction, detached state
value, probability centering, target behavior logits, held-out mask, data
ordering, and training support are byte-identical before the loss mapping.
Held-out rows remain telemetry-only and contribute zero loss and zero gradient.
The main actor optimizer and policy loss are unchanged and remain disjoint
from VoC-Q parameters.  The isolated VoC-Q Adam uses the unchanged actor
learning rate, epsilon, beta values, scaler, gradient clipping, scheduler, and
successful-step counter.  `voc_loss_cost=1.0` remains unchanged.

After a successful Q step, the unchanged transaction updates online Q, then
EMA with exact tau `0.1`, then exactly projects the EMA affine target into the
dedicated gate.  A Q skip or no-support branch still leaves EMA and projected
gate unchanged.  Q, EMA, and projection counters remain lockstep; the
projected gate optimizer remains empty with pristine scheduler/scaler state.
No new parameter, buffer, optimizer slot, scheduler, scaler, tensor key, or
checkpoint state is introduced.

## Exact schema-8 identity

V15 uses strict non-boolean Python integer
`voc_gate_policy_schema_version=8`.  Gate schema 8 is the sole new algorithm
identity; there is no new configuration key.  The persisted configuration,
actor checkpoint, and ModelNet checkpoint each retain the exact 229-key
surface:

```text
229 = 209 v12 stage-neutral keys
    + 6 stage keys
    + 4 path-derived keys
    + 10 v13/v14 protocol keys
```

The six stage keys are `xpid`, `base_seed`, `total_steps`,
`model_warm_up_n`, `actor_unroll_len`, and `use_wandb`.  The four path-derived
keys are `savedir`, `ckpdir`, `cmd`, and `icopro_data_path`.  The ten protocol
keys remain exactly:

```text
voc_gate_policy_schema_version
voc_gate_execution_epsilon
voc_actor_policy_version_barrier
voc_actor_policy_bundle_schema_version
voc_actor_policy_barrier_timeout_s
voc_actor_policy_ray_max_restarts
voc_actor_policy_ray_max_task_retries
actor_amp_init_scale
voc_actor_policy_barrier_runtime
voc_model_input_seal_schema_version
```

Their v15 values remain the v14 values except for gate schema 8:
execution epsilon 0.25; version barrier true; bundle schema 1; timeout 120.0;
Ray actor restart and task-retry counts zero; actor AMP initial scale 32;
barrier runtime true in training; and model-input-seal schema 1.  Missing,
extra, defaulted, coerced, non-finite, or wrongly typed keys fail closed.  The
full surface keyset must be exactly 229; an unknown extra key is not an
extension point.

The 209-key stage-neutral v12 projection remains byte-identical with SHA-256
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.
Configuration, actor metadata, and ModelNet metadata must agree exactly on the
229-key surface and its canonical complete-surface digest.  Each stage's new
complete digest is resolved from its exact v15 paths and command and must be
recorded before launch; it is not guessed or normalized in this document.

Every actor-policy bundle retains the exact existing five keys
`{bundle_schema_version, policy_version, terminal, gate_schema,
actor_state_dict}` with bundle schema 1 and gate schema 8.  Every ack retains
the exact five keys `{bundle_schema_version, gate_schema, rank,
policy_version, terminal}` with the same identities.  Publication-history
events retain the exact seven keys `{predecessor_version, policy_version,
publication_count, terminal, ack_ranks, expected_ack_count, state_sha256}`;
they do not add gate schema.  The surrounding schema-8 checkpoint plus the
validated bundle and acks binds the history.

Model-input-seal schema and persisted evidence do not change.  Every terminal
ModelNet checkpoint retains the exact ten fields and v14 relations:

```text
voc_model_input_seal_schema_version
voc_model_input_sealed
voc_model_input_seal_count
voc_model_terminal_processed_n
voc_model_terminal_drain_update_count
voc_model_terminal_drain_pre_real_step
voc_model_terminal_drain_pre_grad_step_count_m
voc_model_terminal_drain_pre_grad_step_count_p
voc_model_input_late_write_count
voc_model_input_abort_count
```

Seal schema is strict integer 1, sealed is exact true, seal count is 1,
terminal processed n equals final ModelNet real step and is at least configured
total steps, drain count is strict integer zero or one, final m/p counts equal
their pre-drain counts plus drain, and late-write and abort counts are zero.
Drain-zero and drain-one branch relations, finiteness, optimizer/scheduler/
scaler state, durable save, complete-success ordering, and public-finish
ordering remain unchanged.

Training finalization and terminal validation must dispatch schema 8 to a
dedicated `validate_schema8_final_bundle`.  V15 is fresh-only: `ckp=true`, a
resume request, or any preload/parent checkpoint is rejected before training
or state restoration and does not define a positive schema-8 resume path.
Public completed-bundle validation must use the dedicated
`validate_schema8_completed_bundle` route before any environment reset,
action, evaluator-direct/downstream checkpoint tensor use, or output creation.
The authoritative validator's own bound, safe actor/ModelNet deserialization
is part of validation and is not a downstream evaluation load.  Smoke must
validate stored config/actor/model schema-8
identity and the complete actor/ModelNet terminal bundle both before and after
its private evaluation copy.  Fixed evaluation must validate the schema-8
bundle before flags loading, environment construction/reset/action,
evaluator-direct/downstream tensor load, rollout, or output creation.

The authoritative schema-8 validator's JSON-safe `resolved_identity` adds the
exact derived string `voc_q_regression_loss: "half_squared_td"`.  Public,
smoke, and fixed schema-8 protocol/summary/manifest records preserve and
validate that derived value.  It is computed from the already-validated strict
gate schema 8; it is not a configuration, embedded-flags, or checkpoint key,
does not make a 230th persisted surface field, and introduces no tensor state.
Schema-7 and earlier `resolved_identity` keysets and return shapes remain
exactly unchanged.

Schema-8 public JSON-safe records add no tensor-bearing field and preserve the
v14 record shapes except for resolved schema/version/profile identity.  Public,
smoke, and fixed v15 paths must reject schema 7, a v14 stage tuple, missing or
wrong bundle/ack schema, incomplete seal evidence, failed logger completion,
or private markers.  Legacy public/smoke/fixed profiles must reject schema 8
and retain byte-identical schema-6 and schema-7 outputs.

## Immutable stage identities and no-retry rule

Every v15 stage is fresh: `ckp=false`; `preload`, `preload_actor`, and
`voc_parent_checkpoint` are empty; parent-update count is zero; actor policy
version starts at zero; and online-Q, EMA-Q, projected gate, ModelNet, buffers,
and seal state start from the unchanged exact fresh state.  No checkpoint,
optimizer, buffer, version, observation, or result crosses stages.

The only valid tuples
`(xpid, base_seed, total_steps, model_warm_up_n, actor_unroll_len, use_wandb)`
are exactly:

- (`enduro-voc-v15-halfsq-eps25-sentinel-wire1200`, 1, 1200, 512, 41,
  false);
- (`enduro-voc-v15-halfsq-eps25-seed1-qual-fresh-100k`, 1, 100000, 10000,
  201, true); and
- (`enduro-voc-v15-halfsq-eps25-seed5-strict-fresh-300k`, 5, 300000,
  10000, 201, true).

Each xpid is an exact Python string.  Trimming, case change, coercion, or
whitespace normalization is forbidden.  Numeric tuple members are strict
non-boolean Python integers and `use_wandb` is an exact Python boolean.
Configuration, actor metadata, and ModelNet metadata must agree on the tuple;
the normalized `ckpdir` basename must equal the xpid.  `savedir`, `ckpdir`,
`cmd`, and `icopro_data_path` derive from the same immutable v15 snapshot and
must match across all three surfaces.

All stages retain `schedule_total_steps=100000000`, the v14 Enduro data and
network configuration, CUDA devices 0 and 1 only, and Ray resources two GPUs
and 16 CPUs.  The wire is W&B-disabled; qualification and primary require W&B
and the inherited strict two-phase logger close/ack/cleanup before public
finish.

Each stage has exactly one attempt.  The wire is nonqualifying and cannot be
continued into qualification.  Qualification is one separate fresh seed-1
run.  Primary is one separate fresh seed-5 run and may start only after every
qualification gate passes.  There is no resume, extension, fallback seed,
retry, duplicate xpid, selected checkpoint, or replacement run.  Any launch,
identity, artifact, mechanism, numeric, W&B, process, Ray, or GPU cleanup
failure permanently ends the v15 claim at that stage.

## Sequential release gates

The exact release order is:

1. Implement only this document's schema-8 loss mapping and required
   validation/profile/test overlay in the mutable worktree.  Freeze all owned
   bytes and pass two independent code/contract audits.
2. Build a fresh inode-independent immutable snapshot from the authoritative
   v14 source/data baseline plus an exactly enumerated stable v15 overlay.
   Independently audit manifests, modes, binaries, cp310-only runtime binding,
   schema-8 validation, tests, and this document.  No launch is allowed until
   two independent snapshot audits pass.
3. Run exactly one fresh seed-1 1.2k wire with its exact tuple.  Inspect only
   mechanics, artifacts, lifecycle, W&B-disabled completion, source/data
   integrity, and cleanup.  Reward or behavioral acceptance cannot influence
   a wire decision or implementation edit.
4. Only after the wire passes, run exactly one fresh seed-1 100k qualification
   with production warm-up/unroll and W&B enabled.  Apply every inherited v14
   qualification gate unchanged.
5. Only after qualification passes every gate, run exactly one fresh seed-5
   300k primary.  Apply every inherited v14 primary gate unchanged.
6. Only after that primary passes may its terminal checkpoint receive one
   eligible fixed confirmation under exact profile `v15-300k`.

Pong, Space Invaders, another Enduro seed, and any fixed evaluation before an
accepted primary remain out of scope.

## Integrity-wire acceptance

The wire may inspect only immutable binding, exact configuration and CLI,
loss-branch/schema identity, first and final checkpoint completeness, actor
versions/acks/history, Q/EMA/projection transaction counts and state, AMP and
non-finite counters, ModelBuffer seal/drain ordering and evidence, W&B-disabled
logger completion, public finish, source/data manifests, and process/Ray/GPU
cleanup.  It must exercise at least one successful supported Q transaction,
schema-8 half-squared-loss telemetry, a nonterminal actor publication, the sole
terminal publication and ack, the correct seal and zero-or-one drain branch,
durable model save, complete-success, and exact true ModelLearner/SelfPlay
returns.

The wire supplies no qualifying behavioral row.  A live Q skip, actor/gate/
model AMP skip, non-finite value, malformed bundle, timeout, late write, abort,
retry, stale ModelNet checkpoint, missing finish, W&B artifact, or incomplete
cleanup permanently fails v15.  Negative paths need not occur live but must be
covered by frozen tests.

## Frozen 100k qualification

The v14 Frozen 100k decision and all incorporated v13/v12/v11/v9 algebra are
verbatim.  Canonical rows satisfy `70000 < real_step <= 100000`; fixed windows
are `(70000,80000]`, `(80000,90000]`, and `(90000,100000]`; overshoot is
excluded.  Required cells must be finite, rows complete, real steps unique and
strictly increasing, and malformed/duplicate/nonmonotone input fails closed.

The run passes only if all inherited gates pass together:

- teacher gap at least `0.075`, student gap at least `0.05`, retention at
  least `0.50`, and signed margin strictly positive;
- at least two of three windows with both student gap and signed margin
  strictly positive;
- maximum consecutive negative trailing-five pooled gaps at most 3, with
  every positive/negative denominator valid and exact zero nonnegative;
- train and held-out CONTINUE and STOP fractions each strictly above `0.05`;
- wrong-CONTINUE saturation strictly below `0.01`, with wrong-STOP and the
  inherited forced-stop measure reported under their unchanged status;
- online-versus-EMA non-tie sign agreement at least `0.60`;
- held-out EMA selected-action TD RMSE at most `0.5`; and
- actor, online-Q, gate, and ModelNet AMP skips and all non-finite counters
  exactly zero.

The schema-8 half-squared-loss mapping, exact 229/209 identity, Q/EMA/
projection transaction, actor barrier/history, W&B completion, schema-1 model
input seal/exact-ten evidence, public finish, manifests, and cleanup are hard
integrity gates.  They add no numerical qualification threshold.  A failed
qualification permanently ends v15 and forbids primary and fixed evaluation.

## Frozen 300k primary acceptance

The v14 Frozen 300k decision and its incorporated v13/v10 algebra are
verbatim.  Full remains `(100000,300000]`, late remains `(250000,300000]`, and
W1/W2/W3 remain `(270000,280000]`, `(280000,290000]`, and
`(290000,300000]`.  Overshoot above 300000 is excluded.

Every inherited threshold remains unchanged, including learned soft-gate
probability `0.475/0.525`, sampled-control strength `0.525`, conditional
argmax `0.60`, useful-pair coverage `0.95`, sign agreement `0.60`, strict
support fractions above `0.05`, wrong-side saturation and forced-stop rates
below `0.01`, held-out RMSE at most `0.5` where inherited as a training gate,
the exact direction/strength/window requirements, and zero AMP skips or
non-finite events.  Soft behavior/calibration probabilities use training soft
epsilon 0.02; sampled execution, stored likelihood, V-trace, and joint-policy
entropy weighting use execution epsilon 0.25.

All artifact, mechanism, behavior, stability, support, trailing-five,
saturation, forced-stop, calibration, actor-barrier, schema-8 loss, model seal,
AMP, and non-finite gates must pass together.  There is no partial,
diagnostic-only, support-only, mechanism-only, or historical pass.

## Fixed-checkpoint confirmation

The closed fixed profile is exactly `v15-300k` and accepts only the one
accepted seed-5 primary tuple.  It rejects wire, qualification, v14, and every
legacy schema before any rollout or output.  Held-out seeds remain 20260827
through 20260842, exactly 16 streams by 6250 real steps and 100000 total, with
calibration V-trace unroll 201 and all inherited behavior/calibration algebra.

Before any flag normalization, environment construction/reset/action,
evaluator-direct/downstream tensor load, rollout, or output, fixed evaluation
validates the complete schema-8
config/actor/model 229-key identity, 209-key projection, bundle/ack/history,
full actor/Q/EMA/projection optimizer/scheduler/scaler state, complete ModelNet
state, exact ten seal/drain fields and relations, W&B logger completion,
private-marker absence, public finish, source/runtime binding, and primary
stage tuple through the dedicated schema-8 final/completed-bundle route.

Only after immutable validation may an evaluator-private copy disable actor
training, ModelNet training, parallel execution, live barrier waiting, and
live ModelBuffer seal coordination.  It records immutable training soft
epsilon 0.02, execution epsilon 0.25, schema 8, and seal schema 1 while using
runtime soft epsilon zero, execution epsilon zero, barrier wait false, and
model-input-seal coordination false.  It never rewrites config or checkpoints.
Fixed B/calibration probability continues to use the recorded soft learned
gate field, not the epsilon-zero execution likelihood.

Only an accepted primary plus its accepted `v15-300k` fixed confirmation can
support a v15 Enduro claim.

## Frozen test and audit matrix

Before stable implementation or snapshot clearance, frozen tests must cover
at least the following exact matrix.

### Loss values, gradients, and reduction

- Schema 8 computes exact per-row `0.5 * error.square()` in FP32, multiplies
  only by unchanged `q_train_valid.float()`, and sums with no hidden mean,
  clamp, epsilon, coefficient, or extra factor.
- Hand-computed positive, negative, zero, subunit, exact-unit, and tail errors
  bind forward values and gradients, including the exact `0.5` factor.
- Schema-7 beta-1 Smooth-L1 and schema-8 half-squared values and gradients are
  exactly equal for `abs(error) <= 1`; tail values and gradients follow their
  distinct formulas.
- Logged `actor/voc_q_loss` is the summed half-squared loss divided by the
  exact positive training count and never by valid-plus-holdout support.

### Masks, isolation, and dueling gauge

- Mixed train/holdout masks prove held-out rows contribute zero value and zero
  gradient; all-held-out/no-support performs no Q, EMA, or projection update.
- Selected-action gather and STOP/CONTINUE mapping remain exact; the
  unselected action receives only the gradients implied by unchanged dueling
  probability centering, not a new supervised target.
- Actor-policy, dedicated-gate, ModelNet, target, state-value, behavior-logit,
  and held-out tensors retain their existing detach/gradient isolation.
- Common-shift gauge tests prove equal raw-Q shifts cancel under the unchanged
  dueling reconstruction and that schema 8 introduces no common-value head or
  parameter.

### Precision, non-finite, clipping, and AMP

- Squaring occurs after exact FP32 conversion.  Finite inputs whose square and
  gradient are representable in FP32 produce the exact finite result; a large
  finite input that overflows during squaring, or any NaN/Inf input, loss, or
  raw gradient, fails closed before an optimizer, EMA, or projection update.
  The unchanged v14 path does not add a pre-step scan of existing Adam state,
  scaler state, parameters, or later telemetry.  Their recursive finiteness is
  enforced by the existing post-step checkpoint, terminal-bundle, artifact,
  and acceptance validators; any violation hard-fails the run and cannot be
  laundered as an accepted artifact or public finish.
- Existing online-Q evidence remains exactly `actor/voc_total_norm` for the
  raw finite gradient norm, `actor/voc_optimizer_stepped`,
  `voc/amp_scale_before`, `voc/amp_scale_after`, `voc/amp_skip_count`,
  `voc/amp_consecutive_skips`, and
  `voc/nonfinite_gradient_parameter_count`.  The last field counts parameters
  whose gradients were non-finite; it is not a parameter-state counter.  Tests
  exercise the unchanged `clip_grad_norm_` effect and effective
  `clipping * T * B` boundary through gradients/parameter updates; v15 adds no
  Q postclip norm, clipping counter, or checkpoint field.
- A finite over-threshold online-Q gradient is clipped and may step normally.
  Only the inherited AMP-skip, non-finite, or optimizer-failure branches
  suppress Q, EMA, and projection updates; none can be laundered as success.
- Main actor, dedicated gate, and ModelNet AMP/FP32 independence remains
  unchanged.

### Q-to-EMA-to-projection transaction

- A successful schema-8 Q step advances online Q exactly once, EMA exactly
  once at tau 0.1, and exact projection exactly once, with lockstep counters
  and bit-equal projected affine state.
- No-support, a VoC-Q AMP skip, malformed loss, or VoC-Q optimizer failure
  leaves EMA and gate unchanged under the inherited branch contract.  A main
  actor AMP skip is independent: it does not suppress an otherwise successful
  Q-to-EMA-to-projection transaction, although the skip still hard-fails live
  acceptance.
- Periodic and terminal checkpoint serialization preserves the unchanged
  online-Q/EMA/gate state, isolated Adam slots, scheduler, scaler,
  AMP-skip/non-finite counters, and update counts; no new tensor or optimizer
  state exists.  V15 launch guards reject every resume/preload attempt rather
  than treating that preserved state as launch authority.

### Schema, artifacts, public surfaces, and legacy parity

- Schema 8 requires exact 229-key config/actor/model surfaces, exact 209-key
  projection SHA, bundle gate schema 8, ack gate schema 8, unchanged five-key
  bundle/ack and seven-key history shapes, seal schema 1, and exact ten terminal
  model fields.
- Schema-8 authoritative/public/smoke/fixed JSON-safe records require derived
  `voc_q_regression_loss="half_squared_td"`; persisted config/checkpoint
  surfaces remain exactly 229 keys, and schemas at most 7 must not acquire this
  resolved-identity field.
- Dedicated `validate_schema8_final_bundle` and
  `validate_schema8_completed_bundle` routes reject schema 7, schema 9,
  bool/string/float schema, missing/extra keys, cross-surface drift, legacy
  stage, malformed state, incomplete logger/finish, and private markers.
- Schema 6 and schema 7 retain byte-identical keysets, return shapes, runtime
  records, checkpoint behavior, public/smoke outputs, and fixed outputs under
  differential tests against immutable v13/v14 artifacts.
- Public, smoke, and fixed invalid-v15 negatives prove authoritative schema-8
  validation happens before `_load_flags`, live evaluation-spec resolution,
  environment construction/reset/action, data access, any evaluator-direct or
  downstream `torch.load`/checkpoint tensor use, rollout, output directory/file
  creation, or checkpoint rewrite.  The validator's internal bound
  deserialization of actor/ModelNet checkpoints is required to validate full
  state and is explicitly excluded from those downstream counters.  Fakes
  bind each forbidden downstream loader/probe call count to exact zero.
- Positive smoke ordering is exact: validate immutable stored config/actor/
  model identity and the completed schema-8 bundle; create only then the
  private evaluation copy; apply its training/barrier/seal-runtime overrides;
  revalidate the unchanged stored hashes, identity, and completed bundle; and
  only then construct or run the smoke environment.  Tests bind both validator
  calls, require exact equality of pre/post authoritative evidence, and bind
  the pre-copy/private-copy/post-copy order.
- Fixed ordering repeats the v14 regression boundary explicitly: after the
  bound public-module import and requested-profile resolution, schema-8
  authoritative bundle/evidence validation completes before `_load_flags`,
  live spec/environment probe, evaluator-direct/downstream `torch.load`,
  rollout, or output.  Invalid schema-8 evidence leaves every later downstream
  call count and output count at zero; valid evidence is reused unchanged
  rather than reconstructed after a live probe.
- The fixed `v15-300k` profile accepts only the exact primary and preserves
  immutable training identity before applying its private runtime overrides.

### CLI, topology, lifecycle, and no retry

- Real `create_setting(save_flags=False)` tests bind all three exact xpids and
  tuples, exact 229/209 surfaces, schema 8, seal 1, schedule 100M, path/command
  derivation, fresh inputs, topology, and W&B mode.
- Wrong xpid/whitespace, seed, total, warm-up, unroll, W&B, schema, path,
  preload, resume, topology, retry, or resource setting fails before a run
  directory or environment action.  Wrong or missing derived loss identity is
  a schema-8 validator/public/smoke/fixed JSON-evidence negative, not a CLI or
  persisted configuration field.
- Real-Ray tests retain the actor version barrier, model claim/seal/drain,
  logger two-phase completion, timeout, abort, kill, no-restart/no-task-retry,
  private-marker quarantine/cleanup, public-finish, and process/GPU cleanup
  matrices inherited from v14.
- Immutable-snapshot tests require exact manifests and coverage, independent
  inodes/modes, empty runs before launch, snapshot-only cp310 import, no loaded
  cp312/worktree artifact, focused/adversarial/full suites, and unchanged
  post-test hashes.

## Final claim boundary

V15 requires, in order, one accepted immutable snapshot, one accepted
mechanics-only seed-1 wire, one accepted fresh seed-1 100k qualification, one
accepted fresh seed-5 300k primary, and that primary's one accepted
`v15-300k` fixed confirmation.  Any failure is permanent for v15.  No later
mechanics pass, diagnostic, seed, checkpoint, or protocol version can
retroactively change the v14 failure or a v15 stage decision.
