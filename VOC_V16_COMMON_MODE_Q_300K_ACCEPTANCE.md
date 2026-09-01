# Enduro VoC-v16 common-mode-Q preregistered acceptance

This protocol is frozen after the sole VoC-v15 seed-1 100k qualification
failed and before any v16 implementation edit, immutable snapshot, wire,
qualification, primary, or fixed evaluation.  V14 and v15 remain permanent
failures.  V16 is the immediate successor of v15 and gate-policy schema 8.  It
is one separately named, prospective, no-retry experiment that changes exactly
one algorithmic rule:

1. for gate-policy schema 9 only, retain v15's exact half-squared selected-
   action TD loss and add the raw two-action VoC-head mean as one common-mode
   term to both the online and EMA reconstructed Q values.

The exact reconstruction is:

```python
raw = A
common = raw.mean(dim=-1, keepdim=True)
centered = raw - torch.sum(p_det * raw, dim=-1, keepdim=True)
Q = V_det.unsqueeze(-1) + common + centered
```

`p_det` and `V_det` retain their existing definitions and detaches.  This
activates the equal-row raw-head direction while algebraically preserving in
real arithmetic `Q_CONTINUE - Q_STOP = A_CONTINUE - A_STOP`.  It introduces no new head,
parameter, buffer, trainable or persisted tensor state, tensor key or shape,
optimizer, checkpoint state, configuration key, training example, held-out
gradient, target, selected action, gate-projection rule, sampling/execution
rule or configuration, barrier, ModelBuffer mechanism, acceptance population,
or threshold.  `common` and `centered` are expected ephemeral autograd/inference
intermediates, not state.

The two schema-9 derived identities are exactly
`voc_q_regression_loss="half_squared_td"` and
`voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"`.
They appear only in validated JSON-safe resolved identity.  Neither is a CLI,
YAML, persisted-config, embedded-flags, actor-checkpoint, or ModelNet-checkpoint
key, and neither creates a 230th persisted field.

A v16 observation may not be used to tune a second lever, pick a replacement
seed, relax a gate, or authorize a retry.  This document makes no prediction or
guarantee that common-mode reconstruction will improve held-out calibration or
pass qualification.

## Permanent v15 qualification failure

The only v15 qualification was
`enduro-voc-v15-halfsq-eps25-seed1-qual-fresh-100k`, launched once from
immutable snapshot `/tmp/di-voc-v15-halfsq-eps25-final-CKcYfI`.  It was fresh,
seed 1, W&B-enabled, schema 8, seal schema 1, and bound to the exact 229-key
surface and unchanged 209-key projection.  The driver exited zero.  Actor,
online-Q-to-EMA-to-exact-projection, actor-policy barrier/history, sealed
ModelNet input/drain, W&B two-phase completion, public finish, manifests,
process, Ray, and GPU validation all passed.  Successful mechanics cannot
rescue the numeric failure.

The frozen canonical population contains exactly 58 complete CSV rows under
`70000 < real_step <= 100000`, from 70240 through 99920.  Terminal step 100736
is overshoot and is excluded.  The sole failed inherited gate is pooled
held-out EMA selected-action TD RMSE:

```text
sqrt(sum(row_holdout_count * row_ema_selected_action_td_rmse^2) / 8154)
    = 0.5637126651551874 > 0.5
```

The corresponding held-out EMA SSE is `2591.1126340547949`; the unchanged
threshold SSE is `8154 * 0.5^2 = 2038.5`; and the excess is
`552.6126340547949`.  At fixed support, reaching the ceiling would therefore
require a relative observed pooled SSE/MSE reduction of exactly

```text
552.6126340547949 / 2591.1126340547949
    = 0.2132723320444852
    = 21.32723320444852%.
```

This arithmetic is not an estimated treatment effect or a promised reduction.
The online-Q held-out companion is `0.5646944857965209`, only
`0.000981820641333564` above the EMA value.  That near-equality does not support
an EMA-lag-only explanation and does not authorize changing EMA tau.

Every other CSV-observable inherited qualification gate passed:

- teacher gap `0.2517803941787054`, student gap
  `0.24674480056618492`, retention `0.9800000566805596`, and signed margin
  `0.24578266135826818`;
- all three frozen windows had positive student gap and signed margin;
- all 54 trailing-five endpoints had both sign denominators, none was
  negative, and the maximum negative run was zero;
- train CONTINUE/STOP support was `31359/26547`, and held-out
  CONTINUE/STOP support was `4365/3789`;
- online-versus-EMA non-tie delta-sign agreement was
  `45743/66055 = 0.6924986753463023`, above 0.60;
- EMA-Q versus exact-projected-gate sign agreement was `66057/66057`;
- wrong-CONTINUE saturation was zero, wrong-STOP saturation was zero, and the
  report-only forced-stop rate was `0.0032528339`; and
- every actor, online-Q, gate, and CSV-observable protocol AMP-skip,
  consecutive-skip, non-finite, mismatch, malformed-bundle, and barrier-timeout
  counter was zero.

V15 failure permanently forbids a v15 primary, `v15-300k` fixed evaluation,
resume, extension, replacement seed, or retry.  V16 is not a second v15
attempt and may not load a v14 or v15 checkpoint, optimizer, buffer, actor
version, observation, or runtime state.

## Frozen v15 failure artifacts

The v15 qualification run directory is
`/tmp/di-voc-v15-halfsq-eps25-final-CKcYfI/runs/enduro-voc-v15-halfsq-eps25-seed1-qual-fresh-100k`.
Its exact 14 regular-file SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `ckp_actor.tar` | `ee41fa3f0916f5b726b6cd448e12989abbe01554a69c6f4eb5ab8ed434cd2c7c` |
| `ckp_actor.tar_step_480` | `7dae5f509ad1f4c18558c44a6d9347aca38ca3421d090664a5a88b7b53edfd58` |
| `ckp_actor.tar_step_100736` | `938a8545e2f0a38da45582a509a7d9eea37261420b99af2593af27870d90c7b8` |
| `ckp_model.tar` | `8d215ce13489735e685ed1d271ad857b69f23f0b35213447d0d90f67f96b82b8` |
| `ckp_model.tar_step_10000` | `5bc53d4740e431f4b5f0e5f289a8417cad71825bc18f7549fe71de4d5d271011` |
| `ckp_model.tar_step_100736` | `8d215ce13489735e685ed1d271ad857b69f23f0b35213447d0d90f67f96b82b8` |
| `config_c.yaml` | `edd3ce926dbb048543a1ea11ee0bca29bef2f83c88877f3ae1e4d85308e234b2` |
| `finish` | `e2d00bf7baa7276cca91bf1c73e025991e5d1c1ddfc6236ab8cec71c845d8184` |
| `logs.csv` | `359eb6540a6265a2a5b8e77fa8ac01420b5f0077a3b84d67800f0308f448e341` |
| `logs_model.csv` | `360e6d475d635f734e4b14ddfff7a0022ba7997131785ce2ee7a39ea28b6839b` |
| `meta.json` | `28069535c4e792c0d0ca2b8dd9ddfcb638c43374c20ffd565a551538064ca295` |
| `meta_model.json` | `a78d40eeb30c482fe6fbfc057de132d744520af7339459527f5ff029b65ddb50` |
| `out.log` | `54c4ebd93e23623ed9dc9cea217f4cccf5c2e4b31db889534d2c6cb0c8c2c077` |
| `out_model.log` | `b5be765cee5ead620f15912a8506d0f9d8f1dc01588edf5da4176321a9193662` |

The canonical SHA-256 of C-sorted `sha256sum` lines using `./` paths is
`ee400a7df101071287032c65de3c9bf272f03f02b5649421cfdae9d6df5ae5b3`.
The sole launch runtime `/tmp/v15qual-JhNlYp` has launch-provenance SHA-256
`ad1e625579aad7e9c400a4a85130939400ed1e0696b8c71987ce8e3bbfda9ccd`,
driver-log SHA-256
`d8a34610fc994daac81bb80ff654435e684d191d7d41ca12b81a6fa7beee142d`,
and exit-file SHA-256
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
The exit file is exact `0` plus one LF.

The immutable v15 snapshot source manifest is
`1c2d619479a63d75f0b23c41ef293f611b3631432356067a2db425a30ec62cdf`
with 1064 entries.  The data manifest is
`23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343`
with 11 entries.  Terminal config/actor/model identity was exact schema 8,
229 keys, and complete-surface SHA-256
`3e843b740eb8a6b1b742ebaec02e9753fb7edf1b07b4f30cdf2984c8bbc69ff3`.
The 209-key v12 projection SHA-256 was
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.

Actor terminal real step was 100736, policy version and publication count were
149, publication history length was 150, terminal acknowledgement was `1/1`,
and mismatch, malformed, timeout, AMP-skip, and non-finite counters were zero.
Actor state SHA-256 was
`1bf13b2d2508a96486baa2344d1d32f7eb4d934405c0ccd7a57906b9baec3b93`;
publication-history SHA-256 was
`37b9541553950b8c55b49211f23c3856b032473db8a921e895418fc92177a9e8`.
Model input sealed once at terminal processed/final real step 100736 and took
one fresh drain from pre-real 100144 and pre m/p gradient counts 746 to final
counts 747, with zero late writes and aborts.  Authoritative/public schema-8,
W&B request/ack/cleanup, finish, source/data, and process/GPU closure all
passed.  Those mechanics facts are integrity evidence, not numeric acceptance.

The frozen executable RCA notebook is
`notebook/v14_v15_qualification_rca.ipynb`, SHA-256
`45af4c50326188225f97ba1e0f81f6a1573328061a74b88d38ebe1da8f0c4968`.
It validates the v14 and v15 CSV hashes, identical ordered 922-column schema,
complete rows, finite numeric cells, strict integer steps, canonical
populations, sufficient-statistic pooling, and the numeric statements below.
It does not replace terminal bundle, W&B, manifest, or cleanup authority.

## Frozen RCA and prospective rationale

The v14-to-v15 result is not a uniform regression.  Frozen held-out EMA RMSE
moved by window as follows:

| Window | v14 rows/support | v14 EMA RMSE | v15 rows/support | v15 EMA RMSE |
| --- | ---: | ---: | ---: | ---: |
| W1 `(70000,80000]` | 32 / 3647 | `0.3070468494240733` | 19 / 2452 | `0.6350000482991449` |
| W2 `(80000,90000]` | 19 / 2485 | `0.4805763254032396` | 26 / 3509 | `0.5262197892944133` |
| W3 `(90000,100000]` | 12 / 1571 | `0.8753410807478021` | 13 / 2193 | `0.5362964249791570` |
| Full | 63 / 7703 | `0.5247954789453232` | 58 / 8154 | `0.5637126651551874` |

V15 improved the v14 W3 spike, but W1 moved from below the ceiling to the
largest v15 window error and W2 also crossed the ceiling.  All three v15
windows exceed 0.5.  This is an observed temporal redistribution, not proof
that the changed loss caused any window or that a common-mode term will repair
one.

Reconstructed online training RMSE improved from
`0.5867927997052582` in v14 to `0.48946925650925704` in v15, while held-out
online RMSE worsened from `0.5248123566562356` to
`0.5646944857965209`.  EMA train RMSE similarly moved from
`0.5865118456123475` to `0.48913625344503686`, while held-out EMA RMSE moved
from `0.5247954789453232` to `0.5637126651551874`.  V15's logged mean
half-squared Q loss `0.1197900774` reconciles with reconstructed training
half-MSE `0.11979007653386242` to CSV precision.

These are two single-seed, separately generated, policy-coupled on-policy
trajectories.  They contain 63 versus 58 canonical rows, different row cadence,
different support, and no paired states or counterfactual actions.  Rows may
not be paired, interpolated, reweighted across runs, or interpreted as a
randomized train/holdout treatment.  The opposing train/held-out direction is
a descriptive generalization warning, not causal identification.

The old schema-8 dueling reconstruction has an exact common-shift gauge.  For
normalized detached two-action probabilities `p_det` and any scalar row shift
`c`:

```text
old_Q(A + c * [1,1])
  = V_det + (A + c) - sum(p_det * (A + c))
  = V_det + A - sum(p_det * A)
  = old_Q(A).
```

Equivalently, the old weighted reconstructed mean is
`sum(p_det * old_Q) = V_det`, and the equal-row direction has zero selected-Q
gradient.  The existing two-output raw head therefore cannot use its common
direction to fit a state/action-common TD residual even though that direction
already exists in the unchanged affine head.

Schema 9 changes only that gauge.  In exact real arithmetic its new weighted
reconstructed mean is `sum(p_det * Q) = V_det + mean(A)`, and an equal raw
shift `c` shifts both Q actions by `c`.  Common, centered, and detached-value
terms algebraically cancel from the real-arithmetic action difference, leaving
`Q_CONTINUE-Q_STOP=A_CONTINUE-A_STOP`.  The projection rule continues to use
the current stored EMA raw affine delta and does not consume the reconstructed
common term.  This algebra motivates exposing an already-present raw-head direction;
it does not establish that v15's held-out residual is common-mode, recover an
unobserved state/action decomposition, or guarantee an acceptance improvement.

Optimization risk remains explicit.  Across all 149 v15 rows, exactly one
finite online-Q raw gradient norm exceeded the unchanged nominal boundary:
real step 39472 recorded `1767.575439453125`, above
`actor_grad_norm_clipping * actor_unroll_len * actor_batch_size =
0.5 * 201 * 16 = 1608`.  It was clipped and stepped normally; every skip and
non-finite counter stayed zero.  The canonical 70k-100k v15 raw-norm maximum
was `802.5406494140625`.  V16 retains the same clip, scaler, AMP, optimizer,
and fail-closed transaction.  It adds no post-clip telemetry or clipping
counter.

The immediate schema-9 lineage is an attribution choice, not an efficacy
claim.  V16 preserves schema 8's half-squared loss and changes only common-mode
reconstruction.  A hypothetical v14-Huber-plus-common experiment is a
**two-semantics sibling** relative to v15/schema 8: it would change both the
regression loss and the reconstruction.  It is not an equivalent successor,
not an alternate name for v16, and not authorized by this protocol.  Its lower
observed v14 RMSE and gradient norms cannot overcome the confounding introduced
by changing two semantics at once.

## Normative inheritance from v15

The frozen v15 protocol
`VOC_V15_HALF_SQUARED_Q_300K_ACCEPTANCE.md`, SHA-256
`cf5838f577ac55db2990669ebfd712c6cbbeee28e8b5388075833b69f4206b90`,
is incorporated verbatim except for these closed substitutions:

- experiment names and fixed profile change from v15 to v16 identities;
- gate-policy schema 8 changes to schema 9;
- the schema-9 online and EMA Q reconstruction changes exactly as specified;
  and
- schema-9 JSON-safe resolved identity adds the one derived reconstruction
  string while retaining the v15 derived loss string.

V15's exact half-squared selected-action TD objective remains unchanged:

```python
q_error = selected_q.float() - target.float()
q_loss_rows = 0.5 * q_error.square()
q_loss = torch.sum(q_loss_rows * q_train_valid.float())
```

The factor 0.5, FP32 loss operands, sign, elementwise square, unchanged
`q_train_valid`, multiplication, sum reduction, zero-support branch, logged
mean definition, outer `voc_loss_cost=1.0`, and held-out isolation are exact.
There is no return to Huber and no loss coefficient search.

All v13 Changes A-C and v14 Change D remain unchanged.  Training soft epsilon
is exactly 0.02, executed gate epsilon is exactly 0.25, main actor AMP initial
scale is exactly 32, EMA tau is exactly 0.1, and schedule total is exactly
100000000.  The strict actor-policy version barrier, exact five-key bundle and
ack shapes, exact seven-key publication history, W&B two-phase completion,
source-hardcoded no-Ray-retry topology, EMA-to-gate exact projection, and
schema-1 ModelBuffer input-seal mechanism continue as their schema-9
successors.

Every v15 network architecture, affine head, tensor shape, initialization,
optimizer and scheduler topology, parameter ownership, actor/ModelNet
precision rule, target, action mapping, state-value detach, policy-probability
detach, trajectory-generation rule, replay rule, learning-rate schedule,
gradient clip, scalar coefficient, telemetry field, artifact rule, population,
window, sufficient-statistic pooling rule, support floor, behavioral-accuracy
definition, sampled no-op/forced-action exclusion, and acceptance threshold
remains unchanged.  Default behavioral accuracy and no-op telemetry retain
their existing definitions and status; v16 invents no behavioral or no-op
gate.

Unchanged rules do not imply an unchanged learned path.  The new common-mode
gradient can alter online raw-head parameters and their coordinatewise Adam
moments.  Consequently, future raw action differences, EMA state, projected
gate, sampling distribution, realized trajectory, row cadence, supports, and
acceptance direction are not invariant and are not predicted by this
protocol.  Only the fixed-raw real-arithmetic reconstruction identities stated
below are algebraically invariant.

V16 retains the exact Enduro data and configuration, CUDA devices 0 and 1
only, Ray resources two GPUs and 16 CPUs, W&B-disabled wire, W&B-required
qualification/primary, and evaluator-private epsilon/barrier/seal-runtime
overrides.  Pong, Space Invaders, alternate Enduro seeds, early fixed
evaluation, and post-hoc diagnostics as selection inputs remain forbidden.

The only permitted implementation changes are those strictly necessary to
implement schema-9 reconstruction, propagate strict schema 9 through the
existing schema-8 mechanisms, add derived resolved identity, add dedicated
schema-9 validation/public/smoke/fixed routes, add `v16-300k`, and bind the
frozen tests.  A second lever is forbidden.  In particular, v16 may not change:

- half-squared loss, `voc_loss_cost`, learning rate, Adam parameters,
  scheduler, EMA tau, temperature, soft/execution epsilon, action weighting,
  or loss reduction;
- network width/depth, add a head, parameter, state, target, auxiliary loss,
  shared gradient, state-value gradient, normalization, or residual scale;
- replay, batch, unroll, warm-up, trajectory, held-out split, support,
  checkpoint, projection, barrier, seal/drain, logger, or terminal ordering;
  or
- optimizer step count, extra update, selected checkpoint, resume, retry,
  replacement seed, threshold, population, or fixed-evaluation rule.

## Exact schema lineage and common-mode rule

Schemas at most 7 retain byte-, shape-, return-, and behavior-identical
historical beta-1 Smooth-L1 plus the old policy-centered reconstruction.  No
new derived identity appears in their resolved records.

Schema 8 retains byte-, shape-, return-, and behavior-identical v15
half-squared TD plus the old policy-centered reconstruction:

```python
centered = A - torch.sum(p_det * A, dim=-1, keepdim=True)
Q = V_det.unsqueeze(-1) + centered
```

Its resolved identity retains exactly the v15-only derived
`voc_q_regression_loss="half_squared_td"` and does not acquire the schema-9
reconstruction field.

Schema 9 retains the schema-8 half-squared loss byte-for-byte and changes only
both online and EMA reconstruction:

```python
raw = A
common = raw.mean(dim=-1, keepdim=True)
centered = raw - torch.sum(p_det * raw, dim=-1, keepdim=True)
Q = V_det.unsqueeze(-1) + common + centered
```

The action axis has exact size two.  `common` is not detached; it is the mean
of the existing raw online head when training online Q.  `p_det` and `V_det`
remain detached exactly as in v15.  EMA reconstruction uses the same formula
inside the existing no-gradient FP32 EMA inference path.  There is no
additional averaging outside this formula.

For selected action `j` and raw action `k`, the schema-9 reconstruction
Jacobian is exactly

```text
d Q_j / d A_k = 1/2 + 1[j=k] - p_det[k].
```

The sum over the two raw actions is one.  Under schema 8 it is zero because
the `1/2` term is absent.  Thus the unchanged half-squared row error supplies
the equal-row direction under schema 9.  No gradient reaches `p_det`, `V_det`,
target, held-out rows, actor policy, dedicated projected gate, or ModelNet.

In exact real arithmetic, for every normalized `p_det`:

```text
Q_C - Q_S = A_C - A_S
Q(A + c*[1,1]) = Q(A) + c*[1,1].
```

These are algebraic identities, not an assertion that arbitrary finite FP32
additions preserve prior bytes.  Tests may require exact equality in
high-precision algebra and for deliberately rounding-safe representable FP32
values.  They also bind the nonclaim with a large-shift counterexample: for
stored FP32 `A=[1,0]` and `c=2**24`, forming `A+c` rounds both actions to the
same stored value, so the stored raw delta can move from 1 to 0.  V16 does not
require pre-shift and post-shift Q or projected-gate bytes to match after such
input rounding.

At zero raw head output, `common=0` and `centered=0`, so schema-9 Q is exactly
the same detached state value as schema 8.  Fresh zero initialization,
parameter keys, optimizer slots, and pre-update projection state therefore
remain unchanged.

The existing exact projected gate continues to derive its affine target from
the current stored EMA raw-head action-difference affine map, scaled by the
unchanged Q and policy temperatures.  Its runtime invariant is bit equality
with the target recomputed from that same current stored raw state, not byte
invariance before and after an arbitrary FP32 common addition.  Gate weight
and bias projection, empty gate optimizer state, pristine gate
scheduler/scaler, zero parent update count, and Q/EMA/projection lockstep
remain unchanged.

After one successful online-Q step, the existing transaction advances online
Q once, EMA once at tau 0.1, and exact projection once.  No-support, Q AMP
skip, non-finite loss or raw gradient, or optimizer failure advances none of
EMA/projection.  A finite over-threshold gradient is clipped and may step.
Main actor AMP remains transactionally independent but any live skip remains
an acceptance failure.  No schema-9 branch may add an extra optimizer step.

## Exact schema-9 identity

V16 uses strict non-boolean built-in Python integer
`voc_gate_policy_schema_version=9`.  A Boolean, NumPy integer, float, string,
missing/defaulted value, 8, or 10 is not schema 9.  Gate schema 9 plus the
derived reconstruction identity is the sole new algorithm identity.  There
is no new persisted configuration key.

Configuration, actor checkpoint, and ModelNet checkpoint each retain the exact
229-key surface:

```text
229 = 209 v12 stage-neutral keys
    + 6 stage keys
    + 4 path-derived keys
    + 10 v13/v14 protocol keys.
```

The six stage keys remain `xpid`, `base_seed`, `total_steps`,
`model_warm_up_n`, `actor_unroll_len`, and `use_wandb`.  The four path keys
remain `savedir`, `ckpdir`, `cmd`, and `icopro_data_path`.  The exact ten
protocol keys remain:

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

Only the existing gate-schema value becomes 9.  Execution epsilon remains
0.25; barrier true; bundle schema 1; timeout 120.0; Ray actor restart and task
retry zero; actor AMP initial scale 32; training barrier runtime true; and
model-input-seal schema 1.  Full keyset is exactly 229.  Missing, extra,
coerced, non-finite, wrongly typed, or defaulted values fail closed.

The 209-key v12 projection remains byte-identical with SHA-256
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.
Configuration, actor metadata, and ModelNet metadata must agree exactly on all
229 keys and the canonical complete-surface digest.  Each stage's complete
digest is resolved from its exact v16 paths and command before launch and may
not be guessed in this document.

Every actor-policy bundle retains exact keys
`{bundle_schema_version, policy_version, terminal, gate_schema,
actor_state_dict}`, bundle schema 1, and gate schema 9.  Every ack retains exact
keys `{bundle_schema_version, gate_schema, rank, policy_version, terminal}` and
gate schema 9.  Publication-history events remain exact seven-key mappings
`{predecessor_version, policy_version, publication_count, terminal, ack_ranks,
expected_ack_count, state_sha256}`.  History does not add gate schema or either
derived identity.

Model-input-seal schema remains strict integer 1.  Terminal ModelNet evidence
retains the exact ten fields:

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

All exact built-in types, seal count 1, final-real equality, terminal progress,
drain zero/one branches, pre/final m/p relations, finiteness, late-write zero,
abort zero, durable save, complete-success, actor-before-model-before-finish,
and no-post-terminal-action relations remain unchanged.  ModelBuffer's exact
13-key runtime status, claim token linearization, independent 120-second RPC
bounds, seal denial of later claims/writes, and abort behavior are unchanged.

Schema-9 training finalization must use a dedicated
`validate_schema9_final_bundle` before ModelBuffer complete-success.  Public
completion must use a dedicated `validate_schema9_completed_bundle`.
Actor-only terminal validation must use its dedicated schema-9 route.  Shared
dispatch may recognize schemas 6, 7, 8, and 9, but a dedicated schema-9 route
strictly rejects every non-9 value.  Schemas at most 8 never call a dedicated
schema-9 validator and retain their exact return shapes and behavior.

Schema-9 JSON-safe `resolved_identity` retains the schema-8 fields and exact
derived loss string, then adds exactly one derived reconstruction string.  Its
exact keys are:

```text
key_count
v12_projection_key_count
v12_projection_sha256
complete_surface_sha256
stage
paths
gate_schema
voc_gate_policy_schema_version
voc_model_input_seal_schema_version
voc_q_regression_loss
voc_q_reconstruction
```

The last two values must be exactly `"half_squared_td"` and
`"detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"`.
They are derived from validated strict schema 9.  Neither may appear among the
229 keys, embedded flags, CLI, YAML, actor/ModelNet checkpoint top level,
bundle, ack, or history.  Schema-8 resolved identity remains exact and does not
gain `voc_q_reconstruction`; schema-7 and earlier shapes remain unchanged.

Schema-9 public/smoke/fixed top-level JSON-safe record shapes remain the v15
shapes; only resolved schema/version/profile identity and the additional
derived reconstruction entry differ.  They remain non-tensor-bearing and
must preserve both derived strings exactly.  Schema-8 records and canonical
bytes remain unchanged.

## Immutable stage identities and no-retry rule

Every v16 stage is fresh: `ckp=false`; `preload`, `preload_actor`, and
`voc_parent_checkpoint` are empty; parent-update count is zero; actor policy
version starts at zero; and online Q, EMA Q, exact projected gate, ModelNet,
buffers, and seal state start from the unchanged fresh state.  No checkpoint,
optimizer, buffer, version, observation, or result crosses stages.

The only valid tuples
`(xpid, base_seed, total_steps, model_warm_up_n, actor_unroll_len, use_wandb)`
are exactly:

- (`enduro-voc-v16-commonmode-eps25-sentinel-wire1200`, 1, 1200, 512, 41,
  false);
- (`enduro-voc-v16-commonmode-eps25-seed1-qual-fresh-100k`, 1, 100000,
  10000, 201, true); and
- (`enduro-voc-v16-commonmode-eps25-seed5-strict-fresh-300k`, 5, 300000,
  10000, 201, true).

Each xpid is an exact built-in Python string.  Numeric members are strict
non-boolean built-in Python integers.  `use_wandb` is an exact built-in Python
Boolean.  No trimming, normalization, coercion, alias, or alternate tuple is
allowed.  Configuration, actor metadata, and ModelNet metadata must agree on
the tuple.  Normalized real `ckpdir` basename equals xpid; savedir, ckpdir,
command, and Enduro data path bind the same immutable v16 snapshot.

All stages retain `schedule_total_steps=100000000`, exact Enduro paths and
network configuration, CUDA `0,1` only, and Ray two GPUs/16 CPUs.  The wire is
W&B-disabled.  Qualification and primary require authenticated W&B and the
inherited strict request/ack/private-cleanup/public-finish sequence.

Every stage has exactly one attempt.  The wire cannot continue into
qualification.  Qualification is one separate fresh seed-1 run.  Primary is
one separate fresh seed-5 run and may start only after every qualification
gate passes.  There is no resume, preload, extension, fallback seed, retry,
duplicate xpid, selected checkpoint, replacement run, or cross-schema state.
Any identity, launch, artifact, mechanism, numeric, W&B, Ray, process, GPU, or
cleanup failure permanently ends v16 at that stage.

## Sequential release gates

The only release order is:

1. Implement only schema-9 reconstruction, propagation, derived identity,
   dedicated validation/profile, and the frozen tests.  Freeze all bytes and
   pass two independent code/contract audits.
2. Build a fresh inode-independent immutable snapshot from the authoritative
   v15 source/data baseline plus an exactly enumerated v16 overlay.  Pass two
   independent manifest, mode, binary, cp310-binding, schema, test, and
   posthash audits before launch.
3. Run exactly one fresh seed-1 1.2k wire.  Decide mechanics only; behavioral
   or reward evidence cannot influence implementation or wire clearance.
4. Only after the wire passes, run exactly one fresh seed-1 100k
   qualification with W&B and every inherited v15 gate unchanged.
5. Only after qualification passes every gate, run exactly one fresh seed-5
   300k primary with every inherited v15 gate unchanged.
6. Only after the primary passes may its terminal checkpoint receive one
   fixed confirmation under exact profile `v16-300k`.

No primary or fixed evaluation may be launched speculatively or after a
failed qualification.  Pong, Space Invaders, another seed, shortened run, or
diagnostic fixed evaluation remains out of scope until an accepted v16 Enduro
claim exists.

## Integrity-wire acceptance

The wire may inspect only immutable provenance/configuration, schema-9 branch
identity, both derived resolved strings, first/final checkpoints, actor
versions/acks/history, Q/EMA/reconstruction/projection transactions, AMP and
non-finite counters, ModelBuffer seal/drain ordering/evidence, W&B-disabled
logger completion, finish, manifests, and process/Ray/GPU cleanup.

It must exercise at least one supported half-squared Q update using the new
online reconstruction, one EMA update using the same reconstruction, exact
projection from unchanged raw EMA delta, one nonterminal actor publication,
the sole terminal publication/ack, correct seal and drain-zero-or-one branch,
durable model save, complete-success, and exact true ModelLearner/SelfPlay
returns.  Authoritative-validator and public JSON-safe evidence derived from
the checkpoints must carry schema 9 and both derived strings, while checkpoint
bytes and all persisted surfaces stay at their unchanged 229-key shape.

The wire supplies no qualifying behavioral row.  Any Q skip, actor/gate/model
AMP skip, non-finite, timeout, malformed bundle, history error, late write,
abort, retry, stale model checkpoint, missing finish, W&B artifact, source
drift, or incomplete cleanup permanently fails v16.  Negative paths need not
occur live but must be frozen in tests.

## Frozen 100k qualification

The v15 Frozen 100k decision, which incorporates the v14/v13/v12/v11/v9
algebra, is verbatim.  Canonical rows satisfy
`70000 < real_step <= 100000`; windows are `(70000,80000]`,
`(80000,90000]`, and `(90000,100000]`; overshoot is excluded.  Required cells
are finite, rows complete, real steps unique and strictly increasing, and
malformed, duplicate, or nonmonotone input fails closed.

Qualification passes only if every inherited gate passes together:

- teacher gap at least `0.075`, student gap at least `0.05`, retention at
  least `0.50`, and signed margin strictly positive;
- at least two of three windows each have positive student gap and positive
  signed margin;
- maximum consecutive negative trailing-five pooled gaps at most 3, with
  every positive/negative denominator valid and exact zero nonnegative;
- train and held-out CONTINUE and STOP fractions each strictly above `0.05`;
- wrong-CONTINUE saturation strictly below `0.01`, with wrong-STOP and the
  forced-stop diagnostic retaining their inherited status;
- online-versus-EMA non-tie sign agreement at least `0.60`;
- held-out EMA selected-action TD RMSE at most `0.5`; and
- actor, online-Q, gate, ModelNet, protocol, AMP-skip, and non-finite counters
  at their exact inherited zero requirements.

Schema 9, exact229/209, both derived identities, new online/EMA reconstruction,
unchanged half-squared loss, Q/EMA/projection transaction, barrier/history,
W&B completion, seal/exact-ten, finish, manifests, and cleanup are hard
integrity gates.  They add no numeric threshold.  Any failed qualification
permanently ends v16 and forbids primary and fixed evaluation.

## Frozen 300k primary acceptance

The v15 Frozen 300k primary decision and incorporated v13/v10 algebra are
verbatim.  Full remains `(100000,300000]`, late remains
`(250000,300000]`, and W1/W2/W3 remain `(270000,280000]`,
`(280000,290000]`, and `(290000,300000]`.  Overshoot is excluded.

Every inherited threshold remains unchanged, including learned soft-gate
probability `0.475/0.525`, sampled-control strength `0.525`, conditional
argmax accuracy `0.60`, useful-pair coverage `0.95`, sign agreement `0.60`,
strict support fractions above `0.05`, wrong-side saturation and forced-stop
rates below `0.01`, held-out RMSE at most `0.5` where inherited as a training
gate, exact direction/strength/window requirements, and zero AMP-skip or
non-finite safety events.  Absolute support floors and all four frozen
behaviors remain unchanged.

Soft behavior/calibration probabilities use training epsilon 0.02.  Sampled
execution, stored behavior likelihood, V-trace, and joint-policy entropy use
execution epsilon 0.25.  Default behavioral accuracy, sampled no-op, forced
action, support, saturation, and calibration accounting remain exactly the
inherited definitions.

All artifact, provenance, common-reconstruction, mechanism, behavior,
stability, support, trailing-five, saturation, forced-stop, calibration,
barrier, seal, AMP, and non-finite gates pass together.  There is no partial,
diagnostic-only, mechanism-only, or historical pass.

## Fixed-checkpoint confirmation

The closed fixed profile is exactly `v16-300k` and accepts only the one
accepted seed-5 primary tuple.  It rejects wire, qualification, v15, schema 8,
and every legacy profile before rollout or output.  Held-out seeds remain
20260827 through 20260842, exactly 16 streams by 6250 real steps and 100000
total, with calibration V-trace unroll 201 and all inherited behavioral and
calibration algebra.

After importing the checkpoint-bound public module and resolving the requested
profile, fixed evaluation must validate the complete schema-9 primary before
any evaluator-direct/downstream flag load, live spec/environment probe,
environment construction/reset/action, data access, tensor load, rollout, or
output.  The authoritative validator's own bound safe deserialization is part
of validation and excluded from downstream counters.  Prevalidated evidence
must be reused exactly, not trusted as a substitute for the dedicated
validator or reconstructed after a live probe.

Validation includes config/actor/model exact229 identity, 209 projection,
both derived strings, bundle/ack/history, actor/Q/EMA/new reconstruction/
projection optimizer/scheduler/scaler state, complete ModelNet state, exact-ten
seal relations, W&B completion, private-marker absence, public finish,
source/runtime binding, and exact primary tuple.  Schema 8 or a schema-9
checkpoint requested under any legacy/v15 profile fails before every
downstream call and output.

Only after immutable validation may an evaluator-private copy disable actor
training, ModelNet training, parallel execution, live barrier waiting, and
live ModelBuffer seal coordination.  It records immutable training epsilon
0.02, execution epsilon 0.25, schema 9, seal schema 1, and both derived
identities while using runtime soft/execution epsilon 0/0, barrier wait false,
and seal coordination false.  It never rewrites config or checkpoints.

Fixed B/calibration probability continues to use the recorded soft learned
gate field, not epsilon-zero execution likelihood.  Only an accepted primary
and accepted `v16-300k` confirmation can support a v16 Enduro claim.

## Frozen test and audit matrix

Before stable implementation or snapshot clearance, tests must cover at least
the following exact matrix.

### Reconstruction algebra, gradients, and gauge

- Direct online and EMA tests bind exact schema-9 `raw`, unweighted two-action
  mean with `keepdim=True`, detached-policy centering, detached value, and sum
  of common plus centered terms.
- Hand-computed asymmetric raw heads and nonuniform normalized detached
  policies bind both Q actions, selected Q, common, centered, and exact
  Jacobian `1/2 + 1[j=k] - p_det[k]`.
- Real/high-precision algebra and rounding-safe representable FP32 equal raw
  shifts move both schema-9 Q actions by the same shift and produce nonzero
  common-direction gradient; the same shifts cancel under schema 8.
- Those cases bind `Q_CONTINUE-Q_STOP=A_CONTINUE-A_STOP`; selected-action
  mapping remains CONTINUE index 0 and STOP index 1.  A large FP32 shift such
  as `A=[1,0]`, `c=2**24` binds that pre/post stored bytes and deltas are not
  claimed invariant after input rounding.
- Gradients reach only existing online raw-head parameters.  `p_det`,
  `V_det`, targets, actor policy, dedicated gate, ModelNet, and held-out rows
  remain isolated.
- Zero raw-head output gives exact schema-8/schema-9 pre-update Q parity,
  identical fresh tensor state, and no new key, buffer, optimizer slot, or
  initialization.

### Half-squared loss, masks, and precision

- Schema 9 reuses exact schema-8 FP32 `0.5 * error.square()`, unchanged
  `q_train_valid.float()` multiplication, sum reduction, factor 0.5, target,
  selected action, outer cost 1.0, and logged positive-support mean.
- Mixed train/held-out masks prove held-out value and gradient exactly zero;
  all-held-out/no-support performs no Q, EMA, or projection update.
- Positive, negative, zero, subunit, exact-unit, and tail errors bind values
  and gradients and prove schema-8/schema-9 loss byte parity for equal
  reconstructed selected Q.
- Squaring occurs after FP32 conversion.  Representable finite input/loss/raw
  gradient follows the exact path.  Overflow, NaN, Inf, or non-finite raw
  gradient fails before Q/EMA/projection.  Existing post-step/final recursive
  validation rejects non-finite parameter, optimizer, scaler, telemetry, or
  checkpoint state without inventing a new preflight scan.
- A finite over-threshold raw gradient is clipped and may step; explicit tests
  include the v15-style one-finite-clip branch.  AMP skip/non-finite/optimizer
  failure suppresses the transaction.  Existing raw norm, optimizer-stepped,
  scaler, skip, consecutive-skip, and non-finite-gradient-parameter evidence
  remains unchanged; no postclip counter or field is added.

### Online, EMA, projection, and transaction ordering

- Online and EMA reconstruct identical Q from byte-equal raw affine state,
  detached value, and detached probabilities; EMA remains no-gradient FP32.
- One successful Q step advances online Q, EMA tau-0.1 update, and exact
  projection once in order.  Counters are lockstep and projection post-error
  is zero.
- Exact projection continues to use the current stored EMA raw
  action-difference affine map and remains bit-equal to its target recomputed
  from that same state.  Rounding-safe representable common additions bind the
  real-algebra cancellation; arbitrary FP32 additions do not require pre/post
  target bytes to match.
- Gate optimizer state remains empty, scheduler/scaler pristine, parent
  update count zero, and no-support/Q-skip leaves EMA/gate unchanged.  Main
  actor skip independence and live hard-failure status remain unchanged.
- Periodic and terminal checkpoints preserve the existing raw
  online/EMA/gate tensor keysets and shapes plus optimizer/scheduler/scaler
  state.  They do not promise unchanged learned tensor values.  No common-mode
  checkpoint tensor or extra step exists.

### Schema, surfaces, artifacts, and legacy differential

- Strict built-in integer schema 9 activates seal schema 1 and the new
  reconstruction.  Bool, NumPy int, float, string, missing, 8, and 10 reject.
- Exact schema-9 config/actor/model surfaces remain 229; projection is exact
  209 with SHA
  `bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`;
  no 230th key or persisted loss/reconstruction identity is allowed.
- Schema-9 bundle and ack retain exact five-key shapes and gate 9; history
  remains exact seven-key shape with recomputed canonical digest; seal remains
  schema 1 with exact ten terminal fields and branch relations.
- `validate_schema9_final_bundle`, `validate_schema9_completed_bundle`, and
  the actor-terminal schema-9 route require both exact derived strings.
  Wrong/missing/extra identity, 8/10/bool/float/string schema, surface/path/
  stage drift, malformed tensor/metadata/history, logger/finish failure,
  private markers, and forged evidence all fail closed.
- Schema-9 `resolved_identity` has the exact 11-key shape frozen above.
  Schema 8 retains its exact v15 shape and only the loss identity.  Schemas at
  most 7 retain their exact historical shapes and neither field.
- Differential fixtures and immutable v13/v14/v15 artifacts prove schemas at
  most 7 preserve Huber plus old centering, schema 8 preserves half-squared
  plus old centering, and all schema<=8 config/checkpoint/public/smoke/fixed
  keysets, return shapes, behavior, and canonical bytes are unchanged.

### Public, smoke, fixed, ordering, and byte binding

- Public, smoke, and fixed schema-9 validation completes before any
  evaluator-direct/downstream `_load_flags`, live evaluation-spec/environment
  probe, reset/step, data access, direct `torch.load` or tensor use, rollout,
  output creation, or rewrite.  Validator-internal bound deserialization is
  allowed and excluded from downstream zero-call counters.
- The initially validated `config_c.yaml` payload is stable-read, SHA-bound to
  completion evidence, classified from those exact bytes, and consumed by a
  byte-aware loader that never reopens mutable checkpoint config.  Deletion,
  replacement, schema9-to-legacy, legacy-to-schema9, alternate explicit config,
  and probe-to-load swaps fail before downstream use or consume only the
  already bound bytes; final revalidation detects any artifact mutation before
  output.
- Smoke order is exact: immutable schema-9 prevalidation, stable byte binding,
  private runtime copy, evaluator-only overrides, authoritative postvalidation,
  exact pre/post evidence and hash equality, then environment.  Both derived
  strings persist in evidence; stored config/checkpoints remain unchanged.
- Fixed order is exact: bound public import and requested `v16-300k` profile,
  schema-9 dispatch/dedicated validation and evidence equality, then and only
  then flags/live probe/tensor/data/rollout/output.  A schema-9 checkpoint under
  any legacy/v15 profile and any legacy/schema-8 checkpoint under `v16-300k`
  yields zero downstream calls and zero output.
- `v16-300k` accepts only the exact primary stage.  Wire, qualification,
  schema 8/v15, legacy profiles, preload/resume, wrong paths, missing derived
  identity, private markers, or pre/post mutation reject before output.
- Schemas<=8 preserve historical pathname/byte-loader behavior and canonical
  records exactly; feature detection of checkpoint-bound historical public
  modules may not break frozen v13/v14/v15 evaluation.

### CLI, topology, lifecycle, and no retry

- Real `create_setting(save_flags=False)` tests bind all three exact v16
  xpids/tuples, exact229/209, schema 9, seal 1, schedule100M, path/command
  derivation, fresh empty inputs, resources, and W&B modes.
- Wrong xpid/whitespace, seed, total, warm-up, unroll, W&B, schema, path,
  surface, derived resolved identity, preload, resume, topology, retry, or
  resource setting fails before run directory/environment action.  Derived
  identity negatives live in validator JSON evidence, never CLI/YAML/config.
- Schema-9 training with `ckp=true`, preload, parent, resume, or cross-schema
  state is rejected before restoration.  Terminal checkpoint preservation is
  evidence, not resume authority.
- Real-Ray tests inherit actor barrier, claim/seal/drain zero/one, logger
  two-phase completion, timeout, abort, kill, no-restart/no-task-retry,
  private-marker cleanup, finish, and process/GPU cleanup matrices.
- Immutable snapshot tests require exact source/data manifests and coverage,
  independent inodes/modes, empty runs, snapshot-only cp310 import, no cp312 or
  worktree module, focused/adversarial/full suites, and unchanged posthash.

## Final claim boundary

V16 requires, in order, one accepted immutable snapshot, one accepted
mechanics-only seed-1 wire, one accepted fresh seed-1 100k qualification, one
accepted fresh seed-5 300k primary, and that primary's one accepted
`v16-300k` fixed confirmation.  Any failure is permanent for v16.  No later
mechanics pass, diagnostic, seed, checkpoint, sibling protocol, or version can
retroactively change the permanent v14 or v15 failures or any v16 stage
decision.
