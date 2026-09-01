# Enduro VoC-v17 Huber-common-Q preregistered acceptance

This protocol is frozen after the sole VoC-v16 seed-1 100k qualification
failed and before any v17 implementation edit, immutable snapshot, wire,
qualification, primary, or fixed evaluation. V14, v15, and v16 remain
permanent failures. V17 is the immediate successor of v16 and gate-policy
schema 9. It is one separately named, prospective, no-retry experiment that
changes exactly one algorithmic rule:

1. for gate-policy schema 10 only, retain schema 9's exact online and EMA
   common-mode Q reconstruction and replace schema 9's half-squared
   selected-action TD loss with the exact beta-1 Smooth-L1 loss historically
   used by schemas at most 7.

The schema-10 reconstruction remains byte- and behavior-identical to schema 9:

```python
raw = A
common = raw.mean(dim=-1, keepdim=True)
centered = raw - torch.sum(p_det * raw, dim=-1, keepdim=True)
Q = V_det.unsqueeze(-1) + common + centered
```

`p_det`, `V_det`, their detaches, action mapping, targets, train/held-out mask,
online-to-EMA transaction, and exact gate projection are unchanged. Schema 10
changes no reconstruction operation, parameter, buffer, tensor key or shape,
optimizer, scheduler, scaler, clip, target, selected action, training example,
held-out gradient, telemetry field, checkpoint shape, persisted configuration
key, public/smoke/fixed record shape or ordering, execution rule, schedule,
population, threshold, barrier, logger, or ModelBuffer mechanism.

The schema-10 derived identities are exactly:

```text
voc_q_regression_loss="smooth_l1_beta1"
voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
```

Both strings are required in authoritative JSON-safe derived evidence,
including `resolved_identity` and the inherited actor-policy, public, smoke,
and fixed evidence locations that carry derived identity. They are absent from
CLI/YAML keys, persisted training configuration and its exact 229-key surface,
embedded checkpoint flags, and actor/ModelNet checkpoint, bundle, and tensor-
state surfaces; neither creates a 230th persisted field.

A v17 observation may not be used to tune a second lever, pick a replacement
seed, relax a denominator or metric gate, or authorize a retry. This document
makes no causal claim and no prediction or guarantee that restoring beta-1
Smooth-L1 will reduce gradient norms, preserve sign support, improve held-out
calibration, or pass qualification.

## Permanent v16 qualification failure

The only v16 qualification was
`enduro-voc-v16-commonmode-eps25-seed1-qual-fresh-100k`, launched once from
immutable snapshot `/tmp/di-voc-v16-commonmode-eps25-final-mZ6A2C`. It was
fresh, seed 1, W&B-enabled, schema 9, seal schema 1, and bound to the exact
229-key surface and unchanged 209-key projection. The driver exited zero.
Actor, schema-9 online-Q-to-EMA-to-exact-projection, actor-policy
barrier/history, sealed ModelNet input/drain, W&B two-phase completion, public
finish, manifests, process, Ray, and GPU validation all passed. Successful
mechanics cannot rescue either numeric failure.

The frozen actor log is `logs.csv`, SHA-256
`45da4532357f5f150fad15c47bf1cdbb5ec52738ff3a68678afb5dca73ed20d5`,
with 250 complete rows and 922 unique ordered columns. The canonical
qualification population contains exactly 52 complete rows under
`70000 < real_step <= 100000`, from 70032 through 99408. Terminal step 100432
is overshoot and is excluded.

V16 has exactly two CSV-observable inherited hard-gate failures.

### Failure 1: held-out EMA selected-action TD RMSE

The pooled value is:

```text
sqrt(sum(row_holdout_count * row_ema_selected_action_td_rmse^2) / 7535)
    = 0.643399622874774 > 0.5.
```

The corresponding held-out EMA SSE is `3119.21176798055`; the unchanged
threshold SSE is `7535 * 0.5^2 = 1883.75`; and the excess is
`1235.4617679805501`. At fixed support, reaching the ceiling would require an
observed pooled SSE/MSE reduction of exactly `0.3960814012895369`, or
39.60814012895369%.
That arithmetic is not an estimated treatment effect or a promised v17
reduction. The online-Q held-out companion is `0.6432134674`, within about
`0.0001862` of EMA. This descriptive near-equality does not prove that EMA
lag is irrelevant and does not authorize changing tau.

### Failure 2: trailing-five denominator validity

There are 48 eligible trailing-five endpoints. Forty-three have both sign
denominators, positive pooled student gaps, and no negative run. The last five
have zero positive-sign support and therefore undefined, invalid gaps:

| Endpoint step | Positive support | Negative support |
| ---: | ---: | ---: |
| 95344 | 0 | 5871 |
| 96368 | 0 | 5862 |
| 97392 | 0 | 5845 |
| 98416 | 0 | 5875 |
| 99408 | 0 | 5829 |

Undefined gaps are not zero, nonnegative, negative, removable, or replaceable.
The denominator-validity failure is distinct from the maximum-negative-run
gate, whose v16 value remains zero. V16 W3 acceptance sign support is exactly
`1/11722` positive/negative. Neither threshold relaxation nor post-hoc row or
window removal is permitted.

Every other CSV-observable inherited qualification gate passed:

- teacher gap `0.2776639021`, student gap `0.2721106393`, retention
  `0.9800000547`, and signed margin `0.2702639525`;
- all three frozen windows had positive student gap and signed margin;
- train CONTINUE/STOP support was `26897/25775`, and held-out
  CONTINUE/STOP support was `3858/3677`;
- online-versus-EMA non-tie delta-sign agreement was
  `47767/60206 = 0.7933926851`, above 0.60;
- EMA-Q versus exact-projected-gate sign agreement was `60206/60206`;
- wrong-CONTINUE and wrong-STOP saturation were zero, while the report-only
  forced-stop rate was `0.0042599229`; and
- every actor, online-Q, gate, and CSV-observable protocol AMP-skip,
  consecutive-skip, non-finite, mismatch, malformed-bundle, and
  barrier-timeout counter was zero.

V16 failure permanently forbids a v16 primary, `v16-300k` fixed evaluation,
resume, extension, replacement seed, or retry. V17 is not a retry or rescue of
v16 and may not load a v14, v15, or v16 checkpoint, optimizer, buffer, actor
version, observation, or runtime state.

## Frozen v16 failure artifacts and mechanics closure

The v16 qualification run directory is
`/tmp/di-voc-v16-commonmode-eps25-final-mZ6A2C/runs/enduro-voc-v16-commonmode-eps25-seed1-qual-fresh-100k`.
Its exact 14 regular-file SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `ckp_actor.tar` | `c05a2237b495ff36fe633446e8236a93d2e07da9038f01781ac3b5026e947b7d` |
| `ckp_actor.tar_step_480` | `3a9f8f9199726902bf789afe7910f61c89f437cb4099476bcb709ecf6a06873c` |
| `ckp_actor.tar_step_100432` | `ed9500695a774022f4587acf872441a308405324049cfae4b15cdb8dc252bcef` |
| `ckp_model.tar` | `d1ce735542c5dd4a89b4e8cdc550946b5417700a956468820a69e4d80ed9ea58` |
| `ckp_model.tar_step_10000` | `cda4aa134454faf05ee49545dd272235aa86aa69a2a2ad46b61b41ec76e183a4` |
| `ckp_model.tar_step_100432` | `d1ce735542c5dd4a89b4e8cdc550946b5417700a956468820a69e4d80ed9ea58` |
| `config_c.yaml` | `2c91aa82c4744a8f44320ec5cc95f797384c4467c17607420b1dc3a94d9fea6b` |
| `finish` | `1b5f3449f685f876a10bbad21bacad4c0cd28479d53bcb0197f18308e2cb286f` |
| `logs.csv` | `45da4532357f5f150fad15c47bf1cdbb5ec52738ff3a68678afb5dca73ed20d5` |
| `logs_model.csv` | `ef5e39aba9ad5b7c0c2b4bef34ba91bb6f3adf1bcdc940abe9915504b8b18225` |
| `meta.json` | `5d6b34c05115569ac40b0d9d2706e49f5b1f8f269b924b6ac0632542ad2e91e8` |
| `meta_model.json` | `35111313bfab901433e0bd9994e98ecbc90ece7eda564849102d0656c198fa06` |
| `out.log` | `a2b292e7432db6bb25735620e3631529f2de1cd740797fb788e35129d18cc155` |
| `out_model.log` | `47f1422ced5f557a4c2a0cd088aaafcf31591c570b2dc2228c631311cb0797fe` |

The canonical SHA-256 of C-sorted `sha256sum` lines using `./` paths is
`e1a1b76c559cd108593bfdaa13fbf51afe3ae2f75ce1bfb5645781db51839ce5`.
The sole launch runtime `/tmp/v16qual-G3LwWW` has launch-provenance SHA-256
`1b712c604387725ccdc4aa1dbdcb016f4b4cb5c9b0effa22f150e27daf035325`,
driver-log SHA-256
`b1ca5709bdf45d101ae60815ccb94bc06cb926a8d5cf432f582f23e574111ab4`,
and exit-file SHA-256
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
The exit file is exact `0` plus one LF.

The immutable v16 snapshot source manifest is
`f90b970c647b06884f1101a3b39d862eed97aec51d69013614b6b671f54112ea`
with 1065 entries. The data manifest is
`23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343`
with 11 entries. Terminal config/actor/model identity was exact schema 9,
229 keys, complete-surface SHA-256
`d01ce292668d8c7a5f406e1b0bdb5d58981a420d905cfa45e5aa2877ab0f7c41`,
and v12 projection SHA-256
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.

Actor terminal real step was 100432, policy version/publication count was 250,
history length was 251, and terminal acknowledgement was `1/1`. Actor-state
SHA-256 was
`72c2d0a03a412b01cb069ebcbffc9c428f0599110edccdf9a744a59886e5f02f`;
publication-history SHA-256 was
`e0f03da24c156f0b6a7a00a5d542e3bfc0f2700e01d8014d38f452ad87834f9f`.
Mismatch, malformed, timeout, AMP-skip, and non-finite counters were zero.

Model input sealed once at terminal processed/final real step 100432 and took
one fresh drain from pre-real 100240 and pre m/p gradient counts 744 to final
counts 745, with zero late writes and aborts. Authoritative/public schema-9,
W&B request/ack/private cleanup, finish, source/data, and process/GPU closure
all passed. These mechanics facts are integrity evidence, not numeric
acceptance.

## Frozen three-version RCA and 2x2 rationale

The frozen executable notebook is
`notebook/v14_v15_v16_qualification_comparison_rca.ipynb`, SHA-256
`d32ebb602bdf38f8392878f7e76b3e91d32528203565843429dad4d0ad3bc87a`,
406671 bytes. The frozen machine-readable report is
`notebook/v14_v15_v16_qualification_rca_report.json`, SHA-256
`2bcdf68840c8dac4fd0d6569c108a6c5fe0e679e92a4f0f6380f3b90ae999c10`,
66760 bytes and 1158 lines. Both artifacts must retain these exact audited
bytes for this preregistration to remain authoritative.

The notebook validates exact v14/v15/v16 actor/model CSV, protocol, launch,
source-manifest, and data-manifest hashes; identical ordered 922-column actor
and 52-column model schemas; complete finite rows; strict integer steps;
canonical populations; sufficient-statistic pooling; and the statements in
this section. It does not replace terminal-bundle, W&B, manifest, or cleanup
authority.

The observed single-seed 2x2 table is:

| Selected-action loss | Old centered/no-common reconstruction | Schema-9 common reconstruction |
| --- | ---: | ---: |
| beta-1 Smooth-L1 | v14: `0.5247954789453232` (N=7703) | v17: prospective and unobserved |
| half-squared TD | v15: `0.5637126651551874` (N=8154) | v16: `0.643399622874774` (N=7535) |

Entries are pooled held-out EMA selected-action TD RMSE. The rows are three
separately generated, policy-coupled, on-policy trajectories with different
row cadence, state/action support, and no paired states or counterfactual
actions. The table is not a randomized factorial design and does not identify
a main effect, interaction, treatment effect, or causal root cause.

V17 fills the one unobserved loss/reconstruction combination prospectively.
It is the immediate successor of v16 because it changes exactly the loss while
preserving schema-9 common reconstruction. Relative to v14 it changes the
reconstruction, and relative to v15 it differs in two semantics. Those
cross-version descriptions do not make v17 a retry, an equivalent successor
of v14, or a causal factorial completion.

The frozen window results are:

| Version | W1 EMA RMSE | W2 EMA RMSE | W3 EMA RMSE | Full EMA RMSE |
| --- | ---: | ---: | ---: | ---: |
| v14 | `0.3070468494` | `0.4805763254` | `0.8753410807` | `0.5247954789` |
| v15 | `0.6350000483` | `0.5262197893` | `0.5362964250` | `0.5637126652` |
| v16 | `0.4241221974` | `0.9513859414` | `0.8161048403` | `0.6433996229` |

V16's W1 support share is 61.33%; its late W2/W3 deterioration and five
undefined trailing denominators may not be hidden by pooled composition or
reweighted post hoc. V16 EMA train/held-out RMSE are
`0.5999551131/0.6433996229`; online values are
`0.5997760072/0.6432134674`. V15's corresponding EMA values are
`0.4891362534/0.5637126652`. These descriptive comparisons combine algorithm,
trajectory, cadence, support, and state distribution.

V16's pooled EMA bias is only `0.0120960360` against RMSE
`0.6433996229`. Its top ten actor rows carry `0.7408296127` of held-out EMA
SSE. These row-aggregate facts neither expose event-level residual quantiles
nor identify a global-bias or tail mechanism.

Across all actor CSV rows, logged raw VoC-Q norm exceeded the unchanged nominal
boundary 1608 on `0/204` v14 rows, `1/149` v15 rows, and `35/250` v16 rows.
Canonical counts were `0/63`, `0/58`, and `7/52`. V16 canonical raw-norm
median/p95/max were `428.2610778809/1842.8370300293/2687.0109863281`.
These are associations from different trajectories, not proof that the loss
or common reconstruction caused clipping or failure.

The A/B design consensus ranks, first, beta-1 Smooth-L1 plus the unchanged
schema-9 common reconstruction: it is parameter-free, changes one semantic
rule from v16, preserves represented-Q parameterization, and prospectively
fills the unobserved 2x2 cell. Second is an explicit orthonormal coordinate
parameterization of the existing raw head `A`,
`m=(A_CONTINUE+A_STOP)/sqrt(2)` and
`d=(A_CONTINUE-A_STOP)/sqrt(2)`, with separate Adam moments. The second option
would change raw-head parameterization and optimizer state and is not
authorized here. The ranking is a falsifiable experiment-ordering hypothesis,
not a prediction that v17 will pass.

## Sole schema-10 loss delta

Schema 10 uses the exact existing beta-1 Smooth-L1 selected-action TD
objective from schemas at most 7:

```python
selected_q_work = selected_q.float()
target_work = target.float()
q_loss_rows = F.smooth_l1_loss(selected_q_work, target_work, reduction="none")
q_loss = torch.sum(q_loss_rows * q_train_valid.float())
```

This preserves the exact historical callable (`F` is
`torch.nn.functional`), FP32 operands, `reduction="none"`, and call shape. The
omitted beta uses the inherited PyTorch default 1.0. Schema 10 may not
substitute another beta or a merely similar loss.

The normative per-row real-arithmetic value is:

```text
L(e) = 0.5 * e^2          when |e| <= 1
     = |e| - 0.5          when |e| > 1,
where e = selected_q.float() - target.float().
```

Its selected-Q derivative is `e` inside the unit region and `sign(e)` in the
tails, with the common boundary value/derivative at `|e|=1`. Relative to
schema 9's `0.5 * e^2`, value and derivative are identical for `|e|<=1`; only
the tail rule changes from quadratic/unbounded `e` to linear/bounded
`sign(e)`.

The FP32 operands, subtraction sign, target, selected action, action mapping,
unchanged `q_train_valid.float()` multiplication, sum reduction,
zero-positive-support branch, logged positive-support mean, outer
`voc_loss_cost=1.0`, and held-out isolation are exact. No coefficient, beta,
mask, reduction, or target search is allowed. Under schema 10,
`actor/voc_q_loss` means the mean beta-1 Smooth-L1 selected-action training TD
loss on positive `q_train_valid` support. The held-out EMA selected-action TD
RMSE formula, support, population, and ceiling remain unchanged.

For each finite selected-Q row, the raw derivative with respect to the
selected Q is bounded in magnitude by one. This does **not** bound the
aggregate parameter-gradient norm: the loss is summed over supported rows,
the Q Jacobian and activations can amplify parameter gradients, and rows can
align. V17 does not promise that raw norm stays below 1608 or that clipping
never occurs. A finite over-threshold aggregate raw gradient is clipped and
may step exactly as before; skip and non-finite events remain hard failures.

Subtraction, absolute value, square in the unit branch, and reduction occur in
FP32. Representable finite loss/raw gradient follows the inherited transaction.
Overflow, NaN, Inf, or non-finite raw gradient fails before Q/EMA/projection.
Existing post-step/final validation rejects non-finite parameter, optimizer,
scaler, telemetry, or checkpoint state without inventing a new pre-step scan.

## Normative inheritance from v16

The frozen v16 protocol
`VOC_V16_COMMON_MODE_Q_300K_ACCEPTANCE.md`, SHA-256
`1814d68f667748358746c42f4578e17acbb950e9e0adac04a8d78c9352fdb84c`,
is incorporated verbatim except for these closed substitutions:

- experiment names and fixed profile change from v16 to v17 identities;
- gate-policy schema 9 changes to schema 10;
- schema-10 selected-action Q training loss changes exactly from
  half-squared TD to beta-1 Smooth-L1; and
- schema-10 derived loss identity changes from `"half_squared_td"` to
  `"smooth_l1_beta1"`.

Schema 9's online and EMA common reconstruction remains unchanged. All v13
Changes A-C, v14 Change D, v15 half-squared lineage evidence, and v16
common-reconstruction mechanics remain historical context; only the active
schema-10 loss is restored to beta-1 Smooth-L1.

Training soft epsilon is exactly 0.02, executed gate epsilon exactly 0.25,
main actor AMP initial scale exactly 32, EMA tau exactly 0.1, and schedule
total exactly 100000000. The strict actor-policy version barrier, exact
five-key bundle and ack shapes, exact seven-key publication history, W&B
two-phase completion, source-hardcoded no-Ray-retry topology,
EMA-to-gate exact projection, and schema-1 ModelBuffer input-seal mechanism
continue unchanged.

Every v16 network architecture, common reconstruction, affine head, tensor
shape, initialization, optimizer/scheduler/scaler topology, parameter
ownership, actor/ModelNet precision rule, target, action mapping, value and
policy detaches, trajectory/replay rule, learning rate, gradient clip,
telemetry field, artifact rule, population, window, pooling rule, support
floor, behavioral-accuracy definition, sampled no-op/forced-action exclusion,
and acceptance threshold remains unchanged. V17 adds no behavioral or no-op
gate.

Unchanged rules do not imply an unchanged learned path. Changing loss values
and tail gradients can alter raw-head parameters, coordinatewise Adam moments,
EMA state, projected gate, sampling distribution, realized trajectory, row
cadence, support, and acceptance direction. None is invariant or predicted.

V17 retains exact Enduro data/configuration, CUDA devices 0 and 1 only, Ray
resources two GPUs and 16 CPUs, W&B-disabled wire, W&B-required
qualification/primary, and evaluator-private epsilon/barrier/seal-runtime
overrides. Pong, Space Invaders, alternate Enduro seeds, early fixed
evaluation, and post-hoc diagnostics as selection inputs remain forbidden.

The only permitted implementation changes are those strictly necessary to
route schema 10 to the already existing beta-1 Smooth-L1 loss, propagate
strict schema 10 through unchanged schema-9 mechanisms, change derived loss
identity, add dedicated schema-10 validation/public/smoke/fixed routes and
`v17-300k`, and bind frozen tests. No second lever may change reconstruction,
loss beta/coefficient/reduction/mask, learning rate, Adam parameters,
scheduler, EMA tau, temperature, epsilon, action weighting, architecture,
parameterization, normalization, shared gradient, auxiliary loss, clip,
telemetry, replay, batch, unroll, warm-up, held-out split, projection,
barrier, seal/drain, logger, terminal ordering, optimizer step count, retry,
seed, checkpoint selection, threshold, population, or fixed rule.

## Exact schema lineage

Schemas at most 7 retain byte-, shape-, return-, and behavior-identical
historical beta-1 Smooth-L1 plus the old policy-centered/no-common
reconstruction. They retain no derived loss or reconstruction identity.

Schema 8 retains byte-, shape-, return-, and behavior-identical v15
half-squared TD plus the old policy-centered/no-common reconstruction. Its
resolved identity retains exactly
`voc_q_regression_loss="half_squared_td"` and no reconstruction field.

Schema 9 retains byte-, shape-, return-, and behavior-identical v16
half-squared TD plus common reconstruction. Its resolved identity retains
exactly:

```text
voc_q_regression_loss="half_squared_td"
voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
```

Schema 10 retains schema 9's common reconstruction byte-for-byte and changes
only the selected-action loss and derived loss identity:

```text
voc_q_regression_loss="smooth_l1_beta1"
voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
```

All schemas at most 9 must remain exact historical branches. A schema-10
implementation may not refactor them into a shared route that changes their
source-visible behavior, values, keysets, return shapes, error order,
validation order, public bytes, or fixed outputs.

### Reconstruction and transaction invariants

Schema 10's action axis remains exact size two. `common` is the unweighted
raw-head mean with `keepdim=True` and is not detached. `centered` uses the
same normalized detached policy probabilities; the state value remains
detached. EMA uses the identical formula inside the inherited no-gradient
FP32 path.

For selected action `j` and raw action `k`, the real-arithmetic Jacobian
remains exactly:

```text
d Q_j / d A_k = 1/2 + 1[j=k] - p_det[k].
```

The sum over two raw actions is one. No gradient reaches `p_det`, `V_det`,
target, held-out rows, actor policy, dedicated gate, or ModelNet. In exact real
arithmetic:

```text
Q_C - Q_S = A_C - A_S
Q(A + c*[1,1]) = Q(A) + c*[1,1].
```

These are algebraic identities, not a claim that arbitrary finite FP32 common
additions preserve prior bytes. For stored FP32 `A=[1,0]`, adding `c=2**24`
rounds both rows to the same stored value and can move the stored delta from 1
to 0. Runtime exact projection remains bit-equal to its target recomputed from
the same current stored EMA raw state; it is not pre/post-shift invariant.

At zero raw-head output, schema-9/schema-10 reconstructed Q, parameter keys,
optimizer slots, pre-update projection state, and fresh initialization remain
exactly equal. No common-mode checkpoint tensor or extra optimizer step exists.

After one successful online-Q step, the existing transaction advances online
Q once, EMA once at tau 0.1, and exact projection once. No-support, Q AMP
skip, non-finite loss/raw gradient, or optimizer failure advances none of
EMA/projection. Finite clipping may precede a successful step. Main actor AMP
remains transactionally independent, but any live skip remains an acceptance
failure.

## Exact schema-10 identity and surfaces

V17 uses strict non-boolean built-in Python integer
`voc_gate_policy_schema_version=10`. Boolean, NumPy integer, float, string,
missing/defaulted value, 9, or 11 is not schema 10. Gate schema 10 plus the
derived loss value is the sole new algorithm identity. There is no new
persisted configuration key.

Configuration, actor checkpoint, and ModelNet checkpoint each retain exact
229-key surfaces:

```text
229 = 209 v12 stage-neutral keys
    + 6 stage keys
    + 4 path-derived keys
    + 10 v13/v14 protocol keys.
```

The six stage keys remain `xpid`, `base_seed`, `total_steps`,
`model_warm_up_n`, `actor_unroll_len`, and `use_wandb`. The four path keys
remain `savedir`, `ckpdir`, `cmd`, and `icopro_data_path`. The exact ten
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

Only the existing gate-schema value becomes 10. Execution epsilon remains
0.25; barrier true; bundle schema 1; timeout 120.0; Ray actor restart/task
retry zero; actor AMP initial scale 32; training barrier runtime true; and
model-input-seal schema 1. Missing, extra, coerced, non-finite, wrongly typed,
or defaulted values fail closed.

The 209-key v12 projection remains byte-identical with SHA-256
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.
Config, actor metadata, and ModelNet metadata must agree on all 229 keys and
the canonical complete-surface digest. Each stage's complete digest is
resolved from exact v17 paths and command before launch and may not be guessed
in this document.

Every actor-policy bundle retains exact keys
`{bundle_schema_version, policy_version, terminal, gate_schema,
actor_state_dict}`, bundle schema 1, and gate schema 10. Every ack retains exact
keys `{bundle_schema_version, gate_schema, rank, policy_version, terminal}` and
gate schema 10. Publication history remains exact seven-key events
`{predecessor_version, policy_version, publication_count, terminal, ack_ranks,
expected_ack_count, state_sha256}` and adds neither gate schema nor derived
identity.

Model-input-seal schema remains strict integer 1. Terminal ModelNet evidence
retains exact ten fields:

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

All strict types, seal count one, final-real equality, drain zero/one branches,
pre/final m/p relations, durable save, complete-success,
actor-before-model-before-finish, late-write zero, abort zero, and
no-post-terminal-action relations remain unchanged. ModelBuffer exact13
runtime status, claim-token linearization, independent 120-second RPC bounds,
seal denial of later claims/writes, and abort behavior are unchanged.

Schema-10 training finalization must use dedicated
`validate_schema10_final_bundle` before ModelBuffer complete-success. Public
completion must use dedicated `validate_schema10_completed_bundle`, and
actor-only terminal validation must use its dedicated schema-10 route. Shared
dispatch may recognize schemas 6 through 10, but the dedicated schema-10 route
strictly rejects every non-10 value. Schemas at most 9 never call it and retain
their exact return shapes and behavior.

Schema-10 JSON-safe `resolved_identity` has the same exact 11-key shape as
schema 9:

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

The last two values are exactly `"smooth_l1_beta1"` and
`"detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"`. They are
derived from validated strict schema 10 and are required throughout the
authoritative JSON-safe derived-evidence locations specified above. They
remain absent from persisted training configuration, its exact 229-key
surface, and actor/ModelNet checkpoint, bundle, and tensor-state surfaces.
Schema-9 resolved identity retains its exact keyset and old loss value; schema
8 and schemas at most 7 retain their historical shapes.

Schema-10 public/smoke/fixed top-level JSON-safe record shapes remain schema
9's shapes. Only schema/version/stage/profile identity and derived loss value
differ. Records remain non-tensor-bearing and preserve both exact strings.
Schemas at most 9 and their canonical bytes remain unchanged.

## Immutable stages and no-retry rule

Every v17 stage is fresh: `ckp=false`; `preload`, `preload_actor`, and
`voc_parent_checkpoint` are empty; parent-update count is zero; actor policy
version starts at zero; and online Q, EMA Q, exact projected gate, ModelNet,
buffers, and seal state start from unchanged fresh state. No state crosses
stages.

The only valid tuples
`(xpid, base_seed, total_steps, model_warm_up_n, actor_unroll_len, use_wandb)`
are exactly:

- (`enduro-voc-v17-huber-common-eps25-sentinel-wire1200`, 1, 1200, 512, 41,
  false);
- (`enduro-voc-v17-huber-common-eps25-seed1-qual-fresh-100k`, 1, 100000,
  10000, 201, true); and
- (`enduro-voc-v17-huber-common-eps25-seed5-strict-fresh-300k`, 5, 300000,
  10000, 201, true).

Each xpid is an exact built-in string; numeric members are strict non-boolean
built-in integers; `use_wandb` is an exact built-in Boolean. No trimming,
coercion, alias, or alternate tuple is allowed. Config, actor metadata, and
ModelNet metadata must agree. Normalized real ckpdir basename equals xpid and
all paths/command bind the same immutable v17 snapshot.

All stages retain `schedule_total_steps=100000000`, exact Enduro paths and
network configuration, CUDA `0,1` only, and Ray two GPUs/16 CPUs. Wire is
W&B-disabled; qualification/primary require authenticated W&B and the
inherited request/ack/private-cleanup/public-finish sequence.

Every stage has exactly one attempt. Wire cannot continue into qualification.
Qualification is one separate fresh seed-1 run. Primary is one separate fresh
seed-5 run and may start only after every qualification gate passes. There is
no resume, preload, extension, fallback seed, retry, duplicate xpid, selected
checkpoint, replacement run, or cross-schema state. Any failure permanently
ends v17 at that stage.

## Sequential release gates

The only release order is:

1. Implement only schema-10 loss routing, strict propagation, derived identity,
   dedicated validation/profile, and frozen tests. Freeze all bytes and pass
   two independent code/contract audits.
2. Build a fresh inode-independent immutable snapshot from the authoritative
   v16 source/data baseline plus an exactly enumerated v17 overlay. Pass two
   independent manifest, mode, cp310, schema, test, and posthash audits.
3. Run exactly one fresh seed-1 1.2k wire. Decide mechanics only.
4. Only after wire passes, run exactly one fresh seed-1 100k qualification
   with every inherited gate unchanged.
5. Only after qualification passes every gate, run exactly one fresh seed-5
   300k primary with every inherited gate unchanged.
6. Only after primary passes may its terminal checkpoint receive one fixed
   confirmation under exact profile `v17-300k`.

No primary/fixed evaluation may launch speculatively or after failed
qualification. Pong, Space Invaders, alternate seed, shortened run, or
diagnostic fixed evaluation remains out of scope until an accepted v17 Enduro
claim exists.

## Integrity-wire acceptance

Wire may inspect only immutable provenance/config, schema-10 branch identity,
both derived strings, first/final checkpoints, actor versions/acks/history,
Huber/common reconstruction/Q/EMA/projection transactions, AMP/non-finite
counters, ModelBuffer seal/drain ordering, W&B-disabled logger completion,
finish, manifests, and process/Ray/GPU cleanup.

It must exercise at least one supported beta-1 Smooth-L1 Q update using
unchanged common reconstruction, one EMA update using identical reconstruction,
exact projection from unchanged raw EMA delta, one nonterminal actor
publication, terminal publication/ack, correct seal/drain-zero-or-one branch,
durable model save, complete-success, and exact-true worker returns. Validator
and public evidence must carry schema 10 and both strings while checkpoint
bytes/persisted surfaces remain exact229.

Wire supplies no qualifying behavioral row. Any skip, non-finite, timeout,
malformed/history error, late write, abort, retry, stale checkpoint, missing
finish, W&B artifact, source drift, or incomplete cleanup permanently fails
v17. Negative paths need not occur live but must be frozen in tests.

## Frozen 100k qualification

The v16 Frozen 100k decision, incorporating prior algebra, remains verbatim.
Canonical rows satisfy `70000 < real_step <= 100000`; windows are
`(70000,80000]`, `(80000,90000]`, and `(90000,100000]`; overshoot is
excluded. Required cells are finite, rows complete, steps unique/strictly
increasing, and malformed, duplicate, or nonmonotone input fails closed.

Qualification passes only if every inherited gate passes together:

- teacher gap at least `0.075`, student gap at least `0.05`, retention at
  least `0.50`, and signed margin strictly positive;
- at least two of three windows each have positive student gap and margin;
- every trailing-five endpoint has positive and negative denominators, and
  maximum consecutive negative trailing-five pooled gaps is at most 3;
- train and held-out CONTINUE/STOP fractions each strictly above `0.05`;
- wrong-CONTINUE saturation strictly below `0.01`, with wrong-STOP and
  forced-stop diagnostic retaining inherited status;
- online-versus-EMA non-tie sign agreement at least `0.60`;
- held-out EMA selected-action TD RMSE at most `0.5`; and
- actor, online-Q, gate, ModelNet, protocol, AMP-skip, and non-finite counters
  at inherited zero requirements.

Schema10/exact229/209, both identities, exact beta-1 Smooth-L1, unchanged
common reconstruction/Q-to-EMA-to-projection, barrier/history, W&B, seal
exact-ten, finish, manifests, and cleanup are hard integrity gates without new
numeric thresholds. Any failed qualification permanently forbids primary and
fixed evaluation.

## Frozen 300k primary acceptance

The v16 Frozen 300k primary decision remains verbatim. Full is
`(100000,300000]`, late is `(250000,300000]`, and W1/W2/W3 are
`(270000,280000]`, `(280000,290000]`, and `(290000,300000]`. Overshoot is
excluded.

Every threshold remains unchanged, including learned soft-gate probability
`0.475/0.525`, sampled-control strength `0.525`, conditional argmax accuracy
`0.60`, useful-pair coverage `0.95`, sign agreement `0.60`, strict support
fractions above `0.05`, wrong-side saturation and forced-stop rates below
`0.01`, held-out RMSE at most `0.5` where inherited as a training gate, exact
direction/strength/window requirements, absolute supports, four frozen
behaviors, and zero AMP-skip/non-finite events.

Soft behavior/calibration uses training epsilon 0.02. Sampled execution,
stored likelihood, V-trace, and joint entropy use execution epsilon 0.25.
Default behavioral accuracy, sampled no-op, forced action, support,
saturation, and calibration accounting retain inherited definitions.

All artifact, provenance, Huber/common mechanism, behavior, stability,
support, denominator, trailing-five, saturation, forced-stop, calibration,
barrier, seal, AMP, and non-finite gates pass together. There is no partial,
diagnostic-only, mechanism-only, or historical pass.

## Public, smoke, and fixed confirmation

The closed fixed profile is exactly `v17-300k` and accepts only the one
accepted seed-5 primary tuple. It rejects wire, qualification, v16/schema9,
and every legacy profile before rollout/output. Held-out seeds remain
20260827 through 20260842, exactly 16 streams by 6250 real steps and 100000
total, with calibration V-trace unroll 201 and inherited algebra.

After importing checkpoint-bound public code and resolving requested profile,
fixed evaluation must validate complete schema-10 primary before any
evaluator-direct/downstream flag load, live spec/environment probe,
construction/reset/action, data access, direct tensor load/use, rollout, or
output. Validator-internal bound deserialization is allowed and excluded from
downstream counters. Prevalidated evidence is reused exactly but never trusted
as a substitute for the dedicated validator.

Initial `config_c.yaml` bytes are stable-read, SHA-bound to completion
evidence, classified from those bytes, and consumed by the inherited
byte-aware loader without reopening mutable checkpoint config. Deletion,
replacement, schema10-to-legacy, legacy-to-schema10, alternate explicit
config, and probe-to-load swaps fail before downstream use or consume only
bound bytes; final revalidation catches artifact mutation before output.

Smoke order remains exact: immutable schema-10 prevalidation, stable byte
binding, private runtime copy, evaluator-only overrides, authoritative
postvalidation, exact pre/post evidence/hash equality, then environment.
Fixed order remains bound public import/requested profile, schema-10
dispatch/dedicated validation/evidence equality, then downstream use.

Validation covers exact229/209, both derived identities, bundle/ack/history,
actor/Huber/common/Q/EMA/projection state, ModelNet state, seal exact-ten,
W&B completion, private-marker absence, public finish, source/runtime binding,
and primary tuple. Schema 9 under `v17-300k` or another incompatible v17
profile, and schema 10 under any legacy or `v16-300k` profile, yields zero
downstream calls/output. Historical schema 9 remains byte-compatible on its
unchanged v16 route; this compatibility does not authorize evaluation, and
v16 remains permanently failed.

Only after validation may an evaluator-private copy disable actor and
ModelNet training, parallel execution, live barrier waiting, and live seal
coordination. It records immutable epsilon 0.02/0.25, schema 10, seal 1, and
both identities while using runtime epsilon 0/0, barrier wait false, and seal
coordination false. Stored config/checkpoints are never rewritten. Fixed
B/calibration probability continues to use recorded learned-gate fields, not
epsilon-zero execution likelihood.

## Frozen test and audit matrix

Before implementation/snapshot clearance, tests cover at least this matrix.

### Beta-1 Smooth-L1 values, gradients, masks, and precision

- Direct values and gradients for errors `0`, positive/negative subunit,
  `+/-1`, and positive/negative tails bind the exact beta-1 formula. Examples
  include values `0`, `0.125`, `0.5`, and `1.5` for absolute errors
  `0`, `0.5`, `1`, and `2`, with derivatives `0`, signed `0.5`, signed `1`,
  and signed `1`.
- Schema 10 matches schema 9's half-squared value/gradient exactly for
  rounding-safe `|e|<=1` and differs only in tails. Tests bind no hidden
  coefficient, beta, mean reduction, action weight, or mask change.
- FP32 conversion precedes subtraction/loss. Mixed train/held-out masks prove
  held-out value/gradient zero; all-held-out/no-support performs no Q/EMA/
  projection update and emits inherited zero-support evidence.
- Representable finite values follow the transaction. Overflow, NaN, Inf, or
  non-finite raw gradient fails closed. Existing post-step/final recursive
  validation remains unchanged.
- A finite over-threshold aggregate raw gradient is clipped and may step;
  tests explicitly reject any claim that bounded per-example Huber slope
  guarantees no clipping. Existing raw norm, optimizer-stepped, scaler, skip,
  consecutive-skip, and non-finite-gradient-parameter evidence is unchanged;
  no postclip counter is added.

### Reconstruction, isolation, EMA, and projection

- Online and EMA schema-10 reconstruction is byte-identical to schema 9:
  unweighted two-action mean, `keepdim=True`, policy-centered raw head,
  detached policy/value, and sum of common plus centered terms.
- Hand-computed asymmetric heads and nonuniform policy bind Q actions,
  selected Q, common, centered, and Jacobian
  `1/2 + 1[j=k] - p_det[k]`.
- Gradients reach only existing online raw-head parameters. Policy/value,
  target, actor, dedicated gate, ModelNet, and held-out rows remain isolated.
- Zero raw output gives exact schema9/10 pre-update parity and no new key,
  state, optimizer slot, or initialization.
- One successful Q step advances Q, EMA tau-0.1, and exact projection once in
  order. Counters are lockstep; projection post-error zero. No-support/Q-skip
  advances none; actor-skip independence remains.
- Projection uses current stored EMA raw delta and is bit-equal to its target
  recomputed from that state. Real-arithmetic common-shift identities are
  tested separately from FP32 large-shift rounding counterexamples.

### Schema, artifacts, and legacy differential

- Strict built-in integer schema 10 activates seal1, common reconstruction,
  and beta-1 Smooth-L1. Bool, NumPy int, float, string, missing, 9, and 11
  reject.
- Exact schema-10 config/actor/model surfaces remain229; projection exact209
  SHA `bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`;
  no 230th or persisted identity key.
- Bundle/ack remain exact five keys with gate10; history exact seven with
  canonical digest; seal exact10 branch relations unchanged.
- Dedicated schema-10 final/completed/actor validators require exact loss and
  reconstruction strings. Wrong/missing/extra identity, schema, surface,
  path/stage, tensor/metadata/history, logger/finish/private marker, or forged
  evidence fails closed.
- Resolved identity is exact11. Schema9 retains exact11 with half-squared
  loss; schema8 and <=7 shapes remain historical.
- Differential fixtures prove schemas<=7 Huber/no-common, schema8
  half-squared/no-common, schema9 half-squared/common, and schema10
  Huber/common, while every schema<=9 byte/shape/behavior/output remains
  unchanged.

### Public/smoke/fixed ordering and byte binding

- Schema-10 validation precedes downstream `_load_flags`, live spec/env probe,
  reset/step, data, direct tensor load/use, rollout, output, or rewrite.
  Validator-internal safe loads are excluded from counters.
- Stable config payload/hash classification and byte-aware loading resist
  deletion, replacement, cross-schema, alternate-config, and TOCTOU swaps.
- Smoke directly binds prevalidation -> private copy -> postvalidation/evidence
  equality -> environment. Fixed binds public import/profile -> dedicated
  schema10 validation/equality -> downstream use.
- `v17-300k` accepts only exact primary. Wire/qual, schema9/v16, legacy,
  preload/resume, wrong paths/identity/private markers/mutation fail before
  output. Historical checkpoint-bound modules and schema<=9 paths remain
  byte-compatible.

### CLI, lifecycle, and no retry

- Real `create_setting(save_flags=False)` binds all three exact v17 tuples,
  exact229/209, schema10, seal1, schedule100M, fresh inputs, paths/resources,
  and W&B modes. Derived identities never appear in CLI/YAML/config.
- Wrong xpid/whitespace, seed, horizon, warm-up, unroll, W&B, schema, path,
  surface, identity, preload, resume, resource, retry, or topology fails before
  run/environment action.
- Training with `ckp=true`, preload, parent, resume, or cross-schema state is
  rejected before restoration. Terminal checkpoint preservation is evidence,
  not resume authority.
- Real-Ray tests inherit barrier, claim/seal/drain0/1, logger two-phase,
  timeout/abort/kill, no-restart/no-task-retry, private cleanup, finish, and
  process/GPU cleanup.
- Immutable snapshot tests require exact source/data manifests, independent
  inodes/modes, empty runs, snapshot-only cp310 import, no cache/cp312/worktree
  module, focused/adversarial/full suites, and unchanged posthash.

## Final claim boundary

V17 requires, in order, one accepted immutable snapshot, one accepted
mechanics-only seed-1 wire, one accepted fresh seed-1 100k qualification, one
accepted fresh seed-5 300k primary, and that primary's one accepted
`v17-300k` fixed confirmation. Any failure is permanent for v17. No later
mechanics pass, diagnostic, seed, checkpoint, sibling protocol, or version can
retroactively change the permanent v14, v15, or v16 failures or any v17 stage
decision.

This authoritative preregistration does not itself perform or launch an
implementation, snapshot, experiment, retry, primary, fixed evaluation, or
evaluator action. Any later action must satisfy the sequential release gates
above on these exact frozen bytes.
