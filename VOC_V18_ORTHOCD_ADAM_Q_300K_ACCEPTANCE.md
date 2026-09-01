# Enduro VoC-v18 orthogonal common/difference Adam Q preregistered acceptance

This protocol is frozen after the sole VoC-v17 seed-1 100k qualification
failed and before any v18 implementation edit, immutable snapshot, wire,
qualification, primary, or fixed evaluation. V14, v15, v16, and v17 remain
permanent failures. V18 is the immediate successor of v17 and gate-policy
schema 10. It is one separately named, prospective, one-of-one, no-retry
experiment that changes exactly one algorithmic rule:

1. for gate-policy schema 11 only, retain schema 10's exact beta-1 Smooth-L1
   selected-action TD loss and common-mode Q reconstruction, but feed the
   existing Adam optimizer orthonormal common/difference gradients and map its
   common/difference step deltas back to the unchanged raw
   `[CONTINUE, STOP]` head parameters.

The raw `voc_head` weight and bias tensors, storage, state-dict keys and
shapes, forward rows, reconstructed online and EMA Q values, and projected
gate remain in `[CONTINUE, STOP]` order. Schema 11 does **not** reinterpret
the stored parameter rows as coordinates. Only the two rows of Adam's
`exp_avg` and `exp_avg_sq` state for each existing `voc_head` weight or bias
tensor have schema-11 meanings `[common, difference]`.

After inherited finite loss checks, no-argument optimizer zeroing, scaled
backward, and explicit unscale, `_step_voc_optimizer` first computes the raw
global norm. Its finite branch applies the single inherited global clip in
raw `[CONTINUE, STOP]` rows. Only then does schema 11 use the frozen positive
binary32 scalar

```text
s bits  = 0x3f3504f3
s value = 0.7071067690849304
```

and performs add/subtract before multiplication:

```text
g_m = s * (g_C + g_S)
g_d = s * (g_C - g_S).
```

Adam consumes `[g_m, g_d]`, maintains its existing two-row moments in that
coordinate order, and produces actual additive step deltas `[delta_m,
delta_d]`. Before parameter application, schema 11 inverse-maps them, again
with add/subtract before multiplication:

```text
delta_C = s * (delta_m + delta_d)
delta_S = s * (delta_m - delta_d).
```

The staged mapped deltas update the unchanged raw `[C,S]` parameters once on
atomic commit. There is no second gradient clip, optimizer instance, or
optimizer step; no new norm or clipping telemetry; and no new parameter,
buffer, optimizer, scheduler, scaler, tensor, checkpoint, or other telemetry
state. Ephemeral FP32 coordinate gradients, scratch parameters, candidate
state, and deltas are expected; they are not persisted state.

The schema-11 derived identities are exactly:

```text
voc_q_regression_loss="smooth_l1_beta1"
voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
voc_q_optimizer_coordinates="orthonormal_common_difference_adam"
```

All three strings are required in authoritative JSON-safe derived evidence,
including `resolved_identity` and inherited actor-policy, public, smoke, and
fixed evidence locations that carry derived identity. They are absent from
CLI/YAML keys, persisted training configuration and its exact 229-key
surface, embedded checkpoint flags, actor/ModelNet checkpoint and bundle
surfaces, tensor keys, and Adam state-dict keys; none creates a 230th
persisted field.

The delta is a falsifiable hypothesis, not a causal conclusion or pass
prediction. Orthogonal coordinate moments may reduce the frozen diagnostic
cross-coordinate coupling observed under rowwise Adam, but learned parameters,
EMA state, projected gate, sampling distribution, cadence, composition,
support, and acceptance direction are not invariant or predicted. A v18
observation may not select a second lever, a replacement seed, a denominator
pseudocount, relaxed gate, or retry.

## Permanent v17 qualification failure

The only v17 qualification was
`enduro-voc-v17-huber-common-eps25-seed1-qual-fresh-100k`, launched once from
immutable snapshot `/tmp/di-voc-v17-huber-common-eps25-final-Tmc42l`. It was
fresh, seed 1, W&B-enabled, schema 10, seal schema 1, and bound to the exact
229-key surface and unchanged 209-key projection. The driver exited zero.
Actor, beta-1 Smooth-L1/common-Q online-to-EMA-to-exact-projection,
actor-policy barrier/history, sealed ModelNet input, W&B two-phase completion,
public finish, manifests, process, Ray, and GPU validation all passed.
Successful mechanics and a passing pooled RMSE cannot rescue the sole
numeric failure.

The frozen actor log is `logs.csv`, SHA-256
`e50308619550a45e8bc722bd152722f96723f2b448885ae2e5ec65227f21d030`,
with 143 complete rows and 922 unique ordered columns. The canonical
qualification population contains exactly 36 complete rows under
`70000 < real_step <= 100000`, from 70416 through 100000, with no included
overshoot.

### Sole hard-gate failure: trailing-five denominator validity

There are 32 eligible trailing-five endpoints. Twenty-five have both sign
denominators. Seven have zero positive-sign support and therefore undefined,
invalid pooled gaps:

| Endpoint step | Positive support | Negative support |
| ---: | ---: | ---: |
| 74544 | 0 | 5869 |
| 75568 | 0 | 5897 |
| 76576 | 0 | 5867 |
| 77616 | 0 | 5867 |
| 78656 | 0 | 5861 |
| 79680 | 0 | 5849 |
| 80656 | 0 | 5773 |

Undefined gaps are not zero, nonnegative, negative, removable, imputable, or
replaceable. No pseudocount, smoothing, endpoint deletion, shortened window,
post-hoc start step, or denominator relaxation is permitted. Among defined
endpoints the maximum negative run is zero, but that distinct gate does not
repair denominator invalidity.

The pooled held-out EMA selected-action TD RMSE passed:

```text
sqrt(1291.2312621851 / 5748) = 0.4739621233029751 <= 0.5.
```

At fixed support the unchanged threshold SSE is `5748 * 0.5^2 = 1437`, so
the observed SSE is `145.7687378149` below the ceiling. Passing RMSE cannot
substitute for the failed sign-denominator gate. Frozen windows are:

| Window | Held-out support | EMA RMSE | EMA SSE |
| --- | ---: | ---: | ---: |
| W1 `(70000,80000]` | 1461 | `0.5044903887648509` | `371.8399169923` |
| W2 `(80000,90000]` | 1909 | `0.3328853383580048` | `211.5413459745` |
| W3 `(90000,100000]` | 2378 | `0.5455878532763281` | `707.8499992183` |
| Full `(70000,100000]` | 5748 | `0.4739621233029751` | `1291.2312621851` |

The online-Q held-out companion was `0.4695661556492494`; the EMA training
companion was `0.6696040559045617`. These report-only comparisons do not
alter the inherited EMA held-out gate or authorize selecting online Q.

V17 W1 EMA acceptance-sign support was exactly `0/11718`
positive/negative; the online companion was `1454/10264`. Online support was
positive through step 62128, zero from 62992 through 75568, and had its first
canonical reappearance after that collapse at 76576. EMA support was positive
through 69424, zero from 70416 through 80656, and reappeared at 81520, a
4944-real-step later recovery. Tau 1.0 therefore cannot repair the first two
observed invalid endpoints 74544 and 75568, because online support was also
zero there. This is observed temporal evidence, not a causal decomposition.

Every other inherited CSV-observable numeric gate passed, including all
frozen direction/window/margin, support-fraction, saturation, sign-agreement,
held-out RMSE, and safety requirements. Every actor, online-Q, gate, ModelNet,
protocol, AMP-skip, consecutive-skip, non-finite, mismatch,
malformed-bundle, and timeout counter met its inherited zero requirement.
There was no second numeric failure.

V17 failure permanently forbids a v17 primary, `v17-300k` fixed evaluation,
resume, extension, replacement seed, retry, or rescue. V18 is not a retry of
v17 and may not load a v14, v15, v16, or v17 checkpoint, optimizer, buffer,
actor version, observation, or runtime state.

## Frozen v17 failure artifacts and mechanics closure

The v17 qualification run directory is
`/tmp/di-voc-v17-huber-common-eps25-final-Tmc42l/runs/enduro-voc-v17-huber-common-eps25-seed1-qual-fresh-100k`.
Its exact 14 regular files are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `ckp_actor.tar` | 28299347 | `1054956ebbb8a1b75cb1d2eed54611c90d785c83ee3fc9a7d6d40e2c85430eb0` |
| `ckp_actor.tar_step_480` | 28259637 | `d93d16150136f67e8a9d200aefb31b7c2ecbb8a58ed01af4e092c02ce4669b66` |
| `ckp_actor.tar_step_100000` | 28307051 | `7d7eadebd7b333181efda0f40be310d85a4286fce4376b43304873b15456269c` |
| `ckp_model.tar` | 133359317 | `aca6d46ba749fd5394525918f6a6a822f0ac7f3377b7c953b511662012bf1a15` |
| `ckp_model.tar_step_10000` | 133380573 | `61488d3686849efaafe8f8d2dc9144b5925fab7140d23b45f0c930b8a42c2b6c` |
| `ckp_model.tar_step_100000` | 133359317 | `aca6d46ba749fd5394525918f6a6a822f0ac7f3377b7c953b511662012bf1a15` |
| `config_c.yaml` | 7627 | `f5d900bd2b6d0b25a628869b3dc90424e4a8018e888e88ace81a4f88d217b971` |
| `finish` | 3354 | `65f4c77efedbaaeaffe4a90cde00c8c4292942a3b5e3a5c7a39eba4b3acebff7` |
| `logs.csv` | 1442600 | `e50308619550a45e8bc722bd152722f96723f2b448885ae2e5ec65227f21d030` |
| `logs_model.csv` | 113447 | `2801e56a80dc8e3db20dcf0f22109a259e0a7fd6b31e4d1d2054fdd2ef75b564` |
| `meta.json` | 14797 | `c18a89d3352d8fd96a3e374c80ec2ad3d60dd5e22a8107f6432e92a17efc01bc` |
| `meta_model.json` | 14797 | `6d9aa7387bebb1e4fbaa744239c9c9291e752235222528ec2a9e3018c3757d36` |
| `out.log` | 52388 | `e27a8baadffb066032f39b92ed48473489bed1eb0598e09bfc4b6589df3993e4` |
| `out_model.log` | 148219 | `3d57b3674805d9924b569b053268e943d3e9bcdac372001ba04b9b6bc2bedb36` |

The canonical tree binding is the SHA-256 of the 1173-byte concatenation of
C-sorted `sha256sum` records formatted exactly as
`{lowercase_sha256}  ./{basename}\n`; it is
`bbf8f56cbdb35c3ca1156eba47911d95e365424ecb9ae3f17cf50cae1fbca4d3`.

The sole launch runtime `/tmp/v17qual-o7qrEu` has launch-provenance SHA-256
`17aebc1ec40196fa51205eb8ea208fb96b04750941abd44c9fc2bba589232905`
(5028 bytes), driver-log SHA-256
`a98da7f4e081ff353f8b40311ecb0b9e18eba0ef1ee3e8d8b3c7c2456974cdb7`
(389366 bytes), launch-wrapper SHA-256
`0d7d0dc7c0b95911cc310dc15d52102c151276dd797d1599da7258af16bab990`
(3484 bytes), and exit-file SHA-256
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
The exit file is exact `0` plus one LF.

The immutable v17 snapshot source manifest is
`87d17a63953caf41f62b92454e13ff92498c590eef9f6e156282e9e7e1d767ac`
with 1066 entries. The unchanged data manifest is
`23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343`
with 11 entries. The loaded cp310 extension SHA-256 is
`1d4c5026d2a6c002a13829e162428505b30a65bf2af1968f6e982dcfcc16b232`.
Terminal config/actor/model identity was exact schema 10, 229 keys,
qualification complete-surface SHA-256
`ec3c445584c77a3ed82e2855bba7addbb000124da9231e90f311e9a014e85beb`,
and unchanged v12 projection SHA-256
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.

Actor terminal real step was 100000. Q, EMA, exact-projection, actor-policy
version, and publication counts were 143; history length was 144; terminal
acknowledgement was `1/1`. Actor-state SHA-256 was
`6d754d038249572915ce72181163c44af093e13d8d76e018449683ee33c58748`;
publication-history SHA-256 was
`dc32518969b6e98064f1403a9a1f3e45c09199fecf8bce1580254ae18c1d59c2`;
logger-request SHA-256 was
`f5977b77cd4f4efdcb91c5315641483f2e63ef914f2b02954e03cbd1ba091b48`.
All protocol, optimizer-skip, and non-finite counters were zero; gate optimizer
steps remained zero; projection post-error was zero.

Model terminal real and processed step was 100000. Model-input seal schema was
1, sealed true, seal count one, drain-update count zero, pre-real step 100000,
pre and final m/p gradient counts 744/744, late writes zero, and aborts zero.
Finish status/schema were complete/1. W&B was required and completed with
verified ack, private-marker cleanup, and public finish. Source/data payload,
run-tree, process, Ray, and GPU closure all passed. These mechanics facts are
integrity evidence, not numeric acceptance.

## Frozen four-version RCA and design consensus

The authoritative predecessor preregistration is
`VOC_V17_HUBER_COMMON_Q_300K_ACCEPTANCE.md`, SHA-256
`9d032579808f5cc07253e654feea9f4f80fe45151701fd4864896bf486c267a1`.
The frozen executed technical notebook is
`notebook/v14_v15_v16_v17_qualification_comparison_rca.ipynb`, SHA-256
`b4d75b0a11953ef02abd3133e8949fb3d9949b1df56422f704b23b58767708f3`,
663425 bytes. The frozen machine-readable report is
`notebook/v14_v15_v16_v17_qualification_rca_report.json`, SHA-256
`fe45b63c0b10f8b19bef455d8d9ec4a3bb30f2ba8d78f4e354773ac390b67aec`,
190587 bytes. The report passed its MCP validator and one stakeholder render.
All three artifacts must retain these exact bytes for this preregistration to
remain authoritative.

The notebook and report reproduce exact CSV populations, supports, gates,
windows, RMSE/SSE, sign extinction/recovery, gradient norms, clipping,
cadence, composition, checkpoint tensors, Adam moments, manifests, and
mechanics bindings. Hash-bound trusted checkpoints were deserialized
read-only only as disclosed there. They do not replace terminal-bundle, W&B,
manifest, public-finish, or cleanup authority.

The observed four-run table is:

| Version | Loss | Reconstruction | Held-out EMA RMSE | Qualification |
| --- | --- | --- | ---: | --- |
| v14/schema7 | beta-1 Smooth-L1 | centered/no common | `0.5247954789453232` | fail RMSE |
| v15/schema8 | half-squared TD | centered/no common | `0.5637126651551874` | fail RMSE |
| v16/schema9 | half-squared TD | common | `0.643399622874774` | fail RMSE + denominator |
| v17/schema10 | beta-1 Smooth-L1 | common | `0.4739621233029751` | fail denominator only |

These are four separately generated, single-seed, policy-coupled on-policy
trajectories with different row cadence, support, and state/action
composition. They are not paired or randomized and do not identify a loss,
reconstruction, optimizer, EMA, cadence, or treatment effect.

At the common step-480 checkpoint, v16 and v17 online raw head, EMA head,
projected gate, and Adam step/moment tensors are bitwise equal. At the v17
terminal checkpoint, the orthonormal decomposition of the existing rowwise
Adam weight first moment has common/difference norms
`118.4118/61.2779`. Applying the frozen rowwise denominators diagnostically
to common-only and difference-only moment components yields respectively
common-to-difference and difference-to-common update-leakage ratios
`0.3125317498` and `0.3289216883`. These are diagnostic applications of
frozen state, not actual next steps, counterfactual training, or causal
effects. The bias state is excluded from those two ratios.

V17 canonical raw-gradient norm median/p95/max was
`509.1683654785/2808.359375/4498.305664`; the unchanged nominal boundary 1608
was exceeded on `5/36` canonical and `14/143` all-run rows. Huber's bounded
per-example residual slope therefore did not guarantee an aggregate norm
below the clipping boundary. V18 makes no such guarantee.

During the EMA sign-extinction interval 70416 through 80656, exact support
was `0/12822/0` positive/negative/tie. Depth 0, depth 1, and depth 2-3 counts
were `11216/1416/190`, each with zero positive and tie support. The collapse
was not merely a zero denominator created by one disappearing depth bin, but
the aggregate cannot distinguish fixed-feature optimizer coupling from the
changed on-policy composition.

The independent A/B consensus ranks schema-11 orthonormal
common/difference Adam coordinates first. The mechanism was already ranked
second prospectively in the v17 preregistration, before the v17 result. It
keeps the represented Q, raw head, loss, reconstruction, EMA, projection,
configuration, and state shapes fixed while separating Adam's coordinate
moments. Tau 1.0 ranks second as a lag ablation, but online's own zero-positive
run shows it cannot repair the first two observed invalid endpoints. Cadence
control ranks third. No ranking is a pass prediction or proof of mechanism.

## Sole schema-11 optimizer-coordinate delta

For every existing `voc_head` parameter tensor `P` of shape `[2, ...]`, row 0
remains raw CONTINUE and row 1 raw STOP. Forward, checkpoint parameter data,
EMA, and projection read these raw rows exactly as in schema 10. The delta is
implemented only by a schema-11 VoC-Q optimizer subclass/adapter sealed inside
the already existing `thinker/thinker/learn_actor.py`; no new module or source
surface is permitted.

Let the inherited post-unscale, postclip FP32 raw gradient rows be `g_C` and
`g_S`. The adapter never replaces or mutates a live parameter's `.grad`.
Instead it takes detached preserve-format clones of both raw rows and derives
private coordinate-gradient tensors from those clones:

```text
tmp_sum  = add_fp32(g_C_clone, g_S_clone)
tmp_diff = sub_fp32(g_C_clone, g_S_clone)
g_m      = mul_fp32(s, tmp_sum)
g_d      = mul_fp32(s, tmp_diff)
```

Every operation is elementwise; `s` is materialized as binary32 bits
`0x3f3504f3`. No host-double `1/sqrt(2)` recomputation, multiply-first form,
reassociation, alternate reduction, or fused substitute is equivalent. The
operation applies independently to the existing weight and bias tensors.
The private m/d gradients must have exact expected shape, dtype, device, and
finite values before the functional Adam call; a mismatch or non-finite value
is a transform failure and reaches no candidate-state mutation.
Live `.grad` remains branch-exact inherited raw state: clipped raw C/S on the
finite branch, or unscaled non-finite and unclipped raw C/S on the AMP
found-inf branch.

### Pinned production functional-Adam oracle

The canonical production runtime is PyTorch `2.13.0+cu130`. Its
`torch/optim/adam.py` source SHA-256 is
`bde360b0bb9b7869f1cec04a3b41a90b8eabb84a613787d97b88d87f2f3ae1ec`;
its `torch/amp/grad_scaler.py` source SHA-256 is
`97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039`.
Snapshot and wire evidence must bind the version and both source files. They
are runtime provenance, not CLI/config/checkpoint keys.

The external optimizer remains bound to the same raw parameters, parameter
group, state mapping, and state-dict. External parameter-group values remain
byte/value-compatible with inherited Adam, including `foreach=None` and
`fused=None`; schema 11 does not rewrite them to the internally resolved
values. The adapter step executes under inherited optimizer no-grad semantics;
it creates no autograd graph for staging, functional Adam, inverse mapping, or
commit.

On each adapter invocation, before any live mutation, it makes detached clones
of the live raw weight and bias parameters, all existing corresponding
optimizer state, and step tensors into a staged candidate. Existing parameter
order is exactly weight then bias. A first-step empty state is staged exactly
like inherited Adam: exact
`torch.tensor(0.0, dtype=torch.float32, device="cpu")` step tensors, plus
`torch.zeros_like(parameter, memory_format=torch.preserve_format)` FP32
`exp_avg` and `exp_avg_sq` tensors. No first-step state becomes live until a
successful commit.

The adapter allocates detached, `requires_grad=False`, positive-zero FP32
scratch parameters in existing weight-then-bias order. They match each raw
parameter's shape, layout, device, and preserve-format memory behavior but
never enter the external parameter group or state dict. It calls pinned
`torch.optim.adam.adam` exactly once on both scratch parameters, private m/d
gradients, and candidate states, with:

```text
foreach=True
fused=False
capturable=False
differentiable=False
decoupled_weight_decay=False
grad_scale=None
found_inf=None
has_complex=False
amsgrad=False
maximize=False
weight_decay=0
beta1=0.9
beta2=0.999
lr=<current inherited group lr>
eps=<current inherited group eps>
```

`max_exp_avg_sqs` is the inherited empty list and the candidate CPU scalar
steps are supplied in the same weight-then-bias order. The zero-base scratch
post-values themselves are the signed actual additive `delta_m` and
`delta_d`; they are never computed by a custom Adam formula, from a nonzero
proxy, or as post-minus-pre.

The adapter freezes both scratch delta rows before calculating either raw
destination candidate:

```text
tmp_delta_sum  = add_fp32(delta_m_clone, delta_d_clone)
tmp_delta_diff = sub_fp32(delta_m_clone, delta_d_clone)
delta_C        = mul_fp32(s, tmp_delta_sum)
delta_S        = mul_fp32(s, tmp_delta_diff)
P_C_candidate  = add_fp32(P_C_clone, delta_C)
P_S_candidate  = add_fp32(P_S_clone, delta_S)
```

The delta sign is the functional Adam call's actual additive change, not a
positive descent direction. Coordinate values never become raw stored
parameters and an untransformed coordinate delta is never applied to a raw
row.

Before commit, every staged raw parameter, coordinate delta, mapped delta,
`exp_avg`, `exp_avg_sq`, and step is checked for exact expected shape/dtype/
device and finiteness. The adapter then logically commits raw C/S parameters
and candidate m/d optimizer state exactly once. External parameter IDs,
group/state keys, state counts, tensor shapes and dtypes remain unchanged.
The commit is guarded by pre-cloned live raw/state values: an injected
exception during commit triggers exact rollback of every touched live value.
The exception remains fatal with no finish even after verified rollback; a
rollback failure is fatal and forbids any artifact from being accepted.

Before staging, the adapter requires the canonical group/runtime contract:
the exact raw weight/bias parameter order, dense non-complex FP32 tensors,
one inherited group, Adam defaults and state shapes, finite current lr/eps,
external `foreach=None`, `fused=None`, `capturable=False`,
`differentiable=False`, `amsgrad=False`, `maximize=False`, weight decay zero,
betas `(0.9,0.999)`, and the pinned source/runtime hashes. Any disagreement
fails before candidate-state or live parameter/optimizer-state mutation; the
adapter leaves inherited raw `.grad` untouched.

Within schema 11 only, committed Adam state row 0 accumulates `g_m` and row 1
accumulates `g_d`, so both `exp_avg` and `exp_avg_sq` have
`[common,difference]` semantics. There is no conversion of historical moments
and no additional persisted state.

In exact real arithmetic the transform matrix is
`s * [[1,1],[1,-1]]`, with ideal `s=1/sqrt(2)`, and is orthonormal and
self-inverse. The chain rule gives
`g_m=(g_C+g_S)/sqrt(2)` and `g_d=(g_C-g_S)/sqrt(2)`. Because Adam is
coordinatewise and nonlinear, separating its moment histories is not
equivalent to schema-10 rowwise Adam. That non-equivalence is the sole
intended semantic change.

Binary32 arithmetic is normative, not ideal real arithmetic. The frozen `s`
is not exact `1/sqrt(2)`; transform/inverse round trips, norms, parameters,
and checkpoint bytes are not promised bit-invariant. Tests distinguish
real-arithmetic algebra from exact ordered FP32 results, including rounding,
overflow, signed-zero, and cancellation cases.

### Exact inherited transaction and evidence order

The valid schema-11 training order is the inherited v17 order with the
adapter substituted only inside `_step_voc_optimizer`:

1. compute unchanged main actor, VoC-Q, and optional gate losses and require
   each present loss tensor finite before zeroing or backward;
2. call each present optimizer's no-argument `zero_grad()` in inherited actor,
   VoC-Q, then optional gate order, preserving the production default
   `set_to_none` behavior;
3. run the inherited scaled main-actor backward, scaled VoC-Q backward when
   supported, gate-gradient isolation, and optional scaled gate backward;
4. explicitly unscale the main actor, VoC-Q when present, and optional gate
   optimizers in inherited order;
5. call `_step_voc_optimizer` first when Q support exists. It computes the raw
   global gradient norm before any elementwise scan. If that norm is
   non-finite, the inherited branch identifies parameter names by elementwise
   non-finiteness and does not clip. Otherwise it applies the one inherited
   global clip to live raw C/S `.grad`, then `GradScaler.step(adapter)` may
   invoke the schema-11 adapter;
6. perform the optional gate optimizer step second and main actor optimizer
   step third, exactly at their inherited call sites;
7. after those steps, a successful Q step updates raw EMA once at tau 0.1 and
   exact-projects the current raw EMA delta once;
8. later, commit inherited pending training-support counters only for a
   successful Q step, independently commit finite held-out observations even
   for an all-held-out minibatch or recoverable Q AMP skip, then insert
   `voc_optimizer_stepped` and `voc_step_result.total_norm` into `losses`.
   That result is the computed preclip raw norm after a successful Q optimizer
   step and exact `0.0` after a recoverable AMP-skipped Q step. The fatal
   non-finite-norm/every-element-finite branch exits before this later losses
   insertion; norm computation itself emits no losses/stat record;
9. step the main actor scheduler first, then the Q scheduler only for a
   successful Q step, then the optional gate scheduler only for a successful
   gate step, at their inherited locations.

No independent full raw-gradient scan occurs before the norm. The
elementwise-name scan is diagnostic only inside the non-finite-norm branch.
There is no norm recomputation or second clip after coordinate derivation and
no clip of `g_m/g_d`, scratch values, `delta_m/delta_d`, or mapped
`delta_C/delta_S`. There is no second optimizer, scheduler, or scaler step and
no new m/d norm, clip count, or postclip telemetry.

### AMP, failure, and commit semantics

The following table is normative:

| Case | Adapter call / scaler | Live Q parameter and Adam state | Raw `.grad` | Q follow-on |
| --- | --- | --- | --- | --- |
| finite raw path, success | one adapter call inside `GradScaler.step`; atomic commit; then `GradScaler.update` and inherited scale comparison | one committed raw C/S update and one committed m/d state/step update | inherited clipped raw C/S | optional gate, main actor, raw EMA/projection, counters, and successful-Q scheduler continue in inherited order |
| raw AMP found-inf | after unscale, non-finite norm diagnostic; no clip; `GradScaler.step(adapter)` does not call adapter; `update()` backs off; optimizer-stepped false | unchanged | inherited unscaled non-finite raw C/S | no Q optimizer/update-support counter, EMA, projection, or Q scheduler advance; inherited held-out observation and loss evidence are still committed later, and optional gate/main actor follow inherited behavior |
| non-finite raw norm with every raw element finite | inherited diagnostic raises before scaler step, adapter, or scaler update | unchanged | inherited unscaled finite and unclipped raw C/S | fatal, no later step or finish |
| coordinate transform failure | adapter entered, functional call zero; raise before scaler update | unchanged | inherited clipped raw C/S | fatal, no later step or finish |
| functional-Adam failure | pinned functional call entered once on staged values; raise before scaler update | unchanged | inherited clipped raw C/S | fatal, no later step or finish |
| staged non-finite/shape failure | functional call completes only on staged values; validation raises before scaler update | unchanged | inherited clipped raw C/S | fatal, no later step or finish |
| commit exception, rollback verifies | commit attempted once; exact rollback; raise before scaler update | exactly restored | inherited clipped raw C/S | fatal, no later step or finish |
| commit exception, rollback fails | fatal unrecoverable commit fault | no unchanged-state claim | inherited raw branch state unless itself affected by unrecoverable fault | no artifact or finish may be accepted |

On a successful adapter call, scaler update and the inherited scale-based
optimizer-stepped decision follow the commit. Transform, functional, staged
validation, and commit exceptions propagate before `GradScaler.update()`.
The FP32 no-scaler path is analogous: raw non-finite norm raises before the
adapter; a finite branch clips once and invokes the adapter directly; staged
failure or verified rollback advances no live Q parameter/state, EMA,
projection, Q counter, or Q scheduler.

The phrase “advances none” applies only before live commit or after verified
rollback. A failure after a successful Q commit, including a later optional
gate or main-actor fault, follows inherited fatal semantics and forbids a
finish/artifact; this protocol does not falsely claim the earlier Q commit was
undone. Existing post-step/final validation rejects non-finite parameter,
optimizer, scaler, telemetry, or checkpoint state without inventing a scan of
live Adam state before staging. Main actor AMP remains transactionally
independent, while any live skip remains an acceptance failure.

## Normative inheritance from v17

The frozen v17 protocol
`VOC_V17_HUBER_COMMON_Q_300K_ACCEPTANCE.md`, SHA-256
`9d032579808f5cc07253e654feea9f4f80fe45151701fd4864896bf486c267a1`,
is incorporated verbatim except for these closed substitutions:

- experiment names and fixed profile change from v17 to v18 identities;
- gate-policy schema 10 changes to schema 11;
- only the online-Q Adam gradient/moment/update coordinates change exactly as
  specified above; and
- schema-11 authoritative derived evidence adds exactly
  `voc_q_optimizer_coordinates="orthonormal_common_difference_adam"`.

Schema 10's beta-1 Smooth-L1 selected-action TD loss and common-mode online
and EMA reconstruction remain byte- and behavior-identical. The exact loss is
still:

```python
selected_q_work = selected_q.float()
target_work = target.float()
q_loss_rows = F.smooth_l1_loss(selected_q_work, target_work, reduction="none")
q_loss = torch.sum(q_loss_rows * q_train_valid.float())
```

The omitted beta retains the PyTorch default 1.0. FP32 operands, subtraction
sign, selected action, target/action mapping, mask multiplication, sum
reduction, zero-support branch, logged supported mean, outer
`voc_loss_cost=1.0`, and held-out isolation are unchanged. Under schema 11,
`actor/voc_q_loss` retains the schema-10 meaning: mean beta-1 Smooth-L1
selected-action training TD loss on positive `q_train_valid` support.

The online and EMA reconstruction remains exactly:

```python
raw = A
common = raw.mean(dim=-1, keepdim=True)
centered = raw - torch.sum(p_det * raw, dim=-1, keepdim=True)
Q = V_det.unsqueeze(-1) + common + centered
```

`p_det`, `V_det`, their detaches, action mapping, target, held-out split,
online-to-EMA tau-0.1 transaction, and exact projection from the raw EMA
`A_CONTINUE-A_STOP` difference are unchanged. For selected action `j` and raw
action `k`, the real-arithmetic Jacobian remains
`dQ_j/dA_k = 1/2 + 1[j=k] - p_det[k]`. In real arithmetic
`Q_C-Q_S=A_C-A_S`; FP32 values are whatever the exact stored raw rows and
operation order produce.

Training soft epsilon remains 0.02, executed gate epsilon 0.25, main actor
AMP initial scale 32, EMA tau 0.1, schedule total 100000000, and nominal raw-Q
gradient clip boundary 1608. The strict actor-policy version barrier, exact
five-key bundle and ack, exact seven-key publication history, W&B two-phase
completion, source-hardcoded Ray no-retry topology, exact EMA-to-gate
projection, and schema-1 ModelBuffer input seal remain unchanged.

Every v17 network architecture, raw affine head, tensor shape,
initialization, parameter ownership, actor/ModelNet precision rule, loss,
reconstruction, target, action mapping, value/policy detach, trajectory and
replay rule, learning rate, Adam hyperparameters, scheduler, scaler, held-out
split, clip threshold, telemetry field, artifact rule, population, window,
pooling rule, support floor, behavioral-accuracy definition, sampled
no-op/forced-action exclusion, and acceptance threshold remains unchanged.
Only the schema-11 Q optimizer-coordinate adapter changes.

V18 retains exact Enduro data/configuration, CUDA devices 0 and 1 only, Ray
resources two GPUs and 16 CPUs, W&B-disabled wire, W&B-required
qualification/primary, and evaluator-private epsilon/barrier/seal-runtime
overrides. Pong, Space Invaders, alternate Enduro seeds, early fixed
evaluation, and post-hoc diagnostics as selection inputs remain forbidden.

No second lever may change the loss, beta, coefficient, mask, reconstruction,
raw-head representation, network, learning rate, Adam beta/epsilon/weight
decay, scheduler, scaler, raw clip or its order, EMA tau, projection,
temperature, epsilon, action weighting, normalization, shared gradient,
auxiliary loss, telemetry, replay, batch, unroll, warm-up, held-out split,
barrier, seal/drain, logger, terminal order, update cadence, optimizer step
count, retry, seed, checkpoint selection, threshold, population, denominator,
pseudocount, or fixed rule. In particular, tau 1.0, a deterministic cadence,
sign-balanced weighting, difference-margin loss, and a new head or state are
not part of v18.

Unchanged rules do not imply an unchanged learned path. The different Adam
coordinate histories can alter raw parameters, EMA, gate, sampling,
trajectory, row cadence, support, and every learned metric. No direction or
pass outcome is invariant or predicted.

## Exact schema lineage

Schemas at most 7 retain byte-, shape-, return-, path-, and
behavior-identical historical beta-1 Smooth-L1, old centered/no-common
reconstruction, and rowwise C/S Adam. They retain no derived loss,
reconstruction, or optimizer-coordinate identity.

Schema 8 retains byte-, shape-, return-, path-, and behavior-identical v15
half-squared TD, old centered/no-common reconstruction, and rowwise C/S Adam.
Its resolved identity retains exactly
`voc_q_regression_loss="half_squared_td"` and no reconstruction or optimizer
coordinate field.

Schema 9 retains byte-, shape-, return-, path-, and behavior-identical v16
half-squared TD, common reconstruction, and rowwise C/S Adam. Its derived
identity retains exactly:

```text
voc_q_regression_loss="half_squared_td"
voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
```

Schema 10 retains byte-, shape-, return-, path-, and behavior-identical v17
beta-1 Smooth-L1, common reconstruction, and rowwise C/S Adam. Its derived
identity retains exactly:

```text
voc_q_regression_loss="smooth_l1_beta1"
voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
```

Schema 11 retains schema 10's loss, reconstruction, raw forward and parameter
semantics, EMA, and projection, and changes only the optimizer-coordinate
adapter. Its derived identity is exactly:

```text
voc_q_regression_loss="smooth_l1_beta1"
voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
voc_q_optimizer_coordinates="orthonormal_common_difference_adam"
```

All schemas at most 10 remain exact historical branches. A schema-11
implementation may not refactor them into a shared route that changes their
source-visible behavior, values, Adam row semantics, keysets, return shapes,
error order, validation order, public bytes, or fixed outputs.

## Exact schema-11 identity and persisted surfaces

V18 uses strict non-Boolean built-in Python integer
`voc_gate_policy_schema_version=11`. Boolean, NumPy integer, float, string,
missing/defaulted value, 10, or 12 is not schema 11. The existing gate-schema
value plus the new derived optimizer-coordinate value is the sole new
algorithm identity. There is no new persisted configuration key.

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

Only the existing gate-schema value becomes 11. Execution epsilon remains
0.25; barrier true; bundle schema 1; timeout 120.0; Ray actor restart/task
retry zero; actor AMP initial scale 32; training barrier runtime true; and
model-input-seal schema 1. Missing, extra, coerced, non-finite, wrongly typed,
or defaulted values fail closed.

The 209-key v12 projection remains byte-identical with SHA-256
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.
Config, actor metadata, and ModelNet metadata must agree on all 229 keys and
the canonical complete-surface digest. Each digest is resolved from exact
v18 paths and command before launch and is not guessed here.

Every actor-policy bundle retains exact keys
`{bundle_schema_version, policy_version, terminal, gate_schema,
actor_state_dict}`, bundle schema 1, and gate schema 11. Every ack retains
exact keys `{bundle_schema_version, gate_schema, rank, policy_version,
terminal}` and gate schema 11. Publication history remains exact seven-key
events `{predecessor_version, policy_version, publication_count, terminal,
ack_ranks, expected_ack_count, state_sha256}` and adds neither schema nor
derived identity.

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

All strict types, seal count one, final-real equality, drain zero/one
branches, pre/final m/p relations, durable save, complete-success,
actor-before-model-before-finish, late-write zero, abort zero, and
no-post-terminal-action relations remain unchanged. ModelBuffer exact13
runtime status, claim-token linearization, independent 120-second RPC bounds,
seal denial of later claims/writes, and abort behavior remain unchanged.

Schema-11 training finalization must use dedicated
`validate_schema11_final_bundle` before ModelBuffer complete-success. Public
completion must use dedicated `validate_schema11_completed_bundle`, and
actor-only terminal validation must use
`validate_voc_schema11_final_actor_checkpoint`. Shared dispatch may recognize
schemas 6 through 11, but every dedicated schema-11 route strictly rejects
every non-11 or wrongly typed value. Schemas at most 10 never call those
dedicated routes and retain exact return shapes and behavior.

Each schema-11 inner JSON-safe resolved identity has exactly 12 keys:

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
voc_q_optimizer_coordinates
```

The last three values are exactly the three frozen strings above. Versioned
route labels and validator values make only their required schema-10-to-11
discriminator substitution; this preserves the schema-10 container structure.
Required schema-11 derived-evidence keysets then follow one mechanical rule:
every schema-10 evidence mapping that directly contains both
`voc_q_regression_loss` and `voc_q_reconstruction` retains its exact schema-10
keys and adds exactly the single key `voc_q_optimizer_coordinates`.
Concretely:

- the dedicated actor-only final-validator result is its schema-10 keyset plus
  exactly `voc_q_optimizer_coordinates`;
- every final/completed bundle `actor_policy` evidence mapping that directly
  repeats the schema-10 loss/reconstruction pair is its schema-10 keyset plus
  exactly that key;
- the authoritative final/public inner `resolved_identity` is the exact
  12-key mapping enumerated above, and each derived
  `stored_surface_identity.{config,actor_checkpoint,model_checkpoint}` is an
  exact copy of that 12-key mapping;
- an authoritative final-bundle or public completed-record container keeps
  its exact schema-10 container keyset unless it directly repeats the two
  strings, in which case its exact keyset gains only the optimizer-coordinate
  key; all nested mappings obey the preceding bullets;
- smoke local `resolved_identity` and its outer
  `voc_checkpoint_resolved_identity` are each exact-three containers with
  keys `{config, actor_checkpoint, model_checkpoint}`; every inner value is
  the exact 12-key schema-11 identity above. Smoke
  `schema11_final_bundle_validation` keeps the corresponding schema-10
  completed-validation container structure, with every mapping that directly
  repeated the loss/reconstruction pair gaining exactly the
  optimizer-coordinate key;
  and
- fixed `resolved_profile_identity` retains the inherited profile container
  structure and its v18-specific direct identity mapping contains exactly
  `schema11_final_bundle`, `voc_q_regression_loss`, `voc_q_reconstruction`,
  and `voc_q_optimizer_coordinates` in place of the corresponding schema-10
  v17 entries. Any fixed completed-record mapping that directly repeats the
  pair likewise gains exactly the one optimizer-coordinate key.

Here `actor_policy` evidence means the JSON-safe validator/public result
mapping, not the persisted exact-five-key actor-policy bundle. That persisted
bundle remains unchanged except for its existing `gate_schema=11` value.

No schema-11 derived-evidence location may omit the third string, retain a
schema-10-only pair, or add another identity field. These required JSON-safe
evidence keyset additions are explicitly permitted and do not describe
persisted training state. All three names and values remain absent from CLI,
YAML, persisted training configuration and its exact229 surface, embedded
checkpoint flags, actor/ModelNet checkpoints, actor-policy bundle, raw tensor
keys, optimizer state keys, and parameter groups. Schema-10 resolved identity
remains exact11 and every schema at most 10 retains its exact evidence,
container, record, and canonical output bytes.

External `voc_optimizer_state_dict` keysets, parameter IDs, parameter-group
keysets, state counts, tensor shapes, dtypes, and finite/step requirements are
unchanged. For schema 11 only, each existing two-row `exp_avg` and
`exp_avg_sq` tensor corresponding to the raw `voc_head` weight or bias has
row meanings `[common,difference]`; the paired model parameter tensor remains
raw `[CONTINUE,STOP]`. The schema discriminator and derived evidence bind
this interpretation. No validator may infer that a schema-10 rowwise moment
checkpoint can be converted merely because its external shape matches.
The stored tensor bytes alone cannot prove which row semantics produced them;
strict schema 11, the frozen implementation path and tests, fresh provenance,
and authoritative derived evidence are the discriminator.

All three derived names are reserved from persisted schema-11 surfaces.
Dedicated actor and final validators recursively and cycle-safely reject
their explicit presence at any actor or model checkpoint mapping depth,
including mappings nested in lists or tuples, before tensor/state use. Value
does not matter: correct, wrong, null, or forged presence fails. Benign cycles
without reserved keys terminate safely. Schemas at most 10 retain their exact
historical behavior.

## Fresh-only state semantics

Every v18 stage is fresh: `ckp=false`; `preload`, `preload_actor`, and
`voc_parent_checkpoint` are empty; parent-update count is zero; actor-policy
version starts at zero; and raw online Q, raw EMA Q, exact projected gate,
Adam state, ModelNet, buffers, and seal state start from unchanged fresh
state. No state crosses stages.

No v14-v17 rowwise Adam state can be reinterpreted or migrated. No schema-11
wire, qualification, or primary checkpoint may seed another stage. Resume is
forbidden even from schema 11. Terminal state preservation is required
evidence only, never restore authority.

Schema-11 intent is recognized before any persisted config open, run-dir
creation, checkpoint load, or environment action. Malformed V18-prefixed
xpid claims through built-in strings, string subclasses, NumPy strings,
bytes-like UTF-8 values, `os.PathLike`, or other lexical representations are
classified as schema-11 intent and then rejected unless the exact stage xpid
is a built-in string. Conversion or decoding failure also fails before I/O.
This early classifier does not coerce or accept the value.

The only valid tuples
`(xpid, base_seed, total_steps, model_warm_up_n, actor_unroll_len, use_wandb)`
are exactly:

- (`enduro-voc-v18-orthocd-adam-eps25-sentinel-wire1200`, 1, 1200, 512, 41,
  false);
- (`enduro-voc-v18-orthocd-adam-eps25-seed1-qual-fresh-100k`, 1, 100000,
  10000, 201, true); and
- (`enduro-voc-v18-orthocd-adam-eps25-seed5-strict-fresh-300k`, 5, 300000,
  10000, 201, true).

Each xpid is an exact built-in string; numeric members are strict non-Boolean
built-in integers; `use_wandb` is an exact built-in Boolean. No trimming,
coercion, alias, or alternate tuple is allowed. Config, actor metadata, and
ModelNet metadata must agree. Normalized real ckpdir basename equals xpid and
all paths/command bind the same immutable v18 snapshot.

All stages retain `schedule_total_steps=100000000`, exact Enduro paths and
network configuration, CUDA `0,1` only, and Ray two GPUs/16 CPUs. Wire is
W&B-disabled; qualification and primary require authenticated W&B and the
inherited request/ack/private-cleanup/public-finish sequence.

Every stage has exactly one attempt. Wire cannot continue into qualification.
Qualification is one separate fresh seed-1 run. Primary is one separate fresh
seed-5 run and may start only after every qualification gate passes. There is
no resume, preload, extension, fallback seed, retry, duplicate xpid, selected
checkpoint, replacement run, or cross-schema state. Any failure permanently
ends v18 at that stage.

## Sequential release gates

The only release order is:

1. Implement only schema-11 optimizer-coordinate routing, strict propagation,
   derived identity, dedicated validation/profile, and frozen tests. Freeze
   all bytes and pass two independent code/contract audits.
2. Build a fresh inode-independent immutable snapshot from the authoritative
   v17 source/data baseline plus an exactly enumerated v18 overlay. Pass two
   independent manifest, mode, cp310, schema, test, and posthash audits.
3. Run exactly one fresh seed-1 1.2k wire. Decide mechanics only.
4. Only after wire passes, run exactly one fresh seed-1 100k qualification
   with every inherited gate unchanged.
5. Only after qualification passes every gate, run exactly one fresh seed-5
   300k primary with every inherited gate unchanged.
6. Only after primary passes may its terminal checkpoint receive one fixed
   confirmation under exact profile `v18-300k`.

No primary or fixed evaluation may launch speculatively or after failed
qualification. Pong, Space Invaders, alternate seed, shortened run, or
diagnostic fixed evaluation remains forbidden until an accepted v18 Enduro
claim exists.

## Integrity-wire acceptance

Wire may inspect only immutable provenance/config, schema-11 branch identity,
all three derived strings, first/final checkpoints, actor versions/acks/
history, exact coordinate optimizer transaction, Huber/common reconstruction,
Q/EMA/projection transactions, AMP/non-finite counters, ModelBuffer seal/drain
ordering, W&B-disabled logger completion, finish, manifests, and
process/Ray/GPU cleanup. It supplies no qualifying behavioral row.

Wire must exercise at least one supported beta-1 Smooth-L1 online-Q update.
Evidence must show the inherited finite loss checks, no-argument zeroing,
scaled backward, explicit unscale, raw norm-first branch, and single finite-
branch raw clip precede the schema-11 transform; one Adam step advances m/d
moments; inverse-mapped raw C/S parameters feed one raw EMA update; exact
projection follows once; and counters remain lockstep. It must bind production
PyTorch `2.13.0+cu130`, `torch/optim/adam.py` SHA-256
`bde360b0bb9b7869f1cec04a3b41a90b8eabb84a613787d97b88d87f2f3ae1ec`,
`torch/amp/grad_scaler.py` SHA-256
`97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039`,
and the adapter's sole implementation location in the sealed
`thinker/thinker/learn_actor.py`. It must also exercise one nonterminal actor
publication, terminal publication/ack, a valid seal drain-zero-or-one branch,
durable model save, complete-success, exact-true worker returns, and clean
process closure.

Validator/public evidence must carry schema 11 and all three derived strings,
while config/checkpoint/bundle/optimizer/tensor keysets remain unchanged and
exact229. The wire may validate state shape, finiteness, row-semantic binding,
and transaction counters; frozen deterministic tests, not observational
guessing, bind exact FP32 transform arithmetic.

Any skip, non-finite, second clip, second step, raw/coordinate row confusion,
timeout, malformed/history error, late write, abort, retry, stale checkpoint,
missing finish, W&B artifact where forbidden, source drift, or incomplete
cleanup permanently fails v18. Negative paths need not occur live but must be
frozen in tests.

## Frozen 100k qualification

The inherited 100k decision remains unchanged. Canonical rows satisfy
`70000 < real_step <= 100000`; frozen windows are `(70000,80000]`,
`(80000,90000]`, and `(90000,100000]`; overshoot is excluded. Required cells
are finite, rows complete, steps unique and strictly increasing, and
malformed, duplicate, or nonmonotone input fails closed.

Qualification passes only if every inherited gate passes together:

- teacher gap at least `0.075`, student gap at least `0.05`, retention at
  least `0.50`, and signed margin strictly positive;
- at least two of three windows each have positive student gap and margin;
- every trailing-five endpoint has strictly positive positive-sign and
  negative-sign denominators, and maximum consecutive negative trailing-five
  pooled gaps is at most 3;
- train and held-out CONTINUE/STOP fractions are each strictly above `0.05`;
- wrong-CONTINUE saturation is strictly below `0.01`, with wrong-STOP and
  forced-stop diagnostic retaining inherited status;
- online-versus-EMA non-tie sign agreement is at least `0.60`;
- held-out EMA selected-action TD RMSE is at most `0.5`; and
- actor, online-Q, gate, ModelNet, protocol, AMP-skip, and non-finite counters
  meet inherited zero requirements.

Every eligible trailing-five endpoint is required. Zero support makes its
gap undefined and the gate false. Undefined is never coerced to zero or
nonnegative; no pseudocount, smoothing, minimum-support replacement, endpoint
deletion, altered window, or shorter population is permitted. This exact
rule is intentionally unchanged after the v17 failure.

Schema11/exact229/209, all three derived identities, exact beta-1 Smooth-L1,
unchanged common reconstruction, exact optimizer-coordinate transaction,
raw-Q-to-raw-EMA-to-projection, barrier/history, W&B, seal exact-ten, finish,
manifests, and cleanup are hard integrity gates without new numeric
thresholds. Any failed qualification permanently forbids primary and fixed
evaluation.

## Frozen 300k primary acceptance

The inherited 300k primary decision remains unchanged. Full is
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
saturation, denominator, and calibration accounting retain inherited
definitions.

All artifact, provenance, Huber/common/optimizer-coordinate mechanism,
behavior, stability, support, denominator, trailing-five, saturation,
forced-stop, calibration, barrier, seal, AMP, and non-finite gates pass
together. There is no partial, diagnostic-only, mechanism-only, or historical
pass.

## Public, smoke, and fixed confirmation

The closed fixed profile is exactly `v18-300k` and accepts only the one
accepted seed-5 primary tuple. It rejects wire, qualification, schema10/v17,
and every legacy profile before rollout or output. Held-out seeds remain
20260827 through 20260842, exactly 16 streams by 6250 real steps and 100000
total, with calibration V-trace unroll 201 and inherited algebra.

After importing checkpoint-bound public code and resolving requested profile,
fixed evaluation must validate complete schema-11 primary before any
evaluator-direct/downstream flag load, live spec/environment probe,
construction/reset/action, data access, direct tensor load/use, rollout, or
output. Validator-internal bound deserialization is allowed and excluded from
downstream counters. Prevalidated evidence is reused exactly but never trusted
as a substitute for the dedicated validator.

Initial `config_c.yaml` bytes are stable-read, SHA-bound to completion
evidence, classified from those exact bytes, and consumed by the inherited
byte-aware loader without reopening mutable checkpoint config. Deletion,
replacement, schema11-to-legacy, legacy-to-schema11, alternate explicit
config, and probe-to-load swaps fail before downstream use or consume only
bound bytes; final revalidation catches artifact mutation before output.

Smoke order remains exact: immutable schema-11 prevalidation, stable byte
binding, private runtime copy, evaluator-only overrides, authoritative
postvalidation, exact pre/post evidence and checkpoint-hash equality, then
environment. Fixed order remains bound public import/requested profile,
schema-11 dispatch and dedicated validation/evidence equality, then
downstream use.

Validation covers exact229/209, all three identities, bundle/ack/history,
actor raw parameters and m/d Adam-state semantics, Huber/common Q, raw
EMA/projection state, ModelNet state, seal exact-ten, W&B completion,
private-marker absence, public finish, source/runtime binding, and primary
tuple. Wrong/missing/extra/forged identity or evidence fails closed.

Schema 10 under `v18-300k` or another incompatible v18 profile, and schema 11
under any legacy or `v17-300k` profile, yields zero downstream calls/output.
Historical schema 10 remains byte-compatible on its unchanged v17 route;
this compatibility does not authorize evaluation, and v17 remains
permanently failed.

Only after validation may an evaluator-private copy disable actor and
ModelNet training, parallel execution, live barrier waiting, and live seal
coordination. It records immutable epsilon 0.02/0.25, schema 11, seal 1, and
all three identities while using runtime epsilon 0/0, barrier wait false, and
seal coordination false. Stored config/checkpoints are never rewritten.
Fixed B/calibration probability continues to use recorded learned-gate
fields, not epsilon-zero execution likelihood.

## Frozen test and audit matrix

Before implementation or snapshot clearance, tests cover at least this
matrix.

### Exact algebra, Jacobian, and FP32 order

- Construct `s` as FP32 bit pattern `0x3f3504f3` and prove its exact value is
  `0.7071067690849304`. Host-double recomputation, alternate rounded constant,
  or non-FP32 scalar rejects.
- Hand-computed scalar, vector, weight `[2,D]`, and bias `[2]` cases bind
  `g_m=s*(g_C+g_S)` and `g_d=s*(g_C-g_S)`, with add/subtract before multiply.
  Spies or exact reference operations reject multiply-first, reassociated,
  swapped-row, sign-reversed, or fused alternatives.
- Mutation-sensitive cases prove the transform clones both raw source rows
  before writing either temporary row, and the inverse transform clones both
  coordinate deltas before updating either raw parameter row.
- Hand-computed Adam deltas bind
  `delta_C=s*(delta_m+delta_d)` and
  `delta_S=s*(delta_m-delta_d)` with the same ordered FP32 operations and
  actual-additive-delta sign convention.
- Real/high-precision tests establish the orthonormal matrix, self-inverse
  algebra, norm preservation, and chain-rule Jacobian. Separate binary32
  tests cover rounding-safe cases and explicit cancellation, signed-zero,
  large finite, overflow, and non-roundtrip cases; no arbitrary FP32 byte or
  norm invariance is asserted.
- Raw parameter rows before and after a step remain `[CONTINUE,STOP]` and
  forward/state-dict consumers observe raw rows. Coordinate gradients or
  deltas never become stored model parameters.

### Loss, mask, gradient isolation, and common reconstruction

- Schema 11 beta-1 Smooth-L1 values/gradients match schema 10 exactly for
  zero, positive/negative subunit, `+/-1`, and positive/negative tail errors,
  with exact FP32 operands, default beta, `reduction="none"`, mask, and sum.
- Mixed train/held-out masks prove held-out value and gradient zero.
  All-held-out/no-support performs no Adam, EMA, or projection update and
  emits inherited zero-support evidence.
- Online and EMA schema-11 reconstruction is byte-identical to schema 10:
  unweighted mean with `keepdim=True`, policy-centered raw head, detached
  policy/value, and sum of common plus centered terms.
- Hand-computed asymmetric raw heads and nonuniform policy bind Q actions,
  selected Q, common, centered, and the unchanged Jacobian
  `1/2 + 1[j=k] - p_det[k]`.
- Gradients reach only existing online raw-head parameters. Policy/value,
  target, actor, gate, ModelNet, and held-out rows remain isolated.
- Fresh zero raw-head output and empty/zero Adam state preserve exact
  schema10/11 pre-update forward, key, shape, initialization, EMA, and gate
  parity. No post-update trajectory parity is claimed.

### Clip, AMP, non-finite, and transaction order

- Instrumented ordering proves present-loss finite checks -> no-argument
  actor/Q/gate `zero_grad()` calls -> scaled backward -> explicit unscale -> Q
  raw global norm computation first -> conditional non-finite-name diagnosis
  or one raw clip -> Q adapter -> optional gate step -> main actor step ->
  successful-Q raw EMA/projection -> successful-Q training-support counters
  plus independently eligible held-out/loss insertion ->
  actor/Q-success/gate-success schedulers.
- Tests bind canonical no-argument `zero_grad()` and production default
  `set_to_none` behavior. They reject a positional/keyword override or moving
  zeroing after backward.
- A finite loss is required before zero/backward. `_step_voc_optimizer`
  computes raw norm before any elementwise gradient scan; only a non-finite
  norm activates name diagnosis, while a finite norm activates one raw C/S
  clip. No independent pre-norm full-gradient scan is allowed.
- Norm calculation emits no stat there. Later losses insertion publishes
  `actor/voc_total_norm = voc_step_result.total_norm`: the computed preclip raw
  norm after a successful Q optimizer step, or exact `0.0` after a recoverable
  AMP-skipped Q step, alongside optimizer-stepped evidence. The fatal
  non-finite-norm/every-element-finite branch exits before that insertion. No
  m/d norm, postclip norm, clip count, or new telemetry is emitted.
- The adapter never mutates `.grad`. Success and every failure injection prove
  live `.grad` remains inherited clipped raw C/S on the finite branch or
  unscaled non-finite/unclipped raw C/S on the AMP found-inf branch.
- A finite over-threshold raw gradient is clipped before private coordinate
  derivation and may step. Exact call-count spies prove the only clip count is
  one and there is no clip on m/d gradients, scratch values, functional
  deltas, or mapped raw deltas.
- AMP success proves unscale once, `GradScaler.step(adapter)` once, adapter
  call once, pinned functional call once, atomic commit once, and scaler
  update once. Q then optional gate then main actor ordering, raw EMA/
  projection, later evidence, and scheduler ordering are exact.
- Raw AMP found-inf proves unscale once, non-finite norm/name diagnosis, clip
  zero, scaler step once, adapter/function/commit zero, scaler update/backoff
  once, Q optimizer-stepped false, and no Q optimizer/update-support counter,
  EMA, projection, or Q scheduler advance. Inherited held-out observation and
  loss evidence still commit later; gate/main actor retain inherited behavior.
- A non-finite raw norm produced solely by finite raw elements proves the
  inherited fatal diagnostic path: clip/scaler-step/adapter/scaler-update are
  each zero and no live Q state or later transaction advances.
- Transform failure, functional failure, staged-nonfinite/shape failure,
  commit exception with verified rollback, and rollback failure are distinct
  tests. The first four bind no scaler update and the exact state guarantees
  in the normative table; rollback failure binds fatal/no accepted artifact
  without an unchanged-state claim.
- FP32 tests mirror the table without a scaler: raw non-finite norm raises
  before adapter, finite raw gradients clip once, and staged failure or
  verified rollback advances no live Q state or downstream Q transaction.
- Failure after a successful Q commit is fatal/no finish but is never
  mislabeled as a rolled-back Q update. Main actor AMP remains independent;
  every live skip is still an acceptance failure.

### Adam state semantics and checkpoint surfaces

- Snapshot/import and wire tests require production
  `torch==2.13.0+cu130`, `torch/optim/adam.py` SHA
  `bde360b0bb9b7869f1cec04a3b41a90b8eabb84a613787d97b88d87f2f3ae1ec`,
  and `torch/amp/grad_scaler.py` SHA
  `97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039`.
  Wrong version/source fails before adapter staging or live
  parameter/optimizer-state mutation.
- Source-scope tests require the schema11 adapter/helper to live only in the
  already sealed `thinker/thinker/learn_actor.py`; no new module, optimizer
  source file, configuration key, or import surface is allowed.
- Runtime/group preflight attacks each required condition before staging:
  weight/bias order, group count, FP32/dense/noncomplex values, external
  `foreach=None`/`fused=None`, capturable/differentiable/amsgrad/maximize false,
  weight decay zero, betas .9/.999, and finite current lr/eps.
- First-step tests bind inherited lazy state exactly: CPU scalar zero step with
  exact `torch.float32` dtype and preserve-format `zeros_like` FP32 moments.
  Empty live state remains empty if staging fails.
- Deterministic first and later steps call pinned `torch.optim.adam.adam` once
  on positive-zero scratch parameters in weight-then-bias order with explicit
  `foreach=True`, `fused=False`, and every other frozen keyword. Exact scratch
  post-values, candidate steps, `exp_avg`, and `exp_avg_sq` match an independent
  call to that pinned functional oracle byte-for-byte.
- Tests prove zero-base scratch post-values themselves are signed additive m/d
  deltas. Nonzero scratch, custom-formula delta, post-minus-pre, direct m/d
  application to raw rows, or raw C/S moment accumulation fails.
- Inverse mapping uses frozen cloned scratch rows, exact ordered FP32 ops, and
  staged raw candidates. Weight and bias candidates, moments, and steps all
  validate before one logical commit.
- Commit-injection tests cover every touched parameter/state position. A
  verified rollback restores exact pre-cloned bytes and still raises fatal;
  rollback failure cannot produce an accepted checkpoint, finish, or public
  record.
- External state IDs, parameter-group values/keysets, optimizer state keys,
  tensor shapes/dtypes/counts, scheduler, scaler, and checkpoint keysets remain
  schema10-compatible. External `foreach`/`fused` remain `None`; only the
  adapter's private functional call resolves true/false explicitly. Schema11
  alone changes moment row interpretation.
- Tests distinguish raw parameter row 0/1 from moment row common/difference.
  Checkpoint bytes alone are not semantic proof; strict schema11, fresh
  provenance, source binding, transaction tests, and derived evidence are.
- No second optimizer call, persisted scratch/proxy, third moment, migration
  marker, coordinate tensor key, or conversion state is allowed.
- Fresh-only tests reject schema8/9/10/v14-v17 optimizer or actor checkpoints
  despite matching external shapes. Schema11 resume, parent, preload, and
  stage-to-stage restore also reject before state use.
- Terminal actor validation preserves/checks finite Adam state, expected step,
  external shape, and schema11 semantic binding as evidence; it opens no
  resume path.

### EMA, exact projection, and behavior invariants

- EMA reads inverse-mapped raw C/S online parameters, never m/d optimizer
  rows, and updates once at unchanged tau 0.1 after a successful Q step.
- Exact gate projection remains bit-equal to its target recomputed from the
  same stored raw EMA `CONTINUE-STOP` weight and bias difference, unchanged
  scale, and existing projection operation.
- The optimizer-coordinate adapter does not alter loss inputs, target,
  selected action, actor sampling/execution rule, epsilon, held-out split, or
  projection formula. Tests bind those mechanisms, not a learned-trajectory
  invariance claim.
- Q/EMA/projection counters remain lockstep and gate optimizer never steps.
  Projection post-error remains zero.

### Schema, artifacts, forgery, and legacy differential

- Strict built-in integer schema 11 activates seal1, exact Huber/common
  forward, and orthonormal optimizer coordinates. Bool, NumPy int, float,
  string, missing, 10, and 12 reject.
- Exact schema-11 config/actor/model surfaces remain229; projection exact209
  SHA `bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`;
  no 230th or persisted derived identity key exists.
- Bundle/ack remain exact five keys with gate11; history exact seven with
  canonical digest; seal exact10 and drain0/1 relations remain unchanged.
- Dedicated schema-11 final/completed/actor validators require exact loss,
  reconstruction, and optimizer-coordinate strings in derived evidence.
  Wrong/missing/extra identity, schema, surface, path/stage, tensor/metadata,
  Adam shape/step, history, logger/finish/private marker, or forged evidence
  fails closed.
- Exact-keyset tests compare the dedicated actor-only result, every applicable
  `actor_policy` evidence mapping, `stored_surface_identity` copies,
  authoritative final/public completed records, smoke
  `schema11_final_bundle_validation`, and fixed completed evidence and
  `resolved_profile_identity` with their schema-10 counterparts. Every mapping
  that directly carried the loss/reconstruction pair gains exactly
  `voc_q_optimizer_coordinates`; mappings that did not carry that pair keep
  their exact schema-10 keyset. Smoke local `resolved_identity` and outer
  `voc_checkpoint_resolved_identity` are exact-three containers
  `{config, actor_checkpoint, model_checkpoint}` whose inner mappings are each
  exact12; fixed `resolved_profile_identity` remains a distinct fixed-only
  path.
- Actor-only and final-bundle attacks inject each reserved derived key with
  correct, wrong, and null values at top level, nested `resolved_identity`,
  arbitrary nested mappings, and list/tuple-wrapped mappings in actor and
  model checkpoints. All reject before tensor use; a benign cyclic container
  terminates and remains valid if otherwise valid.
- Each schema-11 inner resolved identity is exact12. Schema10 remains exact11;
  schema9 remains its exact historical shape; schema8 and schemas at most 7
  remain historical. The smoke exact-three containers are not themselves
  exact12 identities.
- Differential fixtures prove schemas<=7 Huber/no-common/rowwise Adam,
  schema8 half-squared/no-common/rowwise Adam, schema9
  half-squared/common/rowwise Adam, schema10 Huber/common/rowwise Adam, and
  schema11 Huber/common/orthonormal-coordinate Adam. Every schema<=10
  byte/shape/behavior/path/output remains unchanged.

### Public, smoke, fixed, and TOCTOU ordering

- Schema-11 validation precedes evaluator-direct/downstream `_load_flags`,
  live spec/environment probe, reset/step, data, direct tensor load/use,
  rollout, output, or rewrite. Validator-internal safe bound deserialization
  is excluded from downstream counters.
- Stable config payload/hash classification and byte-aware loading resist
  deletion, replacement, cross-schema, alternate explicit config, and
  probe-to-load TOCTOU swaps. Missing/malformed bound hash fails before
  downstream use.
- Smoke binds prevalidation -> stable bytes -> private copy -> postvalidation
  and exact evidence/hash equality -> environment. It preserves inherited
  CLI override precedence for schemas at most 10; a schema11 checkpoint and
  alternate legacy payload, or legacy checkpoint and schema11 payload, fails
  before loader/environment.
- Fixed binds checkpoint-bound public import and requested profile -> schema11
  dispatch -> dedicated validation and exact evidence reuse/equality ->
  downstream use. The dedicated validator always executes even if a caller
  supplies prevalidated evidence.
- `v18-300k` accepts only exact primary. Wire/qualification, schema10/v17,
  legacy profile, preload/resume, wrong path/identity/private marker/mutation,
  or forged metadata/history/logger/completion record fails before output.
- Validator actor metadata is exact `{key,dtype,shape,numel}` per unique state
  key with built-in types and correct product. Publication-history digest is
  independently recomputed from canonical JSON. Fixed evidence top-level
  keysets, stage member types, path relations, checkpoint hash/size records,
  logger equality, and private-marker paths are exact.
- Historical checkpoint-bound modules lacking schema11 APIs remain valid only
  on their unchanged compatible paths; requested `v18-300k` without required
  APIs fails before downstream use. No failed historical version gains
  evaluation authority.

### CLI, fresh-before-I/O, lifecycle, and no retry

- Real `create_setting(save_flags=False)` binds all three exact v18 tuples,
  exact229/209, schema11, seal1, schedule100M, fresh inputs, paths/resources,
  and W&B modes. Derived identities never appear in CLI/YAML/config.
- Wrong xpid/whitespace, seed, horizon, warm-up, unroll, W&B, schema, path,
  surface, identity, preload, resume, resource, retry, or topology fails
  before run/environment action.
- With `ckp=true`, malformed V18 intent expressed as built-in/string subclass,
  NumPy string, bytes/NumPy bytes, `Path`, custom `PathLike`, or custom
  stringification rejects before config open and before run-dir creation.
  Exact lexical detection never relaxes the final built-in-string check.
- Training with `ckp=true`, preload, parent, resume, or cross-schema state is
  rejected before restoration. Terminal checkpoint preservation is evidence,
  not resume authority.
- Real-Ray tests inherit barrier, claim/seal/drain0/1, logger two-phase,
  timeout/abort/kill, no-restart/no-task-retry, private cleanup, finish, and
  process/GPU cleanup. Schema11 final validation occurs before
  complete-success; validator failure yields abort/no finish.
- Immutable snapshot tests require exact source/data manifests, independent
  inodes and modes, empty runs, snapshot-only production cp310 import, no
  cache/cp312/worktree module, focused/adversarial/full suites, and unchanged
  posthash.

## Final claim boundary

V18 requires, in order, one accepted immutable snapshot, one accepted
mechanics-only seed-1 wire, one accepted fresh seed-1 100k qualification, one
accepted fresh seed-5 300k primary, and that primary's one accepted
`v18-300k` fixed confirmation. Any failure is permanent for v18. No later
mechanics pass, diagnostic, seed, checkpoint, sibling protocol, or version
can retroactively change the permanent v14, v15, v16, or v17 failures or any
v18 stage decision.

This authoritative preregistration does not itself perform or launch an
implementation, snapshot, experiment, retry, primary, fixed evaluation, or
evaluator action. Any later action must satisfy the sequential release gates
above on these exact frozen bytes.
