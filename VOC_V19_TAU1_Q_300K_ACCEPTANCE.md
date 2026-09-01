# Enduro VoC-v19 exact tau-one EMA Q preregistered acceptance

This protocol is frozen after the sole VoC-v18 seed-1 100k qualification
failed and before any v19 implementation edit, immutable snapshot, wire,
qualification, primary, or fixed evaluation. V14, v15, v16, v17, and v18
remain permanent failures. V19 is the immediate successor of v18 and
gate-policy schema 11. It is one separately named, prospective, one-of-one,
no-retry experiment that changes exactly one existing algorithmic value:

1. for gate-policy schema 12 only, the existing
   `voc_gate_target_tau` changes from the inherited built-in float `0.1` to
   the exact built-in float `1.0`.

Schema 12 otherwise retains schema 11's exact beta-1 Smooth-L1
selected-action TD loss, common-mode online and EMA Q reconstruction, raw
`[CONTINUE, STOP]` head parameters and forward rows, orthonormal
common/difference Adam gradient and moment semantics, raw gradient norm and
single clip, optimizer/scaler/scheduler transaction, behavior epsilon,
update cadence mechanism, raw EMA operation order, exact gate projection,
actor-policy barrier/history, ModelBuffer seal, logger, public validation,
populations, and every numeric gate.

The global/default YAML value remains `0.1`. When tau is absent, schemas 1
through 5 retain that historical default; when provided, they retain their
full historical finite non-Boolean numeric acceptance, normalization to
float, `0 < tau <= 1` range, coercions, and error behavior. Atomic schemas 6
through 11 continue to resolve, persist, validate, and execute exact `0.1`
byte- and behavior-identically. Only an exact schema-12 stage profile
atomically resolves the existing field to built-in float `1.0`; no new
CLI/YAML/config/checkpoint field or derived identity field is introduced.
Versioned evidence keeps the inherited container cardinalities under the
required schema-specific route-label substitution.

The inherited raw EMA calculation remains the exact implementation rule:

```python
candidate = (1.0 - tau) * old_ema + tau * poststep_online
```

At finite `tau=1.0`, its result must be `torch.equal` to the corresponding
finite post-step online weight or bias tensor before it is committed. The
inherited arithmetic expression and operation order are normative. Schema 12
does not add a `copy_` shortcut, direct-online loss route, or special EMA
kernel. `torch.equal` is a value/tensor equality requirement, not a byte-view
or signed-zero-bit equality claim: inherited arithmetic may map a stored
negative zero to positive zero while still satisfying `torch.equal`.

The delta is a falsifiable lag ablation, not a causal conclusion or pass
prediction. V18's frozen online held-out RMSE is already
`0.6212603749561236 > 0.5`; eliminating the observed online-to-EMA sign lag
does not imply an RMSE pass or a changed trajectory in any favorable
direction. No denominator relaxation, pseudocount, threshold change,
replacement seed, retry, primary, fixed evaluation, or second lever is
authorized.

## Permanent v18 qualification failure

The only v18 qualification was
`enduro-voc-v18-orthocd-adam-eps25-seed1-qual-fresh-100k`, launched once from
immutable snapshot `/tmp/di-voc-v18-orthocd-adam-eps25-final-2Ii39u`. It was
fresh, seed 1, W&B-enabled, schema 11, seal schema 1, and bound to the exact
229-key surface and schema-11 209-key projection. The driver exited zero.
Actor, Huber/common online Q, orthonormal m/d Adam, raw EMA, exact projection,
actor-policy barrier/history, sealed ModelNet input, W&B two-phase completion,
public finish, manifests, process, Ray, and GPU validation all passed. Those
mechanics cannot rescue either numeric failure.

The frozen actor log is `logs.csv`, SHA-256
`02de18b452ecf343b664ff46b2a9251cc4d2b20cdde1848442df61ff90440f11`,
with 156 complete rows and 922 unique ordered columns. The canonical
qualification population contains exactly 36 complete rows under
`70000 < real_step <= 100000`, from 70624 through 99360. The sole row at
100064 is excluded as an overshoot.

V18 failed exactly two of the unchanged 15 CSV-observable hard gates:

1. trailing-five denominator validity; and
2. held-out EMA selected-action TD RMSE.

Every direction, window, margin, support-fraction, negative-run, saturation,
online/EMA sign-agreement, and CSV safety gate not named above passed. There
was no third numeric failure.

### V18 trailing-five denominator failure

There are 32 eligible trailing-five endpoints. Twenty-nine have both sign
denominators. Exactly three have zero EMA positive-sign support and are
therefore undefined and invalid:

| Endpoint step | EMA positive support | EMA negative support |
| ---: | ---: | ---: |
| 74704 | 0 | 5906 |
| 75744 | 0 | 5891 |
| 76784 | 0 | 5893 |

Undefined gaps are not zero, nonnegative, negative, removable, imputable, or
replaceable. No pseudocount, smoothing, endpoint deletion, shortened window,
post-hoc start step, or denominator relaxation is permitted. Among the 29
defined endpoints, the maximum consecutive negative pooled-gap run is zero;
that separate passing gate does not repair denominator invalidity.

The same trailing-five rule recomputed from frozen online-Q sign counts is
defined at all `32/32` endpoints. Online positive support was zero on logged
rows 61696 through 72672 and had its first canonical reappearance after that
collapse at 73680. EMA positive support was zero on logged rows 64496 through
76784 and reappeared at 77696. The observed reappearance lag is 4016 real
steps. All three EMA-invalid endpoints lie after the online reappearance and
before the EMA reappearance.

This observation directly motivates a tau-one lag ablation. It does not show
that smoothing caused the zero-support interval, because state, actions,
targets, support, update timing, and closed-loop behavior are endogenous.
Tau one changes the future trajectory and cannot be evaluated by substituting
frozen online counts for a new run.

### V18 held-out RMSE failure

The pooled held-out EMA selected-action TD RMSE failed:

```text
sqrt(2283.911753300114 / 5915) = 0.6213871746707746 > 0.5.
```

At fixed support the unchanged threshold SSE is
`5915 * 0.5^2 = 1478.75`. Observed excess SSE is
`805.161753300114`, requiring a nonnegative fractional SSE reduction of
`0.35253627997522413` merely to meet the ceiling at the same support. Frozen
windows are:

| Window | Rows | Held-out support | EMA RMSE | EMA SSE |
| --- | ---: | ---: | ---: | ---: |
| W1 `(70000,80000]` | 10 | 1487 | `0.5198341483850172` | `401.8283546970108` |
| W2 `(80000,90000]` | 13 | 2256 | `0.6682381798094105` | `1007.3993497384664` |
| W3 `(90000,100000]` | 13 | 2172 | `0.6345936073099058` | `874.6840488646368` |
| Full `(70000,100000]` | 36 | 5915 | `0.6213871746707746` | `2283.911753300114` |

The online-Q held-out companion is `0.6212603749561236`. It also fails the
same 0.5 ceiling. Online/EMA Full RMSE proximity is evidence that tau-one has
no frozen aggregate-RMSE pass signal, even though online counts remove the
three frozen denominator failures. Neither frozen online metric is a
counterfactual schema-12 result.

V18 failure permanently forbids a v18 primary, `v18-300k` fixed evaluation,
resume, extension, alternate seed, selected checkpoint, retry, or rescue.
V19 is not a retry of v18 and may not load any v14-v18 checkpoint, optimizer,
buffer, actor version, observation, or runtime state.

## Frozen v18 failure artifacts and mechanics closure

The v18 qualification run directory is
`/tmp/di-voc-v18-orthocd-adam-eps25-final-2Ii39u/runs/enduro-voc-v18-orthocd-adam-eps25-seed1-qual-fresh-100k`.
Its exact 14 regular files are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `ckp_actor.tar` | 28299091 | `b07a7318145e668d86807b413d418e955556fe7f178bf4c2116b0cb972b1f2fc` |
| `ckp_actor.tar_step_480` | 28259637 | `7890c152c388d77d730fde5afb3b610db5b927153f91d41ce57acf4d43bf67e7` |
| `ckp_actor.tar_step_100064` | 28306795 | `1bb2f11271c2b511ca4fdc95d12a74848fe2ffafbfe02508fd85b7410690ad9a` |
| `ckp_model.tar` | 133359317 | `0cd3652a212f8085007ded99301f19d5f3c5094676c0c97753aa60d79257c04c` |
| `ckp_model.tar_step_10000` | 133380573 | `fd7da994a23a2beff4e6d85fc65482de307d95eb4e5af2ff3e02d96325ad8a55` |
| `ckp_model.tar_step_100064` | 133359317 | `0cd3652a212f8085007ded99301f19d5f3c5094676c0c97753aa60d79257c04c` |
| `config_c.yaml` | 7627 | `ba9a3ce191a96be167b25878f2282d6f354d0a89aeeb94a60c3cfa1fad7ce51d` |
| `finish` | 3354 | `9d3bb1eecad4811920e596f1e1c829a716970d8952ff63b6d544ba5b9871b5f4` |
| `logs.csv` | 1544904 | `02de18b452ecf343b664ff46b2a9251cc4d2b20cdde1848442df61ff90440f11` |
| `logs_model.csv` | 121888 | `09ce87d4b23b661665f40714f35b21347beef2f302768af2e10f58198354862e` |
| `meta.json` | 15309 | `f09aa606c3cddb8a473943ea7347c5b7a7ed61da484aff71a9a77106b4b90c6c` |
| `meta_model.json` | 15309 | `c494451c418a9232b8d1bcc14ee00b2447809e34ddec681162259845b82d59fa` |
| `out.log` | 56891 | `8c943cf62464f35fd5f7226027a6bf4b5573c12263fb23966d10576906757f80` |
| `out_model.log` | 159235 | `a96ef26d1798f6a285388ed40dac19140e12364955d3c454c41990706b297566` |

The canonical tree binding is the SHA-256 of the 1173-byte concatenation of
C-sorted `sha256sum` records formatted exactly as
`{lowercase_sha256}  ./{basename}\n`; it is
`14a00e96656bccbce03a51654e07d5d18ef02ae3b93a0b3dc74c39cf86a64eba`.

The sole qualification runtime is `/tmp/v18qual-0dIYy2`. Its exact launch
provenance is `launch_provenance.json`, 2627 bytes, SHA-256
`c356a06868df1841bb27c6586ad5e4ba64ffd552e7ce73e8058163be40f4494d`.
The driver log is 415561 bytes, SHA-256
`9c18ca46896237453c6f048f8c6b3ef147bf3d577ce6eb061c7f2a4ba94c0f14`;
the driver wrapper is 4268 bytes, SHA-256
`88f9503a6493175bf66d069e8026e7034414f0ebd18251150df0a94c75f66c00`;
and `driver.exit_code` is exact `0` plus LF, SHA-256
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
The one-attempt counter `attempt.count` is exact `1` plus LF, SHA-256
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`.

The authoritative v18 preregistration is
`VOC_V18_ORTHOCD_ADAM_Q_300K_ACCEPTANCE.md`, SHA-256
`abfdc806528b34fdf96d67a72accb81ec2ea6b5cd59693439b272f8963621d56`.
The immutable v18 snapshot source manifest is
`f3f6ea5d066076adc9fb5286e83c01ce8735a12e34a2a580fe4115f18ca110c8`
with 1067 entries. The unchanged data manifest is
`23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343`
with 11 entries. The loaded cp310 extension SHA-256 is
`1d4c5026d2a6c002a13829e162428505b30a65bf2af1968f6e982dcfcc16b232`.

Qualification preflight bound exact229, schema 11, seal 1, derived identity
count 12, command SHA-256
`3444766b02fdabc5047daac6a8ba325702f889f3d2021a7613b628bae3b6f3a7`,
complete-surface SHA-256
`3e79420d5580f3d5d9566761731f6ca4bbc360b349f44304a1019631f7c42314`,
and schema-11 209-key projection SHA-256
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.
The accepted prerequisite wire tree is
`937c617793579f15b1d374ef6b9e21361a2add78f87f40f957a7db6aad67e7a6`;
its public evidence SHA-256 is
`3d86f5d417f22f2e6b3b8f0d03f71d445498c28507d5fafdedbded6e7e9df32e`.

The production runtime was PyTorch `2.13.0+cu130` with
`torch/optim/adam.py` SHA-256
`bde360b0bb9b7869f1cec04a3b41a90b8eabb84a613787d97b88d87f2f3ae1ec`
and `torch/amp/grad_scaler.py` SHA-256
`97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039`.
These remain v19 snapshot and wire provenance requirements, not new config
keys.

Actor terminal policy version, successful Q updates, raw EMA updates, and
exact projections were 156; history length was 157 and terminal ack was
verified. Actor-state SHA-256 was
`1aefd1e962594265762fcef492b3d5faeed1a9be182b9cec32190a68f28cdac4`;
publication-history SHA-256 was
`d5b21b380e8d8867a19ed3b8e1ebcd6bd72ef1b65d716b7f01fed92d4adea83e`;
logger-request SHA-256 was
`ed585fe892356e18c66d124682484bb9a03e3f1f9ad0b0fd1b775e1d93716817`.
W&B completion was required, acknowledged, and privately cleaned. Finish was
complete/schema 1. Model seal, terminal save, source/data rehash, run tree,
process, Ray, and GPU closure passed. These mechanics facts are integrity
evidence, not numeric acceptance.

## Frozen V14-V18 RCA and A/B consensus

The authoritative executed technical notebook is
`notebook/v14_v15_v16_v17_v18_qualification_comparison_rca.ipynb`, SHA-256
`f1f4f9d5c8ffb48d41d63ee8d7d17bc338738c98e52409d327b1554c8d8c736f`,
934128 bytes and 1276 lines, with 17 cells, eight continuously executed code
cells, zero errors, and three visually QA-passed charts. The frozen
machine-readable report is
`notebook/v14_v15_v16_v17_v18_qualification_rca_report.json`, SHA-256
`4e56f36911535336fb9adcb4bc5f8b63589cf6fa9aed395498c9db7c808671e8`,
433694 bytes and 7426 lines. Two independent read-only audits found P0=0 and
P1=0. The report also passed its MCP validator and one stakeholder render.
Both artifacts must retain these exact bytes.

The notebook and report independently reproduce the raw CSV populations,
all 15 gates, windows, RMSE/SSE, sign extinction and reappearance, online/EMA
comparisons, cadence, norm/clipping/composition diagnostics, checkpoint m/d
state, source/data/protocol hashes, and mechanics closure. They keep observed
evidence separate from causal hypotheses and do not authorize implementation
or launch.

The five observed qualifications are:

| Version | Sole lineage delta | Held-out EMA RMSE | Undefined trailing endpoints | Decision |
| --- | --- | ---: | ---: | --- |
| v14/schema7 | sealed input baseline | `0.5247954789453232` | 0 | fail RMSE |
| v15/schema8 | half-squared TD | `0.5637126651551873` | 0 | fail RMSE |
| v16/schema9 | common reconstruction | `0.643399622874774` | 5 | fail RMSE + denominator |
| v17/schema10 | beta-1 Huber restored | `0.4739621233029751` | 7 | fail denominator |
| v18/schema11 | orthonormal m/d Adam state | `0.6213871746707746` | 3 | fail RMSE + denominator |

These are five separately generated, single-seed, policy-coupled on-policy
trajectories. They are not paired, randomized, replicated, or a factorial
experiment. State, action, target, support, cadence, and optimizer histories
diverge, so they do not identify a causal loss, reconstruction, optimizer,
EMA, or cadence effect.

The independent A/B consensus ranks exact existing tau `0.1 -> 1.0` first as
the narrowest falsifiable next delta. It mechanically makes the stored raw
EMA tensors value-equal to the raw online tensors at each successful
post-update boundary while leaving the V18 objective and optimizer coordinates
intact. It does not predict the timing or sign of a future logged
reappearance. Deterministic real-step cadence ranks second only after its
quantum, backlog, credit, scheduling, and restore semantics are preregistered.
Tail/clip/coordinate-sensitivity telemetry ranks third before another behavior
change. The ranking is not a causal claim or a pass forecast.

The RCA artifacts use an “exact post-Q copy” shorthand for the tau-one
value-synchronization hypothesis. They are evidence and ranking authorities,
not implementation authorities. Under this preregistration that shorthand
means the `torch.equal` post-update boundary below; it never authorizes a
direct `copy_` calculation, byte-copy contract, or signed-zero-bit equality.

## Sole schema-12 tau delta

The only schema-12 algorithm value change is:

```text
voc_gate_target_tau: built-in float 0.1 -> built-in float 1.0
```

The field already exists in YAML, CLI parsing, the 209-key stage-neutral
projection, the 229-key persisted surface, checkpoint protocol evidence, and
runtime validation. No key is added, removed, renamed, moved, or repurposed.

The source default and `VOC_PROTOCOL_DEFAULTS` remain `0.1`. Schemas 1 through
5 retain their current historical non-atomic requirements and baseline
behavior; atomic schemas 6 through 11 retain their exact-0.1 requirements and
baseline bytes. Exact schema-12 intent must be recognized from the V19 lexical
xpid and complete atomic protocol before any persisted config open,
run-directory creation, checkpoint load, or environment action. During exact
schema-12 stage resolution, the inherited default is replaced by the exact
built-in float
`1.0` before complete-surface construction and persistence. After resolution,
every config, actor metadata, ModelNet metadata, embedded flag surface, and
checkpoint protocol copy must carry exact built-in float `1.0`.

Strict value/type means `type(value) is float`, the value is finite, and it
equals `1.0`. Integer `1`, Boolean `True`, NumPy float or integer, decimal,
string, `0.1`, positive zero, negative value, subunit alternative, value above
one, infinity, NaN, missing value, or defaulted schema-11 value fails closed.
The resolver may install the frozen stage value; it may not coerce an
externally persisted or validator-facing value.

### Exact inherited EMA arithmetic

The schema-12 online Q and old EMA tensors are the same raw FP32 C/S weight
and bias tensors as schema 11. Both are required finite before candidate
construction. For each tensor the inherited operation is exactly:

```python
candidate = (1.0 - tau) * old_ema + tau * poststep_online
```

with `tau` equal to built-in float `1.0`. The implementation retains the
existing expression, tensor dtype/device, evaluation order, candidate
validation, and commit order. It does not use `copy_` as an alternate
calculation, `lerp`, a fused kernel, reordered additions, host conversion,
or loss-time online-Q substitution.

After candidate computation and finite validation, each candidate must
satisfy `torch.equal(candidate, poststep_online)`. Only then are weight and
bias committed, preserving the inherited validate-both-before-mutate rule.
The equality is numeric tensor equality. A test with negative zero explicitly
demonstrates that a candidate can be `torch.equal` to online while its byte
view or sign bit differs. No byte equality or checkpoint-byte invariance is
claimed. The inherited EMA weight and bias remain distinct, non-aliased
storage from the online head.

After a successful Q optimizer step, and after the inherited optional gate
and main actor step call sites, raw EMA updates once with this exact
arithmetic and exact projection runs once from the newly stored raw EMA
`CONTINUE-STOP` difference. Q/EMA/projection counters remain lockstep.

The gate and Q computations for the minibatch that causes the update still
use the inherited pre-step EMA target. Schema 12 does not use current online Q
directly at loss time or retroactively alter the sampled batch. The tau-one
candidate becomes the target for later publication/rollout under the inherited
one-update-delayed sequence. Between successful Q updates, EMA and projection
remain unchanged.

Q optimizer success, not main-actor optimizer success, remains the inherited
condition for advancing raw EMA and projection. The optional gate step and
main actor step still occur before EMA at their inherited call sites. An
independent main-actor AMP skip does not suppress a successful Q target
update, although any live skip remains an acceptance failure. A recoverable Q
AMP skip or unsupported/all-held-out Q batch has no Q optimizer commit and
does not advance live Q parameters, Adam moments/step, EMA, projection, Q
counters, or Q scheduler; raw `.grad` and later held-out/loss evidence retain
their inherited branch-specific semantics. A transform, functional-Adam, or
staged-validation failure before live commit, and a commit exception whose
rollback verifies, likewise advances none of those committed states and is
fatal with no finish. Rollback failure is fatal with no accepted artifact but
carries no unchanged-state promise. A fault after a successful Q commit—
including a later optional-gate, main-actor, EMA, equality, or projection
fault—is fatal with no finish and does not undo or falsely label the already
committed Q update as rolled back.

### No other semantic delta

Schema 12 retains, without value or order change:

- beta-1 Smooth-L1 selected-action TD loss with FP32 operands,
  `reduction="none"`, train mask multiplication, sum, and inherited supported
  mean telemetry;
- common reconstruction
  `V_det.unsqueeze(-1) + raw.mean(-1,keepdim=True) + raw - sum(p_det*raw)`;
- raw `[CONTINUE,STOP]` parameter, state-dict, forward, EMA, and projection
  rows;
- the binary32 `s=0x3f3504f3` orthonormal gradient transform, pinned
  functional Adam staging, m/d moment rows, inverse-mapped raw deltas, atomic
  commit and rollback;
- raw preclip global norm, one raw C/S clip, AMP/scaler branch, main actor and
  Q schedulers, learning rates, Adam betas/epsilon/weight decay, and update
  cadence mechanism;
- training epsilon 0.02, executed epsilon 0.25, gate temperature, exact
  projection scale and formula, support/held-out split, action mapping,
  replay, batch, unroll, warm-up, and schedule total;
- every architecture, parameter, buffer, tensor, optimizer, scheduler,
  scaler, checkpoint, telemetry, bundle, logger, and seal keyset and shape;
  and
- every public/smoke/fixed ordering rule, population, threshold, and
  one-attempt release gate.

No new telemetry measures a tau delta, copy count, EMA byte equality, lag,
or direct-online path. Existing EMA/projection counts and tensor validation
are sufficient mechanics evidence.

## Normative inheritance from v18

The frozen v18 protocol
`VOC_V18_ORTHOCD_ADAM_Q_300K_ACCEPTANCE.md`, SHA-256
`abfdc806528b34fdf96d67a72accb81ec2ea6b5cd59693439b272f8963621d56`,
is incorporated verbatim except for these closed substitutions:

- experiment names and fixed profile change from v18 to v19 identities;
- gate-policy schema 11 changes to schema 12;
- the existing schema-12 stage profile resolves
  `voc_gate_target_tau=1.0` instead of the inherited `0.1`;
- the 209-key projection retains its exact keyset but obtains the new frozen
  value digest specified below; and
- versioned dedicated validators, public records, labels, and profile routes
  make only the schema-11-to-12 discriminator substitution.

Schema 11's loss, common reconstruction, raw head, orthonormal optimizer
coordinates, transaction, raw EMA expression, exact projection, and all
state shapes remain unchanged. The exact three derived identities remain:

```text
voc_q_regression_loss="smooth_l1_beta1"
voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
voc_q_optimizer_coordinates="orthonormal_common_difference_adam"
```

They remain derived-only where already derived-only. Schema 12 adds no fourth
derived identity and does not repeat `voc_gate_target_tau` as a new derived
field; tau is already bound through the inherited persisted/config protocol
surface and its hashes.

The schema-11 adapter remains sealed inside existing
`thinker/thinker/learn_actor.py`. Schema 12 reuses the exact adapter and the
same raw parameter group/state-dict. The production pins remain PyTorch
`2.13.0+cu130`, Adam source
`bde360b0bb9b7869f1cec04a3b41a90b8eabb84a613787d97b88d87f2f3ae1ec`,
and GradScaler source
`97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039`.
There is no optimizer-state conversion or new source module.

No second lever may change loss/beta/coefficient, reconstruction, raw-head
representation, orthonormal scale or operation order, Adam functional call,
learning rate, betas/epsilon/weight decay, scheduler, scaler, raw clip,
projection, epsilon, temperature, action weighting, normalization, shared
gradient, auxiliary loss, telemetry, replay, batch, unroll, warm-up,
held-out split, barrier, seal/drain, logger, terminal order, update cadence,
retry, seed, checkpoint selection, threshold, population, denominator,
pseudocount, or fixed rule.

Unchanged rules do not imply an unchanged learned path. Tau one changes the
gate target presented to future rollout after each successful Q update and
can change projected gate, sampling, trajectory, cadence, support,
composition, errors, and every learned metric. No direction or pass result is
invariant or predicted.

## Exact schema lineage

Schemas 1 through 5 retain byte-, shape-, return-, path-, and
behavior-identical historical beta-1 Smooth-L1, centered/no-common Q,
rowwise C/S Adam, and their historical tau semantics: absent tau defaults to
0.1, while supplied finite non-Boolean numeric values are normalized under
the unchanged `0 < tau <= 1` range and historical coercion/error behavior.

Schemas 6 and 7 retain byte-, shape-, return-, path-, and
behavior-identical historical beta-1 Smooth-L1, centered/no-common Q,
rowwise C/S Adam, exact atomic tau 0.1, and historical evidence shapes.

Schema 8 retains byte-, shape-, return-, path-, and behavior-identical
half-squared TD, centered/no-common Q, rowwise C/S Adam, tau 0.1, and exactly
its historical one-field derived loss identity.

Schema 9 retains byte-, shape-, return-, path-, and behavior-identical
half-squared TD, common Q, rowwise C/S Adam, tau 0.1, and exactly its
historical loss/reconstruction identity pair.

Schema 10 retains byte-, shape-, return-, path-, and behavior-identical
beta-1 Smooth-L1, common Q, rowwise C/S Adam, tau 0.1, and exactly its
historical loss/reconstruction identity pair.

Schema 11 retains byte-, shape-, return-, path-, and behavior-identical v18
beta-1 Smooth-L1, common Q, raw C/S parameters, orthonormal m/d Adam state,
tau 0.1, exact projection, and exact three derived identities.

Schema 12 retains schema 11 in every respect except the existing effective
tau value becomes 1.0. Its derived identity remains the same exact three
strings and 12-key resolved-identity shape. Every schema at most 11 stays on
its source-visible historical path; a schema-12 implementation may not
refactor an older route in a way that changes bytes, values, exceptions,
validation order, output, or behavior.

## Exact schema-12 identity and persisted surfaces

V19 uses strict non-Boolean built-in Python integer
`voc_gate_policy_schema_version=12` and strict built-in float
`voc_gate_target_tau=1.0`. Boolean, NumPy integer/float, built-in integer tau,
float/string schema, missing/defaulted value, schema 11, tau 0.1, or any
other value/type is not schema 12.

Configuration, actor checkpoint, and ModelNet checkpoint each retain the
exact 229-key surface:

```text
229 = 209 stage-neutral keys
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

Among the exact ten protocol values, only gate schema becomes 12. Among the
209 stage-neutral values, only the existing `voc_gate_target_tau` becomes
1.0. The six stage and four path values take their exact closed V19 values;
those identity/path substitutions are not a second algorithm delta. Execution
epsilon remains 0.25; barrier true; bundle schema 1; timeout 120.0; Ray actor
restart/task retry zero; actor AMP initial scale 32; training barrier runtime
true; and model-input-seal schema 1.

The schema-12 209-key projection has exactly the schema-11 projection keyset
and values except `voc_gate_target_tau=1.0`. Its canonical JSON uses sorted
keys, `ensure_ascii=True`, separators `(',', ':')`, `allow_nan=False`, UTF-8,
and no trailing LF. It is 4457 bytes with SHA-256
`ad22b91fdd06a30ac7f53c0135b32fac2530687c3c36dad5dccf06d700f83f82`.
The schema-11 historical `v12_projection` digest
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`
remains authoritative only for atomic schemas 6 through 11; schema 12 must
not claim it. Schemas 1 through 5 retain their historical non-atomic surfaces.
Each V19 complete-surface digest is newly computed from its exact stage,
paths, command, schema 12, and tau-one values before launch.

Every actor-policy bundle retains exact keys
`{bundle_schema_version, policy_version, terminal, gate_schema,
actor_state_dict}`, bundle schema 1, and gate schema 12. Every ack retains
exact keys `{bundle_schema_version, gate_schema, rank, policy_version,
terminal}` and gate schema 12. Publication history remains exact seven-key
events `{predecessor_version, policy_version, publication_count, terminal,
ack_ranks, expected_ack_count, state_sha256}` and gains neither schema nor
tau identity.

Model-input-seal schema remains strict integer 1. Terminal ModelNet evidence
retains exact ten fields and all inherited seal-one, drain-zero/one,
pre/final count, late-write-zero, abort-zero, durable-save,
complete-success, and actor-before-model-before-finish relations.

Core source must expose exact schema-12 API names:

```text
VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION
VOC_GATE_POLICY_SCHEMA12_STAGE_PROFILES
_validate_schema12_stage_profile
_validate_schema12_complete_surface
_validate_schema12_protocol_flags
validate_schema12_final_bundle
validate_voc_schema12_final_actor_checkpoint
```

`VOC_GATE_POLICY_TAU1_Q_SCHEMA_VERSION` is exact built-in integer 12. Public
completion must expose and call dedicated
`validate_schema12_completed_bundle`; payload-aware dispatch must have a
strict schema-12 route. Fixed evaluation must expose dedicated
`validate_v19_final_bundle`. Shared legacy dispatch may recognize schemas 6
through 12, but every dedicated schema-12 API rejects every non-12 and
wrongly typed value. Schemas at most 11 never call those dedicated routes.

Each schema-12 inner JSON-safe resolved identity retains exactly 12 keys:

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

The last three values are the unchanged schema-11 derived strings. The
other exact values are `key_count=229`, `v12_projection_key_count=209`, the
new schema-12 projection digest above, `gate_schema=12`,
`voc_gate_policy_schema_version=12`, and
`voc_model_input_seal_schema_version=1`, plus the exact stage, paths, and
freshly computed complete-surface digest. Authoritative final, actor-only,
public, smoke, and fixed evidence retain their exact schema-11 container
cardinalities and one-to-one mapped keyset shapes, substituting schema12/v19
route labels, schema value 12, stage/path/complete hashes, and tau-bound
surfaces only. Under that schema-specific route-key substitution, no mapping
gains or loses a field.

Smoke local `resolved_identity` and outer
`voc_checkpoint_resolved_identity` remain exact-three containers
`{config, actor_checkpoint, model_checkpoint}` whose inner values are exact
12-key identities. Fixed `resolved_profile_identity` remains a fixed-only
container and replaces the v18 schema11/profile values with exact
schema12/v19 values without a keyset change. Lifecycle `actor_policy`
evidence that never carried derived identity retains its inherited exact
keyset.

The three derived names remain reserved from persisted actor/model checkpoint
surfaces at any mapping/list/tuple depth. Dedicated validators recursively,
cycle-safely reject correct, wrong, null, or forged presence before tensor
use. Tau remains permitted only in its inherited existing protocol locations
with exact value/type; adding a duplicate tau-derived identity key is not
allowed. Schema at most 11 evidence and output bytes remain exact.

For schema 12 with a strict built-in integer EMA update count greater than
zero, every dedicated actor-only, final-bundle, public-completed, smoke, and
fixed validator must compare the stored raw EMA weight and bias separately
with the stored online raw Q-head weight and bias using `torch.equal`. Both
comparisons must pass
before downstream tensor use, flag load, environment action, rollout, or
output. Validator-internal bound deserialization and comparison are allowed
and are not downstream use. A weight mismatch, bias mismatch, or both
mismatches fail closed.
This is numeric tensor equality only: signed-zero byte/sign-bit differences
remain valid, and no validator substitutes byte comparison. The predicate is
activated by a nonzero update count and does not alter schema-at-most-11 or
fresh zero-update validation routes.

External `voc_optimizer_state_dict` IDs, groups, keys, state counts, tensor
shapes/dtypes, and schema-11 m/d row semantics are unchanged. Schema 12 may
not reinterpret, migrate, or convert moments. The schema discriminator,
fresh provenance, implementation source, and derived evidence remain the
semantic authority because external tensor shapes alone are ambiguous.

## Fresh-only state and exact stages

Every v19 stage is fresh: `ckp=false`; `preload`, `preload_actor`, and
`voc_parent_checkpoint` are empty; parent-update count is zero; actor-policy
version starts at zero; and raw online Q, raw EMA Q, projected gate, Adam
state, ModelNet, buffers, and seal state start from unchanged fresh state. No
state crosses stages.

No v14-v18 checkpoint or rowwise/m-d Adam state can seed schema 12. No
schema-12 wire, qualification, or primary checkpoint may seed another v19
stage. Resume is forbidden even from schema 12. Terminal state preservation
is evidence only, never restore authority.

The exact schema-12 lexical prefix is
`enduro-voc-v19-tau1-orthocd-adam-eps25-`. Schema-12 intent is recognized
before any persisted config open, run-dir creation, checkpoint load, or
environment action. Malformed V19-prefixed
xpid claims through built-in strings, string subclasses, NumPy strings,
bytes-like UTF-8 values, `os.PathLike`, or other lexical representations are
classified as schema-12 intent and rejected unless the exact stage xpid is a
built-in string. Conversion/decoding failure also fails before I/O. This
early classifier does not coerce or accept a value.

The only valid stage tuples
`(xpid, base_seed, total_steps, model_warm_up_n, actor_unroll_len, use_wandb)`
are exactly:

- (`enduro-voc-v19-tau1-orthocd-adam-eps25-sentinel-wire1200`, 1, 1200,
  512, 41, false);
- (`enduro-voc-v19-tau1-orthocd-adam-eps25-seed1-qual-fresh-100k`, 1,
  100000, 10000, 201, true); and
- (`enduro-voc-v19-tau1-orthocd-adam-eps25-seed5-strict-fresh-300k`, 5,
  300000, 10000, 201, true).

Each tuple additionally requires effective
`voc_gate_policy_schema_version=12`,
`voc_gate_target_tau=1.0`, barrier true, seal 1, fresh inputs, exact resources,
and the complete atomic schema-12 surface. Each xpid is an exact built-in
string; numeric members are strict non-Boolean built-in integers; W&B is an
exact built-in Boolean; tau is an exact built-in float. No trimming,
coercion, alias, or alternate tuple is allowed.

All stages retain `schedule_total_steps=100000000`, exact Enduro paths and
network configuration, CUDA devices 0 and 1 only, and Ray two GPUs/16 CPUs.
Wire is W&B-disabled; qualification and primary require authenticated W&B
and the inherited request/ack/private-cleanup/public-finish sequence.

Every stage has exactly one attempt. Wire cannot continue into qualification.
Qualification is one separate fresh seed-1 run. Primary is one separate fresh
seed-5 run and may start only after every qualification gate passes. There is
no resume, preload, extension, fallback seed, retry, duplicate xpid, selected
checkpoint, replacement run, or cross-schema state. Any failure permanently
ends v19 at that stage.

## Sequential release gates

The only release order is:

1. Implement only schema-12 tau-one stage resolution, strict propagation,
   inherited EMA arithmetic validation, versioned evidence/public/profile
   routing, and frozen tests. Freeze all bytes and pass two independent
   code/contract audits.
2. Build a fresh inode-independent immutable snapshot from the authoritative
   v18 source/data baseline plus an exactly enumerated v19 overlay. Pass two
   independent manifest, mode, cp310, schema, test, and posthash audits.
3. Run exactly one fresh seed-1 1.2k wire. Decide mechanics only.
4. Only after wire passes, run exactly one fresh seed-1 100k qualification
   with every inherited gate unchanged.
5. Only after qualification passes every gate, run exactly one fresh seed-5
   300k primary with every inherited gate unchanged.
6. Only after primary passes may its terminal checkpoint receive one fixed
   confirmation under exact profile `v19-300k`.

No primary or fixed evaluation may launch speculatively or after failed
qualification. Pong, Space Invaders, alternate seed, shortened run, or
diagnostic fixed evaluation remains forbidden until an accepted v19 Enduro
claim exists.

## Integrity-wire acceptance

Wire may inspect only immutable provenance/config, schema-12/tau-one identity,
the unchanged three derived strings, first/final checkpoints, actor
versions/acks/history, Huber/common/m-d-Adam transaction, exact inherited EMA
arithmetic, projection, AMP/non-finite counters, ModelBuffer seal/drain order,
W&B-disabled logger completion, finish, manifests, and process/Ray/GPU
cleanup. It supplies no qualifying behavioral row.

Wire must exercise at least one supported Q update. Evidence must show the
unchanged finite loss, zero-grad/backward/unscale/raw-norm/one-clip/m-d-Adam/
inverse-map/actor order; exact tau-one candidate arithmetic; candidate
`torch.equal` online weight and bias; one EMA commit; one exact projection;
and lockstep counts. The current minibatch must still use the pre-step EMA and
later publication must use the post-step tau-one result. Direct online use at
loss time fails.

Wire must bind production PyTorch and the inherited Adam/GradScaler source
hashes, adapter location, schema12/exact229, the new 209-key projection digest,
seal1, exact actor bundle/ack/history, one nonterminal publication, terminal
publication/ack, a valid drain-zero-or-one branch, durable model save,
complete-success, true worker returns, and clean process closure.

Any Q skip, non-finite, EMA equality mismatch, changed arithmetic, direct-copy
branch, direct-online loss route, second EMA/projection, count drift, timeout,
malformed/history error, late write, abort, retry, stale checkpoint, missing
finish, W&B artifact where forbidden, source drift, or incomplete cleanup
permanently fails v19. Negative paths need not occur live but must be frozen
in tests.

## Frozen 100k qualification

The 100k decision remains unchanged. Canonical rows satisfy
`70000 < real_step <= 100000`; windows are `(70000,80000]`,
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
  forced-stop diagnostics retaining inherited status;
- online-versus-EMA non-tie sign agreement is at least `0.60`;
- held-out EMA selected-action TD RMSE is at most `0.5`; and
- actor, online-Q, gate, ModelNet, protocol, AMP-skip, non-finite, mismatch,
  malformed, timeout, seal-late-write, and abort counters meet inherited zero
  requirements.

Every eligible trailing-five endpoint is required. Zero support makes its
gap undefined and the gate false. Undefined is never coerced to zero or
nonnegative; no pseudocount, smoothing, minimum-support replacement, endpoint
deletion, altered window, or shorter population is permitted.

Schema12/exact229/209, tau1, all three derived identities, Huber/common/m-d
Adam, exact inherited EMA arithmetic and equality, projection, barrier/history,
W&B, seal exact-ten, finish, manifests, and cleanup are hard integrity gates
without new numeric thresholds. Any failed qualification permanently forbids
primary and fixed evaluation.

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

All artifact, provenance, loss/reconstruction/optimizer/tau mechanism,
behavior, stability, support, denominator, trailing-five, saturation,
forced-stop, calibration, barrier, seal, AMP, and non-finite gates pass
together. There is no partial, diagnostic-only, mechanism-only, or historical
pass.

## Public, smoke, and fixed confirmation

The closed fixed profile is exactly `v19-300k` and accepts only the one
accepted seed-5 primary tuple. It rejects wire, qualification, schema11/v18,
and every legacy profile before rollout or output. Held-out seeds remain
20260827 through 20260842, exactly 16 streams by 6250 real steps and 100000
total, with calibration V-trace unroll 201 and inherited algebra.

After importing checkpoint-bound public code and resolving requested profile,
fixed evaluation must validate complete schema-12 primary before any
evaluator-direct/downstream flag load, live spec/environment probe,
construction/reset/action, data access, direct tensor load/use, rollout, or
output. Validator-internal bound deserialization is allowed and excluded from
downstream counters. Prevalidated evidence is reused exactly but never
trusted as a substitute for the dedicated validator.

Initial `config_c.yaml` bytes are stable-read, SHA-bound to completion
evidence, classified from those exact bytes, and consumed by the inherited
byte-aware loader without reopening mutable checkpoint config. Deletion,
replacement, schema12-to-legacy, legacy-to-schema12, alternate explicit
config, and probe-to-load swaps fail before downstream use or consume only
bound bytes; final revalidation catches artifact mutation before output.

Smoke order remains exact: immutable schema-12 prevalidation, stable byte
binding, private runtime copy, evaluator-only overrides, authoritative
postvalidation, exact pre/post evidence and checkpoint-hash equality, then
environment. Fixed order remains bound public import/requested profile,
schema-12 dispatch and dedicated validation/evidence equality, then downstream
use.

Validation covers exact229, the new exact209 digest, tau1, all three derived
identities, bundle/ack/history, raw parameters and m/d Adam state, Huber/common
Q, raw EMA/projection state, ModelNet state, seal exact-ten, W&B completion,
private-marker absence, public finish, source/runtime binding, and primary
tuple. Wrong/missing/extra/forged identity or evidence fails closed.

Schema 11 under `v19-300k` or another incompatible v19 profile, and schema 12
under any profile other than exact `v19-300k`—including malformed v19,
legacy, or `v18-300k` profiles—yields zero downstream calls/output.
Historical schema 11 remains byte-compatible on its unchanged v18 route; this
does not authorize evaluation, and v18 remains permanently failed.

Only after validation may an evaluator-private copy disable actor and
ModelNet training, parallel execution, live barrier waiting, and live seal
coordination. It records immutable training epsilon 0.02, execution epsilon
0.25, schema 12, seal 1, tau 1.0, and the three derived identities while
using runtime epsilon 0/0, barrier wait false, and seal coordination false.
Stored config/checkpoints are never rewritten. Fixed B/calibration probability
continues to use recorded learned-gate fields, not epsilon-zero execution
likelihood.

## Frozen test and audit matrix

Before implementation or snapshot clearance, tests cover at least this
matrix.

### Tau value, stage override, and schema isolation

- Global/default YAML and legacy `VOC_PROTOCOL_DEFAULTS` remain exact built-in
  float `0.1`; a source diff changing either global value fails. Schemas 1
  through 5 retain their full historical absent-default and supplied finite
  non-Boolean numeric normalization/range/coercion/error behavior. Schemas 6
  through 11 retain exact atomic tau0.1.
- Exact V19 stage inference resolves effective tau to exact built-in float
  `1.0` before complete-surface persistence. All three real
  `create_setting(save_flags=False)` stages produce exact229, exact209 with
  SHA `ad22b91f...`, schema12, seal1, identity12, and no added key.
- Direct stage/profile validation accepts only exact tau `1.0`. It rejects
  built-in `0.1`, zero, negative, subunit alternatives, values above one,
  NaN, infinities, integer 1, Boolean, NumPy scalar, decimal, string, null,
  missing, and defaulted values with no coercion.
- Exact V19 xpid with missing/wrong schema, barrier, seal, stage member, tau,
  or fresh flag rejects before config/run-dir/checkpoint/environment I/O.
  Malformed V19-prefix strings and lexical subclass/bytes/PathLike claims do
  the same.
- Legacy differential probes exercise schemas 1 through 5 with missing tau
  and representative valid nondefault numeric tau values/types: built-in float
  `0.25`, built-in integer `1`, and a NumPy numeric interior value. They bind
  the corresponding normalized built-in floats, the upper boundary, and
  unchanged Boolean/non-numeric/non-finite/out-of-range rejection. Returns,
  paths, coercions, and exceptions remain historical and exact.
  Schemas 6 through 11 retain exact tau0.1 byte/shape/path/error/output
  differentials. Schema11 with tau1 and schema12 with tau0.1 both reject.

### EMA arithmetic, equality, order, and counters

- Exact operation spies bind the inherited expression
  `(1.0-tau)*old + tau*online`, its order, FP32 tensor dtype/device, and
  validate-both-before-mutate behavior at tau1. Direct copy, `lerp`, fused,
  reordered, host, or loss-time-direct-online alternatives reject.
- Finite asymmetric weight/bias tensors prove each candidate is
  `torch.equal` to post-step online before commit. Separate signed-zero tests
  prove `torch.equal` while explicitly permitting different byte/sign-bit
  views; tests must not assert byte equality.
- Q loss and current gate use pre-step EMA. Q optimizer, optional gate, and
  main actor calls precede one EMA update; exact projection follows once;
  later publication/rollout observes the new target. A spy rejects updating
  before loss, retroactive use, extra update, or skipped projection.
- Supported successful Q updates advance Q, EMA, projection, publication
  evidence, and Q scheduler in inherited relations. Unsupported/all-held-out
  batches advance no committed Q parameter/Adam, EMA, or projection state
  while retaining eligible held-out evidence.
- Q AMP found-inf invokes no adapter commit and advances no committed Q
  parameter/Adam, EMA, projection, or Q-scheduler count. Raw `.grad` remains
  the inherited unscaled non-finite, unclipped branch state and held-out/loss
  evidence follows the inherited later path. Independent actor AMP behavior
  remains inherited; a successful Q step still updates EMA even if the actor
  scaler separately skips, and any live skip still fails acceptance.
- Non-finite online/old/candidate tensor, shape/dtype/device drift, candidate
  validation failure, projection failure, or count mismatch is fatal with no
  accepted finish. Candidate weight/bias validation occurs before either EMA
  tensor mutates. A failure after the Q commit does not imply that Q state was
  restored.

### Inherited loss, adapter, AMP, rollback, and runtime pins

- Schema12 beta-1 Smooth-L1 values/gradients, mask, supported mean, held-out
  isolation, common reconstruction, detached value/policy, Q Jacobian, action
  mapping, zero initialization, and raw-head outputs match schema11 exactly.
- Raw norm-first, one raw clip, no m/d clip, no second step, no new telemetry,
  zero-grad/backward/unscale order, Q/gate/actor call order, and scheduler order
  match schema11 exactly.
- The schema11 orthonormal adapter uses the same FP32 scale bits, cloned raw
  gradients, pinned one-call functional Adam, positive-zero scratch, m/d
  moment rows, inverse-mapped raw deltas, atomic commit, and exact rollback.
- Transform, functional, and staged-validation failures before live commit,
  plus a commit exception with verified rollback, prove no live Q parameter or
  Adam state advances, no downstream Q transaction advances, and no scaler
  update occurs. Rollback failure is a distinct fatal/no-accepted-artifact test
  with no unchanged-state promise.
  A fault injected after successful Q commit is fatal/no finish, does not undo
  the Q update, and is never mislabeled as a rollback. Raw found-inf and the
  finite-elements/nonfinite-norm branch retain their exact inherited
  state/scaler/finish guarantees.
- Snapshot/import and wire bind PyTorch `2.13.0+cu130`, Adam source
  `bde360b0...`, GradScaler source `97c411da...`, cp310 extension
  `1d4c5026...`, and adapter location in existing `learn_actor.py`. No new
  module or optimizer source is allowed.

### Surface, keyset, evidence, and forgery tests

- Schema12 config/actor/model surfaces are exact229; projection is exact209
  and exactly the schema11 projection except tau1, with full SHA
  `ad22b91fdd06a30ac7f53c0135b32fac2530687c3c36dad5dccf06d700f83f82`.
  The old `bd386...` digest rejects for schema12 and remains exact for atomic
  schemas6-11; schemas1-5 retain their historical non-atomic surfaces.
- Bundle/ack remain exact five keys with gate12; history remains exact seven
  with recomputed canonical digest; seal remains exact ten with drain0/1
  relations. No tau or derived identity is added to those keysets.
- Dedicated schema12 final/completed/actor validators require schema12,
  exact tau1 throughout existing protocol copies, and unchanged three derived
  strings/identity12. Dedicated routes reject bool/11/13/12.0/string values.
- With strict built-in integer EMA update count greater than zero, dedicated
  actor-only, final, public, smoke, and fixed validators independently require
  `torch.equal` for stored raw EMA weight versus online raw Q weight and stored
  raw EMA bias versus online raw Q bias. Weight-only, bias-only, and dual
  mismatch attacks reject before any downstream tensor/environment/output
  action. Signed-zero byte-different but `torch.equal` pairs pass; no
  byte-equality assertion is permitted.
- Exact-keyset comparisons prove every schema12 final/public/smoke/fixed
  record has the corresponding schema11 cardinality and one-to-one mapped
  keyset under schema-specific route-label substitution. Only required route
  labels, schema/tau values, and stage/path/complete/projection hashes differ.
- Actor/model top-level and nested reserved derived-key attacks remain
  cycle-safe and fail before tensor use. Tau attacks cover missing, duplicate,
  wrong-type, wrong-value, surface drift, embedded/top-level mismatch, and
  forged prevalidated evidence.
- Actor metadata, Adam state semantics, publication history digest, logger
  completion/checkpoint hashes, finish, private markers, source hashes, and
  stored-surface identity copies retain inherited strict built-in types and
  exact equality.
- Actor state metadata remains exact `{key,dtype,shape,numel}` for each unique
  state key with built-in types and exact shape-product relations. Tests
  independently recompute publication-history canonical JSON, bind fixed
  evidence top-level keysets and stage/path relations, and reject wrong
  checkpoint hash/size records, logger equality, or private-marker paths.

### Public, smoke, fixed, TOCTOU, and profile tests

- Schema12 validation occurs before every downstream flag load, live spec/env,
  data, direct tensor load/use, rollout, output, or rewrite. Invalid input
  produces zero downstream calls/output.
- Stable config payload/hash classification and byte-aware loading reject
  deletion, replacement, schema12/legacy swaps, alternate explicit config,
  and probe-to-load mutation. Final revalidation/evidence equality precedes
  output.
- Smoke binds prevalidation -> stable bytes -> private copy -> postvalidation
  and exact evidence/hash equality -> environment. Its exact-three identity
  containers have three exact12 inner mappings and no key drift.
- Fixed binds checkpoint-bound import/profile -> schema12 dispatch -> always-
  executed dedicated validation -> evidence equality -> downstream use.
  Forged caller-supplied evidence never bypasses the dedicated validator.
- `v19-300k` accepts only the exact seed-5 primary tuple. It rejects V19 wire
  and qualification, schema11/v18, schema12 under each legacy/v18 profile,
  schema11 under the v19 profile, preload/resume, wrong stage/path/tau,
  missing APIs, private marker, mutation, and forged history/logger evidence
  before output.
- Historical bound modules without schema12 APIs retain only their unchanged
  compatible route. Requested `v19-300k` without schema12 APIs fails before
  downstream use. No historical failed version gains evaluation authority.

### CLI, lifecycle, immutable snapshot, and no retry

- Real CLI tests bind the three exact V19 tuples, canonical Enduro resources,
  W&B modes, schedule100M, fresh inputs, schema12, tau1, exact229/209, and
  unchanged three derived identities. Derived strings never appear in
  CLI/YAML/config/checkpoint keys.
- Source/API tests require the exact schema12 symbols frozen above, built-in
  integer constant 12, unchanged older dedicated APIs, the existing
  `learn_actor.py` adapter location, and no alias or new optimizer module.
- Wrong whitespace/xpid, seed, horizon, warm-up, unroll, W&B, schema, tau,
  path, preload, parent, resume, resource, retry, or topology fails before
  run/environment action. `ckp=true` and every lexical malformed intent fail
  before persisted config open/run-dir creation.
- Runtime gate-schema propagation accepts strict12 with seal1 and uses
  dedicated schema12 final validation before ModelBuffer complete-success.
  Validator failure yields abort/no finish. Drain0 and drain1, late-write,
  timeout, kill, logger two-phase, no-restart/no-task-retry, process/Ray/GPU
  cleanup, and no-retry relations remain inherited.
- Immutable snapshot tests require an exactly enumerated overlay on the
  accepted V18 source/data baseline, independent inodes/modes, strict
  manifests, empty runs/cache, snapshot-only production cp310 imports,
  exact runtime pins, focused/adversarial/full suites, and unchanged posthash.

## Explicit falsifiers and claim boundary

V19's tau hypothesis is falsified for release by any of the following:

- wire mechanics fail, EMA is not `torch.equal` to poststep online after a
  supported successful Q step, or the inherited arithmetic/order changes;
- current-loss behavior uses online Q directly, publication loses its
  inherited one-update delay, or Q skip advances EMA/projection;
- any second semantic/config/state/key/telemetry change appears;
- any exact schema/type/stage/profile/freshness/source/hash/keyset validator
  fails;
- any 100k hard gate fails, even if denominator validity improves; or
- any 300k or fixed gate fails after all prior release gates pass.

V18 online RMSE `0.6212603749561236` is explicit counterevidence to a pass
forecast. Tau one is evaluated only by one future fresh schema-12 trajectory
under the unchanged gates. No observed V18 online value may be substituted
for that trajectory.

V19 requires, in order, one accepted immutable snapshot, one accepted
mechanics-only seed-1 wire, one accepted fresh seed-1 100k qualification, one
accepted fresh seed-5 300k primary, and that primary's one accepted
`v19-300k` fixed confirmation. Any failure is permanent for v19. No later
mechanics pass, diagnostic, seed, checkpoint, sibling protocol, or version
can retroactively change the permanent v14-v18 failures or a v19 stage
decision.

This authoritative preregistration does not itself perform or launch an
implementation, snapshot, experiment, retry, primary, fixed evaluation, or
evaluator action. Any later action must satisfy the sequential release gates
above on these exact frozen bytes.
