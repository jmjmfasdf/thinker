# Enduro VoC-v20 schema-13 telemetry-only preregistered acceptance

This document is the authoritative preregistration for one prospective
Enduro-only VoC-v20 lineage. It is the immediate successor of the permanently
failed v19/schema-12 qualification and of no other lineage. The sole semantic
delta is a behavior-preserving, hash-bound telemetry surface. Training
objective, gradients, optimizer, parameters, state, scheduler, EMA,
projection, action selection, replay consumption, counters, legacy logging,
W&B metric publication, and every numerical acceptance rule remain v19.

The schema discriminator is strict built-in integer
`voc_gate_policy_schema_version=13`. The exact 229 persisted configuration
keys and exact 209 projection keys remain unchanged. The 209 projection also
retains its exact v19 values, 4457-byte canonical JSON, and SHA-256
`ad22b91fdd06a30ac7f53c0135b32fac2530687c3c36dad5dccf06d700f83f82`.
This is possible because `voc_gate_policy_schema_version` is one of the ten
protocol keys outside the 209 projection. A schema-13 implementation must
prove directly that the schema key is absent from the projection; it may not
obtain the old digest by deleting or normalizing any projected value.

The existing exact three derived identities remain:

```text
voc_q_regression_loss="smooth_l1_beta1"
voc_q_reconstruction="detached_value_plus_raw_head_mean_plus_policy_centered_raw_head"
voc_q_optimizer_coordinates="orthonormal_common_difference_adam"
```

They remain derived-only. There is no telemetry-derived identity, CLI flag,
YAML key, 230th persisted key, checkpoint key, optimizer key, tensor key,
bundle key, ack key, history key, seal key, or public actor-state key.
Each schema-13 resolved identity has the inherited exact 12-key shape with
the two schema fields set to 13 and newly computed complete-surface hashes.

Schema 13 adds exactly five run artifacts:

```text
voc_td_cells.csv
voc_replay_events.csv
voc_q_transactions.csv
voc_telemetry_commits.csv
voc_telemetry_manifest.json
```

The legacy `logs.csv` remains exactly 922 ordered columns. Its schema-12
header is 43550 bytes including LF and has SHA-256
`82488231a631ca3571379e973122dd107007d14f4756fd839a811851dc6accbc`.
Schema 13 requires the same header bytes. The legacy `logs.csv` row values,
order, `FileWriter` tick behavior, and the live W&B metric keys and order are
unchanged. None of the five telemetry artifacts is passed to `wandb.log`.

This document does not predict an RMSE pass, a qualification pass, or a
causal mechanism. The six historical qualifications are unpaired,
single-seed, policy-coupled trajectories. Telemetry is ranked first because
the frozen evidence is insufficient to choose an honest cadence, learning
rate, clip, or coordinate intervention.

## Permanent v19 qualification failure

The only v19 qualification was
`enduro-voc-v19-tau1-orthocd-adam-eps25-seed1-qual-fresh-100k`, launched once
from immutable snapshot
`/tmp/di-voc-v19-tau1-orthocd-adam-eps25-final-tdwY9W`. It was fresh, seed 1,
W&B-enabled, schema 12, model-input-seal schema 1, exact229, exact209, tau
one, beta-1 Huber, common reconstruction, raw C/S parameters, orthonormal
m/d Adam state, and exact raw EMA projection. The driver exited zero.
Actor/model mechanics, 186 successful Q/EMA/projection updates, barrier and
history, W&B request/ack/private cleanup, public finish, source/data hashes,
Ray, process, and GPU closure all passed.

The frozen actor log is `logs.csv`, 1887052 bytes, SHA-256
`e963cedf69f84857a1a1c5adb4e0961fb78fc7d698bb07f668a27ededb135bfa`.
It has 186 complete rows and the exact 922-column header above. The canonical
qualification population is exactly 52 complete rows satisfying
`70000 < real_step <= 100000`, from 70080 through 99632. The single row at
100576 is an overshoot and is excluded.

V19 passes 14 of the 15 unchanged CSV-observable gates. It fails only pooled
held-out EMA selected-action TD RMSE:

```text
sqrt(2106.3285870117097 / 7637) = 0.5251721239021901 > 0.5
```

At fixed support the threshold SSE is `7637 * 0.5^2 = 1909.25`; excess SSE
is `197.0785870117097`. Every one of the 48 eligible trailing-five endpoints
has both positive and negative support. No denominator, sign-run, support,
margin, saturation, agreement, safety, or mechanics gate fails.

Frozen window evidence is:

| Window | Rows | Held-out support | EMA SSE | EMA RMSE | Share of full SSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 `(70000,80000]` | 14 | 1911 | `1722.5797661805962` | `0.9494220793526011` | `81.7812%` |
| W2 `(80000,90000]` | 28 | 4272 | `8.31643801673691` | `0.04412178311776224` | `0.3948%` |
| W3 `(90000,100000]` | 10 | 1454 | `375.4323828143764` | `0.5081403257530148` | `17.8240%` |
| Full | 52 | 7637 | `2106.3285870117097` | `0.5251721239021901` | `100%` |

Rows 71824 and 73600 alone contribute `55.9986%` of full SSE and have
opposite bias signs. The top ten rows contribute `93.7513%`. Pooled bias
`-0.021300` is not calibration evidence. Online and EMA selected-Q/TD
aggregates are equal under the frozen tau-one path; the online RMSE is not a
passing counterfactual. Only 2/52 canonical raw Q norms exceed 1608, and one
of the two largest-error rows has raw norm about 199.76. These aggregates do
not identify a clip or step-size direction.

V19 is permanently failed. It may not be retried, resumed, extended,
reseeded, checkpoint-selected, evaluated under `v19-300k`, or used for a
primary/fixed rescue. Its checkpoints, optimizers, replay, actor versions,
buffers, observations, and runtime state may not seed v20.

## Frozen v19 artifacts and provenance

The v19 qualification run directory is
`/tmp/di-voc-v19-tau1-orthocd-adam-eps25-final-tdwY9W/runs/enduro-voc-v19-tau1-orthocd-adam-eps25-seed1-qual-fresh-100k`.
Its exact 14-file inventory is:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `ckp_actor.tar` | 28301587 | `e0451e4d396e1b8ec464d568a110b8c35d5838ee1f2578aa9604571a14ad88d3` |
| `ckp_actor.tar_step_480` | 28259701 | `3e03e4e3bca90e38ad5eb717dbb183d0ef7169ada8d025a1bd5f030cde999500` |
| `ckp_actor.tar_step_100576` | 28309291 | `6597abf7b16b75888fab951760c11ee36a3996782761a742c6f9e11cd57f3785` |
| `ckp_model.tar` | 133359381 | `320703dc0b8c1ef247e4f3c43075bd994eb52c80a0eae11dfea8c2bb4c2dd8d6` |
| `ckp_model.tar_step_10000` | 133380637 | `8f501ed36082ee52bace908d01f63e3690ec4bfb47daf4447e732f99c5e0575f` |
| `ckp_model.tar_step_100576` | 133359381 | `320703dc0b8c1ef247e4f3c43075bd994eb52c80a0eae11dfea8c2bb4c2dd8d6` |
| `config_c.yaml` | 7672 | `0a2416f2ec79a7d4d4cb285350a360bdbadc4066c82b42319c54b660dbcf8206` |
| `finish` | 3354 | `f3acef31e58a697f3c569aea31125b803b40d4d6a2d117c9406a1c0a830e36bd` |
| `logs.csv` | 1887052 | `e963cedf69f84857a1a1c5adb4e0961fb78fc7d698bb07f668a27ededb135bfa` |
| `logs_model.csv` | 135160 | `81ef29b4090ab862030e87cebaa5e41738ba6395d0945589aaab4adee0412e19` |
| `meta.json` | 15095 | `f2f43edb3169bf21232adfe4c46ade9f4fb1a9f05a5f17d81d7f5f6ab09e092e` |
| `meta_model.json` | 15095 | `869ace15a73322826bfa798040b23b8c260b0e6d278ce1a7bcb564e6685fa7e1` |
| `out.log` | 68574 | `911540b0b64f6084e43500a56b0c6ee2b7efb7d5b619bba103952510acc5a355` |
| `out_model.log` | 177325 | `f6e0bf47f2a14caf49683426f398e413e424ed872c22521d943e008f56a64577` |

The canonical 1173-byte C-sorted record binding uses exact lines
`{lowercase_sha256}  ./{basename}\n` and has SHA-256
`a55381ac4db9be53d6778a4c831549c7916d1f51aa92bac583a98c3961ec6f55`.
The public schema-12 evidence SHA-256 is
`19252d989700bae21b4994e5f4e0e874e58c93bce89c47653c77d4f89e0b911e`.

The immutable v19 source manifest has 1068 records and SHA-256
`df5ad87cc9a091e19ab5314700f0f1ce0fb5c875bbe777b1fa7c73fb5f3e116d`.
The unchanged 11-record data manifest is
`23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343`.
The authoritative v19 preregistration is
`VOC_V19_TAU1_Q_300K_ACCEPTANCE.md`, 57331 bytes and 1069 lines, SHA-256
`c5629b024ea8f0f1740f3ea209dea1322b6643ae80d9d2b25f10f5dd4ff52ce6`.

The v19 runtime is `/tmp/v19qual-vjkLuzEG`. Its launch provenance
`launch_provenance.txt` has SHA-256
`0a676c435285dc40c4c053c45630b71af9782176a9b7ae92249e110a74832a6c`;
the driver log has SHA-256
`3a68c5e063deee0782151f28c522b1a99023ba6fc0868e439166a4d407103d7d`;
and `driver.exit_code` is exact `0\n`, SHA-256
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
The one-attempt file is exact `1\n`, SHA-256
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`.
The accepted v19 wire tree is
`a2ffbc6a15859e3bc4afe49a35658ef71d0e055460cd394c218df2a0d93aa049`;
its public evidence is
`3be2043b1630b19dd4f39999eb308a582121b95ca6a385d1d642406e96adc1cf`.

Terminal policy version, Q updates, EMA updates, and projections are each
186. Publication history SHA-256 is
`1b95dbbe778d06aca55d50c42e5da72febb51580e6c754beef5c1e68c024a6f1`;
logger request SHA-256 is
`ece1ebe9b85dcef63d199e62edfc0402aedc447d135bd93dd2aba730a8d31961`;
actor-state SHA-256 is
`a9541b3c76d73fd072e023f053a40cce8089e69fc9d1283b1ed7d0ee05c09e04`.
W&B completion was required, acknowledged, and privately cleaned.

Production remains CPython 3.10.21 and PyTorch `2.13.0+cu130`. The inherited
Adam source SHA-256 remains
`bde360b0bb9b7869f1cec04a3b41a90b8eabb84a613787d97b88d87f2f3ae1ec`;
GradScaler source remains
`97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039`;
the cp310 extension remains
`1d4c5026d2a6c002a13829e162428505b30a65bf2af1968f6e982dcfcc16b232`.
These are runtime/snapshot evidence, never new config keys.

## Frozen V14-V19 RCA and consensus

The authoritative executed notebook is
`notebook/v14_v15_v16_v17_v18_v19_qualification_comparison_rca.ipynb`,
935244 bytes and 1438 lines, SHA-256
`133891d3945cebf9d8d5961e934997ccc651fcfedfbce775022cea79250f1efc`.
It has 17 cells, eight continuously executed code cells, zero errors, and
three visually QA-passed charts. The authoritative report is
`notebook/v14_v15_v16_v17_v18_v19_qualification_rca_report.json`, 664564
bytes and 11285 lines, SHA-256
`dd828f6170a41752a8301ecc33fc355978d04f499ec358e9c2c9cfeecc3396f5`.
Both must retain exact bytes.

The independent A/B consensus is:

1. collect behavior-preserving schema-13 telemetry before any behavior
   change;
2. consider a deterministic real-step Q-cadence mechanism only after
   telemetry supports and a later preregistration freezes its quantum,
   boundary, backlog, data-use, restore, and ordering semantics; and
3. consider Q-only coordinate/LR/clip sensitivity only after telemetry
   supports a direction and exact value.

The second and third choices are deliberately not implementation-ready.
Values such as a 512-step quantum or half learning rate are unsupported and
withdrawn. This v20 defines no cadence quantum, credit, backlog, reuse,
restore, learning-rate, or clip change. The ordering is an information-gain
priority, not causal attribution or a pass forecast.

## Normative inheritance and sole delta

The complete frozen v19 preregistration is incorporated except for these
closed substitutions:

- V19 names/profile become the exact V20 names below;
- gate-policy schema 12 becomes strict schema 13;
- schema13 writes and validates the five telemetry artifacts defined here;
- schema13 terminal completion uses completion schema 2 and binds the
  telemetry manifest as the fourth `checkpoint_files` record; and
- dedicated util/public/smoke/fixed routes expose the exact telemetry
  evidence object defined below.

Everything else is v19. In particular, tau remains strict built-in float
1.0; inherited FP32 tau arithmetic and `torch.equal` post-update raw
online/EMA checks remain unchanged. Huber beta 1, common reconstruction,
raw `[CONTINUE,STOP]` parameters, orthonormal `[common,difference]` Adam
state rows, the exact FP32 scale bit pattern `0x3f3504f3`, raw clipping,
GradScaler transaction, learning rates, Adam groups, actor/gate/Q ordering,
EMA/projection, scheduler, execution epsilon, action probabilities, replay,
batching, RNG, model, barrier, seal, and every counter are unchanged.

Telemetry values are detached observations only. They may not appear in a
loss, gradient, mask, sampler, action, optimizer, scheduler, scaler, EMA,
projection, counter, checkpoint tensor, actor-policy bundle, ack, history,
model input, W&B metric dictionary, or legacy log dictionary. No training
computation, update, or publication decision may depend on a telemetry value
or successfully written byte. Before version-zero publication, exact
schema/profile/path checks and exclusive four-header creation/fsync may fail
closed; such initialization failure creates no policy/training state, no
transaction evidence, and no finish. After successful header initialization,
per-transaction telemetry reduction/row/hash/commit integrity may only fail
after the inherited publication/ack boundary under the explicit fatal/
no-finish contract below.

Behavior-preserving means that a fixed-input deterministic harness has exact
equality for model/actor outputs, losses, raw gradients, clipped gradients,
optimizer parameters and states, scheduler/scaler states, raw EMA,
projection, RNG states, counters, bundle/history/ack content, legacy stats
dictionary, 922-column CSV row, and W&B metric payload between schema12 and
schema13 after closing only schema/version/xpid/path evidence substitutions.
It does not promise identical asynchronous wall-clock timing, Ray scheduling,
or a realized on-policy trajectory.

Schema13 adds exactly one new production source path,
`thinker/thinker/voc_telemetry.py`. It is a leaf, import-pure module: no
module-level file I/O, RNG, clock, thread, Ray, W&B, logger, device, model, or
optimizer access. It is imported lazily only after exact schema13 intent has
passed. Schemas at most 12 never import, call, hash, or expose it. A
fresh-subprocess legacy test requires the module absent from `sys.modules`
before and after each such route, independent of full-suite import order.

The module owns the frozen schema constants, canonical codec, sidecar writer,
manifest builder, and byte validator. The learner passes only detached
source/candidate values; the module never receives a live model, parameter,
optimizer, scaler, scheduler, replay buffer, or actor-policy handle. Util and
all public callers reuse the same bound validator, while tests also use an
independent oracle. Schema13 `implementation_sources` is the inherited exact
14-key mapping plus only `thinker/voc_telemetry.py`, hence exact15. Schemas at
most 12 remain exact14 without global list drift.

The exact schema13 source keys are:

```text
train.py
thinker/actor_net.py
thinker/bc_loader.py
thinker/cenv.pyx
thinker/dataset_env.py
thinker/dynamic_imitation.py
thinker/gym_add/wrapper.py
thinker/learn_actor.py
thinker/learn_model.py
thinker/logger.py
thinker/main.py
thinker/model_net.py
thinker/self_play.py
thinker/util.py
thinker/voc_telemetry.py
```

## Exact schema lineage and identity

Schemas 1 through 5 retain their full historical finite non-Boolean numeric
tau acceptance, normalization, coercion, exceptions, paths, and behavior.
Absent tau keeps the historical 0.1 default. Schemas 6 through 11 retain
strict atomic tau 0.1 and all historical loss/reconstruction/optimizer
branches. Schema 12 retains exact v19 tau-one behavior and all schema-12
marker/public shapes. Schema 13 is schema 12 plus telemetry evidence only.
No schema-at-most-12 code path may create, require, parse, upload, or return a
schema-13 telemetry artifact or schema-2 marker.

Schema13 retains the exact persisted decomposition:

```text
229 = 209 stage-neutral keys
    + 6 stage keys
    + 4 path-derived keys
    + 10 protocol keys
```

The six stage keys are `xpid`, `base_seed`, `total_steps`,
`model_warm_up_n`, `actor_unroll_len`, and `use_wandb`. The four path keys are
`savedir`, `ckpdir`, `cmd`, and `icopro_data_path`. The ten protocol keys are:

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

Only the schema protocol value becomes 13. The 209 projection excludes
`voc_gate_policy_schema_version` and remains exact v19 projection bytes:
209 keys, 4457 bytes, SHA-256
`ad22b91fdd06a30ac7f53c0135b32fac2530687c3c36dad5dccf06d700f83f82`.
It still contains strict built-in float `voc_gate_target_tau=1.0`. Complete
surface digests are new because schema, stage, paths, xpid, and command are
bound. Canonical JSON remains sorted, ASCII-safe, compact separators,
`allow_nan=False`, UTF-8, and no trailing LF.

Each inner resolved identity remains exact 12 keys:

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

It requires `key_count=229`, projection count 209 and digest above,
`gate_schema=13`, `voc_gate_policy_schema_version=13`, seal schema 1, the
three inherited derived strings, and exact V20 stage/path/complete evidence.
The three derived names remain recursively reserved from actor/model
checkpoint mappings and nested mapping/list/tuple containers, including
correct, wrong, null, and forged values. The cycle-safe rejection occurs
before tensor use. Telemetry names are not checkpoint keys.

Actor-policy bundles retain exact five keys and bundle schema 1; only
`gate_schema=13`. Acks retain exact five keys with gate schema 13. History
retains exact seven keys. Model-input-seal evidence retains exact ten keys,
seal schema 1, one durable terminal save, drain zero or one, late-write zero,
abort zero, actor-before-model-before-finish, and unchanged tensor/state
keysets. Actor/model/config checkpoint keysets and optimizer IDs/groups/state
shapes are unchanged.

Core and public source must expose these dedicated names in the same module
locations as their schema-12 analogues:

```text
VOC_GATE_POLICY_TELEMETRY_Q_SCHEMA_VERSION
VOC_TELEMETRY_SCHEMA_VERSION
VOC_GATE_POLICY_SCHEMA13_STAGE_PROFILES
_validate_schema13_stage_profile
_validate_schema13_complete_surface
_validate_schema13_protocol_flags
validate_schema13_telemetry_manifest
validate_schema13_final_bundle
validate_voc_schema13_final_actor_checkpoint
validate_schema13_completed_bundle
```

The gate constant is exact built-in integer 13, and the telemetry-format
constant is exact built-in integer 1. Every dedicated API rejects
non-built-in/incorrect schema types, wrong stages, and any schema at most 12
as applicable. The manifest, final-bundle, and completed-bundle APIs reject
missing telemetry. Stage/profile/protocol and actor-checkpoint APIs validate
their own available surface and cannot treat absent terminal run artifacts as
success; every terminal public/smoke/fixed route must compose them with the
telemetry-aware final validator. Shared dispatch may recognize schema 13 only
through the dedicated branch. The schema-13 final bundle is the inherited
exact 12-key bundle plus exactly one top-level `telemetry` key, hence exact13.
The normalized telemetry object is defined below.

## Exact stages and fresh-only lineage

The lexical prefix is exactly
`enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-`. Schema-13 intent is
classified before config/checkpoint/run-directory/environment I/O for exact
strings, malformed strings, subclasses, NumPy strings/bytes, UTF-8 bytes,
`os.PathLike`, and arbitrary lexical objects. Intent classification never
coerces acceptance. The valid xpid itself must be a built-in string.

The only exact tuples
`(xpid, base_seed, total_steps, model_warm_up_n, actor_unroll_len, use_wandb)`
are:

- (`enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-sentinel-wire1200`,
  1, 1200, 512, 41, false);
- (`enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-seed1-qual-fresh-100k`,
  1, 100000, 10000, 201, true); and
- (`enduro-voc-v20-telemetry-tau1-orthocd-adam-eps25-seed5-strict-fresh-300k`,
  5, 300000, 10000, 201, true).

Each stage additionally requires strict schema13, tau1, barrier true, bundle
schema1, seal1, execution epsilon0.25, training epsilon0.02, schedule total
100000000, CUDA0/1, Ray2GPU/16CPU, actor/model/self-play GPU allocation
unchanged, `self_play_n=1`, `env_n=16`, and exact229/209. Wire is W&B false;
qualification and primary are W&B true and authenticated. The closed fixed
profile is exactly `v20-300k`.

Every stage command retains the inherited exact 96 ordered flag/value pairs,
including exactly one `--voc_gate_target_tau 1.0`, and carries no
`--voc_gate_policy_schema_version` token. Built-in integer 13 is inferred
only from the exact V20 lexical xpid together with the complete strict atomic
surface. An explicit schema CLI pair, extra telemetry CLI pair, or any 97th
pair is forbidden.

The global `thinker/config/default_actor.yaml` remains byte-identical to V19,
including its historical tau 0.1 default, at SHA-256
`47c60f3a87aa762c62cfa80b6081f84911da533f15adb4d47262c9019150498d`.
Only the exact V20 CLI/stage surface supplies strict built-in float tau1; no
default/YAML edit or new key is allowed.

Every stage is fresh: `ckp=false`; `preload`, `preload_actor`, and
`voc_parent_checkpoint` empty; parent update count zero; actor policy starts
at version zero; Q/EMA/gate/Adam/model/buffers/seal/telemetry files start
fresh. No v14-v19 or earlier V20 state crosses a stage. No resume, recovery,
partial sidecar reuse, tail truncation, repair, extension, seed substitution,
or retry is allowed.

## Canonical telemetry file grammar

All four sidecars are ASCII CSV with a single exact header, unquoted comma
separation, LF-only lines, and no BOM, CR, blank line, trailing space, quote,
or extra column. All values are chosen so quoting is unnecessary. Integers
are canonical base-10 ASCII with no sign or leading zero except `0`.
Booleans are exactly `0` or `1`. Hashes are lowercase 64-hex. Enums are the
lowercase tokens frozen below.

Every data row requires strict built-in integer
`telemetry_schema_version=1` and strict built-in integer `gate_schema=13`.
For each transaction, schema, gate schema, transaction ID, source policy
version, and published policy version agree exactly across all TD, replay, Q,
and commit rows. `real_step_after` agrees across TD, replay, and Q; terminal
agrees across replay and commit; and actor-state/publication-history hashes
agree across replay and commit. No coercion or cross-file drift is accepted.

Every defined finite numeric statistic is a binary64 value serialized by
CPython 3.10 `float.hex()` in lowercase and reparsed by `float.fromhex()`.
Every zero is normalized to positive `0x0.0p+0`. NaN, infinity, negative-zero
spelling, decimal float spelling, or alternate hex spelling is invalid.
The exact uppercase token `NA` is allowed only in Q-transaction fields
explicitly declared unavailable by the status matrix. It is forbidden in
TD cells, replay rows, commit rows, and every otherwise-defined Q field.
Lowercase `na` is never valid.

Sidecar sources are detached FP32 tensors. Telemetry clones the required
tensors at their specified observation boundary without CPU synchronization
or reduction before inherited publication/ack. After ack, canonical flatten
order is time-major then batch-minor and, within a parameter, contiguous
row-major. Each exact FP32 scalar is promoted to binary64. Sums use CPython
3.10 `math.fsum`; products/squares are binary64; L2 is
`math.sqrt(math.fsum(x*x))`. Empty TD cells have count zero and every statistic
equal to positive zero.

Telemetry reparse/recompute fixtures must be bit-exact. Cross-checks against
legacy GPU/PyTorch reductions require exact integer counts but use frozen
forward-error bounds, not impossible bit equality: for a finite legacy sum,
mean, norm, or RMSE, require
`abs(a-b) <= 2**-20 * max(1,abs(a),abs(b)) + 2**-30` unless the inherited gate
already specifies a stricter check. This tolerance is evidence validation
only and never changes an acceptance threshold or logged value.

Fresh initialization creates the four CSVs exclusively as single-link
regular files at exact mode `0600` under umask `077`, writes their exact
headers, flushes/fsyncs them, and rejects
preexistence, symlink, special file, multiple link, wrong owner, or alias.
After all four header fsyncs and before version-zero policy publication, it
fsyncs the run directory so the four names and headers are durable.
No generic `FileWriter` is used. The files are append-only until terminal
seal. Read/validation uses descriptor-bound no-follow opens, pre/post fstat
identity/size checks, and exact bytes; pathname probe-then-reopen is invalid.

### `voc_td_cells.csv`

The exact 22-column header plus LF is 300 bytes with SHA-256
`37c82eea9a7bf7cbe05ee74ffb2b37b6190e4b715b05afbea5b5a06c406473fa`:

```text
telemetry_schema_version,gate_schema,transaction_id,source_policy_version,published_policy_version,real_step_after,q_source,split,selected_action,depth_bin,td_sign,abs_td_bin,count,sum_target,sum_target_sq,sum_selected_q,sum_selected_q_sq,sum_target_selected_q,sum_td,sum_abs_td,sum_td_sq,max_abs_td
```

Each completed policy transaction contributes exactly 720 rows, including
empty cells, in this exact Cartesian order:

1. `q_source`: `online`, then `ema`;
2. `split`: `train`, then `holdout`;
3. `selected_action`: `continue`, then `stop`;
4. `depth_bin`: `0`, `1`, `2_3`, `4_7`, `8_15`, `16_plus`;
5. `td_sign`: `negative`, `zero`, `positive`; and
6. `abs_td_bin`: `0_0p5`, `0p5_1`, `1_2`, `2_4`, `4_inf`.

Depth is not raw `search_steps`. It reuses the inherited decision-depth
definition exactly:

```text
decision_depth = search_steps - (valid & (control_action != STOP))
```

and the existing six mutually exclusive bins 0, 1, 2-3, 4-7, 8-15, and
16+. Negative depth, uncovered depth, or changed STOP adjustment fails.

For each source, selected Q is independently gathered at the actual inherited
gate action: PROCEED/RESET map to CONTINUE row 0 and STOP maps to row 1.
Online uses the reconstructed online Q already returned by
`compute_dynamic_voc_loss`; EMA uses the independently reconstructed raw EMA
Q already used for the gate diagnostics. Both target and selected Q are cast
to the same FP32 work representation used by the inherited Huber path before
detaching. TD is recomputed independently as exact FP32
`target - selected_q` for each source.

TD sign is the exact residual comparison `<0`, `==0`, or `>0`; it is not the
inherited `1e-6` Q-gap tie rule. Absolute bands are exact half-open FP32
intervals `[0,0.5)`, `[0.5,1)`, `[1,2)`, `[2,4)`, and `[4,+inf)`. Every
supported finite event belongs to exactly one source/split/action/depth/sign/
band cell. Train uses inherited `q_train_valid`; holdout uses inherited
`dynamic_voc_holdout_mask`. They remain disjoint and may not change loss
support.

Each cell records count and the exact binary64 reductions:
`sum(target)`, `sum(target^2)`, `sum(selected_q)`, `sum(selected_q^2)`,
`sum(target*selected_q)`, `sum(td)`, `sum(abs(td))`, `sum(td^2)`, and
`max(abs(td))`. These are sufficient to reconstruct support, bias, MAE, SSE,
RMSE, target/Q means and variances, covariance, action/depth composition, and
source-specific tails without retaining event rows.

The authoritative validator also reconstructs the selected-action beta-1
Huber loss from `q_source=online`, `split=train` cells only. Across every
action/depth/sign cell it uses `0.5*sum_td_sq` for bands `0_0p5` and `0p5_1`,
and `sum_abs_td - 0.5*count` for bands `1_2`, `2_4`, and `4_inf`, then combines
terms with `math.fsum`. That reconstructed binary64 sum must agree with the
source-FP32 `q_loss_sum` under the frozen forward-error bound. When
`train_count>0`, `q_loss_sum/train_count` must likewise agree with the
inherited logged supported mean under that bound. EMA cells are never a loss
route, and expected FP32-loss versus binary64-cell reduction rounding is not
mistaken for drift.

### `voc_replay_events.csv`

Here “event” means one inherited consumed replay batch and policy
transaction, not one real transition. The exact 39-column header plus LF is
713 bytes with SHA-256
`eed6226a8a591289125c7f5389b7d6705332b11e32746f103093b5dcd71592e2`:

```text
telemetry_schema_version,gate_schema,transaction_id,source_policy_version,published_policy_version,replay_t,optimized_t,replay_b,actor_ids,actor_ids_sha256,real_step_before,real_step_delta,real_step_after,valid_count,train_count,holdout_count,train_continue_count,train_stop_count,holdout_continue_count,holdout_stop_count,q_status,voc_update_count_before,voc_update_count_after,ema_update_count_before,ema_update_count_after,projection_count_before,projection_count_after,q_scheduler_last_epoch_before,q_scheduler_last_epoch_after,q_scheduler_step_count_before,q_scheduler_step_count_after,q_lr_before,q_lr_used,q_lr_after,publication_count_after,ack_count,terminal,actor_state_sha256,publication_history_sha256
```

There is exactly one row per completed policy transaction. `transaction_id`
is a telemetry-only contiguous ordinal beginning at 1. It is not an inherited
batch ID or scheduling state. It must equal `published_policy_version`;
`source_policy_version` equals `published_policy_version-1`; and
the inherited legacy log tick equals `transaction_id-1`. The tick is checked
against `logs.csv` but is not duplicated in this minimal replay header. The
one replay row, one Q row, 720 TD rows, and one commit row share the
transaction ID.

`replay_t`, `optimized_t`, and `replay_b` are the actual received tensor
dimensions. They must obey
`replay_t=actor_unroll_len+1`, `optimized_t=actor_unroll_len`, and
`replay_b=16`: wire is exactly 42/41/16, while qualification and primary are
exactly 202/201/16. The record reads actual validated values rather than
injecting constants.
`actor_ids` is the actual positional order from the inherited actor-ID tensor,
encoded as semicolon-separated canonical decimal integers with no spaces.
It is not sorted. `actor_ids_sha256` hashes those UTF-8 bytes with no LF.
Existing validation still requires the exact unique set 0 through 15 while
preserving its observed positional order.

`real_step_before` is staged before `consume_data`; `real_step_delta` is the
exact inherited count of true `real_transition[1:]`; and after equals before
plus delta. This row defines no within-batch transition ordering. Supports
are the inherited valid/train/holdout masks and their action subdivisions.
Exact count relations require valid=train+holdout, each split equals its
CONTINUE+STOP subdivision, and every count lies in `[0,optimized_t*replay_b]`.

`q_status` is exactly `stepped`, `no_support`, or `amp_skip`; attempted state
is derived exactly as stepped/amp-skip versus no-support in the Q row.
Counters and Q scheduler epoch/step count are captured immediately before
the Q attempt and at the inherited post-scheduler boundary. `q_lr_before`
is the finite Q param-group LR at the optimizer-decision boundary, and
`q_lr_used` equals it for every row, including the would-be LR when there is
no support. `q_lr_after` changes through exactly one inherited Q-scheduler
step only for `stepped`; it equals before for `no_support` and `amp_skip`.
All three use the canonical hex grammar and are never `NA`. Projection count
is the inherited exact-projection counter, not an invented counter.
For `stepped`, Q/EMA/projection counts and Q-scheduler step count each advance
exactly one; for `no_support` and `amp_skip`, all four remain unchanged. The
scheduler epoch relation and LR change must match that same status.

Publication fields are captured only after the exact publication and ack.
Durable rows therefore require the exact state/history hashes, the observed
publication and ack counts, and a terminal bit matching the
published bundle. The frozen `self_play_n=1`, so `ack_count` is exactly 1 in
wire, qualification, and primary. It counts the one active self-play worker
rank, not the 16 actor streams in `actor_ids`. `publication_count_after`
equals `transaction_id`; every row before the final transaction has terminal
0 and exactly the final row has terminal 1. Fatal attempts cannot reach this
boundary and have no durable replay row.

Schema13 defines no real-step quantum, credit, backlog, produced/queued/
consumed conservation ledger, per-transition row, batch reuse rule, restore
rule, or new scheduler semantics. The primitive batch-boundary facts are
sufficient to evaluate a future cadence hypothesis at existing consumed-batch
boundaries; they do not authorize one.

### `voc_q_transactions.csv`

The exact 59-column header plus LF is 1603 bytes with SHA-256
`e1574cf8c81306818abc2369b5270f98c74f3e3190cb8cb3ceddb000ad4096b3`:

```text
telemetry_schema_version,gate_schema,transaction_id,source_policy_version,published_policy_version,real_step_after,q_status,q_attempted,q_optimizer_committed,q_loss_sum,clip_limit,clip_scale,raw_preclip_total_l2,raw_postclip_total_l2,amp_scale_before,amp_scale_after,nonfinite_gradient_parameter_count,adam_step_before,adam_step_after,raw_preclip_weight_continue_l2,raw_preclip_weight_stop_l2,raw_postclip_weight_continue_l2,raw_postclip_weight_stop_l2,md_postclip_weight_common_l2,md_postclip_weight_difference_l2,adam_m_before_weight_common_l2,adam_m_before_weight_difference_l2,adam_v_before_weight_common_mean,adam_v_before_weight_difference_mean,adam_m_after_weight_common_l2,adam_m_after_weight_difference_l2,adam_v_after_weight_common_mean,adam_v_after_weight_difference_mean,normalized_update_weight_common_l2,normalized_update_weight_difference_l2,coordinate_delta_weight_common_l2,coordinate_delta_weight_difference_l2,mapped_delta_weight_continue_l2,mapped_delta_weight_stop_l2,raw_preclip_bias_continue_l2,raw_preclip_bias_stop_l2,raw_postclip_bias_continue_l2,raw_postclip_bias_stop_l2,md_postclip_bias_common_l2,md_postclip_bias_difference_l2,adam_m_before_bias_common_l2,adam_m_before_bias_difference_l2,adam_v_before_bias_common_mean,adam_v_before_bias_difference_mean,adam_m_after_bias_common_l2,adam_m_after_bias_difference_l2,adam_v_after_bias_common_mean,adam_v_after_bias_difference_mean,normalized_update_bias_common_l2,normalized_update_bias_difference_l2,coordinate_delta_bias_common_l2,coordinate_delta_bias_difference_l2,mapped_delta_bias_continue_l2,mapped_delta_bias_stop_l2
```

There is exactly one row per completed policy transaction. The row observes
only the existing VoC-Q weight then bias parameter order. `q_loss_sum` is the
unchanged selected-action Huber sum before `voc_loss_cost`. `clip_limit` is
exactly `actor_grad_norm_clipping * replay_t * replay_b`, using the
overlap-inclusive `T` passed to `consume_data_single`: wire uses 42*16 and
qualification/primary use 202*16, never optimized 41/201 or the historical
1608 shorthand. Raw preclip clones are taken
after unscale and before the only clip; raw postclip clones are taken after
that clip and before the m/d adapter. `.grad` is never changed by telemetry.
`raw_*_total_l2` and per-row L2 values use the binary64 reduction above.
`clip_scale` is the actual coefficient used by the inherited pinned-Torch
clip call, `min(1, clip_limit/(N+1e-6))`, where `N` is that call's returned
preclip norm. Telemetry captures this coefficient and the raw clones; it
does not run a second behavior-affecting norm or substitute a coefficient.
The separately reduced raw post/pre norms are checked against this coefficient
under the frozen forward-error bound, not required to yield it bit-exactly.

`md_postclip` is the exact adapter transform of the already clipped raw
C/S clones with the inherited FP32 scale. Adam m/v before and after are the
actual staged schema-12 m/d state rows. For `t=adam_step_after`, beta1 0.9,
beta2 0.999, and epsilon 1e-8, the independent formula is
`mhat=m/(1-0.9**t)`, `vhat=v/(1-0.999**t)`, and
`u=mhat/(sqrt(vhat)+1e-8)`. `normalized_update` records the L2 norm of the
actual `coordinate_delta/(-q_lr_used)`, using the same transaction's replay
row LR, while `coordinate_delta` is the
actual signed zero-base functional-Adam scratch result. The formula-derived
`u`, `-q_lr_used*u`, and their norms must agree with the actual scratch and
recorded norms within the frozen forward-error bound above, never by an
impossible universal bit-equality requirement. `mapped_delta` is the actual
inverse-mapped raw C/S delta staged before the atomic commit. These
diagnostics never recompute or replace the production Adam call.

Each `adam_v_*_mean` is computed without `torch.mean`: promote the exact row
elements to binary64 in contiguous order, apply `math.fsum`, and divide in
binary64 by the exact row element count (weight dimension D; one scalar for
each bias coordinate).

The exact status/`NA` matrix is:

| Status | Exact base fields | Exact optimizer-field availability |
| --- | --- | --- |
| `stepped` | attempted/committed/nonfinite count = 1/1/0; finite Q loss and clip limit; finite AMP scales with after >= before; canonical integer Adam steps with after=before+1 | `clip_scale`, both raw totals, and all 40 parameter diagnostics are finite hex |
| `no_support` | attempted/committed/nonfinite count = 0/0/0; Q loss exact `0x0.0p+0`; finite clip limit; equal finite AMP scales; canonical integer Adam steps with after=before | `clip_scale`, both raw totals, and all 40 parameter diagnostics are exact uppercase `NA` |
| `amp_skip` | attempted/committed = 1/0; nonfinite count 1 or 2; finite Q loss and clip limit; finite AMP scales with after < before; canonical integer Adam steps with after=before | `clip_scale`, both raw totals, and all 40 parameter diagnostics are exact uppercase `NA` |

Identity and status integers are always defined. `adam_step_before` and
`adam_step_after` summarize the common weight/bias step only after requiring
the two inherited states to agree; missing inherited first-step state is
normalized observationally to integer zero and is never created or mutated
by telemetry. Missing first-step m/v tensors are likewise represented only in
telemetry as correctly shaped positive-zero before-state summaries; live
optimizer state remains absent until the inherited commit.
`nonfinite_gradient_parameter_count` is the exact inherited
count; parameter names remain in the inherited log/exception path and are
not duplicated in this minimal sidecar. A fatal finite-check, backward,
unscale, norm, transform, functional-Adam, staged validation, commit,
rollback, actor step, EMA, projection, scheduler, publication, or ack fault
produces no durable Q row.

The frozen live stages use FP16. A no-scaler FP32 test fixture serializes both
AMP scales as exact finite 1.0; it does not introduce an unavailable value.
The AMP before/after fields surround the Q optimizer decision/attempt, before
the inherited main actor optimizer step; they are not whole-transaction
scaler snapshots.

### `voc_telemetry_commits.csv`

The exact 22-column header plus LF is 410 bytes with SHA-256
`9105b143dfd260a4c491a2821757c14ded2a32c74207e6f4e7140b717ae62929`:

```text
telemetry_schema_version,gate_schema,transaction_id,source_policy_version,published_policy_version,terminal,td_first_data_row,td_data_row_count,td_block_byte_count,td_block_sha256,replay_first_data_row,replay_data_row_count,replay_block_byte_count,replay_block_sha256,q_first_data_row,q_data_row_count,q_block_byte_count,q_block_sha256,publication_count,ack_count,actor_state_sha256,publication_history_sha256
```

After inherited publication and ack, telemetry serializes the complete TD,
replay, and Q blocks in memory. It appends exactly 720, one, and one
LF-terminated data rows respectively; flushes and fsyncs each file; then
appends one commit row and fsyncs the commit file. `first_data_row` is a
one-based data-row index excluding the header; the first data row is 1.
Block byte count and SHA-256 cover exactly the concatenated LF-terminated
data-row bytes, excluding the header and excluding all other transactions.
For transaction `i` beginning at 1, TD first/count are
`720*(i-1)+1`/720, while replay and Q first/count are `i`/1. The commit row
for transaction `i` is data row `i`.

The manifest records the terminal commit data-row index and hash. Commit IDs
are contiguous; full-file manifest hashes plus sequential block ranges bind
every prior commit without an extra per-row hash-chain column. Ranges do not
overlap or gap, and every non-commit sidecar byte after its header belongs to
exactly one committed block. Orphan, partial, duplicated, reordered,
rewritten, or trailing bytes invalidate the run and are never truncated or
repaired.

## Healthy transaction and failure order

The exact healthy order is:

1. validate the consumed versioned replay batch;
2. record real step before and stage the exact inherited real-step delta;
3. compute detached online/EMA TD sources and stage FP32 clones;
4. execute inherited zero-grad, backward, unscale, raw norm, one clip, m/d
   Adam, optional gate path, and main actor step;
5. on a successful Q step, execute inherited raw EMA update and exact
   projection once;
6. execute inherited counters, actor scheduler, successful-Q scheduler, and
   optional gate scheduler;
7. write the unchanged legacy 922-column row through the existing
   `FileWriter`;
8. publish the next actor-policy bundle, wait for the exact one self-play ack, and
   commit publication state/history;
9. only now reduce/serialize the detached staged telemetry, append/fsync the
   three evidence blocks, append/fsync the commit row;
10. for a nonterminal transaction, close it and flush any deferred checkpoint;
    a terminal transaction instead enters the terminal-only seal below.

No telemetry reduction, CPU read, file open, append, fsync, hash, or
validation occurs between Q computation and publication/ack. Detached tensor
clones may be staged on the existing device stream but cannot participate in
autograd or any inherited input. The terminal transaction follows the same
order through its telemetry commit. Schema13 then uses this exact terminal-only
seal order: save the final actor checkpoint and close the policy transaction;
close/fsync the inherited
`plogger`/`logs.csv`; for each already-bound sidecar writer fd, flush, fsync
data, `fchmod(0400)`, fsync metadata, fstat-validate, and close, followed by
no-follow stable-read validation; atomically publish and validate the manifest
already at exact final mode `0400`; only then send the normal ActorBuffer
`FINISH`; then allow model
terminal seal and the driver logger-finalization branch. That branch publishes
a private request only when `use_wandb=true`; the wire retains its inherited
request-free disabled completion. Legacy logs and
checkpoints retain inherited modes. Schema13 validators reject any telemetry
file with uid/gid other than the launcher effective uid/gid, link count other
than one, alias, write/execute bit, or any group/other permission. The manifest
therefore exists before ActorBuffer `FINISH` and any private logger finish
request, so a W&B-enabled final wildcard upload and pre/post logger validators
bind all five artifacts.

If terminal telemetry finalization fails, the actor successful flag remains
false. Failure close kills/unblocks ActorBuffer without forging `FINISH`; model
seal cannot be accepted, and no logger/public finish is permitted. The
already committed terminal publication/transaction is not rolled back.

If a post-ack telemetry append/fsync/hash/commit fails, prior publication is
not rolled back and no training state is rewritten. The telemetry writer is
poisoned and raises immediately as fatal. No retry/truncation/repair occurs,
and no manifest or public finish may be produced. Partial forensic bytes may
remain exactly as written but are never accepted or reused. There is no
promise that already committed publication/training state is unchanged or
rolled back. Fatal pre-ack paths discard volatile staging and have no durable
transaction rows.

## Terminal telemetry manifest

`voc_telemetry_manifest.json` is canonical compact JSON with sorted keys,
`ensure_ascii=True`, separators `(',', ':')`, `allow_nan=False`, UTF-8, and
one final LF. Its same-directory temporary regular file is opened with
exclusive/no-follow creation at mode `0600` under umask `077`. On the same
bound fd it is written, flushed, content-fsynced, `fchmod(0400)`,
metadata-fsynced, and fstat-validated. Only then is that already read-only
inode published without overwrite using a link/no-replace primitive; a link
method unlinks the temporary name before the final run-directory fsync. The
final name is stable-fd/no-follow validated for the same inode, single link,
mode, owner, size, and bytes. No writable final-name interval is permitted.
Duplicate JSON keys, extra fields, symlink, hardlink, special file, writable
final file, replacement, or identity drift fails. There is no manifest
self-hash field; the completion checkpoint-file record binds the final
manifest bytes. No temporary name remains after successful publication.

Its exact 15-key top-level keyset, in the following documentation order, is:

```text
telemetry_schema_version
gate_schema
status
xpid
fresh
transaction_count
terminal_policy_version
terminal_real_step
terminal_publication_count
terminal_ack_count
actor_state_sha256
publication_history_sha256
legacy_actor_log
artifacts
last_commit
```

Required scalar values include telemetry schema 1, gate schema 13, status
`sealed`, exact xpid, `fresh=true`, positive contiguous
transaction count, terminal policy/publication count equal to transaction
count, terminal ack count exactly 1 from `self_play_n=1`, and
terminal real step at least the stage total. State/history hashes must equal
terminal actor evidence. Q/EMA/projection counts and scheduler state are
reconstructed from replay rows and checked against the terminal actor
checkpoint rather than duplicated in the compact manifest. Immutable
source/data manifests remain launch/completion provenance and are not
duplicated in this telemetry-only mapping.

Schema/count/size/index values are strict built-in non-Boolean integers;
`fresh` is strict built-in Boolean true; names/status/xpid are built-in
strings; digests are lowercase 64-hex. NumPy scalars, coercion, null, NaN,
extra values, and alternate types fail closed.

`legacy_actor_log` and each member of the fixed ordered `artifacts` list use
exact seven-key records:

```text
name
sha256
size
header_sha256
header_size
column_count
data_row_count
```

`legacy_actor_log.name` is `logs.csv`, its header size/count/hash are
43550/922/the frozen digest above, and its data-row count equals transaction
count. `artifacts` is a JSON list of exactly four records in this exact order:
`voc_td_cells.csv`, `voc_replay_events.csv`, `voc_q_transactions.csv`, then
`voc_telemetry_commits.csv`; each record's `name` equals the corresponding
basename. It is never a mapping and accepts no reorder. TD rows equal
`720*transaction_count`; replay, Q, and commit rows each equal transaction
count. Every hash and size is recomputed from bound final bytes.

`last_commit` has exact five keys
`{data_row,transaction_id,sha256,actor_state_sha256,
publication_history_sha256}`. `data_row` is the one-based final commit data
row excluding the header and therefore equals transaction count;
`transaction_id` equals transaction count, and `sha256` hashes that exact
commit row including LF. The referenced row must itself have terminal `1`,
published version equal transaction count, and the same actor/history hashes.

The dedicated validator derives this exact 10-key JSON-safe telemetry
evidence from the bound manifest and sidecars:

```text
telemetry_schema_version
gate_schema
manifest_name
manifest_sha256
manifest_size
transaction_count
terminal_policy_version
terminal_real_step
actor_state_sha256
publication_history_sha256
```

`manifest_name` is exactly `voc_telemetry_manifest.json`; digest and size bind
its full canonical bytes. This object is compared by deep exact equality at
every schema-13 util/public/smoke/fixed boundary. It is evidence, not a
persisted flag or checkpoint field.

## Completion, logger, and public evidence

Schemas at most 12 retain completion schema 1, the inherited exact checkpoint
triplet, logger request/completion schema 1, and exact public shapes.
Schema13 alone uses completion schema 2. The public finish marker retains its
exact seven-key outer keyset
`{schema_version,status,completed_unix,checkpoint_files,
implementation_sources,loaded_extensions,
voc_actor_policy_logger_completion}`. Within that unchanged outer keyset,
`schema_version` becomes 2 and nested `checkpoint_files` expands from the
three inherited records to exactly:

```text
config_c.yaml
ckp_actor.tar
ckp_model.tar
voc_telemetry_manifest.json
```

This is an exact four-key mapping; each value is an exact two-key
`{sha256,size}` record. JSON canonicalization sorts mapping keys and does not
change the required keyset.

The outer field names remain unchanged. Under schema13 only,
`implementation_sources` is exact15 as defined above; `loaded_extensions`
retains its inherited shape. Schemas at most 12 keep their exact14 source
mapping. No raw sidecar is a fifth checkpoint-file entry: the manifest
transitively binds `logs.csv` and all four CSV sidecars. The public finish
marker does not add an outer telemetry field.

The private logger finish request, ack, and returned logger-completion use
dedicated schema version 2 for schema13. The request retains exact six keys
`{schema_version,status,policy_version,state_sha256,
publication_history_sha256,checkpoint_files}`; the ack retains exact three
`{schema_version,status,request_sha256}`; and logger completion retains exact
ten `{schema_version,required,use_wandb,request_sha256,ack_verified,
private_markers_cleaned,policy_version,state_sha256,
publication_history_sha256,checkpoint_files}`. Their nested
`checkpoint_files` is the same exact four-record mapping, so the request hash
cryptographically binds the manifest. Before
the final W&B log/save/finish and again before ack, the logger must stable-read
and fully validate the manifest and sidecars and compare the exact telemetry
object with the authoritative final bundle. The wildcard upload includes
the five new artifacts. After ack, driver revalidation repeats before private
marker cleanup and public finish. W&B live metric keys/order remain unchanged.

A healthy wire or qualification run directory has the inherited exact
14-file stage inventory plus exactly these five telemetry artifacts, hence 19
files. No sixth telemetry file, temporary manifest, orphan sidecar, or partial
row is permitted. Primary has its inherited milestone inventory plus exactly
the same five files; its total count is derived from that inherited milestone
schedule rather than forced to 19. Private logger request/ack markers are
absent by public finish.

`validate_schema13_final_bundle` returns the inherited exact 12 fields plus
exactly `telemetry`, hence exact13. `validate_schema13_completed_bundle` constructs the
public completed record with exact 18 top-level keys: the inherited exact 17
plus only `telemetry`. Lifecycle `actor_policy` remains exact 19 and gains no
derived or telemetry field. Resolved identity remains exact12. Logger
completion outer shape remains unchanged.

Smoke local `resolved_identity` and outer
`voc_checkpoint_resolved_identity` remain exact-three mappings
`{config,actor_checkpoint,model_checkpoint}` with exact12 inner identities.
Schema13 smoke exposes the exact18 final-bundle validation separately and
the exact telemetry object within it; stored-surface identity remains three
deep-equal identity copies and does not absorb telemetry. Fixed evidence is
the inherited exact18-with-private-markers shape plus `telemetry`, therefore
exact19. Fixed `resolved_profile_identity` remains fixed-only and exact in
its inherited shape.

Any missing/extra/renamed checkpoint-file entry, manifest or sidecar;
schema1/2 confusion; malformed telemetry object; manifest/sidecar/log/header
drift; request/completion/finish disagreement; or cross-surface telemetry
drift rejects before environment, data, tensor use, rollout, W&B completion,
or public output. Validator-internal bound reads are not downstream use.

## Public, smoke, fixed, and TOCTOU order

Public evaluation classifies exact config bytes and schema13 before any
downstream action. It stable-reads/hash-binds config, completion marker,
manifest, four sidecars, logs.csv, and checkpoint files; dispatches the
dedicated schema13 validator; compares exact229/209/identity12/telemetry; and
only then loads downstream flags or constructs an environment.

Smoke order is immutable schema13 prevalidation, bound config and telemetry
bytes, private runtime copy, evaluator-only overrides, authoritative
postvalidation with exact pre/post evidence and artifact hashes, then
environment. A private runtime copy may not rewrite or omit telemetry.

Fixed profile `v20-300k` accepts only an accepted seed-5 V20 primary tuple.
It rejects wire, qualification, schema12/v19, schema13 under any non-v20
profile, and all historical profiles before downstream use or output. It
validates the exact19 fixed evidence, manifest and raw files, then creates an
evaluator-private flags copy. Stored config/checkpoints/telemetry are never
rewritten. Held-out fixed seeds remain 20260827 through 20260842, 16 streams
by 6250 real steps, total100000, unroll201, and every inherited fixed metric.

Every pathname-backed read uses stable descriptor-bound bytes. Config or
telemetry deletion, replacement, symlink, hardlink, shortening, growth,
schema13-to-legacy swap, legacy-to-schema13 swap, explicit alternate config,
sidecar swap between probe and parse, manifest swap after parse, and
checkpoint swap fail before downstream. Final revalidation catches changes
before output. No `None` evidence fallback is allowed for schema13.

## Sequential release gates

The only release sequence is:

1. Freeze this preregistration and obtain two independent read-only P0/P1
   audits on the exact bytes.
2. Implement only schema13 telemetry and dedicated validation, freeze all
   source/test bytes, pass two independent code/contract audits, the complete
   native V19 corpus, enumerated successor-admission differentials, and full
   tests.
3. Build a fresh inode-independent immutable snapshot from the authoritative
   v19 source/data baseline plus exactly enumerated schema13 overlay; pass two
   independent manifest/mode/import/test/posthash audits.
4. Run exactly one fresh seed-1 1.2k integrity wire. It decides mechanics and
   telemetry integrity only.
5. Only after wire passes, run exactly one fresh seed-1 100k qualification.
6. Only after every qualification gate passes, run exactly one fresh seed-5
   300k primary.
7. Only after primary passes may its terminal checkpoint/run artifacts receive
   one fixed confirmation under exact profile `v20-300k`.

Any failure permanently ends V20 at that stage. There is no retry, resume,
repair, replacement run, alternate seed, partial telemetry salvage, selected
checkpoint, speculative primary, fixed rescue, Pong, or Space Invaders.

## Integrity-wire acceptance

Wire must exercise at least one supported successful Q update and at least
one complete terminal transaction. It must prove the entire inherited v19
Huber/common/orthonormal/tau1/projection transaction, exact229/209,
identity12, barrier/history/ack, model seal/drain, W&B-disabled completion,
source/runtime pins, finish, and process/Ray/GPU closure.

Telemetry-specific wire gates require:

- exact headers and fresh single-link files;
- exactly 720 TD rows, one replay row, one Q row, and one commit row per
  policy transaction;
- both q sources, all dense empty/nonempty cells, exact depth/sign/band order,
  and TD sufficient-stat reconstruction;
- a stepped Q row whose staged gradients, m/d transform, Adam candidate,
  mapped delta, and terminal checkpoint reconcile;
- exact transaction/log/publication IDs and primitive real-step/support/
  counter/scheduler facts;
- post-ack-only append, per-block hashes/ranges, fsync order, complete commit
  sequence and file hash, terminal manifest, schema2 logger binding and upload, exact telemetry
  evidence equality, and private cleanup; and
- no changed legacy CSV/W&B payload, tensor/state/RNG/counter, or inherited
  artifact beyond closed schema/xpid/path/completion substitutions.

Negative no-support and AMP-skip matrices are frozen in tests and need not be
forced live. Any missing row/cell, noncanonical byte, unsupported `NA`, hash
drift, pre-ack append, telemetry-dependent training branch, Q/EMA/projection
drift, schema1 laundering, or cleanup residue permanently fails wire.

## Frozen 100k qualification

The numerical population and all thresholds are unchanged. Canonical rows
are finite complete legacy `logs.csv` rows satisfying
`70000 < real_step <= 100000`, with windows `(70000,80000]`,
`(80000,90000]`, and `(90000,100000]`; overshoot is excluded. Steps are
unique and strictly increasing. Telemetry never changes row eligibility.

Qualification passes only if all inherited gates pass together:

- teacher gap at least `0.075`, student gap at least `0.05`, retention at
  least `0.50`, and signed margin strictly positive;
- at least two of three windows have positive student gap and margin;
- every trailing-five endpoint has strictly positive positive-sign and
  negative-sign denominators and maximum consecutive negative pooled-gap run
  at most 3;
- train and holdout CONTINUE/STOP fractions each strictly above `0.05`;
- wrong-CONTINUE saturation below `0.01`, with inherited wrong-STOP and
  forced-stop requirements;
- online/EMA non-tie sign agreement at least `0.60`;
- held-out EMA selected-action TD RMSE at most `0.5`; and
- every inherited actor/Q/gate/model/protocol/AMP/nonfinite/mismatch/
  malformed/timeout/seal/abort counter requirement passes.

Undefined is never zero or nonnegative. No pseudocount, endpoint removal,
window shift, support substitution, threshold relaxation, row weighting, or
telemetry-selected subpopulation is allowed.

Schema13 adds integrity gates, not numeric thresholds: manifest/sidecars must
reconstruct exact counts and source-specific sufficient statistics; online
and EMA cells must independently reconcile; primitive replay/counter/
scheduler/publication facts must agree; and fixed-input noninterference must
pass. A telemetry failure fails qualification even if all 15 numerical gates
pass. A numerical failure fails qualification even if telemetry is perfect.

## Frozen 300k primary and fixed acceptance

Primary retains Full `(100000,300000]`, Late `(250000,300000]`, and W1/W2/W3
`(270000,280000]`, `(280000,290000]`, `(290000,300000]`. Overshoot is
excluded. Every inherited threshold remains unchanged, including soft-gate
probability `0.475/0.525`, sampled-control strength `0.525`, conditional
argmax accuracy `0.60`, useful-pair coverage `0.95`, sign agreement `0.60`,
strict support fractions above `0.05`, wrong-side saturation and forced-stop
below `0.01`, held-out RMSE at most `0.5` where inherited, exact window/
direction/strength requirements, absolute supports, four frozen behaviors,
and zero skip/nonfinite requirements.

Training epsilon remains 0.02; execution epsilon remains 0.25. Sampled
execution, likelihood, V-trace, entropy, default behavioral accuracy,
sampled no-op, forced action, calibration, saturation, denominator, and
support definitions remain v19. Telemetry supplies no alternate metric,
threshold, or fixed selection.

The fixed evaluator uses training evidence only after accepted primary and
the exact `v20-300k` route. It must validate schema2 completion and telemetry
before evaluator-private overrides. Runtime epsilon/barrier/seal overrides
remain evaluation-only and may not alter stored training evidence.

## Implementation and test matrix

The anticipated production overlay is confined to the schema/runtime and
public files needed for schema dispatch, telemetry staging/writing,
schema2 finalization, and validation: `thinker/train.py`, `thinker/thinker/util.py`,
`thinker/thinker/actor_net.py`, `thinker/thinker/learn_actor.py`,
`thinker/thinker/main.py`, `thinker/thinker/self_play.py`,
`thinker/thinker/learn_model.py`, `thinker/evaluate_dynamic_imitation.py`,
`thinker/smoke_dynamic_imitation.py`, and
`thinker/evaluate_voc_fixed_checkpoint.py`, plus the sole new production path
`thinker/thinker/voc_telemetry.py`. Tests may use a new dedicated telemetry
test file plus the existing dynamic contract, actor, learner, seal, barrier,
public, smoke, and fixed suites. The final overlay/topology allowlist must be
enumerated and hash-frozen before snapshot; any other new production path or
extra production edit is a preregistration violation.

Required tests include all of the following.

### Schema, surface, lineage, and freshness

- exact constant/API names, strict schema13 built-in integer, exact stages,
  fresh tuple values, profile `v20-300k`, and lexical malformed-prefix
  rejection before all I/O;
- exact 96-pair stage commands with tau1 once, no schema/telemetry CLI pair,
  strict xpid-based inference, and rejection of an explicit schema or 97th
  pair before downstream I/O;
- exact229 keyset, exact209 keyset/4457 bytes/ad22 digest, direct proof that
  `voc_gate_policy_schema_version` is absent from the projection, no 230th
  key, and no telemetry or new derived key in config/checkpoints;
- exact12 inner identity with schema13 and unchanged three strings;
- schema13 with wrong/missing schema, tau, barrier, seal, stage, profile,
  config type, checkpoint/resume/preload/parent, or derived forgery rejects;
- string subclass, NumPy scalar, bytes, PathLike, custom lexical object,
  malformed UTF-8, Boolean/int/float near-value, cycle, nested reserved key,
  and cross-schema attacks reject before run-dir/config/checkpoint I/O;
- schemas1-5 nondefault historical tau differentials and exact schemas6-12
  byte/value/exception/path/marker/public differentials; and
- the native immutable V19 corpus fully green, plus current-production versus
  V19-corpus results green except an exhaustively enumerated exact set of
  forward-unknown assertions whose sole conflict is admitting built-in schema
  13; every schema-at-most-12 parameter, behavior, error, path, marker, and
  public case must pass unchanged; and
- fresh telemetry files only, no reuse, resume, repair, or cross-stage state.

### TD cube algebra and reductions

- exact 22-column header bytes/digest and 720-row Cartesian order;
- strict per-row telemetry schema1/gate13 and exact cross-file transaction,
  version, real-step, terminal, and state/history joins;
- online/EMA selected-Q independently gathered, exact FP32 `target-selectedQ`,
  exact sign rather than Q-gap tie, and action mapping;
- decision-depth STOP adjustment and all six exact bins, boundary depths,
  negative/uncovered rejection, and no raw-search-depth substitution;
- every absolute-band boundary, signed zero to zero sign/band, finite checks,
  dense empty cells, positive-zero encoding, and no overlap/drop;
- exact train/holdout isolation and no effect on q loss/masks;
- online/train-only beta1 Huber reconstruction from the five absolute bands,
  tolerant equality to FP32 `q_loss_sum` and its logged supported mean, and
  explicit rejection of EMA/loss-route substitution;
- deterministic row-major FP32-to-binary64 promotion, `math.fsum`, product,
  square, L2/max, hex serialization/reparse, repeated-run byte identity, and
  count/moment reconstruction;
- legacy GPU aggregate comparisons using the frozen forward-error bound,
  never false bit-equality; and
- injected source swap, mask drift, source-label swap, row reorder, duplicate,
  omission, wrong count, malformed float/hash/enum, CR/BOM/quote/blank/extra
  column rejection.

### Replay and transaction diagnostics

- exactly one replay row per consumed batch/policy transaction and never one
  row per real transition;
- contiguous transaction/publication/log ticks, source predecessor, actual
  T/optimized-T/B, positional actor-ID encoding/hash/set validation,
  real-step before/delta/after, support/action counts, and terminal flag;
- cross-stage replay dimensions exactly 42/41/16 for wire and 202/201/16 for
  qualification/primary, all derived from unroll length rather than a global
  constant;
- exact stepped/no_support/amp_skip status and attempted relation;
- Q/EMA/projection counters and Q scheduler epoch/step/LR before/after,
  publication/ack counts, state/history hashes, and checkpoint reconciliation;
- exact `self_play_n=1` ack count 1 versus 16 positional actor streams, with
  explicit rejection of any actor-count-as-ack substitution;
- explicit proof that no batch ID, quantum, credit, backlog, reuse, restore,
  within-batch schedule, or conservation state is inferred;
- exact Q header and status/`NA` matrix, raw pre/postclip clones, one clip,
  m/d transform, m/v before/after, normalized update, functional coordinate
  delta, inverse raw delta, and no second optimizer calculation;
- pinned functional-Adam oracle, raw `.grad` unchanged by telemetry, candidate
  snapshots read before atomic commit, and exact terminal state reconciliation;
- exact overlap-inclusive clip limits 42*16 and 202*16, captured single-call
  clip coefficient, no second norm/clip, and tolerant raw norm-ratio check;
- exact m/v bias-correction formula at `adam_step_after`, actual scratch delta,
  actual normalized delta/LR, inverse mapping, and formula-to-actual checks
  under the frozen forward-error bound rather than bit equality;
- no-support, FP16 amp-skip, FP32 success, zero gradient, clip/no-clip,
  first-step state, terminal state, nonfinite parameter count, and every field's
  defined/unavailable status; and
- fatal precommit, functional, staged-nonfinite, commit, rollback-failure,
  actor, EMA, projection, scheduler, publication, and ack injections produce
  no durable transaction row and no false unchanged-state promise.

### Append, commit, manifest, and logger

- exact four headers, modes, exclusive creation, no alias/link/special file,
  exact creation mode0600/final mode0400, four-header/run-directory fsync
  durability, and no use of legacy `FileWriter` for sidecars;
- failure injection at every pre-v0 schema/profile/path/header create/write/
  fsync/directory-fsync boundary with no policy/training state, transaction
  evidence, or finish and no reuse of partial initialization bytes;
- assertion that no reduction/I/O occurs before publication/ack and exact
  healthy order through legacy log, publish/ack, blocks, fsyncs, commit,
  close, checkpoint flush, manifest, logger request, upload/ack, finish;
- terminal-only order checkpoint -> legacy-log close/fsync -> sidecar close/
  fsync/validate/mode0400 -> manifest publish/validate -> ActorBuffer FINISH ->
  model seal -> private logger, with injected finalization failure retaining
  successful=false and killing/unblocking without a forged FINISH;
- exact block rows/bytes/ranges/hashes, one-based first-row indexing,
  complete commit-file range/hash binding, transaction counts, no gap/
  overlap/orphan/trailing bytes, and terminal commit identity;
- interruption after each row/block flush/fsync/commit/rename/link/directory
  fsync; append short write; ENOSPC; stale handle; identity replacement; and
  poison/no-retry/no-repair/no-finish behavior;
- descriptor-bound sidecar flush/content-fsync/fchmod0400/metadata-fsync/
  fstat/close order and manifest temp0600 -> content fsync -> fchmod0400 ->
  metadata fsync -> no-replace publication -> temp unlink -> directory fsync,
  with no pathname chmod or writable final-name interval;
- exact compact manifest top15, ordered four-record `artifacts` list, exact7
  file records, exact5 last-commit record, counts, logs header binding,
  state/history/counter relations, atomic publication, read-only seal,
  duplicate-key/extra/missing/wrong-type attacks;
- manifest missing/extra sidecar, sidecar rename/swap, wrong header/hash/size/
  row count, config/checkpoint/manifest cross-run mix, and terminal drift;
- schema2 private request/ack/logger completion/public finish with unchanged
  outer keysets, exact four checkpoint records, pre/post W&B validation,
  wildcard artifact inclusion, private cleanup, and public exact18 telemetry;
- W&B-disabled wire still emits/validates all local telemetry and public
  schema2 completion without a remote request; and
- schemas<=12 retain exact schema1 marker, triplet, logger, public record,
  fixed evidence, and no telemetry file or parser call.

### Public, smoke, fixed, TOCTOU, and noninterference

- dedicated schema13 validation before flag loader, environment, tensor use,
  data, action, rollout, W&B close, or output; no `None` fallback;
- config/manifest/sidecar/checkpoint deletion or swap at every probe/load/
  dispatch/revalidation boundary using bound fds/bytes;
- public exact18, smoke exact18 plus exact-three stored identities, fixed
  exact19, lifecycle actor-policy exact19 unchanged, exact types, deep-copy
  equality, and extra/missing/drift attacks;
- every incompatible legacy/V15-V19 profile and reciprocal schema/profile
  mismatch rejects with zero downstream calls; valid historical routes remain
  byte-compatible but permanently failed runs remain unauthorized;
- fixed-input schema12/schema13 equality for tensors, losses, gradients,
  params, optimizer/scaler/scheduler, EMA/projection, RNG, counters, bundles,
  history, acks, legacy stats/CSV bytes, and W&B metric payload after closing
  identity substitutions;
- telemetry staging order randomized in a test-only oracle yields identical
  training outputs and canonical sidecar bytes; no telemetry-dependent branch
  or RNG consumption; and
- full canonical CPython3.10 no-cache suite, the native V19 corpus and exact
  enumerated successor-admission differential above, pycompile outside source,
  double source/data/posthash, cache/run topology, process/Ray/GPU cleanup,
  and two independent read-only audits.

## Explicit falsifiers and claim boundary

V20 is falsified for release by any behavior/config/state delta beyond the
closed schema/evidence/file substitutions; any 922-column or W&B metric drift;
any missing/malformed/unbound telemetry byte; any reconstruction,
noninterference, append/order, manifest, logger, public, TOCTOU, source, or
freshness failure; any wire mechanics failure; or any unchanged numerical
gate failure at qualification, primary, or fixed.

Telemetry may reveal target tails, Q-source tails, composition, clipping,
coordinate moments/deltas, or cadence associations. It cannot by itself
identify causality, authorize a new knob, select a post-hoc subgroup, relax a
denominator, or predict a pass. A future cadence or coordinate intervention
requires a separate frozen preregistration with exactly one prospectively
chosen semantic delta and value.

V20 requires, in order, accepted exact preregistration bytes, accepted frozen
implementation/tests, an accepted immutable snapshot, exactly one accepted
wire, exactly one accepted fresh seed-1 100k qualification, exactly one
accepted fresh seed-5 300k primary, and exactly one accepted `v20-300k` fixed
confirmation. Any failure is permanent. This document itself implements,
launches, evaluates, retries, repairs, resumes, or authorizes none of them.
