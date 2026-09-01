# Enduro VoC-v14 sealed-model-input preregistered acceptance

This protocol is frozen after the sole VoC-v13 integrity wire failed and
before any v14 immutable snapshot, wire, qualification, primary training, or
fixed-checkpoint evaluation.  V14 is a separately named, result-informed
integrity repair.  Relative to v13 it makes exactly one prospective mechanism
change:

1. schema 7 adds an acknowledged terminal ModelBuffer input seal and a
   strictly bounded zero-or-one-update ModelNet drain before successful
   completion.

V13 changes A-C are inherited unchanged: execution epsilon remains exactly
0.25 while soft training epsilon remains exactly 0.02, the acknowledged actor
policy-version barrier remains enabled, and the main actor GradScaler still
starts at 32.  V14 changes no Q target, loss, EMA target, exact projection,
gate feature, reward, environment, main-action policy, behavioral population,
support floor, window, or numeric acceptance threshold.

## Permanent v13 wire failure and immutable evidence

The only v13 wire,
`enduro-voc-v13-versioned-eps25-sentinel-wire1200`, is a permanent failure.
It may not be continued, retried, repaired in place, or reclassified.  Its
immutable snapshot remains
`/tmp/di-voc-v13-versioned-eps25-final-CnOCd9`, with:

- v13 protocol `VOC_V13_VERSIONED_EPS25_300K_ACCEPTANCE.md` SHA-256
  `c84ff2b0bcde1e5e9eb80ea6cf647d4ffc0b8e238a0a9a265852c6b681b03d73`;
- source manifest SHA-256
  `ace67bd9eb29d47c5d094743aac668c2bea4c43485ad1ded2f671000b06d7ac0`
  over exactly 1061 source files; and
- staged Enduro data manifest SHA-256
  `23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343`
  over exactly 11 files.

The wire launched from that exact snapshot with seed 1, total_steps 1200,
schedule_total_steps 100000000, model warm-up 512, actor unroll 41, W&B false,
CUDA devices 0,1, and Ray requesting two GPUs.  Actor training completed its
sole terminal publication at real_step 1344 and policy version/publication
count 8/8.  The terminal actor step checkpoint was committed at
2026-08-28 10:02:43.359899591 +09:00.  SelfPlay then took no further
environment action and exhausted the exact monotonic 120.0-second model-close
bound; the driver recorded failure at
2026-08-28 10:04:43.220191209 +09:00 and rejected
`return_codes=[False]`.  No public `finish`, private logger-finish request, or
private logger-finish acknowledgement exists.

This was not an optimizer hang.  Model telemetry proves successful lockstep
ModelNet m/p updates through in-memory real_step 688 and 848, with update
counts 4/4 and 8/8.  At process teardown the ModelLearner traceback is exactly
the `model_update=False` replay-throttle status poll in `learn_model.py`, not
an optimizer operation.  The v13 loop copied `status.replay_ratio` but ignored
`status.processed_n` while throttled.  Once the terminal actor stopped all
writes, ModelLearner could neither lower the replay ratio nor advance its
stale progress to total_steps; only ModelLearner could set ModelBuffer finish,
while SelfPlay waited for that finish without stepping the environment.  This
is the circular terminal-drain defect repaired prospectively by Change D.

The failed run's exact post-cleanup artifact SHA-256 values are:

- `config_c.yaml`:
  `29ba1ea6ab604ca0d1edae250f3a5680713da94d4122ad9e3b9a686d55b5556c`;
- `ckp_actor.tar`:
  `485fa58f382b70a0580e489ae9283bdbb473f8eed4e908f031285bd4b05ec6e1`;
- `ckp_actor.tar_step_96`:
  `8ceacc6edf25ecc5a505f3147f67d64212cdc76b66ed5c19909e42cc918499d4`;
- `ckp_actor.tar_step_1344`:
  `8df93cb766563fb2e879b86e0f76eeb848600edf0a533f8db1238ecd49e64cb2`;
- stale `ckp_model.tar`:
  `8249a7b90374000bc01570ec56115969e339a283381ae3d54037412cabc72d4c`;
- `ckp_model.tar_step_512`:
  `a3d154c69c56c8aca1fd4b85b62c0833e70969766b9fd7b0e70df06b50fab8f6`;
- actor/model CSV:
  `b086b1aca15adf4323cbf018ea461ab40b0c0fd4ee1a3433ad6bb3ccf6cf753b`
  and
  `ca2475c753d5354076ffea7e3576a49d7a1e5b1fca729de80db558feac6345dd`;
- actor/model metadata:
  `ccbee33c0afd65bde6e895c99e90f83e4a696ece4f09a6426e643247015e42b3`
  and
  `020cc6f65ef0ecebb3c755617c1d6525458908b666f5d562c17e428d8cd225d5`;
- actor/model text logs:
  `048f0db15c9002bb580a57bf9924b4ff9e3d6af769d368241d2dcee2e4d800cd`
  and
  `e1b5ac3826eb400f37cb6d1c5b793b4973802253d5d59536e5db83d25b5eb072`;
  and
- captured driver log:
  `e090d5c14cfcda32e3d0ee9662882cd8c2ee33f7e1086ce2f99d9759951f1366`.

The canonical C-sorted run-file content-tree digest is
`361188c382761baee634dc7a0abbc16f85840fca558f059ebc9c0fbe5ba18337`.
Its corresponding path/type/mode/link-count/size structure digest is
`6e1e61c3b46d8dd9c791b37b381c23719a446e8fdfdd62659ede8ec8c9461866`.
The preserved runtime `/tmp/v13wire-MZa11y` has content digest
`600d9ed56c6b1b23e73231301b276575ce6ccad9589877482b4dd92beb6dfdcb`
and structure digest
`7c889fb3ae87f47d6a9f15c3d8b1948764b09ca2d627560c646d34ca25c7a6cd`.
The failed terminal ModelNet checkpoint records real_step 512 and m/p gradient
step counts 1/1, so the full final-bundle validator correctly rejects it as
stale against total_steps 1200.  Source and staged-data manifests still pass
strict post-run verification, all v13 runtime/Ray/GPU processes are gone, and
the failed run and runtime evidence remain preserved.

No v13 qualification, primary, or fixed evaluation is authorized.  V14 is
one prospectively frozen replacement experiment, not a second v13 attempt.

## Normative inheritance from v13

The hashed v13 protocol identified above is incorporated verbatim except for
the following closed substitutions and additions:

- the experiment name changes from v13 to v14;
- gate-policy schema 6 changes to schema 7;
- Change D and its one new identity/evidence contract are added;
- the three stage xpids and closed fixed profile change to the v14 identities
  below; and
- release ordering starts with one separately named v14 replacement wire.

V13 sections Change A, Change B, Change C, Unchanged v12 mechanism, the
schema-6 W&B two-phase completion contract, frozen 100k qualification, frozen
300k primary acceptance, and fixed-checkpoint behavioral/calibration algebra
otherwise control verbatim.  In v14, references in those inherited integrity
contracts to schema 6 mean their unchanged schema-7 successor plus Change D.
This substitution cannot weaken a type, topology, timeout, no-retry,
publication-history, logger-completion, artifact, behavior, support, window,
or numeric requirement.

The v12 and earlier permanent failures and immutable identities recorded by
v13 also remain frozen.  V14 cannot pool with, rescue, or reinterpret any
v9-v13 observation.

## Change D: schema-7 sealed ModelNet input and terminal drain

V14 uses `voc_gate_policy_schema_version=7` and introduces the exact identity
`voc_model_input_seal_schema_version=1`.  The latter is a strict non-boolean
Python integer in configuration, runtime state, actor and ModelNet embedded
flags, checkpoints, resume/smoke/public/fixed validation, and resolved
identity records.  Gate-policy schemas 1 through 6 retain the exact legacy
value/default zero and cannot silently acquire sealing behavior.

The actor-policy bundle and acknowledgement shapes do not change.  Every
schema-7 bundle contains exactly
`{bundle_schema_version, policy_version, terminal, gate_schema,
actor_state_dict}`, with strict bundle_schema_version 1 and gate_schema 7.
Every acknowledgement contains exactly
`{bundle_schema_version, gate_schema, rank, policy_version, terminal}`, with
the same strict schema identities.  Publication-history events retain the
existing exact seven-key shape
`{predecessor_version, policy_version, publication_count, terminal,
ack_ranks, expected_ack_count, state_sha256}` and do not add gate_schema; the
surrounding schema-7 checkpoint plus the independently validated bundle and
acknowledgements bind the history to schema 7.  Missing, additional, or
wrongly typed keys and any gate_schema other than 7 fail closed.

The input seal changes terminal coordination only.  All preterminal training
examples, replay sampling/distribution, and optimizer execution are unchanged.
Optimizer algorithms and hyperparameters, losses, learning-rate formulae,
precision, architectures, neural forward outputs, environment transitions,
actions, rewards, and acceptance metrics are unchanged.  Artifact metadata
adds only the schema-7 identity and the exact ten terminal evidence fields
specified below.  Change D may add exactly the documented zero-or-one
post-seal terminal drain update, so the realized terminal optimizer step count
may be one greater than it would have been without the drain; it permits no
other new sample or update.

### Exact terminal ordering

The successful schema-7 order is exactly:

1. The worker completes its last preterminal unroll, issues every associated
   ModelBuffer write, and receives the ActorBuffer enqueue acknowledgement.
2. ActorLearner consumes that version, completes the v13-unchanged actor and
   Q/EMA/projection transaction, and publishes the sole terminal theta_(v+1).
3. The worker loads and acknowledges that terminal policy; ActorLearner then
   validates the complete actor-policy acknowledgement set.  The worker
   performs zero further environment actions.  With the same monotonic
   120.0-second bound, it awaits the exact ObjectRef for its last issued
   ModelBuffer write and requires successful acknowledgement.
4. Still without an environment action, the worker makes one bounded
   `seal_input` request with expected_min equal to total_steps.  ModelBuffer
   atomically closes input, increments its seal count exactly once, freezes
   `terminal_processed_n` as a strict non-boolean integer at least
   total_steps, and acknowledges that exact state.  A later write is rejected
   and counted; no second or conflicting seal is valid.
5. ModelLearner observes the acknowledged seal and snapshots its pre-drain
   real_step and m/p gradient-step counts.  If pre real_step equals
   terminal_processed_n, it performs zero drain updates.  If pre real_step is
   smaller, it cancels or frees every stale pre-seal prefetch, performs
   exactly one fresh post-seal read whose processed_n equals the frozen
   terminal_processed_n, and performs exactly one successful lockstep m+p
   optimizer update.  That one update is the sole terminal exception to the
   replay-ratio cap.  Missing/None/stale data, a progress regression, a
   second drain, an AMP skip, a non-finite value, or a partial optimizer update
   fails closed.
6. Final ModelNet real_step must equal terminal_processed_n.  Final m/p
   gradient-step counts must equal their pre-drain counts plus the drain count.
   ModelNet state, optimizers, schedulers, scalers where applicable, counters,
   and learning-rate state must be recursively finite and mutually exact.
7. ModelLearner force-saves the complete terminal ModelNet checkpoint with all
   ten evidence fields below.  Only after that exact checkpoint has been
   durably saved and validated may ModelBuffer accept `complete_success`, mark
   successful finish, and acknowledge completion.
8. SelfPlay waits for that success finish, resolves ModelLearner to the exact
   Python value `True`, and returns exact `True`.  The driver then performs the
   unchanged full-bundle, logger, and public-finish validation sequence.

Every ordinary schema-7 ModelNet update has one additional bounded
linearization handshake immediately before it consumes data.
`begin_model_update(expected_processed_n)` returns exactly
`{allowed, token, status}`.  `allowed` is an exact Python boolean; an unsealed,
healthy ModelBuffer with no active claim returns true and a new strict
non-boolean Python integer `token`, while a sealed, aborted, or finished buffer
returns false and exact null token.  In both cases `status` is the exact
schema-7 status mapping.  A true claim linearizes that one ordinary update
before a later seal, so the claimed update may finish even if the seal closes
input concurrently.  After its successful consume, ModelLearner calls
independently bounded `end_model_update(token)`, which returns exactly
`{token, status}` with the same strict token and exact schema-7 status mapping,
and clears the claim.  Duplicate, stale, mismatched, malformed, or concurrently
active claims fail closed.  The schema-7 runtime status adds only the
non-persisted exact Python boolean
`voc_model_update_claim_active`; it adds no checkpoint evidence field or
229-key identity field.

Every `status` above is the exact 13-key Python mapping
`{processed_n, warm_up_n, replay_ratio, running, finish,
voc_model_input_seal_schema_version, voc_model_input_sealed,
voc_model_input_seal_count, voc_model_terminal_processed_n,
voc_model_input_late_write_count, voc_model_input_abort_count,
voc_model_input_aborted, voc_model_update_claim_active}`.  Progress and count
fields `processed_n`, `warm_up_n`, seal count, late-write count, and abort
count are strict non-boolean nonnegative Python integers; replay_ratio is an
exact finite nonnegative Python float; the five state fields are exact Python
booleans; and seal schema is strict integer 1.  `running` is equivalent to
processed_n reaching the configured warm-up.  Before sealing, terminal
processed_n is exact null; after sealing it is a strict integer equal to
processed_n, at least total_steps, with seal count equal to the expected
producer count.  Abort has count 1 and forbids finish; no abort has count 0.
Abort or finish forbids an active claim, and successful finish additionally
requires sealed input, zero late writes, and zero aborts.  Missing, extra,
coerced, non-finite, or relationally inconsistent status data fails closed.

A seal closes writes and denies every later begin request even while one
pre-seal claim is finishing.  ModelLearner must end that claim before the
terminal pre-count snapshot and zero-or-one drain decision.  `complete_success`
requires no active claim.  Abort clears an active claim, never launders it into
successful finish, and preserves the ordinary failure semantics below.  Each
begin, end, status, read, complete-success, and abort RPC has its own exact
monotonic 120.0-second bound.

No code path may substitute a stale prefetch for the fresh drain read, stamp a
new real_step onto stale optimizer/scheduler state, set success finish before
the forced checkpoint, or take an environment action after loading the
terminal actor policy.

### Exact terminal ModelNet checkpoint evidence

Every successful schema-7 ModelNet checkpoint contains these exact top-level
fields in addition to the unchanged full ModelNet training state:

- `voc_model_input_seal_schema_version`: strict non-boolean Python integer 1;
- `voc_model_input_sealed`: exact Python boolean true;
- `voc_model_input_seal_count`: strict non-boolean Python integer 1;
- `voc_model_terminal_processed_n`: strict non-boolean positive Python integer;
- `voc_model_terminal_drain_update_count`: strict non-boolean Python integer
  in the closed set {0, 1};
- `voc_model_terminal_drain_pre_real_step`: strict non-boolean nonnegative
  Python integer;
- `voc_model_terminal_drain_pre_grad_step_count_m`: strict non-boolean
  nonnegative Python integer;
- `voc_model_terminal_drain_pre_grad_step_count_p`: strict non-boolean
  nonnegative Python integer;
- `voc_model_input_late_write_count`: strict non-boolean Python integer zero;
  and
- `voc_model_input_abort_count`: strict non-boolean Python integer zero.

The validator requires all of the following relations exactly:

- `voc_model_terminal_processed_n == model real_step >= total_steps`;
- final m and p gradient-step counts are positive and lockstep under the
  unchanged dual-net requirement;
- each final m/p count equals its corresponding pre-drain count plus
  `voc_model_terminal_drain_update_count`;
- drain count zero requires pre-drain real_step equal final real_step and both
  pre-drain counts equal their final counts; and
- drain count one requires pre-drain real_step strictly less than final
  real_step and each final count exactly one greater than its pre-drain count.

Boolean-as-integer values, floats, strings, missing or extra evidence fields,
negative counters, a terminal processed count below total_steps, a pre-drain
real_step above the terminal count, unequal final progress, inconsistent
optimizer/scheduler/counter state, more than one drain update, any late write,
or any abort fails closed.

### Failure and abort semantics

A failed or timed-out last-write acknowledgement, actor terminal
acknowledgement, seal, fresh read, drain update, force-save, validation,
`complete_success`, ModelLearner result, or SelfPlay result is a hard stage
failure.  Duplicate, missing, malformed, future, stale, or conflicting seal
state and every post-seal write are also hard failures.  The failure path
aborts model input when possible, never sets successful ModelBuffer finish,
never returns worker success, never writes logger completion or public
`finish`, and preserves failure evidence.  An abort count can therefore
appear only in a failed/ineligible artifact; every qualifying completed
checkpoint requires it to be exactly zero.

The existing 120.0-second monotonic barrier bound and zero Ray restart/retry
contract apply independently to last-write acknowledgement, input sealing,
ModelNet success finish, ModelLearner result, and SelfPlay result.  A timeout
cannot be converted into a success, retry, longer bound, or replacement run.

## Closed configuration and identity surface

V14 preserves the exact frozen v12 209-key stage-neutral projection and its
canonical SHA-256
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.
It preserves the six closed stage keys and four path/command-derived keys
defined by v13.  Its ten schema-specific keys are exactly the prior nine:

- `actor_amp_init_scale`;
- `voc_gate_execution_epsilon`;
- `voc_actor_policy_version_barrier`;
- `voc_actor_policy_bundle_schema_version`;
- `voc_actor_policy_barrier_timeout_s`;
- `voc_actor_policy_ray_max_restarts`;
- `voc_actor_policy_ray_max_task_retries`;
- `voc_gate_policy_schema_version`; and
- `voc_actor_policy_barrier_runtime`;

plus exactly `voc_model_input_seal_schema_version`.

Their v14 values are the unchanged v13 values except for exact
`voc_gate_policy_schema_version=7` and exact
`voc_model_input_seal_schema_version=1`; barrier runtime remains exact true.
Thus each complete configuration, actor embedded-flags mapping, and ModelNet
embedded-flags mapping contains exactly 229 keys: baseline 209, stage 6,
path/command 4, and schema-specific 10.  The three surfaces must have identical
key sets and values.  Any missing, additional, normalized, coerced, or
cross-surface-mismatched key fails closed.

Public evaluation, smoke, resume, promotion, and fixed evaluation must first
validate the persisted 229-key training identity and the complete terminal
actor and ModelNet checkpoints.  Their resolved JSON-safe records must include
the schema-7 seal identity and all ten terminal ModelNet evidence fields and
relations.  Only after immutable training identity and terminal evidence
validation may a private evaluation-flags copy set training and parallel
execution false, disable live actor-barrier waiting, and make live ModelBuffer
seal coordination ineffective.  That private runtime record must preserve and
report the immutable training values: seal schema 1, soft epsilon 0.02, and
execution epsilon 0.25, while reporting effective barrier wait and model-input
seal coordination false.  For schema 7, public
`evaluation_runtime_flags.immutable_training` contains exact `train_model`
and `voc_model_input_seal_schema_version`, while
`evaluation_runtime_flags.evaluation_copy` contains exact
`train_model=false` and
`effective_model_input_seal_coordination=false`.  The fixed protocol and
manifest contain exact `training_model_input_seal_schema_version=1` and
`runtime_model_input_seal_coordination=false`.  No persisted mapping or
checkpoint may be rewritten.  Schema-6 public runtime-record shape is
unchanged.

Schemas 1 through 6 preserve their historical complete surfaces and exact
seal-schema value/default zero.  Legacy public/fixed profiles reject schema 7
or seal schema 1.  V14 profiles reject a legacy/defaulted seal field, schema 6,
or any v13 complete surface.

## Immutable stage identities and no-retry rule

Every v14 stage is fresh: ckp=false; preload, preload_actor, and
voc_parent_checkpoint are empty; parent-update count is zero; actor policy
version starts at zero; and online-Q, EMA-Q, and projected scalar gate start
from the unchanged exact fresh state.  No checkpoint, buffer, version, seal,
or observation crosses stage boundaries.

The only valid stage tuples
`(xpid, base_seed, total_steps, model_warm_up_n, actor_unroll_len, use_wandb)`
are exactly:

- (`enduro-voc-v14-sealed-eps25-sentinel-wire1200`, 1, 1200, 512, 41,
  false);
- (`enduro-voc-v14-sealed-eps25-seed1-qual-fresh-100k`, 1, 100000, 10000,
  201, true); and
- (`enduro-voc-v14-sealed-eps25-seed5-strict-fresh-300k`, 5, 300000, 10000,
  201, true).

Each xpid is an exact Python string identity.  Trimming, coercion, case change,
or whitespace normalization is forbidden.  Numeric tuple members are strict
non-boolean integers and use_wandb is a strict Python boolean.  Configuration,
actor metadata, and ModelNet metadata must agree on the complete tuple, and
the normalized ckpdir basename must equal the exact xpid.

The seed-1 wire is nonqualifying.  The seed-1 100k qualification is a separate
fresh run, not a continuation of the wire.  The seed-5 300k primary is a
separate fresh run and the one prospectively fixed primary attempt.  Seed 1
cannot back up seed 5.  No stage may be resumed, extended, selected by result,
replaced by another seed, or retried.  Failure of the single v14 wire ends v14;
failure of qualification forbids primary; failure of primary or fixed
confirmation ends the v14 claim.

The closed fixed profile is exactly `v14-300k` and accepts only the primary
tuple.  It rejects wire and qualification artifacts from their stage identity
before any held-out rollout.

## Sequential release gates

The release order is exact:

1. Build a fresh inode-independent immutable v14 snapshot from the preserved
   v13 source/data baseline plus an exactly enumerated stable v14 overlay.
   Independently audit manifests, modes, binaries, runtime binding, schema-7
   validators, tests, and this document before launch.
2. Run exactly one separately named seed-1 1.2k v14 replacement integrity
   wire with the frozen wire tuple, CUDA devices 0,1, Ray requesting two GPUs,
   and no W&B.  It is not a retry of v13.
3. Only after the wire passes every integrity gate, run exactly one fresh
   seed-1 100k qualification with production warm-up/unroll and W&B enabled.
4. Only after qualification passes every inherited v13 qualification gate and
   every schema-7 integrity gate, run exactly one fresh seed-5 300k primary.
5. Only after that primary passes all inherited training/artifact gates may
   its one terminal checkpoint receive one eligible `v14-300k` fixed
   confirmation.

Pong and Space Invaders remain out of scope until every Enduro v14 stage
passes.

## Integrity-wire acceptance

The wire decision may inspect only immutable binding, exact configuration and
provenance, actor-policy versions/acks/history, logger and process lifecycle,
AMP and non-finite counters, ModelBuffer seal/drain ordering and evidence,
checkpoint completeness, recursive finiteness, and cleanup.  Reward,
behavior, support, stability, or acceptance metrics are forbidden inputs to a
wire decision or implementation edit.

The wire must exercise a nonterminal actor publication and the sole terminal
publication, complete every acknowledgement, acknowledge its last model write,
seal exactly once, take the correct zero-or-one terminal drain branch, force
save before success finish, and terminate ModelLearner and SelfPlay with exact
true results.  The complete public actor/ModelNet validators and actual
production-checkpoint smoke must pass.  W&B must remain disabled and its
durable logger-completion evidence must carry the unchanged required=false
semantics.  Public finish is written only after full validation.

Frozen unit, contract, and real-Ray integration tests must cover both drain
branches and at least: strict evidence types/keysets, stale-prefetch discard,
fresh post-seal processed-n equality, replay-cap bypass exactly once,
last-write/seal/complete-success ordering, claim-before-seal completion,
seal-before-claim denial, active-claim completion rejection, duplicate/stale/
wrong claim token, no post-terminal environment step, late write,
duplicate/malformed seal, processed-n regression or shortfall, pre-progress
greater than terminal, missing/None/stale drain data, m-only or p-only update,
AMP skip, non-finite state, second drain, force-save failure, success-finish
laundering, each bounded RPC timeout, worker failure, abort, Ray death/no-retry,
and schemas 1-6 compatibility.  Negative branches need not occur in the live
wire, but any actual skip, late write, abort, timeout, retry, malformed state,
or non-finite value permanently fails it.

## Frozen 100k and 300k decisions

The v13 Frozen 100k qualification section is incorporated verbatim.  Its
population remains `70000 < real_step <= 100000`, with exact windows
`(70000,80000]`, `(80000,90000]`, and `(90000,100000]`; overshoot is excluded.
Every v13 mechanism, trailing-five, support, calibration, saturation, and
safety threshold remains unchanged.  Schema 7 and the exact successful seal,
drain, ModelNet state, and completion evidence are additional integrity gates,
not behavioral thresholds.  Any qualification failure ends v14.

The v13 Frozen 300k primary acceptance section and its incorporated hashed v10
algebra are likewise verbatim.  Full remains `(100000,300000]`, late remains
`(250000,300000]`, and W1/W2/W3 remain `(270000,280000]`,
`(280000,290000]`, and `(290000,300000]`.  Every support floor and numeric
threshold remains unchanged, including soft learned-gate probability
0.475/0.525, sampled-control strength 0.525, conditional argmax 0.60,
useful-pair coverage 0.95, sign agreement 0.60, strict support fractions
above 0.05, wrong-side saturation and forced-stop rates below 0.01, heldout
RMSE at most 0.5 where the inherited training gate requires it, and zero AMP
or non-finite safety events.

Soft B/calibration probabilities still use training soft epsilon 0.02.
Sampled gate behavior, stored likelihood, V-trace, and inherited entropy
weighting still use execution epsilon 0.25.  Change D occurs only after the
terminal actor policy has been loaded and acknowledged and therefore cannot
alter any canonical trajectory row or acceptance population.

All inherited artifact, mechanism, behavior, stability, support,
trailing-five, saturation, forced-stop, calibration, AMP, and non-finite gates
must pass together with every schema-7 seal/drain gate.  There is no partial,
diagnostic, support-only, or mechanism-only pass.

## Fixed-checkpoint confirmation and final claim

Only an accepted seed-5 primary may be evaluated once with closed profile
`v14-300k`.  Held-out seeds remain 20260827 through 20260842, exactly 16
streams by 6250 real steps, 100000 total, with calibration V-trace unroll 201.
Training remains disabled; soft and execution runtime epsilon are both set to
zero only after immutable training validation, and the evaluator-private
`train_model=false` copy makes ModelBuffer seal coordination ineffective while
retaining persisted seal schema 1.  Primary and conditional PROCEED/RESET
actions retain the inherited sampling rules, while the nonzero binary gate is
deterministic and only an exact-zero scalar is sampled at 0.5.

Before any rollout the evaluator must validate and record the complete v13
identity inherited by v14, schema 7, seal schema 1, the exact 229-key
config/actor/model surfaces, the terminal actor bundle and publication
history, full ModelNet state, all ten seal/drain evidence fields and branch
relations, successful public finish/logger completion, absent private
markers, source/runtime binding, and the primary stage tuple.  It must reject
schema 6, seal schema 0/defaulted/missing, an ineligible stage, or any failed
or incomplete terminal artifact.  The validated immutable training mappings
and checkpoints are never rewritten.  Fixed behavioral and calibration
definitions remain exactly those incorporated by v13; Change D adds no
held-out metric or threshold.

The Enduro v14 claim requires immutable artifact/integrity success, the one
seed-1 100k qualification pass, the one seed-5 300k training-telemetry pass,
and its eligible `v14-300k` fixed-checkpoint pass.  Failure at any stage is
permanent for this preregistered experiment.
