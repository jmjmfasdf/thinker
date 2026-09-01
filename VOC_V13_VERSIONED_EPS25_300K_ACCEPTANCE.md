# Enduro VoC-v13 versioned epsilon-0.25 gate preregistered acceptance

This protocol is frozen after the single VoC-v12 seed-4 300k result failed
and before any v13 implementation-dependent wiring, qualification, primary
training, or fixed-checkpoint evaluation.  V13 is a separately named,
result-informed experiment.  It makes exactly three prospective changes to
the v12 production mechanism:

1. the executed scalar-gate epsilon is separated from the soft learned-gate
   epsilon and fixed at 0.25;
2. every actor unroll and every published ActorNet snapshot is governed by a
   strict, acknowledged policy-version barrier; and
3. the main actor GradScaler starts at scale 32.

Nothing else changes.  In particular, v13 does not change the Q target,
online-Q loss, EMA target, exact projection, gate features, reward,
environment, main-action policy, behavioral populations, support floors,
windows, or numeric acceptance thresholds.

## Permanent v12 failure and immutable evidence

V12 remains a permanent failure.  Its canonical full population contains 515
rows from real_step 100288 through 299472.  The terminal overshoot row at
real_step 300880 is excluded.

The first irreversible malformed trailing-five endpoint is real_step 142432.
Its exact five rows are 141824, 141968, 142128, 142272, and 142432, pooling
11378 positive-sign events and zero negative-sign events.  The next endpoint
at 142592 is also zero-denominator, and the final canonical population has 19
such endpoints.  These are failures under the frozen denominator and
malformed-input rules; later observations cannot repair them.

V12 also has the following independent frozen failures:

- main-actor AMP skipped three consecutive updates:
  - at real_step 207408, scale 256 changed to 128, cumulative/consecutive
    skips were 1/1, and 33 gradient parameters were non-finite;
  - at real_step 207568, scale 128 changed to 64, skips were 2/2, and 15
    gradient parameters were non-finite;
  - at real_step 207712, scale 64 changed to 32, skips were 3/3, and 4
    gradient parameters were non-finite.
  The main actor recovered with final cumulative/consecutive counts 3/0, but
  the frozen safety requirement is exactly zero.  Online-Q and dedicated-gate
  AMP skip counts remained zero.
- B3 sampled STOP was 5725/29013 = 0.197325 in full and
  2101/10364 = 0.202721 in late, despite soft negative-sign mean continue
  probabilities 0.403130 and 0.413421.
- Forced-stop rates were 0.105871 full and 0.163567 late, both above the
  unchanged strict 0.01 ceiling.
- Stability W1 failed B1 and B3, W2 failed B3, and W3 had zero defining B3
  support and zero B4 next-positive support.  The three all-direction and
  all-strength results were [false, false, false].

The numeric mechanism criteria and B1, B2, and B4 in both full and late
passing cannot rescue any of these failures.  The v12 fixed-checkpoint
confirmation was never authorized, and no later diagnostic may reclassify,
pool with, extend, retry, or rescue v12.

The immutable v12 decision is bound by these SHA-256 identities:

- protocol VOC_V12_EPSGREEDY_GATE_300K_ACCEPTANCE.md:
  c20657a08a46f69121289d9b220d85a7fffad4ada400654ca5829d95beb7368a;
- snapshot source manifest:
  856f499f90797d1b38c4de6339c8132bd5c97a4400b02d61db4ac2f50f2c7db8;
- staged-data manifest:
  23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343;
- primary xpid enduro-voc-v12-epsgreedy-seed4-strict-fresh-300k, with:
  - finish:
    b5bd1670271e02742d44ff59f7b75b3bb4e9e91322937129d3e5da4ea38f3b0b;
  - actor checkpoint:
    254ce8d1cc5fd91c27263dbac8e6ae9e45af807159a403fd02d464a03b59273b;
  - ModelNet checkpoint:
    43156ff56f280dc4df54f9caa0afc0e04f298da1d1965ab35219b0e27c80eb91;
  - resolved configuration:
    0fae0a3b1e932606d711fcd3b2e538d292f3b77afc737ccbf415ba2871cb7657;
  - actor telemetry:
    c049f5ba7746085b3ba3ba2e386e4f610605825b8d092dc2a6435b89e1955bc2;
  - model telemetry:
    a90be6618f3d0deabc1f326424b4b21e05eac913f8521d736056784cdc739dea.

The earlier normative documents also remain preserved.  In particular,
VOC_V9_PARAM_ALIGN_ACCEPTANCE.md has SHA-256
f832ae998332c5ff0ce03cb9334447a0e456646f80ccfdb044271cfb19681972,
VOC_V10_POWERED_300K_ACCEPTANCE.md has SHA-256
9cf52993ce4dcf1044b867028db9ae4a7d91b4e69516f1823e6abefd3059b6e4,
and VOC_V11_EXACT_PROJECTION_300K_ACCEPTANCE.md has SHA-256
7b4b1bca80a81d31c8aea9ef0fb7fed6b5936e654008d972de4e4539590b4044.
Whenever v13 incorporates a frozen v9-v12 criterion, the exact hashed text
controls.  This summary cannot weaken, strengthen, round, or reinterpret it.

## Change A: separate execution epsilon

V13 introduces voc_gate_execution_epsilon and fixes it to the finite,
non-boolean numeric value exactly equal to 0.25.  It is distinct from the
unchanged voc_train_epsilon exactly equal to 0.02:

- voc_train_epsilon continues to define the frozen soft learned-gate surface
  used for B1-B4 probability, gap, margin, and calibration definitions;
- voc_gate_execution_epsilon defines only the executed binary gate
  distribution used for sampling, stored behavior probability, action
  likelihood, V-trace, and the existing primary/bout actor-entropy
  expectation and weighting.

Let g be the raw scalar log-odds emitted by the exact-projected learned gate.
When CONTINUE and STOP are both legal, v13 training execution is:

- g > 0: P(CONTINUE) = 0.875 and P(STOP) = 0.125;
- g < 0: P(CONTINUE) = 0.125 and P(STOP) = 0.875;
- bit-exact g == 0: P(CONTINUE) = P(STOP) = 0.5.

If only one binary gate action is legal, it has probability one.  The binary
control is sampled from this distribution.  This is an epsilon-greedy scalar
policy, not a Q-sign, Q-value, depth, token, telemetry, or acceptance-slice
hard cut.  Conditional on CONTINUE, PROCEED/RESET is unchanged.  The main
environment action policy is unchanged.

The actual execution probability q also supplies the existing joint-policy
entropy expectation weights.  This adds no entropy loss or coefficient.  The
q gate probability is detached at that weighting, so this path creates no
dedicated gate-head gradient.  Conditional PROCEED/RESET and main-policy
logits and distributions remain unchanged; only the visitation/expectation
weight of their existing entropy gradients follows the same 0.875/0.125 q
used by execution.

The separate soft learned-gate surface remains authoritative for frozen
soft-p criteria.  Schema-6 ActorOut must continue to expose its soft logits
and probability separately from the execution logits and probability.
Substituting execution probability for soft acceptance probability, or soft
probability for behavior likelihood, fails the protocol.

Schemas 1 through 5 retain their historical behavior.  They cannot silently
acquire execution epsilon 0.25.  Where the new field is absent in a legacy
schema, compatibility resolution preserves the prior coupling to
voc_train_epsilon; schema 6 requires an explicit 0.25 value.  Configuration,
actor and ModelNet metadata, resume, smoke, public evaluation, fixed
evaluation, and every resolved identity record must distinguish and bind both
epsilon fields exactly.  Booleans, non-finite values, missing schema-6
values, and adjacent representable mismatches fail closed.

## Change B: schema-6 actor-policy version barrier

V13 uses gate-policy schema 6 and requires
voc_actor_policy_version_barrier=true.  It also requires the actor-policy
bundle identity voc_actor_policy_bundle_schema_version=1 in configuration,
runtime state, actor and ModelNet metadata, checkpoints,
public/smoke/fixed validation, and resolved identity records.  Schemas 1
through 5 retain barrier=false legacy behavior and cannot silently become
v13.

The barrier timeout is the explicit identity
voc_actor_policy_barrier_timeout_s=120.0.  It must be a finite, non-boolean
numeric value exactly equal to 120.0 in configuration, actor and ModelNet
metadata, runtime state, checkpoints, public/smoke/fixed validation, and
resolved identity records.  Tests may inject a fake monotonic clock but may
not change the qualifying timeout identity.

The schema-6 barrier is valid only with the exact resolved topology
parallel_actor=true, ppo_k=1, self_play_n=1, env_n=16, and
actor_batch_size = self_play_n * env_n = 16 after auto_res resolution.
Every actor-training batch therefore contains the complete set of 16
expected actor ids, each exactly once.  Every critical Ray actor and buffer
must use max_restarts=0, and every critical task must use max_retries=0.
The topology values are hard configuration/runtime/checkpoint identities,
not launch defaults.  Critical Ray actors bind the exact identities
voc_actor_policy_ray_max_restarts=0 and
voc_actor_policy_ray_max_task_retries=0 across configuration, actor and
ModelNet metadata, public/smoke/fixed validation, and resolved records.
Every such actor is constructed with those values.  Retry count zero for any
non-actor Ray task is a source-hardcoded protocol constant, with no
configuration override, and is attested from immutable implementation-source
hashes, static contract tests, and live startup Ray task specifications.

GeneralBuffer owns one actor_policy_bundle with exactly these fields:

- bundle_schema_version: the non-boolean integer 1;
- policy_version: a non-boolean integer at least zero;
- terminal: a strict boolean;
- gate_schema: the non-boolean integer 6; and
- actor_state_dict: a complete, recursively finite, immutable CPU snapshot.

Publication is an atomic replacement of the whole bundle.  No consumer may
observe a new version paired with old or partial weights, a mutable alias, a
CUDA-resident tensor, a missing state entry, or a non-finite value.
The fresh initial snapshot theta_0 has policy_version=0 and publication
count zero.

For an unroll produced from policy theta_v:

- temporal row zero carries the sentinel version -1;
- every temporal row from index one onward carries exactly v;
- an actor-training batch contains every expected actor id exactly once and
  has one homogeneous consumed version v; and
- future, stale, mixed, missing, malformed, boolean, negative other than the
  row-zero sentinel, or out-of-range versions fail closed.

The learner consumes theta_v, completes its main-actor update and the
independent online-Q update followed by EMA and exact gate projection, then
validates the complete resulting state before publishing theta_(v+1).  All
expected acknowledgements for the atomic publication must complete before
another version can be published.

Each post-batch publication, terminal or nonterminal, increments both
policy_version and the publication count by exactly one.  Append-only
publication and
acknowledgement telemetry must therefore reconstruct the exact contiguous
sequence 0,1,...,N with no duplicate or omitted version.  In every saved
actor checkpoint, current policy_version must equal publication_count; the
version-mismatch, malformed-bundle, stale/future/mixed-version, and barrier-
timeout counts must all equal zero.  Equality of the two terminal counters is
not by itself evidence of contiguity: training telemetry must also contain
the complete one-step predecessor-to-successor relation for every
publication.

That durable evidence is the append-only top-level
voc_actor_policy_publication_history.  Each event contains exactly these
seven keys: predecessor_version, policy_version, publication_count,
terminal, ack_ranks, expected_ack_count, and state_sha256.  The initial v0
event has predecessor_version=-1, policy_version=0,
publication_count=0, terminal=false, the complete sorted acknowledgement
rank list, and the theta_0 state digest.  Each later event increments both
versions/counts contiguously, contains every and only expected acknowledgement
rank, and binds the digest of that publication.  An event is appended only
after its full acknowledgement set has been validated.  Exactly the final
event is terminal=true; its version, count, acknowledgement set, and digest
must equal the terminal checkpoint evidence described below.

Each worker must first enqueue and receive acknowledgement for its completed
v unroll, then load exactly theta_(v+1) before beginning its next unroll.  It
may not generate ahead, reuse v, skip to a future version, or mix versions.
A stale wait, future bundle, mixed batch, malformed bundle, missing actor id,
duplicate actor id, failed acknowledgement, or barrier invariant violation
is a hard error.

The worker heartbeat is a latest-value liveness channel, not append-only
publication evidence.  It contains exactly rank, policy_version, phase, and
count, and only Python-integer rank zero is valid.  A load_ack heartbeat for
version v has count=2*v+1 and an enqueue heartbeat for version v has
count=2*v+2.  Because GeneralBuffer may overwrite an intermediate latest
value before the driver polls it, successive observed canonical counts must
increase strictly but may jump by more than one.  An exact duplicate is not
progress and must not reset the monotonic deadline; a regression, a changed
payload at the same count, or a malformed phase/version/count relation fails
closed.  Only voc_actor_policy_publication_history, not heartbeat polling,
must reconstruct every contiguous publication.

Every barrier wait uses a monotonic 120-second timeout.  Timeout or any
barrier error must fail fast, abort the run, and must not be hidden by a Ray
task retry.  The protocol is fresh-only; no resume, preload, or parent
checkpoint may import a version lineage.

The one publication produced by the batch that reaches total_steps is the
terminal publication.  It increments v to v+1 and the publication count
exactly once, sets terminal=true on that same atomic bundle, and prevents any
next unroll.  Publishing a nonterminal v+1 followed by a second, duplicate
terminal v+1 is forbidden, as is incrementing again merely to terminalize.
Every expected worker must acknowledge this terminal bundle, and
GeneralBuffer must attest the terminal acknowledgement set, before FINISH
may be written.  A finish marker without the exact terminal bundle and
acknowledgements is invalid.  The terminal actor checkpoint and public
validation record must expose terminal=true, the last contiguous
policy_version and publication_count, zero mismatch/malformed/timeout
counts, and terminal_ack_count equal to expected_ack_count.

The terminal actor checkpoint persists this evidence under these exact
top-level names:

- voc_actor_policy_bundle;
- voc_actor_policy_version;
- voc_actor_policy_publication_count;
- voc_actor_policy_terminal;
- voc_actor_policy_version_mismatch_count;
- voc_actor_policy_malformed_bundle_count;
- voc_actor_policy_barrier_timeout_count;
- voc_actor_policy_terminal_ack_count;
- voc_actor_policy_expected_ack_count;
- voc_actor_policy_state_sha256;
- voc_actor_policy_publication_history; and
- voc_actor_policy_publication_history_sha256.

It also persists the exact identity fields
voc_actor_policy_bundle_schema_version,
voc_actor_policy_version_barrier, voc_actor_policy_barrier_timeout_s,
voc_actor_policy_ray_max_restarts,
voc_actor_policy_ray_max_task_retries, voc_gate_execution_epsilon, and
actor_amp_init_scale.  The public actor validator and smoke/fixed resolved
records must expose all of them.

voc_actor_policy_state_sha256 is a canonical sorted-tensor SHA-256 digest of
the actor state: for each lexicographically sorted state key it binds the
key, dtype, shape, and contiguous CPU tensor bytes.  It is not a pickle-file
digest.  The persisted digest must independently recompute identically from
both the terminal top-level voc_actor_policy_bundle actor_state_dict and the
checkpoint actor_net_state_dict.  Missing, extra, aliased, mutable, non-CPU,
or non-finite tensor state fails before FINISH.

The persisted top-level voc_actor_policy_bundle is the exact terminal bundle,
not a summary.  It contains exactly the five previously defined fields:
bundle_schema_version=1, policy_version equal to the top-level
voc_actor_policy_version, terminal=true, gate_schema=6, and a complete
immutable CPU actor_state_dict.  Its actor_state_dict must equal the
checkpoint actor_net_state_dict key-for-key, dtype-for-dtype,
shape-for-shape, and tensor-byte-for-tensor-byte.  The canonical digest must
recompute from each mapping independently after process exit.  A digest
without this persisted bundle is insufficient evidence and fails closed.

voc_actor_policy_publication_history_sha256 is SHA-256 over the UTF-8 bytes
of the history's canonical compact JSON: sort_keys=true, ensure_ascii=true,
separators=(',', ':'), and allow_nan=false, with no insignificant whitespace.
The persisted history must pass exact seven-key/type validation, reconstruct
the complete sequence 0,1,...,N and all acknowledgement sets, contain only
one terminal event at N, and independently recompute to this digest.  Its
final state_sha256 must equal the digest recomputed from both the terminal
bundle actor_state_dict and checkpoint actor_net_state_dict.

Policy-version fields and acknowledgement counts are integrity telemetry.
They must be exact, complete, finite where numeric, and reconstructible, but
they create no new behavioral threshold or acceptance population.

### Schema-6 W&B two-phase completion

The W&B-enabled qualification and primary runs may commit the public finish
marker only through a strict two-phase logger close.  After every self-play
worker has returned exact success and the terminal actor checkpoint,
publication history, state digests, and ModelNet checkpoint have passed their
final validation, the driver atomically writes the private ckpdir file
voc_actor_policy_logger_finish_request.  It contains exactly:

- schema_version=1;
- status='finish_requested';
- the terminal policy_version;
- state_sha256 equal to the terminal actor-policy state digest;
- publication_history_sha256 equal to the validated complete history digest;
  and
- checkpoint_files containing exactly config_c.yaml, ckp_actor.tar, and
  ckp_model.tar, each mapped to exactly sha256 and size, where sha256 is a
  lowercase 64-hex string and size is a strict non-boolean positive integer.

The request uses the same canonical compact JSON contract as the publication
history: sort_keys=true, ensure_ascii=true, separators=(',', ':'), and
allow_nan=false.  Its request SHA-256 is over those canonical UTF-8 payload
bytes only; the single trailing LF used as the on-disk JSON record terminator
is not part of the digest.  The write must use an exclusive temporary regular
file, flush and fsync it, and atomically create the previously absent target
with create-if-absent hard-link publication.  It fsyncs the ckpdir after
publication and again after removing the temporary link; it must never replace
or overwrite an existing target.  A pre-existing, linked, malformed, partial,
or mismatched private record fails closed.

The schema-6 LogWorker is launched with max_restarts=0 and
max_task_retries=0.  It must remain live until the private request appears,
perform one final strict statistics/artifact upload iteration, and complete
the W&B close.  No exception from logging, upload, visualization, or close may
be swallowed on this path.  Only then may it atomically write the private
ckpdir file voc_actor_policy_logger_finish_ack containing exactly
schema_version=1, status='finish_acknowledged', and request_sha256 equal to
the canonical request digest, and return the exact Python value `True`.

The driver waits with the same monotonic 120-second bound and requires both
the exact true task result and the exact matching acknowledgement; neither is
sufficient alone.  It then deletes both private request/ack files and only
after confirming their absence may atomically commit the public finish
marker.  That public finish contains the durable top-level
voc_actor_policy_logger_completion mapping with exactly these ten keys:
schema_version, required, use_wandb, request_sha256, ack_verified,
private_markers_cleaned, policy_version, state_sha256,
publication_history_sha256, and checkpoint_files.  Its schema_version is 1;
policy_version and both state digests equal the terminal actor/history
evidence; checkpoint_files exactly equals both the private request mapping and
the public finish checkpoint mapping.  For W&B-enabled schema-6 runs,
required=true, use_wandb=true, request_sha256 is the validated request's
canonical digest, ack_verified=true, and private_markers_cleaned=true.

A logger early exit, exception, hang, retry, false/non-boolean
result, missing or mismatched acknowledgement, timeout, failed cleanup, or
public finish visible before this sequence is a hard artifact failure.  The
failure path must never write public finish or a
voc_actor_policy_logger_completion record.  After the driver has confirmed,
within the bounded termination protocol, that the logger task is terminal and
cannot write a late acknowledgement, it may delete only the exact request
atomically created and owned by this finalize attempt, identified by its
`st_dev`, `st_ino`, and `st_ctime_ns`, and an acknowledgement still bound to
that request digest.  It revalidates those identities and payloads immediately
before unlinking and then fsyncs the ckpdir.  If logger death cannot be
confirmed within the bound, or if request/ack ownership, generation, payload,
or binding cannot be proved, deletion is forbidden.  Every existing private
request/ack file is then retained as quarantined forensic evidence, the run
hard-fails, and no public finish or completion attestation may ever be
committed.  A pre-existing/foreign marker collision and an ownership-replaced
marker are therefore quarantine cases just like unconfirmed logger death;
none can be treated as a completed bundle or release evidence.

The W&B-disabled 1.2k wire has no LogWorker handshake, but both private files
must remain absent and public finish still follows successful terminal
checkpoint and worker validation.  Its durable completion record instead has
required=false, use_wandb=false, request_sha256=null, ack_verified=false, and
private_markers_cleaned=true, while binding the same terminal version,
state/history digests, and three checkpoint files.  Public evaluation, smoke,
and fixed validation of every completed schema-6 bundle require the public
finish,
fail closed if either private marker remains, and record strict logger
two-phase completion/cleanup and the exact durable ten-key mapping in their
resolved evidence.  For the W&B-enabled qualification and primary, the W&B
run must additionally be complete with exit code zero.

## Change C: main actor AMP initial scale 32

V13 introduces actor_amp_init_scale and requires the finite, non-boolean
numeric value exactly equal to 32.  It changes only the initial scale of the
main actor GradScaler from v12's 256 to 32.

This is not FP32 actor execution.  It does not change the online-Q, dedicated
gate, or ModelNet scaler, precision, optimizer, scheduler, or counter
semantics.  Main-actor and online-Q updates remain independent: a main-actor
AMP skip does not suppress an otherwise successful online-Q update, EMA
update, exact projection, or their counters.  Conversely, a Q skip preserves
the pre-existing Q/EMA/projected-gate atomic no-update semantics.

The unchanged precision identities are strict booleans: main actor
float16=true and ModelNet model_float16=false.  Configuration, actor metadata,
and ModelNet metadata must all bind those exact values; a consistent rewrite
of all three surfaces is still invalid.

The lower initial scale is not permission to tolerate a skip.  Main actor,
online-Q, gate, and model AMP skip counts and every consecutive-skip and
non-finite counter must remain exactly zero in every qualifying run.  Any
skip is a permanent failure of that stage.

Schema-6 configuration, actor and ModelNet metadata, checkpoints,
resume/smoke/public/fixed validators, and resolved identities must bind
actor_amp_init_scale=32 exactly.  The terminal actor checkpoint and public
record must also preserve initial-scale provenance 32 and show cumulative
and consecutive main-actor AMP skip counts exactly zero.  Legacy schemas
keep their historical constructor default and validation.

## Unchanged v12 mechanism

Except for changes A-C, preserve the complete v12 production configuration:

- dynamic_voc_mode=control and fresh control from step zero;
- voc_gate_policy_schema_version=6 for v13;
- voc_gate_epsilon_greedy_execution=true;
- voc_gate_execution_epsilon=0.25;
- voc_actor_policy_version_barrier=true,
  voc_actor_policy_bundle_schema_version=1, and
  voc_actor_policy_barrier_timeout_s=120.0;
- voc_actor_policy_ray_max_restarts=0 and
  voc_actor_policy_ray_max_task_retries=0, with non-actor task retry
  source-hardcoded at zero;
- parallel_actor=true, ppo_k=1, self_play_n=1, env_n=16, and
  actor_batch_size=16 after auto_res resolution, with Ray restart/retry
  counts fixed at zero;
- actor_amp_init_scale=32;
- float16=true and model_float16=false;
- voc_gate_exact_projection=true;
- voc_gate_param_align=false and voc_gate_param_align_coef=1.0;
- voc_gate_confidence_weighted=false and voc_gate_adam_beta1=0;
- voc_gate_learning_rate=0.001, voc_gate_target_tau=0.1,
  voc_gate_q_temperature=0.05, and voc_gate_temperature=1.0;
- schedule_total_steps=100000000;
- voc_train_epsilon=0.02 and voc_eval_stochastic=true; and
- all other v12 production values unchanged.

“All other v12 production values unchanged” is mechanically bound to the
frozen v12 primary configuration above.  Start from its exact 219-key mapping
and remove only these six stage-varying keys: base_seed, total_steps,
model_warm_up_n, actor_unroll_len, use_wandb, and xpid; and these four
path/command-derived keys: savedir, ckpdir, cmd, and icopro_data_path.  The
remaining exact 209-name projection is serialized with
json.dumps(sort_keys=true, ensure_ascii=true, separators=(',', ':'),
allow_nan=false), UTF-8 encoded with no trailing newline, and has SHA-256
bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407.
The configuration, actor metadata, and ModelNet metadata of every v13 stage
must each contain that exact 209-name projection and recompute that digest.
The exclusion does not make the ten removed values free: the six stage values
are fixed below, while the four path/command values must resolve to the
immutable snapshot, staged data, run directory, and captured launch command.
The nine new schema-6 keys outside this legacy projection are exactly
actor_amp_init_scale, voc_gate_execution_epsilon,
voc_actor_policy_version_barrier,
voc_actor_policy_bundle_schema_version,
voc_actor_policy_barrier_timeout_s,
voc_actor_policy_ray_max_restarts,
voc_actor_policy_ray_max_task_retries,
voc_gate_policy_schema_version, and voc_actor_policy_barrier_runtime.  Their
values are separately mandatory as specified above, including exact
voc_gate_policy_schema_version=6 and
voc_actor_policy_barrier_runtime=true in every immutable training surface.
Thus each complete configuration/actor/ModelNet embedded-flags mapping has
exactly 228 keys: baseline 209, stage 6, path/command 4, and new schema 9.
Any missing or additional key fails closed.  A held-out evaluator may set its
private runtime copy of the barrier-runtime flag false only after validating
and recording the immutable training value true; it must not rewrite any of
the three persisted mappings.

The v12 online-Q optimizer, FP32 EMA Polyak update, successful-Q then EMA then
FP32 exact W/b projection ordering, skip semantics, counters, empty gate
optimizer state, and pristine gate scheduler/scaler remain unchanged.  Stored
gate W/b must be torch.equal to the raw scaled EMA affine target.  Rebuilt
dueling-Q state probabilities remain finite tolerance diagnostics, not
bit-equality gates.

Logged actual behavior-probability and policy-version statistics are
instrumentation only.  The logged statistics may not independently add a
loss, gradient, selection role, behavioral barrier, or post-hoc threshold.
No result may tune either epsilon, the timeout, the barrier, the scaler
initialization, or a Q/depth override.

## Immutable identities and no-retry rule

Every v13 run is fresh: ckp=false, preload and preload_actor are empty,
voc_parent_checkpoint is empty, parent-update count is zero, version lineage
starts at its frozen initial state, and online-Q, EMA-Q, and scalar-gate heads
start exactly zero.  Wire, qualification, and primary runs are separate
processes and output directories; no checkpoint, version state, or
observation passes between them.

- The one 1.2k integrity wire uses base seed exactly 1, xpid exactly
  enduro-voc-v13-versioned-eps25-sentinel-wire1200, and is nonqualifying.
- The one fresh 100k qualification uses base seed exactly 1, xpid exactly
  enduro-voc-v13-versioned-eps25-seed1-qual-fresh-100k, and is not a
  continuation of the wire.
- The one primary 300k run uses base seed exactly 5, xpid exactly
  enduro-voc-v13-versioned-eps25-seed5-strict-fresh-300k, and the next unused
  seed chosen prospectively for v13.  There is exactly one primary attempt.

The only valid stage tuples
(xpid, base_seed, total_steps, model_warm_up_n, actor_unroll_len, use_wandb)
are exactly:

- (enduro-voc-v13-versioned-eps25-sentinel-wire1200, 1, 1200, 512, 41,
  false);
- (enduro-voc-v13-versioned-eps25-seed1-qual-fresh-100k, 1, 100000,
  10000, 201, true); and
- (enduro-voc-v13-versioned-eps25-seed5-strict-fresh-300k, 5, 300000,
  10000, 201, true).

Each xpid is an exact Python string identity; coercion, trimming, case change,
or whitespace normalization is forbidden.  The numeric members are strict
non-boolean integers and use_wandb is a strict Python boolean.  Configuration,
actor metadata, and ModelNet metadata must agree on the complete tuple, and
the normalized ckpdir path basename must equal the exact xpid.  The closed
fixed profile v13-300k accepts only the primary tuple; a wire or qualification
bundle is rejected from its stage identity before any held-out rollout.

Seed 1 is not a backup for seed 5.  No seed may replace, pool with,
majority-vote, or rescue a stage.  Crash, corruption, timeout, version fault,
provenance failure, qualification failure, primary failure, or fixed-
evaluator failure is the failure of that stage.  Same-run extension, resume,
Ray retry, retry-selection, replacement seed, threshold change, and
result-conditioned mechanism change are forbidden.

Create v13 from the immutable v12 snapshot as an independently copied,
separately named snapshot plus an exact enumerated overlay containing only
changes A-C, schema/public/fixed validation, corresponding tests, and this
document.  Preserve the v12 snapshot, run artifacts, normative documents,
unrelated behavior-bearing files, Cython extensions, and staged Enduro data
byte-for-byte.  Freeze source and data manifests outside src, seal modes,
exclude historical run outputs, prohibit links/special/writable source
nodes, and prove inode independence before launch.

## Sequential release gates

The stages are strictly ordered:

1. Freeze and independently audit the v13 snapshot, exact v12-to-v13 overlay,
   manifests, runtime binding, binaries, tests, schema-6 validators, bundle
   barrier, and this document.
2. Run one seed-1 1.2k integrity wire with total_steps=1200,
   model_warm_up_n=512, actor_unroll_len=41, W&B disabled, CUDA devices 0,1,
   Ray requesting two GPUs, and otherwise the production mechanism with
   schedule_total_steps=100000000.
3. Only after the wire passes, run one separate fresh seed-1 100k
   qualification with total_steps=100000, production warm-up 10000, actor
   unroll 201, W&B enabled, and the frozen qualification below.
4. A failed 100k qualification permanently ends v13 and forbids the primary.
   A pass authorizes exactly one separate fresh seed-5 primary with
   total_steps=300000, production warm-up 10000, actor unroll 201, W&B
   enabled, and otherwise identical production configuration.
5. Only if seed-5 300k training telemetry and artifact gates pass may its
   single terminal checkpoint be evaluated with closed profile v13-300k.

Pong and Space Invaders remain out of scope until every Enduro v13 stage
passes.

## Integrity-wire acceptance

The wire may be inspected only for launch/exit, immutable source/data/runtime
binding, schema/config/provenance, the two epsilon surfaces, exact projection,
the version barrier and terminal acknowledgements, AMP32 initialization,
counters, checkpoint completeness, CSV header/width, recursive finiteness,
W&B-disabled identity, and process/GPU/Ray cleanup.  Reward, support,
conditional behavior, stability, and acceptance values are forbidden inputs
to a wire decision or edit.

The wire must exercise and validate at least one complete nonterminal
v-to-v+1 publication/acknowledgement cycle and the terminal acknowledgement
cycle.  Every unroll and batch version stamp must satisfy the exact row and
actor-id contract.  It must demonstrate main actor GradScaler initialization
at 32 and zero actor/Q/gate/model AMP skips.  A supported nonzero gate must
show the separate soft probability and the correct 0.875/0.125 execution
probability when both gate actions are legal.  Stored sampled-action
probability and V-trace likelihood must use execution probability.

Frozen unit/contract tests must exercise positive, negative, and bit-zero
gate logits, one-action legal masks, malformed/mixed/stale/future versions,
missing/duplicate actor ids, publication immutability, acknowledgement and
timeout failures, terminal ordering, Q-skip/no-support, and main-actor-skip
branches.  The wire does not require a skip or every negative branch to occur.
If any occurs, semantics must match the contract; any actual skip, timeout,
version error, retry, or non-finite value fails the wire and blocks
qualification.

## Frozen 100k qualification

The v12 section Frozen 100k qualification is incorporated verbatim except for
the schema-6 identity, execution-epsilon, version-barrier, and AMP32 integrity
additions above.  The canonical population remains
70000 < real_step <= 100000, with windows (70000,80000], (80000,90000],
and (90000,100000].  Overshoot above 100000 is excluded.  Population,
pooling, support, malformed-input, Q-sign tie, trailing-five, calibration,
and safety algebra remain unchanged.

Every numeric qualification gate remains unchanged:

- teacher gap at least 0.075, student gap at least 0.05, retention at least
  0.50, and signed margin strictly positive;
- at least two windows with both student gap and signed margin strictly
  positive;
- maximum consecutive negative trailing-five pooled gaps at most 3, with
  exact zero non-negative and all required denominators positive;
- train and holdout CONTINUE and STOP fractions each strictly greater than
  0.05, with positive denominators;
- wrong-continue saturation strictly below 0.01; wrong-stop remains the
  report-only 100k safety diagnostic;
- online-versus-EMA non-tie sign agreement at least 0.60;
- held-out EMA selected-action TD RMSE at most 0.5; and
- actor, online-Q, gate, and model AMP skips and every non-finite counter
  exactly zero.

Schema 6, both exact epsilon identities, the bundle/version barrier,
successful-Q/EMA/projection count reconciliation, bit-exact W/b projection,
the exact topology and no-retry identity,
voc_actor_policy_barrier_timeout_s=120.0, actor_amp_init_scale=32, and
terminal acknowledgements are additional hard identity/integrity gates, not
new behavioral thresholds.  Failure ends v13 and forbids the 300k run.  A
pass cannot tune the mechanism or thresholds.

## Frozen 300k primary acceptance

The hashed v10 sections Canonical rows and windows, Frozen metric algebra,
Trailing-five definition, Mechanism acceptance, Four required learned
behaviours, and Stability behaviour semantics are incorporated verbatim.
V13 also preserves the v12 dual-surface interpretation.  No label,
population, denominator, support, or number changes:

- full is 100000 < real_step <= 300000;
- late is 250000 < real_step <= 300000;
- W1/W2/W3 are (270000,280000], (280000,290000], and
  (290000,300000];
- support floors remain deep-negative 256 full/64 late, PROCEED and RESET 256
  full/64 late each, and next-positive and next-negative 128 full/32 late;
- soft learned-gate probability thresholds remain 0.475/0.525;
- sampled-control strength remains 0.525, conditional argmax remains 0.60,
  useful-pair coverage remains 0.95, sign agreement remains 0.60, support
  fractions remain strictly above 0.05, both wrong-side saturation rates and
  forced-stop rate remain strictly below 0.01, and heldout RMSE remains at
  most 0.5;
- both full and late must pass mechanism acceptance and all four behaviors;
  all three stability windows must retain every strict direction, at least
  two must meet every 0.525 strength condition, and equality at 0.5 fails
  strict direction;
- the full-window maximum consecutive negative trailing-five run remains at
  most 3, every required denominator must be positive, and every AMP or
  non-finite safety event remains exactly zero.

B1-B4 and stability p_continue values use the soft learned-gate probability
defined by unchanged voc_train_epsilon=0.02.  Sampled successes are controls
drawn from execution epsilon 0.25.  Stored behavior probability, action
likelihood, and V-trace use the execution distribution.  Conditional argmax
uses that execution distribution; nonzero scalar sign agrees with the soft
gate argmax, while correctness against EMA-Q sign remains governed by the
unchanged 0.60 criterion and 1e-6 Q-sign tie rule.

All artifact/provenance, version-barrier, mechanism, behavior, stability,
trailing-five, calibration, support, saturation, forced-stop, AMP, and
non-finite gates must pass.  There is no partial, mechanism-only,
support-only, diagnostic, or historical pass.  The old
(70000,100000] primary-run diagnostic remains report-only.  Overshoot,
optional seeds, continuation, and post-hoc pooling cannot change the
decision.

## Fixed-checkpoint confirmation and final claim

Only after the seed-5 300k training decision passes may its exact terminal
checkpoint be evaluated once with closed profile v13-300k.  The evaluator
uses training disabled, execution epsilon zero, greedy=false, seeds
20260827 through 20260842, exactly 16 streams by 6250 real steps (100000
total), and calibration V-trace unroll 201.  The primary action and
conditional PROCEED/RESET remain sampled.  The binary gate is deterministic
by the sign of every nonzero scalar; only a bit-exact zero is sampled at 0.5.
Diagnostic or shortened evaluations are confirmation-ineligible.

The profile fails closed unless configuration, actor, and ModelNet all bind:

- base_seed=5, total_steps=300000, and schedule_total_steps=100000000;
- fresh control mode with empty preload and parent identity;
- gate-policy schema 6 and
  voc_actor_policy_bundle_schema_version=1;
- voc_actor_policy_version_barrier=true and a valid terminal acknowledged
  bundle;
- voc_actor_policy_barrier_timeout_s=120.0 and the exact 16-actor topology
  with voc_actor_policy_ray_max_restarts=0,
  voc_actor_policy_ray_max_task_retries=0, and finish/source/runtime evidence
  for source-hardcoded zero retry on non-actor tasks;
- voc_train_epsilon=0.02 and voc_gate_execution_epsilon=0.25 exactly;
- voc_eval_stochastic=true and
  voc_gate_epsilon_greedy_execution=true;
- exact projection true, parameter alignment false, and coefficient 1.0;
- actor_amp_init_scale=32, float16=true, and model_float16=false exactly; and
- the unchanged dedicated soft-Q gate and v12 production identity.

The evaluator first attests the immutable training identities
voc_train_epsilon=0.02 and voc_gate_execution_epsilon=0.25.  Only after
validating the unchanged configuration and actor/ModelNet checkpoint
identities plus the terminal bundle may it set both the soft runtime epsilon
and execution runtime epsilon to zero and disable barrier waiting in an
evaluator-only runtime copy.  The summary and manifest must separately
record training soft/execution epsilon 0.02/0.25 and held-out runtime
soft/execution epsilon 0/0.  It must not rewrite checkpoint flags,
configuration, or the evidence manifest.  It validates terminal FP32 gate
W/b against the raw scaled EMA affine target with bit equality, requires
terminal=true, a contiguous policy_version/publication_count relation, zero
version-mismatch, malformed-bundle, timeout, and main-actor AMP skip counts,
complete terminal acknowledgements, and canonical actor-state digest
equality, and preserves the schema-6 soft/execution surfaces.  The resolved
identity must be recorded in the summary, evaluation protocol, and manifest.
The evaluator must also validate and recompute the complete
voc_actor_policy_publication_history and
voc_actor_policy_publication_history_sha256: exact seven-key events must
reconstruct 0,1,...,N with complete acknowledgement ranks, exactly the final
event terminal, and its final state digest equal to the terminal bundle and
checkpoint actor-state digest.  Both the validated history and its digest
must be recorded in the resolved summary, evaluation protocol, and manifest.

The v13 profile is selected and validated from the actual checkpoint state
and the closed v13 specification, not from a normalized default.  Every
legacy v7, v10, v11, or v12 profile must reject schema 6, barrier=true,
bundle schema 1, actor_amp_init_scale=32 as a v13 identity, or execution
epsilon 0.25.  Conversely, the v13 profile must reject schemas 1 through 5 or
legacy-defaulted versions of any v13 field.

Fixed B/calibration probability uses the recorded soft learned-gate field.
Sampled actions, action probability, and V-trace use the epsilon-zero
execution logits.  Apply the same four frozen behavioral definitions and
supports, never count forced stops as sampled or argmax successes, and report
selected-action heldout calibration.  As in v10-v12, fixed-checkpoint
calibration is required reporting and finite audit, not a newly invented
post-hoc RMSE gate unless the hashed fixed-confirmation text explicitly makes
it one.

The Enduro v13 claim requires immutable artifact/integrity success, the one
seed-1 100k qualification pass, the one seed-5 300k training-telemetry pass,
and its eligible v13-300k fixed-checkpoint pass.  Failure at any stage is
permanent for this preregistered experiment.
