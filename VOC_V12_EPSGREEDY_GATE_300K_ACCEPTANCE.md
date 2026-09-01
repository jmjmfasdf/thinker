# Enduro VoC-v12 epsilon-greedy projected-gate preregistered acceptance

This protocol is frozen after the single VoC-v11 seed-3 300k result failed,
after the v12 implementation and contract audit, and before any v12 wiring,
qualification, primary training, or fixed-checkpoint evaluation.  V12 is a
separately named, result-informed experiment.  It changes exactly one
behavior-bearing mechanism relative to v11: execution of the already exact-
projected scalar gate is replaced by the epsilon-greedy sign policy specified
below.  It does not change the gate target, loss, update, features, Q values,
reward, environment, acceptance populations, or thresholds.

V11 remains a permanent failure.  In its canonical W3
`(290000,300000]`, the overshoot row at `real_step=300016` was excluded and
the last canonical row was `real_step=299584`.  B3 had 25 depth-8-plus
negative events and 12 sampled STOP successes: `12/25 = 0.48`.  This failed
both the strict sampled direction `>0.5` and the sampled-strength requirement
`>=0.525`, so the all-four-behavior stability gate failed.  The stored
behavior mean continue probability was `0.4567016`; the learner-current value
was `0.4523981`.  The latter already agrees within about `1.3e-8` with the
v11 epsilon-mixed soft probability
`0.98 * 0.451426576 + 0.01`; the stored-current difference of about
`0.0043035` is asynchronous rollout staleness and changes the expected count
by only about `0.1076` over 25 events.  The exact binomial lower-tail
probability for 12 or fewer STOP events under stored expected STOP
`0.5432984` is about `0.3306`.  Therefore this result is not reclassified as a
runtime, projection, or V-trace fault.  The other full/late mechanism,
B1/B2/B4, support, and safety gates passing cannot rescue W3, and v11's fixed
confirmation was never authorized.

The immutable v11 decision is bound by the following SHA-256 identities:

- protocol `VOC_V11_EXACT_PROJECTION_300K_ACCEPTANCE.md`:
  `7b4b1bca80a81d31c8aea9ef0fb7fed6b5936e654008d972de4e4539590b4044`;
- snapshot source manifest:
  `10afd3d8b55065ad53a2d55816e8577cdb884b0f8008cae5a3ef9d82c761c010`;
- staged-data manifest:
  `23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343`;
- primary xpid `enduro-voc-v11-exactproj-seed3-strict-fresh-300k`, with
  `finish` `9266e1ab473b1c47ac7fb3e2d3d60ecd0b3494b8d025994f06798530216095d7`,
  actor checkpoint
  `553fdc751c040aa751953b4a448708f900ed6686d237761b20f49d589ebf7fa5`,
  ModelNet checkpoint
  `69e6e503192e3416f1d1fde15858b6e45cc58e8a644dd0f2b43690fd03b39e3a`,
  resolved configuration
  `364eadca4c2eaf2c3ce97c3a3ff02dfc615229f8198c49778beb13e8a91ac2d8`,
  actor telemetry
  `7466d20eefaa11bc9bc4460421ae12659993c3d336dab958e17cd638f1f48201`,
  and model telemetry
  `b97bd879bbe3193a81df345b9b82cbceaeeff946edea1f9a411e8c01123c74d9`.

V12 must not reclassify, extend, pool with, or rescue v11.  The earlier
normative documents also remain preserved: `VOC_V9_PARAM_ALIGN_ACCEPTANCE.md`
SHA-256 `f832ae998332c5ff0ce03cb9334447a0e456646f80ccfdb044271cfb19681972`
and `VOC_V10_POWERED_300K_ACCEPTANCE.md` SHA-256
`9cf52993ce4dcf1044b867028db9ae4a7d91b4e69516f1823e6abefd3059b6e4`.
Whenever this document incorporates a frozen v9, v10, or v11 criterion, the
exact hashed text controls; this summary cannot weaken, strengthen, round, or
reinterpret it.

## The one mechanism change

- Keep the complete v11 mechanism and production configuration, changing only
  `voc_gate_epsilon_greedy_execution=false` to
  `voc_gate_epsilon_greedy_execution=true`.  Keep
  `voc_gate_exact_projection=true`, `voc_gate_param_align=false`, and
  `voc_gate_param_align_coef=1.0` exactly.
- Use gate-policy schema 5.  Schema 5 requires explicit
  `voc_gate_epsilon_greedy_execution=true`, exact projection true, parameter
  alignment false, and coefficient exactly 1.0 in configuration, actor and
  ModelNet metadata, resume/smoke/public-evaluation surfaces, and their
  resolved validation records.  Schemas 1--4 retain their historical
  `voc_gate_epsilon_greedy_execution=false` meaning and cannot silently become
  v12.
- Let `g` be the raw scalar log-odds emitted by the exact-projected learned
  gate.  When CONTINUE and STOP are both legal, training execution with the
  existing `voc_train_epsilon=0.02` is:
  - `g > 0`: `P(CONTINUE)=0.99`, `P(STOP)=0.01`;
  - `g < 0`: `P(CONTINUE)=0.01`, `P(STOP)=0.99`;
  - bit-exact `g == 0`: `P(CONTINUE)=P(STOP)=0.5`.
  If only one binary gate action is legal, it has probability one.  The legal
  mask, forced-stop handling, and invalid-row handling are unchanged.
- The binary control is sampled from that execution distribution.  This is an
  epsilon-greedy policy, not a Q/depth hard cut.  Neither Q, depth, control
  token, Q sign, acceptance slice, nor stored telemetry may override the
  learned scalar action.  Conditional on CONTINUE, the existing PROCEED/RESET
  distribution and sampling remain unchanged.  The main environment action
  policy is unchanged.
- Preserve a distinct soft learned-gate surface.  In training it is the v11
  temperature/epsilon-mixed probability and logits; under epsilon-zero fixed
  evaluation it is the temperature-scaled soft probability.  Schema-5
  `ActorOut.misc` exposes `voc_gate_soft_control_logits` and
  `voc_gate_soft_continue_probability`; actual execution remains in
  `search_control_logits` and is additionally exposed as
  `voc_gate_execution_continue_probability` for validation.
- The frozen B1--B4 probability, gap, margin, and calibration definitions use
  the soft learned-gate surface exactly as their v10/v11 predecessors did.
  Sampled-control statistics, action likelihoods, V-trace, and stored behavior
  telemetry use the actual execution distribution.  Substituting execution
  probabilities for the frozen soft `p_continue` criteria, or substituting
  soft probabilities for behavior likelihoods, fails the protocol.
- The v11 online-Q optimizer, FP32 EMA Polyak update, post-EMA exact W/b
  projection, update ordering, skip semantics, counters, and empty gate-
  optimizer/scheduler/scaler state are unchanged.  Stored gate W/b must remain
  `torch.equal` to the raw scaled EMA affine target.  Per-state reconstructed
  dueling-Q/probability residuals remain finite tolerance diagnostics, never
  bit-equality gates.
- Keep `dynamic_voc_mode=control`, fresh control from step zero,
  `voc_gate_confidence_weighted=false`, `voc_gate_adam_beta1=0`,
  `voc_gate_learning_rate=0.001`, `voc_gate_target_tau=0.1`,
  `voc_gate_q_temperature=0.05`, `voc_gate_temperature=1.0`, scheduler horizon
  `100000000`, `voc_train_epsilon=0.02`, `voc_eval_stochastic=true`, and all
  other v11 production values unchanged.

Actual behavior-probability slice statistics are instrumentation only.  They
must be present, finite, temporally aligned, and reconstructible from stored
behavior logits, but they have no new barrier, threshold, loss, gradient,
selection role, or acceptance role.  No result may be used to tune epsilon or
add a sign/depth override.

## Immutable identities and no-retry rule

Every v12 run is fresh: `ckp=false`, `preload=''`, `preload_actor=''`,
`voc_parent_checkpoint=''`, zero parent-update count, and exactly zero online-
Q, EMA-Q, and scalar-gate heads before training.  Wire, qualification, and
primary runs are separate processes and output directories; no checkpoint or
observation passes between them.

- The one 1.2k integrity wire uses base seed exactly 1 and is nonqualifying.
- The one fresh 100k qualification uses base seed exactly 1.  It is a separate
  run, not a continuation of the wire.
- The one primary 300k run uses base seed exactly 4, the next unused seed
  selected prospectively for v12.  There is exactly one primary attempt.

Seed 1 is not a backup for seed 4.  No seed may replace, pool with, majority-
vote, or rescue a frozen stage.  A crash, corruption, provenance failure,
qualification failure, primary failure, or fixed-evaluator failure is the
failure of that stage.  Same-run extension, resume, retry-selection,
replacement seed, threshold change, and result-conditioned mechanism change
are forbidden.

Create v12 from the immutable v11 snapshot as an independently copied,
separately named snapshot plus an exact enumerated overlay containing only the
one execution mechanism/schema, dual-surface telemetry, public/fixed
validation, corresponding tests, and this document.  Preserve the v11
snapshot, result, documents, unrelated behavior-bearing files, Cython
extensions, and staged Enduro data byte-for-byte.  Freeze source and data
manifests outside `src`, seal modes, exclude historical run outputs, prohibit
links/special/writable source nodes, and prove inode independence before any
launch.

## Sequential release gates

The stages are strictly ordered:

1. Freeze and independently audit the v12 snapshot, exact v11-to-v12 overlay,
   manifests, runtime binding, binaries, tests, public/fixed validators, and
   this document.
2. Run one seed-1 1.2k integrity wire with `total_steps=1200`,
   `model_warm_up_n=512`, `actor_unroll_len=41`, W&B disabled, CUDA devices
   `0,1`, Ray requesting two GPUs, and otherwise the production mechanism with
   `schedule_total_steps=100000000`.
3. Only after the wire passes integrity, run one separate fresh seed-1 100k
   qualification with `total_steps=100000`, production warm-up 10000, actor
   unroll 201, W&B enabled, and the frozen qualification below.
4. A failed 100k qualification permanently ends v12 and forbids the primary.
   A pass authorizes exactly one separate fresh seed-4 primary with
   `total_steps=300000`, production warm-up 10000, actor unroll 201, W&B
   enabled, and otherwise identical production configuration.
5. Only if the seed-4 300k training telemetry and artifact gates pass may its
   single terminal checkpoint be evaluated with closed profile `v12-300k`.

Pong and Space Invaders remain out of scope until every Enduro v12 stage
passes.

## Integrity-wire acceptance

The wire may be inspected only for launch/exit, snapshot/data/source binding,
schema/config/provenance, execution and soft-surface mechanics, exact
projection, counters, checkpoint completeness, CSV header/width, recursive
finiteness, W&B-disabled identity, and process/GPU/Ray cleanup.  Reward,
support, conditional behavior, stability, and acceptance values are forbidden
inputs to a decision or edit.

Before a supported update, Q/EMA/gate W/b remain exact zero, the soft and
execution gate probabilities are both 0.5, and the pre-existing first-tie
loss/counter invariants hold.  A later supported nonzero gate must show the
correct soft probability separately from actual `.99/.01` execution when both
gate actions are legal; the stored sampled-action probability and V-trace
likelihood must match the execution surface.  The original PROCEED/RESET and
main-action distributions must remain unchanged.  A positive/negative Q or
depth change may not itself select an action.

The frozen unit/contract tests exercise exact-zero tie, positive and negative
non-ties, one-action legal masks, Q-skip/no-support, and actor-skip branches.
The wire does not require a skip or one-action-only branch to occur.  If such a
branch occurs, its state transition must agree with the contract; any actor or
Q AMP skip makes the qualifying wire fail because all required skip counts are
zero.  A wire failure blocks qualification.  Success authorizes only the one
fresh 100k qualification and supplies no qualifying observations.

## Frozen 100k qualification

The v11 section **Frozen 100k qualification** is incorporated verbatim except
for the schema-5 execution identity and dual-surface consistency additions
below.  Thus the canonical gating population is `70000 < real_step <= 100000`,
the three windows are `(70000,80000]`, `(80000,90000]`, and
`(90000,100000]`, and overshoot above 100000 is excluded.  Population,
pooling, support, malformed-input, Q-sign tie, trailing-five, calibration, and
safety algebra remain the hashed v9/v11 definitions.

Every v11 numeric qualification gate remains unchanged:

- teacher gap at least `0.075`, student gap at least `0.05`, retention at
  least `0.50`, and signed margin strictly positive;
- at least two windows with both student gap and signed margin strictly
  positive;
- maximum consecutive negative trailing-five pooled gaps at most 3, with
  exact zero non-negative;
- train and holdout CONTINUE and STOP fractions each strictly greater than
  `0.05`, with positive denominators;
- wrong-continue saturation strictly below `0.01`; wrong-stop remains the
  report-only 100k safety diagnostic;
- online-versus-EMA non-tie sign agreement at least `0.60`;
- held-out EMA selected-action TD RMSE at most `0.5`;
- actor, online-Q, and gate AMP skips and every non-finite counter exactly
  zero.

Schema 5, exact projection, and epsilon-greedy execution are additional hard
identity/integrity gates, not new behavioral thresholds.  Configuration,
actor, ModelNet, and resolved records must agree; the soft and execution
surfaces must have their frozen meanings; stored behavior probability and
V-trace likelihood must match actual execution; successful Q/EMA/projection
counts and bit-exact W/b projection must reconcile; and gate optimizer state
must remain empty with pristine scheduler/scaler state.  Failure ends v12 and
forbids the 300k run.  A pass cannot tune the mechanism or thresholds.

## Frozen 300k primary acceptance

The hashed v10 sections **Canonical rows and windows**, **Frozen metric
algebra**, **Trailing-five definition**, **Mechanism acceptance**, **Four
required learned behaviours**, and **Stability behaviour semantics** are
incorporated verbatim.  V12 changes none of their labels, populations,
denominators, supports, or numbers:

- full is `100000 < real_step <= 300000`;
- late is `250000 < real_step <= 300000`;
- W1/W2/W3 are `(270000,280000]`, `(280000,290000]`, and
  `(290000,300000]`;
- support floors remain deep-negative 256 full/64 late; PROCEED and RESET 256
  full/64 late each; next-positive and next-negative 128 full/32 late each;
- soft learned-gate probability thresholds remain `0.475/0.525`;
- sampled-control strength remains `0.525`, conditional argmax remains
  `0.60`, useful-pair coverage remains `0.95`, sign agreement remains `0.60`,
  support fractions remain strictly above `0.05`, both wrong-side saturation
  rates and forced-stop rate remain strictly below `0.01`, and heldout RMSE
  remains at most `0.5`;
- both full and late must pass mechanism acceptance and all four behaviors;
  all three stability windows must retain every strict direction, at least two
  must meet every `.525` strength condition, and equality at 0.5 fails strict
  direction;
- the full-window maximum consecutive negative trailing-five run remains at
  most 3, and every AMP/non-finite safety event remains exactly zero.

For avoidance of doubt, B1--B4 and stability `p_continue` values are pooled
from the separate soft learned-gate probability, while sampled successes are
the actual controls drawn from the epsilon-greedy execution distribution.
The same integer-success reconstruction, forced-stop exclusions, and exact
0.5 failure rules apply.  Conditional argmax uses the execution distribution;
for every nonzero scalar it agrees with the soft gate's own argmax, while its
correctness against EMA-Q sign remains subject to the frozen `0.60` criterion
and Q-sign ties remain excluded by the unchanged `1e-6` rule.  Actual
behavior-probability fields are reported but cannot replace soft
`p_continue` or create a new threshold.

All artifact/provenance, mechanism, behavior, stability, trailing-five,
calibration, support, saturation, forced-stop, AMP, and non-finite gates must
pass.  There is no partial, mechanism-only, support-only, diagnostic, or
historical pass.  The old `(70000,100000]` primary-run diagnostic remains
report-only; overshoot, optional seeds, continuation, and post-hoc pooling
cannot change the decision.

## Fixed-checkpoint confirmation and final claim

Only after the seed-4 300k training decision passes may its exact terminal
checkpoint be evaluated once with closed profile `v12-300k`.  The evaluator
uses training disabled, gate epsilon zero, `greedy=False`, seeds
`20260827..20260842`, exactly 16 streams by 6250 real steps (100000 total), and
calibration V-trace unroll 201.  The primary action and conditional
PROCEED/RESET remain sampled.  The binary gate is deterministic by the sign of
every nonzero scalar; only a bit-exact zero is sampled at 0.5.  It must not be
described as a generally stochastic soft gate.  Diagnostic or shortened
evaluations are confirmation-ineligible.

The profile fails closed unless configuration, actor, and ModelNet all bind
`base_seed=4`, `total_steps=300000`, `schedule_total_steps=100000000`, control
mode, schema 5, dedicated soft-Q gate enabled,
`voc_gate_param_align=false`, exact coefficient `1.0`,
`voc_gate_exact_projection=true`,
`voc_gate_epsilon_greedy_execution=true`, `voc_train_epsilon=0.02` exactly,
`voc_eval_stochastic=true`, and fresh empty preload/parent identity.  Runtime
evaluation nevertheless sets gate epsilon to zero as stated above.  It
validates terminal FP32 gate W/b against the raw scaled EMA affine target with
bit equality and preserves the schema-5 dual surface.  Training epsilon must
be a finite non-boolean numeric value exactly equal to `0.02`; booleans,
non-finite values, and adjacent representable mismatches fail.  The resolved
identity, including both epsilon fields, must be recorded in the summary,
evaluation protocol, and manifest.

Fixed B/calibration probability uses `voc_gate_soft_continue_probability`;
sampled actions, action probability, and V-trace use `search_control_logits`.
Apply the same four frozen behavioral definitions and supports, never count
forced stops as sampled or argmax successes, and report selected-action
heldout calibration.  As in v10/v11, fixed-checkpoint calibration is required
reporting and finite audit, not a newly invented post-hoc RMSE gate unless the
hashed fixed-confirmation text explicitly makes it one.

The Enduro v12 claim requires immutable artifact/integrity success, the one
seed-1 100k qualification pass, the one seed-4 300k training-telemetry pass,
and its eligible `v12-300k` fixed-checkpoint pass.  Failure at any stage is
permanent for this preregistered experiment.
