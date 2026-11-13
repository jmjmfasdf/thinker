# VOC diagnostics: teacher fidelity and computation interventions

The offline probes do not update input checkpoints. The explicitly invoked
`train_pilot.py` runs bounded fresh training, with online W&B logs and periodic
reward evaluation. See [training_pipeline.md](training_pipeline.md) for its
configuration, executed checks, resume semantics and limits. Existing campaign
checkpoints are never overwritten.

## Current evaluation policy (2026-09-28)

The user requested no further separate game-score evaluation. Milestone
evaluation now schedules only paired STOP/CONTINUE VOC utility. Disable
`score`, `score_sign`, `score_zero`, and `episodes`; benefit evaluation must
also omit its former calibration and learned/fixed/random scoring stages.
Training-time reward logging and the discounted reward measurement needed
for paired utility remain enabled. Expanded sampling follows the same policy.

The active 0.05 and 0.01 campaigns use an evaluation-only overlay in each
campaign's `score_cancellation_20260928/` directory. Their frozen training
sources and existing results remain intact. Future campaign preparation
copies the paired-only evaluator as well as the scheduling policy.

## T0: teacher–gate KL

`gate_kl.py` reads all native control choices, including RESET. The EMA teacher
is `sigmoid(delta_q / checkpoint_q_temperature)`; the student is the cached
soft continue probability. It reports binary KL, cross entropy, teacher
entropy, probability error, and KL to a constant 0.5 reference. KL is in
nat/control choice. It does not measure whether the teacher is useful.

Aggregation is equal run means within a checkpoint, equal checkpoint means
within a subject, then equal subject means. Training sessions and held-out
sessions are reported separately. The SEM is across subjects. Every input
cache is listed with its config/binding/manifest hashes and file identity.
Invalid control choices are counted; corrupt valid probabilities fail the
cache rather than being silently discarded. Exact probability endpoints use
a reported `1e-12` numerical floor.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python \
research_script/voc_diagnostics/gate_kl.py \
--output research_script/outputs/voc_diagnostics_20260924/t0 --workers 2
```

Use a new output directory when repeating a completed run.

## T1: common-prefix replay

The Cython planner has a **recording** snapshot, not a complete restorable
tree snapshot. Environment `quick_save`/`quick_load` only restores the
underlying environment state. It does not restore the planner's C++ node
pointers, node statistics and latent states, pending real actions, actor
recurrent state, or RNG. Copying the recorded `tree_reps` is insufficient.

`forced_budget_probe.py` therefore reconstructs the same prefix from episode
start in a fresh planner for each arm. It retains the original actor/model
and conditional PROCEED/RESET policy, overriding only the target decision's
binary gate. It intercepts the real-head distribution **before** the target
human action is passed to the execution adapter, then closes that planner.

Required checks in every paired comparison:

- Same seed, deterministic CPU inference and full shared observation history.
- Exact hashes of prefix actor inputs/core/RNG, and target entry
  inputs/core/RNG/current tree observations.
- Actual executed computation count equals requested 0/1/3/5 budget.
- A repeated zero-budget arm reproduces exactly after the other arms.
- Network parameters and buffers are unchanged throughout the probe.

The frozen capture runtimes are selected explicitly through
`VOC_DIAGNOSTIC_RUNTIME`. The Enduro sub1/leave1 checkpoint uses `legacy`;
Pong sub1/leave1 uses `group`. These runtimes support the original campaign's
published 5M milestone interval and validate its bound artifact hashes.
The current root recorder has a different milestone schema, so its rejection
is not bypassed or weakened.

```bash
VOC_DIAGNOSTIC_RUNTIME=/work/jeongmin/thinker-dynamic-imitation/research_script/outputs/allcheckpoints_allsessions_5m_20260923/source/runtimes/legacy/thinker \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python \
research_script/voc_diagnostics/forced_budget_probe.py \
--checkpoint logs/thinker/dynamic-imitation-enduro-voc-sub1-leavesession1 \
--source sub-001/ses-01/sub001-ses01-block2-game0.npz \
--targets 4 16 32 --threads 2 \
--output research_script/outputs/voc_diagnostics_20260924/t1_smoke_enduro
```

Pong uses the `group/thinker` runtime, the corresponding Pong checkpoint,
and source `sub-001/ses-01/sub001-ses01-block1-game1.npz`.

Use `--first-transitions` instead of `--targets` to select the first boundary
in each direction directly from the original human action sequence, without
inspecting intervention effects. Targets are **zero-based action edges**;
the cache's `real_step` is one-based. The first boundary edges are Enduro
42/45 and Pong 33/36 in the selected episodes. The earlier directories
`t1_transitions_enduro` and `t1_transitions_pong` used cache indices without
conversion and therefore contain points **after** the transitions; their
saved labels correctly say active→active and NOOP→NOOP. They are excluded
from directional results. Corrected runs use `t1_transitions_verified_*`.

These are technical smoke tests on a few selected decisions, not subject- or
game-level estimates. Extra search also adds actor recurrent updates. An
equal-update/no-new-search-information control is still needed to isolate
the contribution of search content. Existing carried information is shared;
this tests additional computation at one decision, not the removal of all
past search. Human likelihood is distinct from task-reward VOC validation.

## Remaining experiments

1. Extend paired T1 to preselected held-out runs, all subjects/checkpoints and
   both transition directions. Predeclare practical effect sizes and compute
   confidence intervals at run/subject levels, not event level.
2. Add T2 (original gate versus exact EMA probability), preserving stochastic
   execution in both arms and comparing at matched average computation.
3. Split T3 into human-likelihood usefulness and actual reward-minus-compute
   utility. Validate the current C/S critic using STOP-now versus one
   CONTINUE followed by the existing policy in paired environment rollouts.
4. Choose a learning intervention only after the relevant link fails. A
   single training pilot checks implementation, not cross-checkpoint or
   cross-game reproducibility.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python -m pytest \
research_script/voc_diagnostics/test_gate_kl.py \
research_script/voc_diagnostics/test_forced_budget_probe.py -q
```

The tests include a label-leakage check: changing the target and future human
actions cannot change the pre-target state or target readout. The final
combined report and validated paired CSV can be regenerated with:

```bash
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python \
research_script/voc_diagnostics/summarize.py \
--root research_script/outputs/voc_diagnostics_20260924
```

## Follow-up: T1 content control and T2 gate projection

`intervention_probe.py` adds a three-computation arm with frozen initial
actor-visible information. It uses the normal search arm's exact imagined
primary/control sequence and preserves phase and recurrent update masks.
Frozen fields include tree/model predictions and imagined action/reward
feedback. This is an artificial information intervention; it is not a
trained model-free baseline or a proof of reward utility.

The learned and exact EMA gates both use Bernoulli draws, sharing uniforms
at corresponding control choices. Only the target decision is intervened
on. A single draw stream is used per decision; average computation is not
matched in this initial T2 comparison and is explicitly reported.

`run_interventions.py` predefines the first four human transition/persistence
strata within the first 256 edges of the first episode of the first held-out
source for every subject/game/leave-session checkpoint. Missing strata stay
explicit. It launches CPU evaluation only, verifies source hashes and reuses
completed outputs only under matching code and configuration with `--resume`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python \
research_script/voc_diagnostics/run_interventions.py \
--output research_script/outputs/voc_interventions_20260924/campaign --workers 8

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python \
research_script/voc_diagnostics/summarize_interventions.py \
--campaign research_script/outputs/voc_interventions_20260924/campaign
```

Primary interpretation is computational influence. Human action likelihood
is secondary descriptive fit. Reward-minus-cost utility needs a separate
online experiment; see `reward_validation_protocol.md`. In particular the
current critic charges only the current search stage's remaining cost,
while task rewards extend into future real actions.

`check_online_reconstruction.py` verifies seeded Atari reconstruction in
the installed EnvPool, whose direct state-save API is absent. Seed must be
provided at construction, not merely to the Thinker EnvPool reset wrapper.

## Learning reward and useful VOC decisions

`reward_learning_protocol.md` specifies the new Enduro/Pong 100k-environment-step
control versus EMA-projection pilot. `train_reward_learning.py` trains the real
actor, world model, reward VOC and human imitation jointly, with online W&B.
`reward_learning_eval.py` evaluates independent game scores and paired actual
STOP/CONTINUE returns with a bounded omitted tail, without critic bootstrapping.
`report_reward_learning.py` combines the four cells into curves and confidence
intervals; unfinished cells remain explicit. `monitor_reward_learning.py`
refreshes progress from SLURM and saved results without modifying any job.

The current submission is array 18479 on compute08 (maximum two GPUs), under
`research_script/outputs/voc_reward_learning_20260924/`. Training/evaluation run
from its immutable `experiment_source`, not this editable working directory.
`submission.json` records the source hash, cell mapping and budgets. The
collector updates `live_progress.json` and `learning_report.md` every minute
until the submitted jobs exit. Online W&B group: `voc-reward-learning-20260924`.

The successful 512-step path checks and earlier `train_pilot.py` smoke runs do
not establish reward improvement or useful VOC learning. Those claims require
the milestone evaluations in the new pilot, with uncertainty reported.
