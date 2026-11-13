# Reward-based VOC training pilot

The first training intervention is exact EMA-to-gate projection. It makes the
existing value estimate reach the gate; it does not establish that the estimate
is useful. Actor/world-model RL and human imitation objectives are unchanged.
No human action likelihood or argmax-change target replaces the VOC target.

## Policy and objective

The soft continue probability is `sigmoid((Q_ema,C - Q_ema,S) / 0.05)`.
Projection uses the existing production implementation and checkpoint schema 4.
The separate gate optimizer does not step. The Q optimizer and EMA update
rules are preserved. Projection follows successful Q and EMA updates; skipped Q
updates leave EMA and gate unchanged. Temperature, Bernoulli execution,
conditional PROCEED/RESET policy, training epsilon 0.02, and think cost 0.0005
remain fixed. Confidence weighting and gate optimizer settings stay in config
for provenance, but they are inactive for projection.

With training epsilon enabled, the executed continue probability is
`0.98 * sigmoid(delta_q / 0.05) + 0.01`. The periodic summary reports both raw
teacher-to-execution KL and epsilon-adjusted fidelity. A nonzero raw KL from
the retained exploration mixture is not a projection failure.

The critic's target includes discounted clipped task rewards and only the
remaining cost of the current search stage. It does not include all future
stages' computation costs. Cost is already in the net Q difference and is not
subtracted again at execution. `reward_probe.py` reports finite-horizon and
bootstrapped returns separately; bootstrap is not an independent critic test.

## Real pipeline, bounded workload

`train_pilot.py` uses the production SelfPlayWorker, SActorLearner, SModelLearner,
EnvPool and human sequence loader. It instantiates the synchronous worker
locally instead of launching Ray. Both SR and VP model branches, actor RL,
VOC/EMA, and actual human imitation updates must succeed for a chunk to pass.

For both control and projection, the smoke configuration uses four environments,
32 augmented steps per actor unroll, model unroll 4, model batch 4, human batch
2 with four scored targets, buffer warm-up 32 and capacity 1024, and replay
ratio 1. Architecture, active objective coefficients, search cap and discount
come from the subject-1/leave-session-1 source configuration. Models initialize
fresh; no 5M weights are imported. These are throughput reductions for execution
verification, not an equal-environment-budget efficacy comparison. Actor update
counts are explicit; actual environment steps can differ by gate condition.

Training always opens an online W&B run in the configured `thinker` project.
Actor/model scalar logs and periodic reward evaluation are streamed there, with
local JSONL/CSV copies. Authentication failure stops the pilot rather than
silently training without logging. Attempts have separate W&B identities.

Every four actor updates, training pauses, saves actor/model/config together,
records their individual counters and SHA-256 hashes, and runs paired reward
evaluation in a separate process. Evaluation therefore does not consume the
training RNG or change live parameters. Source Python/Cython, compiled modules
and config files are frozen under `source/runtime`. No existing campaign
checkpoint or running cluster job is overwritten.

## Commands

Run from `/work/jeongmin/thinker-dynamic-imitation`, on an available allocated GPU.
The completed local smoke used the idle development host GPUs; this is not a
cluster submission or a 48-fold retraining campaign.

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python \
research_script/voc_diagnostics/train_pilot.py \
--game pong --mode projection --updates 8 --eval-every 4 \
--output research_script/outputs/MY_NEW_PILOT
```

Use `--game enduro` and/or `--mode control` for the other cells. Outputs must be
new directories. Periodic evaluation defaults to one seed/prefix and 64 real
steps for smoke verification; `--eval-horizon`, `--eval-prefix`,
`--eval-repetitions`, `--eval-seed` configure a larger evaluator.

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python \
research_script/voc_diagnostics/train_pilot.py \
--game pong --mode projection --resume --updates 4 --eval-every 4 \
--output research_script/outputs/MY_NEW_PILOT
```

Resume accepts only a completed matching pilot chunk, unchanged saved checkpoint
hashes and unchanged source runtime. It restores optimizers, schedules, learner
counters, EMA and human-loader RNG. The environment, tree, recurrent state and
online replay restart; it is not bitwise trajectory continuation. All new
warm-up interactions count toward model steps. Existing schema-3 5M control
checkpoints cannot be relabeled as schema 4 or silently resumed into this pilot.

A serialized snapshot can be evaluated again into a new directory with
`reward_probe.py --bundle PATH/TO/bundle.json --output NEW_DIR --policy-epsilon`;
set `VOC_DIAGNOSTIC_RUNTIME` to that pilot's frozen `source/runtime`. To reproduce
the evaluator source as well, use its saved copy under `source/diagnostics/`.
Use `--policy-epsilon` when matching the training policy. Existing-checkpoint
T3 probes used evaluation epsilon 0 and are labeled accordingly.

## Bugs exposed and fixed by actual execution

- Synchronous Env dropped resolved actor flags, including `self_play_n`.
- Synchronous model writes still called remote replay, and replay reads omitted
  trajectory length and batch arguments. Model train/eval modes are now explicit.
- Synchronous human imitation had no source of fresh world-model weights.
- Trace source-index validation rejected valid `[B,E+1]` human training batches.
- Resuming with an empty local replay omitted the new warm-up interactions from
  the model environment-step counter.
- Added unweighted teacher-to-gate KL and teacher entropy logging, so BCE near
  ln(2) cannot be mistaken for poor soft-target fit. No loss coefficient changed.

## Acceptance and limits

Tests cover exact projection, Q/actor AMP skip isolation, save/resume protocol,
per-sequence source indices with real cenv rollout and gradients, synchronous
real-transition attribution, model train/eval modes, live imitation weights and
cold-buffer resume counting. Online evaluation verifies common-prefix actor,
RNG and tree fingerprints, unchanged parameters, and a repeated STOP trajectory.

Completed smoke cells and source/checkpoint binding checks are summarized by:

```bash
/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker/bin/python \
research_script/voc_diagnostics/summarize_training_pipeline.py \
--root research_script/outputs/voc_training_pipeline_20260924
```

These checks establish that the pipeline runs; they do not establish that
projection improves reward. Critic changes require more independent on-policy
states, repeated paired continuations to terminal or a justified tail bound,
held-out calibration, and compute-matched policy comparisons. The existing
critic target remains unchanged until that evidence is available.
