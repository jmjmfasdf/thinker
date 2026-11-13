# sub001 three-game VOC training plan — 2026-09-25

Status: mechanism audit and proposed submission protocol. **No jobs submitted by this plan.**
The user confirmed one fold: human sessions 2–4 for training, session 1 held out,
three games, two gradient conditions. The settings below are recommendations
for the next campaign, not a claim that its runner or GPU validation is complete.

## 1. Question and six training cells

Does allowing reward-based VOC learning to update the tree GRU improve prediction
and allocation of useful computation after sufficient task learning?

| Game | Environment | Human game ID | Detached | VOC → tree GRU |
|---|---|---:|---:|---:|
| Enduro | Enduro-v5 | 0 | 10,000,000 | 10,000,000 |
| Pong | Pong-v5 | 1 | 10,000,000 | 10,000,000 |
| Space Invaders | SpaceInvaders-v5 | 2 | 10,000,000 | 10,000,000 |

- Subject `1` = sub001; train sessions `2,3,4`; held-out session `1`.
- Recommended start: fresh initialization, seed 701, **10M total**, not 10M added
  to old checkpoints. Save one initial actor/model/EMA bundle per game and give
  the two arms identical weights and RNG initialization. Record hashes.
- Total training budget: **60M actual environment actions**, summed over the
  parallel environments. Evaluation, GPU time, imagination and optimizer updates
  are separate counters. Do not combine fresh and old 5M lineages in this table.
- RL plus human imitation remain active. Human likelihood is secondary;
  improvement in actual task reward and allocation efficiency is primary.
- This is one subject, one fold, one training seed. Evaluation-stream confidence
  intervals do not establish training-seed or participant generality.

## 2. Audit: what to keep and what needs work

### Reward target and gradient routing

Keep the current reward-based target. `compute_dynamic_voc_target` combines a
detached recursive task-return target with a detached computation-cost target.
Discount gamma=0.99 applies on real transitions, not on zero-time imagination.
The cost target ends at the current search-stage boundary. Thus the current
objective is **discounted task return minus remaining cost in this planning
stage**; it does not price every computation in all future planning stages.
Keep this definition explicit in the evaluator rather than silently changing it.

`DeltaQ = Q(CONTINUE) - Q(STOP)` already includes cost. Do not subtract cost again
when selecting a gate action. Do not replace this target with action switching,
root-Q argmax changes, or improvement in human imitation likelihood.

Keep selective GRU routing as the only between-arm training intervention:
the VOC gradient updates the VOC head in both arms and additionally the tree
GRU in the connected arm. Targets, EMA and other encoders retain detachment.
The existing feature-leaf/VJP implementation merges the unscaled extra gradient
before actor clipping; simply deleting every `.detach()` would not preserve
the separate optimizer/AMP semantics. Both arms must log the candidate gradient,
whether it was applied, and whether the optimizer actually committed the update.

### Teacher-to-gate connection is already repaired in this lineage

Both completed pilot arms used exact projection of the EMA head into the gate.
The frozen gate has no independent optimizer update; reported projection error
was zero and teacher/gate KL was approximately numerical zero. Historical weak
distillation is not the current bottleneck. Confidence weighting and gate Adam
hyperparameters are not useful fixes while exact projection bypasses that path.

### Execution policy remains a material limitation

For an unconstrained binary gate, current training uses

`P(CONTINUE) = 0.98 * sigmoid(DeltaQ_EMA / 0.05) + 0.01`.

With cost 0.0005, an illustrative perfectly estimated net gain of -0.0005 gives
P(CONTINUE)=0.49755004. At +0.0005 it gives 0.50244999. These were checked with
the actual distribution helper; they are examples, not estimates of typical
checkpoint DeltaQ. A good critic alone need not make this soft policy economical.

Recommendation for the six-run comparison:

- Retain stochastic projected-gate training, temperature and epsilon in both
  arms, so exploration and the gradient intervention remain interpretable.
- Predeclare **two evaluation policies for every checkpoint**: native soft gate
  with epsilon=0, and sign gate with epsilon=0, CONTINUE iff DeltaQ>0. Use STOP
  for exact ties in the proposed deterministic evaluator, subject to legality.
  Keep the conditional PROCEED/RESET and real-action policy unchanged.
- The existing epsilon-greedy helper uses a 50/50 exact-zero tie. A deterministic
  STOP tie therefore needs an explicit evaluation implementation and test.
  Preserve the training checkpoint schema; do not flip incompatible flags on an
  old checkpoint and present it as native training behavior.
- Evaluate reward and compute for each policy separately. Paired VOC calibration
  under the original training continuation policy is the primary critic test.
  Changing continuation to a sign policy changes Q's estimand; separate return
  measurements under that policy are a transfer test, not guaranteed calibration.
- If sign execution helps while native execution wastes compute, a subsequent
  matched training experiment can test epsilon-greedy execution. Do not change
  temperature, cost, execution and gradient routing together after seeing scores.
- Monitor CONTINUE/STOP support by game, count and depth. Early sign execution
  can eliminate the very computations whose benefit must be learned.

Keep `log1p(count)` and `log1p(depth)`, the search cap 30, and depth cap 30 for
this experiment. There is no evidence from this audit that rescaling these
inputs fixes the reward-learning bottleneck. Report cap-forced stops separately
from learned stops. A fixed computation price tests state-dependent allocation;
it does not establish adaptation to an externally varying price/budget input.

### Fair BC exposure and numerical stability

Current `_maybe_compute_imitation` schedules BC per actor update. The completed
100k continuation used 952 vs 1011 successful BC updates in Enduro and 1067 vs
1031 in Pong. Equal real actions therefore did not mean equal human exposure.

For this campaign, propose a common **one BC batch per 128 actual environment
actions**, batch size 8, inherited sequence/valid-target rules, coefficient 1.
This gives 78,125 scheduled batches at 10M. Freeze a shared human sample order
within each game pair. Save scheduling remainder, batch indices, attempted and
successful batches, and valid-target counts. Pending batches cannot disappear
when an actor unroll crosses several thresholds or after resume. BC weighting
must preserve its intended contribution per scheduled batch; equalizing batch
counts alone does not guarantee equal gradients or optimizer dynamics.
This is a shared change from the pilot and requires implementation tests.

Recommend actor and world model FP32 in both arms for the initial long-run
configuration; VOC/projection remain FP32. Connected Enduro previously had eight
AMP-skipped actor steps during a transient encoder/logit explosion. FP32 is a
numerical precaution, not proof that overflow caused the underlying instability.
Keep finite-value checks, gradient clipping, GroupNorm dynamics and zero-weight
auxiliary-loss exclusion. Log encoder/logit maxima, clip ratios, gradient norms,
task-critic error, world-model prediction error and successful update counts.
Do not tune the two arms independently or silently fall back to AMP mid-run.

## 3. Common configuration to validate and freeze

Use the existing small synchronous pilot geometry as the proposed starting
point: env_n=8, actor batch=8/unroll=64, model batch=8/unroll=10/return_n=5,
model replay capacity=100,000, warmup=1024 actions, replay ratio=4. Keep inherited
RL/BC weights, actor LR=3e-4, model LR=5e-5, cost=0.0005, GroupNorm, tree carry,
EMA tau=0.1, gate Q temperature=0.05 and training epsilon=0.02. Use the same
configuration within each pair except selective VOC→GRU enablement.

Set `total_steps=10_000_000` explicitly. Retain `schedule_total_steps=100_000_000`
as in the pilots/historical runs; training length and schedule horizon are
different parameters. Record this choice rather than accidentally decaying the
inherited schedule to zero at 10M. Any throughput-motivated geometry change must
be fixed before scientific initialization, common to the pair, and documented;
do not choose it using held-out outcomes.

The current fresh runner only accepts Pong/Enduro; the continuation runner
requires a parent and does not implement long-run resume. Neither can be made
ready for this campaign by changing `--env-steps` alone.

## 4. Evaluation and interpretation

Save predetermined milestones at 0, 0.2M, 0.5M, 1M, 2M, 3M, 5M, 7.5M, 10M.
Primary endpoint is 10M, never the best observed checkpoint. Do not stop for
poor early reward; historical Pong first crossed -20 near 2.53M (median of 24
runs), and Enduro's historical median rose from 15.3 at 2M to 222.9 at 5M.
Those are historical training episode scores under different settings, not
promised outcomes or current held-out evaluation scores.

Separate four questions:

1. **Task learning:** held-out environment seeds, raw reward, completed full-game
   episode return and episode length. At every milestone use a fixed common
   action horizon for score/1000 actions; at 3M/5M/10M add full-game episodes.
   Do not treat life loss as the end of a full-game episode, mix clipped rewards
   with raw reward, or report a truncated episode as completed. Predeclare the
   final seed list, episode count and timeout in the evaluation manifest before
   submission. A reasonable initial profile is eight streams × 4096 actions for
   the light check and 16 independent full-game episodes, each capped at 50,000
   actions, for the full check; report timeout/censoring explicitly.
2. **Useful budget allocation:** evaluate each policy against zero imagination,
   fixed allocation and state-independent random allocation. Calibrate the last
   two on independent seeds to the learned policy's mean compute; report achieved
   mismatch and compare reward at comparable compute, not just larger budgets.
   At 10M add a fixed-count curve for k=0,1,2,4,8,16,30 on a predeclared shared seed
   subset. Record count versus depth; PROCEED and RESET have distinct topologies.
3. **VOC validity:** paired STOP/CONTINUE from identical environment, planner,
   actor/recurrent and RNG entry states. Compare predicted net DeltaQ with actual
   discounted task-return difference minus current-stage compute-cost difference.
   Distinguish the critic's episodic-life terminal semantics from full-game score.
   Use multiple independently sampled streams and states, not only the previous
   fixed two prefixes. Report target variance, nonzero-benefit support, RMSE versus
   zero and a held-out constant predictor, sign/calibration, and allocation gain.
   A predeclared 32 independent streams × 4 states × 4 paired rollout replicates
   at 5M and 10M is a proposed starting sample; exact horizons and tail bounds must
   be frozen after throughput validation. Both signs and deeper-state support
   must be visible. Cluster uncertainty by stream, not by imagined event.
4. **Situation dependence:** does the gate identify where extra computation
   actually improves reward after conditioning on count/depth? Evaluate at shared
   count/depth strata and on a common held-out state bank where possible. Variable
   counts, gate/STOP AUC, teacher agreement, or decoding depth alone are not proof
   of useful adaptation. Human session-1 likelihood and both action-transition
   directions remain secondary checks, with no checkpoint selection from them.

Keep native-policy VOC calibration and changed-execution outcomes separate.
Within-checkpoint STOP/CONTINUE cloning is mandatory. Between-checkpoint raw
RMSE is not directly comparable when states or continuation policies differ;
use each condition's own zero/constant baselines and independent common probes.
Report learning curves and uncertainty for all three games, including failures
and null results. One game passing does not establish three-game consistency.

## 5. Submission order, restart and monitoring

1. Implement the common six-cell runner/config validation, fresh paired
   initialization, real-action BC schedule, evaluation-mode separation, correct
   episode logging and complete checkpoint metadata. Extend game adapters to SI.
2. Run SLURM GPU smoke tests for all games and both arms, including a short
   deliberate interruption/resume. Verify real RL/BC/model updates, exact
   projection, gradient ownership, fresh weight equality, episode logging and
   online W&B. Smoke results are excluded from scientific learning curves.
3. Run a bounded stability/throughput probe through at least 50k actions for
   Enduro connected and its matched control (the previous transient occurred at
   about 25–27k additional actions). Validate SI's actual environment and human
   loader too. Freeze precision/geometry using operational checks, not held-out
   reward. This is a prerequisite, not a claim of long-term numeric safety.
4. Prepare six one-GPU jobs in three matched game pairs. Prefer the same GPU
   model within each pair. At audit time (2026-09-25 19:05 KST), compute08 had
   2/4 GPUs unallocated; compute07 had 4/4 and compute10 2/2 allocated.
   Recheck before submission. Prefer starting Pong's pair first because the
   delayed-learning question is central, then Enduro, then SI, with other pairs
   starting earlier if suitable GPUs become available. Respect others' jobs.
5. The relevant partitions have a seven-day limit. The completed small runner
   achieved 12.89–14.64 real actions/s: roughly 7.9–9.0 GPU-days per 10M cell,
   **excluding evaluation and before the FP32/BC changes**. Six cells are roughly
   47–54 GPU-days under that obsolete speed; two GPUs imply about 24–27 days of
   training before evaluation/queueing. This is a sizing estimate, not an ETA.
   Benchmark the frozen configuration before promising a completion date.

For planned resumptions, recommend synchronized **2M-action segments in each
pair**, plus periodic atomic snapshots (e.g. every 100k or 30 minutes) and a
pre-time-limit checkpoint signal. Stop at 10M cumulative, not 10M per segment.
If benchmark speed cannot fit 2M safely inside a job, shorten the common segment
before launch. Separate training and evaluation status so evaluation failures
do not restart completed training.

Persist actor/model/EMA/action-prior weights, all optimizers/schedules/scalers,
VOC/projection counters, learner counters, human data signature and BC sample
RNG/schedule, Python/NumPy/torch RNG, source/config hashes, cumulative and attempt
counters, W&B identity, and snapshot consistency. Model/actor real counters must
agree at a committed bundle; record imagined transitions and wall time separately.

Existing resume resets environment, planner, recurrent state and online model
replay. Full serialization would be preferable if supported and tested. If cold
resume is retained, label it explicitly and use the same planned 2M boundaries
in both arms. Replay refill actions still count toward 10M; mark refill and
episode resets. An unscheduled retry is a protocol deviation and is visible.
Never claim bitwise continuation for a cold restart. Restored counters, schedule
and predictions must match; if exact state resume is implemented, also compare
uninterrupted and split trajectories. This long-run path is not yet validated.

All training and probes use online W&B with cumulative real_step as the x-axis.
Include `actor/rmean_episode_return`, completed-game episode count, independent
raw episode statistics, reward evaluation, compute histograms/zero-search rate,
cap-stop rate, VOC prediction metrics and losses/update counts. The legacy
RetBuffer starts max_return at zero, which can display an incorrect best return
for all-negative Pong episodes. Fix or replace that monitoring statistic; before
any completed episode display missing rather than a measured zero.

Scientific runs preserve immutable source/config/data bindings. A failure keeps
its logs and attempts; numeric failures are diagnosed before resubmission.
No automatic efficacy-based sweeps or 48-checkpoint expansion.

## 6. Checks actually completed in this audit

CPU-only pytest: **11 passed, 2 GPU-AMP variants skipped**, 335 deselected, 6.71s.
This is not a GPU campaign-readiness claim. Selected checks cover:

- Selective gradient equals explicit selective autograd (two CPU arms).
- Reward/cost target algebra and recursive delayed reward credit.
- Toy shared-state model can learn easy STOP / hard CONTINUE / post-compute STOP;
  erased state/depth cannot mimic that discrimination (not evidence on Atari).
- Executed-policy likelihoods used for V-trace, separated from soft calibration.
- Execution transform leaves soft-centered Q/loss definitions unchanged.
- Held-out actor streams excluded from Q fitting.
- Exact projection survives an independent simulated actor AMP skip.
- Learner-level exact-projection/sign-execution save/restore lifecycle (this does
  **not** test the new six-cell runner or environment/replay continuation).

Command environment: `CUDA_VISIBLE_DEVICES=''`, one CPU math thread,
`PYTHONPATH=thinker:research_script/voc_diagnostics`, conda environment
`/home/jeongmin/miniconda-dynamic/envs/dynamic-thinker`.
Tests: `test_voc_gru_gradient.py`, selected `test_dynamic_voc_learner.py`, and
`test_dynamic_voc_sequential_mdp.py`. Existing AMP-skip test emitted a scheduler
order warning; this audit did not redefine scheduler semantics or treat it as
the cause of previous divergence. Future runner validation must record scheduler
advancement on skipped steps explicitly.

Main source evidence:

- `thinker/thinker/actor_net.py`: `compute_voc_gate_distribution`, detached gate/Q.
- `thinker/thinker/learn_actor.py`: target, projection, discounts, BC schedule.
- `research_script/voc_diagnostics/voc_gru_gradient.py`: selective VJP route.
- `research_script/voc_diagnostics/train_gru_ablation.py`: bounded continuation.
- `research_script/voc_diagnostics/train_reward_learning.py`: fresh pilot defaults.
- `research_script/voc_diagnostics/train_pilot.py`: existing cold-resume semantics.
- `research_script/outputs/voc_gru_detach_20260925/report.md`: completed pilot.
- That campaign's `historical_pong_horizon/` and `historical_enduro_horizon/`:
  all-24-run horizon audits; legacy learning scale, not current-policy forecasts.

Next concrete deliverable: validated six-cell runner and immutable submission
manifest satisfying the prerequisites above. The present deliverable is the
reviewable mechanism audit and plan, not a submitted 10M campaign.
