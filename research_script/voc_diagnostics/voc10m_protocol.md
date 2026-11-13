# Latest VOC, sub001, three games, 10M each

Authorized 2026-09-25: a single VOC condition, no detach ablation. The GPU-count
question did not authorize extra seeds. Three fresh training runs, seed 701,
Pong / Enduro / SpaceInvaders, human sessions 2–4 train, session 1 held out.
Each run stops at exactly 10,000,000 real environment actions across 8 envs;
total scientific training = 30M. Evaluation and smoke actions are separate.

Latest mechanism = exact EMA→gate projection plus selective VOC→tree GRU VJP.
Keep reward-based VOC, current-stage compute cost 0.0005, gamma .99, log1p
count/depth, search/depth caps 30, tree carry, soft stochastic training gate
(Q temperature .05, uniform gate epsilon .02). No new gate or Q objective.
Actor and world model use FP32 as a declared stability precaution; GroupNorm
dynamics and inactive auxiliary-loss exclusion remain. FP32 does not guarantee
the absence of value divergence. Keep inherited BC-per-actor-update schedule;
there is no between-condition comparison requiring a new schedule. Count actual
BC targets and attempted/committed updates so exposure is explicit.

Geometry: 8 envs, actor batch 8 / unroll 64, model batch 8 / unroll 10 / return 5,
model replay 100k, replay ratio 4, model warmup 1024, BC batch 8.
Stop budget 10M; LR schedule horizon stays 100M. Every fresh game starts from
random weights, without previous 100k/200k/5M weights.

One 24GB GPU per training run, based on real GPU preflight. Additional GPUs
execute immutable-checkpoint evaluations. Pool capacity across bml-compute07,
08 and 10 is 4+4+2; other users keep their allocations and our jobs queue.
The controller submits at most ten active/pending GPU jobs, not ten extra jobs
per node. It consumes one CPU and 1GB on compute08. Free capacity can remain
unused when there are no scientifically scheduled evaluation tasks.

Snapshots at 0/.2/.5/1/2/3/5/7.5/10M; recovery snapshots every ~100k or 30min.
Training exits orderly after at most 5.5 days per six-day allocation, with an
atomic committed actor/model/runner bundle. Controller resumes only successful
PAUSED runs. Numeric failures require inspection, not automatic repeated retry.
Cold resume restores optimizer, EMA, schedule, learner and BC RNG/counters;
environment, planner, recurrent state and model replay restart. New replay-fill
actions count toward 10M. This is explicitly not bitwise trajectory continuation.
All code/config snapshots are frozen before submission. Attempts and failures
remain visible; no overwrite of earlier checkpoints or campaigns.

All training/probes and evaluation jobs log online W&B in the campaign group.
Training x-axis is real_step. Main additional metrics:

| Prefix/file | Meaning |
|---|---|
| actor/rmean_episode_return | Existing rolling raw full-game episode return; use episode count to identify initial placeholders |
| train/mean_last_100_episode_return | Independent completed-game mean, absent before first game completion |
| train/max_completed_episode_return | Correct maximum even when all Pong returns are negative |
| train/raw_reward_per_1000_actions | Accumulated raw reward divided by real actions; distinct from episode return |
| train/count_*_decisions | Completed-decision imagination histogram, plus zero/cap rates |
| train/learned_stop_fraction | STOP among eligible control queries, excluding forced stops |
| train/forced_stop_count | Forced stop count, separate from learned choice |
| train/bc_targets_* | Attempted and committed valid human targets, separate from environment steps |
| gradient/* | Selective VOC gradient, actor GRU gradient, cosine, actual GRU update, commit flag |
| actor/voc_gate_* | EMA TD calibration, exact projection error, gate agreement, support by count/depth; diagnostic, not independent utility |
| model/* | Dynamics/reward/value errors, feature/logit maxima, clipping and update counts |
| resources/* | GPU allocated/reserved peaks and total device memory |
| control_samples.jsonl | Deterministic stream-0 / every-128th-decision sample: count, depth, online/EMA VOC, gate probability, actual control, forced flag, root-policy margin |

Native evaluation retains the training stochastic gate and epsilon; sign
evaluation uses EMA DeltaQ>0, STOP on exact tie, no gate exploration, respecting
legal masks. Real-action sampling and conditional PROCEED/RESET are unchanged.
Each milestone also evaluates zero imagination. At 1/3/5/10M, independently
calibrate mean compute and compare fixed/random allocation, reporting achieved
budget mismatch and paired reward intervals. At 3/5/10M, add 50k actions/stream
to observe completed full-game episodes; report streams without completed games
and do not treat truncated partial games as completed scores.

At 1/3/5/10M, evaluate actual paired STOP/CONTINUE benefits under native
continuation. Match exact environment/planner/recurrent entry identities and
verify repeat STOP, with independent real-action random numbers. Use clipped
discounted task returns minus remaining current-stage compute cost, horizon 1536
or critic-terminal/life loss. Report finite-tail bounds, reward-change support,
EMA/online RMSE versus zero, rank correlation, allocation gain and stream-cluster
intervals. Early probes: 8 streams × 4 entry strata × 2 repetitions = 64 pairs.
At 5M/10M: 16 streams × 4 entry strata × 4 repetitions = 256 pairs. This is
broader than the original pilot but still limited; one training seed is not
seed-robust evidence. Native VOC calibration must not be confused with sign
policy returns, which change the continuation policy.

Success requires improving task reward and beneficial computation allocation
relative to comparable-compute baselines, not just variable counts, teacher
agreement or STOP AUC. Do not choose a checkpoint or tune parameters using
held-out outcomes. Human session-1 downstream replay is not implemented by this
reward-learning launcher and is not claimed as an output of these jobs.
