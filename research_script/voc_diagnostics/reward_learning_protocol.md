# Reward-based VOC learning pilot, 2026-09-24

This is a new exploratory experiment, separate from the completed pipeline smoke tests and the historical v7–v20 acceptance protocols. No existing 5M checkpoint is overwritten or resumed.

## Fixed design

- Enduro and Pong; subject 1, human training sessions 2–4 and held-out session 1; fresh seed 701.
- Per game: existing soft gate (control) versus exact projection of the EMA VOC head into the gate. All other objectives and hyperparameters match. Compare initial actor and model hashes.
- Each cell: 100,000 actual environment actions accumulated over eight parallel environments, including the 1,024-action model warmup. Imagination and evaluation actions do not consume the training budget.
- Actor unroll 64; model batch 8, unroll 10, return length 5; replay ratio 4; replay capacity 100k; BC batch 8. GroupNorm dynamics in both games. These are pilot settings, not an equal-compute 5M experiment.
- Preserve reward-based critic, RL and human imitation. Cost .0005, gamma .99, clipped rewards [-1,1], Q temperature .05, stochastic gate, epsilon .02, EMA tau .1; schedule horizon stays 100M. Gate projection changes signal transmission, not the definition of VOC.
- Projected execution probability is .98 × sigmoid(EMA delta Q / .05) + .01.
- Score checkpoints: 0, 10k, 25k, 50k, 100k. Paired VOC checkpoints: 25k and 100k. Evaluate fixed checkpoints; no selection by outcome.
- Four one-GPU jobs on compute08, maximum two concurrently, without cancelling existing work. Projection cells run first, controls follow. Online W&B group: voc-reward-learning-20260924. Immutable source manifest and per-checkpoint hashes bind all evaluations.

## Independent reward evaluation

At each score milestone, run eight environment streams with a fixed held-out environment seed (19101) for 4,096 real actions each. Sample the policy, including epsilon, with common keyed uniforms. Report raw task score per 1,000 real actions, full-episode scores where completed, and computations per real action. Learning is assessed against initialization and at equal environment steps against control. Bootstrap paired differences over the eight evaluation streams. Training episode returns and actor/model/VOC/BC losses are tracked separately in W&B.

## Does VOC predict useful computation?

At prefixes 128 and 1024 of eight fresh streams (seed 9101), reconstruct exactly the same environment, planner tree, recurrent state and history for STOP versus one forced CONTINUE (conditional PROCEED/RESET). Thereafter execute that checkpoint's policy. Repeat three times with common continuation random numbers. Repeated STOP must reproduce, prefix/tree hashes must match, and all model weights/buffers must remain fixed.

Evaluate actual discounted clipped task rewards for up to 1,536 real actions; stop each stream at the same life-loss/terminal definition used by the critic. Subtract the cost of the current search stage, not costs of future search stages. Do not use a critic bootstrap. A nonterminal paired omitted reward tail is bounded by 2 × .99^1536 / (1-.99) < .00005, less than one tenth of a single computation cost.

Compare pre-computation EMA delta Q with mean paired realized net return: RMSE, zero-predictor RMSE, RMSE improvement, rank correlation, and mean utility. Average repetitions within state, then cluster-bootstrap environment streams (preserving both prefixes). Effective independent sample count is eight streams, not 48 pairs.

Assess local allocation utility against state-independent random continuation with matching expected current-stage computation cost: p_random = sum(p_i cost_i)/sum(cost_i), gain = mean[(p_i-p_random) delta_i]. This is a local allocation comparison under checkpoint continuation. It is not a whole-policy fixed/random-budget stopping experiment.

## Interpretation and outputs

Exploratory positive evidence requires a positive 95% interval for the score difference, VOC RMSE improvement, or allocation gain respectively. Intervals crossing zero mean inconclusive, not equivalence. One subject/seed, few states, noisy rollout targets and only 100k fresh training actions limit conclusions; a null pilot cannot establish that planning is unnecessary.

Gate–teacher KL and projection invariants diagnose transmission only. A low KL or exact projection cannot establish critic accuracy or reward improvement. Retain stochastic execution for the controlled comparison; no deterministic-gate or objective change is mixed into this experiment.

Artifacts: per-cell status, online W&B, immutable milestone actor/model/config bundles, training episodes and losses, score and paired arm records; campaign learning_report.md/json, training_status.csv, score_curve.csv, voc_curve.csv, reward_comparisons.csv and learning_curves.png. Report regenerates after every evaluation and job exit. Missing cells remain INCOMPLETE. No automatic extension to 48-fold retraining.
