# VOC detach continuation pilot — 2026-09-25

Question: Does letting the reward-based VOC loss train the tree GRU improve
prediction and allocation of useful computation? Exploratory; one parent
checkpoint and one continuation seed per game. No 48-checkpoint expansion.

Two conditions per game: detached control and selective VOC -> tree GRU.
Both branch from the previously audited final 100k projected-gate checkpoint
(Enduro update_001041, Pong update_001060). Inherit exact actor/world-model
weights, EMA teacher weights and action-prior EMA. Reset optimizers, schedules,
replay, environment, recurrent state and BC sampling stream in BOTH arms.
Store parent hashes and separate parent/additional counters. The ancestor's
fresh-control origin is retained as historical origin; branch provenance is
recorded explicitly, not described as a fresh random initialization.
Preserve the inherited Q/EMA/projection lifetime update counters to satisfy
the existing trained-gate invariants; report branch-local increments separately.

Each condition gets an additional 100,000 real environment actions aggregated
over eight environments. Seed 90211; inherited RL+BC objectives, learning rates,
GroupNorm world model, log1p counters, gamma .99, cost .0005, stochastic exact
EMA gate projection and exploration epsilon .02. Model replay warmup is 1024
actions within the additional budget. No optimizer state is inherited.

The implementation preserves the production detached head and AMP paths.
A feature leaf gives dL_VOC/dfeatures; a vector-Jacobian product through the
live feature graph selects GRU parameters only. Merge that unscaled gradient
into the actor gradient after AMP unscale and before existing global clipping.
Both arms compute and log the candidate gradient; only the experimental arm
applies it. This equals selective un-detaching by the chain rule. Target,
teacher, visual encoder, phase embedding and final readout parameters receive
no additional VOC gradient. Shared clipping and subsequent changed policies
may mediate effects elsewhere; this is a training intervention.

Validation before science jobs: compare against explicit selective autograd,
confirm identical forward outputs and parent weight hashes, verify actual
GPU AMP/RL/BC/model updates for both games and conditions. Smoke runs use
512 real actions and 32-action warmup (enough to fill a model training sequence);
exclude them from scientific results.
All training, smoke and production, records online W&B.

Save predetermined checkpoints at 0, 25k, 50k and 100k additional actions.
Run independent score evaluation at each. Final evaluation also collects
paired STOP/CONTINUE returns and full-policy fixed/random-budget baselines.
No checkpoint or encoding selection based on held-out outcomes.

Scores: eight held-out streams, seed 34101, 2048 real actions per stream per
milestone. Final budget baselines use the same score horizon/seeds. Calibrate
their mean budget from an independent learned-policy sample (seed 34181,
four streams, 1024 actions). Fixed allocation uses deterministic floor/ceil
counts to implement a fractional mean; random allocation samples a capped
geometric count independently of state with the same expected mean. Report
achieved compute counts and any mismatch, rather than claiming exact equality
of finite samples. Preserve the checkpoint's conditional PROCEED/RESET policy.

Final VOC targets: seed 33101, eight streams, prefixes 128 and 1024, forced
entry counts 0 and 3, two repetitions/state (32 states / 64 pairs/game/arm).
Count/prefix combinations are fixed in advance to broaden Pong coverage.
STOP repetition checks and exact entry identities are mandatory. Use actual
discounted task returns to 1536 real actions or critic terminal, including
episodic life; subtract remaining cost of the current search stage only.
No value bootstrap; report omitted-tail bounds. Real-action common uniforms
are independent of preceding search count. All prefixes/counts/repetitions
from one environment stream remain together for uncertainty estimates.

Report online and EMA VOC RMSE versus zero prediction, correlation with paired
benefit, local allocation gain and mean task/net benefit. Separately report
whole-policy reward and computation versus detached control and calibrated
fixed/random stopping. AUC of gate versus its own sampled STOP is not an
efficacy criterion. Use stream-level paired bootstrap intervals; one training
seed does not establish training-seed robustness. Negative or uncertain
results do not trigger automatic hyperparameter sweeps or longer training.

Resources: SLURM only for GPU work; four one-GPU cells on available compute08
after smoke passes. Respect other allocations; no job cancellation outside
this experiment. Immutable source/config/checkpoint bindings and failures are
retained. Numeric failures are diagnosed before any retry.
