# Latest reward-VOC: sub001, three games, 10M parallel campaign

One condition and one training seed (701) for Pong, Enduro and Space Invaders.
Fresh actor/world-model initialization; human sessions 2–4 train, session 1 held
out. This launcher performs independent environment reward evaluation; it does
not claim to generate held-out human action replay.

## Geometry and provenance

All 72 historical config_c.yaml files used env_n=16, actor_batch_size=16,
actor_unroll_len=201, model_batch_size=32, model_unroll_len=30, BC batch16 with
four scored targets, self_play_n=1 and two GPUs. The number 32 was the model
batch size, not the environment count. Restore that parallel geometry: model
learner one GPU, self-play and actor learner share the other. 10 CPUs per job
fit compute10's 12 CPUs. Ray object store is explicitly bounded by job memory.

Preserve GroupNorm/clamped model dynamics. Actor and model are FP32, unlike the
historical actor AMP. Latest mechanism: exact EMA-Q gate projection and selective
VOC gradient into tree GRU; reward-based VOC target and current-stage cost remain
unchanged. Native training retains stochastic execution and inherited epsilon.
No extra seeds or detached comparison conditions are introduced.

The target is 10,000,000 actor-consumed actual environment actions. Native
parallel training ends at the first completed unroll reaching the target;
small batch overshoot and collector/model lead are explicitly recorded, not
rounded away. The learning-rate schedule retains the historical 100M horizon.
Model warmup=10k, replay capacity=200k, replay ratio4–5, BC per actor update.

## Logs and evaluation

Every training/evaluation run uses online W&B, project jmme425/thinker, grouped
by campaign directory. Collector metrics count physical real actions separately
from actor-consumed actions and imagined transitions. Added diagnostics include
actual actions/second, completed game returns (negative Pong maxima preserved),
raw reward per1k actions, exact count histogram0–30, zero/cap frequency, forced
STOP, RESET, BC targets, GRU gradient/actual update, GPU memory, and sampled
count/depth/root-margin/VOC/gate/control rows. Existing logs retain Q calibration,
support and gate projection error, actor losses, world-model prediction errors.

Immutable evaluation bundles at nominal milestones .2,.5,1,2,3,5,7.5,10M use the
first actor and model snapshot independently at/after each milestone, each in
[milestone,milestone+100k). Exact two counters, lag, hashes and source identity
are included. These asynchronous pairs are not called simultaneous checkpoints.
No untrained zero-step result is fabricated from an already updated checkpoint.

Evaluate native, deterministic sign gate and zero imagination at every milestone.
At every milestone additionally evaluate fixed/random calibrated mean computation
and paired STOP/CONTINUE benefits with actual discounted reward minus current-stage
cost. These asynchronous `benefit` jobs log VOC RMSE versus zero, rank correlation,
mean net benefit, allocation gain and fixed/random reward and compute differences
to W&B with the nominal `real_step` and exact actor/model counters in the saved
checkpoint binding. Achieved compute differences must be checked before claiming
equal-compute superiority. This is evaluation, not an additional training loss.
Routine recovery snapshots do not each trigger a GPU evaluation; the evaluation
checkpoints are the eight milestones listed above. Results arrive after the
evaluation finishes and remain in separate evaluation runs under the same group.

Long50k-action-perstream completed-game evaluations (`episodes`) are disabled by
default. They require an explicit `disabled_kinds: []` in `evaluation_policy.json`.
For an already running campaign, `benefit_from_milestone` can map each cell to its
next milestone: the expanded schedule then starts prospectively, while the
original1/3/5/10M benefit tasks are retained. The20260925 campaign switched on
20260926 with Pong starting at7.5M and Enduro/Space Invaders at5M; past gaps are
not backfilled. Active learning and immutable scientific source are unchanged.
Native-policy
VOC calibration and sign-policy returns remain separate. Held-out reward and
beneficial allocation establish utility; variable computation or gate agreement
alone do not. All planned checkpoints are evaluated, without outcome selection.

Training jobs each request two GPUs on compute07/08/10. From20260926, all future
evaluation submissions use partitionrtx2080 and check for an RTX2080Ti at startup;
the main nodes are reserved for training. Separate evaluation jobs may wait for
SLURM resources; each campaign limits active/pending evaluation jobs to four.
Failure is recorded and never automatically restarted from zero. Regular full
checkpoints are retained for an explicitly reviewed recovery.

The fresh `voc10m_tau001_20260926` campaign changes the training Q-to-gate
temperature to0.01, retaining the same seed/fold and learning geometry. Its
evaluation milestones are2/4/6/8/10M and its source is copied from the verified
0.05 runtime, with controller and GPU routing changes recorded in the manifest.
`prepare_voc_tau001.py` creates configs and scripts; it does not submit jobs.
Three-game short training plus RTX2080Ti benefit smoke must pass before the10M
jobs start. Existing0.05 training and its evaluation milestone schedule remain.
