# Dynamic imitation with previous-human-action initialization and tree carry

## Sequence contract

The first scored action is denoted by $a_t$.  Every training and evaluation
window contains one unscored burn-in transition followed by $L=4$ scored
transitions:

- The initial root is the recorded observation $o_{t-1}$.
- The initial previous action is $a_{t-2}^{human}$.
- The burn-in edge executes $a_{t-1}^{human}$ and is never included in loss,
  NLL, or accuracy.
- The first scored pair is $o_t \rightarrow a_t^{human}$.
- The remaining three scored pairs are consecutive recorded human decisions.

The corresponding batch schema is:

- `obs_seq[B,L+2,C,H,W]`: $o_{t-1}$, the $L$ scored roots, and the final next
  observation.
- `actions_seq[B,L+1]`: the burn-in action followed by the $L$ scored actions.
- `initial_prev_action[B]`: the human action immediately before the burn-in
  action.
- `rewards_seq`, `done_seq`, and `truncated_seq[B,L+1]`: edge-aligned recorded
  outcomes.
- `score_mask=[False,True,True,True,True]`.

Windows are built only inside a recorded episode.  They require two genuine
predecessor actions, never wrap to another sequence, and never synthesize a
terminal transition.  Timestamped recordings are sampled causally at $15$ Hz
($4/60$ seconds per Atari decision).  If an episode ends off that regular
grid, its genuine terminal image is retained as one final, possibly shorter,
transition endpoint.  Each observation contains four oldest-to-newest frames.

## What is teacher-forced

At reset, the same $a_{t-2}^{human}$ initializes all three action-dependent
states: ModelNet's first input action, the root-node action, and Actor
`last_pri`.  The Dynamic Actor then searches normally at $o_{t-1}$.

When Actor requests a real action, its logits, sampled proposal, and argmax are
retained for scoring.  A separate execution tensor replaces only the
`POLICY_REAL` primary action with the current human action.  Actor's imaginary
primary actions and its `PROCEED`, `RESET`, and `STOP` choices are not given
human targets and are not overridden.  The human action drives the recorded
environment step, the real ModelNet update, the carried-child choice, and the
next root's `last_pri`.  The sequence cursor advances only when cenv emits a
full-batch `real_transition`.

## Conditional tree carry

`tree_carry=true` is permission to carry, not a claim that every transition is
carried.  The human-action child becomes the next root only if:

- search expanded that exact human-action child; and
- the recorded transition is nonterminal.

Otherwise cenv creates a fresh root.  The per-root `root_carried` telemetry is
therefore the authoritative carry indicator.  In particular, the carry status
stored for the first scored root describes the result of executing the
unscored burn-in action.

## Training configuration

The initial subject-specific experiments are fixed to:

- subject 1 sessions 1--3 for training and session 4 for holdout evaluation;
- $L=4$ scored actions plus one unscored burn-in action;
- `max_search_steps=20`, `max_depth=20`, and `model_unroll_len=20`;
- fixed `think_cost=0.0005` with `think_cost_anneal=false`;
- Actor mixed precision enabled, but ModelNet training in FP32
  (`float16=true`, `model_float16=false`);
- ModelNet learning rate `0.00005`, gradient clipping `10000`, and
  `model_disable_bn=false` (the effective BatchNorm architecture used by the
  historical Thinker checkpoints);
- depth-0 state projection `clamp` with Smooth-L1 range-loss coefficient `1`;
- normalized cross-entropy coefficient $1$;
- DQfD margin size and coefficient $1$;
- PVP coefficient $0$ and overall BC coefficient $1$;
- action-prior coefficient $1$ and prior EMA $0.05$.

The parameterized Slurm entrypoint encodes these values and obtains the action count at
runtime from the selected environment's zero-based `Discrete` action space.
It then requires every selected behavioral archive, ActorNet, online ModelNet,
behavioral ModelNet, replay environment, tree schema, and action prior to use
that same count.  Enduro/game 0 resolves to 9 actions and Pong/game 1 resolves
to 6 actions; these dimensions are observed contracts, not model-head
constants.

The two prepared command forms are shown below.  They are not run by the
implementation or smoke-test scripts and still require explicit submission:

```bash
cd /home/jeongmin/thinker-dynamic-imitation
sbatch --job-name=dyn-il-enduro-s1 \
  train_dynamic_imitation_compute07.slurm Enduro-v5 0 enduro
sbatch --partition=rtx3090 --nodelist=bml-compute08 \
  --gres=gpu:rtx3090:2 --job-name=dyn-il-pong-s1 \
  train_dynamic_imitation_compute07.slurm Pong-v5 1 pong
```

The first command requests two Titan RTX GPUs on compute07; the second requests
two RTX 3090 GPUs on compute08. Both use model, Actor, environment, and BC batch
sizes $32/16/16/16$. Do not replace
`icopro_train_sessions=1,2,3` with session 4.  The imitation runner uses the
ordinary Dynamic `cenv`; there is no separately maintained `cenv_bc` path.

### Numerical-stability gate

The 20-step recurrent world model is deliberately not trained under FP16.
In the failed Enduro run, the state-reconstruction gradient first overflowed
shortly after model warm-up. The skipped AMP update was followed by a
non-finite replay priority, which contaminated prioritized replay and then the
frozen behavioral planner. Model FP32 removes that observed overflow path and
the lower learning rate reduces decoder growth. Because depth-0 predictions
represent normalized observations, they are also projected into `[0, 1]`
before frame stacking, VP encoding, planning, or Actor encoding. A Smooth-L1
penalty pulls the raw decoder output back toward that interval without an
unbounded squared penalty. Actor AMP remains enabled.

Model input, target, SR/VP losses, predicted states, gradient norm, complete
state dict, replay priorities, normalized sampling probabilities, and
importance weights are now fail-fast checked. A failed learner is propagated
to self-play and prevents the atomic `finish` marker. The model log includes
raw prediction absolute maximum and out-of-bounds fraction, projected minimum
and maximum, range loss, priority min/max, optimizer-step status, AMP scale,
model-precision mode, and learning rate.

Before a week-long run is accepted, the same production path must complete a
30,000-step numerical regression gate. A 70,000-step gate additionally checks
that Dynamic search has not collapsed to forced maximum-budget search. These
gates require finite actor/model logs and checkpoints, matching imitation
schedule/update counts, finite normalized action-prior EMA of runtime width,
and a valid completion bundle.

## Session-4 paired evaluation

The evaluator reconstructs ActorNet and ModelNet from `config_c.yaml`,
`ckp_actor.tar`, and `ckp_model.tar`.  It additionally requires the top-level
`finish` marker written only after both learner futures resolve successfully,
at least one completed imitation optimizer update, and positive actor/model
training progress.  Training removes any stale marker before a fresh or
resumed attempt, then atomically writes a JSON marker whose hashes bind the
final actor/model/config files and the staged training sources.  The evaluator
recomputes those hashes before loading data.  It rejects checkpoints that do not use
the $20/20/20$, `think_cost=0.0005` protocol.  Environment name, behavioral
game ID, subject, train/holdout sessions, preprocessing, and action count are
resolved from the checkpoint plus a live environment rather than an Enduro
table.  Actor and model embedded identities must agree, and the evaluator
recomputes the training-data signature from the selected train-session files
before it opens the holdout sequence.  It then runs each window twice with
identical inputs, greedy action selection, and paired seeds:

```bash
cd /home/jeongmin/thinker-dynamic-imitation/thinker
python evaluate_dynamic_imitation.py \
  --checkpoint-dir /absolute/path/to/checkpoint \
  --data-root /home/jeongmin/thinker-dynamic-imitation/behavioral_data_block \
  --expected-env-name Enduro-v5 \
  --expected-game-id 0 \
  --device cuda
```

The two `--expected-*` arguments are assertions only; they never override the
checkpoint.  For Pong use `Pong-v5` and game ID `1`, or omit both assertions
and let the checkpoint remain the sole identity source.

The canonical stride is $4$.  Thus scored targets do not overlap between
windows; the previous window's final scored action serves only as the next
window's burn-in action.  The summary reports the exact number of eligible,
emitted, unique, and skipped short-tail targets.  `--stride 1` is available as
an explicitly labelled sensitivity analysis in which target actions occur in
multiple planning contexts.

The output directory contains:

- `paired_steps.csv`: one row per scored target, including both NLLs, both
  proposals/argmaxes, source identity, and actual carry status;
- `summary.json`: overall metrics and metrics restricted to rows where the
  carry-enabled run has `root_carried=true`;
- `manifest.json`: hashes for configuration, checkpoints, matched training and
  holdout archives, loaded Python/Cython implementation, CSV, and summary,
  together with runtime environment, protocol, and coverage metadata.

The primary descriptive statistic is
$\Delta NLL_{carry}=NLL_{no-carry}-NLL_{carry}$.  A positive value means the
human action received higher likelihood with carry for that evaluation set.
Overall and actually-carried rows are reported separately because a carry-on
run can legitimately fall back to a fresh root.

## Claim boundaries

- The comparison isolates retaining the searched human branch under a fixed
  checkpoint and fixed recorded inputs; it does not compare Thinker with DQN,
  Dreamer, or another model family.
- A positive held-out $\Delta NLL_{carry}$ supports the claim that carried tree
  context improves human-action prediction for this subject/session.  It does
  not by itself establish that human participants use the same search
  algorithm or neural representation.
- The `root_carried=true` result is a mechanism-conditioned descriptive subset,
  not an independently randomized group.  The evaluator intentionally reports
  no $p$-values.
- Session 4 is never used by the behavioral imitation sampler.  Repeated model
  selection on this output would nevertheless turn it into a validation set
  and should be disclosed or avoided.
- The prepared compute07 jobs use fresh scratch initialization.  Replay-buffer
  state is not checkpointed, so a later training resume is not claimed to be
  bitwise identical to an uninterrupted run even though stale completion
  markers and incompatible imitation protocols are rejected.
