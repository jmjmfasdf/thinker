# Enduro VoC-v15 preregistration design shell

> Status: superseded design shell only; not a frozen preregistration and not
> launch authority.  The authoritative successor is
> `VOC_V15_HALF_SQUARED_Q_300K_ACCEPTANCE.md`.  This shell remains only as the
> audited record of alternatives considered before root selected the single
> half-squared-Q delta.  It authorizes no implementation, snapshot, wire,
> qualification, primary, or fixed evaluation.

## Permanent v14 qualification failure

The sole v14 qualification
`enduro-voc-v14-sealed-eps25-seed1-qual-fresh-100k` is a permanent numeric
qualification failure.  It ran once from immutable snapshot
`/tmp/di-voc-v14-sealed-eps25-final-eKgdrk`; resume, retry, replacement seed,
v14 primary, and v14 fixed evaluation are forbidden.

The driver exited zero after `1918.067587` seconds and all v14 mechanism,
schema-7 seal, strict W&B completion, artifact, and cleanup validators passed.
The frozen canonical population is the 63 complete rows satisfying
`70000 < real_step <= 100000`, from real step 70032 through 99376.  The final
real-step 100112 row is overshoot and is excluded.  The sole failed gate is
the pooled held-out EMA selected-action TD RMSE:

```text
sqrt(sum(row_holdout_count * row_ema_selected_action_td_rmse^2) / 7703)
    = 0.5247954789453232
```

This exceeds the unchanged preregistered ceiling `0.5`.  The online-Q
companion RMSE is `0.5248123566562355`.  This is a numeric qualification
failure, not an implementation P0/P1, and passing mechanism evidence cannot
rescue it.

All other inherited 100k gates passed under the frozen algebra:

- teacher gap `0.27034248188280874`, student gap
  `0.26493562794609854`, retention `0.9799999840977489`, and signed margin
  `0.26223210510810246`;
- all three fixed windows had positive student gap and positive signed
  margin;
- all 59 trailing-five endpoints had both sign denominators, none was
  negative, and the maximum negative run was zero;
- train CONTINUE/STOP support was `28219/25794`, held-out CONTINUE/STOP
  support was `4016/3687`, and online-versus-EMA non-tie delta-sign agreement
  was `46420/61714 = 0.7521794082380011`; the separate EMA-Q versus exact
  projected-gate sign diagnostic was `61715/61715`;
- wrong-CONTINUE saturation was `0/28368`, wrong-STOP saturation was
  `0/25644`, and the report-only forced-stop diagnostic was
  `121/29602 = 0.004087561651239781`; and
- actor, online-Q, gate, and ModelNet AMP-skip and non-finite counters were
  all zero.

The late-window calibration error rules out two tempting post-hoc changes.
A single global held-out bias correction would have an optimistic RMSE lower
bound `0.5245088518192085`, still above `0.5`; changing only that bias is not a
credible v15 mechanism.  Changing only EMA lag is also unsupported because
the online-Q companion is slightly worse than the EMA value.  These are
diagnostic observations, not authority to tune on held-out rows.

## Immutable v14 failure artifacts

The failed run directory is
`/tmp/di-voc-v14-sealed-eps25-final-eKgdrk/runs/enduro-voc-v14-sealed-eps25-seed1-qual-fresh-100k`.
Its exact 14 regular-file SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `ckp_actor.tar` | `a86094cc3131bbe55bd5ac5995203cf8f3830ab36fbaf60e630306b0fe1d107c` |
| `ckp_actor.tar_step_480` | `95db486d6ff0f978e235bb762174bf63f0ef37a0db7c90c409bdec4500747173` |
| `ckp_actor.tar_step_100112` | `76ea5211bf0a6b5ba0070f29b751241769ebc07f3234069fa02a58c36fab3a7a` |
| `ckp_model.tar` | `6d6c2ff3ba5335337476998eb13be6ea80e9962d5ffabc67ade3020cb362af7a` |
| `ckp_model.tar_step_10000` | `f0cbcb5b5ae0a0fab2c4769637d3c32af37d4a481e054c10e32ffe4009152471` |
| `ckp_model.tar_step_100112` | `6d6c2ff3ba5335337476998eb13be6ea80e9962d5ffabc67ade3020cb362af7a` |
| `config_c.yaml` | `97d6a78d78d0d49dbd3e038fa77f268c29d4b21244c98d345060505f70c40350` |
| `finish` | `5ea6592e5f57cafb908902eed9decd7535e78d4e9b9fa33c23f7bf00b16f3b4e` |
| `logs.csv` | `6d72052279a59983d102bed3db767b3792b115d221f155cdd10d4dc29bbe85a8` |
| `logs_model.csv` | `b38dfd254fab250394ca463c4371c573b8c37f332e5f5f0b43c6d9b5fbfc3f64` |
| `meta.json` | `1799dc6bc2fb9be4acddffaea4bbd7c1b09d690ed2d04b92be7315088e0c796f` |
| `meta_model.json` | `04bcd2c1eb1738fd4a513b2aa9ef7a76943e84ea70e645dd1bf73bcf5eb08b47` |
| `out.log` | `67015c9a0896551cdbf19aa49e5d673837d947e50df415bbecdd835a1a8c4006` |
| `out_model.log` | `8e4a649f2b936ee549633001cc49641e01365b0199085d424cac63f5c66313d6` |

Two independent post-exit auditors reported stable aggregate run-tree digests
`61d7f8deec7b3fb38fb5a842bf13a1b465223d3b513e3baa7e3ca49cb36e2cb4`
and, for a separately canonicalized path+size+SHA scan,
`c4440e192aab746477d3d803d303ce37ec9d676bc78ba41e4c6d11b1331a5fbb`.
The individual file hashes above are authoritative when aggregate
canonicalizations differ.

The one launch runtime is `/tmp/v14qual100-hZGppv`:

- `launch_provenance.txt` SHA-256
  `0e9f4ad97ea5774ccc579da43118546b1c837d6e21ceaa3b9be8d7d330af5b54`;
- `driver.log` SHA-256
  `04b18b4110c247425cd6234ebc097eb6efabd99a140ae3865f156615db072833`;
- `driver.exit_code` SHA-256
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`,
  containing exactly `0` plus a trailing LF;
- snapshot source manifest SHA-256
  `4fa507706250f349acfb6034453f1e8c70681c54e9e52718d848cd81755dbfea`
  with 1063 entries; and
- data manifest SHA-256
  `23c32c10236c15a53f27ecbe9e49dc2edbddfc5b132dcece22700ad07ef68343`
  with 11 entries.

Terminal validation resolved the exact 229-key config/actor/model identity to
complete-surface SHA-256
`ad87841c8c5dd18e9b5291b35eae3e473bfc4f22e0a5532872c5cf726d2c31b9`
and the unchanged 209-key v12 projection to
`bd386ad1c78f030409562dea5c34d8866cc128cf99b7f39938f6a090f5987407`.
Actor policy version/publication count was 204 with a 205-event history and
terminal ack `1/1`; state SHA-256 was
`330fa1dee9ad167c46acc1c81eada7852daf3677fd1e0909f72d9a6eff2425fc`
and history SHA-256 was
`a7d1e4f6d515d71cdfa9b936d0fece6fd2f300e71fc077666d8070aca757cde0`.
Model input sealed once at terminal processed/real step 100112, took the
allowed single drain from pre-real 100096 and pre m/p counts 744 to final m/p
counts 745, and recorded zero late writes and aborts.  Strict W&B request/ack,
private-marker cleanup, public finish, source/data posthash, Ray/process, and
GPU cleanup all passed.

## Inherited v14 baseline that v15 must not rewrite

Any v15 option inherits the complete v14 mechanism and all v13 Changes A-C
plus v14 Change D unless the final preregistration explicitly names one
prospective replacement:

- soft training epsilon remains `voc_train_epsilon=0.02`;
- executed gate epsilon remains `voc_gate_execution_epsilon=0.25`, with the
  existing epsilon-greedy action likelihood, V-trace, stored behavior, and
  joint-policy entropy semantics;
- the strict actor-policy bundle/version barrier, bundle schema 1, 120-second
  timeout, no-Ray-retry topology, terminal history, state digests, and W&B
  two-phase completion remain unchanged;
- main actor AMP initial scale remains 32 and all actor/Q/gate/model skip gates
  remain exactly zero;
- exact EMA-to-gate projection and all Q/EMA/projection counter and state
  invariants remain unchanged; and
- the schema-7 ModelBuffer last-write acknowledgement, terminal input seal,
  claim linearization, zero-or-one fresh terminal drain, durable ModelNet save,
  exact ten-field seal evidence, and no-post-terminal-action contract remain
  unchanged.

All v14 populations, sufficient-statistic pooling, overshoot exclusion,
malformed-input rules, windows, support floors, calibration algebra, safety
gates, and thresholds remain unchanged.  In particular, v15 may not relax,
reinterpret, round, or replace the held-out EMA selected-action TD RMSE ceiling
of `0.5`, and held-out rows may not become training input.

## Single-delta precedent

The hashed lineage permits a future version to make a prospective,
mechanically closed change while preserving the prior failure and all
unmentioned gates.  V9 added one gate-only alignment objective; v11 added one
exact projection mechanism; v12 changed one executed gate distribution; and
v14 added one terminal model-input seal.  V10 changed only the preregistered
population/power plan and explicitly left learning unchanged.  V13 is the
exception that enumerated three independent changes A-C before implementation.

The closest precedent for v15 is therefore one train-only calibration
mechanism, stated before implementation, with an exact identity/schema bump,
negative tests proving held-out exclusion, and no threshold change.  A bundle
of result-informed hyperparameter edits is not an acceptable “single delta.”

## Mechanism alternatives awaiting root choice

No option below is selected.  Exact names and values are intentionally absent
until the root decision.

| Option | Prospective single delta | What remains invariant | Main audit risk |
| --- | --- | --- | --- |
| A: state-conditioned common-value residual | Add one train-only residual calibrator whose scalar output is added equally to CONTINUE and STOP online/EMA Q values.  Train it only on the existing non-holdout Q population and include it in the existing successful Q-to-EMA transaction. | The Q difference, gate sign, epsilon-greedy behavior, action likelihood, and exact gate projection remain unchanged by the equal action shift; all v14 thresholds remain unchanged. | Requires a new parameter/state/optimizer or carefully shared optimizer evidence, EMA lifecycle, checkpoint validation, finite/zero-init rules, and proof that no held-out row contributes gradient. |
| B: train-only squared Q regression | Replace only the existing online VoC selected-action Smooth-L1/Huber loss (default transition `beta=1`, summed on `q_train_valid`) with an exactly specified squared TD-error loss on that same non-holdout mask and with an exactly specified reduction; EMA and gate continue to follow the online head as in v14. | Data split, target, architecture, execution, barrier, seal, and thresholds remain unchanged. | It directly optimizes the squared-error family measured by RMSE, but also changes learned Q differences and outlier influence and can regress the already-passing sign/gap/margin and safety gates. |
| C: isolated VoC-Q optimizer rate | Introduce one exact VoC-Q learning-rate field and separate only the existing VoC-Q optimizer rate from the main actor rate; loss, target, features, EMA, and gate remain unchanged. | No new loss or parameter and no held-out training; v14 execution/barrier/seal contracts remain unchanged. | This is the smallest code/config delta but the least targeted; it can alter both Q difference and calibration, and a single 100k result provides no prospective value-selection authority. |

The observed evidence disfavors an EMA-tau-only option and a global-bias-only
option, so neither should be promoted merely because it is easy to implement.
Root may instead reject all three alternatives and request another prospective
single mechanism; that decision must precede code and a final preregistration.

## Version, schema, and stage identity choices

The final preregistration must choose one of these closed identity patterns:

1. **Gate schema 8, recommended for a behavior- or Q-bearing change.**  Set
   `voc_gate_policy_schema_version=8`; every actor-policy bundle and ack uses
   `gate_schema=8`; keep bundle schema 1 and model-input-seal schema 1; add the
   chosen mechanism's exact identity field or fields; validate the new complete
   config/actor/model surface and preserve the 209-key v12 projection hash.
2. **Gate schema 7 plus a separate v15 mechanism schema.**  Keep
   `gate_schema=7` and add one exact non-bool integer mechanism-schema field.
   This minimizes bundle changes but requires public/smoke/fixed validators to
   distinguish v14 and v15 from the full surface and stage tuple; silent v14
   acceptance must fail closed.
3. **Gate schema 7 with only an existing scalar changed.**  This is suitable
   only if root selects an existing-field-only option and explicitly accepts
   the weaker protocol boundary.  Exact stage tuples and complete-surface
   hashes must still distinguish v15.  It is not appropriate for a new head,
   loss family, optimizer topology, or persisted state.

The mechanism tag remains unresolved.  The final closed xpid candidates are
therefore placeholders, not launchable strings:

```text
enduro-voc-v15-<mechanism-tag>-sentinel-wire1200
enduro-voc-v15-<mechanism-tag>-seed1-qual-fresh-100k
enduro-voc-v15-<mechanism-tag>-seed5-strict-fresh-300k
```

The likely closed stage tuples retain v14's seed and stage roles: seed 1 for
the nonqualifying wire, seed 1 for the separate fresh qualification, and the
still-unused seed 5 for the primary.  Root may instead reserve seed 5 and
choose a new unused primary seed, but that must be prospective and frozen with
the exact xpid before implementation.  Wire remains W&B-disabled 1200-step;
qualification remains fresh W&B-enabled 100k with warm-up 10000 and unroll
201; primary remains one fresh W&B-enabled 300k run.  There is no retry,
resume, fallback seed, or promotion of the failed v14 artifact.

## Required content before this shell may become a preregistration

The final v15 document must replace every unresolved choice above and bind:

- exactly one mechanism, its train-only population, algebra, initialization,
  gradient ownership, update ordering, optimizer/scheduler/scaler behavior,
  checkpoint evidence, and failure semantics;
- exact config/checkpoint/public field names, strict Python types and values,
  complete-surface key count/hash rules, schema/bundle/ack values, and legacy
  schema return-shape preservation;
- exact three xpids and stage tuples, seed decision, runtime overrides for
  public/fixed evaluation after immutable validation, and a v15-primary-only
  fixed profile;
- unit, adversarial, resume, public, smoke, fixed, real CLI, wire, artifact,
  W&B, no-retry, and independent immutable-snapshot audit matrices; and
- verbatim inheritance of every v14 numeric gate, including the held-out EMA
  selected-action TD RMSE ceiling `0.5`.

Until root freezes those choices, this document remains a design aid only and
must not be used to edit source/public/fixed files or launch any v15 stage.
