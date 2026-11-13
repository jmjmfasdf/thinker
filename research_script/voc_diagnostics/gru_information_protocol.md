# GRU / VOC information audit

Registered 2026-09-25, before collecting probe results.

Question: Does the detached shared representation deprive the VOC head of
search-count/depth information, and does a direct metadata path improve
prediction of the actual reward benefit of additional computation?

Scope: final 100k Enduro/Pong projection checkpoints from the completed
reward-learning pilot. This is a mechanistic diagnostic of those checkpoints,
not a claim about all subjects or a new whole-policy reward experiment.

1. Verify the loaded tree encoder's normalization modules. On real actor
inputs, backpropagate a nonzero VOC regression loss; record parameter gradients
separately for the head and GRU. As positive controls, evaluate the same head
without detaching its input, and the real policy's supervised loss. Never
update the checkpoint.
2. Collect actual online policy SEARCH rows and separately forced-budget
SEARCH rows (caps 1,4,8,16,30, scheduled independently of rewards). Save the
input token, GRU output, full VOC feature vector, counts, depths, policies and
environment-stream identity. Forced rows are a coverage intervention, not
samples from the trained stopping policy.
3. Predict log1p(count/depth) from frozen representations with ridge probes.
Use outer splits by environment stream and select regularization solely
within training streams. Report MAE in original count units and rounded exact
accuracy, including depth/count bins. Compare GRU, full VOC features, constant
prediction and explicit counter features. This measures decodability, not
whether the trained head uses the information or whether planning is useful.
4. If the information is decodable, examine a direct-path prediction test
against actual paired STOP/CONTINUE returns, including nonzero search counts.
Keep continuation checkpoints frozen. Compare the same regularized readout
on frozen features with and without explicit count/depth. Hold out whole
environment streams, keep paired arms and repetitions together, and report
uncertainty at stream level. Neither teacher Q nor human imitation labels
are ground truth for computation utility.

Reward sampling: two independently seeded batches of eight environment
streams, each examined at counts 0,1,3,6 after a 128-action policy prefix;
two paired repetitions per state. Thus 64 states / 128 pairs per game.
The remaining task-return horizon is 1536 real decisions, or critic terminal
(including episodic life) if earlier, with a bounded omitted discounted tail.
Only costs remaining in the CURRENT search stage are subtracted, matching
the existing critic. A repeated STOP arm verifies reconstruction in each
seed/count cell. All repetitions and counts from a stream stay in one fold.
Counts are homogeneous within an evaluation batch: a mixed-count smoke test
changed model microbatch shapes after early interventions and produced tiny
floating differences (~1e-7) in later entries. Homogeneous-count batches retain
the exact entry-hash requirement; mixed-count results are not used.
Real action sampling uses common uniforms keyed by seed/stream/real decision
and repetition, independent of preceding search count. Imagined actions retain
search-index keys. This variance-reduction correction is bound to
`reward_source_v2`; preliminary `reward/` artifacts are excluded.

Primary readout comparison: full existing VOC features versus those features
plus log1p(count/depth), with the same nested ridge procedure. GRU-only and
metadata-only readouts are secondary. Bootstrap intervals resample whole
environment streams, conditional on the out-of-fold fitted predictions; they
do not include training-seed uncertainty or full model-refitting uncertainty.

Interpretation: absent gradients prove the training-path separation. Strong
decoding contradicts a simple loss-of-count explanation. A replicated
held-out reward-prediction gain from direct metadata would support a readout
bottleneck. A negative or uncertain gain is not proof of equivalence. Full
causal attribution to detached encoder training requires an additional
matched end-to-end retraining intervention; do not infer it from probes.

Existing checkpoints/caches remain read-only. New artifacts record source,
checkpoint hashes, seeds, resource use and all sampling parameters. Probe
fitting is logged to W&B. GPU work uses SLURM allocations.

Exploratory follow-up after inspecting linear decoding: a 100→64→2 GELU
counter readout tests whether the forced-range linear errors reflect nonlinear
encoding rather than absence of information. The same four outer stream folds
are used; epoch selection uses one inner held-stream split only. Fixed AdamW
lr .001, decay .0001, batch 512, maximum 60 epochs and patience 10; refit on
all outer training streams for the selected epoch count. This is explicitly
post-linear exploratory analysis and does not alter the primary reward test.

Exploratory forward sensitivity reconstructs the current GRU update from its
saved token and preceding hidden state, then changes only log1p(count) or
log1p(depth) by the increment corresponding to +1. The existing online VOC
weights are unchanged. Hidden states and original Q differences must reconstruct
within floating-point tolerance. These artificial tokens may conflict with
other metadata; this is evidence of channel dependence, not an intervention
on actual search or its value. It does not alter the primary reward test.

Implementation correction after Pong collection, before reward results: the
installed sklearn squeezes a single target's predictions to (N,), whereas
the shared nested-ridge helper stores targets as (N, 1). The original reward
fit aborted at outer-fold assignment. Explicitly reshape both inner- and
outer-fold predictions to (N, target_count), also preventing accidental
cross-row broadcasting in inner scoring. Single-target vector/column tests
must agree with a duplicated two-target reference; all five probe tests pass.
Keep collected pairs and the running frozen collection source unchanged.
Perform fit-only recovery from copied, hashed corrected analysis files and
record their hashes plus the input-pair hash in each reward report. Fold
grouping, alpha candidates, readouts, bootstrap, targets and acceptance
criteria are unchanged. Failed initial analysis logs remain available.

Validation correction before final reporting: the original nonlinear probe
recomputed GroupKFold assignments. Its forced-coverage assignments differed
from the saved linear assignments despite identical rows and stream IDs;
the mismatch was concentrated in equal-size groups. Do not assume identical
library tie handling. Refit the exploratory nonlinear probe using the saved
linear outer-fold IDs explicitly, preserving all original outputs in
`nonlinear_superseded_unmatched_folds/`. Keep architecture, optimizer, epoch
selection and input data fixed. The final validation still requires exact
outer-fold identity. This correction does not modify the reward comparison.
