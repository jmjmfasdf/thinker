# All-checkpoint, all-session visualization and extended cache

Campaign: `allcheckpoints_allsessions_5m_20260923`.

- Enduro/Pong, subjects 1–6, four leave-session checkpoints per subject/game.
- Each checkpoint replays **all four sessions of the same subject/game**:
  48 captures, 132 distinct human source runs, 528 checkpoint/source pairs.
- Expected complete real transitions across checkpoint replications: 3,798,796.
- The three training sessions and one held-out session are explicitly labelled
  in each cache's `checkpoint_binding.json`. Repeated human trajectories are
  not independent behavioral samples, and this collection is not all held-out.

## Storage

Retained fields use **Blosc Zstd level 9 + bitshuffle**, without numeric
quantization. Only `predicted_xs` and its companion validity field are omitted
from the stored event payload. The model still computes and uses XS, and the
recorder derives the uint8 imagined image **before** omitting the float32 stack.
Real/imagined images, SR/VP latents, tree states, VOC/control/action fields remain.
The omitted stack cannot be reconstructed losslessly from a display image.

Both settings are bound to capture semantics and episode metadata. Validation
rejects mismatched field/compression profiles. Default recorder settings stay
unchanged for other callers; this campaign explicitly passes
`--compression-level 9 --omit-predicted-xs --images all`.

Old `logs/visual_behav/{full,heldout_5m_20260921,pbs,pilot}` directories are deleted
only through the user-requested `purge-old` step. Small provenance/log files and
the deletion receipt are retained under the new output root's `audit/`. Training
checkpoints, source behavioral NPZ files, old numeric caches, and previously
frozen source snapshots are not deleted. Old caches therefore retain their
numeric data, but their old local visualization references are no longer live.

## Locations

Visualization:

```text
/home/jeongmin/thinker-dynamic-imitation/logs/visual_behav/
  allcheckpoints_allsessions_5m_20260923/captures/
    {enduro,pong}-sub{001..006}-leave{01..04}-all/
```

Numeric caches and operations:

```text
/home/jeongmin/thinker-dynamic-imitation/research_script/outputs/
  allcheckpoints_allsessions_5m_20260923/
    00_dynamic_cache/checkpoint_leave{01..04}/sub{001..006}_game{0,1}/
    campaign.json
    submission.json
    reports/task-NN.{running,visual.completed,completed}.json
    pbs/
    subjects/subSSS_gameG.json
    source/
    audit/
    COMPLETED.json  # only after all 48 visualization+cache tasks pass
```

Each job runs visualization → capture integrity/identity validation → cache →
cache coverage validation. Image and latent summary options are enabled; raw
XS summary fields are unavailable, not fabricated from the uint8 image. Image
prediction errors use `imagined_image` versus `real_image` with target-time
availability and action-prefix matching. No 02/03/04 or group analyses run.

## Execution and verification

48 CPU PBS jobs, 8 CPUs/24 GiB/168 hours each, explicitly assigned 12 jobs per
slave5–8. At most three such jobs fit simultaneously on each 28-CPU node. The
frozen legacy runtime handles Enduro plus Pong sub001/leave04; the independently
verified GroupNorm runtime handles the other 23 Pong checkpoints. Only recorder
and native I/O source files are overlaid; model/actor/cenv code is preserved.

Source checksums and the sealed tarball runtime are checked on every node.
Storage is checked before submission and during visualization/cache subprocesses;
below 80 GiB remaining, only the job's own subprocess group is stopped, retaining
partial outputs. A cache does not run after failed visualization. No completed
campaign is reported until every checkpoint-specific cache passes validation.

Current status:

```bash
/usr/bin/python3 -B /home/jeongmin/thinker-dynamic-imitation/research_script/all_session_visual_cache.py status
qstat -u jeongmin -anw
```

The `prepare`, `purge-old`, and `submit` commands are intentionally separate.
`prepare` requires the old reference manifests and refuses an existing campaign;
it is not a resume command. `submit` refuses an existing submission ledger.
Use the frozen sources when diagnosing an already submitted job, and do not
overwrite an incomplete numeric cache without reviewing its failure first.
