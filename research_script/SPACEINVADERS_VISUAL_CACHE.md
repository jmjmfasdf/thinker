# Space Invaders completion chain — 2026-09-28

The 24 tasks cover six subjects and four 5M leave-session checkpoints each.
Every checkpoint uses all 11 source runs across all four sessions of its own
subject/game (66 unique runs, 264 checkpoint/run combinations).

- Tasks 0–3: sub006 leave1–4, visualization → native capture validation →
  extended dynamic cache → cache validation.
- Tasks 4–23: sub001–005 leave1–4, validate and reuse the completed 2026-09-24
  captures, then generate and validate caches. They never replay those captures.
- No downstream model fitting, research analysis, or group analysis runs.

## Locations

Operations, immutable executed sources, and all 24 new caches:

`research_script/outputs/spaceinvaders_allcheckpoints_allsessions_5m_20260928/`

Each cache is in `00_dynamic_cache/checkpoint_leaveLL/subSSS_game2/`.
New sub006 captures are in:

`logs/visual_behav/spaceinvaders_allcheckpoints_allsessions_5m_20260928/captures/`

The 20 reused captures remain read-only in the separate 2026-09-24 campaign.
Enduro/Pong results and all previously executed snapshots are unchanged.

## Storage and runtime

- Retained native arrays: lossless Blosc Zstd9 + bitshuffle.
- Omit only stored `predicted_xs` and its validity field. Keep real/imagined
  images, SR/VP latents, VOC, controls, carry and tree fields.
- Enable prediction-image and latent summaries, bounded batches of 64.
  Unavailable XS remains missing; it is not fabricated from display images.
- Freeze the same adapter-v3 cache implementation used for Enduro/Pong.
- New sub006 runtime: published GroupNorm actor/model/util hash match;
  established instrumented replay implementation, sealed Python tarball,
  Singularity CPU environment.

All tasks request one explicitly assigned slave5–8 node, 8 CPUs, 24 GiB and
168 hours. Sub006 tasks are submitted first so visualization overlaps with
cache-only tasks. At most 12 such jobs fit simultaneously on these four nodes.
Free space below 80 GiB stops only a job's own child process; no old data is
deleted. Existing cache output and duplicate submission are refused.

## Completion contract

Only successful visualization and capture integrity validation allow cache
generation. Cache validation checks subject/game/action count, all 11 source
runs, native real-transition count, unique real-step IDs, STOP/forced-stop
coverage, total additional computation count, and required sidecar files.

`reports/task-NN.completed.json` and the cache's `checkpoint_binding.json`
are written only after those checks. Bindings explicitly label training versus
held-out sessions. Replicated trajectories are not independent human samples.

`subjects/subSSS_game2.json` requires all four checkpoint caches for a subject;
the root `COMPLETED.json` requires all 24 caches. Submission receipts are in
`submission.json`; stage updates and failure details are in `reports/`; live
logs are in `pbs/`.
