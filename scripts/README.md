# Operational utilities

Scientific entry points are under `research_script/00_*.py` through `14_*.py`.
These tools manage jobs, metrics or data layout; none runs on import.

- `training/prepare_hxnorm_grid.py`: prepare an h/x-normalized grid from an existing campaign.
- `maintenance/recover_wandb_realstep.py`: recover metrics from existing run CSVs.
- `maintenance/consolidate_resumed_runs.py`: the preserved resumed-run consolidation protocol.
- `maintenance/merge_dynamic_caches.py`: the preserved 72-checkpoint cache-collection protocol.

The last two retain their historical campaign presets: inspect their options
and source before using them for a different campaign. Existing running jobs,
W&B watchers and downloads continue to use their unchanged frozen sources.

Replay capture itself remains implemented in `thinker/visual_behav.py`.
Completed visualization data are under `logs/visualization/hxln_10m/subXXX/game/`;
its campaign manifests, frozen capture runners, checkpoints and runtime sources
are under `logs/visualization/campaigns/`. Use `logs/visualization/catalog.json`
for the canonical paths. Research results remain under `research_script/outputs/NN/`.
