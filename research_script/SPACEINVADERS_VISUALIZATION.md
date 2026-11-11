# Space Invaders all-session visualization

Campaign: `logs/visual_behav/spaceinvaders_allcheckpoints_allsessions_5m_20260924`.

- Visualization only: no numeric cache or research-analysis job is scheduled.
- Currently published 5M checkpoints: subjects 001–005, leave-session 1–4,
  20 checkpoint-specific captures. Subject 006 has data but no published
  checkpoint directories at preparation time.
- Each checkpoint replays all 11 source runs across its subject's four sessions:
  55 unique human trajectories, 220 checkpoint/run combinations. These are not
  independent human samples; three sessions are training and one is held out.
- Retained arrays: Blosc Zstd level 9 + bitshuffle, no quantization. Only the
  stored `predicted_xs` stack and its validity field are omitted. The model still
  uses its XS and the recorder retains imagined/real images, latents, VOC,
  controls, and tree fields. Enduro/Pong outputs are untouched.

## Runtime and safeguards

Runtime lineage is frozen in `source/provenance.json` with file checksums.
Sub001/leave02 uses the previously validated legacy runtime. Eighteen captures
use the previously validated GroupNorm runtime. Sub001/leave01 uses the legacy
model with its exact published depth-support actor, recovered from the two
known source versions and accepted only if its full source SHA256 matches the
training manifest. Replay instrumentation and evaluation utilities are adapted
code, not claimed to be the complete original training source snapshot.

Every checkpoint's actor/model/config hashes, subject/game identity, selected
NPZ hashes and storage settings are checked before execution. The native
recorder strictly loads model weights and validates complete capture coverage.
Only successful capture validation produces `reports/task-NN.completed.json`.
All 20 proofs are required for the campaign's `COMPLETED.json`.

PBS requests explicitly select slave5–8, 8 CPUs, 24 GiB and 168 hours. Each job
uses three replay workers, two Torch threads each, and the sealed tarball Python
environment inside Singularity. The runtime and source checksums are verified
on each node. A job stops its own subprocess group if shared free space drops
below 80 GiB; it does not delete partial or completed captures.

## Files

- `campaign.json`: input identities, source hashes, task/node mapping.
- `submission.json`: durable PBS submission receipts; duplicate submissions
  are refused.
- `captures/spaceinvaders-subSSS-leaveLL-all/`: native Zarr capture output.
- `reports/`: preflight, progress and completion records.
- `pbs/`: live per-job logs.
- `source/`: immutable executed code and provenance.

The root completion marker refers to visualization only, **not cache creation**.
