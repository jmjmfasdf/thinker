#!/usr/bin/env bash
set -euo pipefail

keys=(
  tree_reps
  thinker_action
  human_action
  status
  actor_policy
  step_times
  env_return
  cur_rewards
  im_vp_vectors
  im_vectors
)

shopt -s nullglob

for npy_path in test/*/video_stat_*.npy; do
  case "$npy_path" in
    test/sub001/*)
      continue
      ;;
  esac

  python filter_npy_keys.py "$npy_path" "${keys[@]}"
done
