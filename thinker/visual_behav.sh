#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <sub> <game> <xpid>" >&2
  exit 1
fi

sub="$1"
game="$2"
xpid="$3"

base_dir="../behavioral_data_block/sub_${sub}/game_${game}"

if [ ! -d "$base_dir" ]; then
  echo "Error: base directory not found: $base_dir" >&2
  exit 1
fi

mapfile -d '' data_files < <(find "$base_dir" -type f -name "*.npz" -print0 | sort -z)

if [ "${#data_files[@]}" -eq 0 ]; then
  echo "Error: no .npz files found under $base_dir" >&2
  exit 1
fi

for data_path in "${data_files[@]}"; do
  echo "Running: visual_behav.py --data $data_path --xpid $xpid"
  python visual_behav.py --data "$data_path" --xpid "$xpid"
done
