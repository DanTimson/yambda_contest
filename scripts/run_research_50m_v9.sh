#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p logs eda_results
stamp="$(date +%Y%m%d_%H%M%S)"
log_path="$repo_dir/logs/research_v9_50m_${stamp}.log"
nohup flock -n "$repo_dir/eda_results/.research_v9_50m.lock" \
  env POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  python -u tools/run_research_v9.py \
    --size 50m \
    --data "data/50m" \
    --cache-dir "data/research_cache_v9_50m" \
    --out "eda_results/research_v9_50m" \
    --task-type CPU \
    --torch-device auto \
    --threads 4 \
    --cohort-users 2000 \
    --train-users 1500 \
    --older-train-users 750 \
    --eval-users 2000 \
    --trees 300 600 1200 \
    --epochs 5 \
  > "$log_path" 2>&1 < /dev/null &
run_pid=$!
printf '%s\n' "$run_pid" > "$repo_dir/logs/research_v9_50m.pid"
printf 'PID: %s\nLog: %s\n' "$run_pid" "$log_path"
printf 'Monitor: tail -f "%s"\n' "$log_path"
