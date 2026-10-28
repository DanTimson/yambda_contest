#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p logs eda_results
stamp="$(date +%Y%m%d_%H%M%S)"
log_path="$repo_dir/logs/research_v9_500m_${stamp}.log"
nohup flock -n "$repo_dir/eda_results/.research_v9_500m.lock" \
  env POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  python -u tools/run_research_v9.py \
    --size 500m \
    --data "data/500m" \
    --cache-dir "data/research_cache_v9_500m" \
    --out "eda_results/research_v9_500m" \
    --task-type CPU \
    --torch-device auto \
    --threads 4 \
    --cohort-users 5000 \
    --train-users 5000 \
    --older-train-users 2500 \
    --eval-users 5000 \
    --trees 600 1200 2000 \
    --epochs 5 \
  > "$log_path" 2>&1 < /dev/null &
run_pid=$!
printf '%s\n' "$run_pid" > "$repo_dir/logs/research_v9_500m.pid"
printf 'PID: %s\nLog: %s\n' "$run_pid" "$log_path"
printf 'Monitor: tail -f "%s"\n' "$log_path"
