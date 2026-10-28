#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ ! -x .venv-catboost-rs-v9/bin/python ]]; then
  uv venv --python 3.12 .venv-catboost-rs-v9
fi
uv pip install --python .venv-catboost-rs-v9/bin/python \
  numpy==2.3.5 catboost==1.2.10 catboost-rs-rocm==0.1.0
rocm_dir="${ROCM_PATH:-/opt/rocm}"
hiprtc_path="$(find -L "$rocm_dir" -name 'libhiprtc.so.7' -print -quit 2>/dev/null || true)"
hip_path="$(find -L "$rocm_dir" -name 'libamdhip64.so.7' -print -quit 2>/dev/null || true)"
if [[ -z "$hiprtc_path" || -z "$hip_path" ]]; then
  printf 'ROCm probe requires system libhiprtc.so.7 and libamdhip64.so.7 under %s. No driver changes made.\n' "$rocm_dir" >&2
  exit 1
fi
mkdir -p eda_results
ROCM_PATH="$rocm_dir" HIP_PATH="$rocm_dir" \
  LD_PRELOAD="$hiprtc_path:$hip_path${LD_PRELOAD:+:$LD_PRELOAD}" \
  .venv-catboost-rs-v9/bin/python tools/probe_catboost_rs_v9.py \
    --expect rocm \
    --out eda_results/catboost_rs_probe_v9.json
