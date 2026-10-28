#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libsuitesparse-dev pkg-config
python -m pip install -r requirements_research_v9.txt
if ! python -c 'import torch' >/dev/null 2>&1; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
python tools/run_research_v9.py --doctor --task-type CPU --torch-device auto
