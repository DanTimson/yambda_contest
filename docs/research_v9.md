# Yambda: local 50M → 500M research

Run all commands in Ubuntu/WSL Bash in the activated yambda_contest Python environment (Python 3.11–3.13). No remote repository writes are performed. This patch only adds new files; the previous submission notebook and results are untouched.

Install once:

```bash
cd "$HOME/yambda_contest" && bash scripts/install_research_v9.sh
```

Existing working PyTorch/ROCm is retained. If PyTorch is missing, installation uses its CPU wheel, without changing drivers. The doctor prints the actual PyTorch device and CatBoost device. CatBoost uses CPU on AMD; its GPU implementation requires NVIDIA CUDA. To require working PyTorch GPU explicitly:

```bash
cd "$HOME/yambda_contest" && python tools/run_research_v9.py --doctor --task-type CPU --torch-device gpu
```

Launch 50M outside the VSCode terminal, from a separate Ubuntu terminal:

```bash
cd "$HOME/yambda_contest" && bash scripts/run_research_50m_v9.sh
```

The script prints the PID and a unique log path. It runs under nohup and flock so closing VSCode does not terminate the job and two jobs for the same experiment cannot run together. Closing WSL or rebooting still terminates it; rerunning reuses completed tables and models. A partially trained neural model restarts that model; committed models and feature tables resume. Keep the activated Python environment when launching.

Follow the latest 50M log:

```bash
cd "$HOME/yambda_contest" && tail -f "$(ls -t logs/research_v9_50m_*.log | head -n 1)"
```

After reviewing 50M results, run 500M:

```bash
cd "$HOME/yambda_contest" && bash scripts/run_research_500m_v9.sh
```

Return `eda_results/research_v9_50m/research_v9_results.zip` first. For 500M, return the analogous ZIP in `research_v9_500m`. Open `research_v9/report.html` inside each output directory to view the generated graphs and tables. CLI and local notebook use the same visible implementation; do not run both simultaneously.

Sources include temporal popularity, artists, like-trained ALS with like/listen profiles, BM25, listen-trained ALS with like/listen profiles, cosine kNN, BPR, EASE, official SANSA, Mult-VAE, SASRec, and two towers with HNSW. Neural candidates use IDs/history; towers additionally use artist metadata, not audio. Compact neural architectures and five epochs test the model families, not the best published configurations.

There are five capped candidate pools and eleven ranking experiments, comparing Logloss/QuerySoftMax/YetiRank Classic, extra features, no-neural/no-listen-space bundles, and a second past training window. Reports contain per-source candidate coverage, additional targets beyond the old pool, paired bootstrap differences, and contest-UID overlap. No guarantee of improvement is assumed. 50M pilot settings: 2000 evaluation / 1500 training / 750 additional-window users, 300/600/1200 trees; 500M: 5000/5000/2500, 600/1200/2000 trees. Within each run comparisons use the same data/cohort; the two scales are not an isolated dataset-size ablation.

The official first-100-event target is retained, including repeated likes. The teaching baseline globally keeps only first likes and suggests filtering seen tracks; that is a different target definition, so its absolute scores are not directly comparable. Report full100 sample counts and use the official_overlap stratum when interpreting contest relevance. No fresh final holdout is claimed.

Existing regular Parquet files are accepted after checksum verification against pinned Hugging Face metadata. Large operations use DuckDB with disk spill. Code records actual source-module hashes instead of checking out or changing your Git branch. Different parameters require a new output directory; old experiments are preserved.

Local synthetic smoke checks cover every source including the official SANSA package and FAISS HNSW, all three ranking losses, five pools, temporal windows, feature finiteness, query handling and cached resume. No real Yambda dataset was downloaded or trained in the assistant environment. GPU/ROCm execution is not tested here.


For the optional Rust/ROCm backend assessment and isolated probe, see `docs/catboost_rs_rocm_v9.md`. It is not installed in the main research environment.
