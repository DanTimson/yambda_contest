# Unified Colab workflow

Open `notebooks/Yambda_unified.ipynb` in Colab. The notebook contains the frozen
runtime additions, pins the repository and dataset revisions, shows charts,
runs synthetic checks, and runs the candidate/ranking experiments. It does not
push to GitHub or mutate the user's remote repository.

## Prepare once locally

This incremental patch assumes `yambda_listen_ranker_v1.patch` has already been
applied, as in the currently running experiment. Apply it after that run finishes;
keep the current results as the control, rather than restarting the run.

```bash
cd "$HOME/yambda_contest" &&
git apply --check yambda_colab_workflow_v1.patch &&
git apply yambda_colab_workflow_v1.patch &&
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -e '.[als]'
else
  .venv/bin/python -m pip install -e '.[als]'
fi &&
POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p test_colab_workflow.py
```

The current 273 MB listening cache covers only the original 5000-user sample
and a restricted interval. To support earlier next-100 validation and eventual
use of all test users without another re-extraction, create an expanded portable
cache once. This reads the existing local 5B source in bounded batches; it does
not redownload it, alter it, or overwrite the old cache. It retains all 26021
requested UIDs and the same independent 1% population sample across the complete
public interval. File size is printed after extraction/export.

```bash
cd "$HOME/yambda_contest" &&
POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python tools/listen_cache_portable.py \
  --eda-run eda_results/listens_debug_20261027_163551 \
  --cache-dir data/eda_cache \
  --data data/5b \
  --users input/test_users.csv \
  --include-all-history \
  --all-test-users \
  --out data/colab_transfer/yambda_listen_cache_fullhistory.zip
```

The ZIP contains **data only**: `events.parquet` and `manifest.json`. It is not a
code overlay. Reuse the resulting ZIP on subsequent sessions; do not rerun the
export just to reopen Colab. A smaller existing-window export is supported by
the CLI but cannot run the longer historical checks below.

## Colab defaults

1. Select a GPU runtime and Run All. Google Drive persistence is enabled; choose
   a new `RUN_LABEL` for a different configuration.
2. Upload `yambda_listen_cache_fullhistory.zip` when asked, or specify its Drive
   path in `BUNDLE_PATH`. Subsequent sessions can reuse the recorded input.
3. Only likes, reactions and mappings are downloaded from the pinned 5B revision.
   The 36.7 GB original listens file and the 13.8 GB audio file are not downloaded.
4. Original EDA and listening EDA run with graphs. Historical local reports are
   visibly labeled as historical. Replaying older baseline training and the
   current QuerySoftMax control is optional and disabled by default, avoiding
   duplicate local work.
5. The default new experiment runs ALS/BM25 retrieval checks on earlier cutoffs,
   then a sampled-Logloss CatBoost ablation on the expanded pool.
6. Download the compact results ZIP. Tables/model checkpoints remain on Drive
   when enabled. The optional final cell attaches a completed local-control
   `eda_for_review.zip` without rerunning it.

Portable import checks the event SHA256, row count/schema, the content hashes of
all six model inputs and user CSV, selected UIDs, seed and requested time range.
Original extraction provenance is retained, not replaced with invented Colab
source paths. A larger cache is filtered by requested UID **before** group-by,
so the default 5000-user experiment does not aggregate every exported test user.

## The two new retrieval approaches

**Population ALS:** past likes from a deterministic UID-hash sample (default
approximately one third), up to 180 days of history, 64 factors, 20 iterations.
Like and listening profiles are folded into the same item space. A listen is
not treated as a future-like target. Source score/rank, dot/cosine, vector norms,
profile coverage and similarities to the last 5/20 tracks are scalar features;
raw latent coordinates are not passed to trees.

**BM25 item-to-item:** co-like relationships offer a different signal from
low-rank ALS. The catalog is bounded at 200000 past-popular items, 80 neighbors
per item, and up to 100 recent seed tracks per population user to limit heavy-user
cooccurrence cost. Both like and listening profiles can query this model.

The existing likes/listens core is retained when adding collaborative items to
a 3000-item pool. Individual source ceilings, equal-budget comparisons and
unbounded union ceilings are reported separately. A bigger union is not itself
a final Recall improvement.

ALS and BM25 fitting explicitly use CPU native implementations. Dense ALS
retrieval uses PyTorch GPU when available, supporting CUDA or ROCm builds; the
Colab preset uses NVIDIA CUDA. CatBoost has an explicit `--task-type GPU` and
fails early when no CUDA device is visible. No blanket silent GPU-to-CPU fallback
is used for CatBoost. GPU execution is not promised to reproduce CPU scores bit
for bit; actual CatBoost parameters and hardware are recorded.

## Ranking and evaluation

The existing 3-mode QuerySoftMax experiment remains available with its original
features and 90/60/30 cutoffs. The new CF mode compares original, listening and
collaborative features on identical candidate rows, using Logloss and up to 300
negatives per training query (half hard, half random). All **retrieved** positives
are retained; no truth item is inserted into retrieval. Random negative weights
correct sampling probability, followed by per-query normalization. This is not
an exposure propensity or counterfactual correction. Sampling is rejected for
QuerySoftMax rather than silently changing its objective.

Short training/validation windows remain censored. The separate 150/120-day
candidate check evaluates next-100 targets over the longer observed future and
shows complete versus partial/active cohorts. It addresses candidate opportunity;
it does not by itself constitute a final long-horizon validation of the trained
ranker. Complete-only cohorts favor active users and overlapping future folds
are correlated. Hidden-test grade thresholds cannot be read off these metrics.

No final contest CSV export is claimed by this notebook. The current full run
means the complete controlled experiment, not a validated final 26021-user model.

## Checkpoints and scope

`train_listen_ranker.py --resume` reuses completed pair tables and atomic model
checkpoints only if source/configuration provenance matches. A mismatched run
requires a new directory. Stage manifests let the notebook reuse complete EDA
and retrieval reports, while incomplete non-resumable stages get a new attempt
directory. Use Drive if recovery after runtime deletion is needed; local Colab
storage otherwise disappears with the runtime.

CF model cache keys include source identity/content provenance, cutoff,
configuration, implementation hash and dependency versions. Verified portable
input hashes permit cache reuse after a Colab re-download with different local
file modification times. Bounded extraction does not imply constant-memory
aggregation or fitting; synthetic tests do not establish full-data RAM/VRAM use.

The delivered notebook is unexecuted. Validation here covers Python/JSON
structure and tiny CPU end-to-end runs, including time safety, portable import,
negative weights, checkpoints, both collaborative sources and existing ranker
regressions. CUDA and full-data Colab execution remain to be checked locally/in
Colab. No actual Yambda fitting was performed in the assistant environment.

## Findings from the friend's notebook

The useful additions were population ALS, separate folded like/listen profiles,
scalar similarities to recent tracks, album/artist affinities, weighted hard +
random negatives, and longer next-100 evaluation. ALS was present in our initial
repository skeleton but was not active in the recent candidate/ranker scripts.

Its saved current complete-target scores are about 6.55% -> 6.62% and
7.03% -> 7.10% after listens, on different cutoffs/users from ours. Its 500M
listens cover only 2201/26021 users at the >=10 qualifying-play threshold.
The larger 50M pilot figures in its historical-results cell refer to different
experiments and populations. They are not the current contest-wide result.

Neural two-tower retrieval remains possible and is not forbidden by the CatBoost
ranking requirement, but ALS and item-to-item supply lower-cost, interpretable
controls first. The homework's similarity and negative-sampling ideas are used
here without claiming that logQ or inverse sampling establishes causal effects.
