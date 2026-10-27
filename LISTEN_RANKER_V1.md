# Listening candidate / feature experiment

The listening EDA has measured retrieval opportunity. This experiment checks
whether CatBoost can rank that richer pool, and whether scalar listening and
trend features add value on the **same** candidate rows.

It adds three files and changes no existing scripts or results. Real-data runs
are local. No downloads or automatic scans of the original listens file occur.

## What changes

For every user/cutoff, two historical candidate families are formed:

- The first 1000 from the original likes/artist candidate ordering.
- Up to 1000 from the listening RRF used by the EDA.

Their equal-family RRF is capped at 2000 by default, retaining the full union
when its size is <=2000. This prevents dropping useful retrieved items before
training. This is a larger budget than the EDA's equal-budget top1000 fusion;
report it as such. Source pools remain fixed at 1000 each.

Three CatBoostRanker models use identical candidate rows, labels, temporal
windows, objective and hyperparameters:

1. `base_features`: the original 32 scalar like/source features, extended to
   candidates that have no earlier likes (zeros and explicit missing ages).
2. `listen_no_sample_trends`: like features plus listening/feedback/rank features,
   without the seven independent-sample listener trend features.
3. `listen_features`: the same features plus 36 scalar features: counts,
   completion/short/organic fractions, recencies, source reciprocal ranks,
   current dislike/unlike state, and distinct-listener/population-sample trends.

No completion, short-play, dislike or unlike action forcibly excludes an item
or overwrites a future-positive target label. Zero labels indicate absence
from the observed target set, not verified negative preference. Organic plays
are not randomized controls. This is predictive ranking, not causal off-policy
evaluation or a propensity-corrected objective.

All models use QuerySoftMax grouped by UID, with group weight 1 / number of
unique observed target items. This is a within-user ranking surrogate;
actual active-user Recall@100 selects tree count on validation. It is not
exact optimization of Recall@100. The objective differs from the old v1
pointwise classifier, so comparisons to that older run do not isolate features;
the within-run feature ablations do. The final two modes isolate the added
value of sample listener trends; all modes already contain the original
like-popularity trend features. Groups with no observed future targets are
excluded from training, but remain in all-user evaluation. A group whose
positives were all missed by retrieval contributes no positive ranking signal.

References:
- https://catboost.ai/docs/en/concepts/loss-functions-ranking#querysoftmax
- https://catboost.ai/docs/en/concepts/python-reference_catboostranker_fit

## Temporal schedule and cache reuse

Defaults, relative to public end E:

- Train history < E-90d; labels [E-90d, E-60d).
- Validation history < E-60d; labels [E-60d, E-30d).
- Development history < E-30d; labels [E-30d, E).

At each cutoff listening history is the previous 90 days. These windows fit
inside the successful listening EDA's existing cache. A cache fingerprint,
cohort, version or time-coverage mismatch causes an explicit error; it never
silently rescans the 4.65-billion-row original file or shortens a window.
Population trends reuse the exact same independently sampled listeners.

This schedule intentionally differs from the first classifier's training and
validation windows. New 60-day-cutoff validation has 30 days of labels; the old
`development_60d` had 60. Do not compare their metric values directly. The last
30-day development target is compatible with the earlier runs for this cohort.
It has already been inspected and is a development check, not a pristine final
audit. Next-100-event labels are chronologically ordered with item ID ties;
Recall uses distinct target IDs, and already liked items remain eligible.

Report all-user, active and complete-100 metrics, candidate ceilings and paired
feature-gain intervals for active and full100 users. Censoring and activity
selection remain unresolved; do not translate an active score into an expected
contest grade. In particular, the EDA's strongest active gains did not translate
to comparable full100 gains.

## Apply and smoke-test

Save the patch in the repository root:

```bash
cd "$HOME/yambda_contest" &&
git apply --check yambda_listen_ranker_v1.patch &&
git apply yambda_listen_ranker_v1.patch &&
POLARS_MAX_THREADS=4 .venv/bin/python -m unittest discover -s tests -p test_listen_ranker.py
```

Tests use tiny synthetic Parquet files only. They check compatibility with the
original feature implementation, new-item defaults, past-only invariance,
cache coverage guards, all model fits, target-window boundaries and users split
across Arrow inference batches. They do not estimate real recommendation quality.

## Local experiment

Run from a separate Ubuntu terminal. This keeps the analysis independent of
VSCode's integrated terminal and preserves the successful memory-limited setup.
It prints stage markers/peak RSS, keeps logs, and creates a new result directory.

```bash
cd "$HOME/yambda_contest" &&
mkdir -p logs &&
(
  set -o pipefail
  YAMBDA_TRAIN_TAG="$(date +%Y%m%d_%H%M%S)"

  systemd-run --user --scope \
    --unit="yambda-listen-ranker-${YAMBDA_TRAIN_TAG}" \
    --property=MemoryHigh=16G \
    --property=MemoryMax=20G \
    --property=MemorySwapMax=2G \
    /usr/bin/env POLARS_MAX_THREADS=4 \
    "$PWD/.venv/bin/python" -u tools/train_listen_ranker.py \
      --data data/5b \
      --users input/test_users.csv \
      --listen-eda-run eda_results/listens_debug_20261027_163551 \
      --cache-dir data/eda_cache \
      --out "eda_results/listen_ranker_v1_${YAMBDA_TRAIN_TAG}" \
      --max-users 5000 \
      --train-users 5000 \
      --train-future-days 90 \
      --val-future-days 60 \
      --eval-future-days 30 \
      --lookback-days 90 \
      --budget 2000 \
      --iterations 200 400 600 \
      --threads 4 \
      --artist-catalog 200000 \
      --seed 42 \
    2>&1 | tee "logs/listen_ranker_v1_${YAMBDA_TRAIN_TAG}.log"

  YAMBDA_TRAIN_STATUS="${PIPESTATUS[0]}"
  printf 'Training exit code: %s\n' "$YAMBDA_TRAIN_STATUS"
  exit "$YAMBDA_TRAIN_STATUS"
)
```

No full-population listens aggregation is attempted here; the needed
user/item aggregates come from the existing 273 MB cache. Aggregation and
CatBoost Pool creation still need RAM. The 20 GiB cgroup cap controls failure
rather than guaranteeing successful completion under every environment. Four
threads are used for Polars, Arrow CPU work and model fitting.

The report and `eda_for_review.zip` paths are printed on success. Return the
ZIP or commit the small HTML/CSV/SVG/JSON results. Keep pairs Parquet, trained
CBM files, and caches local under the existing ignore rules. Model files include
the full fitted ablation models plus `selected_model.cbm`, shrunk to the feature
configuration/tree count chosen on validation. No contest export is made yet.

This run is one controlled ranking experiment, not a new sweep of popularity
variants. Full-population collaborative candidates (ALS/item-item) and embedding
similarity features remain the next major candidate expansion; direct prior
listening alone supported only a minority of future likes in the EDA.
