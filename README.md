# Yambda contest starter

A CPU-first, temporal **candidate retrieval → CatBoost → 100 unique item IDs** pipeline.
This is a starting system for experiments, not a claim of competitive hidden-test quality.
Python 3.11+; Linux/WSL recommended. No CUDA or ROCm is needed.

## Read this before training

The assignment and baseline disagree: the assignment asks for **the next 100 likes**,
while the notebook evaluates **all likes within 30 days**. The hidden cutoff, repeat-like
policy, future observation horizon and empty-target policy are not supplied. They can
change model selection substantially. Ask the instructor the questions in `EXPERIMENT_PLAN.md`.

This implementation assumes:

- one prediction at a common global cutoff, with **no subsequent history updates**;
- targets are the first 100 **first-ever unique liked items** after the cutoff;
- already liked items are excluded using the entire pre-cutoff log;
- timestamps are seconds quantized to 5-second granularity, as used by the supplied
  baseline (`86_400` per day); do not multiply the observed timestamp values by five;
- simultaneous events are ordered by item ID (an explicit reproducibility convention);
- no organic-only target restriction;
- events at the cutoff are future: history uses strict `timestamp < cutoff`.

Both the target rule and censoring are assumptions, not verified contest rules.
The code also supports a fixed time window, including the baseline's 30-day formulation.

## Installation

From the extracted project directory:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[test]'
python -m pytest -q
```

Without uv: `python -m venv .venv`, activate it, then `python -m pip install -e '.[test]'`.
If your proxy is SOCKS, install `socksio` as well. Existing `HTTP_PROXY`, `HTTPS_PROXY`,
`ALL_PROXY`, and HF token configuration are respected; no credentials are included.

## 1. Download a debugging dataset and inspect coverage

```bash
python -m yambda_contest.cli download --data data/50m --size 50m
python -m yambda_contest.cli eda --data data/50m --out eda_50m.json
```

Downloads only likes and artist mappings by default. Each directory is pinned to a concrete
HF revision in `dataset_manifest.json`; don't mix dataset sizes in one directory.
Existing files are reused. Data files may be symlinks into your HF cache; don't delete that
cache while using them. Windows uses copies if symlinks are unavailable.

**Measured on the supplied users:** 50M contains like history for only **245 / 26,021**
test users (0.942%). It has 881,456 raw likes, 8,283 users with likes and 181,304 liked items.
The 10,000-user dataset total counts users with other interaction types too.
`evidence/eda_50m.json` preserves the measured output.

Use 50M to debug; a final system based on 50M alone would lose almost all personalization.
500M is an intermediate experiment; check coverage rather than assuming it covers the test.
The full **5B likes file is about 768 MB compressed**, not 5 billion likes. Its 89.3M likes
are a plausible next step without downloading 4.65B listens. Peak RAM can be much larger
than the compressed download; start with limited cohorts and CF budgets. Full 5B processing
has not been benchmarked by this starter.

```bash
python -m yambda_contest.cli download --data data/5b --size 5b
python -m yambda_contest.cli eda --data data/5b --out eda_5b.json
```

## 2. Reproduce the small real-data pilot

The package includes the 245-user intersection as `evidence/pilot_users.csv` specifically
for this debugging run. It is **not** a replacement for the contest user file.

```bash
python -m yambda_contest.cli backtest \
  --data data/50m --users evidence/pilot_users.csv --out artifacts/pilot \
  --train-cutoffs 13000000 --val-cutoff 17000000 --test-cutoff 21000000 \
  --train-users 150 --eval-users 150 \
  --cf-items 5000 --cf-users 5000 --per-source 100 --max-candidates 500 \
  --iterations 50 100 --threads 4
```

These are elapsed dataset timestamps, not Unix dates. The gaps deliberately leave more
than 30 days to observe future likes. Training labels stop at validation start; validation
labels stop at audit-test start. Test labels use remaining observed future. An optional
`--horizon-days 30 --target-mode window` switches to the notebook's fixed-window target;
`--horizon-days 60 --target-mode next100` means at most 100 first likes within 60 days.

The defaults sample from all users in `input/test_users.csv`, including cold users.
Do not use the full test file with 50M and interpret the active-user score as representative:
most requested users have no data there. On full data start with 1,000 evaluation and
2,000 training users, then increase after measuring runtime and memory.

`backtest` writes:

- `experiment.json`: config, cutoffs, user samples, feature schema, iteration selection;
- `train_pairs.parquet`, `val_candidates.parquet`, `val_truth.parquet`;
- `model.cbm`: the selected model;
- `test_candidates.parquet`, `test_truth.parquet`, `test_predictions.json`;
- `report.json`: CatBoost, RRF, popularity, recent popularity, source coverage, cold/warm
  strata and a paired user-bootstrap comparison of CatBoost versus RRF.

The latest audit block is evaluated **once after iteration selection**. This code does not
refit on validation before that audit, so the reported score is for the exact selected
model. Once you start changing design in response to the audit score, it has become another
validation block; reserve a later block or describe that limitation honestly.

## What the metrics mean

The supplied official code computes:

`hits_in_top100 / min(number_of_unique_true_items, 100)`, averaged per user.

`recall_all` includes every sampled user (zero if no observed target).
`recall_active` conditions on at least one observed target.
`recall_full100` conditions on observing all 100 target items.
Neither conditional estimate is automatically an unbiased estimate of the hidden score:
right censoring and activity-based selection remain. Full100 is a diagnostic, not a reason
to drop less-active users from all evaluation.

Candidate diagnostics distinguish **fraction of relevant items retrieved** from the
**best attainable final Recall@100**. In the all-items window mode, an unbounded candidate
list's hits divided by `min(n_true,100)` could exceed one; the oracle metric caps its
numerator at 100. Ranking can never exceed that attainable ceiling on the same cohort.

The official scorer drops the uid column and compares rows positionally; it does not
validate row counts, uid alignment or uniqueness. This exporter validates them separately.
It writes `uid,item_ids`, matching the example file and assignment; the scorer's default
`ranked_item_ids` is an internal naming discrepancy, not a reason to rename the submission.

## Candidate sources and features

Each source proposes up to `--per-source` unseen tracks. Reciprocal rank fusion (RRF)
combines and caps the union to `--max-candidates`. No target items are injected.

| Source | Signal | Initial implementation |
| --- | --- | --- |
| pop | Global historical popularity | Unique users liking each item in feature history |
| recent | Fresh demand | Sum of exponentially decayed likes, 7-day time constant |
| trend | Growth | Smoothed 7-day / preceding 7-day ratio, weighted by current volume |
| artist | Personal taste | All credited artists; decayed user affinity × current track popularity |
| knn | Co-liking | Sparse cosine ItemKNN on a bounded item/user universe |
| listened | Delayed likes | Recent substantial plays not already liked; optional file |
| als | Latent preference | Optional CPU implicit ALS; fold-in from recent user seeds |
| audio | Acoustic similarity | Optional cosine retrieval to the recent liked-item centroid |

Per-source scores and reciprocal ranks, number of sources, RRF, item popularity at
7/30/history-day scales, growth ratio, first/last observed-like age, user activity/recency,
user–artist affinity, user–item play statistics and audio cosine go into CatBoost.
`pop180` is the feature-history count; the historic name remains if `--history-days` changes.
A first-observed like is **not** a verified track release date.

Training negatives are non-matching candidates from the same retrieval pipeline used at
inference. Unretrieved positives affect the retrieval ceiling but are not artificially
inserted into ranking training. All candidates are retained in the initial training set;
there is no random-negative distribution mismatch. No-positive-future users are omitted
from classifier fitting, with their evaluation treatment still reported explicitly.

The initial ranker is `CatBoostClassifier` with Logloss, selected by actual Recall@100,
not AUC. User weights are inverse target denominator to match macro recall contributions.
A classifier is intentional: within a fixed user list, expected hits are additive.
It is not a guarantee of superiority over `CatBoostRanker`; compare YetiRank later on the
same saved candidates. Raw embedding coordinates and raw numeric uid/item_id are not
fed to the trees. The objective is a surrogate; empirical Recall decides.

The KNN implementation computes small sparse item-product blocks and prunes neighbors;
it never allocates a dense item×item matrix. It still has substantial CPU cost. Increase
`--cf-items` and `--cf-users` separately after profiling. The initial cap favors popular
items and limits tail discovery; record that limitation in ablations. The current audio
and ALS retrieval scans their retained factor/vector catalog per user, so they are pilot
implementations; ANN or batched matrix multiplication should precede a large final export.

## 3. Add listens, ALS and audio independently

```bash
python -m yambda_contest.cli download --data data/50m --size 50m --files listens
uv pip install -e '.[als]'
```

Re-run a backtest with `--als` to add ALS. Listens are used automatically if that local
file exists; `--no-listens` disables them for an ablation. A full 5B listens download is
about 36.7 GB compressed; don't fetch it for the first run. Current code scans it lazily
and retains only requested users in the 90-day pre-cutoff window. For repeated full-data
experiments, pre-aggregate partitioned user/item/artist counters rather than repeatedly
scanning it. Dislike/unlike state handling is deliberately not implemented yet.

Audio vectors are about 13.8 GB in the original file. Download explicitly:

```bash
python -m yambda_contest.cli download --data data/50m --size 50m --files embeddings
python -m yambda_contest.cli audio --data data/50m --out data/audio_early \
  --cutoff 13000000 --max-items 100000
```

Then add `--audio-dir data/audio_early` to the backtest. Extraction streams Parquet and
retains known historical items only. It normalizes vectors and stores memory-mappable NumPy
arrays. The selection cutoff must be no later than the earliest snapshot; later-selected
catalogs are rejected. This conservative fixed catalog misses later items; extracting a
separate pre-cutoff catalog per snapshot is the next improvement. Static audio content is
not intrinsically future behavioral information, but selecting vectors by future popularity
would be. No new neural network training is required to use the supplied vectors.

## 4. Final training and three-file archive

Only after settling the offline setup and confirming the allowable history boundary:

```bash
python -m yambda_contest.cli fit-final \
  --data data/5b --out artifacts/final --as-of 26000000 \
  --train-cutoffs 13000000 17000000 21000000 \
  --iterations 200 --train-users 3000 --threads 4
python -m yambda_contest.cli submit \
  --data data/5b --model-dir artifacts/final --out artifacts/submissions
```

**26000000 above is an example boundary near the observed public-log end, not a known
contest cutoff.** Replace it with the instructor-confirmed cutoff. The command deliberately
requires an explicit `--as-of`. Copy the best iteration count and candidate settings from
experiments; final training does not silently inherit another run's choices. Labels for
all historical snapshots are truncated before `as_of`. The final config and feature schema
are saved next to the model and reused automatically at export.

Export is batched over users and produces `catboost.csv`, `rrf.csv`, `recent.csv`, and
`submissions.zip` containing exactly those three CSVs in test-user order. Each row has
100 distinct integer IDs. These are provisional three methods, not three demonstrated
strongest submissions. Before the actual contest, replace weaker variants with models
that win and complement each other across temporal backtests. Their best-of-three hidden
score is not their union score; don't substitute diversity for measured quality.

```bash
python -m yambda_contest.cli validate artifacts/submissions/catboost.csv
```

The test user list is included; the 20MB identical-popularity submission example is not.

## Validation and provenance

`tests/` checks the official formula, repeat-like and boundary behavior, invariance to
future mutations, cold-user handling, actual training/export, and optional audio/ALS paths.
Optional ALS test skips if `implicit` is not installed. `evidence/` contains measured
pilot reports and dependency versions, not fabricated contest predictions.

Read `EXPERIMENT_PLAN.md` for the split rationale, experiment sequence and unresolved rules.
Source code is deliberately small enough to inspect before expensive runs.
