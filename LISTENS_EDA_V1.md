# Listening evidence before another ranking experiment

This patch adds only `tools/explore_listens.py`, its synthetic tests and this
note. It does not change existing scripts, the Git ignore file, or results.
No model is fitted and no data is downloaded by the script.

## First local run

Save `yambda_listens_eda_v1.patch` in the repository root, then run:

```bash
cd "$HOME/yambda_contest" &&
git apply --check yambda_listens_eda_v1.patch &&
git apply yambda_listens_eda_v1.patch &&
POLARS_MAX_THREADS=4 python -m unittest discover -s tests -p test_explore_listens.py
```

```bash
cd "$HOME/yambda_contest" &&
POLARS_MAX_THREADS=4 python tools/explore_listens.py \
  --data data/5b \
  --users input/test_users.csv \
  --out eda_results/listens_eda_v1 \
  --cache-dir data/eda_cache \
  --baseline-run eda_results/candidate_ranker_v1 \
  --max-users 5000 \
  --seed 42 \
  --future-days 90 60 30 \
  --lookback-days 90 \
  --budget 1000 \
  --population-permille 10 \
  --min-listeners 10 \
  --artist-catalog 200000
```

Open the report directly, or serve it locally:

```bash
cd "$HOME/yambda_contest" &&
python -m http.server 8000 --bind 127.0.0.1 --directory eda_results/listens_eda_v1
```

Then visit http://127.0.0.1:8000/report.html .
Return `eda_results/listens_eda_v1/eda_for_review.zip`, or commit the small
HTML/SVG/CSV/JSON results to the repo. Keep the Parquet caches local.

## What is measured

- Literal next 100 chronological like **events**, distinct-item Recall@100;
  previously liked tracks remain eligible. Timestamp ties use item ID order,
  matching the existing candidate benchmark.
- All, active and complete-100 users are reported separately. Targets at each
  cutoff end at the public likes-file end. In particular the 90-day target is
  **not** the ranker v1's shorter validation-label interval. Only its historical
  candidate lists are reused. Do not compare these scores to its validation
  scores as if they were the same experiment.
- Last 90 days of listening support: any play, >50% play, >=90% repeated play,
  and short-only <=20% pairs. These are exploratory thresholds, not verified
  skip labels. >100% plays are retained.
- Four personalized sources: last-listen recency, decayed qualifying plays,
  decayed organic qualifying plays, repeated near-full plays. Source lists
  are capped at the same budget. Empty histories are not popularity-padded.
- Global sources from a fixed independent UID-hash sample: distinct listeners,
  qualifying listeners, organic qualifying listeners, and week-over-week
  smoothed growth. A listener counts once per item/week. The support threshold
  is measured in sampled users; counts are not extrapolated to the population.
- RRF controls and equal-budget fusions, plus a clearly marked unbounded union
  of the likes-control and capped listen-RRF pools (not all historical tracks). `likes_control` is the old **candidate ordering**, not CatBoost's
  predictions. This experiment tests retrieval opportunity, not model lift.
- Per-user new hits beyond likes-only candidates, paired active-user bootstrap
  changes, repeated time snapshots, target-support and audience-growth charts.
- Potential negative-label contradictions among historically listened pairs:
  later observed positive membership after short-only, qualifying, mixed,
  disliked or unliked evidence. Groups overlap. Cancelled dislikes/unlikes
  are handled using the latest available action; tied timestamps are ambiguous.

A listened-but-not-liked pair is not automatically a negative. No observed
next-100 like is also not proof of dislike: activity and right censoring matter.
Organic/recommendation channels are observational; this is not an identified
counterfactual policy evaluation. Short-play positive fractions are descriptive
pair-level diagnostics, not per-exposure conversion or causal effects.

## Reading order and next decisions

1. **Target support:** how often did a future liked track appear in past listens?
   This is an unbounded opportunity ceiling, not a recommendation metric. Low
   support means direct history retrieval is insufficient; co-listen/ALS or
   other collaborative candidates can still benefit from the denser histories.
2. **Fixed-budget coverage and listen-only hits:** do listens add targets our
   current sources miss? Good coverage with weak top100 ordering calls for
   richer ranker features. New items matter even if a source ranks poorly alone.
3. **Growth charts:** is distinct-listener growth complementary to like trends,
   and do organic/recommendation channels behave differently? No claim that a
   1% sample has the same accuracy as full-population statistics.
4. **Negative audit:** what gets contradicted by later positives? Use this to
   design confidence levels and delayed labels rather than indiscriminately
   adding every non-like as a hard negative.

The next model should keep the likes objective while adding listening-derived
candidates/features and carefully selected negatives. Full-population collaborative
training and embeddings are not implemented by this diagnostic patch.

## I/O, memory and reproducibility

The large listens file is read in bounded Arrow batches **once** to extract:
selected users plus a fixed population sample, within the union of the required
historical time windows. No early completion or organic-channel filter is
applied. The default population sample is 10/1000, approximately 1% of user IDs,
independent of labels and user activity. Population and target cohorts may overlap.
The number of sampled users with events is reported, not assumed.

Later cutoffs reread only the reduced local cache. Group-by aggregation still
needs memory proportional to the retained pairs; this is not a fixed-memory
full-data aggregation guarantee. Start with the given 5000-user command. Cache
size and extraction duration are logged and included in the report; no reliable
wall-clock estimate is assumed for a user's disk/CPU.

Cache signatures include source path, size, mtime, users, seed, time bounds and
sampling version. A complete manifest is written only after extraction succeeds.
Interrupted extraction restarts rather than trusting a partial file. Repeating
the same parameters reuses the cache; changing users/time windows/sample creates
a new cache and requires another full scan. Use a new output directory on rerun;
existing results are never overwritten. Existing likes-statistic caches are reused.

`--baseline-run` reads only local candidate pair files and their cutoff/user
metadata. Missing required pair files or insufficient user coverage cause an
explicit error. Without a matching snapshot the script builds a fresh past-only
likes/artist RRF and records that baseline type. These baseline types can differ,
so interpret within-cutoff comparisons rather than attributing cross-cutoff
changes entirely to listening.

Albums are downloaded and available, but this experiment does not yet use them.
All real scans and training remain local; the supplied tests use tiny generated
Parquet files only.
