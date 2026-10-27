# Experimental plan and initial audit

## 1. Resolve the target before optimizing it

The following are concrete questions for the instructor; none prevents initial offline work:

1. What exact timestamp is the last allowable history event for the hidden target?
   Are hidden labels external future events, or a withheld portion of the public log?
2. Does “next 100 likes” mean 100 chronological like events, 100 distinct items, or
   all likes in a fixed horizon (the notebook uses 30 days)?
3. Can a previously liked then unliked track appear in the target again? Should the
   current liked state or first-ever history determine exclusions?
4. What happens for users with fewer than 100 future likes or none? How were the
   26,021 users selected? Is there a common future observation end?
5. Is the submission column definitely `item_ids` as in the example, with the supplied
   metric adapted accordingly? Are targets both organic and recommendation-driven?

The most consequential error would be to train on public events that are inside the
organizer's withheld interval. Do not infer their permissibility simply from availability.
Conversely, a clear instruction that all public events are allowable would make the
public-log end a valid deployment history boundary; ask once, then follow the answer.

## 2. What was found in the supplied files

- `test_users.csv`: **26,021 unique uids**, range 89–999884.
- `submission_example.csv`: same uid order, exactly 100 IDs per row, but **one identical
  recommendation list for every user**. It is a format check / popularity baseline.
- `recall.py`: macro-average set intersection divided by `min(unique targets,100)`.
  It ignores ranking inside the first 100, and removes uid without joining on it.
  `zip` silently stops at the shorter input. Submission validation must close these gaps.
- `baseline.ipynb`: 30-day future windows, only a 90-day feature history; a CatBoost
  classifier selected using AUC, common popularity candidates and random negative items.
  The random negatives can collide with positives, receive contradictory labels, and differ
  from the inference distribution. These are limitations of that particular starting code,
  not evidence that CatBoost classification itself is unsuitable.
- The baseline deduplicates to the first-ever like before splitting. The starter follows
  that convention provisionally; it is not sufficient evidence of the hidden label rule.
- The baseline keeps one artist by minimum ID. The starter preserves every credited artist.
- Baseline coefficients and printed 50M metric are not meaningful forecasts for the full
  hidden test with very different user coverage and possibly different target semantics.

Measured Yambda-50M EDA appears in `evidence/eda_50m.json`. It has history for only 245
requested users. The package's real-data pilot is explicitly restricted to this intersection.
That makes it useful for detecting implementation problems, not a representative contest
estimate. The example's global recommendation list is not reverse-engineered as a hidden
answer source.

## 3. Offline evaluation design

### Common timeline

Choose cutoff T; every interaction type, candidate model, popularity statistic, user
profile and feature uses events strictly before T. Build targets after T. Do not hold out
each user's last 100 and train global item statistics on other users' later events.
That would allow future trends into the past model.

The Time To Split paper supports global chronological separation and matching validation
and test protocols. Its preferred target choices are studied for sequential **next-item**
prediction. Here we emit one static list for a block of 100 future likes, so adapt the
principle rather than copy the paper's target choice or its sequential history updates.

### Development blocks and audit block

Use at least 2–3 development placements of the full train → validation → audit pipeline.
First use an early block to learn what works. Freeze candidate budgets, feature families,
label semantics and iteration policy. Reserve the latest block for a single final audit.
`backtest` chooses iterations on validation and evaluates the selected model on the audit.
For repeated development, regard that run's audit as development data and keep a separate
later cutoff unused. The code cannot stop a human from repeatedly looking at an audit.

Different snapshot label ranges may overlap in the training set; repeated user observations
are correlated. Training labels are always truncated before validation. Validation labels
end before the audit cutoff. `--embargo-days` can provide an additional gap before validation;
it is a sensitivity test, not an automatic cure for target mismatch.

For the next100 interpretation, label windows have user-dependent duration. Near the log
end many targets are right-censored. Report, per split:

- all requested users, users with observed future likes, and users with 100 observed targets;
- Recall over all / active / full100 cohorts, plus warm/cold history strata;
- target-count distribution, time-to-100 and observation length in a follow-up EDA;
- history length and target popularity distributions, seasonal changes and cohort coverage.

Full100 users are more active and may have different tastes; conditioning on them does
not remove selection bias. A lack of future likes in a short block also does not prove a
track is a true negative. Prefer earlier cutoffs with enough future to inspect censoring,
and compare next100 to 30/60/90-day capped targets until the hidden protocol is clarified.
For official-like macro recall, keep user weighting explicit; don't optimize pooled hits.

### Uncertainty

Use paired per-user deltas on the exact same cohort, with bootstrap intervals. The starter
provides these for CatBoost minus RRF. They quantify sampling uncertainty within the cohort,
not uncertainty from future trend changes or from repeated model selection. Stability
across chronological blocks is more important than an impressive narrow interval on one.
An improvement that wins only one block or only very active users is not yet dependable.

## 4. Candidate-first experiment ladder

Keep retrieval costs and per-source caps in the experiment record. Evaluate candidates
before tuning trees. Increasing the union can improve recall but worsen ranker cost or
introduce harder negatives; examine the full tradeoff.

| Stage | Experiment | Evidence to retain |
| --- | --- | --- |
| 0 | EDA, format checks, coverage at each dataset size | User counts, time range, overlaps, memory |
| 1 | Global pop vs 7/30/90-day pop, exponential decay | Recall@100 across time blocks |
| 2 | Artist candidates + ItemKNN | Individual coverage, incremental union hits, latency |
| 3 | Qualifying listens not already liked | Delayed-like yield, repeat plays, organic share |
| 4 | iALS on likes; then likes + play counts | Incremental recall after existing sources |
| 5 | Audio neighbors and profile ANN | Tail/cold-item gains; missing embeddings; cost |
| 6 | CatBoost on the identical candidate union | Recall@100 vs RRF; feature ablations |
| 7 | Multiple historical snapshots, classifier vs ranker | Block stability and actual macro Recall |
| 8 | Recent/listened/trend variants and final refit | Three strongest justified export variants |

Do not train a large sequential neural model first. It could later be a candidate source,
but the instructor's hints and available baseline favor cheap complementary retrieval and
a strong tabular scorer. Embeddings are not forbidden neural information: pretrained audio
vectors can be used without training an end-to-end neural final scorer.

### Music-specific extensions with likely value

- **Listening → like conversion:** count and recency of substantial plays, repeat listens,
  completion distribution, first play, days since first/last exposure, organic fraction.
  Existing code has counts/recency for plays >=50%; the richer aggregates are next.
- **User–artist / album relationships:** all artists, short/long affinity, unliked catalog
  neighbors. Album mappings are available; album retrieval is not implemented yet.
- **Trends:** popularity over several intervals, smoothed growth, new-to-the-log tracks,
  artist-level new releases as inferred candidates. Distinguish observed appearance from
  actual release date. A feature may use only trend data available at recommendation time.
- **Dislikes and unlikes:** reconstruct state in timestamp order. A past unlike is not
  necessarily permanent dislike; blocking it blindly may hurt if re-likes count as targets.
  Evaluate before adding hard rules. These event states are not implemented initially.
- **Complementarity:** candidates with modest standalone recall can add unique hits beyond
  popularity, e.g. low-popularity audio neighbors. Measure incremental hits after the union,
  not only separate source leaderboards.

### Embeddings into trees

Use scalar and aggregate relationships: user-centroid cosine, cosine to the last/recent
liked tracks, max/mean/top-q seed similarity, distance to each taste cluster, ALS dot product,
artist profile similarity. Contrast recent and long-term user taste. Keep missingness
explicit. The initial code uses audio centroid cosine, retrieval scores and artist affinity;
max-seed and multiple-interest profiles are planned extensions. Centroid-only retrieval
can wash out a user with several distinct musical tastes.

Raw coordinates are not mathematically unusable by trees, but split-based models are a
poor default for rotation-dependent high-dimensional similarity. Compute the relationships
first; do not spend depth rediscovering dot products. A reduced PCA representation can be
an auxiliary experiment, with PCA fit at the same historical boundary.

## 5. Resource strategy

Start on CPU. The first gain is correct history coverage and useful candidates, not GPU
training speed. The full public dataset contains mostly listens: downloading likes alone
is much cheaper than downloading all interaction files.

- Debug on 50M with known warm-user intersection; use small CF item/user caps.
- Inspect 500M if needed, but it is not a substitute for coverage of every requested user.
- Move to 5B likes for global collaborative statistics and requested users' full history.
- Add only requested users' listen features initially, while keeping population likes for
  candidate models. Don't restrict all population data to the 26K target users.
- For larger runs, persist per-cutoff candidate tables and use them to compare rankers
  without rebuilding KNN. The starter saves them, but does not implement a full cache key / resume engine.
- ItemKNN over millions of tracks needs a faster sparse top-k backend or a bounded retrieval
  universe; the initial implementation uses the latter. ALS/audio per-user scans should
  become batched operations or ANN before expanding their universes.
- Do not extrapolate runtime or RAM from the tiny pilot. Full 5B runs, full-listen scans
  and full 26K ALS/audio exports have not been measured in this environment.

## 6. Delivery strategy

Use offline evidence to choose three files. A reasonable initial set is a CatBoost model,
a more recent/trend-focused CatBoost variant, and an alternate retrieval/collaborative
variant that wins in other blocks. Simple RRF and recent popularity are included as
operational fallbacks, not claimed to be the eventual strongest choices.

Averaging scores across CatBoost models or source variants requires compatible score
calibration (or rank fusion) and must be evaluated. Duplicate three near-identical files
only if their underlying variation actually produces gains. If no diversity wins offline,
three strong similar models are preferable to a weak “diverse” one.

Before submitting: freeze config; refit from historical snapshots with all labels before
the confirmed allowable boundary; export in the exact test uid order; validate exactly
100 unique integer IDs; archive precisely three CSVs. There are no contest submissions
in this starter archive because no hidden-boundary assumption has been confirmed.

## Sources

- Official Yambda dataset card and schemas: https://huggingface.co/datasets/yandex/yambda
- Actual full flat file sizes: https://huggingface.co/datasets/yandex/yambda/tree/main/flat/5b
- Yambda paper (2025): https://arxiv.org/abs/2505.22238
- Gusak et al., Time to Split (RecSys 2025): https://arxiv.org/abs/2507.16289
  and https://arxiv.org/html/2507.16289v1
- Paper code: https://github.com/monkey0head/time-to-split
- CPU collaborative filtering implementation: https://github.com/benfred/implicit
- CatBoost ranking losses/metrics: https://catboost.ai/docs/en/concepts/loss-functions-ranking

Dataset content and supplied assessment code were inspected on 2026-10-27. The downloaded
50M file revision is recorded in `evidence/dataset_manifest.json` after the pilot.
