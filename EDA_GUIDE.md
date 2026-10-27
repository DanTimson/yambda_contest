# Explore Yambda before training

This is an EDA-only add-on to the earlier starter. It does not change its training code,
download anything, fit CatBoost, build KNN/ALS models, or write to your dataset directory.
It reads local Parquet files and writes a new output directory. Existing nonempty output
directories are rejected so earlier analyses are preserved.

## Apply locally

Copy these files into the corresponding locations in your repository:

- `tools/explore_yambda.py`
- `tests/test_eda_addon.py`
- `requirements-eda.txt`
- `EDA_GUIDE.md`
- `.gitignore` — inspect it and merge any additional project-specific rules you already use.

The add-on leaves `src/`, `pyproject.toml`, the earlier experiments, and existing notebooks
unchanged. There are no remote Git commands or automatic commits in the script. Future
repository access by the assistant is for inspection; changes are supplied for you to apply.

## Install the small EDA dependencies

In your activated project environment:

```bash
uv pip install -r requirements-eda.txt
```

Or use `python -m pip install -r requirements-eda.txt`.
No CUDA, ROCm, torch, CatBoost or notebook server is required by this script.

## First run: the local likes dataset you intend to use

If you have downloaded full likes into `data/5b`:

```bash
python tools/explore_yambda.py \
  --data data/5b \
  --users input/test_users.csv \
  --out eda_results/5b_likes_v1
```

If your files are under `data/50m` or `data/500m`, replace the data path and output name.
Do not redownload data just to match these names. The script detects the files present.

This run scans the complete local **likes** file for population summaries and historical
popularity, but collects chronological event histories only for the supplied users. On
89M likes this still requires multiple scans and meaningful CPU/RAM; no full-5B resource
claim is made. Let the stages print progress. Sampling requested users limits cohort
processing, not global population scans:

```bash
python tools/explore_yambda.py \
  --data data/5b --users input/test_users.csv \
  --max-users 2000 --out eda_results/5b_sample_v1
```

The 2,000 users are a seeded sample from the supplied user file, not from all population
users. The population and full supplied-user coverage numbers still use the entire user
file. Run without `--max-users` for the complete requested cohort when practical.

## Read and return the output

Open `eda_results/5b_likes_v1/report.html` in a browser. It contains embedded plots and
tables and works offline. It does not depend on a CDN or remote JavaScript.

Send back **`eda_results/5b_likes_v1/eda_for_review.zip`**. It contains:

- `report.html` — readable exploration;
- `summary.json` — structured results, settings, versions and data revision if recorded;
- `tables/*.csv` — complete tables, including per-user activity summaries;
- `plots/*.svg` — exact chart exports.

Or commit the generated report, JSON, tables and plots so they are visible in the repo.
The `.gitignore` allows them on purpose, while ignoring the ZIP duplicate. Reports show
elapsed dataset times and anonymous IDs; they do not contain the event logs or vectors.

The report covers:

1. **Inventory:** local file sizes, row counts, row groups and schemas from metadata.
2. **Coverage:** how many supplied users have like history; missing-user IDs.
3. **Population versus cohort:** distributions of likes per user, including a separate
   accounting of zero-like users; popularity concentration and head/tail event shares.
4. **Timeline:** daily likes, active-user counts and organic shares where available.
   Dataset day numbers are elapsed offsets, not calendar dates.
5. **Repeat likes and cadence:** distinct user–item pairs, repeated-event fraction,
   inter-like gaps and same-timestamp transitions.
6. **Target completeness at multiple cutoffs:** how many users have 100 observed future
   likes, incomplete target counts, activity strata and time-to-100 among complete users.
7. **Target-policy sensitivity:** first 100 like events, first 100 distinct future items,
   and first 100 items never previously liked; previously liked target share.
8. **Cheap references:** strictly pre-cutoff popularity coverage and Recall@100, with and
   without excluding past likes. This tests assumptions without fitting a model.
9. **Artists:** mapping coverage, multiple artist credits and the fraction of future
   target pairs supported by an artist the user previously liked.
10. **Trends:** adjacent-week popularity churn and smoothed growth among tracks with at
    least ten events in the last week.

The initial cohort plot omits zero-like users because its horizontal axis is logarithmic;
the summary and cohort table retain them. Whole-log popularity is descriptive only;
the cutoff diagnostics recompute popularity strictly from the past.

## What “100 likes” means here

The primary analysis takes **the next 100 chronological like events per uid**, not all
likes in a 30-day window. It compares predictions against the set of those item IDs, using
the supplied metric's denominator `min(number of distinct target IDs, 100)`.

Repeated like events can produce fewer than 100 distinct target IDs even when all 100
events are observed. Two additional variants measure whether that matters:

| Variant | Selection |
| --- | --- |
| `next100_events` | First 100 future like events, including repeats |
| `next100_unique` | First 100 distinct items appearing in future likes |
| `next100_new_items` | First 100 distinct future items, excluding every item previously liked |

The earlier starter used the third convention. This EDA does not overwrite that prior
work or silently declare it the contest truth; it quantifies the effect before we choose
hard exclusion and deduplication rules. Same-timestamp events are ordered by item ID for
reproducibility. That order is not actually observed inside the anonymized time bin.

Default cutoffs are **120, 90, 60 and 30 days before the observed log end**. These numbers
specify how much future remains observable, **not** a fixed target horizon. Every cutoff
still targets the next 100 likes. With manual cutoffs:

```bash
python tools/explore_yambda.py \
  --data data/5b --out eda_results/5b_custom_v1 \
  --cutoffs 13000000 17000000 21000000
```

Observed future ends at the dataset end. Users with fewer than 100 events are censored;
their observed prefix is not necessarily their entire future. The time-to-100 distribution
is conditional on completeness and will understate the wait for the whole cohort. The
report keeps zero-future, active-future and complete-100 counts explicit. It does not claim
a complete-user-only score is an unbiased contest estimate.

Inspecting the whole timeline makes it exploratory development data. To retain a genuinely
uninspected final audit later, obtain/choose a separate period that has not informed model
design. This report does not manufacture such an audit from already examined data.

## Optional: listening before future likes

Only if `listens.parquet` is already local:

```bash
python tools/explore_yambda.py \
  --data data/5b --out eda_results/5b_with_listens_v1 \
  --with-listens --listen-users 1000
```

This scans pre-cutoff listens for a reproducible sample of at most 1,000 supplied users,
using a 90-day history before the **latest EDA cutoff**. It measures how many future target
pairs were already listened to, prior play counts and recency, completion/replay summaries,
and a simple recent-listened top-100 reference. No future listens enter those features.
Qualified listens use `played_ratio_pct > 50`; values above 100 are retained as replay
behavior. The reference intentionally leaves seen-like filtering separate.

A sampled user filter can substantially reduce retained data but **does not guarantee a
small physical scan** of a large Parquet file. This option may be much slower than likes
EDA. First run the likes-only report; then add listens if its diagnostic value justifies
the scan. No listens file means a clearly marked skipped section rather than a download.

## Optional: audio coverage and vector audit

Only if `embeddings.parquet` is already local:

```bash
python tools/explore_yambda.py \
  --data data/5b --out eda_results/5b_with_audio_v1 \
  --with-embeddings
```

This reads the item-ID column for cohort item coverage and at most 1,024 vector rows for
shape, finiteness and norm checks. The vector sample is the file's first rows, not a random
or representative catalog sample. It builds no ANN index and does not load all vectors
into RAM. Full-cohort coverage is descriptive; it must not later be mistaken for a
past-cutoff feature-selection procedure.

The two optional flags may be combined. Artist mapping is used automatically if present;
albums, dislikes and unlikes are inventoried but not yet analyzed as state transitions.

## Git hygiene

The supplied `.gitignore` excludes raw data, vector/model formats, common output folders,
archives, virtual environments, caches and credentials. It **keeps** source code, notebooks,
`uv.lock`, the supplied `input/test_users.csv`, and EDA CSV/JSON/HTML/SVG evidence trackable.
It does not blanket-ignore CSV or JSON files, since that would conceal the analysis.

Before committing:

```bash
git status --short --untracked-files=normal
git status --short --ignored
git diff --cached --stat
```

If large files or credentials were already tracked, `.gitignore` alone does not remove
them. Inspect `git ls-files` and, only for the paths you intend to untrack, run e.g.:

```bash
git rm -r --cached -- data artifacts
```

This removes those paths from Git's index, keeping local files. Omit nonexistent or
intentionally tracked paths rather than copying that command blindly. The script does
not run it for you. Previously committed credentials require separate remediation; a
new ignore rule does not remove them from repository history.

Avoid committing large notebook outputs. Keep notebooks themselves if useful and clear
their generated output before pushing when it contains bulky tables or embedded data.

## Checks

```bash
python -m unittest discover -s tests -p test_eda_addon.py
```

Checks cover repeat-like target semantics, complete report generation with all optional
paths, history filtering of listens, preservation of prior output, and actual Git ignore
behavior for datasets versus source/EDA evidence. Tests use temporary synthetic data;
large local dataset behavior and runtime remain for your run to establish.
