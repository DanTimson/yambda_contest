#!/usr/bin/env python3
"""Past-only listen diagnostics and candidate benchmark; no training or downloads."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import explore_yambda as base
import explore_trends as core
import benchmark_candidates as old
import matplotlib.pyplot as plt

DAY = base.DAY
VERSION = 'listens-eda-v1'
COLS = ['uid', 'item_id', 'timestamp', 'is_organic', 'played_ratio_pct', 'track_length_seconds']
KEYS = ['uid', 'item_id']


def population_mask(ids, seed, permille):
    """Stable SplitMix64 user sample, independent of likes, activity and targets."""
    x = np.asarray(ids, dtype=np.uint64)
    with np.errstate(over='ignore'):
        x = x + np.uint64(seed) + np.uint64(0x9e3779b97f4a7c15)
        x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
        x = x ^ (x >> 31)
    return x % np.uint64(1000) < permille


def extract_cache(path, users, lo, hi, cache_dir, seed, permille, batch_size=262144):
    """One bounded-batch scan of the large file; atomically commit a reusable subset."""
    path = Path(path); st = path.stat()
    signature = dict(version=VERSION, source=str(path.resolve()), bytes=st.st_size,
                     mtime_ns=st.st_mtime_ns, users=sorted(users), lo=lo, hi=hi,
                     seed=seed, population_permille=permille)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(cache_dir) / ('listens_' + key)
    meta = root / 'manifest.json'; dest = root / 'events.parquet'
    if meta.exists() and dest.exists():
        base.log(f'Using listen cache: {root}')
        return dest, json.loads(meta.read_text())
    root.mkdir(parents=True, exist_ok=True)
    part = root / 'events.parquet.incomplete'
    pf = pq.ParquetFile(path)
    missing = set(COLS) - set(pf.schema_arrow.names)
    if missing: raise ValueError(f'Missing listen fields: {sorted(missing)}')
    writer = None; chunks = []; buffered = 0; scanned = 0; kept = 0
    invalid = 0; started = time.perf_counter(); last_log = started
    try:
        for batch in pf.iter_batches(batch_size=batch_size, columns=COLS):
            df = pl.from_arrow(batch); scanned += df.height
            df = df.filter((pl.col('timestamp') >= lo) & (pl.col('timestamp') < hi))
            ids = df['uid'].to_numpy()
            cohort = np.isin(ids, users)
            sample = population_mask(ids, seed, permille)
            df = df.with_columns(pl.Series('cohort', cohort), pl.Series('trend_sample', sample))
            df = df.filter(pl.col('cohort') | pl.col('trend_sample'))
            bad = df.select((pl.col('played_ratio_pct').is_null() | (pl.col('played_ratio_pct') < 0) |
                             pl.col('is_organic').is_null() | ~pl.col('is_organic').is_in([0, 1])).sum()).item()
            invalid += int(bad)
            if bad: raise ValueError('Null/negative play ratios or unknown is_organic values in retained data')
            if df.height:
                # Keep >100% ratios and short plays; neither is filtered away.
                df = df.with_columns(pl.col('uid', 'item_id', 'timestamp').cast(pl.Int64))
                chunks.append(df.to_arrow()); buffered += df.height; kept += df.height
            if buffered >= 524288:
                table = pa.concat_tables(chunks)
                if writer is None: writer = pq.ParquetWriter(part, table.schema, compression='zstd')
                writer.write_table(table, row_group_size=262144); chunks = []; buffered = 0
            now = time.perf_counter()
            if now - last_log >= 20:
                base.log(f'LISTEN SCAN {scanned:,}/{pf.metadata.num_rows:,} rows; kept {kept:,}; elapsed {now-started:.0f}s')
                last_log = now
        if chunks:
            table = pa.concat_tables(chunks)
            if writer is None: writer = pq.ParquetWriter(part, table.schema, compression='zstd')
            writer.write_table(table, row_group_size=262144)
        if writer is None:
            schema = pa.schema([(c, pa.int64()) for c in COLS] + [('cohort', pa.bool_()), ('trend_sample', pa.bool_())])
            pq.write_table(pa.Table.from_batches([], schema=schema), part)
    finally:
        if writer is not None: writer.close()
    part.replace(dest)
    info = dict(signature=signature, scanned_rows=scanned, retained_rows=kept,
                invalid_rows=invalid, seconds=time.perf_counter()-started, cache_bytes=dest.stat().st_size)
    temp = meta.with_suffix('.incomplete'); temp.write_text(json.dumps(info, indent=2)); temp.replace(meta)
    base.log(f'LISTEN CACHE ready: {kept:,} retained rows; {root}')
    return dest, info


def pair_stats(events, cut, lookback):
    q = pl.scan_parquet(events).filter(pl.col('cohort') & (pl.col('timestamp') < cut) &
                                     (pl.col('timestamp') >= cut-lookback*DAY))
    return base.collect(q.group_by(KEYS).agg(
        pl.len().alias('plays'), pl.col('timestamp').max().alias('last_ts'),
        (pl.col('played_ratio_pct') > 50).sum().alias('qualified'),
        (pl.col('played_ratio_pct') >= 90).sum().alias('near_full'),
        (pl.col('played_ratio_pct') <= 20).sum().alias('short'),
        (pl.col('played_ratio_pct') > 100).sum().alias('replays'),
        (pl.col('is_organic') == 1).sum().alias('organic'),
        ((pl.col('is_organic') == 1) & (pl.col('played_ratio_pct') > 50)).sum().alias('organic_qualified'),
        pl.col('timestamp').filter(pl.col('played_ratio_pct') > 50).max().alias('last_qualified_ts'),
        pl.col('timestamp').filter((pl.col('is_organic') == 1) & (pl.col('played_ratio_pct') > 50)).max().alias('last_organic_ts'),
        pl.col('timestamp').filter(pl.col('played_ratio_pct') >= 90).max().alias('last_full_ts'),
        pl.col('timestamp').filter(pl.col('played_ratio_pct') <= 20).max().alias('last_short_ts'),
        (pl.col('timestamp') >= cut-7*DAY).sum().alias('plays7'),
    ))


def feedback_pairs(data, users, cut, history):
    result = history.group_by(KEYS).agg(pl.col('timestamp').max().alias('last_like'))
    for name in ['dislikes', 'undislikes', 'unlikes']:
        path = Path(data) / (name + '.parquet')
        if not path.exists(): raise FileNotFoundError(path)
        df = base.collect(pl.scan_parquet(path).filter(pl.col('uid').is_in(users) & (pl.col('timestamp') < cut))
                          .group_by(KEYS).agg(pl.col('timestamp').max().alias('last_' + name)))
        df = df.with_columns(pl.col(KEYS).cast(pl.Int64))
        result = result.join(df, on=KEYS, how='full', coalesce=True)
    # Strict inequalities: same-time actions have unknown ordering.
    return result.with_columns(
        (pl.col('last_dislikes').is_not_null() &
         (pl.col('last_undislikes').is_null() | (pl.col('last_dislikes') > pl.col('last_undislikes')))).alias('active_dislike'),
        (pl.col('last_unlikes').is_not_null() &
         (pl.col('last_like').is_null() | (pl.col('last_unlikes') > pl.col('last_like')))).alias('active_unlike'),
    )


def listen_lists(pairs, cut, budget):
    """Historical fields only. No permanent exclusion of previously liked tracks."""
    specs = {
        'listen_recent': (pl.col('plays') > 0, pl.col('last_ts').cast(pl.Float64)),
        'listen_qualified': (pl.col('qualified') > 0, pl.col('qualified').cast(pl.Float64).log1p() *
                             (-(cut-pl.col('last_qualified_ts'))/(30*DAY)).exp()),
        'listen_organic': (pl.col('organic_qualified') > 0, pl.col('organic_qualified').cast(pl.Float64).log1p() *
                           (-(cut-pl.col('last_organic_ts'))/(30*DAY)).exp()),
        'listen_repeated_full': (pl.col('near_full') >= 2, pl.col('near_full').cast(pl.Float64).log1p() *
                                 (-(cut-pl.col('last_full_ts'))/(30*DAY)).exp()),
    }
    lists = {}
    for name, (mask, score) in specs.items():
        ordered = (pairs.filter(mask).with_columns(score.alias('score'))
                   .sort(['uid', 'score', 'item_id'], descending=[False, True, False])
                   .group_by('uid', maintain_order=True).head(budget))
        lists[name] = base.user_lists(ordered)
    return lists


def trend_lists(events, cut, budget, min_support):
    q = pl.scan_parquet(events).filter(pl.col('trend_sample') & (pl.col('timestamp') < cut) &
                                      (pl.col('timestamp') >= cut-14*DAY))
    bins = pl.when(pl.col('timestamp') >= cut-7*DAY).then(pl.lit('current')).otherwise(pl.lit('previous'))
    df = base.collect(q.group_by('item_id', bins.alias('period')).agg(
        pl.len().alias('plays'),
        pl.col('uid').n_unique().alias('listeners'),
        pl.col('uid').filter(pl.col('played_ratio_pct') > 50).n_unique().alias('qualified_listeners'),
        pl.col('uid').filter((pl.col('played_ratio_pct') > 50) & (pl.col('is_organic') == 1)).n_unique().alias('organic_listeners'),
    ))
    cols = ['plays', 'listeners', 'qualified_listeners', 'organic_listeners']
    cur = df.filter(pl.col('period') == 'current').drop('period')
    prev = df.filter(pl.col('period') == 'previous').drop('period').rename({c: 'prev_' + c for c in cols})
    stats = cur.join(prev, on='item_id', how='full', coalesce=True).fill_null(0)
    stats = stats.with_columns((((pl.col('qualified_listeners')+2.)/(pl.col('prev_qualified_listeners')+2.)).log() *
                               pl.col('qualified_listeners').cast(pl.Float64).log1p()).alias('growth'))
    specs = {'sample_listeners7': 'listeners', 'sample_qualified7': 'qualified_listeners',
             'sample_organic7': 'organic_listeners', 'sample_growth7': 'growth'}
    lists = {}
    for name, field in specs.items():
        f = stats.filter(pl.col(field) > 0)
        if field == 'growth': f = f.filter(pl.col('qualified_listeners') >= min_support)
        lists[name] = f.sort([field, 'item_id'], descending=[True, False]).head(budget)['item_id'].to_list()
    return stats, lists


def baseline_lists(run, cut, users, budget):
    if not run: return None
    for meta_path in sorted((Path(run)/'pairs').glob('*.json')):
        meta = json.loads(meta_path.read_text())
        if meta['cutoff'] != cut: continue
        if not set(users).issubset(meta['users']): raise ValueError('Baseline cache does not cover selected users')
        path = meta_path.with_suffix('.parquet')
        if not path.exists(): raise FileNotFoundError(f'Baseline candidate cache missing: {path}')
        frame = base.collect(pl.scan_parquet(path).filter(pl.col('uid').is_in(users) &
                             (pl.col('candidate_rank') <= budget)).select('uid', 'item_id', 'candidate_rank'))
        return base.user_lists(frame.sort(['uid', 'candidate_rank']))
    base.log(f'No old candidate snapshot at cutoff={cut}; computing fresh likes control instead.')
    return None


def metrics(lists, truth, users, complete, cut, name, oracle=False):
    scores = []; coverage = []; sizes = []
    for u in users:
        t = set(truth.get(u, [])); items = lists.get(u, []); d = max(1, len(t))
        scores.append(len(t & set(items[:100]))/d); coverage.append(len(t & set(items))/d); sizes.append(len(items))
    scores = np.array(scores); coverage = np.array(coverage)
    active = np.array([bool(truth.get(u)) for u in users]); full = np.array([u in complete for u in users])
    def mean(x, mask): return float(x[mask].mean()) if mask.any() else None
    row = dict(cutoff=cut, source=name, users=len(users), active_users=int(active.sum()), full100_users=int(full.sum()),
               mean_candidates=float(np.mean(sizes)), recall_all=None if oracle else float(scores.mean()),
               recall_active=None if oracle else mean(scores, active), recall_full100=None if oracle else mean(scores, full),
               coverage_active=mean(coverage, active), coverage_full100=mean(coverage, full))
    return row, scores, coverage


def audit(pairs, feedback, target, cut, users, report):
    truth_pairs = target.select(KEYS).unique().with_columns(pl.lit(True).alias('future_like'))
    f = (pairs.join(feedback, on=KEYS, how='left').join(truth_pairs, on=KEYS, how='left')
         .with_columns(pl.col('future_like', 'active_dislike', 'active_unlike').fill_null(False)))
    groups = {
        'all_listened_pairs': pl.lit(True),
        'short_only_no_past_like': (pl.col('short') == pl.col('plays')) & pl.col('last_like').is_null(),
        'no_qualified_no_past_like': (pl.col('qualified') == 0) & pl.col('last_like').is_null(),
        'qualified_no_past_like': (pl.col('qualified') > 0) & pl.col('last_like').is_null(),
        'repeated_near_full_no_past_like': (pl.col('near_full') >= 2) & pl.col('last_like').is_null(),
        'mixed_short_and_qualified': (pl.col('short') > 0) & (pl.col('qualified') > 0),
        'active_dislike': pl.col('active_dislike'),
        'active_unlike': pl.col('active_unlike'),
        'past_like': pl.col('last_like').is_not_null(),
    }
    rows = []
    for name, expr in groups.items():
        g = f.filter(expr); hits = int(g['future_like'].sum())
        rows.append(dict(group=name, pairs=g.height, users=g['uid'].n_unique(), future_positive_pairs=hits,
                         observed_next100_positive_fraction=hits/g.height if g.height else None))
    report.table(f'negative_audit_{cut}', pl.DataFrame(rows), f'Potential negative-label contradictions — {cut}',
                 'Overlapping groups. Future positive = membership in observed next-100 likes, not conversion per exposure. '
                 'Unobserved later likes and unequal activity create censoring. These are label-risk diagnostics, not causal effects. '
                 'Dislike/unlike rows cover listened pairs in this lookback only; same-time cancel actions are ambiguous.')
    target_join = target.unique(KEYS, keep='first', maintain_order=True).join(pairs, on=KEYS, how='left')
    joined_history = target_join.join(feedback.select(KEYS + ['last_like']), on=KEYS, how='left')
    masks = {'any_prior_listen': pl.col('plays').is_not_null(),
             'qualified_prior_listen': pl.col('qualified') > 0,
             'repeated_near_full': pl.col('near_full') >= 2,
             'short_only': pl.col('short') == pl.col('plays'),
             'prior_listen_no_prior_like': pl.col('plays').is_not_null() & pl.col('last_like').is_null()}
    support = []
    counts = target.group_by('uid').len(); full = set(counts.filter(pl.col('len') == 100)['uid'].to_list())
    truth = base.user_lists(target)
    for name, mask in masks.items():
        covered = base.user_lists(joined_history.filter(mask))
        m, _, _ = metrics(covered, truth, users, full, cut, name, oracle=True)
        support.append(dict(support=name, fraction_active=m['coverage_active'], fraction_full100=m['coverage_full100']))
    report.table(f'target_support_{cut}', pl.DataFrame(support), f'How much of the target is already in listening history? — {cut}',
                 'Unbounded historical support, not a retrievable fixed-budget score. Same next-100-like-event target as the candidate benchmark.')
    lag = target_join.filter(pl.col('last_ts').is_not_null()).select(((pl.col('timestamp')-pl.col('last_ts'))/DAY).alias('lag_days'))
    quantiles = [{'quantile': q, 'days_from_last_pre_cutoff_listen_to_future_like': float(lag['lag_days'].quantile(q))}
                 for q in [.1, .25, .5, .75, .9]] if lag.height else []
    if quantiles:
        report.table(f'lag_{cut}', pl.DataFrame(quantiles), f'Delay to target likes — {cut}',
                     'Last listen before cutoff, not last listen before the like. Includes waiting from cutoff to target and excludes targets without prior listening.')
    return rows, support


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--users', default='input/test_users.csv')
    p.add_argument('--out', required=True); p.add_argument('--cache-dir', default='data/eda_cache')
    p.add_argument('--max-users', type=int, default=5000); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--future-days', type=int, nargs='+', default=[120, 90, 60, 30])
    p.add_argument('--lookback-days', type=int, default=90); p.add_argument('--budget', type=int, default=1000)
    p.add_argument('--population-permille', type=int, default=10); p.add_argument('--min-listeners', type=int, default=10)
    p.add_argument('--artist-catalog', type=int, default=200000); p.add_argument('--baseline-run')
    a = p.parse_args()
    if a.seed < 0 or a.budget < 100 or a.lookback_days < 14 or not 1 <= a.population_permille <= 1000 or min(a.future_days) <= 0 or a.min_listeners < 1:
        p.error('Require seed>=0, budget>=100, lookback>=14, sample permille in [1,1000], positive horizons/support')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): p.error('Choose a new empty output directory')
    for name in ['likes', 'listens', 'dislikes', 'unlikes', 'undislikes']:
        if not (Path(a.data)/(name+'.parquet')).exists(): p.error(f'Missing {name}.parquet')
    started = time.perf_counter(); ctx = core.load_context(a.data, a.users, a.max_users, a.seed)
    cuts = sorted(set(ctx.end-d*DAY for d in a.future_days))
    if min(cuts) <= ctx.start: p.error('Cutoff must be later than dataset start')
    events, cache = extract_cache(ctx.data/'listens.parquet', ctx.users,
                                  min(cuts)-a.lookback_days*DAY, max(cuts),
                                  a.cache_dir, a.seed, a.population_permille)
    r = core.new_report(out)
    r.prose('Questions for this experiment', 'Can prior listening retrieve likes missing from the current likes-only candidates? '
            'Do distinct-listener and organic trends help? Which apparent negatives later become positives? '
            'No model is trained. Organic listens are observational, not randomized control observations.')
    base.inventory(ctx.data, r)
    overview = base.collect(pl.scan_parquet(events).group_by('cohort', 'trend_sample').agg(
        pl.len().alias('events'), pl.col('uid').n_unique().alias('users'), pl.col('item_id').n_unique().alias('items'),
        (pl.col('played_ratio_pct') <= 20).mean().alias('short_fraction'),
        (pl.col('played_ratio_pct') > 50).mean().alias('qualified_fraction'),
        (pl.col('played_ratio_pct') > 100).mean().alias('over100_fraction'),
        pl.col('is_organic').mean().alias('organic_fraction')))
    r.table('cache_overview', overview, 'Cached listening coverage',
            'Cohort and population-sample membership can overlap. Counts cover the extraction interval, not the entire dataset. '
            'Population users are sampled independently by UID hash; 10 permille means 1% of user IDs in expectation.')
    all_rows = []; audits = []; supports = []; paired = []; overlap = []; source_details = []
    for cut in cuts:
        base.log(f'LISTEN EDA cutoff={cut}; future days={(ctx.end-cut)/DAY:.0f}')
        history, target = core.history_targets(ctx, cut)
        truth = base.user_lists(target)
        complete = set(target.group_by('uid').len().filter(pl.col('len') == 100)['uid'].to_list())
        pairs = pair_stats(events, cut, a.lookback_days)
        feedback = feedback_pairs(ctx.data, ctx.users, cut, history)
        ll = listen_lists(pairs, cut, a.budget)
        trends, tl = trend_lists(events, cut, a.budget, a.min_listeners)
        prior = baseline_lists(a.baseline_run, cut, ctx.users, a.budget)
        # Global likes controls reuse existing past-only statistics caches.
        stats, _ = old.cached_stats(ctx.data, cut, a.cache_dir)
        gl = old.global_lists(stats, a.budget, 10)
        if prior is None:
            artists = old.ArtistSource(ctx.data, history, stats, cut, a.artist_catalog)
            prior = {}
            global_family = core.fuse(list(gl.values()), a.budget)
            for u in ctx.users:
                al = artists.recommend(u, a.budget)
                prior[u] = core.fuse([global_family, core.fuse(list(al.values()), a.budget)], a.budget)
            baseline_kind = 'fresh_hierarchical_rrf'
        else: baseline_kind = 'cached_candidate_rank_order'
        methods = {'likes_control': prior, 'like_pop2': {u: gl['pop2'] for u in ctx.users}, **ll}
        methods.update({name: {u: items for u in ctx.users} for name, items in tl.items()})
        methods['listens_rrf'] = {u: core.fuse([v.get(u, []) for v in ll.values()], a.budget) for u in ctx.users}
        methods['likes_plus_listens_rrf'] = {u: core.fuse([prior.get(u, []), methods['listens_rrf'][u]], a.budget) for u in ctx.users}
        methods['likes_listens_trends_rrf'] = {u: core.fuse([prior.get(u, []), methods['listens_rrf'][u], tl['sample_qualified7']], a.budget) for u in ctx.users}
        methods['union_uncapped_oracle'] = {u: sorted(set(prior.get(u, [])) | set(methods['listens_rrf'][u])) for u in ctx.users}
        active = np.array([bool(truth.get(u)) for u in ctx.users]); scores = {}; covers = {}; rows = []
        for name, lists in methods.items():
            row, score, cov = metrics(lists, truth, ctx.users, complete, cut, name, oracle=name.endswith('oracle'))
            rows.append(row); scores[name] = score; covers[name] = cov
        for name in ['listens_rrf', 'likes_plus_listens_rrf', 'likes_listens_trends_rrf']:
            paired.append(dict(cutoff=cut, source=name, **core.paired_interval((scores[name]-scores['likes_control'])[active], ctx.seed)))
        for u in ctx.users:
            t = set(truth.get(u, [])); prior_set = set(prior.get(u, [])); new_set = set(methods['listens_rrf'][u])
            overlap.append(dict(cutoff=cut, uid=u, target_unique=len(t), full100=u in complete,
                                likes_candidates=len(prior_set), listen_candidates=len(new_set),
                                candidate_overlap=len(prior_set & new_set), listen_only_hits=len(t & (new_set-prior_set)),
                                likes_only_hits=len(t & (prior_set-new_set))))
        audit_rows, support = audit(pairs, feedback, target, cut, ctx.users, r)
        audits.extend(dict(cutoff=cut, **v) for v in audit_rows); supports.extend(dict(cutoff=cut, **v) for v in support)
        source_details.append(dict(cutoff=cut, baseline_kind=baseline_kind, observed_targets_end=ctx.end,
                                   past_pairs=pairs.height, population_min_listeners=a.min_listeners))
        r.table(f'candidate_metrics_{cut}', pl.DataFrame(rows), f'Fixed-budget candidate comparisons — {cut}',
                'Recall = top100; coverage = top budget. Empty lists are not padded with popularity. '
                'Union has a larger budget and is only an opportunity ceiling. All/active/full100 describe different censored cohorts. '
                'Cached old candidates are reused as lists only; old validation labels and scores are not compared.')
        r.table(f'trending_tracks_{cut}', trends.sort(['qualified_listeners','item_id'], descending=[True,False]).head(50),
                f'Inspect sample listening trends — {cut}', 'Counts are from the same independent user sample, not the full population. Growth support is sample listeners, not population-scaled counts.')
        chosen = [v for v in rows if not v['source'].endswith('oracle')]
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax, field, label in zip(axes, ['recall_active', 'coverage_active'], ['Recall@100, active (%)', f'Coverage@{a.budget}, active (%)']):
            ax.barh([v['source'] for v in chosen], [100*(v[field] or 0) for v in chosen]); ax.invert_yaxis(); ax.set_xlabel(label)
        r.figure(f'candidates_{cut}', fig, f'What does listening add? — {cut}', 'Compare top100 ranking and candidate coverage separately. RRF is a diagnostic control, not a trained ranking model.')
        if trends.height:
            rising = trends.filter(pl.col('qualified_listeners') >= a.min_listeners).sort(['growth', 'item_id'], descending=[True, False]).head(15)
            if rising.height:
                fig, ax = plt.subplots(figsize=(10, 6)); y = np.arange(rising.height)
                ax.barh(y-.18, rising['prev_qualified_listeners'], height=.36, label='Previous 7 days')
                ax.barh(y+.18, rising['qualified_listeners'], height=.36, label='Latest 7 days')
                ax.set_yticks(y, [str(i) for i in rising['item_id']]); ax.invert_yaxis()
                ax.set_xlabel('Distinct qualifying listeners in fixed user sample'); ax.set_ylabel('item_id'); ax.legend()
                r.figure(f'listener_growth_{cut}', fig, f'Rising tracks: audience growth versus repeat volume — {cut}',
                         'Ranked by smoothed log growth with sample support threshold. Each user counts once per track per week; repeat plays cannot inflate this count.')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh([v['support'] for v in support], [100*(v['fraction_active'] or 0) for v in support])
        ax.invert_yaxis(); ax.set_xlabel('Mean fraction of next-100 targets, active users (%)')
        r.figure(f'historical_support_{cut}', fig, f'Targets with prior listening evidence — {cut}',
                 'Groups overlap. Unbounded support is an opportunity ceiling, not Recall@100.')
        all_rows.extend(rows)
        del pairs, feedback, methods, trends, stats
        gc.collect()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ['likes_control', 'listens_rrf', 'likes_plus_listens_rrf', 'likes_listens_trends_rrf']:
        selected = sorted([v for v in all_rows if v['source'] == name], key=lambda v: v['cutoff'])
        x = [(ctx.end-v['cutoff'])/DAY for v in selected]
        for ax, field in zip(axes, ['recall_active', 'coverage_active']):
            ax.plot(x, [100*v[field] if v[field] is not None else np.nan for v in selected], marker='o', label=name)
    for ax, label in zip(axes, ['Recall@100, active (%)', f'Coverage@{a.budget}, active (%)']):
        ax.set_xlabel('Days of future observation'); ax.set_ylabel(label); ax.legend(fontsize=8)
    r.figure('temporal_comparison', fig, 'Does improvement persist across time?',
             'Different horizons contain differently censored target cohorts. Trends over horizons are not learning curves.')
    r.table('all_candidate_metrics', pl.DataFrame(all_rows), 'All temporal snapshots')
    r.table('paired_vs_likes', pl.DataFrame(paired), 'Paired Recall@100 changes versus likes control',
            'Active-user bootstrap intervals are conditional on each time snapshot; they do not measure uncertainty about the hidden test.')
    r.table('user_complementarity', pl.DataFrame(overlap), 'Per-user complementarity', limit=10)
    r.summary.update(version=VERSION, arguments=vars(a), seed=ctx.seed, user_file_sha256=ctx.user_sha256,
                     sampled_users=ctx.users, target='next100 chronological like events; distinct-item recall; repeated past likes allowed',
                     cache=cache, metrics=all_rows, negative_audits=audits, target_support=supports,
                     paired=paired, source_details=source_details, seconds=time.perf_counter()-started)
    r.finish()


if __name__ == '__main__': main()
