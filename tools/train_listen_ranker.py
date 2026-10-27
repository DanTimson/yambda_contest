#!/usr/bin/env python3
"""Cached-listen candidate/feature ablation with chronological CatBoost ranking."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostRanker, Pool
import matplotlib.pyplot as plt
import explore_yambda as base
import explore_trends as core
import benchmark_candidates as bench
import train_candidate_ranker as old
import explore_listens as listening

VERSION = 'listen-ranker-v1'
LISTEN_COUNTS = ['plays', 'qualified', 'near_full', 'short', 'organic', 'organic_qualified', 'replays', 'plays7']
LISTEN_TIMES = ['last_ts', 'last_qualified_ts', 'last_organic_ts', 'last_full_ts', 'last_short_ts']
TREND_COLS = ['listeners', 'qualified_listeners', 'organic_listeners', 'prev_listeners',
              'prev_qualified_listeners', 'prev_organic_listeners', 'growth']
EXTRA = ['log_listen_' + c for c in LISTEN_COUNTS] + ['listen_' + c + '_age_days' for c in LISTEN_TIMES] + [
    'listen_qualified_fraction', 'listen_short_fraction', 'listen_organic_fraction', 'listen_full_fraction',
    'active_dislike', 'active_unlike', 'dislike_age_days', 'unlike_age_days'] + [
    'sample_log_' + c if c != 'growth' else 'sample_growth' for c in TREND_COLS] + [
    name + '_rr' for name in ['listen_recent', 'listen_qualified', 'listen_organic', 'listen_repeated_full']] + [
    'listen_family_rr', 'combined_rr', 'in_like_pool', 'in_listen_pool']
FEATURES = old.FEATURES + EXTRA
SAMPLE_FEATURES = ['sample_log_' + c if c != 'growth' else 'sample_growth' for c in TREND_COLS]
MODES = {'base_features': old.FEATURES,
         'listen_no_sample_trends': [c for c in FEATURES if c not in SAMPLE_FEATURES],
         'listen_features': FEATURES}


def log(message):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    base.log(f'{time.strftime("%H:%M:%S")} {message}; peakRSS={peak:.2f} GiB')


def resolve_cache(eda_run, cache_dir, ctx, cuts, lookback):
    """Require the existing subset; never trigger a surprise full-listens scan."""
    report = json.loads((Path(eda_run)/'summary.json').read_text())
    sig = report['cache']['signature']
    if sig['version'] != listening.VERSION: raise ValueError('Unsupported listen-cache extraction version')
    if not set(ctx.users).issubset(sig['users']): raise ValueError('Listen cache does not cover this user cohort')
    if min(cuts)-lookback*core.DAY < sig['lo'] or max(cuts) > sig['hi']:
        raise ValueError('Required listening windows exceed the existing cache; do not silently shorten histories')
    src = ctx.data/'listens.parquet'; st = src.stat()
    if (sig['source'], sig['bytes'], sig['mtime_ns']) != (str(src.resolve()), st.st_size, st.st_mtime_ns):
        raise ValueError('Listen source fingerprint differs from the EDA cache')
    key = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:24]
    folder = Path(cache_dir)/('listens_'+key)
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['signature'] != sig: raise ValueError('Listen manifest mismatch')
    path = folder/'events.parquet'
    if not path.exists(): raise FileNotFoundError(path)
    return path, manifest


class LikeFeatures:
    """Same past-only features as the original model, extended to arbitrary items."""
    def __init__(self, ctx, cut, users, cache_dir, artist_catalog, source_budget=1000):
        self.cut = cut; self.source_budget = source_budget
        self.history = ctx.likes.filter((pl.col('timestamp') < cut) & pl.col('uid').is_in(users))
        stats, _ = bench.cached_stats(ctx.data, cut, cache_dir)
        self.lists = bench.global_lists(stats, source_budget, 10)
        self.global_order = core.fuse(list(self.lists.values()), sum(map(len, self.lists.values())))
        self.global_ranks = {int(i): r for r, i in enumerate(self.global_order, 1)}
        self.global_maps = {k: {int(i): r for r, i in enumerate(v, 1)} for k, v in self.lists.items()}
        self.artists = bench.ArtistSource(ctx.data, self.history, stats, cut, artist_catalog)
        self.index = {int(i): j for j, i in enumerate(stats['item_id'])}
        self.values = stats.select(old.STAT_COLS).to_numpy().astype(np.float32)
        self.histories = {int(u): g for (u,), g in self.history.group_by('uid')}

    def user_sources(self, u):
        personal = self.artists.recommend(u, self.source_budget)
        artist_order = core.fuse(list(personal.values()), sum(map(len, personal.values())))
        full_order = core.fuse([self.global_order, artist_order], len(self.global_order)+len(artist_order))
        return personal, artist_order, full_order

    def features(self, u, ids, sources):
        personal, artist_order, full_order = sources
        n_items = len(ids)
        v = np.zeros((n_items, len(old.STAT_COLS)), np.float32)
        v[:, 8] = 1.  # Neutral ratio7 for tracks with no earlier likes.
        known = [(r, self.index[int(i)]) for r, i in enumerate(ids) if int(i) in self.index]
        if known:
            dst, src = zip(*known); v[list(dst)] = self.values[list(src)]
        v[:, :8] = np.log1p(v[:, :8]); v[:, 11] = (self.cut-v[:, 11])/core.DAY
        for r, i in enumerate(ids):
            if int(i) not in self.index: v[r, 11] = -1.
        h = self.histories.get(u); n = h.height if h is not None else 0
        recent = float((h['timestamp'] >= self.cut-30*core.DAY).mean()) if n else 0.
        recency = (self.cut-int(h['timestamp'].max()))/core.DAY if n else -1.
        seen = {int(i): (int(c), int(t)) for i, c, t in h.group_by('item_id').agg(pl.len(), pl.col('timestamp').max()).iter_rows()} if n else {}
        user = np.empty((n_items, 6), np.float32)
        user[:, 0] = np.log1p(n); user[:, 1] = recent; user[:, 2] = recency
        user[:, 3] = [float(int(i) in seen) for i in ids]
        user[:, 4] = [(self.cut-seen[int(i)][1])/core.DAY if int(i) in seen else -1. for i in ids]
        user[:, 5] = [np.log1p(seen.get(int(i), (0, 0))[0]) for i in ids]
        maps = {**self.global_maps, **{k: {int(i): r for r, i in enumerate(values, 1)} for k, values in personal.items()}}
        channels = np.asarray([[1/maps[k][int(i)] if int(i) in maps[k] else 0. for k in old.CHANNELS] for i in ids], np.float32)
        ar = {int(i): r for r, i in enumerate(artist_order, 1)}
        fr = {int(i): r for r, i in enumerate(full_order, 1)}
        rr = np.asarray([[1/self.global_ranks[int(i)] if int(i) in self.global_ranks else 0.,
                          1/ar[int(i)] if int(i) in ar else 0., 1/fr[int(i)] if int(i) in fr else 0.,
                          float(np.count_nonzero(channels[r]))] for r, i in enumerate(ids)], np.float32)
        return np.concatenate([v, user, channels, rr], axis=1)


class ListenFeatures:
    def __init__(self, events, ctx, cut, users, history, lookback, source_budget):
        self.cut = cut
        log(f'BUILD listen aggregates cutoff={cut}')
        pairs = listening.pair_stats(events, cut, lookback).filter(pl.col('uid').is_in(users))
        self.lists = listening.listen_lists(pairs, cut, source_budget)
        self.pairs = {int(u): g for (u,), g in pairs.group_by('uid')}
        feedback = listening.feedback_pairs(ctx.data, users, cut, history)
        self.feedback = {int(u): g for (u,), g in feedback.group_by('uid')}
        log(f'BUILD sample trends cutoff={cut}')
        trend, _ = listening.trend_lists(events, cut, source_budget, 10)
        self.trend_index = {int(i): j for j, i in enumerate(trend['item_id'])}
        self.trend_values = trend.select(TREND_COLS).to_numpy().astype(np.float32)
        self.trend_values[:, :6] = np.log1p(self.trend_values[:, :6])

    def features(self, u, ids, combined):
        g = self.pairs.get(u)
        lookup = {int(row['item_id']): row for row in g.iter_rows(named=True)} if g is not None else {}
        fb = self.feedback.get(u)
        feedback = {int(row['item_id']): row for row in fb.iter_rows(named=True)} if fb is not None else {}
        maps = {name: {int(i): r for r, i in enumerate(values.get(u, []), 1)} for name, values in self.lists.items()}
        family = core.fuse([values.get(u, []) for values in self.lists.values()], 1000)
        family_map = {int(i): r for r, i in enumerate(family, 1)}
        out = np.zeros((len(ids), len(EXTRA)), np.float32)
        for r, item in enumerate(ids):
            item = int(item); p = lookup.get(item, {}); f = feedback.get(item, {})
            count = float(p.get('plays', 0)); den = max(1., count)
            row = [np.log1p(p.get(c, 0)) for c in LISTEN_COUNTS]
            row += [(self.cut-p[c])/core.DAY if p.get(c) is not None else -1. for c in LISTEN_TIMES]
            row += [p.get(c, 0)/den for c in ['qualified', 'short', 'organic', 'near_full']]
            row += [float(f.get('active_dislike') or False), float(f.get('active_unlike') or False)]
            row += [(self.cut-f[c])/core.DAY if f.get(c) is not None else -1. for c in ['last_dislikes', 'last_unlikes']]
            row += self.trend_values[self.trend_index[item]].tolist() if item in self.trend_index else [0.]*len(TREND_COLS)
            row += [1/m[item] if item in m else 0. for m in maps.values()]
            row += [1/family_map[item] if item in family_map else 0., 1/(r+1),
                    float(item in combined['like']), float(item in combined['listen'])]
            out[r] = row
        return out


def source_rows(ctx, events, cut, users, budget, cache_dir, artist_catalog, lookback):
    likes = LikeFeatures(ctx, cut, users, cache_dir, artist_catalog)
    listens = ListenFeatures(events, ctx, cut, users, likes.history, lookback, 1000)
    for j, u in enumerate(users):
        sources = likes.user_sources(u); like_pool = sources[2][:1000]
        listen_pool = core.fuse([v.get(u, []) for v in listens.lists.values()], 1000)
        ids = np.asarray(core.fuse([like_pool, listen_pool], budget), np.int64)
        if len(ids) < 100: raise ValueError(f'Insufficient historical candidates for uid={u}')
        x = np.concatenate([likes.features(u, ids, sources), listens.features(u, ids, dict(like=set(like_pool), listen=set(listen_pool)))], axis=1)
        if not np.isfinite(x).all(): raise ValueError(f'Non-finite features uid={u}')
        yield u, ids, x, like_pool
        if (j+1) % 500 == 0: log(f'FEATURES cutoff={cut}: {j+1}/{len(users)} users')


def build_pairs(ctx, events, cut, label_end, users, path, budget, cache_dir, artist_catalog, lookback):
    log(f'BUILD cutoff={cut}, labels=[{cut},{label_end}), users={len(users)}')
    target = ctx.likes.filter(pl.col('uid').is_in(users) & (pl.col('timestamp') >= cut) & (pl.col('timestamp') < label_end))
    target = target.group_by('uid', maintain_order=True).head(100)
    truth = base.user_lists(target); complete = set(target.group_by('uid').len().filter(pl.col('len') == 100)['uid'].to_list())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    writer = None; chunks = []; count = 0; baseline = {}; like_baseline = {}; cover = []; full_cover = []
    try:
        for u, ids, x, like_pool in source_rows(ctx, events, cut, users, budget, cache_dir, artist_catalog, lookback):
            true = set(truth.get(u, [])); y = np.asarray([int(i in true) for i in ids], np.int8)
            columns = {name: pa.array(x[:, j]) for j, name in enumerate(FEATURES)}
            columns.update(uid=pa.array(np.full(len(ids), u, np.int64)), item_id=pa.array(ids),
                           candidate_rank=pa.array(np.arange(1, len(ids)+1, dtype=np.int32)), target=pa.array(y),
                           n_true=pa.array(np.full(len(ids), len(true), np.int16)))
            chunks.append(pa.table(columns)); count += len(ids); baseline[u] = ids[:100].tolist(); like_baseline[u] = like_pool[:100]
            if true:
                cover.append(float(y.sum()/len(true)))
                if u in complete: full_cover.append(float(y.sum()/len(true)))
            if len(chunks) >= 16:
                table = pa.concat_tables(chunks)
                if writer is None: writer = pq.ParquetWriter(path, table.schema, compression='zstd')
                writer.write_table(table); chunks = []
        if chunks:
            table = pa.concat_tables(chunks)
            if writer is None: writer = pq.ParquetWriter(path, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None: writer.close()
    meta = dict(cutoff=cut, label_end_exclusive=label_end, users=users, truth=truth, complete=sorted(complete),
                candidate_pairs=count, candidate_recall_active=float(np.mean(cover)) if cover else None,
                candidate_recall_full100=float(np.mean(full_cover)) if full_cover else None,
                rrf=old.metric(baseline, truth, users, complete), likes_rrf=old.metric(like_baseline, truth, users, complete))
    path.with_suffix('.json').write_text(json.dumps(meta))
    log(f'PAIRS ready rows={count:,}, candidate recall={meta["candidate_recall_active"]}')
    return meta


def predict(model, path, features, trees):
    best = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=['uid', 'item_id']+features):
        f = pl.from_arrow(batch)
        scores = model.predict(f.select(features).to_numpy(), ntree_end=trees)
        f = f.select('uid', 'item_id').with_columns(pl.Series('score', scores))
        f = f.sort(['uid', 'score', 'item_id'], descending=[False, True, False]).group_by('uid', maintain_order=True).head(100)
        for (u,), g in f.group_by('uid'):
            values = best.get(int(u), []) + g.select('item_id', 'score').rows()
            best[int(u)] = sorted(values, key=lambda p: (-p[1], p[0]))[:100]
    return {u: [int(i) for i, s in values] for u, values in best.items()}


def evaluate(model, path, meta, features, trees):
    truth = {int(k): v for k, v in meta['truth'].items()}
    pred = predict(model, path, features, trees)
    return old.metric(pred, truth, meta['users'], set(meta['complete'])), pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--users', default='input/test_users.csv')
    p.add_argument('--listen-eda-run', required=True); p.add_argument('--out', required=True)
    p.add_argument('--cache-dir', default='data/eda_cache'); p.add_argument('--max-users', type=int, default=5000)
    p.add_argument('--train-users', type=int, default=5000); p.add_argument('--budget', type=int, default=2000)
    p.add_argument('--train-future-days', type=int, default=90); p.add_argument('--val-future-days', type=int, default=60)
    p.add_argument('--eval-future-days', type=int, default=30); p.add_argument('--lookback-days', type=int, default=90)
    p.add_argument('--iterations', nargs='+', type=int, default=[200, 400, 600]); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--artist-catalog', type=int, default=200000); p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(); out = Path(a.out)
    if out.exists() and any(out.iterdir()): p.error('Choose a new empty output directory')
    if not a.train_future_days > a.val_future_days > a.eval_future_days > 0: p.error('Require train > validation > development horizons >0')
    if min(a.iterations) < 1 or a.train_users < 1 or a.budget < 100 or a.budget > 2000 or a.threads < 1 or a.lookback_days < 14:
        p.error('Require positive iterations/users/threads, budget in [100,2000], lookback>=14')
    pa.set_cpu_count(a.threads)
    ctx = core.load_context(a.data, a.users, a.max_users, a.seed)
    cuts = [ctx.end-d*core.DAY for d in [a.train_future_days, a.val_future_days, a.eval_future_days]]
    if min(cuts) <= ctx.start: p.error('Cutoffs precede available history')
    events, manifest = resolve_cache(a.listen_eda_run, a.cache_dir, ctx, cuts, a.lookback_days)
    train_users = sorted(np.random.default_rng(a.seed+1).choice(ctx.users, min(a.train_users, len(ctx.users)), replace=False).tolist())
    report = core.new_report(out); snapshots = {}
    report.prose('Controlled experiment', 'All CatBoost models use exactly the same expanded candidate pool and QuerySoftMax objective. '
                 'base_features uses the original scalar like features; listen_features additionally uses listen history, feedback state, '
                 'source ranks and sample listener trends. listen_no_sample_trends removes only the sample trend features. '
                 'Future likes define all labels; short plays, dislikes and unlikes never override a positive.')
    for name, cut, end, users in [('train', cuts[0], cuts[1], train_users), ('validation', cuts[1], cuts[2], ctx.users), ('development', cuts[2], ctx.end, ctx.users)]:
        snapshots[name] = build_pairs(ctx, events, cut, end, users, out/'pairs'/f'{name}.parquet', a.budget, a.cache_dir, a.artist_catalog, a.lookback_days)
        gc.collect()
    rows = []; validation = []; models = {}; importance = []; chosen = {}
    for mode, features in MODES.items():
        log(f'TRAIN {mode}: loading training columns')
        frame = base.collect(pl.scan_parquet(out/'pairs/train.parquet').filter(pl.col('n_true') > 0).select(features+['target', 'uid', 'n_true']))
        if not frame.height or not frame['target'].sum(): raise ValueError('No retrieved training positives')
        x = frame.select(features).to_numpy().astype(np.float32, copy=False)
        pool = Pool(x, label=frame['target'].to_numpy(), group_id=frame['uid'].to_numpy(),
                    group_weight=(1./frame['n_true'].to_numpy()).astype(np.float32), feature_names=features,
                    thread_count=a.threads)
        del frame, x; gc.collect()
        model = CatBoostRanker(loss_function='QuerySoftMax', iterations=max(a.iterations), depth=6, learning_rate=.05,
                               l2_leaf_reg=10, random_seed=a.seed, thread_count=a.threads, verbose=100,
                               allow_writing_files=False)
        log(f'TRAIN {mode}')
        model.fit(pool)
        del pool; gc.collect()
        model.save_model(out/f'{mode}.cbm')
        for trees in sorted(set(a.iterations)):
            result, _ = evaluate(model, out/'pairs/validation.parquet', snapshots['validation'], features, trees)
            row = dict(mode=mode, iterations=trees, **result); validation.append(row); log(f'VALIDATION {row}')
        selected = max([v for v in validation if v['mode'] == mode], key=lambda v: (v['recall_active'] if v['recall_active'] is not None else -1., -v['iterations']))
        chosen[mode] = selected; models[mode] = model
        importance.extend(dict(mode=mode, feature=feature, importance=float(value)) for feature, value in zip(features, model.get_feature_importance(type='PredictionValuesChange')))
    best = max(chosen.values(), key=lambda v: (v['recall_active'] if v['recall_active'] is not None else -1., -v['iterations']))
    selected_model = models[best['mode']].copy(); selected_model.shrink(best['iterations']); selected_model.save_model(out/'selected_model.cbm')
    predictions = {}
    for period in ['validation', 'development']:
        meta = snapshots[period]
        rows += [dict(period=period, method='likes_rrf', **meta['likes_rrf']), dict(period=period, method='expanded_rrf', **meta['rrf'])]
        for mode, model in models.items():
            result, pred = evaluate(model, out/'pairs'/f'{period}.parquet', meta, MODES[mode], chosen[mode]['iterations'])
            rows.append(dict(period=period, method=mode, **result)); predictions[(period, mode)] = pred
    paired = []
    meta = snapshots['development']; truth = {int(k): set(v) for k, v in meta['truth'].items()}
    active = [u for u in meta['users'] if truth.get(u)]
    def per_user(mode, users):
        pred = predictions[('development', mode)]
        return np.asarray([len(truth[u] & set(pred.get(u, [])))/len(truth[u]) for u in users])
    for better, reference in [('listen_no_sample_trends', 'base_features'), ('listen_features', 'listen_no_sample_trends'), ('listen_features', 'base_features')]:
        for cohort, users in [('active', active), ('full100', [u for u in active if u in set(meta['complete'])])]:
            paired.append(dict(comparison=f'{better} minus {reference}, development', cohort=cohort, users=len(users),
                               **core.paired_interval(per_user(better, users)-per_user(reference, users), a.seed)))
    report.table('validation_selection', pl.DataFrame(validation), 'Select tree count on validation Recall@100')
    report.table('metrics', pl.DataFrame(rows), 'Matched-pool feature ablation', 'Compare within this experiment. The new 60-day cutoff has only 30 days of labels, unlike the old ranker development_60d. Last-30-day development matches the earlier target window.')
    report.table('candidate_ceilings', pl.DataFrame([dict(period=k, users=len(v['users']),
                 mean_candidates=v['candidate_pairs']/len(v['users']), coverage_active=v['candidate_recall_active'],
                 coverage_full100=v['candidate_recall_full100']) for k,v in snapshots.items()]), 'Expanded-pool candidate ceilings',
                 'Opportunity bounds, not final Recall@100. Active and full100 users have different target censoring and activity.')
    report.table('paired', pl.DataFrame(paired), 'Paired feature gain', 'Intervals condition on observed users/time. Not a hidden-test guarantee.')
    report.table('importance', pl.DataFrame(importance).sort(['mode', 'importance'], descending=[False, True]), 'PredictionValuesChange importance', 'Full fitted models, before prefix selection; not causal importance or Recall contribution.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, period in zip(axes, ['validation', 'development']):
        part = [v for v in rows if v['period'] == period]
        ax.barh([v['method'] for v in part], [100*(v['recall_active'] or 0) for v in part]); ax.invert_yaxis()
        ax.set_xlabel('Recall@100, active users (%)'); ax.set_title(period)
    report.figure('ranker_ablation', fig, 'Does the ranker use listening evidence?', 'All fitted models share candidate rows, labels, users, objective and hyperparameters; their features differ.')
    report.summary.update(version=VERSION, arguments=vars(a), features=MODES, validation=validation, metrics=rows,
                          chosen=chosen, selected_on_validation=best, paired=paired, snapshots=snapshots,
                          listen_cache_signature=manifest['signature'], sampled_users=ctx.users, train_users=train_users,
                          peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                          development_is_not_final_audit=True,
                          target='next100 chronological like events, distinct-item recall; zeros mean no observed target membership, not verified dislike')
    report.finish()


if __name__ == '__main__': main()
