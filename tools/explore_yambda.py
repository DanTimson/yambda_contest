#!/usr/bin/env python3
"""Read-only, model-free EDA for local Yambda files. No downloads or model fitting."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DAY = 86400
KEYS = ['uid', 'item_id']


def collect(q):
    return q.collect(engine='streaming')


def log(message):
    print(message, flush=True)


def fractions(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return {'n': len(a), **{str(p): float(np.quantile(a, p)) for p in [0, .1, .25, .5, .75, .9, .95, .99, 1]}}


def safe_div(a, b):
    return float(a / b) if b else None


def user_lists(frame):
    return {int(u): g['item_id'].to_list() for (u,), g in frame.group_by('uid', maintain_order=True)}


def recall(pred, truth, users):
    active, all_scores = [], []
    for u in users:
        t = set(truth.get(u, []))
        score = len(t & set(pred.get(u, [])[:100])) / min(len(t), 100) if t else 0.
        all_scores.append(score)
        if t:
            active.append(score)
    return {'recall_all': float(np.mean(all_scores)) if all_scores else None,
            'recall_active': float(np.mean(active)) if active else None,
            'n_active': len(active)}


def target_variants(history, future):
    """Chronological event target plus deduplication / exclusion sensitivity analyses."""
    future = future.sort(['uid', 'timestamp', 'item_id'])
    event = future.group_by('uid', maintain_order=True).head(100)
    unique = future.unique(KEYS, keep='first', maintain_order=True).group_by('uid', maintain_order=True).head(100)
    old = history.select(KEYS).unique()
    new = future.join(old, on=KEYS, how='anti').sort(['uid', 'timestamp', 'item_id']).unique(KEYS, keep='first', maintain_order=True)
    new = new.sort(['uid', 'timestamp', 'item_id']).group_by('uid', maintain_order=True).head(100)
    return {'next100_events': event, 'next100_unique': unique, 'next100_new_items': new}


class Report:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'tables').mkdir(exist_ok=True)
        (self.out / 'plots').mkdir(exist_ok=True)
        self.sections = []
        self.files = []
        self.summary = {}

    def prose(self, title, text):
        self.sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p></section>')

    def table(self, name, frame, title, note='', limit=30):
        path = self.out / 'tables' / (name + '.csv')
        frame.write_csv(path)
        self.files.append(path)
        cols = frame.columns
        head = ''.join('<th>' + html.escape(c) + '</th>' for c in cols)
        rows = []
        for row in frame.head(limit).iter_rows():
            cells = [f'{v:.6g}' if isinstance(v, float) else str(v) if v is not None else '—' for v in row]
            rows.append('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in cells) + '</tr>')
        more = f' Showing first {limit} of {frame.height} rows; complete table in tables/{name}.csv.' if frame.height > limit else ''
        self.sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(note + more)}</p>'
                             f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')

    def figure(self, name, fig, title, note=''):
        fig.tight_layout()
        path = self.out / 'plots' / (name + '.svg')
        fig.savefig(path, format='svg', bbox_inches='tight')
        plt.close(fig)
        self.files.append(path)
        svg = path.read_text()
        svg = svg[svg.index('<svg'):]
        self.sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>{svg}</section>')

    def finish(self):
        summary = self.out / 'summary.json'
        summary.write_text(json.dumps(self.summary, indent=2, ensure_ascii=False, allow_nan=False))
        self.files.append(summary)
        content = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yambda exploratory analysis</title><style>
body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f4f6fa;color:#172131}main{max-width:1150px;margin:auto;padding:30px}
h1{font-size:32px}h2{font-size:23px}section{background:white;border:1px solid #dde3ec;border-radius:12px;padding:22px;margin:20px 0}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #e6eaf0;white-space:nowrap}
th{background:#edf2f9;position:sticky;top:0}svg{max-width:100%;height:auto}p{max-width:100ch}code{background:#edf2f9;padding:2px 5px}
</style><main><h1>Yambda exploratory analysis</h1><p>Local data only. No model training. All charts and tables embedded for offline reading.</p>'''
        path = self.out / 'report.html'
        path.write_text(content + ''.join(self.sections) + '</main></html>')
        self.files.append(path)
        archive = self.out / 'eda_for_review.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in self.files:
                z.write(f, f.relative_to(self.out))
        log(f'REPORT: {path.resolve()}\nRETURN: {archive.resolve()}')


def inventory(data, r):
    rows = []
    for path in sorted(Path(data).glob('*.parquet')):
        p = pq.ParquetFile(path)
        rows.append({'file': path.name, 'size_MiB': path.stat().st_size / 2**20,
                     'rows': p.metadata.num_rows, 'row_groups': p.metadata.num_row_groups,
                     'columns': ', '.join(f'{f.name}: {f.type}' for f in p.schema_arrow)})
    r.table('inventory', pl.DataFrame(rows), '1. Available files and schemas',
            'Metadata only; this does not read all listens or audio vectors.')
    r.summary['inventory'] = rows


def population(q, users, r):
    schema = q.collect_schema()
    required = {'uid', 'item_id', 'timestamp'}
    if not required.issubset(schema.names()):
        raise ValueError(f'likes.parquet must contain {sorted(required)}')
    aggs = [pl.len().alias('events'), pl.col('timestamp').min().alias('min_ts'), pl.col('timestamp').max().alias('max_ts')]
    aggs += [pl.col(c).null_count().alias(c + '_nulls') for c in required]
    if 'is_organic' in schema:
        aggs.append(pl.col('is_organic').mean().alias('organic_share'))
    stats = collect(q.select(aggs)).to_dicts()[0]
    if not stats['events'] or any(stats[c + '_nulls'] for c in required):
        raise ValueError('Empty likes or null uid/item_id/timestamp; inspect file before proceeding')
    log('Aggregating population user activity and item popularity (whole likes file)...')
    uc = collect(q.group_by('uid').agg(pl.len().alias('like_events'),
                      pl.col('timestamp').min().alias('first_ts'), pl.col('timestamp').max().alias('last_ts')))
    ic = collect(q.group_by('item_id').len().rename({'len': 'like_events'})).sort(['like_events', 'item_id'], descending=[True, False])
    stats.update(users_with_likes=uc.height, liked_items=ic.height, span_days=(stats['max_ts']-stats['min_ts'])/DAY,
                 population_user_event_quantiles=fractions(uc['like_events']), item_event_quantiles=fractions(ic['like_events']))
    matched = uc.filter(pl.col('uid').is_in(users))
    stats['requested_users'] = len(users)
    stats['requested_users_with_likes'] = matched.height
    stats['requested_user_coverage'] = matched.height / len(users)
    r.summary['population'] = stats
    r.table('population_summary', pl.DataFrame([{'measure': k, 'value': str(v)} for k,v in stats.items() if not isinstance(v, dict)]),
            '2. Population size and requested-user coverage',
            'Absence from likes does not prove absence from listens. Event counts include repeat likes.')
    missing = sorted(set(users) - set(matched['uid'].to_list()))
    r.table('requested_users_without_likes', pl.DataFrame({'uid': pl.Series(missing, dtype=pl.Int64)}),
            'Requested users without any like history', 'These users must not silently disappear from validation.', limit=10)
    fig, ax = plt.subplots(figsize=(10, 4))
    maximum = max(int(uc['like_events'].max()), 2)
    bins = np.geomspace(1, maximum + 1, 40)
    ax.hist(uc['like_events'].to_numpy(), bins=bins, density=True, histtype='step', lw=2, label='Population users with likes')
    if matched.height:
        ax.hist(matched['like_events'].to_numpy(), bins=bins, density=True, histtype='step', lw=2, label='Requested users with likes')
    ax.set(xscale='log', yscale='log', xlabel='Like events per user', ylabel='Density'); ax.legend()
    r.figure('user_activity', fig, 'User activity: requested cohort versus population', 'Both curves condition on at least one like. Zero-like requested users are counted separately above.')
    n = min(10000, ic.height)
    take = np.unique(np.geomspace(1, max(ic.height,1), 150).astype(int))
    cumulative = np.cumsum(ic['like_events'].to_numpy().astype(np.float64)) / stats['events']
    fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(take, cumulative[take-1])
    ax.set(xscale='log', xlabel='Top N tracks by whole-log event count', ylabel='Share of all like events', ylim=(0,1))
    r.figure('popularity_concentration', fig, 'How concentrated are likes?',
             'Whole-log descriptive popularity. This ranking must not be reused as a past-cutoff feature.')
    rows = [{'top_n': k, 'event_share': float(cumulative[min(k,ic.height)-1])} for k in [100,1000,10000,100000] if ic.height]
    r.table('popularity_concentration', pl.DataFrame(rows), 'Head versus tail')
    return stats, uc, ic


def timeline(q, selected, r):
    aggs = [pl.len().alias('like_events'), pl.col('uid').n_unique().alias('active_users')]
    if 'is_organic' in q.collect_schema():
        aggs.append(pl.col('is_organic').mean().alias('organic_share'))
    days = collect(q.with_columns((pl.col('timestamp')//DAY).alias('day')).group_by('day').agg(*aggs).sort('day'))
    # Complete day grid makes zero-activity days visible instead of connecting across missing data.
    grid = pl.DataFrame({'day': range(int(days['day'].min()), int(days['day'].max())+1)})
    days = grid.join(days.with_columns(pl.col('day').cast(pl.Int64)), on='day', how='left').with_columns(pl.col('like_events','active_users').fill_null(0))
    r.table('daily_population', days, '3. Global timeline', 'Day numbers are offsets from an anonymized origin, not calendar dates.', limit=12)
    cohort = selected.with_columns((pl.col('timestamp')//DAY).alias('day')).group_by('day').len().sort('day')
    fig, axes = plt.subplots(2,1,figsize=(11,6),sharex=True)
    axes[0].plot(days['day'], days['like_events'], label='Population'); axes[0].set_ylabel('Like events/day'); axes[0].legend()
    axes[1].plot(cohort['day'],cohort['len'], color='#bd5b34', label='Selected requested users');axes[1].set_ylabel('Like events/day');axes[1].set_xlabel('Elapsed dataset day');axes[1].legend()
    r.figure('timeline', fig, 'Activity and temporal changes', 'Different vertical scales: compare timing and shifts, not the amplitudes of the two panels.')
    if 'organic_share' in days.columns:
        fig, ax=plt.subplots(figsize=(10,3));ax.plot(days['day'],days['organic_share']);ax.set(xlabel='Elapsed dataset day',ylabel='Organic event share',ylim=(0,1))
        r.figure('organic_share',fig,'Organic versus recommendation-driven activity', 'Changes can indicate exposure-policy shifts as well as changes in user taste.')
    return days


def cohort_profile(selected, users, uc, r):
    stats = selected.group_by('uid').agg(pl.len().alias('events'),pl.col('item_id').n_unique().alias('unique_items'),
                pl.col('timestamp').min().alias('first_ts'),pl.col('timestamp').max().alias('last_ts'))
    grid=pl.DataFrame({'uid':pl.Series(users,dtype=pl.Int64)})
    stats=grid.join(stats,on='uid',how='left').with_columns(pl.col('events','unique_items').fill_null(0))
    total=selected.height; uniq=selected.select(KEYS).unique().height
    info={'users':len(users),'events':total,'distinct_user_item_pairs':uniq,'repeat_event_fraction':safe_div(total-uniq,total),
          'event_quantiles_including_zero_users':fractions(stats['events']),
          'unique_item_quantiles_including_zero_users':fractions(stats['unique_items'])}
    r.summary['selected_cohort']=info
    r.table('cohort_users',stats,'4. Cohort activity and repeat likes', 'Per-user summaries only; includes zero-like users. first_ts/last_ts refer to like activity, not account creation.',limit=12)
    r.table('repeat_summary',pl.DataFrame([{'events':total,'distinct_pairs':uniq,'repeat_events':total-uniq,'repeat_event_fraction':info['repeat_event_fraction']}]),
            'How often do users re-like the same item?','Raw repeated events are not necessarily mistakes: unlike/re-like cycles may exist.')
    if selected.height:
        ordered=selected.sort(['uid','timestamp','item_id'])
        gaps=ordered.with_columns(pl.col('timestamp').diff().over('uid').alias('gap')).filter(pl.col('gap').is_not_null())['gap'].to_numpy()/DAY
        r.summary['selected_cohort']['inter_like_gap_days']=fractions(gaps)
        positive=gaps[gaps>0]
        fig,ax=plt.subplots(figsize=(10,3))
        if len(positive):ax.hist(positive,bins=np.geomspace(max(positive.min(),1e-7),max(positive.max(),positive.min()*1.01),40));ax.set_xscale('log')
        ax.set(xlabel='Days between successive likes (positive gaps)',ylabel='Count')
        r.figure('inter_like_gaps',fig,'Timing of likes',f'{int((gaps==0).sum())} zero-gap transitions are omitted from this log-scale plot. Bulk likes at one timestamp are possible; their internal order is unknown.')
        gran=selected['timestamp'].to_numpy()%5
        r.summary['selected_cohort']['fraction_timestamps_divisible_by_5']=float(np.mean(gran==0))
    return stats


def load_artists(data, item_ids, r):
    path=Path(data)/'artist_item_mapping.parquet'
    if not path.exists():
        r.prose('Artist data','artist_item_mapping.parquet is absent; artist diagnostics are skipped.')
        return None
    m=collect(pl.scan_parquet(path).select('item_id','artist_id').filter(pl.col('item_id').is_in(item_ids)).unique()).with_columns(pl.col('item_id').cast(pl.Int64))
    counts=m.group_by('item_id').len()
    info={'cohort_liked_items':len(item_ids),'mapped_items':counts.height,'mapping_coverage':safe_div(counts.height,len(item_ids)),
          'artists_per_mapped_item':fractions(counts['len'])}
    r.summary['artists']=info
    r.table('artist_mapping',pl.DataFrame([{'measure':k,'value':str(v)} for k,v in info.items()]),'5. Artist mapping coverage',
            'All credited artists are retained. Multi-artist joins count credit relationships, not additional tracks.')
    return m


def snapshot_eda(q, selected, users, cutoff, end, artists, r):
    history=selected.filter(pl.col('timestamp')<cutoff)
    future=selected.filter(pl.col('timestamp')>=cutoff)
    variants=target_variants(history,future)
    hcounts=history.group_by('uid').len().rename({'len':'history_events'})
    base=pl.DataFrame({'uid':pl.Series(users,dtype=pl.Int64)}).join(hcounts,on='uid',how='left').with_columns(pl.col('history_events').fill_null(0))
    seen=history.select(KEYS).unique()
    log(f'Cutoff {cutoff}: aggregate pre-cutoff population popularity and next-100 diagnostics...')
    pastpop=collect(q.filter(pl.col('timestamp')<cutoff).group_by('item_id').len()).sort(['len','item_id'],descending=[True,False])
    ranked=[int(i) for i in pastpop['item_id']]
    observed=set(ranked)
    seen_by_user={int(u):set(g['item_id'].to_list()) for (u,),g in seen.group_by('uid')}
    # Generate only diagnostic popularity lists. No collaborative models or trees are fitted.
    pred={u:ranked[:100] for u in users}
    pred_unseen={}
    for u in users:
        unseen=[];old=seen_by_user.get(u,set())
        for item in ranked:
            if item not in old:unseen.append(item)
            if len(unseen)==100:break
        pred_unseen[u]=unseen
    target_rows=[];recall_rows=[];counts_primary=None
    for name,t in variants.items():
        counts=t.group_by('uid').agg(pl.len().alias('target_events'),pl.col('item_id').n_unique().alias('target_unique'),
                                    pl.col('timestamp').max().alias('last_target_ts'))
        joined=base.join(counts,on='uid',how='left').with_columns(pl.col('target_events','target_unique').fill_null(0))
        active=int((joined['target_events']>0).sum());complete=int((joined['target_events']==100).sum())
        unique_pairs=t.select(KEYS).unique()
        repeated=unique_pairs.join(seen,on=KEYS,how='inner').height
        known=unique_pairs.filter(pl.col('item_id').is_in(list(observed))).height
        complete_times=(joined.filter(pl.col('target_events')==100)['last_target_ts']-cutoff).to_numpy()/DAY
        row={'cutoff':cutoff,'variant':name,'future_observation_days':(end-cutoff)/DAY,'users':len(users),
             'active_future_users':active,'users_with_full100':complete,'full100_fraction_all':safe_div(complete,len(users)),
             'median_observed_targets_active':float(joined.filter(pl.col('target_events')>0)['target_events'].median()) if active else None,
             'target_pair_seen_in_user_history_fraction':safe_div(repeated,unique_pairs.height),
             'target_pair_globally_known_item_fraction':safe_div(known,unique_pairs.height),
             'median_days_to100_complete_only':float(np.median(complete_times)) if len(complete_times) else None}
        target_rows.append(row)
        truth=user_lists(t)
        for label,p in [('top100_past_popularity',pred),('top100_past_popularity_exclude_seen',pred_unseen)]:
            recall_rows.append({'cutoff':cutoff,'variant':name,'method':label,**recall(p,truth,users)})
        if name=='next100_events':
            counts_primary=joined
            bands=joined.with_columns(pl.when(pl.col('history_events')==0).then(pl.lit('0'))
                .when(pl.col('history_events')<10).then(pl.lit('1–9')).when(pl.col('history_events')<50).then(pl.lit('10–49'))
                .when(pl.col('history_events')<100).then(pl.lit('50–99')).otherwise(pl.lit('100+')).alias('history_band'))
            bands=bands.group_by('history_band').agg(pl.len().alias('users'),(pl.col('target_events')>0).sum().alias('active_future_users'),
                (pl.col('target_events')==100).mean().alias('full100_fraction'),pl.col('target_events').median().alias('median_observed_events')).sort('history_band')
            r.table(f'activity_bands_{cutoff}',bands,f'Future target completeness by history length — cutoff {cutoff}',
                    'The next 100 like events are the primary target. Completeness depends strongly on activity; complete-only evaluation changes the cohort.')
            row['days_to100_complete_only_quantiles']=fractions(complete_times)
            # Availability of the first 100 targets in a nested popularity pool. Real retrieval coverage, not ranking.
            poolrows=[]
            for k in [100,1000,10000]:
                pool=set(ranked[:k]);scores=[];upper=[]
                for u in users:
                    true=set(truth.get(u,[]))
                    if not true:continue
                    hits=len(true & pool);scores.append(hits/len(true));upper.append(min(hits,100)/min(len(true),100))
                poolrows.append({'cutoff':cutoff,'pool_size':min(k,len(ranked)),
                                 'target_fraction_active':float(np.mean(scores)) if scores else None,
                                 'oracle_recall100_active':float(np.mean(upper)) if upper else None})
            r.table(f'pop_pool_{cutoff}',pl.DataFrame(poolrows),f'Popularity candidate coverage — cutoff {cutoff}',
                    'Pools are learned strictly before the cutoff. Oracle means the best possible choice of 100 items from that pool, not an achievable model score.')
            if artists is not None and artists.height:
                hp=history.select(KEYS).unique().join(artists,on='item_id',how='inner')
                familiar=hp.select('uid','artist_id').unique()
                tp=unique_pairs.join(artists,on='item_id',how='inner')
                supported=tp.join(familiar,on=['uid','artist_id'],how='inner').select(KEYS).unique().height
                target_mapped=tp.select(KEYS).unique().height
                row['familiar_artist_target_pair_fraction_all']=safe_div(supported,unique_pairs.height)
                row['familiar_artist_target_pair_fraction_mapped']=safe_div(supported,target_mapped)
    # Avoid nested objects in CSV while preserving richer data in summary.json.
    flat=[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in target_rows]
    r.table(f'targets_{cutoff}',pl.DataFrame(flat),f'6. Target semantics and censoring — cutoff {cutoff}',
            'next100_events: literal first 100 like events; next100_unique: first 100 distinct future items; next100_new_items: additionally exclude any past likes. Same-timestamp ordering uses item_id as a deterministic tie-break, not an observed order.')
    r.table(f'pop_recall_{cutoff}',pl.DataFrame(recall_rows),f'Effect of seen-item exclusion — cutoff {cutoff}',
            'These are model-free diagnostics, not selected contest submissions. Evaluate the exclusion policy instead of assuming it cannot lose true positives.')
    r.summary.setdefault('snapshots',[]).append({'cutoff':cutoff,'targets':target_rows,'popularity_diagnostics':recall_rows})
    return variants,history,counts_primary


def trends(q, cutoff, r):
    counts=collect(q.filter((pl.col('timestamp')>=cutoff-28*DAY)&(pl.col('timestamp')<cutoff)).group_by('item_id').agg(
        (pl.col('timestamp')>=cutoff-7*DAY).sum().alias('last7'),
        ((pl.col('timestamp')>=cutoff-14*DAY)&(pl.col('timestamp')<cutoff-7*DAY)).sum().alias('prev7'),
        pl.len().alias('last28'))).with_columns(pl.col('item_id').cast(pl.Int64))
    if not counts.height:return
    counts=counts.with_columns(((pl.col('last7')+.5)/(pl.col('prev7')+.5)).alias('growth_ratio'))
    rising=counts.filter(pl.col('last7')>=10).with_columns(((pl.col('growth_ratio').log())*pl.col('last7').log1p()).alias('growth_score')).sort(['growth_score','item_id'],descending=[True,False]).head(30)
    r.table('rising_tracks',rising,'7. Rising tracks at the latest EDA cutoff',
            'All counts precede the cutoff. Growth uses smoothing and a 10-event minimum; item IDs are anonymous and first appearance is not a release date.')
    rows=[]
    for n in [100,1000,10000]:
        a=set(counts.sort(['last7','item_id'],descending=[True,False]).head(n).filter(pl.col('last7')>0)['item_id'])
        b=set(counts.sort(['prev7','item_id'],descending=[True,False]).head(n).filter(pl.col('prev7')>0)['item_id'])
        rows.append({'nominal_top_n':n,'last7_items':len(a),'prev7_items':len(b),'overlap':len(a&b),'jaccard':safe_div(len(a&b),len(a|b))})
    r.table('weekly_popularity_churn',pl.DataFrame(rows),'How stable is the popularity head?',
            'Adjacent seven-day popularity lists. Low overlap suggests testing multiple recency scales; it does not prove a particular decay constant is best.')
    top=counts.sort(['last7','item_id'],descending=[True,False]).head(15)
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(top.height)
    ax.bar(x-.2,top['prev7'].to_numpy(),.4,label='Preceding 7 days');ax.bar(x+.2,top['last7'].to_numpy(),.4,label='Latest 7 days')
    ax.set_xticks(x,[str(i) for i in top['item_id']],rotation=65);ax.set_ylabel('Like events');ax.legend()
    r.figure('weekly_head',fig,'Current popularity head: growth and decline')


def listen_diagnostics(data, users, cutoff, targets, seed, n_users, r):
    path=Path(data)/'listens.parquet'
    if not path.exists():
        r.prose('Optional listens','Requested listen analysis skipped: listens.parquet is absent.');return
    chosen=sorted(np.random.default_rng(seed).choice(users,min(n_users,len(users)),replace=False).tolist())
    log(f'Optional listens: scanning pre-cutoff plays for {len(chosen)} users. A full-file scan may take time.')
    q=pl.scan_parquet(path).filter(pl.col('uid').is_in(chosen)&(pl.col('timestamp')<cutoff)&(pl.col('timestamp')>=cutoff-90*DAY))
    schema=q.collect_schema()
    aggs=[pl.len().alias('play_events'),pl.col('item_id').n_unique().alias('items'),pl.col('timestamp').min().alias('min_ts'),pl.col('timestamp').max().alias('max_ts')]
    if 'played_ratio_pct' in schema:
        aggs.extend([(pl.col('played_ratio_pct')>50).mean().alias('over50_share'),(pl.col('played_ratio_pct')>100).mean().alias('over100_share'),pl.col('played_ratio_pct').mean().alias('mean_played_ratio_pct')])
    stats=collect(q.select(*aggs)).to_dicts()[0]
    valid=q.filter(pl.col('played_ratio_pct')>50) if 'played_ratio_pct' in schema else q
    pairs=collect(valid.group_by('uid','item_id').agg(pl.len().alias('plays'),pl.col('timestamp').max().alias('last_listen_ts'))).with_columns(pl.col('uid','item_id','last_listen_ts').cast(pl.Int64))
    truth=targets.filter(pl.col('uid').is_in(chosen)).select(KEYS).unique()
    joined=truth.join(pairs,on=KEYS,how='left')
    matched=joined.filter(pl.col('plays').is_not_null())
    stats.update(cutoff=cutoff,lookback_days=90,sampled_users=len(chosen),target_pairs=truth.height,
                 target_pairs_with_prior_qualifying_listen=matched.height,
                 prior_listen_target_pair_fraction=safe_div(matched.height,truth.height),
                 prior_play_count_quantiles_for_matched_targets=fractions(matched['plays']),
                 days_since_last_listen_for_matched_targets=fractions((cutoff-matched['last_listen_ts']).to_numpy()/DAY))
    r.summary['listens']=stats
    r.table('listen_summary',pl.DataFrame([{'measure':k,'value':str(v)} for k,v in stats.items()]),'8. Listening before future likes',
            'Only the seeded user sample, 90-day history, latest cutoff. Qualifying means played_ratio_pct >50. Values >100 are retained as replay behavior, not clipped. This measures opportunity, not causal conversion.')
    # A fixed listening-based recommendation list, computed without future labels.
    pairs=pairs.with_columns((pl.col('plays').cast(pl.Float64).log1p()*(-(cutoff-pl.col('last_listen_ts'))/(30*DAY)).exp()).alias('s'))
    ranked=pairs.sort(['uid','s','item_id'],descending=[False,True,False]).group_by('uid',maintain_order=True).head(100)
    metric=recall(user_lists(ranked),user_lists(targets),chosen)
    r.summary['listens']['recent_listened_top100_no_seen_filter']=metric
    r.table('listen_recall',pl.DataFrame([metric]),'Model-free listening candidate diagnostic',
            'This list allows old likes deliberately, to keep the exclusion policy separate. Target overlap is observational and does not estimate recommendation-induced likes.')


def embedding_diagnostics(data, cohort_items, r):
    path=Path(data)/'embeddings.parquet'
    if not path.exists():
        r.prose('Optional embeddings','Requested embedding analysis skipped: embeddings.parquet is absent.');return
    p=pq.ParquetFile(path);names=p.schema_arrow.names
    col='normalized_embed' if 'normalized_embed' in names else 'embed' if 'embed' in names else None
    if col is None:
        r.prose('Optional embeddings','No embed or normalized_embed column found.');return
    log('Optional embeddings: reading IDs for coverage, plus at most 1,024 vector rows for shape/norm audit...')
    matched=collect(pl.scan_parquet(path).select('item_id').filter(pl.col('item_id').is_in(cohort_items)).unique())
    batch=next(p.iter_batches(batch_size=1024,columns=[col]),None)
    vectors=[]
    if batch is not None:
        vectors=[np.asarray(v,dtype=float) for v in batch.column(0).to_pylist() if v is not None]
    finite=[v for v in vectors if np.isfinite(v).all()]
    stats={'cohort_liked_items':len(cohort_items),'items_with_embeddings':matched.height,'coverage':safe_div(matched.height,len(cohort_items)),
           'sample_method':'first at most 1024 rows; not a random catalog sample','vector_column':col,
           'sampled_vectors':len(vectors),'nonfinite_vectors':len(vectors)-len(finite),
           'dimensions':sorted(set(len(v) for v in vectors)),'norm_quantiles':fractions([np.linalg.norm(v) for v in finite])}
    r.summary['embeddings']=stats
    r.table('embedding_summary',pl.DataFrame([{'measure':k,'value':str(v)} for k,v in stats.items()]),'9. Audio embedding coverage and shape',
            'No index is built and no model is trained. Coverage uses the entire observed cohort item set and is descriptive, not a time-safe feature-selection step.')


def run(args):
    start=time.perf_counter();data=Path(args.data);out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is nonempty: {out}. Choose a new --out to preserve prior analysis.')
    if not (data/'likes.parquet').exists():raise FileNotFoundError(f'{data}/likes.parquet')
    user_frame=pl.read_csv(args.users)
    if user_frame.columns!=['uid'] or user_frame['uid'].null_count() or user_frame['uid'].n_unique()!=user_frame.height:
        raise ValueError('Expected exactly one unique non-null uid column')
    all_users=user_frame['uid'].cast(pl.Int64).to_list()
    if not all_users:raise ValueError('Empty user cohort')
    selected_users=sorted(all_users)
    if args.max_users:
        selected_users=sorted(np.random.default_rng(args.seed).choice(all_users,min(args.max_users,len(all_users)),replace=False).tolist())
    r=Report(out)
    r.prose('Purpose and boundaries',
        'Explore the next 100 likes per requested user before choosing recommendation models. Primary targets are the next 100 chronological like events. '
        'Distinct-future-item and first-ever-item variants quantify how much deduplication and past-like exclusion matter. '
        'There is no fixed 30-day target window. The public log end censors future observations. This report is descriptive; inspecting a period makes it development data.')
    inventory(data,r)
    q=pl.scan_parquet(data/'likes.parquet')
    stats,uc,ic=population(q,all_users,r)
    end=int(stats['max_ts'])+1
    cutoffs=sorted(set(args.cutoffs or [end-d*DAY for d in [120,90,60,30]]))
    cutoffs=[c for c in cutoffs if stats['min_ts']<c<end]
    if not cutoffs:raise ValueError('No valid cutoffs: pass elapsed-second values strictly inside the data range')
    log(f'Loading only {len(selected_users)} requested users\' likes; full population logs remain lazy...')
    cols=[c for c in ['uid','item_id','timestamp','is_organic'] if c in q.collect_schema()]
    selected=collect(q.filter(pl.col('uid').is_in(selected_users)).select(cols)).with_columns(pl.col('uid','item_id','timestamp').cast(pl.Int64)).sort(['uid','timestamp','item_id'])
    timeline(q,selected,r);cohort_profile(selected,selected_users,uc,r)
    ids=selected['item_id'].unique().to_list();artists=load_artists(data,ids,r)
    anchor=None
    for cutoff in cutoffs:
        variants,history,counts=snapshot_eda(q,selected,selected_users,cutoff,end,artists,r)
        anchor=variants
    rows=[{k:v for k,v in t.items() if not isinstance(v,dict)} for s in r.summary['snapshots'] for t in s['targets']]
    r.table('all_cutoffs',pl.DataFrame(rows),'Comparison across temporal placements')
    main=[t for s in r.summary['snapshots'] for t in s['targets'] if t['variant']=='next100_events']
    fig,axes=plt.subplots(1,2,figsize=(11,4));x=[(end-t['cutoff'])/DAY for t in main]
    axes[0].plot(x,[t['full100_fraction_all'] for t in main],'o-');axes[0].set(xlabel='Days of observed future',ylabel='Share with 100 future like events',ylim=(0,1))
    axes[1].plot(x,[t['target_pair_seen_in_user_history_fraction'] if t['target_pair_seen_in_user_history_fraction'] is not None else np.nan for t in main],'o-');axes[1].set(xlabel='Days of observed future',ylabel='Future target pairs already liked',ylim=(0,1))
    r.figure('target_completeness',fig,'The next-100 target: completeness and repeated items',
             'Cutoffs share the same log end and are correlated. Complete-only time-to-100 statistics omit censored users and underestimate the wait for the whole cohort.')
    trends(q,cutoffs[-1],r)
    if args.with_listens:listen_diagnostics(data,selected_users,cutoffs[-1],anchor['next100_events'],args.seed,args.listen_users,r)
    else:r.prose('Optional listens skipped','Use --with-listens if listens.parquet is already local. Large files may need a long scan; the user sample limits retained data, not necessarily bytes read.')
    if args.with_embeddings:embedding_diagnostics(data,ids,r)
    else:r.prose('Optional embeddings skipped','Use --with-embeddings to inspect local audio coverage and vector shape without building retrieval models.')
    r.prose('What to decide from these results',
        'Check requested-user coverage before choosing data scale; choose historical cutoffs with enough observed future; compare repeat-like target variants before hard exclusions; '
        'inspect popularity churn before picking recency scales; measure familiar-artist and prior-listen target support before investing in candidate sources. '
        'Popularity diagnostics are references, not a reason to fit CatBoost yet. Keep a later period uninspected for final model auditing.')
    config={'generated_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,
            'arguments':vars(args),'cutoffs':cutoffs,'dataset_end_exclusive':end,
            'user_file_sha256':hashlib.sha256(Path(args.users).read_bytes()).hexdigest(),
            'selected_users':len(selected_users),'python':platform.python_version(),
            'versions':{p:importlib.metadata.version(p) for p in ['polars','numpy','pyarrow','matplotlib']}}
    manifest=data/'dataset_manifest.json'
    if manifest.exists():config['dataset_manifest']=json.loads(manifest.read_text())
    r.summary['run']=config
    r.finish()
    return r.summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,help='Directory containing local likes.parquet')
    p.add_argument('--users',default='input/test_users.csv')
    p.add_argument('--out',required=True,help='New/empty output directory; never overwrite a prior analysis')
    p.add_argument('--cutoffs',nargs='+',type=int,help='Elapsed-second cutoffs; default log end minus 120/90/60/30 days')
    p.add_argument('--max-users',type=int,default=0,help='Seeded requested-user sample; 0 = all. Population summaries still use full likes file.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--with-listens',action='store_true')
    p.add_argument('--listen-users',type=int,default=1000)
    p.add_argument('--with-embeddings',action='store_true')
    args=p.parse_args()
    if args.max_users<0 or args.listen_users<1:p.error('max-users must be nonnegative and listen-users positive')
    run(args)


if __name__=='__main__':main()
