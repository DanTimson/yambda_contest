#!/usr/bin/env python3
"""Block-friendly trend and prior-listen experiments. Local data only; no fitted models."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

try:
    import explore_yambda as base
except ImportError as exc:
    raise ImportError('Keep this script next to the existing tools/explore_yambda.py') from exc

DAY=base.DAY
SOURCES=['pop_all','pop30','pop7','pop2','decay7','decay30','growth7','momentum7']


@dataclass
class Context:
    data: Path
    users: list[int]
    likes: pl.DataFrame
    end: int
    start: int
    seed: int
    user_sha256: str


def load_context(data,users='input/test_users.csv',max_users=0,seed=42):
    root=Path(data);uf=Path(users);frame=pl.read_csv(uf)
    if frame.columns!=['uid'] or frame['uid'].null_count() or frame['uid'].n_unique()!=frame.height:
        raise ValueError('Expected one unique non-null uid column')
    ids=frame['uid'].cast(pl.Int64).to_list()
    if not ids:raise ValueError('Empty cohort')
    if max_users<0:raise ValueError('max_users must be >=0')
    if max_users:ids=np.random.default_rng(seed).choice(ids,min(max_users,len(ids)),replace=False).tolist()
    ids=sorted(ids)
    q=pl.scan_parquet(root/'likes.parquet')
    bounds=base.collect(q.select(pl.col('timestamp').min().alias('start'),pl.col('timestamp').max().alias('last'))).row(0,named=True)
    if bounds['last'] is None:raise ValueError('Empty likes file')
    base.log(f'Loading likes for {len(ids):,} requested users; no population event table is collected.')
    likes=base.collect(q.filter(pl.col('uid').is_in(ids)).select('uid','item_id','timestamp'))
    likes=likes.with_columns(pl.all().cast(pl.Int64)).sort(['uid','timestamp','item_id'])
    return Context(root,ids,likes,int(bounds['last'])+1,int(bounds['start']),seed,hashlib.sha256(uf.read_bytes()).hexdigest())


def history_targets(ctx,cutoff):
    if not ctx.start<cutoff<ctx.end:raise ValueError('Cutoff must be inside the dataset time range')
    h=ctx.likes.filter(pl.col('timestamp')<cutoff)
    # Same explicit primary convention as EDA v1: first 100 chronological like EVENTS.
    t=ctx.likes.filter(pl.col('timestamp')>=cutoff).group_by('uid',maintain_order=True).head(100)
    return h,t


def trend_sources(path,cutoff,pool_size=1000,min_support=10):
    """Every source is built strictly before cutoff. No target information is accepted."""
    q=pl.scan_parquet(path).filter(pl.col('timestamp')<cutoff)
    age=(pl.lit(cutoff)-pl.col('timestamp').cast(pl.Float64))/DAY
    stats=base.collect(q.group_by('item_id').agg(
        pl.len().alias('pop_all'),
        (pl.col('timestamp')>=cutoff-30*DAY).sum().alias('pop30'),
        (pl.col('timestamp')>=cutoff-7*DAY).sum().alias('pop7'),
        (pl.col('timestamp')>=cutoff-2*DAY).sum().alias('pop2'),
        ((pl.col('timestamp')>=cutoff-14*DAY)&(pl.col('timestamp')<cutoff-7*DAY)).sum().alias('prev7'),
        (-age/7).exp().sum().alias('decay7'),(-age/30).exp().sum().alias('decay30'),
        pl.col('uid').filter(pl.col('timestamp')>=cutoff-7*DAY).n_unique().alias('recent_users'),
        pl.col('timestamp').min().alias('first_seen_ts'),
    )).with_columns(pl.col('item_id').cast(pl.Int64))
    # Growth captures proportional expansion; momentum captures absolute excess volume.
    stats=stats.with_columns(((pl.col('pop7')+2.)/(pl.col('prev7')+2.)).alias('ratio7'))
    stats=stats.with_columns((pl.col('ratio7').log()*pl.col('recent_users').cast(pl.Float64).log1p()).alias('growth7'),
                             (pl.col('pop7').cast(pl.Float64)-pl.col('prev7').cast(pl.Float64)).alias('momentum7'))
    result={}
    for name in SOURCES:
        candidate=stats
        if name in ['growth7','momentum7']:
            candidate=candidate.filter((pl.col('recent_users')>=min_support)&(pl.col(name)>0))
        else:candidate=candidate.filter(pl.col(name)>0)
        result[name]=candidate.sort([name,'item_id'],descending=[True,False]).head(pool_size)['item_id'].to_list()
    return stats,result


def fuse(lists,budget=1000,per_source=None):
    scores={}
    for items in lists:
        for rank,item in enumerate(items[:per_source] if per_source else items,1):
            scores[item]=scores.get(item,0.)+1/(60+rank)
    return [i for i,s in sorted(scores.items(),key=lambda p:(-p[1],p[0]))[:budget]]


def cohort_arrays(truth,users,history):
    true=base.user_lists(truth)
    sets=[set(true.get(u,[])) for u in users]
    n=np.array([len(s) for s in sets],dtype=int)
    events=truth.group_by('uid').len()
    complete_ids=set(events.filter(pl.col('len')==100)['uid'].to_list())
    complete=np.array([u in complete_ids for u in users])
    warm_ids=set(history['uid'].to_list())
    warm=np.array([u in warm_ids for u in users])
    return sets,n,complete,warm


def evaluate_list(items,truth_sets,n,complete,warm):
    rank={i:j+1 for j,i in enumerate(items)}
    top=np.array([sum(rank.get(i,10**12)<=100 for i in t) for t in truth_sets],dtype=float)
    candidates=np.array([sum(i in rank for i in t) for t in truth_sets],dtype=float)
    den=np.maximum(n,1);active=n>0
    score=top/den;ceiling=candidates/den
    def mean(a,mask):return float(a[mask].mean()) if mask.any() else None
    return {'pool_items':len(items),'recall100_all':float(score.mean()),'recall100_active':mean(score,active),
            'recall100_full100':mean(score,complete),'coverage_active':mean(ceiling,active),
            'coverage_full100':mean(ceiling,complete),'coverage_warm_active':mean(ceiling,warm&active),
            'coverage_cold_active':mean(ceiling,(~warm)&active)},score,ceiling


def paired_interval(delta,seed=42,repeats=500):
    if not len(delta):return {'delta':None,'low95':None,'high95':None}
    rng=np.random.default_rng(seed)
    means=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(repeats)])
    return {'delta':float(delta.mean()),'low95':float(np.quantile(means,.025)),'high95':float(np.quantile(means,.975))}


def new_report(out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError(f'Choose a new output directory; refusing to overwrite {out}')
    return base.Report(out)


def run_trends(ctx,cutoffs,out,pool_size=1000,min_support=10):
    if pool_size<100 or min_support<1:raise ValueError('pool_size >=100 and min_support >=1 required')
    r=new_report(out);start=time.perf_counter();rows=[];snapshots=[]
    r.prose('Question before this experiment',
        'Do short-lived popularity and growth signals retrieve future likes beyond long-history popularity? '
        'Observe both standalone performance and additional hits. These sources are correlated: eight popularity variants do not replace personalized retrieval.')
    for cutoff in sorted(set(cutoffs)):
        h,t=history_targets(ctx,cutoff);ts,n,complete,warm=cohort_arrays(t,ctx.users,h)
        base.log(f'Building past-only trend statistics at {cutoff}; future observation {(ctx.end-cutoff)/DAY:.0f} days...')
        stats,lists=trend_sources(ctx.data/'likes.parquet',cutoff,pool_size,min_support)
        union=sorted(set(i for values in lists.values() for i in values))
        lists['union_uncapped']=union
        # Equal final budget for fusion, as well as the unbounded opportunity upper bound.
        lists['rrf_fixed_budget']=fuse([lists[k] for k in SOURCES],pool_size)
        evaluated={};metrics={};active=n>0
        for name,items in lists.items():
            m,score,coverage=evaluate_list(items,ts,n,complete,warm)
            if name=='union_uncapped':
                # Sorted IDs define no recommendation order: only union coverage is meaningful.
                m['recall100_all']=m['recall100_active']=m['recall100_full100']=None
            m.update(cutoff=cutoff,source=name,users=len(ctx.users),active_users=int(active.sum()),full100_users=int(complete.sum()))
            metrics[name]=m;evaluated[name]=(score,coverage);rows.append(m)
        marginal=[]
        for name in SOURCES:
            other=sorted(set(i for k in SOURCES if k!=name for i in lists[k]))
            _,_,minus=evaluate_list(other,ts,n,complete,warm)
            unique_gain=evaluated['union_uncapped'][1]-minus
            delta=evaluated[name][0]-evaluated['pop_all'][0]
            marginal.append({'source':name,'unique_union_coverage_gain_active':float(unique_gain[active].mean()) if active.any() else None,
                **{'recall100_vs_pop_all_'+k:v for k,v in paired_interval(delta[active],ctx.seed).items()}})
        r.table(f'sources_{cutoff}',pl.DataFrame(list(metrics.values())),f'Trend source comparison — cutoff {cutoff}',
                'Same cohort, same next-100-like-event target. Coverage refers to up to pool_size candidates; Recall refers to the first 100. Uncapped union is an oracle opportunity, not an equal-cost method.')
        r.table(f'marginal_{cutoff}',pl.DataFrame(marginal),f'What does each source add? — cutoff {cutoff}',
                'Unique union gain removes a source without backfill. Bootstrap intervals compare Recall@100 to pop_all within this cohort; they do not account for time shifts or multiple comparisons.')
        chosen=[name for name in SOURCES if metrics[name]['coverage_active'] is not None]
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        x=np.arange(len(chosen))
        axes[0].barh(x,[metrics[k]['recall100_active']*100 for k in chosen]);axes[0].set_yticks(x,chosen);axes[0].invert_yaxis();axes[0].set_xlabel('Recall@100, active users (%)')
        axes[1].barh(x,[metrics[k]['coverage_active']*100 for k in chosen]);axes[1].set_yticks(x,chosen);axes[1].invert_yaxis();axes[1].set_xlabel(f'Candidate coverage, up to {pool_size} items (%)')
        r.figure(f'source_comparison_{cutoff}',fig,f'Recency versus growth — cutoff {cutoff}',
                 'Read the two panels separately: a good candidate source need not be a good final ordering. Growth/momentum lists may be shorter when support thresholds remove noisy tracks.')
        fig,ax=plt.subplots(figsize=(9,4));values=[m['unique_union_coverage_gain_active'] for m in marginal]
        ax.barh(SOURCES,[v*100 if v is not None else 0 for v in values]);ax.invert_yaxis();ax.set_xlabel('Unique contribution to union coverage (percentage points)')
        r.figure(f'unique_gain_{cutoff}',fig,f'Complementarity, not just standalone scores — cutoff {cutoff}',
                 'A source may score poorly alone but retrieve targets none of the other sources find. This is candidate evidence, not a CatBoost feature importance.')
        top=stats.filter(pl.col('item_id').is_in(lists['growth7'][:20])).sort(['growth7','item_id'],descending=[True,False])
        r.table(f'growth_examples_{cutoff}',top,f'Inspect the rising-track counts — cutoff {cutoff}',
                'pop7 and prev7 count like events. recent_users counts distinct recent users and supplies the growth filter. First observed like is not a verified release date.')
        snapshots.append({'cutoff':cutoff,'future_days':(ctx.end-cutoff)/DAY,'metrics':metrics,'marginal':marginal,
                          'source_lists':{k:v for k,v in lists.items() if k in SOURCES}})
    r.summary={'purpose':'model-free trends workshop','users':len(ctx.users),'user_file_sha256':ctx.user_sha256,
               'seed':ctx.seed,'pool_size':pool_size,'min_recent_distinct_users':min_support,'snapshots':snapshots,
               'seconds':time.perf_counter()-start,'sources':{
               'pop_all':'All historical like-event count','pop30':'Last 30 days','pop7':'Last 7 days','pop2':'Last 2 days',
               'decay7':'sum exp(-age_days/7), 7-day time constant','decay30':'sum exp(-age_days/30), 30-day time constant',
               'growth7':'log((pop7+2)/(prev7+2))*log1p(recent_users), positive only, support filter',
               'momentum7':'pop7-prev7, positive only, same support filter'}}
    r.table('all_sources',pl.DataFrame(rows),'Across temporal placements','Do not choose a time constant from only one cutoff. These are development experiments, not a held-out final audit.')
    r.finish();return r.summary


def run_listens(ctx,cutoff,out,n_users=1000,reference_items=None):
    path=ctx.data/'listens.parquet'
    if not path.exists():raise FileNotFoundError('No local listens.parquet. Skip this block; no automatic download is attempted.')
    if n_users<1:raise ValueError('n_users must be positive')
    h,t=history_targets(ctx,cutoff)
    users=sorted(np.random.default_rng(ctx.seed).choice(ctx.users,min(n_users,len(ctx.users)),replace=False).tolist())
    r=new_report(out)
    base.log(f'Scanning 90-day pre-cutoff listens for {len(users)} sampled users; physical I/O may still be large...')
    q=pl.scan_parquet(path).filter(pl.col('uid').is_in(users)&(pl.col('timestamp')<cutoff)&(pl.col('timestamp')>=cutoff-90*DAY))
    if 'played_ratio_pct' in q.collect_schema():q=q.filter(pl.col('played_ratio_pct')>50)
    pairs=base.collect(q.group_by('uid','item_id').agg(pl.len().alias('plays'),pl.col('timestamp').max().alias('last_ts'))).with_columns(pl.col('uid','item_id','last_ts').cast(pl.Int64))
    pairs=pairs.with_columns((pl.col('plays').cast(pl.Float64).log1p()*(-(cutoff-pl.col('last_ts'))/(30*DAY)).exp()).alias('score'))
    ranked=pairs.sort(['uid','score','item_id'],descending=[False,True,False]).group_by('uid',maintain_order=True).head(1000)
    pred=base.user_lists(ranked);truth=base.user_lists(t)
    seen={u:set(g['item_id'].to_list()) for (u,),g in h.group_by('uid')}
    rows=[];ref=set(reference_items or []);den_active=0;hit_listen=0;hit_listen100=0;hit_reference=0;hit_union=0;hit_additional=0
    for u in users:
        true=set(truth.get(u,[]));items=pred.get(u,[]);pool=set(items)
        row={'uid':u,'target_unique':len(true),'qualifying_listened_candidates':len(items),
             'hits_listened100':len(true&set(items[:100])),'hits_listened1000':len(true&pool),
             'hits_reference_pool':len(true&ref),'hits_union':len(true&(pool|ref)),
             'hits_additional_listens':len((true&pool)-ref),
             'past_liked_candidates':len(pool&seen.get(u,set()))}
        rows.append(row)
        if true:
            den_active+=1;d=len(true);hit_listen+=row['hits_listened1000']/d;hit_listen100+=row['hits_listened100']/d
            hit_reference+=row['hits_reference_pool']/d;hit_union+=row['hits_union']/d;hit_additional+=row['hits_additional_listens']/d
    info={'cutoff':cutoff,'sample_users':len(users),'active_future_users':den_active,'reference_pool_items':len(ref),
          'listened_recall100_active':base.safe_div(hit_listen100,den_active),
          'listened_coverage1000_active':base.safe_div(hit_listen,den_active),
          'reference_coverage_active':base.safe_div(hit_reference,den_active) if ref else None,
          'union_coverage_active':base.safe_div(hit_union,den_active) if ref else None,
          'additional_listen_coverage_active':base.safe_div(hit_additional,den_active) if ref else None}
    r.table('listen_users',pl.DataFrame(rows),'Listen source: per-user diagnostics','Same seeded sample for listening and reference sources. No past-like exclusion is applied; the literal next-100-event target allows re-likes.')
    r.table('listen_summary',pl.DataFrame([info]),'Listening beyond the trend reference','Union coverage is a larger candidate budget and is not a final Recall@100 result. Compare additional hits before constructing a fixed-budget fused source.')
    if den_active:
        labels=['Listens top100','Listens pool1000'];values=[info['listened_recall100_active'],info['listened_coverage1000_active']]
        if ref:labels+=['Trend reference pool','Union pool'];values+=[info['reference_coverage_active'],info['union_coverage_active']]
        fig,ax=plt.subplots(figsize=(9,4));ax.barh(labels,[v*100 for v in values]);ax.invert_yaxis();ax.set_xlabel('Mean fraction of observed targets (%)')
        r.figure('listen_complementarity',fig,'Does listening add personalized candidates?',
                 'Only the first bar is a 100-item recommendation score. The other bars measure larger candidate pools. Prior listening is observational support, not a causal effect.')
    r.summary={'purpose':'prior listens and complementarity','metrics':info,'seed':ctx.seed,'sample_users':users,
               'user_file_sha256':ctx.user_sha256,'reference_items':list(reference_items or [])}
    r.finish();return r.summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['trends','listens'])
    p.add_argument('--data',required=True);p.add_argument('--users',default='input/test_users.csv');p.add_argument('--out',required=True)
    p.add_argument('--cutoffs',nargs='+',type=int);p.add_argument('--max-users',type=int,default=0);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pool-size',type=int,default=1000);p.add_argument('--min-support',type=int,default=10);p.add_argument('--listen-users',type=int,default=1000)
    a=p.parse_args();ctx=load_context(a.data,a.users,a.max_users,a.seed)
    cuts=a.cutoffs or [ctx.end-90*DAY,ctx.end-30*DAY]
    if a.mode=='trends':run_trends(ctx,cuts,a.out,a.pool_size,a.min_support)
    else:
        if len(cuts)!=1:p.error('listens mode requires exactly one --cutoffs value')
        run_listens(ctx,cuts[0],a.out,a.listen_users)


if __name__=='__main__':main()
