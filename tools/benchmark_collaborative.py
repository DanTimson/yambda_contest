#!/usr/bin/env python3
"""Evaluate ALS/BM25 candidate opportunity on longer next-100-like targets."""
import argparse
import gc
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import explore_yambda as base
import explore_trends as core
import explore_listens as listening
import train_listen_ranker as ranker
from collaborative_sources import Sources, Config
from listen_cache_portable import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--users',default='input/test_users.csv')
    p.add_argument('--listen-cache-bundle',required=True);p.add_argument('--cache-dir',default='data/eda_cache')
    p.add_argument('--out',required=True);p.add_argument('--max-users',type=int,default=5000)
    p.add_argument('--eval-users',type=int,default=1500);p.add_argument('--future-days',nargs='+',type=int,default=[150,120])
    p.add_argument('--lookback-days',type=int,default=90);p.add_argument('--budgets',nargs='+',type=int,default=[1000,2000,3000])
    p.add_argument('--cf-mod',type=int,default=3);p.add_argument('--als-factors',type=int,default=64)
    p.add_argument('--als-iterations',type=int,default=20);p.add_argument('--knn-items',type=int,default=200000)
    p.add_argument('--knn-neighbors',type=int,default=80);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--retrieval-device',choices=['auto','gpu','cpu'],default='auto')
    a=p.parse_args();report=core.new_report(a.out)
    if min(a.budgets)<100 or a.eval_users<1: p.error('Invalid budgets/cohort')
    ctx=core.load_context(a.data,a.users,a.max_users,a.seed)
    cuts=sorted(set(ctx.end-d*core.DAY for d in a.future_days))
    if min(cuts)<=ctx.start:p.error('Cutoff precedes available history')
    events,manifest=load_bundle(a.listen_cache_bundle,a.cache_dir,ctx.data,a.users,ctx,cuts,a.lookback_days)
    ids=sorted(np.random.default_rng(a.seed+7).choice(ctx.users,min(a.eval_users,len(ctx.users)),replace=False).tolist())
    cfg=Config(user_mod=a.cf_mod,factors=a.als_factors,iterations=a.als_iterations,knn_items=a.knn_items,
               knn_neighbors=a.knn_neighbors,threads=a.threads,seed=a.seed,retrieval_device=a.retrieval_device)
    rows=[];marginal=[];model_info=[]
    for cut in cuts:
        ranker.log(f'LONG TARGET candidate evaluation cutoff={cut}')
        history,target=core.history_targets(ctx,cut);history=history.filter(pl.col('uid').is_in(ids));target=target.filter(pl.col('uid').is_in(ids))
        truth=base.user_lists(target);complete=set(target.group_by('uid').len().filter(pl.col('len')==100)['uid'].to_list())
        likes=ranker.LikeFeatures(ctx,cut,ids,a.cache_dir,200000)
        hears=ranker.ListenFeatures(events,ctx,cut,ids,history,a.lookback_days,1000)
        cf=Sources(ctx.data,cut,a.cache_dir,cfg,ids,history,hears.pairs)
        model_info.append(dict(cutoff=cut,**cf.info))
        banks={k:{} for k in ['likes','likes_listens','als_likes','als_listens','bm25_likes','bm25_listens','combined_cf']}
        union={}
        for u in ids:
            old=likes.user_sources(u)[2][:1000];heard=core.fuse([v.get(u,[]) for v in hears.lists.values()],1000)
            core_pool=core.fuse([old,heard],2000);channels=cf.lists(u)
            banks['likes'][u]=old;banks['likes_listens'][u]=core_pool
            for name,items in channels.items():banks[name][u]=items
            als=core.fuse([channels['als_likes'],channels['als_listens']],1000)
            knn=core.fuse([channels['bm25_likes'],channels['bm25_listens']],1000)
            family=core.fuse([als,knn],2000)
            order=core.fuse([old,heard,family],len(core_pool)+len(family));keep=set(core_pool)
            for item in order:
                if len(keep)>=max(a.budgets):break
                keep.add(item)
            banks['combined_cf'][u]=[i for i in order if i in keep]
            union[u]=sorted(set(core_pool)|set(i for v in channels.values() for i in v))
            t=set(truth.get(u,[]))
            if t:
                oldset=set(core_pool)
                for name,items in channels.items():
                    marginal.append(dict(cutoff=cut,uid=u,source=name,full100=u in complete,
                                         extra_target_fraction=len((set(items)-oldset)&t)/len(t)))
        for budget in sorted(set(a.budgets)):
            for name,pred in banks.items():
                m,_,_=listening.metrics({u:v[:budget] for u,v in pred.items()},truth,ids,complete,cut,name)
                rows.append(dict(budget=budget,**m))
        oracle,_,_=listening.metrics(union,truth,ids,complete,cut,'all_sources_union_oracle',oracle=True)
        rows.append(dict(budget=-1,**oracle))
        report.table(f'cohort_{cut}',pl.DataFrame([dict(users=len(ids),active=sum(bool(truth.get(u)) for u in ids),full100=len(complete))]),f'Cohort at {cut}')
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for name in ['likes_listens','als_likes','bm25_likes','combined_cf']:
            part=[r for r in rows if r['cutoff']==cut and r['source']==name and r['budget']>0]
            for ax,field in zip(axes,['coverage_active','coverage_full100']):
                ax.plot([r['budget'] for r in part],[100*r[field] if r[field] is not None else np.nan for r in part],marker='o',label=name)
        for ax,label in zip(axes,['Active users','Complete 100-event users']):
            ax.set_title(label);ax.set_xlabel('Candidate budget');ax.set_ylabel('Candidate coverage (%)');ax.legend(fontsize=8)
        report.figure(f'coverage_{cut}',fig,f'Collaborative opportunity: cutoff {cut}','Coverage is an oracle opportunity bound. Different folds overlap in future and are not independent tests.')
        del cf,likes,hears,banks,union;gc.collect()
    report.table('metrics',pl.DataFrame(rows),'Equal-budget candidate comparison','Read complete and active metrics separately. Missing sources remain empty; no synthetic positive insertion.')
    if marginal:
        report.table('marginal',pl.DataFrame(marginal).group_by('cutoff','source','full100').agg(pl.len().alias('active_users'),pl.col('extra_target_fraction').mean()),'Targets beyond the existing core pool')
    report.summary.update(arguments=vars(a),metrics=rows,models=model_info,users=ids,portable_cache=manifest,
                          target='first 100 future like events until public end; distinct-item recall',development_is_not_final_audit=True)
    report.finish()


if __name__=='__main__':main()
