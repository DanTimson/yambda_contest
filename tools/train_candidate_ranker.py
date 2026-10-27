#!/usr/bin/env python3
"""Chronological CatBoost on the benchmark's candidate sources; no contest export."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import benchmark_candidates as bench
import explore_trends as core
import explore_yambda as base

CHANNELS=core.SOURCES+['artist_pop','artist_trend']
STAT_COLS=['pop_all','pop30','pop7','pop2','prev7','decay7','decay30','recent_users','ratio7','growth7','momentum7','first_seen_ts']
FEATURES=['log_'+c for c in STAT_COLS[:8]]+['ratio7','growth7','momentum7','first_seen_age_days']+[
    'log_user_likes','user_recent30_fraction','user_recency_days','seen_like','seen_recency_days','log_seen_count']+[
    s+'_rr' for s in CHANNELS]+['global_rr','artist_rr','fusion_rr','n_sources']


def metric(predictions,truth,users,complete):
    active=[];full=[];all_scores=[]
    for u in users:
        t=set(truth.get(u,[]));score=len(t&set(predictions.get(u,[])[:100]))/len(t) if t else 0.
        all_scores.append(score)
        if t:active.append(score)
        if u in complete:full.append(score)
    return {'recall_all':float(np.mean(all_scores)) if all_scores else None,
            'recall_active':float(np.mean(active)) if active else None,
            'recall_full100':float(np.mean(full)) if full else None,
            'n_users':len(users),'n_active':len(active),'n_full100':len(full)}


def source_features(ctx,cut,users,budget,cache_dir,artist_catalog,min_support):
    history=ctx.likes.filter((pl.col('timestamp')<cut)&pl.col('uid').is_in(users))
    stats,signature=bench.cached_stats(ctx.data,cut,cache_dir)
    lists=bench.global_lists(stats,1000,min_support)
    global_union=set(i for values in lists.values() for i in values)
    global_order=core.fuse(list(lists.values()),len(global_union))
    global_ranks={int(i):r for r,i in enumerate(global_order,1)}
    global_maps={k:{int(i):r for r,i in enumerate(v,1)} for k,v in lists.items()}
    artists=bench.ArtistSource(ctx.data,history,stats,cut,artist_catalog)
    index={int(i):j for j,i in enumerate(stats['item_id'])}
    stat_values=stats.select(STAT_COLS).to_numpy().astype(np.float32)
    histories={int(u):g for (u,),g in history.group_by('uid')}
    del stats;gc.collect()
    for u in users:
        personal=artists.recommend(u,1000)
        artist_union=set(i for values in personal.values() for i in values)
        artist_order=core.fuse(list(personal.values()),len(artist_union))
        # Hierarchical RRF over the complete source union, capped only at the final budget.
        fused=core.fuse([global_order,artist_order],budget)
        if len(fused)<100:raise ValueError(f'Only {len(fused)} candidates for uid={u}; insufficient historical catalog')
        ids=np.asarray(fused,dtype=np.int64);rows=np.asarray([index[int(i)] for i in ids]);v=stat_values[rows].copy()
        v[:,:8]=np.log1p(v[:,:8]);v[:,11]=(cut-v[:,11])/core.DAY
        h=histories.get(u);n=h.height if h is not None else 0
        recent=float((h['timestamp']>=cut-30*core.DAY).mean()) if n else 0.
        recency=(cut-int(h['timestamp'].max()))/core.DAY if n else -1.
        seen={int(i):(int(c),int(t)) for i,c,t in h.group_by('item_id').agg(pl.len(),pl.col('timestamp').max()).iter_rows()} if n else {}
        user=np.empty((len(ids),6),np.float32)
        user[:,0]=np.log1p(n);user[:,1]=recent;user[:,2]=recency
        user[:,3]=[float(int(i) in seen) for i in ids]
        user[:,4]=[(cut-seen[int(i)][1])/core.DAY if int(i) in seen else -1. for i in ids]
        user[:,5]=[np.log1p(seen.get(int(i),(0,0))[0]) for i in ids]
        maps={**global_maps,**{k:{int(i):r for r,i in enumerate(values,1)} for k,values in personal.items()}}
        channels=np.asarray([[1/maps[k][int(i)] if int(i) in maps[k] else 0. for k in CHANNELS] for i in ids],np.float32)
        ar={int(i):r for r,i in enumerate(artist_order,1)}
        rr=np.asarray([[1/global_ranks[int(i)] if int(i) in global_ranks else 0.,
                        1/ar[int(i)] if int(i) in ar else 0.,1/(r+1),float(np.count_nonzero(channels[r]))]
                       for r,i in enumerate(ids)],np.float32)
        x=np.concatenate([v,user,channels,rr],axis=1)
        if not np.isfinite(x).all():raise ValueError('Non-finite features')
        yield u,ids,x


def build_pairs(ctx,cut,label_end,users,out,budget,cache_dir,artist_catalog,min_support):
    """Candidates/features accept past only; labels are attached afterwards."""
    target=ctx.likes.filter(pl.col('uid').is_in(users)&(pl.col('timestamp')>=cut)&(pl.col('timestamp')<label_end))
    target=target.group_by('uid',maintain_order=True).head(100)
    truth=base.user_lists(target)
    complete=set(target.group_by('uid').len().filter(pl.col('len')==100)['uid'].to_list())
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    writer=None;baseline={};coverage=[];count=0;started=time.perf_counter();chunk=[]
    def write_chunk(parts,writer):
        table=pa.concat_tables(parts)
        if writer is None:writer=pq.ParquetWriter(path,table.schema,compression='zstd')
        writer.write_table(table);return writer
    try:
        for u,ids,x in source_features(ctx,cut,users,budget,cache_dir,artist_catalog,min_support):
            true=set(truth.get(u,[]));y=np.array([int(int(i) in true) for i in ids],np.int8)
            columns={name:pa.array(x[:,j]) for j,name in enumerate(FEATURES)}
            columns.update(uid=pa.array(np.full(len(ids),u,np.int64)),item_id=pa.array(ids),
                candidate_rank=pa.array(np.arange(1,len(ids)+1,dtype=np.int32)),target=pa.array(y),
                n_true=pa.array(np.full(len(ids),len(true),np.int16)))
            chunk.append(pa.table(columns));count+=len(ids);baseline[u]=ids[:100].tolist()
            if true:coverage.append(float(y.sum()/len(true)))
            if len(chunk)>=16:writer=write_chunk(chunk,writer);chunk=[]
        if chunk:writer=write_chunk(chunk,writer)
    finally:
        if writer is not None:writer.close()
    result={'cutoff':cut,'label_end_exclusive':label_end,'users':users,'truth':truth,'complete':sorted(complete),
            'candidate_pairs':count,'candidate_recall_active':float(np.mean(coverage)) if coverage else None,
            'rrf':metric(baseline,truth,users,complete),'seconds':time.perf_counter()-started}
    path.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False))
    print(f'PAIRS cutoff={cut}: rows={count:,}, candidate recall={result["candidate_recall_active"]}',flush=True)
    return result


def predict_file(model,path,cap,trees,features=FEATURES,baseline=False):
    """Bounded inference; row groups may split users, so merge their local top-100 lists."""
    best={}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        frame=pl.from_arrow(batch).filter(pl.col('candidate_rank')<=cap)
        if not frame.height:continue
        score=-frame['candidate_rank'].to_numpy().astype(float) if baseline else model.predict_proba(frame.select(features).to_numpy(),ntree_end=trees)[:,1]
        top=frame.select('uid','item_id').with_columns(pl.Series('score',score)).sort(['uid','score','item_id'],descending=[False,True,False]).group_by('uid',maintain_order=True).head(100)
        for (u,),g in top.group_by('uid'):
            values=best.get(int(u),[])+g.select('item_id','score').rows()
            best[int(u)]=sorted(values,key=lambda p:(-p[1],p[0]))[:100]
    return {u:[int(i) for i,s in values] for u,values in best.items()}


def evaluate_file(model,path,meta,cap,trees,baseline=False,return_predictions=False):
    truth={int(k):v for k,v in meta['truth'].items()}
    predictions=predict_file(model,path,cap,trees,baseline=baseline)
    result=metric(predictions,truth,meta['users'],set(meta['complete']))
    return (result,predictions) if return_predictions else result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--users',default='input/test_users.csv');p.add_argument('--out',required=True)
    p.add_argument('--max-users',type=int,default=5000);p.add_argument('--train-users',type=int,default=1500)
    p.add_argument('--train-future-days',type=int,default=150);p.add_argument('--val-future-days',type=int,default=90)
    p.add_argument('--eval-future-days',type=int,nargs='+',default=[60,30]);p.add_argument('--candidate-budgets',type=int,nargs='+',default=[1000,3000])
    p.add_argument('--iterations',type=int,nargs='+',default=[100,200,400]);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--artist-catalog',type=int,default=200000);p.add_argument('--min-support',type=int,default=10)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--cache-dir',default='data/eda_cache')
    a=p.parse_args();out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Choose a new empty --out')
    if not a.train_future_days>a.val_future_days>max(a.eval_future_days)>0:p.error('Require train > val > every evaluation future-days >0')
    if min(a.iterations)<1 or min(a.candidate_budgets)<100 or a.train_users<1:p.error('Invalid training/candidate budget')
    ctx=core.load_context(a.data,a.users,a.max_users,a.seed);out.mkdir(parents=True,exist_ok=True)
    train_cut=ctx.end-a.train_future_days*core.DAY;val_cut=ctx.end-a.val_future_days*core.DAY
    eval_cuts=sorted(set(ctx.end-d*core.DAY for d in a.eval_future_days))
    if train_cut<=ctx.start:p.error('Training cutoff precedes available history')
    val_label_end=min(eval_cuts)
    train_users=sorted(np.random.default_rng(a.seed+1).choice(ctx.users,min(a.train_users,len(ctx.users)),replace=False).tolist())
    capacity=max(a.candidate_budgets);snapshots={}
    schedule=[('train',train_cut,val_cut,train_users),('validation',val_cut,val_label_end,ctx.users)]
    schedule += [(f'development_{int((ctx.end-cut)/core.DAY)}d',cut,ctx.end,ctx.users) for cut in eval_cuts]
    for name,cut,end,users in schedule:
        print(f'BUILD {name}: history < {cut}, labels [{cut}, {end}), users={len(users)}',flush=True)
        snapshots[name]=build_pairs(ctx,cut,end,users,out/'pairs'/f'{name}.parquet',capacity,a.cache_dir,a.artist_catalog,a.min_support)
    # No oracle insertion: every positive must first be retrieved by the historical sources.
    validation=[];models={};chosen=None
    for cap in sorted(set(a.candidate_budgets)):
        print(f'TRAIN CatBoost, candidate budget={cap}',flush=True)
        frame=base.collect(pl.scan_parquet(out/'pairs/train.parquet').filter((pl.col('candidate_rank')<=cap)&(pl.col('n_true')>0)))
        if frame['target'].n_unique()!=2:raise ValueError('Need positive and negative retrieved training examples')
        x=frame.select(FEATURES).to_numpy();y=frame['target'].to_numpy();weights=1/frame['n_true'].to_numpy().clip(1,100)
        model=CatBoostClassifier(iterations=max(a.iterations),depth=6,learning_rate=.05,l2_leaf_reg=10,
            loss_function='Logloss',thread_count=a.threads,random_seed=a.seed,allow_writing_files=False,verbose=100)
        model.fit(x,y,sample_weight=weights);del frame,x,y,weights;gc.collect()
        model.save_model(str(out/f'model_{cap}.cbm'));models[cap]=model
        for trees in sorted(set(a.iterations)):
            row={'budget':cap,'iterations':trees,**evaluate_file(model,out/'pairs/validation.parquet',snapshots['validation'],cap,trees)}
            validation.append(row);print('VALIDATION',row,flush=True)
            if row['n_active']==0:raise ValueError('Validation has no observed future likes')
            if chosen is None or row['recall_active']>chosen['recall_active']:chosen=row
    selected=models[chosen['budget']].copy();selected.shrink(chosen['iterations']);selected.save_model(str(out/'selected_model.cbm'))
    report=base.Report(out);rows=[];paired=[]
    per_budget_best={cap:max((v for v in validation if v['budget']==cap),key=lambda v:v['recall_active']) for cap in models}
    for name,cut,end,users in schedule:
        if name in ['train','validation']:continue
        meta=snapshots[name];path=out/'pairs'/f'{name}.parquet'
        for cap,model in models.items():
            trees=per_budget_best[cap]['iterations']
            baseline,bp=evaluate_file(None,path,meta,cap,0,baseline=True,return_predictions=True)
            result,mp=evaluate_file(model,path,meta,cap,trees,return_predictions=True)
            truth={int(k):set(v) for k,v in meta['truth'].items()}
            delta=np.asarray([(len(set(mp.get(u,[]))&truth[u])-len(set(bp.get(u,[]))&truth[u]))/len(truth[u]) for u in meta['users'] if truth.get(u)])
            paired.append({'period':name,'budget':cap,**core.paired_interval(delta,a.seed,repeats=500)})
            # Each budget uses its own best tree count selected only on validation.
            rows.append({'period':name,'budget':cap,'method':'RRF',**baseline})
            rows.append({'period':name,'budget':cap,'method':'CatBoost','iterations':trees,'selected_budget':cap==chosen['budget'],**result})
    report.table('development_metrics',pl.DataFrame(rows),'CatBoost и RRF на одинаковых кандидатах',
        'Параметры выбраны только на validation. Остальные периоды — проверка разработки, не незатронутый финальный тест. '
        'Цели train/validation ограничены следующим временным срезом: это предотвращает использование будущих меток, но оставляет цензурирование.')
    report.table('paired_improvement',pl.DataFrame(paired),'CatBoost минус RRF: парная разница по активным пользователям','Интервал отражает выборку пользователей, но не будущий временной сдвиг или множественный подбор.')
    report.table('validation_selection',pl.DataFrame(validation),'Выбор числа деревьев и бюджета','Максимизируем реальный Recall@100 на validation, не AUC. Не выбираем модель заново по поздним периодам.')
    imp=selected.get_feature_importance()
    report.table('feature_importance',pl.DataFrame({'feature':FEATURES,'importance':imp}).sort('importance',descending=True),'Важность признаков','Важность среди коррелирующих признаков не является причинным эффектом.')
    for period in sorted(set(r['period'] for r in rows)):
        data=[r for r in rows if r['period']==period];fig,ax=plt.subplots(figsize=(9,4))
        ax.barh([f'{r["method"]}, {r["budget"]}' for r in data],[100*r['recall_active'] for r in data]);ax.invert_yaxis();ax.set_xlabel('Recall@100, активные пользователи (%)')
        report.figure(period,fig,f'Ранжирование: {period}')
    result={'arguments':vars(a),'features':FEATURES,'selected_on_validation':chosen,'validation':validation,'development':rows,'paired_improvement':paired,
            'snapshots':{k:{kk:vv for kk,vv in v.items() if kk not in ['truth','complete','users']} for k,v in snapshots.items()},
            'user_file_sha256':ctx.user_sha256,'evaluation_users':len(ctx.users),'train_users':len(train_users),
            'target':'next 100 chronological like events; unique set metric; repeated old items allowed',
            'development_is_not_final_audit':True}
    report.summary=result;report.finish()


if __name__=='__main__':main()
