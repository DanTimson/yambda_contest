#!/usr/bin/env python3
"""Batch benchmark: temporal popularity, growth, artist candidates and balanced fusion."""
from __future__ import annotations
from collections import defaultdict
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import explore_trends as core
import explore_yambda as base

VERSION='candidate-eda-v1'


def cached_stats(data,cutoff,cache_dir=None):
    path=Path(data)/'likes.parquet';stat=path.stat()
    # Include size and modification time so changing a local file invalidates its cache.
    signature={'version':VERSION,'file':str(path.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'cutoff':cutoff,
               'statistics_code_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
    cache=Path(cache_dir)/f'{key}.parquet' if cache_dir else None
    if cache is not None and cache.exists():
        base.log(f'Using cached past-only statistics: {cache.name}')
        return pl.read_parquet(cache),signature
    # The source function returns the complete statistics; budget does not affect them.
    stats,_=core.trend_sources(path,cutoff,pool_size=100)
    if cache is not None:
        cache.parent.mkdir(parents=True,exist_ok=True)
        temp=cache.with_suffix('.part')
        stats.write_parquet(temp);temp.replace(cache)
        cache.with_suffix('.json').write_text(json.dumps(signature,indent=2))
    return stats,signature


def global_lists(stats,budget,min_support):
    result={}
    for name in core.SOURCES:
        df=stats
        if name in ['growth7','momentum7']:
            df=df.filter((pl.col('recent_users')>=min_support)&(pl.col(name)>0))
        else:df=df.filter(pl.col(name)>0)
        result[name]=df.sort([name,'item_id'],descending=[True,False]).head(budget)['item_id'].to_list()
    return result


class ArtistSource:
    """All credit relationships, bounded past-only catalog, no future labels accepted."""
    def __init__(self,data,history,stats,cutoff,catalog_size=200000,top_artists=5,per_artist=200):
        self.preferences={};self.items={'artist_pop':{},'artist_trend':{}}
        self.info={'available':False,'catalog_items':0,'users_with_profile':0,'top_artists':top_artists,'per_artist':per_artist}
        path=Path(data)/'artist_item_mapping.parquet'
        if not path.exists() or not history.height:return
        # Fixed-budget catalog: 60% general, 20% fresh, 20% rising; round-robin union then fill.
        quotas=[max(1,int(catalog_size*.6)),max(1,int(catalog_size*.2)),max(1,int(catalog_size*.2))]
        columns=['pop_all','decay7','growth7'];chosen=set()
        for col,n in zip(columns,quotas):
            chosen.update(stats.filter(pl.col(col)>0).sort([col,'item_id'],descending=[True,False]).head(n)['item_id'].to_list())
        for item in stats.sort(['decay7','item_id'],descending=[True,False])['item_id']:
            if len(chosen)>=catalog_size:break
            chosen.add(int(item))
        catalog=stats.filter(pl.col('item_id').is_in(list(chosen)))
        hist=history.group_by('uid','item_id').agg(pl.col('timestamp').max())
        hist=hist.filter(pl.col('timestamp')>=cutoff-180*core.DAY)
        needed=set(hist['item_id'].to_list())|chosen
        mapping=base.collect(pl.scan_parquet(path).select('item_id','artist_id').filter(pl.col('item_id').is_in(list(needed))).unique()).with_columns(pl.col('item_id').cast(pl.Int64))
        prof=hist.join(mapping,on='item_id',how='inner').with_columns((-(cutoff-pl.col('timestamp').cast(pl.Float64))/(60*core.DAY)).exp().alias('affinity'))
        prof=prof.group_by('uid','artist_id').agg(pl.col('affinity').sum()).sort(['uid','affinity','artist_id'],descending=[False,True,False]).group_by('uid',maintain_order=True).head(top_artists)
        self.preferences={int(u):g.select('artist_id','affinity').rows() for (u,),g in prof.group_by('uid')}
        tracks=catalog.join(mapping,on='item_id',how='inner').with_columns(pl.col('pop_all').cast(pl.Float64).log1p().alias('artist_pop'),
            (pl.col('decay7').log1p()*pl.col('ratio7').clip(.5,3.)).alias('artist_trend'))
        for name in self.items:
            ordered=tracks.sort(['artist_id',name,'item_id'],descending=[False,True,False]).group_by('artist_id',maintain_order=True).head(per_artist)
            self.items[name]={int(a):g.select('item_id',name).rows() for (a,),g in ordered.group_by('artist_id')}
        self.info.update(available=True,catalog_items=catalog.height,users_with_profile=len(self.preferences),
                         profile_lookback_days=180,profile_time_constant_days=60,
                         formula_artist_pop='sum affinity(artist) * log1p(global_like_count)',
                         formula_artist_trend='sum affinity(artist) * log1p(decay7) * clip(ratio7,0.5,3)')

    def recommend(self,user,budget):
        result={}
        for name,lookup in self.items.items():
            score=defaultdict(float)
            for artist,weight in self.preferences.get(user,[]):
                for item,value in lookup.get(int(artist),[]):score[int(item)]+=float(weight*value)
            result[name]=[i for i,s in sorted(score.items(),key=lambda p:(-p[1],p[0]))[:budget]]
        return result


def scores_for_user(items,truth):
    den=len(truth)
    if not den:return 0.,0.
    return len(truth&set(items[:100]))/den,len(truth&set(items))/den


def arrays_summary(score,coverage,active,complete,warm):
    def mean(a,m):return float(a[m].mean()) if m.any() else None
    return {'recall100_all':float(score.mean()),'recall100_active':mean(score,active),
            'recall100_full100':mean(score,complete),'candidate_recall_active':mean(coverage,active),
            'candidate_recall_full100':mean(coverage,complete),
            'candidate_recall_warm_active':mean(coverage,warm&active),'candidate_recall_cold_active':mean(coverage,(~warm)&active)}


def benchmark(ctx,cutoffs,out,budget=1000,min_support=10,artist_catalog=200000,cache_dir=None,with_artists=True):
    if budget<100 or min_support<1 or artist_catalog<5:raise ValueError('Invalid experiment budget')
    cutoffs=sorted(set(cutoffs))
    if not cutoffs or any(not ctx.start<c<ctx.end for c in cutoffs):raise ValueError('Cutoffs must be inside log range')
    report=core.new_report(out);started=time.perf_counter();all_rows=[];snapshots=[]
    report.prose('Пакет экспериментов: тренды и персональные кандидаты',
        'В одном запуске сравниваем восемь временных источников, два источника по любимым артистам и объединения. '
        'Кандидаты строятся строго до среза. Все сравнения используют одну и ту же группу UID и цель из следующих 100 событий лайка. '
        'Никакие нейросети или деревья здесь не обучаются. Это подготовка кандидатов для последующего CatBoost.')
    for cut in cutoffs:
        tick=time.perf_counter();base.log(f'BENCHMARK cutoff={cut}, future days={(ctx.end-cut)/core.DAY:.0f}')
        h,t=core.history_targets(ctx,cut);ts,n,complete,warm=core.cohort_arrays(t,ctx.users,h);active=n>0
        stats,signature=cached_stats(ctx.data,cut,cache_dir)
        lists=global_lists(stats,budget,min_support)
        trend_rrf=core.fuse([lists[s] for s in core.SOURCES],budget)
        union=set(i for v in lists.values() for i in v)
        fixed_lists={**lists,'rrf_trends':trend_rrf}
        fixed_top={k:set(v[:100]) for k,v in fixed_lists.items()}
        fixed_pool={k:set(v) for k,v in fixed_lists.items()}
        artist=ArtistSource(ctx.data,h,stats,cut,artist_catalog) if with_artists else None
        if artist is None or not artist.info['available']:
            report.prose(f'Артисты — срез {cut}', 'Источник артистов недоступен или отключён. Его пустые списки не следует интерпретировать как измеренную неэффективность метода.')
        names=core.SOURCES+['rrf_trends','artist_pop','artist_trend','rrf_personal','union_oracle']
        scores={name:np.zeros(len(ctx.users)) for name in names};coverage={name:np.zeros(len(ctx.users)) for name in names}
        extra={name:np.zeros(len(ctx.users)) for name in ['artist_beyond_trends','artist_trend_beyond_artist_pop']}
        poolsizes={name:[] for name in names}
        for j,(u,true) in enumerate(zip(ctx.users,ts)):
            personal=artist.recommend(u,budget) if artist else {'artist_pop':[],'artist_trend':[]}
            # Equal-weight families: first fuse eight global sources, then two artist sources,
            # then fuse the two families. Eight global votes cannot swamp one personal vote.
            artist_rrf=core.fuse(list(personal.values()),budget)
            fused=core.fuse([trend_rrf,artist_rrf],budget) if artist_rrf else trend_rrf
            methods={**lists,'rrf_trends':trend_rrf,**personal,'rrf_personal':fused}
            personal_pool=set(personal['artist_pop'])|set(personal['artist_trend'])
            for name,items in methods.items():
                if name in fixed_lists:
                    scores[name][j]=len(true&fixed_top[name])/len(true) if true else 0.
                    coverage[name][j]=len(true&fixed_pool[name])/len(true) if true else 0.
                else:scores[name][j],coverage[name][j]=scores_for_user(items,true)
                poolsizes[name].append(len(items))
            coverage['union_oracle'][j]=len((true&union)|(true&personal_pool))/len(true) if true else 0.
            poolsizes['union_oracle'].append(len(union)+len(personal_pool-union))
            if true:
                extra['artist_beyond_trends'][j]=len((true&personal_pool)-fixed_pool['rrf_trends'])/len(true)
                extra['artist_trend_beyond_artist_pop'][j]=len((true&set(personal['artist_trend']))-set(personal['artist_pop']))/len(true)
        metrics=[]
        for name in names:
            row={'cutoff':cut,'source':name,**arrays_summary(scores[name],coverage[name],active,complete,warm),
                 'mean_pool_size':float(np.mean(poolsizes[name])),'max_pool_size':max(poolsizes[name],default=0)}
            if name=='union_oracle':
                row['recall100_all']=row['recall100_active']=row['recall100_full100']=None
            metrics.append(row);all_rows.append(row)
        deltas=[]
        for name in names:
            if name in ['pop_all','union_oracle']:continue
            # Bootstrap repeats reduced to 300 for an exploratory multi-source sweep.
            d=core.paired_interval((scores[name]-scores['pop_all'])[active],ctx.seed,repeats=300)
            deltas.append({'cutoff':cut,'source':name,**d})
        opportunity={k:float(v[active].mean()) if active.any() else None for k,v in extra.items()}
        cohort={'users':len(ctx.users),'active_users':int(active.sum()),'full100_users':int(complete.sum()),
                'cold_history_users':int((~warm).sum()),'future_days':(ctx.end-cut)/core.DAY}
        report.table(f'metrics_{cut}',pl.DataFrame(metrics),f'Качество источников — срез {cut}',
            'Кандидатные пулы ограничены одинаковым максимумом budget. Короткие списки не дополняются искусственно. union_oracle имеет больший бюджет и не является готовым рекомендателем.')
        report.table(f'deltas_{cut}',pl.DataFrame(deltas),f'Разница Recall@100 с общей популярностью — {cut}',
            'Парный бутстрэп по активным пользователям. Интервалы не учитывают множественный подбор или будущую смену трендов.')
        report.table(f'complementarity_{cut}',pl.DataFrame([opportunity]),f'Дополнительные попадания артистов — {cut}',
            'Это добавочный потенциал большего объединения, а не выигрыш при неизменном бюджете; фиксированный бюджет проверяет rrf_personal.')
        display=[m for m in metrics if m['source']!='union_oracle']
        fig,axes=plt.subplots(1,2,figsize=(13,7));labels=[m['source'] for m in display]
        for ax,key,label in [(axes[0],'recall100_active','Recall@100, активные (%)'),(axes[1],'candidate_recall_active',f'Полнота кандидатов, максимум {budget} (%)')]:
            ax.barh(labels,[m[key]*100 if m[key] is not None else np.nan for m in display]);ax.invert_yaxis();ax.set_xlabel(label)
        report.figure(f'comparison_{cut}',fig,f'Общие и персональные источники — срез {cut}',
            'Популярность и свежесть коррелируют. Артисты добавляют персональный сигнал. Имена методов, формулы и ограничения записаны в инструкции и summary.json.')
        for category,methods in [('recency',['pop_all','pop30','pop7','pop2','decay7','decay30']),('growth',['pop_all','growth7','momentum7']),('personal',['rrf_trends','artist_pop','artist_trend','rrf_personal'])]:
            small=[m for m in metrics if m['source'] in methods]
            fig,ax=plt.subplots(figsize=(9,4));ax.barh([m['source'] for m in small],[m['candidate_recall_active']*100 if m['candidate_recall_active'] is not None else np.nan for m in small]);ax.invert_yaxis();ax.set_xlabel('Полнота кандидатов, активные пользователи (%)')
            report.figure(f'{category}_{cut}',fig,f'{category}: полнота при одинаковом предельном бюджете — {cut}')
        examples=stats.filter(pl.col('item_id').is_in(lists['growth7'][:20])).sort(['growth7','item_id'],descending=[True,False])
        report.table(f'rising_tracks_{cut}',examples,f'Проверка растущих треков — {cut}',
            'prev7/pop7 — число событий, recent_users — число различных пользователей за последнюю неделю. Порог роста применяется по различным пользователям, чтобы один повторяющийся пользователь не создал тренд.')
        snapshot={'cutoff':cut,'cohort':cohort,'metrics':metrics,'deltas_vs_pop_all':deltas,'additional_candidate_opportunities':opportunity,
                  'artist_source':artist.info if artist else {'available':False},'source_lists':lists,'statistics_input':signature,
                  'seconds':time.perf_counter()-tick}
        snapshots.append(snapshot)
        # A completed cutoff is reviewable even if a later cutoff fails. No hidden resume behavior.
        checkpoint=Path(out)/f'checkpoint_{cut}.json';checkpoint.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2));report.files.append(checkpoint)
        del stats,artist,h,t;gc.collect()
    report.table('all_metrics',pl.DataFrame(all_rows),'Сводка по временным срезам','Оцениваем устойчивость относительных выигрышей внутри срезов, а не сравниваем абсолютные Recall разных горизонтов как одну задачу.')
    result={'version':VERSION,'users':len(ctx.users),'seed':ctx.seed,'user_file_sha256':ctx.user_sha256,
            'parameters':{'budget':budget,'min_recent_distinct_users':min_support,'artist_catalog':artist_catalog,'with_artists':with_artists},
            'target':'next 100 chronological like events; set metric; no permanent seen exclusion',
            'snapshots':snapshots,'seconds':time.perf_counter()-started}
    manifest=ctx.data/'dataset_manifest.json'
    if manifest.exists():result['dataset_manifest']=json.loads(manifest.read_text())
    report.summary=result;report.finish();return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--users',default='input/test_users.csv');p.add_argument('--out',required=True)
    p.add_argument('--max-users',type=int,default=5000);p.add_argument('--seed',type=int,default=42);p.add_argument('--budget',type=int,default=1000)
    p.add_argument('--future-days',type=int,nargs='+',default=[120,90,60,30]);p.add_argument('--cutoffs',type=int,nargs='+')
    p.add_argument('--min-support',type=int,default=10);p.add_argument('--artist-catalog',type=int,default=200000)
    p.add_argument('--no-artists',action='store_true');p.add_argument('--cache-dir',default='data/eda_cache')
    p.add_argument('--with-listens',action='store_true');p.add_argument('--listen-users',type=int,default=1000)
    a=p.parse_args()
    if Path(a.out).exists() and any(Path(a.out).iterdir()):p.error('Choose a new empty --out')
    if any(d<=0 for d in a.future_days):p.error('future-days must be positive')
    ctx=core.load_context(a.data,a.users,a.max_users,a.seed)
    cuts=a.cutoffs or [ctx.end-d*core.DAY for d in a.future_days]
    result=benchmark(ctx,cuts,a.out,a.budget,a.min_support,a.artist_catalog,a.cache_dir,not a.no_artists)
    if a.with_listens:
        if not (ctx.data/'listens.parquet').exists():
            base.log('Listens skipped: no local listens.parquet. Main benchmark completed successfully.')
        else:
            cut=max(cuts);reference=core.fuse(list(next(s for s in result['snapshots'] if s['cutoff']==cut)['source_lists'].values()),a.budget)
            core.run_listens(ctx,cut,Path(a.out)/'listens',a.listen_users,reference)
    import zipfile
    archive=Path(a.out)/'all_experiments_for_review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(a.out).rglob('*')):
            if f.is_file() and f.suffix.lower() in {'.html','.json','.csv','.svg'}:
                z.write(f,f.relative_to(a.out))
    base.log(f'RETURN ALL RESULTS: {archive.resolve()}')


if __name__=='__main__':main()
