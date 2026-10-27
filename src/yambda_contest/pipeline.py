from dataclasses import asdict
from pathlib import Path
import csv
import gc
import json
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from .data import DAY, targets
from .retrieval import Snapshot
from .metrics import evaluate, candidate_report, paired_bootstrap


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False))


def label_pairs(pairs, truth):
    return pairs.join(truth.select('uid','item_id').unique().with_columns(pl.lit(1).alias('target')),
                      on=['uid','item_id'],how='left').with_columns(pl.col('target').fill_null(0))


def top_predictions(pairs, score, k=100):
    ranked=pairs.select('uid','item_id').with_columns(pl.Series('score',score))
    ranked=ranked.sort(['uid','score','item_id'],descending=[False,True,False]).group_by('uid',maintain_order=True).head(k)
    return {int(u):g['item_id'].to_list() for (u,),g in ranked.group_by('uid',maintain_order=True)}


def classifier(iterations,seed,threads):
    return CatBoostClassifier(iterations=iterations,depth=6,learning_rate=.05,l2_leaf_reg=10,
        loss_function='Logloss',random_seed=seed,thread_count=threads,allow_writing_files=False,verbose=False)


def fit(pairs,features,iterations,seed,threads):
    if pairs['target'].n_unique()!=2:raise ValueError('Need positive and negative candidate labels; increase training cohort/window')
    # For fixed per-user lists, utility is expected hits / min(n_true,100).
    # This weight aligns classifier contributions with macro Recall rather than prolific users.
    model=classifier(iterations,seed,threads)
    model.fit(pairs.select(features).to_numpy(),pairs['target'].to_numpy(),sample_weight=pairs['weight'].to_numpy())
    return model


def backtest(data,out,users,cfg,train_cutoffs,val_cutoff,test_cutoff,horizon_days=None,
             mode='next100',iterations=(100,200,400),threads=4,max_train_users=2000,eval_users=1000, embargo_days=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not train_cutoffs or max(train_cutoffs)>=val_cutoff or val_cutoff>=test_cutoff:
        raise ValueError('Require train cutoffs < validation < audit test')
    if cfg.max_candidates < 100:raise ValueError('Need at least 100 candidate capacity')
    rng=np.random.default_rng(cfg.seed)
    cohort=sorted(rng.choice(users,min(eval_users,len(users)),replace=False).tolist())
    train_users=sorted(rng.choice(users,min(max_train_users,len(users)),replace=False).tolist())
    train_frames=[];features=None; train_reports=[]
    for cut in train_cutoffs:
        # Prevent later labels from entering training: next100 may otherwise extend past val.
        label_end=val_cutoff-embargo_days*DAY
        if label_end<=cut:raise ValueError('Embargo leaves no training labels')
        effective_days=(label_end-cut)/DAY
        if horizon_days is not None:effective_days=min(effective_days,horizon_days)
        print(f'BUILD train cutoff={cut}, label_end={cut+effective_days*DAY}',flush=True)
        snap=Snapshot(data,cut,train_users,cfg);features=snap.features
        truth=targets(Path(data)/'likes.parquet',cut,train_users,effective_days,mode)
        pairs=snap.generate(train_users)
        train_reports.append({'cutoff':cut,**candidate_report(pairs,truth,train_users)})
        counts=truth.group_by('uid').len().rename({'len':'n_true'})
        pairs=label_pairs(pairs,truth).join(counts,on='uid',how='left').with_columns(
            (1/pl.col('n_true').clip(1,100)).alias('weight'))
        # No-future groups have no observed positives and ambiguous right censoring; not training negatives.
        pairs=pairs.filter(pl.col('n_true').is_not_null()).drop('n_true')
        train_frames.append(pairs)
        del snap;gc.collect()
    train=pl.concat(train_frames,how='vertical_relaxed')
    train.write_parquet(out/'train_pairs.parquet')
    del train_frames;gc.collect()
    print(f'BUILD validation cutoff={val_cutoff}',flush=True)
    val=Snapshot(data,val_cutoff,cohort,cfg)
    vp=val.generate(cohort)
    # Validation labels stop before audit test, irrespective of next100 target selection.
    vh=(test_cutoff-val_cutoff)/DAY
    if horizon_days is not None:vh=min(vh,horizon_days)
    vt=targets(Path(data)/'likes.parquet',val_cutoff,cohort,vh,mode)
    rows=[];best_model=None;best_score=-1.;best_it=None
    for it in sorted(set(iterations)):
        m=fit(train,features,it,cfg.seed,threads)
        rec=top_predictions(vp,m.predict_proba(vp.select(features).to_numpy())[:,1])
        metric=evaluate(rec,vt,cohort)
        rows.append({'iterations':it,**metric})
        print('VALIDATION',rows[-1],flush=True)
        if metric['n_active']==0:raise ValueError('No validation positives')
        if metric['recall_active']>best_score:
            best_score=metric['recall_active'];best_it=it;best_model=m
    best_model.save_model(str(out/'model.cbm'))
    # Preserve the selected model; the audit evaluates exactly this model, without refit on validation.
    vp.write_parquet(out/'val_candidates.parquet');vt.write_parquet(out/'val_truth.parquet')
    del val,vp,vt,train;gc.collect()
    selected={'config':asdict(cfg),'features':features,'best_iterations':best_it,'train_cutoffs':train_cutoffs,
              'val_cutoff':val_cutoff,'test_cutoff':test_cutoff,'horizon_days':horizon_days,'target_mode':mode,
              'cohort':cohort,'train_users':train_users,'train_reports':train_reports,'validation':rows,
              'label_rule':'first-ever unique likes; strict timestamp < cutoff; ties item_id ascending',
              'audit_refit':False,'embargo_days':embargo_days}
    manifest=Path(data)/'dataset_manifest.json'
    selected['dataset_manifest']=json.loads(manifest.read_text()) if manifest.exists() else None
    dump(out/'experiment.json',selected)
    print('BUILD audit test (configuration is frozen)',flush=True)
    snap=Snapshot(data,test_cutoff,cohort,cfg)
    truth=targets(Path(data)/'likes.parquet',test_cutoff,cohort,horizon_days,mode)
    p=snap.generate(cohort);p.write_parquet(out/'test_candidates.parquet');truth.write_parquet(out/'test_truth.parquet')
    pred=top_predictions(p,best_model.predict_proba(p.select(features).to_numpy())[:,1])
    rrf=top_predictions(p,p['rrf'].to_numpy())
    popular={};recent={}
    for u in cohort:
        seen=snap.seen.get(u,set())
        popular[u]=[i for i,_ in snap.pop if i not in seen][:100]
        recent[u]=[i for i,_ in snap.recent if i not in seen][:100]
    report={'catboost':evaluate(pred,truth,cohort),'rrf':evaluate(rrf,truth,cohort),
            'popular':evaluate(popular,truth,cohort),'recent':evaluate(recent,truth,cohort),
            'candidates':candidate_report(p,truth,cohort),
            'catboost_minus_rrf':paired_bootstrap(pred,rrf,truth,cohort),
            'sources':{},'source_leave_one_out':{}}
    for s in snap.sources:
        sub=p.filter(pl.col(s+'_rr')>0)
        report['sources'][s]=candidate_report(sub,truth,cohort)
        without=p.filter((pl.col(s+'_rr')==0) | (pl.col('n_sources')>1))
        report['source_leave_one_out'][s]=candidate_report(without,truth,cohort)
    # Distinguish cold users from users with known likes; don't silently discard them.
    for name,us in [('warm',[u for u in cohort if snap.seen.get(u)]),('cold',[u for u in cohort if not snap.seen.get(u)])]:
        report[name]=evaluate(pred,truth,us)
    dump(out/'test_predictions.json',pred);dump(out/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def final_fit(data,out,users,cfg,train_cutoffs,as_of,iterations=200,horizon_days=None,mode='next100',threads=4,max_train_users=3000):
    """Train only on labels available before deployment cutoff; save model plus schema."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not train_cutoffs or max(train_cutoffs)>=as_of:raise ValueError('Training cutoffs must precede as_of')
    train_users=sorted(np.random.default_rng(cfg.seed).choice(users,min(max_train_users,len(users)),replace=False).tolist())
    frames=[];features=None
    for cut in train_cutoffs:
        snap=Snapshot(data,cut,train_users,cfg);features=snap.features
        days=(as_of-cut)/DAY
        if horizon_days is not None:days=min(days,horizon_days)
        truth=targets(Path(data)/'likes.parquet',cut,train_users,days,mode)
        p=label_pairs(snap.generate(train_users),truth)
        counts=truth.group_by('uid').len().rename({'len':'n_true'})
        p=p.join(counts,on='uid',how='inner').with_columns((1/pl.col('n_true').clip(1,100)).alias('weight')).drop('n_true')
        frames.append(p);del snap;gc.collect()
    model=fit(pl.concat(frames,how='vertical_relaxed'),features,iterations,cfg.seed,threads)
    model.save_model(str(out/'model.cbm'))
    dump(out/'deployment.json',{'config':asdict(cfg),'features':features,'as_of':as_of,'iterations':iterations,'train_cutoffs':train_cutoffs,'horizon_days':horizon_days,'target_mode':mode})


def submit(data,model_dir,out,users,batch_size=128,variants=('catboost','rrf','recent')):
    from .retrieval import Config
    root=Path(model_dir);meta=json.loads((root/'deployment.json').read_text())
    cfg=Config(**meta['config']);as_of=meta['as_of']
    model=CatBoostClassifier();model.load_model(str(root/'model.cbm'))
    snap=Snapshot(data,as_of,users,cfg)
    if snap.features!=meta['features']:raise ValueError('Feature schema differs')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    handles={v:open(out/f'{v}.csv','w',newline='') for v in variants}
    writers={v:csv.writer(h) for v,h in handles.items()}
    for w in writers.values():w.writerow(['uid','item_ids'])
    try:
        for start in range(0,len(users),batch_size):
            batch=users[start:start+batch_size];p=snap.generate(batch)
            preds={'catboost':top_predictions(p,model.predict_proba(p.select(snap.features).to_numpy())[:,1]),
                   'rrf':top_predictions(p,p['rrf'].to_numpy()),
                   'recent':{u:[i for i,s in snap.recent if i not in snap.seen.get(u,set())][:100] for u in batch}}
            for variant,w in writers.items():
                for u in batch:
                    ids=preds[variant].get(u,[])
                    if len(ids)!=100 or len(set(ids))!=100:raise ValueError(f'uid {u}: cannot provide 100 unique recommendations')
                    w.writerow([u,' '.join(map(str,ids))])
            print(f'Exported {min(start+batch_size,len(users))}/{len(users)} users',flush=True)
    finally:
        for h in handles.values():h.close()
    for v in variants:validate_submission(out/f'{v}.csv',users)
    import zipfile
    with zipfile.ZipFile(out/'submissions.zip','w',zipfile.ZIP_DEFLATED) as z:
        for v in variants:z.write(out/f'{v}.csv',f'{v}.csv')


def validate_submission(path,users):
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['uid','item_ids']:raise ValueError('Wrong columns')
        actual=[]
        for row in reader:
            actual.append(int(row['uid']));ids=[int(i) for i in row['item_ids'].split()]
            if len(ids)!=100 or len(set(ids))!=100 or any(i<0 for i in ids):raise ValueError('Need exactly 100 unique nonnegative integer IDs')
    if actual!=users:raise ValueError('uid sequence differs from test_users')
    return {'rows':len(actual),'valid':True}
