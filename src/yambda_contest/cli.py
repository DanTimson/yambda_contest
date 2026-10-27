import argparse
from pathlib import Path
import json
from threadpoolctl import threadpool_limits
from .data import load_users, eda, download
from .retrieval import Config


def main():
    p=argparse.ArgumentParser(description='Yambda: candidate retrieval + chronological CatBoost evaluation')
    sub=p.add_subparsers(dest='cmd',required=True)
    d=sub.add_parser('download');d.add_argument('--data',required=True);d.add_argument('--size',choices=['50m','500m','5b'],default='50m');d.add_argument('--files',nargs='+',default=['likes','artist_item_mapping']);d.add_argument('--revision',default='main')
    e=sub.add_parser('eda');e.add_argument('--data',required=True);e.add_argument('--users',default='input/test_users.csv');e.add_argument('--out')
    a=sub.add_parser('audio');a.add_argument('--data',required=True);a.add_argument('--out',required=True);a.add_argument('--cutoff',type=int,required=True);a.add_argument('--max-items',type=int,default=200000)
    v=sub.add_parser('validate');v.add_argument('file');v.add_argument('--users',default='input/test_users.csv')
    s=sub.add_parser('submit');s.add_argument('--data',required=True);s.add_argument('--model-dir',required=True);s.add_argument('--out',required=True);s.add_argument('--users',default='input/test_users.csv');s.add_argument('--batch-size',type=int,default=128)
    for cmd in ['backtest','fit-final']:
        b=sub.add_parser(cmd);b.add_argument('--data',required=True);b.add_argument('--out',required=True);b.add_argument('--users',default='input/test_users.csv')
        b.add_argument('--train-cutoffs',nargs='+',type=int,required=True)
        if cmd=='backtest':
            b.add_argument('--val-cutoff',required=True,type=int);b.add_argument('--test-cutoff',required=True,type=int);b.add_argument('--iterations',nargs='+',type=int,default=[100,200,400]);b.add_argument('--eval-users',type=int,default=1000);b.add_argument('--embargo-days',type=int,default=0)
        else:
            b.add_argument('--as-of',required=True,type=int);b.add_argument('--iterations',type=int,default=200)
        b.add_argument('--horizon-days',type=int,default=None,help='Optional future observation limit; default all observed future')
        b.add_argument('--target-mode',choices=['next100','window'],default='next100')
        b.add_argument('--train-users',type=int,default=2000);b.add_argument('--threads',type=int,default=4)
        b.add_argument('--history-days',type=int,default=180);b.add_argument('--per-source',type=int,default=150);b.add_argument('--max-candidates',type=int,default=700)
        b.add_argument('--cf-items',type=int,default=30000);b.add_argument('--cf-users',type=int,default=30000);b.add_argument('--knn-neighbors',type=int,default=80)
        b.add_argument('--als',action='store_true');b.add_argument('--audio-dir');b.add_argument('--no-listens',action='store_true');b.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.cmd=='download':return download(args.data,args.size,args.files,args.revision)
    if args.cmd=='audio':
        from .audio import extract
        return extract(args.data,args.out,args.cutoff,args.max_items)
    users=load_users(args.users)
    if args.cmd=='eda':
        r=eda(args.data,users);text=json.dumps(r,indent=2);print(text)
        if args.out:Path(args.out).write_text(text)
        return
    from .pipeline import backtest,final_fit,submit,validate_submission
    if args.cmd=='validate':print(validate_submission(args.file,users));return
    if args.cmd=='submit':
        with threadpool_limits(limits=4):return submit(args.data,args.model_dir,args.out,users,args.batch_size)
    if args.target_mode=='window' and args.horizon_days is None:p.error('window mode requires --horizon-days')
    cfg=Config(history_days=args.history_days,per_source=args.per_source,max_candidates=args.max_candidates,
               cf_items=args.cf_items,cf_users=args.cf_users,knn_neighbors=args.knn_neighbors,use_als=args.als,
               audio_dir=args.audio_dir,use_listens=not args.no_listens,seed=args.seed)
    kwargs=dict(data=args.data,out=args.out,users=users,cfg=cfg,train_cutoffs=args.train_cutoffs,
                horizon_days=args.horizon_days,mode=args.target_mode,iterations=args.iterations,
                threads=args.threads,max_train_users=args.train_users)
    with threadpool_limits(limits=args.threads):
        if args.cmd=='backtest':return backtest(**kwargs,val_cutoff=args.val_cutoff,test_cutoff=args.test_cutoff,eval_users=args.eval_users,embargo_days=args.embargo_days)
        return final_fit(**kwargs,as_of=args.as_of)


if __name__=='__main__':main()
