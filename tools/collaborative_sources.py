"""Past-only population ALS and BM25 item-item sources, with scalar factor features."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import gc
import hashlib
import json
import importlib.metadata
from pathlib import Path
import numpy as np
import polars as pl
from scipy import sparse
from threadpoolctl import threadpool_limits
import explore_yambda as base

SOURCE_NAMES = ['als_likes', 'als_listens', 'bm25_likes', 'bm25_listens']
SIM_NAMES = ['last', 'max5', 'mean5', 'max20', 'mean20']
FEATURES = [s+'_'+k for s in SOURCE_NAMES for k in ['score','rr']] + [
    'cf_als_like_dot','cf_als_listen_dot','cf_als_like_cos','cf_als_listen_cos','cf_item_norm',
    'cf_like_norm','cf_listen_norm','cf_item_known','cf_like_known_fraction','cf_listen_known_fraction'] + [
    'cf_'+kind+'_sim_'+name for kind in ['like','listen'] for name in SIM_NAMES]

@dataclass(frozen=True)
class Config:
    user_mod: int = 3
    history_days: int = 180
    factors: int = 64
    iterations: int = 20
    knn_items: int = 200000
    knn_neighbors: int = 80
    source_budget: int = 1000
    threads: int = 4
    seed: int = 42
    retrieval_device: str = 'auto'


def csr(pairs, users, items, cut, listens=False):
    if pairs.is_empty(): return sparse.csr_matrix((len(users),len(items)),dtype=np.float32),np.zeros(len(users),np.float32)
    uid=np.asarray(pairs['uid']); iid=np.asarray(pairs['item_id']); cols=np.searchsorted(items,iid)
    ok=(cols<len(items));ok &= items[np.minimum(cols,len(items)-1)]==iid
    user_idx={int(u):r for r,u in enumerate(users)}
    rows=np.array([user_idx[int(u)] for u in uid[ok]],np.int32)
    ages=(cut-pairs['timestamp'].to_numpy()[ok].astype(np.int64))/base.DAY
    weights=(1+4*np.exp2(-ages/60)).astype(np.float32)
    if listens: weights=(1+2*np.log1p(np.minimum(pairs['count'].to_numpy()[ok],20))*np.exp2(-ages/30)).astype(np.float32)
    matrix=sparse.csr_matrix((weights,(rows,cols[ok])),shape=(len(users),len(items)),dtype=np.float32)
    total=np.bincount([user_idx[int(u)] for u in uid],minlength=len(users))
    matched=np.bincount(rows,minlength=len(users))
    return matrix,(matched/np.maximum(total,1)).astype(np.float32)


def population_models(data,cut,cache_dir,cfg):
    from implicit.cpu.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import BM25Recommender
    path=Path(data)/'likes.parquet';st=path.stat()
    identity=[str(path.resolve()),st.st_size,st.st_mtime_ns]
    verified_path=Path(data)/'portable_inputs_verified.json'
    if verified_path.exists():
        verified=json.loads(verified_path.read_text())
        if verified['stats'].get('likes.parquet') == identity: identity={'sha256':verified['sha256']['likes.parquet']}
    sig=dict(version='population-cf-v1',source=identity,cut=cut,
             config=asdict(cfg),libraries={p:importlib.metadata.version(p) for p in ['implicit','polars','numpy','scipy']},code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest()[:24]
    folder=Path(cache_dir)/('cf_'+key);folder.mkdir(parents=True,exist_ok=True)
    ap=folder/'als.npz';kp=folder/'bm25.npz'
    if not (ap.exists() and kp.exists() and (folder/'complete.json').exists()):
        corpus_path=folder/'corpus.parquet'
        if not corpus_path.exists():
            print(f'CF CORPUS cutoff={cut}; UID hash sample ~1/{cfg.user_mod}; past {cfg.history_days} days',flush=True)
            q=pl.scan_parquet(path).filter((pl.col('timestamp')<cut)&(pl.col('timestamp')>=cut-cfg.history_days*base.DAY)&
                                          ((pl.col('uid').hash(seed=cfg.seed)%cfg.user_mod)==0))
            pairs=base.collect(q.group_by('uid','item_id').agg(pl.col('timestamp').max()))
            tmp=folder/'corpus.partial.parquet';pairs.write_parquet(tmp);tmp.replace(corpus_path)
        else:pairs=pl.read_parquet(corpus_path)
        if pairs.height<2:raise ValueError('Insufficient past population for collaborative sources')
        users=np.unique(pairs['uid'].to_numpy());items=np.unique(pairs['item_id'].to_numpy())
        matrix,_=csr(pairs,users,items,cut)
        print(f'CF TRAIN users={len(users):,}; tracks={len(items):,}; pairs={matrix.nnz:,}',flush=True)
        with threadpool_limits(limits=1,user_api='blas'):
            if not (ap.exists() and (folder/'als_items.npy').exists()):
                als=AlternatingLeastSquares(factors=cfg.factors,iterations=cfg.iterations,regularization=.1,
                    alpha=1.,random_state=cfg.seed,num_threads=cfg.threads)
                als.fit(matrix,show_progress=True);als.save(folder/'als.partial.npz');(folder/'als.partial.npz').replace(ap)
                np.save(folder/'als_items.npy',items);del als
            if not (kp.exists() and (folder/'knn_items.npy').exists()):
                top=(pairs.group_by('item_id').len().sort(['len','item_id'],descending=[True,False]).head(cfg.knn_items)['item_id'].to_numpy())
                top=np.sort(top)
                recent=pairs.sort(['uid','timestamp','item_id'],descending=[False,True,False]).group_by('uid',maintain_order=True).head(100)
                k_matrix,_=csr(recent,users,top,cut)
                knn=BM25Recommender(K=min(cfg.knn_neighbors,len(top)),num_threads=cfg.threads)
                knn.fit(k_matrix,show_progress=True);knn.save(folder/'bm25.partial.npz');(folder/'bm25.partial.npz').replace(kp)
                np.save(folder/'knn_items.npy',top);del knn,k_matrix,recent
        (folder/'complete.json').write_text(json.dumps(dict(signature=sig,users=len(users),tracks=len(items),pairs=matrix.nnz)))
        del pairs,matrix;gc.collect()
    als=AlternatingLeastSquares.load(ap);als.num_threads=cfg.threads
    knn=BM25Recommender.load(kp);knn.num_threads=cfg.threads
    return als,np.load(folder/'als_items.npy'),knn,np.load(folder/'knn_items.npy'),json.loads((folder/'complete.json').read_text())


class Sources:
    def __init__(self,data,cut,cache_dir,cfg,users,history,listen_pairs):
        self.cfg=cfg;self.users=users;self.ui={int(u):r for r,u in enumerate(users)}
        als,self.items,knn,knn_items,self.info=population_models(data,cut,cache_dir,cfg)
        self.vectors=np.asarray(als.item_factors,dtype=np.float32);self.norms=np.linalg.norm(self.vectors,axis=1)
        lp=history.filter(pl.col('timestamp')>=cut-cfg.history_days*base.DAY).group_by('uid','item_id').agg(pl.col('timestamp').max())
        frames=list(listen_pairs.values())
        if frames:
            heard=pl.concat(frames).filter(pl.col('qualified')>0).select('uid','item_id',pl.col('last_qualified_ts').alias('timestamp'),pl.col('qualified').alias('count'))
            heard=heard.sort(['uid','timestamp','item_id'],descending=[False,True,False]).group_by('uid',maintain_order=True).head(400)
        else:heard=pl.DataFrame(schema={'uid':pl.Int64,'item_id':pl.Int64,'timestamp':pl.Int64,'count':pl.Int64})
        self.seeds={}
        for kind,p in [('like',lp),('listen',heard)]:
            recent=p.sort(['uid','timestamp','item_id'],descending=[False,True,False]).group_by('uid',maintain_order=True).head(20)
            self.seeds[kind]={int(u):g['item_id'].to_numpy() for (u,),g in recent.group_by('uid')}
        self.factors={};self.coverage={};self.ids={};self.scores={}
        torch=None;gpu=False
        if cfg.retrieval_device!='cpu':
            try:
                import torch
                gpu=torch.cuda.is_available()
            except ImportError:pass
            if cfg.retrieval_device=='gpu' and not gpu:raise RuntimeError('GPU retrieval requested, but no CUDA/ROCm PyTorch device is available')
        self.info['retrieval_backend']='torch_gpu' if gpu else 'numpy_cpu'
        print(f'CF RETRIEVAL: {self.info["retrieval_backend"]}; ALS/BM25 fitting: CPU',flush=True)
        gv=torch.as_tensor(self.vectors,device='cuda') if gpu else None
        for kind,p in [('like',lp),('listen',heard)]:
            m,coverage=csr(p,users,self.items,cut,listens=kind=='listen')
            factors=np.zeros((len(users),cfg.factors),np.float32)
            active=np.flatnonzero(np.diff(m.indptr)>0)
            with threadpool_limits(limits=1,user_api='blas'):
                for start in range(0,len(active),128):
                    ix=active[start:start+128];factors[ix]=als.recalculate_user(ix,m[ix])
            self.factors[kind]=factors;self.coverage[kind]=coverage
            k=min(cfg.source_budget,len(self.items));ids=np.full((len(users),k),-1,np.int64);scores=np.zeros((len(users),k),np.float32)
            for start in range(0,len(active),32):
                ix=active[start:start+32]
                if gpu:
                    with torch.no_grad():
                        values,cols=torch.topk(torch.as_tensor(factors[ix],device='cuda')@gv.T,k=k,dim=1)
                        vals=values.cpu().numpy();cols=cols.cpu().numpy()
                else:
                    with threadpool_limits(limits=1,user_api='blas'):values=factors[ix]@self.vectors.T
                    cols=np.argpartition(values,-k,axis=1)[:,-k:];vals=np.take_along_axis(values,cols,axis=1)
                for r,row in enumerate(ix):
                    order=np.lexsort((self.items[cols[r]],-vals[r]));ids[row]=self.items[cols[r][order]];scores[row]=vals[r][order]
            self.ids['als_'+kind+'s']=ids;self.scores['als_'+kind+'s']=scores
            km,_=csr(p,users,knn_items,cut,listens=kind=='listen')
            k=min(cfg.source_budget,len(knn_items));ki=np.full((len(users),k),-1,np.int64);ks=np.zeros((len(users),k),np.float32)
            for row in np.flatnonzero(np.diff(km.indptr)>0):
                cols,vals=knn.recommend(int(row),km[row],N=k,filter_already_liked_items=False)
                cols=np.asarray(cols);vals=np.asarray(vals);ok=(cols>=0)&(cols<len(knn_items))&np.isfinite(vals)&(vals>0)
                cols=cols[ok];vals=vals[ok];order=np.lexsort((knn_items[cols],-vals));n=len(cols)
                ki[row,:n]=knn_items[cols[order]];ks[row,:n]=vals[order]
            self.ids['bm25_'+kind+'s']=ki;self.scores['bm25_'+kind+'s']=ks
        if gv is not None:del gv;torch.cuda.empty_cache()
        del als,knn;gc.collect()

    def lists(self,u):
        row=self.ui[u]
        return {name:[int(i) for i in self.ids[name][row] if i>=0] for name in SOURCE_NAMES}

    def features(self,u,ids):
        row=self.ui[u];cols=np.searchsorted(self.items,ids);known=cols<len(self.items)
        known &= self.items[np.minimum(cols,len(self.items)-1)]==ids
        vec=np.zeros((len(ids),self.cfg.factors),np.float32);vec[known]=self.vectors[cols[known]]
        norms=np.linalg.norm(vec,axis=1);normalized=vec/np.maximum(norms[:,None],1e-12)
        lf=self.factors['like'][row];hf=self.factors['listen'][row];ln=np.linalg.norm(lf);hn=np.linalg.norm(hf)
        maps={name:{int(i):(float(s),1/(r+1)) for r,(i,s) in enumerate(zip(self.ids[name][row],self.scores[name][row])) if i>=0} for name in SOURCE_NAMES}
        source=np.array([[v for name in SOURCE_NAMES for v in maps[name].get(int(i),(0.,0.))] for i in ids],np.float32)
        scalars=np.column_stack([vec@lf,vec@hf,(vec@lf)/np.maximum(norms*ln,1e-12),(vec@hf)/np.maximum(norms*hn,1e-12),
            norms,np.full(len(ids),ln),np.full(len(ids),hn),known.astype(float),
            np.full(len(ids),self.coverage['like'][row]),np.full(len(ids),self.coverage['listen'][row])])
        sims=[]
        for kind in ['like','listen']:
            seeds=self.seeds[kind].get(u,np.array([],dtype=np.int64));ix=np.searchsorted(self.items,seeds)
            ok=ix<len(self.items);ok &= self.items[np.minimum(ix,len(self.items)-1)]==seeds;ix=ix[ok]
            ix=ix[self.norms[ix]>1e-12]
            a=np.full((len(ids),len(SIM_NAMES)),-2.,np.float32)
            if len(ix):
                sim=np.clip(normalized@(self.vectors[ix]/self.norms[ix,None]).T,-1,1)
                a=np.column_stack([sim[:,0],sim[:,:5].max(axis=1),sim[:,:5].mean(axis=1),sim.max(axis=1),sim.mean(axis=1)])
                a[~known]=-2.
            sims.append(a)
        return np.concatenate([source,scalars,*sims],axis=1).astype(np.float32)
