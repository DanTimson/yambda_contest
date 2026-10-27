from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.preprocessing import normalize
from .data import DAY, events, first_likes


@dataclass
class Config:
    history_days: int = 180
    per_source: int = 150
    max_candidates: int = 700
    cf_items: int = 30000
    cf_users: int = 30000
    cf_history: int = 200
    knn_neighbors: int = 80
    knn_block: int = 32
    seed_items: int = 30
    use_als: bool = False
    als_factors: int = 64
    als_iterations: int = 15
    audio_dir: str | None = None
    use_listens: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.history_days <= 0 or self.per_source <= 0 or self.max_candidates < 100:
            raise ValueError('Require positive history/source budgets and at least 100 candidate capacity')
        if self.cf_items < 0 or self.cf_users < 0 or self.knn_block <= 0 or self.knn_neighbors <= 0:
            raise ValueError('Invalid collaborative filtering budget')


class Snapshot:
    """All learned artifacts are fitted inside this cutoff, never on the target window."""
    def __init__(self, data, cutoff, users, cfg):
        self.cutoff, self.cfg = int(cutoff), cfg
        q = pl.scan_parquet(Path(data)/"likes.parquet").filter(pl.col("timestamp") < cutoff)
        history = q.filter(pl.col("timestamp") >= cutoff-cfg.history_days*DAY).collect(engine="streaming")
        if not history.height: raise ValueError("No pre-cutoff training likes")
        self.h = first_likes(history)
        # First-ever exclusion over ALL available history, not just feature lookback.
        seen = q.filter(pl.col("uid").is_in(users)).select("uid","item_id").unique().collect(engine="streaming")
        self.seen = {u:set(g["item_id"].to_list()) for (u,),g in seen.group_by("uid")}
        self.user_history = {u:g.sort(["timestamp","item_id"]) for (u,),g in self.h.filter(pl.col("uid").is_in(users)).group_by("uid")}
        self.item = self.h.group_by("item_id").agg(
            pl.len().cast(pl.Float32).alias("pop180"),
            (pl.col("timestamp") >= cutoff-30*DAY).sum().cast(pl.Float32).alias("pop30"),
            (pl.col("timestamp") >= cutoff-7*DAY).sum().cast(pl.Float32).alias("pop7"),
            (pl.col("timestamp") >= cutoff-14*DAY).sum().cast(pl.Float32).alias("pop14"),
            pl.col("timestamp").max().alias("last_ts"),
            pl.col("timestamp").min().alias("first_ts"),
            ((-(cutoff-pl.col("timestamp").cast(pl.Float64))/(7*DAY)).exp()).sum().cast(pl.Float32).alias("decay"),
        ).with_columns(((pl.col("pop7")+.5)/(pl.col("pop14")-pl.col("pop7")+.5)).alias("trend"))
        self.stats = {r['item_id']:r for r in self.item.to_dicts()}
        self.catalog = set(self.stats)
        def ranked(col):
            return [(int(i),float(s)) for i,s in self.item.sort([col,"item_id"],descending=[True,False]).select("item_id",col).iter_rows()]
        self.pop = ranked("pop180")
        self.recent = ranked("decay")
        # Smoothed growth weighted by current volume, so singleton ratios cannot dominate.
        tr = self.item.with_columns((pl.col("trend")*pl.col("pop7").log1p()).alias("trend_score"))
        self.trend = list(tr.sort(["trend_score","item_id"],descending=[True,False]).select("item_id","trend_score").iter_rows())
        self.artists = defaultdict(list)
        mapping = Path(data)/"artist_item_mapping.parquet"
        if mapping.exists():
            m = pl.scan_parquet(mapping).filter(pl.col("item_id").is_in(list(self.catalog))).select("item_id","artist_id").unique().collect()
            for i,a in m.iter_rows(): self.artists[int(i)].append(int(a))
        self.artist_items = defaultdict(list)
        # Keep every credited artist; no arbitrary min(artist_id) reduction.
        for i,s in self.recent:
            for a in self.artists[i]:
                if len(self.artist_items[a]) < cfg.per_source*2:
                    self.artist_items[a].append((i,s))
        self.listens = {}
        lp = Path(data)/"listens.parquet"
        if cfg.use_listens and lp.exists():
            l = pl.scan_parquet(lp).filter((pl.col("timestamp") < cutoff) & (pl.col("timestamp") >= cutoff-90*DAY) & pl.col("uid").is_in(users))
            if "played_ratio_pct" in l.collect_schema(): l = l.filter(pl.col("played_ratio_pct") >= 50)
            l = l.group_by("uid","item_id").agg(pl.len().alias("listen_count"),pl.col("timestamp").max().alias("listen_ts")).collect(engine="streaming")
            self.listens = {u:{r['item_id']:(r['listen_count'],r['listen_ts']) for r in g.to_dicts()} for (u,),g in l.group_by("uid")}
        self.cf_ids = np.array([i for i,_ in self.pop[:cfg.cf_items]],dtype=np.int64)
        self.cf_index = {int(i):j for j,i in enumerate(self.cf_ids)}
        self.knn = None; self.als = None
        if len(self.cf_ids) > 1 and cfg.cf_users > 0:
            self._fit_cf()
        self.audio = None
        if cfg.audio_dir:
            root=Path(cfg.audio_dir); meta=json.loads((root/'metadata.json').read_text())
            if meta['cutoff'] > cutoff:
                raise ValueError("Audio item selection was made after this cutoff; extract at earliest training cutoff")
            ids=np.load(root/'ids.npy'); vec=np.load(root/'vectors.npy',mmap_mode='r')
            self.audio=(ids,vec,{int(v):i for i,v in enumerate(ids)})
        self.sources = ['pop','recent','trend','artist','knn','listened']
        if cfg.use_als: self.sources.append('als')
        if cfg.audio_dir: self.sources.append('audio')
        self.features = [f'{s}_{kind}' for s in self.sources for kind in ['score','rr']]+[
            'rrf','n_sources','pop180','pop30','pop7','trend_ratio','item_age_days','item_recency_days',
            'user_likes','user_recency_days','artist_affinity','listen_count','listen_recency_days','audio_cos']

    def _fit_cf(self):
        cfg=self.cfg
        # User subsampling limits CPU and memory; item universe always comes from this history only.
        us=np.sort(self.h['uid'].unique().to_numpy())
        if len(us)>cfg.cf_users: us=np.sort(np.random.default_rng(cfg.seed).choice(us,cfg.cf_users,replace=False))
        h=self.h.filter(pl.col('uid').is_in(us.tolist()) & pl.col('item_id').is_in(self.cf_ids.tolist()))
        h=h.sort(['uid','timestamp','item_id']).group_by('uid',maintain_order=True).tail(cfg.cf_history)
        if not h.height: return
        rows=np.searchsorted(us,h['uid'].to_numpy())
        cols=np.array([self.cf_index[int(i)] for i in h['item_id']],dtype=np.int32)
        x=sparse.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(us),len(self.cf_ids)))
        x.sum_duplicates(); x.data[:]=1
        # Cosine similarity, then retain a bounded number of neighbors per item, in small blocks.
        z=normalize(x,axis=0).tocsr(); zt=z.T.tocsr(); rr=[]; cc=[]; vv=[]
        for start in range(0,len(self.cf_ids),cfg.knn_block):
            product=(zt[start:start+cfg.knn_block]@z).tocsr()
            for j in range(product.shape[0]):
                row=product.getrow(j); ids=row.indices; vals=row.data
                mask=ids!=start+j; ids,vals=ids[mask],vals[mask]
                if len(vals)>cfg.knn_neighbors:
                    chosen=np.argpartition(vals,-cfg.knn_neighbors)[-cfg.knn_neighbors:];ids,vals=ids[chosen],vals[chosen]
                rr.extend([start+j]*len(vals));cc.extend(ids.tolist());vv.extend(vals.tolist())
        self.knn=sparse.csr_matrix((vv,(rr,cc)),shape=(len(self.cf_ids),len(self.cf_ids)),dtype=np.float32)
        if cfg.use_als:
            from implicit.cpu.als import AlternatingLeastSquares
            self.als=AlternatingLeastSquares(factors=cfg.als_factors,iterations=cfg.als_iterations,regularization=.1,alpha=20,random_state=cfg.seed,num_threads=4)
            self.als.fit(x,show_progress=False)

    def _profile(self,u):
        h=self.user_history.get(u)
        affinity=Counter(); seeds=[]
        if h is not None:
            seeds=h.tail(self.cfg.seed_items).select('item_id','timestamp').rows()
            for i,t in h.select('item_id','timestamp').iter_rows():
                for a in self.artists[i]: affinity[a]+=float(np.exp(-(self.cutoff-t)/(60*DAY)))
        return h,affinity,seeds

    def candidates(self,u):
        cfg=self.cfg;seen=self.seen.get(u,set());h,aff,seeds=self._profile(u)
        sources={'pop':self.pop,'recent':self.recent,'trend':self.trend}
        artist=Counter()
        for a,w in aff.most_common(10):
            for i,s in self.artist_items[a]: artist[i]+=w*np.log1p(s)
        sources['artist']=artist.most_common()
        knn=Counter()
        for i,t in seeds:
            j=self.cf_index.get(i)
            if j is None or self.knn is None: continue
            row=self.knn.getrow(j);w=np.exp(-(self.cutoff-t)/(30*DAY))
            for k,v in zip(row.indices,row.data):knn[int(self.cf_ids[k])]+=float(w*v)
        sources['knn']=knn.most_common()
        heard=self.listens.get(u,{})
        sources['listened']=sorted(((i,np.log1p(n)*np.exp(-(self.cutoff-t)/(30*DAY))) for i,(n,t) in heard.items()),key=lambda p:(-p[1],p[0]))
        if cfg.use_als:
            src=[]
            profile=[self.cf_index[int(i)] for i,t in seeds if int(i) in self.cf_index]
            if self.als is not None and profile:
                x=sparse.csr_matrix((np.ones(len(profile),np.float32),([0]*len(profile),profile)),shape=(1,len(self.cf_ids)))
                factor=self.als.recalculate_user(0,x)
                scores=self.als.item_factors@factor
                take=min(len(scores),cfg.per_source+len(seen))
                idx=np.argpartition(scores,-take)[-take:]
                src=sorted(((int(self.cf_ids[j]),float(scores[j])) for j in idx),key=lambda p:(-p[1],p[0]))
            sources['als']=src
        ap=None
        if self.audio is not None:
            ids,vec,index=self.audio
            si=[index[i] for i,t in seeds if i in index]
            if si:
                ap=np.asarray(vec[si]).mean(axis=0);ap/=max(np.linalg.norm(ap),1e-12)
                scores=np.asarray(vec@ap);take=min(len(ids),cfg.per_source+len(seen))
                idx=np.argpartition(scores,-take)[-take:]
                sources['audio']=sorted(((int(ids[j]),float(scores[j])) for j in idx),key=lambda p:(-p[1],p[0]))
            else:sources['audio']=[]
        rows={}
        for source,ranked in sources.items():
            count=0
            for i,s in ranked:
                i=int(i)
                if i in seen:continue
                count+=1
                r=rows.setdefault(i,{'uid':int(u),'item_id':i,'rrf':0.,'n_sources':0.})
                r[f'{source}_score']=float(s);r[f'{source}_rr']=1/count
                r['rrf']+=1/(60+count);r['n_sources']+=1
                if count>=cfg.per_source:break
        # The cap uses ONLY past-derived scores, never target membership.
        chosen=sorted(rows.values(),key=lambda r:(-r['rrf'],r['item_id']))[:cfg.max_candidates]
        for r in chosen:
            i=r['item_id'];s=self.stats.get(i,{})
            for source in self.sources:
                r.setdefault(f'{source}_score',0.);r.setdefault(f'{source}_rr',0.)
            r.update(pop180=float(s.get('pop180',0)),pop30=float(s.get('pop30',0)),pop7=float(s.get('pop7',0)),trend_ratio=float(s.get('trend',0)),
                     item_age_days=(self.cutoff-s['first_ts'])/DAY if s else -1.,item_recency_days=(self.cutoff-s['last_ts'])/DAY if s else -1.,
                     user_likes=float(h.height) if h is not None else 0.,user_recency_days=(self.cutoff-h['timestamp'].max())/DAY if h is not None else -1.,
                     artist_affinity=float(sum(aff[a] for a in self.artists[i])),listen_count=float(heard.get(i,(0,0))[0]),
                     listen_recency_days=(self.cutoff-heard[i][1])/DAY if i in heard else -1.,audio_cos=-2.)
            if ap is not None and i in self.audio[2]: r['audio_cos']=float(self.audio[1][self.audio[2][i]]@ap)
        return chosen

    def batches(self,users,batch_size=128):
        for start in range(0,len(users),batch_size):
            rows=[r for u in users[start:start+batch_size] for r in self.candidates(u)]
            if not rows:raise ValueError('No candidates; check catalog/history')
            yield pl.DataFrame(rows).select('uid','item_id',*self.features).with_columns(pl.col(self.features).cast(pl.Float32))

    def generate(self,users):
        return pl.concat(list(self.batches(users)),how='vertical_relaxed')
