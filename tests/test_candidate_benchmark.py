import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import explore_trends as core
import benchmark_candidates as bench


class CandidateTests(unittest.TestCase):
    def fixture(self,root):
        data=root/'data';data.mkdir();rng=np.random.default_rng(7);rows=[]
        for u in range(20):
            for j in range(180):rows.append((u,int((u%4)*80+rng.integers(0,80)),j*2*core.DAY+u*5))
        pl.DataFrame(rows,schema=['uid','item_id','timestamp'],orient='row').with_columns(pl.all().cast(pl.UInt32)).write_parquet(data/'likes.parquet')
        pl.DataFrame({'item_id':list(range(320))+[0], 'artist_id':[i//20 for i in range(320)]+[99]}).write_parquet(data/'artist_item_mapping.parquet')
        users=root/'users.csv';pl.DataFrame({'uid':[0,1,2,3,4,999]}).write_csv(users)
        return data,users

    def test_future_invariance_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,users=self.fixture(root);cut=200*core.DAY
            ctx=core.load_context(data,users);h,t=core.history_targets(ctx,cut)
            stats,sig=bench.cached_stats(data,cut,root/'cache')
            before=bench.global_lists(stats,200,2)
            artist=bench.ArtistSource(data,h,stats,cut,catalog_size=200)
            personal=artist.recommend(0,200)
            cached,_=bench.cached_stats(data,cut,root/'cache')
            self.assertTrue(stats.equals(cached))
            p=data/'likes.parquet';old=pl.read_parquet(p)
            extra=pl.DataFrame({'uid':[0]*200,'item_id':[8888]*200,'timestamp':[(cut+core.DAY)]*200}).cast(old.schema)
            pl.concat([old,extra]).write_parquet(p)
            changed,sig2=bench.cached_stats(data,cut,root/'cache')
            self.assertNotEqual(sig,sig2)
            self.assertEqual(before,bench.global_lists(changed,200,2))
            self.assertEqual(personal,bench.ArtistSource(data,h,changed,cut,catalog_size=200).recommend(0,200))
            self.assertTrue(artist.info['available']);self.assertEqual(artist.recommend(999,200)['artist_pop'],[])

    def test_batch_metrics_and_listen_complementarity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,users=self.fixture(root);ctx=core.load_context(data,users)
            result=bench.benchmark(ctx,[160*core.DAY,240*core.DAY],root/'run',budget=200,min_support=2,artist_catalog=200,cache_dir=root/'cache')
            self.assertEqual(len(result['snapshots']),2)
            for snap in result['snapshots']:
                self.assertEqual(len(snap['metrics']),13)
                for row in snap['metrics']:
                    if row['source']!='union_oracle':
                        self.assertLessEqual(row['max_pool_size'],200)
                        if row['recall100_active'] is not None:self.assertLessEqual(row['recall100_active'],row['candidate_recall_active'])
                self.assertTrue(snap['artist_source']['available'])
            self.assertTrue((root/'run/report.html').exists());self.assertTrue((root/'run/eda_for_review.zip').exists())
            json.loads((root/'run/summary.json').read_text())
            with self.assertRaises(ValueError):bench.benchmark(ctx,[160*core.DAY],root/'run')
            pl.DataFrame({'uid':[0,0,1], 'item_id':[1,2,3], 'timestamp':[230*core.DAY,245*core.DAY,235*core.DAY], 'played_ratio_pct':[120,100,30]}).with_columns(pl.all().cast(pl.UInt32)).write_parquet(data/'listens.parquet')
            listens=core.run_listens(ctx,240*core.DAY,root/'listens',n_users=6,reference_items=[1,2,3])
            self.assertEqual(listens['metrics']['sample_users'],6)
            self.assertTrue((root/'listens/report.html').exists())
            rows=pl.read_csv(root/'listens/tables/listen_users.csv')
            self.assertEqual(rows.filter(pl.col('uid')==0)['qualifying_listened_candidates'][0],1)
            self.assertEqual(rows.filter(pl.col('uid')==1)['qualifying_listened_candidates'][0],0)

    def test_cli_archive_and_absent_optional_listens(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,users=self.fixture(root)
            args=['benchmark_candidates.py','--data',str(data),'--users',str(users),
                  '--out',str(root/'cli'),'--max-users','6','--budget','100',
                  '--artist-catalog','100','--min-support','2','--future-days','90',
                  '--cache-dir',str(root/'cache'),'--with-listens']
            with patch.object(sys,'argv',args):bench.main()
            with zipfile.ZipFile(root/'cli/all_experiments_for_review.zip') as z:
                self.assertIn('report.html',z.namelist());self.assertIn('summary.json',z.namelist())
                self.assertIsNone(z.testzip())

    def test_fixed_budget_fusion(self):
        a=list(range(1000));b=list(range(900,1900))
        fused=core.fuse([a,b],1000)
        self.assertEqual(len(fused),1000);self.assertEqual(len(set(fused)),1000)
        self.assertIn(900,fused)


if __name__=='__main__':unittest.main()
