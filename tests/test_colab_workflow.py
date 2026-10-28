import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import test_listen_ranker as common
import train_listen_ranker as app
import collaborative_sources as cf
import listen_cache_portable as portable


class ColabWorkflowTests(unittest.TestCase):
    def fixture(self,root):
        values=common.ListenRankerTests().fixture(root)
        data,users,ctx,cuts,events,eda=values
        pl.DataFrame({'item_id':range(360),'album_id':[i//12 for i in range(360)]}).write_parquet(data/'album_item_mapping.parquet')
        s=json.loads((eda/'summary.json').read_text());s['user_file_sha256']=portable.sha256(users)
        s['source_details']=[{'observed_targets_end':ctx.end}];(eda/'summary.json').write_text(json.dumps(s))
        return values

    def test_portable_checks_coverage_hashes_and_no_source_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,users,ctx,cuts,events,eda=self.fixture(root);bundle=root/'cache.zip'
            portable.export_bundle(eda,root/'cache',data,users,bundle,include_all_history=True,all_test_users=True)
            (data/'listens.parquet').unlink()
            path,meta=portable.load_bundle(bundle,root/'import',data,users,ctx,cuts,90)
            self.assertTrue(path.exists());self.assertEqual(meta['signature']['hi'],ctx.end)
            args=['explore_listens.py','--data',str(data),'--users',str(users),'--out',str(root/'portable_eda'),
                  '--listen-cache-bundle',str(bundle),'--cache-dir',str(root/'import'),'--max-users','0',
                  '--future-days','90','30','--budget','100','--population-permille','1000','--baseline-source-order','ranker']
            with patch.object(sys,'argv',args):app.listening.main()
            import benchmark_collaborative as bench
            args=['benchmark_collaborative.py','--data',str(data),'--users',str(users),'--listen-cache-bundle',str(bundle),
                  '--cache-dir',str(root/'import'),'--out',str(root/'long'),'--max-users','0','--eval-users','3',
                  '--future-days','150','120','--budgets','100','200','--cf-mod','1','--als-factors','8',
                  '--als-iterations','2','--knn-items','120','--knn-neighbors','10','--retrieval-device','cpu']
            with patch.object(sys,'argv',args):bench.main()
            self.assertTrue((root/'long/summary.json').exists())
            with self.assertRaises(ValueError):portable.load_bundle(bundle,root/'import',data,users,ctx,[ctx.end+1],90)
            old=(data/'likes.parquet').read_bytes();(data/'likes.parquet').write_bytes(old+b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):portable.load_bundle(bundle,root/'import',data,users,ctx,cuts,90)

    def test_collaborative_time_safety_and_scalar_features(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,users,ctx,cuts,events,eda=self.fixture(root);cut=cuts[0]
            h=ctx.likes.filter(pl.col('timestamp')<cut);pairs=app.listening.pair_stats(events,cut,90)
            grouped={int(u):g for (u,),g in pairs.group_by('uid')}
            cfg=cf.Config(user_mod=1,factors=8,iterations=2,knn_items=120,knn_neighbors=10,source_budget=30,threads=2,retrieval_device='cpu')
            a=cf.Sources(data,cut,root/'cf',cfg,ctx.users,h,grouped)
            before=a.lists(0);x=a.features(0,np.array([7,999999]))
            self.assertEqual(x.shape,(2,len(cf.FEATURES)));self.assertTrue(np.isfinite(x).all())
            old=pl.read_parquet(data/'likes.parquet')
            extra=old.head(1).with_columns(pl.lit(cut,dtype=pl.Int64).alias('timestamp'),pl.lit(999999,dtype=pl.Int64).alias('item_id'))
            pl.concat([old,extra]).write_parquet(data/'likes.parquet')
            b=cf.Sources(data,cut,root/'cf',cfg,ctx.users,h,grouped)
            self.assertNotIn(999999,b.items);self.assertEqual(before,b.lists(0))
            np.testing.assert_allclose(x,b.features(0,np.array([7,999999])),atol=1e-5)

    def test_sampled_classifier_collaborative_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,users,ctx,cuts,events,eda=self.fixture(root)
            args=['train_listen_ranker.py','--data',str(data),'--users',str(users),'--listen-eda-run',str(eda),
                  '--cache-dir',str(root/'cache'),'--out',str(root/'run'),'--max-users','0','--train-users','5',
                  '--budget','250','--iterations','2','3','--threads','2','--artist-catalog','200',
                  '--collaborative','--objective','Logloss','--negatives','20','--cf-mod','1',
                  '--als-factors','8','--als-iterations','2','--knn-items','120','--knn-neighbors','10','--retrieval-device','cpu']
            with patch.object(sys,'argv',args):app.main()
            s=json.loads((root/'run/summary.json').read_text())
            self.assertIn('collaborative_features',s['features']);self.assertTrue((root/'run/selected_model.cbm').exists())
            train=pl.read_parquet(root/'run/pairs/train.parquet')
            self.assertTrue((train.group_by('uid').agg(pl.col('sample_weight').sum())['sample_weight']-1).abs().max()<1e-5)
            self.assertLessEqual(train.filter(pl.col('target')==0).group_by('uid').len()['len'].max(),20)

    def test_sampling_keeps_positives_and_corrects_mass(self):
        y=np.zeros(100,np.int8);y[[5,99]]=1
        ix,w=app.sample_indices(y,10,42,123,7)
        self.assertTrue({5,99}.issubset(set(ix)));self.assertEqual(len(ix),12)
        self.assertAlmostEqual(float(w.sum()),1.,places=6)
        self.assertAlmostEqual(float(w[y[ix]==1].sum()),2/100,places=6)


if __name__=='__main__':unittest.main()
