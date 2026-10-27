import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import train_candidate_ranker as ranker
import explore_trends as core


class RankerTests(unittest.TestCase):
    def fixture(self,root):
        data=root/'data';data.mkdir();rng=np.random.default_rng(11);rows=[]
        for u in range(20):
            for j in range(180):rows.append((u,int((u%4)*80+rng.integers(0,80)),j*2*core.DAY+u*5))
        pl.DataFrame(rows,schema=['uid','item_id','timestamp'],orient='row').with_columns(pl.all().cast(pl.UInt32)).write_parquet(data/'likes.parquet')
        pl.DataFrame({'item_id':range(320),'artist_id':[i//20 for i in range(320)]}).write_parquet(data/'artist_item_mapping.parquet')
        users=root/'users.csv';pl.DataFrame({'uid':list(range(12))+[999]}).write_csv(users)
        return data,users

    def test_future_invariance_and_label_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,users=self.fixture(root);cut=150*core.DAY
            ctx=core.load_context(data,users);before=list(ranker.source_features(ctx,cut,[0,999],150,None,200,2))
            old=pl.read_parquet(data/'likes.parquet')
            extra=pl.DataFrame({'uid':[0]*30,'item_id':[9999]*30,'timestamp':[cut+1]*30}).cast(old.schema)
            pl.concat([old,extra]).write_parquet(data/'likes.parquet')
            after=list(ranker.source_features(core.load_context(data,users),cut,[0,999],150,None,200,2))
            for a,b in zip(before,after):
                self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
            meta=ranker.build_pairs(ctx,cut,cut+10*core.DAY,[0,999],root/'pairs.parquet',150,None,200,2)
            expected=set(ctx.likes.filter((pl.col('uid')==0)&(pl.col('timestamp')>=cut)&(pl.col('timestamp')<cut+10*core.DAY))['item_id'].to_list())
            self.assertEqual(set(meta['truth'][0]),expected)
            table=pl.read_parquet(root/'pairs.parquet')
            self.assertEqual(table.select(ranker.FEATURES).width,len(ranker.FEATURES))
            self.assertTrue(all(i in expected for i in table.filter((pl.col('uid')==0)&(pl.col('target')==1))['item_id']))

    def test_full_training_selection_and_batched_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,users=self.fixture(root)
            args=['train_candidate_ranker.py','--data',str(data),'--users',str(users),'--out',str(root/'run'),
                  '--max-users','0','--train-users','8','--train-future-days','240','--val-future-days','180',
                  '--eval-future-days','120','60','--candidate-budgets','100','150','--iterations','3','5',
                  '--artist-catalog','200','--min-support','2','--threads','2','--cache-dir',str(root/'cache')]
            with patch.object(sys,'argv',args):ranker.main()
            report=json.loads((root/'run/summary.json').read_text())
            self.assertEqual(len(report['development']),8)
            self.assertEqual(report['selected_on_validation']['n_users'],13)
            self.assertTrue((root/'run/selected_model.cbm').exists())
            self.assertTrue((root/'run/eda_for_review.zip').exists())
            for row in report['development']:self.assertTrue(0<=row['recall_active']<=1)
            with patch.object(sys,'argv',args),self.assertRaises(SystemExit):ranker.main()

    def test_inference_merges_a_user_split_across_arrow_batches(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'split.parquet';n0=65500;n1=1000
            pl.DataFrame({'uid':[0]*n0+[1]*n1,'item_id':list(range(n0))+list(range(n1)),
                          'candidate_rank':list(range(1,n0+1))+list(range(1,n1+1))}).write_parquet(path)
            predictions=ranker.predict_file(None,path,100000,0,baseline=True)
            self.assertEqual(predictions[1],list(range(100)))

    def test_metric_uses_distinct_items_and_keeps_zero_users(self):
        result=ranker.metric({1:[7,8],2:[3]},{1:[7,7,9]},[1,2],{1})
        self.assertEqual(result['recall_all'],.25);self.assertEqual(result['recall_active'],.5)


if __name__=='__main__':unittest.main()
