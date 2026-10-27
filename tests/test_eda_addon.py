"""Run independently with: python -m unittest discover -s tests -p test_eda_addon.py"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
import numpy as np
import polars as pl

SCRIPT=Path(__file__).resolve().parents[1]/'tools'/'explore_yambda.py'
spec=importlib.util.spec_from_file_location('eda',SCRIPT)
eda=importlib.util.module_from_spec(spec);spec.loader.exec_module(eda)


class EdaTests(unittest.TestCase):
    def test_targets_repeat_and_boundary(self):
        h=pl.DataFrame({'uid':[1,1], 'item_id':[10,11], 'timestamp':[1,2]})
        f=pl.DataFrame({'uid':[1,1,1,1], 'item_id':[10,10,12,13], 'timestamp':[5,6,7,8]})
        variants=eda.target_variants(h,f)
        self.assertEqual(variants['next100_events']['item_id'].to_list(),[10,10,12,13])
        self.assertEqual(variants['next100_unique']['item_id'].to_list(),[10,12,13])
        self.assertEqual(variants['next100_new_items']['item_id'].to_list(),[12,13])

    def test_all_optional_paths_and_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir()
            rows=[]
            rng=np.random.default_rng(42)
            for u in range(12):
                for j in range(180):rows.append((u,int(rng.integers(0,250)),j*eda.DAY*2+u*5,int(j%2)))
            pl.DataFrame(rows,schema=['uid','item_id','timestamp','is_organic'],orient='row').with_columns(pl.all().cast(pl.UInt32)).write_parquet(data/'likes.parquet')
            pl.DataFrame({'item_id':range(250),'artist_id':[i//10 for i in range(250)]}).write_parquet(data/'artist_item_mapping.parquet')
            pl.DataFrame({'uid':[0,0,0], 'item_id':[1,2,3], 'timestamp':[150*eda.DAY,290*eda.DAY,310*eda.DAY], 'played_ratio_pct':[100,120,30]}).with_columns(pl.all().cast(pl.UInt32)).write_parquet(data/'listens.parquet')
            pl.DataFrame({'item_id':range(250),'normalized_embed':np.eye(250,8).tolist()}).write_parquet(data/'embeddings.parquet')
            users=root/'users.csv';pl.DataFrame({'uid':[0,1,2,3,999]}).write_csv(users)
            args=argparse.Namespace(data=str(data),users=str(users),out=str(root/'out'),cutoffs=[200*eda.DAY,300*eda.DAY],max_users=0,seed=42,with_listens=True,listen_users=5,with_embeddings=True)
            result=eda.run(args)
            self.assertEqual(result['population']['requested_users_with_likes'],4)
            self.assertEqual(result['selected_cohort']['users'],5)
            self.assertEqual(len(result['snapshots']),2)
            self.assertEqual(result['listens']['play_events'],1) # only day290 inside [210,300), no future/old plays
            self.assertIn('embeddings',result)
            self.assertTrue((root/'out/report.html').exists())
            parsed=json.loads((root/'out/summary.json').read_text())
            self.assertEqual(parsed['run']['cutoffs'],[200*eda.DAY,300*eda.DAY])
            with zipfile.ZipFile(root/'out/eda_for_review.zip') as z:
                self.assertIsNone(z.testzip());self.assertIn('report.html',z.namelist());self.assertIn('summary.json',z.namelist())
            with self.assertRaisesRegex(ValueError,'nonempty'):eda.run(args)

    def test_cohort_with_no_observed_likes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir()
            pl.DataFrame({'uid':[1,1,1], 'item_id':[10,11,12], 'timestamp':[1,100*eda.DAY,200*eda.DAY]}).write_parquet(data/'likes.parquet')
            users=root/'users.csv';pl.DataFrame({'uid':[999]}).write_csv(users)
            args=argparse.Namespace(data=str(data),users=str(users),out=str(root/'out'),cutoffs=[50*eda.DAY],max_users=0,seed=42,with_listens=False,listen_users=1,with_embeddings=False)
            result=eda.run(args)
            self.assertEqual(result['population']['requested_users_with_likes'],0)
            self.assertEqual(result['snapshots'][0]['targets'][0]['active_future_users'],0)
            self.assertIsNone(result['snapshots'][0]['popularity_diagnostics'][0]['recall_active'])

    def test_gitignore_allows_evidence_but_blocks_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'.gitignore').write_text((SCRIPT.parents[1]/'.gitignore').read_text())
            ignore=['data/5b/likes.parquet','data','input/submission_example.csv','.env','foo/checkpoint.cbm','foo/embeddings.npy','artifacts/output.csv','submissions/a.csv']
            keep=['src/module.py','tools/explore_yambda.py','uv.lock','input/test_users.csv','eda_results/5b_v1/summary.json','eda_results/5b_v1/report.html','eda_results/5b_v1/tables/all_cutoffs.csv','eda_results/5b_v1/plots/timeline.svg']
            for item in ignore:
                self.assertEqual(subprocess.run(['git','-C',str(root),'check-ignore','-q','--no-index',item]).returncode,0,item)
            for item in keep:
                self.assertEqual(subprocess.run(['git','-C',str(root),'check-ignore','-q','--no-index',item]).returncode,1,item)


if __name__=='__main__':unittest.main()
