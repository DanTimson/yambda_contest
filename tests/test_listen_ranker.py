import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import train_listen_ranker as app


class ListenRankerTests(unittest.TestCase):
    def fixture(self, root):
        data = root/'data'; data.mkdir(); likes = []; listens = []
        for u in range(8):
            for j in range(460):
                item = (j*7+u*13) % 360
                likes.append((u, item, j*app.core.DAY+u*5))
                listens.append((u, item, j*app.core.DAY-5, j%2, [10,50,80,100,150][j%5], 180))
        pl.DataFrame(likes, schema=['uid','item_id','timestamp'], orient='row').write_parquet(data/'likes.parquet')
        pl.DataFrame(listens, schema=app.listening.COLS, orient='row').write_parquet(data/'listens.parquet')
        for name, ts in [('dislikes',100),('undislikes',110),('unlikes',120)]:
            pl.DataFrame({'uid':[0], 'item_id':[7], 'timestamp':[ts*app.core.DAY]}).write_parquet(data/(name+'.parquet'))
        pl.DataFrame({'item_id':range(360),'artist_id':[i//30 for i in range(360)]}).write_parquet(data/'artist_item_mapping.parquet')
        users = root/'users.csv'; pl.DataFrame({'uid':list(range(6))+[999]}).write_csv(users)
        ctx = app.core.load_context(data, users)
        cuts = [ctx.end-d*app.core.DAY for d in [90,60,30]]
        events, meta = app.listening.extract_cache(data/'listens.parquet', ctx.users, cuts[0]-90*app.core.DAY,
                                                 cuts[-1], root/'cache', 42, 1000, batch_size=100)
        eda = root/'eda'; eda.mkdir(); (eda/'summary.json').write_text(json.dumps({'cache':meta}))
        return data, users, ctx, cuts, events, eda

    def test_like_feature_compatibility_and_new_items(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data, users, ctx, cuts, events, eda=self.fixture(root)
            selected=[0,999]; cut=cuts[0]
            factory=app.LikeFeatures(ctx,cut,selected,root/'cache',200)
            original=list(app.old.source_features(ctx,cut,selected,100,root/'cache',200,10))
            for u,ids,x in original:
                actual=factory.features(u,ids,factory.user_sources(u))
                np.testing.assert_allclose(actual,x,rtol=1e-6,atol=1e-6)
            unknown=factory.features(0,np.array([999999]),factory.user_sources(0))
            self.assertTrue(np.isfinite(unknown).all())
            self.assertEqual(unknown[0,app.old.FEATURES.index('first_seen_age_days')],-1.)

    def test_listen_and_like_features_ignore_future(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data, users, ctx, cuts, events, eda=self.fixture(root); cut=cuts[0]
            before=list(app.source_rows(ctx,events,cut,[0,999],200,root/'cache',200,90))
            f=pl.read_parquet(events)
            extra=f.head(1).with_columns(pl.lit(cut,dtype=pl.Int64).alias('timestamp'),pl.lit(999999,dtype=pl.Int64).alias('item_id'))
            pl.concat([f,extra]).write_parquet(events)
            new_like=ctx.likes.head(1).with_columns(pl.lit(cut,dtype=pl.Int64).alias('timestamp'),pl.lit(999999,dtype=pl.Int64).alias('item_id'))
            changed=replace(ctx,likes=pl.concat([ctx.likes,new_like]))
            after=list(app.source_rows(changed,events,cut,[0,999],200,root/'cache',200,90))
            for a,b in zip(before,after):
                np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_allclose(a[2],b[2])

    def test_full_ablation_run_and_cache_coverage_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data, users, ctx, cuts, events, eda=self.fixture(root)
            with patch.object(app.listening,'extract_cache',side_effect=AssertionError('no raw listen scan')):
                resolved,_=app.resolve_cache(eda,root/'cache',ctx,cuts,90)
                self.assertEqual(resolved,events)
                with self.assertRaises(ValueError):app.resolve_cache(eda,root/'cache',ctx,[cuts[0]-app.core.DAY,cuts[-1]],90)
            args=['train_listen_ranker.py','--data',str(data),'--users',str(users),'--listen-eda-run',str(eda),
                  '--cache-dir',str(root/'cache'),'--out',str(root/'run'),'--max-users','0','--train-users','5',
                  '--budget','200','--iterations','2','3','--threads','2','--artist-catalog','200']
            with patch.object(sys,'argv',args):app.main()
            s=json.loads((root/'run/summary.json').read_text())
            self.assertEqual(len(s['metrics']),10);self.assertEqual(len(s['validation']),6)
            self.assertTrue((root/'run/selected_model.cbm').exists());self.assertTrue((root/'run/eda_for_review.zip').exists())
            self.assertEqual(len(s['features']['listen_features']),len(app.FEATURES))
            self.assertEqual(s['snapshots']['train']['label_end_exclusive'],cuts[1])
            for r in s['metrics']:self.assertTrue(0<=r['recall_active']<=1)
            with patch.object(sys,'argv',args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):app.main()

    def test_predictions_merge_users_across_arrow_batches(self):
        class Model:
            def predict(self,x,ntree_end):return x[:,0]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pairs.parquet';n=65500
            pl.DataFrame({'uid':[0]*n+[1]*1000,'item_id':list(range(n))+list(range(1000)),
                          'f':list(range(n))+list(range(1000))}).write_parquet(path)
            result=app.predict(Model(),path,['f'],1)
            self.assertEqual(result[1],list(range(999,899,-1)))


if __name__=='__main__':unittest.main()
