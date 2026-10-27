import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import explore_listens as mod


class ListenTests(unittest.TestCase):
    def fixture(self, root):
        data = root/'data'; data.mkdir(); day = mod.DAY
        likes = []; listens = []
        for u in range(1, 6):
            for j in range(180):
                item = (j+u) % 16
                likes.append((u, item, j*2*day+5))
                listens.append((u, item, j*2*day, j % 2, [10, 50, 80, 100, 150][j % 5], 180))
                listens.append((u, item, j*2*day-10, 1, 100, 180))
        cols = ['uid','item_id','timestamp']
        pl.DataFrame(likes, schema=cols, orient='row').write_parquet(data/'likes.parquet')
        pl.DataFrame(listens, schema=mod.COLS, orient='row').write_parquet(data/'listens.parquet', row_group_size=13)
        for name in ['dislikes','unlikes','undislikes']:
            rows = [(1, 2, 100*day)] if name != 'undislikes' else [(1, 2, 101*day)]
            pl.DataFrame(rows, schema=cols, orient='row').write_parquet(data/(name+'.parquet'))
        pl.DataFrame({'uid': [1,2,3,999]}).write_csv(root/'users.csv')
        return data, root/'users.csv'

    def test_scan_boundaries_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data, _ = self.fixture(root)
            path, info = mod.extract_cache(data/'listens.parquet', [1], 0, 100*mod.DAY, root/'cache', 42, 1, batch_size=7)
            f = pl.read_parquet(path).filter(pl.col('cohort'))
            self.assertTrue((f['timestamp'] >= 0).all()); self.assertTrue((f['timestamp'] < 100*mod.DAY).all())
            self.assertGreater(f.filter(pl.col('played_ratio_pct') > 100).height, 0)
            self.assertGreater(f.filter(pl.col('played_ratio_pct') <= 20).height, 0)
            with patch.object(mod.pq, 'ParquetFile', side_effect=AssertionError('must reuse cache')):
                same, again = mod.extract_cache(data/'listens.parquet', [1], 0, 100*mod.DAY, root/'cache', 42, 1)
            self.assertEqual(path, same); self.assertEqual(info, again)

    def test_past_features_ignore_future_and_cancelled_feedback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data, _ = self.fixture(root); cut = 120*mod.DAY
            original = pl.read_parquet(data/'listens.parquet').with_columns(pl.lit(True).alias('cohort'), pl.lit(True).alias('trend_sample'))
            cache = root/'events.parquet'; original.write_parquet(cache)
            before = mod.pair_stats(cache, cut, 90).sort(mod.KEYS)
            _, trends_before = mod.trend_lists(cache, cut, 100, 1)
            x = original.head(1).with_columns(pl.lit(cut).alias('timestamp'), pl.lit(999, dtype=pl.Int64).alias('item_id')).cast(original.schema)
            pl.concat([original, x]).write_parquet(cache)
            after = mod.pair_stats(cache, cut, 90).sort(mod.KEYS)
            self.assertTrue(before.equals(after))
            _, trends_after = mod.trend_lists(cache, cut, 100, 1)
            self.assertEqual(trends_before, trends_after)
            self.assertEqual(mod.listen_lists(before, cut, 100), mod.listen_lists(after, cut, 100))
            h = pl.read_parquet(data/'likes.parquet').filter(pl.col('timestamp') < cut)
            f = mod.feedback_pairs(data, [1], cut, h.filter(pl.col('uid') == 1))
            self.assertFalse(f.filter(pl.col('item_id') == 2)['active_dislike'].item())

    def test_end_to_end_empty_users_and_no_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data, users = self.fixture(root)
            args = ['explore_listens.py', '--data', str(data), '--users', str(users), '--out', str(root/'out'),
                    '--cache-dir', str(root/'cache'), '--max-users','0', '--future-days','90','30',
                    '--budget','100', '--population-permille','1000', '--min-listeners','1','--artist-catalog','50']
            with patch.object(sys, 'argv', args): mod.main()
            report = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(len(report['metrics']), 28)
            self.assertTrue((root/'out/report.html').exists()); self.assertTrue((root/'out/eda_for_review.zip').exists())
            for row in report['metrics']:
                self.assertLessEqual(row['coverage_active'], 1)
                self.assertEqual(row['users'], 4)
            with patch.object(sys, 'argv', args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): mod.main()

    def test_uid_sample_is_batch_independent(self):
        ids = np.arange(10000)
        all_at_once = mod.population_mask(ids, 42, 10)
        np.testing.assert_array_equal(all_at_once, np.concatenate([mod.population_mask(x, 42, 10) for x in np.array_split(ids, 17)]))
        self.assertTrue(50 < int(all_at_once.sum()) < 160)


if __name__ == '__main__': unittest.main()
