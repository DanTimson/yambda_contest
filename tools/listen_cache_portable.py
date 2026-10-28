#!/usr/bin/env python3
"""Export/verify the existing reduced listen cache without copying the 5B source."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import pyarrow.parquet as pq

VERSION = 'yambda-portable-listens-v1'
INPUTS = ['likes.parquet', 'dislikes.parquet', 'unlikes.parquet', 'undislikes.parquet',
          'artist_item_mapping.parquet', 'album_item_mapping.parquet']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def export_bundle(eda_run, cache_dir, data, users, out, include_all_history=False, all_test_users=False):
    report = json.loads((Path(eda_run)/'summary.json').read_text())
    extraction = report['cache']; sig = extraction['signature']
    key = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:24]
    folder = Path(cache_dir)/('listens_'+key)
    if json.loads((folder/'manifest.json').read_text())['signature'] != sig:
        raise ValueError('Native cache signature mismatch')
    if include_all_history or all_test_users:
        from explore_listens import extract_cache
        source = Path(data)/'listens.parquet'
        st = source.stat()
        if (str(source.resolve()), st.st_size, st.st_mtime_ns) != (sig['source'], sig['bytes'], sig['mtime_ns']):
            raise ValueError('Original listens source changed')
        end = report['source_details'][0]['observed_targets_end'] if include_all_history else sig['hi']
        lo = 0 if include_all_history else sig['lo']
        selected = sig['users']
        if all_test_users:
            import polars as pl
            selected = sorted(pl.read_csv(users)['uid'].cast(pl.Int64).to_list())
        events, extraction = extract_cache(source, selected, lo, end, cache_dir, sig['seed'], sig['population_permille'])
        sig = extraction['signature']
    else:
        events = folder/'events.parquet'
    if pq.ParquetFile(events).metadata.num_rows != extraction['retained_rows']:
        raise ValueError('Native cache row count mismatch')
    if sha256(users) != report['user_file_sha256']:
        raise ValueError('User CSV differs from the listening EDA input')
    print('Hashing reduced cache and the six small model inputs; not the 36.7 GB source.', flush=True)
    hashes = {name: sha256(Path(data)/name) for name in INPUTS}
    manifest = dict(version=VERSION, extraction=extraction, input_sha256=hashes,
                    users_sha256=sha256(users), user_csv_base64=base64.b64encode(Path(users).read_bytes()).decode(), events_sha256=sha256(events), events_bytes=events.stat().st_size,
                    source_revision=(json.loads((Path(data)/'dataset_manifest.json').read_text()).get('revision')
                                     if (Path(data)/'dataset_manifest.json').exists() else None),
                    scope='Filtered 5B-derived cohort and UID-hash population sample; not all-population listens')
    out = Path(out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix+'.partial')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED) as z:
        z.write(events, 'events.parquet')
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
    partial.replace(out)
    print(f'PORTABLE CACHE: {out.resolve()} ({out.stat().st_size/1024**2:.1f} MiB)', flush=True)
    return manifest


def load_bundle(bundle, cache_dir, data, users, ctx, cuts, lookback, population_permille=None):
    """Retain original source provenance; validate portable bytes instead of local paths."""
    with zipfile.ZipFile(bundle) as z:
        if sorted(z.namelist()) != ['events.parquet', 'manifest.json']:
            raise ValueError('Bundle must contain exactly events.parquet and manifest.json')
        if z.getinfo('manifest.json').file_size > 2*1024**2: raise ValueError('Oversized manifest')
        m = json.loads(z.read('manifest.json'))
        if m.get('version') != VERSION: raise ValueError('Unsupported portable-cache version')
        sig = m['extraction']['signature']
        if sig['version'] != 'listens-eda-v1': raise ValueError('Unsupported extraction version')
        if not set(ctx.users).issubset(sig['users']): raise ValueError('Portable cache does not cover selected UIDs')
        if max(ctx.start,min(cuts)-lookback*86400) < sig['lo'] or max(cuts) > sig['hi']:
            raise ValueError('Requested time windows exceed portable cache coverage')
        if population_permille is not None and population_permille != sig['population_permille']:
            raise ValueError('Population sample fraction differs from cached extraction')
        if ctx.seed != sig['seed']: raise ValueError('Seed differs from cached extraction')
        if sha256(users) != m['users_sha256']: raise ValueError('User CSV content differs from exported input')
        for name, expected in m['input_sha256'].items():
            if name not in INPUTS: raise ValueError('Unexpected model input in manifest')
            if sha256(Path(data)/name) != expected: raise ValueError(f'Input hash mismatch: {name}')
        if set(m['input_sha256']) != set(INPUTS): raise ValueError('Incomplete model input fingerprints')
        if z.getinfo('events.parquet').file_size != m['events_bytes']: raise ValueError('Cache size mismatch')
        dest = Path(cache_dir)/('portable_'+m['events_sha256'][:24]); dest.mkdir(parents=True, exist_ok=True)
        path = dest/'events.parquet'
        if not path.exists() or sha256(path) != m['events_sha256']:
            temp = dest/'events.parquet.partial'
            with z.open('events.parquet') as src, temp.open('wb') as target: shutil.copyfileobj(src, target, 8*1024**2)
            if sha256(temp) != m['events_sha256']: raise ValueError('Portable event checksum mismatch')
            temp.replace(path)
    pf = pq.ParquetFile(path)
    required = {'uid','item_id','timestamp','is_organic','played_ratio_pct','track_length_seconds','cohort','trend_sample'}
    if not required.issubset(pf.schema_arrow.names) or pf.metadata.num_rows != m['extraction']['retained_rows']:
        raise ValueError('Portable event schema/row count mismatch')
    (dest/'portable_manifest.json').write_text(json.dumps(m, indent=2))
    verified = dict(sha256=m['input_sha256'], stats={name:[str((Path(data)/name).resolve()),(Path(data)/name).stat().st_size,(Path(data)/name).stat().st_mtime_ns] for name in INPUTS})
    (Path(data)/'portable_inputs_verified.json').write_text(json.dumps(verified))
    info = dict(m['extraction'])
    info['portable'] = {k: v for k, v in m.items() if k != 'extraction'}
    print(f'VERIFIED PORTABLE CACHE: {path}; {pf.metadata.num_rows:,} events', flush=True)
    return path, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eda-run', required=True); p.add_argument('--cache-dir', default='data/eda_cache')
    p.add_argument('--data', default='data/5b'); p.add_argument('--users', default='input/test_users.csv')
    p.add_argument('--out', required=True); p.add_argument('--include-all-history', action='store_true'); p.add_argument('--all-test-users', action='store_true'); a = p.parse_args()
    export_bundle(a.eda_run, a.cache_dir, a.data, a.users, a.out, a.include_all_history, a.all_test_users)


if __name__ == '__main__': main()
