"""Stream the large audio file, retain only tracks known at a chosen historical cutoff."""
from pathlib import Path
import json
import numpy as np
import polars as pl
import pyarrow.parquet as pq


def extract(data,out,cutoff,max_items=200000):
    data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    counts=pl.scan_parquet(data/'likes.parquet').filter(pl.col('timestamp')<cutoff).group_by('item_id').len()
    ids=counts.sort(['len','item_id'],descending=[True,False]).head(max_items).collect()['item_id'].to_list()
    wanted=set(ids);found={}
    p=pq.ParquetFile(data/'embeddings.parquet')
    column='normalized_embed' if 'normalized_embed' in p.schema_arrow.names else 'embed'
    for batch in p.iter_batches(batch_size=8192,columns=['item_id',column]):
        frame=pl.from_arrow(batch).filter(pl.col('item_id').is_in(wanted))
        for i,v in frame.iter_rows():
            if v is None:continue
            a=np.asarray(v,dtype=np.float32)
            if not np.isfinite(a).all():continue
            norm=np.linalg.norm(a)
            if norm>0:found[int(i)]=a/norm
        print(f'audio retained={len(found)}/{len(wanted)}',flush=True)
    if not found:raise ValueError('No audio vectors matched known items')
    keys=sorted(found)
    np.save(out/'ids.npy',np.array(keys,dtype=np.int64))
    np.save(out/'vectors.npy',np.stack([found[i] for i in keys]).astype(np.float32))
    (out/'metadata.json').write_text(json.dumps({'cutoff':cutoff,'selected_items':len(wanted),'retained_items':len(keys),'column':column},indent=2))
