from pathlib import Path
import json
import polars as pl

DAY = 86400


def events(path, cutoff=None, users=None):
    """All time filters are applied BEFORE collecting or computing statistics."""
    q = pl.scan_parquet(path)
    if cutoff is not None:
        q = q.filter(pl.col("timestamp") < cutoff)
    if users is not None:
        q = q.filter(pl.col("uid").is_in(list(users)))
    return q.collect(engine="streaming")


def first_likes(df):
    # Re-likes are deliberately treated as the same target; change only if rules clarify otherwise.
    return df.group_by("uid", "item_id").agg(pl.col("timestamp").min()).sort(
        ["uid", "timestamp", "item_id"])


def targets(path, cutoff, users, horizon_days=None, mode="next100"):
    q = pl.scan_parquet(path).filter(pl.col("uid").is_in(list(users)))
    # Deduplicate over complete history first, so old likes cannot become future positives.
    q = q.group_by("uid", "item_id").agg(pl.col("timestamp").min())
    q = q.filter(pl.col("timestamp") >= cutoff)
    if horizon_days is not None:
        q = q.filter(pl.col("timestamp") < cutoff + horizon_days * DAY)
    df = q.sort(["uid", "timestamp", "item_id"]).collect(engine="streaming")
    if mode == "next100":
        df = df.group_by("uid", maintain_order=True).head(100)
    return df


def load_users(path):
    df = pl.read_csv(path)
    if df.columns != ["uid"] or df["uid"].null_count() or df["uid"].n_unique() != df.height:
        raise ValueError("Expected one non-null unique uid column")
    return df["uid"].cast(pl.Int64).to_list()


def eda(data, users):
    path = Path(data) / "likes.parquet"
    q = pl.scan_parquet(path)
    overall = q.select(pl.len().alias("events"), pl.col("uid").n_unique().alias("users"),
                       pl.col("item_id").n_unique().alias("items"),
                       pl.col("timestamp").min().alias("min_ts"),
                       pl.col("timestamp").max().alias("max_ts")).collect().to_dicts()[0]
    counts = q.group_by("uid").len().collect()
    present = set(counts["uid"].to_list())
    overall.update(test_users=len(users), test_with_likes=len(set(users) & present),
                   test_without_likes=len(set(users) - present),
                   span_days=(overall["max_ts"]-overall["min_ts"])/DAY,
                   likes_per_user_quantiles={str(v): float(counts["len"].quantile(v)) for v in [.1,.5,.9,.99]})
    return overall


def download(data, size, names, revision):
    from huggingface_hub import hf_hub_download, HfApi
    import shutil
    root = Path(data); root.mkdir(parents=True, exist_ok=True)
    # Pin a concrete dataset revision, including for later optional downloads.
    manifest = root / "dataset_manifest.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else None
    if old and old["size"] != size:
        raise ValueError("Use a separate directory for each dataset size")
    sha = old["revision"] if old else HfApi().dataset_info("yandex/yambda", revision=revision).sha
    records = old or {"repo": "yandex/yambda", "revision": sha, "size": size, "files": {}}
    for name in names:
        remote = f"{name}.parquet" if name in {"artist_item_mapping", "album_item_mapping", "embeddings"} else f"flat/{size}/{name}.parquet"
        local = hf_hub_download("yandex/yambda", remote, repo_type="dataset", revision=sha)
        dest = root / f"{name}.parquet"
        if not dest.exists():
            # Symlink to HF cache avoids another 13.8GB copy of embeddings. Copy on Windows fallback.
            try: dest.symlink_to(Path(local).resolve())
            except OSError: shutil.copy2(local, dest)
        records["files"][name] = remote
        manifest.write_text(json.dumps(records, indent=2))
        print(dest, flush=True)
