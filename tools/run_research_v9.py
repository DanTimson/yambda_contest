#!/usr/bin/env python3
"""Local 50M/500M candidate and ranking study. Full runs execute only when invoked."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--size", choices=["50m", "500m"], default="50m")
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--out", type=Path)
parser.add_argument("--data", type=Path)
parser.add_argument("--cache-dir", type=Path)
parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
parser.add_argument("--torch-device", choices=["auto", "cpu", "gpu"], default="auto")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--cohort-users", type=int)
parser.add_argument("--train-users", type=int)
parser.add_argument("--older-train-users", type=int)
parser.add_argument("--eval-users", type=int)
parser.add_argument("--trees", type=int, nargs="+")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--skip-neural", action="store_true")
parser.add_argument(
    "--doctor",
    action="store_true",
    help="Verify imports/device only; no data download or training",
)
args = parser.parse_args()


# config
# Параметры эксперимента
DATA_SIZE = args.size  # @param ["50m", "500m"]
RUN_LABEL = "yambda_research_v9_" + DATA_SIZE
TASK_TYPE = args.task_type  # @param ["GPU", "CPU"]
MAX_USERS = 0  # @param {type:"integer"}
TRAIN_USERS = 1500 if DATA_SIZE == "50m" else 5000  # @param {type:"integer"}
EVAL_USERS = 1500  # @param {type:"integer"}
RUN_LISTEN_CONTROL = False  # @param {type:"boolean"}
RUN_COLLABORATIVE_RANKER = False  # @param {type:"boolean"}
RESUME = True  # @param {type:"boolean"}
CF_MOD = 1  # @param {type:"integer"}
ALS_FACTORS = 64  # @param {type:"integer"}
ALS_ITERATIONS = 20  # @param {type:"integer"}
KNN_ITEMS = 200000  # @param {type:"integer"}
KNN_NEIGHBORS = 80  # @param {type:"integer"}
THREADS = 4  # @param {type:"integer"}
SEED = 42  # @param {type:"integer"}
TREE_COUNTS = [200, 400, 600]
LOOKBACK_DAYS = 90
REPO_SHA = "e3acd135b5d61a504eb6f4f9577b6dc9c07fd4b2"  # код загрузки и общие утилиты
DATA_REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
POPULATION_PERMILLE = 100  # @param {type:"integer"}

SCAN_MEMORY_GB = 2
SCAN_THREADS = 2
SPILL_LIMIT_GB = 12
RUN_LEGACY_RETRIEVAL = False  # @param {type:"boolean"}
RUN_LEGACY_REPORT = False  # @param {type:"boolean"}
RESEARCH_COHORT_USERS = 2000 if DATA_SIZE == "50m" else 5000
RESEARCH_EVAL_USERS = 2000 if DATA_SIZE == "50m" else 5000
OLDER_TRAIN_USERS = 750 if DATA_SIZE == "50m" else 2500
RESEARCH_TREES = [300, 600, 1200] if DATA_SIZE == "50m" else [600, 1200, 2000]
RESEARCH_DOWNLOAD = True
LONG_SOURCE_BUDGET = 1000
LONG_CANDIDATE_BUDGET = 3000
LONG_NEGATIVES = 300
LISTEN_ITEMS_PER_USER = 200
LISTEN_ALS_ITEMS = 300000
COSINE_ITEMS = 100000
NEW_SOURCE_BUDGET = 300
BPR_ITERATIONS = 30
EASE_ITEMS = 5000
SANSA_ITEMS = 30000
SANSA_DENSITY = 0.001
SANSA_SHA = "735a782599cceb3663f8df14d75edfdf4c86874c"
NEURAL_ITEMS = 50000
NEURAL_USERS = 50000
RUN_NEURAL_SOURCES = True
VAE_EPOCHS = 5
SASREC_EPOCHS = 5
TOWER_EPOCHS = 5
TORCH_DEVICE = "auto"  # "cpu", "auto", "gpu" (CUDA or ROCm in PyTorch)
TREE_COUNTS = RESEARCH_TREES
EVAL_USERS = RESEARCH_EVAL_USERS
RESEARCH_DOWNLOAD = False
THREADS = args.threads
RESEARCH_COHORT_USERS = args.cohort_users or RESEARCH_COHORT_USERS
TRAIN_USERS = args.train_users or TRAIN_USERS
OLDER_TRAIN_USERS = args.older_train_users or OLDER_TRAIN_USERS
RESEARCH_EVAL_USERS = args.eval_users or RESEARCH_EVAL_USERS
RESEARCH_TREES = args.trees or RESEARCH_TREES
TREE_COUNTS = RESEARCH_TREES
EVAL_USERS = RESEARCH_EVAL_USERS
VAE_EPOCHS = SASREC_EPOCHS = TOWER_EPOCHS = args.epochs
RUN_NEURAL_SOURCES = not args.skip_neural
TORCH_DEVICE = args.torch_device


# environment
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = args.repo.resolve()
if not (REPO / "tools/explore_yambda.py").is_file():
    raise RuntimeError("Set REPO to the existing yambda_contest directory")
DATA = args.data.resolve() if args.data else REPO / "data" / DATA_SIZE
CACHE = (
    args.cache_dir.resolve()
    if args.cache_dir
    else REPO / "data" / ("research_cache_v9_" + DATA_SIZE)
)
RUNS = args.out.resolve() if args.out else REPO / "eda_results" / RUN_LABEL
PERSIST = REPO / "data"
os.environ["POLARS_MAX_THREADS"] = str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = str(THREADS)
for folder in [DATA, CACHE, RUNS]:
    folder.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "tools"))
try:
    REPO_SHA = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
except (OSError, subprocess.CalledProcessError):
    REPO_SHA = "unversioned-local-checkout"
print("Repository:", REPO)
print("Results:", RUNS)


# imports
import gc
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import benchmark_candidates as bench
import duckdb
import explore_trends as core
import explore_yambda as base
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import train_candidate_ranker as old
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from catboost.utils import get_gpu_device_count
from IPython.display import SVG, display
from scipy import sparse
from threadpoolctl import threadpool_limits

DAY = 86400
KEYS = ["uid", "item_id"]
pa.set_cpu_count(THREADS)
versions = {
    p: importlib.metadata.version(p)
    for p in ["numpy", "polars", "pyarrow", "scipy", "implicit", "catboost", "duckdb"]
}
print(versions)
print("CUDA-устройств CatBoost:", get_gpu_device_count())
if TASK_TYPE == "GPU" and get_gpu_device_count() < 1:
    raise RuntimeError("Для CatBoost GPU выберите среду Colab с NVIDIA GPU")


def log(message):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"{time.strftime('%H:%M:%S')} {message}; peakRSS={rss:.2f} GiB", flush=True)


def save_table(name, frame):
    frame = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame
    frame.to_csv(RUNS / (name + ".csv"), index=False)
    display(frame)


def save_figure(name, fig):
    fig.tight_layout()
    path = RUNS / (name + ".svg")
    fig.savefig(path, bbox_inches="tight")
    display(SVG(filename=str(path)))
    plt.close(fig)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


versions.update(
    torch=torch.__version__,
    sansa=importlib.metadata.version("sansa"),
    faiss=importlib.metadata.version("faiss-cpu"),
)
import faiss
from sansa import SANSA

torch.set_num_threads(THREADS)
if TORCH_DEVICE == "gpu" and not torch.cuda.is_available():
    raise RuntimeError("PyTorch GPU requested but CUDA/ROCm is unavailable")
print("PyTorch:", torch.__version__, "ROCm:", torch.version.hip)
print(
    "PyTorch device:",
    "cpu"
    if TORCH_DEVICE == "cpu" or not torch.cuda.is_available()
    else torch.cuda.get_device_name(0),
)
print("CatBoost device:", TASK_TYPE)
if args.doctor:
    print("Dependency/device check passed; no data downloaded or models trained.")
    raise SystemExit(0)


# cache_functions
COLS = [
    "uid",
    "item_id",
    "timestamp",
    "is_organic",
    "played_ratio_pct",
    "track_length_seconds",
]


def population_mask(ids, seed, permille):
    """Stable SplitMix64 user sample, independent of likes, activity and targets."""
    x = np.asarray(ids, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(seed) + np.uint64(11400714819323198485)
        x = (x ^ x >> 30) * np.uint64(13787848793156543929)
        x = (x ^ x >> 27) * np.uint64(10723151780598845931)
        x = x ^ x >> 31
    return x % np.uint64(1000) < permille


# memory_functions
def sql_path(path):
    """Quote a local path as a SQL string literal."""
    return "'" + str(Path(path)).replace("'", "''") + "'"


@contextmanager
def scan_database():
    """Limit database buffers and permit intermediate tables to spill to disk."""
    spill_directory = CACHE / "duckdb_spill"
    spill_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(
        config={
            "memory_limit": f"{SCAN_MEMORY_GB}GB",
            "threads": SCAN_THREADS,
            "temp_directory": str(spill_directory),
            "max_temp_directory_size": f"{SPILL_LIMIT_GB}GB",
            "preserve_insertion_order": False,
        }
    )
    try:
        yield connection
    finally:
        connection.close()
        gc.collect()


def current_rss_gib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024**2
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


@contextmanager
def memory_stage(label):
    started = time.perf_counter()
    print(
        f"BEGIN {label}: RSS={current_rss_gib():.2f} GiB",
        flush=True,
    )
    try:
        yield
    finally:
        gc.collect()
        print(
            f"END {label}: RSS={current_rss_gib():.2f} GiB; "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )


def parquet_source_identity(path):
    source = Path(path)
    return {
        "dataset_revision": DATA_REVISION,
        "file": source.name,
        "bytes": source.stat().st_size,
    }


# context_functions
def load_context_bounded(data, users, max_users=0, seed=42):
    """Scan population likes in DuckDB; materialize only the selected cohort."""
    data = Path(data)
    user_path = Path(users)
    user_frame = pl.read_csv(user_path)
    if (
        user_frame.columns != ["uid"]
        or user_frame["uid"].null_count()
        or user_frame["uid"].n_unique() != user_frame.height
    ):
        raise ValueError("Expected one unique, non-null uid column")
    user_ids = user_frame["uid"].cast(pl.Int64).to_list()
    if not user_ids or max_users < 0:
        raise ValueError("Empty cohort or negative max_users")
    if max_users:
        user_ids = (
            np.random.default_rng(seed)
            .choice(user_ids, min(max_users, len(user_ids)), replace=False)
            .tolist()
        )
    user_ids = sorted(user_ids)
    source = data / "likes.parquet"
    signature = {
        "version": "cohort-likes-duckdb-v1",
        "source": parquet_source_identity(source),
        "users": user_ids,
        "seed": seed,
    }
    cache_key = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()[:24]
    cohort_path = CACHE / f"cohort_likes_{cache_key}.parquet"
    manifest_path = cohort_path.with_suffix(".json")
    if cohort_path.exists() and manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata["signature"] != signature:
            raise ValueError("Cohort likes signature mismatch")
        if pq.ParquetFile(cohort_path).metadata.num_rows != metadata["rows"]:
            raise ValueError("Incomplete cohort likes cache")
    else:
        partial = cohort_path.with_suffix(".partial.parquet")
        partial.unlink(missing_ok=True)
        with scan_database() as connection:
            first_timestamp, last_timestamp = connection.execute(
                f"SELECT min(timestamp), max(timestamp) FROM read_parquet({sql_path(source)})"
            ).fetchone()
            if last_timestamp is None:
                raise ValueError("Empty likes file")
            connection.register("selected_users", pa.table({"uid": user_ids}))
            query = f"""
                SELECT CAST(events.uid AS BIGINT) AS uid,
                       CAST(item_id AS BIGINT) AS item_id,
                       CAST(timestamp AS BIGINT) AS timestamp
                FROM read_parquet({sql_path(source)}) AS events
                INNER JOIN selected_users USING (uid)
                ORDER BY uid, timestamp, item_id
            """
            connection.execute(
                f"COPY ({query}) TO {sql_path(partial)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
            )
        partial.replace(cohort_path)
        metadata = {
            "signature": signature,
            "rows": pq.ParquetFile(cohort_path).metadata.num_rows,
            "start": int(first_timestamp),
            "end": int(last_timestamp) + 1,
        }
        manifest_path.write_text(json.dumps(metadata, indent=2))
    cohort_likes = pl.read_parquet(cohort_path, parallel="none", low_memory=True)
    # Ordering is required for the first-100-event target, including tied timestamps.
    cohort_likes = cohort_likes.sort(["uid", "timestamp", "item_id"])
    return core.Context(
        data=data,
        users=user_ids,
        likes=cohort_likes,
        end=metadata["end"],
        start=metadata["start"],
        seed=seed,
        user_sha256=sha256(user_path),
    )


# cache_extract_functions
def extract_cache(path, users, lo, hi, cache_dir, seed, permille, batch_size=None):
    """Filter raw listens into an on-disk Parquet file without decoding it into Python."""
    if not 0 <= permille <= 1000 or lo >= hi:
        raise ValueError("Invalid population fraction or time interval")
    path = Path(path)
    signature = {
        "version": "listen-cache-duckdb-v5",
        "source": parquet_source_identity(path),
        "users": sorted(users),
        "lo": lo,
        "hi": hi,
        "seed": seed,
        "population_permille": permille,
        "duckdb": duckdb.__version__,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[
        :24
    ]
    folder = Path(cache_dir) / f"listens_{key}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "events.parquet"
    manifest = folder / "manifest.json"
    if output.exists() and manifest.exists():
        metadata = json.loads(manifest.read_text())
        cached_file = pq.ParquetFile(output)
        if metadata["signature"] != signature:
            raise ValueError("Listen cache signature mismatch")
        if cached_file.metadata.num_rows != metadata["retained_rows"]:
            raise ValueError("Incomplete listen cache")
        if not set(COLS + ["cohort", "trend_sample"]).issubset(
            cached_file.schema_arrow.names
        ):
            raise ValueError("Incomplete listen cache schema")
        print(f"Using listen cache: {output}", flush=True)
        return output, metadata

    partial = folder / "events.partial.parquet"
    partial.unlink(missing_ok=True)
    started = time.perf_counter()
    with scan_database() as connection:
        # Only the UID column is scanned here. Sampling is independent of targets.
        population_ids = connection.execute(
            f"SELECT DISTINCT uid FROM read_parquet({sql_path(path)})"
        ).fetchnumpy()["uid"]
        sampled_ids = population_ids[population_mask(population_ids, seed, permille)]
        connection.register(
            "cohort_users", pa.table({"uid": np.asarray(users, dtype=np.int64)})
        )
        connection.register("sampled_users", pa.table({"uid": sampled_ids}))
        print(
            f"Filtering listens: {len(users):,} cohort UIDs; "
            f"{len(sampled_ids):,} population-sample UIDs",
            flush=True,
        )
        query = f"""
            SELECT CAST(events.uid AS BIGINT) AS uid,
                   CAST(item_id AS BIGINT) AS item_id,
                   CAST(timestamp AS BIGINT) AS timestamp,
                   is_organic, played_ratio_pct, track_length_seconds,
                   cohort_users.uid IS NOT NULL AS cohort,
                   sampled_users.uid IS NOT NULL AS trend_sample
            FROM read_parquet({sql_path(path)}) AS events
            LEFT JOIN cohort_users USING (uid)
            LEFT JOIN sampled_users USING (uid)
            WHERE timestamp >= {int(lo)} AND timestamp < {int(hi)}
              AND (cohort_users.uid IS NOT NULL OR sampled_users.uid IS NOT NULL)
        """
        connection.execute(
            f"COPY ({query}) TO {sql_path(partial)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
        )
        invalid_rows = connection.execute(
            f"""
            SELECT count(*) FROM read_parquet({sql_path(partial)})
            WHERE played_ratio_pct IS NULL OR played_ratio_pct < 0
               OR is_organic IS NULL OR is_organic NOT IN (0, 1)
            """
        ).fetchone()[0]
        if invalid_rows:
            raise ValueError(
                f"Invalid listening fields in {invalid_rows:,} retained rows"
            )
    partial.replace(output)
    metadata = {
        "signature": signature,
        "scanned_rows": pq.ParquetFile(path).metadata.num_rows,
        "retained_rows": pq.ParquetFile(output).metadata.num_rows,
        "cache_bytes": output.stat().st_size,
        "invalid_rows": 0,
        "seconds": time.perf_counter() - started,
    }
    temporary_manifest = manifest.with_suffix(".partial.json")
    temporary_manifest.write_text(json.dumps(metadata, indent=2))
    temporary_manifest.replace(manifest)
    return output, metadata


# population_functions
def validate_like_statistics(statistics):
    """Reject invalid aggregates rather than silently replacing them with zeros."""
    numeric_columns = old.STAT_COLS
    invalid = pl.any_horizontal(
        [
            pl.col(column).is_null() | ~pl.col(column).is_finite()
            for column in numeric_columns
        ]
    )
    invalid = invalid | (pl.col("decay7") < 0) | (pl.col("decay30") < 0)
    invalid = invalid | (pl.col("decay7") > pl.col("pop_all") + 1e-6)
    invalid = invalid | (pl.col("decay30") > pl.col("pop_all") + 1e-6)
    examples = statistics.filter(invalid).head(5)
    if examples.height:
        raise ValueError(
            "Invalid past-only like statistics: " + str(examples.to_dicts())
        )


def validate_feature_matrix(user_id, candidate_ids, matrix, feature_names):
    """Report the exact candidate and column when a feature is not finite."""
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature width mismatch: {matrix.shape[1]} vs {len(feature_names)}"
        )
    bad_positions = np.argwhere(~np.isfinite(matrix))
    if len(bad_positions):
        examples = [
            {
                "uid": int(user_id),
                "item_id": int(candidate_ids[row]),
                "feature": feature_names[column],
                "value": str(matrix[row, column]),
            }
            for row, column in bad_positions[:8]
        ]
        raise ValueError("Non-finite feature values: " + json.dumps(examples))


def write_population_corpus(source, cutoff, history_days, user_mod, seed, output):
    """Group population likes on disk, keeping the original Polars UID-hash cohort."""
    source = Path(source)
    output = Path(output)
    lower_bound = cutoff - history_days * DAY
    with scan_database() as connection:
        population_ids = connection.execute(
            f"SELECT DISTINCT uid FROM read_parquet({sql_path(source)}) "
            f"WHERE timestamp >= {int(lower_bound)} AND timestamp < {int(cutoff)}"
        ).fetchnumpy()["uid"]
        original_uid_type = pl.read_parquet_schema(source)["uid"]
        uid_series = pl.Series("uid", population_ids, dtype=original_uid_type)
        selected_ids = uid_series.filter((uid_series.hash(seed=seed) % user_mod) == 0)
        connection.register("cf_population", pa.table({"uid": selected_ids.to_arrow()}))
        query = f"""
            SELECT events.uid, item_id, max(timestamp) AS timestamp
            FROM read_parquet({sql_path(source)}) AS events
            INNER JOIN cf_population USING (uid)
            WHERE timestamp >= {int(lower_bound)} AND timestamp < {int(cutoff)}
            GROUP BY events.uid, item_id
        """
        output.unlink(missing_ok=True)
        connection.execute(
            f"COPY ({query}) TO {sql_path(output)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
        )


def cached_like_statistics(data, cutoff, cache_dir):
    """Compute full-population past-only popularity in a spilling database scan."""
    source = Path(data) / "likes.parquet"
    signature = {
        "version": "past-like-statistics-duckdb-v2",
        "source": parquet_source_identity(source),
        "cutoff": cutoff,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[
        :24
    ]
    output = Path(cache_dir) / f"like_statistics_{key}.parquet"
    if output.exists():
        cached = pl.read_parquet(output, parallel="none", low_memory=True)
        validate_like_statistics(cached)
        return cached, signature
    partial = output.with_suffix(".partial.parquet")
    partial.unlink(missing_ok=True)
    with scan_database() as connection:
        query = f"""
            SELECT CAST(item_id AS BIGINT) AS item_id,
                   count(*) AS pop_all,
                   count(*) FILTER (WHERE timestamp >= {cutoff - 30 * DAY}) AS pop30,
                   count(*) FILTER (WHERE timestamp >= {cutoff - 7 * DAY}) AS pop7,
                   count(*) FILTER (WHERE timestamp >= {cutoff - 2 * DAY}) AS pop2,
                   count(*) FILTER (
                       WHERE timestamp >= {cutoff - 14 * DAY}
                         AND timestamp < {cutoff - 7 * DAY}
                   ) AS prev7,
                   sum(exp((CAST(timestamp AS DOUBLE) - {float(cutoff)}) / {7.0 * DAY})) AS decay7,
                   sum(exp((CAST(timestamp AS DOUBLE) - {float(cutoff)}) / {30.0 * DAY})) AS decay30,
                   count(DISTINCT uid) FILTER (
                       WHERE timestamp >= {cutoff - 7 * DAY}
                   ) AS recent_users,
                   min(timestamp) AS first_seen_ts
            FROM read_parquet({sql_path(source)})
            WHERE timestamp < {int(cutoff)}
            GROUP BY item_id
        """
        connection.execute(
            f"COPY ({query}) TO {sql_path(partial)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
        )
    statistics = pl.read_parquet(partial, parallel="none", low_memory=True)
    statistics = statistics.with_columns(
        ((pl.col("pop7") + 2.0) / (pl.col("prev7") + 2.0)).alias("ratio7")
    ).with_columns(
        (
            pl.col("ratio7").log() * pl.col("recent_users").cast(pl.Float64).log1p()
        ).alias("growth7"),
        (pl.col("pop7").cast(pl.Float64) - pl.col("prev7").cast(pl.Float64)).alias(
            "momentum7"
        ),
    )
    validate_like_statistics(statistics)
    statistics.write_parquet(partial)
    partial.replace(output)
    output.with_suffix(".json").write_text(json.dumps(signature, indent=2))
    return statistics, signature


def population_eda_tables(source):
    with scan_database() as connection:
        population = connection.execute(
            f"""
            SELECT count(*) AS events, count(DISTINCT uid) AS users,
                   count(DISTINCT item_id) AS tracks,
                   min(timestamp) AS start, max(timestamp) AS "end"
            FROM read_parquet({sql_path(source)})
            """
        ).df()
        weekly = connection.execute(
            f"""
            SELECT CAST(floor(timestamp / {7 * DAY}) AS BIGINT) AS week,
                   count(*) AS likes, count(DISTINCT uid) AS active_users
            FROM read_parquet({sql_path(source)})
            GROUP BY week ORDER BY week
            """
        ).df()
    return pl.from_pandas(population), pl.from_pandas(weekly)


def recent_track_counts(source, cutoff):
    output = CACHE / f"eda_weekly_tracks_{cutoff}.parquet"
    if not output.exists():
        with scan_database() as connection:
            query = f"""
                SELECT CAST(floor((timestamp - {cutoff - 28 * DAY}) / {7 * DAY}) AS BIGINT) AS week,
                       item_id, count(*) AS len
                FROM read_parquet({sql_path(source)})
                WHERE timestamp >= {cutoff - 28 * DAY} AND timestamp < {int(cutoff)}
                GROUP BY week, item_id
            """
            partial = output.with_suffix(".partial.parquet")
            partial.unlink(missing_ok=True)
            connection.execute(
                f"COPY ({query}) TO {sql_path(partial)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
            )
            partial.replace(output)
    return pl.read_parquet(output, parallel="none", low_memory=True)


# download
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

INPUTS = [
    "likes.parquet",
    "listens.parquet",
    "dislikes.parquet",
    "unlikes.parquet",
    "undislikes.parquet",
    "artist_item_mapping.parquet",
    "album_item_mapping.parquet",
]
MODEL_INPUTS = [name for name in INPUTS if name != "listens.parquet"]
remote_paths = {
    name: ("flat/" + DATA_SIZE + "/" if name in INPUTS[:5] else "") + name
    for name in INPUTS
}
upstream = []
for name in INPUTS:
    info = get_hf_file_metadata(
        hf_hub_url(
            "yandex/yambda",
            remote_paths[name],
            repo_type="dataset",
            revision=DATA_REVISION,
        )
    )
    if info.size is None:
        raise ValueError("Не получен размер файла: " + name)
    if info.commit_hash != DATA_REVISION:
        raise ValueError("Ревизия источника отличается: " + name)
    upstream.append(
        dict(
            file=name,
            repo_path=remote_paths[name],
            bytes=info.size,
            revision=info.commit_hash,
            etag=info.etag,
        )
    )
missing_bytes = sum(
    row["bytes"] for row in upstream if not (DATA / row["file"]).is_file()
)
headroom = 20 * 1024**3
free = shutil.disk_usage(DATA).free
save_table(
    "download_plan", pd.DataFrame(upstream).assign(GiB=lambda f: f.bytes / 1024**3)
)
print(
    f"Предстоит скачать до {missing_bytes / 1024**3:.1f} GiB; свободно {free / 1024**3:.1f} GiB"
)
if free < missing_bytes + headroom:
    raise RuntimeError(
        f"Нужно не менее {(missing_bytes + headroom) / 1024**3:.1f} GiB свободного места, включая 20 GiB для расчётов"
    )
for row in upstream:
    name = row["file"]
    dest = DATA / name
    if dest.is_file() and dest.stat().st_size != row["bytes"]:
        raise ValueError("Размер уже существующего файла не совпадает: " + str(dest))
    if dest.is_file() and re.fullmatch(r"[0-9a-f]{64}", row["etag"].strip('"')):
        if sha256(dest) != row["etag"].strip('"'):
            raise ValueError(
                "Existing file checksum differs from pinned Hugging Face revision: "
                + str(dest)
            )
        source = dest.resolve()
    else:
        source = Path(
            hf_hub_download(
                "yandex/yambda",
                row["repo_path"],
                repo_type="dataset",
                revision=DATA_REVISION,
            )
        ).resolve()
        if source.stat().st_size != row["bytes"]:
            raise ValueError("Incomplete download: " + name)
        if not dest.exists():
            dest.symlink_to(source)
        elif dest.resolve() != source and sha256(dest) != sha256(source):
            raise ValueError("Existing file differs from pinned download: " + str(dest))
    print("Готово:", name, f"{source.stat().st_size / 1024**3:.2f} GiB", flush=True)
USERS = RUNS / "test_users.csv"
user_bytes = (REPO / "input/test_users.csv").read_bytes()
if USERS.exists() and USERS.read_bytes() != user_bytes:
    raise ValueError("Изменился список пользователей")
USERS.write_bytes(user_bytes)
(RUNS / "dataset_sources.json").write_text(json.dumps(upstream, indent=2))


# research_select_cohort
def v9_select_cohort(data, official_users, output, maximum, seed):
    """Select only by history before the earliest feature cutoff, never by labels."""
    official = pl.read_csv(official_users)["uid"].cast(pl.Int64).to_list()
    with scan_database() as connection:
        first, last = connection.execute(
            f"SELECT min(timestamp), max(timestamp) FROM read_parquet({sql_path(data / 'likes.parquet')})"
        ).fetchone()
        earliest = int(last) + 1 - 240 * DAY
        if earliest <= int(first):
            raise ValueError("Not enough history for the proposed temporal folds")
        available = (
            connection.execute(
                f"SELECT DISTINCT CAST(uid AS BIGINT) AS uid FROM read_parquet({sql_path(data / 'likes.parquet')}) "
                f"WHERE timestamp < {earliest} ORDER BY uid"
            )
            .fetchnumpy()["uid"]
            .tolist()
        )
    overlap = sorted(set(available) & set(official))
    other = sorted(set(available) - set(official))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(overlap, min(maximum, len(overlap)), replace=False).tolist()
    if len(chosen) < maximum and other:
        chosen.extend(
            rng.choice(
                other, min(maximum - len(chosen), len(other)), replace=False
            ).tolist()
        )
    if not chosen:
        raise ValueError("No users with past history")
    frame = pl.DataFrame({"uid": sorted(chosen)})
    if (
        output.exists()
        and pl.read_csv(output)["uid"].to_list() != frame["uid"].to_list()
    ):
        raise ValueError("Cohort changed: choose a new RUN_LABEL")
    frame.write_csv(output)
    info = dict(
        selection_cut=earliest,
        official_users=len(official),
        available_history_users=len(available),
        official_with_early_history=len(overlap),
        selected_users=len(chosen),
        selected_official=len(set(chosen) & set(official)),
        selection_uses_future_labels=False,
    )
    return set(official), info


# research_cohort
OFFICIAL_USERS = USERS
USERS = RUNS / "experiment_users.csv"
OFFICIAL_UIDS, cohort_info = v9_select_cohort(
    DATA, OFFICIAL_USERS, USERS, RESEARCH_COHORT_USERS, SEED
)
(RUNS / "cohort_audit.json").write_text(json.dumps(cohort_info, indent=2))
display(pd.DataFrame([cohort_info]))


# load_context
with memory_stage("Selected-user likes"):
    ctx = load_context_bounded(DATA, USERS, MAX_USERS, SEED)
    all_users = pl.read_csv(USERS)["uid"].cast(pl.Int64).to_list()
    all_cuts = [ctx.end - days * DAY for days in [150, 120, 90, 60, 30]]
    if min(all_cuts) <= ctx.start:
        raise ValueError("Ранний срез предшествует истории датасета")
print(f"Evaluation users: {len(ctx.users):,}; selected likes: {ctx.likes.height:,}")


# extract_listen_cache
with memory_stage("Listening cache: UID sample and filtered Parquet"):
    EVENTS, cache_info = extract_cache(
        DATA / "listens.parquet",
        all_users,
        0,
        ctx.end,
        CACHE,
        SEED,
        POPULATION_PERMILLE,
    )
    (RUNS / "listen_cache_manifest.json").write_text(json.dumps(cache_info, indent=2))
print(f"Retained events: {cache_info['retained_rows']:,}")
print(f"Listening cache: {cache_info['cache_bytes'] / 1024**3:.2f} GiB")


# input_fingerprints
with memory_stage("Input fingerprints"):
    input_hashes = {name: sha256(DATA / name) for name in MODEL_INPUTS}
    verified = dict(
        sha256=input_hashes,
        stats={
            name: [
                str((DATA / name).resolve()),
                (DATA / name).stat().st_size,
                (DATA / name).stat().st_mtime_ns,
            ]
            for name in MODEL_INPUTS
        },
    )
    (DATA / "portable_inputs_verified.json").write_text(json.dumps(verified))


# run_configuration
configuration = dict(
    repo=REPO_SHA,
    source_tools_sha256={
        module.__name__: sha256(module.__file__) for module in [base, core, bench, old]
    },
    torch_device=TORCH_DEVICE,
    data_revision=DATA_REVISION,
    versions=versions,
    users_sha256=sha256(USERS),
    listen_extraction=cache_info["signature"],
    input_sha256=input_hashes,
    max_users=MAX_USERS,
    train_users=TRAIN_USERS,
    eval_users=EVAL_USERS,
    seed=SEED,
    cf_mod=CF_MOD,
    factors=ALS_FACTORS,
    als_iterations=ALS_ITERATIONS,
    knn_items=KNN_ITEMS,
    knn_neighbors=KNN_NEIGHBORS,
    tree_counts=TREE_COUNTS,
    task_type=TASK_TYPE,
    threads=THREADS,
    lookback_days=LOOKBACK_DAYS,
    implementation="research-v9",
    dataset_size=DATA_SIZE,
    scan_memory_gb=SCAN_MEMORY_GB,
    scan_threads=SCAN_THREADS,
    spill_limit_gb=SPILL_LIMIT_GB,
)
config_path = RUNS / "configuration.json"
if config_path.exists() and json.loads(config_path.read_text()) != configuration:
    raise ValueError("Параметры изменились: выберите новый RUN_LABEL")
config_path.write_text(json.dumps(configuration, indent=2))


# input_inventory
save_table(
    "input_schema",
    pd.DataFrame(
        [
            {
                "file": name,
                "rows": pq.ParquetFile(DATA / name).metadata.num_rows,
                "columns": str(pq.ParquetFile(DATA / name).schema_arrow),
            }
            for name in INPUTS
        ]
    ),
)


# population_eda
population, weekly = population_eda_tables(DATA / "likes.parquet")
save_table("population", population)
activity = ctx.likes.group_by("uid").agg(
    pl.len().alias("events"), pl.col("item_id").n_unique().alias("tracks")
)
cohort_activity = (
    pl.DataFrame({"uid": ctx.users}).join(activity, on="uid", how="left").fill_null(0)
)
save_table("cohort_activity", cohort_activity.describe())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(weekly["week"], weekly["likes"])
axes[0].set(xlabel="Week", ylabel="Like events")
axes[1].hist(np.log1p(cohort_activity["events"].to_numpy()), bins=50)
axes[1].set(xlabel="log(1 + user likes)", ylabel="Target users")
save_figure("activity", fig)
weekly.write_csv(RUNS / "weekly_activity.csv")


# target_metric
def temporal_target(context, cut, label_end, users):
    target = context.likes.filter(
        pl.col("uid").is_in(users)
        & (pl.col("timestamp") >= cut)
        & (pl.col("timestamp") < label_end)
    ).sort(["uid", "timestamp", "item_id"])
    target = target.group_by("uid", maintain_order=True).head(100)
    truth = {int(u): set(g["item_id"].to_list()) for (u,), g in target.group_by("uid")}
    complete = set(
        target.group_by("uid").len().filter(pl.col("len") == 100)["uid"].to_list()
    )
    return truth, complete, target


def recall_values(predictions, truth, users):
    return np.array(
        [
            len(set(predictions.get(u, [])) & set(truth.get(u, []))) / len(truth[u])
            if truth.get(u)
            else 0.0
            for u in users
        ]
    )


def recall_metrics(predictions, truth, users, complete):
    values = recall_values(predictions, truth, users)
    active = np.array([bool(truth.get(u)) for u in users])
    full = np.array([u in complete for u in users])
    return dict(
        recall_all=float(values.mean()),
        recall_active=float(values[active].mean()) if active.any() else None,
        recall_full100=float(values[full].mean()) if full.any() else None,
        n_users=len(users),
        n_active=int(active.sum()),
        n_full100=int(full.sum()),
    )


censoring = []
for days in [30, 60, 90, 120, 150]:
    cut = ctx.end - days * DAY
    truth, complete, target = temporal_target(ctx, cut, ctx.end, ctx.users)
    censoring.append(
        dict(
            future_days=days,
            active_users=len(truth),
            full100_users=len(complete),
            active_fraction=len(truth) / len(ctx.users),
            full100_fraction=len(complete) / len(ctx.users),
        )
    )
censoring = pd.DataFrame(censoring)
save_table("future_availability", censoring)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    censoring.future_days,
    100 * censoring.active_fraction,
    "o-",
    label="Any future likes",
)
ax.plot(
    censoring.future_days,
    100 * censoring.full100_fraction,
    "o-",
    label="Complete 100 events",
)
ax.set(xlabel="Observed future (days)", ylabel="Target users (%)")
ax.legend()
save_figure("future_availability", fig)


# trend_eda
cut = ctx.end - 30 * DAY
trend_sample = recent_track_counts(DATA / "likes.parquet", cut)
top_items = (
    trend_sample.group_by("item_id")
    .agg(pl.col("len").sum())
    .sort("len", descending=True)
    .head(8)["item_id"]
    .to_list()
)
totals = trend_sample.group_by("week").agg(pl.col("len").sum().alias("total"))
shares = (
    trend_sample.filter(pl.col("item_id").is_in(top_items))
    .join(totals, on="week")
    .with_columns((pl.col("len") / pl.col("total")).alias("share"))
)
fig, ax = plt.subplots(figsize=(10, 4))
for item in top_items:
    g = shares.filter(pl.col("item_id") == item).sort("week")
    ax.plot(g["week"], 100 * g["share"], "o-", label=str(item))
ax.set(xlabel="Week before cutoff (0–3)", ylabel="Share of population likes (%)")
ax.legend(ncol=4, fontsize=8)
save_figure("track_trends", fig)
save_table("track_trends", shares.sort(["item_id", "week"]))
del trend_sample, shares
gc.collect()


# listen_functions
def pair_stats(events, cut, lookback, users=None):
    q = pl.scan_parquet(events).filter(
        pl.col("cohort")
        & (pl.col("timestamp") < cut)
        & (pl.col("timestamp") >= cut - lookback * DAY)
    )
    if users is not None:
        q = q.filter(pl.col("uid").is_in(users))
    return base.collect(
        q.group_by(KEYS).agg(
            pl.len().alias("plays"),
            pl.col("timestamp").max().alias("last_ts"),
            (pl.col("played_ratio_pct") > 50).sum().alias("qualified"),
            (pl.col("played_ratio_pct") >= 90).sum().alias("near_full"),
            (pl.col("played_ratio_pct") <= 20).sum().alias("short"),
            (pl.col("played_ratio_pct") > 100).sum().alias("replays"),
            (pl.col("is_organic") == 1).sum().alias("organic"),
            ((pl.col("is_organic") == 1) & (pl.col("played_ratio_pct") > 50))
            .sum()
            .alias("organic_qualified"),
            pl.col("timestamp")
            .filter(pl.col("played_ratio_pct") > 50)
            .max()
            .alias("last_qualified_ts"),
            pl.col("timestamp")
            .filter((pl.col("is_organic") == 1) & (pl.col("played_ratio_pct") > 50))
            .max()
            .alias("last_organic_ts"),
            pl.col("timestamp")
            .filter(pl.col("played_ratio_pct") >= 90)
            .max()
            .alias("last_full_ts"),
            pl.col("timestamp")
            .filter(pl.col("played_ratio_pct") <= 20)
            .max()
            .alias("last_short_ts"),
            (pl.col("timestamp") >= cut - 7 * DAY).sum().alias("plays7"),
        )
    )


def feedback_pairs(data, users, cut, history):
    result = history.group_by(KEYS).agg(pl.col("timestamp").max().alias("last_like"))
    for name in ["dislikes", "undislikes", "unlikes"]:
        path = Path(data) / (name + ".parquet")
        if not path.exists():
            raise FileNotFoundError(path)
        df = base.collect(
            pl.scan_parquet(path)
            .filter(pl.col("uid").is_in(users) & (pl.col("timestamp") < cut))
            .group_by(KEYS)
            .agg(pl.col("timestamp").max().alias("last_" + name))
        )
        df = df.with_columns(pl.col(KEYS).cast(pl.Int64))
        result = result.join(df, on=KEYS, how="full", coalesce=True)
    # Strict inequalities: same-time actions have unknown ordering.
    return result.with_columns(
        (
            pl.col("last_dislikes").is_not_null()
            & (
                pl.col("last_undislikes").is_null()
                | (pl.col("last_dislikes") > pl.col("last_undislikes"))
            )
        ).alias("active_dislike"),
        (
            pl.col("last_unlikes").is_not_null()
            & (
                pl.col("last_like").is_null()
                | (pl.col("last_unlikes") > pl.col("last_like"))
            )
        ).alias("active_unlike"),
    )


def listen_lists(pairs, cut, budget):
    """Historical fields only. No permanent exclusion of previously liked tracks."""
    specs = {
        "listen_recent": (pl.col("plays") > 0, pl.col("last_ts").cast(pl.Float64)),
        "listen_qualified": (
            pl.col("qualified") > 0,
            pl.col("qualified").cast(pl.Float64).log1p()
            * (-(cut - pl.col("last_qualified_ts")) / (30 * DAY)).exp(),
        ),
        "listen_organic": (
            pl.col("organic_qualified") > 0,
            pl.col("organic_qualified").cast(pl.Float64).log1p()
            * (-(cut - pl.col("last_organic_ts")) / (30 * DAY)).exp(),
        ),
        "listen_repeated_full": (
            pl.col("near_full") >= 2,
            pl.col("near_full").cast(pl.Float64).log1p()
            * (-(cut - pl.col("last_full_ts")) / (30 * DAY)).exp(),
        ),
    }
    lists = {}
    for name, (mask, score) in specs.items():
        ordered = (
            pairs.filter(mask)
            .with_columns(score.alias("score"))
            .sort(["uid", "score", "item_id"], descending=[False, True, False])
            .group_by("uid", maintain_order=True)
            .head(budget)
        )
        lists[name] = base.user_lists(ordered)
    return lists


def trend_lists(events, cut, budget, min_support):
    q = pl.scan_parquet(events).filter(
        pl.col("trend_sample")
        & (pl.col("timestamp") < cut)
        & (pl.col("timestamp") >= cut - 14 * DAY)
    )
    bins = (
        pl.when(pl.col("timestamp") >= cut - 7 * DAY)
        .then(pl.lit("current"))
        .otherwise(pl.lit("previous"))
    )
    df = base.collect(
        q.group_by("item_id", bins.alias("period")).agg(
            pl.len().alias("plays"),
            pl.col("uid").n_unique().alias("listeners"),
            pl.col("uid")
            .filter(pl.col("played_ratio_pct") > 50)
            .n_unique()
            .alias("qualified_listeners"),
            pl.col("uid")
            .filter((pl.col("played_ratio_pct") > 50) & (pl.col("is_organic") == 1))
            .n_unique()
            .alias("organic_listeners"),
        )
    )
    cols = ["plays", "listeners", "qualified_listeners", "organic_listeners"]
    cur = df.filter(pl.col("period") == "current").drop("period")
    prev = (
        df.filter(pl.col("period") == "previous")
        .drop("period")
        .rename({c: "prev_" + c for c in cols})
    )
    stats = cur.join(prev, on="item_id", how="full", coalesce=True).fill_null(0)
    stats = stats.with_columns(
        (
            (
                (pl.col("qualified_listeners") + 2.0)
                / (pl.col("prev_qualified_listeners") + 2.0)
            ).log()
            * pl.col("qualified_listeners").cast(pl.Float64).log1p()
        ).alias("growth")
    )
    specs = {
        "sample_listeners7": "listeners",
        "sample_qualified7": "qualified_listeners",
        "sample_organic7": "organic_listeners",
        "sample_growth7": "growth",
    }
    lists = {}
    for name, field in specs.items():
        f = stats.filter(pl.col(field) > 0)
        if field == "growth":
            f = f.filter(pl.col("qualified_listeners") >= min_support)
        lists[name] = (
            f.sort([field, "item_id"], descending=[True, False])
            .head(budget)["item_id"]
            .to_list()
        )
    return stats, lists


listening = SimpleNamespace(
    pair_stats=pair_stats,
    feedback_pairs=feedback_pairs,
    listen_lists=listen_lists,
    trend_lists=trend_lists,
)


# listen_eda
cut = ctx.end - 30 * DAY
listen_pairs = pair_stats(EVENTS, cut, LOOKBACK_DAYS, ctx.users)
truth, complete, _ = temporal_target(ctx, cut, ctx.end, ctx.users)
rows = []
for name, mask in [
    ("any_listen", pl.col("plays") > 0),
    ("qualified", pl.col("qualified") > 0),
    ("repeated_full", pl.col("near_full") >= 2),
    ("short_only", (pl.col("short") > 0) & (pl.col("qualified") == 0)),
]:
    groups = {
        int(u): set(g["item_id"].to_list())
        for (u,), g in listen_pairs.filter(mask).group_by("uid")
    }
    active = [u for u in ctx.users if truth.get(u)]
    rows.append(
        dict(
            group=name,
            pairs=int(listen_pairs.filter(mask).height),
            target_share_active=float(recall_values(groups, truth, active).mean()),
            target_share_full100=float(
                recall_values(groups, truth, sorted(complete)).mean()
            )
            if complete
            else None,
        )
    )
save_table("listening_target_support", pd.DataFrame(rows))
fig, ax = plt.subplots(figsize=(9, 4))
active_listening = listen_pairs.group_by("uid").agg(
    pl.col("plays").sum().alias("plays")
)
ax.hist(np.log1p(active_listening["plays"].to_numpy()), bins=50)
ax.set(
    xlabel="log(1 + listening events in history)", ylabel="Users with listening history"
)
save_figure("listening_activity", fig)
del listen_pairs
gc.collect()


# like_features
LISTEN_COUNTS = [
    "plays",
    "qualified",
    "near_full",
    "short",
    "organic",
    "organic_qualified",
    "replays",
    "plays7",
]
LISTEN_TIMES = [
    "last_ts",
    "last_qualified_ts",
    "last_organic_ts",
    "last_full_ts",
    "last_short_ts",
]
TREND_COLS = [
    "listeners",
    "qualified_listeners",
    "organic_listeners",
    "prev_listeners",
    "prev_qualified_listeners",
    "prev_organic_listeners",
    "growth",
]
EXTRA = (
    ["log_listen_" + c for c in LISTEN_COUNTS]
    + ["listen_" + c + "_age_days" for c in LISTEN_TIMES]
    + [
        "listen_qualified_fraction",
        "listen_short_fraction",
        "listen_organic_fraction",
        "listen_full_fraction",
        "active_dislike",
        "active_unlike",
        "dislike_age_days",
        "unlike_age_days",
    ]
    + ["sample_log_" + c if c != "growth" else "sample_growth" for c in TREND_COLS]
    + [
        name + "_rr"
        for name in [
            "listen_recent",
            "listen_qualified",
            "listen_organic",
            "listen_repeated_full",
        ]
    ]
    + ["listen_family_rr", "combined_rr", "in_like_pool", "in_listen_pool"]
)
FEATURES = old.FEATURES + EXTRA
SAMPLE_FEATURES = [
    "sample_log_" + c if c != "growth" else "sample_growth" for c in TREND_COLS
]
MODES = {
    "base_features": old.FEATURES,
    "listen_no_sample_trends": [c for c in FEATURES if c not in SAMPLE_FEATURES],
    "listen_features": FEATURES,
}


class LikeFeatures:
    """Same past-only features as the original model, extended to arbitrary items."""

    def __init__(self, ctx, cut, users, cache_dir, artist_catalog, source_budget=1000):
        self.cut = cut
        self.source_budget = source_budget
        self.history = ctx.likes.filter(
            (pl.col("timestamp") < cut) & pl.col("uid").is_in(users)
        )
        stats, _ = cached_like_statistics(ctx.data, cut, cache_dir)
        self.lists = bench.global_lists(stats, source_budget, 10)
        self.global_order = core.fuse(
            list(self.lists.values()), sum(map(len, self.lists.values()))
        )
        self.global_ranks = {int(i): r for r, i in enumerate(self.global_order, 1)}
        self.global_maps = {
            k: {int(i): r for r, i in enumerate(v, 1)} for k, v in self.lists.items()
        }
        self.artists = bench.ArtistSource(
            ctx.data, self.history, stats, cut, artist_catalog
        )
        self.index = {int(i): j for j, i in enumerate(stats["item_id"])}
        self.values = stats.select(old.STAT_COLS).to_numpy().astype(np.float32)
        self.histories = {int(u): g for (u,), g in self.history.group_by("uid")}

    def user_sources(self, u):
        personal = self.artists.recommend(u, self.source_budget)
        artist_order = core.fuse(
            list(personal.values()), sum(map(len, personal.values()))
        )
        full_order = core.fuse(
            [self.global_order, artist_order],
            len(self.global_order) + len(artist_order),
        )
        return (personal, artist_order, full_order)

    def features(self, user_id, candidate_ids, source_lists):
        personal_sources, artist_order, full_order = source_lists
        n_items = len(candidate_ids)
        item_statistics = np.zeros((n_items, len(old.STAT_COLS)), np.float32)
        item_statistics[:, 8] = 1.0
        known = [
            (candidate_row, self.index[int(item_id)])
            for candidate_row, item_id in enumerate(candidate_ids)
            if int(item_id) in self.index
        ]
        if known:
            destination_rows, statistic_rows = zip(*known)
            item_statistics[list(destination_rows)] = self.values[list(statistic_rows)]
        item_statistics[:, :8] = np.log1p(item_statistics[:, :8])
        item_statistics[:, 11] = (self.cut - item_statistics[:, 11]) / core.DAY
        for candidate_row, item_id in enumerate(candidate_ids):
            if int(item_id) not in self.index:
                item_statistics[candidate_row, 11] = -1.0
        history = self.histories.get(user_id)
        history_count = history.height if history is not None else 0
        recent = (
            float((history["timestamp"] >= self.cut - 30 * core.DAY).mean())
            if history_count
            else 0.0
        )
        recency = (
            (self.cut - int(history["timestamp"].max())) / core.DAY
            if history_count
            else -1.0
        )
        seen = (
            {
                int(item_id): (int(like_count), int(last_like_timestamp))
                for item_id, like_count, last_like_timestamp in history.group_by(
                    "item_id"
                )
                .agg(pl.len(), pl.col("timestamp").max())
                .iter_rows()
            }
            if history_count
            else {}
        )
        user_features = np.empty((n_items, 6), np.float32)
        user_features[:, 0] = np.log1p(history_count)
        user_features[:, 1] = recent
        user_features[:, 2] = recency
        user_features[:, 3] = [float(int(item_id) in seen) for item_id in candidate_ids]
        user_features[:, 4] = [
            (self.cut - seen[int(item_id)][1]) / core.DAY
            if int(item_id) in seen
            else -1.0
            for item_id in candidate_ids
        ]
        user_features[:, 5] = [
            np.log1p(seen.get(int(item_id), (0, 0))[0]) for item_id in candidate_ids
        ]
        source_positions = {
            **self.global_maps,
            **{
                k: {
                    int(item_id): candidate_row
                    for candidate_row, item_id in enumerate(values, 1)
                }
                for k, values in personal_sources.items()
            },
        }
        source_features = np.asarray(
            [
                [
                    1 / source_positions[k][int(item_id)]
                    if int(item_id) in source_positions[k]
                    else 0.0
                    for k in old.CHANNELS
                ]
                for item_id in candidate_ids
            ],
            np.float32,
        )
        artist_positions = {
            int(item_id): candidate_row
            for candidate_row, item_id in enumerate(artist_order, 1)
        }
        fusion_positions = {
            int(item_id): candidate_row
            for candidate_row, item_id in enumerate(full_order, 1)
        }
        fusion_features = np.asarray(
            [
                [
                    1 / self.global_ranks[int(item_id)]
                    if int(item_id) in self.global_ranks
                    else 0.0,
                    1 / artist_positions[int(item_id)]
                    if int(item_id) in artist_positions
                    else 0.0,
                    1 / fusion_positions[int(item_id)]
                    if int(item_id) in fusion_positions
                    else 0.0,
                    float(np.count_nonzero(source_features[candidate_row])),
                ]
                for candidate_row, item_id in enumerate(candidate_ids)
            ],
            np.float32,
        )
        return np.concatenate(
            [item_statistics, user_features, source_features, fusion_features], axis=1
        )


# listen_features
class ListenFeatures:
    def __init__(self, events, ctx, cut, users, history, lookback, source_budget):
        self.cut = cut
        log(f"BUILD listen aggregates cutoff={cut}")
        pairs = listening.pair_stats(events, cut, lookback, users)
        self.lists = listening.listen_lists(pairs, cut, source_budget)
        self.pairs = {int(u): g for (u,), g in pairs.group_by("uid")}
        feedback = listening.feedback_pairs(ctx.data, users, cut, history)
        self.feedback = {int(u): g for (u,), g in feedback.group_by("uid")}
        log(f"BUILD sample trends cutoff={cut}")
        trend, _ = listening.trend_lists(events, cut, source_budget, 10)
        self.trend_index = {int(i): j for j, i in enumerate(trend["item_id"])}
        self.trend_values = trend.select(TREND_COLS).to_numpy().astype(np.float32)
        self.trend_values[:, :6] = np.log1p(self.trend_values[:, :6])

    def features(self, user_id, candidate_ids, combined):
        user_history = self.pairs.get(user_id)
        lookup = (
            {
                int(feature_row["item_id"]): feature_row
                for feature_row in user_history.iter_rows(named=True)
            }
            if user_history is not None
            else {}
        )
        feedback_history = self.feedback.get(user_id)
        feedback = (
            {
                int(feature_row["item_id"]): feature_row
                for feature_row in feedback_history.iter_rows(named=True)
            }
            if feedback_history is not None
            else {}
        )
        maps = {
            name: {
                int(i): candidate_row
                for candidate_row, i in enumerate(values.get(user_id, []), 1)
            }
            for name, values in self.lists.items()
        }
        family = core.fuse(
            [values.get(user_id, []) for values in self.lists.values()], 1000
        )
        family_positions = {
            int(i): candidate_row for candidate_row, i in enumerate(family, 1)
        }
        feature_matrix = np.zeros((len(candidate_ids), len(EXTRA)), np.float32)
        for candidate_row, item_id in enumerate(candidate_ids):
            item_id = int(item_id)
            listen_record = lookup.get(item_id, {})
            feedback_record = feedback.get(item_id, {})
            play_count = float(listen_record.get("plays", 0))
            play_denominator = max(1.0, play_count)
            feature_row = [
                np.log1p(listen_record.get(field, 0)) for field in LISTEN_COUNTS
            ]
            feature_row += [
                (self.cut - listen_record[field]) / core.DAY
                if listen_record.get(field) is not None
                else -1.0
                for field in LISTEN_TIMES
            ]
            feature_row += [
                listen_record.get(field, 0) / play_denominator
                for field in ["qualified", "short", "organic", "near_full"]
            ]
            feature_row += [
                float(feedback_record.get("active_dislike") or False),
                float(feedback_record.get("active_unlike") or False),
            ]
            feature_row += [
                (self.cut - feedback_record[field]) / core.DAY
                if feedback_record.get(field) is not None
                else -1.0
                for field in ["last_dislikes", "last_unlikes"]
            ]
            feature_row += (
                self.trend_values[self.trend_index[item_id]].tolist()
                if item_id in self.trend_index
                else [0.0] * len(TREND_COLS)
            )
            feature_row += [
                1 / source_positions[item_id] if item_id in source_positions else 0.0
                for source_positions in maps.values()
            ]
            feature_row += [
                1 / family_positions[item_id] if item_id in family_positions else 0.0,
                1 / (candidate_row + 1),
                float(item_id in combined["like"]),
                float(item_id in combined["listen"]),
            ]
            feature_matrix[candidate_row] = feature_row
        return feature_matrix


# cf_matrix
SOURCE_NAMES = ["als_likes", "als_listens", "bm25_likes", "bm25_listens"]
SIM_NAMES = ["last", "max5", "mean5", "max20", "mean20"]
CF_FEATURES = (
    [s + "_" + k for s in SOURCE_NAMES for k in ["score", "rr"]]
    + [
        "cf_als_like_dot",
        "cf_als_listen_dot",
        "cf_als_like_cos",
        "cf_als_listen_cos",
        "cf_item_norm",
        "cf_like_norm",
        "cf_listen_norm",
        "cf_item_known",
        "cf_like_known_fraction",
        "cf_listen_known_fraction",
    ]
    + [
        "cf_" + kind + "_sim_" + name
        for kind in ["like", "listen"]
        for name in SIM_NAMES
    ]
)


@dataclass(frozen=True)
class Config:
    user_mod: int = 3
    history_days: int = 180
    factors: int = 64
    iterations: int = 20
    knn_items: int = 200000
    knn_neighbors: int = 80
    source_budget: int = 1000
    threads: int = 4
    seed: int = 42
    retrieval_device: str = "auto"


def csr(pairs, users, items, cut, listens=False):
    if pairs.is_empty():
        return sparse.csr_matrix((len(users), len(items)), dtype=np.float32), np.zeros(
            len(users), np.float32
        )
    uid = np.asarray(pairs["uid"])
    iid = np.asarray(pairs["item_id"])
    cols = np.searchsorted(items, iid)
    ok = cols < len(items)
    ok &= items[np.minimum(cols, len(items) - 1)] == iid
    user_idx = {int(u): r for r, u in enumerate(users)}
    rows = np.array([user_idx[int(u)] for u in uid[ok]], np.int32)
    ages = (cut - pairs["timestamp"].to_numpy()[ok].astype(np.int64)) / base.DAY
    weights = (1 + 4 * np.exp2(-ages / 60)).astype(np.float32)
    if listens:
        weights = (
            1
            + 2
            * np.log1p(np.minimum(pairs["count"].to_numpy()[ok], 20))
            * np.exp2(-ages / 30)
        ).astype(np.float32)
    matrix = sparse.csr_matrix(
        (weights, (rows, cols[ok])), shape=(len(users), len(items)), dtype=np.float32
    )
    total = np.bincount([user_idx[int(u)] for u in uid], minlength=len(users))
    matched = np.bincount(rows, minlength=len(users))
    return matrix, (matched / np.maximum(total, 1)).astype(np.float32)


# cf_fit
CF_IMPLEMENTATION_SHA = (
    "b6971d02f69389fbb5b05dc7dd04958defb3e363c1f98413f0a8cc3f9034e6ee"
)


def population_models(data, cut, cache_dir, cfg):
    from implicit.cpu.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import BM25Recommender

    path = Path(data) / "likes.parquet"
    st = path.stat()
    identity = [str(path.resolve()), st.st_size, st.st_mtime_ns]
    verified_path = Path(data) / "portable_inputs_verified.json"
    if verified_path.exists():
        verified = json.loads(verified_path.read_text())
        if verified["stats"].get("likes.parquet") == identity:
            identity = {"sha256": verified["sha256"]["likes.parquet"]}
    sig = dict(
        version="population-cf-v1",
        source=identity,
        cut=cut,
        config=asdict(cfg),
        libraries={
            p: importlib.metadata.version(p)
            for p in ["implicit", "polars", "numpy", "scipy"]
        },
        code=CF_IMPLEMENTATION_SHA,
    )
    key = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:24]
    folder = Path(cache_dir) / ("cf_" + key)
    folder.mkdir(parents=True, exist_ok=True)
    ap = folder / "als.npz"
    kp = folder / "bm25.npz"
    if not (ap.exists() and kp.exists() and (folder / "complete.json").exists()):
        corpus_path = folder / "corpus.parquet"
        if not corpus_path.exists():
            print(
                f"Building collaborative population: cutoff={cut}, past {cfg.history_days} days",
                flush=True,
            )
            temporary_corpus = folder / "corpus.partial.parquet"
            write_population_corpus(
                path, cut, cfg.history_days, cfg.user_mod, cfg.seed, temporary_corpus
            )
            temporary_corpus.replace(corpus_path)
            pairs = pl.read_parquet(corpus_path, parallel="none", low_memory=True)
        else:
            pairs = pl.read_parquet(corpus_path)
        if pairs.height < 2:
            raise ValueError("Insufficient past population for collaborative sources")
        users = np.unique(pairs["uid"].to_numpy())
        items = np.unique(pairs["item_id"].to_numpy())
        matrix, _ = csr(pairs, users, items, cut)
        print(
            f"CF TRAIN users={len(users):,}; tracks={len(items):,}; pairs={matrix.nnz:,}",
            flush=True,
        )
        with threadpool_limits(limits=1, user_api="blas"):
            if not (ap.exists() and (folder / "als_items.npy").exists()):
                als = AlternatingLeastSquares(
                    factors=cfg.factors,
                    iterations=cfg.iterations,
                    regularization=0.1,
                    alpha=1.0,
                    random_state=cfg.seed,
                    num_threads=cfg.threads,
                )
                als.fit(matrix, show_progress=True)
                als.save(folder / "als.partial.npz")
                (folder / "als.partial.npz").replace(ap)
                np.save(folder / "als_items.npy", items)
                del als
            if not (kp.exists() and (folder / "knn_items.npy").exists()):
                top = (
                    pairs.group_by("item_id")
                    .len()
                    .sort(["len", "item_id"], descending=[True, False])
                    .head(cfg.knn_items)["item_id"]
                    .to_numpy()
                )
                top = np.sort(top)
                recent = (
                    pairs.sort(
                        ["uid", "timestamp", "item_id"], descending=[False, True, False]
                    )
                    .group_by("uid", maintain_order=True)
                    .head(100)
                )
                k_matrix, _ = csr(recent, users, top, cut)
                knn = BM25Recommender(
                    K=min(cfg.knn_neighbors, len(top)), num_threads=cfg.threads
                )
                knn.fit(k_matrix, show_progress=True)
                knn.save(folder / "bm25.partial.npz")
                (folder / "bm25.partial.npz").replace(kp)
                np.save(folder / "knn_items.npy", top)
                del knn, k_matrix, recent
        (folder / "complete.json").write_text(
            json.dumps(
                dict(
                    signature=sig, users=len(users), tracks=len(items), pairs=matrix.nnz
                )
            )
        )
        del pairs, matrix
        gc.collect()
    als = AlternatingLeastSquares.load(ap)
    als.num_threads = cfg.threads
    knn = BM25Recommender.load(kp)
    knn.num_threads = cfg.threads
    return (
        als,
        np.load(folder / "als_items.npy"),
        knn,
        np.load(folder / "knn_items.npy"),
        json.loads((folder / "complete.json").read_text()),
    )


# cf_sources
class Sources:
    def __init__(self, data, cut, cache_dir, config, users, history, listen_pairs):
        self.cfg = config
        self.users = users
        self.ui = {int(user_id): batch_row for batch_row, user_id in enumerate(users)}
        als_model, self.items, bm25_model, bm25_item_ids, self.info = population_models(
            data, cut, cache_dir, config
        )
        self.vectors = np.asarray(als_model.item_factors, dtype=np.float32)
        self.norms = np.linalg.norm(self.vectors, axis=1)
        like_pairs = (
            history.filter(pl.col("timestamp") >= cut - config.history_days * base.DAY)
            .group_by("uid", "item_id")
            .agg(pl.col("timestamp").max())
        )
        frames = list(listen_pairs.values())
        if frames:
            heard = (
                pl.concat(frames)
                .filter(pl.col("qualified") > 0)
                .select(
                    "uid",
                    "item_id",
                    pl.col("last_qualified_ts").alias("timestamp"),
                    pl.col("qualified").alias("count"),
                )
            )
            heard = (
                heard.sort(
                    ["uid", "timestamp", "item_id"], descending=[False, True, False]
                )
                .group_by("uid", maintain_order=True)
                .head(400)
            )
        else:
            heard = pl.DataFrame(
                schema={
                    "uid": pl.Int64,
                    "item_id": pl.Int64,
                    "timestamp": pl.Int64,
                    "count": pl.Int64,
                }
            )
        self.seeds = {}
        for profile_kind, profile_pairs in [("like", like_pairs), ("listen", heard)]:
            recent = (
                profile_pairs.sort(
                    ["uid", "timestamp", "item_id"], descending=[False, True, False]
                )
                .group_by("uid", maintain_order=True)
                .head(20)
            )
            self.seeds[profile_kind] = {
                int(user_id): user_history["item_id"].to_numpy()
                for (user_id,), user_history in recent.group_by("uid")
            }
        self.factors = {}
        self.coverage = {}
        self.ids = {}
        self.scores = {}
        torch = None
        gpu = False
        if config.retrieval_device != "cpu":
            try:
                import torch

                gpu = torch.cuda.is_available()
            except ImportError:
                pass
            if config.retrieval_device == "gpu" and (not gpu):
                raise RuntimeError(
                    "GPU retrieval requested, but no CUDA/ROCm PyTorch device is available"
                )
        self.info["retrieval_backend"] = "torch_gpu" if gpu else "numpy_cpu"
        print(
            f"CF RETRIEVAL: {self.info['retrieval_backend']}; ALS/BM25 fitting: CPU",
            flush=True,
        )
        gpu_item_factors = torch.as_tensor(self.vectors, device="cuda") if gpu else None
        for profile_kind, profile_pairs in [("like", like_pairs), ("listen", heard)]:
            profile_matrix, coverage = csr(
                profile_pairs, users, self.items, cut, listens=profile_kind == "listen"
            )
            factors = np.zeros((len(users), config.factors), np.float32)
            active = np.flatnonzero(np.diff(profile_matrix.indptr) > 0)
            with threadpool_limits(limits=1, user_api="blas"):
                for start in range(0, len(active), 128):
                    profile_rows = active[start : start + 128]
                    factors[profile_rows] = als_model.recalculate_user(
                        profile_rows, profile_matrix[profile_rows]
                    )
            self.factors[profile_kind] = factors
            self.coverage[profile_kind] = coverage
            candidate_count = min(config.source_budget, len(self.items))
            ids = np.full((len(users), candidate_count), -1, np.int64)
            scores = np.zeros((len(users), candidate_count), np.float32)
            for start in range(0, len(active), 32):
                profile_rows = active[start : start + 32]
                if gpu:
                    with torch.no_grad():
                        values, item_columns = torch.topk(
                            torch.as_tensor(factors[profile_rows], device="cuda")
                            @ gpu_item_factors.T,
                            k=candidate_count,
                            dim=1,
                        )
                        top_scores = values.cpu().numpy()
                        item_columns = item_columns.cpu().numpy()
                else:
                    with threadpool_limits(limits=1, user_api="blas"):
                        values = factors[profile_rows] @ self.vectors.T
                    item_columns = np.argpartition(values, -candidate_count, axis=1)[
                        :, -candidate_count:
                    ]
                    top_scores = np.take_along_axis(values, item_columns, axis=1)
                for batch_row, user_row in enumerate(profile_rows):
                    order = np.lexsort(
                        (self.items[item_columns[batch_row]], -top_scores[batch_row])
                    )
                    ids[user_row] = self.items[item_columns[batch_row][order]]
                    scores[user_row] = top_scores[batch_row][order]
            self.ids["als_" + profile_kind + "s"] = ids
            self.scores["als_" + profile_kind + "s"] = scores
            bm25_matrix, _ = csr(
                profile_pairs,
                users,
                bm25_item_ids,
                cut,
                listens=profile_kind == "listen",
            )
            candidate_count = min(config.source_budget, len(bm25_item_ids))
            neighbor_ids = np.full((len(users), candidate_count), -1, np.int64)
            neighbor_scores = np.zeros((len(users), candidate_count), np.float32)
            for user_row in np.flatnonzero(np.diff(bm25_matrix.indptr) > 0):
                item_columns, top_scores = bm25_model.recommend(
                    int(user_row),
                    bm25_matrix[user_row],
                    N=candidate_count,
                    filter_already_liked_items=False,
                )
                item_columns = np.asarray(item_columns)
                top_scores = np.asarray(top_scores)
                valid_items = (
                    (item_columns >= 0)
                    & (item_columns < len(bm25_item_ids))
                    & np.isfinite(top_scores)
                    & (top_scores > 0)
                )
                item_columns = item_columns[valid_items]
                top_scores = top_scores[valid_items]
                order = np.lexsort((bm25_item_ids[item_columns], -top_scores))
                neighbor_count = len(item_columns)
                neighbor_ids[user_row, :neighbor_count] = bm25_item_ids[
                    item_columns[order]
                ]
                neighbor_scores[user_row, :neighbor_count] = top_scores[order]
            self.ids["bm25_" + profile_kind + "s"] = neighbor_ids
            self.scores["bm25_" + profile_kind + "s"] = neighbor_scores
        if gpu_item_factors is not None:
            del gpu_item_factors
            torch.cuda.empty_cache()
        del als_model, bm25_model
        gc.collect()

    def lists(self, user_id):
        user_row = self.ui[user_id]
        return {
            source_name: [
                int(item_id)
                for item_id in self.ids[source_name][user_row]
                if item_id >= 0
            ]
            for source_name in SOURCE_NAMES
        }

    def features(self, user_id, candidate_ids):
        user_row = self.ui[user_id]
        item_positions = np.searchsorted(self.items, candidate_ids)
        known_items = item_positions < len(self.items)
        known_items &= (
            self.items[np.minimum(item_positions, len(self.items) - 1)] == candidate_ids
        )
        item_vectors = np.zeros((len(candidate_ids), self.cfg.factors), np.float32)
        item_vectors[known_items] = self.vectors[item_positions[known_items]]
        vector_norms = np.linalg.norm(item_vectors, axis=1)
        normalized_items = item_vectors / np.maximum(vector_norms[:, None], 1e-12)
        like_profile = self.factors["like"][user_row]
        listen_profile = self.factors["listen"][user_row]
        like_norm = np.linalg.norm(like_profile)
        listen_norm = np.linalg.norm(listen_profile)
        source_ranks = {
            name: {
                int(item_id): (float(score), 1 / (rank + 1))
                for rank, (item_id, score) in enumerate(
                    zip(self.ids[name][user_row], self.scores[name][user_row])
                )
                if item_id >= 0
            }
            for name in SOURCE_NAMES
        }
        source_features = np.array(
            [
                [
                    feature_value
                    for name in SOURCE_NAMES
                    for feature_value in source_ranks[name].get(
                        int(item_id), (0.0, 0.0)
                    )
                ]
                for item_id in candidate_ids
            ],
            np.float32,
        )
        scalar_features = np.column_stack(
            [
                item_vectors @ like_profile,
                item_vectors @ listen_profile,
                item_vectors
                @ like_profile
                / np.maximum(vector_norms * like_norm, 1e-12),
                item_vectors
                @ listen_profile
                / np.maximum(vector_norms * listen_norm, 1e-12),
                vector_norms,
                np.full(len(candidate_ids), like_norm),
                np.full(len(candidate_ids), listen_norm),
                known_items.astype(float),
                np.full(len(candidate_ids), self.coverage["like"][user_row]),
                np.full(len(candidate_ids), self.coverage["listen"][user_row]),
            ]
        )
        similarity_features = []
        for profile_kind in ["like", "listen"]:
            recent_track_ids = self.seeds[profile_kind].get(
                user_id, np.array([], dtype=np.int64)
            )
            recent_item_positions = np.searchsorted(self.items, recent_track_ids)
            supported_seeds = recent_item_positions < len(self.items)
            supported_seeds &= (
                self.items[np.minimum(recent_item_positions, len(self.items) - 1)]
                == recent_track_ids
            )
            recent_item_positions = recent_item_positions[supported_seeds]
            recent_item_positions = recent_item_positions[
                self.norms[recent_item_positions] > 1e-12
            ]
            recent_features = np.full(
                (len(candidate_ids), len(SIM_NAMES)), -2.0, np.float32
            )
            if len(recent_item_positions):
                similarities = np.clip(
                    normalized_items
                    @ (
                        self.vectors[recent_item_positions]
                        / self.norms[recent_item_positions, None]
                    ).T,
                    -1,
                    1,
                )
                recent_features = np.column_stack(
                    [
                        similarities[:, 0],
                        similarities[:, :5].max(axis=1),
                        similarities[:, :5].mean(axis=1),
                        similarities.max(axis=1),
                        similarities.mean(axis=1),
                    ]
                )
                recent_features[~known_items] = -2.0
            similarity_features.append(recent_features)
        return np.concatenate(
            [source_features, scalar_features, *similarity_features], axis=1
        ).astype(np.float32)


# cf_config
cf_cfg = Config(
    user_mod=CF_MOD,
    factors=ALS_FACTORS,
    iterations=ALS_ITERATIONS,
    knn_items=KNN_ITEMS,
    knn_neighbors=KNN_NEIGHBORS,
    threads=THREADS,
    seed=SEED,
    retrieval_device=TORCH_DEVICE,
)
save_table(
    "feature_groups",
    pd.DataFrame(
        [
            {
                "group": "likes",
                "count": len(old.FEATURES),
                "features": ", ".join(old.FEATURES),
            },
            {
                "group": "listens and feedback",
                "count": len(EXTRA),
                "features": ", ".join(EXTRA),
            },
            {
                "group": "collaborative",
                "count": len(CF_FEATURES),
                "features": ", ".join(CF_FEATURES),
            },
        ]
    ),
)


# pair_sources
def source_rows(
    ctx,
    events,
    cut,
    users,
    budget,
    cache_dir,
    artist_catalog,
    lookback,
    cf_cfg=None,
    provenance=None,
):
    likes = LikeFeatures(ctx, cut, users, cache_dir, artist_catalog)
    listens = ListenFeatures(events, ctx, cut, users, likes.history, lookback, 1000)
    cf = None
    if cf_cfg is not None:
        log(f"BUILD collaborative sources cutoff={cut}")
        cf = Sources(
            ctx.data, cut, cache_dir, cf_cfg, users, likes.history, listens.pairs
        )
        if provenance is not None:
            provenance["collaborative"] = cf.info
    for j, u in enumerate(users):
        sources = likes.user_sources(u)
        like_pool = sources[2][:1000]
        listen_pool = core.fuse([v.get(u, []) for v in listens.lists.values()], 1000)
        core_pool = core.fuse([like_pool, listen_pool], min(budget, 2000))
        if cf is not None:
            channels = cf.lists(u)
            als_family = core.fuse(
                [channels["als_likes"], channels["als_listens"]], 1000
            )
            knn_family = core.fuse(
                [channels["bm25_likes"], channels["bm25_listens"]], 1000
            )
            cf_family = core.fuse([als_family, knn_family], 2000)
            order = core.fuse(
                [like_pool, listen_pool, cf_family], len(core_pool) + len(cf_family)
            )
            keep = set(core_pool)
            for item in order:
                if len(keep) >= budget:
                    break
                keep.add(item)
            ids = np.asarray([i for i in order if i in keep], np.int64)
        else:
            ids = np.asarray(core_pool, np.int64)
        if len(ids) < 100:
            raise ValueError(f"Insufficient historical candidates for uid={u}")
        x = np.concatenate(
            [
                likes.features(u, ids, sources),
                listens.features(
                    u, ids, dict(like=set(like_pool), listen=set(listen_pool))
                ),
            ],
            axis=1,
        )
        if cf is not None:
            x = np.concatenate([x, cf.features(u, ids)], axis=1)
        feature_names = FEATURES + CF_FEATURES if cf is not None else FEATURES
        validate_feature_matrix(u, ids, x, feature_names)
        yield u, ids, x, like_pool
        if (j + 1) % 500 == 0:
            log(f"FEATURES cutoff={cut}: {j + 1}/{len(users)} users")


def sample_indices(y, negatives, seed, cut, uid):
    """All retrieved positives + hard/random negatives; weights correct Logloss sampling."""
    if not negatives:
        return np.arange(len(y)), np.full(len(y), 1 / max(1, len(y)), np.float32)
    positive = np.flatnonzero(y)
    negative = np.flatnonzero(y == 0)
    hard = negative[: negatives // 2]
    rest = negative[len(hard) :]
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(cut), int(uid)]))
    chosen = (
        rng.choice(rest, min(negatives - len(hard), len(rest)), replace=False)
        if len(rest)
        else np.array([], np.int64)
    )
    indices = np.concatenate([positive, hard, chosen])
    weight = np.concatenate(
        [
            np.ones(len(positive) + len(hard)),
            np.full(len(chosen), len(rest) / max(1, len(chosen))),
        ]
    )
    order = np.argsort(indices)
    indices = indices[order]
    weight = weight[order]
    return indices, (weight / max(1.0, weight.sum())).astype(np.float32)


# pair_functions
def build_pairs(
    ctx,
    events,
    cut,
    label_end,
    users,
    path,
    budget,
    cache_dir,
    artist_catalog,
    lookback,
    cf_cfg=None,
    negatives=0,
):
    log(f"BUILD cutoff={cut}, labels=[{cut},{label_end}), users={len(users)}")
    target = ctx.likes.filter(
        pl.col("uid").is_in(users)
        & (pl.col("timestamp") >= cut)
        & (pl.col("timestamp") < label_end)
    )
    target = (
        target.sort(["uid", "timestamp", "item_id"])
        .group_by("uid", maintain_order=True)
        .head(100)
    )
    truth = base.user_lists(target)
    complete = set(
        target.group_by("uid").len().filter(pl.col("len") == 100)["uid"].to_list()
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {}
    names = FEATURES
    if cf_cfg is not None:
        names = FEATURES + CF_FEATURES
    writer = None
    chunks = []
    count = 0
    retrieval_count = 0
    baseline = {}
    like_baseline = {}
    cover = []
    full_cover = []
    try:
        for u, ids, x, like_pool in source_rows(
            ctx,
            events,
            cut,
            users,
            budget,
            cache_dir,
            artist_catalog,
            lookback,
            cf_cfg,
            provenance,
        ):
            true = set(truth.get(u, []))
            y = np.asarray([int(i in true) for i in ids], np.int8)
            retrieval_count += len(ids)
            baseline[u] = ids[:100].tolist()
            like_baseline[u] = like_pool[:100]
            if true:
                cover.append(float(y.sum() / len(true)))
                if u in complete:
                    full_cover.append(float(y.sum() / len(true)))
            selected, weights = sample_indices(y, negatives, ctx.seed, cut, u)
            rank = np.arange(1, len(ids) + 1, dtype=np.int32)[selected]
            ids = ids[selected]
            x = x[selected]
            y = y[selected]
            columns = {name: pa.array(x[:, j]) for j, name in enumerate(names)}
            columns.update(
                uid=pa.array(np.full(len(ids), u, np.int64)),
                item_id=pa.array(ids),
                candidate_rank=pa.array(rank),
                target=pa.array(y),
                n_true=pa.array(np.full(len(ids), len(true), np.int16)),
                sample_weight=pa.array(weights),
            )
            chunks.append(pa.table(columns))
            count += len(ids)
            if len(chunks) >= 16:
                table = pa.concat_tables(chunks)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                chunks = []
        if chunks:
            table = pa.concat_tables(chunks)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    meta = dict(
        cutoff=cut,
        label_end_exclusive=label_end,
        users=users,
        truth=truth,
        complete=sorted(complete),
        candidate_pairs=count,
        retrieval_pairs=retrieval_count,
        candidate_recall_active=float(np.mean(cover)) if cover else None,
        candidate_recall_full100=float(np.mean(full_cover)) if full_cover else None,
        rrf=old.metric(baseline, truth, users, complete),
        likes_rrf=old.metric(like_baseline, truth, users, complete),
    )
    meta.update(provenance)
    path.with_suffix(".json").write_text(json.dumps(meta))
    log(
        f"PAIRS ready rows={count:,}, candidate recall={meta['candidate_recall_active']}"
    )
    return meta


def predict(model, path, features, trees):
    best = {}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=65536, columns=["uid", "item_id"] + features
    ):
        f = pl.from_arrow(batch)
        scores = (
            model.predict(
                f.select(features).to_numpy(),
                ntree_end=trees,
                prediction_type="RawFormulaVal",
            )
            if isinstance(model, CatBoostClassifier)
            else model.predict(f.select(features).to_numpy(), ntree_end=trees)
        )
        f = f.select("uid", "item_id").with_columns(pl.Series("score", scores))
        f = (
            f.sort(["uid", "score", "item_id"], descending=[False, True, False])
            .group_by("uid", maintain_order=True)
            .head(100)
        )
        for (u,), g in f.group_by("uid"):
            values = best.get(int(u), []) + g.select("item_id", "score").rows()
            best[int(u)] = sorted(values, key=lambda p: (-p[1], p[0]))[:100]
    return {u: [int(i) for i, s in values] for u, values in best.items()}


def evaluate(model, path, meta, features, trees):
    truth = {int(k): v for k, v in meta["truth"].items()}
    pred = predict(model, path, features, trees)
    return old.metric(pred, truth, meta["users"], set(meta["complete"])), pred


# long_config
V7_CODE_SHA = "fed79dc352c2aa78d663711e5d9be40faa100f06dbae638564301663c8035e72"
"""Chronological long-horizon experiment; run in the prepared v6 notebook namespace."""
import zipfile
from dataclasses import replace
from html import escape

V7_VERSION = "long-horizon-v7"
V7_RATE_FEATURES = ["log_user_like_rate", "user_recent_activity_ratio"]
LONG_RUN_LABEL = globals().get("LONG_RUN_LABEL", "long_horizon_v7")
LONG_TRAIN_USERS = globals().get("LONG_TRAIN_USERS", 5000)
LONG_EVAL_USERS = globals().get("LONG_EVAL_USERS", 5000)
LONG_SOURCE_BUDGET = globals().get("LONG_SOURCE_BUDGET", 1000)
LONG_CANDIDATE_BUDGET = globals().get("LONG_CANDIDATE_BUDGET", 3000)
LONG_NEGATIVES = globals().get("LONG_NEGATIVES", 300)
LONG_PRIMARY_TREES = globals().get("LONG_PRIMARY_TREES", 600)
LONG_TREE_PROBES = globals().get("LONG_TREE_PROBES", [600, 1200, 2000])
LONG_TRAIN_DAYS = globals().get("LONG_TRAIN_DAYS", 210)
LONG_EVAL_DAYS = globals().get("LONG_EVAL_DAYS", 120)
LONG_DOWNLOAD_RESULTS = globals().get("LONG_DOWNLOAD_RESULTS", True)


# long_fast_collaborative_features
def fast_collaborative_features(self, user_id, candidate_ids):
    """Same 28 features, with vectorized source lookup and reused dot products."""
    user_row = self.ui[user_id]
    positions = np.searchsorted(self.items, candidate_ids)
    known = positions < len(self.items)
    known &= self.items[np.minimum(positions, len(self.items) - 1)] == candidate_ids
    item_vectors = np.zeros((len(candidate_ids), self.cfg.factors), np.float32)
    item_vectors[known] = self.vectors[positions[known]]
    item_norms = np.zeros(len(candidate_ids), np.float32)
    item_norms[known] = self.norms[positions[known]]
    normalized = item_vectors / np.maximum(item_norms[:, None], 1e-12)
    like_profile = self.factors["like"][user_row]
    listen_profile = self.factors["listen"][user_row]
    like_norm = np.linalg.norm(like_profile)
    listen_norm = np.linalg.norm(listen_profile)
    like_dot = item_vectors @ like_profile
    listen_dot = item_vectors @ listen_profile
    source_features = np.zeros((len(candidate_ids), 2 * len(SOURCE_NAMES)), np.float32)
    for source_index, source_name in enumerate(SOURCE_NAMES):
        source_ids = self.ids[source_name][user_row]
        valid_positions = np.flatnonzero(source_ids >= 0)
        _, candidate_rows, source_rows = np.intersect1d(
            candidate_ids,
            source_ids[valid_positions],
            assume_unique=True,
            return_indices=True,
        )
        bank_rows = valid_positions[source_rows]
        source_features[candidate_rows, 2 * source_index] = self.scores[source_name][
            user_row, bank_rows
        ]
        source_features[candidate_rows, 2 * source_index + 1] = 1.0 / (bank_rows + 1)
    scalars = np.column_stack(
        [
            like_dot,
            listen_dot,
            like_dot / np.maximum(item_norms * like_norm, 1e-12),
            listen_dot / np.maximum(item_norms * listen_norm, 1e-12),
            item_norms,
            np.full(len(candidate_ids), like_norm),
            np.full(len(candidate_ids), listen_norm),
            known.astype(np.float32),
            np.full(len(candidate_ids), self.coverage["like"][user_row]),
            np.full(len(candidate_ids), self.coverage["listen"][user_row]),
        ]
    )
    recent_features = []
    for profile_kind in ["like", "listen"]:
        seeds = self.seeds[profile_kind].get(user_id, np.array([], np.int64))
        seed_positions = np.searchsorted(self.items, seeds)
        valid = seed_positions < len(self.items)
        valid &= self.items[np.minimum(seed_positions, len(self.items) - 1)] == seeds
        seed_positions = seed_positions[valid]
        seed_positions = seed_positions[self.norms[seed_positions] > 1e-12]
        values = np.full((len(candidate_ids), len(SIM_NAMES)), -2.0, np.float32)
        if len(seed_positions):
            similarities = np.clip(
                normalized
                @ (self.vectors[seed_positions] / self.norms[seed_positions, None]).T,
                -1,
                1,
            )
            values = np.column_stack(
                [
                    similarities[:, 0],
                    similarities[:, :5].max(axis=1),
                    similarities[:, :5].mean(axis=1),
                    similarities.max(axis=1),
                    similarities.mean(axis=1),
                ]
            )
            values[~known] = -2.0
        recent_features.append(values)
    return np.concatenate([source_features, scalars, *recent_features], axis=1).astype(
        np.float32
    )


# long_v7_candidate_views
def v7_candidate_views(like_pool, listen_pool, channels, budget, source_budget):
    """Preserve the core, then compare ALS-only and ALS+BM25 at the same cap."""
    core_pool = core.fuse([like_pool, listen_pool], min(2 * source_budget, budget))
    als_family = core.fuse(
        [channels["als_likes"], channels["als_listens"]], source_budget
    )
    bm25_family = core.fuse(
        [channels["bm25_likes"], channels["bm25_listens"]], source_budget
    )
    hybrid_family = core.fuse([als_family, bm25_family], 2 * source_budget)
    views = {"core": core_pool}
    for name, family in [("als", als_family), ("hybrid", hybrid_family)]:
        order = core.fuse(
            [like_pool, listen_pool, family], len(core_pool) + len(family)
        )
        retained = set(core_pool)
        for item_id in order:
            if len(retained) >= budget:
                break
            retained.add(item_id)
        views[name] = [item_id for item_id in order if item_id in retained]
    return views


# long_v7_fingerprint
def v7_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# research_identity
V9_CODE_SHA = "7680998bd9fbc546822902d228028b29cac34609e7ef1979ae477bd648a87f49"


# research_configuration
"""BPR, bounded EASE, Mult-VAE and SASRec candidate sources, trained strictly in the past."""
"""Official SANSA adapter and a history/metadata two-tower retriever with HNSW."""
"""Controlled 50M/500M candidate and ranking experiments; no submission side effects."""
from contextlib import contextmanager
from dataclasses import replace

V9_NEW_CHANNELS = [
    "listen_space_likes",
    "listen_space_listens",
    "cosine_knn",
    "bpr",
    "ease",
    "multvae",
    "sasrec",
    "sansa",
    "two_tower",
]
V9_EXTRA_FEATURES = [
    name + suffix for name in V9_NEW_CHANNELS for suffix in ["_score", "_rr"]
] + [
    "listen_space_like_dot",
    "listen_space_heard_dot",
    "listen_space_like_cos",
    "listen_space_heard_cos",
    "listen_space_known",
    "listen_space_item_norm",
    "artist_like_count",
    "artist_like30_count",
    "artist_like90_count",
    "artist_like_age",
    "artist_qualified_count",
    "artist_qualified_age",
]
V9_BASE_CF_FEATURES = CF_FEATURES.copy()
V9_BASE_SOURCES = Sources
V9_BASE_FAST = fast_collaborative_features
V9_BASE_VIEWS = v7_candidate_views


# research_CandidateBank
class CandidateBank:
    """Compact storage; materialize a small lookup only for the current user."""

    def __init__(self, users, budget):
        self.ids = np.full((len(users), budget), -1, np.int64)
        self.scores = np.zeros((len(users), budget), np.float32)

    def set(self, row, ids, scores):
        ids, scores = np.asarray(ids), np.asarray(scores)
        valid = (ids >= 0) & np.isfinite(scores)
        ids, scores = ids[valid], scores[valid]
        order = np.lexsort((ids, -scores))[: self.ids.shape[1]]
        self.ids[row, : len(order)] = ids[order]
        self.scores[row, : len(order)] = scores[order]

    def __getitem__(self, row):
        return {
            int(item): (float(score), 1 / (rank + 1))
            for rank, (item, score) in enumerate(zip(self.ids[row], self.scores[row]))
            if item >= 0
        }


# research_v9_torch_device
def v9_torch_device(config):
    import torch

    if config.retrieval_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if config.retrieval_device == "gpu":
        raise RuntimeError("GPU requested, but PyTorch cannot use CUDA/ROCm")
    return torch.device("cpu")


# research_v9_top_scores
def v9_top_scores(bank, row, scores, items, budget):
    count = min(budget, len(items))
    if not count:
        return
    columns = np.argpartition(scores, -count)[-count:]
    bank.set(row, items[columns], scores[columns])


# research_v9_additional_banks
def v9_additional_banks(data, cut, config, users, history):
    """Shared past-only binary matrix and ordered histories; sources have separate objectives."""
    import torch
    from implicit.cpu.bpr import BayesianPersonalizedRanking
    from scipy.linalg import cho_factor, cho_solve
    from torch import nn

    device = v9_torch_device(config)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    identity = dict(
        code=V9_CODE_SHA,
        versions=versions,
        source=parquet_source_identity(data / "likes.parquet"),
        cutoff=cut,
        seed=SEED,
        config=asdict(config),
        ease_items=EASE_ITEMS,
        bpr_iterations=BPR_ITERATIONS,
        neural_items=NEURAL_ITEMS,
        neural_users=NEURAL_USERS,
        vae_epochs=VAE_EPOCHS,
        sasrec_epochs=SASREC_EPOCHS,
        torch=torch.__version__,
        enabled=RUN_NEURAL_SOURCES,
    )
    folder = CACHE / ("additional_" + v7_fingerprint(identity)[:24])
    folder.mkdir(exist_ok=True)
    corpus = folder / "corpus.parquet"
    if not corpus.exists():
        partial = corpus.with_suffix(".partial.parquet")
        write_population_corpus(
            data / "likes.parquet", cut, 180, config.user_mod, SEED, partial
        )
        partial.replace(corpus)
    pairs = pl.read_parquet(corpus, low_memory=True, parallel="none")
    # Corpus is one latest event per pair. Sequences below retain repeated like events.
    population_users = np.unique(pairs["uid"].to_numpy())
    items = np.unique(pairs["item_id"].to_numpy())
    support = (
        pairs.group_by("item_id")
        .len()
        .sort(["len", "item_id"], descending=[True, False])
    )
    matrix, _ = csr(pairs, population_users, items, cut)
    matrix.data[:] = 1
    profile = (
        history.filter(pl.col("timestamp") >= cut - 180 * DAY)
        .group_by("uid", "item_id")
        .agg(pl.col("timestamp").max())
    )
    budget = min(NEW_SOURCE_BUDGET, config.source_budget)
    banks = {
        name: CandidateBank(users, budget)
        for name in ["bpr", "ease", "multvae", "sasrec"]
    }
    bpr_path = folder / "bpr.npz"
    if bpr_path.exists():
        bpr = BayesianPersonalizedRanking.load(bpr_path)
    else:
        bpr = BayesianPersonalizedRanking(
            factors=config.factors,
            iterations=BPR_ITERATIONS,
            learning_rate=0.05,
            regularization=0.01,
            num_threads=THREADS,
            random_state=SEED,
            verify_negative_samples=True,
        )
        with threadpool_limits(limits=1, user_api="blas"):
            bpr.fit(matrix, show_progress=True)
        bpr.save(folder / "bpr.partial.npz")
        (folder / "bpr.partial.npz").replace(bpr_path)
    user_index = {int(user): row for row, user in enumerate(population_users)}
    with threadpool_limits(limits=1, user_api="blas"):
        for row, user in enumerate(users):
            if user in user_index:
                score = bpr.item_factors @ bpr.user_factors[user_index[user]]
                v9_top_scores(banks["bpr"], row, score, items, budget)
    del bpr, matrix
    ease_items = np.sort(support.head(EASE_ITEMS)["item_id"].to_numpy())
    ease_matrix, _ = csr(pairs, population_users, ease_items, cut)
    ease_matrix.data[:] = 1
    coefficients_path = folder / "ease.npy"
    if not coefficients_path.exists():
        with threadpool_limits(limits=THREADS, user_api="blas"):
            gram = (ease_matrix.T @ ease_matrix).toarray().astype(np.float32)
            gram.flat[:: len(ease_items) + 1] += 500.0
            factor = cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
            precision = cho_solve(
                factor, np.eye(len(ease_items), dtype=np.float32), check_finite=False
            )
            coefficients = -precision / np.diag(precision)[None, :]
            np.fill_diagonal(coefficients, 0)
        temporary = folder / "ease.partial.npy"
        np.save(temporary, coefficients)
        temporary.replace(coefficients_path)
        del gram, factor, precision, coefficients
    coefficients = np.load(coefficients_path, mmap_mode="r")
    ease_profiles, _ = csr(profile, users, ease_items, cut)
    ease_profiles.data[:] = 1
    with threadpool_limits(limits=1, user_api="blas"):
        for start in range(0, len(users), 64):
            scores = ease_profiles[start : start + 64] @ coefficients
            for batch_row, score in enumerate(scores):
                row = start + batch_row
                if ease_profiles.indptr[row + 1] > ease_profiles.indptr[row]:
                    v9_top_scores(banks["ease"], row, score, ease_items, budget)
    del ease_matrix, ease_profiles, coefficients
    if not RUN_NEURAL_SOURCES:
        gc.collect()
        return banks
    neural_items = np.sort(support.head(NEURAL_ITEMS)["item_id"].to_numpy())
    rng = np.random.default_rng(SEED)
    fit_users = np.sort(
        rng.choice(
            population_users, min(NEURAL_USERS, len(population_users)), replace=False
        )
    )
    neural_pairs = pairs.filter(pl.col("uid").is_in(fit_users))
    training, _ = csr(neural_pairs, fit_users, neural_items, cut)
    training.data[:] = 1
    profiles, _ = csr(profile, users, neural_items, cut)
    profiles.data[:] = 1

    # Cap dense decoder cost with a declared past-popularity catalog; not all-catalog VAE.
    class MultVAE(nn.Module):
        def __init__(self, item_count):
            super().__init__()
            self.encoder = nn.Linear(item_count, 128)
            self.mean = nn.Linear(128, 64)
            self.log_variance = nn.Linear(128, 64)
            self.decoder = nn.Linear(64, item_count)
            self.dropout = nn.Dropout(0.5)

        def forward(self, values):
            normalized = nn.functional.normalize(values, p=2, dim=1)
            hidden = torch.tanh(self.encoder(self.dropout(normalized)))
            mean = self.mean(hidden)
            logvar = self.log_variance(hidden).clamp(-10, 10)
            latent = (
                mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
                if self.training
                else mean
            )
            return self.decoder(latent), mean, logvar

    vae = MultVAE(len(neural_items)).to(device)
    vae_path = folder / "multvae.pt"
    if vae_path.exists():
        vae.load_state_dict(
            torch.load(vae_path, map_location=device, weights_only=True)
        )
    else:
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        step = 0
        for epoch in range(VAE_EPOCHS):
            vae.train()
            loss_sum = 0.0
            for start in range(0, len(fit_users), 256):
                # A new deterministic user permutation on every epoch.
                if start == 0:
                    order = rng.permutation(len(fit_users))
                selected = order[start : start + 256]
                values = torch.as_tensor(training[selected].toarray(), device=device)
                logits, mean, logvar = vae(values)
                reconstruction = (
                    -(nn.functional.log_softmax(logits, dim=1) * values)
                    .sum(dim=1)
                    .mean()
                )
                kl = (
                    -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(dim=1).mean()
                )
                loss = reconstruction + min(0.2, step / 2000) * kl
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite MultVAE loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
                step += 1
            print(
                f"MultVAE epoch {epoch + 1}/{VAE_EPOCHS}: loss sum {loss_sum:.3f}",
                flush=True,
            )
        torch.save(vae.state_dict(), folder / "multvae.partial.pt")
        (folder / "multvae.partial.pt").replace(vae_path)
        del optimizer
    vae.eval()
    with torch.no_grad():
        for start in range(0, len(users), 128):
            values = torch.as_tensor(
                profiles[start : start + 128].toarray(), device=device
            )
            scores = vae(values)[0].cpu().numpy()
            for batch_row, score in enumerate(scores):
                row = start + batch_row
                if profiles.indptr[row + 1] > profiles.indptr[row]:
                    v9_top_scores(banks["multvae"], row, score, neural_items, budget)
    del vae, training, profiles, neural_pairs
    # Ordered raw events, not the pair-deduplicated ALS corpus.
    sequences = base.collect(
        pl.scan_parquet(data / "likes.parquet")
        .filter(
            (pl.col("timestamp") >= cut - 180 * DAY)
            & (pl.col("timestamp") < cut)
            & pl.col("uid").is_in(fit_users)
            & pl.col("item_id").is_in(neural_items)
        )
        .select(pl.col("uid", "item_id", "timestamp").cast(pl.Int64))
        .sort(["uid", "timestamp", "item_id"])
        .group_by("uid", maintain_order=True)
        .tail(51)
    )
    training_sequences = [
        group["item_id"].to_numpy()
        for _, group in sequences.group_by("uid")
        if group.height >= 2
    ]
    del sequences
    if not training_sequences:
        raise ValueError("No past sequences for SASRec")
    inputs = np.zeros((len(training_sequences), 50), np.int64)
    targets = np.zeros_like(inputs)
    for row, sequence in enumerate(training_sequences):
        encoded = np.searchsorted(neural_items, sequence) + 1
        inputs[row, : len(encoded) - 1] = encoded[:-1]
        targets[row, : len(encoded) - 1] = encoded[1:]

    class SASRec(nn.Module):
        def __init__(self, item_count):
            super().__init__()
            self.items = nn.Embedding(item_count + 1, 64, padding_idx=0)
            self.positions = nn.Embedding(50, 64)
            layer = nn.TransformerEncoderLayer(
                64,
                2,
                dim_feedforward=128,
                dropout=0.2,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(64)

        def forward(self, tokens):
            values = (
                self.items(tokens) * 8
                + self.positions(torch.arange(tokens.shape[1], device=tokens.device))[
                    None, :, :
                ]
            )
            causal = torch.triu(
                torch.ones(
                    tokens.shape[1],
                    tokens.shape[1],
                    device=tokens.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            return self.norm(
                self.blocks(values, mask=causal, src_key_padding_mask=tokens == 0)
            )

    sasrec = SASRec(len(neural_items)).to(device)
    sasrec_path = folder / "sasrec.pt"
    if sasrec_path.exists():
        sasrec.load_state_dict(
            torch.load(sasrec_path, map_location=device, weights_only=True)
        )
    else:
        optimizer = torch.optim.Adam(sasrec.parameters(), lr=0.001)
        for epoch in range(SASREC_EPOCHS):
            sasrec.train()
            order = rng.permutation(len(inputs))
            loss_sum = 0.0
            for start in range(0, len(inputs), 256):
                selected = order[start : start + 256]
                tokens = torch.as_tensor(inputs[selected], device=device)
                positive = torch.as_tensor(targets[selected], device=device)
                negative = torch.randint(
                    1, len(neural_items) + 1, positive.shape, device=device
                )
                negative = torch.where(
                    negative == positive, (negative % len(neural_items)) + 1, negative
                )
                hidden = sasrec(tokens)
                positive_logits = (hidden * sasrec.items(positive)).sum(dim=-1)
                negative_logits = (hidden * sasrec.items(negative)).sum(dim=-1)
                valid = positive > 0
                loss = (
                    nn.functional.softplus(-positive_logits[valid])
                    + nn.functional.softplus(negative_logits[valid])
                ).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite SASRec loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(sasrec.parameters(), 5)
                optimizer.step()
                loss_sum += float(loss.detach())
            print(
                f"SASRec epoch {epoch + 1}/{SASREC_EPOCHS}: loss sum {loss_sum:.3f}",
                flush=True,
            )
        torch.save(sasrec.state_dict(), folder / "sasrec.partial.pt")
        (folder / "sasrec.partial.pt").replace(sasrec_path)
        del optimizer
    profile_sequences = (
        history.filter(
            (pl.col("timestamp") >= cut - 180 * DAY)
            & pl.col("item_id").is_in(neural_items)
        )
        .sort(["uid", "timestamp", "item_id"])
        .group_by("uid", maintain_order=True)
        .tail(50)
    )
    lookup = {
        int(user): group["item_id"].to_numpy()
        for (user,), group in profile_sequences.group_by("uid")
    }
    sasrec.eval()
    with torch.no_grad():
        for start in range(0, len(users), 128):
            active = [
                (row, lookup[user])
                for row, user in enumerate(users[start : start + 128], start)
                if user in lookup
            ]
            if not active:
                continue
            tokens = np.zeros((len(active), 50), np.int64)
            lengths = []
            for row, (_, sequence) in enumerate(active):
                tokens[row, : len(sequence)] = (
                    np.searchsorted(neural_items, sequence) + 1
                )
                lengths.append(len(sequence))
            hidden = sasrec(torch.as_tensor(tokens, device=device))
            query = hidden[
                torch.arange(len(active), device=device),
                torch.as_tensor(lengths, device=device) - 1,
            ]
            scores = (query @ sasrec.items.weight[1:].T).cpu().numpy()
            for (user_row, _), score in zip(active, scores):
                v9_top_scores(banks["sasrec"], user_row, score, neural_items, budget)
    del sasrec, inputs, targets, training_sequences, pairs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    v9_json(folder / "configuration.json", identity)
    return banks


# research_v9_sansa_bank
def v9_sansa_bank(data, cut, config, users, history):
    from sansa import (
        SANSA,
        ICFGramianFactorizerConfig,
        SANSAConfig,
        UMRUnitLowerTriangleInverterConfig,
    )

    identity = dict(
        code=V9_CODE_SHA,
        versions=versions,
        source=parquet_source_identity(data / "likes.parquet"),
        cutoff=cut,
        items=SANSA_ITEMS,
        density=SANSA_DENSITY,
        config=asdict(config),
        sansa_commit=SANSA_SHA,
    )
    folder = CACHE / ("sansa_" + v7_fingerprint(identity)[:24])
    folder.mkdir(exist_ok=True)
    corpus = folder / "corpus.parquet"
    if not corpus.exists():
        partial = corpus.with_suffix(".partial.parquet")
        write_population_corpus(
            data / "likes.parquet", cut, 180, config.user_mod, SEED, partial
        )
        partial.replace(corpus)
    pairs = pl.read_parquet(corpus, low_memory=True, parallel="none")
    items = np.sort(
        pairs.group_by("item_id")
        .len()
        .sort(["len", "item_id"], descending=[True, False])
        .head(SANSA_ITEMS)["item_id"]
        .to_numpy()
    )
    model = SANSA(
        SANSAConfig(
            l2=500.0,
            weight_matrix_density=SANSA_DENSITY,
            gramian_factorizer_config=ICFGramianFactorizerConfig(),
            lower_triangle_inverter_config=UMRUnitLowerTriangleInverterConfig(
                scans=1, finetune_steps=5
            ),
        )
    )
    receipt = folder / "complete.json"
    if receipt.exists():
        model.load_weights(
            (
                sparse.load_npz(folder / "encoder.npz"),
                sparse.load_npz(folder / "decoder.npz"),
            )
        )
    else:
        matrix, _ = csr(pairs, np.unique(pairs["uid"].to_numpy()), items, cut)
        matrix.data[:] = 1
        with threadpool_limits(limits=THREADS, user_api="blas"):
            model.fit(matrix)
        for name, weights in zip(["encoder", "decoder"], model.weights):
            temporary = folder / (name + ".partial.npz")
            sparse.save_npz(temporary, weights)
            temporary.replace(folder / (name + ".npz"))
        v9_json(
            receipt,
            dict(
                identity=identity,
                items=len(items),
                weights_nnz=[int(weight.nnz) for weight in model.weights],
            ),
        )
        del matrix
    profile = (
        history.filter(pl.col("timestamp") >= cut - 180 * DAY)
        .group_by("uid", "item_id")
        .agg(pl.col("timestamp").max())
    )
    profiles, _ = csr(profile, users, items, cut)
    profiles.data[:] = 1
    bank = CandidateBank(users, min(NEW_SOURCE_BUDGET, config.source_budget))
    for start in range(0, len(users), 32):
        batch = profiles[start : start + 32]
        # SANSA's factorized operator has diagonal -1 by construction. Add the
        # input back to obtain a zero self-coefficient without banning re-likes.
        scores = (model.forward(batch) + batch).toarray()
        for row, values in enumerate(scores, start):
            if profiles.indptr[row + 1] > profiles.indptr[row]:
                v9_top_scores(bank, row, values, items, NEW_SOURCE_BUDGET)
    del model, pairs, profiles
    gc.collect()
    return bank


# research_v9_tower_bank
def v9_tower_bank(data, cut, config, users, history):
    import faiss
    import torch
    from torch import nn

    device = v9_torch_device(config)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    identity = dict(
        code=V9_CODE_SHA,
        versions=versions,
        source=parquet_source_identity(data / "likes.parquet"),
        cutoff=cut,
        config=asdict(config),
        items=NEURAL_ITEMS,
        users=NEURAL_USERS,
        epochs=TOWER_EPOCHS,
        torch=torch.__version__,
        faiss=faiss.__version__,
        artist_source=parquet_source_identity(data / "artist_item_mapping.parquet"),
    )
    folder = CACHE / ("tower_" + v7_fingerprint(identity)[:24])
    folder.mkdir(exist_ok=True)
    corpus = folder / "corpus.parquet"
    if not corpus.exists():
        partial = corpus.with_suffix(".partial.parquet")
        write_population_corpus(
            data / "likes.parquet", cut, 180, config.user_mod, SEED, partial
        )
        partial.replace(corpus)
    pairs = pl.read_parquet(corpus, low_memory=True, parallel="none")
    items = np.sort(
        pairs.group_by("item_id")
        .len()
        .sort(["len", "item_id"], descending=[True, False])
        .head(NEURAL_ITEMS)["item_id"]
        .to_numpy()
    )
    rng = np.random.default_rng(SEED)
    population = np.unique(pairs["uid"].to_numpy())
    fit_users = np.sort(
        rng.choice(population, min(NEURAL_USERS, len(population)), replace=False)
    )
    mapping = base.collect(
        pl.scan_parquet(data / "artist_item_mapping.parquet")
        .filter(pl.col("item_id").is_in(items))
        .group_by("item_id")
        .agg(pl.col("artist_id").min())
    ).with_columns(pl.col("item_id").cast(pl.Int64))
    artist_values = np.unique(mapping["artist_id"].to_numpy())
    artist_lookup = {int(item): int(artist) for item, artist in mapping.iter_rows()}
    encoded_artists = np.zeros(len(items) + 1, np.int64)
    for index, item in enumerate(items, 1):
        if int(item) in artist_lookup:
            encoded_artists[index] = (
                np.searchsorted(artist_values, artist_lookup[int(item)]) + 1
            )

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.history_embedding = nn.Embedding(len(items) + 1, 64, padding_idx=0)
            self.item_embedding = nn.Embedding(len(items) + 1, 64, padding_idx=0)
            self.artist_embedding = nn.Embedding(
                len(artist_values) + 1, 32, padding_idx=0
            )
            self.register_buffer("artists", torch.as_tensor(encoded_artists))
            self.user_mlp = nn.Sequential(
                nn.Linear(96, 128), nn.ReLU(), nn.Linear(128, 64)
            )
            self.item_mlp = nn.Sequential(
                nn.Linear(96, 128), nn.ReLU(), nn.Linear(128, 64)
            )

        def item_vectors(self, tokens):
            values = torch.cat(
                [
                    self.item_embedding(tokens),
                    self.artist_embedding(self.artists[tokens]),
                ],
                dim=-1,
            )
            return nn.functional.normalize(self.item_mlp(values), dim=-1)

        def user_vectors(self, tokens):
            mask = tokens > 0
            lengths = mask.sum(dim=1, keepdim=True)
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
            weights = torch.exp2(-(lengths - 1 - positions).clamp(min=0) / 10) * mask
            values = torch.cat(
                [
                    self.history_embedding(tokens),
                    self.artist_embedding(self.artists[tokens]),
                ],
                dim=-1,
            )
            pooled = (values * weights[:, :, None]).sum(dim=1) / weights.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
            return nn.functional.normalize(self.user_mlp(pooled), dim=-1)

    model = TwoTower().to(device)
    path = folder / "model.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    else:
        events = base.collect(
            pl.scan_parquet(data / "likes.parquet")
            .filter(
                (pl.col("timestamp") >= cut - 180 * DAY)
                & (pl.col("timestamp") < cut)
                & pl.col("uid").is_in(fit_users)
                & pl.col("item_id").is_in(items)
            )
            .select(pl.col("uid", "item_id", "timestamp").cast(pl.Int64))
            .sort(["uid", "timestamp", "item_id"])
            .group_by("uid", maintain_order=True)
            .tail(55)
        )
        examples = []
        for _, group in events.group_by("uid"):
            tokens = np.searchsorted(items, group["item_id"].to_numpy()) + 1
            for index in range(max(1, len(tokens) - 5), len(tokens)):
                examples.append(
                    (tokens[max(0, index - 50) : index], int(tokens[index]))
                )
        if not examples:
            raise ValueError("No chronological two-tower training examples")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        for epoch in range(TOWER_EPOCHS):
            model.train()
            loss_sum = 0.0
            order = rng.permutation(len(examples))
            for start in range(0, len(order), 256):
                selected = [examples[index] for index in order[start : start + 256]]
                prefixes = np.zeros((len(selected), 50), np.int64)
                targets = np.asarray([target for _, target in selected], np.int64)
                for row, (prefix, _) in enumerate(selected):
                    prefixes[row, : len(prefix)] = prefix
                tokens = torch.as_tensor(prefixes, device=device)
                positive = torch.as_tensor(targets, device=device)
                logits = (
                    model.user_vectors(tokens) @ model.item_vectors(positive).T / 0.1
                )
                duplicate = (positive[:, None] == positive[None, :]) & ~torch.eye(
                    len(positive), device=device, dtype=torch.bool
                )
                logits = logits.masked_fill(duplicate, -1e9)
                loss = nn.functional.cross_entropy(
                    logits, torch.arange(len(positive), device=device)
                )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite two-tower loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
            print(
                f"Two-tower epoch {epoch + 1}/{TOWER_EPOCHS}: loss sum {loss_sum:.3f}",
                flush=True,
            )
        torch.save(model.state_dict(), folder / "model.partial.pt")
        (folder / "model.partial.pt").replace(path)
        del examples, events, optimizer
    model.eval()
    with torch.no_grad():
        vectors = np.concatenate(
            [
                model.item_vectors(
                    torch.arange(
                        start, min(start + 4096, len(items) + 1), device=device
                    )
                )
                .cpu()
                .numpy()
                for start in range(1, len(items) + 1, 4096)
            ]
        )
    index = faiss.IndexHNSWFlat(64, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = max(128, 2 * NEW_SOURCE_BUDGET)
    # Single-thread index construction avoids scheduling-dependent HNSW topology.
    faiss.omp_set_num_threads(1)
    index.add(np.ascontiguousarray(vectors, np.float32))
    faiss.omp_set_num_threads(THREADS)
    history_ = (
        history.filter(
            (pl.col("timestamp") >= cut - 180 * DAY) & pl.col("item_id").is_in(items)
        )
        .sort(["uid", "timestamp", "item_id"])
        .group_by("uid", maintain_order=True)
        .tail(50)
    )
    lookup = {
        int(user): group["item_id"].to_numpy()
        for (user,), group in history_.group_by("uid")
    }
    bank = CandidateBank(users, min(NEW_SOURCE_BUDGET, config.source_budget))
    quality = []
    with torch.no_grad():
        for start in range(0, len(users), 128):
            active = [
                (row, lookup[user])
                for row, user in enumerate(users[start : start + 128], start)
                if user in lookup
            ]
            if not active:
                continue
            tokens = np.zeros((len(active), 50), np.int64)
            for row, (_, sequence) in enumerate(active):
                tokens[row, : len(sequence)] = np.searchsorted(items, sequence) + 1
            queries = (
                model.user_vectors(torch.as_tensor(tokens, device=device)).cpu().numpy()
            )
            count = min(NEW_SOURCE_BUDGET, config.source_budget, len(items))
            scores, columns = index.search(
                np.ascontiguousarray(queries, np.float32), count
            )
            for local, (user_row, _) in enumerate(active):
                valid = columns[local] >= 0
                bank.set(user_row, items[columns[local, valid]], scores[local, valid])
                if len(quality) < 100:
                    exact = np.argpartition(queries[local] @ vectors.T, -count)[-count:]
                    quality.append(len(set(columns[local]) & set(exact)) / count)
    v9_json(
        folder / "ann_diagnostics.json",
        dict(
            index="HNSW",
            checked_queries=len(quality),
            recall_against_exact=float(np.mean(quality)) if quality else None,
            catalog=len(items),
            item_features=["item_id", "primary_artist"],
            audio_features=False,
        ),
    )
    del model, pairs, index, vectors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bank


# research_v9_json
def v9_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


# research_v9_select_cohort
def v9_select_cohort(data, official_users, output, maximum, seed):
    """Select only by history before the earliest feature cutoff, never by labels."""
    official = pl.read_csv(official_users)["uid"].cast(pl.Int64).to_list()
    with scan_database() as connection:
        first, last = connection.execute(
            f"SELECT min(timestamp), max(timestamp) FROM read_parquet({sql_path(data / 'likes.parquet')})"
        ).fetchone()
        earliest = int(last) + 1 - 240 * DAY
        if earliest <= int(first):
            raise ValueError("Not enough history for the proposed temporal folds")
        available = (
            connection.execute(
                f"SELECT DISTINCT CAST(uid AS BIGINT) AS uid FROM read_parquet({sql_path(data / 'likes.parquet')}) "
                f"WHERE timestamp < {earliest} ORDER BY uid"
            )
            .fetchnumpy()["uid"]
            .tolist()
        )
    overlap = sorted(set(available) & set(official))
    other = sorted(set(available) - set(official))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(overlap, min(maximum, len(overlap)), replace=False).tolist()
    if len(chosen) < maximum and other:
        chosen.extend(
            rng.choice(
                other, min(maximum - len(chosen), len(other)), replace=False
            ).tolist()
        )
    if not chosen:
        raise ValueError("No users with past history")
    frame = pl.DataFrame({"uid": sorted(chosen)})
    if (
        output.exists()
        and pl.read_csv(output)["uid"].to_list() != frame["uid"].to_list()
    ):
        raise ValueError("Cohort changed: choose a new RUN_LABEL")
    frame.write_csv(output)
    info = dict(
        selection_cut=earliest,
        official_users=len(official),
        available_history_users=len(available),
        official_with_early_history=len(overlap),
        selected_users=len(chosen),
        selected_official=len(set(chosen) & set(official)),
        selection_uses_future_labels=False,
    )
    return set(official), info


# research_v9_population_listen_als
def v9_population_listen_als(data, cut, config):
    """Train a separate latent space on qualified population listens, bounded on disk."""
    from implicit.cpu.als import AlternatingLeastSquares

    source = data / "listens.parquet"
    signature = dict(
        code=V9_CODE_SHA,
        revision=DATA_REVISION,
        source=parquet_source_identity(source),
        cutoff=cut,
        days=180,
        max_per_user=LISTEN_ITEMS_PER_USER,
        catalog=LISTEN_ALS_ITEMS,
        config=asdict(config),
        libraries={
            name: versions[name] for name in ["implicit", "numpy", "scipy", "duckdb"]
        },
    )
    key = v7_fingerprint(signature)[:24]
    folder = CACHE / ("listen_population_" + key)
    folder.mkdir(exist_ok=True)
    receipt = folder / "complete.json"
    if receipt.exists():
        if json.loads(receipt.read_text())["signature"] != signature:
            raise ValueError("Listen ALS signature mismatch")
        model = AlternatingLeastSquares.load(folder / "als.npz")
        model.num_threads = THREADS
        return model, np.load(folder / "items.npy"), json.loads(receipt.read_text())
    corpus = folder / "pairs.parquet"
    if not corpus.exists():
        partial = corpus.with_suffix(".partial.parquet")
        partial.unlink(missing_ok=True)
        with scan_database() as connection:
            query = f"""
                WITH pairs AS (
                    SELECT CAST(uid AS BIGINT) AS uid, CAST(item_id AS BIGINT) AS item_id,
                           CAST(max(timestamp) AS BIGINT) AS timestamp,
                           CAST(count(*) AS BIGINT) AS count
                    FROM read_parquet({sql_path(source)})
                    WHERE timestamp >= {cut - 180 * DAY} AND timestamp < {cut}
                      AND played_ratio_pct >= 50 AND track_length_seconds > 0
                    GROUP BY uid, item_id
                )
                SELECT * FROM pairs
                QUALIFY row_number() OVER (PARTITION BY uid ORDER BY timestamp DESC, item_id) <= {int(LISTEN_ITEMS_PER_USER)}
            """
            connection.execute(
                f"COPY ({query}) TO {sql_path(partial)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
            )
        partial.replace(corpus)
    pairs = pl.read_parquet(corpus, low_memory=True, parallel="none")
    if pairs.is_empty():
        raise ValueError("No qualified population listens before cutoff")
    items = np.sort(
        pairs.group_by("item_id")
        .len()
        .sort(["len", "item_id"], descending=[True, False])
        .head(LISTEN_ALS_ITEMS)["item_id"]
        .to_numpy()
    )
    users = np.unique(pairs["uid"].to_numpy())
    matrix, coverage = csr(pairs, users, items, cut, listens=True)
    info = dict(
        signature=signature,
        users=len(users),
        items=len(items),
        pairs=int(matrix.nnz),
        mean_profile_coverage=float(coverage.mean()),
        qualified_min_ratio=50,
    )
    print(
        "POPULATION LISTEN ALS",
        {name: value for name, value in info.items() if name != "signature"},
        flush=True,
    )
    with threadpool_limits(limits=1, user_api="blas"):
        model = AlternatingLeastSquares(
            factors=config.factors,
            iterations=config.iterations,
            regularization=0.1,
            alpha=1.0,
            random_state=SEED,
            num_threads=THREADS,
        )
        model.fit(matrix, show_progress=True)
    model.save(folder / "als.partial.npz")
    (folder / "als.partial.npz").replace(folder / "als.npz")
    np.save(folder / "items.npy", items)
    v9_json(receipt, info)
    del pairs, matrix
    gc.collect()
    return model, items, info


# research_v9_population_cosine
def v9_population_cosine(data, cut, config):
    from implicit.nearest_neighbours import CosineRecommender

    signature = dict(
        code=V9_CODE_SHA,
        source=parquet_source_identity(data / "likes.parquet"),
        cutoff=cut,
        config=asdict(config),
        items=COSINE_ITEMS,
        neighbors=config.knn_neighbors,
    )
    folder = CACHE / ("cosine_" + v7_fingerprint(signature)[:24])
    folder.mkdir(exist_ok=True)
    receipt = folder / "complete.json"
    if receipt.exists():
        model = CosineRecommender.load(folder / "knn.npz")
        model.num_threads = THREADS
        return model, np.load(folder / "items.npy")
    corpus = folder / "pairs.parquet"
    if not corpus.exists():
        partial = corpus.with_suffix(".partial.parquet")
        write_population_corpus(
            data / "likes.parquet", cut, 180, config.user_mod, SEED, partial
        )
        partial.replace(corpus)
    pairs = pl.read_parquet(corpus, low_memory=True, parallel="none")
    items = np.sort(
        pairs.group_by("item_id")
        .len()
        .sort(["len", "item_id"], descending=[True, False])
        .head(COSINE_ITEMS)["item_id"]
        .to_numpy()
    )
    recent = (
        pairs.sort(["uid", "timestamp", "item_id"], descending=[False, True, False])
        .group_by("uid", maintain_order=True)
        .head(100)
    )
    matrix, _ = csr(recent, np.unique(pairs["uid"].to_numpy()), items, cut)
    matrix.data[:] = 1.0
    model = CosineRecommender(
        K=min(config.knn_neighbors, len(items)), num_threads=THREADS
    )
    model.fit(matrix, show_progress=True)
    model.similarity.setdiag(0)
    model.similarity.eliminate_zeros()
    model.save(folder / "knn.partial.npz")
    (folder / "knn.partial.npz").replace(folder / "knn.npz")
    np.save(folder / "items.npy", items)
    v9_json(receipt, signature)
    del pairs, recent, matrix
    gc.collect()
    return model, items


# research_v9_als_recommend
def v9_als_recommend(model, items, profiles, users, cut, config, listens):
    matrix, _ = csr(profiles, users, items, cut, listens=listens)
    active = np.flatnonzero(np.diff(matrix.indptr) > 0)
    factors = np.zeros((len(users), config.factors), np.float32)
    with threadpool_limits(limits=1, user_api="blas"):
        for start in range(0, len(active), 128):
            rows = active[start : start + 128]
            factors[rows] = model.recalculate_user(rows, matrix[rows])
    vectors = np.asarray(model.item_factors, np.float32)
    banks = CandidateBank(users, min(NEW_SOURCE_BUDGET, config.source_budget))
    count = min(NEW_SOURCE_BUDGET, config.source_budget, len(items))
    use_gpu = False
    torch = None
    if config.retrieval_device != "cpu":
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            pass
    if config.retrieval_device == "gpu" and not use_gpu:
        raise RuntimeError("Requested GPU retrieval is unavailable")
    device_vectors = torch.as_tensor(vectors, device="cuda") if use_gpu else None
    with threadpool_limits(limits=1, user_api="blas"):
        for start in range(0, len(active), 32):
            rows = active[start : start + 32]
            if use_gpu:
                with torch.no_grad():
                    values, columns = torch.topk(
                        torch.as_tensor(factors[rows], device="cuda")
                        @ device_vectors.T,
                        count,
                        dim=1,
                    )
                    scores, columns = values.cpu().numpy(), columns.cpu().numpy()
            else:
                values = factors[rows] @ vectors.T
                columns = np.argpartition(values, -count, axis=1)[:, -count:]
                scores = np.take_along_axis(values, columns, axis=1)
            for batch_row, user_row in enumerate(rows):
                order = np.lexsort((items[columns[batch_row]], -scores[batch_row]))
                banks.set(
                    user_row, items[columns[batch_row][order]], scores[batch_row][order]
                )
    if device_vectors is not None:
        del device_vectors
        torch.cuda.empty_cache()
    return factors, banks


# research_ResearchSources
class ResearchSources(V9_BASE_SOURCES):
    """Original sources plus listen-trained ALS, cosine kNN and artist affinities."""

    def __init__(self, data, cut, cache_dir, config, users, history, listen_pairs):
        super().__init__(data, cut, cache_dir, config, users, history, listen_pairs)
        model, self.listen_items, info = v9_population_listen_als(data, cut, config)
        self.info["population_listen_als"] = info
        self.listen_vectors = np.asarray(model.item_factors, np.float32)
        self.listen_norms = np.linalg.norm(self.listen_vectors, axis=1)
        liked = (
            history.filter(pl.col("timestamp") >= cut - 180 * DAY)
            .group_by("uid", "item_id")
            .agg(pl.col("timestamp").max())
        )
        frames = list(listen_pairs.values())
        heard = (
            pl.concat(frames)
            .filter(pl.col("qualified") > 0)
            .select(
                "uid",
                "item_id",
                pl.col("last_qualified_ts").alias("timestamp"),
                pl.col("qualified").alias("count"),
            )
            if frames
            else pl.DataFrame(
                schema={
                    "uid": pl.Int64,
                    "item_id": pl.Int64,
                    "timestamp": pl.Int64,
                    "count": pl.Int64,
                }
            )
        )
        self.listen_profiles = {}
        self.new_banks = {}
        for name, pairs, is_listen in [
            ("likes", liked, False),
            ("listens", heard, True),
        ]:
            factors, banks = v9_als_recommend(
                model, self.listen_items, pairs, users, cut, config, is_listen
            )
            self.listen_profiles[name] = factors
            self.new_banks["listen_space_" + name] = banks
        del model
        cosine, cosine_items = v9_population_cosine(data, cut, config)
        recent = (
            liked.sort(["uid", "timestamp", "item_id"], descending=[False, True, False])
            .group_by("uid", maintain_order=True)
            .head(100)
        )
        matrix, _ = csr(recent, users, cosine_items, cut)
        banks = CandidateBank(users, min(NEW_SOURCE_BUDGET, config.source_budget))
        for user_row in np.flatnonzero(np.diff(matrix.indptr) > 0):
            columns, scores = cosine.recommend(
                int(user_row),
                matrix[user_row],
                N=min(config.source_budget, len(cosine_items)),
                filter_already_liked_items=False,
            )
            valid = (
                (columns >= 0)
                & (columns < len(cosine_items))
                & np.isfinite(scores)
                & (scores > 0)
            )
            columns, scores = np.asarray(columns)[valid], np.asarray(scores)[valid]
            order = np.lexsort((cosine_items[columns], -scores))
            banks.set(user_row, cosine_items[columns[order]], scores[order])
        self.new_banks["cosine_knn"] = banks
        del cosine, matrix
        self.new_banks.update(v9_additional_banks(data, cut, config, users, history))
        self.new_banks["sansa"] = v9_sansa_bank(data, cut, config, users, history)
        self.new_banks["two_tower"] = (
            v9_tower_bank(data, cut, config, users, history)
            if RUN_NEURAL_SOURCES
            else CandidateBank(users, min(NEW_SOURCE_BUDGET, config.source_budget))
        )
        # Only keep catalog rows that can occur in this snapshot, not all 9M tracks.
        needed = set(history["item_id"].to_list())
        for source in self.new_banks.values():
            needed.update(source.ids[source.ids >= 0].tolist())
        for source in self.ids.values():
            needed.update(source[source >= 0].tolist())
        if frames:
            needed.update(pl.concat(frames)["item_id"].to_list())
        # Original popularity/artist pools can contain tracks outside CF banks.
        statistics, _ = cached_like_statistics(data, cut, cache_dir)
        needed.update(statistics["item_id"].to_list())
        mapping = base.collect(
            pl.scan_parquet(data / "artist_item_mapping.parquet")
            .select("item_id", "artist_id")
            .filter(pl.col("item_id").is_in(sorted(needed)))
            .unique()
        ).with_columns(pl.col("item_id").cast(pl.Int64))
        self.artist_mapping = {
            int(item): group["artist_id"].to_list()
            for (item,), group in mapping.group_by("item_id")
        }
        artist_history = (
            history.join(mapping, on="item_id", how="inner")
            .group_by("uid", "artist_id")
            .agg(
                pl.len().alias("count"),
                (pl.col("timestamp") >= cut - 30 * DAY).sum().alias("count30"),
                (pl.col("timestamp") >= cut - 90 * DAY).sum().alias("count90"),
                pl.col("timestamp").max().alias("last"),
            )
        )
        self.artist_likes = {
            (int(user), int(artist)): (
                int(count),
                int(count30),
                int(count90),
                (cut - int(last)) / DAY,
            )
            for user, artist, count, count30, count90, last in artist_history.iter_rows()
        }
        artist_heard = (
            heard.join(mapping, on="item_id", how="inner")
            .group_by("uid", "artist_id")
            .agg(pl.col("count").sum(), pl.col("timestamp").max())
        )
        self.artist_heard = {
            (int(user), int(artist)): (int(count), (cut - int(last)) / DAY)
            for user, artist, count, last in artist_heard.iter_rows()
        }
        gc.collect()

    def lists(self, user_id):
        banks = super().lists(user_id)
        banks.update(
            {
                name: list(source[self.ui[user_id]])
                for name, source in self.new_banks.items()
            }
        )
        return banks

    def extra_features(self, user_id, candidate_ids):
        user_row = self.ui[user_id]
        result = np.zeros((len(candidate_ids), len(V9_EXTRA_FEATURES)), np.float32)
        for channel_index, name in enumerate(V9_NEW_CHANNELS):
            bank = self.new_banks[name][user_row]
            result[:, 2 * channel_index : 2 * channel_index + 2] = [
                bank.get(int(item), (0.0, 0.0)) for item in candidate_ids
            ]
        positions = np.searchsorted(self.listen_items, candidate_ids)
        known = positions < len(self.listen_items)
        known &= (
            self.listen_items[np.minimum(positions, len(self.listen_items) - 1)]
            == candidate_ids
        )
        vectors = np.zeros((len(candidate_ids), self.cfg.factors), np.float32)
        vectors[known] = self.listen_vectors[positions[known]]
        norms = np.linalg.norm(vectors, axis=1)
        liked = self.listen_profiles["likes"][user_row]
        heard = self.listen_profiles["listens"][user_row]
        result[:, 2 * len(V9_NEW_CHANNELS) : 2 * len(V9_NEW_CHANNELS) + 6] = (
            np.column_stack(
                [
                    vectors @ liked,
                    vectors @ heard,
                    (vectors @ liked)
                    / np.maximum(norms * np.linalg.norm(liked), 1e-12),
                    (vectors @ heard)
                    / np.maximum(norms * np.linalg.norm(heard), 1e-12),
                    known,
                    norms,
                ]
            )
        )
        for row, item in enumerate(candidate_ids):
            artists = self.artist_mapping.get(int(item), [])
            likes = [
                self.artist_likes[(user_id, artist)]
                for artist in artists
                if (user_id, artist) in self.artist_likes
            ]
            listens = [
                self.artist_heard[(user_id, artist)]
                for artist in artists
                if (user_id, artist) in self.artist_heard
            ]
            result[
                row, 2 * len(V9_NEW_CHANNELS) + 6 : 2 * len(V9_NEW_CHANNELS) + 12
            ] = [
                np.log1p(max((value[0] for value in likes), default=0)),
                np.log1p(max((value[1] for value in likes), default=0)),
                np.log1p(max((value[2] for value in likes), default=0)),
                min((value[3] for value in likes), default=-1),
                np.log1p(max((value[0] for value in listens), default=0)),
                min((value[1] for value in listens), default=-1),
            ]
        return result


# research_v9_candidate_views
def v9_candidate_views(liked, heard, channels, budget, source_budget):
    baseline = V9_BASE_VIEWS(liked, heard, channels, budget, source_budget)["hybrid"]
    old_cf = core.fuse([channels[name] for name in SOURCE_NAMES], source_budget)
    listen_cf = core.fuse(
        [channels["listen_space_likes"], channels["listen_space_listens"]],
        source_budget,
    )
    original_core = core.fuse([liked, heard], 2 * source_budget)
    shared = [
        original_core,
        old_cf,
        listen_cf,
        channels["cosine_knn"],
        channels["bpr"],
        channels["ease"],
        channels["sansa"],
    ]
    neural = [channels["multvae"], channels["sasrec"], channels["two_tower"]]
    families = shared + neural
    fused = core.fuse(families, budget)
    selected = set()
    quota = min(250, budget // len(families))
    for family in families:
        selected.update(family[:quota])
    full_order = core.fuse(families, sum(map(len, families)))
    for item in full_order:
        if len(selected) >= budget:
            break
        selected.add(item)
    quota_pool = [item for item in full_order if item in selected][:budget]
    return dict(
        core=baseline,
        als=fused,
        hybrid=quota_pool,
        no_neural=core.fuse(shared, budget),
        no_listen_space=core.fuse([shared[0], shared[1], *shared[3:], *neural], budget),
    )


# research_v9_snapshot_implementation
@contextmanager
def v9_snapshot_implementation():
    names = [
        "Sources",
        "fast_collaborative_features",
        "v7_candidate_views",
        "CF_FEATURES",
        "V7_VERSION",
        "V7_CODE_SHA",
    ]
    previous = {name: globals()[name] for name in names}

    def features(source, user, ids):
        return np.concatenate(
            [V9_BASE_FAST(source, user, ids), source.extra_features(user, ids)], axis=1
        )

    globals().update(
        Sources=ResearchSources,
        fast_collaborative_features=features,
        v7_candidate_views=v9_candidate_views,
        CF_FEATURES=V9_BASE_CF_FEATURES + V9_EXTRA_FEATURES,
        V7_VERSION="research-v9",
        V7_CODE_SHA=V9_CODE_SHA,
    )
    try:
        yield
    finally:
        globals().update(previous)


# research_v9_recipes
def v9_recipes():
    old_features = FEATURES + V9_BASE_CF_FEATURES
    all_features = old_features + V9_EXTRA_FEATURES
    return [
        ("baseline_logloss", "core", old_features, "Logloss", "ips", False),
        ("new_pool_old_features", "als", old_features, "Logloss", "ips", False),
        ("new_pool_new_features", "als", all_features, "Logloss", "ips", False),
        ("quota_pool_logloss", "hybrid", all_features, "Logloss", "ips", False),
        (
            "no_neural_bundle",
            "no_neural",
            [
                name
                for name in all_features
                if not name.startswith(("multvae_", "sasrec_", "two_tower_"))
            ],
            "Logloss",
            "ips",
            False,
        ),
        (
            "no_listen_space_bundle",
            "no_listen_space",
            [name for name in all_features if not name.startswith("listen_space_")],
            "Logloss",
            "ips",
            False,
        ),
        ("uniform_logloss", "als", all_features, "Logloss", "uniform", False),
        ("querysoftmax", "als", all_features, "QuerySoftMax", "uniform", False),
        ("yetirank", "als", all_features, "YetiRank", "uniform", False),
        ("multiwindow_logloss", "als", all_features, "Logloss", "ips", True),
        ("multiwindow_yetirank", "als", all_features, "YetiRank", "uniform", True),
    ]


# research_v9_fit
def v9_fit(root, recipe):
    name, view, features, objective, weighting, multiple = recipe
    params = dict(
        loss_function=objective,
        iterations=max(RESEARCH_TREES),
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        border_count=128,
        random_seed=SEED,
        thread_count=THREADS,
        task_type=TASK_TYPE,
        verbose=100,
        allow_writing_files=False,
    )
    if TASK_TYPE == "GPU":
        params.update(devices="0", gpu_ram_part=0.85)
    identity = dict(params=params, recipe=recipe, code=V9_CODE_SHA)
    folder = root / "models"
    folder.mkdir(exist_ok=True)
    path = folder / (name + "_" + v7_fingerprint(identity)[:16] + ".cbm")
    model_type = CatBoostClassifier if objective == "Logloss" else CatBoostRanker
    if path.exists():
        model = model_type()
        model.load_model(path)
        return model
    frames = []
    periods = ["train", "older"] if multiple else ["train"]
    for index, period in enumerate(periods):
        frame = base.collect(
            pl.scan_parquet(root / "pairs" / period / (view + ".parquet"))
            .filter(pl.col("n_true") > 0)
            .select(features + ["uid", "target", "sample_weight"])
        )
        frame = frame.with_columns(
            (pl.col("uid").cast(pl.UInt64) + index * (1 << 32)).alias("query_id")
        )
        frames.append(frame)
    frame = pl.concat(frames).sort("query_id")
    del frames
    if not frame.height or not frame["target"].sum():
        raise ValueError("No retrieved training positives")
    groups = frame["query_id"].to_numpy()
    kwargs = dict(
        label=frame["target"].to_numpy(), feature_names=features, thread_count=THREADS
    )
    if objective == "Logloss":
        if weighting == "ips":
            kwargs["weight"] = frame["sample_weight"].to_numpy()
        else:
            _, inverse, counts = np.unique(
                groups, return_inverse=True, return_counts=True
            )
            kwargs["weight"] = (1 / counts[inverse]).astype(np.float32)
    else:
        # YetiRank ignores object weights; compare to a no-IPS Logloss control.
        # Equal group weights, same sampled rows, no claim of exposure correction.
        kwargs["group_id"] = groups
    pool = Pool(
        frame.select(features).to_numpy().astype(np.float32, copy=False), **kwargs
    )
    del frame, kwargs
    gc.collect()
    pool.quantize(
        border_count=128, used_ram_limit=f"{max(1, SCAN_MEMORY_GB)}gb", random_seed=SEED
    )
    model = model_type(**params)
    model.fit(pool)
    v9_json(path.with_suffix(".json"), dict(recipe=recipe, parameters=params))
    model.save_model(path.with_suffix(".partial.cbm"))
    path.with_suffix(".partial.cbm").replace(path)
    del pool
    gc.collect()
    return model


# research_v9_predictions
def v9_predictions(model, path, features, trees):
    result = {}
    parquet = pq.ParquetFile(path)
    for row_group in range(parquet.num_row_groups):
        frame = pl.from_arrow(parquet.read_row_group(row_group))
        predict_options = (
            dict(prediction_type="RawFormulaVal")
            if isinstance(model, CatBoostClassifier)
            else {}
        )
        score = model.predict(
            Pool(frame.select(features).to_numpy(), feature_names=features),
            ntree_end=trees,
            thread_count=THREADS,
            **predict_options,
        )
        frame = frame.with_columns(pl.Series("score", score))
        for (user,), group in frame.group_by("uid", maintain_order=True):
            if int(user) in result:
                raise ValueError(
                    "A user crosses row groups; evaluation writer must flush complete users"
                )
            result[int(user)] = (
                group.sort(["score", "item_id"], descending=[True, False])
                .head(100)["item_id"]
                .to_list()
            )
    return result


# research_run_research_v9
def run_research_v9():
    if DATA_SIZE not in ["50m", "500m"]:
        raise ValueError("This research notebook is configured for 50M or 500M")
    if (
        min(
            NEW_SOURCE_BUDGET,
            OLDER_TRAIN_USERS,
            TRAIN_USERS,
            RESEARCH_EVAL_USERS,
            LISTEN_ITEMS_PER_USER,
            LISTEN_ALS_ITEMS,
            COSINE_ITEMS,
            EASE_ITEMS,
            SANSA_ITEMS,
            BPR_ITERATIONS,
            NEURAL_ITEMS,
            NEURAL_USERS,
            VAE_EPOCHS,
            SASREC_EPOCHS,
            TOWER_EPOCHS,
        )
        < 1
    ):
        raise ValueError("Source and training budgets must be positive")
    if not 0 < SANSA_DENSITY <= 1 or not 100 <= LONG_CANDIDATE_BUDGET <= 8000:
        raise ValueError("Invalid sparse density or candidate budget")
    root = RUNS / "research_v9"
    root.mkdir(exist_ok=True)
    if not RESEARCH_TREES or min(RESEARCH_TREES) < 1:
        raise ValueError("Tree grid must be nonempty and positive")
    config = replace(cf_cfg, source_budget=LONG_SOURCE_BUDGET)
    users = sorted(ctx.users)
    train = sorted(
        np.random.default_rng(SEED + 1)
        .choice(users, min(TRAIN_USERS, len(users)), replace=False)
        .tolist()
    )
    older = train[: min(OLDER_TRAIN_USERS, len(train))]
    evaluation = sorted(
        np.random.default_rng(SEED + 7)
        .choice(users, min(RESEARCH_EVAL_USERS, len(users)), replace=False)
        .tolist()
    )
    protocol = [
        ("train", ctx.end - 210 * DAY, ctx.end - 120 * DAY, train),
        ("older", ctx.end - 240 * DAY, ctx.end - 150 * DAY, older),
        ("evaluation", ctx.end - 120 * DAY, ctx.end, evaluation),
    ]
    identity = dict(
        dataset=DATA_REVISION,
        dataset_size=DATA_SIZE,
        inputs=configuration["input_sha256"],
        tools=configuration.get("source_tools_sha256"),
        listens=cache_info["signature"],
    )
    signature = dict(
        code=V9_CODE_SHA,
        protocol=protocol,
        config=asdict(config),
        identity=identity,
        budget=LONG_CANDIDATE_BUDGET,
        negatives=LONG_NEGATIVES,
        lookback=LOOKBACK_DAYS,
        seed=SEED,
        listen_catalog=LISTEN_ALS_ITEMS,
        listen_user_items=LISTEN_ITEMS_PER_USER,
        cosine_catalog=COSINE_ITEMS,
        tree_grid=RESEARCH_TREES,
        task_type=TASK_TYPE,
        versions=versions,
        features=FEATURES + V9_BASE_CF_FEATURES + V9_EXTRA_FEATURES,
        additional_sources=dict(
            new_budget=NEW_SOURCE_BUDGET,
            ease_items=EASE_ITEMS,
            bpr_iterations=BPR_ITERATIONS,
            neural_items=NEURAL_ITEMS,
            neural_users=NEURAL_USERS,
            vae_epochs=VAE_EPOCHS,
            sasrec_epochs=SASREC_EPOCHS,
            tower_epochs=TOWER_EPOCHS,
            sansa_items=SANSA_ITEMS,
            sansa_density=SANSA_DENSITY,
            sansa_sha=SANSA_SHA,
            run_neural=RUN_NEURAL_SOURCES,
        ),
    )
    normalized = json.loads(json.dumps(signature))
    manifest = root / "data_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != normalized:
        raise ValueError("Data parameters changed: choose a new RUN_LABEL")
    v9_json(manifest, normalized)
    snapshots = {}
    with v9_snapshot_implementation():
        for period, cut, label_end, cohort in protocol:
            with memory_stage("Research " + period):
                snapshots[period] = v9_build_snapshot(
                    root, period, cut, label_end, cohort, config, identity
                )
    metadata = snapshots["evaluation"]
    truth = {int(user): set(items) for user, items in metadata["truth"].items()}
    complete = set(metadata["complete"])
    metrics = []
    predictions = {}
    importance = []
    for recipe in v9_recipes():
        name, view, features, objective, weighting, multiple = recipe
        with memory_stage("Ranker " + name):
            model = v9_fit(root, recipe)
            for trees in sorted(set(RESEARCH_TREES)):
                pred = v9_predictions(
                    model,
                    root / "pairs" / "evaluation" / (view + ".parquet"),
                    features,
                    trees,
                )
                predictions[(name, trees)] = pred
                for subgroup, cohort in [
                    ("all", evaluation),
                    (
                        "official_overlap",
                        [user for user in evaluation if user in OFFICIAL_UIDS],
                    ),
                    (
                        "population_extra",
                        [user for user in evaluation if user not in OFFICIAL_UIDS],
                    ),
                ]:
                    if cohort:
                        metrics.append(
                            dict(
                                experiment=name,
                                view=view,
                                objective=objective,
                                weighting=weighting,
                                multiwindow=multiple,
                                trees=trees,
                                cohort=subgroup,
                                **recall_metrics(pred, truth, cohort, complete),
                            )
                        )
            importance.extend(
                dict(experiment=name, feature=name_, importance=float(value))
                for name_, value in zip(
                    features,
                    model.get_feature_importance(type="PredictionValuesChange"),
                )
            )
            del model
            gc.collect()
    source_rows = [
        dict(period=period, **row)
        for period, metadata_ in snapshots.items()
        for row in metadata_.get("source_audit", [])
    ]
    pd.DataFrame(source_rows).to_csv(root / "source_audit.csv", index=False)
    save_table("research_source_audit", pd.DataFrame(source_rows))
    frame = pd.DataFrame(metrics)
    save_table("research_metrics", frame)
    frame.to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(importance).to_csv(root / "importance.csv", index=False)
    timings = [
        dict(period=period, stage=name, seconds=seconds)
        for period, metadata_ in snapshots.items()
        for name, seconds in metadata_["timings_seconds"].items()
    ]
    pd.DataFrame(timings).to_csv(root / "feature_timings.csv", index=False)
    coverage = []
    for period, metadata_ in snapshots.items():
        for view, values in metadata_["views"].items():
            coverage.append(
                dict(
                    period=period,
                    view=view,
                    **{key: value for key, value in values.items() if key != "rrf"},
                )
            )
    save_table("research_candidate_coverage", pd.DataFrame(coverage))
    pd.DataFrame(coverage).to_csv(root / "candidate_coverage.csv", index=False)
    # Same-iteration comparisons isolate pool, feature, weighting, objective and extra-window changes.
    comparisons = [
        ("new_pool_old_features", "baseline_logloss"),
        ("new_pool_new_features", "new_pool_old_features"),
        ("quota_pool_logloss", "new_pool_new_features"),
        ("uniform_logloss", "new_pool_new_features"),
        ("yetirank", "uniform_logloss"),
        ("new_pool_new_features", "no_neural_bundle"),
        ("new_pool_new_features", "no_listen_space_bundle"),
        ("querysoftmax", "uniform_logloss"),
        ("multiwindow_logloss", "new_pool_new_features"),
        ("multiwindow_yetirank", "yetirank"),
    ]
    paired = []
    for trees in sorted(set(RESEARCH_TREES)):
        for better, reference in comparisons:
            for cohort, cohort_users in [
                ("active", [user for user in evaluation if truth.get(user)]),
                ("full100", sorted(complete)),
                (
                    "official_full100",
                    [user for user in sorted(complete) if user in OFFICIAL_UIDS],
                ),
            ]:
                if cohort_users:
                    delta = recall_values(
                        predictions[(better, trees)], truth, cohort_users
                    ) - recall_values(
                        predictions[(reference, trees)], truth, cohort_users
                    )
                    paired.append(
                        dict(
                            comparison=better + " minus " + reference,
                            trees=trees,
                            cohort=cohort,
                            users=len(cohort_users),
                            **core.paired_interval(delta, SEED),
                        )
                    )
    save_table("research_paired", pd.DataFrame(paired))
    pd.DataFrame(paired).to_csv(root / "paired.csv", index=False)
    plot = frame[(frame.cohort == "all") & (frame.trees == max(RESEARCH_TREES))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, column in zip(axes, ["recall_active", "recall_full100"]):
        axis.barh(plot.experiment, 100 * plot[column])
        axis.invert_yaxis()
        axis.set(title=column, xlabel="Recall@100 (%)")
    save_figure("research_comparison", fig)
    summary = dict(
        version="research-v9",
        cohort=cohort_info,
        metrics=metrics,
        paired=paired,
        dataset_size=DATA_SIZE,
        development_only=True,
        not_comparable_to_old_5b_without_matching_cohort=True,
    )
    v9_json(root / "summary.json", summary)
    html = [
        "<meta charset='utf-8'><h1>Yambda: controlled development experiments</h1>",
        "<p>Candidate pools: core = previous hybrid; als = expanded RRF; hybrid = reserved quotas. Different cohort from the previous 5B report. No independent final test.</p>",
        "<h2>Cohort</h2><pre>" + escape(json.dumps(cohort_info, indent=2)) + "</pre>",
    ]
    for title, table in [
        ("Metrics", frame),
        ("Candidate coverage", pd.DataFrame(coverage)),
        ("Individual sources", pd.DataFrame(source_rows)),
        ("Paired comparisons", pd.DataFrame(paired)),
    ]:
        html.extend(["<h2>" + title + "</h2>", table.to_html(index=False)])
    html.append((RUNS / "research_comparison.svg").read_text())
    (root / "report.html").write_text("\n".join(html))
    archive = RUNS / "research_v9_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in [".csv", ".json", ".html"]:
                output.write(path, path.name)
        for name in [
            "cohort_audit.json",
            "configuration.json",
            "dataset_sources.json",
            "research_comparison.svg",
            "future_availability.csv",
            "activity.svg",
            "track_trends.svg",
            "feature_groups.csv",
        ]:
            path = RUNS / name
            if path.is_file():
                output.write(path, "eda/" + name)
    print("RESULTS:", archive)
    return root, archive


# research_v9_source_values
def v9_source_values(targets, liked, heard, channels, baseline):
    if not targets:
        return {}
    banks = dict(liked_family=liked, heard_family=heard, **channels)
    reference = set(baseline)
    values = {}
    for name, items in banks.items():
        hits = targets & set(items)
        others = set().union(
            *(set(bank) for other, bank in channels.items() if other != name)
        )
        values[name] = [
            len(hits) / len(targets),
            len(targets & set(items[:100])) / len(targets),
            len(hits - reference) / len(targets),
            len(hits - reference - others) / len(targets),
            len(items),
        ]
    return values


# research_v9_audit_summary
def v9_audit_summary(values, complete):
    rows = []
    names = sorted({name for banks in values.values() for name in banks})
    for name in names:
        active = [value[name] for value in values.values() if name in value]
        full = [
            value[name]
            for user, value in values.items()
            if user in complete and name in value
        ]
        for cohort, selected in [("active", active), ("full100", full)]:
            if selected:
                average = np.asarray(selected).mean(axis=0)
                rows.append(
                    dict(
                        source=name,
                        cohort=cohort,
                        users=len(selected),
                        coverage=average[0],
                        recall100=average[1],
                        additional_over_baseline=average[2],
                        unique_over_baseline_and_other_sources=average[3],
                        mean_candidates=average[4],
                    )
                )
    return rows


# research_v9_build_snapshot
def v9_build_snapshot(root, period, cut, label_end, users, config, source_identity):
    folder = root / "pairs" / period
    folder.mkdir(parents=True, exist_ok=True)
    names = FEATURES + CF_FEATURES + V7_RATE_FEATURES
    signature = dict(
        version=V7_VERSION,
        code=V7_CODE_SHA,
        source=source_identity,
        period=period,
        cutoff=cut,
        label_end=label_end,
        users=users,
        features=names,
        config=asdict(config),
        budget=LONG_CANDIDATE_BUDGET,
        negatives=LONG_NEGATIVES if period == "train" else 0,
        lookback=LOOKBACK_DAYS,
        seed=SEED,
    )
    manifest = folder / "manifest.json"
    paths = {
        view: folder / (view + ".parquet")
        for view in ["core", "als", "hybrid", "no_neural", "no_listen_space"]
    }
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        if metadata["signature"] != signature:
            raise ValueError("Snapshot parameters changed: choose a new LONG_RUN_LABEL")
        for view, path in paths.items():
            if (
                not path.exists()
                or pq.ParquetFile(path).metadata.num_rows
                != metadata["views"][view]["rows"]
            ):
                raise ValueError("Incomplete snapshot: " + str(path))
        print("Reusing feature tables:", folder, flush=True)
        return metadata
    if any(path.exists() for path in paths.values()):
        raise ValueError(
            "Snapshot has committed tables without a manifest: " + str(folder)
        )
    truth_sets, complete, _ = temporal_target(ctx, cut, label_end, users)
    truth = {int(user_id): sorted(targets) for user_id, targets in truth_sets.items()}
    setup_started = time.perf_counter()
    likes = LikeFeatures(
        ctx, cut, users, CACHE, 200000, source_budget=config.source_budget
    )
    hears = ListenFeatures(
        EVENTS, ctx, cut, users, likes.history, LOOKBACK_DAYS, config.source_budget
    )
    collaborative = Sources(DATA, cut, CACHE, config, users, likes.history, hears.pairs)
    setup_seconds = time.perf_counter() - setup_started
    source_audit = {}
    writers = {}
    buffers = {view: [] for view in paths}
    partials = {
        view: path.with_suffix(".partial.parquet") for view, path in paths.items()
    }
    for path in partials.values():
        path.unlink(missing_ok=True)
    counts = {view: 0 for view in paths}
    retrieval_counts = {view: 0 for view in paths}
    predictions = {view: {} for view in paths}
    coverage = {view: [] for view in paths}
    full_coverage = {view: [] for view in paths}
    user_info = {}
    target_slices = {}
    timings = {
        name: 0.0
        for name in [
            "like_features",
            "listen_features",
            "collaborative_features",
            "writing",
        ]
    }
    timings["candidate_generation"] = 0.0
    timings["model_setup"] = setup_seconds
    combined_column = names.index("combined_rr")

    def flush(view):
        if not buffers[view]:
            return
        table = pa.concat_tables(buffers[view])
        if view not in writers:
            writers[view] = pq.ParquetWriter(
                partials[view], table.schema, compression="zstd"
            )
        writers[view].write_table(table, row_group_size=65536)
        buffers[view].clear()

    try:
        with threadpool_limits(limits=1, user_api="blas"):
            for user_index, user_id in enumerate(users):
                candidate_started = time.perf_counter()
                source_lists = likes.user_sources(user_id)
                liked = source_lists[2][: config.source_budget]
                heard = core.fuse(
                    [bank.get(user_id, []) for bank in hears.lists.values()],
                    config.source_budget,
                )
                views = v7_candidate_views(
                    liked,
                    heard,
                    collaborative.lists(user_id),
                    LONG_CANDIDATE_BUDGET,
                    config.source_budget,
                )
                source_audit[user_id] = v9_source_values(
                    truth_sets.get(user_id, set()),
                    liked,
                    heard,
                    collaborative.lists(user_id),
                    views["core"],
                )
                union_ids = np.asarray(
                    sorted(set().union(*map(set, views.values()))), np.int64
                )
                if min(map(len, views.values())) < 100:
                    raise ValueError(f"Insufficient candidates for uid={user_id}")
                timings["candidate_generation"] += (
                    time.perf_counter() - candidate_started
                )
                started = time.perf_counter()
                like_values = likes.features(user_id, union_ids, source_lists)
                timings["like_features"] += time.perf_counter() - started
                started = time.perf_counter()
                listen_values = hears.features(
                    user_id, union_ids, {"like": set(liked), "listen": set(heard)}
                )
                timings["listen_features"] += time.perf_counter() - started
                started = time.perf_counter()
                cf_values = fast_collaborative_features(
                    collaborative, user_id, union_ids
                )
                timings["collaborative_features"] += time.perf_counter() - started
                observed_days = max(1.0, (cut - ctx.start) / DAY)
                history_frame = likes.histories.get(user_id)
                history_count = history_frame.height if history_frame is not None else 0
                recent_fraction = like_values[
                    0, FEATURES.index("user_recent30_fraction")
                ]
                rate_values = np.column_stack(
                    [
                        np.full(
                            len(union_ids), np.log1p(history_count / observed_days)
                        ),
                        np.full(
                            len(union_ids),
                            recent_fraction * observed_days / min(30.0, observed_days),
                        ),
                    ]
                ).astype(np.float32)
                features = np.concatenate(
                    [like_values, listen_values, cf_values, rate_values], axis=1
                )
                validate_feature_matrix(user_id, union_ids, features, names)
                targets = truth_sets.get(user_id, set())
                past_likes = likes.histories.get(user_id)
                past_like_ids = (
                    set(past_likes["item_id"].to_list())
                    if past_likes is not None
                    else set()
                )
                past_hears = hears.pairs.get(user_id)
                past_heard_ids = (
                    set(past_hears["item_id"].to_list())
                    if past_hears is not None
                    else set()
                )
                user_info[user_id] = dict(
                    has_listens=bool(past_heard_ids), history_likes=len(past_like_ids)
                )
                target_slices[user_id] = dict(
                    prior_like=sorted(targets & past_like_ids),
                    heard90_not_liked=sorted(
                        (targets & past_heard_ids) - past_like_ids
                    ),
                    not_heard90_not_liked=sorted(
                        targets - past_heard_ids - past_like_ids
                    ),
                )
                started = time.perf_counter()
                for view, candidate_list in views.items():
                    ids = np.asarray(candidate_list, np.int64)
                    retrieval_counts[view] += len(ids)
                    rows = np.searchsorted(union_ids, ids)
                    values = features[rows].copy()
                    # Combined RRF position belongs to the view, not to the superset ordering.
                    values[:, combined_column] = 1.0 / np.arange(1, len(ids) + 1)
                    labels = np.asarray(
                        [int(item_id in targets) for item_id in ids], np.int8
                    )
                    predictions[view][user_id] = ids[:100].tolist()
                    if targets:
                        value = float(labels.sum() / len(targets))
                        coverage[view].append(value)
                        if user_id in complete:
                            full_coverage[view].append(value)
                    selected, weights = sample_indices(
                        labels,
                        LONG_NEGATIVES if period == "train" else 0,
                        SEED,
                        cut,
                        user_id,
                    )
                    table = {
                        name: pa.array(values[selected, column])
                        for column, name in enumerate(names)
                    }
                    table.update(
                        uid=pa.array(np.full(len(selected), user_id, np.int64)),
                        item_id=pa.array(ids[selected]),
                        target=pa.array(labels[selected]),
                        n_true=pa.array(np.full(len(selected), len(targets), np.int16)),
                        sample_weight=pa.array(weights),
                        candidate_rank=pa.array((selected + 1).astype(np.int32)),
                    )
                    buffers[view].append(pa.table(table))
                    counts[view] += len(selected)
                    if len(buffers[view]) >= 8:
                        flush(view)
                timings["writing"] += time.perf_counter() - started
                if (user_index + 1) % 250 == 0:
                    log(f"LONG {period}: {user_index + 1}/{len(users)} users")
            for view in paths:
                flush(view)
    finally:
        for writer in writers.values():
            writer.close()
    metadata = dict(
        signature=signature,
        users=users,
        truth=truth,
        complete=sorted(complete),
        cutoff=cut,
        label_end=label_end,
        user_info=user_info,
        source_audit=v9_audit_summary(source_audit, complete),
        target_slices=target_slices,
        source_models=collaborative.info,
        timings_seconds=timings,
        views={},
    )
    for view, path in paths.items():
        partials[view].replace(path)
        metadata["views"][view] = dict(
            rows=counts[view],
            retrieval_pairs=retrieval_counts[view],
            mean_candidates=retrieval_counts[view] / len(users),
            coverage_active=float(np.mean(coverage[view])) if coverage[view] else None,
            coverage_full100=float(np.mean(full_coverage[view]))
            if full_coverage[view]
            else None,
            rrf=recall_metrics(predictions[view], truth_sets, users, complete),
        )
    partial_manifest = manifest.with_suffix(".partial.json")
    partial_manifest.write_text(json.dumps(metadata))
    partial_manifest.replace(manifest)
    del likes, hears, collaborative, features
    gc.collect()
    return metadata


# run_research
RESEARCH_OUTPUT, RESEARCH_ARCHIVE = run_research_v9()
