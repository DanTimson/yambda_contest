#!/usr/bin/env python3
"""Small isolated backend probe. Synthetic data only; no Yambda downloads."""

import argparse
import ctypes
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--out", type=Path, default=Path("eda_results/catboost_rs_probe_v9.json")
)
parser.add_argument("--expect", choices=["rocm", "cpu"], default="rocm")
args = parser.parse_args()
result = {
    "expect": args.expect,
    "rows": 512,
    "features": 16,
    "iterations": 20,
    "objectives": [],
}
try:
    distribution = "catboost-rs-rocm" if args.expect == "rocm" else "catboost-rs"
    result["distribution"] = {distribution: importlib.metadata.version(distribution)}
    for other in [
        "catboost-rs",
        "catboost-rs-rocm",
        "catboost-rs-cuda",
        "catboost-rs-wgpu",
    ]:
        if other == distribution:
            continue
        try:
            version = importlib.metadata.version(other)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise RuntimeError(
            f"Conflicting catboost_rs distribution installed: {other}=={version}"
        )
    if args.expect == "rocm":
        runtime = ctypes.CDLL("libamdhip64.so.7")
        count = ctypes.c_int()
        status = runtime.hipGetDeviceCount(ctypes.byref(count))
        result["hip_status"] = status
        result["hip_devices"] = count.value
        if status != 0 or count.value < 1:
            raise RuntimeError("HIP cannot see a GPU; no fit attempted")
    import catboost as reference
    import catboost_rs as alternative

    rng = np.random.default_rng(42)
    features = np.ascontiguousarray(rng.normal(size=(512, 16)), np.float32)
    labels = (
        features[:, 0] + 0.3 * features[:, 1] + rng.normal(scale=0.3, size=512) > 0
    ).astype(np.float32)
    groups = np.repeat(np.arange(32, dtype=np.uint64), 16)
    result["api"] = {
        "pool_quantize": hasattr(alternative.Pool(features), "quantize"),
        "ranker_save_model": hasattr(alternative.CatBoostRanker(), "save_model"),
        "ranker_predict_signature": str(
            inspect.signature(alternative.CatBoostRanker.predict)
        ),
    }
    for objective in ["Logloss", "QuerySoftMax", "YetiRank"]:
        row = {"objective": objective}
        for name, module in [
            ("official_cpu", reference),
            ("rust_" + args.expect, alternative),
        ]:
            try:
                model_class = (
                    module.CatBoostClassifier
                    if objective == "Logloss"
                    else module.CatBoostRanker
                )
                parameters = dict(
                    loss_function=objective,
                    iterations=20,
                    depth=4,
                    learning_rate=0.05,
                    random_seed=42,
                )
                if name == "official_cpu":
                    parameters.update(
                        task_type="CPU",
                        thread_count=4,
                        verbose=False,
                        allow_writing_files=False,
                    )
                model = model_class(**parameters)
                # Published rs wheel chooses backend at build time and rejects task_type.
                pool = module.Pool(features, label=labels, group_id=groups)
                start = time.perf_counter()
                model.fit(pool)
                elapsed = time.perf_counter() - start
                prediction = (
                    model.predict_proba(features)[:, 1]
                    if objective == "Logloss"
                    else model.predict(features)
                )
                if not np.isfinite(prediction).all():
                    raise ValueError("Non-finite scores")
                row[name] = {
                    "status": "ok",
                    "fit_seconds": elapsed,
                    "unique_scores": len(np.unique(prediction)),
                }
                # Training fit is only a functioning check, not generalization evidence.
            except Exception as error:
                row[name] = {
                    "status": "error",
                    "type": type(error).__name__,
                    "message": str(error),
                }
        result["objectives"].append(row)
except Exception as error:
    result["preflight_error"] = {"type": type(error).__name__, "message": str(error)}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("RETURN:", args.out)
if "preflight_error" in result:
    raise SystemExit(1)
