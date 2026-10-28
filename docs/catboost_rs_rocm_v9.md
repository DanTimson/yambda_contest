# catboost-rs-rocm assessment, 2026-10-28

This package is an independent Rust implementation, not the official CatBoost HIP backend. Published `catboost-rs-rocm==0.1.0` requires CPython >=3.12 and glibc >=2.39. Its PyPI provenance links to commit `56e24518ffc928598279d32e22b5148ef86bb0cf`.

The release's Python loss parser supports RMSE, Logloss, CrossEntropy, MAE and LogCosh; QuerySoftMax and YetiRank are rejected. Reading latest main at `6226fdd62e870b314493d4156fa0a500e72f7256` shows added ranking losses, but this is different code from the published wheel. The latest Python ranker interface still does not provide our pipeline's quantized Pool, tree-prefix prediction, save_model or get_feature_importance APIs. Replacing the import would break caching and evaluation, even if a source build runs.

An isolated CPU-wheel synthetic smoke on 64 rows confirmed finite Logloss predictions and rejection of both ranking losses, plus missing Pool.quantize/save_model/get_feature_importance. This does not verify the ROCm binary on AMD or measure its speed. Release-source inspection, independently, confirms the same restricted loss parser for the published ROCm package. No real Yambda run was done for this check.

The maintainers describe validation on gfx1100/ROCm 7.1.1, relevant to RX 7900 XTX, but not confirmation for this user's WSL installation. Their ROCm packaging notes document a HIPRTC JIT library-resolution segfault and recommend loading the system HIP libraries with ROCM_PATH/HIP_PATH/LD_PRELOAD. The optional script uses that documented setup, requires existing system ROCm, and changes no drivers.

Run the isolated probe (512 synthetic rows, 20 trees, three loss-function attempts):

```bash
cd "$HOME/yambda_contest" && bash scripts/probe_catboost_rs_rocm_v9.sh
```

It uses a separate `.venv-catboost-rs-v9`; do not install the `catboost-rs` CPU distribution alongside its ROCm distribution because both provide `catboost_rs`. Official `catboost` has a different import name and can coexist. Return `eda_results/catboost_rs_probe_v9.json`.

A successful Logloss fit would justify a separate candidate-model quality/speed comparison. It would not justify changing the controlled YetiRank/QuerySoftMax study to an unverified implementation. Source-build integration would require a pinned build, adapter for the API gaps, and equality of data/sampling/features; model quality cannot be assumed from the name.

Primary sources:
- https://pypi.org/project/catboost-rs-rocm/0.1.0/
- https://github.com/BectorVoom/catboost_rs/blob/56e24518ffc928598279d32e22b5148ef86bb0cf/crates/catboost-rs-py/src/params.rs
- https://github.com/BectorVoom/catboost_rs/blob/56e24518ffc928598279d32e22b5148ef86bb0cf/crates/catboost-rs-py/pyproject-rocm.toml
- https://github.com/BectorVoom/catboost_rs/blob/6226fdd62e870b314493d4156fa0a500e72f7256/crates/catboost-rs-py/src/ranker.rs
