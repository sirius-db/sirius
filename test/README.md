# Testing this extension
This directory contains all the tests for this extension. The `cpp` directory holds the C++ unit and integration tests, written with [Catch2](https://github.com/catchorg/Catch2).

## C++ unit tests

CI and `make test` run the C++ unit tests with `scripts/run_unit_tests.py`. `make test` builds the release library and wrapper first, and sets `SIRIUS_EXTENSION_PATH`
to include extension-loading checks. `make test_debug` builds the debug library and tests;
set `SIRIUS_EXTENSION_PATH` explicitly to test a compatible wrapper with that build:
```bash
pixi run make test
pixi run python scripts/run_unit_tests.py                                # without rebuilding
pixi run python scripts/run_unit_tests.py --steps shards                 # one step
pixi run python scripts/run_unit_tests.py -- --order rand --rng-seed 5   # Catch2 options for every process
pixi run make test UNITTEST_ARGS="-- --abort"                            # options through make
```

Dynamic scan checks launch `sirius_extension_host`, built alongside `sirius_unittest`.
This DuckDB-only process loads the wrapper without an embedded Sirius copy.

The script runs three steps:

| Step | What runs |
|---|---|
| `shards` | Two Catch2 shards per visible GPU in parallel. Each shard sees one GPU, sets `SIRIUS_TEST_SINGLE_GPU=1` and uses `cpp/integration/integration-shard.yaml`. `[multi_gpu]` and hidden tests are excluded. |
| `multi_gpu` | The `[multi_gpu]` tests with all GPUs visible. Skipped with fewer than two GPUs. |
| `late_mat` | The `[late_mat]`, `[deferred_query]` and `[native_filter]` tests with `SIRIUS_EXP_LATE_MAT=1`. |

It respects `CUDA_VISIBLE_DEVICES` and shards over the GPUs it lists. Each process writes `unittest.log` and `sirius.log` to its own subdirectory of `build/release/test/cpp/log/`, for example `shard-0/` or `late_mat/`.

To run specific tests, call the test binary directly with a Catch2 tag or test name:
```bash
pixi run build/release/test/cpp/sirius_unittest "[uri_parser]"
pixi run build/release/test/cpp/sirius_unittest "uri_parser parses bare absolute paths as file URIs"
```
