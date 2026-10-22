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

## TPC-Havoc queries

[TPC-Havoc](https://github.com/BenchBox-dev/BenchBox/blob/develop/docs/benchmarks/tpc-havoc.md)
rewrites each of the 22 TPC-H queries ten ways: scalar subqueries, CTEs, UNION ALL, window
functions and so on. Each rewrite returns the same rows as the original query, so the set covers
many plan shapes over the same data.

The query files come from [BenchBox](https://github.com/BenchBox-dev/BenchBox), which is MIT-licensed;
TPC-Havoc is derived from TPC-H. They aren't checked in. Instead, `cmake/sirius-tpch-havoc.cmake`
downloads the 22 `qNN.yaml` files and BenchBox's `LICENSE` from a pinned commit into
`build/<preset>/test/tpch_havoc/` when tests are enabled, and checks each file's SHA-256.
Tests load them with `test/cpp/utils/tpch_havoc_queries.hpp`. There are 220 variants, 184 of them
distinct.

```bash
pixi run build/release/test/cpp/sirius_unittest "[tpch_havoc]"
```

To classify them on the GPU outside the test process, one CLI process per query, run
`test/scripts/classify_queries.py --benchmark tpch-havoc`. Its docstring describes the options.

- **Offline builds:** configure with `-DSIRIUS_TPCH_HAVOC_DIR=<dir>` to use a copy of the same files,
  or with `-DSIRIUS_TPCH_HAVOC=OFF` to skip them. With `OFF`, the TPC-Havoc tests skip.
- **Another copy at runtime:** set the `SIRIUS_TPCH_HAVOC_DIR` environment variable.
- **Moving the pin:** update the commit in `cmake/sirius-tpch-havoc.cmake`, then replace its file list
  with the output of:

  ```bash
  commit=<new commit>
  for f in LICENSE benchbox/core/tpchavoc/variant_sets/q{01..22}.yaml; do
    url=https://raw.githubusercontent.com/BenchBox-dev/BenchBox/$commit/$f
    echo "$f=$(curl -sfL "$url" | sha256sum | cut -d' ' -f1)"
  done
  ```
