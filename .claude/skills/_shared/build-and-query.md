# Shared Infrastructure for Sirius Debugging Skills

## Builds and sanitizers

Run builds through `pixi run make <preset>`. The `release` preset is optimized;
`relwithdebinfo` and `clang-relwithdebinfo` include host debug symbols;
`debug` and `clang-debug` include device line information. Root presets build Sirius and its
C++ tests. `pixi run make` also installs the release library and builds the wrapper;
see [sirius-duckdb/README.md](../../../sirius-duckdb/README.md) for shared wrapper
builds. Other root presets build only Sirius and tests. For SQL debugging, install
the selected library and rebuild the wrapper against that prefix with compatible
toolchains. For example, from the repository root:

```bash
pixi run make clang-debug
pixi run cmake --install build/clang-debug --prefix "$PWD/build/clang-debug/install" --component sirius_library
pixi run make -C sirius-duckdb debug SIRIUS_DUCKDB_LINKAGE=shared SIRIUS_INSTALL_DIR="$PWD/build/clang-debug/install"
```

This example uses the wrapper under `sirius-duckdb/build/debug`; choose the
actual wrapper directory for other targets. Take care with existing wrapper build
directories and CMake caches when changing the library or toolchain.

For host sanitizers, use the separate `clang-asan` and `clang-tsan` presets. Their
outputs are in `build/clang-asan` and `build/clang-tsan`; they cannot be combined.
`docs/super-sirius/debugging.md` documents required ASan CUDA options, TSan
suppressions and runtime options, and Compute Sanitizer limitations. In particular,
pooled GPU allocations can hide sub-allocation overruns from memcheck. Compute
Sanitizer does not require an instrumented build; `clang-debug` provides device
line information. Compute Sanitizer and cuda-gdb may not be installed or on
`PATH`; check availability before selecting either tool.

## SQL and results

Use the supplied query, database, Parquet paths, and setup statements when available.
Ask only for information needed to reproduce the issue that cannot be inferred.
Avoid rerunning mutating SQL against user data without authorization. Run normal
SQL after loading the extension; Sirius transparently intercepts supported queries:

```bash
pixi run sirius-duckdb/build/release/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';" <database_path>
```

For CPU comparison, run the same SQL with `SET gpu_execution = false` or without
the extension. Compare results with ordering and query semantics in mind. Repeated
runs can expose inconsistent results, but a difference alone does not prove a race.

## Changes and logs

Respect the requested scope: diagnose and propose fixes for an analysis request;
apply and verify fixes when the request covers them. Preserve existing user changes
and remove temporary instrumentation introduced for the investigation when finished.
Do not stage or commit changes unless requested. Consider the cost of sanitizer
builds and repeated runs alongside the user's resource constraints.

For file logging, set `SIRIUS_LOG_BACKEND=spdlog`, `SIRIUS_LOG_LEVEL=trace`, and a
per-run `SIRIUS_LOG_DIR` as appropriate. The owning sink writes date-stamped
`sirius*.log` files with daily rotation and periodic best-effort flushing; trace level does not guarantee
all entries reach disk before a crash. See `src/log/spdlog_owning_sink.cpp` and
`src/log/logging.hpp`. Test logs live under `build/<preset>/test/cpp/log/`.
Temporary trace statements can use a `[SIRIUS_DIAG]` tag to make them easy to find.

The active engine includes code in both `namespace sirius` and `namespace duckdb`.
Search relevant `src/` paths and follow call sites; namespace alone does not
identify whether code participates in execution.
