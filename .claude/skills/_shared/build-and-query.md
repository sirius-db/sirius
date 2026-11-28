# Shared Infrastructure for Sirius Debugging Skills

## Builds and sanitizers

Run builds through `pixi run make <target>`. The `release` preset is optimized;
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

Use wrapper target `debug` for `debug`/`clang-debug` libraries, and `reldebug` for
`relwithdebinfo`/`clang-relwithdebinfo`. Wrapper outputs are under
`sirius-duckdb/build/debug` or `sirius-duckdb/build/reldebug`, respectively. Take care with existing wrapper build
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
SQL after loading the extension; Sirius intercepts supported queries. Set
`wrapper_build_dir` to the actual wrapper output directory for these commands. This
example uses the default release wrapper:

```bash
wrapper_build_dir="${wrapper_build_dir:-sirius-duckdb/build/release}"
pixi run "$wrapper_build_dir/duckdb" -unsigned -cmd "LOAD '${wrapper_build_dir//\'/\'\'}/extension/sirius/sirius.duckdb_extension';" <database_path>
```

For CPU comparison, use `SET gpu_execution = false` or start the process with
`SIRIUS_DISABLE=1`. The wrapper's DuckDB CLI has the Sirius extension built in
regardless of `SIRIUS_DUCKDB_LINKAGE`, so omitting `LOAD` is insufficient. Compare results with ordering and query semantics in mind. Repeated
runs can expose inconsistent results, but a difference alone does not prove a race.

## Changes and logs

Respect the requested scope: diagnose and propose fixes for an analysis request;
apply and verify fixes when the request covers them. Preserve existing user changes
and remove temporary instrumentation introduced for the investigation when finished.
Do not stage or commit changes unless requested. Consider the cost of sanitizer
builds and repeated runs alongside the user's resource constraints.

For file logging, set `SIRIUS_LOG_BACKEND=spdlog`, `SIRIUS_LOG_LEVEL=trace`, and a
per-run `SIRIUS_LOG_DIR` as appropriate. The owning sink writes date-stamped
`sirius*.log` files with daily rotation. The `sirius_log_flush_seconds` SQL setting
controls the spdlog flush interval (default three seconds). A crash or abrupt exit can
leave entries since the last flush buffered. See `src/log/spdlog_owning_sink.cpp` and
`src/log/logging.hpp`. Test logs live under `<build-dir>/test/cpp/log/`, where
`<build-dir>` is the directory passed to `scripts/run_unit_tests.py --build-dir` (default `build/release`).
Temporary trace statements can use a `[SIRIUS_DIAG]` tag to make them easy to find.

The active engine includes code in both `namespace sirius` and `namespace duckdb`.
Search relevant `src/` paths and follow call sites; namespace alone does not
identify whether code participates in execution.
