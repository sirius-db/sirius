# Shared Infrastructure for Sirius Debugging Skills

This document defines common patterns used across all debugging skills.

## Build Modes

All skills support three build presets:
- `release` -- optimized build, no debug symbols. Default for performance testing and general use.
- `relwithdebinfo` -- optimized build **with** debug symbols. Best for profiling (nsys), cuda-gdb crash debugging, and getting meaningful stack traces at near-production speed. Also available as `clang-relwithdebinfo`.
- `clang-debug` -- clang compiler, unoptimized with full debug symbols. Required for sanitizers (ASan/TSan). Slowest but most debuggable.

**When to use which preset:**
| Preset | Speed | Debug symbols | Sanitizers | Use for |
|--------|-------|---------------|------------|---------|
| `release` | Fastest | No | No | Performance benchmarks, general queries |
| `relwithdebinfo` | Fast | Yes | No | Profiling (nsys), crash backtraces, cuda-gdb |
| `clang-debug` | Slow | Yes | Yes (ASan/TSan) | Race detection, memory error detection |

**Build command pattern (always use pixi):**
```bash
cd /home/bwyogatama/sirius
pixi run -e clang make release
# or
pixi run -e clang make relwithdebinfo
# or
pixi run -e clang make clang-debug
```

**Important:** Always build inside a pixi environment. Pixi manages all dependencies (CUDA toolkit, cuDF, clang, sccache, etc.) via `pixi.toml`.

## AddressSanitizer (ASan)

ASan detects CPU-side memory errors: heap/stack buffer overflows, use-after-free, double-free, and memory leaks. It does **not** detect GPU memory errors (use NVIDIA Compute Sanitizer `memcheck` for that).

**How ASan is configured:**
DuckDB's CMake has `ENABLE_SANITIZER=TRUE` by default, which adds `-fsanitize=address` to Debug builds only (via `CXX_EXTRA_DEBUG`). This means:
- **`clang-debug`**: ASan is **on by default** -- no extra flags needed
- **`release` / `relwithdebinfo`**: ASan is **not active** (the flag only applies to Debug build types)

**To explicitly disable ASan in debug builds** (e.g., when using TSan instead):
```bash
pixi run -e clang make clang-debug EXTRA_CMAKE_FLAGS="-DENABLE_SANITIZER=0"
```

**ASan vs TSan -- mutually exclusive:**
ASan and TSan cannot be used simultaneously. DuckDB will warn and disable ASan if both are enabled. Use separate builds:
- ASan build: `clang-debug` (default, no extra flags)
- TSan build: `clang-debug` with `EXTRA_CMAKE_FLAGS="-DENABLE_TSAN=ON -DENABLE_SANITIZER=0"`

**ASan runtime options:**
```bash
ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:print_legend=1" build/clang-debug/duckdb ...
```
- `detect_leaks=1`: Also report memory leaks at exit
- `halt_on_error=0`: Continue after first error (collect multiple reports)
- `halt_on_error=1` (default): Stop on first error (better for interactive debugging)

**ASan overhead:** ~2x slowdown, ~2-3x memory increase. Much lighter than Valgrind.

**What ASan catches (CPU-side only):**
- Heap buffer overflow / underflow
- Stack buffer overflow
- Use-after-free / use-after-return
- Double-free
- Memory leaks (with `detect_leaks=1`)
- Stack use after scope (with `-fsanitize-address-use-after-scope`)

## SQL Query Execution

All skills accept an optional SQL query from the user. Follow this pattern:

1. Ask the user whether their data is in **DuckDB format** or **Parquet format**
2. Run the query once with `SIRIUS_LOG_LEVEL=trace` to generate detailed logs
3. Read the log from `build/<preset>/log/sirius_<date>.log`
4. Use the log to identify the code path (which operators, pipelines, memory regions were hit)
5. Scope subsequent analysis only to relevant source files

**Query execution -- DuckDB format:**
```bash
export SIRIUS_LOG_LEVEL=trace
build/<preset>/duckdb <path_to_database.duckdb>
```
Then inside the DuckDB CLI:
```sql
CALL gpu_execution('<USER_SQL_QUERY>');
```

**Query execution -- Parquet format:**
Ask the user for the parquet directory path, then:
```bash
export SIRIUS_LOG_LEVEL=trace
build/<preset>/duckdb
```
Then inside the DuckDB CLI, create views for each table from parquet files:
```sql
CREATE OR REPLACE VIEW lineitem AS SELECT * FROM '/path/to/parquet_dir/lineitem/*.parquet';
CREATE OR REPLACE VIEW orders AS SELECT * FROM '/path/to/parquet_dir/orders/*.parquet';
-- ... repeat for each table
CALL gpu_execution('<USER_SQL_QUERY>');
```

## Result Comparison Against DuckDB (CPU Baseline)

All skills that run SQL queries offer the option to compare Sirius GPU results against DuckDB's native CPU execution.

**Pattern:**
1. Run the query via DuckDB CPU (no Sirius extension): `build/release/duckdb <db_path>` then `SELECT ...;`
2. Run the same query via Sirius GPU: `build/release/duckdb <db_path>` then `CALL gpu_execution('SELECT ...');`
3. Diff the results row-by-row (sort both outputs first to handle ordering differences)
4. Report any mismatches: missing rows, extra rows, wrong values, type differences

**Multi-Run Consistency Check:**
For detecting non-deterministic behavior (e.g., race conditions):
1. Run the query N times (default: 3), saving each result set
2. Compare all results pairwise
3. If any differ, report which runs diverged and flag as potential race condition
4. This can automatically trigger the `/race-check` skill if inconsistency is detected

## Code Scope: New Sirius vs Legacy

These skills target **new Sirius** only:
- **New Sirius:** files using `namespace sirius` -- the active codebase
- **Legacy Sirius:** files using `namespace duckdb` -- deprecated, should be ignored

**Exception:** The following legacy files are still used by new Sirius and should be included:
- `src/include/log/*` -- logging infrastructure
- `src/expression_executor/*` -- expression evaluation
- `src/sirius_extension.cpp` -- extension entry point

When searching for relevant code, filter to `namespace sirius` files plus the exceptions above.

## Autonomy Mode

All skills support an **autonomy mode** that controls interactivity:

| Mode | Behavior |
|------|----------|
| `interactive` (default) | Pause after each diagnosis/fix suggestion. Wait for user approval. |
| `autonomous` | Apply fixes, rebuild, re-run, and iterate automatically until resolved or max iterations reached. |
| `semi-autonomous` | Iterate automatically, but pause at key decision points (e.g., choosing between fixes, modifying >3 files). |

In `autonomous` and `semi-autonomous` modes:
- All changes tracked via git commits
- Max iteration limit (default: 5) prevents infinite loops
- Summary of all attempted fixes presented at the end
- End-of-session cleanup prompts user to keep, revert, or cherry-pick changes

**Phase transitions:** All skills with multiple phases must **ask the user before proceeding to the next phase**. Summarize findings, explain the next phase, and ask for confirmation.

## Change Tracking & Easy Revert

All skills must make changes fully reversible.

**Git checkpoint (before any changes):**
```bash
git add -A && git commit -m "SIRIUS_DEBUG_CHECKPOINT: pre-skill state" --allow-empty
```

**Per-iteration snapshots:**
```bash
git add -A && git commit -m "SIRIUS_DEBUG_ITER_<N>: <brief description>"
```

**End-of-session cleanup options:**
1. **Keep all changes** -- squash debug commits into a single clean commit
2. **Keep only the fix** -- revert diagnostic logs, keep the bug fix
3. **Revert everything** -- return to pre-skill checkpoint
4. **Cherry-pick** -- user chooses which iterations to keep

## Debug Log Insertion & Cleanup

Skills can insert `SIRIUS_LOG_TRACE(...)` statements for diagnosis. All use the `[SIRIUS_DIAG]` tag:

```cpp
SIRIUS_LOG_TRACE("[SIRIUS_DIAG] <context>: var={}", var);
```

**Finding and removing diagnostic logs:**
```bash
grep -rn "SIRIUS_DIAG" src/
```

**Log lifecycle:**
1. **Insert** -- add `[SIRIUS_DIAG]` logs during diagnosis
2. **Rebuild & run** -- logs appear in `build/<preset>/log/sirius_<date>.log`
3. **Analyze** -- read logs to narrow down the issue
4. **Iterate** -- add more targeted logs if needed
5. **Cleanup** -- categorize each log:
   - **Promote**: Useful long-term -> change to proper log message with appropriate level
   - **Remove**: Only useful for this investigation -> delete entirely

## Log Analysis

- Logs live in `build/<preset>/log/sirius_<date>.log`
- Unit test logs in `build/<preset>/extension/sirius/test/cpp/log/`
- Log format: `[YYYY-MM-DD HH:MM:SS.mmm] [level] [source_file:line] message`
- Controlled by `SIRIUS_LOG_LEVEL` env var -- **always use `trace`** for maximum detail
- Logging macros: `SIRIUS_LOG_TRACE`, `SIRIUS_LOG_DEBUG`, `SIRIUS_LOG_INFO`, `SIRIUS_LOG_WARN`, `SIRIUS_LOG_ERROR`, `SIRIUS_LOG_FATAL` (defined in `src/include/log/logging.hpp`)

**Per-run log files:**
Create separate log directories per run:
```bash
export SIRIUS_LOG_DIR=build/release/log/run_$(date +%s)
mkdir -p $SIRIUS_LOG_DIR
```

**Immediate flush for crash debugging:**
When `SIRIUS_LOG_LEVEL` is explicitly set, the logger calls `spdlog::flush_on(log_level)` which flushes after every log entry. This ensures no entries are lost before a crash. See `src/include/log/logging.hpp:94-98`.
