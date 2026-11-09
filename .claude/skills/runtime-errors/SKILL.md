---
name: runtime-errors
description: >
  Use this skill when a Sirius query crashes, segfaults, hangs, throws an exception, or unexpectedly
  falls back to CPU. Also use when you see CUDA errors, std::bad_alloc, or the process gets killed.
  Diagnoses issues using log analysis, cuda-gdb, AddressSanitizer, and NVIDIA Compute Sanitizer.
argument-hint: "[sql-query-or-file] [--timeout 30]"
disable-model-invocation: true
---

# Runtime Error Analyzer

Diagnose runtime errors (crashes, segfaults, exceptions, hangs, fallback triggers) by analyzing log files, inserting targeted debug logs, using cuda-gdb and NVIDIA Compute Sanitizer, and re-running queries.

**Reference:** See `.claude/skills/_shared/build-and-query.md` for shared infrastructure (build modes, query execution, changes and logs).

## Triage

1. **Gather context:**
   - Use the supplied query, input paths, and error description; ask only for missing details needed to reproduce.
   - Parse optional `--timeout` argument (default: 30 seconds).
   - Diagnose within the requested scope. Apply fixes only if the request includes them. Preserve existing user work and avoid rerunning mutating SQL against user data without authorization.

2. **Classify the error type** to determine which workflow path to follow:
   - **Hang / timeout** (query never returns, process stuck) -> **Hang Path**
   - **Segmentation fault (SIGSEGV)** (process killed by signal 11) -> **Segfault Path**
   - **Runtime error** (exception, error message, unexpected fallback, wrong exit code) -> **Runtime Error Path**

Choose the diagnostic steps supported by the evidence. Sanitizer builds, repeated runs, and debugger sessions can be expensive; consider the user's resource constraints. Remove temporary instrumentation introduced for diagnosis when finished.

---

## Hang Path

When a query never returns or the process appears stuck. Hangs are typically caused by deadlocks, infinite loops, missing CUDA stream synchronization, or a thread waiting on a condition that is never signaled.

### Phase 1: Reproduce and capture last-known state

1. Run the query with a timeout to confirm the hang:
   ```bash
   export SIRIUS_LOG_BACKEND=spdlog
   export SIRIUS_LOG_LEVEL=trace
   export SIRIUS_LOG_DIR=build/release/log/run_$(date +%s)
   mkdir -p "$SIRIUS_LOG_DIR"
   pixi run timeout <TIMEOUT> sirius-duckdb/build/release/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';" <db_path> <<'EOF'
   <QUERY>
   EOF
   ```
   - Exit code 124 means the timeout elapsed; confirm whether the query is hung or just slow
   - Trace logging may show the last recorded activity, but periodic flushing can leave recent entries unwritten.

2. Read the log file and identify the **last log entry** before the hang:
   - Which pipeline stage was active?
   - Which thread(s) logged last?
   - Was the last operation a lock acquisition, a CUDA kernel launch, a memory allocation, or a queue wait?

3. Look for patterns:
   - **Deadlock indicators:** Multiple threads each waiting on a lock/condition, no progress. Look for `mutex`, `lock`, `wait`, `condition_variable` in the last entries.
   - **Infinite loop indicators:** The same log entry repeated many times, or a loop counter growing without bound.
   - **CUDA sync hang:** Last entry is a `cudaStreamSynchronize`, `cudaDeviceSynchronize`, or `cudaEventSynchronize` -- the GPU kernel may be hung or a stream dependency is unsatisfied.
   - **Queue starvation:** Task Creator waiting for completed tasks, but no tasks are being submitted or completed (empty queue).

### Phase 2: Targeted debug logging

If Phase 1 narrows the hang to a code region but not the exact cause:

1. Insert `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] <location>: reached");` statements at:
   - Entry/exit of suspect functions
   - Before/after lock acquisitions and releases
   - Before/after CUDA synchronization calls
   - Inside loops (with iteration counter) to detect infinite loops
   - Task queue push/pop operations

2. Rebuild and re-run with timeout. Compare new logs against Phase 1 logs:
   - Which new log entries appear? Which don't?
   - If a "before lock" appears but "after lock" doesn't -> deadlock on that lock
   - If loop counter grows without bound -> infinite loop
   - If "before cudaSync" appears but "after cudaSync" doesn't -> GPU hang

3. Repeat as needed while the results narrow the investigation.

### Phase 3: cuda-gdb live attach

For deadlocks and thread-level hangs where log analysis is insufficient:

1. Install a `relwithdebinfo` or `clang-debug` library and rebuild the SQL wrapper
   as described in the shared build guide. Use its actual output directory below
   (`debug` for the shared guide example). Run the query **without** timeout:
   ```bash
   pixi run sirius-duckdb/build/<preset>/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/<preset>/extension/sirius/sirius.duckdb_extension';" <db_path> <<'EOF' &
   <QUERY>
   EOF
   HANG_PID=$!
   ```

2. Wait for it to hang. Check with `ps` and log activity, and identify the actual
   DuckDB process ID (the background Pixi process may be a wrapper), set `HANG_PID`
   to that PID, then attach:
   ```bash
   pixi run cuda-gdb -p $HANG_PID -batch \
     -ex "thread apply all bt" \
     -ex "info threads" \
     -ex quit
   ```

   If attaching is unavailable, launch the reproduction under `cuda-gdb --args`
   and interrupt it after it hangs to collect the backtraces.

3. Analyze the thread backtraces:
   - Identify threads blocked in `pthread_mutex_lock`, `pthread_cond_wait`, `futex`, or CUDA sync calls
   - Map thread IDs back to Sirius components (Thread Coordinator, Task Creator, Pipeline Executor, Scan Executor, GPU scheduling threads)
   - For deadlocks: identify the lock cycle (Thread A holds Lock 1, waits on Lock 2; Thread B holds Lock 2, waits on Lock 1)
   - For CUDA hangs: check if a GPU kernel is still running or if a stream is waiting on an event that was never recorded

4. Stop the hung process started for this investigation: `kill "$HANG_PID"`

### Hang analysis checklist

Common hang causes in Sirius:
- **Deadlock between Thread Coordinator and Task Creator:** Both waiting on each other's condition variables
- **GPU kernel infinite loop:** A CUDA kernel that doesn't terminate (e.g., hash table probe with broken termination condition)
- **Missing pipeline completion signal:** A pipeline finishes but doesn't notify the Task Creator, so it waits forever
- **Memory reservation deadlock:** Thread waiting for memory reservation while holding a lock that blocks the thread that would free memory
- **cuCascade eviction deadlock:** Eviction callback tries to acquire a lock already held by the requesting thread
- **Stream dependency cycle:** CUDA stream A waits on event from stream B, stream B waits on event from stream A

---

## Segfault Path

When the error is a segmentation fault (SIGSEGV) or bus error (SIGBUS). Sirius has a built-in backtrace handler (`src/util/segfault_backtrace_handler.cpp`) that automatically prints a demangled stack trace on crash, so the first step is always to capture that output.

### Phase 1: Capture the automatic backtrace

Sirius installs a signal handler (via `install_segfault_backtrace_handler()`) that catches SIGSEGV/SIGBUS and prints a demangled C++ backtrace to **stderr**. If `SIRIUS_LOG_DIR` is set, it also writes the backtrace to `$SIRIUS_LOG_DIR/segfault_backtrace.txt`.

1. Run the query with trace logging and capture stderr:
   ```bash
   export SIRIUS_LOG_BACKEND=spdlog
   export SIRIUS_LOG_LEVEL=trace
   export SIRIUS_LOG_DIR=build/release/log/run_$(date +%s)
   mkdir -p "$SIRIUS_LOG_DIR"
   pixi run sirius-duckdb/build/release/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';" <db_path> <<'EOF' 2>&1 | tee "$SIRIUS_LOG_DIR/segfault_output.txt"
   <QUERY>
   EOF
   ```

2. Read the backtrace output. Look for the block between `*** SIGSEGV — backtrace ***` and `*** end backtrace ***`. The output includes:
   - **Faulting thread ID** -- identifies which thread crashed
   - **Demangled stack frames** -- shows the full call chain at the crash point
   - The backtrace is also saved to `$SIRIUS_LOG_DIR/segfault_backtrace.txt`

3. Analyze the backtrace:
   - Identify the **top frames** in `namespace sirius` -- this is the crash location
   - Read the source file at that location to understand the crash context
   - Determine whether the crash is **CPU-side** or **GPU-side** based on the call chain
   - Cross-reference with known patterns in `common-segfaults.md`

4. Also read the trace log (`$SIRIUS_LOG_DIR/*.log`) to understand the execution path leading up to the crash -- which pipeline, operator, and data was being processed.

**If the backtrace clearly identifies the crash location**, continue with the crash analysis and report the diagnosis.

**If the backtrace is insufficient** (e.g., frames are in external libraries without symbols, the crash is in a GPU kernel, or the backtrace points to a generic location like a memory allocator), continue to Phase 1b.

### Phase 1b: Binary-search log insertion (when backtrace is insufficient)

When the automatic backtrace doesn't pinpoint the root cause:

- Insert `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] <location>: reached");` at strategic points around the area identified by the backtrace
- For data inspection at the suspected crash point, also insert debug utility calls:
  ```cpp
  sirius::debug_schema(*batch, stream);  // verify data shape before crash
  sirius::debug_head(*batch, 5, stream); // inspect actual values
  sirius::debug_nulls(*batch, stream);   // check for unexpected nulls
  ```
- **Binary search strategy:**
  1. Insert log statements at function entry points across the suspected code path
  2. Rebuild, run -- see which log entry was the last to appear before crash
  3. Insert more log statements within the identified function
  4. Rebuild, run -- narrow down further
  5. Repeat until the exact crashing line is identified
- Trace logs may help narrow the crash location; recent entries can be lost before a flush.

After Phase 1/1b, use the analysis to determine whether the crash is likely **CPU-side** or **GPU-side**, then branch:
- **CPU-side crash** (backtrace/last log entry is in CPU orchestration code, pipeline scheduling, memory management, etc.) -> **Phase 2a: ASan**
- **GPU-side crash** (backtrace/last log entry is before/during a CUDA kernel launch, cuDF call, or GPU memory operation) -> **Phase 2b: Compute Sanitizer**
- **Unclear** -> Try Phase 2a first (faster), then Phase 2b if ASan finds nothing

### Phase 2a: AddressSanitizer -- for CPU-side crashes

Use ASan when the crash appears to be a CPU-side memory error (buffer overflow, use-after-free, dangling pointer). ASan pinpoints the exact memory violation with stack traces for both the bad access and the original allocation.

**Note:** ASan only detects **CPU-side** memory errors. If the crash is GPU-side, skip to Phase 2b (Compute Sanitizer).

- Build the host-instrumented preset with `pixi run make clang-asan`.
- Run a relevant unit-test reproduction using `build/clang-asan/test/cpp/sirius_unittest`
  and the CUDA-compatible ASan options in `docs/super-sirius/debugging.md`. These
  commands do not verify instrumentation of the SQL wrapper; report that limit if
  the issue cannot be reproduced as a unit test.
- Inspect the bad-access and allocation stacks, then trace the relevant source.

### Phase 2b: NVIDIA Compute Sanitizer memcheck -- for GPU-side crashes

Use when Phase 1 points to a GPU memory error (crash during/after a CUDA kernel launch or cuDF operation). Compute Sanitizer catches the exact GPU memory violation with kernel name and line info.

A normal build can be used. `clang-debug` provides device line information; install
the library and rebuild the wrapper as described in the shared reference. Use the
actual wrapper directory. See `docs/super-sirius/debugging.md` for pooled-allocation
limitations:
```bash
pixi run compute-sanitizer --tool memcheck sirius-duckdb/build/<preset>/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/<preset>/extension/sirius/sirius.duckdb_extension';" <db_path> <<'EOF'
<QUERY>
EOF
```
Parse output for: out-of-bounds accesses, misaligned accesses, use-after-free on device memory.

### Phase 3: cuda-gdb

Fallback when ASan and Compute Sanitizer don't find the issue, or when you need a full backtrace / interactive stepping to understand the crash context.

- Build with debug symbols; choose the preset suited to the investigation:
  - `relwithdebinfo` (recommended) -- optimized with debug symbols, faster execution
  - `clang-debug` -- unoptimized, best for stepping through code line-by-line
- Install the chosen library and rebuild the wrapper as described in the shared
  reference. Use its actual build directory below and put the supplied SQL in
  an absolute-path query file:
  ```bash
  pixi run cuda-gdb --batch -ex run -ex bt -ex quit --args sirius-duckdb/build/<preset>/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/<preset>/extension/sirius/sirius.duckdb_extension';" <db_path> < /absolute/path/query.sql
  ```
- Parse the backtrace to identify the exact file, line, and call stack
- For GPU-side crashes, cuda-gdb can inspect device threads and shared memory

### Crash analysis checklist

- Read the source file at the crash point
- Check for common causes:
  - Null pointer dereference
  - Dangling reference (object lifetime issue)
  - Buffer overflow / out-of-bounds access
  - Use-after-free
  - Iterator invalidation
  - GPU memory access violation
  - cuDF column lifetime issues (column destroyed while view still active)
  - DuckDB vector invalidation (vector recycled during scan)
- Trace the data flow to find where the invalid state originated
- Cross-reference with known patterns in `common-segfaults.md` in this directory

---

## Runtime Error Path

When the query produces an exception, error message, unexpected fallback to CPU, or a non-zero exit code without a segfault. This covers C++ exceptions, DuckDB errors, CUDA runtime errors (`cudaError`), cuDF exceptions, and Sirius fallback triggers.

### Phase 1: Log analysis

1. Run the query with trace logging to capture the full execution path:
   ```bash
   export SIRIUS_LOG_BACKEND=spdlog
   export SIRIUS_LOG_LEVEL=trace
   export SIRIUS_LOG_DIR=build/release/log/run_$(date +%s)
   mkdir -p "$SIRIUS_LOG_DIR"
   pixi run sirius-duckdb/build/release/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';" <db_path> <<'EOF'
   <QUERY>
   EOF
   ```
   Capture both stdout/stderr output AND the log file.

2. Read the log file and analyze:
   - **Find the error message** in both stderr and the log. Search for `[error]`, `[critical]`, `Exception`, `Error`, `CUDA error`, `Fallback`.
   - **Trace the execution path** leading up to the error:
     - Which pipelines were created?
     - Which operators were executed before the error?
     - What data was being processed (table names, row counts, column types)?
   - **Identify the error category:**
     - **C++ exception:** `std::runtime_error`, `std::logic_error`, etc. -- look for the throw site in the log/stderr
     - **CUDA runtime error:** `cudaErrorXxx` -- indicates GPU-side failure (OOM, invalid config, driver error)
     - **cuDF exception:** `cudf::logic_error`, `cudf::cuda_error` -- cuDF operation failed
     - **DuckDB error:** Error propagated from DuckDB's execution engine
     - **Sirius fallback:** Query or operator fell back to CPU execution -- search log for `fallback` entries. Search the relevant fallback path to understand which condition triggered it.
     - **Assertion failure:** `assert()` or `DCHECK` failure -- the code hit an unexpected state

3. Read the source file at the error location. Understand the function, the expected preconditions, and what state would cause the error.

### Phase 2: Targeted debug logging

If Phase 1 identifies the general area but not the root cause:

1. Insert `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] ...")` statements to capture:
   - Variable values at the error site (function arguments, object state)
   - Conditional branch outcomes leading to the error (which `if` path was taken?)
   - Data characteristics: row counts, column types, null counts, string lengths
   - Memory state: allocation sizes, reservation amounts, available GPU memory
   - Data state at the error site via debug utilities:
     ```cpp
     sirius::debug_schema(*batch, stream);
     sirius::debug_head(*batch, 10, stream);
     sirius::debug_nulls(*batch, stream);
     ```

2. Rebuild and re-run with a new log directory. Compare against Phase 1 logs.

3. Narrow the scope as needed:
   - Iteration 1: Function-level -- which function is the error in?
   - Iteration 2: Block-level -- which code block within the function?
   - Iteration 3: Line-level -- which exact line and what variable values?

### Phase 3: cuda-gdb

For errors that are hard to reproduce or where the exception obscures the true origin:

- Build with debug symbols; choose the preset suited to the investigation:
  - `relwithdebinfo` (recommended) -- optimized with debug symbols, faster reproduction
  - `clang-debug` -- unoptimized, best for stepping through code line-by-line
- Install the chosen library and rebuild the wrapper as described in the shared
  reference. Use its actual build directory and an absolute-path query file to
  catch the exception at throw time:
  ```bash
  pixi run cuda-gdb --batch \
    -ex "catch throw" \
    -ex run \
    -ex bt \
    -ex quit \
    --args sirius-duckdb/build/<preset>/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/<preset>/extension/sirius/sirius.duckdb_extension';" <db_path> < /absolute/path/query.sql
  ```
  This catches C++ exceptions at the throw site, before unwinding obscures the call stack.

### Runtime error analysis checklist

Common runtime error causes in Sirius:
- **Unsupported data type:** A column type (e.g., `INTERVAL`, `BLOB`, nested types) reached a GPU operator that doesn't handle it. Should trigger fallback but may throw instead.
- **cuDF operation failure:** cuDF throws on invalid input (e.g., mismatched column lengths in join, unsupported aggregation type). Read the cuDF error message carefully -- it usually says exactly what's wrong.
- **GPU OOM:** `cudaErrorMemoryAllocation` -- the GPU ran out of memory. Check memory reservation logic, consider reducing data batch size or GPU region sizes.
- **Unexpected fallback:** A query falls back to CPU when the user expected GPU execution. Inspect the relevant fallback path for its condition. Common triggers: unsupported operator, data too large, unsupported type.
- **Type mismatch:** Sirius type conversion (DuckDB types <-> cuDF types) failed. Inspect the relevant type mapping and conversion code.
- **Pipeline dependency error:** A pipeline tried to read from a Data Repository that hasn't been populated yet (dependency ordering issue in `sirius_meta_pipeline`).
- **Row count overflow:** cuDF uses `int32_t` for row indices (~2B row limit). Large tables can overflow. Should trigger fallback.

---

## Fixes and diagnostic logs

Explain the root cause and proposed fix. If fixing is within the requested scope,
apply it, rebuild, and rerun the relevant reproduction. Report the outcome and
any remaining uncertainty. Preserve pre-existing changes and remove temporary
instrumentation introduced for the investigation; retain useful logging only
when appropriate to the requested change.

---

## Debug Utilities

Sirius provides structured debug utility functions in `src/debug_utils.hpp` (implementation in `src/debug_utils.cpp`). Use these for data inspection at suspected fault points instead of writing ad-hoc log statements. All functions output to `SIRIUS_LOG_DEBUG` with `[SIRIUS_DIAG]` prefix, are thread-safe (buffered single-call output), and are wrapped in try/catch (never crash the pipeline).

### Function Signatures

```cpp
#include "debug_utils.hpp"

// Schema metadata: column names, types, null counts, row count
sirius::debug_schema(batch, stream);
sirius::debug_schema(batch, stream, {"col_a", "col_b"});

// Per-column null counts and percentages (zero GPU cost)
sirius::debug_nulls(batch, stream);

// First N rows in aligned-column or CSV format
sirius::debug_head(batch, /*n=*/10, stream);
sirius::debug_head(batch, /*n=*/5, stream, sirius::DebugFormat::CSV);

// N randomly selected rows (catches bugs not in first rows)
sirius::debug_sample(batch, /*n=*/10, stream);

// Per-column min, max, sum for numeric columns (GPU-side)
sirius::debug_stats(batch, stream);

// Per-column xxhash_64 fingerprint
sirius::debug_checksum(batch, stream);

// Compare two batches: schema, row count, per-column value diff
sirius::debug_diff(batch_a, batch_b, stream);
```

### Usage for Runtime Error Diagnosis

**At suspected fault points** (before/after the crashing operator):
```cpp
// Inspect data shape and types at the fault point:
sirius::debug_schema(*batch, stream);
sirius::debug_nulls(*batch, stream);

// Inspect actual data values:
sirius::debug_head(*batch, 10, stream);

// Check for unexpected nulls or data corruption:
sirius::debug_stats(*batch, stream);
```

**When narrowing down a crash to an operator:**
```cpp
// Insert at operator input and output:
sirius::debug_schema(*input_batch, stream);   // verify input shape
sirius::debug_head(*input_batch, 5, stream);  // check input values
// ... operator executes ...
sirius::debug_schema(*output_batch, stream);  // verify output shape
sirius::debug_head(*output_batch, 5, stream); // check output values
```

**Key benefits over ad-hoc logging:**
- Thread-safe: output buffered into single log call (no interleaving)
- Null-aware: NULL values displayed explicitly, not as garbage
- Try/catch wrapped: debug call never crashes the pipeline (even if the data is corrupted)
- Supports all Sirius types: INT, FLOAT, STRING, DECIMAL, TIMESTAMP, DATE

---

## Known Segfault Patterns

See `common-segfaults.md` in this directory for a catalog of known segfault patterns and their fixes.

## Scope

Analyze active code in both `namespace sirius` and `namespace duckdb`; see shared build-and-query.md.
