---
name: runtime-errors
description: Diagnose runtime errors using Sirius log files, cuda-gdb, and Compute Sanitizer. Use when a query crashes (including segfaults), throws exceptions, hangs, or triggers unexpected fallback to CPU.
argument-hint: [sql-query-or-file] [--timeout 30]
disable-model-invocation: true
---

# Runtime Error Analyzer

Diagnose runtime errors (crashes, segfaults, exceptions, hangs, fallback triggers) by analyzing log files, inserting targeted debug logs, using cuda-gdb and NVIDIA Compute Sanitizer, and re-running queries.

**Reference:** See `.claude/skills/_shared/build-and-query.md` for shared infrastructure (build modes, query execution, autonomy mode, change tracking, debug log conventions).

## Triage

1. **Gather context:**
   - Ask the user for the SQL query (or accept via `$ARGUMENTS`) and the error description
   - Ask about data format (DuckDB or Parquet) -- see shared build-and-query.md
   - Parse optional `--timeout` argument (default: 30 seconds)
   - Determine autonomy mode: `interactive` (default), `autonomous`, or `semi-autonomous`

2. **Classify the error type** to determine which workflow path to follow:
   - **Hang / timeout** (query never returns, process stuck) -> **Hang Path**
   - **Segmentation fault (SIGSEGV)** (process killed by signal 11) -> **Segfault Path**
   - **Runtime error** (exception, error message, unexpected fallback, wrong exit code) -> **Runtime Error Path**

Each path below is self-contained with its own phased workflow. All paths share the same fix iteration and log cleanup steps at the end.

---

## Hang Path

When a query never returns or the process appears stuck. Hangs are typically caused by deadlocks, infinite loops, missing CUDA stream synchronization, or a thread waiting on a condition that is never signaled.

### Phase 1: Reproduce and capture last-known state

1. Run the query with a timeout to confirm the hang:
   ```bash
   export SIRIUS_LOG_LEVEL=trace
   export SIRIUS_LOG_DIR=build/release/log/run_$(date +%s)
   mkdir -p $SIRIUS_LOG_DIR
   timeout <TIMEOUT> build/release/duckdb <db_path> <<'EOF'
   CALL gpu_execution('<QUERY>');
   EOF
   ```
   - If exit code is 124, the process was killed by timeout -- confirmed hang
   - `SIRIUS_LOG_LEVEL=trace` triggers `spdlog::flush_on(trace)` so all log entries before the hang are flushed

2. Read the log file and identify the **last log entry** before the hang:
   - Which pipeline stage was active?
   - Which thread(s) logged last?
   - Was the last operation a lock acquisition, a CUDA kernel launch, a memory allocation, or a queue wait?

3. Look for patterns:
   - **Deadlock indicators:** Multiple threads each waiting on a lock/condition, no progress. Look for `mutex`, `lock`, `wait`, `condition_variable` in the last entries.
   - **Infinite loop indicators:** The same log entry repeated many times, or a loop counter growing without bound.
   - **CUDA sync hang:** Last entry is a `cudaStreamSynchronize`, `cudaDeviceSynchronize`, or `cudaEventSynchronize` -- the GPU kernel may be hung or a stream dependency is unsatisfied.
   - **Queue starvation:** Task Creator waiting for completed tasks, but no tasks are being submitted or completed (empty queue).

### Phase 2: Targeted debug logging (ask user before proceeding)

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

3. Repeat up to 3 iterations, narrowing the scope each time.

### Phase 3: cuda-gdb live attach (ask user before proceeding)

For deadlocks and thread-level hangs where log analysis is insufficient:

1. Run the query **without** timeout (let it hang). Use `relwithdebinfo` or `clang-debug`:
   ```bash
   build/<preset>/duckdb <db_path> <<'EOF' &
   CALL gpu_execution('<QUERY>');
   EOF
   HANG_PID=$!
   ```

2. Wait for it to hang (check with `ps` and log activity), then attach:
   ```bash
   cuda-gdb -p $HANG_PID -batch \
     -ex "thread apply all bt" \
     -ex "info threads" \
     -ex quit
   ```

3. Analyze the thread backtraces:
   - Identify threads blocked in `pthread_mutex_lock`, `pthread_cond_wait`, `futex`, or CUDA sync calls
   - Map thread IDs back to Sirius components (Thread Coordinator, Task Creator, Pipeline Executor, Scan Executor, GPU scheduling threads)
   - For deadlocks: identify the lock cycle (Thread A holds Lock 1, waits on Lock 2; Thread B holds Lock 2, waits on Lock 1)
   - For CUDA hangs: check if a GPU kernel is still running or if a stream is waiting on an event that was never recorded

4. Kill the hung process: `kill $HANG_PID`

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
   export SIRIUS_LOG_LEVEL=trace
   export SIRIUS_LOG_DIR=build/release/log/run_$(date +%s)
   mkdir -p $SIRIUS_LOG_DIR
   build/release/duckdb <db_path> <<'EOF' 2>&1 | tee /tmp/segfault_output.txt
   CALL gpu_execution('<QUERY>');
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

**If the backtrace clearly identifies the crash location**, skip to the Crash Analysis Checklist below and then to the Fix Iteration section.

**If the backtrace is insufficient** (e.g., frames are in external libraries without symbols, the crash is in a GPU kernel, or the backtrace points to a generic location like a memory allocator), continue to Phase 1b.

### Phase 1b: Binary-search log insertion (when backtrace is insufficient)

When the automatic backtrace doesn't pinpoint the root cause:

- Insert `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] <location>: reached");` at strategic points around the area identified by the backtrace
- **Binary search strategy:**
  1. Insert log statements at function entry points across the suspected code path
  2. Rebuild, run -- see which log entry was the last to appear before crash
  3. Insert more log statements within the identified function
  4. Rebuild, run -- narrow down further
  5. Repeat until the exact crashing line is identified
- `SIRIUS_LOG_LEVEL=trace` triggers `spdlog::flush_on(trace)` ensuring no log data is lost before crash (see "Why Immediate Flush Matters" below)

After Phase 1/1b, use the analysis to determine whether the crash is likely **CPU-side** or **GPU-side**, then branch:
- **CPU-side crash** (backtrace/last log entry is in CPU orchestration code, pipeline scheduling, memory management, etc.) -> **Phase 2a: ASan**
- **GPU-side crash** (backtrace/last log entry is before/during a CUDA kernel launch, cuDF call, or GPU memory operation) -> **Phase 2b: Compute Sanitizer**
- **Unclear** -> Try Phase 2a first (faster), then Phase 2b if ASan finds nothing

### Phase 2a: AddressSanitizer -- for CPU-side crashes (ask user before proceeding)

Summarize Phase 1 findings. Offer ASan when the crash appears to be a CPU-side memory error (buffer overflow, use-after-free, dangling pointer). ASan pinpoints the exact memory violation with stack traces for both the bad access and the original allocation.

**Note:** ASan only detects **CPU-side** memory errors. If the crash is GPU-side, skip to Phase 2b (Compute Sanitizer).

- Build with `clang-debug` (ASan is on by default in Debug builds):
  ```bash
  CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make clang-debug
  ```
- Run the reproduction case:
  ```bash
  export SIRIUS_LOG_LEVEL=trace
  export SIRIUS_LOG_DIR=build/clang-debug/log/run_$(date +%s)
  mkdir -p $SIRIUS_LOG_DIR
  ASAN_OPTIONS="detect_leaks=1:halt_on_error=1" build/clang-debug/duckdb <db_path> <<'EOF'
  CALL gpu_execution('<QUERY>');
  EOF
  ```
- Parse ASan output. ASan reports include:
  - **Error type:** heap-buffer-overflow, stack-buffer-overflow, heap-use-after-free, double-free, etc.
  - **Bad access location:** file, line, and full stack trace of the invalid read/write
  - **Allocation context:** where the memory was originally allocated (and freed, for use-after-free)
- Read the source files at both the access and allocation locations
- Trace the data flow to understand how the invalid state arose

**ASan overhead:** ~2x slowdown, ~2-3x memory. See `_shared/build-and-query.md` for full ASan configuration details.

### Phase 2b: NVIDIA Compute Sanitizer memcheck -- for GPU-side crashes (ask user before proceeding)

Use when Phase 1 points to a GPU memory error (crash during/after a CUDA kernel launch or cuDF operation). Compute Sanitizer catches the exact GPU memory violation with kernel name and line info.

Build with debug symbols (`clang-debug` or `relwithdebinfo`):
```bash
compute-sanitizer --tool memcheck build/<preset>/duckdb <<'EOF'
CALL gpu_execution('<QUERY>');
EOF
```
Parse output for: out-of-bounds accesses, misaligned accesses, use-after-free on device memory.

### Phase 3: cuda-gdb (ask user before proceeding)

Fallback when ASan and Compute Sanitizer don't find the issue, or when you need a full backtrace / interactive stepping to understand the crash context.

- Build with debug symbols. Ask the user which preset to use:
  - `relwithdebinfo` (recommended) -- optimized with debug symbols, faster execution
  - `clang-debug` -- unoptimized, best for stepping through code line-by-line
  ```bash
  CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make relwithdebinfo
  # or for full debug:
  CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make clang-debug
  ```
- Run with cuda-gdb (replace `<preset>` with chosen preset):
  ```bash
  cuda-gdb --batch -ex run -ex bt -ex quit --args build/<preset>/duckdb
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
   export SIRIUS_LOG_LEVEL=trace
   export SIRIUS_LOG_DIR=build/release/log/run_$(date +%s)
   mkdir -p $SIRIUS_LOG_DIR
   build/release/duckdb <db_path> <<'EOF'
   CALL gpu_execution('<QUERY>');
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
     - **Sirius fallback:** Query or operator fell back to CPU execution -- search log for `fallback` entries. Check `src/fallback.cpp` to understand which condition triggered it.
     - **Assertion failure:** `assert()` or `DCHECK` failure -- the code hit an unexpected state

3. Read the source file at the error location. Understand the function, the expected preconditions, and what state would cause the error.

### Phase 2: Targeted debug logging (ask user before proceeding)

If Phase 1 identifies the general area but not the root cause:

1. Insert `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] ...")` statements to capture:
   - Variable values at the error site (function arguments, object state)
   - Conditional branch outcomes leading to the error (which `if` path was taken?)
   - Data characteristics: row counts, column types, null counts, string lengths
   - Memory state: allocation sizes, reservation amounts, available GPU memory

2. Rebuild and re-run with a new log directory. Compare against Phase 1 logs.

3. Repeat up to 3 iterations, each time narrowing the scope:
   - Iteration 1: Function-level -- which function is the error in?
   - Iteration 2: Block-level -- which code block within the function?
   - Iteration 3: Line-level -- which exact line and what variable values?

### Phase 3: cuda-gdb (ask user before proceeding)

For errors that are hard to reproduce or where the exception obscures the true origin:

- Build with debug symbols. Ask the user which preset:
  - `relwithdebinfo` (recommended) -- optimized with debug symbols, faster reproduction
  - `clang-debug` -- unoptimized, best for stepping through code line-by-line
- Run with cuda-gdb to catch the exception at throw time (replace `<preset>` with chosen preset):
  ```bash
  cuda-gdb --batch \
    -ex "catch throw" \
    -ex run \
    -ex bt \
    -ex quit \
    --args build/<preset>/duckdb
  ```
  This catches C++ exceptions at the throw site, before unwinding obscures the call stack.
- For CUDA errors, set a breakpoint on the CUDA error handler:
  ```bash
  cuda-gdb --batch \
    -ex "break cudaGetLastError" \
    -ex run \
    -ex bt \
    -ex quit \
    --args build/<preset>/duckdb
  ```

### Runtime error analysis checklist

Common runtime error causes in Sirius:
- **Unsupported data type:** A column type (e.g., `INTERVAL`, `BLOB`, nested types) reached a GPU operator that doesn't handle it. Should trigger fallback but may throw instead.
- **cuDF operation failure:** cuDF throws on invalid input (e.g., mismatched column lengths in join, unsupported aggregation type). Read the cuDF error message carefully -- it usually says exactly what's wrong.
- **GPU OOM:** `cudaErrorMemoryAllocation` -- the GPU ran out of memory. Check memory reservation logic, consider reducing data batch size or GPU region sizes.
- **Unexpected fallback:** A query falls back to CPU when the user expected GPU execution. Check `src/fallback.cpp` for the fallback condition. Common triggers: unsupported operator, data too large, unsupported type.
- **Type mismatch:** Sirius type conversion (DuckDB types <-> cuDF types) failed. Check type mapping in `src/include/cudf/` and `src/include/data/`.
- **Pipeline dependency error:** A pipeline tried to read from a Data Repository that hasn't been populated yet (dependency ordering issue in `sirius_meta_pipeline`).
- **Row count overflow:** cuDF uses `int32_t` for row indices (~2B row limit). Large tables can overflow. Should trigger fallback.

---

## Shared: Fix Iteration

After identifying the root cause (from any path above):

1. **Suggest a fix** with explanation of why it resolves the issue.

2. **Apply and verify** (behavior depends on autonomy mode):
   - Apply the fix (with git checkpoint)
   - Rebuild and re-run the reproduction case
   - Verify the error no longer occurs
   - If the error persists or a new error appears, repeat the relevant path
   - Optionally compare output against DuckDB CPU baseline to ensure correctness
   - Continue until: issue resolved, max iterations (default: 5) reached, or user intervenes

3. **Present final summary:** all attempted fixes, which worked, which didn't.

## Shared: Log Management Summary

At the end of any path, present which `[SIRIUS_DIAG]` log statements should be:
- **Promoted:** Useful long-term -- change from `[SIRIUS_DIAG]` tag to a proper log message at appropriate level (`SIRIUS_LOG_DEBUG`, `SIRIUS_LOG_INFO`, etc.)
- **Removed:** Only useful for this investigation -- delete entirely

---

## Why Immediate Flush Matters

By default, spdlog flushes every 3 seconds (see `SIRIUS_LOG_FLUSH_SEC` in `src/include/log/logging.hpp`). When a segfault or hang occurs, buffered entries are lost. Setting `SIRIUS_LOG_LEVEL=trace` explicitly triggers `spdlog::flush_on(trace)`, which flushes after every log entry -- ensuring the last log before the crash is always visible.

## Known Segfault Patterns

See `common-segfaults.md` in this directory for a catalog of known segfault patterns and their fixes.

## Scope

Only analyze code in `namespace sirius` plus exceptions listed in shared build-and-query.md. Ignore legacy `namespace duckdb` code.
