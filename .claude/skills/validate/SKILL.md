---
name: validate
description: Diagnose incorrect query results by comparing against DuckDB CPU, analyzing per-operator row counts and data checksums to pinpoint the faulty operator. Use when a query returns wrong results.
argument-hint: [sql-query-or-file]
disable-model-invocation: true
---

# Validation Error Analyzer

Diagnose incorrect query results by comparing Sirius GPU output against DuckDB CPU baseline, identifying the faulty operator, and narrowing down the root cause.

**Reference:** See `.claude/skills/_shared/build-and-query.md` for shared infrastructure (build modes, query execution, result comparison, autonomy mode, change tracking, debug log conventions).

## Workflow

1. **Gather context:**
   - Ask the user for the SQL query (or accept via `$ARGUMENTS`)
   - Ask about data format (DuckDB or Parquet)
   - Determine autonomy mode: `interactive` (default), `autonomous`, or `semi-autonomous`

2. **Establish baseline:**
   Run the query via DuckDB CPU to get the expected correct result:
   ```bash
   build/release/duckdb <db_path> -c "SELECT ..." > /tmp/claude-1000/baseline_result.txt
   ```

3. **Run via Sirius GPU:**
   ```bash
   export SIRIUS_LOG_LEVEL=trace
   export SIRIUS_LOG_DIR=build/release/log/run_$(date +%s)
   mkdir -p $SIRIUS_LOG_DIR
   build/release/duckdb <db_path> -c "CALL gpu_execution('SELECT ...');" > /tmp/claude-1000/gpu_result.txt
   ```
   Compare output against baseline (sort both to handle ordering differences).

4. **Handle inconsistent results:**
   If results match on first run but user reports inconsistency:
   - Run the query repeatedly (up to 10 times) until a wrong result appears
   - Record both good run and bad run logs for comparison
   - Flag as potential race condition

5. **Phase 1: Row count analysis with `tools/parse_pipeline_log.py`**
   ```bash
   python3 tools/parse_pipeline_log.py $SIRIUS_LOG_DIR/sirius_*.log
   ```
   - Run on both good and bad run logs
   - Compare per-operator row counts between runs
   - If a mismatch is found, the diverging operator is the likely culprit -- report it

6. **Phase 2: Data validation** (ask user before proceeding)
   If row counts match but results differ, the issue is in data values:
   - Insert diagnostic logging at operator boundaries to compute data checksums:
     - `sum()` of each numeric column
     - `max()` of each numeric column
     - `head(1)` -- first row sample
   - Use `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] operator_name checksum: sum={}, max={}, first_row={}", ...)` format
   - Rebuild and re-run, comparing checksums between GPU and CPU (or good/bad runs)
   - Narrow down to the specific operator where checksums first diverge

7. **Phase 3: Deep dive into faulty operator** (ask user before proceeding)
   Once the faulty operator is identified:
   - Read its implementation (both the `.cpp` and `.cu` files)
   - Add more granular logging/print statements inside the operator
   - Rebuild and re-run to understand exactly where data goes wrong
   - Suggest a fix

8. **Iterative fix loop** (behavior depends on autonomy mode):
   - Apply the fix, rebuild, and verify against DuckDB CPU baseline
   - If still wrong, repeat the analysis
   - Continue until results match or max iterations reached

## Common Validation Error: Stream Synchronization / Garbage Data

The most common cause of wrong results in Sirius is **reading garbage data due to CUDA stream synchronization issues**. This happens when an operator's output is read before the GPU kernel writing it has finished, because the kernel ran on a different stream than expected.

**Root cause:** Sirius uses a stream-per-thread model, but some cuDF operations internally use `cudf::default_stream()` instead of the caller's CUDA stream. When Sirius calls `stream.synchronize()` on its own stream, this does NOT synchronize `cudf::default_stream()`, so data written by cuDF on the default stream may not be visible yet -- resulting in garbage or stale data being read.

**Reference commit:** `69e4c6cf` fixed instances of this pattern where `cudf::default_stream()` was accidentally used.

**Typical symptoms:**
- Data looks like **garbage**: very large numbers, zeros, or partial/stale results rather than logically wrong values
- Wrong results that are **intermittent** (sometimes correct, sometimes wrong)
- Results that change between runs with no code changes
- **More likely to reproduce at larger batch sizes** (GBs) -- small data may complete before the race window opens
- Results that become correct when GPU concurrency is reduced (fewer threads)

**How to diagnose:**

1. **Quick check with `stream_check`:** Use the `stream_check` LD_PRELOAD library (`utils/stream_check/`) to detect default stream usage at runtime. It intercepts `cudf::get_default_stream()` and logs a full stack trace whenever the default stream is accessed from a monitored thread.

   Build and run:
   ```bash
   # Build with stream check enabled
   CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) cmake --build build/release --target stream_check

   # Run the query with stream_check preloaded
   LD_PRELOAD=build/release/libstream_check.so build/release/duckdb <db_path> <<'EOF'
   CALL gpu_execution('<QUERY>');
   EOF
   ```

   Check the output in `default_stream_traces.log`. Each entry shows the **full call stack** where `cudf::get_default_stream()` was called -- these are the code paths that need to use the correct per-thread stream instead.

   **Note:** `stream_check` is already integrated into Sirius via `src/util/stream_check_wrapper.cpp`. The GPU pipeline executor threads automatically enable detection. If `libstream_check.so` is not preloaded, the wrapper gracefully no-ops.

2. **Check with nsys:** Profile the query with `nsys` and examine the stream IDs:
   ```bash
   nsys profile --stats=true -o /tmp/claude-1000/validate_profile build/<preset>/duckdb <db_path> <<'EOF'
   CALL gpu_execution('<QUERY>');
   EOF
   ```
   Look at the CUDA stream IDs in the trace. If `cudf::default_stream` (typically stream 0 or the per-thread default stream) appears where it shouldn't, that's the smoking gun.

3. **Narrow to the faulty operator:** Use Phase 2 (data checksums) to identify which operator produces the first mismatch. Print `sum()`/`max()`/`head(10)` of each operator's output and compare against the correct run.

4. **Confirm with `cudaDeviceSynchronize()`:** Once the faulty operator is identified, insert `cudaDeviceSynchronize()` calls inside that operator -- before reads and after writes:
   ```cpp
   cudaDeviceSynchronize(); // [SIRIUS_DIAG] sync before read
   // ... the suspected read/write ...
   cudaDeviceSynchronize(); // [SIRIUS_DIAG] sync after write
   ```
   - If the wrong result **disappears** with `cudaDeviceSynchronize()`, the issue is confirmed as a stream sync problem.
   - Then narrow down: remove `cudaDeviceSynchronize()` calls one by one to find the exact operation that needs proper stream synchronization.

5. **Fix:** Replace `cudf::default_stream()` with the correct per-thread CUDA stream in the faulty code path. Or ensure the operation explicitly synchronizes the stream it actually uses. **Never leave `cudaDeviceSynchronize()` in production code** -- it serializes all GPU work and destroys performance. It is only a diagnostic tool.

## Key Design Decisions

- `tools/parse_pipeline_log.py` is the first-line tool for row count comparison -- fast and non-invasive
- Data checksum insertion uses `[SIRIUS_DIAG]` tagged log statements for easy cleanup
- All changes tracked via git checkpoints for easy revert
- For inconsistent results, the skill has patience to run many times until it catches a bad result
- Each phase requires user confirmation before proceeding
- Stream sync issues are the #1 cause of validation errors -- always consider this first when results are intermittently wrong or contain garbage data
- `stream_check` (LD_PRELOAD library in `utils/stream_check/`) is the fastest way to find default stream usage -- run it before inserting manual debug logs

## Scope

Only analyze code in `namespace sirius` plus exceptions listed in shared build-and-query.md. Ignore legacy `namespace duckdb` code.
