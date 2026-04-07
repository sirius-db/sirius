# Pitfalls Research

**Domain:** GPU SQL engine debugging utilities (CUDA/cuDF data inspection)
**Researched:** 2026-04-06
**Confidence:** HIGH — based on direct analysis of `src/cuda/print.cu`, `src/include/log/logging.hpp`, `src/include/pipeline/gpu_pipeline_task.hpp`, and the wider codebase.

---

## Critical Pitfalls

### Pitfall 1: Output Goes to stdout, Not the Log Pipeline

**What goes wrong:**
The existing `print_table_contents()` and `print_data_batch_contents()` use `std::printf()` directly. Output goes to stdout, which is invisible in production deployments (no terminal attached), bypasses the spdlog daily-file sink, and is completely missed by the `/validate` and `/runtime-errors` skills that grep log files for `[SIRIUS_DIAG]`.

**Why it happens:**
`std::printf` is the path of least resistance when you just want to see a value during development. It also avoids the complication that `logging.hpp` no-ops under `__CUDACC__`, so developers working in `.cu` files reach for printf because logging literally does nothing there.

**How to avoid:**
All debug utility output must go through `SIRIUS_LOG_DEBUG` or `SIRIUS_LOG_TRACE` macros. Because `logging.hpp` is a no-op under `__CUDACC__`, the debug utility implementation must live in a `.cpp` file (not `.cu`). GPU→host data copies are done in the `.cpp` layer; only raw CUDA kernels (if any are needed at all) go in `.cu` files. The existing `print.cu` is the wrong pattern — new utilities should be in `print.cpp` or `debug_utils.cpp`.

**Warning signs:**
- Any `std::printf`, `fprintf`, `std::cout`, or `std::cerr` in print/debug files
- A `.cu` file calling `SIRIUS_LOG_*` (compiles but silently does nothing)
- Skills reporting "no `[SIRIUS_DIAG]` lines found" when debug calls are present

**Phase to address:**
Phase 1 (core infrastructure) — establish that all output paths go through `SIRIUS_LOG_DEBUG` before building any feature on top.

---

### Pitfall 2: Global cudaDeviceSynchronize() Stalls the Entire Pipeline

**What goes wrong:**
`print_table_contents()` calls `cudaDeviceSynchronize()` at the top of the function. In Sirius, multiple `gpu_pipeline_task` instances execute concurrently on separate CUDA streams (each task receives an `rmm::cuda_stream_view stream` parameter). A `cudaDeviceSynchronize()` blocks the calling CPU thread until ALL outstanding work on ALL streams on the device completes — not just the work relevant to the batch being inspected. In a pipeline with parallel tasks, this serializes every task that happens to call a debug utility, destroying concurrency and potentially causing incorrect timing behavior or test flakes.

**Why it happens:**
Developers reach for `cudaDeviceSynchronize()` because it's the obvious way to ensure GPU work is complete before reading results back to host. The stream-aware alternative (`stream.synchronize()` or `cudaStreamSynchronize(stream)`) requires knowing which stream the relevant GPU work was submitted on, and that information is not currently threaded into `print_table_contents`.

**How to avoid:**
Debug utilities must accept the relevant `rmm::cuda_stream_view` and call `stream.synchronize()` (or the equivalent `cudaStreamSynchronize(stream.value())`) rather than `cudaDeviceSynchronize()`. The stream is available in `gpu_pipeline_task::execute(rmm::cuda_stream_view stream)` and must be passed through to any debug utility called from that context. If the batch was produced by GPU operations submitted on a specific stream, only that stream needs synchronization.

**Warning signs:**
- `cudaDeviceSynchronize()` anywhere in print or debug utility code
- Debug utilities that don't take a `stream` parameter
- Noticeably slower query execution when debug logging is enabled

**Phase to address:**
Phase 1 (core infrastructure) — get the function signature right from the start. Retrofitting stream parameters later requires touching every call site.

---

### Pitfall 3: Null Values Print as Garbage (Null Mask Not Read)

**What goes wrong:**
cuDF columns have an optional validity bitmask (`null_mask`). When a column has nulls, the data buffer at null positions contains undefined memory — whatever was left there from a previous allocation. The existing `print_column_values_signed/unsigned/float/double` functions copy raw data values and print them without consulting the null mask. Null positions appear as valid, potentially meaningful numbers. This produces misleading output during debugging: an operator that produces correct nulls looks like it's producing wrong non-null values.

**Why it happens:**
The cuDF C++ API separates data pointers (`col.data<T>()`) from validity bitmasks (`col.null_mask()`). The data copy is straightforward; null mask handling requires an extra bitmask copy and per-element bit testing (`cudf::bit_is_set(bitmask, i)`), which is easy to skip.

**How to avoid:**
For each column, check `col.nullable()`. If true, copy the null mask alongside the data: `cudaMemcpy(host_mask.data(), col.null_mask(), cudf::bitmask_allocation_size_bytes(n), cudaMemcpyDeviceToHost)`. Then for each row index `i`, test `cudf::bit_is_set(host_mask.data(), i)` before printing the value; print "NULL" if false. The null mask covers the entire column, not just the first `n` rows, so copy `cudf::bitmask_allocation_size_bytes(col.size())` bytes (not `n` bytes) or compute the correct byte count for `n` rows.

**Warning signs:**
- Any column printing that does not also copy and consult `col.null_mask()`
- `debug_nulls()` or `debug_stats()` producing null counts that don't match `debug_head()` output

**Phase to address:**
Phase 1 (core infrastructure) — null handling must be in the base copy-to-host helper. Every formatter built on top inherits correct null behavior automatically.

---

### Pitfall 4: STRING Columns Require Separate Offsets + Chars Copies

**What goes wrong:**
cuDF STRING columns (type_id `STRING`) are not laid out as a single contiguous array of fixed-size values. They use a two-buffer representation: an offsets array (`int32_t[]`, length `n+1`) and a characters array (`char[]`). Calling `col.data<char>()` gives the chars buffer; `col.child(0)` gives the offsets column. Treating a STRING column like a numeric column — copying `n * sizeof(T)` bytes starting at `col.data<T>()` — reads the wrong memory and produces garbage or a segfault. The existing `print.cu` hits the `default:` case and prints "(unprinted type)" rather than attempting this incorrectly, but this still means STRING data is completely invisible in debug output.

**Why it happens:**
STRING is not a fixed-width type. Developers unfamiliar with the cuDF string storage model assume `col.data<T>()` points to string data directly. The correct API is `cudf::strings_column_view(col)` which exposes `.offsets()` and `.chars()` (pre-cuDF 24.x) or `.chars_begin(stream)`/`.chars_end(stream)` (post-24.x).

**How to avoid:**
Implement a dedicated `copy_string_column_to_host` function. Use `cudf::strings_column_view sv(col)`. Copy offsets: `sv.offsets().data<int32_t>()`, length `(n+1) * sizeof(int32_t)`. Copy the chars slice: `sv.chars_begin(stream)` for byte count `offsets_host[n] - offsets_host[0]`. Reconstruct strings on the host. Verify the cuDF version's string API before coding (the `chars()` accessor changed in cuDF 24.x).

**Warning signs:**
- `debug_head` showing "(unprinted type STRING)" for string columns
- `debug_stats` or `debug_checksum` silently skipping string columns
- Any `col.data<char>()` on a STRING column without also accessing the offsets child

**Phase to address:**
Phase 2 (type coverage) — after the core infrastructure handles numeric types correctly, extend to STRING/VARCHAR as a distinct implementation path.

---

### Pitfall 5: DECIMAL Values Print as Raw Integers (Scale Factor Missing)

**What goes wrong:**
cuDF DECIMAL types (DECIMAL32, DECIMAL64, DECIMAL128) store values as scaled integers. A value `12345` with scale `-2` represents `123.45`. Printing the raw integer (as a signed INT32/64) shows `12345` instead of `123.45`. Users comparing debug output against DuckDB CPU results will see numbers that differ by orders of magnitude and misdiagnose the operator as incorrect.

**Why it happens:**
`col.type()` returns `cudf::data_type` with an `id()` (e.g., `DECIMAL64`) and a `scale()` accessor. The scale is easy to overlook when treating DECIMAL the same as INT64. The existing `print.cu` prints "(unprinted type)" for all DECIMAL variants, so this pitfall hasn't been hit yet — but it will be when DECIMAL support is added.

**How to avoid:**
For DECIMAL columns, retrieve `int32_t scale = col.type().scale()`. The stored integer value must be divided by `pow(10, -scale)` (scale is typically negative, meaning divide by `10^|scale|`) before display. Use double arithmetic for display only — precision is not critical for debug output. Format as `%.Nf` with `N = -scale` decimal places. DECIMAL128 requires 128-bit integer handling; use `__int128` or format as a pair of 64-bit halves.

**Warning signs:**
- DECIMAL column values that appear as very large integers in debug output
- `debug_diff()` reporting mismatches on DECIMAL columns that are actually correct
- Calls to `col.type().id() == DECIMAL*` without also calling `col.type().scale()`

**Phase to address:**
Phase 2 (type coverage) — handle DECIMAL during the type coverage phase, after numeric and string foundations are correct.

---

### Pitfall 6: Copying from a Batch Not Currently in GPU Tier

**What goes wrong:**
cuCascade implements tiered memory: a `data_batch`'s underlying data may be in GPU tier, HOST tier, or DISK tier depending on memory pressure. `get_cudf_table_view()` calls `cast<gpu_table_representation>()`, which assumes the batch is in GPU tier. If the batch was spilled to host or disk, calling `col.data<T>()` on the resulting `column_view` gives a stale or null pointer, and `cudaMemcpy` with that pointer causes a device-side illegal address or silent data corruption. The pipeline task header (`gpu_pipeline_task.hpp` line 117) shows the system distinguishes tiers explicitly.

**Why it happens:**
Debug utilities are inserted at arbitrary operator boundaries by the `/validate` skill. The skill doesn't know or check whether the batch being inspected has been spilled. During memory pressure (large queries, low GPU memory), spilling happens silently.

**How to avoid:**
Before calling `get_cudf_table_view()`, check `batch.get_data()->get_current_tier() == cucascade::memory::Tier::GPU`. If not, either log a warning ("batch not in GPU tier, skipping debug dump") or use the cucascade API to materialize the batch back to GPU before inspecting. Never silently proceed with a non-GPU-tier batch.

**Warning signs:**
- Crashes or `cudaErrorIllegalAddress` errors only during debug-instrumented runs on large queries
- `debug_head()` producing zeros or garbage on queries that trigger spilling
- Debug calls inserted after operators that might trigger OOM retries

**Phase to address:**
Phase 1 (core infrastructure) — the tier guard must be in the base entry point of every debug utility.

---

### Pitfall 7: `cudaMemcpy` on In-Flight GPU Data (Missing Stream Sync Before Copy)

**What goes wrong:**
A `cudaMemcpy(..., cudaMemcpyDeviceToHost)` (the non-async form, no stream parameter) enqueues on CUDA's default stream. If the column data was produced by GPU operations submitted on a non-default stream (which is the case for all Sirius pipeline tasks, which receive an `rmm::cuda_stream_view`), the copy can run before those operations complete — reading partially-written device memory. The existing `print.cu` calls `cudaDeviceSynchronize()` to avoid this, but that has the Pitfall 2 problem. The correct fix is stream-specific sync, not global sync.

**Why it happens:**
CUDA's memory model allows GPU operations on different streams to execute concurrently and in any order relative to each other. A DeviceToHost `cudaMemcpy` without a stream does not wait for work on other streams. This is a correctness issue, not just a performance issue — the copied data can be genuinely wrong.

**How to avoid:**
Before any DeviceToHost copy in a debug utility, call `stream.synchronize()` where `stream` is the stream on which the column's producing operations were submitted. Then use regular `cudaMemcpy` (which targets the default stream, serializing with the sync). Alternatively, use `cudaMemcpyAsync(..., stream.value())` if you want the copy itself on the task's stream, followed by a `stream.synchronize()` before accessing host memory.

**Warning signs:**
- Debug output that varies between runs (nondeterministic values) on the same query
- Valgrind/compute-sanitizer reporting use-before-initialization on host buffers after DtoH copies
- Debug output correct on small queries (no concurrency) but wrong on large ones

**Phase to address:**
Phase 1 (core infrastructure) — stream parameter and synchronization discipline must be established before any real data is ever copied.

---

### Pitfall 8: Thread Safety — Concurrent printf Interleaves Output Lines

**What goes wrong:**
Pipeline tasks run on multiple CPU threads concurrently. `std::printf` is nominally thread-safe (POSIX guarantees atomic byte-level writes) but does not guarantee that format strings for a single `printf` call are not interleaved with output from another thread's `printf`. When multiple tasks call debug utilities simultaneously, column-by-column output from different tasks appears jumbled in stdout or log files, making it impossible to associate output lines with specific tasks or operators.

**Why it happens:**
`printf` is not protected by any application-level mutex. It is written assuming single-threaded use. The same problem can occur with `SIRIUS_LOG_*` if log messages are split across multiple macro calls — each `SIRIUS_LOG_DEBUG(...)` call is atomic within spdlog, but two separate calls for the same logical row are not atomically grouped.

**How to avoid:**
Format the entire debug output for a batch (all rows, all columns, the full table) into a single `std::string` using `fmt::format` or `std::ostringstream`, then emit that string in a single `SIRIUS_LOG_DEBUG("{}", full_output_string)` call. spdlog's default logger is thread-safe at the per-call level, so a single call with the full string will not be interleaved. Include a batch ID or task ID as a prefix so output from concurrent tasks is distinguishable in the log.

**Warning signs:**
- Log lines appearing with column data from different operators mixed together
- Inconsistent output structure when running queries with parallelism > 1
- Any debug utility that loops over columns/rows making one log call per iteration

**Phase to address:**
Phase 1 (core infrastructure) — the string-builder approach must be established from the start, not retrofitted.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `std::printf` instead of `SIRIUS_LOG_*` | Works immediately, visible on terminal | Invisible in production, skills can't parse it, not in log rotation | Never — use the log macro from day one |
| `cudaDeviceSynchronize()` instead of stream sync | Simple, always correct | Stalls all GPU work across all concurrent tasks | Never — always pass and use the stream |
| Skip null mask in first iteration | Faster to implement | NULL values print as random numbers, misleads diagnosis | Never — nulls are first-class in SQL |
| Implement only for INT32/INT64 types first | Quick win | DECIMAL/STRING/TIMESTAMP produce "(unprinted)" silently | Acceptable as MVP if all types produce a visible placeholder |
| Hard-code `max_rows=20` | Simpler API | Cannot inspect more rows without recompilation | Acceptable for initial phase; make configurable later |
| Single log call per column instead of full-table string | Easier to implement | Interleaved output under concurrency | Never — always buffer the full table output |

---

## Integration Gotchas

Common mistakes when connecting to the existing Sirius infrastructure.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `logging.hpp` in `.cu` files | Calling `SIRIUS_LOG_*` from CUDA-compiled code — silently no-ops under `__CUDACC__` | Implement debug utilities in `.cpp` files; only pure CUDA kernels (if any) go in `.cu` |
| cuCascade `data_batch` | Calling `get_cudf_table_view()` without checking the tier | Guard with `get_current_tier() == Tier::GPU` before extraction |
| cuDF STRING columns | Treating `col.data<char>()` as a flat char array | Use `cudf::strings_column_view` to access separate offsets and chars buffers |
| spdlog daily sink | Log output only appears after `flush_every` interval expires | For debug utilities that must appear immediately, call `spdlog::default_logger()->flush()` after emitting or rely on the configured flush interval (accept delay) |
| `[SIRIUS_DIAG]` prefix | Forgetting the tag on some utility functions | All output from debug utilities must include `[SIRIUS_DIAG]` so skills can reliably grep it |
| cuDF DECIMAL type | Reading `col.data<int64_t>()` and printing raw integer | Always read `col.type().scale()` and apply: `value / pow(10, -scale)` |

---

## Performance Traps

Patterns that work at small scale but become blocking under realistic workloads.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| `cudaDeviceSynchronize()` in debug path | Query latency doubles or triples with debug enabled | Use stream-scoped sync | Any parallel pipeline (multi-task queries) |
| DtoH copying all rows instead of first N | OOM on host, very slow debug output for large batches | Always copy only `min(N, col.size())` rows | Batches > ~10M rows |
| Allocating host vectors inside tight loops | Heap fragmentation, malloc contention across threads | Allocate once per debug call, pass pre-sized vector | High-frequency operator calls with debug enabled |
| Logging every row as a separate log call | Log file I/O becomes the bottleneck, interleaved output | Buffer the full table into one string, one log call | Any table with more than ~100 rows |
| Computing `debug_stats` with device-side reductions on every call | Triggers new cuDF reduction kernels + stream sync | Acceptable for debug only — document that stats functions are expensive | Not a trap for debugging tools specifically, but affects production query time if left enabled |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **debug_head output**: Verify NULL values print as "NULL", not `0` or random integers — check null mask is being read.
- [ ] **STRING columns**: Verify output shows actual string content, not "(unprinted type)" — check `cudf::strings_column_view` path.
- [ ] **DECIMAL columns**: Verify values are divided by `10^|scale|` — print `1.23` not `123` for scale=-2.
- [ ] **TIMESTAMP/DATE columns**: Verify values print as human-readable dates, not raw epoch integers.
- [ ] **Concurrent safety**: Run the query with parallelism >1 and confirm log output is not interleaved between tasks.
- [ ] **Log file presence**: Confirm output appears in the spdlog file (not just stdout) by checking `$SIRIUS_LOG_DIR/sirius.log`.
- [ ] **`[SIRIUS_DIAG]` tag**: Confirm every log line from debug utilities contains `[SIRIUS_DIAG]` so skills can grep reliably.
- [ ] **Spilled batch guard**: Confirm debug utilities emit a warning (not a crash) when called on a non-GPU-tier batch.
- [ ] **Stream sync correctness**: Confirm values match expected results (not stale data from a previous operator) — verify with a known-correct small query.
- [ ] **Build integration**: Confirm new `.cpp` files are added to `CMakeLists.txt` and build without CUDA separable compilation issues.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Output goes to stdout | LOW | Move all output to `SIRIUS_LOG_*`; update call sites to pass stream; this is a search-and-replace refactor |
| `cudaDeviceSynchronize()` in hot path | LOW | Replace with `stream.synchronize()`; requires adding stream parameter to function signatures throughout the call chain |
| Null mask missing | MEDIUM | Add null mask copy to the base host-copy helper; all formatters built on top automatically get null support |
| STRING type not handled | MEDIUM | Add dedicated `strings_column_view` path; requires cuDF API research for current version |
| DECIMAL scale not applied | LOW | Add `col.type().scale()` lookup and apply division in the formatter; isolated change |
| Thread safety / interleaved output | LOW | Switch from per-row log calls to full-table string buffer; isolated to formatter layer |
| Crashed due to non-GPU-tier batch | LOW | Add tier guard at entry point; single location fix |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Output to stdout instead of SIRIUS_LOG | Phase 1: Core infrastructure — establish `.cpp` files and log macro discipline | Grep for `printf`/`cout` in new files; confirm output appears in log file |
| Global cudaDeviceSynchronize | Phase 1: Core infrastructure — establish stream parameter in all function signatures | Run under `cuda-memcheck --check-api-memory-access`; compare timing with/without debug |
| Null mask not read | Phase 1: Core infrastructure — implement null-aware base DtoH copy helper | Query a table with NULLs; verify "NULL" appears in debug_head output |
| STRING column not handled | Phase 2: Type coverage | Query a table with VARCHAR columns; verify string content appears (not "(unprinted)") |
| DECIMAL scale ignored | Phase 2: Type coverage | Query a DECIMAL column with known values; verify formatted output matches DuckDB CPU output |
| Non-GPU-tier batch | Phase 1: Core infrastructure — add tier guard at entry point | Run debug utility on a query that triggers spilling; verify warning not crash |
| Stream sync before DtoH copy | Phase 1: Core infrastructure | Run compute-sanitizer; confirm no race conditions on device data |
| Thread safety / interleaved output | Phase 1: Core infrastructure — buffer full table before logging | Run parallel query; confirm each `[SIRIUS_DIAG]` block in log is self-contained |
| TIMESTAMP/DATE as raw integers | Phase 2: Type coverage | Query TIMESTAMP column; verify output is date-formatted not epoch nanoseconds |

---

## Sources

- Direct analysis of `src/cuda/print.cu` — existing implementation, identified printf/cudaDeviceSynchronize/null-mask issues firsthand
- Direct analysis of `src/include/log/logging.hpp` — confirmed `__CUDACC__` no-op guard, spdlog daily sink, thread-safe per-call semantics
- Direct analysis of `src/include/pipeline/gpu_pipeline_task.hpp` — confirmed multi-stream concurrency model, per-task stream parameter in `execute(rmm::cuda_stream_view stream)`
- Direct analysis of `src/include/data/data_batch_utils.hpp` — confirmed tier assumption in `get_cudf_table_view()`
- Direct analysis of `src/include/pipeline/gpu_pipeline_task.hpp` line 117 — confirmed tier-checking pattern already used in the codebase
- cuDF column memory model: two-buffer STRING representation (`offsets` + `chars`) is a well-established cuDF internal invariant
- cuDF DECIMAL type: scaled integer representation, `data_type::scale()` accessor
- CUDA stream semantics: `cudaMemcpy` targets default stream; DeviceToHost copies do not wait for non-default stream work

---
*Pitfalls research for: GPU debug utility library (Sirius / CUDA / cuDF)*
*Researched: 2026-04-06*
