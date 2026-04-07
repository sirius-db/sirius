# Project Research Summary

**Project:** Sirius GPU Debug-Print Utilities
**Domain:** GPU SQL engine debug inspection library (CUDA/cuDF/spdlog)
**Researched:** 2026-04-06
**Confidence:** HIGH

## Executive Summary

Sirius needs a structured debug inspection library that replaces ad-hoc `printf` and manually-coded reduction calls scattered across operator `.cpp` files and Claude Code skills. The pattern is well-established across all major data systems (pandas `.head()`/`.describe()`, Spark `.show()`/`.printSchema()`, DuckDB `SUMMARIZE`): five canonical primitives — schema inspection, null counting, row preview, column statistics, and content fingerprinting — cover the majority of GPU pipeline debugging scenarios. All research was verified directly against installed headers in the pixi environment and existing codebase patterns, so confidence is uniformly high with no speculative elements.

The recommended approach is to extend the existing `src/include/print.hpp` and `src/cuda/print.cu` files with six new functions (`debug_schema`, `debug_nulls`, `debug_head`, `debug_stats`, `debug_checksum`, and `debug_diff`). Zero new dependencies are required: cudf 26.02.x, spdlog 1.8.5, and the CUDA runtime already provide every primitive needed. The foundational constraint driving all design decisions is that `SIRIUS_LOG_*` macros are silently no-oped under `__CUDACC__`, which means all debug utility implementations must live in CPU-path code (`.cpp` or the host-code sections of `.cu` files) and route output through `SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] ...")` so the `/validate` and `/runtime-errors` skills can grep log files reliably.

The critical risks are all infrastructure-level and must be resolved before any feature code is written: stream-scoped synchronization instead of `cudaDeviceSynchronize()` (avoids serializing concurrent pipeline tasks), tier-guarding before `get_cudf_table_view()` (avoids illegal memory access on spilled batches), null-mask extraction alongside data (avoids printing garbage for NULL rows), and buffering the entire table output into a single `SIRIUS_LOG_DEBUG` call (avoids interleaved output from concurrent tasks). None of these risks require novel engineering — the codebase already demonstrates the correct patterns in `gpu_pipeline_task.hpp`, `iceberg_scan_task.cpp`, and `gpu_merge_impl.cpp`.

## Key Findings

### Recommended Stack

All required technology is already present in the linked libraries. No new dependencies are needed, and no CMakeLists changes are required because `print.cu` is already in the build.

**Core technologies:**
- **cudf 26.02.x** (`cudf::reduce`, `cudf::hashing::xxhash_64`, `cudf::slice`, `cudf::strings_column_view`): provides all GPU-side statistics, checksums, and efficient first-N slicing — already a Sirius dependency
- **spdlog 1.8.5** (via `SIRIUS_LOG_DEBUG`/`SIRIUS_LOG_TRACE` macros): mandatory output channel so skills can parse `$SIRIUS_LOG_DIR/sirius.log`; bundled fmt 7.1 provides aligned-column formatting with width specifiers
- **CUDA Runtime 12+/13+** (`cudaMemcpy DeviceToHost`, `cudaStreamSynchronize`): direct GPU-to-host copy primitive; already used in `src/cuda/print.cu` and operator files throughout

**Critical version constraint:** spdlog's bundled fmt must be accessed via `<spdlog/fmt/fmt.h>`, not an external `<fmt/format.h>`. The `logging.hpp` no-op guard under `__CUDACC__` is a hard constraint: all new utility code belongs in `.cpp` files or host-code sections of `.cu` files, never in `__global__` or `__device__` functions.

### Expected Features

The five P1 features are the minimum for the `/validate` and `/runtime-errors` skills to replace their current ad-hoc `SIRIUS_LOG_TRACE("[SIRIUS_DIAG] ...")` patterns with named, reusable function calls.

**Must have (table stakes / v1):**
- `debug_schema(batch, names)` — column names, types, null counts, row count; zero GPU-to-host copies needed (metadata only); cheapest function to call
- `debug_nulls(batch, names)` — per-column null count and percentage; wraps schema output with null-focused formatting
- `debug_head(batch, N, names)` — first N rows in aligned + CSV format; full type coverage including STRING (strings_column_view), DECIMAL (scale-aware), TIMESTAMP/DATE (epoch decoding); replaces most ad-hoc printf patterns
- `debug_stats(batch, names)` — per-column min, max, sum via `cudf::reduce`; replaces the `sum()`/`max()` patterns hand-coded in the `/validate` skill
- `debug_checksum(batch, names)` — per-column xxhash_64 fingerprint XOR-reduced to a single uint64 per column; drives `/validate` Phase 2 cross-run comparison

**Should have (competitive / v1.x):**
- `debug_diff(batch_a, batch_b, names)` — row-level comparison for when checksum divergence is detected; add once the five core functions are validated in real debugging sessions
- `debug_sample(batch, N, names)` — N random rows (not just head); add when head() proves unrepresentative for specific bug classes
- Batch ID + thread ID header on all output blocks — enables correlation in concurrent runs; low urgency for initial validation

**Defer (v2+):**
- Dual aligned/CSV output mode as an explicit parameter — CSV mode is valuable for skill automation but aligned suffices initially
- Python wrappers, file dump outside SIRIUS_LOG, GUI viewer, automatic injection — all explicitly out of scope

**Key dependency structure:** `debug_schema` and `debug_nulls` share no GPU data extraction. `debug_head`, `debug_stats`, and `debug_checksum` all depend on a common GPU-to-host copy layer with full type dispatch. `debug_diff` depends on `debug_schema` for schema validation and benefits from running `debug_checksum` as a fast-path filter first.

### Architecture Approach

The architecture is a strict three-layer design within two existing files: a public API layer (declarations in `print.hpp`), a data extraction layer (GPU-to-host copy helpers in anonymous namespace in `print.cu`), and a formatting layer (string assembly + `SIRIUS_LOG_DEBUG` emission, also in `print.cu`). No new files are needed; no CMakeLists changes are required.

**Major components:**
1. **Public API** (`src/include/print.hpp`) — declares `debug_*` function overloads for both `cucascade::data_batch` and `cudf::table_view`; each overload accepts `rmm::cuda_stream_view stream` and `std::vector<std::string> const& names` (with sensible defaults)
2. **Data extraction layer** (`src/cuda/print.cu`, anonymous namespace) — `extract_column_to_host<T>()`, `extract_strings_to_host()`, `extract_null_mask_to_host()`, `compute_column_stats()`, `compute_column_checksum()`; handles all 16+ type IDs via `switch(col.type().id())`; all `cudaMemcpy` calls batched (one copy per column, not one per row)
3. **Formatting layer** (`src/cuda/print.cu`, anonymous namespace) — builds the full table output as a single `std::string` via `fmt::format` before emitting one `SIRIUS_LOG_DEBUG` call; prevents interleaved output under concurrency
4. **Log sink** (`src/include/log/logging.hpp`, existing) — `SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] ...")` routes to daily file sink; consumed by `/validate` and `/runtime-errors` skills via grep

**Key patterns:**
- `cudaStreamSynchronize(stream.value())` once per entry point (not `cudaDeviceSynchronize`)
- Tier guard (`get_current_tier() == Tier::GPU`) before `get_cudf_table_view()`, with warning log on non-GPU tier
- Null mask copy alongside data copy; `cudf::bit_is_set(host_mask, i)` per row before formatting value
- All cuDF API calls wrapped in `try/catch`; `debug_*` functions never throw
- `cudf::slice(table, {0, N})` zero-copy view before any DtoH copy for `debug_head`

### Critical Pitfalls

1. **Output to stdout bypasses skill parsing** — any `std::printf`, `std::cout`, or `SIRIUS_LOG_*` inside `__CUDACC__`-compiled code is invisible to skills. All debug utilities must be in `.cpp` files (or host-code in `.cu`) and use `SIRIUS_LOG_DEBUG("[SIRIUS_DIAG] ...")` exclusively. Prevent by: establishing this as the first test before writing any feature code.

2. **`cudaDeviceSynchronize()` serializes the entire GPU** — blocks all concurrent pipeline streams, not just the relevant one. Use `stream.synchronize()` (stream-scoped sync). Function signatures must accept `rmm::cuda_stream_view stream` from day one — retrofitting later requires changing every call site.

3. **Null mask ignored causes garbage values** — null positions in cuDF data buffers contain undefined memory. Check `col.nullable()`, copy the null bitmask with `cudaMemcpy`, and test `cudf::bit_is_set(host_mask, i)` before formatting each value. Build this into the base host-copy helper so all formatters inherit correct null behavior automatically.

4. **Non-GPU-tier batch causes illegal memory access** — `get_cudf_table_view()` assumes GPU tier; during spilling the pointer is invalid. Guard with `get_current_tier() == Tier::GPU` at each entry point; emit a `SIRIUS_LOG_WARN` and return rather than crash.

5. **Concurrent output interleaving** — multiple pipeline tasks calling debug utilities simultaneously produces jumbled log output if each row is logged separately. Buffer the entire table output into one `std::string` and emit with a single `SIRIUS_LOG_DEBUG` call per function invocation. Include batch ID and thread ID as a prefix.

6. **STRING and DECIMAL type handling is non-obvious** — STRING requires `cudf::strings_column_view` with separate offsets + chars buffers (not `col.data<char>()`); DECIMAL requires reading `col.type().scale()` and dividing the raw integer by `10^|scale|`. Both must be handled before calling any function that touches values "complete."

## Implications for Roadmap

Based on combined research, the dependency structure and pitfall-to-phase mapping from PITFALLS.md directly determine phase order. Infrastructure mistakes made in Phase 1 propagate to every feature built on top.

### Phase 1: Core Infrastructure

**Rationale:** All eight critical pitfalls identified in PITFALLS.md map to infrastructure decisions that must be correct before feature code is written. Stream sync discipline, tier guards, null mask extraction, output routing, and output buffering are foundational. Getting the function signatures and calling conventions wrong in Phase 1 requires changing every call site later.

**Delivers:** The base host-copy helper (`extract_column_to_host<T>` for numeric types with null mask support), stream-scoped sync pattern, tier guard, single-call output buffering, `[SIRIUS_DIAG]` log routing, and `debug_schema`/`debug_nulls` (which need no GPU-to-host data copy, only metadata — making them the ideal first integration test).

**Addresses from FEATURES.md:**
- `debug_schema` (P1, LOW complexity) — verify log routing works end-to-end
- `debug_nulls` (P1, LOW complexity) — verify null count metadata access

**Avoids from PITFALLS.md:**
- Pitfall 1: stdout bypass (establish `.cpp` file and `SIRIUS_LOG_*` discipline)
- Pitfall 2: global `cudaDeviceSynchronize` (establish stream parameter in all signatures)
- Pitfall 3: null mask not read (build null-aware copy helper from the start)
- Pitfall 6: non-GPU-tier batch (add tier guard at entry point)
- Pitfall 7: in-flight data copy (establish stream sync before any DtoH copy)
- Pitfall 8: thread safety / interleaved output (establish single-call buffering pattern)

### Phase 2: Numeric Type Coverage and Row Preview

**Rationale:** `debug_head` and `debug_stats` share the same GPU-to-host copy layer built in Phase 1 and cover the most common debugging patterns. These are the functions explicitly listed in the `/validate` skill as currently hand-coded. Implement for all numeric types (INT8–UINT64, FLOAT32, FLOAT64, BOOL8) first, then validate skill integration before adding complex types.

**Delivers:** `debug_head(batch, N)` for numeric types with aligned + CSV output, `debug_stats(batch)` with `cudf::reduce` min/max/sum for numeric columns, batch ID tagging, and confirmed skill integration via grep test against `[SIRIUS_DIAG]` in log file.

**Uses from STACK.md:**
- `cudf::slice` for zero-copy first-N row view
- `cudf::reduce` with `make_min/max/sum_aggregation` for per-column statistics
- `fmt::format` with width specifiers for aligned-column output

**Implements from ARCHITECTURE.md:**
- Data extraction layer (numeric types only)
- Formatting layer (aligned + CSV string builder)

### Phase 3: Full Type Coverage (STRING, DECIMAL, TIMESTAMP, DATE)

**Rationale:** Full type coverage is blocked on having the numeric path validated (Phase 2) so regressions are detectable. STRING and DECIMAL require dedicated extraction paths that do not share code with numeric extraction. PITFALLS.md identifies these as the most common sources of "(unprinted type)" gaps that make debug output useless at exactly the moment when the suspicious column is of that type.

**Delivers:** STRING columns via `cudf::strings_column_view` (offsets + chars buffers), DECIMAL columns with `col.type().scale()` applied, TIMESTAMP/DATE columns decoded to human-readable format, and `debug_checksum` with `cudf::hashing::xxhash_64` per column.

**Uses from STACK.md:**
- `cudf::strings_column_view::chars_begin(stream)` + `offsets()` for STRING extraction
- `col.type().scale()` for DECIMAL decimal-point rendering
- `cudf::hashing::xxhash_64` with XOR-fold for per-column checksums

**Avoids from PITFALLS.md:**
- Pitfall 4: STRING flat-buffer assumption (use `strings_column_view` dedicated path)
- Pitfall 5: DECIMAL raw integer display (apply scale factor)

### Phase 4: Diff and Sampling (v1.x)

**Rationale:** `debug_diff` and `debug_sample` build entirely on the foundations of Phases 1-3. `debug_diff` depends on `debug_schema` for schema validation and on the host-copy layer for extracting both batches. These are classified P2/P3 in FEATURES.md — add after validating that the five core functions replace the ad-hoc patterns in the skills.

**Delivers:** `debug_diff(batch_a, batch_b)` for row-level comparison with first-N-differences reporting and schema mismatch detection, `debug_sample(batch, N)` for random row selection when head() is unrepresentative.

**Addresses from FEATURES.md:**
- `debug_diff` (P2, HIGH complexity) — requires schema validation + full host copy of both batches
- `debug_sample` (P3, MEDIUM complexity) — index-based row gather or stride sampling

### Phase Ordering Rationale

- Phases 1-3 strictly enforce the dependency order identified in FEATURES.md: the GPU-to-host copy layer must exist before any feature that reads values; type dispatch must be complete before any function claims full type coverage; output routing must be correct before any skill integration can be validated.
- Infrastructure errors (stream sync, tier guard, null mask, output buffering) compound if deferred — every feature built on a wrong foundation needs rework. Phase 1 makes the cost of getting these right minimal.
- `debug_schema` and `debug_nulls` are in Phase 1 not because they are the most valuable features, but because they test log routing and metadata access without any GPU data extraction — making them the lowest-risk first integration test for the new infrastructure.
- `debug_diff` is deferred to Phase 4 because it is the most complex function, requires validating that checksums work correctly first, and its implementation risk (copying two full batches to host) is higher than the other functions.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (STRING extraction):** The `cudf::strings_column_view` API changed in cuDF 24.x (`chars()` vs `chars_begin(stream)`). The pixi-pinned version is 26.02.x — verify exact accessor signatures in installed headers before implementing, as STACK.md notes this version-sensitivity explicitly.
- **Phase 3 (DECIMAL128):** `__int128` formatting has no `printf`/`fmt` built-in support; requires manual string construction or cast to double (acceptable for debug only). Verify `col.type().scale()` returns negative values as documented.
- **Phase 4 (debug_diff float equality):** FEATURES.md flags that float equality is undefined without epsilon; the implementation must use approximate equality for FLOAT32/FLOAT64. The tolerance value needs a decision before implementation.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Infrastructure):** All patterns directly observed in existing codebase (`gpu_pipeline_task.hpp`, `iceberg_scan_task.cpp`, `logging.hpp`). No novel engineering.
- **Phase 2 (Numeric types):** Existing `print.cu` already implements numeric column DtoH copy; this is an extension of a verified pattern.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technologies verified against installed headers in `.pixi/envs/default/include/` and existing source patterns in `src/op/` and `src/cuda/print.cu`; no speculative elements |
| Features | HIGH | Derived from direct codebase analysis of `/validate` and `/runtime-errors` skill SKILL.md files and PROJECT.md; exact feature gaps identified from reading `print.hpp`/`print.cu` |
| Architecture | HIGH | Based on direct inspection of `print.cu`, `logging.hpp`, `data_batch_utils.hpp`, `cucascade` headers, and operator files; all integration points verified |
| Pitfalls | HIGH | All pitfalls identified from direct code analysis, not inference; the `cudaDeviceSynchronize`, stdout, and null-mask issues are visible bugs in the existing `print.cu` |

**Overall confidence:** HIGH

### Gaps to Address

- **spdlog flush latency:** PITFALLS.md notes that log output only appears after `flush_every` interval; if the `/validate` skill must see output immediately after function calls, the configured flush interval may need to be checked or `spdlog::default_logger()->flush()` called explicitly. Validate during Phase 1.
- **DECIMAL128 formatting:** There is no standard `fmt` format specifier for `__int128`. STACK.md recommends casting to double for display only. Confirm this is acceptable precision for debug use (yes, but worth documenting) during Phase 3.
- **cudf::strings::to_host() vs manual copy:** ARCHITECTURE.md mentions `cudf::strings::to_host()` as a potential canonical API, but STACK.md recommends the manual `chars_begin(stream)` + `offsets()` copy pattern (consistent with `iceberg_scan_task.cpp`). Verify which is preferred for the installed cuDF version during Phase 3.

## Sources

### Primary (HIGH confidence)

- Installed headers: `.pixi/envs/default/include/cudf/hashing.hpp`, `reduction.hpp`, `scalar/scalar.hpp`, `column/column_view.hpp` — API signatures verified
- `src/cuda/print.cu` — existing GPU-to-host copy patterns, type dispatch, `cudaDeviceSynchronize` issue
- `src/include/log/logging.hpp` — `__CUDACC__` no-op guard, spdlog MT macros
- `src/include/pipeline/gpu_pipeline_task.hpp` — per-task stream parameter, tier distinction
- `src/include/data/data_batch_utils.hpp` — `get_cudf_table_view()`, tier assumption
- `src/op/scan/iceberg_scan_task.cpp` — `cudf::strings_column_view` host-extraction pattern
- `src/op/merge/gpu_merge_impl.cpp` — `cudf::reduce` with min/max/sum aggregation pattern
- `src/op/partition/gpu_partition_impl.cpp` — `cudf::hash_partition` + `cudf::hash_id` usage
- `.claude/skills/validate/SKILL.md` — Phase 2 patterns that `debug_stats`/`debug_checksum` will replace
- `.claude/skills/runtime-errors/SKILL.md` — data characterization patterns that `debug_head`/`debug_schema` will replace
- `pixi.toml` — `libcudf = "26.02.*"`, `spdlog = "1.8.*"` confirmed pinned versions
- `CMakeLists.txt` — C++20, CUDA 20, no additional formatting libraries linked
- `.planning/PROJECT.md` — explicit key decisions (extend print.hpp/print.cu, no new files)

### Secondary (MEDIUM confidence)

- data-diff library (datafold/data-diff) — checksum-based diff strategy as industry pattern for pipeline comparison
- Spark Deequ / great_expectations — column statistics + checksum as data quality primitives

---
*Research completed: 2026-04-06*
*Ready for roadmap: yes*
