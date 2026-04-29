---
phase: 12-small-sort-vector-rangecheck-fix
plan: 01
subsystem: testing
tags: [gdb, diagnostics, hash_join, sort, mgpu, libstdcxx, _M_range_check]

# Dependency graph
requires:
  - phase: 08-multi-gpu-sql-pipeline-fix
    provides: "scoped_mgpu_env / require_two_gpus / require_gpu_matches_cpu test infrastructure used by the failing TEST_CASE"
provides:
  - "Concrete File:+Line: identifying the std::out_of_range throw site for the small-sort failure"
  - "Confirmation that the fix-site is in Sirius (sirius_physical_hash_join.cpp:623), NOT cucascade"
  - "Reproduction recipe via mcp__project-commands__run_debug mode=gdb (bare-shell gdb is sandbox-isolated from NVIDIA driver)"
affects: ["12-02-PLAN.md", "12-03-PLAN.md", "12-04-PLAN.md"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern: catch-throw-with-typeinfo-filter: GDB Python guard inspects __cxa_throw's tinfo arg and continues past unrelated throws — survives noise from extension load + test setup before reaching the target std::out_of_range"
    - "Pattern: MCP-mediated GDB on sandboxed agents: bare-shell gdb fails NVIDIA driver init under the agent sandbox; mcp__project-commands__run_debug routes gdb through the project-commands daemon which has driver visibility"

key-files:
  created:
    - ".planning/phases/12-small-sort-vector-rangecheck-fix/12-stack-trace.txt"
  modified: []

key-decisions:
  - "Fix-site identified: src/op/sirius_physical_hash_join.cpp:623 — `result.keys = table.select(key_col_indices)` throws when key_col_indices contains 2 on a 2-column input table_view"
  - "Off-by-one shape: the upstream construction of left_key_col_indices / right_key_col_indices (in HASH_JOIN-as-SORT-partitioner plumbing) builds a stale/oversized index list. HASH_JOIN appears in this path because Sirius's distributed-sort plan uses HASH_JOIN as its partitioner (CONTEXT.md pipelines #3, #7)"
  - "Both gpu_pipeline_1 and gpu_pipeline_2 hit the same site, deterministic across pipeline threads — rules out a racy state and confirms a static plumbing defect"
  - "Used mcp__project-commands__run_debug mode=gdb (not raw gdb) because the bare-shell driver path is sandbox-isolated from NVIDIA driver; raw gdb aborts at cucascade::topology_discovery (0 GPUs) before reaching the SORT plan"
  - "Filtered std::out_of_range catchpoint via GDB Python guard inspecting __cxa_throw tinfo arg — bypasses two unrelated std::runtime_error throws in extension-load/test-setup that would otherwise stop the inferior first"

patterns-established:
  - "Pattern: GDB-batch with Python typeinfo guard for filtering catchpoints by exception type — reusable for any future `catch throw <std::*>` triage"
  - "Pattern: phase-12 diagnostic workflow — single Wave 1 plan produces a fix-site artifact, downstream plans read File:+Line: directly"

requirements-completed: []

# Metrics
duration: ~10min
completed: 2026-04-29
---

# Phase 12 Plan 01: Small-Sort _M_range_check Stack Trace Capture Summary

**Pinned the libstdc++ vector::_M_range_check throw to `src/op/sirius_physical_hash_join.cpp:623` (`prepare_join_keys` -> `cudf::table_view::select`), confirming the fix-site is Sirius (not cucascade) and giving 12-02 an unambiguous patch target.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-04-29T19:05:59Z
- **Completed:** 2026-04-29T19:15:00Z (approx)
- **Tasks:** 1
- **Files created:** 1 (12-stack-trace.txt)

## Accomplishments

- Captured `std::out_of_range` backtrace for the failing test `physical_order - small sort stays single-GPU` via GDB-batch.
- Identified the throwing Sirius frame: `src/op/sirius_physical_hash_join.cpp:623` (`table.select(key_col_indices)`).
- Confirmed the throw is libstdc++'s `vector::_M_range_check` (NOT cucascade's `partition_idx out of range`), eliminating cucascade as a candidate.
- Documented `__n=2 >= size=2` with surrounding source context (5 lines before/after) and full bt 30 stack.
- Pinpointed the upstream defect class: `key_col_indices` (i.e. `left_key_col_indices` / `right_key_col_indices` in the HASH_JOIN-as-SORT-partitioner plumbing) gains a stale `2` entry pointing past the 2-column input table_view.

## Task Commits

1. **Task 1: Capture GDB backtrace at std::out_of_range throw** — `e75a45e` (chore)

**Plan metadata:** TBD (final commit forthcoming)

## Files Created/Modified

- `.planning/phases/12-small-sort-vector-rangecheck-fix/12-stack-trace.txt` — fix-site identification artifact for plan 12-02 to consume

## Decisions Made

- **Fix-site:** `src/op/sirius_physical_hash_join.cpp:623`
- **Frame:** `sirius::op::prepare_join_keys(...)` (the FIRST Sirius frame above the libcudf throw)
- **Off-by-one shape:** `key_col_indices.back() = 2` on a 2-column `cudf::table_view`. Likely upstream cause (for 12-02 to confirm + patch): the SORT plan's HASH_JOIN-as-partitioner constructs key indices that include a column index equal to the table's column count (e.g. uses `num_keys` where `num_keys - 1` is expected, or builds left/right key indices positionally without bounds-checking against the actual input table).
- **Proposed fix shape for 12-02:** Add a precondition `SIRIUS_ASSERT(key_col_indices.empty() || *std::max_element(key_col_indices.begin(), key_col_indices.end()) < table.num_columns())` immediately above line 623 to convert the libstdc++ message into a Sirius-attributable error with operator id + plan-graph context. Then walk back through `prepare_join_keys` <- `sirius_physical_hash_join::execute` <- `sirius_physical_hash_join::sink` <- the SORT-as-HASH_JOIN partitioner setup in `sirius_physical_partition*.cpp` / `sirius_physical_sort_*.cpp` to find the upstream off-by-one.
- **Did NOT step past a cucascade-side throw:** the catchpoint guard skipped two unrelated `std::runtime_error` throws (extension-load + test-setup), but no `partition_idx out of range` (cucascade's text) was encountered before or during the target std::out_of_range hit. cucascade is fully ruled out.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Bare-shell GDB cannot initialize NVIDIA driver under the agent sandbox**
- **Found during:** Task 1 (initial GDB invocation via plain Bash)
- **Issue:** Direct `gdb --batch ... ./sirius_unittest ...` aborts before reaching the SORT plan: `Failed to initialize NVML: Driver Not Loaded`, then `cucascade::topology_discovery reported 0 GPUs — refusing to initialize on stub topology (MGPU-01 fail-hard)` with SIGABRT. The agent sandbox blocks driver access; only the project-commands MCP daemon has driver visibility on this host.
- **Fix:** Re-routed GDB through `mcp__project-commands__run_debug` with `mode=gdb` and the same Python catch-throw guard passed via `flags`. The MCP daemon has NVIDIA driver visibility (verified independently with `mcp__project-commands__run_command name=nvidia-smi`).
- **Files modified:** None (test infrastructure only).
- **Verification:** Resulting GDB output reached the test execution and successfully captured two `std::out_of_range` hits (gpu_pipeline_1 and gpu_pipeline_2).
- **Committed in:** N/A (sandbox-isolation workaround, no source change)

**2. [Rule 3 - Blocking] GDB `catch throw std::out_of_range` syntax does not filter by exception type in this libstdc++**
- **Found during:** Task 1 (initial run with plan's verbatim recipe)
- **Issue:** The plan's verbatim GDB recipe (`-ex 'catch throw std::out_of_range'`) sets a *generic* throw catchpoint, not one filtered to `std::out_of_range`. The first throw hit is an unrelated `std::runtime_error` from extension config loading, which exits the catchpoint before reaching the target throw. Note from GDB output: `did not find exception probe (does libstdcxx have SDT probes?)`.
- **Fix:** Replaced with `catch throw` + a Python `commands` block that inspects `__cxa_throw`'s `tinfo` arg and calls `gdb.execute("continue")` for any throw whose typeinfo is not `std::out_of_range`. This survives the unrelated std::runtime_error throws and stops only at the target.
- **Files modified:** None (test infrastructure only).
- **Verification:** Resulting trace contains the literal `_M_range_check` text and the libstdc++ default message format, both of which are required by the plan's verification block.
- **Committed in:** N/A (test infrastructure only, no source change)

---

**Total deviations:** 2 auto-fixed (both Rule 3 blocking — sandbox isolation + GDB catchpoint syntax)
**Impact on plan:** Both deviations were necessary to obtain the artifact at all. Neither expands plan scope. The captured stack trace satisfies all 5 acceptance criteria from 12-01-PLAN.md verbatim.

## Issues Encountered

- **Two non-target throws** were stepped past by the Python guard: a `std::runtime_error` during extension load (line 53 of original raw output: `SiriusContext::initialize`) and a second `std::runtime_error` later in the run. Neither matched `std::out_of_range`, so the guard's `gdb.execute("continue")` correctly resumed past them. No cucascade-side throw was encountered.

- **GDB symbol-table info is unavailable for most Sirius frames** ("No symbol table info available."). This is because the release build lacks `-g`. The frame names + offsets are still resolved by symbol decoding, which is sufficient to identify the file by reading the source. If 12-02 needs `info locals` on Sirius frames (e.g. to inspect `key_col_indices` contents), it should rebuild with `cmake-debug` or add `-g` to release flags.

## User Setup Required

None — no external service configuration required. The artifact is consumed by 12-02 directly from the phase directory.

## Next Phase Readiness

**12-02 is unblocked.** It can begin with:

```
File: src/op/sirius_physical_hash_join.cpp
Line: 623
Frame: sirius::op::prepare_join_keys
```

12-02's recommended approach:

1. Add a debug-build precondition assert at line 623 to upgrade the libstdc++ message into a Sirius-attributable error.
2. Add `SIRIUS_LOG_DEBUG` of `key_col_indices` and `table.num_columns()` at entry to `prepare_join_keys` to identify the upstream caller variant (BUILD_PROBE / MIXED_JOIN / etc.) that's passing the bad indices.
3. Re-run the failing test, observe which call site (`execute` lines 843/873/938/944/1082/1088) fires the assert.
4. Walk back to the SORT-as-HASH_JOIN partitioner setup in `sirius_physical_partition*.cpp` or `sirius_physical_sort_*.cpp` — the actual off-by-one is in how that wires the partition key column indices.
5. Patch + add invariant comment per CONTEXT.md task 3.

**Concerns for 12-02:**
- If `cmake-debug` is required to inspect locals, build time may exceed 12-02's expected wall-clock; consider adding `-g` to release flags as a one-liner deviation.
- The HASH_JOIN-as-SORT-partitioner architecture means the bug may live in either join code OR sort code; bisecting via the `key_col_indices` log will pin which.

## Self-Check: PASSED

- 12-stack-trace.txt: FOUND
- 12-01-SUMMARY.md: FOUND (this file)
- Task 1 commit (e75a45e): FOUND in `git log --all`
- raw intermediate file removed: PASS
- All 5 plan-specified verification checks pass:
  - non-empty: PASS
  - contains `_M_range_check`: PASS
  - `File:` matches `src/(op|pipeline)/.*\.cpp`: PASS
  - `Line:` numeric: PASS
  - `File:` does NOT cite cucascade: PASS

---
*Phase: 12-small-sort-vector-rangecheck-fix*
*Completed: 2026-04-29*
