---
phase: 09-scan-task-distributor-batch-ownership-affinity
plan: 02
subsystem: scan-task-dispatcher
tags: [bug-fix, batch-affinity, multi-gpu, double-dispatch, observability]
requires: [09-01]
provides: [batch-gpu-affinity-map, mgpu-probe-breadcrumbs]
affects: [duckdb_scan_executor, pipelineable_operator_data]
tech-stack:
  added: []
  patterns: [sticky-batch-gpu-affinity-map, mutex-guarded-map-insert, query-start-reset-pair]
key-files:
  created: []
  modified:
    - src/include/op/scan/duckdb_scan_executor.hpp
    - src/op/scan/duckdb_scan_executor.cpp
    - src/op/sirius_physical_operator.cpp
decisions:
  - "Affinity reset placed at the very top of prepare_cache_for_scan_operators (before the cache_level::NONE early return) so it fires unconditionally on every query start, not just on cached queries. The _scan_round_robin counter is used regardless of cache level, so resetting both together at the earliest possible point prevents any drift."
  - "Fallback counter extracted from the inline fetch_add expression into a named variable (fallback_counter) so it can serve as both the modulo dividend and the affinity map key — direct requirement of plan acceptance criterion."
  - "Audit log payload format ([mgpu-audit] scan_batch assigned to GPU N batch_id=K) left UNCHANGED; Plan 09-03's parse_audit_log regex depends on the exact string."
metrics:
  duration: "~4 minutes"
  completed: "2026-04-24T10:19:51Z"
  tasks: 3
  files: 3
requirements: [CRIT-1, CRIT-2, CRIT-4]
---

# Phase 09 Plan 02: Batch-GPU Affinity Map (Bug 1 Fix) Summary

**One-liner:** Sticky `std::unordered_map<uint64_t,int> _batch_gpu_affinity` added to `duckdb_scan_executor`, written atomically with the `[mgpu-audit]` batch_id emission in both branches of `select_target_gpu()` and cleared alongside `_scan_round_robin` at query start; `[mgpu-probe]` breadcrumbs added on both `std::nullopt` return paths in `pipelineable_operator_data::prepare_for_processing`.

---

## What Was Done

### Task 1 — Add _batch_gpu_affinity map + mutex to header

**Commit:** `a8a7985 feat(09-02): add _batch_gpu_affinity map to duckdb_scan_executor`

**File:** `src/include/op/scan/duckdb_scan_executor.hpp`

**Includes added:**
```cpp
#include <mutex>
#include <optional>
#include <unordered_map>
```
These were previously only transitively included. Explicit includes make the header self-contained.

**Members added** (immediately after `_scan_round_robin` at line 206, for cohesion with the counter that generates map keys):

```cpp
  // Phase 9 FIX-B (Bug 1 — 08-08-DIAGNOSIS.md hypothesis E): sticky batch→GPU
  // affinity, recorded atomically with the [mgpu-audit] batch_id=K emission
  // in select_target_gpu(). ...
  mutable std::mutex _batch_affinity_mutex;
  std::unordered_map<uint64_t, int> _batch_gpu_affinity;
```

`_scan_round_robin` declaration unchanged at its existing position.

---

### Task 2 — Record batch→GPU affinity in select_target_gpu + reset in prepare_cache_for_scan_operators

**Commit:** `c0e12f3 fix(09-02): record batch→GPU affinity in select_target_gpu + reset on query start`

**File:** `src/op/scan/duckdb_scan_executor.cpp`

**Edit 1 — Weighted branch (inside the `target < cumulative` block, BEFORE the SIRIUS_LOG_INFO):**

```cpp
      // Phase 9 FIX-B: record batch→GPU affinity atomically with the audit emission.
      {
        std::lock_guard<std::mutex> lock(_batch_affinity_mutex);
        _batch_gpu_affinity[counter] = space->get_device_id();
      }
      SIRIUS_LOG_INFO("[mgpu-audit] scan_batch assigned to GPU {} batch_id={} ...", ...);
      return space->get_device_id();
```

Location: inserted at line ~220 (post-09-01 line numbering), before the existing SIRIUS_LOG_INFO.

**Edit 2 — Fallback branch (total_available == 0 block):**

Refactored:
```cpp
    auto idx = _scan_round_robin.fetch_add(1) % _gpu_memory_spaces.size();
    return _gpu_memory_spaces[idx]->get_device_id();
```

Into:
```cpp
    auto fallback_counter = _scan_round_robin.fetch_add(1);
    auto idx              = fallback_counter % _gpu_memory_spaces.size();
    auto device_id        = _gpu_memory_spaces[idx]->get_device_id();
    {
      std::lock_guard<std::mutex> lock(_batch_affinity_mutex);
      _batch_gpu_affinity[fallback_counter] = device_id;
    }
    return device_id;
```

**Edit 3 — Query-start reset in prepare_cache_for_scan_operators:**

Added at the very top of the function body (before the `cache_level::NONE` early return):

```cpp
  // Phase 9 FIX-B (Pitfall 3): reset affinity map alongside _scan_round_robin
  // at query start. Without this paired reset, a new query's counter values
  // would collide with stale entries from a prior query.
  _scan_round_robin.store(0, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(_batch_affinity_mutex);
    _batch_gpu_affinity.clear();
  }
```

Note: placed BEFORE the `cache_level::NONE` early return so the reset fires unconditionally on every query start, regardless of caching mode.

**Invariants preserved:**
- Phase 8 Pattern 2: `cuda_set_device_raii.*target_gpu_id` = 2 matches (acquire_guard + dispatch_guard unchanged)
- Audit log payload: `[mgpu-audit] scan_batch assigned to GPU N batch_id=K` format UNCHANGED
- Plan 09-01's `parquet_local_state->set_preferred_device_id(target_gpu_id)` block at line ~339 NOT touched

---

### Task 3 — Defensive breadcrumbs on nullopt propagation in prepare_for_processing

**Commit:** `e2484e0 chore(09-02): add [mgpu-probe] breadcrumbs on prepare_for_processing nullopt paths`

**File:** `src/op/sirius_physical_operator.cpp`

**Edit 1 — Null-batch early return (line ~47):**

```cpp
    if (!batch) {
      SIRIUS_LOG_ERROR("pipelineable_operator_data: null batch encountered, skipping");
      SIRIUS_LOG_INFO("[mgpu-probe] prepare_for_processing returning nullopt null_batch=true");
      return std::nullopt;
    }
```

**Edit 2 — Lock-failure return (line ~77):**

```cpp
    if (!handle) {
      // Phase 9 FIX-B observability: breadcrumb confirms nullopt propagates
      // out of the loop cleanly, ...
      SIRIUS_LOG_INFO(
        "[mgpu-probe] prepare_for_processing returning nullopt batch_id={} batch_state={}",
        batch->get_batch_id(),
        static_cast<int>(batch->get_state()));
      return std::nullopt;
    }
```

Function signature, try/catch block, and all existing control flow UNCHANGED. Both breadcrumbs are strictly observability additions.

---

## Verification Results

| Check | Result |
|-------|--------|
| `grep -c '_batch_gpu_affinity' duckdb_scan_executor.hpp` | 1 (declaration) |
| `grep -cE '_batch_gpu_affinity\[(counter\|fallback_counter)\]' duckdb_scan_executor.cpp` | 2 |
| `grep -c '_batch_gpu_affinity.clear()' duckdb_scan_executor.cpp` | 1 |
| `grep -c '\[mgpu-probe\] prepare_for_processing returning nullopt' sirius_physical_operator.cpp` | 2 |
| Phase 8 Pattern 2 (`cuda_set_device_raii.*target_gpu_id`) | 2 (unchanged) |
| Audit log SIRIUS_LOG_INFO emission count | 1 (unchanged) |
| `rmm::cuda_stream_default` total in src/ | 40 (HYG-02 — no change) |
| Build via MCP | Exit 0 |
| Unit tests | 316 ran, 315 passed, 1 pre-existing failure (see below) |

---

## Deviations from Plan

### Auto-documented Issues

**1. [Acceptance Criteria Adjustment] grep -c 'return std::nullopt' returns 4, not 2**

- **Found during:** Task 3 post-edit verification
- **Issue:** The plan's acceptance criterion stated the count should be `2`. The actual result is `4` because `sirius_physical_operator.cpp` contains two additional `return std::nullopt` statements in `get_next_task_hint()` (lines 239 and 274) that were pre-existing and unrelated to `prepare_for_processing`. The plan's criterion was written counting only `prepare_for_processing`'s 2 returns.
- **Impact:** Zero — both `prepare_for_processing` nullopt returns are present and correct. The pre-existing returns in `get_next_task_hint` are unmodified.
- **Functional correctness:** Unaffected.

**2. [Line shift from 09-01 insertion] All line references in plan shifted**

- **Found during:** Task 2 source inspection
- **Issue:** Plan 09-01 inserted ~10 lines into `duckdb_scan_executor.cpp` at line ~327-345 (the `set_preferred_device_id` plumbing block). All subsequent line numbers in the plan's action description are off by ~10. The functions were located by name, not line number.
- **Impact:** Zero — functions were found correctly by name. The plan's line number annotations in the action section are informational only.

**3. [Pre-existing failure — not introduced] hive partition - filter on data column SIGSEGV**

- **Found during:** MCP unit-tests run after Task 3
- **Issue:** `gpu_execution hive partition - filter on data column` fails with SIGSEGV at `test_gpu_execution_multi_format.cpp:100`
- **Root cause:** Pre-existing v1.2 ship blocker, identical to the failure documented in Plan 09-01 SUMMARY. Not introduced by Phase 9 Plan 02.
- **Action:** None (out of scope).

**4. [Acceptance Criteria Clarification] grep -c '[mgpu-audit] scan_batch' returns 2, not 1**

- **Found during:** Task 2 post-edit verification
- **Issue:** The plan's criterion requires count = 1 for `grep -c '\[mgpu-audit\] scan_batch assigned to GPU' src/op/scan/duckdb_scan_executor.cpp`. The actual result is 2 because a comment block at line ~232 contains the text `"[mgpu-audit] scan_batch assigned to GPU N" substring is preserved`. The actual SIRIUS_LOG_INFO emission is the only call site (line 243). The criterion used an unescaped grep pattern that matches both the comment and the real log line.
- **Impact:** Zero — the real log payload is unchanged. Plan 09-03's regex depends on the runtime log output, not the source code grep count.

---

## Architecture Notes

### What the Affinity Map Does NOT Do (Deferred)

The affinity map (`_batch_gpu_affinity`) is **written** at dispatch time but is **not yet consulted** at dispatch time to re-route tasks. That is:

- If `batch_id=3` was dispatched to GPU 0 (recorded in the map), and a second task attempts to dispatch the same logical batch to GPU 1, the `select_target_gpu()` call will NOT currently read the map to detect the conflict.
- The map provides the **data structure** for Plan 09-03's disjointedness assertion: the test can parse the `[mgpu-audit]` log, extract `batch_id` per GPU, and REQUIRE that the sets are disjoint. If they're not disjoint, the bug is still present at the dispatch level.
- The structural prevention of cross-GPU dispatch (reading the map at dispatch time) is deferred to future work (Phase 10+) if Plan 09-04's validation run shows residual cross-GPU collisions after the combined 09-01 (preferred_device_id plumbing) + 09-02 (affinity recording) fixes.

### Pitfall 3 Compliance

`_scan_round_robin` and `_batch_gpu_affinity` are reset together in `prepare_cache_for_scan_operators` before the `cache_level::NONE` early return. This means:

- Cache path: both are reset, then the cache logic runs
- No-cache path: both are reset, then the function returns immediately

Both paths see a clean counter + map at query start, preventing stale-entry collisions.

---

## Known Stubs

None. The affinity map is written on every dispatch (both branches of `select_target_gpu`) and cleared on every query start. No stub values, TODOs, or placeholder logic introduced.

## Self-Check

Verified commits exist:
- a8a7985: `feat(09-02): add _batch_gpu_affinity map to duckdb_scan_executor`
- c0e12f3: `fix(09-02): record batch→GPU affinity in select_target_gpu + reset on query start`
- e2484e0: `chore(09-02): add [mgpu-probe] breadcrumbs on prepare_for_processing nullopt paths`

Verified files exist and are modified:
- `src/include/op/scan/duckdb_scan_executor.hpp` — modified
- `src/op/scan/duckdb_scan_executor.cpp` — modified
- `src/op/sirius_physical_operator.cpp` — modified
