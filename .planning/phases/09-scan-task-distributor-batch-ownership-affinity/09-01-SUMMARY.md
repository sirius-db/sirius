---
phase: 09-scan-task-distributor-batch-ownership-affinity
plan: 01
subsystem: scan-task-dispatcher
tags: [bug-fix, preferred-device-id, multi-gpu, parquet-scan, plumbing]
requires: [08-01, 08-07]
provides: [preferred_device_id-plumbing-bug2]
affects: [parquet_scan_task, duckdb_scan_executor]
tech-stack:
  added: []
  patterns: [two-tier-preferred-device-id-lookup, local-state-per-task-plumbing]
key-files:
  created: []
  modified:
    - src/include/op/scan/parquet_scan_task.hpp
    - src/op/scan/duckdb_scan_executor.cpp
    - src/op/scan/parquet_scan_task.cpp
decisions:
  - "Path B chosen for Task 1: parquet_scan_task_local_state does NOT inherit set/get_preferred_device_id from sirius_pipeline_task_local_state (only the global state has those). Added the accessors directly to the local state class."
  - "Two-tier lookup mirrors gpu_pipeline_task::get_preferred_device_id (gpu_pipeline_task.hpp:188-194) exactly: local wins over global, global is fallback."
  - "Pre-existing unit test failure (hive partition - filter on data column SIGSEGV) confirmed as Phase 8 ship blocker, not introduced by Phase 9 Plan 01."
metrics:
  duration: "~25 minutes"
  completed: "2026-04-24T10:12:43Z"
  tasks: 3
  files: 3
requirements: [CRIT-2, CRIT-4]
---

# Phase 09 Plan 01: preferred_device_id Plumbing (Bug 2 Fix) Summary

**One-liner:** Per-task `preferred_device_id` plumbed from `select_target_gpu()` in `manager_loop` into `parquet_scan_task_local_state`, and `compute_task` updated to use a two-tier local-wins-over-global lookup for `_datasource` io_backend routing.

---

## What Was Done

### Task 1 — Add preferred_device_id accessors to parquet_scan_task_local_state (Path B)

**Commit:** `3b58258 feat(09-01): add preferred_device_id accessors on parquet_scan_task_local_state`

**File:** `src/include/op/scan/parquet_scan_task.hpp`

**Finding at execute time (Path A vs B decision):** `sirius_pipeline_task_local_state` (the base class) does NOT have `set_preferred_device_id` or `get_preferred_device_id` — only `sirius_pipeline_task_global_state` has them. The RESEARCH.md Q4 entry stated the base class name was `sirius_pipeline_task_local_state` but misstated it as having the accessors. The global state has them; the local state base does not. **Path B was required.**

**Lines added (after `get_rg_indices()` in the `public:` block, before `private:`):**

```cpp
  //===----------Preferred device id (Phase 9 Bug 2 fix)----------===//
  void set_preferred_device_id(int device_id) { _preferred_device_id = device_id; }
  [[nodiscard]] std::optional<int> get_preferred_device_id() const
  {
    return _preferred_device_id;
  }
```

**Added to `private:` block:**
```cpp
  std::optional<int> _preferred_device_id;  ///< Per-task GPU assignment (Phase 9 Bug 2 fix)
```

**Pitfall 1 compliance:** `parquet_scan_task_global_state` not touched. Count unchanged (8 references).

---

### Task 2 — Plumb target_gpu_id into parquet_scan_task_local_state before dispatch

**Commit:** `863cc6c fix(09-01): plumb target_gpu_id into parquet_scan_task_local_state`

**File:** `src/op/scan/duckdb_scan_executor.cpp`

**Insertion location:** Inside the existing `is<parquet_scan_task>()` block in `manager_loop`, AFTER `auto* parquet_task = dynamic_cast<parquet_scan_task*>(scan_task);` and BEFORE the cache/reservation logic.

**Lines added (source line ~327, before `_bounded_pool->dispatch` at line 410):**

```cpp
      // Phase 9 FIX-A: plumb target_gpu_id into per-task local state
      if (auto* parquet_local_state = dynamic_cast<parquet_scan_task_local_state*>(
            scan_task->local_state())) {
        parquet_local_state->set_preferred_device_id(target_gpu_id);
      } else {
        SIRIUS_LOG_ERROR("duckdb_scan_executor: parquet_scan_task local_state downcast failed ...");
      }
```

**Key invariants preserved:**
- `select_target_gpu()` call at line 322 UNCHANGED
- `cuda_set_device_raii.*target_gpu_id` count: 2 (acquire_guard + dispatch_guard, both unchanged)
- Placement: line 339 (set) < line 410 (dispatch) — Pitfall 5 compliance

---

### Task 3 — Two-tier preferred_device_id lookup in compute_task

**Commit:** `0c8068e fix(09-01): two-tier preferred_device_id lookup in parquet_scan_task::compute_task`

**File:** `src/op/scan/parquet_scan_task.cpp`

**Edit 1 — probe breadcrumb (lines ~761-769):**

Replaced:
```cpp
    auto const preferred_probe = g_state.get_preferred_device_id();
```
With:
```cpp
    // Phase 9 FIX-A: two-tier preferred_device_id lookup (local-wins-over-global).
    auto const local_preferred_probe = l_state.get_preferred_device_id();
    auto const preferred_probe       = local_preferred_probe.has_value()
      ? local_preferred_probe
      : g_state.get_preferred_device_id();
```

**Edit 2 — _datasource construction (lines ~794-796):**

Replaced:
```cpp
    auto const preferred = g_state.get_preferred_device_id();
    auto backend_it =
      preferred.has_value() ? backends.find(*preferred) : backends.begin();
```
With:
```cpp
    // Phase 9 FIX-A: two-tier lookup (local-wins-over-global).
    auto const local_preferred = l_state.get_preferred_device_id();
    auto const preferred       = local_preferred.has_value()
      ? local_preferred
      : g_state.get_preferred_device_id();
    auto backend_it =
      preferred.has_value() ? backends.find(*preferred) : backends.begin();
```

**Both edits produce the same effective value** — the probe log will match the actual routing.

---

## Verification Results

| Check | Result |
|-------|--------|
| `grep -c 'set_preferred_device_id(target_gpu_id)' duckdb_scan_executor.cpp` | 1 |
| `grep -c 'l_state.get_preferred_device_id()' parquet_scan_task.cpp` | 3 (2 call sites + 1 pre-existing comment — see Deviations) |
| `grep -c 'g_state.get_preferred_device_id()' parquet_scan_task.cpp` | 2 (fallbacks preserved) |
| Phase 8 Pattern 2 grep (`cuda_set_device_raii.*target_gpu_id`) | 2 matches (unchanged) |
| `rmm::cuda_stream_default` in modified files | 0 net-new (HYG-02 pass) |
| `rmm::cuda_stream_default` total in src/ | 40 (no change from Phase 9 Plan 01) |
| Build via MCP | Exit 0 |
| Unit tests | 316 ran, 315 passed, 1 pre-existing failure (see below) |

---

## Deviations from Plan

### Auto-documented Issues

**1. [Confirmation - Path A → Path B] sirius_pipeline_task_local_state does not have preferred_device_id accessors**

- **Found during:** Task 1 source inspection
- **Issue:** The plan stated both paths were possible (A = inherited, B = add). RESEARCH.md Q4 was ambiguous — it listed `sirius_pipeline_task_global_state` as having the accessors and implied the local base might also. Direct inspection confirmed only the global state has them.
- **Fix:** Took Path B as planned. No functional impact.
- **Files modified:** `src/include/op/scan/parquet_scan_task.hpp`
- **Commit:** 3b58258

**2. [Acceptance Criteria Adjustment] grep -c 'l_state.get_preferred_device_id()' returns 3 not 2**

- **Found during:** Task 3 post-edit verification
- **Issue:** Plan's acceptance criterion stated `grep -c 'l_state.get_preferred_device_id()' src/op/scan/parquet_scan_task.cpp` returns `2`. The actual result is `3` because a pre-existing comment at line 783 contains the substring `local_state/global_state get_preferred_device_id()` which grep matches as `l_state...get_preferred_device_id()` is NOT in the comment — wait, the comment says `local_state` not `l_state`. Let me re-examine: the comment at line 783 says "two-tier local_state/global_state get_preferred_device_id() helper". The grep pattern is `l_state.get_preferred_device_id()` (with period). The comment says `local_state/global_state get_preferred_device_id()` — these DO NOT match `l_state.get_preferred_device_id()`. The grep actually found:
  - Line 764: `auto const local_preferred_probe = l_state.get_preferred_device_id();`
  - Line 783: `// so the two-tier local_state/global_state get_preferred_device_id() helper` — this does NOT contain `l_state.get_preferred_device_id()`
  - Line 803: `auto const local_preferred = l_state.get_preferred_device_id();`

  Re-running grep confirms 3 results. The third result must be something else. Checking: `grep -n 'l_state.get_preferred_device_id()' parquet_scan_task.cpp` showed lines 764, 783, 803. The pre-existing comment at line 783 actually does contain `l_state/global_state get_preferred_device_id()` after some edit — but the text in the file says `local_state/global_state get_preferred_device_id()`. Grep for `l_state.get_preferred_device_id()` with the period will match `l_state/global_state get_preferred_device_id()` because `.` in grep matches any character. The `.` in the pattern matches `/` in the comment text.

- **Root cause:** The plan's grep pattern uses `.` (any char in grep) not `\.` (literal period). The existing comment `local_state/global_state get_preferred_device_id()` at line 783 matches `l_state.get_preferred_device_id()` because `.` = any character. So the 3rd match is from a pre-existing comment, not from the new code.
- **Impact:** Zero — the 2 actual call sites are correct. The plan's acceptance criterion used an unescaped `.` in grep which makes it a false positive.
- **Functional correctness:** Unaffected. Two real call sites exist (probe block + datasource block).

**3. [Pre-existing failure - not introduced] hive partition - filter on data column SIGSEGV**

- **Found during:** MCP unit-tests run after Task 3
- **Issue:** `gpu_execution hive partition - filter on data column` fails with SIGSEGV at `test_gpu_execution_multi_format.cpp:100`
- **Root cause:** This is the Phase 8 v1.2 ship blocker documented in STATE.md blockers section. Not introduced by Phase 9 Plan 01.
- **Verification:** The test exercises the same SIGSEGV as `cudaErrorInvalidValue @ cuda_memcpy.cu:42` (now manifesting as SIGSEGV after 1f80c2a) — a pre-existing cross-GPU batch dispatch race. Phase 9 Plans 02 (batch-ownership affinity map) is designed to address this.
- **Action:** Logged here, no fix applied (out of scope for Plan 01).

---

## Runtime Confirmation (Deferred to Plan 09-04)

The key runtime signal — `[mgpu-probe] parquet_scan_task::compute_task entry preferred_device_id=N` (N != -1) — cannot be verified by this agent (no 2-GPU hardware access). After Plan 09-04, the user should grep:

```bash
grep '[mgpu-probe] parquet_scan_task::compute_task entry' $SIRIUS_LOG_DIR/*.log | grep 'preferred_device_id=-1'
# Expected: ZERO matches (the fix worked)
```

---

## Known Stubs

None. The preferred_device_id plumbing is fully wired: `manager_loop` sets it, `compute_task` reads it. No stub values or TODOs introduced.

## Self-Check

Verified commits exist:
- 3b58258: `feat(09-01): add preferred_device_id accessors on parquet_scan_task_local_state`
- 863cc6c: `fix(09-01): plumb target_gpu_id into parquet_scan_task_local_state`
- 0c8068e: `fix(09-01): two-tier preferred_device_id lookup in parquet_scan_task::compute_task`

Verified files exist:
- `src/include/op/scan/parquet_scan_task.hpp` — modified
- `src/op/scan/duckdb_scan_executor.cpp` — modified
- `src/op/scan/parquet_scan_task.cpp` — modified
