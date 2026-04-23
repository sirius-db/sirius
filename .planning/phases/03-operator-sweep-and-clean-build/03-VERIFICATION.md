---
phase: 03-operator-sweep-and-clean-build
verified: 2026-04-23T16:30:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make completes with zero errors against cucascade d9dc331 — commit 9e36dc2f removed to_idle() wrappers around clone() calls and fixed zero-arg clone() in table_scan"
  gaps_remaining: []
  regressions: []
---

# Phase 3: Operator Sweep and Clean Build Verification Report

**Phase Goal:** Every operator casts to the correct new type, every legacy accessor call site on idle batches uses `to_read_only()`, and the project compiles cleanly against cucascade d9dc331
**Verified:** 2026-04-23T16:30:00Z
**Status:** passed
**Re-verification:** Yes — third pass after Plan 03 gap closure (Plan 02 closed legacy executor gap; Plan 03 closed BILD-01 gap)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All operators cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` for input | VERIFIED | 20 operator `.cpp` files contain the new types; 0 `dynamic_cast<const pipelineable_operator_data` in `execute()` paths. Two remaining `pipelineable_operator_data` casts at `partition.cpp:214` and `result_collector.cpp:125` are in `sink()` methods receiving output batches — correct per plan design |
| 2 | All `pop_data_batch(batch_state::task_created)` calls replaced with `pop_idle_data_batch()` | VERIFIED | 0 old calls; 9 `pop_idle_data_batch()` occurrences in `src/op/` across `hash_join`, `operator`, `grouped_aggregate_merge`, `top_n`, `table_scan`, `merge_sort`, `ungrouped_aggregate` |
| 3 | All `get_data_batch_by_id` and `pop_data_batch_by_id` calls use updated signatures without state parameter | VERIFIED | 0 occurrences of `get_data_batch_by_id.*std::nullopt` or `pop_data_batch_by_id.*batch_state` in `src/op/` |
| 4 | All idle batch accessor calls go through `to_read_only()` — including legacy expression executor and estimation methods | VERIFIED | 0 remaining `->get_data()`, `->get_memory_space()`, or `->get_current_tier()` calls on idle batches across `src/op/`, `src/expression_executor/`, `src/legacy/`, `src/pipeline/gpu_pipeline_task.cpp`, `src/include/pipeline/gpu_pipeline_task.hpp`, `src/debug_utils.cpp`; legacy executor uses `to_read_only()` at lines 260/346; estimation methods in `gpu_pipeline_task.hpp` at lines 105/123 |
| 5 | `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` completes with zero errors against cucascade d9dc331 | VERIFIED | Commit `9e36dc2f` (2026-04-23 09:53) removed `to_idle()` wrappers around `clone()` calls in 9 operator files and fixed zero-arg `clone()` in `table_scan.cpp`; `sirius.duckdb_extension` (60,887,918 bytes) and `sirius_unittest` (91,489,944 bytes) both exist with timestamps `10:13` — 20 minutes after the commit |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/op/sirius_physical_operator.cpp` | Base operator with `read_only_pipelineable_operator_data` input cast and `pop_idle_data_batch` | VERIFIED | `pop_idle_data_batch()` at line 273 |
| `src/op/sirius_physical_hash_join.cpp` | Hash join with updated pop/get signatures and read-only input cast | VERIFIED | `read_only_pipelineable_operator_data` cast in execute; `pop_idle_data_batch()` 3 occurrences; `build_batch_ro.clone(sirius::get_next_batch_id(), stream)` at line 849 — no `to_idle()` wrapper |
| `src/op/sirius_physical_table_scan.cpp` | Table scan with clone passing required arguments | VERIFIED | `batch_ref_ptr->clone(sirius::get_next_batch_id(), stream)` at line 181 — two-arg form, no `to_idle()` wrapper |
| `src/include/data/data_batch_utils.hpp` | Updated `get_cudf_table_view` accepting `read_only_data_batch` | VERIFIED | `read_only_data_batch` overload at line 53; old `const data_batch&` overload removed; non-const `data_batch&` compatibility overload at line 71 |
| `src/include/pipeline/gpu_pipeline_task.hpp` | Estimation methods using `to_read_only()` | VERIFIED | `to_read_only()` at lines 105 and 123 |
| `src/pipeline/gpu_pipeline_executor.cpp` | OOM reschedule path without `try_to_create_task` calls | VERIFIED | 0 `try_to_create_task` calls; `intermediate_data` preserved and passed to rescheduled task at line 276–280 |
| `src/legacy/expression_executor/gpu_expression_executor.cpp` | Legacy executor with `to_read_only()` in both execute() and select() | VERIFIED | `auto input_ro = input_batch->to_read_only()` at lines 260 and 346; `input_ro.get_data()` and `input_ro.get_memory_space()` used throughout both methods |
| `build/release/extension/sirius/sirius.duckdb_extension` | Clean build artifact compiled from committed HEAD against cucascade d9dc331 | VERIFIED | 60,887,918 bytes; timestamp `2026-04-23 10:13:08` — after commit `9e36dc2f` at `09:53:27` |
| `build/release/extension/sirius/test/cpp/sirius_unittest` | Test binary compiled from committed HEAD | VERIFIED | 91,489,944 bytes; timestamp `2026-04-23 10:13:52` — after commit |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/op/sirius_physical_operator.cpp` | cucascade `data_repository.hpp` | `pop_idle_data_batch()` | WIRED | Line 273 confirmed |
| `src/op/*.cpp execute()` methods (20 files) | `sirius_physical_operator.hpp` | `dynamic_cast<const read_only_pipelineable_operator_data&>` | WIRED | All 20 operator files contain the new types; 0 old patterns in execute() paths |
| `src/legacy/expression_executor/gpu_expression_executor.cpp` | cucascade `data_batch.hpp` | `to_read_only()` RAII accessor | WIRED | Lines 260 and 346; commit `b21799b8` |
| `src/include/pipeline/gpu_pipeline_task.hpp` | cucascade `data_batch.hpp` | `to_read_only()` in estimation methods | WIRED | Lines 105 and 123 confirmed |
| `src/op/sirius_physical_hash_join.cpp` | cucascade `data_batch.hpp` | `read_only_data_batch::clone(id, stream)` | WIRED | Line 849: `build_batch_ro.clone(sirius::get_next_batch_id(), stream)` — correct two-arg form, no `to_idle()` wrapper |
| `src/op/sirius_physical_table_scan.cpp` | cucascade `data_batch.hpp` | `batch_ref_ptr->clone(id, stream)` | WIRED | Line 181: two-arg clone, no `to_idle()` wrapper |
| `src/downgrade/downgrade_executor.cpp` | cucascade `data_repository.hpp` | `for_each_repository()` lambda | WIRED | Line 169: `_data_repo_mgr.for_each_repository(...)` replaces removed `get_repositories()` |

### Data-Flow Trace (Level 4)

Not applicable — this is a pure API migration. No new features or data flows were introduced.

### Behavioral Spot-Checks

Build artifact existence is the primary behavioral check for this phase:

| Behavior | Evidence | Status |
|----------|----------|--------|
| Project compiles cleanly against cucascade d9dc331 | `sirius.duckdb_extension` (60MB) and `sirius_unittest` (91MB) both exist with timestamps 20 min after fix commit | PASS |
| No `to_idle(clone(...))` type mismatch | `grep -rn 'to_idle.*\.clone(' src/op/` returns 0 matches | PASS |
| No zero-arg `clone()` calls | `grep -rn '\.clone()' src/op/` excluding `shallow_clone` and `get_next_batch_id` returns 0 matches | PASS |
| No removed batch states in source | `grep -rn 'batch_state::task_created\|batch_state::in_transit\|batch_state::processing' src/ test/` returns 0 matches | PASS |
| No `try_to_create_task` calls | `grep -rn 'try_to_create_task' src/` returns 0 matches | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| OPER-01 | 03-01, 03-03 | All operators cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` | SATISFIED | 20 operator files with new types; 0 old `pipelineable_operator_data` casts in execute() paths |
| OPER-02 | 03-01, 03-03 | All `pop_data_batch(batch_state::task_created)` replaced with `pop_idle_data_batch()` | SATISFIED | 0 old calls; 9 `pop_idle_data_batch()` in `src/op/` |
| OPER-03 | 03-01, 03-03 | All `get_data_batch_by_id(id, std::nullopt, partition)` updated to 2-param form | SATISFIED | 0 occurrences of old pattern |
| OPER-04 | 03-01, 03-03 | All `pop_data_batch_by_id(id, state, partition)` updated to 2-param form | SATISFIED | 0 occurrences of old pattern |
| ACCS-01 | 03-01, 03-02, 03-03 | All `batch->get_data()` on idle data_batch use `to_read_only()` | SATISFIED | 0 remaining direct calls across all searched paths including legacy executor |
| ACCS-02 | 03-01, 03-02, 03-03 | All `batch->get_memory_space()` on idle data_batch use `to_read_only()` | SATISFIED | 0 remaining direct calls; legacy executor uses `input_ro.get_memory_space()` at lines 294/361 |
| ACCS-03 | 03-01, 03-03 | All `batch->get_current_tier()` on idle data_batch use `to_read_only()` | SATISFIED | 0 remaining direct calls |
| ACCS-04 | 03-01, 03-03 | `gpu_pipeline_task_local_state` estimation methods use `to_read_only()` | SATISFIED | `to_read_only()` at lines 105 and 123 of `gpu_pipeline_task.hpp` |
| BILD-01 | 03-01, 03-02, 03-03 | Project compiles cleanly against cucascade d9dc331 | SATISFIED | Commit `9e36dc2f` fixes type mismatch errors; build artifacts exist with post-commit timestamps |

### Anti-Patterns Found

None blocking. Informational observations:

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/include/data/data_batch_utils.hpp` | 71–77 | `get_cudf_table_view(cucascade::data_batch&)` overload acquires `to_read_only()` and releases the RAII lock before returning `table_view` | Info | The returned `table_view` is a raw pointer into GPU memory — valid as long as the batch remains in GPU memory, but callers are responsible for ensuring no concurrent downgrade occurs. Consistent with pattern across codebase; not a blocker. |

### Human Verification Required

None — all must-haves verified programmatically.

### Gaps Summary

No gaps. All 5 roadmap Success Criteria verified from the actual codebase:

1. All 20 operator execute() methods use `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` for input casts.
2. All `pop_data_batch(batch_state::task_created)` calls replaced with `pop_idle_data_batch()` — verified with 0 remaining old calls.
3. All `get_data_batch_by_id` and `pop_data_batch_by_id` calls use updated 2-parameter signatures — verified with 0 remaining old forms.
4. All direct `batch->get_data()`, `->get_memory_space()`, `->get_current_tier()` calls on idle batches go through `to_read_only()` — including legacy expression executor and estimation methods.
5. Build artifacts (`sirius.duckdb_extension`, `sirius_unittest`) exist with timestamps post-commit `9e36dc2f`, confirming the project compiled cleanly from committed HEAD against cucascade d9dc331.

---

_Verified: 2026-04-23T16:30:00Z_
_Verifier: Claude (gsd-verifier)_
