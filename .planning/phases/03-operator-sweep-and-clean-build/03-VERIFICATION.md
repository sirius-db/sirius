---
phase: 03-operator-sweep-and-clean-build
verified: 2026-04-23T10:45:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/5
  gaps_closed:
    - "Legacy expression executor (src/legacy/expression_executor/gpu_expression_executor.cpp) migrated to to_read_only() — b21799b8"
  gaps_remaining:
    - "Project compiles cleanly against cucascade d9dc331 — still fails (new root cause)"
  regressions: []
gaps:
  - truth: "CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make completes with zero errors against cucascade d9dc331"
    status: failed
    reason: |
      The committed code at HEAD contains 9+ operator files that call
      `data_batch::to_idle(batch.clone(id, stream))`. The `clone()` method on
      `read_only_data_batch` returns `shared_ptr<data_batch>`, but `to_idle()` only
      accepts `read_only_data_batch&&` or `mutable_data_batch&&` — passing a
      `shared_ptr<data_batch>` is a type mismatch and will not compile.
      Additionally `sirius_physical_table_scan.cpp` line 180 calls
      `batch_ref_ptr->clone()` with zero arguments; `read_only_data_batch::clone()`
      requires `(uint64_t new_batch_id, rmm::cuda_stream_view stream)`.

      The build artifact at `build/release/extension/sirius/sirius.duckdb_extension`
      (timestamp 2026-04-23 09:12) was produced from an UNCOMMITTED working tree
      that had additional API fixes (removing `to_idle()` wrappers and fixing the
      zero-arg clone). Those fixes are present in the working tree but were never
      committed as part of Plan 02. The committed code (HEAD) at the time the
      artifact was produced had these errors — the build succeeded only because
      the developer had forward-patched files on disk without committing them.
    artifacts:
      - path: "src/op/sirius_physical_hash_join.cpp"
        issue: "Line 850: to_idle(build_batch_ro.clone(id, stream)) — clone() returns shared_ptr<data_batch>, to_idle() requires accessor type"
      - path: "src/op/sirius_physical_table_scan.cpp"
        issue: "Line 180: batch_ref_ptr->clone() with zero arguments — no zero-arg overload in read_only_data_batch::clone()"
      - path: "src/op/sirius_physical_concat.cpp"
        issue: "Line 187: to_idle(batch.clone(id, stream)) — same type mismatch"
      - path: "src/op/sirius_physical_grouped_aggregate.cpp"
        issue: "Line 180: to_idle(input_batch.clone(id, stream)) — same type mismatch"
      - path: "src/op/sirius_physical_grouped_aggregate_merge.cpp"
        issue: "Line 208: to_idle(batch.clone(id, stream)) — same type mismatch"
      - path: "src/op/sirius_physical_merge_sort.cpp"
        issue: "Line 91: to_idle(batch.clone(id, stream)) — same type mismatch"
      - path: "src/op/sirius_physical_order.cpp"
        issue: "Line 79: to_idle(batch.clone(id, stream)) — same type mismatch"
      - path: "src/op/sirius_physical_partition.cpp"
        issue: "Line 183: to_idle(input_batch_ro.clone(id, stream)) — same type mismatch"
      - path: "src/op/sirius_physical_ungrouped_aggregate.cpp"
        issue: "Line 500: to_idle(batch.clone(id, stream)) — same type mismatch"
    missing:
      - "Commit the uncommitted working-tree fixes: remove to_idle() wrappers around clone() calls (clone() already returns shared_ptr<data_batch> which is idle) and fix the zero-arg clone() call in table_scan.cpp. Files: sirius_physical_hash_join.cpp, sirius_physical_table_scan.cpp, sirius_physical_concat.cpp, sirius_physical_grouped_aggregate.cpp, sirius_physical_grouped_aggregate_merge.cpp, sirius_physical_merge_sort.cpp, sirius_physical_order.cpp, sirius_physical_partition.cpp, sirius_physical_ungrouped_aggregate.cpp"
      - "Run CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make on the fully committed working tree to confirm zero compilation errors"
---

# Phase 3: Operator Sweep and Clean Build Verification Report

**Phase Goal:** Every operator casts to the correct new type, every legacy accessor call site on idle batches uses `to_read_only()`, and the project compiles cleanly against cucascade d9dc331
**Verified:** 2026-04-23T10:45:00Z
**Status:** gaps_found
**Re-verification:** Yes — after gap closure (Plan 02 closed 1 of 2 previous gaps; new root cause found for remaining gap)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All operators cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` for input | ✓ VERIFIED | 31 occurrences of `read_only_pipelineable_operator_data` in `src/op/*.cpp`; 2 remaining `pipelineable_operator_data` casts are in `sink()` methods (partition.cpp:214, result_collector.cpp:125) — intentionally correct per plan |
| 2 | All `pop_data_batch(batch_state::task_created)` calls replaced with `pop_idle_data_batch()` | ✓ VERIFIED | 0 old calls; 9 `pop_idle_data_batch()` occurrences in `src/op/` |
| 3 | All `get_data_batch_by_id` and `pop_data_batch_by_id` calls use updated signatures without state parameter | ✓ VERIFIED | 0 occurrences of `get_data_batch_by_id.*std::nullopt` or `pop_data_batch_by_id.*batch_state` |
| 4 | All idle batch accessor calls go through `to_read_only()` — including legacy expression executor and estimation methods | ✓ VERIFIED | Legacy executor: `to_read_only()` at lines 260 and 346; `input_ro.get_data()` and `input_ro.get_memory_space()` in both `execute()` and `select()`. No remaining `->get_data()`, `->get_memory_space()`, or `->get_current_tier()` calls on idle batches in `src/op/`, `src/expression_executor/`, `src/legacy/`, `src/pipeline/gpu_pipeline_task.cpp`, `src/include/pipeline/gpu_pipeline_task.hpp`, or `src/debug_utils.cpp`. Estimation methods in `gpu_pipeline_task.hpp` use `to_read_only()` at lines 105 and 123. |
| 5 | `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` completes with zero errors against cucascade d9dc331 | ✗ FAILED | Committed code at HEAD has type mismatch errors: 9 operator files call `data_batch::to_idle(batch.clone(id, stream))` where `clone()` returns `shared_ptr<data_batch>` but `to_idle()` requires an accessor type (`read_only_data_batch&&` or `mutable_data_batch&&`). Additionally `sirius_physical_table_scan.cpp:180` calls `batch_ref_ptr->clone()` with zero arguments (no zero-arg overload exists). Build artifact at 09:12 was produced from an uncommitted working tree that pre-applied these fixes; the committed code would fail to compile. |

**Score:** 4/5 truths verified (SC1–SC4 pass; SC5 still blocked)

### Deferred Items

None — Phase 3 is the last phase in the roadmap. No later phases exist to defer items to.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/op/sirius_physical_operator.cpp` | Base operator with `read_only_pipelineable_operator_data` input cast and `pop_idle_data_batch` | ✓ VERIFIED | `pop_idle_data_batch()` at line 273; `read_only_pipelineable_operator_data` present |
| `src/op/sirius_physical_hash_join.cpp` | Hash join with updated pop/get signatures and read-only input cast | ✓ VERIFIED (partial) | Input cast correct; `pop_idle_data_batch()` correct — but line 850 has `to_idle(clone(...))` type mismatch preventing compilation |
| `src/include/data/data_batch_utils.hpp` | Updated `get_cudf_table_view` accepting `read_only_data_batch` | ✓ VERIFIED | New `read_only_data_batch` overload at lines 53–58; old `const data_batch&` overload removed; non-const `data_batch&` overload retained |
| `src/include/pipeline/gpu_pipeline_task.hpp` | Estimation methods using `to_read_only()` | ✓ VERIFIED | `to_read_only()` at lines 105 and 123 |
| `src/pipeline/gpu_pipeline_executor.cpp` | OOM reschedule path without `try_to_create_task` calls | ✓ VERIFIED | 0 `try_to_create_task` calls in `src/` |
| `src/legacy/expression_executor/gpu_expression_executor.cpp` | Legacy executor with `to_read_only()` accessor | ✓ VERIFIED | Lines 260 and 346: `auto input_ro = input_batch->to_read_only()` in both `execute()` and `select()` |
| `build/release/extension/sirius/sirius.duckdb_extension` | Clean build artifact against cucascade d9dc331 | ✗ STALE | Artifact at 09:12 was built from uncommitted working tree; committed HEAD code does not compile cleanly |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/op/sirius_physical_operator.cpp` | cucascade `data_repository.hpp` | `pop_idle_data_batch()` | ✓ WIRED | Line 273 confirmed |
| `src/op/*.cpp execute()` methods | `sirius_physical_operator.hpp` | `dynamic_cast<const read_only_pipelineable_operator_data&>` | ✓ WIRED | 31 occurrences; all execute() paths use new type |
| `src/expression_executor/gpu_expression_executor.cpp` | cucascade `data_batch.hpp` | `to_read_only()` | ✓ WIRED | `execute()` takes `const read_only_data_batch&` directly; `select()` at line 324 uses `to_read_only()` |
| `src/include/pipeline/gpu_pipeline_task.hpp` | cucascade `data_batch.hpp` | `to_read_only()` in estimation methods | ✓ WIRED | Lines 105 and 123 confirmed |
| `src/legacy/expression_executor/gpu_expression_executor.cpp` | cucascade `data_batch.hpp` | `to_read_only()` RAII accessor | ✓ WIRED | Lines 260 and 346 confirmed; commit b21799b8 |
| `src/op/sirius_physical_hash_join.cpp` | cucascade `data_batch.hpp` | `to_idle(batch.clone(...))` pattern | ✗ TYPE_ERROR | `clone()` returns `shared_ptr<data_batch>` but `to_idle()` requires `accessor&&` — type mismatch; affects 9 operator files |

### Data-Flow Trace (Level 4)

Not applicable — this is a pure API migration. No new features or data flows were introduced; this phase only changes how existing batches are locked and accessed.

### Behavioral Spot-Checks

Step 7b: SKIPPED — the committed code does not compile cleanly, so the extension binary cannot be validated. The artifact on disk was built from a different working tree state.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| OPER-01 | 03-01-PLAN.md | All operators cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` | ✓ SATISFIED | 31 casts to new types in `src/op/` execute() methods; 0 old patterns in execute() paths |
| OPER-02 | 03-01-PLAN.md | All `pop_data_batch(batch_state::task_created)` replaced with `pop_idle_data_batch()` | ✓ SATISFIED | 0 old calls; 9 `pop_idle_data_batch()` in `src/op/` |
| OPER-03 | 03-01-PLAN.md | All `get_data_batch_by_id(id, std::nullopt, partition)` updated to 2-param form | ✓ SATISFIED | 0 occurrences of old pattern |
| OPER-04 | 03-01-PLAN.md | All `pop_data_batch_by_id(id, state, partition)` updated to 2-param form | ✓ SATISFIED | 0 occurrences of old pattern |
| ACCS-01 | 03-01-PLAN.md, 03-02-PLAN.md | All `batch->get_data()` on idle data_batch use `to_read_only()` | ✓ SATISFIED | 0 remaining `->get_data()` calls across all searched paths including legacy executor |
| ACCS-02 | 03-01-PLAN.md, 03-02-PLAN.md | All `batch->get_memory_space()` on idle data_batch use `to_read_only()` | ✓ SATISFIED | 0 remaining `->get_memory_space()` calls; legacy executor: `input_ro.get_memory_space()` at lines 294 and 361 |
| ACCS-03 | 03-01-PLAN.md | All `batch->get_current_tier()` on idle data_batch use `to_read_only()` | ✓ SATISFIED | 0 remaining `->get_current_tier()` calls |
| ACCS-04 | 03-01-PLAN.md | `gpu_pipeline_task_local_state` estimation methods use `to_read_only()` | ✓ SATISFIED | `to_read_only()` at lines 105 and 123 of `gpu_pipeline_task.hpp` |
| BILD-01 | 03-01-PLAN.md, 03-02-PLAN.md | Project compiles cleanly against cucascade d9dc331 | ✗ BLOCKED | 9 operator files have `to_idle(clone(...))` type mismatch (committed code); working tree fixes are uncommitted |

### Anti-Patterns Found

| File | Line(s) | Pattern | Severity | Impact |
|------|---------|---------|----------|--------|
| `src/op/sirius_physical_hash_join.cpp` | 850 | `data_batch::to_idle(build_batch_ro.clone(id, stream))` — `clone()` returns `shared_ptr<data_batch>`, `to_idle()` expects `accessor&&` | Blocker | Type mismatch; committed code will not compile against cucascade d9dc331 |
| `src/op/sirius_physical_table_scan.cpp` | 180 | `data_batch::to_idle(batch_ref_ptr->clone())` — zero-arg `clone()` does not exist in `read_only_data_batch` | Blocker | Compilation error; no zero-arg overload |
| `src/op/sirius_physical_concat.cpp` | 187 | `to_idle(batch.clone(id, stream))` — same type mismatch | Blocker | Will not compile |
| `src/op/sirius_physical_grouped_aggregate.cpp` | 180 | `to_idle(input_batch.clone(id, stream))` — same type mismatch | Blocker | Will not compile |
| `src/op/sirius_physical_grouped_aggregate_merge.cpp` | 208 | `to_idle(batch.clone(id, stream))` — same type mismatch | Blocker | Will not compile |
| `src/op/sirius_physical_merge_sort.cpp` | 91 | `to_idle(batch.clone(id, stream))` — same type mismatch | Blocker | Will not compile |
| `src/op/sirius_physical_order.cpp` | 79 | `to_idle(batch.clone(id, stream))` — same type mismatch | Blocker | Will not compile |
| `src/op/sirius_physical_partition.cpp` | 183 | `to_idle(input_batch_ro.clone(id, stream))` — same type mismatch | Blocker | Will not compile |
| `src/op/sirius_physical_ungrouped_aggregate.cpp` | 500 | `to_idle(batch.clone(id, stream))` — same type mismatch | Blocker | Will not compile |
| Various operators (from previous verification) | Various | `const_cast<read_only_pipelineable_operator_data&>(input).release_read_only_batches()` | Warning | Casts away const on `const operator_data&` parameter; functionally safe but bypasses type system |
| `src/include/data/data_batch_utils.hpp` | 71–77 | `get_cudf_table_view(cucascade::data_batch&)` overload releases RAII lock before returning `table_view` | Warning | Returned `table_view` is valid only while batch is in GPU memory; downgrade between lock release and use would produce dangling pointer |

### Human Verification Required

None — all gaps are programmatically verifiable.

### Gaps Summary

**Root cause:** The `to_idle(accessor&&)` static method on `data_batch` accepts only `read_only_data_batch&&` or `mutable_data_batch&&`. The `read_only_data_batch::clone(id, stream)` method returns `shared_ptr<data_batch>` (an already-idle handle) — not an accessor type. Wrapping a clone in `to_idle()` is both semantically wrong (the batch from `clone()` is already idle, so `to_idle()` is redundant) and a type error (passing `shared_ptr<data_batch>` where an rvalue accessor is expected).

The fix is one line per site: change `data_batch::to_idle(batch.clone(id, stream))` to `batch.clone(id, stream)`. The working-tree already has all 9 fixes applied but they were not committed with Plan 02. An additional fix for `sirius_physical_table_scan.cpp:180` (zero-arg `clone()`) must also be committed.

**Why the SUMMARY claimed a clean build:** The developer ran the build from the working tree which already had these fixes applied on disk (file timestamps at 21:16–21:21 Apr 22, built Apr 23 09:12). The fixes were made outside the GSD workflow and were not committed as part of Plan 02. The SUMMARY accurately described the build result from the working tree but the committed code was in a different state.

**Fix required:** Commit the 9 uncommitted working-tree fixes (already present on disk), then run `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` on the committed tree to validate BILD-01.

---

_Verified: 2026-04-23T10:45:00Z_
_Verifier: Claude (gsd-verifier)_
