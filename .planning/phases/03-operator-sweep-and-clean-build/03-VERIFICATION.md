---
phase: 03-operator-sweep-and-clean-build
verified: 2026-04-22T22:00:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Project compiles cleanly with CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make against cucascade d9dc331"
    status: failed
    reason: |
      The build artifact (sirius.duckdb_extension) is timestamped 17:14 on 2026-04-22.
      The Phase 3 code commits (006aec08 and b4b9c037) are timestamped 19:41, over 2 hours later.
      The build artifact predates the Phase 3 code changes — it was built against pre-migration code.
      Additionally, src/legacy/expression_executor/gpu_expression_executor.cpp (always compiled,
      unconditional in SIRIUS_LEGACY_SOURCES) retains 4 direct calls to data_batch::get_data() and
      data_batch::get_memory_space() via input_batch (a shared_ptr<data_batch>) at lines 260, 293,
      345, 359. These methods are private in cucascade d9dc331 and will produce compilation errors.
      The SUMMARY explicitly acknowledges the build was not validated in the sandbox environment.
    artifacts:
      - path: "src/legacy/expression_executor/gpu_expression_executor.cpp"
        issue: "Lines 260, 293, 345, 359: input_batch->get_data() and input_batch->get_memory_space() on shared_ptr<data_batch> — data_batch::get_data() is private in cucascade d9dc331"
      - path: "build/release/extension/sirius/sirius.duckdb_extension"
        issue: "Build artifact timestamp (17:14) predates Phase 3 commits (19:41) — not a validated clean build"
    missing:
      - "Migrate src/legacy/expression_executor/gpu_expression_executor.cpp Execute() and select() methods to use to_read_only() on input_batch before calling get_data() and get_memory_space()"
      - "Run a full build (CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make) outside sandbox to confirm zero errors against cucascade d9dc331"
---

# Phase 3: Operator Sweep and Clean Build Verification Report

**Phase Goal:** Every operator casts to the correct new type, every legacy accessor call site on idle batches uses `to_read_only()`, and the project compiles cleanly against cucascade d9dc331
**Verified:** 2026-04-22T22:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All operators cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` for input | ✓ VERIFIED | 23 `dynamic_cast` to new types found in `src/op/*.cpp`; all old `pipelineable_operator_data` casts in execute() paths are gone (2 remaining are in `sink()` methods — intentionally kept per plan) |
| 2 | All `pop_data_batch(batch_state::task_created)` calls replaced with `pop_idle_data_batch()` | ✓ VERIFIED | 0 old calls; 9 `pop_idle_data_batch()` occurrences in src/op/ covering all required sites |
| 3 | All `get_data_batch_by_id` and `pop_data_batch_by_id` use updated signatures without state parameter | ✓ VERIFIED | 0 occurrences of `get_data_batch_by_id.*std::nullopt` or `pop_data_batch_by_id.*batch_state` |
| 4 | All `batch->get_data()`, `batch->get_memory_space()`, `batch->get_current_tier()` calls on idle batches go through `to_read_only()` | ✗ FAILED | 4 violations remain in `src/legacy/expression_executor/gpu_expression_executor.cpp` (lines 260, 293, 345, 359): `input_batch->get_data()` and `input_batch->get_memory_space()` on a `shared_ptr<data_batch>`. File is in `SIRIUS_LEGACY_SOURCES` and compiled unconditionally. New (non-legacy) expression executor and all src/op/ files are clean. |
| 5 | `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` completes with zero errors against cucascade d9dc331 | ✗ FAILED | Build artifact is timestamped 17:14, Phase 3 commits are timestamped 19:41. SUMMARY explicitly acknowledges "sandbox write restrictions prevented full artifact production." Legacy executor violations (SC4) would produce compilation errors if built. |

**Score:** 3/5 truths fully verified (SC1–SC3 pass; SC4 partial; SC5 dependent on SC4)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/op/sirius_physical_operator.cpp` | Base operator with `read_only_pipelineable_operator_data` input cast and `pop_idle_data_batch` | ✓ VERIFIED | `pop_idle_data_batch()` at line 273; `read_only_pipelineable_operator_data` present |
| `src/op/sirius_physical_hash_join.cpp` | Hash join with updated pop/get signatures and read-only input cast | ✓ VERIFIED | `pop_idle_data_batch()` at lines 474, 475, 491; `read_only_pipelineable_operator_data` cast at line 813; `build_table_ro_holder` scoping at lines 825, 879, 881 |
| `src/include/data/data_batch_utils.hpp` | Updated `get_cudf_table_view` accepting `read_only_data_batch` | ✓ VERIFIED | New `read_only_data_batch` overload at lines 53-58; old `const data_batch&` overload removed; non-const `data_batch&` overload retained with `to_read_only()` internally |
| `src/include/pipeline/gpu_pipeline_task.hpp` | Estimation methods using `to_read_only()` | ✓ VERIFIED | `to_read_only()` at lines 105 and 123 (both estimation methods) |
| `src/pipeline/gpu_pipeline_executor.cpp` | OOM reschedule path without `try_to_create_task` calls | ✓ VERIFIED | 0 `try_to_create_task` calls in src/ |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/op/sirius_physical_operator.cpp` | cucascade `data_repository.hpp` | `pop_idle_data_batch()` | ✓ WIRED | Line 273 confirmed |
| `src/op/*.cpp execute()` methods | `sirius_physical_operator.hpp` | `dynamic_cast<const read_only_pipelineable_operator_data&>` | ✓ WIRED | 21+ operator files confirmed |
| `src/expression_executor/gpu_expression_executor.cpp` | cucascade `data_batch.hpp` | `to_read_only()` in `select()` | ✓ WIRED | Line 324 confirmed; `execute()` takes `const read_only_data_batch&` directly |
| `src/include/pipeline/gpu_pipeline_task.hpp` | cucascade `data_batch.hpp` | `to_read_only()` in estimation methods | ✓ WIRED | Lines 105 and 123 confirmed |
| `src/legacy/expression_executor/gpu_expression_executor.cpp` | cucascade `data_batch.hpp` | `to_read_only()` accessor | ✗ NOT_WIRED | Lines 260, 293, 345, 359 use `input_batch->get_data()` / `input_batch->get_memory_space()` directly — no `to_read_only()` |

### Data-Flow Trace (Level 4)

Not applicable — this is a pure API migration (no new features, no user-visible data flows added). All operator data flows were pre-existing; this phase only changes how existing batches are locked.

### Behavioral Spot-Checks

Step 7b: SKIPPED (build cannot be run — artifact predates Phase 3 code changes; would require a full clean build in the pixi environment outside sandbox to validate).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| OPER-01 | 03-01-PLAN.md | All operators cast to `read_only_pipelineable_operator_data` or `read_only_partitioned_operator_data` | ✓ SATISFIED | 23 dynamic casts to new types in src/op/ execute() methods; grep for old pattern returns 0 in execute() paths |
| OPER-02 | 03-01-PLAN.md | All `pop_data_batch(batch_state::task_created)` replaced with `pop_idle_data_batch()` | ✓ SATISFIED | 0 old calls; 9 `pop_idle_data_batch()` in src/op/ |
| OPER-03 | 03-01-PLAN.md | All `get_data_batch_by_id(id, std::nullopt, partition)` updated to 2-param form | ✓ SATISFIED | 0 occurrences of `get_data_batch_by_id.*std::nullopt` |
| OPER-04 | 03-01-PLAN.md | All `pop_data_batch_by_id(id, state, partition)` updated to 2-param form | ✓ SATISFIED | 0 occurrences of `pop_data_batch_by_id.*batch_state` |
| ACCS-01 | 03-01-PLAN.md | All `batch->get_data()` on idle data_batch use `to_read_only()` | ✗ BLOCKED | 2 violations in legacy executor (lines 260, 345); new code clean |
| ACCS-02 | 03-01-PLAN.md | All `batch->get_memory_space()` on idle data_batch use `to_read_only()` | ✗ BLOCKED | 2 violations in legacy executor (lines 293, 359); new code clean |
| ACCS-03 | 03-01-PLAN.md | All `batch->get_current_tier()` on idle data_batch use `to_read_only()` | ✓ SATISFIED | 0 remaining occurrences across all searched paths |
| ACCS-04 | 03-01-PLAN.md | `gpu_pipeline_task_local_state` estimation methods use `to_read_only()` | ✓ SATISFIED | `to_read_only()` at lines 105, 123 of gpu_pipeline_task.hpp |
| BILD-01 | 03-01-PLAN.md | Project compiles cleanly against cucascade d9dc331 | ✗ BLOCKED | Build artifact predates Phase 3 code; legacy executor API violations would cause compile errors |

### Anti-Patterns Found

| File | Line(s) | Pattern | Severity | Impact |
|------|---------|---------|----------|--------|
| `src/legacy/expression_executor/gpu_expression_executor.cpp` | 260, 293, 345, 359 | `input_batch->get_data()` / `input_batch->get_memory_space()` on `shared_ptr<data_batch>` | Blocker | `data_batch::get_data()` is private in cucascade d9dc331 — compilation error; file is unconditionally compiled via SIRIUS_LEGACY_SOURCES |
| `src/op/sirius_physical_sort_partition.cpp`, `sort_sample.cpp`, `table_scan.cpp`, `cte.cpp`, `delim_join.cpp`, `column_data_scan.cpp`, `partition.cpp`, `result_collector.cpp` | Various | `const_cast<read_only_pipelineable_operator_data&>(input).release_read_only_batches()` | Warning | Casts away const on `const operator_data&` parameter; functionally safe because operator is sole consumer but bypasses type system (WR-03 from code review) |
| `src/include/data/data_batch_utils.hpp` | 71-77 | `get_cudf_table_view(cucascade::data_batch& batch)` overload releases lock before returning `table_view` | Warning | Returned `table_view` is valid only while batch is in GPU memory; downgrade between lock release and caller's use would produce dangling pointer (WR-01 from code review) |
| `src/op/sirius_physical_operator.cpp` | 272 | `// TODO: later on we will adjust to the new data repository interface in cuCascade` | Info | Pre-existing comment; not a stub |

### Human Verification Required

No human verification items beyond the build — the build itself must be run in a non-sandboxed pixi environment to confirm BILD-01.

### Gaps Summary

**Root cause:** The legacy expression executor (`src/legacy/expression_executor/gpu_expression_executor.cpp`) was identified in the PLAN's task list for migration, but the SUMMARY does not mention it was migrated. The file retains 4 direct calls to private `data_batch` methods (`get_data()` at lines 260/345, `get_memory_space()` at lines 293/359). These methods are private in cucascade d9dc331 — accessible only to `read_only_data_batch` and `mutable_data_batch` (friend classes). Because this file is in `SIRIUS_LEGACY_SOURCES` (unconditionally added to `EXTENSION_SOURCES` in `CMakeLists.txt`), it is compiled in every build configuration. The SUMMARY acknowledged the build was not completed due to sandbox write restrictions, and the build artifact on disk predates the Phase 3 commits by over 2 hours.

**Fix required:** Migrate `GpuExpressionExecutor::Execute()` and `GpuExpressionExecutor::select()` in `src/legacy/expression_executor/gpu_expression_executor.cpp` to call `input_batch->to_read_only()` and use the accessor's `.get_data()` / `.get_memory_space()` methods, then run a full build.

**Scope note on SC4 scope:** The plan's acceptance criterion for Task 2 scoped the search to specific directories: `src/op/`, `src/expression_executor/`, `src/legacy/`, `src/pipeline/gpu_pipeline_task.cpp`, `src/include/pipeline/gpu_pipeline_task.hpp`, `src/debug_utils.cpp`. The `src/legacy/` path was explicitly included in the plan's scope — this is not a gap in the scope definition but a gap in execution.

---

_Verified: 2026-04-22T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
