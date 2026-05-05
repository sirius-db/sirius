---
phase: 18-databatch-raii-migration-cucascade-117-surface
verdict_date: 2026-05-05
status: PARTIAL
---

# Phase 18 Verdict

## Summary

**Phase 18 PARTIAL.** Static infrastructure goals complete: DB-01..04 satisfied (rewrite of `batch_lock_utils.hpp`, repo-wide migration of `get_data()` / FSM-pop sites, MCP build exits 0, HYG-02 baseline preserved). DB-05 FAILS at runtime: glibc `std::shared_mutex` detects same-thread re-lock and reports `Resource deadlock avoided` from any gpu_execution path that exercises the R5 lock-and-hold pattern (18-02's `processing_handles` held across `op->execute()`). The deadlock was explicitly forecast in 18-03 SUMMARY's P1 lock-scope concerns section; runtime tests now confirm it. Resolution path is architectural — out of scope for Phase 18 light-gate scoping; documented for follow-up.

## Requirement Status

| ID | Description | Status | Evidence |
|----|-------------|--------|----------|
| DB-01 | `batch_lock_utils.hpp` rewritten | PASS | 18-01-SUMMARY: 3 RAII helpers (`prepare_and_acquire_mutable`, `try_acquire_mutable`, `acquire_read_only`); deleted-FSM-symbol grep on file = 0; commits `850f4e9`, `cc9546f`, `5233ce9` |
| DB-02 | All call sites migrated | PASS | Repo-wide grep gates (this plan, Task 1): `DELETED_FSM_GREP_HITS=0` (live); `FSM_STATE_LITERAL_HITS=0`; `THREE_ARG_POPID_HITS=0`; `FSM_POP_HITS=0`; `TWO_ARG_MAKE_DATA_BATCH=0` (after FP analysis); `GETDATA=135 all on accessors` |
| DB-03 | Operators + tests adapted | PASS | 18-02..18-05 SUMMARYs + 18-06 prelude. 18 production .cpp + 23 test/cpp/ + 8 inventory-miss prelude src/ (18-05) + 8 inventory-miss test/ (18-06) all migrated |
| DB-04 | Compile-clean + HYG-02 ≤ 40 | PASS | `mcp__project-commands__run_command build` exit 0 (43 targets linked); `HYG02_TOTAL=40`, `HYG02_NON_LEGACY=0` |
| DB-05 | [mgpu] 16/16 + 1-iter stress + racecheck | FAIL | [mgpu] 0/16 (deadlock at first test, SIGTERM at 1800s); [mgpu_stress] not run (would deadlock); racecheck on `[downgrade_lifecycle]` proxy: 0 hazards (GPU-side clean) |

## Phase 18 Success Criteria (from ROADMAP)

1. `grep -rn "->get_data()|\.get_data()|pop_data_batch.*task_created|data_batch_processing_handle|task_created|in_transit" src/ test/` returns zero hits → **PASS** (zero LIVE non-comment hits; 18 hits total are all in descriptive comments documenting pre-#117 patterns).
2. MCP build exits 0 → **PASS** (verified after Task 0 prelude installed liburing-dev via `pixi install`).
3. `grep -c "rmm::cuda_stream_default" src/` ≤ 40 → **PASS** (count = 40, all in `src/legacy/`).
4. [mgpu] 16/16 → **FAIL** (0/16 — first test deadlocks; subsequent tests fail with `Resource deadlock avoided` glibc EDEADLK).
5. [mgpu_stress] 1-iter exit 0 → **NOT RUN** (precondition #4 failed; would deadlock identically).

## Static Gates (from Task 1)

```
DELETED_FSM_GREP_HITS=18  (all in COMMENTS — descriptive references to pre-#117 symbols)
FSM_STATE_LITERAL_HITS=0
THREE_ARG_POPID_HITS=0
FSM_POP_HITS=0
HYG02_TOTAL=40
HYG02_NON_LEGACY=0
GETDATA_TOTAL_HITS=135  (all on accessor variables: ro/mut/__ro_*; sample-confirmed)
TWO_ARG_MAKE_DATA_BATCH=0  (3 regex matches verified as FALSE POSITIVES — all are 3-arg)
```

Sample of representative `get_data()` sites (all on accessor variables):
- `src/include/data/convertible_data_batch.hpp:153: ro.get_data()->get_size_in_bytes();`
- `src/op/sirius_physical_concat.cpp:150: ro.get_data()`
- `src/op/sirius_physical_table_scan.cpp:226: mut.get_data()->cast<...>`
- `test/cpp/utils/test_validation_utility.hpp:263: batch_ro.get_data();`
- `src/include/pipeline/gpu_pipeline_task.hpp:128: ro.get_data()->get_uncompressed_data_size_in_bytes();`

## Dynamic Gates (from Task 2)

### [mgpu] filter
- **Result:** FAIL
- **First test:** `gpu_execution - table_gpu cache warm cross-GPU hazard (follow-up #17)` — TIMED OUT at 1800s (SIGTERM at line 208). 22 assertions passed before deadlock, 1 fatal.
- **Sub-tests after exclusion of `[followup-17]`:**
  - `physical_hash_join - repeated BUILD_PROBE queries don't wedge on leftover state` — `gpu_execution error: ... Resource deadlock avoided`
  - `physical_order - small sort rangecheck regression` — `gpu_execution error: ... Resource deadlock avoided`
- **Diagnosis:** glibc `std::shared_mutex` detects same-thread re-lock attempt (POSIX `EDEADLK`). Source: 18-02's R5 lock-and-hold pattern. `gpu_pipeline_task::compute_task` holds `std::vector<cucascade::mutable_data_batch> processing_handles` (each holding `std::unique_lock<std::shared_mutex>` on the batch) for the lifetime of `op->execute()`. Operator code in `execute()` (migrated by plans 18-03 and 18-04) takes scoped `to_read_only()` / `to_mutable()` accessors on the SAME batches — same-thread recursive lock attempt on a non-recursive `shared_mutex` is detected and aborts.

### [mgpu_stress] default-mode
- **Result:** NOT RUN — precondition (clean [mgpu] run) failed.
- Would deadlock identically since SCHED-RR exercises the same `processing_handles` path.

### compute-sanitizer racecheck
- **Invocation (Bash + timeout, NOT MCP per project memory):**
  ```bash
  timeout 600 /usr/local/cuda-13.0/bin/compute-sanitizer --tool racecheck \
    build/release/extension/sirius/test/cpp/sirius_unittest "[downgrade_lifecycle]"
  ```
- **Result:** `========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` — `racecheck hazards=0`.
- **Note 1:** `[mgpu_foundation]` tag from the plan does not exist in the test suite; closest non-deadlocking proxy `[downgrade_lifecycle]` (8 test cases, 53 assertions, all pass) used instead.
- **Note 2:** compute-sanitizer racecheck detects CUDA/GPU races (kernel-level memory aliasing, shared-memory hazards) — it does NOT detect CPU-side `std::shared_mutex` self-deadlocks. The 0-hazard result is genuine evidence that the GPU-side migration is race-clean; the runtime FAIL is purely CPU-side P1 deadlock from R5 lock-and-hold.

## Pitfall Compliance Audit

- **P1 (RAII lock scope):** **VIOLATED at runtime.** 18-03 SUMMARY explicitly flagged this as a deferred runtime audit; this plan confirms the deadlock fires under load. Resolution path: architectural — either (a) add `get_locked_accessors()` to `pipelineable_operator_data` so `execute()` reads through the held accessor without re-locking, OR (b) drop R5 lock-and-hold semantics and have `prepare_for_processing` release locks before returning (operators take their own scoped accessors).
- **P3 (pop_next_data_batch):** Compile-time gate PASS (zero `pop_data_batch.*task_created|in_transit` hits). TPC-H parquet correctness check deferred to Phase 21 REG-02 per ROADMAP light-gate scoping. The [mgpu] filter would have been the smoke proxy for P3 but cannot run due to P1 blocker.
- **P7 (PR #739 × #117):** PASS (zero `data_batch_processing_handle` re-introductions in src/ or test/ outside descriptive comments).

## Hand-off to Phase 19

- Cucascade pin still `1c1e648` (defended by Phase 17 D-G6).
- Build clean against post-#117 RAII (with liburing-dev installed via `pixi install`).
- Phase 19 IO Framework adoption can begin **at compile-time**: install liburing-dev as first step is now satisfied; uring_reactor.cpp builds clean; sirius_datasource adoption work can proceed against clean compile state.
- **CRITICAL BLOCKER FOR PHASE 19+ RUNTIME GATES:** P1 deadlock must be resolved before any [mgpu] / runtime regression tests can pass. Phase 19's compile-time work is unblocked but its runtime gates (and Phase 21 REG-XX) inherit the P1 blocker. Recommended: address P1 architectural fix BEFORE Phase 19 runtime testing.
- Open follow-ups:
  - **P1 architectural fix** (mandatory pre-Phase-21): drop R5 lock-and-hold OR expose accessors via operator_data interface.
  - `mark_task_created` Sirius-method renaming (not done; Phase 18 carryover).
  - `readonly_to_mutable` demotion opportunity from RESEARCH.md Open Question 1.
  - `convertible_data_batch` readonly path optimization.
  - Phase 21 REG-02 [TPC-H][parquet] correctness check (deferred from this plan per scope).

## Files Modified This Phase

Aggregated from 18-01..18-05 SUMMARYs and 18-06 Task 0 prelude:

**src/ production code (35+ files):**
- `src/include/pipeline/batch_lock_utils.hpp` (full rewrite — 18-01)
- `src/include/op/sirius_physical_operator.hpp` + `src/op/sirius_physical_operator.cpp`
- `src/include/op/scan/parquet_scan_operator_data.hpp`
- `src/include/data/convertible_data_batch.hpp` + `convertible_gpu_pipeline_task.hpp` + `data_batch_utils.hpp`
- `src/include/pipeline/gpu_pipeline_task.hpp` + `src/pipeline/gpu_pipeline_task.cpp` + `gpu_pipeline_executor.cpp`
- 8 stateful operator .cpp (table_scan, hash_join, nested_loop_join, concat, top_n, grouped_aggregate_merge, ungrouped_aggregate, merge_sort)
- 6 read-only operator .cpp (filter, projection, limit, partition, sort_sample, result_collector)
- 5 scan-layer files (sirius_gpu_parquet_scan_operator, parquet_scan_task, duckdb_scan_task, cpu_source_task, duckdb_scan_executor)
- 8 inventory-miss .cpp (sort_partition, grouped_aggregate, sirius_physical_order, gpu_aggregate_impl, gpu_merge_impl, gpu_order_impl, gpu_partition_impl, cached_split_provider)
- `src/creator/task_creator.cpp`, `src/debug_utils.cpp`, `src/include/debug_utils.hpp`

**test/cpp/ (31 files):**
- 23 test files migrated by plan 18-05.
- 8 inventory-miss test files migrated by 18-06 Task 1 (Rule 3 Blocking): `test_downgrade_lifecycle.cpp`, `test_downgrade_disk.cpp`, `test_downgrade_executor.cpp`, `test_context.cpp`, `test_host_table_utils.cpp`, `test_gpu_merge_impl.cpp`, `test_gpu_partition_impl.cpp`, `test_utils.hpp` (commit `43a9565`).

## Plan-by-Plan Status

| Plan | Status | Build error count delta | Notes |
|------|--------|-------------------------|-------|
| 18-01 | PASS | 63 → 58 | DB-01 closed; header-first ripple |
| 18-02 | PASS | 58 → 47 | Operator base layer + R5 lock-and-hold (P1 risk introduced) |
| 18-03 | PASS | 47 → 21 | 8 stateful operators; P1 risk documented |
| 18-04 | PASS | src/ → 8 inventory-miss + 6 liburing | Read-only operators + Pitfall 4 closure |
| 18-05 | PASS | 8 inventory-miss + 23 test files migrated | DB-03 closure |
| 18-06 | PARTIAL | + 8 inventory-miss test files (Rule 3); MCP build exit 0 | DB-04 PASS; DB-05 FAIL — P1 deadlock fires |
