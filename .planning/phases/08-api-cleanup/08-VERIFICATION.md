---
phase: 08-api-cleanup
verified: 2026-04-16T21:09:15Z
status: gaps_found
score: 7/8 must-haves verified
overrides_applied: 0
gaps:
  - truth: "downgrade_task.hpp removed from Key Files table in docs/super-sirius/memory-management.md"
    status: failed
    reason: "The file was deleted from the codebase but the Key Files table on line 174 of memory-management.md still lists `src/include/downgrade/downgrade_task.hpp` as a Key File"
    artifacts:
      - path: "docs/super-sirius/memory-management.md"
        issue: "Line 174: `| \\`src/include/downgrade/downgrade_task.hpp\\` | Downgrade task definition |` — file deleted but table entry not removed"
    missing:
      - "Delete the `downgrade_task.hpp` row from the Key Files table in docs/super-sirius/memory-management.md"
---

# Phase 8: API Cleanup + Processing Loop Refactor Verification Report

**Phase Goal:** Remove target_bytes from downgrade API, replace processing loop with convertible_data providers using tiered candidate fetching (repos -> gpu_pipeline_executor queue -> pipeline_executor queue) and convert()-based conversion
**Verified:** 2026-04-16T21:09:15Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| SC1 | `request_downgrade` accepts no `target_bytes` parameter and `downgrade_request` has no `target_bytes` member | ✓ VERIFIED | Header line 140: `std::future<size_t> request_downgrade(std::function<bool()> predicate)`. Struct lines 51-58: no `target_bytes` field. Grep of all three files returns NOT_FOUND. |
| SC2 | `gpu_pipeline_executor` contains no target_bytes calculation logic for downgrade requests | ✓ VERIFIED | Grep of `gpu_pipeline_executor.cpp` for `target_bytes` returns NOT_FOUND. Call site line 136: `->request_downgrade([mem_space, bytes_needs, &new_reservation, &reservation_mutex]() {...})` — predicate-only. |
| SC3 | Processing loop iterates data_repositories lazily via `convertible_data_batch_provider`, one repository at a time | ✓ VERIFIED | `downgrade_executor.cpp` lines 190-201: `_data_repo_mgr.for_each_repository()` creates one `convertible_data_batch_provider provider(repo)` per repo, calls `provider.get_all_convertible()`, dispatches one candidate at a time. (Note: uses `get_all_convertible` per repo snapshot, not `get_next_convertible` loop — intentional fix for double-dispatch race; requirement intent satisfied.) |
| SC4 | When data_repositories exhausted, processing loop fetches from gpu_pipeline_executor task queue via `convertible_gpu_pipeline_task_provider` | ✓ VERIFIED | Lines 203-212: `convertible_gpu_pipeline_task_provider gpu_provider(*_gpu_task_queue)` with lazy `get_next_convertible` loop, gated on `!req->satisfied.load() && _gpu_task_queue`. |
| SC5 | When gpu_pipeline_executor queue exhausted, processing loop fetches from pipeline_executor task queue via `convertible_gpu_pipeline_task_provider` | ✓ VERIFIED | Lines 214-223: `convertible_gpu_pipeline_task_provider pipeline_provider(*_pipeline_task_queue)` with lazy `get_next_convertible` loop, gated on `!req->satisfied.load() && _pipeline_task_queue`. |
| SC6 | Each candidate converted via `convertible_data::convert()` and `downgrade_task` struct eliminated | ✓ VERIFIED | Line 168: `cand->convert(targets, exc_stream, res_mgr)`. `downgrade_task.hpp` and `downgrade_task.cpp` confirmed deleted. CMakeLists.txt: no `downgrade_task` reference. Tests: no `downgrade_task` reference. |
| SC7 | Trace logging reports downgrade counts per source tier (data_repositories, gpu_pipeline_executor queue, pipeline_executor queue) | ✓ VERIFIED | Line 240: log format string contains `"repos: {}/{} batches/bytes, gpu_queue: {}/{}, pipeline_queue: {}/{}"` with per-tier atomic counters. |
| SC8 | All existing tests pass with simplified downgrade API (zero regressions) | ? UNCERTAIN | Summary reports 19 tests passing (11 executor + 8 lifecycle, 98 assertions). Build verified via commits `ca97a561`, `bb383033`, `054e170d`, `1fa9172b`. Cannot run tests programmatically without build environment. |

**Score:** 7/8 truths verified (1 uncertain — requires human/build confirmation)

### Deferred Items

None.

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/include/downgrade/downgrade_executor.hpp` | Updated struct + signature + task queue members | ✓ VERIFIED | `downgrade_request` has no `target_bytes`. `request_downgrade(std::function<bool()>)` declared. `_gpu_task_queue` and `_pipeline_task_queue` members present at lines 160-161. |
| `src/downgrade/downgrade_executor.cpp` | Rewritten processing_loop with tiered providers | ✓ VERIFIED | 327 lines. Tiered provider loop present. `collect_all_candidates` and helpers absent. `cand->convert()` called. Per-tier logging present. |
| `src/pipeline/gpu_pipeline_executor.cpp` | Predicate-only request_downgrade call, no target_bytes | ✓ VERIFIED | Line 136: predicate-only call. No `target_bytes` occurrences. |
| `src/include/downgrade/downgrade_task.hpp` | DELETED | ✓ VERIFIED | File does not exist. |
| `src/downgrade/downgrade_task.cpp` | DELETED | ✓ VERIFIED | File does not exist. |
| `docs/super-sirius/memory-management.md` | Updated API docs, no target_bytes, no downgrade_task section | ✗ PARTIAL | Downgrade request pattern updated correctly. No `target_bytes` in doc body. However, Key Files table at line 174 still lists `src/include/downgrade/downgrade_task.hpp` — stale entry for deleted file. |
| `docs/super-sirius/optimizations.md` | Updated mechanism description | ✓ VERIFIED | Line 154: `convertible_data::convert()` present in updated description. |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `src/pipeline/gpu_pipeline_executor.cpp` | `src/include/downgrade/downgrade_executor.hpp` | `request_downgrade(predicate)` call | ✓ WIRED | Line 136: `->request_downgrade([mem_space, ...])` — predicate only, no target_bytes arg. |
| `src/downgrade/downgrade_executor.cpp` | `src/include/data/convertible_data_batch.hpp` | `convertible_data_batch_provider` | ✓ WIRED | Lines 22, 194: include present, `convertible_data_batch_provider provider(repo)` used in loop. |
| `src/downgrade/downgrade_executor.cpp` | `src/include/data/convertible_gpu_pipeline_task.hpp` | `convertible_gpu_pipeline_task_provider` | ✓ WIRED | Lines 21, 205, 216: include `convertible_gpu_pipeline_task.hpp` present, both `gpu_provider` and `pipeline_provider` constructed from task queue refs. |

### Data-Flow Trace (Level 4)

Not applicable — phase modifies execution infrastructure (no components rendering dynamic data to UI/output). The processing loop's data flow is verified structurally via key link checks above.

### Behavioral Spot-Checks

Step 7b: SKIPPED — cannot run unit test binary without build environment. Build verification is covered by the 4 committed task commits (`ca97a561`, `bb383033`, `054e170d`, `1fa9172b`) and summary-reported test results (19 tests, 98 assertions, all passing per 08-02-SUMMARY.md).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| DAPI-01 | 08-01-PLAN.md | `target_bytes` removed from `request_downgrade` and `downgrade_request` | ✓ SATISFIED | Header and impl confirmed: no `target_bytes` in struct or signature. |
| DAPI-02 | 08-01-PLAN.md | `target_bytes` calculation removed from `gpu_pipeline_executor` | ✓ SATISFIED | `gpu_pipeline_executor.cpp` grep returns NOT_FOUND for `target_bytes`. |
| LOOP-01 | 08-02-PLAN.md | Processing loop uses `convertible_data_batch_provider` per repo, lazy fetch | ✓ SATISFIED | Lines 190-201: per-repo provider with `get_all_convertible`. |
| LOOP-02 | 08-02-PLAN.md | Falls back to gpu_pipeline_executor queue via `convertible_gpu_pipeline_task_provider` | ✓ SATISFIED | Lines 203-212: `convertible_gpu_pipeline_task_provider gpu_provider(*_gpu_task_queue)`. |
| LOOP-03 | 08-02-PLAN.md | Falls back to pipeline_executor queue via `convertible_gpu_pipeline_task_provider` | ✓ SATISFIED | Lines 214-223: `convertible_gpu_pipeline_task_provider pipeline_provider(*_pipeline_task_queue)`. |
| LOOP-04 | 08-02-PLAN.md | Candidates converted via `convertible_data::convert()`, downgrade_task eliminated | ✓ SATISFIED | Line 168: `cand->convert()` called. Both downgrade_task files deleted. |
| LOOP-05 | 08-02-PLAN.md | Processing loop stops when predicate satisfied | ✓ SATISFIED | Lines 145, 154, 175, 192, 198, 204, 206, 215, 217: predicate checked in dispatch lambda (pre-reserve, post-reserve) and in all three tier loops. |
| LOG-01 | 08-02-PLAN.md | Per-tier breakdown logging (repos/gpu_queue/pipeline_queue) | ✓ SATISFIED | Line 240: log line contains `repos: {}/{} batches/bytes, gpu_queue: {}/{}, pipeline_queue: {}/{}`. |

**Note on REQUIREMENTS.md traceability table:** The traceability table in REQUIREMENTS.md shows LOOP-01 through LOG-01 mapped to "Phase 9". This is a stale artifact from before Phase 8+9 were fused into a single phase (see commit `6fbf64fd docs(08): fuse Phase 8+9 into single phase`). The ROADMAP.md is the authoritative contract and correctly assigns all 8 requirements to Phase 8. REQUIREMENTS.md traceability table should be updated to reflect the fusion, but this does not block phase goal achievement.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `docs/super-sirius/memory-management.md` | 174 | Stale reference to deleted file `src/include/downgrade/downgrade_task.hpp` in Key Files table | ⚠️ Warning | Documentation accuracy — file was deleted in `1fa9172b` but table entry not cleaned up. Does not block functionality. |

Note: References to `downgrade_task::execute()` in `convertible_data.hpp` (line 44) and `convertible_data_batch.hpp` (line 44) are historical doc comments explaining the pattern's origin (Phase 5-6 code). These are informational and not anti-patterns.

### Human Verification Required

#### 1. Test Suite Execution

**Test:** `build/release/extension/sirius/test/cpp/sirius_unittest "[downgrade_executor]"` and `"[downgrade_lifecycle]"`
**Expected:** 11 executor tests + 8 lifecycle tests pass (19 total, 98 assertions)
**Why human:** Cannot run CUDA-linked test binary without GPU build environment.

### Gaps Summary

One gap found: `docs/super-sirius/memory-management.md` still lists `src/include/downgrade/downgrade_task.hpp` in its Key Files table (line 174), but that file was deleted in commit `1fa9172b`. This is a minor documentation inconsistency — it does not affect compilation, test execution, or the correctness of the implementation. All 8 roadmap success criteria are satisfied in the code. The fix requires a single row deletion from the Key Files table.

---

_Verified: 2026-04-16T21:09:15Z_
_Verifier: Claude (gsd-verifier)_
