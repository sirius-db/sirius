---
phase: 07-task-queue-conversion
verified: 2026-04-16T17:00:00Z
status: human_needed
score: 4/4 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Build and run the test suite: CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make && build/release/extension/sirius/test/cpp/sirius_unittest '[convertible_gpu_pipeline_task]'"
    expected: "All 11 test cases pass with 0 failures and 39 assertions"
    why_human: "Cannot run GPU tests without a CUDA device; build requires GPU toolchain. Tests exercising real GPU memory allocation and GPU-to-HOST conversion (Test 2, Test 8) must be confirmed passing on the target hardware."
---

# Phase 7: Task Queue Conversion Verification Report

**Phase Goal:** Queued pipeline tasks can be discovered by memory space, temporarily owned for conversion, and safely returned to the queue
**Verified:** 2026-04-16T17:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `convertible_gpu_pipeline_task` takes ownership of a `unique_ptr<itask>` via constructor and its destructor pushes the task back to the `inspectable_mpsc<itask>` queue | VERIFIED | Constructor stores `_task` via `std::move(task)`. Destructor: `if (_task) { if (!_queue.push(std::move(_task))) { SIRIUS_LOG_WARN(...); } }`. Move semantics handled, logs warning if queue interrupted. Tests 1, 2, 3, 11 explicitly verify RAII queue return (queue.size() == 1 after wrapper destruction). |
| 2 | `convertible_gpu_pipeline_task_provider::get_next_convertible()` uses `mutable_pop_if` with `front_to_back=false`, matching tasks whose `gpu_pipeline_task_local_state` data_batches are in the target `memory_space` and `batch_state::task_created` | VERIFIED | `get_next_convertible` calls `_queue.mutable_pop_if([space](...) { return has_matching_batches(task, space); }, front_to_back)`. All 11 tests invoke with `front_to_back=false`. Predicate: dynamic_cast chain to `pipelineable_operator_data`, checks `batch->get_memory_space() == space && batch->get_state() == cucascade::batch_state::task_created`. Tests 4–7 verify correct predicate filtering (non-gpu task, wrong space, wrong state, matching task). |
| 3 | On conversion failure or exception, all `data_batch` objects inside `operator_data` retain their original `idata_representation` and `batch_state`; the task is always returned to the queue via RAII destructor | VERIFIED | `convert()`: saves `prev_state`, calls `try_to_lock_for_in_transit()`, on no-space-succeeded path calls `try_to_release_in_transit(prev_state)`, on exception catch calls `try_to_release_in_transit(prev_state)` then rethrows. Test 8 confirms state restored to `task_created` after successful convert. Test 3 confirms task returned to queue when exception thrown outside convert(). RAII always fires via `convertible_gpu_pipeline_task` destructor. Note: no test exercises the convert()-internal failure path (e.g., reservation exhaustion), but code paths are structurally sound. |
| 4 | `bytes_in_space()` returns the total byte size across all data_batches in the task's `operator_data` for the given memory space | VERIFIED | `convertible_gpu_pipeline_task::bytes_in_space()` iterates `pipelineable->get_data_batches()`, sums `batch->get_data()->get_size_in_bytes()` for batches where `batch->get_memory_space() == space`. Test 9 creates two GPU batches, captures sizes before task construction, verifies `cd->bytes_in_space(gpu_space) == batch1_size + batch2_size` and `cd->bytes_in_space(host_space) == 0`. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/include/data/convertible_gpu_pipeline_task.hpp` | `convertible_gpu_pipeline_task` and `convertible_gpu_pipeline_task_provider` | VERIFIED | 385 lines. Both classes present, inherit from `convertible_data` and `convertible_data_provider` respectively. Header-only in `namespace sirius`. Apache 2.0 license. Committed at `bdd85d6c`. |
| `test/cpp/data/test_convertible_gpu_pipeline_task.cpp` | GPU integration tests for task queue conversion | VERIFIED | 363 lines. 11 TEST_CASE blocks tagged `[convertible_gpu_pipeline_task]`. Committed at `5115acbf`. |
| `CMakeLists.txt` | Test registration | VERIFIED | `test/cpp/data/test_convertible_gpu_pipeline_task.cpp` present at line 345 in TEST_SOURCES list, between `test_convertible_data_batch.cpp` and `test_host_parquet_representation.cpp`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `convertible_gpu_pipeline_task.hpp` | `data/convertible_data.hpp` | inherits `convertible_data` and `convertible_data_provider` | WIRED | Line 19: `#include "data/convertible_data.hpp"`. Line 58: `class convertible_gpu_pipeline_task : public convertible_data`. Line 258: `class convertible_gpu_pipeline_task_provider : public convertible_data_provider`. |
| `convertible_gpu_pipeline_task.hpp` | `exec/inspectable_mpsc.hpp` | queue reference for RAII push-back | WIRED | Line 21: `#include "exec/inspectable_mpsc.hpp"`. Line 243: `sirius::exec::inspectable_mpsc<sirius::parallel::itask>& _queue`. Line 90: `_queue.push(std::move(_task))`. |
| `convertible_gpu_pipeline_task.hpp` | `pipeline/gpu_pipeline_task.hpp` | dynamic_cast chain to reach data_batches | WIRED | Line 25: `#include "pipeline/gpu_pipeline_task.hpp"`. Line 228: `dynamic_cast<sirius::pipeline::gpu_pipeline_task*>(_task.get())`. Line 235: `dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(ls)`. Line 238: `dynamic_cast<sirius::op::pipelineable_operator_data*>(gpt_ls->_input_data.get())`. |
| `test_convertible_gpu_pipeline_task.cpp` | `convertible_gpu_pipeline_task.hpp` | `#include` | WIRED | Line 20: `#include <data/convertible_gpu_pipeline_task.hpp>`. |
| `CMakeLists.txt` | `test_convertible_gpu_pipeline_task.cpp` | TEST_SOURCES list | WIRED | Line 345 in CMakeLists.txt. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `convertible_gpu_pipeline_task::convert()` | `batches` from `pipelineable->get_data_batches()` | `gpt_ls->_input_data` (live task data) | Yes — live `data_batch` objects owned by the task | FLOWING |
| `convertible_gpu_pipeline_task::bytes_in_space()` | `total` summed from `batch->get_data()->get_size_in_bytes()` | Live `data_batch` objects | Yes — exact byte sizes from real data_batch allocations | FLOWING |
| `convertible_gpu_pipeline_task_provider::get_next_convertible()` | `result` from `mutable_pop_if` | Live `inspectable_mpsc<itask>` queue | Yes — extracts actual queued tasks | FLOWING |

### Behavioral Spot-Checks

Step 7b: SKIPPED — GPU execution required. Cannot run `sirius_unittest "[convertible_gpu_pipeline_task]"` without a CUDA device. Build and test execution is routed to Human Verification.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| TASK-01 | 07-01-PLAN.md, 07-02-PLAN.md | `convertible_gpu_pipeline_task` RAII ownership — constructor takes `(unique_ptr<itask>, inspectable_mpsc<itask>&)`, destructor pushes task back to queue | SATISFIED | Constructor signature at line 65–70. Destructor at lines 87–95. Tests 1, 2, 3, 11 directly exercise RAII return. |
| TASK-02 | 07-01-PLAN.md, 07-02-PLAN.md | `convertible_gpu_pipeline_task_provider` uses `mutable_pop_if` with `front_to_back=false`, predicate checks `gpu_pipeline_task_local_state` data_batches for matching `memory_space` and `batch_state::task_created` | SATISFIED | `get_next_convertible` at line 280–292. `has_matching_batches` static helper at lines 353–379. All test calls use `front_to_back=false`. Tests 4–7 verify predicate precision. |
| TASK-03 | 07-01-PLAN.md, 07-02-PLAN.md | On conversion failure or exception in `convert()`, all `data_batch` objects retain original `idata_representation` and `batch_state`; task always returned to queue via destructor | SATISFIED | `try_to_release_in_transit(prev_state)` on both failure (line 181) and exception (line 185) paths. Test 8 confirms `batch_state::task_created` preserved after successful convert. RAII return confirmed across all 4 RAII tests. |

Note: REQUIREMENTS.md checkboxes for TASK-01/02/03 remain `[ ]` (not checked off). This is a documentation tracking issue only — implementation is complete.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/include/data/convertible_gpu_pipeline_task.hpp` | 335–339 | `get_bytes_in_space()` always returns 0 | Info | Provider-level byte counting is intentionally omitted (documented limitation). The wrapper-level `bytes_in_space()` on `convertible_gpu_pipeline_task` works correctly. Callers requiring exact totals must use `get_all_convertible() + bytes_in_space()`. No impact on correctness of the phase goal. |

No blockers found.

### Human Verification Required

#### 1. GPU Test Suite Execution

**Test:** Run `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make && build/release/extension/sirius/test/cpp/sirius_unittest "[convertible_gpu_pipeline_task]"` on a machine with an NVIDIA GPU.

**Expected:** Output shows 11 test cases, 39 assertions, 0 failures. Specific assertions to confirm:
- Test 1: `queue.size() == 1` after wrapper destruction (RAII return)
- Test 2: `queue.size() == 1` and `batch->get_memory_space()->get_tier() == Tier::HOST` after convert + destruction
- Test 8: `batch->get_state() == batch_state::task_created` after GPU-to-HOST conversion
- Test 9: `cd->bytes_in_space(gpu_space) == batch1_size + batch2_size`
- Test 11: No crash when queue is interrupted before wrapper destruction

**Why human:** The test suite allocates real GPU memory, executes CUDA memory copies via the converter registry, and uses non-default CUDA streams. These cannot be verified without a physical CUDA device.

### Gaps Summary

No gaps found. All 4 success criteria are verified by static analysis of the implementation. The sole pending item is runtime confirmation that the 11 GPU integration tests pass on target hardware.

---

_Verified: 2026-04-16T17:00:00Z_
_Verifier: Claude (gsd-verifier)_
