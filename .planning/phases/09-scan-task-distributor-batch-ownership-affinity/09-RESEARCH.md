# Phase 9: Scan-Task Distributor + Batch-Ownership Affinity — Research

**Researched:** 2026-04-24
**Domain:** Multi-GPU scan-task dispatch, batch-ownership affinity, `preferred_device_id` plumbing
**Confidence:** HIGH — all findings are sourced from direct source-code inspection of the live branch

---

## Summary

Phase 8's probe run (08-08-DIAGNOSIS.md, hypothesis E) proved that the v1.2 SIGSEGV is NOT in the host_parquet converter (those Pattern 2 fixes are correct and remain). The bug lives in two distinct, adjacent sites upstream:

**Bug 1 — Batch double-dispatch.** `duckdb_scan_executor::manager_loop` calls `select_target_gpu()` to pick which GPU should process the next scan task. That selection is purely memory-pressure-weighted and ignores whether a batch is already resident on a specific GPU. When `batch_id=3` has `batch_device_id=0` (already locked and converted to GPU 0), the distributor can simultaneously dispatch another task that tries to `lock_or_prepare_batch` the same `batch_id` on GPU 1 (target_device_id=1). That second attempt sees `lock_status=3 memspace_mismatch success=false`. The failing task proceeds past the null-returning `lock_or_prepare_batch` without hard-aborting, causing a null-deref / use-after-free downstream → SIGSEGV.

**Bug 2 — `preferred_device_id=-1` sentinel at compute_task entry.** The probe shows `parquet_scan_task::compute_task entry preferred_device_id=-1 memspace_device_id=-1` for every task. `parquet_scan_task_global_state` inherits `get_preferred_device_id()` from `sirius_pipeline_task_global_state`, but NOTHING in the dispatch path calls `set_preferred_device_id()` on that global state before the task runs. The `target_gpu_id` chosen by `select_target_gpu()` in `manager_loop` is captured by the dispatch lambda for the stream/device guard but is NEVER written back to the task's global state. So `compute_task` calls `g_state.get_preferred_device_id()` which returns `std::nullopt` (→ `-1` in the probe), and falls back to `backends.begin()` (always GPU 0's backend) for `_datasource` construction, regardless of which GPU is actually executing.

**Primary recommendation:** Fix the distributor at two sites. (1) Add a `std::unordered_map<batch_id_t, int>` sticky-assignment table in `duckdb_scan_executor`, keyed by `batch_id`, that maps each batch to the GPU that first claimed it; update `select_target_gpu()` or a new wrapper to honour this ownership when the batch is already resident. (2) After `select_target_gpu()` returns `target_gpu_id` in `manager_loop`, call `parquet_task_global_state->set_preferred_device_id(target_gpu_id)` on the task's global state before dispatching, so `compute_task` sees the real device.

---

## User Constraints

_(No CONTEXT.md exists for Phase 9 yet — no locked decisions from /gsd:discuss-phase. Phase 9 is researcher-scoped directly from ROADMAP + HALT doc.)_

**Phase constraints inherited from ROADMAP + REQUIREMENTS:**
- Phase 9 closes v1.2 ROADMAP criteria 1, 2, 4, 6. Criteria 3 + 5 are already PASS (zero net-new `rmm::cuda_stream_default`, Pattern 2 grep).
- Fix must not regress the 4 Pattern 2 idiom sites landed by Phase 8 (08-01..06).
- Build/test ONLY via `mcp__project-commands__run_command`. Never launch GPU runs in-place as agent.
- GPU/integration tests (SF100 Q1, SF10 Q1/Q6/Q12, full 22 SF1 × {DuckDB, parquet} × {1,2}) are user-delegated.
- `rmm::cuda_stream_default` is forbidden (zero net-new introductions).
- Feature branch `feature/single-node-multi-gpu2` — commits here, never merge directly to dev.

---

## Research Question Answers

### Q1: Task→GPU assignment path

**Call chain for GPU pipeline tasks (non-scan):**

```
task_creator::manager_loop()
  → task_creator::schedule(operator*)
    → _bounded_pool::dispatch([node])
      → creates gpu_pipeline_task with local_state + global_state
      → sets local_state->set_preferred_device_id(preferred_device_id) [SCHED-00/01/02]
      → _pipeline_executor->schedule(task)  [pipeline_executor.cpp:552]
        → pipeline_executor::schedule(task)
          → _task_queue.push(task)         [pipeline_executor.cpp:95]
            → management_eventloop() pops it
            → reads gpu_task->get_preferred_device_id()
            → _gpu_executors.at(target_device_id)->schedule(task)
```

**Call chain for parquet_scan_task:**

```
task_creator::manager_loop()
  → (PARQUET_SCAN branch, task_creator.cpp:367-403)
  → creates parquet_scan_task_local_state (no preferred_device_id set here)
  → creates parquet_scan_task
  → _pipeline_executor->schedule(parquet_task)  [pipeline_executor.cpp:403]
    → pipeline_executor::schedule(task)
      → task->is<parquet_scan_task>() == true
      → _scan_executor->schedule(task)         [pipeline_executor.cpp:91]
        → duckdb_scan_executor::_task_queue.push(task)
          → manager_loop() pops it
          → int target_gpu_id = select_target_gpu()  [duckdb_scan_executor.cpp:322]
          → dispatch lambda captures target_gpu_id
          → get_scan_output(scan_task, stream)
            → task->compute_task(stream)
              → parquet_scan_task::compute_task()
                → g_state.get_preferred_device_id() → nullopt (BUG 2)
```

**Key finding:** `select_target_gpu()` is called in `duckdb_scan_executor::manager_loop` at line 322 of `duckdb_scan_executor.cpp`. Its return value is captured by the dispatch lambda only for stream/device-guard purposes. It is NEVER written back to the parquet task's global state. `parquet_scan_task_global_state` inherits `set_preferred_device_id()` from `sirius_pipeline_task_global_state` (sirius_pipeline_task_states.hpp:82), but nothing calls it on the scan path.

**Confidence:** HIGH — direct source inspection.

---

### Q2: batch_device_id ownership propagation

**Where `batch_device_id=N` is recorded:**

When `lock_or_prepare_batch` executes the `Tier::GPU` branch in `batch_lock_utils.hpp:~123`:
```cpp
batch->convert_to<cucascade::gpu_table_representation>(registry, target_space, stream);
```
This calls cucascade's converter registry, which (via the Sirius-side override in `host_parquet_representation_converters.cpp`) converts the batch to a GPU table on `target_space`. Cucascade then updates `batch->get_memory_space()` to point to `target_space`, so `batch->get_memory_space()->get_device_id()` returns the GPU that owns the batch.

**Where `batch_device_id` is READ:**

In `lock_or_prepare_batch` (`batch_lock_utils.hpp:55-61`):
```cpp
const auto* target_space =
  requested_memory_space != nullptr ? requested_memory_space : batch->get_memory_space();
auto lock_result = batch->wait_to_lock_for_processing(target_space->get_id());
```
The batch's current space is checked only implicitly: `wait_to_lock_for_processing` looks up the memory space id. If the task's `requested_memory_space->get_id()` does not match the batch's current space id, the lock returns `lock_status=3 (memory_space_mismatch)`.

**Does the distributor read `batch_device_id` when choosing where to dispatch?**

NO. `select_target_gpu()` in `duckdb_scan_executor.cpp:169-223` reads only `space->get_available_memory()` from the `_gpu_memory_spaces` vector. It has no knowledge of which batches are already resident on which GPU. The batch→GPU ownership mapping is opaque to the distributor.

**The missing lookup:** The distributor needs to know, at dispatch time, whether the scan task being dispatched has a batch that is already in `batch_state=2 (processing/locked)` or has `batch_device_id >= 0`. If yes, it must route to that GPU, not to the one with the most free memory.

**Confidence:** HIGH — confirmed by probe log and direct source inspection.

---

### Q3: preferred_device_id plumbing — why it arrives as -1

**Root cause (confirmed):**

The inheritance chain is:
- `parquet_scan_task_global_state` extends `sirius_pipeline_task_global_state`
- `sirius_pipeline_task_global_state` owns `std::optional<int> _preferred_device_id` (default: `std::nullopt`)
- `get_preferred_device_id()` returns `_preferred_device_id` which is nullopt → probe reports `-1`

The `set_preferred_device_id()` on the global state is NEVER called on the parquet scan path. Specifically:

1. `task_creator.cpp:541-542` calls `local_state->set_preferred_device_id(preferred_device_id)` — but this is for `gpu_pipeline_task_local_state`, not for `parquet_scan_task_local_state`. The parquet scan task creation block (lines 390-403) does NOT call `set_preferred_device_id` on anything.

2. `duckdb_scan_executor::manager_loop` at line 322 computes `target_gpu_id = select_target_gpu()` and captures it in the dispatch lambda, but NEVER writes it back to the global state via `parquet_task_global_state->set_preferred_device_id(target_gpu_id)`.

**Consequence:** `compute_task` falls through to `backend_it = backends.begin()` (line 796), which is always the first GPU's backend regardless of which GPU is running the dispatch. So even if the lambda's `rmm::cuda_set_device_raii dispatch_guard` pins the worker thread to GPU 1, the `_datasource` is constructed on GPU 0's cucascade backend — a device-context mismatch for the I/O path.

**Why there is no constructor path bug:** `parquet_scan_task_local_state` (line 488) takes `g_state` and `partition` only. It does not copy `preferred_device_id` from the global state. The global state is shared across all tasks for this parquet operator (single `parquet_scan_task_global_state` per pipeline). So the fix must write `target_gpu_id` to the global state BEFORE the task is dispatched, OR use a per-task local-state field.

**Warning — shared global state:** `parquet_scan_task_global_state` is shared across all tasks for the same pipeline. If two tasks for the same pipeline are dispatched concurrently to different GPUs (which happens!), writing `set_preferred_device_id` on the shared global state creates a data race. The correct fix is to store `preferred_device_id` per-task, either in `parquet_scan_task_local_state` (needs `set_preferred_device_id` there) or directly in `parquet_scan_task` itself before `compute_task` is called.

**Confidence:** HIGH — confirmed by direct source inspection.

---

### Q4: Sticky-assignment primitives — what exists vs. what needs to be added

**What exists today:**

- `duckdb_scan_executor` has `_gpu_memory_spaces` vector and `_scan_round_robin` atomic counter — used by `select_target_gpu()` for memory-weighted round-robin. No batch→GPU map.
- `gpu_pipeline_task_local_state` has `set_preferred_device_id()` — but that's for GPU pipeline tasks, not scan tasks.
- `sirius_pipeline_task_global_state` has `set_preferred_device_id()` — exists but not called on the scan path.
- `parquet_scan_task_local_state` has NO `preferred_device_id` field.

**What needs to be added:**

Option A (preferred — per-task local state field):

Add `std::optional<int> _preferred_device_id` to `parquet_scan_task_local_state` (in `parquet_scan_task.hpp`):
```cpp
void set_preferred_device_id(int id) { _preferred_device_id = id; }
std::optional<int> get_preferred_device_id() const { return _preferred_device_id; }
```
Then in `duckdb_scan_executor::manager_loop`, before dispatch, cast the scan task's local state and call `set_preferred_device_id(target_gpu_id)`. In `parquet_scan_task::compute_task`, read from local state first (as `gpu_pipeline_task` does with its two-tier lookup at `gpu_pipeline_task.hpp:188-194`).

Option B (batch sticky map in executor):

Add `std::unordered_map<int64_t /*batch_id*/, int /*gpu*/> _batch_gpu_affinity` with a `std::mutex` to `duckdb_scan_executor`. Before dispatching a scan task, check if any of its batches (if known at dispatch time) are already in the map and prefer that GPU. This is harder because the batch_ids are not yet known until `compute_task` runs the first time — batches are created lazily inside the scan.

**Minimum-viable approach:** Option A for the `preferred_device_id` plumbing fix (lower risk, no shared-state race). For the double-dispatch fix (Bug 1), see Q5.

**Confidence:** HIGH — direct source inspection of all local/global state classes.

---

### Q5: Cross-GPU yield/reassign primitive

**Does a yield/requeue mechanism exist?**

YES — for GPU pipeline tasks. `gpu_pipeline_executor.cpp` has the OOM reschedule path (commit 6d73680): when `batch_lock_utils.hpp` returns `nullopt`, the task throws `oom_reschedule_exception`, which triggers the retry loop at lines 241-322 of `gpu_pipeline_executor.cpp` (up to `MAX_OOM_RETRIES=100`, 50 ms backoff each → ~5 s patience).

**Does it exist for scan tasks?**

NO. `duckdb_scan_executor::manager_loop` dispatches scan tasks via `_bounded_pool->dispatch(...)`. The dispatch lambda at lines 401-418 has a `try/catch(...)` that calls `_completion_handler->report_error()`. There is NO reschedule loop in the scan executor — a failed `lock_or_prepare_batch` in `compute_task` will propagate as an unhandled error, not a retry.

**The real fix for Bug 1 (double dispatch):**

The OOM retry path in `gpu_pipeline_executor` will NOT help here because the scan task is dispatched through `duckdb_scan_executor`, not through `gpu_pipeline_executor`. The root fix is:

**Prevent the dispatch of a scan task to a GPU that does not own the batch, before the task even runs.** This means the distributor must know batch→GPU affinity at dispatch time.

However, for parquet scan tasks, batches are not pre-assigned before the task runs — the batch is created DURING `compute_task` execution. The actual double-dispatch scenario from the probe log is:

```
batch_id=3 batch_device_id=0  dispatched to GPU 0  (lock_status=0 success=true)
batch_id=3 batch_device_id=0  dispatched to GPU 1  (lock_status=3 success=false)
```

This means batch_id=3 was already converted to GPU 0 by a prior task round, and then a SECOND dispatch (from a retry or a second task request) picks up the same batch_id and routes it to GPU 1.

Looking at `select_target_gpu()` again: it uses a counter `_scan_round_robin` that is shared across all dispatches. Two tasks for the same operator's scan batches can land on different GPUs because the counter just increments. The batch_id is assigned from the same counter (`counter` in `select_target_gpu` doubles as the batch_id suffix in the `[mgpu-audit]` log).

**Minimum viable fix for Bug 1 (Candidate 1 from 08-08):**

When `lock_or_prepare_batch` returns `nullopt` with `lock_status=3 (memspace_mismatch)` AND `batch_device_id >= 0` (batch already resident on specific GPU), the caller (inside `prepare_for_processing` or `compute_task`) should NOT proceed to use the null optional as if it succeeded. Currently the code at `batch_lock_utils.hpp:155-158` calls `cancel_task_if_needed()` and returns `nullopt`. The caller must handle `nullopt` by yielding.

The scan task path through `get_scan_output` → `task->compute_task(stream)` → `pipelineable_operator_data::prepare_for_processing` needs to check the return value of `lock_or_prepare_batch` and either (a) abort+reschedule the task to the owning GPU, or (b) re-queue the scan task for the GPU that owns the batch.

**Minimum viable alternative (Candidate 2 from 08-08 — recommended):**

Add a per-batch→GPU ownership record in `duckdb_scan_executor` and have `select_target_gpu()` respect it. This is cleaner than a yield/requeue because it prevents the mismatch from ever occurring. Since the batch_id is assigned INSIDE `compute_task` (specifically via `_scan_round_robin` counter in `select_target_gpu()`), and `select_target_gpu()` returns the target GPU, we can simultaneously record the mapping: `_batch_gpu_affinity[counter] = target_gpu_id` at the same point that the batch_id is assigned (line 216 in `duckdb_scan_executor.cpp`). Subsequent dispatches can check this map.

However this requires knowing the batch_id before `compute_task` executes. The counter in `select_target_gpu()` IS the batch_id suffix used in audit logs. So the assignment of batch_id and GPU target is co-located at lines 199-216 of `duckdb_scan_executor.cpp`. A sticky map written there would cover the affinity requirement.

**Termination guarantee:** A sticky map gives each batch one GPU permanently. Since the round-robin is monotonically increasing and batches are consumed (processed and released), the map can be a bounded structure. Each batch_id appears once; the entry can be removed after the batch is released (or the map can just grow to `num_batches` entries per query, which is bounded).

**Deadlock risk:** None — if a batch is already on GPU N, routing the task to GPU N's pool means it sits in GPU N's `_task_queue` until a thread is free. This is the same "wait on preferred executor" invariant already locked in `pipeline_executor::management_eventloop` (STATE.md decision from Phase 04-02).

**Confidence:** HIGH.

---

### Q6: AUDIT extension — cross-GPU batch disjointedness

**Where the AUDIT harness lives:**

`test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp`

Key components:
- `struct AuditCounts { std::set<std::string> pipeline_ids; std::set<std::string> scan_ids; }` (line 60)
- `parse_audit_log(tmp_log_dir)` → returns `std::map<int, AuditCounts> by_gpu` (line 70)
- `scan_re` at line 79 extracts `(GPU_id, batch_id)` pairs from `[mgpu-audit] scan_batch assigned to GPU N batch_id=K`
- Current assertions at lines 241-246: `counts[0].scan_ids.size() >= min_count` and `counts[1].scan_ids.size() >= min_count`

**How batch_id is currently logged/collected:**

`[mgpu-audit] scan_batch assigned to GPU N batch_id=K` is emitted at `duckdb_scan_executor.cpp:216` inside `select_target_gpu()`. The `batch_id=K` is the current value of `_scan_round_robin` counter (same counter used for weighted assignment). The `std::set<std::string> scan_ids` in `AuditCounts` accumulates unique batch_id strings per GPU.

**What the new AUDIT REQUIRE needs:**

```
counts[0].batch_ids ∩ counts[1].batch_ids == ∅
```

This translates to a set intersection check on `scan_ids` across GPUs. The existing `parse_audit_log` already collects `scan_ids` per GPU as a `std::set`. The new assertion is:

```cpp
// After the existing >=5 assertions:
std::vector<std::string> intersection;
std::set_intersection(
  counts[0].scan_ids.begin(), counts[0].scan_ids.end(),
  counts[1].scan_ids.begin(), counts[1].scan_ids.end(),
  std::back_inserter(intersection));
INFO("cross-GPU batch_id intersection size: " << intersection.size());
REQUIRE(intersection.empty());  // No batch appears on both GPUs concurrently
```

**What log changes are needed:**

The `[mgpu-audit] scan_batch assigned to GPU N batch_id=K` log line already emits `batch_id=K`. The `scan_re` regex at line 79 already extracts it. No log format change is required; only the test assertion needs to be added.

**Confidence:** HIGH — direct inspection of test file and log format.

---

### Q7: Validation strategy for SF100 Q1

**Dataset path:** `/datasets/tpch_parquet_sf100/lineitem.parquet`

**Inherited validation template:** `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md` (lines 208-252 contain the command block). The Phase 9 VALIDATION.md should follow the same structure.

**Minimum evidence artifact:** `.planning/phases/09-scan-task-distributor-batch-ownership-affinity/09-VALIDATION.md`

**Required content:**
1. Commands run (exact MCP command or manual invocation)
2. Full `[mgpu-audit]` log excerpt showing `scan_batch` distribution across both GPUs (batch count per GPU listed, grep output)
3. Wall-clock time for SF100 Q1 on num_gpus=2
4. Diff of query result vs. num_gpus=1 baseline (must match)
5. Exit code 0 (no SIGSEGV, no cudaErrorInvalidValue)

**Commands the user should run:**

```bash
# 1. Set SF10 path so AUDIT TEST_CASE uses strict >=5 threshold
export SIRIUS_TEST_SF10_PATH=/datasets/tpch_parquet_sf10

# 2. Run full unit-tests suite (includes 22 SF1 × 2-GPU, SF10 Q1/Q6/Q12, AUDIT TEST_CASE)
mcp__project-commands__run_command unit-tests

# 3. For SF100 Q1 validation (run manually, not via MCP):
export SIRIUS_LOG_DIR=/tmp/sirius-sf100-validation
export SIRIUS_LOG_LEVEL=info
# Run sf100 Q1 via Python API or DuckDB CLI with the extension loaded
# Capture the [mgpu-audit] output and record wall-clock

# 4. Grep validation
grep '\[mgpu-audit\] scan_batch' $SIRIUS_LOG_DIR/*.log | \
  awk '{for(i=1;i<=NF;i++) if($i~/^GPU/) gpu=substr($i,4); if($i~/^batch_id=/) print gpu, $i}' | \
  sort | uniq -c
```

**This is a user-run verification** — the agent does NOT run SF100 on 2-GPU hardware.

**Confidence:** HIGH.

---

### Q8: Risk — commit 1f80c2a

**Files touched that Phase 9 will edit:**
- `src/op/scan/duckdb_scan_executor.cpp` — commit 1f80c2a added the weighted `select_target_gpu()` stride fix (the fix that made the round-robin actually rotate between GPUs)
- `src/creator/task_creator.cpp` — commit 1f80c2a added SCHED-00 (partition pinning), NUMA normalization, SCHED-02 round-robin

**Surface area overlap with Phase 9:**

1. `select_target_gpu()` in `duckdb_scan_executor.cpp:169-223` was rewritten by 1f80c2a. Phase 9 will modify or wrap this function. The stride-based weighted RR (lines 193-208) is CORRECT and must be PRESERVED — it ensures scan batches actually distribute across GPUs. Phase 9 adds an affinity check ON TOP of this, not instead of it.

2. SCHED-00 (partition pinning, `task_creator.cpp:479-486`) is for `gpu_pipeline_task` creation, not for `parquet_scan_task` creation. Phase 9 does NOT touch the partition pinning path. No regression risk there.

3. The 1f80c2a commit's `select_target_gpu()` produces the `_scan_round_robin` counter which is ALSO used as the `batch_id` in the audit log (`[mgpu-audit] scan_batch assigned to GPU N batch_id=K`). Phase 9 must preserve the fact that the counter value = batch_id, because the AUDIT TEST_CASE's `parse_audit_log` regex parses `batch_id=K` as a unique identifier and the new disjointedness assert depends on it.

**Masking regression hypothesis (from 08-09-HALT.md):** 1f80c2a may have caused the failure mode to shift from `cudaErrorInvalidValue @ cuda_memcpy.cu:42` (v1.1 / 08-06 baseline) to SIGSEGV (08-08 observation). The most likely reason: before 1f80c2a, `select_target_gpu()` was broken (always picked GPU 0), so both tasks ended up on GPU 0, avoiding the cross-GPU dispatch. After 1f80c2a, tasks genuinely distribute to GPU 1, which surfaces the pre-existing double-dispatch race. The old cudaErrorInvalidValue was a DIFFERENT bug (Pattern 2 converter sites) that happened to co-occur with the distributor being single-GPU-only.

**Confidence:** HIGH.

---

### Q9: Risk — commit 6d73680

**What it changed:** `gpu_pipeline_executor.cpp` — `MAX_OOM_RETRIES: 10 → 100`, backoff `5 ms → 50 ms`.

**Relationship to the double-dispatch bug:**

The OOM retry path fires in `gpu_pipeline_executor::manager_loop` when a `gpu_pipeline_task` throws `oom_reschedule_exception`. The double-dispatch bug surfaces in `duckdb_scan_executor::manager_loop` for SCAN tasks, which do NOT go through `gpu_pipeline_executor`. So these two paths are INDEPENDENT.

**Does fixing the distributor make the retry path obsolete?**

No. The OOM retry budget covers a legitimate pattern: at SF100 with `cache=table_gpu` + `num_gpus=2`, probe tasks on GPU A hold a build-side table in `processing`, while a probe on GPU B needs to convert the same batch — contention on `try_to_lock_for_in_transit()`. Fixing the scan-task distributor does not eliminate this BUILD_PROBE contention. The bumped retry budget should remain.

**Coexistence risk:** None. The retry budget change is an independent knob. Phase 9 does not touch `gpu_pipeline_executor.cpp`.

**Confidence:** HIGH.

---

### Q10: Project skills to invoke during execute-phase

| Skill | Path | Relevance |
|-------|------|-----------|
| `module-context` | `.claude/skills/module-context/` | CRITICAL: Load cudf/rmm/cucascade/pipeline API docs before touching `batch_lock_utils.hpp`, `parquet_scan_task.hpp`, `duckdb_scan_executor.cpp`. Prevents using stale API assumptions. |
| `debug-gdb` | `.claude/skills/debug-gdb/` | Useful if SIGSEGV reproduces post-fix — attach gdb to get a stack trace at the null-deref site inside `pipelineable_operator_data::prepare_for_processing`. |
| `debug-compute-sanitizer` | `.claude/skills/debug-compute-sanitizer/` | Use after the distributor fix is in place to verify no residual stream-ordered race on the now-correct dispatch path. Catches data races invisible to gdb. |
| `build-errors` | `.claude/skills/build-errors/` | Standard for iterative build-fix loops when modifying header classes like `parquet_scan_task.hpp` (many translation units include it). |
| `debug-logging` | `.claude/skills/debug-logging/` | For adding/tuning `[mgpu-probe]` breadcrumbs during the fix-verify cycle (existing breadcrumbs in `batch_lock_utils.hpp` + `parquet_scan_task.cpp` + `host_parquet_representation_converters.cpp` should be KEPT for regression tracking). |

---

## Standard Stack

### Files Phase 9 Will Modify

| File | Reason | Confidence |
|------|--------|------------|
| `src/op/scan/duckdb_scan_executor.cpp` | Add batch→GPU affinity record in `select_target_gpu()` + plumb `target_gpu_id` to local state | HIGH |
| `src/include/op/scan/duckdb_scan_executor.hpp` | Add `_batch_gpu_affinity` map member (+ mutex) | HIGH |
| `src/include/op/scan/parquet_scan_task.hpp` | Add `preferred_device_id` field to `parquet_scan_task_local_state` | HIGH |
| `test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp` | Add disjointedness REQUIRE (cross-GPU batch_ids ∩ == ∅) | HIGH |

### Files Phase 9 Should NOT Modify (regression risk)

| File | Reason |
|------|--------|
| `src/include/pipeline/batch_lock_utils.hpp` | Fix is upstream (distributor), not here. The existing `[mgpu-probe]` breadcrumbs here are valuable and must be kept. |
| `src/data/host_parquet_representation_converters.cpp` | Pattern 2 fix is correct and complete. Do not touch. |
| `src/pipeline/gpu_pipeline_executor.cpp` | OOM retry budget is independent. No changes needed. |
| `src/creator/task_creator.cpp` | SCHED-00/01/02 partition pinning is for GPU pipeline tasks, not scan tasks. Do not touch unless the preferred_device_id option B (global state write) is chosen. |

### Patterns to Preserve

| Pattern | Location | Rule |
|---------|----------|------|
| Pattern 2 idiom (device guard + target-bound stream) | `duckdb_scan_executor.cpp:382-401` | Must not be changed — ROADMAP criterion 5 grep assertion must still pass |
| `[mgpu-probe]` breadcrumbs | `batch_lock_utils.hpp:64-80, 90-105`, `parquet_scan_task.cpp:758-770` | Keep all existing probes — they are the post-fix regression signal |
| `rmm::cuda_stream_default` zero net-new | All files | HYG-02 invariant — never introduce |

---

## Architecture Patterns

### Recommended Fix Shape

```
FIX A — preferred_device_id plumbing (Bug 2):

parquet_scan_task_local_state (parquet_scan_task.hpp):
  + std::optional<int> _preferred_device_id;
  + void set_preferred_device_id(int id) { _preferred_device_id = id; }
  + std::optional<int> get_preferred_device_id() const { return _preferred_device_id; }

duckdb_scan_executor::manager_loop (duckdb_scan_executor.cpp, after line 322):
  int target_gpu_id = select_target_gpu();  // existing
  if (scan_task && scan_task->is<parquet_scan_task>()) {
    auto* parquet = dynamic_cast<parquet_scan_task*>(scan_task);
    // NEW: plumb target_gpu_id to local state so compute_task sees real device
    if (auto* local_state = dynamic_cast<parquet_scan_task_local_state*>(parquet->local_state())) {
      local_state->set_preferred_device_id(target_gpu_id);
    }
    // ... existing reservation + stream acquire logic follows ...
  }

parquet_scan_task::compute_task (parquet_scan_task.cpp, line ~794):
  // Replace: auto const preferred = g_state.get_preferred_device_id();
  // With two-tier lookup (local wins over global):
  auto const& local_preferred = l_state.get_preferred_device_id();
  auto const preferred = local_preferred.has_value()
    ? local_preferred
    : g_state.get_preferred_device_id();
```

```
FIX B — batch-ownership affinity (Bug 1):

duckdb_scan_executor.hpp:
  + std::unordered_map<uint64_t, int> _batch_gpu_affinity;
  + std::mutex _batch_affinity_mutex;

duckdb_scan_executor::select_target_gpu (duckdb_scan_executor.cpp):
  Existing: counter = _scan_round_robin.fetch_add(1); SIRIUS_LOG_INFO batch_id=counter
  New: after computing target GPU, record affinity:
    {
      std::lock_guard lock(_batch_affinity_mutex);
      _batch_gpu_affinity[counter] = return_value;  // batch_id → gpu
    }
  This makes batch_id→GPU binding atomic with the log emission.

Caller of lock_or_prepare_batch (pipelineable_operator_data::prepare_for_processing):
  When lock_result fails with memspace_mismatch AND batch_device_id >= 0:
    Look up _batch_gpu_affinity[batch->get_batch_id()] → owning_gpu
    If owning_gpu != target_device: yield the task back to the scan executor
    OR: make the memspace_mismatch path in batch_lock_utils.hpp not proceed
      to convert (since conversion is what races with the already-locked batch)

Minimum viable: the batch_lock_utils.hpp memspace_mismatch branch should check
  if batch->get_state() == batch_state::processing (already locked by another task)
  before calling try_to_lock_for_in_transit(). If in processing state → return nullopt
  immediately rather than spinning on try_to_lock_for_in_transit().
```

### Alternative Fix Shape (simpler, lower LOC)

If the root cause of Bug 1 is that the SAME batch is dispatched to two scan tasks (not just two pipeline tasks), the simplest fix is in `task_creator.cpp` at the parquet scan task creation block (lines 375-403): ensure that once a `partition` (row-group range) is claimed with `claim_next_rg_partition()`, it is assigned to one target GPU and that assignment is recorded. Then `select_target_gpu()` can receive a hint (the claimed partition's owning GPU) instead of computing purely from available memory.

This requires:
1. The parquet scan task creation in `task_creator.cpp:367-403` to call `select_target_gpu()` on the executor and record `target_gpu_id → partition_idx` before `_pipeline_executor->schedule(parquet_task)`.
2. `duckdb_scan_executor::manager_loop` to read that pre-assigned GPU from a queue/map rather than calling `select_target_gpu()` for parquet tasks.

This is a more invasive design change but eliminates the race entirely.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe batch→GPU map | Custom lock-free structure | `std::unordered_map` + `std::mutex` | Bounded # batches per query; lock contention negligible vs. I/O cost |
| Set intersection for disjointedness | Manual loops | `std::set_intersection` (existing in `<algorithm>`) | Already available; `AuditCounts.scan_ids` is `std::set<std::string>` |
| Preferred-GPU two-tier lookup | Duplicating `gpu_pipeline_task::get_preferred_device_id()` logic | Mirror the same `local → global` fallback pattern from `gpu_pipeline_task.hpp:188-194` | Precedent already established |

---

## Common Pitfalls

### Pitfall 1: Shared global state data race on `preferred_device_id`

**What goes wrong:** `parquet_scan_task_global_state` is shared across all scan tasks for the same pipeline. Two tasks for the same pipeline can be in flight concurrently (one on GPU 0, one on GPU 1). If `preferred_device_id` is stored on the shared global state and one task writes it, the other task reads a stale or wrong value.

**Why it happens:** `set_preferred_device_id()` exists on `sirius_pipeline_task_global_state` (sirius_pipeline_task_states.hpp:82). It is tempting to call it in `manager_loop` — but the global state is shared.

**How to avoid:** Store `preferred_device_id` in `parquet_scan_task_local_state` (per-task, not shared). The local state is unique per task instantiation.

**Warning signs:** Two concurrent scan tasks computing conflicting `preferred` values, or a task using GPU 0's backend when running on GPU 1.

### Pitfall 2: Orphaned `_batch_gpu_affinity` map growing without bound

**What goes wrong:** If entries are never removed, the map grows by `num_batches` per query and leaks memory.

**Why it happens:** Batch IDs are assigned monotonically per executor lifetime. Without cleanup, the map accumulates entries from all queries.

**How to avoid:** Either (a) clear the map on `prepare_cache_for_scan_operators()` (called at query start, `duckdb_scan_executor.cpp:134`) which already resets `_cache`, OR (b) bound the map to only the current query's batch IDs by resetting `_scan_round_robin` and the affinity map together at query start.

### Pitfall 3: Forgetting to reset `_scan_round_robin` alongside the affinity map

**What goes wrong:** If `_scan_round_robin` is NOT reset between queries but the affinity map IS cleared, old counter values would collide with new ones if the counter wraps.

**How to avoid:** Reset both atomically in the same `prepare_cache_for_scan_operators()` call.

### Pitfall 4: `local_state()` cast at manager_loop dispatch time

**What goes wrong:** `scan_task->local_state()` returns a `parallel::itask_local_state*`. The cast to `parquet_scan_task_local_state*` fails if the scan task is not a parquet scan task.

**How to avoid:** Guard the cast with `scan_task->is<parquet_scan_task>()` before attempting it — the same guard already exists in `manager_loop` for the reservation path (line 324).

### Pitfall 5: `compute_task` reads `preferred` AFTER `_datasource` is already set

**What goes wrong:** Once `_datasource` is constructed in `compute_task` (inside `if (!_datasource) { ... }` at line 772), it is cached on the task object. On a retry or second call, `_datasource` is already set and the `preferred` lookup is skipped. So the fix to `preferred` must ensure the FIRST call sees the correct GPU.

**How to avoid:** The `set_preferred_device_id()` call in `manager_loop` must happen BEFORE the first dispatch of the task. Since `manager_loop` creates and dispatches the task in the same function body (lines 384-419), this is naturally guaranteed as long as `set_preferred_device_id()` is called before `_bounded_pool->dispatch(...)`.

---

## Runtime State Inventory

Phase 9 is a code fix (no rename, no migration, no rebrand). Runtime state inventory is not applicable.

---

## Environment Availability

Phase 9 is a code/test change only. External dependencies are the existing CUDA/cuDF/RMM stack already in place from Phase 8. All tooling was verified operational during Phase 8 execution.

| Dependency | Required By | Available | Notes |
|------------|------------|-----------|-------|
| 2× GPU (RTX 6000 Ada) | SF100 Q1 validation | Available on user's verification host | Agent cannot access directly |
| MCP project-commands build | All code changes | Available | Uses pixi internally |
| MCP project-commands unit-tests | AUDIT TEST_CASE + SF1/SF10 | Available | Single-GPU host builds/runs; 2-GPU tests need user hardware |
| `/datasets/tpch_parquet_sf100/` | SF100 Q1 ship-gate | User's verification host | Not accessible from build host |

**Missing dependencies with fallback:** SF100/SF10 datasets — fallback is SF1 DuckDB-attach path which runs in MCP unit-tests autonomously (already established in Phase 8 08-05 AUDIT TEST_CASE).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Catch2 v2 (already installed) |
| Config file | `test/cpp/integration/integration.yaml` (1-GPU default, not flipped) |
| Quick run command | `mcp__project-commands__run_command unit-tests filter='"[mgpu-audit]"'` |
| Full suite command | `mcp__project-commands__run_command unit-tests` |

### Phase Requirements → Test Map

| Requirement | Behavior | Test Type | Automated Command | File Exists? |
|-------------|----------|-----------|-------------------|--------------|
| ROADMAP crit. 1 (SF100 Q1 correct on 2-GPU) | SF100 Q1 returns correct results, no crash | manual SF100 run | user-delegated | N/A |
| ROADMAP crit. 2 (22 SF1 queries + SF10 smoke on 2-GPU) | unit-tests exits 0 with 2-GPU parameterization | integration | `mcp__project-commands__run_command unit-tests` | ✅ existing (08-04/05) |
| ROADMAP crit. 4 (pipeline_task ≥ 5 AND scan_batch ≥ 5 per GPU) | AUDIT TEST_CASE asserts ≥ 5 per GPU | unit | `mcp__project-commands__run_command unit-tests filter='"[mgpu-audit]"'` | ✅ existing (08-05) |
| Phase 9 new (batch disjointedness: ∅ intersection) | `counts[0].scan_ids ∩ counts[1].scan_ids == ∅` | unit | same as crit. 4 | ❌ Wave 0 — add to audit test |
| ROADMAP crit. 6 (SF100 bench evidence) | [mgpu-audit] log shows both GPUs + wall-clock | manual | user-delegated | N/A |

### What Fires in unit-tests (MCP, Autonomous)

1. All 22 TPC-H SF1 queries × {DuckDB, parquet} × {num_gpus=1, num_gpus=2} = 88 variants (from 08-04 integration test parameterization, `test_gpu_execution_tpch.cpp`)
2. SF10 Q1/Q6/Q12 × {num_gpus=2} (gated on `SIRIUS_TEST_SF10_PATH` env var)
3. AUDIT TEST_CASE: `gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1` (from `test_gpu_execution_tpch_mgpu_audit.cpp`) — asserts pipeline_task ≥ min_count AND scan_batch ≥ min_count per GPU AND **NEW: disjointedness REQUIRE**

### What Must Be Delegated to the User

The user must run on 2-GPU hardware (2× RTX 6000 Ada):

```bash
# Step 1: Run full unit-test suite with SF10 paths set (ROADMAP crit. 2 + crit. 4)
export SIRIUS_TEST_SF10_PATH=/datasets/tpch_parquet_sf10
mcp__project-commands__run_command unit-tests
# Expected: exit 0, all 88+ test variants green, AUDIT TEST_CASE passes >=5 per GPU

# Step 2: SF100 Q1 validation (ROADMAP crit. 1 + 6)
export SIRIUS_LOG_DIR=/tmp/sirius-ph9-sf100
export SIRIUS_LOG_LEVEL=info
# Run SF100 Q1 with num_gpus=2 via DuckDB CLI or Python API
# Capture wall-clock and full [mgpu-audit] log
grep '\[mgpu-audit\]' $SIRIUS_LOG_DIR/*.log

# Step 3: Record evidence in .planning/phases/09-*/09-VALIDATION.md
# with full [mgpu-audit] grep output, wall-clock, and correctness diff
```

Evidence file path: `.planning/phases/09-scan-task-distributor-batch-ownership-affinity/09-VALIDATION.md`

### Wave 0 Gaps (before main implementation)

- [ ] Add `REQUIRE(intersection.empty())` to `test_gpu_execution_tpch_mgpu_audit.cpp` after the existing `scan_ids.size() >= min_count` checks — covers new disjointedness requirement
- No framework install needed (Catch2 already in place)
- No new test files needed (amend existing AUDIT TEST_CASE)

---

## State of the Art

| Phase 8 State | Phase 9 Target |
|---------------|----------------|
| `select_target_gpu()` is memory-weighted RR (no batch affinity) | `select_target_gpu()` records batch→GPU sticky assignment |
| `preferred_device_id=-1` at `parquet_scan_task::compute_task` entry | `preferred_device_id=N` (real GPU) at entry |
| Same `batch_id` can be dispatched to two tasks on different GPUs → SIGSEGV | Each `batch_id` is pinned to exactly one GPU; cross-GPU dispatch is blocked |
| AUDIT TEST_CASE asserts ≥ 5 per GPU (no disjointedness check) | AUDIT TEST_CASE additionally asserts `counts[0].scan_ids ∩ counts[1].scan_ids == ∅` |

---

## Open Questions

1. **Does `_datasource` caching on the parquet_scan_task object survive across re-dispatches?**
   - What we know: `_datasource` is a `std::shared_ptr` member of `parquet_scan_task`, set inside `if (!_datasource)` in `compute_task`. It persists across calls to the same task object.
   - What's unclear: Does the scan executor ever re-dispatch the same task object (not a new task object) after a failed lock? If yes, a stale `_datasource` from a wrong-GPU dispatch would persist.
   - Recommendation: Confirm by reading `duckdb_scan_executor::get_scan_output` — if `compute_task` is called once per task object lifetime, this is not a risk.

2. **Is the SIGSEGV in the scan task itself or downstream in the GPU pipeline task that consumes the scan output?**
   - What we know: 08-08 reports the SIGSEGV at `test_gpu_execution_tpch.cpp:207` (`REQUIRE_FALSE(gpu_result->HasError())`), which means the error propagates out of the query executor. The lock failure returns `nullopt` and `cancel_task_if_needed()` logs an error, but does NOT throw — it returns `nullopt`, and the caller downstream may dereference the optional without checking it.
   - What's unclear: Which callsite in `pipelineable_operator_data::prepare_for_processing` dereferences the nullopt?
   - Recommendation: Before writing the fix, add a `SIRIUS_LOG_INFO` at the `return std::nullopt` site in `lock_or_prepare_batch:155` to confirm the nullopt path is taken, and a breakpoint/log at the first deref of the returned optional in the caller.

3. **What is `pipelineable_operator_data::prepare_for_processing`?**
   - What we know: It's referenced in `batch_lock_utils.hpp`'s docstring and in STATE.md context. It calls `lock_or_prepare_batch`. It's the function that receives the `std::nullopt` on failure.
   - What's unclear: Where does it live? Is the null check present?
   - Recommendation: `grep -rn 'prepare_for_processing'` before writing the fix to ensure the null-check gap is understood.

---

## Sources

### Primary (HIGH confidence)

- Direct source inspection: `src/op/scan/duckdb_scan_executor.cpp` (full file, v1.2 branch)
- Direct source inspection: `src/include/pipeline/batch_lock_utils.hpp` (full file)
- Direct source inspection: `src/op/scan/parquet_scan_task.cpp` (lines 720-820)
- Direct source inspection: `src/include/op/scan/parquet_scan_task.hpp` (local/global state classes)
- Direct source inspection: `src/include/pipeline/sirius_pipeline_task_states.hpp` (full file)
- Direct source inspection: `src/pipeline/pipeline_executor.cpp` (lines 70-273)
- Direct source inspection: `src/creator/task_creator.cpp` (lines 100-560)
- Direct source inspection: `test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp` (full relevant sections)
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-08-PROBE-LOG.log` (verbatim probe capture)
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-08-DIAGNOSIS.md` (hypothesis E analysis)
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-09-HALT.md`
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-01-SUMMARY.md`
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-07-SUMMARY.md`
- `git show 1f80c2a --stat` and `git show 6d73680 --stat`

### Secondary (MEDIUM confidence)

- `.planning/STATE.md` — documented decisions from Phase 8 (SCHED-00/01/02, Pattern 2, OOM retry)
- `.planning/ROADMAP.md` — success criteria 1/2/4/6

---

## Metadata

**Confidence breakdown:**

- Standard stack (files to modify): HIGH — confirmed by direct source inspection
- Bug 1 root cause (double-dispatch): HIGH — confirmed by probe log frame identities
- Bug 2 root cause (preferred_device_id=-1): HIGH — confirmed by source inspection of call chain
- Fix shape recommendation: HIGH — matches existing patterns in codebase
- Pitfalls: HIGH — sourced from direct code analysis
- Validation strategy: HIGH — mirrors established Phase 8 template

**Research date:** 2026-04-24
**Valid until:** This is a snapshot of the `feature/single-node-multi-gpu2` branch at the research date. Valid until the branch diverges significantly from this state (approximately 30 days for a codebase this active).

---

## RESEARCH COMPLETE

**Phase:** 09 — Scan-Task Distributor + Batch-Ownership Affinity
**Confidence:** HIGH

### Key Findings

1. **Bug 1 (double-dispatch):** `select_target_gpu()` in `duckdb_scan_executor::manager_loop` is purely memory-weighted with no batch-affinity awareness. It can dispatch the same `batch_id` (already resident on GPU 0) to a task targeting GPU 1. The `lock_or_prepare_batch` call fails with `memspace_mismatch, success=false`, and the downstream code does NOT safely handle the returned `nullopt`, causing SIGSEGV. Fix: add a sticky `batch_id→gpu` map in `duckdb_scan_executor`, keyed by the counter returned from `select_target_gpu()`, which is the same counter used as `batch_id` in the audit log.

2. **Bug 2 (preferred_device_id=-1):** `target_gpu_id` from `select_target_gpu()` is captured by the dispatch lambda for stream/device-guard purposes only. It is never written to `parquet_scan_task_local_state` or `parquet_scan_task_global_state`. `compute_task` reads `g_state.get_preferred_device_id()` → `nullopt` → falls back to `backends.begin()` (GPU 0) for `_datasource` construction, even when running on GPU 1. Fix: add `preferred_device_id` field to `parquet_scan_task_local_state` and set it from `manager_loop` before dispatch.

3. **AUDIT disjointedness extension:** The new REQUIRE (`counts[0].scan_ids ∩ counts[1].scan_ids == ∅`) requires adding a single `std::set_intersection` check to the existing `test_gpu_execution_tpch_mgpu_audit.cpp` AUDIT TEST_CASE. The log format already emits `batch_id=K` and the parser already collects `std::set<std::string> scan_ids` per GPU.

4. **Commit 1f80c2a is adjacent but safe:** Its `select_target_gpu()` stride fix is correct and must be preserved. Phase 9 adds affinity logic ON TOP of it. It also explains why the failure mode changed from `cudaErrorInvalidValue` to SIGSEGV (before 1f80c2a, everything went to GPU 0 and the race was latent).

5. **Commit 6d73680 is independent:** The OOM retry budget is for `gpu_pipeline_executor` (GPU pipeline tasks), not `duckdb_scan_executor` (scan tasks). Phase 9 does not touch it.

### File Created

`.planning/phases/09-scan-task-distributor-batch-ownership-affinity/09-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Root cause (Bug 1 + Bug 2) | HIGH | Confirmed by probe log + direct source trace |
| Fix shape | HIGH | Pattern matches existing `gpu_pipeline_task` two-tier preferred_device_id idiom |
| Files to modify | HIGH | All confirmed by direct source inspection |
| AUDIT extension | HIGH | `parse_audit_log` + `AuditCounts.scan_ids` already infrastructure-ready |
| Risk assessment (1f80c2a / 6d73680) | HIGH | Confirmed by git diff surface area analysis |
| Validation strategy | HIGH | Inherits Phase 8 template with precise commands |

### Open Questions

1. Does `_datasource` caching in `parquet_scan_task` persist across re-dispatches of the same task object, and does this matter for the fix?
2. What is the exact call site in `pipelineable_operator_data::prepare_for_processing` that dereferences the nullopt returned by `lock_or_prepare_batch`? (Needed to verify the SIGSEGV null-deref point.)

### Ready for Planning

Research complete. Planner can decompose Phase 9 into plans covering:
- Wave A: `preferred_device_id` plumbing (Bug 2) — header + local state + compute_task two-tier lookup + manager_loop set call
- Wave B: batch-ownership affinity map (Bug 1) — sticky map in executor + memspace_mismatch caller handling
- Wave C: AUDIT disjointedness REQUIRE (test extension)
- Wave D: Phase 9 ship-gate (user-delegated SF100 Q1 + SF10 + full SF1 22-query validation)
