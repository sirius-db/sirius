---
phase: 08-multi-gpu-sql-pipeline-fix
plan: 03
subsystem: observability

tags: [mgpu-audit, logging, spdlog, multi-gpu, pipeline_executor, scan_executor, audit-payload]

# Dependency graph
requires:
  - phase: 08-multi-gpu-sql-pipeline-fix
    plan: 01
    provides: "FIX-01 per-GPU stream pool map in duckdb_scan_executor; establishes the dispatch sites being audited"
  - milestone: v1.1
    provides: "Baseline [mgpu-audit] info-level dispatch logs (commit fd24174) at src/pipeline/pipeline_executor.cpp:249 and src/op/scan/duckdb_scan_executor.cpp:182"
provides:
  - "Extended pipeline_task [mgpu-audit] emission carrying unique task_id suffix (uint64_t from gpu_pipeline_task::get_task_id()) — Plan 08-05 precondition"
  - "Extended scan_batch [mgpu-audit] emission carrying unique batch_id suffix (_scan_round_robin.fetch_add() counter value) — Plan 08-05 precondition"
  - "Grep-stable payload shape: `[mgpu-audit] <event> <verb> to GPU N <key>=<value>` — the prefix and `GPU N` substring are preserved verbatim so v1.1 verification greps continue to match"
affects:
  - 08-05

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern 3 from 08-RESEARCH.md: [mgpu-audit] log extension for assertion-friendly counting — extend existing INFO emissions with `key=value` suffixes using fmt-style `{}` interpolation via SIRIUS_LOG_INFO. Prefix and leading grep substring preserved verbatim; new fields appended whitespace-separated so tests can awk-split on the suffix."
    - "Reuse-existing-counter pattern: rather than introducing a new atomic member for batch_id, surfaced the already-fetched _scan_round_robin counter (captured as local `counter` in select_target_gpu) into the audit payload. Zero new synchronization primitives, zero new state."

key-files:
  created: []
  modified:
    - "src/pipeline/pipeline_executor.cpp (+9/-1 lines): captured task_id from gpu_pipeline_task::get_task_id() inside the existing dynamic_cast branch; extended SIRIUS_LOG_INFO payload with `task_id={}` after the `GPU {}` field"
    - "src/op/scan/duckdb_scan_executor.cpp (+8/-1 lines): surfaced the pre-existing `counter` local (from `_scan_round_robin.fetch_add(1)`) into the SIRIUS_LOG_INFO payload as `batch_id={}`, positioned between `GPU {}` and the `(available: ... bytes)` trailing parenthetical"

key-decisions:
  - "Used accessor-style task_id (gpu_pipeline_task::get_task_id()) over pointer-as-ID fallback. Rationale: the accessor already exists (src/include/pipeline/gpu_pipeline_task.hpp:204), returns uint64_t assigned at construction, and is semantically correct (unique per task). Pointer fallback was Plan 08-03's VARIANT B contingency; VARIANT A was directly applicable."
  - "Reused `_scan_round_robin` counter as batch_id rather than adding a new atomic member. Rationale: the counter is already monotonic per scan_batch assignment (every call to select_target_gpu's weighted branch does a fetch_add(1)), giving a unique value per batch. Adding a second atomic would be redundant and introduce net-new state. Plan contemplated this as the primary path per RESEARCH Pattern 3's `auto const batch_seq = _scan_round_robin.load()` template; adapted trivially to reuse the already-captured local `counter`."
  - "task_id=0 fallback for non-gpu_pipeline_task branch. Rationale: the dynamic_cast may fail if a non-GPU task somehow ends up in the queue (current code path doesn't allow this but the branch exists for defense-in-depth); logging task_id=0 for that case is consistent and greppable without crashing."
  - "Inserted batch_id BEFORE the `(available: ...)` parenthetical rather than after. Rationale: preserves `(available:` as a stable grep anchor for the trailing memory-state info while still keeping `batch_id=` immediately adjacent to `GPU N` so the whole `GPU N batch_id=K` pair is contiguous for awk-split patterns."

requirements-completed: [AUDIT-01, AUDIT-02, AUDIT-03]

# Metrics
duration: 6min
completed: 2026-04-21
---

# Phase 08 Plan 03: AUDIT Log Payload Extension Summary

**Closed the log-payload side of AUDIT-01/02/03 — both `[mgpu-audit]` INFO emissions now carry unique IDs (`task_id` and `batch_id`) so Plan 08-05's Catch2 audit TEST_CASE can count UNIQUE events per GPU via `grep + awk + sort -u`, robust against log-line duplication from retries.**

## Performance

- **Duration:** ~6 min (wall clock)
- **Started:** 2026-04-21 (parallel with Plan 08-04 in Wave 3)
- **Tasks:** 2 (pipeline_task payload extension + scan_batch payload extension)
- **Files modified:** 2

## Before / After (concrete)

### `src/pipeline/pipeline_executor.cpp`

**Before (v1.1 baseline, commit `fd24174`):**

```cpp
SIRIUS_LOG_DEBUG("management_eventloop: routing task to GPU {}", target_device_id);
SIRIUS_LOG_INFO("[mgpu-audit] pipeline_task dispatched to GPU {}", target_device_id);
```

**After (Plan 08-03, commit `6d86271`):**

```cpp
int target_device_id = _gpu_executors.begin()->first;  // default: first GPU
uint64_t task_id     = 0;
if (auto* gpu_task = dynamic_cast<pipeline::gpu_pipeline_task*>(task.get())) {
  auto pref = gpu_task->get_preferred_device_id();
  if (pref.has_value() && _gpu_executors.count(pref.value())) {
    target_device_id = pref.value();
  }
  task_id = gpu_task->get_task_id();
}

SIRIUS_LOG_DEBUG("management_eventloop: routing task to GPU {}", target_device_id);
SIRIUS_LOG_INFO("[mgpu-audit] pipeline_task dispatched to GPU {} task_id={}",
                target_device_id,
                task_id);
```

**Example emitted log line (format, not a run capture):**

```
[... INFO ...] [mgpu-audit] pipeline_task dispatched to GPU 0 task_id=17
```

### `src/op/scan/duckdb_scan_executor.cpp`

**Before (v1.1 baseline, commit `fd24174`):**

```cpp
auto counter      = _scan_round_robin.fetch_add(1);
// ...
SIRIUS_LOG_INFO("[mgpu-audit] scan_batch assigned to GPU {} (available: {} bytes)",
                space->get_device_id(),
                space->get_available_memory());
```

**After (Plan 08-03, commit `238342a`):**

```cpp
auto counter      = _scan_round_robin.fetch_add(1);
// ...
SIRIUS_LOG_INFO("[mgpu-audit] scan_batch assigned to GPU {} batch_id={} (available: {} bytes)",
                space->get_device_id(),
                counter,
                space->get_available_memory());
```

**Example emitted log line (format, not a run capture):**

```
[... INFO ...] [mgpu-audit] scan_batch assigned to GPU 1 batch_id=42 (available: 23068672000 bytes)
```

## ID Source Determination

| Emission        | Plan contemplated | Applied here         | Source                                                                      |
| --------------- | ----------------- | -------------------- | --------------------------------------------------------------------------- |
| `task_id`       | Accessor vs pointer | **Accessor (VARIANT A)** | `gpu_pipeline_task::get_task_id()` — exists at `gpu_pipeline_task.hpp:204`, returns `uint64_t` assigned at construction per `gpu_pipeline_task.hpp:167` |
| `batch_id`      | Existing counter vs new atomic | **Existing counter** | Pre-existing `_scan_round_robin.fetch_add(1)` local `counter` in `select_target_gpu`'s weighted-branch body; already monotonic per scan_batch assignment |

Neither fallback was needed. Zero new atomics, zero new accessors, zero header changes.

## Task Commits

| Task                                                                     | Commit    | Type |
| ------------------------------------------------------------------------ | --------- | ---- |
| Task 1: Append task_id to pipeline_task [mgpu-audit] emission           | `6d86271` | feat |
| Task 2: Append batch_id to scan_batch [mgpu-audit] emission             | `238342a` | feat |

Plan metadata commit: pending after SUMMARY.md + STATE.md + ROADMAP.md updates.

## Static Invariants (all green)

| Check                                                                                         | Result                                                                     |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `grep -E '\[mgpu-audit\] pipeline_task dispatched to GPU \{\} task_id='` in pipeline_executor.cpp | 1 match (format string includes both `GPU {}` and `task_id=`)              |
| `grep -E '\[mgpu-audit\] scan_batch assigned to GPU \{\} batch_id='` in duckdb_scan_executor.cpp | 1 match (format string includes `GPU {}`, `batch_id=`, and preserves trailing `(available: ... bytes)`) |
| `grep '\[mgpu-audit\] pipeline_task dispatched to GPU'` (baseline substring preserved)       | 1 emission + 1 comment reference — the grep still matches the emission    |
| `grep '\[mgpu-audit\] scan_batch assigned to GPU'` (baseline substring preserved)            | 1 emission + 1 comment reference — the grep still matches the emission    |
| `rmm::cuda_stream_default` in modified files                                                 | 0 matches                                                                  |
| `rmm::cuda_stream_default` across `src/` (baseline HYG-02 check)                             | 41 matches across 12 files — **unchanged** from 08-01/08-02 baseline       |
| MCP `build` (after Task 1)                                                                   | exit 0 (9.2s)                                                              |
| MCP `build` (after Task 2)                                                                   | exit 0 (9.4s)                                                              |
| Files modified match `files_modified` contract                                               | Exactly `src/pipeline/pipeline_executor.cpp` + `src/op/scan/duckdb_scan_executor.cpp` |
| Parallel-file discipline (no edits to test/ or fixtures)                                     | Confirmed: no changes under `test/cpp/integration/`, `test/cpp/utils/`, or `test/cpp/unittest.cpp` |

## Decisions Made

- **Accessor over pointer for task_id.** `gpu_pipeline_task::get_task_id()` already exists and returns a `uint64_t` unique per task. No need for `reinterpret_cast<uintptr_t>(task.get())` fallback. Semantically cleaner, grep-stable as a decimal integer.
- **Reuse `_scan_round_robin` counter for batch_id, not a new atomic.** The counter is already monotonic per batch assignment (`fetch_add(1)` inside `select_target_gpu`'s weighted branch). Adding a second atomic would be duplicate state. Plan's fallback (new `_batch_id_counter`) not needed.
- **task_id=0 defense-in-depth for non-gpu_pipeline_task branch.** Current scheduler guarantees only `gpu_pipeline_task` ends up in `_task_queue`, but the `dynamic_cast` could in principle fail; emitting `task_id=0` in that case is consistent with the format and does not crash.
- **batch_id inserted between `GPU N` and `(available: ...)` parenthetical.** Keeps `GPU N batch_id=K` contiguous as an awk-splittable pair AND preserves `(available:` as a separate stable grep anchor. The trailing `bytes)` closure is unchanged.
- **No test edits, no fixture edits.** Per plan scope and parallel-file discipline with Plan 08-04 (which owns the fixture + test-env work). Plan 08-05 is the consumer of these payload extensions.

## Deviations from Plan

None. Plan 08-03 executed exactly as written:

- Task 1 applied VARIANT A (accessor) as contemplated by the plan.
- Task 2 applied the primary path (existing counter) as contemplated by the plan — the FALLBACK (new atomic member) was not needed.
- Both MCP build gates passed on first attempt.
- Parallel-file discipline preserved: only `src/pipeline/pipeline_executor.cpp` and `src/op/scan/duckdb_scan_executor.cpp` were modified, per the plan's `files_modified` frontmatter.
- HYG invariant preserved: 41 `rmm::cuda_stream_default` matches across `src/` unchanged.

## Handoff to Plan 08-05

Plan 08-05's Catch2 audit TEST_CASE can now assert unique-ID counts per GPU. The canonical grep patterns are:

```bash
# Count UNIQUE tasks dispatched to GPU 0
grep '\[mgpu-audit\] pipeline_task dispatched to GPU 0 ' <logfile> \
  | grep -oE 'task_id=[0-9]+' \
  | sort -u \
  | wc -l

# Count UNIQUE batches assigned to GPU 1
grep '\[mgpu-audit\] scan_batch assigned to GPU 1 ' <logfile> \
  | grep -oE 'batch_id=[0-9]+' \
  | sort -u \
  | wc -l
```

Both patterns are stable across runs (format string is deterministic). ROADMAP criterion 4 (`>= 5 unique tasks AND >= 5 unique batches on BOTH GPUs`) is now directly assertable from log output without worrying about log-line duplication.

## Issues Encountered

None. Both tasks applied cleanly, first build gate of each was green, no additional compile cycles wasted.

## Known Stubs

None. Both emissions are fully wired to real ID sources (gpu_pipeline_task accessor + existing atomic counter). No placeholder values, no TODOs, no data-flow gaps.

## Next Phase Readiness

- **AUDIT-01 / AUDIT-02 log-payload side closed.** The production emissions now carry the IDs Plan 08-05 requires.
- **AUDIT-03 log-payload side closed.** The SF100 ship-gate (Plan 08-06) greps over this same log tag and can now differentiate unique events.
- **Plan 08-05 (Catch2 audit TEST_CASE) is unblocked.** It can assume both payloads carry the `<key>=<value>` suffix.
- **Plan 08-06 (SF100 ship gate) benefits.** The criterion 6 evidence capture becomes more precise — the log-grep count is unique-event-count, not log-line-count.
- **Zero new dependencies, zero new state, zero new CUDA API.** Safe to ship alongside Plan 08-04's test-env work in Wave 3 (no file conflicts, disjoint scope).

## Self-Check: PASSED

**Files verified to exist (modifications):**

- FOUND: `src/pipeline/pipeline_executor.cpp` (line 255 emits `task_id={}` suffix; line 246 emits `task_id = gpu_task->get_task_id()` capture)
- FOUND: `src/op/scan/duckdb_scan_executor.cpp` (line 204 emits `batch_id={}` suffix using pre-existing `counter` local)

**Commits verified to exist:**

- FOUND: `6d86271` (feat — Task 1: task_id payload extension)
- FOUND: `238342a` (feat — Task 2: batch_id payload extension)

**Grep invariants verified:**

- `[mgpu-audit] pipeline_task dispatched to GPU {} task_id=` in pipeline_executor.cpp: 1 match — required >= 1
- `[mgpu-audit] scan_batch assigned to GPU {} batch_id=` in duckdb_scan_executor.cpp: 1 match — required >= 1
- `[mgpu-audit]` across modified files: 4 matches (2 emissions + 2 comment references) — required >= 2 emissions
- `rmm::cuda_stream_default` in modified files: 0 matches — HYG-02 preserved
- `rmm::cuda_stream_default` across `src/`: 41 matches across 12 files — baseline unchanged
- MCP build exits 0 — required

---
*Phase: 08-multi-gpu-sql-pipeline-fix*
*Completed: 2026-04-21*
