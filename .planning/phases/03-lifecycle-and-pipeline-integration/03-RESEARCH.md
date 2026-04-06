# Phase 3: Lifecycle and Pipeline Integration - Research

**Researched:** 2026-04-06
**Domain:** C++ concurrency, GPU memory management, executor lifecycle integration
**Confidence:** HIGH

## Summary

Phase 3 wires the redesigned `downgrade_executor` (completed in Phases 1-2) into its callers: `SiriusContext` for lifecycle management, the monitor loop for memory pressure response, and `gpu_pipeline_executor` for reservation-shortfall recovery. The code changes are concentrated in two areas: (1) adding a `downgrade_executor*` parameter to `gpu_pipeline_executor` and implementing a retry-with-downgrade loop in `manager_loop()`, and (2) writing lifecycle tests that exercise start/stop/drain from SiriusContext's perspective.

The existing code is well-structured for this integration. `pipeline_executor.cpp` has a single construction site for `gpu_pipeline_executor` (line 62) where the downgrade executor pointer can be injected. The `manager_loop()` already has the reservation acquisition at line 98 where the retry loop slots in cleanly. The monitor loop and drain() implementations in `downgrade_executor.cpp` are already complete and functioning correctly from Phase 2.

**Primary recommendation:** Implement PIPE-01/02/03 as the primary code change (modify `gpu_pipeline_executor` constructor + retry loop), then write lifecycle tests for LIFE-01 through LIFE-05 to verify the existing implementations work correctly from SiriusContext's perspective.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** When `reservation->size() < bytes_needs`, call `request_free_memory_and_wait(bytes_needs - reservation->size())` on the downgrade executor, then retry `make_reservation`. Loop up to 5 attempts total.
- **D-02:** No delay between retry attempts -- the blocking `request_free_memory_and_wait` call itself is the wait. Retry immediately after it returns.
- **D-03:** After 5 failed retries (still partial reservation), proceed with the partial reservation and execute the task with reduced memory. Log a warning. This matches the current behavior when `reservation->size() != bytes_needs`.
- **D-04:** Add `downgrade_executor*` as a constructor parameter to `gpu_pipeline_executor`. Direct, explicit dependency -- no runtime lookup via SiriusContext. The constructor already takes `memory_space*`, so adding the executor that manages that space is natural. Update all call sites that construct `gpu_pipeline_executor` to pass the corresponding downgrade executor.
- **D-05:** LIFE-01 through LIFE-05 are already implemented in Phases 1-2. Phase 3 writes unit tests that exercise these from SiriusContext's perspective: start/stop/drain correctness (LIFE-01), drain shared_ptr guarantee (LIFE-02), monitor loop integration (LIFE-03), concurrent API safety (LIFE-04), and CUDA stream lifecycle (LIFE-05). No code changes unless tests reveal gaps.
- **D-06:** Current `drain()` implementation is sufficient for LIFE-02. `pool->wait_all()` ensures all dispatch lambdas have returned, releasing all `shared_ptr<data_batch>` captures. Queue drain drops pending requests that haven't been dispatched (no batch refs). No explicit ref-counting verification needed.

### Claude's Discretion
- Exact retry loop structure (for loop vs while with counter)
- Whether to release and re-acquire reservation on each retry, or attempt to grow the existing one
- Internal logging verbosity for retry attempts (TRACE vs DEBUG vs WARN)
- Test fixture design for lifecycle tests (mock memory_space vs real)
- Whether `gpu_pipeline_executor` stores `downgrade_executor*` directly or wraps in a helper

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LIFE-01 | start(), stop(), drain() compatible with SiriusContext usage | Already implemented; drain() at lines 92-105 of downgrade_executor.cpp correctly interrupts, joins processing thread, waits for pool, drains queue, resumes, and restarts. Tests verify. |
| LIFE-02 | drain() guarantees no shared_ptr<data_batch> references remain | pool->wait_all() ensures all dispatch lambdas (which capture shared_ptr<data_batch>) have completed. Queue drain drops unprocessed requests. No additional code needed per D-06. |
| LIFE-03 | Monitor loop polls should_downgrade_memory() and triggers via blocking API | monitor_loop() at lines 159-179 already polls should_downgrade_memory() and enqueues fire-and-forget requests. It uses the request queue (non-blocking from monitor's perspective). Tests verify integration. |
| LIFE-04 | All public APIs thread-safe without external synchronization | request_free_memory/request_downgrade use atomic queue push (interruptible_mpmc is thread-safe). drain() and stop() use atomic compare_exchange on _running. Tests should exercise concurrent calls. |
| LIFE-05 | CUDA stream created on start, destroyed on stop; workers call cudaSetDevice | start() lines 51-58 create stream and set up per_thread_init with cudaSetDevice. stop() lines 86-89 destroy stream. Already implemented. |
| PIPE-01 | gpu_pipeline_executor calls request_free_memory_and_wait when reservation < bytes_needs | Insertion point: gpu_pipeline_executor.cpp line 98 (after make_reservation). Add retry loop calling _downgrade_executor->request_free_memory_and_wait(shortfall). |
| PIPE-02 | Retry loop runs up to 5 attempts; after 5, proceed with partial reservation | For loop with counter, break when reservation->size() >= bytes_needs or counter hits 5. Fall through to existing warn-and-continue code at lines 108-116. |
| PIPE-03 | gpu_pipeline_executor has access to downgrade_executor | Add downgrade_executor* parameter to constructor. Single construction site at pipeline_executor.cpp:62. Pipeline_executor needs downgrade executor access (passed from SiriusContext). |
</phase_requirements>

## Architecture Patterns

### Integration Point Map

The changes flow through this call chain:

```
SiriusContext::initialize()
  -> creates downgrade_executors_ (already done, lines 179-193)
  -> creates pipeline_executor_ (line 171)
     -> pipeline_executor constructor creates gpu_pipeline_executor per GPU space (line 62)
        ** CHANGE: pass downgrade_executor* here **
  -> pipeline_executor::start() -> gpu_pipeline_executor::start() (already done)

SiriusContext::QueryEnd()
  -> downgrade_executor::drain() for each executor (already done, lines 113-115)

gpu_pipeline_executor::manager_loop()
  -> make_reservation(bytes_needs) (line 98)
  ** CHANGE: if reservation->size() < bytes_needs, retry loop with downgrade **
```

### Recommended Change Structure

```
Modified files:
src/include/pipeline/gpu_pipeline_executor.hpp  -- add downgrade_executor* member + constructor param
src/pipeline/gpu_pipeline_executor.cpp          -- retry loop in manager_loop()
src/include/pipeline/pipeline_executor.hpp      -- store downgrade executor map or accept from outside
src/pipeline/pipeline_executor.cpp              -- pass downgrade_executor* at construction site
src/sirius_context.cpp                          -- pass downgrade executors to pipeline_executor

New files:
test/cpp/downgrade/test_downgrade_lifecycle.cpp -- LIFE-01 through LIFE-05 tests
test/cpp/pipeline/test_gpu_pipeline_retry.cpp   -- PIPE-01 through PIPE-03 tests (optional, may extend existing)
```

### Pattern: Retry Loop in manager_loop()

**Recommendation:** Use a for loop with counter. Release and re-acquire reservation on each retry (release partial, request downgrade for full shortfall, then make fresh reservation). This is simpler and avoids reasoning about "growing" a reservation.

```cpp
// After line 98 in gpu_pipeline_executor.cpp:
auto reservation = _memory_space->make_reservation(bytes_needs);
if (!reservation) {
  // ... existing null-reservation error handling ...
} else if (reservation->size() < bytes_needs && _downgrade_executor) {
  for (int retry = 0; retry < 5 && reservation->size() < bytes_needs; ++retry) {
    size_t shortfall = bytes_needs - reservation->size();
    SIRIUS_LOG_DEBUG(
      "GPU Pipeline Executor: reservation shortfall {} bytes for task {}, "
      "requesting downgrade (attempt {}/5)",
      shortfall, gpu_task->get_task_id(), retry + 1);
    // Release partial reservation before requesting downgrade
    reservation.reset();
    _downgrade_executor->request_free_memory_and_wait(shortfall);
    reservation = _memory_space->make_reservation(bytes_needs);
    if (!reservation) {
      // Reservation system failure -- break and report error
      break;
    }
  }
  if (!reservation) {
    // ... error handling ...
  } else if (reservation->size() != bytes_needs) {
    SIRIUS_LOG_WARN(
      "GPU Pipeline Executor: after 5 downgrade attempts, reservation still partial "
      "({}/{} bytes) for task {} -- proceeding with partial reservation",
      reservation->size(), bytes_needs, gpu_task->get_task_id());
  }
}
```

**Why release-and-reacquire:** The `make_reservation` API returns a new reservation object. There is no "grow" API. Holding a partial reservation while requesting downgrade wastes space -- the partial reservation pins memory that the downgrade could potentially free. Release it, free memory, then try again for the full amount.

### Pattern: Constructor Injection for downgrade_executor

```cpp
// gpu_pipeline_executor.hpp -- add member:
sirius::parallel::downgrade_executor* _downgrade_executor{nullptr};

// Constructor signature change:
explicit gpu_pipeline_executor(
  exec::thread_pool_config config,
  cucascade::memory::memory_space* mem_space,
  exec::publisher<std::unique_ptr<task_request>> task_request_publisher,
  sirius::parallel::downgrade_executor* downgrade_executor = nullptr);
```

Using `nullptr` default preserves backward compatibility for tests that don't need downgrade functionality.

### Pattern: Plumbing Through pipeline_executor

`pipeline_executor` constructs `gpu_pipeline_executor` instances in its constructor (line 45-65 of pipeline_executor.cpp). It iterates GPU memory spaces. The challenge: at construction time, `pipeline_executor` doesn't have access to downgrade executors. Two options:

**Option A (recommended):** Pass the downgrade executors to `pipeline_executor` constructor. This requires SiriusContext to create downgrade executors BEFORE pipeline_executor, which is already the case (lines 179-193 vs line 171 in sirius_context.cpp). Wait -- actually, looking at sirius_context.cpp more carefully:

```
line 171: pipeline_executor_ = ... // created FIRST
line 179-193: create downgrade executors  // created SECOND
```

The downgrade executors are created AFTER pipeline_executor. This means either:
1. Reorder: create downgrade executors first, then pipeline_executor (preferred)
2. Use a setter: `pipeline_executor::set_downgrade_executors(...)` called after both are created
3. Use a late-binding approach: `gpu_pipeline_executor` gets its downgrade_executor* set after construction

**Recommendation:** Reorder initialization in SiriusContext::initialize() so downgrade executors are created before pipeline_executor. This keeps constructor injection clean. The downgrade executors don't depend on pipeline_executor, so the reorder is safe. Their start() should still be called at the end (after pipeline_executor construction).

Revised initialization order:
```
1. memory_manager_ (already first)
2. data_repository_manager_ (already second)
3. downgrade_executors_ (move up from after pipeline_executor)
4. pipeline_executor_ (pass downgrade executors map/vector)
5. task_creator_
6. start everything
```

### Anti-Patterns to Avoid
- **Holding partial reservation during downgrade:** Wastes memory the downgrade could free. Release first.
- **Busy-loop retry without downgrade:** Would spin CPU without making progress. The blocking `request_free_memory_and_wait` is the correct wait mechanism.
- **Looking up downgrade_executor at runtime via SiriusContext:** Creates tight coupling between executor layers and the DuckDB context. Constructor injection per D-04 is cleaner.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe memory reclamation | Custom locking around batch iteration | `request_free_memory_and_wait()` | Already handles candidate selection, concurrent dispatch, predicate-driven stopping |
| Thread pool lifecycle | Manual thread join/interrupt | `bounded_thread_pool::interrupt()/resume()/wait_all()` | RAII slot management prevents resource leaks |
| Future-based async API | Manual condition variables | `std::promise/std::future` in `downgrade_request` | Standard, well-tested pattern already in place |

## Common Pitfalls

### Pitfall 1: Initialization Order in SiriusContext
**What goes wrong:** If downgrade executors are constructed after pipeline_executor (current order), the pipeline_executor can't receive downgrade_executor pointers at construction time.
**Why it happens:** Original code didn't need this dependency.
**How to avoid:** Reorder: create downgrade executors before pipeline_executor. They don't depend on it. Only call `executor->start()` after everything is constructed (to avoid races).
**Warning signs:** Null downgrade_executor pointer in gpu_pipeline_executor during first query.

### Pitfall 2: Reservation Released But Downgrade Frees Nothing
**What goes wrong:** The retry loop releases the partial reservation and requests downgrade, but if no candidates exist (all batches are active/in-transit), request_free_memory_and_wait returns 0. The next make_reservation may get the same (or smaller) partial amount.
**How to avoid:** After downgrade returns 0 freed bytes, break out of the retry loop early -- no point retrying if nothing was freed.
**Warning signs:** Log shows 5 consecutive retries all freeing 0 bytes.

### Pitfall 3: Deadlock Between manager_loop and drain
**What goes wrong:** `gpu_pipeline_executor::manager_loop()` calls `request_free_memory_and_wait()` on the downgrade_executor. If `SiriusContext::QueryEnd()` calls `downgrade_executor::drain()` concurrently, drain interrupts the request queue and joins the processing thread. The request_free_memory_and_wait call blocks forever because the processing thread has exited.
**Why it happens:** QueryEnd is called from the DuckDB thread, while manager_loop runs on the gpu_pipeline_executor's manager thread.
**How to avoid:** This is actually safe because: (1) QueryEnd happens after the query completes, meaning gpu_pipeline_executor's manager_loop is idle (no active tasks), and (2) drain() drains pending requests which fulfills promises with 0. But edge cases during error paths need care -- `drain_after_error()` already stops the gpu_pipeline_executor's manager loop before draining downgrade.
**Warning signs:** Timeout or hang during QueryEnd after a failed query.

### Pitfall 4: Test Isolation With Real GPU Memory
**What goes wrong:** Lifecycle tests that create real memory_space objects and GPU batches can fail if GPU is out of memory from previous test leaks.
**How to avoid:** Use the existing `make_test_memory_manager()` pattern from test_downgrade_executor.cpp. Each test creates its own isolated memory manager. For lifecycle tests that don't need real GPU, pass `nullptr` for memory_space (disables monitor loop).
**Warning signs:** Sporadic CUDA OOM errors in test suite.

### Pitfall 5: Stale Pointer After SiriusContext::terminate()
**What goes wrong:** `terminate()` calls `pipeline_executor_->stop()` then clears `downgrade_executors_`. If gpu_pipeline_executor's manager_loop is still running (race in stop), it may dereference a dangling downgrade_executor pointer.
**How to avoid:** The current terminate() order is correct: pipeline_executor_->stop() is called first (line 210), which stops all gpu_pipeline_executors before downgrade_executors_ are cleared (line 216-219). Verify this order is preserved after changes.
**Warning signs:** Use-after-free crash during shutdown.

## Code Examples

### Retry Loop Implementation
```cpp
// In gpu_pipeline_executor::manager_loop(), after line 98:
auto reservation = _memory_space->make_reservation(bytes_needs);
if (!reservation) {
  SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to acquire memory reservation for task {}",
                   gpu_task->get_task_id());
  if (_completion_handler) {
    _completion_handler->report_error(
      "GPU Pipeline Executor: Failed to acquire memory reservation for task " +
      std::to_string(gpu_task->get_task_id()));
  }
  break;
}

// Retry loop: attempt to reclaim memory via downgrade when reservation is partial
if (reservation->size() < bytes_needs && _downgrade_executor) {
  static constexpr int kMaxDowngradeRetries = 5;
  for (int attempt = 1; attempt <= kMaxDowngradeRetries; ++attempt) {
    size_t shortfall = bytes_needs - reservation->size();
    SIRIUS_LOG_DEBUG(
      "GPU Pipeline Executor: reservation shortfall {} bytes for pipeline {} task {}, "
      "requesting downgrade (attempt {}/{})",
      shortfall, gpu_task->get_pipeline_id(), gpu_task->get_task_id(),
      attempt, kMaxDowngradeRetries);

    reservation.reset();  // release partial reservation
    size_t freed = _downgrade_executor->request_free_memory_and_wait(shortfall);
    if (freed == 0) {
      SIRIUS_LOG_DEBUG("GPU Pipeline Executor: downgrade freed 0 bytes, stopping retry");
      reservation = _memory_space->make_reservation(bytes_needs);
      break;
    }

    reservation = _memory_space->make_reservation(bytes_needs);
    if (!reservation) {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: reservation failed after downgrade for task {}",
                       gpu_task->get_task_id());
      break;
    }
    if (reservation->size() >= bytes_needs) break;
  }
}

// Existing partial-reservation warning (lines 108-116) follows here unchanged
```

### Constructor Change
```cpp
// gpu_pipeline_executor.hpp
#include "downgrade/downgrade_executor.hpp"  // forward declare or include

class gpu_pipeline_executor : public sirius::parallel::itask_executor {
 public:
  explicit gpu_pipeline_executor(
    exec::thread_pool_config config,
    cucascade::memory::memory_space* mem_space,
    exec::publisher<std::unique_ptr<task_request>> task_request_publisher,
    sirius::parallel::downgrade_executor* downgrade_executor = nullptr);

 private:
  sirius::parallel::downgrade_executor* _downgrade_executor{nullptr};
  // ... existing members ...
};
```

### SiriusContext Initialization Reorder
```cpp
// In SiriusContext::initialize(), after data_repository_manager_ creation:

// Create downgrade executors BEFORE pipeline_executor so we can inject them
auto create_executors_for_tier = [&](cucascade::memory::Tier tier) {
  auto spaces = memory_manager_->get_memory_spaces_for_tier(tier);
  auto const& dg_cfg = config_.get_downgrade_executor_config();
  for (auto* space : spaces) {
    auto executor = std::make_unique<sirius::parallel::downgrade_executor>(
      dg_cfg, *data_repository_manager_, space->get_id(),
      const_cast<cucascade::memory::memory_space*>(space), *memory_manager_);
    downgrade_executors_.push_back(std::move(executor));
  }
};
create_executors_for_tier(cucascade::memory::Tier::GPU);

// Now create pipeline_executor, passing downgrade executors
pipeline_executor_ = std::make_unique<sirius::pipeline::pipeline_executor>(
  config_.get_gpu_pipeline_executor_config(),
  config_.get_duckdb_scan_executor_config(),
  *memory_manager_,
  &config_.get_hw_topology(),
  downgrade_executors_);  // NEW parameter

// Start downgrade executors after all construction
for (auto& executor : downgrade_executors_) {
  executor->start();
}
```

### pipeline_executor Constructor Change
```cpp
// pipeline_executor needs to find the right downgrade_executor for each GPU space.
// Since downgrade executors are indexed by space_id, the mapping is:
// space->get_id() -> downgrade_executor with matching get_space_id()

explicit pipeline_executor(
  const exec::thread_pool_config& gpu_executor_config,
  const exec::thread_pool_config& scan_executor_config,
  sirius::memory::sirius_memory_reservation_manager& mem_mgr,
  const cucascade::memory::system_topology_info* sys_topology,
  const std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>>& downgrade_executors);

// In constructor body, when creating gpu_pipeline_executor:
for (auto* space : gpu_spaces) {
  // Find matching downgrade executor
  sirius::parallel::downgrade_executor* dg_exec = nullptr;
  for (auto& de : downgrade_executors) {
    if (de->get_space_id() == space->get_id()) {
      dg_exec = de.get();
      break;
    }
  }
  _gpu_executors.emplace(
    device_id,
    std::make_unique<gpu_pipeline_executor>(config,
                                            const_cast<cucascade::memory::memory_space*>(space),
                                            _task_request_channel.make_publisher(),
                                            dg_exec));
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No retry on partial reservation | Warn and proceed with partial | Pre-Phase 3 | Tasks may OOM or produce suboptimal execution |
| No downgrade integration in pipeline | Retry with downgrade before proceeding | Phase 3 | Better memory utilization, fewer OOM reschedules |
| Collect-all/dispatch-all downgrade | Predicate-driven incremental dispatch | Phase 2 | More efficient, stops early when target is met |

## Open Questions

1. **pipeline_executor constructor backward compatibility**
   - What we know: Adding a new parameter to pipeline_executor constructor. Only one call site (SiriusContext::initialize).
   - What's unclear: Whether any test files construct pipeline_executor directly.
   - Recommendation: Check test files; if they do, add default parameter or update them.

2. **Early break on zero bytes freed**
   - What we know: D-02 says no delay, just retry immediately. But if 0 bytes freed, retrying is pointless.
   - What's unclear: Whether this optimization violates the "5 retries" decision.
   - Recommendation: Breaking early on 0 freed is an optimization within the 5-retry framework. It respects the spirit of D-01/D-02/D-03. Include it.

3. **reservation.reset() before downgrade**
   - What we know: Releasing partial reservation frees memory for downgrade to reclaim.
   - What's unclear: Whether make_reservation can return nullptr after reset (unlikely but possible if another thread grabs all memory).
   - Recommendation: Handle nullptr after re-acquisition as an error, same as existing null-reservation handling.

## Sources

### Primary (HIGH confidence)
- `src/downgrade/downgrade_executor.cpp` -- Full implementation of request API, processing loop, monitor loop, drain
- `src/pipeline/gpu_pipeline_executor.cpp` -- Current manager_loop with reservation acquisition (lines 87-117)
- `src/pipeline/pipeline_executor.cpp` -- Single construction site for gpu_pipeline_executor (line 62)
- `src/sirius_context.cpp` -- Initialization order (lines 141-206), terminate order (lines 209-244), QueryEnd drain (lines 113-115)
- `src/include/exec/bounded_thread_pool.hpp` -- interrupt/resume/wait_all semantics confirmed
- `test/cpp/downgrade/test_downgrade_executor.cpp` -- Existing test patterns and fixtures

### Secondary (MEDIUM confidence)
- `.planning/phases/03-lifecycle-and-pipeline-integration/03-CONTEXT.md` -- All locked decisions D-01 through D-06
- `.planning/REQUIREMENTS.md` -- LIFE-01 through LIFE-05, PIPE-01 through PIPE-03

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all code is internal C++, no external libraries needed beyond what's already in use
- Architecture: HIGH -- single construction site, clear insertion points, well-understood call chain
- Pitfalls: HIGH -- identified from direct code reading, initialization order verified

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (internal codebase, stable)
