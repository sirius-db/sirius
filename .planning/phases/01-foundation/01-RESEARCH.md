# Phase 1: Foundation - Research

**Researched:** 2026-04-06
**Domain:** C++ concurrency refactoring -- decoupling downgrade_executor from itask_executor
**Confidence:** HIGH

## Summary

Phase 1 is a structural refactoring of `downgrade_executor` to remove its inheritance from `itask_executor` and compose its own `bounded_thread_pool`, request queue, and processing thread directly. The existing candidate selection logic (`run_downgrade_pass`, `collect_candidates_from_partition`, two-pass prioritization) is preserved verbatim. A new `downgrade_request` struct is introduced as the unit of work, and requests are processed sequentially by a dedicated processing thread.

All infrastructure primitives already exist in the codebase: `exec::bounded_thread_pool` (reserve/dispatch/wait_all/interrupt/resume), `exec::interruptible_mpmc<T>` (smart-pointer MPMC queue with interrupt support), and `exec::thread_pool_config`. The refactoring is primarily about rearranging ownership -- moving members from the base class into the concrete class and replacing the `itask`/`itask_global_state`/`itask_local_state` hierarchy with a plain struct for `downgrade_task`.

**Primary recommendation:** Execute this as two plans -- (1) restructure downgrade_executor class definition with new members, request struct, processing thread, and lifecycle methods; (2) simplify downgrade_task to a plain struct and wire candidate selection into the new request processing path. Existing tests provide strong regression coverage.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `downgrade_request` has the full skeleton from day one: `std::function<bool()> predicate`, `std::promise<size_t> result`, and `size_t target_bytes`. Only `target_bytes` is exercised in Phase 1; predicate and promise are present but unused until Phase 2 wires them up.
- **D-02:** Request queue uses `exec::interruptible_mpmc<downgrade_request>` -- reuses the proven primitive, supports interrupt/resume for drain semantics.
- **D-03:** Processing thread uses collect-then-dispatch: runs candidate selection first (single-threaded), collects all candidates up to `target_bytes`, dispatches all batch downgrades to the pool at once, then calls `pool->wait_all()`. Phase 2 can evolve to incremental dispatch when it adds predicate-after-each-batch.
- **D-04:** Remove `task_completion_message_queue` from the downgrade path entirely. The processing thread uses `pool->wait_all()` to track completion -- no need to notify `task_creator`. This also removes the `_message_queue` member from `downgrade_executor` and the `_message_queue` reference from `downgrade_task_global_state` (which itself is being removed per D-06).
- **D-05:** Direct composition, no base class. `downgrade_executor` owns a `bounded_thread_pool`, `interruptible_mpmc<downgrade_request>`, processing thread, monitor thread, and `atomic<bool> _running` as direct members. Implements its own `start()/stop()/drain()`. No inheritance from `itask_executor`, no virtual dispatch.
- **D-06:** `downgrade_task` becomes a plain struct with direct members (`shared_ptr<data_batch> batch`, `sirius_memory_reservation_manager& res_mgr`) and an `execute(rmm::cuda_stream_view)` method. The `itask` base class, `downgrade_task_global_state`, and `downgrade_task_local_state` are all removed from the downgrade path. No polymorphism, no `cast<>()` ceremony.
- **D-07:** The candidate selection and ordering logic from `run_downgrade_pass` is ported verbatim, not redesigned.

### Claude's Discretion
- Exact start()/stop()/drain() implementation details (interrupt sequencing, thread join order)
- Whether static helpers remain static or become private methods
- Internal error handling within the processing thread loop

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXEC-01 | The downgrade_executor owns its own `bounded_thread_pool` and does not inherit from `itask_executor` | D-05 defines direct composition. `bounded_thread_pool` API documented below (reserve/dispatch/wait_all/interrupt/resume/stop). Constructor accepts `thread_pool_config` directly. |
| EXEC-02 | Requests are queued and executed one at a time (sequential request processing) | D-02 defines `interruptible_mpmc<downgrade_request>` queue. D-03 defines collect-then-dispatch pattern with `pool->wait_all()` blocking the processing thread until one request completes before dequeuing the next. |
| CAND-01 | Existing candidate selection and prioritization logic is preserved | D-07 requires verbatim port. Current logic in `downgrade_executor.cpp` lines 196-276 documented in Architecture Patterns below. |
| CAND-02 | `collect_candidates_from_partition` and `run_downgrade_pass` selection logic is carried forward | Same functions move into new class as private methods or static helpers (Claude's discretion). Signatures and logic unchanged. |
</phase_requirements>

## Architecture Patterns

### Current Class Hierarchy (Being Removed)

```
itask_executor (base)
  |-- bounded_thread_pool (owned)
  |-- interruptible_mpmc<unique_ptr<itask>> _task_queue
  |-- thread _manager_thread
  |-- atomic<bool> _running
  |-- start()/stop()/drain_and_wait()
  |
  +-- downgrade_executor (derived)
        |-- cudaStream_t _stream
        |-- thread _monitor_thread
        |-- data_repo_mgr, space_id, memory_space, reservation_manager
        |-- task_completion_message_queue _message_queue
        |-- manager_loop() override
        |-- on_start/on_stop/on_stopped overrides
```

### New Class Structure (Phase 1 Target)

```
downgrade_executor (standalone, no base)
  |-- bounded_thread_pool _pool (owned directly)
  |-- interruptible_mpmc<unique_ptr<downgrade_request>> _request_queue
  |-- thread _processing_thread    (replaces _manager_thread)
  |-- thread _monitor_thread       (unchanged)
  |-- atomic<bool> _running        (owned directly)
  |-- cudaStream_t _stream         (unchanged)
  |
  |-- data_repo_mgr, space_id, memory_space, reservation_manager (unchanged)
  |
  |-- start() / stop() / drain()   (own implementation, not virtual)
  |-- get_space_id()               (unchanged)
  |
  |-- processing_loop()            (dequeues requests, runs candidate selection, dispatches to pool)
  |-- monitor_loop()               (unchanged -- polls memory_space, enqueues requests)
  |-- run_downgrade_pass()         (moved from public to internal -- called by processing_loop)
  |-- run_downgrade_pass_all_repos() (called by monitor_loop to enqueue a request)
  |
  |-- static helpers: get_repo_data_size_on_tier, is_partition_active, collect_candidates_from_partition
```

### downgrade_request Struct (D-01)

```cpp
// Source: D-01 from CONTEXT.md
struct downgrade_request {
  size_t target_bytes;                // Phase 1: exercised
  std::function<bool()> predicate;    // Phase 1: present but unused
  std::promise<size_t> result;        // Phase 1: present but unused
};
```

**Critical detail:** `interruptible_mpmc` requires the `smart_pointer` concept (`std::unique_ptr` or `std::shared_ptr`). The queue must be `interruptible_mpmc<std::unique_ptr<downgrade_request>>`. This is consistent with how `itask_executor` uses `interruptible_mpmc<std::unique_ptr<itask>>`.

### downgrade_task Plain Struct (D-06)

```cpp
// Source: D-06 from CONTEXT.md
struct downgrade_task {
  std::shared_ptr<cucascade::data_batch> batch;
  sirius::memory::sirius_memory_reservation_manager& res_mgr;

  void execute(rmm::cuda_stream_view stream);
};
```

The `execute()` body is the same as current `downgrade_task::execute()` but accesses `batch` and `res_mgr` as direct members instead of through `_local_state->cast<>()` and `_global_state->cast<>()`. The `mark_task_completion()` call is removed entirely (D-04).

### Processing Thread Loop Pattern (D-03)

```cpp
void downgrade_executor::processing_loop() {
  while (_running.load()) {
    auto request = _request_queue.pop();  // blocks until request or interrupt
    if (!request) break;                  // interrupted

    // 1. Collect candidates (single-threaded)
    auto candidates = collect_all_candidates(request->target_bytes);

    // 2. Dispatch all to pool at once
    for (auto& batch : candidates) {
      auto slot = _pool->reserve();
      if (!slot) break;  // interrupted
      _pool->dispatch(std::move(slot), [batch = std::move(batch),
                                         &res_mgr = _reservation_manager,
                                         stream = rmm::cuda_stream_view{_stream}]() {
        downgrade_task task{batch, res_mgr};
        try {
          task.execute(stream);
        } catch (const std::exception& e) {
          SIRIUS_LOG_ERROR("[downgrade] batch downgrade failed: {}", e.what());
        }
      });
    }

    // 3. Wait for all dispatched batch downgrades to complete
    _pool->wait_all();
  }
}
```

### Monitor Thread Pattern (Unchanged)

The monitor loop continues polling `should_downgrade_memory()` and now enqueues a `downgrade_request` instead of directly calling `run_downgrade_pass_all_repos`:

```cpp
void downgrade_executor::monitor_loop() {
  while (_running.load()) {
    if (_memory_space && _memory_space->should_downgrade_memory()) {
      size_t amount = _memory_space->get_amount_to_downgrade();
      if (amount > 0) {
        auto req = std::make_unique<downgrade_request>();
        req->target_bytes = amount;
        _request_queue.push(std::move(req));
      }
    }
    std::this_thread::sleep_for(10ms);
  }
}
```

### Lifecycle Methods (Claude's Discretion Area)

**Recommended start() sequence:**
1. CAS `_running` false->true (idempotency guard)
2. Create CUDA stream if `_memory_space` non-null
3. Reactivate `_request_queue`
4. Create `_pool` with `thread_pool_config` and `cudaSetDevice` per-thread init
5. Launch `_processing_thread`
6. Launch `_monitor_thread`

**Recommended stop() sequence:**
1. CAS `_running` true->false (idempotency guard)
2. `_pool->interrupt()` -- unblocks any reserve() in processing_loop
3. `_request_queue.interrupt()` -- unblocks pop() in processing_loop
4. Join `_monitor_thread` (it checks `_running` and exits)
5. Join `_processing_thread` (it sees interrupted queue and exits)
6. `_pool->wait_all()` -- ensure in-flight batch downgrades finish
7. `_pool->stop()` -- join worker threads
8. `_pool.reset()`
9. Destroy CUDA stream

**Recommended drain() sequence:**
1. `_pool->interrupt()` -- unblock processing_loop's reserve()
2. `_request_queue.interrupt()` -- unblock processing_loop's pop()
3. Join `_processing_thread`
4. `_pool->wait_all()` -- wait for in-flight tasks
5. `_request_queue.drain()` -- discard pending requests
6. `_pool->resume()`
7. `_request_queue.reactivate()`
8. Re-launch `_processing_thread`

This matches the pattern in `itask_executor::drain_and_wait()` (lines 58-79 of task_executor.cpp).

### SiriusContext Integration Points

The caller (`sirius_context.cpp`) uses these methods on `downgrade_executor`:
- **Constructor:** `downgrade_executor(config, data_repo_mgr, space_id, memory_space, reservation_manager)` -- signature MUST remain compatible
- **`start()`** -- called after construction in `Initialize()`
- **`stop()`** -- called during `Shutdown()`
- **`drain()`** -- called at QueryEnd before clearing repositories
- **`get_space_id()`** -- used to look up executor by memory space

No other public methods are called from outside. `run_downgrade_pass()` and `run_downgrade_pass_all_repos()` become internal to the class (called by processing_loop and monitor_loop respectively).

### Anti-Patterns to Avoid
- **Keeping virtual dispatch:** Do not add a new base class or abstract interface. The redesign explicitly removes polymorphism from the downgrade path.
- **Shared CUDA stream across threads without synchronization:** The current design uses a single `_stream` shared by all pool workers. This works because `cudaStreamNonBlocking` allows concurrent kernel submission; however, each `batch->convert_to()` internally synchronizes. If this changes in Phase 2, per-worker streams may be needed (out of scope for now).
- **Calling run_downgrade_pass directly from monitor_loop:** The new design routes through the request queue so that sequential processing is enforced.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread pool with bounded concurrency | Custom thread management | `exec::bounded_thread_pool` | Already has reserve/dispatch/wait_all/interrupt/resume; tested in `test/cpp/exec/test_bounded_thread_pool.cpp` |
| Interruptible MPMC queue | mutex + condition_variable queue | `exec::interruptible_mpmc<T>` | Built on `duckdb_moodycamel::BlockingConcurrentQueue`; supports interrupt/drain/reactivate; tested in `test/cpp/exec/test_interruptible_mpmc.cpp` |
| Per-thread CUDA device init | Manual thread-local state | `bounded_thread_pool` constructor's `per_thread_init` callback | Already uses `std::latch` to block until all workers have initialized |

## Common Pitfalls

### Pitfall 1: interruptible_mpmc requires smart_pointer
**What goes wrong:** Attempting to use `interruptible_mpmc<downgrade_request>` directly fails to compile because the `smart_pointer` concept requires `std::shared_ptr` or `std::unique_ptr`.
**Why it happens:** The concept constraint on `interruptible_mpmc` template parameter.
**How to avoid:** Use `interruptible_mpmc<std::unique_ptr<downgrade_request>>`.
**Warning signs:** Compile error referencing `smart_pointer` concept.

### Pitfall 2: std::promise is move-only
**What goes wrong:** `downgrade_request` contains `std::promise<size_t>`. If you try to copy or default-construct a `downgrade_request` vector, compilation fails.
**Why it happens:** `std::promise` is move-only (deleted copy constructor).
**How to avoid:** `downgrade_request` should be move-only. Since it's always wrapped in `unique_ptr` for the queue, this is natural. Just ensure no accidental copies.
**Warning signs:** Compile errors about deleted copy constructor.

### Pitfall 3: drain() must re-launch processing thread
**What goes wrong:** After drain(), the executor appears stopped -- no new requests are processed.
**Why it happens:** drain() joins the processing thread but forgets to restart it.
**How to avoid:** Follow the drain() pattern from `itask_executor::drain_and_wait()` which re-launches the manager thread after draining.
**Warning signs:** Tests pass for first query but fail on second query after drain().

### Pitfall 4: Destroying bounded_thread_pool while workers hold CUDA resources
**What goes wrong:** Segfault or CUDA error during shutdown.
**Why it happens:** `pool->stop()` joins worker threads, but if the CUDA stream is destroyed before workers finish, in-flight `convert_to()` operations crash.
**How to avoid:** In stop(): first `pool->wait_all()`, then `pool->stop()`, then destroy CUDA stream. This is the same order as `itask_executor::stop()` followed by `on_stopped()`.
**Warning signs:** Intermittent crashes during test teardown.

### Pitfall 5: Monitor loop enqueuing after request_queue interrupted
**What goes wrong:** `_request_queue.push()` returns false silently after interrupt; monitor keeps spinning.
**Why it happens:** Monitor checks `_running` but `_request_queue` may be interrupted before `_running` is set to false.
**How to avoid:** Check `_running` as the primary loop condition. The push failure is harmless (returns false, request is dropped). Monitor will exit on next `_running` check.
**Warning signs:** Slow shutdown due to monitor spinning.

### Pitfall 6: Test regression from removing task_completion_message_queue
**What goes wrong:** Existing test `"Single downgrade task executes correctly"` creates a `task_completion_message_queue` and a `downgrade_task_global_state` with it. After D-04/D-06, these types change.
**Why it happens:** Tests directly instantiate the old types.
**How to avoid:** Update test to use the new `downgrade_task` plain struct. The test at `test/cpp/downgrade/test_downgrade_executor.cpp:150-170` needs updating.
**Warning signs:** Test compilation failures.

### Pitfall 7: run_downgrade_pass visibility change
**What goes wrong:** Tests call `executor.run_downgrade_pass(repos, amount)` directly. If this becomes private, tests break.
**Why it happens:** Phase 1 internalizes `run_downgrade_pass` since it's now called from processing_loop.
**How to avoid:** Either (a) keep it public for testability during Phase 1, or (b) test through the request queue by enqueuing requests and verifying batch tier changes. Option (a) is simpler and matches existing test patterns; the signature can be made private in Phase 2 if desired.
**Warning signs:** Test compilation errors about private member access.

## Code Examples

### Current itask_executor::start() for Reference

```cpp
// Source: src/parallel/task_executor.cpp:24-35
void itask_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _bounded_pool = std::make_unique<exec::bounded_thread_pool>(_config.num_threads,
                                                              _config.thread_name_prefix,
                                                              _config.cpu_affinity_list,
                                                              get_per_thread_init());
  _task_queue.reactivate();
  _manager_thread = std::thread([this] { manager_loop(); });
  on_start();
}
```

### Current itask_executor::stop() for Reference

```cpp
// Source: src/parallel/task_executor.cpp:37-49
void itask_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _bounded_pool->interrupt();
  _task_queue.interrupt();
  on_stop();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _bounded_pool->wait_all();
  _bounded_pool->stop();
  _bounded_pool.reset();
  on_stopped();
}
```

### Current downgrade_task::execute() Core Logic (Preserved in New Struct)

```cpp
// Source: src/downgrade/downgrade_task.cpp:32-93
// Key operations to preserve in the new plain struct:
// 1. Check if already on HOST tier -- early return
// 2. Save prev_state, try_to_lock_for_in_transit()
// 3. Request HOST reservation from res_mgr
// 4. convert_to<host_data_representation>() via converter_registry
// 5. try_to_release_in_transit(prev_state)
// 6. Remove: mark_task_completion() call (D-04)
```

### bounded_thread_pool Slot-Based Dispatch Pattern

```cpp
// Source: src/include/exec/bounded_thread_pool.hpp
// Pattern used throughout the codebase:
auto slot = _pool->reserve();   // blocks when at capacity, returns invalid if interrupted
if (!slot) break;                // interrupted -- exit loop
_pool->dispatch(std::move(slot), [...]() {
  // work runs on a pool worker thread
  // slot auto-releases when lambda completes
});
```

## Files Modified in This Phase

| File | Action | Notes |
|------|--------|-------|
| `src/include/downgrade/downgrade_executor.hpp` | **Rewrite** | Remove itask_executor inheritance, add own members, add downgrade_request struct |
| `src/downgrade/downgrade_executor.cpp` | **Rewrite** | Own start/stop/drain, processing_loop, updated monitor_loop, candidate selection preserved |
| `src/include/downgrade/downgrade_task.hpp` | **Rewrite** | Plain struct, remove itask/global_state/local_state hierarchy |
| `src/downgrade/downgrade_task.cpp` | **Rewrite** | Simplified execute(), remove mark_task_completion() |
| `test/cpp/downgrade/test_downgrade_executor.cpp` | **Update** | Adapt to new types (remove task_completion_message_queue usage, update task construction) |
| `CMakeLists.txt` | **No change** | Same source files, same test file |
| `src/sirius_context.cpp` | **No change expected** | Constructor signature preserved, start/stop/drain/get_space_id API preserved |

## Open Questions

1. **Should `run_downgrade_pass` remain public during Phase 1?**
   - What we know: Five existing tests call it directly. Making it private breaks them.
   - What's unclear: Whether to refactor tests now to use the request queue path, or keep the method public.
   - Recommendation: Keep it public for Phase 1 to minimize test churn. Tests verify candidate selection logic (CAND-01, CAND-02) independently of the request queue mechanism. Phase 2 can revisit visibility.

2. **Should `downgrade_request` live in its own header or inside `downgrade_executor.hpp`?**
   - What we know: Phase 2 exposes `request_free_memory()` returning `std::future<size_t>`, which means callers need to see the future but not necessarily the request struct.
   - What's unclear: Whether external callers ever construct requests directly.
   - Recommendation: Define `downgrade_request` in `downgrade_executor.hpp` as a nested or companion struct. It's an implementation detail. Phase 2's public API returns `std::future`, not the request itself.

## Sources

### Primary (HIGH confidence)
- `src/include/exec/bounded_thread_pool.hpp` -- Full API: reserve/dispatch/wait_all/interrupt/resume/stop
- `src/include/exec/interruptible_mpmc.hpp` -- Full API: push/pop/interrupt/drain/reactivate, smart_pointer concept requirement
- `src/include/parallel/task_executor.hpp` -- Base class being removed, documents start/stop/drain_and_wait lifecycle
- `src/parallel/task_executor.cpp` -- Implementation of start/stop/drain_and_wait (reference for new implementation)
- `src/downgrade/downgrade_executor.cpp` -- Current implementation (candidate selection logic to preserve)
- `src/downgrade/downgrade_task.cpp` -- Current task execute logic (core GPU->HOST conversion to preserve)
- `src/sirius_context.cpp` -- Caller integration points
- `test/cpp/downgrade/test_downgrade_executor.cpp` -- Existing test coverage (6 tests)
- `.planning/phases/01-foundation/01-CONTEXT.md` -- All decisions D-01 through D-07

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all primitives exist in-tree, no new dependencies
- Architecture: HIGH -- decisions D-01 through D-07 are specific and complete
- Pitfalls: HIGH -- derived from direct source code analysis of actual types and their constraints

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable -- internal refactoring with no external dependencies)
