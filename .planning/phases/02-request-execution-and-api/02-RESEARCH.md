# Phase 2: Request Execution and API - Research

**Researched:** 2026-04-06
**Domain:** C++ concurrent dispatch loop redesign, std::future/promise API surface
**Confidence:** HIGH

## Summary

Phase 2 transforms the `downgrade_executor::processing_loop` from collect-all/dispatch-all/wait_all into an incremental dispatch loop that checks a predicate after each batch completion. It also exposes three public API methods (`request_free_memory`, `request_free_memory_and_wait`, `request_downgrade`) and removes the legacy `run_downgrade_pass` methods.

The implementation is entirely within the existing `downgrade_executor` class using infrastructure already proven in Phase 1: `bounded_thread_pool` (reserve/dispatch/wait_all), `interruptible_mpmc` (request queue), `downgrade_task` (plain struct execute), and `downgrade_request` (already has predicate/promise/target_bytes fields from Phase 1). No new libraries or external dependencies are needed.

The core technical challenge is the incremental dispatch loop: dispatching up to pool-width batches concurrently, detecting individual batch completions to check the predicate, and stopping new dispatches while letting in-flight batches finish. The `bounded_thread_pool::slot` RAII pattern and `std::atomic<bool>` predicate-satisfied flag provide clean mechanisms for this.

**Primary recommendation:** Implement the incremental dispatch loop using an `atomic<bool> satisfied` flag set by dispatch lambdas after each successful `task.execute()`, with the dispatch loop checking this flag before reserving the next slot. Use `atomic<size_t> bytes_freed` for accounting. All three public methods construct a `downgrade_request` and push to `_request_queue`; the blocking variant calls `.get()` on the returned future.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** The processing loop evolves from collect-all/dispatch-all/wait_all to incremental dispatch: dispatch up to pool-width batches concurrently, check predicate after each batch completion via `atomic<bool> satisfied` flag and condition variable, stop dispatching new batches when predicate is satisfied. In-flight batches finish naturally via `pool->wait_all()` after dispatch loop exits.
- **D-02:** Every request always has a predicate -- there is no null-predicate path. Byte-based requests construct a default predicate that checks `bytes_freed >= target_bytes`. The dispatch loop always calls `req.predicate()` after each batch completion, no conditional.
- **D-03:** `downgrade_request` gains an `atomic<size_t> bytes_freed{0}` member. Each dispatch lambda adds `batch->get_data()->get_size_in_bytes()` after successful `task.execute()`. The default byte-predicate captures a reference to this counter.
- **D-04:** The final `bytes_freed` value (including in-flight batches that finish after predicate is satisfied) is set into `req.result` (the promise) after `pool->wait_all()` returns.
- **D-05:** Three separate public methods with distinct signatures:
  - `std::future<size_t> request_free_memory(size_t bytes)` -- async, byte-based
  - `size_t request_free_memory_and_wait(size_t bytes)` -- blocking, byte-based
  - `std::future<size_t> request_downgrade(std::function<bool()> predicate)` -- async, predicate-based
- **D-06:** All three methods build a `downgrade_request`, push it to `_request_queue`, and return. The blocking variant calls `.get()` on the future. No direct dispatch -- everything goes through the queue and the processing thread.
- **D-07:** Remove both `run_downgrade_pass(repos, bytes)` and `run_downgrade_pass_all_repos(bytes)` from the public and private API. All downgrade work flows through the request queue via the new API methods. The candidate collection logic (`collect_all_candidates`) remains as a private helper called by the processing loop.

### Claude's Discretion
- Exact condition variable / notification mechanism for dispatch-thread wakeup after batch completion
- Whether `request_downgrade(predicate)` also takes a `target_bytes` hint for candidate collection, or collects all available candidates
- Internal error handling within dispatch lambdas (log-and-continue is established from Phase 1)
- Exact thread synchronization details in `processing_loop` between dispatch and completion tracking

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RAPI-01 | Predicate-based request that downgrades batches until predicate returns true or candidates exhausted | D-01, D-02: incremental dispatch with predicate check after each batch; collect_all_candidates provides candidate list |
| RAPI-02 | `request_free_memory(bytes)` returns `std::future<size_t>` (non-blocking) | D-05, D-06: builds downgrade_request with byte-predicate, pushes to queue, returns future from promise |
| RAPI-03 | `request_free_memory_and_wait(bytes)` blocks and returns bytes freed | D-05, D-06: calls request_free_memory then .get() on future |
| RAPI-04 | Predicate-based API supports async via `std::future<size_t>` | D-05: request_downgrade(predicate) returns future |
| RAPI-05 | Partial fulfillment -- frees what is available, returns actual bytes freed | D-03, D-04: atomic bytes_freed counter set into promise after wait_all; naturally handles partial fulfillment when candidates exhausted |
| EXEC-03 | Multiple batch downgrades within a request execute concurrently via thread pool | D-01: dispatch up to pool-width batches; bounded_thread_pool handles concurrency |
| EXEC-04 | Predicate checked after each batch completion; no new dispatches when satisfied, in-flight finish | D-01: atomic<bool> satisfied flag checked in dispatch loop before next reserve() |
| EXEC-05 | Individual batch failures are non-fatal -- logged and skipped | Established pattern from Phase 1: try/catch in dispatch lambda with SIRIUS_LOG_ERROR |
</phase_requirements>

## Architecture Patterns

### Current State (Phase 1 Output)

```
downgrade_executor
  |-- bounded_thread_pool _pool          (reserve/dispatch/wait_all)
  |-- interruptible_mpmc _request_queue  (pop blocks in processing_loop)
  |-- processing_thread                  (collect-all / dispatch-all / wait_all)
  |-- monitor_thread                     (polls should_downgrade_memory)
  |-- downgrade_request { target_bytes, predicate, promise }  (predicate/promise unused)
```

### Target State (Phase 2 Output)

```
downgrade_executor
  |-- bounded_thread_pool _pool
  |-- interruptible_mpmc _request_queue
  |-- processing_thread                  (incremental dispatch with predicate)
  |-- monitor_thread                     (unchanged -- still calls run_downgrade_pass_all_repos
  |   |                                   until Phase 3 wires it to request_free_memory)
  |-- NEW: request_free_memory(bytes) -> future<size_t>
  |-- NEW: request_free_memory_and_wait(bytes) -> size_t
  |-- NEW: request_downgrade(predicate) -> future<size_t>
  |-- REMOVED: run_downgrade_pass(repos, bytes)
  |-- REMOVED: run_downgrade_pass_all_repos(bytes)
```

### Pattern: Incremental Dispatch Loop

**What:** The processing loop dispatches candidates one at a time (up to pool capacity), checks the predicate after each completion, and stops dispatching when satisfied.

**Key mechanism:** Each dispatch lambda, after successful `task.execute()`, atomically adds bytes to `req.bytes_freed` and sets `req.satisfied = true` if `req.predicate()` returns true. The dispatch loop checks `req.satisfied` before calling `pool->reserve()` for the next candidate.

```cpp
// Pseudocode for the new processing_loop body (per request):
auto candidates = collect_all_candidates(repos, req->target_bytes);

for (auto& batch : candidates) {
  if (req->satisfied.load()) break;  // predicate met, stop dispatching

  auto slot = _pool->reserve();
  if (!slot) break;  // interrupted

  auto batch_size = batch->get_data()->get_size_in_bytes();
  _pool->dispatch(std::move(slot),
    [batch = std::move(batch), &req, &res_mgr = _reservation_manager,
     stream = rmm::cuda_stream_view{_stream}, batch_size]() {
      downgrade_task task{batch, res_mgr};
      try {
        task.execute(stream);
        req->bytes_freed.fetch_add(batch_size, std::memory_order_relaxed);
        if (req->predicate()) {
          req->satisfied.store(true, std::memory_order_release);
        }
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("[downgrade] batch downgrade failed: {}", e.what());
      }
    });
}

_pool->wait_all();  // let in-flight batches finish
req->result.set_value(req->bytes_freed.load());
```

### Pattern: Public API Methods

```cpp
std::future<size_t> downgrade_executor::request_free_memory(size_t bytes) {
  auto req = std::make_unique<downgrade_request>();
  req->target_bytes = bytes;
  req->bytes_freed.store(0);
  req->predicate = [&freed = req->bytes_freed, bytes]() {
    return freed.load(std::memory_order_relaxed) >= bytes;
  };
  auto future = req->result.get_future();
  _request_queue.push(std::move(req));
  return future;
}

size_t downgrade_executor::request_free_memory_and_wait(size_t bytes) {
  return request_free_memory(bytes).get();
}

std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate) {
  auto req = std::make_unique<downgrade_request>();
  req->target_bytes = 0;  // collect all available candidates
  req->bytes_freed.store(0);
  req->predicate = std::move(predicate);
  auto future = req->result.get_future();
  _request_queue.push(std::move(req));
  return future;
}
```

### Anti-Patterns to Avoid

- **Checking predicate from dispatch thread only:** The predicate must be checked from the worker thread (inside the dispatch lambda) after each batch completes. The dispatch loop reads the `satisfied` flag but does not call the predicate itself -- this avoids synchronization issues.
- **Using condition_variable for batch completion notification:** The `bounded_thread_pool::reserve()` already blocks when at capacity and unblocks as slots become available. The dispatch loop naturally paces itself. A condition_variable for completion notification is unnecessary complexity; the `atomic<bool> satisfied` flag suffices.
- **Calling predicate without memory ordering:** The predicate lambda captures `&bytes_freed` by reference. Since `bytes_freed` is `atomic<size_t>`, the predicate's `load()` is inherently thread-safe. Use `memory_order_relaxed` for the counter since exact ordering is not critical (we tolerate a few extra batches).
- **Moving the promise before getting the future:** `req->result.get_future()` must be called before `req` is moved into the queue. The current pattern (get future, then push) is correct.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Concurrency limiting | Manual semaphore | `bounded_thread_pool::reserve()` | Already proven, handles interrupt, RAII slot release |
| Task completion tracking | Completion callback system | `pool->wait_all()` + `atomic<bool>` | Simple, no coordination overhead, covers in-flight batches |
| Request queuing | Custom lock-free queue | `interruptible_mpmc` | Already supports interrupt/drain for lifecycle |
| Async result delivery | Callback chains | `std::promise/std::future` | Standard C++, already in downgrade_request struct |

## Common Pitfalls

### Pitfall 1: Promise Lifetime vs. Queue Move
**What goes wrong:** If `get_future()` is called after the request is moved into the queue, the promise is gone and the call is undefined behavior.
**Why it happens:** `std::promise` is move-only; after moving the `unique_ptr<downgrade_request>` into the queue, the caller no longer has access.
**How to avoid:** Always call `req->result.get_future()` before `_request_queue.push(std::move(req))`.
**Warning signs:** Crash or exception on `get_future()` call.

### Pitfall 2: Predicate Captures Dangling Reference
**What goes wrong:** The byte-predicate lambda captures `&req->bytes_freed`. If the request is destroyed before the predicate is called, the reference dangles.
**Why it happens:** The request is owned by `unique_ptr` in the queue, then by the processing loop. The predicate is called inside dispatch lambdas that run concurrently.
**How to avoid:** The processing loop owns the `unique_ptr<downgrade_request>` for the entire duration of dispatch + `wait_all()`. The predicate is only called from dispatch lambdas that run within this window. After `wait_all()` returns, no more lambdas reference the request. This is safe by design -- no extra lifetime management needed.
**Warning signs:** Use-after-free ASAN errors in predicate calls.

### Pitfall 3: batch_size Accounting After Failed Execute
**What goes wrong:** If `task.execute()` throws, the batch was NOT downgraded, but `bytes_freed` might be incremented if accounting is done before the try block.
**Why it happens:** Placing `bytes_freed.fetch_add()` before the try/catch or outside the success path.
**How to avoid:** Only increment `bytes_freed` after `task.execute()` succeeds (inside try, after execute, before catch).
**Warning signs:** `bytes_freed` exceeds actual GPU memory freed; predicate satisfied prematurely.

### Pitfall 4: monitor_loop After run_downgrade_pass Removal
**What goes wrong:** Phase 2 removes `run_downgrade_pass_all_repos` but the monitor_loop currently calls it. If not handled, the monitor_loop has a dangling call.
**Why it happens:** Phase boundary: D-07 removes the methods, but Phase 3 wires monitor_loop to `request_free_memory`.
**How to avoid:** The monitor_loop must be updated in Phase 2 to use `request_free_memory` (or `request_free_memory_and_wait`) instead of `run_downgrade_pass_all_repos`. Even though CONTEXT.md says lifecycle wiring is Phase 3, the removal of the called method forces this change in Phase 2. Alternatively, keep the monitor_loop calling the new API but defer the full lifecycle wiring.
**Warning signs:** Compilation failure after removing run_downgrade_pass_all_repos.

### Pitfall 5: collect_all_candidates with target_bytes=0 for Predicate Requests
**What goes wrong:** `request_downgrade(predicate)` sets `target_bytes = 0`. The current `collect_all_candidates` uses `target_bytes` to limit collection; with 0 it collects nothing.
**Why it happens:** The byte-limiting logic in `collect_candidates_from_partition` checks `if (max_bytes > 0 && collected_bytes >= max_bytes) break;` -- with max_bytes=0, the condition `max_bytes > 0` is false, so it does NOT break early. This means it collects ALL candidates. This is the correct behavior for predicate-only requests.
**How to avoid:** No action needed -- the existing guard handles this correctly. But verify this understanding in the code (line 207 of downgrade_executor.cpp).

### Pitfall 6: Data Race on downgrade_request Members
**What goes wrong:** The processing loop reads `req->bytes_freed` and `req->satisfied` while dispatch lambdas write to them concurrently.
**Why it happens:** Multiple threads accessing the same struct members.
**How to avoid:** Both `bytes_freed` and `satisfied` must be `std::atomic`. The `predicate` field is read-only after construction (never mutated), so no atomicity needed there. The `result` (promise) is only set once from the processing thread after `wait_all()`.

## Code Examples

### downgrade_request Struct Evolution

```cpp
// Phase 1 (current):
struct downgrade_request {
  size_t target_bytes{0};
  std::function<bool()> predicate;  // present but unused
  std::promise<size_t> result;      // present but unused
};

// Phase 2 (target):
struct downgrade_request {
  size_t target_bytes{0};
  std::function<bool()> predicate;
  std::promise<size_t> result;
  std::atomic<size_t> bytes_freed{0};    // NEW: per-request counter
  std::atomic<bool> satisfied{false};     // NEW: predicate-met flag
};
```

**Note:** Adding `atomic` members makes `downgrade_request` non-movable. This is fine because requests are allocated via `std::make_unique<downgrade_request>()` and passed through the queue as `unique_ptr` -- only the pointer moves, never the struct.

### Dispatch Lambda Pattern

```cpp
// Source: established pattern from Phase 1 dispatch + new accounting
_pool->dispatch(std::move(slot),
  [batch = std::move(batch), req_ptr = req.get(),
   &res_mgr = _reservation_manager,
   stream = rmm::cuda_stream_view{_stream}, batch_size]() {
    downgrade_task task{batch, res_mgr};
    try {
      task.execute(stream);
      req_ptr->bytes_freed.fetch_add(batch_size, std::memory_order_relaxed);
      if (req_ptr->predicate()) {
        req_ptr->satisfied.store(true, std::memory_order_release);
      }
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("[downgrade] batch downgrade failed: {}", e.what());
    }
  });
```

**Note:** The lambda captures `req.get()` (raw pointer), not the `unique_ptr`. This is safe because the processing loop holds the `unique_ptr` alive through `wait_all()`.

### Monitor Loop Update (Necessary Due to D-07)

```cpp
// Current monitor_loop calls run_downgrade_pass_all_repos (being removed).
// Must switch to request_free_memory_and_wait or request_free_memory:
void downgrade_executor::monitor_loop() {
  using namespace std::chrono_literals;
  while (_running.load()) {
    if (_memory_space && _memory_space->should_downgrade_memory()) {
      size_t amount = _memory_space->get_amount_to_downgrade();
      if (amount > 0) {
        // Use the new API -- push a request through the queue
        auto req = std::make_unique<downgrade_request>();
        req->target_bytes = amount;
        req->bytes_freed.store(0);
        req->predicate = [&freed = req->bytes_freed, amount]() {
          return freed.load(std::memory_order_relaxed) >= amount;
        };
        _request_queue.push(std::move(req));
      }
    }
    std::this_thread::sleep_for(10ms);
  }
}
```

**Design choice:** The monitor_loop does NOT call `request_free_memory()` (the public API) because that would create a future the monitor doesn't need. Instead, it directly constructs a request and pushes it -- the fire-and-forget pattern. The monitor_loop doesn't need to know when the request completes.

## Research Findings

### Memory Accounting Synchronicity (STATE.md Blocker)
**Question:** Is `memory_space` usage reporting synchronous after `batch->downgrade()` returns?
**Finding:** YES (HIGH confidence). `downgrade_task::execute()` calls `batch->convert_to<host_data_representation>(...)` which is a synchronous GPU-to-HOST transfer. After `task.execute()` returns, the batch's data representation has changed and `get_size_in_bytes()` reflects the pre-transfer size (captured before execute). The `bytes_freed` accounting is reliable. The memory_space's internal tracking updates as part of the conversion, not asynchronously.
**Evidence:** The `convert_to` call in `downgrade_task.cpp:69` blocks until completion. The CUDA stream used is `_stream` which is created with `cudaStreamNonBlocking`, but the `convert_to` call itself synchronizes within cucascade.

### bounded_thread_pool::reserve() as Concurrency Limiter
**Finding:** `reserve()` blocks when all slots are in use (active_ == capacity_). This naturally limits concurrent dispatches to pool width. The dispatch loop calling `reserve()` before each batch dispatch achieves the "dispatch up to pool-width batches concurrently" requirement without explicit counting.
**Confidence:** HIGH -- verified from `bounded_thread_pool.hpp` line 143-148.

### interruptible_mpmc::push() with Predicate-Containing Requests
**Finding:** The queue stores `unique_ptr<downgrade_request>`. Since Phase 2 adds `atomic` members to `downgrade_request` (making it non-movable), the `unique_ptr` wrapper is essential -- only the pointer is moved through the queue, not the struct. This pattern is already in use from Phase 1.
**Confidence:** HIGH -- verified from current code.

### collect_all_candidates with target_bytes=0
**Finding:** When `target_bytes` is 0, `collect_candidates_from_partition` collects ALL idle batches on the source space because the guard `if (max_bytes > 0 && collected_bytes >= max_bytes)` short-circuits on `max_bytes > 0 == false`. This is the correct behavior for predicate-only requests where the caller doesn't know how many bytes they need.
**Confidence:** HIGH -- verified from `downgrade_executor.cpp` line 207.

## Test Impact

### Existing Tests That Must Change
The following tests call `run_downgrade_pass` or `run_downgrade_pass_all_repos` which are being removed (D-07):
- `"run_downgrade_pass with empty repositories returns 0"` -- remove or rewrite using new API
- `"run_downgrade_pass downgrades GPU batches from a single non-partitioned repo"` -- rewrite using `request_free_memory_and_wait`
- `"run_downgrade_pass respects amount_to_downgrade limit"` -- rewrite using `request_free_memory`
- `"run_downgrade_pass prioritizes partitioned repos over non-partitioned"` -- rewrite (candidate selection preserved, just different entry point)
- `"run_downgrade_pass iterates partitions from last to first"` -- rewrite
- `"run_downgrade_pass skips active partitions in first pass"` -- rewrite
- `"run_downgrade_pass skips batches already on HOST"` -- rewrite

### New Tests Needed
- `request_free_memory` returns future that resolves to bytes freed
- `request_free_memory_and_wait` blocks and returns bytes freed
- `request_downgrade` with custom predicate stops dispatching when predicate satisfied
- Partial fulfillment: fewer candidates than target_bytes, returns actual bytes freed
- Individual batch failure does not crash executor or abort request
- Concurrent batch execution within single request (verify pool-width parallelism)

## Project Constraints (from CLAUDE.md)

- **Code style:** C++ 20, 2-space indent, 100-char line limit, clang-format enforced
- **Naming:** snake_case for functions, member variables with underscore suffix for private
- **Namespace:** `sirius::parallel` (existing namespace for downgrade_executor)
- **Smart pointers:** `duckdb::unique_ptr` and `duckdb::shared_ptr` at API boundaries; `std::unique_ptr` acceptable for internal-only types like downgrade_request
- **Logging:** `SIRIUS_LOG_ERROR`, `SIRIUS_LOG_TRACE`, `SIRIUS_LOG_DEBUG` macros
- **Testing:** Catch2 framework, test files in `test/cpp/downgrade/`
- **License:** Apache 2.0 header required on all source files
- **Pre-commit:** clang-format, codespell
- **Build:** `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make`
- **Unit test run:** `build/release/extension/sirius/test/cpp/sirius_unittest "[downgrade_executor]"`

## Open Questions

1. **request_downgrade candidate collection scope**
   - What we know: Byte-based requests use `target_bytes` to limit candidate collection. Predicate-only requests set `target_bytes = 0` which collects all candidates.
   - What's unclear: Should `request_downgrade` accept an optional `target_bytes` hint to limit candidate collection? Large repositories could produce very large candidate lists.
   - Recommendation: Start without the hint (collect all). If profiling shows candidate collection is a bottleneck, add the hint parameter later. The predicate will stop dispatch early regardless.

2. **Monitor loop transition**
   - What we know: D-07 removes `run_downgrade_pass_all_repos`. Monitor loop currently calls it. Phase 3 says it handles lifecycle wiring.
   - What's unclear: Exact boundary -- does Phase 2 update monitor_loop or leave a compile error?
   - Recommendation: Phase 2 MUST update monitor_loop to use the request queue directly (fire-and-forget pattern shown above) since the called method is being removed. This is not lifecycle wiring -- it's just replacing a deleted method call.

## Sources

### Primary (HIGH confidence)
- `src/include/downgrade/downgrade_executor.hpp` -- current class definition with Phase 1 output
- `src/downgrade/downgrade_executor.cpp` -- current processing_loop, candidate collection, run_downgrade_pass
- `src/include/exec/bounded_thread_pool.hpp` -- reserve/dispatch/wait_all mechanics
- `src/include/exec/interruptible_mpmc.hpp` -- queue push/pop/interrupt/drain
- `src/downgrade/downgrade_task.cpp` -- task.execute() synchronous behavior
- `test/cpp/downgrade/test_downgrade_executor.cpp` -- existing test patterns

### Secondary (MEDIUM confidence)
- `.planning/phases/01-foundation/01-CONTEXT.md` -- Phase 1 decisions carried forward
- `.planning/REQUIREMENTS.md` -- requirement definitions
- `.planning/phases/02-request-execution-and-api/02-CONTEXT.md` -- locked decisions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all infrastructure exists from Phase 1
- Architecture: HIGH -- incremental dispatch pattern is straightforward with existing primitives
- Pitfalls: HIGH -- identified from direct code inspection, all verifiable

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable internal refactoring, no external dependencies)
