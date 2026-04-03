# Domain Pitfalls

**Domain:** Concurrent task executor for GPU memory reclamation
**Researched:** 2026-04-03

## Critical Pitfalls

Mistakes that cause deadlocks, data corruption, or rewrites.

### Pitfall 1: Dangling Captures in Dispatch Lambdas

**What goes wrong:** Dispatch lambdas capture references to stack-local variables (`bytes_freed`, `remaining`, `done_cv`). If the request-processing thread returns from `process_request()` before all lambdas complete, those references become dangling pointers. Use-after-free, likely crash.

**Why it happens:** The wait logic has a bug (e.g., spurious wakeup without re-checking predicate, or `remaining` counter is wrong due to off-by-one).

**Consequences:** Memory corruption, intermittent crashes under load, extremely hard to debug.

**Prevention:**
- The atomic counter increment (`remaining.fetch_add(1)`) MUST happen BEFORE `dispatch()`, not after. If dispatch throws or the slot is invalid, the counter is never incremented (check slot validity before incrementing).
- The `done_cv.wait()` predicate MUST re-check `remaining.load() == 0` on every wakeup.
- Add a debug assertion after the wait: `assert(remaining.load() == 0)`.

**Detection:** TSAN (ThreadSanitizer) will flag use-after-free on the captured references. Run tests with `-fsanitize=thread`.

### Pitfall 2: Deadlock Between drain() and request_loop()

**What goes wrong:** `drain()` tries to join the request-processing thread, but that thread is blocked in `bounded_thread_pool::reserve()` waiting for a slot that will never become available because `drain()` already interrupted the pool.

**Why it happens:** Ordering of operations in `drain()`: if you interrupt the pool AFTER the request thread has already entered `reserve()` but BEFORE signaling `_running = false`, the thread is stuck in a `wait()` that was entered before the interrupt.

**Consequences:** `drain()` hangs forever. Query teardown freezes.

**Prevention:**
- Set `_running = false` AND call `_pool->interrupt()` AND notify `_request_cv` in a single critical section (or at least before attempting to join).
- `reserve()` returns an invalid slot when interrupted, which causes the request-processing loop to break.
- Test `drain()` under load: submit a request, then immediately drain while the request is being processed.

**Detection:** Hang in tests. Add a timeout to `drain()` with a LOG_ERROR if it exceeds (say) 30 seconds, to distinguish deadlocks from slow operations.

### Pitfall 3: Promise Set Twice (or Never)

**What goes wrong:** `std::promise::set_value()` called twice throws `std::future_error`. Called never means `future.get()` blocks forever.

**Why it happens:**
- Set twice: exception path sets the value, then normal path also sets it.
- Never set: early return from `process_request()` before reaching `set_value()`.

**Consequences:** Exception crash or permanent hang.

**Prevention:**
- `process_request()` should have a single `set_value()` call at the very end, after the wait-for-completion block.
- Use RAII: wrap the promise in a scope guard that sets value to 0 if `set_value()` was never called (e.g., on exception).
- Alternatively, use `set_value()` unconditionally at the end and ensure no other path calls it.

**Detection:** Unit test: submit a request, verify the future resolves. Submit a request with zero candidates, verify the future resolves with 0.

### Pitfall 4: CUDA Device Not Set on Worker Threads

**What goes wrong:** Worker threads call `batch->downgrade()` which internally does GPU operations, but `cudaSetDevice()` was never called on that thread.

**Why it happens:** The `per_thread_init` callback was forgotten or the pool was recreated without it.

**Consequences:** CUDA operations silently use device 0 (or whatever the default is), causing cross-device memory access errors or silent data corruption.

**Prevention:** The `per_thread_init` lambda must be passed to `bounded_thread_pool` constructor. Add a debug assertion at the start of every dispatch lambda: `int dev; cudaGetDevice(&dev); assert(dev == expected_device);`.

**Detection:** CUDA error codes from `cudaMemcpy` or similar operations. Hard to debug if the wrong device happens to work.

## Moderate Pitfalls

### Pitfall 1: Request Queue Not Drained on Stop

**What goes wrong:** `stop()` joins threads but does not drain pending requests. Their promises are never fulfilled.

**Prevention:** In `stop()`, after joining threads, drain the queue and call `set_value(0)` on each pending request's promise. Or: set an exception via `set_exception()` to signal cancellation.

### Pitfall 2: Monitor Loop Blocks on Full Request Processing

**What goes wrong:** Monitor thread calls `request_free_memory_and_wait()`, which blocks until the request completes. If the request takes a long time (many batches, slow GPU-to-host copies), the monitor stops polling.

**Prevention:** This is actually acceptable behavior -- the monitor does not need to poll while a downgrade is in progress. But document this explicitly. If monitor responsiveness becomes important, switch to async API with periodic polling.

### Pitfall 3: Predicate Evaluated on Stale Memory State

**What goes wrong:** The predicate checks `memory_space->get_current_usage()`, but freed batches have not been reflected in the memory accounting yet (e.g., RMM pool has not reclaimed the memory).

**Prevention:** Understand what `downgrade()` actually does -- if it moves data from GPU to host, the GPU memory should be freed before the downgrade lambda returns. If the free is deferred (e.g., via CUDA stream callback), the predicate may see stale state. Verify that the memory accounting is synchronous with the downgrade operation.

### Pitfall 4: Candidate Collection Races with Concurrent Modifications

**What goes wrong:** Candidates are collected (batch pointers acquired via `shared_ptr`), but between collection and dispatch, the batch state changes (e.g., another component marks it active).

**Prevention:** The `shared_ptr` keeps the batch alive. The downgrade operation itself should handle the case where the batch is no longer idle (return early, do not crash). This is the current behavior -- preserve it.

### Pitfall 5: Latch vs. Atomic Counter Off-by-One

**What goes wrong:** If using a `std::latch`, the count is set to the number of candidates, but early termination means fewer batches are dispatched. `latch.wait()` blocks forever.

**Prevention:** Use `std::atomic<int>` counter instead of `std::latch`. Increment before dispatch, decrement on completion. The counter naturally reaches zero regardless of how many batches were actually dispatched.

## Minor Pitfalls

### Pitfall 1: Sleep Duration in Monitor Loop

**What goes wrong:** 10ms sleep is either too frequent (wastes CPU) or too infrequent (slow response to memory pressure).

**Prevention:** Make it configurable. 10ms is reasonable for now (matches current implementation).

### Pitfall 2: Logging Overhead in Hot Path

**What goes wrong:** TRACE logging for every batch dispatch creates measurable overhead when downgrading hundreds of batches.

**Prevention:** Keep per-batch logging at TRACE level. Log per-request summary at INFO level.

### Pitfall 3: Thread Naming

**What goes wrong:** Unlabeled threads make debugging with `top -H`, `htop`, or `gdb` difficult.

**Prevention:** Use `pthread_setname_np` for the request-processing thread and monitor thread, as the codebase already does for thread pool workers.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Request struct + queue | Promise lifetime management | RAII scope guard on promise |
| Lifecycle (start/stop/drain) | Deadlock in drain | Test drain under active request processing |
| Batch dispatch loop | Dangling captures, counter off-by-one | TSAN testing, assertions on counter |
| Monitor loop integration | Blocking semantics change | Verify monitor behavior matches current |
| Drop itask_executor | Breaking SiriusContext interface | Keep same public method signatures |

## Sources

- Current implementation: `src/downgrade/downgrade_executor.cpp` (existing drain/stop/start patterns)
- `bounded_thread_pool` interrupt semantics: `src/include/exec/bounded_thread_pool.hpp`
- `std::promise` exception behavior: C++20 standard, `std::future_error`
- CUDA device affinity: Current `get_per_thread_init()` in `downgrade_executor.cpp` line 56-63
