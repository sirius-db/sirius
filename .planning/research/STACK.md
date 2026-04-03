# Technology Stack

**Project:** Downgrade Executor Redesign
**Researched:** 2026-04-03

## Recommended Stack

The redesigned downgrade_executor needs no new dependencies. Everything required is available through C++20 standard library primitives and the existing codebase infrastructure (`bounded_thread_pool`, `absl::AnyInvocable`). The design is a composition of well-understood concurrency primitives, not a framework adoption.

### Core Primitives (C++20 Standard Library)

| Primitive | Purpose | Why |
|-----------|---------|-----|
| `std::promise<size_t>` / `std::future<size_t>` | Async result delivery for requests | Standard, zero-dependency, caller gets blocking `.get()` or polling `.wait_for()` for free. The result type is `size_t` (bytes freed). |
| `std::atomic<bool>` | Predicate early-termination flag per request | Lock-free signaling from batch-completion callbacks to the dispatch loop. `memory_order_relaxed` is sufficient since the predicate lambda captures external state that is itself atomic or externally synchronized. |
| `std::mutex` + `std::condition_variable` | Request queue synchronization | The request queue is single-consumer (the request-processing loop) and multi-producer (monitor loop + external callers). A simple mutex+CV is correct, efficient, and debuggable. No need for lock-free queues here because request arrival rate is low (order of seconds, not microseconds). |
| `std::latch` (C++20) | Waiting for all dispatched batches within a request to complete | Each request dispatches N batches. Create a `std::latch(N)`, count down from each batch completion lambda, then `latch.wait()` in the request loop. Clean, one-shot, no reset needed. Preferred over `std::barrier` because requests are one-shot (no reuse). |
| `std::function` or `absl::AnyInvocable` | Predicate type (`() -> bool`) | Use `absl::AnyInvocable<bool() const>` for consistency with the rest of the codebase and because it supports move-only captures (unlike `std::function`). |
| `std::deque<request>` | Request queue storage | FIFO, stable iterators, no need for concurrent access (protected by mutex). `std::queue<request>` wrapping `std::deque` also works. |
| `std::atomic<size_t>` | Accumulating bytes_freed across concurrent batch completions | Each batch completion atomically adds its freed bytes. Read by predicate and by final result. `fetch_add(bytes, memory_order_relaxed)` is sufficient. |

**Confidence:** HIGH -- these are all C++20 standard library facilities, verified against the project's existing use of C++20 (`std::latch` is already used in `bounded_thread_pool.hpp`).

### Existing Codebase Components (Reuse)

| Component | Current Location | Role in Redesign | Modification Needed |
|-----------|-----------------|------------------|---------------------|
| `bounded_thread_pool` | `src/include/exec/bounded_thread_pool.hpp` | Executes batch downgrades concurrently within a request | None. Use `reserve()` + `dispatch()` as-is. The pool's `wait_all()` is NOT used for per-request waiting (use `std::latch` instead, so the pool can be reused across requests without draining). |
| `thread_pool_config` | `src/include/exec/config.hpp` | Configuration for owned pool | None. |
| `absl::AnyInvocable` | Already a dependency | Predicate type, dispatch lambdas | None. Already used throughout. |
| `rmm::cuda_stream_view` | Already a dependency | Pass to batch `execute()` calls | None. |
| `cudaSetDevice` | CUDA runtime | Per-thread-init for pool workers | None. Same pattern as current `get_per_thread_init()`. |

**Confidence:** HIGH -- these are the project's own components, read directly from source.

### Components to Drop

| Component | Why Drop |
|-----------|----------|
| `itask_executor` inheritance | The base class models a queue-of-tasks with a manager loop that pops individual tasks. The redesign needs a queue-of-requests where each request internally dispatches batches. Fighting this abstraction adds complexity. Other executors (`pipeline_executor`, `duckdb_scan_executor`) keep using it. |
| `interruptible_mpmc<unique_ptr<itask>>` task queue | Replaced by a simple `std::deque<request>` protected by mutex+CV. The MPMC queue's high-throughput lock-free design is overkill for request-level granularity (tens of requests per query, not millions of tasks). |
| `itask` / `downgrade_task` class hierarchy | The task abstraction (`itask_local_state`, `itask_global_state`, virtual `execute()`) adds indirection for no benefit. The downgrade operation is a single function call (`data_batch::downgrade()`). Replace with a direct lambda in `dispatch()`. |
| `task_completion_message_queue` | The redesign uses `std::latch` + `std::atomic<size_t>` for intra-request coordination. The message queue's purpose (notifying `task_creator`) is not needed since the downgrade executor is self-contained. |

**Confidence:** HIGH -- based on direct reading of `PROJECT.md` requirements and the current source.

## Detailed Design Patterns

### Pattern 1: The Request Object

```cpp
struct downgrade_request {
  absl::AnyInvocable<bool() const> predicate;  // "are we done?"
  std::promise<size_t> result;                  // delivers bytes_freed to caller
};
```

**Why `std::promise`/`std::future`:** The caller needs either blocking (`get()`) or non-blocking (`wait_for()`) access to the result. `std::promise`/`std::future` provides exactly this with zero custom code. The alternative -- callback-based notification -- adds complexity (caller must manage callback lifetime, handle races between callback and destruction). Futures are value-typed and move-only, matching request semantics perfectly.

**Why NOT `std::async`:** `std::async` launches a new thread or uses an implementation-defined pool. We need precise control: the request loop runs on a dedicated thread, batch execution uses our `bounded_thread_pool` with CUDA device affinity. `std::async` provides neither.

**Confidence:** HIGH.

### Pattern 2: Request Queue (mutex + CV + deque)

```cpp
std::mutex              _request_mu;
std::condition_variable _request_cv;
std::deque<downgrade_request> _request_queue;
std::atomic<bool>       _running{false};
```

The request-processing thread blocks on `_request_cv.wait()` until a request arrives or `_running` becomes false. Producers (monitor loop, external callers) lock, push, notify.

**Why not a lock-free queue:** Request arrival rate is ~1 per 10ms (monitor loop interval) or on-demand from memory pressure callbacks. At this rate, mutex contention is unmeasurable. Lock-free queues (like `moodycamel::ConcurrentQueue`) add complexity without benefit and make debugging harder.

**Why not reuse `interruptible_mpmc`:** That queue wraps `moodycamel::BlockingConcurrentQueue` and is typed on smart pointers. Requests are value types with move-only fields (`std::promise`). Adapting the MPMC queue would require changing its template constraints. A 10-line mutex+CV+deque is simpler, correct, and fits the actual access pattern (single consumer, rare producers).

**Confidence:** HIGH.

### Pattern 3: Intra-Request Batch Coordination (latch + atomic)

For each request, the processing loop:

1. Collects candidate batches (existing `collect_candidates_from_partition` logic)
2. Creates a `std::latch(batch_count)` and `std::atomic<size_t> bytes_freed{0}`
3. For each batch: `reserve()` a slot, check predicate, if false `dispatch()` a lambda that:
   - Calls `batch->downgrade(stream)`
   - `bytes_freed.fetch_add(batch_size, std::memory_order_relaxed)`
   - `latch.count_down()`
4. After dispatching: `latch.wait()` (blocks until all dispatched batches complete)
5. Sets `promise.set_value(bytes_freed.load())`

**Early termination detail:** Between step 3 dispatches, after each `reserve()` returns, check the predicate. If it returns true, stop dispatching new batches but do NOT cancel in-flight ones. Adjust the latch count: create the latch with the actual number of dispatched batches (known at the end of the dispatch loop), not the candidate count.

**Implementation note on latch sizing:** Since the number of dispatched batches is not known upfront (early termination may cut it short), use a two-phase approach:
- Phase A: Dispatch batches, counting how many were actually dispatched (`n_dispatched`)
- Between dispatches, check predicate; stop dispatching if true
- Phase B: Create `std::latch(n_dispatched)` BEFORE starting dispatch? No -- latch must exist before any completion runs.

**Better approach:** Use `std::atomic<int> remaining{0}` + `std::condition_variable`:
```cpp
std::atomic<int>        remaining{0};
std::atomic<size_t>     bytes_freed{0};
std::mutex              done_mu;
std::condition_variable done_cv;

// In dispatch lambda:
bytes_freed.fetch_add(size, std::memory_order_relaxed);
if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
  done_cv.notify_one();
}

// In request loop:
remaining.fetch_add(1, std::memory_order_relaxed);  // before each dispatch
// ... after all dispatches:
std::unique_lock lock(done_mu);
done_cv.wait(lock, [&] { return remaining.load(std::memory_order_acquire) == 0; });
```

This avoids the latch-sizing problem entirely. The atomic counter is incremented before dispatch and decremented after completion. The CV notifies only when the last batch completes.

**Why this over `bounded_thread_pool::wait_all()`:** `wait_all()` blocks until ALL pool work is done. If another request's batches were somehow in the pool (they shouldn't be due to sequential processing, but defense-in-depth), `wait_all()` would wait for those too. The per-request counter is scoped exactly to the current request's batches.

**Confidence:** HIGH.

### Pattern 4: Predicate Checking (relaxed atomics)

The predicate is called from the request-processing thread between batch dispatches. It is NOT called from worker threads (to avoid contention on whatever state the predicate reads). The predicate captures external state -- typically `memory_space->get_current_usage()` compared against a target.

**Memory ordering:** The predicate reads GPU memory usage counters that are themselves updated atomically by RMM/cuCascade. `memory_order_relaxed` on the early-termination flag is fine because:
- The flag is only read by one thread (the dispatch loop)
- Staleness of a few microseconds is acceptable (worst case: one extra batch dispatched)

**Confidence:** HIGH.

### Pattern 5: CUDA Stream Handling

The executor owns a `cudaStream_t` created with `cudaStreamNonBlocking`. All batch downgrades within a request share this stream (passed as `rmm::cuda_stream_view`). This matches the current implementation.

**Thread safety consideration:** Multiple worker threads will call `batch->downgrade(stream)` concurrently on the same CUDA stream. This is safe because:
- CUDA stream operations from different host threads are serialized by the CUDA driver
- `cudaStreamNonBlocking` means the stream does not synchronize with the default stream

However, if batch downgrades involve host-side work after the CUDA operations, the stream serialization may create unnecessary sequencing. If profiling shows this is a bottleneck, consider one stream per worker thread. For now, single stream matches the current design and is the simpler starting point.

**Confidence:** MEDIUM -- the single-stream-multi-thread pattern is correct per CUDA semantics, but performance implications depend on workload characteristics that should be validated.

## What NOT to Use

| Technology | Why Not |
|------------|---------|
| `std::async` | No control over thread pool, CUDA device affinity, or concurrency limits. Would bypass `bounded_thread_pool` entirely. |
| `std::jthread` + `std::stop_token` (C++20) | Tempting for the monitor and request-processing threads, but the codebase consistently uses `std::thread` + `std::atomic<bool>` for lifecycle control. Switching one executor to `jthread` creates inconsistency. The benefit (automatic join, cooperative cancellation) is marginal here since `stop()` already handles join. |
| Coroutines (`co_await`) | C++20 coroutines lack a standard executor/scheduler. Would require a third-party library (e.g., cppcoro, libunifex) or custom infrastructure. Massive complexity for no benefit in a simple request-queue pattern. |
| `std::execution` (C++26 Senders/Receivers) | Not available in C++20. Even with P2300 library implementations, this is experimental and adds a heavy abstraction layer. |
| `folly::Future` / `folly::Executor` | External dependency (Facebook Folly). `std::future` is sufficient for this use case. Folly futures add `.then()` chaining which we don't need. |
| Lock-free request queue | Over-engineered for ~1 request per 10ms. Adds debugging difficulty for zero performance gain. |
| `std::counting_semaphore` | Could replace the atomic counter + CV for batch completion tracking, but `std::latch` or atomic+CV is more readable and the intent is clearer. Semaphores are better for reusable capacity-limiting (which `bounded_thread_pool` already handles). |
| `std::barrier` | For repeating synchronization points. Requests are one-shot, so `std::latch` or atomic counter is the right choice. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Async result | `std::promise`/`std::future` | Callback (`AnyInvocable<void(size_t)>`) | Callback forces caller to manage lifetime and synchronization. Future is simpler for both fire-and-forget and blocking use cases. |
| Batch completion tracking | `std::atomic<int>` + CV | `std::latch` | Latch requires knowing count upfront, which is incompatible with early termination (we don't know how many batches will be dispatched until the loop finishes). Atomic counter allows increment-before-dispatch, decrement-on-complete. |
| Request queue | `std::deque` + mutex + CV | `interruptible_mpmc` | MPMC is typed on smart pointers, overkill for low-rate requests, and harder to debug. |
| Predicate type | `absl::AnyInvocable<bool() const>` | `std::function<bool()>` | `std::function` does not support move-only captures. The predicate may need to capture move-only state (e.g., unique ownership of a memory snapshot). Codebase already uses `absl::AnyInvocable` everywhere. |
| Thread lifecycle | `std::thread` + `std::atomic<bool>` | `std::jthread` + `std::stop_token` | Codebase consistency. `jthread` is better in isolation but would be the only usage in Sirius. |

## Installation / Dependencies

No new dependencies. Everything is already available:

```
C++20 standard library: <future>, <atomic>, <mutex>, <condition_variable>, <latch>, <deque>, <thread>
Existing Sirius: bounded_thread_pool, thread_pool_config, absl::AnyInvocable
Existing CUDA: cuda_runtime_api.h, rmm::cuda_stream_view
```

## Sources

- C++20 `std::latch`: Used in `src/include/exec/bounded_thread_pool.hpp` line 8 (`#include <latch>`) and line 108 (`std::make_unique<std::latch>(capacity)`)
- `bounded_thread_pool` slot/reserve/dispatch pattern: `src/include/exec/bounded_thread_pool.hpp`
- `absl::AnyInvocable` usage: `src/include/exec/bounded_thread_pool.hpp` line 96, `src/include/parallel/task_executor.hpp` line 118
- `interruptible_mpmc` design: `src/include/exec/interruptible_mpmc.hpp`
- Current downgrade_executor: `src/downgrade/downgrade_executor.cpp`, `src/include/downgrade/downgrade_executor.hpp`
- Project requirements: `.planning/PROJECT.md`
- CUDA stream thread safety: CUDA Programming Guide, Section 3.2.8 (concurrent kernel execution and stream semantics)
