# Architecture Patterns

**Domain:** Concurrent task executor for GPU memory reclamation
**Researched:** 2026-04-03

## Recommended Architecture

The downgrade_executor becomes a self-contained component with three internal threads and one shared thread pool:

```
                    External callers
                         |
                    request_free_memory() / request_free_memory_and_wait()
                         |
                         v
                  +------------------+
                  |  Request Queue   |  (mutex + CV + deque<request>)
                  +------------------+
                         |
                         v
              +------------------------+
              | Request Processing     |  <-- dedicated thread
              | Thread                 |
              |                        |
              | for each request:      |
              |   1. collect candidates|
              |   2. for each batch:   |
              |      - reserve() slot  |
              |      - check predicate |
              |      - dispatch()      |
              |   3. wait for all      |
              |      dispatched batches|
              |   4. set promise value  |
              +------------------------+
                         |
                    dispatch() calls
                         |
                         v
              +------------------------+
              | bounded_thread_pool    |  (N worker threads, CUDA device affinity)
              |                        |
              | Worker 1: downgrade()  |
              | Worker 2: downgrade()  |
              | ...                    |
              | Worker N: downgrade()  |
              +------------------------+

              +------------------------+
              | Monitor Thread         |  <-- dedicated thread
              |                        |
              | while (running):       |
              |   if should_downgrade: |
              |     request_free_      |
              |       memory_and_wait()|
              |   sleep(10ms)          |
              +------------------------+
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `downgrade_executor` (public API) | Accepts requests, manages lifecycle | External callers, SiriusContext |
| Request queue | Buffers pending requests, FIFO | Public API (producers), request processing thread (consumer) |
| Request processing thread | Dequeues one request at a time, collects candidates, dispatches batches, waits for completion | Request queue, bounded_thread_pool, candidate selection logic |
| `bounded_thread_pool` | Executes batch downgrades concurrently | Request processing thread (dispatch), CUDA runtime |
| Monitor thread | Polls memory pressure, submits blocking requests | Memory space (reads pressure), request queue (submits requests) |
| Candidate selection | Collects batches to downgrade from repos | Data repository manager, data repositories |

### Data Flow

1. **Request arrival:** Caller constructs a `downgrade_request` with predicate + promise, pushes to queue, returns future from promise.
2. **Request processing:** Dedicated thread pops request from queue. Calls `collect_candidates()` (reusing existing repo-walking logic). Iterates candidates: for each, `reserve()` a pool slot, check predicate (if satisfied, stop dispatching), `dispatch()` a lambda that calls `batch->downgrade(stream)`, increments `bytes_freed`, decrements `remaining` counter.
3. **Batch completion:** Each dispatched lambda finishes, atomically adds freed bytes, counts down remaining. Last batch notifies the request processing thread via CV.
4. **Result delivery:** Request processing thread wakes from `done_cv.wait()`, calls `promise.set_value(bytes_freed)`.
5. **Caller receives result:** `future.get()` returns (blocking API) or `future.wait_for()` polls (async API).

## Patterns to Follow

### Pattern 1: Request-Processing Loop (Sequential Consumer)

**What:** A single dedicated thread that processes requests one at a time from a FIFO queue.

**When:** When work items (requests) must not overlap but each request internally parallelizes.

**Why:** Sequential processing eliminates contention between requests competing for the same candidate batches. The request queue decouples producers (monitor, external callers) from the processing rate.

```cpp
void downgrade_executor::request_loop() {
  while (_running.load(std::memory_order_relaxed)) {
    downgrade_request req;
    {
      std::unique_lock lock(_request_mu);
      _request_cv.wait(lock, [&] {
        return !_request_queue.empty() || !_running.load(std::memory_order_relaxed);
      });
      if (!_running.load(std::memory_order_relaxed)) break;
      req = std::move(_request_queue.front());
      _request_queue.pop_front();
    }
    process_request(std::move(req));
  }
}
```

### Pattern 2: Dispatch-with-Predicate-Check Loop

**What:** Interleave slot reservation with predicate evaluation to enable early exit.

**When:** You need to stop dispatching new work as soon as a condition is met, while letting in-flight work finish.

```cpp
void downgrade_executor::process_request(downgrade_request req) {
  auto candidates = collect_all_candidates();

  std::atomic<size_t> bytes_freed{0};
  std::atomic<int>    remaining{0};
  std::mutex          done_mu;
  std::condition_variable done_cv;

  for (auto& batch : candidates) {
    // Check predicate BEFORE dispatching next batch
    if (req.predicate && req.predicate()) break;

    auto slot = _pool->reserve();
    if (!slot) break;  // pool interrupted (shutdown)

    remaining.fetch_add(1, std::memory_order_relaxed);

    size_t batch_size = batch->get_data()->get_size_in_bytes();
    _pool->dispatch(std::move(slot),
      [&bytes_freed, &remaining, &done_mu, &done_cv,
       batch = std::move(batch), batch_size,
       stream = rmm::cuda_stream_view{_stream}]() {
        try {
          batch->downgrade(stream);
          bytes_freed.fetch_add(batch_size, std::memory_order_relaxed);
        } catch (const std::exception& e) {
          SIRIUS_LOG_ERROR("[downgrade] batch failed: {}", e.what());
        }
        if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          std::lock_guard lock(done_mu);
          done_cv.notify_one();
        }
      });
  }

  // Wait for all dispatched batches
  if (remaining.load(std::memory_order_acquire) > 0) {
    std::unique_lock lock(done_mu);
    done_cv.wait(lock, [&] {
      return remaining.load(std::memory_order_acquire) == 0;
    });
  }

  req.result.set_value(bytes_freed.load(std::memory_order_relaxed));
}
```

### Pattern 3: Lifecycle Management (start/stop/drain)

**What:** Three-method lifecycle matching the existing SiriusContext contract.

```cpp
void start() {
  _running.store(true);
  _pool = std::make_unique<bounded_thread_pool>(config, ...);
  _request_thread = std::thread(&downgrade_executor::request_loop, this);
  _monitor_thread = std::thread(&downgrade_executor::monitor_loop, this);
}

void stop() {
  _running.store(false);
  _request_cv.notify_all();     // wake request loop
  _pool->interrupt();            // unblock any reserve() calls
  if (_monitor_thread.joinable()) _monitor_thread.join();
  if (_request_thread.joinable()) _request_thread.join();
  _pool->stop();
}

void drain() {
  // Interrupt the pool to unblock the request loop's reserve() calls
  _pool->interrupt();
  _request_cv.notify_all();

  // Wait for request thread to finish current request
  if (_request_thread.joinable()) _request_thread.join();

  // Wait for in-flight batches
  _pool->wait_all();

  // Clear pending requests (set their promises to 0)
  {
    std::lock_guard lock(_request_mu);
    while (!_request_queue.empty()) {
      _request_queue.front().result.set_value(0);
      _request_queue.pop_front();
    }
  }

  // Restart
  _pool->resume();
  _running.store(true);
  _request_thread = std::thread(&downgrade_executor::request_loop, this);
}
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Shared Mutable State in Dispatch Lambdas

**What:** Capturing raw pointers to stack-local variables in lambdas dispatched to the thread pool.

**Why bad:** If the request-processing thread proceeds past the wait (e.g., due to a bug), the lambda captures become dangling pointers.

**Instead:** The `process_request()` function MUST block until all dispatched lambdas complete before returning. The atomic counter + CV pattern guarantees this. The captures (`bytes_freed`, `remaining`, `done_mu`, `done_cv`) live on the stack of `process_request()` and are valid for the entire duration of all dispatched lambdas.

### Anti-Pattern 2: Using `wait_all()` for Per-Request Completion

**What:** Calling `bounded_thread_pool::wait_all()` to wait for a request's batches.

**Why bad:** `wait_all()` waits until ALL pool work is done (active count == 0). In the sequential-request model this is technically correct (only one request's batches are in the pool at a time), but it couples correctness to the sequential invariant. If the design ever changes (e.g., limited parallelism between requests), `wait_all()` breaks.

**Instead:** Use the per-request atomic counter + CV. Scoped precisely to the current request's batches.

### Anti-Pattern 3: Checking Predicate from Worker Threads

**What:** Having each worker thread check the predicate after completing a batch.

**Why bad:** The predicate may read memory space state that is updated non-atomically or requires synchronization. Multiple worker threads calling the predicate concurrently creates a data race if the predicate is not thread-safe. Also, worker threads cannot stop other workers' in-flight batches anyway.

**Instead:** Only the request-processing thread checks the predicate, between dispatches. This is single-threaded and safe regardless of predicate implementation.

### Anti-Pattern 4: Destroying the Thread Pool on Drain

**What:** Current `drain()` calls `stop()` then `start()`, which destroys and recreates the thread pool.

**Why bad:** Thread creation is expensive (kernel calls, stack allocation). The pool workers need `cudaSetDevice` initialization. Doing this per-query is wasteful.

**Instead:** Use `interrupt()` / `wait_all()` / `resume()` on the pool. The workers stay alive; only the request-processing thread is restarted.

## Scalability Considerations

| Concern | Current (1 GPU) | Multiple GPUs | Notes |
|---------|-----------------|---------------|-------|
| Request rate | ~1/10ms from monitor | Same per executor | One executor per memory space; no cross-executor contention |
| Batch count per request | Tens to hundreds | Same | Bounded by data in GPU memory |
| Thread pool size | 2-4 workers | Same per executor | Matches current config |
| Memory for request queue | Negligible | Negligible | Requests are tiny (predicate + promise) |

## Sources

- Current architecture: `src/downgrade/downgrade_executor.cpp`, `src/include/downgrade/downgrade_executor.hpp`
- `bounded_thread_pool` API: `src/include/exec/bounded_thread_pool.hpp`
- SiriusContext lifecycle: Referenced in PROJECT.md
- `itask_executor` base class: `src/include/parallel/task_executor.hpp`
