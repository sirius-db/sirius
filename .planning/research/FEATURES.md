# Feature Landscape

**Domain:** Memory reclamation executor for GPU-native SQL engine (downgrade_executor redesign)
**Researched:** 2026-04-03

## Table Stakes

Features the redesigned downgrade_executor must have. Missing any of these means the new design is not shippable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Predicate-driven request completion | Core abstraction shift: "free until condition met" rather than "free N batches." Every memory pressure system (RMM `failure_callback_resource_adaptor`, database buffer pool evictors) uses a predicate or threshold check to decide when to stop. Without this, the executor cannot serve callers with heterogeneous stopping conditions. | Medium | The predicate `() -> bool` is evaluated after each batch completes. Must handle thread-safe evaluation since the dispatch loop thread calls it while workers modify memory state concurrently. |
| Byte-based convenience wrapper | Most callers know bytes, not predicates. `request_free_memory(size_t bytes)` wraps predicate API with a lambda checking `memory_space->get_available_memory()` against target. Mirrors RMM's pattern where OOM handlers receive the failed allocation size. | Low | Thin wrapper over predicate API. The lambda captures the memory_space pointer and target bytes. |
| Blocking (synchronous) API | Callers in the critical allocation path need "free memory and block until done." Every buffer pool manager provides synchronous eviction. RMM's `failure_callback_resource_adaptor` blocks in `do_allocate` until the callback succeeds or gives up. | Low | Returns `size_t` bytes actually freed. Uses `std::promise`/`std::future` internally, caller calls `.get()`. |
| Async (non-blocking) API | Fire-and-forget callers (monitor loop, speculative pre-eviction) need to submit requests without blocking. Returns `std::future<size_t>`. Standard pattern in concurrent task schedulers. | Low | Same internal mechanism as blocking; caller just holds the future without immediately calling `.get()`. |
| Sequential request processing | Only one request active at a time. Prevents two concurrent requests from competing for the same candidate batches, causing double-counting or race conditions on batch state transitions. Database buffer pool managers serialize eviction decisions for the same reason -- candidate selection must see consistent state. | Medium | Requires a request queue (MPMC via existing `interruptible_mpmc`) with a dedicated dispatch loop that pops one request, executes it to completion, then pops the next. |
| Concurrent batch downgrades within a request | A single request fans out to multiple worker threads performing GPU-to-HOST copies in parallel. This is the performance-critical path -- serial batch downgrade would be far too slow under memory pressure because PCIe bandwidth is high but latency per transfer is non-trivial. Current implementation already does this via `bounded_thread_pool`. | Medium | Reuse existing `bounded_thread_pool`. Workers must call `cudaSetDevice` on init (already implemented via `get_per_thread_init()`). Each worker executes one `data_batch::downgrade()` and reports bytes freed. |
| Predicate checked after each batch completion | Early exit: as soon as enough memory is freed, stop dispatching new batches. In-flight batches finish naturally. This prevents over-eviction -- a well-known problem in buffer pool managers where evicting too aggressively causes thrashing and unnecessary I/O. RMM's `failure_callback_resource_adaptor` checks after every retry iteration. | Medium | The dispatch loop iterates over candidates and checks the predicate before dispatching each new batch. A shared atomic counter tracks freed bytes for fast predicate evaluation. In-flight batches complete but no new ones are dispatched. |
| Partial fulfillment semantics | If not enough idle batches exist to satisfy the request, free what is available and return the actual bytes freed. Caller decides whether to retry. This matches RMM's `failure_callback_resource_adaptor` pattern where the callback returns false to signal "give up." | Low | Natural consequence of the predicate API -- when candidates are exhausted, the request completes with whatever was freed. The return value tells the caller exactly what happened. |
| Candidate selection and prioritization | Preserved from current implementation: partitioned repos first, non-active partitions first, last-to-first partition order within each repo. This heuristic minimizes interference with active query execution by preferring to evict data that is not currently being processed. | Low | Existing logic in `run_downgrade_pass` and `collect_candidates_from_partition` is well-tested. Lift into the new design unchanged. The two-pass approach (non-active partitions, then active) is sound. |
| start/stop/drain lifecycle | `SiriusContext` calls `start()`, `stop()`, `drain()`, `get_space_id()`. `drain()` must guarantee no downgrade tasks hold `shared_ptr<data_batch>` references after it returns -- critical for query teardown safety. This is a hard contract. | Medium | New drain must: (1) stop accepting new requests, (2) cancel pending requests in queue, (3) wait for in-flight batch downgrades to complete, (4) clear request queue, (5) resume accepting new requests. Current drain does stop+start which is heavy but correct. |
| Own thread pool (drop itask_executor) | The `itask_executor` base class models a queue-of-tasks; the new design needs a queue-of-requests where each request fans out to multiple concurrent tasks. The inheritance mismatch means fighting `schedule()`, `manager_loop()`, and `_task_queue` abstractions that do not fit. | Medium | Create `bounded_thread_pool` directly in the constructor. Remove inheritance from `itask_executor`. Keep the same thread count config and `cudaSetDevice` per-thread init pattern. Other executors (`pipeline_executor`, `duckdb_scan_executor`) continue using `itask_executor` unmodified. |
| Monitor loop integration | The existing polling loop (`should_downgrade_memory()` every 10ms) is the autonomous memory pressure response path. Must continue to exist, using the new blocking API to trigger downgrade passes instead of directly scheduling tasks. | Low | Monitor thread calls `request_free_memory_and_wait(amount)` instead of directly calling `run_downgrade_pass_all_repos()`. Clean separation between "detect pressure" and "reclaim memory." |
| Thread safety on all public APIs | Multiple threads call into the executor: monitor thread (own thread), allocation-failure callbacks (allocating thread), `SiriusContext` lifecycle methods (query thread). All must be safe without external synchronization by the caller. | Medium | Request queue is inherently thread-safe via `interruptible_mpmc`. Lifecycle methods need a state machine or mutex to prevent concurrent start/stop races. `_running` atomic already provides basic protection. |
| Non-fatal individual batch failures | A single batch downgrade failure (CUDA error, batch state transition race, cuCascade internal error) must not crash the executor or abort the request. Log the error and continue with remaining candidates. | Low | Already implemented in current `manager_loop` with try/catch. Carry forward into dispatch lambda. Failed batches count toward "batches processed" but not "bytes freed." |
| CUDA stream management | The executor creates a non-blocking CUDA stream for downgrade operations and destroys it on stop. Workers share this stream via `rmm::cuda_stream_view`. | Low | Same as current implementation. Stream created in constructor/on_start, destroyed in on_stopped. Single shared stream is sufficient -- profile before adding per-worker streams. |

## Differentiators

Features that improve usability, debuggability, or performance beyond the minimum viable redesign. Not blocking but high value.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Structured completion result type | Instead of returning raw `size_t`, return a `downgrade_result` struct with `bytes_freed`, `batches_dispatched`, `batches_failed`, `predicate_satisfied`, and `elapsed_time`. Richer than a scalar, enables informed retry decisions by callers. | Low | `struct downgrade_result { size_t bytes_freed; size_t batches_dispatched; size_t batches_failed; bool predicate_satisfied; std::chrono::milliseconds elapsed; }`. Future returns this instead of `size_t`. |
| Request-level metrics and logging | Log per-request summary at completion: bytes requested vs freed, batch count, wall time. Essential for debugging memory pressure issues in production. Without metrics, diagnosing "why did the query OOM?" is guesswork. | Low | Use existing spdlog infrastructure. Log at INFO for completed requests, WARN for partial fulfillment, ERROR for zero-byte results. |
| Per-request tracing with request IDs | Correlate log lines across monitor loop, dispatch, and worker threads for a single eviction event. Current implementation generates UUIDs per task but has no request-level grouping. | Low | Generate a monotonic uint64 request ID at submission time. Pass to all log lines within that request's execution. Far cheaper than UUID generation per batch. |
| Cancellation support on pending requests | If a caller submits an async request but the query is torn down before it completes, the request should be cancellable. Prevents wasted work and speeds up drain. | Medium | Add `cancel()` on a lightweight request handle (or use `std::stop_token` pattern). Sets an atomic flag checked alongside the predicate in the dispatch loop. Cancelled requests complete early with partial results. |
| Backpressure / request coalescing | If the monitor loop submits a new request while a previous one is still pending in the queue, coalesce them rather than queuing redundant work. Reduces unnecessary eviction waves under sustained pressure. | Medium | When a new byte-based request arrives and the queue already has a pending byte-based request, merge targets (take the max). Requires inspecting the queue head, which complicates the queue abstraction. Consider as a later optimization. |
| Adaptive polling interval for monitor loop | Fixed 10ms polling is wasteful when idle and potentially too slow under heavy pressure. Adapt the interval based on observed pressure. Common in database checkpoint schedulers. | Low | Exponential backoff: 10ms when pressure detected, double up to 100ms when idle. Reset to 10ms on pressure detection. Simple to implement, meaningful reduction in CPU overhead during idle periods. |
| Freed-bytes fast-path accounting | Track freed bytes locally via atomic counter as workers complete, rather than querying `memory_space->get_available_memory()` after each batch. The memory space query may involve CUDA API calls. Use local counter for fast predicate evaluation, with periodic memory space queries as a consistency check. | Low | Workers atomically add to a `freed_bytes` counter. Byte-based predicate uses this for fast comparison. Predicate-based API callers can still query the memory space directly if they need authoritative numbers. |
| Debug/dry-run mode | Collect candidates and log what would be downgraded without actually performing copies. Invaluable for tuning candidate selection heuristics without risking data movement in production. | Low | Add a `dry_run` flag to the request. Dispatch loop logs candidates but skips `execute()`. Returns projected bytes as if they were freed. |

## Anti-Features

Features to explicitly NOT build. Each would add complexity without proportional value for this internal infrastructure component.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Retry/timeout semantics on requests | Adds state machine complexity (retry count, backoff, timeout threads). The caller is better positioned to decide retry policy -- partial fulfillment already gives the caller the information it needs. Explicitly listed as out of scope in PROJECT.md. | Return partial fulfillment result. Caller retries if needed. Callers can use `std::future::wait_for()` for timeout. |
| Priority queues for requests | Only one request executes at a time and the monitor loop is the primary submitter. Priority adds queue complexity (heap, starvation prevention) with no current consumer. | FIFO queue. Sequential processing. If multiple submitters emerge later, revisit. |
| HOST-to-DISK downgrade | Not implemented in cuCascade yet. Building executor support for a nonexistent tier transition is speculative. Explicitly out of scope in PROJECT.md. | Design the predicate and candidate selection so HOST-to-DISK can be added later (parameterize source/target tier) without redesigning the request API. |
| Pluggable eviction policies (LRU, LFU, ARC) | The current heuristic (partitioned repos first, non-active partitions, last-to-first) works well for the Sirius workload where batch access patterns are pipeline-driven, not random. Abstracting eviction policy adds interface complexity for no demonstrated need. | Keep candidate selection as concrete methods. The logic is already isolated in `collect_candidates_from_partition` and `run_downgrade_pass`, making it easy to modify without a framework. |
| Callback/observer pattern for request completion | Futures already provide the notification mechanism. Adding a separate callback interface creates two ways to observe completion, which is confusing and error-prone. RMM chose callbacks because it operates inside `do_allocate`; we do not have that constraint. | Use `std::future<downgrade_result>`. Callers that need notification use `.get()` or `.wait()`. |
| Cross-executor coordination | Coordinating eviction across multiple memory spaces (e.g., GPU:0 and GPU:1) is a higher-level concern for `SiriusContext`, not the executor. Each executor is bound to one memory space -- keep it that way. | Each executor operates independently on its memory space. `SiriusContext` already creates one executor per memory space and can orchestrate across them if needed. |
| Dynamic thread pool sizing | The thread pool size is set at construction and does not need to change at runtime. GPU-to-HOST copy throughput is limited by PCIe bandwidth, not thread count. A few threads saturate the bus. Adding dynamic sizing adds lifecycle and thread-safety complexity. | Fixed-size thread pool. Configure at construction based on hardware topology. |
| Per-batch CUDA streams | One stream per worker thread for maximum copy parallelism. Adds stream pool management and synchronization complexity. | Single shared stream (current design). Profile first. If stream serialization is measured as a bottleneck, add per-worker streams as a targeted optimization -- not as part of this redesign. |
| Speculative/predictive eviction | Predicting future memory pressure based on query plan analysis. Interesting but far beyond executor scope. | The monitor loop's polling approach is sufficient. Upstream components (planner, reservation manager) can call the request API proactively if they have advance knowledge of upcoming allocations. |
| Async batch cancellation | Cancelling in-flight batches already dispatched to the thread pool. Requires cooperative cancellation inside `data_batch::downgrade()` which is a cuCascade operation we do not control. | Let in-flight batches finish naturally. Early termination only stops NEW dispatches. This is the same pattern used by `bounded_thread_pool::interrupt()`. |

## Feature Dependencies

```
Predicate-driven request completion
  |
  +---> Byte-based convenience wrapper (wraps predicate API)
  |
  +---> Blocking API (uses predicate + future.get() internally)
  |       |
  |       +---> Monitor loop integration (calls blocking API)
  |
  +---> Async API (same mechanism, returns future to caller)
  |       |
  |       +---> Cancellation support (extends async request handle)
  |
  +---> Predicate checked after each batch (evaluation point in dispatch loop)
  |       |
  |       +---> Freed-bytes fast-path accounting (optimizes predicate eval)
  |
  +---> Partial fulfillment (natural consequence of predicate + candidate exhaustion)

Own thread pool (drop itask_executor)
  |
  +---> Sequential request processing (request queue + dispatch loop on own pool)
  |       |
  |       +---> Backpressure / request coalescing (inspects request queue)
  |
  +---> Concurrent batch downgrades (fan-out to workers within a request)
  |       |
  |       +---> CUDA stream management (shared stream passed to workers)
  |
  +---> start/stop/drain lifecycle (own lifecycle, no base class constraints)
          |
          +---> Thread safety on public APIs (lifecycle state machine)

Candidate selection (existing logic, preserved)
  |
  +---> Debug/dry-run mode (logs candidates without executing)

Per-request tracing (request IDs)
  |
  +---> Request-level metrics (metrics tagged with request ID)
  |
  +---> Structured completion result (carries metrics in return type)
```

## MVP Recommendation

Build in this order, driven by dependency chain:

1. **Own thread pool + request queue + sequential dispatch loop** -- Foundation. Drop `itask_executor`, create `bounded_thread_pool` directly, add an `interruptible_mpmc<request>` queue with a dedicated request-processing thread. Without this, nothing else can be built.

2. **Predicate-driven request + concurrent batch execution** -- Core abstraction. Each request takes a `std::function<bool()>` predicate, collects candidates, fans out to workers via `bounded_thread_pool`, checks predicate after each batch completion to enable early exit.

3. **Blocking and async APIs + byte-based wrapper** -- The caller-facing surface. `request_free_memory_and_wait(bytes)` and `request_free_memory(bytes)` returning `std::future`. Byte-based wrapper creates a predicate checking memory consumption.

4. **start/stop/drain + monitor loop** -- Integration with `SiriusContext`. Ensures the new executor is a drop-in replacement from the caller's perspective. Monitor loop calls blocking API.

5. **Candidate selection (preserve existing logic)** -- Lift `run_downgrade_pass`, `collect_candidates_from_partition`, and prioritization heuristics into the new design. These are proven correct.

**Defer to post-MVP (add before merge):**
- Per-request tracing and structured result type: Low complexity, high debuggability value. Add once core is working.
- Request-level metrics: Same -- low cost, add alongside tracing.

**Defer to later milestone:**
- Cancellation support: Add when async API has real callers beyond the monitor loop.
- Backpressure/coalescing: Add if monitor loop double-submission is measured as a problem.
- Adaptive polling interval: Profile first, optimize if idle CPU overhead is significant.
- Debug/dry-run mode: Add during testing/tuning phase if candidate selection needs debugging.

## Sources

- RMM `failure_callback_resource_adaptor` pattern: local header at `.pixi/envs/default/include/rmm/mr/failure_callback_resource_adaptor.hpp` -- retry-on-OOM with callback predicate deciding retry vs give-up (HIGH confidence)
- Current Sirius implementation: `src/downgrade/downgrade_executor.cpp`, `src/include/downgrade/downgrade_executor.hpp`, `src/include/downgrade/downgrade_task.hpp` (HIGH confidence)
- Current `itask_executor` base class: `src/include/parallel/task_executor.hpp` (HIGH confidence)
- Current `bounded_thread_pool`: `src/include/exec/bounded_thread_pool.hpp` (HIGH confidence)
- cuCascade `memory_space` interface: `cucascade/include/cucascade/memory/memory_space.hpp` (HIGH confidence)
- Database buffer pool eviction patterns: [CMU 15-445 Buffer Pool Management (Fall 2025)](https://15445.courses.cs.cmu.edu/fall2025/notes/04-bufferpool.pdf) (MEDIUM confidence)
- Buffer management evolution survey: [Evolution of Buffer Management in Database Systems](https://arxiv.org/html/2512.22995v1) (MEDIUM confidence)
- PROJECT.md requirements and out-of-scope items: `.planning/PROJECT.md` (HIGH confidence)
