# Project Research Summary

**Project:** Downgrade Executor Redesign
**Domain:** Concurrent task executor for GPU memory reclamation
**Researched:** 2026-04-03
**Confidence:** HIGH

## Executive Summary

The downgrade_executor redesign is a well-scoped internal infrastructure refactor: replace an ill-fitting `itask_executor` inheritance model with a purpose-built request-queue executor that matches the actual access pattern (queue-of-requests, each fanning out to concurrent batch downgrades). The central abstraction shift is from "schedule N tasks" to "free memory until a predicate is satisfied," which aligns with how every mature memory reclamation system (RMM's `failure_callback_resource_adaptor`, database buffer pool evictors) operates. No new external dependencies are needed — C++20 standard library primitives (`std::promise`/`std::future`, `std::atomic`, `std::mutex` + `std::condition_variable`) combined with the project's existing `bounded_thread_pool` and `absl::AnyInvocable` cover the entire design.

The recommended architecture is three threads and one shared pool: a monitor thread that detects memory pressure and submits blocking requests, a dedicated request-processing thread that serializes request execution and fans out batch downgrades to the pool, and N worker threads in `bounded_thread_pool` that perform GPU-to-host copies. Sequential request processing eliminates inter-request contention on candidate batches. Per-request coordination uses an `std::atomic<int>` counter (increment before dispatch, decrement on completion, CV notification when zero) rather than `std::latch` — this handles early termination cleanly without requiring the dispatch count to be known upfront.

The primary risks are concurrency correctness issues: dangling lambda captures if the wait logic has a bug, deadlock in `drain()` if pool interruption and thread join are improperly ordered, and `std::promise` set-twice or never-set if exception paths are not guarded. All three are preventable with RAII scope guards on the promise, correct ordering in `drain()`, and TSAN testing. The existing candidate selection logic (`collect_candidates_from_partition`, two-pass prioritization) should be lifted unchanged — it is proven correct and well-isolated.

## Key Findings

### Recommended Stack

The redesign requires zero new dependencies. All primitives are already available: C++20 standard library (`<future>`, `<atomic>`, `<mutex>`, `<condition_variable>`, `<latch>`, `<deque>`, `<thread>`), the project's own `bounded_thread_pool`, and `absl::AnyInvocable`. The key decision is to drop `itask_executor` inheritance entirely — the base class models a flat queue of independent tasks, which fights the request-fan-out model needed here. The `interruptible_mpmc` task queue, `itask`/`downgrade_task` class hierarchy, and `task_completion_message_queue` are all dropped for the same reason.

**Core technologies:**
- `std::promise<size_t>` / `std::future<size_t>`: Async result delivery — standard, zero-dependency, provides both blocking `.get()` and polling `.wait_for()` to callers for free
- `std::atomic<int>` + `std::condition_variable`: Per-request batch completion tracking — supports early termination (unknown dispatch count) unlike `std::latch`, which requires count upfront
- `std::mutex` + `std::deque<downgrade_request>`: Request queue — correct for low-rate (~1/10ms) single-consumer multi-producer access; lock-free queues add complexity with zero measurable benefit here
- `bounded_thread_pool` (existing): Concurrent batch execution — reuse as-is via `reserve()` + `dispatch()`; do NOT use `wait_all()` for per-request synchronization
- `absl::AnyInvocable<bool() const>`: Predicate type — supports move-only captures unlike `std::function`; consistent with codebase convention

### Expected Features

**Must have (table stakes):**
- Predicate-driven request completion — core abstraction; callers pass `() -> bool` stopping condition
- Byte-based convenience wrapper — thin layer over predicate API; most callers know bytes, not predicates
- Blocking API (`request_free_memory_and_wait`) — critical allocation path callers need synchronous eviction
- Async API (`request_free_memory` returning `std::future`) — monitor loop and fire-and-forget callers
- Sequential request processing — prevents inter-request candidate contention
- Concurrent batch downgrades within a request — performance critical; PCIe bandwidth requires parallel transfers
- Predicate checked after each batch completion — enables early exit, prevents over-eviction
- Partial fulfillment semantics — natural consequence; caller decides whether to retry
- Candidate selection (existing logic preserved) — two-pass prioritization is proven correct
- `start()` / `stop()` / `drain()` lifecycle — hard contract with `SiriusContext`; `drain()` must guarantee no `shared_ptr<data_batch>` references remain
- Drop `itask_executor` inheritance — own a `bounded_thread_pool` directly
- Monitor loop integration — existing polling loop calls blocking API instead of directly scheduling tasks
- Thread safety on all public APIs — monitor thread, allocation-failure callbacks, and query threads call concurrently

**Should have (add before merge):**
- Structured completion result (`downgrade_result` struct: `bytes_freed`, `batches_dispatched`, `batches_failed`, `predicate_satisfied`, `elapsed`) — enables informed retry decisions
- Request-level metrics and logging — essential for diagnosing production OOM events
- Per-request monotonic request IDs — correlates log lines across threads for a single eviction event

**Defer (v2+):**
- Cancellation support on pending requests — add when async API has real callers beyond monitor loop
- Backpressure / request coalescing — add only if monitor loop double-submission is measured as a problem
- Adaptive polling interval — profile idle CPU overhead first
- Debug/dry-run mode — add during tuning phase if candidate selection needs debugging
- Per-batch CUDA streams — profile single-stream serialization first

**Explicit anti-features (do not build):**
- Retry/timeout semantics on requests, priority queues, HOST-to-DISK downgrade, pluggable eviction policies, cross-executor coordination, dynamic thread pool sizing, callback/observer pattern for completion

### Architecture Approach

The redesigned executor is a self-contained three-thread, one-pool component. External callers push `downgrade_request` objects (predicate + promise) onto a mutex-protected deque. A dedicated request-processing thread dequeues one request at a time, collects candidates using existing repo-walking logic, iterates candidates with a predicate check before each dispatch, fans out to `bounded_thread_pool` workers, then waits on a per-request atomic counter + CV for all dispatched batches to complete before setting the promise. A monitor thread runs independently, polling memory pressure every 10ms and calling the blocking API when a downgrade pass is warranted.

**Major components:**
1. `downgrade_executor` (public API) — accepts requests, manages all thread lifecycle, exposes `start()`/`stop()`/`drain()`
2. Request queue (`std::mutex` + `std::deque<downgrade_request>` + `std::condition_variable`) — decouples producers from processing rate; FIFO
3. Request-processing thread — sequential consumer; collects candidates, dispatches batches with predicate-gated early exit, waits for per-request completion, fulfills promise
4. `bounded_thread_pool` (reused) — concurrent batch execution; workers call `batch->downgrade(stream)` and update shared `bytes_freed` atomic
5. Monitor thread — autonomous pressure detection; calls blocking request API; acceptable to block while downgrade is in progress
6. Candidate selection (existing logic) — `collect_candidates_from_partition` and two-pass prioritization lifted unchanged

### Critical Pitfalls

1. **Dangling captures in dispatch lambdas** — `process_request()` must block until ALL dispatched lambdas complete. Increment the `remaining` counter BEFORE calling `dispatch()`. The `done_cv.wait()` predicate must re-check `remaining == 0` on every wakeup. Verify with TSAN; add `assert(remaining.load() == 0)` after the wait.

2. **Deadlock between `drain()` and `request_loop()`** — set `_running = false`, call `_pool->interrupt()`, and notify `_request_cv` before attempting to join the request-processing thread. If the thread is blocked in `reserve()` when `interrupt()` fires, `reserve()` returns an invalid slot and the loop exits cleanly. Test drain under active load.

3. **Promise set twice or never** — a single `set_value()` call at the end of `process_request()`, after the wait block. Wrap the promise in an RAII scope guard that calls `set_value(0)` if the function exits abnormally. Unit test: submit a request with zero candidates; verify future resolves with 0.

4. **CUDA device not set on worker threads** — `per_thread_init` lambda (calling `cudaSetDevice`) MUST be passed to the `bounded_thread_pool` constructor. Add a debug assertion in each dispatch lambda: `int dev; cudaGetDevice(&dev); assert(dev == expected_device)`.

5. **Request queue not drained on `stop()`** — after joining threads, iterate the queue and call `set_value(0)` on every remaining promise; otherwise callers block on `future.get()` forever.

## Implications for Roadmap

Based on research, the feature dependency chain maps cleanly to a 4-phase implementation order.

### Phase 1: Foundation — Own Thread Pool + Request Queue

**Rationale:** Everything else depends on this. Drop `itask_executor` inheritance and establish the new structural skeleton before touching any logic.
**Delivers:** A compilable `downgrade_executor` that owns a `bounded_thread_pool` directly, has a mutex-protected `deque<downgrade_request>`, and runs a request-processing thread that can dequeue but does not yet process requests meaningfully.
**Addresses:** "Own thread pool (drop itask_executor)" table-stakes feature; "Sequential request processing" structural requirement; `downgrade_request` struct definition.
**Avoids:** Fighting the `itask_executor` abstraction throughout later phases — remove it first to get a clean slate.

### Phase 2: Core — Predicate-Driven Request Execution

**Rationale:** This is the central new capability. Once the queue infrastructure exists, implement the request processing logic: candidate collection, predicate-gated dispatch loop, per-request atomic counter + CV completion tracking, promise fulfillment.
**Delivers:** A working `process_request()` that executes a single downgrade request end-to-end: collects candidates, fans out to workers, waits for completion, fulfills the promise.
**Addresses:** Predicate-driven request completion, concurrent batch downgrades, predicate checked after each batch, early termination, partial fulfillment semantics, candidate selection (existing logic lifted).
**Avoids:** Dangling captures (Critical Pitfall 1) — implement the atomic counter pattern from the start, not `std::latch`. Latch-vs-counter off-by-one (Moderate Pitfall 5) — use atomic counter for unknown dispatch count.
**Uses:** `bounded_thread_pool::reserve()` + `dispatch()`, `absl::AnyInvocable<bool() const>`, `std::atomic<int>` + CV, existing `collect_candidates_from_partition`.

### Phase 3: API Surface + Lifecycle Integration

**Rationale:** Expose the complete caller-facing API and wire up `SiriusContext` integration, making the new executor a genuine drop-in replacement.
**Delivers:** Blocking and async public APIs with byte-based convenience wrappers, correct `start()`/`stop()`/`drain()` semantics (drain must guarantee no `data_batch` references remain), monitor loop updated to call blocking API.
**Addresses:** Blocking API, async API, byte-based wrapper, `start/stop/drain` lifecycle, monitor loop integration, thread safety on all public APIs, CUDA stream management.
**Avoids:** Deadlock in `drain()` (Critical Pitfall 2) — implement pool interrupt + CV notify + join ordering correctly. Request queue not drained on `stop()` (Critical Pitfall 5) — add promise cleanup explicitly. Monitor loop blocking semantics (Moderate Pitfall 2) — document that blocking during downgrade is acceptable.
**Implements:** Full lifecycle management pattern from ARCHITECTURE.md.

### Phase 4: Observability + Cleanup

**Rationale:** Low-complexity, high-value additions before merge, then final cleanup of old abstractions.
**Delivers:** `downgrade_result` struct replacing raw `size_t`, per-request monotonic IDs, INFO-level per-request completion log; removal of `task_completion_message_queue`, `itask`/`downgrade_task` class hierarchy.
**Addresses:** Structured completion result, request-level metrics and logging, per-request tracing; clean separation from old abstractions.
**Avoids:** Logging overhead in hot path (Minor Pitfall 2) — per-batch logging stays at TRACE; only summary at INFO.

### Phase Ordering Rationale

- Phases 1 and 2 are strictly sequenced by dependency: the queue must exist before request processing can be built.
- Phase 3 is separate from Phase 2 because lifecycle correctness (especially `drain()`) is a distinct concern from execution correctness and introduces its own failure modes (deadlock, promise leaks). Isolating it makes each phase's test surface smaller.
- Phase 4 is last because observability features are lower risk but add noise to PR reviews if included during the correctness-critical phases, and cleanup of old code should happen after the replacement is proven.
- Features explicitly deferred to later milestones (cancellation, coalescing, adaptive polling, dry-run, per-batch streams) should not block merge — they require callers or measurement data that will not exist until post-integration.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (`drain()` semantics):** The exact behavior of `bounded_thread_pool::interrupt()`, `wait_all()`, and `resume()` needs verification against the actual implementation before coding `drain()`. The current `drain()` does `stop()` + `start()` (heavy but correct) — confirming that the lighter-weight interrupt/resume cycle provides the same guarantee is necessary before relying on it.
- **Phase 2 (CUDA stream thread safety under load):** Single-shared-stream with multiple concurrent workers is correct per CUDA semantics but performance implications are workload-dependent. Flag for measurement in Phase 2 integration testing.

Phases with standard patterns (skip additional research):
- **Phase 1 (queue infrastructure):** Mutex + CV + deque is textbook; no research needed.
- **Phase 4 (observability):** spdlog and monotonic counters are already used throughout the codebase; pattern is established.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All primitives verified by direct source inspection: `std::latch` used in `bounded_thread_pool.hpp`, `absl::AnyInvocable` used throughout, C++20 confirmed in build config |
| Features | HIGH | Table stakes derived from direct reading of PROJECT.md requirements, current implementation, and RMM `failure_callback_resource_adaptor` header; out-of-scope items explicitly confirmed in PROJECT.md |
| Architecture | HIGH | Component boundaries and data flow derived from existing source; all patterns verified against `bounded_thread_pool` API and current `downgrade_executor` implementation |
| Pitfalls | HIGH | Critical pitfalls derived from concrete code paths in the proposed design; prevention strategies are testable and specific |

**Overall confidence:** HIGH

### Gaps to Address

- **CUDA single-stream performance:** Whether a single shared stream across N worker threads becomes a bottleneck depends on workload (batch sizes, PCIe saturation). Current design is correct; measure during Phase 2 integration testing. Per-worker streams are the mitigation if needed but should not be pre-built.
- **`bounded_thread_pool::interrupt()` / `resume()` semantics:** The drain redesign (reuse pool without recreating it) depends on `resume()` correctly re-enabling `reserve()` after `interrupt()`. Verify against the pool implementation before implementing Phase 3 `drain()`.
- **Memory accounting synchrony:** Whether `memory_space->get_current_usage()` reflects freed GPU memory synchronously after `batch->downgrade()` returns determines whether predicate-based callers get accurate readings. Verify during Phase 2 testing with a byte-predicate request.

## Sources

### Primary (HIGH confidence)
- `src/include/exec/bounded_thread_pool.hpp` — pool API, `std::latch` usage, `reserve()`/`dispatch()` pattern
- `src/downgrade/downgrade_executor.cpp`, `src/include/downgrade/downgrade_executor.hpp` — current implementation, lifecycle, CUDA stream handling
- `src/include/downgrade/downgrade_task.hpp` — `itask`/`downgrade_task` class hierarchy being dropped
- `src/include/parallel/task_executor.hpp` — `itask_executor` base class being dropped
- `.pixi/envs/default/include/rmm/mr/failure_callback_resource_adaptor.hpp` — predicate/retry pattern for memory reclamation
- `cucascade/include/cucascade/memory/memory_space.hpp` — memory space interface
- `.planning/PROJECT.md` — requirements and explicit out-of-scope items
- C++20 standard: `std::promise`, `std::future`, `std::atomic`, `std::latch`, `std::condition_variable`

### Secondary (MEDIUM confidence)
- CMU 15-445 Buffer Pool Management notes — buffer pool eviction patterns (predicate-driven stopping, serialized eviction decisions)
- Evolution of Buffer Management in Database Systems (arXiv:2512.22995) — historical context for eviction policy design
- CUDA Programming Guide Section 3.2.8 — concurrent stream operations from multiple host threads

---
*Research completed: 2026-04-03*
*Ready for roadmap: yes*
