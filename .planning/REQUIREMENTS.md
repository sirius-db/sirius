# Requirements: Downgrade Executor Redesign

**Defined:** 2026-04-03
**Core Value:** The downgrade executor must reliably free GPU memory on demand with predictable completion semantics.

## v1 Requirements

### Request API

- [x] **RAPI-01**: The fundamental unit of work is a request that takes a predicate `std::function<bool()>` and downgrades data_batches until the predicate returns true or candidates are exhausted
- [x] **RAPI-02**: `request_free_memory(size_t bytes)` wraps the predicate API with a lambda that checks current memory consumption against the target; returns `std::future<size_t>` (non-blocking)
- [x] **RAPI-03**: `request_free_memory_and_wait(size_t bytes)` blocks until the request completes and returns the number of bytes actually freed
- [x] **RAPI-04**: The predicate-based API also supports async usage, returning `std::future<size_t>` for callers that provide a custom predicate
- [x] **RAPI-05**: If not enough idle batches exist to satisfy the request, the executor frees what is available and returns the actual bytes freed (partial fulfillment)

### Execution Engine

- [x] **EXEC-01**: The downgrade_executor owns its own `bounded_thread_pool` and does not inherit from `itask_executor`
- [x] **EXEC-02**: Requests are queued and executed one at a time (sequential request processing); only one request's batch downgrades are active at any moment
- [x] **EXEC-03**: Within a single request, multiple data_batch downgrades execute concurrently via the thread pool
- [x] **EXEC-04**: The predicate is checked after each individual batch downgrade completes; if true, no new batches are dispatched but in-flight batches finish naturally
- [x] **EXEC-05**: Individual batch downgrade failures are non-fatal — logged and skipped, execution continues with remaining candidates

### Lifecycle & Integration

- [ ] **LIFE-01**: `start()`, `stop()`, and `drain()` methods exist with equivalent behavior to today, compatible with `SiriusContext` usage
- [ ] **LIFE-02**: `drain()` guarantees no downgrade tasks hold `shared_ptr<data_batch>` references after it returns
- [ ] **LIFE-03**: The monitor loop continues to exist, polling `should_downgrade_memory()` and using the blocking API (`request_free_memory_and_wait`) to trigger downgrade passes
- [ ] **LIFE-04**: All public APIs are thread-safe without external synchronization by the caller
- [ ] **LIFE-05**: CUDA stream created on start, destroyed on stop; workers call `cudaSetDevice` on thread init

### Candidate Selection

- [x] **CAND-01**: Existing candidate selection and prioritization logic is preserved: partitioned repos first, non-active partitions first, last-to-first partition order, two-pass (non-active then active)
- [x] **CAND-02**: `collect_candidates_from_partition` and `run_downgrade_pass` selection logic is carried forward (modifiable if needed but functionally equivalent)

### Pipeline Integration

- [x] **PIPE-01**: In `gpu_pipeline_executor`, when `reservation->size() < bytes_needs`, call `request_free_memory_and_wait` on the downgrade_executor for the shortfall bytes, then retry `make_reservation`
- [x] **PIPE-02**: The downgrade-then-retry loop runs up to 5 attempts; if after 5 attempts the full reservation cannot be obtained, proceed with the partial reservation (current behavior)
- [x] **PIPE-03**: The `gpu_pipeline_executor` has access to the `downgrade_executor` for its memory space (passed at construction or available via context)

## v2 Requirements

### Observability

- **OBS-01**: Structured result type (`downgrade_result` struct with bytes_freed, batches_dispatched, batches_failed, predicate_satisfied, elapsed) instead of raw `size_t`
- **OBS-02**: Per-request tracing with monotonic request IDs correlated across all log lines
- **OBS-03**: Request-level metrics logging (bytes requested vs freed, batch count, wall time)

### Advanced Features

- **ADV-01**: Cancellation support on pending async requests via atomic flag checked alongside predicate
- **ADV-02**: Backpressure / request coalescing when monitor loop submits while a request is pending
- **ADV-03**: Adaptive polling interval for monitor loop (exponential backoff: 10ms under pressure, up to 100ms when idle)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Retry/timeout semantics on requests | Caller handles retries; partial fulfillment gives enough info |
| HOST-to-DISK downgrade | Not implemented in cuCascade yet |
| Pluggable eviction policies (LRU, LFU, ARC) | Current heuristic works; logic is isolated and easy to modify later |
| Per-batch CUDA streams | Profile first; single shared stream sufficient until measured as bottleneck |
| Dynamic thread pool sizing | PCIe bandwidth is the limit, not thread count |
| Cross-executor coordination | Higher-level concern for SiriusContext, not the executor |
| Changes to itask_executor base class | Other executors still use it |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| RAPI-01 | Phase 2 | Complete |
| RAPI-02 | Phase 2 | Complete |
| RAPI-03 | Phase 2 | Complete |
| RAPI-04 | Phase 2 | Complete |
| RAPI-05 | Phase 2 | Complete |
| EXEC-01 | Phase 1 | Complete |
| EXEC-02 | Phase 1 | Complete |
| EXEC-03 | Phase 2 | Complete |
| EXEC-04 | Phase 2 | Complete |
| EXEC-05 | Phase 2 | Complete |
| LIFE-01 | Phase 3 | Pending |
| LIFE-02 | Phase 3 | Pending |
| LIFE-03 | Phase 3 | Pending |
| LIFE-04 | Phase 3 | Pending |
| LIFE-05 | Phase 3 | Pending |
| CAND-01 | Phase 1 | Complete |
| CAND-02 | Phase 1 | Complete |
| PIPE-01 | Phase 3 | Complete |
| PIPE-02 | Phase 3 | Complete |
| PIPE-03 | Phase 3 | Complete |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0

---
*Requirements defined: 2026-04-03*
*Last updated: 2026-04-03 after roadmap creation*
