# Roadmap: Downgrade Executor Redesign

## Overview

This roadmap transforms the downgrade_executor from an `itask_executor`-derived task scheduler into a purpose-built request-queue executor. Phase 1 establishes the structural skeleton (own thread pool, request queue, candidate selection wired in). Phase 2 implements the core capability: predicate-driven request execution with concurrent batch downgrades and the full public API surface. Phase 3 wires it into the system: lifecycle management for SiriusContext, monitor loop migration, and pipeline executor integration for on-demand memory reclamation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation** - Own thread pool, request queue, sequential processing, and candidate selection
- [ ] **Phase 2: Request Execution and API** - Predicate-driven execution engine with concurrent batches and full public API
- [ ] **Phase 3: Lifecycle and Pipeline Integration** - start/stop/drain semantics, monitor loop, and pipeline executor wiring

## Phase Details

### Phase 1: Foundation
**Goal**: The downgrade_executor compiles and runs with its own thread pool and request queue, completely decoupled from itask_executor
**Depends on**: Nothing (first phase)
**Requirements**: EXEC-01, EXEC-02, CAND-01, CAND-02
**Success Criteria** (what must be TRUE):
  1. downgrade_executor no longer inherits from itask_executor and owns a bounded_thread_pool directly
  2. A downgrade_request struct exists and requests enqueue into a mutex-protected FIFO queue consumed by a dedicated processing thread
  3. Candidate collection (collect_candidates_from_partition, two-pass prioritization) is wired into the request processing path and produces the same candidate ordering as the current implementation
  4. Only one request's processing is active at any time (sequential request execution)
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Rewrite downgrade_task as plain struct and downgrade_executor as standalone class
- [ ] 01-02-PLAN.md — Update tests for new types, build and verify

### Phase 2: Request Execution and API
**Goal**: Callers can submit predicate-based and byte-based downgrade requests and receive results via std::future or blocking call
**Depends on**: Phase 1
**Requirements**: RAPI-01, RAPI-02, RAPI-03, RAPI-04, RAPI-05, EXEC-03, EXEC-04, EXEC-05
**Success Criteria** (what must be TRUE):
  1. A predicate-based request dispatches concurrent batch downgrades and stops dispatching new batches as soon as the predicate returns true (in-flight batches finish naturally)
  2. request_free_memory(bytes) returns a std::future<size_t> that resolves to actual bytes freed, and request_free_memory_and_wait(bytes) blocks and returns actual bytes freed
  3. When fewer idle batches exist than needed to satisfy a request, the executor frees what is available and the returned byte count reflects partial fulfillment
  4. Individual batch downgrade failures are logged and skipped without crashing the executor or aborting the request
  5. Multiple data_batch downgrades within a single request execute concurrently via the thread pool
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: Lifecycle and Pipeline Integration
**Goal**: The redesigned executor is a drop-in replacement: SiriusContext manages it via start/stop/drain, the monitor loop uses it, and gpu_pipeline_executor reclaims memory through it
**Depends on**: Phase 2
**Requirements**: LIFE-01, LIFE-02, LIFE-03, LIFE-04, LIFE-05, PIPE-01, PIPE-02, PIPE-03
**Success Criteria** (what must be TRUE):
  1. start(), stop(), and drain() work correctly with SiriusContext: drain() guarantees no shared_ptr<data_batch> references remain, and stop() fulfills all queued promises with 0
  2. The monitor loop polls should_downgrade_memory() and triggers downgrade passes via request_free_memory_and_wait (blocking API), replacing direct task scheduling
  3. gpu_pipeline_executor calls request_free_memory_and_wait when a reservation falls short, retries up to 5 times, then proceeds with partial reservation
  4. All public APIs are safe to call concurrently from monitor thread, query threads, and allocation-failure callbacks without external synchronization
  5. CUDA stream is created on start and destroyed on stop; worker threads call cudaSetDevice on init
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/2 | Not started | - |
| 2. Request Execution and API | 0/2 | Not started | - |
| 3. Lifecycle and Pipeline Integration | 0/2 | Not started | - |
