# Milestones

## v2.0 Convertible Data Abstraction (Shipped: 2026-04-16)

**Phases completed:** 3 phases, 6 plans, 9 tasks

**Key accomplishments:**

- Formalized data_batch state machine to document task_created<->in_transit transitions with task_created_count preservation tests
- convertible_data and convertible_data_provider abstract interfaces with pure virtual convert/inspect contracts for uniform memory-tier conversion
- Failure-safe convertible_data_batch wrapping data_batch with per-tier conversion and repository-based batch discovery by memory space
- 8 Catch2 GPU integration tests validating GPU-to-HOST conversion, provider discovery with multi-partition iteration, failure safety, and bytes_in_space accuracy using real cuCascade data batches and converter registry
- RAII convertible_gpu_pipeline_task wrapper and mutable_pop_if-based provider for inspectable_mpsc queue task discovery and memory-tier conversion
- 11 Catch2 GPU integration tests validating RAII queue return, predicate filtering, GPU-to-HOST conversion, bytes accounting, and interrupted-queue safety for convertible_gpu_pipeline_task

---

## v1.1 Task Queue Refactor (Shipped: 2026-04-14)

**Phases completed:** 2 phases, 2 plans, 4 tasks

**Key accomplishments:**

- Removed 4 legacy queue classes (gpu_pipeline_queue, pipeline_queue, duckdb_scan_task_queue, itask_queue) -- 6 files deleted, 450 lines removed, zero regressions
- Replaced interruptible_mpmc with inspectable_mpsc in itask_executor base class -- all 868 tests pass with zero regressions

---

## v1.0 MVP (Shipped: 2026-04-14)

**Phases completed:** 2 phases, 3 plans, 5 tasks

**Key accomplishments:**

- Header-only inspectable_mpsc<T> template with mutex+cv blocking, full push/pop/emplace/interrupt/drain API, and 14 Catch2 single-threaded unit tests passing
- 4 multi-threaded stress tests proving thread-safe MPSC operation under 4-producer/1-consumer contention with no data loss, correct blocking, and clean interrupt
- Four predicate-based inspection methods (pop_if, get_if, mutable_pop_if, mutable_get_if) with bidirectional search, completing the inspectable_mpsc class's core value proposition

---
