# inspectable_mpsc & Convertible Data

## What This Is

Thread-safe data infrastructure for the Sirius GPU SQL engine. Started as `inspectable_mpsc<T>` — a predicate-inspectable MPSC queue now integrated as the production task queue. Expanding to include `convertible_data` abstractions that enable uniform memory-space conversion across data batches and queued pipeline tasks.

## Core Value

Thread-safe queue with predicate-based element inspection and selective removal (`pop_if`/`get_if`), enabling consumers to find specific items without draining the queue. Extended by convertible_data interfaces that provide uniform, failure-safe data conversion across memory tiers.

## Current Milestone: v2.0 Convertible Data Abstraction

**Goal:** Create abstract interfaces and concrete implementations for memory-space-aware data conversion, enabling uniform conversion of data batches and queued pipeline tasks.

**Target features:**
- `convertible_data` abstract interface — uniform `convert()` + `bytes_in_space()` API
- `convertible_data_provider` abstract interface — search/iterate convertible items by memory space
- `convertible_data_batch` + `convertible_data_batch_provider` — wrap `data_batch` / `data_repository`
- `convertible_gpu_pipeline_task` + `convertible_gpu_pipeline_task_provider` — wrap `gpu_pipeline_task` / `inspectable_mpsc<itask>`
- Extend `data_batch` state machine: `task_created → in_transit` transition
- Failure safety: all conversions restore original `batch_state` and `idata_representation` on failure

## Requirements

### Validated

Validated in Phase 1: Core Queue
- [x] Class `inspectable_mpsc<T>` as a header-only template in `sirius::exec` namespace
- [x] Internal container: `std::deque<std::unique_ptr<T>>` guarded by `std::mutex` + `std::condition_variable`
- [x] `bool push(std::unique_ptr<T> item)` — enqueue an item; returns false if interrupted
- [x] `bool emplace(Args&&... args)` — construct and enqueue in-place
- [x] `std::unique_ptr<T> pop()` — blocking dequeue using condition_variable wait
- [x] `std::unique_ptr<T> try_pop()` — non-blocking dequeue, returns nullptr if empty
- [x] `void interrupt()` / `void reactivate()` — shutdown and restart semantics matching interruptible_mpmc
- [x] `void drain()` — remove all queued items
- [x] `bool is_open() const noexcept` / `bool is_empty() const noexcept` — state queries
- [x] Thread-safe for MPMC access (designed for MPSC but safe under MPMC)
- [x] Delete copy/move constructors and assignment operators

Validated in Phase 2: Predicate Inspection
- [x] `std::unique_ptr<T> pop_if(std::function<bool(const T&)> predicate, bool front_to_back)` — remove and return first element matching predicate
- [x] `T* get_if(std::function<bool(const T&)> predicate, bool front_to_back)` — return pointer to first matching element without removing
- [x] `std::unique_ptr<T> mutable_pop_if(std::function<bool(T&)> predicate, bool front_to_back)` — pop_if with mutable access in predicate
- [x] `T* mutable_get_if(std::function<bool(T&)> predicate, bool front_to_back)` — get_if with mutable access in predicate

Validated in Phase 3: Dead Code Removal — v1.1
- [x] Verify and remove dead queue code (gpu_pipeline_queue, pipeline_queue, duckdb_scan_task_queue, itask_queue)

Validated in Phase 4: Queue Integration — v1.1
- [x] Replace interruptible_mpmc with inspectable_mpsc in itask_executor and its implementations

### Active

- [ ] `convertible_data` abstract interface with `convert()` and `bytes_in_space()`
- [ ] `convertible_data_provider` abstract interface with `get_next_convertible()`, `get_all_convertible()`, `get_bytes_in_space()`
- [ ] `convertible_data_batch` wrapping `data_batch` with downgrade-style conversion
- [ ] `convertible_data_batch_provider` wrapping `data_repository`
- [ ] `convertible_gpu_pipeline_task` wrapping `gpu_pipeline_task` with RAII queue ownership
- [ ] `convertible_gpu_pipeline_task_provider` wrapping `inspectable_mpsc<itask>`
- [ ] Extend `data_batch` state machine: `task_created → in_transit` transition
- [ ] Failure safety: conversions restore original `batch_state` and `idata_representation` on failure

### Out of Scope

- Lock-free implementation — not needed; mutex+cv is appropriate for the inspection/iteration requirements
- Shared mutex (reader-writer lock) — overhead not justified for MPSC use case where most operations are writes
- Linked-list backing — worse cache locality during iteration outweighs O(1) mid-erase benefit

## Current State

**v2.0 Convertible Data Abstraction** — STARTED 2026-04-15

Building memory-space-aware data conversion interfaces on top of the `inspectable_mpsc<T>` foundation shipped in v1.1.

## Context

Shipped v1.0 (2026-04-14) with 1,153 LOC C++ (295 header + 858 tests). 3 plans, ~69 min.
Shipped v1.1 (2026-04-14) with 2 plans, ~32 min. Removed 450 lines dead code, swapped queue type in itask_executor.
Tech stack: C++20, header-only template, Catch2 test framework.
35 inspectable_mpsc tests passing (231 assertions). Full Sirius suite: 868 tests, 78M+ assertions.
`inspectable_mpsc<itask>` is the production task queue in `itask_executor`, inherited by `gpu_pipeline_executor` and `duckdb_scan_executor`.

- Sirius is a GPU-native SQL engine that extends DuckDB
- `interruptible_mpmc` remains in use for non-itask_executor queues (pipeline_executor, downgrade_executor, task_creator)
- Future work can leverage `pop_if`/`get_if` for predicate-based task scheduling in pipeline executors

## Constraints

- **Tech stack**: C++20, CUDA-compatible, must compile within Sirius build system
- **Pattern**: Header-only template, same style as `interruptible_mpmc.hpp`
- **Location**: `src/include/exec/inspectable_mpsc.hpp`
- **Namespace**: `sirius::exec`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| `std::deque` over `std::list` | Better cache locality for iteration; mid-erase O(n) is acceptable | Validated Phase 1 |
| `std::mutex` + `std::condition_variable` over `std::shared_mutex` | MPSC workload is write-heavy; shared_mutex overhead not justified | Validated Phase 1 |
| `std::unique_ptr<T>` ownership | Class owns elements exclusively; matches intended MPSC semantics | Validated Phase 1 |
| `std::function` for predicate params | Flexibility over template predicates; accepted overhead for internal use | Validated Phase 2 |
| `std::next(rit).base()` for reverse erase | Standard idiom for converting reverse iterator to forward for deque::erase | Validated Phase 2 |
| Mutex held for full predicate scan | Simple correctness over per-element locking; acceptable for MPSC | Validated Phase 2 |
| Raw `T*` return from `get_if` | Avoids ownership transfer; documented invalidation rules; safe under MPSC | Validated Phase 2 |
| Dead code removal before integration | Simplifies codebase before swapping queue types; reduces merge surface | Validated Phase 3 |
| `static_cast<void>` for `[[nodiscard]]` discard | Standard C++ idiom; `schedule()` is fire-and-forget, matching prior semantics | Validated Phase 4 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-15 after v2.0 milestone start*
