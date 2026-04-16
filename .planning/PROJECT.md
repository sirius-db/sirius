# inspectable_mpsc & Convertible Data

## What This Is

Thread-safe data infrastructure for the Sirius GPU SQL engine. Includes `inspectable_mpsc<T>` — a predicate-inspectable MPSC queue integrated as the production task queue — and `convertible_data` abstractions that enable uniform, failure-safe memory-space conversion across data batches and queued pipeline tasks.

## Core Value

Thread-safe queue with predicate-based element inspection and selective removal (`pop_if`/`get_if`), enabling consumers to find specific items without draining the queue. Complemented by convertible_data interfaces providing uniform, failure-safe data conversion across GPU/HOST/DISK memory tiers.

## Shipped Milestones

- **v1.0 MVP** — inspectable_mpsc core queue + predicate inspection (2026-04-14)
- **v1.1 Task Queue Refactor** — dead code removal + production queue integration (2026-04-14)
- **v2.0 Convertible Data Abstraction** — abstract interfaces, batch conversion, task queue conversion (2026-04-16)

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

Validated in Phase 5: State Machine & Interfaces — v2.0
- ✓ `convertible_data` abstract interface with `convert()` and `bytes_in_space()`
- ✓ `convertible_data_provider` abstract interface with `get_next_convertible()`, `get_all_convertible()`, `get_bytes_in_space()`
- ✓ Extend `data_batch` state machine: `task_created → in_transit` transition

Validated in Phase 6: Batch Conversion — v2.0
- ✓ `convertible_data_batch` wrapping `data_batch` with downgrade-style conversion
- ✓ `convertible_data_batch_provider` wrapping `data_repository`
- ✓ Failure safety: batch conversions restore original `batch_state` and `idata_representation` on failure

Validated in Phase 7: Task Queue Conversion — v2.0
- ✓ `convertible_gpu_pipeline_task` wrapping `gpu_pipeline_task` with RAII queue ownership
- ✓ `convertible_gpu_pipeline_task_provider` wrapping `inspectable_mpsc<itask>`
- ✓ Failure safety: task conversions restore original state; task always returned to queue via RAII

Validated in Phase 8: API Cleanup + Processing Loop Refactor — v3.0
- ✓ Removed `target_bytes` from `downgrade_request` struct and `request_downgrade` API
- ✓ Rewrote `processing_loop` with tiered `convertible_data_batch_provider` (repos → gpu queue → pipeline queue)
- ✓ Eliminated `downgrade_task` struct — all conversion via `convertible_data::convert()`
- ✓ Per-tier breakdown logging (repos/gpu_queue/pipeline_queue batches and bytes)
- ✓ Predicate checked both in dispatch loop and after each convert() in workers

### Active

#### Current Milestone: v3.0 Downgrade Executor Integration

**Remaining:**
- Explore refactoring `lock_or_prepare_batch` in `batch_lock_utils.hpp` to use `convertible_data_batch::convert()`

### Out of Scope

- Lock-free implementation — not needed; mutex+cv is appropriate for the inspection/iteration requirements
- Shared mutex (reader-writer lock) — overhead not justified for MPSC use case where most operations are writes
- Linked-list backing — worse cache locality during iteration outweighs O(1) mid-erase benefit

## Current State

**v3.0 Downgrade Executor Integration** — Phase 8 complete, Phase 9 remaining

Previous milestones shipped (v1.0, v1.1, v2.0). 8 phases, 13 plans executed.
Phase 8 completed: removed target_bytes from API, rewrote processing loop with tiered providers, eliminated downgrade_task. Phase 9 (batch_lock exploration) remaining.

## Context

Shipped v1.0 (2026-04-14) with 1,153 LOC C++ (295 header + 858 tests). 3 plans, ~69 min.
Shipped v1.1 (2026-04-14) with 2 plans, ~32 min. Removed 450 lines dead code, swapped queue type in itask_executor.
Shipped v2.0 (2026-04-16) with 6 plans, ~65 min. Added 1,499 LOC across 6 files: abstract interfaces, batch conversion, task queue conversion, and 19 GPU integration tests.
Tech stack: C++20, header-only templates, Catch2 test framework.
54 data infrastructure tests passing (297 assertions). Full Sirius suite: 868+ tests, 78M+ assertions.
`inspectable_mpsc<itask>` is the production task queue in `itask_executor`, inherited by `gpu_pipeline_executor` and `duckdb_scan_executor`.

- Sirius is a GPU-native SQL engine that extends DuckDB
- `interruptible_mpmc` remains in use for non-itask_executor queues (pipeline_executor, downgrade_executor, task_creator)
- `convertible_data` abstractions ready for integration into downgrade executor and memory pressure handling
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
| Documentation-only state machine change | Code already handles task_created↔in_transit; formalize with docs+tests | ✓ Good — Phase 5 |
| Both interfaces in single header | Provider depends on convertible_data; co-location avoids circular includes | ✓ Good — Phase 5 |
| Forward declarations for cucascade types | Minimize header dependencies; memory_space and reservation_manager forward-declared | ✓ Good — Phase 5 |
| `memory_space*` for all memory space params | Non-copyable type; pointer semantics throughout | ✓ Good — Phase 6 |
| Converter registry via singleton | Internal access pattern consistent with existing downgrade_task usage | ✓ Good — Phase 6 |
| `get_bytes_in_space` returns 0 on provider | inspectable_mpsc lacks const iteration; callers use get_all + bytes_in_space | ✓ Good — Phase 7 |
| RAII destructor pushes task back to queue | Guarantees task is never lost even on exception; follows unique_ptr ownership | ✓ Good — Phase 7 |
| Lightweight dynamic_cast predicate | No I/O or allocation in mutable_pop_if predicate; satisfies mpsc contract | ✓ Good — Phase 7 |

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
*Last updated: 2026-04-16 after v3.0 milestone started*
