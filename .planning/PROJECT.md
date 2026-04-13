# inspectable_mpsc

## What This Is

A new thread-safe queue class (`inspectable_mpsc<T>`) for the Sirius GPU SQL engine that supports multiple producers and a single consumer, with the ability to inspect and selectively remove elements by predicate. It lives alongside the existing `interruptible_mpmc` in `sirius::exec` and uses `std::unique_ptr<T>` ownership semantics.

## Core Value

Thread-safe queue with predicate-based element inspection and selective removal (`pop_if`/`get_if`), enabling consumers to find specific items without draining the queue.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Class `inspectable_mpsc<T>` as a header-only template in `sirius::exec` namespace
- [ ] Internal container: `std::deque<std::unique_ptr<T>>` guarded by `std::mutex` + `std::condition_variable`
- [ ] `bool push(std::unique_ptr<T> item)` — enqueue an item; returns false if interrupted
- [ ] `bool emplace(Args&&... args)` — construct and enqueue in-place
- [ ] `std::unique_ptr<T> pop()` — blocking dequeue using condition_variable wait
- [ ] `std::unique_ptr<T> try_pop()` — non-blocking dequeue, returns nullptr if empty
- [ ] `std::unique_ptr<T> pop_if(std::function<bool(const T&)> predicate, bool front_to_back)` — remove and return first element matching predicate
- [ ] `T* get_if(std::function<bool(const T&)> predicate, bool front_to_back)` — return pointer to first matching element without removing
- [ ] `std::unique_ptr<T> mutable_pop_if(std::function<bool(T&)> predicate, bool front_to_back)` — pop_if with mutable access in predicate
- [ ] `T* mutable_get_if(std::function<bool(T&)> predicate, bool front_to_back)` — get_if with mutable access in predicate
- [ ] `void interrupt()` / `void reactivate()` — shutdown and restart semantics matching interruptible_mpmc
- [ ] `void drain()` — remove all queued items
- [ ] `bool is_open() const noexcept` / `bool is_empty() const noexcept` — state queries
- [ ] Thread-safe for MPMC access (designed for MPSC but safe under MPMC)
- [ ] Delete copy/move constructors and assignment operators

### Out of Scope

- Lock-free implementation — not needed; mutex+cv is appropriate for the inspection/iteration requirements
- Shared mutex (reader-writer lock) — overhead not justified for MPSC use case where most operations are writes
- Linked-list backing — worse cache locality during iteration outweighs O(1) mid-erase benefit

## Context

- Sirius is a GPU-native SQL engine that extends DuckDB
- The existing `interruptible_mpmc` class (`src/include/exec/interruptible_mpmc.hpp`) uses a lock-free `BlockingConcurrentQueue` which does not support iteration
- The new class needs iteration for predicate-based inspection, requiring a different internal data structure
- The class will be used in the pipeline execution layer (referenced from `gpu_pipeline_executor.cpp`)
- Header-only template class following the same pattern as `interruptible_mpmc`

## Constraints

- **Tech stack**: C++20, CUDA-compatible, must compile within Sirius build system
- **Pattern**: Header-only template, same style as `interruptible_mpmc.hpp`
- **Location**: `src/include/exec/inspectable_mpsc.hpp`
- **Namespace**: `sirius::exec`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| `std::deque` over `std::list` | Better cache locality for iteration; mid-erase O(n) is acceptable | -- Pending |
| `std::mutex` + `std::condition_variable` over `std::shared_mutex` | MPSC workload is write-heavy; shared_mutex overhead not justified | -- Pending |
| `std::unique_ptr<T>` ownership | Class owns elements exclusively; matches intended MPSC semantics | -- Pending |

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
*Last updated: 2026-04-13 after initialization*
