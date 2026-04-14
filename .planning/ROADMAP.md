# Roadmap: inspectable_mpsc

## Overview

Deliver a header-only `inspectable_mpsc<T>` template class in two phases: first a fully functional thread-safe queue with push/pop/lifecycle/state operations, then the predicate-based inspection methods that justify the class's existence. Phase 1 produces a drop-in replacement for `interruptible_mpmc` usage patterns. Phase 2 adds the differentiating `pop_if`/`get_if` family with bidirectional search.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Core Queue** - Thread-safe queue with push/pop, lifecycle control, state queries, and class scaffolding
- [ ] **Phase 2: Predicate Inspection** - Selective element access via pop_if/get_if with const and mutable variants

## Phase Details

### Phase 1: Core Queue
**Goal**: A complete, testable MPSC queue that can enqueue, dequeue (blocking and non-blocking), manage lifecycle (interrupt/reactivate/drain), and report state -- all thread-safe
**Depends on**: Nothing (first phase)
**Requirements**: STRC-01, STRC-02, STRC-03, CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, LIFE-01, LIFE-02, LIFE-03, STAT-01, STAT-02, STAT-03, SAFE-01, SAFE-02, SAFE-03
**Success Criteria** (what must be TRUE):
  1. Multiple threads can push items concurrently and a single consumer can pop them in FIFO order without data loss or corruption
  2. A consumer calling pop() on an empty queue blocks until an item is pushed or interrupt() is called -- no busy-waiting, no lost wakeups
  3. Calling interrupt() unblocks all waiting consumers and causes push/pop to return failure/nullptr; calling reactivate() restores normal operation
  4. drain() removes all queued items, and is_open()/is_empty()/size() accurately reflect queue state at the point of query
  5. The class compiles as a header-only template in the Sirius build system at `src/include/exec/inspectable_mpsc.hpp` within `sirius::exec` namespace
**Plans:** 2 plans

Plans:
- [ ] 01-01-PLAN.md — TDD: Implement inspectable_mpsc header + single-threaded unit tests
- [ ] 01-02-PLAN.md — Multi-threaded MPSC concurrency stress tests (SAFE-01)

### Phase 2: Predicate Inspection
**Goal**: Consumers can search the queue for specific elements by predicate and selectively remove or inspect them, with control over search direction
**Depends on**: Phase 1
**Requirements**: INSP-01, INSP-02, INSP-03, INSP-04, INSP-05
**Success Criteria** (what must be TRUE):
  1. pop_if with a matching predicate removes and returns the first matching element; the queue retains all non-matching elements in original order
  2. get_if returns a raw pointer to the first matching element without removing it; the element remains in the queue
  3. mutable_pop_if and mutable_get_if behave identically to their const counterparts but the predicate receives a mutable reference, allowing state inspection that requires non-const access
  4. Setting front_to_back=true searches oldest-to-newest; front_to_back=false searches newest-to-oldest; both return the first match in their respective direction
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Queue | 0/2 | Planning complete | - |
| 2. Predicate Inspection | 0/0 | Not started | - |
