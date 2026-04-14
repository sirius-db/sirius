# Phase 1: Core Queue - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-13
**Phase:** 01-core-queue
**Areas discussed:** Test strategy, size() locking semantics

---

## Test strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Match existing pattern (Recommended) | Mirror test_interruptible_mpmc.cpp structure: basic push/pop, emplace, FIFO order, interrupt/reactivate, drain, state queries, plus multi-threaded MPSC stress tests (4 producers, 1 consumer). New file at test/cpp/exec/test_inspectable_mpsc.cpp. | ✓ |
| Minimal unit tests only | Single-threaded correctness tests for each API method. Skip concurrency stress tests. | |
| Comprehensive with edge cases | Full suite plus: interrupt-during-pop race, reactivate-after-drain sequencing, size() accuracy under contention, empty queue state transitions. | |

**User's choice:** Match existing pattern (Recommended)
**Notes:** None

---

## size() locking semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Lock-free racy read (Recommended) | Return deque.size() without locking. Documented as point-in-time snapshot. Matches interruptible_mpmc pattern. | |
| Locked exact count | Acquire the mutex before reading deque.size(). Guarantees accuracy but adds contention. | ✓ |
| Both: size() locked, size_approx() unlocked | Provide two methods for caller choice. | |

**User's choice:** Locked exact count
**Notes:** None

---

## Claude's Discretion

- Copyright header format
- Include guard style
- Internal helper method structure
- Condition_variable notification strategy

## Deferred Ideas

None
