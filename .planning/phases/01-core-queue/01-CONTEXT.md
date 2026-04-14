# Phase 1: Core Queue - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a complete, testable MPSC queue (`inspectable_mpsc<T>`) that can enqueue, dequeue (blocking and non-blocking), manage lifecycle (interrupt/reactivate/drain), and report state -- all thread-safe. This is the foundational class scaffolding and core operations. Predicate-based inspection methods (`pop_if`/`get_if`) are Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Template API compatibility
- **D-01:** Template parameter is `T` (element type), not a smart pointer concept. The class internally manages `std::unique_ptr<T>`. This differs from `interruptible_mpmc` which uses a `smart_pointer` concept accepting both `unique_ptr` and `shared_ptr`.
- **D-02:** No `shared_ptr` support — strictly `unique_ptr<T>` ownership semantics as specified in requirements.

### Blocking pop() semantics
- **D-03:** Use `std::condition_variable::wait()` for true blocking with instant wakeup on push/interrupt. Do not use the 10ms polling pattern from `interruptible_mpmc`. This eliminates busy-waiting and ensures no lost wakeups.

### Test strategy
- **D-04:** Mirror `test_interruptible_mpmc.cpp` structure in a new file `test/cpp/exec/test_inspectable_mpsc.cpp`. Include: basic push/pop, emplace, FIFO order, interrupt/reactivate, drain, state queries, plus multi-threaded MPSC stress tests (4 producers, 1 consumer).
- **D-05:** Use Catch2 framework with `[inspectable_mpsc]` tag, matching existing test conventions.

### size() locking semantics
- **D-06:** `size()` acquires the mutex for an exact count. This guarantees accuracy for callers that use size() in scheduling or diagnostic decisions.
- **D-07:** `is_empty()` also acquires the mutex for consistency with `size()` behavior (both return exact state under lock).

### Claude's Discretion
- Copyright header format (follow Apache 2.0 pattern from existing files)
- Include guard style (`#pragma once` matching existing headers)
- Internal helper method structure
- Exact condition_variable notification strategy (notify_one vs notify_all on push)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Reference implementation
- `src/include/exec/interruptible_mpmc.hpp` -- Existing queue class to mirror style, namespace, and API patterns
- `src/include/exec/channel.hpp` -- Built on interruptible_mpmc; shows publisher/subscriber patterns in sirius::exec

### Test patterns
- `test/cpp/exec/test_interruptible_mpmc.cpp` -- Test structure, Catch2 usage, concurrency test patterns to replicate

### Build integration
- `CMakeLists.txt` -- Test registration patterns for new test files

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `interruptible_mpmc.hpp`: Reference for namespace (`sirius::exec`), coding style (WebKit braces, 2-space indent, 100-char columns), and API surface (push/pop/try_pop/emplace/interrupt/reactivate/drain/is_open/is_empty)
- `test_interruptible_mpmc.cpp`: `test_payload` struct pattern, multi-threaded test structure with timeout guards, Catch2 tag naming

### Established Patterns
- Header-only templates in `src/include/exec/`
- `[[nodiscard]]` on bool-returning methods (push, emplace, is_open)
- `const noexcept` on state query methods
- Deleted copy/move constructors with `= delete` syntax
- `std::memory_order_relaxed` for atomic flag reads in `interruptible_mpmc`

### Integration Points
- New header at `src/include/exec/inspectable_mpsc.hpp` alongside existing exec headers
- New test file at `test/cpp/exec/test_inspectable_mpsc.cpp` alongside existing exec tests
- CMakeLists.txt needs test source registration

</code_context>

<specifics>
## Specific Ideas

No specific requirements -- open to standard approaches following the established patterns in the exec module.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 01-core-queue*
*Context gathered: 2026-04-13*
