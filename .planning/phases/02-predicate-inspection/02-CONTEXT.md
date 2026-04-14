# Phase 2: Predicate Inspection - Context

**Gathered:** 2026-04-14 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Add predicate-based inspection and selective removal methods to the existing `inspectable_mpsc<T>` class. Consumers can search the queue for specific elements by predicate and selectively remove or inspect them, with control over search direction. The core queue (push/pop/lifecycle/state) was delivered in Phase 1.

</domain>

<decisions>
## Implementation Decisions

### Predicate Parameter Type
- **D-01:** Use `std::function<bool(const T&)>` for const variants and `std::function<bool(T&)>` for mutable variants, exactly as specified in REQUIREMENTS.md INSP-01 through INSP-04. Do not templatize the predicate parameter.

### Iterator Strategy for Bidirectional Search
- **D-02:** Use manual forward iterator loops (`begin()`/`end()`) for `front_to_back=true` and manual reverse iterator loops (`rbegin()`/`rend()`) for `front_to_back=false`. Avoid `std::find_if` to keep the iterator position available for `erase()` without reverse-to-forward conversion issues.

### Lock Scope During Predicate Evaluation
- **D-03:** Hold `_mutex` for the entire duration of predicate evaluation across all queue elements. No lock release/reacquire mid-scan. This matches the existing Phase 1 pattern where every method holds the mutex for its full duration. Predicates should be lightweight (no I/O, no heavy computation).

### get_if Raw Pointer Return Safety
- **D-04:** Return `T*` from both `get_if` and `mutable_get_if` as specified in requirements. The pointer is valid under MPSC semantics because only the single consumer calls inspection methods. Add a documentation comment warning that the returned pointer is invalidated by any subsequent mutating operation (`pop()`, `pop_if()`, `drain()`).

### Claude's Discretion
- Whether to use `std::next(rit).base()` or manual index-based erase for reverse `pop_if` -- choose whichever is clearest
- Exact Doxygen comment style for new methods (follow existing `push()`/`pop()` comment pattern)
- Whether to add a private helper for the iteration loop shared between const and mutable variants, or keep each method self-contained

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Implementation target
- `src/include/exec/inspectable_mpsc.hpp` -- Phase 1 implementation to extend with predicate methods

### Reference patterns
- `src/include/exec/interruptible_mpmc.hpp` -- Existing queue class; namespace, style, and API conventions
- `src/include/exec/channel.hpp` -- Publisher/subscriber patterns in sirius::exec

### Test patterns
- `test/cpp/exec/test_inspectable_mpsc.cpp` -- Existing tests to extend with predicate inspection tests

### Requirements
- `.planning/REQUIREMENTS.md` -- INSP-01 through INSP-05 define exact method signatures and behavior

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `inspectable_mpsc.hpp` (Phase 1): Complete class with `std::deque<std::unique_ptr<T>>` backing, `std::mutex`/`std::condition_variable` synchronization, all lifecycle methods. New methods add to this existing class.
- `test_inspectable_mpsc.cpp` (Phase 1): 516 lines of tests including MPSC stress tests. New predicate tests extend this file.

### Established Patterns
- All public methods acquire `_mutex` via `std::unique_lock<std::mutex>` at entry
- `[[nodiscard]]` on bool-returning methods
- `const noexcept` on state query methods
- WebKit braces, 2-space indent, 100-char columns
- `notify_one()` after push, `notify_all()` after interrupt/reactivate/drain
- Doxygen `\brief` and `\return` comments on each public method
- `test_payload` struct for testing with custom types
- Timeout guards in threaded tests (1-5s timeouts with `FAIL()`)

### Integration Points
- New methods added directly to `inspectable_mpsc<T>` class in `inspectable_mpsc.hpp`
- New test cases added to existing `test_inspectable_mpsc.cpp`
- No CMake changes needed (same files, same test binary)

</code_context>

<specifics>
## Specific Ideas

No specific requirements -- open to standard approaches following the established patterns from Phase 1.

</specifics>

<deferred>
## Deferred Ideas

None -- analysis stayed within phase scope.

V2 requirements explicitly deferred in REQUIREMENTS.md:
- EXT-01: `visit_if(predicate, callback)` -- safer callback-under-lock alternative to `get_if`
- EXT-03: `pop_if` with `max_scan_depth` to bound worst-case lock hold time

</deferred>

---

*Phase: 02-predicate-inspection*
*Context gathered: 2026-04-14*
