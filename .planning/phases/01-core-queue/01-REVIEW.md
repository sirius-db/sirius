---
phase: 01-core-queue
reviewed: 2026-04-13T22:15:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - src/include/exec/inspectable_mpsc.hpp
  - test/cpp/exec/test_inspectable_mpsc.cpp
  - CMakeLists.txt
findings:
  critical: 0
  warning: 2
  info: 3
  total: 5
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-04-13T22:15:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 1 delivers the `inspectable_mpsc<T>` header-only template class and its Catch2 unit tests (14 single-threaded + 4 concurrent stress tests). The implementation is clean, follows the existing `interruptible_mpmc` style conventions, and uses `std::mutex` + `std::condition_variable` correctly for thread-safe FIFO operations. The CMakeLists.txt change is a single-line addition in the correct alphabetical position.

Two warnings relate to `noexcept` correctness and a missing test for drain-after-interrupt semantics. Three informational items cover minor documentation and style observations.

No critical issues found. The code is well-structured and production-ready for its Phase 1 scope.

## Warnings

### WR-01: noexcept on methods that acquire std::mutex

**File:** `src/include/exec/inspectable_mpsc.hpp:130-148`
**Issue:** `is_open()`, `is_empty()`, and `size()` are marked `noexcept` but internally call `std::unique_lock<std::mutex> lock(_mutex)`, which can throw `std::system_error` if the underlying pthread mutex operation fails (e.g., EAGAIN from resource exhaustion). If the lock throws, `std::terminate()` will be called due to the `noexcept` specifier.

This is a known design decision (documented in the plan as matching `interruptible_mpmc` convention), and `std::mutex::lock()` throwing is extremely rare in practice. However, `interruptible_mpmc` does not have this exact issue because its `is_open()` reads an `std::atomic<bool>` (no lock), and its `is_empty()` calls `size_approx()` on the lock-free concurrent queue (also no lock). So the `noexcept` precedent from `interruptible_mpmc` does not directly apply to `inspectable_mpsc` which actually acquires a mutex in these methods.

**Fix:** Either remove `noexcept` from the three methods, or add a comment documenting the intentional choice:
```cpp
// noexcept: std::mutex::lock() can theoretically throw std::system_error but
// this is practically impossible in normal operation. Matches project convention.
[[nodiscard]] bool is_open() const noexcept {
```

### WR-02: No test for drain-after-interrupt semantics

**File:** `test/cpp/exec/test_inspectable_mpsc.cpp`
**Issue:** The `pop()` method documents drain-after-interrupt behavior (lines 74-76 of the header): "If the queue is interrupted but still has items, those items are returned before nullptr." However, there is no test that verifies this contract. This is a meaningful behavioral guarantee that should be covered. The existing test "blocking pop returns nullptr after interrupt" only tests interrupt on an empty queue.

**Fix:** Add a test case that pushes items, interrupts, then verifies `pop()` returns remaining items before returning `nullptr`:
```cpp
TEST_CASE("inspectable_mpsc pop drains remaining items after interrupt", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  REQUIRE(queue.push(std::make_unique<int>(1)));
  REQUIRE(queue.push(std::make_unique<int>(2)));
  REQUIRE(queue.push(std::make_unique<int>(3)));

  queue.interrupt();

  // Should drain remaining items in FIFO order
  auto r1 = queue.pop();
  REQUIRE(r1 != nullptr);
  REQUIRE(*r1 == 1);

  auto r2 = queue.pop();
  REQUIRE(r2 != nullptr);
  REQUIRE(*r2 == 2);

  auto r3 = queue.pop();
  REQUIRE(r3 != nullptr);
  REQUIRE(*r3 == 3);

  // Now empty and interrupted -- should return nullptr
  auto r4 = queue.pop();
  REQUIRE(r4 == nullptr);
}
```

## Info

### IN-01: push() does not validate for nullptr input

**File:** `src/include/exec/inspectable_mpsc.hpp:48`
**Issue:** The `push()` method accepts a `std::unique_ptr<T>` but does not validate that it is non-null before enqueueing. The sibling `interruptible_mpmc::push()` has `assert(item != nullptr)` (line 80 of `interruptible_mpmc.hpp`). A nullptr unique_ptr would be silently enqueued, and `pop()`/`try_pop()` would return a moved-from (null) unique_ptr indistinguishable from "queue empty" for callers that only check `result != nullptr`.

**Fix:** Add a debug assertion:
```cpp
[[nodiscard]] bool push(std::unique_ptr<T> item) {
    assert(item != nullptr);
    std::unique_lock<std::mutex> lock(_mutex);
    ...
}
```

### IN-02: test_payload struct defined in anonymous namespace collision risk

**File:** `test/cpp/exec/test_inspectable_mpsc.cpp:33-38`
**Issue:** The `test_payload` struct is defined at file scope without an anonymous namespace, same as in `test_interruptible_mpmc.cpp`. If both test files are linked into the same binary (they are -- `sirius_unittest`), this creates two definitions of `test_payload` with external linkage in the same program. Under C++ ODR, if both definitions are token-identical this is permitted; they are currently identical so this works. However, if one file's `test_payload` diverges in future maintenance, this becomes an ODR violation (undefined behavior). This matches the existing pattern in the codebase so it is not a blocking issue.

**Fix:** Wrap in an anonymous namespace or mark with `static`/internal linkage:
```cpp
namespace {
struct test_payload {
  int id;
  std::string data;
  test_payload(int i, std::string d) : id(i), data(std::move(d)) {}
};
}  // namespace
```

### IN-03: Concurrent test timeout handlers use detach() which leaks threads

**File:** `test/cpp/exec/test_inspectable_mpsc.cpp:175,277,339,395,455,488,509`
**Issue:** Several timeout handlers call `consumer.detach()` before `FAIL(...)`. This prevents the test runner from hanging but leaks the detached thread (which may still hold a reference to the now-destroyed `queue` local variable, causing use-after-free). This is a pragmatic test pattern (also used in `test_interruptible_mpmc.cpp`) -- the alternative of `std::terminate()` on timeout is worse. Not a production code concern.

**Fix:** No change needed. Documenting for awareness. The pattern is consistent with existing test code.

---

_Reviewed: 2026-04-13T22:15:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
