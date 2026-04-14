---
phase: 02-predicate-inspection
reviewed: 2026-04-14T20:45:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - src/include/exec/inspectable_mpsc.hpp
  - test/cpp/exec/test_inspectable_mpsc.cpp
findings:
  critical: 0
  warning: 1
  info: 3
  total: 4
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-04-14T20:45:00Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed the `inspectable_mpsc<T>` header-only template class and its 30 Catch2 test cases. The implementation is correct and well-structured. The four new predicate-inspection methods (`pop_if`, `get_if`, `mutable_pop_if`, `mutable_get_if`) use proper locking, correct reverse-iterator-to-forward-iterator conversion for `std::deque::erase`, and return semantics consistent with the existing queue API.

One warning-level issue was found: `std::function` predicates are passed by value, causing unnecessary heap allocations on every call. Three info-level observations relate to code duplication and minor test coverage gaps.

No critical issues, security vulnerabilities, or correctness bugs were found.

## Warnings

### WR-01: std::function predicate parameters passed by value cause unnecessary heap allocation

**File:** `src/include/exec/inspectable_mpsc.hpp:177,214,241,278`
**Issue:** All four predicate methods accept `std::function<bool(const T&)>` or `std::function<bool(T&)>` by value. Each call copies the `std::function` object, which typically heap-allocates the internal callable. Since these methods are called under the mutex (holding the lock for the full scan duration), the allocation overhead compounds with lock hold time. The sibling class `interruptible_mpmc` avoids `std::function` entirely by using template parameters for its API.

Passing by `const&` eliminates the copy. Alternatively, making the predicate a template parameter enables inlining and avoids `std::function` type-erasure overhead entirely, which is the more idiomatic C++20 approach and matches the project's use of `<concepts>` in `interruptible_mpmc.hpp`.

**Fix (option A -- minimal change, pass by const reference):**
```cpp
std::unique_ptr<T> pop_if(const std::function<bool(const T&)>& predicate,
                           bool front_to_back)
```

**Fix (option B -- template predicate, zero-overhead, preferred):**
```cpp
template <typename Pred>
  requires std::invocable<Pred, const T&>
std::unique_ptr<T> pop_if(Pred&& predicate, bool front_to_back)
{
  std::unique_lock<std::mutex> lock(_mutex);
  // ... same body ...
}
```

Apply the same change to `get_if`, `mutable_pop_if`, and `mutable_get_if`.

## Info

### IN-01: Significant code duplication between const and mutable predicate variants

**File:** `src/include/exec/inspectable_mpsc.hpp:177-199,241-263` and `src/include/exec/inspectable_mpsc.hpp:214-227,278-292`
**Issue:** `pop_if` and `mutable_pop_if` have identical method bodies; only the predicate signature differs (`const T&` vs `T&`). Same for `get_if` and `mutable_get_if`. This doubles the surface area for bugs if the scan/erase logic ever needs to change.
**Fix:** Extract a private template helper parameterized on the predicate type:
```cpp
template <typename Pred>
std::unique_ptr<T> pop_if_impl(Pred&& predicate, bool front_to_back)
{
  std::unique_lock<std::mutex> lock(_mutex);
  // ... shared logic ...
}
```
Then the public methods become one-line delegations. If WR-01 option B is adopted (template predicates), the const/mutable distinction is handled automatically by the caller's lambda capture, and only two public methods are needed (`pop_if` and `get_if`), eliminating `mutable_pop_if` and `mutable_get_if` entirely.

### IN-02: Mutable predicate variants not tested for actual mutation through predicate

**File:** `test/cpp/exec/test_inspectable_mpsc.cpp:740-775,781-825`
**Issue:** The `mutable_pop_if` and `mutable_get_if` tests verify that elements are found and removed correctly, but no test exercises the unique capability of the mutable variants: mutating an element through the `T&` reference inside the predicate. A test that modifies an element's state via the predicate (e.g., setting a flag or changing a field) and then verifies the mutation persisted would confirm the mutable reference is properly forwarded.
**Fix:** Add a test case like:
```cpp
TEST_CASE("inspectable_mpsc mutable_get_if allows mutation via predicate",
          "[inspectable_mpsc]")
{
  inspectable_mpsc<test_payload> queue;
  REQUIRE(queue.emplace(1, "original"));

  auto* ptr = queue.mutable_get_if(
      [](test_payload& p) {
        if (p.id == 1) {
          p.data = "mutated";
          return true;
        }
        return false;
      },
      true);

  REQUIRE(ptr != nullptr);
  REQUIRE(ptr->data == "mutated");

  auto item = queue.try_pop();
  REQUIRE(item->data == "mutated");
}
```

### IN-03: Duplicate-element tests cannot distinguish identity of matched element

**File:** `test/cpp/exec/test_inspectable_mpsc.cpp:658-683,685-711`
**Issue:** The "returns first of duplicates" and "returns last of duplicates" tests for `get_if` push multiple elements with the same value (e.g., two `20`s) and then assert `*ptr == 20`. Since both duplicates have the same value, the assertion cannot distinguish whether the first or last was actually found. The test proves correctness indirectly (by removing one and finding the other), but a more robust approach would use elements with distinct identity (e.g., `test_payload` with same `id` but different `data` fields) to verify which specific element was returned.
**Fix:** Use `test_payload` with distinguishing data:
```cpp
// Push: {20, "first"}, {20, "second"}
// get_if(front_to_back=true) should return the one with data=="first"
```

---

_Reviewed: 2026-04-14T20:45:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
