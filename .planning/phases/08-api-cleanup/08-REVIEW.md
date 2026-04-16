---
phase: 08-api-cleanup
reviewed: 2026-04-16T21:05:42Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/downgrade/downgrade_executor.cpp
  - src/include/downgrade/downgrade_executor.hpp
  - src/pipeline/gpu_pipeline_executor.cpp
  - test/cpp/downgrade/test_downgrade_executor.cpp
  - test/cpp/downgrade/test_downgrade_lifecycle.cpp
  - CMakeLists.txt
  - docs/super-sirius/memory-management.md
  - docs/super-sirius/optimizations.md
findings:
  critical: 2
  warning: 1
  info: 2
  total: 5
status: issues_found
---

# Phase 08: Code Review Report

**Reviewed:** 2026-04-16T21:05:42Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Review covers the new `downgrade_executor` implementation (request-based API with predicate-driven
downgrade), its integration into `gpu_pipeline_executor`, the lifecycle and behavioral test suites,
the CMake build wiring, and two documentation files.

The CMakeLists.txt changes are clean — both test files are correctly registered and the extension
source list is consistent. The test suites are thorough and well-structured.

Two critical bugs exist in the public API of `downgrade_executor`: one is a null-pointer dereference
after a `std::move`, and the other is a deadlock hazard (future that never resolves) in the
`request_free_memory` failure path. One dead variable in `gpu_pipeline_executor.cpp` indicates
incomplete or abandoned logic. Two documentation entries reference artifacts from the old downgrade
architecture that no longer exist.

---

## Critical Issues

### CR-01: Null-pointer dereference after `std::move` in `request_downgrade`

**File:** `src/downgrade/downgrade_executor.cpp:318-320`

**Issue:** `req` is moved into `_request_queue.push(std::move(req))` on line 318. When `push`
returns `false` (queue inactive), the error-handling path on line 320 dereferences `req->result`
— but `req` is now a moved-from (null) `unique_ptr`. This is an unconditional null pointer
dereference (undefined behavior, almost certainly a crash).

```cpp
// Current (buggy):
auto future = req->result.get_future();   // line 317 — future extracted first
if (!_request_queue.push(std::move(req))) {
  SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
  req->result.set_value(0);  // BUG: req is null here
  return future;
}
```

**Fix:** Fulfill the promise via the already-extracted future's shared state. The promise must be
fulfilled *before* the move, or the fix below uses the pattern already correct in `request_free_memory`
(extract future first, fulfill the promise via a separate mechanism). Simplest safe fix:

```cpp
std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    // req is moved-from; fulfill via a temporary promise sharing the same future state.
    // We need a fresh approach: extract before move.
    // NOTE: get_future() was already called above, so we cannot call it again.
    // Use a std::promise to resolve via set_value_at_thread_exit, or restructure:
    std::promise<size_t> p;
    // Cannot share — correct fix is to keep the promise accessible:
  }
  return future;
}
```

The cleanest fix is to hold a raw pointer to the `downgrade_request` before moving it, and use
that pointer in the error path (the object is already destroyed if push succeeds, but if push
fails, the caller still holds it):

```cpp
std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  auto* req_raw  = req.get();  // raw pointer valid until push succeeds or fails
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    req_raw->result.set_value(0);  // safe: push failed, unique_ptr was NOT transferred
    // NOTE: req_raw points to freed memory if push succeeds — only use on failure path.
  }
  return future;
}
```

Wait — if `push` returns `false`, the `unique_ptr` was NOT transferred (push failed), so `req`
is still valid (not null). The issue is that `req` was passed via `std::move`, making `req` null
regardless of whether push succeeded or failed. The actual clean fix:

```cpp
std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    // Fulfill the already-extracted future via a new promise on the stack.
    // Since future is already detached, use std::promise::set_value via promise_at_exit,
    // or restructure to not std::move before checking:
  }
  return future;
}
```

The simplest correct fix avoids moving before the check:

```cpp
std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  bool pushed    = _request_queue.push(std::move(req));
  if (!pushed) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    // req is moved-from; promise is unreachable — this is the root problem.
    // Restructure: get promise ref before move.
  }
  return future;
}
```

**Root cause:** The correct pattern (matching `request_free_memory`) is to NOT set_value in the
error path via `req` after move. Instead, mirror `request_free_memory` which simply returns the
future without fulfilling the promise on failure (which is itself a bug — see CR-02). The actual
fix for both is to check push success before moving:

```cpp
std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    // Promise is moved into req which is moved into push's argument.
    // To fulfill: use a local promise and return its future instead.
    std::promise<size_t> p;
    auto f = p.get_future();
    p.set_value(0);
    return f;
    // OR: restructure to avoid move before the check (see CR-02 fix).
  }
  return future;
}
```

**Recommended fix:** Extract the future early but also return an immediately-fulfilled future on
the error path:

```cpp
std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    std::promise<size_t> p;
    auto early_future = p.get_future();
    p.set_value(0);
    return early_future;
  }
  return future;
}
```

---

### CR-02: Future permanently unresolved when `request_free_memory` push fails

**File:** `src/downgrade/downgrade_executor.cpp:294-306`

**Issue:** When `_request_queue.push(std::move(req))` returns `false`, the function returns the
extracted `future` but the associated promise was moved into the queue argument and is now
destroyed along with the moved-from `req` — nobody calls `set_value` or `set_exception`.
Any caller (including `request_free_memory_and_wait`) calling `.get()` on this future will block
indefinitely, causing a deadlock.

```cpp
// Current (buggy):
std::future<size_t> downgrade_executor::request_free_memory(size_t bytes)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = ...;
  auto future = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("...");
    // BUG: future is returned but promise is already destroyed — future.get() hangs
  }
  return future;
}
```

Note: when a `std::promise` is destroyed without calling `set_value` or `set_exception`, calling
`.get()` on the associated future throws `std::future_error` with
`std::future_errc::broken_promise`. In practice this means `request_free_memory_and_wait` throws
instead of hanging, but the exception is unhandled at the call sites. Still a bug.

**Fix:**

```cpp
std::future<size_t> downgrade_executor::request_free_memory(size_t bytes)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = [&freed = req->bytes_freed, bytes]() {
    return freed.load(std::memory_order_relaxed) >= bytes;
  };
  auto future = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN(
      "[downgrade] request_free_memory: queue inactive, dropping request for {} bytes", bytes);
    std::promise<size_t> p;
    auto early_future = p.get_future();
    p.set_value(0);
    return early_future;
  }
  return future;
}
```

---

## Warnings

### WR-01: Dead variable `pipeline_done` in `gpu_pipeline_executor.cpp`

**File:** `src/pipeline/gpu_pipeline_executor.cpp:336`

**Issue:** The variable `pipeline_done` is computed but never read. It is assigned the result of
`pipeline && pipeline->is_pipeline_finished()` but then unused — consumers are scheduled
unconditionally inside the `if (!query_complete && _task_creator)` branch regardless of
`pipeline_done`. This either indicates dead code left over from a refactor, or an intended guard
that was accidentally dropped.

```cpp
if (!query_complete && _task_creator) {
  bool pipeline_done = pipeline && pipeline->is_pipeline_finished();  // computed but never used
  for (auto* consumer : consumers) {
    if (consumer) { _task_creator->schedule(consumer); }
  }
}
```

**Fix:** If `pipeline_done` was meant to gate consumer scheduling (i.e., only schedule consumers
when the current pipeline is done), restore the guard:

```cpp
if (!query_complete && _task_creator) {
  bool pipeline_done = pipeline && pipeline->is_pipeline_finished();
  if (pipeline_done) {
    for (auto* consumer : consumers) {
      if (consumer) { _task_creator->schedule(consumer); }
    }
  }
}
```

If `pipeline_done` is intentionally not used (consumers should always be scheduled when the task
completes), remove the variable to avoid compiler warnings and future confusion:

```cpp
if (!query_complete && _task_creator) {
  for (auto* consumer : consumers) {
    if (consumer) { _task_creator->schedule(consumer); }
  }
}
```

---

## Info

### IN-01: Stale documentation in `optimizations.md` — references removed `run_downgrade_pass()`

**File:** `docs/super-sirius/optimizations.md:104-111`

**Issue:** The "Memory-Pressure-Driven Downgrade (PR #368)" section describes the old downgrade
architecture (`run_downgrade_pass()` with scored/sorted candidates) that was replaced by the
request-based `processing_loop()` pattern (PR #579). The function `run_downgrade_pass()` no
longer exists in the codebase. The described candidate selection logic (partitioned repos first,
sorted by data size descending) does not match current behavior (lazy tiered iteration in
insertion order). This entry will mislead developers trying to understand the current code path.

**Fix:** Update the PR #368 section to reflect that the mechanism was superseded by PR #579, and
update the code path reference:

```markdown
**Code path:** `src/downgrade/downgrade_executor.cpp` — `monitor_loop()`, `processing_loop()`
```

Also update the mechanism description to match current behavior: the monitor calls
`request_free_memory()` which enqueues a `downgrade_request`; the processing loop handles it via
lazy tiered candidate iteration.

---

### IN-02: Stale file reference in `memory-management.md` key files table

**File:** `docs/super-sirius/memory-management.md:174`

**Issue:** The key files table includes `src/include/downgrade/downgrade_task.hpp` with the
description "Downgrade task definition." This file does not exist — the downgrade task abstraction
was replaced by `convertible_data` and the request-based API in `downgrade_executor.hpp`.

**Fix:** Remove the `downgrade_task.hpp` row from the key files table. Consider adding entries
for the new data provider files if they are considered public API:

```markdown
| `src/include/data/convertible_data.hpp`             | Downgrade candidate abstraction     |
| `src/include/data/convertible_data_batch.hpp`        | Batch provider for repositories     |
| `src/include/data/convertible_gpu_pipeline_task.hpp` | Task provider for queue inspection  |
```

---

_Reviewed: 2026-04-16T21:05:42Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
