---
phase: 04-queue-integration
reviewed: 2026-04-14T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - src/include/parallel/task_executor.hpp
  - src/parallel/task_executor.cpp
findings:
  critical: 1
  warning: 2
  info: 1
  total: 4
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-04-14
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed the new `itask_executor` abstract base class and its implementation
(`task_executor.hpp` / `task_executor.cpp`), which consolidates the common
infrastructure — bounded thread pool, inspectable MPSC queue, and manager
thread — shared by `gpu_pipeline_executor`, `duckdb_scan_executor`, and
`downgrade_executor`.

The overall design is clean. The start/stop CAS guard is correct, the virtual
hook pattern is well-structured, and the `stop()` sequence is sound. Three
issues were found: one critical null pointer dereference, two warnings about
silent task loss and a torn `_running` state in `drain_and_wait`, and one info
item about the silently-swallowed `push()` return value in `schedule()`.

---

## Critical Issues

### CR-01: Null pointer dereference in `drain_and_wait()` after `stop()`

**File:** `src/parallel/task_executor.cpp:64`

**Issue:** `drain_and_wait()` dereferences `_bounded_pool` unconditionally on
line 64 (`_bounded_pool->interrupt()`). However, `stop()` sets `_bounded_pool`
to `nullptr` via `_bounded_pool.reset()` (line 50 of the same file) and also
sets `_running` to `false`. If `drain_and_wait()` is ever called after `stop()`
— or if a subclass destructor calls `stop()` before the base destructor's
`stop()` runs and there is a race — this is an immediate null pointer
dereference.

`wait_all()` already guards against this (`if (_bounded_pool) { ... }`), so the
pattern is established. `drain_and_wait()` must apply the same guard.

**Fix:**
```cpp
void itask_executor::drain_and_wait()
{
  // Guard against calling after stop().
  if (!_running.load() || !_bounded_pool) { return; }

  _bounded_pool->interrupt();
  _task_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _bounded_pool->wait_all();
  _task_queue.drain();
  _bounded_pool->resume();
  _task_queue.reactivate();
  _manager_thread = std::thread([this] { manager_loop(); });
}
```

---

## Warnings

### WR-01: `drain_and_wait()` does not update `_running`, leaving a torn state

**File:** `src/parallel/task_executor.cpp:61-82`

**Issue:** `drain_and_wait()` joins the old manager thread and spawns a new one
without ever touching `_running`. The new manager thread begins executing
`manager_loop()` immediately. If a concurrent call to `stop()` arrives while the
new manager is starting up, `stop()` calls `_bounded_pool->interrupt()` and then
`_task_queue.interrupt()` — both safe — but then does
`_bounded_pool.reset()` while the new manager thread may be in the middle of
`_bounded_pool->reserve()`. This is a benign race today because `stop()` calls
`interrupt()` before `reset()` and `reserve()` returns immediately on interrupt,
but the absence of any state coordination means there is no compile-time or
runtime barrier preventing a future refactor from introducing a true UAF.

A more explicit pattern is to set `_running = false` at the start of draining
and `_running = true` after reactivation so the `stop()` CAS properly serializes
with it:

**Fix:**
```cpp
void itask_executor::drain_and_wait()
{
  if (!_running.load() || !_bounded_pool) { return; }

  _bounded_pool->interrupt();
  _task_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _bounded_pool->wait_all();
  _task_queue.drain();
  _bounded_pool->resume();
  _task_queue.reactivate();
  // Restart the manager. _running remains true throughout — the restart is
  // atomic from the perspective of stop(), which can only proceed if _running
  // is still true (CAS succeeds).
  _manager_thread = std::thread([this] { manager_loop(); });
}
```

In the current code this is a code-smell / latent hazard rather than an
immediately triggered bug, but it is worth addressing before additional callers
of `drain_and_wait()` are added.

### WR-02: `schedule()` silently drops tasks when queue is interrupted

**File:** `src/parallel/task_executor.cpp:22-25`

**Issue:**
```cpp
void itask_executor::schedule(std::unique_ptr<itask> task)
{
  static_cast<void>(_task_queue.push(std::move(task)));
}
```

`inspectable_mpsc::push()` is `[[nodiscard]]` and returns `false` when the
queue is inactive (i.e., has been interrupted). `schedule()` intentionally
discards this return value. During the window inside `drain_and_wait()` between
`_task_queue.interrupt()` (line 67) and `_task_queue.reactivate()` (line 80),
any concurrent call to `schedule()` — for example from a GPU worker thread
completing a task and calling `_task_creator->schedule(consumer)` — will have
its task silently dropped. The moved-from `std::unique_ptr<itask>` is destroyed
without ever being executed.

This can cause a query to stall or produce incorrect results: the consumer
operator never runs, its output port is never populated, and upstream pipelines
wait forever. In error-recovery paths this may be acceptable (the query is
already being torn down), but callers that invoke `drain_and_wait()` mid-query
(the `drain_after_error` path in `pipeline_executor.cpp`) should confirm that
no producers are still actively scheduling new tasks during the drain.

If silent drop is intentional during drain, log a warning so silent drops are
observable:

**Fix:**
```cpp
void itask_executor::schedule(std::unique_ptr<itask> task)
{
  if (!_task_queue.push(std::move(task))) {
    SIRIUS_LOG_WARN("itask_executor::schedule: task dropped — queue is inactive");
  }
}
```

---

## Info

### IN-01: `_bounded_pool` is not null-checked in `drain_leftover_tasks()`

**File:** `src/parallel/task_executor.cpp:59`

**Issue:** `drain_leftover_tasks()` calls `_task_queue.drain()` directly, which
is safe because `_task_queue` is a value member. However, `wait_all()` (line 56)
already applies a `_bounded_pool` null check for consistency. This function
itself does not touch `_bounded_pool`, so there is no bug, but the asymmetry
between `wait_all()` (null-checked) and `drain_leftover_tasks()` (not checked)
could mislead future readers into thinking the latter also depends on
`_bounded_pool`. A brief comment clarifying that `_task_queue` is always valid
would aid readability:

**Fix:**
```cpp
// _task_queue is a value member; always safe to call regardless of pool state.
void itask_executor::drain_leftover_tasks() { _task_queue.drain(); }
```

---

_Reviewed: 2026-04-14_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
