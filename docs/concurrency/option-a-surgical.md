# Option A — Surgical

> **Philosophy:** no new abstractions. Every issue in [the register](00-issue-register.md) is fixed
> at its own call site with the narrowest possible change, using the per-query APIs that **already
> exist** (`drain_query_tasks(query_id)`, `drain(query_index{})`, `size(query_index{})`,
> `drain_pending_tasks(query_id)`, `reset(query_id)`). Correctness of teardown rests on a
> **quiescence proof**: at the moment a query's repositories are erased, no thread holds a borrowed
> pointer into them, because every producer was stopped first and stays stopped.

**Shape:** 12 steps · ~1,400 LOC net · 0 new files in `src/` (2 in `test/`)
**Keeps:** `unique_ptr<data_repository>`, raw `data_repository*`, `enter/leave/wait_for_in_flight`,
`bounded_thread_pool` unchanged.

---

## Step-by-step

### A1 — Concurrency harness + admission knob + scan pool sizing
**Closes:** G1, G2, G3, C3, H4 · **Enables:** everything downstream

- New `test/cpp/utils/concurrent_query_harness.hpp`: N connections from `shared_test_env`, a
  release-gate start barrier, per-thread result/exception capture, deadline watchdog. Salvage the
  primitives currently trapped in `test_query_lifecycle_slot.cpp`'s anonymous namespace
  (`wait_for_workers`, `async_query_result`, `scoped_blocking_window_log_sink`) — move them here
  and have that file include it, so there is one implementation.
- Relax `shared_env_listener` / `scoped_mgpu_env` so a test can opt out of env-pausing (G3).
- Promote `k_max_concurrent_queries` to a real `scan_manager_config::max_concurrent_queries`,
  parsed in `sirius_config.cpp::from_yaml`, **default 1**. Pool stays `num_threads + N`.
  Make `sirius_scan_manager.cpp:597` reject rather than warn.
- Add `sirius_config::max_concurrent_queries` (**default 1**) and turn
  `acquire_query_lifecycle_slot` from a `std::mutex` into a counting semaphore of that size.
  At 1 the behaviour is bit-identical to today, including the same-thread-reacquire error.
- Delete the never-locked `SiriusContext::mutex_`.

**Tests:** harness self-test; existing `test_query_lifecycle_slot.cpp` suite unchanged and green at
N=1. One `[.]`-hidden `[concurrency]` test at N=2 that will be un-hidden by step A11.

---

### A2 — Make dropped work loud
**Closes:** A9

Check the `bool` from every `push()`; on a drop for a still-live query, `report_error` on that
query's completion handler instead of silently destroying the task. Add a `SIRIUS_LOG_ERROR` with
the query id at each site.

> Fixes nothing on its own. Landed second because until it exists, **every** bug in Group A
> presents as an unexplained hang and the remaining steps are much harder to debug.

**Tests:** unit test that an interrupted queue's dropped task surfaces as a query error.

---

### A3 — Stop per-query failures from killing shared subsystems
**Closes:** A1, A3, A4, A10, and the global-halt half of A2

- `terminate_query`: delete the `stop()`. Report to the handler, then `drain_query_tasks(query_id)`.
- Creation-worker catch (`task_creator.cpp:692`): delete the `stop()`.
- GPU executor catch (`:427`, `:432`): delete `_task_creator->stop()`. `completion->report_error`
  on the next line already fails the correct query.
- All four `break`s in `manager_loop` → report to *that task's* handler, then `continue`.
  Only channel/queue closure may break the loop.
- Wrap the `manager_loop` body in try/catch so a throw cannot abort the process.

**Tests:** inject a creation exception and a task exception; assert the scheduler and creator are
still running and a second query completes normally.

---

### A4 — Fix the creator's lock ordering and in-flight coverage
**Closes:** C1, C2, D1, D2, and the self-deadlock half of A2

- Split `_global_state_mutex` from a new `_lifecycle_mutex`; never hold either across
  `_manager_thread.join()`.
- `stop()` takes the same lifecycle mutex as `stop_thread_pool()`/`start_thread_pool()`.
- Forbid `stop()` from a pool worker (assert + post a stop request instead).
- Move `enter_in_flight()` to immediately after the successful `get_query_task_global_state`
  lookup, so the manager thread's own `get_operator_for_next_task(node)` dereference is inside
  the counted region. Release on every early-`continue`.
- Cover the key extractor's `request.node->type` dereference the same way.

**Tests:** a targeted test that `drain_pending_tasks(Q)` returning implies no thread is
dereferencing Q's operators; a stop/start hammer test.

---

### A5 — Per-query completion and error paths
**Closes:** A5, A6, D4

- `wait_for_completion(query_id)`: `_task_queue.size(query_index{q})`,
  `gpu_exec->wait_and_validate_empty(query_id)`, and **delete** the
  `stop_thread_pool()`/`start_thread_pool()` pair — per-query in-flight accounting replaces it.
- `drain_after_error(query_id)`: `_task_queue.drain(query_index{q})` in both places,
  `gpu_exec->drain_query_tasks(query_id)` instead of `drain_and_wait()`, same pool-cycle deletion.
- New `itask_executor::wait_and_validate_empty(query_id)`; reserve the global `drain_and_wait()`
  for `stop()`/teardown only. Add the missing null-pool guard.
- Guard `*_ready_devices.begin()`.

**Tests:** query A completes while query B has queued tasks — A must not throw and B must not lose
work. Same for A erroring.

---

### A6 — Per-query downgrade
**Closes:** A7, B2, D6

- `downgrade_request` gains a `query_id`. `drain(query_id)` cancels only that query's requests.
- Split `drain()` into `quiesce()` / `resume()` so the caller controls when the executor re-arms —
  the current unconditional restart is what makes quiescence expire before `erase()`.
- Add the `_running` guard (`drain()` after `stop()` currently null-derefs and leaks a joinable
  thread).
- Reset `_monitor_request_enqueued` via RAII on the request and on a failed push.
- `data_repository_manager_registry::get_all()` returns `{query_id, manager}` pairs so the sweep
  can attribute candidates.

**Tests:** query A's cleanup must not cancel query B's pending downgrade; monitor latch survives a
drain.

---

### A7 — Close the repository lifetime hole by quiescence
**Closes:** B1, B3, B4, B9

The heart of Option A. No shared ownership — instead, make the "nobody holds a borrowed pointer"
precondition **enforced** rather than documented:

- Add a per-query `closing` flag to `data_repository_manager_registry`. `get_all()` skips closing
  queries, so the sweep can never *start* on a dying query.
- The TIER-1 and TIER-2 candidate paths re-check `closing` **after** the blocking
  `_pool->reserve()` returns, beside the existing `req->satisfied` re-check. This is what closes
  B1's extraction window, which pool-level accounting cannot see (no job exists yet).
- Gate `~convertible_gpu_pipeline_task`'s re-push on `closing`: drop the task instead of
  resurrecting it.
- Reorder `run_mandatory_cleanup` to: **mark closing → downgrade `quiesce(q)` → creator `reset(q)`
  → scheduler/executor drains(q) → repo erase → scan reset → downgrade `resume()`**.
  The current order (queue drain *before* downgrade drain) is precisely what makes B1 fire.
- Take `_mutex` in `cucascade::data_repository_manager::get_repository`.

**Tests:** a stress test that hammers TIER-2 downgrade against a query ending — under ASan and TSan.

---

### A8 — Plan lifetime
**Closes:** B5, B8

- Move the drains so the plan outlives them. Preferred: `StandaloneQueryScope` takes ownership of
  the engine, and `finish()` orders drain → repo erase → **then** engine destruction. Fallback:
  move the drain into `cleanup_internal` before `sirius_active_query.reset()`.
- Add the guarded `data_repository_registry_.erase(query_id)` that
  `drop_query_runtime_state_best_effort` is missing.

**Tests:** force `get_result()` to throw (the path that currently destroys the plan with no drain
having run) and assert clean teardown under ASan.

---

### A9 — Shutdown ordering
**Closes:** B6, B7, B10

- `terminate()`: stop the creator, scan manager and downgrade executors (or null their
  `_pipeline_task_queue`) **before** `task_scheduler_.reset()`. Null `task_scheduler::_task_creator`
  and `gpu_pipeline_executor::_downgrade_executor` in the respective `stop()`s.
- Declare `task_scheduler_` **last** in `SiriusContext` so it is destroyed first, or add a
  last-declared RAII shutdown sentinel running an idempotent `terminate()`.
- Wrap the `terminate()` call in `~SiriusContext` in try/catch + log.

**Tests:** a test that `initialize()` throwing after `task_scheduler_` is constructed tears down
cleanly under ASan.

---

### A10 — Per-query telemetry and prefetch-cache safety
**Closes:** A8, B11

- Add `query_id` to `batch_telemetry_registry`'s placements and ports; `on_query_end(query_id)`
  filters both loops. Re-key `ports` on the existing `port::source_port_uuid` instead of a raw
  `data_repository*` (address recycling silently matches a stale entry).
- `prefetching_cache`: copy `file_entry*` out under the lock and drop the iterator before
  unlocking. Element pointers are stable; iterators are not.

---

### A11 — Turn concurrency on and prove it
**Closes:** the validation half of G1, plus G4, G5, G7

- Raise the default `max_concurrent_queries` above 1.
- Un-hide the `[concurrency]` suite; re-express `test_query_lifecycle_slot.cpp`'s
  serialization assertions against the new semantics (G4); wire up `run_ac13_concurrent_logging`.
- Add a concurrent TPC-H test (two different queries, and the same query twice) and a
  `concurrentloop` SQLLogic file.
- Run the whole suite under ASan and TSan.

> **This is the step where unknown-unknowns surface.** Budget for it accordingly; it is the one
> step in this plan whose size cannot be estimated from the register.

---

### A12 — Fairness, config safety, and hygiene
**Closes:** D3, D5, E1–E7, F1–F9, H1–H3, H5–H11

Largely independent of each other; can be split into several commits or parallelised.

- **Fairness (F1):** make the query bits a fairness index — round-robin across live queries at
  dispatch — instead of a strict priority prefix. Rotate `schedule_lookahead` across live queries
  (D3).
- **Config (E1–E3, E7):** move `operator_params` and the `Config::` static variables into per-`ClientContext`
  settings (the pattern already used for `enable_duckdb_fallback`), snapshotting by value into the
  plan at window begin. Guard the `LOG_*` trio, which is genuinely process-wide. Make
  `original_config` a stack-local RAII.
- **E4:** `logical_plan_` → `shared_ptr<const LogicalOperator>` + an atomic
  `copy_unsupported_` latch.
- **E5, E6:** resolve-or-insert under one `unique_lock`, keyed on `(catalog, schema, table)`;
  atomic packed pair or per-device `call_once`.
- **D5:** split `runtime_unavailable_` into "this query is corrupt" (fail that query) vs "the
  shared runtime is corrupt" (latch).
- **F2:** narrow the plan-time `SlotGuard` to a shared read of the pin registry.
- **F3–F9:** live prefetch-epoch set; fair-share credit on `admission_control`; per-entry reader
  count replacing the `use_count() > 1` heuristic; reservation-driven sizing; remove
  `cudaDeviceSynchronize` and default-stream work.
- **H:** delete H1–H7 and H11; widen `query_id_t` to 64-bit and stop packing it into the signed
  priority (H8); rewrite the five drifted docs and add a per-query-model doc (H9); rename the
  lying `shared_data_repository` aliases (H10).

---

## Pros

- **Smallest total diff and the smallest diff per step.** Every commit is reviewable in isolation
  by someone who knows only the file it touches.
- **No shared primitive is destabilised.** `bounded_thread_pool` is used by four subsystems; not
  touching it means a mistake there cannot take out task creation, GPU dispatch and downgrade at
  once.
- **Uses machinery that already exists and is already tested.** `drain(query_index{})`,
  `size(query_index{})` and `drain_query_tasks(query_id)` all landed in the last six commits.
  Steps A5 and A6 are largely "call the overload that is already there".
- **Lowest risk of a bad revert.** Any single step can be reverted without stranding the others.
- **Memory release stays deterministic.** `erase(query_id)` remains the exact point where a
  query's GPU memory returns, so the `log_pool_stats` leak signature
  (`QueryEnd allocated != QueryBegin allocated`) keeps working — that check is how leaks are
  currently caught.
- **Preserves the leaked-batch diagnostic.** `clear_all_repositories()`'s un-consumed batch report
  is how you learn "operator X didn't drain port Y"; nothing here makes a crash into a silent leak.

## Cons

- **Twelve mechanisms instead of one invariant.** The `closing` flag (A7), the in-flight counter
  (A4), the per-query queue index (A5), the per-query downgrade drain (A6) and the cleanup ordering
  (A7) each independently guarantee a piece of teardown. There is no single place to look to
  answer "is it safe to erase this query's repositories?", and no compiler help — the argument has
  to be re-made by hand at every new call site.
- **Quiescence is order-sensitive, and the order is load-bearing but implicit.** A7 pins a specific
  six-step sequence in `run_mandatory_cleanup`. A future contributor reordering it reintroduces B1
  with no test failure at N=1.
- **A7's re-check-after-reserve is subtle** and easy to regress. It exists because the candidate is
  owned by a processing-thread *local* across a blocking call — a window no pool-level or
  counter-level accounting can observe.
- **`enter_in_flight`/`leave_in_flight`/`wait_for_in_flight` survives** as a one-off in
  `task_creator`, with no equivalent for the GPU executor or downgrade pools. Those two get
  correctness from ordering alone.
- **Step A11 is a cliff.** Eleven steps of unvalidated refactoring land before two queries actually
  run together. The harness exists from step A1, but only at N=1 until A11.
- **Step A12 is enormous** and will want splitting into 4–6 commits, which makes the real step
  count closer to 17.

## Risk register

| Risk | Mitigation |
|---|---|
| A7's ordering silently regresses | Add a debug assertion in `erase(query_id)` that the query is `closing` and all producers are quiesced |
| A11 surfaces issues not in the register | Land A2 (loud drops) first; run TSan from A5 onward, not only at A11 |
| A12 sprawls | Split it up front; F-group items are independently shippable and can land in parallel |
