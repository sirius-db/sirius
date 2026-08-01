# Option D — Recommended (consolidated)

> **This is the plan to execute.** It takes [Option B](option-b-unified-gate.md)'s gate +
> query-aware-pool mechanism and [Option C](option-c-structural.md)'s two structural changes, and
> orders them so the gate is load-bearing from step 2 rather than arriving as scaffolding.

**Scope decisions this plan is built on:**

| Decision | Consequence |
|---|---|
| Target concurrency **3–7** | Scan pool sized `num_threads + N`; head-of-line blocking on the per-GPU manager thread gets its own step |
| **Fairness = FIFO query order** (oldest query wins) | Register F1 is a *policy*, not a bug. No fairness step. See [below](#fairness-is-fifo-and-that-collapses-two-steps) |
| `cucascade` can take a PR | Shared-ownership repositories with `close()` are in scope; the global downgrade drain gets **deleted**, not narrowed |
| Concurrent SQLLogic in scope | The vendored runner already supports it; enablement uses it |

**Shape:** 14 steps (step 14 splits into 4–6 commits) · ~2,000 LOC net · 2 new files in `src/` ·
one `cucascade` PR + submodule bump.

Every step is one commit and leaves `pixi run make test` green.

---

## Ordering rationale

**The gate goes second and does real work immediately.** An earlier draft had it as a deliberate
runtime no-op — wire the checks, keep reporting `open`, prove nothing changed. That was
over-cautious and gave nothing to look at. Instead, step 2 wires it live and step 3 immediately
cashes it in: `drain_after_error`'s own comment explains that it interrupts the *shared* creation
queue precisely so that late `schedule()` calls from completion callbacks return false and get
dropped. The gate does exactly that, **per query, without touching a shared queue** — so the global
`stop_thread_pool()`/`interrupt()` pair can be deleted. That deletion is the gate's first visible
payoff, and it defuses register C1 as a side effect.

**Step 1 stays ahead of it** because it is pure deletion — three rogue `stop()` calls and four
`break`s — needing no new machinery and closing four Criticals. If the project stopped after one
commit, the bugs where a single query's failure hangs the whole engine would already be gone.

**Two things moved late.** The test harness (was step 1) is unusable until enablement — at
`max_concurrent_queries = 1` it can only re-prove serialization, which `test_query_lifecycle_slot.cpp`
already does. And the push-drop check (was step 2) is *better* late: before the gate, "push returned
false" cannot be distinguished from an intended teardown drop, so the check would have been a log
line. After the gate it becomes a real invariant — *any* push-false for a query the gate says is
`open` is a bug — and can `report_error` rather than just warn.

### Fairness is FIFO, and that collapses two steps

`query_id` is packed into the high bits of the scheduling priority and the queue pops lowest-first,
so every task of the older query outranks every task of a newer one. That **is** FIFO query order.
With FIFO accepted as the policy, register F1 stops being a defect:

- **F1's starvation is the policy.** Oldest-first is a defensible choice for a memory-constrained
  GPU engine — it drains queries rather than making partial progress on all of them, which bounds
  peak memory. No round-robin dispatch, no separate fairness index. The dedicated fairness step
  disappears.
- **F4 (single FIFO prefetch pipeline) stops being an issue.** One preparation thread, one prefetch
  thread, one rate limiter, strict FIFO — query 7's prefetch queueing behind queries 1–6 is
  precisely the chosen policy. Dropped from the plan.
- **What survives is small and moves into other steps.** `schedule_lookahead` taking
  `_query_task_global_states.begin()` unconditionally is still a bug — not because oldest-first is
  wrong, but because it will happily serve a query that is finished-but-not-yet-reset, touching
  operators of a dead plan. That becomes a two-line fix in step 4 (skip non-`open` queries, then
  take the oldest). The 31-bit priority mask and 32-bit id wrap (H8) become standalone hygiene.

> **One dependency to state explicitly.** FIFO is safe *provided the downgrade path works*. The
> livelock case is an older query blocked on memory that only a newer query could release by
> finishing — under FIFO the newer query never gets dispatched. The escape hatch is spilling, which
> means **steps 6 and 7 are load-bearing for the fairness policy**, not just for teardown
> correctness. If step 7 slips, revisit this.

---

## Steps

### Step 1 — Stop per-query failures from killing shared subsystems
**Closes:** A1, A3, A4, A10, global-halt half of A2

Pure deletion. No new abstractions, highest severity in the register.

- `task_scheduler::terminate_query` (`task_scheduler.cpp:174`): delete the `stop()`. Report to the
  handler, then `drain_query_tasks(query_id)`. Today one query's creation error stops every GPU
  executor and nothing ever calls `start()` again.
- Creation-worker catch (`task_creator.cpp:692`): delete the `stop()`.
- GPU executor catch (`gpu_pipeline_executor.cpp:427, 432`): delete `_task_creator->stop()` —
  `completion->report_error` on the next line already fails the correct query.
- All four `break`s (`:196, 239, 261, 295`) → report to *that task's* handler, then `continue`.
  Only channel/queue closure may break the loop.
- Wrap the `manager_loop` body in try/catch — it is a `std::thread` entry function that throws
  (`internal_exception` at `:139`) and calls throwing APIs, so an escape aborts the process.

**Tests:** inject a creation exception and a task exception; assert the scheduler and creator are
still running and a subsequent query completes normally.

---

> **Executed note (step 1 → step 2 reordering).** Step 1 deleted the `stop()` calls that used to
> pre-stop the task_creator, which made `drain_after_error → stop_thread_pool()` race a *live*
> manager thread and exposed the C1 deadlock — intermittently hanging the full suite in ~3 of 8
> runs. This plan's own dependency note said *"C1 must be fixed before A5/A6 are exercised, because
> both call `stop_thread_pool()` today"*, and scheduling C1 for step 4 violated it. **C1 and C2 were
> therefore pulled forward into step 2 and are no longer part of step 4.** The lesson generalizes:
> any step that stops pre-stopping a subsystem makes that subsystem's lifecycle races live in the
> same commit.

### Step 2 — `query_lifecycle_registry`, wired live (+ C1/C2 pulled forward)
**Closes:** C1, C2 — plus the gate that steps 3 and 7 build on

New `src/include/exec/query_lifecycle_registry.hpp` + `.cpp`. States `open → quiescing → closed`,
keyed by `query_id_t`, owned by `SiriusContext`. `begin_execution_window` opens;
`run_mandatory_cleanup` moves to `quiescing`, then `closed` after the drains.

Wire the check **live** into all seven enqueue points — a non-`open` query's enqueue becomes a
silent no-op:

| Enqueue point | File |
|---|---|
| `task_creator::schedule` / `schedule_lookahead` | `task_creator.cpp:397, 407, 456` |
| `task_scheduler::schedule` | `task_scheduler.cpp:112` |
| `itask_executor::schedule` | `task_executor.cpp:61` |
| **`~convertible_gpu_pipeline_task`** (the TIER-2 re-push) | `convertible_gpu_pipeline_task.hpp:88` |
| OOM reschedule | `gpu_pipeline_executor.cpp:423` |
| downgrade candidate, re-checked after `reserve()` | `downgrade_executor.cpp:306` |
| `bounded_thread_pool::dispatch` | added in step 5 |

Copy `exec::scoped_dispatcher`'s shape — after `request_stop()`, `enqueue()` is a silent no-op plus
`wait_for_all()`. That pattern is already shipped and tested in this tree.

**Tests — this is the "see it in action" step.** After `quiesce(A)`, a `schedule()` from a
completion callback for A is refused while B's still succeeds; a TIER-2 re-push for a quiescing
query is dropped instead of resurrecting the task. Add a debug-build assertion in
`push()`/`dispatch()` that the query is `open`, so a missed call site fails loudly rather than
silently.

---

### Step 3 — Delete the global interrupts; make completion and error paths per-query
**Closes:** A5, A6, D4 · defuses C1 from the per-query paths

The gate's first payoff. `drain_after_error`'s comment block explains that it interrupts the shared
creation queue so late pushes from completion callbacks are dropped — step 2 now does that per
query, so the shared interrupt is redundant.

- Delete `_task_creator->stop_thread_pool()` / `start_thread_pool()` from `wait_for_completion`
  (`:234, 255, 261`) and `drain_after_error` (`:197, 220`).
- `wait_for_completion(query_id)`: `_task_queue.size(query_index{q})` instead of the global
  `size()` (today query A throws *"task queue not empty"* because query B has work queued);
  `gpu_exec->wait_and_validate_empty(query_id)`.
- `drain_after_error(query_id)`: `_task_queue.drain(query_index{q})` in both places;
  `gpu_exec->drain_query_tasks(query_id)` instead of the unfiltered `drain_and_wait()`.
- New `itask_executor::wait_and_validate_empty(query_id)`; reserve the global `drain_and_wait()`
  for teardown. Add the missing null-pool guard it lacks but `drain_and_wait` has.
- Guard `*_ready_devices.begin()` against an empty vector.

**Tests:** query A completes while B has queued tasks — A must not throw, B must not lose work.
Same for A erroring.

---

### Step 4 — In-flight coverage and lookahead
**Closes:** D1, D2, D3, self-deadlock half of A2 · **prerequisite for step 5**

> C1 and C2 (the `_global_state_mutex`-across-`join()` deadlock and the unlocked `stop()`) moved to
> step 2 — see the note above. `_pool_lifecycle_mutex` already exists.

- Forbid `stop()` from a pool worker (assert; post a stop request).
- Move `enter_in_flight()` to immediately after the `get_query_task_global_state` lookup, so the
  manager thread's own `get_operator_for_next_task(node)` dereference is inside the counted region.
  Release on every early `continue`. Cover the key extractor's `request.node->type` dereference the
  same way.
- `schedule_lookahead`: skip non-`open` queries, then take the oldest — preserving FIFO while no
  longer serving a query whose plan is gone.

**Tests:** `drain_pending_tasks(Q)` returning implies no thread is dereferencing Q's operators;
stop/start hammer test.

---

### Step 5 — Query-aware `bounded_thread_pool` + `drain_and_wait(query_id)`
**Closes:** replaces `enter/leave/wait_for_in_flight`; foundation for step 7

Only safe now: step 3 removed the stop/start-per-query cycle that would otherwise destroy and
recreate the pool — and any per-query state in it — on every completion.

> **RESOLVED IN STEP 5 (success path).** `wait_and_validate_empty(query_id)` now uses
> `bounded_thread_pool::wait_for_query(query_id)` with no quiesce bracket, so a successful query
> completion no longer interrupts the shared queue and can no longer drop a co-tenant's in-transit
> task. `wait_and_drain_query(query_id)` (the ERROR path) deliberately keeps the bracket: a failing
> query can still have tasks the manager may pop at any moment, and there is a window between
> `pop()` and `slot::attach()` where a task belongs to neither the queue nor the per-query count.
> Joining the manager is what closes it, and the caller's next act is to destroy the plan, so
> "almost certainly quiesced" is not good enough there. Closing that last window needs the manager
> to publish its in-hand task's query before leaving the queue's lock — noted as follow-up work.
>
> **Original note, kept for context — A6 was only partly closed in step 3.** `wait_and_validate_empty(query_id)` and
> `wait_and_drain_query(query_id)` still bracket their work in `quiesce_manager()` /
> `resume_manager()`, because `manager_loop()` reserves a pool slot and then blocks in `pop()`: an
> idle manager holds an active slot forever, so `wait_all()` cannot return until the manager is
> interrupted and joined. That interrupt makes `push()` return false for the duration, so a
> co-tenant task **in transit** from the scheduler to a device queue can still be dropped. Step 3
> removed the whole-queue drain that destroyed co-tenants' *queued* work; this step must remove the
> in-transit drop by making the in-flight wait per-query, after which
> `test_task_scheduler.cpp`'s "wait_for_completion validates only its own query's queue" should
> regain the liveness assertion currently documented-but-not-asserted there.

- Tag at `reserve(query_id)`, **not** `dispatch` — that is what puts the manager thread's
  pre-dispatch dereference inside the counted region. Where the query is unknown until after
  `pop()`, use `slot::attach(query_id)`.
- `work_queue_`: `std::queue` → `std::list` + a `query_id → iterator` index so queued work is
  removable by key; erasing a queued item must release its slot.
- Per-query active counter + CV → `drain_and_wait(query_id)`; `close(q)`/`reopen(q)`.
- Convert the four production call sites (`task_creator.cpp:466/501`,
  `gpu_pipeline_executor.cpp:105/306`, downgrade TIER 1 `:238/251`, TIER 2 `:306/314`).
- `data_repository_manager_registry::get_all()` → `{query_id, manager}` pairs (TIER 1 has no query
  id today); `convertible_data` gains a `query_id()` virtual (TIER 2 is type-erased).
- Delete `task_creator`'s bespoke in-flight members.

**Tests:** extend `test_bounded_thread_pool.cpp` — `drain_and_wait(A)` must not wait on B's work;
a queued-then-drained item must release its slot. Convert call sites only after these pass.

---

### Step 6 — cucascade: shared-ownership repositories with `close()`
**Closes:** B3, B4, B9, H10 · **two commits: 6a cucascade PR + submodule bump, 6b Sirius side**

Surface is small: **zero** internal cucascade callers of `get_repository`/`get_repositories`,
**two** Sirius production call sites, and `data_repository` has **no subclasses**.

| # | Change |
|---|---|
| 1 | `_repositories`: `unique_ptr` → `shared_ptr`; `get_repository()` returns by value **under `_mutex`** — it takes no lock today, the only method in the class that doesn't (fixes B9); `get_repositories()` → `vector<shared_ptr>` |
| 2 | **`data_repository::close()`** — take `_mutex`, clear `_data_batches`, set `_closed`. This is what keeps GPU memory release *eager* while object lifetime goes lazy |
| 3 | `add_data_batch` returns a status when closed instead of `void` — otherwise a straggler's publish is a silent no-op, i.e. wrong results, which is worse than the crash it replaces |
| 4 | `clear_all_repositories()` → `close()` each, still counting leaked batches (preserves the drain diagnostic) |
| 5 | `manager::add_data_batch`'s `_repositories.at(...)` → graceful "dropped" path instead of `std::out_of_range` |

Plus: delete `for_each_repository` (zero callers), rename the `shared_data_repository` aliases.
Sirius side is `repository_wiring_materializer.cpp:71` and `downgrade_executor.cpp:226` plus the
registry; `port::repo` stays raw (the plan dies first, so shared ownership there extends lifetime
backwards).

**Tests:** a straggler publishing into a closed repository is observable, not a crash and not a
silent leak. Assert `log_pool_stats` returns to baseline at `QueryEnd` — i.e. that eager `close()`
really released the GPU memory.

---

### Step 7 — Delete the global downgrade drain
**Closes:** A7, B1, B2, D6, F8

With step 6 in place the downgrade executor cannot dangle, so the worst cross-query serialization
point goes away rather than being narrowed. At 3–7 queries this matters ~7× more than it does today.

- **Delete** `for (auto& executor : downgrade_executors_) executor->drain();` from
  `run_mandatory_cleanup`. Today it cancels *every* query's pending spills on *every* query end.
- `downgrade_request` gains a `query_id`; `drain(query_id)` cancels only that query's requests. Add
  the `_running` guard — `drain()` after `stop()` currently null-derefs and leaves a joinable thread.
  RAII-reset `_monitor_request_enqueued` on the request and on a failed push (it latches `true`
  forever today, killing automatic spilling for the rest of the process).
- The TIER-2 window closes three ways over: the gate (step 2) refuses the re-push, the
  post-`reserve()` re-check refuses to dispatch a candidate whose query left `open`, and shared
  ownership (step 6) makes it harmless if either were missed.
- TIER-1 sweep skips non-`open` queries.
- Lift the strict `_pool->wait_all()` serialization (F8).

**Tests:** TIER-2 resurrection stress test under ASan and TSan. Note this is reproducible
**single-query** — the downgrade monitor thread is not slot-gated, so it needs no concurrency
infrastructure.

---

### Step 8 — Repository teardown, plan lifetime, shutdown ordering
**Closes:** B5, B6, B7, B8, B10

- `run_mandatory_cleanup` becomes `gate.quiesce(q)` → `drain_and_wait(q)` on all pools → per-query
  queue drains → repo erase → `gate.close(q)`. Because the gate makes drains idempotent, this
  sequence is **no longer order-sensitive** — the key structural difference from a
  quiescence-only approach.
- Move engine/plan destruction after `run_mandatory_cleanup` (the reorder, not Option C's
  engine-ownership move). Today the plan is destroyed first and pipelines hold non-owning refs
  into it.
- `drop_query_runtime_state_best_effort` erases the repository registry — it currently doesn't, so
  a failed cleanup leaks the manager and all its GPU memory until `terminate()`.
- `terminate()`: stop creator/scan/downgrade **before** `task_scheduler_.reset()`; null
  `task_scheduler::_task_creator` and `gpu_pipeline_executor::_downgrade_executor` in the
  respective `stop()`s.
- Declare `task_scheduler_` last in `SiriusContext` (destroyed first); try/catch in
  `~SiriusContext`, which is `noexcept` but calls a throwing `terminate()`.

**Tests:** force `get_result()` to throw — the path that today destroys the plan with no drain
having run — and assert clean teardown under ASan. Test `initialize()` throwing after
`task_scheduler_` is constructed.

---

### Step 9 — Per-query telemetry and prefetch-cache safety
**Closes:** A8, B11

- `batch_telemetry_registry`: add `query_id` to placements and ports; `on_query_end(query_id)`
  filters both loops. Re-key `ports` on the existing `port::source_port_uuid` instead of a raw
  `data_repository*` — a key is not an owner, and address recycling silently matches stale entries.
- `prefetching_cache`: copy `file_entry*` out under the lock and drop the iterator before
  unlocking (`:311-322`, `:385-390`). Element pointers are stable; iterators are not.

---

### Step 10 — Head-of-line blocking on the per-GPU manager thread
**Closes:** C4

There is exactly one manager thread per GPU, and it performs a blocking `make_reservation` *and* a
blocking downgrade `.get()` while holding a reserved pool slot. Under FIFO this is partly intended
— the older query should win — but it stops *dispatch* entirely, so a newer query cannot run even a
task that needs no new memory.

Move reservation acquisition into the dispatched job, or `make_reservation_or_null` +
requeue-with-backoff.

> F4 (single FIFO prefetch pipeline) was originally paired here and is **dropped** — a FIFO prefetch
> queue is consistent with FIFO query order.

---

### Step 11 — Push-drop invariant
**Closes:** A9

Now a real invariant rather than scaffolding. `multi_index_priority_queue::push` returns false for
exactly one reason — `_active == false`, set only by `interrupt()`. After steps 1, 3 and 7, the
remaining `interrupt()` callers are genuine teardown. So:

- **Any push-false for a query the gate reports as `open` is a bug** → `report_error` on that
  query's completion handler.
- A push-false for a `quiescing`/`closed` query is the documented teardown contract → debug log
  only. (`~convertible_gpu_pipeline_task` returning a task to a closed queue is *intended*; a
  blanket error would fire spuriously.)
- Sites: `task_creator.cpp:404, 414, 456`; `task_scheduler.cpp:112`.
  `itask_executor::schedule:61-63` already logs and is the precedent.

Landing this before enablement means that when concurrency is switched on, lost work surfaces as an
error instead of a hang.

---

### Step 12 — Harness, admission knob, scan pool sizing
**Closes:** G1 (infra), G2, G3, G6, C3, H4

Everything here is an enablement prerequisite — none of it changes behaviour at N=1, which is why
it sits here rather than at the front.

- `test/cpp/utils/concurrent_query_harness.hpp`: N connections, release-gate start barrier,
  per-thread result/exception capture, deadline watchdog. Salvage the primitives from
  `test_query_lifecycle_slot.cpp`'s anonymous namespace (`wait_for_workers`, `async_query_result`,
  `scoped_blocking_window_log_sink`, `held_window_threads`) and the fixture-level
  `make_connection()` from the orphaned `test_transparent_execution.cpp`, then drop that file from
  the allowlist.
- Let a test opt out of `shared_env_listener` / `scoped_mgpu_env` env-pausing.
- Promote `k_max_concurrent_queries` → `scan_manager_config::max_concurrent_queries`, yaml-parsed,
  **default 1**, pool `num_threads + N`. Reject rather than warn past the cap. At 7 the pool goes
  9 → 15 threads.
- Add `sirius_config::max_concurrent_queries` (**default 1**); turn `acquire_query_lifecycle_slot`
  into a counting semaphore of that size. At 1 the behaviour is bit-identical to today, including
  the same-thread-reacquire error.
- Delete the never-locked `SiriusContext::mutex_`.

---

### Step 13 — Turn concurrency on and prove it
**Closes:** G1 (validation), G4, G5, G7; verifies F1

- Raise the default `max_concurrent_queries` above 1.
- Un-hide the `[concurrency]` suite; re-express `test_query_lifecycle_slot.cpp`'s serialization
  assertions against the new semantics; wire up the implemented-but-never-invoked
  `run_ac13_concurrent_logging`.
- **Concurrent SQLLogic:** the vendored runner supports `concurrentloop`/`concurrentforeach`, and
  `ParallelExecuteLoop` constructs a **new `Connection` per parallel iteration** on its own thread
  against the shared `DatabaseInstance` — exactly the shape needed. Constraints: nested parallel
  loops are rejected, and every command in the body must return true from `SupportsConcurrent()`.
  Add a concurrent TPC-H `.test`.
- Concurrent TPC-H at scale under memory pressure — the case steps 6, 7 and 10 were built for.
- **Assert FIFO holds:** an older query's tasks are dispatched ahead of a newer query's, and a
  newer query still completes (no permanent starvation once the older one drains).
- Full suite under ASan and TSan.

> **This is where unknown-unknowns surface** — the one step whose size cannot be estimated from the
> register. Budget accordingly. Step 11 is what makes it tractable.

---

### Step 14 — Config safety and hygiene
**Closes:** D5, E1–E7, F2, F3, F5, F6, F7, F9, H1, H2, H3, H5, H6, H7, H8, H9, H11
**Split into 4–6 commits; largely parallelisable and independent of the 6→7→8 chain.**

- **Config (E1–E3, E7):** move `operator_params` and the `Config::` static variables into per-`ClientContext`
  settings (the pattern already used for `enable_duckdb_fallback`), snapshotting by value into the
  plan at window begin. Guard the genuinely process-wide `LOG_*` trio. Make `original_config` a
  stack-local RAII.
- **E4:** `logical_plan_` → `shared_ptr<const LogicalOperator>` + an atomic `copy_unsupported_` latch.
- **E5, E6:** resolve-or-insert under one `unique_lock` keyed on `(catalog, schema, table)`; atomic
  packed pair or per-device `call_once` for `get_target_ctas`.
- **D5:** split `runtime_unavailable_` into "this query is corrupt" vs "the shared runtime is
  corrupt" — today one query's cleanup failure permanently disables GPU execution process-wide.
- **F2:** narrow the plan-time `SlotGuard` to a shared read of the pin registry.
- **F3, F5, F6, F7, F9:** live prefetch-epoch set; per-entry reader count replacing the
  `use_count() > 1` heuristic; reservation-driven sizing; remove `cudaDeviceSynchronize` and
  default-stream work; per-query dimension on the cucascade reservation/stream waits.
- **H8:** widen `query_id_t` to 64-bit, or keep 32-bit and document the wrap — the priority packing
  masks to 31 bits, so ordering silently inverts at 2³¹ and ids collide with live queries at 2³².
  With FIFO retained, the packing stays; only its robustness needs deciding.
- **H:** delete H1, H2, H3, H5, H6, H7, H11; rewrite the five drifted docs and add a doc for the
  per-query model (H9).

---

## Issue coverage

Every issue in [the register](00-issue-register.md) is closed exactly once. F4 is
**withdrawn** — a FIFO prefetch pipeline is consistent with FIFO query order — and F1 is reclassified
from defect to policy, verified in step 13.

| Group | Issue → step |
|---|---|
| A | A1·1 A2·1,4 A3·1 A4·1 A5·3 A6·3 A7·7 A8·9 A9·11 A10·1 |
| B | B1·7 B2·7 B3·6 B4·6 B5·8 B6·8 B7·8 B8·8 B9·6 B10·8 B11·9 |
| C | C1·2 C2·2 C3·12 C4·10 |
| D | D1·4 D2·4 D3·4 D4·3 D5·14 D6·7 |
| E | E1–E7·14 |
| F | F1·policy (verified 13) · F2·14 F3·14 F4·withdrawn F5·14 F6·14 F7·14 F8·7 F9·14 |
| G | G1·12,13 G2·12 G3·12 G4·13 G5·13 G6·12 G7·13 |
| H | H1–H3·14 H4·12 H5–H9·14 H10·6 H11·14 |

## Critical path

```
1 ─► 2 ─► 3 ─► 4 ─► 5 ─► 6 ─► 7 ─► 8 ─► 10 ─► 11 ─► 12 ─► 13
          ▲     └── 4 before 5: the pool can't hold per-query state while
          │        the stop/start cycle recreates it
          │
          └── 2 before 3: the gate is what replaces the global interrupt
                          6 before 7: deletion needs shared ownership

     9 ──┘  (independent)              14 ── (independent, splits)
```

Steps 9 and 14 are off the critical path and can be pulled forward or parallelised. Step 6a
(cucascade) can start any time and lands on its own cadence.

## Risks

| Risk | Mitigation |
|---|---|
| A missed gate check in step 2 | Debug-build assertion in every `push()`/`dispatch()` that the query is `open`; grep audit as a review checklist item. Step 11 turns the same condition into a runtime error |
| Step 5 destabilises four subsystems at once | Lands behind the N=1 default, so production behaviour is unchanged until step 13; extend `test_bounded_thread_pool.cpp` **before** converting call sites |
| Step 6's `close()` semantics wrong → silent wrong results | `add_data_batch` on a closed repository returns a status the caller must handle; assert-fail in debug; result-correctness test with a forced straggler |
| cucascade coordination stalls | 6a is independently shippable; if it slips, steps 7–13 proceed with a *narrowed* (per-query) downgrade drain and step 7 is revisited later |
| FIFO livelock: older query blocked on memory only a newer query would release | Depends on spilling, i.e. steps 6–7. If step 7 slips, re-evaluate the fairness policy |
| Step 13 surfaces issues not in the register | Step 11 (loud drops) is why this is tractable; run TSan from step 3 onward, not only at 13 |
