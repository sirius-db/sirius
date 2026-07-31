# Option B — Unified Lifecycle Gate + Query-Aware Thread Pool

> **Philosophy:** one invariant instead of twelve. Introduce a single per-query lifecycle authority
> (`open → quiescing → closed`) that **every** enqueue point consults, and teach
> `exec::bounded_thread_pool` to tag jobs with a `query_id` so it can offer
> `drain_and_wait(query_id)`. Once both exist, teardown becomes *idempotent and order-insensitive*:
> work cannot walk back across a drain, so the remaining per-query conversions are mechanical.

This is the option that implements your `drain_and_wait(query_id)` idea — **paired with the gate
that makes it correct**. On its own, `drain_and_wait` is a barrier that work walks straight back
across (see [B1 in the register](00-issue-register.md#group-b--lifetime-ownership-use-after-free)).

**Shape:** 12 steps · ~1,700 LOC net · 2 new files in `src/`
**Keeps:** `unique_ptr<data_repository>` (adds `close()` semantics), plan-before-cleanup ordering
fixed but not inverted.

---

## The two new mechanisms

### 1. `query_lifecycle_registry` — the gate

```
open  ──► quiescing ──► closed
        (drains run)   (erase safe)
```

Owned by `SiriusContext`, keyed by `query_id_t`. Every enqueue point checks it and no-ops when the
query is not `open`:

| Enqueue point | File |
|---|---|
| `task_creator::schedule` / `schedule_lookahead` | `src/creator/task_creator.cpp:397, 407, 456` |
| `task_scheduler::schedule` | `src/pipeline/task_scheduler.cpp:112` |
| `itask_executor::schedule` | `src/parallel/task_executor.cpp:61` |
| **`~convertible_gpu_pipeline_task`** (the TIER-2 re-push) | `src/include/data/convertible_gpu_pipeline_task.hpp:88` |
| `gpu_pipeline_executor`'s OOM reschedule | `src/pipeline/gpu_pipeline_executor.cpp:423` |
| `bounded_thread_pool::dispatch(query_id, ...)` | new |
| downgrade candidate acquisition (post-`reserve()` re-check) | `src/downgrade/downgrade_executor.cpp:306` |

**There is already a working precedent in the tree**: `exec::scoped_dispatcher` layered over a
shared `static_thread_pool`, one per query, where `request_stop()` clears pending work and makes
subsequent `enqueue()` a **silent no-op**, plus `wait_for_all()`. That is exactly "drain + gate",
and it is the design to copy.

### 2. Query-aware `bounded_thread_pool`

- Tag at `reserve(query_id)` — **not** at `dispatch` — so the manager thread's pre-dispatch
  dereference (register D1) falls inside the counted region. Where the query is unknown until
  after `pop()`, use `slot::attach(query_id)`.
- `work_queue_` becomes a `std::list` plus a `query_id → iterator` index so queued work is
  removable by key; erasing a queued item must release its slot.
- Per-query active counter + CV → `drain_and_wait(query_id)`.
- **Replaces** `enter_in_flight` / `leave_in_flight` / `wait_for_in_flight` entirely, and gives the
  GPU executor and downgrade pools the same guarantee `task_creator` has today.

**Feasibility notes from the call-site survey** — 4 `dispatch` sites, 4 `reserve` sites in
production, ~12 in tests:

| Site | `reserve` | `dispatch` | Query id available? |
|---|---|---|---|
| `task_creator::manager_loop` | `task_creator.cpp:466` | `:501` | yes (`request->query_id`) |
| `gpu_pipeline_executor::manager_loop` | `gpu_pipeline_executor.cpp:105` | `:306` | yes (`pipeline->get_query_id()`) |
| downgrade TIER 1 | `downgrade_executor.cpp:238` | `:251` | **no** — needs `get_all()` to return `{query_id, manager}` |
| downgrade TIER 2 | `:306` | `:314` | derivable, but type-erased — needs `query_id()` on `convertible_data` |

One prerequisite: **`task_creator` currently destroys and recreates its pool on every query
completion** (`stop_thread_pool`/`start_thread_pool` from `wait_for_completion` and
`drain_after_error`). Any per-query state stored in the pool evaporates there, so that cycle must
go first — which is also the fix for register C1.

---

## Step-by-step

### B1 — Concurrency harness + admission knob + scan pool sizing
**Closes:** G1, G2, G3, C3, H4

Identical to [Option A step A1](option-a-surgical.md#a1--concurrency-harness--admission-knob--scan-pool-sizing).
Harness, `max_concurrent_queries` config (default 1), semaphore slot, `mutex_` deleted.

---

### B2 — Make dropped work loud
**Closes:** A9

Identical to Option A step A2. Lands early for the same reason: until push failures are loud, every
Group A bug is an unexplained hang.

---

### B3 — Stop per-query failures from killing shared subsystems
**Closes:** A1, A3, A4, A10, global-halt half of A2

Identical to Option A step A3. Delete the three rogue `stop()` calls, convert the four `break`s to
report-and-`continue`, wrap `manager_loop` in try/catch.

> Deliberately kept ahead of the new mechanisms: these are pure deletions, they are the highest
> severity items in the register, and they make the gate's job smaller.

---

### B4 — Introduce `query_lifecycle_registry` (gate only, no behaviour change)
**Closes:** nothing yet — foundation

- New `src/include/exec/query_lifecycle_registry.hpp` + `.cpp`.
- `SiriusContext` owns it; `begin_execution_window` opens a query,
  `run_mandatory_cleanup` moves it `open → quiescing → closed`.
- Wire the **check** into all seven enqueue points listed above, but with the registry always
  reporting `open` for live queries — so this step is a no-op at runtime and provably safe.

**Tests:** unit tests for the state machine; a test that enqueue after `quiescing` is a no-op.

---

### B5 — Remove the creator's stop/start-per-query cycle; fix its lock ordering
**Closes:** C1, C2, D1, D2, self-deadlock half of A2 · **Prerequisite for B6**

- Delete `stop_thread_pool()`/`start_thread_pool()` from `wait_for_completion` and
  `drain_after_error` — the gate (B4) now does what the interrupt was doing, without touching a
  shared pool.
- Split the lifecycle mutex from `_global_state_mutex`; never hold either across `join()`.
- Forbid `stop()` from a pool worker.
- Move `enter_in_flight()` before `get_operator_for_next_task(node)`; cover the key extractor.

---

### B6 — Query-aware `bounded_thread_pool` + `drain_and_wait(query_id)`
**Closes:** replaces `enter/leave/wait_for_in_flight`; foundation for B7/B8

- `reserve(query_id)` / `slot::attach(query_id)`; `std::list`-backed work queue with a per-query
  index; per-query active counter + CV; `drain_and_wait(query_id)`; `close(q)`/`reopen(q)`.
- Convert all four production call sites; delete `task_creator`'s bespoke in-flight members and
  route `drain_pending_tasks(query_id)` through the pool.
- `data_repository_manager_registry::get_all()` → `{query_id, manager}` pairs.
- `convertible_data` gains a `query_id()` virtual.

**Tests:** extend `test_bounded_thread_pool.cpp` with per-query drain semantics — including that
`drain_and_wait(A)` does not wait on B's work, and that a queued-then-drained item releases its slot.

---

### B7 — Per-query completion and error paths
**Closes:** A5, A6, D4

Now mechanical, because the gate makes drains idempotent:

- `wait_for_completion(query_id)` → `size(query_index{q})`, `wait_and_validate_empty(query_id)`,
  no pool cycling.
- `drain_after_error(query_id)` → `drain(query_index{q})`, `drain_query_tasks(query_id)`.
- Guard `*_ready_devices.begin()`.

---

### B8 — Per-query downgrade
**Closes:** A7, B1, B2, B3, D6

The payoff step — the gate and the pool together close the race that neither closes alone:

- `downgrade_request` gains a `query_id`; `drain(query_id)` cancels only that query's requests.
- Split `drain()` into `quiesce()`/`resume()`; add the `_running` guard; RAII-reset
  `_monitor_request_enqueued`.
- **The TIER-2 window closes twice over:** the gate refuses `~convertible_gpu_pipeline_task`'s
  re-push for a non-`open` query, *and* the post-`reserve()` re-check refuses to dispatch a
  candidate whose query has left `open`. The gate handles the processing-thread-local window that
  pool accounting cannot see; `drain_and_wait(q)` handles everything already dispatched.
- TIER-1 sweep skips non-`open` queries.

**Tests:** the TIER-2 resurrection stress test from the register, under ASan and TSan.

---

### B9 — Repository teardown, plan lifetime, shutdown ordering
**Closes:** B4, B5, B6, B7, B8, B9, B10

- `run_mandatory_cleanup` becomes: `gate.quiesce(q)` → `drain_and_wait(q)` on all four pools →
  per-query queue drains → repo erase → `gate.close(q)`. Because the gate makes every drain
  idempotent, **this sequence is no longer order-sensitive** — the key structural difference
  from Option A.
- Move engine/plan destruction after `run_mandatory_cleanup`.
- `drop_query_runtime_state_best_effort` erases the repository registry.
- `terminate()` stops producers before `task_scheduler_.reset()`; fix member declaration order;
  try/catch in `~SiriusContext`.
- Take `_mutex` in `get_repository`.

---

### B10 — Per-query telemetry and prefetch-cache safety
**Closes:** A8, B11

Identical to Option A step A10.

---

### B11 — Turn concurrency on and prove it
**Closes:** validation half of G1, G4, G5, G7

Identical to Option A step A11 — but arriving with a single invariant to reason about when
something does go wrong, and with `drain_and_wait(query_id)` available as a debugging primitive.

---

### B12 — Fairness, config safety, and hygiene
**Closes:** D3, D5, E1–E7, F1–F9, H1–H3, H5–H11

Identical to Option A step A12. Split into 4–6 commits.

---

## Pros

- **One invariant, one place to look.** "Is it safe to erase this query's repositories?" has a
  single answer: the gate says `closed` and every pool reports `drain_and_wait(q)` complete. New
  call sites inherit the guarantee instead of re-deriving it.
- **Teardown becomes order-insensitive.** Option A's correctness depends on a specific six-step
  sequence in `run_mandatory_cleanup`. Here, a gated enqueue point cannot re-arm work after a
  drain regardless of ordering, so a future reorder is not a latent UAF.
- **Implements your `drain_and_wait(query_id)` idea properly.** It uniformly replaces
  `enter/leave/wait_for_in_flight` and extends the same guarantee to the GPU executor and
  downgrade pools, which have nothing equivalent today.
- **Tagging at `reserve()` fixes register D1 for free** — the manager-thread dereference that the
  current in-flight counter misses.
- **Copies a design that already works in this tree.** `scoped_dispatcher`'s "after `request_stop()`,
  enqueue is a silent no-op" is the same shape, already shipped and already tested.
- **Better debugging story.** `drain_and_wait(q)` plus gate state is directly inspectable when a
  query hangs; Option A gives you four separate mechanisms to inspect.

## Cons

- **Touches a primitive shared by four subsystems.** A bug in `bounded_thread_pool` takes out task
  creation, GPU dispatch and downgrade simultaneously. Step B6 needs disproportionate review and
  test attention relative to its LOC.
- **Two foundation steps (B4, B6) deliver no user-visible fix.** B4 is deliberately a runtime
  no-op. That is uncomfortable if the project gets cut short — Option A's steps each close real
  bugs from step 3 onward.
- **The gate is only as good as its coverage.** A missed enqueue point is a silent hole, and the
  compiler will not find it. Needs an explicit audit (and ideally a debug assertion in `push()`
  that the query is `open`).
- **`std::list` + index in the work queue** is more machinery than the current `std::queue`, with
  slot-release bookkeeping on the erase path that must be exactly right.
- **Ordering constraint:** B5 must precede B6 (the pool cannot hold per-query state while it is
  destroyed and recreated every query), which makes the middle of the plan a fixed sequence rather
  than parallelisable work.
- **Still does not fix the register's structural items.** `unique_ptr<data_repository>` with raw
  borrows survives; correctness still rests on quiescence, just centrally enforced quiescence.
  The global `executor->drain()` can be made per-query but not *deleted*.

## Risk register

| Risk | Mitigation |
|---|---|
| A missed gate check | Debug-build assertion in every `push()`/`dispatch()` that the query is `open`; grep audit as a checklist item in B4's review |
| B6 destabilises all four subsystems at once | Land B6 behind the N=1 default so production behaviour is unchanged until B11; extend `test_bounded_thread_pool.cpp` before converting call sites |
| B4/B6 deliver nothing shippable | Sequence B1–B3 first (harness + the three rogue `stop()` deletions) so real value lands before the foundation work |
