# Concurrent Query Execution — Execution Summary

> **STATUS ADDENDUM (2026-08-15, not by the original author).** This summary describes the state
> at the end of the `concurrency3` work and is preserved as written. Much of what it lists as
> remaining has since been fixed on `integration/concurrency-full`: the single-flight mutex is
> now a counted slot pool; the A7/B1/B2/B5/B9/C4/D3/D5/D6 items, the full E group, and F1/F3/F4
> are closed; and the first G-group harness pieces (G1/G2) exist. See
> [01-bringup-triage.md](01-bringup-triage.md) for the current
> per-item status with evidence, and `docs/super-sirius/concurrency-model.md` for the
> as-built architecture documentation. Nothing below this line has been edited.

What was built, what was not, and what to do next.

**Branch:** `concurrency3` (base `a35be838`) · **10 commits** · 32 files, +3142 / −239
**Validation:** full C++ suite (2255 cases, ~32.5M assertions) green, 3 consecutive runs per step
**Not pushed.** Nothing here has been reviewed or merged.

> **Branch note:** this work is on `concurrency3`, not `concurrency2`. The branch was switched
> before the first commit. `concurrency2` still points at the base commit `a35be838`.

---

## 1. Where the project stands

The [issue register](00-issue-register.md) listed 44 issues. **23 are closed.** The single-flight
mutex is still in place, so Sirius still executes one query at a time — but the control plumbing
behind it is now per-query rather than process-wide, which was the actual blocker.

| Group | Closed | Remaining |
|---|---|---|
| A — cross-query blast radius | A1 A2 A3 A4 A5 A9 A10 | A6 (partial), A7, A8 |
| B — lifetime / use-after-free | B6 B7 B8 B10 B11 | B1 B2 B3 B4 B5 B9 |
| C — deadlock | C1 C2 C3 | C4 |
| D — tracking gaps | D1 D2 D3 D4 | D5 D6 |
| E — shared config | — | E1–E7 |
| F — fairness / perf | F1 (policy), F4 (withdrawn) | F2 F3 F5 F6 F7 F8 F9 |
| G — test infra | — | G1–G7 |
| H — hygiene | H4 H8(partial) H10 | rest |

### Commits

| Commit | Step | Closes |
|---|---|---|
| `f966ae00` | 1 — stop per-query failures killing shared subsystems | A1, A3, A4, A10, A2(half) |
| `c19f480e` | 2 — per-query lifecycle gate + the C1 deadlock it exposed | C1, C2 |
| `7bdf050e` | 3 — per-query completion and error paths | A5, D4, A6(partial) |
| `8a09bebc` | — crash handler bounded (unplanned, see §3) | — |
| `5d93da7f` | 4 — in-flight coverage and lookahead | D1, D2, D3, A2(half) |
| `6d22cb6b` | 5 — query-aware thread pool with per-query waits | A6 (success path) |
| `48c114bc` | 8+11 — shutdown ordering, failed-cleanup leak, loud drops | A9, B6, B7, B8, B10 |
| `a9a85ed4` | 9+12 (partial) — prefetch-cache UAF, real `max_concurrent_queries` | B11, C3 |
| `ff33f389`, `e5078162` | plan-doc corrections | — |

### The two mechanisms everything else hangs off

**`exec::query_lifecycle_registry`** (`src/include/exec/query_lifecycle_registry.hpp`) — a per-query
`open → quiescing → closed` gate consulted at six enqueue points, including
`~convertible_gpu_pipeline_task`'s TIER-2 re-push. It replaces "interrupt the shared queue", which
refused *every* query's pushes at once. An unknown query id **accepts** work on purpose: a missed
`open_query()` would otherwise silently stop a query from scheduling, i.e. a hang — the exact
failure class the gate exists to remove.

**Query-aware `bounded_thread_pool`** — slots carry an optional `query_id` via `slot::attach()`,
applied once the task's query is known rather than at `reserve()`. That distinction is load-bearing:
every manager loop reserves a slot and *then* blocks in `pop()`, so an idle manager holds a slot
forever and `wait_all()` can never return. Untagged slots are invisible to per-query waits, which is
what makes `wait_for_query(q)` / `drain_and_wait(q)` usable at all.

---

## 2. What was attempted and backed out

**Steps 6 + 7 — shared-ownership repositories and deleting the global downgrade drain.**

Implemented in full, then reverted. The work is preserved in named stashes, not discarded:

```
cucascade/   stash@{0}  step6-cucascade-shared-ptr-repositories
sirius/      stash@{0}  step6b-7-sirius-side
```

**Symptom:** deterministic `SIGSEGV` at the 6th test (`pin_table compression - result equality vs
uncompressed pin`), 3/3 runs and in isolation, in
`sirius_physical_ungrouped_aggregate_merge::get_next_task_input_data()` on a `task_creator` pool
worker. Faulting instruction `mov 0x8(%rcx),%rdi`; the source line is
`ports.begin()->second->repo->pop_next_data_batch()`.

**Ruled out:** not pre-existing (verified by stashing both halves and rebuilding at step 5); not an
unhooked virtual override (nothing derives from `data_repository`); not an out-of-range partition
(`pop_next_data_batch` range-checks and throws).

**Still open**, in suspicion order:
1. `get_repository()` returning `shared_ptr` **by value** while `materialize_repository_wiring` takes
   `.get()` on the temporary — the one lifetime change on the exact pointer that crashes.
2. The manager's `add_data_batch` moving per-repository calls outside `_mutex`.
3. `close()` leaving `_data_batches` size 0 rather than one empty partition, changing
   `num_partitions()` for anything reading it after close.

**Consequence:** the global `executor->drain()` is still in `run_mandatory_cleanup`. It remains the
worst cross-query serialization point — every query end cancels every other query's pending spills
and stops/joins/restarts each downgrade executor's processing thread.

---

## 3. Things found along the way that were not in the plan

**The crash handler turned every crash into an unkillable spin** (`8a09bebc`).
`segfault_handler` called `backtrace_symbols()`, which goes through `_dl_addr()` and takes the
dynamic loader lock; `__cxa_demangle()` and the log flush allocate. If the crash happened while any
thread held the loader, malloc-arena or logging lock, none of those *fail* — they spin forever. A
dead process then looks alive: 100% CPU, no output, survives SIGTERM.

This is worth calling out because it **cost roughly an hour of misdiagnosis**. It disguised an
ordinary `task_creator` deadlock as a live spinning process, and every symptom (running thread,
`wchan=0`, all other threads idle) pointed the wrong way. The handler is now bounded by a 10s
`SIGALRM` covering the whole handler, and emits the async-signal-safe `backtrace_symbols_fd()` output
*first* so a usable stack survives even if the pretty path blocks. It found the step-5 pool bug
within minutes of landing.

**The plan's own dependency note was violated, and it bit immediately.** The register said *"C1 must
be fixed before A5/A6 are exercised, because both call `stop_thread_pool()` today"* — and C1 was
scheduled for step 4 while step 1 made those paths live. Step 1 removed the `stop()` calls that used
to pre-stop the creator, so `drain_after_error → stop_thread_pool()` began racing a live manager
thread and deadlocked ~3 of 8 full-suite runs. C1/C2 were pulled forward into step 2.

Generalizable lesson, now recorded in the plan: **any step that stops pre-stopping a subsystem makes
that subsystem's lifecycle races live in the same commit.**

**A 9.4 GB core dump was accidentally committed** by a `git add -A` during debugging, taking `.git`
to 3.2 GB. Caught before pushing; the commit was rewritten, the blob garbage-collected, and
`core`/`core.*` added to `.gitignore`.

---

## 4. Known gaps in what *was* delivered

Stated explicitly so they are not mistaken for finished work.

**A6 is only partly closed.** `wait_and_drain_query` (the error path) still brackets its work in
`quiesce_manager()`/`resume_manager()`, which interrupts the shared queue and can drop a co-tenant's
**in-transit** task. The success path no longer does. The bracket cannot simply be removed — a
failing query can still have tasks the manager may pop at any moment, and between `pop()` and
`slot::attach()` a task belongs to neither the queue nor the per-query count. Joining the manager is
what closes that window, and the caller's next act is destroying the plan.

**The concurrency test harness does not exist.** Every test added here is either single-threaded or
exercises one subsystem with synthetic query ids. Nothing runs two *queries* end to end. The four
pre-existing "query_state" tests would still pass with most of the original register bugs present.

**One unexplained intermittent failure**: `query lifecycle slot is released for an unconsumed result`
→ "watchdog child exited abnormally", seen once in ~16 runs and not reproduced in follow-ups. It is a
forked-child test, and the `DISABLE_SIRIUS_SIGNAL_HANDLER=1` debugging harness changes how a crashing
child dies, so it may be an artifact of that.

**A separate pre-existing issue**: running only `[integration]` reproducibly fails
`test_pin_table_mvcc_foundation.cpp:201` (`probe.counts.size() >= 2` yields 1), while the same test
passes in full-suite runs. Order-dependent test isolation, unrelated to this work.

---

## 5. Recommended next steps, in order

### 1. Land the concurrency harness before any more engine work (G1–G3)
Everything above was validated at N=1. The per-query code has still **never run under real
concurrency**, so the confidence from "2255 tests green" is narrower than it looks. Salvage the
primitives trapped in `test_query_lifecycle_slot.cpp`'s anonymous namespace (start gate,
`async_query_result`, `scoped_blocking_window_log_sink`) into `test/cpp/utils/`, and let a test opt
out of `shared_env_listener`'s env-pausing.

Cheap and high-value: `concurrentloop` already works in the vendored DuckDB, and
`ParallelExecuteLoop` constructs a **new `Connection` per parallel iteration** on its own thread
against the shared `DatabaseInstance` — exactly the shape needed.

### 2. Finish steps 6+7 with one instrumentation point
Do **not** resume by re-reading the diff. Add a temporary print of `ports.size()` and
`repo == nullptr` immediately before the deref in `get_next_task_input_data`. That single data point
separates all three hypotheses in §2; none of the black-box debugging did. Then unstash.

This matters more than its position suggests: **FIFO fairness depends on spilling working**, because
the livelock case is an older query blocked on memory only a newer query could release by finishing.
Steps 6/7 are what make the downgrade path per-query.

### 3. Close the error-path window (A6 remainder)
Have the manager publish its in-hand task's query *before* the task leaves the queue's lock — e.g. a
per-query "in-hand" counter incremented under the pop, so `pop()`-to-`attach()` is covered. Then
`wait_and_drain_query` can drop its quiesce bracket like the success path did.

### 4. Per-query batch telemetry (A8)
`batch_telemetry_registry::on_query_end()` takes no query id: it consumes every live placement across
all 16 shards and clears `impl_->ports`, so one query's end silently truncates every co-tenant's
telemetry. Add `query_id` to placements and ports, re-key `ports` on the existing
`port::source_port_uuid` instead of a raw `data_repository*` (a key is not an owner — address
recycling silently matches stale entries), and add an `on_query_end(query_id)` overload.

### 5. Head-of-line blocking (C4)
One manager thread per GPU performs a blocking `make_reservation` *and* a blocking downgrade `.get()`
while holding a reserved slot, so one query's memory-hungry task blocks every other query's dispatch
to that GPU. At the 3–7 target this is the throughput ceiling. Move reservation acquisition into the
dispatched job, or `make_reservation_or_null` + requeue-with-backoff.

### 6. Shared mutable config (E1–E7) — required before removing the mutex
`operator_params` is one non-atomic struct per `DatabaseInstance` written by ~20 `SET` callbacks and
read mid-plan and mid-execution; **the slot is the only thing serializing those writes today**. The
`Config::` static variables have no guard at all, and `EXPRESSION_EVALUATOR_STRATEGY` is read as a *default
argument* on every `expression_evaluator` construction — so one connection's `SET` can change
strategy between two operators of another connection's plan. Move both into per-`ClientContext`
settings and snapshot into the plan at window begin.

### 7. Only then: raise `max_concurrent_queries` and remove the single-flight mutex
The knob and the pool sizing are in place. Raise it, run the harness from step 1 under ASan and TSan,
and expect to find things — 23 issues closed is not 44.

---

## 6. Practical notes for whoever picks this up

- **`DISABLE_SIRIUS_SIGNAL_HANDLER=1`** gives real core dumps fast. Documented in
  `docs/super-sirius/debugging.md`. Invaluable for this work.
- **Put `timeout` *inside* `pixi run`**, not outside — `timeout -s ABRT 600 pixi run ...` signals
  pixi's wrapper and you get a useless Rust backtrace. `pixi run bash -c 'exec timeout -s ABRT ...'`
  signals the test binary.
- **A full suite run is ~345s.** Anything much longer is a hang, not slowness. Watch the
  `[N/2255]` progress counter, and never pipe it to `tail` while diagnosing — `tail` emits nothing
  until its input closes, which makes a live run look identical to a dead one. That mistake cost
  real time here.
- **Validate every step with ≥3 consecutive full-suite runs.** Two of the bugs in this work appeared
  in roughly 1 run in 3.
