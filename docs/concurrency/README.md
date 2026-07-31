# Concurrent Query Execution — Plan

Getting Sirius from "per-query data structures exist" to "N queries actually run together".

| Document | Purpose |
|---|---|
| [00-issue-register.md](00-issue-register.md) | **Read first.** Every remaining issue, with severity, file:line, and failure scenario. All options close all of it. |
| [**option-d-recommended.md**](option-d-recommended.md) | **The plan to execute.** B's step ordering with C's two structural changes folded in, after the scoping answers. |
| [option-a-surgical.md](option-a-surgical.md) | Fix each site in place with the narrowest change. No new abstractions. |
| [option-b-unified-gate.md](option-b-unified-gate.md) | One lifecycle gate + query-aware thread pool. Implements the `drain_and_wait(query_id)` idea. |
| [option-c-structural.md](option-c-structural.md) | Option B plus shared-ownership repositories, plan lifetime inversion, and fairness. |

A, B and C are kept as the reasoning trail — the trade-offs behind each decision in D are argued
there, and A or B remain the fallbacks if the `cucascade` change or the appetite for step 6 changes.

---

## Where the branch stands

The last six commits converted the per-query **data** plumbing — repository registry, task-creator
state, scan-manager state, completion handlers, query-indexed queues. That work is sound and none
of the options redo it.

What was not converted is the per-query **control** plumbing. Every teardown, drain, wait, validate
and error path is still a process-wide stop-the-world, invoked once per query.
`SiriusContext::query_lifecycle_mutex_` is the only thing keeping them from firing.

Two facts that shape all three plans:

- **The new per-query code has never run under concurrency.** All four new tests are
  single-threaded — `test_scan_manager_query_state.cpp:181` is titled *"concurrent queries do not
  collide on operator id"* and spawns zero threads. They prove state *partitioning*, not
  race-freedom.
- **Not everything is latent.** The downgrade monitor thread and the GPU manager loops are not
  slot-gated, so a handful of issues (B1, B2, B3, A7) are live on `dev` today.

## The 44 issues at a glance

| Group | Count | Theme |
|---|---|---|
| A — Cross-query blast radius | 10 | A per-query event stops a shared subsystem. 6 Critical. |
| B — Lifetime / use-after-free | 11 | Raw borrows outliving their owners; teardown ordering. |
| C — Deadlock | 4 | Lock-order inversion in `task_creator`; scan pool sized for one query. |
| D — Gaps in existing per-query tracking | 6 | The in-flight counter has a hole; lookahead serves only the oldest query. |
| E — Shared mutable config | 7 | `operator_params` and `Config::` static variables raced by `SET`. |
| F — Fairness / performance | 9 | Strict head-of-line blocking; global downgrade drain; shared prefetch pipeline. |
| G — Test infrastructure | 7 | No concurrency harness exists. |
| H — Dead code / drift | 11 | Uninitialized dead pointers, orphaned APIs, five drifted docs. |

Four issues are worth naming because they drive the choice between options:

- **A7 + B2 (live).** `run_mandatory_cleanup` drains *every* downgrade executor on *every* query
  end, cancelling other queries' pending spills — and `drain()` **restarts the processing thread as
  its last act**, so quiescence expires before `erase(query_id)` runs 20 lines later.
- **B1 (live).** TIER-2 downgrade pops a task off the shared queue into a processing-thread local,
  carries it across a *blocking* `reserve()` and a full conversion, then `~convertible_gpu_pipeline_task`
  pushes it **back** — potentially after that query's drain already ran.
- **B5.** The plan is destroyed *before* the drains. Pipelines hold non-owning references into it
  and outlive it. The code acknowledges this in a comment.
- **C1.** `stop_thread_pool()` holds `_global_state_mutex` across `_manager_thread.join()` while the
  manager thread needs that same mutex. Reachable on **every query completion**.

## Comparing the options

|  | A — Surgical | B — Unified gate | C — Structural |
|---|---|---|---|
| Steps | 12 (A12 splits → ~17) | 12 (B12 splits → ~17) | 15 (C15 splits → ~20) |
| Net LOC | ~1,400 | ~1,700 | ~2,600 |
| New files in `src/` | 0 | 2 | 3 |
| Touches `cucascade` | no | no | **yes** |
| Touches `bounded_thread_pool` | no | **yes** | **yes** |
| `data_repository` ownership | `unique_ptr` + raw borrows | `unique_ptr` + raw borrows | `shared_ptr` + eager `close()` |
| Teardown correctness rests on | a specific 6-step order | a gate (order-insensitive) | types (hazard unrepresentable) |
| Global downgrade drain | made per-query | made per-query | **deleted** |
| Plan vs cleanup order | fixed | fixed | **inverted** |
| Fairness (F1) | last step | last step | **before enablement** |
| First step that closes a Critical | 3 | 3 | 3 |
| Steps before 2 queries run together | 11 | 11 | 14 |
| Revertibility | high — every step independent | medium — B5→B6 is a fixed chain | low — C8/C9/C10 interlock |

All three share the same first three steps: harness + admission knob + scan pool sizing → make
dropped work loud → delete the three rogue `stop()` calls and convert the four `break`s. That is
where the highest-severity items are, and none of it is architecture-dependent — so it can start
immediately regardless of which option is chosen.

## Recommendation

**Option B**, with two amendments borrowed from C.

The reasoning:

- **A's cost is not its size, it's the twelve independent mechanisms.** Correctness of teardown
  would rest on an implicit six-step ordering in `run_mandatory_cleanup` plus a `closing` flag plus
  an in-flight counter plus a post-`reserve()` re-check. Nothing enforces any of it, and at N=1
  every test still passes if someone reorders it. For a system that already has *four* thread pools
  and *seven* enqueue points, one gate that every enqueue consults is worth its foundation cost.
- **B's `drain_and_wait(query_id)` is your idea, and the survey says it works** — 4 production
  `dispatch` sites, 4 `reserve` sites — **but only paired with the gate.** On its own it cannot see
  the TIER-2 window, because during it no job exists yet: the task is owned by a processing-thread
  local across a blocking `reserve()`. The gate covers that window; the pool covers everything
  already dispatched. Neither alone is sufficient, which is why they are one option and not two.
- **C's C8+C9 are genuinely valuable** — deleting the global downgrade drain rather than narrowing
  it is the difference between "concurrency works" and "concurrency works under memory pressure",
  which is the workload that matters. But C8 modifies the ownership model of the object every
  operator reads and writes through, **in a submodule**, and its bad failure mode is silently
  incomplete results rather than a crash. That is a lot to take on in the same project that is also
  removing the single-flight gate.

**Amendment to B:** fold in C8+C9 (shared-ownership repositories with `close()`, and deleting the
global downgrade drain). C13's fairness work is **not** needed — see the FIFO decision above.

### Revised after the scoping answers

The original write-up put C8+C9 as a follow-on rather than folding them in, on the grounds that
modifying the ownership model of the object every operator reads and writes through — in a
submodule — was too much to take on alongside removing the single-flight gate. Three of those
concerns turned out to be smaller than assumed:

- The cucascade surface is **two Sirius call sites, zero internal cucascade callers, and no
  subclasses** — not a sprawling refactor.
- The "silently incomplete results" failure mode is closed by change #3 in
  [the scope table](option-c-structural.md#cucascade-change-scope) — `add_data_batch` returns a
  status rather than silently accepting on a closed repository.
- At 3–7 concurrent queries, the global downgrade drain is no longer a tolerable tax. Every query
  end currently quiesces *every* downgrade executor on *every* memory space, and query ends become
  ~7× more frequent. C9 deletes it outright; B only narrows it to per-query.

So the recommendation is **Option B's step order, with C8/C9 inserted after B7 and C13 before
enablement** — which is, in practice, Option C. Take the sequencing discipline from B (harness and
the three rogue `stop()` deletions ship in the first three steps, before any foundation work) and
the structural changes from C.

The one part of C to still treat as optional is **C10's engine-ownership move**. Option B's step B9
already fixes the same bug (B5) by reordering destruction; C10 additionally makes
`StandaloneQueryScope` own the engine, which touches all five entry points and DuckDB's
`GlobalSourceState` interaction. Take the reorder; defer the ownership move unless the reorder
proves fragile.

If appetite for the submodule change disappears, **Option B unmodified** remains correct — you
keep the downgrade tax and per-query drain instead of deletion. **Option A** is still defensible
if the priority is landing something quickly with maximum per-step revertibility, but at 3–7
queries its twelve independent teardown mechanisms are the wrong trade.

## Answered scoping questions

### 1. Target concurrency: **3–7**, with **FIFO query order** as the fairness policy

The FIFO decision does most of the work here. `query_id` is already packed into the high bits of
the priority and the queue pops lowest-first — that *is* FIFO, so:

- **F1 is reclassified from defect to policy.** Oldest-first is defensible for a memory-constrained
  GPU engine: it drains queries rather than making partial progress on all of them, which bounds
  peak memory. No round-robin dispatch, no separate fairness index, **no fairness step**.
- **F4 is withdrawn.** A single FIFO prefetch pipeline is *consistent* with FIFO query order —
  query 7's prefetch queueing behind queries 1–6 is the chosen policy, not a bug.
- **What survives is small**: `schedule_lookahead` unconditionally serving
  `_query_task_global_states.begin()` is still wrong, not because oldest-first is wrong but because
  it will serve a finished-but-not-yet-reset query and touch a dead plan. Two-line fix, folded into
  step 4.
- **C4 stays** — the single per-GPU manager thread blocking on `make_reservation` and on the
  downgrade `.get()` stops *dispatch* entirely, so a newer query can't run even a task needing no
  new memory. Gets its own step.

One dependency worth stating: **FIFO is safe provided spilling works.** The livelock case is an
older query blocked on memory only a newer query would release by finishing. Steps 6–7 (shared
ownership + per-query downgrade) are therefore load-bearing for the fairness policy, not just for
teardown.

Scan pool sizing is comfortable: `num_threads` defaults to 8, so `num_threads + 7` takes the pool
from 9 to 15 threads. H8 (32-bit id wrap) stays low priority at this concurrency.

### 2. `cucascade` scope — **more than the pointer swap**

See [the cucascade scope section in Option C](option-c-structural.md#cucascade-change-scope). Short
version: the pointer swap is the easy half and it is the half that does not work on its own.
`close()` is the essential addition. Total surface is small — **zero** internal cucascade callers,
**two** Sirius production call sites, and `data_repository` has **no subclasses**.

### 3. Concurrent SQLLogic: **in scope, and the runner already exists**

The vendored DuckDB fully supports `concurrentloop` / `concurrentforeach`
(`duckdb/test/sqlite/sqllogic_parser.cpp:250`, `sqllogic_test_runner.cpp:1059`). Critically,
`ParallelExecuteLoop` constructs **a new `Connection` per parallel iteration** on its own
`std::thread` against the shared `DatabaseInstance` — i.e. exactly the shape needed: separate
connections, real threads, one shared `SiriusContext`.

Constraints to design around: nested parallel loops are rejected, and every command in the body
must return true from `SupportsConcurrent()`. So the runner is free; the work is authoring the
`.test` files and confirming Sirius's `ClientContext`-registered state behaves under the
per-iteration connections. This lands in step 11/14 (enablement) of whichever option is chosen and
replaces part of the bespoke C++ harness work — the C++ harness is still needed for the
subsystem-level tests in earlier steps, which have no SQL surface.
