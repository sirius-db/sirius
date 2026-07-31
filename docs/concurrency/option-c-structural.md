# Option C — Structural

> **Philosophy:** remove whole *classes* of bug rather than instances. Everything in
> [Option B](option-b-unified-gate.md), plus three structural changes that make the remaining
> hazards unrepresentable: shared ownership of repositories with **eager content release**, plan
> lifetime inverted so the plan outlives every drain, and per-query fairness so concurrency
> actually pays.

The test: after Option C, can a new contributor introduce a teardown UAF by writing ordinary code?
Under A and B the answer is yes (miss an ordering, miss a gate check). Under C the dangerous
pointer types no longer exist.

**Shape:** 15 steps · ~2,600 LOC net · 3 new files in `src/`, changes in `cucascade/`
**Changes:** `shared_ptr<data_repository>` + `close()`, plan destroyed after cleanup, round-robin
fairness, config moved to per-connection.

---

## The three structural changes

### 1. Shared ownership with eager content release

The register's verdict on plain `shared_ptr<data_repository>` is **"partially — worth doing, but
not the fix on its own"**, and it has a real cost: one straggler task keeping one repository alive
keeps **every un-consumed batch in it** alive, and those batches hold GPU memory. That would move
memory release from a deterministic point (`erase(query_id)`) to "whenever the last borrower drops",
which is unbounded on the OOM-reschedule path — and it would break the `log_pool_stats` leak
signature (`QueryEnd allocated != QueryBegin allocated`) that leak detection currently relies on.

So Option C **splits object lifetime from content lifetime**:

```cpp
// shared ownership of the OBJECT — no dangling, ever
std::map<operator_port_key, std::shared_ptr<data_repository>> _repositories;

// deterministic release of the CONTENTS
void data_repository::close();   // takes _mutex, clears _data_batches (GPU memory back NOW),
                                 // sets _closed; later add_data_batch() is a no-op returning
                                 // "dropped" so a straggler's publish is observable, not silent
```

`registry.erase(query_id)` calls `close()` on every repository — **still reporting leaked batch
counts, preserving the diagnostic** — then drops the map entry. Repository *objects* die lazily;
repository *memory* dies eagerly. This is what lets the global `executor->drain()` at
`sirius_context.cpp:322-324` be **deleted** rather than merely made per-query.

Also in scope: `gpu_pipeline_task::_data_repos` and `convertible_data_batch_provider::_repo`
become `shared_ptr`; `port::repo` stays raw (the plan dies first, so shared ownership there would
extend lifetime *backwards*); `batch_telemetry_registry::ports` re-keys on the existing
`port::source_port_uuid` (a key is not an owner — address recycling silently matches stale entries).

`data_batch` itself needs **no change**: it is already `shared_ptr` throughout, non-owning
references are `weak_ptr` or by-id, and there are zero raw `data_batch*` in `src/`.

#### cucascade change scope

The blast radius is much smaller than "modifying a submodule" suggests:

- **Zero internal cucascade callers** of `get_repository` / `get_repositories` — only the class
  itself and its own test file (`cucascade/test/data/test_data_repository_manager.cpp`, ~20 sites,
  all mechanical `auto& repo =` → `auto repo =`).
- **Two Sirius production call sites**: `src/pipeline/repository_wiring_materializer.cpp:71` and
  `src/downgrade/downgrade_executor.cpp:226`.
- **`data_repository` has no subclasses**, so changing virtual signatures is safe.

Five changes, of which the pointer swap is only the first:

| # | Change | Why |
|---|---|---|
| 1 | `_repositories`: `unique_ptr` → `shared_ptr`. `get_repository()` returns `shared_ptr` **by value under `_mutex`**; `get_repositories()` → `vector<shared_ptr>`. | Removes the dangling borrow. Also fixes register **B9** for free — `get_repository` today returns a reference into the map and is the *only* method in the class that takes no lock. |
| 2 | **`data_repository::close()`** — take `_mutex`, clear `_data_batches`, set `_closed`. | **The essential part.** `_data_batches` is `vector<vector<shared_ptr<data_batch>>>`; dropping it is what returns GPU memory. Without this, shared ownership means one straggler task keeps *every* un-consumed batch alive and memory release becomes non-deterministic. |
| 3 | `add_data_batch` returns a status instead of `void` when closed. | Without it, a straggler's publish is a silent no-op — memory-safe but **wrong results**, which is worse than the crash it replaces. Currently `virtual void` with no overrides, so this is safe to change. |
| 4 | `clear_all_repositories()` → `close()` each (still counting leaked batches) then drop entries. | Preserves the leaked-batch diagnostic, which is how "operator X didn't drain port Y" is currently detected. |
| 5 | `data_repository_manager::add_data_batch` uses `_repositories.at(...)`, which throws `std::out_of_range` once the entry is erased. | Needs a graceful "dropped" path rather than an exception on a teardown race. |

Optional hygiene while in there: delete `for_each_repository` (zero callers) and rename the
`shared_data_repository` / `shared_data_repository_manager` aliases (register H10), which claim
shared ownership where there is none — in exactly this code.

**Effort:** ~150 LOC production + mechanical test updates on the cucascade side; ~80 LOC on the
Sirius side across the two call sites and the registry. The cucascade half is independently
shippable — land it, bump the submodule, then land the Sirius half.

### 2. Plan lifetime inversion

Today `cleanup_internal` destroys `sirius_engine` — and therefore `sirius_owned_plan` — *before*
`run_mandatory_cleanup` runs. Every `sirius_pipeline` holds non-owning refs into that freed plan
(`optional_ptr` source/sink, `reference_wrapper` operators) and outlives it via the task global
state. `~gpu_pipeline_task` → `mark_task_completed()` → `notify_downstream_pipelines()` walks all
of them; so does `index_keys_for` from inside `drain_query_tasks`. The code acknowledges this in a
comment ("stop touching a plan that has died").

Option C makes `StandaloneQueryScope` own the engine, so `finish()` orders: drain → repo erase →
**then** engine destruction. After this, "a pipeline referencing a dead plan" is not a state the
system can reach, and the whole `drain_query_tasks`-must-not-run-completion-side-effects problem
evaporates.

### 3. Fairness

`query_id` is currently packed into the **high** bits of the priority and the queue pops
lowest-first, so every task of query 1 outranks every task of query 2. Query 2 runs only when
query 1 has nothing dispatchable — and if query 1 is blocked on memory that only query 2 could
release by finishing, it livelocks. Option C makes the query bits a **fairness index**
(round-robin across live queries at dispatch) rather than a priority prefix, and widens
`query_id_t` to 64-bit so it stops being packed into a signed 31-bit field at all.

---

## Step-by-step

### C1 — Concurrency harness + admission knob + scan pool sizing
**Closes:** G1, G2, G3, C3, H4 · Same as Option A step A1.

### C2 — Make dropped work loud
**Closes:** A9 · Same as Option A step A2.

### C3 — Stop per-query failures from killing shared subsystems
**Closes:** A1, A3, A4, A10, global-halt half of A2 · Same as Option A step A3.

### C4 — Introduce `query_lifecycle_registry` (gate only)
**Closes:** foundation · Same as Option B step B4.

### C5 — Remove the creator's stop/start cycle; fix lock ordering
**Closes:** C1, C2, D1, D2, self-deadlock half of A2 · Same as Option B step B5.

### C6 — Query-aware `bounded_thread_pool` + `drain_and_wait(query_id)`
**Closes:** replaces `enter/leave/wait_for_in_flight` · Same as Option B step B6.

### C7 — Per-query completion and error paths
**Closes:** A5, A6, D4 · Same as Option B step B7.

---

### C8 — Shared-ownership repositories with `close()` *(new to Option C)*
**Closes:** B3, B4, B9, H10 · **Prerequisite for C9's deletion**

- `cucascade`: `_repositories` → `map<key, shared_ptr<data_repository>>`; `get_repository()`
  returns `shared_ptr` **by value under `_mutex`** (which also fixes B9 for free);
  `get_repositories()` → `vector<shared_ptr<...>>`; delete the zero-caller `for_each_repository`.
- Add `data_repository::close()` as described above.
- `registry.erase(query_id)` → `close()` every repository (still counting leaked batches) then
  drop the entry.
- `gpu_pipeline_task::_data_repos` and `convertible_data_batch_provider::_repo` → `shared_ptr`.
- Rename the `shared_data_repository` / `shared_data_repository_manager` aliases, which currently
  claim shared ownership where there was none.

**Tests:** a straggler task publishing into a closed repository must be observable (dropped +
logged), not a crash and not a silent leak. Assert `log_pool_stats` returns to baseline at
`QueryEnd` — i.e. that eager `close()` really did release the GPU memory.

---

### C9 — Delete the global downgrade drain *(new to Option C)*
**Closes:** A7, B1, B2, D6, F8

With C8 in place the downgrade executor can no longer dangle, so the worst cross-query
serialization point in the system goes away entirely rather than being narrowed:

- **Delete** `for (auto& executor : downgrade_executors_) executor->drain();` from
  `run_mandatory_cleanup`.
- `downgrade_request` gains a `query_id`; the sweep skips non-`open` queries; `_running` guard;
  RAII-reset `_monitor_request_enqueued`.
- TIER-2 re-push is gated (never resurrects) *and* harmless if it did (shared ownership).
- Optionally lift the strict `_pool->wait_all()` serialization (F8).

> This is the step that makes concurrency actually *concurrent* under memory pressure. Under
> Options A and B, every query end still quiesces every downgrade executor.

---

### C10 — Plan lifetime inversion *(new to Option C)*
**Closes:** B5, B8

- `StandaloneQueryScope` takes ownership of the engine; `finish()` orders drain → repo erase →
  engine destruction.
- `drop_query_runtime_state_best_effort` erases the repository registry.
- With the plan outliving the drains, remove the defensive machinery that existed only to cope
  with the inversion (the `drain_after_error` comment block explaining why the creator must stay
  interrupted across executor drains describes a hazard that no longer exists).

**Tests:** force `get_result()` to throw — the path that today destroys the plan with no drain
having run — and assert clean teardown under ASan.

---

### C11 — Shutdown ordering
**Closes:** B6, B7, B10 · Same as Option A step A9.

### C12 — Per-query telemetry and prefetch-cache safety
**Closes:** A8, B11 · Same as Option A step A10.

---

### C13 — Fairness *(new to Option C — promoted ahead of enablement)*
**Closes:** F1, D3, H8

- Widen `query_id_t` to 64-bit; stop packing it into the signed priority; make it a separate
  fairness index level.
- Round-robin across live queries at dispatch in `management_eventloop`.
- Rotate `schedule_lookahead` across live queries.

> Deliberately **before** enablement, unlike Options A and B. Turning on concurrency with strict
> query-id-first priority means the first thing the new concurrent tests measure is head-of-line
> blocking, and a livelock (query 1 blocked on memory only query 2 can release) reads as a hang —
> which is exactly the failure mode you would otherwise be trying to debug.

---

### C14 — Turn concurrency on and prove it
**Closes:** validation half of G1, G4, G5, G7

As Option A step A11, plus: concurrent TPC-H at scale under memory pressure (the case C9 and C13
were built for), and a fairness assertion (a short query behind a long query completes in bounded
time).

---

### C15 — Config safety and hygiene
**Closes:** D5, E1–E7, F2–F7, F9, H1–H3, H5–H7, H9, H11

Same content as Option A step A12 minus the fairness items (done in C13) and the alias rename
(done in C8). Split into 4–6 commits.

---

## Pros

- **The dangerous pointer types stop existing.** After C8 and C10 there is no raw
  `data_repository*` crossing a queue hop and no pipeline referencing a dead plan. A new
  contributor cannot reintroduce those bugs by writing ordinary code — which is not true under
  A or B.
- **Deletes the worst serialization point rather than narrowing it.** C9 removes the global
  downgrade drain outright. Under A and B, every query end still quiesces every downgrade executor
  on every memory space — a permanent tax on exactly the workload (concurrent + memory-pressured)
  that this project exists to enable.
- **Solves your `shared_ptr` question without its downside.** Eager `close()` keeps memory release
  deterministic, keeps the `log_pool_stats` leak signature working, and keeps the leaked-batch
  diagnostic — the three real objections to a plain `shared_ptr` conversion.
- **Turns silent failures into observable ones.** A straggler publishing into a closed repository
  is logged and counted rather than being a memory-safe-but-wrong no-op (or a crash).
- **Fairness lands before enablement**, so the first concurrent test results are interpretable.
- **Best foundation for removing the mutex.** The follow-on project (admission queue instead of
  single-flight) is mostly configuration after C14; under A or B there is more to re-verify.

## Cons

- **Largest and longest.** ~2,600 LOC across 15 steps, with C15 realistically splitting into 4–6
  more. Roughly 1.8× Option A.
- **Modifies the `cucascade` submodule** (`data_repository`, `data_repository_manager`). Separate
  review, separate CI, possible coordination with other cucascade consumers — and a submodule bump
  in the middle of the plan.
- **C8 is the highest-consequence single step in any option.** It changes the ownership model of
  the object every operator reads and writes through, in a submodule. If `close()` semantics are
  wrong, the failure mode is "query silently produces incomplete results", which is worse than a
  crash.
- **C10 changes who owns the engine**, touching all five entry points that construct a
  `StandaloneQueryScope` (`physical_sirius_execution.cpp`, `sirius_extension.cpp` ×2,
  `sirius_ffi.cpp`) and DuckDB's `GlobalSourceState` interaction. Real risk of subtle result-lifetime
  regressions on the streaming/pending paths — which is precisely what
  `test_query_lifecycle_slot.cpp` was written to catch, so at least it is covered.
- **Value arrives latest.** Nine steps before anything structurally new lands, and C8–C10 are three
  large consecutive steps with no user-visible change until C14.
- **Hardest to revert.** C8 and C10 are not independently revertible once C9 depends on them.

## Risk register

| Risk | Mitigation |
|---|---|
| C8's `close()` semantics wrong → silent wrong results | Make `add_data_batch` on a closed repository return an explicit "dropped" status that the caller *must* handle; assert-fail in debug builds; add a result-correctness test with a forced straggler |
| cucascade submodule coordination | Land the cucascade half as its own PR first, bump the submodule, then land the Sirius half — C8 splits cleanly in two |
| C10 regresses result lifetime on streaming/pending paths | `test_query_lifecycle_slot.cpp` already covers abandoned streaming and pending results; run it under ASan before and after |
| Three large consecutive steps (C8–C10) with no visible payoff | Sequence C1–C7 first — identical to Option B, so real fixes ship for seven steps before the structural work begins |
