# Concurrency Model

This document describes how Super Sirius runs multiple queries at once: admission, per-query
state, config snapshots, fair scheduling, memory pressure across co-tenant queries, the ordered
per-query cleanup, and how the transparent execution paths participate. The historical issue
register behind this design is `docs/concurrency/00-issue-register.md`; the bring-up triage
(`docs/concurrency/01-bringup-triage.md`) records which register items are fixed and how.
Register IDs cited below (C4, D3, D5, E1, F1, ...) refer to those documents.

The prime directive throughout: **every piece of runtime state is either process-shared and
immutable-per-query, or keyed by query id** — and every teardown step touches exactly one
query's key.

> **Volatility note.** The executor blocking primitives (the quiesce/resume bracket and
> `push_or_bounce`) are under active rework as this is written; that section describes the
> current tip — re-verify it against the code before relying on the details. The
> repository-teardown fences reached their structural end-state (steps 6+7): repositories are
> shared-ownership, the interim sweep gate is gone, and `wait_inflight_request` remains as the
> plan-lifetime fence (see "Teardown fences").

## Query identity: `query_id_t`

**File:** `src/include/query_id.hpp`

One execution window = one query = one `sirius::query_id_t`, a distinct 32-bit enum type (not an
integer alias, so it cannot be silently confused with operator/pipeline/connection ids). It is
minted once when the window opens and is THE identity everywhere downstream: which repository
manager the query owns, which entries its cleanup drops, its scheduling band, and its
log/telemetry correlation key.

`query_priority_bits(id)` packs the id into the high 32 bits of the signed 64-bit
`exec::queue_priority`, giving every query a contiguous, non-overlapping *band* of priorities;
the low 32 bits preserve pipeline order within the query. The bands provide grouping for the
queues' per-query indexes — not cross-query precedence (see [Scheduling](#scheduling)). The id
is masked to 31 bits before shifting (the priority is signed), so band values wrap every 2^31
queries; the fair pops rotate over live ids rather than trusting value order, so a wrap only
shifts the rotation's starting point.

## Admission

**Files:** `src/include/sirius_context.hpp`, `src/sirius_context.cpp`

### The counted slot pool

The Super Sirius runtime is shared across connections. Plan generation and execution used to be
serialized outright by a single-flight mutex; the gate is now a **counted slot pool**
(`query_lifecycle_mutex_` + `query_lifecycle_cv_` + `query_lifecycle_slots_` in `SiriusContext`)
sized once at `initialize()` from `scan_manager_config::max_concurrent_queries` (YAML:
`sirius.executor.scan_manager.max_concurrent_queries`, default 1 — the historical single-flight
behavior, so nothing changes until the config raises it). The same knob bounds the scan
manager's live per-query states and sizes its thread pool, so pool sizing and admission cannot
drift (register C3).

A slot is **scope-bound**: acquired and released in one C++ scope on one thread, never held
across DuckDB's user-visible result lifetime — an abandoned stream or pending result holds
nothing. `acquire_query_lifecycle_slot()` errors on a same-thread reacquire (a nested acquire
would be a silent permanent wait) and, after acquiring, re-checks both runtime health and the
acquiring connection's cancellation before any shared mutation — a cancelled waiter never
late-enters the window. `query_lifecycle_peak()` reports the high-water mark of concurrently
held slots (the bring-up tests assert overlap actually happened).

Two RAII window types own slots:

| Guard | Used for | Body |
|-------|----------|------|
| `SiriusContext::SlotGuard` | Plan-generation windows (`OnFinalizePrepare` validation, explicit-path plan building, runtime-sensitive SET callbacks like pin/unpin) | Slot acquire + config snapshot only; destructor releases |
| `SiriusContext::StandaloneQueryScope` | Full execution windows (transparent execution, `CALL gpu_execution`, `pin_table`) | Slot acquire + config snapshot + begin mutations; explicit `finish()` runs the mandatory cleanup; noexcept destructor is the backstop |

### Window begin, in order

`StandaloneQueryScope`'s constructor:

1. **Mint the window id** — `allocate_window_id()`, *before* the slot is acquired (after
   acquire, no statement on any path allocates). Register H8 wrap guard: the 32-bit counter
   skips raw 0 (`make_query_id(0)` is the "no query" sentinel used by unattributed downgrade
   requests) and any id still live in the lifecycle registry or the repository registry, so a
   wrapped counter cannot collide with a query that is still running and trip
   `create_for_query`'s duplicate check.
2. **Acquire the slot** (health + cancellation re-checked inside).
3. **Install the config snapshot** — see [Config snapshots](#config-snapshots-set-semantics).
4. **`begin_execution_window()`** — the begin mutations, inside the held slot:
   `query_lifecycle_.open_query(query_id)` (the enqueue gate opens before anything can
   schedule), `data_repository_registry_.create_for_query(query_id)`, and
   `task_creator_->set_client_context(query_id, context)` (registers the query's task-creator
   state entry). GPU admission (`set_active_gpu_ids`) runs later, in
   `sirius_engine::initialize_internal()`.

A throw out of the begin mutations is classified (see
[Failure classification](#failure-classification-d5)), backstop-cleaned, and rethrown as the
typed `SiriusBeginWindowFailureException` — entry points must abort the query rather than fall
back to CPU, because the begin may have part-mutated shared registries.

### The lifecycle gate: `query_lifecycle_registry`

**File:** `src/include/exec/query_lifecycle_registry.hpp`

The single authority on "may work still be enqueued for this query?". States: **open**
(window begin) → **quiescing** (top of cleanup; enqueues refused) → **closed** (erased after
the drains). Sirius has several thread pools and enqueue points that fire from *completion
callbacks* — a finishing task schedules its downstream consumers; a downgraded task pushes
itself back — and during teardown those late enqueues must be refused or a drain that already
ran leaves work behind pointing at a dying plan. Historically that was done by interrupting the
*shared* queues, which froze every query at once; the registry does the same thing per query.

Consulted by every producer: `task_creator::schedule()` / `accepts_work()`,
`itask_executor::schedule()`, the TIER-2 downgrade wrapper's RAII re-push, lookahead rotation,
and TIER-2 extraction itself. An **unknown** id is treated as accepting work — the failure mode
of a missed `open_query()` must not be a silent hang, and components constructed without a
registry (most unit tests) behave as before.

## Per-Query State

Every subsystem holds its per-query state in a map keyed by `query_id_t`, handed out as
`shared_ptr` so a reader that resolved an entry keeps it alive against a concurrent erase:

| Subsystem | Entry type | Registered by | Dropped by |
|-----------|-----------|---------------|------------|
| `task_creator` | `query_task_global_state` — per-pipeline task global states, client context, completion handler, lookahead queue, admitted GPU subset | `prepare_for_query(query, handler)` (registration seeded at `set_client_context`) | `reset(query_id)` |
| `sirius_scan_manager` | `query_scan_manager_state` — providers, per-query sequencer, `scoped_dispatcher`, MVCC jobs, checkpoint locks | `prepare_for_query(query, pruning, gpu_ids)` | `reset(query_id)` |
| `SiriusContext` | `data_repository_manager_registry` — one `cucascade::shared_data_repository_manager` per query | `begin_execution_window` → `create_for_query(query_id)` | `run_mandatory_cleanup` → `erase(query_id)` |
| `SiriusContext` | `retired_query_plans_` — parked engine + prepared-statement data (the plan tree owners) | `retire_query_plan(query_id, keepalive)` from `sirius_interface::end_query_internal` | `destroy_retired_query_plan(query_id)` inside the cleanup |
| prefetch cache | live query epoch (`query_epoch_tracker`, register F3) | scan manager's `prepare_for_query` | scan manager's `reset(query_id)` → `finish_query` |

Why per-query maps rather than shared members: **operator ids restart at 0 for every query**, so
`{operator_id, port_id}` repository keys and the creator's `global_states` keys are only unique
*within* a query — a single shared map would let two queries fetch each other's state. And a
shared member is a write-race: the per-query GPU admission (`set_active_gpu_ids`) originally
wrote onto shared `task_creator` members, letting one query's narrowing clamp another query's
tasks (fixed during bring-up; the subset now lives in `query_task_global_state::active_gpu_ids`).

The **completion handler** is per query too: owned by the query's `sirius_engine`, stamped onto
every pipeline's task global state by `task_creator::prepare_for_query`, so tasks report
completion/errors with no shared "current query" handler
(`src/include/pipeline/sirius_pipeline_task_states.hpp`).

## Config Snapshots (SET semantics)

**Files:** `src/include/sirius_config.hpp` (`query_config_snapshot`,
`scoped_query_config_snapshot`), `src/sirius_context.cpp`

SNAPSHOT-AT-WINDOW-BEGIN (registers E1/E2/E3): the execution window copies the SET-mutable
configuration **once** at admission — `SiriusContext::snapshot_query_config()`, taken under
`operator_params_mutex_` so it can never observe a torn write — and installs it thread-locally
via `scoped_query_config_snapshot` (owned by `StandaloneQueryScope`/`SlotGuard`). The snapshot
carries:

- `operator_params` — read via `SiriusContext::query_operator_params()`;
- the expression-evaluator strategy (E2) — operators and scan ingestibles capture it at
  construction and pass it to every evaluator, so one plan uses one strategy;
- `compression_config` (E3) — read via `query_compression_config()`; `pin_table` materializes
  inside its own window and reads the frozen copy (including the `input_plan_dir` string it
  walks), so a concurrent SET can neither tear the string nor reshape a pin mid-run.

**Semantics: a `SET` takes effect for queries ADMITTED after it — never a query mid-plan or
mid-execution.** SET callbacks write through `update_operator_params()` /
`update_compression_config()` under the same mutex and deliberately do **not** occupy a window
slot: with the counted gate a single slot no longer excludes running queries anyway, and
isolation comes from the snapshot, not from blocking the SET. Values that must survive past the
window thread are stamped onto per-query structures at admission — e.g.
`gpu_reservation_max_retries` is copied onto the query's `pipeline_build_context` and read by
the executor's OOM-reschedule path as a per-query constant.

Settings DuckDB stores per connection (`gpu_execution`, `enable_duckdb_fallback`,
`enable_compressed_materialization`) are resolved against the executing connection's
`ClientConfig`, not from Sirius state.

## Scheduling

### The queue: `multi_index_priority_queue`

**File:** `src/include/exec/multi_index_priority_queue.hpp`

Every task queue in the engine — the task creator's creation queue, the task scheduler's
pipeline queue, and each GPU executor's staging queue (`itask_executor::_task_queue`) — is an
`exec::multi_index_priority_queue`. Its spine is an ordered map of priority *levels* (one level =
one pipeline's equal-priority tasks, FIFO); because priority, query, and operator are functions
of the same axis, the query and operator indexes are lookups onto levels, while preferred device
keeps a real per-task side index. This is what makes per-query surgery cheap:
`drain(query_index{id})` drops one query's tasks without touching any other's, and
`try_pop_from(gpu_index{id})` serves device-preferring dispatch.

### Cross-query fairness (F1)

The dispatch-order pops — `pop()` and the front `try_pop_from(gpu_index)` — **round-robin across
query bands** instead of following the packed priority order: the queue remembers the last query
served and serves the next live query id (ascending, wrapping) that has eligible work, taking
that query's best-priority task. Within a query the order is exactly the strict one; with a
single live query this is bit-identical to the pre-fairness queue. Without the rotation, the
query-id-in-high-bits encoding would let every task of an earlier query outrank every task of a
later one — the later query starves, and under memory pressure the pair can livelock. Back pops
and the predicate pops (`try_pop_if` / `mutable_pop_if` — downgrade-victim selection) keep
strict global order: they are not dispatch. Evidence:
`test/cpp/integration/test_concurrent_fairness.cpp`.

### The non-blocking manager (C4)

**File:** `src/pipeline/gpu_pipeline_executor.cpp`

The GPU executor's manager thread **never blocks on memory**: `process_task()` only resolves the
task's completion handler, attributes the pool slot to the task's query
(`slot.attach(query_id)`), and dispatches. The blocking reservation, the downgrade-on-shortfall
round trip, and execution all run in the worker (`prepare_and_execute`), covered by the pool's
per-query accounting. At most one task per executor parks in a blocking memory wait
(`_memory_waiter_parked`); overflow waiters re-queue through the executor's own queue after a
10 ms worker-held backoff (`executor_metrics::tasks_requeued_on_memory_wait`), so a hungry query
can neither stall the device's dispatch loop nor fill every worker slot with parked waits.
Re-queues never consume the OOM retry budget. Evidence:
`test/cpp/pipeline/test_gpu_pipeline_executor_memory_wait.cpp`.

### Lookahead rotation (D3)

**File:** `src/creator/task_creator.cpp` — `schedule_lookahead()`

When the scheduler queue runs empty while devices are idle, `management_eventloop` asks the task
creator to warm up **one** not-yet-activated scan. The rotation is round-robin across the
queries that still accept work (`upper_bound` on the last-served id, wrapping — stale cursor ids
are harmless, the same contract as the queue's fair-pop cursor); a query with nothing warmable,
or one quiescing per the lifecycle gate, does not pin the rotation. Previously only the oldest
registered query ever received lookahead and every newer query started cold. Teardown safety
comes from ordering: `drain_pending_tasks(query_id)` clears the query's lookahead queue under
`lookahead_mutex` *before* its request drain, so a racing walk either lands its push ahead of
the drain (dropped) or finds the queue empty. Evidence:
`test/cpp/creator/test_task_creator_lookahead.cpp`.

## Memory Pressure

**Files:** `src/include/downgrade/downgrade_executor.hpp`, `src/downgrade/downgrade_executor.cpp`,
`src/include/data/data_repository_manager_registry.hpp`,
`src/include/data/convertible_gpu_pipeline_task.hpp`

Memory pressure is a **global** condition: one query's allocation failure is relieved by
spilling whatever the space holds, whoever owns it. One `downgrade_executor` per memory space
runs a monitor thread (fires a pressure request when `should_downgrade_memory()`, at most one
monitor request outstanding, viability-gated) and a processing thread that serves one
`downgrade_request` at a time. See [memory-management.md](memory-management.md) for the request
pattern and candidate mechanics; what matters here is the cross-query contract:

- **Attribution.** Each `downgrade_request` carries the query id of its *waiter*
  (`request_downgrade(query_id, predicate)` from the GPU executor's reservation paths);
  the monitor's own requests are unattributed (`make_query_id(0)`).
- **TIER-1 sweep** walks every in-flight query's repository manager — the registry snapshot in
  *reverse* (newest query first: the query with the least progress pays for the memory, and the
  oldest keeps its working set so it can finish and actually release memory — a FIFO pressure
  response). Managers are `shared_ptr`, so a query ending mid-sweep cannot pull one out from
  under the loop.
- **TIER-2 sweep** extracts queued tasks from the scheduler queue into
  `convertible_gpu_pipeline_task` wrappers, converts their batches, and re-pushes via RAII. The
  wrapper captures the task's `exec::index_keys` at extraction time (while the plan was
  guaranteed alive) and uses them verbatim for both the gate lookup and the re-push (register
  B1); a quiescing query's task is dropped instead of resurrected. Extraction itself also
  consults the gate, so a request starting after `quiesce()` never extracts the ending query's
  tasks.
- **Per-query drain.** `drain(query_id)` fails only that query's queued promises and waits out
  an in-flight request *of that query* — no thread stop/restart, so peer queries' pending
  spills and the monitor's pressure response proceed unaffected (registers A7/B2). Every path
  that destroys a request routes through `fail_request()`, which re-arms the monitor's
  `_monitor_request_enqueued` flag (register D6). The global `drain()` (cancel everything,
  stop-join-restart) is DELETED (step 7): `terminate()` stops executors outright via `stop()`,
  and unattributed requests (the monitor's, external byte targets) can be cancelled with
  `drain(make_query_id(0))`. `stop()` self-serializes through the `_running` CAS.

### Teardown fences (steps 6+7 end-state)

Steps 6+7 replaced the interim repository fences with ownership. The rule: **no raw
`data_repository*` crosses a blocking call anywhere.**

- **Shared-ownership repositories (step 6) — the retired sweep gate.** A TIER-1 sweep co-owns
  everything it borrows: `registry::get_all()` hands out `shared_ptr` managers,
  `data_repository_manager::get_repositories()` hands out `shared_ptr` repositories, and
  batches were always `shared_ptr`. `erase(query_id)`/`clear()` therefore only drop the MAP
  ENTRY — an in-flight sweep keeps its borrowed objects alive until it naturally finishes, and
  teardown never waits for a sweep. The old `begin_sweep()` token that `erase()` fenced on is
  deleted. Leaked-batch accounting moved with the ownership: batches that die un-consumed are
  reported by the repository DESTRUCTOR (wherever and whenever the last holder releases it),
  attributed to their query by the leak handler `create_for_query()` installs. Plan-side
  readers co-own too: operator ports hold `port::repo_owner` (the raw `port::repo` is a cached
  alias), and `gpu_pipeline_task::_data_repos` carries `shared_ptr` destinations across queue
  hops and OOM reschedules (B4).
- **`wait_inflight_request()` — the SURVIVING fence (plan lifetime, not repository lifetime).**
  `drain(query_id)` waits only for the query's *own* in-flight requests — but a peer's (or the
  monitor's) request sweeps by memory space, so its TIER-2 pass may hold the ending query's
  task inside a convertible wrapper across a blocking conversion. The wrapper's RAII drop
  consults the lifecycle gate with extraction-time keys and, for a quiescing query, DESTROYS
  the task instead of re-pushing it — and `~gpu_pipeline_task` walks the task's plan
  (`mark_task_completed` → `notify_downstream_pipelines`). Plan parking (B5) only defers the
  plan's death until cleanup; cleanup must still not destroy it while such a wrapper is alive.
  So query-end cleanup waits out the requests in flight on every executor before destroying
  the parked plan. Bounded, because requests that start later never extract a quiescing
  query's tasks (TIER-2 extraction consults the gate). Shared-ownership repositories
  deliberately did NOT subsume this fence — it guards operator pointers, not repositories.

### Retry cap

The GPU executor's per-task OOM-reschedule cap is `operator_params.gpu_reservation_max_retries`
(default 100 — `exec::default_gpu_reservation_max_retries`; YAML and
`SET gpu_reservation_max_retries`, must be > 0), read per query from the admission snapshot via
`pipeline_build_context` (E1 discipline). Under heavy concurrency with tiny pools, fully-working
spills can legitimately trip the cap (in-use hash tables cannot downgrade), so stress tests
lower it to surface the classified retry-cap error quickly and distinguish it from a real bug.
Exceeding the cap fails only that task's query, through its own completion handler.

## Query End: The Mandatory Cleanup

**File:** `src/sirius_context.cpp` — `run_mandatory_cleanup()`

`StandaloneQueryScope::finish()` runs the cleanup inside the held slot; every step touches only
this query's keys. **Precondition, established by `sirius_interface::end_query_internal`
(register B5):** the engine and the `sirius_prepared_statement_data` that owns the plan tree are
*parked* on the context via `retire_query_plan()` instead of being destroyed inline — pipelines
and tasks hold non-owning operator pointers into that plan, straggler tasks are destroyed by the
drains below, and `~gpu_pipeline_task` walks those pointers
(`mark_task_completed` → `notify_downstream_pipelines`). The plan must die *after* the drains.

The ordered steps:

1. **Quiesce** — `query_lifecycle_.quiesce(query_id)`: every producer now refuses this query's
   work, so no completion callback can enqueue behind a drain that already passed.
2. **Creator reset** — `task_creator_->reset(query_id)`: clears the lookahead queue, drains the
   query's creation requests (raw operator pointers), waits out its in-flight creation lambdas
   via the pool's per-query accounting, and drops the state entry.
3. **Queue sweep** — `task_scheduler_->drain_query_tasks(query_id)`: drops the query's queued
   tasks from the scheduler queue and every executor's staging queue.
4. **Downgrade drain** — `downgrade_executor::drain(query_id)` on every executor: fails only
   this query's queued downgrade promises, unblocking its own (dying) waiters; peers untouched.
5. **Inflight wait** — `wait_inflight_request()` on every executor: waits out any request whose
   TIER-2 pass may hold this query's task in a convertible wrapper (the plan-lifetime fence;
   see above).
6. **Final sweep** — one more `drain_query_tasks(query_id)`: a wrapper that read the gate as
   open just before `quiesce()` landed may have re-pushed a task behind step 3's sweep; every
   wrapper is gone now, and nothing can re-add tasks (producers gate-refused, extraction skips
   quiescing queries).
7. **Parked-plan destruction** — `destroy_retired_query_plan(query_id)`: every drain that can
   destroy one of this query's tasks has run, so the plan is no longer reachable from any task.
   Stays before the repository erase so operators die while the repositories they were wired to
   still exist.
8. **Telemetry close-out** — best-effort (`batch_telemetry_registry::on_query_end()`); a
   telemetry failure never aborts the remaining mandatory steps.
9. **Repository erase** — `data_repository_registry_.erase(query_id)`: drops only the registry's
   map entry — an in-progress downgrade sweep co-owns whatever it borrowed and keeps it alive
   until it finishes (step 6). Un-consumed batches are logged as leaks by the repository
   destructors, attributed to this query.
10. **Scan reset** — `scan_manager_->reset(query_id)`: after the repositories, so downstream
    batches referencing sliced host representations are gone before the providers die.
11. **Close** — `query_lifecycle_.close(query_id)`: the window is over; the entry is erased
    rather than tombstoned so the map stays bounded by in-flight queries.

The slot is released after the cleanup by a non-throwing releaser on every path.

### The error path

A runtime error reaches `sirius_engine::execute`'s `future.get()` catch, which runs
`task_scheduler::drain_after_error(query_id)` (`src/pipeline/task_scheduler.cpp`) **before** the
window's cleanup: quiesce → scheduler-queue drain → per-executor `wait_and_drain_query(query_id)`
→ `drain_pending_tasks(query_id)` → final queue sweep. `wait_and_drain_query` uses the
quiesce/resume manager bracket (`itask_executor::quiesce_manager()`/`resume_manager()`,
serialized by `_manager_lifecycle_mutex` — two queries failing concurrently on the same executor
must not both restart its manager thread); the bracket briefly interrupts the shared queue, and
a co-tenant's push bounced during it is retried via `push_or_bounce` rather than dropped
(register A6 — a dropped successor silently hangs the co-tenant). The success path
(`wait_for_completion(query_id)`) never touches the manager thread: it quiesces, *validates*
that the query has nothing queued (throwing rather than draining, which would hide the bug), and
waits out in-flight pool work.

> The quiesce/resume bracket and `push_or_bounce` are part of the blocking-primitives area under
> active rework (see the volatility note at the top).

Errors raised where no exception may propagate (task destructors, streaming-close callbacks)
route through `task_creator::report_fatal_error(query_id, error)` /
`task_scheduler::terminate_query(handler, error)` — the error goes to that query's own
completion handler and nothing else.

### Failure classification (D5)

**File:** `src/sirius_context.cpp` — `classify_query_failure()`

A begin-window or mandatory-cleanup failure is classified at its catch site. The default verdict
is **per-query**: the query errors, its state is dropped best-effort
(`drop_query_runtime_state_best_effort`), healthy in-flight queries keep running, and new
queries keep admitting (`per_query_cleanup_failures()` counts these). The process-wide
`runtime_unavailable_` latch fires **only** for genuinely shared corruption — today, a sticky
CUDA error that survives a `cudaGetLastError()` clear (the device context is lost for the whole
process). Once latched, every later attempt to enter an execution or plan window gets a stable,
session-preserving `SiriusRuntimeUnavailableException` (never INTERNAL/FATAL — CPU queries must
continue). A cleanup failure never runs a second pass over half-cleaned state: `finish()`'s
catch classifies, drops best-effort, releases the slot, and lets the query error; the destructor
backstop runs the cleanup only when `finish()` never completed.

## Transparent Execution Under Concurrency

**Files:** `src/include/sirius_context.hpp` (`SiriusConnectionState`), `src/sirius_context.cpp`,
`src/transparent/sirius_optimizer_extension.cpp`, `src/transparent/physical_sirius_execution.cpp`

- **Per-connection capture.** The optimizer hook copies the optimized logical plan into the
  connection's own `SiriusConnectionState` (registered per `ClientContext`), stamped with a
  per-connection planning generation — a stale capture from an earlier attempt is dropped, and
  one connection's planning can no longer clobber another's. Connection-scoped flags
  (`InternalQueryGuard` and `CpuFallbackGuard` depths, the query ordinal) live there too, so the
  guards no longer disable other connections' lifecycle callbacks.
- **`OnFinalizePrepare`** consumes the capture (generation-checked), validates it for GPU
  translation inside a `SlotGuard` plan window, and on success replaces the physical plan with a
  `PhysicalSiriusExecution`, stashing the CPU plan for runtime fallback
  (`install_transparent_execution`). `FORCE_MATERIALIZED`/`IN_MEMORY` are fixed on the shared
  fallback stash here, at prepare time — never written per-execution (part of E4).
- **Execution.** `PhysicalSiriusExecution::GetDataInternal` opens a `StandaloneQueryScope`,
  copies the plan template, plans/executes through `sirius_interface`, and calls
  `window->finish()` before exposing the result. The operator is **immutable across executions**
  (register E4): `logical_plan_` is never `reset()` — every execution takes its own
  `copy_logical_plan()` and a monotonic atomic (`plan_copy_unsupported_`) latches the
  non-copyable case onto the SQL-replan path — so concurrent executions of one prepared
  statement share only read-only state.
- **PREPARE/EXECUTE.** `OnExecutePrepared` requests a rebind on each execute of a reusable
  prepared statement so GPU eligibility is re-decided against current stats. SQL-level
  `EXECUTE <name>` is intercepted by `try_intercept_execute_statement`: the stored statement is
  recovered by name from `ClientData`, its parameterless SELECT re-planned fresh under an
  `InternalQueryGuard`, and the shared install tail taken. Parameterized prepared statements
  stay on DuckDB's CPU path (re-planning the unbound statement cannot bind EXECUTE-time values).
- **CPU fallback.** Plan-time declines fall back silently. A runtime GPU failure (window
  finished, error captured) replays the stashed CPU plan on a private executor in the same
  transaction, outside any Sirius window, under a `CpuFallbackGuard` (which the `sirius_httpfs`
  s3:// guard reads — S3 is GPU-only, so s3-reading queries error instead of falling back).
  `SiriusBeginWindowFailureException` is never a fallback candidate. The explicit
  `CALL gpu_execution(...)` path follows the same window + fallback shape in
  `SiriusExtension::GPUExecutionFunction` (`src/sirius_extension.cpp`); its per-execution
  `ClientConfig` save/restore is an RAII on the executing stack (`scoped_client_config` in
  `ExtractPlan`), not on the shared bind data (register E7).

## Testing

**Files:** `test/cpp/integration/test_concurrent_queries.cpp`,
`test_concurrent_adversarial.cpp`, `test_concurrent_config_races.cpp`,
`test_concurrent_fairness.cpp`; shared harness `test/cpp/utils/concurrent_test_utils.hpp`

- **The bring-up harness** (`test_concurrent_queries.cpp`): `concurrent_env` builds a per-test
  DuckDB + config YAML with the concurrency knob under test; `run_workers` runs barrier-synced
  worker threads. It proves both correctness under genuine overlap and that overlap *happened*
  (`query_lifecycle_peak() > 1` — a silent re-serialization must not pass on correctness alone).
  GPU-engagement assertions (`executions >= N`, `runtime_fallbacks == 0`) catch a silent CPU
  fallback. Slots/workers/iterations/pool bytes are env-tunable
  (`SIRIUS_TEST_CONCURRENCY_{SLOTS,WORKERS,ITERS,POOL_BYTES}`) for bisection grids.
- **The adversarial suite** (`test_concurrent_adversarial.cpp` + `concurrent_test_utils.hpp`):
  workload-shaped attacks on the register's MUST-FIX cluster — spill storm with constant
  query-end churn (A7/B1/B2/D6), mixed failure storms (per-query containment), pin churn, 2-GPU
  admission. `scoped_watchdog` aborts with a diagnostic instead of hanging;
  `scoped_downgrade_log_counter` asserts spills actually fired. Config-race scenarios
  (`test_concurrent_config_races.cpp`) drive SET storms against running queries and restore
  every knob they perturb.
- **Co-tenancy rules.** Test envs use small *absolute* GPU pools (default 4 GiB; the pool is
  allocated eagerly at init, and shared GPUs have co-tenants at env-construction time — a
  fraction-of-card default flakes with `bad_alloc`), and per-PID temp files for configs and
  parquet fixtures, since several test binaries from sibling worktrees may run concurrently on
  one box. Catch2 v2 assertion macros are not thread-safe: workers collect failure strings; the
  main thread `REQUIRE`s.
- Deterministic unit tests cover the fair pops, the lookahead rotation
  (`test/cpp/creator/test_task_creator_lookahead.cpp`), the TIER-2 wrapper
  (`test/cpp/data/test_convertible_gpu_pipeline_task.cpp`), the C4 memory wait
  (`test/cpp/pipeline/test_gpu_pipeline_executor_memory_wait.cpp`), the config snapshot
  (`test/cpp/config/`), and the window-id wrap guard (via `set_next_window_id_for_test`).

## Key Files

| File | Purpose |
|------|---------|
| `src/include/query_id.hpp` | Window/query identity, priority-band packing |
| `src/include/sirius_context.hpp` / `src/sirius_context.cpp` | Slot pool, execution windows, begin/cleanup, failure classification, snapshots |
| `src/include/exec/query_lifecycle_registry.hpp` | The per-query enqueue gate |
| `src/include/data/data_repository_manager_registry.hpp` | Per-query repository managers (shared ownership; per-query leak attribution) |
| `src/include/creator/task_creator.hpp` / `src/creator/task_creator.cpp` | Per-query creation state, lookahead rotation, per-query drains |
| `src/include/pipeline/task_scheduler.hpp` / `src/pipeline/task_scheduler.cpp` | Per-query start/terminate/drain, fair dispatch |
| `src/include/pipeline/gpu_pipeline_executor.hpp` / `src/pipeline/gpu_pipeline_executor.cpp` | C4 non-blocking manager, memory-wait slot, per-query OOM cap |
| `src/include/parallel/task_executor.hpp` / `src/parallel/task_executor.cpp` | Gated `schedule()`, `push_or_bounce`, quiesce/resume bracket, per-query executor drains |
| `src/include/downgrade/downgrade_executor.hpp` / `src/downgrade/downgrade_executor.cpp` | Attributed requests, per-query drain, tiered sweeps, inflight wait |
| `src/include/data/convertible_gpu_pipeline_task.hpp` | Gated TIER-2 RAII wrapper with extraction-time keys |
| `src/include/exec/multi_index_priority_queue.hpp` | Fair pops, per-query drain, secondary indexes |
| `src/include/sirius_config.hpp` | `query_config_snapshot`, `scoped_query_config_snapshot` |
| `src/include/scan_manager/sirius_scan_manager.hpp` | Per-query scan state, `max_concurrent_queries` |
| `src/include/io/cache/query_epoch_tracker.hpp` / `src/include/io/cache/fair_band_queue.hpp` | Prefetch-cache live epochs (F3) and per-query request fairness (F4) |
| `src/include/transparent/physical_sirius_execution.hpp` | Immutable prepared-statement operator (E4), runtime CPU fallback |
