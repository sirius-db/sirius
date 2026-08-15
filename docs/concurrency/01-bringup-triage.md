# Concurrency Bring-up — Issue Register Triage

**Branch:** `integration/concurrency-full` · **Date:** 2026-08-14
**Context:** the counted query-lifecycle gate is in (slot pool sized by
`scan_manager.max_concurrent_queries`), so the register's framing — "everything
needed before `query_lifecycle_mutex_` can be replaced by an admission queue" —
is no longer hypothetical. Concurrency is REACHABLE via SQL, and a first
multi-connection harness exists (`test/cpp/integration/test_concurrent_queries.cpp`).
This file records where each register group stands NOW, with bring-up evidence.

## What the bring-up demonstrated

| Scenario (4 shapes over parquet views) | Result |
|---|---|
| 4 workers x 4 slots, happy path | PASS — peak 4 overlapping windows, 28 GPU executions, results byte-identical, 3/3 repeats |
| 4 workers x 1 slot (gate off) | PASS — serialized, correct |
| Runtime-failing query beside 3 healthy workers | PASS after the quiesce-bracket fix below (was: hang) — 7/7 fallbacks contained, healthy queries stayed on GPU |
| Memory pressure (3 GiB pool, 20M rows) | PASS — peak 4, 28 GPU executions, correct |
| Pinned-entry serving + unpin/re-pin churn | PASS — unpin+re-pin landed mid-serving, results correct (1 cycle; intensify later) |

## Fixed during bring-up (this branch, on top of steps 1–12)

| Register ID | Fix |
|---|---|
| new (unlisted) | Concurrent cleanups raced `downgrade_executor::drain()`'s thread join — SIGABRT via noexcept `run_mandatory_cleanup`, or a wedge. `_lifecycle_mutex` now serializes `drain()`/`stop()`. This is the crash the register's A7/B2 analysis predicts but does not list as its own entry. |
| new (unlisted) | `set_active_gpu_ids` (#1442, post-register) wrote the per-query GPU admission onto shared `task_creator` members — one query's narrowing clamped another's tasks. Moved into `query_task_global_state`. |
| A6 (the live half) | The error-path quiesce bracket (`wait_and_drain_query`) kept the device queue interrupted until `resume_manager()`, BOUNCING co-tenant pushes — a bounced successor = that query hangs (observed: repeatedly-failing query starved 3 healthy ones). Fixed: the queue reactivates the moment the manager join lands, and `itask_executor::schedule` bounce-retries via the new `push_or_bounce` instead of dropping (drop — still loud — only at real shutdown). |
| C3 | `max_concurrent_queries` is real config (step 12) and the gate reads the same knob, so pool sizing and admission can no longer drift. |
| G1/G2 (start) | `test_concurrent_queries.cpp`: reusable multi-connection barrier harness (`concurrent_env` + `run_workers`), GPU-engagement assertions (`executions >= N`, `runtime_fallbacks == 0`), `query_lifecycle_peak()` overlap proof, env-tunable slots/workers/pools grid. |
| new (unlisted) | Concurrent `drain_after_error` from two failing queries hit `std::terminate` in `itask_executor::resume_manager` (`std::thread` assigned over a joinable manager thread). `_manager_lifecycle_mutex` now serializes `stop()`/`drain_and_wait()`/`wait_and_drain_query()`. |
| B1 | The TIER-2 RAII re-push no longer re-derives the task's queue keys through the plan (`index_keys_for` → `pipe->get_source()->type`, freed once teardown destroyed the plan). `convertible_gpu_pipeline_task` captures `exec::index_keys` at extraction time (task still in the queue ⇒ plan alive) and uses them verbatim for both the lifecycle-gate lookup and a new key-supplied `multi_index_priority_queue::push` overload — mirroring `task_creation_request`'s pre-resolved keys. `run_mandatory_cleanup` also re-sweeps the query's queues once after the downgrade drains join every wrapper, closing the check-then-push race against `quiesce()`. Deterministic unit tests in `test_convertible_gpu_pipeline_task.cpp` fail pre-fix. |
| B5 | The plan now dies AFTER the drains. `sirius_interface::end_query_internal` parks the engine + `sirius_prepared_statement_data` (which owns the plan tree the collector references) on `SiriusContext` via `retire_query_plan()`; `run_mandatory_cleanup` destroys them once every drain for the query has run, still before the repository erase (operators die while their wired repositories exist). Backstop paths (`drop_query_runtime_state_best_effort`, `terminate()`) clear parked state too. `[teardown_races]` harness scenario exercises the B1/B5 interlock. |
| F1 | The queues' dispatch pops now round-robin across query bands instead of following the packed priority order: `multi_index_priority_queue::pop()` and the front `try_pop_from(gpu_index)` remember the last query served and serve the next live query id (wrapping) its best-priority task, under the queue's existing mutex. Within a query the order is untouched; single-query behavior is bit-identical (whole legacy suite). Covers the task_creator queue, the task_scheduler dispatch, and the GPU executors' staging queues in one change; back/predicate pops (downgrade-victim selection) keep strict order. `query_priority_bits()` packing stays — it now provides band separation for the per-query indexes, not cross-query precedence. Evidence: `test_concurrent_fairness.cpp` — heavy query admitted first beside 16 shorts, 1 GiB pool: before 0/16 shorts completed while it ran (first-short latency ~2.1 s vs 3.5 ms baseline); after 16/16 (median 13 ms), heavy elapsed unchanged within noise. |
| E1 | SNAPSHOT-AT-WINDOW-BEGIN: `sirius::query_config_snapshot` is copied ONCE per query at admission (installed thread-locally by `StandaloneQueryScope`/`SlotGuard`); every plan-time/execution-time reader uses `SiriusContext::query_operator_params()`, never the live struct. SET callbacks write via `update_operator_params` under `operator_params_mutex_` (the snapshot's lock) and no longer occupy a window slot. Semantics: a SET takes effect for queries ADMITTED after it — never mid-plan. |
| E2 | `duckdb::Config` scalars → `std::atomic` (source-compatible); `LOG_*` strings → copy-on-read `ConfigString` (shared_ptr swap under mutex). The expression strategy rides the E1 snapshot: operators and scan ingestibles capture it at construction (window thread) and pass it to every evaluator at execute time, so one plan uses one strategy. |
| E4 | `PhysicalSiriusExecution` is immutable across executions: `logical_plan_` lost `mutable` and is never `reset()` (a monotonic atomic `plan_copy_unsupported_` latches the non-copyable case); the shared CPU-fallback stash gets `FORCE_MATERIALIZED`/`IN_MEMORY` fixed at `OnFinalizePrepare` instead of being written per-execution. DIAGNOSIS CORRECTION: the documented "streaming EXECUTEs silently skip Sirius" signature was misattributed — instrumented runs show streaming engages on EVERY iteration (the GPU run completes inside `Execute()`'s single ClientContext lock hold), while SQL-level `PREPARE`/`EXECUTE` never engaged at all: a pre-existing single-threaded transparency gap, not a race. That gap is now CLOSED (see the PREPARE/EXECUTE row below); the prepared-statement scenario asserts per-phase engagement equalities including phase C. |
| E7 | `gpu_execution`'s ClientConfig save/restore moved off the shared bind data onto the executing stack (RAII in `ExtractPlan`); concurrent executions of one prepared statement can no longer clobber each other's saved copy. New `[config_race]` scenario asserts the connection's `enable_optimizer` survives concurrent executions. FOLLOW-UP FIXED: the `enable_optimizer` named parameter used to be registered but ignored (`GPUExecutionBind` hard-coded `true`); the bind now honors the user's value (default stays true), and a second `[config_race]` scenario runs `gpu_execution('…', enable_optimizer=false)` concurrently and asserts both the reference result and the connection's own config restore. |
| E3 | `compression_config` joined the E1 snapshot: `query_config_snapshot` carries it, the pin (which runs inside a `StandaloneQueryScope` window) reads `SiriusContext::query_compression_config()` — a coherent copy frozen at admission, so the `input_plan_dir` string it hands to `fs::directory_iterator` can no longer be torn/freed by a concurrent SET — and the four `pin_table_compression*` setters route through `update_compression_config` under `operator_params_mutex_`, mirroring their operator_params neighbours. SET-storm rotation extended with the compression knobs (RESET guard included); dedicated pin-churn-vs-SET-storm scenario in `test_concurrent_config_races.cpp`. |
| E5 | `plan_register` keys are the pin's RESOLVED cache identity — `cache_entry_info::compression_plan_key()`: duckdb `catalog.schema.table` or the canonicalized parquet file set, i.e. the same identity the pinned-entry serving matchers use — never the bare user-supplied pin name, so same-named tables in different ATTACHed databases keep separate plan entries (the plan FILE is still looked up by the pin name; that is the on-disk `<name>.<ext>` contract). The resolve-miss-load-set-resolve check-then-act collapsed into one atomic `get_or_load_table_plan(key, loader)` (loader runs under the registry's exclusive lock; a nullopt/empty result stores nothing, preserving the rescan-next-pin semantics). Two-ATTACHed-DBs scenario asserts each database's pin registers ITS OWN plan; threaded loads-exactly-once unit test in `test_compression.cpp`. |
| E6 | `get_target_ctas()` (CUDA host-side, `common.cuh`): the `{device, value}` pair of mutable function-local static variables with a non-atomic check-then-use — tearable across two GPUs — replaced by a per-device array of `std::atomic<uint32_t>` slots (0 = unfilled; device ids beyond the 64-slot bound recompute uncached). A read is one word; concurrent first-computes store the same deterministic value. Threaded unit test `test_decode_target_ctas.cpp` asserts every call returns the CALLING device's value, alternating devices when ≥ 2 GPUs are visible (the tear is only OBSERVABLE on heterogeneous devices, documented in the test). |
| new (PREPARE/EXECUTE transparency) | SQL-level `PREPARE … AS SELECT` / `EXECUTE` now runs transparently. The finalize hook DOES fire for the EXECUTE statement (its plan is a `PhysicalExecute` wrapping the stored prepared statement's CPU plan) — Sirius simply declined non-SELECT statement types. `SiriusContext::try_intercept_execute_statement` recovers the stored prepared statement by name from `ClientData`, re-plans its parameterless SELECT fresh (optimizer included, under `InternalQueryGuard`), and takes the shared `install_transparent_execution` tail — so every EXECUTE re-decides GPU eligibility against current stats, and the runtime CPU fallback stashes the `PhysicalExecute` wrapper. Parameterized prepared statements (EXECUTE-time values) stay on DuckDB's CPU path: re-planning the unbound statement cannot bind those values. Phase C of the prepared-statement scenario flipped from `== 0` to full engagement equality; a single-threaded engage+runtime-fallback+parameterized-decline test lives in `test_transparent_runtime_fallback.cpp`. No duckdb submodule change was needed. |
| new (OOM retry cap) | The GPU executor's per-task OOM-reschedule cap (`MAX_RETRIES`, hard-coded 100) is now the config value `operator_params.gpu_reservation_max_retries` (YAML + `SET gpu_reservation_max_retries`, default 100, must be > 0). E1 discipline: the engine stamps the admission-time snapshot value onto the query's `pipeline_build_context`, every `sirius_pipeline` copies it, and the executor's reschedule path reads that per-query constant — never the live struct mid-execution (pipeline-less tests fall back to `exec::default_gpu_reservation_max_retries`). Stress tests with tiny pools can now lower the cap to surface the classified retry-cap error quickly and distinguish it from a real bug. |

## MUST FIX — live now, with bring-up evidence

Ranked; the first cluster currently hangs a test.

1. ~~**Failure containment cluster — A7, B2, D6.**~~ FIXED. `downgrade_request`
   carries its query id; `drain(query_id)` fails only that query's promises
   with no thread stop/restart (A7+B2), every request-destruction path routes
   through `fail_request()` which re-arms `_monitor_request_enqueued` (D6), and
   the registry's sweep gate fences repository teardown against in-progress
   sweeps. INTERLOCK with B5 (coordinator): a PEER's in-flight sweep can hold
   the ending query's task across a blocking conversion, so cleanup calls
   `wait_inflight_request()` per executor before the final sweep + plan
   destruction, and TIER-2 extraction consults the lifecycle gate so later
   requests never extract a quiescing query's tasks. William's steps 6+7
   (shared-ownership repositories) remain the structural end-state.
2. ~~**B5 — plan destroyed before the drains.**~~ FIXED, see above.
3. ~~**E1/E2/E4/E7 — shared mutable config.**~~ FIXED, see above. The
   remaining E-batch — E3 (compression-config setters), E5 (plan_register
   check-then-act keyed by bare table name), E6 (`get_target_ctas()` mutable
   function-local static) — is now ALSO FIXED, see the table above; Group E is
   closed. Both follow-ups surfaced here are FIXED too (see the table):
   SQL-level `PREPARE`/`EXECUTE` now runs transparently (parameterless
   SELECTs; parameterized statements stay on CPU), and `gpu_execution`'s
   `enable_optimizer` named parameter is honored.
4. ~~**D5 — `runtime_unavailable_` process-wide latch.**~~ FIXED.
   `classify_query_failure()` at every former latch site: default verdict is
   per-query (query errors, state dropped best-effort, healthy queries and new
   admissions unaffected, counted by `per_query_cleanup_failures()`); the
   process-wide latch fires only when a sticky CUDA error survives a
   `cudaGetLastError()` clear. Follow-up: step-level classification inside
   `run_mandatory_cleanup` once steps 6+7 land; multi-device sticky probe.
5. ~~**B9 — `get_repository()` unlocked map reference.**~~ FIXED (cucascade):
   map stores `shared_ptr`, new locked `get_repository_shared()`; old accessor
   kept deprecated until downstreams migrate.
6. ~~**F1 — absolute cross-query priority.**~~ FIXED (round-robin across query
   bands in the queue pops), see above.

## DEFER — real but not blocking correctness bring-up

- **A8** (telemetry truncation cross-query) — observability, not execution.
- **F2** (plan-generation excluded from execution windows) — the counted gate
  already admits N planners; narrowing to a read-view lock is a follow-up win.
- **F3–F9** — fairness/throughput; matter after correctness burn-in. F5
  (spurious PIN rejection from the all-entries snapshot) will surface in the
  pin-churn test and may get promoted.
- **H group** — hygiene; H8 (query-id wrap at 2^31/2^32) deserves a guard
  before long-running deployments.

## Test gaps still to close (G7 spirit)

- Memory-pressure grid (small pool, downgrades firing while queries end).
- Pinned serving + unpin/re-pin churn (F5 watch).
- TPC-H shapes concurrently (not just the 4 bring-up shapes); 2-GPU config
  (now meaningful — the admission race is fixed).
- SET-vs-execution race test once the E batch lands.
