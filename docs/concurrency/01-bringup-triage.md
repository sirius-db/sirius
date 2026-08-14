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

## MUST FIX — live now, with bring-up evidence

Ranked; the first cluster currently hangs a test.

1. **Failure containment cluster — A7, B2, D6 (B1 and A6's live half now
   fixed, see above).** The error-path test passes, but the remaining three
   interlock under memory pressure and heavier churn: query-end
   `drain()` cancels PEERS' downgrade promises (A7), `drain()` restarts the
   processing thread before quiescence is used (B2), and a
   drain racing the monitor can permanently latch `_monitor_request_enqueued`
   (D6 — kills automatic spilling for the process). William's backed-out
   steps 6+7 (shared-ownership repositories, delete the global drain) are the
   structural fix; the interim path is per-query attribution on downgrade
   requests + scoping `drain_after_error` to the failing query.
2. ~~**B5 — plan destroyed before the drains.**~~ FIXED, see above.
3. **E1/E2/E4/E7 — shared mutable config.** The slot no longer serializes
   SET-vs-execution. E4 (transparent prepared statement `mutable
   logical_plan_`) is the sharpest: two concurrent EXECUTEs of one prepared
   statement is an ordinary workload. E1/E2 need at least torn-read safety
   (atomics / snapshot-at-window-begin); E2's `std::string` static variables can UAF.
4. **D5 — `runtime_unavailable_` process-wide latch.** With N queries, one
   unlucky cleanup poisons N−1 healthy queries and every future one. Needs a
   per-query verdict plus a separate genuinely-shared-corruption latch.
5. **B9 — `get_repository()` unlocked map reference.** Trivial to hit with
   two queries creating/clearing repositories; trivial to fix (take `_mutex`,
   return by value/shared_ptr — cucascade change).
6. ~~**F1 — absolute cross-query priority.**~~ FIXED (round-robin across query
   bands in the queue pops), see above.

## DEFER — real but not blocking correctness bring-up

- **A8** (telemetry truncation cross-query) — observability, not execution.
- **F2** (plan-generation excluded from execution windows) — the counted gate
  already admits N planners; narrowing to a read-view lock is a follow-up win.
- **F3–F9** — fairness/throughput; matter after correctness burn-in. F5
  (spurious PIN rejection from the all-entries snapshot) will surface in the
  pin-churn test and may get promoted.
- **E3/E5/E6** — narrower config races; fix with the E batch.
- **H group** — hygiene; H8 (query-id wrap at 2^31/2^32) deserves a guard
  before long-running deployments.

## Test gaps still to close (G7 spirit)

- Memory-pressure grid (small pool, downgrades firing while queries end).
- Pinned serving + unpin/re-pin churn (F5 watch).
- TPC-H shapes concurrently (not just the 4 bring-up shapes); 2-GPU config
  (now meaningful — the admission race is fixed).
- SET-vs-execution race test once the E batch lands.
