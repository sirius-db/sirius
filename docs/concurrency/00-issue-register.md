# Concurrent Query Execution — Consolidated Issue Register

**Branch:** `concurrency2` (`a35be838`) · **Baseline:** `dev`
**Scope:** everything needed before `SiriusContext::query_lifecycle_mutex_` can be replaced by an
admission queue. `src/legacy/` is out of scope.

This is the single source of truth for the remaining work. Every plan option in this directory
closes **all** of these issues; they differ only in *mechanism* and *ordering*.

---

## 0. Where we actually are

The last six commits converted the per-query **data** plumbing:

| Already done — do not redo | Commit |
|---|---|
| `task_creator::query_task_global_state` map keyed by `query_id_t`; per-query in-flight counter; `reset(query_id)` / `drain_pending_tasks(query_id)` | `a8002e38` |
| `pipeline::index_keys_for` as the one key extractor; executor queues → `multi_index_priority_queue` with `query_index`; `drain(query_index)` | `59e0df9e` |
| `completion_handler` moved onto `sirius_pipeline_task_global_state`; `task_scheduler::_query` deleted | `49dc8421` |
| `SiriusContext::query_` deleted; `planner::query` owned by `sirius_engine`; `completion_handler_` outlives the engine | `f0883cee` |
| `query_scan_manager_state` with its own `scoped_dispatcher`; pin table → `shared_ptr<pinned_entry>` | `cecfcfe5` |

What was **not** converted is the per-query **control** plumbing. Every teardown, drain, wait,
validate and error path is still a process-wide stop-the-world, and each is invoked once per
query. `query_lifecycle_mutex_` is the only thing keeping them from firing.

Two consequences worth stating plainly:

1. **The feature is currently unreachable.** `_query_task_global_states` and `_query_states` can
   never hold more than one live entry via SQL, so none of the new per-query code has ever run
   under real concurrency.
2. **Some issues are live today**, not latent — the downgrade monitor thread and the GPU manager
   loops are not slot-gated. These are marked **LIVE**.

---

## Group A — Cross-query blast radius

A per-query event tears down a shared subsystem. These are the loudest bugs and the first to fire.

| ID | Severity | Issue | Location |
|---|---|---|---|
| **A1** | Critical | `task_scheduler::terminate_query()` reports to the right handler and then calls `stop()` — closes the request channel, joins the management thread, stops **every** GPU executor. Nothing ever calls `start()` again. One query's creation error hangs every other query and every future query in the process. | `src/pipeline/task_scheduler.cpp:171-176` |
| **A2** | Critical | The creation-worker catch calls `stop()` **from inside its own pool worker**. `do_stop_thread_pool()` → `_bounded_pool->wait_all()` blocks until `active_ == 0`, but the caller *is* an active slot. Guaranteed self-deadlock; if it got past, `_bounded_pool.reset()` joins the calling thread with itself inside a `noexcept` function. | `src/creator/task_creator.cpp:688-693` → `:363-373` |
| **A3** | Critical | GPU task exception → `_task_creator->stop()`, interrupting the shared creation queue and tearing down the shared pool. Every other query's `schedule()` push then silently returns false. The `completion->report_error` on the next line already fails the correct query. | `src/pipeline/gpu_pipeline_executor.cpp:427, 432` |
| **A4** | Critical | Four `break`s in `manager_loop` (reservation failure, downgrade cancelled, post-downgrade reservation failure, local-state cast failure) exit the `while (_running)` loop permanently. `_running` stays `true`, nothing restarts it, that GPU stops dispatching for **all** queries, and the popped task is dropped with no error. | `src/pipeline/gpu_pipeline_executor.cpp:196, 239, 261, 295` |
| **A5** | Critical | `wait_for_completion(query_id)` is per-query in name only: `_task_creator->stop_thread_pool()` halts creation for all queries; `_task_queue.size() != 0 → throw` sees **every** query's tasks, so A fails because B has work queued; `gpu_exec->wait_and_validate_empty()` interrupts and waits across all queries. | `src/pipeline/task_scheduler.cpp:234, 237, 250` |
| **A6** | Critical | `drain_after_error(query_id)` likewise: `_task_queue.drain()` **unfiltered** (twice), `gpu_exec->drain_and_wait()` for all devices, plus a global `stop_thread_pool()`/`start_thread_pool()` pair. One query's error destroys every other query's queued work → they hang with no error. | `src/pipeline/task_scheduler.cpp:197, 201, 205-207, 218, 220` |
| **A7** | Critical **LIVE** | `run_mandatory_cleanup` calls `executor->drain()` on **every** downgrade executor on **every** query end. `drain()` calls `cancel_pending_requests()`, which sets an exception on every queued promise — including query B's, whose GPU manager thread is blocked in `request_downgrade(...).get()`. B's manager thread then takes the `break` at `:239` (A4) and that GPU dies. | `src/sirius_context.cpp:322-324` → `src/downgrade/downgrade_executor.cpp:129-142, 496-506` |
| **A8** | High | `batch_telemetry_registry::on_query_end()` takes no query id. It consumes every live placement across all 16 shards as `reason=query_end` and then `impl_->ports.clear()`. Query A's end silently truncates query B's telemetry for the rest of B's life. | `src/sirius_context.cpp:331` → `src/telemetry/batch_telemetry.cpp:475-497` |
| **A9** | High | `multi_index_priority_queue::push` returns `false` and destroys the item when interrupted. `task_creator::schedule`/`schedule_lookahead` and `task_scheduler::schedule` all **discard** the return. Every interrupt window in A3/A5/A6 therefore loses work with zero trace — a dropped task is a pipeline that never finishes, i.e. a silent hang. | `src/creator/task_creator.cpp:404, 414, 456`; `src/pipeline/task_scheduler.cpp:112` |
| **A10** | Critical | `manager_loop` has no try/catch but throws (`internal_exception` at `:139`) and calls throwing APIs (`make_reservation`, `acquire_stream`, telemetry). It is a `std::thread` entry function — an escaping exception aborts the process. `:139` is specifically reachable when a stale/foreign task reaches a device queue, i.e. exactly what multi-query drain bugs produce. | `src/pipeline/gpu_pipeline_executor.cpp:95-478` |

---

## Group B — Lifetime, ownership, use-after-free

| ID | Severity | Issue | Location |
|---|---|---|---|
| **B1** | Critical **LIVE** | **TIER-2 downgrade resurrects a drained task.** `mutable_pop_if` removes a task from the shared scheduler queue into a processing-thread local; ownership then crosses a *blocking* `_pool->reserve()` and a full H2D/D2H conversion; `~convertible_gpu_pipeline_task` pushes it **back**. If the query's `drain_query_tasks(q)` ran in that window, the task returns after the drain. Two distinct faults follow: the re-push runs `index_keys_for` → `pipe->get_source()->type` on an **already-destroyed plan** (B5), and the resurrected task holds raw `data_repository*` into a manager about to be erased. | `src/downgrade/downgrade_executor.cpp:297-314`; `src/include/data/convertible_gpu_pipeline_task.hpp:86-89` |
| **B2** | Critical **LIVE** | `downgrade_executor::drain()` **restarts the processing thread** as its last act (`_pool->resume(); _request_queue.reactivate(); _processing_thread = std::thread(...)`). Quiescence therefore expires *before* `erase(query_id)` runs 20 lines later. The monitor thread (~10 ms period, never stopped by `drain()`) reliably refills the queue. With >1 executor, executor[0] restarts while executor[1] is still draining. | `src/downgrade/downgrade_executor.cpp:138-141` vs `src/sirius_context.cpp:323` / `:345` |
| **B3** | High **LIVE** | TIER-1 sweep holds raw `shared_data_repository*` from `manager->get_repositories()` across a blocking `reserve()`. The `shared_ptr<manager>` protects the *manager*, not the repositories inside it — `erase()` → `clear_all_repositories()` destroys the `unique_ptr`s. The registry header states this precondition explicitly; nothing enforces it. | `src/downgrade/downgrade_executor.cpp:223-236`; `src/include/data/data_repository_manager_registry.hpp:124-142` |
| **B4** | High | `gpu_pipeline_task::_data_repos` is `vector<shared_data_repository*>` carried across every queue hop and re-copied on the OOM reschedule path (which includes a 50 ms sleep). Any task alive when its manager is erased UAFs in `publish_output`. | `src/include/pipeline/gpu_pipeline_task.hpp:276`; `src/pipeline/gpu_pipeline_executor.cpp:404-423` |
| **B5** | High | **The plan is destroyed before the drains.** `cleanup_internal` → `sirius_active_query.reset()` destroys `sirius_engine` and therefore `sirius_owned_plan`; only *then* does `window->finish()` run `run_mandatory_cleanup`. `sirius_pipeline::source`/`operators`/`sink` are non-owning refs into that freed plan, and the pipeline outlives it (held by `shared_ptr` from the task global state). `~gpu_pipeline_task` → `mark_task_completed()` → `notify_downstream_pipelines()` dereferences all of them. The code acknowledges the inversion in a comment. | `src/sirius_interface.cpp:124` vs `src/sirius_context.cpp:314-318`; `src/include/pipeline/sirius_pipeline.hpp:243-247` |
| **B6** | Critical | `terminate()` does `task_scheduler_.reset()` **before** stopping the downgrade executors and the creator pool. Each downgrade executor holds `_pipeline_task_queue` = a raw pointer into the destroyed `task_scheduler::_task_queue` and dereferences it in `processing_loop`; creator lambdas hold `_task_scheduler`. | `src/sirius_context.cpp:817-828`; `src/include/downgrade/downgrade_executor.hpp:210` |
| **B7** | High | `SiriusContext` member declaration order makes `task_scheduler_` destroy *after* `task_creator_` and `downgrade_executors_`, but `task_scheduler` holds raw pointers to both. Reachable when `initialize()` throws after `task_scheduler_` is constructed (`is_initialized_` stays false, so `~SiriusContext` skips `terminate()`). | `src/include/sirius_context.hpp:621-624` |
| **B8** | High | `drop_query_runtime_state_best_effort` resets the creator, scheduler queues and scan manager but **never erases the repository registry**. A failed cleanup therefore leaks the query's manager and every batch in it (GPU memory never returned) until `terminate()`, and the downgrade executors keep sweeping it on every monitor cycle forever. | `src/sirius_context.cpp:404-430` |
| **B9** | Medium | `data_repository_manager::get_repository()` returns a **reference into the map** and is the only method in the class that does not take `_mutex`. A concurrent `add_new_repository` rehash or `clear_all_repositories` races it. | `cucascade/include/cucascade/data/data_repository_manager.hpp:155-158` |
| **B10** | Medium | `~SiriusContext() noexcept` calls `terminate()`, which starts with `throw_if_not_initialized()` and goes on to `reset_all()` / `registry.clear()` / `memory_manager_->shutdown()`. Any throw → `std::terminate` during DB teardown. | `src/sirius_context.cpp:134-137` |
| **B11** | High | `prefetching_cache` unlocks the map mutex and *then* uses the iterator (`lk.unlock(); ...; return *it->second;`). A concurrent insert rehashes `_file_cache` and the reader touches a freed bucket. Element pointers are stable; iterators are not. Single-query runs first-touch everything in one prepare, so this only becomes routine with two queries opening different files. | `src/io/cache/prefetching_cache.cpp:311-322, 385-390` |

---

## Group C — Deadlock and lock ordering

| ID | Severity | Issue | Location |
|---|---|---|---|
| **C1** | Critical | `stop_thread_pool()` holds `_global_state_mutex` across `_manager_thread.join()`; the manager thread calls `get_query_task_global_state()`, which takes **the same mutex**. Manager blocks on the mutex, stopper blocks in `join()` holding it. Reachable on **every query completion** (`wait_for_completion` → `stop_thread_pool`), and far likelier with a second query keeping the creation queue non-empty. | `src/creator/task_creator.cpp:375-379` → `:369`, vs `:486` → `:112` |
| **C2** | High | `task_creator::stop()` does **not** take `_global_state_mutex` while `stop_thread_pool()`/`start_thread_pool()` do. A worker calling `stop()` can be inside `do_stop_thread_pool()` while the DuckDB thread — seeing the `_running` CAS already false — returns immediately and calls `start_thread_pool()`, reassigning `_bounded_pool` and `_manager_thread` under the other thread. | `src/creator/task_creator.cpp:340-344` vs `:346-361`, `:375-379` |
| **C3** | Critical | `k_max_concurrent_queries = 1`, and the scan pool is sized `num_threads + k_max_concurrent_queries`. Each query parks one **blocking** coalescer sequencer in `queue.wait_dequeue`, unblocked only by that query's own split tasks on the same pool. At Q ≥ pool size this is a hard deadlock. Exceeding the cap currently only emits `SIRIUS_LOG_WARN`. The header carries its own `TODO: … RAISE IT`. | `src/include/scan_manager/sirius_scan_manager.hpp:284-292`; `src/scan_manager/sirius_scan_manager.cpp:272-280, 596-610` |
| **C4** | High | There is exactly **one** manager thread per GPU, and it performs both a blocking `make_reservation` and a blocking downgrade `.get()` while holding a reserved pool slot. One query's memory-hungry task blocks every other query's dispatch to that GPU. If the memory that would be released requires a downstream task to be dispatched, it livelocks; only the `IDLE` fallback in cucascade prevents a hard hang in the all-quiet case. | `src/pipeline/gpu_pipeline_executor.cpp:187, 234`; `cucascade/src/memory/memory_space.cpp:258-268` |

---

## Group D — Gaps in the per-query tracking that already exists

| ID | Severity | Issue | Location |
|---|---|---|---|
| **D1** | High | `get_operator_for_next_task(node)` runs on the manager thread **before** `enter_in_flight()` and recursively calls `node->get_next_task_hint()`. `drain_pending_tasks(Q)` can see `in_flight == 0` and return while that dereference is in progress — a hole in the exact invariant the mechanism exists to provide. | `src/creator/task_creator.cpp:493` vs `:499` |
| **D2** | High | The creation queue's key extractor dereferences `request.node->type` **at push time**, inside the queue mutex. A `schedule()` racing teardown reads a freed operator. Not covered by the in-flight counter. | `src/creator/task_creator.cpp:58-62` |
| **D3** | Medium | `schedule_lookahead` hard-codes `_query_task_global_states.begin()` — only the *oldest* query ever gets lookahead, so every newer query starts cold. It also dereferences operators under only `lookahead_mutex`, uncounted by the in-flight tracker. Carries its own TODO. | `src/creator/task_creator.cpp:419-459` |
| **D4** | Medium | `if (_task_queue.empty()) { schedule_lookahead(*_ready_devices.begin()); }` — `_ready_devices` can be empty when woken by a `task_available` event. B1's extraction window makes `_task_queue.empty()` return true while a task is temporarily held by a downgrade worker, so concurrency makes this reachable routinely. | `src/pipeline/task_scheduler.cpp:309-313` |
| **D5** | High | `runtime_unavailable_` is a process-wide permanent latch set by *one* query's cleanup failure. Defensible when only one query can be mid-cleanup; with N queries one unlucky query poisons N−1 healthy ones and every future query. It conflates "this query's state is corrupt" with "the shared runtime is corrupt". | `src/include/sirius_context.hpp:575`; `src/sirius_context.cpp:335-338` |
| **D6** | Medium | `_monitor_request_enqueued` latches `true` and is cleared **only** inside `processing_loop`. `cancel_pending_requests()` eats queued monitor requests without resetting it, and the push returns `false` when interrupted. Any `drain()` that races the monitor leaves the flag stuck → **automatic memory-pressure downgrade for that space is dead for the rest of the process**, and every later query OOMs instead of spilling. | `src/downgrade/downgrade_executor.cpp:456, 473-474, 496-506` |

---

## Group E — Shared mutable configuration

| ID | Severity | Issue | Location |
|---|---|---|---|
| **E1** | High | `operator_params` is one plain non-atomic struct per `DatabaseInstance`, written by ~20 `SET` callbacks and read mid-plan and mid-execution. The **only** thing serializing those writes is the slot this project removes. Semantics are already wrong today: B's `SET scan_task_batch_size` silently changes A's next query. | `src/include/sirius_config.hpp:245`; `src/sirius_extension.cpp:1788-1794` and 20 setters; readers `src/sirius_engine.cpp:211`, `src/planner/sirius_physical_plan_generator.cpp:83` |
| **E2** | High | `duckdb::Config` process-wide static variables written by `SET` callbacks with **no** guard at all. `EXPRESSION_EVALUATOR_STRATEGY` is read as a **default argument** on every `expression_evaluator` construction, so B's `SET` can change strategy between two operators of A's plan. The `LOG_*` `std::string`s are worse — concurrent reassign/read is a torn read or UAF on the string buffer. | `src/include/config.hpp:30-77`; writers `src/sirius_extension.cpp:1638-1758, 1859-1894` |
| **E3** | High | Five compression-config setters write `get_config().get_compression_config()` with **no slot**, inconsistent with their `operator_params` neighbours. The reader (`PinTableFunction`) does hold a window and passes the `std::string` straight into `fs::directory_iterator`. | `src/sirius_extension.cpp:1942-1983` vs `:1287-1337` |
| **E4** | High | `PhysicalSiriusExecution::logical_plan_` is `mutable` and `reset()` from a `const` source method. DuckDB shares one prepared `PhysicalOperator` across executions and calls `GetDataInternal` `const` precisely because it assumes re-entrancy. Two concurrent `EXECUTE`s of the same transparent prepared statement → one thread dereferences while the other destroys. | `src/include/transparent/physical_sirius_execution.hpp:74`; `src/transparent/physical_sirius_execution.cpp:198, 202` |
| **E5** | Medium | `plan_register::global()` check-then-act across three separate critical sections, keyed by bare table name with no catalog/schema. Two connections pinning same-named tables in different ATTACHed DBs both miss, both write, and the loser pins with the other database's plan DSL. | `src/sirius_extension.cpp:1293, 1301, 1317` |
| **E6** | Medium | `get_target_ctas()` uses `static int cached_device = -1; static uint32_t cached = 0;` with a non-atomic check-then-use — the only genuine mutable function-local static in the operator/CUDA layer. Two queries race both words; on multi-GPU the pair can tear and return the other device's CTA count. | `src/include/cuda/scan/strings/common.cuh:150-158` |
| **E7** | Medium | `SiriusTableFunctionData::original_config` saves/restores `ClientConfig` on **shared bind data**. Two concurrent executions of one `gpu_execution(...)` prepared statement clobber each other's saved config, leaving a connection with `enable_optimizer` permanently flipped. | `src/sirius_extension.cpp:247, 252, 266` |

---

## Group F — Fairness and performance under concurrency

These do not corrupt anything, but concurrency is the point — a correct engine that serializes is not a win.

| ID | Severity | Issue | Location |
|---|---|---|---|
| **F1** | High | `query_id` is packed into the **high** bits of the scheduling priority and the queue pops lowest-first, so **every** task of query 1 outranks **every** task of query 2. Query 2 runs only when query 1 has nothing dispatchable; if query 1 waits on memory only query 2 could release, it livelocks. | `src/include/query_id.hpp:66-69`; `src/creator/task_creator.cpp:256-260` |
| **F2** | Medium | The plan-time `SlotGuard` makes plan generation mutually exclusive with *execution*. Planning is CPU-only and needs only a read view of the pin table, which `_pinned_entries_mutex` + `shared_ptr<pinned_entry>` now provide. Today query A executing blocks query B from even being planned. The cheapest lock to narrow. | `src/sirius_context.cpp:1236` |
| **F3** | Medium | Prefetch-cache query epoch is one global `_ticker`; scoring is newest-wins, so B's `prepare_for_query` demotes every chunk A prefetched-but-not-yet-read to eviction tier 0. Documented as a KNOWN GAP, not fixed. Related: `chunk_lifecycle::on_request` resets A's counters when B touches the same chunk. | `src/include/io/cache/prefetching_cache.hpp:228`; `src/io/cache/prefetching_cache.cpp:659`; `src/include/io/cache/types.hpp:368-370, 407-412` |
| **F4** | Medium | One `_preparation_thread`, one `_prefetch_thread`, one `_evictor_thread`, one process-wide `_rate_limiter` with no query dimension. Strict FIFO: query A's 20k chunk requests fully block query B's first prefetch. | `src/include/io/cache/prefetching_cache.hpp:251-262`; `src/exec/admission_control.cpp` |
| **F5** | Medium | The pin/unpin guard rejects on `use_count() > 1`, but the extra reference comes from an **all-entries** snapshot. Query A anywhere inside `try_match_cached_entry` holds a ref to *every* pinned entry, so connection B's `PIN` on a completely unrelated table spuriously fails. | `src/scan_manager/sirius_scan_manager.cpp:1002-1007, 1272-1276, 1482-1491` |
| **F6** | Medium | Sizing reads whole-device/whole-host free memory: sort partition count from `get_available_memory()`, result-collector host space from `max_element` over free bytes then an unreserved `make_reservation_or_null`. Two concurrent sorts each size to a fraction of the *same* free bytes and overshoot; two collectors pick the same host space and one gets `nullptr`. | `src/op/sirius_physical_sort_sample.cpp:266`; `src/op/sirius_physical_result_collector.cpp:160-193` |
| **F7** | Medium | `cudaDeviceSynchronize()` in the dynamic-filter publish fallback blocks **all** of query B's kernels on every stream. Plus default-stream work in the orphan-pairing path, parquet footer read, and string decode — the build does not enable per-thread default stream, so the legacy default stream implicitly synchronizes with every blocking stream on the device. | `src/op/sirius_physical_hash_join.cpp:2023, 1183`; `src/op/scan/parquet_gpu_ingestible.cpp:461` |
| **F8** | Medium | Downgrade is strictly serial (one request at a time, ending in `_pool->wait_all()`), and TIER 2 sizes its budget from the whole shared queue and pops **other queries'** ready tasks. Defensible — memory pressure is global — but it is a throughput cliff and it briefly removes a co-tenant's dispatchable work. | `src/downgrade/downgrade_executor.cpp:144-398` |
| **F9** | Medium | Shared blocking primitives with no query dimension that will become the next bottleneck once the slot is gone: cucascade's `memory_reservation_manager::_wait_cv` (blocks when no space can satisfy a reservation) and `exclusive_stream_pool::_cv` (blocks when all streams are checked out). | `cucascade/include/cucascade/memory/memory_reservation_manager.hpp:276-277`; `cucascade/include/cucascade/memory/stream_pool.hpp:99-100` |

---

## Group G — Test infrastructure

| ID | Severity | Issue |
|---|---|---|
| **G1** | Critical | **No reusable concurrency harness exists.** All four new branch tests (`test_task_creator_query_state.cpp`, `test_scan_manager_query_state.cpp`, `test_per_query_completion_handler.cpp`, `test_task_index_keys.cpp`) are **single-threaded** — they prove state *partitioning*, not race-freedom. `test_scan_manager_query_state.cpp:181` is literally titled *"concurrent queries do not collide on operator id"* and spawns zero threads. |
| **G2** | High | The only multi-threaded multi-connection code lives in an anonymous namespace inside `test/cpp/integration/test_query_lifecycle_slot.cpp` (start gate, `async_query_result`, `scoped_blocking_window_log_sink`, `held_window_threads` — ~200 lines). Nothing in `test/cpp/utils/` spawns a thread. Mainstream fixtures (`GpuExecutionFixture`, `GPUExecutionFixtureBase`) hold **one** connection and have no `make_connection()`. |
| **G3** | High | The harness actively **forbids** parallel tests: `shared_env_listener` and `scoped_mgpu_env`'s ctor *pause* (destroy the `DuckDB` instance of) every other env before each test. |
| **G4** | Medium | `test_query_lifecycle_slot.cpp` exists to prove **serialization** — e.g. `run_ac2_gpu_single_flight` asserts the second query *waits on the slot*. These assertions need to be re-expressed against the new semantics, not deleted. |
| **G5** | Medium | `run_ac13_concurrent_logging` is fully implemented (`:1919-2045`) and dispatchable (`:2082`) but **no `TEST_CASE` invokes it**. |
| **G6** | Low | `test/cpp/integration/test_transparent_execution.cpp:304-366` asserts the *old* whole-query serialization semantics (`REQUIRE(is_query_lifecycle_active())`, `REQUIRE(second_elapsed >= 150ms)`). It is orphaned (not in `CMakeLists.txt`, listed in `orphan_test_allowlist.txt:13`) and uses a nonexistent `pg_sleep()`. Dead — but it is the **only** file with a fixture-level `make_connection()`, so it is the natural salvage point. |
| **G7** | Medium | Zero SQLLogic concurrency (`concurrentloop` appears nowhere). No TPC-H/TPC-DS query is ever run concurrently with another. |

---

## Group H — Dead code, drift, and hygiene

| ID | Issue | Location |
|---|---|---|
| **H1** | `task_creator::_client_context` — uninitialized raw pointer, not in the ctor init list, never read. Superseded by the per-query `query_task_global_state::client_context`. A trap waiting for someone to "fix" it by reading it. | `src/include/creator/task_creator.hpp:240` |
| **H2** | `sirius_engine::wait_for_query_finish()` + `query_finish_mutex` / `query_finish_cv` / `query_finished` — **no definition, no caller**. Delete before someone turns them into a global barrier. | `src/include/sirius_engine.hpp:102-108` |
| **H3** | `task_completion_message_queue` — never instantiated. | `src/include/task_completion.hpp:108, 111` |
| **H4** | `SiriusContext::mutable std::mutex mutex_` — **never locked anywhere**. | `src/include/sirius_context.hpp:560` |
| **H5** | `exec::inspectable_mpsc` — orphaned; referenced only by its own test and a stale comment. | `src/include/exec/inspectable_mpsc.hpp` |
| **H6** | `task_creator::schedule(node, query_id)` — zero callers. | `src/include/creator/task_creator.hpp:180` |
| **H7** | `task_creation_request::device_id` — set only by `schedule_lookahead`; the queue is only ever `pop()`ed, never `try_pop_from(gpu_index{})`. | `src/include/creator/task_creator.hpp:72` |
| **H8** | Query ids wrap at 2³² and the priority packing masks to **31 bits**. A wrapped id collides with a live query → `create_for_query` throws "already registered"; at 2³¹ the priority ordering silently inverts. | `src/include/sirius_context.hpp:579`; `src/include/query_id.hpp:66` |
| **H9** | Doc drift: five files describe deleted APIs (`pipeline-execution.md:296`, `architecture-overview.md:130`, `task-creator.md:31,239`, `scan.md:16,158`, `optimizations.md:25`). No doc exists for the new per-query model. Dangling doxygen: `/// \brief Get the current query.` now sits above `get_config()` (`sirius_context.hpp:498`). Header comment claims `std::map` above an `unordered_map` (`task_scheduler.hpp:216-219`). |
| **H10** | Misleading aliases: `using shared_data_repository = data_repository;` and `shared_data_repository_manager` imply shared ownership where there is none — in exactly the area under discussion. | `cucascade/.../data_repository.hpp:319`, `data_repository_manager.hpp:251` |
| **H11** | Dead APIs in cucascade: `data_repository_manager::for_each_repository` (zero callers). Dead test accessor `testable_task_creator::query_id()`. Uninitialized members `sirius_physical_table_scan::physical_table_scan` / `column_size` / `mask_size`. | various |

---

## Dependency notes for planning

A few hard ordering constraints that every option must respect:

1. **C3 (scan pool sizing) gates every end-to-end concurrent test.** Nothing can be validated at
   Q > 1 until the pool is sized for it.
2. **A4 must be fixed before A7**, or fixing the downgrade drain just relocates the hang.
3. **B5 (plan lifetime) is upstream of B1's second fault.** Gating the TIER-2 re-push stops the
   resurrection, but any surviving task still dereferences a dead plan on its completion path.
4. **C1 must be fixed before A5/A6 are exercised**, because both call `stop_thread_pool()` today.
5. **A9 is a diagnosability multiplier** — until push failures are loud, every other bug in
   Group A presents as an unexplained hang. Worth landing early even though it fixes nothing.
6. **G1/G2 gate confidence in everything else.** Without a harness, no fix in this register can be
   demonstrated to work; the four existing "query_state" tests would all still pass with every
   Group A bug present.
