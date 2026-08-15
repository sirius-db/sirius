/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "data/data_repository_manager_registry.hpp"
#include "exec/bounded_thread_pool.hpp"
#include "exec/config.hpp"
#include "exec/interruptible_mpmc.hpp"
#include "exec/multi_index_priority_queue.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "parallel/task.hpp"
#include "query_id.hpp"

#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace sirius {
namespace parallel {

/**
 * @brief A request to free GPU memory by downgrading data batches.
 *
 * Enqueued into the executor's request queue by the monitor loop or by
 * external callers. The processing loop dequeues one request at a time
 * and dispatches batch downgrades to the thread pool.
 * WARNING: The predicate function may be called by multiple threads concurrently.
 * Thread safety of the predicate function is the responsibility of the caller.
 */
struct downgrade_request {
  std::function<bool()> predicate;
  std::promise<size_t> result;
  std::atomic<size_t> bytes_freed{0};
  std::atomic<size_t> batches_downgraded{0};
  std::atomic<bool> satisfied{false};
  bool is_monitor_request{false};
  /// The query this request works FOR (its waiter), so the per-query drain can fail exactly
  /// the ending query's promises and no one else's. `make_query_id(0)` — a value execution
  /// windows never mint (window ids start at 1) — marks unattributed requests: the monitor's
  /// own pressure requests and external byte-target requests.
  sirius::query_id_t query_id{sirius::make_query_id(0)};
};

/**
 * @brief Executor specialized for performing memory downgrade operations across tier hierarchies.
 *
 * Each downgrade_executor is bound to a specific memory space (e.g., GPU:0, HOST:0) and
 * monitors it for memory pressure. When `should_downgrade_memory()` triggers, it enqueues
 * a downgrade_request. The processing thread dequeues requests sequentially, collects
 * candidate batches, and dispatches downgrade tasks to the thread pool.
 *
 * This is a standalone class with its own thread pool and request queue.
 */
class downgrade_executor {
 public:
  /**
   * @brief Constructs a new downgrade_executor bound to a specific memory space.
   *
   * @param config Configuration for the thread pool (thread count, etc.)
   * @param data_repo_registry Registry of every in-flight query's data repository manager
   * @param space_id The memory space this executor is responsible for downgrading FROM
   * @param memory_space Pointer to the memory space (for pressure queries; nullptr disables
   * monitor)
   * @param reservation_manager Reference to the memory reservation manager
   * @param pipeline_task_queue Optional pointer to pipeline task queue for tiered fallback
   */
  explicit downgrade_executor(
    exec::downgrade_executor_config config,
    sirius::data::data_repository_manager_registry& data_repo_registry,
    cucascade::memory::memory_space_id space_id,
    cucascade::memory::memory_space* memory_space,
    sirius::memory::sirius_memory_reservation_manager& reservation_manager,
    sirius::exec::multi_index_priority_queue<sirius::parallel::itask>* pipeline_task_queue =
      nullptr);

  ~downgrade_executor();

  // Non-copyable and non-movable
  downgrade_executor(const downgrade_executor&)            = delete;
  downgrade_executor& operator=(const downgrade_executor&) = delete;
  downgrade_executor(downgrade_executor&&)                 = delete;
  downgrade_executor& operator=(downgrade_executor&&)      = delete;

  void start();
  void stop();

  // NOTE (step 7): the global drain() — cancel EVERY queued request and stop-join-restart the
  // processing thread — is gone. Its last legitimate callers were terminate-adjacent paths and
  // tests: terminate() stops executors outright via stop(), per-query cleanup uses
  // drain(query_id), and tests use stop()/start() or drain(make_query_id(0)) (the id monitor
  // and external byte-target requests carry). Nothing else needed a whole-executor quiesce
  // whose quiescence expired the moment it returned.

  /**
   * @brief Per-query drain, for one query's end-of-window cleanup.
   *
   * Fails ONLY @p query_id's queued promises (unblocking that query's own waiters, which are
   * being torn down anyway) and waits for an in-flight request of that query to complete. It
   * never interrupts the request queue, the pool, or the processing/monitor threads, so peer
   * queries' pending spills and the monitor's pressure response proceed unaffected — and it
   * needs no lifecycle serialization because it never touches thread lifetimes.
   *
   * Precondition: the query is quiesced (no producer can enqueue new requests for it), which
   * run_mandatory_cleanup guarantees before calling this. Repository teardown needs no fence
   * at all: sweeps co-own every manager/repository/batch they borrow (step 6), so the
   * registry's erase() just drops its map entry.
   */
  void drain(sirius::query_id_t query_id);

  /**
   * @brief Wait until no downgrade request is being processed, regardless of owner.
   *
   * THE SURVIVING FENCE — this wait is about PLAN lifetime, not repository lifetime, and
   * shared-ownership repositories (step 6) deliberately did NOT subsume it:
   *
   * drain(query_id) waits only for the query's OWN in-flight request — but a PEER's (or the
   * monitor's) request sweeps by memory space, not by query, so its TIER-2 pass can hold the
   * ending query's task inside a convertible wrapper across a blocking conversion. When that
   * wrapper's RAII drop runs, the lifecycle gate (consulted with extraction-time keys) refuses
   * the re-push for a quiescing query and the wrapper DESTROYS the task instead — and
   * ~gpu_pipeline_task walks the task's plan (mark_task_completed ->
   * notify_downstream_pipelines over raw operator pointers). Plan parking (B5) only defers the
   * plan's death until cleanup; cleanup must still not destroy it while such a wrapper is
   * alive. Query-end cleanup therefore calls this after the per-query drains and BEFORE
   * destroying the parked plan: when it returns, every wrapper created by a request that was
   * in flight has been destroyed (a request joins its wrappers before completing), so one
   * final queue sweep leaves nothing of the query for a later request to find.
   *
   * A request popped but not yet published as in-flight can slip past this wait; that is
   * closed from the other side — TIER-2 extraction consults the query-lifecycle gate, so a
   * request starting after quiesce() never extracts the ending query's tasks. Those two
   * properties (extraction gate + this bounded wait) are exactly what makes destroying the
   * plan safe; keep them together.
   */
  void wait_inflight_request();

  /**
   * @brief Get the memory space this executor is responsible for.
   */
  cucascade::memory::memory_space_id get_space_id() const { return _space_id; }

  /**
   * @brief Asynchronously request GPU memory reclamation.
   *
   * Constructs a predicate that checks bytes_freed >= bytes and enqueues
   * a downgrade request. Returns immediately with a future.
   *
   * @param bytes Target bytes to free
   * @return std::future<size_t> Resolves to actual bytes freed (may be less than requested)
   */
  std::future<size_t> request_free_memory(size_t bytes);

  /**
   * @brief Synchronously request GPU memory reclamation.
   *
   * Blocks until the request completes and returns the actual bytes freed.
   *
   * @param bytes Target bytes to free
   * @return size_t Actual bytes freed (may be less than requested)
   */
  size_t request_free_memory_and_wait(size_t bytes);

  /**
   * @brief Set the pipeline task queue pointer for tiered downgrade scanning.
   *
   * Must be called before start(). Allows deferred wiring when the queue
   * is not available at construction time.
   *
   * @param pipeline_task_queue Pointer to the task_scheduler's task queue
   */
  void set_pipeline_task_queue(
    sirius::exec::multi_index_priority_queue<sirius::parallel::itask>* pipeline_task_queue);

  /**
   * @brief Bind the per-query lifecycle gate.
   *
   * Forwarded to every convertible_gpu_pipeline_task the TIER-2 sweep creates, so a task
   * extracted from the shared queue is dropped rather than re-pushed once its query starts
   * tearing down. Without one (the unit-test default) the re-push is ungated, as before.
   */
  void set_query_lifecycle_registry(sirius::exec::query_lifecycle_registry* registry) noexcept
  {
    _query_lifecycle = registry;
  }

  /**
   * @brief Asynchronously request a predicate-driven downgrade, unattributed.
   *
   * Equivalent to request_downgrade(make_query_id(0), predicate): the request belongs to no
   * query and is only ever cancelled by stop() or a drain(make_query_id(0)). Callers whose
   * waiter belongs to a query (the GPU pipeline executor's reservation paths) must use the
   * attributed overload so a per-query drain can find their request.
   *
   * @param predicate Callable returning true when the caller's condition is met
   * @return std::future<size_t> Resolves to total bytes freed
   */
  std::future<size_t> request_downgrade(std::function<bool()> predicate);

  /**
   * @brief Asynchronously request a predicate-driven downgrade on behalf of @p query_id.
   *
   * Dispatches batch downgrades until the predicate returns true or candidates
   * are exhausted. In-flight batches finish naturally. If @p query_id's cleanup runs while
   * the request is still queued, drain(query_id) fails the returned future's promise.
   *
   * @param query_id  The query whose waiter blocks on the returned future
   * @param predicate Callable returning true when the caller's condition is met
   * @return std::future<size_t> Resolves to total bytes freed
   */
  std::future<size_t> request_downgrade(sirius::query_id_t query_id,
                                        std::function<bool()> predicate);

  /**
   * @brief Whether a DISK tier is configured (an effectively unbounded spill sink).
   *
   * Used by callers (e.g. the GPU pipeline executor) to decide whether an unsatisfiable
   * reservation can ever be relieved by spilling, or whether retrying is futile.
   */
  bool has_disk_tier() const;

  /**
   * @brief Number of downgrade requests the monitor loop has issued (test-only).
   *
   * Lets tests observe whether the monitor has gone quiescent (count stops rising)
   * or is actively issuing requests.
   */
  size_t monitor_requests_issued_for_testing() const
  {
    return _monitor_requests_issued.load(std::memory_order_relaxed);
  }

 private:
  void processing_loop();
  void monitor_loop();
  void cancel_pending_requests();

  /**
   * @brief Fail every queued request belonging to @p query_id; leave the rest queued.
   *
   * Pops the whole queue and re-pushes the survivors. Safe against the concurrently popping
   * processing thread (anything it takes is covered by drain(query_id)'s in-flight wait) and
   * bounded: the only unattributed repeating producer, the monitor, keeps at most one request
   * outstanding.
   */
  void cancel_pending_requests_for_query(sirius::query_id_t query_id);

  /**
   * @brief Fail one request's promise and, for a monitor request, re-arm the monitor.
   *
   * Every path that destroys a request without the processing loop seeing it MUST go through
   * here: the processing loop is the only other place that clears _monitor_request_enqueued,
   * so silently eating a monitor request would leave the flag latched true and
   * memory-pressure downgrade for this space dead for the rest of the process.
   */
  void fail_request(std::unique_ptr<downgrade_request> request);

  /**
   * @brief Whether a downgrade from this executor's source tier could plausibly free memory.
   *
   * DISK is an effectively unbounded sink, so if it is configured a downgrade can always make
   * progress. Otherwise progress is only possible if some HOST space still has capacity to accept
   * data. Re-evaluated on every monitor cycle so the monitor backs off when stuck and resumes the
   * instant conditions change -- no latched state, no missed wakeup.
   */
  bool has_viable_downgrade_target() const;

 private:
  exec::downgrade_executor_config _config;
  std::unique_ptr<exec::bounded_thread_pool> _pool;
  exec::interruptible_mpmc<std::unique_ptr<downgrade_request>> _request_queue;
  std::thread _processing_thread;
  std::thread _monitor_thread;
  /// Which request the processing loop is currently executing (it handles one at a time).
  /// drain(query_id) waits on this so a query's cleanup proceeds only once the request its
  /// waiter is blocked on has fulfilled its promise. Guarded by _in_flight_mutex; the cv is
  /// notified on every clear.
  std::mutex _in_flight_mutex;
  std::condition_variable _in_flight_cv;
  bool _in_flight_active{false};
  sirius::query_id_t _in_flight_query{sirius::make_query_id(0)};
  std::atomic<bool> _monitor_request_enqueued{false};
  std::atomic<bool> _running{false};
  std::atomic<size_t> _monitor_requests_issued{0};
  std::unique_ptr<cucascade::memory::exclusive_stream_pool> _stream_pool;

  std::mutex _monitor_cv_mutex;
  std::condition_variable _monitor_cv;

  /// Every in-flight query's repository manager. Memory pressure is a global condition, so
  /// spill candidates are drawn from across all live queries, not just one.
  sirius::data::data_repository_manager_registry& _data_repo_registry;
  cucascade::memory::memory_space_id _space_id;
  cucascade::memory::memory_space* _memory_space;
  std::string _source_label;
  sirius::memory::sirius_memory_reservation_manager& _reservation_manager;
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask>* _pipeline_task_queue{nullptr};
  /// Non-owning; owned by SiriusContext and outlives this executor. Null in unit tests.
  sirius::exec::query_lifecycle_registry* _query_lifecycle{nullptr};
};

}  // namespace parallel
}  // namespace sirius
