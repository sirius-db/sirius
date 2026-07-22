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

#include "exec/bounded_thread_pool.hpp"
#include "exec/config.hpp"
#include "exec/inspectable_priority_queue.hpp"
#include "exec/interruptible_mpmc.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "parallel/task.hpp"

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
   * @param data_repo_mgr Reference to the data repository manager
   * @param space_id The memory space this executor is responsible for downgrading FROM
   * @param memory_space Pointer to the memory space (for pressure queries; nullptr disables
   * monitor)
   * @param reservation_manager Reference to the memory reservation manager
   * @param pipeline_task_queue Optional pointer to pipeline task queue for tiered fallback
   */
  explicit downgrade_executor(
    exec::downgrade_executor_config config,
    cucascade::shared_data_repository_manager& data_repo_mgr,
    cucascade::memory::memory_space_id space_id,
    cucascade::memory::memory_space* memory_space,
    sirius::memory::sirius_memory_reservation_manager& reservation_manager,
    sirius::exec::inspectable_priority_queue<sirius::parallel::itask>* pipeline_task_queue =
      nullptr);

  ~downgrade_executor();

  // Non-copyable and non-movable
  downgrade_executor(const downgrade_executor&)            = delete;
  downgrade_executor& operator=(const downgrade_executor&) = delete;
  downgrade_executor(downgrade_executor&&)                 = delete;
  downgrade_executor& operator=(downgrade_executor&&)      = delete;

  void start();
  void stop();
  void drain();

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
    sirius::exec::inspectable_priority_queue<sirius::parallel::itask>* pipeline_task_queue);

  /**
   * @brief Asynchronously request a predicate-driven downgrade.
   *
   * Dispatches batch downgrades until the predicate returns true or candidates
   * are exhausted. In-flight batches finish naturally.
   *
   * @param predicate Callable returning true when the caller's condition is met
   * @return std::future<size_t> Resolves to total bytes freed
   */
  std::future<size_t> request_downgrade(std::function<bool()> predicate);

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
  std::atomic<bool> _monitor_request_enqueued{false};
  std::atomic<bool> _running{false};
  std::atomic<size_t> _monitor_requests_issued{0};
  std::unique_ptr<cucascade::memory::exclusive_stream_pool> _stream_pool;

  std::mutex _monitor_cv_mutex;
  std::condition_variable _monitor_cv;

  cucascade::shared_data_repository_manager& _data_repo_mgr;
  cucascade::memory::memory_space_id _space_id;
  cucascade::memory::memory_space* _memory_space;
  std::string _source_label;
  sirius::memory::sirius_memory_reservation_manager& _reservation_manager;
  sirius::exec::inspectable_priority_queue<sirius::parallel::itask>* _pipeline_task_queue{nullptr};
};

}  // namespace parallel
}  // namespace sirius
