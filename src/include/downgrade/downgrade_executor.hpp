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

#include "downgrade/downgrade_task.hpp"
#include "exec/bounded_thread_pool.hpp"
#include "exec/config.hpp"
#include "exec/interruptible_mpmc.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"

#include <cuda_runtime_api.h>

#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <atomic>
#include <functional>
#include <future>
#include <memory>
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
 */
struct downgrade_request {
  size_t target_bytes{0};
  std::function<bool()> predicate;
  std::promise<size_t> result;
  std::atomic<size_t> bytes_freed{0};
  std::atomic<size_t> batches_downgraded{0};
  std::atomic<bool> satisfied{false};
};

/**
 * @brief Information about a repository to consider for downgrade candidate selection.
 */
struct downgrade_repository_info {
  cucascade::shared_data_repository* repo;
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
   */
  explicit downgrade_executor(
    exec::downgrade_executor_config config,
    cucascade::shared_data_repository_manager& data_repo_mgr,
    cucascade::memory::memory_space_id space_id,
    cucascade::memory::memory_space* memory_space,
    sirius::memory::sirius_memory_reservation_manager& reservation_manager);

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
   * @brief Asynchronously request a predicate-driven downgrade.
   *
   * Collects up to target_bytes worth of candidates and dispatches batch
   * downgrades until the predicate returns true or candidates are exhausted.
   * In-flight batches finish naturally.
   *
   * @param target_bytes Approximate bytes to collect as candidates
   * @param predicate Callable returning true when the caller's condition is met
   * @return std::future<size_t> Resolves to total bytes freed
   */
  std::future<size_t> request_downgrade(size_t target_bytes, std::function<bool()> predicate);

 private:
  void processing_loop();
  void monitor_loop();

  std::vector<std::weak_ptr<cucascade::data_batch>> collect_all_candidates(
    const std::vector<downgrade_repository_info>& repositories, size_t amount_to_downgrade);

  static size_t get_repo_data_size_on_tier(cucascade::shared_data_repository* repo,
                                           cucascade::memory::Tier tier);

  static bool is_partition_active(cucascade::shared_data_repository* repo, size_t partition_idx);

  static std::vector<std::weak_ptr<cucascade::data_batch>> collect_candidates_from_partition(
    cucascade::shared_data_repository* repo,
    size_t partition_idx,
    cucascade::memory::memory_space_id source_space,
    size_t max_bytes,
    size_t& collected_bytes);

 private:
  exec::downgrade_executor_config _config;
  std::unique_ptr<exec::bounded_thread_pool> _pool;
  exec::interruptible_mpmc<std::unique_ptr<downgrade_request>> _request_queue;
  std::thread _processing_thread;
  std::thread _monitor_thread;
  std::atomic<bool> _running{false};
  cudaStream_t _stream{nullptr};

  cucascade::shared_data_repository_manager& _data_repo_mgr;
  cucascade::memory::memory_space_id _space_id;
  cucascade::memory::memory_space* _memory_space;
  sirius::memory::sirius_memory_reservation_manager& _reservation_manager;
};

}  // namespace parallel
}  // namespace sirius
