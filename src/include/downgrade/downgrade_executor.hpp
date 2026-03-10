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

#include "downgrade/downgrade_queue.hpp"
#include "downgrade/downgrade_task.hpp"
#include "parallel/task_executor.hpp"
#include "task_completion.hpp"

#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <thread>
#include <vector>

namespace sirius {
namespace parallel {

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
 * monitors it for memory pressure. When `should_downgrade_memory()` triggers, it automatically
 * iterates all repositories in the data_repository_manager and schedules downgrade tasks.
 *
 * The executor runs a monitor thread that polls the memory space and a worker thread that
 * executes the downgrade tasks (GPU→HOST copies, etc.).
 */
class downgrade_executor : public itask_executor {
 public:
  /**
   * @brief Constructs a new downgrade_executor bound to a specific memory space.
   *
   * @param config Configuration for the task executor (thread count, retry policy, etc.)
   * @param data_repo_mgr Reference to the data repository manager
   * @param space_id The memory space this executor is responsible for downgrading FROM
   * @param memory_space Pointer to the memory space (for pressure queries; nullptr disables
   * monitor)
   * @param reservation_manager Reference to the memory reservation manager
   */
  explicit downgrade_executor(
    task_executor_config config,
    cucascade::shared_data_repository_manager& data_repo_mgr,
    cucascade::memory::memory_space_id space_id,
    cucascade::memory::memory_space* memory_space,
    sirius::memory::sirius_memory_reservation_manager& reservation_manager)
    : itask_executor(std::make_unique<downgrade_task_queue>(), config),
      _data_repo_mgr(data_repo_mgr),
      _space_id(space_id),
      _memory_space(memory_space),
      _reservation_manager(reservation_manager)
  {
  }

  /**
   * @brief Get the memory space this executor is responsible for.
   */
  cucascade::memory::memory_space_id get_space_id() const { return _space_id; }

  ~downgrade_executor() override = default;

  // Non-copyable and non-movable
  downgrade_executor(const downgrade_executor&)            = delete;
  downgrade_executor& operator=(const downgrade_executor&) = delete;
  downgrade_executor(downgrade_executor&&)                 = delete;
  downgrade_executor& operator=(downgrade_executor&&)      = delete;

  void schedule_downgrade_task(std::unique_ptr<downgrade_task> downgrade_task)
  {
    this->schedule(std::move(downgrade_task));
  }

  void worker_loop(int worker_id) override;
  void start() override;
  void stop() override;

  /**
   * @brief Drain all pending and in-flight downgrade tasks.
   *
   * Must be called before clearing data repositories (e.g., at QueryEnd) to ensure
   * no downgrade tasks hold shared_ptr<data_batch> references to batches that are
   * about to be destroyed.
   *
   * Closes the queue (discarding pending tasks), waits for in-flight tasks to finish,
   * then re-opens the queue so new tasks can be scheduled for the next query.
   */
  void drain();

  /**
   * @brief Perform a downgrade pass with an explicit list of repositories.
   *
   * Uses this executor's bound memory space as the source tier and its stored
   * reservation manager. Walks through the provided repositories using the
   * prioritization rules:
   * 1. Partitioned repos first, then by descending data size on this tier
   * 2. Within each repo, iterate partitions from last to first
   * 3. First pass: non-active partitions; second pass: active partitions
   *
   * @param repositories Vector of repository pointers to scan
   * @param amount_to_downgrade Target bytes of data to downgrade
   * @return size_t Number of downgrade tasks scheduled
   */
  size_t run_downgrade_pass(std::vector<downgrade_repository_info> repositories,
                            size_t amount_to_downgrade);

 private:
  /**
   * @brief Monitor loop that polls the memory space for pressure and triggers downgrades.
   *
   * Iterates all repositories via data_repository_manager::for_each_repository().
   */
  void monitor_loop();

  static size_t get_repo_data_size_on_tier(cucascade::shared_data_repository* repo,
                                           cucascade::memory::Tier tier);

  static bool is_partition_active(cucascade::shared_data_repository* repo, size_t partition_idx);

  static std::vector<std::shared_ptr<cucascade::data_batch>> collect_candidates_from_partition(
    cucascade::shared_data_repository* repo,
    size_t partition_idx,
    cucascade::memory::memory_space_id source_space,
    size_t max_bytes,
    size_t& collected_bytes);

 private:
  cucascade::shared_data_repository_manager& _data_repo_mgr;
  cucascade::memory::memory_space_id _space_id;
  cucascade::memory::memory_space* _memory_space;
  sirius::memory::sirius_memory_reservation_manager& _reservation_manager;
  task_completion_message_queue _message_queue;  ///< Owned; receives downgrade task completions

  std::thread _monitor_thread;
};

}  // namespace parallel
}  // namespace sirius
