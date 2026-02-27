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

#include "config.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "parallel/task_executor.hpp"
#include "task_completion.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/common.hpp>

#include <cstdint>
#include <memory>

namespace sirius {
namespace parallel {

/**
 * @brief Global state shared across all downgrade tasks in an operation.
 *
 * This class holds references to shared resources that all downgrade tasks within
 * an operation need access to, including the data repository for storing results
 * and the message queue for task completion notifications.
 */
class downgrade_task_global_state : public itask_global_state {
 public:
  /**
   * @brief Construct a new downgrade_task_global_state object
   *
   * @param data_repo_mgr Reference to the data repository manager for storing task outputs
   * @param message_queue Reference to the message queue for task completion notifications
   */
  explicit downgrade_task_global_state(
    sirius::memory::sirius_memory_reservation_manager& reservation_manager,
    cucascade::shared_data_repository_manager& data_repo_mgr,
    task_completion_message_queue& message_queue)
    : _reservation_manager(reservation_manager),
      _data_repo_mgr(data_repo_mgr),
      _message_queue(message_queue)
  {
  }

  sirius::memory::sirius_memory_reservation_manager& _reservation_manager;
  cucascade::shared_data_repository_manager&
    _data_repo_mgr;  ///< Repository for storing and retrieving data batches
  task_completion_message_queue&
    _message_queue;  ///< Message queue to notify task_creator about task completion
};

/**
 * @brief Local state for a downgrade task
 *
 * This class holds the local state for a downgrade task, including the task ID,
 * the pipeline ID, and the data batch view.
 */
class downgrade_task_local_state : public itask_local_state {
 public:
  explicit downgrade_task_local_state(uint64_t task_id,
                                      uint64_t pipeline_id,
                                      std::shared_ptr<cucascade::data_batch> batch)
    : _task_id(task_id), _pipeline_id(pipeline_id), _batch(std::move(batch))
  {
  }
  uint64_t _task_id;
  uint64_t _pipeline_id;
  std::shared_ptr<cucascade::data_batch> _batch;
};

/**
 * @brief A task representing a unit of work in a memory downgrade operation.
 *
 * This class encapsulates the necessary information to execute a task within a memory
 * downgrade operation. Memory downgrading involves moving data from higher-tier (faster)
 * memory to lower-tier (slower) memory to free up space.
 */
class downgrade_task : public itask {
 public:
  /**
   * @brief Construct a new downgrade_task object
   *
   * @param local_state The local state specific to this task
   * @param global_state The global state shared across multiple tasks
   */
  downgrade_task(std::unique_ptr<downgrade_task_local_state> local_state,
                 std::shared_ptr<downgrade_task_global_state> global_state)
    : itask(std::move(local_state), std::move(global_state))
  {
  }

  /**
   * @brief Executes the memory downgrade operation for this task
   *
   * This method performs the actual downgrading of data from a higher memory tier
   * to a lower memory tier.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void execute(rmm::cuda_stream_view stream) override;

  /**
   * @brief Get the unique identifier for this task
   *
   * @return uint64_t The task ID
   */
  uint64_t get_task_id() const;

  /**
   * @brief Marks this task as completed and notifies dependent tasks
   *
   * This method informs the task_creator that the task has been completed, allowing
   * it to schedule any tasks that were dependent on this task's completion. This
   * method should be called after successfully pushing the task's output to the
   * Data Repository.
   */
  void mark_task_completion();
};

}  // namespace parallel
}  // namespace sirius
