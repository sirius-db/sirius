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
#include "creator/task_completion.hpp"
<<<<<<< HEAD
=======
#include "data/data_repository.hpp"
>>>>>>> 842a98a2 (Using shared_ptr<data_batch> instead of data_batch_view in preparation for cucascade)
#include "gpu_pipeline.hpp"
#include "parallel/task_executor.hpp"

#include <data/data_batch.hpp>
#include <data/data_repository.hpp>
#include <data/data_repository_manager.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace sirius {
namespace pipeline {

/**
 * @brief Global state shared across all GPU pipeline tasks in an execution context.
 *
 * This class maintains resources and state that are shared among multiple tasks
 * within the same execution context. It provides access to the data repository
 * for retrieving input data and a message queue for notifying the TaskCreator
 * about task completion events.
 */
class gpu_pipeline_task_global_state : public sirius::parallel::itask_global_state {
 public:
  /**
   * @brief Construct a new gpu_pipeline_task_global_state object
   *
   * @param pipeline Shared pointer to the GPU pipeline to execute
   */
<<<<<<< HEAD
  explicit gpu_pipeline_task_global_state(uint64_t pipeline_id,
                                          duckdb::shared_ptr<duckdb::GPUPipeline> pipeline,
                                          cucascade::shared_data_repository_manager& data_repo_mgr,
                                          task_completion_message_queue& message_queue)
    : _pipeline_id(pipeline_id),
      _pipeline(std::move(pipeline)),
      _data_repo_mgr(data_repo_mgr),
      _message_queue(message_queue)
  {
  }

  cucascade::shared_data_repository_manager&
    _data_repo_mgr;  ///< Reference to the data repository manager
  task_completion_message_queue&
    _message_queue;  ///< Message queue to notify TaskCreator about task completion
=======
  explicit gpu_pipeline_task_global_state(duckdb::shared_ptr<duckdb::GPUPipeline> pipeline)
    : _pipeline(std::move(pipeline))
  {
  }
>>>>>>> 842a98a2 (Using shared_ptr<data_batch> instead of data_batch_view in preparation for cucascade)
  duckdb::shared_ptr<duckdb::GPUPipeline>
    _pipeline;  ///< Shared pointer to the GPU pipeline to execute
};

/**
 * @brief Local state specific to an individual GPU pipeline task instance.
 *
 * This class encapsulates the state and data that is unique to a single task
 * execution. It holds the task and pipeline identifiers, the GPU pipeline to
 * execute, and the data batch views that serve as input to the pipeline.
 */
class gpu_pipeline_task_local_state : public sirius::parallel::itask_local_state {
 public:
  /**
   * @brief Construct a new gpu_pipeline_task_local_state object
   *
<<<<<<< HEAD
   * @param task_id Unique identifier for this task
   * @param batches Vector of data batches serving as input to the pipeline
   */
  explicit gpu_pipeline_task_local_state(
    uint64_t task_id,
    std::vector<std::shared_ptr<cucascade::data_batch>> batches,
    std::unique_ptr<cucascade::memory::reservation> res = nullptr)
    : _task_id(task_id), _batches(std::move(batches)), _reservation(std::move(res))
  {
  }

  uint64_t _task_id;  ///< Unique identifier for this task
  std::vector<std::shared_ptr<cucascade::data_batch>>
    _batches;  ///< Input data batches for the pipeline
=======
   * @param batch_views Vector of data batches serving as input to the pipeline
   * @param res Memory reservation for GPU resources
   */
  explicit gpu_pipeline_task_local_state(
    std::vector<std::shared_ptr<cucascade::data_batch>> batch_views,
    std::unique_ptr<cucascade::memory::reservation> res = nullptr)
    : _batch_views(std::move(batch_views)), _reservation(std::move(res))
  {
  }

  std::vector<std::shared_ptr<cucascade::data_batch>>
    _batch_views;  ///< Input data batches for the pipeline
>>>>>>> 842a98a2 (Using shared_ptr<data_batch> instead of data_batch_view in preparation for cucascade)

  void set_reservation(std::unique_ptr<cucascade::memory::reservation> res)
  {
    _reservation = std::move(res);
  }

 private:
  std::unique_ptr<cucascade::memory::reservation>
    _reservation;  ///< Memory reservation for GPU resources
  // TODO: for now, reservation is passed as a local state, will be null when the task is first
  // created, and will be set when reservation is made
};

/**
 * @brief A task representing a unit of work in a GPU pipeline.
 *
 * This class encapsulates the necessary information to execute a task within a pipeline on the GPU.
 * These task will be created by the TaskCreator and be scheduled for execution on the
 * gpu_pipeline_executor.
 *
 * Note that this class will be further derived to represent specific types of tasks such as build,
 * aggregation, etc..
 */
class gpu_pipeline_task : public sirius::parallel::itask {
 public:
  /**
   * @brief Construct a new gpu_pipeline_task object
   *
   * @param task_id The unique identifier for this task
   * @param data_repos The data repositories to push the output of this task to
   * @param local_state The local state specific to this task
   * @param global_state The global state shared across multiple tasks
   */
  gpu_pipeline_task(uint64_t task_id,
                    std::vector<cucascade::idata_repository*> data_repos,
                    std::unique_ptr<sirius::parallel::itask_local_state> local_state,
                    std::shared_ptr<sirius::parallel::itask_global_state> global_state);

  /**
   * @brief Method to actually execute the task
   */
  void execute() override;

  /**
   * @brief Get the unique identifier for this task
   *
   * @return uint64_t The task ID
   */
  uint64_t get_task_id() const;

  /**
   * @brief Get the GPU pipeline associated with this task
   *
   * @return const duckdb::GPUPipeline* Pointer to the GPU pipeline
   */
  const duckdb::GPUPipeline* get_pipeline() const;

  /**
   * @brief Method to mark that this task is completed
   *
   * This method informs that TaskCreator that the task is completed so that it can start scheduling
   * tasks that were dependent on this task. This method should be called after pushing the output
   * of this task to the Data Repository.
   */
  void mark_task_completion();

<<<<<<< HEAD
  /**
   * @brief Method to push the output of this task to the Data Repository
   *
   * @param batch The data batch to push
   * @param pipeline_id The id of the pipeline that produced this data batch
   */
  void push_data_batch(std::shared_ptr<cucascade::data_batch> batch, uint64_t pipeline_id);
=======
 private:
  uint64_t _task_id;
<<<<<<< HEAD
  cucascade::idata_repository& _data_repo;
>>>>>>> 842a98a2 (Using shared_ptr<data_batch> instead of data_batch_view in preparation for cucascade)
=======
  std::vector<cucascade::idata_repository*> _data_repos;
>>>>>>> e15ed165 (Create pipeline task)
};

}  // namespace pipeline
}  // namespace sirius
