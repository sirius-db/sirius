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
#include "parallel/task_executor.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_itask.hpp"
#include "pipeline/sirius_pipeline_itask_local_state.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/memory_reservation.hpp>

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
  explicit gpu_pipeline_task_global_state(duckdb::shared_ptr<sirius_pipeline> pipeline)
    : _pipeline(std::move(pipeline))
  {
  }
  duckdb::shared_ptr<sirius_pipeline> _pipeline;  ///< Shared pointer to the GPU pipeline to execute
};

/**
 * @brief Local state specific to an individual GPU pipeline task instance.
 *
 * This class encapsulates the state and data that is unique to a single task
 * execution. It holds the task and pipeline identifiers, the GPU pipeline to
 * execute, and the data batch views that serve as input to the pipeline.
 */
class gpu_pipeline_task_local_state : public sirius_pipeline_itask_local_state {
 public:
  /**
   * @brief Construct a new gpu_pipeline_task_local_state object
   *
   * @param batch_views Vector of data batches serving as input to the pipeline
   * @param res Memory reservation for GPU resources
   */
  explicit gpu_pipeline_task_local_state(
    std::vector<std::shared_ptr<cucascade::data_batch>> batches)
    : _batches(std::move(batches))
  {
  }

  std::vector<std::shared_ptr<cucascade::data_batch>>
    _batches;  ///< Input data batches for the pipeline

  /**
   * @brief Get a const pointer to the reservation (non-owning).
   *
   * @return const cucascade::memory::reservation* Pointer to the reservation, or nullptr
   */
  // WSM TODO: remove this method?
  const cucascade::memory::reservation* get_reservation() const { return _reservation.get(); }
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
class gpu_pipeline_task : public sirius_pipeline_itask {
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
                    std::vector<cucascade::shared_data_repository*> data_repos,
                    std::unique_ptr<sirius_pipeline_itask_local_state> local_state,
                    std::shared_ptr<sirius::parallel::itask_global_state> global_state);

  ~gpu_pipeline_task() override;

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
   * @return const duckdb::sirius_pipeline* Pointer to the GPU pipeline
   */
  const sirius_pipeline* get_pipeline() const;

  /**
   * @brief Compute and return the output data batches for this task.
   *
   * Executes the GPU pipeline on the input batches and returns the computed results.
   *
   * @return std::vector<std::shared_ptr<cucascade::data_batch>> The computed output batches
   */
  std::vector<std::shared_ptr<cucascade::data_batch>> compute_task() override;

  /**
   * @brief Publish the computed output batches to data repositories.
   *
   * Pushes the output batches to the configured data repositories.
   *
   * @param output_batches The data batches to publish
   */
  void publish_output(std::vector<std::shared_ptr<cucascade::data_batch>> output_batches) override;

  /**
   * @brief Get the input size for this task
   *
   * @return std::size_t The input size
   */
  std::size_t get_input_size() const;

  std::size_t get_estimated_reservation_size() const override;

  /// @brief Get the output consumer operators for this task.
  std::vector<op::sirius_physical_operator*> get_output_consumers() override;

 private:
  uint64_t _task_id;
  std::vector<cucascade::shared_data_repository*> _data_repos;
};

}  // namespace pipeline
}  // namespace sirius
