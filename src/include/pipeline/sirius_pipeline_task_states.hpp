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

#include "parallel/task.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/memory/memory_reservation.hpp>

#include <memory>

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
class sirius_pipeline_task_global_state : public sirius::parallel::itask_global_state {
 public:
  /**
   * @brief Construct a new sirius_pipeline_task_global_state object
   *
   * @param pipeline Shared pointer to the GPU pipeline to execute
   */
  explicit sirius_pipeline_task_global_state(duckdb::shared_ptr<sirius_pipeline> pipeline)
    : _pipeline(std::move(pipeline))
  {
  }

  [[nodiscard]] const sirius_pipeline* get_pipeline() const { return _pipeline.get(); }

  [[nodiscard]] sirius_pipeline* get_pipeline() { return _pipeline.get(); }

  [[nodiscard]] size_t get_pipeline_id() const { return _pipeline->get_pipeline_id(); }

 private:
  duckdb::shared_ptr<sirius_pipeline> _pipeline;  ///< Shared pointer to the GPU pipeline to execute
};

/**
 * @brief Interface for pipeline task local states that manage memory reservations.
 *
 * This class extends itask_local_state to provide memory reservation management
 * capabilities for pipeline tasks. It serves as a common base for both GPU pipeline
 * tasks and DuckDB scan tasks that need to manage memory reservations.
 */
// WSM TODO: consider merging this with itask_local_state
class sirius_pipeline_task_local_state : public parallel::itask_local_state {
 public:
  /**
   * @brief Destructor for proper cleanup of derived classes.
   */
  ~sirius_pipeline_task_local_state() override = default;

  /**
   * @brief Release and return the memory reservation held by this task.
   *
   * This method transfers ownership of the reservation to the caller.
   * After calling this method, the task no longer holds a reservation.
   *
   * @return std::unique_ptr<cucascade::memory::reservation> The released reservation,
   *         or nullptr if no reservation was held.
   */
  std::unique_ptr<cucascade::memory::reservation> release_reservation()
  {
    return std::move(_reservation);
  }

  /**
   * @brief Set a memory reservation for this task.
   *
   * This method transfers ownership of the provided reservation to the task.
   * Any previously held reservation will be released.
   *
   * @param res The memory reservation to set (ownership transferred to the task)
   */
  void set_reservation(std::unique_ptr<cucascade::memory::reservation> res)
  {
    _reservation = std::move(res);
  }

 protected:
  /**
   * @brief Protected default constructor.
   *
   * This constructor is protected to ensure the class can only be instantiated
   * through derived classes.
   */
  sirius_pipeline_task_local_state() = default;

  std::unique_ptr<cucascade::memory::reservation>
    _reservation;  ///< Memory reservation for GPU resources
};

}  // namespace pipeline
}  // namespace sirius
