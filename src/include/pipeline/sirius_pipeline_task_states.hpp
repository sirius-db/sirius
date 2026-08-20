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

#include "exec/queue_priority.hpp"
#include "parallel/task.hpp"
#include "pipeline/pipeline_memory_history.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/memory/memory_reservation.hpp>

#include <memory>
#include <optional>
#include <utility>

namespace sirius {
namespace telemetry {
class telemetry_context;
}  // namespace telemetry

namespace pipeline {

/**
 * @brief Breakdown of a task's pre-execution memory reservation estimate.
 *
 * Computed by get_estimated_reservation_size_info() and cached on the local state via
 * set_reservation() so that execute() can access all components without re-computing.
 */
struct reservation_size_info {
  std::size_t input_basis = 0;  ///< Estimation basis (e.g. input data size)
  std::size_t bytes_to_materialize_input =
    0;  ///< Cost to materialize input into the task's target space (host/disk upgrades plus
        ///< cross-GPU clones); 0 for scans
  std::size_t peak_memory_estimate = 0;  ///< Predicted operator peak; 2*input_basis if no history
  std::size_t reservation_size     = 0;  ///< peak_memory_estimate + bytes_to_materialize_input
  bool had_history                 = false;  ///< True if estimate came from pipeline_memory_history
};

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
   * @param telemetry_context SiriusContext-wide telemetry context for task events
   */
  explicit sirius_pipeline_task_global_state(
    duckdb::shared_ptr<sirius_pipeline> pipeline,
    std::shared_ptr<const telemetry::telemetry_context> telemetry_context)
    : _pipeline(std::move(pipeline)), _telemetry_context(std::move(telemetry_context))
  {
  }

  [[nodiscard]] const sirius_pipeline* get_pipeline() const { return _pipeline.get(); }

  [[nodiscard]] sirius_pipeline* get_pipeline() { return _pipeline.get(); }

  [[nodiscard]] size_t get_pipeline_id() const
  {
    return _pipeline ? _pipeline->get_pipeline_id() : 0;
  }

  void set_pipeline(duckdb::shared_ptr<sirius_pipeline> pipeline)
  {
    _pipeline = std::move(pipeline);
  }

  [[nodiscard]] const telemetry::telemetry_context& get_telemetry_context() const noexcept
  {
    return *_telemetry_context;
  }

  /**
   * @brief Get the memory history for this pipeline's tasks.
   *
   * Used to record and query historical memory consumption patterns so that
   * future tasks can make better reservation estimates.
   *
   * The history is owned by the pipeline itself, not by this global state, so that the
   * data size estimator can reach a producer pipeline's ratio through `port::src_pipeline`
   * without going through the task_creator. There is exactly one global state per pipeline
   * (task_creator::prepare_for_query), so this delegation preserves the previous 1:1
   * lifetime. Falls back to a per-state history for the test paths that construct a global
   * state with a null pipeline.
   */
  pipeline_memory_history& get_memory_history()
  {
    return _pipeline ? _pipeline->get_memory_history() : _detached_memory_history;
  }
  const pipeline_memory_history& get_memory_history() const
  {
    return _pipeline ? _pipeline->get_memory_history() : _detached_memory_history;
  }

  /**
   * @brief Set the preferred GPU device ID for this pipeline's tasks.
   *
   * This is a pipeline-level default that can be overridden per-task
   * via gpu_pipeline_task_local_state::set_preferred_device_id().
   *
   * @param device_id The GPU device ID where data locality is highest
   */
  void set_preferred_device_id(int device_id) { _preferred_device_id = device_id; }

  /**
   * @brief Get the preferred GPU device ID for this pipeline's tasks.
   *
   * @return The preferred device ID, or std::nullopt if not set
   */
  [[nodiscard]] std::optional<int> get_preferred_device_id() const { return _preferred_device_id; }

  /**
   * @brief Set the scheduling priority shared by all tasks of this pipeline.
   *
   * Lower values are scheduled first by the pipeline-level priority queue. Assigned once per
   * query by task_creator::prepare_for_query() based on the pipeline's position in the plan.
   */
  void set_priority(exec::queue_priority priority) { _priority = priority; }

  /**
   * @brief Get the scheduling priority for this pipeline's tasks (default 0).
   */
  [[nodiscard]] exec::queue_priority get_priority() const { return _priority; }

 private:
  duckdb::shared_ptr<sirius_pipeline> _pipeline;  ///< Shared pointer to the GPU pipeline to execute
  /// Stand-in history used only when @c _pipeline is null (tests that construct a global
  /// state without a pipeline). The real history lives on sirius_pipeline.
  pipeline_memory_history _detached_memory_history;
  std::optional<int> _preferred_device_id;  ///< Pipeline-level preferred GPU device
  exec::queue_priority _priority{0};        ///< Pipeline-level scheduling priority
  std::shared_ptr<const telemetry::telemetry_context>
    _telemetry_context;  ///< SiriusContext telemetry
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
    if (_reservation) {
      _reservation_bytes = _reservation->size();
    } else {
      _reservation_bytes = 0;
    }
  }

  /**
   * @brief Set a memory reservation and cache the pre-computed size breakdown.
   *
   * Combines set_reservation() with storing the reservation_size_info produced by
   * get_estimated_reservation_size_info(), so execute() can read all estimation
   * components from _reservation_size_info without re-computing them.
   *
   * @param res  The memory reservation to set (ownership transferred to the task)
   * @param info The pre-computed reservation size breakdown
   */
  void set_reservation(std::unique_ptr<cucascade::memory::reservation> res,
                       reservation_size_info info)
  {
    set_reservation(std::move(res));
    _reservation_size_info = info;
  }

  [[nodiscard]] std::size_t get_reservation_bytes() const { return _reservation_bytes; }

  [[nodiscard]] const std::optional<reservation_size_info>& get_reservation_size_info()
    const noexcept
  {
    return _reservation_size_info;
  }

  /**
   * @brief Non-owning accessor for the held reservation.
   *
   * @return Pointer to the reservation, or nullptr if none is held.
   */
  cucascade::memory::reservation* reservation() noexcept { return _reservation.get(); }

  /**
   * @brief Get the basis for estimating task memory consumption.
   *
   * This method allows for different task types to provide their own logic for providing something
   * as a starting point for memory reservation estimation.
   *
   * @return The value to use as the basis for memory consumption estimation (e.g., input data size,
   * number of rows, etc.)
   */
  [[nodiscard]] virtual std::size_t get_task_consumption_basis() const = 0;

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
  std::size_t _reservation_bytes = 0;
  std::optional<reservation_size_info> _reservation_size_info;  ///< Cached estimation breakdown
};

}  // namespace pipeline
}  // namespace sirius
