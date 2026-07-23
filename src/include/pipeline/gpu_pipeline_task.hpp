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
#include "pipeline/sirius_pipeline_task_states.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace sirius {
namespace op {
class operator_data;
}

namespace pipeline {

/**
 * @brief Global state shared across all GPU pipeline tasks in an execution context.
 *
 * This is an alias to sirius_pipeline_task_global_state for backward compatibility
 * and semantic clarity in GPU pipeline contexts.
 */
using gpu_pipeline_task_global_state = sirius_pipeline_task_global_state;

/**
 * @brief Local state specific to an individual GPU pipeline task instance.
 *
 * This class encapsulates the state and data that is unique to a single task
 * execution. It holds the task and pipeline identifiers, the GPU pipeline to
 * execute, and the data batch views that serve as input to the pipeline.
 */
class gpu_pipeline_task_local_state : public sirius_pipeline_task_local_state {
 public:
  /**
   * @brief Construct a new gpu_pipeline_task_local_state object
   *
   * @param batch_views Vector of data batches serving as input to the pipeline
   * @param res Memory reservation for GPU resources
   */
  explicit gpu_pipeline_task_local_state(std::unique_ptr<op::operator_data> input_data,
                                         size_t start_operator_index = 0)
    : _input_data(std::move(input_data)), _start_operator_index(start_operator_index)
  {
  }

  std::unique_ptr<op::operator_data> _input_data;  ///< Input data batches for the pipeline
  size_t _start_operator_index = 0;  ///< Operator index to resume from (0 = start of pipeline)

  /**
   * @brief Set the preferred GPU device ID for this task based on data locality.
   *
   * @param device_id The GPU device ID where the majority of input data resides
   */
  void set_preferred_device_id(int device_id) { _preferred_device_id = device_id; }

  /**
   * @brief Get the preferred GPU device ID for this task.
   *
   * @return The preferred device ID, or std::nullopt if not set
   */
  [[nodiscard]] std::optional<int> get_preferred_device_id() const { return _preferred_device_id; }

  /// Number of times this task has been retried due to OOM (0 = first attempt).
  uint32_t retry_count = 0;
  /// Task ID of the original (non-retried) task; only meaningful when retry_count > 0.
  std::optional<uint64_t> original_task_id = std::nullopt;

  [[nodiscard]] std::size_t get_task_consumption_basis() const override
  {
    if (_reservation_size_info) { return _reservation_size_info->input_basis; }
    // Fallback for code paths that call this before get_estimated_reservation_size_info()
    // (e.g. tests that bypass the normal executor flow).
    return _input_data ? _input_data->get_estimated_size_in_bytes() : 0;
  }

  /**
   * @brief Estimate the bytes prepare_for_processing will allocate in the target space.
   *
   * Counts inputs that are not GPU-resident (host/disk upgrades) and, when @p target_space is
   * given, GPU-resident inputs living in a different memory space — those are cloned into the
   * target space by lock_or_prepare_batch, so their bytes are part of the task's footprint.
   * Resident scan inputs cached in HOST are also counted, since they require an upload before
   * execution.
   */
  [[nodiscard]] std::size_t get_estimated_bytes_to_materialize_input(
    const cucascade::memory::memory_space* target_space) const;

 private:
  std::optional<int> _preferred_device_id;  ///< Preferred GPU device based on data locality
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
                    std::unique_ptr<sirius_pipeline_task_local_state> local_state,
                    std::shared_ptr<sirius_pipeline_task_global_state> global_state);

  ~gpu_pipeline_task() override;

  /**
   * @brief Method to actually execute the task
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void execute(rmm::cuda_stream_view stream) override;

  /**
   * @brief Get the preferred GPU device ID for this task.
   *
   * Checks local_state first (per-task override), then global_state (pipeline default).
   *
   * @return The preferred device ID, or std::nullopt if not set at either level
   */
  [[nodiscard]] std::optional<int> get_preferred_device_id() const
  {
    if (auto* ls = dynamic_cast<const gpu_pipeline_task_local_state*>(_local_state.get())) {
      if (ls->get_preferred_device_id().has_value()) { return ls->get_preferred_device_id(); }
    }
    if (auto gs = std::dynamic_pointer_cast<const gpu_pipeline_task_global_state>(_global_state)) {
      return gs->get_preferred_device_id();
    }
    return std::nullopt;
  }

  /**
   * @brief Get the scheduling priority for this task.
   *
   * Priority is a pipeline-level property carried on the shared global state, so every task of a
   * pipeline reports the same value. Lower priority values are dispatched first by the pipeline-
   * level priority queue (priority ascends with execution order). Defaults to 0 when no global
   * state is attached.
   *
   * @return The scheduling priority for this task.
   */
  [[nodiscard]] exec::queue_priority get_priority() const
  {
    if (auto gs = std::dynamic_pointer_cast<const gpu_pipeline_task_global_state>(_global_state)) {
      return gs->get_priority();
    }
    return 0;
  }

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
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return std::vector<std::shared_ptr<cucascade::data_batch>> The computed output batches
   */
  std::unique_ptr<op::operator_data> compute_task(rmm::cuda_stream_view stream) override;

  /**
   * @brief Publish the computed output batches to data repositories.
   *
   * Pushes the output batches to the configured data repositories.
   *
   * @param output_batches The data batches to publish
   */
  void publish_output(op::operator_data& output_batches, rmm::cuda_stream_view stream) override;

  /**
   * @brief Get the input size for this task
   *
   * @return std::size_t The input size
   */
  std::size_t get_input_size() const;

  [[nodiscard]] pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* target_space) const override;

  /// @brief Get the output consumer operators for this task.
  std::vector<op::sirius_physical_operator*> get_output_consumers() override;

  /**
   * @brief Get the data repositories for output publishing.
   *
   * Used by the executor to create a rescheduled task with the same output destinations.
   */
  [[nodiscard]] const std::vector<cucascade::shared_data_repository*>& get_data_repos()
    const noexcept
  {
    return _data_repos;
  }

  /**
   * @brief Get the shared global state.
   *
   * Used by the executor to create a rescheduled task sharing the same pipeline context.
   */
  [[nodiscard]] std::shared_ptr<sirius_pipeline_task_global_state> get_shared_global_state() const
  {
    return std::dynamic_pointer_cast<sirius_pipeline_task_global_state>(_global_state);
  }

  /**
   * @brief Create a rescheduled task after an OOM event.
   *
   * Derived classes can override this to ensure the rescheduled task has the correct
   * dynamic type and any additional state needed for re-execution.
   *
   * @param task_id The unique identifier for the new task
   * @param local_state The local state with intermediate data and resume index
   * @return A new task ready to be scheduled for execution
   */
  virtual std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id, std::unique_ptr<sirius_pipeline_task_local_state> local_state);

 private:
  std::vector<cucascade::shared_data_repository*> _data_repos;
  cucascade::memory::reservation_aware_resource_adaptor* _allocator = nullptr;
  /// Non-owning subscription ledger: the input data_batches this task subscribed to in its
  /// constructor so that the downgrade_executor can know that the data_baches are in a task.
  /// weak_ptr so that memory can be released as soon as the last owner drops.
  /// This is used in the destructor to unsubscribe.
  std::vector<std::weak_ptr<cucascade::data_batch>> _subscribed_batches;
};

}  // namespace pipeline
}  // namespace sirius
