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
    _bytes_to_materialize_input = get_estimated_bytes_to_materialize_input();
  }

  std::unique_ptr<op::operator_data> _input_data;  ///< Input data batches for the pipeline
  size_t _start_operator_index = 0;  ///< Operator index to resume from (0 = start of pipeline)

  /// Number of times this task has been retried due to OOM (0 = first attempt).
  uint32_t retry_count = 0;
  /// Task ID of the original (non-retried) task; only meaningful when retry_count > 0.
  std::optional<uint64_t> original_task_id = std::nullopt;

  // We want to cache the estimation basis, because its going to be calculated from the inputs, but
  // if the inputs change due to preparing for processing then we will loose this information and we
  // can't then provide it to the operator history
  mutable std::optional<std::size_t> _estimation_basis = std::nullopt;

  // The peak bytes observed to materialize the input data. We need to track this so we can subtract
  // it from the peak bytes observed to compute the operators' peak bytes.
  size_t _bytes_to_materialize_input = 0;

  /**
   * @brief Get a const pointer to the reservation (non-owning).
   *
   * @return const cucascade::memory::reservation* Pointer to the reservation, or nullptr
   */
  const cucascade::memory::reservation* get_reservation() const { return _reservation.get(); }

  [[nodiscard]] std::size_t get_task_consumption_basis() const override
  {
    if (_estimation_basis) { return *_estimation_basis; }
    _estimation_basis = _input_data ? _input_data->get_estimated_size_in_bytes() : 0;
    return *_estimation_basis;
  }

  [[nodiscard]] std::size_t get_estimated_bytes_to_materialize_input() const
  {
    std::size_t input_size = 0;
    auto* pipelineable_input =
      dynamic_cast<const op::pipelineable_operator_data*>(_input_data.get());
    if (pipelineable_input) {
      for (const auto& batch : pipelineable_input->get_data_batches()) {
        if (batch && batch->get_data() &&
            batch->get_data()->get_current_tier() != cucascade::memory::Tier::GPU) {
          input_size += batch->get_data()->get_uncompressed_data_size_in_bytes();
        }
      }
    }
    return input_size;
  }
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

  std::size_t get_estimated_reservation_size() const override;

  /// @brief Get the output consumer operators for this task.
  std::vector<op::sirius_physical_operator*> get_output_consumers() override;

  /**
   * @brief Mark this task as rescheduled due to OOM.
   *
   * When set, the destructor will NOT call mark_task_completed() on the pipeline,
   * since the rescheduled replacement task will handle that instead.
   */
  void mark_as_rescheduled() noexcept { _oom_rescheduled = true; }

  /**
   * @brief Check if this task was rescheduled due to OOM.
   */
  [[nodiscard]] bool is_rescheduled() const noexcept { return _oom_rescheduled; }

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
  uint64_t _task_id;
  std::vector<cucascade::shared_data_repository*> _data_repos;
  bool _oom_rescheduled                                             = false;
  cucascade::memory::reservation_aware_resource_adaptor* _allocator = nullptr;
};

}  // namespace pipeline
}  // namespace sirius
