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

#include "op/sirius_physical_operator.hpp"
#include "parallel/task.hpp"
#include "pipeline/sirius_pipeline_itask_local_state.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>

#include <memory>
#include <vector>

namespace sirius {
namespace pipeline {

/**
 * @brief Interface for pipeline tasks that compute and publish data batches.
 *
 * This class extends itask to provide a common interface for pipeline tasks
 * that process data batches. It serves as a common base for both GPU pipeline
 * tasks and DuckDB scan tasks, separating the computation logic from the
 * output publishing logic.
 */
// WSM TODO: consider merging this with itask
class sirius_pipeline_itask : public parallel::itask {
 public:
  /**
   * @brief Destructor for proper cleanup of derived classes.
   */
  ~sirius_pipeline_itask() override = default;

  /**
   * @brief Compute and return the output data batches for this task.
   *
   * This method performs the actual computation work of the task and returns
   * the resulting data batches. The computation may involve reading input batches,
   * executing GPU operators, scanning database tables, etc.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return std::vector<std::shared_ptr<cucascade::data_batch>> The computed output
   *         data batches, which may be empty if no output is produced.
   */
  virtual std::unique_ptr<op::operator_data> compute_task(rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Publish the computed output batches to appropriate destinations.
   *
   * This method handles the publishing of output batches to data repositories,
   * notification of task creators, and any other post-computation activities.
   * It separates the concerns of computation from output management.
   *
   * @param output_batches The data batches to publish (typically the result of compute_task())
   */
  virtual void publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Get the estimated reservation memory size needed for this task.
   *
   * This method returns the estimated reservation size for this task.
   *
   * @return std::size_t The estimated reservation size
   */
  virtual std::size_t get_estimated_reservation_size() const = 0;

  /// @brief Get the output consumer operators for this task.
  virtual std::vector<op::sirius_physical_operator*> get_output_consumers() = 0;

  void execute(rmm::cuda_stream_view stream) override
  {
    auto output_batches = compute_task(stream);
    if (output_batches) { publish_output(*output_batches, stream); }
  }

 protected:
  /**
   * @brief Protected constructor for derived classes.
   *
   * @param local_state The local state specific to this task
   * @param global_state The global state shared across multiple tasks
   */
  sirius_pipeline_itask(std::unique_ptr<sirius_pipeline_itask_local_state> local_state,
                        std::shared_ptr<parallel::itask_global_state> global_state)
    : itask(std::move(local_state), std::move(global_state))
  {
  }
};

}  // namespace pipeline
}  // namespace sirius
