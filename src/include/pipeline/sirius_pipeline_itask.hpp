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
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "telemetry/runtime_fsm_handle.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>

#include <cstdint>
#include <memory>
#include <string>
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
  ~sirius_pipeline_itask() noexcept override;

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
   * @brief Compute the full breakdown of the pre-execution memory reservation estimate.
   *
   * Derived classes implement this to fill all fields of reservation_size_info.
   * The caller (executor) uses reservation_size_info::reservation_size to acquire a
   * reservation, then passes the struct to set_reservation() so execute() can read
   * the components without re-computing them.
   *
   * @param target_space The memory space the task will execute in, so inputs residing outside it
   *                     (host/disk tiers, or GPU data on a different device that prepare will
   *                     clone) are counted in bytes_to_materialize_input. nullptr means no single
   *                     target space is known; only non-GPU-tier inputs are counted.
   * @return reservation_size_info with input_basis, bytes_to_materialize_input,
   *         peak_memory_estimate, reservation_size, and had_history populated.
   */
  [[nodiscard]] virtual pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space* target_space) const = 0;

  /// @brief Get the output consumer operators for this task.
  virtual std::vector<op::sirius_physical_operator*> get_output_consumers() = 0;

  void execute(rmm::cuda_stream_view stream) override
  {
    auto output_batches = compute_task(stream);
    if (output_batches) { publish_output(*output_batches, stream); }
  }

  // Tasks that can exist without an attached pipeline should override this.
  [[nodiscard]] size_t get_pipeline_id() const
  {
    return _global_state->cast<sirius_pipeline_task_global_state>().get_pipeline_id();
  }

  struct telemetry_queued_data {
    quent::Uuid queue_resource_id;
    uint64_t queue_capacity_entries;
  };

  struct telemetry_routing_data {
    int64_t preferred_device_id;
    quent::Uuid manager_thread_resource_id;
  };

  struct telemetry_reserving_data {
    uint64_t requested_bytes;
    uint64_t input_basis;
    uint64_t peak_estimate;
    uint64_t bytes_to_materialize;
    quent::Uuid manager_thread_resource_id;
  };

  struct telemetry_downgrading_data {
    uint64_t shortfall_bytes;
    uint64_t partial_bytes;
    quent::Uuid manager_thread_resource_id;
  };

  struct telemetry_preparing_data {
    std::string origin_tier;
    std::string target_tier;
    uint64_t input_bytes;
    quent::Uuid executor_thread_resource_id;
    quent::Uuid reservation_resource_id;
    uint64_t reservation_capacity_bytes;
  };

  struct telemetry_computing_data {
    uint32_t current_operator_id;
    uint64_t input_bytes;
    uint64_t peak_allocated_bytes;
    quent::Uuid executor_thread_resource_id;
    quent::Uuid reservation_resource_id;
    uint64_t reservation_capacity_bytes;
  };

  [[nodiscard]] quent::Uuid telemetry_uuid() const;
  void telemetry_queued(telemetry_queued_data data);
  void telemetry_routing(telemetry_routing_data data);
  void telemetry_reserving(telemetry_reserving_data data);
  void telemetry_downgrading(telemetry_downgrading_data data);
  void telemetry_preparing(telemetry_preparing_data data);
  void telemetry_computing(telemetry_computing_data data);
  void finalize_telemetry(bool success);

 protected:
  /**
   * @brief Protected constructor for derived classes.
   *
   * @param task_id The unique identifier for this task
   * @param local_state The local state specific to this task
   * @param global_state The global state shared across multiple tasks
   */
  sirius_pipeline_itask(uint64_t task_id,
                        std::unique_ptr<sirius_pipeline_task_local_state> local_state,
                        std::shared_ptr<sirius_pipeline_task_global_state> global_state);

 private:
  using task_state = telemetry::runtime_fsm_handle<quent::Task,
                                                   quent::task_state::Created,
                                                   quent::task_state::Queued,
                                                   quent::task_state::Routing,
                                                   quent::task_state::Reserving,
                                                   quent::task_state::Downgrading,
                                                   quent::task_state::Preparing,
                                                   quent::task_state::Computing,
                                                   quent::task_state::Finalizing>;

  task_state _telemetry_task_state;
  bool _telemetry_finalized{false};
};

}  // namespace pipeline
}  // namespace sirius
