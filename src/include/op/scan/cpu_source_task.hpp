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

#include <cudf/utilities/default_stream.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/main/client_context.hpp>
#include <op/sirius_physical_cpu_source.hpp>
#include <parallel/task.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <pipeline/sirius_pipeline_itask.hpp>
#include <pipeline/sirius_pipeline_task_states.hpp>
#include <pipeline/task_scheduler.hpp>
#include <sirius_context.hpp>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// CPU Source Task Global State
//===----------------------------------------------------------------------===//
class cpu_source_task_global_state : public pipeline::sirius_pipeline_task_global_state {
 public:
  cpu_source_task_global_state(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
                               sirius_physical_cpu_source* source_op);

  sirius_physical_cpu_source& get_source_op() { return _op; }

  std::vector<sirius_physical_operator*> get_output_consumers() const noexcept
  {
    std::vector<sirius_physical_operator*> output_consumers;
    for (const auto& next_port : _op.get_next_ports_after_sink()) {
      output_consumers.push_back(next_port.next_operator);
    }
    return output_consumers;
  }

 private:
  sirius_physical_cpu_source& _op;
};

//===----------------------------------------------------------------------===//
// CPU Source Task Local State
//===----------------------------------------------------------------------===//
class cpu_source_task_local_state : public pipeline::sirius_pipeline_task_local_state {
 public:
  cpu_source_task_local_state() = default;

  // Required override of the base class's pure virtual. The consumption
  // basis feeds pipeline_memory_history (via estimate_peak_memory /
  // record_on_failure / record) so the estimator can learn a task's memory
  // footprint over time. cpu_source_task never exercises that code path: it
  // does not call get_estimated_reservation_size(), does not record memory
  // history, and is only scheduled through the CPU_SOURCE branch in
  // task_creator/duckdb_scan_executor (neither reads this value). Return 0
  // to satisfy the interface; the field is otherwise unused for this task.
  [[nodiscard]] std::size_t get_task_consumption_basis() const override { return 0; }
};

//===----------------------------------------------------------------------===//
// CPU Source Task
//===----------------------------------------------------------------------===//

/// A scan-side task that reads pre-computed DuckDB data (ColumnDataCollection,
/// empty result, or dummy scan) and publishes it as data_batch objects into
/// the pipeline's data repository.
class cpu_source_task : public pipeline::sirius_pipeline_itask {
 public:
  cpu_source_task(uint64_t task_id,
                  cucascade::shared_data_repository* data_repo,
                  std::unique_ptr<cpu_source_task_local_state> local_state,
                  std::shared_ptr<cpu_source_task_global_state> global_state,
                  duckdb::ClientContext& client_ctx);

  ~cpu_source_task() override;

  std::unique_ptr<op::operator_data> compute_task(rmm::cuda_stream_view stream) override;

  void publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t get_estimated_reservation_size() const override { return 0; }

  std::vector<op::sirius_physical_operator*> get_output_consumers() override;

 private:
  cucascade::shared_data_repository* _data_repo;
  duckdb::ClientContext& _client_ctx;
};

}  // namespace sirius::op::scan
