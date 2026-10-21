/*
 * Copyright 2026, Sirius Contributors.
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
#include "op/window/whole_partition.hpp"

namespace sirius::op {

// Pipelineable adapter. Its child must publish exactly one complete bucket per batch.
// Partition routing and the FULL barrier belong to PARTITION/CONCAT, not this evaluator.
class sirius_physical_window : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::WINDOW;

  sirius_physical_window(duckdb::vector<logical_type> output_types,
                         duckdb::vector<logical_type> prepared_types,
                         std::size_t input_width,
                         std::vector<cudf::size_type> partition_columns,
                         std::vector<whole_partition_aggregate> aggregates,
                         std::size_t estimated_cardinality);

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(operator_data const& input,
                                         ::cuda::stream_ref stream) override;
  std::size_t no_history_peak_memory_estimate(input_stats const& stats) const override;

  std::vector<cudf::size_type> const& partition_columns() const { return _partition_columns; }

 private:
  std::size_t workspace_bytes(std::size_t rows, std::size_t input_bytes) const;
  duckdb::vector<logical_type> _prepared_types;
  std::size_t _input_width;
  std::vector<cudf::size_type> _partition_columns;
  std::vector<whole_partition_aggregate> _aggregates;
};

}  // namespace sirius::op
