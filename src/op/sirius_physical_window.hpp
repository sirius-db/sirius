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
#include "op/window/cumulative.hpp"
#include "op/window/ranking.hpp"
#include "op/window/whole_partition.hpp"

namespace sirius::op {

/**
 * @brief Pipelineable adapter that evaluates window functions over complete hash buckets.
 *
 * A window evaluates aggregates over the whole partition, ranking functions over the window ORDER
 * BY, or running aggregates over the window ORDER BY. The child must publish exactly one complete
 * bucket per batch. PARTITION and CONCAT do the partition routing and apply the FULL barrier, not
 * this operator.
 *
 * A task evaluates one bucket as one cuDF table on one GPU. A bucket holds every row of each
 * partition it contains, so a single partition is never split. Limits on a bucket:
 * - Its row count must stay below the `cudf::size_type` limit (2^31 - 1). CONCAT rejects a bucket
 *   above it.
 * - Its input and results must fit in the GPU's memory. `execute` rejects a bucket whose known
 *   lower bound exceeds the memory space's maximum.
 *
 * A partition larger than either limit is not handled by this operator.
 */
class sirius_physical_window : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::WINDOW;

  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    // Hash buckets do not preserve the input's global sequence.
    return sirius::OrderPreservationType::NO_ORDER;
  }

  /// A window of aggregates over the whole partition.
  sirius_physical_window(duckdb::vector<logical_type> output_types,
                         duckdb::vector<logical_type> prepared_types,
                         std::size_t input_width,
                         std::vector<cudf::size_type> partition_columns,
                         std::vector<window_aggregate> aggregates,
                         std::size_t estimated_cardinality);

  /// A window of ranking functions over the window ORDER BY.
  sirius_physical_window(duckdb::vector<logical_type> output_types,
                         duckdb::vector<logical_type> prepared_types,
                         std::size_t input_width,
                         std::vector<cudf::size_type> partition_columns,
                         window_order order,
                         std::vector<ranking_function> ranking_functions,
                         std::size_t estimated_cardinality);

  /// A window of running aggregates over the window ORDER BY.
  sirius_physical_window(duckdb::vector<logical_type> output_types,
                         duckdb::vector<logical_type> prepared_types,
                         std::size_t input_width,
                         std::vector<cudf::size_type> partition_columns,
                         window_order order,
                         std::vector<window_aggregate> cumulative_aggregates,
                         std::size_t estimated_cardinality);

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(operator_data const& input,
                                         ::cuda::stream_ref stream) override;
  std::size_t no_history_peak_memory_estimate(input_stats const& stats) const override;

  /// The indices of the partition key columns in the prepared input.
  std::vector<cudf::size_type> const& partition_columns() const { return _partition_columns; }

 private:
  /// Cold-start estimate of the memory a task needs for a bucket.
  std::size_t workspace_bytes(std::size_t rows, std::size_t input_bytes) const;
  duckdb::vector<logical_type> _prepared_types;
  std::size_t _input_width;
  std::vector<cudf::size_type> _partition_columns;
  std::vector<window_aggregate> _aggregates;
  window_order _ranking_order;
  std::vector<ranking_function> _ranking_functions;
  std::vector<window_aggregate> _cumulative_aggregates;
};

}  // namespace sirius::op
