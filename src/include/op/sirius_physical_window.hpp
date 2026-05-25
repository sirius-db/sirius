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

#include "duckdb/common/vector.hpp"
#include "duckdb/planner/expression.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "op/window/window_op_util.hpp"

#include <vector>

namespace sirius {
namespace op {

/// Phase 1 window-function operator: ROW_NUMBER / RANK / DENSE_RANK over a single PARTITION BY /
/// ORDER BY shared by all expressions of one DuckDB LogicalWindow.
///
/// Ranking is not decomposable, so (unlike HASH_GROUP_BY's local+merge) this operator consumes
/// already-partitioned data: the pipeline converter hash-partitions the input by the PARTITION BY
/// columns and feeds whole partitions here (same shape as MERGE_GROUP_BY consuming PARTITION, but
/// without a local-partial stage and without CONCAT). Per partition, execute() concatenates the
/// batches, stably sorts by (PARTITION BY, ORDER BY), runs a grouped rank scan, and appends the
/// resulting BIGINT rank columns. Output schema = child columns ++ one rank column per expression.
class sirius_physical_window : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::WINDOW;

 public:
  /// @param types        output schema = child output columns ++ one BIGINT per window expression
  /// @param window_exprs the LogicalWindow's BoundWindowExpression list (Phase 1 ranking only,
  ///                     sharing one PARTITION BY / ORDER BY; validated by create_plan)
  sirius_physical_window(duckdb::vector<sirius::logical_type> types,
                         duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> window_exprs,
                         std::size_t estimated_cardinality);

  // cuDF compute definitions (see window_op_util).
  std::vector<int> partition_idx;
  std::vector<int> order_idx;
  std::vector<cudf::order> order_dirs;
  std::vector<cudf::null_order> order_null;
  std::vector<window_rank_kind> ranks;

  std::size_t current_partition_index = 0;

  /// PARTITION BY column indices, consumed by sirius_physical_partition's WINDOW branch to choose
  /// the hash-partition keys. Returns std::vector<int> to match _partition_keys.
  const std::vector<int>& get_partition_key_indices() const { return partition_idx; }

  // Source interface
  bool is_source() const override { return true; }

  // Window sorts internally and emits in that order; outer ORDER BY (if any) re-orders downstream.
  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

  // Sink interface
  bool is_sink() const override { return true; }

  bool sink_order_dependent() const override { return false; }

  // partition-consumer: drain all batches of one partition per task (mirrors MERGE_GROUP_BY).
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;
};

}  // namespace op
}  // namespace sirius
