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

#include "cudf/aggregation.hpp"
#include "cudf/types.hpp"
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/radix_partitioned_hashtable.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/storage/data_table.hpp"
#include "expression/ast/node.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <vector>

namespace sirius {
namespace op {

class sirius_physical_grouped_aggregate : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::HASH_GROUP_BY;

 public:
  sirius_physical_grouped_aggregate(duckdb::vector<sirius::logical_type> types,
                                    duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
                                    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups,
                                    std::size_t estimated_cardinality);

  sirius_physical_grouped_aggregate(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups,
    duckdb::vector<duckdb::GroupingSet> grouping_sets,
    duckdb::vector<duckdb::unsafe_vector<std::size_t>> grouping_functions,
    std::size_t estimated_cardinality,
    duckdb::TupleDataValidityType group_validity,
    duckdb::TupleDataValidityType distinct_validity,
    std::vector<std::optional<std::uint64_t>> aggregate_input_max_abs = {});

  duckdb::vector<duckdb::GroupingSet> grouping_sets;

  /// The grouping sets as positions in `group_idx`, and the arguments of each GROUPING()
  /// function. Used when the aggregate computes several grouping sets or GROUPING().
  std::vector<std::set<std::size_t>> grouping_set_keys;
  std::vector<std::vector<std::size_t>> grouping_functions;

  // Grouped aggregatge definitions for cudf compute
  std::vector<int> group_idx;
  std::vector<cudf::aggregation::Kind> cudf_aggregates;
  std::vector<int> cudf_aggregate_idx;
  std::vector<std::vector<int>> cudf_aggregate_struct_col_indices;
  /// Parallel to cudf_aggregates: the plan-time bound on |unscaled value| of a decimal SUM input
  /// (planner::resolve_decimal_sum_input_bounds); nullopt when unproven.
  std::vector<std::optional<std::uint64_t>> cudf_aggregate_input_max_abs;

  // AVG decomposition metadata
  std::vector<AggregateSlot> aggregate_slots;
  bool has_avg            = false;
  bool has_count_distinct = false;

 public:
  /// Whether the aggregate computes several grouping sets or GROUPING() functions. The local
  /// output then has a set id column and one column per GROUPING() function after the keys.
  [[nodiscard]] bool has_grouping_sets() const noexcept
  {
    return grouping_sets.size() > 1 || !grouping_functions.empty();
  }

  /// Number of leading output columns that the merge groups by.
  [[nodiscard]] std::size_t num_output_group_columns() const noexcept
  {
    return group_idx.size() + (has_grouping_sets() ? 1 + grouping_functions.size() : 0);
  }

  std::vector<int> get_output_grouping_indices() const
  {
    std::vector<int> indices(num_output_group_columns());
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }

  //! Runtime schema of the local COUNT(DISTINCT) accumulator. The local aggregate and PARTITION
  //! carry LIST sets; MERGE_GROUP_BY later converts those sets to the declared BIGINT count.
  [[nodiscard]] duckdb::vector<sirius::logical_type> get_count_distinct_local_output_types() const;

  // Source interface
  bool is_source() const override { return true; }

  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

  // Sink interface
  bool is_sink() const override { return true; }

  bool sink_order_dependent() const override { return false; }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;
};

}  // namespace op
}  // namespace sirius
