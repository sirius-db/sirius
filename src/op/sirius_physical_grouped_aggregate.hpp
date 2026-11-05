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

#include <atomic>
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
    duckdb::vector<duckdb::unsafe_vector<duckdb::ProjectionIndex>> grouping_functions,
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
  /// Parallel to cudf_aggregates: the plan-time bound on |unscaled value| of a decimal or wide
  /// integer SUM input (planner::resolve_sum_input_bounds); nullopt when unproven.
  std::vector<std::optional<std::uint64_t>> cudf_aggregate_input_max_abs;
  /// Parallel to cudf_aggregates: true for the SUM of a wide integer SUM or AVG
  /// (AggregateSlot::is_wide_integer_sum), whose partial result may be DECIMAL128.
  std::vector<bool> cudf_aggregate_is_wide_integer_sum;

  /// Max groups in the previous batch for a widened BIGINT SUM to be summed as two INT64 halves
  /// instead of over DECIMAL128. From a group-count sweep on an RTX PRO 6000: the halves win up to
  /// 10K groups (contention on cuDF's 128-bit CAS), tie at 100K-300K, and lose from 1M on (a second
  /// atomic per row). 64K sits inside the tie range, well below the losses.
  static constexpr std::int64_t split_wide_sum_max_groups = 1 << 16;

  /// Groups in the most recent batch this operator aggregated; -1 until the first one. Batches
  /// that start before any count is known split: a wasted split costs a few percent at most,
  /// while a wasted DECIMAL128 sum with few groups costs up to 3x.
  std::atomic<std::int64_t> _last_batch_groups{-1};

  // Partial aggregate layout and finalization metadata
  std::vector<AggregateSlot> aggregate_slots;
  duckdb::vector<sirius::logical_type> local_types;
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

  [[nodiscard]] bool has_stddev_samp() const;

  //! Alternate carriers of the local output, for set_alternate_physical_types(). The partial of a
  //! wide integer SUM, or of an aggregate whose result is HUGEINT, is INT64 or DECIMAL128 with
  //! scale 0; its column's alternate is whichever of the two is not its native carrier. Empty when
  //! no slot has two carriers or `types` is not one column per cuDF aggregate after the group
  //! columns.
  [[nodiscard]] std::vector<std::optional<cudf::data_type>> wide_integer_sum_partial_types() const;

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
