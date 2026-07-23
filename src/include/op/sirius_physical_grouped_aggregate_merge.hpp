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
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"

#include <memory>
#include <numeric>

namespace sirius {
namespace op {

class sirius_physical_grouped_aggregate_merge : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::MERGE_GROUP_BY;

 public:
  sirius_physical_grouped_aggregate_merge(
    sirius_physical_grouped_aggregate* grouped_aggregate,
    uint64_t hash_partition_bytes = config::DEFAULT_HASH_PARTITION_BYTES);

  sirius_physical_grouped_aggregate_merge(
    duckdb::vector<sirius::logical_type> types,
    std::vector<int> group_idx,
    std::vector<cudf::aggregation::Kind> cudf_aggregates,
    std::vector<int> cudf_aggregate_idx,
    std::vector<std::vector<int>> cudf_aggregate_struct_col_indices,
    std::vector<AggregateSlot> aggregate_slots,
    bool has_avg,
    bool has_count_distinct,
    std::size_t estimated_cardinality);

  sirius_physical_grouped_aggregate_merge(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups,
    std::size_t estimated_cardinality);

  sirius_physical_grouped_aggregate_merge(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups,
    duckdb::vector<duckdb::GroupingSet> grouping_sets,
    duckdb::vector<duckdb::unsafe_vector<std::size_t>> grouping_functions,
    std::size_t estimated_cardinality,
    duckdb::TupleDataValidityType group_validity,
    duckdb::TupleDataValidityType distinct_validity);

  //! The grouping sets
  duckdb::GroupedAggregateData grouped_aggregate_data;

  duckdb::vector<duckdb::GroupingSet> grouping_sets;
  //! The radix partitioned hash tables (one per grouping set)
  duckdb::vector<duckdb::HashAggregateGroupingData> groupings;
  duckdb::unique_ptr<duckdb::DistinctAggregateCollectionInfo> distinct_collection_info;
  //! A recreation of the input chunk, with nulls for everything that isn't a group
  duckdb::vector<sirius::logical_type> input_group_types;

  // Filters given to sink and friends
  duckdb::unsafe_vector<std::size_t> non_distinct_filter;
  duckdb::unsafe_vector<std::size_t> distinct_filter;

  sirius_physical_operator* child_op;
  sirius_physical_operator* get_child_op() const { return child_op; }

  // Grouped aggregatge definitions for cudf compute
  std::vector<int> group_idx;
  std::vector<cudf::aggregation::Kind> cudf_aggregates;
  std::vector<int> cudf_aggregate_idx;
  std::vector<std::vector<int>> cudf_aggregate_struct_col_indices;

  // AVG and COUNT DISTINCT decomposition metadata
  std::vector<AggregateSlot> aggregate_slots;
  bool has_avg            = false;
  bool has_count_distinct = false;

  std::size_t current_partition_index = 0;

 public:
  std::vector<int> get_output_grouping_indices() const
  {
    std::vector<int> indices(group_idx.size());
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }

  // Source interface
  bool is_source() const override { return true; }

  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

  // Sink interface
  bool is_sink() const override { return true; }

  bool sink_order_dependent() const override { return false; }

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Decide the partition count for the upstream PARTITION operator that feeds this merge
  partition_strategy get_partition_strategy(const partition_sizing_input& in) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;
  std::string params_to_string() const override;
};

}  // namespace op
}  // namespace sirius
