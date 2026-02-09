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
#include "op/sirius_physical_operator.hpp"

namespace sirius {
namespace op {

class sirius_physical_grouped_aggregate : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::HASH_GROUP_BY;

 public:
  sirius_physical_grouped_aggregate(
    duckdb::ClientContext& context,
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups,
    duckdb::idx_t estimated_cardinality);

  sirius_physical_grouped_aggregate(
    duckdb::ClientContext& context,
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups,
    duckdb::vector<duckdb::GroupingSet> grouping_sets,
    duckdb::vector<duckdb::unsafe_vector<duckdb::idx_t>> grouping_functions,
    duckdb::idx_t estimated_cardinality,
    duckdb::TupleDataValidityType group_validity,
    duckdb::TupleDataValidityType distinct_validity);

  duckdb::vector<duckdb::GroupingSet> grouping_sets;

  // TODO: we may need some of these variables later when we implement grouping sets

  // //! The grouping sets
  // duckdb::GroupedAggregateData grouped_aggregate_data;

  // //! The radix partitioned hash tables (one per grouping set)
  // duckdb::vector<duckdb::HashAggregateGroupingData> groupings;
  // duckdb::unique_ptr<duckdb::DistinctAggregateCollectionInfo> distinct_collection_info;
  // //! A recreation of the input chunk, with nulls for everything that isn't a group
  // duckdb::vector<duckdb::LogicalType> input_group_types;

  // // Filters given to sink and friends
  // duckdb::unsafe_vector<duckdb::idx_t> non_distinct_filter;
  // duckdb::unsafe_vector<duckdb::idx_t> distinct_filter;

  // duckdb::unordered_map<duckdb::Expression*, size_t> filter_indexes;

  // Grouped aggregatge definitions for cudf compute
  std::vector<int> group_idx;
  std::vector<cudf::aggregation::Kind> cudf_aggregates;
  std::vector<int> cudf_aggregate_idx;

 public:
  // Source interface
  bool is_source() const override { return true; }

  duckdb::OrderPreservationType source_order() const override
  {
    return duckdb::OrderPreservationType::NO_ORDER;
  }

  // Sink interface
  bool is_sink() const override { return true; }

  bool sink_order_dependent() const override { return false; }

  std::vector<std::shared_ptr<::cucascade::data_batch>> execute(
    const std::vector<std::shared_ptr<::cucascade::data_batch>>& input_batches,
    rmm::cuda_stream_view stream = cudf::get_default_stream()) override;
};

}  // namespace op
}  // namespace sirius
