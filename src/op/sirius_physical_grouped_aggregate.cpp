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

#include "op/sirius_physical_grouped_aggregate.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"

namespace sirius {
namespace op {

// TODO: we may need some of these functions later when we implement grouping sets
// static duckdb::vector<duckdb::LogicalType> create_group_chunk_types(
//   duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups)
// {
//   duckdb::set<duckdb::idx_t> group_indices;

//   if (groups.empty()) { return {}; }

//   for (auto& group : groups) {
//     D_ASSERT(group->type == duckdb::ExpressionType::BOUND_REF);
//     auto& bound_ref = group->Cast<duckdb::BoundReferenceExpression>();
//     group_indices.insert(bound_ref.index);
//   }
//   duckdb::idx_t highest_index = *group_indices.rbegin();
//   duckdb::vector<duckdb::LogicalType> types(highest_index + 1, duckdb::LogicalType::SQLNULL);
//   for (auto& group : groups) {
//     auto& bound_ref        = group->Cast<duckdb::BoundReferenceExpression>();
//     types[bound_ref.index] = bound_ref.return_type;
//   }
//   return types;
// }

sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::ClientContext& context,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups_p,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_grouped_aggregate(context,
                                      std::move(types),
                                      std::move(expressions),
                                      std::move(groups_p),
                                      {},
                                      {},
                                      estimated_cardinality,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES)
{
}

// expressions is the list of aggregates to be computed. Each aggregates has a bound_ref expression
// to a column groups_p is the list of group by columns. Each group by column is a bound_ref
// expression to a column grouping_sets_p is the list of grouping set. Each grouping set is a set of
// indexes to the group by columns. Seems like DuckDB group the groupby columns into several sets
// and for every grouping set there is one radix_table grouping_functions_p is a list of indexes to
// the groupby expressions (groups_p) for each grouping_sets. The first level of the vector is the
// grouping set and the second level is the indexes to the groupby expression for that set.
sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::ClientContext& context,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups_p,
  duckdb::vector<duckdb::GroupingSet> grouping_sets_p,
  duckdb::vector<duckdb::unsafe_vector<duckdb::idx_t>> grouping_functions_p,
  duckdb::idx_t estimated_cardinality,
  duckdb::TupleDataValidityType group_validity,
  duckdb::TupleDataValidityType distinct_validity)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
    grouping_sets(std::move(grouping_sets_p))
{
  // TODO: for now commenting out this code because we are not using grouping sets yet. Will add it
  // back later when necessary.

  // // get a list of all aggregates to be computed
  // const duckdb::idx_t group_count = groups_p.size();
  // if (grouping_sets.empty()) {
  //   duckdb::GroupingSet set;
  //   for (duckdb::idx_t i = 0; i < group_count; i++) {
  //     set.insert(i);
  //   }
  //   grouping_sets.push_back(std::move(set));
  // }
  // input_group_types = create_group_chunk_types(groups_p);

  // grouped_aggregate_data.InitializeGroupby(
  //   std::move(groups_p), std::move(expressions), std::move(grouping_functions_p));

  // auto& aggregates = grouped_aggregate_data.aggregates;
  // // filter_indexes must be pre-built, not lazily instantiated in parallel...
  // // Because everything that lives in this class should be read-only at execution time
  // idx_t aggregate_input_idx = 0;
  // for (idx_t i = 0; i < aggregates.size(); i++) {
  //   auto& aggregate = aggregates[i];
  //   auto& aggr      = aggregate->Cast<duckdb::BoundAggregateExpression>();
  //   aggregate_input_idx += aggr.children.size();
  //   if (aggr.aggr_type == duckdb::AggregateType::DISTINCT) {
  //     distinct_filter.push_back(i);
  //   } else if (aggr.aggr_type == duckdb::AggregateType::NON_DISTINCT) {
  //     non_distinct_filter.push_back(i);
  //   } else {  // LCOV_EXCL_START
  //     throw duckdb::NotImplementedException(
  //       "AggregateType not implemented in PhysicalHashAggregate");
  //   }  // LCOV_EXCL_STOP
  // }

  // for (idx_t i = 0; i < aggregates.size(); i++) {
  //   auto& aggregate = aggregates[i];
  //   auto& aggr      = aggregate->Cast<duckdb::BoundAggregateExpression>();
  //   if (aggr.filter) {
  //     auto& bound_ref_expr = aggr.filter->Cast<duckdb::BoundReferenceExpression>();
  //     if (!filter_indexes.count(aggr.filter.get())) {
  //       // Replace the bound reference expression's index with the corresponding index of the
  //       // payload chunk
  //       // TODO: Still not quite sure why duckdb replace the index
  //       filter_indexes[aggr.filter.get()] = bound_ref_expr.index;
  //       bound_ref_expr.index              = aggregate_input_idx;
  //     }
  //     aggregate_input_idx++;
  //   }
  // }

  // distinct_collection_info =
  //   duckdb::DistinctAggregateCollectionInfo::Create(grouped_aggregate_data.aggregates);

  // for (idx_t i = 0; i < grouping_sets.size(); i++) {
  //   groupings.emplace_back(grouping_sets[i],
  //                          grouped_aggregate_data,
  //                          distinct_collection_info,
  //                          group_validity,
  //                          distinct_validity);
  // }

  // // The output of groupby is ordered as the grouping columns first followed by the aggregate
  // // columns See RadixHTLocalSourceState::Scan for more details
  // idx_t total_output_columns = 0;
  // for (auto& aggregate : aggregates) {
  //   auto& aggr = aggregate->Cast<duckdb::BoundAggregateExpression>();
  //   total_output_columns++;
  // }
  // total_output_columns += grouped_aggregate_data.GroupCount();

  auto cudf_defs     = convert_duckdb_aggregates_to_cudf(groups_p, expressions);
  group_idx          = std::move(cudf_defs.group_idx);
  cudf_aggregates    = std::move(cudf_defs.cudf_aggregates);
  cudf_aggregate_idx = std::move(cudf_defs.cudf_aggregate_idx);
}

std::vector<std::shared_ptr<::cucascade::data_batch>> sirius_physical_grouped_aggregate::execute(
  const std::vector<std::shared_ptr<::cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  for (auto& input_batch : input_batches) {
    auto result = gpu_aggregate_impl::local_grouped_aggregate(input_batch,
                                                              group_idx,
                                                              cudf_aggregates,
                                                              cudf_aggregate_idx,
                                                              stream,
                                                              *input_batch->get_memory_space());
    results.push_back(std::move(result));
  }
  return results;
}
}  // namespace op
}  // namespace sirius
