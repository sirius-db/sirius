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

#include "op/sirius_physical_grouped_aggregate_merge.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"

namespace sirius {
namespace op {

static duckdb::vector<duckdb::LogicalType> create_group_chunk_types(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups)
{
  duckdb::set<duckdb::idx_t> group_indices;

  if (groups.empty()) { return {}; }

  for (auto& group : groups) {
    D_ASSERT(group->type == duckdb::ExpressionType::BOUND_REF);
    auto& bound_ref = group->Cast<duckdb::BoundReferenceExpression>();
    group_indices.insert(bound_ref.index);
  }
  duckdb::idx_t highest_index = *group_indices.rbegin();
  duckdb::vector<duckdb::LogicalType> types(highest_index + 1, duckdb::LogicalType::SQLNULL);
  for (auto& group : groups) {
    auto& bound_ref        = group->Cast<duckdb::BoundReferenceExpression>();
    types[bound_ref.index] = bound_ref.return_type;
  }
  return types;
}

// Helper to deep copy a vector of Expression unique_ptrs
static duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> copy_expressions(
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& src)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> result;
  result.reserve(src.size());
  for (const auto& expr : src) {
    result.push_back(expr->Copy());
  }
  return result;
}

// Helper to convert vector<vector<idx_t>> to vector<unsafe_vector<idx_t>>
static duckdb::vector<duckdb::unsafe_vector<duckdb::idx_t>> convert_grouping_functions(
  const duckdb::vector<duckdb::vector<duckdb::idx_t>>& src)
{
  duckdb::vector<duckdb::unsafe_vector<duckdb::idx_t>> result;
  result.reserve(src.size());
  for (const auto& inner : src) {
    duckdb::unsafe_vector<duckdb::idx_t> converted;
    for (auto val : inner) {
      converted.push_back(val);
    }
    result.push_back(std::move(converted));
  }
  return result;
}

sirius_physical_grouped_aggregate_merge::sirius_physical_grouped_aggregate_merge(
  sirius_physical_grouped_aggregate* grouped_aggregate)
  : sirius_physical_grouped_aggregate_merge(
      grouped_aggregate->types,  // copied by value
      copy_expressions(grouped_aggregate->grouped_aggregate_data.aggregates),
      copy_expressions(grouped_aggregate->grouped_aggregate_data.groups),
      grouped_aggregate->grouping_sets,  // copied by value
      convert_grouping_functions(grouped_aggregate->grouped_aggregate_data.GetGroupingFunctions()),
      grouped_aggregate->estimated_cardinality,
      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,  // default
      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES)  // default
{
  child_op = grouped_aggregate;
}

// expressions is the list of aggregates to be computed. Each aggregates has a bound_ref expression
// to a column groups_p is the list of group by columns. Each group by column is a bound_ref
// expression to a column grouping_sets_p is the list of grouping set. Each grouping set is a set of
// indexes to the group by columns. Seems like DuckDB group the groupby columns into several sets
// and for every grouping set there is one radix_table grouping_functions_p is a list of indexes to
// the groupby expressions (groups_p) for each grouping_sets. The first level of the vector is the
// grouping set and the second level is the indexes to the groupby expression for that set.
sirius_physical_grouped_aggregate_merge::sirius_physical_grouped_aggregate_merge(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups_p,
  duckdb::vector<duckdb::GroupingSet> grouping_sets_p,
  duckdb::vector<duckdb::unsafe_vector<duckdb::idx_t>> grouping_functions_p,
  duckdb::idx_t estimated_cardinality,
  duckdb::TupleDataValidityType group_validity,
  duckdb::TupleDataValidityType distinct_validity)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_GROUP_BY, std::move(types), estimated_cardinality),
    grouping_sets(std::move(grouping_sets_p))
{
  // get a list of all aggregates to be computed
  const duckdb::idx_t group_count = groups_p.size();
  if (grouping_sets.empty()) {
    duckdb::GroupingSet set;
    for (duckdb::idx_t i = 0; i < group_count; i++) {
      set.insert(i);
    }
    grouping_sets.push_back(std::move(set));
  }
  input_group_types = create_group_chunk_types(groups_p);

  grouped_aggregate_data.InitializeGroupby(
    std::move(groups_p), std::move(expressions), std::move(grouping_functions_p));

  auto& aggregates = grouped_aggregate_data.aggregates;
  // filter_indexes must be pre-built, not lazily instantiated in parallel...
  // Because everything that lives in this class should be read-only at execution time
  idx_t aggregate_input_idx = 0;
  for (idx_t i = 0; i < aggregates.size(); i++) {
    auto& aggregate = aggregates[i];
    auto& aggr      = aggregate->Cast<duckdb::BoundAggregateExpression>();
    aggregate_input_idx += aggr.children.size();
    if (aggr.aggr_type == duckdb::AggregateType::DISTINCT) {
      distinct_filter.push_back(i);
    } else if (aggr.aggr_type == duckdb::AggregateType::NON_DISTINCT) {
      non_distinct_filter.push_back(i);
    } else {  // LCOV_EXCL_START
      throw duckdb::NotImplementedException(
        "AggregateType not implemented in PhysicalHashAggregate");
    }  // LCOV_EXCL_STOP
  }

  for (idx_t i = 0; i < aggregates.size(); i++) {
    auto& aggregate = aggregates[i];
    auto& aggr      = aggregate->Cast<duckdb::BoundAggregateExpression>();
    if (aggr.filter) {
      auto& bound_ref_expr = aggr.filter->Cast<duckdb::BoundReferenceExpression>();
      if (!filter_indexes.count(aggr.filter.get())) {
        // Replace the bound reference expression's index with the corresponding index of the
        // payload chunk
        // TODO: Still not quite sure why duckdb replace the index
        filter_indexes[aggr.filter.get()] = bound_ref_expr.index;
        bound_ref_expr.index              = aggregate_input_idx;
      }
      aggregate_input_idx++;
    }
  }

  distinct_collection_info =
    duckdb::DistinctAggregateCollectionInfo::Create(grouped_aggregate_data.aggregates);

  for (idx_t i = 0; i < grouping_sets.size(); i++) {
    groupings.emplace_back(grouping_sets[i],
                           grouped_aggregate_data,
                           distinct_collection_info,
                           group_validity,
                           distinct_validity);
  }

  // The output of groupby is ordered as the grouping columns first followed by the aggregate
  // columns See RadixHTLocalSourceState::Scan for more details
  idx_t total_output_columns = 0;
  for (auto& aggregate : aggregates) {
    auto& aggr = aggregate->Cast<duckdb::BoundAggregateExpression>();
    total_output_columns++;
  }
  total_output_columns += grouped_aggregate_data.GroupCount();
}

}  // namespace op
}  // namespace sirius
