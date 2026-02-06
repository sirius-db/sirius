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
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/merge/gpu_merge_impl.hpp"

#include <iostream>

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
  : sirius_physical_grouped_aggregate_merge(grouped_aggregate->types,  // copied by value
                                            grouped_aggregate->group_idx,
                                            grouped_aggregate->cudf_aggregates,
                                            grouped_aggregate->cudf_aggregate_idx,
                                            grouped_aggregate->estimated_cardinality)
{
  child_op = grouped_aggregate;
}

sirius_physical_grouped_aggregate_merge::sirius_physical_grouped_aggregate_merge(
  duckdb::vector<duckdb::LogicalType> types,
  std::vector<int> group_idx,
  std::vector<cudf::aggregation::Kind> cudf_aggregates,
  std::vector<int> cudf_aggregate_idx,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::MERGE_GROUP_BY, std::move(types), estimated_cardinality),
    group_idx(std::move(group_idx)),
    cudf_aggregates(std::move(cudf_aggregates)),
    cudf_aggregate_idx(std::move(cudf_aggregate_idx))
{
}

sirius_physical_grouped_aggregate_merge::sirius_physical_grouped_aggregate_merge(
  duckdb::ClientContext& context,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups_p,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_grouped_aggregate_merge(context,
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
sirius_physical_grouped_aggregate_merge::sirius_physical_grouped_aggregate_merge(
  duckdb::ClientContext& context,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups_p,
  duckdb::vector<duckdb::GroupingSet> grouping_sets_p,
  duckdb::vector<duckdb::unsafe_vector<duckdb::idx_t>> grouping_functions_p,
  duckdb::idx_t estimated_cardinality,
  duckdb::TupleDataValidityType group_validity,
  duckdb::TupleDataValidityType distinct_validity)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::MERGE_GROUP_BY, std::move(types), estimated_cardinality),
    grouping_sets(std::move(grouping_sets_p))
{
  // Convert input parameters to cudf compute definitions BEFORE moving them
  auto cudf_defs     = convert_duckdb_aggregates_to_cudf(groups_p, expressions);
  group_idx          = std::move(cudf_defs.group_idx);
  cudf_aggregates    = std::move(cudf_defs.cudf_aggregates);
  cudf_aggregate_idx = std::move(cudf_defs.cudf_aggregate_idx);
}

std::optional<std::vector<std::shared_ptr<::cucascade::data_batch>>>
sirius_physical_grouped_aggregate_merge::get_next_task_input_batch()
{
  // we need to lock, then pull all the batches from one partition and return them, and increment
  // the partition index
  std::lock_guard<std::mutex> lg(lock);
  if (current_partition_index < ports.begin()->second->repo->num_partitions()) {
    std::vector<::std::shared_ptr<::cucascade::data_batch>> input_batch;
    bool found_batch = true;
    while (found_batch) {
      auto batch = ports.begin()->second->repo->pop_data_batch(
        ::cucascade::batch_state::task_created, current_partition_index);
      if (batch) {
        input_batch.push_back(std::move(batch));
      } else {
        found_batch = false;
      }
    }
    current_partition_index++;
    return input_batch;
  } else {
    return std::nullopt;
  }
}

std::vector<std::shared_ptr<::cucascade::data_batch>>
sirius_physical_grouped_aggregate_merge::execute(
  const std::vector<std::shared_ptr<::cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  if (input_batches.size() == 0) {
    throw std::runtime_error(
      "We expect at least one input batch for grouped aggregate merge operator");
  }
  // if there is only one batch, return it. We are assuming it was already aggregated.
  if (input_batches.size() == 1) { return input_batches; }

  auto result = gpu_merge_impl::merge_grouped_aggregate(input_batches,
                                                        group_idx.size(),
                                                        cudf_aggregates,
                                                        stream,
                                                        *input_batches[0]->get_memory_space());
  return {result};
}
}  // namespace op
}  // namespace sirius
