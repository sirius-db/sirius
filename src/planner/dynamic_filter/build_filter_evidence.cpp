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

#include "planner/dynamic_filter/build_filter_evidence.hpp"

#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "sirius_config.hpp"

namespace sirius::planner {
namespace {

bool is_plain_deduplication(duckdb::LogicalAggregate const& aggregate)
{
  if (aggregate.groups.empty() || !aggregate.expressions.empty() ||
      !aggregate.grouping_functions.empty() || aggregate.grouping_sets.size() > 1) {
    return false;
  }
  for (auto const& group : aggregate.groups) {
    if (!group) { return false; }
  }
  // An empty vector is DuckDB's implicit ordinary GROUP BY representation.
  if (!aggregate.grouping_sets.empty()) {
    auto const& grouping_set = aggregate.grouping_sets.front();
    if (grouping_set.size() != aggregate.groups.size()) { return false; }
    for (auto index : duckdb::ProjectionIndex::GetIndexes(aggregate.groups.size())) {
      if (grouping_set.count(index) == 0) { return false; }
    }
  }
  return true;
}

}  // namespace

bool build_subtree_is_filtering(duckdb::LogicalOperator const& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET:
      return op.Cast<duckdb::LogicalGet>().table_filters.HasFilters() ||
             op.Cast<duckdb::LogicalGet>().table_filters.HasMultiColumnFilters();
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N: return true;
    default: break;
  }
  for (auto const& child : op.children) {
    if (child && build_subtree_is_filtering(*child)) { return true; }
  }
  return false;
}

bool build_relation_is_opaque(duckdb::LogicalOperator const& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_GET:
    case duckdb::LogicalOperatorType::LOGICAL_CTE_REF: return true;
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
      // Computed groups are valid: the filter uses actual build output, not input-key lineage.
      if (!is_plain_deduplication(op.Cast<duckdb::LogicalAggregate>())) { return false; }
      [[fallthrough]];
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
      return op.children.size() == 1 && op.children.front() != nullptr &&
             build_relation_is_opaque(*op.children.front());
    default: return false;
  }
}

bool build_subtree_has_filtering_join(duckdb::LogicalOperator const& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
      op.type == duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN) {
    switch (op.Cast<duckdb::LogicalComparisonJoin>().join_type) {
      case duckdb::JoinType::SEMI:
      case duckdb::JoinType::ANTI:
      case duckdb::JoinType::RIGHT_SEMI:
      case duckdb::JoinType::RIGHT_ANTI: return true;
      default: break;
    }
  }
  for (auto const& child : op.children) {
    if (child && build_subtree_has_filtering_join(*child)) { return true; }
  }
  return false;
}

bool right_family_publication_enabled(sirius::operator_params const& params)
{
  return params.enable_dynamic_filter && params.enable_dynamic_filter_multi_partition &&
         params.max_dynamic_filter_bloom_bytes_per_gpu() > 0;
}

}  // namespace sirius::planner
