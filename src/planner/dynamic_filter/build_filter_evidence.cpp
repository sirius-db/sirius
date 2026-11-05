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
#include "duckdb/planner/operator/logical_get.hpp"
#include "transparent/parquet_footer_statistics.hpp"

#include <cstddef>
#include <optional>

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

/// Share of the table's rows above which a GET filter is not filter evidence.
constexpr double k_negligible_filter_keep_ratio = 0.99;

std::optional<std::size_t> table_rows(duckdb::LogicalGet const& get, duckdb::ClientContext& context)
{
  if (get.function.cardinality && get.bind_data) {
    auto const table = get.function.cardinality(context, get.bind_data.get());
    if (table && table->has_estimated_cardinality) { return table->estimated_cardinality; }
  }
  return sirius::transparent::cached_parquet_row_count(get);
}

bool keeps_nearly_all_rows(duckdb::LogicalGet const& get, duckdb::ClientContext& context)
{
  if (!get.has_estimated_cardinality) { return false; }
  try {
    auto const rows = table_rows(get, context);
    if (!rows || *rows == 0) { return false; }
    return static_cast<double>(get.estimated_cardinality) >=
           k_negligible_filter_keep_ratio * static_cast<double>(*rows);
  } catch (...) {
    // Optional evidence must not fail query planning.
    return false;
  }
}

}  // namespace

bool build_subtree_is_filtering(duckdb::LogicalOperator const& op, duckdb::ClientContext* context)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET: {
      auto const& get = op.Cast<duckdb::LogicalGet>();
      if (!get.table_filters.HasFilters() && !get.table_filters.HasMultiColumnFilters()) {
        return false;
      }
      return context == nullptr || !keeps_nearly_all_rows(get, *context);
    }
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N: return true;
    default: break;
  }
  for (auto const& child : op.children) {
    if (child && build_subtree_is_filtering(*child, context)) { return true; }
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

}  // namespace sirius::planner
