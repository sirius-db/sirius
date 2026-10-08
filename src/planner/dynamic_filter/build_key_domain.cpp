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

#include "planner/dynamic_filter/build_key_domain.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "planner/scan_column_origin.hpp"

namespace sirius::planner {

namespace detail {

duckdb::LogicalGet const* resolve_pass_through_scan(duckdb::LogicalOperator const& subtree,
                                                    std::size_t output_ordinal) noexcept
{
  auto const origin = resolve_scan_column_origin(subtree, output_ordinal);
  return origin ? origin->get : nullptr;
}

std::vector<duckdb::LogicalGet const*> resolve_build_key_scans(
  duckdb::LogicalComparisonJoin const& join)
{
  std::vector<duckdb::LogicalGet const*> scans(join.conditions.size(), nullptr);
  if (join.children.size() != 2) { return scans; }
  for (std::size_t condition_index = 0; condition_index < join.conditions.size();
       ++condition_index) {
    auto const& build_side = *join.conditions[condition_index].right;
    if (build_side.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { continue; }
    auto const ordinal =
      static_cast<std::size_t>(build_side.Cast<duckdb::BoundReferenceExpression>().index);
    scans[condition_index] = resolve_pass_through_scan(*join.children[1], ordinal);
  }
  return scans;
}

}  // namespace detail

duckdb_base_table_cardinality::duckdb_base_table_cardinality(
  duckdb::ClientContext& context) noexcept
  : _context{&context}
{
}

std::optional<std::size_t> duckdb_base_table_cardinality::operator()(
  duckdb::LogicalGet const& get) const noexcept
{
  // Other table functions may report estimates below the true domain.
  if (get.function.name != "seq_scan" || !get.function.cardinality || !get.bind_data) {
    return std::nullopt;
  }
  try {
    auto const stats = get.function.cardinality(*_context, get.bind_data.get());
    if (!stats || !stats->has_max_cardinality) { return std::nullopt; }
    return static_cast<std::size_t>(stats->max_cardinality);
  } catch (...) {
    // Optional evidence must not fail query planning.
    return std::nullopt;
  }
}

}  // namespace sirius::planner
