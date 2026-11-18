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

#include "planner/build_side_choice.hpp"

#include "planner/dynamic_filter/build_filter_evidence.hpp"

#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/enums/join_type.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <cstdint>
#include <utility>

namespace sirius::planner {

namespace {

bool swappable(duckdb::LogicalOperator const& op)
{
  if (op.type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return false; }
  auto const& join = op.Cast<duckdb::LogicalComparisonJoin>();
  return join.join_type == duckdb::JoinType::INNER && join.duplicate_eliminated_columns.empty() &&
         std::all_of(join.conditions.begin(),
                     join.conditions.end(),
                     [](auto const& condition) { return condition.IsComparison(); }) &&
         std::any_of(join.conditions.begin(), join.conditions.end(), [](auto const& condition) {
           return condition.GetComparisonType() == duckdb::ExpressionType::COMPARE_EQUAL;
         });
}

/// DuckDB's BuildProbeSideOptimizer::FlipChildren for an inner comparison join.
void flip(duckdb::LogicalComparisonJoin& join)
{
  std::swap(join.children[0], join.children[1]);
  for (auto& condition : join.conditions) {
    condition =
      duckdb::JoinCondition(condition.RightReference()->Copy(),
                            condition.LeftReference()->Copy(),
                            duckdb::FlipComparisonExpression(condition.GetComparisonType()));
  }
  std::swap(join.left_projection_map, join.right_projection_map);
}

/// The bytes of one row @p op returns. A variable-size column counts as 16 bytes, the rule of
/// push_narrowing_expressions.
std::uint64_t row_bytes(duckdb::LogicalOperator& op)
{
  op.ResolveOperatorTypes();
  std::uint64_t bytes = 0;
  for (auto const& type : op.types) {
    auto const physical = type.InternalType();
    bytes += duckdb::TypeIsConstantSize(physical) ? duckdb::GetTypeIdSize(physical) : 16;
  }
  return bytes;
}

/// Whether @p rows rows of @p width bytes each are at least @p bytes, without overflow.
bool at_least(std::uint64_t rows, std::uint64_t width, std::uint64_t bytes)
{
  if (width == 0) { return bytes == 0; }
  return rows >= bytes / width + (bytes % width == 0 ? 0 : 1);
}

}  // namespace

std::size_t swap_unfiltered_builds(duckdb::ClientContext& context,
                                   duckdb::LogicalOperator& plan,
                                   std::uint64_t min_build_bytes)
{
  std::size_t swapped = 0;
  for (auto& child : plan.children) {
    swapped += swap_unfiltered_builds(context, *child, min_build_bytes);
  }
  if (!swappable(plan)) { return swapped; }
  auto& probe = *plan.children[0];
  auto& build = *plan.children[1];
  if (build_subtree_is_filtering(build) || build_relation_is_opaque(build) ||
      !build_subtree_is_filtering(probe)) {
    return swapped;
  }
  auto const build_rows  = build.EstimateCardinality(context);
  auto const build_width = row_bytes(build);
  auto const swap        = at_least(build_rows, build_width, min_build_bytes);
  SIRIUS_LOG_INFO(
    "[build_side_choice] inner join with an unfiltered build of {} estimated rows of {} bytes and "
    "a filtered probe, swapped from {} bytes of build: {}.",
    build_rows,
    build_width,
    min_build_bytes,
    swap ? "swapped" : "kept");
  if (swap) {
    flip(plan.Cast<duckdb::LogicalComparisonJoin>());
    ++swapped;
  }
  return swapped;
}

}  // namespace sirius::planner
