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

#include "planner/scan_column_origin.hpp"

#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_join.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>

namespace sirius::planner {

namespace {

struct ordinal_origin {
  std::size_t child_index;
  std::size_t child_ordinal;
};

/// Maps @p ordinal through @p projection_map onto child @p child_index, or passes it through
/// when the map is empty.
std::optional<ordinal_origin> mapped_origin(std::size_t child_index,
                                            duckdb::vector<duckdb::idx_t> const& projection_map,
                                            std::size_t ordinal,
                                            std::size_t child_width)
{
  if (projection_map.empty()) {
    if (ordinal >= child_width) { return std::nullopt; }
    return ordinal_origin{child_index, ordinal};
  }
  if (ordinal >= projection_map.size()) { return std::nullopt; }
  return ordinal_origin{child_index, static_cast<std::size_t>(projection_map[ordinal])};
}

/// Join output per LogicalJoin::ResolveTypes: the projected left block, then the projected right
/// block for join types that emit both sides. MARK appends a BOOLEAN after the left block.
std::optional<ordinal_origin> join_origin(duckdb::LogicalJoin const& join,
                                          std::size_t ordinal,
                                          origin_policy policy)
{
  if (join.children.size() != 2) { return std::nullopt; }
  auto const left_width  = join.children[0]->types.size();
  auto const right_width = join.children[1]->types.size();
  auto const left_block =
    join.left_projection_map.empty() ? left_width : join.left_projection_map.size();
  auto const left = [&] { return mapped_origin(0, join.left_projection_map, ordinal, left_width); };
  auto const right = [&] {
    return mapped_origin(1, join.right_projection_map, ordinal - left_block, right_width);
  };
  switch (join.join_type) {
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
    case duckdb::JoinType::MARK:
      if (ordinal >= left_block) { return std::nullopt; }
      return left();
    case duckdb::JoinType::RIGHT_SEMI:
    case duckdb::JoinType::RIGHT_ANTI:
      return mapped_origin(1, join.right_projection_map, ordinal, right_width);
    case duckdb::JoinType::SINGLE:
      // At most one match per left row: the left side is a row subset, the right is padded.
      if (ordinal < left_block) { return left(); }
      if (policy == origin_policy::row_subset) { return std::nullopt; }
      return right();
    default:
      // Either side may be duplicated, so only the values survive.
      if (policy == origin_policy::row_subset) { return std::nullopt; }
      if (ordinal < left_block) { return left(); }
      return right();
  }
}

/// The child ordinal whose column output @p ordinal of @p op forwards, when @p op is one that
/// forwards columns under @p policy.
std::optional<ordinal_origin> step_origin(duckdb::LogicalOperator const& op,
                                          std::size_t ordinal,
                                          origin_policy policy)
{
  if (op.children.empty()) { return std::nullopt; }
  auto const child_width = op.children[0]->types.size();
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
      auto const& projection = op.Cast<duckdb::LogicalProjection>();
      if (ordinal >= projection.expressions.size()) { return std::nullopt; }
      auto const& expression = *projection.expressions[ordinal];
      if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        return std::nullopt;
      }
      return ordinal_origin{
        0, static_cast<std::size_t>(expression.Cast<duckdb::BoundReferenceExpression>().index)};
    }
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
      return mapped_origin(
        0, op.Cast<duckdb::LogicalFilter>().projection_map, ordinal, child_width);
    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
      return mapped_origin(0, op.Cast<duckdb::LogicalOrder>().projection_map, ordinal, child_width);
    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
    case duckdb::LogicalOperatorType::LOGICAL_DISTINCT:
      return mapped_origin(0, {}, ordinal, child_width);
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
      auto const& aggregate = op.Cast<duckdb::LogicalAggregate>();
      // Several grouping sets repeat rows; the keys a set omits are padded with NULL.
      if (policy == origin_policy::row_subset && aggregate.grouping_sets.size() > 1) {
        return std::nullopt;
      }
      if (ordinal >= aggregate.groups.size()) { return std::nullopt; }
      auto const& group = *aggregate.groups[ordinal];
      if (group.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return std::nullopt; }
      return ordinal_origin{
        0, static_cast<std::size_t>(group.Cast<duckdb::BoundReferenceExpression>().index)};
    }
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
      return join_origin(op.Cast<duckdb::LogicalJoin>(), ordinal, policy);
    case duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_ASOF_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN:
      if (policy == origin_policy::row_subset) { return std::nullopt; }
      return join_origin(op.Cast<duckdb::LogicalJoin>(), ordinal, policy);
    case duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT: {
      if (policy == origin_policy::row_subset || op.children.size() != 2) { return std::nullopt; }
      if (ordinal < child_width) { return ordinal_origin{0, ordinal}; }
      return mapped_origin(1, {}, ordinal - child_width, op.children[1]->types.size());
    }
    default: return std::nullopt;
  }
}

/// Table in-out functions and scans fed by a child have no single base column domain.
bool admissible_base_scan(duckdb::LogicalGet const& get, std::size_t ordinal)
{
  if (!get.children.empty() || !get.projected_input.empty()) { return false; }
  auto const width =
    get.projection_ids.empty() ? get.GetColumnIds().size() : get.projection_ids.size();
  return ordinal < width;
}

}  // namespace

std::optional<scan_column_origin> resolve_scan_column_origin(duckdb::LogicalOperator const& subtree,
                                                             std::size_t output_ordinal,
                                                             origin_policy policy)
{
  auto const* node = &subtree;
  auto ordinal     = output_ordinal;
  while (node->type != duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto const step = step_origin(*node, ordinal, policy);
    if (!step || step->child_index >= node->children.size()) { return std::nullopt; }
    node    = node->children[step->child_index].get();
    ordinal = step->child_ordinal;
  }
  auto const& get = node->Cast<duckdb::LogicalGet>();
  if (!admissible_base_scan(get, ordinal)) { return std::nullopt; }
  return scan_column_origin{&get, ordinal};
}

}  // namespace sirius::planner
