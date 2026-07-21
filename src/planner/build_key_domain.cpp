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

#include "planner/build_key_domain.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"

namespace sirius::planner {

namespace {

/// Which child of `op`, at which of that child's output ordinals, produces `op`'s output ordinal
/// `output_ordinal` -- when `op` produces it by value-preserving pass-through and `op`'s rows are
/// an injective image of that child's rows. Absent otherwise. The only place that knows the
/// LogicalOperator taxonomy.
struct ordinal_origin {
  std::size_t child_index;
  std::size_t child_ordinal;
};

std::optional<ordinal_origin> pass_through_origin(duckdb::LogicalOperator const& op,
                                                  std::size_t output_ordinal) noexcept
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
      auto const& projection = op.Cast<duckdb::LogicalProjection>();
      if (output_ordinal >= projection.expressions.size()) { return std::nullopt; }
      auto const& expression = *projection.expressions[output_ordinal];
      // A plain reference only: a computed expression -- including a cast -- does not pass the
      // base column's values through.
      if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        return std::nullopt;
      }
      return ordinal_origin{
        0, static_cast<std::size_t>(expression.Cast<duckdb::BoundReferenceExpression>().index)};
    }
    case duckdb::LogicalOperatorType::LOGICAL_FILTER: {
      auto const& filter = op.Cast<duckdb::LogicalFilter>();
      if (!filter.HasProjectionMap()) { return ordinal_origin{0, output_ordinal}; }
      if (output_ordinal >= filter.projection_map.size()) { return std::nullopt; }
      return ordinal_origin{0, static_cast<std::size_t>(filter.projection_map[output_ordinal])};
    }
    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY: {
      auto const& order = op.Cast<duckdb::LogicalOrder>();
      if (!order.HasProjectionMap()) { return ordinal_origin{0, output_ordinal}; }
      if (output_ordinal >= order.projection_map.size()) { return std::nullopt; }
      return ordinal_origin{0, static_cast<std::size_t>(order.projection_map[output_ordinal])};
    }
    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
    case duckdb::LogicalOperatorType::LOGICAL_DISTINCT:
      // Row truncation or deduplication: a subset of the child's rows, no column remapping.
      return ordinal_origin{0, output_ordinal};
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
      auto const& aggregate = op.Cast<duckdb::LogicalAggregate>();
      // Output layout: groups, then aggregate expressions, then grouping functions. Only a group
      // that is a plain reference passes base-column values through; one row per distinct group
      // is a subset of the child's rows. Multiple grouping sets repeat rows across sets.
      if (aggregate.grouping_sets.size() > 1) { return std::nullopt; }
      if (output_ordinal >= aggregate.groups.size()) { return std::nullopt; }
      auto const& group = *aggregate.groups[output_ordinal];
      if (group.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return std::nullopt; }
      return ordinal_origin{
        0, static_cast<std::size_t>(group.Cast<duckdb::BoundReferenceExpression>().index)};
    }
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
      auto const& join = op.Cast<duckdb::LogicalComparisonJoin>();
      if (join.children.size() != 2) { return std::nullopt; }
      // Output layout (LogicalJoin::ResolveTypes): the left block through left_projection_map,
      // then the right block through right_projection_map; SEMI/ANTI emit only the left block,
      // RIGHT_SEMI/RIGHT_ANTI only the right block, MARK the left block plus one appended
      // BOOLEAN column. Continue only when the join's output rows are, by join semantics alone,
      // an injective image of the traced side's rows.
      auto const left_width  = join.left_projection_map.empty() ? join.children[0]->types.size()
                                                                : join.left_projection_map.size();
      auto const left_origin = [&join](std::size_t ordinal) {
        return ordinal_origin{0,
                              join.left_projection_map.empty()
                                ? ordinal
                                : static_cast<std::size_t>(join.left_projection_map[ordinal])};
      };
      switch (join.join_type) {
        case duckdb::JoinType::SEMI:
        case duckdb::JoinType::ANTI:
          // Output is only the left block, a subset of the left rows.
          if (output_ordinal >= left_width) { return std::nullopt; }
          return std::optional{left_origin(output_ordinal)};
        case duckdb::JoinType::RIGHT_SEMI:
        case duckdb::JoinType::RIGHT_ANTI: {
          // Output is only the right block, a subset of the right rows.
          auto const right_width = join.right_projection_map.empty()
                                     ? join.children[1]->types.size()
                                     : join.right_projection_map.size();
          if (output_ordinal >= right_width) { return std::nullopt; }
          return std::optional{ordinal_origin{
            1,
            join.right_projection_map.empty()
              ? output_ordinal
              : static_cast<std::size_t>(join.right_projection_map[output_ordinal])}};
        }
        case duckdb::JoinType::MARK:
        case duckdb::JoinType::SINGLE:
          // MARK: one output row per left row; the appended mark ordinal is not a base column.
          // SINGLE: exactly one right row per left row; right-block values may repeat.
          if (output_ordinal >= left_width) { return std::nullopt; }
          return std::optional{left_origin(output_ordinal)};
        default:
          // INNER, LEFT, RIGHT, OUTER, and anything unmodelled: the opposite side may multiply
          // the traced side's rows, which would inflate the coverage ratio and over-fire the
          // gate.
          return std::nullopt;
      }
    }
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN:
      // Refused explicitly rather than by omission: dynamic-filter routing excludes CTE/DELIM
      // producer paths entirely, and this walk must not quietly disagree.
      return std::nullopt;
    default: return std::nullopt;
  }
}

/// A LOGICAL_GET terminates the walk successfully only when it is a plain base-table scan and the
/// ordinal lies within the scan's own width. A table-in-out function appends its child's columns
/// after the scan columns, so "the base table's row count" is not meaningful for such a node at
/// all; refuse the whole node rather than the ordinal.
bool admissible_base_scan(duckdb::LogicalGet const& get, std::size_t ordinal) noexcept
{
  if (!get.children.empty() || !get.projected_input.empty()) { return false; }
  auto const scan_width =
    get.projection_ids.empty() ? get.GetColumnIds().size() : get.projection_ids.size();
  return ordinal < scan_width;
}

}  // namespace

namespace detail {

duckdb::LogicalGet const* resolve_pass_through_scan(duckdb::LogicalOperator const& subtree,
                                                    std::size_t output_ordinal) noexcept
{
  // A path, not a tree traversal: each step strictly descends into exactly one child, so the loop
  // terminates on any finite tree and can never wander into the wrong subtree.
  auto const* node = &subtree;
  auto ordinal     = output_ordinal;
  while (node->type != duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto const step = pass_through_origin(*node, ordinal);
    if (!step || step->child_index >= node->children.size()) { return nullptr; }
    node    = node->children[step->child_index].get();
    ordinal = step->child_ordinal;
  }
  auto const& get = node->Cast<duckdb::LogicalGet>();
  return admissible_base_scan(get, ordinal) ? &get : nullptr;
}

std::vector<duckdb::LogicalGet const*> resolve_build_key_scans(
  duckdb::LogicalComparisonJoin const& join)
{
  std::vector<duckdb::LogicalGet const*> scans(join.conditions.size(), nullptr);
  if (join.children.size() != 2) { return scans; }
  for (std::size_t condition_index = 0; condition_index < join.conditions.size();
       ++condition_index) {
    auto const& build_side = *join.conditions[condition_index].right;
    // Post-ColumnBindingResolver a plain build key is a BOUND_REF whose index is an output
    // ordinal of the build child. A cast or computed side stays null: a cast key is never
    // admitted, and a computed key's values are not the base column's values.
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
  // Allowlist: only DuckDB's native table scan, whose cardinality callback returns
  // NodeStatistics::max_cardinality = committed rows plus transaction-local inserts -- a true
  // upper bound on the table's rows. Every other table function promises only an expected
  // cardinality, which may under-state the domain and over-fire the gate.
  if (get.function.name != "seq_scan" || !get.function.cardinality || !get.bind_data) {
    return std::nullopt;
  }
  try {
    auto const stats = get.function.cardinality(*_context, get.bind_data.get());
    if (!stats || !stats->has_max_cardinality) { return std::nullopt; }
    return static_cast<std::size_t>(stats->max_cardinality);
  } catch (...) {
    // An escaped exception here would fail query planning for the sake of an optional
    // optimization; refusal is the only failure mode.
    return std::nullopt;
  }
}

}  // namespace sirius::planner
