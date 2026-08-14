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

#include "planner/late_mat_plan_pass.hpp"

#include "expression/ast/node.hpp"
#include "expression/ast/utils.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_projection.hpp"

#include <optional>
#include <variant>

namespace sirius::planner {

namespace {

using op::sirius_physical_operator;
using op::SiriusPhysicalOperatorType;

/// Whether `expr` reads the column at `pos`.
bool reads_column(ast::node const& expr, std::size_t pos)
{
  bool found = false;
  ast::visit_references(expr, [&](ast::reference const& ref) {
    if (ref.column_index == pos) { found = true; }
  });
  return found;
}

/// What one operator does to a column arriving from `from` at input position
/// `in_pos`.
///
/// `std::nullopt` means the operator reads it — the column's life ends here.
/// A value is the output position it moves to, and the column lives on.
std::optional<std::size_t> trace_through(sirius_physical_operator const& node,
                                         sirius_physical_operator const& from,
                                         std::size_t in_pos)
{
  switch (node.type) {
    case SiriusPhysicalOperatorType::HASH_JOIN: {
      auto const& join = static_cast<op::sirius_physical_hash_join const&>(node);
      // INNER only. Every other join type either emits NULLs for an unmatched
      // side — and a deferred column would have to produce a null value from a
      // null rowid — or collects just one side's columns, so a payload riding
      // on the other is not in the output at all.
      if (join.join_type != duckdb::JoinType::INNER) { return std::nullopt; }
      if (node.children.size() != 2) { return std::nullopt; }

      bool const from_lhs = node.children[0].get() == &from;
      if (!from_lhs && node.children[1].get() != &from) { return std::nullopt; }

      // A key is read: the join compares its values. The conditions' left
      // nodes address the lhs and their right nodes the rhs, so only this
      // side's half is consulted.
      for (auto const& condition : join.conditions) {
        auto const& side = from_lhs ? condition.left : condition.right;
        if (side && reads_column(*side, in_pos)) { return std::nullopt; }
      }

      std::vector<int> const lhs(join.lhs_output_columns.col_idxs.begin(),
                                 join.lhs_output_columns.col_idxs.end());
      std::vector<int> const rhs(join.rhs_output_columns.col_idxs.begin(),
                                 join.rhs_output_columns.col_idxs.end());
      return join_output_position(from_lhs, lhs, rhs, in_pos);
    }

    case SiriusPhysicalOperatorType::FILTER: {
      auto const& filter = static_cast<op::sirius_physical_filter const&>(node);
      // The predicate is the only thing a filter reads; it decides which ROWS
      // survive, never what is in the columns it passes on.
      if (filter.expression && reads_column(*filter.expression, in_pos)) { return std::nullopt; }
      // The output mask is an explicit positional map, so a filter that folds a
      // projection into its gather is still transparent to everything it keeps.
      return std::visit(
        [&](auto const& mask) -> std::optional<std::size_t> {
          using T = std::decay_t<decltype(mask)>;
          if constexpr (std::is_same_v<T, op::passthrough>) {
            return in_pos;
          } else {
            for (std::size_t out = 0; out < mask.size(); ++out) {
              if (static_cast<std::size_t>(mask[out]) == in_pos) { return out; }
            }
            return std::nullopt;  // dropped here; nothing downstream can want it
          }
        },
        filter.output_columns);
    }

    case SiriusPhysicalOperatorType::PROJECTION: {
      auto const& projection = static_cast<op::sirius_physical_projection const&>(node);
      // A bare column reference MOVES the column. Anything else computes with
      // it, which is a read.
      std::optional<std::size_t> moved_to;
      for (std::size_t out = 0; out < projection.select_list.size(); ++out) {
        auto const& expr = projection.select_list[out];
        if (!expr) { continue; }
        if (auto const* ref = std::get_if<ast::reference>(&expr->v)) {
          if (ref->column_index == in_pos && !moved_to.has_value()) { moved_to = out; }
          continue;
        }
        if (reads_column(*expr, in_pos)) { return std::nullopt; }
      }
      return moved_to;  // nullopt when the projection simply drops it
    }

    default:
      // Fail closed: an unmodelled shape is assumed to read everything. This
      // can only shorten a lifetime, so an operator missing from this switch
      // costs a deferral rather than permitting a wrong one.
      return std::nullopt;
  }
}

}  // namespace

std::optional<std::size_t> join_output_position(bool from_lhs,
                                                std::vector<int> const& lhs_projection,
                                                std::vector<int> const& rhs_projection,
                                                std::size_t in_position)
{
  auto const& own = from_lhs ? lhs_projection : rhs_projection;
  for (std::size_t i = 0; i < own.size(); ++i) {
    if (static_cast<std::size_t>(own[i]) == in_position) {
      // The output is lhs-then-rhs, so an rhs column carries the lhs's emitted
      // width as an offset — not the lhs's INPUT width, which is larger
      // whenever the join projects only part of its left side.
      return from_lhs ? i : lhs_projection.size() + i;
    }
  }
  return std::nullopt;
}

std::vector<column_lifetime> analyze_column_lifetimes(sirius_physical_operator const& scan)
{
  std::vector<column_lifetime> lifetimes;
  lifetimes.reserve(scan.types.size());

  for (std::size_t col = 0; col < scan.types.size(); ++col) {
    column_lifetime life;
    life.scan_output_position = col;
    life.position_at_reader   = col;

    std::size_t position = col;
    int boundaries       = 0;
    // `from` is which child the column arrives through, which a join needs in
    // order to know whether it is looking at its lhs or its rhs.
    auto const* from = static_cast<sirius_physical_operator const*>(&scan);
    for (auto const* node = scan.get_parent_op(); node != nullptr;
         from = node, node = node->get_parent_op()) {
      ++boundaries;
      auto const moved = trace_through(*node, *from, position);
      if (!moved.has_value()) {
        // Read here — or dropped here, which for a deferral is the same
        // answer: this is as far as the column travels.
        life.first_reader       = node;
        life.boundaries         = boundaries;
        life.position_at_reader = position;
        break;
      }
      position = *moved;
    }
    if (life.first_reader == nullptr) {
      // Nothing ever read it: the query carries this column to its output
      // untouched, which is the longest ride there is.
      life.boundaries         = boundaries;
      life.position_at_reader = position;
    }
    lifetimes.push_back(life);
  }
  return lifetimes;
}

}  // namespace sirius::planner
