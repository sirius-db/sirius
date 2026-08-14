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

/// What one operator does to a column arriving at input position `in_pos`.
///
/// `std::nullopt` means the operator reads it — the column's life ends here.
/// A value is the output position it moves to, and the column lives on.
std::optional<std::size_t> trace_through(sirius_physical_operator const& node, std::size_t in_pos)
{
  switch (node.type) {
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
    for (auto const* node = scan.get_parent_op(); node != nullptr; node = node->get_parent_op()) {
      ++boundaries;
      auto const moved = trace_through(*node, position);
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
