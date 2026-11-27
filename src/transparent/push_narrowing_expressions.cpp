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

#include "transparent/push_narrowing_expressions.hpp"

#include <duckdb/common/types.hpp>
#include <duckdb/common/types/decimal.hpp>
#include <duckdb/optimizer/column_binding_replacer.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <optional>
#include <vector>

namespace sirius::transparent {

namespace {

/// Whether @p expr is DECIMAL `+`, `-`, `*` over columns, constants and widening DECIMAL casts.
bool movable(duckdb::Expression const& expr)
{
  switch (expr.GetExpressionClass()) {
    case duckdb::ExpressionClass::BOUND_COLUMN_REF:
      return expr.Cast<duckdb::BoundColumnRefExpression>().Depth() == 0;
    case duckdb::ExpressionClass::BOUND_CONSTANT: return true;
    case duckdb::ExpressionClass::BOUND_FUNCTION: {
      auto const& function = expr.Cast<duckdb::BoundFunctionExpression>();
      if (duckdb::BoundCastExpression::IsCast(expr)) {
        auto const& child  = duckdb::BoundCastExpression::Child(function);
        auto const& source = child.GetReturnType();
        auto const& target = function.GetReturnType();
        if (duckdb::BoundCastExpression::IsTryCast(function) ||
            source.id() != duckdb::LogicalTypeId::DECIMAL ||
            target.id() != duckdb::LogicalTypeId::DECIMAL) {
          return false;
        }
        auto const source_scale = duckdb::DecimalType::GetScale(source);
        auto const target_scale = duckdb::DecimalType::GetScale(target);
        return target_scale >= source_scale &&
               duckdb::DecimalType::GetWidth(target) - target_scale >=
                 duckdb::DecimalType::GetWidth(source) - source_scale &&
               movable(child);
      }
      auto const& name = function.Function().GetName().GetIdentifierName();
      if ((name != "+" && name != "-" && name != "*") ||
          function.GetReturnType().id() != duckdb::LogicalTypeId::DECIMAL) {
        return false;
      }
      return std::all_of(function.GetChildren().begin(),
                         function.GetChildren().end(),
                         [](auto const& child) { return movable(*child); });
    }
    default: return false;
  }
}

std::vector<duckdb::ColumnBinding> columns_of(duckdb::Expression& expr)
{
  std::vector<duckdb::ColumnBinding> columns;
  duckdb::ExpressionIterator::VisitExpression<duckdb::BoundColumnRefExpression>(
    expr, [&](duckdb::BoundColumnRefExpression const& column) {
      if (std::find(columns.begin(), columns.end(), column.Binding()) == columns.end()) {
        columns.push_back(column.Binding());
      }
    });
  return columns;
}

class reference_counter : public duckdb::LogicalOperatorVisitor {
 public:
  explicit reference_counter(duckdb::ColumnBinding binding) : _binding(binding) {}

  void count_in(duckdb::LogicalOperator& op)
  {
    EnumerateExpressions(
      op, [this](duckdb::unique_ptr<duckdb::Expression>* expr) { VisitExpression(expr); });
  }

  void count_in(duckdb::unique_ptr<duckdb::Expression>& expr) { VisitExpression(&expr); }

  [[nodiscard]] std::size_t count() const { return _count; }

 protected:
  duckdb::unique_ptr<duckdb::Expression> VisitReplace(
    duckdb::BoundColumnRefExpression& expr, duckdb::unique_ptr<duckdb::Expression>*) override
  {
    if (expr.Binding() == _binding) { ++_count; }
    return nullptr;
  }

 private:
  duckdb::ColumnBinding _binding;
  std::size_t _count = 0;
};

std::size_t width_of(duckdb::LogicalType const& type)
{
  auto const physical = type.InternalType();
  // A variable-size column counts as wide: dropping one always narrows the row.
  return duckdb::TypeIsConstantSize(physical) ? duckdb::GetTypeIdSize(physical) : 16;
}

std::size_t push_below(duckdb::LogicalProjection& projection, duckdb::Binder& binder)
{
  if (projection.children.size() != 1 ||
      projection.children[0]->type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    return 0;
  }
  auto& join = projection.children[0]->Cast<duckdb::LogicalComparisonJoin>();
  if (join.join_type != duckdb::JoinType::INNER) { return 0; }

  std::size_t moved = 0;
  for (std::size_t side = 0; side < 2; ++side) {
    auto& child         = join.children[side];
    auto const bindings = child->GetColumnBindings();
    child->ResolveOperatorTypes();
    auto const types = child->types;
    auto const position_of =
      [&](duckdb::ColumnBinding const& binding) -> std::optional<std::size_t> {
      auto const it = std::find(bindings.begin(), bindings.end(), binding);
      if (it == bindings.end()) { return std::nullopt; }
      return static_cast<std::size_t>(it - bindings.begin());
    };

    // The child columns that stop being needed, and the projection's expressions that move.
    std::vector<bool> dropped(bindings.size(), false);
    std::vector<std::size_t> moving;
    for (std::size_t k = 0; k < projection.expressions.size(); ++k) {
      auto& expr = projection.expressions[k];
      if (expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_COLUMN_REF ||
          expr->IsVolatile() || !movable(*expr)) {
        continue;
      }
      auto const inputs = columns_of(*expr);
      if (inputs.empty()) { continue; }
      std::size_t input_width = 0;
      bool only_here          = true;
      for (auto const& input : inputs) {
        auto const position = position_of(input);
        if (!position || dropped[*position]) {
          only_here = false;
          break;
        }
        reference_counter in_expr(input);
        in_expr.count_in(expr);
        reference_counter elsewhere(input);
        elsewhere.count_in(projection);
        elsewhere.count_in(join);
        if (elsewhere.count() != in_expr.count()) {
          only_here = false;
          break;
        }
        input_width += width_of(types[*position]);
      }
      if (!only_here || width_of(expr->GetReturnType()) >= input_width) { continue; }
      for (auto const& input : inputs) {
        dropped[*position_of(input)] = true;
      }
      moving.push_back(k);
    }
    if (moving.empty()) { continue; }

    auto const table_index = binder.GenerateTableIndex();
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
    duckdb::ColumnBindingReplacer replacer;
    std::vector<std::optional<std::size_t>> kept_at(bindings.size());
    for (std::size_t i = 0; i < bindings.size(); ++i) {
      if (dropped[i]) { continue; }
      kept_at[i] = select_list.size();
      replacer.replacement_bindings.emplace_back(
        bindings[i],
        duckdb::ColumnBinding(table_index, duckdb::ProjectionIndex(select_list.size())));
      select_list.push_back(
        duckdb::make_uniq<duckdb::BoundColumnRefExpression>(types[i], bindings[i]));
    }
    auto& map = side == 0 ? join.left_projection_map : join.right_projection_map;
    duckdb::vector<duckdb::ProjectionIndex> new_map;
    for (auto const& visible : map) {
      if (auto const kept = kept_at[static_cast<duckdb::idx_t>(visible)]) {
        new_map.emplace_back(*kept);
      }
    }
    for (auto const k : moving) {
      auto& expr       = projection.expressions[k];
      auto const alias = expr->GetAlias();
      auto const type  = expr->GetReturnType();
      duckdb::ColumnBinding const output(table_index, duckdb::ProjectionIndex(select_list.size()));
      if (!map.empty()) { new_map.emplace_back(select_list.size()); }
      select_list.push_back(std::move(expr));
      expr = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(type, output);
      expr->SetAlias(alias);
    }
    if (!map.empty()) { map = std::move(new_map); }

    auto lower = duckdb::make_uniq<duckdb::LogicalProjection>(table_index, std::move(select_list));
    if (child->has_estimated_cardinality) {
      lower->SetEstimatedCardinality(child->estimated_cardinality);
    }
    lower->children.push_back(std::move(child));
    child = std::move(lower);

    // The projection redefines every binding, so nothing above it references the child's.
    replacer.stop_operator = child.get();
    replacer.VisitOperator(projection);
    moved += moving.size();
  }
  if (moved > 0) { projection.ResolveOperatorTypes(); }
  return moved;
}

}  // namespace

std::size_t push_narrowing_expressions(duckdb::LogicalOperator& plan, duckdb::Binder& binder)
{
  std::size_t moved = 0;
  for (auto& child : plan.children) {
    moved += push_narrowing_expressions(*child, binder);
  }
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    auto const here = push_below(plan.Cast<duckdb::LogicalProjection>(), binder);
    if (here > 0) {
      SIRIUS_LOG_INFO("Moved {} projection expression(s) below an inner join", here);
    }
    moved += here;
  }
  return moved;
}

}  // namespace sirius::transparent
