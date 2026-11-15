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

#include "transparent/narrow_outer_join_counts.hpp"

#include <duckdb/common/types/value.hpp>
#include <duckdb/optimizer/column_binding_replacer.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <optional>
#include <vector>

namespace sirius::transparent {

namespace {

class binding_reference_counter : public duckdb::LogicalOperatorVisitor {
 public:
  explicit binding_reference_counter(duckdb::ColumnBinding binding) : _binding(binding) {}

  void count_in(duckdb::LogicalOperator& op)
  {
    EnumerateExpressions(op, [this](duckdb::unique_ptr<duckdb::Expression>* expr) {
      VisitExpression(expr);
    });
  }

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

/// The column a plain COUNT(col) reads, when narrowing it to a flag would save bytes.
std::optional<duckdb::ColumnBinding> counted_column(duckdb::Expression const& expr)
{
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
    return std::nullopt;
  }
  auto const& aggr = expr.Cast<duckdb::BoundAggregateExpression>();
  if (aggr.Function().GetName().GetIdentifierName() != "count" ||
      aggr.GetChildren().size() != 1 || aggr.IsDistinct() || aggr.GetFilter() ||
      aggr.GetOrderBys()) {
    return std::nullopt;
  }
  auto const& arg = *aggr.GetChildren()[0];
  if (arg.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
    return std::nullopt;
  }
  auto const& type = arg.GetReturnType();
  if (type.id() == duckdb::LogicalTypeId::BOOLEAN ||
      (duckdb::TypeIsConstantSize(type.InternalType()) &&
       duckdb::GetTypeIdSize(type.InternalType()) <= 1)) {
    return std::nullopt;
  }
  return arg.Cast<duckdb::BoundColumnRefExpression>().Binding();
}

duckdb::unique_ptr<duckdb::Expression> present_flag(duckdb::LogicalType const& type,
                                                 duckdb::ColumnBinding binding)
{
  auto present = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
    duckdb::ExpressionType::OPERATOR_IS_NOT_NULL, duckdb::LogicalType::BOOLEAN);
  present->GetChildrenMutable().push_back(
    duckdb::make_uniq<duckdb::BoundColumnRefExpression>(type, binding));
  return duckdb::make_uniq<duckdb::BoundCaseExpression>(
    std::move(present),
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::BOOLEAN(true)),
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value(duckdb::LogicalType::BOOLEAN)));
}

std::size_t narrow_counts_below(duckdb::LogicalAggregate& agg, duckdb::Binder& binder)
{
  if (agg.children.size() != 1 ||
      agg.children[0]->type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    return 0;
  }
  auto& join = agg.children[0]->Cast<duckdb::LogicalComparisonJoin>();
  if (join.join_type != duckdb::JoinType::LEFT && join.join_type != duckdb::JoinType::RIGHT) {
    return 0;
  }
  auto& nulled = join.children[join.join_type == duckdb::JoinType::LEFT ? 1 : 0];
  auto const bindings = nulled->GetColumnBindings();
  nulled->ResolveOperatorTypes();
  auto const types = nulled->types;

  std::vector<bool> narrow(bindings.size(), false);
  std::size_t narrowed = 0;
  for (auto const& expr : agg.expressions) {
    auto const column = counted_column(*expr);
    if (!column) { continue; }
    auto const it = std::find(bindings.begin(), bindings.end(), *column);
    if (it == bindings.end()) { continue; }
    binding_reference_counter refs(*column);
    refs.count_in(agg);
    refs.count_in(join);
    if (refs.count() != 1) { continue; }
    narrow[static_cast<std::size_t>(it - bindings.begin())] = true;
    ++narrowed;
  }
  if (narrowed == 0) { return 0; }

  // The projection keeps the child's column order, so the join's projection maps stay valid.
  auto const table_index = binder.GenerateTableIndex();
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
  duckdb::ColumnBindingReplacer replacer;
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    duckdb::ColumnBinding const replacement(table_index, duckdb::ProjectionIndex(i));
    if (narrow[i]) {
      select_list.push_back(present_flag(types[i], bindings[i]));
      replacer.replacement_bindings.emplace_back(
        bindings[i], replacement, duckdb::LogicalType::BOOLEAN);
    } else {
      select_list.push_back(
        duckdb::make_uniq<duckdb::BoundColumnRefExpression>(types[i], bindings[i]));
      replacer.replacement_bindings.emplace_back(bindings[i], replacement);
    }
  }
  auto projection = duckdb::make_uniq<duckdb::LogicalProjection>(table_index, std::move(select_list));
  if (nulled->has_estimated_cardinality) {
    projection->SetEstimatedCardinality(nulled->estimated_cardinality);
  }
  projection->children.push_back(std::move(nulled));
  nulled = std::move(projection);

  // The aggregate redefines every binding, so nothing above it can reference the replaced ones.
  replacer.stop_operator = nulled.get();
  replacer.VisitOperator(agg);
  agg.ResolveOperatorTypes();
  return narrowed;
}

}  // namespace

std::size_t narrow_outer_join_counts(duckdb::LogicalOperator& plan, duckdb::Binder& binder)
{
  std::size_t narrowed = 0;
  for (auto& child : plan.children) {
    narrowed += narrow_outer_join_counts(*child, binder);
  }
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    auto const here = narrow_counts_below(plan.Cast<duckdb::LogicalAggregate>(), binder);
    if (here > 0) {
      SIRIUS_LOG_DEBUG("Narrowed {} outer-join COUNT argument(s) to presence flags", here);
    }
    narrowed += here;
  }
  return narrowed;
}

}  // namespace sirius::transparent
