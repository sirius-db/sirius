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

#include "expression/ast/from_duckdb.hpp"

// sirius
#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/between.hpp"
#include "expression/ast/case_expr.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/coalesce.hpp"
#include "expression/ast/comparison.hpp"
#include "expression/ast/conjunction.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/in_list.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/unary_op.hpp"
#include "expression/function_id.hpp"
#include "expression/join_condition.hpp"  // sirius::comparison_type, sirius::from_duckdb(ExpressionType)
#include "expression/value.hpp"           // sirius::from_duckdb(Value const&, logical_type const&)
#include "helper/type_conversions.hpp"  // sirius::from_duckdb(LogicalType const&)

// duckdb
#include <duckdb/common/assert.hpp>
#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace sirius::ast {

namespace {

// Translate a sequence of DuckDB child expressions (skipping the first @p start
// entries), short-circuiting to std::nullopt the moment any child is
// unsupported (i.e. from_duckdb returns null). Callers treat nullopt as "this
// whole expression is unsupported" and propagate a null node upward.
template <class Children>
std::optional<std::vector<std::unique_ptr<node>>> translate_children(Children const& children,
                                                                     std::size_t start = 0)
{
  std::vector<std::unique_ptr<node>> out;
  out.reserve(children.size() > start ? children.size() - start : 0);
  for (std::size_t i = start; i < children.size(); ++i) {
    auto translated = from_duckdb(*children[i]);
    if (!translated) { return std::nullopt; }
    out.push_back(std::move(translated));
  }
  return out;
}

std::unique_ptr<node> translate_reference(duckdb::BoundReferenceExpression const& expr)
{
  return std::make_unique<node>(
    reference{static_cast<uint32_t>(expr.Index()), sirius::from_duckdb(expr.GetReturnType())});
}

std::unique_ptr<node> translate_constant(duckdb::BoundConstantExpression const& expr)
{
  auto return_type = sirius::from_duckdb(expr.GetReturnType());
  auto payload     = sirius::from_duckdb(expr.GetValue(), return_type);
  return std::make_unique<node>(constant{std::move(payload), std::move(return_type)});
}

// Comparisons, casts and BETWEENs are all BoundFunctionExpression upstream,
// discriminated by ExpressionType; the Bound*Expression names are now only
// static-helper namespaces over that representation.
std::unique_ptr<node> translate_comparison(duckdb::BoundFunctionExpression const& expr)
{
  // Filter the BoundComparisonExpression ExpressionType down to the set that the
  // Sirius comparison node carries (sirius::comparison_type). The set mirrors
  // sirius::from_duckdb(duckdb::ExpressionType) in src/expression/join_condition.cpp.
  switch (expr.GetExpressionType()) {
    case duckdb::ExpressionType::COMPARE_EQUAL:
    case duckdb::ExpressionType::COMPARE_NOTEQUAL:
    case duckdb::ExpressionType::COMPARE_LESSTHAN:
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
    case duckdb::ExpressionType::COMPARE_GREATERTHAN:
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
    case duckdb::ExpressionType::COMPARE_DISTINCT_FROM:
    case duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM: break;
    default: return nullptr;
  }
  auto left  = from_duckdb(duckdb::BoundComparisonExpression::Left(expr));
  auto right = from_duckdb(duckdb::BoundComparisonExpression::Right(expr));
  if (!left || !right) { return nullptr; }
  return std::make_unique<node>(comparison{
    /*op=*/sirius::from_duckdb(expr.GetExpressionType()),
    /*left=*/std::move(left),
    /*right=*/std::move(right),
  });
}

std::unique_ptr<node> translate_conjunction(duckdb::BoundConjunctionExpression const& expr)
{
  conjunction::kind op{conjunction::kind::invalid};
  switch (expr.GetExpressionType()) {
    case duckdb::ExpressionType::CONJUNCTION_AND: op = conjunction::kind::op_and; break;
    case duckdb::ExpressionType::CONJUNCTION_OR: op = conjunction::kind::op_or; break;
    default: return nullptr;
  }
  auto children = translate_children(expr.GetChildren());
  if (!children) { return nullptr; }
  return std::make_unique<node>(conjunction{op, std::move(*children)});
}

std::unique_ptr<node> translate_between(duckdb::BoundFunctionExpression const& expr)
{
  auto input = from_duckdb(duckdb::BoundBetweenExpression::Input(expr));
  auto lower = from_duckdb(duckdb::BoundBetweenExpression::LowerBound(expr));
  auto upper = from_duckdb(duckdb::BoundBetweenExpression::UpperBound(expr));
  if (!input || !lower || !upper) { return nullptr; }
  return std::make_unique<node>(between{
    /*input=*/std::move(input),
    /*lower=*/std::move(lower),
    /*upper=*/std::move(upper),
    /*lower_inclusive=*/duckdb::BoundBetweenExpression::LowerInclusive(expr),
    /*upper_inclusive=*/duckdb::BoundBetweenExpression::UpperInclusive(expr),
  });
}

std::unique_ptr<node> translate_case(duckdb::BoundCaseExpression const& expr)
{
  std::vector<case_expr::when_then> cases;
  cases.reserve(expr.CaseChecks().size());
  for (auto const& check : expr.CaseChecks()) {
    auto when_node = from_duckdb(*check.when_expr);
    auto then_node = from_duckdb(*check.then_expr);
    if (!when_node || !then_node) { return nullptr; }
    cases.push_back(case_expr::when_then{std::move(when_node), std::move(then_node)});
  }
  auto else_node = from_duckdb(expr.Else());
  if (!else_node) { return nullptr; }
  return std::make_unique<node>(
    case_expr{std::move(cases), std::move(else_node), sirius::from_duckdb(expr.GetReturnType())});
}

std::unique_ptr<node> translate_cast(duckdb::BoundFunctionExpression const& expr)
{
  auto child = from_duckdb(duckdb::BoundCastExpression::Child(expr));
  if (!child) { return nullptr; }
  auto target_type = sirius::from_duckdb(expr.GetReturnType());
  return std::make_unique<node>(cast{
    /*child=*/std::move(child),
    /*target_type=*/std::move(target_type),
    /*try_cast=*/duckdb::BoundCastExpression::IsTryCast(expr),
  });
}

std::unique_ptr<node> translate_function(duckdb::BoundFunctionExpression const& expr)
{
  auto func_id_opt =
    sirius::from_duckdb_function_name(expr.Function().GetName().GetIdentifierName());
  if (!func_id_opt.has_value()) { return nullptr; }
  auto arguments = translate_children(expr.GetChildren());
  if (!arguments) { return nullptr; }
  auto return_type = sirius::from_duckdb(expr.GetReturnType());
  return std::make_unique<node>(
    function_call{*func_id_opt, std::move(*arguments), std::move(return_type)});
}

std::unique_ptr<node> translate_aggregate(duckdb::BoundAggregateExpression const& aggr)
{
  auto agg_id_opt =
    sirius::from_duckdb_aggregate_name(aggr.Function().GetName().GetIdentifierName());
  if (!agg_id_opt.has_value()) { return nullptr; }
  auto arguments = translate_children(aggr.GetChildren());
  if (!arguments) { return nullptr; }
  auto return_type = sirius::from_duckdb(aggr.GetReturnType());
  return std::make_unique<node>(
    aggregate{*agg_id_opt, std::move(*arguments), std::move(return_type), aggr.IsDistinct()});
}

std::unique_ptr<node> translate_operator(duckdb::BoundOperatorExpression const& expr)
{
  auto const op_type = expr.GetExpressionType();

  // Unary operators share a single AST node; only the kind tag differs.
  unary_op::kind unary_kind{unary_op::kind::invalid};
  switch (op_type) {
    case duckdb::ExpressionType::OPERATOR_NOT: unary_kind = unary_op::kind::op_not; break;
    case duckdb::ExpressionType::OPERATOR_IS_NULL: unary_kind = unary_op::kind::op_is_null; break;
    case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL:
      unary_kind = unary_op::kind::op_is_not_null;
      break;
    case duckdb::ExpressionType::OPERATOR_TRY: unary_kind = unary_op::kind::op_try; break;
    default: break;
  }
  if (unary_kind != unary_op::kind::invalid) {
    D_ASSERT(!expr.GetChildren().empty());
    auto child = from_duckdb(*expr.GetChildren()[0]);
    if (!child) { return nullptr; }
    return std::make_unique<node>(unary_op{unary_kind, std::move(child)});
  }

  if (op_type == duckdb::ExpressionType::OPERATOR_COALESCE) {
    auto children = translate_children(expr.GetChildren());
    if (!children) { return nullptr; }
    return std::make_unique<node>(
      coalesce{std::move(*children), sirius::from_duckdb(expr.GetReturnType())});
  }

  if (op_type == duckdb::ExpressionType::COMPARE_IN ||
      op_type == duckdb::ExpressionType::COMPARE_NOT_IN) {
    D_ASSERT(!expr.GetChildren().empty());
    auto probe = from_duckdb(*expr.GetChildren()[0]);
    if (!probe) { return nullptr; }
    auto values = translate_children(expr.GetChildren(), /*start=*/1);
    if (!values) { return nullptr; }
    return std::make_unique<node>(in_list{
      /*probe=*/std::move(probe),
      /*values=*/std::move(*values),
      /*negated=*/op_type == duckdb::ExpressionType::COMPARE_NOT_IN,
    });
  }

  return nullptr;
}

}  // namespace

std::unique_ptr<node> from_duckdb(duckdb::Expression const& expr)
{
  switch (expr.GetExpressionClass()) {
    case duckdb::ExpressionClass::BOUND_AGGREGATE:
      return translate_aggregate(expr.Cast<duckdb::BoundAggregateExpression>());
    case duckdb::ExpressionClass::BOUND_CASE:
      return translate_case(expr.Cast<duckdb::BoundCaseExpression>());
    case duckdb::ExpressionClass::BOUND_CONJUNCTION:
      return translate_conjunction(expr.Cast<duckdb::BoundConjunctionExpression>());
    case duckdb::ExpressionClass::BOUND_CONSTANT:
      return translate_constant(expr.Cast<duckdb::BoundConstantExpression>());
    case duckdb::ExpressionClass::BOUND_FUNCTION: {
      // Casts, BETWEENs and comparisons are function expressions upstream;
      // split them back out before falling through to a real function call.
      auto const& function_expr = expr.Cast<duckdb::BoundFunctionExpression>();
      if (duckdb::BoundCastExpression::IsCast(expr)) { return translate_cast(function_expr); }
      if (expr.GetExpressionType() == duckdb::ExpressionType::COMPARE_BETWEEN) {
        return translate_between(function_expr);
      }
      if (duckdb::BoundComparisonExpression::IsComparison(expr)) {
        return translate_comparison(function_expr);
      }
      return translate_function(function_expr);
    }
    case duckdb::ExpressionClass::BOUND_OPERATOR:
      return translate_operator(expr.Cast<duckdb::BoundOperatorExpression>());
    case duckdb::ExpressionClass::BOUND_PARAMETER: return nullptr;
    case duckdb::ExpressionClass::BOUND_REF:
      return translate_reference(expr.Cast<duckdb::BoundReferenceExpression>());
    default:
      throw duckdb::InternalException("[sirius::ast::from_duckdb] Unknown ExpressionClass: {}",
                                      expr.GetExpressionClass());
  }
}

}  // namespace sirius::ast
