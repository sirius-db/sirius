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

#include <duckdb/common/value_operations/value_operations.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <op/scan/parquet_source_filter.hpp>

namespace sirius::op::scan {
namespace {
source_filter_result evaluate_source_comparison(duckdb::ExpressionType comparison_type,
                                                duckdb::Value const& value,
                                                duckdb::Value const& constant)
{
  using result = source_filter_result;
  switch (comparison_type) {
    case duckdb::ExpressionType::COMPARE_EQUAL:
    case duckdb::ExpressionType::COMPARE_NOTEQUAL:
    case duckdb::ExpressionType::COMPARE_LESSTHAN:
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
    case duckdb::ExpressionType::COMPARE_GREATERTHAN:
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: break;
    default: return result::UNKNOWN;
  }
  if (constant.IsNull() || value.IsNull()) { return result::SQL_NULL; }
  if (constant.type() != value.type()) { return result::UNKNOWN; }
  bool matches = false;
  switch (comparison_type) {
    case duckdb::ExpressionType::COMPARE_EQUAL:
      matches = duckdb::ValueOperations::Equals(value, constant);
      break;
    case duckdb::ExpressionType::COMPARE_NOTEQUAL:
      matches = duckdb::ValueOperations::NotEquals(value, constant);
      break;
    case duckdb::ExpressionType::COMPARE_LESSTHAN:
      matches = duckdb::ValueOperations::LessThan(value, constant);
      break;
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
      matches = duckdb::ValueOperations::LessThanEquals(value, constant);
      break;
    case duckdb::ExpressionType::COMPARE_GREATERTHAN:
      matches = duckdb::ValueOperations::GreaterThan(value, constant);
      break;
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      matches = duckdb::ValueOperations::GreaterThanEquals(value, constant);
      break;
    default: return result::UNKNOWN;
  }
  return matches ? result::MATCH : result::NO_MATCH;
}

source_filter_result evaluate_source_in(duckdb::vector<duckdb::Value> const& values,
                                        duckdb::Value const& value)
{
  using result = source_filter_result;
  if (value.IsNull()) { return result::SQL_NULL; }
  bool has_null = false;
  bool unknown  = false;
  for (auto const& candidate : values) {
    if (candidate.IsNull()) {
      has_null = true;
      continue;
    }
    if (candidate.type() != value.type()) {
      unknown = true;
      continue;
    }
    if (duckdb::ValueOperations::Equals(candidate, value)) { return result::MATCH; }
  }
  return unknown ? result::UNKNOWN : has_null ? result::SQL_NULL : result::NO_MATCH;
}

source_filter_result evaluate_source_expression(duckdb::Expression const& expression,
                                                duckdb::Value const& value)
{
  using result    = source_filter_result;
  using type      = duckdb::ExpressionType;
  auto const kind = expression.GetExpressionType();
  if (kind == type::CONJUNCTION_AND || kind == type::CONJUNCTION_OR) {
    bool unknown = false, null = false;
    for (auto const& child : expression.Cast<duckdb::BoundConjunctionExpression>().GetChildren()) {
      auto r = evaluate_source_expression(*child, value);
      if (kind == type::CONJUNCTION_AND && r == result::NO_MATCH) return r;
      if (kind == type::CONJUNCTION_OR && r == result::MATCH) return r;
      unknown |= r == result::UNKNOWN;
      null |= r == result::SQL_NULL;
    }
    return unknown                         ? result::UNKNOWN
           : null                          ? result::SQL_NULL
           : kind == type::CONJUNCTION_AND ? result::MATCH
                                           : result::NO_MATCH;
  }
  auto is_input = [](duckdb::Expression const& e) {
    return e.GetExpressionType() == type::BOUND_REF &&
           e.Cast<duckdb::BoundReferenceExpression>().Index() == 0;
  };
  if (expression.GetExpressionClass() == duckdb::ExpressionClass::BOUND_OPERATOR) {
    auto const& children = expression.Cast<duckdb::BoundOperatorExpression>().GetChildren();
    if (children.empty() || !is_input(*children.front())) return result::UNKNOWN;
    if (kind == type::OPERATOR_IS_NULL) return value.IsNull() ? result::MATCH : result::NO_MATCH;
    if (kind == type::OPERATOR_IS_NOT_NULL)
      return value.IsNull() ? result::NO_MATCH : result::MATCH;
    if (kind == type::COMPARE_IN) {
      duckdb::vector<duckdb::Value> values;
      for (size_t i = 1; i < children.size(); ++i) {
        if (children[i]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT)
          return result::UNKNOWN;
        values.push_back(children[i]->Cast<duckdb::BoundConstantExpression>().GetValue());
      }
      return evaluate_source_in(values, value);
    }
  }
  if (duckdb::BoundComparisonExpression::IsComparison(expression)) {
    auto const& comparison = expression.Cast<duckdb::BoundFunctionExpression>();
    auto const& left       = duckdb::BoundComparisonExpression::Left(comparison);
    auto const& right      = duckdb::BoundComparisonExpression::Right(comparison);
    if (is_input(left) && right.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT)
      return evaluate_source_comparison(
        kind, value, right.Cast<duckdb::BoundConstantExpression>().GetValue());
  }
  return result::UNKNOWN;
}
}  // namespace

source_filter_result evaluate_parquet_source_filter(duckdb::TableFilter const& filter,
                                                    duckdb::Value const& value)
{
  using result = source_filter_result;
  using type   = duckdb::TableFilterType;
  if (value.type() != duckdb::LogicalType::VARCHAR &&
      value.type() != duckdb::LogicalType::UBIGINT) {
    return result::UNKNOWN;
  }
  switch (filter.filter_type) {
    case type::EXPRESSION_FILTER:
      return evaluate_source_expression(*filter.Cast<duckdb::ExpressionFilter>().expr, value);
    case type::LEGACY_IS_NULL: return value.IsNull() ? result::MATCH : result::NO_MATCH;
    case type::LEGACY_IS_NOT_NULL: return value.IsNull() ? result::NO_MATCH : result::MATCH;
    case type::LEGACY_CONSTANT_COMPARISON: {
      auto const& comparison = filter.Cast<duckdb::LegacyConstantFilter>();
      return evaluate_source_comparison(comparison.comparison_type, value, comparison.constant);
    }
    case type::LEGACY_IN_FILTER:
      return evaluate_source_in(filter.Cast<duckdb::LegacyInFilter>().values, value);
    case type::LEGACY_CONJUNCTION_AND:
    case type::LEGACY_CONJUNCTION_OR: {
      bool const is_and    = filter.filter_type == type::LEGACY_CONJUNCTION_AND;
      auto const& children = is_and
                               ? filter.Cast<duckdb::LegacyConjunctionAndFilter>().child_filters
                               : filter.Cast<duckdb::LegacyConjunctionOrFilter>().child_filters;
      bool unknown         = false;
      bool has_null        = false;
      for (auto const& child : children) {
        auto const r = evaluate_parquet_source_filter(*child, value);
        if (is_and && r == result::NO_MATCH) { return result::NO_MATCH; }
        if (!is_and && r == result::MATCH) { return result::MATCH; }
        unknown |= r == result::UNKNOWN;
        has_null |= r == result::SQL_NULL;
      }
      if (unknown) { return result::UNKNOWN; }
      if (has_null) { return result::SQL_NULL; }
      return is_and ? result::MATCH : result::NO_MATCH;
    }
    default: return result::UNKNOWN;
  }
}

}  // namespace sirius::op::scan
