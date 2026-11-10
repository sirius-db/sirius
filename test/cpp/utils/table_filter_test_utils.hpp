#pragma once

#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/dynamic_filter.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/filter/table_filter_functions.hpp>
#include <duckdb/planner/table_filter_set.hpp>

namespace sirius::test {

inline duckdb::unique_ptr<duckdb::ExpressionFilter> constant_filter(
  duckdb::ExpressionType comparison, duckdb::Value constant, duckdb::LogicalType const& column_type)
{
  auto ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(column_type, 0);
  return duckdb::make_uniq<duckdb::ExpressionFilter>(duckdb::BoundComparisonExpression::Create(
    comparison,
    std::move(ref),
    duckdb::make_uniq<duckdb::BoundConstantExpression>(std::move(constant))));
}

inline duckdb::unique_ptr<duckdb::ExpressionFilter> constant_filter(
  duckdb::ExpressionType comparison, duckdb::Value constant)
{
  auto type = constant.type();
  return constant_filter(comparison, std::move(constant), type);
}

inline duckdb::unique_ptr<duckdb::ExpressionFilter> null_filter(duckdb::LogicalType const& type,
                                                                bool is_not_null = false)
{
  return duckdb::make_uniq<duckdb::ExpressionFilter>(
    duckdb::ExpressionFilter::CreateNullCheckExpression(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, 0),
      is_not_null ? duckdb::ExpressionType::OPERATOR_IS_NOT_NULL
                  : duckdb::ExpressionType::OPERATOR_IS_NULL));
}

inline duckdb::unique_ptr<duckdb::ExpressionFilter> in_filter(duckdb::vector<duckdb::Value> values)
{
  auto ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(values.at(0).type(), 0);
  return duckdb::make_uniq<duckdb::ExpressionFilter>(
    duckdb::ExpressionFilter::CreateInExpression(std::move(ref), std::move(values)));
}

inline duckdb::unique_ptr<duckdb::ExpressionFilter> conjunction_filter(
  duckdb::ExpressionType type,
  duckdb::unique_ptr<duckdb::TableFilter> left,
  duckdb::unique_ptr<duckdb::TableFilter> right)
{
  auto expr = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(type);
  expr->GetChildrenMutable().push_back(std::move(left->Cast<duckdb::ExpressionFilter>().expr));
  expr->GetChildrenMutable().push_back(std::move(right->Cast<duckdb::ExpressionFilter>().expr));
  return duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(expr));
}

inline duckdb::unique_ptr<duckdb::ExpressionFilter> optional_filter(
  duckdb::unique_ptr<duckdb::TableFilter> child, duckdb::LogicalType const& type)
{
  return duckdb::make_uniq<duckdb::ExpressionFilter>(duckdb::CreateOptionalFilterExpression(
    std::move(child->Cast<duckdb::ExpressionFilter>().expr), type));
}

inline duckdb::unique_ptr<duckdb::ExpressionFilter> dynamic_filter(duckdb::LogicalType const& type)
{
  return duckdb::make_uniq<duckdb::ExpressionFilter>(duckdb::CreateDynamicFilterExpression(
    duckdb::make_shared_ptr<duckdb::DynamicFilterData>(
      duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value(type)),
    type));
}

}  // namespace sirius::test
