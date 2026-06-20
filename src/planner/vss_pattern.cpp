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

#include "vss/vss_pattern.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"

#include <string>

namespace sirius::vss {
namespace {

std::optional<cuvs::distance::DistanceType> metric_for_function(std::string const& name)
{
  if (name == "array_distance") { return cuvs::distance::DistanceType::L2SqrtExpanded; }
  if (name == "array_cosine_distance") { return cuvs::distance::DistanceType::CosineExpanded; }
  return std::nullopt;
}

std::optional<std::vector<float>> extract_float_array(duckdb::Value const& value)
{
  auto const& type = value.type();
  if (type.id() != duckdb::LogicalTypeId::ARRAY) { return std::nullopt; }
  if (value.IsNull()) { return std::nullopt; }
  if (duckdb::ArrayType::GetChildType(type).id() != duckdb::LogicalTypeId::FLOAT) {
    return std::nullopt;
  }

  auto const& children = duckdb::ArrayValue::GetChildren(value);
  std::vector<float> out;
  out.reserve(children.size());
  for (auto const& child : children) {
    if (child.IsNull()) { return std::nullopt; }
    out.push_back(child.GetValue<float>());
  }
  return out;
}

struct parsed_distance {
  cuvs::distance::DistanceType metric;
  cudf::size_type vector_column_index;  // index into the projection's input
  std::vector<float> query;
  int64_t dim;
};

std::optional<parsed_distance> parse_distance_expression(duckdb::Expression const& expr)
{
  if (expr.expression_class != duckdb::ExpressionClass::BOUND_FUNCTION) { return std::nullopt; }
  auto const& fn    = expr.Cast<duckdb::BoundFunctionExpression>();
  auto const metric = metric_for_function(fn.function.name);
  if (!metric.has_value()) { return std::nullopt; }
  if (fn.children.size() != 2) { return std::nullopt; }

  duckdb::BoundReferenceExpression const* ref     = nullptr;  // column reference
  duckdb::BoundConstantExpression const* constant = nullptr;  // query vector literal
  for (auto const& child : fn.children) {
    if (child->expression_class == duckdb::ExpressionClass::BOUND_REF) {
      ref = &child->Cast<duckdb::BoundReferenceExpression>();
    } else if (child->expression_class == duckdb::ExpressionClass::BOUND_CONSTANT) {
      constant = &child->Cast<duckdb::BoundConstantExpression>();
    }
  }

  if (ref == nullptr || constant == nullptr) { return std::nullopt; }
  if (ref->return_type.id() != duckdb::LogicalTypeId::ARRAY) { return std::nullopt; }
  if (duckdb::ArrayType::GetChildType(ref->return_type).id() != duckdb::LogicalTypeId::FLOAT) {
    return std::nullopt;
  }

  auto query = extract_float_array(constant->value);
  if (!query.has_value() || query->empty()) { return std::nullopt; }

  auto const dim = static_cast<int64_t>(query->size());
  if (static_cast<int64_t>(duckdb::ArrayType::GetSize(ref->return_type)) != dim) {
    return std::nullopt;
  }

  parsed_distance parsed;
  parsed.metric              = *metric;
  parsed.vector_column_index = static_cast<cudf::size_type>(ref->index);
  parsed.query               = std::move(*query);
  parsed.dim                 = dim;
  return parsed;
}

}  // namespace

std::optional<vss_top_k_pattern> match_vss_top_n(duckdb::LogicalTopN const& op)
{
  if (op.orders.size() != 1) { return std::nullopt; }
  auto const& order = op.orders[0];

  if (order.type != duckdb::OrderType::ASCENDING) { return std::nullopt; }
  if (order.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }
  auto const distance_index = order.expression->Cast<duckdb::BoundReferenceExpression>().index;

  if (op.children.size() != 1) { return std::nullopt; }
  if (op.children[0]->type != duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    return std::nullopt;
  }
  auto const& proj  = op.children[0]->Cast<duckdb::LogicalProjection>();
  auto const& exprs = proj.expressions;
  if (distance_index >= exprs.size()) { return std::nullopt; }

  auto parsed = parse_distance_expression(*exprs[distance_index]);
  if (!parsed.has_value()) { return std::nullopt; }

  std::vector<vss_output_column> output_columns;
  output_columns.reserve(exprs.size());
  for (duckdb::idx_t i = 0; i < exprs.size(); ++i) {
    if (i == distance_index) {
      output_columns.push_back({vss_output_column::kind::distance, -1});
      continue;
    }
    auto const& e = *exprs[i];
    if (e.expression_class != duckdb::ExpressionClass::BOUND_REF) { return std::nullopt; }
    output_columns.push_back(
      {vss_output_column::kind::gather_input,
       static_cast<cudf::size_type>(e.Cast<duckdb::BoundReferenceExpression>().index)});
  }

  vss_top_k_pattern pattern;
  pattern.vector_column_index   = parsed->vector_column_index;
  pattern.query                 = std::move(parsed->query);
  pattern.dim                   = parsed->dim;
  pattern.metric                = parsed->metric;
  pattern.output_columns        = std::move(output_columns);
  pattern.distance_output_index = static_cast<cudf::size_type>(distance_index);
  return pattern;
}

}  // namespace sirius::vss
