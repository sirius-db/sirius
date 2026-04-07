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

// sirius
#include <expression_executor/gpu_expression_translator.hpp>
#include <log/logging.hpp>

// cudf
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace sirius {

using expr_ref = std::reference_wrapper<cudf::ast::expression const>;

namespace {

/// @brief Convert a cudf::ast::ast_operator to a human-readable string.
std::string ast_operator_to_string(cudf::ast::ast_operator op)
{
  switch (op) {
    case cudf::ast::ast_operator::ADD: return "+";
    case cudf::ast::ast_operator::SUB: return "-";
    case cudf::ast::ast_operator::MUL: return "*";
    case cudf::ast::ast_operator::DIV: return "/";
    case cudf::ast::ast_operator::TRUE_DIV: return "true_div";
    case cudf::ast::ast_operator::FLOOR_DIV: return "floor_div";
    case cudf::ast::ast_operator::MOD: return "%";
    case cudf::ast::ast_operator::PYMOD: return "pymod";
    case cudf::ast::ast_operator::POW: return "**";
    case cudf::ast::ast_operator::EQUAL: return "==";
    case cudf::ast::ast_operator::NULL_EQUAL: return "null_eq";
    case cudf::ast::ast_operator::NOT_EQUAL: return "!=";
    case cudf::ast::ast_operator::LESS: return "<";
    case cudf::ast::ast_operator::GREATER: return ">";
    case cudf::ast::ast_operator::LESS_EQUAL: return "<=";
    case cudf::ast::ast_operator::GREATER_EQUAL: return ">=";
    case cudf::ast::ast_operator::BITWISE_AND: return "&";
    case cudf::ast::ast_operator::BITWISE_OR: return "|";
    case cudf::ast::ast_operator::BITWISE_XOR: return "^";
    case cudf::ast::ast_operator::LOGICAL_AND: return "AND";
    case cudf::ast::ast_operator::NULL_LOGICAL_AND: return "null_and";
    case cudf::ast::ast_operator::LOGICAL_OR: return "OR";
    case cudf::ast::ast_operator::NULL_LOGICAL_OR: return "null_or";
    case cudf::ast::ast_operator::IDENTITY: return "identity";
    case cudf::ast::ast_operator::IS_NULL: return "IS_NULL";
    case cudf::ast::ast_operator::SIN: return "sin";
    case cudf::ast::ast_operator::COS: return "cos";
    case cudf::ast::ast_operator::TAN: return "tan";
    case cudf::ast::ast_operator::ARCSIN: return "arcsin";
    case cudf::ast::ast_operator::ARCCOS: return "arccos";
    case cudf::ast::ast_operator::ARCTAN: return "arctan";
    case cudf::ast::ast_operator::SINH: return "sinh";
    case cudf::ast::ast_operator::COSH: return "cosh";
    case cudf::ast::ast_operator::TANH: return "tanh";
    case cudf::ast::ast_operator::ARCSINH: return "arcsinh";
    case cudf::ast::ast_operator::ARCCOSH: return "arccosh";
    case cudf::ast::ast_operator::ARCTANH: return "arctanh";
    case cudf::ast::ast_operator::EXP: return "exp";
    case cudf::ast::ast_operator::LOG: return "log";
    case cudf::ast::ast_operator::SQRT: return "sqrt";
    case cudf::ast::ast_operator::CBRT: return "cbrt";
    case cudf::ast::ast_operator::CEIL: return "ceil";
    case cudf::ast::ast_operator::FLOOR: return "floor";
    case cudf::ast::ast_operator::ABS: return "abs";
    case cudf::ast::ast_operator::RINT: return "rint";
    case cudf::ast::ast_operator::BIT_INVERT: return "~";
    case cudf::ast::ast_operator::NOT: return "NOT";
    case cudf::ast::ast_operator::CAST_TO_INT64: return "cast_to_int64";
    case cudf::ast::ast_operator::CAST_TO_UINT64: return "cast_to_uint64";
    case cudf::ast::ast_operator::CAST_TO_FLOAT64: return "cast_to_float64";
    default: return "unknown_op";
  }
}

/// @brief Recursively produce a human-readable string for a cuDF AST expression.
std::string expression_to_string(cudf::ast::expression const& expr)
{
  if (auto const* op = dynamic_cast<cudf::ast::operation const*>(&expr)) {
    auto const& operands = op->get_operands();
    auto op_str          = ast_operator_to_string(op->get_operator());
    if (operands.size() == 1) {
      return op_str + "(" + expression_to_string(operands[0].get()) + ")";
    } else if (operands.size() == 2) {
      return "(" + expression_to_string(operands[0].get()) + " " + op_str + " " +
             expression_to_string(operands[1].get()) + ")";
    }
    return op_str + "(?)";
  }
  if (auto const* col_ref = dynamic_cast<cudf::ast::column_reference const*>(&expr)) {
    std::string table_str =
      (col_ref->get_table_source() == cudf::ast::table_reference::LEFT) ? "L" : "R";
    return table_str + "[" + std::to_string(col_ref->get_column_index()) + "]";
  }
  if (auto const* col_name_ref = dynamic_cast<cudf::ast::column_name_reference const*>(&expr)) {
    return col_name_ref->get_column_name();
  }
  if (auto const* lit = dynamic_cast<cudf::ast::literal const*>(&expr)) {
    auto const& s   = lit->get_scalar();
    auto const dt   = s.type();
    auto const name = cudf::type_to_name(dt);

    // Try to read the scalar value for supported types.
    auto val_str = std::string("?");
    switch (dt.id()) {
      case cudf::type_id::INT8:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<int8_t> const&>(s).value());
        break;
      case cudf::type_id::INT16:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<int16_t> const&>(s).value());
        break;
      case cudf::type_id::INT32:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<int32_t> const&>(s).value());
        break;
      case cudf::type_id::INT64:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<int64_t> const&>(s).value());
        break;
      case cudf::type_id::UINT8:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<uint8_t> const&>(s).value());
        break;
      case cudf::type_id::UINT16:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<uint16_t> const&>(s).value());
        break;
      case cudf::type_id::UINT32:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<uint32_t> const&>(s).value());
        break;
      case cudf::type_id::UINT64:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<uint64_t> const&>(s).value());
        break;
      case cudf::type_id::FLOAT32:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<float> const&>(s).value());
        break;
      case cudf::type_id::FLOAT64:
        val_str = std::to_string(static_cast<cudf::numeric_scalar<double> const&>(s).value());
        break;
      case cudf::type_id::BOOL8:
        val_str = static_cast<cudf::numeric_scalar<bool> const&>(s).value() ? "true" : "false";
        break;
      case cudf::type_id::TIMESTAMP_DAYS:
        val_str = std::to_string(static_cast<cudf::timestamp_scalar<cudf::timestamp_D> const&>(s)
                                   .value()
                                   .time_since_epoch()
                                   .count());
        break;
      case cudf::type_id::TIMESTAMP_SECONDS:
        val_str = std::to_string(static_cast<cudf::timestamp_scalar<cudf::timestamp_s> const&>(s)
                                   .value()
                                   .time_since_epoch()
                                   .count());
        break;
      case cudf::type_id::TIMESTAMP_MILLISECONDS:
        val_str = std::to_string(static_cast<cudf::timestamp_scalar<cudf::timestamp_ms> const&>(s)
                                   .value()
                                   .time_since_epoch()
                                   .count());
        break;
      case cudf::type_id::TIMESTAMP_MICROSECONDS:
        val_str = std::to_string(static_cast<cudf::timestamp_scalar<cudf::timestamp_us> const&>(s)
                                   .value()
                                   .time_since_epoch()
                                   .count());
        break;
      case cudf::type_id::TIMESTAMP_NANOSECONDS:
        val_str = std::to_string(static_cast<cudf::timestamp_scalar<cudf::timestamp_ns> const&>(s)
                                   .value()
                                   .time_since_epoch()
                                   .count());
        break;
      case cudf::type_id::DECIMAL32: {
        auto const& fps = static_cast<cudf::fixed_point_scalar<numeric::decimal32> const&>(s);
        val_str = "rep=" + std::to_string(fps.value()) + ",scale=" + std::to_string(dt.scale());
        break;
      }
      case cudf::type_id::DECIMAL64: {
        auto const& fps = static_cast<cudf::fixed_point_scalar<numeric::decimal64> const&>(s);
        val_str = "rep=" + std::to_string(fps.value()) + ",scale=" + std::to_string(dt.scale());
        break;
      }
      case cudf::type_id::STRING: {
        val_str = "\"" + static_cast<cudf::string_scalar const&>(s).to_string() + "\"";
        break;
      }
      default: break;
    }
    return "literal(" + name + ":" + val_str + ")";
  }
  return "<unknown>";
}

}  // anonymous namespace

std::string gpu_expression_translator::translated_expression::to_string() const
{
  if (tree.size() == 0) { return "<empty>"; }
  return expression_to_string(tree.back());
}

std::optional<gpu_expression_translator::translated_expression>
gpu_expression_translator::translate_expression(duckdb::Expression const& expr,
                                                cudf::ast::table_reference const table_src)
{
  reset_tree();
  auto expr_ref = add_expression(expr, table_src);
  if (!expr_ref) { return std::nullopt; }
  translated_expression result;
  result.tree           = std::move(_ast_tree);
  result.owned_literals = std::move(_literal_scalars);
  return result;
}

std::optional<gpu_expression_translator::translated_expression>
gpu_expression_translator::translate_expression_with_names(
  duckdb::Expression const& expr, column_name_resolver_fxn column_name_resolver)
{
  reset_tree();
  _column_name_resolver = std::move(column_name_resolver);
  auto expr_ref         = add_expression(expr, cudf::ast::table_reference::LEFT);
  _column_name_resolver = nullptr;
  if (!expr_ref) { return std::nullopt; }
  translated_expression result;
  result.tree           = std::move(_ast_tree);
  result.owned_literals = std::move(_literal_scalars);
  return result;
}

std::optional<expr_ref> gpu_expression_translator::add_join_condition(
  duckdb::JoinCondition const& condition, bool swap_sides)
{
  auto left_table_ref =
    swap_sides ? cudf::ast::table_reference::RIGHT : cudf::ast::table_reference::LEFT;
  auto right_table_ref =
    swap_sides ? cudf::ast::table_reference::LEFT : cudf::ast::table_reference::RIGHT;

  auto left_expr = add_expression(*condition.left, left_table_ref);
  if (!left_expr) { return std::nullopt; }

  auto right_expr = add_expression(*condition.right, right_table_ref);
  if (!right_expr) { return std::nullopt; }

  switch (condition.comparison) {
    case duckdb::ExpressionType::COMPARE_EQUAL:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::EQUAL, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_NOTEQUAL:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::NOT_EQUAL, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_LESSTHAN:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_GREATERTHAN:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::GREATER, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS_EQUAL, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::GREATER_EQUAL, *left_expr, *right_expr);
    default:
      SIRIUS_LOG_DEBUG("[expression_translator] Unsupported join condition comparison type: {}",
                       static_cast<int>(condition.comparison));
      return std::nullopt;
  }
}

std::optional<gpu_expression_translator::translated_expression>
gpu_expression_translator::translate_join_condition(duckdb::JoinCondition const& condition)
{
  reset_tree();
  auto cond_ref = add_join_condition(condition);
  if (!cond_ref) { return std::nullopt; }
  translated_expression result;
  result.tree           = std::move(_ast_tree);
  result.owned_literals = std::move(_literal_scalars);
  return result;
}

std::optional<gpu_expression_translator::translated_expression>
gpu_expression_translator::translate_join_conditions(
  duckdb::vector<duckdb::JoinCondition> const& conditions,
  std::size_t start_idx,
  std::size_t end_idx,
  bool swap_sides)
{
  if (start_idx >= end_idx) { return std::nullopt; }

  reset_tree();

  auto combined = add_join_condition(conditions[start_idx], swap_sides);
  if (!combined) { return std::nullopt; }

  for (std::size_t i = start_idx + 1; i < end_idx; ++i) {
    auto next = add_join_condition(conditions[i], swap_sides);
    if (!next) { return std::nullopt; }
    combined = _ast_tree.emplace<cudf::ast::operation>(
      cudf::ast::ast_operator::LOGICAL_AND, *combined, *next);
  }

  translated_expression result;
  result.tree           = std::move(_ast_tree);
  result.owned_literals = std::move(_literal_scalars);
  return result;
}

std::optional<expr_ref> gpu_expression_translator::add_expression(
  duckdb::Expression const& expr, cudf::ast::table_reference const table_src)
{
  switch (expr.GetExpressionClass()) {
    case duckdb::ExpressionClass::BOUND_BETWEEN:
      return add_expression(expr.Cast<duckdb::BoundBetweenExpression>(), table_src);
    case duckdb::ExpressionClass::BOUND_CASE: {
      SIRIUS_LOG_DEBUG(
        "[expression_translator] CASE expressions cannot be translated to cuDF ASTs: {}",
        expr.ToString());
      return std::nullopt;
    }
    case duckdb::ExpressionClass::BOUND_CAST:
      return add_expression(expr.Cast<duckdb::BoundCastExpression>(), table_src);
    case duckdb::ExpressionClass::BOUND_COMPARISON:
      return add_expression(expr.Cast<duckdb::BoundComparisonExpression>(), table_src);
    case duckdb::ExpressionClass::BOUND_CONJUNCTION:
      return add_expression(expr.Cast<duckdb::BoundConjunctionExpression>(), table_src);
    case duckdb::ExpressionClass::BOUND_CONSTANT:
      return add_expression(expr.Cast<duckdb::BoundConstantExpression>(), table_src);
    case duckdb::ExpressionClass::BOUND_FUNCTION:
      return add_expression(expr.Cast<duckdb::BoundFunctionExpression>(), table_src);
    case duckdb::ExpressionClass::BOUND_OPERATOR:
      return add_expression(expr.Cast<duckdb::BoundOperatorExpression>(), table_src);
    case duckdb::ExpressionClass::BOUND_PARAMETER: {
      SIRIUS_LOG_DEBUG(
        "[expression_translator] Cannot translate parameter expressions to cuDF ASTs: {}",
        expr.ToString());
      return std::nullopt;
    }
    case duckdb::ExpressionClass::BOUND_REF:
      return add_expression(expr.Cast<duckdb::BoundReferenceExpression>(), table_src);
    default:
      throw duckdb::InternalException("[expression_translator] Unknown ExpressionClass: {}",
                                      expr.GetExpressionClass());
  }
}

//===----------BETWEEN----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  duckdb::BoundBetweenExpression const& expr, cudf::ast::table_reference const table_src)
{
  // Add the children.
  auto input_expr = add_expression(*expr.input, table_src);
  auto lower_expr = add_expression(*expr.lower, table_src);
  auto upper_expr = add_expression(*expr.upper, table_src);

  // Check for failure in translating children
  if (!input_expr || !lower_expr || !upper_expr) { return std::nullopt; }

  // Construct the BETWEEN expression
  auto const& lower_cmp_op = _ast_tree.emplace<cudf::ast::operation>(
    cudf::ast::ast_operator::GREATER_EQUAL, *input_expr, *lower_expr);
  auto const& upper_cmp_op = _ast_tree.emplace<cudf::ast::operation>(
    cudf::ast::ast_operator::LESS_EQUAL, *input_expr, *upper_expr);
  return _ast_tree.emplace<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, lower_cmp_op, upper_cmp_op);
}

//===----------CAST----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  duckdb::BoundCastExpression const& expr, cudf::ast::table_reference const table_src)
{
  // Add the child
  auto child_expr = add_expression(*expr.child, table_src);

  // Check for failure in translating child
  if (!child_expr) { return std::nullopt; }

  // Construct the CAST expression, if possible
  // CuDF AST only supports casts to INT64, UINT64, FLOAT64
  auto const cudf_return_type = GetCudfType(expr.return_type);
  switch (cudf_return_type.id()) {
    case cudf::type_id::INT64:
      return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_INT64,
                                                     *child_expr);
    case cudf::type_id::UINT64:
      return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_UINT64,
                                                     *child_expr);
    case cudf::type_id::FLOAT64:
      return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_FLOAT64,
                                                     *child_expr);
    default: {
      SIRIUS_LOG_DEBUG("[expression_translator] Unsupported cast type_id: {}",
                       static_cast<int>(cudf_return_type.id()));
      return std::nullopt;
    }
  }
}

//===----------COMPARISON----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  duckdb::BoundComparisonExpression const& expr, cudf::ast::table_reference const table_src)
{
  // Add the children
  auto left_expr  = add_expression(*expr.left, table_src);
  auto right_expr = add_expression(*expr.right, table_src);

  // Check for failure in translating children
  if (!left_expr || !right_expr) { return std::nullopt; }

  // Construct the comparison expression
  switch (expr.GetExpressionType()) {
    case duckdb::ExpressionType::COMPARE_EQUAL:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::EQUAL, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_NOTEQUAL:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::NOT_EQUAL, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_LESSTHAN:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_GREATERTHAN:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::GREATER, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS_EQUAL, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::GREATER_EQUAL, *left_expr, *right_expr);
    case duckdb::ExpressionType::COMPARE_DISTINCT_FROM:
    case duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM: {
      SIRIUS_LOG_DEBUG(
        "[expression_translator] DISTINCT comparisons not supported in expression translator");
      return std::nullopt;
    }
    default:
      throw duckdb::InternalException("[expression_translator] Unknown comparison type: {}",
                                      expr.GetExpressionType());
  }
}

//===----------CONJUNCTION----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  duckdb::BoundConjunctionExpression const& expr, cudf::ast::table_reference const table_src)
{
  // If there are no children, return
  if (expr.children.empty()) { return std::nullopt; }

  // Add the children and combine with AND/OR operations as we go
  auto result = add_expression(*expr.children[0], table_src);
  if (!result) { return std::nullopt; }

  for (size_t i = 1; i < expr.children.size(); ++i) {
    // Add child expression
    auto child_expr = add_expression(*expr.children[i], table_src);

    // Check for failure in translating child
    if (!child_expr) { return std::nullopt; }

    // Combine with previous children using AND/OR
    if (expr.GetExpressionType() == duckdb::ExpressionType::CONJUNCTION_AND) {
      result = _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LOGICAL_AND, *result, *child_expr);
    } else if (expr.GetExpressionType() == duckdb::ExpressionType::CONJUNCTION_OR) {
      result = _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LOGICAL_OR, *result, *child_expr);
    } else {
      throw duckdb::InternalException("[expression_translator] Unknown conjunction type: {}",
                                      expr.GetExpressionType());
    }
  }
  return result;
}

//===----------CONSTANT----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  duckdb::BoundConstantExpression const& expr, cudf::ast::table_reference const table_src)
{
  auto const cudf_type = GetCudfType(expr.return_type);
  // TODO: Expand type support as needed. See gpu_execute_constant.cpp.
  switch (cudf_type.id()) {
    case cudf::type_id::INT8: {
      return add_literal_expression<cudf::numeric_scalar<int8_t>>(
        expr.value.GetValue<int8_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::INT16: {
      return add_literal_expression<cudf::numeric_scalar<int16_t>>(
        expr.value.GetValue<int16_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::INT32: {
      return add_literal_expression<cudf::numeric_scalar<int32_t>>(
        expr.value.GetValue<int32_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::INT64: {
      return add_literal_expression<cudf::numeric_scalar<int64_t>>(
        expr.value.GetValue<int64_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::UINT8: {
      return add_literal_expression<cudf::numeric_scalar<uint8_t>>(
        expr.value.GetValue<uint8_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::UINT16: {
      return add_literal_expression<cudf::numeric_scalar<uint16_t>>(
        expr.value.GetValue<uint16_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::UINT32: {
      return add_literal_expression<cudf::numeric_scalar<uint32_t>>(
        expr.value.GetValue<uint32_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::UINT64: {
      return add_literal_expression<cudf::numeric_scalar<uint64_t>>(
        expr.value.GetValue<uint64_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::FLOAT32: {
      return add_literal_expression<cudf::numeric_scalar<float_t>>(
        expr.value.GetValue<float_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::FLOAT64: {
      return add_literal_expression<cudf::numeric_scalar<double_t>>(
        expr.value.GetValue<double_t>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::BOOL8: {
      return add_literal_expression<cudf::numeric_scalar<bool>>(
        expr.value.GetValue<bool>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::STRING: {
      return add_literal_expression<cudf::string_scalar>(
        expr.value.GetValue<std::string>(), true, _stream, _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_DAYS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_D>>(
        cudf::duration_D{expr.value.GetValue<duckdb::date_t>().days}, true, _stream, _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_SECONDS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_s>>(
        cudf::duration_s{expr.value.GetValue<duckdb::timestamp_sec_t>().value},
        true,
        _stream,
        _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_MILLISECONDS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_ms>>(
        cudf::duration_ms{expr.value.GetValue<duckdb::timestamp_ms_t>().value},
        true,
        _stream,
        _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_us>>(
        cudf::duration_us{expr.value.GetValue<duckdb::timestamp_tz_t>().value},
        true,
        _stream,
        _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_ns>>(
        cudf::duration_ns{expr.value.GetValue<duckdb::timestamp_tz_t>().value},
        true,
        _stream,
        _resource_ref);
    }
    // cudf decimal type uses negative scale
    /// TODO: Uncomment the decimal code once the following bug fix is released in cuDF and we
    /// update to that version:
    /// https://github.com/rapidsai/cudf/pull/21447
    /// For now, fallthrough to default case to return nullopt and effectively disallow decimal
    /// types.
    case cudf::type_id::DECIMAL32: {
      // return add_literal_expression<cudf::fixed_point_scalar<numeric::decimal32>>(
      //   expr.value.GetValueUnsafe<typename numeric::decimal32::rep>(),
      //   numeric::scale_type{-duckdb::DecimalType::GetScale(expr.value.type())},
      //   true,
      //   _stream,
      //   _resource_ref);
    }
    case cudf::type_id::DECIMAL64: {
      // return add_literal_expression<cudf::fixed_point_scalar<numeric::decimal64>>(
      //   expr.value.GetValueUnsafe<typename numeric::decimal64::rep>(),
      //   numeric::scale_type{-duckdb::DecimalType::GetScale(expr.value.type())},
      //   true,
      //   _stream,
      //   _resource_ref);
    }
    case cudf::type_id::DECIMAL128: {
      // duckdb::hugeint_t const value = expr.value.GetValueUnsafe<duckdb::hugeint_t>();
      // __int128_t rep                = (static_cast<__int128_t>(value.upper) << 64) | value.lower;
      // return add_literal_expression<cudf::fixed_point_scalar<numeric::decimal128>>(
      //   rep,
      //   numeric::scale_type{-duckdb::DecimalType::GetScale(expr.value.type())},
      //   true,
      //   _stream,
      //   _resource_ref);
      SIRIUS_LOG_DEBUG(
        "[expression_translator] DECIMAL types are not supported in expression translator due to "
        "cuDF bug");
      return std::nullopt;
    }
    default: {
      SIRIUS_LOG_DEBUG("[expression_translator] Unsupported constant type_id: {}",
                       static_cast<int>(cudf_type.id()));
      return std::nullopt;
    }
  }
}

//===----------FUNCTION----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  duckdb::BoundFunctionExpression const& expr, cudf::ast::table_reference const table_src)
{
  // cuDF AST only supports numeric binary functions
  auto const& func_str = expr.function.name;
  if (func_str == "+") {
    return add_function_expression<cudf::ast::ast_operator::ADD>(expr, table_src);
  } else if (func_str == "-") {
    return add_function_expression<cudf::ast::ast_operator::SUB>(expr, table_src);
  } else if (func_str == "*") {
    return add_function_expression<cudf::ast::ast_operator::MUL>(expr, table_src);
  } else if (func_str == "/" || func_str == "//") {
    return add_function_expression<cudf::ast::ast_operator::DIV>(expr, table_src);
  } else if (func_str == "%") {
    return add_function_expression<cudf::ast::ast_operator::MOD>(expr, table_src);
  }
  SIRIUS_LOG_DEBUG("[expression_translator] Unsupported function: {}", func_str);
  return std::nullopt;
}

//===----------OPERATOR----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  duckdb::BoundOperatorExpression const& expr, cudf::ast::table_reference const table_src)
{
  switch (expr.type) {
    case duckdb::ExpressionType::COMPARE_IN:  // Fallthrough
    case duckdb::ExpressionType::COMPARE_NOT_IN: {
      // [KEVIN]: It may be wise to limit the number of children for IN expressions that we
      // attempt to translate, as a large number of children could lead to a very large/complex
      // AST that could be very slow. For now, we will optimistically attempt to translate all
      // IN expressions regardless of number of children, but we can revisit this if it becomes an
      // issue.
      assert(expr.children.size() > 1);  // IN expressions must have at least 2 children (test
                                         // expression and at least 1 comparator expression)

      // Translate the first comparison expression
      auto test_expr = add_expression(*expr.children[0], table_src);
      if (!test_expr) { return std::nullopt; }
      auto comparator_expr = add_expression(*expr.children[1], table_src);
      if (!comparator_expr) { return std::nullopt; }
      expr_ref comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::EQUAL, *test_expr, *comparator_expr);

      // Loop over children, building an OR tree of comparisons.
      // Re-translate the test expression each time to avoid shared AST subgraphs.
      for (size_t child = 2; child < expr.children.size(); ++child) {
        auto test_expr = add_expression(*expr.children[0], table_src);
        if (!test_expr) { return std::nullopt; }
        auto comparator_expr = add_expression(*expr.children[child], table_src);
        if (!comparator_expr) { return std::nullopt; }

        expr_ref next_comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::EQUAL, *test_expr, *comparator_expr);
        comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::LOGICAL_OR, comparison_expr, next_comparison_expr);
      }

      if (expr.type == duckdb::ExpressionType::COMPARE_IN) { return comparison_expr; }
      return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, comparison_expr);
    }
    case duckdb::ExpressionType::OPERATOR_COALESCE: {
      SIRIUS_LOG_DEBUG(
        "[expression_translator] COALESCE operator not supported in expression translator");
      return std::nullopt;
    }
    case duckdb::ExpressionType::OPERATOR_TRY: {
      SIRIUS_LOG_DEBUG(
        "[expression_translator] TRY operator not supported in expression translator");
      return std::nullopt;
    }
    case duckdb::ExpressionType::OPERATOR_NOT: {
      auto child_expr = add_expression(*expr.children[0], table_src);
      if (!child_expr) { return std::nullopt; }

      return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, *child_expr);
    }
    case duckdb::ExpressionType::OPERATOR_IS_NULL:  // Fallthrough
    case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL: {
      auto child_expr = add_expression(*expr.children[0], table_src);
      if (!child_expr) { return std::nullopt; }

      // Add IS_NULL followed by NOT to represent IS_NOT_NULL
      expr_ref is_null_op =
        _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL, *child_expr);
      if (expr.type == duckdb::ExpressionType::OPERATOR_IS_NULL) { return is_null_op; }
      return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, is_null_op);
    }
    default:
      throw duckdb::InternalException("[expression_translator] Unknown operator type: {}",
                                      expr.GetExpressionType());
  }
}

//===----------REFERENCE----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  duckdb::BoundReferenceExpression const& expr, cudf::ast::table_reference const table_src)
{
  // We need to disable DECIMAL types in the expression translator for now because of cuDF bug.
  auto const cudf_type = GetCudfType(expr.return_type);
  if (cudf_type.id() == cudf::type_id::DECIMAL32 || cudf_type.id() == cudf::type_id::DECIMAL64 ||
      cudf_type.id() == cudf::type_id::DECIMAL128) {
    SIRIUS_LOG_DEBUG(
      "[expression_translator] DECIMAL types are not supported in expression translator due to "
      "cuDF bug");
    return std::nullopt;
  }

  if (_column_name_resolver) {
    return _ast_tree.emplace<cudf::ast::column_name_reference>(_column_name_resolver(expr.index));
  }
  return _ast_tree.emplace<cudf::ast::column_reference>(expr.index, table_src);
}

}  // namespace sirius