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
#include <expression/ast/aggregate.hpp>
#include <expression_evaluator/gpu_expression_translator_internal.hpp>
#include <log/logging.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

// standard library
#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <variant>

namespace sirius {

using expr_ref = std::reference_wrapper<cudf::ast::expression const>;

std::string ast_operator_to_string(cudf::ast::ast_operator op)
{
  switch (op) {
    case cudf::ast::ast_operator::ADD: return "+";
    case cudf::ast::ast_operator::SUB: return "-";
    case cudf::ast::ast_operator::MUL: return "*";
    case cudf::ast::ast_operator::DIV: return "/";
    case cudf::ast::ast_operator::TRUE_DIV: return "TRUE_DIV";
    case cudf::ast::ast_operator::FLOOR_DIV: return "FLOOR_DIV";
    case cudf::ast::ast_operator::MOD: return "%";
    case cudf::ast::ast_operator::PYMOD: return "PYMOD";
    case cudf::ast::ast_operator::POW: return "POW";
    case cudf::ast::ast_operator::EQUAL: return "==";
    case cudf::ast::ast_operator::NULL_EQUAL: return "NULL_EQUAL";
    case cudf::ast::ast_operator::NOT_EQUAL: return "!=";
    case cudf::ast::ast_operator::LESS: return "<";
    case cudf::ast::ast_operator::GREATER: return ">";
    case cudf::ast::ast_operator::LESS_EQUAL: return "<=";
    case cudf::ast::ast_operator::GREATER_EQUAL: return ">=";
    case cudf::ast::ast_operator::BITWISE_AND: return "&";
    case cudf::ast::ast_operator::BITWISE_OR: return "|";
    case cudf::ast::ast_operator::BITWISE_XOR: return "^";
    case cudf::ast::ast_operator::LOGICAL_AND: return "&&";
    case cudf::ast::ast_operator::NULL_LOGICAL_AND: return "NULL_LOGICAL_AND";
    case cudf::ast::ast_operator::LOGICAL_OR: return "||";
    case cudf::ast::ast_operator::NULL_LOGICAL_OR: return "NULL_LOGICAL_OR";
    case cudf::ast::ast_operator::IDENTITY: return "IDENTITY";
    case cudf::ast::ast_operator::IS_NULL: return "IS_NULL";
    case cudf::ast::ast_operator::SIN: return "SIN";
    case cudf::ast::ast_operator::COS: return "COS";
    case cudf::ast::ast_operator::TAN: return "TAN";
    case cudf::ast::ast_operator::ARCSIN: return "ARCSIN";
    case cudf::ast::ast_operator::ARCCOS: return "ARCCOS";
    case cudf::ast::ast_operator::ARCTAN: return "ARCTAN";
    case cudf::ast::ast_operator::SINH: return "SINH";
    case cudf::ast::ast_operator::COSH: return "COSH";
    case cudf::ast::ast_operator::TANH: return "TANH";
    case cudf::ast::ast_operator::ARCSINH: return "ARCSINH";
    case cudf::ast::ast_operator::ARCCOSH: return "ARCCOSH";
    case cudf::ast::ast_operator::ARCTANH: return "ARCTANH";
    case cudf::ast::ast_operator::EXP: return "EXP";
    case cudf::ast::ast_operator::LOG: return "LOG";
    case cudf::ast::ast_operator::SQRT: return "SQRT";
    case cudf::ast::ast_operator::CBRT: return "CBRT";
    case cudf::ast::ast_operator::CEIL: return "CEIL";
    case cudf::ast::ast_operator::FLOOR: return "FLOOR";
    case cudf::ast::ast_operator::ABS: return "ABS";
    case cudf::ast::ast_operator::RINT: return "RINT";
    case cudf::ast::ast_operator::BIT_INVERT: return "~";
    case cudf::ast::ast_operator::NOT: return "NOT";
    case cudf::ast::ast_operator::CAST_TO_INT64: return "CAST_TO_INT64";
    case cudf::ast::ast_operator::CAST_TO_UINT64: return "CAST_TO_UINT64";
    case cudf::ast::ast_operator::CAST_TO_FLOAT64: return "CAST_TO_FLOAT64";
    default: return "UNKNOWN_OP";
  }
}

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

std::string gpu_expression_translator::translated_expression::to_string() const
{
  if (tree.size() == 0) { return "<empty>"; }
  return expression_to_string(tree.back());
}

std::optional<gpu_expression_translator::translated_expression>
gpu_expression_translator::translate_expression(sirius::ast::node const& expr,
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
  sirius::ast::node const& expr, column_name_resolver_fxn column_name_resolver)
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
  sirius::join_condition const& condition, bool swap_sides)
{
  auto left_table_ref =
    swap_sides ? cudf::ast::table_reference::RIGHT : cudf::ast::table_reference::LEFT;
  auto right_table_ref =
    swap_sides ? cudf::ast::table_reference::LEFT : cudf::ast::table_reference::RIGHT;

  auto const* left_node = condition.left.get();
  if (!left_node) { return std::nullopt; }
  auto left_expr = add_expression(*left_node, left_table_ref);
  if (!left_expr) { return std::nullopt; }

  auto const* right_node = condition.right.get();
  if (!right_node) { return std::nullopt; }
  auto right_expr = add_expression(*right_node, right_table_ref);
  if (!right_expr) { return std::nullopt; }

  switch (condition.comparison) {
    using enum sirius::comparison_type;
    case equal:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::EQUAL, *left_expr, *right_expr);
    case not_equal:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::NOT_EQUAL, *left_expr, *right_expr);
    case lt:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS, *left_expr, *right_expr);
    case gt:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::GREATER, *left_expr, *right_expr);
    case le:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS_EQUAL, *left_expr, *right_expr);
    case ge:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::GREATER_EQUAL, *left_expr, *right_expr);
    case not_distinct_from:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::NULL_EQUAL, *left_expr, *right_expr);
    default:
      SIRIUS_LOG_DEBUG("[expression_translator] Unsupported join condition comparison type: {}",
                       static_cast<int>(condition.comparison));
      return std::nullopt;
  }
}

std::optional<gpu_expression_translator::translated_expression>
gpu_expression_translator::translate_join_condition(sirius::join_condition const& condition)
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
  duckdb::vector<sirius::join_condition> const& conditions,
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

namespace {

/// @brief Recover the SQL type identifier of an AST node, where it is carried.
///
/// Used by the function arm's decimal-propagation guard. Node kinds that carry
/// a logical type (reference, constant, cast, function_call) return its id;
/// alternatives without one return INVALID, treated as a non-DECIMAL sentinel
/// by the caller. References must be included so that direct decimal-column
/// arithmetic (e.g. col + col) trips the guard, matching the pre-AST translator
/// which checked every function child's return_type.
sirius::type_id node_logical_type_id(sirius::ast::node const& expr)
{
  return std::visit(
    [](auto const& alt) -> sirius::type_id {
      using T = std::decay_t<decltype(alt)>;
      if constexpr (std::is_same_v<T, sirius::ast::reference>) {
        return alt.return_type().id();
      } else if constexpr (std::is_same_v<T, sirius::ast::constant>) {
        return alt.return_type().id();
      } else if constexpr (std::is_same_v<T, sirius::ast::cast>) {
        return alt.target_type.id();
      } else if constexpr (std::is_same_v<T, sirius::ast::function_call>) {
        return alt.return_type().id();
      } else {
        return sirius::type_id::INVALID;
      }
    },
    expr.v);
}

}  // namespace

std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::node const& expr, cudf::ast::table_reference const table_src)
{
  return std::visit(
    [this, table_src](auto const& alt) -> std::optional<expr_ref> {
      using T = std::decay_t<decltype(alt)>;
      if constexpr (std::is_same_v<T, sirius::ast::reference> ||
                    std::is_same_v<T, sirius::ast::constant> ||
                    std::is_same_v<T, sirius::ast::comparison> ||
                    std::is_same_v<T, sirius::ast::conjunction> ||
                    std::is_same_v<T, sirius::ast::between> ||
                    std::is_same_v<T, sirius::ast::case_expr> ||
                    std::is_same_v<T, sirius::ast::cast> ||
                    std::is_same_v<T, sirius::ast::unary_op> ||
                    std::is_same_v<T, sirius::ast::coalesce> ||
                    std::is_same_v<T, sirius::ast::in_list> ||
                    std::is_same_v<T, sirius::ast::function_call> ||
                    std::is_same_v<T, sirius::ast::aggregate>) {
        return this->add_expression(alt, table_src);
      } else {
        static_assert(sizeof(T) == 0,
                      "Unhandled sirius::ast alternative in add_expression dispatcher");
      }
    },
    expr.v);
}

//===----------REFERENCE----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::reference const& alt, cudf::ast::table_reference const table_src)
{
  if (_column_name_resolver) {
    return _ast_tree.emplace<cudf::ast::column_name_reference>(
      _column_name_resolver(alt.column_index));
  }
  return _ast_tree.emplace<cudf::ast::column_reference>(alt.column_index, table_src);
}

//===----------CONSTANT----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::constant const& alt, cudf::ast::table_reference const /*table_src*/)
{
  // A typed NULL constant has no cuDF AST literal representation; signal
  // untranslatable so the caller falls back to DuckDB execution rather than
  // unwrapping the empty payload below.
  if (std::holds_alternative<sirius::null_value>(alt.payload)) {
    SIRIUS_LOG_DEBUG("[expression_translator] NULL constants cannot be translated to cuDF ASTs");
    return std::nullopt;
  }

  auto const cudf_type = sirius::get_cudf_type(alt.return_type());
  // TODO: Expand type support as needed. See constant.cpp.
  switch (cudf_type.id()) {
    case cudf::type_id::INT8: {
      return add_literal_expression<cudf::numeric_scalar<int8_t>>(
        std::get<int8_t>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::INT16: {
      return add_literal_expression<cudf::numeric_scalar<int16_t>>(
        std::get<int16_t>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::INT32: {
      return add_literal_expression<cudf::numeric_scalar<int32_t>>(
        std::get<int32_t>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::INT64: {
      return add_literal_expression<cudf::numeric_scalar<int64_t>>(
        std::get<int64_t>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::UINT8: {
      return add_literal_expression<cudf::numeric_scalar<uint8_t>>(
        std::get<uint8_t>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::UINT16: {
      return add_literal_expression<cudf::numeric_scalar<uint16_t>>(
        std::get<uint16_t>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::UINT32: {
      return add_literal_expression<cudf::numeric_scalar<uint32_t>>(
        std::get<uint32_t>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::UINT64: {
      return add_literal_expression<cudf::numeric_scalar<uint64_t>>(
        std::get<uint64_t>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::FLOAT32: {
      return add_literal_expression<cudf::numeric_scalar<float>>(
        std::get<float>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::FLOAT64: {
      return add_literal_expression<cudf::numeric_scalar<double>>(
        std::get<double>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::BOOL8: {
      return add_literal_expression<cudf::numeric_scalar<bool>>(
        std::get<bool>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::STRING: {
      return add_literal_expression<cudf::string_scalar>(
        std::get<std::string>(alt.payload), true, _stream, _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_DAYS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_D>>(
        cudf::duration_D{std::get<sirius::date_value>(alt.payload).days},
        true,
        _stream,
        _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_SECONDS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_s>>(
        cudf::duration_s{std::get<sirius::timestamp_sec_value>(alt.payload).value},
        true,
        _stream,
        _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_MILLISECONDS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_ms>>(
        cudf::duration_ms{std::get<sirius::timestamp_ms_value>(alt.payload).value},
        true,
        _stream,
        _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_us>>(
        cudf::duration_us{std::get<sirius::timestamp_us_value>(alt.payload).value},
        true,
        _stream,
        _resource_ref);
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      return add_literal_expression<cudf::timestamp_scalar<cudf::timestamp_ns>>(
        cudf::duration_ns{std::get<sirius::timestamp_ns_value>(alt.payload).value},
        true,
        _stream,
        _resource_ref);
    }
    // cudf decimal type uses negative scale
    case cudf::type_id::DECIMAL32: {
      return add_literal_expression<cudf::fixed_point_scalar<numeric::decimal32>>(
        std::get<sirius::decimal32>(alt.payload).value,
        numeric::scale_type{-static_cast<int32_t>(alt.return_type().decimal_scale())},
        true,
        _stream,
        _resource_ref);
    }
    case cudf::type_id::DECIMAL64: {
      return add_literal_expression<cudf::fixed_point_scalar<numeric::decimal64>>(
        std::get<sirius::decimal64>(alt.payload).value,
        numeric::scale_type{-static_cast<int32_t>(alt.return_type().decimal_scale())},
        true,
        _stream,
        _resource_ref);
    }
    case cudf::type_id::DECIMAL128: {
      return add_literal_expression<cudf::fixed_point_scalar<numeric::decimal128>>(
        std::get<sirius::decimal128>(alt.payload).value,
        numeric::scale_type{-static_cast<int32_t>(alt.return_type().decimal_scale())},
        true,
        _stream,
        _resource_ref);
    }
    default: {
      SIRIUS_LOG_DEBUG("[expression_translator] Unsupported constant type_id: {}",
                       static_cast<int>(cudf_type.id()));
      return std::nullopt;
    }
  }
}

//===----------COMPARISON----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::comparison const& alt, cudf::ast::table_reference const table_src)
{
  // Add the children
  auto left_expr  = add_expression(*alt.left, table_src);
  auto right_expr = add_expression(*alt.right, table_src);

  // Check for failure in translating children
  if (!left_expr || !right_expr) { return std::nullopt; }

  // Construct the comparison expression
  switch (alt.op) {
    using enum sirius::comparison_type;
    case equal:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::EQUAL, *left_expr, *right_expr);
    case not_equal:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::NOT_EQUAL, *left_expr, *right_expr);
    case lt:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS, *left_expr, *right_expr);
    case gt:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::GREATER, *left_expr, *right_expr);
    case le:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS_EQUAL, *left_expr, *right_expr);
    case ge:
      return _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::GREATER_EQUAL, *left_expr, *right_expr);
    case distinct_from:
    case not_distinct_from: {
      SIRIUS_LOG_DEBUG(
        "[expression_translator] DISTINCT comparisons not supported in expression translator");
      return std::nullopt;
    }
    default:
      SIRIUS_LOG_DEBUG("[expression_translator] Unsupported comparison type: {}",
                       static_cast<int>(alt.op));
      return std::nullopt;
  }
}

//===----------CONJUNCTION----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::conjunction const& alt, cudf::ast::table_reference const table_src)
{
  // If there are no children, return
  if (alt.children.empty()) { return std::nullopt; }

  // Resolve the cuDF operator once up front rather than re-checking per child.
  cudf::ast::ast_operator cudf_op{};
  using enum sirius::ast::conjunction::kind;
  // SQL AND/OR use Kleene three-valued logic (TRUE OR NULL = TRUE), so use cuDF's
  // NULL_LOGICAL_* operators rather than the null-propagating LOGICAL_*.
  switch (alt.op) {
    case op_and: cudf_op = cudf::ast::ast_operator::NULL_LOGICAL_AND; break;
    case op_or: cudf_op = cudf::ast::ast_operator::NULL_LOGICAL_OR; break;
    default:
      SIRIUS_LOG_DEBUG("[expression_translator] Unsupported conjunction type: {}",
                       static_cast<int>(alt.op));
      return std::nullopt;
  }

  // Seed with the first child, then fold the rest, combining with AND/OR as we go.
  auto seed = add_expression(*alt.children.front(), table_src);
  if (!seed) { return std::nullopt; }

  return std::accumulate(
    std::next(alt.children.begin()),
    alt.children.end(),
    seed,
    [&](std::optional<expr_ref> acc,
        std::unique_ptr<sirius::ast::node> const& child) -> std::optional<expr_ref> {
      // A previous child failed to translate; propagate the failure.
      if (!acc) { return std::nullopt; }

      // Add child expression and check for failure in translating it.
      auto child_expr = add_expression(*child, table_src);
      if (!child_expr) { return std::nullopt; }

      return _ast_tree.emplace<cudf::ast::operation>(cudf_op, *acc, *child_expr);
    });
}

//===----------BETWEEN----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::between const& alt, cudf::ast::table_reference const table_src)
{
  // Add the children.
  auto input_expr = add_expression(*alt.input, table_src);
  auto lower_expr = add_expression(*alt.lower, table_src);
  auto upper_expr = add_expression(*alt.upper, table_src);

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

//===----------CASE----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::case_expr const& /*alt*/, cudf::ast::table_reference const /*table_src*/)
{
  SIRIUS_LOG_DEBUG("[expression_translator] CASE expressions cannot be translated to cuDF ASTs");
  return std::nullopt;
}

//===----------CAST----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::cast const& alt, cudf::ast::table_reference const table_src)
{
  // Add the child
  auto child_expr = add_expression(*alt.child, table_src);

  // Check for failure in translating child
  if (!child_expr) { return std::nullopt; }

  // Construct the CAST expression, if possible
  // CuDF AST only supports casts to INT64, UINT64, FLOAT64
  auto const cudf_return_type = sirius::get_cudf_type(alt.target_type);
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

//===----------UNARY OPERATOR----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::unary_op const& alt, cudf::ast::table_reference const table_src)
{
  auto child_expr = add_expression(*alt.child, table_src);
  if (!child_expr) { return std::nullopt; }

  switch (alt.op) {
    using enum sirius::ast::unary_op::kind;
    case op_not:
      return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, *child_expr);
    case op_is_null:
      return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL, *child_expr);
    case op_is_not_null: {
      // Add IS_NULL followed by NOT to represent IS_NOT_NULL
      expr_ref is_null_op =
        _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL, *child_expr);
      return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, is_null_op);
    }
    case op_try:
    default:
      // cuDF AST cannot express TRY.
      SIRIUS_LOG_DEBUG(
        "[expression_translator] TRY operator not supported in expression translator");
      return std::nullopt;
  }
}

//===----------COALESCE----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::coalesce const& /*alt*/, cudf::ast::table_reference const /*table_src*/)
{
  SIRIUS_LOG_DEBUG(
    "[expression_translator] COALESCE operator not supported in expression translator");
  return std::nullopt;
}

//===----------IN / NOT IN----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::in_list const& alt, cudf::ast::table_reference const table_src)
{
  // IN expressions must have at least one comparator value.
  if (alt.values.empty()) { return std::nullopt; }

  // Translate the first comparison expression
  auto test_expr = add_expression(*alt.probe, table_src);
  if (!test_expr) { return std::nullopt; }
  auto comparator_expr = add_expression(*alt.values[0], table_src);
  if (!comparator_expr) { return std::nullopt; }
  expr_ref comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, *test_expr, *comparator_expr);

  // Loop over values, building an OR tree of comparisons.
  // Re-translate the probe expression each time to avoid shared AST subgraphs.
  for (size_t i = 1; i < alt.values.size(); ++i) {
    auto probe_expr = add_expression(*alt.probe, table_src);
    if (!probe_expr) { return std::nullopt; }
    auto value_expr = add_expression(*alt.values[i], table_src);
    if (!value_expr) { return std::nullopt; }

    expr_ref next_comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
      cudf::ast::ast_operator::EQUAL, *probe_expr, *value_expr);
    comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
      cudf::ast::ast_operator::LOGICAL_OR, comparison_expr, next_comparison_expr);
  }

  if (!alt.negated) { return comparison_expr; }
  return _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, comparison_expr);
}

//===----------FUNCTION----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::function_call const& alt, cudf::ast::table_reference const table_src)
{
  // cuDF AST only supports numeric binary functions. We must also disable any
  // operation that propagates decimal types up the expression tree, pending the
  // cuDF fix: https://github.com/rapidsai/cudf/pull/21996. Unsupported functions
  // fall through to the default arm below, so checking decimal propagation once
  // up front (rather than per supported case) is equivalent.
  if (std::ranges::any_of(alt.arguments(), [](auto const& arg) {
        return node_logical_type_id(*arg) == sirius::type_id::DECIMAL;
      })) {
    SIRIUS_LOG_DEBUG(
      "[expression_translator] Blocking function because it propagates decimal types");
    return std::nullopt;
  }

  switch (alt.function()) {
    using enum sirius::function_id;
    case add: return add_function_expression<cudf::ast::ast_operator::ADD>(alt, table_src);
    case sub: return add_function_expression<cudf::ast::ast_operator::SUB>(alt, table_src);
    case mul: return add_function_expression<cudf::ast::ast_operator::MUL>(alt, table_src);
    case div:
    case int_div: return add_function_expression<cudf::ast::ast_operator::DIV>(alt, table_src);
    case mod: return add_function_expression<cudf::ast::ast_operator::MOD>(alt, table_src);
    default:
      SIRIUS_LOG_DEBUG("[expression_translator] Unsupported function: {}",
                       static_cast<int>(alt.function()));
      return std::nullopt;
  }
}

//===----------AGGREGATE----------===//
std::optional<expr_ref> gpu_expression_translator::add_expression(
  sirius::ast::aggregate const& /*alt*/, cudf::ast::table_reference const /*table_src*/)
{
  throw not_implemented_exception(
    "[gpu_expression_translator] aggregate nodes cannot be lowered to a cuDF AST (#863).");
}

}  // namespace sirius
