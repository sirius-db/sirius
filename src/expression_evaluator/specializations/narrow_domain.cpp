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

// sirius
#include <expression/ast/node.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>

// standard library
#include <optional>
#include <variant>

namespace sirius {
using ast_result      = expression_evaluator::ast_result;
using evaluate_result = expression_evaluator::evaluate_result;
using evaluation_mode = expression_evaluator::evaluation_mode;

namespace {

/// Host-exact value of a numeric constant as a degenerate [v, v] range, or nullopt when the
/// constant is not a supported narrowable-domain literal (or its payload alternative disagrees
/// with its declared logical type). Typed NULLs return nullopt here; callers treat them as
/// representable in any carrier because only their validity bit is materialized.
std::optional<numeric_range> constant_numeric_range(sirius::ast::constant const& expr)
{
  auto const& payload = expr.payload;
  switch (expr.return_type().id()) {
    case type_id::TINYINT:
      if (auto const* v = std::get_if<int8_t>(&payload)) { return signed_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::SMALLINT:
      if (auto const* v = std::get_if<int16_t>(&payload)) { return signed_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::INTEGER:
      if (auto const* v = std::get_if<int32_t>(&payload)) { return signed_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::BIGINT:
      if (auto const* v = std::get_if<int64_t>(&payload)) { return signed_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::UTINYINT:
      if (auto const* v = std::get_if<uint8_t>(&payload)) { return unsigned_integer_range(*v, *v); }
      return std::nullopt;
    case type_id::USMALLINT:
      if (auto const* v = std::get_if<uint16_t>(&payload)) {
        return unsigned_integer_range(*v, *v);
      }
      return std::nullopt;
    case type_id::UINTEGER:
      if (auto const* v = std::get_if<uint32_t>(&payload)) {
        return unsigned_integer_range(*v, *v);
      }
      return std::nullopt;
    case type_id::UBIGINT:
      if (auto const* v = std::get_if<uint64_t>(&payload)) {
        return unsigned_integer_range(*v, *v);
      }
      return std::nullopt;
    case type_id::DECIMAL: {
      auto const scale = expr.return_type().decimal_scale();
      if (auto const* v = std::get_if<sirius::decimal32>(&payload)) {
        return decimal_range(v->value, v->value, scale);
      }
      if (auto const* v = std::get_if<sirius::decimal64>(&payload)) {
        return decimal_range(v->value, v->value, scale);
      }
      if (auto const* v = std::get_if<sirius::decimal128>(&payload)) {
        return decimal_range(v->value, v->value, scale);
      }
      return std::nullopt;
    }
    default: return std::nullopt;
  }
}

/// Raw unscaled/integral host value of a validated narrowable-domain constant. Callers must have
/// established representability through constant_numeric_range + numeric_range_fits.
__int128_t constant_host_value(sirius::ast::constant const& expr)
{
  auto const range = constant_numeric_range(expr);
  if (!range) {
    throw internal_exception(
      "[expression_evaluator:narrow_domain] constant lost its numeric payload between "
      "representability check and scalar creation");
  }
  return range->minimum;
}

}  // namespace

std::optional<cudf::data_type> expression_evaluator::narrow_domain_carrier(
  sirius::ast::node const& column_operand,
  std::initializer_list<sirius::ast::node const*> constant_operands) const
{
  if (!column_operand.holds<sirius::ast::reference>()) { return std::nullopt; }
  auto const& ref     = column_operand.get<sirius::ast::reference>();
  auto const& logical = ref.return_type();
  if (!is_narrowable_numeric_type(logical)) { return std::nullopt; }
  if (ref.column_index >= static_cast<std::uint32_t>(_input_table.num_columns())) {
    return std::nullopt;
  }
  auto const carrier = _input_table.column(ref.column_index).type();
  if (!can_restore_to(carrier, sirius::get_cudf_type(logical))) { return std::nullopt; }

  for (auto const* operand : constant_operands) {
    if (!operand || !operand->holds<sirius::ast::constant>()) { return std::nullopt; }
    auto const& constant = operand->get<sirius::ast::constant>();
    // A typed NULL materializes only a validity bit, so it fits any carrier.
    if (std::holds_alternative<sirius::null_value>(constant.payload)) { continue; }
    auto const range = constant_numeric_range(constant);
    if (!range || !numeric_range_fits(carrier, *range)) { return std::nullopt; }
  }
  return carrier;
}

evaluate_result expression_evaluator::evaluate_narrow_domain_operand(
  sirius::ast::node const& operand,
  std::optional<cudf::data_type> narrow_carrier,
  evaluation_mode mode)
{
  if (!narrow_carrier) { return evaluate(operand, mode); }
  if (operand.holds<sirius::ast::reference>()) {
    // The reference passes through as its raw input column, bypassing restoration.
    auto const& ref = operand.get<sirius::ast::reference>();
    if (mode == evaluation_mode::AST) {
      auto const& col_expr = _ast_tree.emplace<cudf::ast::column_reference>(ref.column_index);
      return evaluate_result(ast_result(col_expr));
    }
    return evaluate_result(_input_table.column(ref.column_index));
  }
  return evaluate_constant_in_carrier(operand.get<sirius::ast::constant>(), *narrow_carrier, mode);
}

evaluate_result expression_evaluator::evaluate_constant_in_carrier(
  sirius::ast::constant const& expr, cudf::data_type carrier, evaluation_mode mode)
{
  bool const is_valid = !std::holds_alternative<sirius::null_value>(expr.payload);
  auto const value    = is_valid ? constant_host_value(expr) : __int128_t{};

  switch (carrier.id()) {
    case cudf::type_id::INT8: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int8_t>>(
        static_cast<int8_t>(value), is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::INT16: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int16_t>>(
        static_cast<int16_t>(value), is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::INT32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int32_t>>(
        static_cast<int32_t>(value), is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::UINT8: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint8_t>>(
        static_cast<uint8_t>(value), is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::UINT16: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint16_t>>(
        static_cast<uint16_t>(value), is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::UINT32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint32_t>>(
        static_cast<uint32_t>(value), is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::DECIMAL32: {
      auto scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        static_cast<int32_t>(value), numeric::scale_type{carrier.scale()}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::DECIMAL64: {
      auto scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        static_cast<int64_t>(value), numeric::scale_type{carrier.scale()}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    default:
      // narrow_domain_carrier only yields strict narrowings, so INT64/UINT64/DECIMAL128 (and any
      // non-numeric carrier) cannot reach a constant conversion.
      throw internal_exception(
        "[expression_evaluator:narrow_domain] unsupported narrow constant carrier: {}",
        cudf::type_to_name(carrier));
  }
}

}  // namespace sirius
