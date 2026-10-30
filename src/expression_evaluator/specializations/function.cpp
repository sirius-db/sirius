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
#include <config.hpp>
#include <expression/ast/node.hpp>
#include <expression/date_trunc_unit.hpp>
#include <expression/function_id.hpp>
#include <expression/substring_slice.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/ast_supported_types.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/like_multiliteral.hpp>
#include <expression_evaluator/regex/cudf_regex_pattern.hpp>
#include <expression_evaluator/regex/regex_playground.hpp>
#include <expression_evaluator/round_floating_point.hpp>
#include <helper/logical_type.hpp>
#include <helper/numeric_narrowing.hpp>
#include <helper/timestamp_semantics.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/assert.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/datetime.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

// rmm
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <ranges>
#include <regex>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

namespace {

/**
 * @brief Read-only preflight for // and %; the caller performs the arithmetic.
 *
 * Safe pairs and pairs containing NULL return normally. A valid zero divisor or
 * signed-minimum/-1 pair throws not_implemented_exception before cuDF arithmetic,
 * requesting runtime CPU replay when enabled or safe query failure otherwise.
 * This guard neither mutates operands nor produces an arithmetic result. Replay
 * supplies DuckDB's errors/settings and guarded CASE semantics; the guard does not.
 */
void require_safe_division(evaluate_result const& left_result,
                           evaluate_result const& right_result,
                           ::cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  if ((!left_result.is_scalar() && left_result.get_column_view().is_empty()) ||
      (!right_result.is_scalar() && right_result.get_column_view().is_empty())) {
    return;
  }
  // Comparisons need a column on one side. Retain only one row for a scalar; the caller
  // keeps the original scalar for cuDF's scalar-column arithmetic overload.
  auto left_scalar        = left_result.is_scalar()
                              ? cudf::make_column_from_scalar(left_result.get_scalar(), 1, stream, mr)
                              : nullptr;
  auto right_scalar       = right_result.is_scalar()
                              ? cudf::make_column_from_scalar(right_result.get_scalar(), 1, stream, mr)
                              : nullptr;
  auto const left         = left_scalar ? left_scalar->view() : left_result.get_column_view();
  auto const right        = right_scalar ? right_scalar->view() : right_result.get_column_view();
  auto const boolean_type = cudf::data_type{cudf::type_id::BOOL8};
  // A one-row predicate represents a scalar operand. Broadcast it with the scalar overload,
  // preserving its validity; scalar_column_view is supported by transform, not binary_operation.
  auto combine = [&](cudf::column_view lhs, cudf::column_view rhs, cudf::binary_operator op) {
    if (lhs.size() == rhs.size()) {
      return cudf::binary_operation(lhs, rhs, op, boolean_type, stream, mr);
    }
    auto const scalar = cudf::get_element(lhs.size() == 1 ? lhs : rhs, 0, stream, mr);
    return lhs.size() == 1 ? cudf::binary_operation(*scalar, rhs, op, boolean_type, stream, mr)
                           : cudf::binary_operation(lhs, *scalar, op, boolean_type, stream, mr);
  };
  auto const zero = cudf::numeric_scalar<int32_t>(0, true, stream, mr);
  auto invalid =
    cudf::binary_operation(right, zero, cudf::binary_operator::EQUAL, boolean_type, stream, mr);

  std::unique_ptr<cudf::scalar> minimum;
  if (left.type().id() == cudf::type_id::DECIMAL128 && left.type().scale() == 0) {
    // HUGEINT uses this physical carrier. SQL DECIMAL(38, 0) cannot reach its minimum.
    minimum = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
      std::numeric_limits<__int128_t>::min(), numeric::scale_type{0}, true, stream, mr);
  } else if (cudf::is_integral_not_bool(left.type()) && !cudf::is_unsigned(left.type())) {
    minimum =
      cudf::type_dispatcher(left.type(), [&]<typename T>() -> std::unique_ptr<cudf::scalar> {
        if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
          return std::make_unique<cudf::numeric_scalar<T>>(
            std::numeric_limits<T>::min(), true, stream, mr);
        } else {
          throw internal_exception("division minimum needs a signed integer type");
        }
      });
  }
  if (minimum) {
    auto const minus_one = cudf::numeric_scalar<int32_t>(-1, true, stream, mr);
    auto is_minimum      = cudf::binary_operation(
      left, *minimum, cudf::binary_operator::EQUAL, boolean_type, stream, mr);
    auto is_minus_one = cudf::binary_operation(
      right, minus_one, cudf::binary_operator::EQUAL, boolean_type, stream, mr);
    auto overflow =
      combine(is_minimum->view(), is_minus_one->view(), cudf::binary_operator::LOGICAL_AND);
    invalid = combine(invalid->view(), overflow->view(), cudf::binary_operator::LOGICAL_OR);
  }
  // A zero divisor is harmless when the numerator is NULL. Right-side NULLs have already
  // propagated through the comparisons; contains ignores every invalid predicate row.
  auto left_valid = cudf::is_valid(left, stream, mr);
  invalid = combine(invalid->view(), left_valid->view(), cudf::binary_operator::LOGICAL_AND);
  auto const true_value = cudf::numeric_scalar<bool>(true, true, stream, mr);
  if (cudf::contains(invalid->view(), true_value, stream)) {
    throw not_implemented_exception(
      "[expression_evaluator:function] Division by zero or signed division overflow; "
      "falling back to DuckDB");
  }
}

// The largest magnitude a HUGEINT result may have; the minimum's 2^127 is excluded with it, since
// DuckDB cannot negate or divide it by -1 either.
constexpr unsigned __int128 hugeint_max_magnitude = std::numeric_limits<__int128_t>::max();

constexpr unsigned __int128 magnitude(__int128_t v)
{
  auto const bits = static_cast<unsigned __int128>(v);
  return v < 0 ? static_cast<unsigned __int128>(0) - bits : bits;
}

// The largest magnitude among an integer type's values; nullopt for any other type.
std::optional<unsigned __int128> integer_type_magnitude(logical_type const& type)
{
  switch (type.id()) {
    case type_id::TINYINT: return magnitude(std::numeric_limits<int8_t>::min());
    case type_id::SMALLINT: return magnitude(std::numeric_limits<int16_t>::min());
    case type_id::INTEGER: return magnitude(std::numeric_limits<int32_t>::min());
    case type_id::BIGINT: return magnitude(std::numeric_limits<int64_t>::min());
    case type_id::UTINYINT: return std::numeric_limits<uint8_t>::max();
    case type_id::USMALLINT: return std::numeric_limits<uint16_t>::max();
    case type_id::UINTEGER: return std::numeric_limits<uint32_t>::max();
    case type_id::UBIGINT: return std::numeric_limits<uint64_t>::max();
    default: return std::nullopt;
  }
}

// The magnitude of an integer or scale-0 DECIMAL128 (HUGEINT) literal; nullopt otherwise.
std::optional<unsigned __int128> constant_magnitude(value const& payload)
{
  return std::visit(
    []<typename T>(T const& v) -> std::optional<unsigned __int128> {
      if constexpr (std::is_same_v<T, null_value>) {
        return 0;
      } else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        if constexpr (std::is_signed_v<T>) {
          return magnitude(v);
        } else {
          return v;
        }
      } else if constexpr (std::is_same_v<T, decimal128>) {
        if (v.scale != 0) { return std::nullopt; }
        return magnitude(v.value);
      } else {
        return std::nullopt;
      }
    },
    payload);
}

// The largest magnitude among @p column's values, 0 when it has none. Synchronizes the stream.
std::optional<unsigned __int128> column_magnitude(cudf::column_view const& column,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  if (column.is_empty() || column.null_count() == column.size()) { return 0; }
  auto const range = compute_exact_numeric_range(column, stream, mr);
  if (!range) { return std::nullopt; }
  return std::max(magnitude(range->minimum), magnitude(range->maximum));
}

// a + b and a * b, or nullopt when the result passes hugeint_max_magnitude.
std::optional<unsigned __int128> bounded_sum(unsigned __int128 a, unsigned __int128 b)
{
  if (a > hugeint_max_magnitude || b > hugeint_max_magnitude - a) { return std::nullopt; }
  return a + b;
}

std::optional<unsigned __int128> bounded_product(unsigned __int128 a, unsigned __int128 b)
{
  if (a > hugeint_max_magnitude || b > hugeint_max_magnitude) { return std::nullopt; }
  if (a != 0 && b > hugeint_max_magnitude / a) { return std::nullopt; }
  return a * b;
}

// Whether a signed integer column holds its type's minimum, whose negation and abs overflow.
// Synchronizes the stream.
bool holds_signed_minimum(cudf::column_view const& column,
                          ::cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr)
{
  if (!cudf::is_integral(column.type()) || cudf::is_unsigned(column.type()) || column.is_empty()) {
    return false;
  }
  auto const lowest =
    cudf::type_dispatcher(column.type(), [&]<typename T>() -> std::unique_ptr<cudf::scalar> {
      if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        return std::make_unique<cudf::numeric_scalar<T>>(
          std::numeric_limits<T>::lowest(), true, stream, mr);
      } else {
        throw internal_exception("holds_signed_minimum needs a signed integer type");
      }
    });
  return cudf::contains(column, *lowest, stream);
}

}  // namespace

std::optional<unsigned __int128> expression_evaluator::hugeint_magnitude_bound(
  sirius::ast::node const& node)
{
  if (node.holds<sirius::ast::constant>()) {
    return constant_magnitude(node.get<sirius::ast::constant>().payload);
  }
  if (node.holds<sirius::ast::cast>()) {
    auto const& cast  = node.get<sirius::ast::cast>();
    auto const source = cast.child->return_type();
    if (auto const bound = integer_type_magnitude(source)) { return bound; }
    if (source.id() == type_id::HUGEINT) { return hugeint_magnitude_bound(*cast.child); }
    return std::nullopt;
  }
  if (node.holds<sirius::ast::reference>()) {
    auto const& ref = node.get<sirius::ast::reference>();
    if (ref.return_type().id() != type_id::HUGEINT) { return std::nullopt; }
    auto const cached = std::ranges::find(_hugeint_reference_bounds,
                                          ref.column_index,
                                          &decltype(_hugeint_reference_bounds)::value_type::first);
    if (cached != _hugeint_reference_bounds.end()) { return cached->second; }
    auto const bound = column_magnitude(_input_table.column(ref.column_index), _stream, _mr);
    _hugeint_reference_bounds.emplace_back(ref.column_index, bound);
    return bound;
  }
  if (node.holds<sirius::ast::function_call>()) {
    auto const& call = node.get<sirius::ast::function_call>();
    auto const& args = call.arguments();
    if (call.return_type().id() != type_id::HUGEINT || args.size() != 2) { return std::nullopt; }
    auto const left = hugeint_magnitude_bound(*args[0]);
    if (!left) { return std::nullopt; }
    switch (call.function()) {
      // |a // b| and |a % b| are at most |a|; evaluation guards zero and signed overflow.
      case function_id::int_div:
      case function_id::mod: return left;
      case function_id::add:
      case function_id::sub:
      case function_id::mul: {
        auto const right = hugeint_magnitude_bound(*args[1]);
        if (!right) { return std::nullopt; }
        return call.function() == function_id::mul ? bounded_product(*left, *right)
                                                   : bounded_sum(*left, *right);
      }
      default: return std::nullopt;
    }
  }
  return std::nullopt;
}

void expression_evaluator::require_hugeint_arithmetic_in_range(
  sirius::ast::function_call const& call, evaluate_result const& left, evaluate_result const& right)
{
  // An operand without a static bound, such as a CASE, is measured after evaluation.
  auto const operand_bound = [&](
                               sirius::ast::node const& node,
                               evaluate_result const& operand) -> std::optional<unsigned __int128> {
    if (auto const bound = hugeint_magnitude_bound(node)) { return bound; }
    if (operand.is_scalar()) {
      auto const column = cudf::make_column_from_scalar(operand.get_scalar(), 1, _stream, _mr);
      return column_magnitude(column->view(), _stream, _mr);
    }
    return column_magnitude(operand.get_column_view(), _stream, _mr);
  };
  auto const& args = call.arguments();
  auto const a     = operand_bound(*args[0], left);
  auto const b     = operand_bound(*args[1], right);
  auto const bound = !a || !b                              ? std::nullopt
                     : call.function() == function_id::mul ? bounded_product(*a, *b)
                                                           : bounded_sum(*a, *b);
  if (!bound) {
    throw not_implemented_exception(
      "[expression_evaluator:function] HUGEINT arithmetic may overflow; falling back to DuckDB");
  }
}

like_multiliteral_cache::entry_ptr const& expression_evaluator::get_or_classify_like(
  std::string_view pattern)
{
  if (auto const found = _like_classifications.find(pattern);
      found != _like_classifications.end()) {
    return found->second;
  }

  auto classification = _like_cache->get_or_classify(pattern);
  ++_like_shared_cache_lookup_count;
  return _like_classifications.emplace(std::string(pattern), std::move(classification))
    .first->second;
}

evaluate_result expression_evaluator::evaluate(sirius::ast::function_call const& alt,
                                               evaluation_mode mode)
{
  auto const resolved_id = alt.function();
  auto const& args       = alt.arguments();

  auto const ast_supported = ast_supports_function(alt);

  auto const ast_op_count = alt.cudf_ast_op_count();

  if (ast_supported && _policy.strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    // Only numeric binary functions are supported in AST currently.
    D_ASSERT(args.size() == 2);

    auto function_type_switch_ast = [](function_id id) -> cudf::ast::ast_operator {
      switch (id) {
        case function_id::add: return cudf::ast::ast_operator::ADD;
        case function_id::sub: return cudf::ast::ast_operator::SUB;
        case function_id::mul: return cudf::ast::ast_operator::MUL;
        case function_id::div:
        case function_id::int_div: return cudf::ast::ast_operator::DIV;
        case function_id::mod: return cudf::ast::ast_operator::MOD;
        default:
          throw invalid_input_exception(
            "[expression_evaluator:function] unsupported AST function id {}", static_cast<int>(id));
      }
    };

    auto left             = evaluate(*args[0], evaluation_mode::AST);
    auto right            = evaluate(*args[1], evaluation_mode::AST);
    auto const& func_expr = _ast_tree.emplace<cudf::ast::operation>(
      function_type_switch_ast(resolved_id), left.get_expr(), right.get_expr());

    if (mode == evaluation_mode::AST) {
      //===----------1: AST Mode----------===//
      return evaluate_result(compose(func_expr, {&left, &right}));
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(func_expr);
    release_temporaries({&left, &right});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with solo operation----------===//
  if (mode == evaluation_mode::AST) {
    auto result = evaluate(alt, evaluation_mode::MATERIALIZE);
    if (!result.is_owned_column()) {
      throw internal_exception(
        "[expression_evaluator:function]: Expected an owned column after executing function "
        "expression.");
    }
    return materialize_as_ast_column(result.release_column());
  }

  // libcudf may return a narrower numeric type than the SQL result (e.g. INT32 for length, INT16
  // for month where DuckDB declares BIGINT). cuDF AST operators and constant IN-lists require the
  // declared type.
  auto result = evaluate_function_kernel(alt);
  if (result.is_scalar()) { return result; }
  auto const declared_type = sirius::require_cudf_type(alt.return_type());
  auto const result_type   = result.get_column_view().type();
  if (result_type == declared_type || !cudf::is_numeric(result_type) ||
      !cudf::is_numeric(declared_type)) {
    return result;
  }
  return evaluate_result(cudf::cast(result.get_column_view(), declared_type, _stream, _mr));
}

evaluate_result expression_evaluator::evaluate_function_kernel(
  sirius::ast::function_call const& alt)
{
  auto const resolved_id = alt.function();
  auto const& args       = alt.arguments();
  auto const output_type = sirius::require_cudf_type(alt.return_type());

  if (auto const date_operation = alt.date_operation(); date_operation.has_value()) {
    auto left               = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    auto right              = evaluate(*args[1], evaluation_mode::MATERIALIZE);
    auto as_transform_input = [&](evaluate_result& value) -> cudf::transform_input {
      if (!value.is_scalar()) { return value.get_column_view(); }
      // Retain the one-row column in the evaluation result while its view is broadcast.
      value = evaluate_result(cudf::make_column_from_scalar(value.get_scalar(), 1, _stream, _mr));
      return cudf::scalar_column_view{value.get_column_view()};
    };
    auto const left_input  = as_transform_input(left);
    auto const right_input = as_transform_input(right);
    return evaluate_result(temporal::date_arithmetic_checked(
      left_input, right_input, *date_operation, _input_table.num_rows(), _stream, _mr));
  }

  //----------Numeric Binary Functions----------//
  auto execute_numeric_binary_func = [&](cudf::binary_operator op) -> evaluate_result {
    auto left = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    // cuDF has no scalar-scalar binary operation; broadcast one side when both are constants.
    auto right = left.is_scalar() ? evaluate_as_column(*args[1])
                                  : evaluate(*args[1], evaluation_mode::MATERIALIZE);
    if (division_requires_check(alt)) { require_safe_division(left, right, _stream, _mr); }
    if (alt.return_type().id() == type_id::HUGEINT &&
        (op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB ||
         op == cudf::binary_operator::MUL)) {
      require_hugeint_arithmetic_in_range(alt, left, right);
    }
    if (left.is_scalar()) {
      return evaluate_result(cudf::binary_operation(
        left.get_scalar(), right.get_column_view(), op, output_type, _stream, _mr));
    }
    if (right.is_scalar()) {
      return evaluate_result(cudf::binary_operation(
        left.get_column_view(), right.get_scalar(), op, output_type, _stream, _mr));
    }
    return evaluate_result(cudf::binary_operation(
      left.get_column_view(), right.get_column_view(), op, output_type, _stream, _mr));
  };
  if (resolved_id == function_id::add) {
    return execute_numeric_binary_func(cudf::binary_operator::ADD);
  }
  if (resolved_id == function_id::sub) {
    return execute_numeric_binary_func(cudf::binary_operator::SUB);
  }
  if (resolved_id == function_id::mul) {
    return execute_numeric_binary_func(cudf::binary_operator::MUL);
  }
  if (resolved_id == function_id::div || resolved_id == function_id::int_div) {
    return execute_numeric_binary_func(cudf::binary_operator::DIV);
  }
  if (resolved_id == function_id::mod) {
    return execute_numeric_binary_func(cudf::binary_operator::MOD);
  }

  //----------Negation----------//
  if (resolved_id == function_id::negate) {
    auto input = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    if (input.is_scalar()) {
      input = evaluate_result(
        cudf::make_column_from_scalar(input.get_scalar(), _input_table.num_rows(), _stream, _mr));
    }
    auto const column = input.get_column_view();
    if (holds_signed_minimum(column, _stream, _mr)) {
      throw invalid_input_exception("Overflow in negation of numeric value!");
    }
    return evaluate_result(
      cudf::unary_operation(column, cudf::unary_operator::NEGATE, _stream, _mr));
  }

  //----------Substring Function----------//
  if (resolved_id == function_id::substring) {
    // DuckDB binds substring as (VARCHAR, BIGINT[, BIGINT]); planning admits only constant bounds
    // that gpu_substring_slice accepts.
    D_ASSERT(args.size() == 2 || args.size() == 3);
    auto const bigint_arg = [&](std::size_t i) {
      D_ASSERT(args[i]->holds<sirius::ast::constant>());
      return std::get<int64_t>(args[i]->get<sirius::ast::constant>().payload);
    };
    auto const offset = bigint_arg(1);
    auto const length = args.size() == 3 ? bigint_arg(2) : substring_default_length;
    auto const slice  = gpu_substring_slice(offset, length);
    if (!slice) {
      throw invalid_input_exception(
        "[expression_evaluator:function] unsupported substring bounds (offset {}, length {})",
        offset,
        length);
    }

    auto input               = evaluate_as_column(*args[0]);
    auto const input_strings = cudf::strings_column_view(input.get_column_view());
    auto result_column       = cudf::strings::slice_strings(
      input_strings, slice->start, slice->stop, std::optional<cudf::size_type>{1}, _stream, _mr);
    return evaluate_result(std::move(result_column));
  }

  //----------String Matching Functions----------//
  auto setup_string_matching = [&]() -> std::pair<evaluate_result, std::string_view> {
    D_ASSERT(args.size() == 2);
    D_ASSERT(args[1]->holds<sirius::ast::constant>());

    auto input            = evaluate_as_column(*args[0]);
    auto const& match_str = std::get<std::string>(args[1]->get<sirius::ast::constant>().payload);
    return {evaluate_result(std::move(input)), std::string_view(match_str)};
  };
  if (resolved_id == function_id::like || resolved_id == function_id::not_like) {
    auto [input, match_str] = setup_string_matching();
    auto const invert       = resolved_id == function_id::not_like;

    // `%lit1%lit2%...%litN%` patterns take a SWAR digram fast path (NOT fused in);
    // everything else — and any ineligible column layout — takes cudf::strings::like.
    if (_like_swar_fastpath) {
      auto const& parsed = get_or_classify_like(match_str);
      if (*parsed) {
        if (auto result_column = like_multiliteral(
              cudf::strings_column_view(input.get_column_view()), **parsed, invert, _stream, _mr)) {
          return evaluate_result(std::move(result_column));
        }
      }
    }

    // cuDF fallback
    auto result_column = cudf::strings::like(cudf::strings_column_view(input.get_column_view()),
                                             std::string_view(match_str),
                                             std::string_view(),
                                             _stream,
                                             _mr);
    if (invert) {
      result_column =
        cudf::unary_operation(result_column->view(), cudf::unary_operator::NOT, _stream, _mr);
    }
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::contains) {
    auto [input, match_str] = setup_string_matching();
    auto result_column = cudf::strings::contains(cudf::strings_column_view(input.get_column_view()),
                                                 cudf::string_scalar(match_str, true, _stream, _mr),
                                                 _stream,
                                                 _mr);
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::prefix) {
    auto [input, match_str] = setup_string_matching();
    auto result_column =
      cudf::strings::starts_with(cudf::strings_column_view(input.get_column_view()),
                                 cudf::string_scalar(match_str, true, _stream, _mr),
                                 _stream,
                                 _mr);
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::suffix) {
    auto [input, match_str] = setup_string_matching();
    auto result_column =
      cudf::strings::ends_with(cudf::strings_column_view(input.get_column_view()),
                               cudf::string_scalar(match_str, true, _stream, _mr),
                               _stream,
                               _mr);
    return evaluate_result(std::move(result_column));
  }

  //----------DateTime Extraction Functions----------//
  auto execute_datetime_extract_func =
    [&](cudf::datetime::datetime_component component) -> evaluate_result {
    D_ASSERT(args.size() == 1);
    auto input = evaluate_as_column(*args[0]);
    auto result_column =
      cudf::datetime::extract_datetime_component(input.get_column_view(), component, _stream, _mr);
    // cuDF returns INT16, while SQL extraction returns BIGINT. Restore the declared type before
    // a parent AST operation or CASE consumes this result; final-output reconciliation is too late.
    if (result_column->type() != output_type) {
      result_column = cudf::cast(result_column->view(), output_type, _stream, _mr);
    }
    return evaluate_result(std::move(result_column));
  };
  if (resolved_id == function_id::year) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::YEAR);
  }
  if (resolved_id == function_id::month) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MONTH);
  }
  if (resolved_id == function_id::day) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::DAY);
  }
  if (resolved_id == function_id::hour) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::HOUR);
  }
  if (resolved_id == function_id::minute) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MINUTE);
  }
  if (resolved_id == function_id::second) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::SECOND);
  }
  if (resolved_id == function_id::millisecond || resolved_id == function_id::microsecond) {
    D_ASSERT(args.size() == 1);
    auto input         = evaluate_as_column(*args[0]);
    auto const extract = [&](cudf::datetime::datetime_component component) {
      return cudf::datetime::extract_datetime_component(
        input.get_column_view(), component, _stream, _mr);
    };

    // cuDF returns separate base-1000 components; DuckDB includes the seconds
    // within the minute. In particular, MICROSECOND excludes whole milliseconds.
    // cuDF's time-of-day decomposition also handles pre-epoch timestamps without
    // a signed epoch remainder. Widen before arithmetic: 59999 does not fit INT16.
    auto const output_type = cudf::data_type{cudf::type_id::INT64};
    cudf::numeric_scalar<int64_t> thousand(1000, true, _stream, _mr);
    auto seconds = extract(cudf::datetime::datetime_component::SECOND);
    auto millis  = extract(cudf::datetime::datetime_component::MILLISECOND);
    auto scaled  = cudf::binary_operation(
      seconds->view(), thousand, cudf::binary_operator::MUL, output_type, _stream, _mr);
    auto result = cudf::binary_operation(
      scaled->view(), millis->view(), cudf::binary_operator::ADD, output_type, _stream, _mr);
    if (resolved_id == function_id::microsecond) {
      auto micros = extract(cudf::datetime::datetime_component::MICROSECOND);
      scaled      = cudf::binary_operation(
        result->view(), thousand, cudf::binary_operator::MUL, output_type, _stream, _mr);
      result = cudf::binary_operation(
        scaled->view(), micros->view(), cudf::binary_operator::ADD, output_type, _stream, _mr);
    }
    auto finite = temporal::finite_mask(input.get_column_view(), _stream, _mr);
    cudf::numeric_scalar<int64_t> null_result(0, false, _stream, _mr);
    result = cudf::copy_if_else(result->view(), null_result, finite->view(), _stream, _mr);
    return evaluate_result(std::move(result));
  }

  //----------Date Truncation Function----------//
  if (resolved_id == function_id::date_trunc) {
    D_ASSERT(args.size() == 2);
    // The first child is the frequency, which should be a constant string
    D_ASSERT(args[0]->holds<sirius::ast::constant>());
    auto const& freq_str = std::get<std::string>(args[0]->get<sirius::ast::constant>().payload);
    auto const unit      = parse_gpu_date_trunc_unit(freq_str);
    if (!unit) {
      throw invalid_input_exception(
        "[expression_evaluator:function] unrecognized/unsupported date_trunc frequency: {}",
        freq_str);
    }
    auto input = evaluate_as_column(*args[1]);

    auto frequency = [](date_trunc_unit unit) -> cudf::datetime::rounding_frequency {
      switch (unit) {
        case date_trunc_unit::day: return cudf::datetime::rounding_frequency::DAY;
        case date_trunc_unit::hour: return cudf::datetime::rounding_frequency::HOUR;
        case date_trunc_unit::minute: return cudf::datetime::rounding_frequency::MINUTE;
        case date_trunc_unit::second: return cudf::datetime::rounding_frequency::SECOND;
        case date_trunc_unit::millisecond: return cudf::datetime::rounding_frequency::MILLISECOND;
        case date_trunc_unit::microsecond: return cudf::datetime::rounding_frequency::MICROSECOND;
      }
      throw internal_exception("[expression_evaluator:function] unknown date_trunc unit");
    };

    auto result_column =
      cudf::datetime::floor_datetimes(input.get_column_view(), frequency(*unit), _stream, _mr);
    return evaluate_result(std::move(result_column));
  }

  //----------String Concatenation----------//
  if (resolved_id == function_id::concat || resolved_id == function_id::concat_operator) {
    // Evaluate every argument to a materialized column or scalar.
    std::vector<evaluate_result> arg_results;
    arg_results.reserve(args.size());
    for (auto const& arg : args) {
      arg_results.push_back(evaluate(*arg, evaluation_mode::MATERIALIZE));
    }

    // cudf::strings::concatenate takes a table_view — scalars must be expanded
    // to full-length columns first.
    auto const num_rows = _input_table.num_rows();
    std::vector<std::unique_ptr<cudf::column>> scalar_cols;
    std::vector<cudf::column_view> col_views;
    col_views.reserve(arg_results.size());
    for (auto const& res : arg_results) {
      if (res.is_scalar()) {
        scalar_cols.push_back(
          cudf::make_column_from_scalar(res.get_scalar(), num_rows, _stream, _mr));
        col_views.push_back(scalar_cols.back()->view());
      } else {
        col_views.push_back(res.get_column_view());
      }
    }

    // DuckDB's concat() ignores NULL arguments (concat(NULL, 'x') = 'x'), whereas
    // the || operator propagates NULL ('a' || NULL = NULL). The narep scalar
    // selects the behaviour: a valid empty string replaces NULLs with "" (ignore),
    // an invalid narep short-circuits the whole row to NULL (propagate).
    bool const propagate_nulls = (resolved_id == function_id::concat_operator);
    cudf::string_scalar narep("", /*is_valid=*/!propagate_nulls, _stream, _mr);
    cudf::table_view concat_table(col_views);
    auto result_column = cudf::strings::concatenate(
      concat_table,
      cudf::string_scalar("", true, _stream, _mr),  // empty separator between parts
      narep,
      cudf::strings::separator_on_nulls::NO,
      _stream,
      _mr);
    return evaluate_result(std::move(result_column));
  }

  //----------Unary Functions----------//
  if (resolved_id == function_id::upper || resolved_id == function_id::lower) {
    D_ASSERT(args.size() == 1);
    auto input         = evaluate_as_column(*args[0]);
    auto const strings = cudf::strings_column_view(input.get_column_view());
    // Use cuDF's Unicode mappings, including multi-character expansions. These can
    // differ from DuckDB's single-code-point mappings (e.g. upper('ß'), lower('İ')).
    auto result_column = resolved_id == function_id::upper
                           ? cudf::strings::to_upper(strings, _stream, _mr)
                           : cudf::strings::to_lower(strings, _stream, _mr);
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::round) {
    // Planning admits FLOAT and DOUBLE inputs with an optional constant INTEGER precision.
    D_ASSERT(args.size() == 1 || args.size() == 2);
    int32_t precision = 0;
    if (args.size() == 2) {
      D_ASSERT(args[1]->holds<sirius::ast::constant>());
      precision = std::get<int32_t>(args[1]->get<sirius::ast::constant>().payload);
    }
    auto input = evaluate_as_column(*args[0]);
    return evaluate_result(round_floating_point(input.get_column_view(), precision, _stream, _mr));
  }
  if (resolved_id == function_id::abs) {
    // Planning admits FLOAT, DOUBLE, DECIMAL and integer inputs other than HUGEINT.
    D_ASSERT(args.size() == 1);
    auto input = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    if (input.is_scalar()) {
      input = evaluate_result(
        cudf::make_column_from_scalar(input.get_scalar(), _input_table.num_rows(), _stream, _mr));
    }
    // cuDF wraps abs of a signed minimum; fall back so DuckDB raises its overflow error.
    auto const column = input.get_column_view();
    if (holds_signed_minimum(column, _stream, _mr)) {
      throw not_implemented_exception(
        "[expression_evaluator:function] abs overflows for the smallest value of a signed "
        "integer type, expected to fall back to CPU execution");
    }
    return evaluate_result(cudf::unary_operation(column, cudf::unary_operator::ABS, _stream, _mr));
  }
  if (resolved_id == function_id::strlen) {
    D_ASSERT(args.size() == 1);
    auto input = evaluate_as_column(*args[0]);
    auto result_column =
      cudf::strings::count_bytes(cudf::strings_column_view(input.get_column_view()), _stream, _mr);
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::length) {
    D_ASSERT(args.size() == 1);
    auto input         = evaluate_as_column(*args[0]);
    auto result_column = cudf::strings::count_characters(
      cudf::strings_column_view(input.get_column_view()), _stream, _mr);
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::regexp_replace) {
    // The input should be <input column, pattern string scalar, replace string scalar>
    D_ASSERT(args.size() == 3);
    D_ASSERT(args[1]->holds<sirius::ast::constant>());
    D_ASSERT(args[2]->holds<sirius::ast::constant>());

    auto input = evaluate_as_column(*args[0]);

    auto const& pattern_str = std::get<std::string>(args[1]->get<sirius::ast::constant>().payload);
    auto const& replace_str = std::get<std::string>(args[2]->get<sirius::ast::constant>().payload);
    // The planner validated the RE2 pattern; give cuDF RE2's end-of-text anchor semantics.
    auto const cudf_pattern = ::sirius::regex::to_cudf_regex_pattern(pattern_str);
    auto const has_backrefs = std::regex_search(replace_str, std::regex(R"(\\[0-9])"));
    if (has_backrefs) {
      if (_policy.regex_jit) {
        if (pattern_str == R"(^https?://(?:www\.)?([^/]+)/.*$)" && replace_str == R"(\1)") {
          return ::sirius::regex::regex_playground::jit_transform_clickbench_q28_regex(
            input.get_column_view(), _stream, _mr);
        }
      }
      // replace_with_backrefs has no replacement-count parameter. Anchor a lazy prefix to
      // the beginning of the input so the program can match only once, at the first original
      // match. Restore the prefix in the replacement; cuDF preserves the unmatched suffix.
      // [\s\S] includes newlines without changing the original pattern's dot semantics.
      // Group 1 is the prefix, group 2 is the whole original match (RE2's \0), and original
      // capture groups move by two. Braced references avoid treating e.g. \12 as group 12:
      // DuckDB interprets it as group 1 followed by the literal character '2'.
      auto const first_pattern      = std::string{"^([\\s\\S]*?)("} + cudf_pattern + ")";
      auto regex_prog               = cudf::strings::regex_program::create(first_pattern);
      std::string first_replacement = "${1}";
      for (std::size_t i = 0; i < replace_str.size(); ++i) {
        if (replace_str[i] == '\\') {
          auto const group = replace_str[++i] - '0';
          first_replacement += "${" + std::to_string(group + 2) + "}";
        } else {
          first_replacement += replace_str[i];
        }
      }
      return cudf::strings::replace_with_backrefs(
        cudf::strings_column_view(input.get_column_view()),
        *regex_prog,
        first_replacement,
        _stream,
        _mr);
    } else {
      auto regex_prog = cudf::strings::regex_program::create(cudf_pattern);
      return cudf::strings::replace_re(cudf::strings_column_view(input.get_column_view()),
                                       *regex_prog,
                                       cudf::string_scalar(replace_str, true, _stream, _mr),
                                       1,
                                       _stream,
                                       _mr);
    }
  }

  //----------Struct Functions----------//
  // row() and struct_pack() both construct a struct column from their child expressions.
  // row() is used by DuckDB for tuple constructors like (col1, col2).
  if (resolved_id == function_id::row || resolved_id == function_id::struct_pack) {
    D_ASSERT(!args.empty());
    std::vector<std::unique_ptr<cudf::column>> child_cols;
    for (const auto& a : args) {
      auto result = evaluate(*a, evaluation_mode::MATERIALIZE);
      if (result.is_scalar()) {
        child_cols.push_back(cudf::make_column_from_scalar(
          result.get_scalar(), _input_table.num_rows(), _stream, _mr));
      } else {
        child_cols.push_back(
          std::make_unique<cudf::column>(result.get_column_view(), _stream, _mr));
      }
    }
    auto const num_rows = child_cols[0]->size();
    return cudf::make_structs_column(num_rows,
                                     std::move(child_cols),
                                     0,
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                     _stream,
                                     _mr);
  }

  //----------constant_or_null----------//
  // constant_or_null(c, a1, ..., an) is c in every row where all of a1..an are non-NULL and
  // NULL in the other rows.
  if (resolved_id == function_id::constant_or_null) {
    D_ASSERT(args.size() >= 2);
    auto const num_rows = _input_table.num_rows();
    auto null_mask      = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, _stream, _mr);
    cudf::size_type null_count = 0;
    {
      bool any_null_scalar = false;
      std::vector<evaluate_result> column_results;
      std::vector<cudf::column_view> column_views;
      column_results.reserve(args.size() - 1);
      column_views.reserve(args.size() - 1);
      for (std::size_t i = 1; i < args.size(); ++i) {
        auto result = evaluate(*args[i], evaluation_mode::MATERIALIZE);
        if (result.is_scalar()) {
          any_null_scalar |= !result.get_scalar().is_valid(_stream);
        } else {
          column_views.push_back(result.get_column_view());
          column_results.push_back(std::move(result));
        }
      }
      if (any_null_scalar) {
        null_mask  = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, _stream, _mr);
        null_count = num_rows;
      } else if (!column_views.empty()) {
        std::tie(null_mask, null_count) =
          cudf::bitmask_and(cudf::table_view(column_views), _stream, _mr);
      }
    }

    auto const value = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    D_ASSERT(value.is_scalar());
    auto result_column = cudf::make_column_from_scalar(value.get_scalar(), num_rows, _stream, _mr);
    if (null_count > 0) {
      result_column->set_null_mask(std::move(null_mask), null_count);
      if (!cudf::is_fixed_width(result_column->type())) {
        // Rows that become NULL still hold the constant's payload, which cuDF expects to be empty.
        result_column = cudf::purge_nonempty_nulls(result_column->view(), _stream, _mr);
      }
    }
    return evaluate_result(std::move(result_column));
  }

  // `error()` is a runtime-error-raising function. We deliberately do not
  // evaluate it on-device here; the upstream fallback path is expected to
  // handle it on the CPU.
  if (resolved_id == function_id::error) {
    throw not_implemented_exception(
      "[expression_evaluator:function] error() is not dispatched on GPU; "
      "expected to fall back to CPU execution");
  }

  // Invariant violation: alt.function() returned a valid function_id but no
  // dispatch arm above claimed it. This means an entry was added to
  // function_id without a corresponding GPU handler here.
  throw internal_exception(
    "[expression_evaluator:function]: registered function_id has no GPU dispatch arm: id={}",
    static_cast<int>(resolved_id));
}

}  // namespace sirius
