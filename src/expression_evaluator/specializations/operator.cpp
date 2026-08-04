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
#include <expression/ast/node.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <helper/logical_type.hpp>
#include <log/logging.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/assert.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/search.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/timestamps.hpp>

// rmm
#include <rmm/device_uvector.hpp>

// standard library
#include <algorithm>
#include <string>
#include <variant>

namespace {

// Read a typed constant from one of the values in a sirius::ast::in_list. Assumes
// the AST translator has guaranteed every entry holds sirius::ast::constant.
template <typename T>
T const& get_const_payload(::sirius::ast::node const& n)
{
  auto const& c = n.get<::sirius::ast::constant>();
  return std::get<T>(c.payload);
}

template <typename T>
std::unique_ptr<cudf::column> execute_numeric_in_ast(const ::sirius::ast::in_list& alt,
                                                     const cudf::column_view& input_view,
                                                     rmm::device_async_resource_ref mr,
                                                     rmm::cuda_stream_view stream)
{
  std::vector<T> children_vals;
  children_vals.reserve(alt.values.size());
  for (auto const& v : alt.values) {
    children_vals.push_back(get_const_payload<T>(*v));
  }
  rmm::device_uvector<T> children_vals_d(children_vals.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(children_vals_d.data(),
                                children_vals.data(),
                                children_vals.size() * sizeof(T),
                                cudaMemcpyHostToDevice,
                                stream));
  cudf::column_view children_view(input_view.type(),
                                  static_cast<cudf::size_type>(children_vals.size()),
                                  children_vals_d.data(),
                                  nullptr,
                                  0,
                                  0);
  return cudf::contains(children_view, input_view, stream, mr);
}

// For decimal (fixed_point) types. Reads the rep directly from sirius::decimal{32,64,128}.
template <typename DecimalT>
std::unique_ptr<cudf::column> execute_decimal_in_ast(const ::sirius::ast::in_list& alt,
                                                     const cudf::column_view& input_view,
                                                     rmm::device_async_resource_ref mr,
                                                     rmm::cuda_stream_view stream)
{
  using Rep = typename DecimalT::rep;
  std::vector<Rep> children_vals;
  children_vals.reserve(alt.values.size());
  for (auto const& v : alt.values) {
    auto const& payload = v->get<::sirius::ast::constant>().payload;
    if constexpr (std::is_same_v<DecimalT, numeric::decimal32>) {
      children_vals.push_back(std::get<::sirius::decimal32>(payload).value);
    } else if constexpr (std::is_same_v<DecimalT, numeric::decimal64>) {
      children_vals.push_back(std::get<::sirius::decimal64>(payload).value);
    } else if constexpr (std::is_same_v<DecimalT, numeric::decimal128>) {
      children_vals.push_back(std::get<::sirius::decimal128>(payload).value);
    } else {
      static_assert(sizeof(DecimalT) == 0, "Unsupported decimal type");
    }
  }
  rmm::device_uvector<Rep> children_vals_d(children_vals.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(children_vals_d.data(),
                                children_vals.data(),
                                children_vals.size() * sizeof(Rep),
                                cudaMemcpyHostToDevice,
                                stream));
  // input_view.type() carries the scale, so the haystack column matches the needle's type.
  cudf::column_view children_view(input_view.type(),
                                  static_cast<cudf::size_type>(children_vals.size()),
                                  children_vals_d.data(),
                                  nullptr,
                                  0,
                                  0);
  return cudf::contains(children_view, input_view, stream, mr);
}

// For timestamp types. Pulls the underlying integral count out of the matching sirius wrapper.
template <typename CudfTimestampT>
std::unique_ptr<cudf::column> execute_timestamp_in_ast(const ::sirius::ast::in_list& alt,
                                                       const cudf::column_view& input_view,
                                                       rmm::device_async_resource_ref mr,
                                                       rmm::cuda_stream_view stream)
{
  using Rep = typename CudfTimestampT::rep;
  std::vector<Rep> children_vals;
  children_vals.reserve(alt.values.size());
  for (auto const& v : alt.values) {
    auto const& payload = v->get<::sirius::ast::constant>().payload;
    if constexpr (std::is_same_v<CudfTimestampT, cudf::timestamp_D>) {
      children_vals.push_back(std::get<::sirius::date_value>(payload).days);
    } else if constexpr (std::is_same_v<CudfTimestampT, cudf::timestamp_s>) {
      children_vals.push_back(std::get<::sirius::timestamp_sec_value>(payload).value);
    } else if constexpr (std::is_same_v<CudfTimestampT, cudf::timestamp_ms>) {
      children_vals.push_back(std::get<::sirius::timestamp_ms_value>(payload).value);
    } else if constexpr (std::is_same_v<CudfTimestampT, cudf::timestamp_us>) {
      children_vals.push_back(std::get<::sirius::timestamp_us_value>(payload).value);
    } else if constexpr (std::is_same_v<CudfTimestampT, cudf::timestamp_ns>) {
      children_vals.push_back(std::get<::sirius::timestamp_ns_value>(payload).value);
    } else {
      static_assert(sizeof(CudfTimestampT) == 0, "Unsupported cudf timestamp type");
    }
  }
  rmm::device_uvector<Rep> children_vals_d(children_vals.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(children_vals_d.data(),
                                children_vals.data(),
                                children_vals.size() * sizeof(Rep),
                                cudaMemcpyHostToDevice,
                                stream));
  cudf::column_view children_view(input_view.type(),
                                  static_cast<cudf::size_type>(children_vals.size()),
                                  children_vals_d.data(),
                                  nullptr,
                                  0,
                                  0);
  return cudf::contains(children_view, input_view, stream, mr);
}

// BOOL8 has the cudf storage type uint8_t but the AST constant payload variant
// holds the value as `bool`. Use a separate helper that reads `bool` from the
// variant and uploads as uint8_t (matching the cudf BOOL8 storage convention).
std::unique_ptr<cudf::column> execute_bool_in_ast(const ::sirius::ast::in_list& alt,
                                                  const cudf::column_view& input_view,
                                                  rmm::device_async_resource_ref mr,
                                                  rmm::cuda_stream_view stream)
{
  std::vector<uint8_t> children_vals;
  children_vals.reserve(alt.values.size());
  for (auto const& v : alt.values) {
    children_vals.push_back(get_const_payload<bool>(*v) ? uint8_t{1} : uint8_t{0});
  }
  rmm::device_uvector<uint8_t> children_vals_d(children_vals.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(children_vals_d.data(),
                                children_vals.data(),
                                children_vals.size() * sizeof(uint8_t),
                                cudaMemcpyHostToDevice,
                                stream));
  cudf::column_view children_view(input_view.type(),
                                  static_cast<cudf::size_type>(children_vals.size()),
                                  children_vals_d.data(),
                                  nullptr,
                                  0,
                                  0);
  return cudf::contains(children_view, input_view, stream, mr);
}

std::unique_ptr<cudf::column> execute_string_in_ast(const ::sirius::ast::in_list& alt,
                                                    const cudf::column_view& input_view,
                                                    rmm::device_async_resource_ref mr,
                                                    rmm::cuda_stream_view stream)
{
  auto const num_strings = static_cast<cudf::size_type>(alt.values.size());
  auto const num_offsets = num_strings + 1;

  std::vector<char> chars;
  std::vector<cudf::size_type> offsets;
  cudf::size_type offset = 0;
  for (auto const& v : alt.values) {
    auto const& child_string = get_const_payload<std::string>(*v);
    chars.insert(chars.end(), child_string.begin(), child_string.end());
    offsets.push_back(offset);
    offset += static_cast<cudf::size_type>(child_string.size());
  }
  offsets.push_back(offset);

  rmm::device_uvector<char> chars_buffer(offset, stream, mr);
  rmm::device_uvector<cudf::size_type> offsets_buffer(num_offsets, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(chars_buffer.data(),
                                chars.data(),
                                chars.size() * sizeof(char),
                                cudaMemcpyHostToDevice,
                                stream));
  CUDF_CUDA_TRY(cudaMemcpyAsync(offsets_buffer.data(),
                                offsets.data(),
                                offsets.size() * sizeof(cudf::size_type),
                                cudaMemcpyHostToDevice,
                                stream));

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                                    num_offsets,
                                                    std::move(offsets_buffer).release(),
                                                    rmm::device_buffer{},
                                                    0);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(offsets_col));
  auto in_strings_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                       num_strings,
                                                       std::move(chars_buffer).release(),
                                                       rmm::device_buffer{},
                                                       0,
                                                       std::move(children));

  return cudf::contains(in_strings_col->view(), input_view, stream, mr);
}

}  // namespace

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

//===----------------------------------------------------------------------===//
// unary_op
//===----------------------------------------------------------------------===//

evaluate_result expression_evaluator::evaluate(sirius::ast::unary_op const& alt,
                                               evaluation_mode mode)
{
  if (alt.op == sirius::ast::unary_op::kind::op_try) {
    throw not_implemented_exception(
      "[expression_evaluator] evaluate called on an unsupported TRY operator expression.");
  }

  auto const ast_op_count = alt.cudf_ast_op_count();

  if (_strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    auto child        = evaluate(*alt.child, evaluation_mode::AST);
    auto build_output = [&](expr_ref const& produced) -> evaluate_result {
      return evaluate_result(compose(produced, {&child}));
    };

    evaluate_result output = [&]() -> evaluate_result {
      switch (alt.op) {
        case sirius::ast::unary_op::kind::op_not: {
          auto const& not_expr =
            _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, child.get_expr());
          return build_output(not_expr);
        }
        case sirius::ast::unary_op::kind::op_is_null: {
          auto const& is_null_expr = _ast_tree.emplace<cudf::ast::operation>(
            cudf::ast::ast_operator::IS_NULL, child.get_expr());
          return build_output(is_null_expr);
        }
        case sirius::ast::unary_op::kind::op_is_not_null: {
          auto const& is_null_expr = _ast_tree.emplace<cudf::ast::operation>(
            cudf::ast::ast_operator::IS_NULL, child.get_expr());
          auto const& not_expr =
            _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, is_null_expr);
          return build_output(not_expr);
        }
        default:
          throw internal_exception(
            "[expression_evaluator] AST mode with unknown/unsupported unary_op kind: {}",
            static_cast<int>(alt.op));
      }
    }();

    if (mode == evaluation_mode::AST) {
      //===----------1: AST Mode----------===//
      return output;
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(output.get_expr());
    release_temporaries({&output});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  if (mode == evaluation_mode::AST) {
    auto result = evaluate(alt, evaluation_mode::MATERIALIZE);
    if (!result.is_owned_column()) {
      throw internal_exception(
        "[expression_evaluator:operator]: Expected an owned column after executing operator "
        "expression.");
    }
    return materialize_as_ast_column(result.release_column());
  }

  auto child = evaluate(*alt.child, evaluation_mode::MATERIALIZE);
  switch (alt.op) {
    case sirius::ast::unary_op::kind::op_not:
      return evaluate_result(
        cudf::unary_operation(child.get_column_view(), cudf::unary_operator::NOT, _stream, _mr));
    case sirius::ast::unary_op::kind::op_is_null:
      return evaluate_result(cudf::is_null(child.get_column_view(), _stream, _mr));
    case sirius::ast::unary_op::kind::op_is_not_null: {
      auto is_null_result = cudf::is_null(child.get_column_view(), _stream, _mr);
      return evaluate_result(
        cudf::unary_operation(is_null_result->view(), cudf::unary_operator::NOT, _stream, _mr));
    }
    default:
      throw internal_exception(
        "[expression_evaluator] evaluate called on an operator expression with "
        "unknown/unsupported unary_op kind: {}",
        static_cast<int>(alt.op));
  }
}

//===----------------------------------------------------------------------===//
// in_list
//===----------------------------------------------------------------------===//

evaluate_result expression_evaluator::evaluate(sirius::ast::in_list const& alt,
                                               evaluation_mode mode)
{
  if (alt.values.empty()) {
    throw invalid_input_exception(
      "[expression_evaluator:in_list] in_list has no values — malformed AST.");
  }
  auto const ast_op_count = alt.cudf_ast_op_count();

  if (_strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    D_ASSERT(!alt.values.empty());

    auto test                = evaluate(*alt.probe, evaluation_mode::AST);
    auto comparator          = evaluate(*alt.values[0], evaluation_mode::AST);
    expr_ref comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
      cudf::ast::ast_operator::EQUAL, test.get_expr(), comparator.get_expr());
    auto output = evaluate_result(compose(comparison_expr, {&test, &comparator}));

    for (std::size_t value_idx = 1; value_idx < alt.values.size(); ++value_idx) {
      auto next_comparator          = evaluate(*alt.values[value_idx], evaluation_mode::AST);
      expr_ref next_comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::EQUAL, test.get_expr(), next_comparator.get_expr());
      comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LOGICAL_OR, comparison_expr, next_comparison_expr);
      output = evaluate_result(compose(comparison_expr, {&output, &next_comparator}));
    }

    if (!alt.negated) {
      // produce IN result (positive)
      if (mode == evaluation_mode::AST) { return output; }
      auto result_column = evaluate_ast(output.get_expr());
      release_temporaries({&output});
      return evaluate_result(std::move(result_column));
    }
    auto const& not_expr =
      _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, comparison_expr);
    auto not_output = evaluate_result(compose(not_expr, {&output}));

    if (mode == evaluation_mode::AST) {
      //===----------1: AST Mode----------===//
      return not_output;
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(not_output.get_expr());
    release_temporaries({&not_output});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  if (mode == evaluation_mode::AST) {
    auto result = evaluate(alt, evaluation_mode::MATERIALIZE);
    if (!result.is_owned_column()) {
      throw internal_exception(
        "[expression_evaluator:operator]: Expected an owned column after executing operator "
        "expression.");
    }
    return materialize_as_ast_column(result.release_column());
  }

  D_ASSERT(!alt.values.empty());
  auto test = evaluate(*alt.probe, evaluation_mode::MATERIALIZE);
  D_ASSERT(!test.is_scalar());  // IN with scalar LHS should have been already resolved

  // Optimization: special handling when every haystack entry is a constant.
  auto const all_constants = std::all_of(alt.values.begin(), alt.values.end(), [](auto const& v) {
    return v && v->template holds<sirius::ast::constant>();
  });

  if (all_constants) {
    std::unique_ptr<cudf::column> contains_column;
    switch (test.get_column_view().type().id()) {
      case cudf::type_id::INT8:
        contains_column = execute_numeric_in_ast<int8_t>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::INT16:
        contains_column =
          execute_numeric_in_ast<int16_t>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::INT32:
        contains_column =
          execute_numeric_in_ast<int32_t>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::INT64:
        contains_column =
          execute_numeric_in_ast<int64_t>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::UINT8:
        contains_column =
          execute_numeric_in_ast<uint8_t>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::UINT16:
        contains_column =
          execute_numeric_in_ast<uint16_t>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::UINT32:
        contains_column =
          execute_numeric_in_ast<uint32_t>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::UINT64:
        contains_column =
          execute_numeric_in_ast<uint64_t>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::FLOAT32:
        contains_column = execute_numeric_in_ast<float>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::FLOAT64:
        contains_column = execute_numeric_in_ast<double>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::BOOL8:
        contains_column = execute_bool_in_ast(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::DECIMAL32:
        contains_column =
          execute_decimal_in_ast<numeric::decimal32>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::DECIMAL64:
        contains_column =
          execute_decimal_in_ast<numeric::decimal64>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::DECIMAL128:
        contains_column =
          execute_decimal_in_ast<numeric::decimal128>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::TIMESTAMP_DAYS:
        contains_column =
          execute_timestamp_in_ast<cudf::timestamp_D>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::TIMESTAMP_SECONDS:
        contains_column =
          execute_timestamp_in_ast<cudf::timestamp_s>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::TIMESTAMP_MILLISECONDS:
        contains_column =
          execute_timestamp_in_ast<cudf::timestamp_ms>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::TIMESTAMP_MICROSECONDS:
        contains_column =
          execute_timestamp_in_ast<cudf::timestamp_us>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::TIMESTAMP_NANOSECONDS:
        contains_column =
          execute_timestamp_in_ast<cudf::timestamp_ns>(alt, test.get_column_view(), _mr, _stream);
        break;
      case cudf::type_id::STRING:
        contains_column = execute_string_in_ast(alt, test.get_column_view(), _mr, _stream);
        break;
      default:
        throw not_implemented_exception(
          "[expression_evaluator] evaluate IN called with unsupported scalar haystack type {}",
          static_cast<int>(test.get_column_view().type().id()));
    }
    if (alt.negated) {
      contains_column =
        cudf::unary_operation(contains_column->view(), cudf::unary_operator::NOT, _stream, _mr);
    }
    return evaluate_result(std::move(contains_column));
  }

  // Some haystack referent is not a scalar — fall back to a chained OR of EQUAL comparisons.
  auto const output_type = cudf::data_type{cudf::type_id::BOOL8};
  auto comparator        = evaluate(*alt.values[0], evaluation_mode::MATERIALIZE);
  auto comparison_column = cudf::binary_operation(test.get_column_view(),
                                                  comparator.get_column_view(),
                                                  cudf::binary_operator::EQUAL,
                                                  output_type,
                                                  _stream,
                                                  _mr);
  auto output            = evaluate_result(std::move(comparison_column));

  for (std::size_t value_idx = 1; value_idx < alt.values.size(); ++value_idx) {
    auto next_comparator = evaluate(*alt.values[value_idx], evaluation_mode::MATERIALIZE);
    auto next_eq_column  = cudf::binary_operation(test.get_column_view(),
                                                 next_comparator.get_column_view(),
                                                 cudf::binary_operator::EQUAL,
                                                 output_type,
                                                 _stream,
                                                 _mr);
    auto combined        = cudf::binary_operation(output.get_column_view(),
                                           next_eq_column->view(),
                                           cudf::binary_operator::LOGICAL_OR,
                                           output_type,
                                           _stream,
                                           _mr);
    output               = evaluate_result(std::move(combined));
  }

  if (!alt.negated) { return output; }
  auto not_column =
    cudf::unary_operation(output.get_column_view(), cudf::unary_operator::NOT, _stream, _mr);
  return evaluate_result(std::move(not_column));
}

//===----------------------------------------------------------------------===//
// coalesce — always materializes; no AST representation.
//===----------------------------------------------------------------------===//

evaluate_result expression_evaluator::evaluate(sirius::ast::coalesce const& alt,
                                               evaluation_mode mode)
{
  D_ASSERT(alt.children.size() > 1);

  auto current_result = evaluate(*alt.children[0], evaluation_mode::MATERIALIZE);
  if (current_result.is_scalar()) {
    current_result = evaluate_result(cudf::make_column_from_scalar(
      current_result.get_scalar(), _input_table.num_rows(), _stream, _mr));
  }

  for (std::size_t child_idx = 1; child_idx < alt.children.size(); ++child_idx) {
    // Short-circuit if the null_count is zero
    if (!current_result.get_column_view().has_nulls()) { break; }
    auto replacement = evaluate(*alt.children[child_idx], evaluation_mode::MATERIALIZE);
    if (replacement.is_scalar()) {
      current_result = evaluate_result(cudf::replace_nulls(
        current_result.get_column_view(), replacement.get_scalar(), _stream, _mr));
    } else {
      current_result = evaluate_result(cudf::replace_nulls(
        current_result.get_column_view(), replacement.get_column_view(), _stream, _mr));
    }
  }

  if (mode == evaluation_mode::AST) {
    if (!current_result.is_owned_column()) {
      throw internal_exception(
        "[expression_evaluator:operator]: Expected an owned column after executing COALESCE "
        "expression.");
    }
    return materialize_as_ast_column(current_result.release_column());
  }
  return current_result;
}

}  // namespace sirius
