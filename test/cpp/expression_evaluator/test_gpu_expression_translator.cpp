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

// test
#include "ast_test_support.hpp"

#include <catch.hpp>

// sirius — node-construction surface (build every test expression as a
// sirius::ast::node directly; no DuckDB Bound*Expression in the test body).
#include <expression/ast/between.hpp>
#include <expression/ast/case_expr.hpp>
#include <expression/ast/cast.hpp>
#include <expression/ast/coalesce.hpp>
#include <expression/ast/comparison.hpp>
#include <expression/ast/conjunction.hpp>
#include <expression/ast/constant.hpp>
#include <expression/ast/function_call.hpp>
#include <expression/ast/in_list.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/reference.hpp>
#include <expression/ast/unary_op.hpp>
#include <expression/function_id.hpp>
#include <expression/join_condition.hpp>  // sirius::comparison_type, sirius::from_duckdb (enum mapper)
#include <expression/value.hpp>
#include <expression_evaluator/gpu_expression_translator_internal.hpp>
#include <helper/logical_type.hpp>

// duckdb — only the comparison-type enum the join-condition test parameterizes over.
#include <duckdb/common/enums/expression_type.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

// cuda
#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace sirius::expr_test;

namespace {

auto const stream = cudf::get_default_stream();
auto const mr     = cudf::get_current_device_resource_ref();

using sirius::logical_type;
using sirius::type_id;

//===----------------------------------------------------------------------===//
// Translator bridge — translate a hand-built Sirius AST node directly.
//===----------------------------------------------------------------------===//

std::optional<sirius::gpu_expression_translator::translated_expression> translate(
  sirius::gpu_expression_translator& translator, ast_node const& node)
{
  return translator.translate_expression(node);
}

//===----------------------------------------------------------------------===//
// Table-building helpers
//===----------------------------------------------------------------------===//

/// Build a single-column table from a host vector of int32 values.
std::unique_ptr<cudf::table> make_int32_table(std::vector<int32_t> const& values)
{
  auto const n = static_cast<cudf::size_type>(values.size());
  auto col     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * n,
             cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Build a two-column table from host vectors of int32 values.
std::unique_ptr<cudf::table> make_int32x2_table(std::vector<int32_t> const& col0_vals,
                                                std::vector<int32_t> const& col1_vals)
{
  auto const n = static_cast<cudf::size_type>(col0_vals.size());
  auto c0      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto c1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(c0->mutable_view().data<int32_t>(),
             col0_vals.data(),
             sizeof(int32_t) * n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(c1->mutable_view().data<int32_t>(),
             col1_vals.data(),
             sizeof(int32_t) * n,
             cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(c0));
  cols.push_back(std::move(c1));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Build a single-column table from a host vector of float64 values.
std::unique_ptr<cudf::table> make_float64_table(std::vector<double> const& values)
{
  auto const n = static_cast<cudf::size_type>(values.size());
  auto col     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(
    col->mutable_view().data<double>(), values.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Build a two-column table: col0=int32, col1=int64
std::unique_ptr<cudf::table> make_mixed_table(std::vector<int32_t> const& i32_vals,
                                              std::vector<int64_t> const& i64_vals)
{
  auto const n = static_cast<cudf::size_type>(i32_vals.size());
  auto c0      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto c1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(c0->mutable_view().data<int32_t>(),
             i32_vals.data(),
             sizeof(int32_t) * n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(c1->mutable_view().data<int64_t>(),
             i64_vals.data(),
             sizeof(int64_t) * n,
             cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(c0));
  cols.push_back(std::move(c1));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Build a single-column DECIMAL32 table from unscaled int32 representations.
/// `cudf_scale` is cudf's signed scale (negative for digits after the decimal point).
std::unique_ptr<cudf::table> make_decimal32_table(std::vector<int32_t> const& reps,
                                                  int32_t cudf_scale)
{
  auto const n = static_cast<cudf::size_type>(reps.size());
  auto col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL32, cudf_scale},
                                           n,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(
    col->mutable_view().data<int32_t>(), reps.data(), sizeof(int32_t) * n, cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Build a single-column DECIMAL64 table from unscaled int64 representations.
std::unique_ptr<cudf::table> make_decimal64_table(std::vector<int64_t> const& reps,
                                                  int32_t cudf_scale)
{
  auto const n = static_cast<cudf::size_type>(reps.size());
  auto col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL64, cudf_scale},
                                           n,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(
    col->mutable_view().data<int64_t>(), reps.data(), sizeof(int64_t) * n, cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Build a single-column STRING table from a host vector of std::string values.
std::unique_ptr<cudf::table> make_string_table(std::vector<std::string> const& values)
{
  auto const n = static_cast<cudf::size_type>(values.size());

  std::vector<cudf::size_type> offsets(static_cast<std::size_t>(n + 1), 0);
  for (cudf::size_type i = 0; i < n; ++i) {
    offsets[i + 1] = offsets[i] + static_cast<cudf::size_type>(values[i].size());
  }
  auto const total_chars = offsets[n];

  std::vector<char> chars(static_cast<std::size_t>(total_chars));
  cudf::size_type cursor = 0;
  for (cudf::size_type i = 0; i < n; ++i) {
    std::memcpy(chars.data() + cursor, values[i].data(), values[i].size());
    cursor += static_cast<cudf::size_type>(values[i].size());
  }

  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               static_cast<cudf::size_type>(offsets.size()),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
  cudaMemcpy(offsets_col->mutable_view().data<cudf::size_type>(),
             offsets.data(),
             offsets.size() * sizeof(cudf::size_type),
             cudaMemcpyHostToDevice);

  rmm::device_buffer chars_buf(static_cast<std::size_t>(total_chars), stream, mr);
  if (total_chars > 0) {
    cudaMemcpy(chars_buf.data(),
               chars.data(),
               static_cast<std::size_t>(total_chars),
               cudaMemcpyHostToDevice);
  }

  auto col = cudf::make_strings_column(
    n, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{0, stream, mr});
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

//===----------------------------------------------------------------------===//
// Translator helper
//===----------------------------------------------------------------------===//

::sirius::gpu_expression_translator make_translator()
{
  return ::sirius::gpu_expression_translator(stream, mr);
}

sirius::join_condition make_reference_join_condition(duckdb::ExpressionType comparison)
{
  // The join_condition's operands are hand-built Sirius AST reference nodes; the
  // comparison-enum mapping still goes through sirius::from_duckdb(ExpressionType),
  // which is the join_condition comparison mapper (NOT the expression from_duckdb).
  return sirius::join_condition{make_ref(0), make_ref(0), sirius::from_duckdb(comparison)};
}

}  // namespace

//===----------------------------------------------------------------------===//
// Test: Column reference
//===----------------------------------------------------------------------===//

TEST_CASE("translator: column reference produces identity", "[expression_translator]")
{
  std::vector<int32_t> values = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col_ref(0)
  auto expr = make_ref(0);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());
  REQUIRE(host_vals == values);
}

//===----------------------------------------------------------------------===//
// Test: Comparison operators
//===----------------------------------------------------------------------===//

TEST_CASE("translator: comparison EQUAL", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5, 3, 7, 3, 9, 10};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) == 3
  auto expr = make_cmp(sirius::comparison_type::equal, make_ref(0), make_int_const(3));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back(v == 3 ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: comparison NOT EQUAL", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto expr = make_cmp(sirius::comparison_type::not_equal, make_ref(0), make_int_const(3));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back(v != 3 ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: comparison LESS THAN", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto expr = make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(3));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back(v < 3 ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: comparison GREATER THAN", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto expr = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(3));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back(v > 3 ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: comparison LESS THAN OR EQUAL", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto expr = make_cmp(sirius::comparison_type::le, make_ref(0), make_int_const(3));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back(v <= 3 ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: comparison GREATER THAN OR EQUAL", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto expr = make_cmp(sirius::comparison_type::ge, make_ref(0), make_int_const(3));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back(v >= 3 ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: comparison between two columns", "[expression_translator]")
{
  std::vector<int32_t> col0 = {1, 5, 3, 8, 2};
  std::vector<int32_t> col1 = {4, 2, 3, 7, 9};
  auto table                = make_int32x2_table(col0, col1);
  auto tv                   = table->view();

  // Build: col(0) > col(1)
  auto expr = make_cmp(sirius::comparison_type::gt, make_ref(0), make_ref(1));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (size_t i = 0; i < col0.size(); ++i) {
    expected.push_back(col0[i] > col1[i] ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: DISTINCT FROM returns nullopt", "[expression_translator]")
{
  auto expr = make_cmp(sirius::comparison_type::distinct_from, make_ref(0), make_int_const(3));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

TEST_CASE("translator: NULL constant returns nullopt", "[expression_translator]")
{
  // A typed NULL has no cuDF AST literal representation; the translator must
  // report untranslatable rather than unwrapping the empty constant payload.
  auto expr = make_null_const(logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

//===----------------------------------------------------------------------===//
// Test: Join condition translation
//===----------------------------------------------------------------------===//

TEST_CASE("translator: join condition uses requested comparison operator",
          "[expression_translator]")
{
  struct join_test_case {
    duckdb::ExpressionType comparison;
    cudf::ast::ast_operator expected_operator;
  };

  std::vector<join_test_case> const test_cases = {
    {duckdb::ExpressionType::COMPARE_EQUAL, cudf::ast::ast_operator::EQUAL},
    {duckdb::ExpressionType::COMPARE_NOTEQUAL, cudf::ast::ast_operator::NOT_EQUAL},
    {duckdb::ExpressionType::COMPARE_LESSTHAN, cudf::ast::ast_operator::LESS},
    {duckdb::ExpressionType::COMPARE_GREATERTHAN, cudf::ast::ast_operator::GREATER},
    {duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO, cudf::ast::ast_operator::LESS_EQUAL},
    {duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, cudf::ast::ast_operator::GREATER_EQUAL},
  };

  for (auto const& test_case : test_cases) {
    CAPTURE(test_case.comparison);
    auto condition  = make_reference_join_condition(test_case.comparison);
    auto translator = make_translator();
    auto translated = translator.translate_join_condition(condition);

    REQUIRE(translated.has_value());
    REQUIRE(translated->size() == 3);

    auto const* root_operation = dynamic_cast<cudf::ast::operation const*>(&translated->back());
    REQUIRE(root_operation != nullptr);
    REQUIRE(root_operation->get_operator() == test_case.expected_operator);

    auto const& operands = root_operation->get_operands();
    REQUIRE(operands.size() == 2);

    auto const* left_ref = dynamic_cast<cudf::ast::column_reference const*>(&operands[0].get());
    REQUIRE(left_ref != nullptr);
    REQUIRE(left_ref->get_table_source() == cudf::ast::table_reference::LEFT);
    REQUIRE(left_ref->get_column_index() == 0);

    auto const* right_ref = dynamic_cast<cudf::ast::column_reference const*>(&operands[1].get());
    REQUIRE(right_ref != nullptr);
    REQUIRE(right_ref->get_table_source() == cudf::ast::table_reference::RIGHT);
    REQUIRE(right_ref->get_column_index() == 0);
  }
}

TEST_CASE("translator: join condition unsupported comparison returns nullopt",
          "[expression_translator]")
{
  auto condition  = make_reference_join_condition(duckdb::ExpressionType::COMPARE_DISTINCT_FROM);
  auto translator = make_translator();
  auto translated = translator.translate_join_condition(condition);

  REQUIRE_FALSE(translated.has_value());
}

//===----------------------------------------------------------------------===//
// Test: Arithmetic function expressions (+, -, *, /, %)
//===----------------------------------------------------------------------===//

TEST_CASE("translator: addition col(0) + 10", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) + 10
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(10));
  auto expr =
    make_func(sirius::function_id::add, std::move(args), logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());

  std::vector<int32_t> expected;
  for (auto v : values) {
    expected.push_back(v + 10);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: subtraction col(0) - 3", "[expression_translator]")
{
  std::vector<int32_t> values = {10, 20, 30, 40, 50};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(3));
  auto expr =
    make_func(sirius::function_id::sub, std::move(args), logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());

  std::vector<int32_t> expected;
  for (auto v : values) {
    expected.push_back(v - 3);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: multiplication col(0) * 2", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(2));
  auto expr =
    make_func(sirius::function_id::mul, std::move(args), logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());

  std::vector<int32_t> expected;
  for (auto v : values) {
    expected.push_back(v * 2);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: division col(0) / 3", "[expression_translator]")
{
  std::vector<int32_t> values = {9, 12, 15, 18, 21};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(3));
  auto expr =
    make_func(sirius::function_id::div, std::move(args), logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());

  std::vector<int32_t> expected;
  for (auto v : values) {
    expected.push_back(v / 3);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: integer division col(0) // 3", "[expression_translator]")
{
  std::vector<int32_t> values = {10, 11, 12, 13, 14};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(3));
  auto expr =
    make_func(sirius::function_id::int_div, std::move(args), logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());

  std::vector<int32_t> expected;
  for (auto v : values) {
    expected.push_back(v / 3);  // integer division
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: modulo col(0) % 3", "[expression_translator]")
{
  std::vector<int32_t> values = {7, 8, 9, 10, 11, 12};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(3));
  auto expr =
    make_func(sirius::function_id::mod, std::move(args), logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());

  std::vector<int32_t> expected;
  for (auto v : values) {
    expected.push_back(v % 3);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: col(0) + col(1)", "[expression_translator]")
{
  std::vector<int32_t> col0 = {1, 2, 3, 4, 5};
  std::vector<int32_t> col1 = {10, 20, 30, 40, 50};
  auto table                = make_int32x2_table(col0, col1);
  auto tv                   = table->view();

  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_ref(1));
  auto expr =
    make_func(sirius::function_id::add, std::move(args), logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());

  std::vector<int32_t> expected;
  for (size_t i = 0; i < col0.size(); ++i) {
    expected.push_back(col0[i] + col1[i]);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: unsupported function returns nullopt", "[expression_translator]")
{
  // A non-arithmetic function (substring) has no cuDF-AST binary-op lowering;
  // the translator must report untranslatable.
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref(0));
  auto expr = make_func(
    sirius::function_id::substring, std::move(args), logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

//===----------------------------------------------------------------------===//
// Test: Conjunction (AND / OR)
//===----------------------------------------------------------------------===//
TEST_CASE("translator: conjunction with unsupported first child returns nullopt",
          "[expression_translator]")
{
  std::vector<std::unique_ptr<ast_node>> unsupported_args;
  unsupported_args.push_back(make_ref(0));
  auto unsupported = make_func(sirius::function_id::substring,
                               std::move(unsupported_args),
                               logical_type::make(type_id::INTEGER));

  auto supported = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(0));

  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(std::move(unsupported));
  children.push_back(std::move(supported));
  auto conjunction = make_conj(sirius::ast::conjunction::kind::op_and, std::move(children));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *conjunction);
  REQUIRE_FALSE(ast_tree.has_value());
}

TEST_CASE("translator: conjunction AND", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) > 3 AND col(0) < 8
  auto cmp1 = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(3));
  auto cmp2 = make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(8));

  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(std::move(cmp1));
  children.push_back(std::move(cmp2));
  auto conj = make_conj(sirius::ast::conjunction::kind::op_and, std::move(children));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *conj);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back((v > 3 && v < 8) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: conjunction OR", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) <= 2 OR col(0) >= 9
  auto cmp1 = make_cmp(sirius::comparison_type::le, make_ref(0), make_int_const(2));
  auto cmp2 = make_cmp(sirius::comparison_type::ge, make_ref(0), make_int_const(9));

  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(std::move(cmp1));
  children.push_back(std::move(cmp2));
  auto conj = make_conj(sirius::ast::conjunction::kind::op_or, std::move(children));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *conj);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back((v <= 2 || v >= 9) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: conjunction with three children", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) > 2 AND col(0) < 9 AND col(0) != 5
  auto cmp1 = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(2));
  auto cmp2 = make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(9));
  auto cmp3 = make_cmp(sirius::comparison_type::not_equal, make_ref(0), make_int_const(5));

  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(std::move(cmp1));
  children.push_back(std::move(cmp2));
  children.push_back(std::move(cmp3));
  auto conj = make_conj(sirius::ast::conjunction::kind::op_and, std::move(children));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *conj);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back((v > 2 && v < 9 && v != 5) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: empty conjunction returns nullopt", "[expression_translator]")
{
  // No children
  auto conj = make_conj(sirius::ast::conjunction::kind::op_and, {});

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *conj);
  REQUIRE_FALSE(ast_tree.has_value());
}

//===----------------------------------------------------------------------===//
// Test: BETWEEN expression
//===----------------------------------------------------------------------===//

TEST_CASE("translator: BETWEEN expression", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) BETWEEN 3 AND 7  ->  col(0) >= 3 AND col(0) <= 7
  auto between = make_between(make_ref(0), make_int_const(3), make_int_const(7), true, true);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *between);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back((v >= 3 && v <= 7) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: BETWEEN with tight range", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // BETWEEN 3 AND 3 -> only 3 matches
  auto between = make_between(make_ref(0), make_int_const(3), make_int_const(3), true, true);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *between);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected = {0, 0, 1, 0, 0};
  REQUIRE(host_vals == expected);
}

//===----------------------------------------------------------------------===//
// Test: CAST expressions
//===----------------------------------------------------------------------===//

TEST_CASE("translator: CAST to INT64", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: CAST(col(0) AS BIGINT)
  auto cast_expr = make_cast(make_ref(0), logical_type::make(type_id::BIGINT), /*try_cast=*/false);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *cast_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int64_t>(result->view());

  std::vector<int64_t> expected;
  for (auto v : values) {
    expected.push_back(static_cast<int64_t>(v));
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: CAST to FLOAT64", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto cast_expr = make_cast(make_ref(0), logical_type::make(type_id::DOUBLE), /*try_cast=*/false);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *cast_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<double>(result->view());

  std::vector<double> expected;
  for (auto v : values) {
    expected.push_back(static_cast<double>(v));
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: unsupported CAST returns nullopt", "[expression_translator]")
{
  // CAST to VARCHAR is not supported
  auto cast_expr = make_cast(make_ref(0), logical_type::make(type_id::VARCHAR), /*try_cast=*/false);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *cast_expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

//===----------------------------------------------------------------------===//
// Test: IN / NOT IN operators
//===----------------------------------------------------------------------===//

TEST_CASE("translator: IN operator", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) IN (2, 5, 8)
  std::vector<std::unique_ptr<ast_node>> in_values;
  in_values.push_back(make_int_const(2));
  in_values.push_back(make_int_const(5));
  in_values.push_back(make_int_const(8));
  auto in_expr = make_in(make_ref(0), std::move(in_values), /*negated=*/false);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *in_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back((v == 2 || v == 5 || v == 8) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: NOT IN operator", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) NOT IN (2, 4)
  std::vector<std::unique_ptr<ast_node>> in_values;
  in_values.push_back(make_int_const(2));
  in_values.push_back(make_int_const(4));
  auto in_expr = make_in(make_ref(0), std::move(in_values), /*negated=*/true);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *in_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back((v != 2 && v != 4) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: IN with single comparator", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) IN (2) -- single-element IN list
  std::vector<std::unique_ptr<ast_node>> in_values;
  in_values.push_back(make_int_const(2));
  auto in_expr = make_in(make_ref(0), std::move(in_values), /*negated=*/false);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *in_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected = {0, 1, 0};
  REQUIRE(host_vals == expected);
}

//===----------------------------------------------------------------------===//
// Test: NOT / IS NULL / IS NOT NULL operators
//===----------------------------------------------------------------------===//

TEST_CASE("translator: NOT operator", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: NOT (col(0) > 3)
  auto cmp      = make_cmp(sirius::comparison_type::gt, make_ref(0), make_int_const(3));
  auto not_expr = make_unary(sirius::ast::unary_op::kind::op_not, std::move(cmp));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *not_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back(!(v > 3) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: IS NULL operator", "[expression_translator]")
{
  // Column without nulls -- IS NULL should be all false
  std::vector<int32_t> values = {1, 2, 3};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto is_null_expr = make_unary(sirius::ast::unary_op::kind::op_is_null, make_ref(0));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *is_null_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  // No nulls -> all false
  std::vector<uint8_t> expected = {0, 0, 0};
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: IS NOT NULL operator", "[expression_translator]")
{
  // Column without nulls -- IS NOT NULL should be all true
  std::vector<int32_t> values = {1, 2, 3};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto is_not_null_expr = make_unary(sirius::ast::unary_op::kind::op_is_not_null, make_ref(0));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *is_not_null_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  // No nulls -> all true
  std::vector<uint8_t> expected = {1, 1, 1};
  REQUIRE(host_vals == expected);
}

//===----------------------------------------------------------------------===//
// Test: Unsupported expression types return nullopt
//===----------------------------------------------------------------------===//

TEST_CASE("translator: CASE expression returns nullopt", "[expression_translator]")
{
  // CASE WHEN col(0)=1 THEN 10 ELSE 0 END
  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.push_back(sirius::ast::case_expr::when_then{
    make_cmp(sirius::comparison_type::equal, make_ref(0), make_int_const(1)), make_int_const(10)});
  auto case_expr = std::make_unique<ast_node>(sirius::ast::case_expr{
    std::move(cases), make_int_const(0), logical_type::make(type_id::INTEGER)});

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *case_expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

TEST_CASE("translator: COALESCE operator returns nullopt", "[expression_translator]")
{
  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_int_const(0));
  auto coalesce_expr = std::make_unique<ast_node>(
    sirius::ast::coalesce{std::move(children), logical_type::make(type_id::INTEGER)});

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *coalesce_expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

TEST_CASE("translator: TRY operator returns nullopt", "[expression_translator]")
{
  auto try_expr = make_unary(sirius::ast::unary_op::kind::op_try, make_ref(0));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *try_expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

//===----------------------------------------------------------------------===//
// Test: Complex / nested expressions
//===----------------------------------------------------------------------===//

TEST_CASE("translator: nested arithmetic (col(0) + 1) * (col(1) - 2)", "[expression_translator]")
{
  std::vector<int32_t> col0 = {1, 2, 3, 4, 5};
  std::vector<int32_t> col1 = {10, 20, 30, 40, 50};
  auto table                = make_int32x2_table(col0, col1);
  auto tv                   = table->view();

  // Build: (col(0) + 1) * (col(1) - 2)
  std::vector<std::unique_ptr<ast_node>> add_args;
  add_args.push_back(make_ref(0));
  add_args.push_back(make_int_const(1));
  auto add_expr =
    make_func(sirius::function_id::add, std::move(add_args), logical_type::make(type_id::INTEGER));

  std::vector<std::unique_ptr<ast_node>> sub_args;
  sub_args.push_back(make_ref(1));
  sub_args.push_back(make_int_const(2));
  auto sub_expr =
    make_func(sirius::function_id::sub, std::move(sub_args), logical_type::make(type_id::INTEGER));

  std::vector<std::unique_ptr<ast_node>> mul_args;
  mul_args.push_back(std::move(add_expr));
  mul_args.push_back(std::move(sub_expr));
  auto mul_expr =
    make_func(sirius::function_id::mul, std::move(mul_args), logical_type::make(type_id::INTEGER));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *mul_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());

  std::vector<int32_t> expected;
  for (size_t i = 0; i < col0.size(); ++i) {
    expected.push_back((col0[i] + 1) * (col1[i] - 2));
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: comparison with arithmetic col(0)*2 > col(1)+3", "[expression_translator]")
{
  std::vector<int32_t> col0 = {5, 10, 1, 20, 3};
  std::vector<int32_t> col1 = {9, 15, 0, 35, 2};
  auto table                = make_int32x2_table(col0, col1);
  auto tv                   = table->view();

  // Build LHS: col(0) * 2
  std::vector<std::unique_ptr<ast_node>> lhs_args;
  lhs_args.push_back(make_ref(0));
  lhs_args.push_back(make_int_const(2));
  auto lhs =
    make_func(sirius::function_id::mul, std::move(lhs_args), logical_type::make(type_id::INTEGER));

  // Build RHS: col(1) + 3
  std::vector<std::unique_ptr<ast_node>> rhs_args;
  rhs_args.push_back(make_ref(1));
  rhs_args.push_back(make_int_const(3));
  auto rhs =
    make_func(sirius::function_id::add, std::move(rhs_args), logical_type::make(type_id::INTEGER));

  auto cmp = make_cmp(sirius::comparison_type::gt, std::move(lhs), std::move(rhs));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *cmp);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (size_t i = 0; i < col0.size(); ++i) {
    expected.push_back((col0[i] * 2 > col1[i] + 3) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: IN combined with AND/OR and comparison", "[expression_translator]")
{
  std::vector<int32_t> col0_vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int64_t> col1_vals = {50, 40, 30, 20, 10, 60, 70, 80, 90, 100};
  auto table                     = make_mixed_table(col0_vals, col1_vals);
  auto tv                        = table->view();

  // Build: col(0) IN (1, 3, 5, 7, 9) AND col(1) >= 50
  std::vector<std::unique_ptr<ast_node>> in_values;
  in_values.push_back(make_int_const(1));
  in_values.push_back(make_int_const(3));
  in_values.push_back(make_int_const(5));
  in_values.push_back(make_int_const(7));
  in_values.push_back(make_int_const(9));
  auto in_expr = make_in(make_ref(0), std::move(in_values), /*negated=*/false);

  auto cmp = make_cmp(sirius::comparison_type::ge, make_ref(1), make_bigint_const(50));

  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(std::move(in_expr));
  children.push_back(std::move(cmp));
  auto conj = make_conj(sirius::ast::conjunction::kind::op_and, std::move(children));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *conj);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (size_t i = 0; i < col0_vals.size(); ++i) {
    bool in_set = col0_vals[i] == 1 || col0_vals[i] == 3 || col0_vals[i] == 5 ||
                  col0_vals[i] == 7 || col0_vals[i] == 9;
    expected.push_back((in_set && col1_vals[i] >= 50) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: deeply nested NOT(col(0) BETWEEN 3 AND 7)", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: NOT (col(0) BETWEEN 3 AND 7)
  auto between  = make_between(make_ref(0), make_int_const(3), make_int_const(7), true, true);
  auto not_expr = make_unary(sirius::ast::unary_op::kind::op_not, std::move(between));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *not_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back(!(v >= 3 && v <= 7) ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

//===----------------------------------------------------------------------===//
// Test: Float64 arithmetic
//===----------------------------------------------------------------------===//

TEST_CASE("translator: float64 arithmetic col(0) + 0.5", "[expression_translator]")
{
  std::vector<double> values = {1.0, 2.5, 3.75, 4.0, 5.5};
  auto table                 = make_float64_table(values);
  auto tv                    = table->view();

  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_double_const(0.5));
  auto func_expr =
    make_func(sirius::function_id::add, std::move(args), logical_type::make(type_id::DOUBLE));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *func_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<double>(result->view());

  REQUIRE(host_vals.size() == values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    REQUIRE(host_vals[i] == Approx(values[i] + 0.5));
  }
}

TEST_CASE("translator: float64 comparison col(0) < 3.5", "[expression_translator]")
{
  std::vector<double> values = {1.0, 2.5, 3.5, 4.0, 5.5};
  auto table                 = make_float64_table(values);
  auto tv                    = table->view();

  auto expr = make_cmp(sirius::comparison_type::lt, make_ref(0), make_double_const(3.5));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected;
  for (auto v : values) {
    expected.push_back(v < 3.5 ? 1U : 0U);
  }
  REQUIRE(host_vals == expected);
}

//===----------------------------------------------------------------------===//
// Test: Translator is reusable (translate_expression can be called again)
//===----------------------------------------------------------------------===//

TEST_CASE("translator: reuse translator for multiple expressions", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto translator = make_translator();

  // First translation: col(0) + 10
  {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref(0));
    args.push_back(make_int_const(10));
    auto func_expr =
      make_func(sirius::function_id::add, std::move(args), logical_type::make(type_id::INTEGER));

    auto ast_tree = translate(translator, *func_expr);
    REQUIRE(ast_tree.has_value());

    auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
    auto host_vals = copy_column_to_host<int32_t>(result->view());

    std::vector<int32_t> expected;
    for (auto v : values) {
      expected.push_back(v + 10);
    }
    REQUIRE(host_vals == expected);
  }

  // Second translation with same translator: col(0) * 3
  {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref(0));
    args.push_back(make_int_const(3));
    auto func_expr =
      make_func(sirius::function_id::mul, std::move(args), logical_type::make(type_id::INTEGER));

    auto ast_tree = translate(translator, *func_expr);
    REQUIRE(ast_tree.has_value());

    auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
    auto host_vals = copy_column_to_host<int32_t>(result->view());

    std::vector<int32_t> expected;
    for (auto v : values) {
      expected.push_back(v * 3);
    }
    REQUIRE(host_vals == expected);
  }
}

//===----------------------------------------------------------------------===//
// Test: CAST chained with arithmetic
//===----------------------------------------------------------------------===//

TEST_CASE("translator: CAST(col(0) AS BIGINT) + 100", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: CAST(col(0) AS BIGINT) + 100
  auto cast_expr = make_cast(make_ref(0), logical_type::make(type_id::BIGINT), /*try_cast=*/false);

  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(std::move(cast_expr));
  args.push_back(make_bigint_const(100));
  auto func_expr =
    make_func(sirius::function_id::add, std::move(args), logical_type::make(type_id::BIGINT));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *func_expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int64_t>(result->view());

  std::vector<int64_t> expected;
  for (auto v : values) {
    expected.push_back(static_cast<int64_t>(v) + 100);
  }
  REQUIRE(host_vals == expected);
}

//===----------------------------------------------------------------------===//
// Test: Edge cases
//===----------------------------------------------------------------------===//

TEST_CASE("translator: single-row table", "[expression_translator]")
{
  std::vector<int32_t> values = {42};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto expr = make_cmp(sirius::comparison_type::equal, make_ref(0), make_int_const(42));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  std::vector<uint8_t> expected = {1};
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: 100-row table with complex expression", "[expression_translator]")
{
  // Build a 100-row table
  std::vector<int32_t> col0(100), col1(100);
  for (int i = 0; i < 100; ++i) {
    col0[i] = i;
    col1[i] = 100 - i;
  }
  auto table = make_int32x2_table(col0, col1);
  auto tv    = table->view();

  // Build: (col(0) + col(1)) == 100
  std::vector<std::unique_ptr<ast_node>> add_args;
  add_args.push_back(make_ref(0));
  add_args.push_back(make_ref(1));
  auto add_expr =
    make_func(sirius::function_id::add, std::move(add_args), logical_type::make(type_id::INTEGER));

  auto cmp = make_cmp(sirius::comparison_type::equal, std::move(add_expr), make_int_const(100));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *cmp);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  // col0[i] + col1[i] == i + (100-i) == 100 for all rows
  std::vector<uint8_t> expected(100, 1);
  REQUIRE(host_vals == expected);
}

//===----------------------------------------------------------------------===//
// Test: Decimal column vs. decimal literal comparisons
//
// NOTE: decimal column-vs-literal comparisons work once rapidsai/cudf#21447 is
// included. Decimal arithmetic (function_call children returning DECIMAL) is
// currently blocked by the translator pending rapidsai/cudf#21996, so the
// positive tests here stay inside the comparison pattern and a negative test
// below pins the blocking behavior.
//===----------------------------------------------------------------------===//

TEST_CASE("translator: decimal32 column EQUAL decimal literal", "[expression_translator]")
{
  // Column: DECIMAL(5,2) values 1.00, 2.50, 3.00, 2.50, 4.00
  std::vector<int32_t> col_reps = {100, 250, 300, 250, 400};
  auto table                    = make_decimal32_table(col_reps, -2);
  auto tv                       = table->view();

  auto const dec52 = logical_type::make_decimal(5, 2);

  // col(0) == 2.50
  auto expr =
    make_cmp(sirius::comparison_type::equal, make_ref_typed(0, dec52), make_dec32_const(250, 5, 2));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());
  REQUIRE(host_vals == std::vector<uint8_t>{0, 1, 0, 1, 0});
}

TEST_CASE("translator: decimal64 column LESS decimal literal", "[expression_translator]")
{
  // Column: DECIMAL(12,4) values 1.0000, 2.0000, 3.0000, 4.0000
  std::vector<int64_t> col_reps = {10000, 20000, 30000, 40000};
  auto table                    = make_decimal64_table(col_reps, -4);
  auto tv                       = table->view();

  auto const dec124 = logical_type::make_decimal(12, 4);

  // col(0) < 2.5000
  auto expr = make_cmp(
    sirius::comparison_type::lt, make_ref_typed(0, dec124), make_dec64_const(25000, 12, 4));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());
  REQUIRE(host_vals == std::vector<uint8_t>{1, 1, 0, 0});
}

TEST_CASE("translator: nested decimal comparison col(0) >= 2.00 AND col(0) <= 4.00",
          "[expression_translator]")
{
  // Column: DECIMAL(5,2) values 1.00, 2.00, 3.00, 4.00, 5.00
  std::vector<int32_t> col_reps = {100, 200, 300, 400, 500};
  auto table                    = make_decimal32_table(col_reps, -2);
  auto tv                       = table->view();

  auto const dec52 = logical_type::make_decimal(5, 2);

  // col(0) >= 2.00
  auto geq =
    make_cmp(sirius::comparison_type::ge, make_ref_typed(0, dec52), make_dec32_const(200, 5, 2));

  // col(0) <= 4.00
  auto leq =
    make_cmp(sirius::comparison_type::le, make_ref_typed(0, dec52), make_dec32_const(400, 5, 2));

  // (col(0) >= 2.00) AND (col(0) <= 4.00)
  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(std::move(geq));
  children.push_back(std::move(leq));
  auto conj = make_conj(sirius::ast::conjunction::kind::op_and, std::move(children));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *conj);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());
  REQUIRE(host_vals == std::vector<uint8_t>{0, 1, 1, 1, 0});
}

TEST_CASE("translator: nested decimal arithmetic returns nullopt", "[expression_translator]")
{
  // Any function_call whose children have DECIMAL return_type must be blocked by
  // the translator (pending rapidsai/cudf#21996). Build a nested expression
  // (col(0) + col(0)) * 2.00 over a DECIMAL(5,2) column and confirm the
  // translator refuses both the inner and outer functions.
  auto const dec52 = logical_type::make_decimal(5, 2);

  // col(0) + col(0)
  std::vector<std::unique_ptr<ast_node>> add_args;
  add_args.push_back(make_ref_typed(0, dec52));
  add_args.push_back(make_ref_typed(0, dec52));
  auto add = make_func(sirius::function_id::add, std::move(add_args), dec52);

  // (col(0) + col(0)) * 2.00
  std::vector<std::unique_ptr<ast_node>> mul_args;
  mul_args.push_back(std::move(add));
  mul_args.push_back(make_dec32_const(200, 5, 2));
  auto mul = make_func(sirius::function_id::mul, std::move(mul_args), dec52);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *mul);
  REQUIRE_FALSE(ast_tree.has_value());
}

TEST_CASE("translator: direct decimal column arithmetic returns nullopt", "[expression_translator]")
{
  // Direct column-vs-column decimal arithmetic (col(0) + col(0)) must also be
  // blocked (pending rapidsai/cudf#21996), not just arithmetic whose children
  // are decimal constants or nested functions. The function's arguments are
  // bare references here, so the guard depends on reference nodes carrying their
  // DECIMAL return_type.
  auto const dec52 = logical_type::make_decimal(5, 2);

  std::vector<std::unique_ptr<ast_node>> add_args;
  add_args.push_back(make_ref_typed(0, dec52));
  add_args.push_back(make_ref_typed(0, dec52));
  auto add = make_func(sirius::function_id::add, std::move(add_args), dec52);

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *add);
  REQUIRE_FALSE(ast_tree.has_value());
}

//===----------------------------------------------------------------------===//
// Test: String column vs. string literal comparisons
//===----------------------------------------------------------------------===//

TEST_CASE("translator: string column EQUAL string literal", "[expression_translator]")
{
  auto table = make_string_table({"alpha", "bravo", "charlie", "bravo", "delta"});
  auto tv    = table->view();

  // col(0) == 'bravo'
  auto expr = make_cmp(sirius::comparison_type::equal, make_ref(0), make_str_const("bravo"));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());
  REQUIRE(host_vals == std::vector<uint8_t>{0, 1, 0, 1, 0});
}

TEST_CASE("translator: string column NOT EQUAL string literal", "[expression_translator]")
{
  auto table = make_string_table({"foo", "bar", "foo", "baz"});
  auto tv    = table->view();

  // col(0) != 'foo'
  auto expr = make_cmp(sirius::comparison_type::not_equal, make_ref(0), make_str_const("foo"));

  auto translator = make_translator();
  auto ast_tree   = translate(translator, *expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());
  REQUIRE(host_vals == std::vector<uint8_t>{0, 1, 0, 1});
}
