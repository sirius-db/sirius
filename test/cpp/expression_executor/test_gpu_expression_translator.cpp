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
#include <catch.hpp>

// sirius
#include <expression_executor/gpu_expression_translator.hpp>

// duckdb
#include <duckdb/common/helper.hpp>

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
#include <memory>
#include <vector>

using namespace duckdb;

namespace {

auto const stream = cudf::get_default_stream();
auto const mr     = cudf::get_current_device_resource_ref();

//===----------------------------------------------------------------------===//
// GPU <-> host copy helpers
//===----------------------------------------------------------------------===//

template <typename T>
std::vector<T> copy_column_to_host(cudf::column_view const& col)
{
  std::vector<T> host(col.size());
  if (col.size() > 0) {
    cudaMemcpy(host.data(), col.data<T>(), sizeof(T) * col.size(), cudaMemcpyDeviceToHost);
  }
  return host;
}

std::vector<uint8_t> copy_bool_column_to_host(cudf::column_view const& col)
{
  std::vector<uint8_t> host(col.size());
  if (col.size() > 0) {
    cudaMemcpy(host.data(), col.head(), sizeof(uint8_t) * col.size(), cudaMemcpyDeviceToHost);
  }
  return host;
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

/// Build a single-column table from a host vector of int64 values.
std::unique_ptr<cudf::table> make_int64_table(std::vector<int64_t> const& values)
{
  auto const n = static_cast<cudf::size_type>(values.size());
  auto col     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(col->mutable_view().data<int64_t>(),
             values.data(),
             sizeof(int64_t) * n,
             cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
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

/// Build a single-column table from a host vector of int16 values.
std::unique_ptr<cudf::table> make_int16_table(std::vector<int16_t> const& values)
{
  auto const n = static_cast<cudf::size_type>(values.size());
  auto col     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT16}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(col->mutable_view().data<int16_t>(),
             values.data(),
             sizeof(int16_t) * n,
             cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Build a single-column table from a host vector of float32 values.
std::unique_ptr<cudf::table> make_float32_table(std::vector<float> const& values)
{
  auto const n = static_cast<cudf::size_type>(values.size());
  auto col     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(
    col->mutable_view().data<float>(), values.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
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

duckdb::JoinCondition make_reference_join_condition(duckdb::ExpressionType comparison)
{
  duckdb::JoinCondition condition;
  condition.left =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  condition.right =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  condition.comparison = comparison;
  return condition;
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
  auto expr = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());
  REQUIRE(host_vals == values);
}

//===----------------------------------------------------------------------===//
// Test: Integer constant
//===----------------------------------------------------------------------===//

TEST_CASE("translator: integer constant (INT32)", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: 42  (constant)
  auto expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(42));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int32_t>(result->view());
  std::vector<int32_t> expected(5, 42);
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: integer constant (INT64)", "[expression_translator]")
{
  std::vector<int64_t> values = {1, 2, 3};
  auto table                  = make_int64_table(values);
  auto tv                     = table->view();

  auto expr = duckdb::make_uniq<BoundConstantExpression>(Value::BIGINT(9999));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int64_t>(result->view());
  std::vector<int64_t> expected(3, 9999);
  REQUIRE(host_vals == expected);
}

TEST_CASE("translator: float64 constant", "[expression_translator]")
{
  std::vector<double> values = {1.0, 2.0};
  auto table                 = make_float64_table(values);
  auto tv                    = table->view();

  auto expr = duckdb::make_uniq<BoundConstantExpression>(Value::DOUBLE(3.14));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<double>(result->view());
  REQUIRE(host_vals.size() == 2);
  REQUIRE(host_vals[0] == Approx(3.14));
  REQUIRE(host_vals[1] == Approx(3.14));
}

TEST_CASE("translator: boolean constant", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  auto expr = duckdb::make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());
  REQUIRE(host_vals.size() == 3);
  REQUIRE(host_vals[0] == 1);
  REQUIRE(host_vals[1] == 1);
  REQUIRE(host_vals[2] == 1);
}

TEST_CASE("translator: int16 constant", "[expression_translator]")
{
  std::vector<int16_t> values = {10, 20, 30};
  auto table                  = make_int16_table(values);
  auto tv                     = table->view();

  auto expr = duckdb::make_uniq<BoundConstantExpression>(Value::SMALLINT(7));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<int16_t>(result->view());
  std::vector<int16_t> expected(3, 7);
  REQUIRE(host_vals == expected);
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
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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

  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_NOTEQUAL, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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

  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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

  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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

  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHANOREQUALTO, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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

  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHANOREQUALTO, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1);
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_DISTINCT_FROM, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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

  // Build: col(0) + 10  via BoundFunctionExpression with name "+"
  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "+", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(10)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
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

  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "-", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
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

  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "*", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
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

  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "/", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
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

  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "//", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
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

  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "%", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
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

  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "+", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
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
  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction("abs", {LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

//===----------------------------------------------------------------------===//
// Test: Conjunction (AND / OR)
//===----------------------------------------------------------------------===//

TEST_CASE("translator: conjunction AND", "[expression_translator]")
{
  std::vector<int32_t> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto table                  = make_int32_table(values);
  auto tv                     = table->view();

  // Build: col(0) > 3 AND col(0) < 8
  auto left1  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right1 = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto cmp1   = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left1), std::move(right1));

  auto left2  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right2 = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(8));
  auto cmp2   = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, std::move(left2), std::move(right2));

  auto conj = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  conj->children.push_back(std::move(cmp1));
  conj->children.push_back(std::move(cmp2));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*conj);
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
  auto left1  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right1 = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2));
  auto cmp1   = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHANOREQUALTO, std::move(left1), std::move(right1));

  auto left2  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right2 = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(9));
  auto cmp2   = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHANOREQUALTO, std::move(left2), std::move(right2));

  auto conj = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_OR);
  conj->children.push_back(std::move(cmp1));
  conj->children.push_back(std::move(cmp2));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*conj);
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
  auto cmp1 = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));

  auto cmp2 = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(9)));

  auto cmp3 = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_NOTEQUAL,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5)));

  auto conj = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  conj->children.push_back(std::move(cmp1));
  conj->children.push_back(std::move(cmp2));
  conj->children.push_back(std::move(cmp3));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*conj);
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
  auto conj = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  // No children

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*conj);
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
  auto input_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto lower_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto upper_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(7));

  auto between = duckdb::make_uniq<BoundBetweenExpression>(
    std::move(input_expr), std::move(lower_expr), std::move(upper_expr), true, true);

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*between);
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
  auto input_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto lower_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto upper_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));

  auto between = duckdb::make_uniq<BoundBetweenExpression>(
    std::move(input_expr), std::move(lower_expr), std::move(upper_expr), true, true);

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*between);
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
  auto child = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto cast_expr =
    BoundCastExpression::AddDefaultCastToType(std::move(child), LogicalType{LogicalTypeId::BIGINT});

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*cast_expr);
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

  auto child = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto cast_expr =
    BoundCastExpression::AddDefaultCastToType(std::move(child), LogicalType{LogicalTypeId::DOUBLE});

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*cast_expr);
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
  auto child = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto cast_expr = BoundCastExpression::AddDefaultCastToType(std::move(child),
                                                             LogicalType{LogicalTypeId::VARCHAR});

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*cast_expr);
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
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                            LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(8)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*in_expr);
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
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_NOT_IN,
                                                            LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(4)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*in_expr);
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
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                            LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*in_expr);
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
  auto cmp = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  auto not_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_NOT,
                                                             LogicalType{LogicalTypeId::BOOLEAN});
  not_expr->children.push_back(std::move(cmp));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*not_expr);
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

  auto is_null_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_IS_NULL, LogicalType{LogicalTypeId::BOOLEAN});
  is_null_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*is_null_expr);
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

  auto is_not_null_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType{LogicalTypeId::BOOLEAN});
  is_not_null_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*is_not_null_expr);
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
  // BoundCaseExpression: CASE WHEN col(0)=1 THEN 10 ELSE 0 END
  auto check = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  auto result_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(10));

  duckdb::BoundCaseCheck case_check;
  case_check.when_expr = std::move(check);
  case_check.then_expr = std::move(result_expr);

  auto else_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0));
  auto case_expr = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
  case_expr->else_expr = std::move(else_expr);
  case_expr->case_checks.push_back(std::move(case_check));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*case_expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

TEST_CASE("translator: COALESCE operator returns nullopt", "[expression_translator]")
{
  auto coalesce_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_COALESCE, LogicalType{LogicalTypeId::INTEGER});
  coalesce_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  coalesce_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*coalesce_expr);
  REQUIRE_FALSE(ast_tree.has_value());
}

TEST_CASE("translator: TRY operator returns nullopt", "[expression_translator]")
{
  auto try_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_TRY,
                                                             LogicalType{LogicalTypeId::INTEGER});
  try_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*try_expr);
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
  auto add_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "+", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  add_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  add_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));

  auto sub_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "-", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  sub_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1));
  sub_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));

  auto mul_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "*", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  mul_expr->children.push_back(std::move(add_expr));
  mul_expr->children.push_back(std::move(sub_expr));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*mul_expr);
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
  auto lhs = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "*", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  lhs->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  lhs->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));

  // Build RHS: col(1) + 3
  auto rhs = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "+", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  rhs->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1));
  rhs->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  auto cmp = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(lhs), std::move(rhs));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*cmp);
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
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                            LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(7)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(9)));

  auto cmp = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHANOREQUALTO,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BIGINT}, 1),
    duckdb::make_uniq<BoundConstantExpression>(Value::BIGINT(50)));

  auto conj = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  conj->children.push_back(std::move(in_expr));
  conj->children.push_back(std::move(cmp));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*conj);
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
  auto between = duckdb::make_uniq<BoundBetweenExpression>(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(7)),
    true,
    true);

  auto not_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_NOT,
                                                             LogicalType{LogicalTypeId::BOOLEAN});
  not_expr->children.push_back(std::move(between));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*not_expr);
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

  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::DOUBLE},
    ScalarFunction("+", {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::DOUBLE}, 0));
  func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::DOUBLE(0.5)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
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

  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::DOUBLE}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::DOUBLE(3.5));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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
    auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
      LogicalType{LogicalTypeId::INTEGER},
      ScalarFunction(
        "+", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
      duckdb::vector<duckdb::unique_ptr<Expression>>{},
      nullptr);
    func_expr->children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
    func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(10)));

    auto ast_tree = translator.translate_expression(*func_expr);
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
    auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
      LogicalType{LogicalTypeId::INTEGER},
      ScalarFunction(
        "*", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
      duckdb::vector<duckdb::unique_ptr<Expression>>{},
      nullptr);
    func_expr->children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
    func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

    auto ast_tree = translator.translate_expression(*func_expr);
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
// Test: Float32 constant
//===----------------------------------------------------------------------===//

TEST_CASE("translator: float32 constant", "[expression_translator]")
{
  std::vector<float> values = {1.0f, 2.0f, 3.0f};
  auto table                = make_float32_table(values);
  auto tv                   = table->view();

  auto expr = duckdb::make_uniq<BoundConstantExpression>(Value::FLOAT(2.5f));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_column_to_host<float>(result->view());
  REQUIRE(host_vals.size() == 3);
  REQUIRE(host_vals[0] == Approx(2.5f));
  REQUIRE(host_vals[1] == Approx(2.5f));
  REQUIRE(host_vals[2] == Approx(2.5f));
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
  auto cast_child =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto cast_expr = BoundCastExpression::AddDefaultCastToType(std::move(cast_child),
                                                             LogicalType{LogicalTypeId::BIGINT});

  auto func_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::BIGINT},
    ScalarFunction("+", {LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::BIGINT, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  func_expr->children.push_back(std::move(cast_expr));
  func_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::BIGINT(100)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*func_expr);
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

  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(42));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, std::move(left), std::move(right));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*expr);
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
  auto add_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "+", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  add_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  add_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1));

  auto cmp = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL,
    std::move(add_expr),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(100)));

  auto translator = make_translator();
  auto ast_tree   = translator.translate_expression(*cmp);
  REQUIRE(ast_tree.has_value());

  auto result    = cudf::compute_column(tv, ast_tree->back(), stream, mr);
  auto host_vals = copy_bool_column_to_host(result->view());

  // col0[i] + col1[i] == i + (100-i) == 100 for all rows
  std::vector<uint8_t> expected(100, 1);
  REQUIRE(host_vals == expected);
}
