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
#include <utils/utils.hpp>

// sirius
#include <data/data_repository_manager.hpp>
#include <data/gpu_data_representation.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <memory/sirius_memory_manager.hpp>

// duckdb
#include <duckdb/common/helper.hpp>

// cudf, etc.
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cstdint>

using namespace duckdb;
using namespace duckdb::sirius;
using namespace cucascade;
using namespace cucascade::memory;
using data_repository_mgr = data_repository_manager<std::shared_ptr<data_batch>>;
using memory_mgr          = ::sirius::memory_manager;

namespace {

void initialize_memory_manager()
{
  memory_mgr::reset_for_testing();
  std::vector<memory_reservation_manager::memory_space_config> configs;
  configs.emplace_back(Tier::GPU, 0, 256 * 1024 * 1024);
  memory_mgr::initialize(std::move(configs));
}

memory_space* get_default_gpu_space()
{
  initialize_memory_manager();
  auto& manager = memory_mgr::get();
  return const_cast<memory_space*>(manager.get_memory_space(Tier::GPU, 0));
}

rmm::device_async_resource_ref get_resource_ref(memory_space& space)
{
  return rmm::to_device_async_resource_ref_checked(space.get_default_allocator());
}

template <typename T>
std::vector<T> copy_column_to_host(const cudf::column_view& col)
{
  std::vector<T> host(col.size());
  if (col.size() > 0) {
    cudaMemcpy(host.data(), col.data<T>(), sizeof(T) * col.size(), cudaMemcpyDeviceToHost);
  }
  return host;
}

std::vector<uint8_t> copy_bool_column_to_host(const cudf::column_view& col)
{
  std::vector<uint8_t> host(col.size());
  if (col.size() > 0) {
    cudaMemcpy(host.data(), col.head(), sizeof(uint8_t) * col.size(), cudaMemcpyDeviceToHost);
  }
  return host;
}

std::vector<std::string> copy_string_column_to_host(const cudf::column_view& col)
{
  std::vector<std::string> host;
  if (col.size() == 0) { return host; }

  cudf::strings_column_view str_col(col);
  std::vector<cudf::size_type> offsets(col.size() + 1);
  cudaMemcpy(offsets.data(),
             str_col.offsets().data<cudf::size_type>(),
             (col.size() + 1) * sizeof(cudf::size_type),
             cudaMemcpyDeviceToHost);

  std::vector<char> chars(offsets.back());
  if (!chars.empty()) {
    cudaMemcpy(chars.data(),
               str_col.chars_begin(cudf::get_default_stream()),
               offsets.back(),
               cudaMemcpyDeviceToHost);
  }

  host.reserve(col.size());
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    auto start = offsets[i];
    auto end   = offsets[i + 1];
    if (chars.empty()) {
      host.emplace_back();
    } else {
      host.emplace_back(chars.data() + start, chars.data() + end);
    }
  }
  return host;
}

std::shared_ptr<data_batch> make_input_batch(
  data_repository_mgr& repo_mgr,
  memory_space& space,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<std::optional<std::pair<int, int>>>& ranges)
{
  auto mr    = get_resource_ref(space);
  auto table = ::sirius::create_cudf_table_with_random_data(
    128, column_types, ranges, cudf::get_default_stream(), mr);
  auto gpu_repr = std::make_unique<gpu_table_representation>(*table, space);
  auto batch_id = repo_mgr.get_next_data_batch_id();
  return std::make_shared<data_batch>(batch_id, repo_mgr, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_int32_batch_with_nulls(data_repository_mgr& repo_mgr,
                                                        memory_space& space,
                                                        const std::vector<int32_t>& values,
                                                        const std::vector<bool>& valids)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto null_mask = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  auto* mask_ptr = static_cast<cudf::bitmask_type*>(null_mask.data());

  cudf::size_type null_count = 0;
  for (cudf::size_type i = 0; i < size; ++i) {
    if (!valids[i]) {
      cudf::set_null_mask(mask_ptr, i, i + 1, false, stream);
      ++null_count;
    }
  }

  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, size, std::move(null_mask), null_count, stream, mr);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<gpu_table_representation>(*table, space);
  auto batch_id = repo_mgr.get_next_data_batch_id();
  return std::make_shared<data_batch>(batch_id, repo_mgr, std::move(gpu_repr));
}

}  // namespace

TEST_CASE("gpu_expression_executor execute(data_batch) projects references",
          "[expression_executor]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 100}},
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 1000}}};

  auto input_batch = make_input_batch(repo_mgr, *space, column_types, ranges);

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  exprs.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BIGINT}, 1));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.execute(input_batch, repo_mgr, cudf::get_default_stream());

  REQUIRE(output_batch != nullptr);
  REQUIRE(output_batch->get_memory_space() == input_batch->get_memory_space());
  REQUIRE(output_batch->get_batch_id() == input_batch->get_batch_id() + 1);

  auto& input_repr  = input_batch->get_data()->cast<gpu_table_representation>();
  auto& output_repr = output_batch->get_data()->cast<gpu_table_representation>();
  auto input_view   = input_repr.get_table().view();
  auto output_view  = output_repr.get_table().view();

  REQUIRE(output_view.num_columns() == input_view.num_columns());
  REQUIRE(output_view.num_rows() == input_view.num_rows());

  auto input_col0  = copy_column_to_host<int32_t>(input_view.column(0));
  auto output_col0 = copy_column_to_host<int32_t>(output_view.column(0));
  REQUIRE(output_col0 == input_col0);

  auto input_col1  = copy_column_to_host<int64_t>(input_view.column(1));
  auto output_col1 = copy_column_to_host<int64_t>(output_view.column(1));
  REQUIRE(output_col1 == input_col1);
}

TEST_CASE("gpu_expression_executor execute(data_batch) handles constants and comparisons",
          "[expression_executor]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 100}},
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 1000}}};

  auto input_batch = make_input_batch(repo_mgr, *space, column_types, ranges);

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  exprs.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(42)));

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BIGINT}, 1);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value::BIGINT(500));
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, std::move(left_expr), std::move(right_expr)));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.execute(input_batch, repo_mgr, cudf::get_default_stream());

  REQUIRE(output_batch != nullptr);

  auto& input_repr  = input_batch->get_data()->cast<gpu_table_representation>();
  auto& output_repr = output_batch->get_data()->cast<gpu_table_representation>();
  auto input_view   = input_repr.get_table().view();
  auto output_view  = output_repr.get_table().view();

  REQUIRE(output_view.num_columns() == 3);
  REQUIRE(output_view.num_rows() == input_view.num_rows());

  auto input_col0  = copy_column_to_host<int32_t>(input_view.column(0));
  auto output_col0 = copy_column_to_host<int32_t>(output_view.column(0));
  REQUIRE(output_col0 == input_col0);

  std::vector<int32_t> expected_constants(input_view.num_rows(), 42);
  auto output_col1 = copy_column_to_host<int32_t>(output_view.column(1));
  REQUIRE(output_col1 == expected_constants);

  auto input_col1 = copy_column_to_host<int64_t>(input_view.column(1));
  std::vector<uint8_t> expected_bool;
  expected_bool.reserve(input_col1.size());
  for (auto value : input_col1) {
    expected_bool.push_back(value < 500 ? 1U : 0U);
  }
  auto output_col2 = copy_bool_column_to_host(output_view.column(2));
  REQUIRE(output_col2 == expected_bool);
}

TEST_CASE("gpu_expression_executor select(data_batch) filters rows", "[expression_executor]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 9}}};

  auto input_batch = make_input_batch(repo_mgr, *space, column_types, ranges);

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5));
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left_expr), std::move(right_expr)));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.select(input_batch, repo_mgr, cudf::get_default_stream());

  REQUIRE(output_batch != nullptr);
  REQUIRE(output_batch->get_memory_space() == input_batch->get_memory_space());
  REQUIRE(output_batch->get_batch_id() == input_batch->get_batch_id() + 1);

  auto& input_repr  = input_batch->get_data()->cast<gpu_table_representation>();
  auto& output_repr = output_batch->get_data()->cast<gpu_table_representation>();
  auto input_view   = input_repr.get_table().view();
  auto output_view  = output_repr.get_table().view();

  REQUIRE(output_view.num_columns() == 1);

  auto input_values = copy_column_to_host<int32_t>(input_view.column(0));
  std::vector<int32_t> expected;
  expected.reserve(input_values.size());
  for (auto value : input_values) {
    if (value > 5) { expected.push_back(value); }
  }

  auto output_values = copy_column_to_host<int32_t>(output_view.column(0));
  REQUIRE(output_values == expected);
  REQUIRE(output_view.num_rows() == static_cast<cudf::size_type>(expected.size()));
}

TEST_CASE("gpu_expression_executor select(data_batch) handles conjunction and IN",
          "[expression_executor]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 9}},
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 50}}};

  auto input_batch = make_input_batch(repo_mgr, *space, column_types, ranges);

  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                            LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(7)));

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BIGINT}, 1);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value::BIGINT(20));
  auto comparison = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHANOREQUALTO, std::move(left_expr), std::move(right_expr));

  auto conjunction = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  conjunction->children.push_back(std::move(in_expr));
  conjunction->children.push_back(std::move(comparison));

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(std::move(conjunction));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.select(input_batch, repo_mgr, cudf::get_default_stream());

  REQUIRE(output_batch != nullptr);
  REQUIRE(output_batch->get_memory_space() == input_batch->get_memory_space());
  REQUIRE(output_batch->get_batch_id() == input_batch->get_batch_id() + 1);

  auto& input_repr  = input_batch->get_data()->cast<gpu_table_representation>();
  auto& output_repr = output_batch->get_data()->cast<gpu_table_representation>();
  auto input_view   = input_repr.get_table().view();
  auto output_view  = output_repr.get_table().view();

  REQUIRE(output_view.num_columns() == input_view.num_columns());

  auto input_col0 = copy_column_to_host<int32_t>(input_view.column(0));
  auto input_col1 = copy_column_to_host<int64_t>(input_view.column(1));
  std::vector<int32_t> expected_col0;
  std::vector<int64_t> expected_col1;
  expected_col0.reserve(input_col0.size());
  expected_col1.reserve(input_col1.size());

  for (size_t i = 0; i < input_col0.size(); ++i) {
    auto value  = input_col0[i];
    bool in_set = value == 1 || value == 3 || value == 5 || value == 7;
    if (in_set && input_col1[i] >= 20) {
      expected_col0.push_back(value);
      expected_col1.push_back(input_col1[i]);
    }
  }

  auto output_col0 = copy_column_to_host<int32_t>(output_view.column(0));
  auto output_col1 = copy_column_to_host<int64_t>(output_view.column(1));
  REQUIRE(output_col0 == expected_col0);
  REQUIRE(output_col1 == expected_col1);
  REQUIRE(output_view.num_rows() == static_cast<cudf::size_type>(expected_col0.size()));
}

TEST_CASE("gpu_expression_executor select(data_batch) handles empty result",
          "[expression_executor]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 9}}};

  auto input_batch = make_input_batch(repo_mgr, *space, column_types, ranges);

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1000));
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left_expr), std::move(right_expr)));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.select(input_batch, repo_mgr, cudf::get_default_stream());

  REQUIRE(output_batch != nullptr);
  REQUIRE(output_batch->get_memory_space() == input_batch->get_memory_space());
  REQUIRE(output_batch->get_batch_id() == input_batch->get_batch_id() + 1);

  auto& input_repr  = input_batch->get_data()->cast<gpu_table_representation>();
  auto& output_repr = output_batch->get_data()->cast<gpu_table_representation>();
  auto input_view   = input_repr.get_table().view();
  auto output_view  = output_repr.get_table().view();

  REQUIRE(output_view.num_columns() == input_view.num_columns());
  REQUIRE(output_view.num_rows() == 0);
}

TEST_CASE("gpu_expression_executor select(data_batch) respects null mask behavior",
          "[expression_executor]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  std::vector<bool> valids    = {true, false, true, false, true};

  auto input_batch = make_int32_batch_with_nulls(repo_mgr, *space, values, valids);

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2));
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left_expr), std::move(right_expr)));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.select(input_batch, repo_mgr, cudf::get_default_stream());

  REQUIRE(output_batch != nullptr);
  REQUIRE(output_batch->get_memory_space() == input_batch->get_memory_space());
  REQUIRE(output_batch->get_batch_id() == input_batch->get_batch_id() + 1);

  auto& output_repr = output_batch->get_data()->cast<gpu_table_representation>();
  auto output_view  = output_repr.get_table().view();

  std::vector<int32_t> expected;
  for (size_t i = 0; i < values.size(); ++i) {
    if (valids[i] && values[i] > 2) { expected.push_back(values[i]); }
  }

  REQUIRE(output_view.num_columns() == 1);
  auto output_values = copy_column_to_host<int32_t>(output_view.column(0));
  REQUIRE(output_values == expected);
  REQUIRE(output_view.num_rows() == static_cast<cudf::size_type>(expected.size()));
}

TEST_CASE("gpu_expression_executor select(data_batch) filters strings", "[expression_executor]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::STRING}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{1, 3}}};

  auto input_batch = make_input_batch(repo_mgr, *space, column_types, ranges);

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value("str_2"));
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, std::move(left_expr), std::move(right_expr)));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.select(input_batch, repo_mgr, cudf::get_default_stream());

  REQUIRE(output_batch != nullptr);
  REQUIRE(output_batch->get_memory_space() == input_batch->get_memory_space());
  REQUIRE(output_batch->get_batch_id() == input_batch->get_batch_id() + 1);

  auto& input_repr  = input_batch->get_data()->cast<gpu_table_representation>();
  auto& output_repr = output_batch->get_data()->cast<gpu_table_representation>();
  auto input_view   = input_repr.get_table().view();
  auto output_view  = output_repr.get_table().view();

  REQUIRE(output_view.num_columns() == 1);

  auto input_strings = copy_string_column_to_host(input_view.column(0));
  std::vector<std::string> expected;
  expected.reserve(input_strings.size());
  for (const auto& value : input_strings) {
    if (value == "str_2") { expected.push_back(value); }
  }

  auto output_strings = copy_string_column_to_host(output_view.column(0));
  REQUIRE(output_strings == expected);
  REQUIRE(output_view.num_rows() == static_cast<cudf::size_type>(expected.size()));
}
