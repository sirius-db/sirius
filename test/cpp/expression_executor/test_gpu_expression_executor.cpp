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
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

// duckdb
#include <duckdb/common/helper.hpp>

// cudf, etc.
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>

using namespace duckdb;
using namespace duckdb::sirius;
using namespace cucascade;
using namespace cucascade::memory;
using memory_mgr = ::sirius::memory::sirius_memory_reservation_manager;

namespace {

std::unique_ptr<memory_mgr> initialize_memory_manager()
{
  ::sirius::converter_registry::reset_for_testing();
  reservation_manager_configurator builder;
  auto constexpr gpu_capacity  = 256ull << 20;  // 256MB
  auto constexpr host_capacity = 512ull << 20;  // 512MB
  auto constexpr limit_ratio   = 0.75;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);
  auto configs = builder.build();
  auto manager = std::make_unique<memory_mgr>(std::move(configs));
  ::sirius::converter_registry::initialize();
  return manager;
}

memory_space* get_default_gpu_space()
{
  static auto manager = initialize_memory_manager();
  return const_cast<memory_space*>(manager->get_memory_space(Tier::GPU, 0));
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
  memory_space& space,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<std::optional<std::pair<int, int>>>& ranges)
{
  auto mr    = get_resource_ref(space);
  auto table = ::sirius::create_cudf_table_with_random_data(
    128, column_types, ranges, cudf::get_default_stream(), mr);
  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<data_batch>(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_int32_batch_with_nulls(memory_space& space,
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

  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<data_batch>(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_decimal64_batch(memory_space& space,
                                                 int8_t scale,
                                                 const std::vector<int64_t>& values)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -scale},
                                           size,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(col->mutable_view().data<int64_t>(),
             values.data(),
             sizeof(int64_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<data_batch>(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_decimal64_two_col_batch(memory_space& space,
                                                         int8_t scale,
                                                         const std::vector<int64_t>& values_a,
                                                         const std::vector<int64_t>& values_b)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values_a.size());

  auto make_col = [&](const std::vector<int64_t>& values) {
    auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -scale},
                                             size,
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr);
    cudaMemcpy(col->mutable_view().data<int64_t>(),
               values.data(),
               sizeof(int64_t) * values.size(),
               cudaMemcpyHostToDevice);
    return col;
  };

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_col(values_a));
  cols.push_back(make_col(values_b));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<data_batch>(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_decimal32_batch(memory_space& space,
                                                 int8_t scale,
                                                 const std::vector<int32_t>& values)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL32, -scale},
                                           size,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<data_batch>(batch_id, std::move(gpu_repr));
}

std::shared_ptr<data_batch> make_decimal128_batch(memory_space& space,
                                                  int8_t scale,
                                                  const std::vector<__int128_t>& values)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();
  auto size   = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL128, -scale},
                                           size,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(col->mutable_view().data<__int128_t>(),
             values.data(),
             sizeof(__int128_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<data_batch>(batch_id, std::move(gpu_repr));
}

}  // namespace

TEST_CASE("gpu_expression_executor execute(data_batch) projects references",
          "[expression_executor]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 100}},
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 1000}}};

  auto input_batch = make_input_batch(*space, column_types, ranges);

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  exprs.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BIGINT}, 1));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.execute(input_batch, cudf::get_default_stream());

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

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 100}},
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 1000}}};

  auto input_batch = make_input_batch(*space, column_types, ranges);

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
  auto output_batch = executor.execute(input_batch, cudf::get_default_stream());

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

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 9}}};

  auto input_batch = make_input_batch(*space, column_types, ranges);

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5));
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left_expr), std::move(right_expr)));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.select(input_batch, cudf::get_default_stream());

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

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 9}},
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 50}}};

  auto input_batch = make_input_batch(*space, column_types, ranges);

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
  auto output_batch = executor.select(input_batch, cudf::get_default_stream());

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

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{0, 9}}};

  auto input_batch = make_input_batch(*space, column_types, ranges);

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1000));
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left_expr), std::move(right_expr)));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.select(input_batch, cudf::get_default_stream());

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

  std::vector<int32_t> values = {1, 2, 3, 4, 5};
  std::vector<bool> valids    = {true, false, true, false, true};

  auto input_batch = make_int32_batch_with_nulls(*space, values, valids);

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2));
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left_expr), std::move(right_expr)));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.select(input_batch, cudf::get_default_stream());

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

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::STRING}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>{std::pair<int, int>{1, 3}}};

  auto input_batch = make_input_batch(*space, column_types, ranges);

  auto left_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0);
  auto right_expr = duckdb::make_uniq<BoundConstantExpression>(Value("str_2"));
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL, std::move(left_expr), std::move(right_expr)));

  GpuExpressionExecutor executor(exprs, get_resource_ref(*space));
  auto output_batch = executor.select(input_batch, cudf::get_default_stream());

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

// =============================================================================
// experimental gpu_expression_executor tests
// =============================================================================

namespace {
using exp_executor = ::sirius::experimental::gpu_expression_executor;
auto constexpr MAT = ::sirius::experimental::expression_executor_strategy::MATERIALIZE;

// Shorthand: build executor, run execute(), return output table view and input table view.
struct exec_result {
  std::shared_ptr<data_batch> input;
  std::shared_ptr<data_batch> output;
  cudf::table_view input_view;
  cudf::table_view output_view;
};

exec_result run_execute(memory_space& space,
                        std::shared_ptr<data_batch> const& input_batch,
                        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs)
{
  exp_executor executor(exprs, MAT, get_resource_ref(space));
  auto output_batch = executor.execute(input_batch);
  REQUIRE(output_batch != nullptr);
  auto& in_repr  = input_batch->get_data()->cast<gpu_table_representation>();
  auto& out_repr = output_batch->get_data()->cast<gpu_table_representation>();
  return {input_batch, output_batch, in_repr.get_table().view(), out_repr.get_table().view()};
}

exec_result run_select(memory_space& space,
                       std::shared_ptr<data_batch> const& input_batch,
                       duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs)
{
  exp_executor executor(exprs, MAT, get_resource_ref(space));
  auto output_batch = executor.select(input_batch);
  REQUIRE(output_batch != nullptr);
  auto& in_repr  = input_batch->get_data()->cast<gpu_table_representation>();
  auto& out_repr = output_batch->get_data()->cast<gpu_table_representation>();
  return {input_batch, output_batch, in_repr.get_table().view(), out_repr.get_table().view()};
}

// Helper: make a BoundFunctionExpression with the given name, arg types, return type, and children.
duckdb::unique_ptr<BoundFunctionExpression> make_func_expr(
  const std::string& name,
  LogicalType const& return_type,
  std::vector<LogicalType> arg_types,
  duckdb::vector<duckdb::unique_ptr<Expression>> children)
{
  auto expr = duckdb::make_uniq<BoundFunctionExpression>(
    return_type,
    ScalarFunction(name, std::move(arg_types), return_type, nullptr),
    duckdb::vector<duckdb::unique_ptr<Expression>>{},
    nullptr);
  for (auto& child : children) {
    expr->children.push_back(std::move(child));
  }
  return expr;
}
}  // namespace

// ---------------------------------------------------------------------------
// execute() — reference, constant, comparison (basic smoke test per type)
// ---------------------------------------------------------------------------

TEST_CASE("experimental execute projects references, constants, and comparisons",
          "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  SECTION("INT32")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
    exprs.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(42)));
    auto lhs = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
    auto rhs = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(50));
    exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
      ExpressionType::COMPARE_GREATERTHAN, std::move(lhs), std::move(rhs)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
    REQUIRE(ov.num_columns() == 3);
    REQUIRE(ov.num_rows() == iv.num_rows());

    auto in0  = copy_column_to_host<int32_t>(iv.column(0));
    auto out0 = copy_column_to_host<int32_t>(ov.column(0));
    REQUIRE(out0 == in0);

    std::vector<int32_t> expected_const(iv.num_rows(), 42);
    REQUIRE(copy_column_to_host<int32_t>(ov.column(1)) == expected_const);

    std::vector<uint8_t> expected_cmp;
    for (auto v : in0) {
      expected_cmp.push_back(v > 50 ? 1U : 0U);
    }
    REQUIRE(copy_bool_column_to_host(ov.column(2)) == expected_cmp);
  }

  SECTION("DECIMAL64")
  {
    uint8_t const scale = 5;
    std::vector<int64_t> raw(128);
    std::iota(raw.begin(), raw.end(), 0);
    auto input = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw);

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType::DECIMAL(18, scale), 0));
    exprs.push_back(
      duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int64_t{42}, 18, scale)));
    auto lhs = duckdb::make_uniq<BoundReferenceExpression>(LogicalType::DECIMAL(18, scale), 0);
    auto rhs = duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int64_t{64}, 18, scale));
    exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
      ExpressionType::COMPARE_LESSTHAN, std::move(lhs), std::move(rhs)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
    REQUIRE(ov.num_columns() == 3);
    REQUIRE(ov.num_rows() == iv.num_rows());

    auto in0 = copy_column_to_host<int64_t>(iv.column(0));
    REQUIRE(copy_column_to_host<int64_t>(ov.column(0)) == in0);

    std::vector<int64_t> expected_const(iv.num_rows(), 42);
    REQUIRE(copy_column_to_host<int64_t>(ov.column(1)) == expected_const);

    std::vector<uint8_t> expected_cmp;
    for (auto v : in0) {
      expected_cmp.push_back(v < 64 ? 1U : 0U);
    }
    REQUIRE(copy_bool_column_to_host(ov.column(2)) == expected_cmp);
  }

  SECTION("STRING")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::STRING}}, {std::pair<int, int>{1, 5}});

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0));
    exprs.push_back(duckdb::make_uniq<BoundConstantExpression>(Value("hello")));
    auto lhs = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0);
    auto rhs = duckdb::make_uniq<BoundConstantExpression>(Value("str_3"));
    exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
      ExpressionType::COMPARE_EQUAL, std::move(lhs), std::move(rhs)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
    REQUIRE(ov.num_columns() == 3);
    REQUIRE(ov.num_rows() == iv.num_rows());

    auto in0 = copy_string_column_to_host(iv.column(0));
    REQUIRE(copy_string_column_to_host(ov.column(0)) == in0);

    std::vector<std::string> expected_const(iv.num_rows(), "hello");
    REQUIRE(copy_string_column_to_host(ov.column(1)) == expected_const);

    std::vector<uint8_t> expected_cmp;
    for (auto const& v : in0) {
      expected_cmp.push_back(v == "str_3" ? 1U : 0U);
    }
    REQUIRE(copy_bool_column_to_host(ov.column(2)) == expected_cmp);
  }
}

// ---------------------------------------------------------------------------
// select() — basic filter + edge cases
// ---------------------------------------------------------------------------

TEST_CASE("experimental select filters rows and handles edge cases",
          "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  SECTION("basic INT32 filter")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 9}});

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    auto lhs = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
    auto rhs = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5));
    exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
      ExpressionType::COMPARE_GREATERTHAN, std::move(lhs), std::move(rhs)));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
    std::vector<int32_t> expected;
    for (auto v : in_vals) {
      if (v > 5) { expected.push_back(v); }
    }
    REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
  }

  SECTION("empty result")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 9}});

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    auto lhs = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
    auto rhs = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1000));
    exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
      ExpressionType::COMPARE_GREATERTHAN, std::move(lhs), std::move(rhs)));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    REQUIRE(ov.num_rows() == 0);
    REQUIRE(ov.num_columns() == iv.num_columns());
  }

  SECTION("DECIMAL64 filter")
  {
    uint8_t const scale = 2;
    std::vector<int64_t> raw(128);
    std::iota(raw.begin(), raw.end(), 0);
    auto input = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw);

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    auto lhs = duckdb::make_uniq<BoundReferenceExpression>(LogicalType::DECIMAL(18, scale), 0);
    auto rhs = duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int64_t{64}, 18, scale));
    exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
      ExpressionType::COMPARE_GREATERTHAN, std::move(lhs), std::move(rhs)));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    auto in_vals                       = copy_column_to_host<int64_t>(iv.column(0));
    std::vector<int64_t> expected;
    for (auto v : in_vals) {
      if (v > 64) { expected.push_back(v); }
    }
    REQUIRE(copy_column_to_host<int64_t>(ov.column(0)) == expected);
  }

  SECTION("STRING equality filter")
  {
    auto input = make_input_batch(
      *space, {cudf::data_type{cudf::type_id::STRING}}, {std::pair<int, int>{1, 3}});

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    auto lhs = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0);
    auto rhs = duckdb::make_uniq<BoundConstantExpression>(Value("str_2"));
    exprs.push_back(duckdb::make_uniq<BoundComparisonExpression>(
      ExpressionType::COMPARE_EQUAL, std::move(lhs), std::move(rhs)));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    auto in_strs                       = copy_string_column_to_host(iv.column(0));
    std::vector<std::string> expected;
    for (auto const& v : in_strs) {
      if (v == "str_2") { expected.push_back(v); }
    }
    REQUIRE(copy_string_column_to_host(ov.column(0)) == expected);
  }
}

// ---------------------------------------------------------------------------
// Arithmetic functions (AST-capable): col + const, col * col
// ---------------------------------------------------------------------------

TEST_CASE("experimental execute arithmetic functions", "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{1, 50}, std::pair<int, int>{1, 10}});

  SECTION("col0 + 10")
  {
    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
    children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(10)));
    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(
      make_func_expr("+",
                     LogicalType{LogicalTypeId::INTEGER},
                     {LogicalType{LogicalTypeId::INTEGER}, LogicalType{LogicalTypeId::INTEGER}},
                     std::move(children)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
    REQUIRE(ov.num_columns() == 1);
    auto in0  = copy_column_to_host<int32_t>(iv.column(0));
    auto out0 = copy_column_to_host<int32_t>(ov.column(0));
    for (size_t i = 0; i < in0.size(); ++i) {
      REQUIRE(out0[i] == in0[i] + 10);
    }
  }

  SECTION("col0 * col1")
  {
    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
    children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1));
    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(
      make_func_expr("*",
                     LogicalType{LogicalTypeId::INTEGER},
                     {LogicalType{LogicalTypeId::INTEGER}, LogicalType{LogicalTypeId::INTEGER}},
                     std::move(children)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
    auto in0                           = copy_column_to_host<int32_t>(iv.column(0));
    auto in1                           = copy_column_to_host<int32_t>(iv.column(1));
    auto out0                          = copy_column_to_host<int32_t>(ov.column(0));
    for (size_t i = 0; i < in0.size(); ++i) {
      REQUIRE(out0[i] == in0[i] * in1[i]);
    }
  }

  SECTION("col0 % 3")
  {
    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
    children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));
    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(
      make_func_expr("%",
                     LogicalType{LogicalTypeId::INTEGER},
                     {LogicalType{LogicalTypeId::INTEGER}, LogicalType{LogicalTypeId::INTEGER}},
                     std::move(children)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
    auto in0                           = copy_column_to_host<int32_t>(iv.column(0));
    auto out0                          = copy_column_to_host<int32_t>(ov.column(0));
    for (size_t i = 0; i < in0.size(); ++i) {
      REQUIRE(out0[i] == in0[i] % 3);
    }
  }
}

// ---------------------------------------------------------------------------
// Decimal arithmetic (MATERIALIZE path — AST is disabled for decimal return
// types pending cudf#21996, so all decimal function evaluation flows through
// cudf::binary_operation on fixed_point columns/scalars).
// ---------------------------------------------------------------------------

TEST_CASE("experimental execute decimal arithmetic (DECIMAL64)",
          "[expression_executor][experimental][decimal]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width = 18;
  uint8_t const scale = 2;
  auto const dec_type = LogicalType::DECIMAL(width, scale);

  // Values: 0.00, 0.01, ... 0.09 (stored as 0, 1, ... 9 at scale 2)
  std::vector<int64_t> raw_a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> raw_b = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  SECTION("col + literal (col + 1.25)")
  {
    auto input = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw_a);

    // 1.25 at scale 2 → raw value 125
    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(duckdb::make_uniq<BoundReferenceExpression>(dec_type, 0));
    children.push_back(
      duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int64_t{125}, width, scale)));

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(make_func_expr("+", dec_type, {dec_type, dec_type}, std::move(children)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
    REQUIRE(ov.num_columns() == 1);
    REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
    REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

    auto out0 = copy_column_to_host<int64_t>(ov.column(0));
    std::vector<int64_t> expected;
    expected.reserve(raw_a.size());
    for (auto v : raw_a) {
      expected.push_back(v + 125);
    }
    REQUIRE(out0 == expected);
  }

  SECTION("col - col (two-column batch)")
  {
    auto input = make_decimal64_two_col_batch(*space, static_cast<int8_t>(scale), raw_b, raw_a);

    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(duckdb::make_uniq<BoundReferenceExpression>(dec_type, 0));
    children.push_back(duckdb::make_uniq<BoundReferenceExpression>(dec_type, 1));

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(make_func_expr("-", dec_type, {dec_type, dec_type}, std::move(children)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
    REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
    REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

    auto out0 = copy_column_to_host<int64_t>(ov.column(0));
    std::vector<int64_t> expected;
    expected.reserve(raw_a.size());
    for (size_t i = 0; i < raw_a.size(); ++i) {
      expected.push_back(raw_b[i] - raw_a[i]);
    }
    REQUIRE(out0 == expected);
  }

  SECTION("col * integer literal (decimal with scale 0)")
  {
    auto input = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw_a);

    // 2 as DECIMAL(18, 0) → same cudf type_id (DECIMAL64), scale 0.
    // cudf MUL: output scale = lhs_scale + rhs_scale = -2 + 0 = -2.
    auto const dec_int_type = LogicalType::DECIMAL(width, 0);

    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(duckdb::make_uniq<BoundReferenceExpression>(dec_type, 0));
    children.push_back(
      duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int64_t{2}, width, uint8_t{0})));

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(make_func_expr("*", dec_type, {dec_type, dec_int_type}, std::move(children)));

    auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
    REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
    REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

    auto out0 = copy_column_to_host<int64_t>(ov.column(0));
    std::vector<int64_t> expected;
    expected.reserve(raw_a.size());
    for (auto v : raw_a) {
      expected.push_back(v * 2);
    }
    REQUIRE(out0 == expected);
  }
}

TEST_CASE("experimental execute decimal arithmetic (DECIMAL32)",
          "[expression_executor][experimental][decimal]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Width 8 → INT32 physical (DuckDB) → DECIMAL32 (cudf). Exercises the
  // DECIMAL32 branches in gpu_execute_constant.cpp and GetCudfType.
  uint8_t const width = 8;
  uint8_t const scale = 2;
  auto const dec_type = LogicalType::DECIMAL(width, scale);

  // 1.00, 2.00, 3.00 ... 10.00 (raw: 100, 200, ... 1000 at scale 2)
  std::vector<int32_t> raw = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
  auto input               = make_decimal32_batch(*space, static_cast<int8_t>(scale), raw);

  // col0 + 0.50 → raw add 50
  duckdb::vector<duckdb::unique_ptr<Expression>> children;
  children.push_back(duckdb::make_uniq<BoundReferenceExpression>(dec_type, 0));
  children.push_back(
    duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int32_t{50}, width, scale)));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(make_func_expr("+", dec_type, {dec_type, dec_type}, std::move(children)));

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL32);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

  auto out0 = copy_column_to_host<int32_t>(ov.column(0));
  std::vector<int32_t> expected;
  expected.reserve(raw.size());
  for (auto v : raw) {
    expected.push_back(v + 50);
  }
  REQUIRE(out0 == expected);
}

TEST_CASE("experimental execute nested decimal arithmetic (col + 1.00) * 2",
          "[expression_executor][experimental][decimal]")
{
  // Regression coverage: before disabling AST for decimal-returning functions,
  // nested decimal arithmetic would feed intermediate fixed_point results into
  // the cudf AST kernel and trip cudf#21996. With AST disabled for decimal
  // return types, the inner `+` materializes via cudf::binary_operation and the
  // outer `*` consumes the materialized column via another cudf::binary_operation.
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width = 18;
  uint8_t const scale = 2;
  auto const dec_type = LogicalType::DECIMAL(width, scale);
  auto const dec_int  = LogicalType::DECIMAL(width, 0);

  // 0.00, 0.01 ... 0.09
  std::vector<int64_t> raw = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto input               = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw);

  // Inner: col + 1.00 (both scale 2) → result scale 2
  duckdb::vector<duckdb::unique_ptr<Expression>> add_children;
  add_children.push_back(duckdb::make_uniq<BoundReferenceExpression>(dec_type, 0));
  add_children.push_back(
    duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int64_t{100}, width, scale)));
  auto add_expr = make_func_expr("+", dec_type, {dec_type, dec_type}, std::move(add_children));

  // Outer: (col + 1.00) * 2 where 2 is DECIMAL(18, 0) so MUL output scale = -2 + 0 = -2
  duckdb::vector<duckdb::unique_ptr<Expression>> mul_children;
  mul_children.push_back(std::move(add_expr));
  mul_children.push_back(
    duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int64_t{2}, width, uint8_t{0})));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(make_func_expr("*", dec_type, {dec_type, dec_int}, std::move(mul_children)));

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

  auto out0 = copy_column_to_host<int64_t>(ov.column(0));
  std::vector<int64_t> expected;
  expected.reserve(raw.size());
  for (auto v : raw) {
    expected.push_back((v + 100) * 2);
  }
  REQUIRE(out0 == expected);
}

TEST_CASE("experimental execute decimal arithmetic (DECIMAL128)",
          "[expression_executor][experimental][decimal]")
{
  // Width 38 → INT128 physical (DuckDB) → DECIMAL128 (cudf). Exercises the
  // hugeint_t → __int128_t conversion branch in gpu_execute_constant.cpp for
  // decimal constants, plus cudf::binary_operation on DECIMAL128 columns.
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width = 38;
  uint8_t const scale = 2;
  auto const dec_type = LogicalType::DECIMAL(width, scale);

  // 1.00, 2.00, 3.00 (raw: 100, 200, 300 at scale 2)
  std::vector<__int128_t> raw = {100, 200, 300};
  auto input                  = make_decimal128_batch(*space, static_cast<int8_t>(scale), raw);

  // col + 1.25 → raw add 125
  duckdb::vector<duckdb::unique_ptr<Expression>> children;
  children.push_back(duckdb::make_uniq<BoundReferenceExpression>(dec_type, 0));
  children.push_back(duckdb::make_uniq<BoundConstantExpression>(
    Value::DECIMAL(duckdb::hugeint_t{125}, width, scale)));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(make_func_expr("+", dec_type, {dec_type, dec_type}, std::move(children)));

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL128);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

  auto out0 = copy_column_to_host<__int128_t>(ov.column(0));
  REQUIRE(out0.size() == raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    // __int128_t has no stream operator, so compare via int64 cast (values fit).
    REQUIRE(static_cast<int64_t>(out0[i]) == static_cast<int64_t>(raw[i] + 125));
  }
}

TEST_CASE("experimental execute decimal DIV (DECIMAL64)",
          "[expression_executor][experimental][decimal]")
{
  // DIV has a distinct output-scale rule from ADD/SUB/MUL:
  //   cudf fixed_point DIV output scale = lhs_scale - rhs_scale
  // Here: col(scale -2) / literal(scale 0) → output scale -2.
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width = 18;
  uint8_t const scale = 2;
  auto const dec_type = LogicalType::DECIMAL(width, scale);
  auto const dec_int  = LogicalType::DECIMAL(width, 0);

  // 4.00, 5.00, 6.00, 7.00 (raw 400, 500, 600, 700 at scale 2)
  std::vector<int64_t> raw = {400, 500, 600, 700};
  auto input               = make_decimal64_batch(*space, static_cast<int8_t>(scale), raw);

  // col / 2 (2 as DECIMAL(18, 0)) → fixed_point div truncates toward zero in
  // the output scale; 500/2 = 250, 700/2 = 350.
  duckdb::vector<duckdb::unique_ptr<Expression>> children;
  children.push_back(duckdb::make_uniq<BoundReferenceExpression>(dec_type, 0));
  children.push_back(
    duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int64_t{2}, width, uint8_t{0})));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(make_func_expr("/", dec_type, {dec_type, dec_int}, std::move(children)));

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale));

  auto out0 = copy_column_to_host<int64_t>(ov.column(0));
  std::vector<int64_t> expected;
  expected.reserve(raw.size());
  for (auto v : raw) {
    expected.push_back(v / 2);
  }
  REQUIRE(out0 == expected);
}

TEST_CASE("experimental execute decimal TPC-H Q1 shape price * (1 - discount)",
          "[expression_executor][experimental][decimal]")
{
  // Mirrors TPC-H Q1's `l_extendedprice * (1 - l_discount)`:
  //   inner SUB has a scalar on the LEFT (1.00) and a column on the RIGHT
  //   (discount), which hits the left-scalar branch of execute_numeric_binary_func;
  //   outer MUL then consumes the materialized inner result as a column.
  //   MUL output scale = lhs_scale + rhs_scale = -2 + -2 = -4, so the return
  //   type is DECIMAL(18, 4).
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  uint8_t const width     = 18;
  uint8_t const scale2    = 2;
  uint8_t const scale4    = 4;
  auto const price_type   = LogicalType::DECIMAL(width, scale2);  // DECIMAL(18,2)
  auto const discount_t   = LogicalType::DECIMAL(width, scale2);  // DECIMAL(18,2)
  auto const mul_ret_type = LogicalType::DECIMAL(width, scale4);  // DECIMAL(18,4)

  // price: 1000.00, 2000.00, 3000.00
  std::vector<int64_t> raw_price = {100000, 200000, 300000};
  // discount: 0.05, 0.10, 0.15
  std::vector<int64_t> raw_discount = {5, 10, 15};
  auto input =
    make_decimal64_two_col_batch(*space, static_cast<int8_t>(scale2), raw_price, raw_discount);

  // Inner: 1.00 - discount (scalar on left, column on right)
  duckdb::vector<duckdb::unique_ptr<Expression>> sub_children;
  sub_children.push_back(
    duckdb::make_uniq<BoundConstantExpression>(Value::DECIMAL(int64_t{100}, width, scale2)));
  sub_children.push_back(duckdb::make_uniq<BoundReferenceExpression>(discount_t, 1));
  auto sub_expr =
    make_func_expr("-", discount_t, {discount_t, discount_t}, std::move(sub_children));

  // Outer: price * (1.00 - discount) → return type DECIMAL(18, 4)
  duckdb::vector<duckdb::unique_ptr<Expression>> mul_children;
  mul_children.push_back(duckdb::make_uniq<BoundReferenceExpression>(price_type, 0));
  mul_children.push_back(std::move(sub_expr));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(
    make_func_expr("*", mul_ret_type, {price_type, discount_t}, std::move(mul_children)));

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.column(0).type().id() == cudf::type_id::DECIMAL64);
  REQUIRE(ov.column(0).type().scale() == -static_cast<int32_t>(scale4));

  // Expected: price * (100 - discount) interpreted at scale -4.
  //   1000.00 * 0.95 = 950.0000 → raw 100000 * 95 = 9500000
  //   2000.00 * 0.90 = 1800.0000 → raw 200000 * 90 = 18000000
  //   3000.00 * 0.85 = 2550.0000 → raw 300000 * 85 = 25500000
  auto out0 = copy_column_to_host<int64_t>(ov.column(0));
  std::vector<int64_t> expected;
  expected.reserve(raw_price.size());
  for (size_t i = 0; i < raw_price.size(); ++i) {
    expected.push_back(raw_price[i] * (100 - raw_discount[i]));
  }
  REQUIRE(out0 == expected);
}

// ---------------------------------------------------------------------------
// LIKE / NOT LIKE (AST breakers — string functions always materialize)
// ---------------------------------------------------------------------------

TEST_CASE("experimental select LIKE and NOT LIKE", "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Generates strings "str_1", "str_2", "str_3"
  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::STRING}}, {std::pair<int, int>{1, 3}});

  SECTION("LIKE 'str_2'")
  {
    // col0 LIKE 'str_2' — exact match via LIKE
    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0));
    children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value("str_2")));
    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(
      make_func_expr("~~",
                     LogicalType{LogicalTypeId::BOOLEAN},
                     {LogicalType{LogicalTypeId::VARCHAR}, LogicalType{LogicalTypeId::VARCHAR}},
                     std::move(children)));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    auto in_strs                       = copy_string_column_to_host(iv.column(0));
    std::vector<std::string> expected;
    for (auto const& s : in_strs) {
      if (s == "str_2") { expected.push_back(s); }
    }
    REQUIRE(copy_string_column_to_host(ov.column(0)) == expected);
  }

  SECTION("LIKE 'str_%' — wildcard")
  {
    // col0 LIKE 'str_%' — should match all rows
    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0));
    children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value("str_%")));
    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(
      make_func_expr("~~",
                     LogicalType{LogicalTypeId::BOOLEAN},
                     {LogicalType{LogicalTypeId::VARCHAR}, LogicalType{LogicalTypeId::VARCHAR}},
                     std::move(children)));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    REQUIRE(ov.num_rows() == iv.num_rows());
  }

  SECTION("NOT LIKE 'str_1'")
  {
    // col0 NOT LIKE 'str_1' — should exclude 'str_1' rows
    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0));
    children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value("str_1")));
    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(
      make_func_expr("!~~",
                     LogicalType{LogicalTypeId::BOOLEAN},
                     {LogicalType{LogicalTypeId::VARCHAR}, LogicalType{LogicalTypeId::VARCHAR}},
                     std::move(children)));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    auto in_strs                       = copy_string_column_to_host(iv.column(0));
    std::vector<std::string> expected;
    for (auto const& s : in_strs) {
      if (s != "str_1") { expected.push_back(s); }
    }
    REQUIRE(copy_string_column_to_host(ov.column(0)) == expected);
  }
}

// ---------------------------------------------------------------------------
// CASE/WHEN (always materializes — AST breaker)
// ---------------------------------------------------------------------------

TEST_CASE("experimental execute CASE expression", "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // CASE WHEN col0 > 50 THEN 1 ELSE 0 END
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

  auto when_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(50)));
  auto then_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1));
  auto else_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0));

  auto case_expr = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
  BoundCaseCheck check;
  check.when_expr = std::move(when_expr);
  check.then_expr = std::move(then_expr);
  case_expr->case_checks.push_back(std::move(check));
  case_expr->else_expr = std::move(else_expr);

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(std::move(case_expr));

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
  REQUIRE(ov.num_columns() == 1);
  REQUIRE(ov.num_rows() == iv.num_rows());

  auto in_vals  = copy_column_to_host<int32_t>(iv.column(0));
  auto out_vals = copy_column_to_host<int32_t>(ov.column(0));
  for (size_t i = 0; i < in_vals.size(); ++i) {
    REQUIRE(out_vals[i] == (in_vals[i] > 50 ? 1 : 0));
  }
}

TEST_CASE("experimental execute CASE with multiple WHEN branches",
          "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // CASE WHEN col0 < 30 THEN -1 WHEN col0 > 70 THEN 1 ELSE 0 END
  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

  auto case_expr = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});

  BoundCaseCheck check1;
  check1.when_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(30)));
  check1.then_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(-1));
  case_expr->case_checks.push_back(std::move(check1));

  BoundCaseCheck check2;
  check2.when_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(70)));
  check2.then_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1));
  case_expr->case_checks.push_back(std::move(check2));

  case_expr->else_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(std::move(case_expr));

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  auto out_vals                      = copy_column_to_host<int32_t>(ov.column(0));
  for (size_t i = 0; i < in_vals.size(); ++i) {
    int32_t expected = in_vals[i] < 30 ? -1 : (in_vals[i] > 70 ? 1 : 0);
    REQUIRE(out_vals[i] == expected);
  }
}

// ---------------------------------------------------------------------------
// BETWEEN (decomposed into two comparisons + AND)
// ---------------------------------------------------------------------------

TEST_CASE("experimental select BETWEEN", "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

  // col0 BETWEEN 20 AND 50 (inclusive)
  auto between = duckdb::make_uniq<BoundBetweenExpression>(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(20)),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(50)),
    true,   // lower_inclusive
    true);  // upper_inclusive

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(std::move(between));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  std::vector<int32_t> expected;
  for (auto v : in_vals) {
    if (v >= 20 && v <= 50) { expected.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
}

// ---------------------------------------------------------------------------
// IN / NOT IN (AST breaker when constant list — uses cudf::contains)
// ---------------------------------------------------------------------------

TEST_CASE("experimental select IN and NOT IN", "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 9}});

  SECTION("IN (2, 4, 6)")
  {
    auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
    in_expr->children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
    in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));
    in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(4)));
    in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(6)));

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(std::move(in_expr));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
    std::vector<int32_t> expected;
    for (auto v : in_vals) {
      if (v == 2 || v == 4 || v == 6) { expected.push_back(v); }
    }
    REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
  }

  SECTION("NOT IN (0, 1, 2, 3)")
  {
    auto not_in_expr = duckdb::make_uniq<BoundOperatorExpression>(
      ExpressionType::COMPARE_NOT_IN, LogicalType{LogicalTypeId::BOOLEAN});
    not_in_expr->children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
    not_in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0)));
    not_in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
    not_in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));
    not_in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(std::move(not_in_expr));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
    std::vector<int32_t> expected;
    for (auto v : in_vals) {
      if (v != 0 && v != 1 && v != 2 && v != 3) { expected.push_back(v); }
    }
    REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
  }
}

// ---------------------------------------------------------------------------
// IS NULL / IS NOT NULL / NOT (operator expressions)
// ---------------------------------------------------------------------------

TEST_CASE("experimental select IS NULL and IS NOT NULL", "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  std::vector<int32_t> values = {10, 20, 30, 40, 50};
  std::vector<bool> valids    = {true, false, true, false, true};
  auto input                  = make_int32_batch_with_nulls(*space, values, valids);

  SECTION("IS NULL")
  {
    auto is_null = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL,
                                                              LogicalType{LogicalTypeId::BOOLEAN});
    is_null->children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(std::move(is_null));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    // Null rows should pass the IS NULL filter
    REQUIRE(ov.num_rows() == 2);
  }

  SECTION("IS NOT NULL")
  {
    auto is_not_null = duckdb::make_uniq<BoundOperatorExpression>(
      ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType{LogicalTypeId::BOOLEAN});
    is_not_null->children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));

    duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
    exprs.push_back(std::move(is_not_null));

    auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
    // Non-null rows should pass
    REQUIRE(ov.num_rows() == 3);
    auto out_vals = copy_column_to_host<int32_t>(ov.column(0));
    REQUIRE(out_vals == std::vector<int32_t>{10, 30, 50});
  }
}

TEST_CASE("experimental select NOT operator", "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 9}});

  // NOT (col0 > 5) — should keep rows where col0 <= 5
  auto cmp = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5)));

  auto not_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_NOT,
                                                             LogicalType{LogicalTypeId::BOOLEAN});
  not_expr->children.push_back(std::move(cmp));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(std::move(not_expr));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  std::vector<int32_t> expected;
  for (auto v : in_vals) {
    if (v <= 5) { expected.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
}

// ---------------------------------------------------------------------------
// Conjunction with AST breaker: AND/OR mixing AST-capable and non-AST nodes
// ---------------------------------------------------------------------------

TEST_CASE("experimental select conjunction with AST breaker", "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // Table: (INT32 col0, STRING col1)
  auto input = make_input_batch(
    *space,
    {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::STRING}},
    {std::pair<int, int>{0, 9}, std::pair<int, int>{1, 5}});

  // WHERE col0 > 3 AND col1 LIKE 'str_%'
  // col0 > 3 is AST-capable, LIKE is not — forces mixed materialization
  auto cmp = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  duckdb::vector<duckdb::unique_ptr<Expression>> like_children;
  like_children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 1));
  like_children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value("str_%")));
  auto like =
    make_func_expr("~~",
                   LogicalType{LogicalTypeId::BOOLEAN},
                   {LogicalType{LogicalTypeId::VARCHAR}, LogicalType{LogicalTypeId::VARCHAR}},
                   std::move(like_children));

  auto conjunction = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  conjunction->children.push_back(std::move(cmp));
  conjunction->children.push_back(std::move(like));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(std::move(conjunction));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
  auto in_col0                       = copy_column_to_host<int32_t>(iv.column(0));
  auto in_col1                       = copy_string_column_to_host(iv.column(1));

  std::vector<int32_t> expected_col0;
  for (auto v : in_col0) {
    // "str_%" matches everything produced by make_input_batch (all start with "str_")
    if (v > 3) { expected_col0.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected_col0);
}

TEST_CASE("experimental select OR conjunction", "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 20}});

  // col0 < 3 OR col0 > 17
  auto lt = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));
  auto gt = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(17)));

  auto disjunction = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_OR);
  disjunction->children.push_back(std::move(lt));
  disjunction->children.push_back(std::move(gt));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(std::move(disjunction));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  std::vector<int32_t> expected;
  for (auto v : in_vals) {
    if (v < 3 || v > 17) { expected.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
}

// ---------------------------------------------------------------------------
// Nested compound: CASE inside a comparison inside a conjunction
// Exercises AST breaker (CASE) nested within AST-capable nodes
// ---------------------------------------------------------------------------

TEST_CASE("experimental select with nested CASE in predicate",
          "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input = make_input_batch(
    *space, {cudf::data_type{cudf::type_id::INT32}}, {std::pair<int, int>{0, 100}});

  // WHERE (CASE WHEN col0 > 50 THEN col0 ELSE 0 END) > 75
  auto case_expr = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
  BoundCaseCheck check;
  check.when_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(50)));
  check.then_expr =
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  case_expr->case_checks.push_back(std::move(check));
  case_expr->else_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0));

  auto outer_cmp = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    std::move(case_expr),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(75)));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(std::move(outer_cmp));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
  auto in_vals                       = copy_column_to_host<int32_t>(iv.column(0));
  std::vector<int32_t> expected;
  for (auto v : in_vals) {
    int32_t case_result = v > 50 ? v : 0;
    if (case_result > 75) { expected.push_back(v); }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected);
}

// ---------------------------------------------------------------------------
// IN with conjunction (multi-column filter, mirrors a TPC-H style predicate)
// ---------------------------------------------------------------------------

TEST_CASE("experimental select IN with conjunction multi-column",
          "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}},
                     {std::pair<int, int>{0, 9}, std::pair<int, int>{0, 50}});

  // WHERE col0 IN (1, 3, 5, 7) AND col1 >= 20
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                            LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(7)));

  auto cmp = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHANOREQUALTO,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BIGINT}, 1),
    duckdb::make_uniq<BoundConstantExpression>(Value::BIGINT(20)));

  auto conjunction = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  conjunction->children.push_back(std::move(in_expr));
  conjunction->children.push_back(std::move(cmp));

  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
  exprs.push_back(std::move(conjunction));

  auto [in_batch, out_batch, iv, ov] = run_select(*space, input, exprs);
  auto in_col0                       = copy_column_to_host<int32_t>(iv.column(0));
  auto in_col1                       = copy_column_to_host<int64_t>(iv.column(1));
  std::vector<int32_t> expected_col0;
  std::vector<int64_t> expected_col1;
  for (size_t i = 0; i < in_col0.size(); ++i) {
    bool in_set = in_col0[i] == 1 || in_col0[i] == 3 || in_col0[i] == 5 || in_col0[i] == 7;
    if (in_set && in_col1[i] >= 20) {
      expected_col0.push_back(in_col0[i]);
      expected_col1.push_back(in_col1[i]);
    }
  }
  REQUIRE(copy_column_to_host<int32_t>(ov.column(0)) == expected_col0);
  REQUIRE(copy_column_to_host<int64_t>(ov.column(1)) == expected_col1);
}

// ---------------------------------------------------------------------------
// Arithmetic in projection combined with CASE — complex execute() output
// ---------------------------------------------------------------------------

TEST_CASE("experimental execute mixed arithmetic and CASE projection",
          "[expression_executor][experimental]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto input =
    make_input_batch(*space,
                     {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                     {std::pair<int, int>{1, 50}, std::pair<int, int>{1, 10}});

  // Output: [col0 + col1, CASE WHEN col0 > 25 THEN col1 * 2 ELSE col1 END]
  duckdb::vector<duckdb::unique_ptr<Expression>> exprs;

  // Expression 0: col0 + col1
  {
    duckdb::vector<duckdb::unique_ptr<Expression>> children;
    children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
    children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1));
    exprs.push_back(
      make_func_expr("+",
                     LogicalType{LogicalTypeId::INTEGER},
                     {LogicalType{LogicalTypeId::INTEGER}, LogicalType{LogicalTypeId::INTEGER}},
                     std::move(children)));
  }

  // Expression 1: CASE WHEN col0 > 25 THEN col1 * 2 ELSE col1 END
  {
    duckdb::vector<duckdb::unique_ptr<Expression>> mul_children;
    mul_children.push_back(
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1));
    mul_children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));
    auto then_expr =
      make_func_expr("*",
                     LogicalType{LogicalTypeId::INTEGER},
                     {LogicalType{LogicalTypeId::INTEGER}, LogicalType{LogicalTypeId::INTEGER}},
                     std::move(mul_children));

    auto case_expr = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
    BoundCaseCheck check;
    check.when_expr = duckdb::make_uniq<BoundComparisonExpression>(
      ExpressionType::COMPARE_GREATERTHAN,
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
      duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(25)));
    check.then_expr = std::move(then_expr);
    case_expr->case_checks.push_back(std::move(check));
    case_expr->else_expr =
      duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1);
    exprs.push_back(std::move(case_expr));
  }

  auto [in_batch, out_batch, iv, ov] = run_execute(*space, input, exprs);
  REQUIRE(ov.num_columns() == 2);
  REQUIRE(ov.num_rows() == iv.num_rows());

  auto in0  = copy_column_to_host<int32_t>(iv.column(0));
  auto in1  = copy_column_to_host<int32_t>(iv.column(1));
  auto out0 = copy_column_to_host<int32_t>(ov.column(0));
  auto out1 = copy_column_to_host<int32_t>(ov.column(1));
  for (size_t i = 0; i < in0.size(); ++i) {
    REQUIRE(out0[i] == in0[i] + in1[i]);
    REQUIRE(out1[i] == (in0[i] > 25 ? in1[i] * 2 : in1[i]));
  }
}
