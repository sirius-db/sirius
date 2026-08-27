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

#include "op/aggregate/group_key_labels.hpp"
#include "operator/operator_test_utils.hpp"
#include "operator/operator_type_traits.hpp"
#include "utils/data_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace sirius::test::operator_utils;
using sirius::op::detail::make_group_key_labels;
using sirius::test::vector_to_cudf_column;

std::unique_ptr<cudf::table> make_table(std::vector<std::unique_ptr<cudf::column>> columns)
{
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> make_table(std::unique_ptr<cudf::column> column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return make_table(std::move(columns));
}

std::unique_ptr<cudf::column> with_nulls(
  std::unique_ptr<cudf::column> column,
  std::vector<bool> const& valid,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  REQUIRE(static_cast<std::size_t>(column->size()) == valid.size());
  auto valid_column       = vector_to_cudf_column<gpu_type_traits<bool>>(valid, stream, mr);
  auto [mask, null_count] = cudf::bools_to_mask(valid_column->view(), stream, mr);
  column->set_null_mask(std::move(*mask), null_count);
  return column;
}

void require_columns_equal(cudf::column_view const& actual,
                           cudf::column_view const& expected,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  REQUIRE(actual.type() == expected.type());
  REQUIRE(actual.size() == expected.size());
  REQUIRE(actual.null_count() == expected.null_count());
  if (actual.is_empty()) { return; }

  auto equal           = cudf::binary_operation(actual,
                                      expected,
                                      cudf::binary_operator::NULL_EQUALS,
                                      cudf::data_type{cudf::type_id::BOOL8},
                                      stream,
                                      mr);
  auto all_aggregation = cudf::make_all_aggregation<cudf::reduce_aggregation>();
  auto all_equal       = cudf::reduce(
    equal->view(), *all_aggregation, cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
  REQUIRE(static_cast<cudf::numeric_scalar<bool>&>(*all_equal).value(stream));
}

void require_tables_equal(cudf::table_view const& actual,
                          cudf::table_view const& expected,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  REQUIRE(actual.num_columns() == expected.num_columns());
  REQUIRE(actual.num_rows() == expected.num_rows());
  for (cudf::size_type index = 0; index < actual.num_columns(); ++index) {
    require_columns_equal(actual.column(index), expected.column(index), stream, mr);
  }
}

void require_encode_contract(
  cudf::table_view const& keys,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto expected = cudf::encode(keys, stream, mr);
  auto actual   = make_group_key_labels(keys, stream, mr);

  require_tables_equal(actual.sorted_unique_keys->view(), expected.first->view(), stream, mr);
  require_columns_equal(actual.labels->view(), expected.second->view(), stream, mr);

  auto round_trip = cudf::gather(actual.sorted_unique_keys->view(),
                                 actual.labels->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
  require_tables_equal(round_trip->view(), keys, stream, mr);
}

template <typename T>
struct host_column {
  std::vector<T> values;
  std::vector<std::uint8_t> valid;
};

template <typename T>
host_column<T> copy_to_host(cudf::column_view const& column,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  host_column<T> result{std::vector<T>(column.size()), std::vector<std::uint8_t>(column.size(), 1)};
  REQUIRE(cudaMemcpyAsync(result.values.data(),
                          column.data<T>(),
                          result.values.size() * sizeof(T),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);

  std::unique_ptr<cudf::column> validity;
  if (column.nullable()) {
    validity = cudf::mask_to_bools(
      column.null_mask(), column.offset(), column.offset() + column.size(), stream, mr);
    REQUIRE(cudaMemcpyAsync(result.valid.data(),
                            validity->view().data<bool>(),
                            result.valid.size(),
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
  }
  stream.synchronize();
  return result;
}

void require_floating_columns_equivalent(cudf::column_view const& actual,
                                         cudf::column_view const& expected,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  REQUIRE(actual.type() == expected.type());
  REQUIRE(actual.size() == expected.size());
  auto const actual_host   = copy_to_host<double>(actual, stream, mr);
  auto const expected_host = copy_to_host<double>(expected, stream, mr);
  for (std::size_t index = 0; index < actual_host.values.size(); ++index) {
    REQUIRE(actual_host.valid[index] == expected_host.valid[index]);
    if (actual_host.valid[index] != 0) {
      auto const both_nan =
        std::isnan(actual_host.values[index]) && std::isnan(expected_host.values[index]);
      REQUIRE((both_nan || actual_host.values[index] == expected_host.values[index]));
    }
  }
}

}  // namespace

TEST_CASE("group key labels match encode for Q16-like mixed keys", "[aggregate][group_key_labels]")
{
  std::vector<std::string> const brands = {"Brand#12", "Brand#34", "Brand#45", "Brand#51"};
  std::vector<std::string> const types  = {"STANDARD ANODIZED TIN", "SMALL PLATED COPPER", "PROMO"};
  std::vector<int32_t> const sizes      = {3, 9, 14, 23};

  std::vector<std::string> brand_column;
  std::vector<std::string> type_column;
  std::vector<int32_t> size_column;
  std::minstd_rand random{7};
  for (int index = 0; index < 4096; ++index) {
    brand_column.push_back(brands[random() % brands.size()]);
    type_column.push_back(types[random() % types.size()]);
    size_column.push_back(sizes[random() % sizes.size()]);
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(brand_column));
  columns.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(type_column));
  columns.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>(size_column));
  auto keys = make_table(std::move(columns));

  require_encode_contract(keys->view());
}

TEST_CASE("group key labels handle empty and degenerate inputs", "[aggregate][group_key_labels]")
{
  SECTION("empty input preserves schema")
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>({}));
    columns.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>({}));
    auto keys   = make_table(std::move(columns));
    auto result = make_group_key_labels(
      keys->view(), cudf::get_default_stream(), cudf::get_current_device_resource_ref());

    REQUIRE(result.sorted_unique_keys->num_columns() == 2);
    REQUIRE(result.sorted_unique_keys->num_rows() == 0);
    REQUIRE(result.labels->type().id() == cudf::type_id::INT32);
    REQUIRE(result.labels->size() == 0);
    REQUIRE_FALSE(result.labels->nullable());
  }

  SECTION("one row")
  {
    auto keys = make_table(
      vector_to_cudf_column<gpu_type_traits<string_tag>>(std::vector<std::string>{"one"}));
    require_encode_contract(keys->view());
  }

  SECTION("one repeated group")
  {
    auto keys =
      make_table(vector_to_cudf_column<gpu_type_traits<int32_t>>(std::vector<int32_t>(1024, 42)));
    require_encode_contract(keys->view());
  }

  SECTION("all rows null")
  {
    auto keys = make_table(
      with_nulls(vector_to_cudf_column<gpu_type_traits<int32_t>>(std::vector<int32_t>(257, 0)),
                 std::vector<bool>(257, false)));
    require_encode_contract(keys->view());
  }

  SECTION("all rows distinct")
  {
    std::vector<int32_t> values(4096);
    for (int32_t index = 0; index < static_cast<int32_t>(values.size()); ++index) {
      values[index] = index * 7919;
    }
    auto keys = make_table(vector_to_cudf_column<gpu_type_traits<int32_t>>(values));
    require_encode_contract(keys->view());
  }

  SECTION("zero columns are rejected")
  {
    REQUIRE_THROWS_AS(
      make_group_key_labels(
        cudf::table_view{}, cudf::get_default_stream(), cudf::get_current_device_resource_ref()),
      cudf::logic_error);
  }
}

TEST_CASE("group key labels preserve nullable tuple ordering", "[aggregate][group_key_labels]")
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(with_nulls(vector_to_cudf_column<gpu_type_traits<string_tag>>(
                                 std::vector<std::string>{"b", "ignored", "a", "ignored"}),
                               {true, false, true, false}));
  columns.push_back(
    with_nulls(vector_to_cudf_column<gpu_type_traits<int32_t>>(std::vector<int32_t>{1, 0, 2, 0}),
               {true, false, true, false}));
  auto keys = make_table(std::move(columns));

  require_encode_contract(keys->view());
  auto result = make_group_key_labels(
    keys->view(), cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const labels = copy_to_host<int32_t>(
    result.labels->view(), cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  REQUIRE(labels.values == std::vector<int32_t>{1, 2, 0, 2});
}

TEST_CASE("group key labels preserve floating-point key semantics", "[aggregate][group_key_labels]")
{
  double const quiet_nan   = std::numeric_limits<double>::quiet_NaN();
  double const payload_nan = std::nan("1");
  REQUIRE(std::bit_cast<std::uint64_t>(quiet_nan) != std::bit_cast<std::uint64_t>(payload_nan));

  std::vector<double> const values = {quiet_nan,
                                      payload_nan,
                                      -0.0,
                                      0.0,
                                      std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity(),
                                      1.5,
                                      99.0,
                                      quiet_nan};
  std::vector<bool> const valid    = {true, true, true, true, true, true, true, false, true};
  auto keys = make_table(with_nulls(vector_to_cudf_column<gpu_type_traits<double>>(values), valid));

  auto stream   = cudf::get_default_stream();
  auto mr       = cudf::get_current_device_resource_ref();
  auto expected = cudf::encode(keys->view(), stream, mr);
  auto actual   = make_group_key_labels(keys->view(), stream, mr);

  require_columns_equal(actual.labels->view(), expected.second->view(), stream, mr);
  require_floating_columns_equivalent(
    actual.sorted_unique_keys->view().column(0), expected.first->view().column(0), stream, mr);

  auto round_trip = cudf::gather(actual.sorted_unique_keys->view(),
                                 actual.labels->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
  require_floating_columns_equivalent(
    round_trip->view().column(0), keys->view().column(0), stream, mr);

  auto const labels = copy_to_host<int32_t>(actual.labels->view(), stream, mr).values;
  REQUIRE(labels[0] == labels[1]);
  REQUIRE(labels[0] == labels[8]);
  REQUIRE(labels[2] == labels[3]);
  REQUIRE(labels[0] != labels[7]);
}

TEST_CASE("group key labels match encode for deterministic randomized keys",
          "[aggregate][group_key_labels]")
{
  std::vector<std::string> string_a;
  std::vector<std::string> string_b;
  std::vector<int32_t> integers;
  std::vector<bool> string_valid;
  std::vector<bool> integer_valid;
  std::mt19937 random{20260817};
  for (int index = 0; index < 50000; ++index) {
    string_a.push_back("alpha-" + std::to_string(random() % 10));
    string_b.push_back("beta-" + std::to_string(random() % 5));
    integers.push_back(static_cast<int32_t>(random() % 4));
    string_valid.push_back(random() % 17 != 0);
    integer_valid.push_back(random() % 13 != 0);
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    with_nulls(vector_to_cudf_column<gpu_type_traits<string_tag>>(string_a), string_valid));
  columns.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(string_b));
  columns.push_back(
    with_nulls(vector_to_cudf_column<gpu_type_traits<int32_t>>(integers), integer_valid));
  auto keys = make_table(std::move(columns));

  require_encode_contract(keys->view());
}

TEST_CASE("group key labels use the supplied stream and memory resource",
          "[aggregate][group_key_labels]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* memory_space  = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(memory_space != nullptr);
  auto mr = get_resource_ref(*memory_space);
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>(
    std::vector<int32_t>{2, 1, 2, 3, 1}, stream.view(), mr));
  columns.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(
    std::vector<std::string>{"b", "a", "b", "c", "a"}, stream.view(), mr));
  auto keys = make_table(std::move(columns));

  auto expected = cudf::encode(keys->view(), stream.view(), mr);
  auto actual   = make_group_key_labels(keys->view(), stream.view(), mr);
  auto rebuilt  = cudf::gather(actual.sorted_unique_keys->view(),
                              actual.labels->view(),
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              stream.view(),
                              mr);

  require_tables_equal(
    actual.sorted_unique_keys->view(), expected.first->view(), stream.view(), mr);
  require_columns_equal(actual.labels->view(), expected.second->view(), stream.view(), mr);
  require_tables_equal(rebuilt->view(), keys->view(), stream.view(), mr);

  auto contents = actual.labels->release();
  REQUIRE(contents.data != nullptr);
  REQUIRE(contents.data->memory_resource() == mr);
}
