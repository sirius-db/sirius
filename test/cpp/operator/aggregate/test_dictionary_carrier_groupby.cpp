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

/**
 * @file test_dictionary_carrier_groupby.cpp
 * @brief A grouped aggregate over DICTIONARY32 carriers matches the one over decoded strings.
 *
 * A dictionary group key groups on its codes and leaves as STRING, the end of its carrier; a COUNT
 * over a dictionary value counts its codes.
 */

#include "data/data_batch_utils.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "operator/operator_test_utils.hpp"
#include "operator/operator_type_traits.hpp"
#include "utils/data_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {

using sirius::test::vector_to_cudf_column;
using sirius::test::operator_utils::gpu_type_traits;
using sirius::test::operator_utils::string_tag;
using Kind = cudf::aggregation::Kind;

constexpr int num_rows = 10'000;

std::unique_ptr<cudf::column> strings_with_nulls(int modulus, int null_every)
{
  std::vector<std::string> values;
  for (int row = 0; row < num_rows; ++row) {
    values.push_back("value_" + std::to_string(row % modulus));
  }
  auto column = vector_to_cudf_column<gpu_type_traits<string_tag>>(values);
  std::vector<cudf::bitmask_type> mask(
    cudf::bitmask_allocation_size_bytes(num_rows) / sizeof(cudf::bitmask_type), 0);
  for (int row = 0; row < num_rows; ++row) {
    if (row % null_every != 0) { mask[row / 32] |= cudf::bitmask_type{1} << (row % 32); }
  }
  auto device_mask = cudf::create_null_mask(num_rows, cudf::mask_state::UNINITIALIZED);
  REQUIRE(cudaMemcpy(device_mask.data(),
                     mask.data(),
                     mask.size() * sizeof(cudf::bitmask_type),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  column->set_null_mask(std::move(device_mask),
                        num_rows / null_every + (num_rows % null_every != 0 ? 1 : 0));
  return column;
}

std::unique_ptr<cudf::column> encode(cudf::column_view strings)
{
  return cudf::dictionary::encode(strings, cudf::data_type{cudf::type_id::INT32});
}

std::vector<std::optional<std::string>> strings_to_host(cudf::column_view column)
{
  auto const stream = cudf::get_default_stream();
  cudf::strings_column_view const strings(column);
  std::vector<int32_t> offsets(column.size() + 1);
  cudaMemcpy(offsets.data(),
             strings.offsets().data<int32_t>() + column.offset(),
             offsets.size() * sizeof(int32_t),
             cudaMemcpyDeviceToHost);
  std::string chars(static_cast<std::size_t>(strings.chars_size(stream)), '\0');
  cudaMemcpy(chars.data(), strings.chars_begin(stream), chars.size(), cudaMemcpyDeviceToHost);
  std::vector<cudf::bitmask_type> mask;
  if (column.nullable()) {
    mask.resize(cudf::num_bitmask_words(column.size()));
    cudaMemcpy(mask.data(),
               column.null_mask(),
               mask.size() * sizeof(cudf::bitmask_type),
               cudaMemcpyDeviceToHost);
  }
  std::vector<std::optional<std::string>> values;
  for (cudf::size_type row = 0; row < column.size(); ++row) {
    if (!mask.empty() && !((mask[row / 32] >> (row % 32)) & 1u)) {
      values.emplace_back();
      continue;
    }
    values.emplace_back(chars.substr(offsets[row], offsets[row + 1] - offsets[row]));
  }
  return values;
}

template <typename T>
std::vector<T> to_host(cudf::column_view const& column)
{
  std::vector<T> values(column.size());
  cudaMemcpy(values.data(), column.data<T>(), values.size() * sizeof(T), cudaMemcpyDeviceToHost);
  return values;
}

using group_row = std::tuple<int64_t, int64_t, int64_t>;  // SUM(v), COUNT(s), COUNT(*)

// Run SUM(v), COUNT(s), COUNT(*) grouped by k over [k, v, s], and return each group's results
// keyed by its key, with the type the key column left as.
std::pair<std::map<std::optional<std::string>, group_row>, cudf::type_id> grouped(
  cudf::column_view key, cudf::column_view value, cudf::column_view counted)
{
  auto* space       = sirius::test::operator_utils::get_default_gpu_space();
  auto const stream = cudf::get_default_stream();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::make_unique<cudf::column>(key));
  columns.push_back(std::make_unique<cudf::column>(value));
  columns.push_back(std::make_unique<cudf::column>(counted));
  auto batch  = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                       *space,
                                       stream,
                                       sirius::telemetry::batch_telemetry_info{});
  auto output = sirius::op::gpu_aggregate_impl::local_grouped_aggregate(
    batch->to_read_only(),
    {0},
    {Kind::SUM, Kind::COUNT_VALID, Kind::COUNT_ALL},
    {1, 2, 0},
    {},
    {},
    {},
    stream,
    *space,
    {},
    {},
    {},
    false,
    nullptr,
    true);
  auto const view = sirius::get_cudf_table_view(*output);
  auto const type = view.column(0).type().id();
  if (type != cudf::type_id::STRING || view.num_rows() == 0) { return {{}, type}; }
  auto const names  = strings_to_host(view.column(0));
  auto const sums   = to_host<int64_t>(view.column(1));
  auto const counts = to_host<int64_t>(view.column(2));
  auto const all    = to_host<int64_t>(view.column(3));
  std::map<std::optional<std::string>, group_row> groups;
  for (std::size_t row = 0; row < names.size(); ++row) {
    REQUIRE(groups.emplace(names[row], group_row{sums[row], counts[row], all[row]}).second);
  }
  return {std::move(groups), type};
}

}  // namespace

TEST_CASE("grouped aggregate over dictionary carriers matches decoded strings",
          "[aggregate][dictionary_carrier]")
{
  auto const key     = strings_with_nulls(7, 50);
  auto const counted = strings_with_nulls(13, 3);
  std::vector<int64_t> values(num_rows);
  for (int row = 0; row < num_rows; ++row) {
    values[row] = row % 101 - 50;
  }
  auto const value                  = vector_to_cudf_column<gpu_type_traits<int64_t>>(values);
  auto const [expected, plain_type] = grouped(key->view(), value->view(), counted->view());
  REQUIRE(plain_type == cudf::type_id::STRING);
  REQUIRE(expected.size() == 8);  // seven values and the NULL group

  auto const dictionary_key     = encode(key->view());
  auto const dictionary_counted = encode(counted->view());
  SECTION("a dictionary key leaves as strings")
  {
    auto const [actual, type] =
      grouped(dictionary_key->view(), value->view(), dictionary_counted->view());
    CHECK(type == cudf::type_id::STRING);
    CHECK(actual == expected);
  }
  SECTION("a string key with a counted dictionary value")
  {
    auto const [actual, type] = grouped(key->view(), value->view(), dictionary_counted->view());
    CHECK(type == cudf::type_id::STRING);
    CHECK(actual == expected);
  }
}

TEST_CASE("a null-free dictionary key of few values leaves the low-cardinality path as strings",
          "[aggregate][dictionary_carrier]")
{
  std::vector<std::string> names;
  for (int row = 0; row < num_rows; ++row) {
    names.push_back("value_" + std::to_string(row % 7));
  }
  auto const key = vector_to_cudf_column<gpu_type_traits<string_tag>>(names);
  std::vector<int64_t> values(num_rows, 1);
  auto const value          = vector_to_cudf_column<gpu_type_traits<int64_t>>(values);
  auto const [expected, _]  = grouped(key->view(), value->view(), key->view());
  auto const dictionary_key = encode(key->view());
  auto const [actual, type] = grouped(dictionary_key->view(), value->view(), key->view());
  CHECK(type == cudf::type_id::STRING);
  CHECK(actual == expected);
}

TEST_CASE("a zero-row dictionary key groups to an empty string key",
          "[aggregate][dictionary_carrier]")
{
  auto const value   = cudf::make_empty_column(cudf::type_id::INT64);
  auto const counted = cudf::make_empty_column(cudf::type_id::STRING);
  SECTION("without children")
  {
    auto const key            = cudf::make_empty_column(cudf::type_id::DICTIONARY32);
    auto const [groups, type] = grouped(key->view(), value->view(), counted->view());
    CHECK(type == cudf::type_id::STRING);
    CHECK(groups.empty());
  }
  SECTION("with children")
  {
    auto const key            = encode(cudf::make_empty_column(cudf::type_id::STRING)->view());
    auto const [groups, type] = grouped(key->view(), value->view(), counted->view());
    CHECK(type == cudf::type_id::STRING);
    CHECK(groups.empty());
  }
}
