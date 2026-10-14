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

#include "op/aggregate/low_cardinality_groupby.hpp"
#include "operator/operator_type_traits.hpp"
#include "utils/data_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using sirius::test::vector_to_cudf_column;
using sirius::test::operator_utils::decimal64_tag;
using sirius::test::operator_utils::gpu_type_traits;
using sirius::test::operator_utils::string_tag;
using Kind = cudf::aggregation::Kind;

/// Host copy of a column's fixed-width values.
template <typename T>
std::vector<T> to_host(cudf::column_view const& column)
{
  std::vector<T> values(column.size());
  cudaMemcpy(values.data(), column.data<T>(), values.size() * sizeof(T), cudaMemcpyDeviceToHost);
  return values;
}

/// Host copy of a strings column whose values are all one character long.
std::string single_chars_to_host(cudf::column_view const& column)
{
  cudf::strings_column_view const strings(column);
  auto const stream = cudf::get_default_stream();
  std::string chars(static_cast<std::size_t>(strings.chars_size(stream)), '\0');
  cudaMemcpy(chars.data(), strings.chars_begin(stream), chars.size(), cudaMemcpyDeviceToHost);
  return chars;
}

struct batch {
  std::unique_ptr<cudf::table>
    table;  ///< [key0 STRING, key1 STRING, value0 DECIMAL64, value1 DECIMAL64]
  std::vector<std::string> key0;
  std::vector<std::string> key1;
  std::vector<int64_t> value0;
  std::vector<int64_t> value1;
};

/// Random rows whose keys are drawn from @p key0_alphabet x @p key1_alphabet.
batch make_batch(int num_rows, std::string const& key0_alphabet, std::string const& key1_alphabet)
{
  std::mt19937 rng(42);
  std::uniform_int_distribution<std::size_t> pick0(0, key0_alphabet.size() - 1);
  std::uniform_int_distribution<std::size_t> pick1(0, key1_alphabet.size() - 1);
  std::uniform_int_distribution<int64_t> value(-1'000'000, 1'000'000);
  batch out;
  for (int row = 0; row < num_rows; ++row) {
    out.key0.emplace_back(1, key0_alphabet[pick0(rng)]);
    out.key1.emplace_back(1, key1_alphabet[pick1(rng)]);
    out.value0.push_back(value(rng));
    out.value1.push_back(value(rng));
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(out.key0));
  columns.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>(out.key1));
  columns.push_back(vector_to_cudf_column<gpu_type_traits<decimal64_tag>>(out.value0));
  columns.push_back(vector_to_cudf_column<gpu_type_traits<decimal64_tag>>(out.value1));
  out.table = std::make_unique<cudf::table>(std::move(columns));
  return out;
}

std::unique_ptr<cudf::table> run(cudf::table_view const& input,
                                 std::vector<int> const& group_idx,
                                 std::vector<Kind> const& aggregates,
                                 std::vector<int> const& aggregate_idx)
{
  return sirius::op::try_low_cardinality_groupby(input,
                                                 group_idx,
                                                 aggregates,
                                                 aggregate_idx,
                                                 cudf::get_default_stream(),
                                                 cudf::get_current_device_resource_ref());
}

/// Runs SUM(value0), SUM(value1), COUNT_VALID(value0), COUNT_ALL(key0) grouped by both keys and
/// checks every group against a host computation.
void check_against_host(batch const& input)
{
  auto result = run(input.table->view(),
                    {0, 1},
                    {Kind::SUM, Kind::SUM, Kind::COUNT_VALID, Kind::COUNT_ALL},
                    {2, 3, 2, 0});
  REQUIRE(result != nullptr);
  REQUIRE(result->num_columns() == 6);
  CHECK(result->get_column(2).type() == cudf::data_type{cudf::type_id::DECIMAL128, -2});
  CHECK(result->get_column(4).type() == cudf::data_type{cudf::type_id::INT64});

  struct totals {
    __int128_t sum0 = 0;
    __int128_t sum1 = 0;
    int64_t count   = 0;
  };
  std::map<std::string, totals> expected;
  for (std::size_t row = 0; row < input.key0.size(); ++row) {
    auto& group = expected[input.key0[row] + input.key1[row]];
    group.sum0 += input.value0[row];
    group.sum1 += input.value1[row];
    ++group.count;
  }

  auto sorted      = cudf::sort_by_key(result->view(), result->view().select({0, 1}));
  auto const keys0 = single_chars_to_host(sorted->get_column(0));
  auto const keys1 = single_chars_to_host(sorted->get_column(1));
  auto const sums0 = to_host<__int128_t>(sorted->get_column(2));
  auto const sums1 = to_host<__int128_t>(sorted->get_column(3));
  auto const valid = to_host<int64_t>(sorted->get_column(4));
  auto const all   = to_host<int64_t>(sorted->get_column(5));
  REQUIRE(sorted->num_rows() == static_cast<cudf::size_type>(expected.size()));
  std::size_t group_idx = 0;
  for (auto const& [key, group] : expected) {
    INFO("group " << key);
    CHECK(std::string{keys0[group_idx], keys1[group_idx]} == key);
    CHECK(sums0[group_idx] == group.sum0);
    CHECK(sums1[group_idx] == group.sum1);
    CHECK(valid[group_idx] == group.count);
    CHECK(all[group_idx] == group.count);
    ++group_idx;
  }
}

}  // namespace

TEST_CASE("low-cardinality group-by matches a host computation", "[aggregate][low_cardinality]")
{
  SECTION("4 groups (register kernel, up to 4 groups)")
  {
    check_against_host(make_batch(200'000, "AN", "FP"));
  }
  SECTION("6 groups (register kernel, up to 8 groups)")
  {
    check_against_host(make_batch(200'000, "ANR", "FP"));
  }
  SECTION("21 groups (shared-memory fallback)")
  {
    check_against_host(make_batch(200'000, "ANR", "ABCDEFG"));
  }
}

TEST_CASE("low-cardinality group-by declines inputs it does not handle",
          "[aggregate][low_cardinality]")
{
  SECTION("more than 64 groups")
  {
    auto input = make_batch(50'000, "ABCDEFGHI", "ABCDEFGH");  // 72 groups
    CHECK(run(input.table->view(), {0, 1}, {Kind::SUM}, {2}) == nullptr);
  }
  SECTION("an unsupported aggregate")
  {
    auto input = make_batch(1'000, "AN", "FP");
    CHECK(run(input.table->view(), {0, 1}, {Kind::MIN}, {2}) == nullptr);
  }
  SECTION("a key that is not one character long")
  {
    auto input   = make_batch(1'000, "AN", "FP");
    auto key0    = input.key0;
    key0[500]    = "AB";
    auto columns = input.table->release();
    columns[0]   = vector_to_cudf_column<gpu_type_traits<string_tag>>(key0);
    cudf::table const table(std::move(columns));
    CHECK(run(table.view(), {0, 1}, {Kind::SUM}, {2}) == nullptr);
  }
  SECTION("a non-string key")
  {
    auto input = make_batch(1'000, "AN", "FP");
    CHECK(run(input.table->view(), {2}, {Kind::SUM}, {3}) == nullptr);
  }
}
