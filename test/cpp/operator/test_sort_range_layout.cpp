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

#include "operator_test_utils.hpp"
#include "sort_range_test_utils.hpp"

#include <catch.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>

using namespace sirius::test::operator_utils;
using namespace sirius::test::sort_range;

namespace {

duckdb::BoundOrderByNode order(duckdb::idx_t column,
                               duckdb::LogicalType type      = duckdb::LogicalType::INTEGER,
                               duckdb::OrderType direction   = duckdb::OrderType::ASCENDING,
                               duckdb::OrderByNullType nulls = duckdb::OrderByNullType::NULLS_LAST)
{
  return duckdb::BoundOrderByNode(
    direction, nulls, duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, column));
}

duckdb::vector<duckdb::BoundOrderByNode> one_order(
  duckdb::LogicalType type      = duckdb::LogicalType::INTEGER,
  duckdb::OrderType direction   = duckdb::OrderType::ASCENDING,
  duckdb::OrderByNullType nulls = duckdb::OrderByNullType::NULLS_LAST)
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(order(0, type, direction, nulls));
  return orders;
}

std::vector<std::shared_ptr<cucascade::data_batch>> int_batches(
  std::vector<std::vector<std::int32_t>> const& values)
{
  auto* space = get_default_gpu_space();
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (auto const& batch : values) {
    batches.push_back(make_numeric_batch<std::int32_t>(*space, batch, cudf::type_id::INT32));
  }
  return batches;
}

bool precedes(key_row const& a,
              key_row const& b,
              duckdb::vector<duckdb::BoundOrderByNode> const& orders)
{
  for (std::size_t i = 0; i < orders.size(); ++i) {
    auto const& x = a[i];
    auto const& y = b[i];
    if (x == y) { continue; }
    bool const x_null = std::holds_alternative<std::monostate>(x);
    bool const y_null = std::holds_alternative<std::monostate>(y);
    if (x_null || y_null) {
      return x_null == (orders[i].null_order == duckdb::OrderByNullType::NULLS_FIRST);
    }
    bool const ascending = orders[i].type == duckdb::OrderType::ASCENDING;
    return ascending ? x < y : y < x;
  }
  return false;
}

void check_layout(layout const& result,
                  std::vector<std::shared_ptr<cucascade::data_batch>> const& batches,
                  duckdb::vector<duckdb::BoundOrderByNode> const& orders)
{
  CHECK(result.input_batches == batches.size());
  REQUIRE(result.counted.size() == result.published.size());
  REQUIRE(result.keys.size() == result.counted.size());
  std::size_t total_rows = 0;
  std::vector<key_row> output;
  for (std::size_t range = 0; range < result.counted.size(); ++range) {
    auto const& counted = result.counted[range];
    auto const& actual  = result.published[range];
    CHECK(counted.rows == actual.rows);
    CHECK(counted.bytes == actual.bytes);
    CHECK(counted.slices == actual.slices);
    total_rows += actual.rows;
    output.insert(output.end(), result.keys[range].begin(), result.keys[range].end());
  }
  std::vector<std::size_t> columns;
  for (auto const& node : orders) {
    columns.push_back(node.expression->Cast<duckdb::BoundReferenceExpression>().Index());
  }
  std::vector<key_row> input;
  for (auto const& batch : batches) {
    auto keys = read_keys(*batch, columns);
    input.insert(input.end(), keys.begin(), keys.end());
  }
  CHECK(total_rows == input.size());
  REQUIRE(output.size() == input.size());
  for (std::size_t i = 1; i < output.size(); ++i) {
    CHECK_FALSE(precedes(output[i], output[i - 1], orders));
  }
  std::map<key_row, std::size_t> in_counts, out_counts;
  for (auto const& row : input) {
    ++in_counts[row];
  }
  for (auto const& row : output) {
    ++out_counts[row];
  }
  CHECK(in_counts == out_counts);
}

layout run_checked(std::vector<std::shared_ptr<cucascade::data_batch>> const& batches,
                   duckdb::vector<duckdb::BoundOrderByNode> const& orders,
                   std::uint64_t sample_bytes,
                   std::uint64_t target_bytes,
                   bool finished                   = true,
                   std::size_t max_sample_key_rows = 0)
{
  std::size_t cardinality = 0;
  for (auto const& batch : batches) {
    cardinality += sirius::get_cudf_table_view(*batch).num_rows();
  }
  auto result = run_layout(
    batches, orders, sample_bytes, target_bytes, cardinality, finished, max_sample_key_rows);
  check_layout(result, batches, orders);
  return result;
}

}  // namespace

TEST_CASE("Sort ranges preserve uniform random keys", "[sort_range_layout]")
{
  std::mt19937 rng(2579);
  std::vector<std::vector<std::int32_t>> values(4, std::vector<std::int32_t>(250));
  for (auto& batch : values) {
    for (auto& value : batch) {
      value = static_cast<std::int32_t>(rng() % 10000);
    }
    std::sort(batch.begin(), batch.end());
  }
  auto batches = int_batches(values);
  auto result  = run_checked(batches, one_order(), 100000, 80);
  CHECK(result.sample.num_ranges > 4);
}

TEST_CASE("Sort ranges count the single-range path", "[sort_range_layout]")
{
  auto batches = int_batches({{1, 2, 3}, {4, 5, 6}});
  auto result  = run_checked(batches, one_order(), 100000, 100000);
  REQUIRE(result.counted.size() == 1);
  CHECK(result.counted.front().rows == 6);
  CHECK(result.counted.front().slices == 2);
}

TEST_CASE("Sort ranges keep a heavy key together (current behaviour, #2579)", "[sort_range_layout]")
{
  std::vector<std::vector<std::int32_t>> values(4);
  for (auto& batch : values) {
    for (int i = 0; i < 150; ++i) {
      batch.push_back(i);
    }
    batch.insert(batch.end(), 100, 75);
    std::sort(batch.begin(), batch.end());
  }
  auto batches = int_batches(values);
  auto result  = run_checked(batches, one_order(), 100000, 80);
  REQUIRE(result.sample.repeated_boundaries.has_value());
  CHECK(*result.sample.repeated_boundaries > 0);
  auto empty = std::count_if(
    result.counted.begin(), result.counted.end(), [](auto const& r) { return r.rows == 0; });
  CHECK(empty >= *result.sample.repeated_boundaries);
  auto ranges_with_key =
    std::count_if(result.keys.begin(), result.keys.end(), [](auto const& rows) {
      return std::any_of(rows.begin(), rows.end(), [](auto const& row) {
        return row[0] == key_cell{std::int64_t{75}};
      });
    });
  CHECK(ranges_with_key == 1);
}

TEST_CASE("Sort ranges with all equal keys use one range (current behaviour, #2579)",
          "[sort_range_layout]")
{
  auto batches =
    int_batches(std::vector<std::vector<std::int32_t>>(4, std::vector<std::int32_t>(250, 7)));
  auto result = run_checked(batches, one_order(), 100000, 80);
  CHECK(std::count_if(result.counted.begin(), result.counted.end(), [](auto const& r) {
          return r.rows > 0;
        }) == 1);
}

TEST_CASE("Sort ranges preserve NULL-heavy orders", "[sort_range_layout]")
{
  auto* space = get_default_gpu_space();
  for (auto direction : {duckdb::OrderType::ASCENDING, duckdb::OrderType::DESCENDING}) {
    for (auto nulls : {duckdb::OrderByNullType::NULLS_FIRST, duckdb::OrderByNullType::NULLS_LAST}) {
      std::vector<std::int32_t> values(200);
      std::iota(values.begin(), values.end(), 0);
      if (direction == duckdb::OrderType::DESCENDING) {
        std::reverse(values.begin(), values.end());
      }
      std::vector<bool> valid(200, true);
      if (nulls == duckdb::OrderByNullType::NULLS_FIRST) {
        std::fill(valid.begin(), valid.begin() + 100, false);
      } else {
        std::fill(valid.begin() + 100, valid.end(), false);
      }
      auto batch =
        make_numeric_batch_with_nulls<std::int32_t>(*space, values, valid, cudf::type_id::INT32);
      auto orders = one_order(duckdb::LogicalType::INTEGER, direction, nulls);
      run_checked({batch}, orders, 100000, 80);
    }
  }
}

TEST_CASE("Sort ranges preserve two keys", "[sort_range_layout]")
{
  auto* space = get_default_gpu_space();
  std::vector<std::int64_t> first(300, 5), second(300);
  std::iota(second.begin(), second.end(), 0);
  auto batch =
    make_two_column_batch<std::int64_t, std::int64_t>(*space, first, second, cudf::type_id::INT64);
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(order(0, duckdb::LogicalType::BIGINT));
  orders.push_back(order(1, duckdb::LogicalType::BIGINT));
  run_checked({batch}, orders, 100000, 80);
}

TEST_CASE("Sort ranges preserve string keys", "[sort_range_layout]")
{
  auto* space = get_default_gpu_space();
  std::vector<std::string> values;
  for (int i = 0; i < 300; ++i) {
    values.push_back("key-" + std::to_string(1000 + i));
  }
  auto batch = make_string_batch(*space, values);
  run_checked({batch}, one_order(duckdb::LogicalType::VARCHAR), 100000, 120);
}

TEST_CASE("Sort ranges count wide-row bytes", "[sort_range_layout]")
{
  auto* space = get_default_gpu_space();
  std::vector<std::int64_t> keys(300);
  std::iota(keys.begin(), keys.end(), 0);
  std::vector<std::string> payload;
  for (auto key : keys) {
    payload.emplace_back(10 + key, 'x');
  }
  auto batch =
    make_two_column_batch<std::int64_t, std::string>(*space, keys, payload, cudf::type_id::STRING);
  auto result      = run_checked({batch}, one_order(duckdb::LogicalType::BIGINT), 1000000, 1000);
  auto minmax_rows = std::minmax_element(
    result.counted.begin(), result.counted.end(), [](auto const& a, auto const& b) {
      return a.rows < b.rows;
    });
  auto minmax_bytes = std::minmax_element(
    result.counted.begin(), result.counted.end(), [](auto const& a, auto const& b) {
      return a.bytes < b.bytes;
    });
  INFO("wide rows: rows min=" << minmax_rows.first->rows << " max=" << minmax_rows.second->rows
                              << ", bytes min=" << minmax_bytes.first->bytes
                              << " max=" << minmax_bytes.second->bytes);
  CHECK(minmax_bytes.second->bytes > minmax_bytes.first->bytes);
  CHECK(result.sample.complete_input);
}

TEST_CASE("Sort ranges cluster later batches in last range when chosen after one batch",
          "[sort_range_layout]")
{
  std::vector<std::vector<std::int32_t>> values(4, std::vector<std::int32_t>(100));
  for (int b = 0; b < 4; ++b) {
    std::iota(values[b].begin(), values[b].end(), b * 100);
  }
  auto batches = int_batches(values);
  auto bytes   = batches.front()->to_read_only().get_data()->get_size_in_bytes();
  auto result  = run_checked(batches, one_order(), bytes, 50, false);
  CHECK(result.sample.sample_batches == 1);
  CHECK(result.counted.back().rows >= 300);
  for (std::size_t range = 0; range + 1 < result.keys.size(); ++range) {
    for (auto const& row : result.keys[range]) {
      CHECK(std::get<std::int64_t>(row[0]) < 100);
    }
  }
}

TEST_CASE("Sort ranges cluster reversed arrival in first range when chosen after one batch",
          "[sort_range_layout]")
{
  std::vector<std::vector<std::int32_t>> values(4, std::vector<std::int32_t>(100));
  for (int b = 0; b < 4; ++b) {
    std::iota(values[b].begin(), values[b].end(), b * 100);
  }
  std::reverse(values.begin(), values.end());
  auto batches = int_batches(values);
  auto bytes   = batches.front()->to_read_only().get_data()->get_size_in_bytes();
  auto result  = run_checked(batches, one_order(), bytes, 50, false);
  CHECK(result.sample.sample_batches == 1);
  CHECK(result.counted.front().rows >= 300);
  for (std::size_t range = 1; range < result.keys.size(); ++range) {
    for (auto const& row : result.keys[range]) {
      CHECK(std::get<std::int64_t>(row[0]) >= 300);
    }
  }
}

TEST_CASE("Sort ranges balance batches that arrive in key order when chosen after all input",
          "[sort_range_layout]")
{
  std::vector<std::vector<std::int32_t>> values(4, std::vector<std::int32_t>(100));
  for (int b = 0; b < 4; ++b) {
    std::iota(values[b].begin(), values[b].end(), b * 100);
  }
  bool const reversed = GENERATE(false, true);
  if (reversed) { std::reverse(values.begin(), values.end()); }
  auto result = run_checked(int_batches(values), one_order(), 100000, 50);
  CHECK(result.sample.complete_input);
  CHECK(result.sample.sample_batches == 4);
  REQUIRE(result.sample.num_ranges == 32);
  for (auto const& range : result.counted) {
    // 400 distinct keys in 32 ranges.
    CHECK(range.rows >= 12);
    CHECK(range.rows <= 13);
  }
}

TEST_CASE("Sort ranges balance key-row samples with a stride above one", "[sort_range_layout]")
{
  std::mt19937 rng(2579);
  std::uniform_int_distribution<std::int32_t> key(0, 1000000);
  std::vector<std::vector<std::int32_t>> values(3, std::vector<std::int32_t>(20000));
  for (auto& batch : values) {
    for (auto& value : batch) {
      value = key(rng);
    }
    std::sort(batch.begin(), batch.end());
  }
  auto result = run_checked(int_batches(values), one_order(), 1000000, 30000, true, 4096);
  REQUIRE(result.sample.sample_stride > 1);
  REQUIRE(result.sample.num_ranges == 8);
  for (auto const& range : result.counted) {
    CHECK(std::abs(static_cast<double>(range.rows) / 7500.0 - 1.0) <= 0.05);
  }
}

TEST_CASE("Sort ranges weight unequal batches by their row counts", "[sort_range_layout]")
{
  std::mt19937 rng(2580);
  std::uniform_int_distribution<std::int32_t> wide(0, 1000000);
  std::uniform_int_distribution<std::int32_t> narrow(450000, 550000);
  std::vector<std::vector<std::int32_t>> values(2);
  values[0].resize(40000);
  values[1].resize(4000);
  for (auto& value : values[0]) {
    value = wide(rng);
  }
  for (auto& value : values[1]) {
    value = narrow(rng);
  }
  for (auto& batch : values) {
    std::sort(batch.begin(), batch.end());
  }
  auto result = run_checked(int_batches(values), one_order(), 1000000, 22000, true, 4096);
  REQUIRE(result.sample.sample_stride > 1);
  REQUIRE(result.sample.num_ranges == 8);
  for (auto const& range : result.counted) {
    CHECK(std::abs(static_cast<double>(range.rows) / 5500.0 - 1.0) <= 0.05);
  }
}

TEST_CASE("Sort ranges are no more than the sampled key rows", "[sort_range_layout]")
{
  std::vector<std::int32_t> keys(4096);
  std::iota(keys.begin(), keys.end(), 0);
  // The byte target asks for many more ranges than the 16 key rows kept.
  auto result = run_checked(int_batches({keys}), one_order(), 1000000, 64, true, 16);
  REQUIRE(result.sample.sample_key_rows == 16);
  CHECK(result.sample.num_ranges == 16);
  REQUIRE(result.sample.repeated_boundaries.has_value());
  CHECK(*result.sample.repeated_boundaries == 0);
  REQUIRE(result.counted.size() == 16);
  for (auto const& range : result.counted) {
    CHECK(range.rows > 0);
  }
}

TEST_CASE("Sort ranges retain rows in a batch too short to give a key row", "[sort_range_layout]")
{
  std::vector<std::int32_t> large(40000);
  std::iota(large.begin(), large.end(), 0);
  auto result =
    run_checked(int_batches({large, {10, 11, 12}}), one_order(), 1000000, 20000, true, 4096);
  // The first key row of a batch is past its third row at this stride.
  REQUIRE(result.sample.sample_stride > 6);
  CHECK(result.sample.sample_key_rows < result.sample.sample_rows);
}
