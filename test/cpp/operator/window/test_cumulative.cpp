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

#include "helper/type_conversions.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/window/cumulative.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>

namespace {
using sirius::op::cumulative_aggregate;
using sirius::op::evaluate_cumulative_window;
using sirius::op::ranking_order;
using function = sirius::aggregate_id;

void set_validity(cudf::column& column,
                  std::vector<bool> const& valid,
                  ::cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
{
  if (valid.empty()) { return; }
  REQUIRE(valid.size() == static_cast<std::size_t>(column.size()));
  auto mask = cudf::create_null_mask(column.size(), cudf::mask_state::ALL_VALID, stream, mr);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < column.size(); ++i) {
    if (!valid[i]) {
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), i, i + 1, false, stream);
      ++nulls;
    }
  }
  column.set_null_mask(std::move(mask), nulls);
}

// A device column of @p values, with the validity given by @p valid when it is not empty.
template <typename T>
std::unique_ptr<cudf::column> column(std::vector<T> const& values,
                                     cudf::data_type type,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr,
                                     std::vector<bool> const& valid = {})
{
  auto result =
    cudf::is_fixed_point(type) ? cudf::make_fixed_point_column(
                                   type, values.size(), cudf::mask_state::UNALLOCATED, stream, mr)
    : cudf::is_timestamp(type)
      ? cudf::make_timestamp_column(type, values.size(), cudf::mask_state::UNALLOCATED, stream, mr)
      : cudf::make_numeric_column(type, values.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(result->mutable_view().template data<T>(),
                            values.data(),
                            values.size() * sizeof(T),
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
  }
  set_validity(*result, valid, stream, mr);
  // Input vectors can be temporaries. Finish their host-to-device transfers before returning.
  stream.sync();
  return result;
}

template <typename T>
std::vector<T> host(cudf::column_view input, ::cuda::stream_ref stream)
{
  std::vector<T> values(input.size());
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(values.data(),
                            input.data<T>(),
                            values.size() * sizeof(T),
                            cudaMemcpyDeviceToHost,
                            stream.get()) == cudaSuccess);
  }
  stream.sync();
  return values;
}

std::vector<bool> validity(cudf::column_view input, ::cuda::stream_ref stream)
{
  std::vector<bool> result(input.size(), true);
  if (!input.nullable()) { return result; }
  std::vector<cudf::bitmask_type> mask((input.size() + input.offset() + 31) / 32);
  REQUIRE(cudaMemcpyAsync(mask.data(),
                          input.null_mask(),
                          mask.size() * sizeof(cudf::bitmask_type),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  for (cudf::size_type i = 0; i < input.size(); ++i) {
    auto bit  = input.offset() + i;
    result[i] = (mask[bit / 32] & (1u << (bit % 32))) != 0;
  }
  return result;
}

struct direction {
  char const* sql;
  duckdb::OrderType type;
  duckdb::OrderByNullType nulls;
};

constexpr direction ascending_nulls_first{
  "ASC NULLS FIRST", duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_FIRST};
constexpr direction ascending_nulls_last{
  "ASC NULLS LAST", duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST};
constexpr direction descending_nulls_first{
  "DESC NULLS FIRST", duckdb::OrderType::DESCENDING, duckdb::OrderByNullType::NULLS_FIRST};
constexpr direction descending_nulls_last{
  "DESC NULLS LAST", duckdb::OrderType::DESCENDING, duckdb::OrderByNullType::NULLS_LAST};

// An ORDER BY over @p columns, every key in the same direction.
ranking_order make_order(std::vector<cudf::size_type> columns, direction const& key)
{
  ranking_order order;
  order.columns = std::move(columns);
  for (std::size_t i = 0; i < order.columns.size(); ++i) {
    order.directions.push_back(sirius::op::to_cudf_order(key.type));
    order.nulls.push_back(sirius::op::to_cudf_null_order(key.type, key.nulls));
  }
  return order;
}

// One input row of the model. Columns of the input table: k (partition key), o (order key), v.
struct row {
  std::optional<int32_t> k;
  std::optional<int32_t> o;
  std::optional<int32_t> v;
};

// Rows with NULL partition keys, tied and NULL order keys and NULL arguments. The seed is fixed.
std::vector<row> make_rows(std::size_t count)
{
  std::mt19937 generator(12345);
  auto const pick = [&](int low, int high, int null_percent) -> std::optional<int32_t> {
    if (static_cast<int>(generator() % 100) < null_percent) { return std::nullopt; }
    return low + static_cast<int32_t>(generator() % static_cast<uint32_t>(high - low + 1));
  };
  std::vector<row> rows(count);
  for (auto& r : rows) {
    r = {pick(0, 4, 10), pick(0, 9, 15), pick(-50, 50, 25)};
  }
  return rows;
}

// What a running aggregate holds after some rows of a partition.
struct running {
  std::optional<__int128_t> sum;
  std::optional<__int128_t> min;
  std::optional<__int128_t> max;
  int64_t count      = 0;
  int64_t count_star = 0;
};

// The result of each aggregate for each row, by the definition of the frame ROWS BETWEEN UNBOUNDED
// PRECEDING AND CURRENT ROW. Rows are ordered by the partition key and then the order key in
// the SQL sense, and ties keep their input order.
std::vector<running> model(std::vector<row> const& rows, bool keyed, direction const& dir)
{
  auto const ascending  = dir.type == duckdb::OrderType::ASCENDING;
  auto const null_first = dir.nulls == duckdb::OrderByNullType::NULLS_FIRST;
  auto const order_key  = [&](row const& r) -> int64_t {
    if (!r.o) {
      return null_first ? std::numeric_limits<int64_t>::min() : std::numeric_limits<int64_t>::max();
    }
    return ascending ? *r.o : -static_cast<int64_t>(*r.o);
  };
  auto const partition_key = [&](row const& r) -> int64_t { return keyed ? (r.k ? *r.k : -1) : 0; };
  std::vector<std::size_t> index(rows.size());
  std::iota(index.begin(), index.end(), 0);
  std::stable_sort(index.begin(), index.end(), [&](std::size_t a, std::size_t b) {
    auto const pa = partition_key(rows[a]);
    auto const pb = partition_key(rows[b]);
    return pa != pb ? pa < pb : order_key(rows[a]) < order_key(rows[b]);
  });
  std::vector<running> result(rows.size());
  running state;
  std::optional<int64_t> current;
  for (auto const i : index) {
    if (current != partition_key(rows[i])) {
      current = partition_key(rows[i]);
      state   = {};
    }
    ++state.count_star;
    if (rows[i].v) {
      auto const v = static_cast<__int128_t>(*rows[i].v);
      state.sum    = state.sum.value_or(0) + v;
      state.min    = state.min ? std::min(*state.min, v) : v;
      state.max    = state.max ? std::max(*state.max, v) : v;
      ++state.count;
    }
    result[i] = state;
  }
  return result;
}

// Checks every function over an argument column of @p type against the model, for the rows in
// @p rows. The input has the columns k, o and v, in that order.
template <typename T>
void check_against_model(cudf::data_type type,
                         bool keyed,
                         direction const& dir,
                         bool has_sum,
                         ::cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
{
  INFO("type " << static_cast<int>(type.id()) << " keyed " << keyed << " " << dir.sql);
  auto const rows = make_rows(200);
  std::vector<int32_t> k, o;
  std::vector<T> v;
  std::vector<bool> k_valid, o_valid, v_valid;
  for (auto const& r : rows) {
    k.push_back(r.k.value_or(0));
    o.push_back(r.o.value_or(0));
    v.push_back(static_cast<T>(r.v.value_or(0)));
    k_valid.push_back(r.k.has_value());
    o_valid.push_back(r.o.has_value());
    v_valid.push_back(r.v.has_value());
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column<int32_t>(k, cudf::data_type{cudf::type_id::INT32}, stream, mr, k_valid));
  columns.push_back(column<int32_t>(o, cudf::data_type{cudf::type_id::INT32}, stream, mr, o_valid));
  columns.push_back(column<T>(v, type, stream, mr, v_valid));
  cudf::table input(std::move(columns));

  std::vector<cumulative_aggregate> aggregates;
  if (has_sum) { aggregates.push_back({function::sum, 2}); }
  aggregates.push_back({function::min, 2});
  aggregates.push_back({function::max, 2});
  aggregates.push_back({function::count, 2});
  aggregates.push_back({function::count_star, 2});
  auto result = evaluate_cumulative_window(
    input.view(),
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
    make_order({1}, dir),
    aggregates,
    stream,
    mr);
  REQUIRE(result->num_columns() == static_cast<cudf::size_type>(aggregates.size()));

  auto const expected = model(rows, keyed, dir);
  int column_index    = 0;
  auto const next = [&]() -> cudf::column_view { return result->view().column(column_index++); };
  auto const check_extremum = [&](bool is_min) {
    auto const view = next();
    CHECK(view.type() == type);
    auto const values = host<T>(view, stream);
    auto const valid  = validity(view, stream);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      auto const& want = is_min ? expected[i].min : expected[i].max;
      INFO("row " << i);
      REQUIRE(valid[i] == want.has_value());
      if (want) { CHECK(static_cast<__int128_t>(values[i]) == *want); }
    }
  };
  if (has_sum) {
    auto const view = next();
    CHECK(view.type() == (cudf::is_fixed_point(type)
                            ? cudf::data_type{cudf::type_id::DECIMAL128, type.scale()}
                            : cudf::data_type{cudf::type_id::DECIMAL128, 0}));
    auto const values = host<__int128_t>(view, stream);
    auto const valid  = validity(view, stream);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      INFO("sum row " << i);
      REQUIRE(valid[i] == expected[i].sum.has_value());
      if (expected[i].sum) { CHECK(values[i] == *expected[i].sum); }
    }
  }
  check_extremum(true);
  check_extremum(false);
  for (bool const star : {false, true}) {
    auto const view = next();
    CHECK(view.type() == cudf::data_type{cudf::type_id::INT64});
    CHECK(view.null_count() == 0);
    auto const values = host<int64_t>(view, stream);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      INFO((star ? "count_star" : "count") << " row " << i);
      CHECK(values[i] == (star ? expected[i].count_star : expected[i].count));
    }
  }
}
}  // namespace

TEST_CASE("cumulative evaluator matches the model for every direction and NULL order",
          "[window_evaluator][window_cumulative][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream    = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr        = cudf::get_current_device_resource_ref();
  auto const dir = GENERATE(
    ascending_nulls_first, ascending_nulls_last, descending_nulls_first, descending_nulls_last);
  auto const keyed = GENERATE(false, true);
  check_against_model<int32_t>(cudf::data_type{cudf::type_id::INT32}, keyed, dir, true, stream, mr);
  check_against_model<int64_t>(cudf::data_type{cudf::type_id::INT64}, keyed, dir, true, stream, mr);
  check_against_model<int32_t>(
    cudf::data_type{cudf::type_id::DECIMAL32, -2}, keyed, dir, true, stream, mr);
  check_against_model<int64_t>(
    cudf::data_type{cudf::type_id::DECIMAL64, -2}, keyed, dir, true, stream, mr);
  check_against_model<__int128_t>(
    cudf::data_type{cudf::type_id::DECIMAL128, -2}, keyed, dir, true, stream, mr);
}

TEST_CASE("cumulative evaluator takes extrema of dates",
          "[window_evaluator][window_cumulative][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  check_against_model<int32_t>(
    cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, true, ascending_nulls_last, false, stream, mr);
}

TEST_CASE("cumulative evaluator does not carry a value across partitions",
          "[window_evaluator][window_cumulative][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  // Partition 0 is 10, 20. Partition 1 starts with NULLs and then has 5. Partition 2 is all NULL.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({0, 0, 1, 1, 1, 2, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(
    column<int32_t>({1, 2, 1, 2, 3, 1, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<int64_t>({10, 20, 0, 0, 5, 0, 0},
                                    cudf::data_type{cudf::type_id::INT64},
                                    stream,
                                    mr,
                                    {true, true, false, false, true, false, false}));
  cudf::table input(std::move(columns));
  auto result = evaluate_cumulative_window(
    input.view(),
    {0},
    make_order({1}, ascending_nulls_last),
    {{function::sum, 2}, {function::max, 2}, {function::count, 2}, {function::count_star, 2}},
    stream,
    mr);
  CHECK(validity(result->view().column(0), stream) ==
        std::vector<bool>{true, true, false, false, true, false, false});
  CHECK(validity(result->view().column(1), stream) ==
        std::vector<bool>{true, true, false, false, true, false, false});
  auto sums = host<__int128_t>(result->view().column(0), stream);
  auto maxs = host<int64_t>(result->view().column(1), stream);
  CHECK(sums[0] == 10);
  CHECK(sums[1] == 30);
  CHECK(sums[4] == 5);
  CHECK(maxs[1] == 20);
  CHECK(maxs[4] == 5);
  CHECK(host<int64_t>(result->view().column(2), stream) ==
        std::vector<int64_t>{1, 2, 0, 0, 1, 0, 0});
  CHECK(host<int64_t>(result->view().column(3), stream) ==
        std::vector<int64_t>{1, 2, 1, 2, 3, 1, 2});
}

TEST_CASE(
  "cumulative evaluator keeps a NULL in the middle of a partition from hiding the running value",
  "[window_evaluator][window_cumulative][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({1, 2, 3, 4, 5}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<int64_t>({3, 0, 0, 7, 1},
                                    cudf::data_type{cudf::type_id::INT64},
                                    stream,
                                    mr,
                                    {true, false, false, true, true}));
  cudf::table input(std::move(columns));
  auto result =
    evaluate_cumulative_window(input.view(),
                               {},
                               make_order({0}, ascending_nulls_first),
                               {{function::sum, 1}, {function::min, 1}, {function::max, 1}},
                               stream,
                               mr);
  for (int c = 0; c < 3; ++c) {
    CHECK(result->view().column(c).null_count() == 0);
  }
  CHECK(host<__int128_t>(result->view().column(0), stream) ==
        std::vector<__int128_t>{3, 3, 3, 10, 11});
  CHECK(host<int64_t>(result->view().column(1), stream) == std::vector<int64_t>{3, 3, 3, 3, 1});
  CHECK(host<int64_t>(result->view().column(2), stream) == std::vector<int64_t>{3, 3, 3, 7, 7});
}

TEST_CASE("cumulative evaluator sums floating point values",
          "[window_evaluator][window_cumulative][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  for (auto const id : {cudf::type_id::FLOAT32, cudf::type_id::FLOAT64}) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(
      column<int32_t>({3, 1, 2, 4}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
    if (id == cudf::type_id::FLOAT32) {
      columns.push_back(column<float>({0.5f, 0.25f, 1.0f, 2.0f}, cudf::data_type{id}, stream, mr));
    } else {
      columns.push_back(column<double>({0.5, 0.25, 1.0, 2.0}, cudf::data_type{id}, stream, mr));
    }
    cudf::table input(std::move(columns));
    auto result = evaluate_cumulative_window(
      input.view(), {}, make_order({0}, ascending_nulls_last), {{function::sum, 1}}, stream, mr);
    CHECK(result->get_column(0).type() == cudf::data_type{cudf::type_id::FLOAT64});
    CHECK(host<double>(result->view().column(0), stream) ==
          std::vector<double>{1.75, 0.25, 1.25, 3.75});
  }
}

TEST_CASE("cumulative evaluator reports a decimal sum that exceeds DECIMAL(38)",
          "[window_evaluator][window_cumulative][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  // 6e37 + 6e37 reaches 10^38 without wrapping in the 128 bit accumulator. 9e37 + 9e37 passes
  // 2^127 and wraps, in either sign.
  int const digit = GENERATE(6, 9, -9);
  INFO(digit);
  __int128_t const value =
    static_cast<__int128_t>(digit) * 10'000'000'000'000'000'000LL * 1'000'000'000'000'000'000LL;
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column<int32_t>({1, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(
    column<__int128_t>({value, value}, cudf::data_type{cudf::type_id::DECIMAL128, 0}, stream, mr));
  cudf::table input(std::move(columns));
  CHECK_THROWS_AS(
    evaluate_cumulative_window(
      input.view(), {}, make_order({0}, ascending_nulls_last), {{function::sum, 1}}, stream, mr),
    std::overflow_error);
}

TEST_CASE("cumulative evaluator returns typed empty columns",
          "[window_evaluator][window_cumulative][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column<int32_t>({}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<int32_t>({}, cudf::data_type{cudf::type_id::DECIMAL32, -2}, stream, mr));
  cudf::table input(std::move(columns));
  auto result =
    evaluate_cumulative_window(input.view(),
                               {0},
                               make_order({0}, ascending_nulls_last),
                               {{function::sum, 1}, {function::max, 1}, {function::count, 1}},
                               stream,
                               mr);
  REQUIRE(result->num_columns() == 3);
  CHECK(result->get_column(0).type() == cudf::data_type{cudf::type_id::DECIMAL128, -2});
  CHECK(result->get_column(1).type() == cudf::data_type{cudf::type_id::DECIMAL32, -2});
  CHECK(result->get_column(2).type() == cudf::data_type{cudf::type_id::INT64});
  CHECK(result->num_rows() == 0);
}

TEST_CASE("cumulative evaluator rejects invalid requests",
          "[window_evaluator][window_cumulative][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column<int32_t>({1, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(
    column<double>({1.0, 2.0}, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr));
  cudf::table input(std::move(columns));
  auto const order = make_order({0}, ascending_nulls_last);
  auto call        = [&](std::vector<cudf::size_type> const& partitions,
                  ranking_order const& keys,
                  std::vector<cumulative_aggregate> const& aggregates) {
    return evaluate_cumulative_window(input.view(), partitions, keys, aggregates, stream, mr);
  };
  CHECK_NOTHROW(call({}, order, {{function::sum, 1}}));
  CHECK_THROWS_AS(call({}, order, {}), std::invalid_argument);
  CHECK_THROWS_AS(call({}, ranking_order{}, {{function::sum, 1}}), std::invalid_argument);
  CHECK_THROWS_AS(call({}, order, {{function::sum, 2}}), std::invalid_argument);
  CHECK_THROWS_AS(call({2}, order, {{function::sum, 1}}), std::invalid_argument);
  CHECK_THROWS_AS(call({}, make_order({2}, ascending_nulls_last), {{function::sum, 1}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(call({}, order, {{function::avg, 1}}), std::invalid_argument);
  // Floating point extrema are not supported.
  CHECK_THROWS_AS(call({}, order, {{function::min, 1}}), std::invalid_argument);
  CHECK_THROWS_AS(call({}, order, {{function::max, 1}}), std::invalid_argument);
}
