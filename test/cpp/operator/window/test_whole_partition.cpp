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

#include "op/window/whole_partition.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

namespace {
using sirius::op::evaluate_whole_partition_window;
using sirius::op::whole_partition_aggregate;
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

template <typename T>
std::unique_ptr<cudf::column> column(std::vector<T> const& values,
                                     cudf::data_type type,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr,
                                     std::vector<bool> const& valid = {})
{
  auto result =
    (type.id() == cudf::type_id::DECIMAL32 || type.id() == cudf::type_id::DECIMAL64 ||
     type.id() == cudf::type_id::DECIMAL128)
      ? cudf::make_fixed_point_column(
          type, values.size(), cudf::mask_state::UNALLOCATED, stream, mr)
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

template <typename T>
constexpr cudf::type_id decimal_id()
{
  return sizeof(T) == 4   ? cudf::type_id::DECIMAL32
         : sizeof(T) == 8 ? cudf::type_id::DECIMAL64
                          : cudf::type_id::DECIMAL128;
}

__int128_t wide(duckdb::Value const& value)
{
  auto v = value.GetValue<duckdb::hugeint_t>();
  return static_cast<__int128_t>(v.upper) * (static_cast<__int128_t>(1) << 64) + v.lower;
}
}  // namespace

TEST_CASE("whole-partition evaluator matches DuckDB while preserving unsorted input rows",
          "[window_evaluator][gpu]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto setup = con.Query(
    "CREATE TABLE w(id INTEGER, k INTEGER, q VARCHAR, v BIGINT, f FLOAT, payload INTEGER);"
    "INSERT INTO w VALUES (0,2,'b',10,1,99),(1,1,'a',20,2,99),(2,2,'b',NULL,3,99),"
    "(3,NULL,'a',5,4,99),(4,1,'a',40,NULL,99),(5,NULL,'a',7,6,99),"
    "(6,2,'c',NULL,NULL,99),(7,2,'c',NULL,NULL,99),(8,1,NULL,3,9,99),(9,1,NULL,4,10,99)");
  REQUIRE_FALSE(setup->HasError());
  rmm::cuda_stream owning_stream;
  auto stream = owning_stream.view();
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({2, 1, 2, 0, 1, 0, 2, 2, 1, 1},
                    cudf::data_type{cudf::type_id::INT32},
                    stream,
                    mr,
                    {true, true, true, false, true, false, true, true, true, true}));
  auto strings = sirius::test::operator_utils::make_string_column(
    {"b", "a", "b", "a", "a", "a", "c", "c", "", ""}, stream, mr);
  set_validity(
    *strings, {true, true, true, true, true, true, true, true, false, false}, stream, mr);
  columns.push_back(std::move(strings));
  columns.push_back(
    column<int64_t>({10, 20, 0, 5, 40, 7, 0, 0, 3, 4},
                    cudf::data_type{cudf::type_id::INT64},
                    stream,
                    mr,
                    {true, true, false, true, true, true, false, false, true, true}));
  columns.push_back(column<float>({1, 2, 3, 4, 0, 6, 0, 0, 9, 10},
                                  cudf::data_type{cudf::type_id::FLOAT32},
                                  stream,
                                  mr,
                                  {true, true, true, true, false, true, false, false, true, true}));
  columns.push_back(column<int32_t>(
    {99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  cudf::table input(std::move(columns));
  auto const keyed  = GENERATE(false, true);
  auto const sliced = GENERATE(false, true);
  auto view         = sliced ? cudf::slice(input.view(), {2, 9}, stream).front() : input.view();
  // Column 2 appears in a SUM and in two AVGs, so the three requests share one argument.
  auto result = evaluate_whole_partition_window(
    view,
    keyed ? std::vector<cudf::size_type>{1, 0} : std::vector<cudf::size_type>{},
    {{function::sum, 2}, {function::avg, 2}, {function::sum, 3}, {function::avg, 2}},
    stream,
    mr);
  auto window   = keyed ? "PARTITION BY k,q" : "";
  auto expected = con.Query(std::string("SELECT sum(v) OVER (") + window + "), avg(v) OVER (" +
                            window + "),sum(f) OVER (" + window + "),avg(v) OVER (" + window +
                            ") FROM w " + (sliced ? "WHERE id>=2 AND id<9 " : "") + "ORDER BY id");
  REQUIRE_FALSE(expected->HasError());
  REQUIRE(result->num_rows() == expected->RowCount());
  REQUIRE(result->num_columns() == 4);
  CHECK(result->get_column(0).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
  auto expected_rows = expected->Fetch();
  REQUIRE(expected_rows);
  REQUIRE(expected_rows->size() == expected->RowCount());
  auto sums = host<__int128_t>(result->view().column(0), stream);
  for (cudf::size_type c = 0; c < result->num_columns(); ++c) {
    auto valid   = validity(result->view().column(c), stream);
    auto doubles = c ? host<double>(result->view().column(c), stream) : std::vector<double>{};
    for (cudf::size_type row = 0; row < result->num_rows(); ++row) {
      auto value = expected_rows->GetValue(c, row);
      REQUIRE(valid[row] == !value.IsNull());
      if (!valid[row]) { continue; }
      if (c == 0) {
        CHECK(sums[row] == wide(value));
      } else {
        CHECK(doubles[row] == Catch::Approx(value.GetValue<double>()));
      }
    }
  }
}

TEST_CASE("whole-partition evaluator widens before summation and handles cancellation",
          "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream    = owning_stream.view();
  auto mr        = cudf::get_current_device_resource_ref();
  auto const max = std::numeric_limits<int64_t>::max();
  auto const min = std::numeric_limits<int64_t>::min();
  SECTION("64 bit integers widen to 128 bits")
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(
      column<int32_t>({0, 0, 1, 1, 2, 2, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
    columns.push_back(column<int64_t>(
      {max, max, min, min, max, -max, 1}, cudf::data_type{cudf::type_id::INT64}, stream, mr));
    auto input  = cudf::table(std::move(columns));
    auto result = evaluate_whole_partition_window(
      input.view(), {0}, {{function::sum, 1}, {function::avg, 1}}, stream, mr);
    auto sums = host<__int128_t>(result->view().column(0), stream);
    auto avgs = host<double>(result->view().column(1), stream);
    CHECK(sums == std::vector<__int128_t>{2 * static_cast<__int128_t>(max),
                                          2 * static_cast<__int128_t>(max),
                                          2 * static_cast<__int128_t>(min),
                                          2 * static_cast<__int128_t>(min),
                                          1,
                                          1,
                                          1});
    CHECK(avgs[0] == static_cast<double>(max));
    CHECK(avgs[2] == static_cast<double>(min));
    CHECK(avgs[4] == Catch::Approx(1.0 / 3));
  }

  SECTION("float sums accumulate in double")
  {
    auto floats =
      column<float>({16777216, 1, -16777216}, cudf::data_type{cudf::type_id::FLOAT32}, stream, mr);
    auto floating = evaluate_whole_partition_window(
      cudf::table_view{{floats->view()}}, {}, {{function::sum, 0}, {function::avg, 0}}, stream, mr);
    CHECK(host<double>(floating->view().column(0), stream) == std::vector<double>{1, 1, 1});
    CHECK(host<double>(floating->view().column(1), stream)[0] == Catch::Approx(1.0 / 3));
  }
}

TEMPLATE_TEST_CASE("whole-partition evaluator preserves decimal scale and null results",
                   "[window_evaluator][gpu]",
                   int32_t,
                   int64_t,
                   __int128_t)
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  auto id          = decimal_id<TestType>();
  auto const keyed = GENERATE(false, true);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({0, 0, 1, 1, 2, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<TestType>({100, 200, 0, 0, 450, 50},
                                     cudf::data_type{id, -2},
                                     stream,
                                     mr,
                                     {true, true, false, false, true, true}));
  auto input  = cudf::table(std::move(columns));
  auto result = evaluate_whole_partition_window(
    input.view(),
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
    {{function::sum, 1}, {function::avg, 1}},
    stream,
    mr);
  CHECK(result->get_column(0).type() == cudf::data_type{cudf::type_id::DECIMAL128, -2});
  auto valid = validity(result->view().column(0), stream);
  CHECK(valid == (keyed ? std::vector<bool>{true, true, false, false, true, true}
                        : std::vector<bool>(6, true)));
  CHECK(validity(result->view().column(1), stream) == valid);
  auto sums = host<__int128_t>(result->view().column(0), stream);
  auto avgs = host<double>(result->view().column(1), stream);
  CHECK(sums[0] == (keyed ? 300 : 800));
  CHECK(sums[4] == (keyed ? 500 : 800));
  CHECK(avgs[0] == Catch::Approx(keyed ? 1.5 : 2.0));
  CHECK(avgs[4] == Catch::Approx(keyed ? 2.5 : 2.0));
}

TEST_CASE("whole-partition evaluator returns typed empties and nulls for all-null input",
          "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  auto const keyed = GENERATE(false, true);
  auto const empty = GENERATE(false, true);
  auto input       = column<int32_t>(empty ? std::vector<int32_t>{} : std::vector<int32_t>{0, 0, 0},
                               cudf::data_type{cudf::type_id::INT32},
                               stream,
                               mr,
                               empty ? std::vector<bool>{} : std::vector<bool>(3, false));
  auto result = evaluate_whole_partition_window(
    cudf::table_view{{input->view()}},
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
    {{function::sum, 0},
     {function::avg, 0},
     {function::count, 0},
     {function::count_star, 0},
     {function::min, 0},
     {function::max, 0}},
    stream,
    mr);
  REQUIRE(result->num_rows() == (empty ? 0 : 3));
  REQUIRE(result->num_columns() == 6);
  CHECK(result->get_column(0).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
  CHECK(result->get_column(1).type() == cudf::data_type{cudf::type_id::FLOAT64});
  CHECK(result->get_column(0).null_count() == result->num_rows());
  CHECK(result->get_column(1).null_count() == result->num_rows());
  for (auto i : {2, 3}) {
    CHECK(result->get_column(i).type() == cudf::data_type{cudf::type_id::INT64});
    CHECK(result->get_column(i).null_count() == 0);
  }
  CHECK(host<int64_t>(result->view().column(2), stream) ==
        std::vector<int64_t>(result->num_rows(), 0));
  CHECK(host<int64_t>(result->view().column(3), stream) ==
        std::vector<int64_t>(result->num_rows(), 3));
  for (auto i : {4, 5}) {
    CHECK(result->get_column(i).type() == input->type());
    CHECK(result->get_column(i).null_count() == result->num_rows());
  }
}

TEST_CASE("whole-partition evaluator reports decimal accumulator overflow",
          "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream                 = owning_stream.view();
  auto mr                     = cudf::get_current_device_resource_ref();
  auto const limit            = sirius::op::int128_pow10(38) - 1;
  auto const carrier_overflow = GENERATE(false, true);
  // Two half-limits cross DECIMAL(38)'s precision bound without overflowing int128.
  auto const negative  = GENERATE(false, true);
  auto const magnitude = carrier_overflow ? limit : limit / 2 + 1;
  auto const value     = negative ? -magnitude : magnitude;
  auto input =
    column<__int128_t>({value, value}, cudf::data_type{cudf::type_id::DECIMAL128, 0}, stream, mr);
  auto const keyed = GENERATE(false, true);
  auto keys        = column<int32_t>({0, 0}, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  CHECK_THROWS_AS(evaluate_whole_partition_window(
                    cudf::table_view{{keys->view(), input->view()}},
                    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
                    {{function::sum, 1}},
                    stream,
                    mr),
                  std::overflow_error);
}

TEST_CASE("whole-partition evaluator counts and takes extrema over NULL and non-NULL values",
          "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  auto const keyed = GENERATE(false, true);
  auto const dec   = cudf::data_type{cudf::type_id::DECIMAL64, -2};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({0, 0, 0, 1, 1, 2, 2, 3}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<int64_t>({5, 0, 3, 0, 0, 7, 0, 0},
                                    cudf::data_type{cudf::type_id::INT64},
                                    stream,
                                    mr,
                                    {true, false, true, false, false, true, false, false}));
  auto strings = sirius::test::operator_utils::make_string_column(
    {"pear", "apple", "", "", "kiwi", "fig", "", ""}, stream, mr);
  set_validity(*strings, {true, true, false, false, true, true, false, false}, stream, mr);
  columns.push_back(std::move(strings));
  columns.push_back(column<int64_t>({150, 20, 0, 99, 0, 5, 0, 0},
                                    dec,
                                    stream,
                                    mr,
                                    {true, true, false, true, false, true, false, false}));
  cudf::table input(std::move(columns));
  auto result = evaluate_whole_partition_window(
    input.view(),
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
    {{function::count, 1},
     {function::count_star, 1},
     {function::min, 1},
     {function::max, 1},
     {function::min, 2},
     {function::max, 2},
     {function::min, 3},
     {function::max, 3}},
    stream,
    mr);
  REQUIRE(result->num_columns() == 8);
  REQUIRE(result->num_rows() == 8);
  for (auto c : {0, 1, 2, 3}) {
    CHECK(result->get_column(c).type() == cudf::data_type{cudf::type_id::INT64});
  }
  CHECK(result->get_column(4).type() == cudf::data_type{cudf::type_id::STRING});
  CHECK(result->get_column(5).type() == cudf::data_type{cudf::type_id::STRING});
  CHECK(result->get_column(6).type() == dec);
  CHECK(result->get_column(7).type() == dec);

  auto const check_ints = [&](cudf::size_type c,
                              std::vector<int64_t> const& expected,
                              std::vector<bool> const& expected_valid) {
    auto const column = result->view().column(c);
    CHECK(validity(column, stream) == expected_valid);
    auto const values = host<int64_t>(column, stream);
    for (std::size_t row = 0; row < expected.size(); ++row) {
      if (expected_valid[row]) { CHECK(values[row] == expected[row]); }
    }
  };
  auto const check_strings = [&](cudf::size_type c,
                                 std::vector<std::string> const& expected,
                                 std::vector<bool> const& expected_valid) {
    auto const column = result->view().column(c);
    CHECK(validity(column, stream) == expected_valid);
    stream.synchronize();
    auto const values = sirius::test::operator_utils::copy_column_to_host<std::string>(column);
    for (std::size_t row = 0; row < expected.size(); ++row) {
      if (expected_valid[row]) { CHECK(values[row] == expected[row]); }
    }
  };
  std::vector<bool> const all_valid(8, true);
  std::vector<bool> const keyed_valid{true, true, true, false, false, true, true, false};

  // COUNT skips NULLs and COUNT(*) does not.
  check_ints(0,
             keyed ? std::vector<int64_t>{2, 2, 2, 0, 0, 1, 1, 0} : std::vector<int64_t>(8, 3),
             all_valid);
  check_ints(1,
             keyed ? std::vector<int64_t>{3, 3, 3, 2, 2, 2, 2, 1} : std::vector<int64_t>(8, 8),
             all_valid);
  check_ints(2,
             keyed ? std::vector<int64_t>{3, 3, 3, 0, 0, 7, 7, 0} : std::vector<int64_t>(8, 3),
             keyed ? keyed_valid : all_valid);
  check_ints(3,
             keyed ? std::vector<int64_t>{5, 5, 5, 0, 0, 7, 7, 0} : std::vector<int64_t>(8, 7),
             keyed ? keyed_valid : all_valid);
  auto const keyed_string_valid =
    std::vector<bool>{true, true, true, true, true, true, true, false};
  check_strings(
    4,
    keyed ? std::vector<std::string>{"apple", "apple", "apple", "kiwi", "kiwi", "fig", "fig", ""}
          : std::vector<std::string>(8, "apple"),
    keyed ? keyed_string_valid : all_valid);
  check_strings(
    5,
    keyed ? std::vector<std::string>{"pear", "pear", "pear", "kiwi", "kiwi", "fig", "fig", ""}
          : std::vector<std::string>(8, "pear"),
    keyed ? keyed_string_valid : all_valid);
  check_ints(6,
             keyed ? std::vector<int64_t>{20, 20, 20, 99, 99, 5, 5, 0} : std::vector<int64_t>(8, 5),
             keyed ? keyed_string_valid : all_valid);
  check_ints(
    7,
    keyed ? std::vector<int64_t>{150, 150, 150, 99, 99, 5, 5, 0} : std::vector<int64_t>(8, 150),
    keyed ? keyed_string_valid : all_valid);
}

TEMPLATE_TEST_CASE("whole-partition evaluator orders NaN above every value in min and max",
                   "[window_evaluator][gpu]",
                   float,
                   double)
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  auto const keyed = GENERATE(false, true);
  auto const type  = cudf::data_type{std::is_same_v<TestType, float> ? cudf::type_id::FLOAT32
                                                                     : cudf::type_id::FLOAT64};
  auto const nan   = std::numeric_limits<TestType>::quiet_NaN();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column<int32_t>(
    {0, 0, 0, 1, 1, 2, 2, 3, 4, 4}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(
    column<TestType>({1, nan, 2, nan, nan, nan, 0, 0, 5, 3},
                     type,
                     stream,
                     mr,
                     {true, true, true, true, true, true, false, false, true, true}));
  cudf::table input(std::move(columns));
  auto result = evaluate_whole_partition_window(
    input.view(),
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
    {{function::min, 1}, {function::max, 1}},
    stream,
    mr);
  REQUIRE(result->num_columns() == 2);
  REQUIRE(result->num_rows() == 10);
  CHECK(result->get_column(0).type() == type);
  CHECK(result->get_column(1).type() == type);

  // A partition's maximum is NaN when it holds any NaN. Its minimum is NaN only when every
  // non-NULL value is NaN, and a partition of NULLs gives NULL.
  std::vector<TestType> const keyed_min{1, 1, 1, nan, nan, nan, nan, 0, 3, 3};
  std::vector<TestType> const keyed_max{nan, nan, nan, nan, nan, nan, nan, 0, 5, 5};
  std::vector<bool> const keyed_valid{true, true, true, true, true, true, true, false, true, true};
  auto const check = [&](cudf::size_type c,
                         std::vector<TestType> const& expected,
                         std::vector<bool> const& expected_valid) {
    auto const column = result->view().column(c);
    CHECK(validity(column, stream) == expected_valid);
    auto const values = host<TestType>(column, stream);
    for (std::size_t row = 0; row < expected.size(); ++row) {
      if (!expected_valid[row]) { continue; }
      if (std::isnan(expected[row])) {
        CHECK(std::isnan(values[row]));
      } else {
        CHECK(values[row] == expected[row]);
      }
    }
  };
  check(0,
        keyed ? keyed_min : std::vector<TestType>(10, 1),
        keyed ? keyed_valid : std::vector<bool>(10, true));
  check(1,
        keyed ? keyed_max : std::vector<TestType>(10, nan),
        keyed ? keyed_valid : std::vector<bool>(10, true));
}

TEMPLATE_TEST_CASE("whole-partition evaluator groups by decimal partition keys",
                   "[window_evaluator][gpu]",
                   int32_t,
                   int64_t,
                   __int128_t)
{
  rmm::cuda_stream owning_stream;
  auto stream = owning_stream.view();
  auto mr     = cudf::get_current_device_resource_ref();
  auto id     = decimal_id<TestType>();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column<TestType>({100, 250, 100, 0, 250, 0},
                                     cudf::data_type{id, -2},
                                     stream,
                                     mr,
                                     {true, true, true, false, true, false}));
  columns.push_back(
    column<int64_t>({1, 2, 3, 4, 5, 6}, cudf::data_type{cudf::type_id::INT64}, stream, mr));
  cudf::table input(std::move(columns));
  auto result = evaluate_whole_partition_window(
    input.view(), {0}, {{function::sum, 1}, {function::count_star, 1}}, stream, mr);
  CHECK(host<__int128_t>(result->view().column(0), stream) ==
        std::vector<__int128_t>{4, 7, 4, 10, 7, 10});
  CHECK(host<int64_t>(result->view().column(1), stream) == std::vector<int64_t>{2, 2, 2, 2, 2, 2});
}

TEMPLATE_TEST_CASE("whole-partition evaluator sums narrow integers exactly",
                   "[window_evaluator][gpu]",
                   int8_t,
                   int16_t,
                   int32_t)
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  auto const keyed = GENERATE(false, true);
  auto const id    = sizeof(TestType) == 1   ? cudf::type_id::INT8
                     : sizeof(TestType) == 2 ? cudf::type_id::INT16
                                             : cudf::type_id::INT32;
  auto const max   = std::numeric_limits<TestType>::max();
  auto const min   = std::numeric_limits<TestType>::min();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({0, 0, 0, 1, 1, 1, 2, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<TestType>({max, max, max, min, min, 0, max, max},
                                     cudf::data_type{id},
                                     stream,
                                     mr,
                                     {true, true, true, true, true, false, true, true}));
  cudf::table input(std::move(columns));
  auto result = evaluate_whole_partition_window(
    input.view(),
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
    {{function::sum, 1}, {function::avg, 1}},
    stream,
    mr);
  CHECK(result->get_column(0).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
  CHECK(result->get_column(1).type() == cudf::data_type{cudf::type_id::FLOAT64});
  auto const sums = host<__int128_t>(result->view().column(0), stream);
  auto const avgs = host<double>(result->view().column(1), stream);
  auto const big  = static_cast<__int128_t>(max);
  auto const low  = static_cast<__int128_t>(min);
  // Without partition columns the total of five maxima and two minima is outside the column type.
  auto const total = 5 * big + 2 * low;
  CHECK(sums[0] == (keyed ? 3 * big : total));
  CHECK(sums[3] == (keyed ? 2 * low : total));
  CHECK(sums[6] == (keyed ? 2 * big : total));
  CHECK(avgs[0] ==
        Catch::Approx(keyed ? static_cast<double>(max) : static_cast<double>(total) / 7));
  CHECK(avgs[3] ==
        Catch::Approx(keyed ? static_cast<double>(min) : static_cast<double>(total) / 7));
  CHECK(avgs[6] ==
        Catch::Approx(keyed ? static_cast<double>(max) : static_cast<double>(total) / 7));
}

TEST_CASE("whole-partition evaluator treats sum_no_overflow as sum", "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  auto const keyed = GENERATE(false, true);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({0, 0, 1, 1, 1}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<int64_t>({4, -1, 7, 2, 9},
                                    cudf::data_type{cudf::type_id::INT64},
                                    stream,
                                    mr,
                                    {true, true, true, false, true}));
  columns.push_back(column<int64_t>(
    {150, 25, -10, 0, 5}, cudf::data_type{cudf::type_id::DECIMAL64, -2}, stream, mr));
  cudf::table input(std::move(columns));
  auto const partition_columns =
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{};
  auto result = evaluate_whole_partition_window(input.view(),
                                                partition_columns,
                                                {{function::sum_no_overflow, 1},
                                                 {function::sum, 1},
                                                 {function::sum_no_overflow, 2},
                                                 {function::sum, 2}},
                                                stream,
                                                mr);
  REQUIRE(result->num_columns() == 4);
  CHECK(result->get_column(0).type() == result->get_column(1).type());
  CHECK(result->get_column(2).type() == result->get_column(3).type());
  CHECK(host<__int128_t>(result->view().column(0), stream) ==
        host<__int128_t>(result->view().column(1), stream));
  CHECK(host<__int128_t>(result->view().column(2), stream) ==
        host<__int128_t>(result->view().column(3), stream));
  auto const sums = host<__int128_t>(result->view().column(0), stream);
  CHECK(sums[0] == (keyed ? 3 : 19));
  CHECK(sums[2] == (keyed ? 16 : 19));
}

TEST_CASE("whole-partition evaluator rejects invalid requests", "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = owning_stream.view();
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column<int32_t>({0, 0, 1}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<int64_t>({1, 2, 3}, cudf::data_type{cudf::type_id::INT64}, stream, mr));
  columns.push_back(
    column<double>({1.5, 2.5, 3.5}, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr));
  columns.push_back(sirius::test::operator_utils::make_string_column({"a", "b", "c"}, stream, mr));
  cudf::table input(std::move(columns));
  auto const evaluate = [&](std::vector<cudf::size_type> const& partition_columns,
                            std::vector<whole_partition_aggregate> const& aggregates) {
    return evaluate_whole_partition_window(input.view(), partition_columns, aggregates, stream, mr);
  };

  SECTION("no aggregates") { CHECK_THROWS_AS(evaluate({0}, {}), std::invalid_argument); }
  SECTION("partition column outside the input")
  {
    CHECK_THROWS_AS(evaluate({4}, {{function::sum, 1}}), std::invalid_argument);
    CHECK_THROWS_AS(evaluate({-1}, {{function::sum, 1}}), std::invalid_argument);
  }
  SECTION("argument column outside the input")
  {
    CHECK_THROWS_AS(evaluate({0}, {{function::sum, 4}}), std::invalid_argument);
    CHECK_THROWS_AS(evaluate({0}, {{function::count, -1}}), std::invalid_argument);
  }
  SECTION("unsupported aggregate")
  {
    CHECK_THROWS_AS(evaluate({0}, {{function::first, 1}}), std::invalid_argument);
    CHECK_THROWS_AS(evaluate({0}, {{function::stddev_samp, 1}}), std::invalid_argument);
  }
  SECTION("unsupported partition column type")
  {
    CHECK_THROWS_AS(evaluate({2}, {{function::sum, 1}}), std::invalid_argument);
  }
  SECTION("unsupported argument type")
  {
    CHECK_THROWS_AS(evaluate({0}, {{function::sum, 3}}), std::invalid_argument);
    CHECK_THROWS_AS(evaluate({0}, {{function::avg, 3}}), std::invalid_argument);
  }
}

TEST_CASE("whole-partition evaluator widens only the sum columns that could overflow",
          "[window_evaluator][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream      = owning_stream.view();
  auto mr          = cudf::get_current_device_resource_ref();
  auto const keyed = GENERATE(false, true);
  auto const max64 = std::numeric_limits<int64_t>::max();
  auto const max32 = std::numeric_limits<int32_t>::max();
  auto const dec64 = cudf::data_type{cudf::type_id::DECIMAL64, -2};
  auto const dec32 = cudf::data_type{cudf::type_id::DECIMAL32, -2};
  std::vector<int64_t> const small64{1, 2, 3, -4};
  std::vector<int64_t> const large64{max64, max64, 5, 6};
  std::vector<int64_t> const small_dec64{100, 250, -5, 7};
  std::vector<int64_t> const large_dec64{max64, max64, 5, 6};
  std::vector<int32_t> const small_dec32{100, 200, 300, -50};
  std::vector<int32_t> const large_dec32{max32, max32, 1, 2};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({0, 0, 1, 1}, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column<int64_t>(small64, cudf::data_type{cudf::type_id::INT64}, stream, mr));
  columns.push_back(column<int64_t>(large64, cudf::data_type{cudf::type_id::INT64}, stream, mr));
  columns.push_back(column<int64_t>(small_dec64, dec64, stream, mr));
  columns.push_back(column<int64_t>(large_dec64, dec64, stream, mr));
  columns.push_back(column<int32_t>(small_dec32, dec32, stream, mr));
  columns.push_back(column<int32_t>(large_dec32, dec32, stream, mr));
  cudf::table input(std::move(columns));
  std::vector<whole_partition_aggregate> aggregates;
  for (cudf::size_type c = 1; c <= 6; ++c) {
    aggregates.push_back({function::sum, c});
  }
  auto result = evaluate_whole_partition_window(
    input.view(),
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{},
    aggregates,
    stream,
    mr);
  REQUIRE(result->num_columns() == 6);
  auto const group = [&](std::size_t row) { return keyed ? row / 2 : 0; };
  auto const check = [&](cudf::size_type c, auto const& values, int scale) {
    auto const column = result->view().column(c);
    CHECK(column.type() == cudf::data_type{cudf::type_id::DECIMAL128, scale});
    auto const sums = host<__int128_t>(column, stream);
    for (std::size_t row = 0; row < values.size(); ++row) {
      __int128_t expected = 0;
      for (std::size_t other = 0; other < values.size(); ++other) {
        if (group(other) == group(row)) { expected += values[other]; }
      }
      CHECK(sums[row] == expected);
    }
  };
  check(0, small64, 0);
  check(1, large64, 0);
  check(2, small_dec64, -2);
  check(3, large_dec64, -2);
  check(4, small_dec32, -2);
  check(5, large_dec32, -2);
}
