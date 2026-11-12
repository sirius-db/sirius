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

#include "op/window/framed.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {
using sirius::op::evaluate_framed_window;
using sirius::op::frame_current_row;
using sirius::op::frame_range_offset;
using sirius::op::frame_row_offset;
using sirius::op::frame_unbounded;
using sirius::op::frame_unit;
using sirius::op::framed_aggregate;
using sirius::op::window_frame;
using sirius::op::window_order;
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
    cudf::is_fixed_point(type) ? cudf::make_fixed_point_column(
                                   type, values.size(), cudf::mask_state::UNALLOCATED, stream, mr)
    : cudf::is_timestamp(type)
      ? cudf::make_timestamp_column(type, values.size(), cudf::mask_state::UNALLOCATED, stream, mr)
      : cudf::make_numeric_column(type, values.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(result->mutable_view().head(),
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

// A column from optional values: NULL where empty.
template <typename T>
std::unique_ptr<cudf::column> column(std::vector<std::optional<T>> const& values,
                                     cudf::data_type type,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  std::vector<T> data;
  std::vector<bool> valid;
  for (auto const& value : values) {
    data.push_back(value.value_or(T{}));
    valid.push_back(value.has_value());
  }
  return column<T>(data, type, stream, mr, valid);
}

template <typename T>
std::vector<T> host(cudf::column_view input, ::cuda::stream_ref stream)
{
  return sirius::test::operator_utils::copy_column_to_host<T>(input, stream);
}

std::vector<bool> validity(cudf::column_view input, ::cuda::stream_ref stream)
{
  stream.sync();
  return sirius::test::operator_utils::copy_validity_to_host(input);
}

__int128_t wide(duckdb::Value const& value)
{
  auto v = value.GetValueUnsafe<duckdb::hugeint_t>();
  return static_cast<__int128_t>(v.upper) * (static_cast<__int128_t>(1) << 64) + v.lower;
}

// Checks an optional host value against a column.
template <typename T>
void check_column(cudf::column_view result,
                  std::vector<std::optional<T>> const& expected,
                  ::cuda::stream_ref stream)
{
  REQUIRE(result.size() == static_cast<cudf::size_type>(expected.size()));
  auto const values = host<T>(result, stream);
  auto const valid  = validity(result, stream);
  for (std::size_t row = 0; row < expected.size(); ++row) {
    INFO("row " << row);
    REQUIRE(valid[row] == expected[row].has_value());
    if (!expected[row]) { continue; }
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(*expected[row])) {
        CHECK(std::isnan(values[row]));
      } else {
        CHECK(values[row] == Catch::Approx(*expected[row]));
      }
    } else {
      CHECK(values[row] == *expected[row]);
    }
  }
}

window_frame rows(sirius::op::frame_bound start, sirius::op::frame_bound end)
{
  return {frame_unit::rows, start, end};
}

window_frame range(sirius::op::frame_bound start, sirius::op::frame_bound end)
{
  return {frame_unit::range, start, end};
}

frame_row_offset preceding_rows(std::int64_t n) { return {n, true}; }
frame_row_offset following_rows(std::int64_t n) { return {n, false}; }
frame_range_offset preceding_range(__int128_t d) { return {d, true}; }
frame_range_offset following_range(__int128_t d) { return {d, false}; }

window_order ascending(std::vector<cudf::size_type> columns)
{
  return {columns,
          std::vector<cudf::order>(columns.size(), cudf::order::ASCENDING),
          std::vector<cudf::null_order>(columns.size(), cudf::null_order::AFTER)};
}

// The order key types the randomised comparison runs over. A raw key value r is the integer r,
// the decimal r * 0.25, the date r days after the epoch and the timestamp r seconds after it.
enum class key_kind { integer, decimal, date, timestamp };

struct key_type {
  key_kind kind;
  char const* name;
};

// The SQL expression of the key from its raw INTEGER column `kraw`.
std::string key_sql(key_kind kind)
{
  switch (kind) {
    case key_kind::integer: return "kraw";
    case key_kind::decimal: return "CAST(kraw * 0.25 AS DECIMAL(9,2))";
    case key_kind::date: return "DATE '1970-01-01' + kraw";
    case key_kind::timestamp: return "TIMESTAMP '1970-01-01 00:00:00' + to_seconds(kraw)";
  }
  return "";
}

// A RANGE offset of `d` raw key units in SQL, and in the unit of the key's cuDF type.
std::string offset_sql(key_kind kind, int d)
{
  switch (kind) {
    case key_kind::integer: return std::to_string(d);
    case key_kind::decimal: return "CAST(" + std::to_string(d) + " * 0.25 AS DECIMAL(9,2))";
    case key_kind::date: return "INTERVAL " + std::to_string(d) + " DAY";
    case key_kind::timestamp: return "INTERVAL " + std::to_string(d) + " SECOND";
  }
  return "";
}

__int128_t offset_units(key_kind kind, int d)
{
  switch (kind) {
    case key_kind::integer:
    case key_kind::date: return d;
    case key_kind::decimal: return d * 25;
    case key_kind::timestamp: return static_cast<__int128_t>(d) * 1'000'000;
  }
  return 0;
}

std::unique_ptr<cudf::column> key_column(key_kind kind,
                                         std::vector<std::optional<std::int32_t>> const& raw,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  switch (kind) {
    case key_kind::integer: return column(raw, cudf::data_type{cudf::type_id::INT32}, stream, mr);
    case key_kind::decimal: {
      std::vector<std::optional<std::int32_t>> unscaled;
      for (auto const& r : raw) {
        unscaled.push_back(r ? std::optional<std::int32_t>{*r * 25} : std::nullopt);
      }
      return column(unscaled, cudf::data_type{cudf::type_id::DECIMAL32, -2}, stream, mr);
    }
    case key_kind::date:
      return column(raw, cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, stream, mr);
    case key_kind::timestamp: {
      std::vector<std::optional<std::int64_t>> micros;
      for (auto const& r : raw) {
        micros.push_back(r ? std::optional<std::int64_t>{std::int64_t{*r} * 1'000'000}
                           : std::nullopt);
      }
      return column(micros, cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, stream, mr);
    }
  }
  return nullptr;
}

// One frame of the randomised comparison, in SQL and as a descriptor.
struct sql_frame {
  std::string sql;
  window_frame frame;
};

std::vector<sql_frame> comparison_frames(key_kind kind)
{
  auto const off = [&](int d) { return offset_sql(kind, d); };
  auto const big = std::int64_t{3'000'000'000};
  return {
    // ROWS. Huge offsets are as good as unbounded on the widening side and empty on the other.
    {"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
     rows(frame_unbounded{}, frame_current_row{})},
    {"ROWS BETWEEN CURRENT ROW AND CURRENT ROW", rows(frame_current_row{}, frame_current_row{})},
    {"ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING", rows(preceding_rows(2), following_rows(1))},
    {"ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING", rows(preceding_rows(3), preceding_rows(1))},
    {"ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING", rows(following_rows(1), following_rows(3))},
    {"ROWS BETWEEN 3 PRECEDING AND 5 PRECEDING", rows(preceding_rows(3), preceding_rows(5))},
    {"ROWS BETWEEN 2 FOLLOWING AND 1 FOLLOWING", rows(following_rows(2), following_rows(1))},
    {"ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
     rows(frame_current_row{}, frame_unbounded{})},
    {"ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING",
     rows(following_rows(1), frame_unbounded{})},
    {"ROWS BETWEEN UNBOUNDED PRECEDING AND 2 PRECEDING",
     rows(frame_unbounded{}, preceding_rows(2))},
    {"ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
     rows(frame_unbounded{}, frame_unbounded{})},
    {"ROWS BETWEEN 3000000000 PRECEDING AND 3000000000 FOLLOWING",
     rows(preceding_rows(big), following_rows(big))},
    {"ROWS BETWEEN 3000000000 FOLLOWING AND UNBOUNDED FOLLOWING",
     rows(following_rows(big), frame_unbounded{})},
    {"ROWS BETWEEN UNBOUNDED PRECEDING AND 3000000000 PRECEDING",
     rows(frame_unbounded{}, preceding_rows(big))},
    // RANGE with peer endpoints.
    {"RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
     range(frame_unbounded{}, frame_current_row{})},
    {"RANGE BETWEEN CURRENT ROW AND CURRENT ROW", range(frame_current_row{}, frame_current_row{})},
    {"RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
     range(frame_current_row{}, frame_unbounded{})},
    {"RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
     range(frame_unbounded{}, frame_unbounded{})},
    // RANGE with offsets, including frames off the current row and frames whose start is past
    // their end.
    {"RANGE BETWEEN " + off(3) + " PRECEDING AND " + off(2) + " FOLLOWING",
     range(preceding_range(offset_units(kind, 3)), following_range(offset_units(kind, 2)))},
    {"RANGE BETWEEN " + off(3) + " PRECEDING AND CURRENT ROW",
     range(preceding_range(offset_units(kind, 3)), frame_current_row{})},
    {"RANGE BETWEEN CURRENT ROW AND " + off(2) + " FOLLOWING",
     range(frame_current_row{}, following_range(offset_units(kind, 2)))},
    {"RANGE BETWEEN UNBOUNDED PRECEDING AND " + off(1) + " PRECEDING",
     range(frame_unbounded{}, preceding_range(offset_units(kind, 1)))},
    {"RANGE BETWEEN " + off(1) + " FOLLOWING AND UNBOUNDED FOLLOWING",
     range(following_range(offset_units(kind, 1)), frame_unbounded{})},
    {"RANGE BETWEEN " + off(3) + " PRECEDING AND " + off(1) + " PRECEDING",
     range(preceding_range(offset_units(kind, 3)), preceding_range(offset_units(kind, 1)))},
    {"RANGE BETWEEN " + off(1) + " FOLLOWING AND " + off(3) + " FOLLOWING",
     range(following_range(offset_units(kind, 1)), following_range(offset_units(kind, 3)))},
    {"RANGE BETWEEN " + off(1) + " PRECEDING AND " + off(3) + " PRECEDING",
     range(preceding_range(offset_units(kind, 1)), preceding_range(offset_units(kind, 3)))},
    {"RANGE BETWEEN " + off(3) + " FOLLOWING AND " + off(1) + " FOLLOWING",
     range(following_range(offset_units(kind, 3)), following_range(offset_units(kind, 1)))},
    {"RANGE BETWEEN " + off(0) + " PRECEDING AND " + off(0) + " FOLLOWING",
     range(preceding_range(0), following_range(0))},
    {"RANGE BETWEEN " + off(1000) + " PRECEDING AND " + off(1000) + " FOLLOWING",
     range(preceding_range(offset_units(kind, 1000)), following_range(offset_units(kind, 1000)))},
  };
}

// The aggregates computed over every frame, with their SQL.
struct sql_aggregate {
  function id;
  char const* sql;
};

std::vector<sql_aggregate> const comparison_aggregates{{function::sum, "sum(v)"},
                                                       {function::avg, "avg(v)"},
                                                       {function::count, "count(v)"},
                                                       {function::count_star, "count(*)"},
                                                       {function::min, "min(v)"},
                                                       {function::max, "max(v)"}};

// Checks the result column of `id` over BIGINT `v` against DuckDB's column `c`.
void check_against(cudf::column_view result,
                   function id,
                   duckdb::DataChunk& expected,
                   std::size_t c,
                   ::cuda::stream_ref stream)
{
  REQUIRE(static_cast<std::size_t>(result.size()) == expected.size());
  auto const valid = validity(result, stream);
  std::vector<__int128_t> sums;
  std::vector<double> averages;
  std::vector<std::int64_t> integers;
  switch (id) {
    case function::sum:
      REQUIRE(result.type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
      sums = host<__int128_t>(result, stream);
      break;
    case function::avg:
      REQUIRE(result.type() == cudf::data_type{cudf::type_id::FLOAT64});
      averages = host<double>(result, stream);
      break;
    default:
      REQUIRE(result.type() == cudf::data_type{cudf::type_id::INT64});
      integers = host<std::int64_t>(result, stream);
      break;
  }
  for (std::size_t row = 0; row < expected.size(); ++row) {
    auto const value = expected.GetValue(c, row);
    INFO("row " << row << ", expected " << value.ToString());
    REQUIRE(valid[row] == !value.IsNull());
    if (value.IsNull()) { continue; }
    switch (id) {
      case function::sum: CHECK(sums[row] == wide(value)); break;
      case function::avg:
        CHECK(averages[row] == Catch::Approx(value.GetValue<double>()).epsilon(1e-12));
        break;
      default: CHECK(integers[row] == value.GetValue<std::int64_t>()); break;
    }
  }
}

}  // namespace

TEST_CASE("framed evaluator matches DuckDB over ROWS and RANGE frames", "[window][framed][gpu]")
{
  sirius::test::scoped_sirius_disable disable;
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();

  auto const keyed       = GENERATE(false, true);
  auto const descending  = GENERATE(false, true);
  auto const nulls_first = GENERATE(false, true);
  auto const key         = GENERATE(key_type{key_kind::integer, "integer"},
                            key_type{key_kind::decimal, "decimal"},
                            key_type{key_kind::date, "date"},
                            key_type{key_kind::timestamp, "timestamp"});
  CAPTURE(keyed, descending, nulls_first, key.name);

  // Random rows: small groups and key ranges give ties and gaps, some groups of one row, and NULL
  // groups, keys and values.
  struct row {
    std::optional<std::int32_t> g;
    std::optional<std::int32_t> k;
    std::int32_t k2;
    std::optional<std::int64_t> v;
  };
  std::mt19937 rng(20261111);
  auto const pick = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
  std::vector<row> data;
  for (int i = 0; i < 240; ++i) {
    row r;
    if (keyed && pick(0, 19) != 0) { r.g = pick(0, 6); }
    if (keyed && i >= 236) { r.g = 100 + i; }  // Groups of one row.
    if (pick(0, 9) != 0) { r.k = pick(-12, 12); }
    r.k2 = pick(0, 2);
    if (pick(0, 6) != 0) {
      r.v = pick(0, 9) == 0 ? std::int64_t{pick(-9, 9)} << 50 : pick(-99, 99);
    }
    data.push_back(r);
  }
  // Sort on (g, k, k2) as the evaluator requires. The direction of the group key is free.
  std::stable_sort(data.begin(), data.end(), [&](row const& a, row const& b) {
    if (a.g != b.g) { return a.g < b.g; }  // nullopt sorts first.
    if (a.k != b.k) {
      if (!a.k || !b.k) { return !a.k == nulls_first; }
      return descending ? *a.k > *b.k : *a.k < *b.k;
    }
    return a.k2 < b.k2;
  });

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  REQUIRE_FALSE(
    con.Query("CREATE TABLE t(pos INTEGER, g INTEGER, kraw INTEGER, k2 INTEGER, v BIGINT)")
      ->HasError());
  {
    auto const sql = [](auto const& value) {
      return value ? std::to_string(*value) : std::string("NULL");
    };
    std::string insert = "INSERT INTO t VALUES ";
    for (std::size_t i = 0; i < data.size(); ++i) {
      auto const& r = data[i];
      insert += std::string(i ? "," : "") + "(" + std::to_string(i) + "," + sql(r.g) + "," +
                sql(r.k) + "," + std::to_string(r.k2) + "," + sql(r.v) + ")";
    }
    REQUIRE_FALSE(con.Query(insert)->HasError());
  }

  std::vector<std::optional<std::int32_t>> groups, keys;
  std::vector<std::int32_t> keys2;
  std::vector<std::optional<std::int64_t>> values;
  for (auto const& r : data) {
    groups.push_back(r.g);
    keys.push_back(r.k);
    keys2.push_back(r.k2);
    values.push_back(r.v);
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column(groups, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(key_column(key.kind, keys, stream, mr));
  columns.push_back(column(keys2, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  columns.push_back(column(values, cudf::data_type{cudf::type_id::INT64}, stream, mr));
  cudf::table input(std::move(columns));
  auto const group_columns =
    keyed ? std::vector<cudf::size_type>{0} : std::vector<cudf::size_type>{};

  auto const direction =
    std::string(descending ? " DESC" : " ASC") + (nulls_first ? " NULLS FIRST" : " NULLS LAST");
  auto const partition      = std::string(keyed ? "PARTITION BY g " : "");
  auto const cudf_direction = descending ? cudf::order::DESCENDING : cudf::order::ASCENDING;
  // cuDF's NULL order is in the ascending sense and reversed for a descending key.
  auto const cudf_nulls =
    nulls_first != descending ? cudf::null_order::BEFORE : cudf::null_order::AFTER;

  auto const run = [&](window_order const& order,
                       std::vector<sql_frame> const& frames,
                       std::string const& order_sql,
                       std::string const& rows_order_sql) {
    std::vector<framed_aggregate> aggregates;
    std::string select;
    for (auto const& frame : frames) {
      for (auto const& agg : comparison_aggregates) {
        aggregates.push_back({agg.id, 3, frame.frame});
        auto const& by = frame.frame.unit == frame_unit::rows ? rows_order_sql : order_sql;
        select += std::string(select.empty() ? "" : ", ") + agg.sql + " OVER (" + partition +
                  "ORDER BY " + by + " " + frame.sql + ")";
      }
    }
    auto const sql = "WITH s AS (SELECT pos, g, " + key_sql(key.kind) +
                     " AS k, k2, v FROM t) SELECT " + select + " FROM s ORDER BY pos";
    auto expected = con.Query(sql);
    INFO(sql);
    REQUIRE_FALSE(expected->HasError());
    // Every row fits one chunk.
    auto expected_rows = expected->Fetch();
    REQUIRE(expected_rows);
    REQUIRE(expected_rows->size() == data.size());
    auto result =
      evaluate_framed_window(input.view(), group_columns, order, aggregates, stream, mr);
    REQUIRE(result->num_columns() == static_cast<cudf::size_type>(aggregates.size()));
    for (std::size_t c = 0; c < aggregates.size(); ++c) {
      INFO(frames[c / comparison_aggregates.size()].sql << " " << comparison_aggregates[c % 6].sql);
      check_against(result->view().column(c), aggregates[c].function, *expected_rows, c, stream);
    }
  };

  SECTION("one order key")
  {
    // ROWS frames over tied keys depend on the order of the peers, so DuckDB breaks ties by
    // position, which is the evaluator's row order.
    run({{1}, {cudf_direction}, {cudf_nulls}},
        comparison_frames(key.kind),
        "k" + direction,
        "k" + direction + ", pos");
  }
  SECTION("two order keys")
  {
    std::vector<sql_frame> frames{
      {"RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
       range(frame_unbounded{}, frame_current_row{})},
      {"RANGE BETWEEN CURRENT ROW AND CURRENT ROW",
       range(frame_current_row{}, frame_current_row{})},
      {"RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
       range(frame_current_row{}, frame_unbounded{})},
      {"ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING", rows(preceding_rows(1), following_rows(1))},
    };
    run({{1, 2}, {cudf_direction, cudf::order::ASCENDING}, {cudf_nulls, cudf::null_order::AFTER}},
        frames,
        "k" + direction + ", k2 ASC NULLS LAST",
        "k" + direction + ", k2 ASC NULLS LAST, pos");
  }
}

TEST_CASE("framed evaluator treats a RANGE frame without order keys as the whole partition",
          "[window][framed][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  auto groups =
    column<std::int32_t>({1, 1, 2, 2, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto values =
    column<std::int64_t>({1, 2, 3, 4, 5}, cudf::data_type{cudf::type_id::INT64}, stream, mr);
  auto result =
    evaluate_framed_window(cudf::table_view{{groups->view(), values->view()}},
                           {0},
                           window_order{},
                           {{function::count_star, 1, sirius::op::default_window_frame()},
                            {function::max, 1, rows(frame_unbounded{}, frame_current_row{})}},
                           stream,
                           mr);
  check_column<std::int64_t>(result->view().column(0), {2, 2, 3, 3, 3}, stream);
  check_column<std::int64_t>(result->view().column(1), {1, 2, 3, 4, 5}, stream);
}

TEST_CASE("framed evaluator computes a DECIMAL128 running total and maximum",
          "[window][framed][gpu]")
{
  // The q51 shape: SUM and MAX over ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW.
  rmm::cuda_stream owning_stream;
  auto stream  = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr      = cudf::get_current_device_resource_ref();
  auto const t = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  auto groups =
    column<std::int32_t>({0, 0, 0, 1, 1}, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto keys =
    column<std::int32_t>({1, 2, 3, 1, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto values =
    column<__int128_t>({150, -25, 1000, 7, 0}, t, stream, mr, {true, true, true, false, true});
  auto const frame = rows(frame_unbounded{}, frame_current_row{});
  auto result =
    evaluate_framed_window(cudf::table_view{{groups->view(), keys->view(), values->view()}},
                           {0},
                           ascending({1}),
                           {{function::sum, 2, frame}, {function::max, 2, frame}},
                           stream,
                           mr);
  REQUIRE(result->get_column(0).type() == t);
  REQUIRE(result->get_column(1).type() == t);
  check_column<__int128_t>(result->view().column(0), {150, 125, 1125, std::nullopt, 0}, stream);
  check_column<__int128_t>(result->view().column(1), {150, 150, 1000, std::nullopt, 0}, stream);
}

TEST_CASE("framed evaluator computes string MIN and MAX", "[window][framed][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream  = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr      = cudf::get_current_device_resource_ref();
  auto strings = sirius::test::operator_utils::make_string_column(
    {"pear", "", "", "apple", "fig", "kiwi"}, stream, mr);
  set_validity(*strings, {true, false, false, true, true, true}, stream, mr);
  auto const frame = rows(preceding_rows(1), frame_current_row{});
  auto result      = evaluate_framed_window(cudf::table_view{{strings->view()}},
                                            {},
                                       window_order{},
                                            {{function::min, 0, frame}, {function::max, 0, frame}},
                                       stream,
                                       mr);
  REQUIRE(result->get_column(0).type().id() == cudf::type_id::STRING);
  // Row 2's frame holds two NULLs, so its extrema are NULL.
  std::vector<bool> const valid{true, true, false, true, true, true};
  auto const check = [&](cudf::column_view col, std::vector<std::string> const& expected) {
    stream.sync();
    auto const values = sirius::test::operator_utils::copy_column_to_host<std::string>(col);
    CHECK(validity(col, stream) == valid);
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (valid[i]) { CHECK(values[i] == expected[i]); }
    }
  };
  check(result->view().column(0), {"pear", "pear", "", "apple", "apple", "fig"});
  check(result->view().column(1), {"pear", "pear", "", "apple", "fig", "kiwi"});
}

TEMPLATE_TEST_CASE("framed evaluator orders NaN above every value in MIN and MAX",
                   "[window][framed][gpu]",
                   float,
                   double)
{
  rmm::cuda_stream owning_stream;
  auto stream     = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr         = cudf::get_current_device_resource_ref();
  auto const nan  = std::numeric_limits<TestType>::quiet_NaN();
  auto const type = cudf::data_type{cudf::type_to_id<TestType>()};
  auto values     = column<TestType>(
    {1, nan, nan, 2, 0, 3}, type, stream, mr, {true, true, true, true, false, true});
  auto const frame = rows(preceding_rows(1), frame_current_row{});
  auto result      = evaluate_framed_window(cudf::table_view{{values->view()}},
                                            {},
                                       window_order{},
                                            {{function::min, 0, frame},
                                             {function::max, 0, frame},
                                             {function::sum, 0, frame},
                                             {function::avg, 0, frame}},
                                       stream,
                                       mr);
  REQUIRE(result->get_column(0).type() == type);
  REQUIRE(result->get_column(2).type() == cudf::data_type{cudf::type_id::FLOAT64});
  using opt = std::optional<TestType>;
  // Frames: {1}, {1,NaN}, {NaN,NaN}, {NaN,2}, {2,NULL}, {NULL,3}.
  check_column<TestType>(
    result->view().column(0), {opt{1}, opt{1}, opt{nan}, opt{2}, opt{2}, opt{3}}, stream);
  check_column<TestType>(
    result->view().column(1), {opt{1}, opt{nan}, opt{nan}, opt{nan}, opt{2}, opt{3}}, stream);
  using dopt = std::optional<double>;
  check_column<double>(
    result->view().column(2), {dopt{1}, dopt{nan}, dopt{nan}, dopt{nan}, dopt{2}, dopt{3}}, stream);
  check_column<double>(
    result->view().column(3), {dopt{1}, dopt{nan}, dopt{nan}, dopt{nan}, dopt{2}, dopt{3}}, stream);
}

TEST_CASE("framed evaluator result types", "[window][framed][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  using cudf::type_id;
  auto const [input, sum] = GENERATE(
    std::pair{cudf::data_type{type_id::INT8}, cudf::data_type{type_id::DECIMAL128, 0}},
    std::pair{cudf::data_type{type_id::INT16}, cudf::data_type{type_id::DECIMAL128, 0}},
    std::pair{cudf::data_type{type_id::INT32}, cudf::data_type{type_id::DECIMAL128, 0}},
    std::pair{cudf::data_type{type_id::INT64}, cudf::data_type{type_id::DECIMAL128, 0}},
    std::pair{cudf::data_type{type_id::DECIMAL32, -2}, cudf::data_type{type_id::DECIMAL128, -2}},
    std::pair{cudf::data_type{type_id::DECIMAL64, -3}, cudf::data_type{type_id::DECIMAL128, -3}},
    std::pair{cudf::data_type{type_id::DECIMAL128, -4}, cudf::data_type{type_id::DECIMAL128, -4}},
    std::pair{cudf::data_type{type_id::FLOAT32}, cudf::data_type{type_id::FLOAT64}},
    std::pair{cudf::data_type{type_id::FLOAT64}, cudf::data_type{type_id::FLOAT64}});
  CAPTURE(static_cast<int>(input.id()));
  auto const size = cudf::size_of(input);
  auto values     = cudf::is_fixed_point(input)
                      ? cudf::make_fixed_point_column(input, 3, cudf::mask_state::ALL_VALID, stream, mr)
                      : cudf::make_numeric_column(input, 3, cudf::mask_state::ALL_VALID, stream, mr);
  REQUIRE(cudaMemsetAsync(values->mutable_view().head(), 0, 3 * size, stream.get()) == cudaSuccess);
  auto const frame = rows(preceding_rows(1), following_rows(1));
  auto const empty = GENERATE(false, true);
  cudf::table_view const view{{values->view()}};
  auto result = evaluate_framed_window(empty ? cudf::slice(view, {0, 0}, stream).front() : view,
                                       {},
                                       window_order{},
                                       {{function::sum, 0, frame},
                                        {function::sum_no_overflow, 0, frame},
                                        {function::avg, 0, frame},
                                        {function::count, 0, frame},
                                        {function::count_star, 0, frame},
                                        {function::min, 0, frame},
                                        {function::max, 0, frame}},
                                       stream,
                                       mr);
  REQUIRE(result->num_rows() == (empty ? 0 : 3));
  CHECK(result->get_column(0).type() == sum);
  CHECK(result->get_column(1).type() == sum);
  CHECK(result->get_column(2).type() == cudf::data_type{type_id::FLOAT64});
  CHECK(result->get_column(3).type() == cudf::data_type{type_id::INT64});
  CHECK(result->get_column(4).type() == cudf::data_type{type_id::INT64});
  CHECK(result->get_column(5).type() == input);
  CHECK(result->get_column(6).type() == input);
  if (!empty) { check_column<std::int64_t>(result->view().column(3), {2, 3, 2}, stream); }
}

TEST_CASE("framed evaluator gives empty frames NULL aggregates and zero counts",
          "[window][framed][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  // Two groups; the values of group 1 are all NULL.
  auto groups =
    column<std::int32_t>({0, 0, 0, 1, 1}, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto values         = column<std::int64_t>({1, 2, 3, 0, 0},
                                     cudf::data_type{cudf::type_id::INT64},
                                     stream,
                                     mr,
                                             {true, true, true, false, false});
  auto const reversed = rows(following_rows(1), preceding_rows(1));
  auto const whole    = rows(frame_unbounded{}, frame_unbounded{});
  auto result         = evaluate_framed_window(cudf::table_view{{groups->view(), values->view()}},
                                               {0},
                                       window_order{},
                                               {{function::sum, 1, reversed},
                                                {function::avg, 1, reversed},
                                                {function::min, 1, reversed},
                                                {function::count, 1, reversed},
                                                {function::count_star, 1, reversed},
                                                {function::sum, 1, whole},
                                                {function::count, 1, whole},
                                                {function::count_star, 1, whole}},
                                       stream,
                                       mr);
  for (int c = 0; c < 3; ++c) {
    CHECK(result->get_column(c).null_count() == 5);
  }
  check_column<std::int64_t>(result->view().column(3), {0, 0, 0, 0, 0}, stream);
  check_column<std::int64_t>(result->view().column(4), {0, 0, 0, 0, 0}, stream);
  check_column<__int128_t>(result->view().column(5), {6, 6, 6, std::nullopt, std::nullopt}, stream);
  check_column<std::int64_t>(result->view().column(6), {3, 3, 3, 0, 0}, stream);
  check_column<std::int64_t>(result->view().column(7), {3, 3, 3, 2, 2}, stream);
}

TEST_CASE("framed evaluator shares sums between SUM and AVG and keeps request order",
          "[window][framed][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr     = cudf::get_current_device_resource_ref();
  auto keys = column<std::int32_t>({1, 2, 2, 4}, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto values =
    column<std::int32_t>({10, 20, 30, 40}, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto const running = sirius::op::default_window_frame();
  auto const window  = range(preceding_range(1), following_range(1));
  auto result        = evaluate_framed_window(cudf::table_view{{keys->view(), values->view()}},
                                              {},
                                       ascending({0}),
                                              {{function::avg, 1, running},
                                               {function::sum, 1, window},
                                               {function::sum, 1, running},
                                               {function::avg, 1, window},
                                               {function::sum, 1, running}},
                                       stream,
                                       mr);
  using dopt         = std::optional<double>;
  check_column<double>(result->view().column(0), {dopt{10}, dopt{20}, dopt{20}, dopt{25}}, stream);
  check_column<__int128_t>(result->view().column(1), {60, 60, 60, 40}, stream);
  check_column<__int128_t>(result->view().column(2), {10, 60, 60, 100}, stream);
  check_column<double>(result->view().column(3), {dopt{20}, dopt{20}, dopt{20}, dopt{40}}, stream);
  check_column<__int128_t>(result->view().column(4), {10, 60, 60, 100}, stream);
}

TEST_CASE("framed window plan shares work and is decided from types alone", "[window][framed]")
{
  using plan_type = sirius::op::framed_window_plan;
  using cudf::type_id;
  std::vector<cudf::data_type> const types{cudf::data_type{type_id::INT32},
                                           cudf::data_type{type_id::INT32},
                                           cudf::data_type{type_id::FLOAT64},
                                           cudf::data_type{type_id::DECIMAL128, -2}};
  auto const running  = sirius::op::default_window_frame();
  auto const reversed = rows(preceding_rows(3), preceding_rows(5));
  auto const offset   = range(preceding_range(3), preceding_range(1));
  auto const backward = range(preceding_range(1), preceding_range(3));
  auto const plan     = sirius::op::plan_framed_window(types,
                                                       {0},
                                                   ascending({1}),
                                                       {{function::sum, 1, running},
                                                        {function::avg, 1, running},
                                                        {function::count, 1, running},
                                                        {function::max, 2, reversed},
                                                        {function::sum, 3, offset},
                                                        {function::count_star, 1, backward}});
  REQUIRE(plan.outputs.size() == 6);
  REQUIRE(plan.frames.size() == 4);
  // SUM and AVG share one sum of the cast argument, AVG and COUNT one count of the raw argument.
  CHECK(plan.frames[0].requests.size() == 2);
  auto const& sum = std::get<plan_type::output_request>(plan.outputs[0].recipe);
  auto const& avg = std::get<plan_type::output_average>(plan.outputs[1].recipe);
  auto const& cnt = std::get<plan_type::output_request>(plan.outputs[2].recipe);
  CHECK(avg.sum == sum.request);
  CHECK(avg.count == cnt.request);
  CHECK(plan.values[plan.requests[sum.request].values].how == plan_type::preparation::cast);
  CHECK(plan.requests[sum.request].type == cudf::data_type{type_id::DECIMAL128, 0});
  CHECK(plan.requests[cnt.request].type == cudf::data_type{type_id::INT64});
  CHECK(std::holds_alternative<plan_type::peer_range_frame>(plan.frames[0].strategy));
  // A ROWS frame whose start is after its end is empty.
  CHECK(std::holds_alternative<plan_type::empty_frame>(plan.frames[1].strategy));
  CHECK(std::holds_alternative<plan_type::output_float_extremum>(plan.outputs[3].recipe));
  {
    auto const& extremum = std::get<plan_type::output_float_extremum>(plan.outputs[3].recipe);
    // The extremum reads the NaN free values and NaN comes back from the maximum of the NaN flags.
    CHECK(plan.values[plan.requests[extremum.extremum].values].how ==
          plan_type::preparation::nans_to_nulls);
    CHECK(plan.requests[extremum.any_nan].kind == plan_type::aggregation::max);
    CHECK(plan.values[plan.requests[extremum.any_nan].values].how ==
          plan_type::preparation::nan_flags);
    CHECK(plan.requests[extremum.any_nan].type == cudf::data_type{type_id::BOOL8});
  }
  // A DECIMAL128 SUM is already in its accumulator type, so it reads the input as is and checks
  // DECIMAL(38).
  auto const& decimal_sum = std::get<plan_type::output_request>(plan.outputs[4].recipe);
  CHECK(plan.values[plan.requests[decimal_sum.request].values].how == plan_type::preparation::none);
  CHECK(plan.outputs[4].check_decimal38);
  CHECK_FALSE(plan.outputs[0].check_decimal38);
  auto const& forward = std::get<plan_type::offset_range_frame>(plan.frames[2].strategy);
  CHECK(forward.start.delta == 3);
  CHECK(forward.end.delta == -1);
  CHECK_FALSE(forward.clamp_reversed);
  CHECK(std::get<plan_type::offset_range_frame>(plan.frames[3].strategy).clamp_reversed);
  // Every value column is read by a request.
  for (std::size_t v = 0; v < plan.values.size(); ++v) {
    CHECK(std::any_of(plan.requests.begin(), plan.requests.end(), [&](auto const& request) {
      return request.values == v;
    }));
  }
  // The arguments need their logical types, and so does the order key of a RANGE offset frame,
  // but not the key of a frame of peers or the group columns.
  CHECK(plan.argument_columns == std::vector<cudf::size_type>{1, 2, 3});
  using columns = std::vector<cudf::size_type>;
  CHECK(sirius::op::logical_type_columns(plan) == columns{1, 2, 3});
  CHECK(sirius::op::logical_type_columns(sirius::op::plan_framed_window(
          types, {0}, ascending({1}), {{function::sum, 3, offset}})) == columns{3, 1});
  CHECK(sirius::op::logical_type_columns(sirius::op::plan_framed_window(
          types, {0}, ascending({1}), {{function::sum, 3, running}})) == columns{3});
  // Huge ROWS offsets are clamped so that cuDF's bound arithmetic cannot overflow.
  auto const huge = sirius::op::plan_framed_window(
    types,
    {},
    window_order{},
    {{function::count_star,
      0,
      rows(preceding_rows(std::numeric_limits<std::int64_t>::max()),
           following_rows(std::numeric_limits<std::int64_t>::max()))}});
  auto const& huge_rows = std::get<plan_type::rows_frame>(huge.frames[0].strategy);
  CHECK(huge_rows.preceding == std::numeric_limits<cudf::size_type>::max());
  CHECK(huge_rows.following == std::numeric_limits<cudf::size_type>::max() - 1);
}

TEST_CASE("framed evaluator rejects an input that does not match its plan", "[window][framed][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream     = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr         = cudf::get_current_device_resource_ref();
  auto const plan = sirius::op::plan_framed_window(
    {cudf::data_type{cudf::type_id::INT32}},
    {},
    window_order{},
    {{function::max, 0, rows(frame_unbounded{}, frame_current_row{})}});
  auto values = column<std::int64_t>({1, 2}, cudf::data_type{cudf::type_id::INT64}, stream, mr);
  CHECK_THROWS_AS(evaluate_framed_window(plan, cudf::table_view{{values->view()}}, stream, mr),
                  std::invalid_argument);
  CHECK_THROWS_AS(evaluate_framed_window(plan, cudf::table_view{}, stream, mr),
                  std::invalid_argument);
}

TEST_CASE("framed evaluator reports a DECIMAL128 sum past DECIMAL(38)", "[window][framed][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream      = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr          = cudf::get_current_device_resource_ref();
  auto const limit = sirius::op::int128_pow10(38) - 1;
  auto const half  = limit / 2 + 1;
  auto const sign  = GENERATE(1, -1);
  auto values      = column<__int128_t>(
    {sign * half, sign * half}, cudf::data_type{cudf::type_id::DECIMAL128, 0}, stream, mr);
  auto const frame = rows(frame_unbounded{}, frame_current_row{});
  CHECK_THROWS_AS(evaluate_framed_window(cudf::table_view{{values->view()}},
                                         {},
                                         window_order{},
                                         {{function::sum, 0, frame}},
                                         stream,
                                         mr),
                  std::overflow_error);
  // AVG has no DECIMAL(38) bound.
  CHECK_NOTHROW(evaluate_framed_window(cudf::table_view{{values->view()}},
                                       {},
                                       window_order{},
                                       {{function::avg, 0, frame}},
                                       stream,
                                       mr));
}

TEST_CASE("framed evaluator rejects invalid requests", "[window][framed][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream  = static_cast<::cuda::stream_ref>(owning_stream);
  auto mr      = cudf::get_current_device_resource_ref();
  auto keys    = column<std::int8_t>({1, 2}, cudf::data_type{cudf::type_id::INT8}, stream, mr);
  auto values  = column<std::int32_t>({1, 2}, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto floats  = column<double>({1, 2}, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
  auto strings = sirius::test::operator_utils::make_string_column({"a", "b"}, stream, mr);
  cudf::table_view const input{{keys->view(), values->view(), floats->view(), strings->view()}};
  auto const frame = rows(frame_unbounded{}, frame_current_row{});
  auto const fails = [&](std::vector<cudf::size_type> const& groups,
                         window_order const& order,
                         std::vector<framed_aggregate> const& aggregates) {
    CHECK_THROWS_AS(evaluate_framed_window(input, groups, order, aggregates, stream, mr),
                    std::invalid_argument);
  };
  fails({}, ascending({0}), {});
  fails({}, ascending({0}), {{function::sum, 4, frame}});
  fails({7}, ascending({0}), {{function::sum, 1, frame}});
  fails({}, ascending({7}), {{function::sum, 1, frame}});
  fails({}, {{0}, {cudf::order::ASCENDING}, {}}, {{function::sum, 1, frame}});
  fails({}, ascending({0}), {{function::first, 1, frame}});
  fails({}, ascending({0}), {{function::sum, 3, frame}});
  fails({}, ascending({0}), {{function::avg, 3, frame}});
  // Offsets in the wrong unit, negative, or with other than one order key.
  fails({}, ascending({0}), {{function::sum, 1, rows(preceding_range(1), frame_current_row{})}});
  fails({}, ascending({0}), {{function::sum, 1, range(preceding_rows(1), frame_current_row{})}});
  fails({}, ascending({0}), {{function::sum, 1, rows(preceding_rows(-1), frame_current_row{})}});
  fails({}, ascending({0}), {{function::sum, 1, range(preceding_range(-1), frame_current_row{})}});
  fails(
    {}, ascending({0, 1}), {{function::sum, 1, range(preceding_range(1), frame_current_row{})}});
  fails({}, window_order{}, {{function::sum, 1, range(preceding_range(1), frame_current_row{})}});
  // The offset must fit the key type, and the key must have a type that offsets apply to.
  fails({}, ascending({0}), {{function::sum, 1, range(preceding_range(200), frame_current_row{})}});
  fails({}, ascending({2}), {{function::sum, 1, range(preceding_range(1), frame_current_row{})}});
  fails({}, ascending({3}), {{function::sum, 1, range(preceding_range(1), frame_current_row{})}});
  CHECK_NOTHROW(
    evaluate_framed_window(input,
                           {},
                           ascending({0}),
                           {{function::sum, 1, range(preceding_range(127), following_range(127))}},
                           stream,
                           mr));
}
