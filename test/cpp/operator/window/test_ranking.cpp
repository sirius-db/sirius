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

#include "op/cudf_sort_order.hpp"
#include "op/window/ranking.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>

#include <limits>
#include <string>

namespace {
using sirius::op::evaluate_ranking_window;
using sirius::op::ranking_function;
using sirius::op::ranking_order;

void set_validity(cudf::column& column,
                  std::vector<bool> const& valid,
                  ::cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
{
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
                                     cudf::type_id id,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr,
                                     std::vector<bool> const& valid = {})
{
  auto result = cudf::make_numeric_column(
    cudf::data_type{id}, values.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(result->mutable_view().template data<T>(),
                            values.data(),
                            values.size() * sizeof(T),
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
  }
  if (!valid.empty()) { set_validity(*result, valid, stream, mr); }
  // Input vectors can be temporaries. Finish their host-to-device transfers before returning.
  stream.sync();
  return result;
}

std::vector<int64_t> host(cudf::column_view input, ::cuda::stream_ref stream)
{
  std::vector<int64_t> values(input.size());
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(values.data(),
                            input.data<int64_t>(),
                            values.size() * sizeof(int64_t),
                            cudaMemcpyDeviceToHost,
                            stream.get()) == cudaSuccess);
  }
  stream.sync();
  return values;
}

struct direction {
  char const* sql;
  duckdb::OrderType type;
  duckdb::OrderByNullType nulls;
};

// 12 rows. Partition key k has NULLs, order key v has NULLs and ties, w breaks some of them.
constexpr char const* create_table =
  "CREATE TABLE w(id INTEGER, k INTEGER, q VARCHAR, v BIGINT, w INTEGER);"
  "INSERT INTO w VALUES (0,1,'a',5,1),(1,2,'b',3,1),(2,1,'a',5,2),(3,NULL,'a',7,2),"
  "(4,2,'b',NULL,1),(5,1,'a',2,1),(6,NULL,'a',7,2),(7,2,'b',3,2),(8,1,'a',NULL,1),"
  "(9,1,'a',9,1),(10,NULL,'a',1,2),(11,2,'b',3,1)";

std::unique_ptr<cudf::table> make_input(::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column<int32_t>({1, 2, 1, 0, 2, 1, 0, 2, 1, 1, 0, 2},
                                    cudf::type_id::INT32,
                                    stream,
                                    mr,
                                    {1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1}));
  columns.push_back(column<int64_t>({5, 3, 5, 7, 0, 2, 7, 3, 0, 9, 1, 3},
                                    cudf::type_id::INT64,
                                    stream,
                                    mr,
                                    {1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1}));
  columns.push_back(
    column<int32_t>({1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1}, cudf::type_id::INT32, stream, mr));
  columns.push_back(
    column<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, cudf::type_id::INT32, stream, mr));
  return std::make_unique<cudf::table>(std::move(columns));
}

void check_against_duckdb(duckdb::Connection& con,
                          std::string const& window,
                          cudf::table_view result,
                          std::vector<std::string> const& functions,
                          ::cuda::stream_ref stream)
{
  std::string select;
  for (auto const& function : functions) {
    select += (select.empty() ? "" : ",") + function + "() OVER (" + window + ")";
  }
  auto expected = con.Query("SELECT " + select + " FROM w ORDER BY id");
  REQUIRE_FALSE(expected->HasError());
  auto rows = expected->Fetch();
  REQUIRE(rows);
  REQUIRE(result.num_rows() == expected->RowCount());
  REQUIRE(result.num_columns() == static_cast<cudf::size_type>(functions.size()));
  for (cudf::size_type c = 0; c < result.num_columns(); ++c) {
    CHECK(result.column(c).type() == cudf::data_type{cudf::type_id::INT64});
    CHECK(result.column(c).null_count() == 0);
    auto values = host(result.column(c), stream);
    for (cudf::size_type row = 0; row < result.num_rows(); ++row) {
      INFO("function " << functions[c] << " row " << row);
      CHECK(values[row] == rows->GetValue(c, row).GetValue<int64_t>());
    }
  }
}
}  // namespace

TEST_CASE("ranking evaluator matches DuckDB for every direction and NULL order",
          "[window_evaluator][window_ranking][gpu]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  REQUIRE_FALSE(con.Query(create_table)->HasError());
  rmm::cuda_stream owning_stream;
  auto stream = owning_stream.view();
  auto mr     = cudf::get_current_device_resource_ref();
  auto input  = make_input(stream, mr);

  auto const order = GENERATE(
    direction{
      "ASC NULLS FIRST", duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_FIRST},
    direction{"ASC NULLS LAST", duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST},
    direction{
      "DESC NULLS FIRST", duckdb::OrderType::DESCENDING, duckdb::OrderByNullType::NULLS_FIRST},
    direction{
      "DESC NULLS LAST", duckdb::OrderType::DESCENDING, duckdb::OrderByNullType::NULLS_LAST});
  auto const sliced = GENERATE(false, true);
  INFO(order.sql << (sliced ? " sliced" : ""));
  auto view = sliced ? cudf::slice(input->view(), {2, 11}, stream).front() : input->view();
  ranking_order spec;
  spec.columns   = {1};
  spec.direction = sirius::op::to_cudf_order(order.type);
  spec.nulls     = sirius::op::to_cudf_null_order(order.type, order.nulls);
  auto result    = evaluate_ranking_window(
    view, {0}, spec, {ranking_function::rank, ranking_function::dense_rank}, stream, mr);

  // The sliced table is the rows with id 2 to 10 in DuckDB.
  std::string const rows   = sliced ? "(SELECT * FROM w WHERE id >= 2 AND id < 11) w" : "w";
  std::string const window = std::string("PARTITION BY k ORDER BY v ") + order.sql;
  std::string select       = "rank() OVER (" + window + "),dense_rank() OVER (" + window + ")";
  auto expected            = con.Query("SELECT " + select + " FROM " + rows + " ORDER BY id");
  REQUIRE_FALSE(expected->HasError());
  auto fetched = expected->Fetch();
  REQUIRE(fetched);
  REQUIRE(result->num_rows() == expected->RowCount());
  for (cudf::size_type c = 0; c < 2; ++c) {
    auto values = host(result->view().column(c), stream);
    for (cudf::size_type row = 0; row < result->num_rows(); ++row) {
      INFO("column " << c << " row " << row);
      CHECK(values[row] == fetched->GetValue(c, row).GetValue<int64_t>());
    }
  }
}

TEST_CASE("ranking evaluator ranks over several order keys and numbers rows",
          "[window_evaluator][window_ranking][gpu]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  REQUIRE_FALSE(con.Query(create_table)->HasError());
  rmm::cuda_stream owning_stream;
  auto stream = owning_stream.view();
  auto mr     = cudf::get_current_device_resource_ref();
  auto input  = make_input(stream, mr);

  SECTION("two order keys, one of them with NULLs")
  {
    ranking_order spec;
    spec.columns   = {1, 2};
    spec.direction = cudf::order::DESCENDING;
    spec.nulls     = sirius::op::to_cudf_null_order(duckdb::OrderType::DESCENDING,
                                                duckdb::OrderByNullType::NULLS_LAST);
    auto result    = evaluate_ranking_window(
      input->view(), {0}, spec, {ranking_function::rank, ranking_function::dense_rank}, stream, mr);
    check_against_duckdb(con,
                         "PARTITION BY k ORDER BY v DESC NULLS LAST, w DESC NULLS LAST",
                         result->view(),
                         {"rank", "dense_rank"},
                         stream);
  }
  SECTION("row_number follows the input order of peers")
  {
    // The id column breaks every tie, so DuckDB's numbering is determined.
    ranking_order spec;
    spec.columns   = {1, 3};
    spec.direction = cudf::order::ASCENDING;
    spec.nulls     = sirius::op::to_cudf_null_order(duckdb::OrderType::ASCENDING,
                                                duckdb::OrderByNullType::NULLS_FIRST);
    auto result =
      evaluate_ranking_window(input->view(), {0}, spec, {ranking_function::row_number}, stream, mr);
    check_against_duckdb(
      con, "PARTITION BY k ORDER BY v NULLS FIRST, id", result->view(), {"row_number"}, stream);

    // Peers keep their input order.
    spec.columns = {1};
    result =
      evaluate_ranking_window(input->view(), {0}, spec, {ranking_function::row_number}, stream, mr);
    auto values = host(result->view().column(0), stream);
    CHECK(values[0] < values[2]);
    CHECK(values[3] < values[6]);
    CHECK(values[1] < values[7]);
    CHECK(values[7] < values[11]);
  }
  SECTION("several functions share one sort")
  {
    ranking_order spec;
    spec.columns   = {1};
    spec.direction = cudf::order::DESCENDING;
    spec.nulls     = sirius::op::to_cudf_null_order(duckdb::OrderType::DESCENDING,
                                                duckdb::OrderByNullType::NULLS_FIRST);
    auto result    = evaluate_ranking_window(input->view(),
                                             {0},
                                          spec,
                                             {ranking_function::dense_rank,
                                              ranking_function::rank,
                                              ranking_function::rank,
                                              ranking_function::dense_rank},
                                          stream,
                                          mr);
    check_against_duckdb(con,
                         "PARTITION BY k ORDER BY v DESC NULLS FIRST",
                         result->view(),
                         {"dense_rank", "rank", "rank", "dense_rank"},
                         stream);
  }
}

TEST_CASE("ranking evaluator partitions by strings and several columns, and orders floats",
          "[window_evaluator][window_ranking][gpu]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  REQUIRE_FALSE(
    con
      .Query("CREATE TABLE w(id INTEGER, q VARCHAR, k INTEGER, f DOUBLE);"
             "INSERT INTO w VALUES (0,'x',1,1.5),(1,'y',1,'NaN'),(2,'x',1,1.5),(3,NULL,1,'inf'),"
             "(4,'x',2,'-inf'),(5,'y',1,'NaN'),(6,NULL,1,-0.0),(7,'x',1,0.0),(8,'y',1,2.5),"
             "(9,NULL,1,NULL),(10,'x',2,-1.0),(11,'x',2,'-inf')")
      ->HasError());
  rmm::cuda_stream owning_stream;
  auto stream    = owning_stream.view();
  auto mr        = cudf::get_current_device_resource_ref();
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  auto const inf = std::numeric_limits<double>::infinity();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    column<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, cudf::type_id::INT32, stream, mr));
  auto strings = sirius::test::operator_utils::make_string_column(
    {"x", "y", "x", "", "x", "y", "", "x", "y", "", "x", "x"}, stream, mr);
  set_validity(*strings, {1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1}, stream, mr);
  columns.push_back(std::move(strings));
  columns.push_back(
    column<int32_t>({1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2}, cudf::type_id::INT32, stream, mr));
  columns.push_back(column<double>({1.5, nan, 1.5, inf, -inf, nan, -0.0, 0.0, 2.5, 0, -1.0, -inf},
                                   cudf::type_id::FLOAT64,
                                   stream,
                                   mr,
                                   {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1}));
  cudf::table input(std::move(columns));
  ranking_order spec;
  spec.columns   = {3};
  spec.direction = cudf::order::ASCENDING;
  spec.nulls     = sirius::op::to_cudf_null_order(duckdb::OrderType::ASCENDING,
                                              duckdb::OrderByNullType::NULLS_FIRST);
  auto result    = evaluate_ranking_window(
    input.view(), {1, 2}, spec, {ranking_function::rank, ranking_function::dense_rank}, stream, mr);
  check_against_duckdb(con,
                       "PARTITION BY q, k ORDER BY f NULLS FIRST",
                       result->view(),
                       {"rank", "dense_rank"},
                       stream);
}

TEST_CASE("ranking evaluator handles empty input and rejects invalid requests",
          "[window_evaluator][window_ranking][gpu]")
{
  rmm::cuda_stream owning_stream;
  auto stream = owning_stream.view();
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column<int32_t>({}, cudf::type_id::INT32, stream, mr));
  columns.push_back(column<int64_t>({}, cudf::type_id::INT64, stream, mr));
  cudf::table empty(std::move(columns));
  ranking_order spec;
  spec.columns = {1};

  SECTION("empty input gives empty INT64 columns")
  {
    auto result = evaluate_ranking_window(
      empty.view(), {0}, spec, {ranking_function::rank, ranking_function::row_number}, stream, mr);
    REQUIRE(result->num_columns() == 2);
    CHECK(result->num_rows() == 0);
    CHECK(result->view().column(0).type() == cudf::data_type{cudf::type_id::INT64});
  }
  SECTION("invalid requests throw")
  {
    auto input = make_input(stream, mr);
    auto call  = [&](std::vector<cudf::size_type> const& partitions,
                    ranking_order const& order,
                    std::vector<ranking_function> const& functions) {
      return evaluate_ranking_window(input->view(), partitions, order, functions, stream, mr);
    };
    CHECK_THROWS_AS(call({0}, spec, {}), std::invalid_argument);
    CHECK_THROWS_AS(call({}, spec, {ranking_function::rank}), std::invalid_argument);
    CHECK_THROWS_AS(call({7}, spec, {ranking_function::rank}), std::invalid_argument);
    CHECK_THROWS_AS(call({0}, ranking_order{}, {ranking_function::rank}), std::invalid_argument);
    spec.columns = {-1};
    CHECK_THROWS_AS(call({0}, spec, {ranking_function::rank}), std::invalid_argument);
    spec.columns = {4};
    CHECK_THROWS_AS(call({0}, spec, {ranking_function::rank}), std::invalid_argument);
  }
}
