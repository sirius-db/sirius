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

#include "catch.hpp"
#include "duckdb.hpp"
#include "pin_table.hpp"

using namespace duckdb;

namespace {

void RunPinTableQuery(const std::string& query)
{
  DuckDB db(nullptr);
  Connection con(db);
  auto result = con.Query(query);
  REQUIRE(!result->HasError());
}

}  // namespace

TEST_CASE("pin_table forwards required parameters", "[pin_table]")
{
  pin_table_testing::clear_recorded_calls();

  RunPinTableQuery("CALL pin_table('address_to_file/*.parquet', tier='host', name='lineitem')");

  const auto& calls = pin_table_testing::recorded_calls();
  REQUIRE(calls.size() == 1);
  REQUIRE(calls[0].path == "address_to_file/*.parquet");
  REQUIRE(calls[0].tier == "host");
  REQUIRE(calls[0].name == "lineitem");
  REQUIRE_FALSE(calls[0].cols.has_value());
  REQUIRE_FALSE(calls[0].n_rows.has_value());
}

TEST_CASE("pin_table forwards optional cols and n_rows", "[pin_table]")
{
  pin_table_testing::clear_recorded_calls();

  RunPinTableQuery(
    "CALL pin_table('data/*.parquet', tier='host', name='lineitem', "
    "cols=['l_orderkey', 'l_partkey'], n_rows=1000)");

  const auto& calls = pin_table_testing::recorded_calls();
  REQUIRE(calls.size() == 1);
  REQUIRE(calls[0].path == "data/*.parquet");
  REQUIRE(calls[0].tier == "host");
  REQUIRE(calls[0].name == "lineitem");
  REQUIRE(calls[0].cols.has_value());
  REQUIRE(calls[0].cols->size() == 2);
  REQUIRE((*calls[0].cols)[0] == "l_orderkey");
  REQUIRE((*calls[0].cols)[1] == "l_partkey");
  REQUIRE(calls[0].n_rows.has_value());
  REQUIRE(*calls[0].n_rows == 1000);
}

TEST_CASE("pin_table requires tier and name", "[pin_table]")
{
  pin_table_testing::clear_recorded_calls();

  DuckDB db(nullptr);
  Connection con(db);

  auto missing_tier = con.Query("CALL pin_table('data/*.parquet', name='lineitem')");
  REQUIRE(missing_tier->HasError());

  auto missing_name = con.Query("CALL pin_table('data/*.parquet', tier='host')");
  REQUIRE(missing_name->HasError());

  REQUIRE(pin_table_testing::recorded_calls().empty());
}
