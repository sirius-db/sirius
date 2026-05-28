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

#include "helper/transaction_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/transaction/duck_transaction.hpp>

namespace {

duckdb::Catalog& default_catalog_for(duckdb::ClientContext& context)
{
  const auto& name = duckdb::DatabaseManager::Get(context).GetDefaultDatabase(context);
  return duckdb::Catalog::GetCatalog(context, name);
}

}  // namespace

TEST_CASE("get_query_start_time matches DuckTransaction::start_time on the default catalog",
          "[helper][transaction]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection conn(db);
  auto& context = *conn.context;

  conn.Query("BEGIN TRANSACTION");

  auto& catalog = default_catalog_for(context);
  auto expected = duckdb::DuckTransaction::Get(context, catalog).start_time;
  auto helper_v = sirius::helper::get_query_start_time(context);

  REQUIRE(helper_v == expected);

  conn.Query("COMMIT");
}

TEST_CASE("get_query_start_time is monotonic across committed transactions",
          "[helper][transaction]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection conn(db);
  auto& context = *conn.context;

  conn.Query("BEGIN TRANSACTION");
  // Touch the catalog so DuckDB bumps its transaction-id counter on commit,
  // otherwise read-only transactions may share a start_time with the next one.
  conn.Query("CREATE TABLE t (x INTEGER)");
  auto first = sirius::helper::get_query_start_time(context);
  conn.Query("COMMIT");

  conn.Query("BEGIN TRANSACTION");
  conn.Query("INSERT INTO t VALUES (1)");
  auto second = sirius::helper::get_query_start_time(context);
  conn.Query("COMMIT");

  REQUIRE(second >= first);
}
