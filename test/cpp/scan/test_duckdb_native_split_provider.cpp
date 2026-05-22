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

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <op/scan/duckdb_native_scan_info.hpp>
#include <scan_manager/duckdb_native_split_provider.hpp>
#include <utils/utils.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;
using namespace sirius::scan_manager;

namespace {

void exec_ok(duckdb::Connection& con, std::string const& q)
{
  auto result = con.Query(q);
  REQUIRE(result);
  if (result->HasError()) {
    INFO("query failed: " << q << "\n  error: " << result->GetError());
    REQUIRE_FALSE(result->HasError());
  }
}

duckdb::DataTable& get_storage(duckdb::Connection& con, std::string const& table_name)
{
  exec_ok(con, "BEGIN TRANSACTION");
  auto& ctx     = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");
  auto entry   = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(entry);
  return entry->Cast<duckdb::DuckTableEntry>().GetStorage();
}

projected_column real_col(duckdb::idx_t col_id)
{
  projected_column pc;
  pc.storage_idx = duckdb::StorageIndex(col_id);
  pc.is_rowid    = false;
  return pc;
}

/// Storage + context only; caller fills in projected_cols / projected_types
/// / approximate_batch_size.
duckdb_native_scan_info make_scan_info(duckdb::DataTable& storage, duckdb::ClientContext& ctx)
{
  duckdb_native_scan_info info;
  info.storage = &storage;
  info.context = &ctx;
  return info;
}

/// Drain all splits and return the count.
std::size_t count_splits(duckdb_native_split_provider& provider)
{
  std::size_t n = 0;
  while (provider.has_more_splits()) {
    auto factory = provider.next_split_provider();
    if (!factory) break;
    auto payloads = factory();
    if (payloads.empty()) break;
    ++n;
  }
  return n;
}

}  // namespace

TEST_CASE("duckdb_native_split_provider throws on null storage",
          "[scan][duckdb_native_split_provider]")
{
  duckdb_native_scan_info info;
  // info.storage stays null.
  REQUIRE_THROWS_AS(duckdb_native_split_provider{std::move(info)}, std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider throws on null context",
          "[scan][duckdb_native_split_provider]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  duckdb_native_scan_info info;
  info.storage = &storage;
  // info.context stays null.
  REQUIRE_THROWS_AS(duckdb_native_split_provider{std::move(info)}, std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider throws on projected vectors size mismatch",
          "[scan][duckdb_native_split_provider]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto info            = make_scan_info(storage, *con.context);
  info.projected_cols  = {real_col(0)};
  info.projected_types = {};  // empty — parallel-vector violation
  REQUIRE_THROWS_AS(duckdb_native_split_provider{std::move(info)}, std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider throws when walker rejects the table",
          "[scan][duckdb_native_split_provider]")
{
  // HUGEINT is not supported by the gpu decoder → walker reports non-viable
  // → split_provider surfaces the rejection as a query error.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a HUGEINT)");
  exec_ok(con, "INSERT INTO t VALUES (1), (2), (3)");
  auto& storage = get_storage(con, "t");

  auto info            = make_scan_info(storage, *con.context);
  info.projected_cols  = {real_col(0)};
  info.projected_types = {sirius::logical_type::make(sirius::type_id::HUGEINT)};
  REQUIRE_THROWS_AS(duckdb_native_split_provider{std::move(info)}, std::runtime_error);
}

TEST_CASE("duckdb_native_split_provider emits one batch for a small INTEGER table",
          "[scan][duckdb_native_split_provider]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1024)");
  auto& storage = get_storage(con, "t");

  auto info                   = make_scan_info(storage, *con.context);
  info.projected_cols         = {real_col(0)};
  info.projected_types        = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info.approximate_batch_size = 0;  // single-batch fast path

  duckdb_native_split_provider provider{std::move(info)};
  REQUIRE(provider.has_more_splits());

  auto factory = provider.next_split_provider();
  REQUIRE(factory);
  auto payloads = factory();
  REQUIRE(payloads.size() == 1);

  auto* batch = dynamic_cast<duckdb_native_split_provider::split_payload*>(payloads[0].get());
  REQUIRE(batch != nullptr);
  REQUIRE(!batch->row_groups.empty());
  REQUIRE(batch->scan_info != nullptr);

  // After draining, both signals must report exhaustion: has_more_splits()
  // goes false and next_split_provider() returns an empty std::function.
  REQUIRE_FALSE(provider.has_more_splits());
  auto exhausted_factory = provider.next_split_provider();
  REQUIRE_FALSE(exhausted_factory);
}

TEST_CASE("duckdb_native_split_provider splits across batches when batch_size caps",
          "[scan][duckdb_native_split_provider]")
{
  // STANDARD_VECTOR_SIZE is 2048; a row group is up to 122,880 rows in DuckDB
  // (60 vectors). 800k INTEGER rows guarantees several row groups, and a low
  // batch cap forces multi-batch output.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 800000)");
  auto& storage = get_storage(con, "t");

  auto info                   = make_scan_info(storage, *con.context);
  info.projected_cols         = {real_col(0)};
  info.projected_types        = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info.approximate_batch_size = 256 * 1024;  // ~64k INTEGER rows per batch budget

  duckdb_native_split_provider provider{std::move(info)};
  std::size_t splits = count_splits(provider);
  REQUIRE(splits > 1);  // must have produced at least two batches
}

TEST_CASE("duckdb_native_split_provider keeps a single oversized row group as its own batch",
          "[scan][duckdb_native_split_provider]")
{
  // Set approximate_batch_size below a single row group's decoded byte cost.
  // The provider must still produce one singleton batch — not stall on an
  // unsatisfiable cap.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 200000)");
  auto& storage = get_storage(con, "t");

  auto info                   = make_scan_info(storage, *con.context);
  info.projected_cols         = {real_col(0)};
  info.projected_types        = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info.approximate_batch_size = 1;  // anything > 0 row groups exceeds

  duckdb_native_split_provider provider{std::move(info)};
  std::size_t splits = count_splits(provider);
  REQUIRE(splits >= 1);
}

TEST_CASE("duckdb_native_split_provider populates payload with scan_info shared_ptr",
          "[scan][duckdb_native_split_provider]")
{
  // The payload's scan_info must alias the same data even across multiple
  // batches so downstream tasks observe a consistent projection.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 400000)");
  auto& storage = get_storage(con, "t");

  auto info                   = make_scan_info(storage, *con.context);
  info.projected_cols         = {real_col(0)};
  info.projected_types        = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info.approximate_batch_size = 64 * 1024;

  duckdb_native_split_provider provider{std::move(info)};

  std::shared_ptr<op::scan::duckdb_native_scan_info const> first_scan_info;
  bool any = false;
  while (provider.has_more_splits()) {
    auto factory = provider.next_split_provider();
    if (!factory) break;
    auto payloads = factory();
    if (payloads.empty()) break;
    any         = true;
    auto* batch = dynamic_cast<duckdb_native_split_provider::split_payload*>(payloads[0].get());
    REQUIRE(batch != nullptr);
    REQUIRE(batch->scan_info != nullptr);
    if (!first_scan_info) {
      first_scan_info = batch->scan_info;
    } else {
      REQUIRE(batch->scan_info.get() == first_scan_info.get());
    }
  }
  REQUIRE(any);
}
