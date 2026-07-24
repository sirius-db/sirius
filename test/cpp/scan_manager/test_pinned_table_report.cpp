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

// Unit tests for make_pinned_table_report: field derivation from a pinned
// entry, mvcc-vs-parquet NULL rules, and best-effort live-table probes against
// a real duckdb table. The sirius_pinned_tables() SQL surface end-to-end lives
// in test_pin_table_observability.cpp.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/transaction/duck_transaction.hpp>
#include <scan_manager/duckdb_mvcc_metadata.hpp>
#include <scan_manager/pinned_table_report.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <unistd.h>

#include <memory>
#include <string>
#include <vector>

using sirius::scan_manager::make_pinned_table_report;
using sirius::scan_manager::pinned_entry;

namespace {

void exec_ok(duckdb::Connection& con, const std::string& q)
{
  auto result = con.Query(q);
  REQUIRE(result);
  if (result->HasError()) {
    INFO("query failed: " << q << "\n  error: " << result->GetError());
    REQUIRE_FALSE(result->HasError());
  }
}

struct report_test_db {
  std::string path;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  report_test_db()
  {
    static int counter = 0;
    path               = "/tmp/sirius_pinned_report_test_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + ".db";
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
    db  = std::make_unique<duckdb::DuckDB>(path);
    con = std::make_unique<duckdb::Connection>(*db);
    con->Query("SET gpu_execution = false;");
  }
  ~report_test_db()
  {
    con.reset();
    db.reset();
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
  }
};

duckdb::DataTable& resolve_storage(duckdb::Connection& con, const std::string& table_name)
{
  auto& ctx     = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");
  auto entry   = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(entry);
  return entry->Cast<duckdb::DuckTableEntry>().GetStorage();
}

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager>& mem_mgr()
{
  static auto mgr = sirius::test::operator_utils::initialize_memory_manager();
  return mgr;
}

std::shared_ptr<cudf::column> make_i32_chunk(std::size_t n, cucascade::memory::memory_space& space)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(n),
                                       cudf::mask_state::UNALLOCATED,
                                       cudf::get_default_stream(),
                                       space.get_default_allocator());
  return std::shared_ptr<cudf::column>(std::move(col));
}

/// GPU entry over one INT32 column "k", one chunk of @p base_rows rows, keyed
/// to the search-path table @p table_name (empty catalog => search path).
pinned_entry make_duckdb_entry(std::string const& table_name,
                               std::size_t base_rows,
                               duckdb::transaction_t v_base,
                               cucascade::memory::memory_space& space)
{
  pinned_entry entry;
  entry.tier                   = cucascade::memory::Tier::GPU;
  entry.memory_space           = &space;
  entry.cache_info.schema_name = "main";
  entry.cache_info.table_name  = table_name;
  entry.cache_info.column_ids.push_back(duckdb::ColumnIndex(0));
  entry.cache_info.names = {"k"};
  entry.data_batches_by_column["k"].push_back(make_i32_chunk(base_rows, space));
  entry.chunk_memory_spaces.push_back(&space);
  entry.num_rows     = base_rows;
  entry.mvcc         = std::make_unique<sirius::scan_manager::duckdb_mvcc_metadata>();
  entry.mvcc->v_base = v_base;
  entry.mvcc->base_row_count_per_chunk = {base_rows};
  return entry;
}

}  // namespace

TEST_CASE("pinned table report: a parquet entry has no mvcc or live-table fields",
          "[pinned_table_report][scan_manager]")
{
  report_test_db env;
  auto* space = mem_mgr()->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  pinned_entry entry;
  entry.tier                           = cucascade::memory::Tier::GPU;
  entry.memory_space                   = space;
  entry.cache_info.resolved_file_paths = {"/data/a.parquet"};
  entry.cache_info.column_ids.push_back(duckdb::ColumnIndex(0));
  entry.cache_info.column_ids.push_back(duckdb::ColumnIndex(1));
  entry.cache_info.names = {"a", "b"};
  entry.data_batches_by_column["a"].push_back(make_i32_chunk(100, *space));
  entry.data_batches_by_column["b"].push_back(make_i32_chunk(100, *space));
  entry.chunk_memory_spaces.push_back(space);
  entry.num_rows = 100;

  auto report = make_pinned_table_report("p", entry, *env.con->context);
  REQUIRE(report.format == "parquet");
  REQUIRE(report.tier == "gpu");
  REQUIRE(report.column_count == 2);
  REQUIRE(report.chunk_count == 1);
  REQUIRE(report.base_rows == 100);
  REQUIRE(report.is_valid);
  REQUIRE_FALSE(report.v_base.has_value());
  REQUIRE_FALSE(report.promoted_chunks.has_value());
  REQUIRE_FALSE(report.delta_insert_rows.has_value());
}

TEST_CASE("pinned table report: promotion status reflects a sticky disable",
          "[pinned_table_report][scan_manager]")
{
  report_test_db env;
  auto* space = mem_mgr()->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  // No matching table in the catalog: the live probe fails gracefully and the
  // live fields stay absent, but the mvcc-resident fields still populate.
  auto entry = make_duckdb_entry("does_not_exist", 100, /*v_base=*/1, *space);
  entry.mvcc->promotion.promoted_chunks = 3;
  entry.mvcc->promotion.promoted_rows   = 300;
  entry.mvcc->promotion.disabled_reason = "delta-starts-mid-row-group";

  auto report = make_pinned_table_report("t", entry, *env.con->context);
  REQUIRE(report.format == "duckdb");
  REQUIRE(report.v_base == duckdb::transaction_t{1});
  REQUIRE(report.promoted_chunks == 3);
  REQUIRE(report.promoted_rows == 300);
  REQUIRE(report.promotion_status.has_value());
  REQUIRE(report.promotion_status->find("delta-starts-mid-row-group") != std::string::npos);
  REQUIRE_FALSE(report.delta_insert_rows.has_value());  // table absent -> live fields NULL
}

TEST_CASE("pinned table report: live delta counts match the table's state",
          "[pinned_table_report][scan_manager]")
{
  report_test_db env;
  auto* space = mem_mgr()->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(300000)");
  exec_ok(*env.con, "CHECKPOINT");

  // v_base captured before the mutations, so the report's query snapshot is newer.
  exec_ok(*env.con, "BEGIN TRANSACTION");
  duckdb::transaction_t v_base =
    duckdb::DuckTransaction::Get(*env.con->context, resolve_storage(*env.con, "t").GetAttached())
      .start_time;
  exec_ok(*env.con, "ROLLBACK");

  exec_ok(*env.con, "DELETE FROM t WHERE k IN (0, 5, 299999)");
  exec_ok(*env.con, "INSERT INTO t SELECT (300000+range)::INTEGER FROM range(1000)");

  // Empty catalog name => the live probe resolves 't' via the search path.
  auto entry = make_duckdb_entry("t", /*base_rows=*/300000, v_base, *space);

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto report = make_pinned_table_report("t", entry, *env.con->context);
  REQUIRE(report.base_rows == 300000);
  REQUIRE(report.is_valid);
  REQUIRE(report.delta_insert_rows == 1000);
  REQUIRE(report.delta_delete_rows == 3);
  REQUIRE(report.dirty_chunks == 1);
  exec_ok(*env.con, "ROLLBACK");
}
