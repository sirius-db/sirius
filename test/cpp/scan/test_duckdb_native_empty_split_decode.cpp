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

// Empty row-group lists must preserve the projected zero-row schema, including
// ARRAY columns that require cuDF's LIST factory. The integration case also
// verifies that fully pruned plans remain viable on the GPU path.

#include "test_utils.hpp"

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <op/scan/duckdb_native_decoder.hpp>
#include <op/scan/duckdb_native_gpu_ingestible.hpp>
#include <op/scan/duckdb_native_metadata.hpp>
#include <unistd.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;

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

// File-backed database; the metadata walk reads checkpointed row groups.
struct empty_split_test_db {
  std::string path;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

  empty_split_test_db()
  {
    static int counter = 0;
    path               = "/tmp/sirius_empty_split_decode_test_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + ".db";
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
    db  = std::make_unique<duckdb::DuckDB>(path);
    con = std::make_unique<duckdb::Connection>(*db);
    con->Query("SET gpu_execution = false;");
  }

  ~empty_split_test_db()
  {
    con.reset();
    db.reset();
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
  }
};

// Requires an active transaction because catalog access needs one.
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

projected_column real_col(duckdb::idx_t col_id)
{
  projected_column pc;
  pc.storage_idx = duckdb::StorageIndex(col_id);
  pc.is_rowid    = false;
  return pc;
}

projected_column rowid_col()
{
  projected_column pc;
  pc.is_rowid = true;
  return pc;
}

sirius::logical_type array_of(sirius::type_id child, std::uint32_t size)
{
  return sirius::logical_type::make_array(sirius::logical_type::make(child), size);
}

}  // namespace

TEST_CASE("empty split with an ARRAY projection decodes to a 0-row LIST column",
          "[scan][duckdb_native_empty_split]")
{
  auto mem_mgr    = initialize_memory_manager();
  auto* gpu_space = sirius::scan_test_utils::get_space(*mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;

  // The empty branch returns before touching storage/context, so the
  // table_info may carry nullptrs there.
  duckdb_native_ingestible_table_info info;
  info.projected_cols  = {real_col(0)};
  info.projected_types = {array_of(sirius::type_id::INTEGER, 3)};

  std::unique_ptr<cudf::table> table;
  REQUIRE_NOTHROW(table = decode_duckdb_native_split(
                    {}, info, /*datasource=*/nullptr, *gpu_space, stream.view()));
  REQUIRE(table != nullptr);
  REQUIRE(table->num_columns() == 1);
  REQUIRE(table->num_rows() == 0);
  REQUIRE(table->get_column(0).type().id() == cudf::type_id::LIST);
  cudf::lists_column_view const lists(table->get_column(0).view());
  REQUIRE(lists.child().type().id() == cudf::type_id::INT32);
}

TEST_CASE("empty split preserves rowid, ARRAY, and scalar projection order",
          "[scan][duckdb_native_empty_split]")
{
  auto mem_mgr    = initialize_memory_manager();
  auto* gpu_space = sirius::scan_test_utils::get_space(*mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;

  duckdb_native_ingestible_table_info info;
  info.projected_cols  = {rowid_col(), real_col(0), real_col(1)};
  info.projected_types = {sirius::logical_type::make(sirius::type_id::BIGINT),
                          array_of(sirius::type_id::INTEGER, 3),
                          sirius::logical_type::make(sirius::type_id::INTEGER)};

  auto table =
    decode_duckdb_native_split({}, info, /*datasource=*/nullptr, *gpu_space, stream.view());
  REQUIRE(table->num_columns() == 3);
  REQUIRE(table->num_rows() == 0);
  REQUIRE(table->get_column(0).type().id() == cudf::type_id::INT64);
  REQUIRE(table->get_column(1).type().id() == cudf::type_id::LIST);
  cudf::lists_column_view const lists(table->get_column(1).view());
  REQUIRE(lists.child().type().id() == cudf::type_id::INT32);
  REQUIRE(table->get_column(2).type().id() == cudf::type_id::INT32);
}

TEST_CASE("all-pruned split stays viable and decodes to a 0-row table",
          "[scan][duckdb_native_empty_split]")
{
  empty_split_test_db env;
  exec_ok(*env.con, "CREATE TABLE t(id INTEGER, a INTEGER[3])");
  exec_ok(*env.con,
          "INSERT INTO t SELECT range, [range, range + 1, range + 2] FROM range(0, 3000)");
  exec_ok(*env.con, "CHECKPOINT");
  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");

  std::vector<projected_column> cols      = {real_col(0), real_col(1)};
  std::vector<sirius::logical_type> types = {sirius::logical_type::make(sirius::type_id::INTEGER),
                                             array_of(sirius::type_id::INTEGER, 3)};

  // id ranges over [0, 3000), so id >= 1,000,000 is FILTER_ALWAYS_FALSE for
  // every row group. Keyed by the relative scan-column index (0 = id) with the
  // parallel column_ids mapping it back to storage column 0.
  duckdb::TableFilterSet filters;
  filters.filters[0] = duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(1000000));
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.push_back(duckdb::ColumnIndex(0));

  auto plan =
    prepare_duckdb_native_walk(storage, *env.con->context, cols, types, &filters, &column_ids);
  REQUIRE(plan.viable);
  REQUIRE(plan.n_row_groups > 0);
  REQUIRE(plan.pruned_row_groups == plan.n_row_groups);

  auto range = walk_duckdb_native_row_group_range(plan, 0, plan.n_row_groups);
  REQUIRE(range.viable);
  REQUIRE(range.row_groups.empty());

  duckdb_native_ingestible_table_info info;
  info.storage         = &storage;
  info.context         = env.con->context.get();
  info.projected_cols  = cols;
  info.projected_types = types;

  auto mem_mgr    = initialize_memory_manager();
  auto* gpu_space = sirius::scan_test_utils::get_space(*mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;

  std::unique_ptr<cudf::table> table;
  REQUIRE_NOTHROW(table = decode_duckdb_native_split(
                    range.row_groups, info, /*datasource=*/nullptr, *gpu_space, stream.view()));
  REQUIRE(table != nullptr);
  REQUIRE(table->num_columns() == 2);
  REQUIRE(table->num_rows() == 0);
  REQUIRE(table->get_column(0).type().id() == cudf::type_id::INT32);
  REQUIRE(table->get_column(1).type().id() == cudf::type_id::LIST);
  cudf::lists_column_view const lists(table->get_column(1).view());
  REQUIRE(lists.child().type().id() == cudf::type_id::INT32);
}
