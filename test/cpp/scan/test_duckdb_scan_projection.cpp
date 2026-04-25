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

// test
#include <catch.hpp>
#include <scan/test_utils.hpp>
#include <utils/utils.hpp>

// sirius
#include <helper/type_conversions.hpp>
#include <op/scan/duckdb_scan_task.hpp>

// duckdb
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/catalog/catalog_transaction.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/parallel/thread_context.hpp>

// standard library
#include <memory>
#include <string>
#include <vector>

using namespace sirius;
using namespace sirius::scan_test_utils;

namespace {

// Variant of make_physical_table_scan that takes explicit column_ids and projection_ids
// so a test can construct a non-identity projection (the case `estimate_rows_per_batch`
// has to handle when DuckDB pushes down a filter on a column the output doesn't include).
std::unique_ptr<sirius::op::sirius_physical_duckdb_scan> make_projecting_scan(
  duckdb::ClientContext& ctx,
  std::string const& table_name,
  std::vector<size_t> const& column_id_indices,
  std::vector<size_t> const& projection_id_indices)
{
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");

  auto table_entry = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(table_entry);
  auto& table_catalog_entry = table_entry->Cast<duckdb::TableCatalogEntry>();

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<std::string> column_names;
  for (auto idx : column_id_indices) {
    column_ids.emplace_back(idx);
    column_names.push_back(table_catalog_entry.GetColumn(duckdb::LogicalIndex(idx)).GetName());
  }
  for (auto pid : projection_id_indices) {
    projection_ids.push_back(pid);
  }

  auto bind_data           = std::make_unique<duckdb::TableScanBindData>(table_catalog_entry);
  auto table_scan_function = duckdb::TableScanFunction::GetFunction();
  duckdb::ExtraOperatorInfo extra_info;
  auto virtual_columns = table_catalog_entry.GetVirtualColumns();

  // scanned_types is built by the constructor from column_ids / projection_ids.
  return std::make_unique<sirius::op::sirius_physical_duckdb_scan>(
    sirius::from_duckdb_vec(table_catalog_entry.GetTypes()),
    table_scan_function,
    std::move(bind_data),
    sirius::from_duckdb_vec(table_catalog_entry.GetTypes()),
    std::move(column_ids),
    std::move(projection_ids),
    std::move(column_names),
    nullptr,
    0,
    std::move(extra_info),
    duckdb::vector<duckdb::Value>(),
    std::move(virtual_columns));
}

}  // namespace

// Regression: estimate_rows_per_batch must resolve the storage column through
// projection_ids when projection_ids is non-empty. Pre-fix, the code used column_ids[i]
// directly — for SELECT c1 FROM t WHERE c0 LIKE '%' (column_ids=[0,1], projection_ids=[1]),
// it read stats for c0 instead of c1. If c0 had a smaller max_string_length than c1, the
// resulting allocation was too small for c1's actual values; with has_metadata_bound=true,
// chunk_fits was skipped and process_column wrote past the buffer — silent corruption.
//
// This test sets c0=max_len=1 and c1=max_len=200, projects only c1, and asserts that the
// single output column_builder was sized using c1's metadata (not c0's).
TEST_CASE("duckdb_scan_task - projection_ids picks the projected storage column for stats",
          "[duckdb_scan_task][varchar][projection][shared_context]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();

  // Create a 2-column varchar table with sharply different max_string_length values.
  REQUIRE_FALSE(con.Query("DROP TABLE IF EXISTS t_proj_regress")->HasError());
  REQUIRE_FALSE(con.Query("CREATE TABLE t_proj_regress(c0 VARCHAR, c1 VARCHAR)")->HasError());
  REQUIRE_FALSE(
    con.Query("INSERT INTO t_proj_regress SELECT 'x', repeat('y', 200) FROM range(100)")
      ->HasError());
  REQUIRE_FALSE(con.Query("CHECKPOINT")->HasError());  // persist stats

  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());
  REQUIRE(sirius_ctx != nullptr);

  REQUIRE_FALSE(con.Query("BEGIN TRANSACTION")->HasError());

  // column_ids=[0,1] (we read both — for instance, the optimizer needed both to push a
  // filter on c0); projection_ids=[1] (we only output c1).
  auto physical_scan = make_projecting_scan(client_ctx,
                                            "t_proj_regress",
                                            /*column_ids=*/{0, 1},
                                            /*projection_ids=*/{1});
  REQUIRE(physical_scan != nullptr);
  REQUIRE(physical_scan->scanned_types.size() == 1);  // sanity: 1 output column
  REQUIRE(physical_scan->scanned_types[0].is_varchar());

  auto& task_sched  = sirius_ctx->get_task_scheduler();
  auto global_state = std::make_shared<op::scan::duckdb_scan_task_global_state>(
    nullptr, task_sched, client_ctx, physical_scan.get());

  auto thread_context = std::make_unique<duckdb::ThreadContext>(client_ctx);
  auto execution_context =
    std::make_unique<duckdb::ExecutionContext>(client_ctx, *thread_context, nullptr);

  // Constructor invokes estimate_rows_per_batch + initialize_builders — that's the
  // path where the bug lived.
  auto local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
    *global_state, *execution_context, /*approximate_batch_size=*/1'000'000);

  auto const& builders = local_state->column_builders_for_testing();
  REQUIRE(builders.size() == 1);
  REQUIRE(builders[0].type.is_varchar());
  REQUIRE(builders[0].has_metadata_bound);

  // The decisive check: type_size must reflect c1's max_string_length (200), not c0's
  // (1). Pre-fix would give 1; post-fix gives at least 200.
  CHECK(builders[0].type_size >= 200);

  REQUIRE_FALSE(con.Query("COMMIT")->HasError());
  con.Query("DROP TABLE t_proj_regress");
}
