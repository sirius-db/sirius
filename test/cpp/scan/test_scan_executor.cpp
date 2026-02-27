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
#include <op/scan/duckdb_scan_executor.hpp>
#include <op/scan/duckdb_scan_task.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <pipeline/pipeline_executor.hpp>

// cucascade
#include <cucascade/data/data_repository.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/catalog/catalog_transaction.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parallel/thread_context.hpp>

// standard library
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

using namespace sirius;
using namespace sirius::scan_test_utils;
using namespace cucascade::memory;

/**
 * @brief Create a PhysicalTableScan for the given table
 */
static std::unique_ptr<sirius::op::sirius_physical_duckdb_scan> make_physical_table_scan(
  duckdb::ClientContext& ctx, std::string const& table_name)
{
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");

  auto table_entry = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(table_entry);

  auto& table_catalog_entry = table_entry->Cast<duckdb::TableCatalogEntry>();

  // Get all column IDs as ColumnIndex
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  for (size_t i = 0; i < table_catalog_entry.GetColumns().LogicalColumnCount(); ++i) {
    column_ids.push_back(duckdb::ColumnIndex(i));
    projection_ids.push_back(i);  // Map output column i to internal column i
  }

  // Create bind data
  auto bind_data = std::make_unique<duckdb::TableScanBindData>(table_catalog_entry);

  // Get the table scan function
  auto table_scan_function = duckdb::TableScanFunction::GetFunction();

  // Get column names
  duckdb::vector<std::string> column_names;
  for (size_t i = 0; i < table_catalog_entry.GetColumns().LogicalColumnCount(); ++i) {
    column_names.push_back(table_catalog_entry.GetColumn(duckdb::LogicalIndex(i)).GetName());
  }

  // Create extra operator info (must be a variable, not a temporary)
  duckdb::ExtraOperatorInfo extra_info;

  auto virtual_columns = table_catalog_entry.GetVirtualColumns();

  // Create sirius_physical_duckdb_scan with all required parameters
  auto physical_scan = std::make_unique<sirius::op::sirius_physical_duckdb_scan>(
    table_catalog_entry.GetTypes(),   // types
    table_scan_function,              // function
    std::move(bind_data),             // bind_data
    table_catalog_entry.GetTypes(),   // returned_types
    std::move(column_ids),            // column_ids
    std::move(projection_ids),        // projection_ids (maps output to internal columns)
    std::move(column_names),          // names
    nullptr,                          // table_filters
    0,                                // estimated_cardinality
    std::move(extra_info),            // extra_info
    duckdb::vector<duckdb::Value>(),  // parameters
    std::move(virtual_columns)        // virtual_columns
  );

  return physical_scan;
}

/**
 * @brief Helper to run a scan test with the given parameters
 *
 * This encapsulates the common test setup and execution logic.
 */
static void run_scan_test(std::string const& table_name,
                          size_t num_rows,
                          int num_threads,
                          size_t batch_size)
{
  // Setup DuckDB database
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  // Create and populate table
  create_synthetic_table(con, table_name, num_rows);

  // Get client context
  auto& client_ctx = *con.context;
  auto sirius_ctx  = sirius::get_sirius_context(con, get_test_config_path());

  // Verify memory manager is initialized
  auto& mem_mgr   = sirius_ctx->get_memory_manager();
  auto* mem_space = get_space(mem_mgr, Tier::HOST);
  REQUIRE(mem_space != nullptr);

  // Begin transaction for catalog access
  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  // Create physical table scan
  auto physical_scan = make_physical_table_scan(client_ctx, table_name);
  REQUIRE(physical_scan);

  // Get pipeline executor from sirius context (owns the scan executor)
  auto& pipeline_executor = sirius_ctx->get_pipeline_executor();
  auto& scan_executor     = pipeline_executor.get_scan_executor();

  // Create global state
  auto global_state = std::make_shared<op::scan::duckdb_scan_task_global_state>(
    nullptr, pipeline_executor, client_ctx, physical_scan.get());

  // Create data repository manager (empty, unused for this test)
  cucascade::shared_data_repository data_repo;

  // Create a single execution context for all scan tasks.
  auto thread_context = std::make_unique<duckdb::ThreadContext>(client_ctx);
  auto execution_context =
    std::make_unique<duckdb::ExecutionContext>(client_ctx, *thread_context, nullptr);

  // Run tasks
  pipeline_executor.start();
  for (int i = 0; i < scan_executor.get_num_threads(); ++i) {
    auto local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
      *global_state, *execution_context, batch_size);
    auto task = std::make_unique<op::scan::duckdb_scan_task>(
      static_cast<uint64_t>(i + 1), &data_repo, std::move(local_state), global_state);
    pipeline_executor.schedule(std::move(task));
  }
  while (!global_state->is_source_drained()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  pipeline_executor.stop();

  // Validate repository data batches
  auto batches = drain_data_repo(data_repo);
  validate_scanned_batches(batches, num_rows, mem_mgr);

  // End the transaction
  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  // Cleanup
  con.Query("DROP TABLE " + table_name);
}

//===----------------------------------------------------------------------===//
// Test: Single-threaded scan executor
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - single threaded small table", "[scan_executor][single_thread]")
{
  // Use 10MB batch size to ensure multiple 1MB blocks are allocated
  run_scan_test("test_small", 100, 1, 10000000);
}

TEST_CASE("scan_executor - single threaded with small batches", "[scan_executor][single_thread]")
{
  // Use a small batch size to force multiple batches
  // With 4 columns (INT + BIGINT + DOUBLE + VARCHAR(256)) = 4 + 8 + 8 + 256 = 276 bytes per row
  // So ~600000 bytes should fit about 2175 rows (1 vector)
  run_scan_test("test_medium", 10000, 1, 600000);
}

//===----------------------------------------------------------------------===//
// Test: Multi-threaded scan executor
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - multi threaded small table", "[scan_executor][multi_thread]")
{
  run_scan_test("test_mt_small", 1000, 4, 1000000);
}

TEST_CASE("scan_executor - multi threaded medium table", "[scan_executor][multi_thread]")
{
  run_scan_test("test_mt_medium", 100000, 4, 1000000);
}

TEST_CASE("scan_executor - multi threaded large table", "[scan_executor][multi_thread]")
{
  run_scan_test("test_mt_large", 500000, 8, 10000000);
}

//===----------------------------------------------------------------------===//
// Test: Edge cases
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - empty table", "[scan_executor][edge_case]")
{
  run_scan_test("test_empty", 0, 1, 1000000);
}

TEST_CASE("scan_executor - single row table", "[scan_executor][edge_case]")
{
  run_scan_test("test_single_row", 1, 1, 1000000);
}
