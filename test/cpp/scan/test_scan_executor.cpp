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

// catch2
#include <catch.hpp>

// test utilities
#include "test_utils.hpp"

// sirius
#include <data/data_batch.hpp>
#include <data/data_repository.hpp>
#include <operator/gpu_physical_table_scan.hpp>
#include <scan/duckdb_scan_executor.hpp>
#include <scan/duckdb_scan_task.hpp>

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
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

using idx_t = duckdb::idx_t;
using namespace sirius;

//===----------------------------------------------------------------------===//
// Test Scan Task - Custom task that appends column_builder data to table
//===----------------------------------------------------------------------===//

/**
 * @brief Test version of duckdb_scan_task that appends scanned data to a DuckDB table
 *
 * This task executes the full scan pipeline (get_next_chunk -> process_chunk)
 * and then reads data from the column_builders to append to a staging table. It is essentially a
 * replica of duckdb_scan_task that adds an append() hook at the end for validation.
 */
class test_scan_task : public op::scan::duckdb_scan_task {
 public:
  test_scan_task(uint64_t task_id,
                 cucascade::shared_data_repository* data_repo,
                 duckdb::Connection& con,
                 std::string const& table_name,
                 std::unique_ptr<op::scan::duckdb_scan_task_local_state> l_state,
                 std::shared_ptr<op::scan::duckdb_scan_task_global_state> g_state)
    : duckdb_scan_task(task_id, data_repo, std::move(l_state), g_state),
      con_(con),
      table_name_(table_name)
  {
  }

  void execute() override
  {
    auto& l_state = this->_local_state->cast<op::scan::duckdb_scan_task_local_state>();
    auto& g_state = this->_global_state->cast<op::scan::duckdb_scan_task_global_state>();

    // Initialize the data chunk
    l_state.chunk.Initialize(duckdb::Allocator::Get(l_state.exec_ctx.client),
                             g_state.op.returned_types);

    // Scan loop - process chunks into column builders
    while (get_next_chunk(l_state, g_state)) {
      if (!chunk_fits(l_state)) {
        throw std::runtime_error("Chunk does not fit in allocated buffers");
      }

      // Process the chunk into column builders
      process_chunk(l_state);

      // Termination condition
      if (STANDARD_VECTOR_SIZE + l_state.row_offset >= l_state.estimated_rows_per_batch) { break; }
    }

    // Add tasks back to the queue if the scan is not finished
    if (!g_state.is_source_drained()) {
      auto const new_task_id = this->task_id + g_state.max_threads;

      // Create a new local state, passing the existing local_tf_state to continue the scan
      // This ensures DuckDB continues scanning from the current position rather than starting over
      auto new_local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
        g_state,
        l_state.exec_ctx,
        l_state.approximate_batch_size,
        l_state.default_varchar_size,
        std::move(l_state.local_tf_state));  // Pass the moved state to constructor

      // Create a new reference to the global state
      auto shared_global_state =
        std::static_pointer_cast<op::scan::duckdb_scan_task_global_state>(this->_global_state);
      auto next_task = std::make_unique<test_scan_task>(new_task_id,
                                                        _data_repo,
                                                        con_,
                                                        table_name_,
                                                        std::move(new_local_state),
                                                        shared_global_state);
      g_state.scan_executor.schedule(std::move(next_task));
    }

    // Append data from column_builders to staging table
    // NOTE: this makes the scan slow due to mutex and row-wise appends
    append_to_table(l_state);
  }

 private:
  /**
   * @brief Helper to check if a bit in a validity mask is set (1 = valid, 0 = invalid)
   */
  static inline bool is_valid(uint8_t current_mask, idx_t row_idx)
  {
    auto const bit_idx = row_idx % 8;
    return (current_mask & (1 << bit_idx)) != 0;
  }

  /**
   * @brief Append data from column_builders to the staging table
   *
   * Reads data directly from the column_builder buffers (data_blocks_accessor,
   * mask_blocks_accessor, offset_blocks_accessor) and appends to DuckDB table.
   *
   * NOTE: Uses a mutex to protect DuckDB connection access since DuckDB connections
   * are not thread-safe for concurrent writes.
   */
  void append_to_table(op::scan::duckdb_scan_task_local_state& l_state)
  {
    auto const num_rows = l_state.row_offset;
    if (num_rows == 0) {
      return;  // Nothing to append
    }

    // Lock the mutex to protect DuckDB connection access
    // DuckDB connections are not thread-safe for concurrent writes
    std::lock_guard<std::mutex> lock(append_mutex_);

    duckdb::Appender app(con_, table_name_);
    auto& column_builders = l_state.column_builders;
    auto& allocation      = l_state.allocation;

    // Ensure allocation exists
    if (!allocation) { throw std::runtime_error("Allocation is null in append_to_table"); }

    // First, reset the cursors of all column builders to their initial positions
    for (auto& builder : column_builders) {
      // Reset accessors to their starting byte offsets in the packed allocation
      builder.data_blocks_accessor.reset_cursor();
      builder.mask_blocks_accessor.reset_cursor();

      // Only reset offset accessor for VARCHAR columns (it's only initialized for VARCHAR)
      if (builder.type.InternalType() == duckdb::PhysicalType::VARCHAR) {
        builder.offset_blocks_accessor.reset_cursor();
      }
    }

    for (size_t i = 0; i < num_rows; ++i) {
      app.BeginRow();

      for (auto& builder : column_builders) {
        auto const& type = builder.type;

        // Check validity - advance mask accessor every 8 rows
        if (i > 0 && i % 8 == 0) { builder.mask_blocks_accessor.advance(); }
        bool valid = is_valid(builder.mask_blocks_accessor.get_current(allocation), i);

        if (!valid) {
          app.Append(duckdb::Value());  // NULL value
          continue;
        }

        // Type switch
        switch (type.id()) {
          case duckdb::LogicalTypeId::CHAR:  // Fallthrough
          case duckdb::LogicalTypeId::VARCHAR: {
            auto const beg = builder.offset_blocks_accessor.get_current(allocation);
            builder.offset_blocks_accessor.advance();
            auto const end = builder.offset_blocks_accessor.get_current(allocation);
            auto const len = end - beg;
            // We need to copy the string data from the multiple blocks allocation to a contiguous
            // buffer.
            std::string str(len, '\0');
            builder.data_blocks_accessor.memcpy_to(allocation, str.data(), len);
            app.Append<duckdb::string_t>(str);
            break;
          }
          case duckdb::LogicalTypeId::INTEGER: {
            auto const int_val = builder.data_blocks_accessor.get_current_as<int32_t>(allocation);
            app.Append<int32_t>(int_val);
            builder.data_blocks_accessor.advance_as<int32_t>();
            break;
          }
          case duckdb::LogicalTypeId::BIGINT: {
            auto const bigint_val =
              builder.data_blocks_accessor.get_current_as<int64_t>(allocation);
            app.Append<int64_t>(bigint_val);
            builder.data_blocks_accessor.advance_as<int64_t>();
            break;
          }
          case duckdb::LogicalTypeId::DOUBLE: {
            auto const double_val = builder.data_blocks_accessor.get_current_as<double>(allocation);
            app.Append<double>(double_val);
            builder.data_blocks_accessor.advance_as<double>();
            break;
          }
          case duckdb::LogicalTypeId::FLOAT: {
            auto const float_val = builder.data_blocks_accessor.get_current_as<float>(allocation);
            app.Append<float>(float_val);
            builder.data_blocks_accessor.advance_as<float>();
            break;
          }
          case duckdb::LogicalTypeId::DECIMAL: {
            auto width = duckdb::DecimalType::GetWidth(type);
            auto scale = duckdb::DecimalType::GetScale(type);

            switch (type.InternalType()) {
              case duckdb::PhysicalType::INT16: {
                auto const dec_val =
                  builder.data_blocks_accessor.get_current_as<int16_t>(allocation);
                app.Append(duckdb::Value::DECIMAL(dec_val, width, scale));
                builder.data_blocks_accessor.advance_as<int16_t>();
                break;
              }
              case duckdb::PhysicalType::INT32: {
                auto const dec_val =
                  builder.data_blocks_accessor.get_current_as<int32_t>(allocation);
                app.Append(duckdb::Value::DECIMAL(dec_val, width, scale));
                builder.data_blocks_accessor.advance_as<int32_t>();
                break;
              }
              case duckdb::PhysicalType::INT64: {
                auto const dec_val =
                  builder.data_blocks_accessor.get_current_as<int64_t>(allocation);
                app.Append(duckdb::Value::DECIMAL(dec_val, width, scale));
                builder.data_blocks_accessor.advance_as<int64_t>();
                break;
              }
              case duckdb::PhysicalType::INT128: {
                auto const dec_val =
                  builder.data_blocks_accessor.get_current_as<duckdb::hugeint_t>(allocation);
                app.Append(duckdb::Value::DECIMAL(dec_val, width, scale));
                builder.data_blocks_accessor.advance_as<duckdb::hugeint_t>();
                break;
              }
              default: FAIL("Unsupported decimal internal type");
            }
            break;
          }
          case duckdb::LogicalTypeId::DATE: {
            auto const date_val =
              builder.data_blocks_accessor.get_current_as<duckdb::date_t>(allocation);
            app.Append<duckdb::date_t>(date_val);
            builder.data_blocks_accessor.advance_as<duckdb::date_t>();
            break;
          }
          default: FAIL("Type not handled in test scan task appender");
        }
      }

      app.EndRow();
    }

    app.Close();
  }

  duckdb::Connection& con_;
  std::string table_name_;

  // Static mutex to protect DuckDB connection access across all task instances
  // DuckDB connections are not thread-safe for concurrent writes
  static std::mutex append_mutex_;
};

// Define the static mutex
std::mutex test_scan_task::append_mutex_;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/**
 * @brief Create a simple synthetic table with multiple columns and rows
 */
static void create_synthetic_table(duckdb::Connection& con,
                                   std::string const& table_name,
                                   size_t num_rows)
{
  // Create table with INTEGER, BIGINT, DOUBLE, and VARCHAR columns
  // clang-format off
  std::string create_sql = "CREATE TABLE " + table_name + " \
                            ( \
                              id INTEGER, \
                              value BIGINT, \
                              price DOUBLE, \
                              name VARCHAR \
                            );";
  // clang-format on
  auto result = con.Query(create_sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());

  // Insert data in batches
  constexpr size_t BATCH_SIZE = 1000;
  for (size_t start = 0; start < num_rows; start += BATCH_SIZE) {
    size_t end             = std::min(start + BATCH_SIZE, num_rows);
    std::string insert_sql = "INSERT INTO " + table_name + " VALUES ";

    for (size_t i = start; i < end; ++i) {
      if (i > start) { insert_sql += ", "; }
      // Generate predictable test data
      auto id          = static_cast<int32_t>(i);
      auto value       = static_cast<int64_t>(i * 100);
      auto price       = static_cast<double>(i) * 1.5;
      std::string name = "item_" + std::to_string(i);

      insert_sql += "(" + std::to_string(id) + ", " + std::to_string(value) + ", " +
                    std::to_string(price) + ", " + "'" + name + "')";
    }

    result = con.Query(insert_sql);
    REQUIRE(result);
    REQUIRE(!result->HasError());
  }
}

/**
 * @brief Validate that two tables have identical content
 */
static void validate_tables_equal(duckdb::Connection& con,
                                  std::string const& ref_table,
                                  std::string const& stage_table)
{
  auto cnt_ref = con.Query("SELECT COUNT(*) FROM " + ref_table + ";");
  auto cnt_stg = con.Query("SELECT COUNT(*) FROM " + stage_table + ";");
  REQUIRE(cnt_ref);
  REQUIRE(!cnt_ref->HasError());
  REQUIRE(cnt_stg);
  REQUIRE(!cnt_stg->HasError());

  auto ref_n = cnt_ref->GetValue<int64_t>(0, 0);
  auto stg_n = cnt_stg->GetValue<int64_t>(0, 0);
  REQUIRE(ref_n == stg_n);

  // Differences present in ref but not in stage
  // clang-format off
  auto missing = con.Query(
    "SELECT COUNT(*) \
       FROM ( SELECT * \
                FROM " + ref_table + " \
                  EXCEPT ALL SELECT * \
                               FROM " + stage_table + ");");
  // clang-format on
  REQUIRE(missing);
  REQUIRE(!missing->HasError());
  auto missing_n = missing->GetValue<int64_t>(0, 0);

  // Differences present in stage but not in ref
  // clang-format off
  auto extra = con.Query(
    "SELECT COUNT(*) \
       FROM ( SELECT * \
                FROM " + stage_table + " \
                  EXCEPT ALL SELECT * \
                               FROM " + ref_table + ");");
  // clang-format on
  REQUIRE(extra);
  REQUIRE(!extra->HasError());
  auto extra_n = extra->GetValue<int64_t>(0, 0);

  if (missing_n != 0 || extra_n != 0) {
    // Dump a few rows to help debugging
    // clang-format off
    auto diff1 = con.Query("SELECT * \
                              FROM " + ref_table + " \
                                EXCEPT ALL SELECT * \
                                             FROM " + stage_table + " \
                                             LIMIT 10;");
    auto diff2 = con.Query("SELECT * \
                              FROM " + stage_table + " \
                                EXCEPT ALL SELECT * \
                                             FROM " + ref_table + " \
                                             LIMIT 10;");
    // clang-format on
    std::cout << "REFERENCE TABLE: " + ref_table << "\n";
    std::cout << "MISSING:\n";
    diff1->Print();
    std::cout << "EXTRA:\n";
    diff2->Print();
  }
  REQUIRE(missing_n == 0);
  REQUIRE(extra_n == 0);
}

/**
 * @brief Create a PhysicalTableScan for the given table
 */
static std::unique_ptr<duckdb::GPUPhysicalTableScan> make_physical_table_scan(
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

  // Create GPUPhysicalTableScan with all required parameters
  auto physical_scan = std::make_unique<duckdb::GPUPhysicalTableScan>(
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
    duckdb::virtual_column_map_t());

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
                          size_t batch_size,
                          uint64_t pipeline_id)
{
  // Initialize memory manager for tests
  initialize_memory_manager();

  // Verify memory manager is initialized
  auto& mem_mgr   = sirius::memory_manager::get();
  auto* mem_space = mem_mgr.get_memory_space(Tier::HOST, 0);
  REQUIRE(mem_space != nullptr);

  // Setup DuckDB database
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  // Create and populate table
  create_synthetic_table(con, table_name, num_rows);

  // Get client context
  auto& client_ctx = *con.context;

  // Begin transaction for catalog access
  auto begin_result = con.Query("BEGIN TRANSACTION");
  REQUIRE(begin_result);
  REQUIRE(!begin_result->HasError());

  // Create physical table scan
  auto physical_scan = make_physical_table_scan(client_ctx, table_name);
  REQUIRE(physical_scan);

  // Create staging table for scanned data
  std::string staging_table = table_name + "_scanned";
  auto create_result =
    con.Query("CREATE TABLE " + staging_table + " AS SELECT * FROM " + table_name + " WHERE 1=0");
  REQUIRE(create_result);
  REQUIRE(!create_result->HasError());

  // Create scan executor (task scheduler)
  op::scan::duckdb_scan_executor scan_executor({num_threads, false});

  // Create execution context using dummy query
  auto dummy_query = "SELECT * FROM " + table_name + " LIMIT 0";
  auto prepared    = con.Prepare(dummy_query);
  REQUIRE(prepared);
  REQUIRE(!prepared->HasError());
  auto dummy_result = prepared->Execute();
  REQUIRE(dummy_result);
  REQUIRE(!dummy_result->HasError());

  duckdb::ThreadContext thread_ctx(client_ctx);
  duckdb::ExecutionContext exec_ctx(client_ctx, thread_ctx, nullptr);

  // Create global state
  auto global_state = std::make_shared<op::scan::duckdb_scan_task_global_state>(
    nullptr, scan_executor, client_ctx, physical_scan.get());

  // Create data repository manager (empty, unused for this test)
  cucascade::shared_data_repository data_repo;

  // Create local state
  auto local_state =
    std::make_unique<op::scan::duckdb_scan_task_local_state>(*global_state, exec_ctx, batch_size);

  // Create and schedule test task
  uint64_t task_id = 1;
  auto task        = std::make_unique<test_scan_task>(
    task_id, &data_repo, con, staging_table, std::move(local_state), global_state);

  // Run task
  scan_executor.start();
  scan_executor.schedule(std::move(task));
  scan_executor.wait();
  scan_executor.stop();

  // Validate tables are identical
  validate_tables_equal(con, table_name, staging_table);

  // Release DuckDB table function state before committing
  // The global_tf_state holds a shared checkpoint lock on the source table
  // (acquired via InitializeParallelScan -> SharedLockTable) that must be
  // released before COMMIT can proceed
  global_state.reset();

  // End the transaction
  auto commit_result = con.Query("COMMIT");
  REQUIRE(commit_result);
  REQUIRE(!commit_result->HasError());

  // Cleanup
  con.Query("DROP TABLE " + staging_table);
  con.Query("DROP TABLE " + table_name);
}

//===----------------------------------------------------------------------===//
// Test: Single-threaded scan executor
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - single threaded small table", "[scan_executor][single_thread]")
{
  // Use 10MB batch size to ensure multiple 1MB blocks are allocated
  run_scan_test("test_small", 100, 1, 10000000, 1);
}

TEST_CASE("scan_executor - single threaded with small batches", "[scan_executor][single_thread]")
{
  // Use a small batch size to force multiple batches
  // With 4 columns (INT + BIGINT + DOUBLE + VARCHAR(256)) = 4 + 8 + 8 + 256 = 276 bytes per row
  // So ~600000 bytes should fit about 2175 rows (1 vector)
  run_scan_test("test_medium", 10000, 1, 600000, 2);
}

//===----------------------------------------------------------------------===//
// Test: Multi-threaded scan executor
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - multi threaded small table", "[scan_executor][multi_thread]")
{
  run_scan_test("test_mt_small", 1000, 4, 1000000, 3);
}

TEST_CASE("scan_executor - multi threaded medium table", "[scan_executor][multi_thread]")
{
  // Use a medium batch size to force multiple batches across multiple threads
  // With 4 columns (INT + BIGINT + DOUBLE + VARCHAR(256)) = 4 + 8 + 8 + 256 = 276 bytes per row
  // So ~1000000 bytes should fit about 3623 rows (~2 vectors)
  run_scan_test("test_mt_medium", 100000, 4, 1000000, 4);
}

TEST_CASE("scan_executor - multi threaded large table", "[scan_executor][multi_thread]")
{
  run_scan_test("test_mt_large", 500000, 8, 1000000, 5);
}

//===----------------------------------------------------------------------===//
// Test: Edge cases
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - empty table", "[scan_executor][edge_case]")
{
  run_scan_test("test_empty", 0, 1, 1000000, 6);
}

TEST_CASE("scan_executor - single row table", "[scan_executor][edge_case]")
{
  run_scan_test("test_single_row", 1, 1, 1000000, 7);
}
