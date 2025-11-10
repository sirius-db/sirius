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

// sirius
#include <data/data_repository.hpp>
#include <scan/duckdb_scan_executor.hpp>
#include <scan/duckdb_scan_task.hpp>
#include <scan/physical_table_scan_adapter.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parallel/thread_context.hpp>

// standard library
#include <chrono>
#include <memory>
#include <string>
#include <thread>

using idx_t = duckdb::idx_t;
using namespace sirius;

//===----------------------------------------------------------------------===//
// Test Scan Task - Custom task that appends column_builder data to table
//===----------------------------------------------------------------------===//

/**
 * @brief Test version of duckdb_scan_task that appends scanned data to a DuckDB table
 *
 * This task executes the full scan pipeline (get_next_chunk -> process_chunk)
 * and then reads data from the column_builders to append to a staging table.
 */
class test_scan_task : public parallel::duckdb_scan_task {
 public:
  test_scan_task(uint64_t task_id,
                 duckdb::Connection& con,
                 std::string const& table_name,
                 sirius::unique_ptr<parallel::duckdb_scan_task_local_state> l_state,
                 sirius::shared_ptr<parallel::duckdb_scan_task_global_state> g_state)
    : duckdb_scan_task(task_id,
                       *reinterpret_cast<data_repository_manager*>(std::nullptr),  // Not used
                       std::move(l_state),
                       g_state),
      con_(con),
      table_name_(table_name)
  {
  }

  void execute() override
  {
    auto& l_state = this->_local_state->cast<parallel::duckdb_scan_task_local_state>();
    auto& g_state = this->_global_state->cast<parallel::duckdb_scan_task_global_state>();

    // Initialize the data chunk
    l_state.chunk.Initialize(duckdb::Allocator::Get(l_state.exec_ctx.client),
                             g_state.op.returned_types);

    // Scan loop - process chunks into column builders
    while (get_next_chunk(l_state, g_state)) {
      if (!chunk_fits(l_state)) {
        throw duckdb::InternalException("Chunk does not fit in allocated buffers");
      }

      // Process the chunk into column builders
      process_chunk(l_state);
      l_state.row_offset += l_state.chunk.size();

      // Termination condition
      if (STANDARD_VECTOR_SIZE * l_state.row_offset >= l_state.estimated_rows_per_batch) { break; }
    }

    // Add tasks back to the queue if the scan is not finished
    if (!g_state.IsSourceDrained()) {
      auto const new_task_id = this->task_id + g_state.max_threads;
      auto new_local_state   = sirius::make_unique<duckdb_scan_task_local_state>(
        g_state, l_state.exec_ctx, l_state.approximate_batch_size);
      auto shared_global_state =
        std::static_pointer_cast<duckdb_scan_task_global_state>(this->_global_state);
      auto next_task = sirius::make_unique<duckdb_scan_task>(
        new_task_id, dr_mgr, std::move(new_local_state), shared_global_state);
      g_state.scan_executor.schedule(std::move(next_task));
    }

    // Append data from column_builders to staging table
    append_to_table(l_state);
  }

 private:
  /**
   * @brief Helper to check if a bit in a validity mask is set (1 = valid, 0 = invalid)
   */
  static inline bool is_valid(uint8_t const* mask, idx_t row_idx)
  {
    auto const byte_idx = row_idx / 8;
    auto const bit_idx  = row_idx % 8;
    return (mask[byte_idx] & (1 << bit_idx)) != 0;
  }

  /**
   * @brief Append data from column_builders to the staging table
   *
   * Reads data directly from the column_builder buffers (data_blocks_accessor,
   * mask_blocks_accessor, offset_blocks_accessor) and appends to DuckDB table.
   */
  void append_to_table(parallel::duckdb_scan_task_local_state& l_state)
  {
    auto const num_rows = l_state.row_offset;
    if (num_rows == 0) {
      return;  // Nothing to append
    }

    duckdb::Appender app(con_, table_name_);
    auto const& column_builders = l_state.column_builders;

    for (idx_t i = 0; i < num_rows; ++i) {
      app.BeginRow();

      for (idx_t col = 0; col < column_builders.size(); ++col) {
        auto const& builder = column_builders[col];
        auto const& type    = builder.type;

        // Get raw pointers to the data
        auto const* data_ptr = builder.data_blocks_accessor.get_base_ptr();
        auto const* mask_ptr = builder.mask_blocks_accessor.get_base_ptr();

        // Check validity
        bool valid = true;
        if (mask_ptr) { valid = is_valid(mask_ptr, i); }

        if (!valid) {
          app.Append(duckdb::Value());  // NULL value
          continue;
        }

        // Type switch
        switch (type.id()) {
          case duckdb::LogicalTypeId::CHAR:  // Fallthrough
          case duckdb::LogicalTypeId::VARCHAR: {
            auto const* offset_ptr = builder.offset_blocks_accessor.get_base_ptr();
            auto const beg         = offset_ptr[i];
            auto const end         = offset_ptr[i + 1];
            auto const* str_ptr    = reinterpret_cast<const char*>(data_ptr + beg);
            auto const len         = end - beg;
            app.Append<duckdb::string_t>(std::string(str_ptr, len));
            break;
          }
          case duckdb::LogicalTypeId::INTEGER: {
            auto const* int_ptr = reinterpret_cast<const int32_t*>(data_ptr);
            app.Append<int32_t>(int_ptr[i]);
            break;
          }
          case duckdb::LogicalTypeId::BIGINT: {
            auto const* bigint_ptr = reinterpret_cast<const int64_t*>(data_ptr);
            app.Append<int64_t>(bigint_ptr[i]);
            break;
          }
          case duckdb::LogicalTypeId::DOUBLE: {
            auto const* double_ptr = reinterpret_cast<const double*>(data_ptr);
            app.Append<double>(double_ptr[i]);
            break;
          }
          case duckdb::LogicalTypeId::FLOAT: {
            auto const* float_ptr = reinterpret_cast<const float*>(data_ptr);
            app.Append<float>(float_ptr[i]);
            break;
          }
          case duckdb::LogicalTypeId::DECIMAL: {
            auto width = duckdb::DecimalType::GetWidth(type);
            auto scale = duckdb::DecimalType::GetScale(type);

            switch (type.InternalType()) {
              case duckdb::PhysicalType::INT16: {
                auto const* dec_ptr = reinterpret_cast<const int16_t*>(data_ptr);
                app.Append(duckdb::Value::DECIMAL(dec_ptr[i], width, scale));
                break;
              }
              case duckdb::PhysicalType::INT32: {
                auto const* dec_ptr = reinterpret_cast<const int32_t*>(data_ptr);
                app.Append(duckdb::Value::DECIMAL(dec_ptr[i], width, scale));
                break;
              }
              case duckdb::PhysicalType::INT64: {
                auto const* dec_ptr = reinterpret_cast<const int64_t*>(data_ptr);
                app.Append(duckdb::Value::DECIMAL(dec_ptr[i], width, scale));
                break;
              }
              case duckdb::PhysicalType::INT128: {
                auto const* dec_ptr = reinterpret_cast<const duckdb::hugeint_t*>(data_ptr);
                app.Append(duckdb::Value::DECIMAL(dec_ptr[i], width, scale));
                break;
              }
              default: FAIL("Unsupported decimal internal type");
            }
            break;
          }
          case duckdb::LogicalTypeId::DATE: {
            auto const* date_ptr = reinterpret_cast<const int32_t*>(data_ptr);
            app.Append<duckdb::date_t>(duckdb::date_t(date_ptr[i]));
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
};

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
      int32_t id       = static_cast<int32_t>(i);
      int64_t value    = static_cast<int64_t>(i * 100);
      double price     = static_cast<double>(i) * 1.5;
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
static std::unique_ptr<duckdb::PhysicalTableScan> make_physical_table_scan(
  duckdb::ClientContext& ctx, std::string const& table_name)
{
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, INVALID_CATALOG);
  auto& schema  = catalog.GetSchema(ctx, DEFAULT_SCHEMA);

  auto table_entry = schema.GetEntry(ctx, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(table_entry);

  auto& table_catalog_entry = table_entry->Cast<duckdb::TableCatalogEntry>();

  // Get all column IDs
  std::vector<duckdb::column_t> column_ids;
  for (size_t i = 0; i < table_catalog_entry.GetColumns().LogicalColumnCount(); ++i) {
    column_ids.push_back(static_cast<duckdb::column_t>(i));
  }

  // Create bind data
  auto bind_data = sirius::make_unique<duckdb::TableScanBindData>(table_catalog_entry);

  // Get the table scan function
  auto table_scan_function = duckdb::TableScanFunction::GetFunction();

  // Create PhysicalTableScan
  auto physical_scan = sirius::make_unique<duckdb::PhysicalTableScan>(
    table_catalog_entry.GetTypes(),
    table_scan_function,
    std::move(bind_data),
    column_ids,
    std::vector<duckdb::LogicalType>(),                  // projection_ids (empty = all columns)
    std::vector<std::unique_ptr<duckdb::Expression>>(),  // table_filters
    0                                                    // estimated_cardinality
  );

  return physical_scan;
}

//===----------------------------------------------------------------------===//
// Test: Single-threaded scan executor
//===----------------------------------------------------------------------===//

TEST_CASE("scan_executor - single threaded small table", "[scan_executor][single_thread]")
{
  // Setup DuckDB database
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  std::string table_name = "test_small";
  size_t num_rows        = 100;
  int num_threads        = 1;

  // Create and populate table
  create_synthetic_table(con, table_name, num_rows);

  // Get client context
  auto& client_ctx = *con.context;

  // Create physical table scan
  auto physical_scan = make_physical_table_scan(client_ctx, table_name);
  REQUIRE(physical_scan);

  // Create physical table scan adapter
  duckdb::physical_table_scan_adapter ptsa(*physical_scan);

  // Create staging table for scanned data
  std::string staging_table = table_name + "_scanned";
  auto create_result =
    con.Query("CREATE TABLE " + staging_table + " AS SELECT * FROM " + table_name + " WHERE 1=0");
  REQUIRE(create_result);
  REQUIRE(!create_result->HasError());

  // Create global state
  uint64_t pipeline_id = 1;
  parallel::duckdb_scan_executor scan_executor({num_threads, false});
  auto global_state = sirius::make_shared<parallel::duckdb_scan_task_global_state>(
    pipeline_id, scan_executor, client_ctx, ptsa);

  // Create and execute test task
  auto local_state = sirius::make_unique<parallel::duckdb_scan_task_local_state>(
    *global_state, client_ctx, 1000000);  // Large batch size to scan all rows

  uint64_t task_id = 1;
  auto task        = sirius::make_unique<test_scan_task>(
    task_id, con, staging_table, std::move(local_state), global_state);

  // Execute the task (will append to staging_table)
  task->execute();
  REQUIRE(task->get_global_state()->IsSourceDrained());

  // Validate tables are identical
  validate_tables_equal(con, table_name, staging_table);

  // Cleanup
  con.Query("DROP TABLE " + staging_table);
  con.Query("DROP TABLE " + table_name);
}

// TEST_CASE("scan_executor - single threaded medium table", "[scan_executor][single_thread]")
// {
//   duckdb::DuckDB db(nullptr);
//   duckdb::Connection con(db);

//   std::string table_name = "test_medium";
//   size_t num_rows        = 10000;

//   create_synthetic_table(con, table_name, num_rows);

//   auto& client_ctx   = *con.context;
//   auto physical_scan = make_physical_table_scan(client_ctx, table_name);
//   parallel::physical_table_scan_adapter ptsa(*physical_scan);

//   std::string staging_table = table_name + "_scanned";
//   con.Query("CREATE TABLE " + staging_table + " AS SELECT * FROM " + table_name + " WHERE 1=0");

//   uint64_t pipeline_id = 2;
//   parallel::duckdb_scan_executor scan_executor(1);
//   auto global_state = sirius::make_shared<parallel::duckdb_scan_task_global_state>(
//     pipeline_id, scan_executor, client_ctx, ptsa);
//   auto local_state =
//     sirius::make_unique<parallel::duckdb_scan_task_local_state>(*global_state, client_ctx,
//     1000000);
//   auto task = sirius::make_unique<test_scan_task>(
//     1, con, staging_table, std::move(local_state), global_state);

//   task->execute();

//   validate_tables_equal(con, table_name, staging_table);

//   con.Query("DROP TABLE " + staging_table);
//   con.Query("DROP TABLE " + table_name);
// }

// //===----------------------------------------------------------------------===//
// // Test: Multi-threaded scan executor
// //===----------------------------------------------------------------------===//

// TEST_CASE("scan_executor - multi threaded small table", "[scan_executor][multi_thread]")
// {
//   duckdb::DuckDB db(nullptr);
//   duckdb::Connection con(db);

//   std::string table_name = "test_mt_small";
//   size_t num_rows        = 1000;

//   create_synthetic_table(con, table_name, num_rows);

//   auto& client_ctx   = *con.context;
//   auto physical_scan = make_physical_table_scan(client_ctx, table_name);
//   parallel::physical_table_scan_adapter ptsa(*physical_scan);

//   // Create staging table
//   std::string staging_table = table_name + "_scanned";
//   con.Query("CREATE TABLE " + staging_table + " AS SELECT * FROM " + table_name + " WHERE 1=0");

//   uint64_t pipeline_id = 3;
//   parallel::duckdb_scan_executor scan_executor(4);
//   auto global_state = sirius::make_shared<parallel::duckdb_scan_task_global_state>(
//     pipeline_id, scan_executor, client_ctx, ptsa);

//   // Create multiple tasks for multi-threaded execution
//   // Each task will append to the staging table (DuckDB handles concurrent appends)
//   std::vector<sirius::unique_ptr<test_scan_task>> tasks;
//   for (int i = 0; i < 4; ++i) {
//     auto local_state = sirius::make_unique<parallel::duckdb_scan_task_local_state>(
//       *global_state, client_ctx, 1000000);
//     tasks.push_back(sirius::make_unique<test_scan_task>(
//       i, con, staging_table, std::move(local_state), global_state));
//   }

//   // Execute tasks in parallel using threads
//   std::vector<std::thread> threads;
//   for (auto& task : tasks) {
//     threads.emplace_back([&task]() { task->execute(); });
//   }

//   // Wait for all threads
//   for (auto& thread : threads) {
//     thread.join();
//   }

//   // Validate results
//   validate_tables_equal(con, table_name, staging_table);

//   con.Query("DROP TABLE " + staging_table);
//   con.Query("DROP TABLE " + table_name);
// }

// TEST_CASE("scan_executor - multi threaded large table", "[scan_executor][multi_thread]")
// {
//   duckdb::DuckDB db(nullptr);
//   duckdb::Connection con(db);

//   std::string table_name = "test_mt_large";
//   size_t num_rows        = 100000;

//   create_synthetic_table(con, table_name, num_rows);

//   auto& client_ctx   = *con.context;
//   auto physical_scan = make_physical_table_scan(client_ctx, table_name);
//   parallel::physical_table_scan_adapter ptsa(*physical_scan);

//   std::string staging_table = table_name + "_scanned";
//   con.Query("CREATE TABLE " + staging_table + " AS SELECT * FROM " + table_name + " WHERE 1=0");

//   uint64_t pipeline_id = 4;
//   parallel::duckdb_scan_executor scan_executor(8);
//   auto global_state = sirius::make_shared<parallel::duckdb_scan_task_global_state>(
//     pipeline_id, scan_executor, client_ctx, ptsa);

//   auto start = std::chrono::high_resolution_clock::now();

//   // Create multiple tasks for multi-threaded execution
//   std::vector<sirius::unique_ptr<test_scan_task>> tasks;
//   for (int i = 0; i < 8; ++i) {
//     auto local_state = sirius::make_unique<parallel::duckdb_scan_task_local_state>(
//       *global_state, client_ctx, 1000000);
//     tasks.push_back(sirius::make_unique<test_scan_task>(
//       i, con, staging_table, std::move(local_state), global_state));
//   }

//   // Execute tasks in parallel
//   std::vector<std::thread> threads;
//   for (auto& task : tasks) {
//     threads.emplace_back([&task]() { task->execute(); });
//   }

//   for (auto& thread : threads) {
//     thread.join();
//   }

//   auto end      = std::chrono::high_resolution_clock::now();
//   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//   std::cout << "Multi-threaded scan (8 threads) of " << num_rows << " rows took "
//             << duration.count() << "ms\n";

//   validate_tables_equal(con, table_name, staging_table);

//   con.Query("DROP TABLE " + staging_table);
//   con.Query("DROP TABLE " + table_name);
// }

// //===----------------------------------------------------------------------===//
// // Test: Multiple concurrent pipelines
// //===----------------------------------------------------------------------===//

// TEST_CASE("scan_executor - multiple concurrent pipelines", "[scan_executor][multi_thread]")
// {
//   duckdb::DuckDB db(nullptr);
//   duckdb::Connection con(db);

//   // Create multiple tables
//   std::vector<std::string> tables = {"pipeline_table_1", "pipeline_table_2", "pipeline_table_3"};
//   std::vector<size_t> row_counts  = {5000, 10000, 7500};

//   for (size_t i = 0; i < tables.size(); ++i) {
//     create_synthetic_table(con, tables[i], row_counts[i]);
//   }

//   auto& client_ctx = *con.context;

//   // Create staging tables
//   std::vector<std::string> staging_tables;
//   for (size_t i = 0; i < tables.size(); ++i) {
//     std::string staging = tables[i] + "_scanned";
//     staging_tables.push_back(staging);
//     con.Query("CREATE TABLE " + staging + " AS SELECT * FROM " + tables[i] + " WHERE 1=0");
//   }

//   std::vector<std::thread> threads;

//   for (size_t i = 0; i < tables.size(); ++i) {
//     uint64_t pipeline_id = 100 + i;
//     parallel::duckdb_scan_executor scan_executor(4);
//     auto physical_scan = make_physical_table_scan(client_ctx, tables[i]);
//     parallel::physical_table_scan_adapter ptsa(*physical_scan);

//     auto global_state = sirius::make_shared<parallel::duckdb_scan_task_global_state>(
//       pipeline_id, scan_executor, client_ctx, ptsa);
//     auto local_state = sirius::make_unique<parallel::duckdb_scan_task_local_state>(
//       *global_state, client_ctx, 1000000);
//     auto task = sirius::make_unique<test_scan_task>(
//       i, con, staging_tables[i], std::move(local_state), global_state);

//     // Execute each pipeline in its own thread
//     threads.emplace_back([task = std::move(task)]() { task->execute(); });
//   }

//   // Wait for all threads
//   for (auto& thread : threads) {
//     thread.join();
//   }

//   // Verify each pipeline's results
//   for (size_t i = 0; i < tables.size(); ++i) {
//     validate_tables_equal(con, tables[i], staging_tables[i]);
//   }

//   // Cleanup
//   for (auto const& staging : staging_tables) {
//     con.Query("DROP TABLE " + staging);
//   }
//   for (auto const& table : tables) {
//     con.Query("DROP TABLE " + table);
//   }
// }

// //===----------------------------------------------------------------------===//
// // Test: Empty table
// //===----------------------------------------------------------------------===//

// TEST_CASE("scan_executor - empty table", "[scan_executor][edge_case]")
// {
//   duckdb::DuckDB db(nullptr);
//   duckdb::Connection con(db);

//   std::string table_name = "test_empty";

//   // Create empty table
//   create_synthetic_table(con, table_name, 0);

//   auto& client_ctx   = *con.context;
//   auto physical_scan = make_physical_table_scan(client_ctx, table_name);
//   parallel::physical_table_scan_adapter ptsa(*physical_scan);

//   std::string staging_table = table_name + "_scanned";
//   con.Query("CREATE TABLE " + staging_table + " AS SELECT * FROM " + table_name + " WHERE 1=0");

//   uint64_t pipeline_id = 5;
//   parallel::duckdb_scan_executor scan_executor(1);
//   auto global_state = sirius::make_shared<parallel::duckdb_scan_task_global_state>(
//     pipeline_id, scan_executor, client_ctx, ptsa);
//   auto local_state =
//     sirius::make_unique<parallel::duckdb_scan_task_local_state>(*global_state, client_ctx,
//     1000000);
//   auto task = sirius::make_unique<test_scan_task>(
//     1, con, staging_table, std::move(local_state), global_state);

//   task->execute();

//   // Empty table should produce no rows
//   validate_tables_equal(con, table_name, staging_table);

//   con.Query("DROP TABLE " + staging_table);
//   con.Query("DROP TABLE " + table_name);
// }

// //===----------------------------------------------------------------------===//
// // Test: Single row table
// //===----------------------------------------------------------------------===//

// TEST_CASE("scan_executor - single row table", "[scan_executor][edge_case]")
// {
//   duckdb::DuckDB db(nullptr);
//   duckdb::Connection con(db);

//   std::string table_name = "test_single_row";

//   create_synthetic_table(con, table_name, 1);

//   auto& client_ctx   = *con.context;
//   auto physical_scan = make_physical_table_scan(client_ctx, table_name);
//   parallel::physical_table_scan_adapter ptsa(*physical_scan);

//   std::string staging_table = table_name + "_scanned";
//   con.Query("CREATE TABLE " + staging_table + " AS SELECT * FROM " + table_name + " WHERE 1=0");

//   uint64_t pipeline_id = 6;
//   parallel::duckdb_scan_executor scan_executor(1);
//   auto global_state = sirius::make_shared<parallel::duckdb_scan_task_global_state>(
//     pipeline_id, scan_executor, client_ctx, ptsa);
//   auto local_state =
//     sirius::make_unique<parallel::duckdb_scan_task_local_state>(*global_state, client_ctx,
//     1000000);
//   auto task = sirius::make_unique<test_scan_task>(
//     1, con, staging_table, std::move(local_state), global_state);

//   task->execute();

//   validate_tables_equal(con, table_name, staging_table);

//   con.Query("DROP TABLE " + staging_table);
//   con.Query("DROP TABLE " + table_name);
// }
