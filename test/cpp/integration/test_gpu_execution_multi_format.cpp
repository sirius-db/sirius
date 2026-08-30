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

/**
 * @file test_gpu_execution_multi_format.cpp
 * @brief Integration tests for multi-format scan support through gpu_execution.
 *
 * Each test runs the query through both GPU and CPU and compares results. The live
 * content covers hive-partitioned parquet scans; the CSV section (read_csv via the
 * generic duckdb_scan path) is currently disabled.
 */

#include <cudf/utilities/default_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utils/child_process_environment.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

static fs::path get_project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT);
#else
  return fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
#endif
}

struct sirius_config_env_guard {
  sirius_config_env_guard(const std::string& config_path)
  {
    setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
  }
  ~sirius_config_env_guard() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

/**
 * @brief Base fixture providing compare_gpu_vs_cpu for multi-format tests.
 */
class MultiFormatFixtureBase {
 public:
  MultiFormatFixtureBase()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<sirius_config_env_guard>(cfg_path.string());
      db           = std::make_unique<duckdb::DuckDB>(nullptr);
      con          = std::make_unique<duckdb::Connection>(*db);
    }
  }

  static bool is_floating_point(duckdb::LogicalTypeId id)
  {
    return id == duckdb::LogicalTypeId::FLOAT || id == duckdb::LogicalTypeId::DOUBLE;
  }

  /// Collect all rows from a MaterializedQueryResult as sorted vectors of stringified values.
  static std::vector<std::vector<std::string>> collect_rows(duckdb::MaterializedQueryResult& result)
  {
    std::vector<std::vector<std::string>> rows;
    for (duckdb::idx_t r = 0; r < result.RowCount(); r++) {
      std::vector<std::string> row;
      row.reserve(result.ColumnCount());
      for (duckdb::idx_t c = 0; c < result.ColumnCount(); c++) {
        row.push_back(result.GetValue(c, r).ToString());
      }
      rows.push_back(std::move(row));
    }
    std::sort(rows.begin(), rows.end());
    return rows;
  }

  void compare_gpu_vs_cpu(const std::string& query,
                          std::optional<float> float_tolerance = std::nullopt)
  {
    // Enable transparent GPU execution
    con->Query("SET gpu_execution = true;");
    auto before_gpu_stats = sirius::test::get_transparent_execution_stats(*con);

    // Run on GPU (transparent — plain SQL goes through Sirius optimizer hook)
    auto gpu_result = con->Query(query);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("transparent GPU execution error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());
    auto after_gpu_stats = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(before_gpu_stats, after_gpu_stats, 1, 0, 1);

    // Run on CPU (disable transparent execution)
    con->Query("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    con->Query("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());
    auto after_cpu_stats = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(after_gpu_stats, after_cpu_stats, 0, 0, 0);

    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    // Build a per-column flag for which columns are floating-point.
    std::vector<bool> col_is_float(gpu_result->ColumnCount());
    for (duckdb::idx_t c = 0; c < gpu_result->ColumnCount(); c++) {
      col_is_float[c] = is_floating_point(gpu_result->types[c].id());
    }

    // Collect and sort rows from already-materialized results for deterministic comparison.
    auto& gpu_mat = gpu_result->Cast<duckdb::MaterializedQueryResult>();
    auto& cpu_mat = cpu_result->Cast<duckdb::MaterializedQueryResult>();
    auto gpu_rows = collect_rows(gpu_mat);
    auto cpu_rows = collect_rows(cpu_mat);

    for (duckdb::idx_t r = 0; r < gpu_rows.size(); r++) {
      for (duckdb::idx_t c = 0; c < gpu_rows[r].size(); c++) {
        if (float_tolerance.has_value() && col_is_float[c]) {
          double gpu_d = std::stod(gpu_rows[r][c]);
          double cpu_d = std::stod(cpu_rows[r][c]);
          double diff  = std::fabs(gpu_d - cpu_d);
          if (diff > static_cast<double>(float_tolerance.value())) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " float mismatch: GPU=[" << gpu_d
                                 << "] CPU=[" << cpu_d << "] diff=" << diff
                                 << " tolerance=" << float_tolerance.value());
            REQUIRE(diff <= static_cast<double>(float_tolerance.value()));
          }
        } else {
          if (gpu_rows[r][c] != cpu_rows[r][c]) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_rows[r][c]
                                 << "] CPU=[" << cpu_rows[r][c] << "]");
          }
          REQUIRE(gpu_rows[r][c] == cpu_rows[r][c]);
        }
      }
    }
  }

  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::unique_ptr<sirius_config_env_guard> config_guard;
};

/**
 * @brief CSV test fixture.
 *
 * Generates CSV files from the existing parquet test data into a temp directory,
 * then creates views using read_csv(). This tests the generic duckdb_scan path
 * that routes non-parquet table functions through DuckDB's scan infrastructure.
 */
// class GPUExecutionCSVFixture : public MultiFormatFixtureBase {
//  public:
//   GPUExecutionCSVFixture()
//   {
//     auto parquet_dir = fs::path(__FILE__).parent_path() / "data/parquet";
//     csv_dir          = fs::temp_directory_path() / "sirius_test_csv";
//     fs::create_directories(csv_dir);

//     // Export parquet to CSV
//     std::vector<std::string> tables = {
//       "nation", "region", "customer", "orders", "lineitem", "part", "partsupp", "supplier"};

//     for (const auto& tbl : tables) {
//       auto pq_path  = parquet_dir / (tbl + ".parquet");
//       auto csv_path = csv_dir / (tbl + ".csv");
//       if (!fs::exists(pq_path)) continue;

//       auto result = con->Query("COPY (SELECT * FROM read_parquet('" + pq_path.string() +
//                                "')) TO '" + csv_path.string() + "' (HEADER, DELIMITER ',');");
//       REQUIRE(result);
//       REQUIRE_FALSE(result->HasError());
//     }

//     // Create views from CSV files
//     for (const auto& tbl : tables) {
//       auto csv_path = csv_dir / (tbl + ".csv");
//       if (!fs::exists(csv_path)) continue;

//       auto result = con->Query("CREATE VIEW " + tbl + " AS SELECT * FROM read_csv('" +
//                                csv_path.string() + "');");
//       REQUIRE(result);
//       REQUIRE_FALSE(result->HasError());
//     }
//   }

//   ~GPUExecutionCSVFixture() { fs::remove_all(csv_dir); }

//   fs::path csv_dir;
// };

// //===----------------------------------------------------------------------===//
// // CSV Scan tests
// //===----------------------------------------------------------------------===//

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - scan single column",
//                  "[.][integration][gpu_execution][csv][scan]")
// {
//   compare_gpu_vs_cpu("select n_nationkey from nation;");
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - scan multiple columns",
//                  "[.][integration][gpu_execution][csv][scan]")
// {
//   compare_gpu_vs_cpu("select n_nationkey, n_regionkey, n_name from nation;");
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - scan all columns",
//                  "[.][integration][gpu_execution][csv][scan]")
// {
//   compare_gpu_vs_cpu("select * from region;");
// }

// //===----------------------------------------------------------------------===//
// // CSV Filter tests (exercises BoundConstantExpression with various types)
// //===----------------------------------------------------------------------===//

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - filter integer equality",
//                  "[.][integration][gpu_execution][csv][filter]")
// {
//   compare_gpu_vs_cpu("select n_nationkey, n_name from nation where n_regionkey = 1;");
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - filter string equality",
//                  "[.][integration][gpu_execution][csv][filter]")
// {
//   compare_gpu_vs_cpu("select r_regionkey from region where r_name = 'EUROPE';");
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - filter date comparison",
//                  "[.][integration][gpu_execution][csv][filter]")
// {
//   // This tests the TIMESTAMP_DAYS constant materializer fix
//   compare_gpu_vs_cpu(
//     "select o_orderkey, o_totalprice from orders "
//     "where o_orderdate >= date '1995-01-01' and o_orderdate < date '1995-04-01';");
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - filter date between",
//                  "[.][integration][gpu_execution][csv][filter]")
// {
//   // DuckDB may rewrite >= AND < to BETWEEN, exercising the BoundBetweenExpression path
//   compare_gpu_vs_cpu(
//     "select o_orderkey from orders "
//     "where o_orderdate between date '1995-01-01' and date '1995-03-31';");
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - filter float comparison",
//                  "[.][integration][gpu_execution][csv][filter]")
// {
//   // CSV reads DECIMAL columns as DOUBLE — tests FLOAT64 filter path
//   compare_gpu_vs_cpu("select l_orderkey from lineitem where l_discount > 0.05;", 0.001f);
// }

// //===----------------------------------------------------------------------===//
// // CSV Aggregation tests
// //===----------------------------------------------------------------------===//

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - group by with sum",
//                  "[.][integration][gpu_execution][csv][aggregate]")
// {
//   compare_gpu_vs_cpu("select n_regionkey, count(*) as cnt from nation group by n_regionkey;");
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - aggregate with float columns",
//                  "[.][integration][gpu_execution][csv][aggregate]")
// {
//   // Tests SUM/AVG on DOUBLE (CSV-inferred type)
//   compare_gpu_vs_cpu(
//     "select l_returnflag, sum(l_quantity) as sum_qty, avg(l_extendedprice) as avg_price "
//     "from lineitem group by l_returnflag;",
//     0.01f);
// }

// //===----------------------------------------------------------------------===//
// // CSV Join tests
// //===----------------------------------------------------------------------===//

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - inner join",
//                  "[.][integration][gpu_execution][csv][join]")
// {
//   compare_gpu_vs_cpu(
//     "select n.n_name, r.r_name from nation n inner join region r "
//     "on n.n_regionkey = r.r_regionkey;");
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - multi table join",
//                  "[.][integration][gpu_execution][csv][join]")
// {
//   compare_gpu_vs_cpu(
//     "select c.c_name, n.n_name from customer c "
//     "inner join nation n on c.c_nationkey = n.n_nationkey "
//     "inner join region r on n.n_regionkey = r.r_regionkey "
//     "where r.r_name = 'EUROPE' "
//     "order by c.c_name limit 10;");
// }

// //===----------------------------------------------------------------------===//
// // CSV TPC-H representative queries
// //===----------------------------------------------------------------------===//

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - tpch q1 pricing summary",
//                  "[.][integration][gpu_execution][csv][tpch]")
// {
//   compare_gpu_vs_cpu(
//     "select l_returnflag, l_linestatus, "
//     "sum(l_quantity) as sum_qty, "
//     "sum(l_extendedprice) as sum_base_price, "
//     "sum(l_extendedprice * (1 - l_discount)) as sum_disc_price "
//     "from lineitem "
//     "where l_shipdate <= date '1998-09-02' "
//     "group by l_returnflag, l_linestatus "
//     "order by l_returnflag, l_linestatus;",
//     0.01f);
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - tpch q3 shipping priority",
//                  "[.][integration][gpu_execution][csv][tpch]")
// {
//   compare_gpu_vs_cpu(
//     "select l_orderkey, "
//     "sum(l_extendedprice * (1 - l_discount)) as revenue, "
//     "o_orderdate, o_shippriority "
//     "from customer "
//     "inner join orders on c_custkey = o_custkey "
//     "inner join lineitem on l_orderkey = o_orderkey "
//     "where c_mktsegment = 'BUILDING' "
//     "and o_orderdate < date '1995-03-15' "
//     "and l_shipdate > date '1995-03-15' "
//     "group by l_orderkey, o_orderdate, o_shippriority "
//     "order by revenue desc limit 10;",
//     0.01f);
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - tpch q4 order priority",
//                  "[.][integration][gpu_execution][csv][tpch]")
// {
//   compare_gpu_vs_cpu(
//     "select o_orderpriority, count(*) as order_count "
//     "from orders "
//     "where o_orderdate >= date '1996-10-01' "
//     "and o_orderdate < date '1997-01-01' "
//     "and exists ( "
//     "  select * from lineitem "
//     "  where l_orderkey = o_orderkey "
//     "  and l_commitdate < l_receiptdate "
//     ") "
//     "group by o_orderpriority "
//     "order by o_orderpriority;");
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - tpch q6 revenue forecast",
//                  "[.][integration][gpu_execution][csv][tpch]")
// {
//   // Tests date + float filters together (the original "Unknown cudf type: 12" trigger)
//   compare_gpu_vs_cpu(
//     "select sum(l_extendedprice * l_discount) as revenue "
//     "from lineitem "
//     "where l_shipdate >= date '1997-01-01' "
//     "and l_shipdate < date '1998-01-01' "
//     "and l_discount between 0.07 - 0.01 and 0.07 + 0.01 "
//     "and l_quantity < 25;",
//     0.01f);
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - tpch q10 returned item reporting",
//                  "[.][integration][gpu_execution][csv][tpch]")
// {
//   compare_gpu_vs_cpu(
//     "select c_custkey, c_name, "
//     "sum(l_extendedprice * (1 - l_discount)) as revenue, "
//     "c_acctbal, n_name, c_address, c_phone, c_comment "
//     "from customer inner join orders on c_custkey = o_custkey "
//     "inner join lineitem on l_orderkey = o_orderkey "
//     "inner join nation on c_nationkey = n_nationkey "
//     "where o_orderdate >= date '1993-07-01' "
//     "and o_orderdate < date '1993-10-01' "
//     "and l_returnflag = 'R' "
//     "group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment "
//     "order by revenue desc limit 20;",
//     0.01f);
// }

// //===----------------------------------------------------------------------===//
// // CSV Order By / Limit tests
// //===----------------------------------------------------------------------===//

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - order by with limit",
//                  "[.][integration][gpu_execution][csv][order]")
// {
//   compare_gpu_vs_cpu(
//     "select o_orderkey, o_totalprice, o_orderdate "
//     "from orders order by o_totalprice desc limit 10;",
//     0.01f);
// }

// TEST_CASE_METHOD(GPUExecutionCSVFixture,
//                  "gpu_execution csv - order by date column",
//                  "[.][integration][gpu_execution][csv][order]")
// {
//   compare_gpu_vs_cpu(
//     "select o_orderkey, o_orderdate from orders "
//     "where o_orderstatus = 'F' order by o_orderdate limit 10;");
// }

//===----------------------------------------------------------------------===//
// Hive-partitioned parquet scan tests
//===----------------------------------------------------------------------===//

/**
 * @brief Test fixture for hive-partitioned parquet scans via gpu_execution.
 *
 * Dataset: test/cpp/integration/data/hive_partitioned/
 *   year=2024/month=01/data.parquet  (id=1, name=alice, amount=100.5)
 *   year=2024/month=02/data.parquet  (id=2, name=bob,   amount=200.75)
 *   year=2025/month=01/data.parquet  (id=3, name=charlie, amount=300.25)
 *
 * Partition columns (year, month) are NOT in the parquet files — their
 * values come from the directory paths.
 */
class HivePartitionDataset {
 public:
  HivePartitionDataset() : scratch{"hive_count_star"}, hive_dir{scratch.path()}
  {
    auto const source_root = get_project_root() / "test/cpp/integration/data/hive_partitioned";
    auto const y2024_m01   = source_root / "year=2024/month=01/data.parquet";
    auto const y2024_m02   = source_root / "year=2024/month=02/data.parquet";
    auto const y2025_m01   = source_root / "year=2025/month=01/data.parquet";
    REQUIRE(fs::exists(y2024_m01));
    REQUIRE(fs::exists(y2024_m02));
    REQUIRE(fs::exists(y2025_m01));

    fs::create_directories(hive_dir / "h/year=2024/month=01");
    fs::create_directories(hive_dir / "h/year=2024/month=02");
    fs::create_directories(hive_dir / "h/year=2025/month=01");
    fs::create_directories(hive_dir / "flat");

    fs::copy_file(y2024_m01,
                  hive_dir / "h/year=2024/month=01/data.parquet",
                  fs::copy_options::overwrite_existing);
    fs::copy_file(y2024_m02,
                  hive_dir / "h/year=2024/month=02/data.parquet",
                  fs::copy_options::overwrite_existing);
    fs::copy_file(y2025_m01,
                  hive_dir / "h/year=2025/month=01/data.parquet",
                  fs::copy_options::overwrite_existing);

    fs::copy_file(
      y2024_m01, hive_dir / "flat/part_2024_01.parquet", fs::copy_options::overwrite_existing);
    fs::copy_file(
      y2024_m02, hive_dir / "flat/part_2024_02.parquet", fs::copy_options::overwrite_existing);
    fs::copy_file(
      y2025_m01, hive_dir / "flat/part_2025_01.parquet", fs::copy_options::overwrite_existing);

    hive_path   = (hive_dir / "h/year=*/month=*/*.parquet").string();
    flat_path   = (hive_dir / "flat/*.parquet").string();
    config_path = hive_dir / "watchdog_integration.yaml";
    write_watchdog_config(config_path);
    is_hive_available = true;
  }

  struct watchdog_result {
    bool timed_out{false};
    duckdb::idx_t row_count{0};
    duckdb::idx_t column_count{0};
    std::vector<std::vector<std::string>> rows;
    std::string error;
  };

  std::string hive_scan() const
  {
    return "read_parquet(" + sirius::test::sql_literal(hive_path) + ", hive_partitioning=true)";
  }

  std::string flat_scan() const
  {
    return "read_parquet(" + sirius::test::sql_literal(flat_path) + ")";
  }

  static std::vector<std::string> split_row(std::string const& line)
  {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string part;
    while (std::getline(ss, part, '\t')) {
      parts.push_back(std::move(part));
    }
    return parts;
  }

  static void write_watchdog_result(fs::path const& path, watchdog_result const& result)
  {
    std::ofstream out(path);
    out << "ERROR\t" << result.error << "\n";
    out << "SHAPE\t" << result.row_count << "\t" << result.column_count << "\n";
    for (auto const& row : result.rows) {
      out << "ROW";
      for (auto const& value : row) {
        out << '\t' << value;
      }
      out << "\n";
    }
  }

  static void write_watchdog_config(fs::path const& path)
  {
    std::ofstream out(path);
    out << R"(sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.2
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: 8000000000
      initial_number_pools: 4
      pool_size: 512
      block_size: 1048576
  executor:
    pipeline:
      num_threads: 2
    task_creator:
      num_threads: 1
    downgrade:
      num_threads: 1
      monitor_period: 10ms
  operator_params:
    scan_task_batch_size: 100000000
    max_sort_partition_bytes: 0
    hash_partition_bytes: 100000000
    concat_batch_bytes: 100000000
    max_build_hash_table_bytes: 90000000
)";
  }

  static watchdog_result read_watchdog_result(fs::path const& path)
  {
    watchdog_result result;
    std::ifstream in(path);
    if (!in) {
      result.error = "watchdog child did not write a result file";
      return result;
    }
    std::string line;
    while (std::getline(in, line)) {
      auto parts = split_row(line);
      if (parts.empty()) { continue; }
      if (parts[0] == "ERROR") {
        if (parts.size() > 1) { result.error = parts[1]; }
      } else if (parts[0] == "SHAPE") {
        if (parts.size() >= 3) {
          result.row_count    = static_cast<duckdb::idx_t>(std::stoull(parts[1]));
          result.column_count = static_cast<duckdb::idx_t>(std::stoull(parts[2]));
        }
      } else if (parts[0] == "ROW") {
        parts.erase(parts.begin());
        result.rows.push_back(std::move(parts));
      }
    }
    return result;
  }

  watchdog_result run_gpu_query_with_watchdog(std::string const& query,
                                              std::chrono::seconds timeout)
  {
    static std::atomic<std::uint64_t> next_child_id{0};
    auto const output_path =
      hive_dir / ("watchdog_result_" + std::to_string(next_child_id.fetch_add(1)) + ".txt");
    std::vector<std::string> child_arguments{"sirius_unittest",
                                             "gpu_execution hive partition watchdog child runner"};
    std::vector<char*> child_argv;
    child_argv.reserve(child_arguments.size() + 1);
    for (auto& argument : child_arguments) {
      child_argv.push_back(argument.data());
    }
    child_argv.push_back(nullptr);

    sirius::test::child_process_environment child_environment{
      {{"SIRIUS_HIVE_WATCHDOG_QUERY", query},
       {"SIRIUS_HIVE_WATCHDOG_OUTPUT", output_path.string()},
       {"SIRIUS_HIVE_WATCHDOG_CONFIG", config_path.string()},
       {"SIRIUS_CONFIG_FILE", config_path.string()}},
      {"SIRIUS_DISABLE"}};

    pid_t pid{};
    auto const spawn_result = ::posix_spawn(
      &pid, "/proc/self/exe", nullptr, nullptr, child_argv.data(), child_environment.data());
    REQUIRE(spawn_result == 0);

    int status      = 0;
    auto const stop = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < stop) {
      auto const waited = ::waitpid(pid, &status, WNOHANG);
      if (waited == pid) {
        auto result = read_watchdog_result(output_path);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
          result.error = result.error.empty() ? "watchdog child exited abnormally" : result.error;
        }
        return result;
      }
      if (waited < 0) {
        watchdog_result result;
        result.error = "waitpid failed while waiting for watchdog child";
        return result;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }

    (void)::kill(pid, SIGKILL);
    (void)::waitpid(pid, &status, 0);
    watchdog_result out;
    out.timed_out = true;
    out.error     = "query timed out after " + std::to_string(timeout.count()) + " seconds";
    return out;
  }

  void require_gpu_rows(std::string const& query,
                        std::vector<std::vector<std::string>> const& expected_rows)
  {
    auto result = run_gpu_query_with_watchdog(query, std::chrono::seconds{60});
    INFO(query);
    INFO(result.error);
    REQUIRE_FALSE(result.timed_out);
    REQUIRE(result.error.empty());
    CHECK(result.rows == expected_rows);
  }

  sirius::test::scratch_dir scratch;
  fs::path hive_dir;
  fs::path config_path;
  std::string hive_path;
  std::string flat_path;
  bool is_hive_available = false;  // No extension is needed for DuckDB hive partition discovery.
};

class EscapedHivePartitionDataset {
 public:
  EscapedHivePartitionDataset() : scratch{"hive_unescape"}, hive_dir{scratch.path()}
  {
    auto const source =
      get_project_root() /
      "test/cpp/integration/data/hive_partitioned/year=2024/month=01/data.parquet";
    REQUIRE(fs::exists(source));

    copy_partition(source, hive_dir / "space/city=New%20York/data.parquet");
    copy_partition(source, hive_dir / "slash/city=Path%2FTeam/data.parquet");
    copy_partition(source, hive_dir / "percent/city=100%25/data.parquet");
    copy_partition(source, hive_dir / "multiple/city=Los%20Angeles%2FWest/data.parquet");
    copy_partition(source, hive_dir / "two_columns/city=New%20York/dept=R%26D/data.parquet");
    copy_partition(source, hive_dir / "plain/city=Boston/data.parquet");
  }

  std::string partition_scan(fs::path const& relative_glob) const
  {
    return "read_parquet(" + sirius::test::sql_literal((hive_dir / relative_glob).string()) +
           ", hive_partitioning=true)";
  }

 private:
  static void copy_partition(fs::path const& source, fs::path const& target)
  {
    fs::create_directories(target.parent_path());
    fs::copy_file(source, target, fs::copy_options::overwrite_existing);
  }

  sirius::test::scratch_dir scratch;
  fs::path hive_dir;
};

class GPUExecutionHivePartitionFixture : public MultiFormatFixtureBase,
                                         public HivePartitionDataset {};

class GPUExecutionEscapedHivePartitionFixture : public MultiFormatFixtureBase,
                                                public EscapedHivePartitionDataset {};

TEST_CASE("gpu_execution hive partition watchdog child runner",
          "[.][gpu_execution][hive_partition][watchdog_child]")
{
  auto const* query      = std::getenv("SIRIUS_HIVE_WATCHDOG_QUERY");
  auto const* output_raw = std::getenv("SIRIUS_HIVE_WATCHDOG_OUTPUT");
  if (query == nullptr || output_raw == nullptr) { return; }

  HivePartitionDataset::watchdog_result out;
  try {
    auto const* config_raw = std::getenv("SIRIUS_HIVE_WATCHDOG_CONFIG");
    if (config_raw == nullptr) { out.error = "watchdog child missing config path"; }

    std::unique_ptr<duckdb::DuckDB> db;
    std::unique_ptr<duckdb::Connection> con;
    if (out.error.empty()) {
      db  = std::make_unique<duckdb::DuckDB>(nullptr);
      con = std::make_unique<duckdb::Connection>(*db);
    }

    auto set_gpu = out.error.empty() ? con->Query("SET gpu_execution = true;") : nullptr;
    if (!set_gpu) {
      out.error = "SET gpu_execution returned nullptr";
    } else if (set_gpu->HasError()) {
      out.error = set_gpu->GetError();
    }

    auto before_gpu_stats = out.error.empty()
                              ? sirius::test::get_transparent_execution_stats(*con)
                              : duckdb::SiriusContext::transparent_execution_stats{};
    if (out.error.empty()) {
      auto result = con->Query(query);
      if (!result) {
        out.error = "query returned nullptr";
      } else if (result->HasError()) {
        out.error = result->GetError();
      } else {
        out.row_count      = result->RowCount();
        out.column_count   = result->ColumnCount();
        auto& materialized = result->Cast<duckdb::MaterializedQueryResult>();
        out.rows           = MultiFormatFixtureBase::collect_rows(materialized);
      }
    }

    if (out.error.empty()) {
      auto after_gpu_stats = sirius::test::get_transparent_execution_stats(*con);
      if (after_gpu_stats.successful_rebinds != before_gpu_stats.successful_rebinds + 1 ||
          after_gpu_stats.fallbacks != before_gpu_stats.fallbacks ||
          after_gpu_stats.executions != before_gpu_stats.executions + 1) {
        out.error = "transparent execution stats delta mismatch";
      }
    }
  } catch (std::exception const& e) {
    out.error = e.what();
  } catch (...) {
    out.error = "query threw an unknown exception";
  }

  HivePartitionDataset::write_watchdog_result(output_raw, out);
  REQUIRE(out.error.empty());
}

TEST_CASE_METHOD(HivePartitionDataset,
                 "gpu_execution hive partition count star completes under watchdog",
                 "[gpu_execution][hive_partition][count_star][watchdog]")
{
  SECTION("hive count star") { require_gpu_rows("SELECT count(*) FROM " + hive_scan(), {{"3"}}); }

  SECTION("flat count star control")
  {
    require_gpu_rows("SELECT count(*) FROM " + flat_scan(), {{"3"}});
  }

  SECTION("partition-filtered count star")
  {
    require_gpu_rows("SELECT count(*) FROM " + hive_scan() + " WHERE year = 2024", {{"2"}});
  }

  SECTION("partition column select")
  {
    require_gpu_rows("SELECT year FROM " + hive_scan() + " ORDER BY year",
                     {{"2024"}, {"2024"}, {"2025"}});
  }
}

TEST_CASE_METHOD(GPUExecutionEscapedHivePartitionFixture,
                 "gpu_execution hive partition - unescapes varchar partition values",
                 "[integration][gpu_execution][hive_partition][unescape]")
{
  SECTION("projects a space-escaped partition column")
  {
    compare_gpu_vs_cpu("SELECT city FROM " + partition_scan("space/city=*/*.parquet"));
  }

  SECTION("projects data and a space-escaped partition column")
  {
    compare_gpu_vs_cpu("SELECT id, city FROM " + partition_scan("space/city=*/*.parquet"));
  }

  SECTION("unescapes a slash")
  {
    compare_gpu_vs_cpu("SELECT city FROM " + partition_scan("slash/city=*/*.parquet"));
  }

  SECTION("unescapes a percent sign")
  {
    compare_gpu_vs_cpu("SELECT city FROM " + partition_scan("percent/city=*/*.parquet"));
  }

  SECTION("unescapes multiple sequences in one value")
  {
    compare_gpu_vs_cpu("SELECT city FROM " + partition_scan("multiple/city=*/*.parquet"));
  }

  SECTION("unescapes every partition column")
  {
    compare_gpu_vs_cpu("SELECT id, city, dept FROM " +
                       partition_scan("two_columns/city=*/dept=*/*.parquet"));
  }

  SECTION("preserves an unescaped partition value")
  {
    compare_gpu_vs_cpu("SELECT id, city FROM " + partition_scan("plain/city=*/*.parquet"));
  }
}

TEST_CASE_METHOD(GPUExecutionHivePartitionFixture,
                 "gpu_execution hive partition - basic scan with partition columns",
                 "[.][integration][gpu_execution][hive_partition]")
{
  if (!is_hive_available) {
    WARN("hive extension not available — skipping");
    return;
  }
  compare_gpu_vs_cpu("SELECT * FROM read_parquet('" + hive_path +
                     "', hive_partitioning=true) ORDER BY id");
}

TEST_CASE_METHOD(GPUExecutionHivePartitionFixture,
                 "gpu_execution hive partition - filter on data column",
                 "[.][integration][gpu_execution][hive_partition]")
{
  if (!is_hive_available) {
    WARN("hive extension not available — skipping");
    return;
  }
  compare_gpu_vs_cpu("SELECT * FROM read_parquet('" + hive_path +
                     "', hive_partitioning=true) WHERE id >= 2 ORDER BY id");
}

TEST_CASE_METHOD(GPUExecutionHivePartitionFixture,
                 "gpu_execution hive partition - filter on partition column",
                 "[.][integration][gpu_execution][hive_partition]")
{
  if (!is_hive_available) {
    WARN("hive extension not available — skipping");
    return;
  }
  compare_gpu_vs_cpu("SELECT id, name, year FROM read_parquet('" + hive_path +
                     "', hive_partitioning=true) WHERE year = 2024 ORDER BY id");
}

TEST_CASE_METHOD(GPUExecutionHivePartitionFixture,
                 "gpu_execution hive partition - group by partition column",
                 "[.][integration][gpu_execution][hive_partition]")
{
  if (!is_hive_available) {
    WARN("hive extension not available — skipping");
    return;
  }
  compare_gpu_vs_cpu("SELECT year, SUM(amount) as total FROM read_parquet('" + hive_path +
                     "', hive_partitioning=true) GROUP BY year ORDER BY year");
}

TEST_CASE_METHOD(GPUExecutionHivePartitionFixture,
                 "gpu_execution hive partition - reversed column order",
                 "[.][integration][gpu_execution][hive_partition]")
{
  if (!is_hive_available) {
    WARN("hive extension not available — skipping");
    return;
  }
  compare_gpu_vs_cpu("SELECT year, month, amount, name, id FROM read_parquet('" + hive_path +
                     "', hive_partitioning=true) ORDER BY id");
}

TEST_CASE_METHOD(GPUExecutionHivePartitionFixture,
                 "gpu_execution hive partition - aggregation on data column",
                 "[.][integration][gpu_execution][hive_partition]")
{
  if (!is_hive_available) {
    WARN("hive extension not available — skipping");
    return;
  }
  compare_gpu_vs_cpu("SELECT SUM(amount) as total FROM read_parquet('" + hive_path +
                     "', hive_partitioning=true)");
}
