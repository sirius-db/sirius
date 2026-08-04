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
#include <sys/wait.h>
#include <unistd.h>
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

std::string sql_string_literal(std::string const& value)
{
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('\'');
  for (auto const c : value) {
    if (c == '\'') { escaped.push_back('\''); }
    escaped.push_back(c);
  }
  escaped.push_back('\'');
  return escaped;
}

/**
 * @brief How a query is expected to be routed by the transparent optimizer.
 *
 * Correctness alone cannot distinguish "the GPU computed the right answer" from "the GPU
 * declined and DuckDB computed the right answer" — both return the same rows. Every
 * comparison therefore asserts the route as well, so a silent regression to CPU shows up as a
 * test failure rather than a pass. This matters most for the iceberg cases below, where the
 * delete gate deliberately routes some tables to CPU: when a delete kind starts running on the
 * GPU, its expectation flips from `plan_fallback` to `gpu` and the assertion proves it.
 */
enum class gpu_route {
  gpu,              ///< Plan generated and executed on the GPU.
  plan_fallback,    ///< Declined at plan time (NotImplementedException) — DuckDB CPU ran it.
  runtime_fallback  ///< GPU plan installed, then failed at execute time — CPU fallback ran it.
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

  /// Assert that `query` took `route`, in terms of the transparent-execution counters.
  static void require_route(const duckdb::SiriusContext::transparent_execution_stats& before,
                            const duckdb::SiriusContext::transparent_execution_stats& after,
                            gpu_route route)
  {
    switch (route) {
      case gpu_route::gpu:
        sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 0);
        break;
      case gpu_route::plan_fallback:
        // create_plan threw NotImplementedException: no rebind, no GPU execution, one fallback.
        sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0, 0);
        break;
      case gpu_route::runtime_fallback:
        // The rebind succeeded and the GPU operator ran, then bailed to the stashed CPU plan.
        sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 1);
        break;
    }
  }

  void compare_gpu_vs_cpu(const std::string& query,
                          std::optional<float> float_tolerance = std::nullopt,
                          gpu_route route                      = gpu_route::gpu)
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
    require_route(before_gpu_stats, after_gpu_stats, route);

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
// Iceberg scan tests
//
// These cases were hidden ([.] tag) and hard-skipped (iceberg_available = false) while iceberg
// was out of the build, so they passed for a month without running any SQL — which is how the
// rot went unnoticed. They are now unhidden, unskipped, and a missing iceberg extension is a
// hard failure rather than a WARN-and-return.
//
// Every case asserts three things: the rows are right, DuckDB's own reader agrees, and the
// query took the route it is supposed to take. The literal row expectations matter as much as
// the GPU-vs-CPU comparison — an engine that drops deletes and a reader that drops deletes
// agree with each other perfectly.
//===----------------------------------------------------------------------===//

/**
 * @brief Iceberg test fixture.
 *
 * Loads the community iceberg extension and resolves the fixture tables under
 * test/cpp/integration/data/. Tests compare gpu_execution against DuckDB's native iceberg
 * reader and against literal expected rows.
 */
class GPUExecutionIcebergFixture : public MultiFormatFixtureBase {
 public:
  // Community-extension builds these expectations were verified against. A different build is
  // not automatically wrong, so a mismatch warns rather than fails — but it is recorded, so a
  // behaviour change arriving with an extension upgrade is attributable instead of mysterious.
  // (Under 75726455 the reader applies equality deletes; under the older 1.4.4-era build it
  // silently ignored them, which is why the pre-removal versions of these tests avoided
  // comparing against it.)
  static constexpr const char* kVerifiedIcebergVersion = "75726455";

  // Which delete kinds the GPU scan path applies itself. Positional deletes and V3 deletion
  // vectors are applied by iceberg_gpu_ingestible (they share one per-file position map), so
  // those tables must execute as GPU_SCAN — asserting the route is what distinguishes "the GPU
  // applied the deletes" from "the GPU quietly handed the table back to DuckDB", since both
  // return the same rows. Equality deletes match on key values and need the key columns
  // force-projected into the scan, which is not wired; the gate still sends them to CPU.
  static constexpr gpu_route kPositionalDeleteRoute = gpu_route::gpu;
  static constexpr gpu_route kDeletionVectorRoute   = gpu_route::gpu;
  static constexpr gpu_route kEqualityDeleteRoute   = gpu_route::plan_fallback;

  GPUExecutionIcebergFixture()
  {
    require_repo_root_cwd();
    load_iceberg_extension();

    auto root = get_project_root();
    v1_path   = (root / "test/cpp/integration/data/iceberg_v1").string();
    v2_path   = (root / "test/cpp/integration/data/iceberg_v2_delete").string();
  }

  /**
   * @brief Fail early, and legibly, when the process cwd is not the repo root.
   *
   * Fixture metadata (manifest lists, manifests, data files) stores repo-relative paths, so
   * iceberg_scan resolves them against the process cwd — passing an absolute table path does
   * not help. Without this check every iceberg case fails with an opaque
   * `IO Error: Cannot open file "test/cpp/..."` that reads like a missing fixture.
   */
  static void require_repo_root_cwd()
  {
    if (!fs::exists("test/cpp/integration/data/iceberg_v1/metadata")) {
      FAIL(
        "iceberg fixture metadata stores repo-relative paths, so these tests must run with "
        "the repo root as cwd; cwd is "
        << fs::current_path().string());
    }
  }

  /**
   * @brief Load the iceberg extension, failing hard if it is unavailable.
   *
   * Deliberately LOAD without INSTALL: a test must not reach the network, and an INSTALL here
   * would silently paper over an unprovisioned environment. The extension is a build/CI
   * dependency and is installed once, out of band.
   */
  void load_iceberg_extension()
  {
    auto load = con->Query("LOAD iceberg;");
    REQUIRE(load);
    if (load->HasError()) {
      FAIL("the iceberg extension is required by these tests but could not be loaded: "
           << load->GetError()
           << "\nprovision it once with:  duckdb -unsigned -c \"INSTALL avro; INSTALL iceberg;\"");
    }

    auto version = con->Query(
      "SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'iceberg';");
    REQUIRE(version);
    REQUIRE_FALSE(version->HasError());
    REQUIRE(version->RowCount() == 1);
    auto const installed = version->GetValue(0, 0).ToString();
    if (installed != kVerifiedIcebergVersion) {
      WARN("iceberg extension version " << installed << " differs from the verified "
                                        << kVerifiedIcebergVersion
                                        << "; reader behaviour differences trace to this");
    }
  }

  /// Non-data manifest entries (V2 positional, V2 equality, V3 deletion vectors — the last
  /// reported as content='POSITION_DELETES', file_format='PUFFIN') at the snapshot being read.
  int64_t delete_file_count(const std::string& table_path, const std::string& extra_args = "")
  {
    auto result = con->Query("SELECT count(*) FROM iceberg_metadata('" + table_path + "'" +
                             extra_args + ") WHERE content <> 'EXISTING';");
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("iceberg_metadata error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->RowCount() == 1);
    return result->GetValue(0, 0).GetValue<int64_t>();
  }

  /**
   * @brief Canary: prove the table really carries delete files before asserting deletes work.
   *
   * The engine once fabricated empty delete data, and the delete tests still looked correct —
   * with nothing to delete, a right-looking row set proves nothing. Asserting the delete-file
   * count up front means a delete test can no longer pass by having no deletes to apply.
   */
  void require_delete_files(const std::string& table_path, int64_t expected)
  {
    INFO("delete-file canary for " << table_path);
    REQUIRE(delete_file_count(table_path) == expected);
  }

  /**
   * @brief Run a query on the GPU and assert route, oracle agreement, and literal rows.
   *
   * `expected` is compared as stringified rows in sorted order, so it works for any result
   * shape. It is not redundant with the GPU-vs-CPU check: if DuckDB's reader ever stops
   * applying deletes, a GPU that also ignores them would still match it.
   */
  void expect_iceberg_rows(const std::string& query,
                           gpu_route route,
                           std::vector<std::vector<std::string>> expected)
  {
    INFO("query: " << query);

    con->Query("SET gpu_execution = true;");
    auto before     = sirius::test::get_transparent_execution_stats(*con);
    auto gpu_result = con->Query(query);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("transparent GPU execution error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());
    require_route(before, sirius::test::get_transparent_execution_stats(*con), route);

    con->Query("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    con->Query("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    if (cpu_result->HasError()) {
      UNSCOPED_INFO("DuckDB reader error: " << cpu_result->GetError());
    }
    REQUIRE_FALSE(cpu_result->HasError());

    auto gpu_rows = collect_rows(gpu_result->Cast<duckdb::MaterializedQueryResult>());
    auto cpu_rows = collect_rows(cpu_result->Cast<duckdb::MaterializedQueryResult>());
    std::sort(expected.begin(), expected.end());

    REQUIRE(gpu_rows == cpu_rows);
    REQUIRE(gpu_rows == expected);
  }

  std::string v1_path;
  std::string v2_path;
};

//===----------------------------------------------------------------------===//
// V1 — append-only. Runs on the GPU: iceberg data files are parquet, and iceberg_scan binds
// them into the same MultiFileBindData read_parquet produces.
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - V1 basic scan",
                 "[integration][gpu_execution][iceberg]")
{
  REQUIRE(delete_file_count(v1_path) == 0);  // append-only: nothing for the gate to catch
  expect_iceberg_rows("SELECT fruit, count FROM iceberg_scan('" + v1_path + "') ORDER BY count;",
                      gpu_route::gpu,
                      {{"apple", "1"}, {"banana", "2"}, {"cherry", "3"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - V1 count(*)",
                 "[integration][gpu_execution][iceberg]")
{
  expect_iceberg_rows(
    "SELECT count(*) FROM iceberg_scan('" + v1_path + "');", gpu_route::gpu, {{"3"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - V1 filter and order by",
                 "[integration][gpu_execution][iceberg]")
{
  expect_iceberg_rows(
    "SELECT fruit, count FROM iceberg_scan('" + v1_path + "') WHERE count > 1 ORDER BY count;",
    gpu_route::gpu,
    {{"banana", "2"}, {"cherry", "3"}});
}

//===----------------------------------------------------------------------===//
// V2 positional deletes.
//
// Fixture: 5 rows apple/1..elderberry/5; the delete file removes positions 1 and 3
// (banana/2, date/4). Surviving: apple/1, cherry/3, elderberry/5.
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - V2 positional deletes basic scan",
                 "[integration][gpu_execution][iceberg]")
{
  require_delete_files(v2_path, 1);
  expect_iceberg_rows("SELECT fruit, count FROM iceberg_scan('" + v2_path + "') ORDER BY count;",
                      kPositionalDeleteRoute,
                      {{"apple", "1"}, {"cherry", "3"}, {"elderberry", "5"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - V2 positional deletes count(*)",
                 "[integration][gpu_execution][iceberg]")
{
  require_delete_files(v2_path, 1);
  // 5 data rows minus 2 deleted. A count that reads row counts from parquet footers without
  // applying deletes returns 5 — this is the case that catches it.
  expect_iceberg_rows(
    "SELECT count(*) FROM iceberg_scan('" + v2_path + "');", kPositionalDeleteRoute, {{"3"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - V2 positional deletes order by desc",
                 "[integration][gpu_execution][iceberg]")
{
  require_delete_files(v2_path, 1);
  expect_iceberg_rows(
    "SELECT fruit, count FROM iceberg_scan('" + v2_path + "') ORDER BY count DESC;",
    kPositionalDeleteRoute,
    {{"apple", "1"}, {"cherry", "3"}, {"elderberry", "5"}});
}

//===----------------------------------------------------------------------===//
// V2 equality deletes.
//
// Fixture: 5 rows apple/1..elderberry/5; an equality-delete file keyed on (fruit, count)
// removes banana/2 and date/4. Surviving: apple/1, cherry/3, elderberry/5.
//===----------------------------------------------------------------------===//

class GPUExecutionIcebergEqualityDeleteFixture : public GPUExecutionIcebergFixture {
 public:
  GPUExecutionIcebergEqualityDeleteFixture()
  {
    eq_path =
      (get_project_root() / "test/cpp/integration/data/iceberg_v2_equality_delete").string();
  }

  std::string eq_path;
};

TEST_CASE_METHOD(GPUExecutionIcebergEqualityDeleteFixture,
                 "gpu_execution iceberg - V2 equality deletes basic scan",
                 "[integration][gpu_execution][iceberg]")
{
  require_delete_files(eq_path, 1);
  expect_iceberg_rows("SELECT fruit, count FROM iceberg_scan('" + eq_path + "') ORDER BY count;",
                      kEqualityDeleteRoute,
                      {{"apple", "1"}, {"cherry", "3"}, {"elderberry", "5"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergEqualityDeleteFixture,
                 "gpu_execution iceberg - V2 equality deletes count(*)",
                 "[integration][gpu_execution][iceberg]")
{
  require_delete_files(eq_path, 1);
  expect_iceberg_rows(
    "SELECT count(*) FROM iceberg_scan('" + eq_path + "');", kEqualityDeleteRoute, {{"3"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergEqualityDeleteFixture,
                 "gpu_execution iceberg - V2 equality deletes filter on surviving rows",
                 "[integration][gpu_execution][iceberg]")
{
  require_delete_files(eq_path, 1);
  expect_iceberg_rows(
    "SELECT fruit, count FROM iceberg_scan('" + eq_path + "') WHERE count > 2 ORDER BY count;",
    kEqualityDeleteRoute,
    {{"cherry", "3"}, {"elderberry", "5"}});
}

//===----------------------------------------------------------------------===//
// V2 equality delete edge cases: single-column keys, multiple delete files, everything
// deleted, equality combined with positional, and multiple data files.
//===----------------------------------------------------------------------===//

class GPUExecutionIcebergEqEdgeCaseFixture : public GPUExecutionIcebergFixture {
 public:
  GPUExecutionIcebergEqEdgeCaseFixture()
  {
    auto root       = get_project_root();
    single_col_path = (root / "test/cpp/integration/data/iceberg_v2_eq_single_col").string();
    multi_del_path  = (root / "test/cpp/integration/data/iceberg_v2_eq_multi_delete_file").string();
    all_del_path    = (root / "test/cpp/integration/data/iceberg_v2_eq_all_deleted").string();
    combined_path   = (root / "test/cpp/integration/data/iceberg_v2_eq_pos_combined").string();
    multi_data_path = (root / "test/cpp/integration/data/iceberg_v2_eq_multi_data_file").string();
  }

  std::string single_col_path;
  std::string multi_del_path;
  std::string all_del_path;
  std::string combined_path;
  std::string multi_data_path;
};

TEST_CASE_METHOD(GPUExecutionIcebergEqEdgeCaseFixture,
                 "gpu_execution iceberg - V2 equality deletes single-column key",
                 "[integration][gpu_execution][iceberg]")
{
  require_delete_files(single_col_path, 1);
  // The key is the fruit column alone (field_id=1); count is not part of it.
  // Delete entries "banana" and "date" must match on fruit regardless of count.
  expect_iceberg_rows(
    "SELECT fruit, count FROM iceberg_scan('" + single_col_path + "') ORDER BY count;",
    kEqualityDeleteRoute,
    {{"apple", "1"}, {"cherry", "3"}, {"elderberry", "5"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergEqEdgeCaseFixture,
                 "gpu_execution iceberg - V2 equality deletes multiple delete files",
                 "[integration][gpu_execution][iceberg]")
{
  // Two separate one-row equality-delete files — exercises the concatenate/deduplicate path.
  require_delete_files(multi_del_path, 2);
  expect_iceberg_rows(
    "SELECT fruit, count FROM iceberg_scan('" + multi_del_path + "') ORDER BY count;",
    kEqualityDeleteRoute,
    {{"apple", "1"}, {"cherry", "3"}, {"elderberry", "5"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergEqEdgeCaseFixture,
                 "gpu_execution iceberg - V2 equality deletes all rows deleted",
                 "[integration][gpu_execution][iceberg]")
{
  // The delete file covers every data row — an all-false mask, which is where an
  // apply_boolean_mask that mishandles the empty result shows up.
  require_delete_files(all_del_path, 1);
  expect_iceberg_rows(
    "SELECT fruit, count FROM iceberg_scan('" + all_del_path + "');", kEqualityDeleteRoute, {});
}

TEST_CASE_METHOD(GPUExecutionIcebergEqEdgeCaseFixture,
                 "gpu_execution iceberg - V2 equality and positional deletes combined",
                 "[integration][gpu_execution][iceberg]")
{
  // Both delete kinds against one table. The positional delete removes row 0 (apple/1). The
  // equality delete sits at the same sequence number as the data, so per the Iceberg spec it
  // does NOT apply — deletes only affect data files with strictly lower sequence numbers.
  // banana/2 surviving is the assertion that the sequence-number rule is honoured.
  require_delete_files(combined_path, 2);
  expect_iceberg_rows(
    "SELECT fruit, count FROM iceberg_scan('" + combined_path + "') ORDER BY count;",
    kEqualityDeleteRoute,
    {{"banana", "2"}, {"cherry", "3"}, {"date", "4"}, {"elderberry", "5"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergEqEdgeCaseFixture,
                 "gpu_execution iceberg - V2 equality deletes multiple data files",
                 "[integration][gpu_execution][iceberg]")
{
  // Two data files, one delete file removing a row from each: file 1 loses banana/2, file 2
  // loses elderberry/5. Exercises applying one delete set across per-file boundaries — the
  // same boundary the GPU path's one-file-per-batch coalescing rule exists to protect.
  require_delete_files(multi_data_path, 1);
  expect_iceberg_rows(
    "SELECT fruit, count FROM iceberg_scan('" + multi_data_path + "') ORDER BY count;",
    kEqualityDeleteRoute,
    {{"apple", "1"}, {"cherry", "3"}, {"date", "4"}, {"fig", "6"}});
}

//===----------------------------------------------------------------------===//
// V3 deletion vectors (Puffin / Roaring bitmap).
//
// Same data and same surviving rows as the V2 positional fixture — deliberately, so the DV
// path is checked to produce identical output to the positional path.
//===----------------------------------------------------------------------===//

class GPUExecutionIcebergDVFixture : public GPUExecutionIcebergFixture {
 public:
  GPUExecutionIcebergDVFixture()
  {
    dv_path =
      (get_project_root() / "test/cpp/integration/data/iceberg_v3_deletion_vector").string();
  }

  std::string dv_path;
};

TEST_CASE_METHOD(GPUExecutionIcebergDVFixture,
                 "gpu_execution iceberg - V3 deletion vector basic scan",
                 "[integration][gpu_execution][iceberg]")
{
  require_delete_files(dv_path, 1);
  expect_iceberg_rows("SELECT fruit, count FROM iceberg_scan('" + dv_path + "') ORDER BY count;",
                      kDeletionVectorRoute,
                      {{"apple", "1"}, {"cherry", "3"}, {"elderberry", "5"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergDVFixture,
                 "gpu_execution iceberg - V3 deletion vector count(*)",
                 "[integration][gpu_execution][iceberg]")
{
  require_delete_files(dv_path, 1);
  expect_iceberg_rows(
    "SELECT count(*) FROM iceberg_scan('" + dv_path + "');", kDeletionVectorRoute, {{"3"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergDVFixture,
                 "gpu_execution iceberg - V3 deletion vector filter",
                 "[integration][gpu_execution][iceberg]")
{
  // A filter over the surviving rows: a deleted row must not reappear because a predicate
  // happens to select it.
  require_delete_files(dv_path, 1);
  expect_iceberg_rows(
    "SELECT fruit, count FROM iceberg_scan('" + dv_path + "') WHERE count > 2 ORDER BY count;",
    kDeletionVectorRoute,
    {{"cherry", "3"}, {"elderberry", "5"}});
}

//===----------------------------------------------------------------------===//
// Schema evolution, snapshot selection, and sequence numbers.
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - schema evolution added column",
                 "[integration][gpu_execution][iceberg]")
{
  auto path = (get_project_root() / "test/cpp/integration/data/iceberg_schema_evolution").string();
  REQUIRE(delete_file_count(path) == 0);  // append-only — the gate lets this through

  // File 1 holds {fruit, count}; a later snapshot added `color`, so file 2 holds
  // {fruit, count, color} and file 1's rows must read back with color NULL.
  //
  // The GPU scan has no way to synthesize a column that is absent from a data file, so the
  // plan is accepted and then fails at execute time with "Projected column 'color' not found
  // in parquet file", and the stashed CPU plan returns the correct rows. Asserting
  // runtime_fallback (not gpu) records that this is unsupported-but-correct; the fix is the
  // MISSING-column entry kind in scan_plan, at which point this becomes gpu_route::gpu.
  expect_iceberg_rows("SELECT * FROM iceberg_scan('" + path + "') ORDER BY count;",
                      gpu_route::runtime_fallback,
                      {{"apple", "1", "NULL"},
                       {"banana", "2", "NULL"},
                       {"cherry", "3", "NULL"},
                       {"date", "4", "brown"},
                       {"elderberry", "5", "purple"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - snapshot-aware deletes",
                 "[integration][gpu_execution][iceberg]")
{
  auto path = (get_project_root() / "test/cpp/integration/data/iceberg_snapshot_deletes").string();

  // Latest snapshot carries a delete file, so the gate declines it and DuckDB applies it.
  require_delete_files(path, 1);
  expect_iceberg_rows("SELECT * FROM iceberg_scan('" + path + "') ORDER BY count;",
                      kPositionalDeleteRoute,
                      {{"apple", "1"}, {"cherry", "3"}, {"elderberry", "5"}});

  // Snapshot 1 predates the delete: all 5 rows, and — the point of this case — the gate must
  // probe the SAME snapshot the scan reads, so this one runs on the GPU. A gate that ignored
  // snapshot_from_id would see the table's delete file and needlessly decline.
  auto const snap1_args = ", snapshot_from_id = 9400000000000001";
  REQUIRE(delete_file_count(path, snap1_args) == 0);
  expect_iceberg_rows(
    "SELECT * FROM iceberg_scan('" + path + "'" + snap1_args + ") ORDER BY count;",
    gpu_route::gpu,
    {{"apple", "1"}, {"banana", "2"}, {"cherry", "3"}, {"date", "4"}, {"elderberry", "5"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - equality delete sequence numbers",
                 "[integration][gpu_execution][iceberg]")
{
  auto path = (get_project_root() / "test/cpp/integration/data/iceberg_eq_seq_number").string();
  // seq1 = [apple/1, banana/2, cherry/3], seq2 = delete(banana), seq3 = [banana/4, date/5].
  // The re-inserted banana/4 must survive: its data sequence number is above the delete's, and
  // an implementation that matches on key alone would wrongly drop it.
  require_delete_files(path, 1);
  expect_iceberg_rows("SELECT fruit, count FROM iceberg_scan('" + path + "') ORDER BY count;",
                      kEqualityDeleteRoute,
                      {{"apple", "1"}, {"cherry", "3"}, {"banana", "4"}, {"date", "5"}});
}

TEST_CASE_METHOD(GPUExecutionIcebergFixture,
                 "gpu_execution iceberg - deflate compressed manifests",
                 "[integration][gpu_execution][iceberg]")
{
  auto path = (get_project_root() / "test/cpp/integration/data/iceberg_deflate_manifest").string();
  REQUIRE(delete_file_count(path) == 0);
  expect_iceberg_rows("SELECT * FROM iceberg_scan('" + path + "') ORDER BY count;",
                      gpu_route::gpu,
                      {{"apple", "1"}, {"banana", "2"}, {"cherry", "3"}});
}

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
  HivePartitionDataset()
  {
    auto const source_root = get_project_root() / "test/cpp/integration/data/hive_partitioned";
    auto const y2024_m01   = source_root / "year=2024/month=01/data.parquet";
    auto const y2024_m02   = source_root / "year=2024/month=02/data.parquet";
    auto const y2025_m01   = source_root / "year=2025/month=01/data.parquet";
    REQUIRE(fs::exists(y2024_m01));
    REQUIRE(fs::exists(y2024_m02));
    REQUIRE(fs::exists(y2025_m01));

    static std::atomic<std::uint64_t> next_fixture_id{0};
    hive_dir = fs::temp_directory_path() /
               ("sirius_hive_count_star_" + std::to_string(next_fixture_id.fetch_add(1)));
    fs::remove_all(hive_dir);
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

  ~HivePartitionDataset() { fs::remove_all(hive_dir); }

  struct watchdog_result {
    bool timed_out{false};
    duckdb::idx_t row_count{0};
    duckdb::idx_t column_count{0};
    std::vector<std::vector<std::string>> rows;
    std::string error;
  };

  std::string hive_scan() const
  {
    return "read_parquet(" + sql_string_literal(hive_path) + ", hive_partitioning=true)";
  }

  std::string flat_scan() const { return "read_parquet(" + sql_string_literal(flat_path) + ")"; }

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
    auto const pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      ::setenv("SIRIUS_HIVE_WATCHDOG_QUERY", query.c_str(), 1);
      ::setenv("SIRIUS_HIVE_WATCHDOG_OUTPUT", output_path.string().c_str(), 1);
      ::setenv("SIRIUS_HIVE_WATCHDOG_CONFIG", config_path.string().c_str(), 1);
      ::setenv("SIRIUS_CONFIG_FILE", config_path.string().c_str(), 1);
      ::execl("/proc/self/exe",
              "sirius_unittest",
              "gpu_execution hive partition watchdog child runner",
              static_cast<char*>(nullptr));
      ::_exit(127);
    }

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

  fs::path hive_dir;
  fs::path config_path;
  std::string hive_path;
  std::string flat_path;
  bool is_hive_available = false;  // No extension is needed for DuckDB hive partition discovery.
};

class EscapedHivePartitionDataset {
 public:
  EscapedHivePartitionDataset()
  {
    auto const source =
      get_project_root() /
      "test/cpp/integration/data/hive_partitioned/year=2024/month=01/data.parquet";
    REQUIRE(fs::exists(source));

    static std::atomic<std::uint64_t> next_fixture_id{0};
    hive_dir = fs::temp_directory_path() /
               ("sirius_hive_unescape_" + std::to_string(next_fixture_id.fetch_add(1)));
    fs::remove_all(hive_dir);

    copy_partition(source, hive_dir / "space/city=New%20York/data.parquet");
    copy_partition(source, hive_dir / "slash/city=Path%2FTeam/data.parquet");
    copy_partition(source, hive_dir / "percent/city=100%25/data.parquet");
    copy_partition(source, hive_dir / "multiple/city=Los%20Angeles%2FWest/data.parquet");
    copy_partition(source, hive_dir / "two_columns/city=New%20York/dept=R%26D/data.parquet");
    copy_partition(source, hive_dir / "plain/city=Boston/data.parquet");
  }

  ~EscapedHivePartitionDataset() { fs::remove_all(hive_dir); }

  std::string partition_scan(fs::path const& relative_glob) const
  {
    return "read_parquet(" + sql_string_literal((hive_dir / relative_glob).string()) +
           ", hive_partitioning=true)";
  }

 private:
  static void copy_partition(fs::path const& source, fs::path const& target)
  {
    fs::create_directories(target.parent_path());
    fs::copy_file(source, target, fs::copy_options::overwrite_existing);
  }

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

    std::unique_ptr<sirius_config_env_guard> config_guard;
    std::unique_ptr<duckdb::DuckDB> db;
    std::unique_ptr<duckdb::Connection> con;
    if (out.error.empty()) {
      config_guard = std::make_unique<sirius_config_env_guard>(config_raw);
      unsetenv("SIRIUS_DISABLE");
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
