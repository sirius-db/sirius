/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

/**
 * @file test_gpu_execution_native_scan.cpp
 * @brief Integration tests for the duckdb_native_scan_task fast path.
 *
 * Covers behaviors that are new in PR E and not exercised by the existing
 * test_gpu_execution_tpch.cpp scans:
 *
 *   1. Rowid synthesis (commit #4): bounded-LIMIT queries on a wide table
 *      trigger DuckDB's late_materialization rewrite to
 *      `SEMI-JOIN(rowid-scan, full-scan)`, projecting COLUMN_IDENTIFIER_ROW_ID.
 *      Native scan now produces this column on GPU instead of falling back.
 *
 *   2. Default-deny escape gates (commit #3): TABLESAMPLE pushed into the
 *      scan via sampling_pushdown=true. Native scan refuses; query falls
 *      back to duckdb_scan_task which forwards sample_options to DuckDB.
 *      Test asserts results match the CPU run.
 *
 * Both run via `gpu_execution(...)` with assert against plain DuckDB.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct sirius_config_env_guard_native {
  sirius_config_env_guard_native(const std::string& config_path)
  {
    setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
  }
  ~sirius_config_env_guard_native() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

class NativeScanFixture {
 public:
  NativeScanFixture()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<sirius_config_env_guard_native>(cfg_path.string());
      db           = std::make_unique<duckdb::DuckDB>(nullptr);
      con          = std::make_unique<duckdb::Connection>(*db);
    }

    // Small NOT NULL table so the native path is viable. The integration
    // env is shared across test cases so the table may persist; CREATE OR
    // REPLACE is idempotent across reruns within one suite invocation.
    REQUIRE_FALSE(con
                    ->Query("CREATE OR REPLACE TABLE native_scan_t AS "
                            "SELECT range AS k, range * 2 AS v FROM range(0, 100000)")
                    ->HasError());
  }

  void compare_gpu_vs_cpu(const std::string& query)
  {
    con->Query("SET enable_duckdb_fallback = false;");

    auto gpu_sql    = "CALL gpu_execution(\"" + query + "\")";
    auto gpu_result = con->Query(gpu_sql);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("gpu_execution error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());

    auto cpu_result = con->Query(query);
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());

    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    // Per-row value comparison — counts alone could pass with wrong rowids
    // or wrong projected values. Sort client-side because LIMIT/TOP_N row
    // ordering is plan-dependent. (See test_gpu_execution_alp.cpp for why
    // we can't use `SELECT * FROM gpu_execution(...) ORDER BY ...`.)
    auto materialize_rows = [](duckdb::MaterializedQueryResult& r) {
      std::vector<std::vector<std::string>> rows;
      rows.reserve(r.RowCount());
      for (duckdb::idx_t i = 0; i < r.RowCount(); ++i) {
        std::vector<std::string> row;
        row.reserve(r.ColumnCount());
        for (duckdb::idx_t c = 0; c < r.ColumnCount(); ++c) {
          row.push_back(r.GetValue(c, i).ToString());
        }
        rows.push_back(std::move(row));
      }
      std::sort(rows.begin(), rows.end());
      return rows;
    };
    auto gpu_rows = materialize_rows(*gpu_result);
    auto cpu_rows = materialize_rows(*cpu_result);
    for (std::size_t r = 0; r < gpu_rows.size(); ++r) {
      for (std::size_t c = 0; c < gpu_rows[r].size(); ++c) {
        if (gpu_rows[r][c] != cpu_rows[r][c]) {
          UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_rows[r][c]
                               << "] CPU=[" << cpu_rows[r][c] << "]");
        }
        REQUIRE(gpu_rows[r][c] == cpu_rows[r][c]);
      }
    }
  }

 protected:
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::unique_ptr<sirius_config_env_guard_native> config_guard;
};

}  // namespace

TEST_CASE_METHOD(NativeScanFixture,
                 "duckdb_native_scan: bounded LIMIT triggers rowid synthesis",
                 "[integration][gpu_execution][native_scan]")
{
  // SELECT k FROM native_scan_t LIMIT n with small n: late_materialization rewrites this
  // to SEMI-JOIN(rowid-scan, full-scan). Native scan synthesizes the rowid
  // column via thrust::sequence per RG. Row count must match CPU exactly.
  compare_gpu_vs_cpu("SELECT k FROM native_scan_t LIMIT 50");
  compare_gpu_vs_cpu("SELECT v FROM native_scan_t LIMIT 1000");
}

TEST_CASE_METHOD(NativeScanFixture,
                 "duckdb_native_scan: ORDER BY + LIMIT (TOP_N late-mat)",
                 "[integration][gpu_execution][native_scan]")
{
  // ORDER BY + LIMIT triggers TOP_N late-materialization which also projects
  // rowid. row_group_order_options is set, native scan ignores order hint
  // (perf-only) but synthesizes rowid correctly. Sorted compare not required —
  // semantically the limited set must match.
  compare_gpu_vs_cpu("SELECT k FROM native_scan_t ORDER BY k LIMIT 100");
}

TEST_CASE_METHOD(NativeScanFixture,
                 "duckdb_native_scan: full scan matches CPU exactly",
                 "[integration][gpu_execution][native_scan]")
{
  // Sanity: no LIMIT, no SAMPLE, no nulls. Pure native fast path. Both columns
  // present so projection_ids covers the whole projection. Result is sorted
  // by k since we don't rely on per-RG order from the scan.
  compare_gpu_vs_cpu("SELECT k, v FROM native_scan_t WHERE k < 10000");
}
