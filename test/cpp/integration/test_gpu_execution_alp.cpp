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
 * @file test_gpu_execution_alp.cpp
 * @brief Integration tests for ALP and ALPRD floating-point decoders in gpu_execution.
 *
 * Creates temporary DuckDB files with columns compressed using ALP (Adaptive
 * Lossless floating-Point) and ALPRD (ALP Real Doubles) codecs, then verifies
 * that GPU scan results match CPU scan results for all combinations of codec
 * and type (FLOAT, DOUBLE) including NULL-handling.
 */

#include "op/sirius_physical_partition.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static fs::path get_project_root_alp()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT);
#else
  return fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
#endif
}

struct sirius_config_env_guard_alp {
  sirius_config_env_guard_alp(const std::string& config_path)
  {
    setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
  }
  ~sirius_config_env_guard_alp() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

/**
 * @brief Base fixture providing compare_gpu_vs_cpu — duplicated from multi_format
 * to keep this file self-contained and independently compilable.
 */
class ALPFixtureBase {
 public:
  ALPFixtureBase()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<sirius_config_env_guard_alp>(cfg_path.string());
      db           = std::make_unique<duckdb::DuckDB>(nullptr);
      con          = std::make_unique<duckdb::Connection>(*db);
    }
  }

  static bool is_floating_point(duckdb::LogicalTypeId id)
  {
    return id == duckdb::LogicalTypeId::FLOAT || id == duckdb::LogicalTypeId::DOUBLE;
  }

  void compare_gpu_vs_cpu(const std::string& query,
                          std::optional<float> float_tolerance = std::nullopt)
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

    // Sort both result sets client-side. We can't use
    // `SELECT * FROM gpu_execution(...) ORDER BY ...` because Sirius's
    // pipeline converter doesn't recognize gpu_execution as a scan source
    // when wrapped in an outer query — that path throws "Unsupported scan
    // function: gpu_execution" from sirius_pipeline_converter.cpp.
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

    for (std::size_t r = 0; r < gpu_rows.size(); r++) {
      for (std::size_t c = 0; c < gpu_rows[r].size(); c++) {
        auto& gpu_str = gpu_rows[r][c];
        auto& cpu_str = cpu_rows[r][c];
        bool is_float = float_tolerance.has_value() && is_floating_point(gpu_result->types[c].id());

        if (is_float) {
          if (gpu_str == "NULL" || cpu_str == "NULL") {
            REQUIRE(gpu_str == cpu_str);
            continue;
          }
          double gpu_d = std::stod(gpu_str);
          double cpu_d = std::stod(cpu_str);
          double diff  = std::fabs(gpu_d - cpu_d);
          if (diff > static_cast<double>(float_tolerance.value())) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " float mismatch: GPU=[" << gpu_d
                                 << "] CPU=[" << cpu_d << "] diff=" << diff);
          }
          REQUIRE(diff <= static_cast<double>(float_tolerance.value()));
        } else {
          if (gpu_str != cpu_str) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_str << "] CPU=["
                                 << cpu_str << "]");
          }
          REQUIRE(gpu_str == cpu_str);
        }
      }
    }
  }

  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::unique_ptr<sirius_config_env_guard_alp> config_guard;
};

/**
 * @brief Fixture that creates a temp DuckDB file with ALP / ALPRD compressed
 * floating-point columns and attaches it to the Sirius integration connection.
 *
 * Tables (each 10 240 rows):
 *   alp_doubles  — DOUBLE column compressed with ALP
 *   alp_floats   — FLOAT  column compressed with ALP
 *   alprd_doubles — DOUBLE column compressed with ALPRD
 *   alprd_floats  — FLOAT  column compressed with ALPRD
 *   alp_nulls    — DOUBLE column (ALP) with every 10th value NULL
 */
class GPUExecutionALPFixture : public ALPFixtureBase {
 public:
  GPUExecutionALPFixture()
  {
    static std::atomic<int> counter{0};
    temp_path = fs::temp_directory_path() / ("sirius_alp_" + std::to_string(counter++) + ".duckdb");

    build_database();

    auto attach = con->Query("ATTACH '" + temp_path.string() + "' AS alp_db (READ_ONLY)");
    if (!attach || attach->HasError()) {
      auto msg = attach ? attach->GetError() : std::string("null result");
      FAIL("Failed to attach ALP test database: " + msg);
    }
  }

  ~GPUExecutionALPFixture()
  {
    con->Query("DETACH alp_db");
    fs::remove(temp_path);
    fs::remove(fs::path(temp_path.string() + ".wal"));
  }

 private:
  void exec(duckdb::Connection& c, const std::string& sql)
  {
    auto r = c.Query(sql);
    if (!r || r->HasError()) {
      throw std::runtime_error("ALP fixture setup failed [" + sql.substr(0, 60) +
                               "]: " + (r ? r->GetError() : "null"));
    }
  }

  void build_database()
  {
    duckdb::DuckDB plain_db(temp_path.string());
    duckdb::Connection c(plain_db);

    exec(c, "SET force_compression = 'alp'");
    exec(c,
         "CREATE TABLE alp_doubles AS "
         "SELECT i::INTEGER AS id, "
         "       (i * 1234567.89012 + 0.123456)::DOUBLE AS val "
         "FROM range(10240) t(i)");
    exec(c,
         "CREATE TABLE alp_floats AS "
         "SELECT i::INTEGER AS id, "
         "       (i * 1234.56789 + 0.12345)::FLOAT AS val "
         "FROM range(10240) t(i)");
    exec(c,
         "CREATE TABLE alp_nulls AS "
         "SELECT i::INTEGER AS id, "
         "       CASE WHEN i % 10 = 0 THEN NULL "
         "            ELSE (i * 999.12345 + 0.001)::DOUBLE END AS val "
         "FROM range(10240) t(i)");

    exec(c, "SET force_compression = 'alprd'");
    exec(c,
         "CREATE TABLE alprd_doubles AS "
         "SELECT i::INTEGER AS id, "
         "       ((i % 7 + 1) * 1000003.0 / (i % 11 + 1))::DOUBLE AS val "
         "FROM range(10240) t(i)");
    exec(c,
         "CREATE TABLE alprd_floats AS "
         "SELECT i::INTEGER AS id, "
         "       ((i % 7 + 1) * 1000.003 / (i % 11 + 1))::FLOAT AS val "
         "FROM range(10240) t(i)");

    exec(c, "CHECKPOINT");
  }

  fs::path temp_path;
};

//===----------------------------------------------------------------------===//
// ALP DOUBLE tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alp - double sum",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT SUM(val) FROM alp_db.alp_doubles", 1e-4f);
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alp - double count",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT COUNT(val) FROM alp_db.alp_doubles");
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alp - double filter",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM alp_db.alp_doubles WHERE val > 5000000.0");
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alp - double row scan",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT id, val FROM alp_db.alp_doubles WHERE id < 50", 1e-6f);
}

//===----------------------------------------------------------------------===//
// ALP FLOAT tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alp - float sum",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT SUM(val) FROM alp_db.alp_floats", 1e-1f);
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alp - float count",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT COUNT(val) FROM alp_db.alp_floats");
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alp - float row scan",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT id, val FROM alp_db.alp_floats WHERE id < 50", 1e-4f);
}

//===----------------------------------------------------------------------===//
// ALPRD DOUBLE tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alprd - double sum",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT SUM(val) FROM alp_db.alprd_doubles", 1e-4f);
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alprd - double count",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT COUNT(val) FROM alp_db.alprd_doubles");
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alprd - double filter",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM alp_db.alprd_doubles WHERE val > 500000.0");
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alprd - double row scan",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT id, val FROM alp_db.alprd_doubles WHERE id < 50", 1e-6f);
}

//===----------------------------------------------------------------------===//
// ALPRD FLOAT tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alprd - float sum",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT SUM(val) FROM alp_db.alprd_floats", 1e-1f);
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alprd - float count",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT COUNT(val) FROM alp_db.alprd_floats");
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alprd - float row scan",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT id, val FROM alp_db.alprd_floats WHERE id < 50", 1e-4f);
}

//===----------------------------------------------------------------------===//
// NULL handling
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alp - null count",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT COUNT(*) AS total, COUNT(val) AS non_null FROM alp_db.alp_nulls");
}

TEST_CASE_METHOD(GPUExecutionALPFixture,
                 "gpu_execution alp - null sum",
                 "[integration][gpu_execution][alp]")
{
  compare_gpu_vs_cpu("SELECT SUM(val) FROM alp_db.alp_nulls", 1e-4f);
}
