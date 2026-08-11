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

#pragma once

// Fixture scaffolding shared by the compressed-materialization integration tests
// (test_compressed_materialization_gate.cpp and
// test_compressed_materialization_partition.cpp). Each test file keeps its own value recipes,
// fixture constants, and configuration values; this header holds only the mechanics they share:
// result checking, parquet generation, YAML configuration, shared-env pausing, and GPU-vs-CPU
// result comparison. Everything is defined inline in a named namespace because two translation
// units include this header.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace sirius::test::compmat {

inline void require_ok(duckdb::unique_ptr<duckdb::MaterializedQueryResult> const& r,
                       char const* what)
{
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO(what << " error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
}

/// Write @p select_sql 's result to @p path as parquet with @p row_group_size rows per row group,
/// using a throwaway DuckDB instance with the Sirius extension disabled so generation never
/// touches the GPU under test.
inline void generate_parquet(std::filesystem::path const& path,
                             std::string const& select_sql,
                             std::size_t row_group_size)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB gen_db(nullptr);
  duckdb::Connection gen(gen_db);
  auto r = gen.Query("COPY (" + select_sql + ") TO " + sirius::test::sql_literal(path.string()) +
                     " (FORMAT PARQUET, ROW_GROUP_SIZE " + std::to_string(row_group_size) + ");");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
}

/// Operator parameters that differ between the compressed-materialization fixtures; every other
/// configuration value is shared verbatim by write_config.
struct config_values {
  std::size_t scan_batch_bytes;
  std::size_t hash_partition_bytes;
  std::size_t concat_batch_bytes;
  /// Per-partition build-bytes cap for hash-join BUILD_PROBE eligibility. A build side at or over
  /// this stays on the STANDARD partitioned join path; fixtures that must observe an engaged hash
  /// partition set it below their build size so a small build cannot elect BUILD_PROBE/broadcast.
  std::size_t max_build_hash_table_bytes = 90000000;
};

inline void write_config(std::filesystem::path const& yaml_path, config_values const& values)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_bytes: "
    << (2ull << 30)
    << "\n"
       "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 32000000000\n"
       "      initial_number_pools: 10\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 4\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period: 10ms\n"
       "  operator_params:\n"
       "    scan_task_batch_size: "
    << values.scan_batch_bytes
    << "\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: "
    << values.hash_partition_bytes
    << "\n"
       "    concat_batch_bytes: "
    << values.concat_batch_bytes
    << "\n"
       "    max_build_hash_table_bytes: "
    << values.max_build_hash_table_bytes << "\n";
}

/// Pause whichever shared test environments are active so a test-local SiriusContext owns the
/// GPU.
inline void pause_shared_envs()
{
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }
}

/// Run @p query on the GPU (a failure surfaces loudly: duckdb fallback is disabled by every
/// caller), then on the CPU, and compare the two result sets as sorted stringified rows (robust
/// to result-order ties).
inline void compare_gpu_vs_cpu(duckdb::Connection& con, std::string const& query)
{
  auto gpu_result = con.Query(query);
  require_ok(gpu_result, "gpu query");

  require_ok(con.Query("SET gpu_execution = false;"), "disable gpu");
  auto cpu_result = con.Query(query);
  require_ok(cpu_result, "cpu query");
  require_ok(con.Query("SET gpu_execution = true;"), "re-enable gpu");

  REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
  REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

  auto collect_rows = [](duckdb::MaterializedQueryResult& result) {
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
  };
  auto gpu_rows = collect_rows(*gpu_result);
  auto cpu_rows = collect_rows(*cpu_result);
  for (std::size_t r = 0; r < gpu_rows.size(); r++) {
    for (std::size_t c = 0; c < gpu_rows[r].size(); c++) {
      if (gpu_rows[r][c] != cpu_rows[r][c]) {
        UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_rows[r][c]
                             << "] CPU=[" << cpu_rows[r][c] << "]");
      }
      REQUIRE(gpu_rows[r][c] == cpu_rows[r][c]);
    }
  }
}

}  // namespace sirius::test::compmat
