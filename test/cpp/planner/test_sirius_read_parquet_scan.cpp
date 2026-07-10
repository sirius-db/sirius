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

// Regression test: the tree-based pipeline build (USE_TREE_BASED_PIPELINE_BUILD, the
// default) must plan `sirius_read_parquet` — the internal rewrite target for
// read_parquet('s3://…') inside gpu_execution — like the legacy converter does.
// Before the fix, wrap_table_scan_source threw "Unsupported scan function:
// sirius_read_parquet", so S3 SQL failed during planning (and S3 has no CPU fallback;
// the CPU-side table function only throws). The scan is URI-agnostic — the resolved
// URI travels in parameters[0] — so a local parquet file exercises the exact same plan
// path without an S3 harness. A passing query also proves the GPU path served it: the
// DuckDB-side execute callback for this function unconditionally throws.

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kRows = 100'000;

// Throwaway, Sirius-disabled DuckDB writes the parquet so the extension callback does
// not build a SiriusContext on it.
void generate_parquet(fs::path const& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r = gen.Query("COPY (SELECT range AS k, range * 2 AS v FROM range(" +
                       std::to_string(kRows) + ")) TO '" + path.string() + "' (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_fraction: 0.4\n"
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
       "    scan_task_batch_size: 100000000\n"
       "    default_scan_task_varchar_size: 256\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 100000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

}  // namespace

// NB: no [integration]/[shared_context] tag — this TEST_CASE builds its own SiriusContext
// and manages (pauses) the shared envs itself, mirroring the isolated-context pin tests.
TEST_CASE("tree pipeline build plans sirius_read_parquet scans",
          "[planner][sirius_read_parquet][tree_pipeline]")
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

  auto tmp = fs::temp_directory_path() / ("sirius-srp-tree-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto parquet_path = tmp / "kv.parquet";
  generate_parquet(parquet_path);

  auto yaml_path = tmp / "srp_tree.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // A planning failure must surface as an error, not a CPU replay (which would throw
    // the function's "internal rewrite target" error anyway).
    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    auto res = con.Query("SELECT max(k), count(*) FROM sirius_read_parquet('" +
                         parquet_path.string() + "');");
    REQUIRE(res);
    if (res->HasError()) { UNSCOPED_INFO("sirius_read_parquet query error: " << res->GetError()); }
    REQUIRE_FALSE(res->HasError());  // pre-fix: "Unsupported scan function: sirius_read_parquet"
    REQUIRE(res->GetValue(0, 0).GetValue<int64_t>() == kRows - 1);
    REQUIRE(res->GetValue(1, 0).GetValue<int64_t>() == kRows);
  }

  fs::remove_all(tmp, ec);
}
