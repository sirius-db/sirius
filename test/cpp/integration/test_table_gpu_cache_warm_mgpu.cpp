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

// Phase 8 follow-up #17: table_gpu cache warm cross-GPU hazard.
//
// Stand-alone repro for the SF100 parquet + cache=table_gpu + num_gpus=2
// failure observed at Q11 warm. Hypothesis: on the cold pass, the first
// downstream consumer mutates a cached data_batch in place via
// lock_or_prepare_batch (batch_lock_utils.hpp:87). The cache stores the
// same shared_ptr; on warm, task_creator's SCHED-00/01/02 can route the
// next consumer to a different GPU, triggering a cross-device p2p
// conversion through cucascade's convert_gpu_to_gpu. If any step in that
// chain dereferences a pointer while the current device is wrong, the
// next stream sync surfaces as cudaErrorIllegalAddress at
// sirius_physical_operator.cpp:59.
//
// Test surface is deliberately small so it can run in the unit-test
// harness: 8 parquet files × 500k rows (>= the 32 MiB multi-partition
// floor `num_gpus * 16 MiB`), a self hash-join that exercises
// sirius_physical_partition's MGPU path, then repeated gpu_execution
// invocations so the cache_scan_results_for_query hash match triggers
// preload_mode on every call after the first.

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace {

/**
 * @brief Write a dedicated Sirius YAML for the follow-up #17 repro.
 *
 * Keep aligned with integration-2gpu.yaml except for the two knobs
 * that actually drive the hazard:
 *   - operator_params.hash_partition_bytes: 1 MiB (force partitioning
 *     of the ~64 MiB self-join working set so sirius_physical_partition's
 *     MGPU floor of `num_gpus` partitions kicks in).
 */
void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 2\n"
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
       "      monitor_period_ms: 10\n"
       "  operator_params:\n"
       "    scan_task_batch_size: 100000000\n"
       "    default_scan_task_varchar_size: 256\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 1000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

/**
 * @brief Generate 8 parquet files using a vanilla DuckDB instance.
 *
 * SIRIUS_DISABLE=1 is set around the generator to keep the extension
 * callback from building a SiriusContext on this throwaway DuckDB —
 * the real Sirius instance is built later from our yaml.
 */
void generate_parquet_surface(fs::path const& tmp_dir, int num_files, int rows_per_file)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);

    auto r = gen.Query("CREATE TABLE t AS SELECT range AS k, range * 2 AS v FROM range(" +
                       std::to_string(rows_per_file) + ");");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());

    for (int i = 0; i < num_files; ++i) {
      auto path = tmp_dir / ("p" + std::to_string(i) + ".parquet");
      auto copy = gen.Query("COPY t TO '" + path.string() + "' (FORMAT PARQUET);");
      REQUIRE(copy);
      REQUIRE_FALSE(copy->HasError());
    }
  }
  unsetenv("SIRIUS_DISABLE");
}

}  // namespace

TEST_CASE("gpu_execution - table_gpu cache warm cross-GPU hazard (follow-up #17)",
          "[mgpu][followup-17][gpu_execution]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN(
      "follow-up #17 repro requires >=2 GPUs; single-GPU host — skipping "
      "(per Catch2 v2 WARN+return convention)");
    return;
  }

  // The Catch2 listener in unittest.cpp pauses shared envs for TEST_CASEs
  // without [shared_context] or [integration] tags. Be defensive: walk the
  // known envs and pause any that are still holding the extension lock so
  // our local env is the only Sirius instance in the process.
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }

  auto tmp = fs::temp_directory_path() / ("sirius-fu17-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  // 8 files × 500k rows × 16 B/row ≈ 64 MiB total — comfortably above the
  // 32 MiB (num_gpus * 16 MiB) multi-partition floor so the hash-join
  // partitioner produces at least 2 partitions routed across both GPUs.
  constexpr int kNumFiles    = 8;
  constexpr int kRowsPerFile = 500000;
  generate_parquet_surface(tmp, kNumFiles, kRowsPerFile);

  auto yaml_path = tmp / "fu17.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  // shared_test_env's constructor creates the DuckDB + SiriusContext from
  // SIRIUS_CONFIG_FILE, which it sets to our yaml.
  sirius::test::shared_test_env local_env(yaml_path);

  auto glob = (tmp / "*.parquet").string();
  auto inner_query =
    "SELECT t1.k, count(*) "
    "FROM read_parquet('" +
    glob +
    "') t1 "
    "JOIN read_parquet('" +
    glob +
    "') t2 ON t1.k = t2.v "
    "GROUP BY t1.k "
    "ORDER BY t1.k "
    "LIMIT 10;";
  auto gpu_sql = "CALL gpu_execution(\"" + inner_query + "\");";

  // One cold pass populates the cache; N warm passes replay it. Each warm
  // pass is an independent chance for task_creator to route a downstream
  // consumer onto a GPU different from the one where the cold pass
  // mutated the cached batch — so running multiple warm passes raises the
  // probability of tripping the hazard when it is present.
  constexpr int kWarmIterations = 5;

  std::string cold_error;
  std::vector<std::string> warm_errors(kWarmIterations);
  bool cold_ok = false;
  std::vector<bool> warm_ok(kWarmIterations, false);
  {
    auto con = std::make_unique<duckdb::Connection>(local_env.make_connection());
    auto fb  = con->Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    auto r_cold = con->Query(gpu_sql);
    REQUIRE(r_cold);
    if (r_cold->HasError()) {
      cold_error = r_cold->GetError();
    } else {
      cold_ok = true;
    }

    for (int i = 0; i < kWarmIterations; ++i) {
      auto r_warm = con->Query(gpu_sql);
      REQUIRE(r_warm);
      if (r_warm->HasError()) {
        warm_errors[i] = r_warm->GetError();
      } else {
        warm_ok[i] = true;
      }
    }
  }

  INFO("cold error: " << cold_error);
  for (int i = 0; i < kWarmIterations; ++i) {
    INFO("warm[" << i << "] error: " << warm_errors[i]);
  }

  // Cold must pass for the warm assertions to be meaningful.
  REQUIRE(cold_ok);

  // Any warm failure with cudaErrorIllegalAddress / reduce failed to
  // synchronize / pipelineable_operator_data::prepare_for_processing is
  // the hazard the follow-up tracks. This TEST_CASE will fail (revealing
  // the hazard) while the bug is open and pass (locking in the fix) once
  // resolved.
  for (int i = 0; i < kWarmIterations; ++i) {
    REQUIRE(warm_ok[i]);
  }

  fs::remove_all(tmp, ec);
}
