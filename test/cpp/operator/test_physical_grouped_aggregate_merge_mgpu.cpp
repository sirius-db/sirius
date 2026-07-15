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

// Per-operator MGPU integration test for grouped_aggregate_merge.
//
// grouped_aggregate_merge is partition-based (like hash_join) — it builds a
// cuco GPU hash table per partition. cuco tables cannot span GPUs, so every
// task for a given partition must pin to the same GPU via
// `partition_idx % num_gpus` in task_creator.cpp. These TEST_CASEs exercise
// the cross-GPU routing path in isolation:
//
//   1. High-cardinality GROUP BY with hash_partition_bytes small enough to
//      force multi-partition execution. The `num_gpus` floor in
//      sirius_physical_partition.cpp:242 guarantees >=2 partitions, so both
//      GPUs should see pipeline_task dispatches.
//   2. Single-key GROUP BY (all k=0). Tiny aggregate — small-table-bytes
//      path skips the floor; just a correctness check that the one-partition
//      grouped_aggregate_merge still works under num_gpus=2.
//   3. COUNT(*)-only GROUP BY across both GPUs — sanity check that the
//      COUNT aggregation-type enum dispatches correctly through the merge
//      path when SUM is not in play.

#include "mgpu_test_utils.hpp"

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <string>

namespace fs = std::filesystem;
using namespace sirius::test::mgpu;

namespace {

constexpr int kNumFiles    = 8;
constexpr int kRowsPerFile = 500000;

mgpu_env_params make_params()
{
  mgpu_env_params p;
  p.cache                    = "none";
  p.hash_partition_bytes     = 1'000'000;  // 1 MiB → force many partitions
  p.pipeline_num_threads     = 4;
  p.task_creator_num_threads = 4;
  return p;
}

fs::path make_tmp_dir(std::string const& tag)
{
  auto tmp =
    fs::temp_directory_path() / ("sirius-mgpu-gagg-" + std::to_string(::getpid()) + "-" + tag);
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  return tmp;
}

}  // namespace

TEST_CASE("grouped_aggregate_merge - group by with high cardinality distributes across both GPUs",
          "[mgpu][operator-mgpu][grouped_aggregate_merge][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("highcard");
  auto yaml = tmp / "mgpu.yaml";

  // Independent keys across files: range(0..500000) each — total ~4M rows
  // with ~500k distinct keys. At ~16 B/row that is ~64 MiB input; with
  // hash_partition_bytes=1 MiB sirius_physical_partition produces many
  // partitions (>= num_gpus floor of 2).
  generate_parquet_surface(
    tmp,
    "SELECT range AS k, range * 2 AS v FROM range(" + std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  auto glob = parquet_glob(tmp);
  auto inner_query =
    "SELECT k, SUM(v) AS sum_v, COUNT(*) AS cnt "
    "FROM read_parquet('" +
    glob +
    "') "
    "GROUP BY k "
    "ORDER BY k "
    "LIMIT 50";

  std::map<int, size_t> tasks_per_gpu;
  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    auto& scheduler = env.get_task_scheduler(*con);
    con.reset();
    scheduler.visit_executors(
      [&](int device_id, const sirius::pipeline::gpu_pipeline_executor& exec) {
        tasks_per_gpu[device_id] = exec.get_metrics().tasks_executed;
      });
  }

  INFO("gpu0 tasks=" << tasks_per_gpu[0] << " gpu1 tasks=" << tasks_per_gpu[1]);
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("grouped_aggregate_merge - group by with single key forces single-GPU path",
          "[mgpu][operator-mgpu][grouped_aggregate_merge][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("singlekey");
  auto yaml = tmp / "mgpu.yaml";

  // Every row shares k=0 → aggregate result is a single group. Below the
  // small-table-bytes threshold where the num_gpus floor is skipped, so
  // only one partition exists and the merge runs on a single GPU. We just
  // assert correctness here, not distribution.
  generate_parquet_surface(
    tmp,
    "SELECT 0 AS k, range * 2 AS v FROM range(" + std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  auto glob = parquet_glob(tmp);
  auto inner_query =
    "SELECT k, SUM(v) AS sum_v, COUNT(*) AS cnt "
    "FROM read_parquet('" +
    glob +
    "') "
    "GROUP BY k "
    "ORDER BY k";

  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    con.reset();
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("grouped_aggregate_merge - count(*)-only aggregate across two GPUs",
          "[mgpu][operator-mgpu][grouped_aggregate_merge][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("countstar");
  auto yaml = tmp / "mgpu.yaml";

  // Same high-cardinality surface as test #1 so we hit the multi-partition
  // MGPU path, but the aggregate is COUNT(*) only — catches cudf
  // aggregation-type enum dispatch issues that SUM would otherwise mask.
  generate_parquet_surface(
    tmp,
    "SELECT range AS k, range * 2 AS v FROM range(" + std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  auto glob = parquet_glob(tmp);
  auto inner_query =
    "SELECT k, COUNT(*) AS cnt "
    "FROM read_parquet('" +
    glob +
    "') "
    "GROUP BY k "
    "ORDER BY k "
    "LIMIT 50";

  std::map<int, size_t> tasks_per_gpu;
  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    auto& scheduler = env.get_task_scheduler(*con);
    con.reset();
    scheduler.visit_executors(
      [&](int device_id, const sirius::pipeline::gpu_pipeline_executor& exec) {
        tasks_per_gpu[device_id] = exec.get_metrics().tasks_executed;
      });
  }

  INFO("gpu0 tasks=" << tasks_per_gpu[0] << " gpu1 tasks=" << tasks_per_gpu[1]);
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  std::error_code ec;
  fs::remove_all(tmp, ec);
}
