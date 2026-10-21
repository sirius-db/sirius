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

// Multi-GPU tests for sirius_physical_nested_loop_join.
//
// A nested-loop join is never hash-partitioned. On several GPUs a small right
// side is replicated to every GPU (one pinned slot each) and every left batch is
// routed to the slot on its own GPU, so no join task copies an input. Before
// #2200 the join ran as one unpinned partition: each task landed where its left
// batch was and cloned the right side there, once per task.

#include "log/logging.hpp"
#include "mgpu_test_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

using sirius::test::mgpu::generate_parquet_surface;
using sirius::test::mgpu::mgpu_env_params;
using sirius::test::mgpu::parquet_glob;
using sirius::test::mgpu::require_gpu_matches_cpu;
using sirius::test::mgpu::require_two_gpus;
using sirius::test::mgpu::scoped_log_dir;
using sirius::test::mgpu::scoped_mgpu_env;
using sirius::test::mgpu::write_mgpu_yaml;

fs::path make_tmp(std::string const& tag)
{
  return fs::temp_directory_path() /
         ("sirius-mgpu-nlj-" + tag + "-" + std::to_string(::getpid()));
}

/// Whether any log file under @p log_dir contains @p needle. The placement decision is only
/// visible in the DEBUG log: a single-GPU fallback returns the same rows.
bool log_dir_contains(fs::path const& log_dir, std::string const& needle)
{
  (void)sirius::log::get_sink()->flush();
  std::error_code ec;
  for (auto const& entry : fs::recursive_directory_iterator(log_dir, ec)) {
    if (!entry.is_regular_file()) { continue; }
    std::ifstream in(entry.path());
    std::stringstream ss;
    ss << in.rdbuf();
    if (ss.str().find(needle) != std::string::npos) { return true; }
  }
  return false;
}

}  // namespace

TEST_CASE("physical_nested_loop_join - small right side broadcasts across two GPUs",
          "[mgpu][operator-mgpu][nested_loop_join][broadcast][gpu_execution][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("broadcast");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto left_dir  = tmp / "left";
  auto right_dir = tmp / "right";
  // Several left files so the scan lands batches on both GPUs; one small right file (a few KiB,
  // far below the 2 * 16 MiB small-table threshold). Each right row matches the two left keys
  // strictly inside its (lo, hi) range.
  generate_parquet_surface(
    left_dir, "SELECT range AS k, range * 7 AS v FROM range(200000)", /*num_files=*/4);
  generate_parquet_surface(
    right_dir, "SELECT range * 40 AS lo, range * 40 + 3 AS hi FROM range(200)", /*num_files=*/1);

  // DEBUG level so partition_exchange logs the layout it resolved.
  scoped_log_dir log_dir(tmp / "logs", "debug");

  mgpu_env_params params{};
  params.cache   = "none";
  auto yaml_path = tmp / "broadcast.yaml";
  write_mgpu_yaml(yaml_path, params);
  scoped_mgpu_env env(yaml_path);

  // Inequalities only, so the planner picks the nested-loop join.
  auto inner_query =
    "SELECT l.k, l.v, r.lo "
    "FROM read_parquet('" +
    parquet_glob(left_dir) +
    "') AS l "
    "JOIN read_parquet('" +
    parquet_glob(right_dir) +
    "') AS r "
    "  ON l.k > r.lo AND l.k < r.hi "
    "ORDER BY l.k, l.v, r.lo "
    "LIMIT 100";

  std::map<int, size_t> tasks_per_gpu;
  {
    auto con = env.make_connection();
    require_gpu_matches_cpu(con, inner_query, /*force_cpu_reference=*/true);
    auto& scheduler = env.get_task_scheduler(con);
    scheduler.visit_executors(
      [&](int device_id, const sirius::pipeline::gpu_pipeline_executor& exec) {
        tasks_per_gpu[device_id] = exec.get_metrics().tasks_executed;
      });
  }

  INFO("log dir: " << log_dir.path());
  // The right side was replicated rather than left to a single unpinned partition.
  REQUIRE(log_dir_contains(log_dir.path(), "2 partitions, mode broadcast"));

  // Each slot joined its own left batches against its local copy of the right side.
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  fs::remove_all(tmp, ec);
}
