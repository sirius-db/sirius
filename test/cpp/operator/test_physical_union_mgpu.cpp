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

// Per-operator MGPU integration test for UNION ALL (physical_union).
//
// UNION ALL is the one wrapper-fed operator that is deliberately NOT
// partition-based. Its arms are joined by PASSTHROUGH_SINK, which forwards a
// batch unpartitioned: the `pipelineable_operator_data` it hands on carries no
// `partition_idx`, so the `partition_idx % num_gpus` pin every other MGPU
// operator relies on never fires. Placement instead falls to the locality
// block in task_creator.cpp:450-475, which reads the memory space of the
// inbound batch and picks the GPU already holding the most bytes.
//
// That makes `hash_partition_bytes` — the lever test_physical_order_mgpu.cpp
// and test_physical_grouped_aggregate_merge_mgpu.cpp pull to force
// multi-partition execution — the wrong knob here. The lever is
// `scan_task_batch_size`: small enough that a wide arm produces many batches
// while a narrow one produces few, which is the unequal-arm shape the
// starved-arm task hint exists to serve.
//
// The shared integration configs cannot express that shape. Both
// integration.yaml:22 and integration-2gpu.yaml pin scan_task_batch_size at
// 100 MB, so every fixture table is one batch per arm and no suite run has
// ever driven an arm past a single round. These TEST_CASEs generate their own
// yaml via write_mgpu_yaml instead, which is why they change no config file.
//
//   1. Unequal arms drain across many batches — SINGLE GPU. The wide arm
//      produces many scan batches, the narrow one a single batch. Correctness
//      against a CPU oracle. This is the first fixture that exercises the
//      starved-arm hint across more than one round per arm.
//   2. Three arms of descending width — SINGLE GPU. Drives `_arm_cursor` /
//      `_wait_cursor` past arm 0, which the 100 MB fixtures cannot.
//   3. Balanced arms distribute across two GPUs — needs 2 GPUs.
//   4. Unequal arms do not strand work on one GPU — needs 2 GPUs. The
//      stranding failure the hint exists to prevent.
//
// TEST_CASEs 1 and 2 carry [single-gpu] and no [mgpu] tag, so they run on a
// single-GPU host; 3 and 4 carry [mgpu] and gate on require_two_gpus().
//
// NOT asserted here: zero cross-device migration — that a UNION task ran on
// the GPU that produced its input batch. `tasks_executed` per executor proves
// distribution, not locality, and the suite has no mechanism for the stronger
// property. Demonstrating it needs either an nsys run or a negative control
// that forces `partition_idx == 0` onto the sink, and the latter would mean a
// behaviour hook in production code. See the union-all note set's
// gpu-handoff.md §2.

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
#include <vector>

namespace fs = std::filesystem;
using namespace sirius::test::mgpu;

namespace {

// 1 MB is below the per-file size of every surface here, so the coalescer
// gives each file its own batch: 8 batches for the wide arm, 2 for the middle
// one, 1 for the narrow one. That is the asymmetry the starved-arm hint exists
// to serve, at a row count that keeps the CPU-oracle comparison cheap enough
// for CI. Mirrors test/cpp/scan_manager/test_pin_table_multi_gpu.cpp:139.
// hash_partition_bytes is left at its default: UNION ALL does not partition,
// so shrinking it would only add noise from the scan side.
constexpr uint64_t kSmallScanBatchBytes = 1'000'000;

mgpu_env_params make_params(int num_gpus)
{
  mgpu_env_params p;
  p.cache                    = "none";
  p.num_gpus                 = num_gpus;
  p.scan_task_batch_size     = kSmallScanBatchBytes;
  p.pipeline_num_threads     = 4;
  p.task_creator_num_threads = 4;
  return p;
}

fs::path make_tmp_dir(std::string const& tag)
{
  auto tmp =
    fs::temp_directory_path() / ("sirius-mgpu-union-" + std::to_string(::getpid()) + "-" + tag);
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  return tmp;
}

// 8 files x 100k rows x 16 B ~ 12.8 MB, so each ~1.6 MB file is its own batch
// at kSmallScanBatchBytes and the arm needs eight rounds to drain.
void generate_wide_arm(fs::path const& dir)
{
  generate_parquet_surface(dir, "SELECT range AS k, range * 2 AS v FROM range(100000)", 8);
}

// One file, 1k rows — a single batch however the coalescer is configured.
void generate_narrow_arm(fs::path const& dir)
{
  generate_parquet_surface(dir, "SELECT range AS k, range * 3 AS v FROM range(1000)", 1);
}

// A middle arm, wide enough for a handful of batches but well short of the
// wide arm, so a three-arm union has three distinct drain lengths.
void generate_middle_arm(fs::path const& dir)
{
  generate_parquet_surface(dir, "SELECT range AS k, range * 5 AS v FROM range(200000)", 2);
}

std::string union_of(std::vector<fs::path> const& dirs)
{
  std::string sql;
  for (size_t i = 0; i < dirs.size(); ++i) {
    if (i != 0) sql += " UNION ALL ";
    sql += "SELECT k, v FROM read_parquet('" + parquet_glob(dirs[i]) + "')";
  }
  return sql;
}

// Run `query` under a generated env and return tasks_executed per device.
std::map<int, size_t> run_and_collect(fs::path const& yaml, std::string const& query)
{
  std::map<int, size_t> tasks_per_gpu;
  scoped_mgpu_env env(yaml);
  auto con = std::make_unique<duckdb::Connection>(env.make_connection());
  require_gpu_matches_cpu(*con, query, /*force_cpu_reference=*/true);
  auto& scheduler = env.get_task_scheduler(*con);
  con.reset();  // flush sinks before reading metrics

  scheduler.visit_executors(
    [&](int device_id, sirius::pipeline::gpu_pipeline_executor const& exec) {
      tasks_per_gpu[device_id] = exec.get_metrics().tasks_executed;
    });
  return tasks_per_gpu;
}

}  // namespace

TEST_CASE("physical_union - unequal arms drain across many batches",
          "[single-gpu][operator-mgpu][union_all][gpu_execution]")
{
  auto tmp  = make_tmp_dir("unequal-1gpu");
  auto yaml = tmp / "mgpu.yaml";
  write_mgpu_yaml(yaml, make_params(/*num_gpus=*/1));
  REQUIRE(fs::exists(yaml));

  auto wide   = tmp / "wide";
  auto narrow = tmp / "narrow";
  generate_wide_arm(wide);
  generate_narrow_arm(narrow);

  // 8 x 100k + 1 x 1k. The arms are bag-unioned, so every row of both survives.
  constexpr size_t kExpectedRows = 8 * 100000 + 1000;

  auto query = union_of({wide, narrow});
  {
    scoped_mgpu_env env(yaml);
    auto con = env.make_connection();
    require_gpu_matches_cpu(con, query, /*force_cpu_reference=*/true);

    auto rows = con.Query("SELECT count(*) FROM (" + query + ") t;");
    REQUIRE(rows);
    REQUIRE_FALSE(rows->HasError());
    REQUIRE(rows->GetValue(0, 0).GetValue<int64_t>() == static_cast<int64_t>(kExpectedRows));
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("physical_union - three arms of descending width",
          "[single-gpu][operator-mgpu][union_all][gpu_execution]")
{
  auto tmp  = make_tmp_dir("three-arm-1gpu");
  auto yaml = tmp / "mgpu.yaml";
  write_mgpu_yaml(yaml, make_params(/*num_gpus=*/1));
  REQUIRE(fs::exists(yaml));

  auto wide   = tmp / "wide";
  auto middle = tmp / "middle";
  auto narrow = tmp / "narrow";
  generate_wide_arm(wide);
  generate_middle_arm(middle);
  generate_narrow_arm(narrow);

  constexpr size_t kExpectedRows = 8 * 100000 + 2 * 200000 + 1000;

  auto query = union_of({wide, middle, narrow});
  {
    scoped_mgpu_env env(yaml);
    auto con = env.make_connection();
    require_gpu_matches_cpu(con, query, /*force_cpu_reference=*/true);

    auto rows = con.Query("SELECT count(*) FROM (" + query + ") t;");
    REQUIRE(rows);
    REQUIRE_FALSE(rows->HasError());
    REQUIRE(rows->GetValue(0, 0).GetValue<int64_t>() == static_cast<int64_t>(kExpectedRows));
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("physical_union - balanced arms distribute across two GPUs",
          "[mgpu][operator-mgpu][union_all][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("balanced-2gpu");
  auto yaml = tmp / "mgpu.yaml";
  write_mgpu_yaml(yaml, make_params(/*num_gpus=*/2));
  REQUIRE(fs::exists(yaml));

  auto left  = tmp / "left";
  auto right = tmp / "right";
  generate_wide_arm(left);
  generate_wide_arm(right);

  auto tasks_per_gpu = run_and_collect(yaml, union_of({left, right}));

  INFO("gpu0 tasks=" << tasks_per_gpu[0] << " gpu1 tasks=" << tasks_per_gpu[1]);
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("physical_union - unequal arms do not strand work on one GPU",
          "[mgpu][operator-mgpu][union_all][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("unequal-2gpu");
  auto yaml = tmp / "mgpu.yaml";
  write_mgpu_yaml(yaml, make_params(/*num_gpus=*/2));
  REQUIRE(fs::exists(yaml));

  // The asymmetry is the point: the narrow arm finishes in one round while the
  // wide arm is still draining. If the starved-arm hint stopped nominating the
  // wide arm, its remaining batches would strand on whichever GPU took the
  // first one.
  auto wide   = tmp / "wide";
  auto narrow = tmp / "narrow";
  generate_wide_arm(wide);
  generate_narrow_arm(narrow);

  auto tasks_per_gpu = run_and_collect(yaml, union_of({wide, narrow}));

  INFO("gpu0 tasks=" << tasks_per_gpu[0] << " gpu1 tasks=" << tasks_per_gpu[1]);
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  std::error_code ec;
  fs::remove_all(tmp, ec);
}
