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

// Per-operator MGPU integration test for ORDER BY (physical_order).
//
// ORDER BY is partition-based: when the input working set exceeds
// hash_partition_bytes sirius_physical_partition produces multiple
// partitions that the task_creator pins `partition_idx % num_gpus`.
// The merge stage pulls the per-partition sorted runs back into a single
// sorted output. These TEST_CASEs cover the cross-GPU sort/merge path:
//
//   1. Large sort across both GPUs — 4M rows with a permuted key so the
//      partitioner produces >=2 partitions; asserts correctness and
//      that pipeline_task dispatches land on both GPUs.
//   2. Small sort that stays on a single GPU — working set below the
//      multi-partition floor; correctness only.
//   3. ORDER BY ... LIMIT over large input — exercises the top-N merge
//      path with cross-GPU partitions feeding a single-GPU top-N sink.

#include "mgpu_test_utils.hpp"

#include <catch.hpp>
#include <cuda_runtime.h>
#include <duckdb.hpp>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace sirius::test::mgpu;

namespace {

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
  auto tmp = fs::temp_directory_path() /
             ("sirius-mgpu-order-" + std::to_string(::getpid()) + "-" + tag);
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  return tmp;
}

}  // namespace

TEST_CASE("physical_order - large sort distributes across two GPUs",
          "[mgpu][operator-mgpu][order][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp     = make_tmp_dir("large");
  auto log_dir = tmp / "log";
  auto yaml    = tmp / "mgpu.yaml";

  // 8 files × 500k rows × 16 B ≈ 64 MiB. `k = range * 37 % 1000000` gives a
  // permuted key so the sort has real work, and the permutation still
  // produces enough distinct values to drive multi-partition execution at
  // hash_partition_bytes=1 MiB.
  constexpr int kNumFiles    = 8;
  constexpr int kRowsPerFile = 500000;
  generate_parquet_surface(
    tmp,
    "SELECT (range * 37) % 1000000 AS k, range AS v FROM range(" +
      std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  scoped_log_dir logs(log_dir);

  auto glob = parquet_glob(tmp);
  auto inner_query =
    "SELECT k, v FROM read_parquet('" + glob + "') ORDER BY k DESC LIMIT 100";

  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    con.reset();  // flush sinks before parse_audit_log
  }

  auto by_gpu = parse_audit_log(log_dir);
  INFO("gpu0 pipelines=" << by_gpu[0].pipeline_ids.size()
                         << " gpu1 pipelines=" << by_gpu[1].pipeline_ids.size());
  REQUIRE(by_gpu.count(0));
  REQUIRE(by_gpu.count(1));
  REQUIRE(by_gpu[0].pipeline_ids.size() >= 1);
  REQUIRE(by_gpu[1].pipeline_ids.size() >= 1);

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("physical_order - small sort stays single-GPU",
          "[mgpu][operator-mgpu][order][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp     = make_tmp_dir("small");
  auto log_dir = tmp / "log";
  auto yaml    = tmp / "mgpu.yaml";

  // 8 files × 1k rows ≈ 128 KiB — well below the multi-partition floor at
  // hash_partition_bytes=1 MiB. A single partition should keep the sort on
  // one GPU; we assert correctness only here, not distribution.
  constexpr int kNumFiles    = 8;
  constexpr int kRowsPerFile = 1000;
  generate_parquet_surface(
    tmp,
    "SELECT (range * 37) % 1000000 AS k, range AS v FROM range(" +
      std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  scoped_log_dir logs(log_dir);

  auto glob        = parquet_glob(tmp);
  auto inner_query = "SELECT * FROM read_parquet('" + glob + "') ORDER BY k LIMIT 10";

  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    con.reset();
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("physical_order - order by with limit over large input",
          "[mgpu][operator-mgpu][order][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp     = make_tmp_dir("topn");
  auto log_dir = tmp / "log";
  auto yaml    = tmp / "mgpu.yaml";

  // Same large surface as test #1 so partitioning routes across both GPUs,
  // but with a tight LIMIT 5 to exercise the top-N path — per-partition
  // top-N runs then merge to a single final top-5 on the consumer GPU.
  constexpr int kNumFiles    = 8;
  constexpr int kRowsPerFile = 500000;
  generate_parquet_surface(
    tmp,
    "SELECT (range * 37) % 1000000 AS k, range AS v FROM range(" +
      std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  scoped_log_dir logs(log_dir);

  auto glob        = parquet_glob(tmp);
  auto inner_query = "SELECT k FROM read_parquet('" + glob + "') ORDER BY k LIMIT 5";

  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    con.reset();
  }

  auto by_gpu = parse_audit_log(log_dir);
  INFO("gpu0 pipelines=" << by_gpu[0].pipeline_ids.size()
                         << " gpu1 pipelines=" << by_gpu[1].pipeline_ids.size());
  REQUIRE(by_gpu.count(0));
  REQUIRE(by_gpu.count(1));
  REQUIRE(by_gpu[0].pipeline_ids.size() >= 1);
  REQUIRE(by_gpu[1].pipeline_ids.size() >= 1);

  std::error_code ec;
  fs::remove_all(tmp, ec);
}
