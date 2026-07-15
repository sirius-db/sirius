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

// Regression for partition device-pin enforcement on cuco-backed operators.
//
// A BUILD_PROBE hash_join's single cuco table is only valid on the GPU it was
// built on; a probe task on another GPU trips cudaErrorInvalidValue (SIGABRT).
// Two fixed defects let a probe task scatter to the wrong GPU: (1) partition
// affinity indexed the physical topology, so num_gpus<physical mapped onto a
// device with no executor (pin dropped); (2) OOM reschedule dropped the pin.
//
// A small dim (build) + large fact (probe) keeps it in BUILD_PROBE; a small
// usage_limit_bytes forces OOM reschedules. The mgpu case SIGABRT'd before the
// fixes (on a host with >2 GPUs); the single-GPU case is a control where the
// cross-device hazard can't occur. Both assert GPU result == true CPU reference.

#include "mgpu_test_utils.hpp"

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace {

using sirius::test::mgpu::generate_parquet_surface;
using sirius::test::mgpu::mgpu_env_params;
using sirius::test::mgpu::parquet_glob;
using sirius::test::mgpu::require_gpu_matches_cpu;
using sirius::test::mgpu::scoped_mgpu_env;
using sirius::test::mgpu::write_mgpu_yaml;

fs::path make_tmp(std::string const& tag)
{
  return fs::temp_directory_path() /
         ("sirius-mgpu-partmemspace-" + tag + "-" + std::to_string(::getpid()));
}

// Read a uint64 tuning knob from the environment, or fall back to `dflt`
// (lets the repro be tuned per host without recompiling).
uint64_t env_u64(char const* name, uint64_t dflt)
{
  if (char const* v = std::getenv(name)) {
    try {
      return static_cast<uint64_t>(std::stoull(v));
    } catch (...) {
      // fall through to default on garbage input
    }
  }
  return dflt;
}

// Small dim (build) + large fact (probe) → BUILD_PROBE (one shared cuco table),
// aggregated to one row so the oracle stays tiny while the join drives many
// probe tasks. fk = range % dim_rows so every probe row matches one build row.
struct join_surface {
  fs::path dim_dir;
  fs::path fact_dir;
};

join_surface build_join_surface(fs::path const& tmp,
                                uint64_t fact_rows,
                                uint64_t dim_rows,
                                int fact_files)
{
  join_surface s{tmp / "dim", tmp / "fact"};
  generate_parquet_surface(
    s.dim_dir,
    "SELECT range AS k, (range % 1000) AS v FROM range(" + std::to_string(dim_rows) + ")",
    /*num_files=*/1);
  generate_parquet_surface(s.fact_dir,
                           "SELECT (range % " + std::to_string(dim_rows) +
                             ") AS fk, (range % 1000) AS v FROM range(" +
                             std::to_string(fact_rows) + ")",
                           fact_files);
  return s;
}

std::string join_query(join_surface const& s)
{
  return "SELECT COUNT(*) AS c, SUM(f.v) AS sf, SUM(d.v) AS sd "
         "FROM read_parquet('" +
         parquet_glob(s.fact_dir) + "') f JOIN read_parquet('" + parquet_glob(s.dim_dir) +
         "') d ON f.fk = d.k";
}

// Memory params shared by both variants. The only difference between the
// multi- and single-GPU TEST_CASEs is num_gpus.
mgpu_env_params tight_params(int num_gpus)
{
  mgpu_env_params params{};
  params.num_gpus = num_gpus;
  params.cache    = "none";
  // Small ABSOLUTE per-GPU budget => OOM/reschedule fires regardless of
  // physical capacity. Default 512 MiB; override with the env knob below.
  params.usage_limit_bytes = env_u64("SIRIUS_TEST_PART_MEMSPACE_GPU_BYTES", 512ULL * 1024 * 1024);
  // Large enough that the small dim stays in BUILD_PROBE mode (one shared
  // build table) rather than partitioning into MIXED_JOIN.
  params.max_build_hash_table_bytes = 256'000'000;
  params.hash_partition_bytes       = 10'000'000;
  params.concat_batch_bytes         = 50'000'000;
  params.scan_task_batch_size       = 20'000'000;
  params.pipeline_num_threads       = 4;
  params.task_creator_num_threads   = 4;
  params.duckdb_scan_num_threads    = 2;
  return params;
}

}  // namespace

//===----------------------------------------------------------------------===//
// MULTI-GPU regression. Passes with the pin fixes; before them a probe task on
// the wrong GPU SIGABRT'd in cuco (so a regression aborts the unittest binary).
//===----------------------------------------------------------------------===//
TEST_CASE("partition memory-space is enforced across OOM reschedule on multi-GPU",
          "[mgpu][operator-mgpu][partition][memspace][hash_join][gpu_execution]")
{
  if (!sirius::test::mgpu::require_two_gpus()) return;

  auto tmp = make_tmp("mgpu");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  // Per-FILE fact rows (total probe rows = rows * fact_files). Kept under
  // usage_limit_bytes so the join completes rather than exhausting OOM retries;
  // the crash this guards comes from probe-task scatter, not from overflow.
  uint64_t const rows     = env_u64("SIRIUS_TEST_PART_MEMSPACE_ROWS", 2'000'000);
  uint64_t const dim_rows = env_u64("SIRIUS_TEST_PART_MEMSPACE_DIM_ROWS", 200'000);
  int const iterations    = static_cast<int>(env_u64("SIRIUS_TEST_PART_MEMSPACE_ITERS", 5));

  auto surface = build_join_surface(tmp, rows, dim_rows, /*fact_files=*/8);

  auto yaml_path = tmp / "part-memspace-mgpu.yaml";
  write_mgpu_yaml(yaml_path, tight_params(/*num_gpus=*/2));

  scoped_mgpu_env env(yaml_path);
  auto con   = env.make_connection();
  auto query = join_query(surface);

  // Loop to widen the OOM/reschedule race window; a wrong-GPU dispatch fails
  // the embedded REQUIREs in require_gpu_matches_cpu (HasError or row mismatch).
  for (int i = 0; i < iterations; ++i) {
    INFO("multi-gpu iteration " << i << " (rows=" << rows
                                << ", gpu_bytes=" << tight_params(2).usage_limit_bytes << ")");
    require_gpu_matches_cpu(con, query, /*force_cpu_reference=*/true);
  }

  // SIRIUS_TEST_PART_MEMSPACE_KEEP=1 preserves the tmp dir (parquet + log) for
  // inspection (e.g. grep the log for the "OOM reschedule" WARN).
  if (!std::getenv("SIRIUS_TEST_PART_MEMSPACE_KEEP")) {
    fs::remove_all(tmp, ec);
  } else {
    WARN("kept repro artifacts at " << tmp.string());
  }
}

//===----------------------------------------------------------------------===//
// SINGLE-GPU control: same partitioned-join-under-OOM path with num_gpus=1.
// Runs on any host with >=1 GPU. On one device the cross-GPU hazard cannot
// occur, so this asserts the OOM-reschedule machinery itself stays correct.
//===----------------------------------------------------------------------===//
TEST_CASE("partition memory-space repro - single-GPU OOM-reschedule control",
          "[operator-mgpu][partition][memspace][hash_join][gpu_execution][single_gpu]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 1) {
    WARN("partition memspace single-GPU control requires >=1 GPU; none visible — skipping");
    return;
  }

  auto tmp = make_tmp("1gpu");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  // Per-FILE fact rows (total probe rows = rows * fact_files). Kept under
  // usage_limit_bytes so the join completes rather than exhausting OOM retries;
  // the crash this guards comes from probe-task scatter, not from overflow.
  uint64_t const rows     = env_u64("SIRIUS_TEST_PART_MEMSPACE_ROWS", 2'000'000);
  uint64_t const dim_rows = env_u64("SIRIUS_TEST_PART_MEMSPACE_DIM_ROWS", 200'000);
  int const iterations    = static_cast<int>(env_u64("SIRIUS_TEST_PART_MEMSPACE_ITERS", 5));

  auto surface = build_join_surface(tmp, rows, dim_rows, /*fact_files=*/8);

  auto yaml_path = tmp / "part-memspace-1gpu.yaml";
  write_mgpu_yaml(yaml_path, tight_params(/*num_gpus=*/1));

  scoped_mgpu_env env(yaml_path);
  auto con   = env.make_connection();
  auto query = join_query(surface);

  for (int i = 0; i < iterations; ++i) {
    INFO("single-gpu iteration " << i << " (rows=" << rows << ")");
    require_gpu_matches_cpu(con, query, /*force_cpu_reference=*/true);
  }

  if (!std::getenv("SIRIUS_TEST_PART_MEMSPACE_KEEP")) {
    fs::remove_all(tmp, ec);
  } else {
    WARN("kept repro artifacts at " << tmp.string());
  }
}
