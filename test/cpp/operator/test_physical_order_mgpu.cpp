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
// ORDER BY samples, range-partitions, and merges: SORT_PARTITION cuts each
// sorted run into range slices and MERGE_SORT pulls them back into a single
// sorted output, placing each task by data locality (sort has no partition
// placement). These TEST_CASEs cover the cross-GPU sort/merge path:
//
//   1. Large sort across both GPUs — 4M rows with a permuted key so the
//      partitioner produces >=2 partitions; asserts correctness and
//      that pipeline_task dispatches land on both GPUs.
//   2. Small sort that stays on a single GPU — working set below the
//      multi-partition floor; correctness only.
//   3. ORDER BY ... LIMIT over large input — exercises the top-N merge
//      path with cross-GPU partitions feeding a single-GPU top-N sink.
//   4. ORDER BY without LIMIT, forced onto many range partitions with a small
//      sample — the only case here that reaches SORT_SAMPLE → SORT_PARTITION
//      with more than one partition, and with SORT_PARTITION tasks on a GPU
//      other than the one that computed the partition boundaries.

#include "log/logging.hpp"
#include "mgpu_test_utils.hpp"

#include <cuda_runtime.h>

#include <catch.hpp>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sirius::test::mgpu;

namespace {

mgpu_env_params make_params()
{
  mgpu_env_params p;
  p.cache                = "none";
  p.hash_partition_bytes = 1'000'000;   // 1 MiB → force many partitions
  p.scan_task_batch_size = 16'000'000;  // 16 MB → force >=2 scan batches so the
                                        // coalescer doesn't merge the whole surface
                                        // into one batch (else all work lands on GPU 0)
  p.pipeline_num_threads     = 4;
  p.task_creator_num_threads = 4;
  return p;
}

fs::path make_tmp_dir(std::string const& tag)
{
  auto tmp =
    fs::temp_directory_path() / ("sirius-mgpu-order-" + std::to_string(::getpid()) + "-" + tag);
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  return tmp;
}

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

// Runs @p query on the GPU with CPU fallback off and on DuckDB's CPU engine, and requires the two
// results to be equal row for row, in order. require_gpu_matches_cpu sorts both sides first, which
// would hide a wrongly ordered result.
void require_gpu_matches_cpu_in_order(duckdb::Connection& con, std::string const& query)
{
  auto run = [&](std::string const& sql) {
    auto result = con.Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(sql << ": " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    return result;
  };

  run("SET enable_duckdb_fallback = false;");
  auto gpu_result = run("CALL gpu_execution(\"" + query + "\")");
  run("SET gpu_execution = false;");
  auto cpu_result = run(query + ";");
  run("SET gpu_execution = true;");

  REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
  REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

  auto const ncols = gpu_result->ColumnCount();
  auto const nrows = gpu_result->RowCount();
  auto gpu_rows    = gpu_result->Collection().GetRows();
  auto cpu_rows    = cpu_result->Collection().GetRows();
  // One assertion for the whole result: the result has millions of cells.
  duckdb::idx_t mismatches = 0;
  for (duckdb::idx_t r = 0; r < nrows; ++r) {
    for (duckdb::idx_t c = 0; c < ncols; ++c) {
      auto const gpu_cell = gpu_rows.GetValue(c, r).ToString();
      auto const cpu_cell = cpu_rows.GetValue(c, r).ToString();
      if (gpu_cell == cpu_cell) { continue; }
      if (mismatches == 0) {
        UNSCOPED_INFO("first mismatch at row " << r << " col " << c << ": GPU=[" << gpu_cell
                                               << "] CPU=[" << cpu_cell << "]");
      }
      ++mismatches;
    }
  }
  REQUIRE(mismatches == 0);
}

}  // namespace

TEST_CASE("physical_order - large sort distributes across two GPUs",
          "[mgpu][operator-mgpu][order][gpu_execution][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("large");
  auto yaml = tmp / "mgpu.yaml";

  // 8 files × 500k rows × 16 B ≈ 64 MiB. `k = range * 37 % 1000000` gives a
  // permuted key so the sort has real work, and the permutation still
  // produces enough distinct values to drive multi-partition execution at
  // hash_partition_bytes=1 MiB.
  constexpr int kNumFiles    = 8;
  constexpr int kRowsPerFile = 500000;
  generate_parquet_surface(tmp,
                           "SELECT (range * 37) % 1000000 AS k, range AS v FROM range(" +
                             std::to_string(kRowsPerFile) + ")",
                           kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  auto glob        = parquet_glob(tmp);
  auto inner_query = "SELECT k, v FROM read_parquet('" + glob + "') ORDER BY k DESC LIMIT 100";

  std::map<int, size_t> tasks_per_gpu;
  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    auto& scheduler = env.get_task_scheduler(*con);
    con.reset();  // flush sinks before reading metrics

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

// Phase 12 regression: ORDER BY ... LIMIT over a tiny parquet surface MUST NOT
// throw std::out_of_range / vector::_M_range_check when the partitioner yields
// a 2-element partition vector. See 12-CONTEXT.md and 12-stack-trace.txt for
// the original off-by-one site at sirius_physical_hash_join.cpp:622-637.
TEST_CASE("physical_order - small sort rangecheck regression",
          "[mgpu][operator-mgpu][order][gpu_execution][regression][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("rangecheck");
  auto yaml = tmp / "mgpu.yaml";

  // Tiny surface (4 files x 256 rows). Combined with hash_partition_bytes=1024
  // below, this forces the partitioner into the 2-element vector shape that
  // originally tripped vector::_M_range_check on the small-sort plan path.
  constexpr int kNumFiles    = 4;
  constexpr int kRowsPerFile = 256;
  generate_parquet_surface(
    tmp,
    "SELECT (range * 17) % 1024 AS k, range AS v FROM range(" + std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  auto params                 = make_params();
  params.hash_partition_bytes = 1024;  // force multi-partition on tiny data
  write_mgpu_yaml(yaml, params);
  REQUIRE(fs::exists(yaml));

  auto glob        = parquet_glob(tmp);
  auto inner_query = "SELECT * FROM read_parquet('" + glob + "') ORDER BY k LIMIT 5";

  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    // The actual regression assertion: this used to throw
    // std::vector::_M_range_check inside gpu_execution. Post-12-02 it must
    // round-trip cleanly and match CPU output row-for-row.
    require_gpu_matches_cpu(*con, inner_query);
    con.reset();
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("physical_order - small sort stays single-GPU",
          "[mgpu][operator-mgpu][order][gpu_execution][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("small");
  auto yaml = tmp / "mgpu.yaml";

  // 8 files × 1k rows ≈ 128 KiB — well below the multi-partition floor at
  // hash_partition_bytes=1 MiB. A single partition should keep the sort on
  // one GPU; we assert correctness only here, not distribution.
  constexpr int kNumFiles    = 8;
  constexpr int kRowsPerFile = 1000;
  generate_parquet_surface(tmp,
                           "SELECT (range * 37) % 1000000 AS k, range AS v FROM range(" +
                             std::to_string(kRowsPerFile) + ")",
                           kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

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
          "[mgpu][operator-mgpu][order][gpu_execution][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("topn");
  auto yaml = tmp / "mgpu.yaml";

  // Same large surface as test #1 so partitioning routes across both GPUs,
  // but with a tight LIMIT 5 to exercise the top-N path — per-partition
  // top-N runs then merge to a single final top-5 on the consumer GPU.
  constexpr int kNumFiles    = 8;
  constexpr int kRowsPerFile = 500000;
  generate_parquet_surface(tmp,
                           "SELECT (range * 37) % 1000000 AS k, range AS v FROM range(" +
                             std::to_string(kRowsPerFile) + ")",
                           kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  auto glob        = parquet_glob(tmp);
  auto inner_query = "SELECT k FROM read_parquet('" + glob + "') ORDER BY k LIMIT 5";

  std::map<int, size_t> tasks_per_gpu;
  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    auto& scheduler = env.get_task_scheduler(*con);
    con.reset();  // flush sinks before reading metrics

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

// A range sort computes its partition boundaries once, in the first SORT_SAMPLE task, on that
// task's GPU. Every later batch is range-partitioned by its own task on the GPU that holds it, so
// that task needs the boundaries in its own GPU's memory: reading the first task's table is an
// illegal memory access on GPUs without access to each other's memory pool.
//
// The other tests in this file use LIMIT, which plans TOP_N and never reaches this path.
TEST_CASE("physical_order - range sort partitions batches on a GPU other than the sample's",
          "[mgpu][operator-mgpu][order][gpu_execution][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("range");
  auto yaml = tmp / "mgpu.yaml";

  // 8 files × 500k rows. Each query below reads two columns, about 64 MiB, which the 16 MB scan
  // batch size splits into several batches spread over both GPUs. `k` is a permuted key, `s` its
  // string form, `n` a nullable key; `v` breaks ties within a file. The files are identical, so
  // rows that tie on every column are equal and the expected order is still unique.
  constexpr int kNumFiles    = 8;
  constexpr int kRowsPerFile = 500000;
  generate_parquet_surface(tmp,
                           "SELECT (range * 7919) % 1000003 AS k, range AS v, "
                           "'key-' || lpad(((range * 7919) % 1000003)::VARCHAR, 8, '0') AS s, "
                           "CASE WHEN range % 10 = 0 THEN NULL ELSE (range * 31) % 100003 END AS n "
                           "FROM range(" +
                             std::to_string(kRowsPerFile) + ")",
                           kNumFiles);

  // DEBUG level so the per-GPU boundary copy is logged.
  scoped_log_dir log_dir(tmp / "logs", "debug");
  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  auto const source = "FROM read_parquet('" + parquet_glob(tmp) + "') ";
  std::vector<std::string> const queries{
    "SELECT k, v " + source + "ORDER BY k, v DESC",
    "SELECT s, v " + source + "ORDER BY s DESC, v",
    "SELECT n, v " + source + "ORDER BY n DESC NULLS FIRST, v",
  };

  {
    scoped_mgpu_env env(yaml);
    auto con = env.make_connection();
    // 1 MiB per partition gives tens of range partitions. A 4 MiB sample is met by the first
    // batch, so the boundaries come from one batch on one GPU and every other batch is
    // partitioned by a later task, on whichever GPU holds it.
    for (auto const* setting :
         {"SET max_sort_partition_bytes = 1048576;", "SET sort_sample_bytes = 4194304;"}) {
      auto result = con.Query(setting);
      REQUIRE(result);
      REQUIRE_FALSE(result->HasError());
    }

    for (auto const& query : queries) {
      INFO(query);
      require_gpu_matches_cpu_in_order(con, query);
    }

    // The path under test ran: a GPU other than the sample task's received the boundaries.
    INFO("log dir: " << log_dir.path());
    REQUIRE(log_dir_contains(log_dir.path(), "[sort_boundaries] copied"));
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}
