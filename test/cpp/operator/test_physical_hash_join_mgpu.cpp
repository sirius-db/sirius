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

// Multi-GPU integration tests for sirius_physical_hash_join.
//
// hash_join is the highest-risk operator in the num_gpus>1 routing matrix
// because both BUILD_PROBE and MIXED_JOIN paths use cuco hash tables that
// cannot span GPUs. task_creator pins hash_join inputs via SCHED-00 using
// `partition_idx % num_gpus`:
//   - BUILD_PROBE: operator_id is the partition index, so every probe task
//     pins to the same GPU as the single shared build table.
//   - MIXED_JOIN:  the real partition index is used, so partitions spread
//     across GPUs (one partition per GPU on 2-GPU hosts).
//
// A SF100 Q11 repro (cache=table_gpu + num_gpus=2) surfaced a race in the
// BUILD_PROBE state machine: task_creator calls
// get_next_task_input_data_for_build_probe while _hash_table_build_state is
// SCHEDULED (between dispatching the build task and the build task running
// execute()), and that branch logs "invalid hash table build state 2" and
// returns nullptr. The task_creator's caller handles the nullptr gracefully,
// but the downstream symptom is that probe tasks get dispatched against
// batches whose processing state has been left in a bad spot. See
// .planning/phases/08-multi-gpu-sql-pipeline-fix/08-FOLLOWUPS.md #17.
//
// These tests assert:
//   1. BUILD_PROBE correctness under num_gpus=2 with enough probe batches
//      to cover the SCHEDULING → SCHEDULED → BUILT transition window.
//   2. MIXED_JOIN distributes partitions across both GPUs and returns the
//      same result set as DuckDB CPU.
//   3. Repeated back-to-back invocations of BUILD_PROBE don't wedge on the
//      leftover hash table state from the prior query (DESTROYED → NOT_BUILT).

#include "log/logging.hpp"
#include "mgpu_test_utils.hpp"

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>

#include <cstdlib>
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

/**
 * @brief Build a unique tmp dir for this TEST_CASE. The `tag` disambiguates
 * when several TEST_CASEs run in one process.
 */
fs::path make_tmp(std::string const& tag)
{
  return fs::temp_directory_path() /
         ("sirius-mgpu-hashjoin-" + tag + "-" + std::to_string(::getpid()));
}

/**
 * @brief Generate a "small" parquet surface — a few files, a few thousand
 * rows each. Intended as the build side for BUILD_PROBE. Columns: `k`
 * (bigint key), `v` (bigint value).
 */
void generate_small_build_side(fs::path const& dir)
{
  generate_parquet_surface(dir,
                           "SELECT range AS k, range * 3 AS v FROM range(5000)",
                           /*num_files=*/2);
}

/**
 * @brief Generate a "large" parquet surface — many files, many rows each,
 * comfortably above the 32 MiB multi-partition floor so hash_partition
 * produces enough partitions to force both GPUs into play on num_gpus=2.
 * Columns: `k` (bigint key joining the build side), `v` (bigint value).
 */
void generate_large_probe_side(fs::path const& dir)
{
  // 8 files × 500k rows × 16 B/row ≈ 64 MiB — above the num_gpus * 16 MiB
  // floor the partition operator enforces in configure_partition_min_partitions
  // (src/pipeline/sirius_pipeline_converter.cpp:782).
  generate_parquet_surface(
    dir, "SELECT range AS k, range * 7 AS v FROM range(500000)", /*num_files=*/8);
}

/**
 * @brief Scan every file under @p log_dir for @p needle. Used to assert that a
 * runtime code path actually engaged: correctness alone cannot distinguish
 * "broadcast replicated the build" from "fell back to single-GPU routing", nor
 * "dynamic filter published" from "no filter published" — both fall back to a
 * correct (but slower) result. The distinguishing signal is the log marker.
 * Requires the log level to include the marker (broadcast markers are DEBUG).
 */
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

//===----------------------------------------------------------------------===//
// BUILD_PROBE: probe side is large, build side is small. The build operator
// pins to a single GPU via operator_id partition index; every probe task
// must land on the same GPU. The probe side has enough batches that
// task_creator issues many probe tasks while the build task is still in
// SCHEDULED state, which is exactly the window the SF100 Q11 race exercises.
//===----------------------------------------------------------------------===//
TEST_CASE("physical_hash_join - BUILD_PROBE probe-heavy join across two GPUs",
          "[mgpu][operator-mgpu][hash_join][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("build-probe");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto build_dir = tmp / "build";
  auto probe_dir = tmp / "probe";
  generate_small_build_side(build_dir);
  generate_large_probe_side(probe_dir);

  mgpu_env_params params{};
  params.cache                    = "none";
  params.hash_partition_bytes     = 1'000'000;
  params.pipeline_num_threads     = 4;
  params.task_creator_num_threads = 4;
  auto yaml_path                  = tmp / "fu17-hj.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);

  // BUILD_PROBE trigger: build side MUST be smaller than
  // max_build_hash_table_bytes (configured 90 MiB above) so the planner
  // chooses BUILD_PROBE rather than MIXED_JOIN. 5000-row × 16 B ≈ 80 KiB,
  // well under the threshold.
  auto inner_query =
    "SELECT probe.k, probe.v, build.v AS build_v "
    "FROM read_parquet('" +
    parquet_glob(probe_dir) +
    "') AS probe "
    "JOIN read_parquet('" +
    parquet_glob(build_dir) +
    "') AS build "
    "  ON probe.k = build.k "
    "ORDER BY probe.k, probe.v "
    "LIMIT 100";

  std::map<int, size_t> tasks_per_gpu;
  {
    auto con = env.make_connection();
    require_gpu_matches_cpu(con, inner_query);
    auto& scheduler = env.get_task_scheduler(con);
    scheduler.visit_executors(
      [&](int device_id, const sirius::pipeline::gpu_pipeline_executor& exec) {
        tasks_per_gpu[device_id] = exec.get_metrics().tasks_executed;
      });
  }

  // Both GPUs saw SOME pipeline work.
  INFO("gpu0 tasks=" << tasks_per_gpu[0] << " gpu1 tasks=" << tasks_per_gpu[1]);
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// MIXED_JOIN: both sides are big enough that the planner builds the real
// partitioned path. Each partition lands on partition_idx % num_gpus, so
// a 2-GPU run with >=2 partitions should see pipeline work on both GPUs.
//===----------------------------------------------------------------------===//
TEST_CASE("physical_hash_join - MIXED_JOIN large-vs-large join distributes partitions",
          "[mgpu][operator-mgpu][hash_join][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("mixed-join");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto left_dir  = tmp / "left";
  auto right_dir = tmp / "right";
  // Both sides >> max_build_hash_table_bytes so neither side is auto-promoted
  // to BUILD_PROBE. Use the same generator function so the key ranges
  // overlap and the join produces non-trivial output.
  generate_large_probe_side(left_dir);
  generate_large_probe_side(right_dir);

  mgpu_env_params params{};
  params.cache                = "none";
  params.hash_partition_bytes = 1'000'000;
  // Force MIXED_JOIN by lowering max_build_hash_table_bytes below our
  // 64 MiB per-side surface.
  params.max_build_hash_table_bytes = 1'000'000;
  params.pipeline_num_threads       = 4;
  params.task_creator_num_threads   = 4;
  auto yaml_path                    = tmp / "fu17-mj.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);

  auto inner_query =
    "SELECT l.k, COUNT(*) AS n "
    "FROM read_parquet('" +
    parquet_glob(left_dir) +
    "') AS l "
    "JOIN read_parquet('" +
    parquet_glob(right_dir) +
    "') AS r "
    "  ON l.k = r.k "
    "GROUP BY l.k "
    "ORDER BY l.k "
    "LIMIT 50";

  std::map<int, size_t> tasks_per_gpu;
  {
    auto con = env.make_connection();
    require_gpu_matches_cpu(con, inner_query);
    auto& scheduler = env.get_task_scheduler(con);
    scheduler.visit_executors(
      [&](int device_id, const sirius::pipeline::gpu_pipeline_executor& exec) {
        tasks_per_gpu[device_id] = exec.get_metrics().tasks_executed;
      });
  }

  // MIXED_JOIN with >=2 partitions must schedule pipeline work on BOTH GPUs.
  INFO("gpu0 tasks=" << tasks_per_gpu[0] << " gpu1 tasks=" << tasks_per_gpu[1]);
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Repeated BUILD_PROBE invocations in the same process. The hash_join's
// _hash_table_build_state cycles NOT_BUILT → SCHEDULING → SCHEDULED →
// BUILT → DESTROYED → (next query) NOT_BUILT; a bug that leaks state
// across queries would surface here even if the first call passes.
//===----------------------------------------------------------------------===//
TEST_CASE("physical_hash_join - repeated BUILD_PROBE queries don't wedge on leftover state",
          "[mgpu][operator-mgpu][hash_join][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("build-probe-repeat");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto build_dir = tmp / "build";
  auto probe_dir = tmp / "probe";
  generate_small_build_side(build_dir);
  generate_large_probe_side(probe_dir);

  mgpu_env_params params{};
  params.cache                    = "none";
  params.hash_partition_bytes     = 1'000'000;
  params.pipeline_num_threads     = 4;
  params.task_creator_num_threads = 4;
  auto yaml_path                  = tmp / "fu17-hj-repeat.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);

  auto inner_query =
    "SELECT probe.k, build.v AS build_v "
    "FROM read_parquet('" +
    parquet_glob(probe_dir) +
    "') AS probe "
    "JOIN read_parquet('" +
    parquet_glob(build_dir) +
    "') AS build "
    "  ON probe.k = build.k "
    "ORDER BY probe.k "
    "LIMIT 50";

  constexpr int kIterations = 4;
  {
    auto con = env.make_connection();
    for (int i = 0; i < kIterations; ++i) {
      INFO("iteration " << i);
      require_gpu_matches_cpu(con, inner_query);
    }
  }

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Bisecting the SF10-scale cold correctness bug.
//
// The Q11-like scale-up TEST_CASE (below) produces wrong top-20 rows on cold.
// These narrower TEST_CASEs peel off the suspected contributors one at a time
// using the same ~512 MiB partsupp-like surface: each should PASS if the
// contributor is not the culprit. Tagged [bisect-cold] so a single filter
// runs them in isolation.
//
// Variants (in order of complexity):
//   1. Simple JOIN + GROUP BY + ORDER BY with cache=none — baseline.
//   2. Same with cache=table_gpu — isolates cache's contribution.
//   3. Same shape + HAVING scalar subquery with cache=none — isolates
//      subquery/DELIM_JOIN contribution.
//   4. Full Q11 shape with cache=table_gpu — the failing baseline (already
//      in the other TEST_CASE; duplicated here without [stress] so the
//      bisect filter sweeps all four in one invocation).
//===----------------------------------------------------------------------===//
namespace {

struct bisect_surface {
  fs::path tmp;
  fs::path ps_dir;
  fs::path s_dir;
  fs::path n_dir;
  fs::path yaml_path;
  std::string glob_ps;
  std::string glob_s;
  std::string glob_n;
  std::unique_ptr<scoped_mgpu_env> env;
};

/**
 * @brief Prepare the three-table partsupp/supplier/nation surface used by
 * every bisect TEST_CASE so they share identical input data and only the
 * query shape / cache knob differs. The cache knob is passed via @p cache.
 */
bisect_surface make_bisect_surface(std::string const& tag, std::string const& cache)
{
  bisect_surface s;
  s.tmp = make_tmp("bisect-" + tag);
  std::error_code ec;
  fs::remove_all(s.tmp, ec);
  fs::create_directories(s.tmp);

  s.ps_dir = s.tmp / "partsupp";
  s.s_dir  = s.tmp / "supplier";
  s.n_dir  = s.tmp / "nation";
  // ps_partkey is unique per row (range), ps_suppkey joins into supplier
  // (0..49999), and ps_supplycost is `range` cast to double so SUM is unique
  // per partkey — needed so ORDER BY value DESC has no ties and the
  // comparison oracle can deterministically match top-N row sets against CPU.
  generate_parquet_surface(s.ps_dir,
                           "SELECT range AS ps_partkey, (range % 50000) AS ps_suppkey, "
                           "       CAST(range AS DOUBLE) AS ps_supplycost, "
                           "       CAST(1 AS DOUBLE) AS ps_availqty "
                           "FROM range(2000000)",
                           /*num_files=*/8);
  generate_parquet_surface(s.s_dir,
                           "SELECT range AS s_suppkey, (range % 25) AS s_nationkey "
                           "FROM range(50000)",
                           /*num_files=*/4);
  generate_parquet_surface(
    s.n_dir,
    "SELECT range AS n_nationkey, CASE WHEN range = 7 THEN 'GERMANY' "
    "                                 ELSE concat('N', CAST(range AS VARCHAR)) END AS n_name "
    "FROM range(25)",
    /*num_files=*/1);

  mgpu_env_params params{};
  params.cache                      = cache;
  params.hash_partition_bytes       = 10'000'000;
  params.max_build_hash_table_bytes = 90'000'000;
  params.pipeline_num_threads       = 8;
  params.task_creator_num_threads   = 4;
  params.duckdb_scan_num_threads    = 4;
  s.yaml_path                       = s.tmp / "bisect.yaml";
  write_mgpu_yaml(s.yaml_path, params);

  s.glob_ps = parquet_glob(s.ps_dir);
  s.glob_s  = parquet_glob(s.s_dir);
  s.glob_n  = parquet_glob(s.n_dir);

  s.env = std::make_unique<scoped_mgpu_env>(s.yaml_path);
  return s;
}

}  // namespace

TEST_CASE("hash_join bisect 1 - simple JOIN+GROUP BY+ORDER BY, cache=none",
          "[mgpu][operator-mgpu][hash_join][bisect-cold][gpu_execution]")
{
  if (!require_two_gpus()) return;
  auto s = make_bisect_surface("1", "none");

  auto q =
    "SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value "
    "FROM read_parquet('" +
    s.glob_ps +
    "') ps "
    "JOIN read_parquet('" +
    s.glob_s +
    "') sup ON ps.ps_suppkey = sup.s_suppkey "
    "JOIN read_parquet('" +
    s.glob_n +
    "') n ON sup.s_nationkey = n.n_nationkey "
    "WHERE n.n_name = 'GERMANY' "
    "GROUP BY ps_partkey "
    "ORDER BY value DESC "
    "LIMIT 20";
  {
    auto con = s.env->make_connection();
    require_gpu_matches_cpu(con, q);
  }
  s.env.reset();
  std::error_code ec;
  fs::remove_all(s.tmp, ec);
}

TEST_CASE("hash_join bisect 2 - simple JOIN+GROUP BY+ORDER BY, cache=table_gpu",
          "[mgpu][operator-mgpu][hash_join][bisect-cold][gpu_execution]")
{
  if (!require_two_gpus()) return;
  auto s = make_bisect_surface("2", "table_gpu");

  auto q =
    "SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value "
    "FROM read_parquet('" +
    s.glob_ps +
    "') ps "
    "JOIN read_parquet('" +
    s.glob_s +
    "') sup ON ps.ps_suppkey = sup.s_suppkey "
    "JOIN read_parquet('" +
    s.glob_n +
    "') n ON sup.s_nationkey = n.n_nationkey "
    "WHERE n.n_name = 'GERMANY' "
    "GROUP BY ps_partkey "
    "ORDER BY value DESC "
    "LIMIT 20";
  {
    auto con = s.env->make_connection();
    require_gpu_matches_cpu(con, q);
  }
  s.env.reset();
  std::error_code ec;
  fs::remove_all(s.tmp, ec);
}

TEST_CASE("hash_join bisect 3 - Q11 shape with HAVING subquery, cache=none",
          "[mgpu][operator-mgpu][hash_join][bisect-cold][gpu_execution]")
{
  if (!require_two_gpus()) return;
  auto s = make_bisect_surface("3", "none");

  auto q =
    "SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value "
    "FROM read_parquet('" +
    s.glob_ps +
    "') ps "
    "JOIN read_parquet('" +
    s.glob_s +
    "') sup ON ps.ps_suppkey = sup.s_suppkey "
    "JOIN read_parquet('" +
    s.glob_n +
    "') n ON sup.s_nationkey = n.n_nationkey "
    "WHERE n.n_name = 'GERMANY' "
    "GROUP BY ps_partkey "
    "HAVING SUM(ps_supplycost * ps_availqty) > ("
    "  SELECT SUM(ps_supplycost * ps_availqty) * 0.00001 "
    "  FROM read_parquet('" +
    s.glob_ps +
    "') ps "
    "  JOIN read_parquet('" +
    s.glob_s +
    "') sup ON ps.ps_suppkey = sup.s_suppkey "
    "  JOIN read_parquet('" +
    s.glob_n +
    "') n ON sup.s_nationkey = n.n_nationkey "
    "  WHERE n.n_name = 'GERMANY'"
    ") "
    "ORDER BY value DESC "
    "LIMIT 20";
  {
    auto con = s.env->make_connection();
    require_gpu_matches_cpu(con, q);
  }
  s.env.reset();
  std::error_code ec;
  fs::remove_all(s.tmp, ec);
}

//===----------------------------------------------------------------------===//
// Scale-up test targeting follow-up #17's BUILD_PROBE SCHEDULED state race.
//
// SF100 Q11 crashes on cold with "invalid hash table build state 2 (SCHEDULED)
// in operator 10" followed by batch_lock_utils retry exhaustion; none of the
// smaller BUILD_PROBE TEST_CASEs above reproduce it. The candidate triggers
// are: (a) a build-side big enough that the hash_join SCHEDULING → SCHEDULED
// → BUILT window stays open long enough for task_creator to issue many probe
// hint/input_data calls during the race window; (b) cache=table_gpu so the
// cached-batch lifecycle interacts with repeated BUILD_PROBE cycles;
// (c) a Q11-like scalar subquery pattern whose DELIM_JOIN encodes as nested
// HASH_JOINs feeding each other.
//
// This TEST_CASE tries all three in one shot at the largest size that fits
// unit-test budgets (~500 MiB per side). It is tagged [stress] so CI can
// skip when budget is tight; the [hash_join] tag still matches it for local
// follow-up-17 runs.
//===----------------------------------------------------------------------===//
TEST_CASE("physical_hash_join - follow-up #17 scale-up: Q11-like BUILD_PROBE with table_gpu cache",
          "[mgpu][operator-mgpu][hash_join][stress][followup-17][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("fu17-scaleup");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  // Three "tables" modeled on Q11's partsupp × supplier × nation shape, each
  // sized to stay under the unit-test budget while being large enough that
  // BUILD_PROBE's build task takes measurable wall time (opens the SCHEDULED
  // race window). Columns chosen so join predicates and the sum(a*b)
  // expression match Q11's arithmetic.
  //
  //   partsupp-like: 8 files × 2M rows × 32 B ≈ 512 MiB — the probe side,
  //                  generates many cached batches under table_gpu.
  //   supplier-like: 4 files × 50k rows × 24 B ≈ 5 MiB — build side, small
  //                  enough to land in BUILD_PROBE mode (below the 90 MiB
  //                  max_build_hash_table_bytes in mgpu_env_params).
  //   nation-like:   1 file × 25 rows — matches TPC-H exactly, drives the
  //                  WHERE n_name = 'GERMANY' filter.
  auto ps_dir = tmp / "partsupp";
  auto s_dir  = tmp / "supplier";
  auto n_dir  = tmp / "nation";
  // ps_supplycost = range, ps_availqty = 1.0 → SUM per partkey is `range` and
  // thus unique; ORDER BY value DESC has no ties so the comparison oracle
  // matches row sets deterministically.
  generate_parquet_surface(ps_dir,
                           "SELECT range AS ps_partkey, (range % 50000) AS ps_suppkey, "
                           "       CAST(range AS DOUBLE) AS ps_supplycost, "
                           "       CAST(1 AS DOUBLE) AS ps_availqty "
                           "FROM range(2000000)",
                           /*num_files=*/8);
  generate_parquet_surface(s_dir,
                           "SELECT range AS s_suppkey, (range % 25) AS s_nationkey "
                           "FROM range(50000)",
                           /*num_files=*/4);
  generate_parquet_surface(
    n_dir,
    "SELECT range AS n_nationkey, CASE WHEN range = 7 THEN 'GERMANY' "
    "                                 ELSE concat('N', CAST(range AS VARCHAR)) END AS n_name "
    "FROM range(25)",
    /*num_files=*/1);

  mgpu_env_params params{};
  params.cache                      = "table_gpu";
  params.hash_partition_bytes       = 10'000'000;
  params.max_build_hash_table_bytes = 90'000'000;
  params.pipeline_num_threads       = 8;
  params.task_creator_num_threads   = 4;
  params.duckdb_scan_num_threads    = 4;
  auto yaml_path                    = tmp / "fu17-scaleup.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);

  // Q11-shaped query: the inner aggregate feeds a scalar HAVING subquery,
  // which DuckDB encodes as a DELIM_JOIN over the same three-table join.
  // The outer and inner joins both land as BUILD_PROBE (supplier fits the
  // build threshold) — on a 2-GPU host this exercises:
  //   - two BUILD_PROBE hash joins in one query
  //   - cache=table_gpu sharing partsupp scan results across the outer and
  //     subquery scans (different pipeline_ids, so separate cache entries)
  //   - the SCHEDULED state race window while the build task is running
  auto inner_query =
    "SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value "
    "FROM read_parquet('" +
    parquet_glob(ps_dir) +
    "') ps "
    "JOIN read_parquet('" +
    parquet_glob(s_dir) +
    "') s ON ps.ps_suppkey = s.s_suppkey "
    "JOIN read_parquet('" +
    parquet_glob(n_dir) +
    "') n ON s.s_nationkey = n.n_nationkey "
    "WHERE n.n_name = 'GERMANY' "
    "GROUP BY ps_partkey "
    "HAVING SUM(ps_supplycost * ps_availqty) > ("
    "  SELECT SUM(ps_supplycost * ps_availqty) * 0.00001 "
    "  FROM read_parquet('" +
    parquet_glob(ps_dir) +
    "') ps "
    "  JOIN read_parquet('" +
    parquet_glob(s_dir) +
    "') s ON ps.ps_suppkey = s.s_suppkey "
    "  JOIN read_parquet('" +
    parquet_glob(n_dir) +
    "') n ON s.s_nationkey = n.n_nationkey "
    "  WHERE n.n_name = 'GERMANY'"
    ") "
    "ORDER BY value DESC "
    "LIMIT 20";

  // Three iterations: iteration 0 populates the cache (cold), iterations
  // 1..2 replay under preload (warm). Any hash_join state-machine race
  // during cold or stale-cache hazard during warm should surface in this
  // matrix.
  constexpr int kIterations = 3;
  std::map<int, size_t> tasks_per_gpu;
  {
    auto con = env.make_connection();
    for (int i = 0; i < kIterations; ++i) {
      INFO("iteration " << i);
      require_gpu_matches_cpu(con, inner_query);
    }
    auto& scheduler = env.get_task_scheduler(con);
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

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Broadcast small-build BUILD_PROBE. When the build side is small
// (< small_table_bytes == num_gpus * 16 MiB), the PARTITION operator replicates
// it to every GPU (one hash table per GPU) instead of funneling the whole build
// to a single GPU. Correctness alone cannot prove the broadcast path engaged —
// a single-GPU fallback also returns the right rows — so these tests also assert
// the runtime log markers ("[broadcast]" from the partition decision; "Pushed
// ... dynamic filter(s)" from the publisher). They are 2-GPU-gated.
//===----------------------------------------------------------------------===//
TEST_CASE("physical_hash_join - broadcast small-build BUILD_PROBE replicates across two GPUs",
          "[mgpu][operator-mgpu][hash_join][build_probe][broadcast][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("broadcast-replicate");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto build_dir = tmp / "build";
  auto probe_dir = tmp / "probe";
  generate_small_build_side(build_dir);  // ~80 KiB << small_table_bytes (2 * 16 MiB)
  generate_large_probe_side(probe_dir);

  // DEBUG level so the partition "[broadcast]" decision marker is emitted.
  scoped_log_dir log_dir(tmp / "logs", "debug");

  mgpu_env_params params{};
  params.cache   = "none";
  auto yaml_path = tmp / "broadcast-replicate.yaml";
  write_mgpu_yaml(yaml_path, params);
  scoped_mgpu_env env(yaml_path);

  auto inner_query =
    "SELECT probe.k, probe.v, build.v AS build_v "
    "FROM read_parquet('" +
    parquet_glob(probe_dir) +
    "') AS probe "
    "JOIN read_parquet('" +
    parquet_glob(build_dir) +
    "') AS build "
    "  ON probe.k = build.k "
    "ORDER BY probe.k, probe.v "
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

  // The broadcast path engaged: the small build was replicated, not funneled to one GPU.
  INFO("log dir: " << log_dir.path());
  REQUIRE(log_dir_contains(log_dir.path(), "[broadcast]"));

  // Both GPUs built their own hash table from the replicated build and did probe work.
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// A broadcast join publishes dynamic filters: every partition holds the full
// replicated build, so the first GPU to arrive wins the OPEN->PUBLISHING race
// and publishes the membership filter (exactly once). Before the broadcast
// fix, >1 partition disabled publication entirely. The build-side predicate
// supplies the filter evidence discovery requires to wire a producer, and the
// filtered build (2500 keys) vs the large probe domain (500000 keys)
// guarantees a membership filter is emitted. 2-GPU-gated.
//===----------------------------------------------------------------------===//
TEST_CASE("physical_hash_join - broadcast BUILD_PROBE publishes dynamic filters across two GPUs",
          "[mgpu][operator-mgpu][hash_join][build_probe][broadcast][dynamic_filter][gpu_execution]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("broadcast-dynfilter");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto build_dir = tmp / "build";
  auto probe_dir = tmp / "probe";
  generate_small_build_side(build_dir);
  generate_large_probe_side(probe_dir);

  scoped_log_dir log_dir(tmp / "logs", "debug");

  mgpu_env_params params{};
  params.cache   = "none";
  auto yaml_path = tmp / "broadcast-dynfilter.yaml";
  write_mgpu_yaml(yaml_path, params);
  scoped_mgpu_env env(yaml_path);

  // The build-side predicate is required evidence: discovery refuses an unfiltered base-relation
  // build (its filter would keep every probe row), so a bare read_parquet build never wires a
  // producer and this test would assert against a query that cannot publish.
  auto inner_query =
    "SELECT probe.k, probe.v, build.v AS build_v "
    "FROM read_parquet('" +
    parquet_glob(probe_dir) +
    "') AS probe "
    "JOIN read_parquet('" +
    parquet_glob(build_dir) +
    "') AS build "
    "  ON probe.k = build.k "
    "WHERE build.k % 2 = 0 "
    "ORDER BY probe.k, probe.v "
    "LIMIT 100";

  {
    auto con = env.make_connection();
    // Dynamic-filter pushdown is on by default; keep the result a correctness oracle.
    require_gpu_matches_cpu(con, inner_query, /*force_cpu_reference=*/true);
  }

  INFO("log dir: " << log_dir.path());
  // Broadcast engaged AND the publication winner's success summary appears; no delivery may log
  // the not-published diagnostic.
  REQUIRE(log_dir_contains(log_dir.path(), "[broadcast]"));
  REQUIRE(log_dir_contains(log_dir.path(), "dynamic-filter publication:"));
  REQUIRE_FALSE(log_dir_contains(log_dir.path(), "dynamic filter NOT published"));

  fs::remove_all(tmp, ec);
}
