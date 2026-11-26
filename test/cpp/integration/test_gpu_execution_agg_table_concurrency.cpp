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

/**
 * @file test_gpu_execution_agg_table_concurrency.cpp
 * @brief The table path of GROUP BY under concurrency: many pipeline threads, several queries at
 *        once, several GPUs, and a GPU with little memory.
 *
 * One table serves every task of an aggregate on its GPU, and there is one table per GPU. A group
 * whose rows are spread over many tasks, or over several GPUs, must come out once with the right
 * values. The inputs here put every key into every scan batch (one batch per row group), so each
 * group is fed by every task that folds a batch, on whichever GPU the task runs.
 *
 * Environments:
 *  - `[agg_table_threads]` and `[agg_table_pressure]` cases build a Sirius instance of their own
 *    from a YAML written here (sixteen pipeline threads; four concurrent queries; a 120 MB GPU
 *    limit). They carry no [integration] tag, so the Catch2 listener leaves every shared
 *    environment paused, as for the other tests with a private instance.
 *  - `[integration][multi_gpu]` cases use the shared two-GPU environment (integration-2gpu.yaml).
 *  - The `[multi_gpu]` case without [integration] builds an instance over every visible GPU.
 *
 * Multi-GPU cases return early when fewer than two GPUs are visible. scripts/run_unit_tests.py
 * runs them in its `multi_gpu` step, in one process that sees all GPUs.
 *
 * Races are not deterministic, so each check is repeated; a failure that appears in one
 * repetition out of several is a finding, not noise.
 */

#include <cuda_runtime.h>

#include <catch.hpp>
#include <unistd.h>
#include <utils/agg_table_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using sirius::test::scoped_sirius_setting;
using sirius::test::agg_table::compare_on_off;
using sirius::test::agg_table::kEnableSetting;
using sirius::test::agg_table::kPartitionBytes;
using sirius::test::agg_table::kScanBatch;

namespace {

/// What the private instances vary. Everything else follows integration.yaml.
struct engine_config {
  int num_gpus               = 1;
  int pipeline_threads       = 4;
  int max_concurrent_queries = 1;
  /// One line of `sirius.memory.gpu`: the share or the bytes of each GPU the instance takes.
  std::string gpu_limit = "usage_limit_fraction: 0.2";
  /// hash_partition_bytes, concat_batch_bytes and max_build_hash_table_bytes.
  std::uint64_t operator_bytes = 100000000;
};

fs::path write_config(fs::path const& directory, engine_config const& config)
{
  auto const path = directory / "sirius.yaml";
  std::ofstream out(path);
  out << "sirius:\n"
      << "  max_concurrent_queries: " << config.max_concurrent_queries << "\n"
      << "  topology:\n"
      << "    num_gpus: " << config.num_gpus << "\n"
      << "  memory:\n"
      << "    gpu:\n"
      << "      " << config.gpu_limit << "\n"
      << "      reservation_limit_fraction: 1.0\n"
      << "    host:\n"
      << "      capacity_bytes: 32000000000\n"
      << "      initial_number_pools: 2\n"
      << "      pool_size: 128\n"
      << "      block_size: 1048576\n"
      << "  executor:\n"
      << "    pipeline:\n"
      << "      num_threads: " << config.pipeline_threads << "\n"
      << "    task_creator:\n"
      << "      num_threads: 2\n"
      << "    downgrade:\n"
      << "      num_threads: 1\n"
      << "      monitor_period: 10ms\n"
      << "  operator_params:\n"
      << "    scan_task_batch_size: 100000000\n"
      << "    max_sort_partition_bytes: 0\n"
      << "    hash_partition_bytes: " << config.operator_bytes << "\n"
      << "    concat_batch_bytes: " << config.operator_bytes << "\n"
      << "    max_build_hash_table_bytes: " << config.operator_bytes << "\n";
  out.close();
  REQUIRE(fs::exists(path));
  return path;
}

/// Only one Sirius instance may hold the extension at a time.
void pause_shared_envs()
{
  for (auto* env : {sirius::test::g_shared_env,
                    sirius::test::g_integration_env,
                    sirius::test::g_integration_env_2gpu}) {
    if (env != nullptr && env->is_active()) { env->pause(); }
  }
}

void run_ok(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

/// A private Sirius instance with a file-backed database, removed with the instance.
class local_engine {
 public:
  local_engine(std::string const& name, engine_config const& config)
    : _directory(fs::temp_directory_path() /
                 ("sirius-agg-table-" + name + "-" + std::to_string(::getpid())))
  {
    pause_shared_envs();
    std::error_code ignored;
    fs::remove_all(_directory, ignored);
    fs::create_directories(_directory);
    _env = std::make_unique<sirius::test::shared_test_env>(write_config(_directory, config));
    con  = connect();
  }

  ~local_engine()
  {
    con.reset();
    _env.reset();
    std::error_code ignored;
    fs::remove_all(_directory, ignored);
  }

  local_engine(local_engine const&)            = delete;
  local_engine& operator=(local_engine const&) = delete;

  /// A new connection that reads and writes the instance's file-backed database.
  std::unique_ptr<duckdb::Connection> connect()
  {
    auto connection = std::make_unique<duckdb::Connection>(_env->make_connection());
    run_ok(*connection,
           "ATTACH IF NOT EXISTS '" + (_directory / "data.db").string() + "' AS agg_data");
    run_ok(*connection, "USE agg_data");
    return connection;
  }

  std::unique_ptr<duckdb::Connection> con;

 private:
  fs::path _directory;
  std::unique_ptr<sirius::test::shared_test_env> _env;
};

/**
 * @brief Creates the inputs on @p con's current database, on the CPU, and checkpoints them.
 *
 * `shared_keys`: every row group holds all 1000 ordinary keys, the keys -1 and -2, NULL keys and
 * one key on a quarter of the rows; `big` passes INT64 when summed per key.
 * `growing`: @p rows / 2 keys, each seen twice, half the table apart.
 */
void create_tables(duckdb::Connection& con, std::uint64_t rows)
{
  auto const n = std::to_string(rows);
  run_ok(con, "SET gpu_execution = false");
  run_ok(con,
         "CREATE TABLE shared_keys AS SELECT CASE WHEN i % 40 = 0 THEN -1 WHEN i % 40 = 1 THEN -2 "
         "  WHEN i % 40 = 2 THEN NULL WHEN i % 4 = 3 THEN 7 ELSE hash(i) % 1000 END::BIGINT AS k, "
         "(i % 1000)::BIGINT AS v, "
         "CASE WHEN i % 5 = 0 THEN 4000000000000000000 ELSE i % 100 END::BIGINT AS big, "
         "((i % 9000) / 100.0)::DECIMAL(12,2) AS p FROM range(" +
           n + ") r(i)");
  run_ok(con,
         "CREATE TABLE growing AS SELECT (i % (" + n +
           " // 2))::BIGINT AS k, i::BIGINT AS v, (i % 3)::INTEGER AS w FROM range(" + n +
           ") r(i)");
  run_ok(con, "CHECKPOINT");
  run_ok(con, "SET gpu_execution = true");
}

constexpr char kSharedAll[] =
  "SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx FROM shared_keys GROUP BY k";
constexpr char kSharedWide[] =
  "SELECT k, count(*) AS c, sum(big) AS sb, sum(p) AS sp, min(p) AS mnp, max(p) AS mxp "
  "FROM shared_keys GROUP BY k";
constexpr char kSharedCount[] = "SELECT k, count(*) AS c FROM shared_keys GROUP BY k";
constexpr char kGrowingTotals[] =
  "SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s, sum(mx - mn) AS spread, sum(sw) AS sw "
  "FROM (SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx, sum(w) AS sw "
  "FROM growing GROUP BY k) t";
constexpr char kGrowingSample[] =
  "SELECT k, count(*) AS c, sum(v) AS s FROM growing GROUP BY k HAVING k % 9973 = 5";

/// The three-way comparison of every query above, @p repeats times, with one scan batch per row
/// group and then also with a small partition target for the exchange of partial rows.
void check_queries(duckdb::Connection& con, int repeats)
{
  scoped_sirius_setting const scan_batches{con, kScanBatch, std::uint64_t{1}};
  for (int repeat = 0; repeat < repeats; repeat++) {
    INFO("repeat " << repeat);
    compare_on_off(con, kSharedAll);
    compare_on_off(con, kSharedWide);
    compare_on_off(con, kSharedCount);
    compare_on_off(con, kGrowingTotals);
    compare_on_off(con, kGrowingSample);
  }
  scoped_sirius_setting const partitions{con, kPartitionBytes, std::uint64_t{1} << 20};
  for (int repeat = 0; repeat < repeats; repeat++) {
    INFO("small partitions, repeat " << repeat);
    compare_on_off(con, kSharedAll);
    compare_on_off(con, kGrowingTotals);
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// One GPU, many tasks
//===----------------------------------------------------------------------===//

TEST_CASE("agg table - sixteen pipeline threads fold the batches of one input",
          "[agg_table][gpu_execution][aggregate][agg_table_threads]")
{
  // 3000000 rows are 25 row groups: 25 scan batches for 16 threads.
  local_engine engine("threads", {.pipeline_threads = 16});
  create_tables(*engine.con, 3000000);
  check_queries(*engine.con, 5);
}

TEST_CASE("agg table - four queries at once, with the table path on and off",
          "[agg_table][gpu_execution][aggregate][agg_table_threads]")
{
  local_engine engine("queries", {.pipeline_threads = 8, .max_concurrent_queries = 4});
  create_tables(*engine.con, 2000000);
  sirius::test::agg_table::require_enable_setting(*engine.con);

  std::vector<std::string> const queries{kSharedAll, kSharedWide, kGrowingTotals, kGrowingSample};
  std::vector<std::vector<std::vector<std::string>>> expected;
  run_ok(*engine.con, "SET gpu_execution = false");
  for (auto const& query : queries) {
    auto result = engine.con->Query(query);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    expected.push_back(sirius::test::collect_rows(*result, true));
  }
  run_ok(*engine.con, "SET gpu_execution = true");

  // Eight connections: each query once with the table path on and once with it off, all at the
  // same time, six rounds each. Catch2 assertions are not thread-safe, so the workers only
  // record what they saw.
  constexpr std::size_t kRounds = 6;
  struct worker {
    std::unique_ptr<duckdb::Connection> con;
    std::size_t query = 0;
    bool table_path   = false;
    std::vector<std::string> failures;
  };
  std::vector<worker> workers(2 * queries.size());
  for (std::size_t i = 0; i < workers.size(); i++) {
    workers[i].con        = engine.connect();
    workers[i].query      = i / 2;
    workers[i].table_path = i % 2 == 0;
    run_ok(
      *workers[i].con,
      std::string{"SET "} + kEnableSetting + " = " + (workers[i].table_path ? "true" : "false"));
    run_ok(*workers[i].con, std::string{"SET "} + kScanBatch + " = 1");
  }

  auto const before = sirius::test::get_transparent_execution_stats(*engine.con);
  {
    std::vector<std::thread> threads;
    threads.reserve(workers.size());
    for (auto& w : workers) {
      threads.emplace_back([&w, &queries, &expected] {
        for (std::size_t round = 0; round < kRounds; round++) {
          try {
            auto result = w.con->Query(queries[w.query]);
            if (!result) {
              w.failures.push_back("round " + std::to_string(round) + ": no result");
            } else if (result->HasError()) {
              w.failures.push_back("round " + std::to_string(round) + ": " + result->GetError());
            } else if (sirius::test::collect_rows(*result, true) != expected[w.query]) {
              w.failures.push_back("round " + std::to_string(round) + ": rows differ from the CPU");
            }
          } catch (std::exception const& error) {
            w.failures.push_back("round " + std::to_string(round) + ": " + error.what());
          }
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  auto const after = sirius::test::get_transparent_execution_stats(*engine.con);

  for (auto const& w : workers) {
    INFO("query " << queries[w.query] << " with " << kEnableSetting << " = "
                  << (w.table_path ? "true" : "false"));
    for (auto const& failure : w.failures) {
      UNSCOPED_INFO(failure);
    }
    CHECK(w.failures.empty());
  }
  // Every run was a GPU execution: none fell back at plan time or at runtime.
  CHECK(after.executions == before.executions + workers.size() * kRounds);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  for (auto& w : workers) {
    w.con.reset();
  }
}

//===----------------------------------------------------------------------===//
// A GPU with little memory
//===----------------------------------------------------------------------===//

TEST_CASE("agg table - a 120 MB GPU limit: two million groups do not fit the table's share",
          "[agg_table][gpu_execution][aggregate][agg_table_pressure]")
{
  // The table may hold a share of the GPU's memory space (a quarter by default), here 30 MB, and
  // two million groups with five aggregates need several times that: most rows are returned by
  // the fold and take the existing path, under real memory pressure on every other operator.
  local_engine engine("pressure",
                      {.pipeline_threads = 8,
                       .gpu_limit        = "usage_limit_bytes: 120000000",
                       .operator_bytes   = 8000000});
  create_tables(*engine.con, 4000000);
  check_queries(*engine.con, 2);
  // The engine builds no table below 64 MiB, so the default share (30 MB) leaves every row on
  // the old path. With 0.6 of the space (72 MB) a table exists, cannot hold the groups, and
  // lives under the memory pressure of every other operator.
  scoped_sirius_setting const share{
    *engine.con, sirius::test::agg_table::kMemoryFractionSetting, std::string_view{"0.6"}};
  auto const before = sirius::test::agg_table::table_stats();
  check_queries(*engine.con, 2);
  auto const delta = sirius::test::agg_table::table_stats().since(before);
  INFO("folded batches " << delta.folded_batches << ", boundary flushes " << delta.boundary_flushes
                         << ", batches on the old path " << delta.legacy_batches);
  CHECK(delta.folded_batches > 0);
}

//===----------------------------------------------------------------------===//
// Several GPUs
//===----------------------------------------------------------------------===//

TEST_CASE("agg table - one group fed from two GPUs",
          "[integration][gpu_execution][aggregate][agg_table][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  auto* env = sirius::test::acquire_integration_env_for(2);
  REQUIRE(env != nullptr);
  REQUIRE(env->is_active());
  auto const database =
    fs::temp_directory_path() / ("sirius_agg_table_mgpu_" + std::to_string(::getpid()) + ".db");
  std::error_code ignored;
  fs::remove(database, ignored);
  fs::remove(database.string() + ".wal", ignored);
  {
    auto con = std::make_unique<duckdb::Connection>(env->make_connection());
    run_ok(*con, "ATTACH '" + database.string() + "' AS agg_table_mgpu");
    run_ok(*con, "USE agg_table_mgpu");
    // 49 row groups: enough batches for both GPUs to fold every key.
    create_tables(*con, 6000000);
    auto const before = sirius::test::agg_table::table_stats();
    check_queries(*con, 3);
    // The point of the case: tables on both GPUs took batches of the same groups.
    auto const delta = sirius::test::agg_table::table_stats().since(before);
    INFO("folded batches: " << delta.folded_batches
                            << ", GPUs that folded: " << delta.gpus_folded());
    CHECK(delta.gpus_folded() == 2);
    con->Query("USE memory");
    con->Query("DETACH agg_table_mgpu");
  }
  fs::remove(database, ignored);
  fs::remove(database.string() + ".wal", ignored);
}

TEST_CASE("agg table - one group fed from every visible GPU",
          "[agg_table][gpu_execution][aggregate][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  int gpus = 0;
  REQUIRE(cudaGetDeviceCount(&gpus) == cudaSuccess);
  // Two pipeline threads per GPU, and 16 row groups per GPU so every GPU folds every key.
  local_engine engine("all-gpus",
                      {.num_gpus         = gpus,
                       .pipeline_threads = std::max(8, 2 * gpus),
                       .gpu_limit        = "usage_limit_fraction: 0.4"});
  create_tables(*engine.con, static_cast<std::uint64_t>(gpus) * 16 * 122880);
  auto const before = sirius::test::agg_table::table_stats();
  check_queries(*engine.con, 3);
  // Tables on several GPUs took batches of the same groups. Which GPUs get tasks is the
  // scheduler's choice, so only "more than one" is required; the count is reported.
  auto const delta = sirius::test::agg_table::table_stats().since(before);
  INFO("folded batches: " << delta.folded_batches << ", GPUs that folded: " << delta.gpus_folded()
                          << " of " << gpus);
  CHECK(delta.gpus_folded() >= 2);
  if (delta.gpus_folded() < static_cast<std::size_t>(gpus)) {
    WARN("tables folded on " << delta.gpus_folded() << " of " << gpus << " GPUs");
  }
}
