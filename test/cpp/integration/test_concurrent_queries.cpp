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

// Concurrent transparent GPU execution across connections.
//
// The query-lifecycle gate is a counted slot pool sized by
// sirius.executor.scan_manager.max_concurrent_queries (default 1 = the
// historical single-flight behavior; every other test in this binary runs
// that way). This test raises it and proves two things the single-flight
// suite cannot:
//   1. results stay correct when queries genuinely overlap, and
//   2. overlap actually happened (query_lifecycle_peak() > 1) — otherwise a
//      regression that silently re-serializes the gate would pass on
//      correctness alone.
//
// Assertions from worker threads are collected and REQUIREd on the main
// thread: Catch2 v2 assertion macros are not thread-safe.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/client_context.hpp>
#include <sirius_context.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr std::int64_t kRows    = 2'000'000;
constexpr std::int64_t kDimRows = 97;

// Bisection knobs (deadlock triage runs the same binary at several points of
// the slots x workers grid without rebuilding). Defaults are the shipped
// test shape.
int env_int(const char* name, int fallback)
{
  if (const char* v = std::getenv(name); v != nullptr) { return std::atoi(v); }
  return fallback;
}
int slots() { return env_int("SIRIUS_TEST_CONCURRENCY_SLOTS", 4); }
int workers() { return env_int("SIRIUS_TEST_CONCURRENCY_WORKERS", 4); }
int iters_per_worker() { return env_int("SIRIUS_TEST_CONCURRENCY_ITERS", 6); }
int pipeline_threads() { return env_int("SIRIUS_TEST_PIPELINE_THREADS", 4); }
int creator_threads() { return env_int("SIRIUS_TEST_CREATOR_THREADS", 2); }

// integration.yaml (1 GPU, half the card) plus the concurrency knob under
// test. Written fresh per run so the test owns its whole config.
std::string concurrent_config_yaml()
{
  return R"(sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.5
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: 32000000000
      initial_number_pools: 10
      pool_size: 512
      block_size: 1048576
  executor:
    pipeline:
      num_threads: )" +
         std::to_string(pipeline_threads()) + R"(
    task_creator:
      num_threads: )" +
         std::to_string(creator_threads()) + R"(
    downgrade:
      num_threads: 1
      monitor_period: 10ms
    scan_manager:
      max_concurrent_queries: )" +
         std::to_string(slots()) + R"(
  operator_params:
    scan_task_batch_size: 100000000
    max_sort_partition_bytes: 0
    hash_partition_bytes: 100000000
    concat_batch_bytes: 100000000
    max_build_hash_table_bytes: 90000000
)";
}

class scoped_config_env {
 public:
  explicit scoped_config_env(const fs::path& config_path)
  {
    if (const char* current = std::getenv("SIRIUS_CONFIG_FILE"); current != nullptr) {
      original_     = current;
      had_original_ = true;
    }
    // The shared env sets SIRIUS_DISABLE=1 after creating its DB so stray
    // DuckDB instances in other tests skip SiriusContext creation. This test
    // WANTS its own SiriusContext, so lift the kill switch for its window.
    had_disable_ = std::getenv("SIRIUS_DISABLE") != nullptr;
    unsetenv("SIRIUS_DISABLE");
    setenv("SIRIUS_CONFIG_FILE", config_path.string().c_str(), 1);
  }
  ~scoped_config_env()
  {
    if (had_original_) {
      setenv("SIRIUS_CONFIG_FILE", original_.c_str(), 1);
    } else {
      unsetenv("SIRIUS_CONFIG_FILE");
    }
    if (had_disable_) { setenv("SIRIUS_DISABLE", "1", 1); }
  }

 private:
  std::string original_;
  bool had_original_ = false;
  bool had_disable_  = false;
};

std::string materialize(duckdb::QueryResult& result) { return result.ToString(); }

// Every query carries an ORDER BY so a result compares by string equality.
const std::vector<std::string>& query_shapes()
{
  static const std::vector<std::string> queries = {
    // grouped aggregate over the fact table
    "SELECT k, count(*) AS c, sum(v) AS s FROM fact GROUP BY k ORDER BY k",
    // join against the dim table with a filter on the build side
    "SELECT d.bucket, count(*) AS c FROM fact f JOIN dim d ON f.k = d.k "
    "WHERE d.w > 250 GROUP BY d.bucket ORDER BY d.bucket",
    // selective filter + global aggregate
    "SELECT count(*) AS c, sum(v) AS s, min(id) AS lo, max(id) AS hi FROM fact WHERE k < 13",
    // two-key grouped aggregate with a projection expression
    "SELECT k % 10 AS kb, v % 7 AS vb, count(*) AS c, sum(id) AS s FROM fact "
    "GROUP BY kb, vb ORDER BY kb, vb",
  };
  return queries;
}

}  // namespace

TEST_CASE("concurrent transparent GPU queries overlap and stay correct",
          "[concurrency][isolated_context]")
{
  // Own DB + config: the shared env is paused by the [isolated_context]
  // listener, so the SiriusContext built here reads the raised slot count.
  auto config_path = fs::temp_directory_path() / "sirius_concurrent_queries_test.yaml";
  {
    std::ofstream out(config_path);
    out << concurrent_config_yaml();
  }
  scoped_config_env env_guard(config_path);
  duckdb::DuckDB db(nullptr);

  // Seed data once as PARQUET, then expose it through views: the GPU scan
  // serves parquet (and file-backed duckdb) sources; an in-memory CREATE
  // TABLE would plan-time-fall-back every query to CPU and reduce this test
  // to concurrent DuckDB. The COPY itself is CPU-side setup.
  auto const fact_path = fs::temp_directory_path() / "sirius_concurrent_fact.parquet";
  auto const dim_path  = fs::temp_directory_path() / "sirius_concurrent_dim.parquet";
  {
    duckdb::Connection con(db);
    auto r1 = con.Query("COPY (SELECT range AS id, range % " + std::to_string(kDimRows) +
                        " AS k, (range * 13) % 1000 AS v FROM range(" + std::to_string(kRows) +
                        ")) TO '" + fact_path.string() + "' (FORMAT parquet)");
    REQUIRE_FALSE(r1->HasError());
    auto r2 = con.Query(
      "COPY (SELECT range AS k, (range * 7) % 500 AS w, range % 5 AS bucket "
      "FROM range(" +
      std::to_string(kDimRows) + ")) TO '" + dim_path.string() + "' (FORMAT parquet)");
    REQUIRE_FALSE(r2->HasError());
    auto r3 =
      con.Query("CREATE VIEW fact AS SELECT * FROM read_parquet('" + fact_path.string() + "')");
    REQUIRE_FALSE(r3->HasError());
    auto r4 =
      con.Query("CREATE VIEW dim AS SELECT * FROM read_parquet('" + dim_path.string() + "')");
    REQUIRE_FALSE(r4->HasError());
  }

  // Single-threaded reference pass (still transparently GPU-executed; the
  // point of the test is overlap-correctness, not GPU-vs-CPU — the rest of
  // the suite covers that). The SiriusContext handle is captured here, from a
  // connection that has executed: registered_state gains "sirius_state"
  // lazily on first execution, so a fresh connection would hand back null.
  std::vector<std::string> reference;
  duckdb::shared_ptr<duckdb::SiriusContext> sirius_ctx;
  {
    duckdb::Connection con(db);
    for (const auto& q : query_shapes()) {
      auto r = con.Query(q);
      REQUIRE_FALSE(r->HasError());
      reference.push_back(materialize(*r));
    }
    sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
  }

  // Concurrent phase: kWorkers threads, own connection each, barrier-synced
  // start, each cycling the query shapes from a different offset so distinct
  // shapes overlap.
  std::mutex failures_mutex;
  std::vector<std::string> failures;
  std::atomic<int> ready{0};
  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool go = false;

  auto worker = [&](int wid) {
    duckdb::Connection con(db);
    {
      std::unique_lock<std::mutex> lock(start_mutex);
      ++ready;
      start_cv.wait(lock, [&] { return go; });
    }
    for (int i = 0; i < iters_per_worker(); ++i) {
      const auto qi   = (wid + i) % query_shapes().size();
      const auto& sql = query_shapes()[qi];
      auto r          = con.Query(sql);
      if (r->HasError()) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("worker " + std::to_string(wid) + " iter " + std::to_string(i) +
                           " query " + std::to_string(qi) + " ERROR: " + r->GetError());
        continue;
      }
      auto got = materialize(*r);
      if (got != reference[qi]) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("worker " + std::to_string(wid) + " iter " + std::to_string(i) +
                           " query " + std::to_string(qi) + " WRONG RESULT:\n--- got ---\n" + got +
                           "\n--- expected ---\n" + reference[qi]);
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(workers());
  for (int w = 0; w < workers(); ++w) {
    threads.emplace_back(worker, w);
  }
  while (ready.load() < workers()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  {
    std::lock_guard<std::mutex> lock(start_mutex);
    go = true;
  }
  start_cv.notify_all();
  for (auto& t : threads) {
    t.join();
  }

  for (const auto& f : failures) {
    UNSCOPED_INFO(f);
  }
  REQUIRE(failures.empty());

  // Prove the gate actually admitted overlapping windows: a silent
  // re-serialization (or a config knob that stopped reaching the gate) would
  // otherwise pass on correctness alone.
  INFO("query_lifecycle_peak = " << sirius_ctx->query_lifecycle_peak());
  if (slots() > 1 && workers() > 1) { REQUIRE(sirius_ctx->query_lifecycle_peak() > 1); }

  // And prove the overlapping queries actually EXECUTED on the GPU: a
  // plan-time fallback regression would otherwise reduce this test to
  // concurrent DuckDB CPU execution, which proves nothing about Sirius.
  const auto stats = sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << stats.executions << " fallbacks=" << stats.fallbacks
                     << " runtime_fallbacks=" << stats.runtime_fallbacks);
  const auto total_queries =
    query_shapes().size() + static_cast<std::size_t>(workers()) * iters_per_worker();
  REQUIRE(stats.executions >= total_queries);
  REQUIRE(stats.runtime_fallbacks == 0);
}
