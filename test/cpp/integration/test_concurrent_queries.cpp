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

// Default absolute GPU pool. Small on purpose: the pool is allocated EAGERLY
// at init, and these tests run on shared GPUs (parallel agent sessions, CI
// neighbors). The old default — usage_limit_fraction 0.5, ~half the card —
// flaked with bad_alloc whenever a co-tenant held memory at env-construction
// time. Every scenario here was designed against a 3 GiB pool, so 4 GiB is
// roomy; override via SIRIUS_TEST_CONCURRENCY_POOL_BYTES for bigger sweeps.
std::uint64_t default_pool_bytes()
{
  if (const char* v = std::getenv("SIRIUS_TEST_CONCURRENCY_POOL_BYTES")) {
    return std::strtoull(v, nullptr, 10);
  }
  return 4ULL << 30;
}

// integration.yaml (1 GPU, small absolute pool) plus the concurrency knob
// under test. Written fresh per run so the test owns its whole config.
std::string concurrent_config_yaml(std::uint64_t gpu_pool_bytes = 0)
{
  // gpu_pool_bytes == 0: the shared-GPU-friendly default above. Non-zero: an
  // absolute cap, used by the memory-pressure scenarios to force downgrades.
  if (gpu_pool_bytes == 0) { gpu_pool_bytes = default_pool_bytes(); }
  std::string gpu_mem = "      usage_limit_bytes: " + std::to_string(gpu_pool_bytes);
  return R"(sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
)" + gpu_mem +
         R"(
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: 8000000000
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

// Shared fixture: own DB + config, parquet-backed views, single-threaded
// reference results, and the SiriusContext handle (captured from a connection
// that has executed — registered_state gains "sirius_state" lazily on first
// execution, so a fresh connection would hand back null).
struct concurrent_env {
  fs::path config_path;
  std::unique_ptr<scoped_config_env> env_guard;
  std::unique_ptr<duckdb::DuckDB> db;
  fs::path fact_path;
  fs::path dim_path;
  std::vector<std::string> reference;
  duckdb::shared_ptr<duckdb::SiriusContext> sirius_ctx;

  explicit concurrent_env(std::uint64_t gpu_pool_bytes = 0, std::int64_t rows = kRows)
  {
    config_path = fs::temp_directory_path() / "sirius_concurrent_queries_test.yaml";
    {
      std::ofstream out(config_path);
      out << concurrent_config_yaml(gpu_pool_bytes);
    }
    env_guard = std::make_unique<scoped_config_env>(config_path);
    db        = std::make_unique<duckdb::DuckDB>(nullptr);

    // Seed data as PARQUET, exposed through views: the GPU scan serves
    // parquet (and file-backed duckdb) sources; an in-memory CREATE TABLE
    // would plan-time-fall-back every query to CPU and reduce these tests to
    // concurrent DuckDB. The COPY itself is CPU-side setup.
    fact_path = fs::temp_directory_path() / "sirius_concurrent_fact.parquet";
    dim_path  = fs::temp_directory_path() / "sirius_concurrent_dim.parquet";
    duckdb::Connection con(*db);
    auto r1 = con.Query("COPY (SELECT range AS id, range % " + std::to_string(kDimRows) +
                        " AS k, (range * 13) % 1000 AS v FROM range(" + std::to_string(rows) +
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

    // Single-threaded reference pass (still transparently GPU-executed; the
    // point of these tests is overlap-correctness, not GPU-vs-CPU — the rest
    // of the suite covers that).
    for (const auto& q : query_shapes()) {
      auto r = con.Query(q);
      REQUIRE_FALSE(r->HasError());
      reference.push_back(materialize(*r));
    }
    sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
  }
};

// Run @p n_workers threads (own connection each, barrier-synced start), each
// executing @p per_worker(worker_id, iteration) -> SQL, checking against
// @p expected(worker_id, iteration) (empty string = only require no error).
// Returns the failure descriptions (Catch2 v2 assertions are not thread-safe,
// so workers only collect).
template <typename SqlFn, typename ExpectFn>
std::vector<std::string> run_workers(
  duckdb::DuckDB& db, int n_workers, int n_iters, SqlFn per_worker, ExpectFn expected)
{
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
    for (int i = 0; i < n_iters; ++i) {
      const std::string sql = per_worker(wid, i);
      auto r                = con.Query(sql);
      if (r->HasError()) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("worker " + std::to_string(wid) + " iter " + std::to_string(i) +
                           " ERROR: " + r->GetError());
        continue;
      }
      const std::string want = expected(wid, i);
      if (want.empty()) { continue; }
      auto got = materialize(*r);
      if (got != want) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("worker " + std::to_string(wid) + " iter " + std::to_string(i) +
                           " WRONG RESULT:\n--- got ---\n" + got + "\n--- expected ---\n" + want);
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(n_workers));
  for (int w = 0; w < n_workers; ++w) {
    threads.emplace_back(worker, w);
  }
  while (ready.load() < n_workers) {
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
  return failures;
}

void require_no_failures(const std::vector<std::string>& failures)
{
  for (const auto& f : failures) {
    UNSCOPED_INFO(f);
  }
  REQUIRE(failures.empty());
}

}  // namespace

TEST_CASE("concurrent transparent GPU queries overlap and stay correct",
          "[concurrency][isolated_context]")
{
  concurrent_env env;

  auto failures = run_workers(
    *env.db,
    workers(),
    iters_per_worker(),
    [&](int wid, int i) { return query_shapes()[(wid + i) % query_shapes().size()]; },
    [&](int wid, int i) { return env.reference[(wid + i) % query_shapes().size()]; });
  require_no_failures(failures);

  // Prove the gate actually admitted overlapping windows: a silent
  // re-serialization (or a config knob that stopped reaching the gate) would
  // otherwise pass on correctness alone.
  INFO("query_lifecycle_peak = " << env.sirius_ctx->query_lifecycle_peak());
  if (slots() > 1 && workers() > 1) { REQUIRE(env.sirius_ctx->query_lifecycle_peak() > 1); }

  // And prove the overlapping queries actually EXECUTED on the GPU: a
  // plan-time fallback regression would otherwise reduce this test to
  // concurrent DuckDB CPU execution, which proves nothing about Sirius.
  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << stats.executions << " fallbacks=" << stats.fallbacks
                     << " runtime_fallbacks=" << stats.runtime_fallbacks);
  const auto total_queries =
    query_shapes().size() + static_cast<std::size_t>(workers()) * iters_per_worker();
  REQUIRE(stats.executions >= total_queries);
  REQUIRE(stats.runtime_fallbacks == 0);
}

TEST_CASE("a runtime-failing query beside healthy concurrent queries",
          "[concurrency][isolated_context]")
{
  // Worker 0 repeatedly runs a shape that FAILS on the GPU at runtime
  // (DISTINCT aggregates throw mid-execution) and completes via DuckDB CPU
  // fallback; the other workers run supported shapes. This exercises the
  // per-query failure containment (terminate_query -> drain_after_error ->
  // run_mandatory_cleanup) WHILE other queries hold slots — historically the
  // shape of the 4x4 wedge. Every result, including the failing worker's
  // CPU-fallback results, must stay correct, and the healthy queries must
  // still run on the GPU.
  concurrent_env env;

  const std::string distinct_sql =
    "SELECT count(DISTINCT k) AS dk, count(DISTINCT v % 101) AS dv FROM fact ORDER BY 1";
  std::string distinct_ref;
  {
    duckdb::Connection con(*env.db);
    auto r = con.Query(distinct_sql);
    REQUIRE_FALSE(r->HasError());
    distinct_ref = materialize(*r);
  }
  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();

  auto failures = run_workers(
    *env.db,
    workers(),
    iters_per_worker(),
    [&](int wid, int i) {
      if (wid == 0) { return distinct_sql; }
      return query_shapes()[(wid + i) % query_shapes().size()];
    },
    [&](int wid, int i) {
      if (wid == 0) { return distinct_ref; }
      return env.reference[(wid + i) % query_shapes().size()];
    });
  require_no_failures(failures);

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << stats.executions << " fallbacks=" << stats.fallbacks
                     << " runtime_fallbacks=" << stats.runtime_fallbacks);
  // Worker 0's every iteration must have attempted the GPU and fallen back at
  // runtime; the healthy workers must not have been dragged down with it.
  REQUIRE(stats.runtime_fallbacks - stats_before.runtime_fallbacks >=
          static_cast<std::uint64_t>(iters_per_worker()));
  const auto healthy = static_cast<std::size_t>(workers() - 1) * iters_per_worker();
  REQUIRE(stats.executions - stats_before.executions >= healthy);
}

TEST_CASE("concurrent queries stay correct under GPU memory pressure",
          "[concurrency][memory_pressure][isolated_context]")
{
  // A deliberately small GPU pool with a larger fact table: overlapping
  // queries now contend for reservations and push batches through the
  // downgrade path while other queries end (the global drain cancels peers'
  // pending spills — the known structural debt this test keeps honest).
  // Results must stay correct; runtime fallbacks are reported but tolerated
  // (an OOM-retry cap trip completes via CPU with correct results).
  constexpr std::uint64_t kSmallPool = 3ULL * 1024 * 1024 * 1024;  // 3 GiB
  concurrent_env env(kSmallPool, /*rows=*/20'000'000);

  auto failures = run_workers(
    *env.db,
    workers(),
    iters_per_worker(),
    [&](int wid, int i) { return query_shapes()[(wid + i) % query_shapes().size()]; },
    [&](int wid, int i) { return env.reference[(wid + i) % query_shapes().size()]; });
  require_no_failures(failures);

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << stats.executions << " fallbacks=" << stats.fallbacks
                     << " runtime_fallbacks=" << stats.runtime_fallbacks
                     << " peak=" << env.sirius_ctx->query_lifecycle_peak());
  REQUIRE(stats.executions >= query_shapes().size());
}

TEST_CASE(
  "query teardown races in-flight downgrades without resurrecting or dereferencing "
  "a dead plan",
  "[concurrency][memory_pressure][teardown_races][isolated_context]")
{
  // The B1/B5 interlock. A tiny GPU pool keeps the downgrade monitor firing
  // (10ms period), so TIER-2 sweeps repeatedly extract queued tasks from the
  // shared scheduler queue and carry them across blocking conversion windows.
  // Meanwhile worker 0 runs a shape that FAILS on the GPU at runtime every
  // iteration, so error-path teardowns (quiesce -> drains -> plan destruction)
  // land continuously while peers' tasks are mid-extraction. Pre-fix, two
  // things could go wrong in that window:
  //   - B1: the TIER-2 RAII re-push re-derived the task's queue keys through
  //     the plan (pipe->get_source()->type) after the owning query destroyed
  //     it, and could resurrect the task behind its own drain;
  //   - B5: the engine (and with it the plan) was destroyed BEFORE the
  //     window's mandatory cleanup drained the queues, so every task the
  //     drains destroyed walked freed operator pointers in
  //     ~gpu_pipeline_task -> mark_task_completed -> notify_downstream_pipelines.
  // Post-fix the plan is parked until the drains complete and the re-push
  // uses extraction-time keys, so this must survive with correct results.
  //
  // The GPU pool is deliberately tiny (1.5 GB, vs the harness's 4 GiB shared
  // default; the host pool is the harness default 8 GB) so several test runs
  // can coexist on a shared box and pressure stays high.
  constexpr std::uint64_t kTinyPool = 1'500'000'000ULL;  // 1.5 GB
  concurrent_env env(kTinyPool, /*rows=*/20'000'000);

  const std::string distinct_sql =
    "SELECT count(DISTINCT k) AS dk, count(DISTINCT v % 101) AS dv FROM fact ORDER BY 1";
  std::string distinct_ref;
  {
    duckdb::Connection con(*env.db);
    auto r = con.Query(distinct_sql);
    REQUIRE_FALSE(r->HasError());
    distinct_ref = materialize(*r);
  }
  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();

  auto failures = run_workers(
    *env.db,
    workers(),
    iters_per_worker(),
    [&](int wid, int i) {
      if (wid == 0) { return distinct_sql; }
      return query_shapes()[(wid + i) % query_shapes().size()];
    },
    [&](int wid, int i) {
      if (wid == 0) { return distinct_ref; }
      return env.reference[(wid + i) % query_shapes().size()];
    });
  require_no_failures(failures);

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << stats.executions << " fallbacks=" << stats.fallbacks
                     << " runtime_fallbacks=" << stats.runtime_fallbacks
                     << " peak=" << env.sirius_ctx->query_lifecycle_peak());
  // The failing worker must actually have exercised the error-path teardown.
  REQUIRE(stats.runtime_fallbacks - stats_before.runtime_fallbacks >=
          static_cast<std::uint64_t>(iters_per_worker()));

  // B5 bookkeeping: every parked engine/plan must have been destroyed by its own
  // query's mandatory cleanup — a leftover means a cleanup never destroyed its plan
  // (it would then survive, holding GPU memory, until process teardown).
  REQUIRE(env.sirius_ctx->retired_query_plan_count() == 0);
}

TEST_CASE("concurrent queries serve from a pinned table across unpin/re-pin churn",
          "[concurrency][pin_table][isolated_context]")
{
  // Queries serve from GPU-pinned parquet entries while a churn thread
  // unpins and re-pins the same entry: William's shared_ptr pin map promises
  // an unpin mid-scan only drops the map slot — serving providers co-own the
  // data. Every result must stay correct throughout the churn.
  concurrent_env env;

  {
    duckdb::Connection con(*env.db);
    auto pin =
      con.Query("CALL pin_table('" + env.fact_path.string() + "', tier='gpu', name='fact_pin')");
    REQUIRE_FALSE(pin->HasError());
    auto pin2 =
      con.Query("CALL pin_table('" + env.dim_path.string() + "', tier='gpu', name='dim_pin')");
    REQUIRE_FALSE(pin2->HasError());
  }

  std::atomic<bool> churn_stop{false};
  std::atomic<int> churn_cycles{0};
  std::thread churn([&] {
    duckdb::Connection con(*env.db);
    while (!churn_stop.load()) {
      auto un = con.Query("CALL unpin_table('fact_pin')");
      if (un->HasError()) { break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      auto re =
        con.Query("CALL pin_table('" + env.fact_path.string() + "', tier='gpu', name='fact_pin')");
      if (re->HasError()) { break; }
      ++churn_cycles;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  auto failures = run_workers(
    *env.db,
    workers(),
    iters_per_worker() * 2,  // longer window so churn cycles land mid-stream
    [&](int wid, int i) { return query_shapes()[(wid + i) % query_shapes().size()]; },
    [&](int wid, int i) { return env.reference[(wid + i) % query_shapes().size()]; });
  churn_stop.store(true);
  churn.join();
  require_no_failures(failures);

  INFO("churn cycles completed: " << churn_cycles.load());
  REQUIRE(churn_cycles.load() > 0);

  duckdb::Connection con(*env.db);
  auto un1 = con.Query("CALL unpin_table('fact_pin')");
  auto un2 = con.Query("CALL unpin_table('dim_pin')");
  REQUIRE_FALSE(un1->HasError());
  REQUIRE_FALSE(un2->HasError());
}
