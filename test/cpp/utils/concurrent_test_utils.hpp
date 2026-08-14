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

// Shared harness for the adversarial concurrency suite
// (test_concurrent_adversarial.cpp, test_concurrent_config_races.cpp).
//
// Generalizes the helpers of test_concurrent_queries.cpp:
//   - adversarial_env: per-test DuckDB + SiriusContext with an absolute (tiny)
//     GPU pool, bounded host pool, per-PID temp files (several test binaries
//     from sibling worktrees may run concurrently on this box), arbitrary
//     query-shape list with single-threaded reference results.
//   - run_workers: barrier-synced worker threads with PER-WORKER iteration
//     counts (the spill-storm scenario needs short queries ending constantly
//     while long ones run).
//   - scoped_watchdog: aborts the process with a diagnostic when a scenario
//     wedges — a reproducible abort signature beats an eternal hang.
//   - scoped_downgrade_log_counter: a log-sink shim that counts completed
//     downgrade requests / batches ("[downgrade] ... done: N batches") and the
//     "queue inactive, dropping request" drops, so tests can ASSERT that
//     spills actually fired instead of hoping the pool was small enough.
//
// Catch2 v2 assertion macros are not thread-safe: workers only collect
// failure strings; the main thread REQUIREs.

#pragma once

#include "log/logging.hpp"
#include "log/sink.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/client_context.hpp>
#include <sirius_context.hpp>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace sirius::test::concurrent {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Env knobs (same names as test_concurrent_queries.cpp where they overlap)
// ---------------------------------------------------------------------------
inline int env_int(const char* name, int fallback)
{
  if (const char* v = std::getenv(name); v != nullptr) { return std::atoi(v); }
  return fallback;
}

inline std::int64_t env_i64(const char* name, std::int64_t fallback)
{
  if (const char* v = std::getenv(name); v != nullptr) { return std::atoll(v); }
  return fallback;
}

inline int slots() { return env_int("SIRIUS_TEST_CONCURRENCY_SLOTS", 4); }
inline int workers() { return env_int("SIRIUS_TEST_CONCURRENCY_WORKERS", 4); }
inline int iters_per_worker() { return env_int("SIRIUS_TEST_CONCURRENCY_ITERS", 6); }
inline int pipeline_threads() { return env_int("SIRIUS_TEST_PIPELINE_THREADS", 4); }
inline int creator_threads() { return env_int("SIRIUS_TEST_CREATOR_THREADS", 2); }

/// Per-scenario watchdog deadline. SIRIUS_TEST_ADVERSARIAL_TIMEOUT_S overrides.
inline std::chrono::seconds scenario_timeout(int fallback_seconds)
{
  return std::chrono::seconds(env_int("SIRIUS_TEST_ADVERSARIAL_TIMEOUT_S", fallback_seconds));
}

// ---------------------------------------------------------------------------
// Watchdog: a hang IS a finding — turn it into an abort with a signature.
// ---------------------------------------------------------------------------
class scoped_watchdog {
 public:
  scoped_watchdog(std::string label, std::chrono::seconds timeout) : label_(std::move(label))
  {
    thread_ = std::thread([this, timeout] {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!cv_.wait_for(lock, timeout, [&] { return disarmed_; })) {
        std::fprintf(stderr,
                     "\n[adversarial-watchdog] scenario '%s' still running after %lld s — "
                     "concurrency hang (register groups A/C are the usual suspects). "
                     "Aborting so the run yields a stack instead of silence.\n",
                     label_.c_str(),
                     static_cast<long long>(timeout.count()));
        std::fflush(stderr);
        std::abort();
      }
    });
  }

  ~scoped_watchdog()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      disarmed_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }

  scoped_watchdog(const scoped_watchdog&)            = delete;
  scoped_watchdog& operator=(const scoped_watchdog&) = delete;

 private:
  std::string label_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool disarmed_ = false;
  std::thread thread_;
};

// ---------------------------------------------------------------------------
// Downgrade-observing log sink.
//
// downgrade_executor::processing_loop logs one DEBUG line per completed
// request: "[downgrade] [<label>] request [monitor ]done: <N> batches, ...".
// The only test-visible downgrade counter today is monitor_requests_issued
// (requests ISSUED, not work done), so this shim is how a test proves spills
// actually moved batches. should_log admits everything so the DEBUG line is
// formatted even when the configured sink filters it; forwarding still honors
// the downstream level.
// ---------------------------------------------------------------------------
struct downgrade_log_totals {
  std::size_t requests_done         = 0;  ///< completed downgrade requests (incl. 0-batch ones)
  std::size_t requests_with_batches = 0;  ///< completed requests that moved >= 1 batch
  std::size_t batches_downgraded    = 0;  ///< total batches moved off the source tier
  std::size_t inactive_drops        = 0;  ///< "request_downgrade: queue inactive" drops
};

class downgrade_counting_sink final : public sirius::log::sink {
 public:
  explicit downgrade_counting_sink(std::shared_ptr<sirius::log::sink> downstream)
    : downstream_(std::move(downstream))
  {
  }

  void set_level(sirius::log::level level) override { downstream_->set_level(level); }

  bool should_log(sirius::log::level) const override { return true; }

  void log(sirius::log::level level,
           std::source_location const& location,
           std::string_view message) override
  {
    if (message.find("[downgrade]") != std::string_view::npos) {
      if (auto pos = message.find(" done: "); pos != std::string_view::npos) {
        requests_done_.fetch_add(1, std::memory_order_relaxed);
        std::size_t batches = 0;
        for (auto rest = message.substr(pos + 7);
             !rest.empty() && rest.front() >= '0' && rest.front() <= '9';
             rest = rest.substr(1)) {
          batches = batches * 10 + static_cast<std::size_t>(rest.front() - '0');
        }
        batches_downgraded_.fetch_add(batches, std::memory_order_relaxed);
        if (batches > 0) { requests_with_batches_.fetch_add(1, std::memory_order_relaxed); }
      }
      if (message.find("queue inactive, dropping request") != std::string_view::npos) {
        inactive_drops_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    // Keep WARN+ lines for the scenario report (capped: storms are chatty).
    if (level == sirius::log::level::warn || level == sirius::log::level::error ||
        level == sirius::log::level::critical) {
      std::lock_guard<std::mutex> lock(captured_mutex_);
      if (captured_.size() < 64) { captured_.emplace_back(message); }
    }
    if (downstream_->should_log(level)) { downstream_->log(level, location, message); }
  }

  bool flush() override { return downstream_->flush(); }

  downgrade_log_totals totals() const
  {
    downgrade_log_totals t;
    t.requests_done         = requests_done_.load(std::memory_order_relaxed);
    t.requests_with_batches = requests_with_batches_.load(std::memory_order_relaxed);
    t.batches_downgraded    = batches_downgraded_.load(std::memory_order_relaxed);
    t.inactive_drops        = inactive_drops_.load(std::memory_order_relaxed);
    return t;
  }

  std::vector<std::string> captured_warnings() const
  {
    std::lock_guard<std::mutex> lock(captured_mutex_);
    return captured_;
  }

 private:
  std::shared_ptr<sirius::log::sink> downstream_;
  std::atomic<std::size_t> requests_done_{0};
  std::atomic<std::size_t> requests_with_batches_{0};
  std::atomic<std::size_t> batches_downgraded_{0};
  std::atomic<std::size_t> inactive_drops_{0};
  mutable std::mutex captured_mutex_;
  std::vector<std::string> captured_;
};

class scoped_downgrade_log_counter {
 public:
  scoped_downgrade_log_counter()
    : downstream_(sirius::log::get_sink()),
      sink_(std::make_shared<downgrade_counting_sink>(downstream_))
  {
    try {
      sirius::log::set_sink(sink_);
    } catch (...) {
      try {
        sirius::log::set_sink(downstream_);
      } catch (...) {
      }
      throw;
    }
  }

  ~scoped_downgrade_log_counter() noexcept
  {
    try {
      if (sirius::log::get_sink() == sink_) { sirius::log::set_sink(downstream_); }
    } catch (...) {
    }
  }

  scoped_downgrade_log_counter(const scoped_downgrade_log_counter&)            = delete;
  scoped_downgrade_log_counter& operator=(const scoped_downgrade_log_counter&) = delete;

  downgrade_log_totals totals() const { return sink_->totals(); }
  std::vector<std::string> captured_warnings() const { return sink_->captured_warnings(); }

 private:
  std::shared_ptr<sirius::log::sink> downstream_;
  std::shared_ptr<downgrade_counting_sink> sink_;
};

// ---------------------------------------------------------------------------
// Config + environment
// ---------------------------------------------------------------------------
struct env_options {
  // Tiny absolute pools by default: this suite runs on a shared box next to
  // sibling agents' test binaries. Do NOT switch these to card fractions.
  std::uint64_t gpu_pool_bytes      = 1'500'000'000;
  std::uint64_t host_capacity_bytes = 8'000'000'000;
  std::int64_t rows                 = 2'000'000;
  std::int64_t dim_rows             = 97;
  int num_gpus                      = 1;
  int max_concurrent_queries        = slots();
  std::string monitor_period        = "10ms";
  // One split per 100M rows = effectively one scan task per file. Scenarios
  // that need a long-running query with a steady stream of dispatchable tasks
  // (the fairness suite) lower this so the scan fans out.
  std::int64_t scan_task_batch_size = 100'000'000;
};

inline std::string adversarial_config_yaml(const env_options& opt)
{
  return R"(sirius:
  topology:
    num_gpus: )" +
         std::to_string(opt.num_gpus) + R"(
  memory:
    gpu:
      usage_limit_bytes: )" +
         std::to_string(opt.gpu_pool_bytes) + R"(
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: )" +
         std::to_string(opt.host_capacity_bytes) + R"(
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
      monitor_period: )" +
         opt.monitor_period + R"(
    scan_manager:
      max_concurrent_queries: )" +
         std::to_string(opt.max_concurrent_queries) + R"(
  operator_params:
    scan_task_batch_size: )" +
         std::to_string(opt.scan_task_batch_size) + R"(
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
    // DuckDB instances in other tests skip SiriusContext creation. These tests
    // WANT their own SiriusContext, so lift the kill switch for their window.
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

  scoped_config_env(const scoped_config_env&)            = delete;
  scoped_config_env& operator=(const scoped_config_env&) = delete;

 private:
  std::string original_;
  bool had_original_ = false;
  bool had_disable_  = false;
};

inline std::string materialize(duckdb::QueryResult& result) { return result.ToString(); }

/// The bring-up shapes from test_concurrent_queries.cpp (every query carries
/// an ORDER BY or is a scalar aggregate so results compare by string equality).
inline const std::vector<std::string>& default_shapes()
{
  static const std::vector<std::string> queries = {
    "SELECT k, count(*) AS c, sum(v) AS s FROM fact GROUP BY k ORDER BY k",
    "SELECT d.bucket, count(*) AS c FROM fact f JOIN dim d ON f.k = d.k "
    "WHERE d.w > 250 GROUP BY d.bucket ORDER BY d.bucket",
    "SELECT count(*) AS c, sum(v) AS s, min(id) AS lo, max(id) AS hi FROM fact WHERE k < 13",
    "SELECT k % 10 AS kb, v % 7 AS vb, count(*) AS c, sum(id) AS s FROM fact "
    "GROUP BY kb, vb ORDER BY kb, vb",
  };
  return queries;
}

/// Per-test DuckDB + SiriusContext + parquet-backed `fact`/`dim` views +
/// single-threaded reference results for @p shapes. Temp files are per-PID so
/// sibling worktrees' binaries can run this suite concurrently.
struct adversarial_env {
  env_options options;
  std::vector<std::string> shapes;
  fs::path config_path;
  std::unique_ptr<scoped_config_env> env_guard;
  std::unique_ptr<duckdb::DuckDB> db;
  fs::path fact_path;
  fs::path dim_path;
  std::vector<std::string> reference;
  duckdb::shared_ptr<duckdb::SiriusContext> sirius_ctx;

  explicit adversarial_env(env_options opt                        = {},
                           const std::vector<std::string>& shapes = default_shapes())
    : options(opt), shapes(shapes)
  {
    const std::string pid = std::to_string(::getpid());
    config_path           = fs::temp_directory_path() / ("sirius_adversarial_" + pid + ".yaml");
    {
      std::ofstream out(config_path);
      out << adversarial_config_yaml(options);
    }
    env_guard = std::make_unique<scoped_config_env>(config_path);
    db        = std::make_unique<duckdb::DuckDB>(nullptr);

    // Seed data as PARQUET behind views: the GPU scan serves parquet sources;
    // an in-memory CREATE TABLE would plan-time-fall-back every query to CPU
    // and reduce these tests to concurrent DuckDB.
    fact_path = fs::temp_directory_path() / ("sirius_adversarial_fact_" + pid + ".parquet");
    dim_path  = fs::temp_directory_path() / ("sirius_adversarial_dim_" + pid + ".parquet");
    duckdb::Connection con(*db);
    auto r1 =
      con.Query("COPY (SELECT range AS id, range % " + std::to_string(options.dim_rows) +
                " AS k, (range * 13) % 1000 AS v FROM range(" + std::to_string(options.rows) +
                ")) TO '" + fact_path.string() + "' (FORMAT parquet)");
    REQUIRE_FALSE(r1->HasError());
    auto r2 = con.Query(
      "COPY (SELECT range AS k, (range * 7) % 500 AS w, range % 5 AS bucket "
      "FROM range(" +
      std::to_string(options.dim_rows) + ")) TO '" + dim_path.string() + "' (FORMAT parquet)");
    REQUIRE_FALSE(r2->HasError());
    auto r3 =
      con.Query("CREATE VIEW fact AS SELECT * FROM read_parquet('" + fact_path.string() + "')");
    REQUIRE_FALSE(r3->HasError());
    auto r4 =
      con.Query("CREATE VIEW dim AS SELECT * FROM read_parquet('" + dim_path.string() + "')");
    REQUIRE_FALSE(r4->HasError());

    // Single-threaded reference pass (still transparently GPU-executed; these
    // tests prove overlap-correctness, not GPU-vs-CPU — the rest of the suite
    // covers that).
    for (const auto& q : this->shapes) {
      reference.push_back(reference_for(con, q));
    }
    sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
  }

  ~adversarial_env()
  {
    // The DB (and with it the SiriusContext) must go before the config guard.
    db.reset();
    std::error_code ec;
    fs::remove(fact_path, ec);
    fs::remove(dim_path, ec);
    fs::remove(config_path, ec);
  }

  adversarial_env(const adversarial_env&)            = delete;
  adversarial_env& operator=(const adversarial_env&) = delete;

  static std::string reference_for(duckdb::Connection& con, const std::string& sql)
  {
    auto r = con.Query(sql);
    REQUIRE_FALSE(r->HasError());
    return materialize(*r);
  }

  std::string reference_for(const std::string& sql)
  {
    duckdb::Connection con(*db);
    return reference_for(con, sql);
  }
};

// ---------------------------------------------------------------------------
// Workers
// ---------------------------------------------------------------------------

/// Run @p n_workers threads (own connection each, barrier-synced start).
/// Worker @p wid executes @p iters_for(wid) iterations of
/// @p per_worker(wid, i) -> SQL, comparing against @p expected(wid, i)
/// (empty string = only require no error). Returns failure descriptions.
template <typename SqlFn, typename ExpectFn>
std::vector<std::string> run_workers(duckdb::DuckDB& db,
                                     int n_workers,
                                     const std::function<int(int)>& iters_for,
                                     SqlFn per_worker,
                                     ExpectFn expected)
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
    const int n_iters = iters_for(wid);
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

/// Uniform-iteration overload (matches the original helper's shape).
template <typename SqlFn, typename ExpectFn>
std::vector<std::string> run_workers(
  duckdb::DuckDB& db, int n_workers, int n_iters, SqlFn per_worker, ExpectFn expected)
{
  return run_workers(db, n_workers, [n_iters](int) { return n_iters; }, per_worker, expected);
}

inline void require_no_failures(const std::vector<std::string>& failures)
{
  for (const auto& f : failures) {
    UNSCOPED_INFO(f);
  }
  REQUIRE(failures.empty());
}

/// Sum of monitor_requests_issued_for_testing() across all downgrade executors.
inline std::size_t total_monitor_requests(const duckdb::SiriusContext& ctx)
{
  std::size_t total = 0;
  for (const auto& exec : ctx.get_downgrade_executors()) {
    if (exec) { total += exec->monitor_requests_issued_for_testing(); }
  }
  return total;
}

}  // namespace sirius::test::concurrent
