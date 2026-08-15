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

// F6 (docs/concurrency/00-issue-register.md): operator sizing used to read
// whole-device / whole-host free memory —
//   - SORT_SAMPLE derived its partition count from get_available_memory(),
//     so two concurrent sorts each budgeted a fraction of the SAME free bytes
//     and overshot together;
//   - the result collector picked the host space by max_element over free
//     bytes and only then reserved, so two collectors could pick the same
//     space and the loser proceeded unreserved.
// Post-fix, SORT_SAMPLE divides the free-memory read by the live-query count
// and the collector reserves FIRST across candidate spaces. These scenarios
// pin the behavioral bar: N concurrent ORDER BY-heavy / wide-result queries
// must ALL complete with correct results on a small pool — slower is fine,
// crashing or wrong results is not.
//
// Result verification is intentionally not string-compare: a multi-million-row
// ORDER BY result is checked client-side for (a) exact row count, (b) exact
// column checksums, and (c) lexicographic (v, id) sortedness — strong enough
// to catch dropped/duplicated/unordered partitions without materializing
// hundred-MB reference strings per worker.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/concurrent_test_utils.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace sirius::test::concurrent;

struct sort_reference {
  std::int64_t rows   = 0;
  std::int64_t sum_v  = 0;
  std::int64_t sum_id = 0;
};

/// Client-side verification of one materialized `SELECT id, v ... ORDER BY v, id`
/// result: row count, column checksums, and (v, id) non-decreasing order.
/// Returns an empty string on success, else a failure description.
std::string verify_sorted_result(duckdb::QueryResult& result, const sort_reference& ref)
{
  std::int64_t rows       = 0;
  std::int64_t sum_v      = 0;
  std::int64_t sum_id     = 0;
  std::int64_t prev_v     = 0;
  std::int64_t prev_id    = 0;
  bool have_prev          = false;
  std::int64_t violations = 0;

  while (true) {
    auto chunk = result.Fetch();
    if (!chunk || chunk->size() == 0) { break; }
    for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
      auto const id = chunk->GetValue(0, i).GetValue<std::int64_t>();
      auto const v  = chunk->GetValue(1, i).GetValue<std::int64_t>();
      ++rows;
      sum_v += v;
      sum_id += id;
      if (have_prev && (v < prev_v || (v == prev_v && id < prev_id))) { ++violations; }
      prev_v    = v;
      prev_id   = id;
      have_prev = true;
    }
  }

  std::string failure;
  if (rows != ref.rows) {
    failure += "row count " + std::to_string(rows) + " != " + std::to_string(ref.rows) + "; ";
  }
  if (sum_v != ref.sum_v) {
    failure += "sum(v) " + std::to_string(sum_v) + " != " + std::to_string(ref.sum_v) + "; ";
  }
  if (sum_id != ref.sum_id) {
    failure += "sum(id) " + std::to_string(sum_id) + " != " + std::to_string(ref.sum_id) + "; ";
  }
  if (violations != 0) { failure += std::to_string(violations) + " (v, id) ordering violations; "; }
  return failure;
}

}  // namespace

TEST_CASE("sizing: concurrent ORDER BY-heavy queries all complete with ordered, complete results",
          "[concurrency][adversarial][memory_pressure][sizing][isolated_context]")
{
  scoped_watchdog dog("concurrent sorts", scenario_timeout(900));

  // Full-table sorts (ORDER BY -> SORT_SAMPLE/SORT_PARTITION/MERGE_SORT) big
  // enough that partition sizing matters against the pool, small enough that
  // several run concurrently. Pre-fix, every concurrent sort sized its
  // partitions from the same whole-device free bytes and they overshot
  // together under overlap.
  const std::int64_t rows = env_i64("SIRIUS_TEST_SIZING_ROWS", 8'000'000);
  env_options opt;
  opt.gpu_pool_bytes =
    static_cast<std::uint64_t>(env_i64("SIRIUS_TEST_SIZING_POOL_BYTES", 2'000'000'000));
  opt.rows = rows;
  // Reference scalars only — the sort results are verified client-side.
  adversarial_env env(opt, {"SELECT count(*) AS c, sum(v) AS sv, sum(id) AS si FROM fact"});

  const std::string sort_sql = "SELECT id, v FROM fact ORDER BY v, id";

  sort_reference ref;
  {
    duckdb::Connection con(*env.db);
    auto r = con.Query("SELECT count(*), sum(v), sum(id) FROM fact");
    REQUIRE_FALSE(r->HasError());
    auto chunk = r->Fetch();
    REQUIRE(chunk != nullptr);
    ref.rows   = chunk->GetValue(0, 0).GetValue<std::int64_t>();
    ref.sum_v  = chunk->GetValue(1, 0).GetValue<std::int64_t>();
    ref.sum_id = chunk->GetValue(2, 0).GetValue<std::int64_t>();
    REQUIRE(ref.rows == rows);

    // The sort must run on the GPU for the scenario to mean anything.
    auto probe = con.Query(sort_sql);
    REQUIRE_FALSE(probe->HasError());
    REQUIRE(verify_sorted_result(*probe, ref).empty());
  }

  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();

  const int n_workers = env_int("SIRIUS_TEST_SIZING_WORKERS", 4);
  const int n_iters   = env_int("SIRIUS_TEST_SIZING_ITERS", 2);

  std::mutex failures_mutex;
  std::vector<std::string> failures;
  std::atomic<int> ready{0};
  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool go = false;

  auto worker = [&](int wid) {
    duckdb::Connection con(*env.db);
    {
      std::unique_lock<std::mutex> lock(start_mutex);
      ++ready;
      start_cv.wait(lock, [&] { return go; });
    }
    for (int i = 0; i < n_iters; ++i) {
      auto r = con.Query(sort_sql);
      if (r->HasError()) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("worker " + std::to_string(wid) + " iter " + std::to_string(i) +
                           " ERROR: " + r->GetError());
        continue;
      }
      if (auto failure = verify_sorted_result(*r, ref); !failure.empty()) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("worker " + std::to_string(wid) + " iter " + std::to_string(i) +
                           " WRONG RESULT: " + failure);
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
  require_no_failures(failures);

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << (stats.executions - stats_before.executions) << " runtime_fallbacks="
                     << (stats.runtime_fallbacks - stats_before.runtime_fallbacks)
                     << " peak=" << env.sirius_ctx->query_lifecycle_peak());
  // Overlap actually happened (the whole point) ...
  if (slots() > 1 && n_workers > 1) { REQUIRE(env.sirius_ctx->query_lifecycle_peak() > 1); }
  // ... and every sort ran on the GPU: a concurrent-overshoot OOM completing
  // via the CPU fallback would hide exactly the regression this scenario pins.
  REQUIRE(stats.executions - stats_before.executions >=
          static_cast<std::uint64_t>(n_workers) * n_iters);
  REQUIRE(stats.runtime_fallbacks == stats_before.runtime_fallbacks);
}

TEST_CASE("sizing: concurrent wide-result queries reserve host space and all complete",
          "[concurrency][adversarial][memory_pressure][sizing][isolated_context]")
{
  scoped_watchdog dog("concurrent collectors", scenario_timeout(900));

  // Every worker materializes a wide result at the same time, so every
  // result collector wants host space for its GPU->HOST clones in the same
  // window. Pre-fix, the collectors chose the space by a shared free-bytes
  // snapshot and reserved after: contenders picked the same space and the
  // losers fell to the unreserved path (WARN "proceeding without
  // reservation", converter allowed to OOM). Post-fix reservation comes
  // first, so at this sizing (well within host capacity) every clone must be
  // reserved — the WARN doubles as the regression signal.
  const std::int64_t rows = env_i64("SIRIUS_TEST_SIZING_ROWS", 8'000'000);
  env_options opt;
  opt.rows = rows;
  adversarial_env env(opt, {"SELECT count(*) AS c FROM fact WHERE k % 3 = 1"});

  // ~1/3 of the fact table, two BIGINT columns: a couple hundred MB of host
  // result per query at the default 8 GB host capacity.
  const std::string wide_sql = "SELECT id, v FROM fact WHERE k % 3 = 1";
  std::int64_t expected_rows = 0;
  {
    duckdb::Connection con(*env.db);
    auto r = con.Query("SELECT count(*) FROM fact WHERE k % 3 = 1");
    REQUIRE_FALSE(r->HasError());
    auto chunk = r->Fetch();
    REQUIRE(chunk != nullptr);
    expected_rows = chunk->GetValue(0, 0).GetValue<std::int64_t>();
    REQUIRE(expected_rows > 0);
  }

  // Captures WARN+ lines (the unreserved-fallback signal) during the storm.
  scoped_downgrade_log_counter log_capture;

  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();
  const int n_workers     = env_int("SIRIUS_TEST_SIZING_WORKERS", 4);
  const int n_iters       = env_int("SIRIUS_TEST_SIZING_ITERS", 2);

  std::mutex failures_mutex;
  std::vector<std::string> failures;
  std::atomic<int> ready{0};
  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool go = false;

  auto worker = [&](int wid) {
    duckdb::Connection con(*env.db);
    {
      std::unique_lock<std::mutex> lock(start_mutex);
      ++ready;
      start_cv.wait(lock, [&] { return go; });
    }
    for (int i = 0; i < n_iters; ++i) {
      auto r = con.Query(wide_sql);
      if (r->HasError()) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("worker " + std::to_string(wid) + " iter " + std::to_string(i) +
                           " ERROR: " + r->GetError());
        continue;
      }
      std::int64_t rows_seen = 0;
      while (true) {
        auto chunk = r->Fetch();
        if (!chunk || chunk->size() == 0) { break; }
        rows_seen += static_cast<std::int64_t>(chunk->size());
      }
      if (rows_seen != expected_rows) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("worker " + std::to_string(wid) + " iter " + std::to_string(i) +
                           " WRONG RESULT: " + std::to_string(rows_seen) +
                           " rows != " + std::to_string(expected_rows));
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
  require_no_failures(failures);

  // Nobody fell to the unreserved path: with reserve-first ordering and this
  // sizing every collector clone must have obtained a host reservation.
  for (const auto& w : log_capture.captured_warnings()) {
    if (w.find("proceeding without reservation") != std::string::npos) {
      UNSCOPED_INFO("unreserved collector fallback fired: " << w);
      REQUIRE(w.find("proceeding without reservation") == std::string::npos);
    }
  }

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << (stats.executions - stats_before.executions) << " runtime_fallbacks="
                     << (stats.runtime_fallbacks - stats_before.runtime_fallbacks)
                     << " peak=" << env.sirius_ctx->query_lifecycle_peak());
  if (slots() > 1 && n_workers > 1) { REQUIRE(env.sirius_ctx->query_lifecycle_peak() > 1); }
  REQUIRE(stats.executions - stats_before.executions >=
          static_cast<std::uint64_t>(n_workers) * n_iters);
  REQUIRE(stats.runtime_fallbacks == stats_before.runtime_fallbacks);
}
