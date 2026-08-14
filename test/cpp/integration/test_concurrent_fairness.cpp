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

// Cross-query scheduling fairness (register issue F1).
//
// The scheduling priority packs the query id into its high bits, so without a
// fairness policy EVERY task of an earlier query outranks every task of a
// later one: a long query admitted first starves short queries admitted after
// it until it finishes, and under memory pressure that ordering can livelock
// (the early query waits on memory only a later query's completion could
// release, and the later query never dispatches).
//
// The fix rotates the queue pops round-robin across query bands. This test
// admits one heavy query FIRST beside short queries admitted after, under a
// deliberately small GPU pool, and asserts the short queries complete while
// the heavy one is still running. Before the fix the short queries' latencies
// all collapse onto the heavy query's completion time. The small pool plus the
// scoped_watchdog also make this a livelock probe: a wedge aborts with a
// signature instead of hanging CI.
//
// Catch2 v2 assertion macros are not thread-safe: workers collect failure
// strings and timestamps; the main thread REQUIREs.

#include "utils/concurrent_test_utils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace sirius::test::concurrent;

namespace {

using steady_clock = std::chrono::steady_clock;

double ms_between(steady_clock::time_point from, steady_clock::time_point to)
{
  return std::chrono::duration<double, std::milli>(to - from).count();
}

}  // namespace

TEST_CASE("fairness: short queries complete while a heavy query runs",
          "[concurrency][fairness][memory_pressure][isolated_context]")
{
  // Small absolute pool on purpose (shared GPUs + the livelock probe needs
  // memory pressure), a large fact table, and small scan splits so the heavy
  // query keeps a steady stream of dispatchable tasks queued for seconds —
  // exactly the condition under which the packed priorities starved everyone
  // else before the fix.
  env_options opt;
  opt.rows = env_i64("SIRIUS_TEST_FAIRNESS_ROWS", 100'000'000);
  opt.gpu_pool_bytes =
    static_cast<std::uint64_t>(env_i64("SIRIUS_TEST_FAIRNESS_POOL_BYTES", 1'000'000'000));
  opt.scan_task_batch_size = env_i64("SIRIUS_TEST_FAIRNESS_SCAN_BATCH", 1'000'000);

  // Shape 0 is the heavy query (join + wide group-by + sort over the whole
  // fact table); shape 1 is a short scalar aggregate over the tiny dim table.
  const std::vector<std::string> shapes = {
    "SELECT f.k, d.bucket, count(*) AS c, sum(f.v) AS sv, sum(f.id) AS si "
    "FROM fact f JOIN dim d ON f.k = d.k GROUP BY f.k, d.bucket ORDER BY f.k, d.bucket",
    "SELECT count(*) AS c, sum(w) AS s, min(k) AS lo, max(k) AS hi FROM dim WHERE k < 50",
  };
  adversarial_env env(opt, shapes);

  constexpr int kShortWorkers        = 2;
  constexpr int kItersPerShortWorker = 8;
  constexpr int kTotalShortIters     = kShortWorkers * kItersPerShortWorker;

  scoped_watchdog dog("fairness: heavy-first starvation", scenario_timeout(600));

  std::mutex failures_mutex;
  std::vector<std::string> failures;
  auto record_failure = [&](std::string msg) {
    std::lock_guard<std::mutex> lock(failures_mutex);
    failures.push_back(std::move(msg));
  };

  // --- Heavy query, admitted FIRST -----------------------------------------
  steady_clock::time_point heavy_start{};
  std::atomic<steady_clock::time_point::rep> heavy_end_ticks{0};
  std::thread heavy_thread([&] {
    duckdb::Connection con(*env.db);
    heavy_start = steady_clock::now();
    auto r      = con.Query(env.shapes[0]);
    heavy_end_ticks.store(steady_clock::now().time_since_epoch().count());
    if (r->HasError()) {
      record_failure("heavy query ERROR: " + r->GetError());
    } else if (auto got = materialize(*r); got != env.reference[0]) {
      record_failure("heavy query WRONG RESULT");
    }
  });

  // Wait until the heavy query holds an execution window (its query id is
  // minted at admission, so every short query below gets a HIGHER id — the
  // exact ordering that starved them before the fix).
  const auto admission_deadline = steady_clock::now() + std::chrono::seconds(30);
  while (!env.sirius_ctx->is_query_lifecycle_active() && steady_clock::now() < admission_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (!env.sirius_ctx->is_query_lifecycle_active()) {
    // Either the heavy query failed at admission or it finished implausibly
    // fast; surface its error (if any) instead of a bare timeout.
    heavy_thread.join();
    require_no_failures(failures);
    FAIL("heavy query never held an observable execution window");
  }
  // Give the heavy query a head start so its tasks are queued ahead.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // --- Short queries, admitted after ----------------------------------------
  std::mutex latency_mutex;
  std::vector<double> short_latencies_ms;
  std::vector<steady_clock::time_point> short_completions;

  std::vector<std::thread> short_threads;
  short_threads.reserve(kShortWorkers);
  for (int w = 0; w < kShortWorkers; ++w) {
    short_threads.emplace_back([&, w] {
      duckdb::Connection con(*env.db);
      for (int i = 0; i < kItersPerShortWorker; ++i) {
        const auto begin = steady_clock::now();
        auto r           = con.Query(env.shapes[1]);
        const auto end   = steady_clock::now();
        if (r->HasError()) {
          record_failure("short worker " + std::to_string(w) + " iter " + std::to_string(i) +
                         " ERROR: " + r->GetError());
          continue;
        }
        if (auto got = materialize(*r); got != env.reference[1]) {
          record_failure("short worker " + std::to_string(w) + " iter " + std::to_string(i) +
                         " WRONG RESULT");
          continue;
        }
        std::lock_guard<std::mutex> lock(latency_mutex);
        short_latencies_ms.push_back(ms_between(begin, end));
        short_completions.push_back(end);
      }
    });
  }

  for (auto& t : short_threads) {
    t.join();
  }
  heavy_thread.join();

  require_no_failures(failures);
  REQUIRE(short_latencies_ms.size() == static_cast<std::size_t>(kTotalShortIters));

  const auto heavy_end  = steady_clock::time_point(steady_clock::duration(heavy_end_ticks.load()));
  const double heavy_ms = ms_between(heavy_start, heavy_end);

  std::vector<double> sorted_latencies = short_latencies_ms;
  std::sort(sorted_latencies.begin(), sorted_latencies.end());
  const double median_short_ms = sorted_latencies[sorted_latencies.size() / 2];
  const double max_short_ms    = sorted_latencies.back();

  const auto completed_before_heavy = static_cast<int>(
    std::count_if(short_completions.begin(), short_completions.end(), [&](const auto& tp) {
      return tp < heavy_end;
    }));

  INFO("heavy_ms=" << heavy_ms << " median_short_ms=" << median_short_ms << " max_short_ms="
                   << max_short_ms << " completed_before_heavy=" << completed_before_heavy << "/"
                   << kTotalShortIters << " peak=" << env.sirius_ctx->query_lifecycle_peak());

  // Overlap must have genuinely happened, or this scenario proved nothing.
  REQUIRE(env.sirius_ctx->query_lifecycle_peak() > 1);

  if (heavy_ms < 1000.0) {
    // The heavy query finished too fast to starve anyone — the fairness
    // assertions below would be noise. Correctness still held; retune via
    // SIRIUS_TEST_FAIRNESS_ROWS / SIRIUS_TEST_FAIRNESS_POOL_BYTES if this
    // starts happening routinely.
    WARN("heavy query finished in " << heavy_ms << " ms; fairness window too small to judge");
    return;
  }

  // The fairness assertions, with room for scheduling noise:
  //  - most short iterations must complete while the heavy query is running
  //    (before the fix: ~0 of them do — they unblock only when it finishes);
  //  - the median short latency must be far below the heavy elapsed time
  //    (before the fix it collapses onto the heavy query's remaining runtime).
  REQUIRE(completed_before_heavy >= kTotalShortIters / 2);
  REQUIRE(median_short_ms * 2.0 < heavy_ms);
}
