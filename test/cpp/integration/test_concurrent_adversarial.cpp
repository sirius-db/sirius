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

// Adversarial concurrency scenarios: workload-shaped attacks on the
// concurrent-query bring-up (docs/concurrency/00-issue-register.md,
// 01-bringup-triage.md). Unlike the bring-up tests in
// test_concurrent_queries.cpp these deliberately aim at the register's
// MUST-FIX cluster:
//
//   1. Spill storm + query-end churn — A7 (query-end drain cancels peers'
//      downgrade promises), B1 (TIER-2 task resurrection), B2 (drain restarts
//      the processing thread before quiescence is consumed), D6 (drain racing
//      the monitor latches _monitor_request_enqueued -> spilling dies).
//   2. Mixed failure storm — per-query failure containment while failures and
//      successes interleave on every worker under memory pressure.
//   3. Intensified pin churn — F5 (spurious PIN rejection from the
//      all-entries use_count snapshot) plus the shared_ptr pin machinery.
//   4. 2-GPU concurrency — the per-query GPU admission fix (task_creator
//      admission moved into query_task_global_state).
//
// A failing assertion here is a deliverable: scenarios that fail on current
// code are tagged [!mayfail] with the failure signature documented at the
// assertion site, so the suite stays green while the repro ships.

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/concurrent_test_utils.hpp>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace sirius::test::concurrent;

}  // namespace

TEST_CASE("adversarial: spill storm with constant query-end churn",
          "[concurrency][adversarial][memory_pressure][isolated_context]")
{
  // Long memory-hungry queries force downgrades on a tiny (1.5 GB) pool while
  // short queries END constantly — every query end runs run_mandatory_cleanup
  // -> downgrade_executor::drain(query_id) on the shared executors, i.e. the
  // exact A7/B1/B2/D6 window. After the churn a memory-heavy query with CPU
  // fallback DISABLED must still succeed on the GPU: if a cancelled request
  // racing the monitor latched _monitor_request_enqueued (D6), automatic
  // spilling for the space is dead and that query dies in OOM-retry instead.
  scoped_watchdog dog("spill storm", scenario_timeout(900));

  const std::int64_t rows               = env_i64("SIRIUS_TEST_SPILL_ROWS", 8'000'000);
  const std::vector<std::string> shapes = {
    // [0] heavy: fact-fact self join — the build side alone dwarfs the pool.
    "SELECT count(*) AS c, sum(a.v + b.v) AS s FROM fact a JOIN fact b ON a.id = b.id",
    // [1] heavy: high-cardinality aggregate (4M groups) + top-n.
    "SELECT g, c, s FROM (SELECT id % 4000000 AS g, count(*) AS c, sum(v) AS s FROM fact "
    "GROUP BY g) ORDER BY c DESC, g LIMIT 10",
    // [2] quick: dim-only aggregate — a full GPU window that ends in milliseconds.
    "SELECT count(*) AS c, sum(w) AS s FROM dim WHERE w > 100",
    // [3] quick: selective fact filter.
    "SELECT count(*) AS c, min(id) AS lo, max(id) AS hi FROM fact WHERE k = 7",
  };

  env_options opt;
  opt.gpu_pool_bytes =
    static_cast<std::uint64_t>(env_i64("SIRIUS_TEST_SPILL_POOL_BYTES", 1'500'000'000));
  opt.rows = rows;
  adversarial_env env(opt, shapes);

  // Installed AFTER the env: extension load reinstalls the configured global
  // sink (install_configured_log_sink in LoadInternal), which would evict a
  // shim installed earlier. The reference pass's downgrades are not counted;
  // the assertions below use deltas across the storm only.
  scoped_downgrade_log_counter downgrade_log;

  const auto stats_before     = env.sirius_ctx->get_transparent_execution_stats();
  const auto downgrade_before = downgrade_log.totals();
  const auto monitor_before   = total_monitor_requests(*env.sirius_ctx);

  constexpr int kHeavyWorkers = 2;
  constexpr int kQuickWorkers = 3;
  const int heavy_iters       = env_int("SIRIUS_TEST_SPILL_HEAVY_ITERS", 3);
  const int quick_iters       = env_int("SIRIUS_TEST_SPILL_QUICK_ITERS", 24);

  auto shape_of = [&](int wid, int i) {
    if (wid < kHeavyWorkers) { return static_cast<std::size_t>((wid + i) % 2); }
    return static_cast<std::size_t>(2 + (wid + i) % 2);
  };
  auto failures = run_workers(
    *env.db,
    kHeavyWorkers + kQuickWorkers,
    [&](int wid) { return wid < kHeavyWorkers ? heavy_iters : quick_iters; },
    [&](int wid, int i) { return shapes[shape_of(wid, i)]; },
    [&](int wid, int i) { return env.reference[shape_of(wid, i)]; });
  require_no_failures(failures);

  const auto stats           = env.sirius_ctx->get_transparent_execution_stats();
  const auto downgrade_after = downgrade_log.totals();
  const auto monitor_after   = total_monitor_requests(*env.sirius_ctx);
  for (const auto& w : downgrade_log.captured_warnings()) {
    UNSCOPED_INFO("captured WARN/ERROR: " << w);
  }
  INFO("executions=" << (stats.executions - stats_before.executions) << " runtime_fallbacks="
                     << (stats.runtime_fallbacks - stats_before.runtime_fallbacks)
                     << " peak=" << env.sirius_ctx->query_lifecycle_peak());
  INFO("downgrade requests done="
       << (downgrade_after.requests_done - downgrade_before.requests_done) << " with_batches="
       << (downgrade_after.requests_with_batches - downgrade_before.requests_with_batches)
       << " batches=" << (downgrade_after.batches_downgraded - downgrade_before.batches_downgraded)
       << " inactive_drops=" << (downgrade_after.inactive_drops - downgrade_before.inactive_drops)
       << " monitor_requests_issued=" << (monitor_after - monitor_before));

  // The scenario is only meaningful if spills actually fired during the storm.
  // If this fails, the pool/row sizing no longer produces pressure — raise
  // SIRIUS_TEST_SPILL_ROWS or shrink SIRIUS_TEST_SPILL_POOL_BYTES.
  REQUIRE(downgrade_after.batches_downgraded > downgrade_before.batches_downgraded);

  // Every iteration attempted the GPU (runtime fallbacks are tolerated during
  // the storm — an OOM-retry cap trip completes correctly via CPU — but are
  // reported above).
  const auto total_iters = static_cast<std::uint64_t>(kHeavyWorkers) * heavy_iters +
                           static_cast<std::uint64_t>(kQuickWorkers) * quick_iters;
  REQUIRE(stats.executions - stats_before.executions >= total_iters);
  if (slots() > 1) { REQUIRE(env.sirius_ctx->query_lifecycle_peak() > 1); }

  // D6 latch probe: after all that drain-vs-monitor churn, a memory-heavy
  // query must still be able to spill its way to completion ON THE GPU.
  {
    duckdb::Connection con(*env.db);
    auto set = con.Query("SET enable_duckdb_fallback = false");
    REQUIRE_FALSE(set->HasError());
    const auto monitor_pre_probe  = total_monitor_requests(*env.sirius_ctx);
    auto probe                    = con.Query(shapes[0]);
    const auto monitor_post_probe = total_monitor_requests(*env.sirius_ctx);
    INFO("post-churn probe monitor_requests_issued delta = "
         << (monitor_post_probe - monitor_pre_probe));
    if (probe->HasError()) { UNSCOPED_INFO("post-churn probe ERROR: " << probe->GetError()); }
    REQUIRE_FALSE(probe->HasError());
    REQUIRE(materialize(*probe) == env.reference[0]);
  }
}

TEST_CASE("adversarial: mixed failure storm keeps healthy executions flowing",
          "[concurrency][adversarial][isolated_context]")
{
  // Every worker ALTERNATES between a runtime-failing shape (DISTINCT
  // aggregates throw mid-GPU-execution and complete via CPU fallback) and
  // healthy shapes, under a small pool. This differs from the bring-up error
  // test (one dedicated failing worker): here failure cleanup
  // (terminate_query -> drain_after_error -> run_mandatory_cleanup) fires on
  // EVERY connection interleaved with its own healthy queries, so a
  // containment bug poisons the very next iteration of the same worker.
  //
  // REGRESSION HISTORY: before the _manager_lifecycle_mutex fix
  // (fix(concurrency): serialize task_executor manager-thread lifecycle
  // transitions) this scenario reliably ABORTED the process (2/2):
  // std::terminate in itask_executor::resume_manager() <-
  // task_scheduler::drain_after_error. With >= 2 queries failing
  // concurrently, two unserialized wait_and_drain_query() brackets
  // interleaved on the same executor and the second resume_manager()
  // assigned onto a joinable _manager_thread. Unlisted A6-family; same class
  // as the downgrade_executor drain race fixed by _lifecycle_mutex. This
  // test is the live regression guard for that fix.
  scoped_watchdog dog("mixed failure storm", scenario_timeout(900));

  env_options opt;
  opt.gpu_pool_bytes = 2'000'000'000;
  opt.rows           = env_i64("SIRIUS_TEST_FAILURE_STORM_ROWS", 6'000'000);
  adversarial_env env(opt);

  const std::string distinct_sql =
    "SELECT count(DISTINCT k) AS dk, count(DISTINCT v % 101) AS dv FROM fact ORDER BY 1";
  const std::string distinct_ref = env.reference_for(distinct_sql);

  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();

  const int n_workers  = workers();
  const int n_iters    = env_int("SIRIUS_TEST_FAILURE_STORM_ITERS", 9);
  auto is_failing_iter = [](int wid, int i) { return (wid + i) % 3 == 0; };

  auto failures = run_workers(
    *env.db,
    n_workers,
    n_iters,
    [&](int wid, int i) {
      if (is_failing_iter(wid, i)) { return distinct_sql; }
      return env.shapes[(wid + i) % env.shapes.size()];
    },
    [&](int wid, int i) {
      if (is_failing_iter(wid, i)) { return distinct_ref; }
      return env.reference[(wid + i) % env.shapes.size()];
    });
  require_no_failures(failures);

  std::uint64_t failing_iters = 0;
  for (int w = 0; w < n_workers; ++w) {
    for (int i = 0; i < n_iters; ++i) {
      if (is_failing_iter(w, i)) { ++failing_iters; }
    }
  }
  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << (stats.executions - stats_before.executions) << " runtime_fallbacks="
                     << (stats.runtime_fallbacks - stats_before.runtime_fallbacks)
                     << " failing_iters=" << failing_iters
                     << " peak=" << env.sirius_ctx->query_lifecycle_peak());

  // Every failing iteration must have attempted the GPU and fallen back at
  // runtime; every iteration (healthy or not) must have reached execution —
  // i.e. no worker got wedged or silently plan-time-fallback'd by a peer's
  // failure cleanup.
  REQUIRE(stats.runtime_fallbacks - stats_before.runtime_fallbacks >= failing_iters);
  REQUIRE(stats.executions - stats_before.executions >=
          static_cast<std::uint64_t>(n_workers) * n_iters);
  if (slots() > 1 && n_workers > 1) { REQUIRE(env.sirius_ctx->query_lifecycle_peak() > 1); }
}

TEST_CASE("adversarial: rapid pin/unpin churn under concurrent scans",
          "[concurrency][adversarial][pin_table][isolated_context]")
{
  // The bring-up pin test churns at 50 ms and BREAKS on the first pin error.
  // This one churns at ~10 ms for the whole run, never breaks, and COUNTS:
  // F5 predicts spurious re-pin rejections ("a query is currently reading the
  // pinned entry") because any query inside try_match_cached_entry holds a
  // snapshot reference to EVERY pinned entry. Spurious rejections are
  // documented (rate reported below); wrong results and unexpected pin errors
  // are failures.
  //
  // Observed on integration/concurrency-full (2026-08-14): 0 spurious
  // rejections in 15 churn ops beside 96 executions — F5's predicted
  // over-rejection did not surface at this cadence (a pin lands in ~1-2 s
  // gaps between snapshot holds). Keep counting; a nonzero rate is worth
  // knowing about, not failing on.
  scoped_watchdog dog("pin churn", scenario_timeout(600));

  adversarial_env env;

  {
    duckdb::Connection con(*env.db);
    auto pin =
      con.Query("CALL pin_table('" + env.fact_path.string() + "', tier='gpu', name='fact_pin')");
    REQUIRE_FALSE(pin->HasError());
    auto pin2 =
      con.Query("CALL pin_table('" + env.dim_path.string() + "', tier='gpu', name='dim_pin')");
    REQUIRE_FALSE(pin2->HasError());
  }

  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();

  std::atomic<bool> churn_stop{false};
  std::atomic<int> unpin_ok{0};
  std::atomic<int> unpin_fail{0};
  std::atomic<int> pin_ok{0};
  std::atomic<int> pin_busy{0};   // F5 signature: "currently reading the pinned entry"
  std::atomic<int> pin_other{0};  // anything else is a real bug
  std::mutex sample_mutex;
  std::vector<std::string> error_samples;
  auto record_sample = [&](const std::string& what, const std::string& err) {
    std::lock_guard<std::mutex> lock(sample_mutex);
    if (error_samples.size() < 10) { error_samples.push_back(what + ": " + err); }
  };

  std::thread churn([&] {
    duckdb::Connection con(*env.db);
    bool pinned = true;
    while (!churn_stop.load()) {
      if (pinned) {
        auto un = con.Query("CALL unpin_table('fact_pin')");
        if (un->HasError()) {
          ++unpin_fail;
          record_sample("unpin", un->GetError());
        } else {
          ++unpin_ok;
          pinned = false;
        }
      } else {
        auto re = con.Query("CALL pin_table('" + env.fact_path.string() +
                            "', tier='gpu', name='fact_pin')");
        if (re->HasError()) {
          const auto err = re->GetError();
          if (err.find("currently reading the pinned entry") != std::string::npos) {
            ++pin_busy;
          } else {
            ++pin_other;
            record_sample("pin", err);
          }
        } else {
          ++pin_ok;
          pinned = true;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });

  // A pin/unpin round costs seconds (parquet rescan + GPU upload), not the
  // 10 ms inter-op sleep, so the worker run must be long enough for cycles to
  // accumulate — 2x iterations yielded only ~6 churn ops.
  auto failures = run_workers(
    *env.db,
    workers(),
    iters_per_worker() * 4,
    [&](int wid, int i) { return env.shapes[(wid + i) % env.shapes.size()]; },
    [&](int wid, int i) { return env.reference[(wid + i) % env.shapes.size()]; });
  churn_stop.store(true);
  churn.join();
  require_no_failures(failures);

  for (const auto& s : error_samples) {
    UNSCOPED_INFO("churn error sample — " << s);
  }
  const int pin_attempts = pin_ok.load() + pin_busy.load() + pin_other.load();
  INFO("churn: unpin_ok=" << unpin_ok.load() << " unpin_fail=" << unpin_fail.load()
                          << " pin_ok=" << pin_ok.load() << " pin_busy(F5)=" << pin_busy.load()
                          << " pin_other=" << pin_other.load() << " (spurious-rejection rate "
                          << pin_busy.load() << "/" << pin_attempts << ")");

  // The churn must actually have exercised the machinery. Calibration: a full
  // unpin+re-pin cycle costs ~1-2 s here, so a several-minute worker window
  // yields roughly a dozen ops; below 8 the scenario is not testing anything.
  REQUIRE(unpin_ok.load() + pin_ok.load() >= 8);
  // F5 rejections are tolerated-and-reported; any OTHER pin/unpin error is a
  // real bug (e.g. a re-pin merge mismatch or a torn map).
  REQUIRE(pin_other.load() == 0);
  REQUIRE(unpin_fail.load() == 0);

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << (stats.executions - stats_before.executions) << " runtime_fallbacks="
                     << (stats.runtime_fallbacks - stats_before.runtime_fallbacks));
  // Serving from a churning pinned entry must never fail at runtime — an
  // unpin mid-scan only drops the map slot; providers co-own the data.
  REQUIRE(stats.runtime_fallbacks - stats_before.runtime_fallbacks == 0);
  REQUIRE(stats.executions - stats_before.executions >=
          static_cast<std::uint64_t>(workers()) * (iters_per_worker() * 4));

  duckdb::Connection con(*env.db);
  // The churn thread may have stopped in the unpinned state; only dim_pin is
  // guaranteed to still exist.
  auto un1 = con.Query("CALL unpin_table('fact_pin')");
  (void)un1;
  auto un2 = con.Query("CALL unpin_table('dim_pin')");
  REQUIRE_FALSE(un2->HasError());
}

TEST_CASE("adversarial: re-pinning one table is never rejected by queries on another",
          "[concurrency][adversarial][pin_table][isolated_context]")
{
  // F5, the exact scenario: a churn worker RE-PINS table X (the merge path,
  // whose use_count() > 1 guard protects in-place mutation of a live entry)
  // while query workers run continuously against table Y only. The old
  // try_match_cached_entry snapshot took a shared_ptr to EVERY pinned entry —
  // including X's, which Y's queries can never serve from — so X's re-pin
  // spuriously failed with "a query is currently reading the pinned entry"
  // whenever any worker was mid-match. The snapshot now takes references only
  // to entries that pass the identity gate, so Y-only queries hold no
  // reference to X's entry and ZERO spurious rejections is structural, not
  // statistical.
  scoped_watchdog dog("pin isolation churn", scenario_timeout(600));

  env_options opt;
  // Small tables: the churn RE-materializes table X on every merge re-pin, and
  // short Y queries maximize match-snapshot traffic.
  opt.rows                              = env_i64("SIRIUS_TEST_PIN_ISOLATION_ROWS", 200'000);
  opt.dim_rows                          = 200'000;
  const std::vector<std::string> shapes = {
    // Y-only shapes: no worker ever scans `fact`, so no worker can ever have a
    // legitimate serving reference to fact_pin's entry.
    "SELECT count(*) AS c, sum(w) AS s FROM dim WHERE w > 100",
    "SELECT bucket, count(*) AS c, min(k) AS lo FROM dim GROUP BY bucket ORDER BY bucket",
  };
  adversarial_env env(opt, shapes);

  {
    duckdb::Connection con(*env.db);
    // X: pinned with a column subset so later re-pins take the same-row-count
    // MERGE path (the use_count-guarded one) instead of building a fresh entry.
    auto pin = con.Query("CALL pin_table('" + env.fact_path.string() +
                         "', tier='gpu', name='fact_pin', cols=['id', 'k'])");
    REQUIRE_FALSE(pin->HasError());
    // Y: pinned too, so the workers' matches do real serving work.
    auto pin2 =
      con.Query("CALL pin_table('" + env.dim_path.string() + "', tier='gpu', name='dim_pin')");
    REQUIRE_FALSE(pin2->HasError());
  }

  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();

  std::atomic<bool> churn_stop{false};
  std::atomic<int> repin_ok{0};
  std::atomic<int> repin_busy{0};   // F5 signature: "currently reading the pinned entry"
  std::atomic<int> repin_other{0};  // anything else is a real bug
  std::mutex sample_mutex;
  std::vector<std::string> error_samples;

  std::thread churn([&] {
    duckdb::Connection con(*env.db);
    bool flip = false;
    while (!churn_stop.load()) {
      // Alternating column subsets keep every iteration on the merge path of a
      // LIVE entry (identical chunk boundaries: same file, all-BIGINT subsets).
      const std::string cols = flip ? "cols=['id', 'v']" : "cols=['id', 'k']";
      flip                   = !flip;
      auto re                = con.Query("CALL pin_table('" + env.fact_path.string() +
                          "', tier='gpu', name='fact_pin', " + cols + ")");
      if (re->HasError()) {
        const auto err = re->GetError();
        if (err.find("currently reading the pinned entry") != std::string::npos) {
          ++repin_busy;
        } else {
          ++repin_other;
        }
        std::lock_guard<std::mutex> lock(sample_mutex);
        if (error_samples.size() < 10) { error_samples.push_back(err); }
      } else {
        ++repin_ok;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });

  auto failures = run_workers(
    *env.db,
    workers(),
    iters_per_worker() * 20,
    [&](int wid, int i) { return env.shapes[(wid + i) % env.shapes.size()]; },
    [&](int wid, int i) { return env.reference[(wid + i) % env.shapes.size()]; });
  churn_stop.store(true);
  churn.join();
  require_no_failures(failures);

  for (const auto& s : error_samples) {
    UNSCOPED_INFO("re-pin error sample — " << s);
  }
  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("re-pins ok=" << repin_ok.load() << " busy(F5)=" << repin_busy.load()
                     << " other=" << repin_other.load() << " executions="
                     << (stats.executions - stats_before.executions) << " runtime_fallbacks="
                     << (stats.runtime_fallbacks - stats_before.runtime_fallbacks));

  // The churn must have exercised the merge path a meaningful number of times
  // while queries flowed.
  REQUIRE(repin_ok.load() >= 8);
  REQUIRE(stats.executions - stats_before.executions >=
          static_cast<std::uint64_t>(workers()) * (iters_per_worker() * 20));
  // THE F5 assertion: queries on Y hold no reference to X's entry, so X's
  // merge re-pin can never see a foreign use_count — zero spurious rejections.
  REQUIRE(repin_busy.load() == 0);
  REQUIRE(repin_other.load() == 0);

  duckdb::Connection con(*env.db);
  auto un1 = con.Query("CALL unpin_table('fact_pin')");
  REQUIRE_FALSE(un1->HasError());
  auto un2 = con.Query("CALL unpin_table('dim_pin')");
  REQUIRE_FALSE(un2->HasError());
}

TEST_CASE("adversarial: plan generation is not blocked by a held execution slot",
          "[concurrency][adversarial][plan_view][isolated_context]")
{
  // F2: the plan-time guard used to occupy a full execution-window slot, so a
  // query A EXECUTING blocked query B from even being PLANNED. With ONE slot
  // (the strictest shape), a heavy query holds it for seconds; the peer's
  // Prepare (which runs the transparent OnFinalizePrepare rebind, i.e. full
  // Sirius plan generation) must complete WHILE that slot is held. Before the
  // fix the peer's Prepare parked in acquire_query_lifecycle_slot until the
  // heavy query finished, so zero prepares could land inside the held window.
  scoped_watchdog dog("plan view vs execution", scenario_timeout(600));

  env_options opt;
  opt.max_concurrent_queries = 1;
  opt.gpu_pool_bytes =
    static_cast<std::uint64_t>(env_i64("SIRIUS_TEST_SPILL_POOL_BYTES", 1'500'000'000));
  opt.rows = env_i64("SIRIUS_TEST_PLAN_VIEW_ROWS", 8'000'000);
  adversarial_env env(opt);

  // Heavy: the fact-fact self join from the spill storm — several seconds of
  // window hold on this pool. Its reference is computed up front.
  const std::string heavy_sql =
    "SELECT count(*) AS c, sum(a.v + b.v) AS s FROM fact a JOIN fact b ON a.id = b.id";
  const std::string heavy_ref = env.reference_for(heavy_sql);

  std::atomic<bool> heavy_done{false};
  std::string heavy_error;
  std::string heavy_result;
  std::thread heavy([&] {
    duckdb::Connection con(*env.db);
    auto r = con.Query(heavy_sql);
    if (r->HasError()) {
      heavy_error = r->GetError();
    } else {
      heavy_result = materialize(*r);
    }
    heavy_done.store(true);
  });

  // Wait for the heavy query's execution window (the single slot) to be held.
  while (!env.sirius_ctx->is_query_lifecycle_active() && !heavy_done.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Peer prepares while the slot is held. A prepare counts only when the slot
  // was held BEFORE and AFTER it (the heavy window spanned it): with one slot
  // and one peer connection, a slot-consuming plan guard can never satisfy
  // that — the prepare would park until the heavy query released the slot.
  const auto stats_before  = env.sirius_ctx->get_transparent_execution_stats();
  int prepares_inside_hold = 0;
  int prepares_total       = 0;
  {
    duckdb::Connection con(*env.db);
    while (!heavy_done.load()) {
      const bool held_before = env.sirius_ctx->is_query_lifecycle_active();
      auto prepared          = con.Prepare(
        "SELECT count(*) AS c, sum(v) AS s, min(id) AS lo, max(id) AS hi "
                 "FROM fact WHERE k < 13");
      REQUIRE_FALSE(prepared->HasError());
      ++prepares_total;
      if (held_before && env.sirius_ctx->is_query_lifecycle_active() && !heavy_done.load()) {
        ++prepares_inside_hold;
      }
    }
  }
  heavy.join();
  REQUIRE(heavy_error.empty());
  REQUIRE(heavy_result == heavy_ref);

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("prepares_total=" << prepares_total << " prepares_inside_hold=" << prepares_inside_hold
                         << " rebinds="
                         << (stats.successful_rebinds - stats_before.successful_rebinds)
                         << " peak=" << env.sirius_ctx->query_lifecycle_peak());
  // The prepares must be real Sirius plan generations (transparent rebinds),
  // not plan-time CPU declines that never touched the Sirius planner.
  REQUIRE(stats.successful_rebinds - stats_before.successful_rebinds >=
          static_cast<std::uint64_t>(prepares_total));
  // At least one full plan generation landed strictly inside the held window.
  REQUIRE(prepares_inside_hold >= 1);
  // Planning does not consume the execution window: the single-slot peak can
  // never exceed 1 no matter how many prepares overlapped the execution.
  REQUIRE(env.sirius_ctx->query_lifecycle_peak() == 1);
}

TEST_CASE("adversarial: concurrent queries across two GPUs",
          "[concurrency][adversarial][mgpu][isolated_context]")
{
  // The per-query GPU admission fix moved set_active_gpu_ids state off the
  // shared task_creator; this proves concurrent queries on num_gpus=2 stay
  // correct. Requires two visible CUDA devices; skips politely otherwise.
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("2-GPU adversarial scenario requires >=2 GPUs; found " << device_count << " — skipping");
    return;
  }

  scoped_watchdog dog("2-GPU concurrency", scenario_timeout(900));

  env_options opt;
  opt.num_gpus = 2;
  opt.rows     = env_i64("SIRIUS_TEST_MGPU_ROWS", 4'000'000);
  adversarial_env env(opt);

  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();

  auto failures = run_workers(
    *env.db,
    workers(),
    iters_per_worker(),
    [&](int wid, int i) { return env.shapes[(wid + i) % env.shapes.size()]; },
    [&](int wid, int i) { return env.reference[(wid + i) % env.shapes.size()]; });
  require_no_failures(failures);

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << (stats.executions - stats_before.executions) << " runtime_fallbacks="
                     << (stats.runtime_fallbacks - stats_before.runtime_fallbacks)
                     << " peak=" << env.sirius_ctx->query_lifecycle_peak());
  REQUIRE(stats.executions - stats_before.executions >=
          static_cast<std::uint64_t>(workers()) * iters_per_worker());
  REQUIRE(stats.runtime_fallbacks - stats_before.runtime_fallbacks == 0);
  if (slots() > 1 && workers() > 1) { REQUIRE(env.sirius_ctx->query_lifecycle_peak() > 1); }
}
