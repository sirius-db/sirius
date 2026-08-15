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

// Group-E (shared mutable configuration) adversarial scenarios from
// docs/concurrency/00-issue-register.md:
//
//   - E4: PhysicalSiriusExecution::logical_plan_ is `mutable` and reset from
//     a const source method; DuckDB shares one prepared PhysicalOperator
//     across EXECUTEs. Two threads EXECUTE-ing one prepared statement race
//     copy_logical_plan(*logical_plan_) against logical_plan_.reset().
//   - E1: operator_params is one plain non-atomic struct per
//     DatabaseInstance, written by `SET` callbacks and read mid-plan and
//     mid-execution — the removed lifecycle slot was the only serialization.
//   - E2: duckdb::Config process-wide static variables
//     (EXPRESSION_EVALUATOR_STRATEGY is read as a default argument on every
//     expression_evaluator construction) written by `SET` with no guard.
//
// A failing assertion here is a deliverable: scenarios that fail on current
// code are tagged [!mayfail] with the failure signature documented at the
// assertion site, so the suite stays green while the repro ships.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/stream_query_result.hpp>
#include <utils/concurrent_test_utils.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace sirius::test::concurrent;

// Barrier-synced N-thread runner over an arbitrary callable (no per-thread
// connection — the prepared-statement scenarios deliberately SHARE state).
void run_racers(int n_threads, const std::function<void(int)>& body)
{
  std::atomic<int> ready{0};
  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool go = false;

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(n_threads));
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t] {
      {
        std::unique_lock<std::mutex> lock(start_mutex);
        ++ready;
        start_cv.wait(lock, [&] { return go; });
      }
      body(t);
    });
  }
  while (ready.load() < n_threads) {
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
}

}  // namespace

TEST_CASE("adversarial: one prepared statement EXECUTEd concurrently and repeatedly",
          "[concurrency][adversarial][prepared][isolated_context]")
{
  // E4's workload: PREPARE once, EXECUTE many times from several threads, in
  // all three DuckDB entry forms: C++ materialized, C++ streaming, and SQL
  // PREPARE/EXECUTE. The shared PhysicalSiriusExecution is immutable across
  // executions since the E4 fix (per-execution plan capture; no
  // logical_plan_.reset() from the const source path), so every EXECUTE must
  // rebuild its own Sirius plan and ENGAGE the GPU — correctness alone would
  // also pass on a silently retained CPU plan, hence the per-phase
  // executions-delta equalities below.
  //
  // The register's earlier failure signature for this scenario ("streaming
  // EXECUTEs silently skip Sirius") was misattributed: instrumenting every
  // decline path shows streaming EXECUTEs engage on every iteration (the GPU
  // run completes inside Execute()'s single ClientContext lock hold, before
  // any sibling can invalidate the stream), while SQL-level EXECUTE never
  // engages — PREPARE/EXECUTE bypasses OnFinalizePrepare entirely (the PREPARE
  // statement itself is not a SELECT, and Binder::Bind(ExecuteStatement)
  // re-plans via Planner::PrepareSQLStatement, which has no finalize hook).
  // That is a pre-existing, single-threaded transparency limitation, not a
  // race — see the note in test_transparent_runtime_fallback.cpp. Phase C
  // therefore asserts correctness under contention plus ZERO transparent
  // engagement, so a future PREPARE/EXECUTE interception flips it consciously.
  scoped_watchdog dog("prepared double-EXECUTE", scenario_timeout(600));

  adversarial_env env;
  const std::string& sql = env.shapes[0];
  const std::string& ref = env.reference[0];

  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();
  const int iters         = env_int("SIRIUS_TEST_PREPARED_ITERS", 15);

  std::mutex failures_mutex;
  std::vector<std::string> failures;
  auto add_failure = [&](const std::string& f) {
    std::lock_guard<std::mutex> lock(failures_mutex);
    failures.push_back(f);
  };

  // Phase 0 — single-threaded streaming baseline: one EXECUTE with
  // allow_stream_result=true, materialized by hand. Establishes whether a
  // lone streaming EXECUTE engages the GPU at all, so the racing phases below
  // can tell "streaming never executes transparently" apart from "the race
  // broke it".
  {
    duckdb::Connection con(*env.db);
    auto stmt = con.Prepare(sql);
    REQUIRE_FALSE(stmt->HasError());
    duckdb::vector<duckdb::Value> values;
    auto r = stmt->Execute(values, /*allow_stream_result=*/true);
    REQUIRE_FALSE(r->HasError());
    std::string got;
    if (r->type == duckdb::QueryResultType::STREAM_RESULT) {
      auto materialized = r->Cast<duckdb::StreamQueryResult>().Materialize();
      REQUIRE_FALSE(materialized->HasError());
      got = materialize(*materialized);
    } else {
      got = materialize(*r);
    }
    REQUIRE(got == ref);
    const auto stats_now = env.sirius_ctx->get_transparent_execution_stats();
    INFO("phase 0 (single streaming EXECUTE): executions delta="
         << (stats_now.executions - stats_before.executions)
         << " rebinds delta=" << (stats_now.successful_rebinds - stats_before.successful_rebinds)
         << " plan_time_fallbacks delta=" << (stats_now.fallbacks - stats_before.fallbacks));
    CHECK(stats_now.executions - stats_before.executions >= 1);
  }
  const auto stats_after_phase0 = env.sirius_ctx->get_transparent_execution_stats();

  // Phase A — C++ API, materialized results, 3 threads on ONE PreparedStatement.
  std::uint64_t phase_a_execs = 0;
  {
    duckdb::Connection con(*env.db);
    auto stmt = con.Prepare(sql);
    REQUIRE_FALSE(stmt->HasError());
    run_racers(3, [&](int tid) {
      for (int i = 0; i < iters; ++i) {
        duckdb::vector<duckdb::Value> values;
        auto r = stmt->Execute(values, /*allow_stream_result=*/false);
        if (r->HasError()) {
          add_failure("phase A thread " + std::to_string(tid) + " iter " + std::to_string(i) +
                      " ERROR: " + r->GetError());
          continue;
        }
        if (auto got = materialize(*r); got != ref) {
          add_failure("phase A thread " + std::to_string(tid) + " iter " + std::to_string(i) +
                      " WRONG RESULT:\n--- got ---\n" + got + "\n--- expected ---\n" + ref);
        }
      }
    });
    phase_a_execs = static_cast<std::uint64_t>(3) * iters;
  }
  const auto stats_after_phase_a = env.sirius_ctx->get_transparent_execution_stats();
  INFO("phase A: executions delta="
       << (stats_after_phase_a.executions - stats_after_phase0.executions));
  // Every materialized EXECUTE ran through the Sirius operator exactly once.
  REQUIRE(stats_after_phase_a.executions - stats_after_phase0.executions == phase_a_execs);

  // Phase B — C++ API, STREAMING results, 2 threads on one PreparedStatement.
  // DuckDB invalidates an open stream when the next query starts on the same
  // connection, so "result invalidated/closed" errors are tolerated (counted);
  // a wrong RESULT or a crash is not.
  std::atomic<int> phase_b_ok{0};
  std::atomic<int> phase_b_invalidated{0};
  std::mutex invalidated_mutex;
  std::vector<std::string> invalidated_samples;
  {
    duckdb::Connection con(*env.db);
    auto stmt = con.Prepare(sql);
    REQUIRE_FALSE(stmt->HasError());
    run_racers(2, [&](int tid) {
      for (int i = 0; i < iters; ++i) {
        duckdb::vector<duckdb::Value> values;
        std::string got;
        // Tolerated-and-counted: DuckDB invalidates an open stream when the
        // next query starts on the same connection, so racing streams
        // legitimately error. Real bugs surface as wrong RESULTS or crashes.
        auto tolerate = [&](const std::string& err) {
          ++phase_b_invalidated;
          std::lock_guard<std::mutex> lock(invalidated_mutex);
          if (invalidated_samples.size() < 5) { invalidated_samples.push_back(err); }
        };
        try {
          auto r = stmt->Execute(values, /*allow_stream_result=*/true);
          if (r->HasError()) {
            tolerate(r->GetError());
            continue;
          }
          if (r->type == duckdb::QueryResultType::STREAM_RESULT) {
            // ToString() on a stream renders "[[STREAM RESULT]]"; materialize
            // it (draining the stream, which may observe invalidation).
            auto materialized = r->Cast<duckdb::StreamQueryResult>().Materialize();
            if (materialized->HasError()) {
              tolerate(materialized->GetError());
              continue;
            }
            got = materialize(*materialized);
          } else {
            got = materialize(*r);
          }
        } catch (std::exception& e) {
          tolerate(e.what());
          continue;
        }
        if (got != ref) {
          add_failure("phase B thread " + std::to_string(tid) + " iter " + std::to_string(i) +
                      " WRONG RESULT:\n--- got ---\n" + got + "\n--- expected ---\n" + ref);
          continue;
        }
        ++phase_b_ok;
      }
    });
  }
  {
    // Dump early failures before the phase gate below aborts the test.
    std::lock_guard<std::mutex> lock(failures_mutex);
    for (std::size_t f = 0; f < failures.size() && f < 3; ++f) {
      UNSCOPED_INFO(failures[f]);
    }
  }
  for (const auto& s : invalidated_samples) {
    UNSCOPED_INFO("phase B tolerated error sample: " << s);
  }
  const auto stats_after_phase_b = env.sirius_ctx->get_transparent_execution_stats();
  INFO("phase B: ok=" << phase_b_ok.load() << " invalidated=" << phase_b_invalidated.load()
                      << " executions delta="
                      << (stats_after_phase_b.executions - stats_after_phase_a.executions));
  REQUIRE(phase_b_ok.load() >= 1);
  // THE E4 assertion: every streaming EXECUTE engaged the GPU — including the
  // ones whose stream a sibling later invalidated (the transparent execution
  // completes inside Execute()'s lock hold, before the stream is exposed).
  // Equality keeps this a live tripwire in both directions: a silent CPU
  // degrade AND a double-execution both trip it.
  REQUIRE(stats_after_phase_b.executions - stats_after_phase_a.executions ==
          static_cast<std::uint64_t>(2) * iters);

  // Phase C — SQL PREPARE / EXECUTE, 2 threads sharing the SAME connection.
  std::uint64_t phase_c_execs = 0;
  {
    duckdb::Connection con(*env.db);
    auto prep = con.Query("PREPARE adversarial_p1 AS " + sql);
    REQUIRE_FALSE(prep->HasError());
    run_racers(2, [&](int tid) {
      for (int i = 0; i < iters; ++i) {
        auto r = con.Query("EXECUTE adversarial_p1");
        if (r->HasError()) {
          add_failure("phase C thread " + std::to_string(tid) + " iter " + std::to_string(i) +
                      " ERROR: " + r->GetError());
          continue;
        }
        if (auto got = materialize(*r); got != ref) {
          add_failure("phase C thread " + std::to_string(tid) + " iter " + std::to_string(i) +
                      " WRONG RESULT:\n--- got ---\n" + got + "\n--- expected ---\n" + ref);
        }
      }
    });
    phase_c_execs = static_cast<std::uint64_t>(2) * iters;
  }

  require_no_failures(failures);

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << (stats.executions - stats_after_phase0.executions) << " runtime_fallbacks="
                     << (stats.runtime_fallbacks - stats_after_phase0.runtime_fallbacks)
                     << " plan_time_fallbacks=" << (stats.fallbacks - stats_after_phase0.fallbacks)
                     << " successful_rebinds="
                     << (stats.successful_rebinds - stats_after_phase0.successful_rebinds));
  // SQL-level PREPARE/EXECUTE is NOT intercepted by Sirius (pre-existing
  // single-threaded limitation, see the test header): phase C's value here is
  // correctness under contention (checked per-iteration above). Assert the
  // zero engagement so a future PREPARE/EXECUTE interception — or a regression
  // that starts double-counting — flips this consciously.
  (void)phase_c_execs;
  REQUIRE(stats.executions - stats_after_phase_b.executions == 0);
  // No mid-flight plan was corrupted into the runtime-fallback path anywhere.
  REQUIRE(stats.runtime_fallbacks - stats_after_phase0.runtime_fallbacks == 0);
  REQUIRE(stats.fallbacks - stats_after_phase0.fallbacks == 0);
}

TEST_CASE("adversarial: SET storm on operator params races concurrent executions",
          "[concurrency][adversarial][config_race][isolated_context]")
{
  // E1/E2 torn-read regression: one worker hammers a rotation across the FULL
  // SET-mutable surface — operator_params sizing knobs (E1: one shared struct,
  // previously written mid-flight under everyone's plans), plan-shape toggles
  // (dynamic filter pushdown, runtime distinct build-probe) and the
  // process-global expression_evaluator_strategy (E2: previously re-read as a
  // default argument BETWEEN two operators of a peer's plan) — while the other
  // workers run verified queries.
  //
  // With SNAPSHOT-AT-WINDOW-BEGIN, each query freezes the whole config at
  // admission: results must stay correct under every interleaving, no query
  // may fall back at runtime (a runtime fallback here means a mid-flight plan
  // was corrupted), and every query must still engage the GPU (a torn sizing
  // read that poisons plan generation would surface as a plan-time fallback).
  scoped_watchdog dog("SET-vs-execution race", scenario_timeout(600));

  adversarial_env env;

  // The rotation below perturbs settings, and expression_evaluator_strategy is
  // PROCESS-GLOBAL (duckdb::Config static) — it outlives this test's isolated
  // DB. A mid-rotation exit, including a REQUIRE throw, must not leak the
  // perturbed half into the rest of the suite: under 'materialize', inequality
  // nested-loop/mixed-join shapes runtime-fall-back, which failed 13 unrelated
  // gpu_execution tests when the storm happened to end on that half. RESET
  // every knob in the rotation on every exit path.
  struct storm_reset_guard {
    duckdb::DuckDB& db;
    ~storm_reset_guard()
    {
      static constexpr const char* knobs[] = {"scan_task_batch_size",
                                              "expression_evaluator_strategy",
                                              "hash_partition_bytes",
                                              "max_build_hash_table_bytes",
                                              "enable_dynamic_filter_pushdown",
                                              "concat_batch_bytes",
                                              "max_sort_partition_bytes",
                                              "sort_sample_bytes",
                                              "enable_runtime_distinct_build_probe",
                                              "mark_join_build_switch_ratio",
                                              "dynamic_filter_keep_threshold",
                                              "max_broadcast_join_size"};
      try {
        duckdb::Connection con(db);
        for (const char* knob : knobs) {
          (void)con.Query(std::string("RESET ") + knob);
        }
      } catch (...) {  // never mask the test's own failure
      }
    }
  } reset_guard{*env.db};

  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();

  const int query_workers = std::max(2, workers() - 1);
  const int query_iters   = env_int("SIRIUS_TEST_SET_RACE_QUERY_ITERS", 10);
  const int set_iters     = env_int("SIRIUS_TEST_SET_RACE_SET_ITERS", 60);

  auto set_sql = [](int i) -> std::string {
    // Alternate every knob between a small/perturbing value and its
    // config-default-ish value so mid-storm admissions see wildly different
    // — but each internally consistent — snapshots.
    static const std::vector<std::string> statements = {
      "SET scan_task_batch_size = 262144",
      "SET expression_evaluator_strategy = 'materialize'",
      "SET hash_partition_bytes = 4194304",
      "SET max_build_hash_table_bytes = 1048576",
      "SET enable_dynamic_filter_pushdown = false",
      "SET concat_batch_bytes = 4194304",
      "SET max_sort_partition_bytes = 8388608",
      "SET sort_sample_bytes = 1048576",
      "SET enable_runtime_distinct_build_probe = false",
      "SET mark_join_build_switch_ratio = 1.0",
      "SET dynamic_filter_keep_threshold = 0.1",
      "SET max_broadcast_join_size = 1048576",
      "SET scan_task_batch_size = 100000000",
      "SET expression_evaluator_strategy = 'ast_interpret'",
      "SET hash_partition_bytes = 100000000",
      "SET max_build_hash_table_bytes = 90000000",
      "SET enable_dynamic_filter_pushdown = true",
      "SET concat_batch_bytes = 100000000",
      "SET max_sort_partition_bytes = 0",
      "SET sort_sample_bytes = 100000000",
      "SET enable_runtime_distinct_build_probe = true",
      "SET mark_join_build_switch_ratio = 8.0",
      "SET dynamic_filter_keep_threshold = 0.9",
      "SET max_broadcast_join_size = 268435456",
    };
    return statements[static_cast<std::size_t>(i) % statements.size()];
  };

  auto failures = run_workers(
    *env.db,
    query_workers + 1,
    [&](int wid) { return wid == 0 ? set_iters : query_iters; },
    [&](int wid, int i) {
      if (wid == 0) { return set_sql(i); }
      return env.shapes[(wid + i) % env.shapes.size()];
    },
    [&](int wid, int i) -> std::string {
      if (wid == 0) { return {}; }  // SETs: only require no error
      return env.reference[(wid + i) % env.shapes.size()];
    });
  require_no_failures(failures);

  const auto stats = env.sirius_ctx->get_transparent_execution_stats();
  INFO("executions=" << (stats.executions - stats_before.executions) << " runtime_fallbacks="
                     << (stats.runtime_fallbacks - stats_before.runtime_fallbacks)
                     << " plan_time_fallbacks=" << (stats.fallbacks - stats_before.fallbacks)
                     << " peak=" << env.sirius_ctx->query_lifecycle_peak());
  REQUIRE(stats.executions - stats_before.executions >=
          static_cast<std::uint64_t>(query_workers) * query_iters);
  REQUIRE(stats.runtime_fallbacks - stats_before.runtime_fallbacks == 0);
  // A torn sizing read poisoning plan generation would surface here: all four
  // shapes are GPU-supported under every value in the storm rotation.
  REQUIRE(stats.fallbacks - stats_before.fallbacks == 0);
  if (slots() > 1) { REQUIRE(env.sirius_ctx->query_lifecycle_peak() > 1); }
}

TEST_CASE("adversarial: concurrent gpu_execution() executions restore the connection config",
          "[concurrency][adversarial][config_race][isolated_context]")
{
  // E7: SiriusTableFunctionData used to save/restore ClientConfig on SHARED
  // bind data — two executions of one gpu_execution(...) prepared statement
  // could clobber each other's saved copy, leaving the connection with
  // enable_optimizer permanently flipped. The save/restore is per-execution
  // (RAII on the executing stack) now; this scenario runs many overlapping
  // executions of ONE prepared statement on a connection whose
  // enable_optimizer differs from the value gpu_execution installs while
  // planning, then asserts the connection got its own value back.
  scoped_watchdog dog("gpu_execution config restore", scenario_timeout(600));

  adversarial_env env;
  duckdb::Connection con(*env.db);

  // gpu_execution's ExtractPlan installs enable_optimizer = true while it
  // plans; start from FALSE so a clobbered restore is observable.
  con.context->config.enable_optimizer = false;

  const std::string& sql = env.shapes[2];  // scalar aggregate: order-stable
  const std::string& ref = env.reference[2];
  auto stmt              = con.Prepare("SELECT * FROM gpu_execution('" + sql + "')");
  REQUIRE_FALSE(stmt->HasError());

  const int iters = env_int("SIRIUS_TEST_GPU_EXECUTION_ITERS", 10);
  std::mutex failures_mutex;
  std::vector<std::string> failures;

  run_racers(2, [&](int tid) {
    for (int i = 0; i < iters; ++i) {
      duckdb::vector<duckdb::Value> values;
      auto r = stmt->Execute(values, /*allow_stream_result=*/false);
      if (r->HasError()) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("thread " + std::to_string(tid) + " iter " + std::to_string(i) +
                           " ERROR: " + r->GetError());
        continue;
      }
      if (auto got = materialize(*r); got != ref) {
        std::lock_guard<std::mutex> lock(failures_mutex);
        failures.push_back("thread " + std::to_string(tid) + " iter " + std::to_string(i) +
                           " WRONG RESULT:\n--- got ---\n" + got + "\n--- expected ---\n" + ref);
      }
    }
  });
  require_no_failures(failures);

  // The E7 assertion: after every execution's save/restore has unwound, the
  // connection still has ITS configuration — not a value some overlapping
  // execution saved mid-plan and wrote back last.
  REQUIRE(con.context->config.enable_optimizer == false);

  // And the mid-plan flip never leaked to a peer connection's config either.
  duckdb::Connection peer(*env.db);
  REQUIRE(peer.context->config.enable_optimizer == true);
}
