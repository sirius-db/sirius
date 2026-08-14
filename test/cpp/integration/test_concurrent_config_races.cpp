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
          "[concurrency][adversarial][prepared][isolated_context][!mayfail]")
{
  // E4's workload: PREPARE once, EXECUTE many times from several threads.
  // DuckDB's ClientContext lock serializes the executions internally, so the
  // sharp UAF window (copy_logical_plan vs logical_plan_.reset()) needs the
  // reset path (non-serializable plan) to open fully — but the repeated-reuse
  // path (fresh Sirius physical plan per EXECUTE off one shared mutable
  // logical_plan_) is exercised on every iteration, from alternating threads,
  // in all three DuckDB entry forms: C++ materialized, C++ streaming, and SQL
  // PREPARE/EXECUTE.
  //
  // KNOWN FAILURE on integration/concurrency-full (2026-08-14), hence
  // [!mayfail]: the final GPU-engagement REQUIRE fails. Signature:
  //   phase 0 (single streaming EXECUTE): executions delta=1  -> streaming
  //     alone transparently executes on the GPU;
  //   phase A (3 threads, materialized): 45/45 executions+rebinds;
  //   phase C (2 threads, SQL EXECUTE):  30/30 executions+rebinds;
  //   phase B (2 threads, STREAMING):    ok=16 correct results but
  //     executions delta = 0, successful_rebinds = 2/30,
  //     plan_time_fallbacks = 0, runtime_fallbacks = 0.
  // I.e. when a streaming EXECUTE's bind overlaps a sibling's still-draining
  // transparent execution window on the SAME connection, the bind silently
  // skips Sirius (no capture, no rebind, no fallback accounting) and the
  // query runs DuckDB's retained CPU plan. Results stay correct; GPU
  // engagement and observability are silently lost. Register: E4-family
  // (shared prepared-statement state; per-connection capture/guard state
  // raced by concurrent EXECUTEs).
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
  INFO("phase B: ok=" << phase_b_ok.load() << " invalidated=" << phase_b_invalidated.load());
  REQUIRE(phase_b_ok.load() >= 1);

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
  // Every successful iteration must have gone through the Sirius operator —
  // a prepared statement that silently degraded to a CPU plan would pass on
  // correctness alone.
  REQUIRE(stats.executions - stats_after_phase0.executions >=
          phase_a_execs + static_cast<std::uint64_t>(phase_b_ok.load()) + phase_c_execs);
  REQUIRE(stats.runtime_fallbacks - stats_after_phase0.runtime_fallbacks == 0);
}

TEST_CASE("adversarial: SET on operator params races concurrent executions",
          "[concurrency][adversarial][config_race][isolated_context]")
{
  // One worker hammers `SET scan_task_batch_size` (E1: plain struct written
  // mid-flight, also reshapes scan batching under everyone's feet) and
  // `SET expression_evaluator_strategy` (E2: process-wide static read as a
  // default argument between two operators of a peer's plan) while the other
  // workers run verified queries. Results must stay correct under every
  // torn/interleaved combination; a runtime fallback here means a strategy
  // switch corrupted a mid-flight plan.
  scoped_watchdog dog("SET-vs-execution race", scenario_timeout(600));

  adversarial_env env;

  const auto stats_before = env.sirius_ctx->get_transparent_execution_stats();

  const int query_workers = std::max(2, workers() - 1);
  const int query_iters   = env_int("SIRIUS_TEST_SET_RACE_QUERY_ITERS", 10);
  const int set_iters     = env_int("SIRIUS_TEST_SET_RACE_SET_ITERS", 60);

  auto set_sql = [](int i) -> std::string {
    switch (i % 4) {
      case 0: return "SET scan_task_batch_size = 262144";
      case 1: return "SET expression_evaluator_strategy = 'materialize'";
      case 2: return "SET scan_task_batch_size = 100000000";
      default: return "SET expression_evaluator_strategy = 'ast_interpret'";
    }
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
                     << " peak=" << env.sirius_ctx->query_lifecycle_peak());
  REQUIRE(stats.executions - stats_before.executions >=
          static_cast<std::uint64_t>(query_workers) * query_iters);
  REQUIRE(stats.runtime_fallbacks - stats_before.runtime_fallbacks == 0);
  if (slots() > 1) { REQUIRE(env.sirius_ctx->query_lifecycle_peak() > 1); }
}
