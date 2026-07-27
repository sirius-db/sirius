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

/**
 * @file test_query_lifecycle_slot.cpp
 * @brief Regression test for a known query-lifecycle-lock defect.
 *
 * Sirius acquires a per-DatabaseInstance query-lifecycle lock in QueryBegin and
 * releases it in QueryEnd. DuckDB does not call QueryEnd for an unconsumed
 * streaming or pending prepared-statement result (their destructors do not run
 * cleanup and there is no result-destruction hook), so leaving such a result
 * unconsumed on an idle connection keeps the lock held; a subsequent query on any
 * connection of the same DatabaseInstance then waits in acquire instead of
 * completing.
 *
 * This test pins that behaviour so the fix can be verified. It runs the scenario
 * in a short-lived child process guarded by a bounded deadline, reusing the
 * repository's existing subprocess-watchdog pattern (see the hive-partition tests
 * in test_gpu_execution_multi_format.cpp): a query that should return but instead
 * waits is detected as the child not completing within the deadline. On the
 * unfixed engine the child does not complete (the reproduced wait); once the fix
 * lands, the child completes and the case passes.
 */

#include "sirius_extension.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_search_path.hpp>
#include <duckdb/main/client_config.hpp>
#include <duckdb/main/client_data.hpp>
#include <duckdb/main/pending_query_result.hpp>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utils/transparent_execution_test_utils.hpp>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

constexpr char const* kChildRunnerCase = "query lifecycle slot watchdog child runner";
constexpr char const* kEnvVariant      = "SIRIUS_SLOT_WATCHDOG_VARIANT";
constexpr char const* kEnvOutput       = "SIRIUS_SLOT_WATCHDOG_OUTPUT";
constexpr char const* kEnvConfig       = "SIRIUS_SLOT_WATCHDOG_CONFIG";

// Result the child process reports back to the parent through a small text file.
struct slot_watchdog_result {
  std::string error;         // non-empty if the child reported a failure
  bool timed_out   = false;  // the child did not complete within the deadline
  bool b_started   = false;  // the child reached the variant's observed workload
  bool b_completed = false;  // the observed workload completed successfully
};

template <typename T>
bool require_success(T* result, char const* operation, std::string& error)
{
  if (result == nullptr) {
    error = std::string(operation) + " returned nullptr";
    return false;
  }
  if (result->HasError()) {
    error = std::string(operation) + " failed: " + result->GetError();
    return false;
  }
  return true;
}

template <typename T>
bool require_success(T* result, char const* operation, slot_watchdog_result& out)
{
  return require_success(result, operation, out.error);
}

void write_result(fs::path const& path, slot_watchdog_result const& result)
{
  std::ofstream out(path);
  out << "ERROR\t" << result.error << "\n";
  out << "STARTED\t" << (result.b_started ? 1 : 0) << "\n";
  out << "COMPLETED\t" << (result.b_completed ? 1 : 0) << "\n";
}

slot_watchdog_result read_result(fs::path const& path)
{
  slot_watchdog_result result;
  std::ifstream in(path);
  if (!in) {
    result.error = "watchdog child did not write a result file";
    return result;
  }
  std::string key;
  while (in >> key) {
    if (key == "ERROR") {
      std::string rest;
      std::getline(in, rest);
      if (!rest.empty() && rest.front() == '\t') { rest.erase(rest.begin()); }
      result.error = rest;
    } else if (key == "STARTED") {
      int started = 0;
      in >> started;
      result.b_started = (started != 0);
    } else if (key == "COMPLETED") {
      int done = 0;
      in >> done;
      result.b_completed = (done != 0);
    }
  }
  return result;
}

bool require_scalar_result(duckdb::QueryResult* result,
                           char const* operation,
                           std::string const& expected,
                           std::string& error)
{
  if (!require_success(result, operation, error)) { return false; }

  auto chunk = result->Fetch();
  if (chunk == nullptr || chunk->size() != 1 || chunk->ColumnCount() != 1) {
    error = std::string(operation) + " did not return exactly one scalar row";
    return false;
  }
  auto const actual = chunk->GetValue(0, 0).ToString();

  while (auto extra = result->Fetch()) {
    if (extra->size() != 0) {
      error = std::string(operation) + " returned more than one row";
      return false;
    }
  }
  if (result->HasError()) {
    error = std::string(operation) + " failed while draining: " + result->GetError();
    return false;
  }
  if (actual != expected) {
    error = std::string(operation) + " returned " + actual + ", expected " + expected;
    return false;
  }
  return true;
}

bool run_statement(duckdb::Connection& connection,
                   std::string const& sql,
                   char const* operation,
                   std::string& error)
{
  auto result = connection.Query(sql);
  return require_success(result.get(), operation, error);
}

bool run_scalar_query(duckdb::Connection& connection,
                      std::string const& sql,
                      std::string const& expected,
                      char const* operation,
                      std::string& error,
                      std::chrono::steady_clock::time_point* query_returned_at = nullptr)
{
  auto result = connection.Query(sql);
  if (query_returned_at != nullptr) { *query_returned_at = std::chrono::steady_clock::now(); }
  return require_scalar_result(result.get(), operation, expected, error);
}

bool run_prepared_scalar(duckdb::PreparedStatement& prepared,
                         std::string const& expected,
                         char const* operation,
                         std::string& error)
{
  duckdb::vector<duckdb::Value> parameters;
  auto result = prepared.Execute(parameters, false);
  return require_scalar_result(result.get(), operation, expected, error);
}

std::string range_sum(std::uint64_t count)
{
  auto const sum = count % 2 == 0 ? (count / 2) * (count - 1) : count * ((count - 1) / 2);
  return std::to_string(sum);
}

struct async_query_result {
  std::string error;
  std::chrono::steady_clock::time_point completed_at;
  std::atomic<bool> completed{false};
};

void run_async_scalar_query(duckdb::Connection& connection,
                            std::string const& sql,
                            std::string const& expected,
                            char const* operation,
                            async_query_result& out)
{
  try {
    (void)run_scalar_query(connection, sql, expected, operation, out.error, &out.completed_at);
  } catch (std::exception const& error) {
    out.error = std::string(operation) + " threw: " + error.what();
    if (out.completed_at == std::chrono::steady_clock::time_point{}) {
      out.completed_at = std::chrono::steady_clock::now();
    }
  } catch (...) {
    out.error = std::string(operation) + " threw an unknown exception";
    if (out.completed_at == std::chrono::steady_clock::time_point{}) {
      out.completed_at = std::chrono::steady_clock::now();
    }
  }
  out.completed.store(true, std::memory_order_release);
}

bool create_range_table(duckdb::Connection& connection,
                        std::string const& table_name,
                        std::uint64_t count,
                        std::string& error,
                        bool replace = false)
{
  auto const sql = std::string{"CREATE "} + (replace ? "OR REPLACE " : "") + "TABLE " + table_name +
                   " AS SELECT range AS i FROM range(" + std::to_string(count) + ");";
  return run_statement(connection, sql, "CREATE range table", error);
}

bool set_gpu_execution(duckdb::Connection& connection, bool enabled, std::string& error)
{
  return run_statement(connection,
                       enabled ? "SET gpu_execution=true;" : "SET gpu_execution=false;",
                       enabled ? "SET gpu_execution=true" : "SET gpu_execution=false",
                       error);
}

bool require_transparent_execution_delta(
  duckdb::SiriusContext::transparent_execution_stats const& before,
  duckdb::SiriusContext::transparent_execution_stats const& after,
  std::uint64_t expected_executions,
  char const* operation,
  std::string& error)
{
  if (after.executions != before.executions + expected_executions ||
      after.fallbacks != before.fallbacks || after.runtime_fallbacks != before.runtime_fallbacks) {
    error = std::string(operation) + " did not execute on GPU without fallback";
    return false;
  }
  return true;
}

void mark_workload_started(fs::path const& output_path, slot_watchdog_result& out)
{
  out.b_started = true;
  write_result(output_path, out);
}

void run_abandoned_result_scenario(std::string const& variant,
                                   duckdb::Connection& a,
                                   duckdb::Connection& b,
                                   fs::path const& output_path,
                                   slot_watchdog_result& out)
{
  auto const gpu_follow_up = variant == "wave1_ac1_pending_gpu";
  if (!set_gpu_execution(a, false, out.error) || !set_gpu_execution(b, gpu_follow_up, out.error) ||
      !create_range_table(a, "t", 200000, out.error) ||
      !run_statement(a, "CHECKPOINT;", "CHECKPOINT", out.error)) {
    return;
  }

  // Connection A establishes and then abandons a result, per the variant.
  std::unique_ptr<duckdb::PreparedStatement> prepared;
  std::unique_ptr<duckdb::QueryResult> streamed;
  std::unique_ptr<duckdb::PendingQueryResult> pending;

  if (variant == "stream") {
    if (!set_gpu_execution(a, true, out.error)) { return; }
    auto const stats_before = sirius::test::get_transparent_execution_stats(a);
    prepared                = a.Prepare("SELECT i FROM t;");
    if (!require_success(prepared.get(), "stream Prepare", out)) { return; }
    streamed = prepared->Execute();  // streaming by default
    if (!require_success(streamed.get(), "stream Execute", out)) { return; }
    if (streamed->type != duckdb::QueryResultType::STREAM_RESULT) {
      out.error = "stream Execute did not return a streaming result";
      return;
    }
    auto chunk = streamed->Fetch();
    if (!chunk || chunk->size() == 0) {
      out.error = "stream Execute produced no first chunk";
      return;
    }
    auto const stats_after = sirius::test::get_transparent_execution_stats(a);
    if (!require_transparent_execution_delta(
          stats_before, stats_after, 1, "stream query", out.error)) {
      return;
    }
  } else if (variant == "pending" || gpu_follow_up) {
    if (!set_gpu_execution(a, true, out.error)) { return; }
    prepared = a.Prepare("SELECT i FROM t;");
    if (!require_success(prepared.get(), "pending Prepare", out)) { return; }
    duckdb::shared_ptr<duckdb::SiriusContext> ac1_context;
    if (gpu_follow_up) { ac1_context = sirius::test::get_registered_sirius_context(a); }
    auto const pending_execution_baseline =
      ac1_context ? ac1_context->get_transparent_execution_stats().executions : 0;
    pending = prepared->PendingQuery();  // created, never executed to a result
    if (!require_success(pending.get(), "prepared PendingQuery", out)) { return; }
    if (gpu_follow_up) {
      // DuckDB may legally schedule a PendingQuery's pipeline on a background
      // worker before the caller explicitly drives it. Let that one execution
      // settle into the shared stats baseline so the strict +1 assertion below
      // remains scoped to connection B's follow-up query.
      auto const target      = pending_execution_baseline + 1;
      auto const deadline    = std::chrono::steady_clock::now() + std::chrono::seconds{10};
      std::uint64_t observed = pending_execution_baseline;
      bool saw_target_once   = false;
      bool settled           = false;
      while (std::chrono::steady_clock::now() < deadline) {
        observed = ac1_context->get_transparent_execution_stats().executions;
        if (observed > target) {
          out.error = "AC-1 pending query produced more than one background execution";
          return;
        }
        if (observed == target) {
          if (saw_target_once) {
            settled = true;
            break;
          }
          saw_target_once = true;
        } else {
          saw_target_once = false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
      }
      if (!settled) {
        out.error =
          "AC-1 pending background execution did not settle at exactly +1 within 10s "
          "(baseline=" +
          std::to_string(pending_execution_baseline) + ", observed=" + std::to_string(observed) +
          ")";
        return;
      }
    }
  } else if (variant == "cached_no_rebind") {
    // A plain CPU prepared statement keeps DuckDB on the cached DO_NOT_REBIND
    // execute path (no re-plan), which does not call OnFinalizePrepare.
    prepared = a.Prepare("SELECT i FROM t;");
    if (!require_success(prepared.get(), "cached Prepare", out)) { return; }
    {
      auto first = prepared->Execute();
      if (!require_success(first.get(), "cached first Execute", out)) { return; }
      if (first->type != duckdb::QueryResultType::STREAM_RESULT) {
        out.error = "cached first Execute did not return a streaming result";
        return;
      }
      while (auto chunk = first->Fetch()) {
        (void)chunk;
      }
      if (first->HasError()) {
        out.error = "cached first Execute failed while draining: " + first->GetError();
        return;
      }
    }
    streamed = prepared->Execute();  // second execution: cached, no rebind
    if (!require_success(streamed.get(), "cached second Execute", out)) { return; }
    if (streamed->type != duckdb::QueryResultType::STREAM_RESULT) {
      out.error = "cached second Execute did not return a streaming result";
      return;
    }
    auto chunk = streamed->Fetch();
    if (!chunk || chunk->size() == 0) {
      out.error = "cached second Execute produced no first chunk";
      return;
    }
  } else {
    out.error = "unknown abandoned-result variant: " + variant;
    return;
  }

  // Persist this before entering B so the watchdog can distinguish the expected
  // blocked B query from a setup or connection-A wait.
  mark_workload_started(output_path, out);

  if (gpu_follow_up) {
    auto const stats_before = sirius::test::get_transparent_execution_stats(b);
    if (!run_scalar_query(
          b, "SELECT sum(i) FROM t;", range_sum(200000), "AC-1 GPU aggregate", out.error)) {
      return;
    }
    auto const stats_after = sirius::test::get_transparent_execution_stats(b);
    if (!require_transparent_execution_delta(
          stats_before, stats_after, 1, "AC-1 GPU aggregate", out.error)) {
      return;
    }
  } else {
    auto rb = b.Query("SELECT 42;");
    if (!require_success(rb.get(), "connection B SELECT 42", out)) { return; }
  }
  out.b_completed = true;
}

bool wait_for_workers(std::atomic<unsigned>& ready, unsigned expected)
{
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  while (ready.load(std::memory_order_acquire) != expected) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::yield();
  }
  return true;
}

void run_ac2_gpu_single_flight(duckdb::Connection& a,
                               duckdb::Connection& b,
                               fs::path const& output_path,
                               slot_watchdog_result& out)
{
  constexpr std::uint64_t kCountA = 1000000;
  constexpr std::uint64_t kCountB = 750000;
  if (!set_gpu_execution(a, false, out.error) || !set_gpu_execution(b, false, out.error) ||
      !create_range_table(a, "gpu_a", kCountA, out.error) ||
      !create_range_table(a, "gpu_b", kCountB, out.error) ||
      !run_statement(a, "CHECKPOINT;", "CHECKPOINT", out.error) ||
      !set_gpu_execution(a, true, out.error) || !set_gpu_execution(b, true, out.error)) {
    return;
  }

  auto sirius_context     = sirius::test::get_registered_sirius_context(a);
  auto const stats_before = sirius_context->get_transparent_execution_stats();
  std::atomic<unsigned> ready{0};
  std::atomic<bool> start{false};
  async_query_result result_a;
  async_query_result result_b;

  auto worker = [&](duckdb::Connection& connection,
                    std::string const& sql,
                    std::string const& expected,
                    char const* operation,
                    async_query_result& result) {
    ready.fetch_add(1, std::memory_order_release);
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    run_async_scalar_query(connection, sql, expected, operation, result);
  };
  std::thread thread_a([&]() {
    worker(
      a, "SELECT sum(i) FROM gpu_a;", range_sum(kCountA), "AC-2 connection A aggregate", result_a);
  });
  std::thread thread_b;
  try {
    thread_b = std::thread([&]() {
      worker(b,
             "SELECT sum(i) FROM gpu_b;",
             range_sum(kCountB),
             "AC-2 connection B aggregate",
             result_b);
    });
  } catch (std::exception const& error) {
    start.store(true, std::memory_order_release);
    thread_a.join();
    out.error = std::string{"AC-2 could not start connection B worker: "} + error.what();
    return;
  } catch (...) {
    start.store(true, std::memory_order_release);
    thread_a.join();
    out.error = "AC-2 could not start connection B worker";
    return;
  }

  auto const both_ready = wait_for_workers(ready, 2);
  mark_workload_started(output_path, out);
  start.store(true, std::memory_order_release);
  thread_a.join();
  thread_b.join();

  if (!both_ready) {
    out.error = "AC-2 workers did not reach the shared start gate";
    return;
  }
  if (!result_a.error.empty()) {
    out.error = result_a.error;
    return;
  }
  if (!result_b.error.empty()) {
    out.error = result_b.error;
    return;
  }
  auto const stats_after = sirius_context->get_transparent_execution_stats();
  if (!require_transparent_execution_delta(
        stats_before, stats_after, 2, "AC-2 concurrent aggregates", out.error)) {
    return;
  }
  out.b_completed = true;
}

constexpr std::uint64_t kAc3CpuCount = 1000000;

bool run_ac3_attempt(duckdb::Connection& a,
                     duckdb::Connection& b,
                     duckdb::shared_ptr<duckdb::SiriusContext> const& sirius_context,
                     std::uint64_t large_count,
                     fs::path const& output_path,
                     slot_watchdog_result& out)
{
  struct release_window_hold {
    duckdb::SiriusContext& context;
    bool released = false;
    void release() noexcept
    {
      if (released) { return; }
      context.release_test_window_hold();
      released = true;
    }
    ~release_window_hold() { release(); }
  } hold{*sirius_context};

  auto const stats_before = sirius_context->get_transparent_execution_stats();
  async_query_result gpu_result;
  std::thread gpu_thread(run_async_scalar_query,
                         std::ref(a),
                         "SELECT sum(i) FROM gpu_large;",
                         range_sum(large_count),
                         "AC-3 large GPU aggregate",
                         std::ref(gpu_result));

  auto const start_deadline  = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  bool gpu_execution_started = false;
  while (std::chrono::steady_clock::now() < start_deadline) {
    auto const stats = sirius_context->get_transparent_execution_stats();
    if (stats.executions >= stats_before.executions + 1 &&
        sirius_context->is_query_lifecycle_active()) {
      gpu_execution_started = true;
      break;
    }
    if (gpu_result.completed.load(std::memory_order_acquire)) { break; }
    std::this_thread::yield();
  }

  if (!gpu_execution_started || gpu_result.completed.load(std::memory_order_acquire)) {
    hold.release();
    gpu_thread.join();
    if (!gpu_result.error.empty()) {
      out.error = gpu_result.error;
      return false;
    }
    if (!gpu_execution_started) {
      auto const stats_after = sirius_context->get_transparent_execution_stats();
      if (!require_transparent_execution_delta(
            stats_before, stats_after, 1, "AC-3 large GPU aggregate", out.error)) {
        return false;
      }
      out.error = "AC-3 GPU execution did not start within 30s";
    } else {
      out.error =
        "AC-3 GPU execution completed before the CPU probe could start despite the "
        "deterministic window hold";
    }
    return false;
  }

  std::string cpu_error;
  bool cpu_ok = false;
  try {
    mark_workload_started(output_path, out);
    cpu_ok = run_scalar_query(
      b, "SELECT sum(i) FROM cpu_small;", range_sum(kAc3CpuCount), "AC-3 CPU probe", cpu_error);
  } catch (std::exception const& error) {
    cpu_error = std::string{"AC-3 CPU probe threw: "} + error.what();
  } catch (...) {
    cpu_error = "AC-3 CPU probe threw an unknown exception";
  }
  auto const gpu_completed_before_release = gpu_result.completed.load(std::memory_order_acquire);
  hold.release();
  gpu_thread.join();

  if (!gpu_result.error.empty()) {
    out.error = gpu_result.error;
    return false;
  }
  if (!cpu_ok) {
    out.error = cpu_error;
    return false;
  }
  auto const stats_after = sirius_context->get_transparent_execution_stats();
  if (!require_transparent_execution_delta(
        stats_before, stats_after, 1, "AC-3 large GPU aggregate", out.error)) {
    return false;
  }
  if (gpu_completed_before_release) {
    out.error = "AC-3 CPU probe did not complete before the held GPU window was released";
    return false;
  }
  return true;
}

void run_ac3_cpu_bypasses_gpu(duckdb::Connection& a,
                              duckdb::Connection& b,
                              fs::path const& output_path,
                              slot_watchdog_result& out)
{
  constexpr std::uint64_t kGpuCount = 20000000;
  if (!set_gpu_execution(a, false, out.error) || !set_gpu_execution(b, false, out.error) ||
      !create_range_table(a, "cpu_small", kAc3CpuCount, out.error) ||
      !create_range_table(a, "gpu_large", kGpuCount, out.error) ||
      !run_statement(a, "CHECKPOINT;", "CHECKPOINT", out.error) ||
      !set_gpu_execution(a, true, out.error)) {
    return;
  }

  auto sirius_context = sirius::test::get_registered_sirius_context(a);
  sirius_context->arm_test_window_hold();

  if (run_ac3_attempt(a, b, sirius_context, kGpuCount, output_path, out)) {
    out.b_completed = true;
  }
}

void run_ac4_concurrent_planning(duckdb::Connection& a,
                                 duckdb::Connection& b,
                                 fs::path const& output_path,
                                 slot_watchdog_result& out)
{
  constexpr std::uint64_t kCountA = 10000;
  constexpr std::uint64_t kCountB = 12000;
  if (!set_gpu_execution(a, false, out.error) || !set_gpu_execution(b, false, out.error) ||
      !create_range_table(a, "plan_a", kCountA, out.error) ||
      !create_range_table(a, "plan_b", kCountB, out.error) ||
      !run_statement(a, "CHECKPOINT;", "CHECKPOINT", out.error) ||
      !set_gpu_execution(a, true, out.error) || !set_gpu_execution(b, true, out.error)) {
    return;
  }

  std::atomic<unsigned> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> abort{false};
  async_query_result result_a;
  async_query_result result_b;

  auto worker = [&](duckdb::Connection& connection,
                    std::string const& table_name,
                    std::uint64_t base_sum,
                    std::uint64_t multiplier,
                    std::uint64_t offset,
                    char const* prepare_operation,
                    char const* execute_operation,
                    async_query_result& result) {
    ready.fetch_add(1, std::memory_order_release);
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    try {
      for (std::uint64_t iteration = 0; iteration < 8; ++iteration) {
        if (abort.load(std::memory_order_acquire)) { break; }
        auto const sql = "SELECT sum(i * " + std::to_string(multiplier) + ") + " +
                         std::to_string(offset + iteration) + " FROM " + table_name + ";";
        auto prepared = connection.Prepare(sql);
        if (!require_success(prepared.get(), prepare_operation, result.error)) {
          abort.store(true, std::memory_order_release);
          break;
        }
        auto const expected = std::to_string(base_sum * multiplier + offset + iteration);
        if (!run_prepared_scalar(*prepared, expected, execute_operation, result.error)) {
          abort.store(true, std::memory_order_release);
          break;
        }
      }
    } catch (std::exception const& error) {
      result.error = std::string(execute_operation) + " threw: " + error.what();
      abort.store(true, std::memory_order_release);
    } catch (...) {
      result.error = std::string(execute_operation) + " threw an unknown exception";
      abort.store(true, std::memory_order_release);
    }
    result.completed_at = std::chrono::steady_clock::now();
    result.completed.store(true, std::memory_order_release);
  };

  std::thread thread_a([&]() {
    worker(a,
           "plan_a",
           std::stoull(range_sum(kCountA)),
           1,
           0,
           "AC-4 connection A Prepare",
           "AC-4 connection A Execute",
           result_a);
  });
  std::thread thread_b;
  try {
    thread_b = std::thread([&]() {
      worker(b,
             "plan_b",
             std::stoull(range_sum(kCountB)),
             2,
             100,
             "AC-4 connection B Prepare",
             "AC-4 connection B Execute",
             result_b);
    });
  } catch (std::exception const& error) {
    abort.store(true, std::memory_order_release);
    start.store(true, std::memory_order_release);
    thread_a.join();
    out.error = std::string{"AC-4 could not start connection B worker: "} + error.what();
    return;
  } catch (...) {
    abort.store(true, std::memory_order_release);
    start.store(true, std::memory_order_release);
    thread_a.join();
    out.error = "AC-4 could not start connection B worker";
    return;
  }

  auto const both_ready = wait_for_workers(ready, 2);
  mark_workload_started(output_path, out);
  start.store(true, std::memory_order_release);
  thread_a.join();
  thread_b.join();

  if (!both_ready) {
    out.error = "AC-4 workers did not reach the shared start gate";
    return;
  }
  if (!result_a.error.empty()) {
    out.error = result_a.error;
    return;
  }
  if (!result_b.error.empty()) {
    out.error = result_b.error;
    return;
  }
  out.b_completed = true;
}

void run_ac5_explicit_reexecution(duckdb::Connection& connection,
                                  fs::path const& output_path,
                                  slot_watchdog_result& out)
{
  constexpr std::uint64_t kCount = 200000;
  if (!set_gpu_execution(connection, false, out.error) ||
      !create_range_table(connection, "explicit_t", kCount, out.error) ||
      !run_statement(connection, "CHECKPOINT;", "CHECKPOINT", out.error) ||
      !run_statement(connection,
                     "SET enable_duckdb_fallback=false;",
                     "SET enable_duckdb_fallback=false",
                     out.error) ||
      !run_statement(connection,
                     "CALL pin_table(format='duckdb', name='explicit_t', tier='gpu');",
                     "AC-5 pin_table",
                     out.error)) {
    return;
  }

  auto prepared =
    connection.Prepare("SELECT * FROM gpu_execution('SELECT sum(i) FROM explicit_t;');");
  if (!require_success(prepared.get(), "AC-5 explicit Prepare", out)) {
    (void)connection.Query("CALL unpin_table('explicit_t');");
    return;
  }

  mark_workload_started(output_path, out);
  auto const expected = range_sum(kCount);
  if (!run_prepared_scalar(*prepared, expected, "AC-5 first Execute", out.error)) {
    (void)connection.Query("CALL unpin_table('explicit_t');");
    return;
  }
  if (!run_statement(
        connection, "CALL unpin_table('explicit_t');", "AC-5 unpin_table", out.error)) {
    return;
  }
  if (!run_prepared_scalar(*prepared, expected, "AC-5 second Execute", out.error)) { return; }
  out.b_completed = true;
}

void run_ac6_capture_generation(duckdb::Connection& connection,
                                fs::path const& output_path,
                                slot_watchdog_result& out)
{
  if (!set_gpu_execution(connection, false, out.error) ||
      !run_statement(connection, "CREATE SCHEMA old_scope;", "CREATE old_scope", out.error) ||
      !run_statement(connection, "CREATE SCHEMA new_scope;", "CREATE new_scope", out.error) ||
      !create_range_table(connection, "old_scope.t", 3, out.error) ||
      !create_range_table(connection, "new_scope.t", 7, out.error) ||
      !run_statement(connection, "CHECKPOINT;", "CHECKPOINT", out.error) ||
      !run_statement(
        connection, "SET search_path='old_scope';", "SET old search_path", out.error) ||
      !run_statement(connection,
                     "SET enable_duckdb_fallback=false;",
                     "SET enable_duckdb_fallback=false",
                     out.error) ||
      !set_gpu_execution(connection, true, out.error) ||
      !run_statement(connection, "BEGIN TRANSACTION;", "BEGIN TRANSACTION", out.error)) {
    return;
  }

  {
    auto old_plan = connection.ExtractPlan("SELECT count(*) FROM t;");
    if (old_plan == nullptr) {
      out.error = "AC-6 ExtractPlan returned nullptr";
      return;
    }
  }

  // Change binding context without running an intervening statement: the stale
  // ExtractPlan capture must be rejected specifically by the next Prepare's
  // planning generation, not incidentally cleared by another QueryBegin.
  connection.context->client_data->catalog_search_path->Set(
    duckdb::CatalogSearchEntry::Parse("new_scope"), duckdb::CatalogSetPathType::SET_SCHEMAS);
  auto& client_config            = duckdb::ClientConfig::GetConfig(*connection.context);
  client_config.enable_optimizer = false;

  auto const stats_before_prepare = sirius::test::get_transparent_execution_stats(connection);
  mark_workload_started(output_path, out);
  auto prepared = connection.Prepare("SELECT count(*) FROM t;");
  if (!require_success(prepared.get(), "AC-6 Prepare", out)) {
    client_config.enable_optimizer = true;
    (void)connection.Query("ROLLBACK;");
    return;
  }
  auto const stats_after_prepare = sirius::test::get_transparent_execution_stats(connection);
  if (stats_after_prepare.successful_rebinds != stats_before_prepare.successful_rebinds) {
    client_config.enable_optimizer = true;
    (void)connection.Query("ROLLBACK;");
    out.error = "AC-6 first Prepare changed successful_rebinds from " +
                std::to_string(stats_before_prepare.successful_rebinds) + " to " +
                std::to_string(stats_after_prepare.successful_rebinds);
    return;
  }

  bool execute_ok = false;
  try {
    execute_ok = run_prepared_scalar(*prepared, "7", "AC-6 Execute", out.error);
  } catch (...) {
    client_config.enable_optimizer = true;
    throw;
  }
  client_config.enable_optimizer = true;
  if (!execute_ok) {
    (void)connection.Query("ROLLBACK;");
    return;
  }

  if (!run_statement(connection, "COMMIT;", "COMMIT", out.error)) { return; }
  out.b_completed = true;
}

void run_ac7_load_union(fs::path const& database_path,
                        fs::path const& output_path,
                        slot_watchdog_result& out)
{
  duckdb::DBConfig config;
  config.options.load_extensions = false;
  duckdb::DuckDB db(database_path.string(), &config);
  duckdb::Connection connection(db);

  if (!run_statement(connection,
                     "SET disabled_optimizers='filter_pushdown';",
                     "AC-7 user disabled_optimizers SET",
                     out.error)) {
    return;
  }
  mark_workload_started(output_path, out);
  db.LoadStaticExtension<duckdb::SiriusExtension>();

  duckdb::Value current_setting;
  auto lookup = connection.context->TryGetCurrentSetting("disabled_optimizers", current_setting);
  if (!lookup || current_setting.IsNull()) {
    out.error = "AC-7 could not read disabled_optimizers immediately after LOAD";
    return;
  }
  auto const actual                             = current_setting.ToString();
  constexpr std::array<char const*, 4> expected = {
    "filter_pushdown", "in_clause", "compressed_materialization", "late_materialization"};
  for (auto const* optimizer : expected) {
    if (actual.find(optimizer) == std::string::npos) {
      out.error = "AC-7 disabled_optimizers is missing " + std::string(optimizer) + ": " + actual;
      return;
    }
  }
  out.b_completed = true;
}

// The scenario for one variant. Runs inside the existing child process and
// reports progress through the existing watchdog result file.
slot_watchdog_result run_scenario(std::string const& variant,
                                  fs::path const& database_path,
                                  fs::path const& output_path)
{
  slot_watchdog_result out;
  try {
    if (variant == "wave1_ac7_load_union") {
      run_ac7_load_union(database_path, output_path, out);
      return out;
    }

    duckdb::DuckDB db(database_path.string());
    duckdb::Connection a(db);
    duckdb::Connection b(db);

    if (variant == "stream" || variant == "pending" || variant == "cached_no_rebind" ||
        variant == "wave1_ac1_pending_gpu") {
      run_abandoned_result_scenario(variant, a, b, output_path, out);
    } else if (variant == "wave1_ac2_gpu_single_flight") {
      run_ac2_gpu_single_flight(a, b, output_path, out);
    } else if (variant == "wave1_ac3_cpu_bypasses_gpu") {
      run_ac3_cpu_bypasses_gpu(a, b, output_path, out);
    } else if (variant == "wave1_ac4_concurrent_planning") {
      run_ac4_concurrent_planning(a, b, output_path, out);
    } else if (variant == "wave1_ac5_explicit_reexecution") {
      run_ac5_explicit_reexecution(a, output_path, out);
    } else if (variant == "wave1_ac6_capture_generation") {
      run_ac6_capture_generation(a, output_path, out);
    } else {
      out.error = "unknown variant: " + variant;
    }
  } catch (std::exception const& e) {
    out.error = e.what();
  } catch (...) {
    out.error = "scenario threw an unknown exception";
  }
  return out;
}

}  // namespace

// Hidden child-runner: re-entered via execl by the watchdog below. Not part of any
// standard gate; runs only when selected by its exact name.
TEST_CASE(kChildRunnerCase, "[.][query_lifecycle][watchdog_child]")
{
  auto const* variant_raw = std::getenv(kEnvVariant);
  auto const* output_raw  = std::getenv(kEnvOutput);
  auto const* config_raw  = std::getenv(kEnvConfig);
  if (variant_raw == nullptr || output_raw == nullptr) { return; }

  slot_watchdog_result out;
  if (config_raw == nullptr) {
    out.error = "watchdog child missing config path";
  } else {
    ::setenv("SIRIUS_CONFIG_FILE", config_raw, 1);
    ::unsetenv("SIRIUS_DISABLE");
    auto const output_path = fs::path(output_raw);
    auto const database_path =
      output_path.parent_path() / ("scenario_" + std::string(variant_raw) + ".duckdb");
    out = run_scenario(variant_raw, database_path, output_path);
  }
  write_result(output_raw, out);
}

class QueryLifecycleSlotFixture {
 public:
  QueryLifecycleSlotFixture()
  {
    static std::atomic<std::uint64_t> next_id{0};
    work_dir =
      fs::temp_directory_path() / ("sirius_slot_leak_" + std::to_string(next_id.fetch_add(1)));
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    config_path =
      fs::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" / "integration.yaml";
    REQUIRE(fs::exists(config_path));
  }

  ~QueryLifecycleSlotFixture() { fs::remove_all(work_dir); }

  // Runs one variant in a child process guarded by a bounded deadline. If the
  // child does not complete in time, the child process is ended and the result is
  // marked timed_out (the reproduced wait).
  slot_watchdog_result run_variant(std::string const& variant, std::chrono::seconds deadline)
  {
    static std::atomic<std::uint64_t> next_child{0};
    auto const output_path =
      work_dir / ("result_" + variant + "_" + std::to_string(next_child.fetch_add(1)) + ".txt");

    auto const pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      ::setenv(kEnvVariant, variant.c_str(), 1);
      ::setenv(kEnvOutput, output_path.string().c_str(), 1);
      ::setenv(kEnvConfig, config_path.string().c_str(), 1);
      // A child that inherited a live CUDA context must re-exec before running an
      // engine query, so re-invoke the unit-test binary for the child-runner case.
      ::execl("/proc/self/exe", "sirius_unittest", kChildRunnerCase, static_cast<char*>(nullptr));
      ::_exit(127);
    }

    int status      = 0;
    auto const stop = std::chrono::steady_clock::now() + deadline;
    while (std::chrono::steady_clock::now() < stop) {
      auto const waited = ::waitpid(pid, &status, WNOHANG);
      if (waited == pid) {
        auto result = read_result(output_path);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
          if (result.error.empty()) { result.error = "watchdog child exited abnormally"; }
        }
        return result;
      }
      if (waited < 0) {
        if (errno == EINTR) { continue; }
        (void)::kill(pid, SIGKILL);
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        slot_watchdog_result result;
        result.error = "waitpid failed while waiting for watchdog child";
        return result;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }

    // Deadline elapsed: the child is still waiting on the follow-up query. End it
    // and report the reproduced wait.
    (void)::kill(pid, SIGKILL);
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    auto out      = read_result(output_path);
    out.timed_out = true;
    if (out.b_started) {
      out.error = "variant " + variant + " did not complete within " +
                  std::to_string(deadline.count()) + "s after its observed workload started";
    } else if (out.error.empty()) {
      out.error = "variant " + variant + " timed out during setup";
    }
    return out;
  }

  void require_variant_succeeds(std::string const& variant,
                                std::chrono::seconds deadline = std::chrono::seconds{60})
  {
    auto result = run_variant(variant, deadline);
    INFO("variant: " << variant);
    INFO("error: " << result.error);
    REQUIRE(result.b_started);
    REQUIRE_FALSE(result.timed_out);
    REQUIRE(result.error.empty());
    REQUIRE(result.b_completed);
  }

  void require_b_completes(std::string const& variant) { require_variant_succeeds(variant); }

  fs::path work_dir;
  fs::path config_path;
};

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "query lifecycle slot is released for an unconsumed result",
                 "[query_lifecycle][slot_leak]")
{
  SECTION("unconsumed streaming result (cross-connection)") { require_b_completes("stream"); }
  SECTION("unexecuted pending result") { require_b_completes("pending"); }
  SECTION("cached prepared, no rebind") { require_b_completes("cached_no_rebind"); }
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "wave-1 AC-1: pending result releases the slot before a GPU follow-up",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("wave1_ac1_pending_gpu");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "wave-1 AC-2: concurrent transparent GPU queries both complete correctly",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("wave1_ac2_gpu_single_flight");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "wave-1 AC-3: a CPU query completes before an in-flight GPU query",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("wave1_ac3_cpu_bypasses_gpu", std::chrono::seconds{240});
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "wave-1 AC-4: concurrent prepared planning preserves query ownership",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("wave1_ac4_concurrent_planning");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "wave-1 AC-5: explicit prepared execution observes a pin-state change",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("wave1_ac5_explicit_reexecution");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "wave-1 AC-6: stale capture is not consumed after a planning generation change",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("wave1_ac6_capture_generation");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "wave-1 AC-7: extension load unions optimizer masks with the user setting",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("wave1_ac7_load_union");
}
