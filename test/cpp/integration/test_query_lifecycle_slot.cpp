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

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/pending_query_result.hpp>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utils/transparent_execution_test_utils.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
  bool b_started   = false;  // the child reached connection B's follow-up query
  bool b_completed = false;  // connection B's follow-up query returned
};

template <typename T>
bool require_success(T* result, char const* operation, slot_watchdog_result& out)
{
  if (result == nullptr) {
    out.error = std::string(operation) + " returned nullptr";
    return false;
  }
  if (result->HasError()) {
    out.error = std::string(operation) + " failed: " + result->GetError();
    return false;
  }
  return true;
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

// ---------------------------------------------------------------------------
// Concurrent-planning wrong-results repro (issue #1294, problem 2).
//
// The transparent capture is a single un-owned slot on the shared
// SiriusContext, and plain Prepare() runs outside the query-lifecycle lock, so
// two connections planning at the same time can interleave set/take and one
// connection's prepared statement executes the OTHER connection's plan. Each
// worker below prepares and executes queries whose scalar results are unique
// to its own table and expression, so a cross-consumed plan shows up as a
// wrong value (some runs surface the same interleaving as a lock-ordering
// stall instead, which the parent's deadline catches).
// ---------------------------------------------------------------------------
std::uint64_t range_sum(std::uint64_t count)
{
  return count % 2 == 0 ? (count / 2) * (count - 1) : count * ((count - 1) / 2);
}

struct concurrent_worker_result {
  std::string error;
};

void run_prepared_loop(duckdb::Connection& connection,
                       std::string const& table_name,
                       std::uint64_t base_sum,
                       std::uint64_t multiplier,
                       std::uint64_t offset,
                       std::atomic<bool>& start,
                       std::atomic<bool>& abort_flag,
                       concurrent_worker_result& out)
{
  while (!start.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  try {
    for (std::uint64_t iteration = 0; iteration < 8; ++iteration) {
      if (abort_flag.load(std::memory_order_acquire)) { break; }
      auto const sql = "SELECT sum(i * " + std::to_string(multiplier) + ") + " +
                       std::to_string(offset + iteration) + " FROM " + table_name + ";";
      auto prepared = connection.Prepare(sql);
      if (!prepared || prepared->HasError()) {
        out.error = "Prepare failed: " + (prepared ? prepared->GetError() : "null result");
        break;
      }
      duckdb::vector<duckdb::Value> parameters;
      auto result = prepared->Execute(parameters, /*allow_stream_result=*/false);
      if (!result || result->HasError()) {
        out.error = "Execute failed: " + (result ? result->GetError() : "null result");
        break;
      }
      auto chunk = result->Fetch();
      if (!chunk || chunk->size() != 1 || chunk->ColumnCount() != 1) {
        out.error = sql + " did not return exactly one scalar row";
        break;
      }
      auto const expected = std::to_string(base_sum * multiplier + offset + iteration);
      auto const actual   = chunk->GetValue(0, 0).ToString();
      if (actual != expected) {
        out.error = sql + " returned " + actual + ", expected " + expected;
        break;
      }
    }
  } catch (std::exception const& e) {
    out.error = e.what();
  } catch (...) {
    out.error = "worker threw an unknown exception";
  }
  if (!out.error.empty()) { abort_flag.store(true, std::memory_order_release); }
}

// The A/B scenario for one variant. Runs inside the child process. Connection A
// leaves a result unconsumed; connection B then issues a follow-up query that, on
// the unfixed engine, waits (so this function does not return and the parent's
// deadline elapses). Returns normally only once B completes.
slot_watchdog_result run_scenario(std::string const& variant,
                                  fs::path const& database_path,
                                  fs::path const& output_path)
{
  slot_watchdog_result out;
  try {
    duckdb::DuckDB db(database_path.string());
    duckdb::Connection a(db);
    duckdb::Connection b(db);

    auto b_cpu = b.Query("SET gpu_execution=false;");
    if (!require_success(b_cpu.get(), "connection B SET gpu_execution=false", out)) { return out; }

    auto a_cpu = a.Query("SET gpu_execution=false;");
    if (!require_success(a_cpu.get(), "connection A SET gpu_execution=false", out)) { return out; }
    auto create = a.Query("CREATE TABLE t AS SELECT range AS i FROM range(200000);");
    if (!require_success(create.get(), "CREATE TABLE", out)) { return out; }
    auto checkpoint = a.Query("CHECKPOINT;");
    if (!require_success(checkpoint.get(), "CHECKPOINT", out)) { return out; }

    // Connection A establishes and then abandons a result, per the variant.
    std::unique_ptr<duckdb::PreparedStatement> prepared;
    std::unique_ptr<duckdb::QueryResult> streamed;
    std::unique_ptr<duckdb::PendingQueryResult> pending;

    if (variant == "stream") {
      auto set_gpu = a.Query("SET gpu_execution=true;");
      if (!require_success(set_gpu.get(), "connection A SET gpu_execution=true", out)) {
        return out;
      }
      auto const stats_before = sirius::test::get_transparent_execution_stats(a);
      prepared                = a.Prepare("SELECT i FROM t;");
      if (!require_success(prepared.get(), "stream Prepare", out)) { return out; }
      streamed = prepared->Execute();  // streaming by default
      if (!require_success(streamed.get(), "stream Execute", out)) { return out; }
      if (streamed->type != duckdb::QueryResultType::STREAM_RESULT) {
        out.error = "stream Execute did not return a streaming result";
        return out;
      }
      auto chunk = streamed->Fetch();
      if (!chunk || chunk->size() == 0) {
        out.error = "stream Execute produced no first chunk";
        return out;
      }
      auto const stats_after = sirius::test::get_transparent_execution_stats(a);
      if (stats_after.executions != stats_before.executions + 1 ||
          stats_after.fallbacks != stats_before.fallbacks ||
          stats_after.runtime_fallbacks != stats_before.runtime_fallbacks) {
        out.error = "stream query did not execute exactly once on GPU without fallback";
        return out;
      }
    } else if (variant == "pending") {
      auto set_gpu = a.Query("SET gpu_execution=true;");
      if (!require_success(set_gpu.get(), "connection A SET gpu_execution=true", out)) {
        return out;
      }
      prepared = a.Prepare("SELECT i FROM t;");
      if (!require_success(prepared.get(), "pending Prepare", out)) { return out; }
      pending = prepared->PendingQuery();  // created, never executed to a result
      if (!require_success(pending.get(), "prepared PendingQuery", out)) { return out; }
    } else if (variant == "cached_no_rebind") {
      // A plain CPU prepared statement keeps DuckDB on the cached DO_NOT_REBIND
      // execute path (no re-plan), which does not call OnFinalizePrepare.
      prepared = a.Prepare("SELECT i FROM t;");
      if (!require_success(prepared.get(), "cached Prepare", out)) { return out; }
      {
        auto first = prepared->Execute();
        if (!require_success(first.get(), "cached first Execute", out)) { return out; }
        if (first->type != duckdb::QueryResultType::STREAM_RESULT) {
          out.error = "cached first Execute did not return a streaming result";
          return out;
        }
        while (auto chunk = first->Fetch()) {
          (void)chunk;
        }
        if (first->HasError()) {
          out.error = "cached first Execute failed while draining: " + first->GetError();
          return out;
        }
      }
      streamed = prepared->Execute();  // second execution: cached, no rebind
      if (!require_success(streamed.get(), "cached second Execute", out)) { return out; }
      if (streamed->type != duckdb::QueryResultType::STREAM_RESULT) {
        out.error = "cached second Execute did not return a streaming result";
        return out;
      }
      auto chunk = streamed->Fetch();
      if (!chunk || chunk->size() == 0) {
        out.error = "cached second Execute produced no first chunk";
        return out;
      }
    } else if (variant == "concurrent_planning") {
      if (!require_success(
            a.Query("SET gpu_execution=true;").get(), "connection A SET gpu_execution=true", out)) {
        return out;
      }
      if (!require_success(
            b.Query("SET gpu_execution=true;").get(), "connection B SET gpu_execution=true", out)) {
        return out;
      }
      constexpr std::uint64_t kCountA = 10000;
      constexpr std::uint64_t kCountB = 12000;
      if (!require_success(a.Query("CREATE TABLE plan_a AS SELECT range AS i FROM range(" +
                                   std::to_string(kCountA) + ");")
                             .get(),
                           "CREATE plan_a",
                           out)) {
        return out;
      }
      if (!require_success(a.Query("CREATE TABLE plan_b AS SELECT range AS i FROM range(" +
                                   std::to_string(kCountB) + ");")
                             .get(),
                           "CREATE plan_b",
                           out)) {
        return out;
      }
      if (!require_success(a.Query("CHECKPOINT;").get(), "CHECKPOINT plan tables", out)) {
        return out;
      }

      out.b_started = true;
      write_result(output_path, out);

      std::atomic<bool> start{false};
      std::atomic<bool> abort_flag{false};
      concurrent_worker_result result_a;
      concurrent_worker_result result_b;
      std::thread thread_a([&]() {
        run_prepared_loop(a, "plan_a", range_sum(kCountA), 1, 0, start, abort_flag, result_a);
      });
      std::thread thread_b([&]() {
        run_prepared_loop(b, "plan_b", range_sum(kCountB), 2, 100, start, abort_flag, result_b);
      });
      start.store(true, std::memory_order_release);
      thread_a.join();
      thread_b.join();

      if (!result_a.error.empty()) {
        out.error = "connection A worker: " + result_a.error;
        return out;
      }
      if (!result_b.error.empty()) {
        out.error = "connection B worker: " + result_b.error;
        return out;
      }
      out.b_completed = true;
      return out;
    } else {
      out.error = "unknown variant: " + variant;
      return out;
    }

    // Persist this before entering B so the watchdog can distinguish the expected
    // blocked B query from a setup or connection-A hang.
    out.b_started = true;
    write_result(output_path, out);

    auto rb = b.Query("SELECT 42;");
    if (!require_success(rb.get(), "connection B SELECT 42", out)) { return out; }
    out.b_completed = true;
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
      out.error = "connection B did not complete within " + std::to_string(deadline.count()) +
                  "s (query-lifecycle lock not released)";
    } else if (out.error.empty()) {
      out.error = "watchdog child timed out before starting connection B's query";
    }
    return out;
  }

  void require_b_completes(std::string const& variant)
  {
    auto result = run_variant(variant, std::chrono::seconds{60});
    INFO("variant: " << variant);
    INFO("error: " << result.error);
    // The fix makes connection B complete; the unfixed engine leaves it waiting.
    REQUIRE(result.b_started);
    REQUIRE_FALSE(result.timed_out);
    REQUIRE(result.error.empty());
    REQUIRE(result.b_completed);
  }

  fs::path work_dir;
  fs::path config_path;
};

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "query lifecycle slot is released for an unconsumed result",
                 "[.][query_lifecycle][slot_leak]")
{
  SECTION("unconsumed streaming result (cross-connection)") { require_b_completes("stream"); }
  SECTION("unexecuted pending result") { require_b_completes("pending"); }
  SECTION("cached prepared, no rebind") { require_b_completes("cached_no_rebind"); }
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "concurrent prepared planning returns each connection's own result",
                 "[.][query_lifecycle][slot_leak]")
{
  require_b_completes("concurrent_planning");
}
