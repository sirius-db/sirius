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
 * @brief Query-lifecycle slot regression and concurrency coverage.
 *
 * Historically, Sirius acquired a per-DatabaseInstance query-lifecycle lock in
 * QueryBegin and released it in QueryEnd. DuckDB does not call QueryEnd for an
 * unconsumed streaming or pending prepared-statement result, so abandoning one
 * could leave the slot occupied and prevent another connection from proceeding.
 *
 * The slot is now scoped to engine-owned planning and execution windows. This
 * suite retains the abandoned-result regression and covers lifecycle concurrency,
 * cancellation, runtime-health, operator-id and keyed-log behavior. Scenarios run
 * in isolated child processes with bounded deadlines.
 */

#include "log/logging.hpp"
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

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <source_location>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr char const* kChildRunnerCase = "query lifecycle slot watchdog child runner";
constexpr char const* kEnvVariant      = "SIRIUS_SLOT_WATCHDOG_VARIANT";
constexpr char const* kEnvOutput       = "SIRIUS_SLOT_WATCHDOG_OUTPUT";
constexpr char const* kEnvConfig       = "SIRIUS_SLOT_WATCHDOG_CONFIG";

static_assert(SIRIUS_ACTIVE_LOG_LEVEL <= SIRIUS_LOG_LEVEL_DEBUG,
              "query-lifecycle window tests require DEBUG logging");

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
                      std::string& error)
{
  auto result = connection.Query(sql);
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

bool run_scalar_query_capture(duckdb::Connection& connection,
                              std::string const& sql,
                              char const* operation,
                              std::string& actual,
                              std::string& error)
{
  auto result = connection.Query(sql);
  if (!require_success(result.get(), operation, error)) { return false; }

  auto chunk = result->Fetch();
  if (chunk == nullptr || chunk->size() != 1 || chunk->ColumnCount() != 1) {
    error = std::string(operation) + " did not return exactly one scalar row";
    return false;
  }
  actual = chunk->GetValue(0, 0).ToString();
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
  return true;
}

std::string ascii_lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

bool contains_case_insensitive(std::string const& value, std::string const& needle)
{
  return ascii_lower(value).find(ascii_lower(needle)) != std::string::npos;
}

bool run_query_expect_error(duckdb::Connection& connection,
                            std::string const& sql,
                            char const* operation,
                            std::string const& expected_substring,
                            std::string& error)
{
  std::string actual;
  try {
    auto result = connection.Query(sql);
    if (result == nullptr) {
      error = std::string(operation) + " returned nullptr";
      return false;
    }
    if (!result->HasError()) {
      error = std::string(operation) + " unexpectedly succeeded";
      return false;
    }
    actual = result->GetError();
  } catch (std::exception const& exception) {
    actual = exception.what();
  } catch (...) {
    error = std::string(operation) + " threw an unknown exception";
    return false;
  }

  if (!contains_case_insensitive(actual, expected_substring)) {
    error = std::string(operation) + " returned an unexpected error: " + actual;
    return false;
  }
  return true;
}

std::string range_sum(std::uint64_t count)
{
  auto const sum = count % 2 == 0 ? (count / 2) * (count - 1) : count * ((count - 1) / 2);
  return std::to_string(sum);
}

std::string sql_quote(std::string value)
{
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('\'');
  for (auto const ch : value) {
    if (ch == '\'') { quoted.push_back('\''); }
    quoted.push_back(ch);
  }
  quoted.push_back('\'');
  return quoted;
}

template <typename Predicate>
bool wait_until(Predicate&& predicate, std::chrono::steady_clock::duration timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return true;
}

struct async_query_result {
  std::string error;
  std::atomic<bool> completed{false};
};

void run_async_scalar_query(duckdb::Connection& connection,
                            std::string const& sql,
                            std::string const& expected,
                            char const* operation,
                            async_query_result& out)
{
  try {
    (void)run_scalar_query(connection, sql, expected, operation, out.error);
  } catch (std::exception const& error) {
    out.error = std::string(operation) + " threw: " + error.what();
  } catch (...) {
    out.error = std::string(operation) + " threw an unknown exception";
  }
  out.completed.store(true, std::memory_order_release);
}

void run_async_pending_scalar(duckdb::PendingQueryResult& pending,
                              std::string const& expected,
                              char const* operation,
                              async_query_result& out)
{
  try {
    auto result = pending.Execute();
    (void)require_scalar_result(result.get(), operation, expected, out.error);
  } catch (std::exception const& error) {
    out.error = std::string(operation) + " threw: " + error.what();
  } catch (...) {
    out.error = std::string(operation) + " threw an unknown exception";
  }
  out.completed.store(true, std::memory_order_release);
}

class blocking_window_log_sink final : public sirius::log::sink {
 public:
  explicit blocking_window_log_sink(std::shared_ptr<sirius::log::sink> downstream)
    : downstream_(std::move(downstream))
  {
  }

  void set_level(sirius::log::level level) override { downstream_->set_level(level); }

  bool should_log(sirius::log::level level) const override
  {
    // The gate must still see INFO window events when the configured sink
    // filters them. Forwarding below continues to honor the configured level.
    return level == sirius::log::level::info || downstream_->should_log(level);
  }

  void log(sirius::log::level level,
           std::source_location const& location,
           std::string_view message) override
  {
    std::exception_ptr forwarding_error;
    try {
      if (downstream_->should_log(level)) { downstream_->log(level, location, message); }
    } catch (...) {
      forwarding_error = std::current_exception();
    }

    auto const is_window_begin =
      level == sirius::log::level::info && message.starts_with("[window] begin instance=");
    if (is_window_begin && armed_.exchange(false, std::memory_order_acq_rel)) {
      std::unique_lock lock(mutex_);
      claimed_ = true;
      if (!released_) {
        blocked_ = true;
        condition_.notify_all();
        if (!condition_.wait_for(lock, std::chrono::seconds{120}, [&]() { return released_; })) {
          timed_out_ = true;
          released_  = true;
        }
        blocked_ = false;
      }
      lock.unlock();
      condition_.notify_all();
    }

    if (forwarding_error) { std::rethrow_exception(forwarding_error); }
  }

  bool flush() override { return downstream_->flush(); }

  void arm()
  {
    std::lock_guard lock(mutex_);
    claimed_   = false;
    blocked_   = false;
    released_  = false;
    timed_out_ = false;
    armed_.store(true, std::memory_order_release);
  }

  void release() noexcept
  {
    armed_.store(false, std::memory_order_release);
    {
      std::lock_guard lock(mutex_);
      released_ = true;
    }
    condition_.notify_all();
  }

  bool wait_until_blocked(std::chrono::steady_clock::duration timeout)
  {
    std::unique_lock lock(mutex_);
    (void)condition_.wait_for(
      lock, timeout, [&]() { return blocked_ || timed_out_ || (claimed_ && released_); });
    return blocked_ && !timed_out_;
  }

  bool is_blocked() const
  {
    std::lock_guard lock(mutex_);
    return blocked_ && !timed_out_;
  }

  bool timed_out() const
  {
    std::lock_guard lock(mutex_);
    return timed_out_;
  }

 private:
  std::shared_ptr<sirius::log::sink> downstream_;
  std::atomic<bool> armed_{false};
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool claimed_   = false;
  bool blocked_   = false;
  bool released_  = true;
  bool timed_out_ = false;
};

class scoped_blocking_window_log_sink {
 public:
  scoped_blocking_window_log_sink()
    : downstream_(sirius::log::get_sink()),
      sink_(std::make_shared<blocking_window_log_sink>(downstream_))
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

  ~scoped_blocking_window_log_sink() noexcept
  {
    sink_->release();
    try {
      if (sirius::log::get_sink() == sink_) { sirius::log::set_sink(downstream_); }
    } catch (...) {
    }
  }

  scoped_blocking_window_log_sink(scoped_blocking_window_log_sink const&)            = delete;
  scoped_blocking_window_log_sink& operator=(scoped_blocking_window_log_sink const&) = delete;

  void arm() { sink_->arm(); }
  void release() noexcept { sink_->release(); }
  bool wait_until_blocked(std::chrono::steady_clock::duration timeout)
  {
    return sink_->wait_until_blocked(timeout);
  }
  bool is_blocked() const { return sink_->is_blocked(); }
  bool timed_out() const { return sink_->timed_out(); }

 private:
  std::shared_ptr<sirius::log::sink> downstream_;
  std::shared_ptr<blocking_window_log_sink> sink_;
};

struct held_window_threads {
  void arm() { window_hold.arm(); }
  bool wait_until_blocked(std::chrono::steady_clock::duration timeout)
  {
    return window_hold.wait_until_blocked(timeout);
  }
  bool is_blocked() const { return window_hold.is_blocked(); }
  bool timed_out() const { return window_hold.timed_out(); }

  void release() noexcept
  {
    if (released) { return; }
    window_hold.release();
    released = true;
  }

  void join_all()
  {
    for (auto& thread : threads) {
      if (thread.joinable()) { thread.join(); }
    }
  }

  ~held_window_threads()
  {
    release();
    join_all();
  }

  scoped_blocking_window_log_sink window_hold;
  std::vector<std::thread> threads;
  bool released = false;
};

bool variant_requires_file_log(std::string const& variant)
{
  return variant == "ac9_cancelled_waiter" || variant == "ac12_operator_ids" ||
         variant == "ac13_concurrent_logging";
}

fs::path variant_log_dir(fs::path const& output_path)
{
  return output_path.parent_path() / (output_path.stem().string() + "_logs");
}

bool read_variant_log_lines(std::vector<std::string>& lines, std::string& error)
{
  if (!sirius::log::get_sink()->flush()) {
    error = "could not flush the Sirius file log";
    return false;
  }
  auto const* log_dir_raw = std::getenv("SIRIUS_LOG_DIR");
  if (log_dir_raw == nullptr) {
    error = "logging variant is missing SIRIUS_LOG_DIR";
    return false;
  }

  std::vector<fs::path> log_files;
  for (auto const& entry : fs::directory_iterator(fs::path(log_dir_raw))) {
    if (entry.is_regular_file()) { log_files.push_back(entry.path()); }
  }
  std::sort(log_files.begin(), log_files.end());
  if (log_files.empty()) {
    error = "logging variant produced no regular log files";
    return false;
  }
  for (auto const& path : log_files) {
    std::ifstream input(path);
    if (!input) {
      error = "could not read Sirius log file: " + path.string();
      return false;
    }
    std::string line;
    while (std::getline(input, line)) {
      lines.push_back(line);
    }
  }
  return true;
}

std::size_t count_window_begins_for_sql_marker(std::vector<std::string> const& lines,
                                               std::string const& sql_marker,
                                               std::string& error)
{
  using query_key = std::tuple<std::string, std::uint64_t, std::uint64_t>;
  static std::regex const sql_re{
    R"(QueryBegin: instance=(\S+) connection=(\d+) query=(\d+) SQL: (.*)$)"};
  static std::regex const window_re{
    R"(\[window\] begin instance=(\S+) connection=(\d+) window=\d+ query=(\d+) outcome=-)"};

  std::set<query_key> keys;
  for (auto const& line : lines) {
    std::smatch match;
    if (std::regex_search(line, match, sql_re) &&
        match[4].str().find(sql_marker) != std::string::npos) {
      keys.emplace(match[1].str(), std::stoull(match[2].str()), std::stoull(match[3].str()));
    }
  }
  if (keys.empty()) {
    error = "log contains no keyed SQL line for " + sql_marker;
    return 0;
  }

  std::size_t count = 0;
  for (auto const& line : lines) {
    std::smatch match;
    if (!std::regex_search(line, match, window_re)) { continue; }
    auto const key =
      query_key{match[1].str(), std::stoull(match[2].str()), std::stoull(match[3].str())};
    if (keys.find(key) != keys.end()) { ++count; }
  }
  return count;
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
  auto const gpu_follow_up = variant == "ac1_pending_gpu";
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
  auto const stats_before = sirius_context->get_transparent_execution_stats();
  async_query_result gpu_result;
  held_window_threads workers;
  workers.arm();
  workers.threads.emplace_back(run_async_scalar_query,
                               std::ref(a),
                               "SELECT sum(i) FROM gpu_large;",
                               range_sum(large_count),
                               "AC-3 large GPU aggregate",
                               std::ref(gpu_result));

  auto const blocked               = workers.wait_until_blocked(std::chrono::seconds{30});
  auto const stats_at_block        = sirius_context->get_transparent_execution_stats();
  auto const gpu_execution_started = blocked &&
                                     stats_at_block.executions == stats_before.executions + 1 &&
                                     sirius_context->is_query_lifecycle_active() &&
                                     !gpu_result.completed.load(std::memory_order_acquire);

  if (!gpu_execution_started) {
    auto const hold_timed_out = workers.timed_out();
    workers.release();
    workers.join_all();
    if (!gpu_result.error.empty()) {
      out.error = gpu_result.error;
      return false;
    }
    if (hold_timed_out) {
      out.error = "AC-3 blocking log sink reached its 120s safety limit";
    } else {
      auto const stats_after = sirius_context->get_transparent_execution_stats();
      if (!require_transparent_execution_delta(
            stats_before, stats_after, 1, "AC-3 large GPU aggregate", out.error)) {
        return false;
      }
      out.error = "AC-3 GPU execution did not reach the blocked window within 30s";
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
  auto const gpu_completed_before_release  = gpu_result.completed.load(std::memory_order_acquire);
  auto const hold_timed_out_before_release = workers.timed_out();
  workers.release();
  workers.join_all();

  if (!gpu_result.error.empty()) {
    out.error = gpu_result.error;
    return false;
  }
  if (!cpu_ok) {
    out.error = cpu_error;
    return false;
  }
  if (hold_timed_out_before_release) {
    out.error = "AC-3 blocking log sink reached its 120s safety limit";
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

bool measure_ac8_cpu_baseline(duckdb::Connection& connection,
                              std::string const& probe_sql,
                              std::string& expected,
                              double& p95_seconds,
                              std::string& error)
{
  constexpr std::size_t kWarmups      = 3;
  constexpr std::size_t kMeasurements = 20;
  for (std::size_t repetition = 0; repetition < kWarmups; ++repetition) {
    if (repetition == 0) {
      if (!run_scalar_query_capture(
            connection, probe_sql, "AC-8 CPU baseline warmup", expected, error)) {
        return false;
      }
    } else if (!run_scalar_query(
                 connection, probe_sql, expected, "AC-8 CPU baseline warmup", error)) {
      return false;
    }
  }

  std::vector<double> measurements;
  measurements.reserve(kMeasurements);
  for (std::size_t repetition = 0; repetition < kMeasurements; ++repetition) {
    auto const begin = std::chrono::steady_clock::now();
    if (!run_scalar_query(
          connection, probe_sql, expected, "AC-8 CPU baseline measurement", error)) {
      return false;
    }
    measurements.push_back(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
  }
  std::sort(measurements.begin(), measurements.end());
  // nearest-rank p95 of 20 measurements is the 19th sorted observation.
  p95_seconds = measurements[18];
  return true;
}

void run_ac8_worker_pressure(duckdb::DuckDB& db,
                             duckdb::Connection& holder_connection,
                             duckdb::Connection& cpu_connection,
                             fs::path const& output_path,
                             slot_watchdog_result& out)
{
  constexpr std::uint64_t kCount = 5000000;
  constexpr std::size_t kWaiters = 4;
  auto const* tpch_dir_raw       = std::getenv("SIRIUS_TEST_TPCH_DIR");
  if (tpch_dir_raw == nullptr) {
    out.error = "AC-8 child is missing SIRIUS_TEST_TPCH_DIR";
    return;
  }
  auto const lineitem_path = fs::path(tpch_dir_raw) / "lineitem.parquet";
  if (!fs::is_regular_file(lineitem_path)) {
    out.error = "AC-8 lineitem parquet does not exist: " + lineitem_path.string();
    return;
  }

  if (!run_statement(holder_connection, "SET threads TO 4;", "AC-8 SET threads=4", out.error) ||
      !set_gpu_execution(holder_connection, false, out.error) ||
      !set_gpu_execution(cpu_connection, false, out.error) ||
      !create_range_table(holder_connection, "slot_bench_range", kCount, out.error) ||
      !run_statement(holder_connection, "CHECKPOINT;", "AC-8 CHECKPOINT", out.error)) {
    return;
  }

  auto const probe_sql =
    "SELECT sum(l_extendedprice * l_discount) FROM read_parquet(" +
    sql_quote(lineitem_path.string()) +
    ") WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01';";
  std::string probe_expected;
  double baseline_p95_seconds = 0;
  if (!measure_ac8_cpu_baseline(
        cpu_connection, probe_sql, probe_expected, baseline_p95_seconds, out.error)) {
    return;
  }

  // Build and schedule four GPU queries with no background workers. Once the
  // holder is parked, one external Execute() plus the three workers created by
  // SET threads=4 drive all four pre-planned sources into the execution-slot wait.
  if (!run_statement(holder_connection, "SET threads TO 1;", "AC-8 SET threads=1", out.error)) {
    return;
  }
  std::array<std::unique_ptr<duckdb::Connection>, kWaiters> waiter_connections;
  std::array<std::unique_ptr<duckdb::PreparedStatement>, kWaiters> waiter_prepared;
  std::array<std::unique_ptr<duckdb::PendingQueryResult>, kWaiters> waiter_pending;
  auto const gpu_sql      = "SELECT sum(i) FROM slot_bench_range;";
  auto const gpu_expected = range_sum(kCount);
  for (std::size_t index = 0; index < kWaiters; ++index) {
    waiter_connections[index] = std::make_unique<duckdb::Connection>(db);
    if (!set_gpu_execution(*waiter_connections[index], true, out.error)) { return; }
    waiter_prepared[index] = waiter_connections[index]->Prepare(gpu_sql);
    if (!require_success(waiter_prepared[index].get(), "AC-8 waiter Prepare", out)) { return; }
    waiter_pending[index] = waiter_prepared[index]->PendingQuery();
    if (!require_success(waiter_pending[index].get(), "AC-8 waiter PendingQuery", out)) { return; }
  }
  if (!set_gpu_execution(holder_connection, true, out.error)) { return; }

  auto context            = sirius::test::get_registered_sirius_context(holder_connection);
  auto const stats_before = context->get_transparent_execution_stats();
  async_query_result holder_result;
  std::array<async_query_result, kWaiters> waiter_results;
  std::array<std::atomic<bool>, kWaiters> waiter_started{};
  held_window_threads workers;
  workers.arm();
  auto const scenario_begin = std::chrono::steady_clock::now();
  workers.threads.emplace_back(run_async_scalar_query,
                               std::ref(holder_connection),
                               gpu_sql,
                               gpu_expected,
                               "AC-8 holder aggregate",
                               std::ref(holder_result));

  auto const holder_blocked = workers.wait_until_blocked(std::chrono::seconds{30});
  auto const holder_stats   = context->get_transparent_execution_stats();
  if (!holder_blocked || workers.timed_out() || !context->is_query_lifecycle_active() ||
      holder_stats.executions != stats_before.executions + 1 ||
      holder_result.completed.load(std::memory_order_acquire)) {
    out.error = "AC-8 holder did not enter and hold its execution window within 30s";
    return;
  }
  mark_workload_started(output_path, out);

  workers.threads.emplace_back([&]() {
    waiter_started[0].store(true, std::memory_order_release);
    run_async_pending_scalar(
      *waiter_pending[0], gpu_expected, "AC-8 external waiter aggregate", waiter_results[0]);
  });
  if (!wait_until(
        [&]() {
          auto const stats = context->get_transparent_execution_stats();
          return workers.timed_out() ||
                 (stats.executions == stats_before.executions + 2 &&
                  !holder_result.completed.load(std::memory_order_acquire) &&
                  !waiter_results[0].completed.load(std::memory_order_acquire));
        },
        std::chrono::seconds{30})) {
    out.error =
      "AC-8 external waiter did not enter the pre-acquire path before worker-pool expansion";
    return;
  }
  if (workers.timed_out() || !workers.is_blocked() ||
      !waiter_started[0].load(std::memory_order_acquire) ||
      holder_result.completed.load(std::memory_order_acquire) ||
      waiter_results[0].completed.load(std::memory_order_acquire)) {
    out.error = "AC-8 holder or external waiter completed before worker-pool expansion";
    return;
  }
  if (!run_statement(cpu_connection, "SET threads TO 4;", "AC-8 restore threads=4", out.error)) {
    return;
  }
  if (!wait_until(
        [&]() {
          return workers.timed_out() || context->get_transparent_execution_stats().executions ==
                                          stats_before.executions + 5;
        },
        std::chrono::seconds{30})) {
    out.error = "AC-8 did not observe five execution attempts within 30s";
    return;
  }
  if (workers.timed_out() || !workers.is_blocked() ||
      context->get_transparent_execution_stats().executions != stats_before.executions + 5 ||
      holder_result.completed.load(std::memory_order_acquire) ||
      waiter_results[0].completed.load(std::memory_order_acquire)) {
    out.error = "AC-8 holder or waiter completed before all execution attempts were registered";
    return;
  }

  // The other three pending queries are already executing on DuckDB workers.
  // Attach result drainers only after the +5 observation so they cannot satisfy
  // the worker-pressure admission oracle themselves.
  for (std::size_t index = 1; index < kWaiters; ++index) {
    workers.threads.emplace_back([&, index]() {
      waiter_started[index].store(true, std::memory_order_release);
      run_async_pending_scalar(*waiter_pending[index],
                               gpu_expected,
                               "AC-8 scheduler waiter aggregate",
                               waiter_results[index]);
    });
  }
  if (!wait_until(
        [&]() {
          return std::all_of(waiter_started.begin(), waiter_started.end(), [](auto const& started) {
            return started.load(std::memory_order_acquire);
          });
        },
        std::chrono::seconds{10})) {
    out.error = "AC-8 waiter result drainers did not start within 10s";
    return;
  }
  auto const any_waiter_completed = [&]() {
    return std::any_of(waiter_results.begin(), waiter_results.end(), [](auto const& result) {
      return result.completed.load(std::memory_order_acquire);
    });
  };
  if (any_waiter_completed() || holder_result.completed.load(std::memory_order_acquire) ||
      workers.timed_out() || !workers.is_blocked()) {
    out.error = "AC-8 held or waiting GPU query completed before the CPU probe";
    return;
  }

  auto const probe_begin = std::chrono::steady_clock::now();
  if (!run_scalar_query(
        cpu_connection, probe_sql, probe_expected, "AC-8 contended CPU probe", out.error)) {
    return;
  }
  auto const probe_seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - probe_begin).count();
  if (holder_result.completed.load(std::memory_order_acquire) || any_waiter_completed() ||
      workers.timed_out() || !workers.is_blocked()) {
    out.error = "AC-8 CPU probe did not finish before the held GPU window was released";
    return;
  }
  if (probe_seconds > 10.0) {
    out.error =
      "AC-8 CPU probe exceeded the 10s absolute gate: " + std::to_string(probe_seconds) + "s";
    return;
  }
  if (baseline_p95_seconds <= 0 || probe_seconds > 20.0 * baseline_p95_seconds) {
    out.error =
      "AC-8 CPU probe exceeded the 20x idle-p95 gate: probe=" + std::to_string(probe_seconds) +
      "s, p95=" + std::to_string(baseline_p95_seconds) + "s";
    return;
  }

  workers.release();
  workers.join_all();
  if (!holder_result.error.empty()) {
    out.error = holder_result.error;
    return;
  }
  for (auto const& waiter_result : waiter_results) {
    if (!waiter_result.error.empty()) {
      out.error = waiter_result.error;
      return;
    }
  }

  auto const stats_after = context->get_transparent_execution_stats();
  if (!require_transparent_execution_delta(
        stats_before, stats_after, 5, "AC-8 holder and waiters", out.error)) {
    return;
  }
  auto const scenario_seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - scenario_begin).count();
  if (scenario_seconds > 60.0) {
    out.error =
      "AC-8 pressure scenario exceeded the 60s gate: " + std::to_string(scenario_seconds) + "s";
    return;
  }
  out.b_completed = true;
}

void run_ac9_cancelled_waiter(duckdb::Connection& holder_connection,
                              duckdb::Connection& waiter_connection,
                              fs::path const& output_path,
                              slot_watchdog_result& out)
{
  constexpr std::uint64_t kCount = 500000;
  if (!run_statement(holder_connection, "SET threads TO 1;", "AC-9 SET threads=1", out.error) ||
      !set_gpu_execution(holder_connection, false, out.error) ||
      !set_gpu_execution(waiter_connection, false, out.error) ||
      !create_range_table(holder_connection, "ac9_holder", kCount, out.error) ||
      !create_range_table(holder_connection, "ac9_waiter", kCount, out.error) ||
      !run_statement(holder_connection, "CHECKPOINT;", "AC-9 CHECKPOINT", out.error) ||
      !set_gpu_execution(holder_connection, true, out.error) ||
      !set_gpu_execution(waiter_connection, true, out.error)) {
    return;
  }

  auto const waiter_sql      = "SELECT sum(i) + 9 FROM ac9_waiter;";
  auto const waiter_expected = std::to_string(std::stoull(range_sum(kCount)) + 9);
  auto waiter_prepared       = waiter_connection.Prepare(waiter_sql);
  if (!require_success(waiter_prepared.get(), "AC-9 waiter Prepare", out)) { return; }
  auto waiter_pending = waiter_prepared->PendingQuery();
  if (!require_success(waiter_pending.get(), "AC-9 waiter PendingQuery", out)) { return; }

  auto context            = sirius::test::get_registered_sirius_context(holder_connection);
  auto const stats_before = context->get_transparent_execution_stats();
  async_query_result holder_result;
  async_query_result waiter_result;
  held_window_threads workers;
  workers.arm();
  auto const holder_sql      = "SELECT sum(i) FROM ac9_holder;";
  auto const holder_expected = range_sum(kCount);
  workers.threads.emplace_back(run_async_scalar_query,
                               std::ref(holder_connection),
                               holder_sql,
                               holder_expected,
                               "AC-9 holder aggregate",
                               std::ref(holder_result));
  auto const holder_blocked = workers.wait_until_blocked(std::chrono::seconds{30});
  auto const holder_stats   = context->get_transparent_execution_stats();
  if (!holder_blocked || workers.timed_out() || !context->is_query_lifecycle_active() ||
      holder_stats.executions != stats_before.executions + 1 ||
      holder_result.completed.load(std::memory_order_acquire)) {
    out.error = "AC-9 holder did not enter and hold its execution window within 30s";
    return;
  }
  mark_workload_started(output_path, out);

  workers.threads.emplace_back(run_async_pending_scalar,
                               std::ref(*waiter_pending),
                               waiter_expected,
                               "AC-9 cancelled waiter",
                               std::ref(waiter_result));
  if (!wait_until(
        [&]() {
          auto const stats = context->get_transparent_execution_stats();
          return workers.timed_out() || (stats.executions == stats_before.executions + 2 &&
                                         !holder_result.completed.load(std::memory_order_acquire) &&
                                         !waiter_result.completed.load(std::memory_order_acquire));
        },
        std::chrono::seconds{30})) {
    out.error = "AC-9 waiter did not enter the pre-acquire path within 30s";
    return;
  }
  if (workers.timed_out() || !workers.is_blocked() ||
      holder_result.completed.load(std::memory_order_acquire) ||
      waiter_result.completed.load(std::memory_order_acquire)) {
    out.error = "AC-9 holder or waiter completed before cancellation and explicit release";
    return;
  }

  waiter_connection.Interrupt();
  workers.release();
  workers.join_all();
  if (!holder_result.error.empty()) {
    out.error = holder_result.error;
    return;
  }
  if (waiter_result.error.empty() || !contains_case_insensitive(waiter_result.error, "interrupt")) {
    out.error = "AC-9 waiter did not return an interruption error: " + waiter_result.error;
    return;
  }
  auto const stats_after_cancel = context->get_transparent_execution_stats();
  if (stats_after_cancel.executions != stats_before.executions + 2 ||
      stats_after_cancel.fallbacks != stats_before.fallbacks ||
      stats_after_cancel.runtime_fallbacks != stats_before.runtime_fallbacks) {
    out.error = "AC-9 cancelled waiter changed execution admission state after cancellation";
    return;
  }

  auto const follow_up_expected = std::to_string(std::stoull(range_sum(kCount)) + 19);
  if (!run_scalar_query(waiter_connection,
                        "SELECT sum(i) + 19 FROM ac9_waiter;",
                        follow_up_expected,
                        "AC-9 waiter-connection follow-up",
                        out.error)) {
    return;
  }
  auto const stats_after_follow_up = context->get_transparent_execution_stats();
  if (!require_transparent_execution_delta(
        stats_after_cancel, stats_after_follow_up, 1, "AC-9 follow-up", out.error)) {
    return;
  }

  std::vector<std::string> log_lines;
  if (!read_variant_log_lines(log_lines, out.error)) { return; }
  auto const waiter_window_count =
    count_window_begins_for_sql_marker(log_lines, "ac9_waiter", out.error);
  if (!out.error.empty()) { return; }
  if (waiter_window_count != 1) {
    out.error =
      "AC-9 expected only the successful follow-up window for the waiter SQL marker, "
      "observed " +
      std::to_string(waiter_window_count);
    return;
  }
  out.b_completed = true;
}

void run_ac10_unavailable_matrix(duckdb::Connection& connection,
                                 fs::path const& output_path,
                                 slot_watchdog_result& out)
{
  constexpr std::uint64_t kCount = 200000;
  if (!set_gpu_execution(connection, false, out.error) ||
      !create_range_table(connection, "ac10_health", kCount, out.error) ||
      !run_statement(connection, "CHECKPOINT;", "AC-10 CHECKPOINT", out.error) ||
      !run_statement(
        connection, "SET enable_duckdb_fallback=false;", "AC-10 disable fallback", out.error) ||
      !set_gpu_execution(connection, true, out.error)) {
    return;
  }

  auto context = sirius::test::get_registered_sirius_context(connection);
  mark_workload_started(output_path, out);
  context->mark_runtime_unavailable();
  if (context->get_runtime_health() != duckdb::SiriusContext::runtime_health::UNAVAILABLE ||
      context->is_query_lifecycle_active()) {
    out.error = "AC-10 runtime was not unavailable with no active lifecycle slot";
    return;
  }

  constexpr char const* s3_query =
    "SELECT * FROM read_parquet('s3://sirius-ac10-bogus/x.parquet');";
  if (!run_query_expect_error(connection,
                              s3_query,
                              "AC-10 fallback-disabled unavailable S3 query",
                              "unavailable",
                              out.error)) {
    return;
  }

  // CPU paths remain usable after the Sirius runtime is unavailable.
  if (!set_gpu_execution(connection, false, out.error) ||
      !run_scalar_query(
        connection, "SELECT 42;", "42", "AC-10 CPU query after runtime unavailable", out.error)) {
    return;
  }

  // A local transparent query may retain and execute its DuckDB plan when
  // fallback is enabled. Depending on whether health is observed while
  // finalizing or entering execution, either fallback counter is valid.
  if (!run_statement(
        connection, "SET enable_duckdb_fallback=true;", "AC-10 enable fallback", out.error) ||
      !set_gpu_execution(connection, true, out.error) ||
      !run_query_expect_error(connection,
                              s3_query,
                              "AC-10 fallback-enabled unavailable S3 query",
                              "unavailable",
                              out.error)) {
    return;
  }
  auto const fallback_before = context->get_transparent_execution_stats();
  if (!run_scalar_query(connection,
                        "SELECT sum(i) FROM ac10_health;",
                        range_sum(kCount),
                        "AC-10 unavailable CPU fallback",
                        out.error)) {
    return;
  }
  auto const fallback_after      = context->get_transparent_execution_stats();
  auto const plan_fallback_delta = fallback_after.fallbacks - fallback_before.fallbacks;
  auto const runtime_fallback_delta =
    fallback_after.runtime_fallbacks - fallback_before.runtime_fallbacks;
  if (plan_fallback_delta + runtime_fallback_delta != 1) {
    out.error = "AC-10 fallback-enabled query did not record exactly one plan/runtime fallback";
    return;
  }

  if (!run_statement(connection,
                     "SET enable_duckdb_fallback=false;",
                     "AC-10 disable fallback after runtime unavailable",
                     out.error) ||
      !run_query_expect_error(connection,
                              "SELECT sum(i) FROM ac10_health;",
                              "AC-10 fallback-disabled transparent query",
                              "unavailable",
                              out.error) ||
      !run_query_expect_error(connection,
                              "CALL pin_table(format='duckdb', name='ac10_health', tier='gpu');",
                              "AC-10 unavailable pin_table",
                              "unavailable",
                              out.error)) {
    return;
  }

  // Per-connection routing settings remain writable; shared-runtime settings do not.
  if (!set_gpu_execution(connection, false, out.error) ||
      !set_gpu_execution(connection, true, out.error) ||
      !run_query_expect_error(connection,
                              "SET sirius_log_level='info';",
                              "AC-10 unavailable shared-runtime SET",
                              "unavailable",
                              out.error) ||
      !set_gpu_execution(connection, false, out.error) ||
      !run_scalar_query(
        connection, "SELECT 84;", "84", "AC-10 final CPU sanity query", out.error)) {
    return;
  }
  out.b_completed = true;
}

void run_ac11_planning_error_retry(duckdb::Connection& connection,
                                   fs::path const& output_path,
                                   slot_watchdog_result& out)
{
  constexpr std::uint64_t kCount = 250000;
  if (!set_gpu_execution(connection, false, out.error) ||
      !create_range_table(connection, "ac11_valid", kCount, out.error) ||
      !run_statement(connection, "CHECKPOINT;", "AC-11 CHECKPOINT", out.error) ||
      !run_statement(
        connection, "SET enable_duckdb_fallback=false;", "AC-11 disable fallback", out.error) ||
      !set_gpu_execution(connection, true, out.error)) {
    return;
  }

  auto context            = sirius::test::get_registered_sirius_context(connection);
  auto const stats_before = context->get_transparent_execution_stats();
  mark_workload_started(output_path, out);
  if (!run_query_expect_error(connection,
                              "SELECT sum(i) FROM ac11_missing_table;",
                              "AC-11 bind failure",
                              "ac11_missing_table",
                              out.error)) {
    return;
  }
  auto const stats_after_error = context->get_transparent_execution_stats();
  if (context->is_query_lifecycle_active() ||
      context->get_runtime_health() != duckdb::SiriusContext::runtime_health::OK ||
      stats_after_error.successful_rebinds != stats_before.successful_rebinds ||
      stats_after_error.executions != stats_before.executions ||
      stats_after_error.fallbacks != stats_before.fallbacks ||
      stats_after_error.runtime_fallbacks != stats_before.runtime_fallbacks) {
    out.error = "AC-11 bind failure changed Sirius state or retained the lifecycle slot";
    return;
  }

  if (!run_scalar_query(connection,
                        "SELECT sum(i) FROM ac11_valid;",
                        range_sum(kCount),
                        "AC-11 GPU retry",
                        out.error)) {
    return;
  }
  auto const stats_after_retry = context->get_transparent_execution_stats();
  if (!require_transparent_execution_delta(
        stats_after_error, stats_after_retry, 1, "AC-11 GPU retry", out.error) ||
      stats_after_retry.successful_rebinds != stats_after_error.successful_rebinds + 1) {
    if (out.error.empty()) {
      out.error = "AC-11 GPU retry did not record exactly one successful rebind";
    }
    return;
  }
  out.b_completed = true;
}

struct ac12_operator_window {
  std::vector<std::uint64_t> operator_ids;
  bool saw_runtime_plan_creation = false;
  bool saw_query_plan            = false;
  std::string outcome;
};

bool parse_ac12_operator_windows(std::vector<std::string> const& lines,
                                 std::vector<ac12_operator_window>& windows,
                                 std::string& error)
{
  static std::regex const begin_re{
    R"(\[window\] begin instance=(\S+) connection=(\d+) window=(\d+) query=(\d+) outcome=-)"};
  static std::regex const event_re{
    R"(\[window\] (begin|end) instance=(\S+) connection=(\d+) window=(\d+))"
    R"( query=(\d+) outcome=(\S+))"};
  static std::regex const pipeline_re{R"(^Pipeline #\d+:)"};
  static std::regex const operator_re{R"(\(id=(\d+)\))"};

  for (std::size_t begin_index = 0; begin_index < lines.size(); ++begin_index) {
    std::smatch begin_match;
    if (!std::regex_search(lines[begin_index], begin_match, begin_re)) { continue; }
    auto const instance      = begin_match[1].str();
    auto const connection_id = begin_match[2].str();
    auto const window_id     = begin_match[3].str();

    std::size_t end_index = lines.size();
    std::string outcome;
    for (std::size_t index = begin_index + 1; index < lines.size(); ++index) {
      std::smatch event_match;
      if (!std::regex_search(lines[index], event_match, event_re)) { continue; }
      if (event_match[1].str() == "end" && event_match[2].str() == instance &&
          event_match[3].str() == connection_id && event_match[4].str() == window_id) {
        end_index = index;
        outcome   = event_match[6].str();
        break;
      }
    }
    if (end_index == lines.size()) {
      error = "AC-12 found a runtime window without its matching end";
      return false;
    }

    ac12_operator_window window;
    window.outcome            = outcome;
    bool in_pipeline_overview = false;
    for (std::size_t index = begin_index; index <= end_index; ++index) {
      auto const& line = lines[index];
      // Plan provenance: the window either rebuilt the Sirius plan or (first execution only)
      // consumed the one OnFinalizePrepare built. Both satisfy AC-12.
      if (line.find("Creating sirius physical plan") != std::string::npos ||
          line.find("reusing finalize-validated Sirius plan") != std::string::npos) {
        window.saw_runtime_plan_creation = true;
      }
      if (line.find("Query Plan:") != std::string::npos) { window.saw_query_plan = true; }
      if (line.find("=== Pipeline Overview ===") != std::string::npos) {
        in_pipeline_overview = true;
        continue;
      }
      if (line.find("=== Query Plan DAG ===") != std::string::npos) {
        in_pipeline_overview = false;
      }
      if (!in_pipeline_overview || !std::regex_search(line, pipeline_re)) { continue; }
      for (std::sregex_iterator match(line.begin(), line.end(), operator_re), end; match != end;
           ++match) {
        window.operator_ids.push_back(std::stoull((*match)[1].str()));
      }
    }
    windows.push_back(std::move(window));
    begin_index = end_index;
  }
  return true;
}

void run_ac12_operator_ids(duckdb::Connection& connection,
                           fs::path const& output_path,
                           slot_watchdog_result& out)
{
  constexpr std::uint64_t kCount = 300000;
  if (!set_gpu_execution(connection, false, out.error) ||
      !create_range_table(connection, "ac12_operator_ids", kCount, out.error) ||
      !run_statement(connection, "CHECKPOINT;", "AC-12 CHECKPOINT", out.error) ||
      !run_statement(
        connection, "SET enable_duckdb_fallback=false;", "AC-12 disable fallback", out.error) ||
      !set_gpu_execution(connection, true, out.error)) {
    return;
  }

  auto prepared = connection.Prepare("SELECT sum(i) FROM ac12_operator_ids;");
  if (!require_success(prepared.get(), "AC-12 Prepare", out)) { return; }
  std::vector<std::string> before_lines;
  if (!read_variant_log_lines(before_lines, out.error)) { return; }

  auto context            = sirius::test::get_registered_sirius_context(connection);
  auto const stats_before = context->get_transparent_execution_stats();
  mark_workload_started(output_path, out);
  auto const expected = range_sum(kCount);
  if (!run_prepared_scalar(*prepared, expected, "AC-12 first Execute", out.error) ||
      !run_prepared_scalar(*prepared, expected, "AC-12 second Execute", out.error)) {
    return;
  }
  auto const stats_after = context->get_transparent_execution_stats();
  if (!require_transparent_execution_delta(
        stats_before, stats_after, 2, "AC-12 repeated execution", out.error)) {
    return;
  }

  std::vector<std::string> all_lines;
  if (!read_variant_log_lines(all_lines, out.error)) { return; }
  if (all_lines.size() < before_lines.size()) {
    out.error = "AC-12 log shrank between the pre-execute and post-execute snapshots";
    return;
  }
  std::vector<std::string> execution_lines(
    all_lines.begin() + static_cast<std::ptrdiff_t>(before_lines.size()), all_lines.end());
  std::vector<ac12_operator_window> windows;
  if (!parse_ac12_operator_windows(execution_lines, windows, out.error)) { return; }
  if (windows.size() != 2) {
    out.error =
      "AC-12 expected two runtime execution windows, observed " + std::to_string(windows.size());
    return;
  }
  for (auto const& window : windows) {
    if (window.outcome != "ok" || !window.saw_runtime_plan_creation || !window.saw_query_plan ||
        window.operator_ids.empty() ||
        // Id 0 may belong to a plan-root operator absorbed from the pipeline overview.
        *std::min_element(window.operator_ids.begin(), window.operator_ids.end()) > 1) {
      out.error =
        "AC-12 runtime window did not contain a complete id-zero Sirius plan with outcome=ok";
      return;
    }
  }
  if (windows[0].operator_ids != windows[1].operator_ids) {
    out.error = "AC-12 repeated executions produced different operator-id sequences";
    return;
  }
  out.b_completed = true;
}

void run_ac13_concurrent_logging(duckdb::Connection& a,
                                 duckdb::Connection& b,
                                 fs::path const& output_path,
                                 slot_watchdog_result& out)
{
  constexpr std::uint64_t kCountA = 100000;
  constexpr std::uint64_t kCountB = 120000;
  if (!set_gpu_execution(a, false, out.error) || !set_gpu_execution(b, false, out.error) ||
      !create_range_table(a, "ac13_a", kCountA, out.error) ||
      !create_range_table(a, "ac13_b", kCountB, out.error) ||
      !run_statement(a, "CHECKPOINT;", "AC-13 CHECKPOINT", out.error) ||
      !set_gpu_execution(a, true, out.error) || !set_gpu_execution(b, true, out.error)) {
    return;
  }

  auto context            = sirius::test::get_registered_sirius_context(a);
  auto const stats_before = context->get_transparent_execution_stats();
  async_query_result result_a;
  async_query_result result_b;
  auto const sum_a = std::stoull(range_sum(kCountA));
  auto const sum_b = std::stoull(range_sum(kCountB));
  auto worker      = [](duckdb::Connection& connection,
                   std::array<std::string, 2> const& sql,
                   std::array<std::string, 2> const& expected,
                   char const* operation,
                   async_query_result& result) {
    try {
      for (std::size_t index = 0; index < sql.size(); ++index) {
        if (!run_scalar_query(connection, sql[index], expected[index], operation, result.error)) {
          break;
        }
      }
    } catch (std::exception const& error) {
      result.error = std::string(operation) + " threw: " + error.what();
    } catch (...) {
      result.error = std::string(operation) + " threw an unknown exception";
    }
    result.completed.store(true, std::memory_order_release);
  };
  std::array<std::string, 2> const sql_a      = {"SELECT sum(i) + 101 FROM ac13_a;",
                                                 "SELECT sum(i) + 102 FROM ac13_a;"};
  std::array<std::string, 2> const expected_a = {std::to_string(sum_a + 101),
                                                 std::to_string(sum_a + 102)};
  std::array<std::string, 2> const sql_b      = {"SELECT sum(i) + 201 FROM ac13_b;",
                                                 "SELECT sum(i) + 202 FROM ac13_b;"};
  std::array<std::string, 2> const expected_b = {std::to_string(sum_b + 201),
                                                 std::to_string(sum_b + 202)};
  held_window_threads workers;
  workers.arm();

  workers.threads.emplace_back(worker,
                               std::ref(a),
                               std::cref(sql_a),
                               std::cref(expected_a),
                               "AC-13 connection A",
                               std::ref(result_a));
  auto const holder_blocked = workers.wait_until_blocked(std::chrono::seconds{30});
  auto const holder_stats   = context->get_transparent_execution_stats();
  if (!holder_blocked || workers.timed_out() || !context->is_query_lifecycle_active() ||
      holder_stats.executions != stats_before.executions + 1 ||
      result_a.completed.load(std::memory_order_acquire)) {
    out.error = "AC-13 connection A did not enter the held runtime window within 30s";
    return;
  }
  mark_workload_started(output_path, out);
  workers.threads.emplace_back(worker,
                               std::ref(b),
                               std::cref(sql_b),
                               std::cref(expected_b),
                               "AC-13 connection B",
                               std::ref(result_b));
  auto const b_wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  bool b_query_observed      = false;
  while (std::chrono::steady_clock::now() < b_wait_deadline && !workers.timed_out()) {
    std::vector<std::string> log_lines;
    if (!read_variant_log_lines(log_lines, out.error)) { return; }
    b_query_observed = std::any_of(log_lines.begin(), log_lines.end(), [](auto const& line) {
      return line.find("QueryBegin: instance=") != std::string::npos &&
             line.find("+ 201") != std::string::npos && line.find("ac13_b") != std::string::npos;
    });
    if (b_query_observed && !result_a.completed.load(std::memory_order_acquire) &&
        !result_b.completed.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
  }
  if (!b_query_observed && !workers.timed_out()) {
    out.error = "AC-13 connection B did not enter the pre-acquire path within 30s";
    return;
  }
  if (workers.timed_out() || !workers.is_blocked() ||
      result_a.completed.load(std::memory_order_acquire) ||
      result_b.completed.load(std::memory_order_acquire)) {
    out.error = "AC-13 connection completed before the held window was released";
    return;
  }
  workers.release();
  workers.join_all();
  if (!result_a.error.empty()) {
    out.error = result_a.error;
    return;
  }
  if (!result_b.error.empty()) {
    out.error = result_b.error;
    return;
  }
  auto const stats_after = context->get_transparent_execution_stats();
  if (!require_transparent_execution_delta(
        stats_before, stats_after, 4, "AC-13 concurrent queries", out.error)) {
    return;
  }

  auto pending_prepared = a.Prepare("SELECT sum(i) + 301 FROM ac13_a;");
  if (!require_success(pending_prepared.get(), "AC-13 abandoned Prepare", out)) { return; }
  auto pending = pending_prepared->PendingQuery();
  if (!require_success(pending.get(), "AC-13 abandoned PendingQuery", out)) { return; }
  if (!sirius::log::get_sink()->flush()) {
    out.error = "AC-13 could not flush the Sirius file log";
    return;
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
    if (variant == "ac7_load_union") {
      run_ac7_load_union(database_path, output_path, out);
      return out;
    }

    duckdb::DuckDB db(database_path.string());
    duckdb::Connection a(db);
    duckdb::Connection b(db);

    if (variant == "stream" || variant == "pending" || variant == "cached_no_rebind" ||
        variant == "ac1_pending_gpu") {
      run_abandoned_result_scenario(variant, a, b, output_path, out);
    } else if (variant == "ac2_gpu_single_flight") {
      run_ac2_gpu_single_flight(a, b, output_path, out);
    } else if (variant == "ac3_cpu_bypasses_gpu") {
      run_ac3_cpu_bypasses_gpu(a, b, output_path, out);
    } else if (variant == "ac4_concurrent_planning") {
      run_ac4_concurrent_planning(a, b, output_path, out);
    } else if (variant == "ac5_explicit_reexecution") {
      run_ac5_explicit_reexecution(a, output_path, out);
    } else if (variant == "ac6_capture_generation") {
      run_ac6_capture_generation(a, output_path, out);
    } else if (variant == "ac8_worker_pressure") {
      run_ac8_worker_pressure(db, a, b, output_path, out);
    } else if (variant == "ac9_cancelled_waiter") {
      run_ac9_cancelled_waiter(a, b, output_path, out);
    } else if (variant == "ac10_unavailable_matrix") {
      run_ac10_unavailable_matrix(a, output_path, out);
    } else if (variant == "ac11_planning_error_retry") {
      run_ac11_planning_error_retry(a, output_path, out);
    } else if (variant == "ac12_operator_ids") {
      run_ac12_operator_ids(a, output_path, out);
    } else if (variant == "ac13_concurrent_logging") {
      run_ac13_concurrent_logging(a, b, output_path, out);
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
    auto const log_dir = variant_log_dir(output_path);
    if (variant_requires_file_log(variant)) { fs::create_directories(log_dir); }

    auto const pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      ::setenv(kEnvVariant, variant.c_str(), 1);
      ::setenv(kEnvOutput, output_path.string().c_str(), 1);
      ::setenv(kEnvConfig, config_path.string().c_str(), 1);
      if (variant_requires_file_log(variant)) {
        ::setenv("SIRIUS_LOG_BACKEND", "spdlog", 1);
        ::setenv("SIRIUS_LOG_DIR", log_dir.string().c_str(), 1);
        ::setenv("SIRIUS_LOG_LEVEL", "debug", 1);
      }
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
                 "pending result releases the slot before a GPU follow-up",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac1_pending_gpu");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "concurrent transparent GPU queries both complete correctly",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac2_gpu_single_flight");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "a CPU query completes before an in-flight GPU query",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac3_cpu_bypasses_gpu", std::chrono::seconds{240});
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "concurrent prepared planning preserves query ownership",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac4_concurrent_planning");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "explicit prepared execution observes a pin-state change",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac5_explicit_reexecution");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "stale capture is not consumed after a planning generation change",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac6_capture_generation");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "extension load unions optimizer masks with the user setting",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac7_load_union");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "worker pressure leaves bounded CPU capacity",
                 "[.][query_lifecycle][slot_leak_gate]")
{
  auto const* tpch_dir_raw = std::getenv("SIRIUS_TEST_TPCH_DIR");
  if (tpch_dir_raw == nullptr) {
    FAIL("SIRIUS_TEST_TPCH_DIR is unset; the worker-pressure gate requires its fixture");
    return;
  }
  auto const lineitem_path = fs::path(tpch_dir_raw) / "lineitem.parquet";
  if (!fs::is_regular_file(lineitem_path)) {
    FAIL("lineitem.parquet is absent; the worker-pressure gate requires its fixture");
    return;
  }
  require_variant_succeeds("ac8_worker_pressure", std::chrono::seconds{240});
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "a cancelled waiter cannot enter a later execution window",
                 "[.][query_lifecycle][slot_leak_gate]")
{
  require_variant_succeeds("ac9_cancelled_waiter", std::chrono::seconds{90});
}

TEST_CASE_METHOD(
  QueryLifecycleSlotFixture,
  "an unavailable runtime keeps CPU paths usable and rejects shared-runtime settings",
  "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac10_unavailable_matrix");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "a binding error does not retain lifecycle state",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac11_planning_error_retry");
}

TEST_CASE_METHOD(QueryLifecycleSlotFixture,
                 "operator ids restart for each repeated execution",
                 "[query_lifecycle][slot_leak]")
{
  require_variant_succeeds("ac12_operator_ids");
}
