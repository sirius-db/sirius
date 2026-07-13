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

#include "catch.hpp"
#include "config.hpp"
#include "log/log_backend.hpp"
#include "log/logging.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using namespace sirius;

namespace {

struct recorded_message {
  log_level level;
  std::string file;
  uint32_t line;
  std::string message;
};

class recording_backend final : public log_backend {
 public:
  void log(log_level level, const std::source_location& loc, std::string_view message) override
  {
    std::lock_guard lock(_mutex);
    _messages.push_back({level, loc.file_name(), loc.line(), std::string{message}});
  }

  bool flush() override
  {
    ++_flush_count;
    return true;
  }

  std::vector<recorded_message> messages() const
  {
    std::lock_guard lock(_mutex);
    return _messages;
  }

  size_t count() const
  {
    std::lock_guard lock(_mutex);
    return _messages.size();
  }

  int flush_count() const { return _flush_count; }

 private:
  mutable std::mutex _mutex;
  std::vector<recorded_message> _messages;
  std::atomic<int> _flush_count{0};
};

/// Installs a recording backend for the scope of a test and restores the
/// process-wide logger (initialized in unittest.cpp's main) on exit.
class scoped_recording_backend {
 public:
  explicit scoped_recording_backend(std::string_view level = "trace")
    : _backend(std::make_shared<recording_backend>())
  {
    InitGlobalLogger(_backend, level);
  }

  ~scoped_recording_backend()
  {
    using duckdb::Config;
    InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_MS, Config::LOG_BACKEND);
  }

  recording_backend& backend() { return *_backend; }
  std::shared_ptr<recording_backend> backend_ptr() { return _backend; }

 private:
  std::shared_ptr<recording_backend> _backend;
};

}  // namespace

TEST_CASE("ShouldLog gates by the global level", "[log]")
{
  scoped_recording_backend scoped;

  SetGlobalLogLevel("info");
  CHECK_FALSE(ShouldLog(log_level::trace));
  CHECK_FALSE(ShouldLog(log_level::debug));
  CHECK(ShouldLog(log_level::info));
  CHECK(ShouldLog(log_level::warn));
  CHECK(ShouldLog(log_level::error));
  CHECK(ShouldLog(log_level::critical));

  SetGlobalLogLevel("off");
  CHECK_FALSE(ShouldLog(log_level::trace));
  CHECK_FALSE(ShouldLog(log_level::critical));

  SetGlobalLogLevel("trace");
  CHECK(ShouldLog(log_level::trace));
  CHECK(ShouldLog(log_level::critical));
}

TEST_CASE("Unknown level names fall back to info", "[log]")
{
  scoped_recording_backend scoped;

  SetGlobalLogLevel("verbose");
  CHECK_FALSE(ShouldLog(log_level::debug));
  CHECK(ShouldLog(log_level::info));
}

TEST_CASE("SIRIUS_LOG macros format and attribute the call site", "[log]")
{
  scoped_recording_backend scoped;

  SIRIUS_LOG_INFO("x={}", 42);
  const uint32_t expected_line = __LINE__ - 1;

  auto messages = scoped.backend().messages();
  REQUIRE(messages.size() == 1);
  CHECK(messages[0].level == log_level::info);
  CHECK(messages[0].message == "x=42");
  CHECK(messages[0].file.ends_with("test_logging.cpp"));
  CHECK(messages[0].line == expected_line);
}

TEST_CASE("LogAt enforces the level gate", "[log]")
{
  scoped_recording_backend scoped;

  SetGlobalLogLevel("off");
  LogAt(log_level::error, std::source_location::current(), "must not appear");
  CHECK(scoped.backend().count() == 0);

  SetGlobalLogLevel("error");
  LogAt(log_level::error, std::source_location::current(), "must appear");
  CHECK(scoped.backend().count() == 1);
}

TEST_CASE("FlushGlobalLogger forwards to the backend and reports reliability", "[log]")
{
  scoped_recording_backend scoped;

  CHECK(FlushGlobalLogger());
  CHECK(scoped.backend().flush_count() == 1);
}

TEST_CASE("A null backend resets to the pre-init state", "[log]")
{
  scoped_recording_backend scoped;

  InitGlobalLogger(nullptr, "trace");
  CHECK_FALSE(ShouldLog(log_level::critical));
  SIRIUS_LOG_ERROR("dropped {}", 1);
  LogAt(log_level::critical, std::source_location::current(), "dropped");
  CHECK_FALSE(FlushGlobalLogger());
  SetGlobalLogLevel("trace");
  // The level gate opens but there is still no backend; logging stays safe.
  SIRIUS_LOG_ERROR("still dropped");
  CHECK(scoped.backend().count() == 0);
}

TEST_CASE("Concurrent logging survives backend swaps", "[log]")
{
  scoped_recording_backend scoped;
  auto backend_a = scoped.backend_ptr();
  auto backend_b = std::make_shared<recording_backend>();

  std::atomic<bool> stop{false};
  std::vector<std::thread> loggers;
  loggers.reserve(4);
  for (int t = 0; t < 4; t++) {
    loggers.emplace_back([&stop, t] {
      while (!stop.load(std::memory_order_relaxed)) {
        SIRIUS_LOG_INFO("thread {} says hi", t);
      }
    });
  }

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  bool use_a    = false;
  while (std::chrono::steady_clock::now() < deadline) {
    InitGlobalLogger(use_a ? backend_a : backend_b, "trace");
    use_a = !use_a;
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto& t : loggers) {
    t.join();
  }

  CHECK(backend_a->count() + backend_b->count() > 0);
}

TEST_CASE("log_backend_type round-trips through its string form", "[log]")
{
  for (auto type : {log_backend_type::spdlog, log_backend_type::noop}) {
    std::string name;
    REQUIRE(enum_to_string(type, name));
    log_backend_type parsed{};
    REQUIRE(string_to_enum(name, parsed));
    CHECK(parsed == type);
  }

  log_backend_type parsed{};
  CHECK_FALSE(string_to_enum("bogus", parsed));
}

TEST_CASE("The noop backend accepts and discards everything", "[log]")
{
  auto noop = make_noop_backend();
  REQUIRE(noop != nullptr);
  noop->log(log_level::critical, std::source_location::current(), "into the void");
  CHECK(noop->flush());

  scoped_recording_backend scoped;
  InitGlobalLogger(noop, "trace");
  SIRIUS_LOG_ERROR("also into the void");
  CHECK(scoped.backend().count() == 0);
}

TEST_CASE("The spdlog backend writes the documented line format", "[log]")
{
  // The multi-GPU audit tests (mgpu_test_utils.hpp parse_audit_log) and the
  // REST retry-logging test parse this exact byte format from sirius.log.
  auto temp_dir = std::filesystem::temp_directory_path() / "sirius_test_logging_spdlog_format";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);

  {
    scoped_recording_backend scoped;  // restores the process logger on exit
    auto backend = make_spdlog_backend({temp_dir.string(), std::nullopt});
    REQUIRE(backend != nullptr);
    InitGlobalLogger(backend, "info");
    SIRIUS_LOG_INFO("hello");
    REQUIRE(FlushGlobalLogger());
  }

  std::string line;
  for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
    std::ifstream file(entry.path());
    std::getline(file, line);
    if (!line.empty()) { break; }
  }
  std::regex pattern(
    R"(^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[info\] \[test_logging\.cpp:\d+\] hello$)");
  INFO("log line: " << line);
  CHECK(std::regex_match(line, pattern));

  std::filesystem::remove_all(temp_dir);
}
