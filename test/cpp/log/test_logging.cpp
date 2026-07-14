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
#include "log/logging.hpp"
#include "log/noop_sink.hpp"
#include "log/spdlog_sink.hpp"
#include "utils/log_test_utils.hpp"

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

using namespace sirius::log;

using recording_backend        = sirius::test::recording_log_backend;
using scoped_recording_backend = sirius::test::scoped_recording_log_backend;

TEST_CASE("The backend level gates by severity", "[log]")
{
  scoped_recording_backend scoped;
  auto& backend = scoped.backend();

  backend.set_level(level::info);
  CHECK_FALSE(backend.should_log(level::trace));
  CHECK_FALSE(backend.should_log(level::debug));
  CHECK(backend.should_log(level::info));
  CHECK(backend.should_log(level::warn));
  CHECK(backend.should_log(level::error));
  CHECK(backend.should_log(level::critical));

  backend.set_level(level::off);
  CHECK_FALSE(backend.should_log(level::trace));
  CHECK_FALSE(backend.should_log(level::critical));

  backend.set_level(level::trace);
  CHECK(backend.should_log(level::trace));
  CHECK(backend.should_log(level::critical));
}

TEST_CASE("Unknown level names are rejected and leave the default", "[log]")
{
  // The config layer relies on this: it seeds `info` and keeps it on a bad name.
  level lvl = level::info;
  CHECK_FALSE(string_to_enum("verbose", lvl));
  CHECK(lvl == level::info);
  CHECK(string_to_enum("warn", lvl));
  CHECK(lvl == level::warn);
}

TEST_CASE("SIRIUS_LOG macros format and attribute the call site", "[log]")
{
  scoped_recording_backend scoped;

  SIRIUS_LOG_INFO("x={}", 42);
  const uint32_t expected_line = __LINE__ - 1;

  auto messages = scoped.backend().records();
  REQUIRE(messages.size() == 1);
  CHECK(messages[0].level == level::info);
  CHECK(messages[0].message == "x=42");
  CHECK(messages[0].file.ends_with("test_logging.cpp"));
  CHECK(messages[0].line == expected_line);
}

TEST_CASE("The sink enforces the level gate", "[log]")
{
  scoped_recording_backend scoped;

  scoped.backend().set_level(level::off);
  get_sink()->log(level::error, std::source_location::current(), "must not appear");
  CHECK(scoped.backend().count() == 0);

  scoped.backend().set_level(level::error);
  get_sink()->log(level::error, std::source_location::current(), "must appear");
  CHECK(scoped.backend().count() == 1);
}

TEST_CASE("get_sink()->flush() forwards and reports reliability", "[log]")
{
  scoped_recording_backend scoped;

  CHECK(get_sink()->flush());
  CHECK(scoped.backend().flush_count() == 1);
}

TEST_CASE("set_sink(nullptr) installs a discarding backend", "[log]")
{
  scoped_recording_backend scoped;

  set_sink(nullptr);  // swaps in a noop, detaching the recorder
  SIRIUS_LOG_ERROR("dropped {}", 1);
  get_sink()->log(level::critical, std::source_location::current(), "dropped");
  CHECK(get_sink()->flush());  // the noop flush is vacuously reliable
  CHECK(scoped.backend().count() == 0);
}

TEST_CASE("Concurrent logging survives backend swaps", "[log]")
{
  scoped_recording_backend scoped;
  auto backend_a = scoped.backend_ptr();
  auto backend_b = std::make_shared<recording_backend>();
  backend_a->set_level(level::trace);
  backend_b->set_level(level::trace);

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
    set_sink(use_a ? backend_a : backend_b);
    use_a = !use_a;
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto& t : loggers) {
    t.join();
  }

  CHECK(backend_a->count() + backend_b->count() > 0);
}

TEST_CASE("The noop sink accepts and discards everything", "[log]")
{
  auto noop = make_noop_sink();
  REQUIRE(noop != nullptr);
  noop->log(level::critical, std::source_location::current(), "into the void");
  CHECK(noop->flush());

  scoped_recording_backend scoped;
  set_sink(noop);
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
    auto backend = make_spdlog_sink({temp_dir.string(), std::nullopt});
    REQUIRE(backend != nullptr);
    backend->set_level(level::info);
    set_sink(backend);
    SIRIUS_LOG_INFO("hello");
    REQUIRE(get_sink()->flush());
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
