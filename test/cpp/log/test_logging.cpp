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
#include "log/logging.hpp"
#include "utils/log_test_utils.hpp"

#include <atomic>
#include <chrono>
#include <latch>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using sirius::log::get_sink;
using sirius::log::level;
using sirius::log::set_sink;
using sirius::log::string_to_enum;

using recording_sink        = sirius::test::recording_log_sink;
using scoped_recording_sink = sirius::test::scoped_recording_log_sink;

TEST_CASE("Unknown level names are rejected and leave the default", "[log]")
{
  // The config layer relies on this: it seeds `info` and keeps it on a bad name.
  level lvl = level::info;
  CHECK_FALSE(string_to_enum("verbose", lvl));
  CHECK(lvl == level::info);
  CHECK(string_to_enum("warn", lvl));
  CHECK(lvl == level::warn);
}

// test cases using SIRIUS_LOG_* only make sense when logging is compiled in.
#if SIRIUS_ACTIVE_LOG_LEVEL != SIRIUS_LOG_LEVEL_OFF

TEST_CASE("SIRIUS_LOG macros format and attribute the call site", "[log]")
{
  scoped_recording_sink scoped;

  SIRIUS_LOG_FATAL("x={}", 42);
  const uint32_t expected_line = __LINE__ - 1;

  auto messages = scoped.sink().records();
  REQUIRE(messages.size() == 1);
  CHECK(messages[0].level == level::critical);
  CHECK(messages[0].message == "x=42");
  CHECK(messages[0].file.ends_with("test_logging.cpp"));
  CHECK(messages[0].line == expected_line);
}

TEST_CASE("Concurrent logging survives sink swaps", "[log]")
{
  scoped_recording_sink scoped;
  auto sink_a = scoped.sink_ptr();
  auto sink_b = std::make_shared<recording_sink>();
  sink_a->set_level(level::trace);
  sink_b->set_level(level::trace);

  std::atomic<bool> stop{false};
  std::latch ready{4};
  std::vector<std::thread> loggers;
  loggers.reserve(4);
  for (int t = 0; t < 4; t++) {
    loggers.emplace_back([&stop, &ready, t] {
      ready.count_down();
      while (!stop.load(std::memory_order_relaxed)) {
        SIRIUS_LOG_FATAL("thread {} says hi", t);
      }
    });
  }

  // Wait until every logger thread is up so the swaps below race with concurrent
  // logging rather than with thread startup.
  ready.wait();

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  bool use_a    = false;
  while (std::chrono::steady_clock::now() < deadline) {
    set_sink(use_a ? sink_a : sink_b);
    use_a = !use_a;
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto& t : loggers) {
    t.join();
  }

  CHECK(sink_a->count() + sink_b->count() > 0);
}

#endif

TEST_CASE("set_sink(nullptr) installs a discarding sink", "[log]")
{
  scoped_recording_sink scoped;

  set_sink(nullptr);  // swaps in a noop, detaching the recorder
  SIRIUS_LOG_ERROR("dropped {}", 1);
  get_sink()->log(level::critical, std::source_location::current(), "dropped");
  CHECK(get_sink()->flush());  // the noop flush is vacuously reliable
  CHECK(scoped.sink().count() == 0);
}
