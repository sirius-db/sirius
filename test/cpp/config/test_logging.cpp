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
#include "sirius/sirius_logging.hpp"
#include "util/env_guard.hpp"
#include "utils/log_test_utils.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

using sirius::duckdb_extension::apply_logging_setting;
using sirius::duckdb_extension::logging_options;
using sirius::duckdb_extension::logging_setting;
using sirius::log::get_sink;
using sirius::log::level;

TEST_CASE("DuckDB logging defaults remain unchanged", "[log][logging_config]")
{
  logging_options defaults;
  CHECK(defaults.backend == "spdlog");
  CHECK(defaults.level == "info");
  CHECK(defaults.directory == "log");
  CHECK(defaults.flush_seconds == 3);
}

TEST_CASE("Invalid SQL logging updates preserve the sink", "[log][logging_config]")
{
  sirius::test::scoped_recording_log_sink recording;
  auto previous = get_sink();
  CHECK_THROWS_WITH(apply_logging_setting({.backend = "syslog"}, logging_setting::backend),
                    Catch::Matchers::ContainsSubstring("Unknown sirius_log_backend"));
  CHECK_THROWS_WITH(apply_logging_setting({.level = "verbose"}, logging_setting::level),
                    Catch::Matchers::ContainsSubstring("sirius_log_level must be one of"));
  CHECK_THROWS_WITH(
    apply_logging_setting({.flush_seconds = -1}, logging_setting::flush_seconds),
    Catch::Matchers::ContainsSubstring("sirius_log_flush_seconds must be non-negative"));
  CHECK(get_sink() == previous);
  CHECK(get_sink()->should_log(level::trace));
}

TEST_CASE("SQL logging updates reuse sinks where appropriate", "[log][logging_config]")
{
  sirius::test::scoped_recording_log_sink recording;
  auto previous = get_sink();
  apply_logging_setting({.backend = "noop", .directory = "unused"}, logging_setting::directory);
  apply_logging_setting({.backend = "noop", .flush_seconds = 0}, logging_setting::flush_seconds);
  CHECK(get_sink() == previous);
  apply_logging_setting({.level = "error"}, logging_setting::level);
  CHECK(get_sink() == previous);
  CHECK_FALSE(get_sink()->should_log(level::warn));
  CHECK(get_sink()->should_log(level::error));
  apply_logging_setting({.backend = "duckdb"}, logging_setting::backend);
  CHECK(get_sink() == previous);
  apply_logging_setting({.backend = "noop"}, logging_setting::backend);
  CHECK(get_sink() != previous);
  CHECK_FALSE(get_sink()->should_log(level::critical));
}

TEST_CASE("Extension logging validates environment values before installing",
          "[log][logging_config]")
{
  sirius::test::scoped_recording_log_sink recording;
  auto previous = get_sink();
  sirius::util::env_guard backend{"SIRIUS_LOG_BACKEND", "noop"};
  sirius::util::env_guard level_env{"SIRIUS_LOG_LEVEL", "warn"};
  sirius::util::env_guard directory{"SIRIUS_LOG_DIR", "unused"};

  SECTION("valid overrides are returned as plain startup options")
  {
    auto options = sirius::duckdb_extension::logging_options_from_environment();
    CHECK(options.backend == "noop");
    CHECK(options.level == "warn");
    CHECK(options.directory == "unused");
  }
  SECTION("unknown backend")
  {
    sirius::util::env_guard invalid{"SIRIUS_LOG_BACKEND", "syslog"};
    CHECK_THROWS_WITH(sirius::duckdb_extension::logging_options_from_environment(),
                      Catch::Matchers::ContainsSubstring("SIRIUS_LOG_BACKEND"));
  }
  SECTION("unknown level")
  {
    sirius::util::env_guard invalid{"SIRIUS_LOG_LEVEL", "verbose"};
    CHECK_THROWS_WITH(sirius::duckdb_extension::logging_options_from_environment(),
                      Catch::Matchers::ContainsSubstring("SIRIUS_LOG_LEVEL"));
  }
  CHECK(get_sink() == previous);
}

TEST_CASE("Integration sink construction failures preserve the previous sink",
          "[log][logging_config]")
{
  sirius::test::scoped_recording_log_sink recording;
  auto root = std::filesystem::temp_directory_path() /
              ("sirius-logging-adapter-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);
  struct cleanup_directory {
    std::filesystem::path path;
    ~cleanup_directory()
    {
      std::error_code ec;
      std::filesystem::remove_all(path, ec);
    }
  } cleanup{root};
  std::ofstream(root / "blocker") << "file";
  logging_options current{
    .level = "error", .directory = (root / "valid").string(), .flush_seconds = 0};
  sirius::duckdb_extension::install_log_sink(current);
  auto previous = get_sink();
  auto invalid  = (root / "blocker" / "child").string();

  CHECK_THROWS(apply_logging_setting({.directory = invalid}, logging_setting::directory));
  CHECK(get_sink() == previous);
  current.flush_seconds = 1;
  apply_logging_setting(current, logging_setting::flush_seconds);
  CHECK(get_sink() != previous);
  CHECK_FALSE(get_sink()->should_log(level::warn));
  previous = get_sink();
}
