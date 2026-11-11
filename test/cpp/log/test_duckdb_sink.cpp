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
#include "log/duckdb_sink.hpp"
#include "log/logging.hpp"
#include "log/noop_sink.hpp"
#include "sirius_context.hpp"
#include "util/env_guard.hpp"

#include <duckdb.hpp>  // IWYU pragma: keep

#include <memory>
#include <source_location>
#include <string>

using sirius::log::level;

namespace {

// Restores whatever sink the test suite installed (unittest.cpp's main sets one).
struct sink_restorer {
  std::shared_ptr<sirius::log::sink> prev = sirius::log::get_sink();
  ~sink_restorer() { sirius::log::set_sink(prev); }
};

// Restores the process-wide log settings that loading Sirius takes from the environment.
struct log_backend_restorer {
  std::string backend = duckdb::Config::LOG_BACKEND;
  std::string dir     = duckdb::Config::LOG_DIR;
  std::string level   = duckdb::Config::LOG_LEVEL;
  ~log_backend_restorer()
  {
    duckdb::Config::LOG_BACKEND = backend;
    duckdb::Config::LOG_DIR     = dir;
    duckdb::Config::LOG_LEVEL   = level;
  }
};

}  // namespace

TEST_CASE("duckdb sink forwards Sirius logs into duckdb_logs", "[log]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto sink = sirius::log::make_duckdb_sink(*db.instance);

  // DuckDB logging is off by default: the sink defers and drops everything.
  CHECK_FALSE(sink->should_log(level::error));

  REQUIRE_FALSE(con.Query("CALL enable_logging()")->HasError());

  // With logging enabled (default INFO threshold), error-and-above passes.
  CHECK(sink->should_log(level::error));

  sink_restorer restore;
  sirius::log::set_sink(sink);
  SIRIUS_LOG_ERROR("duckdb sink marker {}", 4242);
  // flush() forwards to storage but reports false — best-effort, no delivery guarantee.
  CHECK_FALSE(sink->flush());

  auto result = con.Query("SELECT message FROM duckdb_logs WHERE type = 'Sirius'");
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() >= 1);

  bool found = false;
  for (duckdb::idx_t i = 0; i < result->RowCount(); ++i) {
    if (result->Collection().GetValue(0, i).ToString().find("duckdb sink marker 4242") !=
        std::string::npos) {
      found = true;
    }
  }
  CHECK(found);
}

TEST_CASE("duckdb sink is a safe no-op after the database is destroyed", "[log]")
{
  std::shared_ptr<sirius::log::sink> sink;
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    REQUIRE_FALSE(con.Query("CALL enable_logging()")->HasError());
    sink = sirius::log::make_duckdb_sink(*db.instance);
    CHECK(sink->should_log(level::error));
  }
  // The DatabaseInstance is gone; the sink's weak reference has expired.
  CHECK_FALSE(sink->should_log(level::error));
  sink->log(level::error, std::source_location::current(), "after destroy");  // must not crash
  CHECK_FALSE(sink->flush());  // safe no-op, reports false once the db is gone
}

TEST_CASE("duckdb backend captures messages logged while loading the configuration", "[log]")
{
  sink_restorer restore_sink;
  log_backend_restorer restore_backend;
  sirius::util::env_guard disable_env("SIRIUS_DISABLE", "1");
  sirius::util::env_guard backend_env("SIRIUS_LOG_BACKEND", "duckdb");

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  REQUIRE_FALSE(con.Query("CALL enable_logging()")->HasError());
  duckdb::SiriusContextExtensionCallback callback(db.instance.get());

  auto result = con.Query(
    "SELECT count(*) FROM duckdb_logs WHERE type = 'Sirius' "
    "AND message LIKE '%Sirius disabled via SIRIUS_DISABLE%'");
  REQUIRE_FALSE(result->HasError());
  CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == 1);
}

TEST_CASE("a failed load restores the previous log sink", "[log]")
{
  sink_restorer restore_sink;
  log_backend_restorer restore_backend;
  auto const previous = sirius::log::make_noop_sink();
  sirius::log::set_sink(previous);
  duckdb::Config::LOG_BACKEND = "noop";
  duckdb::Config::LOG_DIR     = "previous-log-dir";
  duckdb::Config::LOG_LEVEL   = "warn";
  sirius::util::env_guard enable_env("SIRIUS_DISABLE", "0");
  sirius::util::env_guard backend_env("SIRIUS_LOG_BACKEND", "duckdb");
  sirius::util::env_guard dir_env("SIRIUS_LOG_DIR", "failed-load-log-dir");
  sirius::util::env_guard level_env("SIRIUS_LOG_LEVEL", "debug");
  sirius::util::env_guard config_env("SIRIUS_CONFIG_FILE", "/nonexistent/sirius.yaml");

  CHECK_THROWS_WITH(
    duckdb::DuckDB(nullptr),
    Catch::Matchers::ContainsSubstring("SIRIUS_CONFIG_FILE points to non-existent"));
  CHECK(sirius::log::get_sink() == previous);
  CHECK(duckdb::Config::LOG_BACKEND == "noop");
  CHECK(duckdb::Config::LOG_DIR == "previous-log-dir");
  CHECK(duckdb::Config::LOG_LEVEL == "warn");
}

TEST_CASE("a successful load keeps the sink it installs", "[log]")
{
  sink_restorer restore_sink;
  log_backend_restorer restore_backend;
  auto const previous = sirius::log::make_noop_sink();
  sirius::log::set_sink(previous);
  sirius::util::env_guard disable_env("SIRIUS_DISABLE", "1");
  sirius::util::env_guard backend_env("SIRIUS_LOG_BACKEND", "duckdb");

  duckdb::DuckDB db(nullptr);
  CHECK(sirius::log::get_sink() != previous);
  CHECK(duckdb::Config::LOG_BACKEND == "duckdb");
}
