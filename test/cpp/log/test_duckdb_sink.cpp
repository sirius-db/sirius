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
#include "log/duckdb_sink.hpp"
#include "log/logging.hpp"

#include <duckdb.hpp>

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

}  // namespace

TEST_CASE("duckdb sink forwards Sirius logs into duckdb_logs", "[log]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto sink = sirius::log::make_duckdb_sink(*db.instance);

  // DuckDB logging is off by default: the sink defers and drops everything.
  CHECK_FALSE(sink->should_log(level::error));

  REQUIRE_FALSE(con.Query("PRAGMA enable_logging")->HasError());

  // With logging enabled (default INFO threshold), error-and-above passes.
  CHECK(sink->should_log(level::error));

  sink_restorer restore;
  sirius::log::set_sink(sink);
  SIRIUS_LOG_ERROR("duckdb sink marker {}", 4242);
  CHECK(sink->flush());

  auto result = con.Query("SELECT message FROM duckdb_logs WHERE type = 'Sirius'");
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() >= 1);

  bool found = false;
  for (duckdb::idx_t i = 0; i < result->RowCount(); ++i) {
    if (result->GetValue(0, i).ToString().find("duckdb sink marker 4242") != std::string::npos) {
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
    REQUIRE_FALSE(con.Query("PRAGMA enable_logging")->HasError());
    sink = sirius::log::make_duckdb_sink(*db.instance);
    CHECK(sink->should_log(level::error));
  }
  // The DatabaseInstance is gone; the sink's weak reference has expired.
  CHECK_FALSE(sink->should_log(level::error));
  sink->log(level::error, std::source_location::current(), "after destroy");  // must not crash
  CHECK(sink->flush());
}
