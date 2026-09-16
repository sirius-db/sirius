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

/**
 * @file test_pin_registry_epoch.cpp
 * @brief A plan validated at finalize is reused only while the pinned registry is unchanged.
 *
 * pin_table/unpin_table take the query-lifecycle slot, so they cannot interleave with a plan
 * or execution window — but they can land BETWEEN the finalize window that validates a plan
 * and the execution window that runs it. The plan bakes in pin-derived decisions, so it
 * carries the registry epoch it was built against and is discarded when that epoch moves.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/log_test_utils.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

using PinEpochFixture = sirius::test::GpuExecutionFixture;

namespace {

void query_ok(duckdb::Connection& con, const std::string& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO("query failed: " << sql << "\n  error: " << (result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

std::uint64_t epoch_of(duckdb::Connection& con)
{
  return sirius::test::get_registered_sirius_context(con)->get_scan_manager().pin_registry_epoch();
}

bool logged_containing(const sirius::test::scoped_recording_log_sink& logs, std::string_view needle)
{
  for (auto const& record : logs.records()) {
    if (record.message.find(needle) != std::string::npos) { return true; }
  }
  return false;
}

}  // namespace

TEST_CASE_METHOD(PinEpochFixture,
                 "pin and unpin move the pinned-registry epoch",
                 "[integration][gpu_execution][pin_registry_epoch]")
{
  auto& con = *this->con;
  query_ok(con, "CREATE OR REPLACE TABLE epoch_t AS SELECT range AS i FROM range(1000);");
  query_ok(con, "CHECKPOINT;");

  auto const before_pin = epoch_of(con);
  query_ok(con, "CALL pin_table(format='duckdb', name='epoch_t', tier='gpu');");
  auto const after_pin = epoch_of(con);
  REQUIRE(after_pin > before_pin);

  query_ok(con, "CALL unpin_table('epoch_t');");
  auto const after_unpin = epoch_of(con);
  REQUIRE(after_unpin > after_pin);
}

TEST_CASE_METHOD(PinEpochFixture,
                 "a pinned-registry change between finalize and execute discards the plan",
                 "[integration][gpu_execution][pin_registry_epoch]")
{
  auto& con = *this->con;
  query_ok(con, "CREATE OR REPLACE TABLE epoch_reuse_t AS SELECT range AS i FROM range(1000);");
  query_ok(con, "CHECKPOINT;");
  query_ok(con, "SET gpu_execution = true;");

  auto const expected = 1000LL * 999 / 2;
  auto const sql      = std::string{"SELECT sum(i) FROM epoch_reuse_t;"};

  SECTION("registry unchanged: the validated plan is reused")
  {
    query_ok(con, "CALL pin_table(format='duckdb', name='epoch_reuse_t', tier='gpu');");
    sirius::test::scoped_recording_log_sink logs("info");
    auto prepared = con.Prepare(sql);
    REQUIRE(prepared);
    REQUIRE_FALSE(prepared->HasError());
    auto result = prepared->Execute();
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    auto chunk = result->Fetch();
    REQUIRE(chunk);
    REQUIRE(chunk->size() == 1);
    REQUIRE(chunk->GetValue(0, 0).ToString() == std::to_string(expected));
    CHECK(logged_containing(logs, "reusing finalize-validated Sirius plan"));
    CHECK_FALSE(logged_containing(logs, "discarding finalize-validated Sirius plan"));
    query_ok(con, "CALL unpin_table('epoch_reuse_t');");
  }

  SECTION("a registry change between the windows discards the validated plan")
  {
    query_ok(con, "CALL pin_table(format='duckdb', name='epoch_reuse_t', tier='gpu');");
    sirius::test::scoped_recording_log_sink logs("info");
    // The real race is microseconds wide — between the finalize window releasing the
    // lifecycle slot and the execution window acquiring it — so it is injected here rather
    // than scheduled.
    query_ok(con, "SET sirius_test_inject_pin_registry_change = true;");

    auto prepared = con.Prepare(sql);
    REQUIRE(prepared);
    REQUIRE_FALSE(prepared->HasError());
    auto result = prepared->Execute();
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    auto chunk = result->Fetch();
    REQUIRE(chunk);
    REQUIRE(chunk->size() == 1);
    REQUIRE(chunk->GetValue(0, 0).ToString() == std::to_string(expected));

    CHECK(logged_containing(logs, "discarding finalize-validated Sirius plan"));
    CHECK_FALSE(logged_containing(logs, "reusing finalize-validated Sirius plan"));

    query_ok(con, "SET sirius_test_inject_pin_registry_change = false;");
    query_ok(con, "CALL unpin_table('epoch_reuse_t');");
  }

  SECTION("another connection unpins between the windows: results stay correct")
  {
    query_ok(con, "CALL pin_table(format='duckdb', name='epoch_reuse_t', tier='gpu');");
    sirius::test::scoped_recording_log_sink logs("info");

    auto prepared = con.Prepare(sql);
    REQUIRE(prepared);
    REQUIRE_FALSE(prepared->HasError());

    // DuckDB re-binds the prepared statement on every Execute, so Sirius re-finalizes and the
    // plan actually run is built after this unpin. Pin this behaviour down: the result must be
    // correct however the engine got there.
    duckdb::Connection unpinner(*con.context->db);
    query_ok(unpinner, "CALL unpin_table('epoch_reuse_t');");

    auto result = prepared->Execute();
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    auto chunk = result->Fetch();
    REQUIRE(chunk);
    REQUIRE(chunk->size() == 1);
    REQUIRE(chunk->GetValue(0, 0).ToString() == std::to_string(expected));
  }
}
