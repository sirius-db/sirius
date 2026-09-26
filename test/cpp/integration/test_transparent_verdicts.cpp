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

#include "op/scan/table_scan/scan_contract.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "transparent/read_view_registry.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb/main/client_config.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/planner/logical_operator.hpp>

using namespace sirius::op::scan;

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a transparent verdict routing and recertification",
                 "[transparent][verdict][integration]")
{
  run_ok("CREATE TABLE r2a_routing(i INTEGER)");
  run_ok("INSERT INTO r2a_routing VALUES (1),(2)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET sirius_test_inject_scan_verdict='unsupported'");
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query("SELECT sum(i) FROM r2a_routing");
  REQUIRE_FALSE(result->HasError());
  CHECK(result->GetValue(0, 0).GetValue<int64_t>() == 3);
  auto after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.fallbacks == before.fallbacks + 1);
  CHECK(after.semantic_verdicts[1] == before.semantic_verdicts[1] + 1);
  CHECK(after.scan_lowerings == before.scan_lowerings);
  CHECK(after.window_tasks_started == before.window_tasks_started);

  run_ok("SET enable_duckdb_fallback=false");
  result = con->Query("SELECT i FROM r2a_routing");
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("Injected GPU scan certification unsupported") !=
        std::string::npos);
  run_ok("SET enable_duckdb_fallback=true");
  run_ok("SET sirius_test_inject_scan_verdict=''");

  auto prepared = con->Prepare("SELECT sum(i) FROM r2a_routing");
  REQUIRE_FALSE(prepared->HasError());
  for (int execution = 0; execution < 2; ++execution) {
    before                = sirius::test::get_transparent_execution_stats(*con);
    auto execution_result = prepared->Execute();
    REQUIRE_FALSE(execution_result->HasError());
    after = sirius::test::get_transparent_execution_stats(*con);
    CHECK(after.semantic_verdicts[0] == before.semantic_verdicts[0] + 1);
    CHECK(after.execution_rebuilds == before.execution_rebuilds);
    CHECK(after.executions == before.executions + 1);
    CHECK(after.window_tasks_started > before.window_tasks_started);
  }
  run_ok("SET sirius_test_inject_pin_registry_change=true");
  before = sirius::test::get_transparent_execution_stats(*con);
  run_ok("SELECT sum(i) FROM r2a_routing");
  after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.execution_rebuilds == before.execution_rebuilds + 1);
  CHECK(after.semantic_verdicts[0] == before.semantic_verdicts[0] + 2);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a rebuild latches verdict options again and declines at runtime",
                 "[transparent][verdict][integration]")
{
  run_ok("CREATE TABLE r2a_rebuild(i INTEGER)");
  run_ok("INSERT INTO r2a_rebuild VALUES (1),(2)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET sirius_test_inject_pin_registry_change=true");
  auto before  = sirius::test::get_transparent_execution_stats(*con);
  auto pending = con->PendingQuery("SELECT sum(i) FROM r2a_rebuild");
  REQUIRE_FALSE(pending->HasError());
  // Planning is complete and no execution task is running. Change only the local
  // option via DuckDB's pinned API so the active pending statement is not replaced.
  auto& config = duckdb::DBConfig::GetConfig(*con->context);
  duckdb::optional_ptr<const duckdb::ConfigurationOption> option;
  auto index = config.TryGetSettingIndex("sirius_test_inject_scan_verdict", option);
  REQUIRE(index.IsValid());
  con->context->config.user_settings.SetUserSetting(index.GetIndex(), duckdb::Value("unsupported"));
  auto result = pending->Execute();
  REQUIRE_FALSE(result->HasError());
  auto chunk = result->Fetch();
  REQUIRE(chunk);
  CHECK(chunk->GetValue(0, 0).GetValue<int64_t>() == 3);
  auto after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.execution_rebuilds == before.execution_rebuilds + 1);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.semantic_verdicts[0] == before.semantic_verdicts[0] + 1);
  CHECK(after.semantic_verdicts[1] == before.semantic_verdicts[1] + 1);
  CHECK(after.window_tasks_started == before.window_tasks_started);
  CHECK(after.setting_lookups_per_attempt == before.setting_lookups_per_attempt + 32);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a production budget exceedance still executes on GPU",
                 "[transparent][verdict][integration]")
{
  run_ok("CREATE TABLE r2a_measured(i INTEGER)");
  run_ok("INSERT INTO r2a_measured VALUES (1),(2)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET sirius_test_inject_certification_delay_ms=51");
  run_ok("SET sirius_test_inject_certification_bytes=8388609");
  run_ok("SET sirius_test_budget_declines=false");
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query("SELECT sum(i) FROM r2a_measured");
  REQUIRE_FALSE(result->HasError());
  CHECK(result->GetValue(0, 0).GetValue<int64_t>() == 3);
  auto after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.budget_exceeded[0] == before.budget_exceeded[0] + 1);
  CHECK(after.budget_exceeded[1] == before.budget_exceeded[1] + 1);
  CHECK(after.semantic_verdicts[0] == before.semantic_verdicts[0] + 1);
  CHECK(after.executions == before.executions + 1);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(after.window_tasks_started > before.window_tasks_started);
  CHECK(after.semantic_declines[static_cast<std::size_t>(verdict_reason::budget_time)] ==
        before.semantic_declines[static_cast<std::size_t>(verdict_reason::budget_time)]);
  CHECK(after.semantic_declines[static_cast<std::size_t>(verdict_reason::budget_bytes)] ==
        before.semantic_declines[static_cast<std::size_t>(verdict_reason::budget_bytes)]);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a encrypted native storage refuses before lowering and direct pin",
                 "[transparent][verdict][integration]")
{
  auto path = temp_db_path + ".encrypted";
  struct remove_fixture {
    std::string path;
    duckdb::Connection& connection;
    ~remove_fixture()
    {
      connection.Query("DETACH r2a_encrypted");
      connection.Query("SET force_mbedtls_unsafe=false");
      std::filesystem::remove(path);
      std::filesystem::remove(path + ".wal");
    }
  } cleanup{path, *con};
  // This disposable fixture tests encrypted-storage admission, not cryptography.
  // The test binary's built-in crypto module requires opt-in for fixture writes.
  run_ok("SET force_mbedtls_unsafe=true");
  run_ok("ATTACH '" + path + "' AS r2a_encrypted (ENCRYPTION_KEY 'r2a-fixture-only')");
  run_ok("CREATE TABLE r2a_encrypted.t(i INTEGER)");
  run_ok("INSERT INTO r2a_encrypted.t VALUES (7)");
  run_ok("CHECKPOINT r2a_encrypted");
  run_ok("DETACH r2a_encrypted");
  run_ok("ATTACH '" + path + "' AS r2a_encrypted (ENCRYPTION_KEY 'r2a-fixture-only', READ_ONLY)");
  run_ok("SET force_mbedtls_unsafe=false");
  run_ok("SET gpu_execution=true");
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query("SELECT i FROM r2a_encrypted.t");
  REQUIRE_FALSE(result->HasError());
  CHECK(result->GetValue(0, 0).GetValue<int32_t>() == 7);
  auto after  = sirius::test::get_transparent_execution_stats(*con);
  auto reason = static_cast<std::size_t>(verdict_reason::native_encrypted);
  CHECK(after.semantic_declines[reason] == before.semantic_declines[reason] + 1);
  CHECK(after.fallbacks == before.fallbacks + 1);
  CHECK(after.scan_lowerings == before.scan_lowerings);
  CHECK(after.window_tasks_started == before.window_tasks_started);
  auto pin = con->Query("CALL pin_table(format='duckdb', name='r2a_encrypted.main.t', tier='gpu')");
  REQUIRE(pin->HasError());
  CHECK(pin->GetError().find("encrypted storage is not GPU-decodable") != std::string::npos);
  run_ok("DETACH r2a_encrypted");
}
