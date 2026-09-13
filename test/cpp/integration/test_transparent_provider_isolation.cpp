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

#include "scan/dict_fsst_fixture.hpp"
#include "transparent/connection_provenance.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/operator/logical_get.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

using sirius::test::get_transparent_execution_stats;
using sirius::test::dict_fsst::query;
using sirius::transparent::connection_provenance;
using sirius::transparent::decline_reason;

struct isolation_fixture {
  sirius::test::dict_fsst::fixture disk;
  duckdb::Connection a;
  duckdb::Connection b;
  std::string hidden;

  isolation_fixture()
    : a(sirius::test::g_integration_env->make_connection()),
      b(sirius::test::g_integration_env->make_connection())
  {
    hidden = "hidden_" + std::filesystem::path(disk.directory).filename().string();
    query(a, "SET GLOBAL gpu_execution = true");
    query(a, "ATTACH '" + disk.path + "' AS " + hidden + " (HIDDEN true)");
    query(a, "CREATE TABLE " + hidden + ".main.t(i INTEGER)");
    query(a, "INSERT INTO " + hidden + ".main.t VALUES (1), (2), (3)");
    query(a, "CHECKPOINT " + hidden);
  }

  ~isolation_fixture()
  {
    b.Query("USE memory");
    a.Query("DETACH " + hidden);
    a.Query("RESET GLOBAL gpu_execution");
  }

  void gpu(bool enabled)
  {
    query(a, std::string("SET GLOBAL gpu_execution = ") + (enabled ? "true" : "false"));
  }

  void provider()
  {
    query(b, "USE " + hidden);
    query(b, "SET SESSION catalog_error_max_schemas = 0");
  }

  static auto state(duckdb::Connection& con)
  {
    auto result = duckdb::get_sirius_connection_state(*con.context);
    REQUIRE(result);
    return result;
  }

  static void unchanged_gpu(duckdb::SiriusContext::transparent_execution_stats const& before,
                            duckdb::SiriusContext::transparent_execution_stats const& after)
  {
    sirius::test::require_transparent_execution_delta(before, after, 0, 0, 0);
  }

  auto hidden_query(std::string const& sql)
  {
    auto before = get_transparent_execution_stats(a);
    auto result = query(a, sql);
    auto after  = get_transparent_execution_stats(a);
    unchanged_gpu(before, after);
    REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips + 1);
    REQUIRE(after.provider_internal_skips == before.provider_internal_skips);
    REQUIRE(after.classification_failures == before.classification_failures);
    REQUIRE(state(a)->provenance() == connection_provenance::user);
    REQUIRE(state(a)->attempt_declined());
    REQUIRE(state(a)->attempt_decline_reason() == decline_reason::hidden_catalog);
    return result;
  }
};

struct drop_capture_info : duckdb::OptimizerExtensionInfo {
  duckdb::ClientContext* target = nullptr;
  bool visited                  = false;
};

void drop_capture(duckdb::OptimizerExtensionInput& input,
                  duckdb::unique_ptr<duckdb::LogicalOperator>&)
{
  auto& info = static_cast<drop_capture_info&>(*input.info);
  if (info.target != &input.context) { return; }
  info.visited = true;
  auto state   = duckdb::get_sirius_connection_state(input.context);
  if (state) { state->clear_captured_plan(); }
}

}  // namespace

TEST_CASE_METHOD(isolation_fixture,
                 "provider connection declines a table query with GPU on or off",
                 "[integration][transparent][provenance]")
{
  gpu(GENERATE(true, false));
  provider();
  auto before = get_transparent_execution_stats(b);
  auto result = query(b, "SELECT i FROM t ORDER BY i");
  auto after  = get_transparent_execution_stats(b);
  unchanged_gpu(before, after);
  REQUIRE(after.provider_internal_skips == before.provider_internal_skips + 1);
  REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips);
  REQUIRE(after.classification_failures == before.classification_failures);
  REQUIRE(state(b)->provenance() == connection_provenance::provider_internal);
  REQUIRE(state(b)->attempt_decline_reason() == decline_reason::provider_internal);
  REQUIRE(result->RowCount() == 3);
  for (int i = 0; i < 3; ++i) {
    REQUIRE(result->GetValue(0, i).GetValue<int32_t>() == i + 1);
  }
}

TEST_CASE_METHOD(isolation_fixture,
                 "provider classification covers queries without LogicalGet",
                 "[integration][transparent][provenance]")
{
  gpu(GENERATE(true, false));
  provider();
  auto before = get_transparent_execution_stats(b);
  auto result = query(b, "SELECT 42::INTEGER");
  auto after  = get_transparent_execution_stats(b);
  unchanged_gpu(before, after);
  REQUIRE(after.provider_internal_skips == before.provider_internal_skips + 1);
  REQUIRE(state(b)->provenance() == connection_provenance::provider_internal);
  REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 42);
}

TEST_CASE_METHOD(isolation_fixture,
                 "provider latch survives search path and GPU setting changes",
                 "[integration][transparent][provenance]")
{
  gpu(false);
  provider();
  query(b, "SELECT 1");
  REQUIRE(state(b)->provenance() == connection_provenance::provider_internal);
  query(b, "USE memory");
  query(b, "SET SESSION catalog_error_max_schemas = 100");
  gpu(true);
  auto before = get_transparent_execution_stats(b);
  query(b, "SELECT 42");
  auto after = get_transparent_execution_stats(b);
  unchanged_gpu(before, after);
  REQUIRE(after.provider_internal_skips == before.provider_internal_skips + 1);
  REQUIRE(state(b)->provenance() == connection_provenance::provider_internal);
}

TEST_CASE_METHOD(isolation_fixture,
                 "provider latch survives a catalog alias change inside the transaction",
                 "[integration][transparent][provenance]")
{
  auto const gpu_enabled = GENERATE(true, false);
  CAPTURE(gpu_enabled);
  gpu(gpu_enabled);
  auto const renamed = hidden + "_k";
  struct cleanup_alias {
    duckdb::Connection& user;
    duckdb::Connection& provider;
    std::string const& original;
    std::string const& renamed;
    ~cleanup_alias()
    {
      provider.Query("ROLLBACK");
      provider.Query("USE memory");
      // Catalog aliases survive ROLLBACK; cleanup must cover either side of the rename.
      user.Query("DETACH DATABASE IF EXISTS " + renamed);
      user.Query("DETACH DATABASE IF EXISTS " + original);
    }
  } guard{a, b, hidden, renamed};

  query(b, "BEGIN");
  query(b, "USE " + hidden);
  query(b, "ALTER DATABASE " + hidden + " SET ALIAS TO " + renamed);
  query(b, "SET SESSION catalog_error_max_schemas = 0");
  auto provider_query = [&](std::string const& sql) {
    auto before = get_transparent_execution_stats(b);
    auto result = query(b, sql);
    auto after  = get_transparent_execution_stats(b);
    unchanged_gpu(before, after);
    REQUIRE(after.provider_internal_skips == before.provider_internal_skips + 1);
    REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips);
    REQUIRE(after.classification_failures == before.classification_failures);
    REQUIRE(state(b)->provenance() == connection_provenance::provider_internal);
    REQUIRE(state(b)->attempt_declined());
    REQUIRE(state(b)->attempt_decline_reason() == decline_reason::provider_internal);
    return result;
  };
  auto result = provider_query("SELECT 42::INTEGER");
  REQUIRE(result->RowCount() == 1);
  REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 42);
  for (auto const& name : {hidden, renamed}) {
    CAPTURE(name);
    auto rows = provider_query("SELECT i FROM " + name + ".main.t ORDER BY i");
    REQUIRE(rows->RowCount() == 3);
    for (int row = 0; row < 3; ++row) {
      REQUIRE(rows->GetValue(0, row).GetValue<int32_t>() == row + 1);
    }
  }
  query(b, "ROLLBACK");

  auto before             = get_transparent_execution_stats(a);
  auto user_result        = query(a, "SELECT 42::INTEGER");
  auto after              = get_transparent_execution_stats(a);
  auto const expected_gpu = gpu_enabled ? 1u : 0u;
  sirius::test::require_transparent_execution_delta(before, after, expected_gpu, 0, expected_gpu);
  REQUIRE(user_result->RowCount() == 1);
  REQUIRE(user_result->GetValue(0, 0).GetValue<int32_t>() == 42);
  REQUIRE(after.provider_internal_skips == before.provider_internal_skips);
  REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips);
  REQUIRE(after.classification_failures == before.classification_failures);
  REQUIRE(state(a)->provenance() == connection_provenance::user);
  REQUIRE_FALSE(state(a)->attempt_declined());
}

TEST_CASE_METHOD(isolation_fixture,
                 "hidden catalog decline survives optimizer scan elimination",
                 "[integration][transparent][provenance]")
{
  gpu(GENERATE(true, false));
  auto shape     = GENERATE(0, 1, 2);
  auto prefix    = shape == 1 ? "SELECT count(*) FROM " : "SELECT i FROM ";
  auto predicate = shape == 2 ? " WHERE 1 = 0" : "";
  hidden_query(std::string(prefix) + hidden + ".main.t" + predicate);
}

TEST_CASE_METHOD(isolation_fixture,
                 "hidden catalog decline survives disabled extension optimizer hooks",
                 "[integration][transparent][provenance]")
{
  auto const gpu_enabled = GENERATE(true, false);
  gpu(gpu_enabled);
  struct reset_optimizers {
    duckdb::Connection& con;
    ~reset_optimizers() { con.Query("RESET disabled_optimizers"); }
  } guard{a};
  query(a, "SET disabled_optimizers = 'extension'");
  hidden_query("SELECT i FROM " + hidden + ".main.t");
  hidden_query("SELECT count(*) FROM " + hidden + ".main.t");

  auto before             = get_transparent_execution_stats(a);
  auto result             = query(a, "SELECT 42::INTEGER");
  auto after              = get_transparent_execution_stats(a);
  auto const expected_gpu = gpu_enabled ? 1u : 0u;
  sirius::test::require_transparent_execution_delta(before, after, expected_gpu, 0, expected_gpu);
  REQUIRE(result->RowCount() == 1);
  REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 42);
  REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips);
  REQUIRE(after.provider_internal_skips == before.provider_internal_skips);
  REQUIRE(after.classification_failures == before.classification_failures);
  REQUIRE(state(a)->provenance() == connection_provenance::user);
  REQUIRE_FALSE(state(a)->attempt_declined());
  REQUIRE(state(a)->attempt_decline_reason() == decline_reason::none);
}

TEST_CASE_METHOD(
  isolation_fixture,
  "hidden catalog decline survives a DETACH by another connection inside a transaction",
  "[integration][transparent][provenance]")
{
  gpu(GENERATE(true, false));
  struct reset_optimizers {
    duckdb::Connection& con;
    ~reset_optimizers() { con.Query("RESET disabled_optimizers"); }
  } optimizer_guard{a};
  query(a, "SET disabled_optimizers = 'extension'");
  query(a, "BEGIN");
  struct rollback_transaction {
    duckdb::Connection& con;
    ~rollback_transaction() { con.Query("ROLLBACK"); }
  } transaction_guard{a};

  REQUIRE(hidden_query("SELECT i FROM " + hidden + ".main.t")->RowCount() == 3);
  query(b, "DETACH " + hidden);
  auto rows = hidden_query("SELECT i FROM " + hidden + ".main.t ORDER BY i");
  REQUIRE(rows->RowCount() == 3);
  for (int row = 0; row < 3; ++row) {
    REQUIRE(rows->GetValue(0, row).GetValue<int32_t>() == row + 1);
  }
  auto count = hidden_query("SELECT count(*) FROM " + hidden + ".main.t");
  REQUIRE(count->RowCount() == 1);
  REQUIRE(count->GetValue(0, 0).GetValue<int64_t>() == 3);
  query(a, "ROLLBACK");
}

TEST_CASE_METHOD(isolation_fixture,
                 "statement inspection fails closed on a stale or unknown catalog identity",
                 "[integration][transparent][provenance]")
{
  using sirius::transparent::inspect_statement_for_hidden_catalog;
  auto& manager          = duckdb::DatabaseManager::Get(*a.context);
  auto hidden_database   = manager.GetDatabase(hidden);
  auto ordinary_database = manager.GetDatabase("memory");
  REQUIRE(hidden_database);
  REQUIRE(ordinary_database);
  auto const oid = hidden_database->oid;
  duckdb::StatementProperties props;

  props.read_databases["no_such_catalog"] = {12345, duckdb::optional_idx()};
  CHECK(inspect_statement_for_hidden_catalog(*a.context, props) ==
        decline_reason::classification_failed);
  props.read_databases.clear();
  props.read_databases[hidden] = {oid + 1, duckdb::optional_idx()};
  CHECK(inspect_statement_for_hidden_catalog(*a.context, props) ==
        decline_reason::classification_failed);
  props.read_databases[hidden] = {oid, duckdb::optional_idx()};
  CHECK(inspect_statement_for_hidden_catalog(*a.context, props) == decline_reason::hidden_catalog);
  props.read_databases.clear();
  props.modified_databases[hidden] = {{oid, duckdb::optional_idx()},
                                      duckdb::DatabaseModificationType::INSERT_DATA};
  CHECK(inspect_statement_for_hidden_catalog(*a.context, props) == decline_reason::hidden_catalog);
  props.modified_databases.clear();
  props.read_databases["memory"] = {ordinary_database->oid, duckdb::optional_idx()};
  CHECK(inspect_statement_for_hidden_catalog(*a.context, props) == decline_reason::none);
}

TEST_CASE_METHOD(isolation_fixture,
                 "hidden catalog decline is cleared for the next user attempt",
                 "[integration][transparent][provenance]")
{
  hidden_query("SELECT i FROM " + hidden + ".main.t");
  auto before = get_transparent_execution_stats(a);
  auto result = query(a, "SELECT 42::INTEGER");
  auto after  = get_transparent_execution_stats(a);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  REQUIRE_FALSE(state(a)->attempt_declined());
  REQUIRE(state(a)->provenance() == connection_provenance::user);
  REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 42);
}

TEST_CASE_METHOD(isolation_fixture,
                 "hidden prepared statement stays on CPU across executions",
                 "[integration][transparent][provenance]")
{
  gpu(GENERATE(true, false));
  auto before   = get_transparent_execution_stats(a);
  auto prepared = a.Prepare("SELECT i FROM " + hidden + ".main.t ORDER BY i");
  REQUIRE_FALSE(prepared->HasError());
  auto after_prepare = get_transparent_execution_stats(a);
  REQUIRE(after_prepare.hidden_catalog_skips == before.hidden_catalog_skips + 1);
  for (int n = 0; n < 2; ++n) {
    auto result = prepared->Execute();
    REQUIRE_FALSE(result->HasError());
    size_t rows = 0;
    while (auto chunk = result->Fetch()) {
      for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
        REQUIRE(chunk->GetValue(0, i).GetValue<int32_t>() == static_cast<int32_t>(++rows));
      }
    }
    REQUIRE_FALSE(result->HasError());
    REQUIRE(rows == 3);
  }
  unchanged_gpu(before, get_transparent_execution_stats(a));
}

TEST_CASE_METHOD(isolation_fixture,
                 "missing capture cannot bypass an earlier hidden catalog decline",
                 "[integration][transparent][provenance]")
{
  gpu(GENERATE(true, false));
  auto info    = duckdb::make_shared_ptr<drop_capture_info>();
  info->target = a.context.get();
  struct disarm {
    drop_capture_info& info;
    ~disarm() { info.target = nullptr; }
  } guard{*info};
  duckdb::OptimizerExtension extension;
  extension.optimizer_info    = info;
  extension.optimize_function = drop_capture;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*a.context),
                                       std::move(extension));
  hidden_query("SELECT count(*) FROM " + hidden + ".main.t");
  REQUIRE(info->visited);
}

TEST_CASE_METHOD(isolation_fixture,
                 "hidden default without provider setting remains a user connection",
                 "[integration][transparent][provenance]")
{
  gpu(GENERATE(true, false));
  query(b, "USE " + hidden);
  query(b, "SET SESSION catalog_error_max_schemas = 100");
  auto before = get_transparent_execution_stats(b);
  query(b, "SELECT i FROM t");
  auto after = get_transparent_execution_stats(b);
  unchanged_gpu(before, after);
  REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips + 1);
  REQUIRE(after.provider_internal_skips == before.provider_internal_skips);
  REQUIRE(state(b)->provenance() == connection_provenance::user);
}

TEST_CASE("hidden catalog inspection fails closed when get_bind_info throws",
          "[transparent][provenance]")
{
  duckdb::TableFunction function;
  function.get_bind_info = [](duckdb::optional_ptr<duckdb::FunctionData>) -> duckdb::BindInfo {
    throw std::runtime_error("injected get_bind_info failure");
  };
  duckdb::LogicalGet get(0, std::move(function), nullptr, {duckdb::LogicalType::INTEGER}, {"i"});
  REQUIRE(sirius::transparent::inspect_plan_for_hidden_catalog(get) ==
          decline_reason::classification_failed);
}

TEST_CASE_METHOD(isolation_fixture,
                 "provenance classification failure declines each attempt and then recovers",
                 "[integration][transparent][provenance]")
{
  gpu(GENERATE(true, false));
  query(b, "SELECT 42::INTEGER");
  auto connection_state = state(b);
  auto const provenance = connection_state->provenance();
  REQUIRE(provenance == connection_provenance::user);
  struct reset_fault {
    duckdb::Connection& con;
    ~reset_fault() { con.Query("RESET SESSION sirius_test_inject_provenance_failure"); }
  } guard{b};
  query(b, "SET SESSION sirius_test_inject_provenance_failure = true");
  for (int attempt = 0; attempt < 2; ++attempt) {
    CAPTURE(attempt);
    auto before = get_transparent_execution_stats(b);
    auto result = query(b, "SELECT 42::INTEGER");
    auto after  = get_transparent_execution_stats(b);
    REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 42);
    unchanged_gpu(before, after);
    REQUIRE(after.classification_failures == before.classification_failures + 1);
    REQUIRE(after.provider_internal_skips == before.provider_internal_skips);
    REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips);
    REQUIRE(connection_state->provenance() == provenance);
    REQUIRE(connection_state->attempt_declined());
    REQUIRE(connection_state->attempt_decline_reason() == decline_reason::classification_failed);
  }
  query(b, "SET SESSION sirius_test_inject_provenance_failure = false");
  gpu(true);
  auto before = get_transparent_execution_stats(b);
  auto result = query(b, "SELECT 42::INTEGER");
  auto after  = get_transparent_execution_stats(b);
  REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 42);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  REQUIRE(after.classification_failures == before.classification_failures);
  REQUIRE(after.provider_internal_skips == before.provider_internal_skips);
  REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips);
  REQUIRE(connection_state->provenance() == provenance);
  REQUIRE_FALSE(connection_state->attempt_declined());
  REQUIRE(connection_state->attempt_decline_reason() == decline_reason::none);
}

namespace {

bool load_ducklake(duckdb::Connection& con)
{
  auto result = con.Query("LOAD ducklake");
  if (!result || result->HasError()) {
    auto required = std::getenv("SIRIUS_TEST_REQUIRE_DUCKLAKE");
    if (required && std::string(required) == "1") {
      FAIL("required DuckLake LOAD failed: " << (result ? result->GetError() : "no result"));
    }
    WARN("DuckLake not covered");
    return false;
  }
  auto version = query(con,
                       "SELECT extension_version FROM duckdb_extensions() WHERE extension_name = "
                       "'ducklake' AND loaded");
  REQUIRE(version->RowCount() == 1);
  REQUIRE(version->GetValue(0, 0).ToString() == "d8a1881e");
  return true;
}

void attach_lake(duckdb::Connection& con, std::string const& directory)
{
  query(con,
        "ATTACH 'ducklake:" + directory + "/meta.ducklake' AS lake (DATA_PATH '" + directory +
          "/data/')");
}

void populate_lake(duckdb::Connection& con)
{
  query(con, "CREATE TABLE lake.t(id INTEGER)");
  query(con, "INSERT INTO lake.t SELECT i FROM range(100) r(i)");
  query(con, "INSERT INTO lake.t VALUES (100), (101)");
  query(con, "DELETE FROM lake.t WHERE id = 5");
  query(con, "ALTER TABLE lake.t ADD COLUMN extra INTEGER DEFAULT 7");
  query(con, "INSERT INTO lake.t SELECT i, 7 FROM range(102, 132) r(i)");
  query(con, "CALL ducklake_flush_inlined_data('lake')");
}

void check_lake_rows(duckdb::Connection& con)
{
  auto rows = query(con, "SELECT id, extra FROM lake.t ORDER BY id");
  REQUIRE(rows->RowCount() == 131);
  for (int row = 0; row < 131; ++row) {
    REQUIRE(rows->GetValue(0, row).GetValue<int32_t>() == (row < 5 ? row : row + 1));
    REQUIRE(rows->GetValue(1, row).GetValue<int32_t>() == 7);
  }
  auto aggregates = query(con, "SELECT count(*), sum(id), sum(extra) FROM lake.t");
  REQUIRE(aggregates->GetValue(0, 0).GetValue<int64_t>() == 131);
  REQUIRE(aggregates->GetValue(1, 0).GetValue<int64_t>() == 8641);
  REQUIRE(aggregates->GetValue(2, 0).GetValue<int64_t>() == 917);
}

auto lake_config() { return std::filesystem::path(__FILE__).parent_path() / "integration.yaml"; }

}  // namespace

TEST_CASE("DuckLake provider SQL declines while ordinary user SQL remains GPU eligible",
          "[transparent][provenance][ducklake]")
{
  sirius::test::dict_fsst::fixture disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) { return; }
  query(con, "SET GLOBAL gpu_execution = true");
  auto before = get_transparent_execution_stats(con);
  attach_lake(con, disk.directory);
  populate_lake(con);
  query(con, "SELECT id, extra FROM lake.t ORDER BY id");
  auto after = get_transparent_execution_stats(con);
  REQUIRE(after.provider_internal_skips > before.provider_internal_skips);
  REQUIRE(after.successful_rebinds == before.successful_rebinds);
  REQUIRE(after.executions == before.executions);
  REQUIRE(after.classification_failures == before.classification_failures);
  before = after;
  query(con, "SELECT 42::INTEGER");
  after = get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  check_lake_rows(con);
}

TEST_CASE("DuckLake flush preserves rows and file paths after closing every connection",
          "[transparent][provenance][ducklake]")
{
  sirius::test::dict_fsst::fixture disk;
  {
    sirius::test::shared_test_env env(lake_config());
    auto con = env.make_connection();
    if (!load_ducklake(con)) { return; }
    query(con, "SET GLOBAL gpu_execution = true");
    attach_lake(con, disk.directory);
    populate_lake(con);
    check_lake_rows(con);
    query(con, "DETACH lake");
  }
  {
    sirius::test::shared_test_env env(lake_config());
    auto con = env.make_connection();
    REQUIRE(load_ducklake(con));
    query(con, "SET GLOBAL gpu_execution = true");
    attach_lake(con, disk.directory);
    check_lake_rows(con);
    auto files = query(con, "SELECT data_file FROM ducklake_list_files('lake', 't')");
    REQUIRE(files->RowCount() > 0);
    for (duckdb::idx_t row = 0; row < files->RowCount(); ++row) {
      REQUIRE_FALSE(files->GetValue(0, row).IsNull());
      REQUIRE(std::filesystem::is_regular_file(files->GetValue(0, row).ToString()));
    }
    query(con, "DETACH lake");
    query(con, "ATTACH '" + disk.directory + "/meta.ducklake' AS meta (READ_ONLY)");
    query(con, "SET SESSION gpu_execution = false");
    auto paths = query(
      con,
      "SELECT f.path, f.path_is_relative, t.path, t.path_is_relative, s.path, s.path_is_relative "
      "FROM meta.ducklake_data_file f JOIN meta.ducklake_table t USING (table_id) "
      "JOIN meta.ducklake_schema s USING (schema_id) "
      "WHERE t.end_snapshot IS NULL AND s.end_snapshot IS NULL");
    REQUIRE(paths->RowCount() > 0);
    for (duckdb::idx_t row = 0; row < paths->RowCount(); ++row) {
      REQUIRE_FALSE(paths->GetValue(0, row).IsNull());
      auto path = std::filesystem::path(paths->GetValue(0, row).ToString());
      if (paths->GetValue(1, row).GetValue<bool>()) {
        auto table_path = std::filesystem::path(paths->GetValue(2, row).ToString());
        if (paths->GetValue(3, row).GetValue<bool>()) {
          auto schema_path = std::filesystem::path(paths->GetValue(4, row).ToString());
          if (paths->GetValue(5, row).GetValue<bool>()) {
            schema_path = std::filesystem::path(disk.directory) / "data" / schema_path;
          }
          table_path = schema_path / table_path;
        }
        path = table_path / path;
      }
      REQUIRE(std::filesystem::is_regular_file(path));
    }
  }
}
