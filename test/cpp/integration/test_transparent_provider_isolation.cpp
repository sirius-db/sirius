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

#include "ducklake/binding_bridge.hpp"
#include "ducklake/scan_ingestible.hpp"
#include "exec/streaming_fragment.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "transparent/connection_provenance.hpp"
#include "utils/dynamic_filter_test_utils.hpp"
#include "utils/scoped_temp_directory.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/main/client_config.hpp>
#include <duckdb/main/client_context_state.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <set>
#include <string>

namespace {

using sirius::test::get_transparent_execution_stats;
using sirius::test::query;
using sirius::transparent::connection_provenance;
using sirius::transparent::decline_reason;

struct isolation_fixture {
  sirius::test::scoped_temp_directory disk;
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
                 "provider marker alone latches before the first statement",
                 "[integration][transparent][provenance][provider_marker]")
{
  gpu(GENERATE(true, false));
  auto const catalog = hidden + "_user";
  auto const table   = catalog + ".t";
  struct detach_catalog {
    duckdb::Connection& con;
    std::string const& name;
    ~detach_catalog() { con.Query("DETACH DATABASE IF EXISTS " + name); }
  } guard{a, catalog};
  query(a, "ATTACH '" + disk.directory + "/user.duckdb' AS " + catalog);
  query(a, "CREATE TABLE " + table + " AS SELECT i::INTEGER AS i FROM range(3) r(i)");
  query(a, "CHECKPOINT " + catalog);
  REQUIRE(state(b)->provenance() == connection_provenance::unclassified);
  b.context->registered_state->Insert("ducklake_internal_connection",
                                      duckdb::make_shared_ptr<duckdb::ClientContextState>());

  for (int attempt = 0; attempt < 2; ++attempt) {
    CAPTURE(attempt);
    auto before = get_transparent_execution_stats(b);
    auto result = query(b, "SELECT i FROM " + table + " ORDER BY i");
    auto after  = get_transparent_execution_stats(b);
    unchanged_gpu(before, after);
    REQUIRE(after.provider_internal_skips == before.provider_internal_skips + 1);
    REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips);
    REQUIRE(after.classification_failures == before.classification_failures);
    REQUIRE(state(b)->provenance() == connection_provenance::provider_internal);
    REQUIRE(state(b)->attempt_declined());
    REQUIRE(state(b)->attempt_decline_reason() == decline_reason::provider_internal);
    REQUIRE(result->RowCount() == 3);
    for (int row = 0; row < 3; ++row) {
      REQUIRE(result->Collection().GetValue(0, row).GetValue<int32_t>() == row);
    }
    b.context->registered_state->Remove("ducklake_internal_connection");
  }
}

TEST_CASE_METHOD(isolation_fixture,
                 "provider marker added after a user statement is honoured on the next attempt",
                 "[integration][transparent][provenance][provider_marker]")
{
  auto c      = sirius::test::g_integration_env->make_connection();
  auto before = get_transparent_execution_stats(c);
  auto result = query(c, "SELECT 42::INTEGER");
  auto user   = get_transparent_execution_stats(c);
  sirius::test::require_transparent_execution_delta(before, user, 1, 0, 1);
  REQUIRE(result->Collection().GetValue(0, 0).GetValue<int32_t>() == 42);
  REQUIRE(state(c)->provenance() == connection_provenance::user);
  REQUIRE_FALSE(state(c)->attempt_declined());
  REQUIRE(user.provider_internal_skips == before.provider_internal_skips);
  REQUIRE(user.hidden_catalog_skips == before.hidden_catalog_skips);
  REQUIRE(user.classification_failures == before.classification_failures);
  c.context->registered_state->Insert("ducklake_internal_connection",
                                      duckdb::make_shared_ptr<duckdb::ClientContextState>());

  for (int attempt = 0; attempt < 2; ++attempt) {
    CAPTURE(attempt);
    auto previous = get_transparent_execution_stats(c);
    auto marked   = query(c, "SELECT 43::INTEGER");
    auto after    = get_transparent_execution_stats(c);
    unchanged_gpu(previous, after);
    REQUIRE(after.successful_rebinds == user.successful_rebinds);
    REQUIRE(after.provider_internal_skips == previous.provider_internal_skips + 1);
    REQUIRE(after.hidden_catalog_skips == previous.hidden_catalog_skips);
    REQUIRE(after.classification_failures == previous.classification_failures);
    REQUIRE(state(c)->provenance() == connection_provenance::provider_internal);
    REQUIRE(state(c)->attempt_declined());
    REQUIRE(state(c)->attempt_decline_reason() == decline_reason::provider_internal);
    REQUIRE(marked->Collection().GetValue(0, 0).GetValue<int32_t>() == 43);
  }
}

TEST_CASE_METHOD(isolation_fixture,
                 "provider marker takes precedence over a hidden catalog without the fingerprint",
                 "[integration][transparent][provenance][provider_marker]")
{
  gpu(GENERATE(true, false));
  query(b, "USE " + hidden);
  query(b, "SET SESSION catalog_error_max_schemas = 100");
  REQUIRE(state(b)->provenance() == connection_provenance::user);
  b.context->registered_state->Insert("ducklake_internal_connection",
                                      duckdb::make_shared_ptr<duckdb::ClientContextState>());
  auto before = get_transparent_execution_stats(b);
  auto result = query(b, "SELECT i FROM t ORDER BY i");
  auto after  = get_transparent_execution_stats(b);
  unchanged_gpu(before, after);
  REQUIRE(after.provider_internal_skips == before.provider_internal_skips + 1);
  REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips);
  REQUIRE(after.classification_failures == before.classification_failures);
  REQUIRE(state(b)->provenance() == connection_provenance::provider_internal);
  REQUIRE(state(b)->attempt_declined());
  REQUIRE(state(b)->attempt_decline_reason() == decline_reason::provider_internal);
  REQUIRE(result->RowCount() == 3);
  for (int row = 0; row < 3; ++row) {
    REQUIRE(result->Collection().GetValue(0, row).GetValue<int32_t>() == row + 1);
  }
}

TEST_CASE_METHOD(isolation_fixture,
                 "provenance failure precedes a provider marker without latching the connection",
                 "[integration][transparent][provenance][provider_marker]")
{
  gpu(GENERATE(true, false));
  duckdb::ExtensionOption fault;
  REQUIRE(duckdb::DBConfig::GetConfig(*b.context)
            .TryGetExtensionOption("sirius_test_inject_provenance_failure", fault));
  auto& settings = duckdb::ClientConfig::GetConfig(*b.context).user_settings;
  // Configure the fault without planning a SET statement on the fresh connection.
  settings.SetUserSetting(fault.setting_index.GetIndex(), duckdb::Value::BOOLEAN(true));
  b.context->registered_state->Insert("ducklake_internal_connection",
                                      duckdb::make_shared_ptr<duckdb::ClientContextState>());
  REQUIRE(state(b)->provenance() == connection_provenance::unclassified);
  for (int attempt = 0; attempt < 2; ++attempt) {
    CAPTURE(attempt);
    auto before = get_transparent_execution_stats(b);
    auto result = query(b, "SELECT 42::INTEGER");
    auto after  = get_transparent_execution_stats(b);
    unchanged_gpu(before, after);
    REQUIRE(after.classification_failures == before.classification_failures + 1);
    REQUIRE(after.provider_internal_skips == before.provider_internal_skips);
    REQUIRE(after.hidden_catalog_skips == before.hidden_catalog_skips);
    REQUIRE(state(b)->provenance() == connection_provenance::unclassified);
    REQUIRE(state(b)->attempt_declined());
    REQUIRE(state(b)->attempt_decline_reason() == decline_reason::classification_failed);
    REQUIRE(result->Collection().GetValue(0, 0).GetValue<int32_t>() == 42);
  }
}

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
    REQUIRE(result->Collection().GetValue(0, i).GetValue<int32_t>() == i + 1);
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
  REQUIRE(result->Collection().GetValue(0, 0).GetValue<int32_t>() == 42);
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
  REQUIRE(result->Collection().GetValue(0, 0).GetValue<int32_t>() == 42);
  for (auto const& name : {hidden, renamed}) {
    CAPTURE(name);
    auto rows = provider_query("SELECT i FROM " + name + ".main.t ORDER BY i");
    REQUIRE(rows->RowCount() == 3);
    for (int row = 0; row < 3; ++row) {
      REQUIRE(rows->Collection().GetValue(0, row).GetValue<int32_t>() == row + 1);
    }
  }
  query(b, "ROLLBACK");

  auto before             = get_transparent_execution_stats(a);
  auto user_result        = query(a, "SELECT 42::INTEGER");
  auto after              = get_transparent_execution_stats(a);
  auto const expected_gpu = gpu_enabled ? 1u : 0u;
  sirius::test::require_transparent_execution_delta(before, after, expected_gpu, 0, expected_gpu);
  REQUIRE(user_result->RowCount() == 1);
  REQUIRE(user_result->Collection().GetValue(0, 0).GetValue<int32_t>() == 42);
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
  sirius::test::disabled_optimizers_guard const guard(a, "extension");
  hidden_query("SELECT i FROM " + hidden + ".main.t");
  hidden_query("SELECT count(*) FROM " + hidden + ".main.t");

  auto before             = get_transparent_execution_stats(a);
  auto result             = query(a, "SELECT 42::INTEGER");
  auto after              = get_transparent_execution_stats(a);
  auto const expected_gpu = gpu_enabled ? 1u : 0u;
  sirius::test::require_transparent_execution_delta(before, after, expected_gpu, 0, expected_gpu);
  REQUIRE(result->RowCount() == 1);
  REQUIRE(result->Collection().GetValue(0, 0).GetValue<int32_t>() == 42);
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
  sirius::test::disabled_optimizers_guard const optimizer_guard(a, "extension");
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
    REQUIRE(rows->Collection().GetValue(0, row).GetValue<int32_t>() == row + 1);
  }
  auto count = hidden_query("SELECT count(*) FROM " + hidden + ".main.t");
  REQUIRE(count->RowCount() == 1);
  REQUIRE(count->Collection().GetValue(0, 0).GetValue<int64_t>() == 3);
  query(a, "ROLLBACK");
}

TEST_CASE_METHOD(isolation_fixture,
                 "statement inspection fails closed on a stale or unknown catalog identity",
                 "[integration][transparent][provenance]")
{
  using sirius::transparent::inspect_statement_for_hidden_catalog;
  auto& manager          = duckdb::DatabaseManager::Get(*a.context);
  auto hidden_database   = manager.GetDatabase(duckdb::Identifier(hidden));
  auto ordinary_database = manager.GetDatabase("memory");
  REQUIRE(hidden_database);
  REQUIRE(ordinary_database);
  auto const oid = hidden_database->oid;
  duckdb::StatementProperties props;

  props.read_databases["no_such_catalog"] = {12345, duckdb::optional_idx()};
  CHECK(inspect_statement_for_hidden_catalog(*a.context, props) ==
        decline_reason::classification_failed);
  props.read_databases.clear();
  props.read_databases[duckdb::Identifier(hidden)] = {oid + 1, duckdb::optional_idx()};
  CHECK(inspect_statement_for_hidden_catalog(*a.context, props) ==
        decline_reason::classification_failed);
  props.read_databases[duckdb::Identifier(hidden)] = {oid, duckdb::optional_idx()};
  CHECK(inspect_statement_for_hidden_catalog(*a.context, props) == decline_reason::hidden_catalog);
  props.read_databases.clear();
  props.modified_databases[duckdb::Identifier(hidden)] = {
    {oid, duckdb::optional_idx()}, duckdb::DatabaseModificationType::INSERT_DATA};
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
  REQUIRE(result->Collection().GetValue(0, 0).GetValue<int32_t>() == 42);
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

TEST_CASE("hidden catalog inspection fails closed when get_table_entry throws",
          "[transparent][provenance]")
{
  duckdb::TableFunction function;
  function.get_table_entry = [](duckdb::optional_ptr<const duckdb::FunctionData>)
    -> duckdb::optional_ptr<duckdb::TableCatalogEntry> {
    throw std::runtime_error("injected get_table_entry failure");
  };
  duckdb::LogicalGet get(duckdb::TableIndex(0),
                         duckdb::BoundTableFunction(function),
                         nullptr,
                         {duckdb::LogicalType::INTEGER},
                         {"i"});
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
    REQUIRE(result->Collection().GetValue(0, 0).GetValue<int32_t>() == 42);
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
  REQUIRE(result->Collection().GetValue(0, 0).GetValue<int32_t>() == 42);
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
  auto path = std::getenv("SIRIUS_TEST_DUCKLAKE_EXTENSION");
  auto result =
    con.Query(path && *path ? "LOAD " + duckdb::Value(path).ToSQLString() : "LOAD ducklake");
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
  REQUIRE(version->Collection().GetValue(0, 0).ToString() == "80e6155e41");
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
    REQUIRE(rows->Collection().GetValue(0, row).GetValue<int32_t>() == (row < 5 ? row : row + 1));
    REQUIRE(rows->Collection().GetValue(1, row).GetValue<int32_t>() == 7);
  }
  auto aggregates = query(con, "SELECT count(*), sum(id), sum(extra) FROM lake.t");
  REQUIRE(aggregates->Collection().GetValue(0, 0).GetValue<int64_t>() == 131);
  REQUIRE(aggregates->Collection().GetValue(1, 0).GetValue<int64_t>() == 8641);
  REQUIRE(aggregates->Collection().GetValue(2, 0).GetValue<int64_t>() == 917);
}

struct provider_query_observer : duckdb::OptimizerExtensionInfo {
  std::function<void(duckdb::ClientContext&)> observe;
  std::function<void(duckdb::ClientContext&, duckdb::LogicalOperator&)> inspect_plan;
};
void observe_provider_query(duckdb::OptimizerExtensionInput& input,
                            duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  auto& observer = static_cast<provider_query_observer&>(*input.info);
  if (observer.observe) observer.observe(input.context);
  if (observer.inspect_plan) observer.inspect_plan(input.context, *plan);
}

auto lake_config() { return sirius::test::integration_config_path(); }

}  // namespace

TEST_CASE("DuckLake provider SQL declines while ordinary user SQL remains GPU eligible",
          "[transparent][provenance][ducklake][isolated_context]")
{
  sirius::test::scoped_temp_directory disk;
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
  auto expected_executions = sirius::ducklake::bridge_available() ? 1 : 0;
  REQUIRE(after.successful_rebinds == before.successful_rebinds + expected_executions);
  REQUIRE(after.executions == before.executions + expected_executions);
  REQUIRE(after.classification_failures == before.classification_failures);
  before = after;
  query(con, "SELECT 42::INTEGER");
  after = get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  check_lake_rows(con);
}

TEST_CASE("DuckLake flush preserves rows and file paths after closing every connection",
          "[transparent][provenance][ducklake][isolated_context]")
{
  sirius::test::scoped_temp_directory disk;
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
      REQUIRE_FALSE(files->Collection().GetValue(0, row).IsNull());
      REQUIRE(std::filesystem::is_regular_file(files->Collection().GetValue(0, row).ToString()));
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
      REQUIRE_FALSE(paths->Collection().GetValue(0, row).IsNull());
      auto path = std::filesystem::path(paths->Collection().GetValue(0, row).ToString());
      if (paths->Collection().GetValue(1, row).GetValue<bool>()) {
        auto table_path = std::filesystem::path(paths->Collection().GetValue(2, row).ToString());
        if (paths->Collection().GetValue(3, row).GetValue<bool>()) {
          auto schema_path = std::filesystem::path(paths->Collection().GetValue(4, row).ToString());
          if (paths->Collection().GetValue(5, row).GetValue<bool>()) {
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

TEST_CASE("DuckLake SQL preserves snapshot and transaction visibility",
          "[transparent][provenance][ducklake][isolated_context]")
{
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  query(con, "SET GLOBAL gpu_execution = true");
  attach_lake(con, disk.directory);
  query(con, "CREATE TABLE lake.t(id BIGINT)");
  query(con, "INSERT INTO lake.t VALUES (1), (2)");
  auto snapshot  = query(con,
                        "SELECT snapshot_id, snapshot_time FROM ducklake_snapshots('lake') ORDER "
                         "BY snapshot_id DESC LIMIT 1");
  auto version   = snapshot->Collection().GetValue(0, 0).ToString();
  auto timestamp = snapshot->Collection().GetValue(1, 0).ToSQLString();
  query(con, "INSERT INTO lake.t VALUES (3)");
  query(con, "BEGIN");
  query(con, "INSERT INTO lake.t VALUES (4)");
  query(con, "DELETE FROM lake.t WHERE id = 1");
  auto before  = get_transparent_execution_stats(con);
  auto current = query(con, "SELECT sum(id)::BIGINT FROM lake.t");
  REQUIRE(current->Collection().GetValue(0, 0).GetValue<int64_t>() == 9);
  for (auto const& selector : {"VERSION => " + version, "TIMESTAMP => " + timestamp}) {
    // Historical reads can still include transaction-local changes. Preserve the
    // provider's visibility rather than assuming a committed-only snapshot.
    auto sql = "SELECT sum(id)::BIGINT FROM lake.t AT (" + selector + ")";
    query(con, "SET SESSION gpu_execution = false");
    auto baseline = query(con, sql);
    query(con, "SET SESSION gpu_execution = true");
    auto selected = query(con, sql);
    REQUIRE(selected->Collection().GetValue(0, 0) == baseline->Collection().GetValue(0, 0));
  }
  auto prepared = con.Prepare("SELECT sum(id)::BIGINT FROM lake.t WHERE id >= ?");
  REQUIRE_FALSE(prepared->HasError());
  for (auto threshold : {2, 3}) {
    auto result = prepared->Execute(threshold);
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->Collection().GetValue(0, 0).GetValue<int64_t>() == (threshold == 2 ? 9 : 7));
  }
  auto after               = get_transparent_execution_stats(con);
  auto expected_executions = sirius::ducklake::bridge_available() ? 5 : 0;
  REQUIRE(after.executions == before.executions + expected_executions);
  REQUIRE(after.successful_rebinds == before.successful_rebinds + expected_executions);
  query(con, "ROLLBACK");
  auto committed = query(con, "SELECT sum(id)::BIGINT FROM lake.t");
  REQUIRE(committed->Collection().GetValue(0, 0).GetValue<int64_t>() == 6);
}

TEST_CASE("DuckLake unsupported columns respect explicit and transparent fallback settings",
          "[transparent][provenance][ducklake][isolated_context]")
{
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  query(con, "SET GLOBAL gpu_execution = true");
  attach_lake(con, disk.directory);
  query(con, "CREATE TABLE lake.t(id UUID)");
  auto value = std::string("550e8400-e29b-41d4-a716-446655440000");
  query(con, "INSERT INTO lake.t VALUES ('" + value + "')");
  auto sql          = std::string("SELECT id FROM lake.t");
  auto explicit_sql = "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")";
  for (auto const& statement : {sql, explicit_sql}) {
    query(con, "SET enable_duckdb_fallback = true");
    auto result = query(con, statement);
    REQUIRE(result->RowCount() == 1);
    REQUIRE(result->Collection().GetValue(0, 0).ToString() == value);
    query(con, "SET enable_duckdb_fallback = false");
    REQUIRE(con.Query(statement)->HasError());
  }
}

namespace {
std::vector<std::string> ducklake_result_bag(duckdb::QueryResult& result)
{
  if (result.HasError()) throw std::runtime_error(result.GetError());
  std::vector<std::string> rows;
  while (auto chunk = result.Fetch()) {
    for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
      std::string value;
      for (duckdb::idx_t column = 0; column < chunk->ColumnCount(); ++column)
        value += chunk->GetValue(column, row).ToString() + "|";
      rows.push_back(std::move(value));
    }
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}
}  // namespace
TEST_CASE("DuckLake Parquet scans honor the configured batch target",
          "[isolated_context][ducklake][ducklake_sql]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto target = GENERATE(uint64_t{1}, uint64_t{64 * 1024 * 1024});
  INFO(target);
  sirius::test::scoped_temp_directory disk;
  auto config = YAML::LoadFile(lake_config().string());
  config["sirius"]["operator_params"]["scan_task_batch_size"] = target;
  auto path = std::filesystem::path(disk.directory) / "batch.yaml";
  {
    std::ofstream output(path);
    output << config;
  }
  sirius::test::shared_test_env env(path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  attach_lake(con, disk.directory);
  query(con, "SET gpu_execution=false");
  query(con, "CALL lake.set_option('data_inlining_row_limit', 0)");
  query(con, "CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(524288) r(i)");
  auto sql      = std::string("SELECT SUM(id) FROM lake.q");
  auto expected = ducklake_result_bag(*query(con, sql));
  auto runtime  = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  query(con, "SET gpu_execution=true");
  query(con, "SET enable_duckdb_fallback=false");
  for (auto const& statement :
       {sql, "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")"}) {
    decoded = 0;
    CHECK(ducklake_result_bag(*query(con, statement)) == expected);
    if (target == 1) {
      CHECK(decoded.load() > 1);
    } else {
      CHECK(decoded.load() == 1);
    }
  }
}

TEST_CASE("Production DuckLake SQL lowers complete bound scans",
          "[isolated_context][ducklake][ducklake_sql]")
{
  auto shape = GENERATE(std::string("parquet"), std::string("mixed"), std::string("inline"));
  INFO(shape);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");
  run("CREATE TABLE lake.q(id BIGINT, value BIGINT, label VARCHAR)");
  run("INSERT INTO lake.q VALUES (1,11,'one'),(2,22,'two'),(2,22,'two'),(3,NULL,'three')");
  if (shape != "inline") run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (4,44,'four'),(5,55,NULL)");
  if (shape == "parquet") run("CALL ducklake_flush_inlined_data('lake')");
  run("ATTACH '" + (std::filesystem::path(disk.directory) / "native.db").string() +
      "' AS native_store");
  run("CREATE TABLE native_store.native_keys(id BIGINT)");
  run("INSERT INTO native_store.native_keys VALUES (2),(4)");
  run("CHECKPOINT native_store");
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  // Admission must not depend on diagnostic counters being enabled.
  run("SET enable_duckdb_fallback=false");
  for (bool track : {false, true}) {
    counters->track_units = track;
    auto before           = runtime->get_transparent_execution_stats();
    CHECK(run("SELECT id FROM lake.q")->RowCount() == 6);
    CHECK(run("CALL gpu_execution('SELECT id FROM lake.q')")->RowCount() == 6);
    auto after = runtime->get_transparent_execution_stats();
    CHECK(after.executions == before.executions + 1);
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  }
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  run("SET enable_duckdb_fallback=false");
  std::vector<std::string> queries = {
    "SELECT id FROM lake.q",
    "SELECT 1 FROM lake.q",
    "SELECT value,id,label FROM lake.q WHERE id>=2",
    "SELECT id FROM lake.q WHERE id=1 OR id=4",
    "SELECT COUNT(*) FROM lake.q",
    "SELECT COUNT(*) FROM lake.q WHERE id>=2",
    "SELECT COUNT(*) FROM lake.q WHERE id%2=0",
    "SELECT COUNT(*) FROM lake.q WHERE id>90",
    "SELECT id,COUNT(*) FROM lake.q GROUP BY id",
    "SELECT a.id,b.value FROM lake.q a JOIN lake.q b ON a.id=b.id WHERE b.id>=2",
    "SELECT a.id FROM lake.q a JOIN native_store.native_keys b ON a.id=b.id",
    "SELECT id FROM lake.q WHERE id>90"};
  for (auto const& sql : queries) {
    INFO(sql);
    run("SET gpu_execution=false");
    auto expected = ducklake_result_bag(*run(sql));
    run("SET gpu_execution=true");
    for (bool explicit_entry : {false, true}) {
      INFO(explicit_entry);
      auto decoded_before  = decoded.load();
      auto imported_before = counters->arrow_rows_imported.load();
      auto before          = runtime->get_transparent_execution_stats();
      auto result          = con.Query(
        explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
      INFO((result->HasError() ? result->GetError() : "GPU query completed"));
      REQUIRE_FALSE(result->HasError());
      CHECK(ducklake_result_bag(*result) == expected);
      auto after = runtime->get_transparent_execution_stats();
      CHECK(after.executions == before.executions + (explicit_entry ? 0 : 1));
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
      CHECK(after.certificate_incompletes == before.certificate_incompletes);
      CHECK(after.certificate_mismatches == before.certificate_mismatches);
      if (sql == "SELECT id FROM lake.q" || sql == "SELECT 1 FROM lake.q") {
        CHECK((decoded.load() > decoded_before) == (shape != "inline"));
        CHECK((counters->arrow_rows_imported.load() > imported_before) == (shape != "parquet"));
      }
    }
  }
  auto prepared = con.Prepare("SELECT id FROM lake.q WHERE id>=?");
  REQUIRE_FALSE(prepared->HasError());
  for (int64_t value : {2, 4, 1}) {
    auto before = runtime->get_transparent_execution_stats();
    auto result = prepared->Execute(value);
    INFO((result->HasError() ? result->GetError() : "prepared GPU execution completed"));
    REQUIRE_FALSE(result->HasError());
    auto actual = ducklake_result_bag(*result);
    auto after  = runtime->get_transparent_execution_stats();
    CHECK(after.executions == before.executions + 1);
    CHECK(after.fallbacks == before.fallbacks);
    run("SET gpu_execution=false");
    CHECK(actual ==
          ducklake_result_bag(*run("SELECT id FROM lake.q WHERE id>=" + std::to_string(value))));
    run("SET gpu_execution=true");
  }
}
TEST_CASE("Production DuckLake SQL retires failed GPU work before replay",
          "[isolated_context][ducklake][ducklake_recovery]")
{
  auto explicit_entry = GENERATE(false, true);
  INFO(explicit_entry);
  auto mode = GENERATE(std::string("replay"),
                       std::string("disabled"),
                       std::string("replay_failure"),
                       std::string("interrupted"));
  INFO(mode);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");
  run("CREATE TABLE lake.q(id BIGINT, value BIGINT)");
  for (int batch = 0; batch < 4; ++batch) {
    run("INSERT INTO lake.q SELECT i, i%7 FROM range(" + std::to_string(batch * 4096) + "," +
        std::to_string((batch + 1) * 4096) + ") r(i)");
    run("CALL ducklake_flush_inlined_data('lake')");
  }
  run("INSERT INTO lake.q VALUES (20000,5),(20000,5),(20001,NULL)");
  std::string sql = "SELECT id,value FROM lake.q WHERE id%3<>0";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0}, replay_checks{0};
  uint64_t reserved_at_replay = UINT64_MAX;
  std::string failed_file;
  counters->parquet_phase_for_testing = [&](std::string const& file, bool footer) {
    if (!footer && ++decoded == 3) {
      failed_file = file;
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io, "injected DuckLake decode failure");
    }
  };
  runtime->cpu_replay_hook_for_testing = [&] {
    ++replay_checks;
    reserved_at_replay = hosts[0]->get_total_reserved_memory();
    if (mode == "replay_failure") std::filesystem::rename(failed_file, failed_file + ".saved");
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    runtime->cpu_replay_hook_for_testing = {};
    if (!failed_file.empty() && std::filesystem::exists(failed_file + ".saved"))
      std::filesystem::rename(failed_file + ".saved", failed_file);
  });
  run("SET sirius_test_sync_cpu_replay=true");
  run(std::string("SET sirius_test_interrupt_before_replay=") +
      (mode == "interrupted" ? "true" : "false"));
  run(std::string("SET enable_duckdb_fallback=") + (mode == "disabled" ? "false" : "true"));
  auto before = runtime->get_transparent_execution_stats();
  auto result = con.Query(
    explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
  INFO((result->HasError() ? result->GetError() : "replayed successfully"));
  CHECK(decoded >= 3);
  if (mode == "replay") {
    REQUIRE_FALSE(result->HasError());
    CHECK(ducklake_result_bag(*result) == expected);
  } else
    CHECK(result->HasError());
  bool replayed = mode == "replay" || mode == "replay_failure";
  CHECK(replay_checks == (mode == "disabled" ? 0 : 1));
  if (mode != "disabled") CHECK(reserved_at_replay == reserved);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  auto after = runtime->get_transparent_execution_stats();
  if (mode == "interrupted") {
    auto cancelled = static_cast<size_t>(sirius::transparent::late_failure_condition::cancelled);
    CHECK(after.late_failure_no_replay[cancelled] == before.late_failure_no_replay[cancelled] + 1);
  }
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (replayed ? 1 : 0));
  counters->parquet_phase_for_testing  = {};
  runtime->cpu_replay_hook_for_testing = {};
  if (mode == "replay_failure") std::filesystem::rename(failed_file + ".saved", failed_file);
  run("SET sirius_test_sync_cpu_replay=false");
  run("SET sirius_test_interrupt_before_replay=false");
  run("SET enable_duckdb_fallback=false");
  auto next = runtime->get_transparent_execution_stats();
  CHECK(ducklake_result_bag(*run(sql)) == expected);
  CHECK(runtime->get_transparent_execution_stats().executions == next.executions + 1);
}
TEST_CASE("Production DuckLake SQL preserves its view across concurrent commits",
          "[isolated_context][ducklake][ducklake_concurrency]")
{
  auto mode = GENERATE(std::string("commit"), std::string("flush"));
  INFO(mode);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q(id BIGINT)");
  run("INSERT INTO lake.q VALUES (1),(2),(2)");
  run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (3),(4)");
  std::string sql = "SELECT id FROM lake.q";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto writer           = env.make_connection();
  auto write            = [&](std::string const& text) {
    auto r = writer.Query(text);
    if (r->HasError()) throw std::runtime_error(r->GetError());
  };
  write("SET gpu_execution=false");
  run("SET enable_duckdb_fallback=false");
  std::promise<void> arrived, release;
  auto at_capture = arrived.get_future();
  auto resume     = release.get_future().share();
  std::atomic<bool> first{true};
  counters->after_certify_for_testing = [&] {
    if (first.exchange(false)) {
      arrived.set_value();
      if (resume.wait_for(std::chrono::seconds(15)) != std::future_status::ready)
        throw std::runtime_error("capture rendezvous timed out");
    }
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->after_certify_for_testing = {}; });
  auto before = runtime->get_transparent_execution_stats();
  auto active =
    std::async(std::launch::async, [&] { return ducklake_result_bag(*con.Query(sql)); });
  bool released = false;
  auto unblock  = std::shared_ptr<void>(nullptr, [&](void*) {
    if (!released) release.set_value();
  });
  REQUIRE(at_capture.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  write("INSERT INTO lake.q VALUES (9)");
  if (mode == "flush") write("CALL ducklake_flush_inlined_data('lake')");
  release.set_value();
  released = true;
  REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  CHECK(active.get() == expected);
  counters->after_certify_for_testing = {};
  auto after                          = runtime->get_transparent_execution_stats();
  CHECK(after.executions == before.executions + 1);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  expected.push_back("9|");
  std::sort(expected.begin(), expected.end());
  CHECK(ducklake_result_bag(*run(sql)) == expected);
}

TEST_CASE("Production DuckLake SQL cancellation drains blocked reads without replay",
          "[isolated_context][ducklake][ducklake_cancel]")
{
  auto footer = GENERATE(false, true);
  INFO(footer);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(10000) r(i)");
  run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (10001)");
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::promise<void> arrived, release;
  auto at_read = arrived.get_future();
  auto resume  = release.get_future().share();
  std::atomic<bool> first{true};
  std::atomic<unsigned> replays{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool at_footer) {
    if (at_footer == footer && first.exchange(false)) {
      arrived.set_value();
      if (resume.wait_for(std::chrono::seconds(15)) != std::future_status::ready)
        throw std::runtime_error("read rendezvous timed out");
    }
  };
  runtime->cpu_replay_hook_for_testing = [&] { ++replays; };
  auto clear                           = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  run("SET sirius_test_sync_cpu_replay=true");
  auto before = runtime->get_transparent_execution_stats();
  auto active =
    std::async(std::launch::async, [&] { return con.Query("SELECT id FROM lake.q WHERE id%2=0"); });
  bool released = false;
  auto unblock  = std::shared_ptr<void>(nullptr, [&](void*) {
    if (!released) release.set_value();
  });
  REQUIRE(at_read.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  con.Interrupt();
  release.set_value();
  released = true;
  REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  auto result = active.get();
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(result->GetError().find("Interrupt") != std::string::npos);
  CHECK(replays == 0);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  counters->parquet_phase_for_testing = {};
  run("SET sirius_test_sync_cpu_replay=false");
  run("SET enable_duckdb_fallback=false");
  CHECK(ducklake_result_bag(*run("SELECT id FROM lake.q WHERE id=10001")) ==
        std::vector<std::string>{"10001|"});
}
TEST_CASE("Production DuckLake SQL combines dynamic filters with local visibility",
          "[isolated_context][ducklake][ducklake_dynamic]")
{
  auto shape   = GENERATE(std::string("parquet"), std::string("mixed"), std::string("inline"));
  auto deletes = GENERATE(std::string("ordinary"), std::string("dv"), std::string("inline"));
  INFO(shape);
  INFO(deletes);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CALL lake.set_option('data_inlining_row_limit'," +
      std::string(shape == "inline" ? "20000" : "0") + ")");
  run("CALL lake.set_option('parquet_row_group_size',2048)");
  run("CALL lake.set_option('write_deletion_vectors'," +
      std::string(deletes == "dv" ? "true" : "false") + ")");
  run(
    "CREATE TABLE lake.q AS SELECT range::BIGINT id, CASE WHEN range%11=0 THEN NULL ELSE range%17 "
    "END::BIGINT value FROM range(10000)");
  if (shape == "mixed") {
    run("CALL lake.set_option('data_inlining_row_limit',20000)");
    run("INSERT INTO lake.q SELECT range::BIGINT, (range%17)::BIGINT FROM range(2700,3300)");
  }
  run("CALL lake.set_option('data_inlining_row_limit'," +
      std::string(deletes == "inline" ? "20000" : "0") + ")");
  run("DELETE FROM lake.q WHERE id%7=0");
  run("ATTACH '" + (std::filesystem::path(disk.directory) / "keys.db").string() +
      "' AS native_store");
  run("CREATE TABLE native_store.keys AS SELECT range::BIGINT k FROM range(17)");
  run("CHECKPOINT native_store");
  std::string sql =
    "SELECT q.value,q.id FROM lake.q q JOIN native_store.keys b ON q.value=b.k WHERE q.id>=2048 "
    "AND q.id<4096 AND b.k%4=1";
  auto expected = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  run("SET enable_duckdb_fallback=false");
  run("SET enable_dynamic_filter=true");
  run("SET dynamic_filter_domain_coverage_threshold=2.0");
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto before         = runtime->get_transparent_execution_stats();
    auto dynamic_before = runtime->get_dynamic_filter_stats_snapshot();
    CHECK(ducklake_result_bag(*run(sql)) == expected);
    auto after         = runtime->get_transparent_execution_stats();
    auto dynamic_after = runtime->get_dynamic_filter_stats_snapshot();
    CHECK(after.executions == before.executions + 1);
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(dynamic_after.producers_enabled > dynamic_before.producers_enabled);
    CHECK(dynamic_after.filters_pushed > dynamic_before.filters_pushed);
    CHECK(dynamic_after.publications_failed == dynamic_before.publications_failed);
  }
  CHECK((decoded.load() > 0) == (shape != "inline"));
  CHECK((counters->arrow_rows_imported.load() > 0) == (shape != "parquet"));
}

TEST_CASE("Production DuckLake SQL refuses a grant that cannot progress both join scans",
          "[isolated_context][ducklake][ducklake_admission]")
{
  auto fallback = GENERATE(false, true);
  INFO(fallback);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q(id BIGINT)");
  run("INSERT INTO lake.q VALUES (1),(2),(2),(3)");
  std::string sql = "SELECT a.id,b.id FROM lake.q a JOIN lake.q b ON a.id=b.id";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto provider         = std::make_shared<sirius::scan_manager::test::test_reservation_provider>();
  auto envelope         = sirius::ducklake::scan_ingestible::envelope();
  provider->grant_bytes = 2 * envelope.retained;
  for (auto bytes : envelope.temporary)
    provider->grant_bytes += bytes;
  counters->preparation_provider_for_testing = provider;
  auto clear                                 = std::shared_ptr<void>(
    nullptr, [&](void*) { counters->preparation_provider_for_testing.reset(); });
  run(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
  auto result = con.Query(sql);
  if (fallback) {
    REQUIRE_FALSE(result->HasError());
    CHECK(ducklake_result_bag(*result) == expected);
  } else
    CHECK(result->HasError());
  CHECK(provider->seen->requests == 1);
  CHECK(provider->seen->allocations == 0);
  CHECK(provider->outstanding() == 0);
  CHECK(counters->arrow_batches_imported == 0);
}

TEST_CASE("Production DuckLake SQL isolates concurrent prepared executions",
          "[isolated_context][ducklake][ducklake_concurrency]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  auto concurrent_config                                = YAML::LoadFile(lake_config().string());
  concurrent_config["sirius"]["max_concurrent_queries"] = 2;
  if (concurrent_config["sirius"]["executor"]["scan_manager"]["max_concurrent_queries"])
    concurrent_config["sirius"]["executor"]["scan_manager"]["max_concurrent_queries"] = 2;
  auto config_path = std::filesystem::path(disk.directory) / "concurrent.yaml";
  {
    std::ofstream output(config_path);
    output << concurrent_config;
  }
  sirius::test::shared_test_env env(config_path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q AS SELECT range::BIGINT id,(range%17)::BIGINT value FROM range(2048)");
  run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (7,70),(8,80)");
  std::array<int64_t, 4> limits{0, 500, 1000, 1500};
  std::vector<std::vector<std::string>> expected;
  for (auto v : limits)
    expected.push_back(
      ducklake_result_bag(*run("SELECT value,id FROM lake.q WHERE id>=" + std::to_string(v) +
                               " AND id<" + std::to_string(v + 300))));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  auto sibling = env.make_connection();
  run("SET enable_duckdb_fallback=false");
  auto setting = sibling.Query("SET enable_duckdb_fallback=false");
  REQUIRE_FALSE(setting->HasError());
  REQUIRE(runtime->get_config().max_concurrent_queries() == 2);
  std::mutex rendezvous_mutex;
  std::condition_variable rendezvous;
  unsigned arrived = 0;
  std::atomic<unsigned> first_footers{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer || first_footers.fetch_add(1) >= 2) return;
    std::unique_lock lock(rendezvous_mutex);
    ++arrived;
    rendezvous.notify_all();
    if (!rendezvous.wait_for(lock, std::chrono::seconds(15), [&] { return arrived == 2; }))
      throw std::runtime_error("two DuckLake queries did not reach preparation concurrently");
  };
  auto before = runtime->get_transparent_execution_stats();
  std::promise<void> release;
  auto ready   = release.get_future().share();
  auto execute = [&](duckdb::Connection& con, int shift) {
    ready.wait();
    auto prepared = con.Prepare("SELECT value,id FROM lake.q WHERE id>=? AND id<?");
    if (prepared->HasError()) throw std::runtime_error(prepared->GetError());
    for (int repeat = 0; repeat < 8; ++repeat) {
      auto index  = (repeat + shift) % limits.size();
      auto v      = limits[index];
      auto result = prepared->Execute(v, v + 300);
      if (ducklake_result_bag(*result) != expected[index]) return false;
    }
    return true;
  };
  auto a = std::async(std::launch::async, [&] { return execute(con, 0); });
  auto b = std::async(std::launch::async, [&] { return execute(sibling, 2); });
  release.set_value();
  REQUIRE(a.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  REQUIRE(b.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  CHECK(a.get());
  CHECK(b.get());
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.executions == before.executions + 16);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(arrived == 2);
}

TEST_CASE(
  "Production DuckLake SQL preserves transaction local changes and new statement identities",
  "[isolated_context][ducklake][ducklake_transaction]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q(id BIGINT,value BIGINT)");
  run("INSERT INTO lake.q VALUES (1,10),(2,20),(3,30)");
  run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (4,40),(5,50)");
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  run("SET gpu_execution=false");
  run("BEGIN");
  run("INSERT INTO lake.q VALUES (6,60),(6,60)");
  run("DELETE FROM lake.q WHERE id IN (1,4)");
  auto expected = ducklake_result_bag(*run("SELECT value,id FROM lake.q"));
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  auto before = runtime->get_transparent_execution_stats();
  CHECK(ducklake_result_bag(*run("SELECT value,id FROM lake.q")) == expected);
  CHECK(runtime->get_transparent_execution_stats().executions == before.executions + 1);
  run("ROLLBACK");
  run("SET gpu_execution=false");
  expected = ducklake_result_bag(*run("SELECT value,id FROM lake.q"));
  run("SET gpu_execution=true");
  auto prepared = con.Prepare("SELECT value,id FROM lake.q");
  REQUIRE_FALSE(prepared->HasError());
  CHECK(ducklake_result_bag(*prepared->Execute()) == expected);
  run("SET gpu_execution=false");
  run("DELETE FROM lake.q WHERE id=2");
  run("INSERT INTO lake.q VALUES (7,70)");
  expected = ducklake_result_bag(*run("SELECT value,id FROM lake.q"));
  run("SET gpu_execution=true");
  auto fresh = runtime->get_transparent_execution_stats();
  CHECK(ducklake_result_bag(*prepared->Execute()) == expected);
  CHECK(runtime->get_transparent_execution_stats().executions == fresh.executions + 1);
}

TEST_CASE("Production DuckLake SQL drains publication races before deciding replay",
          "[isolated_context][ducklake][ducklake_publication]")
{
  auto explicit_entry = GENERATE(false, true);
  auto hold_batch     = GENERATE(false, true);
  auto fallback       = GENERATE(false, true);
  CAPTURE(explicit_entry, hold_batch, fallback);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',100)");
  run("CREATE TABLE lake.q(id BIGINT, value BIGINT)");
  for (int batch = 0; batch < 3; ++batch) {
    run("INSERT INTO lake.q SELECT i,i%7 FROM range(" + std::to_string(batch * 4096) + "," +
        std::to_string((batch + 1) * 4096) + ") r(i)");
    run("CALL ducklake_flush_inlined_data('lake')");
  }
  run("INSERT INTO lake.q VALUES (20000,5),(20000,5),(20001,NULL)");
  std::string sql = "SELECT value,COUNT(*),SUM(id) FROM lake.q WHERE id%3<>0 GROUP BY value";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> footers{0}, replays{0};
  std::promise<void> arrived, release;
  auto blocked                        = arrived.get_future();
  auto resume                         = release.get_future().share();
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (footer && ++footers == 2) {
      arrived.set_value();
      if (resume.wait_for(std::chrono::seconds(20)) != std::future_status::ready)
        throw std::runtime_error("publication rendezvous timed out");
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io,
        "injected later DuckLake footer failure");
    }
  };
  uint64_t reserved_at_replay          = UINT64_MAX;
  runtime->cpu_replay_hook_for_testing = [&] {
    ++replays;
    reserved_at_replay = hosts[0]->get_total_reserved_memory();
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  run("SET sirius_test_sync_cpu_replay=true");
  run("SET scan_task_batch_size=1");
  // Enable the shared publication observer without holding a real footer.
  run("SET sirius_test_hold_footer_index=" + std::to_string(UINT64_MAX));
  run(std::string("SET sirius_test_hold_published_batch=") + (hold_batch ? "true" : "false"));
  run(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
  auto before    = runtime->get_transparent_execution_stats();
  auto active    = std::async(std::launch::async, [&] {
    return con.Query(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")"
                                    : sql);
  });
  bool reached   = blocked.wait_for(std::chrono::seconds(15)) == std::future_status::ready;
  bool published = reached && runtime->get_scan_manager().wait_for_publication_for_testing(
                                std::chrono::seconds(15));
  release.set_value();
  auto result = active.get();
  REQUIRE(reached);
  REQUIRE(published);
  INFO((result->HasError() ? result->GetError() : "replayed successfully"));
  CHECK(result->HasError() == !fallback);
  if (fallback) CHECK(ducklake_result_bag(*result) == expected);
  CHECK(replays == (fallback ? 1 : 0));
  if (fallback) CHECK(reserved_at_replay == reserved);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (fallback ? 1 : 0));
  if (hold_batch) CHECK(after.discarded_speculative_work > before.discarded_speculative_work);
}

TEST_CASE("Production DuckLake SQL refuses replay when native checkpoint cleanup fails",
          "[isolated_context][ducklake][ducklake_cleanup]")
{
  auto explicit_entry = GENERATE(false, true);
  CAPTURE(explicit_entry);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',100)");
  run("CREATE TABLE lake.q(id BIGINT,value BIGINT)");
  for (int batch = 0; batch < 4; ++batch)
    run("INSERT INTO lake.q SELECT i,i%7 FROM range(" + std::to_string(batch * 4096) + "," +
        std::to_string((batch + 1) * 4096) + ") r(i)");
  run("INSERT INTO lake.q VALUES (20000,5),(20000,5),(20001,NULL)");
  run("ATTACH '" + (std::filesystem::path(disk.directory) / "keys.db").string() +
      "' AS native_store");
  run("CREATE TABLE native_store.keys AS SELECT range::BIGINT k FROM range(7)");
  run("CHECKPOINT native_store");
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0}, replays{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer && ++decoded == 3)
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io, "injected DuckLake decode failure");
  };
  runtime->cpu_replay_hook_for_testing = [&] { ++replays; };
  auto clear                           = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    runtime->cpu_replay_hook_for_testing = {};
    // The injection happens after worker drains. Release retained test keys before teardown.
    runtime->get_scan_manager().reset_all();
  });
  run("SET enable_duckdb_fallback=true");
  run("SET sirius_test_sync_cpu_replay=true");
  run("SET sirius_test_inject_checkpoint_cleanup_failure=true");
  auto before     = runtime->get_transparent_execution_stats();
  std::string sql = "SELECT q.id,b.k FROM lake.q q JOIN native_store.keys b ON q.value=b.k";
  auto result     = con.Query(
    explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(decoded >= 3);
  CHECK(replays == 0);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  CHECK(runtime->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  CHECK(runtime->get_scan_manager().checkpoint_key_count() > 0);
}

TEST_CASE("Production DuckLake SQL does not reuse rows after replacing a file at the same path",
          "[isolated_context][ducklake][ducklake_object_identity]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id,(i%7)::BIGINT value FROM range(2048) r(i)");
  std::string sql = "SELECT value,id FROM lake.q";
  auto original   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  run("SET enable_duckdb_fallback=false");
  for (int repeat = 0; repeat < 2; ++repeat)
    CHECK(ducklake_result_bag(*run(sql)) == original);
  REQUIRE(decoded >= 2);
  run("SET gpu_execution=false");
  auto files = run("SELECT data_file FROM ducklake_list_files('lake', 'q')");
  REQUIRE(files->RowCount() == 1);
  auto path        = files->Collection().GetValue(0, 0).ToString();
  auto replacement = (std::filesystem::path(disk.directory) / "replacement.parquet").string();
  run("COPY (SELECT id+10000 AS id,value+100 AS value FROM read_parquet(" +
      duckdb::Value(path).ToSQLString() + ")) TO " + duckdb::Value(replacement).ToSQLString() +
      " (FORMAT PARQUET, FIELD_IDS {id:1,value:2})");
  std::filesystem::rename(replacement, path);
  auto expected = ducklake_result_bag(*run(sql));
  REQUIRE(expected != original);
  run("SET gpu_execution=true");
  auto before = runtime->get_transparent_execution_stats();
  auto result = con.Query(sql);
  if (result->HasError()) {
    // Catalog file evidence may reject external replacement; replay must read the new object.
    run("SET enable_duckdb_fallback=true");
    result = con.Query(sql);
    REQUIRE_FALSE(result->HasError());
    CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks ==
          before.runtime_fallbacks + 1);
  } else {
    CHECK(runtime->get_transparent_execution_stats().executions == before.executions + 1);
    CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  }
  CHECK(ducklake_result_bag(*result) == expected);
}

TEST_CASE("Production DuckLake SQL cancellation waits for its blocked provider query",
          "[isolated_context][ducklake][ducklake_provider_cancel]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',20000)");
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id,(i%7)::BIGINT value FROM range(4096) r(i)");
  std::string sql = "SELECT value,id FROM lake.q WHERE id%3<>0";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  run("SET enable_duckdb_fallback=true");
  run("SET sirius_test_sync_cpu_replay=true");
  std::atomic<unsigned> replay_checks{0};
  runtime->cpu_replay_hook_for_testing = [&] { ++replay_checks; };
  std::promise<void> arrived, release;
  auto blocked = arrived.get_future();
  auto resume  = release.get_future().share();
  std::atomic<bool> first{true};
  std::thread::id execution_owner, provider_owner;
  auto observer     = duckdb::make_shared_ptr<provider_query_observer>();
  observer->observe = [&](duckdb::ClientContext& provider) {
    auto state = duckdb::get_sirius_connection_state(provider);
    // Admission distinguishes reader work from metadata SQL used while binding the user query.
    if (!state || state->provenance() != connection_provenance::provider_internal ||
        hosts[0]->get_total_reserved_memory() <= reserved || !first.exchange(false))
      return;
    provider_owner = std::this_thread::get_id();
    arrived.set_value();
    if (resume.wait_for(std::chrono::seconds(20)) != std::future_status::ready)
      throw std::runtime_error("provider query rendezvous timed out");
  };
  duckdb::OptimizerExtension hook;
  hook.optimizer_info        = observer;
  hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), std::move(hook));
  auto clear   = std::shared_ptr<void>(nullptr, [&](void*) {
    observer->observe                    = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  auto before  = runtime->get_transparent_execution_stats();
  auto active  = std::async(std::launch::async, [&] {
    execution_owner = std::this_thread::get_id();
    return con.Query(sql);
  });
  bool reached = blocked.wait_for(std::chrono::seconds(15)) == std::future_status::ready;
  con.Interrupt();
  auto while_blocked = active.wait_for(std::chrono::milliseconds(50));
  release.set_value();
  auto result = active.get();
  REQUIRE(reached);
  CHECK(provider_owner == execution_owner);
  CHECK(while_blocked == std::future_status::timeout);
  REQUIRE(result->HasError());
  CHECK(replay_checks == 0);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  observer->observe = {};
  run("SET enable_duckdb_fallback=false");
  auto next = runtime->get_transparent_execution_stats();
  CHECK(ducklake_result_bag(*run(sql)) == expected);
  CHECK(runtime->get_transparent_execution_stats().executions == next.executions + 1);
}

TEST_CASE("Production DuckLake mixed scans execute and recover across two GPUs",
          "[isolated_context][ducklake_multi_gpu][multi_gpu]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  if (!sirius::test::has_gpus(2)) SKIP("DuckLake device placement requires two GPUs");
  sirius::test::scoped_temp_directory disk;
  auto config                              = YAML::LoadFile(lake_config().string());
  config["sirius"]["topology"]["num_gpus"] = 2;
  auto config_path                         = std::filesystem::path(disk.directory) / "devices.yaml";
  {
    std::ofstream output(config_path);
    output << config;
  }
  sirius::test::shared_test_env env(config_path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  auto use_dv = GENERATE(false, true);
  INFO(use_dv);
  run(std::string("CALL lake.set_option('write_deletion_vectors',") + (use_dv ? "true" : "false") +
      ")");
  run("CREATE TABLE lake.q(id BIGINT,value BIGINT)");
  for (int batch = 0; batch < 16; ++batch) {
    run(
      "INSERT INTO lake.q SELECT i,CASE WHEN i%7=0 THEN NULL ELSE i%17 END "
      "FROM range(" +
      std::to_string(batch * 32768) + "," + std::to_string((batch + 1) * 32768) + ") r(i)");
  }
  run("DELETE FROM lake.q WHERE id%13=0");
  run("ALTER TABLE lake.q RENAME id TO renamed_id");
  run("CALL lake.set_option('data_inlining_row_limit',100)");
  run("INSERT INTO lake.q VALUES (42,2),(42,2),(43,NULL)");
  std::string sql =
    "SELECT value,COUNT(*),SUM(renamed_id) FROM lake.q WHERE renamed_id%3=0 OR renamed_id%7=0 "
    "GROUP BY value";
  auto expected = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::mutex mutex;
  std::set<int> devices;
  bool inject = false, fired = false;
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (footer) return;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cannot observe GPU device");
    std::lock_guard lock(mutex);
    devices.insert(device);
    if (inject && devices.size() == 2 && !std::exchange(fired, true))
      throw std::runtime_error("DuckLake test failure after both devices decoded");
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  for (bool explicit_entry : {false, true}) {
    INFO(explicit_entry);
    for (bool fail : {false, true, false}) {
      INFO(fail);
      devices.clear();
      inject = fail;
      fired  = false;
      run(std::string("SET enable_duckdb_fallback=") + (fail ? "true" : "false"));
      auto before   = runtime->get_transparent_execution_stats();
      auto imported = counters->arrow_rows_imported.load();
      auto result =
        run(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
      CHECK(ducklake_result_bag(*result) == expected);
      result.reset();
      CHECK(devices.size() == 2);
      CHECK(fired == fail);
      auto after = runtime->get_transparent_execution_stats();
      if (!explicit_entry) {
        CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (fail ? 1 : 0));
        CHECK(after.executions == before.executions + (fail ? 0 : 1));
      }
      if (!fail) CHECK(counters->arrow_rows_imported.load() > imported);
      CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    }
  }
}

TEST_CASE("Production DuckLake inline batch sizing preserves wide nullable rows",
          "[isolated_context][ducklake][ducklake_inline_batches]")
{
  auto width = GENERATE(2, 5, 32);
  INFO(width);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',20000)");
  std::string fields = "i::BIGINT id";
  for (int col = 1; col < width; ++col) {
    fields += ",CASE WHEN i%13=0 THEN NULL ELSE ";
    fields += col % 3 == 0 ? "i%2=0" : "'value_'||CAST(i%17 AS VARCHAR)";
    fields += " END AS c" + std::to_string(col);
  }
  run("CREATE TABLE lake.q AS SELECT " + fields + " FROM range(4097) r(i)");
  auto expected = ducklake_result_bag(*run("SELECT * FROM lake.q"));
  auto runtime  = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  for (bool explicit_entry : {false, true}) {
    auto batches = counters->arrow_batches_imported.load();
    auto rows    = counters->arrow_rows_imported.load();
    auto result =
      run(explicit_entry ? "CALL gpu_execution('SELECT * FROM lake.q')" : "SELECT * FROM lake.q");
    CHECK(ducklake_result_bag(*result) == expected);
    result.reset();
    CHECK(counters->arrow_rows_imported.load() - rows == 4097);
    auto imported = counters->arrow_batches_imported.load() - batches;
    CHECK(imported > 0);
    // Provider chunk boundaries and empty carriers affect the exact count.
    // A narrow scan must still need fewer imports than 256-row data batches alone.
    if (width == 2) CHECK(imported < 17);
    if (width == 32) CHECK(imported >= 17);
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  }
}

TEST_CASE("DuckLake result fragments execute original bindings without CPU replay",
          "[isolated_context][ducklake][ducklake_fragment]")
{
  auto shape       = GENERATE(std::string("parquet"),
                        std::string("mixed"),
                        std::string("inline"),
                        std::string("failed_parquet"));
  bool fail_decode = shape == "failed_parquet";
  INFO(shape);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit',100)");
  query(con, "CREATE TABLE lake.q(id BIGINT)");
  query(con, "INSERT INTO lake.q VALUES (1),(2),(2)");
  if (shape != "inline") query(con, "CALL ducklake_flush_inlined_data('lake')");
  query(con, "INSERT INTO lake.q VALUES (3),(NULL)");
  if (shape == "parquet" || fail_decode) query(con, "CALL ducklake_flush_inlined_data('lake')");
  auto sql      = std::string("SELECT id FROM lake.q");
  auto expected = ducklake_result_bag(*query(con, sql));
  auto runtime  = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  con.context->registered_state->Insert(
    sirius::exec::stream_bind_catalog::kStateKey,
    duckdb::make_shared_ptr<sirius::exec::stream_bind_catalog>());
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) {
      ++decoded;
      if (fail_decode) throw std::runtime_error("fragment decode failure");
    }
  };
  auto observer          = duckdb::make_shared_ptr<provider_query_observer>();
  unsigned visits        = 0;
  bool inside            = false;
  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    if (&context != con.context.get() || inside) return;
    inside     = true;
    auto leave = std::shared_ptr<void>(nullptr, [&](void*) { inside = false; });
    ++visits;
    sirius::exec::fragment_spec spec;
    spec.plan_source = [&](duckdb::ClientContext& current) {
      REQUIRE(&current == &context);
      auto prepared = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
        duckdb::StatementType::SELECT_STATEMENT);
      prepared->types = {duckdb::LogicalType::BIGINT};
      prepared->names = {"id"};
      return sirius::exec::bound_plan{plan.Copy(current), std::move(prepared)};
    };
    sirius::exec::streaming_fragment fragment(context, std::move(spec));
    fragment.build();
    if (fail_decode) {
      REQUIRE_THROWS_WITH(fragment.run(),
                          Catch::Matchers::ContainsSubstring("fragment decode failure"));
      REQUIRE_THROWS(fragment.take_result());
    } else {
      fragment.run();
      CHECK(ducklake_result_bag(*fragment.take_result()) == expected);
    }
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    observer->inspect_plan              = {};
    counters->parquet_phase_for_testing = {};
  });
  duckdb::OptimizerExtension extension;
  extension.optimizer_info    = observer;
  extension.optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), extension);
  auto imported = counters->arrow_rows_imported.load();
  auto before   = runtime->get_transparent_execution_stats();
  CHECK(ducklake_result_bag(*query(con, sql)) == expected);
  CHECK(visits == 1);
  CHECK((decoded.load() > 0) == (shape != "inline"));
  CHECK((counters->arrow_rows_imported.load() > imported) == (shape != "parquet" && !fail_decode));
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE("DuckLake scalar subqueries and floating predicates match CPU results",
          "[isolated_context][ducklake][ducklake_scalar_nan]")
{
  auto parquet = GENERATE(false, true);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit',100)");
  query(con, "CREATE TABLE lake.scalar_value AS SELECT 42::BIGINT AS x");
  query(con, "CREATE TABLE lake.floats(f FLOAT, d DOUBLE)");
  query(con,
        "INSERT INTO lake.floats SELECT x::FLOAT,x FROM (VALUES "
        "('-inf'::DOUBLE),(-1),(0),('inf'),('NaN'),(NULL)) t(x)");
  if (parquet) query(con, "CALL ducklake_flush_inlined_data('lake')");
  std::vector<std::string> queries{"SELECT (SELECT x FROM lake.scalar_value)",
                                   "SELECT first(x),count(*) FROM lake.scalar_value",
                                   "SELECT (SELECT x FROM lake.scalar_value WHERE x%2=1)",
                                   "SELECT first(NULL::BIGINT),count(*) FROM lake.scalar_value"};
  for (auto column : {std::string("f"), std::string("d")}) {
    for (auto op : {"=", "<>", ">", ">=", "<", "<=", "IS DISTINCT FROM", "IS NOT DISTINCT FROM"}) {
      queries.push_back("SELECT a." + column + " " + op + " b." + column +
                        " FROM lake.floats a CROSS JOIN lake.floats b");
      queries.push_back("SELECT count(*) FROM lake.floats WHERE " + column + " " + op + " 'NaN'");
    }
    for (auto predicate : {" BETWEEN '-inf' AND 'NaN'",
                           " IN ('NaN', 1)",
                           " NOT IN ('NaN', 1)",
                           " IN ('NaN', NULL)",
                           " NOT IN ('NaN', NULL)",
                           " > 1e300"}) {
      queries.push_back("SELECT " + column + predicate + " FROM lake.floats");
      queries.push_back("SELECT count(*) FROM lake.floats WHERE " + column + predicate);
    }
  }
  std::vector<std::vector<std::string>> expected;
  for (auto const& sql : queries)
    expected.push_back(ducklake_result_bag(*query(con, sql)));
  query(con, "SET gpu_execution=true");
  query(con, "SET enable_duckdb_fallback=false");
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (size_t i = 0; i < queries.size(); ++i) {
      INFO(queries[i]);
      auto before = get_transparent_execution_stats(con);
      CHECK(ducklake_result_bag(*query(con, queries[i])) == expected[i]);
      auto after = get_transparent_execution_stats(con);
      CHECK(after.executions == before.executions + 1);
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    }
  }
}

TEST_CASE("DuckLake preserves discovery errors when GPU fallback is refused",
          "[isolated_context][ducklake][ducklake_discovery_error]")
{
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  query(con, "SET gpu_execution=false");
  auto attach = "ATTACH 'ducklake:" + disk.directory + "/meta.db' AS lake (DATA_PATH '" +
                disk.directory + "/data', METADATA_CATALOG 'meta', DATA_INLINING_ROW_LIMIT 0)";
  query(con, attach);
  query(con, "CREATE TABLE lake.t(id INTEGER)");
  query(con, "INSERT INTO lake.t VALUES (1), (2)");
  query(con, "DELETE FROM lake.t WHERE id=2");
  std::vector<std::string> statements{
    "SELECT * FROM lake.t",
    "SELECT * FROM ducklake_table_insertions('lake','main','t',0,2)",
    "SELECT * FROM ducklake_table_deletions('lake','main','t',2,3)"};
  for (auto const& sql : statements) {
    query(con, "SET gpu_execution=false");
    auto expected = ducklake_result_bag(*query(con, sql));
    query(con, "SET gpu_execution=true");
    query(con, "SET enable_duckdb_fallback=true");
    CHECK(ducklake_result_bag(*query(con, sql)) == expected);
  }
  query(con, "SET gpu_execution=false");
  query(con, "UPDATE meta.ducklake_data_file SET file_format='foo'");
  query(con, "DETACH lake");
  query(con, attach);
  for (auto const& sql : statements) {
    query(con, "SET gpu_execution=false");
    auto cpu = con.Query(sql);
    REQUIRE(cpu->HasError());
    REQUIRE(cpu->GetError().find("Unsupported data file format: foo") != std::string::npos);
    query(con, "SET gpu_execution=true");
    for (bool fallback : {true, false}) {
      query(con, std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
      for (auto const& statement :
           {sql, "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")"}) {
        auto result = con.Query(statement);
        REQUIRE(result->HasError());
        INFO(result->GetError());
        if (fallback) {
          CHECK(result->GetError().find("Unsupported data file format: foo") != std::string::npos);
          CHECK(result->GetError().find("source discovery incomplete") == std::string::npos);
        } else {
          CHECK((result->GetError().find("GPU") != std::string::npos ||
                 result->GetError().find("SiriusExecuteQuery error:") != std::string::npos));
        }
      }
    }
  }
}

TEST_CASE("DuckLake Top-N prunes unopened files and resets prepared bounds",
          "[isolated_context][ducklake][ducklake_topn]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  // Use multiple batches so the runtime boundary can prune later files.
  auto config = YAML::LoadFile(lake_config().string());
  config["sirius"]["operator_params"]["scan_task_batch_size"] = 8192;
  auto config_path = std::filesystem::path(disk.directory) / "topn.yaml";
  {
    std::ofstream output(config_path);
    output << config;
  }
  sirius::test::shared_test_env env(config_path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit', 0)");
  query(con, "CREATE TABLE lake.events(ts TIMESTAMP, id BIGINT)");
  constexpr int files = 256;
  for (int i = 0; i < files; ++i) {
    auto start = std::to_string(i * 1000);
    query(con,
          "INSERT INTO lake.events SELECT TIMESTAMP '2026-01-01' + INTERVAL (i) SECOND, i "
          "FROM range(" +
            start + "," + std::to_string((i + 1) * 1000) + ") r(i)");
  }
  // Warm CUDA kernels before measuring asynchronous file pruning.
  query(con, "SET gpu_execution=true");
  query(con, "SET enable_duckdb_fallback=false");
  query(con, "SELECT * FROM lake.events WHERE id < 1000 ORDER BY ts ASC LIMIT 2");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::mutex files_lock;
  std::set<std::string> opened, decoded;
  counters->parquet_phase_for_testing = [&](std::string const& path, bool footer) {
    std::lock_guard<std::mutex> guard(files_lock);
    (footer ? opened : decoded).insert(path);
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  for (auto const& order : {"ASC", "DESC"}) {
    auto sql = std::string("SELECT * FROM lake.events ORDER BY ts ") + order + " LIMIT 2";
    query(con, "SET gpu_execution=false");
    auto expected = ducklake_result_bag(*query(con, sql));
    query(con, "SET gpu_execution=true");
    query(con, "SET enable_duckdb_fallback=false");
    opened.clear();
    decoded.clear();
    auto before = get_transparent_execution_stats(con);
    CHECK(ducklake_result_bag(*query(con, sql)) == expected);
    sirius::test::require_transparent_execution_delta(
      before, get_transparent_execution_stats(con), 1, 0, 1);
    INFO("Opened " << opened.size() << ", decoded " << decoded.size() << " of " << files);
    CHECK_FALSE(decoded.empty());
    CHECK(opened.size() < files);
    CHECK(decoded.size() < files);
  }
  for (auto const& suffix :
       {"WHERE ts > TIMESTAMP '2026-01-01 01:00:00' ORDER BY ts DESC LIMIT 5 OFFSET 3",
        "ORDER BY ts DESC, id ASC LIMIT 2",
        "ORDER BY ts ASC NULLS FIRST LIMIT 2"}) {
    auto sql = std::string("SELECT * FROM lake.events ") + suffix;
    query(con, "SET gpu_execution=false");
    auto expected = ducklake_result_bag(*query(con, sql));
    query(con, "SET gpu_execution=true");
    CHECK(ducklake_result_bag(*query(con, sql)) == expected);
  }
  query(con, "SET gpu_execution=false");
  query(con, "CREATE TABLE lake.ties(ts TIMESTAMP, id BIGINT)");
  for (int i = 0; i < 5; ++i) {
    query(con,
          "INSERT INTO lake.ties VALUES (TIMESTAMP '2026-01-01', " + std::to_string(i) +
            "), (NULL, " + std::to_string(i + 10) + ")");
  }
  for (auto const& suffix : {"ORDER BY ts ASC NULLS FIRST, id DESC LIMIT 3",
                             "ORDER BY ts DESC NULLS LAST, id DESC LIMIT 3",
                             "ORDER BY ts ASC NULLS LAST, id ASC LIMIT 20",
                             "ORDER BY ts DESC NULLS LAST, id DESC LIMIT 4 OFFSET 3"}) {
    auto sql = std::string("SELECT * FROM lake.ties ") + suffix;
    query(con, "SET gpu_execution=false");
    auto expected = ducklake_result_bag(*query(con, sql));
    query(con, "SET gpu_execution=true");
    CHECK(ducklake_result_bag(*query(con, sql)) == expected);
  }
  auto prepared = con.Prepare("SELECT id FROM lake.events ORDER BY ts DESC LIMIT 2");
  REQUIRE_FALSE(prepared->HasError());
  CHECK(ducklake_result_bag(*prepared->Execute()) ==
        std::vector<std::string>{std::to_string(files * 1000 - 2) + "|",
                                 std::to_string(files * 1000 - 1) + "|"});
  query(con, "SET gpu_execution=false");
  query(con, "DELETE FROM lake.events WHERE id >= 1000");
  query(con, "SET gpu_execution=true");
  CHECK(ducklake_result_bag(*prepared->Execute()) == std::vector<std::string>{"998|", "999|"});
}

TEST_CASE("DuckLake reuses delete preparation capacity across files and scans",
          "[isolated_context][ducklake][ducklake_delete_budget]")
{
  auto files = GENERATE(1, 2, 8);
  INFO(files);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',10)");
  run("CREATE TABLE lake.q(i BIGINT)");
  for (int file = 0; file < files; ++file) {
    run("INSERT INTO lake.q SELECT i FROM range(" + std::to_string(file * 100) + "," +
        std::to_string((file + 1) * 100) + ") t(i)");
  }
  run("DELETE FROM lake.q WHERE i%100=0");
  run("INSERT INTO lake.q VALUES (-1),(-2)");
  std::vector<std::string> sql{"SELECT count(*),sum(i) FROM lake.q",
                               "SELECT count(*) FROM lake.q",
                               "SELECT i FROM lake.q WHERE i=50 OR i::VARCHAR LIKE '%9%'",
                               "SELECT count(*),sum(a.i) FROM lake.q a JOIN lake.q b ON a.i=b.i"};
  std::vector<std::vector<std::string>> expected;
  for (auto const& text : sql)
    expected.push_back(ducklake_result_bag(*run(text)));
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto const reserved = hosts[0]->get_total_reserved_memory();
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (size_t i = 0; i < sql.size(); ++i) {
      INFO(sql[i]);
      auto before = get_transparent_execution_stats(con);
      CHECK(ducklake_result_bag(*run(sql[i])) == expected[i]);
      auto after = get_transparent_execution_stats(con);
      CHECK(after.executions == before.executions + 1);
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
      CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    }
  }
  if (files > 1) {
    auto counters = runtime->physical_counters();
    std::atomic<unsigned> decoded{0};
    counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
      if (!footer && ++decoded == 2) throw std::runtime_error("delete visibility decode failure");
    };
    auto clear =
      std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
    run("SET enable_duckdb_fallback=true");
    auto before = get_transparent_execution_stats(con);
    CHECK(ducklake_result_bag(*run(sql[0])) == expected[0]);
    CHECK(decoded.load() >= 2);
    CHECK(get_transparent_execution_stats(con).runtime_fallbacks == before.runtime_fallbacks + 1);
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    clear.reset();
    run("SET enable_duckdb_fallback=false");
    CHECK(ducklake_result_bag(*run(sql[0])) == expected[0]);
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  }
}
