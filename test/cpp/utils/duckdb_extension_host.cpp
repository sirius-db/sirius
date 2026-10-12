// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "utils/loadable_extension.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/scan_callback_replacements.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/multi_file/multi_file_function.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/main/extension_helper.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>

namespace {
using sirius::test::sql_literal;

std::string phase()
{
  auto const* value = std::getenv("SIRIUS_REGISTRY_TRUST_PHASE");
  REQUIRE(value != nullptr);
  return value;
}

void load_sirius(duckdb::Connection& connection)
{
  auto const* config = std::getenv("SIRIUS_TEST_SHARED_CONFIG_OVERRIDE");
  REQUIRE(config != nullptr);
  setenv("SIRIUS_CONFIG_FILE", config, 1);
  setenv("SIRIUS_ENABLE_TEST_OPTIONS", "1", 1);
  unsetenv("SIRIUS_DISABLE");
  auto const extension = sirius::test::loadable_extension_path();
  REQUIRE(std::filesystem::is_regular_file(extension));
  auto result = connection.Query("LOAD " + sql_literal(extension.string()));
  INFO((result->HasError() ? result->GetError() : "success"));
  REQUIRE_FALSE(result->HasError());
  REQUIRE_FALSE(connection.Query("SET gpu_execution=false")->HasError());
}

duckdb::unique_ptr<duckdb::FunctionData> iceberg_bind(duckdb::ClientContext&,
                                                      duckdb::TableFunctionBindInput&,
                                                      duckdb::vector<duckdb::LogicalType>& types,
                                                      duckdb::vector<std::string>& names)
{
  types = {duckdb::LogicalType::INTEGER};
  names = {"id"};
  return duckdb::make_uniq<duckdb::MultiFileBindData>();
}

void require_unverified(duckdb::Connection& connection, std::string const& query)
{
  auto rejected = connection.Query(query);
  INFO((rejected->HasError() ? rejected->GetError() : "success"));
  REQUIRE(rejected->HasError());
  CHECK(rejected->GetError().find("unverified callbacks") != std::string::npos);
}
}  // namespace

TEST_CASE("Scan registry rejects replacement before Sirius load child", "[.][extension_host]")
{
  auto const test_phase = phase();
  REQUIRE(test_phase.starts_with("preload_dynamic"));
  duckdb::DBConfig config;
  config.options.load_extensions = false;
  config.SetOptionByName("allow_unsigned_extensions", duckdb::Value::BOOLEAN(true));
  duckdb::DuckDB database(nullptr, &config);
  duckdb::ExtensionHelper::LoadExtension(database, "parquet");
  duckdb::Connection connection(database);
  sirius::test::scratch_dir files("registry_preload");
  REQUIRE_FALSE(connection
                  .Query("COPY (SELECT 1::INTEGER AS i) TO " + files.file_literal("input.parquet") +
                         " (FORMAT PARQUET)")
                  ->HasError());

  duckdb::ExtensionLoader loader(*database.instance, "registry_preload_test");
  auto original    = loader.GetTableFunction("read_parquet").functions.functions.front();
  auto replacement = original;
  sirius::test::replace_callback(replacement, test_phase);
  duckdb::CreateTableFunctionInfo info(replacement);
  info.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
  loader.RegisterFunction(std::move(info));
  sirius::test::require_registered_callbacks(
    loader.GetTableFunction("read_parquet").functions.functions.front(), replacement);

  load_sirius(connection);
  auto const query = "SELECT i FROM read_parquet(" + files.file_literal("input.parquet") + ")";
  if (test_phase.ends_with("init_global") || test_phase.ends_with("init_local")) {
    auto cpu = connection.Query(query);
    REQUIRE(cpu->HasError());
    CHECK(cpu->GetError().find(test_phase.ends_with("init_global")
                                 ? "replacement global initializer"
                                 : "replacement local initializer") != std::string::npos);
  }
  REQUIRE_FALSE(connection.Query("SET gpu_execution=true")->HasError());
  REQUIRE_FALSE(connection.Query("SET enable_duckdb_fallback=false")->HasError());
  require_unverified(connection, query);

  duckdb::CreateTableFunctionInfo restore(original);
  restore.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
  loader.RegisterFunction(std::move(restore));
  auto accepted = connection.Query(query);
  INFO((accepted->HasError() ? accepted->GetError() : "success"));
  REQUIRE_FALSE(accepted->HasError());
  CHECK(accepted->GetValue(0, 0).GetValue<int32_t>() == 1);
}

TEST_CASE("Iceberg trust bootstrap load order child", "[.][extension_host]")
{
  auto const test_phase = phase();
  REQUIRE((test_phase == "iceberg_first_dynamic" || test_phase == "iceberg_last_dynamic"));
  bool const iceberg_first = test_phase == "iceberg_first_dynamic";
  duckdb::DBConfig config;
  config.options.load_extensions = false;
  config.SetOptionByName("allow_unsigned_extensions", duckdb::Value::BOOLEAN(true));
  duckdb::DuckDB database(nullptr, &config);
  duckdb::ExtensionHelper::LoadExtension(database, "parquet");
  duckdb::ExtensionHelper::LoadExtension(database, "core_functions");
  duckdb::Connection connection(database);
  auto load_iceberg = [&] {
    auto result = connection.Query("LOAD iceberg");
    INFO((result->HasError() ? result->GetError() : "success"));
    REQUIRE_FALSE(result->HasError());
  };
  if (iceberg_first)
    load_iceberg();
  else
    load_sirius(connection);

  duckdb::ExtensionLoader loader(*database.instance, "iceberg_bootstrap_test");
  loader.RegisterFunction(duckdb::TableFunction(
    "iceberg_scan", {duckdb::LogicalType::INTEGER}, sirius::test::fake_scan, iceberg_bind));
  if (!iceberg_first) {
    // Cache an admission failure before Iceberg loads; its bootstrap must replace it.
    REQUIRE_FALSE(connection.Query("SET gpu_execution=true")->HasError());
    REQUIRE_FALSE(connection.Query("SET enable_duckdb_fallback=false")->HasError());
    require_unverified(connection, "SELECT * FROM iceberg_scan(42)");
    REQUIRE_FALSE(connection.Query("SET gpu_execution=false")->HasError());
  }
  if (iceberg_first)
    load_sirius(connection);
  else
    load_iceberg();

  REQUIRE_FALSE(connection.Query("SET gpu_execution=true")->HasError());
  REQUIRE_FALSE(connection.Query("SET enable_duckdb_fallback=false")->HasError());
  require_unverified(connection, "SELECT * FROM iceberg_scan(42)");
  REQUIRE_FALSE(connection.Query("SET sirius_test_inject_transparent_gpu_error='t6'")->HasError());
  auto const path =
    std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/integration/data/iceberg_snapshot_deletes";
  auto genuine = connection.Query("SELECT sum(count) FROM iceberg_scan(" + sql_literal(path) +
                                  ", snapshot_from_id=9400000000000002)");
  INFO((genuine->HasError() ? genuine->GetError() : "success"));
  REQUIRE(genuine->HasError());
  CHECK(genuine->GetError().find("injected transparent GPU failure: t6") != std::string::npos);
}
