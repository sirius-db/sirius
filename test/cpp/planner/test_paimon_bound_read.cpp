// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "planner/paimon_bound_read.hpp"
#include "planner/verified_scan_callbacks.hpp"
#include "utils/paimon_binding_fixture.hpp"

#include <catch2/catch_all.hpp>
#include <dlfcn.h>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string_view>

namespace {
using sirius::planner::paimon_bound_read;
using sirius::planner::paimon_read_provider;
using namespace sirius::test::paimon_fixture;

void sql(duckdb::Connection& con, std::string const& text)
{
  auto result = con.Query(text);
  INFO((result->HasError() ? result->GetError() : "success"));
  REQUIRE_FALSE(result->HasError());
}

std::string quote(std::string const& text) { return duckdb::Value(text).ToSQLString(); }

struct binding {
  duckdb::BoundTableFunction function;
  duckdb::unique_ptr<duckdb::FunctionData> data;
};

binding bind_paimon(duckdb::Connection& con, std::string const& warehouse, int entry)
{
  con.BeginTransaction();
  try {
    binding result;
    auto& context = *con.context;
    if (entry == 2) {
      auto& table = duckdb::Catalog::GetEntry<duckdb::TableCatalogEntry>(
        context, duckdb::QualifiedName("pm", "test", "t"));
      result.function = duckdb::BoundTableFunction(table.GetScanFunction(context, result.data));
    } else {
      auto& functions = duckdb::Catalog::GetEntry<duckdb::TableFunctionCatalogEntry>(
        context, duckdb::QualifiedName(SYSTEM_CATALOG, DEFAULT_SCHEMA, "paimon_scan"));
      duckdb::vector<duckdb::LogicalType> arguments(entry == 0 ? 1 : 3,
                                                    duckdb::LogicalType::VARCHAR);
      result.function =
        duckdb::BoundTableFunction(functions.functions.GetFunctionByArguments(context, arguments));
      duckdb::vector<duckdb::Value> values =
        entry == 0 ? duckdb::vector<duckdb::Value>{duckdb::Value(warehouse + "/test.db/t")}
                   : duckdb::vector<duckdb::Value>{
                       duckdb::Value(warehouse), duckdb::Value("test"), duckdb::Value("t")};
      duckdb::named_argument_map_t parameters;
      result.function.SetCallArguments(values, parameters);
      duckdb::vector<duckdb::LogicalType> input_types, return_types;
      duckdb::vector<duckdb::Identifier> input_names, names;
      duckdb::TableFunctionRef reference;
      duckdb::TableFunctionBindInput input(
        values, parameters, input_types, input_names, nullptr, nullptr, result.function, reference);
      result.data = result.function.bind(context, input, return_types, names);
    }
    con.Commit();
    return result;
  } catch (...) {
    con.Rollback();
    throw;
  }
}

struct warehouse_directory {
  std::filesystem::path path;
  warehouse_directory()
  {
    auto pattern =
      (std::filesystem::temp_directory_path() / "sirius-paimon-binding-XXXXXX").string();
    auto* created = ::mkdtemp(pattern.data());
    if (!created) throw std::runtime_error("Cannot create Paimon binding fixture directory");
    path = created;
  }
  ~warehouse_directory()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

std::string_view text(PaimonReadString value) { return {value.data, value.size}; }

void load(duckdb::Connection& con)
{
  auto const* path = std::getenv("SIRIUS_PAIMON_TEST_EXTENSION");
  REQUIRE(path != nullptr);
  REQUIRE(std::filesystem::is_regular_file(path));
  sql(con, "LOAD " + quote(path));
}

void dynamic_config(duckdb::DBConfig& config)
{
  config.options.load_extensions = false;
  config.SetOptionByName("allow_unsigned_extensions", duckdb::Value::BOOLEAN(true));
}

struct fixture_mode {
  std::optional<std::string> previous;
  explicit fixture_mode(int value)
  {
    auto const* old = std::getenv("SIRIUS_PAIMON_BINDING_FIXTURE_MODE");
    if (old) previous = old;
    REQUIRE(::setenv("SIRIUS_PAIMON_BINDING_FIXTURE_MODE", std::to_string(value).c_str(), 1) == 0);
  }
  ~fixture_mode()
  {
    if (previous)
      ::setenv("SIRIUS_PAIMON_BINDING_FIXTURE_MODE", previous->c_str(), 1);
    else
      ::unsetenv("SIRIUS_PAIMON_BINDING_FIXTURE_MODE");
  }
};

std::filesystem::path fixture_path()
{
  return std::filesystem::read_symlink("/proc/self/exe").parent_path() /
         "paimon_binding_fixture/paimon.duckdb_extension";
}

uint64_t fixture_releases()
{
  auto* module = ::dlopen(fixture_path().c_str(), RTLD_NOW | RTLD_NOLOAD);
  REQUIRE(module != nullptr);
  auto close = [](void* value) { ::dlclose(value); };
  std::unique_ptr<void, decltype(close)> owner(module, close);
  auto get = reinterpret_cast<uint64_t (*)()>(::dlsym(module, "sirius_paimon_fixture_released"));
  REQUIRE(get != nullptr);
  return get();
}
}  // namespace

TEST_CASE("Paimon binding requires a loaded ABI-checked provider", "[paimon_binding]")
{
  duckdb::DBConfig config;
  dynamic_config(config);
  duckdb::DuckDB db(nullptr, &config);
  CHECK_THROWS_WITH(paimon_read_provider::from_loaded(*db.instance),
                    "Paimon descriptor refused: provider is not dynamically loaded");
  duckdb::Connection con(db);
  sql(con, "SET allow_extensions_metadata_mismatch=true");
  CHECK_THROWS_WITH(paimon_read_provider::from_loaded(*db.instance),
                    "Paimon descriptor refused: extension ABI checking is disabled");
}

TEST_CASE("Shared callback verification includes semantic settings", "[paimon_binding]")
{
  auto trusted   = duckdb::TableScanFunction::GetFunction();
  auto candidate = duckdb::BoundTableFunction(trusted);
  REQUIRE(sirius::planner::detail::same_scan_callbacks(trusted, candidate));
  SECTION("filter pruning") { candidate.filter_prune = !trusted.filter_prune; }
  SECTION("projection pushdown") { candidate.projection_pushdown = !trusted.projection_pushdown; }
  SECTION("initialization") { candidate.init_global = nullptr; }
  SECTION("statistics")
  {
    REQUIRE(trusted.statistics_extended != nullptr);
    candidate.statistics_extended = nullptr;
  }
  SECTION("table entry")
  {
    REQUIRE(trusted.get_table_entry != nullptr);
    candidate.get_table_entry = nullptr;
  }
  SECTION("SQL reconstruction")
  {
    REQUIRE(trusted.to_sql != nullptr);
    candidate.to_sql = nullptr;
  }
  SECTION("return mode")
  {
    candidate.return_type = duckdb::TableFunctionReturnType::SET_RETURNING_FUNCTION;
  }
  SECTION("statement result") { candidate.call_return_type = duckdb::StatementReturnType::NOTHING; }
  SECTION("serialization verification")
  {
    candidate.verify_serialization = !trusted.verify_serialization;
  }
  SECTION("name") { candidate.SetName("forged_scan"); }
  REQUIRE_FALSE(sirius::planner::detail::same_scan_callbacks(trusted, candidate));
}

TEST_CASE("Paimon acquisition rejects malformed and foreign descriptor APIs", "[paimon_binding]")
{
  // These are real independent hidden DSOs, with synthetic descriptor responses.
  auto const mode = GENERATE(bad_magic,
                             bad_version,
                             bad_size,
                             missing_callback,
                             foreign_callback,
                             missing_accepts,
                             missing_describe,
                             missing_release);
  fixture_mode selected(mode);
  duckdb::DBConfig config;
  dynamic_config(config);
  duckdb::DuckDB db(nullptr, &config);
  duckdb::Connection con(db);
  sql(con, "LOAD " + quote(fixture_path().string()));
  CHECK_THROWS_WITH(paimon_read_provider::from_loaded(*db.instance),
                    mode < 4 ? "Paimon descriptor refused: unsupported bridge ABI"
                             : "Paimon descriptor refused: foreign or missing bridge callback");
}

TEST_CASE("Paimon acquisition rejects an initializer supplied by a dependency", "[paimon_binding]")
{
  fixture_mode selected(valid);
  duckdb::DBConfig config;
  dynamic_config(config);
  duckdb::DuckDB db(nullptr, &config);
  duckdb::Connection con(db);
  auto proxy =
    fixture_path().parent_path().parent_path() / "paimon_binding_proxy/paimon.duckdb_extension";
  sql(con, "LOAD " + quote(proxy.string()));
  CHECK_THROWS_WITH(paimon_read_provider::from_loaded(*db.instance),
                    "Paimon descriptor refused: provider initializer is unavailable or foreign");
}

TEST_CASE("Paimon retain refuses incomplete evidence and releases failed outputs",
          "[paimon_binding]")
{
  auto const mode = GENERATE(null_output, partial_error, describe_error, invalid_view);
  fixture_mode selected(mode);
  duckdb::DBConfig config;
  dynamic_config(config);
  duckdb::DuckDB db(nullptr, &config);
  duckdb::Connection con(db);
  sql(con, "LOAD " + quote(fixture_path().string()));
  auto provider = paimon_read_provider::from_loaded(*db.instance);
  auto bound    = bind_paimon(con, "/ignored", 0);
  auto before   = fixture_releases();
  CHECK_THROWS(provider.retain(bound.function, bound.data.get()));
  CHECK(fixture_releases() - before == (mode == null_output ? 0 : 1));
}

TEST_CASE("Paimon descriptor move assignment releases each owner once", "[paimon_binding]")
{
  auto const mode = GENERATE(valid, valid_v2);
  fixture_mode selected(mode);
  duckdb::DBConfig config;
  dynamic_config(config);
  duckdb::DuckDB db(nullptr, &config);
  duckdb::Connection con(db);
  sql(con, "LOAD " + quote(fixture_path().string()));
  auto provider = paimon_read_provider::from_loaded(*db.instance);
  auto bound    = bind_paimon(con, "/ignored", 0);
  auto before   = fixture_releases();
  {
    auto first  = provider.retain(bound.function, bound.data.get());
    auto second = provider.retain(bound.function, bound.data.get());
    first       = std::move(second);
    CHECK(fixture_releases() - before == 1);
    CHECK(first.view().snapshot_id == 5);
    CHECK_THROWS(second.view());
  }
  CHECK(fixture_releases() - before == 2);
}

TEST_CASE("Paimon descriptor consumer authenticates real loaded bindings",
          "[.][paimon_binding_dynamic]")
{
  int const entry = GENERATE(0, 1, 2);
  warehouse_directory warehouse;
  duckdb::DBConfig config;
  dynamic_config(config);
  duckdb::DuckDB db(nullptr, &config);
  duckdb::Connection con(db);
  load(con);
  sql(con, "ATTACH " + quote(warehouse.path.string()) + " AS pm (TYPE paimon)");
  sql(con, "CREATE SCHEMA pm.test");
  sql(con, "CREATE TABLE pm.test.t (id INTEGER, value INTEGER)");
  sql(con, "INSERT INTO pm.test.t VALUES (1, 10)");
  auto bound    = bind_paimon(con, warehouse.path.string(), entry);
  auto provider = paimon_read_provider::from_loaded(*db.instance);
  auto original = provider.retain(bound.function, bound.data.get());
  REQUIRE(original.view().has_snapshot == 1);
  auto const snapshot = original.view().snapshot_id;
  auto const schema   = std::string(text(original.view().schema_json));

  SECTION("retained identity survives later writes and binding destruction")
  {
    sql(con, "INSERT INTO pm.test.t VALUES (2, 20)");
    auto fresh_binding = bind_paimon(con, warehouse.path.string(), entry);
    auto fresh         = provider.retain(fresh_binding.function, fresh_binding.data.get());
    CHECK(fresh.view().snapshot_id > snapshot);
    bound.data.reset();
    CHECK(original.view().snapshot_id == snapshot);
    CHECK(text(original.view().schema_json) == schema);
    auto moved = std::move(original);
    CHECK_THROWS(original.view());
    CHECK(moved.view().snapshot_id == snapshot);
  }
  SECTION("foreign info is rejected before wrapper access")
  {
    auto forged          = bound.function;
    forged.function_info = duckdb::make_shared_ptr<duckdb::TableFunctionInfo>();
    CHECK_THROWS_WITH(provider.retain(forged, bound.data.get()),
                      "Paimon descriptor refused: unverified callbacks or function-info");
  }
  SECTION("matching scan body cannot hide semantic callback replacement")
  {
    auto forged         = bound.function;
    forged.filter_prune = !forged.filter_prune;
    CHECK_THROWS(provider.retain(forged, bound.data.get()));
    forged            = bound.function;
    forged.init_local = nullptr;
    CHECK_THROWS(provider.retain(forged, bound.data.get()));
    CHECK_THROWS(provider.retain(bound.function, nullptr));
    duckdb::TableFunctionData foreign_bind;
    CHECK_THROWS(provider.retain(bound.function, &foreign_bind));
  }
  SECTION("table, SQL, return and serialization semantics are authenticated")
  {
    auto forged            = bound.function;
    forged.get_table_entry = +[](duckdb::optional_ptr<duckdb::FunctionData const>)
      -> duckdb::optional_ptr<duckdb::TableCatalogEntry> { return nullptr; };
    CHECK_THROWS(provider.retain(forged, bound.data.get()));
    forged        = bound.function;
    forged.to_sql = +[](duckdb::ClientContext&, duckdb::LogicalGet const&) {
      return duckdb::TableFunctionToSQLResult{};
    };
    CHECK_THROWS(provider.retain(forged, bound.data.get()));
    forged             = bound.function;
    forged.return_type = duckdb::TableFunctionReturnType::SET_RETURNING_FUNCTION;
    CHECK_THROWS(provider.retain(forged, bound.data.get()));
    forged                  = bound.function;
    forged.call_return_type = duckdb::StatementReturnType::NOTHING;
    CHECK_THROWS(provider.retain(forged, bound.data.get()));
    forged                      = bound.function;
    forged.verify_serialization = !forged.verify_serialization;
    CHECK_THROWS(provider.retain(forged, bound.data.get()));
    forged = bound.function;
    forged.SetName("forged_paimon_scan");
    CHECK_THROWS(provider.retain(forged, bound.data.get()));
  }
  SECTION("mutable catalog cannot grant trust")
  {
    auto replacement          = duckdb::TableFunction(bound.function.GetName(), {}, nullptr);
    replacement.function_info = duckdb::make_shared_ptr<duckdb::TableFunctionInfo>();
    duckdb::ExtensionLoader loader(*db.instance, "binding_replacement");
    duckdb::CreateTableFunctionInfo info(replacement);
    info.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
    loader.RegisterFunction(std::move(info));
    auto independent = paimon_read_provider::from_loaded(*db.instance);
    CHECK_THROWS(independent.retain(duckdb::BoundTableFunction(replacement), bound.data.get()));
    // The independently obtained original definition remains the trust authority.
    CHECK(independent.retain(bound.function, bound.data.get()).view().snapshot_id == snapshot);
  }
}

TEST_CASE("Paimon descriptors keep module ownership after database teardown",
          "[.][paimon_binding_dynamic]")
{
  warehouse_directory warehouse;
  std::optional<paimon_bound_read> retained;
  {
    duckdb::DBConfig config;
    dynamic_config(config);
    duckdb::DuckDB db(nullptr, &config);
    duckdb::Connection con(db);
    CHECK_THROWS(paimon_read_provider::from_loaded(*db.instance));
    load(con);
    auto provider = paimon_read_provider::from_loaded(*db.instance);
    sql(con, "ATTACH " + quote(warehouse.path.string()) + " AS pm (TYPE paimon)");
    sql(con, "CREATE SCHEMA pm.test");
    sql(con, "CREATE TABLE pm.test.t (id INTEGER, value INTEGER)");
    auto bound = bind_paimon(con, warehouse.path.string(), 0);
    retained.emplace(provider.retain(bound.function, bound.data.get()));
    CHECK(retained->view().has_snapshot == 0);
    CHECK(retained->view().snapshot_id == 0);
    sql(con, "INSERT INTO pm.test.t VALUES (1, 10)");
    CHECK(retained->view().has_snapshot == 0);
  }
  REQUIRE(retained.has_value());
  CHECK(retained->view().has_snapshot == 0);
  CHECK(text(retained->view().table) == "t");
  retained.reset();
}
