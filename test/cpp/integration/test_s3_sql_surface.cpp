/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "utils/s3_container.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string env_or(std::string_view name, std::string fallback = {})
{
  auto const* value = std::getenv(std::string{name}.c_str());
  return value ? std::string{value} : std::move(fallback);
}

bool truthy_env(std::string_view name)
{
  auto value = env_or(name);
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

struct s3_test_env {
  std::string endpoint;
  std::string https_endpoint;
  std::string ca_bundle_path;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::string session_token;
  fs::path local_dir;
};

std::optional<s3_test_env> load_s3_test_env()
{
  if (!sirius::test::ensure_s3_container_env()) { return std::nullopt; }

  auto endpoint   = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket     = env_or("SIRIUS_TEST_S3_BUCKET");
  auto local_dir  = env_or("SIRIUS_TEST_S3_LOCAL_DIR");

  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty() ||
      local_dir.empty()) {
    return std::nullopt;
  }

  return s3_test_env{std::move(endpoint),
                     env_or("SIRIUS_TEST_S3_HTTPS_ENDPOINT"),
                     env_or("SIRIUS_TEST_S3_CA_BUNDLE"),
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket),
                     env_or("SIRIUS_TEST_S3_SESSION_TOKEN"),
                     fs::path{std::move(local_dir)}};
}

bool should_skip_s3_env(std::optional<s3_test_env> const& env)
{
  if (env) { return false; }
  if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
    FAIL("SIRIUS_TEST_S3_* environment is required in strict mode");
  }
  SUCCEED("SIRIUS_TEST_S3_* not set; skipping live S3 SQL-surface test");
  return true;
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::string sql_quote(std::string_view value)
{
  std::string out{"'"};
  for (char c : value) {
    if (c == '\'') { out.push_back('\''); }
    out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

std::string yaml_quote(std::string const& value) { return sql_quote(value); }

std::string read_text_file(fs::path const& path)
{
  std::ifstream in(path);
  REQUIRE(in);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

void load_sirius_extension(duckdb::DuckDB& db)
{
  try {
    db.LoadStaticExtension<duckdb::SiriusExtension>();
  } catch (std::exception const& e) {
    auto const msg = std::string{e.what()};
    if (msg.find("already exists") == std::string::npos &&
        msg.find("already loaded") == std::string::npos) {
      throw;
    }
  }
}

struct sirius_memory_limits {
  std::string gpu_usage{"256 MiB"};
  std::string gpu_reservation{"128 MiB"};
  std::string host_capacity{"512 MiB"};
  std::string disk_capacity{"2 GiB"};
  std::optional<bool> enable_prefetch_cache;
  bool rest_perf_instrumentation{false};
};

sirius_memory_limits large_sirius_memory_limits(bool enable_prefetch_cache)
{
  sirius_memory_limits limits;
  limits.gpu_usage                 = "5 GiB";
  limits.gpu_reservation           = "2 GiB";
  limits.host_capacity             = "8 GiB";
  limits.disk_capacity             = "32 GiB";
  limits.enable_prefetch_cache     = enable_prefetch_cache;
  limits.rest_perf_instrumentation = true;
  return limits;
}

class sirius_config_env_guard {
 public:
  explicit sirius_config_env_guard(s3_test_env const& env,
                                   sirius_memory_limits limits             = {},
                                   std::optional<std::string> signing_mode = std::nullopt,
                                   std::optional<std::string> endpoint     = std::nullopt,
                                   std::optional<std::string> ca_bundle    = std::nullopt,
                                   std::optional<bool> tls_verify          = std::nullopt)
  {
    if (auto* current = std::getenv("SIRIUS_CONFIG_FILE"); current != nullptr) {
      had_original_config_env_ = true;
      original_config_env_     = current;
    }
    if (auto* current = std::getenv("SIRIUS_DISABLE"); current != nullptr) {
      had_original_disable_env_ = true;
      original_disable_env_     = current;
    }

    auto const unique = std::to_string(reinterpret_cast<std::uintptr_t>(this));
    dir_              = fs::temp_directory_path() / ("sirius_b3_s3_sql_" + unique);
    config_path_      = dir_ / "sirius.yaml";
    fs::create_directories(dir_);

    std::ofstream out(config_path_);
    auto const object_endpoint = endpoint.value_or(env.endpoint);
    out << "sirius:\n"
           "  space:\n"
           "    gpu:\n"
           "      - device_id: 0\n"
           "        per_stream_reservation: false\n"
           "        reservation_limit_fraction: 0.4\n"
           "        downgrade_trigger_fraction: 0.8\n"
           "        downgrade_stop_fraction: 0.6\n"
           "        memory_capacity: "
        << limits.gpu_usage
        << "\n"
           "    host:\n"
           "      - numa_id: -1\n"
           "        reservation_limit_fraction: 0.9\n"
           "        downgrade_trigger_fraction: 0.8\n"
           "        downgrade_stop_fraction: 0.6\n"
           "        memory_capacity: "
        << limits.host_capacity
        << "\n"
           "    disk:\n"
           "      - disk_id: 0\n"
           "        mount_path: "
        << yaml_quote((dir_ / "disk_memory").string())
        << "\n"
           "        memory_capacity: "
        << limits.disk_capacity
        << "\n"
           "  executor:\n"
           "    scan_manager:\n";
    if (limits.enable_prefetch_cache.has_value()) {
      out << "      enable_prefetch_cache: " << (*limits.enable_prefetch_cache ? "true" : "false")
          << "\n";
    }
    out << "      object_store:\n"
           "        endpoint: "
        << yaml_quote(object_endpoint)
        << "\n"
           "        region: "
        << yaml_quote(env.region)
        << "\n"
           "        access_key: "
        << yaml_quote(env.access_key)
        << "\n"
           "        secret_key: "
        << yaml_quote(env.secret_key) << "\n";
    if (!env.session_token.empty()) {
      out << "        session_token: " << yaml_quote(env.session_token) << "\n";
    }
    if (signing_mode.has_value()) {
      out << "        signing_mode: " << yaml_quote(*signing_mode) << "\n";
    }
    if (ca_bundle.has_value() && !ca_bundle->empty()) {
      out << "        ca_bundle_path: " << yaml_quote(*ca_bundle) << "\n";
    }
    if (tls_verify.has_value()) {
      out << "        tls_verify: " << (*tls_verify ? "true" : "false") << "\n";
    } else {
      out << "        tls_verify: false\n";
    }
    out << "      rest:\n"
           "        max_connections: 8\n"
           "        request_timeout_s: 30\n";
    if (limits.rest_perf_instrumentation) { out << "        perf_instrumentation: true\n"; }
    out.close();
    REQUIRE(out);

    setenv("SIRIUS_CONFIG_FILE", config_path_.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
  }

  ~sirius_config_env_guard()
  {
    if (had_original_config_env_) {
      setenv("SIRIUS_CONFIG_FILE", original_config_env_.c_str(), 1);
    } else {
      unsetenv("SIRIUS_CONFIG_FILE");
    }
    if (had_original_disable_env_) {
      setenv("SIRIUS_DISABLE", original_disable_env_.c_str(), 1);
    } else {
      unsetenv("SIRIUS_DISABLE");
    }
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  [[nodiscard]] fs::path const& config_path() const noexcept { return config_path_; }

 private:
  fs::path dir_;
  fs::path config_path_;
  std::string original_config_env_;
  std::string original_disable_env_;
  bool had_original_config_env_{false};
  bool had_original_disable_env_{false};
};

class s3_sql_fixture {
 public:
  explicit s3_sql_fixture(s3_test_env const& env,
                          sirius_memory_limits limits             = {},
                          std::optional<std::string> signing_mode = std::nullopt,
                          std::optional<std::string> endpoint     = std::nullopt,
                          std::optional<std::string> ca_bundle    = std::nullopt,
                          std::optional<bool> tls_verify          = std::nullopt)
    : config_env(env,
                 std::move(limits),
                 std::move(signing_mode),
                 std::move(endpoint),
                 std::move(ca_bundle),
                 tls_verify),
      db(nullptr),
      con(db)
  {
    load_sirius_extension(db);
    REQUIRE(con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state"));
    setenv("SIRIUS_DISABLE", "1", 1);
  }

  sirius_config_env_guard config_env;
  duckdb::DuckDB db;
  duckdb::Connection con;
};

std::unique_ptr<duckdb::MaterializedQueryResult> require_query_ok(duckdb::Connection& con,
                                                                  std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  return std::unique_ptr<duckdb::MaterializedQueryResult>(
    static_cast<duckdb::MaterializedQueryResult*>(result.release()));
}

std::string gpu_execution_sql(std::string const& inner_sql)
{
  std::string escaped;
  escaped.reserve(inner_sql.size() + 8);
  for (char c : inner_sql) {
    if (c == '\'') { escaped.push_back('\''); }
    escaped.push_back(c);
  }
  return "SELECT * FROM gpu_execution('" + escaped + "')";
}

std::vector<std::vector<std::string>> collect_rows(duckdb::MaterializedQueryResult& result)
{
  std::vector<std::vector<std::string>> rows;
  for (duckdb::idx_t r = 0; r < result.RowCount(); ++r) {
    std::vector<std::string> row;
    row.reserve(result.ColumnCount());
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); ++c) {
      row.push_back(result.GetValue(c, r).ToString());
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

void check_rows_equal_with_tolerant_columns(duckdb::MaterializedQueryResult& actual,
                                            duckdb::MaterializedQueryResult& expected,
                                            std::vector<duckdb::idx_t> const& tolerant_columns = {})
{
  REQUIRE(actual.RowCount() == expected.RowCount());
  REQUIRE(actual.ColumnCount() == expected.ColumnCount());
  for (duckdb::idx_t r = 0; r < actual.RowCount(); ++r) {
    for (duckdb::idx_t c = 0; c < actual.ColumnCount(); ++c) {
      auto const actual_value   = actual.GetValue(c, r).ToString();
      auto const expected_value = expected.GetValue(c, r).ToString();
      INFO("row=" << r << " column=" << c);
      if (std::find(tolerant_columns.begin(), tolerant_columns.end(), c) !=
          tolerant_columns.end()) {
        CHECK(std::stod(actual_value) ==
              Approx(std::stod(expected_value)).epsilon(1e-10).margin(1e-8));
      } else {
        CHECK(actual_value == expected_value);
      }
    }
  }
}

fs::path local_parquet_path(s3_test_env const& env, std::string_view table)
{
  return env.local_dir / "parquet" / (std::string{table} + ".parquet");
}

fs::path local_sf10_lineitem_path()
{
  auto override_path = env_or("SIRIUS_PR6_LARGE_LOCAL_PARQUET");
  if (!override_path.empty()) { return fs::path{override_path}; }
  auto work_dir = env_or("SIRIUS_BENCH_WORK_DIR");
  if (!work_dir.empty()) { return fs::path{work_dir} / "lineitem_sf10.parquet"; }
  return fs::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" / "s3" / "fixtures" /
         "generated" / "lineitem_sf10.parquet";
}

std::string local_parquet_scan(s3_test_env const& env, std::string_view table)
{
  return "read_parquet(" + sql_quote(local_parquet_path(env, table).string()) + ")";
}

std::string local_parquet_file_scan(fs::path const& path)
{
  return "read_parquet(" + sql_quote(path.string()) + ")";
}

std::string s3_parquet_scan(s3_test_env const& env, std::string_view table)
{
  auto const key = "parquet/" + std::string{table} + ".parquet";
  return "read_parquet(" + sql_quote(s3_uri(env.bucket, key)) + ")";
}

std::string s3_sirius_parquet_scan(s3_test_env const& env, std::string_view table)
{
  auto const key = "parquet/" + std::string{table} + ".parquet";
  return "sirius_read_parquet(" + sql_quote(s3_uri(env.bucket, key)) + ")";
}

std::string sf10_lineitem_key()
{
  return env_or("SIRIUS_PR6_LARGE_S3_KEY",
                env_or("SIRIUS_BENCH_S3_KEY", "tpch/lineitem_sf10.parquet"));
}

std::string s3_large_lineitem_uri(s3_test_env const& env)
{
  return s3_uri(env.bucket, sf10_lineitem_key());
}

std::string s3_large_lineitem_scan(s3_test_env const& env)
{
  return "read_parquet(" + sql_quote(s3_large_lineitem_uri(env)) + ")";
}

std::string s3_sirius_large_lineitem_scan(s3_test_env const& env)
{
  return "sirius_read_parquet(" + sql_quote(s3_large_lineitem_uri(env)) + ")";
}

duckdb::SiriusContext& require_sirius_context(s3_sql_fixture& fixture)
{
  auto sirius_ctx =
    fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  return *sirius_ctx;
}

sirius::io::rest::rest_ioctx& require_rest_ioctx(s3_sql_fixture& fixture, std::string const& uri)
{
  auto& sirius_ctx = require_sirius_context(fixture);
  auto datasource  = sirius_ctx.get_scan_manager().create_datasource(uri);
  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  auto* rest_ctx = dynamic_cast<sirius::io::rest::rest_ioctx*>(datasource->io_ctx().get());
  REQUIRE(rest_ctx != nullptr);
  return *rest_ctx;
}

std::uint64_t rest_chunk_get_count(s3_sql_fixture& fixture, std::string const& uri)
{
  return require_rest_ioctx(fixture, uri).perf_snapshot().chunk_get_count;
}

std::uint64_t rest_terminal_failures(s3_sql_fixture& fixture, std::string const& uri)
{
  return require_rest_ioctx(fixture, uri).perf_snapshot().terminal_failures_total;
}

struct large_lineitem_fixture {
  std::string uri;
  fs::path local_path;
  duckdb::idx_t total_num_rows{0};
};

std::optional<large_lineitem_fixture> read_large_lineitem_fixture(s3_sql_fixture& fixture,
                                                                  s3_test_env const& env)
{
  if (!truthy_env("SIRIUS_TEST_S3_LARGE")) {
    SUCCEED("SIRIUS_TEST_S3_LARGE not set; skipping large S3 SQL test");
    return std::nullopt;
  }

  large_lineitem_fixture out;
  out.uri        = s3_large_lineitem_uri(env);
  out.local_path = local_sf10_lineitem_path();
  if (!fs::exists(out.local_path)) {
    if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
      FAIL("SF10 local parquet fixture is required in strict mode: " + out.local_path.string());
    }
    SUCCEED("SF10 local parquet fixture is absent; skipping large S3 SQL test");
    return std::nullopt;
  }

  try {
    auto& sirius_ctx   = require_sirius_context(fixture);
    auto bind_info     = sirius_ctx.get_scan_manager().describe_parquet(out.uri);
    out.total_num_rows = static_cast<duckdb::idx_t>(bind_info.total_num_rows);
  } catch (std::exception const& e) {
    if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
      FAIL("SF10 S3 parquet fixture is required in strict mode at " + out.uri + ": " + e.what());
    }
    SUCCEED("SF10 S3 parquet fixture is absent; skipping large S3 SQL test");
    return std::nullopt;
  }
  return out;
}

duckdb::idx_t local_parquet_row_count(s3_test_env const& env, std::string_view table)
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto result = require_query_ok(con, "SELECT count(*) FROM " + local_parquet_scan(env, table));
  REQUIRE(result->RowCount() == 1);
  auto const rows = result->GetValue(0, 0).GetValue<int64_t>();
  REQUIRE(rows >= 0);
  return static_cast<duckdb::idx_t>(rows);
}

duckdb::idx_t local_parquet_file_row_count(fs::path const& path)
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto result =
    require_query_ok(con, "SELECT count(l_orderkey) FROM " + local_parquet_file_scan(path));
  REQUIRE(result->RowCount() == 1);
  auto const rows = result->GetValue(0, 0).GetValue<int64_t>();
  REQUIRE(rows >= 0);
  return static_cast<duckdb::idx_t>(rows);
}

std::string explain_text(duckdb::Connection& con, std::string const& sql)
{
  auto result = require_query_ok(con, "EXPLAIN " + sql);
  std::string out;
  for (duckdb::idx_t r = 0; r < result->RowCount(); ++r) {
    for (duckdb::idx_t c = 0; c < result->ColumnCount(); ++c) {
      out += result->GetValue(c, r).ToString();
      out.push_back('\n');
    }
  }
  return out;
}

bool plan_mentions_cardinality(std::string plan_text, duckdb::idx_t row_count)
{
  plan_text.erase(std::remove(plan_text.begin(), plan_text.end(), ','), plan_text.end());
  auto const rows = std::to_string(row_count);
  return plan_text.find("~" + rows + " rows") != std::string::npos ||
         plan_text.find("EC: " + rows) != std::string::npos ||
         plan_text.find("Estimated Cardinality: " + rows) != std::string::npos;
}

duckdb::unique_ptr<duckdb::FunctionData> bind_sirius_read_parquet(
  duckdb::ClientContext& ctx,
  std::string const& uri,
  duckdb::TableFunction& table_function,
  duckdb::vector<duckdb::LogicalType>& types,
  duckdb::vector<std::string>& names)
{
  duckdb::unique_ptr<duckdb::FunctionData> bind_data;
  ctx.RunFunctionInTransaction([&] {
    auto& entry = duckdb::Catalog::GetEntry<duckdb::TableFunctionCatalogEntry>(
      ctx, INVALID_CATALOG, DEFAULT_SCHEMA, "sirius_read_parquet");

    duckdb::vector<duckdb::LogicalType> arg_types;
    arg_types.emplace_back(duckdb::LogicalType::VARCHAR);
    table_function = entry.functions.GetFunctionByArguments(ctx, arg_types);

    duckdb::vector<duckdb::Value> inputs;
    inputs.emplace_back(uri);

    duckdb::named_parameter_map_t named_parameters;
    duckdb::vector<duckdb::LogicalType> input_table_types;
    duckdb::vector<std::string> input_table_names;

    duckdb::TableFunctionRef ref;
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
    children.push_back(duckdb::make_uniq<duckdb::ConstantExpression>(duckdb::Value(uri)));
    ref.function = duckdb::make_uniq<duckdb::FunctionExpression>(
      "sirius_read_parquet", std::move(children), nullptr, nullptr, false, false, false);

    duckdb::TableFunctionBindInput bind_input(inputs,
                                              named_parameters,
                                              input_table_types,
                                              input_table_names,
                                              nullptr,
                                              nullptr,
                                              table_function,
                                              ref);
    bind_data = table_function.bind(ctx, bind_input, types, names);
  });
  return bind_data;
}

std::string tpch_q3_shape_query(std::string const& customer_scan,
                                std::string const& orders_scan,
                                std::string const& lineitem_scan)
{
  return "SELECT l_orderkey, "
         "sum(l_extendedprice * (1 - l_discount)) AS revenue, "
         "o_orderdate, "
         "o_shippriority "
         "FROM " +
         customer_scan + " c, " + orders_scan + " o, " + lineitem_scan +
         " l "
         "WHERE c_mktsegment = 'BUILDING' "
         "AND c_custkey = o_custkey "
         "AND l_orderkey = o_orderkey "
         "AND o_orderdate < DATE '1995-03-15' "
         "AND l_shipdate > DATE '1995-03-15' "
         "GROUP BY l_orderkey, o_orderdate, o_shippriority "
         "ORDER BY revenue DESC, o_orderdate, l_orderkey "
         "LIMIT 10";
}

std::string tpch_q1_shape_query(std::string const& lineitem_scan)
{
  return "SELECT l_returnflag, "
         "l_linestatus, "
         "sum(l_quantity) AS sum_qty, "
         "sum(l_extendedprice) AS sum_base_price, "
         "sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, "
         "sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, "
         "avg(l_quantity) AS avg_qty, "
         "avg(l_extendedprice) AS avg_price, "
         "avg(l_discount) AS avg_disc, "
         "count(*) AS count_order "
         "FROM " +
         lineitem_scan +
         " WHERE l_shipdate BETWEEN DATE '1996-01-01' AND DATE '1996-06-30' "
         "GROUP BY l_returnflag, l_linestatus "
         "ORDER BY l_returnflag, l_linestatus";
}

std::string large_lineitem_orders_join_query(std::string const& lineitem_scan,
                                             std::string const& orders_scan)
{
  return "SELECT count(*) FROM " + lineitem_scan + " l JOIN " + orders_scan +
         " o ON l.l_orderkey = o.o_orderkey "
         "WHERE o.o_orderdate < DATE '1995-03-15'";
}

void compare_s3_gpu_to_local_cpu(s3_sql_fixture& fixture,
                                 std::string const& s3_query,
                                 std::string const& local_query,
                                 std::vector<duckdb::idx_t> const& tolerant_columns = {})
{
  auto s3_result = require_query_ok(fixture.con, gpu_execution_sql(s3_query));

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto local_result = require_query_ok(baseline_con, local_query);

  check_rows_equal_with_tolerant_columns(*s3_result, *local_result, tolerant_columns);
}

}  // namespace

TEST_CASE("internal sirius_read_parquet is registered as a one-argument table function",
          "[sql][s3][registration]")
{
  duckdb::DuckDB db(nullptr);
  load_sirius_extension(db);
  duckdb::Connection con(db);

  auto result = require_query_ok(con,
                                 "SELECT function_name, parameter_types "
                                 "FROM duckdb_functions() "
                                 "WHERE function_name = 'sirius_read_parquet' "
                                 "ORDER BY function_name");
  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).ToString() == "sirius_read_parquet");
  CHECK(result->GetValue(1, 0).ToString().find("VARCHAR") != std::string::npos);
}

TEST_CASE("S3 SQL config guard writes nested object_store options only when configured",
          "[s3][config]")
{
  s3_test_env env{"http://127.0.0.1:9000",
                  "",
                  "",
                  "us-east-1",
                  "temporary-access-key",
                  "temporary-secret-key",
                  "sirius-test",
                  "",
                  fs::temp_directory_path()};

  {
    sirius_config_env_guard guard(env);
    auto const yaml = read_text_file(guard.config_path());
    CHECK(yaml.find("object_store_config:") == std::string::npos);
    CHECK(yaml.find("executor:") != std::string::npos);
    CHECK(yaml.find("scan_manager:") != std::string::npos);
    CHECK(yaml.find("object_store:") != std::string::npos);
    CHECK(yaml.find("session_token:") == std::string::npos);
    CHECK(yaml.find("signing_mode:") == std::string::npos);
    CHECK(yaml.find("enable_chunk_prewarm") == std::string::npos);
  }

  env.session_token = "temporary-session-token";
  {
    sirius_config_env_guard guard(env, {}, std::string{"header"});
    auto const yaml = read_text_file(guard.config_path());
    CHECK(yaml.find("session_token: 'temporary-session-token'") != std::string::npos);
    CHECK(yaml.find("signing_mode: 'header'") != std::string::npos);
  }
}

TEST_CASE("gpu_execution rewrites S3 read_parquet and scans through Sirius",
          "[s3][integration][sql][gpu_execution]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const s3_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                        s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  auto const local_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                           local_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  compare_s3_gpu_to_local_cpu(fixture, s3_query, local_query);

  auto result = require_query_ok(fixture.con, gpu_execution_sql(s3_query));
  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 3);
  std::array<int, 5> region_counts{};
  for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
    auto const nation_key = result->GetValue(0, row).GetValue<int32_t>();
    auto const region_key = result->GetValue(2, row).GetValue<int32_t>();
    CHECK(nation_key == static_cast<int32_t>(row));
    REQUIRE(region_key >= 0);
    REQUIRE(region_key < static_cast<int32_t>(region_counts.size()));
    ++region_counts[static_cast<std::size_t>(region_key)];
  }
  CHECK(result->GetValue(1, 0).ToString() == "ALGERIA");
  for (auto const count : region_counts) {
    CHECK(count == 5);
  }
}

TEST_CASE("gpu_execution S3 SQL surface returns empty result sets cleanly",
          "[s3][integration][sql][gpu_execution]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const s3_query =
    "SELECT n_nationkey FROM " + s3_parquet_scan(*env, "nation") + " WHERE n_regionkey = 99";
  auto result = require_query_ok(fixture.con, gpu_execution_sql(s3_query));
  CHECK(result->RowCount() == 0);
}

TEST_CASE("gpu_execution S3 SQL surface counts every uploaded TPCH parquet table",
          "[s3][integration][sql][gpu_execution]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  std::vector<std::pair<std::string, duckdb::idx_t>> tables = {
    {"nation", 25}, {"region", 5}, {"customer", 15000}, {"orders", 150000}, {"lineitem", 600572}};

  for (auto const& [table, expected_rows] : tables) {
    auto const sql = "SELECT count(*) FROM " + s3_parquet_scan(*env, table);
    auto result    = require_query_ok(fixture.con, gpu_execution_sql(sql));
    REQUIRE(result->RowCount() == 1);
    CHECK(result->GetValue(0, 0).GetValue<int64_t>() == static_cast<int64_t>(expected_rows));
  }
}

TEST_CASE("gpu_execution S3 SQL surface scans all orders row groups",
          "[s3][integration][sql][gpu_execution]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const aggregate_sql =
    "SELECT count(*), min(o_orderdate), max(o_orderdate) FROM " + s3_parquet_scan(*env, "orders");
  auto const local_sql = "SELECT count(*), min(o_orderdate), max(o_orderdate) FROM " +
                         local_parquet_scan(*env, "orders");
  compare_s3_gpu_to_local_cpu(fixture, aggregate_sql, local_sql);
}

TEST_CASE("gpu_execution S3 SQL surface matches local TPC-H Q1 shape",
          "[s3][integration][sql][gpu_execution]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  compare_s3_gpu_to_local_cpu(fixture,
                              tpch_q1_shape_query(s3_parquet_scan(*env, "lineitem")),
                              tpch_q1_shape_query(local_parquet_scan(*env, "lineitem")),
                              {6, 7, 8});
}

TEST_CASE("gpu_execution S3 SQL surface matches local TPC-H Q3 shape",
          "[s3][integration][sql][gpu_execution]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  compare_s3_gpu_to_local_cpu(fixture,
                              tpch_q3_shape_query(s3_parquet_scan(*env, "customer"),
                                                  s3_parquet_scan(*env, "orders"),
                                                  s3_parquet_scan(*env, "lineitem")),
                              tpch_q3_shape_query(local_parquet_scan(*env, "customer"),
                                                  local_parquet_scan(*env, "orders"),
                                                  local_parquet_scan(*env, "lineitem")),
                              {1});
}

TEST_CASE("gpu_execution S3 window query reports unsupported S3 CPU fallback",
          "[s3][integration][sql][gpu_execution][fallback]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const s3_query = "SELECT n_nationkey, ROW_NUMBER() OVER (ORDER BY n_nationkey) AS rn FROM " +
                        s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  auto result = fixture.con.Query(gpu_execution_sql(s3_query));
  REQUIRE(result);
  REQUIRE(result->HasError());
  auto const error = result->GetError();
  INFO(error);
  CHECK(error.find("S3 CPU fallback is not supported") != std::string::npos);
  CHECK((error.find("window") != std::string::npos || error.find("Window") != std::string::npos ||
         error.find("WINDOW") != std::string::npos));
  CHECK(error.find("No filesystem") == std::string::npos);
  CHECK(error.find("no filesystem") == std::string::npos);
}

TEST_CASE("plain DuckDB read_parquet over s3 is not the Sirius SQL surface",
          "[s3][integration][sql][gpu_execution]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto result    = fixture.con.Query("SELECT count(*) FROM read_parquet('" + uri + "')");
  REQUIRE(result);
  REQUIRE(result->HasError());
  auto const error = result->GetError();
  INFO(error);
  CHECK((error.find("s3") != std::string::npos || error.find("S3") != std::string::npos));
}

TEST_CASE("internal sirius_read_parquet bind returns row-count metadata for cardinality",
          "[s3][integration][sql][planner-metadata]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const uri                  = s3_uri(env->bucket, "parquet/orders.parquet");
  auto const expected_orders_rows = local_parquet_row_count(*env, "orders");
  duckdb::TableFunction table_function;
  duckdb::vector<duckdb::LogicalType> return_types;
  duckdb::vector<std::string> names;

  auto bind_data =
    bind_sirius_read_parquet(*fixture.con.context, uri, table_function, return_types, names);

  REQUIRE(bind_data != nullptr);
  auto const* typed = dynamic_cast<duckdb::SiriusReadParquetBindData const*>(bind_data.get());
  REQUIRE(typed != nullptr);
  CHECK(typed->uri == uri);
  CHECK(typed->total_num_rows == expected_orders_rows);
  CHECK_FALSE(return_types.empty());
  CHECK_FALSE(names.empty());
  REQUIRE(table_function.cardinality != nullptr);

  auto stats = table_function.cardinality(*fixture.con.context, bind_data.get());
  REQUIRE(stats != nullptr);
  CHECK(stats->has_estimated_cardinality);
  CHECK(stats->estimated_cardinality == expected_orders_rows);
  CHECK(stats->has_max_cardinality);
  CHECK(stats->max_cardinality == expected_orders_rows);
}

TEST_CASE("internal sirius_read_parquet exposes S3 row count to DuckDB EXPLAIN",
          "[s3][integration][sql][planner-metadata]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const expected_orders_rows = local_parquet_row_count(*env, "orders");
  auto const plan =
    explain_text(fixture.con, "SELECT * FROM " + s3_sirius_parquet_scan(*env, "orders"));
  INFO(plan);
  CHECK(plan_mentions_cardinality(plan, expected_orders_rows));
}

TEST_CASE("internal sirius_read_parquet exposes distinct S3 table cardinalities in joins",
          "[s3][integration][sql][planner-metadata]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const expected_orders_rows = local_parquet_row_count(*env, "orders");
  auto const expected_nation_rows = local_parquet_row_count(*env, "nation");
  auto const sql = "SELECT count(*) FROM " + s3_sirius_parquet_scan(*env, "orders") + " o JOIN " +
                   s3_sirius_parquet_scan(*env, "nation") +
                   " n ON (o.o_custkey % 25) = n.n_nationkey";
  auto const plan = explain_text(fixture.con, sql);
  INFO(plan);
  CHECK(plan_mentions_cardinality(plan, expected_orders_rows));
  CHECK(plan_mentions_cardinality(plan, expected_nation_rows));
}

TEST_CASE("gpu_execution S3 SQL telemetry reaches the REST ioctx",
          "[s3][integration][sql][gpu_execution]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  limits.rest_perf_instrumentation = true;
  s3_sql_fixture fixture(*env, limits);
  auto const uri         = s3_uri(env->bucket, "parquet/orders.parquet");
  auto const before_gets = rest_chunk_get_count(fixture, uri);

  auto const sql =
    "SELECT count(*), min(o_orderdate), max(o_orderdate) FROM " + s3_parquet_scan(*env, "orders");
  auto result = require_query_ok(fixture.con, gpu_execution_sql(sql));

  CHECK(rest_chunk_get_count(fixture, uri) > before_gets);
  CHECK(rest_terminal_failures(fixture, uri) == 0);
  REQUIRE(result->RowCount() == 1);
}

TEST_CASE("gpu_execution S3 SQL supports both configured SigV4 signing modes",
          "[s3][integration][sql][gpu_execution][config]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto const s3_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                        s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  auto const local_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                           local_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";

  SECTION("presigned")
  {
    s3_sql_fixture fixture(*env, {}, std::string{"presigned"});
    compare_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  }

  SECTION("header")
  {
    s3_sql_fixture fixture(*env, {}, std::string{"header"});
    compare_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  }
}

TEST_CASE("gpu_execution S3 SQL works over TLS with the harness CA bundle",
          "[s3][integration][sql][gpu_execution][config]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }
  if (env->https_endpoint.empty() || env->ca_bundle_path.empty()) {
    if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
      FAIL("SIRIUS_TEST_S3_HTTPS_ENDPOINT and SIRIUS_TEST_S3_CA_BUNDLE are required");
    }
    SUCCEED("HTTPS MinIO endpoint not configured; skipping TLS S3 SQL test");
    return;
  }

  s3_sql_fixture fixture(*env, {}, std::nullopt, env->https_endpoint, env->ca_bundle_path, true);
  compare_s3_gpu_to_local_cpu(
    fixture,
    "SELECT n_nationkey, n_name FROM " + s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey",
    "SELECT n_nationkey, n_name FROM " + local_parquet_scan(*env, "nation") +
      " ORDER BY n_nationkey");
}

TEST_CASE("gpu_execution S3 SQL preserves correctness with prefetch cache disabled",
          "[s3][integration][sql][gpu_execution][config]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  limits.enable_prefetch_cache = false;
  s3_sql_fixture fixture(*env, limits);
  compare_s3_gpu_to_local_cpu(
    fixture,
    "SELECT count(*), sum(o_totalprice) FROM " + s3_parquet_scan(*env, "orders"),
    "SELECT count(*), sum(o_totalprice) FROM " + local_parquet_scan(*env, "orders"),
    {1});
}

TEST_CASE("gpu_execution S3 SQL survives a constrained GPU memory config with disk tier",
          "[s3][integration][sql][gpu_execution][tiering]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  limits.gpu_usage       = "256 MiB";
  limits.gpu_reservation = "128 MiB";
  limits.host_capacity   = "512 MiB";
  limits.disk_capacity   = "2 GiB";
  s3_sql_fixture fixture(*env, limits);
  compare_s3_gpu_to_local_cpu(
    fixture,
    "SELECT sum(l_extendedprice * (1 - l_discount)) FROM " + s3_parquet_scan(*env, "lineitem") +
      " WHERE l_shipdate BETWEEN DATE '1996-01-01' AND DATE '1996-06-30'",
    "SELECT sum(l_extendedprice * (1 - l_discount)) FROM " + local_parquet_scan(*env, "lineitem") +
      " WHERE l_shipdate BETWEEN DATE '1996-01-01' AND DATE '1996-06-30'",
    {0});
}

TEST_CASE("gpu_execution large S3 lineitem count matches the local parquet oracle",
          "[.][s3][sql][large][large-count][gpu_execution][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits(/*enable_prefetch_cache=*/true));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto const expected_rows  = local_parquet_file_row_count(large->local_path);
  auto const before_gets    = rest_chunk_get_count(fixture, large->uri);
  auto const s3_count_query = "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env);
  auto s3_result            = require_query_ok(fixture.con, gpu_execution_sql(s3_count_query));
  auto const get_delta      = rest_chunk_get_count(fixture, large->uri) - before_gets;

  REQUIRE(s3_result->RowCount() == 1);
  CHECK(s3_result->GetValue(0, 0).GetValue<int64_t>() == static_cast<int64_t>(expected_rows));
  CHECK(large->total_num_rows == expected_rows);
  CHECK(expected_rows > 50'000'000);
  CHECK(get_delta > 0);
}

TEST_CASE("gpu_execution large S3 lineitem TPC-H Q1 shape matches local CPU",
          "[.][s3][sql][large][large-q1][gpu_execution][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits(/*enable_prefetch_cache=*/true));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto const before_gets = rest_chunk_get_count(fixture, large->uri);
  compare_s3_gpu_to_local_cpu(fixture,
                              tpch_q1_shape_query(s3_large_lineitem_scan(*env)),
                              tpch_q1_shape_query(local_parquet_file_scan(large->local_path)),
                              {6, 7, 8});
  CHECK(rest_chunk_get_count(fixture, large->uri) > before_gets);
}

TEST_CASE("gpu_execution large S3 lineitem join uses planner cardinality and matches local CPU",
          "[.][s3][sql][large][large-join][gpu_execution][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits(/*enable_prefetch_cache=*/true));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto const expected_lineitem_rows = local_parquet_file_row_count(large->local_path);
  auto const expected_orders_rows   = local_parquet_row_count(*env, "orders");
  compare_s3_gpu_to_local_cpu(
    fixture,
    large_lineitem_orders_join_query(s3_large_lineitem_scan(*env), s3_parquet_scan(*env, "orders")),
    large_lineitem_orders_join_query(local_parquet_file_scan(large->local_path),
                                     local_parquet_scan(*env, "orders")));

  auto const explain_sql = large_lineitem_orders_join_query(s3_sirius_large_lineitem_scan(*env),
                                                            s3_sirius_parquet_scan(*env, "orders"));
  auto const plan        = explain_text(fixture.con, explain_sql);
  INFO(plan);
  CHECK(plan_mentions_cardinality(plan, expected_lineitem_rows));
  CHECK(plan_mentions_cardinality(plan, expected_orders_rows));
}

TEST_CASE("gpu_execution large S3 lineitem count matches without prefetch cache",
          "[.][s3][sql][large][large-count-no-prewarm][gpu_execution][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits(/*enable_prefetch_cache=*/false));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto const expected_rows  = local_parquet_file_row_count(large->local_path);
  auto const before_gets    = rest_chunk_get_count(fixture, large->uri);
  auto const s3_count_query = "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env);
  auto s3_result            = require_query_ok(fixture.con, gpu_execution_sql(s3_count_query));
  auto const get_delta      = rest_chunk_get_count(fixture, large->uri) - before_gets;

  REQUIRE(s3_result->RowCount() == 1);
  CHECK(s3_result->GetValue(0, 0).GetValue<int64_t>() == static_cast<int64_t>(expected_rows));
  CHECK(get_delta > 0);
}

TEST_CASE("gpu_execution large S3 lineitem Q1 shape matches local CPU without prefetch cache",
          "[.][s3][sql][large][large-q1-no-prewarm][gpu_execution][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits(/*enable_prefetch_cache=*/false));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  compare_s3_gpu_to_local_cpu(fixture,
                              tpch_q1_shape_query(s3_large_lineitem_scan(*env)),
                              tpch_q1_shape_query(local_parquet_file_scan(large->local_path)),
                              {6, 7, 8});
}

TEST_CASE("gpu_execution large S3 lineitem join matches local CPU without prefetch cache",
          "[.][s3][sql][large][large-join-no-prewarm][gpu_execution][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits(/*enable_prefetch_cache=*/false));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  compare_s3_gpu_to_local_cpu(
    fixture,
    large_lineitem_orders_join_query(s3_large_lineitem_scan(*env), s3_parquet_scan(*env, "orders")),
    large_lineitem_orders_join_query(local_parquet_file_scan(large->local_path),
                                     local_parquet_scan(*env, "orders")));
}
