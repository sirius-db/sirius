/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"

#include <duckdb.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
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
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
};

std::optional<s3_test_env> read_s3_test_env()
{
  auto endpoint   = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket     = env_or("SIRIUS_TEST_S3_BUCKET");

  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty()) {
    return std::nullopt;
  }

  return s3_test_env{std::move(endpoint),
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket)};
}

bool skip_if_no_s3_env(std::optional<s3_test_env> const& env)
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

class sirius_config_env_guard {
 public:
  explicit sirius_config_env_guard(s3_test_env const& env)
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
    dir_              = fs::temp_directory_path() / ("sirius_pr6_s3_sql_" + unique);
    config_path_      = dir_ / "sirius.yaml";
    fs::create_directories(dir_);

    std::ofstream out(config_path_);
    out << "sirius:\n"
           "  memory:\n"
           "    gpu:\n"
           "      usage_limit_bytes: 256 MiB\n"
           "      reservation_limit_bytes: 128 MiB\n"
           "    host:\n"
           "      capacity_bytes: 512 MiB\n"
           "  object_store_config:\n"
           "    endpoint: "
        << yaml_quote(env.endpoint)
        << "\n"
           "    region: "
        << yaml_quote(env.region)
        << "\n"
           "    access_key: "
        << yaml_quote(env.access_key)
        << "\n"
           "    secret_key: "
        << yaml_quote(env.secret_key) << "\n";
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
  explicit s3_sql_fixture(s3_test_env const& env) : config_env(env), db(nullptr), con(db)
  {
    load_sirius_extension(db);
    REQUIRE(con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state"));

    // Match the shared integration fixture pattern: after the SiriusContext is
    // created for this DuckDB instance, keep unrelated DuckDB instances from
    // allocating a second context.
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

fs::path local_parquet_path(std::string_view table)
{
  return fs::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" / "data" / "parquet" /
         (std::string{table} + ".parquet");
}

std::string local_parquet_scan(std::string_view table)
{
  return "read_parquet(" + sql_quote(local_parquet_path(table).string()) + ")";
}

std::string s3_parquet_scan(s3_test_env const& env, std::string_view table)
{
  auto const key = "parquet/" + std::string{table} + ".parquet";
  return "read_parquet(" + sql_quote(s3_uri(env.bucket, key)) + ")";
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

}  // namespace

TEST_CASE("sirius_read_parquet is registered as a one-argument table function",
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

TEST_CASE("sirius_read_parquet executes directly via transparent execution",
          "[s3][sql][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");

  auto const sql =
    "SELECT n_nationkey, n_name, n_regionkey "
    "FROM sirius_read_parquet('" +
    uri + "') ORDER BY n_nationkey";
  auto result = require_query_ok(fixture.con, sql);

  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 3);
  CHECK(result->GetValue(0, 0).GetValue<int32_t>() == 0);
  CHECK(result->GetValue(1, 0).ToString() == "ALGERIA");
  CHECK(result->GetValue(2, 0).GetValue<int32_t>() == 0);
}

TEST_CASE("gpu_execution rewrites S3 read_parquet and scans through Sirius",
          "[s3][sql][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto const sql = gpu_execution_sql(
    "SELECT n_nationkey, n_name, n_regionkey "
    "FROM read_parquet('" +
    uri + "') ORDER BY n_nationkey");
  auto result = require_query_ok(fixture.con, sql);

  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 3);
  CHECK(result->GetValue(0, 0).GetValue<int32_t>() == 0);
  CHECK(result->GetValue(1, 0).ToString() == "ALGERIA");
  CHECK(result->GetValue(2, 0).GetValue<int32_t>() == 0);
}

TEST_CASE("gpu_execution can call sirius_read_parquet directly",
          "[s3][sql][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto const sql = gpu_execution_sql(
    "SELECT n_nationkey, n_name, n_regionkey "
    "FROM sirius_read_parquet('" +
    uri + "') ORDER BY n_nationkey");
  auto result = require_query_ok(fixture.con, sql);

  REQUIRE(result->RowCount() == 25);
  REQUIRE(result->ColumnCount() == 3);
  CHECK(result->GetValue(0, 0).GetValue<int32_t>() == 0);
  CHECK(result->GetValue(1, 0).ToString() == "ALGERIA");
  CHECK(result->GetValue(2, 0).GetValue<int32_t>() == 0);
}

TEST_CASE("gpu_execution S3 SQL surface returns empty result sets cleanly",
          "[s3][sql][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto const sql = gpu_execution_sql(
    "SELECT n_nationkey "
    "FROM read_parquet('" +
    uri + "') WHERE n_regionkey = 99");
  auto result = require_query_ok(fixture.con, sql);

  CHECK(result->RowCount() == 0);
}

TEST_CASE("gpu_execution S3 SQL surface matches local TPC-H Q3 shape",
          "[s3][sql][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const s3_query = tpch_q3_shape_query(s3_parquet_scan(*env, "customer"),
                                            s3_parquet_scan(*env, "orders"),
                                            s3_parquet_scan(*env, "lineitem"));
  auto s3_result      = require_query_ok(fixture.con, gpu_execution_sql(s3_query));

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto const local_query = tpch_q3_shape_query(
    local_parquet_scan("customer"), local_parquet_scan("orders"), local_parquet_scan("lineitem"));
  auto baseline_result = require_query_ok(baseline_con, local_query);

  CHECK(s3_result->RowCount() <= 10);
  REQUIRE(s3_result->RowCount() == baseline_result->RowCount());
  REQUIRE(s3_result->ColumnCount() == baseline_result->ColumnCount());
  CHECK(collect_rows(*s3_result) == collect_rows(*baseline_result));
}

TEST_CASE("gpu_execution S3 SQL surface scans all orders row groups",
          "[s3][sql][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  auto const aggregate_sql =
    "SELECT count(*), min(o_orderdate), max(o_orderdate) FROM " + s3_parquet_scan(*env, "orders");

  s3_sql_fixture fixture(*env);
  auto s3_result = require_query_ok(fixture.con, gpu_execution_sql(aggregate_sql));

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto const local_sql =
    "SELECT count(*), min(o_orderdate), max(o_orderdate) FROM " + local_parquet_scan("orders");
  auto baseline_result = require_query_ok(baseline_con, local_sql);

  REQUIRE(s3_result->RowCount() == 1);
  REQUIRE(s3_result->ColumnCount() == 3);
  REQUIRE(baseline_result->RowCount() == 1);
  REQUIRE(baseline_result->ColumnCount() == 3);
  CHECK(collect_rows(*s3_result) == collect_rows(*baseline_result));
}
