/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"

#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>
#include <io/sirius_datasource.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
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
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::string session_token;
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
                     std::move(bucket),
                     env_or("SIRIUS_TEST_S3_SESSION_TOKEN")};
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

std::optional<s3_test_env> require_aws_live_env()
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return std::nullopt; }
  if (env->session_token.empty()) {
    FAIL(
      "SIRIUS_TEST_S3_SESSION_TOKEN is required for live AWS tests; use assume-role "
      "temporary credentials");
  }
  return env;
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::string sf10_lineitem_key()
{
  return env_or("SIRIUS_PR6_LARGE_S3_KEY",
                env_or("SIRIUS_BENCH_S3_KEY", "tpch/lineitem_sf10.parquet"));
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
  // A small disk tier so the downgrade executor can spill instead of looping
  // when GPU pressure is hit. Without it, on large-VRAM GPUs (where RMM's pool
  // can exceed the 256 MiB usage cap at init) even a tiny scan spins forever at
  // the no-disk-spill boundary. `large_sirius_memory_limits()` overrides this.
  std::string disk_capacity{"2 GiB"};
  std::optional<bool> enable_chunk_prewarm;
};

sirius_memory_limits large_sirius_memory_limits()
{
  sirius_memory_limits limits;
  limits.gpu_usage       = "5 GiB";
  limits.gpu_reservation = "2 GiB";
  limits.host_capacity   = "8 GiB";
  // SF10 scans can trigger downgrade pressure on the CI RTX 3060; provide a
  // disk tier so the large correctness tests exercise Sirius tiering instead
  // of failing at the no-disk spill boundary.
  limits.disk_capacity = "32 GiB";
  return limits;
}

sirius_memory_limits large_sirius_memory_limits_without_chunk_prewarm()
{
  auto limits                 = large_sirius_memory_limits();
  limits.enable_chunk_prewarm = false;
  return limits;
}

sirius_memory_limits large_sirius_memory_limits_with_chunk_prewarm(bool enabled)
{
  auto limits                 = large_sirius_memory_limits();
  limits.enable_chunk_prewarm = enabled;
  return limits;
}

class sirius_config_env_guard {
 public:
  explicit sirius_config_env_guard(s3_test_env const& env,
                                   sirius_memory_limits limits             = {},
                                   std::optional<std::string> signing_mode = std::nullopt,
                                   std::optional<bool> use_async_backend   = std::nullopt)
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
    out << "sirius:\n";
    if (limits.disk_capacity.empty()) {
      out << "  memory:\n"
             "    gpu:\n"
             "      usage_limit_bytes: "
          << limits.gpu_usage
          << "\n"
             "      reservation_limit_bytes: "
          << limits.gpu_reservation
          << "\n"
             "    host:\n"
             "      capacity_bytes: "
          << limits.host_capacity << "\n";
    } else {
      out << "  space:\n"
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
          << limits.disk_capacity << "\n";
    }
    out << "  object_store_config:\n"
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
    if (!env.session_token.empty()) {
      out << "    session_token: " << yaml_quote(env.session_token) << "\n";
    }
    if (signing_mode.has_value()) {
      out << "    signing_mode: " << yaml_quote(*signing_mode) << "\n";
    }
    if (use_async_backend.has_value()) {
      out << "    s3_use_async_backend: " << (*use_async_backend ? "true" : "false") << "\n";
    }
    if (limits.enable_chunk_prewarm.has_value()) {
      out << "  executor:\n"
             "    scan_manager:\n"
             "      enable_prefetch_cache: true\n"
             "      enable_chunk_prewarm: "
          << (*limits.enable_chunk_prewarm ? "true" : "false") << "\n";
    }
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
                          std::optional<bool> use_async_backend   = std::nullopt)
    : config_env(env, std::move(limits), std::move(signing_mode), use_async_backend),
      db(nullptr),
      con(db)
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

void check_rows_equal_with_tolerant_columns(duckdb::MaterializedQueryResult& actual,
                                            duckdb::MaterializedQueryResult& expected,
                                            std::vector<duckdb::idx_t> const& tolerant_columns)
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
              Approx(std::stod(expected_value)).epsilon(1e-10).margin(1e-9));
      } else {
        CHECK(actual_value == expected_value);
      }
    }
  }
}

bool within_large_s3_byte_budget(std::uint64_t byte_delta, std::size_t object_size)
{
  // B1 Phase 3a (newplan §26.6) measured every config (cache-off /
  // cache-on+prewarm-on / cache-on+prewarm-off) at 1.00x object_size on SF10;
  // §26's earlier 1.87x double-fetch did not reproduce on current code.
  // 1.1x is a tight regression guard; measured worst case is the join at
  // ~1.003x (lineitem + orders both read against the lineitem object_size).
  return byte_delta <=
         static_cast<std::uint64_t>(object_size) + static_cast<std::uint64_t>(object_size / 10);
}

bool within_no_prewarm_s3_byte_budget(std::uint64_t byte_delta, std::size_t object_size)
{
  // Same 1.1x ceiling; cache-on + prewarm-off also measured 1.00x (§26.6).
  return byte_delta <=
         static_cast<std::uint64_t>(object_size) + static_cast<std::uint64_t>(object_size / 10);
}

fs::path local_parquet_path(std::string_view table)
{
  return fs::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" / "data" / "parquet" /
         (std::string{table} + ".parquet");
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

std::string local_parquet_scan(std::string_view table)
{
  return "read_parquet(" + sql_quote(local_parquet_path(table).string()) + ")";
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

sirius::io::s3::s3_ioctx& require_async_s3_ioctx(s3_sql_fixture& fixture, std::string const& uri)
{
  auto& sirius_ctx = require_sirius_context(fixture);
  auto datasource  = sirius_ctx.get_scan_manager().create_datasource(uri);
  REQUIRE(datasource != nullptr);
  auto* s3_ctx = dynamic_cast<sirius::io::s3::s3_ioctx*>(datasource->io_ctx().get());
  REQUIRE(s3_ctx != nullptr);
  return *s3_ctx;
}

struct large_lineitem_fixture {
  std::string uri;
  fs::path local_path;
  std::size_t object_size{0};
  duckdb::idx_t total_num_rows{0};
};

std::optional<large_lineitem_fixture> read_large_lineitem_fixture(s3_sql_fixture& fixture,
                                                                  s3_test_env const& env)
{
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
    out.object_size    = bind_info.object_size;
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

duckdb::idx_t local_parquet_row_count(std::string_view table)
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto result = require_query_ok(con, "SELECT count(*) FROM " + local_parquet_scan(table));
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
  // Keep the TPC-H Q1 operator shape, but use a bounded date window so SF10
  // row-group pruning keeps the CI GPU under its 5 GiB Sirius memory cap.
  // The full-date Q1 shape currently needs more GPU intermediates than this
  // host can reserve; large-scale tier engagement is tracked separately.
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

struct b1_cache_counters {
  std::uint64_t hit_count{0};
  std::uint64_t hit_after_wait{0};
  std::uint64_t partial_miss{0};
  std::uint64_t full_miss{0};
  std::uint64_t range_miss{0};
};

b1_cache_counters snapshot_cache(sirius::io::prefetching_cache const& cache)
{
  return b1_cache_counters{cache.hit_count_total(),
                           cache.hit_after_wait_total(),
                           cache.partial_miss_count_total(),
                           cache.full_miss_count_total(),
                           cache.range_miss_count_total()};
}

b1_cache_counters operator-(b1_cache_counters const& after, b1_cache_counters const& before)
{
  return b1_cache_counters{after.hit_count - before.hit_count,
                           after.hit_after_wait - before.hit_after_wait,
                           after.partial_miss - before.partial_miss,
                           after.full_miss - before.full_miss,
                           after.range_miss - before.range_miss};
}

struct b1_metrics {
  double wall_clock_ms{0};
  std::uint64_t bytes_read{0};
  std::uint64_t fsmr_borrows{0};
  b1_cache_counters cache;
};

struct b1_run_record {
  std::string query;
  std::string config;
  int iteration{0};
  b1_metrics metrics;
  bool has_cache_metrics{false};
};

struct b1_summary {
  double median{0};
  double min{0};
  double max{0};
};

using b1_metric_getter = double (*)(b1_metrics const&);

double median_sorted(std::vector<double> values)
{
  REQUIRE_FALSE(values.empty());
  std::sort(values.begin(), values.end());
  auto const mid = values.size() / 2;
  if (values.size() % 2 == 1) { return values[mid]; }
  return (values[mid - 1] + values[mid]) / 2.0;
}

b1_summary summarize_metric(std::vector<b1_run_record> const& records,
                            std::string_view query,
                            std::string_view config,
                            b1_metric_getter getter)
{
  std::vector<double> values;
  for (auto const& record : records) {
    if (std::string_view{record.query} == query && std::string_view{record.config} == config) {
      values.push_back(getter(record.metrics));
    }
  }
  REQUIRE_FALSE(values.empty());
  return b1_summary{median_sorted(values),
                    *std::min_element(values.begin(), values.end()),
                    *std::max_element(values.begin(), values.end())};
}

std::optional<b1_summary> summarize_metric_if_available(std::vector<b1_run_record> const& records,
                                                        std::string_view query,
                                                        std::string_view config,
                                                        b1_metric_getter getter,
                                                        bool requires_cache)
{
  std::vector<double> values;
  for (auto const& record : records) {
    if (std::string_view{record.query} != query || std::string_view{record.config} != config) {
      continue;
    }
    if (requires_cache && !record.has_cache_metrics) { continue; }
    values.push_back(getter(record.metrics));
  }
  if (values.empty()) { return std::nullopt; }
  return b1_summary{median_sorted(values),
                    *std::min_element(values.begin(), values.end()),
                    *std::max_element(values.begin(), values.end())};
}

std::string format_double(double value, int precision = 2)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

std::string format_summary(b1_summary const& summary, bool integral = false)
{
  if (integral) {
    return std::to_string(static_cast<std::uint64_t>(std::llround(summary.median))) + " [" +
           std::to_string(static_cast<std::uint64_t>(std::llround(summary.min))) + ", " +
           std::to_string(static_cast<std::uint64_t>(std::llround(summary.max))) + "]";
  }
  return format_double(summary.median) + " [" + format_double(summary.min) + ", " +
         format_double(summary.max) + "]";
}

std::string format_optional_summary(std::optional<b1_summary> summary, bool integral = false)
{
  if (!summary) { return "n/a"; }
  return format_summary(*summary, integral);
}

std::string format_ratio(double numerator, double denominator)
{
  if (denominator == 0.0) { return "n/a"; }
  return format_double(numerator / denominator, 3);
}

std::string format_optional_ratio(std::optional<b1_summary> numerator,
                                  std::optional<b1_summary> denominator)
{
  if (!numerator || !denominator) { return "n/a"; }
  return format_ratio(numerator->median, denominator->median);
}

double metric_wall_ms(b1_metrics const& metrics) { return metrics.wall_clock_ms; }
double metric_bytes_read(b1_metrics const& metrics)
{
  return static_cast<double>(metrics.bytes_read);
}
double metric_fsmr_borrows(b1_metrics const& metrics)
{
  return static_cast<double>(metrics.fsmr_borrows);
}
double metric_hit_count(b1_metrics const& metrics)
{
  return static_cast<double>(metrics.cache.hit_count);
}
double metric_hit_after_wait(b1_metrics const& metrics)
{
  return static_cast<double>(metrics.cache.hit_after_wait);
}
double metric_partial_miss(b1_metrics const& metrics)
{
  return static_cast<double>(metrics.cache.partial_miss);
}
double metric_full_miss(b1_metrics const& metrics)
{
  return static_cast<double>(metrics.cache.full_miss);
}
double metric_range_miss(b1_metrics const& metrics)
{
  return static_cast<double>(metrics.cache.range_miss);
}

struct b1_metric_row {
  std::string_view name;
  b1_metric_getter getter;
  bool integral{true};
  bool scale_by_object_size{false};
  bool requires_cache{false};
};

constexpr std::array<b1_metric_row, 9> b1_metric_rows{{
  {"wall_clock_ms", metric_wall_ms, false, false, false},
  {"bytes_read_total", metric_bytes_read, true, false, false},
  {"bytes_read / object_size", metric_bytes_read, false, true, false},
  {"fsmr_borrows_total", metric_fsmr_borrows, true, false, false},
  {"hit_count_total", metric_hit_count, true, false, true},
  {"hit_after_wait_total", metric_hit_after_wait, true, false, true},
  {"partial_miss_count_total", metric_partial_miss, true, false, true},
  {"full_miss_count_total", metric_full_miss, true, false, true},
  {"range_miss_count_total", metric_range_miss, true, false, true},
}};

constexpr std::string_view kB1CacheOffConfig      = "cache_off";
constexpr std::string_view kB1CacheOnPrewarmOn    = "cache_on_prewarm_on";
constexpr std::string_view kB1CacheOnPrewarmOff   = "cache_on_prewarm_off";
constexpr std::string_view kB1CountStarQuery      = "count_star";
constexpr std::string_view kB1CountProjectedQuery = "count_l_orderkey";
constexpr std::string_view kB1Q1Query             = "q1";
constexpr std::string_view kB1JoinQuery           = "join";

constexpr std::array<std::string_view, 3> b1_config_names{
  kB1CacheOffConfig, kB1CacheOnPrewarmOn, kB1CacheOnPrewarmOff};
constexpr std::array<std::string_view, 4> b1_query_names{
  kB1CountStarQuery, kB1CountProjectedQuery, kB1Q1Query, kB1JoinQuery};

std::string b1_config_label(std::string_view config)
{
  if (config == kB1CacheOffConfig) { return "cache OFF (production default)"; }
  if (config == kB1CacheOnPrewarmOn) { return "cache ON + prewarm ON"; }
  if (config == kB1CacheOnPrewarmOff) { return "cache ON + prewarm OFF"; }
  return std::string{config};
}

std::string b1_query_label(std::string_view query)
{
  if (query == kB1CountStarQuery) { return "count(*)"; }
  if (query == kB1CountProjectedQuery) { return "count(l_orderkey)"; }
  if (query == kB1Q1Query) { return "q1 — TPC-H Q1 shape (narrowed-date filter)"; }
  if (query == kB1JoinQuery) { return "join — lineitem×orders"; }
  return std::string{query};
}

b1_cache_counters sum_cache_counters(std::vector<b1_run_record> const& records,
                                     std::string_view query,
                                     std::string_view config)
{
  b1_cache_counters out;
  for (auto const& record : records) {
    if (std::string_view{record.query} != query || std::string_view{record.config} != config ||
        !record.has_cache_metrics) {
      continue;
    }
    out.hit_count += record.metrics.cache.hit_count;
    out.hit_after_wait += record.metrics.cache.hit_after_wait;
    out.partial_miss += record.metrics.cache.partial_miss;
    out.full_miss += record.metrics.cache.full_miss;
    out.range_miss += record.metrics.cache.range_miss;
  }
  return out;
}

std::string cache_hit_rate(b1_cache_counters const& counters)
{
  auto const misses = counters.partial_miss + counters.full_miss + counters.range_miss;
  auto const total  = counters.hit_count + misses;
  if (total == 0) { return "n/a"; }
  return format_double(100.0 * static_cast<double>(counters.hit_count) /
                       static_cast<double>(total));
}

std::string dominant_miss_counter(b1_cache_counters const& counters)
{
  auto dominant = std::pair<std::string_view, std::uint64_t>{"partial_miss", counters.partial_miss};
  if (counters.full_miss > dominant.second) { dominant = {"full_miss", counters.full_miss}; }
  if (counters.range_miss > dominant.second) { dominant = {"range_miss", counters.range_miss}; }
  return std::string{dominant.first} + "=" + std::to_string(dominant.second);
}

bool fsmr_nonzero_for_all_records(std::vector<b1_run_record> const& records,
                                  std::string_view query,
                                  std::string_view config)
{
  bool saw_record = false;
  for (auto const& record : records) {
    if (std::string_view{record.query} != query || std::string_view{record.config} != config) {
      continue;
    }
    saw_record = true;
    if (record.metrics.fsmr_borrows == 0) { return false; }
  }
  return saw_record;
}

fs::path b1_bench_output_path()
{
  auto path = env_or("SIRIUS_BENCH_OUTPUT_PATH");
  if (!path.empty()) { return fs::path{path}; }
  return fs::path{SIRIUS_PROJECT_ROOT} / "pr6-b1-bench-results.md";
}

void write_b1_bench_markdown(fs::path const& path,
                             std::vector<b1_run_record> const& records,
                             std::size_t object_size)
{
  if (!path.parent_path().empty()) { fs::create_directories(path.parent_path()); }
  std::ofstream out(path);
  REQUIRE(out);

  auto const branch = env_or("SIRIUS_BENCH_BRANCH", "unknown");
  auto const sha    = env_or("SIRIUS_BENCH_SHA", "unknown");
  auto const host   = env_or("SIRIUS_BENCH_HOST", "unknown");
  auto const date   = env_or("SIRIUS_BENCH_DATE", "unknown");

  out << "# B1 Phase 3a — bench results\n\n"
      << "**Branch / SHA:** " << branch << " @ " << sha << "\n"
      << "**CI host:** " << host << "\n"
      << "**Date:** " << date << "\n"
      << "**SF10 lineitem object size:** " << object_size << " bytes (~"
      << format_double(static_cast<double>(object_size) / (1024.0 * 1024.0 * 1024.0), 2)
      << " GiB)\n\n"
      << "## Raw iteration data\n\n"
      << "| query | config | iteration | wall_clock_ms | bytes_read_total | "
         "fsmr_borrows_total | hit_count_total | hit_after_wait_total | "
         "partial_miss_count_total | full_miss_count_total | range_miss_count_total |\n"
      << "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";

  for (auto const& record : records) {
    auto const cache_hit = record.has_cache_metrics ? std::to_string(record.metrics.cache.hit_count)
                                                    : std::string{"n/a"};
    auto const cache_wait   = record.has_cache_metrics
                                ? std::to_string(record.metrics.cache.hit_after_wait)
                                : std::string{"n/a"};
    auto const partial_miss = record.has_cache_metrics
                                ? std::to_string(record.metrics.cache.partial_miss)
                                : std::string{"n/a"};
    auto const full_miss = record.has_cache_metrics ? std::to_string(record.metrics.cache.full_miss)
                                                    : std::string{"n/a"};
    auto const range_miss = record.has_cache_metrics
                              ? std::to_string(record.metrics.cache.range_miss)
                              : std::string{"n/a"};
    out << "|" << record.query << "|" << record.config << "|" << record.iteration << "|"
        << format_double(record.metrics.wall_clock_ms) << "|" << record.metrics.bytes_read << "|"
        << record.metrics.fsmr_borrows << "|" << cache_hit << "|" << cache_wait << "|"
        << partial_miss << "|" << full_miss << "|" << range_miss << "|\n";
  }

  out << "\n## Cache OFF (production default)\n\n"
      << "| query | wall_clock_ms median [min, max] | bytes_read / object_size | "
         "fsmr_borrows_total median [min, max] |\n"
      << "|---|---:|---:|---:|\n";
  for (auto const query : b1_query_names) {
    auto wall    = summarize_metric(records, query, kB1CacheOffConfig, metric_wall_ms);
    auto bytes   = summarize_metric(records, query, kB1CacheOffConfig, metric_bytes_read);
    auto borrows = summarize_metric(records, query, kB1CacheOffConfig, metric_fsmr_borrows);
    bytes        = b1_summary{bytes.median / static_cast<double>(object_size),
                       bytes.min / static_cast<double>(object_size),
                       bytes.max / static_cast<double>(object_size)};
    out << "|" << b1_query_label(query) << "|" << format_summary(wall) << "|"
        << format_summary(bytes) << "|" << format_summary(borrows, true) << "|\n";
  }

  out << "\n## Per-query results\n\n";
  for (auto const query : b1_query_names) {
    out << "### " << b1_query_label(query);
    out << "\n\n"
        << "| metric | cache OFF median [min, max] | cache ON + prewarm ON median [min, max] | "
           "cache ON + prewarm OFF median [min, max] | cache ON prewarm OFF/ON ratio |\n"
        << "|---|---:|---:|---:|---:|\n";

    for (auto const& row : b1_metric_rows) {
      auto cache_off = summarize_metric_if_available(
        records, query, kB1CacheOffConfig, row.getter, row.requires_cache);
      auto prewarm_on = summarize_metric_if_available(
        records, query, kB1CacheOnPrewarmOn, row.getter, row.requires_cache);
      auto prewarm_off = summarize_metric_if_available(
        records, query, kB1CacheOnPrewarmOff, row.getter, row.requires_cache);
      if (row.scale_by_object_size) {
        auto const scale = static_cast<double>(object_size);
        if (cache_off) {
          cache_off =
            b1_summary{cache_off->median / scale, cache_off->min / scale, cache_off->max / scale};
        }
        if (prewarm_on) {
          prewarm_on = b1_summary{
            prewarm_on->median / scale, prewarm_on->min / scale, prewarm_on->max / scale};
        }
        if (prewarm_off) {
          prewarm_off = b1_summary{
            prewarm_off->median / scale, prewarm_off->min / scale, prewarm_off->max / scale};
        }
      }
      out << "|" << row.name << "|"
          << format_optional_summary(cache_off, row.integral && !row.scale_by_object_size) << "|"
          << format_optional_summary(prewarm_on, row.integral && !row.scale_by_object_size) << "|"
          << format_optional_summary(prewarm_off, row.integral && !row.scale_by_object_size) << "|"
          << format_optional_ratio(prewarm_off, prewarm_on) << "|\n";
    }
    out << "\n";
  }

  out << "## count(*) vs count(l_orderkey)\n\n"
      << "| config | count(*) bytes/object | count(l_orderkey) bytes/object | "
         "count(*) / count(l_orderkey) bytes | count(*) wall ms | "
         "count(l_orderkey) wall ms | count(*) / count(l_orderkey) wall |\n"
      << "|---|---:|---:|---:|---:|---:|---:|\n";
  for (auto const config : b1_config_names) {
    auto count_star_bytes = summarize_metric(records, kB1CountStarQuery, config, metric_bytes_read);
    auto count_key_bytes =
      summarize_metric(records, kB1CountProjectedQuery, config, metric_bytes_read);
    auto count_star_wall = summarize_metric(records, kB1CountStarQuery, config, metric_wall_ms);
    auto count_key_wall = summarize_metric(records, kB1CountProjectedQuery, config, metric_wall_ms);
    out << "|" << b1_config_label(config) << "|"
        << format_double(count_star_bytes.median / static_cast<double>(object_size)) << "|"
        << format_double(count_key_bytes.median / static_cast<double>(object_size)) << "|"
        << format_ratio(count_star_bytes.median, count_key_bytes.median) << "|"
        << format_double(count_star_wall.median) << "|" << format_double(count_key_wall.median)
        << "|" << format_ratio(count_star_wall.median, count_key_wall.median) << "|\n";
  }
  out << "\n";

  out << "## Observations\n\n";
  bool direction_a_trigger = false;
  bool direction_c_trigger = true;
  for (auto const query : b1_query_names) {
    auto const cache_off_wall = summarize_metric(records, query, kB1CacheOffConfig, metric_wall_ms);
    auto const on_wall  = summarize_metric(records, query, kB1CacheOnPrewarmOn, metric_wall_ms);
    auto const off_wall = summarize_metric(records, query, kB1CacheOnPrewarmOff, metric_wall_ms);
    auto const cache_off_bytes =
      summarize_metric(records, query, kB1CacheOffConfig, metric_bytes_read);
    auto const on_bytes = summarize_metric(records, query, kB1CacheOnPrewarmOn, metric_bytes_read);
    auto const off_bytes =
      summarize_metric(records, query, kB1CacheOnPrewarmOff, metric_bytes_read);
    auto const on_cache  = sum_cache_counters(records, query, kB1CacheOnPrewarmOn);
    auto const off_cache = sum_cache_counters(records, query, kB1CacheOnPrewarmOff);
    auto const total_on_cache =
      on_cache.hit_count + on_cache.partial_miss + on_cache.full_miss + on_cache.range_miss;
    auto const hit_rate = total_on_cache == 0 ? 0.0
                                              : static_cast<double>(on_cache.hit_count) /
                                                  static_cast<double>(total_on_cache);
    auto const prewarm_saves_wall =
      off_wall.median > 0.0 ? (off_wall.median - on_wall.median) / off_wall.median : 0.0;
    if (prewarm_saves_wall >= 0.10 && hit_rate > 0.50) { direction_a_trigger = true; }
    if (!(prewarm_saves_wall < 0.10 || hit_rate < 0.05)) { direction_c_trigger = false; }

    out << "- " << b1_query_label(query) << ": cache-OFF bytes/object="
        << format_double(cache_off_bytes.median / static_cast<double>(object_size))
        << ", cache-OFF wall/cache-ON+prewarm-ON wall="
        << format_ratio(cache_off_wall.median, on_wall.median)
        << ", cache-ON prewarm OFF/ON bytes=" << format_ratio(off_bytes.median, on_bytes.median)
        << ", cache-ON prewarm OFF/ON wall=" << format_ratio(off_wall.median, on_wall.median)
        << ", prewarm-ON cache hit rate=" << cache_hit_rate(on_cache)
        << "%, dominant miss ON=" << dominant_miss_counter(on_cache)
        << ", dominant miss OFF=" << dominant_miss_counter(off_cache)
        << ", FSMR borrow count nonzero on cache-OFF and cache-ON+prewarm-OFF="
        << (fsmr_nonzero_for_all_records(records, query, kB1CacheOffConfig) &&
                fsmr_nonzero_for_all_records(records, query, kB1CacheOnPrewarmOff)
              ? "yes"
              : "no")
        << ", FSMR borrow count zero on cache-ON+prewarm-ON="
        << (fsmr_nonzero_for_all_records(records, query, kB1CacheOnPrewarmOn) ? "no" : "yes")
        << ".\n";
  }

  auto const cache_off_count_star_bytes =
    summarize_metric(records, kB1CountStarQuery, kB1CacheOffConfig, metric_bytes_read);
  auto const cache_off_count_key_bytes =
    summarize_metric(records, kB1CountProjectedQuery, kB1CacheOffConfig, metric_bytes_read);
  auto const cache_off_count_star_ratio =
    cache_off_count_star_bytes.median / static_cast<double>(object_size);
  auto const cache_off_count_key_ratio =
    cache_off_count_key_bytes.median / static_cast<double>(object_size);

  out << "\n## Phase 3 decision trigger\n\n"
      << "- If cache-OFF count(*) bytes/object > 1.5 while count(l_orderkey) stays <= 1.2: "
         "file cache-OFF empty-projection redundancy as a separate backlog item.\n"
      << "- If cache-OFF count(*) bytes/object is already ~= 1.0: treat §26's earlier 1.87x "
         "as stale or non-reproduced on current code.\n"
      << "- If cache-ON prewarm ON saves >= 10% wall-clock and hit rate > 50%: Phase 3 can "
         "consider cache hit alignment; otherwise prefer the measured default/cache policy "
         "follow-up.\n"
      << "- Trigger evaluation from this run: ";
  if (cache_off_count_star_ratio > 1.5 && cache_off_count_key_ratio <= 1.2) {
    out << "cache-OFF count(*) redundancy reproduced; keep it separate from the cache-ON "
           "prewarm result.\n";
  } else if (cache_off_count_star_ratio <= 1.2) {
    out << "cache-OFF count(*) redundancy did not reproduce on current code; §26's 1.87x "
           "measurement looks stale or configuration-specific.\n";
  } else {
    out << "cache-OFF count(*) is elevated but not enough to classify cleanly; review the raw "
           "bytes before tightening guards.\n";
  }
  out << "- Cache-ON prewarm trigger evaluation: ";
  if (direction_a_trigger) {
    out << "direction A condition met.\n";
  } else if (direction_c_trigger) {
    out << "direction C condition met.\n";
  } else {
    out << "mixed signal: prewarm ON saves wall-clock, but cache hit rate stays below 50%; "
           "Phase 3 should review before choosing A vs C.\n";
  }
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

TEST_CASE("S3 SQL config guard writes AWS-only options only when configured", "[s3][config]")
{
  s3_test_env env{"https://s3.us-east-2.amazonaws.com",
                  "us-east-2",
                  "temporary-access-key",
                  "temporary-secret-key",
                  "sirius-s3-test",
                  ""};

  {
    sirius_config_env_guard guard(env);
    auto const yaml = read_text_file(guard.config_path());
    CHECK(yaml.find("object_store_config:") != std::string::npos);
    CHECK(yaml.find("session_token:") == std::string::npos);
    CHECK(yaml.find("signing_mode:") == std::string::npos);
  }

  env.session_token = "temporary-session-token";
  {
    sirius_config_env_guard guard(env, {}, std::string{"header"});
    auto const yaml = read_text_file(guard.config_path());
    CHECK(yaml.find("object_store_config:") != std::string::npos);
    CHECK(yaml.find("session_token: 'temporary-session-token'") != std::string::npos);
    CHECK(yaml.find("signing_mode: 'header'") != std::string::npos);
  }
}

TEST_CASE("gpu_execution rewrites S3 read_parquet and scans through Sirius",
          "[.][s3][integration][sql][gpu_execution]")
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

TEST_CASE("gpu_execution S3 SQL results match with the async backend flag",
          "[.][s3][integration][sql][gpu_execution][asynccurl]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto const query =
    "SELECT n_nationkey, n_name, n_regionkey "
    "FROM read_parquet('" +
    uri + "') ORDER BY n_nationkey";

  auto run_query = [&](bool use_async_backend) {
    s3_sql_fixture fixture(*env, {}, std::nullopt, use_async_backend);
    std::uint64_t before_bytes          = 0;
    sirius::io::s3::s3_ioctx* async_ctx = nullptr;
    if (use_async_backend) {
      async_ctx    = &require_async_s3_ioctx(fixture, uri);
      before_bytes = async_ctx->bytes_read_total();
    }

    auto result = require_query_ok(fixture.con, gpu_execution_sql(query));

    std::uint64_t byte_delta = 0;
    if (async_ctx != nullptr) { byte_delta = async_ctx->bytes_read_total() - before_bytes; }
    return std::pair{collect_rows(*result), byte_delta};
  };

  auto const blocking = run_query(false);
  auto const async    = run_query(true);

  CHECK(async.first == blocking.first);
  CHECK(async.first.size() == 25);
  CHECK(async.second > 0);
}

TEST_CASE("gpu_execution reads real AWS S3 parquet through Sirius SigV4",
          "[.][s3][aws][live][sql][gpu_execution]")
{
  auto env = require_aws_live_env();
  if (!env) { return; }

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto const local_query =
    "SELECT n_nationkey, n_name, n_regionkey "
    "FROM " +
    local_parquet_scan("nation") + " ORDER BY n_nationkey";
  auto baseline_result = require_query_ok(baseline_con, local_query);

  auto run_s3 = [&](std::optional<std::string> signing_mode) {
    s3_sql_fixture fixture(*env, {}, std::move(signing_mode));
    auto const s3_query =
      "SELECT n_nationkey, n_name, n_regionkey "
      "FROM " +
      s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
    return require_query_ok(fixture.con, gpu_execution_sql(s3_query));
  };

  SECTION("presigned")
  {
    auto s3_result = run_s3(std::nullopt);
    REQUIRE(s3_result->RowCount() == 25);
    REQUIRE(s3_result->ColumnCount() == 3);
    check_rows_equal_with_tolerant_columns(*s3_result, *baseline_result, {});
  }

  SECTION("header")
  {
    auto s3_result = run_s3(std::string{"header"});
    REQUIRE(s3_result->RowCount() == 25);
    REQUIRE(s3_result->ColumnCount() == 3);
    check_rows_equal_with_tolerant_columns(*s3_result, *baseline_result, {});
  }
}

TEST_CASE("gpu_execution reads real AWS S3 parquet through the async backend flag",
          "[.][s3][aws][live][sql][gpu_execution][asynccurl]")
{
  auto env = require_aws_live_env();
  if (!env) { return; }

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto const local_query =
    "SELECT n_nationkey, n_name, n_regionkey "
    "FROM " +
    local_parquet_scan("nation") + " ORDER BY n_nationkey";
  auto baseline_result = require_query_ok(baseline_con, local_query);

  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  s3_sql_fixture fixture(*env, {}, std::nullopt, true);
  auto& async_ctx   = require_async_s3_ioctx(fixture, uri);
  auto const before = async_ctx.bytes_read_total();
  auto const s3_query =
    "SELECT n_nationkey, n_name, n_regionkey FROM read_parquet('" + uri + "') ORDER BY n_nationkey";
  auto const s3_result = require_query_ok(fixture.con, gpu_execution_sql(s3_query));
  auto const bytes     = async_ctx.bytes_read_total() - before;

  REQUIRE(s3_result->RowCount() == 25);
  REQUIRE(s3_result->ColumnCount() == 3);
  check_rows_equal_with_tolerant_columns(*s3_result, *baseline_result, {});
  CHECK(bytes > 0);
}

TEST_CASE("gpu_execution aggregates real AWS S3 parquet through Sirius SigV4",
          "[.][s3][aws][live][sql][gpu_execution]")
{
  auto env = require_aws_live_env();
  if (!env) { return; }

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto const local_query =
    "SELECT count(*) AS c, min(n_nationkey) AS lo, max(n_nationkey) AS hi "
    "FROM " +
    local_parquet_scan("nation");
  auto baseline_result = require_query_ok(baseline_con, local_query);

  auto run_s3 = [&](std::optional<std::string> signing_mode) {
    s3_sql_fixture fixture(*env, {}, std::move(signing_mode));
    auto const s3_query =
      "SELECT count(*) AS c, min(n_nationkey) AS lo, max(n_nationkey) AS hi "
      "FROM " +
      s3_parquet_scan(*env, "nation");
    return require_query_ok(fixture.con, gpu_execution_sql(s3_query));
  };

  SECTION("presigned")
  {
    auto s3_result = run_s3(std::nullopt);
    REQUIRE(s3_result->RowCount() == 1);
    REQUIRE(s3_result->ColumnCount() == 3);
    check_rows_equal_with_tolerant_columns(*s3_result, *baseline_result, {});
  }

  SECTION("header")
  {
    auto s3_result = run_s3(std::string{"header"});
    REQUIRE(s3_result->RowCount() == 1);
    REQUIRE(s3_result->ColumnCount() == 3);
    check_rows_equal_with_tolerant_columns(*s3_result, *baseline_result, {});
  }
}

TEST_CASE("gpu_execution S3 window query reports unsupported S3 CPU fallback",
          "[.][s3][integration][sql][gpu_execution][fallback]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const s3_query =
    "SELECT n_nationkey, ROW_NUMBER() OVER (ORDER BY n_nationkey) AS rn "
    "FROM " +
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

TEST_CASE("DuckDB CPU read_parquet does not register a Sirius S3 filesystem",
          "[.][s3][integration][sql][filesystem]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto result    = fixture.con.Query("SELECT count(*) FROM read_parquet('" + uri + "')");
  REQUIRE(result);
  REQUIRE(result->HasError());
  auto const error = result->GetError();
  INFO(error);
  CHECK((error.find("s3") != std::string::npos || error.find("S3") != std::string::npos));
}

TEST_CASE("gpu_execution S3 SQL surface returns empty result sets cleanly",
          "[.][s3][integration][sql][gpu_execution]")
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
          "[.][s3][integration][sql][gpu_execution]")
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

TEST_CASE("internal sirius_read_parquet bind returns row-count metadata for cardinality",
          "[.][s3][integration][sql][planner-metadata]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const uri                  = s3_uri(env->bucket, "parquet/orders.parquet");
  auto const expected_orders_rows = local_parquet_row_count("orders");
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
          "[.][s3][integration][sql][planner-metadata]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const expected_orders_rows = local_parquet_row_count("orders");
  auto const plan =
    explain_text(fixture.con, "SELECT * FROM " + s3_sirius_parquet_scan(*env, "orders"));

  INFO(plan);
  CHECK(plan_mentions_cardinality(plan, expected_orders_rows));
}

TEST_CASE("internal sirius_read_parquet exposes distinct S3 table cardinalities in joins",
          "[.][s3][integration][sql][planner-metadata]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const expected_orders_rows = local_parquet_row_count("orders");
  auto const expected_nation_rows = local_parquet_row_count("nation");
  auto const sql = "SELECT count(*) FROM " + s3_sirius_parquet_scan(*env, "orders") + " o JOIN " +
                   s3_sirius_parquet_scan(*env, "nation") +
                   " n ON (o.o_custkey % 25) = n.n_nationkey";
  auto const plan = explain_text(fixture.con, sql);

  INFO(plan);
  CHECK(plan_mentions_cardinality(plan, expected_orders_rows));
  CHECK(plan_mentions_cardinality(plan, expected_nation_rows));
}

TEST_CASE("gpu_execution S3 SQL surface scans all orders row groups",
          "[.][s3][integration][sql][gpu_execution]")
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

TEST_CASE("gpu_execution large S3 lineitem count matches the local parquet oracle",
          "[.][s3][sql][large][large-count][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits());
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto& s3_ctx              = require_async_s3_ioctx(fixture, large->uri);
  auto const expected_rows  = local_parquet_file_row_count(large->local_path);
  auto const before_bytes   = s3_ctx.bytes_read_total();
  auto const s3_count_query = "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env);
  auto s3_result            = require_query_ok(fixture.con, gpu_execution_sql(s3_count_query));
  auto const byte_delta     = s3_ctx.bytes_read_total() - before_bytes;

  REQUIRE(s3_result->RowCount() == 1);
  REQUIRE(s3_result->ColumnCount() == 1);
  CHECK(s3_result->GetValue(0, 0).GetValue<int64_t>() == static_cast<int64_t>(expected_rows));
  CHECK(large->total_num_rows == expected_rows);
  CHECK(expected_rows > 50'000'000);
  CHECK(large->object_size == fs::file_size(large->local_path));
  INFO("byte_delta=" << byte_delta << " object_size=" << large->object_size);
  CHECK(within_large_s3_byte_budget(byte_delta, large->object_size));
}

TEST_CASE("gpu_execution large S3 lineitem count matches with the async backend flag",
          "[.][s3][sql][large][large-count][gpu_execution][integration][asynccurl]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  struct run_result {
    int64_t count{0};
    std::uint64_t bytes_read{0};
    std::uint64_t device_copies{0};
    std::uint64_t device_stream_syncs{0};
  };

  auto run_query = [&](bool use_async_backend) -> std::optional<run_result> {
    s3_sql_fixture fixture(*env, large_sirius_memory_limits(), std::nullopt, use_async_backend);
    auto large = read_large_lineitem_fixture(fixture, *env);
    if (!large) { return std::nullopt; }

    auto const query = "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env);
    if (!use_async_backend) {
      auto result = require_query_ok(fixture.con, gpu_execution_sql(query));
      REQUIRE(result->RowCount() == 1);
      return run_result{result->GetValue(0, 0).GetValue<int64_t>(), 0, 0, 0};
    }

    auto& async_ctx          = require_async_s3_ioctx(fixture, large->uri);
    auto const before_bytes  = async_ctx.bytes_read_total();
    auto const before_copies = async_ctx.device_copies_total();
    auto const before_syncs  = async_ctx.device_stream_sync_total();
    auto result              = require_query_ok(fixture.con, gpu_execution_sql(query));
    auto const byte_delta    = async_ctx.bytes_read_total() - before_bytes;
    auto const copy_delta    = async_ctx.device_copies_total() - before_copies;
    auto const sync_delta    = async_ctx.device_stream_sync_total() - before_syncs;

    REQUIRE(result->RowCount() == 1);
    return run_result{
      result->GetValue(0, 0).GetValue<int64_t>(), byte_delta, copy_delta, sync_delta};
  };

  auto const blocking = run_query(false);
  if (!blocking) { return; }
  auto const async = run_query(true);
  if (!async) { return; }

  CHECK(async->count == blocking->count);
  CHECK(blocking->count > 50'000'000);
  CHECK(async->bytes_read > 0);
  CHECK(async->device_copies > 0);
  CHECK(async->device_stream_syncs == 0);
}

TEST_CASE("gpu_execution large S3 lineitem TPC-H Q1 shape matches local CPU",
          "[.][s3][sql][large][large-q1][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits());
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto& s3_ctx              = require_async_s3_ioctx(fixture, large->uri);
  auto const before_bytes   = s3_ctx.bytes_read_total();
  auto const before_borrows = s3_ctx.fsmr_borrows_total();
  auto const s3_query       = tpch_q1_shape_query(s3_large_lineitem_scan(*env));
  auto s3_result            = require_query_ok(fixture.con, gpu_execution_sql(s3_query));
  auto const byte_delta     = s3_ctx.bytes_read_total() - before_bytes;
  auto const borrow_delta   = s3_ctx.fsmr_borrows_total() - before_borrows;

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto const local_query = tpch_q1_shape_query(local_parquet_file_scan(large->local_path));
  auto baseline_result   = require_query_ok(baseline_con, local_query);

  REQUIRE(s3_result->RowCount() == baseline_result->RowCount());
  REQUIRE(s3_result->ColumnCount() == baseline_result->ColumnCount());
  check_rows_equal_with_tolerant_columns(*s3_result, *baseline_result, {6, 7, 8});
  CHECK(byte_delta > 0);
  INFO("byte_delta=" << byte_delta << " object_size=" << large->object_size);
  CHECK(within_large_s3_byte_budget(byte_delta, large->object_size));
  CHECK(borrow_delta > 0);
}

TEST_CASE("gpu_execution large S3 lineitem join uses planner cardinality and matches local CPU",
          "[.][s3][sql][large][large-join][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits());
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto const expected_lineitem_rows = local_parquet_file_row_count(large->local_path);
  auto const expected_orders_rows   = local_parquet_row_count("orders");
  auto const s3_query =
    large_lineitem_orders_join_query(s3_large_lineitem_scan(*env), s3_parquet_scan(*env, "orders"));
  auto s3_result = require_query_ok(fixture.con, gpu_execution_sql(s3_query));

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto const local_query = large_lineitem_orders_join_query(
    local_parquet_file_scan(large->local_path), local_parquet_scan("orders"));
  auto baseline_result = require_query_ok(baseline_con, local_query);

  REQUIRE(s3_result->RowCount() == baseline_result->RowCount());
  REQUIRE(s3_result->ColumnCount() == baseline_result->ColumnCount());
  CHECK(collect_rows(*s3_result) == collect_rows(*baseline_result));

  auto const explain_sql = large_lineitem_orders_join_query(s3_sirius_large_lineitem_scan(*env),
                                                            s3_sirius_parquet_scan(*env, "orders"));
  auto const plan        = explain_text(fixture.con, explain_sql);
  INFO(plan);
  CHECK(plan_mentions_cardinality(plan, expected_lineitem_rows));
  CHECK(plan_mentions_cardinality(plan, expected_orders_rows));
}

TEST_CASE("gpu_execution large S3 lineitem count stays within byte budget without chunk prewarm",
          "[.][s3][sql][large][large-count-no-prewarm][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits_without_chunk_prewarm());
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto& s3_ctx = require_async_s3_ioctx(fixture, large->uri);
  REQUIRE(s3_ctx.cache() != nullptr);
  auto const expected_rows       = local_parquet_file_row_count(large->local_path);
  auto const before_bytes        = s3_ctx.bytes_read_total();
  auto const before_hits         = s3_ctx.cache()->hit_count_total();
  auto const before_range_misses = s3_ctx.cache()->range_miss_count_total();
  auto const s3_count_query      = "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env);
  auto s3_result                 = require_query_ok(fixture.con, gpu_execution_sql(s3_count_query));
  auto const byte_delta          = s3_ctx.bytes_read_total() - before_bytes;
  auto const hit_delta           = s3_ctx.cache()->hit_count_total() - before_hits;
  auto const range_miss_delta    = s3_ctx.cache()->range_miss_count_total() - before_range_misses;

  REQUIRE(s3_result->RowCount() == 1);
  REQUIRE(s3_result->ColumnCount() == 1);
  CHECK(s3_result->GetValue(0, 0).GetValue<int64_t>() == static_cast<int64_t>(expected_rows));
  INFO("byte_delta=" << byte_delta << " object_size=" << large->object_size);
  CHECK(within_no_prewarm_s3_byte_budget(byte_delta, large->object_size));
  CHECK(hit_delta == 0);
  CHECK(range_miss_delta > 0);
}

TEST_CASE("gpu_execution large S3 lineitem Q1 shape matches local CPU without chunk prewarm",
          "[.][s3][sql][large][large-q1-no-prewarm][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits_without_chunk_prewarm());
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto const s3_query = tpch_q1_shape_query(s3_large_lineitem_scan(*env));
  auto s3_result      = require_query_ok(fixture.con, gpu_execution_sql(s3_query));

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto const local_query = tpch_q1_shape_query(local_parquet_file_scan(large->local_path));
  auto baseline_result   = require_query_ok(baseline_con, local_query);

  REQUIRE(s3_result->RowCount() == baseline_result->RowCount());
  REQUIRE(s3_result->ColumnCount() == baseline_result->ColumnCount());
  check_rows_equal_with_tolerant_columns(*s3_result, *baseline_result, {6, 7, 8});
}

TEST_CASE("gpu_execution large S3 lineitem join matches local CPU without chunk prewarm",
          "[.][s3][sql][large][large-join-no-prewarm][gpu_execution][integration]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits_without_chunk_prewarm());
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto const s3_query =
    large_lineitem_orders_join_query(s3_large_lineitem_scan(*env), s3_parquet_scan(*env, "orders"));
  auto s3_result = require_query_ok(fixture.con, gpu_execution_sql(s3_query));

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto const local_query = large_lineitem_orders_join_query(
    local_parquet_file_scan(large->local_path), local_parquet_scan("orders"));
  auto baseline_result = require_query_ok(baseline_con, local_query);

  REQUIRE(s3_result->RowCount() == baseline_result->RowCount());
  REQUIRE(s3_result->ColumnCount() == baseline_result->ColumnCount());
  CHECK(collect_rows(*s3_result) == collect_rows(*baseline_result));
}

TEST_CASE("B1 Phase 3a cache-mode bench records SF10 S3 SQL telemetry",
          "[!benchmark][b1_bench][s3][sql][large]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  constexpr int kIterations = 3;
  struct bench_config {
    std::string_view name;
    sirius_memory_limits limits;
    bool expects_cache{false};
    bool expects_fsmr_borrows{true};
  };
  struct bench_query {
    std::string_view name;
    std::string sql;
  };

  std::array<bench_config, 3> const configs{{
    {kB1CacheOffConfig, large_sirius_memory_limits(), false, true},
    {kB1CacheOnPrewarmOn, large_sirius_memory_limits_with_chunk_prewarm(true), true, false},
    {kB1CacheOnPrewarmOff, large_sirius_memory_limits_with_chunk_prewarm(false), true, true},
  }};

  std::vector<b1_run_record> records;
  std::optional<std::size_t> object_size;

  for (auto const& config : configs) {
    for (int query_index = 0; query_index < static_cast<int>(b1_query_names.size());
         ++query_index) {
      for (int iteration = 1; iteration <= kIterations; ++iteration) {
        s3_sql_fixture fixture(*env, config.limits);
        auto large = read_large_lineitem_fixture(fixture, *env);
        if (!large) { return; }
        if (!object_size) { object_size = large->object_size; }

        std::array<bench_query, 4> const queries{{
          {kB1CountStarQuery, "SELECT count(*) FROM " + s3_large_lineitem_scan(*env)},
          {kB1CountProjectedQuery, "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env)},
          {kB1Q1Query, tpch_q1_shape_query(s3_large_lineitem_scan(*env))},
          {kB1JoinQuery,
           large_lineitem_orders_join_query(s3_large_lineitem_scan(*env),
                                            s3_parquet_scan(*env, "orders"))},
        }};

        auto& s3_ctx = require_async_s3_ioctx(fixture, large->uri);
        auto* cache  = s3_ctx.cache();
        if (config.expects_cache) {
          REQUIRE(cache != nullptr);
        } else {
          REQUIRE(cache == nullptr);
        }
        auto const& query         = queries[static_cast<std::size_t>(query_index)];
        auto const before_bytes   = s3_ctx.bytes_read_total();
        auto const before_borrows = s3_ctx.fsmr_borrows_total();
        auto const before_cache   = cache ? std::optional{snapshot_cache(*cache)} : std::nullopt;

        auto const start = std::chrono::steady_clock::now();
        auto result      = require_query_ok(fixture.con, gpu_execution_sql(query.sql));
        auto const stop  = std::chrono::steady_clock::now();

        auto const after_cache  = cache ? std::optional{snapshot_cache(*cache)} : std::nullopt;
        auto const wall_ms      = std::chrono::duration<double, std::milli>(stop - start).count();
        auto const borrow_delta = s3_ctx.fsmr_borrows_total() - before_borrows;
        auto cache_delta        = b1_cache_counters{};
        if (before_cache && after_cache) { cache_delta = *after_cache - *before_cache; }

        REQUIRE(result->RowCount() > 0);
        if (config.expects_fsmr_borrows) {
          CHECK(borrow_delta > 0);
        } else {
          CHECK(borrow_delta == 0);
        }

        records.push_back(b1_run_record{
          std::string{query.name},
          std::string{config.name},
          iteration,
          b1_metrics{wall_ms, s3_ctx.bytes_read_total() - before_bytes, borrow_delta, cache_delta},
          cache != nullptr,
        });
      }
    }
  }

  REQUIRE(object_size.has_value());
  REQUIRE(records.size() == configs.size() * b1_query_names.size() * kIterations);
  auto const output_path = b1_bench_output_path();
  write_b1_bench_markdown(output_path, records, *object_size);
  CHECK(fs::exists(output_path));
}
