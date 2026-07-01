/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/s3/s3_object_ref.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "utils/s3_container.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/utilities/span.hpp>

#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
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
  std::optional<std::size_t> rest_n_reactors;
  std::optional<std::size_t> rest_max_connections;
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
           "        block_size: 1 MiB\n"
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
    if (limits.rest_n_reactors.has_value()) {
      out << "      rest_n_reactors: " << *limits.rest_n_reactors << "\n";
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
           "        max_connections: "
        << limits.rest_max_connections.value_or(std::size_t{8})
        << "\n"
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

using bench_clock = std::chrono::steady_clock;

double elapsed_ms(bench_clock::time_point start, bench_clock::time_point stop)
{
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

double mean_ns_as_ms(std::uint64_t total_ns, std::uint64_t count)
{
  if (count == 0) { return 0.0; }
  return static_cast<double>(total_ns) / static_cast<double>(count) / 1'000'000.0;
}

std::uint64_t sat_sub(std::uint64_t after, std::uint64_t before)
{
  return after >= before ? after - before : 0;
}

sirius::io::rest::rest_perf_snapshot delta_snapshot(
  sirius::io::rest::rest_perf_snapshot const& after,
  sirius::io::rest::rest_perf_snapshot const& before)
{
  sirius::io::rest::rest_perf_snapshot out;
  out.chunk_get_ns_total    = sat_sub(after.chunk_get_ns_total, before.chunk_get_ns_total);
  out.chunk_get_count       = sat_sub(after.chunk_get_count, before.chunk_get_count);
  out.chunk_get_ns_max      = after.chunk_get_ns_max;
  out.queue_wait_ns_total   = sat_sub(after.queue_wait_ns_total, before.queue_wait_ns_total);
  out.queue_wait_count      = sat_sub(after.queue_wait_count, before.queue_wait_count);
  out.ttfb_ns               = sat_sub(after.ttfb_ns, before.ttfb_ns);
  out.h2d_observed_ns_total = sat_sub(after.h2d_observed_ns_total, before.h2d_observed_ns_total);
  out.h2d_observed_count    = sat_sub(after.h2d_observed_count, before.h2d_observed_count);
  out.h2d_observed_ns_max   = after.h2d_observed_ns_max;
  out.retries_total         = sat_sub(after.retries_total, before.retries_total);
  out.terminal_failures_total =
    sat_sub(after.terminal_failures_total, before.terminal_failures_total);
  out.device_stream_sync_total =
    sat_sub(after.device_stream_sync_total, before.device_stream_sync_total);
  out.payload_bytes_read_total =
    sat_sub(after.payload_bytes_read_total, before.payload_bytes_read_total);
  return out;
}

struct rest_bench_measurement {
  double wall_clock_ms{0.0};
  double open_ms{0.0};
  double footer_fetch_ms{0.0};
  double metadata_parse_ms{0.0};
  double scan_ms{0.0};
  std::uint64_t payload_bytes_read{0};
  duckdb::idx_t rows{0};
  sirius::io::rest::rest_perf_snapshot micro;
};

std::unique_ptr<cudf::io::datasource::buffer> read_parquet_footer_for_bench(
  cudf::io::datasource& source)
{
  auto constexpr footer_tail_size = sizeof(cudf::io::parquet::file_ender_s);
  auto const file_size            = source.size();
  REQUIRE(file_size >= footer_tail_size);

  auto tail                 = source.host_read(file_size - footer_tail_size, footer_tail_size);
  std::uint32_t footer_size = 0;
  std::memcpy(&footer_size, tail->data(), sizeof(footer_size));
  INFO("file_size=" << file_size << " footer_size=" << footer_size);
  REQUIRE(file_size >= footer_tail_size + footer_size);

  return source.host_read(file_size - footer_tail_size - footer_size, footer_size);
}

rest_bench_measurement run_rest_parquet_scan(s3_sql_fixture& fixture,
                                             std::string const& uri,
                                             std::vector<std::string> const& columns)
{
  auto& manager = require_sirius_context(fixture).get_scan_manager();

  auto const wall_start = bench_clock::now();

  auto const open_start = bench_clock::now();
  auto datasource       = manager.create_datasource(uri);
  auto const open_stop  = bench_clock::now();
  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  auto io_ctx = datasource->io_ctx();
  auto* rest  = dynamic_cast<sirius::io::rest::rest_ioctx*>(io_ctx.get());
  REQUIRE(rest != nullptr);
  auto const before = rest->perf_snapshot();

  auto const footer_start = bench_clock::now();
  auto footer_buffer      = read_parquet_footer_for_bench(*datasource);
  auto const footer_stop  = bench_clock::now();

  auto opts = cudf::io::parquet_reader_options::builder().column_names(columns).build();

  auto const parse_start = bench_clock::now();
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer_buffer->data(), footer_buffer->size()), opts};
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  metadatas.push_back(reader.parquet_metadata());
  auto const parse_stop = bench_clock::now();

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(datasource->duplicate());

  auto const scan_start  = bench_clock::now();
  auto [table, metadata] = cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts);
  (void)metadata;
  auto const scan_stop = bench_clock::now();
  auto const wall_stop = bench_clock::now();
  auto const after     = rest->perf_snapshot();
  auto const micro     = delta_snapshot(after, before);

  return rest_bench_measurement{elapsed_ms(wall_start, wall_stop),
                                elapsed_ms(open_start, open_stop),
                                elapsed_ms(footer_start, footer_stop),
                                elapsed_ms(parse_start, parse_stop),
                                elapsed_ms(scan_start, scan_stop),
                                micro.payload_bytes_read_total,
                                static_cast<duckdb::idx_t>(table->num_rows()),
                                micro};
}

struct metric_delta {
  std::string metric;
  double baseline{0.0};
  double current{0.0};
  double delta_pct{0.0};
};

struct bench_record {
  std::string scenario;
  double wall_clock_ms{0.0};
  double open_ms{0.0};
  double footer_fetch_ms{0.0};
  double metadata_parse_ms{0.0};
  double scan_ms{0.0};
  std::uint64_t payload_bytes_read{0};
  duckdb::idx_t row_count{0};
  double effective_bytes_per_sec{0.0};
  std::uint64_t chunk_get_ns_total{0};
  std::uint64_t chunk_get_count{0};
  std::uint64_t chunk_get_ns_max{0};
  double chunk_get_ms_mean{0.0};
  std::uint64_t queue_wait_ns_total{0};
  std::uint64_t queue_wait_count{0};
  double queue_wait_ms_mean{0.0};
  std::uint64_t h2d_observed_ns_total{0};
  std::uint64_t h2d_observed_count{0};
  std::uint64_t h2d_observed_ns_max{0};
  double h2d_observed_ms_mean{0.0};
  std::uint64_t ttfb_ns{0};
  std::uint64_t retries_total{0};
  std::uint64_t terminal_failures_total{0};
  std::uint64_t device_stream_sync_total{0};
  std::vector<metric_delta> comparisons;
};

bench_record make_record(std::string scenario,
                         rest_bench_measurement measurement,
                         std::uint64_t dataset_bytes)
{
  auto seconds = measurement.wall_clock_ms / 1000.0;
  auto effective =
    seconds > 0.0 ? static_cast<double>(dataset_bytes) / seconds : static_cast<double>(0);
  auto const& micro = measurement.micro;
  return bench_record{std::move(scenario),
                      measurement.wall_clock_ms,
                      measurement.open_ms,
                      measurement.footer_fetch_ms,
                      measurement.metadata_parse_ms,
                      measurement.scan_ms,
                      measurement.payload_bytes_read,
                      measurement.rows,
                      effective,
                      micro.chunk_get_ns_total,
                      micro.chunk_get_count,
                      micro.chunk_get_ns_max,
                      mean_ns_as_ms(micro.chunk_get_ns_total, micro.chunk_get_count),
                      micro.queue_wait_ns_total,
                      micro.queue_wait_count,
                      mean_ns_as_ms(micro.queue_wait_ns_total, micro.queue_wait_count),
                      micro.h2d_observed_ns_total,
                      micro.h2d_observed_count,
                      micro.h2d_observed_ns_max,
                      mean_ns_as_ms(micro.h2d_observed_ns_total, micro.h2d_observed_count),
                      micro.ttfb_ns,
                      micro.retries_total,
                      micro.terminal_failures_total,
                      micro.device_stream_sync_total,
                      {}};
}

std::string json_escape(std::string_view value)
{
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\':
      case '"':
        escaped.push_back('\\');
        escaped.push_back(c);
        break;
      case '\n': escaped += "\\n"; break;
      default: escaped.push_back(c); break;
    }
  }
  return escaped;
}

fs::path unittest_log_dir()
{
#ifdef SIRIUS_UNITTEST_LOG_DIR
  return fs::path{SIRIUS_UNITTEST_LOG_DIR};
#else
  return fs::path(SIRIUS_PROJECT_ROOT) / "build" / "release" / "extension" / "sirius" / "test" /
         "cpp" / "log";
#endif
}

fs::path perf_json_path()
{
  auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  return unittest_log_dir() / ("s3_rest_perf_" + std::to_string(stamp) + ".json");
}

fs::path perf_baseline_path()
{
  return fs::path(SIRIUS_PROJECT_ROOT) / "doc" / "s3support" / "perf-baseline-minio.json";
}

std::optional<std::string> read_optional_text_file(fs::path const& path)
{
  std::ifstream in(path);
  if (!in) { return std::nullopt; }
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

std::optional<std::string> find_result_object(std::string const& json, std::string_view scenario)
{
  auto const needle = "\"scenario\": \"" + std::string{scenario} + "\"";
  auto const pos    = json.find(needle);
  if (pos == std::string::npos) { return std::nullopt; }
  auto const start = json.rfind('{', pos);
  if (start == std::string::npos) { return std::nullopt; }

  int depth = 0;
  for (std::size_t i = start; i < json.size(); ++i) {
    if (json[i] == '{') {
      ++depth;
    } else if (json[i] == '}') {
      --depth;
      if (depth == 0) { return json.substr(start, i - start + 1); }
    }
  }
  return std::nullopt;
}

std::optional<double> extract_json_number(std::string const& json_object, std::string_view key)
{
  auto const needle = "\"" + std::string{key} + "\"";
  auto pos          = json_object.find(needle);
  if (pos == std::string::npos) { return std::nullopt; }
  pos = json_object.find(':', pos + needle.size());
  if (pos == std::string::npos) { return std::nullopt; }
  ++pos;
  while (pos < json_object.size() &&
         std::isspace(static_cast<unsigned char>(json_object[pos])) != 0) {
    ++pos;
  }
  auto end = pos;
  while (end < json_object.size()) {
    auto const c = json_object[end];
    if (!(std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '-' || c == '+' || c == '.' ||
          c == 'e' || c == 'E')) {
      break;
    }
    ++end;
  }
  if (end == pos) { return std::nullopt; }
  try {
    return std::stod(json_object.substr(pos, end - pos));
  } catch (...) {
    return std::nullopt;
  }
}

void attach_baseline_comparison(bench_record& record, std::optional<std::string> const& baseline)
{
  if (!baseline) { return; }
  auto object = find_result_object(*baseline, record.scenario);
  if (!object) { return; }

  auto add_delta = [&](std::string metric, double current) {
    auto baseline_value = extract_json_number(*object, metric);
    if (!baseline_value || *baseline_value == 0.0) { return; }
    auto const delta_pct = ((current - *baseline_value) / *baseline_value) * 100.0;
    record.comparisons.push_back(
      metric_delta{std::move(metric), *baseline_value, current, delta_pct});
  };

  add_delta("footer_fetch_ms", record.footer_fetch_ms);
  add_delta("chunk_get_ms_mean", record.chunk_get_ms_mean);

  for (auto const& delta : record.comparisons) {
    if (delta.metric == "footer_fetch_ms" && delta.delta_pct > 50.0) {
      WARN(record.scenario << " footer_fetch_ms is " << delta.delta_pct
                           << "% above perf-baseline-minio.json");
    }
  }
}

void write_perf_json(fs::path const& path,
                     s3_test_env const& env,
                     std::uint64_t dataset_bytes,
                     std::vector<bench_record> const& records)
{
  fs::create_directories(path.parent_path());
  std::ofstream out(path);
  REQUIRE(out);

  out << "{\n";
  out << "  \"git_sha\": \"" << json_escape(env_or("SIRIUS_BENCH_GIT_SHA", "unknown")) << "\",\n";
  out << "  \"host\": \"" << json_escape(env_or("HOSTNAME", "unknown")) << "\",\n";
  out << "  \"backend\": \"rest_minio\",\n";
  out << "  \"dataset_bytes\": " << dataset_bytes << ",\n";
  out << "  \"config\": {\"bucket\": \"" << json_escape(env.bucket) << "\", \"key\": \""
      << json_escape(sf10_lineitem_key()) << "\"},\n";
  out << "  \"results\": [\n";
  for (std::size_t i = 0; i < records.size(); ++i) {
    auto const& r = records[i];
    out << "    {\"scenario\": \"" << json_escape(r.scenario) << "\", "
        << "\"wall_clock_ms\": " << std::fixed << std::setprecision(3) << r.wall_clock_ms << ", "
        << "\"open_ms\": " << r.open_ms << ", "
        << "\"footer_fetch_ms\": " << r.footer_fetch_ms << ", "
        << "\"metadata_parse_ms\": " << r.metadata_parse_ms << ", "
        << "\"scan_ms\": " << r.scan_ms << ", "
        << "\"payload_bytes_read\": " << r.payload_bytes_read << ", "
        << "\"row_count\": " << r.row_count << ", "
        << "\"effective_bytes_per_sec\": " << r.effective_bytes_per_sec << ", "
        << "\"chunk_get_ns_total\": " << r.chunk_get_ns_total << ", "
        << "\"chunk_get_count\": " << r.chunk_get_count << ", "
        << "\"chunk_get_ns_max\": " << r.chunk_get_ns_max << ", "
        << "\"chunk_get_ms_mean\": " << r.chunk_get_ms_mean << ", "
        << "\"queue_wait_ns_total\": " << r.queue_wait_ns_total << ", "
        << "\"queue_wait_count\": " << r.queue_wait_count << ", "
        << "\"queue_wait_ms_mean\": " << r.queue_wait_ms_mean << ", "
        << "\"h2d_observed_ns_total\": " << r.h2d_observed_ns_total << ", "
        << "\"h2d_observed_count\": " << r.h2d_observed_count << ", "
        << "\"h2d_observed_ns_max\": " << r.h2d_observed_ns_max << ", "
        << "\"h2d_observed_ms_mean\": " << r.h2d_observed_ms_mean << ", "
        << "\"ttfb_ns\": " << r.ttfb_ns << ", "
        << "\"retries_total\": " << r.retries_total << ", "
        << "\"terminal_failures_total\": " << r.terminal_failures_total << ", "
        << "\"device_stream_sync_total\": " << r.device_stream_sync_total;
    if (!r.comparisons.empty()) {
      out << ", \"comparison\": {";
      for (std::size_t c = 0; c < r.comparisons.size(); ++c) {
        auto const& cmp = r.comparisons[c];
        out << "\"" << json_escape(cmp.metric) << "\": {\"baseline\": " << cmp.baseline
            << ", \"current\": " << cmp.current << ", \"delta_pct\": " << cmp.delta_pct << "}";
        if (c + 1 != r.comparisons.size()) { out << ", "; }
      }
      out << "}";
    }
    out << "}";
    out << (i + 1 == records.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
}

void require_perf_json_schema(fs::path const& path)
{
  auto const json = read_text_file(path);
  for (auto key : {"\"git_sha\"",
                   "\"host\"",
                   "\"backend\"",
                   "\"dataset_bytes\"",
                   "\"scenario\"",
                   "\"async_http\"",
                   "\"async_https\"",
                   "\"compat_http\"",
                   "\"compat_https\"",
                   "\"wall_clock_ms\"",
                   "\"open_ms\"",
                   "\"footer_fetch_ms\"",
                   "\"metadata_parse_ms\"",
                   "\"scan_ms\"",
                   "\"payload_bytes_read\"",
                   "\"row_count\"",
                   "\"effective_bytes_per_sec\"",
                   "\"chunk_get_ns_total\"",
                   "\"chunk_get_count\"",
                   "\"chunk_get_ns_max\"",
                   "\"chunk_get_ms_mean\"",
                   "\"queue_wait_ns_total\"",
                   "\"queue_wait_count\"",
                   "\"queue_wait_ms_mean\"",
                   "\"h2d_observed_ns_total\"",
                   "\"h2d_observed_count\"",
                   "\"h2d_observed_ns_max\"",
                   "\"h2d_observed_ms_mean\"",
                   "\"ttfb_ns\"",
                   "\"device_stream_sync_total\"",
                   "\"retries_total\"",
                   "\"terminal_failures_total\"",
                   "\"config\""}) {
    CHECK(json.find(key) != std::string::npos);
  }
}

std::optional<large_lineitem_fixture> read_large_lineitem_bench_fixture(s3_test_env const& env)
{
  if (!truthy_env("SIRIUS_TEST_S3_LARGE")) {
    SUCCEED("SIRIUS_TEST_S3_LARGE not set; skipping S3 perf benchmark");
    return std::nullopt;
  }

  large_lineitem_fixture out;
  out.uri        = s3_large_lineitem_uri(env);
  out.local_path = local_sf10_lineitem_path();
  if (!fs::exists(out.local_path)) {
    if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
      FAIL("SF10 local parquet fixture is required in strict mode: " + out.local_path.string());
    }
    SUCCEED("SF10 local parquet fixture is absent; skipping S3 perf benchmark");
    return std::nullopt;
  }
  out.total_num_rows = local_parquet_file_row_count(out.local_path);
  return out;
}

std::vector<std::string> bench_single_column_projection() { return {"l_orderkey"}; }

std::vector<std::string> bench_compat_projection()
{
  return {"l_orderkey",
          "l_partkey",
          "l_suppkey",
          "l_linenumber",
          "l_quantity",
          "l_extendedprice",
          "l_discount"};
}

bench_record run_rest_minio_bench_scenario(
  s3_test_env const& env,
  large_lineitem_fixture const& large,
  std::string scenario,
  std::string const& endpoint,
  std::optional<std::string> ca_bundle,
  bool tls_verify,
  bool perf_instrumentation,
  std::vector<std::string> columns,
  std::optional<std::size_t> rest_max_connections = std::nullopt,
  std::optional<std::size_t> rest_n_reactors      = std::nullopt,
  bool enable_prefetch_cache                      = true)
{
  INFO("scenario=" << scenario << " columns=" << columns.size()
                   << " max_connections=" << rest_max_connections.value_or(std::size_t{8})
                   << " rest_n_reactors=" << rest_n_reactors.value_or(std::size_t{2})
                   << " enable_prefetch_cache=" << enable_prefetch_cache);
  auto limits                      = large_sirius_memory_limits(enable_prefetch_cache);
  limits.rest_perf_instrumentation = perf_instrumentation;
  limits.rest_max_connections      = rest_max_connections;
  limits.rest_n_reactors           = rest_n_reactors;
  if (columns.size() > 1) {
    // The compatibility run mirrors the old #982 seven-column async-S3 baseline.
    // SF10 materializes a much wider cuDF table than the single-column CI
    // baseline, so give only that scenario a larger GPU budget.
    limits.gpu_usage       = "8 GiB";
    limits.gpu_reservation = "3 GiB";
  }
  if (rest_max_connections.value_or(std::size_t{8}) >= std::size_t{32}) {
    // The historical #982 async-S3 bench used mc=32.  The REST reactor parks one
    // host bounce slot per connection, so give that compatibility scenario a
    // larger host budget without changing the production-shape scenarios above.
    limits.host_capacity = "12 GiB";
  }
  s3_sql_fixture fixture(env, limits, std::nullopt, endpoint, std::move(ca_bundle), tls_verify);
  auto measurement = run_rest_parquet_scan(fixture, large.uri, columns);
  CHECK(measurement.rows == large.total_num_rows);
  CHECK(measurement.payload_bytes_read > 0);
  CHECK(measurement.open_ms > 0.0);
  CHECK(measurement.footer_fetch_ms > 0.0);
  CHECK(measurement.metadata_parse_ms > 0.0);
  CHECK(measurement.scan_ms > 0.0);
  CHECK(measurement.wall_clock_ms + 0.001 >=
        measurement.open_ms + measurement.footer_fetch_ms + measurement.scan_ms);
  CHECK(measurement.micro.device_stream_sync_total == 0);
  CHECK(measurement.micro.terminal_failures_total == 0);
  return make_record(std::move(scenario), measurement, measurement.payload_bytes_read);
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

TEST_CASE("S3 bench STS session token reaches presigned URLs",
          "[s3][authorizer][credential_provider]")
{
  sirius::io::s3::static_credentials creds;
  creds.access_key_id     = "AKIAFAKEBENCHKEY";
  creds.secret_access_key = "fake-secret-key";
  creds.session_token     = "fake-session-token";

  sirius::io::s3::sirius_sigv4_presigned_authorizer authorizer{
    std::move(creds), "us-east-2", "https://s3.us-east-2.amazonaws.com"};
  auto request = authorizer.authorize(
    sirius::io::s3::s3_object_ref{"sirius-bench", "tpch/lineitem_sf10.parquet"},
    sirius::io::s3::s3_request_method::GET,
    std::chrono::seconds{60});

  CHECK(request.headers.empty());
  CHECK(request.url.find("X-Amz-Security-Token=") != std::string::npos);
  CHECK(request.url.find("fake-session-token") != std::string::npos);
}

TEST_CASE("S3 REST bench perf instrumentation gate keeps micro counters zero", "[.][s3][bench]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }
  auto large = read_large_lineitem_bench_fixture(*env);
  if (!large) { return; }

  auto record = run_rest_minio_bench_scenario(*env,
                                              *large,
                                              "async_http_gate_off",
                                              env->endpoint,
                                              std::nullopt,
                                              false,
                                              /*perf_instrumentation=*/false,
                                              bench_single_column_projection());
  CHECK(record.row_count == large->total_num_rows);
  CHECK(record.payload_bytes_read > 0);
  CHECK(record.chunk_get_count == 0);
  CHECK(record.chunk_get_ns_total == 0);
  CHECK(record.chunk_get_ns_max == 0);
  CHECK(record.queue_wait_count == 0);
  CHECK(record.queue_wait_ns_total == 0);
  CHECK(record.h2d_observed_count == 0);
  CHECK(record.h2d_observed_ns_total == 0);
  CHECK(record.h2d_observed_ns_max == 0);
  CHECK(record.ttfb_ns == 0);
  CHECK(record.retries_total == 0);
  CHECK(record.terminal_failures_total == 0);
  CHECK(record.device_stream_sync_total == 0);
}

TEST_CASE("S3 REST perf benchmark emits HTTP and HTTPS JSON baseline", "[.][s3][bench]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }
  if (env->https_endpoint.empty() || env->ca_bundle_path.empty()) {
    if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
      FAIL(
        "SIRIUS_TEST_S3_HTTPS_ENDPOINT and SIRIUS_TEST_S3_CA_BUNDLE are required in strict "
        "s3-bench mode");
    }
    SUCCEED("HTTPS MinIO endpoint is absent; skipping S3 REST perf benchmark");
    return;
  }
  auto large = read_large_lineitem_bench_fixture(*env);
  if (!large) { return; }

  auto baseline_json = read_optional_text_file(perf_baseline_path());
  std::vector<bench_record> records;
  records.reserve(4);

  // compat_* mirrors the old async-S3 benchmark shape: seven projected columns,
  // mc=32, and no scan-manager prefetch cache mixed into the raw S3 read path.
  auto compat_http  = run_rest_minio_bench_scenario(*env,
                                                   *large,
                                                   "compat_http",
                                                   env->endpoint,
                                                   std::nullopt,
                                                   false,
                                                   /*perf_instrumentation=*/true,
                                                   bench_compat_projection(),
                                                   std::size_t{32},
                                                   std::size_t{1},
                                                   /*enable_prefetch_cache=*/false);
  auto compat_https = run_rest_minio_bench_scenario(*env,
                                                    *large,
                                                    "compat_https",
                                                    env->https_endpoint,
                                                    env->ca_bundle_path,
                                                    true,
                                                    /*perf_instrumentation=*/true,
                                                    bench_compat_projection(),
                                                    std::size_t{32},
                                                    std::size_t{1},
                                                    /*enable_prefetch_cache=*/false);
  auto http         = run_rest_minio_bench_scenario(*env,
                                            *large,
                                            "async_http",
                                            env->endpoint,
                                            std::nullopt,
                                            false,
                                            /*perf_instrumentation=*/true,
                                            bench_single_column_projection());
  auto https        = run_rest_minio_bench_scenario(*env,
                                             *large,
                                             "async_https",
                                             env->https_endpoint,
                                             env->ca_bundle_path,
                                             true,
                                             /*perf_instrumentation=*/true,
                                             bench_single_column_projection());

  for (auto const& r :
       {std::cref(http), std::cref(https), std::cref(compat_http), std::cref(compat_https)}) {
    auto const& record = r.get();
    CHECK(record.row_count == large->total_num_rows);
    CHECK(record.payload_bytes_read > 0);
    CHECK(record.chunk_get_count > 0);
    CHECK(record.chunk_get_ns_total > 0);
    CHECK(record.chunk_get_ns_max > 0);
    CHECK(record.chunk_get_ns_max <= record.chunk_get_ns_total);
    CHECK(record.queue_wait_count > 0);
    CHECK(record.queue_wait_ns_total > 0);
    CHECK(record.h2d_observed_count > 0);
    CHECK(record.h2d_observed_ns_total > 0);
    CHECK(record.h2d_observed_ns_max > 0);
    CHECK(record.h2d_observed_ns_max <= record.h2d_observed_ns_total);
    CHECK(record.ttfb_ns > 0);
    CHECK(record.device_stream_sync_total == 0);
    CHECK(record.terminal_failures_total == 0);
  }

  CHECK(http.row_count == https.row_count);
  CHECK(http.payload_bytes_read == https.payload_bytes_read);
  CHECK(compat_http.row_count == compat_https.row_count);
  CHECK(compat_http.payload_bytes_read == compat_https.payload_bytes_read);
  if (https.footer_fetch_ms > http.footer_fetch_ms * 3.0) {
    WARN("HTTPS footer_fetch_ms is more than 3x HTTP on this MinIO run: http="
         << http.footer_fetch_ms << "ms https=" << https.footer_fetch_ms << "ms");
  }
  if (compat_https.footer_fetch_ms > compat_http.footer_fetch_ms * 3.0) {
    WARN("HTTPS compat footer_fetch_ms is more than 3x HTTP on this MinIO run: http="
         << compat_http.footer_fetch_ms << "ms https=" << compat_https.footer_fetch_ms << "ms");
  }
  WARN("S3 REST perf async_http throughput=" << http.effective_bytes_per_sec
                                             << " B/s footer_fetch_ms=" << http.footer_fetch_ms
                                             << " chunk_get_ms_mean=" << http.chunk_get_ms_mean);
  WARN("S3 REST perf async_https throughput=" << https.effective_bytes_per_sec
                                              << " B/s footer_fetch_ms=" << https.footer_fetch_ms
                                              << " chunk_get_ms_mean=" << https.chunk_get_ms_mean);
  WARN("S3 REST perf compat_http throughput="
       << compat_http.effective_bytes_per_sec << " B/s footer_fetch_ms="
       << compat_http.footer_fetch_ms << " chunk_get_ms_mean=" << compat_http.chunk_get_ms_mean
       << " chunk_get_count=" << compat_http.chunk_get_count);
  WARN("S3 REST perf compat_https throughput="
       << compat_https.effective_bytes_per_sec << " B/s footer_fetch_ms="
       << compat_https.footer_fetch_ms << " chunk_get_ms_mean=" << compat_https.chunk_get_ms_mean
       << " chunk_get_count=" << compat_https.chunk_get_count);

  attach_baseline_comparison(http, baseline_json);
  attach_baseline_comparison(https, baseline_json);
  attach_baseline_comparison(compat_http, baseline_json);
  attach_baseline_comparison(compat_https, baseline_json);
  records.push_back(std::move(http));
  records.push_back(std::move(https));
  records.push_back(std::move(compat_http));
  records.push_back(std::move(compat_https));
  REQUIRE(records.size() >= 4);

  auto const path = perf_json_path();
  write_perf_json(
    path, *env, static_cast<std::uint64_t>(fs::file_size(large->local_path)), records);
  require_perf_json_schema(path);
  WARN("Wrote S3 REST perf JSON baseline to " << path.string());
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
