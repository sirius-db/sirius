/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/io_context.hpp"
#include "io/rest/authorizer.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/rest/s3/sigv4_authorizer.hpp"
#include "io/s3/sirius_httpfs.hpp"
#include "scan_manager/config.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "utils/s3_container.hpp"
#include "utils/tpch_queries.hpp"
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
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
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

class scoped_env_var {
 public:
  scoped_env_var(std::string name, std::string value) : name_(std::move(name))
  {
    if (auto* current = std::getenv(name_.c_str()); current != nullptr) {
      had_original_ = true;
      original_     = current;
    }
    setenv(name_.c_str(), value.c_str(), 1);
  }

  ~scoped_env_var()
  {
    if (had_original_) {
      setenv(name_.c_str(), original_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  scoped_env_var(scoped_env_var const&)            = delete;
  scoped_env_var& operator=(scoped_env_var const&) = delete;

 private:
  std::string name_;
  std::string original_;
  bool had_original_{false};
};

class scoped_env_vars {
 public:
  explicit scoped_env_vars(std::vector<std::string> names)
  {
    originals_.reserve(names.size());
    for (auto& name : names) {
      std::optional<std::string> value;
      if (auto const* current = std::getenv(name.c_str()); current != nullptr) {
        value = std::string{current};
      }
      originals_.push_back({std::move(name), std::move(value)});
    }
  }

  scoped_env_vars(scoped_env_vars const&)            = delete;
  scoped_env_vars& operator=(scoped_env_vars const&) = delete;

  ~scoped_env_vars()
  {
    for (auto const& [name, value] : originals_) {
      if (value.has_value()) {
        setenv(name.c_str(), value->c_str(), 1);
      } else {
        unsetenv(name.c_str());
      }
    }
  }

  void set(std::string const& name, std::string const& value)
  {
    setenv(name.c_str(), value.c_str(), 1);
  }

  void unset(std::string const& name) { unsetenv(name.c_str()); }

 private:
  std::vector<std::pair<std::string, std::optional<std::string>>> originals_;
};

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

enum class aws_live_env_decision { ready, skip, fail };

struct aws_live_env_result {
  aws_live_env_decision decision{aws_live_env_decision::skip};
  std::optional<s3_test_env> env;
  std::string message;
};

bool is_regional_aws_s3_endpoint(std::string const& endpoint, std::string const& region)
{
  if (region.empty()) { return false; }
  auto const expected = "https://s3." + region + ".amazonaws.com";
  return endpoint == expected;
}

aws_live_env_result classify_aws_live_env()
{
  auto endpoint      = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key    = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key    = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket        = env_or("SIRIUS_TEST_S3_BUCKET");
  auto region        = env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  auto session_token = env_or("SIRIUS_TEST_S3_SESSION_TOKEN");
  auto local_dir =
    env_or("SIRIUS_TEST_S3_LOCAL_DIR",
           (fs::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" / "data").string());
  auto const strict = truthy_env("SIRIUS_TEST_S3_STRICT");

  auto skip_or_fail = [&](std::string message) {
    return aws_live_env_result{strict ? aws_live_env_decision::fail : aws_live_env_decision::skip,
                               std::nullopt,
                               std::move(message)};
  };

  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty()) {
    return skip_or_fail("SIRIUS_TEST_S3_* real-AWS environment is not complete");
  }
  if (session_token.empty()) {
    return skip_or_fail(
      "SIRIUS_TEST_S3_SESSION_TOKEN is required for real-AWS tests; use assume-role temporary "
      "credentials");
  }
  if (!is_regional_aws_s3_endpoint(endpoint, region)) {
    return skip_or_fail(
      "SIRIUS_TEST_S3_ENDPOINT must be regional https://s3.<region>.amazonaws.com");
  }

  return aws_live_env_result{aws_live_env_decision::ready,
                             s3_test_env{std::move(endpoint),
                                         "",
                                         "",
                                         std::move(region),
                                         std::move(access_key),
                                         std::move(secret_key),
                                         std::move(bucket),
                                         std::move(session_token),
                                         fs::path{std::move(local_dir)}},
                             ""};
}

std::optional<s3_test_env> read_aws_live_env()
{
  auto result = classify_aws_live_env();
  if (result.decision == aws_live_env_decision::ready) { return std::move(result.env); }
  if (result.decision == aws_live_env_decision::fail) { FAIL(result.message); }
  SUCCEED(result.message);
  return std::nullopt;
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
  std::optional<sirius::scan_manager::io_backend> backend;
  std::optional<std::string> rest_footer_probe_bytes;
  std::optional<std::size_t> rest_list_max_matches;
  std::optional<std::size_t> rest_list_max_scanned;
};

sirius_memory_limits large_sirius_memory_limits(bool enable_prefetch_cache)
{
  sirius_memory_limits limits;
  limits.gpu_usage             = "5 GiB";
  limits.gpu_reservation       = "2 GiB";
  limits.host_capacity         = "8 GiB";
  limits.disk_capacity         = "32 GiB";
  limits.enable_prefetch_cache = enable_prefetch_cache;
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
      out << "      cache:\n"
             "        mode: "
          << (*limits.enable_prefetch_cache ? "sirius" : "none") << "\n";
    }
    if (limits.backend.has_value()) {
      std::string backend_name;
      REQUIRE(sirius::scan_manager::enum_to_string(*limits.backend, backend_name));
      out << "      backend: " << backend_name << "\n";
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
           "        request_timeout_s: 30\n";
    if (limits.rest_footer_probe_bytes.has_value()) {
      out << "        footer_probe_bytes: " << yaml_quote(*limits.rest_footer_probe_bytes) << "\n";
    }
    if (limits.rest_list_max_matches.has_value()) {
      out << "        list_max_matches: " << *limits.rest_list_max_matches << "\n";
    }
    if (limits.rest_list_max_scanned.has_value()) {
      out << "        list_max_scanned: " << *limits.rest_list_max_scanned << "\n";
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

void query_or_throw_on_error(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  if (!result) { throw std::runtime_error("DuckDB returned a null query result"); }
  if (result->HasError()) { result->ThrowError(); }
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

void set_gpu_execution(duckdb::Connection& con, bool enabled)
{
  auto result = con.Query(std::string{"SET gpu_execution = "} + (enabled ? "true" : "false"));
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
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

struct watchdog_query_result {
  duckdb::idx_t row_count{0};
  duckdb::idx_t column_count{0};
  std::vector<std::string> column_names;
  std::vector<std::vector<std::string>> rows;
  std::string error;
};

watchdog_query_result require_query_ok_with_watchdog(std::shared_ptr<s3_sql_fixture> fixture,
                                                     std::string sql,
                                                     std::chrono::seconds timeout)
{
  struct shared_state {
    std::mutex mtx;
    std::condition_variable cv;
    bool done{false};
    watchdog_query_result result;
  };

  auto state      = std::make_shared<shared_state>();
  auto worker_sql = sql;
  std::thread worker([fixture, sql = std::move(worker_sql), state]() {
    watchdog_query_result out;
    try {
      auto result = fixture->con.Query(sql);
      if (!result) {
        out.error = "query returned nullptr";
      } else if (result->HasError()) {
        out.error = result->GetError();
      } else {
        out.row_count    = result->RowCount();
        out.column_count = result->ColumnCount();
        out.column_names.reserve(result->ColumnCount());
        for (duckdb::idx_t c = 0; c < result->ColumnCount(); ++c) {
          out.column_names.push_back(result->ColumnName(c));
        }
        out.rows = collect_rows(*result);
      }
    } catch (std::exception const& e) {
      out.error = e.what();
    } catch (...) {
      out.error = "query threw an unknown exception";
    }

    {
      std::lock_guard<std::mutex> lock(state->mtx);
      state->result = std::move(out);
      state->done   = true;
    }
    state->cv.notify_one();
  });

  {
    std::unique_lock<std::mutex> lock(state->mtx);
    if (!state->cv.wait_for(lock, timeout, [&] { return state->done; })) {
      worker.detach();
      FAIL("query timed out after " << timeout.count() << " seconds: " << sql);
    }
  }

  worker.join();
  INFO(state->result.error);
  REQUIRE(state->result.error.empty());
  return std::move(state->result);
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

void check_rows_equal_with_tolerant_columns(watchdog_query_result const& actual,
                                            duckdb::MaterializedQueryResult& expected,
                                            std::vector<duckdb::idx_t> const& tolerant_columns = {})
{
  REQUIRE(actual.row_count == expected.RowCount());
  REQUIRE(actual.column_count == expected.ColumnCount());
  for (duckdb::idx_t r = 0; r < actual.row_count; ++r) {
    for (duckdb::idx_t c = 0; c < actual.column_count; ++c) {
      auto const& actual_value  = actual.rows[r][c];
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

std::string lowercase(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

void require_nested_operation_unsupported(s3_sql_fixture& fixture,
                                          std::string const& sql,
                                          std::string_view column_name)
{
  auto result = fixture.con.Query(gpu_execution_sql(sql));
  REQUIRE(result);
  REQUIRE(result->HasError());

  auto const error       = result->GetError();
  auto const lower_error = lowercase(error);
  INFO(error);
  CHECK(lower_error.find("nested column operation") != std::string::npos);
  CHECK((lower_error.find("unsupported") != std::string::npos ||
         lower_error.find("not supported") != std::string::npos));
  CHECK(error.find(std::string{column_name}) != std::string::npos);
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

std::string local_parquet_glob_scan(s3_test_env const& env,
                                    std::string_view pattern,
                                    std::string_view options = {})
{
  return "read_parquet(" + sql_quote((env.local_dir / std::string{pattern}).string()) +
         std::string{options} + ")";
}

std::string s3_parquet_scan(s3_test_env const& env, std::string_view table)
{
  auto const key = "parquet/" + std::string{table} + ".parquet";
  return "read_parquet(" + sql_quote(s3_uri(env.bucket, key)) + ")";
}

std::string s3_parquet_glob_scan(s3_test_env const& env,
                                 std::string_view pattern,
                                 std::string_view options = {})
{
  return "read_parquet(" + sql_quote(s3_uri(env.bucket, pattern)) + std::string{options} + ")";
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

void require_s3_keys_listed(s3_sql_fixture& fixture,
                            s3_test_env const& env,
                            std::vector<std::string_view> const& expected_keys)
{
  auto& rest  = require_rest_ioctx(fixture, s3_uri(env.bucket, "parquet/nation.parquet"));
  auto listed = rest.list_objects(env.bucket, "glob-enc/");
  for (auto const expected : expected_keys) {
    INFO("expected LIST key=" << expected);
    REQUIRE(std::any_of(
      listed.begin(), listed.end(), [&](auto const& entry) { return entry.key == expected; }));
  }
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

void compare_transparent_s3_gpu_to_local_cpu(
  s3_sql_fixture& fixture,
  std::string const& s3_query,
  std::string const& local_query,
  std::vector<duckdb::idx_t> const& tolerant_columns = {})
{
  auto s3_result = require_query_ok(fixture.con, s3_query);

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto local_result = require_query_ok(baseline_con, local_query);

  check_rows_equal_with_tolerant_columns(*s3_result, *local_result, tolerant_columns);
}

void compare_s3_gpu_to_local_cpu_with_watchdog(
  std::shared_ptr<s3_sql_fixture> const& fixture,
  std::string_view label,
  std::string const& s3_query,
  std::string const& local_query,
  std::chrono::seconds timeout,
  std::vector<duckdb::idx_t> const& tolerant_columns = {})
{
  INFO("watchdog case: " << label);
  auto s3_result = require_query_ok_with_watchdog(fixture, gpu_execution_sql(s3_query), timeout);

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto local_result = require_query_ok(baseline_con, local_query);

  if (tolerant_columns.empty()) {
    auto local_rows = collect_rows(*local_result);
    REQUIRE(s3_result.row_count == local_result->RowCount());
    REQUIRE(s3_result.column_count == local_result->ColumnCount());
    CHECK(s3_result.rows == local_rows);
  } else {
    check_rows_equal_with_tolerant_columns(s3_result, *local_result, tolerant_columns);
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
  sirius::io::rest::s3::static_credentials creds;
  creds.access_key_id     = "AKIAFAKEBENCHKEY";
  creds.secret_access_key = "fake-secret-key";
  creds.session_token     = "fake-session-token";

  sirius::io::rest::s3::sigv4_presigned_authorizer authorizer{
    std::move(creds), "us-east-2", "https://s3.us-east-2.amazonaws.com"};
  auto request =
    authorizer.authorize(sirius::io::rest::object_ref{"sirius-bench", "tpch/lineitem_sf10.parquet"},
                         sirius::io::rest::request_method::GET,
                         std::chrono::seconds{60});

  CHECK(request.headers.empty());
  CHECK(request.url.find("X-Amz-Security-Token=") != std::string::npos);
  CHECK(request.url.find("fake-session-token") != std::string::npos);
}

TEST_CASE("real-AWS live env guard requires regional endpoint and temporary credentials",
          "[s3][aws][env]")
{
  scoped_env_vars env{{"SIRIUS_TEST_S3_ENDPOINT",
                       "SIRIUS_TEST_S3_ACCESS_KEY",
                       "SIRIUS_TEST_S3_SECRET_KEY",
                       "SIRIUS_TEST_S3_BUCKET",
                       "SIRIUS_TEST_S3_REGION",
                       "SIRIUS_TEST_S3_SESSION_TOKEN",
                       "SIRIUS_TEST_S3_LOCAL_DIR",
                       "SIRIUS_TEST_S3_STRICT"}};

  auto set_complete_base = [&]() {
    env.set("SIRIUS_TEST_S3_ENDPOINT", "https://s3.us-east-2.amazonaws.com");
    env.set("SIRIUS_TEST_S3_ACCESS_KEY", "AKIAFAKE");
    env.set("SIRIUS_TEST_S3_SECRET_KEY", "fake-secret");
    env.set("SIRIUS_TEST_S3_BUCKET", "sirius-s3-test");
    env.set("SIRIUS_TEST_S3_REGION", "us-east-2");
    env.set("SIRIUS_TEST_S3_LOCAL_DIR",
            (fs::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" / "data").string());
  };

  SECTION("missing session token skips outside strict mode")
  {
    set_complete_base();
    env.unset("SIRIUS_TEST_S3_SESSION_TOKEN");
    env.unset("SIRIUS_TEST_S3_STRICT");
    auto result = classify_aws_live_env();
    CHECK(result.decision == aws_live_env_decision::skip);
    CHECK(result.message.find("SESSION_TOKEN") != std::string::npos);
  }

  SECTION("missing session token fails in strict mode")
  {
    set_complete_base();
    env.unset("SIRIUS_TEST_S3_SESSION_TOKEN");
    env.set("SIRIUS_TEST_S3_STRICT", "1");
    auto result = classify_aws_live_env();
    CHECK(result.decision == aws_live_env_decision::fail);
    CHECK(result.message.find("assume-role") != std::string::npos);
  }

  SECTION("non-regional endpoint skips or fails before any AWS work")
  {
    set_complete_base();
    env.set("SIRIUS_TEST_S3_ENDPOINT", "https://s3.amazonaws.com");
    env.set("SIRIUS_TEST_S3_SESSION_TOKEN", "temporary-token");
    env.unset("SIRIUS_TEST_S3_STRICT");
    auto result = classify_aws_live_env();
    CHECK(result.decision == aws_live_env_decision::skip);
    CHECK(result.message.find("regional") != std::string::npos);

    env.set("SIRIUS_TEST_S3_STRICT", "1");
    result = classify_aws_live_env();
    CHECK(result.decision == aws_live_env_decision::fail);
  }

  SECTION("complete temporary regional env is accepted")
  {
    set_complete_base();
    env.set("SIRIUS_TEST_S3_SESSION_TOKEN", "temporary-token");
    env.unset("SIRIUS_TEST_S3_STRICT");
    auto result = classify_aws_live_env();
    REQUIRE(result.decision == aws_live_env_decision::ready);
    REQUIRE(result.env.has_value());
    CHECK(result.env->endpoint == "https://s3.us-east-2.amazonaws.com");
    CHECK(result.env->session_token == "temporary-token");
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

TEST_CASE("transparent read_parquet over S3 scans through Sirius REST",
          "[s3][integration][sql][gpu_execution][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);

  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto& rest     = require_rest_ioctx(fixture, uri);
  CHECK(rest.type() == sirius::io::io_context_type::restful);

  auto const s3_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                        s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  auto const local_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                           local_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
}

TEST_CASE("transparent read_parquet over S3 keeps REST routing when local Sirius datasource is off",
          "[s3][integration][sql][gpu_execution][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  limits.backend = sirius::scan_manager::io_backend::kvikio;
  s3_sql_fixture fixture(*env, limits);
  set_gpu_execution(fixture.con, true);

  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto& rest     = require_rest_ioctx(fixture, uri);
  CHECK(rest.type() == sirius::io::io_context_type::restful);

  auto result =
    require_query_ok(fixture.con, "SELECT count(*) FROM read_parquet(" + sql_quote(uri) + ")");
  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).GetValue<int64_t>() == 25);
}

TEST_CASE("transparent S3 read_parquet expands globbed parquet files",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);

  auto const before_stats = sirius::test::get_transparent_execution_stats(fixture.con);
  auto const s3_scan      = s3_parquet_glob_scan(*env, "glob/multi/nation_*.parquet");
  auto const local_scan   = local_parquet_glob_scan(*env, "glob/multi/nation_*.parquet");
  auto const s3_query     = "SELECT n_nationkey, n_name, n_regionkey FROM " + s3_scan +
                        " ORDER BY n_nationkey, n_name, n_regionkey";
  auto const local_query = "SELECT n_nationkey, n_name, n_regionkey FROM " + local_scan +
                           " ORDER BY n_nationkey, n_name, n_regionkey";

  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  auto const after_stats = sirius::test::get_transparent_execution_stats(fixture.con);
  sirius::test::require_transparent_execution_delta(before_stats, after_stats, 1, 0, 1);
}

TEST_CASE("transparent S3 glob opens the literal percent key instead of its slash decoy",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/a%2Fb.parquet", "glob-enc/a/b.parquet"});

  auto const s3_scan    = s3_parquet_glob_scan(*env, "glob-enc/a*.parquet");
  auto const local_scan = local_parquet_glob_scan(*env, "glob-enc/a*.parquet");
  auto const s3_query =
    "SELECT n_nationkey, n_name FROM " + s3_scan + " ORDER BY n_nationkey, n_name";
  auto const local_query =
    "SELECT n_nationkey, n_name FROM " + local_scan + " ORDER BY n_nationkey, n_name";

  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  auto result = require_query_ok(fixture.con, "SELECT count(*) FROM " + s3_scan);
  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).GetValue<int64_t>() == 25);
}

TEST_CASE("transparent S3 glob opens keys containing URI fragment and query delimiters",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/x#1.parquet", "glob-enc/y?v.parquet"});

  for (auto const pattern :
       {std::string_view{"glob-enc/x*.parquet"}, std::string_view{"glob-enc/y*.parquet"}}) {
    DYNAMIC_SECTION("pattern=" << pattern)
    {
      auto const s3_query    = "SELECT count(*) FROM " + s3_parquet_glob_scan(*env, pattern);
      auto const local_query = "SELECT count(*) FROM " + local_parquet_glob_scan(*env, pattern);
      compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
    }
  }
}

TEST_CASE("transparent S3 glob opens a key containing a literal percent byte",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/100%.parquet"});

  auto const s3_query =
    "SELECT count(*) FROM " + s3_parquet_glob_scan(*env, "glob-enc/100*.parquet");
  auto const local_query =
    "SELECT count(*) FROM " + local_parquet_glob_scan(*env, "glob-enc/100*.parquet");
  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
}

TEST_CASE("transparent S3 glob decodes a percent-encoded Hive value exactly once",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/t/col=a%20b/p0.parquet"});

  auto const options     = std::string_view{", hive_partitioning=true"};
  auto const s3_scan     = s3_parquet_glob_scan(*env, "glob-enc/t/*/*.parquet", options);
  auto const local_scan  = local_parquet_glob_scan(*env, "glob-enc/t/*/*.parquet", options);
  auto const s3_query    = "SELECT col, count(*) FROM " + s3_scan + " GROUP BY col ORDER BY col";
  auto const local_query = "SELECT col, count(*) FROM " + local_scan + " GROUP BY col ORDER BY col";

  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  auto result = require_query_ok(fixture.con, s3_query);
  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).ToString() == "a b");
  CHECK(result->GetValue(1, 0).GetValue<int64_t>() == 25);
}

TEST_CASE("S3 direct and glob routes share the literal object cache identity",
          "[s3][integration][filesystem][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto& manager          = require_sirius_context(fixture).get_scan_manager();
  auto const direct_uri  = s3_uri(env->bucket, "glob-enc/a%2Fb.parquet");
  auto direct_datasource = manager.create_datasource(direct_uri);
  REQUIRE(direct_datasource != nullptr);

  auto glob_files =
    sirius::io::s3::expand_glob(s3_uri(env->bucket, "glob-enc/a*.parquet"), manager);
  REQUIRE(glob_files.size() == 1);
  auto glob_datasource = manager.create_datasource(glob_files.front().path);
  REQUIRE(glob_datasource != nullptr);

  CHECK(glob_files.front().path == direct_uri);
  CHECK(glob_datasource->get_io_object().object_path() ==
        direct_datasource->get_io_object().object_path());
  CHECK(glob_datasource->get_io_object().raw_file_cache_id() ==
        direct_datasource->get_io_object().raw_file_cache_id());
  CHECK(glob_datasource->get_io_object().size() == direct_datasource->get_io_object().size());
}

TEST_CASE("transparent S3 non-glob reads distinguish literal percent keys from spaces",
          "[s3][integration][sql][gpu_execution][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);

  auto const literal_scan =
    "read_parquet(" + sql_quote(s3_uri(env->bucket, "glob-enc/f%20g.parquet")) + ")";
  auto const literal_local = local_parquet_glob_scan(*env, "glob-enc/f%20g.parquet");
  auto const space_scan =
    "read_parquet(" + sql_quote(s3_uri(env->bucket, "glob-enc/f g.parquet")) + ")";
  auto const space_local = local_parquet_glob_scan(*env, "glob-enc/f g.parquet");

  compare_transparent_s3_gpu_to_local_cpu(fixture,
                                          "SELECT count(*) FROM " + literal_scan + " ORDER BY 1",
                                          "SELECT count(*) FROM " + literal_local + " ORDER BY 1");
  compare_transparent_s3_gpu_to_local_cpu(fixture,
                                          "SELECT count(*) FROM " + space_scan + " ORDER BY 1",
                                          "SELECT count(*) FROM " + space_local + " ORDER BY 1");

  auto literal_result = require_query_ok(fixture.con, "SELECT count(*) FROM " + literal_scan);
  auto space_result   = require_query_ok(fixture.con, "SELECT count(*) FROM " + space_scan);
  CHECK(literal_result->GetValue(0, 0).GetValue<int64_t>() == 25);
  CHECK(space_result->GetValue(0, 0).GetValue<int64_t>() == 5);
}

TEST_CASE("transparent S3 glob rejects a literal question mark in a Hive partition segment",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/q/col=a?b/p0.parquet"});

  auto const scan =
    s3_parquet_glob_scan(*env, "glob-enc/q/col=a?b/*.parquet", ", hive_partitioning=true");
  CHECK_THROWS_WITH(query_or_throw_on_error(fixture.con, "SELECT count(*) FROM " + scan),
                    Catch::Contains("literal '?'"));
}

TEST_CASE("transparent S3 glob rejects a question mark before the Hive partition separator",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/guard-before/co?l=value/p0.parquet"});

  auto const scan =
    s3_parquet_glob_scan(*env, "glob-enc/guard-before/*/*.parquet", ", hive_partitioning=true");
  CHECK_THROWS_WITH(query_or_throw_on_error(fixture.con, "SELECT count(n_nationkey) FROM " + scan),
                    Catch::Contains("literal '?'"));
}

TEST_CASE("transparent S3 glob permits a question mark in the terminal filename",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/guard-filename/report=foo?bar.parquet"});

  auto const options = std::string_view{", hive_partitioning=true"};
  auto const s3_scan =
    s3_parquet_glob_scan(*env, "glob-enc/guard-filename/report*.parquet", options);
  auto const local_scan =
    local_parquet_glob_scan(*env, "glob-enc/guard-filename/report*.parquet", options);
  auto const s3_query    = "SELECT count(n_nationkey) FROM " + s3_scan;
  auto const local_query = "SELECT count(n_nationkey) FROM " + local_scan;

  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  auto result = require_query_ok(fixture.con, s3_query);
  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).GetValue<int64_t>() == 25);
}

TEST_CASE("transparent S3 glob supports an encoded question mark in a Hive partition value",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/q/col=a%3Fb/p0.parquet"});

  auto const options     = std::string_view{", hive_partitioning=true"};
  auto const s3_scan     = s3_parquet_glob_scan(*env, "glob-enc/q/col=a%3F*/*.parquet", options);
  auto const local_scan  = local_parquet_glob_scan(*env, "glob-enc/q/col=a%3F*/*.parquet", options);
  auto const s3_query    = "SELECT col, count(*) FROM " + s3_scan + " GROUP BY col ORDER BY col";
  auto const local_query = "SELECT col, count(*) FROM " + local_scan + " GROUP BY col ORDER BY col";

  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  auto result = require_query_ok(fixture.con, s3_query);
  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).ToString() == "a?b");
  CHECK(result->GetValue(1, 0).GetValue<int64_t>() == 25);
}

TEST_CASE("S3 glob results are sorted by raw literal key bytes",
          "[s3][integration][filesystem][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto& manager = require_sirius_context(fixture).get_scan_manager();
  auto files    = sirius::io::s3::expand_glob(s3_uri(env->bucket, "glob-enc/*.parquet"), manager);

  std::vector<std::string> actual;
  actual.reserve(files.size());
  for (auto const& file : files) {
    actual.push_back(file.path);
  }
  std::vector<std::string> const expected{
    s3_uri(env->bucket, "glob-enc/100%.parquet"),
    s3_uri(env->bucket, "glob-enc/a%2Fb.parquet"),
    s3_uri(env->bucket, "glob-enc/f g.parquet"),
    s3_uri(env->bucket, "glob-enc/f%20g.parquet"),
    s3_uri(env->bucket, "glob-enc/x#1.parquet"),
    s3_uri(env->bucket, "glob-enc/y?v.parquet"),
  };
  CHECK(actual == expected);
}

TEST_CASE("transparent S3 glob ignores percent-encoded keys outside the match",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/a%2Fb.parquet", "glob-enc/y?v.parquet"});

  auto const pattern     = std::string_view{"glob-enc/y*.parquet"};
  auto const s3_query    = "SELECT count(*) FROM " + s3_parquet_glob_scan(*env, pattern);
  auto const local_query = "SELECT count(*) FROM " + local_parquet_glob_scan(*env, pattern);
  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
}

TEST_CASE("transparent S3 glob preserves hive partition columns",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);

  auto const before_stats = sirius::test::get_transparent_execution_stats(fixture.con);
  auto const options      = ", hive_partitioning=true";
  auto const s3_scan      = s3_parquet_glob_scan(*env, "glob/hive/year=*/nation.parquet", options);
  auto const local_scan = local_parquet_glob_scan(*env, "glob/hive/year=*/nation.parquet", options);
  auto const s3_query   = "SELECT year, count(*), min(n_nationkey), max(n_nationkey) FROM " +
                        s3_scan + " GROUP BY year ORDER BY year";
  auto const local_query = "SELECT year, count(*), min(n_nationkey), max(n_nationkey) FROM " +
                           local_scan + " GROUP BY year ORDER BY year";

  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  auto const after_stats = sirius::test::get_transparent_execution_stats(fixture.con);
  sirius::test::require_transparent_execution_delta(before_stats, after_stats, 1, 0, 1);
}

TEST_CASE("transparent S3 glob scans remain correct with parquet footer probes",
          "[s3][integration][sql][gpu_execution][transparent][glob][footerbind]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  s3_sql_fixture fixture(*env, limits);
  set_gpu_execution(fixture.con, true);

  auto const s3_scan     = s3_parquet_glob_scan(*env, "glob/multi/nation_*.parquet");
  auto const local_scan  = local_parquet_glob_scan(*env, "glob/multi/nation_*.parquet");
  auto const s3_query    = "SELECT count(n_nationkey), min(n_name), max(n_name) FROM " + s3_scan;
  auto const local_query = "SELECT count(n_nationkey), min(n_name), max(n_name) FROM " + local_scan;

  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
}

TEST_CASE("transparent S3 glob remains correct with a straddled footer-probe window",
          "[s3][integration][sql][gpu_execution][transparent][glob][footerbind]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  limits.rest_footer_probe_bytes = "512 B";
  s3_sql_fixture fixture(*env, limits);
  set_gpu_execution(fixture.con, true);

  auto const s3_scan     = s3_parquet_glob_scan(*env, "glob/multi/nation_*.parquet");
  auto const local_scan  = local_parquet_glob_scan(*env, "glob/multi/nation_*.parquet");
  auto const s3_query    = "SELECT count(n_nationkey), min(n_name), max(n_name) FROM " + s3_scan;
  auto const local_query = "SELECT count(n_nationkey), min(n_name), max(n_name) FROM " + local_scan;

  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
}

TEST_CASE("transparent S3 glob repeated scans remain correct",
          "[s3][integration][sql][gpu_execution][transparent][glob][footerbind]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  s3_sql_fixture fixture(*env, limits);
  set_gpu_execution(fixture.con, true);

  auto const s3_scan     = s3_parquet_glob_scan(*env, "glob/multi/nation_*.parquet");
  auto const local_scan  = local_parquet_glob_scan(*env, "glob/multi/nation_*.parquet");
  auto const s3_query    = "SELECT count(n_nationkey), min(n_name), max(n_name) FROM " + s3_scan;
  auto const local_query = "SELECT count(n_nationkey), min(n_name), max(n_name) FROM " + local_scan;

  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
}

TEST_CASE("transparent S3 glob matcher semantics match DuckDB segment globs",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);

  auto check_count = [&](std::string_view s3_pattern,
                         std::string_view local_pattern,
                         std::int64_t expected) {
    auto const s3_query    = "SELECT count(*) FROM " + s3_parquet_glob_scan(*env, s3_pattern);
    auto const local_query = "SELECT count(*) FROM " + local_parquet_glob_scan(*env, local_pattern);
    compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
    auto result = require_query_ok(fixture.con, s3_query);
    REQUIRE(result->RowCount() == 1);
    CHECK(result->GetValue(0, 0).GetValue<int64_t>() == expected);
  };

  check_count("glob/multi/nation_?.parquet", "glob/multi/nation_?.parquet", 50);
  check_count("glob/multi/nation_[ab].parquet", "glob/multi/nation_[ab].parquet", 50);
  check_count("glob/hive/**/nation.parquet", "glob/hive/**/nation.parquet", 50);
  check_count("root_*.parquet", "root_*.parquet", 50);

  auto uppercase_root_uri = s3_uri(env->bucket, "root_*.parquet");
  uppercase_root_uri.replace(0, 2, "S3");
  auto const uppercase_root_query =
    "SELECT count(n_nationkey) FROM read_parquet(" + sql_quote(uppercase_root_uri) + ")";
  auto const local_root_query =
    "SELECT count(n_nationkey) FROM " + local_parquet_glob_scan(*env, "root_*.parquet");
  compare_transparent_s3_gpu_to_local_cpu(fixture, uppercase_root_query, local_root_query);
  auto uppercase_root = require_query_ok(fixture.con, uppercase_root_query);
  REQUIRE(uppercase_root->RowCount() == 1);
  CHECK(uppercase_root->GetValue(0, 0).GetValue<int64_t>() == 50);

  auto wildcard_bucket =
    fixture.con.Query("SELECT count(*) FROM read_parquet('s3://*/glob/multi/nation_*.parquet')");
  REQUIRE(wildcard_bucket);
  REQUIRE(wildcard_bucket->HasError());
  INFO(wildcard_bucket->GetError());
  CHECK(wildcard_bucket->GetError().find("bucket") != std::string::npos);
}

TEST_CASE("transparent S3 glob scans 1001 parquet objects across LIST pages",
          "[.][s3][integration][sql][gpu_execution][transparent][glob][large][glob-scale]")
{
  if (!truthy_env("SIRIUS_TEST_S3_GLOB_SCALE")) {
    SUCCEED("SIRIUS_TEST_S3_GLOB_SCALE is not enabled");
    return;
  }

  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);

  auto const query =
    "SELECT count(n_nationkey), sum(n_nationkey), min(n_nationkey), max(n_nationkey) FROM " +
    s3_parquet_glob_scan(*env, "glob-scale/part_*.parquet");
  auto result = require_query_ok(fixture.con, query);

  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).GetValue<int64_t>() == 25'025);
  CHECK(result->GetValue(1, 0).GetValue<int64_t>() == 300'300);
  CHECK(result->GetValue(2, 0).GetValue<int64_t>() == 0);
  CHECK(result->GetValue(3, 0).GetValue<int64_t>() == 24);
}

TEST_CASE("transparent S3 glob reports no-files and GPU-only errors clearly",
          "[s3][integration][sql][gpu_execution][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  SECTION("zero-match glob reports a no-files class error")
  {
    s3_sql_fixture fixture(*env);
    set_gpu_execution(fixture.con, true);

    auto result = fixture.con.Query("SELECT count(*) FROM " +
                                    s3_parquet_glob_scan(*env, "glob/multi/no-match-*.parquet"));
    REQUIRE(result);
    REQUIRE(result->HasError());
    auto const error = result->GetError();
    INFO(error);
    CHECK(
      (error.find("No files") != std::string::npos || error.find("no files") != std::string::npos));
    CHECK(error.find("glob/wildcard patterns are not supported") == std::string::npos);
    CHECK(error.find("No filesystem") == std::string::npos);
    CHECK(error.find("no filesystem") == std::string::npos);
  }

  SECTION("gpu_execution=false rejects globbed S3 at the filesystem gate")
  {
    s3_sql_fixture fixture(*env);
    set_gpu_execution(fixture.con, false);

    auto result = fixture.con.Query("SELECT count(*) FROM " +
                                    s3_parquet_glob_scan(*env, "glob/multi/nation_*.parquet"));
    REQUIRE(result);
    REQUIRE(result->HasError());
    auto const error = result->GetError();
    INFO(error);
    CHECK(error.find("S3 is GPU-only") != std::string::npos);
    CHECK(error.find("SET gpu_execution=true") != std::string::npos);
  }

  SECTION("configured glob match cap rejects overly broad matches")
  {
    sirius_memory_limits limits;
    limits.rest_list_max_matches = 1;
    s3_sql_fixture fixture(*env, limits);
    set_gpu_execution(fixture.con, true);

    auto result = fixture.con.Query("SELECT count(n_nationkey) FROM " +
                                    s3_parquet_glob_scan(*env, "glob/multi/nation_*.parquet"));
    REQUIRE(result);
    REQUIRE(result->HasError());
    auto const error = result->GetError();
    INFO(error);
    CHECK(error.find("narrow the glob prefix") != std::string::npos);
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

TEST_CASE("S3 pushdown all-pruned filter completes with an empty result",
          "[.][s3][pushdown][deadlock]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto fixture        = std::make_shared<s3_sql_fixture>(*env);
  auto const s3_query = "SELECT n_nationkey FROM " + s3_parquet_scan(*env, "nation") +
                        " WHERE n_regionkey = 99 ORDER BY n_nationkey";

  auto result =
    require_query_ok_with_watchdog(fixture, gpu_execution_sql(s3_query), std::chrono::seconds{120});
  CHECK(result.row_count == 0);
  REQUIRE(result.column_count == 1);
  REQUIRE(result.column_names.size() == 1);
  CHECK(result.column_names[0] == "n_nationkey");
  CHECK(result.rows.empty());
}

TEST_CASE("S3 pushdown all-pruned aggregate returns zero rows counted",
          "[.][s3][pushdown][deadlock][agg-identity]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto fixture = std::make_shared<s3_sql_fixture>(*env);
  auto const s3_query =
    "SELECT count(*) AS c FROM " + s3_parquet_scan(*env, "nation") + " WHERE n_regionkey = 99";

  auto result =
    require_query_ok_with_watchdog(fixture, gpu_execution_sql(s3_query), std::chrono::seconds{120});
  REQUIRE(result.row_count == 1);
  REQUIRE(result.column_count == 1);
  REQUIRE(result.rows.size() == 1);
  REQUIRE(result.rows[0].size() == 1);
  CHECK(result.rows[0][0] == "0");
}

TEST_CASE("S3 pushdown zero-input ungrouped count emits the aggregate identity row",
          "[.][s3][pushdown][agg-identity]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto fixture = std::make_shared<s3_sql_fixture>(*env);
  auto const s3_query =
    "SELECT count(*) AS c FROM " + s3_parquet_scan(*env, "nation") + " WHERE n_regionkey = 99";

  auto result =
    require_query_ok_with_watchdog(fixture, gpu_execution_sql(s3_query), std::chrono::seconds{120});
  REQUIRE(result.row_count == 1);
  REQUIRE(result.column_count == 1);
  REQUIRE(result.rows.size() == 1);
  REQUIRE(result.rows[0].size() == 1);
  CHECK(result.rows[0][0] == "0");
}

TEST_CASE("S3 pushdown zero-input ungrouped aggregates emit SQL identity and null values",
          "[.][s3][pushdown][agg-identity]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto fixture = std::make_shared<s3_sql_fixture>(*env);
  auto const s3_query =
    "SELECT count(*) AS c_all, count(n_name) AS c_name, sum(n_nationkey) AS sum_key, "
    "min(n_name) AS min_name, max(n_name) AS max_name, avg(n_nationkey) AS avg_key, "
    "first(n_name) AS first_name FROM " +
    s3_parquet_scan(*env, "nation") + " WHERE n_regionkey = 99";

  auto result =
    require_query_ok_with_watchdog(fixture, gpu_execution_sql(s3_query), std::chrono::seconds{120});
  REQUIRE(result.row_count == 1);
  REQUIRE(result.column_count == 7);
  REQUIRE(result.rows.size() == 1);
  REQUIRE(result.rows[0].size() == 7);
  CHECK(result.rows[0] ==
        std::vector<std::string>{"0", "0", "NULL", "NULL", "NULL", "NULL", "NULL"});
}

TEST_CASE("S3 pushdown zero-input grouped aggregate still emits no groups",
          "[.][s3][pushdown][agg-identity]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto fixture        = std::make_shared<s3_sql_fixture>(*env);
  auto const s3_query = "SELECT n_regionkey, count(*) AS c FROM " +
                        s3_parquet_scan(*env, "nation") +
                        " WHERE n_regionkey = 99 GROUP BY n_regionkey ORDER BY n_regionkey";

  auto result =
    require_query_ok_with_watchdog(fixture, gpu_execution_sql(s3_query), std::chrono::seconds{120});
  CHECK(result.row_count == 0);
  REQUIRE(result.column_count == 2);
  CHECK(result.rows.empty());
}

TEST_CASE("S3 pushdown non-pruned aggregate still matches the local parquet oracle",
          "[.][s3][pushdown][agg-identity]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto fixture = std::make_shared<s3_sql_fixture>(*env);
  auto const s3_query =
    "SELECT count(*) AS c, min(o_orderdate) AS min_date, max(o_orderdate) AS max_date FROM " +
    s3_parquet_scan(*env, "orders") + " WHERE o_orderdate >= DATE '1994-01-01'";
  auto const local_query =
    "SELECT count(*) AS c, min(o_orderdate) AS min_date, max(o_orderdate) AS max_date FROM " +
    local_parquet_scan(*env, "orders") + " WHERE o_orderdate >= DATE '1994-01-01'";

  auto s3_result =
    require_query_ok_with_watchdog(fixture, gpu_execution_sql(s3_query), std::chrono::seconds{120});

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto local_result = require_query_ok(baseline_con, local_query);
  auto local_rows   = collect_rows(*local_result);

  REQUIRE(s3_result.row_count == 1);
  REQUIRE(s3_result.column_count == local_result->ColumnCount());
  CHECK(s3_result.rows == local_rows);
}

TEST_CASE("S3 pushdown selective filters still match the local parquet oracle",
          "[.][s3][pushdown][deadlock]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  // A partially pruned scan must still avoid premature completion and match the oracle.
  auto fixture     = std::make_shared<s3_sql_fixture>(*env);
  auto const shape = std::string{
    "SELECT l_returnflag, count(*) AS c FROM %s "
    "WHERE l_shipdate BETWEEN DATE '1996-01-01' AND DATE '1996-06-30' "
    "GROUP BY l_returnflag ORDER BY l_returnflag"};
  auto const s3_query = [&] {
    auto query = shape;
    auto scan  = s3_parquet_scan(*env, "lineitem");
    query.replace(query.find("%s"), 2, scan);
    return query;
  }();
  auto const local_query = [&] {
    auto query = shape;
    auto scan  = local_parquet_scan(*env, "lineitem");
    query.replace(query.find("%s"), 2, scan);
    return query;
  }();

  auto s3_result =
    require_query_ok_with_watchdog(fixture, gpu_execution_sql(s3_query), std::chrono::seconds{120});

  duckdb::DuckDB baseline_db(nullptr);
  duckdb::Connection baseline_con(baseline_db);
  auto local_result = require_query_ok(baseline_con, local_query);
  auto local_rows   = collect_rows(*local_result);

  CHECK_FALSE(s3_result.rows.empty());
  CHECK(s3_result.rows == local_rows);
}

TEST_CASE("S3 pushdown shape-C zero-side joins match the local parquet oracle",
          "[.][s3][integration][pushdown][shape-c]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto fixture = std::make_shared<s3_sql_fixture>(*env);

  auto make_scans = [&](bool use_s3) {
    struct scans {
      std::string nation;
      std::string region;
      std::string pruned_nation;
      std::string pruned_region;
    };

    auto nation = use_s3 ? s3_parquet_scan(*env, "nation") : local_parquet_scan(*env, "nation");
    auto region = use_s3 ? s3_parquet_scan(*env, "region") : local_parquet_scan(*env, "region");
    return scans{
      nation,
      region,
      "(SELECT * FROM " + nation + " WHERE n_regionkey = 99)",
      "(SELECT * FROM " + region + " WHERE r_regionkey = 99)",
    };
  };

  struct query_case {
    std::string_view label;
    std::string sql;
  };

  auto compare_cases = [&](std::string_view matrix_label,
                           std::vector<query_case> const& s3_queries,
                           std::vector<query_case> const& local_queries) {
    INFO("shape-c matrix: " << matrix_label);
    REQUIRE(s3_queries.size() == local_queries.size());
    for (std::size_t i = 0; i < s3_queries.size(); ++i) {
      REQUIRE(s3_queries[i].label == local_queries[i].label);
      compare_s3_gpu_to_local_cpu_with_watchdog(fixture,
                                                s3_queries[i].label,
                                                s3_queries[i].sql,
                                                local_queries[i].sql,
                                                std::chrono::seconds{120});
    }
  };

  auto select_cases = [](std::vector<query_case> const& queries,
                         std::vector<std::string_view> const& labels) {
    std::vector<query_case> selected;
    selected.reserve(labels.size());
    for (auto const label : labels) {
      auto iter = std::find_if(queries.begin(), queries.end(), [&](query_case const& candidate) {
        return candidate.label == label;
      });
      REQUIRE(iter != queries.end());
      selected.push_back(*iter);
    }
    return selected;
  };

  auto compare_selected_cases = [&](std::string_view matrix_label,
                                    std::vector<query_case> const& s3_queries,
                                    std::vector<query_case> const& local_queries,
                                    std::vector<std::string_view> const& labels) {
    compare_cases(
      matrix_label, select_cases(s3_queries, labels), select_cases(local_queries, labels));
  };

  auto build_zero_side_queries = [](auto const& s) {
    return std::vector<query_case>{
      {"hash inner dead left",
       "SELECT n.n_nationkey, r.r_name FROM " + s.pruned_nation + " n INNER JOIN " + s.region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"hash inner dead right",
       "SELECT n.n_nationkey, r.r_name FROM " + s.nation + " n INNER JOIN " + s.pruned_region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"hash left dead left",
       "SELECT n.n_nationkey, r.r_name FROM " + s.pruned_nation + " n LEFT JOIN " + s.region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"hash left dead right",
       "SELECT n.n_nationkey, r.r_name FROM " + s.nation + " n LEFT JOIN " + s.pruned_region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"hash right dead left",
       "SELECT n.n_nationkey, r.r_name FROM " + s.pruned_nation + " n RIGHT JOIN " + s.region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"hash right dead right",
       "SELECT n.n_nationkey, r.r_name FROM " + s.nation + " n RIGHT JOIN " + s.pruned_region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"hash full outer dead left",
       "SELECT n.n_nationkey, r.r_name FROM " + s.pruned_nation + " n FULL OUTER JOIN " + s.region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"hash full outer dead right",
       "SELECT n.n_nationkey, r.r_name FROM " + s.nation + " n FULL OUTER JOIN " + s.pruned_region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"hash not exists dead inner",
       "SELECT n.n_nationkey FROM " + s.nation + " n WHERE NOT EXISTS (SELECT 1 FROM " +
         s.pruned_region + " r WHERE r.r_regionkey = n.n_regionkey) ORDER BY n.n_nationkey"},
      {"hash in mark dead inner",
       "SELECT n.n_nationkey, n.n_regionkey IN (SELECT r_regionkey FROM " + s.pruned_region +
         ") AS in_pruned FROM " + s.nation + " n ORDER BY n.n_nationkey"},
      {"hash exists dead inner",
       "SELECT n.n_nationkey FROM " + s.nation + " n WHERE EXISTS (SELECT 1 FROM " +
         s.pruned_region + " r WHERE r.r_regionkey = n.n_regionkey) ORDER BY n.n_nationkey"},
      {"hash count over zero-side join",
       "SELECT count(*) FROM " + s.pruned_nation + " n INNER JOIN " + s.region +
         " r ON n.n_regionkey = r.r_regionkey"},
      {"hash both sides pruned",
       "SELECT n.n_nationkey, r.r_name FROM " + s.pruned_nation + " n INNER JOIN " +
         s.pruned_region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"nlj left dead right",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.nation + " n LEFT JOIN " + s.pruned_region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"nlj right dead left",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.pruned_nation + " n RIGHT JOIN " + s.region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"nlj full outer dead left",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.pruned_nation + " n FULL OUTER JOIN " +
         s.region + " r ON n.n_regionkey < r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"nlj full outer dead right",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.nation + " n FULL OUTER JOIN " +
         s.pruned_region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"nlj anti dead right",
       "SELECT n.n_nationkey FROM " + s.nation + " n ANTI JOIN " + s.pruned_region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY n.n_nationkey"},
      {"nlj inner dead left",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.pruned_nation + " n INNER JOIN " + s.region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"nlj inner dead right",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.nation + " n INNER JOIN " + s.pruned_region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"nlj mark both alive",
       "SELECT n.n_nationkey, n.n_regionkey < ANY (SELECT r_regionkey FROM " + s.region +
         ") AS lt_any_region FROM " + s.nation + " n ORDER BY n.n_nationkey"},
      {"nlj mark dead right",
       "SELECT n.n_nationkey, n.n_regionkey < ANY (SELECT r_regionkey FROM " + s.pruned_region +
         ") AS lt_any_region FROM " + s.nation + " n ORDER BY n.n_nationkey"},
      {"nlj mark dead left",
       "SELECT n.n_nationkey, n.n_regionkey < ANY (SELECT r_regionkey FROM " + s.region +
         ") AS lt_any_region FROM " + s.pruned_nation + " n ORDER BY n.n_nationkey"},
    };
  };

  auto build_empty_batch_queries = [](auto const& s) {
    return std::vector<query_case>{
      {"hash fallback left dead right",
       "SELECT n.n_nationkey, r.r_name FROM " + s.nation + " n LEFT JOIN " + s.pruned_region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"hash fallback right dead left",
       "SELECT n.n_nationkey, r.r_name FROM " + s.pruned_nation + " n RIGHT JOIN " + s.region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"hash fallback full outer dead left",
       "SELECT n.n_nationkey, r.r_name FROM " + s.pruned_nation + " n FULL OUTER JOIN " + s.region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"hash fallback full outer dead right",
       "SELECT n.n_nationkey, r.r_name FROM " + s.nation + " n FULL OUTER JOIN " + s.pruned_region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"hash fallback anti dead right",
       "SELECT n.n_nationkey FROM " + s.nation + " n ANTI JOIN " + s.pruned_region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY n.n_nationkey"},
      {"nlj fallback left dead right",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.nation + " n LEFT JOIN " + s.pruned_region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"nlj fallback right dead left",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.pruned_nation + " n RIGHT JOIN " + s.region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"nlj fallback full outer dead left",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.pruned_nation + " n FULL OUTER JOIN " +
         s.region + " r ON n.n_regionkey < r.r_regionkey ORDER BY r.r_regionkey, n.n_nationkey"},
      {"nlj fallback full outer dead right",
       "SELECT n.n_nationkey, r.r_regionkey FROM " + s.nation + " n FULL OUTER JOIN " +
         s.pruned_region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY n.n_nationkey, r.r_regionkey"},
      {"nlj fallback anti dead right",
       "SELECT n.n_nationkey FROM " + s.nation + " n ANTI JOIN " + s.pruned_region +
         " r ON n.n_regionkey < r.r_regionkey ORDER BY n.n_nationkey"},
      {"nlj fallback mark dead right",
       "SELECT n.n_nationkey, n.n_regionkey < ANY (SELECT r_regionkey FROM " + s.pruned_region +
         ") AS lt_any_region FROM " + s.nation + " n ORDER BY n.n_nationkey"},
    };
  };

  auto const empty_batch_s3_queries    = build_empty_batch_queries(make_scans(/*use_s3=*/true));
  auto const empty_batch_local_queries = build_empty_batch_queries(make_scans(/*use_s3=*/false));
  auto const zero_side_s3_queries      = build_zero_side_queries(make_scans(/*use_s3=*/true));
  auto const zero_side_local_queries   = build_zero_side_queries(make_scans(/*use_s3=*/false));

  SECTION("hash fallback empty-batch pins")
  {
    compare_selected_cases("hash fallback empty-batch pins",
                           empty_batch_s3_queries,
                           empty_batch_local_queries,
                           {
                             "hash fallback left dead right",
                             "hash fallback right dead left",
                             "hash fallback full outer dead left",
                             "hash fallback full outer dead right",
                             "hash fallback anti dead right",
                           });
  }

  SECTION("all-pruned-side hash and non-MARK NLJ joins")
  {
    compare_selected_cases(
      "all-pruned-side hash and non-MARK NLJ joins",
      zero_side_s3_queries,
      zero_side_local_queries,
      {
        "hash inner dead left",      "hash inner dead right",      "hash left dead left",
        "hash left dead right",      "hash right dead left",       "hash right dead right",
        "hash full outer dead left", "hash full outer dead right", "hash not exists dead inner",
        "hash in mark dead inner",   "hash exists dead inner",     "hash count over zero-side join",
        "hash both sides pruned",    "nlj left dead right",        "nlj right dead left",
        "nlj full outer dead left",  "nlj full outer dead right",  "nlj anti dead right",
        "nlj inner dead left",       "nlj inner dead right",
      });
  }

  SECTION("AC-R1 NLJ fallback empty-batch cells")
  {
    compare_selected_cases("AC-R1 NLJ fallback empty-batch cells",
                           empty_batch_s3_queries,
                           empty_batch_local_queries,
                           {
                             "nlj fallback left dead right",
                             "nlj fallback right dead left",
                             "nlj fallback full outer dead left",
                             "nlj fallback full outer dead right",
                             "nlj fallback anti dead right",
                           });
  }

  SECTION("AC-R2 MARK NLJ both sides alive")
  {
    compare_selected_cases("AC-R2 MARK NLJ both sides alive",
                           zero_side_s3_queries,
                           zero_side_local_queries,
                           {"nlj mark both alive"});
  }

  SECTION("AC-R2 MARK NLJ dead build side")
  {
    compare_selected_cases("AC-R2 MARK NLJ dead build side",
                           zero_side_s3_queries,
                           zero_side_local_queries,
                           {"nlj mark dead right"});
  }

  SECTION("AC-R2 MARK NLJ dead probe side")
  {
    compare_selected_cases("AC-R2 MARK NLJ dead probe side",
                           zero_side_s3_queries,
                           zero_side_local_queries,
                           {"nlj mark dead left"});
  }

  SECTION("AC-R2 MARK NLJ fallback dead build side")
  {
    compare_selected_cases("AC-R2 MARK NLJ fallback dead build side",
                           empty_batch_s3_queries,
                           empty_batch_local_queries,
                           {"nlj fallback mark dead right"});
  }
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

TEST_CASE("S3 read_parquet op shape T1 matches outer join semantics",
          "[s3][integration][sql][t1-outer-join]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const s3_region      = s3_parquet_scan(*env, "region");
  auto const s3_nation      = s3_parquet_scan(*env, "nation");
  auto const s3_supplier    = s3_parquet_scan(*env, "supplier");
  auto const local_region   = local_parquet_scan(*env, "region");
  auto const local_nation   = local_parquet_scan(*env, "nation");
  auto const local_supplier = local_parquet_scan(*env, "supplier");

  compare_s3_gpu_to_local_cpu(
    fixture,
    "SELECT r.r_regionkey, n.n_nationkey FROM " + s3_region + " r LEFT JOIN " + s3_nation +
      " n ON n.n_regionkey = r.r_regionkey AND n.n_nationkey > 100 "
      "ORDER BY r.r_regionkey, n.n_nationkey",
    "SELECT r.r_regionkey, n.n_nationkey FROM " + local_region + " r LEFT JOIN " + local_nation +
      " n ON n.n_regionkey = r.r_regionkey AND n.n_nationkey > 100 "
      "ORDER BY r.r_regionkey, n.n_nationkey");

  compare_s3_gpu_to_local_cpu(
    fixture,
    "SELECT r.r_regionkey, n.n_nationkey FROM " + s3_nation + " n RIGHT JOIN " + s3_region +
      " r ON n.n_regionkey = r.r_regionkey AND n.n_nationkey > 100 "
      "ORDER BY r.r_regionkey, n.n_nationkey",
    "SELECT r.r_regionkey, n.n_nationkey FROM " + local_nation + " n RIGHT JOIN " + local_region +
      " r ON n.n_regionkey = r.r_regionkey AND n.n_nationkey > 100 "
      "ORDER BY r.r_regionkey, n.n_nationkey");

  compare_s3_gpu_to_local_cpu(fixture,
                              "SELECT n.n_nationkey, count(s.s_suppkey) AS supplier_count FROM " +
                                s3_nation + " n LEFT JOIN " + s3_supplier +
                                " s ON s.s_nationkey = n.n_nationkey "
                                "GROUP BY n.n_nationkey ORDER BY n.n_nationkey",
                              "SELECT n.n_nationkey, count(s.s_suppkey) AS supplier_count FROM " +
                                local_nation + " n LEFT JOIN " + local_supplier +
                                " s ON s.s_nationkey = n.n_nationkey "
                                "GROUP BY n.n_nationkey ORDER BY n.n_nationkey");
}

TEST_CASE("S3 read_parquet op shape T2 matches grouped distinct aggregates",
          "[s3][integration][sql][t2-groupby]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto const grouped_query = [](std::string const& scan) {
    return "SELECT l_returnflag, l_linestatus, count(*) AS item_count, "
           "count(DISTINCT l_orderkey) AS distinct_orders, sum(l_quantity) AS quantity "
           "FROM " +
           scan +
           " GROUP BY l_returnflag, l_linestatus HAVING count(*) > 100 "
           "ORDER BY l_returnflag, l_linestatus";
  };

  s3_sql_fixture fixture(*env);
  compare_s3_gpu_to_local_cpu(fixture,
                              grouped_query(s3_parquet_scan(*env, "lineitem")),
                              grouped_query(local_parquet_scan(*env, "lineitem")),
                              {4});
}

TEST_CASE("S3 read_parquet op shape T3 matches string predicates",
          "[s3][integration][sql][t3-string]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const s3_part      = s3_parquet_scan(*env, "part");
  auto const s3_nation    = s3_parquet_scan(*env, "nation");
  auto const local_part   = local_parquet_scan(*env, "part");
  auto const local_nation = local_parquet_scan(*env, "nation");

  compare_s3_gpu_to_local_cpu(fixture,
                              "SELECT count(*) AS green_count FROM " + s3_part +
                                " WHERE p_name LIKE '%green%' ORDER BY green_count",
                              "SELECT count(*) AS green_count FROM " + local_part +
                                " WHERE p_name LIKE '%green%' ORDER BY green_count");

  compare_s3_gpu_to_local_cpu(fixture,
                              "SELECT p_partkey, p_name FROM " + s3_part +
                                " WHERE p_name LIKE 'forest%' ORDER BY p_partkey LIMIT 100",
                              "SELECT p_partkey, p_name FROM " + local_part +
                                " WHERE p_name LIKE 'forest%' ORDER BY p_partkey LIMIT 100");

  compare_s3_gpu_to_local_cpu(
    fixture,
    "SELECT n_nationkey, n_name FROM " + s3_nation +
      " WHERE n_name IN ('FRANCE', 'GERMANY', 'BRAZIL') ORDER BY n_nationkey",
    "SELECT n_nationkey, n_name FROM " + local_nation +
      " WHERE n_name IN ('FRANCE', 'GERMANY', 'BRAZIL') ORDER BY n_nationkey");
}

TEST_CASE("S3 read_parquet op shape T4 mixes local and S3 scans",
          "[s3][integration][sql][t4-mixed]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto const local_nation   = local_parquet_scan(*env, "nation");
  auto const local_supplier = local_parquet_scan(*env, "supplier");
  auto const s3_supplier    = s3_parquet_scan(*env, "supplier");

  s3_sql_fixture fixture(*env);
  compare_s3_gpu_to_local_cpu(
    fixture,
    "SELECT n.n_name, s.s_name FROM " + local_nation + " n JOIN " + s3_supplier +
      " s ON s.s_nationkey = n.n_nationkey "
      "WHERE n.n_regionkey = 1 ORDER BY s.s_suppkey LIMIT 50",
    "SELECT n.n_name, s.s_name FROM " + local_nation + " n JOIN " + local_supplier +
      " s ON s.s_nationkey = n.n_nationkey "
      "WHERE n.n_regionkey = 1 ORDER BY s.s_suppkey LIMIT 50");
}

TEST_CASE("S3 read_parquet op shape T5 preserves null decimal and timestamp values",
          "[s3][integration][sql][t5-edge-types]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto const edge_query = [](std::string const& scan) {
    return "SELECT id, n, d, ts FROM " + scan +
           " WHERE n IS NULL OR d < CAST(0 AS DECIMAL(18,4)) ORDER BY id";
  };
  auto const aggregate_query = [](std::string const& scan) {
    return "SELECT count(*) AS total, count(n) AS non_null_n, sum(d) AS sum_d, "
           "min(ts) AS min_ts, max(ts) AS max_ts FROM " +
           scan + " ORDER BY total, non_null_n, sum_d, min_ts, max_ts";
  };

  s3_sql_fixture fixture(*env);
  compare_s3_gpu_to_local_cpu(fixture,
                              edge_query(s3_parquet_scan(*env, "edge_types")),
                              edge_query(local_parquet_scan(*env, "edge_types")),
                              {2});
  compare_s3_gpu_to_local_cpu(fixture,
                              aggregate_query(s3_parquet_scan(*env, "edge_types")),
                              aggregate_query(local_parquet_scan(*env, "edge_types")),
                              {2});
}

TEST_CASE("S3 read_parquet op shape T6 matches CTE and scalar subquery results",
          "[s3][integration][sql][t6-cte]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto const cte_query = [](std::string const& lineitem_scan, std::string const& orders_scan) {
    return "WITH revenue AS ("
           "SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) AS revenue "
           "FROM " +
           lineitem_scan +
           " GROUP BY l_orderkey) "
           "SELECT o.o_orderkey, revenue.revenue FROM " +
           orders_scan +
           " o JOIN revenue ON revenue.l_orderkey = o.o_orderkey "
           "WHERE o.o_orderstatus = 'O' "
           "ORDER BY revenue.revenue DESC, o.o_orderkey LIMIT 200";
  };
  auto const scalar_query = [](std::string const& orders_scan) {
    return "SELECT o_orderkey FROM " + orders_scan +
           " WHERE o_totalprice > (SELECT avg(o_totalprice) FROM " + orders_scan +
           ") ORDER BY o_orderkey LIMIT 100";
  };

  auto fixture = std::make_shared<s3_sql_fixture>(*env);
  compare_s3_gpu_to_local_cpu_with_watchdog(
    fixture,
    "T6 CTE aggregate join",
    cte_query(s3_parquet_scan(*env, "lineitem"), s3_parquet_scan(*env, "orders")),
    cte_query(local_parquet_scan(*env, "lineitem"), local_parquet_scan(*env, "orders")),
    std::chrono::seconds{120},
    {1});
  compare_s3_gpu_to_local_cpu_with_watchdog(fixture,
                                            "T6 scalar subquery",
                                            scalar_query(s3_parquet_scan(*env, "orders")),
                                            scalar_query(local_parquet_scan(*env, "orders")),
                                            std::chrono::seconds{120});
}

TEST_CASE("gpu_execution S3 nested parquet projections match local DuckDB CPU",
          "[.][s3][integration][sql][gpu_execution][nested]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  struct nested_projection_case {
    std::string_view table;
    std::string_view select_list;
  };

  constexpr std::array<nested_projection_case, 5> cases{{
    {"nested_struct", "id, payload"},
    {"nested_list", "id, items"},
    {"nested_map", "id, attrs"},
    {"nested_deep", "id, struct_of_list, list_of_struct, tail"},
    {"nested_chunk_boundary", "id, items, payload, attrs, tail"},
  }};

  s3_sql_fixture fixture(*env);
  for (auto const& test_case : cases) {
    auto const s3_query = "SELECT " + std::string{test_case.select_list} + " FROM " +
                          s3_parquet_scan(*env, test_case.table) + " ORDER BY id";
    auto const local_query = "SELECT " + std::string{test_case.select_list} + " FROM " +
                             local_parquet_scan(*env, test_case.table) + " ORDER BY id";

    INFO("table=" << test_case.table);
    compare_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  }
}

TEST_CASE("gpu_execution rejects operations on nested S3 parquet columns cleanly",
          "[.][s3][integration][sql][gpu_execution][nested][unsupported]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);

  auto const struct_scan = s3_parquet_scan(*env, "nested_struct");
  require_nested_operation_unsupported(
    fixture, "SELECT id FROM " + struct_scan + " WHERE payload IS NULL", "payload");

  require_nested_operation_unsupported(
    fixture,
    "SELECT id FROM " + struct_scan + " WHERE payload = struct_pack(a := 10, b := 'alpha')",
    "payload");

  auto const list_scan = s3_parquet_scan(*env, "nested_list");
  require_nested_operation_unsupported(
    fixture, "SELECT id FROM " + list_scan + " WHERE items IS NULL", "items");

  require_nested_operation_unsupported(
    fixture, "SELECT items, count(*) FROM " + list_scan + " GROUP BY items", "items");

  require_nested_operation_unsupported(
    fixture,
    "SELECT l.id FROM " + list_scan + " l JOIN " + list_scan + " r ON l.items = r.items",
    "items");

  auto const map_scan = s3_parquet_scan(*env, "nested_map");
  require_nested_operation_unsupported(
    fixture, "SELECT id FROM " + map_scan + " WHERE attrs IS NULL", "attrs");
}

TEST_CASE("transparent S3 window query reports unsupported S3 CPU fallback",
          "[s3][integration][sql][gpu_execution][fallback][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  auto const s3_query = "SELECT n_nationkey, ROW_NUMBER() OVER (ORDER BY n_nationkey) AS rn FROM " +
                        s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  auto result = fixture.con.Query(s3_query);
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

TEST_CASE("transparent S3 projection fallback reports prose instead of an exception envelope",
          "[s3][integration][sql][gpu_execution][fallback][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  auto const query =
    "SELECT abs(n_nationkey - 100) FROM " + s3_parquet_scan(*env, "nation") + " LIMIT 1";
  auto result = fixture.con.Query(query);
  REQUIRE(result);
  REQUIRE(result->HasError());

  auto const error = result->GetError();
  INFO(error);
  CHECK(error.find("S3 CPU fallback is not supported") != std::string::npos);
  CHECK(error.find("Underlying GPU error: Unsupported expression in projection") !=
        std::string::npos);
  CHECK(error.find("\"exception_type\"") == std::string::npos);
  CHECK(error.find("\"exception_message\"") == std::string::npos);
}

TEST_CASE("transparent S3 view fallback is rejected instead of replaying on CPU",
          "[s3][integration][sql][gpu_execution][fallback][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  REQUIRE_FALSE(fixture.con
                  .Query("CREATE VIEW v_s3_nation AS "
                         "SELECT n_nationkey FROM " +
                         s3_parquet_scan(*env, "nation"))
                  ->HasError());

  auto result = fixture.con.Query(
    "SELECT n_nationkey, ROW_NUMBER() OVER (ORDER BY n_nationkey) AS rn "
    "FROM v_s3_nation ORDER BY n_nationkey");
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

TEST_CASE("S3 read_parquet is rejected when transparent GPU execution is disabled",
          "[s3][integration][sql][gpu_execution][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, false);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto result    = fixture.con.Query("SELECT count(*) FROM read_parquet('" + uri + "')");
  REQUIRE(result);
  REQUIRE(result->HasError());
  auto const error = result->GetError();
  INFO(error);
  CHECK(error.find("S3 is GPU-only") != std::string::npos);
  CHECK(error.find("SET gpu_execution=true") != std::string::npos);
  CHECK(error.find("No filesystem") == std::string::npos);
  CHECK(error.find("no filesystem") == std::string::npos);
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

TEST_CASE("gpu_execution reads real AWS S3 parquet through Sirius SigV4",
          "[.][s3][aws][live][sql][gpu_execution]")
{
  auto env = read_aws_live_env();
  if (!env) { return; }

  auto const s3_query = std::string{"SELECT n_nationkey, n_name, n_regionkey FROM read_parquet("} +
                        sql_quote(s3_uri(env->bucket, "fixtures/nation.parquet")) +
                        ") ORDER BY n_nationkey";
  auto const local_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                           local_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";

  SECTION("presigned")
  {
    s3_sql_fixture fixture(*env, {}, std::string{"presigned"}, env->endpoint, std::nullopt, true);
    compare_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  }

  SECTION("header")
  {
    s3_sql_fixture fixture(*env, {}, std::string{"header"}, env->endpoint, std::nullopt, true);
    compare_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  }
}

TEST_CASE("gpu_execution aggregates real AWS S3 parquet through Sirius SigV4",
          "[.][s3][aws][live][sql][gpu_execution]")
{
  auto env = read_aws_live_env();
  if (!env) { return; }

  auto const s3_query =
    std::string{"SELECT count(*), min(n_nationkey), max(n_nationkey) FROM read_parquet("} +
    sql_quote(s3_uri(env->bucket, "fixtures/nation.parquet")) + ")";
  auto const local_query = "SELECT count(*), min(n_nationkey), max(n_nationkey) FROM " +
                           local_parquet_scan(*env, "nation");

  SECTION("presigned")
  {
    s3_sql_fixture fixture(*env, {}, std::string{"presigned"}, env->endpoint, std::nullopt, true);
    compare_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  }

  SECTION("header")
  {
    s3_sql_fixture fixture(*env, {}, std::string{"header"}, env->endpoint, std::nullopt, true);
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
  auto const s3_count_query = "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env);
  auto s3_result            = require_query_ok(fixture.con, gpu_execution_sql(s3_count_query));

  REQUIRE(s3_result->RowCount() == 1);
  CHECK(s3_result->GetValue(0, 0).GetValue<int64_t>() == static_cast<int64_t>(expected_rows));
  CHECK(large->total_num_rows == expected_rows);
  CHECK(expected_rows > 50'000'000);
}

TEST_CASE("gpu_execution large S3 lineitem TPC-H Q1 shape matches local CPU",
          "[.][s3][sql][large][large-q1][gpu_execution][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits(/*enable_prefetch_cache=*/true));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  compare_s3_gpu_to_local_cpu(fixture,
                              tpch_q1_shape_query(s3_large_lineitem_scan(*env)),
                              tpch_q1_shape_query(local_parquet_file_scan(large->local_path)),
                              {6, 7, 8});
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

  // Keep this plan unfiltered. The filtered query above checks correctness;
  // filter pushdown makes EXPLAIN report a post-filter estimate.
  auto const explain_sql = "SELECT count(*) FROM " + s3_sirius_large_lineitem_scan(*env) +
                           " l JOIN " + s3_sirius_parquet_scan(*env, "orders") +
                           " o ON l.l_orderkey = o.o_orderkey";
  auto const plan = explain_text(fixture.con, explain_sql);
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
  auto const s3_count_query = "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env);
  auto s3_result            = require_query_ok(fixture.con, gpu_execution_sql(s3_count_query));

  REQUIRE(s3_result->RowCount() == 1);
  CHECK(s3_result->GetValue(0, 0).GetValue<int64_t>() == static_cast<int64_t>(expected_rows));
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
