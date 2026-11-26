/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/s3/duckdb_secret_config.hpp"
#include "io/s3/sirius_httpfs.hpp"
#include "op/scan/field_id_mapping.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "planner/connector_registry.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "transparent/parquet_footer_statistics.hpp"
#include "utils/isolated_checkpoint_test.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/mock_iceberg_rest_catalog.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/s3_backend.hpp"
#include "utils/s3_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/multi_file/table_function_multi_file.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using sirius::test::s3::env_or;
using sirius::test::s3::sql_quote;
using sirius::test::s3::truthy_env;

namespace fs = std::filesystem;

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
  if (!sirius::test::ensure_s3_test_env()) { return std::nullopt; }

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
  return sirius::test::s3::skip_or_fail_unless(env.has_value(),
                                               "SIRIUS_TEST_S3_* environment is not configured");
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

std::string yaml_quote(std::string const& value) { return sql_quote(value); }

std::string read_text_file(fs::path const& path)
{
  std::ifstream in(path);
  REQUIRE(in);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

std::vector<std::uint8_t> read_binary_file(fs::path const& path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  REQUIRE(in);
  auto const size = in.tellg();
  REQUIRE(size >= 0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  REQUIRE(in);
  return bytes;
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
  std::string host_capacity{"512 MiB"};
  std::string disk_capacity{"2 GiB"};
  bool disk_tier{true};
  std::optional<std::string> cache_mode;
  std::optional<sirius::scan_manager::io_backend> backend;
  std::optional<std::string> rest_footer_probe_bytes;
  std::optional<std::size_t> rest_list_max_matches;
  std::optional<std::int64_t> footer_remote_reads;
  std::optional<std::int64_t> footer_local_reads;
};

sirius_memory_limits large_sirius_memory_limits(std::string cache_mode)
{
  sirius_memory_limits limits;
  limits.gpu_usage     = "3 GiB";
  limits.host_capacity = "8 GiB";
  limits.disk_capacity = "32 GiB";
  limits.cache_mode    = std::move(cache_mode);
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
    object_store_.endpoint       = endpoint.value_or(env.endpoint);
    object_store_.region         = env.region;
    object_store_.access_key     = env.access_key;
    object_store_.secret_key     = env.secret_key;
    object_store_.session_token  = env.session_token;
    object_store_.ca_bundle_path = ca_bundle.value_or("");
    object_store_.tls_verify     = tls_verify.value_or(false);
    if (signing_mode.has_value()) {
      REQUIRE(cucascade::io::string_to_enum(*signing_mode, object_store_.s3_signing_mode));
    }
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
           "        block_size: 1 MiB\n";
    if (limits.disk_tier) {
      out << "    disk:\n"
             "      - disk_id: 0\n"
             "        mount_path: "
          << yaml_quote((dir_ / "disk_memory").string())
          << "\n"
             "        memory_capacity: "
          << limits.disk_capacity << "\n";
    }
    out << "  executor:\n"
           "    scan_manager:\n";
    if (limits.cache_mode.has_value()) {
      out << "      cache:\n"
             "        mode: "
          << *limits.cache_mode << "\n";
    }
    if (limits.backend.has_value()) {
      std::string backend_name;
      REQUIRE(sirius::scan_manager::enum_to_string(*limits.backend, backend_name));
      out << "      backend: " << backend_name << "\n";
    }
    out << "      rest:\n"
           "        request_timeout_s: 30\n";
    if (limits.rest_footer_probe_bytes.has_value()) {
      out << "        footer_probe_bytes: " << yaml_quote(*limits.rest_footer_probe_bytes) << "\n";
    }
    if (limits.rest_list_max_matches.has_value()) {
      out << "        list_max_matches: " << *limits.rest_list_max_matches << "\n";
    }
    if (limits.footer_remote_reads || limits.footer_local_reads) {
      out << "  operator_params:\n";
      if (limits.footer_remote_reads) {
        out << "    parquet_footer_statistics_max_remote_reads: " << *limits.footer_remote_reads
            << '\n';
      }
      if (limits.footer_local_reads) {
        out << "    parquet_footer_statistics_max_local_reads: " << *limits.footer_local_reads
            << '\n';
      }
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
  [[nodiscard]] cucascade::io::object_store_config const& object_store() const noexcept
  {
    return object_store_;
  }

 private:
  fs::path dir_;
  fs::path config_path_;
  cucascade::io::object_store_config object_store_;
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
      db(nullptr, &db_config),
      con(db)
  {
    load_sirius_extension(db);
    auto context = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(context);
    context->get_config().set_object_store_config(config_env.object_store());
    context->get_scan_manager().install_s3_config("s3://" + env.bucket, config_env.object_store());
    setenv("SIRIUS_DISABLE", "1", 1);
  }

  sirius_config_env_guard config_env;
  // Avro and Iceberg are built locally against the pinned DuckDB revision.
  duckdb::DBConfig db_config{{{"allow_unsigned_extensions", duckdb::Value::BOOLEAN(true)}}, false};
  duckdb::DuckDB db;
  duckdb::Connection con;
};

// Starts Sirius with no programmatic object-store fallback. Tests using this
// fixture must make every S3 credential available through DuckDB secrets.
class s3_secret_only_sql_fixture {
 public:
  explicit s3_secret_only_sql_fixture(s3_test_env const& env)
    : config_env(env), db(nullptr), con(db)
  {
    load_sirius_extension(db);
    setenv("SIRIUS_DISABLE", "1", 1);
  }

  sirius_config_env_guard config_env;
  duckdb::DuckDB db;
  duckdb::Connection con;
};

std::string create_sirius_s3_secret_sql(s3_test_env const& env,
                                        std::string_view name,
                                        std::optional<std::string> scope = std::nullopt)
{
  auto sql = "CREATE SECRET " + std::string{name} + " (TYPE SIRIUS_S3";
  if (scope.has_value()) { sql += ", SCOPE " + sql_quote(*scope); }
  sql += ", KEY_ID " + sql_quote(env.access_key) + ", SECRET " + sql_quote(env.secret_key) +
         ", REGION " + sql_quote(env.region) + ", ENDPOINT " + sql_quote(env.endpoint) +
         ", USE_SSL " + (env.endpoint.rfind("https://", 0) == 0 ? "true" : "false");
  if (!env.session_token.empty()) { sql += ", SESSION_TOKEN " + sql_quote(env.session_token); }
  return sql + ")";
}

std::unique_ptr<duckdb::QueryResult> require_query_ok(duckdb::Connection& con,
                                                      std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  return std::unique_ptr<duckdb::QueryResult>(static_cast<duckdb::QueryResult*>(result.release()));
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

std::vector<std::vector<std::string>> collect_rows(duckdb::QueryResult& result)
{
  std::vector<std::vector<std::string>> rows;
  for (duckdb::idx_t r = 0; r < result.RowCount(); ++r) {
    std::vector<std::string> row;
    row.reserve(result.ColumnCount());
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); ++c) {
      row.push_back(result.Collection().GetValue(c, r).ToString());
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
          out.column_names.push_back(result->ColumnName(c).GetIdentifierName());
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

void check_rows_equal_with_tolerant_columns(duckdb::QueryResult& actual,
                                            duckdb::QueryResult& expected,
                                            std::vector<duckdb::idx_t> const& tolerant_columns = {})
{
  REQUIRE(actual.RowCount() == expected.RowCount());
  REQUIRE(actual.ColumnCount() == expected.ColumnCount());
  for (duckdb::idx_t r = 0; r < actual.RowCount(); ++r) {
    for (duckdb::idx_t c = 0; c < actual.ColumnCount(); ++c) {
      auto const actual_value   = actual.Collection().GetValue(c, r).ToString();
      auto const expected_value = expected.Collection().GetValue(c, r).ToString();
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
                                            duckdb::QueryResult& expected,
                                            std::vector<duckdb::idx_t> const& tolerant_columns = {})
{
  REQUIRE(actual.row_count == expected.RowCount());
  REQUIRE(actual.column_count == expected.ColumnCount());
  for (duckdb::idx_t r = 0; r < actual.row_count; ++r) {
    for (duckdb::idx_t c = 0; c < actual.column_count; ++c) {
      auto const& actual_value  = actual.rows[r][c];
      auto const expected_value = expected.Collection().GetValue(c, r).ToString();
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
  if (!work_dir.empty()) {
    WARN(
      "SIRIUS_BENCH_WORK_DIR is deprecated; set SIRIUS_PR6_LARGE_LOCAL_PARQUET to the SF10 "
      "lineitem file");
    return fs::path{work_dir} / "lineitem_sf10.parquet";
  }
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
  if (std::getenv("SIRIUS_PR6_LARGE_S3_KEY") != nullptr) {
    WARN(
      "SIRIUS_PR6_LARGE_S3_KEY is deprecated; set SIRIUS_BENCH_S3_KEY, which the harness also uses "
      "for the upload");
  }
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

cucascade::io::rest::rest_ioctx& require_rest_ioctx(s3_sql_fixture& fixture, std::string const& uri)
{
  return *sirius::test::s3::require_rest_ioctx(
    require_sirius_context(fixture).get_scan_manager().create_datasource(uri));
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
  if (sirius::test::s3::skip_or_fail_unless(truthy_env("SIRIUS_TEST_S3_LARGE"),
                                            "SIRIUS_TEST_S3_LARGE is not enabled")) {
    return std::nullopt;
  }

  large_lineitem_fixture out;
  out.uri        = s3_large_lineitem_uri(env);
  out.local_path = local_sf10_lineitem_path();
  if (sirius::test::s3::skip_or_fail_unless(
        fs::exists(out.local_path),
        "SF10 local parquet fixture is absent: " + out.local_path.string())) {
    return std::nullopt;
  }

  try {
    auto& sirius_ctx   = require_sirius_context(fixture);
    auto bind_info     = sirius_ctx.get_scan_manager().describe_parquet(out.uri);
    out.total_num_rows = static_cast<duckdb::idx_t>(bind_info.total_num_rows);
  } catch (std::exception const& e) {
    if (sirius::test::s3::skip_or_fail_unless(
          false, "SF10 S3 parquet fixture is absent at " + out.uri + ": " + e.what())) {
      return std::nullopt;
    }
  }
  return out;
}

duckdb::idx_t local_parquet_row_count(s3_test_env const& env, std::string_view table)
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto result = require_query_ok(con, "SELECT count(*) FROM " + local_parquet_scan(env, table));
  REQUIRE(result->RowCount() == 1);
  auto const rows = result->Collection().GetValue(0, 0).GetValue<int64_t>();
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
  auto const rows = result->Collection().GetValue(0, 0).GetValue<int64_t>();
  REQUIRE(rows >= 0);
  return static_cast<duckdb::idx_t>(rows);
}

std::string explain_text(duckdb::Connection& con, std::string const& sql)
{
  auto result = require_query_ok(con, "EXPLAIN " + sql);
  std::string out;
  for (duckdb::idx_t r = 0; r < result->RowCount(); ++r) {
    for (duckdb::idx_t c = 0; c < result->ColumnCount(); ++c) {
      out += result->Collection().GetValue(c, r).ToString();
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
  duckdb::BoundTableFunction& table_function,
  duckdb::vector<duckdb::LogicalType>& types,
  duckdb::vector<duckdb::Identifier>& names)
{
  duckdb::unique_ptr<duckdb::FunctionData> bind_data;
  ctx.RunFunctionInTransaction([&] {
    auto& entry = duckdb::Catalog::GetEntry<duckdb::TableFunctionCatalogEntry>(
      ctx, INVALID_CATALOG, DEFAULT_SCHEMA, "sirius_read_parquet");

    duckdb::vector<duckdb::LogicalType> arg_types;
    arg_types.emplace_back(duckdb::LogicalType::VARCHAR);
    table_function =
      duckdb::BoundTableFunction(entry.functions.GetFunctionByArguments(ctx, arg_types));

    duckdb::vector<duckdb::Value> inputs;
    inputs.emplace_back(uri);

    duckdb::named_argument_map_t named_parameters;
    duckdb::vector<duckdb::LogicalType> input_table_types;
    duckdb::vector<duckdb::Identifier> input_table_names;

    duckdb::TableFunctionRef ref;
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
    children.push_back(duckdb::ConstantExpression::String(uri));
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
          "[sql][s3]")
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
  CHECK(result->Collection().GetValue(0, 0).ToString() == "sirius_read_parquet");
  CHECK(result->Collection().GetValue(1, 0).ToString().find("VARCHAR") != std::string::npos);

  con.context->RunFunctionInTransaction([&] {
    auto& entry = duckdb::Catalog::GetEntry<duckdb::TableFunctionCatalogEntry>(
      *con.context, INVALID_CATALOG, DEFAULT_SCHEMA, "sirius_read_parquet");
    auto function =
      entry.functions.GetFunctionByArguments(*con.context, {duckdb::LogicalType::VARCHAR});
    REQUIRE(function->get_virtual_columns != nullptr);
    auto const columns = function->get_virtual_columns(*con.context, nullptr);
    CHECK(columns.at(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME).type ==
          duckdb::LogicalType::VARCHAR);
    CHECK(columns.at(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX).type ==
          duckdb::LogicalType::UBIGINT);
    CHECK(columns.at(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER).type ==
          duckdb::LogicalType::BIGINT);
  });
}

TEST_CASE("S3 SQL config guard keeps object-store credentials out of YAML", "[s3][config]")
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
    CHECK(yaml.find("executor:") != std::string::npos);
    CHECK(yaml.find("scan_manager:") != std::string::npos);
    CHECK(yaml.find("object_store:") == std::string::npos);
    CHECK(yaml.find("temporary-access-key") == std::string::npos);
    CHECK(yaml.find("temporary-secret-key") == std::string::npos);
    CHECK(yaml.find("session_token:") == std::string::npos);
    CHECK(yaml.find("signing_mode:") == std::string::npos);
  }

  env.session_token = "temporary-session-token";
  {
    sirius_config_env_guard guard(env, {}, std::string{"header"});
    auto const yaml = read_text_file(guard.config_path());
    CHECK(yaml.find("session_token:") == std::string::npos);
    CHECK(yaml.find("signing_mode:") == std::string::npos);
    CHECK(guard.object_store().session_token == "temporary-session-token");
    CHECK(guard.object_store().s3_signing_mode ==
          cucascade::io::object_store_config::signing_mode::header);
  }
}

TEST_CASE("Statements that carry credentials are logged without their text", "[s3][sql][log]")
{
  // No object store is contacted: this is about what the QueryBegin callback writes.
  s3_test_env env{"http://127.0.0.1:9000",
                  "",
                  "",
                  "us-east-1",
                  "temporary-access-key",
                  "temporary-secret-key",
                  "sirius-test",
                  "",
                  fs::temp_directory_path()};
  sirius_config_env_guard guard(env);
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  load_sirius_extension(db);
  setenv("SIRIUS_DISABLE", "1", 1);

  // Each statement creates a secret and holds the same synthetic value as a literal. DuckDB
  // hands the callback the statement as written, so a classifier that reads only its first
  // words, or ends a nested comment at the first "*/", logs the literal.
  auto const value      = std::string{"synthetic-credential-9c2e"};
  auto const secret_sql = "SECRET logged (TYPE SIRIUS_S3, KEY_ID 'k', SECRET '" + value + "')";
  std::vector<std::string> const statements{
    "CREATE /* a /* nested */ comment */ OR REPLACE " + secret_sql,
    "EXPLAIN CREATE OR REPLACE " + secret_sql,
    "EXPLAIN ANALYZE CREATE OR REPLACE " + secret_sql,
    "EXPLAIN ANALYSE CREATE OR REPLACE " + secret_sql,
    "EXPLAIN (ANALYZE) /* why */ CREATE OR REPLACE " + secret_sql,
  };
  sirius::test::scoped_recording_log_sink logs{"trace"};
  for (auto const& sql : statements) {
    INFO(sql.substr(0, sql.find("SECRET logged")));
    require_query_ok(con, sql);
  }
  // A statement of no such kind is the control: its text is logged.
  require_query_ok(con, "SELECT 'logged-in-full' AS control");
  auto const records = logs.records();
  auto const logged  = [&](std::string_view text) {
    return std::ranges::count_if(
      records, [&](auto const& record) { return record.message.find(text) != std::string::npos; });
  };
  CHECK(logged(value) == 0);
  CHECK(logged("SQL: CREATE SECRET <text withheld") == statements.size());
  CHECK(logged("SQL: SELECT 'logged-in-full' AS control") == 1);
}

TEST_CASE("S3 bench STS session token reaches presigned URLs", "[s3][sigv4]")
{
  cucascade::io::rest::s3::static_credentials creds;
  creds.access_key_id     = "AKIAFAKEBENCHKEY";
  creds.secret_access_key = "fake-secret-key";
  creds.session_token     = "fake-session-token";

  cucascade::io::rest::s3::sigv4_presigned_authorizer authorizer{
    std::move(creds), "us-east-2", "https://s3.us-east-2.amazonaws.com"};
  auto request = authorizer.authorize(
    cucascade::io::rest::object_ref{"sirius-bench", "tpch/lineitem_sf10.parquet"},
    cucascade::io::rest::request_method::GET,
    std::chrono::seconds{60});

  CHECK(request.headers.empty());
  CHECK(request.url.find("X-Amz-Security-Token=") != std::string::npos);
  CHECK(request.url.find("fake-session-token") != std::string::npos);
}

TEST_CASE("real-AWS live env guard requires regional endpoint and temporary credentials",
          "[s3][aws]")
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
          "[s3][integration][sql]")
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
    auto const nation_key = result->Collection().GetValue(0, row).GetValue<int32_t>();
    auto const region_key = result->Collection().GetValue(2, row).GetValue<int32_t>();
    CHECK(nation_key == static_cast<int32_t>(row));
    REQUIRE(region_key >= 0);
    REQUIRE(region_key < static_cast<int32_t>(region_counts.size()));
    ++region_counts[static_cast<std::size_t>(region_key)];
  }
  CHECK(result->Collection().GetValue(1, 0).ToString() == "ALGERIA");
  for (auto const count : region_counts) {
    CHECK(count == 5);
  }
}

TEST_CASE("transparent read_parquet over S3 scans through Sirius REST",
          "[s3][integration][sql][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto const uri      = s3_uri(env->bucket, "parquet/nation.parquet");
  auto const s3_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                        s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  auto const local_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                           local_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";

  SECTION("the query opens the object without a pre-resolved datasource")
  {
    sirius_memory_limits limits;
    limits.cache_mode = "none";
    limits.disk_tier  = false;
    s3_sql_fixture fixture(*env, limits);
    set_gpu_execution(fixture.con, true);
    auto s3_result = require_query_ok(fixture.con, s3_query);
    REQUIRE(s3_result->RowCount() == 25);
    REQUIRE(s3_result->ColumnCount() == 3);
    CHECK(s3_result->Collection().GetValue(0, 0).GetValue<int32_t>() == 0);
    CHECK(s3_result->Collection().GetValue(1, 0).ToString() == "ALGERIA");
    CHECK(s3_result->Collection().GetValue(2, 0).GetValue<int32_t>() == 0);

    duckdb::DuckDB local_db(nullptr);
    duckdb::Connection local_con(local_db);
    auto local_result = require_query_ok(local_con, local_query);
    check_rows_equal_with_tolerant_columns(*s3_result, *local_result);
  }

  SECTION("the REST datasource is resolved before the query")
  {
    s3_sql_fixture fixture(*env);
    set_gpu_execution(fixture.con, true);
    auto& rest = require_rest_ioctx(fixture, uri);
    CHECK(rest.type() == cucascade::io::io_context_type::restful);
    compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
  }
}

TEST_CASE("transparent read_parquet over S3 routes to REST when backend is native",
          "[s3][integration][sql][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  limits.backend = sirius::scan_manager::io_backend::native;
  s3_sql_fixture fixture(*env, limits);
  set_gpu_execution(fixture.con, true);

  // Native routing keeps S3 reads on REST, independently of the local backend.
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  CHECK(require_rest_ioctx(fixture, uri).type() == cucascade::io::io_context_type::restful);

  auto result =
    require_query_ok(fixture.con, "SELECT count(*) FROM read_parquet(" + sql_quote(uri) + ")");
  REQUIRE(result->RowCount() == 1);
  CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == 25);

  auto const s3_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                        s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  auto const local_query = "SELECT n_nationkey, n_name, n_regionkey FROM " +
                           local_parquet_scan(*env, "nation") + " ORDER BY n_nationkey";
  compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
}

TEST_CASE("S3 virtual columns preserve file identity and row positions through both SQL entries",
          "[s3][integration][sql][virtual_columns]")
{
  auto const transparent = GENERATE(false, true);
  CAPTURE(transparent);
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  // The explicit gpu_execution wrapper must remain a DuckDB table-function
  // call; its inner SQL is planned by Sirius. Transparent mode plans plain SQL.
  set_gpu_execution(fixture.con, transparent);
  require_query_ok(fixture.con, "SET enable_duckdb_fallback = false");
  auto compare = [&](std::string const& remote_sql, std::string const& local_sql) {
    if (transparent) {
      compare_transparent_s3_gpu_to_local_cpu(fixture, remote_sql, local_sql);
    } else {
      compare_s3_gpu_to_local_cpu(fixture, remote_sql, local_sql);
    }
  };
  auto const uri    = s3_uri(env->bucket, "parquet/nation.parquet");
  auto const remote = s3_parquet_scan(*env, "nation");
  auto const local  = local_parquet_scan(*env, "nation");
  // The CPU oracle reads the identical local fixture. Only the filename is
  // replaced with the expected S3 URI; row numbers come from native Parquet.
  auto const expected_prefix =
    "SELECT " + sql_quote(uri) + " AS filename, file_index, file_row_number, n_nationkey FROM ";
  auto const actual_prefix = "SELECT filename, file_index, file_row_number, n_nationkey FROM ";
  for (auto const& condition : {std::string{},
                                std::string{" WHERE n_nationkey IN (2, 9, 23)"},
                                std::string{" WHERE file_row_number > 7 AND n_nationkey < 13"},
                                std::string{" WHERE file_index = 1"}}) {
    auto const s3_sql  = actual_prefix + remote + condition + " ORDER BY file_row_number";
    auto const cpu_sql = expected_prefix + local + condition + " ORDER BY file_row_number";
    compare(s3_sql, cpu_sql);
  }

  auto const virtual_only = "SELECT file_index, file_row_number FROM ";
  compare(virtual_only + remote + " ORDER BY file_row_number",
          virtual_only + local + " ORDER BY file_row_number");

  compare(
    virtual_only + s3_parquet_scan(*env, "virtual_nested_only") + " ORDER BY file_row_number",
    virtual_only + local_parquet_scan(*env, "virtual_nested_only") + " ORDER BY file_row_number");

  // A filter-only virtual column must be bound even when no virtual is projected.
  compare("SELECT n_nationkey FROM " + remote + " WHERE filename = " + sql_quote(uri) +
            " ORDER BY n_nationkey",
          "SELECT n_nationkey FROM " + local + " ORDER BY n_nationkey");

  // Non-contiguous groups plus row filtering must preserve file-local positions.
  auto const provenance_uri = s3_uri(env->bucket, "parquet/virtual_provenance.parquet");
  auto const predicate = " WHERE (marker < 2048 OR marker >= 6144) AND keep = 1 ORDER BY marker";
  compare("SELECT filename, file_index, file_row_number, marker FROM " +
            s3_parquet_scan(*env, "virtual_provenance") + predicate,
          "SELECT " + sql_quote(provenance_uri) + ", file_index, file_row_number, marker FROM " +
            local_parquet_scan(*env, "virtual_provenance") + predicate);

  // A physical column sharing a virtual name wins binding, including its type
  // and null semantics; registration must not replace it with a constant.
  for (auto const& suffix : {std::string{" ORDER BY marker"},
                             std::string{" WHERE filename IS NULL ORDER BY marker"},
                             std::string{" WHERE filename IS NOT NULL ORDER BY marker"}}) {
    auto const prefix = "SELECT filename, file_index, file_row_number FROM ";
    compare(prefix + s3_parquet_scan(*env, "virtual_physical_names") + suffix,
            prefix + local_parquet_scan(*env, "virtual_physical_names") + suffix);
  }

  // Value::ToString() cannot distinguish a SQL NULL from the VARCHAR "NULL" and
  // also loses the logical types. Check the actual remote result at both SQL entries.
  auto query = [&](std::string const& sql) {
    return require_query_ok(fixture.con, transparent ? sql : gpu_execution_sql(sql));
  };
  duckdb::vector<duckdb::LogicalType> const virtual_types{
    duckdb::LogicalType::VARCHAR, duckdb::LogicalType::UBIGINT, duckdb::LogicalType::BIGINT};
  for (bool empty : {false, true}) {
    auto result = query("SELECT filename, file_index, file_row_number FROM " + remote +
                        (empty ? " WHERE file_index = 99" : "") + " ORDER BY file_row_number");
    CHECK(result->GetTypes() == virtual_types);
    REQUIRE(result->RowCount() == (empty ? 0 : 25));
    for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
      for (duckdb::idx_t col = 0; col < 3; ++col) {
        CHECK_FALSE(result->Collection().GetValue(col, row).IsNull());
      }
      CHECK(result->Collection().GetValue(0, row).GetValue<std::string>() == uri);
      CHECK(result->Collection().GetValue(1, row).GetValue<uint64_t>() == 0);
      CHECK(result->Collection().GetValue(2, row).GetValue<int64_t>() == row);
    }
  }
  auto star = query("SELECT * FROM " + remote + " LIMIT 1");
  REQUIRE(star->ColumnCount() == 4);
  CHECK(star->GetTypes() == duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER,
                                                                duckdb::LogicalType::VARCHAR,
                                                                duckdb::LogicalType::INTEGER,
                                                                duckdb::LogicalType::VARCHAR});
  for (duckdb::idx_t col = 0; col < 4; ++col) {
    CHECK(star->ColumnName(col).GetIdentifierName() ==
          std::vector<std::string>{"n_nationkey", "n_name", "n_regionkey", "n_comment"}[col]);
  }
  auto const physical_remote = s3_parquet_scan(*env, "virtual_physical_names");
  auto physical              = query("SELECT * FROM " + physical_remote + " ORDER BY marker");
  REQUIRE(physical->ColumnCount() == 4);
  CHECK(physical->GetTypes() == duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT,
                                                                    duckdb::LogicalType::VARCHAR,
                                                                    duckdb::LogicalType::BIGINT,
                                                                    duckdb::LogicalType::INTEGER});
  REQUIRE(physical->RowCount() == 3);
  CHECK(physical->Collection().GetValue(1, 0).IsNull());
  CHECK_FALSE(physical->Collection().GetValue(1, 1).IsNull());
  CHECK(physical->Collection().GetValue(1, 1).GetValue<std::string>() == "NULL");
  CHECK(physical->Collection().GetValue(1, 2).GetValue<std::string>() == "physical");
  for (duckdb::idx_t row = 0; row < physical->RowCount(); ++row) {
    CHECK(physical->Collection().GetValue(2, row).GetValue<int64_t>() == 77);
    CHECK(physical->Collection().GetValue(3, row).GetValue<int32_t>() == 88);
  }
  auto only_null = query("SELECT marker FROM " + physical_remote + " WHERE filename IS NULL");
  REQUIRE(only_null->RowCount() == 1);
  CHECK(only_null->Collection().GetValue(0, 0).GetValue<int64_t>() == 0);
  auto text_null = query("SELECT marker FROM " + physical_remote + " WHERE filename = 'NULL'");
  REQUIRE(text_null->RowCount() == 1);
  CHECK(text_null->Collection().GetValue(0, 0).GetValue<int64_t>() == 1);
}

TEST_CASE("SIRIUS_S3 CREATE SECRET credentials are scoped and refreshed for Sirius scans",
          "[s3][integration][sql][gpu_execution][secret]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto fixture_env       = *env;
  fixture_env.access_key = "invalid-fallback-key";
  fixture_env.secret_key = "invalid-fallback-secret";
  s3_sql_fixture fixture(fixture_env);
  set_gpu_execution(fixture.con, true);
  auto const uri           = s3_uri(env->bucket, "parquet/nation.parquet");
  auto const scope         = sql_quote("s3://" + env->bucket + "/parquet/");
  auto const outside_scope = sql_quote("s3://" + env->bucket + "/unrelated/");
  auto create_secret       = [&](std::string const& name,
                           std::string const& secret_scope,
                           std::string const& key_id,
                           std::string const& secret) {
    return "CREATE OR REPLACE SECRET " + name + " (TYPE SIRIUS_S3, SCOPE " + secret_scope +
           ", KEY_ID " + sql_quote(key_id) + ", SECRET " + sql_quote(secret) + ", REGION " +
           sql_quote(env->region) + ", ENDPOINT " + sql_quote(env->endpoint) + ", USE_SSL " +
           (env->endpoint.rfind("https://", 0) == 0 ? "true" : "false") + ")";
  };

  // An invalid credential scoped elsewhere must not shadow the matching
  // secret. The successful GPU scan proves Sirius's own SIRIUS_S3 secret reaches
  // its REST IO without relying on DuckDB's httpfs extension.
  require_query_ok(
    fixture.con, create_secret("s3_out_of_scope", outside_scope, "invalid-key", "invalid-secret"));
  require_query_ok(fixture.con,
                   create_secret("s3_matching", scope, env->access_key, env->secret_key));
  auto const scan_sql = "SELECT count(*) FROM read_parquet(" + sql_quote(uri) + ")";
  auto scan           = require_query_ok(fixture.con, gpu_execution_sql(scan_sql));
  REQUIRE(scan->RowCount() == 1);
  CHECK(scan->Collection().GetValue(0, 0).GetValue<int64_t>() == 25);

  // CREATE OR REPLACE changes future opens without requiring a new connection.
  require_query_ok(
    fixture.con,
    create_secret("s3_matching", scope, "rotated-invalid-key", "rotated-invalid-secret"));
  auto failed_scan = fixture.con.Query(gpu_execution_sql(scan_sql));
  REQUIRE(failed_scan);
  CHECK(failed_scan->HasError());
  CHECK(failed_scan->GetError().find("rotated-invalid-secret") == std::string::npos);

  require_query_ok(fixture.con,
                   create_secret("s3_matching", scope, env->access_key, env->secret_key));
  auto rescanned = require_query_ok(fixture.con, gpu_execution_sql(scan_sql));
  CHECK(rescanned->Collection().GetValue(0, 0).GetValue<int64_t>() == 25);
}

TEST_CASE("SIRIUS_S3 secret authenticates every file in an explicit parquet list",
          "[s3][integration][sql][gpu_execution][secret]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_secret_only_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_query_ok(
    fixture.con,
    create_sirius_s3_secret_sql(*env, "explicit_list", "s3://" + env->bucket + "/glob/multi/"));

  auto const first  = sql_quote(s3_uri(env->bucket, "glob/multi/nation_a.parquet"));
  auto const second = sql_quote(s3_uri(env->bucket, "glob/multi/nation_b.parquet"));
  auto const scan_sql =
    "SELECT count(*) FROM read_parquet([" + first + ", " + second + "], union_by_name=false)";
  auto result = require_query_ok(fixture.con, gpu_execution_sql(scan_sql));
  REQUIRE(result->RowCount() == 1);
  CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == 50);
}

TEST_CASE("SIRIUS_S3 secret authenticates an uppercase-scheme parquet glob",
          "[s3][integration][sql][gpu_execution][secret][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_secret_only_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_query_ok(
    fixture.con, create_sirius_s3_secret_sql(*env, "uppercase_glob", "s3://" + env->bucket + "/"));

  auto uri = s3_uri(env->bucket, "root_*.parquet");
  uri.replace(0, 2, "S3");
  auto const scan_sql = "SELECT count(*) FROM read_parquet(" + sql_quote(uri) + ")";
  auto result         = require_query_ok(fixture.con, scan_sql);
  REQUIRE(result->RowCount() == 1);
  CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == 50);
}

TEST_CASE("S3 LIST keeps its REST context alive while credentials rotate",
          "[s3][integration][filesystem][secret][rotation]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto sirius_ctx =
    fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  auto& manager      = sirius_ctx->get_scan_manager();
  auto const scope   = "s3://" + env->bucket;
  auto const prefix  = scope + "/glob/multi/";
  auto rotated       = fixture.config_env.object_store();
  rotated.tls_verify = !rotated.tls_verify;

  std::size_t pages = 0;
  manager.list_objects_paged(prefix,
                             /*page_size=*/1,
                             [&](cucascade::io::rest::s3::list_objects_v2_page const&) {
                               ++pages;
                               if (pages == 1) {
                                 // Replacing the final scope retires the registry's owner of the
                                 // old context. The LIST call must retain its own shared owner
                                 // until all pages have been consumed.
                                 manager.install_s3_config(scope, rotated);
                               }
                               return true;
                             });
  CHECK(pages >= 2);
}

TEST_CASE("S3 planned read opens with its own config after the path's is replaced",
          "[s3][integration][filesystem][secret][rotation]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto sirius_ctx =
    fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  auto& manager   = sirius_ctx->get_scan_manager();
  auto const path = s3_uri(env->bucket, "parquet/nation.parquet");
  using signing   = cucascade::io::object_store_config::signing_mode;

  // Differs from every other installed config, so this path alone maps to its snapshot.
  auto own = fixture.config_env.object_store();
  own.s3_signing_mode =
    own.s3_signing_mode == signing::header ? signing::presigned : signing::header;
  auto const planned = manager.install_s3_config(path, own);
  REQUIRE(planned);
  // Another connection resolves the same path to credentials that cannot read it.
  auto other       = fixture.config_env.object_store();
  other.secret_key = "wrong";
  manager.install_s3_config(path, other);

  CHECK_THROWS(manager.create_datasource(path));
  auto opened = manager.create_datasource(path, cucascade::io::open_hint::generic, &*planned);
  REQUIRE(opened);
  // No path maps to that config any more, so nothing would retire a context the manager kept.
  std::weak_ptr<cucascade::io::ioctx> context = opened->io_ctx();
  opened.reset();
  CHECK(context.expired());

  // The bind-time footer read and the LIST of a glob take the caller's config the same way.
  CHECK_THROWS(manager.describe_parquet(path));
  CHECK(manager.describe_parquet(path, &*planned).total_num_rows == 25);
  auto const pattern = s3_uri(env->bucket, "parquet/nation*.parquet");
  auto const listed  = manager.install_s3_config(pattern, own);
  REQUIRE(listed);
  manager.install_s3_config(pattern, other);
  CHECK_THROWS(sirius::io::s3::expand_glob(pattern, manager));
  CHECK_FALSE(sirius::io::s3::expand_glob(pattern, manager, std::nullopt, &*listed).empty());
  manager.install_s3_config(pattern, fixture.config_env.object_store());
  manager.install_s3_config(path, fixture.config_env.object_store());

  // The same through SQL. Between the bind's or the glob's install and its read, another
  // connection resolves the path to the credentials that cannot read it.
  //
  // Each statement sums a column, so that it scans its files: DuckDB answers a bare count(*)
  // over Parquet from the footers. With the fallback off, a scan that cannot open a file is
  // the statement's error.
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con, "SET enable_duckdb_fallback=false");
  auto const glob_query =
    "SELECT sum(n_nationkey) FROM " + s3_parquet_glob_scan(*env, "glob/multi/nation_*.parquet");
  auto const glob_rows = collect_rows(*require_query_ok(fixture.con, glob_query));
  using rows_t         = std::vector<std::vector<std::string>>;
  // Every data file the three statements scan.
  std::vector<std::string> const scanned{path,
                                         s3_uri(env->bucket, "glob/multi/nation_a.parquet"),
                                         s3_uri(env->bucket, "glob/multi/nation_b.parquet")};
  auto const counters = sirius_ctx->physical_counters();
  // The install callback below runs wherever a file is opened, and DuckDB may read the footers
  // of a glob's files on several of its workers: every access to `replaced` takes the mutex.
  std::mutex replaced_mutex;
  std::vector<std::string> replaced;
  auto const installs_of = [&](std::string const& installed) {
    std::lock_guard lock{replaced_mutex};
    return std::ranges::count(replaced, installed);
  };
  std::size_t planned_statements = 0;
  struct reset_hooks {
    std::shared_ptr<sirius::op::scan::physical_check_counters> counters;
    ~reset_hooks()
    {
      counters->s3_config_installed_for_testing = {};
      counters->after_lowering_for_testing      = {};
    }
  } reset{counters};
  // Fires after every install that a read follows: the bind, the glob, and each file DuckDB's
  // own reader opens through sirius_httpfs.
  counters->s3_config_installed_for_testing = [&](std::string const& installed) {
    {
      std::lock_guard lock{replaced_mutex};
      replaced.push_back(installed);
    }
    manager.install_s3_config(installed, other);
  };
  // Fires once every scan of a statement is lowered. By then the statement has installed and
  // recorded the configuration of every data file it scans, and installs none again; the
  // scans open the files afterwards, and must do so with what was recorded.
  counters->after_lowering_for_testing = [&] {
    ++planned_statements;
    for (auto const& file : scanned)
      manager.install_s3_config(file, other);
  };
  auto const executions_before = sirius::test::get_transparent_execution_stats(fixture.con);
  auto bound =
    fixture.con.Query("SELECT sum(n_nationkey) FROM " + s3_sirius_parquet_scan(*env, "nation"));
  auto const planned_after_bound = planned_statements;
  auto globbed                   = fixture.con.Query(glob_query);
  auto const bind_installs       = installs_of(path);
  auto const glob_installs       = installs_of(s3_uri(env->bucket, "glob/multi/nation_*.parquet"));
  // A transparent read of the one file: DuckDB's binder opens it through sirius_httpfs before
  // the statement is planned, and the GPU scan opens it again when it runs.
  {
    std::lock_guard lock{replaced_mutex};
    replaced.clear();
  }
  auto direct =
    fixture.con.Query("SELECT sum(n_nationkey) FROM " + s3_parquet_scan(*env, "nation"));
  auto const opened_by_duckdb               = installs_of(path);
  auto const planned_at_end                 = planned_statements;
  counters->s3_config_installed_for_testing = {};
  counters->after_lowering_for_testing      = {};
  auto const executions_after = sirius::test::get_transparent_execution_stats(fixture.con);
  // Every path is left on the configuration that cannot read it.
  for (auto const& file : scanned)
    CHECK_THROWS(manager.create_datasource(file));
  for (auto const& file : scanned)
    manager.install_s3_config(file, fixture.config_env.object_store());
  manager.install_s3_config(s3_uri(env->bucket, "glob/multi/nation_*.parquet"),
                            fixture.config_env.object_store());
  CHECK(bind_installs > 0);
  CHECK(glob_installs > 0);
  CHECK(planned_after_bound == 1);
  CHECK(planned_at_end == 3);
  CHECK(opened_by_duckdb > 0);
  // All three ran on the GPU, on the configurations they planned. Each is checked by itself,
  // so that one failing does not hide the others; a failed statement shows its error.
  CHECK(executions_after.executions == executions_before.executions + 3);
  auto const rows_of = [](duckdb::QueryResult& result) {
    return result.HasError() ? rows_t{{result.GetError()}} : collect_rows(result);
  };
  CHECK(rows_of(*bound) == rows_t{{"300"}});
  CHECK(rows_of(*globbed) == glob_rows);
  CHECK(rows_of(*direct) == rows_t{{"300"}});
}

TEST_CASE("S3 idle REST contexts are bounded while credentials rotate",
          "[s3][integration][filesystem][secret][rotation]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto sirius_ctx =
    fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  auto& manager = sirius_ctx->get_scan_manager();
  // A planned query latches the counters' test hooks.
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con,
                   "SELECT count(*) FROM read_parquet(" +
                     sql_quote(s3_uri(env->bucket, "parquet/nation.parquet")) + ")");
  auto const counters = sirius_ctx->physical_counters();
  struct reset_limit {
    std::shared_ptr<sirius::op::scan::physical_check_counters> counters;
    ~reset_limit() { counters->idle_rest_context_limit_for_testing.reset(); }
  } reset{counters};
  counters->idle_rest_context_limit_for_testing = 1;

  // Four configurations that all reach the store, each for a path of its own that is opened
  // once and not again: what a catalog vending new credentials on every load leaves behind.
  using signing = cucascade::io::object_store_config::signing_mode;
  std::vector<cucascade::io::object_store_config> rotated;
  for (auto const mode : {signing::presigned, signing::header}) {
    for (bool const verify : {true, false}) {
      rotated.push_back(fixture.config_env.object_store());
      rotated.back().s3_signing_mode = mode;
      rotated.back().tls_verify      = verify;
    }
  }
  std::array<std::uint8_t, 1> const byte{0};
  auto const open = [&](std::size_t index) {
    auto const key = "rotation/object-" + std::to_string(index);
    REQUIRE(sirius::test::put_s3_test_object(key, byte));
    auto const path = s3_uri(env->bucket, key);
    manager.install_s3_config(path, rotated.at(index));
    auto source = manager.create_datasource(path);
    REQUIRE(source);
    return source;
  };

  auto const in_flight                           = open(0);
  std::weak_ptr<cucascade::io::ioctx> first_idle = open(1)->io_ctx();
  open(2);
  open(3);
  // The read in flight, the context just built, and one idle context. Without the bound every
  // configuration keeps its own.
  CHECK(manager.rest_context_count() <= 3);
  CHECK(first_idle.expired());
  // The read in flight kept its context in the manager: the same one serves its path again.
  CHECK(manager.create_datasource(s3_uri(env->bucket, "rotation/object-0"))->io_ctx() ==
        in_flight->io_ctx());
  // A dropped context is rebuilt when one of its paths is opened again.
  CHECK(manager.create_datasource(s3_uri(env->bucket, "rotation/object-1")));
}

TEST_CASE("S3 idle REST contexts are released when the statement that held them ends",
          "[s3][integration][sql][secret][rotation]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto sirius_ctx =
    fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  auto& manager = sirius_ctx->get_scan_manager();
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con, "SET enable_duckdb_fallback=false");
  // Two tables, each under a secret of its own. The secrets differ in a field the store does
  // not check, so each resolves to a configuration, and so a REST context, of its own.
  auto const nation = s3_uri(env->bucket, "parquet/nation.parquet");
  auto const region = s3_uri(env->bucket, "parquet/region.parquet");
  auto const secret = [&](std::string const& name, std::string const& scope, bool verify_ssl) {
    auto sql = create_sirius_s3_secret_sql(*env, name, scope);
    sql.pop_back();
    return sql + ", VERIFY_SSL " + (verify_ssl ? "true" : "false") + ")";
  };
  require_query_ok(fixture.con, secret("rotation_nation", nation, true));
  require_query_ok(fixture.con, secret("rotation_region", region, false));
  // One statement reads both, so its scans hold both contexts for as long as it runs.
  auto const both = "SELECT count(*) FROM read_parquet(" + sql_quote(nation) +
                    ") JOIN read_parquet(" + sql_quote(region) + ") ON n_regionkey = r_regionkey";
  auto const run = [&] {
    auto const before = sirius::test::get_transparent_execution_stats(fixture.con);
    auto result       = require_query_ok(fixture.con, both);
    auto const after  = sirius::test::get_transparent_execution_stats(fixture.con);
    CHECK(collect_rows(*result) == std::vector<std::vector<std::string>>{{"25"}});
    CHECK(after.executions == before.executions + 1);
  };
  // A planned query latches the counters' test hooks.
  run();
  auto const counters = sirius_ctx->physical_counters();
  struct reset_limit {
    std::shared_ptr<sirius::op::scan::physical_check_counters> counters;
    ~reset_limit() { counters->idle_rest_context_limit_for_testing.reset(); }
  } reset{counters};

  // Within the bound both stay cached once the statement has ended.
  counters->idle_rest_context_limit_for_testing = 2;
  run();
  CHECK(manager.rest_context_count() == 2);
  std::weak_ptr<cucascade::io::ioctx> const nation_context =
    manager.create_datasource(nation)->io_ctx();
  std::weak_ptr<cucascade::io::ioctx> const region_context =
    manager.create_datasource(region)->io_ctx();
  REQUIRE_FALSE(nation_context.expired());
  REQUIRE_FALSE(region_context.expired());
  CHECK(nation_context.lock() != region_context.lock());

  // The same statement again, with the bound lowered once its scans are lowered. By then its
  // bind has closed its file handles, so nothing gives a context back before the scans take
  // both; while it runs, both are in use and neither can be dropped. Once its scans are
  // released, one of the two is given back, without a later statement having to ask.
  struct reset_hook {
    std::shared_ptr<sirius::op::scan::physical_check_counters> counters;
    ~reset_hook() { counters->after_lowering_for_testing = {}; }
  } reset_lowering{counters};
  counters->after_lowering_for_testing = [&] { counters->idle_rest_context_limit_for_testing = 1; };
  run();
  CHECK(manager.rest_context_count() == 1);
  CHECK(nation_context.expired() != region_context.expired());
}

TEST_CASE(
  "S3 idle REST contexts are released when a file handle closes or retained Parquet sources "
  "are dropped",
  "[s3][integration][sql][secret][rotation]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits("sirius"));
  auto sirius_ctx =
    fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  auto& manager = sirius_ctx->get_scan_manager();
  set_gpu_execution(fixture.con, true);
  // Two files, each under a secret of its own. The secrets differ in a field the store does not
  // check, so each file has a REST context of its own.
  auto const nation = s3_uri(env->bucket, "parquet/nation.parquet");
  auto const region = s3_uri(env->bucket, "parquet/region.parquet");
  auto const secret = [&](std::string const& name, std::string const& scope, bool verify_ssl) {
    auto sql = create_sirius_s3_secret_sql(*env, name, scope);
    sql.pop_back();
    return sql + ", VERIFY_SSL " + (verify_ssl ? "true" : "false") + ")";
  };
  require_query_ok(fixture.con, secret("reader_nation", nation, true));
  require_query_ok(fixture.con, secret("reader_region", region, false));
  // A planned query latches the counters' test hooks.
  require_query_ok(fixture.con,
                   "SELECT sum(n_nationkey) FROM read_parquet(" + sql_quote(nation) + ")");
  auto const counters = sirius_ctx->physical_counters();
  struct reset_limit {
    std::shared_ptr<sirius::op::scan::physical_check_counters> counters;
    ~reset_limit() { counters->idle_rest_context_limit_for_testing.reset(); }
  } reset{counters};
  counters->idle_rest_context_limit_for_testing = 1;

  // From here on no statement runs, and no context is built after the readers of a step are
  // open: a context that goes is given back because its last reader went.
  {
    auto& fs           = duckdb::FileSystem::GetFileSystem(*fixture.con.context);
    auto nation_handle = fs.OpenFile(nation, duckdb::FileFlags::FILE_FLAGS_READ);
    auto region_handle = fs.OpenFile(region, duckdb::FileFlags::FILE_FLAGS_READ);
    // Each handle holds its context: neither is idle.
    CHECK(manager.rest_context_count() == 2);
    nation_handle.reset();
    // One idle context is within the bound.
    CHECK(manager.rest_context_count() == 2);
    region_handle.reset();
    CHECK(manager.rest_context_count() == 1);
  }
  // The same for Parquet pins, which hold the datasources of their files until removed.
  REQUIRE(manager.pin_parquet_ranges("reader_nation", {nation}, std::nullopt) > 0);
  REQUIRE(manager.pin_parquet_ranges("reader_region", {region}, std::nullopt) > 0);
  CHECK(manager.rest_context_count() == 2);
  manager.remove_pinned_entry("reader_nation");
  CHECK(manager.rest_context_count() == 2);
  manager.remove_pinned_entry("reader_region");
  CHECK(manager.rest_context_count() == 1);
  // Resetting the caches drops the datasources of every pin.
  REQUIRE(manager.pin_parquet_ranges("reader_nation", {nation}, std::nullopt) > 0);
  REQUIRE(manager.pin_parquet_ranges("reader_region", {region}, std::nullopt) > 0);
  CHECK(manager.rest_context_count() == 2);
  sirius_ctx->reset_caches();
  CHECK(manager.rest_context_count() == 1);
  // Pinning a name again drops the datasources it held. With no idle context allowed, the
  // context that only they held goes.
  REQUIRE(manager.pin_parquet_ranges("reader_nation", {nation}, std::nullopt) > 0);
  REQUIRE(manager.pin_parquet_ranges("reader_region", {region}, std::nullopt) > 0);
  CHECK(manager.rest_context_count() == 2);
  counters->idle_rest_context_limit_for_testing = 0;
  REQUIRE(manager.pin_parquet_ranges("reader_nation", {region}, std::nullopt) > 0);
  CHECK(manager.rest_context_count() == 1);
}

TEST_CASE("transparent S3 read_parquet expands globbed parquet files",
          "[s3][integration][sql][transparent][glob]")
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

namespace {

duckdb::LogicalGet const* footer_admission_scan(duckdb::LogicalOperator const& plan)
{
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto const& get  = plan.Cast<duckdb::LogicalGet>();
    auto const& name = get.function.GetName().GetIdentifierName();
    if (name == "read_parquet" || name == "parquet_scan") { return &get; }
  }
  for (auto const& child : plan.children) {
    if (auto const* get = footer_admission_scan(*child)) { return get; }
  }
  return nullptr;
}

}  // namespace

TEST_CASE("Footer admission preflight binds a mixed local and SeaweedFS scan",
          "[.][s3][integration][parquet_footer_statistics][footer_admission_probe]")
{
  auto env = load_s3_test_env();
  REQUIRE(env.has_value());
  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con, "SET parquet_footer_statistics=true");
  sirius::test::scratch_dir directory("footer_admission_probe");
  duckdb::DuckDB writer_db(nullptr);
  duckdb::Connection writer(writer_db);
  require_query_ok(writer,
                   "COPY (SELECT range AS x FROM range(10)) TO " +
                     directory.file_literal("local.parquet") + " (FORMAT PARQUET)");
  require_query_ok(writer,
                   "COPY (SELECT range AS x FROM range(100, 110)) TO " +
                     directory.file_literal("remote.parquet") + " (FORMAT PARQUET)");
  auto const key = "footer-admission-probe/" + std::to_string(::getpid()) + "/remote.parquet";
  REQUIRE(
    sirius::test::put_s3_test_object(key, read_binary_file(directory.file("remote.parquet"))));
  auto const remote = s3_uri(env->bucket, key);
  REQUIRE(duckdb::FileSystem::IsRemoteFile(remote));
  auto plan = fixture.con.ExtractPlan("SELECT x FROM read_parquet([" +
                                      directory.file_literal("local.parquet") + ", " +
                                      sql_quote(remote) + "])");
  REQUIRE(plan);
  auto const* get = footer_admission_scan(*plan);
  REQUIRE(get != nullptr);
  auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get->bind_data.get());
  REQUIRE(bind != nullptr);
  REQUIRE(bind->file_list->GetAllFiles().size() == 2);
  CHECK_FALSE(bind->file_options.column_statistics.IsNull());
  CHECK(sirius::transparent::parquet_scan_row_count(*get) == std::optional<std::size_t>{20});
}

namespace {

namespace remote_footer_detail = sirius::transparent::detail;

struct remote_footer_cache_scope {
  remote_footer_detail::cache_capacities previous =
    remote_footer_detail::set_cache_capacities_for_testing(
      remote_footer_detail::default_cache_capacities);
  remote_footer_cache_scope() { remote_footer_detail::clear_caches_for_testing(); }
  ~remote_footer_cache_scope()
  {
    remote_footer_detail::clear_caches_for_testing();
    (void)remote_footer_detail::set_cache_capacities_for_testing(previous);
  }
};

struct footer_remote_files {
  sirius::test::scratch_dir directory{"footer_admission"};
  std::string glob;
  std::vector<std::string> remote;
  std::vector<std::string> local;

  footer_remote_files(s3_test_env const& env, int remote_count, int local_count = 0)
  {
    static std::atomic<unsigned> sequence{0};
    auto const prefix =
      "footer-admission-" + std::to_string(::getpid()) + "-" + std::to_string(sequence++) + "/";
    glob = s3_uri(env.bucket, prefix + "*.parquet");
    duckdb::DuckDB writer_db(nullptr);
    duckdb::Connection writer(writer_db);
    for (int index = 0; index < remote_count + local_count; ++index) {
      bool const upload = index < remote_count;
      auto const first  = upload ? 1000 + index * 100 : (index - remote_count) * 100;
      auto const name   = "part-" + std::to_string(index) + ".parquet";
      require_query_ok(writer,
                       "COPY (SELECT range AS x FROM range(" + std::to_string(first) + ", " +
                         std::to_string(first + 100) + ")) TO " + directory.file_literal(name) +
                         " (FORMAT PARQUET)");
      if (upload) {
        REQUIRE(
          sirius::test::put_s3_test_object(prefix + name, read_binary_file(directory.file(name))));
        remote.push_back(s3_uri(env.bucket, prefix + name));
        REQUIRE(duckdb::FileSystem::IsRemoteFile(remote.back()));
      } else {
        local.push_back(directory.file(name));
        fs::last_write_time(local.back(), fs::file_time_type::clock::now() - std::chrono::hours(1));
        REQUIRE_FALSE(duckdb::FileSystem::IsRemoteFile(local.back()));
      }
    }
  }

  std::string relation(bool mixed = false) const
  {
    if (!mixed) { return "read_parquet(" + sql_quote(glob) + ")"; }
    std::string paths;
    for (auto const* list : {&local, &remote}) {
      for (auto const& path : *list) {
        if (!paths.empty()) { paths += ", "; }
        paths += sql_quote(path);
      }
    }
    return "read_parquet([" + paths + "])";
  }
};

void check_remote_footer(duckdb::LogicalOperator const& plan,
                         std::size_t files,
                         std::optional<std::size_t> rows)
{
  auto const* get = footer_admission_scan(plan);
  REQUIRE(get != nullptr);
  auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get->bind_data.get());
  REQUIRE(bind != nullptr);
  CHECK(bind->file_list->GetAllFiles().size() == files);
  CHECK(bind->file_options.column_statistics.IsNull() == !rows.has_value());
  CHECK(sirius::transparent::parquet_scan_row_count(*get) == rows);
}

class footer_version_mask final : public sirius::io::s3::sirius_httpfs {
 public:
  explicit footer_version_mask(bool weak) : weak_(weak) {}
  std::string GetVersionTag(duckdb::FileHandle& handle) override
  {
    auto const tag = sirius::io::s3::sirius_httpfs::GetVersionTag(handle);
    return weak_ ? "W/" + tag : "";
  }
  duckdb::vector<duckdb::OpenFileInfo> Glob(std::string const& path,
                                            duckdb::FileOpener* opener) override
  {
    auto files = sirius::io::s3::sirius_httpfs::Glob(path, opener);
    for (auto& file : files) {
      if (!file.extended_info) { continue; }
      auto& options    = file.extended_info->options;
      auto const found = options.find("etag");
      if (found == options.end()) { continue; }
      if (weak_) {
        found->second = duckdb::Value("W/" + found->second.GetValue<std::string>());
      } else {
        options.erase(found);
      }
    }
    return files;
  }

 private:
  bool weak_;
};

}  // namespace

TEST_CASE("Footer admission counts remote misses at both sides of the boundary",
          "[.][s3][integration][parquet_footer_statistics][footer_admission_feature]")
{
  auto env = load_s3_test_env();
  REQUIRE(env);
  auto const count = GENERATE(2, 3, 4);
  CAPTURE(count);
  s3_sql_fixture fixture(*env);
  remote_footer_cache_scope caches;
  footer_remote_files files(*env, count);
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con, "SET parquet_footer_statistics_max_remote_reads=3");
  require_query_ok(fixture.con, "SET parquet_footer_statistics_max_local_reads=0");
  auto const before = remote_footer_detail::footer_read_count();
  auto plan         = fixture.con.ExtractPlan("SELECT x FROM " + files.relation());
  REQUIRE(plan);
  bool const admitted = count <= 3;
  CHECK(remote_footer_detail::footer_read_count() - before == (admitted ? count : 0));
  check_remote_footer(
    *plan, count, admitted ? std::optional<std::size_t>{count * 100} : std::nullopt);
  auto result = require_query_ok(
    fixture.con, "SELECT count(*)::BIGINT, sum(x)::BIGINT FROM " + files.relation());
  auto const rows = count * 100;
  std::vector<std::vector<std::string>> expected{
    {std::to_string(rows), std::to_string(rows * (2000 + rows - 1) / 2)}};
  CHECK(collect_rows(*result) == expected);
}

TEST_CASE("Footer admission skips a mixed scan when either independent cap is exceeded",
          "[.][s3][integration][parquet_footer_statistics][footer_admission_feature]")
{
  auto env = load_s3_test_env();
  REQUIRE(env);
  auto const exceeded = std::string(GENERATE("local", "remote", "neither"));
  CAPTURE(exceeded);
  s3_sql_fixture fixture(*env);
  remote_footer_cache_scope caches;
  footer_remote_files files(*env, 2, 2);
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con,
                   "SET parquet_footer_statistics_max_remote_reads=" +
                     std::string(exceeded == "remote" ? "1" : "2"));
  require_query_ok(fixture.con,
                   "SET parquet_footer_statistics_max_local_reads=" +
                     std::string(exceeded == "local" ? "1" : "2"));
  auto const before = remote_footer_detail::footer_read_count();
  auto plan         = fixture.con.ExtractPlan("SELECT x FROM " + files.relation(true));
  REQUIRE(plan);
  auto const admitted = exceeded == "neither";
  CHECK(remote_footer_detail::footer_read_count() - before == (admitted ? 4 : 0));
  check_remote_footer(*plan, 4, admitted ? std::optional<std::size_t>{400} : std::nullopt);
  auto result = require_query_ok(
    fixture.con, "SELECT count(*)::BIGINT, sum(x)::BIGINT FROM " + files.relation(true));
  CHECK(collect_rows(*result) == std::vector<std::vector<std::string>>{{"400", "239800"}});
}

TEST_CASE("Footer admission treats weak and missing remote validators as repeated misses",
          "[.][s3][integration][parquet_footer_statistics][footer_admission_feature]")
{
  auto env = load_s3_test_env();
  REQUIRE(env);
  auto const weak = GENERATE(false, true);
  CAPTURE(weak);
  s3_sql_fixture fixture(*env);
  remote_footer_cache_scope caches;
  footer_remote_files files(*env, 2);
  set_gpu_execution(fixture.con, true);
  fixture.db.GetFileSystem().UnregisterSubSystem("SiriusHttpFS");
  fixture.db.GetFileSystem().RegisterSubSystem(duckdb::make_uniq<footer_version_mask>(weak));
  require_query_ok(fixture.con, "SET parquet_footer_statistics_max_remote_reads=2");
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto const before = remote_footer_detail::footer_read_count();
    auto plan         = fixture.con.ExtractPlan("SELECT x FROM " + files.relation());
    REQUIRE(plan);
    CHECK(remote_footer_detail::footer_read_count() - before == 2);
    check_remote_footer(*plan, 2, 200);
  }
  require_query_ok(fixture.con, "SET parquet_footer_statistics_max_remote_reads=1");
  auto const before = remote_footer_detail::footer_read_count();
  auto skipped      = fixture.con.ExtractPlan("SELECT x FROM " + files.relation());
  REQUIRE(skipped);
  CHECK(remote_footer_detail::footer_read_count() == before);
  check_remote_footer(*skipped, 2, std::nullopt);
}

TEST_CASE("Footer admission counts cached remote files of an explicit list as reads",
          "[.][s3][integration][parquet_footer_statistics][footer_admission_feature]")
{
  auto env = load_s3_test_env();
  REQUIRE(env);
  s3_sql_fixture fixture(*env);
  remote_footer_cache_scope caches;
  footer_remote_files files(*env, 2);
  set_gpu_execution(fixture.con, true);
  for (auto const& path : files.remote) {
    auto analyzed = require_query_ok(
      fixture.con, "SELECT files FROM sirius_analyze_parquet(" + sql_quote(path) + ")");
    CHECK(analyzed->Collection().GetValue(0, 0).GetValue<uint64_t>() == 1);
  }
  REQUIRE(remote_footer_detail::statistics_cache_size() == 2);
  auto const before = remote_footer_detail::footer_read_count();
  require_query_ok(fixture.con, "SET parquet_footer_statistics_max_remote_reads=1");
  auto skipped = fixture.con.ExtractPlan("SELECT x FROM " + files.relation(true));
  REQUIRE(skipped);
  check_remote_footer(*skipped, 2, std::nullopt);
  require_query_ok(fixture.con, "SET parquet_footer_statistics_max_remote_reads=2");
  auto admitted = fixture.con.ExtractPlan("SELECT x FROM " + files.relation(true));
  REQUIRE(admitted);
  check_remote_footer(*admitted, 2, 200);
  CHECK(remote_footer_detail::footer_read_count() == before);
}

TEST_CASE("Footer admission inherits the remote engine cap until a connection overrides it",
          "[.][s3][integration][parquet_footer_statistics][footer_admission_feature]")
{
  auto env = load_s3_test_env();
  REQUIRE(env);
  sirius_memory_limits limits;
  limits.footer_remote_reads = 1;
  limits.footer_local_reads  = 3;
  s3_sql_fixture fixture(*env, limits);
  remote_footer_cache_scope caches;
  footer_remote_files files(*env, 2);
  duckdb::Connection second(fixture.db);
  set_gpu_execution(fixture.con, true);
  set_gpu_execution(second, true);
  auto expect = [&](duckdb::Connection& connection, bool admitted) {
    remote_footer_detail::clear_caches_for_testing();
    auto const before = remote_footer_detail::footer_read_count();
    auto plan         = connection.ExtractPlan("SELECT x FROM " + files.relation());
    REQUIRE(plan);
    check_remote_footer(*plan, 2, admitted ? std::optional<std::size_t>{200} : std::nullopt);
    CHECK(remote_footer_detail::footer_read_count() - before == (admitted ? 2 : 0));
  };
  auto initial = require_query_ok(
    fixture.con, "SELECT current_setting('parquet_footer_statistics_max_remote_reads')");
  CHECK(initial->Collection().GetValue(0, 0).IsNull());
  expect(fixture.con, false);
  require_query_ok(fixture.con, "SET parquet_footer_statistics_max_remote_reads=2");
  expect(fixture.con, true);
  expect(second, false);
  auto other = require_query_ok(
    second, "SELECT current_setting('parquet_footer_statistics_max_remote_reads')");
  CHECK(other->Collection().GetValue(0, 0).IsNull());
  require_query_ok(fixture.con, "RESET parquet_footer_statistics_max_remote_reads");
  expect(fixture.con, false);
  require_query_ok(fixture.con, "SET parquet_footer_statistics_max_remote_reads=2");
  require_query_ok(fixture.con, "SET parquet_footer_statistics_max_remote_reads=NULL");
  expect(fixture.con, false);
}

TEST_CASE("transparent S3 footer statistics follow an equal-size overwrite",
          "[s3][integration][sql][transparent][glob][parquet_footer_statistics]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);

  // Plain, uncompressed files of equal row counts have equal sizes whatever the values.
  sirius::test::scratch_dir dir("s3_footer_statistics");
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto write_plain = [&](std::string const& name, int from, int to) {
    require_query_ok(con,
                     "COPY (SELECT range AS x FROM range(" + std::to_string(from) + ", " +
                       std::to_string(to) + ")) TO " + dir.file_literal(name) +
                       " (FORMAT parquet, COMPRESSION uncompressed)");
    std::ifstream in(dir.file(name), std::ios::binary);
    REQUIRE(in.is_open());
    return std::vector<std::uint8_t>{std::istreambuf_iterator<char>{in},
                                     std::istreambuf_iterator<char>{}};
  };
  auto const low         = write_plain("low.parquet", 0, 100);
  auto const middle      = write_plain("middle.parquet", 100, 200);
  auto const replacement = write_plain("replacement.parquet", 1000, 1100);
  REQUIRE(middle.size() == replacement.size());
  if (!sirius::test::put_s3_test_object("footer-statistics/a.parquet", low)) {
    SUCCEED("managed SeaweedFS is required for the footer statistics overwrite test");
    return;
  }
  REQUIRE(sirius::test::put_s3_test_object("footer-statistics/b.parquet", middle));

  auto const glob = s3_uri(env->bucket, "footer-statistics/*.parquet");
  auto analyzed   = require_query_ok(
    fixture.con, "SELECT files FROM sirius_analyze_parquet(" + sql_quote(glob) + ")");
  REQUIRE(collect_rows(*analyzed) == std::vector<std::vector<std::string>>{{"2"}});

  auto const query = "SELECT count(*) FROM read_parquet(" + sql_quote(glob) + ") WHERE x >= 1000";
  REQUIRE(collect_rows(*require_query_ok(fixture.con, query)) ==
          std::vector<std::vector<std::string>>{{"0"}});
  REQUIRE(sirius::test::put_s3_test_object("footer-statistics/b.parquet", replacement));
  CHECK(collect_rows(*require_query_ok(fixture.con, query)) ==
        std::vector<std::vector<std::string>>{{"100"}});
}

TEST_CASE("transparent S3 glob refuses semantic type drift before decoding",
          "[s3][integration][sql][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);

  sirius::test::scratch_dir dir("s3_schema_drift");
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  require_query_ok(con,
                   "COPY (SELECT 1::INTEGER AS x UNION ALL SELECT 2) TO " +
                     dir.file_literal("a.parquet") + " (FORMAT PARQUET)");
  require_query_ok(con,
                   "COPY (SELECT 2.5::DOUBLE AS x UNION ALL SELECT 17.5) TO " +
                     dir.file_literal("b.parquet") + " (FORMAT PARQUET)");

  auto read_bytes = [](std::string const& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.is_open());
    std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>{in},
                                    std::istreambuf_iterator<char>{}};
    REQUIRE_FALSE(in.bad());
    REQUIRE_FALSE(bytes.empty());
    return bytes;
  };
  auto const bytes_a = read_bytes(dir.file("a.parquet"));
  auto const bytes_b = read_bytes(dir.file("b.parquet"));
  if (!sirius::test::put_s3_test_object("schema-drift/a.parquet", bytes_a)) {
    SUCCEED("managed SeaweedFS is required for the schema drift test");
    return;
  }
  REQUIRE(sirius::test::put_s3_test_object("schema-drift/b.parquet", bytes_b));

  auto before = sirius::test::get_transparent_execution_stats(fixture.con);
  auto result =
    fixture.con.Query("SELECT sum(x) FROM " + s3_parquet_glob_scan(*env, "schema-drift/*.parquet"));
  REQUIRE(result);
  INFO(result->ToString());
  REQUIRE(result->HasError());
  auto const error = result->GetError();
  CHECK(error.find("S3 CPU fallback is not supported") != std::string::npos);
  CHECK(error.find("Parquet column 'x' has unqualified type drift") != std::string::npos);
  auto after = sirius::test::get_transparent_execution_stats(fixture.con);
  CHECK(after.parquet_type_refusals == before.parquet_type_refusals + 1);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE("transparent S3 glob opens the literal percent key instead of its slash decoy",
          "[s3][integration][sql][transparent][glob]")
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
  CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == 25);
}

TEST_CASE("transparent S3 glob opens keys containing URI fragment and query delimiters",
          "[s3][integration][sql][transparent][glob]")
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
      if (pattern == "glob-enc/y*.parquet") {
        require_s3_keys_listed(fixture, *env, {"glob-enc/a%2Fb.parquet"});
      }
      auto const s3_query    = "SELECT count(*) FROM " + s3_parquet_glob_scan(*env, pattern);
      auto const local_query = "SELECT count(*) FROM " + local_parquet_glob_scan(*env, pattern);
      compare_transparent_s3_gpu_to_local_cpu(fixture, s3_query, local_query);
    }
  }
}

TEST_CASE("transparent S3 glob opens a key containing a literal percent byte",
          "[s3][integration][sql][transparent][glob]")
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
          "[s3][integration][sql][transparent][glob]")
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
  CHECK(result->Collection().GetValue(0, 0).ToString() == "a b");
  CHECK(result->Collection().GetValue(1, 0).GetValue<int64_t>() == 25);
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
          "[s3][integration][sql][transparent]")
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
  CHECK(literal_result->Collection().GetValue(0, 0).GetValue<int64_t>() == 25);
  CHECK(space_result->Collection().GetValue(0, 0).GetValue<int64_t>() == 5);
}

TEST_CASE("transparent S3 glob rejects a literal question mark in a Hive partition segment",
          "[s3][integration][sql][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/q/col=a?b/p0.parquet"});

  auto const scan =
    s3_parquet_glob_scan(*env, "glob-enc/q/col=a?b/*.parquet", ", hive_partitioning=true");
  CHECK_THROWS_WITH(query_or_throw_on_error(fixture.con, "SELECT count(*) FROM " + scan),
                    Catch::Matchers::ContainsSubstring("literal '?'"));
}

TEST_CASE("transparent S3 glob rejects a question mark before the Hive partition separator",
          "[s3][integration][sql][transparent][glob]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_s3_keys_listed(fixture, *env, {"glob-enc/guard-before/co?l=value/p0.parquet"});

  auto const scan =
    s3_parquet_glob_scan(*env, "glob-enc/guard-before/*/*.parquet", ", hive_partitioning=true");
  CHECK_THROWS_WITH(query_or_throw_on_error(fixture.con, "SELECT count(n_nationkey) FROM " + scan),
                    Catch::Matchers::ContainsSubstring("literal '?'"));
}

TEST_CASE("transparent S3 glob permits a question mark in the terminal filename",
          "[s3][integration][sql][transparent][glob]")
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
  CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == 25);
}

TEST_CASE("transparent S3 glob supports an encoded question mark in a Hive partition value",
          "[s3][integration][sql][transparent][glob]")
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
  CHECK(result->Collection().GetValue(0, 0).ToString() == "a?b");
  CHECK(result->Collection().GetValue(1, 0).GetValue<int64_t>() == 25);
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

TEST_CASE("transparent S3 glob preserves hive partition columns",
          "[s3][integration][sql][transparent][glob]")
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

TEST_CASE("transparent S3 glob remains correct with a straddled footer-probe window",
          "[s3][integration][sql][transparent][glob][footerbind]")
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
          "[s3][integration][sql][transparent][glob][footerbind]")
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
          "[s3][integration][sql][transparent][glob]")
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
    CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == expected);
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
  CHECK(uppercase_root->Collection().GetValue(0, 0).GetValue<int64_t>() == 50);

  auto wildcard_bucket =
    fixture.con.Query("SELECT count(*) FROM read_parquet('s3://*/glob/multi/nation_*.parquet')");
  REQUIRE(wildcard_bucket);
  REQUIRE(wildcard_bucket->HasError());
  INFO(wildcard_bucket->GetError());
  CHECK(wildcard_bucket->GetError().find("bucket") != std::string::npos);
}

TEST_CASE("transparent S3 glob scans 1001 parquet objects across LIST pages",
          "[.][s3][integration][sql][transparent][glob][large][glob-scale]")
{
  if (sirius::test::s3::skip_or_fail_unless(truthy_env("SIRIUS_TEST_S3_GLOB_SCALE"),
                                            "SIRIUS_TEST_S3_GLOB_SCALE is not enabled")) {
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
  CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == 25'025);
  CHECK(result->Collection().GetValue(1, 0).GetValue<int64_t>() == 300'300);
  CHECK(result->Collection().GetValue(2, 0).GetValue<int64_t>() == 0);
  CHECK(result->Collection().GetValue(3, 0).GetValue<int64_t>() == 24);
}

TEST_CASE("transparent S3 glob reports no-files and GPU-only errors clearly",
          "[s3][integration][sql][transparent][glob]")
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

TEST_CASE("S3 pushdown all-pruned filter completes with an empty result",
          "[s3][integration][sql][pushdown]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  auto fixture = std::make_shared<s3_sql_fixture>(*env);

  SECTION("unordered projection returns no rows through a direct query")
  {
    auto const s3_query =
      "SELECT n_nationkey FROM " + s3_parquet_scan(*env, "nation") + " WHERE n_regionkey = 99";
    auto result = require_query_ok(fixture->con, gpu_execution_sql(s3_query));
    CHECK(result->RowCount() == 0);
  }

  SECTION("ordered projection completes under the watchdog")
  {
    auto const s3_query = "SELECT n_nationkey FROM " + s3_parquet_scan(*env, "nation") +
                          " WHERE n_regionkey = 99 ORDER BY n_nationkey";

    auto result = require_query_ok_with_watchdog(
      fixture, gpu_execution_sql(s3_query), std::chrono::seconds{120});
    CHECK(result.row_count == 0);
    REQUIRE(result.column_count == 1);
    REQUIRE(result.column_names.size() == 1);
    CHECK(result.column_names[0] == "n_nationkey");
    CHECK(result.rows.empty());
  }
}

TEST_CASE("S3 pushdown zero-input ungrouped count emits the aggregate identity row",
          "[.][s3][integration][pushdown]")
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
          "[.][s3][integration][pushdown]")
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
          "[.][s3][integration][pushdown]")
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
          "[.][s3][integration][pushdown]")
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
          "[.][s3][integration][pushdown]")
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
          "[.][s3][integration][pushdown]")
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
      {"hash anti dead right",
       "SELECT n.n_nationkey FROM " + s.nation + " n ANTI JOIN " + s.pruned_region +
         " r ON n.n_regionkey = r.r_regionkey ORDER BY n.n_nationkey"},
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

  auto const zero_side_s3_queries    = build_zero_side_queries(make_scans(/*use_s3=*/true));
  auto const zero_side_local_queries = build_zero_side_queries(make_scans(/*use_s3=*/false));

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

  SECTION("hash ANTI JOIN dead build side")
  {
    compare_selected_cases("hash ANTI JOIN dead build side",
                           zero_side_s3_queries,
                           zero_side_local_queries,
                           {"hash anti dead right"});
  }

  SECTION("MARK NLJ both sides alive")
  {
    compare_selected_cases("MARK NLJ both sides alive",
                           zero_side_s3_queries,
                           zero_side_local_queries,
                           {"nlj mark both alive"});
  }

  SECTION("MARK NLJ dead build side")
  {
    compare_selected_cases("MARK NLJ dead build side",
                           zero_side_s3_queries,
                           zero_side_local_queries,
                           {"nlj mark dead right"});
  }

  SECTION("MARK NLJ dead probe side")
  {
    compare_selected_cases("MARK NLJ dead probe side",
                           zero_side_s3_queries,
                           zero_side_local_queries,
                           {"nlj mark dead left"});
  }
}

TEST_CASE("gpu_execution S3 SQL surface counts rows in five uploaded TPC-H tables",
          "[s3][integration][sql]")
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
    CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() ==
          static_cast<int64_t>(expected_rows));
  }
}

TEST_CASE("gpu_execution S3 SQL surface scans all orders row groups", "[s3][integration][sql]")
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

TEST_CASE("gpu_execution S3 scan keeps column values under column aliases",
          "[s3][integration][sql]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  // The aliases trade the names of nation's two key columns, which have one type. A scan that
  // resolved its columns by alias would return each one's values under the other's name.
  auto const query = [](std::string const& scan) {
    return "SELECT n_nationkey, n_regionkey, n_name FROM " + scan +
           " AS t(n_regionkey, n_name, n_nationkey, n_comment) ORDER BY 1, 2, 3";
  };
  compare_s3_gpu_to_local_cpu(
    fixture, query(s3_parquet_scan(*env, "nation")), query(local_parquet_scan(*env, "nation")));
  compare_s3_gpu_to_local_cpu(fixture,
                              query(s3_sirius_parquet_scan(*env, "nation")),
                              query(local_parquet_scan(*env, "nation")));
}

TEST_CASE("gpu_execution S3 SQL surface matches local TPC-H Q1 shape", "[s3][integration][sql]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  compare_s3_gpu_to_local_cpu(fixture,
                              tpch_q1_shape_query(s3_parquet_scan(*env, "lineitem")),
                              tpch_q1_shape_query(local_parquet_scan(*env, "lineitem")),
                              {6, 7, 8});
}

TEST_CASE("gpu_execution S3 SQL surface matches local TPC-H Q3 shape", "[s3][integration][sql]")
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

TEST_CASE("S3 read_parquet op shape T1 matches outer join semantics", "[s3][integration][sql]")
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
          "[s3][integration][sql]")
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

TEST_CASE("S3 read_parquet op shape T3 matches string predicates", "[s3][integration][sql]")
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

TEST_CASE("S3 read_parquet op shape T4 mixes local and S3 scans", "[s3][integration][sql]")
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
          "[s3][integration][sql]")
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
          "[s3][integration][sql]")
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
          "[.][s3][integration][sql][nested]")
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
          "[.][s3][integration][sql][nested]")
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
          "[s3][integration][sql][fallback][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  auto const s3_query =
    "SELECT n_nationkey, PERCENT_RANK() OVER (ORDER BY n_nationkey) AS pr FROM " +
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
          "[s3][integration][sql][fallback][transparent]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  auto const query = "SELECT md5(n_name) FROM " + s3_parquet_scan(*env, "nation") + " LIMIT 1";
  auto result      = fixture.con.Query(query);
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
          "[s3][integration][sql][fallback][transparent]")
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
    "SELECT n_nationkey, PERCENT_RANK() OVER (ORDER BY n_nationkey) AS pr "
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

TEST_CASE("transparent S3 read-view mismatch preserves the source veto",
          "[s3][integration][sql][transparent][fallback]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  REQUIRE_FALSE(fixture.con
                  .Query("CREATE VIEW v_s3_read_view AS SELECT n_nationkey FROM " +
                         s3_parquet_scan(*env, "nation"))
                  ->HasError());
  REQUIRE_FALSE(
    fixture.con.Query("SET sirius_test_inject_read_view_mismatch = 'finalize'")->HasError());
  for (auto const fallback : {true, false}) {
    REQUIRE_FALSE(
      fixture.con
        .Query(std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"))
        ->HasError());
    auto const before = sirius::test::get_transparent_execution_stats(fixture.con);

    auto result = fixture.con.Query("SELECT sum(n_nationkey) FROM v_s3_read_view");
    REQUIRE(result);
    REQUIRE(result->HasError());
    auto const error = result->GetError();
    INFO(error);
    CHECK(error.find("S3 CPU fallback is not supported") != std::string::npos);
    CHECK(error.find("read-view mismatch") != std::string::npos);

    auto const after = sirius::test::get_transparent_execution_stats(fixture.con);
    CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.executions == before.executions);
  }
}

TEST_CASE("transparent S3 eligibility covers copy and SQL-replan correspondence",
          "[s3][integration][sql][transparent][fallback]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  REQUIRE_FALSE(fixture.con
                  .Query("CREATE VIEW v_s3_matrix AS SELECT n_nationkey FROM " +
                         s3_parquet_scan(*env, "nation"))
                  ->HasError());
  auto const single = "SELECT sum(n_nationkey) FROM v_s3_matrix";
  auto const multi =
    "SELECT sum(a.n_nationkey + b.n_nationkey) "
    "FROM v_s3_matrix a JOIN v_s3_matrix b USING (n_nationkey)";
  struct optimizer_reset {
    duckdb::Connection& connection;
    ~optimizer_reset() { connection.Query("RESET disabled_optimizers"); }
  } reset{fixture.con};

  for (auto const fallback : {true, false}) {
    CAPTURE(fallback);
    REQUIRE_FALSE(
      fixture.con
        .Query(std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"))
        ->HasError());
    for (auto const& query : {single, multi}) {
      auto const before = sirius::test::get_transparent_execution_stats(fixture.con);
      auto result       = fixture.con.Query(query);
      REQUIRE(result);
      REQUIRE_FALSE(result->HasError());
      REQUIRE(result->Collection().GetValue(0, 0).ToString() == (query == single ? "300" : "600"));
      auto const after = sirius::test::get_transparent_execution_stats(fixture.con);
      sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
      CHECK(after.read_view_mismatches == before.read_view_mismatches);
    }
  }

  REQUIRE_FALSE(fixture.con.Query("SET disabled_optimizers = 'extension'")->HasError());
  for (auto const fallback : {true, false}) {
    CAPTURE(fallback);
    REQUIRE_FALSE(
      fixture.con
        .Query(std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"))
        ->HasError());

    auto before = sirius::test::get_transparent_execution_stats(fixture.con);
    auto result = fixture.con.Query(single);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->Collection().GetValue(0, 0).ToString() == "300");
    auto after = sirius::test::get_transparent_execution_stats(fixture.con);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    CHECK(after.read_view_mismatches == before.read_view_mismatches);

    before = after;
    result = fixture.con.Query(multi);
    REQUIRE(result);
    REQUIRE(result->HasError());
    auto const error = result->GetError();
    INFO(error);
    CHECK(error.find("S3 CPU fallback is not supported") != std::string::npos);
    CHECK(error.find("reason=no_correspondence") != std::string::npos);
    CHECK(error.find("correspondence=none") != std::string::npos);
    after = sirius::test::get_transparent_execution_stats(fixture.con);
    CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
    sirius::test::require_transparent_execution_delta(before, after, 0, 0, 0);
  }
}

TEST_CASE("transparent S3 execution rebuild preserves read-view correspondence",
          "[s3][integration][sql][transparent][fallback]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }
  s3_sql_fixture fixture(*env);
  auto& con = fixture.con;
  set_gpu_execution(con, true);
  REQUIRE_FALSE(con
                  .Query("CREATE VIEW v_s3_rebuild AS SELECT n_nationkey FROM " +
                         s3_parquet_scan(*env, "nation"))
                  ->HasError());
  struct restore_settings {
    duckdb::Connection& con;
    ~restore_settings()
    {
      con.Query("RESET disabled_optimizers");
      con.Query("SET sirius_test_inject_read_view_mismatch = 'off'");
      con.Query("SET sirius_test_inject_pin_registry_change = false");
      con.Query("SET enable_duckdb_fallback = true");
    }
  } restore{con};
  REQUIRE_FALSE(con.Query("SET sirius_test_inject_pin_registry_change = true")->HasError());
  for (bool hooks : {false, true}) {
    REQUIRE_FALSE(
      con.Query(hooks ? "RESET disabled_optimizers" : "SET disabled_optimizers = 'extension'")
        ->HasError());
    REQUIRE_FALSE(con
                    .Query(hooks
                             ? "SET sirius_test_inject_read_view_mismatch = 'execute_copy_fails'"
                             : "SET sirius_test_inject_read_view_mismatch = 'off'")
                    ->HasError());
    for (bool fallback : {true, false}) {
      REQUIRE_FALSE(
        con.Query(std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"))
          ->HasError());
      for (bool multi : {false, true}) {
        // Hooks-off multi-scan queries decline at finalize, already covered above.
        if (!hooks && multi) continue;
        CAPTURE(hooks, fallback, multi);
        auto const before = sirius::test::get_transparent_execution_stats(con);
        auto result =
          con.Query(multi ? "SELECT sum(a.n_nationkey + b.n_nationkey) FROM v_s3_rebuild a "
                            "JOIN v_s3_rebuild b USING (n_nationkey)"
                          : "SELECT sum(n_nationkey) FROM v_s3_rebuild");
        REQUIRE(result);
        INFO((result->HasError() ? result->GetError() : "success"));
        REQUIRE_FALSE(result->HasError());
        CHECK(result->Collection().GetValue(0, 0).ToString() == (multi ? "600" : "300"));
        auto const after = sirius::test::get_transparent_execution_stats(con);
        CHECK(after.execution_rebuilds == before.execution_rebuilds + 1);
        CHECK(after.read_view_mismatches == before.read_view_mismatches);
        sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
      }
    }
  }
}

TEST_CASE("S3 read_parquet is rejected when transparent GPU execution is disabled",
          "[s3][integration][sql][transparent]")
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
          "[s3][integration][sql]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  auto const uri                  = s3_uri(env->bucket, "parquet/orders.parquet");
  auto const expected_orders_rows = local_parquet_row_count(*env, "orders");
  duckdb::BoundTableFunction table_function;
  duckdb::vector<duckdb::LogicalType> return_types;
  duckdb::vector<duckdb::Identifier> names;

  auto bind_data =
    bind_sirius_read_parquet(*fixture.con.context, uri, table_function, return_types, names);

  REQUIRE(bind_data != nullptr);
  auto const* typed = dynamic_cast<duckdb::SiriusReadParquetBindData const*>(bind_data.get());
  REQUIRE(typed != nullptr);
  CHECK(typed->uri == uri);
  CHECK(typed->total_num_rows == expected_orders_rows);
  CHECK_FALSE(return_types.empty());
  CHECK_FALSE(names.empty());
  CHECK(typed->bound_types == return_types);
  CHECK(typed->bound_names == duckdb::IdentifiersToStrings(names));
  REQUIRE(table_function.cardinality != nullptr);

  auto stats = table_function.cardinality(*fixture.con.context, bind_data.get());
  REQUIRE(stats != nullptr);
  CHECK(stats->has_estimated_cardinality);
  CHECK(stats->estimated_cardinality == expected_orders_rows);
  CHECK(stats->has_max_cardinality);
  CHECK(stats->max_cardinality == expected_orders_rows);
}

TEST_CASE("Sirius S3 capture uses the fresh schema from a name-only rebind",
          "[s3][integration][sql][footerbind]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env);
  sirius::test::scratch_dir scratch{"s3_name_only_rebind"};
  auto const first_path  = scratch.path() / "first.parquet";
  auto const second_path = scratch.path() / "second.parquet";
  require_query_ok(fixture.con,
                   "COPY (SELECT 1::INTEGER AS original_name) TO " +
                     sql_quote(first_path.string()) + " (FORMAT PARQUET)");
  require_query_ok(fixture.con,
                   "COPY (SELECT 1::INTEGER AS rebound_name) TO " +
                     sql_quote(second_path.string()) + " (FORMAT PARQUET)");

  auto const first_key  = "rebind/name-only-first.parquet";
  auto const second_key = "rebind/name-only-second.parquet";
  if (!sirius::test::put_s3_test_object(first_key, read_binary_file(first_path))) {
    SUCCEED("managed SeaweedFS is required for the name-only rebind test");
    return;
  }
  REQUIRE(sirius::test::put_s3_test_object(second_key, read_binary_file(second_path)));

  auto const first_uri  = s3_uri(env->bucket, first_key);
  auto const second_uri = s3_uri(env->bucket, second_key);
  duckdb::BoundTableFunction first_function;
  duckdb::vector<duckdb::LogicalType> first_types;
  duckdb::vector<duckdb::Identifier> first_names;
  auto first_bind = bind_sirius_read_parquet(
    *fixture.con.context, first_uri, first_function, first_types, first_names);
  REQUIRE(first_bind != nullptr);
  REQUIRE(first_names == duckdb::vector<duckdb::Identifier>{"original_name"});
  duckdb::LogicalGet first_get(duckdb::TableIndex(91),
                               duckdb::BoundTableFunction(first_function),
                               std::move(first_bind),
                               first_types,
                               first_names);
  first_get.parameters.emplace_back(first_uri);
  sirius::op::scan::bound_read_view first_captured;
  fixture.con.context->RunFunctionInTransaction([&] {
    first_captured = sirius::op::scan::capture_bound_read_view(first_get, *fixture.con.context);
  });
  REQUIRE(first_captured.identity != nullptr);
  REQUIRE(first_captured.identity->bound_names == duckdb::IdentifiersToStrings(first_names));

  duckdb::BoundTableFunction rebound_function;
  duckdb::vector<duckdb::LogicalType> rebound_types;
  duckdb::vector<duckdb::Identifier> rebound_names;
  auto rebound_bind = bind_sirius_read_parquet(
    *fixture.con.context, second_uri, rebound_function, rebound_types, rebound_names);
  REQUIRE(rebound_bind != nullptr);
  REQUIRE(rebound_types == first_types);
  REQUIRE(rebound_names == duckdb::vector<duckdb::Identifier>{"rebound_name"});

  // Model the node state that makes B1 important: a rebind has replaced the
  // bind payload, while LogicalGet::names still reflects the previous bind.
  duckdb::LogicalGet rebound_get(duckdb::TableIndex(91),
                                 duckdb::BoundTableFunction(rebound_function),
                                 std::move(rebound_bind),
                                 rebound_types,
                                 first_names);
  rebound_get.parameters.emplace_back(second_uri);
  REQUIRE(rebound_get.names == first_names);

  sirius::op::scan::bound_read_view captured;
  fixture.con.context->RunFunctionInTransaction([&] {
    captured = sirius::op::scan::capture_bound_read_view(rebound_get, *fixture.con.context);
  });
  REQUIRE(captured.identity != nullptr);
  CHECK(captured.identity->bound_types == rebound_types);
  CHECK(captured.identity->bound_names == duckdb::IdentifiersToStrings(rebound_names));
  CHECK(captured.identity->bound_names != duckdb::IdentifiersToStrings(rebound_get.names));
}

TEST_CASE("internal sirius_read_parquet exposes S3 row count to DuckDB EXPLAIN",
          "[s3][integration][sql]")
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
          "[s3][integration][sql]")
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
          "[s3][integration][sql][config]")
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

TEST_CASE("gpu_execution reads real AWS S3 parquet through Sirius SigV4", "[.][s3][aws][sql]")
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

TEST_CASE("gpu_execution aggregates real AWS S3 parquet through Sirius SigV4", "[.][s3][aws][sql]")
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
          "[s3][integration][sql][config]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }
  if (sirius::test::s3::skip_or_fail_unless(
        !env->https_endpoint.empty() && !env->ca_bundle_path.empty(),
        "SIRIUS_TEST_S3_HTTPS_ENDPOINT and SIRIUS_TEST_S3_CA_BUNDLE are required")) {
    return;
  }

  s3_sql_fixture fixture(*env, {}, std::nullopt, env->https_endpoint, env->ca_bundle_path, true);
  compare_s3_gpu_to_local_cpu(
    fixture,
    "SELECT n_nationkey, n_name FROM " + s3_parquet_scan(*env, "nation") + " ORDER BY n_nationkey",
    "SELECT n_nationkey, n_name FROM " + local_parquet_scan(*env, "nation") +
      " ORDER BY n_nationkey");
}

TEST_CASE("gpu_execution S3 SQL preserves correctness with cache.mode none",
          "[s3][integration][sql][config]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  limits.cache_mode = "none";
  s3_sql_fixture fixture(*env, limits);
  compare_s3_gpu_to_local_cpu(
    fixture,
    "SELECT count(*), sum(o_totalprice) FROM " + s3_parquet_scan(*env, "orders"),
    "SELECT count(*), sum(o_totalprice) FROM " + local_parquet_scan(*env, "orders"),
    {1});
}

TEST_CASE("gpu_execution S3 SQL matches the local oracle with a 256 MiB GPU tier and a disk tier",
          "[s3][integration][sql]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  sirius_memory_limits limits;
  limits.gpu_usage     = "256 MiB";
  limits.host_capacity = "512 MiB";
  limits.disk_capacity = "2 GiB";
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
          "[.][s3][sql][large][large-cache][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits("sirius"));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto const expected_rows  = local_parquet_file_row_count(large->local_path);
  auto const s3_count_query = "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env);
  auto s3_result            = require_query_ok(fixture.con, gpu_execution_sql(s3_count_query));

  REQUIRE(s3_result->RowCount() == 1);
  CHECK(s3_result->Collection().GetValue(0, 0).GetValue<int64_t>() ==
        static_cast<int64_t>(expected_rows));
  CHECK(large->total_num_rows == expected_rows);
  CHECK(expected_rows > 50'000'000);
}

TEST_CASE("gpu_execution large S3 lineitem TPC-H Q1 shape matches local CPU",
          "[.][s3][sql][large][large-cache][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits("sirius"));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  compare_s3_gpu_to_local_cpu(fixture,
                              tpch_q1_shape_query(s3_large_lineitem_scan(*env)),
                              tpch_q1_shape_query(local_parquet_file_scan(large->local_path)),
                              {6, 7, 8});
}

TEST_CASE("gpu_execution large S3 lineitem join uses planner cardinality and matches local CPU",
          "[.][s3][sql][large][large-cache][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits("sirius"));
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

TEST_CASE("gpu_execution large S3 lineitem count matches with cache.mode none",
          "[.][s3][sql][large][large-nocache][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits("none"));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  auto const expected_rows  = local_parquet_file_row_count(large->local_path);
  auto const s3_count_query = "SELECT count(l_orderkey) FROM " + s3_large_lineitem_scan(*env);
  auto s3_result            = require_query_ok(fixture.con, gpu_execution_sql(s3_count_query));

  REQUIRE(s3_result->RowCount() == 1);
  CHECK(s3_result->Collection().GetValue(0, 0).GetValue<int64_t>() ==
        static_cast<int64_t>(expected_rows));
}

TEST_CASE("gpu_execution large S3 lineitem Q1 shape matches local CPU with cache.mode none",
          "[.][s3][sql][large][large-nocache][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits("none"));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  compare_s3_gpu_to_local_cpu(fixture,
                              tpch_q1_shape_query(s3_large_lineitem_scan(*env)),
                              tpch_q1_shape_query(local_parquet_file_scan(large->local_path)),
                              {6, 7, 8});
}

TEST_CASE("gpu_execution large S3 lineitem join matches local CPU with cache.mode none",
          "[.][s3][sql][large][large-nocache][integration]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }

  s3_sql_fixture fixture(*env, large_sirius_memory_limits("none"));
  auto large = read_large_lineitem_fixture(fixture, *env);
  if (!large) { return; }

  compare_s3_gpu_to_local_cpu(
    fixture,
    large_lineitem_orders_join_query(s3_large_lineitem_scan(*env), s3_parquet_scan(*env, "orders")),
    large_lineitem_orders_join_query(local_parquet_file_scan(large->local_path),
                                     local_parquet_scan(*env, "orders")));
}

TEST_CASE("native walk failure in a mixed S3 plan preserves execution-time source veto",
          "[s3][integration][sql][transparent][fallback]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }
  if (sirius::test::run_isolated()) { return; }
  for (bool explicit_path : {false, true}) {
    s3_sql_fixture fixture(*env);
    set_gpu_execution(fixture.con, true);
    auto path = fs::temp_directory_path() / ("sirius_native_s3_" + std::to_string(::getpid()) +
                                             (explicit_path ? "_explicit.db" : "_transparent.db"));
    REQUIRE_FALSE(fs::exists(path));
    require_query_ok(fixture.con, "ATTACH '" + path.string() + "' AS lease_native");
    require_query_ok(fixture.con,
                     "CREATE TABLE lease_native.main.native_lease_t AS SELECT range::BIGINT i "
                     "FROM range(300000)");
    require_query_ok(fixture.con, "CHECKPOINT lease_native");
    require_query_ok(fixture.con, "SET enable_duckdb_fallback = true");
    require_query_ok(fixture.con, "SET sirius_test_inject_native_walk_failure = 'native_lease_t'");
    auto context = sirius::test::get_registered_sirius_context(fixture.con);
    auto before  = context->get_transparent_execution_stats();
    auto sql     = "SELECT count(*) FROM lease_native.main.native_lease_t n JOIN " +
               s3_parquet_scan(*env, "nation") + " s ON n.i = s.n_nationkey";
    auto result = fixture.con.Query(explicit_path ? gpu_execution_sql(sql) : sql);
    REQUIRE(result);
    REQUIRE(result->HasError());
    INFO(result->GetError());
    CHECK(result->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
    CHECK(result->GetError().find("injected native metadata walk failure") != std::string::npos);
    CHECK(result->GetError().find("GPU plan generation failed:") == std::string::npos);
    auto after = context->get_transparent_execution_stats();
    CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
    if (!explicit_path) {
      CHECK(after.successful_rebinds == before.successful_rebinds + 1);
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.executions == before.executions + 1);
    }
    require_query_ok(fixture.con, "DETACH lease_native");
    fs::remove(path);
  }
}

TEST_CASE("never-entered native and S3 windows preserve the runtime-unavailable error",
          "[s3][integration][sql][transparent][fallback]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }
  if (sirius::test::run_isolated()) { return; }
  for (bool explicit_path : {false, true}) {
    s3_sql_fixture fixture(*env);
    set_gpu_execution(fixture.con, true);
    auto path = fs::temp_directory_path() / ("sirius_unavailable_s3_" + std::to_string(::getpid()) +
                                             (explicit_path ? "_explicit.db" : "_transparent.db"));
    REQUIRE_FALSE(fs::exists(path));
    require_query_ok(fixture.con, "ATTACH '" + path.string() + "' AS lease_native");
    require_query_ok(fixture.con,
                     "CREATE TABLE lease_native.main.native_lease_t AS SELECT range::BIGINT i "
                     "FROM range(300000)");
    require_query_ok(fixture.con, "CHECKPOINT lease_native");
    require_query_ok(fixture.con, "SET enable_duckdb_fallback = true");
    require_query_ok(fixture.con, "SET sirius_test_mark_runtime_unavailable_before_window = true");
    auto context = sirius::test::get_registered_sirius_context(fixture.con);
    auto before  = context->get_transparent_execution_stats();
    auto sql     = "SELECT count(*) FROM lease_native.main.native_lease_t n JOIN " +
               s3_parquet_scan(*env, "nation") + " s ON n.i = s.n_nationkey";
    auto result = fixture.con.Query(explicit_path ? gpu_execution_sql(sql) : sql);
    REQUIRE(result);
    REQUIRE(result->HasError());
    INFO(result->GetError());
    CHECK(result->GetErrorType() == duckdb::ExceptionType::EXECUTOR);
    CHECK(result->GetError().find("Sirius GPU runtime is unavailable") != std::string::npos);
    CHECK(result->GetError().find("S3 CPU fallback is not supported") == std::string::npos);
    CHECK(context->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
    CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
    auto after = context->get_transparent_execution_stats();
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
    if (!explicit_path) {
      CHECK(after.successful_rebinds == before.successful_rebinds + 1);
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.executions == before.executions + 1);
    }
    // The poisoned runtime is destroyed with this fixture. Files are unique to this child.
    fs::remove(path);
  }
}

TEST_CASE("Explicit replay rejects S3 behind a view before CPU replay starts",
          "[s3][integration][sql][fallback]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) return;
  if (sirius::test::run_isolated()) return;
  s3_sql_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con,
                   "CREATE VIEW explicit_remote_view AS SELECT n_nationkey FROM " +
                     s3_parquet_scan(*env, "nation"));
  require_query_ok(fixture.con, "SET enable_duckdb_fallback = true");
  require_query_ok(fixture.con, "SET sirius_test_sync_cpu_replay = true");
  auto context                         = sirius::test::get_registered_sirius_context(fixture.con);
  unsigned replays                     = 0;
  context->cpu_replay_hook_for_testing = [&] { ++replays; };
  struct reset_hook {
    duckdb::SiriusContext& context;
    ~reset_hook() { context.cpu_replay_hook_for_testing = {}; }
  } reset{*context};

  auto const query =
    "SELECT n_nationkey, percent_rank() OVER (ORDER BY n_nationkey) "
    "FROM explicit_remote_view";
  SECTION("plan generation fails") {}
  SECTION("window entry fails")
  {
    require_query_ok(fixture.con, "SET sirius_test_mark_runtime_unavailable_before_window = true");
  }
  auto result = fixture.con.Query(gpu_execution_sql(query));
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(result->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
  CHECK(result->GetError().find("Underlying GPU error:") != std::string::npos);
  CHECK(replays == 0);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
}

TEST_CASE("S3 semantic verdict keeps the source replay veto",
          "[s3][integration][transparent][verdict]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) return;
  s3_sql_fixture fixture(*env);
  auto& con = fixture.con;
  set_gpu_execution(con, true);
  REQUIRE_FALSE(con.Query("SET sirius_test_inject_scan_verdict='unsupported'")->HasError());
  auto before = sirius::test::get_transparent_execution_stats(con);
  auto result = con.Query("SELECT n_nationkey FROM " + s3_parquet_scan(*env, "nation"));
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
  auto after = sirius::test::get_transparent_execution_stats(con);
  CHECK(after.semantic_verdicts[1] == before.semantic_verdicts[1] + 1);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(after.scan_lowerings == before.scan_lowerings);
  CHECK(after.window_tasks_started == before.window_tasks_started);
}

TEST_CASE("S3 late physical refusal follows a published GPU batch and never replays",
          "[s3][integration][transparent][late_failure]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) return;
  s3_sql_fixture fixture(*env, large_sirius_memory_limits("sirius"));
  sirius::test::scratch_dir directory("s3_late_physical");
  set_gpu_execution(fixture.con, false);
  require_query_ok(fixture.con,
                   "COPY (SELECT i::INTEGER x FROM range(4096) t(i)) TO " +
                     directory.file_literal("a.parquet") +
                     " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  require_query_ok(fixture.con,
                   "COPY (SELECT x::DOUBLE x FROM (VALUES (1.2),(2.0)) t(x)) TO " +
                     directory.file_literal("b.parquet") + " (FORMAT PARQUET)");
  for (auto file : {"a.parquet", "b.parquet"}) {
    if (!sirius::test::put_s3_test_object(std::string("late-scan-refusal/") + file,
                                          read_binary_file(directory.path() / file))) {
      SUCCEED("managed MinIO is required to upload the late-failure fixture");
      return;
    }
  }
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con, "SET scan_task_batch_size=1");
  require_query_ok(fixture.con, "SET sirius_test_hold_footer_index=2");
  auto& context = require_sirius_context(fixture);
  auto sql      = "SELECT sum(x) FROM " + s3_parquet_glob_scan(*env, "late-scan-refusal/*.parquet");
  // The explicit SQL rewriter accepts a single S3 URI, not a glob. The view
  // preserves this same two-file bound scan without rewriting the glob as an object key.
  require_query_ok(fixture.con,
                   "CREATE VIEW late_s3_input AS SELECT * FROM " +
                     s3_parquet_glob_scan(*env, "late-scan-refusal/*.parquet"));
  for (bool explicit_entry : {false, true}) {
    auto before = context.get_transparent_execution_stats();
    std::promise<void> footer_started;
    auto started = footer_started.get_future();
    std::atomic<bool> notified{false};
    auto counters                       = context.physical_counters();
    counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
      if (footer && !notified.exchange(true)) footer_started.set_value();
    };
    auto pending   = std::async(std::launch::async, [&] {
      return fixture.con.Query(
        explicit_entry ? "CALL gpu_execution('SELECT sum(x) FROM late_s3_input')" : sql);
    });
    auto ready     = started.wait_for(std::chrono::seconds(20)) == std::future_status::ready;
    auto published = ready && context.get_scan_manager().wait_for_publication_for_testing(
                                std::chrono::seconds(20));
    context.get_scan_manager().release_footer_hold_for_testing(2);
    auto result                         = pending.get();
    counters->parquet_phase_for_testing = {};
    INFO("explicit=" << explicit_entry);
    if (result->HasError()) UNSCOPED_INFO(result->GetError());
    REQUIRE(ready);
    REQUIRE(published);
    REQUIRE(result->HasError());
    CHECK(result->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
    auto after = context.get_transparent_execution_stats();
    auto cause = static_cast<size_t>(sirius::transparent::late_failure_cause::physical_input);
    CHECK(after.late_failures[cause] == before.late_failures[cause] + 1);
    CHECK(after.late_replays == before.late_replays);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
    std::vector<sirius::op::scan::scan_publication_observation> observations;
    for (auto const& [query, observation] : after.publications_by_query) {
      if (!before.publications_by_query.contains(query)) observations.push_back(observation);
    }
    REQUIRE(observations.size() == 1);
    auto const& registrations = observations.front().readahead_registrations;
    CHECK(registrations.contains(s3_uri(env->bucket, "late-scan-refusal/a.parquet")));
    CHECK_FALSE(registrations.contains(s3_uri(env->bucket, "late-scan-refusal/b.parquet")));
  }
}

TEST_CASE("S3 mixed Parquet schemas flush between files on first and repeated reads",
          "[s3][integration][sql][transparent][glob][d7]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) return;
  auto limits = large_sirius_memory_limits("sirius");
  s3_sql_fixture fixture(*env, limits);
  sirius::test::scratch_dir directory("s3_d7_mixed");
  set_gpu_execution(fixture.con, false);
  require_query_ok(
    fixture.con,
    "COPY (SELECT 10::INTEGER x) TO " + directory.file_literal("a.parquet") + " (FORMAT PARQUET)");
  require_query_ok(
    fixture.con,
    "COPY (SELECT 20::DOUBLE x) TO " + directory.file_literal("b.parquet") + " (FORMAT PARQUET)");
  for (auto file : {"a.parquet", "b.parquet"}) {
    REQUIRE(sirius::test::put_s3_test_object(std::string("schema-validation/") + file,
                                             read_binary_file(directory.path() / file)));
  }
  auto local = require_query_ok(
    fixture.con, "SELECT x FROM read_parquet(" + directory.file_literal("*.parquet") + ")");
  auto expected = collect_rows(*local);
  std::sort(expected.begin(), expected.end());
  REQUIRE(expected.size() == 2);
  CHECK(expected[0][0] == "10");
  CHECK(expected[1][0] == "20");
  set_gpu_execution(fixture.con, true);
  auto local_query = "SELECT x FROM read_parquet(" + directory.file_literal("*.parquet") + ")";
  for (int attempt = 0; attempt != 2; ++attempt) {
    auto before = sirius::test::get_transparent_execution_stats(fixture.con);
    auto result = require_query_ok(fixture.con, local_query);
    auto rows   = collect_rows(*result);
    std::sort(rows.begin(), rows.end());
    CHECK(rows == expected);
    CHECK(result->GetTypes() == local->GetTypes());
    auto after = sirius::test::get_transparent_execution_stats(fixture.con);
    CHECK(after.successful_rebinds == before.successful_rebinds + 1);
    CHECK(after.split_flushed_for_schema == before.split_flushed_for_schema + 1);
  }
  auto& manager     = require_sirius_context(fixture).get_scan_manager();
  auto local_source = manager.create_datasource((directory.path() / "a.parquet").string());
  REQUIRE(local_source);
  CHECK_FALSE(local_source->prefers_bulk_io());
  // Give pin_table pristine paths. The ordinary local scan above has warmed
  // a.parquet and b.parquet, so using those paths would only exercise a
  // metadata cache hit rather than the pin-first footer producer.
  for (auto file : {"a.parquet", "b.parquet"}) {
    std::filesystem::copy_file(directory.path() / file,
                               directory.path() / (std::string("pin-") + file));
  }
  auto pin_glob    = directory.file_literal("pin-*.parquet");
  auto cold_source = manager.create_datasource(directory.file("pin-a.parquet"));
  REQUIRE(cold_source);
  CHECK(cold_source->metadata() == nullptr);
  require_query_ok(fixture.con,
                   "CALL pin_table(" + pin_glob +
                     ", name='schema_validation_pin', format='parquet', tier='parquet')");
  CHECK(cold_source->metadata() != nullptr);
  auto local_before = sirius::test::get_transparent_execution_stats(fixture.con);
  auto pinned       = require_query_ok(fixture.con, "SELECT x FROM read_parquet(" + pin_glob + ")");
  auto pinned_rows  = collect_rows(*pinned);
  std::sort(pinned_rows.begin(), pinned_rows.end());
  CHECK(pinned_rows == expected);
  CHECK(pinned->GetTypes() == local->GetTypes());
  auto local_after = sirius::test::get_transparent_execution_stats(fixture.con);
  CHECK(local_after.successful_rebinds == local_before.successful_rebinds + 1);
  CHECK(local_after.executions == local_before.executions + 1);
  CHECK(local_after.runtime_fallbacks == local_before.runtime_fallbacks);
  CHECK(local_after.split_physical_rejections == local_before.split_physical_rejections);
  auto query    = "SELECT x FROM " + s3_parquet_glob_scan(*env, "schema-validation/*.parquet");
  auto counters = require_sirius_context(fixture).physical_counters();
  std::mutex hits_mutex;
  std::map<std::string, int> hits;
  counters->parquet_metadata_for_testing = [&](std::string const& file, bool hit) {
    if (hit) {
      std::lock_guard lock(hits_mutex);
      ++hits[file];
    }
  };
  struct reset_metadata_hook {
    std::shared_ptr<sirius::op::scan::physical_check_counters> counters;
    ~reset_metadata_hook() { counters->parquet_metadata_for_testing = {}; }
  } reset_hook{counters};
  for (int attempt = 0; attempt != 2; ++attempt) {
    {
      std::lock_guard lock(hits_mutex);
      hits.clear();
    }
    auto before = sirius::test::get_transparent_execution_stats(fixture.con);
    auto result = require_query_ok(fixture.con, query);
    auto rows   = collect_rows(*result);
    std::sort(rows.begin(), rows.end());
    CHECK(rows == expected);
    CHECK(result->GetTypes() == local->GetTypes());
    auto after = sirius::test::get_transparent_execution_stats(fixture.con);
    CHECK(after.successful_rebinds == before.successful_rebinds + 1);
    CHECK(after.executions == before.executions + 1);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(after.split_flushed_for_schema == before.split_flushed_for_schema + 1);
    CHECK(after.budget_exceeded == before.budget_exceeded);
    for (auto file : {"a.parquet", "b.parquet"}) {
      auto uri    = s3_uri(env->bucket, std::string("schema-validation/") + file);
      auto source = manager.create_datasource(uri);
      REQUIRE(source);
      CHECK(source->prefers_bulk_io());
      CHECK(source->metadata() != nullptr);
      if (attempt == 1) {
        std::lock_guard lock(hits_mutex);
        CHECK(hits[uri] > 0);
      }
    }
  }
}

TEST_CASE("S3 Puffin planning refuses eager and preserves deferred open failure",
          "[s3][integration][iceberg][scan_preparation][s3_dv]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) return;
  s3_sql_fixture fixture(*env);
  auto& con = fixture.con;
  set_gpu_execution(con, false);
  require_query_ok(con, "LOAD avro");
  require_query_ok(con, "LOAD iceberg");
  require_query_ok(con, "SET unsafe_enable_version_guessing=true");
  sirius::test::scratch_dir directory("s3_dv");
  std::string const prefix = "preparation-dv";
  auto remote              = "s3://" + env->bucket + "/" + prefix;
  auto shell_quote         = [](std::string const& value) {
    std::string result(1, '\'');
    for (char c : value) {
      if (c == '\'') {
        result += '\'';
        result += '\\';
        result += '\'';
        result += '\'';
      } else {
        result += c;
      }
    }
    return result + '\'';
  };
  auto command = "python3 -B test/cpp/integration/data/generate_inventory_fixtures.py " +
                 shell_quote(directory.path().string()) + " " + shell_quote(remote);
  REQUIRE(std::system(command.c_str()) == 0);
  auto local = directory.path() / "dv_bounded";
  for (auto const& entry : fs::recursive_directory_iterator(local)) {
    if (!entry.is_regular_file()) continue;
    auto relative = fs::relative(entry.path(), local);
    // Leave the Puffin sidecar absent so this case exercises an actual open failure.
    if (relative == fs::path("data/a.puffin")) continue;
    auto key = prefix + "/dv_bounded/" + relative.generic_string();
    if (!sirius::test::put_s3_test_object(key, read_binary_file(entry.path()))) {
      if (sirius::test::s3::skip_or_fail_unless(false, "managed S3 backend required for DV upload"))
        return;
    }
  }
  // Bind the fixture's exact snapshot; remote reads remain GPU-only.
  auto metadata_bytes = read_binary_file(local / "metadata/v1.metadata.json");
  std::string metadata(metadata_bytes.begin(), metadata_bytes.end());
  auto key = metadata.find("current-snapshot-id");
  REQUIRE(key != std::string::npos);
  auto colon = metadata.find(':', key);
  REQUIRE(colon != std::string::npos);
  auto snapshot = std::stoll(metadata.substr(colon + 1));
  auto sql      = "SELECT fruit, count FROM iceberg_scan(" +
             sql_quote(remote + "/dv_bounded/metadata/v1.metadata.json") +
             ", snapshot_from_id=" + std::to_string(snapshot) + ")";
  auto& context         = require_sirius_context(fixture);
  auto counters         = context.physical_counters();
  auto provider         = std::make_shared<sirius::scan_manager::test::test_reservation_provider>();
  provider->grant_bytes = 1024 * 1024;
  struct reset_hooks {
    decltype(counters) value;
    ~reset_hooks()
    {
      value->preparation_provider_for_testing.reset();
      value->iceberg_preparation_route_for_testing = {};
      value->iceberg_dv_phase_for_testing          = {};
    }
  } reset{counters};
  counters->preparation_provider_for_testing = provider;
  set_gpu_execution(con, true);
  for (bool deferred : {false, true}) {
    for (bool fallback : {false, true}) {
      provider->return_null = !deferred;
      size_t routes = 0, reads_started = 0;
      bool selected = !deferred;
      std::string opened_path;
      counters->iceberg_preparation_route_for_testing = [&](auto, bool route) {
        ++routes;
        selected = route;
      };
      counters->iceberg_dv_phase_for_testing = [&](auto const& path, bool start) {
        if (start) {
          opened_path = path;
          ++reads_started;
        }
      };
      require_query_ok(con,
                       std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
      auto before = context.get_transparent_execution_stats();
      auto result = con.Query(sql);
      REQUIRE(result);
      REQUIRE(result->HasError());
      INFO(result->GetError());
      if (!deferred) {
        // A remote Puffin can only use the deferred datasource route. Planning
        // must decline before route publication, permit ownership, or a
        // Puffin open; the CPU fallback may still report its own missing-file
        // error when explicitly enabled.
        CHECK(routes == 0);
        CHECK(opened_path.empty());
        CHECK(reads_started == 0);
      } else {
        CHECK(routes == 1);
        CHECK(selected);
        CHECK(opened_path == remote + "/dv_bounded/data/a.puffin");
        CHECK(reads_started == 1);
        CHECK(result->GetError().find("Cannot open file") != std::string::npos);
      }
      auto after = context.get_transparent_execution_stats();
      auto cause = static_cast<size_t>(sirius::transparent::late_failure_cause::reader_io);
      CHECK(after.late_failures[cause] == before.late_failures[cause] + (deferred ? 1 : 0));
      CHECK(after.late_replays[cause] == before.late_replays[cause]);
      auto eager_reason =
        static_cast<size_t>(sirius::op::scan::verdict_reason::iceberg_puffin_remote_eager);
      CHECK(after.semantic_declines[eager_reason] ==
            before.semantic_declines[eager_reason] + (deferred ? 0 : 1));
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
      CHECK(after.fallbacks == before.fallbacks);

      CHECK(provider->outstanding() == 0);
      CHECK(provider->seen->allocated_bytes == 0);
    }
  }

  // A supported remote Puffin must still be refused before any payload open if
  // the deferred arena cannot be constructed after admission. This protects
  // the planner fallback from accidentally retaining a stale deferred decision
  // and entering the eager payload path.
  provider->return_null           = false;
  provider->seen->fail_allocation = true;
  size_t allocation_routes        = 0;
  size_t allocation_reads_started = 0;
  std::string allocation_opened_path;
  counters->iceberg_preparation_route_for_testing = [&](auto, bool) { ++allocation_routes; };
  counters->iceberg_dv_phase_for_testing          = [&](auto const& path, bool start) {
    if (start) {
      allocation_opened_path = path;
      ++allocation_reads_started;
    }
  };
  require_query_ok(con, "SET enable_duckdb_fallback=false");
  auto before = context.get_transparent_execution_stats();
  auto result = con.Query(sql);
  REQUIRE(result);
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
  CHECK(allocation_routes == 0);
  CHECK(allocation_opened_path.empty());
  CHECK(allocation_reads_started == 0);
  auto after = context.get_transparent_execution_stats();
  auto eager_reason =
    static_cast<size_t>(sirius::op::scan::verdict_reason::iceberg_puffin_remote_eager);
  CHECK(after.semantic_declines[eager_reason] == before.semantic_declines[eager_reason] + 1);
  CHECK(provider->outstanding() == 0);
  CHECK(provider->seen->allocated_bytes == 0);
  provider->seen->fail_allocation = false;
}

TEST_CASE("S3 remote Puffin deferred route matches an independent local CPU reference",
          "[s3][integration][iceberg][scan_preparation][s3_dv]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) return;
  s3_sql_fixture fixture(*env);
  auto& con = fixture.con;
  require_query_ok(con, "LOAD avro");
  require_query_ok(con, "LOAD iceberg");
  require_query_ok(con, "SET unsafe_enable_version_guessing=true");

  sirius::test::scratch_dir local_directory("s3_dv_cpu_reference");
  sirius::test::scratch_dir remote_directory("s3_dv_remote_reference");
  auto shell_quote = [](std::string const& value) {
    std::string result(1, '\'');
    for (char c : value) {
      if (c == '\'') {
        result += '\'';
        result += '\\';
        result += '\'';
        result += '\'';
      } else {
        result += c;
      }
    }
    return result + '\'';
  };
  auto generate = [&](std::filesystem::path const& directory, std::string const& remote) {
    auto command = "python3 -B test/cpp/integration/data/generate_inventory_fixtures.py " +
                   shell_quote(directory.string());
    if (!remote.empty()) command += " " + shell_quote(remote);
    REQUIRE(std::system(command.c_str()) == 0);
  };

  auto const prefix = std::string{"deferred-dv-positive"};
  auto const remote = "s3://" + env->bucket + "/" + prefix;
  generate(local_directory.path(), "");
  generate(remote_directory.path(), remote);

  auto const local        = local_directory.path() / "dv_bounded";
  auto const remote_local = remote_directory.path() / "dv_bounded";
  for (auto const& entry : fs::recursive_directory_iterator(remote_local)) {
    if (!entry.is_regular_file()) continue;
    auto relative = fs::relative(entry.path(), remote_directory.path());
    auto key      = prefix + "/" + relative.generic_string();
    if (!sirius::test::put_s3_test_object(key, read_binary_file(entry.path()))) {
      if (sirius::test::s3::skip_or_fail_unless(false, "managed S3 backend required for DV upload"))
        return;
    }
  }

  auto metadata_bytes = read_binary_file(local / "metadata/v1.metadata.json");
  std::string metadata(metadata_bytes.begin(), metadata_bytes.end());
  auto key = metadata.find("current-snapshot-id");
  REQUIRE(key != std::string::npos);
  auto colon = metadata.find(':', key);
  REQUIRE(colon != std::string::npos);
  auto snapshot  = std::stoll(metadata.substr(colon + 1));
  auto local_sql = "SELECT fruit, count FROM iceberg_scan(" +
                   sql_quote((local / "metadata/v1.metadata.json").string()) +
                   ", snapshot_from_id=" + std::to_string(snapshot) + ")";
  set_gpu_execution(con, false);
  auto local_result = require_query_ok(con, local_sql);
  auto expected     = collect_rows(*local_result);

  auto& context         = require_sirius_context(fixture);
  auto counters         = context.physical_counters();
  auto provider         = std::make_shared<sirius::scan_manager::test::test_reservation_provider>();
  provider->grant_bytes = 1024 * 1024;
  bool deferred_route   = false;
  std::string opened_path;
  struct reset_hooks {
    std::shared_ptr<sirius::op::scan::physical_check_counters> counters;
    ~reset_hooks()
    {
      counters->preparation_provider_for_testing.reset();
      counters->iceberg_preparation_route_for_testing = {};
      counters->iceberg_dv_phase_for_testing          = {};
    }
  } reset{counters};
  counters->preparation_provider_for_testing      = provider;
  counters->iceberg_preparation_route_for_testing = [&](auto, bool deferred) {
    deferred_route = deferred;
  };
  counters->iceberg_dv_phase_for_testing = [&](auto const& path, bool start) {
    if (start) opened_path = path;
  };

  set_gpu_execution(con, true);
  require_query_ok(con, "SET enable_duckdb_fallback=false");
  auto remote_sql = "SELECT fruit, count FROM iceberg_scan(" +
                    sql_quote(remote + "/dv_bounded/metadata/v1.metadata.json") +
                    ", snapshot_from_id=" + std::to_string(snapshot) + ")";
  auto remote_result = con.Query(remote_sql);
  REQUIRE(remote_result);
  REQUIRE_FALSE(remote_result->HasError());
  CHECK(collect_rows(*remote_result) == expected);
  CHECK(deferred_route);
  CHECK(opened_path == remote + "/dv_bounded/data/a.puffin");
}

TEST_CASE("S3 mapped Parquet normalizes IDs on the bulk reader",
          "[s3][integration][field_id_mapping]")
{
  using namespace sirius::op::scan;
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) return;
  s3_sql_fixture fixture(*env, large_sirius_memory_limits("cucs"));
  sirius::test::scratch_dir directory("s3_field_id_mapping");
  set_gpu_execution(fixture.con, false);
  require_query_ok(fixture.con,
                   "COPY (SELECT (range - 2)::INTEGER old_count FROM range(5)) TO " +
                     directory.file_literal("data.parquet") +
                     " (FORMAT PARQUET, FIELD_IDS {old_count: 2})");
  REQUIRE(sirius::test::put_s3_test_object("field-id-mapping/data.parquet",
                                           read_binary_file(directory.path() / "data.parquet")));
  auto uri    = s3_uri(env->bucket, "field-id-mapping/data.parquet");
  auto source = require_sirius_context(fixture).get_scan_manager().create_datasource(uri);
  REQUIRE(source);
  REQUIRE(source->prefers_bulk_io());
  auto options  = cudf::io::parquet_reader_options::builder().build();
  auto resolved = resolve_parquet_metadata_with_identity(*source, 71, uri, options);
  materialization_program program{
    71,
    1,
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}},
    0,
    false};
  program.input_types = program.output_types;
  auto mapping        = field_id_mapping::compile(
    resolved.artifact,
    resolved.identity,
    program,
    {{2, duckdb::LogicalType::BIGINT, {}, true},
            {3, duckdb::LogicalType::INTEGER, duckdb::Value::INTEGER(42), false}});
  bound_table_scan contract;
  contract.contract_id = 71;
  contract.profiles    = std::make_shared<physical_profile_table>();
  parquet_source input{source, resolved.artifact->file_metadata(), {0}};
  std::vector<std::size_t> groups{0};
  rmm::cuda_stream stream;
  auto batch = materialize_mapped_parquet(
    input, mapping, contract, groups, stream, cudf::get_current_device_resource_ref());
  stream.synchronize();
  REQUIRE(batch.rows == 5);
  std::array<int64_t, 5> values;
  std::array<int32_t, 5> defaults;
  REQUIRE(cudaMemcpy(values.data(),
                     batch.table.view().column(0).head<int64_t>(),
                     sizeof(values),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(defaults.data(),
                     batch.table.view().column(1).head<int32_t>(),
                     sizeof(defaults),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(values == std::array<int64_t, 5>{-2, -1, 0, 1, 2});
  CHECK(defaults == std::array<int32_t, 5>{42, 42, 42, 42, 42});
  REQUIRE(batch.lineage.runs.size() == 1);
  CHECK(batch.lineage.runs[0].file_row_offset == 0);
}

TEST_CASE("S3 metadata-only provenance skips pruned sources and data readers",
          "[s3][integration][virtual_columns][metadata_only]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) { return; }
  s3_sql_fixture fixture(*env);
  namespace scan  = sirius::op::scan;
  auto const uri  = s3_uri(env->bucket, "parquet/virtual_provenance.parquet");
  auto& context   = require_sirius_context(fixture);
  auto datasource = context.get_scan_manager().create_datasource(uri);
  REQUIRE(datasource);
  auto ioctx = datasource->io_ctx();
  auto* rest = dynamic_cast<cucascade::io::rest::rest_ioctx*>(ioctx.get());
  REQUIRE(rest);
  auto info                    = std::make_unique<scan::parquet_ingestible_table_info>();
  info->resolved_file_paths    = {s3_uri(env->bucket, "does-not-exist-pruned.parquet"), uri};
  info->names                  = {"marker", "keep"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::BIGINT),
                                  sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto const index_id          = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
  auto const row_id            = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  info->column_ids             = {duckdb::ColumnIndex(index_id), duckdb::ColumnIndex(row_id)};
  info->projected_output_count = 2;
  info->virtual_columns        = {
    {index_id, "file_index", sirius::logical_type::make(sirius::type_id::UBIGINT), std::nullopt},
    {row_id, "file_row_number", sirius::logical_type::make(sirius::type_id::BIGINT), std::nullopt}};
  info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(0),
    sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL,
                                  duckdb::Value::UBIGINT(1)));
  auto reader = scan::make_ingestible(std::move(info));
  std::vector<std::string> resolved;
  auto task = reader->next_split_provider([&](std::string_view path) {
    resolved.emplace_back(path);
    return ioctx;
  });
  REQUIRE(task);
  auto file = task({});
  REQUIRE(resolved == std::vector<std::string>{uri});
  REQUIRE(reader->has_processed_all_metadata());
  CHECK(file->fadvise_hints().empty());
  auto coalescer = reader->create_batch_coalescer();
  CHECK(coalescer->push(std::move(file)).empty());
  auto splits = coalescer->flush();
  REQUIRE(splits.size() == 1);
  CHECK(splits.front()->fadvise_hints().empty());
  auto const* space =
    context.get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto decoded = reader->materialize_metadata_to_table(
    *splits.front(), *space, cudf::get_default_stream(), false, nullptr);
  auto output = reader->post_filter_and_project(
    std::move(decoded), *space, cudf::get_default_stream(), false, nullptr, nullptr, {});
  REQUIRE(output);
  CHECK(output->num_rows() == 8193);
  CHECK(reader->reader_calls() == 0);
}

TEST_CASE("S3 Iceberg catalog table runs on the GPU with catalog-vended credentials",
          "[s3][integration][iceberg][catalog]")
{
  auto env = load_s3_test_env();
  if (should_skip_s3_env(env)) return;
  // No programmatic or static S3 credential: every read must use what the catalog vends.
  sirius_config_env_guard config_env(*env);
  duckdb::DBConfig db_config{{{"allow_unsigned_extensions", duckdb::Value::BOOLEAN(true)}}, false};
  duckdb::DuckDB db(nullptr, &db_config);
  duckdb::Connection con(db);
  load_sirius_extension(db);
  setenv("SIRIUS_DISABLE", "1", 1);
  for (auto const* extension : {"httpfs", "avro", "iceberg"}) {
    auto loaded = con.Query(std::string("LOAD ") + extension);
    if (sirius::test::s3::skip_or_fail_unless(
          loaded && !loaded->HasError(),
          "the httpfs, avro and iceberg extensions must be installed"))
      return;
  }

  // A committed local table, relocated under the bucket by rewriting its paths and uploaded.
  // @p omit, relative to the table root, is written locally and never uploaded.
  auto const relocate = [&](std::string const& table,
                            fs::path const& out,
                            std::string const& to,
                            std::string const& omit = {}) {
    auto const command = "python3 -B test/cpp/integration/data/generate_remote_table_fixture.py " +
                         table + " '" + out.string() + "' '" + to + "'";
    REQUIRE(std::system(command.c_str()) == 0);
    for (auto const& entry : fs::recursive_directory_iterator(out)) {
      if (!entry.is_regular_file()) continue;
      auto const relative = fs::relative(entry.path(), out).generic_string();
      if (relative == omit) continue;
      auto const key = to.substr(("s3://" + env->bucket + "/").size()) + "/" + relative;
      if (!sirius::test::put_s3_test_object(key, read_binary_file(entry.path()))) return false;
    }
    return true;
  };
  // Every hook a block of this test installs captures its locals: clear them all when the
  // block ends, however it ends.
  struct reset_hooks {
    std::shared_ptr<sirius::op::scan::physical_check_counters> counters;
    ~reset_hooks()
    {
      counters->iceberg_dv_phase_for_testing      = {};
      counters->iceberg_equality_open_for_testing = {};
      counters->after_lowering_for_testing        = {};
      counters->idle_rest_context_limit_for_testing.reset();
    }
  };
  auto const source = std::string{"test/cpp/integration/data/iceberg_v1"};
  auto const remote = "s3://" + env->bucket + "/catalog-form/fruit";
  sirius::test::scratch_dir directory("s3_catalog");
  if (sirius::test::s3::skip_or_fail_unless(
        relocate(source, directory.path(), remote),
        "managed S3 backend required for the catalog table upload"))
    return;

  std::map<std::string, std::string> vended{{"s3.access-key-id", env->access_key},
                                            {"s3.secret-access-key", env->secret_key},
                                            {"s3.endpoint", env->endpoint},
                                            {"s3.path-style-access", "true"},
                                            {"client.region", env->region}};
  if (!env->session_token.empty()) vended["s3.session-token"] = env->session_token;
  sirius::test::mock_iceberg_rest_catalog catalog(vended);
  auto const metadata = [&](char const* name) {
    return sirius::test::mock_iceberg_rest_catalog::table{
      remote + "/metadata/" + name, read_text_file(directory.path() / "metadata" / name)};
  };
  catalog.put("ns", "fruit", metadata("v1.metadata.json"));
  catalog.put("ns", "fruit_again", metadata("v1.metadata.json"));
  catalog.put("ns", "empty", metadata("empty.metadata.json"));
  {
    sirius::test::scoped_recording_log_sink logs{"info"};
    require_query_ok(con,
                     "ATTACH '' AS lake (TYPE ICEBERG, AUTHORIZATION_TYPE 'none', ENDPOINT '" +
                       catalog.uri() + "')");
    // A broader static secret with wrong credentials stays in place throughout: the vended
    // secret is more specific and must win, for the statement and for connections reading on
    // its behalf.
    // The nested comment sits between the two keywords that name the statement. DuckDB hands
    // the callback the statement with it in place, so a classifier that ends the comment at
    // the first "*/" reads `still` where SECRET stands and logs the whole text.
    require_query_ok(con,
                     "CREATE /* setup /* nested */ still a comment */ SECRET broad (TYPE S3, "
                     "KEY_ID 'wrong', SECRET 'synthetic-credential-5e1f', REGION " +
                       sql_quote(env->region) + ", ENDPOINT " + sql_quote(env->endpoint) +
                       ", URL_STYLE 'path', USE_SSL " +
                       (env->endpoint.rfind("https://", 0) == 0 ? "true" : "false") + ")");
    // Both statements hold credentials as literals: the log names their kind, not their text.
    auto const records = logs.records();
    auto const logged  = [&](std::string_view text) {
      return std::ranges::any_of(records, [&](auto const& record) {
        return record.message.find(text) != std::string::npos;
      });
    };
    CHECK(logged("SQL: ATTACH <text withheld"));
    CHECK(logged("SQL: CREATE SECRET <text withheld"));
    CHECK_FALSE(logged("AUTHORIZATION_TYPE"));
    CHECK_FALSE(logged("synthetic-credential-5e1f"));
  }

  using rows_t = std::vector<std::vector<std::string>>;
  rows_t const fruit{{"apple", "1"}, {"banana", "2"}, {"cherry", "3"}};
  auto const query = std::string{"SELECT fruit, count FROM lake.ns.fruit ORDER BY count"};
  auto const data_file =
    remote + "/data/00000-0-a1b2c3d4-0001-0001-0001-000000000003-00001.parquet";
  auto const no_snapshot =
    static_cast<size_t>(sirius::op::scan::verdict_reason::iceberg_no_snapshot_id);
  auto const stats = [&] { return sirius::test::get_transparent_execution_stats(con); };
  // One GPU execution reading only @p file, with @p walks delete-discovery walks and no fallback.
  auto const expect_gpu =
    [&](std::string const& sql, rows_t const& expected, uint64_t walks, std::string const& file) {
      INFO(sql);
      auto const before = stats();
      auto result       = require_query_ok(con, sql);
      auto const after  = stats();
      CHECK(collect_rows(*result) == expected);
      CHECK(after.executions == before.executions + 1);
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
      CHECK(after.iceberg_manifest_walks == before.iceberg_manifest_walks + walks);
      auto reads = after.parquet_reader_calls;
      for (auto const& [file, count] : before.parquet_reader_calls)
        reads[file] -= count;
      std::erase_if(reads, [](auto const& entry) { return entry.second == 0; });
      CHECK(reads.size() == 1);
      CHECK(reads.count(file) == 1);
    };
  // DuckDB without Sirius reads the committed local table these fixtures were relocated from.
  auto const cpu_rows = [&](std::string const& table) {
    duckdb::DuckDB cpu_db(nullptr, &db_config);
    duckdb::Connection cpu(cpu_db);
    require_query_ok(cpu, "LOAD avro");
    require_query_ok(cpu, "LOAD iceberg");
    return collect_rows(*require_query_ok(cpu,
                                          "SELECT fruit, count FROM iceberg_scan('" + table +
                                            "/metadata/v1.metadata.json') ORDER BY count"));
  };

  set_gpu_execution(con, true);
  require_query_ok(con, "SET enable_duckdb_fallback=false");
  auto const loads_before = catalog.load_table_requests();
  expect_gpu(query, fruit, 1, data_file);
  // The binding and the delete discovery each resolve the table through the catalog.
  CHECK(catalog.load_table_requests() - loads_before >= 2);
  CHECK(cpu_rows(source) == fruit);

  // httpfs is loaded only for the catalog client and the secret type: s3:// stays GPU-only.
  set_gpu_execution(con, false);
  auto refused = con.Query(query);
  REQUIRE(refused);
  REQUIRE(refused->HasError());
  CHECK(refused->GetError().find("S3 is GPU-only") != std::string::npos);
  // The refusal leaves the connection usable, and each table vends its own secret.
  set_gpu_execution(con, true);
  expect_gpu("SELECT count(*) FROM lake.ns.fruit a JOIN lake.ns.fruit_again b USING (fruit)",
             {{"3"}},
             2,
             data_file);
  // Those two tables share a location, so one vended secret serves both. A copy under another
  // prefix is a table of its own, and the catalog vends it the same credentials for another
  // region, which this store does not check. One statement reads both tables: each data file
  // then resolves, in the statement's transaction, to what its own table vended.
  {
    auto const sirius_ctx =
      con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);
    auto const counters = sirius_ctx->physical_counters();
    reset_hooks const reset{counters};
    auto const copy_remote = "s3://" + env->bucket + "/catalog-form-copy/fruit";
    sirius::test::scratch_dir copy_directory("s3_catalog_copy");
    REQUIRE(relocate(source, copy_directory.path(), copy_remote));
    auto const copy_file =
      copy_remote + "/data/00000-0-a1b2c3d4-0001-0001-0001-000000000003-00001.parquet";
    auto const vend_copy = [&](std::string const& region) {
      auto copy_vended             = vended;
      copy_vended["client.region"] = region;
      catalog.put("ns",
                  "fruit_copy",
                  {copy_remote + "/metadata/v1.metadata.json",
                   read_text_file(copy_directory.path() / "metadata/v1.metadata.json"),
                   std::move(copy_vended)});
    };
    auto const region_of = [&](std::string const& file) {
      return sirius::io::s3::resolve_duckdb_s3_secret(*con.context, file, {}).region;
    };
    vend_copy("eu-central-1");
    REQUIRE(env->region != "eu-central-1");
    require_query_ok(con, "BEGIN TRANSACTION");
    auto const before_both = stats();
    auto both              = require_query_ok(
      con, "SELECT count(*) FROM lake.ns.fruit a JOIN lake.ns.fruit_copy b USING (fruit)");
    CHECK(collect_rows(*both) == rows_t{{"3"}});
    CHECK(stats().executions == before_both.executions + 1);
    CHECK(region_of(data_file) == env->region);
    CHECK(region_of(copy_file) == "eu-central-1");
    require_query_ok(con, "COMMIT");

    // The catalog rotates what it vends for the copy between loads. Each statement reads with
    // the credentials of its own load, on a REST context of its own; the ones earlier
    // statements left behind are given back once the statement that follows has ended.
    counters->idle_rest_context_limit_for_testing = 1;
    for (auto const* region : {"eu-west-1", "ap-south-1", "eu-west-1"}) {
      INFO(region);
      vend_copy(region);
      require_query_ok(con, "BEGIN TRANSACTION");
      expect_gpu("SELECT fruit, count FROM lake.ns.fruit_copy ORDER BY count", fruit, 1, copy_file);
      CHECK(region_of(copy_file) == region);
      require_query_ok(con, "COMMIT");
      CHECK(sirius_ctx->get_scan_manager().rest_context_count() <= 1);
    }
  }
  require_query_ok(con, "SET sirius_s3_header_signing=true");
  expect_gpu(query, fruit, 1, data_file);
  require_query_ok(con, "SET sirius_s3_header_signing=false");

  // A V3 deletion-vector table: its manifests and its Puffin file are opened by path, after
  // the binding, with the vended credentials.
  {
    sirius::test::scratch_dir dv_directory("s3_catalog_dv");
    auto const dv_prefix = std::string{"catalog-form-dv"};
    auto const dv_remote = "s3://" + env->bucket + "/" + dv_prefix + "/dv_bounded";
    auto const generate  = "python3 -B test/cpp/integration/data/generate_inventory_fixtures.py '" +
                          dv_directory.path().string() + "' 's3://" + env->bucket + "/" +
                          dv_prefix + "'";
    REQUIRE(std::system(generate.c_str()) == 0);
    auto const local = dv_directory.path() / "dv_bounded";
    for (auto const& entry : fs::recursive_directory_iterator(local)) {
      if (!entry.is_regular_file()) continue;
      auto const key =
        dv_prefix + "/" + fs::relative(entry.path(), dv_directory.path()).generic_string();
      REQUIRE(sirius::test::put_s3_test_object(key, read_binary_file(entry.path())));
    }
    catalog.put("ns",
                "dv",
                {dv_remote + "/metadata/v1.metadata.json",
                 read_text_file(local / "metadata/v1.metadata.json")});
    rows_t const kept{{"apple", "1"}, {"cherry", "3"}, {"elderberry", "5"}};
    // Just before the statement opens its Puffin file, a second connection resolves the same
    // path. It sees only the broad secret, whose credentials cannot read the file; the statement
    // must still open it with the credentials it planned with.
    auto const sirius_ctx =
      con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);
    auto const counters = sirius_ctx->physical_counters();
    reset_hooks const reset{counters};
    duckdb::Connection other(db);
    std::string replaced_with;
    counters->iceberg_dv_phase_for_testing = [&](std::string const& puffin, bool start) {
      if (!start) return;
      auto config   = sirius::io::s3::resolve_duckdb_s3_secret(*other.context, puffin, {});
      replaced_with = config.access_key;
      sirius_ctx->get_scan_manager().install_s3_config(puffin, std::move(config));
    };
    expect_gpu(
      "SELECT fruit, count FROM lake.ns.dv ORDER BY count", kept, 1, dv_remote + "/data/a.parquet");
    counters->iceberg_dv_phase_for_testing = {};
    CHECK(replaced_with == "wrong");
    // The same for a data file. The hook runs on the planning thread once every scan of the
    // statement is lowered: by then the statement has installed and recorded the configuration
    // of every file it scans, and no scan has opened one. There the second connection resolves
    // the data file's path to the broad secret. The scan then opens the file with what the
    // statement recorded.
    replaced_with.clear();
    std::size_t lowered                  = 0;
    counters->after_lowering_for_testing = [&] {
      ++lowered;
      auto replaced = sirius::io::s3::resolve_duckdb_s3_secret(*other.context, data_file, {});
      replaced_with = replaced.access_key;
      sirius_ctx->get_scan_manager().install_s3_config(data_file, std::move(replaced));
    };
    auto const rebuilds_before = stats().execution_rebuilds;
    expect_gpu(query, fruit, 1, data_file);
    counters->after_lowering_for_testing = {};
    CHECK(lowered == 1);
    CHECK(replaced_with == "wrong");
    // The plan was not built a second time, which would have installed the good one again.
    CHECK(stats().execution_rebuilds == rebuilds_before);
    CHECK(cpu_rows("test/cpp/integration/data/iceberg_v3_deletion_vector") == kept);
  }

  // Equality deletes on S3. Both delete files are read while planning, with the credentials
  // this statement resolved for them.
  {
    auto const eq_source = std::string{"test/cpp/integration/data/iceberg_v2_eq_multi_delete_file"};
    auto const eq_remote = "s3://" + env->bucket + "/catalog-form-eq";
    sirius::test::scratch_dir eq_directory("s3_catalog_eq");
    REQUIRE(relocate(eq_source, eq_directory.path(), eq_remote));
    // No manifest lists this object. Opening it would fail the statement.
    std::array<std::uint8_t, 4> const not_parquet{'P', 'A', 'R', '1'};
    REQUIRE(sirius::test::put_s3_test_object("catalog-form-eq/data/delete-eq-unlisted.parquet",
                                             not_parquet));
    auto const eq_metadata = [&](char const* name) {
      return sirius::test::mock_iceberg_rest_catalog::table{
        eq_remote + "/metadata/" + name, read_text_file(eq_directory.path() / "metadata" / name)};
    };
    catalog.put("ns", "eq", eq_metadata("v1.metadata.json"));
    rows_t const kept{{"apple", "1"}, {"cherry", "3"}, {"elderberry", "5"}};
    auto const eq_query = std::string{"SELECT fruit, count FROM lake.ns.eq ORDER BY count"};
    auto const eq_data =
      eq_remote + "/data/00000-0-e5f6a7b8-0005-0005-0005-000000000001-00001.parquet";
    auto const eq_delete =
      std::string{"data/delete-eq-1-e5f6a7b8-0005-0005-0005-000000000002-00001.parquet"};
    auto const remote_declines = [&] {
      return stats().semantic_declines[static_cast<size_t>(
        sirius::op::scan::verdict_reason::iceberg_equality_remote)];
    };

    // Just before each delete file is opened, a second connection resolves its path to the
    // broad secret, which cannot read it. The statement opens it with what it planned with.
    auto const sirius_ctx =
      con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);
    auto const counters = sirius_ctx->physical_counters();
    reset_hooks const reset{counters};
    duckdb::Connection other(db);
    std::vector<std::string> replaced_with;
    counters->iceberg_equality_open_for_testing = [&](std::string const& path) {
      auto config = sirius::io::s3::resolve_duckdb_s3_secret(*other.context, path, {});
      replaced_with.push_back(config.access_key);
      sirius_ctx->get_scan_manager().install_s3_config(path, std::move(config));
    };
    auto const before          = stats();
    auto const declines_before = remote_declines();
    auto const eq_loads_before = catalog.load_table_requests();
    expect_gpu(eq_query, kept, 1, eq_data);
    counters->iceberg_equality_open_for_testing = {};
    CHECK(replaced_with == std::vector<std::string>{"wrong", "wrong"});
    CHECK(stats().iceberg_delete_payload_loads == before.iceberg_delete_payload_loads + 2);
    CHECK(remote_declines() == declines_before);
    // The binding and the delete lookup each ask the catalog once. Checking that the lookup
    // resolved the bound table asks it nothing more.
    CHECK(catalog.load_table_requests() - eq_loads_before == 2);
    CHECK(cpu_rows(eq_source) == kept);

    // The load holds the context that reads the delete files until it ends, and an execution
    // need not follow it: a load that fails ends the statement. With no idle context allowed,
    // the load gives its contexts back when it ends. When it succeeds, none is left once the
    // statement's scans are lowered, before any of them has opened a file. When it fails
    // before its second file, none is left after the statement.
    counters->idle_rest_context_limit_for_testing = 0;
    std::optional<std::size_t> contexts_after_load;
    counters->after_lowering_for_testing = [&] {
      contexts_after_load = sirius_ctx->get_scan_manager().rest_context_count();
    };
    auto const loads_before = stats().iceberg_delete_payload_loads;
    expect_gpu(eq_query, kept, 1, eq_data);
    counters->after_lowering_for_testing = {};
    CHECK(stats().iceberg_delete_payload_loads == loads_before + 2);
    REQUIRE(contexts_after_load);
    CHECK(*contexts_after_load == 0);
    std::size_t opened                          = 0;
    counters->iceberg_equality_open_for_testing = [&](std::string const&) {
      if (++opened == 2) { throw std::runtime_error("injected: second delete file"); }
    };
    auto const failed                           = con.Query(eq_query);
    counters->iceberg_equality_open_for_testing = {};
    REQUIRE(failed);
    CHECK(failed->HasError());
    CHECK(opened == 2);
    CHECK(sirius_ctx->get_scan_manager().rest_context_count() == 0);
    counters->idle_rest_context_limit_for_testing.reset();

    // The path form reads the same files with a static secret.
    require_query_ok(con,
                     "CREATE SECRET eq_static (TYPE S3, KEY_ID " + sql_quote(env->access_key) +
                       ", SECRET " + sql_quote(env->secret_key) + ", REGION " +
                       sql_quote(env->region) + ", ENDPOINT " + sql_quote(env->endpoint) +
                       ", URL_STYLE 'path', USE_SSL " +
                       (env->endpoint.rfind("https://", 0) == 0 ? "true" : "false") + ", SCOPE " +
                       sql_quote(eq_remote + "/") + ")");
    expect_gpu("SELECT fruit, count FROM iceberg_scan('" + eq_remote +
                 "/metadata/v1.metadata.json') ORDER BY count",
               kept,
               1,
               eq_data);
    require_query_ok(con, "DROP SECRET eq_static");

    // A delete file that is not the size its manifest records is refused before its keys are
    // decoded. The statement fails: the rows that file removes are not returned.
    auto const intact = read_binary_file(eq_directory.path() / eq_delete);
    REQUIRE(sirius::test::put_s3_test_object("catalog-form-eq/" + eq_delete,
                                             std::span(intact).first(intact.size() - 1)));
    auto truncated = con.Query(eq_query);
    REQUIRE(truncated);
    REQUIRE(truncated->HasError());
    CHECK(truncated->GetError().find("size differs from its manifest entry") != std::string::npos);
    REQUIRE(sirius::test::put_s3_test_object("catalog-form-eq/" + eq_delete, intact));
    expect_gpu(eq_query, kept, 1, eq_data);

    // A delete file that cannot be opened at all: the same table under a prefix where that
    // object was never uploaded. The statement fails. It does not go on without the file,
    // which would return the rows the file removes.
    {
      auto const missing_remote = "s3://" + env->bucket + "/catalog-form-eq-missing";
      sirius::test::scratch_dir missing_directory("s3_catalog_eq_missing");
      REQUIRE(relocate(eq_source, missing_directory.path(), missing_remote, eq_delete));
      catalog.put("ns",
                  "eq_missing",
                  {missing_remote + "/metadata/v1.metadata.json",
                   read_text_file(missing_directory.path() / "metadata/v1.metadata.json")});
      auto const before_missing = stats();
      auto missing = con.Query("SELECT fruit, count FROM lake.ns.eq_missing ORDER BY count");
      auto const after_missing = stats();
      REQUIRE(missing);
      INFO((missing->HasError() ? missing->GetError() : "returned rows"));
      REQUIRE(missing->HasError());
      CHECK(missing->GetError().find(fs::path(eq_delete).filename().string()) != std::string::npos);
      CHECK(after_missing.executions == before_missing.executions);
    }

    // Table names that only work quoted. The delete lookup addresses the table by its qualified
    // name on a connection of its own, where that name is split at dots outside quotes: with a
    // dot in it, an unquoted name is another table's, and a quote in it must be escaped. A
    // keyword or a space would not do: they split the same with and without quotes.
    for (auto const* name : {"fruit.v2", "the \"fruit\" table"}) {
      INFO(name);
      catalog.put("ns", name, eq_metadata("v1.metadata.json"));
      expect_gpu("SELECT fruit, count FROM lake.ns." +
                   duckdb::KeywordHelper::WriteQuotedAndEscaped(name, '"') + " ORDER BY count",
                 kept,
                 1,
                 eq_data);
    }

    // The table replaced under its name once the statement has bound it: another table UUID, a
    // snapshot of the same ID over the same data file, and no deletes. The delete lookup
    // resolves the name to the replacement on its own connection. Its inventory would return
    // the rows the bound table deleted, so the statement is refused instead.
    auto const unverified = static_cast<size_t>(sirius::op::scan::verdict_reason::evidence_missing);
    catalog.replace_after_next_load("ns", "eq", eq_metadata("replaced.metadata.json"));
    auto const before_replaced = stats();
    auto replaced              = con.Query(eq_query);
    auto const after_replaced  = stats();
    REQUIRE(replaced);
    REQUIRE(replaced->HasError());
    // An Iceberg build that accepts expected_table_uuid on iceberg_metadata() refuses the lookup
    // itself; with any other, Sirius's comparison does. Both say "Table UUID mismatch", and
    // Sirius's also says what the name now resolves to.
    // A catalog function over nested columns: not a query for the GPU.
    set_gpu_execution(con, false);
    auto const uuid_probe = require_query_ok(
      con,
      "SELECT count(*) FROM duckdb_functions() WHERE function_name = 'iceberg_metadata' AND "
      "list_contains(parameters, 'expected_table_uuid')");
    set_gpu_execution(con, true);
    bool const extension_checks_uuid = uuid_probe->Fetch()->GetValue(0, 0).GetValue<int64_t>() > 0;
    auto const mismatch              = replaced->GetError();
    CAPTURE(extension_checks_uuid, mismatch);
    CHECK(mismatch.find("Table UUID mismatch") != std::string::npos);
    if (!extension_checks_uuid) {
      CHECK(mismatch.find("now resolves to table 00000000-0000-4000-8000-0000000000ff") !=
            std::string::npos);
      CHECK(mismatch.find("would be read from another table") != std::string::npos);
      CHECK(mismatch.find("could not read this table's metadata") == std::string::npos);
    }
    CHECK(after_replaced.semantic_declines[unverified] ==
          before_replaced.semantic_declines[unverified] + 1);
    CHECK(after_replaced.executions == before_replaced.executions);
    // Bound afresh, the replacement is a table of its own, without deletes.
    expect_gpu(
      eq_query,
      {{"apple", "1"}, {"banana", "2"}, {"cherry", "3"}, {"date", "4"}, {"elderberry", "5"}},
      1,
      eq_data);

    // The table dropped once the statement has bound it. The delete lookup finds no table under
    // the name: which table it would read is unknown, not known to be another one, and the
    // refusal says that rather than reporting a mismatch. The table sits in a namespace of its
    // own: this catalog answers 404 for a namespace as well, and the client then takes the
    // whole namespace for missing.
    catalog.put("dropped", "eq", eq_metadata("v1.metadata.json"));
    catalog.remove_after_next_load("dropped", "eq");
    auto const before_dropped = stats();
    auto dropped             = con.Query("SELECT fruit, count FROM lake.dropped.eq ORDER BY count");
    auto const after_dropped = stats();
    REQUIRE(dropped);
    REQUIRE(dropped->HasError());
    auto const unknown = dropped->GetError();
    CAPTURE(unknown);
    if (!extension_checks_uuid) {
      CHECK(unknown.find("could not identify the table") != std::string::npos);
      CHECK(unknown.find("Table UUID mismatch") == std::string::npos);
      CHECK(unknown.find("another table") == std::string::npos);
    }
    CHECK(after_dropped.semantic_declines[unverified] ==
          before_dropped.semantic_declines[unverified] + 1);
    CHECK(after_dropped.executions == before_dropped.executions);
  }

  // An internal connection reads by path with the secret of the statement that opened it; an
  // ordinary second connection has no credential for the same path.
  auto const manifest = remote + "/metadata/a1b2c3d4-0001-0001-0001-000000000002-m0.avro";
  auto const by_path  = "SELECT count(*) FROM read_avro('" + manifest + "')";
  require_query_ok(con, "BEGIN TRANSACTION");
  expect_gpu(query, fruit, 1, data_file);
  {
    auto internal = duckdb::SiriusContext::open_internal_connection(*con.context);
    auto read     = internal.Query(by_path);
    REQUIRE(read);
    INFO((read->HasError() ? read->GetError() : ""));
    REQUIRE_FALSE(read->HasError());
    CHECK(read->Fetch()->GetValue(0, 0).GetValue<int64_t>() == 1);
    duckdb::Connection unrelated(db);
    auto denied = unrelated.Query(by_path);
    REQUIRE(denied);
    CHECK(denied->HasError());
    // What it resolves for the path is the broad secret, not what the catalog vended.
    CHECK(sirius::io::s3::resolve_duckdb_s3_secret(*unrelated.context, manifest, {}).access_key ==
          "wrong");
  }
  // A connection reading for this statement resolves its vended secret and signs the same way.
  require_query_ok(con, "SET sirius_s3_header_signing=true");
  {
    duckdb::Connection reader(db);
    auto const state = duckdb::get_sirius_connection_state(*reader.context);
    REQUIRE(state);
    state->set_internal_outer(con.context);
    auto const resolved = sirius::io::s3::resolve_duckdb_s3_secret(*reader.context, manifest, {});
    CHECK(resolved.access_key == env->access_key);
    CHECK(resolved.s3_signing_mode == cucascade::io::object_store_config::signing_mode::header);
  }
  require_query_ok(con, "SET sirius_s3_header_signing=false");
  require_query_ok(con, "COMMIT");

  // The lookup accepts scan info only on a catalog table's binding, with the extension's callbacks.
  con.BeginTransaction();
  auto logical = con.ExtractPlan(query);
  auto* node   = logical.get();
  while (node->type != duckdb::LogicalOperatorType::LOGICAL_GET)
    node = node->children.front().get();
  auto& get = node->Cast<duckdb::LogicalGet>();
  REQUIRE(get.function.function_info);
  CHECK(sirius::planner::lookup_connector(get, *con.context) != nullptr);
  auto const table_entry       = get.function.get_table_entry;
  get.function.get_table_entry = nullptr;
  CHECK(sirius::planner::lookup_connector(get, *con.context) == nullptr);
  get.function.get_table_entry = table_entry;
  // A callback that is not the extension's is compared with the registered one, never called.
  static int foreign_bind_info_calls = 0;
  auto const bind_info               = get.function.get_bind_info;
  get.function.get_bind_info         = [](duckdb::TableFunctionGetBindInfoInput&) {
    ++foreign_bind_info_calls;
    return duckdb::BindInfo(duckdb::ScanType::TABLE);
  };
  CHECK(sirius::planner::lookup_connector(get, *con.context) == nullptr);
  CHECK(foreign_bind_info_calls == 0);
  get.function.get_bind_info = bind_info;
  // Scan info of another type on a genuine table binding: the scan kind alone would accept it.
  struct foreign_info : duckdb::TableFunctionInfo {};
  auto const scan_info       = get.function.function_info;
  get.function.function_info = duckdb::make_shared_ptr<foreign_info>();
  CHECK(sirius::planner::lookup_connector(get, *con.context) == nullptr);
  get.function.function_info = scan_info;
  CHECK(sirius::planner::lookup_connector(get, *con.context) != nullptr);
  // Two genuine catalog bindings carry scan info of one type, which the hook cannot tell apart.
  // Nothing Sirius reads comes from it: with the two infos swapped, the lookup accepts both
  // bindings, and each still names the table, snapshot and files it was bound to. Its delete
  // lookup addresses that table and finds the UUID it bound.
  {
    auto other_plan = con.ExtractPlan("SELECT fruit, count FROM lake.ns.dv ORDER BY count");
    auto* other     = other_plan.get();
    while (other->type != duckdb::LogicalOperatorType::LOGICAL_GET)
      other = other->children.front().get();
    auto& other_get = other->Cast<duckdb::LogicalGet>();
    struct bound_to {
      std::string table;
      std::string table_uuid;
      int64_t snapshot_id;
      std::string read_view;
      std::optional<std::string> selector;
      std::string inventory_table;
      std::vector<std::string> data_files;
      bool operator==(bound_to const&) const = default;
    };
    auto const bound = [&](duckdb::LogicalGet& scan) {
      auto const* connector = sirius::planner::lookup_connector(scan, *con.context);
      REQUIRE(connector);
      auto const snapshot = sirius::op::scan::capture_iceberg_snapshot(
        scan.function, scan.bind_data.get(), *con.context);
      REQUIRE(snapshot);
      auto const view = sirius::op::scan::capture_bound_read_view(scan, *con.context);
      REQUIRE(view.identity);
      sirius::planner::scan_contract_provenance provenance;
      auto const pre = connector->decline_reason(scan, *con.context, provenance);
      INFO((pre.decline ? pre.decline->text : ""));
      REQUIRE_FALSE(pre.decline);
      REQUIRE(pre.inventory);
      bound_to result{scan.GetTable()->name.GetIdentifierName(),
                      snapshot->table_uuid,
                      snapshot->snapshot_id,
                      sirius::op::scan::canonical_read_view_text(view),
                      view.logical_selector_evidence,
                      pre.inventory->table,
                      {}};
      for (auto const& entry : pre.inventory->entries)
        if (entry.content == "DATA") result.data_files.push_back(entry.file_path);
      std::ranges::sort(result.data_files);
      return result;
    };
    auto const fruit_bound = bound(get);
    auto const other_bound = bound(other_get);
    CHECK(fruit_bound.table == "fruit");
    CHECK(fruit_bound.inventory_table == R"("lake"."ns"."fruit")");
    CHECK(fruit_bound.data_files == std::vector<std::string>{data_file});
    CHECK(other_bound.table == "dv");
    CHECK(other_bound.table_uuid != fruit_bound.table_uuid);
    CHECK(other_bound.data_files != fruit_bound.data_files);
    REQUIRE(other_get.function.function_info);
    REQUIRE(other_get.function.function_info != get.function.function_info);
    std::swap(get.function.function_info, other_get.function.function_info);
    CHECK(bound(get) == fruit_bound);
    CHECK(bound(other_get) == other_bound);
    std::swap(get.function.function_info, other_get.function.function_info);
  }
  logical.reset();
  con.Rollback();
  con.BeginTransaction();
  auto by_path_plan =
    con.ExtractPlan("SELECT fruit FROM iceberg_scan('" + source + "/metadata/v1.metadata.json')");
  node = by_path_plan.get();
  while (node->type != duckdb::LogicalOperatorType::LOGICAL_GET)
    node = node->children.front().get();
  auto& path_get = node->Cast<duckdb::LogicalGet>();
  CHECK(sirius::planner::lookup_connector(path_get, *con.context) != nullptr);
  path_get.function.function_info = scan_info;
  CHECK(sirius::planner::lookup_connector(path_get, *con.context) == nullptr);
  by_path_plan.reset();
  con.Rollback();

  // A table whose only snapshot is retained and not current. Read by version, the statement
  // binds that snapshot; the delete lookup resolves the name without a version and binds none.
  // The lookup still learns which table it reached, so the scan is not refused.
  catalog.put("ns", "detached", metadata("detached.metadata.json"));
  auto const mismatches_before =
    stats()
      .semantic_declines[static_cast<size_t>(sirius::op::scan::verdict_reason::evidence_missing)];
  expect_gpu(
    "SELECT fruit, count FROM lake.ns.detached AT (VERSION => 1000000000000000001) ORDER BY count",
    fruit,
    1,
    data_file);
  CHECK(stats().semantic_declines[static_cast<size_t>(
          sirius::op::scan::verdict_reason::evidence_missing)] == mismatches_before);

  // A table without a snapshot publishes no snapshot identity, so the GPU path declines it.
  require_query_ok(con, "SET enable_duckdb_fallback=true");
  auto const before_empty = stats();
  auto empty             = require_query_ok(con, "SELECT fruit FROM lake.ns.empty WHERE count > 1");
  auto const after_empty = stats();
  CHECK(collect_rows(*empty).empty());
  CHECK(after_empty.semantic_declines[no_snapshot] ==
        before_empty.semantic_declines[no_snapshot] + 1);
  CHECK(after_empty.executions == before_empty.executions);
  // Without the fallback the refusal names the table, not arguments the query never wrote.
  require_query_ok(con, "SET enable_duckdb_fallback=false");
  auto unread = con.Query("SELECT fruit FROM lake.ns.empty WHERE count > 1");
  REQUIRE(unread->HasError());
  CHECK(unread->GetError().find(R"(Iceberg table "lake"."ns"."empty" has no committed snapshot)") !=
        std::string::npos);
  CHECK(unread->GetError().find("snapshot_from_id") == std::string::npos);
}
