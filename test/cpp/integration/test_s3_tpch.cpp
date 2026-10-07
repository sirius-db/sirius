/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "utils/s3_backend.hpp"
#include "utils/s3_test_env.hpp"
#include "utils/tpch_queries.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <duckdb.hpp>

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
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using sirius::test::s3::env_or;
using sirius::test::s3::sql_quote;

namespace fs = std::filesystem;

constexpr std::array<std::string_view, 8> kS3TpchTables = {
  "nation", "region", "customer", "orders", "part", "partsupp", "supplier", "lineitem"};

struct s3_tpch_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::string session_token;
  fs::path tiny_local_dir;
  fs::path sf1_local_dir;
};

std::optional<s3_tpch_env> load_s3_tpch_env()
{
  if (!sirius::test::ensure_s3_test_env()) return std::nullopt;

  auto endpoint   = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket     = env_or("SIRIUS_TEST_S3_BUCKET");
  auto local_dir  = env_or("SIRIUS_TEST_S3_LOCAL_DIR");
  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty() ||
      local_dir.empty()) {
    return std::nullopt;
  }

  return s3_tpch_env{std::move(endpoint),
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket),
                     env_or("SIRIUS_TEST_S3_SESSION_TOKEN"),
                     fs::path{std::move(local_dir)} / "parquet",
                     fs::path{env_or("SIRIUS_TEST_S3_TPCH_LOCAL_DIR")}};
}

bool should_skip_s3_tpch_env(std::optional<s3_tpch_env> const& env)
{
  return sirius::test::s3::skip_or_fail_unless(env.has_value(),
                                               "SIRIUS_TEST_S3_* environment is not configured");
}

class s3_tpch_config_guard {
 public:
  s3_tpch_config_guard(s3_tpch_env const& /*env*/, bool sf1)
  {
    if (auto const* current = std::getenv("SIRIUS_CONFIG_FILE"); current != nullptr) {
      original_config_ = current;
    }
    if (auto const* current = std::getenv("SIRIUS_DISABLE"); current != nullptr) {
      original_disable_ = current;
    }

    auto const unique = std::to_string(reinterpret_cast<std::uintptr_t>(this));
    dir_              = fs::temp_directory_path() / ("sirius_s3_tpch_" + unique);
    config_path_      = dir_ / "sirius.yaml";
    fs::create_directories(dir_);

    auto const gpu_capacity  = sf1 ? "2 GiB" : "512 MiB";
    auto const host_capacity = sf1 ? "4 GiB" : "1 GiB";
    auto const disk_capacity = sf1 ? "16 GiB" : "2 GiB";
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
        << gpu_capacity
        << "\n"
           "    host:\n"
           "      - numa_id: -1\n"
           "        reservation_limit_fraction: 0.9\n"
           "        downgrade_trigger_fraction: 0.8\n"
           "        downgrade_stop_fraction: 0.6\n"
           "        memory_capacity: "
        << host_capacity
        << "\n"
           "        block_size: 1 MiB\n"
           "    disk:\n"
           "      - disk_id: 0\n"
           "        mount_path: "
        << sql_quote((dir_ / "disk_memory").string())
        << "\n"
           "        memory_capacity: "
        << disk_capacity
        << "\n"
           "  executor:\n"
           "    scan_manager:\n"
           "      rest:\n"
           "        request_timeout_s: 30\n";
    out.close();
    REQUIRE(out);

    setenv("SIRIUS_CONFIG_FILE", config_path_.c_str(), /*overwrite=*/1);
    unsetenv("SIRIUS_DISABLE");
  }

  ~s3_tpch_config_guard()
  {
    if (original_config_.has_value()) {
      setenv("SIRIUS_CONFIG_FILE", original_config_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv("SIRIUS_CONFIG_FILE");
    }
    if (original_disable_.has_value()) {
      setenv("SIRIUS_DISABLE", original_disable_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv("SIRIUS_DISABLE");
    }
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

 private:
  fs::path dir_;
  fs::path config_path_;
  std::optional<std::string> original_config_;
  std::optional<std::string> original_disable_;
};

void tpch_require_query_ok(duckdb::Connection& connection, std::string const& sql)
{
  auto result = connection.Query(sql);
  REQUIRE(result);
  if (result->HasError()) UNSCOPED_INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
}

void tpch_load_sirius_extension(duckdb::DuckDB& db)
{
  try {
    db.LoadStaticExtension<duckdb::SiriusExtension>();
  } catch (std::exception const& error) {
    auto const message = std::string{error.what()};
    if (message.find("already exists") == std::string::npos &&
        message.find("already loaded") == std::string::npos) {
      throw;
    }
  }
}

std::vector<std::vector<std::string>> tpch_collect_rows(duckdb::MaterializedQueryResult& result)
{
  std::vector<std::vector<std::string>> rows;
  rows.reserve(result.RowCount());
  for (duckdb::idx_t row_index = 0; row_index < result.RowCount(); ++row_index) {
    std::vector<std::string> row;
    row.reserve(result.ColumnCount());
    for (duckdb::idx_t column_index = 0; column_index < result.ColumnCount(); ++column_index) {
      row.push_back(result.GetValue(column_index, row_index).ToString());
    }
    rows.push_back(std::move(row));
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

bool tpch_is_floating_point(duckdb::LogicalTypeId type)
{
  return type == duckdb::LogicalTypeId::FLOAT || type == duckdb::LogicalTypeId::DOUBLE;
}

class s3_tpch_suite {
 public:
  s3_tpch_suite(s3_tpch_env const& env, std::string object_prefix, fs::path local_dir, bool sf1)
    : env_(env),
      object_prefix_(std::move(object_prefix)),
      local_dir_(std::move(local_dir)),
      config_guard_(env, sf1),
      cpu_db_(nullptr),
      cpu_connection_(cpu_db_)
  {
    REQUIRE_FALSE(local_dir_.empty());
    create_local_views();
    reset_gpu_catalog();
  }

  void run_all_queries()
  {
    for (auto const& query : sirius::test::kTpchQueries) {
      INFO("TPC-H Q" << query.number);
      if (!query.retry_once) {
        compare_query(query);
        continue;
      }

      try {
        compare_query(query);
      } catch (std::exception const& first_error) {
        WARN("TPC-H Q" << query.number
                       << " first attempt failed; retrying once: " << first_error.what());
        reset_gpu_catalog();
        compare_query(query);
      }
    }
  }

  void require_gpu_off_rejected()
  {
    tpch_require_query_ok(*gpu_connection_, "SET gpu_execution = false;");
    auto result = gpu_connection_->Query("SELECT count(*) FROM nation");
    REQUIRE(result);
    REQUIRE(result->HasError());
    INFO(result->GetError());
    CHECK(result->GetError().find("S3 is GPU-only") != std::string::npos);
    tpch_require_query_ok(*gpu_connection_, "SET gpu_execution = true;");
  }

 private:
  void create_views(duckdb::Connection& connection, std::string const& root, bool s3)
  {
    for (auto const table : kS3TpchTables) {
      auto const path = s3 ? root + "/" + std::string{table} + ".parquet"
                           : (fs::path{root} / (std::string{table} + ".parquet")).string();
      tpch_require_query_ok(connection,
                            "CREATE OR REPLACE VIEW " + std::string{table} +
                              " AS SELECT * FROM read_parquet(" + sql_quote(path) + ");");
    }
  }

  void create_local_views()
  {
    for (auto const table : kS3TpchTables) {
      auto const parquet = local_dir_ / (std::string{table} + ".parquet");
      REQUIRE(fs::exists(parquet));
    }
    create_views(cpu_connection_, local_dir_.string(), false);
  }

  void reset_gpu_catalog()
  {
    gpu_connection_.reset();
    gpu_db_.reset();
    gpu_db_ = std::make_unique<duckdb::DuckDB>(nullptr);
    tpch_load_sirius_extension(*gpu_db_);
    gpu_connection_ = std::make_unique<duckdb::Connection>(*gpu_db_);
    auto context =
      gpu_connection_->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(context);
    sirius::io::object_store_config object_store;
    object_store.endpoint      = env_.endpoint;
    object_store.region        = env_.region;
    object_store.access_key    = env_.access_key;
    object_store.secret_key    = env_.secret_key;
    object_store.session_token = env_.session_token;
    object_store.tls_verify    = false;
    context->get_config().set_object_store_config(object_store);
    context->get_scan_manager().install_s3_config("s3://" + env_.bucket, object_store);
    auto const root = "s3://" + env_.bucket + "/" + object_prefix_;
    create_views(*gpu_connection_, root, true);
    tpch_require_query_ok(*gpu_connection_, "SET gpu_execution = true;");
  }

  std::unique_ptr<duckdb::MaterializedQueryResult> run_gpu_query(std::string const& sql)
  {
    auto result = gpu_connection_->Query(sql);
    if (!result) throw std::runtime_error("GPU query returned no result");
    if (result->HasError()) throw std::runtime_error(result->GetError());
    return result;
  }

  void compare_query(sirius::test::tpch_query_spec const& query)
  {
    auto const before = sirius::test::get_transparent_execution_stats(*gpu_connection_);
    auto gpu_result   = run_gpu_query(std::string{query.sql});
    auto const after  = sirius::test::get_transparent_execution_stats(*gpu_connection_);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);

    auto cpu_result = cpu_connection_.Query(std::string{query.sql});
    REQUIRE(cpu_result);
    if (cpu_result->HasError()) UNSCOPED_INFO(cpu_result->GetError());
    REQUIRE_FALSE(cpu_result->HasError());
    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    std::vector<bool> floating_columns(gpu_result->ColumnCount());
    for (duckdb::idx_t column = 0; column < gpu_result->ColumnCount(); ++column) {
      floating_columns[column] = tpch_is_floating_point(gpu_result->types[column].id());
    }

    auto gpu_rows = tpch_collect_rows(*gpu_result);
    auto cpu_rows = tpch_collect_rows(cpu_result->Cast<duckdb::MaterializedQueryResult>());
    REQUIRE(gpu_rows.size() == cpu_rows.size());
    for (std::size_t row = 0; row < gpu_rows.size(); ++row) {
      REQUIRE(gpu_rows[row].size() == cpu_rows[row].size());
      for (std::size_t column = 0; column < gpu_rows[row].size(); ++column) {
        if (query.float_tolerance.has_value() && floating_columns[column]) {
          auto const gpu_value = std::stod(gpu_rows[row][column]);
          auto const cpu_value = std::stod(cpu_rows[row][column]);
          CHECK(std::fabs(gpu_value - cpu_value) <= *query.float_tolerance);
        } else {
          CHECK(gpu_rows[row][column] == cpu_rows[row][column]);
        }
      }
    }
  }

  s3_tpch_env const& env_;
  std::string object_prefix_;
  fs::path local_dir_;
  s3_tpch_config_guard config_guard_;
  // Outlive the GPU database: each Sirius context restores the device memory
  // resource that was current when it was created.
  duckdb::DuckDB cpu_db_;
  duckdb::Connection cpu_connection_;
  std::unique_ptr<duckdb::DuckDB> gpu_db_;
  std::unique_ptr<duckdb::Connection> gpu_connection_;
};

}  // namespace

TEST_CASE("transparent S3 TPC-H Q1-Q22 match the tiny local CPU oracle",
          "[.][s3][integration][sql][tpch]")
{
  auto env = load_s3_tpch_env();
  if (should_skip_s3_tpch_env(env)) return;

  s3_tpch_suite suite(*env, "parquet", env->tiny_local_dir, false);
  suite.run_all_queries();
  suite.require_gpu_off_rejected();
}

TEST_CASE("transparent S3 TPC-H Q1-Q22 match the SF1 local CPU oracle",
          "[.][s3][integration][sql][tpch][large]")
{
  if (sirius::test::s3::skip_or_fail_unless(sirius::test::s3::truthy_env("SIRIUS_TEST_S3_TPCH"),
                                            "SIRIUS_TEST_S3_TPCH is not enabled")) {
    return;
  }

  auto env = load_s3_tpch_env();
  if (should_skip_s3_tpch_env(env)) return;
  REQUIRE_FALSE(env->sf1_local_dir.empty());

  s3_tpch_suite suite(*env, "tpch/sf1", env->sf1_local_dir, true);
  suite.run_all_queries();
}
