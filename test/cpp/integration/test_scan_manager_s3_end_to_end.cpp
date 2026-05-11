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

#include "catch.hpp"
#include "exec/thread_pool.hpp"
#include "helper/logical_type.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/sirius_sigv4_credential_provider.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"
#include "operator/operator_test_utils.hpp"
#include "scan_manager/parquet_split_provider.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>

#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::sirius_sigv4_credential_provider;
using sirius::io::s3::static_credentials;
using sirius::op::scan::parquet_scan_data;
using sirius::scan_manager::parquet_split_provider;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;
using sirius::scan_manager::split_connector;
using sirius::test::operator_utils::copy_column_to_host;

namespace {

using namespace std::chrono_literals;

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
  bool strict{false};
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
                     truthy_env("SIRIUS_TEST_S3_STRICT")};
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

s3_ioctx_config make_s3_config(s3_test_env const& env, bool bad_credentials = false)
{
  static_credentials creds;
  creds.access_key_id     = bad_credentials ? "definitely-wrong" : env.access_key;
  creds.secret_access_key = bad_credentials ? "definitely-wrong" : env.secret_key;
  auto provider           = std::make_shared<sirius_sigv4_credential_provider>(
    std::move(creds), env.region, env.endpoint, 30min);

  s3_ioctx_config cfg{};
  cfg.creds              = std::move(provider);
  cfg.max_connections    = 4;
  cfg.request_timeout_s  = 20;
  cfg.max_retry_attempts = 3;
  cfg.retry_backoff_base = 10ms;
  cfg.retry_jitter       = 0ms;
  cfg.honor_retry_after  = false;
  return cfg;
}

sirius_scan_manager make_scan_manager(s3_test_env const& env, bool bad_credentials = false)
{
  scan_manager_config cfg{};
  cfg.s3_config                         = make_s3_config(env, bad_credentials);
  cfg.s3_thread_pool.num_threads        = 4;
  cfg.s3_thread_pool.thread_name_prefix = "s3_e2e";
  return sirius_scan_manager(std::move(cfg));
}

duckdb::vector<duckdb::ColumnIndex> all_column_ids(std::size_t n)
{
  duckdb::vector<duckdb::ColumnIndex> ids;
  ids.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    ids.emplace_back(duckdb::ColumnIndex(i));
  }
  return ids;
}

duckdb::vector<sirius::logical_type> nation_returned_types()
{
  return {sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::VARCHAR),
          sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::VARCHAR)};
}

parquet_split_provider make_provider(std::vector<std::string> paths, sirius_scan_manager& manager)
{
  auto returned_types = nation_returned_types();
  auto const arity    = returned_types.size();
  return parquet_split_provider(returned_types,
                                paths,
                                all_column_ids(arity),
                                /*projection_ids*/ {},
                                /*names*/ {},
                                arity,
                                /*table_filter_set*/ nullptr,
                                /*partition_indices*/ {},
                                /*approximate_batch_size*/ std::size_t{1} << 30,
                                parquet_split_provider::DEFAULT_MAX_FILE_PROCESSED,
                                manager);
}

std::vector<std::unique_ptr<parquet_scan_data>> drive_provider(parquet_split_provider& provider)
{
  sirius::exec::static_thread_pool pool(2, "s3_e2e_psp");
  split_connector connector;
  provider.run(pool, connector);

  std::vector<std::unique_ptr<parquet_scan_data>> drained;
  while (true) {
    auto next = connector.get_next_split();
    if (!next.has_value()) { break; }
    std::unique_ptr<sirius::op::operator_data> base = std::move(*next);
    auto* raw                                       = base.release();
    auto* parquet                                   = dynamic_cast<parquet_scan_data*>(raw);
    REQUIRE(parquet != nullptr);
    drained.emplace_back(std::unique_ptr<parquet_scan_data>(parquet));
  }
  return drained;
}

cudf::io::table_with_metadata read_parquet_through_scan_manager(sirius_scan_manager& manager,
                                                                std::string const& path)
{
  auto* io_ctx = manager.io_ctx_for(path);
  REQUIRE(io_ctx != nullptr);

  auto io_object = io_ctx->create_io_object(path);
  REQUIRE(io_object != nullptr);
  CHECK(io_object->object_path() == path);

  auto datasource = io_ctx->make_datasource(io_object);
  auto options    = cudf::io::parquet_reader_options::builder().build();
  options.set_source(cudf::io::source_info{datasource.get()});
  return cudf::io::read_parquet(options);
}

bool saw_scheme(std::vector<std::unique_ptr<parquet_scan_data>> const& splits,
                std::string_view scheme)
{
  for (auto const& split : splits) {
    for (auto const& slice : split->rg_slices) {
      if (slice.file_path.rfind(scheme, 0) == 0) { return true; }
    }
  }
  return false;
}

}  // namespace

TEST_CASE("scan_manager S3 end-to-end reads nation parquet through sirius_datasource",
          "[scan_manager][s3][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping scan_manager S3 end-to-end test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  auto manager = make_scan_manager(*env);
  auto path    = s3_uri(env->bucket, "parquet/nation.parquet");

  auto result = read_parquet_through_scan_manager(manager, path);
  REQUIRE(result.tbl != nullptr);
  auto view = result.tbl->view();

  REQUIRE(view.num_rows() == 25);
  REQUIRE(view.num_columns() == 4);

  auto nation_keys = copy_column_to_host<int32_t>(view.column(0));
  auto names       = copy_column_to_host<std::string>(view.column(1));
  auto region_keys = copy_column_to_host<int32_t>(view.column(2));

  CHECK(nation_keys.front() == 0);
  CHECK(names.front() == "ALGERIA");
  CHECK(region_keys.front() == 0);
  CHECK(nation_keys.back() == 24);
  CHECK(names.back() == "UNITED STATES");
  CHECK(region_keys.back() == 1);
}

TEST_CASE("scan_manager S3 end-to-end routes parquet_split_provider through S3 ioctx",
          "[scan_manager][s3][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping scan_manager S3 split-provider test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  auto manager  = make_scan_manager(*env);
  auto path     = s3_uri(env->bucket, "parquet/nation.parquet");
  auto provider = make_provider({path}, manager);
  auto splits   = drive_provider(provider);

  REQUIRE_FALSE(splits.empty());
  for (auto const& split : splits) {
    for (auto const& slice : split->rg_slices) {
      CHECK(slice.file_path == path);
      CHECK(slice.io_ctx.get() == manager.io_ctx_for(path));
      CHECK(dynamic_cast<s3_ioctx*>(slice.io_ctx.get()) != nullptr);
      CHECK(slice.io_object != nullptr);
    }
  }
}

TEST_CASE("scan_manager S3 end-to-end dispatches mixed local and S3 parquet paths",
          "[scan_manager][s3][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping scan_manager mixed S3 test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  auto manager    = make_scan_manager(*env);
  auto local_path = std::filesystem::absolute("test/cpp/integration/data/parquet/nation.parquet");
  auto local_uri  = "file://" + local_path.string();
  auto s3_path    = s3_uri(env->bucket, "parquet/nation.parquet");

  auto provider = make_provider({local_uri, s3_path}, manager);
  auto splits   = drive_provider(provider);

  REQUIRE(saw_scheme(splits, "file://"));
  REQUIRE(saw_scheme(splits, "s3://"));
  for (auto const& split : splits) {
    for (auto const& slice : split->rg_slices) {
      CHECK(slice.io_ctx.get() == manager.io_ctx_for(slice.file_path));
      CHECK(slice.io_object != nullptr);
    }
  }
}

TEST_CASE("scan_manager S3 end-to-end surfaces missing-key failures after retries",
          "[scan_manager][s3][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping scan_manager missing-key S3 test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  auto manager      = make_scan_manager(*env);
  auto missing_path = s3_uri(env->bucket, "definitely-does-not-exist.parquet");
  auto provider     = make_provider({missing_path}, manager);

  CHECK_THROWS_AS(drive_provider(provider), std::runtime_error);
}

TEST_CASE("scan_manager S3 end-to-end does not retry bad credentials indefinitely",
          "[scan_manager][s3][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN(
      "Skipping scan_manager bad-credentials S3 test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  auto manager = make_scan_manager(*env, true);
  auto path    = s3_uri(env->bucket, "parquet/nation.parquet");
  auto* io_ctx = manager.io_ctx_for(path);
  REQUIRE(io_ctx != nullptr);

  CHECK_THROWS_AS(io_ctx->create_io_object(path), std::runtime_error);
}
