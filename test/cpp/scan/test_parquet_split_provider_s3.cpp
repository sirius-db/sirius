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
#include "io/prefetching_cache.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"
#include "scan_manager/parquet_split_provider.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "scan_manager/split_connector.hpp"

// Include this last among sirius/test headers: it transitively pulls
// liburing.h, whose BLOCK_SIZE macro collides with blockingconcurrentqueue.h.
// clang-format off
#include <scan/test_helpers_ioctx.hpp>
// clang-format on

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <duckdb.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

using sirius::io::buffer_pool;
using sirius::io::sirius_ioctx;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::sirius_sigv4_presigned_authorizer;
using sirius::io::s3::static_credentials;
using sirius::op::scan::parquet_scan_data;
using sirius::scan_manager::parquet_split_provider;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;
using sirius::scan_manager::split_connector;

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

std::size_t cache_capacity_bytes(std::size_t block_size, std::uint32_t max_slabs)
{
  return block_size * static_cast<std::size_t>(buffer_pool::CHUNKS_PER_SLAB) *
         static_cast<std::size_t>(max_slabs);
}

struct host_cache_memory {
  static constexpr std::uint32_t max_slabs = 3;
  static constexpr std::size_t block_size  = 4096;

  host_cache_memory()
    : upstream(0, true),
      host_mr(0,
              upstream,
              cache_capacity_bytes(block_size, max_slabs),
              cache_capacity_bytes(block_size, max_slabs),
              block_size,
              static_cast<std::size_t>(buffer_pool::CHUNKS_PER_SLAB),
              1)
  {
  }

  cucascade::memory::numa_region_pinned_host_memory_resource upstream;
  cucascade::memory::fixed_size_host_memory_resource host_mr;
};

std::filesystem::path fresh_tmp_dir(std::string const& tag)
{
  auto dir = std::filesystem::temp_directory_path() / ("psp_s3_test_" + tag);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);
  return dir;
}

std::filesystem::path write_parquet_file(duckdb::Connection& con,
                                         std::filesystem::path const& dir,
                                         std::string const& name,
                                         std::size_t num_rows)
{
  std::filesystem::create_directories(dir);
  std::string const table = "psp_s3_tmp_" + name;
  auto result             = con.Query("CREATE OR REPLACE TABLE " + table +
                          " AS SELECT range::INTEGER AS id, "
                                      "('name_' || range)::VARCHAR AS name "
                                      "FROM range(" +
                          std::to_string(num_rows) + ")");
  REQUIRE(result);
  REQUIRE(!result->HasError());

  auto const path = dir / (name + ".parquet");
  std::string const copy_sql =
    "COPY " + table + " TO '" + path.string() + "' (FORMAT PARQUET, ROW_GROUP_SIZE 128)";
  result = con.Query(copy_sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());

  result = con.Query("DROP TABLE " + table);
  REQUIRE(result);
  REQUIRE(!result->HasError());
  return path;
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

duckdb::vector<sirius::logical_type> local_returned_types()
{
  return {sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::VARCHAR)};
}

duckdb::vector<sirius::logical_type> nation_returned_types()
{
  return {sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::VARCHAR),
          sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::VARCHAR)};
}

s3_ioctx_config make_live_s3_config(s3_test_env const& env)
{
  static_credentials creds;
  creds.access_key_id     = env.access_key;
  creds.secret_access_key = env.secret_key;
  auto provider           = std::make_shared<sirius_sigv4_presigned_authorizer>(
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

sirius_scan_manager make_scan_manager(cucascade::memory::fixed_size_host_memory_resource& host_mr,
                                      std::optional<s3_test_env> const& env = std::nullopt)
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource = true;
  cfg.uring_n_reactors      = 1;

  std::vector<std::shared_ptr<sirius_ioctx>> borrowed_ioctxs;
  auto gpu_ioctxs = sirius::scan_test_utils::make_test_gpu_ioctxs(1);
  REQUIRE_FALSE(gpu_ioctxs.empty());
  borrowed_ioctxs.push_back(gpu_ioctxs.begin()->second);

  if (env) {
    auto s3_cfg                           = make_live_s3_config(*env);
    s3_cfg.host_memory_resource           = &host_mr;
    cfg.s3_config                         = s3_cfg;
    cfg.s3_thread_pool.num_threads        = 4;
    cfg.s3_thread_pool.thread_name_prefix = "s3_psp";
    borrowed_ioctxs.push_back(std::make_shared<s3_ioctx>(std::move(s3_cfg)));
  }
  return sirius_scan_manager(std::move(cfg), std::move(borrowed_ioctxs));
}

parquet_split_provider make_provider(std::vector<std::string> paths,
                                     duckdb::vector<sirius::logical_type> returned_types,
                                     sirius_scan_manager& manager)
{
  auto const arity = returned_types.size();
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
  sirius::exec::static_thread_pool pool(2, "psp_s3");
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

bool contains_path_with_scheme(std::vector<std::unique_ptr<parquet_scan_data>> const& splits,
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

TEST_CASE("parquet_split_provider routes pure local batches through scan_manager uring backend",
          "[scan_manager][parquet_split_provider][s3]")
{
  auto const dir = fresh_tmp_dir("local");
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto path     = write_parquet_file(con, dir, "local", 256);
  auto file_uri = "file://" + path.string();

  host_cache_memory memory;
  auto manager  = make_scan_manager(memory.host_mr);
  auto provider = make_provider({file_uri}, local_returned_types(), manager);
  auto splits   = drive_provider(provider);

  REQUIRE_FALSE(splits.empty());
  for (auto const& split : splits) {
    for (auto const& slice : split->rg_slices) {
      CHECK(slice.file_path == file_uri);
      CHECK(slice.io_ctx.get() == manager.io_ctx_for(slice.file_path));
      CHECK(slice.io_object != nullptr);
    }
  }
  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_split_provider routes pure S3 batches through scan_manager S3 backend",
          "[.][s3][integration][scan_manager][parquet_split_provider]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 parquet_split_provider test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  host_cache_memory memory;
  auto manager  = make_scan_manager(memory.host_mr, env);
  auto path     = s3_uri(env->bucket, "parquet/nation.parquet");
  auto provider = make_provider({path}, nation_returned_types(), manager);
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

TEST_CASE("parquet_split_provider dispatches mixed local and S3 batches per path",
          "[.][s3][integration][scan_manager][parquet_split_provider]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN(
      "Skipping mixed S3 parquet_split_provider test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  auto local_path = std::filesystem::absolute("test/cpp/integration/data/parquet/nation.parquet");
  auto local_uri  = "file://" + local_path.string();
  auto s3_path    = s3_uri(env->bucket, "parquet/nation.parquet");

  host_cache_memory memory;
  auto manager  = make_scan_manager(memory.host_mr, env);
  auto provider = make_provider({local_uri, s3_path}, nation_returned_types(), manager);
  auto splits   = drive_provider(provider);

  REQUIRE(contains_path_with_scheme(splits, "file://"));
  REQUIRE(contains_path_with_scheme(splits, "s3://"));

  for (auto const& split : splits) {
    for (auto const& slice : split->rg_slices) {
      CHECK(slice.io_ctx.get() == manager.io_ctx_for(slice.file_path));
      CHECK(slice.io_object != nullptr);
    }
  }
}

TEST_CASE("parquet_split_provider reports unsupported paths through the connector",
          "[scan_manager][parquet_split_provider][s3]")
{
  host_cache_memory memory;
  auto manager = make_scan_manager(memory.host_mr);
  auto provider =
    make_provider({"unsupported://bucket/key.parquet"}, local_returned_types(), manager);

  try {
    (void)drive_provider(provider);
    FAIL("unsupported path unexpectedly produced parquet splits");
  } catch (std::runtime_error const& e) {
    CHECK(std::string{e.what()}.find("no backend supports path") != std::string::npos);
  }
}
