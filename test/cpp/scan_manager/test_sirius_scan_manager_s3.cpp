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
#include "io/prefetching_cache.hpp"
#include "io/s3/credential_provider.hpp"
#include "io/s3/mock_credential_provider.hpp"
#include "io/s3/s3_io_object.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/sirius_sigv4_credential_provider.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using sirius::io::buffer_pool;
using sirius::io::s3::credential_provider;
using sirius::io::s3::mock_credential_provider;
using sirius::io::s3::presign_method;
using sirius::io::s3::s3_io_object;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::sirius_sigv4_credential_provider;
using sirius::io::s3::static_credentials;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;

s3_ioctx_config make_mock_s3_config()
{
  s3_ioctx_config cfg{};
  cfg.creds             = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  cfg.max_connections   = 2;
  cfg.request_timeout_s = 1;
  cfg.max_retry_attempts = 1;
  cfg.retry_backoff_base = std::chrono::milliseconds{0};
  cfg.retry_jitter       = std::chrono::milliseconds{0};
  cfg.honor_retry_after  = false;
  return cfg;
}

std::string env_or(std::string_view name, std::string fallback = {})
{
  auto const* value = std::getenv(std::string{name}.c_str());
  return value ? std::string{value} : std::move(fallback);
}

struct s3_test_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  fs::path local_dir;
};

std::optional<s3_test_env> read_s3_test_env()
{
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
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket),
                     fs::path{local_dir}};
}

std::vector<std::uint8_t> read_binary_file(fs::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::shared_ptr<s3_io_object> make_s3_object(std::string bucket, std::string key, std::size_t size)
{
  auto path = s3_uri(bucket, key);
  return std::make_shared<s3_io_object>(std::move(bucket), std::move(key), size, std::move(path));
}

class counting_credential_provider final : public credential_provider {
 public:
  explicit counting_credential_provider(s3_test_env const& env) : _endpoint(env.endpoint)
  {
    static_credentials creds;
    creds.access_key_id     = env.access_key;
    creds.secret_access_key = env.secret_key;
    _delegate               = std::make_shared<sirius_sigv4_credential_provider>(
      std::move(creds), env.region, env.endpoint, 30min);
  }

  std::string get_presigned_url(s3_object_ref const& obj, presign_method method) override
  {
    if (method == presign_method::GET) { _get_count.fetch_add(1, std::memory_order_relaxed); }
    return _delegate->get_presigned_url(obj, method);
  }

  [[nodiscard]] int get_count() const noexcept
  {
    return _get_count.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::string const& endpoint() const noexcept { return _endpoint; }

 private:
  std::shared_ptr<credential_provider> _delegate;
  std::string _endpoint;
  std::atomic<int> _get_count{0};
};

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

bool wait_until_cached(sirius::io::sirius_ioctx& io_ctx,
                       s3_io_object& obj,
                       cudf::io::text::byte_range_info range,
                       std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto view = io_ctx.cache()->read(obj,
                                         static_cast<std::size_t>(range.offset()),
                                         static_cast<std::size_t>(range.size()),
                                         nullptr);
        view) {
      return true;
    }
    std::this_thread::sleep_for(10ms);
  }
  return false;
}

scan_manager_config make_live_s3_scan_manager_config(std::shared_ptr<credential_provider> creds,
                                                     bool enable_cache)
{
  s3_ioctx_config s3_cfg{};
  s3_cfg.creds             = std::move(creds);
  s3_cfg.max_connections   = 4;
  s3_cfg.request_timeout_s = 20;

  scan_manager_config cfg{};
  cfg.use_sirius_datasource = true;
  cfg.uring_n_reactors      = 1;
  cfg.enable_prefetch_cache = enable_cache;
  cfg.prefetch_buffer_pool_bytes =
    cache_capacity_bytes(host_cache_memory::block_size, host_cache_memory::max_slabs);
  cfg.prefetch_inflight_budget_chunks = 8;
  cfg.s3_config                       = std::move(s3_cfg);
  return cfg;
}

std::string make_file_uri(std::string const& tag)
{
  auto path = std::filesystem::temp_directory_path() / ("sirius_scan_manager_s3_" + tag);
  std::ofstream out(path);
  out << "local";
  out.close();
  return "file://" + path.string();
}

sirius::sirius_config make_context_config()
{
  auto const config_path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                           "integration" / "integration.yaml";
  sirius::sirius_config cfg{};
  cfg.load_from_file(config_path);
  REQUIRE_FALSE(cfg.get_memory_space_configs().empty());
  return cfg;
}

}  // namespace

TEST_CASE("sirius_scan_manager constructs S3 backend and dispatches by path", "[scan_manager][s3]")
{
  auto const local_uri = make_file_uri("dispatch.dat");

  scan_manager_config cfg{};
  cfg.use_sirius_datasource             = true;
  cfg.uring_n_reactors                  = 1;
  cfg.s3_config                         = make_mock_s3_config();
  cfg.s3_thread_pool.num_threads        = 2;
  cfg.s3_thread_pool.thread_name_prefix = "s3_io_test";

  host_cache_memory memory;
  sirius_scan_manager manager(std::move(cfg), &memory.host_mr);

  auto* default_ctx = manager.io_ctx();
  REQUIRE(default_ctx != nullptr);

  CHECK(manager.io_ctx_for(local_uri) == default_ctx);

  auto* s3_ctx = manager.io_ctx_for("s3://bucket/key.parquet");
  REQUIRE(s3_ctx != nullptr);
  CHECK(s3_ctx != default_ctx);
  CHECK(dynamic_cast<s3_ioctx*>(s3_ctx) != nullptr);

  CHECK(manager.io_ctx_for("unsupported://bucket/key.parquet") == nullptr);
}

TEST_CASE("sirius_scan_manager leaves S3 disabled when s3_config is empty", "[scan_manager][s3]")
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource = true;
  cfg.uring_n_reactors      = 1;
  host_cache_memory memory;
  sirius_scan_manager manager(std::move(cfg), &memory.host_mr);

  REQUIRE(manager.io_ctx() != nullptr);
  CHECK(manager.io_ctx_for("s3://bucket/key.parquet") == nullptr);
  CHECK(manager.io_ctx_for("unsupported://bucket/key.parquet") == nullptr);
}

TEST_CASE("sirius_scan_manager stop is idempotent with both uring and S3 backends",
          "[scan_manager][s3]")
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource = true;
  cfg.uring_n_reactors      = 1;
  cfg.s3_config             = make_mock_s3_config();

  host_cache_memory memory;
  sirius_scan_manager manager(std::move(cfg), &memory.host_mr);
  REQUIRE(manager.io_ctx_for("s3://bucket/key.parquet") != nullptr);

  manager.stop();
  manager.stop();
}

TEST_CASE("sirius_config carries object_store_config and defaults keep S3 disabled",
          "[sirius][config][s3]")
{
  sirius::sirius_config cfg{};

  CHECK(cfg.object_store_config.endpoint.empty());
  CHECK(cfg.object_store_config.access_key.empty());
  CHECK_FALSE(cfg.get_scan_manager_config().s3_config.has_value());
}

TEST_CASE("sirius_scan_manager wires S3 ioctx cache and serves repeated host reads from it",
          "[s3][scan_manager][cache][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 scan_manager test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 128);

  auto provider = std::make_shared<counting_credential_provider>(*env);
  auto cfg      = make_live_s3_scan_manager_config(provider, true);
  host_cache_memory memory;
  sirius_scan_manager manager(std::move(cfg), &memory.host_mr);

  auto const path = s3_uri(env->bucket, key);
  auto* io_ctx    = manager.io_ctx_for(path);
  REQUIRE(io_ctx != nullptr);
  REQUIRE(io_ctx->cache() != nullptr);

  auto obj = make_s3_object(env->bucket, key, local.size());
  std::vector<cudf::io::text::byte_range_info> ranges{{0, 128}};
  for (int i = 0; i < 64; ++i) {
    io_ctx->cache()->insert(*obj, nullptr, ranges);
  }
  REQUIRE(wait_until_cached(*io_ctx, *obj, ranges.front(), 10s));

  std::vector<std::uint8_t> first(128);
  std::vector<std::uint8_t> second(128);
  CHECK(io_ctx->host_read(*obj, 0, first.size(), first.data()) == first.size());
  CHECK(io_ctx->host_read(*obj, 0, second.size(), second.data()) == second.size());
  CHECK(first == second);
  CHECK(provider->get_count() == 1);
}

TEST_CASE("sirius_scan_manager leaves S3 ioctx cache disabled when prefetch cache is off",
          "[s3][scan_manager][cache]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping S3 scan_manager cache-disabled test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  auto provider = std::make_shared<counting_credential_provider>(*env);
  auto cfg      = make_live_s3_scan_manager_config(provider, false);
  host_cache_memory memory;
  sirius_scan_manager manager(std::move(cfg), &memory.host_mr);

  auto* io_ctx = manager.io_ctx_for(s3_uri(env->bucket, "medium.bin"));
  REQUIRE(io_ctx != nullptr);
  CHECK(io_ctx->cache() == nullptr);
}

TEST_CASE("SiriusContext initialize keeps empty object_store_config inert",
          "[sirius][context][s3][isolated_context]")
{
  auto cfg = make_context_config();

  duckdb::SiriusContext context;
  context.initialize(cfg);
  CHECK(context.get_scan_manager().io_ctx_for("s3://bucket/key.parquet") == nullptr);
  context.terminate();
}

TEST_CASE("SiriusContext initialize wires populated object_store_config into scan_manager",
          "[sirius][context][s3][isolated_context]")
{
  auto cfg                           = make_context_config();
  cfg.object_store_config.endpoint   = "http://127.0.0.1:9000";
  cfg.object_store_config.region     = "us-east-1";
  cfg.object_store_config.access_key = "minioadmin";
  cfg.object_store_config.secret_key = "minioadmin";

  duckdb::SiriusContext context;
  context.initialize(cfg);

  auto* s3_ctx = context.get_scan_manager().io_ctx_for("s3://bucket/nation.parquet");
  REQUIRE(s3_ctx != nullptr);
  CHECK(dynamic_cast<s3_ioctx*>(s3_ctx) != nullptr);

  context.terminate();
}
