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
#include "io/cache/prefetching_cache.hpp"
#include "io/object_store_config.hpp"
#include "io/s3/mock_request_authorizer.hpp"
#include "io/s3/s3_blocking_io_object.hpp"
#include "io/s3/s3_blocking_ioctx.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <algorithm>
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
using sirius::io::object_store_config;
using sirius::io::sirius_ioctx;
using sirius::io::s3::mock_request_authorizer;
using sirius::io::s3::s3_authorized_request;
using sirius::io::s3::s3_blocking_io_object;
using sirius::io::s3::s3_blocking_ioctx;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::s3_request_authorizer;
using sirius::io::s3::s3_request_method;
using sirius::io::s3::sirius_sigv4_presigned_authorizer;
using sirius::io::s3::static_credentials;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;

s3_ioctx_config make_mock_s3_config()
{
  s3_ioctx_config cfg{};
  cfg.creds = std::make_shared<mock_request_authorizer>(
    s3_authorized_request{"http://127.0.0.1:1/not-used", {}});
  cfg.max_connections    = 2;
  cfg.request_timeout_s  = 1;
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

void require_cuda_success(cudaError_t status)
{
  INFO(cudaGetErrorString(status));
  REQUIRE(status == cudaSuccess);
}

std::vector<std::uint8_t> copy_device_to_host(rmm::device_buffer const& device, std::size_t bytes)
{
  REQUIRE(bytes <= device.size());
  std::vector<std::uint8_t> host(bytes);
  if (bytes > 0) {
    require_cuda_success(cudaMemcpy(host.data(), device.data(), bytes, cudaMemcpyDeviceToHost));
  }
  return host;
}

std::vector<std::uint8_t> read_s3_object_to_device(duckdb::SiriusContext& context,
                                                   std::string const& uri,
                                                   std::size_t bytes)
{
  auto s3_ctx = context.get_scan_manager().s3_ioctx();
  REQUIRE(s3_ctx != nullptr);
  auto obj = s3_ctx->create_io_object(uri);
  REQUIRE(obj != nullptr);
  REQUIRE(obj->size() >= bytes);

  rmm::cuda_stream stream;
  rmm::device_buffer dst(bytes, stream);
  auto const got =
    s3_ctx->device_read(*obj, 0, bytes, static_cast<std::uint8_t*>(dst.data()), stream);
  CHECK(got == bytes);
  return copy_device_to_host(dst, got);
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::shared_ptr<s3_blocking_io_object> make_s3_object(std::string bucket,
                                                      std::string key,
                                                      std::size_t size)
{
  auto path = s3_uri(bucket, key);
  return std::make_shared<s3_blocking_io_object>(
    std::move(bucket), std::move(key), size, std::move(path));
}

class counting_request_authorizer final : public s3_request_authorizer {
 public:
  explicit counting_request_authorizer(s3_test_env const& env) : _endpoint(env.endpoint)
  {
    static_credentials creds;
    creds.access_key_id     = env.access_key;
    creds.secret_access_key = env.secret_key;
    _delegate               = std::make_shared<sirius_sigv4_presigned_authorizer>(
      std::move(creds), env.region, env.endpoint, 30min);
  }

  s3_authorized_request authorize(s3_object_ref const& obj,
                                  s3_request_method method,
                                  std::chrono::seconds timeout) override
  {
    if (method == s3_request_method::GET) { _get_count.fetch_add(1, std::memory_order_relaxed); }
    return _delegate->authorize(obj, method, timeout);
  }

  [[nodiscard]] int get_count() const noexcept
  {
    return _get_count.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::string const& endpoint() const noexcept { return _endpoint; }

 private:
  std::shared_ptr<s3_request_authorizer> _delegate;
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
                       sirius::io::sirius_io_object& obj,
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

scan_manager_config make_live_s3_scan_manager_config(std::shared_ptr<s3_request_authorizer> creds,
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

sirius::sirius_config make_context_config(std::string_view filename = "integration.yaml")
{
  auto const config_path =
    std::filesystem::path(__FILE__).parent_path().parent_path() / "integration" / filename;
  sirius::sirius_config cfg{};
  cfg.load_from_file(config_path);
  REQUIRE_FALSE(cfg.get_memory_space_configs().empty());
  return cfg;
}

}  // namespace

TEST_CASE("sirius_scan_manager routes borrowed S3 backend and dispatches by path",
          "[scan_manager][s3]")
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

  auto* routed_s3_ctx = manager.io_ctx_for("s3://bucket/key.parquet");
  REQUIRE(routed_s3_ctx != nullptr);
  CHECK(routed_s3_ctx != default_ctx);
  CHECK(dynamic_cast<s3_blocking_ioctx*>(routed_s3_ctx) != nullptr);

  CHECK(manager.io_ctx_for("unsupported://bucket/key.parquet") == nullptr);
}

TEST_CASE("sirius_scan_manager leaves S3 disabled when s3_config is empty", "[scan_manager][s3]")
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource = true;
  cfg.uring_n_reactors      = 1;
  sirius_scan_manager manager(std::move(cfg));

  REQUIRE(manager.io_ctx() != nullptr);
  CHECK(manager.io_ctx_for("s3://bucket/key.parquet") == nullptr);
  CHECK(manager.io_ctx_for("unsupported://bucket/key.parquet") == nullptr);
}

TEST_CASE("sirius_scan_manager stop does not destroy borrowed S3 backend", "[scan_manager][s3]")
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource = true;
  cfg.uring_n_reactors      = 1;
  cfg.s3_config             = make_mock_s3_config();

  host_cache_memory memory;
  sirius_scan_manager manager(std::move(cfg), &memory.host_mr);
  REQUIRE(manager.io_ctx_for("s3://bucket/key.parquet") != nullptr);

  auto s3_ctx = manager.s3_ioctx();
  REQUIRE(s3_ctx != nullptr);
  std::weak_ptr<sirius_ioctx> weak_s3{s3_ctx};
  s3_ctx.reset();

  manager.stop();
  manager.stop();
  CHECK_FALSE(weak_s3.expired());
}

TEST_CASE("sirius_scan_manager with no datasource or S3 config constructs no backends",
          "[scan_manager][s3][ownership]")
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource = false;

  sirius_scan_manager manager(std::move(cfg));

  CHECK(manager.io_ctx() == nullptr);
  CHECK(manager.io_ctx_for("file:///tmp/does-not-matter.parquet") == nullptr);
  CHECK(manager.io_ctx_for("s3://bucket/key.parquet") == nullptr);
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
          "[.][s3][integration][scan_manager][cache]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 scan_manager test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 128);

  auto provider = std::make_shared<counting_request_authorizer>(*env);
  auto cfg      = make_context_config("integration_s3cache.yaml");
  cfg.set_scan_manager_config(make_live_s3_scan_manager_config(provider, true));

  duckdb::SiriusContext context;
  context.initialize(cfg);

  auto const path = s3_uri(env->bucket, key);
  auto s3_ctx     = context.get_scan_manager().s3_ioctx();
  REQUIRE(s3_ctx != nullptr);
  REQUIRE(dynamic_cast<s3_ioctx*>(s3_ctx.get()) != nullptr);
  CHECK(context.get_scan_manager().io_ctx_for(path) == s3_ctx.get());
  REQUIRE(s3_ctx->cache() != nullptr);

  auto obj = s3_ctx->create_io_object(path);
  REQUIRE(obj != nullptr);
  REQUIRE(obj->size() == local.size());
  std::vector<cudf::io::text::byte_range_info> ranges{{0, 128}};
  for (int i = 0; i < 64; ++i) {
    s3_ctx->cache()->insert(*obj, nullptr, ranges);
  }
  REQUIRE(wait_until_cached(*s3_ctx, *obj, ranges.front(), 10s));

  std::vector<std::uint8_t> first(128);
  std::vector<std::uint8_t> second(128);
  CHECK(s3_ctx->host_read(*obj, 0, first.size(), first.data()) == first.size());
  CHECK(s3_ctx->host_read(*obj, 0, second.size(), second.data()) == second.size());
  CHECK(first == second);
  CHECK(provider->get_count() == 1);

  s3_ctx.reset();
  context.terminate();
}

TEST_CASE("sirius_scan_manager leaves S3 ioctx cache disabled when prefetch cache is off",
          "[.][s3][integration][scan_manager][cache]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping S3 scan_manager cache-disabled test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  auto provider = std::make_shared<counting_request_authorizer>(*env);
  auto cfg      = make_context_config("integration_s3cache.yaml");
  cfg.set_scan_manager_config(make_live_s3_scan_manager_config(provider, false));

  duckdb::SiriusContext context;
  context.initialize(cfg);

  auto s3_ctx = context.get_scan_manager().s3_ioctx();
  REQUIRE(s3_ctx != nullptr);
  CHECK(context.get_scan_manager().io_ctx_for(s3_uri(env->bucket, "medium.bin")) == s3_ctx.get());
  CHECK(s3_ctx->cache() == nullptr);

  context.terminate();
}

TEST_CASE("SiriusContext object_store_config header signing reads MinIO bytes",
          "[.][s3][integration][scan_manager][authorizer]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping S3 header-signing integration because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  std::string const key = "small.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 32);

  auto cfg                                = make_context_config("integration_s3cache.yaml");
  cfg.object_store_config.endpoint        = env->endpoint;
  cfg.object_store_config.region          = env->region;
  cfg.object_store_config.access_key      = env->access_key;
  cfg.object_store_config.secret_key      = env->secret_key;
  cfg.object_store_config.s3_signing_mode = object_store_config::signing_mode::header;

  auto scan_cfg                              = cfg.get_scan_manager_config();
  scan_cfg.use_sirius_datasource             = true;
  scan_cfg.s3_thread_pool.num_threads        = 4;
  scan_cfg.s3_thread_pool.thread_name_prefix = "s3_header";
  cfg.set_scan_manager_config(std::move(scan_cfg));

  duckdb::SiriusContext context;
  context.initialize(cfg);

  auto s3_ctx = context.get_scan_manager().s3_ioctx();
  REQUIRE(s3_ctx != nullptr);
  auto* s3 = dynamic_cast<s3_ioctx*>(s3_ctx.get());
  REQUIRE(s3 != nullptr);
  auto obj = s3->create_io_object(s3_uri(env->bucket, key));
  REQUIRE(obj != nullptr);
  REQUIRE(obj->size() == local.size());
  std::vector<std::uint8_t> got(32);
  REQUIRE(s3->host_read(*obj, 0, got.size(), got.data()) == got.size());
  CHECK(std::equal(got.begin(), got.end(), local.begin()));

  s3_ctx.reset();
  context.terminate();
}

TEST_CASE("SiriusContext initialize keeps empty object_store_config inert",
          "[sirius][context][s3][isolated_context]")
{
  auto cfg = make_context_config();

  duckdb::SiriusContext context;
  context.initialize(cfg);
  CHECK(context.get_scan_manager().s3_ioctx() == nullptr);
  CHECK(context.get_scan_manager().io_ctx_for("s3://bucket/key.parquet") == nullptr);
  CHECK_FALSE(context.get_config().get_scan_manager_config().s3_config.has_value());
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

  auto owned_s3_ctx = context.get_scan_manager().s3_ioctx();
  REQUIRE(owned_s3_ctx != nullptr);

  auto* s3_ctx = context.get_scan_manager().io_ctx_for("s3://bucket/nation.parquet");
  REQUIRE(s3_ctx != nullptr);
  CHECK(dynamic_cast<s3_ioctx*>(s3_ctx) != nullptr);
  CHECK(s3_ctx == owned_s3_ctx.get());

  auto const& stored_scan_config = context.get_config().get_scan_manager_config();
  REQUIRE(stored_scan_config.s3_config.has_value());
  CHECK(stored_scan_config.s3_config->async_thread_pool == nullptr);

  owned_s3_ctx.reset();
  context.terminate();
}

TEST_CASE("SiriusContext owns S3 ioctx beyond borrowed scan_manager lifetime",
          "[sirius][context][s3][ownership][isolated_context]")
{
  auto cfg                           = make_context_config();
  cfg.object_store_config.endpoint   = "http://127.0.0.1:9000";
  cfg.object_store_config.region     = "us-east-1";
  cfg.object_store_config.access_key = "minioadmin";
  cfg.object_store_config.secret_key = "minioadmin";

  std::weak_ptr<sirius_ioctx> weak_s3;

  duckdb::SiriusContext context;
  context.initialize(cfg);
  {
    auto owned_s3_ctx = context.get_scan_manager().s3_ioctx();
    REQUIRE(owned_s3_ctx != nullptr);
    weak_s3 = owned_s3_ctx;

    sirius_scan_manager borrowed_manager(context.get_config().get_scan_manager_config());
    CHECK(borrowed_manager.io_ctx_for("s3://bucket/key.parquet") != nullptr);
  }

  CHECK_FALSE(weak_s3.expired());
  context.terminate();
  CHECK(weak_s3.expired());
}

TEST_CASE("SiriusContext with s3_use_async_backend=false selects the blocking backend",
          "[.][s3][integration][scan_manager]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping blocking-backend selection test because SIRIUS_TEST_S3_* is not configured");
    return;
  }

  auto cfg                                     = make_context_config("integration_s3cache.yaml");
  cfg.object_store_config.endpoint             = env->endpoint;
  cfg.object_store_config.region               = env->region;
  cfg.object_store_config.access_key           = env->access_key;
  cfg.object_store_config.secret_key           = env->secret_key;
  cfg.object_store_config.s3_use_async_backend = false;

  duckdb::SiriusContext context;
  context.initialize(cfg);

  auto s3_ctx = context.get_scan_manager().s3_ioctx();
  REQUIRE(s3_ctx != nullptr);
  CHECK(dynamic_cast<s3_blocking_ioctx*>(s3_ctx.get()) != nullptr);
  CHECK(dynamic_cast<s3_ioctx*>(s3_ctx.get()) == nullptr);

  s3_ctx.reset();
  context.terminate();
}

TEST_CASE("async and blocking S3 backends device_read the same bytes as the local oracle",
          "[.][s3][integration][scan_manager][asynccurl]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping S3 device_read parity test because SIRIUS_TEST_S3_* is not configured");
    return;
  }
  if (env->local_dir.empty()) {
    WARN("Skipping S3 device_read parity test because SIRIUS_TEST_S3_LOCAL_DIR is not configured");
    return;
  }

  std::string const key = "small.bin";
  auto const oracle     = read_binary_file(env->local_dir / key);
  REQUIRE_FALSE(oracle.empty());
  REQUIRE(oracle.size() < sirius::io::CHUNK_SIZE);
  auto const uri = s3_uri(env->bucket, key);

  auto run_with_backend = [&](bool use_async) {
    auto cfg                                     = make_context_config("integration_s3cache.yaml");
    cfg.object_store_config.endpoint             = env->endpoint;
    cfg.object_store_config.region               = env->region;
    cfg.object_store_config.access_key           = env->access_key;
    cfg.object_store_config.secret_key           = env->secret_key;
    cfg.object_store_config.s3_use_async_backend = use_async;

    duckdb::SiriusContext context;
    context.initialize(cfg);
    {
      auto s3_ctx = context.get_scan_manager().s3_ioctx();
      REQUIRE(s3_ctx != nullptr);
      if (use_async) {
        REQUIRE(dynamic_cast<s3_ioctx*>(s3_ctx.get()) != nullptr);
      } else {
        REQUIRE(dynamic_cast<s3_blocking_ioctx*>(s3_ctx.get()) != nullptr);
      }
    }
    auto bytes = read_s3_object_to_device(context, uri, oracle.size());
    context.terminate();
    return bytes;
  };

  auto const blocking_bytes = run_with_backend(false);
  auto const async_bytes    = run_with_backend(true);

  CHECK(blocking_bytes == oracle);
  CHECK(async_bytes == oracle);
  CHECK(async_bytes == blocking_bytes);
}

TEST_CASE("SiriusContext initializes prefetch cache on real local gpu ioctxs",
          "[sirius][context][scan_manager][cache][isolated_context]")
{
  auto cfg                       = make_context_config();
  auto scan_cfg                  = cfg.get_scan_manager_config();
  scan_cfg.use_sirius_datasource = true;
  scan_cfg.uring_n_reactors      = 1;
  scan_cfg.enable_prefetch_cache = true;
  scan_cfg.prefetch_buffer_pool_bytes =
    cache_capacity_bytes(host_cache_memory::block_size, host_cache_memory::max_slabs);
  scan_cfg.prefetch_inflight_budget_chunks = 8;
  cfg.set_scan_manager_config(std::move(scan_cfg));

  duckdb::SiriusContext context;
  context.initialize(cfg);

  auto* ioctx = context.get_scan_manager().io_ctx();
  REQUIRE(ioctx != nullptr);
  CHECK(ioctx->cache() != nullptr);

  context.terminate();
}
