/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/prefetching_cache.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/sirius_sigv4_credential_provider.hpp"
#include "scan_manager/parquet_metadata.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

using sirius::io::buffer_pool;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::sirius_sigv4_credential_provider;
using sirius::io::s3::static_credentials;
using sirius::scan_manager::parquet_metadata;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

using namespace std::chrono_literals;

constexpr std::size_t one_mib         = 1ULL << 20;
constexpr std::size_t one_hundred_mib = 100ULL << 20;

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

bool skip_if_no_s3_env(std::optional<s3_test_env> const& env)
{
  if (env) { return false; }
  if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
    FAIL("SIRIUS_TEST_S3_* environment is required in strict mode");
  }
  SUCCEED("SIRIUS_TEST_S3_* not set; skipping live S3 describe_parquet test");
  return true;
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

s3_ioctx_config make_s3_config(s3_test_env const& env)
{
  static_credentials creds;
  creds.access_key_id     = env.access_key;
  creds.secret_access_key = env.secret_key;

  s3_ioctx_config cfg{};
  cfg.creds = std::make_shared<sirius_sigv4_credential_provider>(
    std::move(creds), env.region, env.endpoint, 30min);
  cfg.max_connections    = 4;
  cfg.request_timeout_s  = 20;
  cfg.max_retry_attempts = 3;
  cfg.retry_backoff_base = 10ms;
  cfg.retry_jitter       = 0ms;
  cfg.honor_retry_after  = false;
  return cfg;
}

sirius_scan_manager make_s3_manager(s3_test_env const& env,
                                    cucascade::memory::fixed_size_host_memory_resource* host_mr,
                                    bool enable_cache)
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource             = true;
  cfg.uring_n_reactors                  = 1;
  cfg.s3_config                         = make_s3_config(env);
  cfg.s3_thread_pool.num_threads        = 4;
  cfg.s3_thread_pool.thread_name_prefix = "pr6_s3";
  cfg.enable_prefetch_cache             = enable_cache;
  cfg.prefetch_buffer_pool_bytes =
    cache_capacity_bytes(host_cache_memory::block_size, host_cache_memory::max_slabs);
  cfg.prefetch_inflight_budget_chunks = 8;
  return sirius_scan_manager(std::move(cfg), host_mr);
}

}  // namespace

TEST_CASE("describe_parquet reports no matching backend with the URI in the error",
          "[scan_manager][describe_parquet]")
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource = false;
  sirius_scan_manager manager(std::move(cfg));

  REQUIRE_THROWS_WITH(manager.describe_parquet("s3://missing-backend/object.parquet"),
                      Catch::Contains("s3://missing-backend/object.parquet"));
}

TEST_CASE("describe_parquet surfaces S3 missing-object errors cleanly", "[s3][describe_parquet]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  host_cache_memory memory;
  auto manager   = make_s3_manager(*env, &memory.host_mr, false);
  auto const uri = s3_uri(env->bucket, "parquet/definitely-missing-pr6-object.parquet");

  REQUIRE_THROWS_WITH(manager.describe_parquet(uri), Catch::Contains("404"));
}

TEST_CASE("describe_parquet fetches only the footer over S3", "[s3][describe_parquet]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  host_cache_memory memory;
  auto manager   = make_s3_manager(*env, &memory.host_mr, false);
  auto const key = env_or("SIRIUS_PR6_LARGE_S3_KEY", "parquet/lineitem.parquet");
  auto const uri = s3_uri(env->bucket, key);

  auto* base_ctx = manager.io_ctx_for(uri);
  REQUIRE(base_ctx != nullptr);
  auto* s3_ctx = dynamic_cast<s3_ioctx*>(base_ctx);
  REQUIRE(s3_ctx != nullptr);

  auto const before = s3_ctx->bytes_read_total();
  auto result       = manager.describe_parquet(uri);
  auto const delta  = s3_ctx->bytes_read_total() - before;

  if (result.object_size < one_hundred_mib) {
    WARN(
      "R8 is using a fixture smaller than 100 MiB; set SIRIUS_PR6_LARGE_S3_KEY "
      "to exercise the exact large-file acceptance case");
  }
  CHECK(result.object_size > 0);
  CHECK(delta <= one_mib);
}

TEST_CASE("describe_parquet inserts parsed parquet metadata into the prefetch cache",
          "[s3][describe_parquet][cache]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  host_cache_memory memory;
  auto manager   = make_s3_manager(*env, &memory.host_mr, true);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto bind_info = manager.describe_parquet(uri);
  auto* base_ctx = manager.io_ctx_for(uri);
  auto* s3_ctx   = dynamic_cast<s3_ioctx*>(base_ctx);
  REQUIRE(s3_ctx != nullptr);
  REQUIRE(s3_ctx->cache() != nullptr);

  auto io_object = s3_ctx->create_io_object(uri);
  auto metadata  = s3_ctx->cache()->get_metadata(*io_object);
  REQUIRE(metadata != nullptr);

  auto parquet = std::dynamic_pointer_cast<parquet_metadata>(metadata);
  REQUIRE(parquet != nullptr);
  REQUIRE(parquet->file_metadata() != nullptr);
  CHECK(parquet->footer_byte_len() > 0);
  CHECK_FALSE(bind_info.names.empty());
  CHECK(parquet->file_metadata()->schema.size() >= bind_info.names.size() + 1);
}

TEST_CASE("describe_parquet exposes the parquet footer row count for planner metadata",
          "[s3][describe_parquet][planner-metadata]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  host_cache_memory memory;
  auto manager   = make_s3_manager(*env, &memory.host_mr, true);
  auto const uri = s3_uri(env->bucket, "parquet/orders.parquet");

  auto bind_info = manager.describe_parquet(uri);

  auto* base_ctx = manager.io_ctx_for(uri);
  auto* s3_ctx   = dynamic_cast<s3_ioctx*>(base_ctx);
  REQUIRE(s3_ctx != nullptr);
  REQUIRE(s3_ctx->cache() != nullptr);

  auto io_object = s3_ctx->create_io_object(uri);
  auto metadata  = s3_ctx->cache()->get_metadata(*io_object);
  auto parquet   = std::dynamic_pointer_cast<parquet_metadata>(metadata);
  REQUIRE(parquet != nullptr);
  REQUIRE(parquet->file_metadata() != nullptr);

  CHECK(bind_info.total_num_rows > 0);
  CHECK(bind_info.total_num_rows == static_cast<std::size_t>(parquet->file_metadata()->num_rows));
}
