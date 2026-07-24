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

#pragma once

#include "catch.hpp"
#include "io/object_store_config.hpp"
#include "io/rdma/cuobj_rdma_client.hpp"
#include "io/rdma/cuobj_rdma_reactor.hpp"
#include "io/rdma/mock_rdma_client.hpp"
#include "io/rdma/rdma_client.hpp"
#include "io/s3/s3_rdma_ioctx.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"
#include "io/s3/static_credentials.hpp"
#include "io/sirius_datasource.hpp"
#ifdef SIRIUS_HAVE_TESTCONTAINERS
#include "utils/s3_container.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace s3_rdma_p3a_tests {

using sirius::io::object_store_config;
using sirius::io::rdma::curl_s3_control_client;
using sirius::io::rdma::rx_route;
using sirius::io::s3::s3_rdma_ioctx;

constexpr std::string_view k_bucket_env     = "SIRIUS_TEST_S3_BUCKET";
constexpr std::string_view k_region_env     = "SIRIUS_TEST_S3_REGION";
constexpr std::string_view k_endpoint_env   = "SIRIUS_TEST_S3_ENDPOINT";
constexpr std::string_view k_access_key_env = "SIRIUS_TEST_S3_ACCESS_KEY";
constexpr std::string_view k_secret_key_env = "SIRIUS_TEST_S3_SECRET_KEY";
constexpr std::string_view k_local_dir_env  = "SIRIUS_TEST_S3_LOCAL_DIR";
constexpr std::string_view k_fixture_key    = "small.bin";
constexpr std::size_t k_max_inflight        = 2;
constexpr std::size_t k_arena_slot_size     = 64UL << 10;

enum class signing_case { presigned, header };

struct minio_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::filesystem::path local_dir;
};

std::string env_or(std::string_view name, std::string fallback = {})
{
  if (auto* value = std::getenv(std::string{name}.c_str()); value != nullptr) { return value; }
  return fallback;
}

std::string require_env(std::string_view name)
{
  auto value = env_or(name);
  REQUIRE_FALSE(value.empty());
  return value;
}

#ifdef SIRIUS_HAVE_TESTCONTAINERS
bool ensure_minio_env()
{
  if (sirius::test::ensure_s3_container_env()) { return true; }
  SUCCEED("SIRIUS_TEST_S3_* not set; skipping S3 RDMA real-client MinIO test");
  return false;
}
#endif

minio_env read_minio_env()
{
  return minio_env{require_env(k_endpoint_env),
                   env_or(k_region_env, "us-east-1"),
                   require_env(k_access_key_env),
                   require_env(k_secret_key_env),
                   require_env(k_bucket_env),
                   std::filesystem::path{require_env(k_local_dir_env)}};
}

std::vector<std::uint8_t> read_binary_file(std::filesystem::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> chars((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return std::vector<std::uint8_t>(chars.begin(), chars.end());
}

void require_bytes_equal(std::span<const std::uint8_t> got, std::span<const std::uint8_t> expected)
{
  REQUIRE(got.size() == expected.size());
  CHECK(std::equal(got.begin(), got.end(), expected.begin(), expected.end()));
}

object_store_config object_store_cfg(minio_env const& env,
                                     signing_case signing,
                                     std::string secret_override = {})
{
  object_store_config cfg;
  cfg.endpoint             = env.endpoint;
  cfg.region               = env.region;
  cfg.access_key           = env.access_key;
  cfg.secret_key           = secret_override.empty() ? env.secret_key : std::move(secret_override);
  cfg.s3_transport         = object_store_config::transport::RDMA;
  cfg.s3_rdma_max_inflight = k_max_inflight;
  cfg.s3_rdma_arena_slot_size = k_arena_slot_size;
  cfg.s3_signing_mode         = signing == signing_case::presigned
                                  ? object_store_config::signing_mode::presigned
                                  : object_store_config::signing_mode::header;
  cfg.tls_verify              = false;
  return cfg;
}

std::shared_ptr<sirius::io::s3::s3_request_authorizer> make_authorizer(
  object_store_config const& cfg)
{
  auto creds = sirius::io::s3::static_credentials_from(cfg);
  if (cfg.s3_signing_mode == object_store_config::signing_mode::header) {
    return std::make_shared<sirius::io::s3::sirius_sigv4_header_authorizer>(
      std::move(creds), cfg.region, cfg.endpoint);
  }
  return std::make_shared<sirius::io::s3::sirius_sigv4_presigned_authorizer>(
    std::move(creds), cfg.region, cfg.endpoint);
}

std::shared_ptr<curl_s3_control_client> make_real_client(object_store_config const& cfg)
{
  return std::make_shared<curl_s3_control_client>(
    make_authorizer(cfg), cfg.ca_bundle_path, cfg.tls_verify);
}

sirius::io::rdma::rdma_transport_clients make_transport_clients(object_store_config const& cfg)
{
  return sirius::io::rdma::rdma_transport_clients{
    make_real_client(cfg), std::make_shared<sirius::io::rdma::mock_rdma_data_session_factory>()};
}

std::string signing_name(signing_case signing)
{
  return signing == signing_case::presigned ? "presigned" : "header";
}

}  // namespace s3_rdma_p3a_tests

TEST_CASE("mock RDMA data sessions expose registration lifetime", "[s3][rdma][client]")
{
  auto factory = std::make_shared<sirius::io::rdma::mock_rdma_data_session_factory>();
  auto session = factory->acquire();
  REQUIRE(session != nullptr);
  std::array<std::uint8_t, 16> host{};

  CHECK_NOTHROW(session->register_memory(host.data(), host.size()));
  CHECK_NOTHROW(session->deregister_memory(host.data()));
  CHECK(factory->register_count() == 1);
  CHECK(factory->deregister_count() == 1);
}

TEST_CASE("cuobj_rdma_reactor flush decision matches GPUDirect ordering classes",
          "[s3][rdma][client]")
{
  constexpr int k_no_ordering         = 0;
  constexpr int k_owner_ordered       = 100;
  constexpr int k_all_devices_ordered = 200;

  CHECK(sirius::io::rdma::flush_required(k_no_ordering));
  CHECK_FALSE(sirius::io::rdma::flush_required(k_owner_ordered));
  CHECK_FALSE(sirius::io::rdma::flush_required(k_all_devices_ordered));
}

#ifdef SIRIUS_HAVE_TESTCONTAINERS

TEST_CASE("curl S3 control client HEADs MinIO objects with both signing modes",
          "[s3][rdma][client][integration]")
{
  using namespace s3_rdma_p3a_tests;
  if (!ensure_minio_env()) { return; }

  auto const env     = read_minio_env();
  auto const payload = read_binary_file(env.local_dir / std::string{k_fixture_key});

  for (auto signing : {signing_case::presigned, signing_case::header}) {
    DYNAMIC_SECTION(signing_name(signing))
    {
      auto client      = make_real_client(object_store_cfg(env, signing));
      auto const found = client->head(rx_route{env.bucket, std::string{k_fixture_key}});
      CHECK(found.outcome.http_status == 200);
      CHECK(found.outcome.transport_error.empty());
      CHECK(found.object_size == payload.size());

      auto const missing = client->head(rx_route{env.bucket, "does-not-exist-for-rdma-client.bin"});
      CHECK(missing.outcome.http_status == 404);
      CHECK(missing.outcome.transport_error.empty());
    }
  }
}

TEST_CASE("curl S3 control client reads MinIO ranges exactly with both signing modes",
          "[s3][rdma][client][integration]")
{
  using namespace s3_rdma_p3a_tests;
  if (!ensure_minio_env()) { return; }

  auto const env     = read_minio_env();
  auto const payload = read_binary_file(env.local_dir / std::string{k_fixture_key});
  REQUIRE(payload.size() > 8192);

  for (auto signing : {signing_case::presigned, signing_case::header}) {
    DYNAMIC_SECTION(signing_name(signing))
    {
      auto client = make_real_client(object_store_cfg(env, signing));
      rx_route route{env.bucket, std::string{k_fixture_key}};

      std::vector<std::uint8_t> start(4096);
      auto const start_result = client->range_get(route, 0, start.size(), start.data());
      REQUIRE(start_result.outcome.http_status == 206);
      REQUIRE(start_result.delivered_bytes == start.size());
      require_bytes_equal(start, std::span<const std::uint8_t>(payload.data(), start.size()));

      std::vector<std::uint8_t> mid(2048);
      constexpr std::size_t mid_offset = 1234;
      auto const mid_result = client->range_get(route, mid_offset, mid.size(), mid.data());
      REQUIRE(mid_result.outcome.http_status == 206);
      REQUIRE(mid_result.delivered_bytes == mid.size());
      require_bytes_equal(mid,
                          std::span<const std::uint8_t>(payload.data() + mid_offset, mid.size()));

      std::vector<std::uint8_t> crossing(64, std::uint8_t{0xaa});
      auto const tail_offset = payload.size() - 17;
      auto const crossing_result =
        client->range_get(route, tail_offset, crossing.size(), crossing.data());
      REQUIRE(crossing_result.outcome.http_status == 206);
      REQUIRE(crossing_result.delivered_bytes == 17);
      require_bytes_equal(std::span<const std::uint8_t>(crossing.data(), 17),
                          std::span<const std::uint8_t>(payload.data() + tail_offset, 17));

      std::array<std::uint8_t, 8> eof{};
      eof.fill(std::uint8_t{0xcc});
      auto const at_eof = client->range_get(route, payload.size(), eof.size(), eof.data());
      CHECK(at_eof.outcome.http_status == 416);
      CHECK(at_eof.delivered_bytes == 0);
      auto const past_eof = client->range_get(route, payload.size() + 99, eof.size(), eof.data());
      CHECK(past_eof.outcome.http_status == 416);
      CHECK(past_eof.delivered_bytes == 0);
      CHECK(std::all_of(eof.begin(), eof.end(), [](std::uint8_t b) { return b == 0xcc; }));
    }
  }
}

TEST_CASE("s3_rdma_ioctx uses the real client for host reads through MinIO",
          "[s3][rdma][client][integration]")
{
  using namespace s3_rdma_p3a_tests;
  if (!ensure_minio_env()) { return; }

  auto const env     = read_minio_env();
  auto const payload = read_binary_file(env.local_dir / std::string{k_fixture_key});

  for (auto signing : {signing_case::presigned, signing_case::header}) {
    DYNAMIC_SECTION(signing_name(signing))
    {
      auto cfg = object_store_cfg(env, signing);
      auto ctx = std::make_shared<s3_rdma_ioctx>(cfg, make_transport_clients(cfg));
      ctx->start();

      auto datasource =
        ctx->open_datasource("s3://" + env.bucket + "/" + std::string{k_fixture_key});
      REQUIRE(datasource != nullptr);
      CHECK(datasource->size() == payload.size());

      std::vector<std::uint8_t> got(4096);
      constexpr std::size_t offset = 257;
      REQUIRE(datasource->host_read(offset, got.size(), got.data()) == got.size());
      require_bytes_equal(got, std::span<const std::uint8_t>(payload.data() + offset, got.size()));

      ctx->shutdown();
    }
  }
}

TEST_CASE("curl S3 control client lists MinIO objects with pagination",
          "[s3][rdma][client][integration][list]")
{
  using namespace s3_rdma_p3a_tests;
  if (!ensure_minio_env()) { return; }

  auto const env = read_minio_env();

  for (auto signing : {signing_case::presigned, signing_case::header}) {
    DYNAMIC_SECTION(signing_name(signing))
    {
      auto client = make_real_client(object_store_cfg(env, signing));

      auto const first = client->list_page(env.bucket, "parquet/", 1, "");
      REQUIRE(first.outcome.http_status == 200);
      REQUIRE(first.outcome.transport_error.empty());
      REQUIRE(first.page.entries.size() == 1);
      REQUIRE(first.page.is_truncated);
      REQUIRE_FALSE(first.page.next_continuation_token.empty());

      auto const second =
        client->list_page(env.bucket, "parquet/", 1, first.page.next_continuation_token);
      REQUIRE(second.outcome.http_status == 200);
      REQUIRE(second.outcome.transport_error.empty());
      REQUIRE(second.page.entries.size() == 1);
      CHECK(second.page.entries.front().key > first.page.entries.front().key);

      auto const clamped = client->list_page(env.bucket, "parquet/", 0, "");
      REQUIRE(clamped.outcome.http_status == 200);
      REQUIRE(clamped.outcome.transport_error.empty());
      CHECK(clamped.page.entries.size() > 1);
      CHECK(clamped.page.entries.size() <= 1000);
    }
  }
}

TEST_CASE("s3_rdma_ioctx lists MinIO objects through the real control plane",
          "[s3][rdma][client][integration][list]")
{
  using namespace s3_rdma_p3a_tests;
  if (!ensure_minio_env()) { return; }

  auto const env = read_minio_env();

  for (auto signing : {signing_case::presigned, signing_case::header}) {
    DYNAMIC_SECTION(signing_name(signing))
    {
      auto cfg = object_store_cfg(env, signing);
      auto ctx = std::make_shared<s3_rdma_ioctx>(cfg, make_transport_clients(cfg));
      ctx->start();

      std::vector<sirius::io::s3::list_entry> entries;
      std::size_t pages = 0;
      ctx->list_objects_paged(
        env.bucket, "parquet/", 1, [&](sirius::io::s3::list_objects_v2_page const& page) {
          ++pages;
          entries.insert(entries.end(), page.entries.begin(), page.entries.end());
          return true;
        });

      REQUIRE(pages >= 2);
      REQUIRE(entries.size() >= 2);
      CHECK(std::is_sorted(entries.begin(), entries.end(), [](auto const& lhs, auto const& rhs) {
        return lhs.key < rhs.key;
      }));
      CHECK(std::all_of(entries.begin(), entries.end(), [](auto const& entry) {
        return entry.key.starts_with("parquet/");
      }));

      ctx->shutdown();
    }
  }
}

TEST_CASE("curl S3 control client reports MinIO authentication failures",
          "[s3][rdma][client][integration]")
{
  using namespace s3_rdma_p3a_tests;
  if (!ensure_minio_env()) { return; }

  auto const env = read_minio_env();

  for (auto signing : {signing_case::presigned, signing_case::header}) {
    DYNAMIC_SECTION(signing_name(signing))
    {
      auto client = make_real_client(object_store_cfg(env, signing, "wrong-secret-key"));

      auto const head = client->head(rx_route{env.bucket, std::string{k_fixture_key}});
      CHECK(head.outcome.http_status >= 400);
      CHECK(head.outcome.transport_error.empty());

      std::array<std::uint8_t, 16> dst{};
      auto const get = client->range_get(
        rx_route{env.bucket, std::string{k_fixture_key}}, 0, dst.size(), dst.data());
      CHECK(get.outcome.http_status >= 400);
      CHECK(get.outcome.transport_error.empty());
    }
  }
}

#endif

TEST_CASE("s3_rdma_ioctx start rejects a missing data-session factory", "[s3][rdma][client]")
{
  using namespace s3_rdma_p3a_tests;

  object_store_config cfg;
  cfg.s3_transport = object_store_config::transport::RDMA;
  sirius::io::rdma::rdma_transport_clients clients{
    std::make_shared<sirius::io::rdma::mock_s3_control_client>(), nullptr};
  s3_rdma_ioctx ctx{std::move(cfg), std::move(clients)};

  CHECK_THROWS_WITH(ctx.start(), Catch::Contains("RDMA") && Catch::Contains("initialization"));
}
