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
#include "utils/s3_container.hpp"

#include <cuda_runtime.h>

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
using sirius::io::rdma::cuobj_rdma_client;
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

bool ensure_minio_env()
{
  if (sirius::test::ensure_s3_container_env()) { return true; }
  SUCCEED("SIRIUS_TEST_S3_* not set; skipping S3 RDMA real-client MinIO test");
  return false;
}

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

bool contains(std::string_view haystack, std::string_view needle)
{
  return haystack.find(needle) != std::string_view::npos;
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

std::shared_ptr<cuobj_rdma_client> make_real_client(object_store_config const& cfg)
{
  return std::make_shared<cuobj_rdma_client>(
    make_authorizer(cfg), cfg.ca_bundle_path, cfg.tls_verify);
}

std::string signing_name(signing_case signing)
{
  return signing == signing_case::presigned ? "presigned" : "header";
}

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA real-client device-path test: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

class default_registration_client final : public sirius::io::rdma::rdma_client {
 public:
  std::size_t head(std::string_view, std::string_view) override { return 0; }

  std::size_t get(std::string_view, std::string_view, std::size_t, std::size_t, void*) override
  {
    return 0;
  }
};

}  // namespace s3_rdma_p3a_tests

TEST_CASE("rdma_client registration hooks are backward-compatible no-ops", "[s3][rdma][client]")
{
  s3_rdma_p3a_tests::default_registration_client base_client;
  std::array<std::uint8_t, 16> host{};

  CHECK_NOTHROW(base_client.register_memory(host.data(), host.size()));
  CHECK_NOTHROW(base_client.deregister_memory(host.data()));

  sirius::io::rdma::mock_rdma_client mock_client;
  CHECK_NOTHROW(mock_client.register_memory(host.data(), host.size()));
  CHECK_NOTHROW(mock_client.deregister_memory(host.data()));
}

TEST_CASE("cuobj_rdma_reactor flush decision matches GPUDirect ordering classes",
          "[s3][rdma][client]")
{
  constexpr int k_no_ordering         = 0;
  constexpr int k_owner_ordered       = 100;
  constexpr int k_all_devices_ordered = 200;

  CHECK(sirius::io::rdma::flush_required(k_no_ordering));
  CHECK(sirius::io::rdma::flush_required(k_owner_ordered));
  CHECK_FALSE(sirius::io::rdma::flush_required(k_all_devices_ordered));
}

TEST_CASE("cuobj_rdma_client HEADs MinIO objects with both signing modes",
          "[s3][rdma][client][integration]")
{
  using namespace s3_rdma_p3a_tests;
  if (!ensure_minio_env()) { return; }

  auto const env     = read_minio_env();
  auto const payload = read_binary_file(env.local_dir / std::string{k_fixture_key});

  for (auto signing : {signing_case::presigned, signing_case::header}) {
    DYNAMIC_SECTION(signing_name(signing))
    {
      auto client = make_real_client(object_store_cfg(env, signing));
      CHECK(client->head(env.bucket, k_fixture_key) == payload.size());

      try {
        (void)client->head(env.bucket, "does-not-exist-for-rdma-client.bin");
        FAIL("expected HEAD on an absent object to throw");
      } catch (std::runtime_error const& e) {
        CHECK(s3_rdma_p3a_tests::contains(e.what(), "does-not-exist-for-rdma-client.bin"));
      }
    }
  }
}

TEST_CASE("cuobj_rdma_client host GET reads MinIO ranges exactly with both signing modes",
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

      std::vector<std::uint8_t> start(4096);
      REQUIRE(client->get(env.bucket, k_fixture_key, 0, start.size(), start.data()) ==
              start.size());
      require_bytes_equal(start, std::span<const std::uint8_t>(payload.data(), start.size()));

      std::vector<std::uint8_t> mid(2048);
      constexpr std::size_t mid_offset = 1234;
      REQUIRE(client->get(env.bucket, k_fixture_key, mid_offset, mid.size(), mid.data()) ==
              mid.size());
      require_bytes_equal(mid,
                          std::span<const std::uint8_t>(payload.data() + mid_offset, mid.size()));

      std::vector<std::uint8_t> crossing(64, std::uint8_t{0xaa});
      auto const tail_offset = payload.size() - 17;
      REQUIRE(client->get(
                env.bucket, k_fixture_key, tail_offset, crossing.size(), crossing.data()) == 17);
      require_bytes_equal(std::span<const std::uint8_t>(crossing.data(), 17),
                          std::span<const std::uint8_t>(payload.data() + tail_offset, 17));

      std::array<std::uint8_t, 8> eof{};
      eof.fill(std::uint8_t{0xcc});
      CHECK(client->get(env.bucket, k_fixture_key, payload.size(), eof.size(), eof.data()) == 0);
      CHECK(client->get(env.bucket, k_fixture_key, payload.size() + 99, eof.size(), eof.data()) ==
            0);
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
      auto ctx = std::make_shared<s3_rdma_ioctx>(cfg, make_real_client(cfg));
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

TEST_CASE("cuobj_rdma_client device GET fails loudly when RDMA SDK support is disabled",
          "[s3][rdma][client][integration][gpu]")
{
  using namespace s3_rdma_p3a_tests;
  if (!ensure_minio_env()) { return; }
  if (!cuda_device_available()) { return; }

  auto const env = read_minio_env();
  auto client    = make_real_client(object_store_cfg(env, signing_case::presigned));

  void* device_dst = nullptr;
  REQUIRE(cudaMalloc(&device_dst, 16) == cudaSuccess);
  try {
    (void)client->get(env.bucket, k_fixture_key, 0, 16, device_dst);
    FAIL("expected device GET to fail in the default build");
  } catch (std::runtime_error const& e) {
    CHECK(s3_rdma_p3a_tests::contains(e.what(), "SIRIUS_ENABLE_S3_RDMA"));
  }
  CHECK(cudaFree(device_dst) == cudaSuccess);
}

TEST_CASE("cuobj_rdma_client surfaces MinIO authentication failures",
          "[s3][rdma][client][integration]")
{
  using namespace s3_rdma_p3a_tests;
  if (!ensure_minio_env()) { return; }

  auto const env = read_minio_env();

  for (auto signing : {signing_case::presigned, signing_case::header}) {
    DYNAMIC_SECTION(signing_name(signing))
    {
      auto client = make_real_client(object_store_cfg(env, signing, "wrong-secret-key"));

      CHECK_THROWS_AS(client->head(env.bucket, k_fixture_key), std::runtime_error);

      std::array<std::uint8_t, 16> dst{};
      CHECK_THROWS_AS(client->get(env.bucket, k_fixture_key, 0, dst.size(), dst.data()),
                      std::runtime_error);
    }
  }
}
