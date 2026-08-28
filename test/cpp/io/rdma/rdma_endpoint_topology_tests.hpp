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
#include "io/s3/s3_rdma_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "rdma_test_transport.hpp"
#include "utils/log_test_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace s3_rdma_endpoint_topology_tests {

using sirius::io::object_store_config;
using sirius::io::s3::detect_endpoint_topology;
using sirius::io::s3::endpoint_topology;
using sirius::io::s3::s3_rdma_ioctx;
using sirius::test::rdma::mock_transport_fixture;
using sirius::test::rdma::seeded_mock_transport;
using namespace std::chrono_literals;

constexpr std::size_t k_slot_size = 64UL << 10;

struct endpoint_pair {
  std::string_view name;
  std::string_view host;
  std::string_view data;
};

object_store_config make_config(std::string_view host_endpoint, std::string_view data_endpoint)
{
  object_store_config cfg;
  cfg.endpoint                     = host_endpoint;
  cfg.region                       = "us-east-1";
  cfg.access_key                   = "host-access-key";
  cfg.secret_key                   = "host-secret-key";
  cfg.s3_signing_mode              = object_store_config::signing_mode::header;
  cfg.s3_transport                 = object_store_config::transport::RDMA;
  cfg.s3_rdma_max_inflight         = 1;
  cfg.s3_rdma_arena_slot_size      = k_slot_size;
  cfg.s3_rdma_data.endpoint        = data_endpoint;
  cfg.s3_rdma_data.region          = cfg.region;
  cfg.s3_rdma_data.access_key      = cfg.access_key;
  cfg.s3_rdma_data.secret_key      = cfg.secret_key;
  cfg.s3_rdma_data.s3_signing_mode = object_store_config::signing_mode::header;
  return cfg;
}

std::shared_ptr<s3_rdma_ioctx> make_ioctx(object_store_config cfg,
                                          std::shared_ptr<mock_transport_fixture> const& transport)
{
  return std::make_shared<s3_rdma_ioctx>(std::move(cfg), transport->clients());
}

bool contains_all(std::string_view message, std::initializer_list<std::string_view> tokens)
{
  return std::all_of(tokens.begin(), tokens.end(), [&](std::string_view token) {
    return message.find(token) != std::string_view::npos;
  });
}

std::size_t count_messages(std::span<const sirius::test::recording_log_sink::record> records,
                           std::initializer_list<std::string_view> tokens)
{
  return static_cast<std::size_t>(
    std::count_if(records.begin(), records.end(), [&](auto const& record) {
      return contains_all(record.message, tokens);
    }));
}

std::size_t count_warnings(std::span<const sirius::test::recording_log_sink::record> records,
                           std::initializer_list<std::string_view> tokens)
{
  return static_cast<std::size_t>(
    std::count_if(records.begin(), records.end(), [&](auto const& record) {
      return record.level == sirius::log::level::warn && contains_all(record.message, tokens);
    }));
}

std::vector<std::uint8_t> pattern_bytes(std::size_t size)
{
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + 29U) & 0xffU);
  }
  return bytes;
}

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA endpoint-topology transfer test: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

std::vector<std::uint8_t> copy_device_to_host(void const* device_data,
                                              std::size_t size,
                                              rmm::cuda_stream_view stream)
{
  std::vector<std::uint8_t> bytes(size);
  auto const result =
    cudaMemcpyAsync(bytes.data(), device_data, size, cudaMemcpyDeviceToHost, stream.value());
  REQUIRE(result == cudaSuccess);
  stream.synchronize();
  return bytes;
}

void require_transfer_semantics(object_store_config cfg, std::string_view key)
{
  auto const payload = pattern_bytes(4096);
  auto transport =
    seeded_mock_transport("bucket", std::string{key}, std::vector<std::uint8_t>{payload});
  auto ctx = make_ioctx(std::move(cfg), transport);
  ctx->start();
  auto datasource = ctx->open_datasource("s3://bucket/" + std::string{key});

  std::vector<std::uint8_t> host_bytes(payload.size());
  REQUIRE(datasource->host_read(0, host_bytes.size(), host_bytes.data()) == payload.size());
  CHECK(host_bytes == payload);

  rmm::cuda_stream stream;
  rmm::device_buffer device_bytes(payload.size(), stream);
  auto future = datasource->device_read_async(0,
                                              payload.size(),
                                              static_cast<std::uint8_t*>(device_bytes.data()),
                                              rmm::cuda_stream_view{stream});
  REQUIRE(future.wait_for(5s) == std::future_status::ready);
  REQUIRE(future.get() == payload.size());
  CHECK(copy_device_to_host(device_bytes.data(), payload.size(), stream) == payload);

  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fail_stop_total == 0);
  CHECK(snapshot.retries_total == 0);
  ctx->shutdown();
}

}  // namespace s3_rdma_endpoint_topology_tests

TEST_CASE("s3_rdma endpoint topology normalizes provably equal addresses", "[s3][rdma][topology]")
{
  using namespace s3_rdma_endpoint_topology_tests;

  constexpr std::array rows{
    endpoint_pair{"identical", "http://store.example", "http://store.example"},
    endpoint_pair{"scheme-and-host-case", "HTTP://Store.Example", "http://store.example"},
    endpoint_pair{"http-default-port", "http://store.example:80", "http://store.example"},
    endpoint_pair{"https-default-port", "https://store.example", "https://store.example:443"},
    endpoint_pair{"trailing-slash", "http://store.example/", "http://store.example"},
    endpoint_pair{"combined", "HTTPS://Store.Example/", "https://store.example:443"},
  };

  for (auto const& row : rows) {
    DYNAMIC_SECTION(row.name)
    {
      CHECK(detect_endpoint_topology(row.host, row.data) == endpoint_topology::same_address);
    }
  }
}

TEST_CASE("s3_rdma endpoint topology conservatively keeps unequal addresses split",
          "[s3][rdma][topology]")
{
  using namespace s3_rdma_endpoint_topology_tests;

  constexpr std::array rows{
    endpoint_pair{"different-host", "http://store-a.example", "http://store-b.example"},
    endpoint_pair{"different-port", "http://store.example:8080", "http://store.example:8081"},
    endpoint_pair{"different-scheme", "http://store.example", "https://store.example"},
    endpoint_pair{"different-path", "http://store.example/a", "http://store.example/b"},
    endpoint_pair{"empty-data-endpoint", "http://store.example", ""},
    endpoint_pair{"cname-looking-alias", "http://store.example", "http://store-alias.example"},
  };

  for (auto const& row : rows) {
    DYNAMIC_SECTION(row.name)
    {
      CHECK(detect_endpoint_topology(row.host, row.data) == endpoint_topology::split);
    }
  }
}

TEST_CASE("s3_rdma ioctx exposes same-address topology at construction", "[s3][rdma][topology]")
{
  using namespace s3_rdma_endpoint_topology_tests;

  auto transport = std::make_shared<mock_transport_fixture>();
  auto ctx = make_ioctx(make_config("HTTP://Store.Example/", "http://store.example:80"), transport);
  CHECK(ctx->topology() == endpoint_topology::same_address);
}

TEST_CASE("s3_rdma ioctx exposes split topology at construction", "[s3][rdma][topology]")
{
  using namespace s3_rdma_endpoint_topology_tests;

  auto transport = std::make_shared<mock_transport_fixture>();
  auto ctx = make_ioctx(make_config("http://control.example", "http://data.example"), transport);
  CHECK(ctx->topology() == endpoint_topology::split);
}

TEST_CASE("s3_rdma startup logs the endpoint consistency model", "[s3][rdma][topology]")
{
  using namespace s3_rdma_endpoint_topology_tests;

  SECTION("same-address model names immutable keys")
  {
    sirius::test::scoped_recording_log_sink logs{"trace"};
    auto transport = std::make_shared<mock_transport_fixture>();
    auto ctx =
      make_ioctx(make_config("http://store.example", "HTTP://STORE.EXAMPLE:80/"), transport);
    ctx->start();
    ctx->shutdown();

    auto const records = logs.records();
    CHECK(count_messages(records, {"same_address", "immutable"}) == 1);
  }

  SECTION("split model names the publisher barrier")
  {
    sirius::test::scoped_recording_log_sink logs{"trace"};
    auto transport = std::make_shared<mock_transport_fixture>();
    auto ctx = make_ioctx(make_config("http://control.example", "http://data.example"), transport);
    ctx->start();
    ctx->shutdown();

    auto const records = logs.records();
    CHECK(count_messages(records, {"split", "barrier"}) == 1);
  }
}

TEST_CASE("s3_rdma warns only for same-address credential conflicts", "[s3][rdma][topology]")
{
  using namespace s3_rdma_endpoint_topology_tests;

  SECTION("same-address explicit credentials differ")
  {
    auto cfg                    = make_config("http://store.example", "http://store.example");
    cfg.s3_rdma_data.access_key = "different-data-access-key";
    cfg.s3_rdma_data.secret_key = "different-data-secret-key";
    sirius::test::scoped_recording_log_sink logs{"trace"};
    auto transport = std::make_shared<mock_transport_fixture>();
    auto ctx       = make_ioctx(std::move(cfg), transport);
    ctx->start();
    ctx->shutdown();

    auto const records = logs.records();
    CHECK(count_warnings(records, {"credentials differ"}) == 1);
  }

  SECTION("same-address inherited credentials do not warn")
  {
    auto cfg                    = make_config("http://store.example", "http://store.example");
    cfg.s3_rdma_data.access_key = "";
    cfg.s3_rdma_data.secret_key = "";
    sirius::test::scoped_recording_log_sink logs{"trace"};
    auto transport = std::make_shared<mock_transport_fixture>();
    auto ctx       = make_ioctx(std::move(cfg), transport);
    ctx->start();
    ctx->shutdown();

    auto const records = logs.records();
    CHECK(count_warnings(records, {"credentials differ"}) == 0);
  }

  SECTION("split endpoints may use distinct credentials")
  {
    auto cfg                    = make_config("http://control.example", "http://data.example");
    cfg.s3_rdma_data.access_key = "different-data-access-key";
    cfg.s3_rdma_data.secret_key = "different-data-secret-key";
    sirius::test::scoped_recording_log_sink logs{"trace"};
    auto transport = std::make_shared<mock_transport_fixture>();
    auto ctx       = make_ioctx(std::move(cfg), transport);
    ctx->start();
    ctx->shutdown();

    auto const records = logs.records();
    CHECK(count_warnings(records, {"credentials differ"}) == 0);
  }
}

TEST_CASE("s3_rdma transfer semantics do not depend on endpoint topology",
          "[s3][rdma][topology][gpu]")
{
  using namespace s3_rdma_endpoint_topology_tests;
  if (!cuda_device_available()) { return; }

  struct transfer_row {
    std::string_view name;
    object_store_config cfg;
  };
  std::array rows{
    transfer_row{"same-address", make_config("http://store.example", "http://store.example:80/")},
    transfer_row{"split", make_config("http://control.example", "http://data.example")},
  };

  for (auto& row : rows) {
    DYNAMIC_SECTION(row.name)
    {
      require_transfer_semantics(std::move(row.cfg), "topology-" + std::string{row.name});
    }
  }
}
