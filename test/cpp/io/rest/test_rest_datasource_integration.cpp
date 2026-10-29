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
#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/s3_backend.hpp"
#include "utils/s3_test_env.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using cucascade::io::io_context_type;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;
using sirius::test::s3::env_or;
using sirius::test::s3::require_env;
using sirius::test::s3::require_rest_ioctx;
using sirius::test::s3::single_gpu_index;
using namespace std::chrono_literals;

struct scan_manager_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  std::shared_ptr<const sirius::memory::topology_index> topology =
    single_gpu_index(/*numa_node=*/0);
};

scan_manager_config make_s3_rest_config()
{
  scan_manager_config cfg{};
  cfg.backend                 = sirius::scan_manager::io_backend::native;
  cfg.object_store.endpoint   = require_env("SIRIUS_TEST_S3_ENDPOINT");
  cfg.object_store.region     = env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  cfg.object_store.access_key = require_env("SIRIUS_TEST_S3_ACCESS_KEY");
  cfg.object_store.secret_key = require_env("SIRIUS_TEST_S3_SECRET_KEY");
  cfg.object_store.tls_verify = false;
  cfg.rest.request_timeout_s  = 30;
  cfg.rest.max_connections    = 8;
  cfg.rest_n_reactors         = 1;
  cfg.cache.mode              = cucascade::io::cache::cache_mode::none;
  return cfg;
}

scan_manager_config make_tls_s3_rest_config()
{
  auto cfg                        = make_s3_rest_config();
  cfg.object_store.endpoint       = require_env("SIRIUS_TEST_S3_HTTPS_ENDPOINT");
  cfg.object_store.tls_verify     = true;
  cfg.object_store.ca_bundle_path = require_env("SIRIUS_TEST_S3_CA_BUNDLE");
  return cfg;
}

std::filesystem::path local_fixture_path(std::string const& key)
{
  return std::filesystem::path{require_env("SIRIUS_TEST_S3_LOCAL_DIR")} / key;
}

std::vector<std::uint8_t> read_binary_file(std::filesystem::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> chars((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return std::vector<std::uint8_t>(chars.begin(), chars.end());
}

void require_bytes_equal(std::span<std::uint8_t const> got, std::span<std::uint8_t const> expected)
{
  REQUIRE(got.size() == expected.size());
  CHECK(std::equal(got.begin(), got.end(), expected.begin(), expected.end()));
}

std::vector<std::uint8_t> copy_device_to_host(rmm::device_buffer const& device,
                                              std::size_t size,
                                              rmm::cuda_stream const& stream)
{
  std::vector<std::uint8_t> out(size);
  REQUIRE(
    cudaMemcpyAsync(out.data(), device.data(), size, cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  stream.synchronize();
  return out;
}

bool s3_env_unavailable()
{
  return sirius::test::s3::skip_or_fail_unless(sirius::test::ensure_s3_test_env(),
                                               "SeaweedFS test environment is not available");
}

constexpr std::array<std::uint8_t, 16> HELLO_BYTES{
  's', 'i', 'r', 'i', 'u', 's', '-', 's', '3', '-', 'h', 'e', 'l', 'l', 'o', '\n'};

}  // namespace

TEST_CASE("rest datasource reads the SeaweedFS hello fixture through scan_manager",
          "[s3][integration][rest]")
{
  if (s3_env_unavailable()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle, make_s3_rest_config(), *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://" + bucket + "/hello.txt");
  require_rest_ioctx(datasource);
  CHECK(datasource->io_ctx()->type() == io_context_type::restful);
  REQUIRE(datasource->size() == HELLO_BYTES.size());

  std::array<std::uint8_t, 16> got{};
  REQUIRE(datasource->host_read(0, got.size(), got.data()) == got.size());
  CHECK(got == HELLO_BYTES);
}

TEST_CASE("rest datasource reads exact host ranges and clips EOF on SeaweedFS fixtures",
          "[s3][integration][rest]")
{
  if (s3_env_unavailable()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const small  = read_binary_file(local_fixture_path("small.bin"));
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle, make_s3_rest_config(), *fixture.memory, fixture.topology};
  auto datasource = manager.create_datasource("s3://" + bucket + "/small.bin");
  require_rest_ioctx(datasource);

  REQUIRE(datasource->size() == small.size());

  std::vector<std::uint8_t> full(small.size());
  REQUIRE(datasource->host_read(0, full.size(), full.data()) == full.size());
  require_bytes_equal(full, small);

  std::vector<std::uint8_t> mid(4096);
  std::size_t const mid_offset = 1234;
  REQUIRE(datasource->host_read(mid_offset, mid.size(), mid.data()) == mid.size());
  require_bytes_equal(mid, std::span<std::uint8_t const>(small.data() + mid_offset, mid.size()));

  std::vector<std::uint8_t> crossing(64, std::uint8_t{0xaa});
  std::size_t const tail_offset = small.size() - 17;
  REQUIRE(datasource->host_read(tail_offset, crossing.size(), crossing.data()) == 17);
  require_bytes_equal(std::span<std::uint8_t const>(crossing.data(), 17),
                      std::span<std::uint8_t const>(small.data() + tail_offset, 17));

  std::array<std::uint8_t, 8> eof{};
  eof.fill(std::uint8_t{0xcc});
  CHECK(datasource->host_read(small.size(), eof.size(), eof.data()) == 0);
  CHECK(datasource->host_read(small.size() + 99, eof.size(), eof.data()) == 0);
  CHECK(std::all_of(eof.begin(), eof.end(), [](std::uint8_t b) { return b == 0xcc; }));
}

TEST_CASE("rest datasource fans out host_read_ranges_async against the SeaweedFS medium fixture",
          "[s3][integration][rest]")
{
  if (s3_env_unavailable()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const medium = read_binary_file(local_fixture_path("medium.bin"));
  scan_manager_fixture fixture;
  auto cfg                 = make_s3_rest_config();
  cfg.rest.max_connections = 4;
  cfg.rest.merge_max_gap   = 0;  // no logical gap coalescing; physical GET sizing is worker-owned
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, cfg, *fixture.memory, fixture.topology};
  auto datasource = manager.create_datasource("s3://" + bucket + "/medium.bin");
  require_rest_ioctx(datasource);

  std::vector<std::pair<std::size_t, std::size_t>> const ranges{
    {17, 257},
    {64 * 1024 + 9, 1024},
    {2 * 1024 * 1024 + 11, 4096},
    {medium.size() - 333, 333},
  };
  std::vector<std::vector<std::uint8_t>> buffers;
  std::vector<cucascade::io::slice> segments;
  buffers.reserve(ranges.size());
  segments.reserve(ranges.size());
  std::size_t total = 0;
  for (auto const& [offset, size] : ranges) {
    buffers.emplace_back(size);
    segments.emplace_back(offset, size, buffers.back().data());
    total += size;
  }

  auto future = datasource->host_read_ranges_async(segments);
  REQUIRE(future.wait_for(5s) == std::future_status::ready);
  REQUIRE(future.get() == total);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto const [offset, size] = ranges[i];
    require_bytes_equal(buffers[i], std::span<std::uint8_t const>(medium.data() + offset, size));
  }
}

TEST_CASE("rest datasource stages device reads through host memory for SeaweedFS fixtures",
          "[s3][integration][rest]")
{
  if (s3_env_unavailable()) { return; }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    WARN("Skipping rest datasource device-read integration test: no CUDA device");
    return;
  }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const small  = read_binary_file(local_fixture_path("small.bin"));
  auto const medium = read_binary_file(local_fixture_path("medium.bin"));

  scan_manager_fixture fixture;
  auto cfg                 = make_s3_rest_config();
  cfg.rest.max_connections = 4;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, cfg, *fixture.memory, fixture.topology};

  SECTION("single chunk")
  {
    auto datasource = manager.create_datasource("s3://" + bucket + "/small.bin");
    require_rest_ioctx(datasource);

    rmm::cuda_stream stream;
    rmm::device_buffer dst(small.size(), stream);
    REQUIRE(datasource->device_read(
              0, small.size(), static_cast<std::uint8_t*>(dst.data()), stream) == small.size());
    auto got = copy_device_to_host(dst, small.size(), stream);
    require_bytes_equal(got, small);
  }

  SECTION("multi chunk")
  {
    auto datasource = manager.create_datasource("s3://" + bucket + "/medium.bin");
    require_rest_ioctx(datasource);

    std::size_t const offset = 123;
    std::size_t const size   = (3UL << 20) + 17;
    rmm::cuda_stream stream;
    rmm::device_buffer dst(size, stream);
    REQUIRE(datasource->device_read(offset, size, static_cast<std::uint8_t*>(dst.data()), stream) ==
            size);
    auto got = copy_device_to_host(dst, size, stream);
    require_bytes_equal(got, std::span<std::uint8_t const>(medium.data() + offset, size));
  }

  SECTION("vectored ranges")
  {
    auto datasource = manager.create_datasource("s3://" + bucket + "/medium.bin");
    require_rest_ioctx(datasource);

    std::size_t const first_offset  = 7;
    std::size_t const first_size    = 4096;
    std::size_t const second_offset = (1UL << 20) + 3;
    std::size_t const second_size   = (2UL << 20) + 5;
    rmm::cuda_stream stream;
    rmm::device_buffer first(first_size, stream);
    rmm::device_buffer second(second_size, stream);
    std::array<cucascade::io::slice, 2> const slices{
      cucascade::io::slice{first_offset, first_size, static_cast<std::uint8_t*>(first.data())},
      cucascade::io::slice{second_offset, second_size, static_cast<std::uint8_t*>(second.data())}};
    auto future = datasource->device_read_ranges_async(slices, stream);
    REQUIRE(future.wait_for(10s) == std::future_status::ready);
    REQUIRE(future.get() == first_size + second_size);
    require_bytes_equal(copy_device_to_host(first, first_size, stream),
                        std::span<std::uint8_t const>(medium.data() + first_offset, first_size));
    require_bytes_equal(copy_device_to_host(second, second_size, stream),
                        std::span<std::uint8_t const>(medium.data() + second_offset, second_size));
  }
}

TEST_CASE("rest datasource reads through the TLS SeaweedFS endpoint with the harness CA bundle",
          "[s3][integration][rest]")
{
  if (s3_env_unavailable()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle, make_tls_s3_rest_config(), *fixture.memory, fixture.topology};
  auto datasource = manager.create_datasource("s3://" + bucket + "/hello.txt");
  require_rest_ioctx(datasource);

  std::array<std::uint8_t, 16> got{};
  REQUIRE(datasource->host_read(0, got.size(), got.data()) == got.size());
  CHECK(got == HELLO_BYTES);
}
