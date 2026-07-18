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
#include "io/rdma/cuobj_rdma_reactor.hpp"
#include "io/rdma/mock_rdma_client.hpp"
#include "io/s3/s3_rdma_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "rdma_test_transport.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace s3_rdma_p4a_tests {

using sirius::io::object_store_config;
using sirius::io::rdma::cuobj_rdma_io_object;
using sirius::io::rdma::cuobj_rdma_reactor;
using sirius::io::rdma::rdma_perf_snapshot;
using sirius::io::s3::s3_rdma_ioctx;
using sirius::test::rdma::mock_transport_fixture;
using sirius::test::rdma::seeded_mock_transport;
using namespace std::chrono_literals;

constexpr std::size_t k_max_inflight = 2;
constexpr std::size_t k_slot_size    = 64UL << 10;
constexpr std::string_view k_bucket  = "bucket";

std::size_t ceil_div(std::size_t value, std::size_t divisor)
{
  return (value + divisor - 1) / divisor;
}

object_store_config make_rdma_object_store_config()
{
  object_store_config cfg;
  cfg.s3_transport            = object_store_config::transport::RDMA;
  cfg.s3_rdma_max_inflight    = k_max_inflight;
  cfg.s3_rdma_arena_slot_size = k_slot_size;
  cfg.endpoint                = "mock-rdma-endpoint";
  cfg.region                  = "us-east-1";
  cfg.access_key              = "mock-access-key";
  cfg.secret_key              = "mock-secret-key";
  return cfg;
}

cuobj_rdma_reactor::config make_reactor_config()
{
  cuobj_rdma_reactor::config cfg;
  cfg.max_inflight    = k_max_inflight;
  cfg.arena_slot_size = k_slot_size;
  return cfg;
}

std::vector<std::uint8_t> pattern_bytes(std::size_t size, std::uint8_t salt = 19)
{
  std::vector<std::uint8_t> out(size);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::uint8_t>((i * 131U + salt) & 0xffU);
  }
  return out;
}

std::span<const std::uint8_t> byte_slice(std::vector<std::uint8_t> const& bytes,
                                         std::size_t offset,
                                         std::size_t size)
{
  return std::span<const std::uint8_t>(bytes.data() + offset, size);
}

void require_bytes_equal(std::span<const std::uint8_t> got, std::span<const std::uint8_t> expected)
{
  REQUIRE(got.size() == expected.size());
  CHECK(std::equal(got.begin(), got.end(), expected.begin(), expected.end()));
}

void require_zero_snapshot(rdma_perf_snapshot const& snapshot)
{
  CHECK(snapshot.bytes_total == 0);
  CHECK(snapshot.requests_total == 0);
  CHECK(snapshot.retries_total == 0);
  CHECK(snapshot.short_read_total == 0);
  CHECK(snapshot.error_total == 0);
  CHECK(snapshot.slot_wait_total == 0);
  CHECK(snapshot.flush_total == 0);
  CHECK(snapshot.inflight_peak == 0);
  CHECK(snapshot.envelope_wait_total == 0);
  CHECK(snapshot.envelope_wait_ns_total == 0);
  CHECK(snapshot.envelope_depth_peak == 0);
  CHECK(snapshot.slots_in_use_peak == 0);
  CHECK(snapshot.fail_stop_total == 0);
  CHECK(snapshot.arena_leak_total == 0);
}

void require_snapshots_equal(rdma_perf_snapshot const& lhs, rdma_perf_snapshot const& rhs)
{
  CHECK(lhs.bytes_total == rhs.bytes_total);
  CHECK(lhs.requests_total == rhs.requests_total);
  CHECK(lhs.retries_total == rhs.retries_total);
  CHECK(lhs.short_read_total == rhs.short_read_total);
  CHECK(lhs.error_total == rhs.error_total);
  CHECK(lhs.slot_wait_total == rhs.slot_wait_total);
  CHECK(lhs.flush_total == rhs.flush_total);
  CHECK(lhs.inflight_peak == rhs.inflight_peak);
  CHECK(lhs.envelope_wait_total == rhs.envelope_wait_total);
  CHECK(lhs.envelope_wait_ns_total == rhs.envelope_wait_ns_total);
  CHECK(lhs.envelope_depth_peak == rhs.envelope_depth_peak);
  CHECK(lhs.slots_in_use_peak == rhs.slots_in_use_peak);
  CHECK(lhs.fail_stop_total == rhs.fail_stop_total);
  CHECK(lhs.arena_leak_total == rhs.arena_leak_total);
}

std::shared_ptr<mock_transport_fixture> seeded_transport(std::string key,
                                                         std::vector<std::uint8_t> bytes,
                                                         std::string bucket = std::string{k_bucket})
{
  return seeded_mock_transport(std::move(bucket), std::move(key), std::move(bytes));
}

std::shared_ptr<s3_rdma_ioctx> make_started_ioctx(std::shared_ptr<mock_transport_fixture> transport)
{
  auto ctx = std::make_shared<s3_rdma_ioctx>(make_rdma_object_store_config(), transport->clients());
  ctx->start();
  return ctx;
}

std::unique_ptr<cuobj_rdma_reactor> make_started_reactor(
  std::shared_ptr<mock_transport_fixture> transport)
{
  auto reactor =
    std::make_unique<cuobj_rdma_reactor>(std::make_shared<cuobj_rdma_reactor::reactor_context>(
      make_reactor_config(), transport->clients()));
  reactor->start();
  return reactor;
}

std::unique_ptr<sirius::io::sirius_datasource> open_ds(std::shared_ptr<s3_rdma_ioctx> const& ctx,
                                                       std::string const& key)
{
  return ctx->open_datasource("s3://" + std::string{k_bucket} + "/" + key);
}

template <typename T>
T require_ready_value(std::future<T>& fut, std::chrono::milliseconds timeout = 5s)
{
  REQUIRE(fut.wait_for(timeout) == std::future_status::ready);
  return fut.get();
}

std::string require_ready_error(std::future<std::size_t>& fut,
                                std::chrono::milliseconds timeout = 5s)
{
  REQUIRE(fut.wait_for(timeout) == std::future_status::ready);
  try {
    (void)fut.get();
    FAIL("expected RDMA future to fail");
  } catch (std::exception const& e) {
    return e.what();
  }
  return {};
}

bool wait_until(std::function<bool()> predicate, std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) { return true; }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA retry/metrics GPU test: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

std::vector<std::uint8_t> copy_device_to_host(void const* device_data,
                                              std::size_t size,
                                              rmm::cuda_stream_view stream)
{
  std::vector<std::uint8_t> out(size);
  if (size == 0) { return out; }
  auto err = cudaMemcpyAsync(out.data(), device_data, size, cudaMemcpyDeviceToHost, stream.value());
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync D2H failed: ") + cudaGetErrorString(err));
  }
  stream.synchronize();
  return out;
}

std::future<std::size_t> issue_device_read(sirius::io::sirius_datasource& ds,
                                           std::size_t offset,
                                           std::size_t size,
                                           rmm::device_buffer& device,
                                           rmm::cuda_stream_view stream)
{
  return ds.device_read_async(offset, size, static_cast<std::uint8_t*>(device.data()), stream);
}

}  // namespace s3_rdma_p4a_tests

TEST_CASE("s3_rdma retry metrics start at zero on a fresh ioctx", "[s3][rdma][metrics]")
{
  using namespace s3_rdma_p4a_tests;

  auto ctx = make_started_ioctx(std::make_shared<mock_transport_fixture>());
  require_zero_snapshot(ctx->perf_snapshot());
}

TEST_CASE("s3_rdma ioctx perf snapshot is the single reactor aggregate", "[s3][rdma][metrics]")
{
  using namespace s3_rdma_p4a_tests;

  auto payload = pattern_bytes(2048);

  auto direct_transport = seeded_transport("direct", payload);
  auto direct_reactor   = make_started_reactor(direct_transport);
  cuobj_rdma_io_object direct_obj("s3://bucket/direct", "bucket", "direct", payload.size());
  std::vector<std::uint8_t> direct_got(payload.size());
  REQUIRE(direct_reactor->host_read(direct_obj, 0, direct_got.size(), direct_got.data()) ==
          direct_got.size());
  require_bytes_equal(direct_got, payload);

  auto ioctx_transport = seeded_transport("via-ioctx", payload);
  auto ctx             = make_started_ioctx(ioctx_transport);
  auto ds              = open_ds(ctx, "via-ioctx");
  std::vector<std::uint8_t> ioctx_got(payload.size());
  REQUIRE(ds->host_read(0, ioctx_got.size(), ioctx_got.data()) == ioctx_got.size());
  require_bytes_equal(ioctx_got, payload);

  require_snapshots_equal(ctx->perf_snapshot(), direct_reactor->perf_snapshot());
}

TEST_CASE("s3_rdma clean device read metrics count logical requests and bytes",
          "[s3][rdma][metrics][gpu]")
{
  using namespace s3_rdma_p4a_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(2 * k_slot_size + 31);
  auto transport = seeded_transport("metrics-clean", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = open_ds(ctx, "metrics-clean");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto fut = issue_device_read(*ds, 0, payload.size(), device, stream);
  REQUIRE(require_ready_value(fut) == payload.size());

  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.requests_total == 1);
  CHECK(snapshot.bytes_total == payload.size());
  CHECK(snapshot.error_total == 0);
  CHECK(snapshot.retries_total == 0);
  CHECK(snapshot.inflight_peak >= 1);
  CHECK(snapshot.inflight_peak <= k_max_inflight);
  CHECK(snapshot.slots_in_use_peak >= 1);
  CHECK(snapshot.slots_in_use_peak <= k_max_inflight);
  CHECK(snapshot.envelope_depth_peak == 1);
  CHECK(snapshot.flush_total == 0);
}

TEST_CASE("s3_rdma slot wait remains zero under gated backpressure", "[s3][rdma][metrics][gpu]")
{
  using namespace s3_rdma_p4a_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(5 * k_slot_size + 1);
  auto transport = seeded_transport("gated-metrics", payload);
  transport->close_get_gate();

  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "gated-metrics");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto fut = issue_device_read(*ds, 0, payload.size(), device, stream);
  REQUIRE(wait_until([&] { return transport->gets_issued() >= k_max_inflight; }));
  CHECK(transport->peak_concurrent_gets() <= k_max_inflight);

  transport->open_get_gate();
  REQUIRE(require_ready_value(fut) == payload.size());
  CHECK(ctx->perf_snapshot().slot_wait_total == 0);
  CHECK(ctx->perf_snapshot().slots_in_use_peak <= k_max_inflight);
}

TEST_CASE("s3_rdma perf counters accumulate monotonically and can be sampled in flight",
          "[s3][rdma][metrics][gpu]")
{
  using namespace s3_rdma_p4a_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(2 * k_slot_size);
  auto transport = seeded_transport("monotonic", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = open_ds(ctx, "monotonic");
  rmm::cuda_stream stream;
  rmm::device_buffer first(payload.size(), stream);
  rmm::device_buffer second(payload.size(), stream);

  auto first_fut = issue_device_read(*ds, 0, payload.size(), first, stream);
  REQUIRE(require_ready_value(first_fut) == payload.size());
  auto const after_first = ctx->perf_snapshot();

  transport->close_get_gate();
  auto second_fut = issue_device_read(*ds, 0, payload.size(), second, stream);
  REQUIRE(wait_until(
    [&] { return transport->gets_issued() >= ceil_div(payload.size(), k_slot_size) + 1; }));
  auto const during_second = ctx->perf_snapshot();
  CHECK(during_second.bytes_total >= after_first.bytes_total);
  CHECK(during_second.requests_total >= after_first.requests_total);
  transport->open_get_gate();

  REQUIRE(require_ready_value(second_fut) == payload.size());
  auto const after_second = ctx->perf_snapshot();

  CHECK(after_second.bytes_total == 2 * payload.size());
  CHECK(after_second.requests_total == 2);
  CHECK(after_second.retries_total == after_first.retries_total);
  CHECK(after_second.error_total == 0);
  CHECK(after_second.envelope_depth_peak >= after_first.envelope_depth_peak);
  CHECK(after_second.slots_in_use_peak >= after_first.slots_in_use_peak);
}
