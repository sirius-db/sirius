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
#include "io/rdma/cuobj_rdma_reactor.hpp"
#include "io/rdma/mock_rdma_client.hpp"
#include "io/s3/s3_rdma_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "io/templated_ioctx.hpp"
#include "rdma_test_transport.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
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

static_assert(sirius::io::io_reactor_c<sirius::io::rdma::cuobj_rdma_reactor>);
static_assert(sirius::io::reactor_has_device_rx<sirius::io::rdma::cuobj_rdma_reactor>);
static_assert(!sirius::io::reactor_has_host_to_device_rx<sirius::io::rdma::cuobj_rdma_reactor>);
static_assert(!sirius::io::reactor_has_vector_host_rx<sirius::io::rdma::cuobj_rdma_reactor>);

namespace {

using sirius::io::object_store_config;
using sirius::io::rdma::cuobj_rdma_reactor;
using sirius::io::s3::s3_rdma_ioctx;
using sirius::test::rdma::mock_transport_fixture;
using sirius::test::rdma::seeded_mock_transport;
using namespace std::chrono_literals;

constexpr std::size_t kMaxInflight = 2;
constexpr std::size_t kSlotSize    = 64UL << 10;

std::size_t ceil_div(std::size_t value, std::size_t divisor)
{
  return (value + divisor - 1) / divisor;
}

object_store_config make_mock_rdma_config(std::size_t max_inflight = kMaxInflight,
                                          std::size_t slot_size    = kSlotSize)
{
  object_store_config cfg;
  cfg.s3_transport            = object_store_config::transport::RDMA;
  cfg.s3_rdma_max_inflight    = max_inflight;
  cfg.s3_rdma_arena_slot_size = slot_size;
  cfg.endpoint                = "mock-rdma-endpoint";
  cfg.region                  = "us-east-1";
  cfg.access_key              = "mock-access-key";
  cfg.secret_key              = "mock-secret-key";
  return cfg;
}

std::vector<std::uint8_t> pattern_bytes(std::size_t size, std::uint8_t salt = 17)
{
  std::vector<std::uint8_t> out(size);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::uint8_t>((i * 131U + salt) & 0xffU);
  }
  return out;
}

std::span<const std::uint8_t> slice(std::vector<std::uint8_t> const& bytes,
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

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA mock reactor GPU test: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

std::shared_ptr<mock_transport_fixture> seeded_transport(std::string key,
                                                         std::vector<std::uint8_t> bytes,
                                                         std::string bucket = "bucket")
{
  return seeded_mock_transport(std::move(bucket), std::move(key), std::move(bytes));
}

std::shared_ptr<s3_rdma_ioctx> make_started_ioctx(std::shared_ptr<mock_transport_fixture> transport,
                                                  object_store_config cfg = make_mock_rdma_config())
{
  auto ctx = std::make_shared<s3_rdma_ioctx>(std::move(cfg), transport->clients());
  ctx->start();
  return ctx;
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
    FAIL("expected RDMA read future to fail");
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

std::vector<std::uint8_t> read_device_object_or_throw(std::shared_ptr<s3_rdma_ioctx> const& ctx,
                                                      std::string const& key,
                                                      std::size_t size)
{
  if (cudaSetDevice(0) != cudaSuccess) { throw std::runtime_error("cudaSetDevice failed"); }

  auto ds = ctx->open_datasource("s3://bucket/" + key);
  rmm::cuda_stream stream;
  rmm::device_buffer device(size, stream);
  auto fut = ds->device_read_async(0, size, static_cast<std::uint8_t*>(device.data()), stream);
  if (fut.wait_for(5s) != std::future_status::ready) {
    throw std::runtime_error("device read timed out");
  }
  auto const n = fut.get();
  if (n != size) { throw std::runtime_error("device read returned the wrong byte count"); }
  return copy_device_to_host(device.data(), size, stream);
}

}  // namespace

TEST_CASE("cuobj_rdma_reactor exposes the P2 structural contract", "[s3][rdma][reactor]")
{
  CHECK(cuobj_rdma_reactor::supports("s3://bucket/key"));
  CHECK_FALSE(cuobj_rdma_reactor::supports("file:///tmp/object"));
  CHECK_FALSE(cuobj_rdma_reactor::supports("https://bucket/key"));
  CHECK(cuobj_rdma_reactor::preferred_prefetching_stage() ==
        sirius::io::cache::prefetching_stage::none);

  auto ctx = make_started_ioctx(std::make_shared<mock_transport_fixture>());
  CHECK(ctx->type() == sirius::io::io_context_type::rdma);
  CHECK(ctx->supports("s3://bucket/key"));
  CHECK(ctx->supports_device_read());
  CHECK_FALSE(ctx->supports_host_to_device_read());
  CHECK_FALSE(ctx->supports_vector_host_read());
  CHECK_FALSE(ctx->can_use_prefetching_cache());
}

TEST_CASE("mock RDMA control client reports absent objects as HTTP results", "[s3][rdma][reactor]")
{
  sirius::io::rdma::mock_s3_control_client client;
  auto const result = client.head(sirius::io::rdma::rx_route{"bucket", "missing"});
  CHECK(result.outcome.http_status == 404);
  CHECK(result.outcome.transport_error.empty());
}

TEST_CASE("s3_rdma_ioctx mock host reads exact and EOF-clipped ranges", "[s3][rdma][reactor]")
{
  auto payload   = pattern_bytes(512);
  auto transport = seeded_transport("host-object", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = ctx->open_datasource("s3://bucket/host-object");

  REQUIRE(ds->size() == payload.size());

  std::vector<std::uint8_t> got(41);
  auto const n = ds->host_read(17, got.size(), got.data());
  REQUIRE(n == got.size());
  require_bytes_equal(got, slice(payload, 17, got.size()));

  std::vector<std::uint8_t> clipped(64);
  auto const clipped_n = ds->host_read(payload.size() - 19, clipped.size(), clipped.data());
  REQUIRE(clipped_n == 19);
  clipped.resize(clipped_n);
  require_bytes_equal(clipped, slice(payload, payload.size() - 19, clipped_n));
}

TEST_CASE("s3_rdma_ioctx mock device read chunks and delivers bytes", "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(3 * kSlotSize + 123);
  auto transport = seeded_transport("device-object", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = ctx->open_datasource("s3://bucket/device-object");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto fut =
    ds->device_read_async(0, payload.size(), static_cast<std::uint8_t*>(device.data()), stream);
  REQUIRE(require_ready_value(fut) == payload.size());

  auto got = copy_device_to_host(device.data(), payload.size(), stream);
  require_bytes_equal(got, payload);
  CHECK(transport->gets_issued() == ceil_div(payload.size(), kSlotSize));
}

TEST_CASE("s3_rdma_ioctx mock device read honors offsets across chunks", "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  auto payload          = pattern_bytes(4 * kSlotSize + 37);
  constexpr auto offset = kSlotSize - 17;
  constexpr auto size   = 2 * kSlotSize + 33;
  auto transport        = seeded_transport("offset-object", payload);
  auto ctx              = make_started_ioctx(transport);
  auto ds               = ctx->open_datasource("s3://bucket/offset-object");
  rmm::cuda_stream stream;
  rmm::device_buffer device(size, stream);

  auto fut = ds->device_read_async(offset, size, static_cast<std::uint8_t*>(device.data()), stream);
  REQUIRE(require_ready_value(fut) == size);

  auto got = copy_device_to_host(device.data(), size, stream);
  require_bytes_equal(got, slice(payload, offset, size));
}

TEST_CASE("cuobj_rdma_reactor mock gate enforces max inflight backpressure",
          "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(5 * kSlotSize + 1);
  auto transport = seeded_transport("gated-object", payload);
  transport->close_get_gate();

  auto ctx = make_started_ioctx(transport);
  auto ds  = ctx->open_datasource("s3://bucket/gated-object");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto fut =
    ds->device_read_async(0, payload.size(), static_cast<std::uint8_t*>(device.data()), stream);

  REQUIRE(wait_until([&] { return transport->gets_issued() >= kMaxInflight; }));
  CHECK(transport->peak_concurrent_gets() == kMaxInflight);
  auto const blocked_snapshot = ctx->perf_snapshot();
  CHECK(blocked_snapshot.inflight_peak == kMaxInflight);
  CHECK(blocked_snapshot.slots_in_use_peak == kMaxInflight);
  CHECK(fut.wait_for(50ms) != std::future_status::ready);

  transport->open_get_gate();
  REQUIRE(require_ready_value(fut) == payload.size());
  CHECK(transport->gets_issued() == ceil_div(payload.size(), kSlotSize));

  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.requests_total == 1);
  CHECK(snapshot.envelope_depth_peak == 1);
  CHECK(snapshot.inflight_peak == kMaxInflight);
  CHECK(snapshot.slots_in_use_peak == kMaxInflight);
  CHECK(snapshot.envelope_wait_total == 0);
}

TEST_CASE("cuobj_rdma_reactor reports short mock reads as failed futures",
          "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  constexpr std::size_t kShort = kSlotSize / 2;
  auto payload                 = pattern_bytes(kSlotSize);
  auto transport               = seeded_transport("short-object", payload);
  transport->short_write(kShort);

  auto ctx = make_started_ioctx(transport);
  auto ds  = ctx->open_datasource("s3://bucket/short-object");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto fut =
    ds->device_read_async(0, payload.size(), static_cast<std::uint8_t*>(device.data()), stream);
  auto const message = require_ready_error(fut);
  CHECK((message.find("short") != std::string::npos ||
         message.find(std::to_string(kShort)) != std::string::npos ||
         message.find(std::to_string(payload.size())) != std::string::npos));
  CHECK(transport->gets_issued() == 1);

  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.retries_total == 0);
  CHECK(snapshot.short_read_total == 1);
  CHECK(snapshot.fail_stop_total == 1);
  CHECK(snapshot.arena_leak_total == 1);

  auto follow_up =
    ds->device_read_async(0, payload.size(), static_cast<std::uint8_t*>(device.data()), stream);
  CHECK_FALSE(require_ready_error(follow_up).empty());
  CHECK(transport->gets_issued() == 1);
  CHECK(ctx->perf_snapshot().fail_stop_total == 1);
}

TEST_CASE("cuobj_rdma_reactor propagates mock transport errors", "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(kSlotSize);
  auto transport = seeded_transport("error-object", payload);
  transport->fail_gets("mock transport down");

  auto ctx = make_started_ioctx(transport);
  auto ds  = ctx->open_datasource("s3://bucket/error-object");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto fut =
    ds->device_read_async(0, payload.size(), static_cast<std::uint8_t*>(device.data()), stream);
  CHECK(require_ready_error(fut).find("mock transport down") != std::string::npos);
  CHECK(transport->gets_issued() == 1);

  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.retries_total == 0);
  CHECK(snapshot.fail_stop_total == 1);
  CHECK(snapshot.arena_leak_total == 1);

  auto follow_up =
    ds->device_read_async(0, payload.size(), static_cast<std::uint8_t*>(device.data()), stream);
  CHECK_FALSE(require_ready_error(follow_up).empty());
  CHECK(transport->gets_issued() == 1);
  CHECK(ctx->perf_snapshot().fail_stop_total == 1);
}

TEST_CASE("s3_rdma_ioctx mock open fails on missing object", "[s3][rdma][reactor]")
{
  auto ctx = make_started_ioctx(std::make_shared<mock_transport_fixture>());
  CHECK_THROWS_AS(ctx->open_datasource("s3://bucket/absent"), std::runtime_error);
}

TEST_CASE("cuobj_rdma_reactor resolves zero-length device reads without work",
          "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(128);
  auto transport = seeded_transport("zero-object", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = ctx->open_datasource("s3://bucket/zero-object");
  rmm::cuda_stream stream;
  rmm::device_buffer device(1, stream);

  auto zero = ds->device_read_async(0, 0, static_cast<std::uint8_t*>(device.data()), stream);
  REQUIRE(require_ready_value(zero, 250ms) == 0);

  auto clipped = ds->device_read_async(
    payload.size() + 32, 4096, static_cast<std::uint8_t*>(device.data()), stream);
  REQUIRE(require_ready_value(clipped, 250ms) == 0);
}

TEST_CASE("cuobj_rdma_reactor shutdown completes issued GETs and errors unissued chunks",
          "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(2 * kSlotSize + 7);
  auto transport = seeded_transport("shutdown-object", payload);
  transport->close_get_gate();

  auto ctx = make_started_ioctx(transport);
  auto ds  = ctx->open_datasource("s3://bucket/shutdown-object");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto fut =
    ds->device_read_async(0, payload.size(), static_cast<std::uint8_t*>(device.data()), stream);
  REQUIRE(wait_until([&] { return transport->gets_issued() >= kMaxInflight; }));

  std::promise<void> first_shutdown_started;
  std::promise<void> second_shutdown_started;
  auto first_started        = first_shutdown_started.get_future();
  auto second_started       = second_shutdown_started.get_future();
  auto first_shutdown       = std::async(std::launch::async, [&] {
    first_shutdown_started.set_value();
    ctx->shutdown();
  });
  auto second_shutdown      = std::async(std::launch::async, [&] {
    second_shutdown_started.set_value();
    ctx->shutdown();
  });
  auto const first_entered  = first_started.wait_for(5s) == std::future_status::ready;
  auto const second_entered = second_started.wait_for(5s) == std::future_status::ready;
  auto const first_blocked =
    first_entered && first_shutdown.wait_for(50ms) != std::future_status::ready;
  auto const second_blocked =
    second_entered && second_shutdown.wait_for(50ms) != std::future_status::ready;

  transport->open_get_gate();
  auto const first_finished  = first_shutdown.wait_for(5s) == std::future_status::ready;
  auto const second_finished = second_shutdown.wait_for(5s) == std::future_status::ready;
  REQUIRE(first_finished);
  REQUIRE(second_finished);
  first_shutdown.get();
  second_shutdown.get();
  CHECK(first_entered);
  CHECK(second_entered);
  CHECK(first_blocked);
  CHECK(second_blocked);
  ctx->shutdown();

  auto const message = require_ready_error(fut);
  CHECK(message.find("transport closed") != std::string::npos);
  CHECK(transport->gets_issued() == kMaxInflight);

  constexpr auto completed_bytes = kMaxInflight * kSlotSize;
  auto const snapshot            = ctx->perf_snapshot();
  CHECK(snapshot.bytes_total == completed_bytes);
  auto got = copy_device_to_host(device.data(), completed_bytes, stream);
  require_bytes_equal(got, slice(payload, 0, completed_bytes));
}

TEST_CASE("cuobj_rdma_reactor recycles arena slots across sequential reads",
          "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(3 * kSlotSize + 5);
  auto transport = seeded_transport("reuse-object", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = ctx->open_datasource("s3://bucket/reuse-object");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  for (int pass = 0; pass < 2; ++pass) {
    auto fut =
      ds->device_read_async(0, payload.size(), static_cast<std::uint8_t*>(device.data()), stream);
    REQUIRE(require_ready_value(fut) == payload.size());
    auto got = copy_device_to_host(device.data(), payload.size(), stream);
    require_bytes_equal(got, payload);
  }
}

TEST_CASE("cuobj_rdma_reactor fails device reads issued after shutdown cleanly",
          "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(kSlotSize);
  auto transport = seeded_transport("post-shutdown-object", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = ctx->open_datasource("s3://bucket/post-shutdown-object");
  ctx->shutdown();

  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  bool failed = false;
  try {
    auto fut =
      ds->device_read_async(0, payload.size(), static_cast<std::uint8_t*>(device.data()), stream);
    auto const message = require_ready_error(fut);
    failed             = !message.empty();
  } catch (std::exception const&) {
    failed = true;
  }
  CHECK(failed);
}

TEST_CASE("cuobj_rdma_reactor serves concurrent mock readers through one ioctx",
          "[s3][rdma][reactor][gpu]")
{
  if (!cuda_device_available()) { return; }

  auto left      = pattern_bytes(2 * kSlotSize + 11, 3);
  auto right     = pattern_bytes(2 * kSlotSize + 29, 97);
  auto transport = std::make_shared<mock_transport_fixture>();
  transport->put_object("bucket", "left", left);
  transport->put_object("bucket", "right", right);
  auto ctx = make_started_ioctx(transport);

  auto left_future = std::async(
    std::launch::async, [&] { return read_device_object_or_throw(ctx, "left", left.size()); });
  auto right_future = std::async(
    std::launch::async, [&] { return read_device_object_or_throw(ctx, "right", right.size()); });

  REQUIRE(left_future.wait_for(5s) == std::future_status::ready);
  REQUIRE(right_future.wait_for(5s) == std::future_status::ready);
  auto left_got  = left_future.get();
  auto right_got = right_future.get();
  require_bytes_equal(left_got, left);
  require_bytes_equal(right_got, right);
}
