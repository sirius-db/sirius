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
#include "io/rdma/rdma_client.hpp"
#include "io/s3/s3_rdma_ioctx.hpp"
#include "io/sirius_datasource.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace s3_rdma_f01_tests {

using sirius::io::object_store_config;
using sirius::io::rdma::cuda_delivery_ops;
using sirius::io::rdma::mock_rdma_client;
using sirius::io::rdma::rdma_client;
using sirius::io::s3::s3_rdma_ioctx;
using namespace std::chrono_literals;

constexpr std::size_t k_max_inflight = 2;
constexpr std::size_t k_slot_size    = 64UL << 10;
constexpr std::string_view k_bucket  = "bucket";

class recording_rdma_client final : public rdma_client {
 public:
  explicit recording_rdma_client(std::shared_ptr<mock_rdma_client> inner) : _inner(std::move(inner))
  {
    if (!_inner) { throw std::invalid_argument("recording_rdma_client: null inner client"); }
  }

  void put_object(std::string key,
                  std::vector<std::uint8_t> bytes,
                  std::string bucket = std::string{k_bucket})
  {
    _inner->put_object(std::move(bucket), std::move(key), std::move(bytes));
  }

  [[nodiscard]] std::size_t get_count() const
  {
    std::lock_guard lk{_mtx};
    return _get_destinations.size();
  }

  [[nodiscard]] void* get_destination(std::size_t zero_based_call) const
  {
    std::lock_guard lk{_mtx};
    return _get_destinations.at(zero_based_call);
  }

  [[nodiscard]] std::size_t register_count() const
  {
    std::lock_guard lk{_mtx};
    return _registered_bases.size();
  }

  [[nodiscard]] std::size_t deregister_count() const
  {
    std::lock_guard lk{_mtx};
    return _deregistered_bases.size();
  }

  std::size_t head(std::string_view bucket, std::string_view key) override
  {
    return _inner->head(bucket, key);
  }

  std::size_t get(std::string_view bucket,
                  std::string_view key,
                  std::size_t offset,
                  std::size_t size,
                  void* dst) override
  {
    {
      std::lock_guard lk{_mtx};
      _get_destinations.push_back(dst);
    }
    return _inner->get(bucket, key, offset, size, dst);
  }

  void register_memory(void* base, std::size_t bytes) override
  {
    {
      std::lock_guard lk{_mtx};
      _registered_bases.push_back(base);
    }
    _inner->register_memory(base, bytes);
  }

  void deregister_memory(void* base) noexcept override
  {
    {
      std::lock_guard lk{_mtx};
      _deregistered_bases.push_back(base);
    }
    _inner->deregister_memory(base);
  }

 private:
  std::shared_ptr<mock_rdma_client> _inner;
  mutable std::mutex _mtx;
  std::vector<void*> _get_destinations;
  std::vector<void*> _registered_bases;
  std::vector<void*> _deregistered_bases;
};

object_store_config make_config(std::size_t max_inflight = k_max_inflight)
{
  object_store_config cfg;
  cfg.s3_transport            = object_store_config::transport::RDMA;
  cfg.s3_rdma_max_inflight    = max_inflight;
  cfg.s3_rdma_arena_slot_size = k_slot_size;
  cfg.endpoint                = "mock-rdma-endpoint";
  cfg.region                  = "us-east-1";
  cfg.access_key              = "mock-access-key";
  cfg.secret_key              = "mock-secret-key";
  return cfg;
}

std::vector<std::uint8_t> pattern_bytes(std::size_t size, std::uint8_t salt = 41)
{
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + salt) & 0xffU);
  }
  return bytes;
}

std::shared_ptr<recording_rdma_client> seeded_client(std::string key,
                                                     std::vector<std::uint8_t> bytes)
{
  auto client = std::make_shared<recording_rdma_client>(std::make_shared<mock_rdma_client>());
  client->put_object(std::move(key), std::move(bytes));
  return client;
}

std::shared_ptr<s3_rdma_ioctx> make_started_ioctx(std::shared_ptr<recording_rdma_client> client,
                                                  cuda_delivery_ops ops    = {},
                                                  std::size_t max_inflight = k_max_inflight)
{
  auto ctx =
    std::make_shared<s3_rdma_ioctx>(make_config(max_inflight), std::move(client), std::move(ops));
  ctx->start();
  return ctx;
}

std::unique_ptr<sirius::io::sirius_datasource> open_ds(std::shared_ptr<s3_rdma_ioctx> const& ctx,
                                                       std::string const& key)
{
  return ctx->open_datasource("s3://" + std::string{k_bucket} + "/" + key);
}

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA F01 GPU test: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

std::string require_ready_error(std::future<std::size_t>& future,
                                std::chrono::milliseconds timeout = 5s)
{
  REQUIRE(future.wait_for(timeout) == std::future_status::ready);
  try {
    (void)future.get();
    FAIL("expected the RDMA read to fail");
  } catch (std::exception const& error) {
    return error.what();
  }
  return {};
}

std::size_t require_ready_value(std::future<std::size_t>& future,
                                std::chrono::milliseconds timeout = 5s)
{
  REQUIRE(future.wait_for(timeout) == std::future_status::ready);
  return future.get();
}

bool message_contains(std::string const& message, std::string_view needle)
{
  return message.find(needle) != std::string::npos;
}

std::vector<std::uint8_t> copy_device_to_host(void const* device_data,
                                              std::size_t size,
                                              rmm::cuda_stream_view stream)
{
  std::vector<std::uint8_t> bytes(size);
  auto const rc =
    cudaMemcpyAsync(bytes.data(), device_data, size, cudaMemcpyDeviceToHost, stream.value());
  if (rc != cudaSuccess) {
    throw std::runtime_error(std::string{"cudaMemcpyAsync D2H failed: "} + cudaGetErrorString(rc));
  }
  stream.synchronize();
  return bytes;
}

std::future<std::size_t> issue_device_read(sirius::io::sirius_datasource& ds,
                                           std::size_t size,
                                           rmm::device_buffer& device,
                                           rmm::cuda_stream_view stream)
{
  return ds.device_read_async(0, size, static_cast<std::uint8_t*>(device.data()), stream);
}

}  // namespace s3_rdma_f01_tests

TEST_CASE("s3_rdma creates the completion event before enqueueing D2D",
          "[s3][rdma][reactor][slot-lifetime][gpu]")
{
  using namespace s3_rdma_f01_tests;
  if (!cuda_device_available()) { return; }

  auto payload = pattern_bytes(k_slot_size);
  auto client  = seeded_client("event-create", payload);
  std::atomic<int> create_calls{0};
  std::atomic<int> memcpy_calls{0};
  std::atomic<int> destroy_calls{0};

  cuda_delivery_ops ops;
  ops.event_create = [&](cudaEvent_t* event, unsigned int flags) {
    if (create_calls.fetch_add(1) == 0) { return cudaErrorMemoryAllocation; }
    return cudaEventCreateWithFlags(event, flags);
  };
  ops.memcpy_async =
    [&](void* dst, void const* src, std::size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
      memcpy_calls.fetch_add(1);
      return cudaMemcpyAsync(dst, src, size, kind, stream);
    };
  ops.event_destroy = [&](cudaEvent_t event) {
    destroy_calls.fetch_add(1);
    return cudaEventDestroy(event);
  };

  auto ctx = make_started_ioctx(client, std::move(ops));
  auto ds  = open_ds(ctx, "event-create");
  rmm::cuda_stream stream;
  rmm::device_buffer first(payload.size(), stream);

  auto first_future = issue_device_read(*ds, payload.size(), first, stream);
  auto const error  = require_ready_error(first_future);
  CHECK(message_contains(error, "event"));
  CHECK(memcpy_calls.load() == 0);
  CHECK(destroy_calls.load() == 0);
  CHECK(ctx->perf_snapshot().delivery_fatal_total == 0);

  rmm::device_buffer follow_up(payload.size(), stream);
  auto follow_up_future = issue_device_read(*ds, payload.size(), follow_up, stream);
  REQUIRE(require_ready_value(follow_up_future) == payload.size());
  CHECK(memcpy_calls.load() == 1);
  CHECK(destroy_calls.load() == 1);
}

TEST_CASE("s3_rdma default delivery ops preserve byte-exact multi-chunk reads",
          "[s3][rdma][reactor][slot-lifetime][gpu]")
{
  using namespace s3_rdma_f01_tests;
  if (!cuda_device_available()) { return; }

  auto payload = pattern_bytes(3 * k_slot_size + 211, 37);
  auto client  = seeded_client("success-regression", payload);
  auto ctx     = make_started_ioctx(client, cuda_delivery_ops{});
  auto ds      = open_ds(ctx, "success-regression");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto future = issue_device_read(*ds, payload.size(), device, stream);
  REQUIRE(require_ready_value(future) == payload.size());
  auto const got = copy_device_to_host(device.data(), payload.size(), stream);
  REQUIRE(got.size() == payload.size());
  CHECK(std::equal(got.begin(), got.end(), payload.begin(), payload.end()));
}

TEST_CASE("s3_rdma ioctx snapshot aggregates all delivery safety counters",
          "[s3][rdma][reactor][slot-lifetime][metrics][gpu]")
{
  using namespace s3_rdma_f01_tests;
  if (!cuda_device_available()) { return; }

  auto payload = pattern_bytes(k_slot_size, 43);
  auto client  = seeded_client("safety-counters", payload);
  std::atomic<int> destroy_calls{0};

  cuda_delivery_ops ops;
  ops.event_destroy = [&](cudaEvent_t event) {
    destroy_calls.fetch_add(1);
    auto const real_result = cudaEventDestroy(event);
    return real_result == cudaSuccess ? cudaErrorUnknown : real_result;
  };

  auto ctx = make_started_ioctx(client, std::move(ops), 1);
  auto ds  = open_ds(ctx, "safety-counters");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto future = issue_device_read(*ds, payload.size(), device, stream);
  REQUIRE(require_ready_value(future) == payload.size());

  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.bytes_total == payload.size());
  CHECK(snapshot.requests_total == 1);
  CHECK(snapshot.error_total == 0);
  CHECK(snapshot.delivery_fatal_total == 0);
  CHECK(snapshot.arena_leak_total == 0);
  CHECK(destroy_calls.load() == 1);
}

TEST_CASE("s3_rdma event RAII destroys exactly created events without overriding results",
          "[s3][rdma][reactor][slot-lifetime][gpu]")
{
  using namespace s3_rdma_f01_tests;
  if (!cuda_device_available()) { return; }

  auto payload        = pattern_bytes(k_slot_size, 47);
  auto success_client = seeded_client("destroy-failure", payload);
  std::atomic<int> destroy_calls{0};

  cuda_delivery_ops destroy_ops;
  destroy_ops.event_destroy = [&](cudaEvent_t event) {
    destroy_calls.fetch_add(1);
    auto const real_result = cudaEventDestroy(event);
    return real_result == cudaSuccess ? cudaErrorUnknown : real_result;
  };
  auto success_ctx = make_started_ioctx(success_client, std::move(destroy_ops), 1);
  auto success_ds  = open_ds(success_ctx, "destroy-failure");
  rmm::cuda_stream success_stream;
  rmm::device_buffer success_buffer(payload.size(), success_stream);
  auto success_future =
    issue_device_read(*success_ds, payload.size(), success_buffer, success_stream);
  REQUIRE(require_ready_value(success_future) == payload.size());
  CHECK(destroy_calls.load() == 1);
  CHECK(success_ctx->perf_snapshot().delivery_fatal_total == 0);

  auto create_client = seeded_client("create-without-destroy", payload);
  std::atomic<int> create_destroy_calls{0};
  std::atomic<int> memcpy_calls{0};
  cuda_delivery_ops create_ops;
  create_ops.event_create  = [](cudaEvent_t*, unsigned int) { return cudaErrorMemoryAllocation; };
  create_ops.event_destroy = [&](cudaEvent_t event) {
    create_destroy_calls.fetch_add(1);
    return cudaEventDestroy(event);
  };
  create_ops.memcpy_async =
    [&](void* dst, void const* src, std::size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
      memcpy_calls.fetch_add(1);
      return cudaMemcpyAsync(dst, src, size, kind, stream);
    };
  auto create_ctx = make_started_ioctx(create_client, std::move(create_ops), 1);
  auto create_ds  = open_ds(create_ctx, "create-without-destroy");
  rmm::cuda_stream create_stream;
  rmm::device_buffer create_buffer(payload.size(), create_stream);
  auto create_future = issue_device_read(*create_ds, payload.size(), create_buffer, create_stream);
  CHECK_FALSE(require_ready_error(create_future).empty());
  CHECK(create_destroy_calls.load() == 0);
  CHECK(memcpy_calls.load() == 0);
}
