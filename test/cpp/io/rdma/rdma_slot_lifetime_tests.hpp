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
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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

class manual_gate {
 public:
  void arrive_and_wait()
  {
    std::unique_lock lk{_mtx};
    ++_arrivals;
    _cv.notify_all();
    _cv.wait(lk, [&] { return _open; });
  }

  [[nodiscard]] bool wait_for_arrivals(std::size_t count, std::chrono::milliseconds timeout = 5s)
  {
    std::unique_lock lk{_mtx};
    return _cv.wait_for(lk, timeout, [&] { return _arrivals >= count; });
  }

  void open()
  {
    {
      std::lock_guard lk{_mtx};
      _open = true;
    }
    _cv.notify_all();
  }

 private:
  std::mutex _mtx;
  std::condition_variable _cv;
  std::size_t _arrivals{0};
  bool _open{false};
};

class open_gate_on_exit {
 public:
  explicit open_gate_on_exit(std::shared_ptr<manual_gate> gate) : _gate(std::move(gate)) {}
  ~open_gate_on_exit() { _gate->open(); }

  open_gate_on_exit(open_gate_on_exit const&)            = delete;
  open_gate_on_exit& operator=(open_gate_on_exit const&) = delete;

 private:
  std::shared_ptr<manual_gate> _gate;
};

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

  void gate_get_call(std::size_t one_based_call, std::shared_ptr<manual_gate> gate)
  {
    std::lock_guard lk{_mtx};
    _gated_get_call = one_based_call;
    _get_gate       = std::move(gate);
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
    std::shared_ptr<manual_gate> gate;
    {
      std::lock_guard lk{_mtx};
      _get_destinations.push_back(dst);
      if (_gated_get_call && _get_destinations.size() == *_gated_get_call) { gate = _get_gate; }
    }
    if (gate) { gate->arrive_and_wait(); }
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
  std::optional<std::size_t> _gated_get_call;
  std::shared_ptr<manual_gate> _get_gate;
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

bool wait_until(std::function<bool()> predicate, std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) { return true; }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
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

TEST_CASE("s3_rdma retains the slot and future until record failure is quiesced",
          "[s3][rdma][reactor][slot-lifetime][gpu]")
{
  using namespace s3_rdma_f01_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size, 7);
  auto client    = seeded_client("record-failure", payload);
  auto sync_gate = std::make_shared<manual_gate>();
  std::atomic<int> record_calls{0};

  cuda_delivery_ops ops;
  ops.event_record = [&](cudaEvent_t event, cudaStream_t stream) {
    auto const real_result = cudaEventRecord(event, stream);
    if (real_result != cudaSuccess) { return real_result; }
    return record_calls.fetch_add(1) == 0 ? cudaErrorUnknown : cudaSuccess;
  };
  ops.stream_synchronize = [sync_gate](cudaStream_t stream) {
    sync_gate->arrive_and_wait();
    return cudaStreamSynchronize(stream);
  };

  auto ctx = make_started_ioctx(client, std::move(ops));
  auto ds  = open_ds(ctx, "record-failure");
  rmm::cuda_stream first_stream;
  rmm::cuda_stream second_stream;
  rmm::device_buffer first(payload.size(), first_stream);
  rmm::device_buffer second(payload.size(), second_stream);
  open_gate_on_exit release_gate{sync_gate};

  auto first_future = issue_device_read(*ds, payload.size(), first, first_stream);
  REQUIRE(sync_gate->wait_for_arrivals(1));
  REQUIRE(wait_until([&] { return client->get_count() >= 1; }));
  auto* const held_slot = client->get_destination(0);

  auto second_future = issue_device_read(*ds, payload.size(), second, second_stream);
  REQUIRE(wait_until([&] { return client->get_count() >= 2; }));
  CHECK(client->get_destination(1) != held_slot);
  CHECK(first_future.wait_for(50ms) != std::future_status::ready);
  REQUIRE(require_ready_value(second_future) == payload.size());

  sync_gate->open();
  CHECK(message_contains(require_ready_error(first_future), "event"));
  CHECK(ctx->perf_snapshot().fallback_stream_sync_total >= 1);

  rmm::device_buffer follow_up(payload.size(), first_stream);
  auto follow_up_future = issue_device_read(*ds, payload.size(), follow_up, first_stream);
  REQUIRE(require_ready_value(follow_up_future) == payload.size());
}

TEST_CASE("s3_rdma stream sync proves quiescence after event sync failure",
          "[s3][rdma][reactor][slot-lifetime][gpu]")
{
  using namespace s3_rdma_f01_tests;
  if (!cuda_device_available()) { return; }

  auto payload = pattern_bytes(k_slot_size, 11);
  auto client  = seeded_client("event-sync", payload);
  std::atomic<int> event_sync_calls{0};
  std::atomic<int> stream_sync_calls{0};

  cuda_delivery_ops ops;
  ops.event_synchronize = [&](cudaEvent_t event) {
    auto const real_result = cudaEventSynchronize(event);
    if (real_result != cudaSuccess) { return real_result; }
    return event_sync_calls.fetch_add(1) == 0 ? cudaErrorUnknown : cudaSuccess;
  };
  ops.stream_synchronize = [&](cudaStream_t stream) {
    stream_sync_calls.fetch_add(1);
    return cudaStreamSynchronize(stream);
  };

  auto ctx = make_started_ioctx(client, std::move(ops), 1);
  auto ds  = open_ds(ctx, "event-sync");
  rmm::cuda_stream stream;
  rmm::device_buffer first(payload.size(), stream);

  auto first_future = issue_device_read(*ds, payload.size(), first, stream);
  CHECK(message_contains(require_ready_error(first_future), "event"));
  REQUIRE(client->get_count() == 1);
  auto* const first_slot = client->get_destination(0);
  CHECK(stream_sync_calls.load() >= 1);
  CHECK(ctx->perf_snapshot().delivery_fatal_total == 0);

  rmm::device_buffer follow_up(payload.size(), stream);
  auto follow_up_future = issue_device_read(*ds, payload.size(), follow_up, stream);
  REQUIRE(require_ready_value(follow_up_future) == payload.size());
  REQUIRE(client->get_count() == 2);
  CHECK(client->get_destination(1) == first_slot);
}

TEST_CASE("s3_rdma context-fatal delivery failure fail-stops and drains queued chunks",
          "[s3][rdma][reactor][slot-lifetime][gpu]")
{
  using namespace s3_rdma_f01_tests;
  if (!cuda_device_available()) { return; }

  auto payload = pattern_bytes(4 * k_slot_size + 17, 13);
  auto client  = seeded_client("fatal-drain", payload);
  std::atomic<int> record_calls{0};

  cuda_delivery_ops ops;
  ops.event_record = [&](cudaEvent_t event, cudaStream_t stream) {
    auto const real_result = cudaEventRecord(event, stream);
    if (real_result != cudaSuccess) { return real_result; }
    return record_calls.fetch_add(1) == 0 ? cudaErrorUnknown : cudaSuccess;
  };
  ops.stream_synchronize = [](cudaStream_t stream) {
    auto const real_result = cudaStreamSynchronize(stream);
    return real_result == cudaSuccess ? cudaErrorIllegalAddress : real_result;
  };

  auto ctx = make_started_ioctx(client, std::move(ops), 1);
  auto ds  = open_ds(ctx, "fatal-drain");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);

  auto future               = issue_device_read(*ds, payload.size(), device, stream);
  auto const original_error = require_ready_error(future);
  CHECK_FALSE(original_error.empty());
  CHECK(client->get_count() == 1);
  CHECK(ctx->perf_snapshot().delivery_fatal_total == 1);

  rmm::device_buffer subsequent(k_slot_size, stream);
  auto subsequent_future      = issue_device_read(*ds, k_slot_size, subsequent, stream);
  auto const subsequent_error = require_ready_error(subsequent_future);
  CHECK_FALSE(subsequent_error.empty());
  CHECK((message_contains(subsequent_error, "fatal") ||
         message_contains(subsequent_error, "event") ||
         message_contains(subsequent_error, "illegal")));
  CHECK(client->get_count() == 1);
  CHECK(ctx->perf_snapshot().delivery_fatal_total == 1);
}

TEST_CASE("s3_rdma double-active fatal transition resolves chunks through their owners",
          "[s3][rdma][reactor][slot-lifetime][gpu]")
{
  using namespace s3_rdma_f01_tests;
  if (!cuda_device_available()) { return; }

  auto client = seeded_client("fatal-owner", pattern_bytes(k_slot_size, 17));
  client->put_object("active-owner", pattern_bytes(k_slot_size, 19));
  client->put_object("queued-left", pattern_bytes(k_slot_size, 23));
  client->put_object("queued-right", pattern_bytes(k_slot_size, 29));
  auto fatal_gate = std::make_shared<manual_gate>();
  auto get_gate   = std::make_shared<manual_gate>();
  client->gate_get_call(2, get_gate);
  std::atomic<int> record_calls{0};
  std::atomic<int> stream_sync_calls{0};
  std::atomic<int> memcpy_calls{0};

  cuda_delivery_ops ops;
  ops.event_record = [&](cudaEvent_t event, cudaStream_t stream) {
    auto const real_result = cudaEventRecord(event, stream);
    if (real_result != cudaSuccess) { return real_result; }
    if (record_calls.fetch_add(1) == 0) {
      fatal_gate->arrive_and_wait();
      return cudaErrorUnknown;
    }
    return cudaSuccess;
  };
  ops.stream_synchronize = [&](cudaStream_t stream) {
    auto const real_result = cudaStreamSynchronize(stream);
    if (real_result != cudaSuccess) { return real_result; }
    return stream_sync_calls.fetch_add(1) == 0 ? cudaErrorIllegalAddress : cudaSuccess;
  };
  ops.memcpy_async =
    [&](void* dst, void const* src, std::size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
      memcpy_calls.fetch_add(1);
      return cudaMemcpyAsync(dst, src, size, kind, stream);
    };

  auto ctx             = make_started_ioctx(client, std::move(ops), 2);
  auto fatal_ds        = open_ds(ctx, "fatal-owner");
  auto active_ds       = open_ds(ctx, "active-owner");
  auto queued_left_ds  = open_ds(ctx, "queued-left");
  auto queued_right_ds = open_ds(ctx, "queued-right");
  rmm::cuda_stream fatal_stream;
  rmm::cuda_stream active_stream;
  rmm::cuda_stream queued_left_stream;
  rmm::cuda_stream queued_right_stream;
  rmm::device_buffer fatal_buffer(k_slot_size, fatal_stream);
  rmm::device_buffer active_buffer(k_slot_size, active_stream);
  rmm::device_buffer queued_left_buffer(k_slot_size, queued_left_stream);
  rmm::device_buffer queued_right_buffer(k_slot_size, queued_right_stream);
  open_gate_on_exit release_fatal_gate{fatal_gate};
  open_gate_on_exit release_get_gate{get_gate};

  auto fatal_future = issue_device_read(*fatal_ds, k_slot_size, fatal_buffer, fatal_stream);
  REQUIRE(fatal_gate->wait_for_arrivals(1));
  auto active_future = issue_device_read(*active_ds, k_slot_size, active_buffer, active_stream);
  REQUIRE(get_gate->wait_for_arrivals(1));
  REQUIRE(client->get_count() == 2);
  auto queued_left_future =
    issue_device_read(*queued_left_ds, k_slot_size, queued_left_buffer, queued_left_stream);
  auto queued_right_future =
    issue_device_read(*queued_right_ds, k_slot_size, queued_right_buffer, queued_right_stream);

  fatal_gate->open();
  CHECK_FALSE(require_ready_error(fatal_future).empty());
  CHECK_FALSE(require_ready_error(queued_left_future).empty());
  CHECK_FALSE(require_ready_error(queued_right_future).empty());
  CHECK(active_future.wait_for(50ms) != std::future_status::ready);
  CHECK(client->get_count() == 2);
  CHECK(ctx->perf_snapshot().delivery_fatal_total == 1);

  get_gate->open();
  CHECK_FALSE(require_ready_error(active_future).empty());
  CHECK(client->get_count() == 2);
  CHECK(memcpy_calls.load() == 1);
  CHECK(ctx->perf_snapshot().delivery_fatal_total == 1);
}

TEST_CASE("s3_rdma shutdown leaks an arena when device quiescence cannot be established",
          "[s3][rdma][reactor][slot-lifetime][gpu]")
{
  using namespace s3_rdma_f01_tests;
  if (!cuda_device_available()) { return; }

  auto payload = pattern_bytes(k_slot_size, 31);
  auto client  = seeded_client("fatal-teardown", payload);
  std::atomic<int> record_calls{0};
  std::atomic<int> device_sync_calls{0};

  cuda_delivery_ops ops;
  ops.event_record = [&](cudaEvent_t event, cudaStream_t stream) {
    auto const real_result = cudaEventRecord(event, stream);
    if (real_result != cudaSuccess) { return real_result; }
    return record_calls.fetch_add(1) == 0 ? cudaErrorUnknown : cudaSuccess;
  };
  ops.stream_synchronize = [](cudaStream_t stream) {
    auto const real_result = cudaStreamSynchronize(stream);
    return real_result == cudaSuccess ? cudaErrorIllegalAddress : real_result;
  };
  ops.device_synchronize = [&] {
    device_sync_calls.fetch_add(1);
    auto const real_result = cudaDeviceSynchronize();
    return real_result == cudaSuccess ? cudaErrorIllegalAddress : real_result;
  };

  auto ctx = make_started_ioctx(client, std::move(ops), 1);
  auto ds  = open_ds(ctx, "fatal-teardown");
  rmm::cuda_stream stream;
  rmm::device_buffer device(payload.size(), stream);
  auto future = issue_device_read(*ds, payload.size(), device, stream);
  CHECK_FALSE(require_ready_error(future).empty());
  REQUIRE(client->register_count() == 1);

  auto const start = std::chrono::steady_clock::now();
  ctx->shutdown();
  auto const elapsed = std::chrono::steady_clock::now() - start;
  CHECK(elapsed < 5s);
  CHECK(device_sync_calls.load() >= 1);
  CHECK(ctx->perf_snapshot().delivery_fatal_total == 1);
  CHECK(ctx->perf_snapshot().arena_leak_total == 1);
  CHECK(client->deregister_count() == 0);

  ds.reset();
  ctx.reset();
  CHECK(client->deregister_count() == 0);
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
  std::atomic<int> record_calls{0};
  std::atomic<int> event_sync_calls{0};
  std::atomic<int> stream_sync_calls{0};

  cuda_delivery_ops ops;
  ops.event_record = [&](cudaEvent_t event, cudaStream_t stream) {
    auto const real_result = cudaEventRecord(event, stream);
    if (real_result != cudaSuccess) { return real_result; }
    return record_calls.fetch_add(1) == 1 ? cudaErrorUnknown : cudaSuccess;
  };
  ops.event_synchronize = [&](cudaEvent_t event) {
    auto const real_result = cudaEventSynchronize(event);
    if (real_result != cudaSuccess) { return real_result; }
    return event_sync_calls.fetch_add(1) == 0 ? cudaErrorUnknown : cudaSuccess;
  };
  ops.stream_synchronize = [&](cudaStream_t stream) {
    auto const real_result = cudaStreamSynchronize(stream);
    if (real_result != cudaSuccess) { return real_result; }
    return stream_sync_calls.fetch_add(1) == 0 ? cudaSuccess : cudaErrorIllegalAddress;
  };
  ops.device_synchronize = [] {
    auto const real_result = cudaDeviceSynchronize();
    return real_result == cudaSuccess ? cudaErrorIllegalAddress : real_result;
  };

  auto ctx = make_started_ioctx(client, std::move(ops), 1);
  auto ds  = open_ds(ctx, "safety-counters");
  rmm::cuda_stream stream;
  rmm::device_buffer first(payload.size(), stream);
  rmm::device_buffer second(payload.size(), stream);

  auto recoverable = issue_device_read(*ds, payload.size(), first, stream);
  CHECK_FALSE(require_ready_error(recoverable).empty());
  auto fatal = issue_device_read(*ds, payload.size(), second, stream);
  CHECK_FALSE(require_ready_error(fatal).empty());
  ctx->shutdown();

  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fallback_stream_sync_total >= 2);
  CHECK(snapshot.delivery_fatal_total == 1);
  CHECK(snapshot.arena_leak_total == 1);
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
