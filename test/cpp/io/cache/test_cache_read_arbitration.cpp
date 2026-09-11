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

// These tests are the cache/read arbitration contract.
//
// A cache handle names an entire split, but one cuDF read usually asks for only
// a few of its chunks. Waiting at the datasource therefore creates false
// dependencies: an unrelated read stalls behind prefetch I/O it will never use.
// The cache is the first layer that knows both the read range and each chunk
// state, so it alone can make the three-way choice exercised below:
//
//   overlapping prefetch load  -> wait, then reuse the cache
//   unrelated prefetch load    -> dispatch the read immediately
//   demand-owned loading chunk -> dispatch through reactor-owned bounce staging

#include "catch.hpp"
#include "io/cache/config.hpp"
#include "io/io_request.hpp"
#include "io/sirius_datasource.hpp"
#include "io/templated_ioctx.hpp"
#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace {

constexpr std::size_t chunk_size = 1U << 20;
constexpr std::size_t read_size  = 4096;

struct controlled_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{1};
};

class controlled_object final : public sirius::io::io_object {
 public:
  explicit controlled_object(std::string path) : _path(std::move(path)) {}

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return 4 * chunk_size; }

 private:
  std::string _path;
};

class controlled_reactor {
 public:
  using io_object_type                  = controlled_object;
  using reactor_config_type             = controlled_config;
  static constexpr bool prefers_bulk_io = false;

  [[nodiscard]] controlled_config const& get_config() const noexcept { return _config; }

  void enqueue(std::unique_ptr<sirius::io::grouped_io_request> request) noexcept
  {
    {
      std::lock_guard lock(_mutex);
      _requests.push_back(std::move(request));
    }
    _ready.notify_one();
  }

  [[nodiscard]] std::unique_ptr<sirius::io::grouped_io_request> take_next(
    std::chrono::milliseconds timeout = 2s)
  {
    std::unique_lock lock(_mutex);
    if (!_ready.wait_for(lock, timeout, [this] { return !_requests.empty(); })) { return nullptr; }
    auto request = std::move(_requests.front());
    _requests.pop_front();
    return request;
  }

  [[nodiscard]] std::size_t pending() const
  {
    std::lock_guard lock(_mutex);
    return _requests.size();
  }

  [[nodiscard]] std::size_t queued_bytes() const noexcept { return 0; }

  std::size_t host_read(controlled_object const&,
                        std::size_t,
                        std::size_t size,
                        std::uint8_t*) const
  {
    return size;
  }

  void start() {}
  void shutdown() {}
  void interrupt() {}

  [[nodiscard]] static std::unique_ptr<controlled_object> create_io_object(std::string path)
  {
    return std::make_unique<controlled_object>(std::move(path));
  }

  [[nodiscard]] static bool supports(std::string_view) { return true; }

  [[nodiscard]] static std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<cudf::io::text::byte_range_info const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

 private:
  controlled_config _config;
  mutable std::mutex _mutex;
  std::condition_variable _ready;
  std::deque<std::unique_ptr<sirius::io::grouped_io_request>> _requests;
};

class controlled_context final : public sirius::io::templated_ioctx<controlled_reactor> {
 public:
  using templated_ioctx::templated_ioctx;

  [[nodiscard]] sirius::io::io_context_type type() const noexcept override
  {
    return sirius::io::io_context_type::uring;
  }

  [[nodiscard]] controlled_reactor& reactor() noexcept { return *_reactors.front(); }
};

std::shared_ptr<const sirius::memory::topology_index> single_gpu_topology()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

struct cache_fixture {
  cache_fixture()
    : memory(initialize_memory_manager(1)), context(std::make_shared<controlled_context>(1, [] {
        return std::make_unique<controlled_reactor>();
      }))
  {
    sirius::io::cache::config config;
    config.mode = sirius::io::cache::cache_mode::sirius;
    config.apply_mode();
    context->initialize_cache(*memory, config, single_gpu_topology());
    datasource = context->open_datasource("controlled://cache-read-arbitration");
  }

  ~cache_fixture()
  {
    datasource.reset();
    context->shutdown_cache();
  }

  void advise(std::size_t offset)
  {
    std::array<cudf::io::text::byte_range_info, 1> ranges{cudf::io::text::byte_range_info{
      static_cast<std::int64_t>(offset), static_cast<std::int64_t>(read_size)}};
    datasource->fadvise(ranges, 0);
    REQUIRE(datasource->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  }

  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory;
  std::shared_ptr<controlled_context> context;
  std::unique_ptr<sirius::io::sirius_datasource> datasource;
};

void complete_success(sirius::io::grouped_io_request& request, std::uint8_t value = 0x5a)
{
  while (!request.empty()) {
    auto slice = request.take_front();
    if (slice.h_buffer.is_contiguous()) {
      auto* destination = std::get<std::uint8_t*>(slice.h_buffer.buffer);
      std::fill_n(destination, slice.size(), value);
    } else if (slice.h_buffer.is_fragmented()) {
      for (auto* chunk : slice.h_buffer.fragments()) {
        auto const within_chunk = slice.offset() - chunk->offset;
        std::fill_n(chunk->data + within_chunk, slice.size(), value);
      }
    }
    if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), true); }
    request.coordinator->on_complete();
  }
}

}  // namespace

TEST_CASE("an overlapping cache read waits for prefetch and reuses its chunk",
          "[cache][cache_handle][read-arbitration]")
{
  cache_fixture fixture;
  fixture.advise(0);

  std::atomic<bool> prefetched{false};
  REQUIRE(fixture.datasource->prefetch_async([&prefetched](bool ok) noexcept {
    prefetched.store(ok, std::memory_order_release);
  }) == sirius::io::prefetch_refusal::issued);
  auto prefetch = fixture.context->reactor().take_next();
  REQUIRE(prefetch != nullptr);

  std::array<std::uint8_t, read_size> destination{};
  auto read_call = std::async(std::launch::async, [&] {
    return fixture.datasource->host_read_async(0, destination.size(), destination.data());
  });

  // The only backend request is the prefetch held above. The read cannot return
  // a future yet because doing so would require issuing duplicate I/O.
  CHECK(read_call.wait_for(50ms) == std::future_status::timeout);
  CHECK(fixture.context->reactor().pending() == 0);

  complete_success(*prefetch);
  REQUIRE(read_call.wait_for(2s) == std::future_status::ready);
  auto read = read_call.get();

  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == destination.size());
  CHECK(std::ranges::all_of(destination, [](auto byte) { return byte == 0x5a; }));
  CHECK(prefetched.load(std::memory_order_acquire));
  CHECK(fixture.context->reactor().pending() == 0);
}

TEST_CASE("a cached device read retires its pin through the stream completion poll",
          "[cache][cache_handle][completion-poll][gpu_execution]")
{
  cache_fixture fixture;
  fixture.advise(0);

  std::atomic<bool> prefetched{false};
  REQUIRE(fixture.datasource->prefetch_async([&prefetched](bool ok) noexcept {
    prefetched.store(ok, std::memory_order_release);
  }) == sirius::io::prefetch_refusal::issued);
  auto prefetch = fixture.context->reactor().take_next();
  REQUIRE(prefetch != nullptr);
  complete_success(*prefetch, 0x6c);
  REQUIRE(prefetched.load(std::memory_order_acquire));

  // This read is entirely cache-resident, so the controlled reactor must see
  // no request. Its future can become ready only when the production
  // completion poll observes the stream ticket, releases the chunk pin, and
  // settles the cached-copy coordinator credit.
  rmm::cuda_stream stream;
  rmm::device_buffer destination{read_size, stream.view()};
  auto read = fixture.datasource->device_read_async(
    0, read_size, static_cast<std::uint8_t*>(destination.data()), stream.view());

  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == read_size);
  CHECK(fixture.context->reactor().pending() == 0);

  std::array<std::uint8_t, read_size> host{};
  REQUIRE(cudaMemcpyAsync(
            host.data(), destination.data(), host.size(), cudaMemcpyDeviceToHost, stream.value()) ==
          cudaSuccess);
  stream.synchronize();
  CHECK(std::ranges::all_of(host, [](auto byte) { return byte == 0x6c; }));
}

TEST_CASE("a cache read does not wait for unrelated chunks in the same handle",
          "[cache][cache_handle][read-arbitration]")
{
  cache_fixture fixture;
  fixture.advise(0);

  REQUIRE(fixture.datasource->prefetch_async([](bool) noexcept {}) ==
          sirius::io::prefetch_refusal::issued);
  auto prefetch = fixture.context->reactor().take_next();
  REQUIRE(prefetch != nullptr);

  std::array<std::uint8_t, read_size> destination{};
  auto read_call = std::async(std::launch::async, [&] {
    return fixture.datasource->host_read_async(
      2 * chunk_size, destination.size(), destination.data());
  });

  auto const returned_without_prefetch = read_call.wait_for(100ms) == std::future_status::ready;
  CHECK(returned_without_prefetch);

  // Always release the held prefetch before a REQUIRE can leave this scope.
  if (!returned_without_prefetch) { complete_success(*prefetch); }
  REQUIRE(read_call.wait_for(2s) == std::future_status::ready);
  auto read = read_call.get();

  auto demand = fixture.context->reactor().take_next();
  REQUIRE(demand != nullptr);
  CHECK(demand->front().h_buffer.is_contiguous());
  complete_success(*demand, 0x2b);
  if (prefetch->coordinator->tasks_remaining() != 0) { complete_success(*prefetch); }

  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == destination.size());
  CHECK(std::ranges::all_of(destination, [](auto byte) { return byte == 0x2b; }));
}

TEST_CASE("a demand-owned loading chunk uses reactor bounce staging",
          "[cache][cache_handle][read-arbitration]")
{
  cache_fixture fixture;
  fixture.advise(chunk_size);

  // The first executor read, not the prefetcher, wins the chunk's loading CAS.
  // Its prepared slice writes into the cache allocation.
  std::uint8_t first_destination{};
  auto first = fixture.datasource->device_read_async(
    chunk_size, read_size, &first_destination, rmm::cuda_stream_default);
  auto cache_load = fixture.context->reactor().take_next();
  REQUIRE(cache_load != nullptr);
  CHECK(cache_load->front().h_buffer.is_fragmented());

  // The same handle sees the chunk in loading state, but its producer is only
  // prepared, not prefetching. Waiting would serialize executor reads and can
  // deadlock behind queue pressure; a staged device-only slice gives the
  // reactor ownership of a bounce slot instead.
  std::uint8_t second_destination{};
  auto second_call                   = std::async(std::launch::async, [&] {
    return fixture.datasource->device_read_async(
      chunk_size, read_size, &second_destination, rmm::cuda_stream_default);
  });
  auto const returned_without_loader = second_call.wait_for(100ms) == std::future_status::ready;
  CHECK(returned_without_loader);

  // If the assertion failed, settle the first load so cleanup cannot hang.
  if (!returned_without_loader) { complete_success(*cache_load); }
  REQUIRE(second_call.wait_for(2s) == std::future_status::ready);
  auto second = second_call.get();

  auto bounced = fixture.context->reactor().take_next();
  REQUIRE(bounced != nullptr);
  CHECK(bounced->front().is_staged());
  CHECK_FALSE(bounced->front().is_fragmented());
  CHECK(bounced->front().has_device_request());

  complete_success(*bounced);
  if (cache_load->coordinator->tasks_remaining() != 0) { complete_success(*cache_load); }

  REQUIRE(first.wait_for(2s) == std::future_status::ready);
  REQUIRE(second.wait_for(2s) == std::future_status::ready);
  CHECK(first.get() == read_size);
  CHECK(second.get() == read_size);
}
