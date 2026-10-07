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
#include "io/cache/prefetching_cache.hpp"
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

  /// @p staging_block_size mirrors what a real reactor takes from its host
  /// resource; 0 means "no staging", which opts the reactor out of the
  /// chunk-size check in ioctx::initialize_cache.
  explicit controlled_reactor(std::size_t staging_block_size = 0)
    : _staging_block_size(staging_block_size)
  {
  }

  [[nodiscard]] controlled_config const& get_config() const noexcept { return _config; }

  [[nodiscard]] std::size_t staging_block_size() const noexcept { return _staging_block_size; }

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
  std::size_t _staging_block_size{0};
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
  /// @p eviction_threshold_fraction sizes the cache's hard cap on resident
  /// chunk bytes as a fraction of the host tier.
  explicit cache_fixture(double eviction_threshold_fraction = 0.8)
    : memory(initialize_memory_manager(1)), context(std::make_shared<controlled_context>(1, [] {
        return std::make_unique<controlled_reactor>();
      }))
  {
    sirius::io::cache::config config;
    config.mode                        = sirius::io::cache::cache_mode::sirius;
    config.eviction_threshold_fraction = eviction_threshold_fraction;
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
  rmm::device_buffer destination{read_size, stream};
  auto read = fixture.datasource->device_read_async(
    0, read_size, static_cast<std::uint8_t*>(destination.data()), stream);

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
    chunk_size, read_size, &first_destination, ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
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
      chunk_size,
      read_size,
      &second_destination,
      ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
  });
  auto const returned_without_loader = second_call.wait_for(100ms) == std::future_status::ready;
  CHECK(returned_without_loader);

  // If the assertion failed, settle the first load so cleanup cannot hang.
  if (!returned_without_loader) { complete_success(*cache_load); }
  REQUIRE(second_call.wait_for(2s) == std::future_status::ready);
  auto second = second_call.get();

  auto bounced = fixture.context->reactor().take_next();
  REQUIRE(bounced != nullptr);
  CHECK(bounced->front().needs_staging());
  CHECK_FALSE(bounced->front().is_fragmented());
  CHECK(bounced->front().has_device_request());

  complete_success(*bounced);
  if (cache_load->coordinator->tasks_remaining() != 0) { complete_success(*cache_load); }

  REQUIRE(first.wait_for(2s) == std::future_status::ready);
  REQUIRE(second.wait_for(2s) == std::future_status::ready);
  CHECK(first.get() == read_size);
  CHECK(second.get() == read_size);
}

TEST_CASE("a cache whose chunk size differs from the reactor staging block is refused", "[cache]")
{
  // The reactors plan a fragmented fill as fill_span(fill, chunk->offset,
  // staging block size).  If that size is not the cache's chunk size the extent
  // is wrong -- a larger staging block writes past the end of the pinned chunk.
  auto memory = initialize_memory_manager(1);

  sirius::io::cache::config config;
  config.mode = sirius::io::cache::cache_mode::sirius;
  config.apply_mode();

  // A reactor that opts out of staging (0) reports what the pool actually chose.
  auto probe =
    std::make_shared<controlled_context>(1, [] { return std::make_unique<controlled_reactor>(); });
  probe->initialize_cache(*memory, config, single_gpu_topology());
  REQUIRE(probe->cache() != nullptr);
  auto const chunk_size = probe->cache()->chunk_size();
  REQUIRE(chunk_size > 1);
  probe->shutdown_cache();

  auto mismatched = std::make_shared<controlled_context>(
    1, [chunk_size] { return std::make_unique<controlled_reactor>(chunk_size / 2); });
  mismatched->initialize_cache(*memory, config, single_gpu_topology());
  CHECK(mismatched->cache() == nullptr);

  auto matched = std::make_shared<controlled_context>(
    1, [chunk_size] { return std::make_unique<controlled_reactor>(chunk_size); });
  matched->initialize_cache(*memory, config, single_gpu_topology());
  CHECK(matched->cache() != nullptr);
  matched->shutdown_cache();
}

// ---------------------------------------------------------------------------
// The budget is a hard cap on cache-resident bytes
// ---------------------------------------------------------------------------
//
// eviction_threshold_fraction of the host tier bounds what the pool may hand
// out.  An allocation that would cross it is refused like an exhausted pool;
// the read over such a chunk still succeeds through reactor-owned staging but
// leaves nothing cached.  0.0007 of the test host tier (3-4 GiB) is two 1 MiB
// chunks, so a 4-chunk object can over-run the cap on its own.

namespace {

constexpr double two_chunk_cap = 0.0007;

void fadvise_chunks(sirius::io::sirius_datasource& ds, std::size_t first, std::size_t n)
{
  std::array<cudf::io::text::byte_range_info, 1> ranges{cudf::io::text::byte_range_info{
    static_cast<std::int64_t>(first * chunk_size), static_cast<std::int64_t>(n * chunk_size)}};
  ds.fadvise(ranges, 0);
}

}  // namespace

TEST_CASE("the cache cap is two chunks under the test fraction", "[cache][cache_cap]")
{
  cache_fixture fixture(two_chunk_cap);
  auto* cache = fixture.context->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->chunk_size() == chunk_size);
  CHECK(cache->max_prefetching_budget_bytes() == 2 * chunk_size);
}

TEST_CASE("a demand read under the cache cap populates a cache chunk", "[cache][cache_cap]")
{
  cache_fixture fixture(two_chunk_cap);
  auto* cache = fixture.context->cache();
  REQUIRE(cache->max_prefetching_budget_bytes() == 2 * chunk_size);
  fixture.advise(chunk_size);
  CHECK(cache->claimed_bytes() == chunk_size);

  std::uint8_t destination{};
  auto read = fixture.datasource->device_read_async(
    chunk_size, read_size, &destination, ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
  auto load = fixture.context->reactor().take_next();
  REQUIRE(load != nullptr);
  // file -> cache chunk -> device: the slice fills the cache allocation.
  CHECK(load->front().h_buffer.is_fragmented());
  complete_success(*load);
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == read_size);

  INFO("cache: " << cache->summary());
  CHECK(cache->claimed_bytes() == chunk_size);
  CHECK(cache->summary().find("uncached_over_budget=0") != std::string::npos);
}

TEST_CASE("a demand read over the cache cap succeeds uncached", "[cache][cache_cap]")
{
  cache_fixture fixture(two_chunk_cap);
  auto* cache       = fixture.context->cache();
  auto const budget = cache->max_prefetching_budget_bytes();
  REQUIRE(budget == 2 * chunk_size);

  // A live request holds the whole cap.
  auto filler = fixture.context->open_datasource("controlled://cache-cap-filler");
  fadvise_chunks(*filler, 0, 2);
  REQUIRE(filler->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  REQUIRE(cache->claimed_bytes() == budget);

  // The demand-side preparation is refused rather than grown past the cap.
  std::array<cudf::io::text::byte_range_info, 1> ranges{cudf::io::text::byte_range_info{
    static_cast<std::int64_t>(chunk_size), static_cast<std::int64_t>(read_size)}};
  fixture.datasource->fadvise(ranges, 0);
  CHECK(fixture.datasource->prepare_prefetch(false) ==
        sirius::io::prepare_result::allocation_failed);
  CHECK(cache->claimed_bytes() <= budget);

  std::uint8_t destination{};
  auto read = fixture.datasource->device_read_async(
    chunk_size, read_size, &destination, ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
  auto bounced = fixture.context->reactor().take_next();
  REQUIRE(bounced != nullptr);
  CHECK(bounced->front().needs_staging());
  CHECK_FALSE(bounced->front().is_fragmented());
  CHECK(bounced->front().has_device_request());
  complete_success(*bounced);
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == read_size);

  INFO("cache: " << cache->summary());
  CHECK(cache->claimed_bytes() <= budget);
  CHECK(cache->summary().find("uncached_over_budget=1") != std::string::npos);
  CHECK(cache->summary().find("cap_refusals=0") == std::string::npos);
}

TEST_CASE("a prefetch allocation over the cache cap recovers after disposal",
          "[cache][cache_cap][prepare]")
{
  cache_fixture fixture(two_chunk_cap);
  auto* cache       = fixture.context->cache();
  auto const budget = cache->max_prefetching_budget_bytes();
  REQUIRE(budget == 2 * chunk_size);

  {
    auto filler = fixture.context->open_datasource("controlled://cache-cap-filler");
    fadvise_chunks(*filler, 0, 2);
    REQUIRE(filler->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  }  // disposed, but LRU keeps its chunks resident: the pool is still at the cap

  REQUIRE(cache->claimed_bytes() == budget);
  fadvise_chunks(*fixture.datasource, 2, 2);
  CHECK(fixture.datasource->prepare_prefetch(false) ==
        sirius::io::prepare_result::allocation_failed);
  CHECK(cache->claimed_bytes() <= budget);

  // The blocking retry waits for the eviction the refusal requested; the
  // disposed request's chunks make room, and the cap holds afterwards.
  CHECK(fixture.datasource->prepare_prefetch(true) == sirius::io::prepare_result::prepared);
  CHECK(cache->claimed_bytes() <= budget);
}

TEST_CASE("an empty cache admits one request larger than its cap", "[cache][cache_cap][prepare]")
{
  cache_fixture fixture(two_chunk_cap);
  auto* cache = fixture.context->cache();
  REQUIRE(cache->max_prefetching_budget_bytes() == 2 * chunk_size);
  REQUIRE(cache->claimed_bytes() == 0);

  // Four chunks against a two-chunk cap: refused, it would never run at all.
  fadvise_chunks(*fixture.datasource, 0, 4);
  CHECK(fixture.datasource->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  CHECK(cache->claimed_bytes() == 4 * chunk_size);

  // ...but it runs alone: nothing else is admitted while it is resident.
  auto other = fixture.context->open_datasource("controlled://cache-cap-other");
  fadvise_chunks(*other, 0, 1);
  CHECK(other->prepare_prefetch(false) == sirius::io::prepare_result::allocation_failed);
  CHECK(cache->claimed_bytes() == 4 * chunk_size);
}
