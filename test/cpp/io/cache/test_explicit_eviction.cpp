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

// Coverage of prefetching_cache::evict(bytes) and claimed_bytes(), driven
// through a real uring-backed sirius_datasource so the chunk states, the
// prepare path and the evictor are the engine's own.
//
// The whole point of the explicit demand is that it works when the cache's own
// trigger does not, so every case here runs under the `lru` policy with a pool
// nowhere near its threshold: without the demand the evictor has no reason to
// reclaim anything, which is what makes "it reclaimed" a real signal.

#include "catch.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"
#include "io/sirius_datasource.hpp"
#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index_for_evict()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

/// A real on-disk file, so the uring backend can open it and the cache sizes
/// its slot table from a genuine object size.
struct temp_data_file {
  std::filesystem::path path;

  explicit temp_data_file(std::size_t bytes)
  {
    path = std::filesystem::temp_directory_path() /
           ("sirius_explicit_evict_" + std::to_string(::getpid()) + "_" +
            std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::ofstream out(path, std::ios::binary);
    std::vector<char> data(bytes, 'z');
    out.write(data.data(), static_cast<std::streamsize>(bytes));
  }

  ~temp_data_file()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

/// Poll rather than sleep once: the evictor is a background thread, so a sweep
/// that is merely slow must not read as one that never happened.
bool claimed_drops_below(sirius::io::cache::prefetching_cache& cache,
                         std::size_t target,
                         std::chrono::milliseconds budget)
{
  auto const deadline = std::chrono::steady_clock::now() + budget;
  while (std::chrono::steady_clock::now() < deadline) {
    if (cache.claimed_bytes() < target) { return true; }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return cache.claimed_bytes() < target;
}

/// `lru`, so the evictor sweeps only under its own pressure rule -- which a
/// handful of megabytes will never trip.  That is the point: anything reclaimed
/// in these tests was reclaimed because it was asked for.
scan_manager_config lru_config()
{
  scan_manager_config cfg;
  cfg.thread_pool.num_threads = 3;
  cfg.uring_n_reactors        = 1;
  cfg.cache.mode              = sirius::io::cache::cache_mode::sirius;
  cfg.cache.eviction          = sirius::io::cache::eviction_policy::lru;
  cfg.apply_cache_mode();
  return cfg;
}

}  // namespace

TEST_CASE("claimed_bytes tracks the staging memory the cache is holding",
          "[cache][eviction][explicit]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius_scan_manager manager{lru_config(), *memory, topology};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());

  CHECK(cache->claimed_bytes() == 0);

  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, 4ll << 20);
  auto ds = manager.create_datasource(file.path.string());
  REQUIRE(ds != nullptr);
  ds->fadvise(ranges, 0);
  // Nothing attaches staging buffers on its own -- fadvise only registers the
  // request and its chunks.  Until this call the request names chunks but holds
  // no memory, which is exactly what claimed_bytes should report.
  CHECK(cache->claimed_bytes() == 0);
  REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);

  INFO("cache: " << cache->summary());
  CHECK(cache->claimed_bytes() > 0);
}

TEST_CASE("an explicit evict reclaims a disposed request the pressure rule would not",
          "[cache][eviction][explicit]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius_scan_manager manager{lru_config(), *memory, topology};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());

  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, 4ll << 20);
  {
    auto ds = manager.create_datasource(file.path.string());
    REQUIRE(ds != nullptr);
    ds->fadvise(ranges, 0);
    // Attach staging buffers before the request is disposed, so the chunks are
    // genuinely reclaimable rather than never allocated in the first place.
    REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  }  // ~sirius_datasource -> ~prefetching_handle -> the consumer is disposed

  auto const claimed = cache->claimed_bytes();
  REQUIRE(claimed > 0);

  // Nothing has crossed the pool's threshold, so the evictor has no reason of
  // its own to sweep: the memory stays put until it is asked for.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  REQUIRE(cache->claimed_bytes() == claimed);

  cache->evict(claimed);
  INFO("cache: " << cache->summary());
  CHECK(claimed_drops_below(*cache, claimed, std::chrono::milliseconds(2000)));
}

TEST_CASE("an explicit evict frees at least what was asked for", "[cache][eviction][explicit]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius_scan_manager manager{lru_config(), *memory, topology};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);

  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, 4ll << 20);
  {
    auto ds = manager.create_datasource(file.path.string());
    REQUIRE(ds != nullptr);
    ds->fadvise(ranges, 0);
    REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  }

  auto const claimed = cache->claimed_bytes();
  REQUIRE(claimed > 0);

  // Ask for a single byte.  Chunks are the only granularity there is, so this
  // frees exactly one -- the assertion is that the demand is a floor and not a
  // suggestion, not that the arithmetic is byte-exact.
  cache->evict(1);
  CHECK(claimed_drops_below(*cache, claimed, std::chrono::milliseconds(2000)));
}

TEST_CASE("a zero-byte evict is a no-op", "[cache][eviction][explicit]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius_scan_manager manager{lru_config(), *memory, topology};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);

  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, 4ll << 20);
  {
    auto ds = manager.create_datasource(file.path.string());
    REQUIRE(ds != nullptr);
    ds->fadvise(ranges, 0);
    REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  }

  auto const claimed = cache->claimed_bytes();
  REQUIRE(claimed > 0);

  cache->evict(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  CHECK(cache->claimed_bytes() == claimed);
}
