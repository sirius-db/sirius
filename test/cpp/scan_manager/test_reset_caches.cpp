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

// Coverage of sirius_scan_manager::reset_caches(), which is what the
// reset_sirius_cache() table function calls: every ioctx's prefetching cache is
// dropped and rebuilt empty, or left alone where the configuration gives no
// cache in the first place.

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
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index_for_reset()
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
           ("sirius_reset_caches_" + std::to_string(::getpid()) + "_" +
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

scan_manager_config config_with_cache(sirius::io::cache::cache_mode mode)
{
  scan_manager_config cfg;
  cfg.thread_pool.num_threads = 3;
  cfg.uring_n_reactors        = 1;
  cfg.cache.mode              = mode;
  cfg.cache.eviction          = sirius::io::cache::eviction_policy::lru;
  cfg.apply_cache_mode();
  return cfg;
}

/// Populate the default ioctx's cache with staging buffers, so a reset has
/// something real to drop.  Returns the bytes it ended up holding.
std::size_t claim_some_cache(sirius_scan_manager& manager, temp_data_file const& file)
{
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);

  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, 4ll << 20);
  auto ds = manager.create_datasource(file.path.string());
  REQUIRE(ds != nullptr);
  ds->fadvise(ranges, 0);
  // Nothing attaches staging buffers on its own; fadvise only registers the
  // request and the chunks it names.
  REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  return cache->claimed_bytes();
}

/// Pull `global[... evictions=N ...]` out of prefetching_cache::summary().  The
/// counters live on the cache object, so a non-zero count going back to zero is
/// the observable proof that the object was replaced rather than emptied --
/// pointer identity is not, since the allocator readily hands back the address
/// it just freed.
std::uint64_t evictions_from(std::string const& summary)
{
  std::smatch m;
  static std::regex const re(R"(global\[[^\]]*evictions=(\d+))");
  if (!std::regex_search(summary, m, re)) { return 0; }
  return std::stoull(m[1].str());
}

/// Poll rather than sleep once: the evictor is a background thread, so a sweep
/// that is merely slow must not read as one that never happened.
std::uint64_t evictions_within(sirius::io::cache::prefetching_cache& cache,
                               std::chrono::milliseconds budget)
{
  auto const deadline = std::chrono::steady_clock::now() + budget;
  std::uint64_t seen  = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    seen = evictions_from(cache.summary());
    if (seen > 0) { return seen; }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return seen;
}

}  // namespace

TEST_CASE("reset_caches replaces a populated cache with an empty one",
          "[scan_manager][cache][reset_cache]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_reset();

  sirius_scan_manager manager{
    config_with_cache(sirius::io::cache::cache_mode::sirius), *memory, topology};
  auto* before = manager.io_ctx()->cache();
  REQUIRE(before != nullptr);
  bool const was_armed = before->is_armed();

  auto const claimed = claim_some_cache(manager, file);
  REQUIRE(claimed > 0);
  // Move a counter off zero, so "the counters are zero again" afterwards means
  // the object was replaced rather than just found in its initial state.
  before->evict(claimed);
  REQUIRE(evictions_within(*before, std::chrono::milliseconds(2000)) > 0);

  manager.reset_caches();

  auto* after = manager.io_ctx()->cache();
  REQUIRE(after != nullptr);
  // A genuinely new cache, not the old one emptied in place: the file entries,
  // their chunk arenas and the counters all go with the object, which is the
  // only way to be sure nothing of the previous query's residency survives.
  CHECK(after->claimed_bytes() == 0);
  CHECK(evictions_from(after->summary()) == 0);
  // Rebuilt from the same config against the same backend, so it comes up in
  // the same armed state -- a reset must not quietly disable prefetching.
  CHECK(after->is_armed() == was_armed);
}

TEST_CASE("a rebuilt cache is usable again", "[scan_manager][cache][reset_cache]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_reset();

  sirius_scan_manager manager{
    config_with_cache(sirius::io::cache::cache_mode::sirius), *memory, topology};
  REQUIRE(claim_some_cache(manager, file) > 0);
  manager.reset_caches();

  // The point of rebuilding rather than just dropping: the next query has to
  // find a working cache, not an absent one.
  REQUIRE(manager.io_ctx()->cache() != nullptr);
  CHECK(claim_some_cache(manager, file) > 0);
}

TEST_CASE("reset_caches is idempotent", "[scan_manager][cache][reset_cache]")
{
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_reset();

  sirius_scan_manager manager{
    config_with_cache(sirius::io::cache::cache_mode::sirius), *memory, topology};
  REQUIRE(manager.io_ctx()->cache() != nullptr);

  manager.reset_caches();
  manager.reset_caches();

  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  CHECK(cache->claimed_bytes() == 0);
}

TEST_CASE("reset_caches reclaims still-resident chunk buffers, not just evicted ones",
          "[scan_manager][cache][reset_cache]")
{
  // claim_some_cache() only attaches buffers (fadvise + prepare_prefetch); unlike
  // the "replaces a populated cache" test above, nothing here evicts them before
  // reset_caches() runs. This is the exact shape that used to leak: the old
  // prefetching_cache destructor tore down _file_cache and _pool without ever
  // returning a still-resident chunk's buffer to the underlying
  // fixed_size_host_memory_resource, so the block was gone from its free list
  // for the rest of the process's life even though the cache object reporting
  // it was destroyed.
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_reset();

  auto* host_space = sirius::scan_test_utils::get_space(*memory, cucascade::memory::Tier::HOST);
  REQUIRE(host_space != nullptr);
  auto* host_mr =
    host_space->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  REQUIRE(host_mr != nullptr);

  sirius_scan_manager manager{
    config_with_cache(sirius::io::cache::cache_mode::sirius), *memory, topology};
  REQUIRE(manager.io_ctx()->cache() != nullptr);

  // Baseline AFTER construction: the scan manager's own io/uring buffers are
  // blocks too, and they legitimately stay held while `manager` is alive. Only
  // the delta the cache claim adds on top is what a reset has to give back.
  auto const free_before = host_mr->get_free_blocks();

  REQUIRE(claim_some_cache(manager, file) > 0);
  // The claim checked real blocks out of the free list -- confirms the test is
  // actually exercising the resident (non-evicted) case, not a no-op.
  REQUIRE(host_mr->get_free_blocks() < free_before);

  manager.reset_caches();

  // Every block the populated cache was holding must be back on the free list --
  // not merely accounted as "reservation released" but actually deallocated
  // from the shared resource that later resets and queries draw from.
  CHECK(host_mr->get_free_blocks() == free_before);
}

TEST_CASE("reset_caches is a no-op where the configuration does not cache",
          "[scan_manager][cache][reset_cache]")
{
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_reset();

  sirius_scan_manager manager{
    config_with_cache(sirius::io::cache::cache_mode::none), *memory, topology};
  // Caching off, so no cache was ever built...
  REQUIRE(manager.io_ctx()->cache() == nullptr);

  manager.reset_caches();

  // ...and the reset must not be what teaches this context to cache.
  CHECK(manager.io_ctx()->cache() == nullptr);
}
