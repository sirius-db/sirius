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

// Coverage of sirius_scan_manager::pin_parquet_ranges()'s all-or-nothing
// contract: a pin that throws part way through must leave nothing pinned, and a
// failed re-pin must leave the previous pin exactly as it was.
//
// The observable is evictability rather than the bytes themselves.  A pinned
// file's datasource owns the cache_handle whose live consumer makes the
// evictor skip the request entirely, so chunks that survive an evict_sync() are
// still pinned and chunks it reclaims are not.

#include "catch.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"
#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index_for_pin()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

scan_manager_config config_with_sirius_cache()
{
  scan_manager_config cfg;
  cfg.thread_pool.num_threads = 3;
  cfg.uring_n_reactors        = 1;
  cfg.cache.mode              = sirius::io::cache::cache_mode::sirius;
  cfg.cache.eviction          = sirius::io::cache::eviction_policy::lru;
  cfg.apply_cache_mode();
  return cfg;
}

std::string good_parquet()
{
  return (std::filesystem::path{SIRIUS_PROJECT_ROOT} / "test" / "cpp" / "integration" / "data" /
          "parquet" / "customer.parquet")
    .string();
}

std::string missing_parquet() { return "/nonexistent/sirius_pin_rollback.parquet"; }

/// Bytes still held after asking the evictor to reclaim everything it can.
/// Anything left over is held by a live pin, which is the property under test.
std::size_t bytes_after_full_eviction(sirius::io::cache::prefetching_cache& cache)
{
  cache.evict_sync(cache.claimed_bytes());
  return cache.claimed_bytes();
}

}  // namespace

TEST_CASE("a pin that throws part way through pins nothing", "[scan_manager][cache][pin_parquet]")
{
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_pin();

  sirius_scan_manager manager{config_with_sirius_cache(), *memory, topology};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());

  // The good file pins first, then the missing one fails: the failure is the
  // second iteration, so the first file's handles already exist.
  REQUIRE_THROWS(
    manager.pin_parquet_ranges("rollback", {good_parquet(), missing_parquet()}, std::nullopt));

  // Nothing may be left pinned: with the partial entry retained, the first
  // file's request still has a live consumer and the evictor skips it forever.
  CHECK(bytes_after_full_eviction(*cache) == 0);
}

TEST_CASE("a failed re-pin leaves the previous pin intact", "[scan_manager][cache][pin_parquet]")
{
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_pin();

  sirius_scan_manager manager{config_with_sirius_cache(), *memory, topology};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());

  REQUIRE(manager.pin_parquet_ranges("reptin", {good_parquet()}, std::nullopt) > 0);
  auto const pinned_bytes = bytes_after_full_eviction(*cache);
  // A pin that survives an eviction sweep is what the rest of the case measures
  // against, so it has to be real.
  REQUIRE(pinned_bytes > 0);

  // Fails on its first file, so the re-pin never pins anything of its own --
  // whatever is still resident afterwards can only be the original pin.
  REQUIRE_THROWS(
    manager.pin_parquet_ranges("reptin", {missing_parquet(), good_parquet()}, std::nullopt));

  CHECK(bytes_after_full_eviction(*cache) == pinned_bytes);
}

TEST_CASE("reset_caches drops parquet pins rather than leave them stale",
          "[scan_manager][cache][pin_parquet][reset_cache]")
{
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_pin();

  sirius_scan_manager manager{config_with_sirius_cache(), *memory, topology};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());

  REQUIRE(manager.pin_parquet_ranges("resetpin", {good_parquet()}, std::nullopt) > 0);
  REQUIRE(bytes_after_full_eviction(*cache) > 0);

  manager.reset_caches();

  // The chunks died with the old cache, so the registry must stop naming the
  // pin: a retained datasource here would claim a residency the rebuilt cache
  // does not have, and its handle would outlive the cache that minted it.
  auto* rebuilt = manager.io_ctx()->cache();
  REQUIRE(rebuilt != nullptr);
  CHECK(rebuilt->claimed_bytes() == 0);
  CHECK(manager.pinned_parquet_count_for_testing() == 0);
}
