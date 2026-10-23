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
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
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
    if (bytes != 0) {
      out.seekp(static_cast<std::streamoff>(bytes - 1));
      out.put('z');
    }
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

scan_manager_config constrained_lru_config()
{
  auto cfg = lru_config();
  // Keep automatic pressure eviction disabled while the fixture deliberately
  // fills the small host tier. Only failed preparations' explicit requests may
  // free the disposable requests below.
  cfg.cache.min_prefetching_budget_fraction = 0.25;
  cfg.cache.eviction_threshold_fraction     = 1.0;
  return cfg;
}

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> constrained_memory_manager()
{
  sirius::converter_registry::reset_for_testing();

  reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2ull << 30)
    .set_reservation_fraction_per_gpu(0.75)
    // Large enough for the allocator's baseline pools, but small enough that a
    // handful of cache requests deterministically exhaust it.
    .set_per_numa_region_capacity(256ull << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(1.0);

  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
  sirius::converter_registry::initialize();
  return manager;
}

}  // namespace

TEST_CASE("claimed_bytes tracks the staging memory the cache is holding",
          "[cache][eviction][explicit]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
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

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
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
  }  // ~sirius_datasource -> ~cache_handle -> the consumer is disposed

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

TEST_CASE("blocking prepare waits for earlier asynchronous eviction before retrying",
          "[cache][eviction][explicit][prepare]")
{
  constexpr std::size_t request_bytes = 32ull << 20;
  temp_data_file file(1ull << 30);
  auto memory   = constrained_memory_manager();
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, constrained_lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);

  // Fill whatever portion of the constrained tier remains after ioctx's own
  // baseline allocations. Every successful request is immediately disposable,
  // but LRU retains its buffers because the automatic threshold is disabled.
  std::size_t offset = 0;
  std::shared_ptr<sirius::io::sirius_datasource> target;
  for (std::size_t attempt = 0; attempt < 32; ++attempt) {
    std::vector<cudf::io::text::byte_range_info> ranges;
    ranges.emplace_back(static_cast<std::int64_t>(offset),
                        static_cast<std::int64_t>(request_bytes));
    auto filler = manager.create_datasource(file.path.string());
    REQUIRE(filler != nullptr);
    filler->fadvise(ranges, 0);
    if (filler->prepare_prefetch(false) == sirius::io::prepare_result::allocation_failed) {
      target = std::move(filler);
      break;
    }
    offset += request_bytes;
  }
  REQUIRE(target != nullptr);

  // The first attempt queued eviction without waiting. The blocking retry
  // waits for that pass rather than doubling the requested shortfall.
  REQUIRE(target->prepare_prefetch(true) == sirius::io::prepare_result::prepared);
}

TEST_CASE("failed nonblocking preparation starts eviction without waiting",
          "[cache][eviction][explicit][prepare]")
{
  constexpr std::size_t mib = 1ull << 20;
  temp_data_file file(1ull << 30);
  auto memory   = constrained_memory_manager();
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, constrained_lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);

  // Provide an unsubscribed victim, then fill the remaining capacity with
  // live requests that cannot be reclaimed by the evictor's first pass.
  {
    auto victim = manager.create_datasource(file.path.string());
    REQUIRE(victim != nullptr);
    std::vector<cudf::io::text::byte_range_info> ranges;
    ranges.emplace_back(0, static_cast<std::int64_t>(16 * mib));
    victim->fadvise(ranges, 0);
    REQUIRE(victim->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  }

  std::vector<std::shared_ptr<sirius::io::sirius_datasource>> fillers;
  bool exhausted = false;
  for (std::size_t i = 0; i < 32; ++i) {
    auto filler = manager.create_datasource(file.path.string());
    REQUIRE(filler != nullptr);
    std::vector<cudf::io::text::byte_range_info> ranges;
    ranges.emplace_back(static_cast<std::int64_t>((16 + 32 * i) * mib),
                        static_cast<std::int64_t>(32 * mib));
    filler->fadvise(ranges, 0);
    auto const before = cache->claimed_bytes();
    if (filler->prepare_prefetch(false) == sirius::io::prepare_result::allocation_failed) {
      exhausted = true;
      // No explicit evict() or blocking prepare(true): the failed attempt
      // itself must have woken the evictor.
      CHECK(claimed_drops_below(*cache, before, std::chrono::milliseconds(2000)));
      break;
    }
    fillers.push_back(std::move(filler));
  }
  REQUIRE(exhausted);
}

TEST_CASE("prepare does not publish a request the retry's eviction left short of buffers",
          "[cache][eviction][explicit][prepare]")
{
  constexpr std::size_t mib = 1ull << 20;  // also the pool's chunk size
  temp_data_file file(2ull << 30);
  auto memory   = constrained_memory_manager();
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  // The shortfall comes from the cache's own 40 MiB cap, not from a full host
  // tier: a full tier is spill pressure, under which the evictor takes idle
  // chunks only and never reaches the subscriber-ignoring pass under test.
  auto cfg                              = constrained_lru_config();
  cfg.cache.eviction_threshold_fraction = 40.0 / 256.0;
  sirius_scan_manager manager{scan_lifecycle, cfg, *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);

  REQUIRE(cache->chunk_size() == mib);
  REQUIRE(cache->max_prefetching_budget_bytes() == 40 * mib);

  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, static_cast<std::int64_t>(36 * mib));
  auto victim = manager.create_datasource(file.path.string());
  REQUIRE(victim != nullptr);
  victim->fadvise(ranges, 0);
  REQUIRE(victim->prepare_prefetch(false) == sirius::io::prepare_result::prepared);

  // The target names 32 MiB of the victim's chunks -- still buffered, so the
  // shortfall count skips them -- plus 32 MiB of empty ones past the gap.
  std::vector<cudf::io::text::byte_range_info> target_ranges;
  target_ranges.emplace_back(0, static_cast<std::int64_t>(32 * mib));
  target_ranges.emplace_back(static_cast<std::int64_t>(36 * mib),
                             static_cast<std::int64_t>(32 * mib));
  auto target = manager.create_datasource(file.path.string());
  REQUIRE(target != nullptr);
  target->fadvise(target_ranges, 0);

  cache->evict_sync(mib);
  REQUIRE(cache->claimed_bytes() == 36 * mib);

  // Buffered, then disposed: the only chunks in the tier the evictor may take.
  // Its last 4 MiB are named by nothing else below, so the request stays an
  // eviction candidate instead of being retired the moment it is released.
  victim.reset();

  // 36 MiB held plus 32 MiB wanted is 28 MiB over the cap.  The victim's
  // unsubscribed tail covers 4 of them, so the subscriber-ignoring pass takes
  // 24 of the target's own prefix chunks as well.  The attach loop then walks
  // those first and spends the buffers on them, leaving part of the empty tail
  // with none -- and a request whose chunks cannot all be claimed for loading
  // is not `prepared`.
  INFO("cache: " << cache->summary());
  CHECK(target->prepare_prefetch(true) == sirius::io::prepare_result::allocation_failed);
}

TEST_CASE("prepare abandons immediately after the consumer reaches the split",
          "[cache][eviction][explicit][prepare]")
{
  temp_data_file file(8ull << 20);
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, lru_config(), *memory, topology};
  // Built before the datasource registers its request; the default cache is
  // otherwise only built when a query reads through it.
  REQUIRE(manager.ensure_default_cache() != nullptr);
  auto ds = manager.create_datasource(file.path.string());
  REQUIRE(ds != nullptr);

  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, 4ll << 20);
  ds->fadvise(ranges, 0);
  ds->update(sirius::io::cache::scan_stage::preparing);

  CHECK(ds->prepare_prefetch(true) == sirius::io::prepare_result::fallen_behind);
  CHECK(manager.ensure_default_cache()->claimed_bytes() == 0);
}

TEST_CASE("an explicit evict frees at least what was asked for", "[cache][eviction][explicit]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
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
  cache->evict_sync(1);
  CHECK(cache->claimed_bytes() < claimed);
}

TEST_CASE("a zero-byte evict is a no-op", "[cache][eviction][explicit]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
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

TEST_CASE("the evictor retires disposed requests while the pool is under its threshold",
          "[cache][eviction][explicit]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());

  // Requests that fall behind: fadvise registers the chunks, nothing ever
  // prepares them, so they stay `empty` and the pool never approaches the
  // pressure threshold.  Each fadvise is also what wakes the evictor for a
  // round, and every round must retire the requests disposed before it --
  // otherwise the batch is one entry per request for the life of the cache.
  constexpr int n_requests = 64;
  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, 4ll << 20);
  for (int i = 0; i < n_requests; ++i) {
    auto ds = manager.create_datasource(file.path.string());
    REQUIRE(ds != nullptr);
    ds->fadvise(ranges, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }  // ~sirius_datasource -> the consumer is disposed before the next round

  // The batch is a background thread's state, so give it a deadline rather than
  // a single read.  Steady state is the round's own arrivals plus the previous
  // round's last request; the bound is loose enough to absorb a round that
  // absorbed two of them at once and tight enough that "one entry per request
  // ever issued" cannot pass.
  constexpr std::size_t max_tracked = 8;
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  std::size_t tracked = cache->eviction_batch_size_for_testing();
  while (tracked > max_tracked && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    tracked = cache->eviction_batch_size_for_testing();
  }
  INFO("cache: " << cache->summary());
  CHECK(tracked <= max_tracked);
}

TEST_CASE("residency reports the loaded share of an object by path", "[cache][residency]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());
  auto const chunk = cache->chunk_size();
  REQUIRE(2 * chunk <= (8ull << 20));

  // A path the cache has never seen: nothing held, and no size to report either.
  auto const unseen = cache->residency(file.path.string());
  CHECK(unseen.resident_bytes == 0);
  CHECK(unseen.object_bytes == 0);

  // Read two chunks through the cache. Only loaded chunks count; the rest of the file does not.
  auto ds = manager.create_datasource(file.path.string());
  REQUIRE(ds != nullptr);
  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, static_cast<std::int64_t>(2 * chunk));
  ds->fadvise(ranges, 0);
  REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  std::vector<std::uint8_t> bytes(2 * chunk);
  REQUIRE(ds->host_read(0, bytes.size(), bytes.data()) == bytes.size());

  auto const held = cache->residency(file.path.string());
  INFO("cache: " << cache->summary());
  CHECK(held.object_bytes == (8ull << 20));
  CHECK(held.resident_bytes == 2 * chunk);
}

TEST_CASE("release_idle gives back idle chunks above the reservation and keeps live ones",
          "[cache][eviction][release_idle]")
{
  constexpr std::size_t mib = 1ull << 20;  // also the pool's chunk size
  temp_data_file file(256ull << 20);
  auto memory   = constrained_memory_manager();
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, constrained_lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->chunk_size() == mib);

  auto prepare = [&](std::int64_t offset, std::int64_t bytes) {
    std::vector<cudf::io::text::byte_range_info> ranges;
    ranges.emplace_back(offset, bytes);
    auto ds = manager.create_datasource(file.path.string());
    REQUIRE(ds != nullptr);
    ds->fadvise(ranges, 0);
    REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
    return ds;
  };
  // 64 MiB of the 256 MiB tier is the pool's reservation (constrained_lru_config).
  auto live = prepare(0, static_cast<std::int64_t>(8 * mib));
  auto idle = prepare(static_cast<std::int64_t>(8 * mib), static_cast<std::int64_t>(92 * mib));
  REQUIRE(cache->claimed_bytes() == 100 * mib);
  idle.reset();

  // Only the 36 MiB above the reservation reach the host tier; the live request's chunks stay.
  CHECK(cache->release_idle(1ull << 30) == 36 * mib);
  CHECK(cache->claimed_bytes() == 64 * mib);
  CHECK(cache->release_idle(1ull << 30) == 0);
  live.reset();
}

TEST_CASE("the pool grows past its reservation only while the host tier is below its stop mark",
          "[cache][eviction][release_idle]")
{
  constexpr std::size_t mib = 1ull << 20;
  temp_data_file file(256ull << 20);
  auto memory   = constrained_memory_manager();
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, constrained_lru_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);
  auto host_spaces = memory->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(host_spaces.size() == 1);
  auto* host_mr = host_spaces.front()->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  REQUIRE(host_mr != nullptr);

  auto make = [&](std::int64_t offset, std::int64_t bytes) {
    std::vector<cudf::io::text::byte_range_info> ranges;
    ranges.emplace_back(offset, bytes);
    auto ds = manager.create_datasource(file.path.string());
    REQUIRE(ds != nullptr);
    ds->fadvise(ranges, 0);
    return ds;
  };
  auto first = make(0, static_cast<std::int64_t>(32 * mib));
  REQUIRE(first->prepare_prefetch(false) == sirius::io::prepare_result::prepared);

  // Spilled data stand-in: everything the tier has left but a few chunks, which takes it past
  // its downgrade stop threshold (65% by default).
  auto const available = host_mr->get_available_memory();
  REQUIRE(available > 8 * mib);
  auto pressure = host_mr->allocate_multiple_blocks((available / mib - 8) * mib);
  REQUIRE(pressure != nullptr);
  REQUIRE_FALSE(host_spaces.front()->should_stop_downgrading_memory());

  // 32 MiB held plus 48 MiB wanted is past the 64 MiB reservation: refused under pressure.
  auto second = make(static_cast<std::int64_t>(32 * mib), static_cast<std::int64_t>(48 * mib));
  CHECK(second->prepare_prefetch(false) == sirius::io::prepare_result::allocation_failed);
  CHECK(cache->claimed_bytes() == 32 * mib);

  // Once the pressure is gone the same request fits.
  pressure.reset();
  REQUIRE(host_spaces.front()->should_stop_downgrading_memory());
  CHECK(second->prepare_prefetch(true) == sirius::io::prepare_result::prepared);
  CHECK(cache->claimed_bytes() == 80 * mib);
}

TEST_CASE("an uncapped pool over the host's stop line gives way and keeps serving",
          "[cache][eviction][release_idle]")
{
  constexpr std::size_t mib = 1ull << 20;
  temp_data_file file(512ull << 20);
  auto memory   = constrained_memory_manager();
  auto topology = single_gpu_index_for_evict();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  auto cfg                              = constrained_lru_config();
  cfg.cache.eviction_threshold_fraction = 0.0;  // uncapped: only the host tier limits the cache
  sirius_scan_manager manager{scan_lifecycle, cfg, *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);
  auto host_spaces = memory->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(host_spaces.size() == 1);
  auto const* host = host_spaces.front();
  auto* host_mr    = host->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  REQUIRE(host_mr != nullptr);

  auto read = [&](std::size_t i) {
    std::vector<cudf::io::text::byte_range_info> ranges;
    ranges.emplace_back(static_cast<std::int64_t>(i * 16 * mib),
                        static_cast<std::int64_t>(16 * mib));
    auto ds = manager.create_datasource(file.path.string());
    REQUIRE(ds != nullptr);
    ds->fadvise(ranges, 0);
    return ds->prepare_prefetch(true);
  };

  // 80 MiB of idle chunks, past the pool's 64 MiB reservation and under the stop line.
  for (std::size_t i = 0; i < 5; ++i) {
    REQUIRE(read(i) == sirius::io::prepare_result::prepared);
  }
  REQUIRE(cache->claimed_bytes() == 80 * mib);
  REQUIRE(host->should_stop_downgrading_memory());

  // Spilled data stand-in: take the tier well past the stop line, by more than one request.
  std::vector<
    std::unique_ptr<cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation>>
    spill;
  while (host->get_amount_to_downgrade() < 40 * mib) {
    spill.push_back(host_mr->allocate_multiple_blocks(8 * mib));
    REQUIRE(spill.back() != nullptr);
  }

  // Every request is still served: the cache evicts its overshoot from idle chunks and then
  // fits, rather than refusing for as long as the tier stays over the line.
  for (std::size_t i = 5; i < 9; ++i) {
    INFO("request " << i << ", cache: " << cache->summary());
    CHECK(read(i) == sirius::io::prepare_result::prepared);
  }
  CHECK(cache->claimed_bytes() < 80 * mib);
  CHECK(host->get_amount_to_downgrade() < 40 * mib);
}

TEST_CASE("readahead bounds memory retries and returns capacity on abandonment",
          "[cache][eviction][readahead][retry]")
{
  using namespace std::chrono_literals;
  using sirius::io::cache::scan_stage;
  using sirius::op::scan::scan_info;
  using sirius::scan_manager::prefetch_outcome_kind;
  using sirius::scan_manager::readahead_scan_manager;
  using sirius::scan_manager::shared_scan_budget;

  // The cache reserves 64 MiB at construction. Exhaust the remaining tier and
  // ask for more than that reserve so preparation really encounters pressure.
  constexpr std::int64_t request_bytes = 96ll << 20;
  temp_data_file file(2 * request_bytes);
  auto memory = constrained_memory_manager();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager scans{
    lifecycle, constrained_lru_config(), *memory, single_gpu_index_for_evict()};
  // The default cache is built on first use; build it before the tier is exhausted.
  REQUIRE(scans.ensure_default_cache() != nullptr);
  auto host     = memory->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
  auto* host_mr = host->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  auto pressure = host_mr->allocate_multiple_blocks(host_mr->get_available_memory());
  REQUIRE(pressure);
  REQUIRE(host_mr->get_available_memory() == 0);

  // A real query order and real cache pressure exercise the production worker,
  // preparation, eviction and completion paths without injected allocation failures.
  struct scan_operator : sirius::op::sirius_physical_operator {
    scan_operator()
      : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::GPU_SCAN, {}, 0)
    {
      operator_id = 1;
    }
  } op;
  auto telemetry = sirius::test::make_test_telemetry_context();
  sirius::pipeline::pipeline_build_context ctx{nullptr, true};
  auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(ctx);
  sirius::pipeline::sirius_pipeline_build_state build;
  build.set_pipeline_source(*pipeline, op);
  auto id = sirius::make_query_id(1);
  sirius::planner::query query(
    {pipeline},
    telemetry->context(),
    id,
    sirius::telemetry::query_telemetry_info{quent::now_v7(), telemetry->worker_id().raw(), id});
  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  auto budget    = std::make_shared<shared_scan_budget>();
  auto readahead = std::make_shared<readahead_scan_manager>(*publisher, 1, budget, 1);
  readahead->prepare_for_query(query);
  auto make_split = [&](std::int64_t offset) {
    auto ds = scans.create_datasource(file.path.string());
    std::vector<cudf::io::text::byte_range_info> ranges{{offset, request_bytes}};
    ds->fadvise(ranges, 0);
    return std::make_shared<scan_info>(std::vector<scan_info::fadvise_entry>{{ds, ranges}});
  };
  auto split = make_split(0);
  readahead->register_scan_task(split, op.get_operator_id());
  auto backend = reinterpret_cast<std::uintptr_t>(scans.io_ctx());
  auto await   = [](auto predicate) {
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(1ms);
    return predicate();
  };

  SECTION("persistent pressure is accounted once and a later candidate can progress")
  {
    readahead->start();
    REQUIRE(await([&] {
      return readahead->counters().outcome(prefetch_outcome_kind::skipped_memory_pressure) == 1;
    }));
    CHECK(readahead->counters().candidates_taken == 1);
    CHECK(readahead->counters().memory_retries == 2);
    CHECK(budget->active(backend) == 0);
    // Returning memory lets a subsequent candidate prove both capacity tickets
    // were released and that exhaustion did not stop the prefetch worker.
    pressure.reset();
    auto next = make_split(request_bytes);
    readahead->register_scan_task(next, op.get_operator_id());
    REQUIRE(
      await([&] { return readahead->counters().outcome(prefetch_outcome_kind::prefetched) == 1; }));
    readahead->stop();
    CHECK(readahead->counters().candidates_taken == 2);
    CHECK(readahead->counters().outcome(prefetch_outcome_kind::skipped_memory_pressure) == 1);
    // The IO completion returns its shared ticket after recording the outcome.
    CHECK(await([&] { return budget->active(backend) == 0; }));
  }
  SECTION("stop during memory retries settles the candidate exactly once")
  {
    readahead->start();
    REQUIRE(await([&] { return readahead->counters().memory_retries >= 1; }));
    readahead->stop();
    CHECK(readahead->counters().candidates_taken == 1);
    CHECK(readahead->counters().memory_retries <= 2);
    CHECK(readahead->counters().outcome(prefetch_outcome_kind::skipped_memory_pressure) == 1);
    CHECK(budget->active(backend) == 0);
  }
  SECTION("cancellation while waiting leaves another reader's debt intact")
  {
    auto demand = budget->acquire({{backend, 1}}, sirius::make_query_id(2), {}, true);
    readahead->start();
    REQUIRE(await([&] { return budget->pending() == 1; }));
    readahead->stop();
    CHECK(readahead->counters().candidates_taken == 1);
    CHECK(readahead->counters().memory_retries == 0);
    CHECK(readahead->counters().outcome(prefetch_outcome_kind::skipped_memory_pressure) == 1);
    CHECK(budget->pending() == 0);
    CHECK(budget->active(backend) == 1);
  }
}
