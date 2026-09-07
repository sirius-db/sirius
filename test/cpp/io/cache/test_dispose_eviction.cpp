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

// End-to-end coverage of the `cache.eviction: idle` (dispose_on_idle) reclaim
// path, driven through a real uring-backed sirius_datasource so the chunk
// states, the prepare loop and the evictor are the engine's own.

#include "catch.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"
#include "io/sirius_datasource.hpp"
#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index_for_dispose()
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
           ("sirius_dispose_evict_" + std::to_string(::getpid()) + "_" +
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

/// Pull `global[... evictions=N ...]` out of prefetching_cache::summary().
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

scan_manager_config dispose_on_idle_config()
{
  scan_manager_config cfg;
  cfg.thread_pool.num_threads = 3;
  cfg.uring_n_reactors        = 1;
  cfg.cache.mode              = sirius::io::cache::cache_mode::sirius;
  cfg.cache.eviction          = sirius::io::cache::eviction_policy::idle;
  cfg.apply_cache_mode();
  return cfg;
}

}  // namespace

TEST_CASE("dispose_on_idle reclaims a disposed request's chunks on the next sweep",
          "[cache][eviction][dispose]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_dispose();

  sirius_scan_manager manager{dispose_on_idle_config(), *memory, topology};
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
    // Nothing does this on its own -- fadvise only registers the request and the
    // chunks it names -- so without this call there is no buffer to reclaim and
    // the evictor correctly evicts nothing.
    REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  }  // ~sirius_datasource -> ~prefetching_handle -> the consumer is disposed

  // The evictor only sweeps when a request reaches it, so a second insert is
  // what gives the disposed request its sweep. Reclaiming
  // then is the behavior under test: before the size_t/double ternary fix, the
  // `need` target computed to 0 and the reclaim passes were skipped outright,
  // so a disposed request's chunks were never freed under any amount of
  // sweeping.
  std::vector<cudf::io::text::byte_range_info> other;
  other.emplace_back(6ll << 20, 1ll << 20);
  auto ds2 = manager.create_datasource(file.path.string());
  REQUIRE(ds2 != nullptr);
  ds2->fadvise(other, 0);

  INFO("cache: " << cache->summary());
  CHECK(evictions_within(*cache, std::chrono::milliseconds(2000)) > 0);
}
