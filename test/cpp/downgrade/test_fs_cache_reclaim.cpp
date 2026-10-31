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

#include "catch.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "utils/sirius_test_env.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/io_context.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::io::cache {

// The cache's existing friend seam lets these integration tests retain the handle and wait
// for subscriber disposal deterministically, without reaching into the evictor's queues.
struct fs_cache_test_access {
  static cache_handle prepared_reclaim_handle(fs_cache& cache,
                                              io_object const& object,
                                              std::span<const byte_range> ranges)
  {
    auto handle = cache.initiate_prefetching_request(object, ranges);
    REQUIRE(cache.prepare(handle, false) == prepare_result::prepared);
    return handle;
  }
};

}  // namespace cucascade::io::cache

namespace {

using namespace std::chrono_literals;
using cucascade::memory::Tier;
using sirius::parallel::downgrade_executor;
namespace cache = cucascade::io::cache;

struct temporary_file {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
                               ("sirius_fs_cache_reclaim_" + std::to_string(::getpid()) + "_" +
                                std::to_string(reinterpret_cast<std::uintptr_t>(this)));

  explicit temporary_file(std::size_t bytes)
  {
    std::ofstream output(path, std::ios::binary);
    std::vector<char> data(bytes, 'x');
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
    REQUIRE(output.good());
  }

  explicit temporary_file(std::string const& contents)
  {
    std::ofstream output(path);
    output << contents;
    REQUIRE(output.good());
  }

  ~temporary_file()
  {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
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

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> reclaim_memory_manager()
{
  sirius::converter_registry::reset_for_testing();
  cucascade::memory::reservation_manager_configurator builder;
  // Unlike the general scan fixture's 75% reservation limit / 85% trigger,
  // these thresholds can be crossed with reservations alone (no large allocation).
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2ull << 30)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_numa_region_capacity(512ull << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(1.0)
    .set_downgrade_fractions_per_numa_region(0.8, 0.7);
  auto memory =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
  sirius::converter_registry::initialize();
  return memory;
}

sirius::scan_manager::scan_manager_config cached_scan_config(double floor_fraction)
{
  sirius::scan_manager::scan_manager_config config;
  config.thread_pool.num_threads               = 1;
  config.uring_reactors_per_gpu                = 1;
  config.cache.mode                            = cache::cache_mode::cucs;
  config.cache.eviction                        = cache::eviction_policy::lru;
  config.cache.min_prefetching_budget_fraction = floor_fraction;
  config.apply_cache_mode();
  return config;
}

struct reclaim_fixture {
  temporary_file file{16ull << 20};
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    reclaim_memory_manager();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius::data::data_repository_manager_registry repositories;
  sirius::scan_manager::sirius_scan_manager scans;
  std::unique_ptr<downgrade_executor> executor;

  explicit reclaim_fixture(std::optional<std::size_t> floor_chunks = 0)
    : scans(lifecycle,
            cached_scan_config([&] {
              if (!floor_chunks) { return cache::config{}.min_prefetching_budget_fraction; }
              if (*floor_chunks == 0) { return 0.0; }
              auto* host       = sirius::scan_test_utils::get_space(*memory, Tier::HOST);
              auto const chunk = host->get_memory_resource_of<Tier::HOST>()->get_block_size();
              return static_cast<double>(*floor_chunks * chunk - chunk / 2) /
                     static_cast<double>(host->get_max_memory());
            }()),
            *memory,
            single_gpu_topology())
  {
  }

  cache::fs_cache& fs_cache()
  {
    auto* result = scans.io_ctx()->cache();
    REQUIRE(result != nullptr);
    REQUIRE(result->is_armed());
    return *result;
  }

  cache::cache_handle prepare(std::size_t first_chunk, std::size_t count)
  {
    auto& cache = fs_cache();
    auto object = scans.io_ctx()->open_io_object(file.path.string());
    std::vector<cucascade::io::byte_range> ranges{
      {static_cast<std::int64_t>(first_chunk * cache.chunk_size()),
       static_cast<std::int64_t>(count * cache.chunk_size())}};
    auto handle = cache::fs_cache_test_access::prepared_reclaim_handle(cache, *object, ranges);
    REQUIRE(handle.chunks() != nullptr);
    REQUIRE(handle.chunks()->size() == count);
    return handle;
  }

  void start(Tier tier                                = Tier::HOST,
             bool bind_cache                          = true,
             std::chrono::milliseconds monitor_period = 0ms)
  {
    auto* space = sirius::scan_test_utils::get_space(*memory, tier);
    REQUIRE(space != nullptr);
    sirius::exec::downgrade_executor_config config{
      .thread_pool    = {.num_threads = 1, .thread_name_prefix = "cache_reclaim"},
      .monitor_period = monitor_period};
    executor = std::make_unique<downgrade_executor>(
      lifecycle, config, repositories, space->get_id(), space, *memory);
    if (bind_cache) { executor->set_fs_cache(&fs_cache()); }
    executor->start();
  }
};

void dispose_and_wait(reclaim_fixture& fixture, cache::cache_handle& handle)
{
  handle.update(cache::scan_stage::disposed);
  // Disposal alone does not wake the evictor to release subscribers. An empty-range
  // fadvise request wakes it without allocating or evicting chunks needed by the test.
  auto wake = fixture.scans.create_datasource(fixture.file.path.string());
  wake->fadvise({}, 0);
  auto const chunks = handle.chunks();
  REQUIRE(chunks != nullptr);
  auto const unsubscribed = [&] {
    return std::ranges::all_of(*chunks,
                               [](auto* chunk) { return chunk->state.get_subscribers() == 0; });
  };
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (!unsubscribed() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(unsubscribed());
}

// Merely sizing this candidate proves the repository path was entered. No converter is
// needed: an idle cache hit must satisfy the request before any repository work is planned.
class counted_host_representation : public cucascade::idata_representation {
 public:
  counted_host_representation(cucascade::memory::memory_space& space,
                              std::atomic<std::size_t>& size_reads)
    : idata_representation(space), size_reads_(size_reads)
  {
  }

  std::size_t get_size_in_bytes() const override
  {
    size_reads_.fetch_add(1);
    return 4096;
  }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return get_size_in_bytes(); }
  std::unique_ptr<cucascade::idata_representation> clone(::cuda::stream_ref) const override
  {
    throw std::logic_error("the repository candidate must not be spilled");
  }

 private:
  std::atomic<std::size_t>& size_reads_;
};

}  // namespace

TEST_CASE("HOST reclaim keeps chunks inside the prefetch reservation",
          "[downgrade_executor][fs_cache]")
{
  reclaim_fixture fixture{8};
  auto idle = fixture.prepare(0, 2);
  auto live = fixture.prepare(2, 2);
  dispose_and_wait(fixture, idle);
  auto& cache = fixture.fs_cache();
  auto* host  = sirius::scan_test_utils::get_space(*fixture.memory, Tier::HOST);
  REQUIRE(host != nullptr);
  auto const available = host->get_available_memory();
  auto const claimed   = cache.claimed_bytes();
  REQUIRE(claimed == 4 * cache.chunk_size());
  REQUIRE(cache.releasable_bytes() == 0);
  fixture.start();

  CHECK(fixture.executor->request_free_memory_and_wait(claimed) == 0);
  CHECK(cache.claimed_bytes() == claimed);
  CHECK(host->get_available_memory() == available);
  CHECK(cache.releasable_bytes() == 0);
  for (auto const* handle : {&idle, &live}) {
    for (auto* chunk : *handle->chunks()) {
      CHECK(chunk->state.get_state() == cache::chunk_state::allocated);
    }
  }
}

TEST_CASE("HOST reclaim credits only bytes above the prefetch reservation",
          "[downgrade_executor][fs_cache]")
{
  reclaim_fixture fixture{2};
  auto idle = fixture.prepare(0, 4);
  dispose_and_wait(fixture, idle);
  auto& cache = fixture.fs_cache();
  auto* host  = sirius::scan_test_utils::get_space(*fixture.memory, Tier::HOST);
  REQUIRE(host != nullptr);
  auto const available = host->get_available_memory();
  auto const chunk     = cache.chunk_size();
  REQUIRE(cache.claimed_bytes() == 4 * chunk);
  REQUIRE(cache.releasable_bytes() == 2 * chunk);
  fixture.start();

  CHECK(fixture.executor->request_free_memory_and_wait(4 * chunk) == 2 * chunk);
  CHECK(cache.claimed_bytes() == 2 * chunk);
  CHECK(host->get_available_memory() - available == 2 * chunk);
  CHECK(fixture.executor->request_free_memory_and_wait(chunk) == 0);
  CHECK(cache.claimed_bytes() == 2 * chunk);
}

TEST_CASE("HOST monitor does not re-fire on a cache it cannot shrink",
          "[downgrade_executor][fs_cache]")
{
  reclaim_fixture fixture{8};
  auto live          = fixture.prepare(0, 4);
  auto& cache        = fixture.fs_cache();
  auto const claimed = cache.claimed_bytes();
  auto* host         = sirius::scan_test_utils::get_space(*fixture.memory, Tier::HOST);
  REQUIRE(host != nullptr);
  auto pressure = host->make_reservation_upto(host->get_max_memory());
  REQUIRE(host->should_downgrade_memory());
  fixture.start(Tier::HOST, true, 1ms);

  std::this_thread::sleep_for(100ms);
  CHECK(fixture.executor->monitor_requests_issued_for_testing() == 0);
  CHECK(cache.claimed_bytes() == claimed);
  pressure.reset();
}

TEST_CASE("HOST monitor reclaims the cache once and then backs off",
          "[downgrade_executor][fs_cache]")
{
  reclaim_fixture fixture{0};
  auto live   = fixture.prepare(0, 4);
  auto& cache = fixture.fs_cache();
  auto* host  = sirius::scan_test_utils::get_space(*fixture.memory, Tier::HOST);
  REQUIRE(host != nullptr);
  auto pressure = host->make_reservation_upto(host->get_max_memory());
  REQUIRE(host->should_downgrade_memory());
  fixture.start(Tier::HOST, true, 1ms);

  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (cache.claimed_bytes() != 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(cache.claimed_bytes() == 0);
  auto const requests = fixture.executor->monitor_requests_issued_for_testing();
  REQUIRE(requests >= 1);
  std::this_thread::sleep_for(50ms);
  CHECK(fixture.executor->monitor_requests_issued_for_testing() == requests);
  pressure.reset();
}

TEST_CASE("HOST downgrade satisfies a target from idle cache before repositories",
          "[downgrade_executor][fs_cache]")
{
  std::atomic<std::size_t> size_reads{0};
  reclaim_fixture fixture;
  auto idle = fixture.prepare(0, 2);
  auto live = fixture.prepare(2, 2);
  dispose_and_wait(fixture, idle);

  auto const query_id = sirius::make_query_id(1);
  fixture.lifecycle.open_query(query_id);
  auto& manager = *fixture.repositories.create_for_query(query_id);
  auto* host    = sirius::scan_test_utils::get_space(*fixture.memory, Tier::HOST);
  REQUIRE(host != nullptr);
  auto batch = cucascade::data_batch::make(
    1, std::make_unique<counted_host_representation>(*host, size_reads));
  auto repository = std::make_unique<cucascade::shared_data_repository>();
  repository->add_data_batch(batch);
  manager.add_new_repository(1, "out", std::move(repository));

  auto& cache        = fixture.fs_cache();
  auto const claimed = cache.claimed_bytes();
  auto const reads   = size_reads.load();
  fixture.start();
  auto const freed = fixture.executor->request_free_memory_and_wait(1);

  CHECK(freed == cache.chunk_size());
  CHECK(cache.claimed_bytes() == claimed - freed);
  CHECK(size_reads.load() == reads);
  CHECK(batch->to_read_only().get_memory_space()->get_tier() == Tier::HOST);
  // Meeting the target in the idle pass must not also force-evict subscribed chunks.
  for (auto* chunk : *live.chunks()) {
    CHECK(chunk->state.get_state() == cache::chunk_state::allocated);
  }
}

TEST_CASE("HOST downgrade force-reclaims only an unmet byte target",
          "[downgrade_executor][fs_cache]")
{
  reclaim_fixture fixture;
  auto live          = fixture.prepare(0, 2);
  auto& cache        = fixture.fs_cache();
  auto const claimed = cache.claimed_bytes();
  fixture.start();

  CHECK(fixture.executor->request_free_memory_and_wait(0) == 0);
  CHECK(fixture.executor->request_downgrade([] { return false; }).get() == 0);
  CHECK(cache.claimed_bytes() == claimed);

  auto const freed = fixture.executor->request_free_memory_and_wait(1);
  CHECK(freed == cache.chunk_size());
  CHECK(cache.claimed_bytes() == claimed - freed);
  // An unattainable target returns only the chunks actually handed back, not its demand.
  CHECK(fixture.executor->request_free_memory_and_wait(claimed * 2) == claimed - freed);
  CHECK(cache.claimed_bytes() == 0);
}

TEST_CASE("GPU downgrade ignores the HOST cache binding", "[downgrade_executor][fs_cache]")
{
  reclaim_fixture fixture;
  auto live          = fixture.prepare(0, 2);
  auto const claimed = fixture.fs_cache().claimed_bytes();
  fixture.start(Tier::GPU);

  CHECK(fixture.executor->request_free_memory_and_wait(claimed) == 0);
  CHECK(fixture.fs_cache().claimed_bytes() == claimed);
}

TEST_CASE("HOST downgrade tolerates no cache and paused rebinding",
          "[downgrade_executor][fs_cache][downgrade_lifecycle]")
{
  reclaim_fixture fixture;
  fixture.start(Tier::HOST, false);
  CHECK(fixture.executor->request_free_memory_and_wait(1) == 0);
  CHECK_THROWS_AS(fixture.executor->set_fs_cache(&fixture.fs_cache()), std::logic_error);

  fixture.executor->pause();
  fixture.executor->pause();
  fixture.executor->set_fs_cache(&fixture.fs_cache());
  fixture.executor->resume();
  fixture.executor->resume();
  auto live = fixture.prepare(0, 1);
  CHECK(fixture.executor->request_free_memory_and_wait(1) == fixture.fs_cache().chunk_size());

  fixture.executor->pause();
  fixture.executor->set_fs_cache(nullptr);
  fixture.executor->stop();
  REQUIRE_NOTHROW(fixture.executor->resume());
  fixture.executor->start();
  CHECK(fixture.executor->request_free_memory_and_wait(1) == 0);
}

TEST_CASE("SiriusContext reset rebinds HOST reclaim to each rebuilt local cache",
          "[downgrade_executor][fs_cache][reset_cache][isolated_context]")
{
  temporary_file config{std::string{R"yaml(sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.05
      reservation_limit_fraction: 0.04
      downgrade_trigger_fraction: 0.04
      downgrade_stop_fraction: 0.02
    host:
      capacity_bytes: 536870912
      reservation_limit_fraction: 0.9
      downgrade_trigger_fraction: 0.8
      downgrade_stop_fraction: 0.7
  executor:
    pipeline:
      num_threads: 1
    downgrade:
      num_threads: 1
      monitor_period: 60000ms
    scan_manager:
      num_threads: 1
      uring_reactors_per_gpu: 1
      cache:
        mode: cucs
        eviction: lru
        min_prefetching_budget_fraction: 0.0
  telemetry:
    enable_quent: false
)yaml"}};
  temporary_file file{16ull << 20};
  auto database = sirius::test::open_sirius_db(nullptr, config.path);
  duckdb::Connection connection(*database);
  auto context = connection.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(context != nullptr);
  duckdb::SiriusContext::SlotGuard maintenance{
    *context, *connection.context, sirius::exec::query_admission::access::maintenance};
  auto& scans = context->get_scan_manager();
  auto hosts  = context->get_memory_manager().get_memory_spaces_for_tier(Tier::HOST);
  REQUIRE_FALSE(hosts.empty());
  auto& executor = context->get_downgrade_executor(hosts.front()->get_id());

  auto populate = [&] {
    auto datasource = scans.create_datasource(file.path.string());
    REQUIRE(datasource != nullptr);
    auto* cache = scans.io_ctx()->cache();
    REQUIRE(cache != nullptr);
    std::vector<cudf::io::text::byte_range_info> ranges{
      {0, static_cast<std::int64_t>(2 * cache->chunk_size())}};
    datasource->fadvise(ranges, 0);
    REQUIRE(datasource->prepare_prefetch(false) == cucascade::io::prepare_result::prepared);
    REQUIRE(cache->claimed_bytes() > 0);
    return datasource;
  };

  // Prime the initial cache too: the first reset must safely discard resident buffers.
  populate().reset();
  for (int round = 0; round < 2; ++round) {
    CAPTURE(round);
    context->reset_caches();
    REQUIRE(scans.io_ctx()->cache() != nullptr);
    CHECK(scans.io_ctx()->cache()->claimed_bytes() == 0);
    auto live          = populate();
    auto const claimed = scans.io_ctx()->cache()->claimed_bytes();
    CHECK(executor.request_free_memory_and_wait(claimed) == claimed);
    CHECK(scans.io_ctx()->cache()->claimed_bytes() == 0);
  }
}
