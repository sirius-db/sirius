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
#include "duckdb/common/exception.hpp"
#include "event/query_event_publisher.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"
#include "scan_manager/preparation.hpp"
#include "scan_manager/preparation_fixture.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/io_context.hpp>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index_for_prefetch()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

/// A real on-disk file, so the uring backend opens it and the cache sizes its
/// slot table from a genuine object size.
struct temp_data_file {
  std::filesystem::path path;

  explicit temp_data_file(std::size_t bytes)
  {
    path = std::filesystem::temp_directory_path() /
           ("sirius_scan_input_prefetch_" + std::to_string(::getpid()) + "_" +
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

  temp_data_file(temp_data_file const&)            = delete;
  temp_data_file& operator=(temp_data_file const&) = delete;
};

/// `cache.mode: cucs` with `cache.eviction: idle`.
scan_manager_config dispose_on_idle_config()
{
  scan_manager_config cfg;
  cfg.thread_pool.num_threads = 3;
  cfg.uring_reactors_per_gpu  = 1;
  cfg.cache.mode              = cucascade::io::cache::cache_mode::cucs;
  cfg.cache.eviction          = cucascade::io::cache::eviction_policy::idle;
  cfg.apply_cache_mode();
  return cfg;
}

/// Pull `global[... evictions=N ...]` out of fs_cache::summary().
std::uint64_t evictions_from(std::string const& summary)
{
  std::smatch m;
  static std::regex const re(R"(global\[[^\]]*evictions=(\d+))");
  if (!std::regex_search(summary, m, re)) { return 0; }
  return std::stoull(m[1].str());
}

/// Poll until claimed_bytes() drops below @p before; the evictor is a background thread.
std::size_t claimed_after_sweep(cucascade::io::cache::fs_cache& cache,
                                std::size_t before,
                                std::chrono::milliseconds budget)
{
  // The evictor may release the disposed request's chunks over several passes:
  // keep polling until everything is reclaimed or the budget runs out, and
  // return the lowest value seen so a partial sweep is still reported.
  auto const deadline = std::chrono::steady_clock::now() + budget;
  auto lowest         = before;
  while (std::chrono::steady_clock::now() < deadline) {
    auto const now = cache.claimed_bytes();
    lowest         = std::min(lowest, now);
    if (now == 0) { return 0; }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return std::min(lowest, cache.claimed_bytes());
}

}  // namespace

TEST_CASE("Deferred scan inputs dispose cache requests when publication is rejected",
          "[cache][scan][scan_preparation]")
{
  using sirius::op::scan::scan_info;
  using sirius::op::scan::scan_operator_input;
  struct interrupting_info : scan_info {
    using scan_info::scan_info;
    bool* interrupted = nullptr;
    std::size_t estimated_bytes() const noexcept override
    {
      if (interrupted) *interrupted = true;
      return 4096;
    }
  };
  temp_data_file file(8ull << 20);
  auto memory = initialize_memory_manager(1);
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle, dispose_on_idle_config(), *memory, single_gpu_index_for_prefetch()};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  auto ds = manager.create_datasource(file.path.string());
  std::vector<scan_info::fadvise_entry> hints{{ds, {cudf::io::text::byte_range_info{0, 4096}}}};
  auto info      = std::make_shared<scan_info>(hints);
  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  // No operator is registered: trying to enqueue readahead must fail.
  auto readahead = std::make_shared<sirius::scan_manager::readahead_scan_manager>(*publisher, 1);
  bool fail_registration = false, discard = false, cached_cancel = false;
  int coordinator_rejection = -1;
  SECTION("cached worker observes interruption after sizing an insert-delta input")
  {
    cached_cancel = true;
  }
  SECTION("cancelled construction leaves no cache request") { discard = true; }
  SECTION("admitted initialization prepares the buffers") { readahead.reset(); }
  SECTION("failed registration disposes a partially initialized request")
  {
    fail_registration = true;
  }
  SECTION("coordinator rejects prepared cache requests without publishing readahead")
  {
    // Normal stop, user cancellation, and failure after the buffers are attached.
    coordinator_rejection = GENERATE(0, 1, 2);
  }
  if (coordinator_rejection >= 0) {
    using namespace sirius::scan_manager;
    using namespace sirius::test::preparation;
    auto neighbor = manager.create_datasource(file.path.string());
    neighbor->fadvise(hints.front().ranges, 0);
    REQUIRE(neighbor->prepare_prefetch(false) == cucascade::io::prepare_result::prepared);
    auto const chunk = cache->claimed_bytes();
    REQUIRE(chunk > 0);
    REQUIRE(2 * chunk <= (8ull << 20));
    auto exclusive = manager.create_datasource(file.path.string());
    hints.push_back({exclusive, {cudf::io::text::byte_range_info{static_cast<int64_t>(chunk), 1}}});
    info = std::make_shared<scan_info>(hints);
    fixture f;
    auto out         = std::make_shared<sink>();
    auto source      = make_source(preparation_coordinator::execution_domain::metadata_worker);
    source.try_claim = single_job([] { return std::make_unique<tagged>(1); });
    source.construct = [&](auto) {
      return preparation_coordinator::publication{
        std::make_unique<scan_operator_input>(
          info, readahead, 0, 0, scan_operator_input::prefetch_start::deferred),
        4096};
    };
    source.close = [out] { out->close(); };
    f.register_source(std::move(source), out);
    size_t prepared_bytes = 0;
    f.coordinator.before_publication_for_testing([&] {
      prepared_bytes = cache->claimed_bytes();
      if (coordinator_rejection == 2) throw std::runtime_error("prepared publication failed");
      auto reason = coordinator_rejection == 0 ? stop_reason::normal_eos : stop_reason::user_cancel;
      f.coordinator.request_stop(reason);
      f.coordinator.request_stop(reason);
    });
    auto guard = f.shutdown_guard();
    f.start();
    f.finish();
    CHECK(prepared_bytes == 2 * chunk);
    CHECK(f.completion.has_error() == (coordinator_rejection != 0));
    if (coordinator_rejection == 2)
      CHECK(f.completion.failure().detail == "prepared publication failed");
    CHECK(out->count() == 0);
    CHECK(out->close_calls == 1);
    CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
    CHECK(info->get_scan_stage() == cucascade::io::cache::scan_stage::disposed);
    for (auto const& rejected : {ds, exclusive}) {
      rejected->update(cucascade::io::cache::scan_stage::disposed);
      CHECK(rejected->prepare_prefetch(false) == cucascade::io::prepare_result::fallen_behind);
    }
    // A cache sweep is separate from query drain. Repeated disposal cannot drop
    // the live neighbor's subscription or return its shared buffer twice.
    cache->evict_sync(chunk);
    CHECK(cache->claimed_bytes() == chunk);
    cache->evict_sync(chunk);
    CHECK(cache->claimed_bytes() == chunk);
    neighbor.reset();
    cache->evict_sync(chunk);
    CHECK(cache->claimed_bytes() == 0);
    cache->evict_sync(chunk);
    CHECK(cache->claimed_bytes() == 0);
    return;
  }
  if (cached_cancel) {
    using namespace sirius::scan_manager;
    struct single_input : databatch_provider {
      batch next;
      batch get_next_batch() override { return std::exchange(next, {}); }
    } provider;
    bool interrupted        = false;
    auto delta              = std::make_unique<interrupting_info>(hints);
    delta->interrupted      = &interrupted;
    provider.next.scan_info = std::move(delta);
    auto gate               = std::make_shared<preparation_gate>();
    gate->interrupted       = [&] { return interrupted; };
    split_connector connector;
    std::stop_source stop;
    load_balancing_scan_batch_coalescer::drain_cached_provider(
      provider, connector, stop.get_token(), false, 0, 0, nullptr, false, false, nullptr, 0, gate);
    CHECK_THROWS_AS(connector.get_next_split(), duckdb::InterruptException);
    CHECK(gate->closed);
    CHECK(cache->generation_count(file.path.string()) == 0);
    CHECK(cache->claimed_bytes() == 0);
    CHECK(ds->prepare_prefetch(false) == cucascade::io::prepare_result::nothing_to_prepare);
    return;
  }
  auto input = std::make_unique<scan_operator_input>(
    info, readahead, 0, 0, scan_operator_input::prefetch_start::deferred);
  CHECK(cache->generation_count(file.path.string()) == 0);
  CHECK(cache->claimed_bytes() == 0);
  CHECK(ds->prepare_prefetch(false) == cucascade::io::prepare_result::nothing_to_prepare);
  if (!discard) {
    if (fail_registration) {
      CHECK_THROWS_AS(input->initialize_prefetch(), std::out_of_range);
    } else {
      REQUIRE_NOTHROW(input->initialize_prefetch());
      CHECK(ds->prepare_prefetch(false) == cucascade::io::prepare_result::prepared);
    }
    CHECK(cache->generation_count(file.path.string()) == 1);
    CHECK(cache->claimed_bytes() > 0);
    CHECK_THROWS_AS(input->initialize_prefetch(), std::logic_error);
  }
  input.reset();
  CHECK(info->get_scan_stage() == cucascade::io::cache::scan_stage::disposed);
  if (discard) {
    CHECK(cache->generation_count(file.path.string()) == 0);
    CHECK(cache->claimed_bytes() == 0);
  } else {
    CHECK(ds->prepare_prefetch(false) == cucascade::io::prepare_result::fallen_behind);
  }
}

TEST_CASE("cache.eviction idle reclaims a disposed scan input's chunks",
          "[cache][scan][eviction][dispose]")
{
  temp_data_file file(8ull << 20);
  auto memory = initialize_memory_manager(1);
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle, dispose_on_idle_config(), *memory, single_gpu_index_for_prefetch()};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());
  REQUIRE(cache->claimed_bytes() == 0);

  std::size_t claimed = 0;
  {
    auto ds = manager.create_datasource(file.path.string());
    REQUIRE(ds != nullptr);
    std::vector<sirius::op::scan::scan_info::fadvise_entry> hints;
    hints.push_back({ds, {cudf::io::text::byte_range_info{0, 4ll << 20}}});
    auto info  = std::make_shared<sirius::op::scan::scan_info>(std::move(hints));
    auto input = std::make_unique<sirius::op::scan::scan_operator_input>(info);
    claimed    = cache->claimed_bytes();
    REQUIRE(claimed > 0);
    input.reset();
    CHECK(info->get_scan_stage() == cucascade::io::cache::scan_stage::disposed);
  }

  // The evictor sweeps when a request reaches it; this insert claims no buffers of its own.
  auto other = manager.create_datasource(file.path.string());
  REQUIRE(other != nullptr);
  std::vector<cudf::io::text::byte_range_info> untouched;
  untouched.emplace_back(6ll << 20, 1ll << 20);
  other->fadvise(untouched, 0);

  INFO("cache: " << cache->summary());
  // Primary assertion: the disposed input's chunks were evicted. The claimed
  // total must drop; it reaches zero unless `other`'s hint shares a chunk with
  // the disposed range (chunk size is a host-pool property, not fixed here).
  auto const after = claimed_after_sweep(*cache, claimed, std::chrono::milliseconds(5000));
  CHECK(evictions_from(cache->summary()) > 0);
  CHECK(after < claimed);
}
