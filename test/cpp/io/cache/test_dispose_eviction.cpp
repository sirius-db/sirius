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
#include "duckdb/common/exception.hpp"
#include "event/query_event_publisher.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"
#include "io/sirius_datasource.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "planner/query.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

/// Pull a counter from either section of a cache report.
std::uint64_t report_counter(std::string const& summary,
                             std::string const& name,
                             std::string const& section = "global")
{
  std::smatch m;
  auto const re = std::regex(section + "\\[[^\\]]*" + name + "=(\\d+)");
  if (!std::regex_search(summary, m, re)) {
    throw std::runtime_error("Missing cache counter: " + summary);
  }
  return std::stoull(m[1].str());
}

/// Pull the eviction count out of prefetching_cache::summary().
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

/// Poll rather than sleep once: the evictor is a background thread, so a sweep
/// that is merely slow must not read as one that never happened.
std::size_t claimed_after_sweep(sirius::io::cache::prefetching_cache& cache,
                                std::size_t before,
                                std::chrono::milliseconds budget)
{
  auto const deadline = std::chrono::steady_clock::now() + budget;
  while (std::chrono::steady_clock::now() < deadline) {
    auto const now = cache.claimed_bytes();
    if (now < before) { return now; }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return cache.claimed_bytes();
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

// Insert real cache activity while the scan manager emits its INFO report. This
// exercises the old gap between that report and the baseline update, without
// adding synchronization hooks or counter setters to the production cache.
class read_during_summary_log final : public sirius::log::sink {
 public:
  read_during_summary_log(std::shared_ptr<sirius::log::sink> delegate, std::function<void()> read)
    : delegate_(std::move(delegate)), read_(std::move(read))
  {
  }
  void set_level(sirius::log::level level) override { delegate_->set_level(level); }
  bool should_log(sirius::log::level level) const override { return delegate_->should_log(level); }
  bool flush() override { return delegate_->flush(); }
  void log(sirius::log::level level,
           std::source_location const& location,
           std::string_view message) override
  {
    delegate_->log(level, location, message);
    if (message.starts_with("[sirius_scan_manager] cache summary:") && !read_done_.exchange(true))
      read_();
  }

 private:
  std::shared_ptr<sirius::log::sink> delegate_;
  std::function<void()> read_;
  std::atomic<bool> read_done_{false};
};

}  // namespace

TEST_CASE("dispose_on_idle reclaims a disposed request's chunks on the next sweep",
          "[cache][eviction][dispose]")
{
  temp_data_file file(8ull << 20);  // 8 MiB
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_dispose();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, dispose_on_idle_config(), *memory, topology};
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
    // Nothing does this on its own -- fadvise only registers the request and the
    // chunks it names -- so without this call there is no buffer to reclaim and
    // the evictor correctly evicts nothing.
    REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  }  // ~sirius_datasource -> ~cache_handle -> the consumer is disposed

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

TEST_CASE("dispose_on_idle keeps a chunk a live request still shares", "[cache][eviction][dispose]")
{
  constexpr std::size_t file_bytes = 32ull << 20;
  temp_data_file file(file_bytes);
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index_for_dispose();

  sirius::exec::query_lifecycle_registry scan_lifecycle;

  sirius_scan_manager manager{scan_lifecycle, dispose_on_idle_config(), *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());

  // A single prepared byte measures the cache's chunk size, so the ranges below
  // can be laid out in whole chunks whatever the host allocator's block size
  // is.  The probe stays alive for the rest of the test: its chunk keeps a
  // subscriber, so no sweep may touch it and it is a constant in every total.
  auto probe = manager.create_datasource(file.path.string());
  REQUIRE(probe != nullptr);
  std::vector<cudf::io::text::byte_range_info> probe_range;
  probe_range.emplace_back(0, 1);
  probe->fadvise(probe_range, 0);
  REQUIRE(probe->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  auto const chunk = cache->claimed_bytes();
  REQUIRE(chunk > 0);
  REQUIRE(6 * chunk <= file_bytes);

  // Two requests on the same file, overlapping on exactly one chunk -- what
  // adjacent parquet splits of one file do to their boundary chunk.
  std::vector<cudf::io::text::byte_range_info> first;
  first.emplace_back(static_cast<std::int64_t>(chunk), static_cast<std::int64_t>(2 * chunk));
  auto ds1 = manager.create_datasource(file.path.string());
  REQUIRE(ds1 != nullptr);
  ds1->fadvise(first, 0);
  REQUIRE(ds1->prepare_prefetch(false) == sirius::io::prepare_result::prepared);

  std::vector<cudf::io::text::byte_range_info> second;
  second.emplace_back(static_cast<std::int64_t>(2 * chunk), static_cast<std::int64_t>(2 * chunk));
  auto ds2 = manager.create_datasource(file.path.string());
  REQUIRE(ds2 != nullptr);
  ds2->fadvise(second, 0);
  REQUIRE(ds2->prepare_prefetch(false) == sirius::io::prepare_result::prepared);

  // Probe plus chunks 1, 2 and 3 -- the shared chunk 2 is claimed once.
  REQUIRE(cache->claimed_bytes() == 4 * chunk);

  ds1.reset();  // ~sirius_datasource -> ~cache_handle -> the consumer is disposed

  // The evictor only sweeps when a request reaches it, so a fresh insert is
  // what gives the disposed request its sweep.  fadvise alone claims no
  // buffers, so this one does not move the totals.
  std::vector<cudf::io::text::byte_range_info> untouched;
  untouched.emplace_back(static_cast<std::int64_t>(5 * chunk), static_cast<std::int64_t>(chunk));
  auto ds3 = manager.create_datasource(file.path.string());
  REQUIRE(ds3 != nullptr);
  ds3->fadvise(untouched, 0);

  // Only chunk 1 went idle: chunk 2 is still named by ds2, and `idle` reclaims
  // a chunk when it is idle -- not when one of several subscribers lets go.
  INFO("cache: " << cache->summary());
  CHECK(claimed_after_sweep(*cache, 4 * chunk, std::chrono::milliseconds(2000)) == 3 * chunk);
}

TEST_CASE("scan inputs populate the cache without a readahead manager",
          "[cache][scan][no_readahead]")
{
  temp_data_file file(8ull << 20);
  auto memory             = initialize_memory_manager(1);
  auto topology           = single_gpu_index_for_dispose();
  auto cfg                = dispose_on_idle_config();
  cfg.cache.eviction      = sirius::io::cache::eviction_policy::lru;
  cfg.max_readahead_scans = 0;
  cfg.apply_cache_mode();
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, cfg, *memory, topology};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());

  auto make_input = [&]() {
    auto ds = manager.create_datasource(file.path.string());
    std::vector<sirius::op::scan::scan_info::fadvise_entry> hints;
    hints.push_back({ds, {cudf::io::text::byte_range_info{0, 4096}}});
    auto info = std::make_shared<sirius::op::scan::scan_info>(std::move(hints));
    return std::pair{ds, std::make_unique<sirius::op::scan::scan_operator_input>(info)};
  };

  auto [first_ds, first_input] = make_input();
  // A later readahead preparation of the same handle is idempotent.
  REQUIRE(first_ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  std::vector<std::uint8_t> bytes(4096);
  REQUIRE(first_ds->host_read(0, bytes.size(), bytes.data()) == bytes.size());
  CHECK(bytes.front() == static_cast<std::uint8_t>('z'));
  INFO(cache->summary());
  CHECK(report_counter(cache->summary(), "h2d") > 0);

  first_input.reset();
  auto [second_ds, second_input] = make_input();
  REQUIRE(second_ds->host_read(0, bytes.size(), bytes.data()) == bytes.size());
  INFO(cache->summary());
  CHECK(report_counter(cache->summary(), "hits") > 0);
}

TEST_CASE("cache refresh reports each increment once across overlapping refreshes",
          "[cache][cache_summary]")
{
  temp_data_file file(8ull << 20);
  auto memory             = initialize_memory_manager(1);
  auto cfg                = dispose_on_idle_config();
  cfg.cache.eviction      = sirius::io::cache::eviction_policy::lru;
  cfg.max_readahead_scans = 0;
  cfg.apply_cache_mode();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager manager{lifecycle, cfg, *memory, single_gpu_index_for_dispose()};
  // The default cache is built on first use.
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache);
  REQUIRE(cache->is_armed());
  auto ds = manager.create_datasource(file.path.string());
  std::array ranges{cudf::io::text::byte_range_info{0, 4096}};
  ds->fadvise(ranges, 0);
  REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  std::vector<std::uint8_t> bytes(4096);
  auto read = [&] {
    if (ds->host_read(0, bytes.size(), bytes.data()) != bytes.size())
      throw std::runtime_error("Incomplete cache read");
  };
  read();  // Populate the chunk; subsequent reads count as hits.
  auto const before = cache->prepare_for_query();
  std::vector<std::string> reports;
  constexpr std::array names{"reads", "hits", "h2d", "miss", "evictions"};

  SECTION("diagnostic observations do not consume a reporting interval")
  {
    read();
    auto const observed = cache->summary();
    CHECK(cache->summary() == observed);
    auto const epoch = cache->query_epoch();
    reports.push_back(cache->prepare_for_query());
    CHECK(reports.back() == observed);
    CHECK(cache->query_epoch() == epoch + 1);
    reports.push_back(cache->prepare_for_query());
    for (auto const* name : names)
      CHECK(report_counter(reports.back(), name, "last_cycle") == 0);
    read();
  }
  SECTION("two refreshers race with actual cache reads")
  {
    // Seed a nonzero interval even if the refreshers finish before the read loop.
    read();
    std::barrier start{3};
    std::array<std::vector<std::string>, 2> concurrent_reports;
    std::array<std::exception_ptr, 2> errors;
    auto refresh = [&](std::size_t index) {
      start.arrive_and_wait();
      try {
        for (int i = 0; i < 100; ++i)
          concurrent_reports[index].push_back(cache->prepare_for_query());
      } catch (...) {
        errors[index] = std::current_exception();
      }
    };
    {
      std::jthread first(refresh, 0), second(refresh, 1);
      start.arrive_and_wait();
      for (int i = 0; i < 100; ++i)
        read();
    }
    for (std::size_t i = 0; i < errors.size(); ++i) {
      REQUIRE_FALSE(errors[i]);
      reports.insert(reports.end(), concurrent_reports[i].begin(), concurrent_reports[i].end());
    }
  }
  SECTION("scan manager logs the sample it closes and preserves activity during logging")
  {
    read();
    auto tctx           = sirius::test::make_test_telemetry_context();
    auto const query_id = sirius::make_query_id(7);
    sirius::telemetry::query_telemetry_info tinfo{
      quent::now_v7(), tctx->worker_id().raw(), query_id};
    sirius::planner::query empty{std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>>{},
                                 tctx->context(),
                                 query_id,
                                 tinfo};
    sirius::test::scoped_recording_log_sink logs;
    sirius::log::set_sink(std::make_shared<read_during_summary_log>(logs.sink_ptr(), read));
    manager.prepare_for_query(empty, true, {});
    std::string info_report, trace_report;
    for (auto const& record : logs.records()) {
      auto const pos = record.message.find("prefetching_cache: global[");
      if (pos == std::string::npos) continue;
      if (record.level == sirius::log::level::info) info_report = record.message.substr(pos);
      if (record.level == sirius::log::level::trace) trace_report = record.message.substr(pos);
    }
    REQUIRE_FALSE(info_report.empty());
    CHECK(info_report == trace_report);
    CHECK(report_counter(info_report, "hits", "last_cycle") == 1);
    reports.push_back(info_report);
    // The read injected by the INFO sink belongs to the next interval.
    CHECK(report_counter(cache->summary(), "hits", "last_cycle") == 1);
  }

  reports.push_back(cache->prepare_for_query());
  auto const after = cache->summary();
  CHECK(report_counter(after, "hits") > report_counter(before, "hits"));
  for (auto const* name : names) {
    INFO(name);
    std::uint64_t reported = 0;
    for (auto const& report : reports)
      reported += report_counter(report, name, "last_cycle");
    CHECK(reported == report_counter(after, name) - report_counter(before, name));
  }
}

TEST_CASE("Deferred scan inputs initialize prefetch only when admitted",
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
    scan_lifecycle, dispose_on_idle_config(), *memory, single_gpu_index_for_dispose()};
  auto* cache = manager.ensure_default_cache();
  REQUIRE(cache != nullptr);
  auto ds = manager.create_datasource(file.path.string());
  std::vector<scan_info::fadvise_entry> hints{{ds, {cudf::io::text::byte_range_info{0, 4096}}}};
  auto info      = std::make_shared<scan_info>(hints);
  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  // No operator is registered: trying to enqueue readahead must fail.
  auto readahead = std::make_shared<sirius::scan_manager::readahead_scan_manager>(*publisher, 1);
  bool fail_registration = false, discard = false, cached_cancel = false;
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
    CHECK(ds->prepare_prefetch(false) == sirius::io::prepare_result::nothing_to_prepare);
    return;
  }
  auto input = std::make_unique<scan_operator_input>(
    info, readahead, 0, 0, scan_operator_input::prefetch_start::deferred);
  CHECK(cache->generation_count(file.path.string()) == 0);
  CHECK(cache->claimed_bytes() == 0);
  CHECK(ds->prepare_prefetch(false) == sirius::io::prepare_result::nothing_to_prepare);
  if (!discard) {
    if (fail_registration) {
      CHECK_THROWS_AS(input->initialize_prefetch(), std::out_of_range);
    } else {
      REQUIRE_NOTHROW(input->initialize_prefetch());
      CHECK(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
    }
    CHECK(cache->generation_count(file.path.string()) == 1);
    CHECK(cache->claimed_bytes() > 0);
    CHECK_THROWS_AS(input->initialize_prefetch(), std::logic_error);
  }
  input.reset();
  CHECK(info->get_scan_stage() == sirius::io::cache::scan_stage::disposed);
  if (discard) {
    CHECK(cache->generation_count(file.path.string()) == 0);
    CHECK(cache->claimed_bytes() == 0);
  } else {
    CHECK(ds->prepare_prefetch(false) == sirius::io::prepare_result::fallen_behind);
  }
}
