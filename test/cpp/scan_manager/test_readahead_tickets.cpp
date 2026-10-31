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

// The readahead's ticket protocol: one owner of a split's two credits (the per-query gatekeeper
// slot and the runtime-wide shared_scan_budget token), which a demand read joins instead of
// charging a second ticket, and which settles once the prefetch's IO settled.  Driven against a
// real fs_cache over a backend whose IO the test holds and completes by hand
// (utils/controlled_io.hpp), plus the bounded memory retry against real cache pressure.
//
// Re-homed from test/cpp/io/cache/test_cache_read_arbitration.cpp and
// test_explicit_eviction.cpp; their cache-only halves now live in cuCascade.

#include "catch.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "query_id.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "scan_manager/shared_scan_budget.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/controlled_io.hpp"
#include "utils/readahead_test_utils.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace {

namespace cio = sirius::test::controlled_io;
using cio::chunk_size;
using cio::complete_failure;
using cio::complete_success;
using cio::held_request;
using sirius::scan_manager::prefetch_outcome_kind;
using sirius::scan_manager::readahead_scan_manager;
using sirius::scan_manager::shared_scan_budget;

constexpr std::size_t read_size = 4096;

/// A cached controlled backend: `cache.mode: cucs` over a held reactor.
struct cache_fixture {
  explicit cache_fixture(cio::controlled_reactor_config backend = {})
    : memory(initialize_memory_manager(1)),
      context(std::make_shared<cio::controlled_ioctx>(backend))
  {
    cucascade::io::cache::config config;
    config.mode = cucascade::io::cache::cache_mode::cucs;
    config.apply_mode();
    context->start();
    context->initialize_cache(*memory, config, sirius::test::single_gpu_index_for_prefetch());
    REQUIRE(context->cache() != nullptr);
    REQUIRE(context->cache()->chunk_size() == chunk_size);
  }

  ~cache_fixture()
  {
    context->shutdown();
    context->shutdown_cache();
  }

  cache_fixture(cache_fixture const&)            = delete;
  cache_fixture& operator=(cache_fixture const&) = delete;

  [[nodiscard]] cio::controlled_reactor& reactor() noexcept { return context->reactor(); }
  [[nodiscard]] std::uintptr_t backend() const noexcept
  {
    return reinterpret_cast<std::uintptr_t>(context.get());
  }

  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory;
  std::shared_ptr<cio::controlled_ioctx> context;
};

bool await_condition(auto predicate, std::chrono::milliseconds timeout = 2s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  return predicate();
}

/// One query's readahead over one scan operator (operator id 1), with a budget of one slot and
/// a shared-budget limit of one per backend unless told otherwise.
struct readahead_query {
  using scan_info = sirius::op::scan::scan_info;
  using stage     = cucascade::io::cache::scan_stage;
  using outcome   = prefetch_outcome_kind;

  readahead_query(std::shared_ptr<shared_scan_budget> budget, int id)
    : manager(std::make_shared<readahead_scan_manager>(*publisher, 1, std::move(budget), 1))
  {
    sirius::test::test_scan op;
    auto telemetry = sirius::test::make_test_telemetry_context();
    sirius::pipeline::pipeline_build_context ctx{nullptr, true};
    auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(ctx);
    sirius::pipeline::sirius_pipeline_build_state build;
    build.set_pipeline_source(*pipeline, op);
    auto query_id = sirius::make_query_id(id);
    sirius::planner::query query({pipeline},
                                 telemetry->context(),
                                 query_id,
                                 sirius::telemetry::query_telemetry_info{
                                   quent::now_v7(), telemetry->worker_id().raw(), query_id});
    manager->prepare_for_query(query);
  }

  /// What a scan input does: publish the stage, tell the cache handles, then the readahead.
  void update(std::shared_ptr<scan_info> const& split, stage value)
  {
    split->set_scan_stage(value);
    for (auto const& ds : split->datasources())
      ds->update(value);
    manager->update_scan_state(1, split.get(), value);
  }

  [[nodiscard]] static std::shared_ptr<cucascade::io::datasource> open(
    std::shared_ptr<cio::controlled_ioctx> const& context, std::string path)
  {
    return cucascade::io::open_datasource(context, std::move(path));
  }

  std::shared_ptr<scan_info> add_split(std::shared_ptr<cio::controlled_ioctx> const& context,
                                       std::string path,
                                       bool partial = false)
  {
    auto ds = open(context, std::move(path));
    std::vector<cudf::io::text::byte_range_info> ranges{{0, read_size}};
    ds->fadvise(ranges, 0);
    std::vector<scan_info::fadvise_entry> entries{{ds, ranges}};
    // A datasource without a cache request refuses prefetch, leaving demand
    // responsible for that part even when the first datasource succeeds.
    if (partial) { entries.push_back({open(context, "controlled://unadvised"), ranges}); }
    auto split = std::make_shared<scan_info>(std::move(entries));
    manager->register_scan_task(split, 1);
    return split;
  }

  [[nodiscard]] std::uint64_t outcomes(outcome kind) const
  {
    return manager->counters().outcome(kind);
  }

  std::shared_ptr<sirius::event::query_event_publisher> publisher =
    std::make_shared<sirius::event::query_event_publisher>();
  std::shared_ptr<readahead_scan_manager> manager;
};

}  // namespace

// ===========================================================================
// host tier (upstream #2006)
// ===========================================================================

TEST_CASE("demand joining prefetch issue spends one ticket and completion frees both budgets",
          "[cache][readahead][scan-tickets]")
{
  cache_fixture fixture;
  auto budget  = std::make_shared<shared_scan_budget>();
  auto backend = fixture.backend();
  readahead_query query(budget, 1);
  auto split = query.add_split(fixture.context, "controlled://overlap");
  std::promise<void> resume;
  auto resumed                    = resume.get_future().share();
  auto first                      = std::make_shared<std::atomic<bool>>(true);
  auto pause_timed_out            = std::make_shared<std::atomic<bool>>(false);
  fixture.reactor().after_enqueue = [first, resumed, pause_timed_out] {
    if (first->exchange(false))
      *pause_timed_out = resumed.wait_for(5s) != std::future_status::ready;
  };
  query.manager->start();
  held_request io{fixture.reactor().take_next()};
  // The issuing thread has not returned from enqueue, so scan_info::prefetch
  // has not finished issuing/publishing its outcome. Demand races that window.
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 1);
  CHECK(query.manager->counters().cold_read_tickets == 0);
  CHECK(query.manager->counters().borrowed == 0);
  resume.set_value();
  REQUIRE(io);
  complete_success(*io);
  REQUIRE(await_condition(
    [&] { return query.outcomes(readahead_query::outcome::wait_for_prefetch) == 1; }));
  CHECK_FALSE(pause_timed_out->load());
  CHECK(budget->active(backend) == 0);
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 0);

  // A younger query can use the shared backend while the first split remains
  // undisposed. This is the regression: the old demand charge survived here.
  readahead_query other(budget, 2);
  auto peer = other.add_split(fixture.context, "controlled://peer");
  other.manager->start();
  held_request peer_io{fixture.reactor().take_next()};
  REQUIRE(peer_io);
  complete_success(*peer_io);
  REQUIRE(await_condition([&] { return budget->active(backend) == 0; }));
  other.manager->stop();

  // The first query's local credit was also returned, exactly once. With one
  // new prefetch held, another candidate must wait at the local gate.
  auto next = query.add_split(fixture.context, "controlled://next");
  auto last = query.add_split(fixture.context, "controlled://last");
  held_request next_io{fixture.reactor().take_next()};
  REQUIRE(next_io);
  auto timeouts = query.manager->counters().gate_timeouts.load();
  CHECK(await_condition([&] { return query.manager->counters().gate_timeouts > timeouts; }));
  CHECK(budget->pending() == 0);
  CHECK(fixture.reactor().pending() == 0);
  complete_success(*next_io);
  held_request last_io{fixture.reactor().take_next()};
  REQUIRE(last_io);
  complete_success(*last_io);
  query.update(split, readahead_query::stage::disposed);
  query.update(split, readahead_query::stage::disposed);
  CHECK(await_condition([&] { return budget->active(backend) == 0; }));
  query.manager->stop();
  fixture.reactor().after_enqueue = {};
}

TEST_CASE("prefetch completing a subset retains the ticket for a demand-owned chunk",
          "[cache][readahead][scan-tickets][prefetch-coverage]")
{
  cache_fixture fixture;
  auto budget      = std::make_shared<shared_scan_budget>();
  auto backend     = fixture.backend();
  bool device_read = false;
  SECTION("host demand") {}
  SECTION("device demand") { device_read = true; }

  std::array<std::uint8_t, read_size> destination{};
  rmm::cuda_stream stream;
  ::cuda::stream_ref const stream_ref{stream};
  rmm::device_buffer device_destination(device_read ? read_size : 0, stream_ref);
  std::future<std::size_t> read;
  std::atomic<bool> first{true};
  readahead_query query(budget, 1);
  readahead_query peer(budget, 2);
  std::shared_ptr<cucascade::io::datasource> ds =
    readahead_query::open(fixture.context, "controlled://split-coverage");
  std::vector<cudf::io::text::byte_range_info> ranges{
    {0, read_size}, {static_cast<std::int64_t>(chunk_size), static_cast<std::int64_t>(read_size)}};
  ds->fadvise(ranges, 0);
  auto split = std::make_shared<readahead_query::scan_info>(
    std::vector<readahead_query::scan_info::fadvise_entry>{{ds, ranges}});
  query.manager->register_scan_task(split, 1);

  // Run demand at the claim boundary, after producer loading is visible but
  // before any chunk is loading. Its wait check passes and it claims only A.
  // Backend requests stay pending until this test explicitly completes them.
  fixture.context->cache()->set_before_prefetch_claim_for_testing([&] {
    if (!first.exchange(false)) return;
    query.update(split, readahead_query::stage::reading);
    read = device_read
             ? ds->device_read_async(
                 0, read_size, static_cast<std::uint8_t*>(device_destination.data()), stream_ref)
             : ds->host_read_async(0, read_size, destination.data());
  });
  query.manager->start();
  held_request demand_io{fixture.reactor().take_next()};
  held_request prefetch_io{fixture.reactor().take_next()};
  REQUIRE(demand_io);
  REQUIRE(prefetch_io);
  CHECK(budget->active(backend) == 1);
  CHECK(query.manager->counters().cold_read_tickets == 0);
  CHECK(query.manager->counters().borrowed == 0);

  // Prefetch skipped A and issued only B. Completing B must not free A's slot.
  complete_success(*prefetch_io);
  CHECK(await_condition(
    [&] { return query.outcomes(readahead_query::outcome::wait_for_prefetch) == 1; }));
  CHECK(read.wait_for(0ms) == std::future_status::timeout);
  CHECK(budget->active(backend) == 1);
  auto peer_split = peer.add_split(fixture.context, "controlled://coverage-peer");
  peer.manager->start();
  CHECK(await_condition([&] { return budget->pending() == 1; }));
  CHECK(fixture.reactor().pending() == 0);

  complete_success(*demand_io);
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == read_size);
  if (device_read) {
    std::vector<std::uint8_t> copied(read_size);
    REQUIRE(
      cudaMemcpy(copied.data(), device_destination.data(), read_size, cudaMemcpyDeviceToHost) ==
      cudaSuccess);
    CHECK(std::ranges::all_of(copied, [](auto b) { return b == 0x5a; }));
  } else {
    CHECK(std::ranges::all_of(destination, [](auto b) { return b == 0x5a; }));
  }
  CHECK(budget->active(backend) == 1);
  query.update(split, readahead_query::stage::disposed);
  query.update(split, readahead_query::stage::disposed);
  held_request peer_io{fixture.reactor().take_next()};
  REQUIRE(peer_io);
  CHECK(budget->active(backend) == 1);
  complete_success(*peer_io);
  CHECK(await_condition([&] { return budget->active(backend) == 0; }));
  query.manager->stop();
  peer.manager->stop();
  fixture.context->cache()->set_before_prefetch_claim_for_testing({});
  stream.synchronize();
}

TEST_CASE("incomplete prefetch transfers its ticket to demand until disposal",
          "[cache][readahead][scan-tickets]")
{
  cache_fixture fixture;
  auto budget  = std::make_shared<shared_scan_budget>();
  auto backend = fixture.backend();
  readahead_query query(budget, 1);
  bool partial                  = false;
  bool demand_before_completion = true;
  SECTION("IO failure with overlapping demand") {}
  SECTION("partial coverage with overlapping demand") { partial = true; }
  SECTION("IO failure before demand") { demand_before_completion = false; }
  SECTION("partial coverage before demand")
  {
    partial                  = true;
    demand_before_completion = false;
  }
  auto split = query.add_split(fixture.context, "controlled://incomplete", partial);
  query.manager->start();
  held_request io{fixture.reactor().take_next()};
  REQUIRE(io);
  if (demand_before_completion) query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 1);
  if (partial)
    complete_success(*io);
  else
    complete_failure(*io);
  auto expected = demand_before_completion ? readahead_query::outcome::wait_for_prefetch
                                           : readahead_query::outcome::prefetched;
  REQUIRE(await_condition([&] { return query.outcomes(expected) == 1; }));
  CHECK(budget->active(backend) == (demand_before_completion ? 1 : 0));
  query.update(split, readahead_query::stage::reading);
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 1);
  CHECK(query.manager->counters().cold_read_tickets == (demand_before_completion ? 0 : 1));
  auto next = query.add_split(fixture.context, "controlled://after-disposal");
  query.update(split, readahead_query::stage::disposed);
  query.update(split, readahead_query::stage::disposed);
  // A stale reading report cannot resurrect the disposed split's ticket.
  query.manager->update_scan_state(1, split.get(), readahead_query::stage::reading);
  held_request next_io{fixture.reactor().take_next()};
  REQUIRE(next_io);
  CHECK(budget->active(backend) == 1);
  complete_success(*next_io);
  CHECK(await_condition([&] { return budget->active(backend) == 0; }));
  query.manager->stop();
}

TEST_CASE("demand wins ownership while speculative acquisition waits",
          "[cache][readahead][scan-tickets]")
{
  cache_fixture fixture;
  auto budget  = std::make_shared<shared_scan_budget>();
  auto backend = fixture.backend();
  auto blocker = budget->acquire({{backend, 1}}, sirius::make_query_id(0), {}, true);
  readahead_query query(budget, 1);
  auto split = query.add_split(fixture.context, "controlled://demand-first");
  query.manager->start();
  REQUIRE(await_condition([&] { return budget->pending() == 1; }));
  query.update(split, readahead_query::stage::reading);
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 2);
  CHECK(query.manager->counters().cold_read_tickets == 1);
  blocker.reset();
  query.update(split, readahead_query::stage::disposed);
  REQUIRE(await_condition(
    [&] { return query.outcomes(readahead_query::outcome::skipped_fell_behind) == 1; }));
  CHECK(budget->active(backend) == 0);
  CHECK(fixture.reactor().pending() == 0);
  auto next = query.add_split(fixture.context, "controlled://after-demand");
  held_request io{fixture.reactor().take_next()};
  REQUIRE(io);
  complete_success(*io);
  CHECK(await_condition([&] { return budget->active(backend) == 0; }));
  query.manager->stop();
}

TEST_CASE("disposal and teardown retain speculative capacity until IO completes",
          "[cache][readahead][scan-tickets]")
{
  cache_fixture fixture;
  auto budget  = std::make_shared<shared_scan_budget>();
  auto backend = fixture.backend();
  readahead_query query(budget, 1);
  auto split = query.add_split(fixture.context, "controlled://late-completion");
  query.manager->start();
  held_request io{fixture.reactor().take_next()};
  REQUIRE(io);
  query.update(split, readahead_query::stage::reading);
  query.update(split, readahead_query::stage::disposed);
  query.update(split, readahead_query::stage::disposed);
  SECTION("manager survives") {}
  SECTION("manager stopped") { query.manager->stop(); }
  SECTION("manager destroyed") { query.manager.reset(); }
  CHECK(budget->active(backend) == 1);
  split.reset();
  complete_success(*io);
  CHECK(await_condition([&] { return budget->active(backend) == 0; }));
}

TEST_CASE("inline prefetch completion records coverage without retaining capacity",
          "[cache][readahead][scan-tickets]")
{
  cache_fixture fixture;
  auto budget  = std::make_shared<shared_scan_budget>();
  auto backend = fixture.backend();
  readahead_query query(budget, 1);
  fixture.reactor().after_enqueue = [&] {
    held_request io{fixture.reactor().take_next()};
    if (io) complete_success(*io);
  };
  auto split = query.add_split(fixture.context, "controlled://inline");
  query.manager->start();
  REQUIRE(
    await_condition([&] { return query.outcomes(readahead_query::outcome::prefetched) == 1; }));
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 0);
  CHECK(query.manager->counters().cold_read_tickets == 0);
  query.update(split, readahead_query::stage::disposed);
  auto next = query.add_split(fixture.context, "controlled://inline-next");
  REQUIRE(
    await_condition([&] { return query.outcomes(readahead_query::outcome::prefetched) == 2; }));
  query.manager->stop();
  CHECK(budget->active(backend) == 0);
  fixture.reactor().after_enqueue = {};
}

// ===========================================================================
// bounded memory retry against real cache pressure
// ===========================================================================

namespace {

using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

/// `lru`, so the evictor sweeps only under its own pressure rule; with the threshold at 1.0 only
/// a failed preparation's explicit eviction frees anything.
scan_manager_config constrained_lru_config()
{
  scan_manager_config cfg;
  cfg.thread_pool.num_threads               = 3;
  cfg.uring_reactors_per_gpu                = 1;
  cfg.cache.mode                            = cucascade::io::cache::cache_mode::cucs;
  cfg.cache.eviction                        = cucascade::io::cache::eviction_policy::lru;
  cfg.cache.min_prefetching_budget_fraction = 0.25;
  cfg.cache.eviction_threshold_fraction     = 1.0;
  cfg.apply_cache_mode();
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

/// A scan manager whose host tier is exhausted, and a one-operator query for its readahead.
struct pressure_fixture {
  // The cache reserves 64 MiB at construction. Exhaust the remaining tier and
  // ask for more than that reserve so preparation really encounters pressure.
  static constexpr std::int64_t request_bytes = 96ll << 20;

  pressure_fixture()
  {
    auto* host    = memory->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
    auto* host_mr = host->get_memory_resource_of<cucascade::memory::Tier::HOST>();
    pressure      = host_mr->allocate_multiple_blocks(host_mr->get_available_memory());
    REQUIRE(pressure);
    REQUIRE(host_mr->get_available_memory() == 0);
  }

  [[nodiscard]] std::shared_ptr<sirius::op::scan::scan_info> make_split(std::int64_t offset)
  {
    auto ds = scans.create_datasource(file.path.string());
    std::vector<cudf::io::text::byte_range_info> ranges{{offset, request_bytes}};
    ds->fadvise(ranges, 0);
    std::vector<sirius::op::scan::scan_info::fadvise_entry> entries{{ds, ranges}};
    return std::make_shared<sirius::op::scan::scan_info>(std::move(entries));
  }

  [[nodiscard]] std::uintptr_t backend() const noexcept
  {
    return reinterpret_cast<std::uintptr_t>(scans.io_ctx());
  }

  sirius::test::temp_data_file file{static_cast<std::size_t>(2 * request_bytes)};
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    constrained_memory_manager();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager scans{
    lifecycle, constrained_lru_config(), *memory, sirius::test::single_gpu_index_for_prefetch()};
  cucascade::memory::fixed_multiple_blocks_allocation pressure;
  sirius::test::single_scan_query query;
  std::shared_ptr<sirius::event::query_event_publisher> publisher =
    std::make_shared<sirius::event::query_event_publisher>();
};

}  // namespace

TEST_CASE("readahead bounds memory retries and returns capacity on abandonment",
          "[cache][eviction][readahead][retry]")
{
  pressure_fixture fixture;
  auto budget    = std::make_shared<shared_scan_budget>();
  auto readahead = std::make_shared<readahead_scan_manager>(*fixture.publisher, 1, budget, 1);
  readahead->prepare_for_query(fixture.query.get());
  auto split = fixture.make_split(0);
  readahead->register_scan_task(split, 1);
  auto backend = fixture.backend();
  auto await   = [](auto predicate) { return await_condition(predicate, 5s); };

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
    fixture.pressure.reset();
    auto next = fixture.make_split(pressure_fixture::request_bytes);
    readahead->register_scan_task(next, 1);
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
