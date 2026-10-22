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
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "utils/telemetry_utils.hpp"

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
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
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
  explicit controlled_object(std::string path, std::string version = {})
    : _path(std::move(path)), _version(std::move(version))
  {
  }

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return 4 * chunk_size; }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return _version; }

 private:
  std::string _path, _version;
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
    if (after_enqueue) after_enqueue();
  }

  // Configured before starting the worker. Lets tests pause issue before its
  // fan-in guard drops, or complete the request inline on the issuing thread.
  std::function<void()> after_enqueue;

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
  cache_fixture()
    : memory(initialize_memory_manager(1)), context(std::make_shared<controlled_context>(1, [] {
        return std::make_unique<controlled_reactor>();
      }))
  {
    sirius::io::cache::config config;
    config.mode = sirius::io::cache::cache_mode::sirius;
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

void complete_failure(sirius::io::grouped_io_request& request)
{
  while (!request.empty()) {
    auto slice = request.take_front();
    if (slice.on_complete) (*slice.on_complete)(slice.h_buffer.fragments(), false);
    request.coordinator->report_error(std::make_error_code(std::errc::io_error));
  }
}

// Pending test IO must settle even when an assertion unwinds the fixture.
struct held_request {
  std::unique_ptr<sirius::io::grouped_io_request> request;
  ~held_request()
  {
    if (request) complete_failure(*request);
  }
  explicit operator bool() const { return bool(request); }
  sirius::io::grouped_io_request& operator*() const { return *request; }
};

bool await_condition(auto predicate)
{
  auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  return predicate();
}

struct readahead_query {
  using scan_info = sirius::op::scan::scan_info;
  using stage     = sirius::io::cache::scan_stage;
  using outcome   = sirius::scan_manager::prefetch_outcome_kind;
  struct scan_operator : sirius::op::sirius_physical_operator {
    scan_operator()
      : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::GPU_SCAN, {}, 0)
    {
      operator_id = 1;
    }
  };

  readahead_query(std::shared_ptr<sirius::scan_manager::shared_scan_budget> budget, int id)
    : manager(std::make_shared<sirius::scan_manager::readahead_scan_manager>(
        *publisher, 1, std::move(budget), 1))
  {
    scan_operator op;
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

  void update(std::shared_ptr<scan_info> const& split, stage value)
  {
    split->set_scan_stage(value);
    for (auto const& ds : split->datasources())
      ds->update(value);
    manager->update_scan_state(1, split.get(), value);
  }

  std::shared_ptr<scan_info> add_split(controlled_context& context,
                                       std::string path,
                                       bool partial = false)
  {
    std::shared_ptr<sirius::io::sirius_datasource> ds = context.open_datasource(std::move(path));
    std::vector<cudf::io::text::byte_range_info> ranges{{0, read_size}};
    ds->fadvise(ranges, 0);
    std::vector<scan_info::fadvise_entry> entries{{ds, ranges}};
    // A datasource without a cache request refuses prefetch, leaving demand
    // responsible for that part even when the first datasource succeeds.
    if (partial) { entries.push_back({context.open_datasource("controlled://unadvised"), ranges}); }
    auto split = std::make_shared<scan_info>(std::move(entries));
    manager->register_scan_task(split, 1);
    return split;
  }

  std::shared_ptr<sirius::event::query_event_publisher> publisher =
    std::make_shared<sirius::event::query_event_publisher>();
  std::shared_ptr<sirius::scan_manager::readahead_scan_manager> manager;
};

}  // namespace

TEST_CASE("demand joining prefetch issue spends one ticket and completion frees both budgets",
          "[cache][readahead][scan-tickets]")
{
  cache_fixture fixture;
  auto budget  = std::make_shared<sirius::scan_manager::shared_scan_budget>();
  auto backend = reinterpret_cast<std::uintptr_t>(fixture.context.get());
  readahead_query query(budget, 1);
  auto split = query.add_split(*fixture.context, "controlled://overlap");
  std::promise<void> resume;
  auto resumed                             = resume.get_future().share();
  auto first                               = std::make_shared<std::atomic<bool>>(true);
  auto pause_timed_out                     = std::make_shared<std::atomic<bool>>(false);
  fixture.context->reactor().after_enqueue = [first, resumed, pause_timed_out] {
    if (first->exchange(false))
      *pause_timed_out = resumed.wait_for(5s) != std::future_status::ready;
  };
  query.manager->start();
  auto io = fixture.context->reactor().take_next();
  // The issuing thread has not returned from enqueue, so scan_info::prefetch
  // has not finished issuing/publishing its outcome. Demand races that window.
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 1);
  CHECK(query.manager->counters().cold_read_tickets == 0);
  CHECK(query.manager->counters().borrowed == 0);
  resume.set_value();
  REQUIRE(io);
  complete_success(*io);
  REQUIRE(await_condition([&] {
    return query.manager->counters().outcome(readahead_query::outcome::wait_for_prefetch) == 1;
  }));
  CHECK_FALSE(pause_timed_out->load());
  CHECK(budget->active(backend) == 0);
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 0);

  // A younger query can use the shared backend while the first split remains
  // undisposed. This is the regression: the old demand charge survived here.
  readahead_query other(budget, 2);
  auto peer = other.add_split(*fixture.context, "controlled://peer");
  other.manager->start();
  auto peer_io = fixture.context->reactor().take_next();
  REQUIRE(peer_io);
  complete_success(*peer_io);
  REQUIRE(await_condition([&] { return budget->active(backend) == 0; }));
  other.manager->stop();

  // The first query's local credit was also returned, exactly once. With one
  // new prefetch held, another candidate must wait at the local gate.
  auto next    = query.add_split(*fixture.context, "controlled://next");
  auto last    = query.add_split(*fixture.context, "controlled://last");
  auto next_io = fixture.context->reactor().take_next();
  REQUIRE(next_io);
  auto timeouts = query.manager->counters().gate_timeouts.load();
  CHECK(await_condition([&] { return query.manager->counters().gate_timeouts > timeouts; }));
  CHECK(budget->pending() == 0);
  CHECK(fixture.context->reactor().pending() == 0);
  complete_success(*next_io);
  auto last_io = fixture.context->reactor().take_next();
  REQUIRE(last_io);
  complete_success(*last_io);
  query.update(split, readahead_query::stage::disposed);
  query.update(split, readahead_query::stage::disposed);
  CHECK(await_condition([&] { return budget->active(backend) == 0; }));
}

TEST_CASE("prefetch completing a subset retains the ticket for a demand-owned chunk",
          "[cache][readahead][scan-tickets][prefetch-coverage]")
{
  cache_fixture fixture;
  auto budget      = std::make_shared<sirius::scan_manager::shared_scan_budget>();
  auto backend     = reinterpret_cast<std::uintptr_t>(fixture.context.get());
  bool device_read = false;
  SECTION("host demand") {}
  SECTION("device demand") { device_read = true; }

  std::array<std::uint8_t, read_size> destination{};
  rmm::cuda_stream stream;
  rmm::device_buffer device_destination(device_read ? read_size : 0, stream.view());
  std::future<std::size_t> read;
  std::atomic<bool> first{true};
  readahead_query query(budget, 1);
  readahead_query peer(budget, 2);
  std::shared_ptr<sirius::io::sirius_datasource> ds =
    fixture.context->open_datasource("controlled://split-coverage");
  std::vector<cudf::io::text::byte_range_info> ranges{{0, read_size}, {chunk_size, read_size}};
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
                 0, read_size, static_cast<std::uint8_t*>(device_destination.data()), stream.view())
             : ds->host_read_async(0, read_size, destination.data());
  });
  query.manager->start();
  held_request demand_io{fixture.context->reactor().take_next()};
  held_request prefetch_io{fixture.context->reactor().take_next()};
  REQUIRE(demand_io);
  REQUIRE(prefetch_io);
  CHECK(budget->active(backend) == 1);
  CHECK(query.manager->counters().cold_read_tickets == 0);
  CHECK(query.manager->counters().borrowed == 0);

  // Prefetch skipped A and issued only B. Completing B must not free A's slot.
  complete_success(*prefetch_io);
  CHECK(await_condition([&] {
    return query.manager->counters().outcome(readahead_query::outcome::wait_for_prefetch) == 1;
  }));
  CHECK(read.wait_for(0ms) == std::future_status::timeout);
  CHECK(budget->active(backend) == 1);
  auto peer_split = peer.add_split(*fixture.context, "controlled://coverage-peer");
  peer.manager->start();
  CHECK(await_condition([&] { return budget->pending() == 1; }));
  CHECK(fixture.context->reactor().pending() == 0);

  complete_success(*demand_io);
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == read_size);
  CHECK(budget->active(backend) == 1);
  query.update(split, readahead_query::stage::disposed);
  query.update(split, readahead_query::stage::disposed);
  held_request peer_io{fixture.context->reactor().take_next()};
  REQUIRE(peer_io);
  CHECK(budget->active(backend) == 1);
  complete_success(*peer_io);
  CHECK(await_condition([&] { return budget->active(backend) == 0; }));
  query.manager->stop();
  peer.manager->stop();
  fixture.context->cache()->set_before_prefetch_claim_for_testing({});
}

TEST_CASE("prefetch coverage distinguishes resident extents from successful submitted IO",
          "[cache][read-arbitration][prefetch-coverage]")
{
  cache_fixture fixture;
  // Independent opens share cached bytes only with a stable object identity.
  // Without a validator, each open deliberately gets its own generation.
  auto open = [&] {
    return std::make_unique<sirius::io::sirius_datasource>(
      fixture.context, std::make_shared<controlled_object>("controlled://resident-coverage", "v1"));
  };
  fixture.datasource = open();
  fixture.advise(0);
  REQUIRE(fixture.datasource->prefetch_async([](sirius::io::cache::prefetch_result) noexcept {}) ==
          sirius::io::prefetch_refusal::issued);
  held_request warm{fixture.context->reactor().take_next()};
  REQUIRE(warm);
  complete_success(*warm);

  bool include_b = true;
  bool wider_a   = false;
  bool fail_b    = false;
  SECTION("resident A and successful B") {}
  SECTION("resident A is narrower than requested") { wider_a = true; }
  SECTION("all requested bytes already resident") { include_b = false; }
  SECTION("resident A and failed B") { fail_b = true; }
  auto ds = open();
  std::vector<cudf::io::text::byte_range_info> ranges{
    {0, static_cast<std::int64_t>(wider_a ? 2 * read_size : read_size)}};
  if (include_b) ranges.emplace_back(chunk_size, read_size);
  ds->fadvise(ranges, 0);
  REQUIRE(ds->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  std::promise<sirius::io::cache::prefetch_result> done;
  auto result = done.get_future();
  std::atomic<int> calls{0};
  auto const issued = ds->prefetch_async([&](sirius::io::cache::prefetch_result out) noexcept {
    if (calls.fetch_add(1) == 0) done.set_value(out);
  });
  // Issue is synchronous even though completion is controlled by the test.
  // Own unexpected IO too, so a failed no-IO assertion cannot hang teardown.
  held_request io{fixture.context->reactor().take_next(0ms)};
  if (include_b) {
    CHECK(issued == sirius::io::prefetch_refusal::issued);
    REQUIRE(io);
    if (fail_b)
      complete_failure(*io);
    else
      complete_success(*io);
  } else {
    CHECK(issued != sirius::io::prefetch_refusal::issued);
    CHECK_FALSE(io);
    if (io) complete_failure(*io);
  }
  REQUIRE(result.wait_for(2s) == std::future_status::ready);
  auto const out = result.get();
  CHECK(out.ok == !fail_b);
  CHECK(out.covered == (!wider_a && !fail_b));
  CHECK(calls == 1);
}

TEST_CASE("incomplete prefetch transfers its ticket to demand until disposal",
          "[cache][readahead][scan-tickets]")
{
  cache_fixture fixture;
  auto budget  = std::make_shared<sirius::scan_manager::shared_scan_budget>();
  auto backend = reinterpret_cast<std::uintptr_t>(fixture.context.get());
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
  auto split = query.add_split(*fixture.context, "controlled://incomplete", partial);
  query.manager->start();
  auto io = fixture.context->reactor().take_next();
  REQUIRE(io);
  if (demand_before_completion) query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 1);
  if (partial)
    complete_success(*io);
  else
    complete_failure(*io);
  auto expected = demand_before_completion ? readahead_query::outcome::wait_for_prefetch
                                           : readahead_query::outcome::prefetched;
  REQUIRE(await_condition([&] { return query.manager->counters().outcome(expected) == 1; }));
  CHECK(budget->active(backend) == (demand_before_completion ? 1 : 0));
  query.update(split, readahead_query::stage::reading);
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 1);
  CHECK(query.manager->counters().cold_read_tickets == (demand_before_completion ? 0 : 1));
  auto next = query.add_split(*fixture.context, "controlled://after-disposal");
  query.update(split, readahead_query::stage::disposed);
  query.update(split, readahead_query::stage::disposed);
  // A stale reading report cannot resurrect the disposed split's ticket.
  query.manager->update_scan_state(1, split.get(), readahead_query::stage::reading);
  auto next_io = fixture.context->reactor().take_next();
  REQUIRE(next_io);
  CHECK(budget->active(backend) == 1);
  complete_success(*next_io);
  CHECK(await_condition([&] { return budget->active(backend) == 0; }));
}

TEST_CASE("demand wins ownership while speculative acquisition waits",
          "[cache][readahead][scan-tickets]")
{
  cache_fixture fixture;
  auto budget  = std::make_shared<sirius::scan_manager::shared_scan_budget>();
  auto backend = reinterpret_cast<std::uintptr_t>(fixture.context.get());
  auto blocker = budget->acquire({{backend, 1}}, sirius::make_query_id(0), {}, true);
  readahead_query query(budget, 1);
  auto split = query.add_split(*fixture.context, "controlled://demand-first");
  query.manager->start();
  REQUIRE(await_condition([&] { return budget->pending() == 1; }));
  query.update(split, readahead_query::stage::reading);
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 2);
  CHECK(query.manager->counters().cold_read_tickets == 1);
  blocker.reset();
  query.update(split, readahead_query::stage::disposed);
  REQUIRE(await_condition([&] {
    return query.manager->counters().outcome(readahead_query::outcome::skipped_fell_behind) == 1;
  }));
  CHECK(budget->active(backend) == 0);
  CHECK(fixture.context->reactor().pending() == 0);
  auto next = query.add_split(*fixture.context, "controlled://after-demand");
  auto io   = fixture.context->reactor().take_next();
  REQUIRE(io);
  complete_success(*io);
  CHECK(await_condition([&] { return budget->active(backend) == 0; }));
}

TEST_CASE("disposal and teardown retain speculative capacity until IO completes",
          "[cache][readahead][scan-tickets]")
{
  cache_fixture fixture;
  auto budget  = std::make_shared<sirius::scan_manager::shared_scan_budget>();
  auto backend = reinterpret_cast<std::uintptr_t>(fixture.context.get());
  readahead_query query(budget, 1);
  auto split = query.add_split(*fixture.context, "controlled://late-completion");
  query.manager->start();
  auto io = fixture.context->reactor().take_next();
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
  auto budget  = std::make_shared<sirius::scan_manager::shared_scan_budget>();
  auto backend = reinterpret_cast<std::uintptr_t>(fixture.context.get());
  readahead_query query(budget, 1);
  fixture.context->reactor().after_enqueue = [&] {
    auto io = fixture.context->reactor().take_next();
    if (io) complete_success(*io);
  };
  auto split = query.add_split(*fixture.context, "controlled://inline");
  query.manager->start();
  REQUIRE(await_condition(
    [&] { return query.manager->counters().outcome(readahead_query::outcome::prefetched) == 1; }));
  query.update(split, readahead_query::stage::reading);
  CHECK(budget->active(backend) == 0);
  CHECK(query.manager->counters().cold_read_tickets == 0);
  query.update(split, readahead_query::stage::disposed);
  auto next = query.add_split(*fixture.context, "controlled://inline-next");
  REQUIRE(await_condition(
    [&] { return query.manager->counters().outcome(readahead_query::outcome::prefetched) == 2; }));
  query.manager->stop();
  CHECK(budget->active(backend) == 0);
}

TEST_CASE("an overlapping cache read waits for prefetch and reuses its chunk",
          "[cache][cache_handle][read-arbitration]")
{
  cache_fixture fixture;
  fixture.advise(0);

  std::atomic<bool> prefetched{false};
  REQUIRE(fixture.datasource->prefetch_async(
            [&prefetched](sirius::io::cache::prefetch_result result) noexcept {
              bool const ok = result.ok;
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
  REQUIRE(fixture.datasource->prefetch_async(
            [&prefetched](sirius::io::cache::prefetch_result result) noexcept {
              bool const ok = result.ok;
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

  REQUIRE(fixture.datasource->prefetch_async([](sirius::io::cache::prefetch_result) noexcept {}) ==
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

TEST_CASE("raw cache validates identities across opens", "[cache][object_identity]")
{
  cache_fixture fixture;
  int const mode            = GENERATE(0, 1, 2);  // same version, changed version, no validator
  std::string const version = mode == 2 ? "" : "v1";
  auto open                 = [&](std::string tag) {
    return std::make_unique<sirius::io::sirius_datasource>(
      fixture.context,
      std::make_shared<controlled_object>("controlled://versioned", std::move(tag)));
  };
  fixture.datasource = open(version);
  fixture.advise(0);
  REQUIRE(fixture.datasource->prefetch_async([](sirius::io::cache::prefetch_result) noexcept {}) ==
          sirius::io::prefetch_refusal::issued);
  auto fill = fixture.context->reactor().take_next();
  REQUIRE(fill);
  complete_success(*fill, 0x5a);
  auto reopened = open(mode == 1 ? "v2" : version);
  std::array<std::uint8_t, read_size> bytes{};
  auto read = reopened->host_read_async(0, bytes.size(), bytes.data());
  if (mode != 0) {
    auto request = fixture.context->reactor().take_next();
    REQUIRE(request);
    complete_success(*request, 0x7b);
  }
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == bytes.size());
  CHECK(std::ranges::all_of(bytes, [&](auto b) { return b == (mode == 0 ? 0x5a : 0x7b); }));
  CHECK(fixture.context->reactor().pending() == 0);
}

TEST_CASE("a late old raw-cache fill cannot overwrite the new version", "[cache][object_identity]")
{
  cache_fixture fixture;
  auto open = [&](std::string tag) {
    return std::make_unique<sirius::io::sirius_datasource>(
      fixture.context,
      std::make_shared<controlled_object>("controlled://versioned", std::move(tag)));
  };
  fixture.datasource = open("v1");
  fixture.advise(0);
  REQUIRE(fixture.datasource->prefetch_async([](sirius::io::cache::prefetch_result) noexcept {}) ==
          sirius::io::prefetch_refusal::issued);
  auto old_fill = fixture.context->reactor().take_next();
  REQUIRE(old_fill);
  auto old_source    = std::move(fixture.datasource);
  fixture.datasource = open("v2");
  fixture.advise(0);
  REQUIRE(fixture.datasource->prefetch_async([](sirius::io::cache::prefetch_result) noexcept {}) ==
          sirius::io::prefetch_refusal::issued);
  auto new_fill = fixture.context->reactor().take_next();
  REQUIRE(new_fill);
  complete_success(*new_fill, 0x7b);
  complete_success(*old_fill, 0x5a);
  auto reopened = open("v2");
  std::array<std::uint8_t, read_size> bytes{};
  auto read = reopened->host_read_async(0, bytes.size(), bytes.data());
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == bytes.size());
  CHECK(std::ranges::all_of(bytes, [](auto b) { return b == 0x7b; }));
  read = old_source->host_read_async(0, bytes.size(), bytes.data());
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == bytes.size());
  CHECK(std::ranges::all_of(bytes, [](auto b) { return b == 0x5a; }));
  CHECK(fixture.context->reactor().pending() == 0);
}

TEST_CASE("superseded raw cache entries release their arenas and handles without pressure",
          "[cache][object_identity][raw_retirement]")
{
  cache_fixture fixture;
  bool const validated = GENERATE(true, false);
  std::vector<std::weak_ptr<controlled_object>> versions;
  for (int i = 0; i < 24; ++i) {
    auto object = std::make_shared<controlled_object>("controlled://retirement",
                                                      validated ? std::to_string(i) : "");
    versions.emplace_back(object);
    fixture.datasource = std::make_unique<sirius::io::sirius_datasource>(fixture.context, object);
    fixture.advise(0);
    REQUIRE(fixture.datasource->prefetch_async([](sirius::io::cache::prefetch_result) noexcept {
    }) == sirius::io::prefetch_refusal::issued);
    auto fill = fixture.context->reactor().take_next();
    REQUIRE(fill);
    complete_success(*fill);
    CHECK(fixture.context->cache()->file_entry_count_for_testing() == 1);
  }
  fixture.datasource.reset();
  // No query, allocation pressure or explicit evict call is needed to retire old versions.
  auto const deadline = std::chrono::steady_clock::now() + 3s;
  auto old_released   = [&] {
    return std::all_of(
      versions.begin(), versions.end() - 1, [](auto const& v) { return v.expired(); });
  };
  while (!old_released() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  CHECK(old_released());
  CHECK_FALSE(versions.back().expired());
  CHECK(fixture.context->cache()->claimed_bytes() == chunk_size);
  fixture.context->shutdown_cache();
  CHECK(versions.back().expired());
}

TEST_CASE("retired raw arena survives an abandoned in-flight fill until physical completion",
          "[cache][object_identity][raw_retirement]")
{
  cache_fixture fixture;
  auto object = std::make_shared<controlled_object>("controlled://retirement", "v1");
  std::weak_ptr<controlled_object> old = object;
  fixture.datasource = std::make_unique<sirius::io::sirius_datasource>(fixture.context, object);
  object.reset();
  fixture.advise(0);
  REQUIRE(fixture.datasource->prefetch_async([](sirius::io::cache::prefetch_result) noexcept {}) ==
          sirius::io::prefetch_refusal::issued);
  auto fill = fixture.context->reactor().take_next();
  REQUIRE(fill);
  fixture.datasource = std::make_unique<sirius::io::sirius_datasource>(
    fixture.context, std::make_shared<controlled_object>("controlled://retirement", "v2"));
  fixture.advise(0);
  std::this_thread::sleep_for(250ms);  // let the evictor release the disposed v1 request
  CHECK_FALSE(old.expired());
  CHECK(fixture.context->cache()->claimed_bytes() == 2 * chunk_size);
  complete_success(*fill);
  fill.reset();
  auto const deadline = std::chrono::steady_clock::now() + 3s;
  while (!old.expired() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  CHECK(old.expired());
  CHECK(fixture.context->cache()->claimed_bytes() == chunk_size);
}

TEST_CASE("retired raw arena survives a cache-hit CUDA copy without a prefetch handle",
          "[cache][object_identity][raw_retirement][gpu_execution]")
{
  cache_fixture fixture;
  auto object = std::make_shared<controlled_object>("controlled://retirement", "v1");
  std::weak_ptr<controlled_object> old = object;
  fixture.datasource = std::make_unique<sirius::io::sirius_datasource>(fixture.context, object);
  object.reset();
  fixture.advise(0);
  REQUIRE(fixture.datasource->prefetch_async([](sirius::io::cache::prefetch_result) noexcept {}) ==
          sirius::io::prefetch_refusal::issued);
  auto fill = fixture.context->reactor().take_next();
  REQUIRE(fill);
  complete_success(*fill, 0x6c);
  fill.reset();
  auto reader = std::make_unique<sirius::io::sirius_datasource>(
    fixture.context, std::make_shared<controlled_object>("controlled://retirement", "v1"));
  rmm::cuda_stream stream;
  rmm::device_buffer destination{read_size, stream};
  struct gate {
    cudaStream_t stream;
    std::atomic<bool> released{false};
    ~gate()
    {
      released.store(true);
      std::ignore = cudaStreamSynchronize(stream);
    }
  } blocked{stream.value()};
  REQUIRE(cudaLaunchHostFunc(
            stream.value(),
            [](void* ptr) {
              auto& ready = static_cast<gate*>(ptr)->released;
              while (!ready.load()) {
                std::this_thread::sleep_for(1ms);
              }
            },
            &blocked) == cudaSuccess);
  auto read =
    reader->device_read_async(0, read_size, static_cast<std::uint8_t*>(destination.data()), stream);
  reader.reset();
  fixture.datasource = std::make_unique<sirius::io::sirius_datasource>(
    fixture.context, std::make_shared<controlled_object>("controlled://retirement", "v2"));
  fixture.advise(0);
  std::this_thread::sleep_for(250ms);
  CHECK(read.wait_for(0ms) != std::future_status::ready);
  CHECK_FALSE(old.expired());
  blocked.released.store(true);
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == read_size);
  std::array<std::uint8_t, read_size> host{};
  REQUIRE(cudaMemcpyAsync(
            host.data(), destination.data(), host.size(), cudaMemcpyDeviceToHost, stream.value()) ==
          cudaSuccess);
  stream.synchronize();
  CHECK(std::ranges::all_of(host, [](auto byte) { return byte == 0x6c; }));
  auto const deadline = std::chrono::steady_clock::now() + 3s;
  while (!old.expired() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  CHECK(old.expired());
}
