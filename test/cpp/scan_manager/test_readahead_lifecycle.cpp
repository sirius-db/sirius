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
#include "exec/config.hpp"
#include "io/kvikio/kvikio_context.hpp"
#include "io/rest/config.hpp"
#include "io/uring/config.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "query_id.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/gatekeeper.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/telemetry_utils.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using sirius::io::cache::cache_mode;
using sirius::io::cache::eviction_policy;
using sirius::io::cache::scan_stage;
using sirius::scan_manager::gatekeeper;
using sirius::scan_manager::readahead_scan_manager;

namespace {
/// A live event publisher for the readahead under test to register its mailbox
/// with.  Held by shared_ptr because the subscriber keeps only a weak reference,
/// and declared before the readahead in each test so it outlives it.
auto make_event_publisher() { return std::make_shared<sirius::event::query_event_publisher>(); }
}  // namespace
using sirius::scan_manager::scan_manager_config;

namespace {
constexpr auto PIPELINE_THREADS =
  static_cast<std::size_t>(sirius::exec::default_gpu_pipeline_num_threads);
}  // namespace

// ===========================================================================
// per-backend scan budget
// ===========================================================================

TEST_CASE("each backend publishes its own default scan budget", "[scan_manager][readahead]")
{
  CHECK(sirius::io::uring::config{}.n_max_concurrent_scans == PIPELINE_THREADS);
  CHECK(sirius::io::rest::config{}.n_max_concurrent_scans == 2 * PIPELINE_THREADS);
  // kvikIO has no prefetching cache to read ahead into, so it publishes no
  // depth at all and there is no knob that could give it one.
  sirius::io::kvikio_context kvikio;
  CHECK_FALSE(kvikio.can_use_prefetching_cache());
  CHECK(kvikio.n_max_concurrent_scans() == 0);
}

TEST_CASE("the readahead budget follows the cache mode when unset", "[scan_manager][readahead]")
{
  // Stand-in for the widest n_max_concurrent_scans a live backend publishes.
  constexpr std::size_t backend_budget = 6;

  auto budget_for = [](cache_mode mode) {
    scan_manager_config cfg;
    cfg.cache.mode = mode;
    cfg.apply_cache_mode();
    return cfg.resolve_readahead(backend_budget, sirius::scan_manager::prefetch_strategy::eager)
      .budget;
  };

  CHECK(budget_for(cache_mode::none) == 0);
  // `os` has no prefetching cache, but ordering scans ahead of demand still
  // warms the page cache, so readahead is on.
  CHECK(budget_for(cache_mode::os) == backend_budget);
  CHECK(budget_for(cache_mode::sirius) == backend_budget);
}

TEST_CASE("a partially prepared split retries before issuing prefetch",
          "[scan_manager][readahead][prepare]")
{
  using outcome = sirius::op::scan::scan_info::prepare_outcome;

  CHECK_FALSE(outcome{}.ready());
  CHECK(outcome{.prepared = 1}.ready());
  CHECK_FALSE(outcome{.prepared = 1, .failed = 1}.ready());
  CHECK_FALSE(outcome{.prepared = 1, .fell_behind = 1}.ready());
}

TEST_CASE("readahead backend selection considers only the supplied query contexts",
          "[scan_manager][readahead]")
{
  using sirius::scan_manager::backend_readahead_policy;
  using sirius::scan_manager::prefetch_strategy;
  using sirius::scan_manager::select_readahead_backend;

  // A REST context retained from a preceding query is intentionally absent:
  // the current local query must retain the uring policy.
  constexpr std::array local_query = {
    backend_readahead_policy{.budget = 4, .strategy = prefetch_strategy::opportunistic}};
  auto selected = select_readahead_backend(local_query);
  CHECK(selected.budget == 4);
  CHECK(selected.strategy == prefetch_strategy::opportunistic);

  // Likewise, a REST context created only to LIST objects must not enable
  // readahead for a current kvikIO query that publishes no budget.
  constexpr std::array kvikio_query = {
    backend_readahead_policy{.budget = 0, .strategy = prefetch_strategy::opportunistic}};
  selected = select_readahead_backend(kvikio_query);
  CHECK(selected.budget == 0);
  CHECK(selected.strategy == prefetch_strategy::opportunistic);

  // A genuinely mixed current query still takes the widest backend and keeps
  // that backend's strategy paired with its budget.
  constexpr std::array mixed_query = {
    backend_readahead_policy{.budget = 4, .strategy = prefetch_strategy::opportunistic},
    backend_readahead_policy{.budget = 8, .strategy = prefetch_strategy::eager}};
  selected = select_readahead_backend(mixed_query);
  CHECK(selected.budget == 8);
  CHECK(selected.strategy == prefetch_strategy::eager);
}

TEST_CASE("explicit readahead settings override the selected backend policy",
          "[scan_manager][readahead]")
{
  scan_manager_config cfg;
  cfg.cache.mode          = cache_mode::sirius;
  cfg.max_readahead_scans = 3;
  cfg.readahead_strategy  = sirius::scan_manager::prefetch_strategy::eager;
  cfg.pipeline_width      = 9;
  cfg.apply_cache_mode();

  auto const plan =
    cfg.resolve_readahead(4, sirius::scan_manager::prefetch_strategy::opportunistic);
  CHECK(plan.budget == 3);
  CHECK(plan.strategy == sirius::scan_manager::prefetch_strategy::eager);
}

TEST_CASE("apply_cache_mode leaves the other derived knobs alone", "[scan_manager][readahead]")
{
  scan_manager_config cfg;
  cfg.cache.mode     = cache_mode::sirius;
  cfg.cache.eviction = eviction_policy::idle;
  cfg.apply_cache_mode();

  CHECK(cfg.cache.use_prefetching_cache());
  CHECK(cfg.cache.dispose_on_idle);
  CHECK(cfg.uring.use_odirect);
}

// ===========================================================================
// gatekeeper
// ===========================================================================
//
// The budget is construction-time config, and the gate starts CLOSED: that is
// what lets the worker treat "not armed yet" and "no ticket free right now" as
// the same wait, instead of needing a started flag of its own.

namespace {
constexpr auto INSTANT = std::chrono::milliseconds{0};
constexpr auto BRIEF   = std::chrono::milliseconds{50};
}  // namespace

TEST_CASE("a fresh gatekeeper hands out nothing until it is armed", "[scan_manager][gatekeeper]")
{
  gatekeeper g{4};
  CHECK(g.available() == 0);
  CHECK_FALSE(g.acquire_for(INSTANT));

  g.reload();
  CHECK(g.available() == 4);
  CHECK(g.acquire_for(INSTANT));
  CHECK(g.available() == 3);
}

TEST_CASE("the budget bounds how many tickets are out at once", "[scan_manager][gatekeeper]")
{
  gatekeeper g{2};
  g.reload();

  REQUIRE(g.acquire_for(INSTANT));
  REQUIRE(g.acquire_for(INSTANT));
  // Exhausted: the third caller must wait rather than over-subscribe.
  CHECK_FALSE(g.acquire_for(BRIEF));

  g.release();
  CHECK(g.acquire_for(INSTANT));
}

TEST_CASE("an executor read borrows rather than waits", "[scan_manager][gatekeeper]")
{
  // The whole point of the counter being signed: a read the readahead never
  // covered is the query's critical path and can never be made to wait, so it
  // is allowed to push the count negative.
  gatekeeper g{1};
  g.reload();

  CHECK_FALSE(g.acquire_or_borrow());  // took the free one
  CHECK(g.available() == 0);

  CHECK(g.acquire_or_borrow());  // nothing free -- borrowed
  CHECK(g.available() == -1);
  CHECK(g.deficit() == 1);

  // Debt is repaid before the readahead may start anything new: the returning
  // ticket brings the count to zero, which is still not acquirable.
  g.release();
  CHECK(g.deficit() == 0);
  CHECK_FALSE(g.acquire_for(INSTANT));

  g.release();
  CHECK(g.acquire_for(INSTANT));
}

TEST_CASE("reload clears outstanding debt", "[scan_manager][gatekeeper]")
{
  // Debt describes how the executor WAS competing; a re-arm says that is no
  // longer the question.
  gatekeeper g{2};
  g.reload();
  REQUIRE_FALSE(g.acquire_or_borrow());  // covered by the budget
  REQUIRE_FALSE(g.acquire_or_borrow());
  REQUIRE(g.acquire_or_borrow());  // budget spent -- this one is debt
  REQUIRE(g.deficit() == 1);

  g.reload();
  CHECK(g.deficit() == 0);
  CHECK(g.available() == 2);
}

TEST_CASE("stop interrupts a waiting acquire", "[scan_manager][gatekeeper]")
{
  gatekeeper g{1};
  g.reload();
  REQUIRE(g.acquire_for(INSTANT));  // take the only ticket

  std::atomic<bool> returned{false};
  std::thread waiter{[&] {
    // Would otherwise sit here for the full 10s: nothing is going to release.
    CHECK_FALSE(g.acquire_for(std::chrono::seconds{10}));
    returned.store(true);
  }};

  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  REQUIRE_FALSE(returned.load());  // genuinely parked, not racing through

  auto const before = std::chrono::steady_clock::now();
  g.stop();
  waiter.join();
  CHECK(returned.load());
  CHECK(std::chrono::steady_clock::now() - before < std::chrono::seconds{5});
}

TEST_CASE("a stopped gate stays stopped until reloaded", "[scan_manager][gatekeeper]")
{
  gatekeeper g{2};
  g.reload();
  g.stop();
  // Tickets are free, but the gate is shut: teardown must not start new work.
  CHECK_FALSE(g.acquire_for(BRIEF));

  g.reload();
  CHECK(g.acquire_for(INSTANT));
}

TEST_CASE("wait_for_all reports on the tickets, not on why it woke", "[scan_manager][gatekeeper]")
{
  gatekeeper g{2};
  g.reload();
  CHECK(g.wait_for_all(INSTANT));  // nothing out

  REQUIRE(g.acquire_for(INSTANT));
  CHECK_FALSE(g.wait_for_all(BRIEF));  // one still out

  g.release();
  CHECK(g.wait_for_all(INSTANT));
}

TEST_CASE("stop does not fake a drain", "[scan_manager][gatekeeper]")
{
  // Draining is about tickets coming back, which a stop says nothing about.
  // Reporting success here would let teardown log a settled query while IO was
  // still in flight.
  gatekeeper g{2};
  g.reload();
  REQUIRE(g.acquire_for(INSTANT));
  g.stop();
  CHECK_FALSE(g.wait_for_all(INSTANT));
}

TEST_CASE("a returning ticket wakes a waiter", "[scan_manager][gatekeeper]")
{
  gatekeeper g{1};
  g.reload();
  REQUIRE(g.acquire_for(INSTANT));

  std::thread releaser{[&] {
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    g.release();
  }};
  CHECK(g.acquire_for(std::chrono::seconds{5}));
  releaser.join();
}

// ===========================================================================
// worker lifecycle
// ===========================================================================

namespace {
/// Minimal concrete scan carrying only the operator id the readahead keys its
/// work queue on.
struct test_scan : sirius::op::sirius_physical_operator {
  test_scan()
    : sirius::op::sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::GPU_SCAN, {}, 0)
  {
    operator_id = 1;
  }
};

/// A query with one GPU_SCAN operator, which is what it takes to park the
/// worker: with no work queues at all it runs out of order and exits on its
/// own, and a worker that is already gone hides everything a second start does.
class single_scan_query {
 public:
  single_scan_query()
  {
    auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(_ctx);
    _build.set_pipeline_source(*pipeline, *_scan);
    auto const id = sirius::make_query_id(1);
    _query        = std::make_unique<sirius::planner::query>(
      std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>>{pipeline},
      _telemetry->context(),
      id,
      sirius::telemetry::query_telemetry_info{
        _telemetry->engine_id(), _telemetry->worker_id(), id});
  }

  [[nodiscard]] const sirius::planner::query& get() const { return *_query; }

 private:
  std::shared_ptr<const sirius::telemetry::telemetry_context> _telemetry =
    sirius::test::make_test_telemetry_context();
  sirius::pipeline::pipeline_build_context _ctx{nullptr, true};
  sirius::pipeline::sirius_pipeline_build_state _build;
  std::unique_ptr<test_scan> _scan = std::make_unique<test_scan>();
  std::unique_ptr<sirius::planner::query> _query;
};
}  // namespace

TEST_CASE("a zero budget means the backend opted out and no worker runs",
          "[scan_manager][readahead]")
{
  auto sm = make_event_publisher();
  readahead_scan_manager m{*sm, 0};
  m.start();
  CHECK_FALSE(m.is_running());

  // stop() on a manager that never started must be a no-op, not a hang.
  m.stop();
  CHECK_FALSE(m.is_running());
}

TEST_CASE("start runs a worker and stop joins it", "[scan_manager][readahead]")
{
  auto sm = make_event_publisher();
  readahead_scan_manager m{*sm, 4};
  REQUIRE_FALSE(m.is_running());

  m.start();
  CHECK(m.is_running());

  m.stop();
  CHECK_FALSE(m.is_running());
}

TEST_CASE("a second start on a parked worker does not join it", "[scan_manager][readahead]")
{
  // The failure here is a hang, not a wrong value: a second start that moves a
  // fresh jthread over the live one joins a worker whose stop token nothing
  // will ever request, and the caller never comes back.
  single_scan_query query;
  auto sm = make_event_publisher();
  readahead_scan_manager m{*sm, 4};
  m.prepare_for_query(query.get());

  std::promise<void> done;
  auto finished = done.get_future();
  std::jthread caller{[&] {
    m.start();
    m.start();  // already running -- must not spawn a second worker
    m.stop();
    m.stop();  // already stopped
    done.set_value();
  }};

  REQUIRE(finished.wait_for(std::chrono::seconds{2}) == std::future_status::ready);
  CHECK_FALSE(m.is_running());
}

TEST_CASE("the destructor stops a running worker", "[scan_manager][readahead]")
{
  // The interesting failure here is a hang, not a wrong value: a worker parked
  // on the gate with nothing to wake it would never be joined.
  auto sm = make_event_publisher();
  auto m  = std::make_unique<readahead_scan_manager>(*sm, 4);
  m->start();
  REQUIRE(m->is_running());
  m.reset();
  SUCCEED("destructor joined the worker");
}

TEST_CASE("teardown does not wait out the drain on a gate that never armed",
          "[scan_manager][readahead]")
{
  // Opportunistic never arms without an idle signal, so the gate holds no
  // tickets and has none outstanding -- but it still reads as undrained, and
  // waiting on that would cost the full timeout for nothing.
  auto sm = make_event_publisher();
  readahead_scan_manager m{*sm, 4};
  m.start(sirius::scan_manager::prefetch_strategy::opportunistic);
  REQUIRE(m.is_running());

  auto const before = std::chrono::steady_clock::now();
  m.stop();
  CHECK(std::chrono::steady_clock::now() - before < std::chrono::milliseconds{150});
}

TEST_CASE("update is safe on a manager that was never started", "[scan_manager][readahead]")
{
  auto sm = make_event_publisher();
  readahead_scan_manager m{*sm, 4};
  m.update_scan_state(7, nullptr, scan_stage::reading);
  m.update_scan_state(7, nullptr, scan_stage::disposed);
  CHECK_FALSE(m.is_running());
}

TEST_CASE("update tolerates a null split", "[scan_manager][readahead]")
{
  // A resident cached batch has no scan_info and reports a null task.
  auto sm = make_event_publisher();
  readahead_scan_manager m{*sm, 4};
  m.start();

  for (int i = 0; i < 200; ++i) {
    m.update_scan_state(1, nullptr, scan_stage::reading);
    m.update_scan_state(2, nullptr, scan_stage::queued);
    m.update_scan_state(1, nullptr, scan_stage::disposed);
  }

  CHECK(m.is_running());
  m.stop();
  CHECK_FALSE(m.is_running());
}

TEST_CASE("a stopped manager stays stopped", "[scan_manager][readahead]")
{
  auto sm = make_event_publisher();
  readahead_scan_manager m{*sm, 4};
  m.start();
  m.stop();
  REQUIRE_FALSE(m.is_running());

  // Start-once, stop-once: stop() shuts the gate for good and tears the event
  // subscriber down, so a restarted worker could only spin on gate timeouts
  // issuing nothing.  No worker is born instead.
  m.start();
  CHECK_FALSE(m.is_running());
  std::this_thread::sleep_for(std::chrono::milliseconds{150});
  CHECK(m.counters().gate_timeouts.load() == 0);
}

// ===========================================================================
// how a settled prefetch is judged
// ===========================================================================

TEST_CASE("a prefetch is judged by what the consumer was doing when it settled",
          "[scan_manager][readahead][counters]")
{
  using kind          = sirius::scan_manager::prefetch_outcome_kind;
  auto const classify = &readahead_scan_manager::classify_prefetch;

  SECTION("allocation failure outranks everything")
  {
    // Even with a live split and an untouched consumer, no buffers means there
    // was never an attempt to be early or late for.
    CHECK(classify(/*allocation_failed=*/true,
                   /*split_alive=*/true,
                   /*issued_io=*/false,
                   scan_stage::none) == kind::skipped_memory_pressure);
    CHECK(classify(true, true, true, scan_stage::reading) == kind::skipped_memory_pressure);
  }

  SECTION("an expired split is the readahead running behind")
  {
    CHECK(classify(false, /*split_alive=*/false, true, scan_stage::none) ==
          kind::skipped_fell_behind);
  }

  SECTION("IO that landed before the consumer arrived is the win")
  {
    for (auto stage : {scan_stage::none, scan_stage::initialized, scan_stage::queued}) {
      CHECK(classify(false, true, /*issued_io=*/true, stage) == kind::prefetched);
    }
  }

  SECTION("reading is carved out of preparing-or-higher")
  {
    // The two rules overlap here and the more specific one wins: the prefetch
    // did land, the consumer is simply already on this split waiting for it.
    CHECK(classify(false, true, true, scan_stage::reading) == kind::wait_for_prefetch);
    CHECK(classify(false, true, true, scan_stage::preparing) == kind::skipped_fell_behind);
    CHECK(classify(false, true, true, scan_stage::disposed) == kind::skipped_fell_behind);
  }

  SECTION("nothing issued is a miss only if the consumer moved on")
  {
    CHECK(classify(false, true, /*issued_io=*/false, scan_stage::queued) == kind::nothing_to_issue);
    CHECK(classify(false, true, false, scan_stage::preparing) == kind::skipped_fell_behind);
    CHECK(classify(false, true, false, scan_stage::reading) == kind::skipped_fell_behind);
  }
}

TEST_CASE("every outcome kind has its own slot", "[scan_manager][readahead][counters]")
{
  // The array is indexed by the enum cast to an integer, so a kind added
  // without bumping klast would silently share -- or overrun -- a slot.
  using kind = sirius::scan_manager::prefetch_outcome_kind;
  sirius::scan_manager::readahead_counters c;

  for (auto k : {kind::prefetched,
                 kind::wait_for_prefetch,
                 kind::skipped_memory_pressure,
                 kind::skipped_fell_behind,
                 kind::nothing_to_issue}) {
    REQUIRE(c.outcome(k) == 0);
    c.record(k);
    CHECK(c.outcome(k) == 1);
  }
}

TEST_CASE("a fresh manager reports an all-zero readahead summary",
          "[scan_manager][readahead][counters]")
{
  using kind = sirius::scan_manager::prefetch_outcome_kind;
  auto sm    = make_event_publisher();
  readahead_scan_manager m{*sm, 4};
  auto const& c = m.counters();

  CHECK(c.outcome(kind::prefetched) == 0);
  CHECK(c.outcome(kind::wait_for_prefetch) == 0);
  CHECK(c.outcome(kind::skipped_memory_pressure) == 0);
  CHECK(c.outcome(kind::skipped_fell_behind) == 0);
  CHECK(c.outcome(kind::nothing_to_issue) == 0);

  auto const line = m.summary();
  INFO(line);
  CHECK(line.find("issued=0[prefetched=0 wait_for_prefetch=0]") != std::string::npos);
  CHECK(line.find("skipped=0[memory_pressure=0 fell_behind=0 nothing_to_issue=0]") !=
        std::string::npos);
  CHECK(line.find("executor_reads=0[borrowed=0]") != std::string::npos);
}

// ===========================================================================
// a zero budget is not built at all
// ===========================================================================

namespace {
std::shared_ptr<const sirius::memory::topology_index> single_gpu_index()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

/// A query with no operators at all.  prepare_for_query settles the readahead
/// before it looks for scan operators, so this exercises that decision on its
/// own without needing a real parquet file behind a scan.
class empty_query {
 public:
  empty_query()
  {
    auto const id = sirius::make_query_id(2);
    _query        = std::make_unique<sirius::planner::query>(
      std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>>{},
      _telemetry->context(),
      id,
      sirius::telemetry::query_telemetry_info{
        _telemetry->engine_id(), _telemetry->worker_id(), id});
  }

  [[nodiscard]] const sirius::planner::query& get() const { return *_query; }

 private:
  std::shared_ptr<const sirius::telemetry::telemetry_context> _telemetry =
    sirius::test::make_test_telemetry_context();
  std::unique_ptr<sirius::planner::query> _query;
};

scan_manager_config config_with_readahead_budget(std::size_t budget)
{
  scan_manager_config cfg;
  cfg.thread_pool.num_threads = 2;
  cfg.uring_n_reactors        = 1;
  cfg.cache.mode              = cache_mode::sirius;
  cfg.max_readahead_scans     = budget;
  cfg.apply_cache_mode();
  return cfg;
}
}  // namespace

TEST_CASE("a zero readahead budget builds no manager for the query", "[scan_manager][readahead]")
{
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index();
  empty_query query;

  // Control: a usable budget does build one, so the check below is about the
  // budget rather than about the query having nothing to scan.
  {
    sirius::scan_manager::sirius_scan_manager manager{
      config_with_readahead_budget(4), *memory, topology};
    manager.prepare_for_query(query.get(), false, std::vector<int>{0});
    CHECK(manager.has_readahead_for_testing());
  }

  // A zero budget must leave it unbuilt.  Building one and merely not starting
  // its worker still subscribes it to the publisher, which then buffers one
  // event per deployed task in a mailbox nothing drains -- for the whole query.
  {
    sirius::scan_manager::sirius_scan_manager manager{
      config_with_readahead_budget(0), *memory, topology};
    manager.prepare_for_query(query.get(), false, std::vector<int>{0});
    CHECK_FALSE(manager.has_readahead_for_testing());
  }
}

TEST_CASE("the kvikIO backend builds no readahead however the cache is configured",
          "[scan_manager][readahead]")
{
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index();
  empty_query query;

  // A prefetching cache is asked for and the readahead is left to the backend.
  // The kvikIO ioctx cannot use that cache, so it is dropped before backend
  // selection rather than merely publishing a zero: none of the depths the
  // sibling backend configs still carry, nor the pipeline width an
  // opportunistic strategy would schedule against, can reach the plan.
  scan_manager_config cfg;
  cfg.thread_pool.num_threads = 2;
  cfg.uring_n_reactors        = 1;
  cfg.backend                 = sirius::scan_manager::io_backend::kvikio;
  cfg.cache.mode              = cache_mode::sirius;
  cfg.pipeline_width          = PIPELINE_THREADS;
  cfg.apply_cache_mode();
  REQUIRE(cfg.uring.n_max_concurrent_scans > 0);
  REQUIRE(cfg.rest.n_max_concurrent_scans > 0);
  REQUIRE_FALSE(cfg.max_readahead_scans.has_value());
  REQUIRE_FALSE(cfg.readahead_strategy.has_value());

  sirius::scan_manager::sirius_scan_manager manager{cfg, *memory, topology};
  REQUIRE(manager.io_ctx() != nullptr);
  REQUIRE_FALSE(manager.io_ctx()->can_use_prefetching_cache());

  manager.prepare_for_query(query.get(), false, std::vector<int>{0});
  CHECK_FALSE(manager.has_readahead_for_testing());
}
