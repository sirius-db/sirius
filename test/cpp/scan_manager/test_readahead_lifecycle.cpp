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
#include "io/kvikio/config.hpp"
#include "io/rest/config.hpp"
#include "io/uring/config.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/gatekeeper.hpp"
#include "scan_manager/readahead_scan_manager.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>

using sirius::io::cache::cache_mode;
using sirius::io::cache::eviction_policy;
using sirius::io::cache::scan_stage;
using sirius::scan_manager::gatekeeper;
using sirius::scan_manager::readahead_scan_manager;

namespace {
/// A live stage manager for the readahead under test to register its mailbox
/// with.  Held by shared_ptr because the listener keeps only a weak reference,
/// and declared before the readahead in each test so it outlives it.
auto make_stage_manager() { return std::make_shared<sirius::exec::query_stage_manager>(); }
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
  // kvikIO drives its own process-global task pool, so it opts out.
  CHECK(sirius::io::kvikio_config{}.n_max_concurrent_scans == 0);
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

TEST_CASE("a zero budget means the backend opted out and no worker runs",
          "[scan_manager][readahead]")
{
  auto sm = make_stage_manager();
  readahead_scan_manager m{*sm, 0};
  m.start();
  CHECK_FALSE(m.is_running());

  // stop() on a manager that never started must be a no-op, not a hang.
  m.stop();
  CHECK_FALSE(m.is_running());
}

TEST_CASE("start runs a worker and stop joins it", "[scan_manager][readahead]")
{
  auto sm = make_stage_manager();
  readahead_scan_manager m{*sm, 4};
  REQUIRE_FALSE(m.is_running());

  m.start();
  CHECK(m.is_running());

  m.stop();
  CHECK_FALSE(m.is_running());
}

TEST_CASE("start and stop are idempotent", "[scan_manager][readahead]")
{
  auto sm = make_stage_manager();
  readahead_scan_manager m{*sm, 4};

  m.start();
  m.start();  // already running -- must not spawn a second worker
  CHECK(m.is_running());

  m.stop();
  m.stop();  // already stopped
  CHECK_FALSE(m.is_running());
}

TEST_CASE("the destructor stops a running worker", "[scan_manager][readahead]")
{
  // The interesting failure here is a hang, not a wrong value: a worker parked
  // on the gate with nothing to wake it would never be joined.
  auto sm = make_stage_manager();
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
  auto sm = make_stage_manager();
  readahead_scan_manager m{*sm, 4};
  m.start(sirius::scan_manager::prefetch_strategy::opportunistic);
  REQUIRE(m.is_running());

  auto const before = std::chrono::steady_clock::now();
  m.stop();
  CHECK(std::chrono::steady_clock::now() - before < std::chrono::milliseconds{150});
}

TEST_CASE("update is safe on a manager that was never started", "[scan_manager][readahead]")
{
  auto sm = make_stage_manager();
  readahead_scan_manager m{*sm, 4};
  m.update_scan_state(7, nullptr, scan_stage::reading);
  m.update_scan_state(7, nullptr, scan_stage::disposed);
  CHECK_FALSE(m.is_running());
}

TEST_CASE("update tolerates a null split", "[scan_manager][readahead]")
{
  // A resident cached batch has no scan_info and reports a null task.
  auto sm = make_stage_manager();
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

TEST_CASE("a stopped manager can be restarted", "[scan_manager][readahead]")
{
  auto sm = make_stage_manager();
  readahead_scan_manager m{*sm, 4};
  m.start();
  m.stop();
  REQUIRE_FALSE(m.is_running());

  // The stop_source is replaced on start, so the second worker is not born
  // already-stopped -- and the gate is re-armed rather than left shut by the
  // first stop().
  m.start();
  CHECK(m.is_running());
  m.update_scan_state(1, nullptr, scan_stage::reading);
  m.stop();
  CHECK_FALSE(m.is_running());
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
  auto sm    = make_stage_manager();
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
