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
#include "op/sirius_physical_operator.hpp"
#include "planner/query_index.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/readahead_scan_manager.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

using sirius::io::cache::scan_stage;
using sirius::scan_manager::cache_mode;
using sirius::scan_manager::readahead_scan_manager;
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

TEST_CASE("readahead is enabled for every cache mode but none", "[scan_manager][readahead]")
{
  auto enabled_for = [](cache_mode mode) {
    scan_manager_config cfg;
    cfg.cache = mode;
    cfg.apply_cache_mode();
    return cfg.enable_readahead;
  };

  CHECK_FALSE(enabled_for(cache_mode::none));
  // `os` has no prefetching cache, but ordering scans ahead of demand still
  // warms the page cache, so readahead is on.
  CHECK(enabled_for(cache_mode::os));
  CHECK(enabled_for(cache_mode::persistent));
  CHECK(enabled_for(cache_mode::prefetch));
}

TEST_CASE("apply_cache_mode leaves the other derived knobs alone", "[scan_manager][readahead]")
{
  scan_manager_config cfg;
  cfg.cache = cache_mode::prefetch;
  cfg.apply_cache_mode();

  CHECK(cfg.enable_readahead);
  CHECK(cfg.enable_prefetch_cache);
  CHECK(cfg.prefetch_cache.dispose_on_idle);
  CHECK(cfg.uring.use_odirect);
}

// ===========================================================================
// worker lifecycle
// ===========================================================================

TEST_CASE("a zero budget means the backend opted out and no worker runs",
          "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  m.start(0);
  CHECK_FALSE(m.is_running());

  // stop() on a manager that never started must be a no-op, not a hang.
  m.stop();
  CHECK_FALSE(m.is_running());
}

TEST_CASE("start runs a worker and stop joins it", "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  REQUIRE_FALSE(m.is_running());

  m.start(4);
  CHECK(m.is_running());

  m.stop();
  CHECK_FALSE(m.is_running());
}

TEST_CASE("start and stop are idempotent", "[scan_manager][readahead]")
{
  readahead_scan_manager m;

  m.start(4);
  m.start(8);  // already running -- must not spawn a second worker
  CHECK(m.is_running());

  m.stop();
  m.stop();  // already stopped
  CHECK_FALSE(m.is_running());
}

TEST_CASE("the destructor stops a running worker", "[scan_manager][readahead]")
{
  // The interesting failure here is a hang, not a wrong value: a worker parked
  // on the condvar without a stop-aware wait would never be joined.
  auto m = std::make_unique<readahead_scan_manager>();
  m->start(4);
  REQUIRE(m->is_running());
  m.reset();
  SUCCEED("destructor joined the worker");
}

TEST_CASE("update wakes the worker without deadlocking", "[scan_manager][readahead]")
{
  // update() takes the same mutex the worker waits on, so this exercises the
  // notify-outside-the-lock path.
  readahead_scan_manager m;
  m.start(4);

  for (int i = 0; i < 200; ++i) {
    m.update_scan_state(1, nullptr, scan_stage::reading);
    m.update_scan_state(2, nullptr, scan_stage::queued);
    m.update_scan_state(1, nullptr, scan_stage::disposed);
  }

  CHECK(m.is_running());
  m.stop();
  CHECK_FALSE(m.is_running());
}

TEST_CASE("update is safe on a manager that was never started", "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  m.update_scan_state(7, nullptr, scan_stage::reading);
  m.update_scan_state(7, nullptr, scan_stage::disposed);
  CHECK_FALSE(m.is_running());
}

// ===========================================================================
// ongoing-scan accounting
// ===========================================================================

namespace {

/// A split with no file ranges: enough to be registered and reported on, and
/// the base already returns an empty fadvise list.
using test_split = sirius::op::scan::scan_info;

constexpr std::size_t OP = 1;

}  // namespace

TEST_CASE("an unprefetched split counts as ongoing only once it is reading",
          "[scan_manager][readahead]")
{
  // No worker: start() is never called, so nothing prefetches these splits out
  // from under the assertions.
  readahead_scan_manager m;
  auto split = std::make_shared<test_split>();
  m.register_scan_task(split, OP);

  CHECK(m.ongoing_scans() == 0);

  for (auto stage : {scan_stage::initialized, scan_stage::queued, scan_stage::preparing}) {
    m.update_scan_state(OP, split.get(), stage);
    // Nothing is doing IO yet -- the split has not been handed to a reader.
    CHECK(m.ongoing_scans() == 0);
  }

  m.update_scan_state(OP, split.get(), scan_stage::reading);
  CHECK(m.ongoing_scans() == 1);

  m.update_scan_state(OP, split.get(), scan_stage::disposed);
  CHECK(m.ongoing_scans() == 0);
}

TEST_CASE("splits are counted independently", "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  auto a = std::make_shared<test_split>();
  auto b = std::make_shared<test_split>();
  m.register_scan_task(a, OP);
  m.register_scan_task(b, OP);

  m.update_scan_state(OP, a.get(), scan_stage::reading);
  CHECK(m.ongoing_scans() == 1);  // b is still only registered

  m.update_scan_state(OP, b.get(), scan_stage::reading);
  CHECK(m.ongoing_scans() == 2);

  // One split disposing must not retire the other -- the whole reason update()
  // carries the split identity rather than just the operator id.
  m.update_scan_state(OP, a.get(), scan_stage::disposed);
  CHECK(m.ongoing_scans() == 1);
}

TEST_CASE("a split that is destroyed stops counting", "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  auto split = std::make_shared<test_split>();
  m.register_scan_task(split, OP);
  m.update_scan_state(OP, split.get(), scan_stage::reading);
  REQUIRE(m.ongoing_scans() == 1);

  // A split whose task went away without reporting disposed still frees its
  // slot; the weak_ptr expiring is the backstop.
  split.reset();
  CHECK(m.ongoing_scans() == 0);
}

TEST_CASE("registering the same split twice is idempotent", "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  auto split = std::make_shared<test_split>();
  m.register_scan_task(split, OP);
  m.register_scan_task(split, OP);

  m.update_scan_state(OP, split.get(), scan_stage::reading);
  CHECK(m.ongoing_scans() == 1);  // not 2
}

TEST_CASE("updates for an unknown split still record the operator stage",
          "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  auto split   = std::make_shared<test_split>();
  auto unknown = std::make_shared<test_split>();
  m.register_scan_task(split, OP);

  // A resident cached batch has no scan_info and reports a null task; it must
  // not be mistaken for one of the registered splits.
  m.update_scan_state(OP, nullptr, scan_stage::reading);
  m.update_scan_state(OP, unknown.get(), scan_stage::reading);
  CHECK(m.ongoing_scans() == 0);

  m.update_scan_state(OP, split.get(), scan_stage::reading);
  CHECK(m.ongoing_scans() == 1);
}

// ===========================================================================
// operator retirement
// ===========================================================================
//
// Retirement is one-way: prefetching_scheduler::advance() only ever moves the
// cursor forward, so an operator retired while its producer is still emitting
// splits is dropped from the prefetch order for the rest of the query.  These
// pin the rule that only mark_operator_closed() can retire an operator.

namespace {

/// Minimal concrete scan carrying only the operator id the scheduler keys on.
struct test_scan : sirius::op::sirius_physical_operator {
  explicit test_scan(std::size_t id)
    : sirius::op::sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::GPU_SCAN, {}, 0)
  {
    operator_id = id;
  }
};

/// Seeds @p m with a single pipeline-mode scan for operator @ref OP.
struct single_scan_order {
  test_scan scan{OP};
  std::vector<sirius::planner::prefetch_step> steps{
    sirius::planner::prefetch_step{&scan, 0, sirius::planner::scheduling_mode::pipeline, 1}};

  void seed(readahead_scan_manager& m) { m.prepare_for_order(steps); }
};

}  // namespace

TEST_CASE("later groups fill idle readahead budget as soon as a split is registered",
          "[scan_manager][readahead]")
{
  constexpr std::size_t FIRST = 10;
  constexpr std::size_t LATER = 20;
  test_scan first{FIRST};
  test_scan later{LATER};
  std::vector<sirius::planner::prefetch_step> steps{
    {&first, 1, sirius::planner::scheduling_mode::barrier_all, 1},
    {&later, 2, sirius::planner::scheduling_mode::barrier_all, 1},
  };

  readahead_scan_manager m;
  m.prepare_for_order(steps);
  m.start(1);

  // The preferred group has emitted nothing. Registration of a later group's
  // split must wake the worker, which should select it as lookahead rather than
  // leave the budget idle.
  auto split = std::make_shared<test_split>();
  m.register_scan_task(split, LATER);
  for (int i = 0; i < 100 && m.has_unprefetched_work(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  CHECK_FALSE(m.has_unprefetched_work());
  // Lookahead must not consume or advance the preferred group's cursor.
  auto* preferred = m.get_next_prefetching_operator();
  REQUIRE(preferred != nullptr);
  CHECK(preferred->get_operator_id() == FIRST);
  m.stop();
}

TEST_CASE("a lull between split waves does not retire the operator", "[scan_manager][readahead]")
{
  // The regression: with only one split emitted so far, "every split I know
  // about is disposed" also describes an operator whose producer is between
  // waves.  Retiring there loses every split the producer has yet to emit.
  readahead_scan_manager m;
  single_scan_order order;
  order.seed(m);

  auto first = std::make_shared<test_split>();
  m.register_scan_task(first, OP);
  m.update_scan_state(OP, first.get(), scan_stage::reading);
  m.update_scan_state(OP, first.get(), scan_stage::disposed);

  REQUIRE(m.get_next_prefetching_operator() != nullptr);

  // The second wave arrives and must still be reachable.
  auto second = std::make_shared<test_split>();
  m.register_scan_task(second, OP);
  CHECK(m.get_next_prefetching_operator() != nullptr);
}

TEST_CASE("closing the producer retires an operator whose splits are done",
          "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  single_scan_order order;
  order.seed(m);

  auto split = std::make_shared<test_split>();
  m.register_scan_task(split, OP);
  m.update_scan_state(OP, split.get(), scan_stage::disposed);
  REQUIRE(m.get_next_prefetching_operator() != nullptr);

  // Close is itself a retirement edge: the last split disposed before the
  // producer closed, so nothing else would re-evaluate depletion.
  m.mark_operator_closed(OP);
  CHECK(m.get_next_prefetching_operator() == nullptr);
}

TEST_CASE("a closed operator with live splits is not retired until they finish",
          "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  single_scan_order order;
  order.seed(m);

  auto split = std::make_shared<test_split>();
  m.register_scan_task(split, OP);
  m.update_scan_state(OP, split.get(), scan_stage::reading);

  m.mark_operator_closed(OP);
  CHECK(m.get_next_prefetching_operator() != nullptr);

  m.update_scan_state(OP, split.get(), scan_stage::disposed);
  CHECK(m.get_next_prefetching_operator() == nullptr);
}

TEST_CASE("mark_operator_closed is idempotent and tolerates unknown operators",
          "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  single_scan_order order;
  order.seed(m);

  m.mark_operator_closed(OP);
  m.mark_operator_closed(OP);
  m.mark_operator_closed(OP + 99);  // never seeded
  SUCCEED("no crash and no deadlock");
}

TEST_CASE("a stopped manager can be restarted", "[scan_manager][readahead]")
{
  readahead_scan_manager m;
  m.start(4);
  m.stop();
  REQUIRE_FALSE(m.is_running());

  // The stop_source is replaced on start, so the second worker is not born
  // already-stopped.
  m.start(4);
  CHECK(m.is_running());
  m.update_scan_state(1, nullptr, scan_stage::reading);
  m.stop();
  CHECK_FALSE(m.is_running());
}
