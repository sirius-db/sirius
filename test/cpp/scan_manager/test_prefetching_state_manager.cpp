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

// TODO(phase4): two things change here once the implementations land — drop the [.] tag, and
// restore the broad [scan_manager] tag alongside [prefetch_api].
//
// Nearly every method here is declared noexcept — including the constructor — and their Phase-1
// bodies throw, so an unhidden case would abort the whole test binary instead of failing. A Catch2
// test spec *includes* hidden cases, so the broad tag stays off until then: with it, running
// `sirius_unittest "[scan_manager]"` would pull these in and abort.
//
// The per-query prefetch bookkeeping. Two properties are what this file exists to pin:
//   - the counters are complete. A rung is recorded above scan_operator_input's metadata check, so
//     a fully-pinned query — every split resident, no datasource, no IO — still reports the ladder
//     it climbed instead of zeros.
//   - a straggler is harmless. A split can outlive its query (~split_connector runs when the
//     pipelines are destroyed, which is before the scan manager is reset, and a task can still be
//     in flight on a GPU executor thread), so a decrement arriving after clean_up must not corrupt
//     anything. That is also why the scan manager builds a fresh instance per query rather than
//     resetting a long-lived one.
//
// GPU-free and cache-free: the class touches no CUDA, no cucascade and no datasource.

#include "planner/query.hpp"
#include "query_id.hpp"
#include "scan_manager/prefetching_state_manager.hpp"
#include "utils/telemetry_utils.hpp"

#include <catch.hpp>
#include <io/cache/types.hpp>

#include <cstddef>
#include <cstdint>

namespace {

using sirius::io::cache::prefetching_stage;
using sirius::scan_manager::prefetching_state_manager;

prefetching_state_manager::config test_config()
{
  return {.memory_threshold = 1024, .max_concurrent_scan = 4};
}

/// A query carrying nothing but an id — which is all prepare_for_query is allowed to keep, since
/// planner::query is destroyed before the scan manager is reset.
sirius::planner::query make_query(std::uint32_t id)
{
  auto telemetry      = sirius::test::make_test_telemetry_context();
  const auto query_id = sirius::make_query_id(id);
  return {duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>>{},
          telemetry->context(),
          query_id,
          sirius::telemetry::query_telemetry_info{
            telemetry->engine_id(), telemetry->worker_id(), query_id}};
}

}  // namespace

TEST_CASE("a fresh manager reports zero counters", "[.][prefetch_api][prefetching_state_manager]")
{
  prefetching_state_manager manager{test_config()};
  auto const counters = manager.snapshot();

  CHECK(counters.n_inputs_created == 0);
  CHECK(counters.n_inputs_disposed == 0);
  CHECK(counters.n_metadata_created == 0);
  CHECK(counters.n_task_queued == 0);
  CHECK(counters.n_task_prepared == 0);
  CHECK(counters.n_task_completed == 0);
  CHECK(counters.n_live == 0);
  CHECK(sirius::value_of(manager.query_id()) == 0);
}

TEST_CASE("the manager reports the tunables it was constructed with",
          "[.][prefetch_api][prefetching_state_manager]")
{
  prefetching_state_manager manager{test_config()};

  CHECK(manager.get_config().memory_threshold == 1024);
  CHECK(manager.get_config().max_concurrent_scan == 4);
  CHECK_FALSE(manager.summary().empty());
}

TEST_CASE("update maps each ladder rung to its own counter",
          "[.][prefetch_api][prefetching_state_manager]")
{
  prefetching_state_manager manager{test_config()};

  SECTION("metadata_created")
  {
    manager.update(prefetching_stage::metadata_created);
    auto const counters = manager.snapshot();
    CHECK(counters.n_metadata_created == 1);
    CHECK(counters.n_task_queued == 0);
    CHECK(counters.n_task_prepared == 0);
    CHECK(counters.n_task_completed == 0);
  }

  SECTION("task_queued")
  {
    manager.update(prefetching_stage::task_queued);
    auto const counters = manager.snapshot();
    CHECK(counters.n_metadata_created == 0);
    CHECK(counters.n_task_queued == 1);
    CHECK(counters.n_task_prepared == 0);
    CHECK(counters.n_task_completed == 0);
  }

  SECTION("task_preprocessing")
  {
    manager.update(prefetching_stage::task_preprocessing);
    auto const counters = manager.snapshot();
    CHECK(counters.n_metadata_created == 0);
    CHECK(counters.n_task_queued == 0);
    CHECK(counters.n_task_prepared == 1);
    CHECK(counters.n_task_completed == 0);
  }

  SECTION("disposable")
  {
    manager.update(prefetching_stage::disposable);
    auto const counters = manager.snapshot();
    CHECK(counters.n_metadata_created == 0);
    CHECK(counters.n_task_queued == 0);
    CHECK(counters.n_task_prepared == 0);
    CHECK(counters.n_task_completed == 1);
  }

  SECTION("the none rung is ignored")
  {
    // `none` is not a rung: io_context uses it to mean "this backend never wants prefetch
    // activated", and it must not be mistaken for progress.
    manager.update(prefetching_stage::none);
    auto const counters = manager.snapshot();
    CHECK(counters.n_metadata_created == 0);
    CHECK(counters.n_task_queued == 0);
    CHECK(counters.n_task_prepared == 0);
    CHECK(counters.n_task_completed == 0);
  }

  SECTION("a rung climbed twice is counted twice")
  {
    manager.update(prefetching_stage::task_queued);
    manager.update(prefetching_stage::task_queued);
    CHECK(manager.snapshot().n_task_queued == 2);
  }
}

TEST_CASE("the live gauge tracks construction and disposal",
          "[.][prefetch_api][prefetching_state_manager]")
{
  prefetching_state_manager manager{test_config()};

  constexpr std::size_t kCreated  = 5;
  constexpr std::size_t kDisposed = 2;
  for (std::size_t i = 0; i < kCreated; ++i) {
    manager.on_input_created();
  }
  for (std::size_t i = 0; i < kDisposed; ++i) {
    manager.on_input_disposed();
  }

  auto const counters = manager.snapshot();
  // The two totals are monotonic; only the gauge moves in both directions.
  CHECK(counters.n_inputs_created == kCreated);
  CHECK(counters.n_inputs_disposed == kDisposed);
  CHECK(counters.n_live == static_cast<std::int64_t>(kCreated - kDisposed));
}

TEST_CASE("prepare_for_query zeroes the counters and binds the query id",
          "[.][prefetch_api][prefetching_state_manager]")
{
  prefetching_state_manager manager{test_config()};
  manager.on_input_created();
  manager.update(prefetching_stage::metadata_created);
  REQUIRE(manager.snapshot().n_metadata_created == 1);

  auto const query = make_query(4242);
  manager.prepare_for_query(query);

  auto const counters = manager.snapshot();
  CHECK(counters.n_inputs_created == 0);
  CHECK(counters.n_inputs_disposed == 0);
  CHECK(counters.n_metadata_created == 0);
  CHECK(counters.n_task_queued == 0);
  CHECK(counters.n_task_prepared == 0);
  CHECK(counters.n_task_completed == 0);
  CHECK(counters.n_live == 0);
  CHECK(manager.query_id() == query.query_id());
}

TEST_CASE("counters survive disposal after clean_up",
          "[.][prefetch_api][prefetching_state_manager]")
{
  // The straggler-split case, and the reason each query gets a fresh instance. ~split_connector
  // runs when the query's pipelines are destroyed, which is before the scan manager is reset, and
  // a task can still be in flight on a GPU executor thread — so a decrement lands after clean_up.
  // It must be harmless: nobody reads these counters again.
  prefetching_state_manager manager{test_config()};
  auto const query = make_query(7);
  manager.prepare_for_query(query);
  manager.on_input_created();
  manager.on_input_created();
  manager.on_input_disposed();

  REQUIRE_NOTHROW(manager.clean_up());

  REQUIRE_NOTHROW(manager.on_input_disposed());
  REQUIRE_NOTHROW(manager.update(prefetching_stage::disposable));

  auto const counters = manager.snapshot();
  CHECK(counters.n_inputs_created == 2);
  CHECK(counters.n_inputs_disposed == 2);
  CHECK(counters.n_live == 0);
  CHECK(counters.n_task_completed == 1);
}
