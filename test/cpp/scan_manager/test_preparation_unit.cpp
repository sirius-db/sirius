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
#include "op/scan/iceberg_delete_set.hpp"
#include "pipeline/completion_handler.hpp"
#include "scan_manager/preparation.hpp"
#include "scan_manager/preparation_test_support.hpp"

#include <cudf/io/parquet_schema.hpp>

#include <array>
#include <barrier>
#include <thread>

using namespace sirius::scan_manager;
namespace {
required_input_set needs(required_input input)
{
  required_input_set set;
  set.set(static_cast<size_t>(input));
  return set;
}
}  // namespace
TEST_CASE("A pending delete set must not be interpreted as no deletes",
          "[scan_preparation][preparation_unit]")
{
  preparation_unit unit({3, 4}, needs(required_input::delete_set));
  CHECK(unit.record().state == unit_state::pending);
  CHECK_FALSE(unit.record().deps);
  CHECK_THROWS_AS(unit.complete_input(delete_set_input{}), std::invalid_argument);
  REQUIRE(unit.complete_input(
    delete_set_input{std::make_shared<sirius::op::scan::iceberg_delete_set const>("file")}));
  auto ready = unit.record();
  CHECK(ready.state == unit_state::ready);
  REQUIRE(ready.deps->delete_set);
  CHECK(ready.deps->delete_set->positions.empty());
  CHECK_FALSE(unit.cancel());
  CHECK_FALSE(unit.complete_input(delete_set_input{}));  // late/duplicate ignored
  CHECK(unit.record().deps == ready.deps);
}
TEST_CASE("Required inputs are derived from actual scan checks and arrive in any order",
          "[scan_preparation][preparation_unit]")
{
  sirius::op::scan::later_check_set checks;
  checks.set(static_cast<size_t>(sirius::op::scan::later_check::segments_per_range));
  checks.set(static_cast<size_t>(sirius::op::scan::later_check::key_held));
  auto required = required_inputs(checks, true);
  CHECK_FALSE(required[static_cast<size_t>(required_input::footer)]);
  preparation_unit unit({1, 2}, required);
  REQUIRE(unit.complete_input(checkpoint_input{7}));
  CHECK(unit.record().state == unit_state::pending);
  REQUIRE(unit.complete_input(
    delete_set_input{std::make_shared<sirius::op::scan::iceberg_delete_set const>("file")}));
  CHECK(unit.record().state == unit_state::pending);
  REQUIRE(unit.complete_input(
    segments_input{std::make_shared<sirius::op::scan::physical_profile_table>()}));
  CHECK(unit.record().state == unit_state::ready);
  CHECK(unit.record().deps->checkpoint_iteration == 7);
  CHECK_FALSE(unit.complete_input(checkpoint_input{8}));
  CHECK(unit.record().deps->checkpoint_iteration == 7);
}
TEST_CASE("Preparation terminal races commit exactly once", "[scan_preparation][preparation_unit]")
{
  for (int iteration = 0; iteration < 50; ++iteration) {
    preparation_unit unit({1, 1}, needs(required_input::delete_set));
    std::barrier rendezvous(3);
    bool completed = false, cancelled = false;
    std::jthread first([&] {
      rendezvous.arrive_and_wait();
      completed = unit.complete_input(
        delete_set_input{std::make_shared<sirius::op::scan::iceberg_delete_set const>("file")});
    });
    std::jthread second([&] {
      rendezvous.arrive_and_wait();
      cancelled = unit.cancel();
    });
    rendezvous.arrive_and_wait();
    first.join();
    second.join();
    CHECK(completed != cancelled);
    CHECK(unit.record().state != unit_state::pending);
  }
}
TEST_CASE("Failure carriers preserve the original exception and reject invalid verdict reasons",
          "[scan_preparation][preparation_unit]")
{
  preparation_failure failure{sirius::transparent::late_failure_cause::resource,
                              {},
                              "HOST allocation",
                              std::make_exception_ptr(std::bad_alloc())};
  preparation_unit unit({1, 1}, needs(required_input::footer));
  REQUIRE(unit.fail(failure));
  CHECK(unit.record().failure->original == failure.original);
  CHECK_FALSE(unit.cancel());
  CHECK_FALSE(unit.complete_input(footer_input{}));
  failure.reason = sirius::op::scan::verdict_reason::iceberg_delete_corrupt;
  CHECK_THROWS_AS(failure.validate(), std::invalid_argument);
  CHECK_THROWS_AS(classify_failure(failure), std::invalid_argument);
  failure.cause = sirius::transparent::late_failure_cause::physical_input;
  CHECK_NOTHROW(failure.validate());
}
TEST_CASE("Typed preparation errors retain their cause and wake before GPU completion",
          "[scan_preparation][preparation_unit]")
{
  sirius::pipeline::completion_handler completion;
  auto future = completion.get_awaitable();
  completion.begin_preparation();
  auto original = std::make_exception_ptr(std::bad_alloc());
  completion.report_error(preparation_failure{
    sirius::transparent::late_failure_cause::resource, {}, "allocation inside bound", original});
  REQUIRE(future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready);
  CHECK_THROWS_AS(future.get(), std::bad_alloc);
  CHECK(completion.failure().cause == sirius::transparent::late_failure_cause::resource);
  completion.mark_completed();
  completion.close_preparation_inputs();
  completion.preparation_quiescent();
  CHECK(completion.has_error());
}
TEST_CASE("Cancellation and failure wake independently of preparation closure",
          "[scan_preparation][preparation_unit]")
{
  for (int iteration = 0; iteration < 50; ++iteration) {
    sirius::pipeline::completion_handler completion;
    auto future = completion.get_awaitable();
    completion.begin_preparation();
    completion.mark_completed();
    std::barrier rendezvous(3);
    std::jthread cancel([&] {
      rendezvous.arrive_and_wait();
      completion.report_error("cancelled");
    });
    std::jthread fail([&] {
      rendezvous.arrive_and_wait();
      completion.report_error(
        preparation_failure{sirius::transparent::late_failure_cause::resource, {}, "resource", {}});
    });
    rendezvous.arrive_and_wait();
    cancel.join();
    fail.join();
    REQUIRE(future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready);
    CHECK_THROWS(future.get());
    CHECK(completion.has_error());
    auto winner = completion.failure();
    completion.close_preparation_inputs();
    completion.preparation_quiescent();
    CHECK(completion.failure().cause == winner.cause);
    CHECK(completion.failure().detail == winner.detail);
  }
}

TEST_CASE("Typed physical failure keeps the terminal winner's verdict reason",
          "[scan_preparation][preparation_unit]")
{
  sirius::pipeline::completion_handler completion;
  auto future = completion.get_awaitable();
  completion.begin_preparation();
  completion.report_error(preparation_failure{
    sirius::transparent::late_failure_cause::physical_input,
    sirius::op::scan::verdict_reason::iceberg_delete_corrupt,
    "corrupt DV",
    std::make_exception_ptr(std::logic_error("original footer contradiction"))});
  completion.report_error(
    preparation_failure{sirius::transparent::late_failure_cause::resource, {}, "later OOM", {}});
  CHECK_THROWS_WITH(future.get(), "original footer contradiction");
  CHECK(completion.failure_reason() == sirius::op::scan::verdict_reason::iceberg_delete_corrupt);
  CHECK(completion.failure().cause == sirius::transparent::late_failure_cause::physical_input);
}
TEST_CASE("Footer completion retains its independent metadata and approval owners",
          "[scan_preparation][preparation_unit]")
{
  unit_record snapshot;
  auto footer = std::make_shared<cudf::io::parquet::FileMetaData>();
  std::weak_ptr<cudf::io::parquet::FileMetaData> weak = footer;
  {
    preparation_unit unit({1, 1}, needs(required_input::footer));
    CHECK_THROWS_AS(unit.complete_input(footer_input{footer, {}, {}}), std::invalid_argument);
    REQUIRE(unit.complete_input(footer_input{
      std::move(footer), std::make_shared<sirius::op::scan::parquet_input_approval const>(), {}}));
    snapshot = unit.record();
    CHECK(snapshot.state == unit_state::ready);
    CHECK_FALSE(weak.expired());
    CHECK_FALSE(unit.complete_input(footer_input{}));
  }
  CHECK_FALSE(weak.expired());
  snapshot = {};
  CHECK(weak.expired());
}

TEST_CASE("Late preparation payloads are released and cannot replace a terminal result",
          "[scan_preparation][preparation_unit]")
{
  preparation_unit unit({4, 5}, needs(required_input::delete_set));
  REQUIRE(unit.cancel());
  auto value = std::make_shared<sirius::op::scan::iceberg_delete_set const>("late file");
  std::weak_ptr<sirius::op::scan::iceberg_delete_set const> weak = value;
  CHECK_FALSE(unit.complete_input(delete_set_input{std::move(value)}));
  CHECK(weak.expired());
  CHECK(unit.record().state == unit_state::cancelled);
  CHECK_FALSE(unit.record().deps);
}

TEST_CASE("Delete result validation rejects invalid positions and releases its real backing",
          "[scan_preparation][ledger][delete_set]")
{
  sirius::scan_manager::test::test_reservation_provider provider;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{8, 8, 16, 20}}, 0, 0, true}};
  REQUIRE(ledger.admit(scans, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  ledger.register_unit({1, 1}, 20);
  auto permit    = ledger.acquire_permit({1, 1});
  auto block     = ledger.allocator(permit).allocate_retained(16);
  auto positions = reinterpret_cast<int64_t*>(block.data());
  SECTION("unsorted")
  {
    positions[0] = 2;
    positions[1] = 1;
  }
  SECTION("duplicate")
  {
    positions[0] = 2;
    positions[1] = 2;
  }
  SECTION("negative")
  {
    positions[0] = -1;
    positions[1] = 2;
  }
  CHECK_THROWS_AS(sirius::op::scan::iceberg_delete_set("file", std::move(block), 2, 1),
                  std::invalid_argument);
  CHECK(ledger.charged_bytes({1, 1}) == 0);
  CHECK(provider.seen->allocated_bytes == 0);
}
