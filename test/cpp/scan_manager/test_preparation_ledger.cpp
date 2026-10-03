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
#include "scan_manager/preparation_ledger.hpp"
#include "scan_manager/preparation_test_support.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <array>
#include <future>
#include <limits>
#include <thread>

using namespace sirius::scan_manager;
using sirius::scan_manager::test::test_reservation_provider;
namespace {
auto const host = memory_space_id(cucascade::memory::Tier::HOST, 0);
std::array<scan_envelope, 1> sample() { return {scan_envelope{{{8, 8, 16, 20}}, 0, 0, true}}; }
}  // namespace
TEST_CASE("Admission uses the one actual grant and releases all unsuccessful grants",
          "[r2b][ledger]")
{
  test_reservation_provider provider;
  SECTION("valid grant")
  {
    preparation_ledger ledger(provider);
    auto d = ledger.admit(sample(), host);
    CHECK(d.deferred);
    CHECK(d.sigma_retained == 20);
    CHECK(d.w == 32);
    CHECK(d.c_obtained == 64);
    CHECK(d.n_permits == 1);
    CHECK(provider.outstanding() == 1);
    CHECK_THROWS_AS(ledger.admit(sample(), host), std::logic_error);
    CHECK(provider.seen->requests == 1);
  }
  SECTION("short grant")
  {
    provider.grant_bytes = 51;
    preparation_ledger ledger(provider);
    CHECK_FALSE(ledger.admit(sample(), host).deferred);
    CHECK(provider.outstanding() == 0);
  }
  SECTION("null grant")
  {
    provider.return_null = true;
    preparation_ledger ledger(provider);
    CHECK_FALSE(ledger.admit(sample(), host).deferred);
    CHECK(provider.outstanding() == 0);
  }
  CHECK(provider.outstanding() == 0);
}
TEST_CASE("Admission sums retained results across scans and accounts for allocation granularity",
          "[r2b][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  auto s = sample()[0];
  SECTION("two scans exceed C although each fits")
  {
    std::array scans{s, s};
    auto d = ledger.admit(scans, host);
    CHECK_FALSE(d.deferred);
    CHECK(d.sigma_retained == 40);
    CHECK(d.w == 32);
  }
  SECTION("physical rounding is included before the request")
  {
    provider.quantum = 16;
    auto d           = ledger.admit(sample(), host);
    CHECK_FALSE(d.deferred);
    CHECK(d.sigma_retained == 32);
    CHECK(d.w == 48);
  }
  CHECK(provider.outstanding() == 0);
}
TEST_CASE("Unqualified bounds and overflow route to legacy before any request", "[r2b][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  auto s = sample();
  SECTION("missing proof") { s[0].qualified = false; }
  SECTION("overflow") { s[0].units[0].blob = std::numeric_limits<uint64_t>::max(); }
  CHECK_FALSE(ledger.admit(s, host).deferred);
  CHECK(provider.seen->requests == 0);
  CHECK_FALSE(footer_envelope(1024).has_value());
  CHECK_FALSE(roaring_envelope(512, 20).has_value());
}
TEST_CASE("No DV work does not create a W permit or divide by zero", "[r2b][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{}, 0, 0, true}};
  auto d = ledger.admit(scans, host);
  CHECK(d.deferred);
  CHECK(d.w == 0);
  CHECK(d.n_permits == 0);
  CHECK(provider.seen->requests == 0);
  CHECK_FALSE(ledger.acquire_permit({1, 1}));
}
TEST_CASE("Whole W remains held until real temporary buffers are freed", "[r2b][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  REQUIRE(ledger.admit(sample(), host).deferred);
  ledger.register_unit({1, 1}, 20);
  ledger.register_unit({1, 2}, 0);
  auto permit = ledger.acquire_permit({1, 1});
  REQUIRE(permit);
  CHECK_FALSE(ledger.acquire_permit({1, 1}));
  CHECK_FALSE(ledger.acquire_permit({1, 2}));
  auto allocator = ledger.allocator(permit);
  auto temporary = allocator.allocate(32);
  auto retained  = allocator.allocate_retained(20);
  permit.release();
  CHECK(ledger.permits_in_flight() == 1);
  CHECK_FALSE(ledger.acquire_permit({1, 2}));
  CHECK_THROWS_AS(allocator.allocate(1), std::logic_error);
  temporary.release();
  CHECK(ledger.permits_in_flight() == 0);
  CHECK(ledger.charged_bytes({1, 1}) == 20);
  CHECK(ledger.acquire_permit({1, 2}));
  retained.release();
  CHECK(ledger.charged_bytes({1, 1}) == 0);
  CHECK(provider.seen->allocated_bytes == 0);
  CHECK(provider.seen->requests == 1);
}
TEST_CASE("Allocation bounds fail before touching backing and retain distinct failure causes",
          "[r2b][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  REQUIRE(ledger.admit(sample(), host).deferred);
  ledger.register_unit({1, 1}, 20);
  auto permit    = ledger.acquire_permit({1, 1});
  auto allocator = ledger.allocator(permit);
  CHECK_THROWS_AS(allocator.allocate(33), preparation_resource_error);
  CHECK(provider.seen->allocations == 0);
  CHECK(ledger.bound_exceeded() == 1);
  CHECK(ledger.allocation_failures() == 0);
  provider.seen->fail_allocation = true;
  try {
    allocator.allocate(16);
    FAIL("expected allocation failure");
  } catch (preparation_resource_error const& e) {
    CHECK_FALSE(e.bound_exceeded);
    CHECK(e.original != nullptr);
    CHECK(e.cause == sirius::transparent::late_failure_cause::resource);
  }
  CHECK(ledger.bound_exceeded() == 1);
  CHECK(ledger.allocation_failures() == 1);
  CHECK(ledger.charged_bytes({1, 1}) == 0);
}
TEST_CASE("Results keep backing after plan and attempt owners finish", "[r2b][ledger]")
{
  test_reservation_provider provider;
  auto ledger = std::make_unique<preparation_ledger>(provider);
  auto d      = ledger->admit(sample(), host);
  REQUIRE(d.deferred);
  preparation_admission owner(std::move(ledger), d);
  auto attempt = owner.consume();
  CHECK_THROWS_AS(owner.consume(), std::logic_error);
  attempt->register_unit({1, 1}, 20);
  auto permit = attempt->acquire_permit({1, 1});
  auto block  = attempt->allocator(permit).allocate_retained(16);
  auto* p     = reinterpret_cast<int64_t*>(block.data());
  p[0]        = 2;
  p[1]        = 9;
  auto result =
    std::make_shared<sirius::op::scan::iceberg_delete_set const>("file", std::move(block), 2, 1);
  permit.release();
  attempt->close();
  attempt.reset();
  owner.finish();
  CHECK(provider.outstanding() == 1);
  CHECK(result->positions[1] == 9);
  result.reset();
  CHECK(provider.outstanding() == 0);
  CHECK(provider.seen->allocated_bytes == 0);
}
TEST_CASE("An unexecuted plan releases its grant and cannot be reused after finish",
          "[r2b][ledger]")
{
  test_reservation_provider provider;
  {
    auto ledger = std::make_unique<preparation_ledger>(provider);
    auto d      = ledger->admit(sample(), host);
    preparation_admission owner(std::move(ledger), d);
    CHECK(provider.outstanding() == 1);
    owner.finish();
    owner.finish();
    CHECK(provider.outstanding() == 0);
    CHECK_THROWS_AS(owner.consume(), std::logic_error);
  }
  CHECK(provider.outstanding() == 0);
}
TEST_CASE("Statement DV threshold only chooses preparation route including legacy scans",
          "[r2b][ledger]")
{
  std::array counts{scan_dv_count{1, 150 * 1024 * 1024}, scan_dv_count{2, 150 * 1024 * 1024}};
  CHECK_FALSE(statement_dv_route_allowed(counts));
  counts[1].live_positions = 1;
  CHECK(statement_dv_route_allowed(counts));
  counts[1].contract       = 1;
  counts[1].live_positions = counts[0].live_positions;
  CHECK(statement_dv_route_allowed(counts));  // repeated scan reference counted once
  counts[1].live_positions = 2;               // inconsistent evidence cannot prove the route
  CHECK_FALSE(statement_dv_route_allowed(counts));
}

TEST_CASE("Multiple W permits are independent and close rejects further admission", "[r2b][ledger]")
{
  test_reservation_provider provider;
  provider.grant_bytes = 84;
  preparation_ledger ledger(provider);
  REQUIRE(ledger.admit(sample(), host).n_permits == 2);
  ledger.register_unit({1, 1}, 20);
  ledger.register_unit({1, 2}, 0);
  ledger.register_unit({1, 3}, 0);
  auto first  = ledger.acquire_permit({1, 1});
  auto second = ledger.acquire_permit({1, 2});
  REQUIRE(first);
  REQUIRE(second);
  CHECK_FALSE(ledger.acquire_permit({1, 3}));
  auto a = ledger.allocator(first).allocate(32);
  auto b = ledger.allocator(second).allocate(32);
  CHECK(provider.seen->allocated_bytes == 64);
  ledger.close();
  CHECK_FALSE(ledger.acquire_permit({1, 3}));
  CHECK_THROWS_AS(ledger.allocator(first).allocate(1), std::logic_error);
  first.release();
  second.release();
  CHECK(ledger.permits_in_flight() == 2);
  a.release();
  CHECK(ledger.permits_in_flight() == 1);
  b.release();
  CHECK(ledger.permits_in_flight() == 0);
  CHECK(provider.seen->requests == 1);
}
TEST_CASE("HOST adapter allocates real reserved blocks and returns them after the last consumer",
          "[r2b][ledger][host]")
{
  cucascade::memory::host_memory_space_config config;
  config.numa_id              = 0;
  config.memory_capacity      = 16 * 1024;
  config.block_size           = 1024;
  config.pool_size            = 16;
  config.initial_number_pools = 1;
  cucascade::memory::memory_reservation_manager manager({config});
  auto* space = manager.get_memory_space(cucascade::memory::Tier::HOST, 0);
  REQUIRE(space);
  auto* resource    = space->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  auto initial_free = resource->get_free_blocks();
  host_reservation_provider provider(manager);
  charged_block retained;
  {
    preparation_ledger ledger(provider);
    auto d = ledger.admit(sample(), host);
    REQUIRE(d.deferred);
    CHECK(d.sigma_retained == 1024);
    CHECK(d.w == 3072);
    CHECK(manager.get_active_reservation_count() == 1);
    ledger.register_unit({1, 1}, 20);
    auto permit = ledger.acquire_permit({1, 1});
    retained    = ledger.allocator(permit).allocate_retained(20);
    CHECK(resource->get_free_blocks() == initial_free - 1);
    CHECK(ledger.charged_bytes({1, 1}) == 1024);
    permit.release();
    ledger.close();
  }
  CHECK(manager.get_active_reservation_count() == 1);
  retained.release();
  CHECK(manager.get_active_reservation_count() == 0);
  CHECK(resource->get_free_blocks() == initial_free);
}

TEST_CASE("Closing during allocation keeps the grant alive until the allocation and result exit",
          "[r2b][ledger]")
{
  struct paused_backing : reservation_backing {
    std::shared_ptr<reservation_backing> base;
    std::promise<void> entered;
    std::shared_future<void> release;
    uint64_t charge_size(uint64_t bytes) const override { return base->charge_size(bytes); }
    backing_allocation allocate(uint64_t bytes) override
    {
      entered.set_value();
      release.wait();
      return base->allocate(bytes);
    }
  };
  struct provider_type : reservation_provider {
    test_reservation_provider fake;
    std::shared_ptr<paused_backing> pause = std::make_shared<paused_backing>();
    std::optional<reservation_grant> request(memory_space_id space, uint64_t bytes) override
    {
      auto grant    = fake.request(space, bytes);
      pause->base   = std::move(grant->handle);
      grant->handle = pause;
      return grant;
    }
  } provider;
  std::promise<void> unblock;
  provider.pause->release = unblock.get_future().share();
  auto entered            = provider.pause->entered.get_future();
  auto ledger             = std::make_unique<preparation_ledger>(provider);
  REQUIRE(ledger->admit(sample(), host).deferred);
  ledger->register_unit({1, 1}, 20);
  auto permit    = ledger->acquire_permit({1, 1});
  auto allocator = ledger->allocator(permit);
  charged_block result;
  std::exception_ptr failure;
  std::jthread worker([&] {
    try {
      result = allocator.allocate_retained(16);
    } catch (...) {
      failure = std::current_exception();
    }
  });
  auto started = entered.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  if (!started) {
    unblock.set_value();
    worker.join();
    REQUIRE(started);
  }
  permit.release();
  ledger->close();
  ledger.reset();
  CHECK(provider.fake.outstanding() == 1);
  unblock.set_value();
  worker.join();
  REQUIRE_FALSE(failure);
  provider.pause.reset();
  CHECK(provider.fake.outstanding() == 1);
  result.release();
  CHECK(provider.fake.outstanding() == 0);
}

TEST_CASE("Dropping an owner closes allocation even if a worker still owns its ticket",
          "[r2b][ledger]")
{
  test_reservation_provider provider;
  auto ledger = std::make_unique<preparation_ledger>(provider);
  REQUIRE(ledger->admit(sample(), host).deferred);
  ledger->register_unit({1, 1}, 20);
  auto permit    = ledger->acquire_permit({1, 1});
  auto allocator = ledger->allocator(permit);
  ledger.reset();
  CHECK(provider.outstanding() == 1);
  CHECK_THROWS_AS(allocator.allocate(1), std::logic_error);
  permit.release();
  CHECK(provider.outstanding() == 0);
}

TEST_CASE("Large blob and small final positions still require the whole temporary peak",
          "[r2b][ledger]")
{
  test_reservation_provider provider;
  provider.grant_bytes = 76;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{64, 4, 4, 4}}, 0, 0, true}};
  auto d = ledger.admit(scans, host);
  REQUIRE(d.deferred);
  CHECK(d.sigma_retained == 4);
  CHECK(d.w == 72);
  CHECK(d.n_permits == 1);
  ledger.register_unit({1, 1}, 4);
  ledger.register_unit({1, 2}, 0);
  auto permit = ledger.acquire_permit({1, 1});
  auto blob   = ledger.allocator(permit).allocate(64);
  permit.release();
  CHECK_FALSE(ledger.acquire_permit({1, 2}));
  blob.release();
  CHECK(ledger.acquire_permit({1, 2}));
}
TEST_CASE("A fresh lowering owner is required for every execution", "[r2b][ledger]")
{
  test_reservation_provider provider;
  for (int execution = 0; execution < 2; ++execution) {
    auto ledger = std::make_unique<preparation_ledger>(provider);
    auto d      = ledger->admit(sample(), host);
    preparation_admission owner(std::move(ledger), d);
    auto attempt = owner.consume();
    CHECK_THROWS_AS(owner.consume(), std::logic_error);
    attempt.reset();
    CHECK(provider.outstanding() == 0);
  }
  CHECK(provider.seen->requests == 2);
}

TEST_CASE("Retained reallocation accounts for the simultaneous old and new buffers",
          "[r2b][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  REQUIRE(ledger.admit(sample(), host).deferred);
  ledger.register_unit({1, 1}, 20);
  auto permit    = ledger.acquire_permit({1, 1});
  auto allocator = ledger.allocator(permit);
  auto old       = allocator.allocate_retained(16);
  CHECK_THROWS_AS(allocator.allocate_retained(8), preparation_resource_error);
  CHECK(provider.seen->allocations == 1);
  CHECK(ledger.charged_bytes({1, 1}) == 16);
  old.release();
  auto replacement = allocator.allocate_retained(8);
  CHECK(ledger.charged_bytes({1, 1}) == 8);
}

TEST_CASE("Retained positions without a preparation work bound stay on the original path",
          "[r2b][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{0, 0, 0, 20}}, 0, 0, true}};
  auto decision = ledger.admit(scans, host);
  CHECK_FALSE(decision.deferred);
  CHECK(provider.seen->requests == 0);
}

TEST_CASE("Destroying an unexecuted admission releases its reservation without explicit finish",
          "[r2b][ledger]")
{
  test_reservation_provider provider;
  {
    auto ledger   = std::make_unique<preparation_ledger>(provider);
    auto decision = ledger->admit(sample(), host);
    REQUIRE(decision.deferred);
    preparation_admission plan(std::move(ledger), decision);
    CHECK(provider.outstanding() == 1);
  }
  CHECK(provider.outstanding() == 0);
}

TEST_CASE("HOST reservation failure does not retry or leave a partial grant", "[r2b][ledger][host]")
{
  cucascade::memory::host_memory_space_config config;
  config.numa_id              = 0;
  config.memory_capacity      = 8 * 1024;
  config.block_size           = 1024;
  config.pool_size            = 8;
  config.initial_number_pools = 1;
  bool occupy                 = true;
  SECTION("another reservation consumes the available capacity") {}
  SECTION("minimum grant exceeds the space reservation limit")
  {
    config.memory_capacity = 4 * 1024;
    config.pool_size       = 4;
    occupy                 = false;
  }
  cucascade::memory::memory_reservation_manager manager({config});
  auto* space    = manager.get_memory_space(cucascade::memory::Tier::HOST, 0);
  auto occupying = occupy ? space->make_reservation_or_null(4 * 1024) : nullptr;
  REQUIRE(bool(occupying) == occupy);
  host_reservation_provider provider(manager);
  preparation_ledger ledger(provider);
  CHECK_FALSE(ledger.admit(sample(), host).deferred);
  CHECK(manager.get_active_reservation_count() == (occupy ? 1 : 0));
  CHECK_THROWS_AS(ledger.admit(sample(), host), std::logic_error);
  occupying.reset();
  CHECK(manager.get_active_reservation_count() == 0);
}

TEST_CASE("Failure to construct the lowering grant selects the original path once", "[r2b][ledger]")
{
  struct failing_provider : reservation_provider {
    int requests = 0;
    std::optional<reservation_grant> request(memory_space_id, uint64_t) override
    {
      ++requests;
      throw std::bad_alloc();
    }
  } provider;
  preparation_ledger ledger(provider);
  auto decision = ledger.admit(sample(), host);
  CHECK_FALSE(decision.deferred);
  CHECK(decision.reason == admission_reason::no_grant);
  CHECK(ledger.granted_capacity() == 0);
  CHECK(provider.requests == 1);
  CHECK_THROWS_AS(ledger.admit(sample(), host), std::logic_error);
  CHECK(provider.requests == 1);
}
