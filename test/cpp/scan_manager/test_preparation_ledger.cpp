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
#include "exec/scoped_dispatcher.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/iceberg_delete_set.hpp"
#include "op/scan/iceberg_dv_preparation.hpp"
#include "op/scan/iceberg_gpu_ingestible.hpp"
#include "scan_manager/preparation_fixture.hpp"
#include "scan_manager/preparation_ledger.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "scan_manager/split_provider.hpp"
#include "transparent/read_view_registry.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <array>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace sirius::scan_manager;
using sirius::scan_manager::test::test_reservation_provider;
namespace {
auto const host = memory_space_id(cucascade::memory::Tier::HOST, 0);
std::array<scan_envelope, 1> sample() { return {scan_envelope{{{{8, 8, 16}, 20}}, {}, true}}; }

struct dv_precondition_fixture {
  test_reservation_provider provider;
  std::vector<std::string> paths{"a.parquet"};
  std::shared_ptr<sirius::transparent::read_view_registry> read_views;
  std::shared_ptr<sirius::op::scan::iceberg_dv_preparation> prepared;
  std::shared_ptr<preparation_admission> admission;

  explicit dv_precondition_fixture(size_t file_count = 1)
    : read_views(std::make_shared<sirius::transparent::read_view_registry>())
  {
    using namespace sirius::op::scan;
    for (size_t i = 1; i < file_count; ++i)
      paths.push_back(std::to_string(i) + ".parquet");
    provider.grant_bytes = 64 * 1024;
    auto discovery       = std::make_shared<iceberg_delete_discovery>();
    IcebergDeleteFileEntry dv;
    dv.file_format           = "puffin";
    dv.file_path             = "a.puffin";
    dv.content_offset        = 4;
    dv.content_size_in_bytes = 44;
    dv.record_count          = 2;
    dv.file_size_in_bytes    = 400;
    for (auto const& path : paths) {
      dv.file_path            = path == paths.front() ? "a.puffin" : path + ".puffin";
      dv.referenced_data_file = path;
      discovery->deletion_vector_entries.push_back(dv);
    }
    auto ledger   = std::make_unique<preparation_ledger>(provider);
    auto envelope = iceberg_dv_preparation::envelope(*discovery, paths);
    auto decision = ledger->admit(std::array{envelope}, host, file_count);
    REQUIRE(decision.deferred);
    auto contract =
      read_views->entries().empty()
        ? allocate_scan_contract(
            *read_views, {}, 0, 1, nullptr, {}, {}, {materializer_kind::iceberg, "iceberg.v1"})
        : read_views->entries().front().contract.contract_id;
    prepared =
      std::make_shared<iceberg_dv_preparation>(contract, "table", paths, discovery, *ledger);
    admission = std::make_shared<preparation_admission>(std::move(ledger), decision);
  }

  std::shared_ptr<preparation_ledger> consume()
  {
    prepared->bind_admission(admission);
    return std::shared_ptr<preparation_ledger>(admission->consume());
  }

  std::unique_ptr<sirius::op::scan::iceberg_ingestible_table_info> info(bool deferred = true) const
  {
    using namespace sirius::op::scan;
    auto bind                    = std::make_unique<iceberg_ingestible_table_info>();
    bind->contract_id            = prepared->contract();
    bind->read_views             = read_views;
    bind->resolved_file_paths    = paths;
    bind->names                  = {"value"};
    bind->returned_types         = {sirius::logical_type::make(sirius::type_id::INTEGER)};
    bind->column_ids             = {duckdb::ColumnIndex(0)};
    bind->projected_output_count = 1;
    bind->table_path             = "table";
    if (deferred)
      bind->deferred = prepared;
    else
      bind->delete_data = std::make_shared<IcebergDeleteData>();
    return bind;
  }
};

class contract_table_info final : public sirius::op::scan::ingestible_table_info {
 public:
  bool cache_allowed = false;
  std::span<std::string const> column_names() const override { return {}; }
  std::span<std::string const> file_paths() const override { return {}; }
  std::string display_name() const override { return "contract-adapter"; }
  bool allows_cached_rows() const override { return cache_allowed; }
};

class contract_ingestible : public sirius::op::scan::gpu_ingestible {
 public:
  std::shared_ptr<sirius::scan_manager::preparation_admission> admission;
  std::shared_ptr<sirius::scan_manager::preparation_ledger> attached;
  unsigned stop_calls   = 0;
  unsigned finish_calls = 0;
  unsigned claim_calls  = 0;
  bool custom_stop_seen = false;

  std::unique_ptr<sirius::op::scan::batch_coalescer> create_batch_coalescer() const override
  {
    return nullptr;
  }
  bool has_processed_all_metadata() const override { return false; }
  metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver) override
  {
    return [](preparation_stop_token) { return std::make_unique<sirius::op::scan::scan_info>(); };
  }
  sirius::op::scan::filtered_table materialize_metadata_to_table(
    sirius::op::scan::scan_info const&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<const sirius::like_multiliteral_cache>,
    sirius::expression_query_policy) override
  {
    return {};
  }
  std::unique_ptr<cudf::table> post_filter_and_project(
    sirius::op::scan::filtered_table&&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<const sirius::like_multiliteral_cache>,
    std::unique_ptr<cudf::column>*,
    std::span<std::size_t const>,
    sirius::expression_query_policy) override
  {
    return nullptr;
  }
  sirius::op::scan::ingestible_table_info const& table_info() const noexcept override
  {
    return info;
  }
  std::vector<std::size_t> materialized_column_order() const override { return {}; }
  std::shared_ptr<sirius::scan_manager::preparation_admission> preparation_admission()
    const override
  {
    return admission;
  }
  void attach_preparation_ledger(
    std::shared_ptr<sirius::scan_manager::preparation_ledger> ledger) override
  {
    attached = std::move(ledger);
  }
  metadata_claim try_claim_split(sirius::io::ioctx_resolver) override
  {
    ++claim_calls;
    return metadata_job{[this](preparation_stop_token stop) {
      custom_stop_seen = true;
      if (stop.stop_requested())
        throw std::system_error(std::make_error_code(std::errc::operation_canceled));
      return std::make_unique<sirius::op::scan::scan_info>();
    }};
  }
  void stop_preparation() noexcept override { ++stop_calls; }
  void finish_preparation() noexcept override { ++finish_calls; }

  contract_table_info info;
};
}  // namespace
TEST_CASE("Admission uses the one actual grant and releases all unsuccessful grants",
          "[scan_preparation][ledger]")
{
  test_reservation_provider provider;
  SECTION("valid grant")
  {
    preparation_ledger ledger(provider);
    auto d = ledger.admit(sample(), host);
    CHECK(d.deferred);
    CHECK(d.sigma_retained == 20);
    CHECK(d.w == 32);
    CHECK(d.allocation_granularity == 1);
    CHECK(d.c_requested == 52);
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

TEST_CASE("Non-Iceberg adapter consumes the shared preparation capability contract",
          "[scan_preparation][contract]")
{
  test_reservation_provider provider;
  auto ledger   = std::make_unique<preparation_ledger>(provider);
  auto decision = ledger->admit(sample(), host);
  REQUIRE(decision.deferred);
  auto admission = std::make_shared<preparation_admission>(std::move(ledger), decision);

  contract_ingestible adapter;
  adapter.admission = admission;
  CHECK(adapter.preparation_admission() == admission);
  auto consumed = admission->consume();
  REQUIRE(consumed);
  adapter.attach_preparation_ledger(std::shared_ptr<preparation_ledger>(std::move(consumed)));
  CHECK(adapter.attached);

  auto claim = adapter.try_claim_split({});
  REQUIRE(std::holds_alternative<preparation_job>(claim));
  std::stop_source source;
  source.request_stop();
  CHECK_THROWS_AS(std::get<preparation_job>(claim).run(preparation_stop_token{source.get_token()}),
                  std::system_error);
  claim = adapter.try_claim_split({});
  CHECK(std::get<preparation_job>(claim).run({}));

  adapter.stop_preparation();
  adapter.finish_preparation();
  CHECK(adapter.stop_calls == 1);
  CHECK(adapter.finish_calls == 1);
  CHECK_FALSE(adapter.info.allows_cached_rows());
  adapter.attached->close();
  adapter.attached.reset();
  CHECK(provider.outstanding() == 0);
}

TEST_CASE("Preparation claim uses the adapter capability without an admission",
          "[scan_preparation][contract]")
{
  contract_ingestible adapter;
  sirius::scan_manager::split_provider provider(adapter, {});

  auto sources = provider.make_preparation_sources();
  REQUIRE(sources.size() == 1);
  auto claim = sources.front().try_claim();
  REQUIRE(std::holds_alternative<preparation_job>(claim));
  CHECK(adapter.claim_calls == 1);

  std::stop_source source;
  source.request_stop();
  CHECK_THROWS_AS(std::get<preparation_job>(claim).run(preparation_stop_token{source.get_token()}),
                  std::system_error);
  CHECK(adapter.custom_stop_seen);
}
TEST_CASE("Ordinary split provider cancellation drains queued and running metadata",
          "[scan_preparation][split_provider][cancellation]")
{
  using namespace sirius::test::preparation;
  class cancellable_ingestible final : public contract_ingestible {
   public:
    metadata_scan_task_t work;
    bool claimed = false;
    bool has_processed_all_metadata() const override { return claimed; }
    metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver) override
    {
      if (std::exchange(claimed, true)) return {};
      return std::move(work);
    }
    metadata_claim try_claim_split(sirius::io::ioctx_resolver resolve) override
    {
      return gpu_ingestible::try_claim_split(std::move(resolve));
    }
  } adapter;
  auto queued = GENERATE(false, true);
  fixture f(1, 1);
  auto blocked = f.make_hold();
  auto out     = std::make_shared<sink>();
  std::promise<void> submitted;
  auto accepted       = submitted.get_future();
  unsigned work_calls = 0, datasource_opens = 0;
  adapter.work = [&](preparation_stop_token stop) -> std::unique_ptr<sirius::op::scan::scan_info> {
    ++work_calls;
    blocked->wait();
    if (stop.stop_requested())
      throw std::system_error(std::make_error_code(std::errc::operation_canceled));
    ++datasource_opens;
    return std::make_unique<tagged>(0);
  };
  split_provider provider(adapter, {});
  auto sources = provider.make_preparation_sources();
  REQUIRE(sources.size() == 1);
  auto source        = make_source(preparation_execution_domain::metadata_worker);
  source.try_claim   = std::move(sources.front().try_claim);
  source.close       = std::move(sources.front().close);
  source.bind_wakeup = std::move(sources.front().bind_wakeup);
  f.register_source(std::move(source), out);
  auto guard = f.shutdown_guard();
  if (queued) f.dispatcher.enqueue([&] { blocked->wait(); });
  f.coordinator.before_submission_for_testing([&] { submitted.set_value(); });
  f.start();
  REQUIRE(accepted.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  REQUIRE(blocked->await());
  f.coordinator.request_stop(stop_reason::normal_eos);
  blocked->release();
  f.finish();
  CHECK(work_calls == (queued ? 0 : 1));
  CHECK(datasource_opens == 0);
  CHECK(out->count() == 0);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Admission sums retained results across scans and accounts for allocation granularity",
          "[scan_preparation][ledger]")
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
    CHECK(d.allocation_granularity == 16);
    CHECK(d.c_requested == 80);
  }
  CHECK(provider.outstanding() == 0);
}
TEST_CASE("Unqualified bounds and overflow route to legacy before any request",
          "[scan_preparation][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  auto s = sample();
  SECTION("missing proof") { s[0].qualified = false; }
  SECTION("overflow") { s[0].units[0].temporary[0] = std::numeric_limits<uint64_t>::max(); }
  CHECK_FALSE(ledger.admit(s, host).deferred);
  CHECK(provider.seen->requests == 0);
}
TEST_CASE("Parallel DV admission backs off without losing the one-permit route",
          "[scan_preparation][ledger]")
{
  enum class response { reject, partial, allocation_failure };
  auto mode     = GENERATE(response::reject, response::partial, response::allocation_failure);
  auto capacity = GENERATE(208u, 144u, 112u, 111u);
  struct budget_provider : reservation_provider {
    uint64_t capacity;
    response mode;
    std::vector<uint64_t> requests;
    test_reservation_provider backing;
    budget_provider(uint64_t capacity, response mode) : capacity(capacity), mode(mode) {}
    std::optional<reservation_grant> request(memory_space_id space, uint64_t bytes) override
    {
      CHECK(backing.outstanding() == 0);  // Failed grants must be released before retrying.
      requests.push_back(bytes);
      if (bytes > capacity) {
        if (mode == response::reject) return std::nullopt;
        if (mode == response::allocation_failure) throw std::bad_alloc();
      }
      backing.grant_bytes = std::min(bytes, capacity);
      return backing.request(space, bytes);
    }
  } provider(capacity, mode);
  {
    preparation_ledger ledger(provider);
    std::array scans{scan_envelope{std::vector<unit_envelope>(4, {{8, 8, 16}, 20}), {}, true}};
    auto d = ledger.admit(scans, host, 4);
    CHECK(d.sigma_retained == 80);
    CHECK(d.w == 32);
    CHECK(provider.requests.front() == 208);
    REQUIRE(d.deferred == (capacity >= 112));
    if (d.deferred) {
      auto expected = (capacity - 80) / 32;
      CHECK(d.n_permits == expected);
      CHECK(d.c_obtained >= d.sigma_retained + d.n_permits * d.w);
      CHECK(provider.backing.outstanding() == 1);
      std::vector<w_permit> permits;
      for (uint64_t i = 0; i < 4; ++i)
        ledger.register_unit({1, i}, 20);
      for (uint64_t i = 0; i < expected; ++i) {
        permits.push_back(ledger.acquire_permit({1, i}));
        REQUIRE(permits.back());
      }
      if (expected < 4) CHECK_FALSE(ledger.acquire_permit({1, expected}));
      auto previous = ledger.permits_in_flight();
      permits.front().release();
      CHECK(ledger.permits_in_flight() == previous - 1);
      CHECK(ledger.acquire_permit({1, 0}));
    } else {
      CHECK(d.n_permits == 0);
      CHECK(provider.backing.outstanding() == 0);
    }
    auto expected_requests = mode == response::partial && capacity >= 112
                               ? std::vector<uint64_t>{208}
                             : capacity >= 208 ? std::vector<uint64_t>{208}
                             : capacity >= 144 ? std::vector<uint64_t>{208, 144}
                                               : std::vector<uint64_t>{208, 144, 112};
    CHECK(provider.requests == expected_requests);
  }
  CHECK(provider.backing.outstanding() == 0);
}

TEST_CASE("DV permits are capped by work and configured concurrency even with an overgrant",
          "[scan_preparation][ledger]")
{
  auto limit = GENERATE(1u, 2u, 8u);
  test_reservation_provider provider;
  provider.grant_bytes = 4096;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{std::vector<unit_envelope>(3, {{8, 8, 16}, 20}), {}, true}};
  auto d = ledger.admit(scans, host, limit);
  REQUIRE(d.deferred);
  CHECK(d.n_permits == std::min(limit, 3u));
  CHECK(d.c_requested == 60 + 32 * std::min(limit, 3u));
  CHECK(provider.seen->requests == 1);
}

TEST_CASE("Parallel temporary capacity overflow preserves a representable single permit",
          "[scan_preparation][ledger]")
{
  test_reservation_provider provider;
  provider.grant_bytes = std::numeric_limits<uint64_t>::max();
  preparation_ledger ledger(provider);
  auto w = std::numeric_limits<uint64_t>::max() / 2 + 1;
  std::array scans{scan_envelope{{{{w, 0, 0}, 0}, {{w, 0, 0}, 0}}, {}, true}};
  auto d = ledger.admit(scans, host, 2);
  REQUIRE(d.deferred);
  CHECK(d.c_requested == w);
  CHECK(d.n_permits == 1);
}

TEST_CASE("Scans without DV work need no temporary-memory permits", "[scan_preparation][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{}, {}, true}};
  auto d = ledger.admit(scans, host);
  CHECK(d.deferred);
  CHECK(d.w == 0);
  CHECK(d.n_permits == 0);
  CHECK(provider.seen->requests == 0);
  CHECK_FALSE(ledger.acquire_permit({1, 1}));
}
TEST_CASE("The full temporary-memory permit stays held until its buffers are freed",
          "[scan_preparation][ledger]")
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
          "[scan_preparation][ledger]")
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
TEST_CASE("Results keep backing after plan and attempt owners finish", "[scan_preparation][ledger]")
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
TEST_CASE("Unexecuted admission releases its grant on finish or destruction",
          "[scan_preparation][ledger]")
{
  bool explicit_finish = GENERATE(false, true);
  test_reservation_provider provider;
  {
    auto ledger = std::make_unique<preparation_ledger>(provider);
    auto d      = ledger->admit(sample(), host);
    REQUIRE(d.deferred);
    preparation_admission owner(std::move(ledger), d);
    CHECK(provider.outstanding() == 1);
    if (explicit_finish) {
      owner.finish();
      owner.finish();
      CHECK(provider.outstanding() == 0);
      CHECK_THROWS_AS(owner.consume(), std::logic_error);
    }
  }
  CHECK(provider.outstanding() == 0);
}
TEST_CASE("Statement DV threshold only chooses preparation route including legacy scans",
          "[scan_preparation][ledger]")
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

TEST_CASE("Temporary-memory permits are independent and closure rejects further admission",
          "[scan_preparation][ledger]")
{
  test_reservation_provider provider;
  provider.grant_bytes = 84;
  preparation_ledger ledger(provider);
  auto scans = sample();
  scans[0].units.push_back({{8, 8, 16}, 0});
  REQUIRE(ledger.admit(scans, host, 2).n_permits == 2);
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
          "[scan_preparation][ledger][host]")
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
    CHECK(d.allocation_granularity == 1024);
    CHECK(d.c_requested == 4096);
    CHECK(d.c_obtained == 4096);
    CHECK(d.n_permits == 1);
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
          "[scan_preparation][ledger]")
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
          "[scan_preparation][ledger]")
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
          "[scan_preparation][ledger]")
{
  test_reservation_provider provider;
  provider.grant_bytes = 76;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{{64, 4, 4}, 4}}, {}, true}};
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
TEST_CASE("A fresh lowering owner is required for every execution", "[scan_preparation][ledger]")
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
          "[scan_preparation][ledger]")
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

TEST_CASE("Statement resource requests share backing and roll back failed preparation",
          "[scan_preparation][ledger]")
{
  auto failure = GENERATE(0, 1, 2);  // success, prepare failure, bind failure
  test_reservation_provider provider;
  provider.grant_bytes = 128;
  provider.quantum     = 8;
  std::array<charged_block, 2> descriptors;
  std::array<std::shared_ptr<preparation_admission>, 2> admissions;
  std::array<uint64_t, 2> retained{7, 17};
  size_t rollbacks = 0;
  std::vector<preparation_request> requests;
  for (size_t i = 0; i != 2; ++i) {
    requests.push_back(
      {scan_envelope{
         {{i == 0 ? std::vector<uint64_t>{9, 1} : std::vector<uint64_t>{3}, retained[i]}},
         {},
         true},
       [&, i](preparation_ledger& ledger) {
         auto permit = ledger.try_acquire_unit({i + 1, 1}, retained[i]);
         if (!permit) throw std::runtime_error("missing planning permit");
         descriptors[i] = ledger.allocator(permit).allocate_retained(retained[i]);
         if (i == 1 && failure == 1) throw std::runtime_error("prepare failed");
       },
       [&, i](auto admission) {
         admissions[i] = std::move(admission);
         if (i == 1 && failure == 2) throw std::runtime_error("bind failed");
       },
       [&, i]() noexcept {
         descriptors[i].release();
         admissions[i].reset();
         ++rollbacks;
       }});
  }
  if (failure) {
    REQUIRE_THROWS_WITH(prepare_statement_resources(provider, requests, host),
                        failure == 1 ? "prepare failed" : "bind failed");
    CHECK(rollbacks == 2);
    CHECK(provider.outstanding() == 0);
    CHECK(provider.seen->allocated_bytes == 0);
  } else {
    auto decision = prepare_statement_resources(provider, requests, host);
    REQUIRE(decision.deferred);
    CHECK(decision.sigma_retained == 32);
    CHECK(decision.w == 24);
    CHECK(decision.c_requested == 56);
    CHECK(provider.seen->requests == 1);
    REQUIRE(admissions[0] == admissions[1]);
    auto ledger = admissions[0]->consume();
    auto first  = ledger->acquire_permit({1, 1});
    REQUIRE(first);
    CHECK_FALSE(ledger->acquire_permit({2, 1}));
    first.release();
    auto second = ledger->acquire_permit({2, 1});
    REQUIRE(second);
    second.release();
    ledger.reset();
    CHECK(provider.outstanding() == 1);
    descriptors[0].release();
    CHECK(provider.seen->allocated_bytes == 24);
    descriptors[1].release();
    CHECK(provider.outstanding() == 0);
  }
}

TEST_CASE("Retained-only preparation jobs acquire permits without temporary storage",
          "[scan_preparation][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  auto bytes = GENERATE(uint64_t{0}, uint64_t{20});
  std::array scans{scan_envelope{{{{}, bytes}}, {}, true}};
  auto decision = ledger.admit(scans, host);
  REQUIRE(decision.deferred);
  REQUIRE(decision.n_permits == 1);
  auto permit = ledger.try_acquire_unit({1, 1}, bytes);
  REQUIRE(permit);
  auto output = ledger.allocator(permit).allocate_retained(bytes);
  CHECK_FALSE(ledger.try_acquire_unit({2, 1}, 0));
  permit.release();
  auto next = ledger.try_acquire_unit({2, 1}, 0);
  CHECK(next);
  CHECK(output.size() == bytes);
  CHECK_THROWS_AS(ledger.allocator(next).allocate(1), preparation_resource_error);
  CHECK_THROWS_AS(ledger.allocator(next).allocate_retained(1), preparation_resource_error);
  CHECK(ledger.bound_exceeded() == 2);
}

TEST_CASE("HOST reservation failure does not retry or leave a partial grant",
          "[scan_preparation][ledger][host]")
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

TEST_CASE("Failure to construct the lowering grant selects the original path once",
          "[scan_preparation][ledger]")
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

TEST_CASE("DV profile qualifies only complete bounded inputs and keeps append-only quota at zero",
          "[scan_preparation][iceberg][ledger]")
{
  using namespace sirius::op::scan;
  iceberg_delete_discovery discovery;
  std::vector<std::string> paths{"a.parquet", "b.parquet"};
  auto empty = iceberg_dv_preparation::envelope(discovery, {});
  CHECK(empty.qualified);
  CHECK(empty.units.empty());
  IcebergDeleteFileEntry dv;
  dv.file_format           = "puffin";
  dv.content_offset        = 4;
  dv.content_size_in_bytes = 44;
  dv.record_count          = 2;
  dv.file_size_in_bytes    = 400;
  dv.referenced_data_file  = "a.parquet";
  discovery.deletion_vector_entries.push_back(dv);
  auto e = iceberg_dv_preparation::envelope(discovery, paths);
  REQUIRE(e.qualified);
  CHECK(e.units.size() == 1);
  CHECK(e.units.front().retained == 16);
  CHECK(e.units.front().temporary.size() == 2);  // Roaring expansion allocates no temporary block
  SECTION("a blob length is not a complete container bound")
  {
    discovery.deletion_vector_entries.front().file_size_in_bytes = 44;
    CHECK_FALSE(iceberg_dv_preparation::envelope(discovery, paths).qualified);
  }
  SECTION("missing container bound stays legacy")
  {
    discovery.deletion_vector_entries.front().file_size_in_bytes = -1;
    CHECK_FALSE(iceberg_dv_preparation::envelope(discovery, paths).qualified);
  }
  SECTION("unprovable JSON size stays legacy")
  {
    discovery.deletion_vector_entries.front().file_size_in_bytes =
      std::numeric_limits<int64_t>::max();
    CHECK_FALSE(iceberg_dv_preparation::envelope(discovery, paths).qualified);
  }
  SECTION("mixed positional input keeps legacy")
  {
    discovery.positional_delete_files.push_back("pos.parquet");
    CHECK_FALSE(iceberg_dv_preparation::envelope(discovery, paths).qualified);
  }
}

TEST_CASE("Charged DV descriptor names outlive the plan through the last published split consumer",
          "[scan_preparation][iceberg][ledger]")
{
  using namespace sirius::op::scan;
  test_reservation_provider provider;
  provider.grant_bytes = 64 * 1024;
  auto ledger          = std::make_shared<preparation_ledger>(provider);
  auto discovery       = std::make_shared<iceberg_delete_discovery>();
  IcebergDeleteFileEntry dv;
  dv.file_format           = "puffin";
  dv.file_path             = "a.puffin";
  dv.content_offset        = 4;
  dv.content_size_in_bytes = 44;
  dv.record_count          = 2;
  dv.file_size_in_bytes    = 400;
  dv.referenced_data_file  = "a.parquet";
  discovery->deletion_vector_entries.push_back(dv);
  std::vector<std::string> paths{"a.parquet", "b.parquet"};
  auto e = iceberg_dv_preparation::envelope(*discovery, paths);
  REQUIRE(ledger->admit(std::array{e}, host).deferred);
  auto prepared = std::make_shared<iceberg_dv_preparation>(7, "table", paths, discovery, *ledger);
  REQUIRE(prepared->files().size() == 2);
  CHECK(prepared->files()[0].dv != nullptr);
  CHECK(prepared->files()[1].dv == nullptr);
  auto arena_charge = ledger->charged_bytes({7, 0});
  CHECK(arena_charge == e.retained.front());
  CHECK(ledger->tracked_units() == 1);
  auto attempt = std::make_shared<iceberg_dv_attempt>(prepared, ledger);
  auto claim   = attempt->try_claim();
  auto job     = std::get<std::shared_ptr<iceberg_dv_attempt::file_job>>(std::move(claim));
  job->commit();
  auto allocation = job->allocator().allocate_retained(16);
  auto* values    = reinterpret_cast<int64_t*>(allocation.data());
  values[0]       = 1;
  values[1]       = 3;
  auto result     = std::make_shared<iceberg_delete_set const>(
    job->descriptor().path, std::move(allocation), 2, 1, job->path_owner());
  job->complete(result);
  job.reset();
  CHECK(ledger->tracked_units() == 2);
  SECTION("a retained result survives descriptor retirement")
  {
    attempt->finish();
    prepared.reset();
    ledger->close();
    CHECK(result->data_file == "a.parquet");
    CHECK(result->positions[1] == 3);
    CHECK(provider.seen->allocated_bytes == arena_charge + 16);
    result.reset();
    CHECK(ledger->tracked_units() == 0);
  }
  SECTION("an attempt's weak slot does not retain the result")
  {
    CHECK(attempt->result(0, prepared->files()[0].path) == result);
    result.reset();
    CHECK(attempt->result(0, prepared->files()[0].path) == nullptr);
    CHECK(provider.seen->allocated_bytes == arena_charge);
    CHECK(ledger->tracked_units() == 1);
    attempt->finish();
    prepared.reset();
    ledger->close();
    CHECK(ledger->tracked_units() == 0);
  }
  ledger.reset();
  CHECK(provider.seen->allocated_bytes == 0);
  CHECK(provider.outstanding() == 0);
}

TEST_CASE("DV descriptor accounting preserves the base layout and rounded admission",
          "[scan_preparation][ledger]")
{
  using namespace sirius::op::scan;
  static_assert(sizeof(iceberg_dv_preparation::file) == 24 &&
                sizeof(std::weak_ptr<iceberg_delete_set const>) == 16 &&
                sizeof(iceberg_dv_preparation::file*) == 8);
  test_reservation_provider provider;
  provider.grant_bytes = 64 * 1024;
  std::vector<std::string> paths{"a.parquet", "b.parquet"};
  uint64_t expected_raw = 116, expected_charge = 116, expected_retained = 132;
  SECTION("byte-granular storage for two short paths") {}
  SECTION("sixteen-byte granularity for ten long paths")
  {
    paths.clear();
    for (char value : std::string("abcdefghij"))
      paths.emplace_back(200, value);
    provider.quantum  = 16;
    expected_raw      = 2490;
    expected_charge   = 2496;
    expected_retained = 2512;
  }
  auto discovery = std::make_shared<iceberg_delete_discovery>();
  auto empty     = iceberg_dv_preparation::envelope(*discovery, paths);
  CHECK(empty.qualified);
  CHECK(empty.units.empty());
  CHECK(empty.retained.empty());
  IcebergDeleteFileEntry dv;
  dv.file_format           = "puffin";
  dv.file_path             = "a.puffin";
  dv.content_offset        = 4;
  dv.content_size_in_bytes = 44;
  dv.record_count          = 2;
  dv.file_size_in_bytes    = 400;
  dv.referenced_data_file  = paths.front();
  discovery->deletion_vector_entries.push_back(dv);
  auto envelope = iceberg_dv_preparation::envelope(*discovery, paths);
  REQUIRE(envelope.qualified);
  REQUIRE(envelope.units.size() == 1);
  CHECK(envelope.retained.front() == expected_raw);
  CHECK(envelope.retained.size() == 1);
  CHECK(envelope.units.front().retained == 16);
  {
    preparation_ledger ledger(provider);
    auto decision = ledger.admit(std::array{envelope}, host);
    REQUIRE(decision.deferred);
    CHECK(decision.n_permits == 1);
    CHECK(decision.reason == admission_reason::admitted);
    CHECK(decision.route_reason == scan_eligibility::eligible);
    CHECK(decision.sigma_retained == expected_retained);
    CHECK(decision.allocation_granularity == provider.quantum);
    CHECK(decision.c_obtained == 65536);
    CHECK(provider.seen->requests == 1);
    {
      iceberg_dv_preparation prepared(7, "table", paths, discovery, ledger);
      CHECK(ledger.charged_bytes({7, 0}) == expected_charge);
      CHECK(provider.seen->allocated_bytes == expected_charge);
      CHECK(ledger.tracked_units() == 1);
      CHECK(ledger.permits_in_flight() == 0);
    }
    CHECK(ledger.tracked_units() == 0);
    CHECK(provider.seen->allocated_bytes == 0);
    CHECK(provider.seen->allocations == 1);
  }
  CHECK(provider.outstanding() == 0);
}

TEST_CASE("DV admission rejects null and replacement bindings without losing the first owner",
          "[scan_preparation][ledger]")
{
  dv_precondition_fixture f;
  CHECK_THROWS_AS(f.prepared->bind_admission(nullptr), std::logic_error);
  CHECK_FALSE(f.prepared->admission());
  REQUIRE_NOTHROW(f.prepared->bind_admission(f.admission));
  REQUIRE(f.prepared->admission() == f.admission);
  SECTION("the same admission cannot be rebound")
  {
    CHECK_THROWS_AS(f.prepared->bind_admission(f.admission), std::logic_error);
  }
  SECTION("a different admission cannot replace the first")
  {
    dv_precondition_fixture other;
    CHECK_THROWS_AS(f.prepared->bind_admission(other.admission), std::logic_error);
    CHECK_FALSE(other.prepared->admission());
  }
  SECTION("a null binding cannot clear the first")
  {
    CHECK_THROWS_AS(f.prepared->bind_admission(nullptr), std::logic_error);
  }
  CHECK(f.prepared->admission() == f.admission);
  CHECK(f.provider.seen->requests == 1);
}

TEST_CASE("DV jobs validate completion and retain result identity until retirement",
          "[scan_preparation][ledger]")
{
  using namespace sirius::op::scan;
  static_assert(!std::is_copy_constructible_v<iceberg_dv_attempt>);
  static_assert(!std::is_copy_assignable_v<iceberg_dv_attempt>);
  dv_precondition_fixture f;
  auto ledger  = f.consume();
  auto attempt = std::make_shared<iceberg_dv_attempt>(f.prepared, ledger);
  CHECK_THROWS_AS((iceberg_dv_attempt{f.prepared, ledger}), std::logic_error);
  auto claim  = attempt->try_claim();
  auto job    = std::get<std::shared_ptr<iceberg_dv_attempt::file_job>>(std::move(claim));
  auto result = std::make_shared<iceberg_delete_set const>(f.paths.front());
  CHECK_THROWS_AS(job->complete(result), std::logic_error);
  job->commit();
  SECTION("completed result remains available until GPU retirement")
  {
    REQUIRE_NOTHROW(job->complete(result));
    CHECK_THROWS_AS(job->complete(result), std::logic_error);
    CHECK(attempt->result(0, f.paths.front()) == result);
    CHECK(attempt->result(f.prepared->files().size(), f.paths.front()) == nullptr);
    CHECK(attempt->result(0, "other.parquet") == nullptr);
    job.reset();
    attempt->stop();
    CHECK(attempt->result(0, f.paths.front()) == result);
  }
  SECTION("late completion cannot write after stop or finish")
  {
    if (GENERATE(false, true))
      attempt->finish();
    else
      attempt->stop();
    CHECK_THROWS_AS(job->complete(result), std::logic_error);
    CHECK(attempt->result(0, f.paths.front()) == nullptr);
    job.reset();
  }
  attempt->finish();
  CHECK(f.prepared->files().empty());
  CHECK_THROWS_AS(attempt->try_claim(), std::logic_error);
  CHECK(attempt->result(0, f.paths.front()) == nullptr);
  REQUIRE_NOTHROW(attempt->finish());
  CHECK(ledger->tracked_units() == 0);
  ledger->close();
  ledger.reset();
  CHECK(f.provider.seen->allocated_bytes == 0);
  CHECK(f.provider.outstanding() == 0);
}

TEST_CASE("Discarding a committed DV job does not invalidate a running sibling",
          "[scan_preparation][ledger][claim]")
{
  using namespace sirius::op::scan;
  dv_precondition_fixture f(3);
  auto ledger  = f.consume();
  auto attempt = std::make_shared<iceberg_dv_attempt>(f.prepared, ledger);
  auto queued  = std::get<std::shared_ptr<iceberg_dv_attempt::file_job>>(attempt->try_claim());
  queued->commit();
  auto running = std::get<std::shared_ptr<iceberg_dv_attempt::file_job>>(attempt->try_claim());
  running->commit();
  CHECK(ledger->permits_in_flight() == 2);
  auto result = std::make_shared<iceberg_delete_set const>(f.paths[1]);

  // Normal EOS can dispose a queued job while another worker is just before complete().
  queued.reset();
  auto next = std::get<std::shared_ptr<iceberg_dv_attempt::file_job>>(attempt->try_claim());
  CHECK(next->descriptor().path == f.paths[2]);
  next->commit();
  next.reset();
  CHECK(std::holds_alternative<preparation_exhausted>(attempt->try_claim()));
  REQUIRE_NOTHROW(running->complete(result));
  CHECK(attempt->result(1, f.paths[1]) == result);
  running.reset();
  attempt->stop();
  attempt->finish();
  CHECK(ledger->permits_in_flight() == 0);
  CHECK(ledger->tracked_units() == 0);
  ledger->close();
  ledger.reset();
  CHECK(f.provider.outstanding() == 0);
}

TEST_CASE("External retirement waits for a DV claim before discarding queued work",
          "[scan_preparation][ledger][claim]")
{
  using namespace sirius::op::scan;
  using namespace sirius::test::preparation;
  dv_precondition_fixture dv(2);
  auto ledger  = dv.consume();
  auto attempt = std::make_shared<iceberg_dv_attempt>(dv.prepared, ledger);
  fixture f(1, 4, k_interrupt_check_interval, preparation_options{2, 4, 1, 1});
  auto worker = f.make_hold();
  auto claim  = f.make_hold();
  auto out    = std::make_shared<sink>();
  std::weak_ptr<iceberg_dv_attempt::file_job> queued;
  std::atomic<size_t> claims{0}, reads{0};
  std::atomic<bool> closed{false};
  std::promise<void> stop_requested;
  auto requested = stop_requested.get_future();
  std::future<void> retiring;
  auto guard = f.shutdown_guard();

  // Occupy the only worker so the first DV job stays in the dispatcher queue.
  f.dispatcher.enqueue([worker] { worker->wait(); });
  REQUIRE(worker->await());
  auto source      = make_source(preparation_coordinator::execution_domain::metadata_worker);
  source.try_claim = [&]() -> preparation_coordinator::claim_result {
    auto index = claims++;
    if (index == 1) claim->wait();
    auto claimed = attempt->try_claim();
    auto* ready  = std::get_if<std::shared_ptr<iceberg_dv_attempt::file_job>>(&claimed);
    if (!ready) return preparation_exhausted{};
    auto owned = std::move(*ready);
    owned->commit();
    if (index == 0) queued = owned;
    return preparation_coordinator::job{[&, owned = std::move(owned)] {
      ++reads;
      owned->complete(
        std::make_shared<iceberg_delete_set const>(std::string(owned->descriptor().path)));
      return std::make_unique<tagged>(0);
    }};
  };
  source.close = [&] {
    attempt->stop();
    closed = true;
    out->close();
  };
  f.register_source(std::move(source), out);
  f.start();
  REQUIRE(claim->await());
  retiring = std::async(std::launch::async, [&] {
    f.coordinator.request_stop(stop_reason::normal_eos);
    stop_requested.set_value();
    f.coordinator.drain();
  });
  REQUIRE(requested.wait_for(1s) == std::future_status::ready);
  CHECK_FALSE(queued.expired());
  CHECK_FALSE(closed);
  CHECK(retiring.wait_for(0ms) == std::future_status::timeout);
  claim->release();
  worker->release();
  f.finish();
  retiring.get();

  CHECK_FALSE(f.completion.has_error());
  f.completion.mark_completed();
  CHECK_NOTHROW(f.completion.get_awaitable().get());
  CHECK(claims == 2);
  CHECK(reads == 0);
  CHECK(queued.expired());
  CHECK(out->count() == 0);
  CHECK(out->close_calls == 1);
  CHECK(out->closing_thread == f.coordinator.snapshot().owner);
  CHECK(ledger->permits_in_flight() == 0);
  attempt->finish();
  CHECK(ledger->tracked_units() == 0);
  ledger->close();
  ledger.reset();
  CHECK(dv.provider.outstanding() == 0);
}

TEST_CASE("Iceberg ledger installation rejects missing repeated and undeferred state",
          "[scan_preparation][ledger]")
{
  using namespace sirius::op::scan;
  dv_precondition_fixture f;
  auto ledger     = f.consume();
  auto ingestible = make_ingestible(f.info());
  CHECK_THROWS_AS(ingestible->attach_preparation_ledger(nullptr), std::logic_error);
  auto legacy = make_ingestible(f.info(false));
  CHECK_THROWS_AS(legacy->attach_preparation_ledger(ledger), std::logic_error);
  REQUIRE_NOTHROW(ingestible->attach_preparation_ledger(ledger));
  CHECK_THROWS_WITH(ingestible->attach_preparation_ledger(ledger),
                    "DV preparation attempt already started");
  ingestible->finish_preparation();
  CHECK(ledger.use_count() == 1);
  CHECK(f.prepared->files().empty());
  CHECK(ledger->tracked_units() == 0);
  ledger->close();
  ledger.reset();
  CHECK(f.provider.seen->allocated_bytes == 0);
  CHECK(f.provider.outstanding() == 0);
}

TEST_CASE("Uninstalled Iceberg preparation releases descriptors at final cleanup",
          "[scan_preparation][ledger]")
{
  dv_precondition_fixture f;
  auto ledger     = f.consume();
  auto ingestible = make_ingestible(f.info());

  REQUIRE_FALSE(f.prepared->files().empty());
  ingestible->finish_preparation();
  CHECK(f.prepared->files().empty());
  CHECK(f.provider.seen->allocated_bytes == 0);

  REQUIRE_NOTHROW(ingestible->finish_preparation());
  ledger->close();
  ledger.reset();
  CHECK(f.provider.outstanding() == 0);
}

TEST_CASE("Rejected pre-start Iceberg claims leave the first file available",
          "[scan_preparation][ledger]")
{
  using namespace sirius::op::scan;
  dv_precondition_fixture f;
  auto ledger     = f.consume();
  auto ingestible = make_ingestible(f.info());
  auto resolve    = [](std::string_view) -> std::shared_ptr<sirius::io::ioctx> {
    throw std::runtime_error("claim-only resolver was executed");
  };
  REQUIRE(f.paths.size() == 1);
  REQUIRE_FALSE(ingestible->has_processed_all_metadata());
  CHECK_THROWS_AS(ingestible->try_claim_split(resolve), std::logic_error);
  REQUIRE_FALSE(ingestible->has_processed_all_metadata());
  REQUIRE_NOTHROW(ingestible->attach_preparation_ledger(ledger));
  auto claim = ingestible->try_claim_split(resolve);
  auto work  = std::move(std::get<gpu_ingestible::metadata_job>(claim));
  CHECK(ingestible->has_processed_all_metadata());
  CHECK(std::holds_alternative<preparation_exhausted>(ingestible->try_claim_split(resolve)));
  work = {};
  CHECK(ledger->permits_in_flight() == 0);
  CHECK(ledger->tracked_units() == 1);
  ingestible->stop_preparation();
  CHECK(ledger.use_count() == 1);
  CHECK_THROWS_AS(ingestible->try_claim_split(resolve), std::logic_error);
  CHECK(f.prepared->files().size() == 1);
  ingestible->finish_preparation();
  CHECK(ledger->tracked_units() == 0);
  ledger->close();
  ledger.reset();
  CHECK(f.provider.seen->allocated_bytes == 0);
  CHECK(f.provider.outstanding() == 0);
}

TEST_CASE("Pending resource claims leave no registration and wake after physical release",
          "[scan_preparation][ledger][claim]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  REQUIRE(ledger.admit(sample(), host).n_permits == 1);
  auto gate = std::make_shared<preparation_gate>();
  ledger.bind_wakeup(preparation_wakeup{gate});
  auto first = ledger.try_acquire_unit({1, 1}, 8);
  REQUIRE(first);
  auto bytes = ledger.allocator(first).allocate(8);
  for (unsigned n = 0; n != 3; ++n) {
    CHECK_FALSE(ledger.try_acquire_unit({1, 2}, 12));
    CHECK(ledger.tracked_units() == 1);
  }
  first.release();
  ledger.retire_unit({1, 1});
  CHECK(gate->wake_generation == 0);
  bytes = {};
  CHECK(gate->wake_generation == 1);
  auto second = ledger.try_acquire_unit({1, 2}, 12);
  REQUIRE(second);
  second.release();
  ledger.retire_unit({1, 2});
  CHECK(ledger.tracked_units() == 0);
}

TEST_CASE("DV claim owns descriptors and resources after execution finish",
          "[scan_preparation][ledger][claim]")
{
  using namespace sirius::op::scan;
  dv_precondition_fixture f;
  auto ledger  = f.consume();
  auto attempt = std::make_shared<iceberg_dv_attempt>(f.prepared, ledger);
  auto claimed = attempt->try_claim();
  auto& owned  = std::get<std::shared_ptr<iceberg_dv_attempt::file_job>>(claimed);
  REQUIRE(owned);
  CHECK(owned->descriptor().path == f.paths.front());
  CHECK(ledger->permits_in_flight() == 1);
  attempt->finish();
  CHECK(f.prepared->files().empty());
  CHECK(owned->descriptor().path == "a.parquet");
  CHECK(owned->descriptor().dv->file_path == "a.puffin");
  CHECK_THROWS_AS(owned->commit(), std::logic_error);
  auto allocator = owned->allocator();
  auto temporary = allocator.allocate(8);
  owned.reset();
  CHECK(ledger->permits_in_flight() == 1);
  temporary.release();
  CHECK_THROWS_AS(allocator.allocate(8), std::logic_error);
  CHECK(ledger->permits_in_flight() == 0);
  CHECK(ledger->tracked_units() == 0);
  CHECK(f.provider.seen->allocated_bytes == 0);
}

TEST_CASE("Iceberg pending claims preserve both cursors and discarded jobs release their permits",
          "[scan_preparation][ledger][claim]")
{
  using namespace sirius::op::scan;
  dv_precondition_fixture f;
  auto ledger     = f.consume();
  auto ingestible = make_ingestible(f.info());
  ingestible->attach_preparation_ledger(ledger);
  auto resolve = [](std::string_view) -> std::shared_ptr<sirius::io::ioctx> {
    throw std::runtime_error("claim must not read files");
  };
  auto occupied = ledger->try_acquire_unit({99, 1}, 0);
  REQUIRE(occupied);
  auto registrations = ledger->tracked_units();
  for (int i = 0; i != 3; ++i) {
    CHECK(std::holds_alternative<preparation_pending>(ingestible->try_claim_split(resolve)));
    CHECK_FALSE(ingestible->has_processed_all_metadata());
    CHECK(ledger->tracked_units() == registrations);
  }
  occupied.release();
  ledger->retire_unit({99, 1});
  {
    auto claim  = ingestible->try_claim_split(resolve);
    auto& ready = std::get<gpu_ingestible::metadata_job>(claim);
    REQUIRE(ready.requirement);
    CHECK(ready.requirement->key == unit_key{f.prepared->contract(), 1});
    CHECK(
      ready.requirement->required ==
      required_inputs(f.read_views->entry(f.prepared->contract()).eligibility.later_checks, true));
    CHECK(ingestible->has_processed_all_metadata());
  }
  CHECK(std::holds_alternative<preparation_exhausted>(ingestible->try_claim_split(resolve)));
  CHECK(ledger->permits_in_flight() == 0);
  ingestible->finish_preparation();
  CHECK(ledger->tracked_units() == 0);
}

TEST_CASE("Resource wake handles cannot execute arbitrary throwing cleanup hooks",
          "[scan_preparation][ledger][claim]")
{
  static_assert(std::is_nothrow_invocable_v<preparation_wakeup>);
  static_assert(!std::is_constructible_v<preparation_wakeup, std::function<void()>>);
  preparation_wakeup expired;
  {
    auto gate = std::make_shared<preparation_gate>();
    expired   = preparation_wakeup{gate};
    expired();
    CHECK(gate->wake_generation == 1);
  }
  REQUIRE_NOTHROW(expired());
}

TEST_CASE("Admission preserves output-holding scans minimum progress credits",
          "[scan_preparation][ledger]")
{
  test_reservation_provider provider;
  auto scan            = sample().front();
  scan.minimum_permits = 1;
  std::array scans{scan, scan};
  SECTION("configured limit cannot serialize mutually dependent scans")
  {
    preparation_ledger ledger(provider);
    auto decision = ledger.admit(scans, host, 1);
    CHECK_FALSE(decision.deferred);
    CHECK(decision.reason == admission_reason::capacity);
    CHECK(provider.seen->requests == 0);
  }
  SECTION("short grant cannot drop a required permit")
  {
    provider.grant_bytes = 103;
    preparation_ledger ledger(provider);
    auto decision = ledger.admit(scans, host, 2);
    CHECK_FALSE(decision.deferred);
    CHECK(decision.reason == admission_reason::short_grant);
    CHECK(provider.seen->requests == 1);
    CHECK(provider.outstanding() == 0);
  }
  SECTION("each scan can hold output while another acquires credit")
  {
    provider.grant_bytes = 104;
    preparation_ledger ledger(provider);
    auto decision = ledger.admit(scans, host, 2);
    REQUIRE(decision.deferred);
    CHECK(decision.n_permits == 2);
    auto a = ledger.try_acquire_unit({1, 0}, 20);
    auto b = ledger.try_acquire_unit({2, 0}, 20);
    REQUIRE(a);
    REQUIRE(b);
    auto output = ledger.allocator(a).allocate(32);
    a.release();
    CHECK(ledger.permits_in_flight() == 2);
    b.release();
    auto next = ledger.try_acquire_unit({2, 1}, 0);
    CHECK(bool(next));
  }
}

TEST_CASE("Input accounting includes temporary and retained allocations from its permit",
          "[scan_preparation][ledger]")
{
  test_reservation_provider provider;
  preparation_ledger ledger(provider);
  REQUIRE(ledger.admit(sample(), host).deferred);
  auto permit = ledger.try_acquire_unit({1, 2}, 20);
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  auto temporary = allocator.allocate(8);
  auto retained  = allocator.allocate_retained(12);
  CHECK(ledger.charged_bytes(permit) == 20);
  CHECK(ledger.charged_bytes(unit_key{1, 2}) == 12);
  temporary.release();
  CHECK(ledger.charged_bytes(permit) == 12);
  retained.release();
  CHECK(ledger.charged_bytes(permit) == 0);
  preparation_ledger foreign(provider);
  CHECK_THROWS_AS(foreign.charged_bytes(permit), std::logic_error);
  ledger.retire_unit({1, 2});
  permit.release();
  CHECK_THROWS_AS(ledger.charged_bytes(permit), std::logic_error);
}
