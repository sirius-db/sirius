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
#include "op/scan/composite_scan_driver.hpp"
#include "operator/operator_test_utils.hpp"
#include "scan_manager/preparation_identity.hpp"
#include "scan_manager/preparation_slot.hpp"
#include "scan_manager/preparation_test_support.hpp"

#include <cudf/column/column_factories.hpp>

#include <duckdb/common/exception.hpp>

#include <array>
#include <future>
#include <numeric>

using namespace sirius::op::scan;
namespace sm = sirius::scan_manager;
namespace {
using domain = sm::preparation_execution_domain;
struct test_split : scan_info {
  explicit test_split(scan_contract_id contract = 77) { set_contract_payload(contract, {}, {}); }
  size_t estimated_bytes() const noexcept override { return 17; }
  size_t preparation_bytes() const noexcept override { return 23; }
};
struct test_validator : split_validator {
  uint64_t generation = 9;
  bool accept         = true;
  mutable later_check_set seen;
  bool accepts(scan_info const& input) const noexcept override
  {
    return accept && dynamic_cast<test_split const*>(&input);
  }
  split_validation_evidence inspect(split_validation_context const& context,
                                    scan_info const&) const override
  {
    seen = context.required;
    return {{}, generation};
  }
};
class test_coalescer : public batch_coalescer {
  bool buffer_;
  scan_contract_id contract_;
  std::unique_ptr<scan_info> pending_;

 public:
  explicit test_coalescer(bool buffer, scan_contract_id contract)
    : buffer_(buffer), contract_(contract)
  {
  }
  cursor_step advance(scan_info&, size_t&, size_t) override
  {
    if (buffer_) {
      pending_ = std::make_unique<test_split>(contract_);
      return {nullptr, false};
    }
    return {std::make_unique<test_split>(contract_), true};
  }
  size_t retained_bytes() const noexcept override { return pending_ ? 17 : 0; }
  std::unique_ptr<scan_info> partial_emit() override { return std::move(pending_); }
  std::vector<std::unique_ptr<scan_info>> flush() override
  {
    std::vector<std::unique_ptr<scan_info>> out;
    if (pending_) out.push_back(std::move(pending_));
    return out;
  }
  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info>) override
  {
    throw std::logic_error("bounded only");
  }
};
struct test_info : ingestible_table_info {
  std::span<std::string const> column_names() const override { return {}; }
  std::span<std::string const> file_paths() const override { return {}; }
  std::string display_name() const override { return "test child"; }
};
struct test_child : gpu_ingestible {
  test_info info;
  scan_contract_id split_contract = 77;
  domain execution                = domain::metadata_worker;
  size_t lanes                    = 1;
  int claims = 0, closes = 0, stops = 0, finishes = 0, preflights = 0, projections = 0;
  bool ready = true, empty = false, buffered = false, fail_close = false, fail_bind = false;
  bool fail_read = false, already_projected = false;
  std::shared_ptr<void> retained;
  void preflight_for_query() override { ++preflights; }
  void attach_preparation_ledger(std::shared_ptr<sm::preparation_ledger>) override {}
  void stop_preparation() noexcept override { ++stops; }
  void finish_preparation() noexcept override
  {
    ++finishes;
    retained.reset();
  }
  bool has_processed_all_metadata() const override { return claims > 0; }
  metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver) override { return {}; }
  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override
  {
    return std::make_unique<test_coalescer>(buffered, split_contract);
  }
  std::vector<preparation_source> make_preparation_sources(sirius::io::ioctx_resolver) override
  {
    std::vector<preparation_source> out;
    for (size_t n = 0; n < lanes; ++n) {
      preparation_source source;
      source.execution = execution;
      source.coalescer = create_batch_coalescer();
      source.try_claim = [this]() -> metadata_claim {
        if (!ready) return sm::preparation_pending{};
        if (claims++) return sm::preparation_exhausted{};
        return metadata_job{[this](sm::preparation_stop_token stop) -> std::unique_ptr<scan_info> {
          if (stop.stop_requested()) throw duckdb::InterruptException();
          if (fail_read) throw std::runtime_error("child read failed");
          return empty ? nullptr : std::make_unique<test_split>();
        }};
      };
      source.close = [this] {
        ++closes;
        if (fail_close) throw std::runtime_error("child close failed");
      };
      source.bind_wakeup = [this](sm::preparation_wakeup) {
        if (fail_bind) throw std::runtime_error("child bind failed");
      };
      out.push_back(std::move(source));
    }
    return out;
  }
  ingestible_table_info const& table_info() const noexcept override { return info; }
  std::vector<size_t> materialized_column_order() const override { return {0}; }
  filtered_table materialize_metadata_to_table(
    scan_info const&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<sirius::like_multiliteral_cache const>,
    sirius::expression_query_policy) override
  {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
    return {
      owning_table_view(std::make_unique<cudf::table>(std::move(cols))),
      already_projected ? filter_state::ROW_FILTERED_AND_PROJECTED : filter_state::UNFILTERED};
  }
  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    cucascade::memory::memory_space const& space,
    ::cuda::stream_ref stream,
    bool,
    std::shared_ptr<sirius::like_multiliteral_cache const>,
    std::unique_ptr<cudf::column>*,
    std::span<size_t const>,
    sirius::expression_query_policy) override
  {
    ++projections;
    return input.table.release(stream, space.get_default_allocator());
  }
};
struct harness {
  sm::test::test_reservation_provider provider;
  std::shared_ptr<sm::preparation_ledger> ledger =
    std::make_shared<sm::preparation_ledger>(provider);
  std::shared_ptr<test_validator> validator = std::make_shared<test_validator>();
  std::vector<std::shared_ptr<test_child>> children;
  size_t discoveries = 0;
  composite_scan_driver driver;
  std::vector<gpu_ingestible::preparation_source> sources;
  explicit harness(std::vector<std::shared_ptr<test_child>> inputs)
    : children(std::move(inputs)),
      driver(77,
             9,
             check_bit(later_check::host_staged),
             4096,
             [this]() -> std::optional<composite_scan_child> {
               if (discoveries == children.size()) return {};
               return composite_scan_child{
                 children[discoveries++], validator, check_bit(later_check::host_staged)};
             })
  {
    std::vector<sm::scan_envelope> envelopes{{{{{8}, 16}}, {}, true, 1}};
    REQUIRE(
      ledger->admit(envelopes, sm::memory_space_id{cucascade::memory::Tier::HOST, 0}).deferred);
    sources = driver.make_preparation_sources({}, ledger, {});
    for (auto& source : sources)
      source.bind_wakeup({});
  }
  std::unique_ptr<scan_info> run(size_t lane)
  {
    auto claim = sources[lane].try_claim();
    REQUIRE(std::holds_alternative<sm::preparation_job>(claim));
    return std::get<sm::preparation_job>(std::move(claim)).run({});
  }
  std::unique_ptr<scan_info> coalesce(size_t lane, scan_info& input)
  {
    size_t cursor = 0;
    auto result   = sources[lane].coalescer->advance(input, cursor, 1);
    REQUIRE(result.finished);
    return std::move(result.batch);
  }
  ~harness()
  {
    for (auto& source : sources) {
      try {
        source.close();
      } catch (...) {
      }
    }
    sources.clear();
    driver.finish_preparation();
    ledger->close();
  }
};
}  // namespace

TEST_CASE("Composite scans switch children only after jobs and metadata close", "[composite_scan]")
{
  auto first        = std::make_shared<test_child>();
  auto second       = std::make_shared<test_child>();
  second->execution = domain::query_owner;
  harness h({first, second});
  CHECK_FALSE(h.run(0));
  auto job = h.sources[1].try_claim();
  REQUIRE(std::holds_alternative<sm::preparation_job>(job));
  CHECK(std::holds_alternative<sm::preparation_pending>(h.sources[1].try_claim()));
  CHECK(std::holds_alternative<sm::preparation_pending>(h.sources[0].try_claim()));
  CHECK(first->closes == 0);
  auto input  = std::get<sm::preparation_job>(std::move(job)).run({});
  auto output = h.coalesce(1, *input);
  CHECK(output->estimated_bytes() == 17);
  CHECK(output->preparation_bytes() == 23);
  input.reset();
  CHECK(std::holds_alternative<sm::preparation_pending>(h.sources[1].try_claim()));
  CHECK(first->closes == 1);
  CHECK_FALSE(h.run(0));
  CHECK(h.discoveries == 2);
  CHECK(first->finishes == 0);
  output.reset();
  CHECK(first->finishes == 1);
  auto next  = h.run(0);
  auto batch = h.coalesce(0, *next);
  CHECK(std::holds_alternative<sm::preparation_pending>(h.sources[0].try_claim()));
  CHECK_FALSE(h.run(0));
  CHECK(h.driver.exhausted());
  CHECK(second->closes == 1);
  for (auto& lane : h.sources)
    CHECK(lane.coalescer->flush().empty());
}

TEST_CASE("Composite scans emit exactly one carrier only for a whole empty source",
          "[composite_scan]")
{
  auto child      = std::make_shared<test_child>();
  child->empty    = true;
  auto with_child = GENERATE(false, true);
  harness h(with_child ? std::vector{child} : std::vector<std::shared_ptr<test_child>>{});
  CHECK_FALSE(h.run(0));
  if (with_child) {
    CHECK_FALSE(h.run(1));
    CHECK(std::holds_alternative<sm::preparation_pending>(h.sources[1].try_claim()));
    CHECK_FALSE(h.run(0));
  }
  CHECK(h.driver.exhausted());
  CHECK(h.sources[0].coalescer->flush().empty());
  auto tail = h.sources[1].coalescer->flush();
  REQUIRE(tail.size() == 1);
  CHECK(tail[0]->preparation_bytes() == 0);
  CHECK(h.driver.validator()->accepts(*tail[0]));
  CHECK(h.driver.validator()->inspect({77, {}, {}}, *tail[0]).generation == 9);
  CHECK(h.sources[1].coalescer->flush().empty());
}

TEST_CASE("Composite validator rejects foreign routes and delegates child evidence",
          "[composite_scan]")
{
  auto child = std::make_shared<test_child>();
  harness h({child}), other({});
  h.run(0);
  auto input = h.run(1);
  CHECK_FALSE(h.driver.validator()->accepts(*input));
  auto output = h.coalesce(1, *input);
  CHECK_FALSE(other.driver.validator()->accepts(*output));
  CHECK(h.driver.validator()->inspect({77, {}, {}}, *output).generation == 9);
  CHECK(h.validator->seen == check_bit(later_check::host_staged));
  h.validator->generation = 10;
  CHECK_THROWS_AS(h.driver.validator()->inspect({77, {}, {}}, *output), certificate_incomplete);
  h.validator->generation = 9;
  h.validator->accept     = false;
  CHECK_THROWS_AS(h.driver.validator()->inspect({77, {}, {}}, *output), certificate_incomplete);
  CHECK_THROWS_AS(
    h.driver.validator()->inspect({77, check_bit(later_check::schema_per_file), {}}, *output),
    certificate_incomplete);
}

TEST_CASE("Composite scans preserve pending state and abandoned job progress", "[composite_scan]")
{
  auto child = std::make_shared<test_child>();
  harness h({child});
  h.run(0);
  child->ready = false;
  CHECK(std::holds_alternative<sm::preparation_pending>(h.sources[1].try_claim()));
  CHECK(child->claims == 0);
  child->ready = true;
  {
    auto abandoned = h.sources[1].try_claim();
    REQUIRE(std::holds_alternative<sm::preparation_job>(abandoned));
  }
  CHECK(std::holds_alternative<sm::preparation_pending>(h.sources[1].try_claim()));
  CHECK_FALSE(h.run(0));
  CHECK(h.driver.exhausted());
}

TEST_CASE("Composite scans retain children for partial coalescer output", "[composite_scan]")
{
  auto child      = std::make_shared<test_child>();
  child->buffered = true;
  harness h({child});
  h.run(0);
  auto input      = h.run(1);
  size_t cursor   = 0;
  auto& coalescer = *h.sources[1].coalescer;
  CHECK_FALSE(coalescer.advance(*input, cursor, 1).finished);
  CHECK(coalescer.retained_bytes() == 17);
  auto batch = coalescer.partial_emit();
  REQUIRE(batch);
  CHECK(coalescer.retained_bytes() == 0);
  CHECK(h.driver.validator()->accepts(*batch));
  h.driver.finish_preparation();
  CHECK(child->finishes == 0);
  input.reset();
  batch.reset();
  // The lane's active coalescer also holds its child until source drain.
  h.sources.clear();
  CHECK(child->finishes == 1);
}

TEST_CASE("Composite scans clean up initialization and close failures", "[composite_scan]")
{
  auto child = std::make_shared<test_child>();
  harness h({child});
  SECTION("binding failure keeps every created lane reachable for close")
  {
    child->lanes     = 2;
    child->fail_bind = true;
    CHECK_THROWS_WITH(h.run(0), "child bind failed");
    h.sources[1].close();
    CHECK(child->closes == 2);
  }
  SECTION("one failing close does not skip other lanes")
  {
    child->lanes      = 2;
    child->fail_close = true;
    h.run(0);
    CHECK_THROWS_WITH(h.sources[1].close(), "child close failed");
    CHECK(child->closes == 2);
    CHECK_NOTHROW(h.sources[1].close());
  }
  SECTION("unsupported execution domain is rejected")
  {
    child->execution = domain::native_reader;
    CHECK_THROWS_WITH(h.run(0), "Composite child requested an unsupported execution domain");
  }
  SECTION("reader failure propagates")
  {
    child->fail_read = true;
    h.run(0);
    CHECK_THROWS_WITH(h.run(1), "child read failed");
  }
  h.sources.clear();
  h.driver.finish_preparation();
  CHECK(child->finishes == 1);
}

TEST_CASE("Composite materialization projects each child once", "[composite_scan][gpu]")
{
  auto* space              = sirius::test::operator_utils::get_default_gpu_space();
  auto stream              = space->acquire_stream();
  auto child               = std::make_shared<test_child>();
  child->already_projected = GENERATE(false, true);
  harness h({child});
  h.run(0);
  auto input  = h.run(1);
  auto output = h.coalesce(1, *input);
  auto batch  = h.driver.materialize(*output, *space, stream, false, {}, {});
  REQUIRE(batch);
  CHECK(batch->state == filter_state::ROW_FILTERED_AND_PROJECTED);
  CHECK(child->projections == (child->already_projected ? 0 : 1));
  harness empty({});
  empty.run(0);
  empty.sources[0].coalescer->flush();
  auto tail = empty.sources[1].coalescer->flush();
  CHECK_FALSE(empty.driver.materialize(*tail.front(), *space, stream, false, {}, {}));
}

TEST_CASE("Scan preparation identities do not alias across concurrent unit sequences",
          "[composite_scan]")
{
  sm::preparation_identity identity;
  std::array<std::future<std::vector<uint64_t>>, 4> producers;
  for (auto& producer : producers) {
    producer = std::async(std::launch::async, [&] {
      std::vector<uint64_t> keys;
      for (size_t batch = 0; batch < 1024; ++batch)
        keys.push_back(identity.next());
      return keys;
    });
  }
  std::vector<uint64_t> keys;
  for (auto& producer : producers) {
    auto batch = producer.get();
    keys.insert(keys.end(), batch.begin(), batch.end());
  }
  std::sort(keys.begin(), keys.end());
  REQUIRE(keys.size() == 4096);
  CHECK(std::adjacent_find(keys.begin(), keys.end()) == keys.end());
  CHECK(keys.front() == 0);
  CHECK(identity.next() == keys.back() + 1);
  sm::preparation_identity other_scan;
  CHECK(other_scan.next() == 0);
}

TEST_CASE("Scan retained preparation slots reuse capacity without crossing scan ownership",
          "[composite_scan][ducklake_delete_budget]")
{
  namespace sm = sirius::scan_manager;
  sm::test::test_reservation_provider provider;
  auto ledger = std::make_shared<sm::preparation_ledger>(provider);
  sm::scan_envelope envelope{{{{8}, 16}}, {}, true, 1};
  std::vector<sm::scan_envelope> scans{envelope, envelope};
  auto decision = ledger->admit(scans, sm::memory_space_id{cucascade::memory::Tier::HOST, 0}, 2);
  REQUIRE(decision.deferred);
  REQUIRE(decision.n_permits == 2);
  auto first  = std::make_shared<sm::preparation_slot>(ledger, sm::unit_key{1, 1}, 16);
  auto second = std::make_shared<sm::preparation_slot>(ledger, sm::unit_key{2, 1}, 16);
  for (int file = 0; file < 32; ++file) {
    auto a = first->try_acquire();
    REQUIRE(a);
    CHECK_FALSE(first->try_acquire());
    auto b = second->try_acquire();
    REQUIRE(b);
    auto bytes = ledger->allocator(a).allocate_retained(16);
    CHECK(ledger->charged_bytes(first->key()) == 16);
    CHECK_THROWS_AS(ledger->allocator(a).allocate_retained(1), sm::preparation_resource_error);
    CHECK(ledger->tracked_units() == 2);
    bytes.release();
    CHECK(provider.seen->allocated_bytes == 0);
  }
  CHECK(ledger->permits_in_flight() == 0);
  first.reset();
  second.reset();
  CHECK(ledger->tracked_units() == 0);
  ledger.reset();
  CHECK(provider.outstanding() == 0);
}

TEST_CASE("Scan retained slot retirement preserves outstanding backing",
          "[composite_scan][ducklake_delete_budget]")
{
  namespace sm = sirius::scan_manager;
  sm::test::test_reservation_provider provider;
  auto ledger = std::make_shared<sm::preparation_ledger>(provider);
  std::vector<sm::scan_envelope> scans{{{{{8}, 16}}, {}, true, 1}};
  REQUIRE(ledger->admit(scans, sm::memory_space_id{cucascade::memory::Tier::HOST, 0}).deferred);
  auto slot   = std::make_shared<sm::preparation_slot>(ledger, sm::unit_key{1, 1}, 16);
  auto permit = slot->try_acquire();
  REQUIRE(permit);
  auto bytes = ledger->allocator(permit).allocate_retained(16);
  slot.reset();
  permit.release();
  CHECK(ledger->tracked_units() == 1);
  ledger->close();
  ledger.reset();
  CHECK(provider.seen->allocated_bytes == 16);
  CHECK(provider.outstanding() == 1);
  bytes.release();
  CHECK(provider.seen->allocated_bytes == 0);
  CHECK(provider.outstanding() == 0);
}

TEST_CASE("Composite discovery releases its owner lane on abandonment and cancellation",
          "[composite_scan]")
{
  harness h({std::make_shared<test_child>()});
  {
    auto abandoned = h.sources[0].try_claim();
    REQUIRE(std::holds_alternative<sm::preparation_job>(abandoned));
  }
  CHECK(h.discoveries == 0);
  auto claim = h.sources[0].try_claim();
  REQUIRE(std::holds_alternative<sm::preparation_job>(claim));
  std::stop_source stop;
  stop.request_stop();
  CHECK_THROWS_AS(std::get<sm::preparation_job>(std::move(claim))
                    .run(sm::preparation_stop_token(stop.get_token())),
                  duckdb::InterruptException);
  CHECK(h.discoveries == 0);
  CHECK_FALSE(h.run(0));
  CHECK(h.discoveries == 1);
}

TEST_CASE("Composite routing cannot hide a child contract mismatch", "[composite_scan]")
{
  auto child            = std::make_shared<test_child>();
  child->split_contract = 78;
  harness h({child});
  h.run(0);
  auto input  = h.run(1);
  auto output = h.coalesce(1, *input);
  CHECK(output->contract_id() == 77);
  CHECK_THROWS_WITH(h.driver.validator()->inspect({77, {}, {}}, *output),
                    "scan split contract mismatch: expected 77, got 78");
}
