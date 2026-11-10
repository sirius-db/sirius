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

/*
 * Tests `build_filter_construction`: filters constructed from any split of a certified build, in
 * any order and from several streams, equal those of the same build as one batch (O1) and those of
 * the column constructors of the same kind; claims are validated; contributions never wait on the
 * host; failures join accepted work; and the drop rules (overflow, DECIMAL128 fit, root admission)
 * remove only the affected filter. Also pins `in_list_storage_upper_bound` against the sets it
 * bounds.
 */

#include "operator/construction_parity_utils.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <helper/numeric_narrowing.hpp>
#include <op/dynamic_filter/detail/bounds_accumulator.hpp>
#include <op/dynamic_filter/detail/exact_key_stager.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>

namespace {

namespace op   = sirius::op;
namespace acc  = sirius::test::accumulation;
namespace cons = sirius::test::construction;
using sirius::test::columns::families;
using sirius::test::columns::make_column;
using sirius::test::columns::validity;
using sirius::test::columns::values;

/// Uneven splits of @p column into @p pieces batches; five pieces include an empty batch.
std::vector<std::vector<cudf::column_view>> split(cudf::column_view const& column,
                                                  std::size_t pieces)
{
  std::vector<cudf::size_type> bounds{0};
  if (pieces == 2) { bounds.push_back(column.size() / 3); }
  if (pieces == 5) {
    bounds.insert(bounds.end(), {1, 1, column.size() / 2, column.size() / 2 + 1});
  }
  bounds.push_back(column.size());
  std::vector<std::vector<cudf::column_view>> tables;
  for (std::size_t index = 0; index + 1 < bounds.size(); ++index) {
    tables.push_back({cudf::slice(column, {bounds[index], bounds[index + 1]}).front()});
  }
  return tables;
}

/**
 * @brief The membership sizing a join is planned with. Construction always reads the device L2, so
 * every sizing uses it; a zero IN-list fraction makes a Bloom key of any build too large for a
 * small IN-list.
 */
struct sizing {
  std::string_view name;
  double inlist_max_l2_fraction;
};

std::array<sizing, 2> constexpr sizings{{{"bloom", 0.0}, {"exact", 0.125}}};

/// Leases a construction for @p subject over @p batches, as a session's creator does.
op::build_filter_construction create(acc::fixture const& fixture,
                                     cons::join const& subject,
                                     std::vector<acc::batch_ptr> const& batches)
{
  auto certified = acc::fixture::certify(batches);
  REQUIRE(std::holds_alternative<op::complete_build_inventory>(certified));
  auto const& inventory = std::get<op::complete_build_inventory>(certified);
  auto construction     = op::build_filter_construction::try_create(
    op::plan_construction(inventory.facts(),
                          subject.plan.admitted_keys(),
                          op::binding_facts_of(subject.plan),
                          subject.limits),
    inventory,
    fixture.spaces);
  REQUIRE(construction);
  return std::move(*construction);
}

/// Finishes @p construction on GPU 0's publication stream.
op::build_filter_construction::constructed_filters finish(
  acc::fixture const& fixture, op::build_filter_construction& construction)
{
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto finished = construction.finish({.space   = fixture.gpu(0),
                                       .stream  = fixture.publication_stream(0),
                                       .scratch = fixture.gpu(0).get_default_allocator()});
  REQUIRE(finished);
  return std::move(*finished);
}

/// Contributes every batch of @p batches, in inventory order, on GPU 0's task stream.
void contribute_all(acc::fixture const& fixture,
                    op::build_filter_construction& construction,
                    std::vector<acc::batch_ptr> const& batches,
                    std::optional<rmm::device_async_resource_ref> scratch = std::nullopt)
{
  auto certified        = acc::fixture::certify(batches);
  auto const& inventory = std::get<op::complete_build_inventory>(certified);
  auto const stream     = fixture.task_stream(0);
  for (auto const& batch : batches) {
    auto source     = batch->to_read_only();
    auto const view = sirius::get_cudf_table_view(source);
    auto const index =
      static_cast<std::size_t>(inventory.find(batch->get_batch_id()) - inventory.batches().data());
    auto const ticket =
      construction.reserve(index, rmm::cuda_device_id{0}, op::batch_key_facts::of(view));
    REQUIRE(ticket);
    REQUIRE(construction.contribute(
      *ticket, view, stream, scratch.value_or(fixture.gpu(0).get_default_allocator())));
    stream.sync();
  }
}

/// Finishes @p construction on GPU 0 as the task that owns @p stream and its thread's tracker.
op::build_filter_construction::constructed_filters finish_on(
  acc::fixture const& fixture,
  op::build_filter_construction& construction,
  ::cuda::stream_ref stream)
{
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto finished = construction.finish(
    {.space = fixture.gpu(0), .stream = stream, .scratch = fixture.gpu(0).get_default_allocator()});
  REQUIRE(finished);
  return std::move(*finished);
}

/**
 * @brief A task's reservation attached to the calling thread's tracker, as `gpu_pipeline_task`
 * attaches it before running operators.
 */
class tracked_task {
 public:
  tracked_task(cucascade::memory::memory_space& space,
               std::size_t bytes,
               ::cuda::stream_ref stream,
               std::unique_ptr<cucascade::memory::reservation_limit_policy> policy = nullptr)
    : _allocator{space.get_memory_resource_of<cucascade::memory::Tier::GPU>()}, _stream{stream}
  {
    REQUIRE(_allocator != nullptr);
    auto reservation = space.make_reservation_or_null(bytes);
    REQUIRE(reservation);
    REQUIRE(_allocator->attach_reservation_to_tracker(
      _stream, std::move(reservation), std::move(policy)));
  }
  ~tracked_task() { _allocator->reset_stream_reservation(_stream); }
  tracked_task(tracked_task const&)            = delete;
  tracked_task& operator=(tracked_task const&) = delete;

 private:
  cucascade::memory::reservation_aware_resource_adaptor* _allocator;
  ::cuda::stream_ref _stream;
};

/// Requires @p left and @p right to hold the same filters for key 0.
void require_same_key(cons::constructed const& left,
                      cons::constructed const& right,
                      cudf::column_view const& probe,
                      ::cuda::stream_ref stream)
{
  auto const* a = cons::key_of(left, 0);
  auto const* b = cons::key_of(right, 0);
  REQUIRE((a == nullptr) == (b == nullptr));
  if (a == nullptr) { return; }
  CHECK(a->domain == b->domain);
  CHECK(a->dropped == b->dropped);
  cons::require_same_membership(a->membership.get(), b->membership.get(), probe, stream);
  cons::require_same_zone_map(a->zone_map.get(), b->zone_map.get(), stream);
}

}  // namespace

TEST_CASE("filters from any split equal one batch and the column constructors",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.publication_stream(0);
  for (auto const& tested : families()) {
    for (auto const& sized : sizings) {
      CAPTURE(tested.name, sized.name);
      auto const build  = tested.carrier(stream);
      auto const oracle = tested.oracle(stream);
      cons::join const subject{
        fixture, {tested.construction}, cons::l2_bytes(), true, sized.inlist_max_l2_fraction};
      auto const whole =
        cons::construct(fixture, subject, cons::make_batches(fixture, {{build->view()}}));
      auto const* single = cons::key_of(whole, 0);
      REQUIRE(single != nullptr);
      REQUIRE(single->membership);
      REQUIRE(single->zone_map);

      // The whole build equals the column constructor of the same kind over the whole build, and
      // its bounds equal one reduction; its one key is judged and kept. The plan builds at the
      // carrier when it stays in the recorded type's family.
      auto const& built_from =
        whole.plan.keys.front().construction_type == build->type() ? build->view() : oracle->view();
      cons::require_same_membership(
        single->membership.get(),
        cons::column_oracle(*single->membership, built_from, stream).get(),
        oracle->view(),
        stream);
      cons::require_zone_map_bounds(single->zone_map.get(), oracle->view(), stream);
      CHECK(whole.plan.selection.keys_considered == 1);
      CHECK(whole.plan.selection.keys_with_known_domain == 0);
      CHECK(whole.plan.selection.keys_skipped_domain_gate == 0);
      CHECK(whole.plan.selection.keys_skipped_type_mismatch == 0);

      // O1: every split, also contributed in reverse order, equals the single batch.
      for (std::size_t const pieces : {2, 5}) {
        auto const batches = cons::make_batches(fixture, split(build->view(), pieces));
        require_same_key(whole, cons::construct(fixture, subject, batches), oracle->view(), stream);
        std::vector<std::size_t> reversed;
        for (auto index = batches.size(); index > 0; --index) {
          reversed.push_back(index - 1);
        }
        require_same_key(
          whole, cons::construct(fixture, subject, batches, reversed), oracle->view(), stream);
      }
    }
  }
}

TEST_CASE("a small build constructs the column constructor's small IN-list from any split",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.publication_stream(0);
  auto const int64  = cudf::data_type{cudf::type_id::INT64};
  auto const keys   = make_column(int64, values<std::int64_t>(10), stream, validity(10));
  cons::join const subject{fixture, {int64}, cons::l2_bytes(), true};
  auto const whole =
    cons::construct(fixture, subject, cons::make_batches(fixture, {{keys->view()}}));
  auto const* single = cons::key_of(whole, 0);
  REQUIRE(single != nullptr);
  CHECK(dynamic_cast<op::sirius_dynamic_small_in_list_filter const*>(single->membership.get()) !=
        nullptr);
  cons::require_same_membership(
    single->membership.get(),
    cons::column_oracle(*single->membership, keys->view(), stream).get(),
    keys->view(),
    stream);
  require_same_key(
    whole,
    cons::construct(fixture, subject, cons::make_batches(fixture, split(keys->view(), 5))),
    keys->view(),
    stream);
}

TEST_CASE("heterogeneous carriers construct the recorded type's filters",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.publication_stream(0);
  auto const int64  = cudf::data_type{cudf::type_id::INT64};
  for (std::size_t const l2 : {std::size_t{0}, cons::l2_bytes()}) {
    CAPTURE(l2);
    auto const narrow =
      make_column(cudf::data_type{cudf::type_id::INT16}, values<std::int16_t>(100), stream);
    auto const middle =
      make_column(cudf::data_type{cudf::type_id::INT32}, values<std::int32_t>(300), stream);
    auto const native = make_column(int64, values<std::int64_t>(50), stream);
    cons::join const subject{fixture, {int64}, l2, true};
    auto const mixed = cons::construct(
      fixture,
      subject,
      cons::make_batches(fixture, {{narrow->view()}, {middle->view()}, {native->view()}}));

    // The oracle is the concatenation cast to the recorded type, as one batch.
    std::vector<std::unique_ptr<cudf::column>> cast;
    for (auto const* part : {narrow.get(), middle.get(), native.get()}) {
      cast.push_back(cudf::cast(part->view(), int64, stream));
    }
    auto const whole = cudf::concatenate(
      std::vector<cudf::column_view>{cast[0]->view(), cast[1]->view(), cast[2]->view()}, stream);
    stream.sync();  // `make_batches` copies on its own stream.
    auto const single =
      cons::construct(fixture, subject, cons::make_batches(fixture, {{whole->view()}}));
    require_same_key(mixed, single, whole->view(), stream);
    CHECK(cons::key_of(mixed, 0)->domain == op::classify_membership_key(int64));
  }
}

TEST_CASE("concurrent contributions on several streams construct the same filters",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.publication_stream(0);
  for (auto const& tested : families()) {
    for (std::size_t const l2 : {std::size_t{0}, cons::l2_bytes()}) {
      CAPTURE(tested.name, l2);
      auto const build = tested.carrier(stream);
      cons::join const subject{fixture, {tested.construction}, l2, true};
      auto const batches  = cons::make_batches(fixture, split(build->view(), 5));
      auto const serial   = cons::construct(fixture, subject, batches);
      auto const parallel = cons::construct(fixture, subject, batches, {}, /*threads=*/4);
      require_same_key(serial, parallel, tested.oracle(stream)->view(), stream);
    }
  }
}

TEST_CASE("a construction contribution never waits on the host", "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  rmm::mr::cuda_async_memory_resource async_resource;  // cudaFree would wait for the device.
  rmm::device_async_resource_ref const scratch{async_resource};
  auto const stream = fixture.task_stream(0);
  for (auto const& tested : families()) {
    if (tested.construction.id() == cudf::type_id::STRING) { continue; }  // Its bounds wait.
    for (std::size_t const l2 : {std::size_t{0}, cons::l2_bytes()}) {
      CAPTURE(tested.name, l2);
      auto const build = tested.carrier(stream);
      cons::join const subject{fixture, {tested.construction}, l2, true};
      auto const batches = cons::make_batches(fixture, {{build->view()}});
      auto construction  = create(fixture, subject, batches);
      auto source        = batches.front()->to_read_only();
      auto const view    = sirius::get_cudf_table_view(source);
      auto const ticket =
        construction.reserve(0, rmm::cuda_device_id{0}, op::batch_key_facts::of(view));
      REQUIRE(ticket);
      // The first contribution loads the kernels lazily; a repeat of it is gated.
      REQUIRE(construction.contribute(*ticket, view, stream, scratch));
      stream.sync();
      acc::stream_gate gate{stream};
      REQUIRE(construction.contribute(*ticket, view, stream, scratch));
      cudaEvent_t after{};
      REQUIRE(cudaEventCreateWithFlags(&after, cudaEventDisableTiming) == cudaSuccess);
      REQUIRE(cudaEventRecord(after, stream.get()) == cudaSuccess);
      CHECK(cudaEventQuery(after) == cudaErrorNotReady);
      gate.open();
      REQUIRE(cudaEventSynchronize(after) == cudaSuccess);
      REQUIRE(cudaEventDestroy(after) == cudaSuccess);
      CHECK_FALSE(gate.timed_out());
    }
  }
}

TEST_CASE("a claim must match the inventory's rows, carriers, and null counts",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.publication_stream(0);
  auto const int32  = cudf::data_type{cudf::type_id::INT32};
  auto const keys   = make_column(int32, values<std::int32_t>(20), stream, validity(20));
  cons::join const subject{fixture, {int32}, cons::l2_bytes(), true};
  auto const construction = create(fixture, subject, cons::make_batches(fixture, {{keys->view()}}));
  auto const facts        = op::batch_key_facts::of(cudf::table_view{{keys->view()}});
  auto const device       = rmm::cuda_device_id{0};
  REQUIRE(facts.columns[0].null_count);
  CHECK(construction.reserve(0, device, facts));
  CHECK_FALSE(construction.reserve(1, device, facts));
  auto other_nulls                  = facts;
  other_nulls.columns[0].null_count = *facts.columns[0].null_count + 1;
  CHECK_FALSE(construction.reserve(0, device, other_nulls));
  auto other_carrier               = facts;
  other_carrier.columns[0].carrier = cudf::data_type{cudf::type_id::INT16};
  CHECK_FALSE(construction.reserve(0, device, other_carrier));
  auto other_rows = facts;
  ++other_rows.rows;
  CHECK_FALSE(construction.reserve(0, device, other_rows));
  auto missing = facts;
  missing.columns.clear();
  CHECK_FALSE(construction.reserve(0, device, missing));
}

TEST_CASE("an overflowed staging slice drops only that key's IN-list",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.publication_stream(0);
  auto const int32  = cudf::data_type{cudf::type_id::INT32};
  auto const keys   = make_column(int32, values<std::int32_t>(20), stream);
  cons::join const subject{fixture, {int32, int32}, cons::l2_bytes(), true};
  auto const batches = cons::make_batches(fixture, {{keys->view(), keys->view()}});
  auto construction  = create(fixture, subject, batches);
  auto source        = batches.front()->to_read_only();
  auto const view    = sirius::get_cudf_table_view(source);
  auto ticket = construction.reserve(0, rmm::cuda_device_id{0}, op::batch_key_facts::of(view));
  REQUIRE(ticket);
  cons::test_access::shrink_slice(*ticket, 1);
  auto const task = fixture.task_stream(0);
  REQUIRE(construction.contribute(*ticket, view, task, fixture.gpu(0).get_default_allocator()));
  task.sync();
  auto const finished = finish(fixture, construction);
  CHECK(finished.keys[0].membership);
  CHECK_FALSE(finished.keys[1].membership);
  CHECK(finished.keys[1].dropped == op::dynamic_filter_key_reason::construction_invariant);
  CHECK(finished.keys[1].zone_map);
  CHECK(finished.counts.filters_skipped_construction_invariant == 1);
}

TEST_CASE("a failed contribution joins the work already queued on its stream",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  auto const int64  = cudf::data_type{cudf::type_id::INT64};
  auto const keys   = make_column(int64, values<std::int64_t>(600), stream);
  // No L2 makes a Bloom key, whose inserts are queued before the bounds accumulator fails.
  cons::join const subject{fixture, {int64}, 0, true};
  auto const batches = cons::make_batches(fixture, {{keys->view()}});
  auto construction  = create(fixture, subject, batches);
  auto source        = batches.front()->to_read_only();
  auto const view    = sirius::get_cudf_table_view(source);
  auto ticket = construction.reserve(0, rmm::cuda_device_id{0}, op::batch_key_facts::of(view));
  REQUIRE(ticket);
  auto const scratch = fixture.gpu(0).get_default_allocator();
  REQUIRE(construction.contribute(*ticket, view, stream, scratch));
  stream.sync();
  cons::test_access::corrupt_batch(*ticket);

  acc::stream_gate gate{stream};
  auto opener = std::async(std::launch::async, [&gate] {
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    gate.open();
  });
  CHECK_THROWS_AS(construction.contribute(*ticket, view, stream, scratch),
                  op::detail::accumulation_invariant_error);
  // The failure propagated only after the queued inserts ran.
  CHECK(cudaStreamQuery(stream.get()) == cudaSuccess);
  opener.get();
  CHECK_FALSE(gate.timed_out());
}

TEST_CASE("an all-null key constructs an empty membership filter and no zone map",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.publication_stream(0);
  auto const int32  = cudf::data_type{cudf::type_id::INT32};
  auto const nulls =
    make_column(int32, values<std::int32_t>(6), stream, std::vector<bool>(6, false));
  for (auto const& sized : sizings) {
    CAPTURE(sized.name);
    cons::join const subject{
      fixture, {int32}, cons::l2_bytes(), true, sized.inlist_max_l2_fraction};
    auto const result =
      cons::construct(fixture, subject, cons::make_batches(fixture, split(nulls->view(), 2)));
    auto const* key = cons::key_of(result, 0);
    REQUIRE(key != nullptr);
    REQUIRE(key->membership);
    CHECK_FALSE(key->zone_map);
    cons::require_same_membership(
      key->membership.get(),
      cons::column_oracle(*key->membership, nulls->view(), stream).get(),
      nulls->view(),
      stream);
  }
}

TEST_CASE("a DECIMAL128 key publishes membership only when its values fit int64",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream  = fixture.publication_stream(0);
  auto const decimal = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  for (bool const large : {false, true}) {
    for (auto const& sized : sizings) {
      CAPTURE(large, sized.name);
      auto wide = values<__int128_t>(40);
      if (large) { wide[17] = __int128_t{1} << 70; }  // Wraps when narrowed to int64.
      auto const keys = make_column(decimal, wide, stream);
      cons::join const subject{
        fixture, {decimal}, cons::l2_bytes(), true, sized.inlist_max_l2_fraction};
      auto const result =
        cons::construct(fixture, subject, cons::make_batches(fixture, split(keys->view(), 2)));
      auto const* key = cons::key_of(result, 0);
      REQUIRE(key != nullptr);
      CHECK(key->zone_map);  // Bounds are exact at DECIMAL128 either way.
      CHECK(static_cast<bool>(key->membership) == !large);
      CHECK((key->dropped == op::dynamic_filter_key_reason::decimal128_unfit) == large);
      // The fit rule, applied to the whole column independently.
      CHECK(sirius::column_values_fit(keys->view(),
                                      cudf::data_type{cudf::type_id::DECIMAL64, -2},
                                      stream,
                                      cudf::get_current_device_resource_ref()) == !large);
      if (!large) {
        cons::require_same_membership(
          key->membership.get(),
          cons::column_oracle(*key->membership, keys->view(), stream).get(),
          keys->view(),
          stream);
      }
      cons::require_zone_map_bounds(key->zone_map.get(), keys->view(), stream);
    }
  }
}

TEST_CASE("finish on a thread holding the task's reservation builds the root filters",
          "[dynamic_filter][construction][tracked]")
{
  acc::fixture fixture;  // Per-thread tracking, as the engine runs.
  auto const stream = fixture.publication_stream(0);
  auto const int64  = cudf::data_type{cudf::type_id::INT64};
  auto const keys   = make_column(int64, values<std::int64_t>(600), stream, validity(600));
  cons::join const subject{fixture, {int64}, cons::l2_bytes(), true};
  auto const batches = cons::make_batches(fixture, {{keys->view()}});
  auto construction  = create(fixture, subject, batches);
  REQUIRE(std::holds_alternative<op::exact_membership>(construction.plan().keys[0].membership));
  contribute_all(fixture, construction, batches);

  auto const task = fixture.task_stream(0);
  tracked_task const tracked{fixture.gpu(0), 64 * acc::mib, task};
  auto const finished = finish_on(fixture, construction, task);
  CHECK(finished.keys[0].membership);
  CHECK(finished.keys[0].zone_map);
  CHECK_FALSE(finished.keys[0].dropped);
  CHECK(finished.counts.filters_skipped_construction_admission == 0);
}

TEST_CASE("a root filter without room left in the task's memory is dropped alone",
          "[dynamic_filter][construction][tracked]")
{
  acc::fixture fixture;
  auto const stream      = fixture.publication_stream(0);
  auto const int64       = cudf::data_type{cudf::type_id::INT64};
  std::size_t const rows = 200'000;
  auto const keys        = make_column(int64, values<std::int64_t>(rows), stream);
  // An L2 this large plans a hash IN-list for the key.
  cons::join const subject{fixture, {int64}, std::size_t{1} << 30, true, /*fraction=*/1.0};
  auto const batches = cons::make_batches(fixture, {{keys->view()}});
  auto construction  = create(fixture, subject, batches);
  REQUIRE(std::holds_alternative<op::exact_membership>(construction.plan().keys[0].membership));
  contribute_all(fixture, construction, batches);

  // The task holds 1 MiB, and every unreserved byte is taken elsewhere: the IN-list's bound does
  // not fit the room left, the zone map does.
  auto& space = fixture.gpu(0);
  REQUIRE(op::in_list_storage_upper_bound(rows, op::membership_key_rep::i64) > acc::mib);
  auto const task = fixture.task_stream(0);
  std::unique_ptr<rmm::device_buffer> filler;  // Released after the task's tracker detaches.
  tracked_task const tracked{space, acc::mib, task};
  filler              = std::async(std::launch::async, [&] {
             auto const unreserved = space.get_available_memory() / 256 * 256;
             return std::make_unique<rmm::device_buffer>(
               unreserved, stream, space.get_default_allocator());
           }).get();
  auto const finished = finish_on(fixture, construction, task);
  CHECK_FALSE(finished.keys[0].membership);
  CHECK(finished.keys[0].dropped == op::dynamic_filter_key_reason::construction_admission);
  CHECK(finished.keys[0].zone_map);
  CHECK(finished.counts.filters_skipped_construction_admission == 1);
}

TEST_CASE("a root filter whose construction runs out of the task's memory is dropped alone",
          "[dynamic_filter][construction][tracked]")
{
  acc::fixture fixture;
  auto const stream      = fixture.publication_stream(0);
  auto const int64       = cudf::data_type{cudf::type_id::INT64};
  std::size_t const rows = 20'000;
  auto const keys        = make_column(int64, values<std::int64_t>(rows), stream);
  cons::join const subject{fixture, {int64}, std::size_t{1} << 30, true, /*fraction=*/1.0};
  auto const batches = cons::make_batches(fixture, {{keys->view()}});
  auto construction  = create(fixture, subject, batches);
  REQUIRE(std::holds_alternative<op::exact_membership>(construction.plan().keys[0].membership));
  contribute_all(fixture, construction, batches);

  // The room check counts unreserved memory, but this task may not grow past its 64 KiB: the set's
  // allocation fails, the zone map fits.
  std::size_t const held = 64 * 1024;
  REQUIRE(op::in_list_storage_upper_bound(rows, op::membership_key_rep::i64) > held);
  auto const task = fixture.task_stream(0);
  tracked_task const tracked{fixture.gpu(0),
                             held,
                             task,
                             std::make_unique<cucascade::memory::fail_reservation_limit_policy>()};
  auto const finished = finish_on(fixture, construction, task);
  CHECK_FALSE(finished.keys[0].membership);
  CHECK(finished.keys[0].dropped == op::dynamic_filter_key_reason::construction_admission);
  CHECK(finished.keys[0].zone_map);
  CHECK(finished.counts.filters_skipped_construction_admission == 1);
}

TEST_CASE("root construction takes no more than its bound from the task's allocator",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.publication_stream(0);
  auto const int64  = cudf::data_type{cudf::type_id::INT64};
  // Narrower carriers restore their bounds at publish; the key builds a hash IN-list and a zone
  // map.
  auto const narrow =
    make_column(cudf::data_type{cudf::type_id::INT16}, values<std::int16_t>(300), stream);
  auto const middle =
    make_column(cudf::data_type{cudf::type_id::INT32}, values<std::int32_t>(900), stream);
  auto const native = make_column(int64, values<std::int64_t>(500), stream, validity(500));
  cons::join const subject{fixture, {int64}, std::size_t{1} << 30, true, /*fraction=*/1.0};
  auto const batches =
    cons::make_batches(fixture, {{narrow->view()}, {middle->view()}, {native->view()}});
  auto construction = create(fixture, subject, batches);
  REQUIRE(std::holds_alternative<op::exact_membership>(construction.plan().keys[0].membership));
  contribute_all(fixture, construction, batches);

  rmm::mr::cuda_async_memory_resource async_resource;
  rmm::mr::statistics_resource_adaptor stats{
    cuda::mr::any_resource<cuda::mr::device_accessible>{async_resource}};
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto const finished = construction.finish(
    {.space = fixture.gpu(0), .stream = stream, .scratch = rmm::device_async_resource_ref{stats}});
  REQUIRE(finished);
  REQUIRE(finished->keys[0].membership);
  REQUIRE(finished->keys[0].zone_map);
  auto const peak = static_cast<std::size_t>(stats.get_bytes_counter().peak);
  CHECK(peak > 0);
  CHECK(peak <= construction.root_construction_bytes() + construction.task_scratch_bytes());
}

TEST_CASE("task scratch covers contributions and the Bloom merge", "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream      = fixture.publication_stream(0);
  auto const int32       = cudf::data_type{cudf::type_id::INT32};
  auto const keys        = make_column(int32, values<std::int32_t>(600), stream);
  auto const batches     = cons::make_batches(fixture, {{keys->view()}});
  auto const scratch_for = [&](std::size_t l2, bool zone_maps) {
    return create(fixture, cons::join{fixture, {int32}, l2, zone_maps}, batches)
      .task_scratch_bytes();
  };
  // One GPU: a Bloom key merges nothing; an exact key ranks and scans; bounds reduce.
  auto const exact  = op::detail::exact_key_stager::contribution_scratch_bytes(600);
  auto const bounds = op::detail::bounds_accumulator::contribution_scratch_bytes(600);
  CHECK(exact >= 600 * sizeof(cudf::size_type));
  CHECK(bounds > 0);
  CHECK(scratch_for(0, false) == 0);
  CHECK(scratch_for(0, true) == bounds);
  CHECK(scratch_for(cons::l2_bytes(), false) == exact);
  CHECK(scratch_for(cons::l2_bytes(), true) == exact + bounds);
}

TEST_CASE("a contribution takes no more than the task scratch bound, STRING keys included",
          "[dynamic_filter][construction]")
{
  acc::fixture fixture;
  auto const stream = fixture.publication_stream(0);
  for (auto const& tested : {sirius::test::columns::string_family(), families().front()}) {
    for (std::size_t const l2 : {std::size_t{0}, std::size_t{1} << 30}) {
      CAPTURE(tested.name, l2);
      auto const build = tested.carrier(stream);
      cons::join const subject{fixture, {tested.construction}, l2, true, /*fraction=*/1.0};
      auto const batches = cons::make_batches(fixture, {{build->view()}});
      auto construction  = create(fixture, subject, batches);
      rmm::mr::cuda_async_memory_resource async_resource;
      rmm::mr::statistics_resource_adaptor stats{
        cuda::mr::any_resource<cuda::mr::device_accessible>{async_resource}};
      contribute_all(fixture, construction, batches, rmm::device_async_resource_ref{stats});
      CHECK(static_cast<std::size_t>(stats.get_bytes_counter().peak) <=
            construction.task_scratch_bytes());
    }
  }
}

TEST_CASE("the IN-list storage bound covers every set it may build and is never zero",
          "[dynamic_filter][construction][in_list_bound]")
{
  for (auto const rep : {op::membership_key_rep::i32,
                         op::membership_key_rep::i64,
                         op::membership_key_rep::u32,
                         op::membership_key_rep::u64}) {
    auto const rep_bytes = op::membership_rep_bytes(rep);
    for (std::size_t keys = 0; keys < 5000; ++keys) {
      auto const bound = op::in_list_storage_upper_bound(keys, rep);
      REQUIRE(bound > 0);
      // Every capacity factor the constructor may choose, across many cuco extent boundaries.
      for (std::size_t const factor : {2, 3, 4}) {
        REQUIRE(bound >= op::sirius_dynamic_in_list_filter::set_slots(keys, factor) * rep_bytes +
                           keys * rep_bytes);
      }
    }
  }
}

TEST_CASE("constructed IN-list sets fit the storage bound",
          "[dynamic_filter][construction][in_list_bound]")
{
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  for (std::size_t keys = 1; keys < 2048; keys += 97) {
    CAPTURE(keys);
    auto const column =
      make_column(cudf::data_type{cudf::type_id::INT64}, values<std::int64_t>(keys), stream);
    op::sirius_dynamic_in_list_filter const set{
      column->view(), stream, cudf::get_current_device_resource_ref()};
    stream.sync();
    CHECK(set.storage_bytes() > 0);
    CHECK(set.storage_bytes() <=
          op::in_list_storage_upper_bound(keys, op::membership_key_rep::i64));
  }
}
