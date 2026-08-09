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

#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <future>
#include <latch>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr int kDeviceId                 = 0;
constexpr std::size_t kProbeColumnIndex = 7;

using sirius::op::dynamic_filter_publish_plan;
using sirius::op::dynamic_filter_route_class;

constexpr auto kInt64 = cudf::data_type{cudf::type_id::INT64};

template <typename MemoryManager>
std::vector<sirius::op::dynamic_filter_replica_space> get_replica_spaces(
  MemoryManager& memory_manager)
{
  auto const gpu_spaces  = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto const host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(gpu_spaces.size() == 1);
  REQUIRE_FALSE(host_spaces.empty());

  auto* gpu_space = memory_manager.get_memory_space(cucascade::memory::Tier::GPU,
                                                    gpu_spaces.front()->get_device_id());
  REQUIRE(gpu_space != nullptr);
  auto const local_host =
    std::find_if(host_spaces.begin(), host_spaces.end(), [gpu_space](auto const* host_space) {
      return host_space->get_device_id() == gpu_space->get_device_id();
    });
  auto const* host_space = local_host == host_spaces.end() ? host_spaces.front() : *local_host;
  return {{*gpu_space, *host_space}};
}

// Build an admitted INT64 key; a nonzero domain marks it unique and enables coverage gating.
dynamic_filter_publish_plan::admitted_key make_int64_key(std::size_t condition_index,
                                                         cudf::size_type build_key_ordinal,
                                                         std::size_t domain_cardinality = 0)
{
  return dynamic_filter_publish_plan::admitted_key{
    .planner_condition_index      = condition_index,
    .build_key_ordinal            = build_key_ordinal,
    .storage_type                 = kInt64,
    .key_shape                    = {},
    .build_key_domain_cardinality = domain_cardinality,
    .build_key_proven_unique      = domain_cardinality > 0};
}

// GPU resources and key columns used by publisher tests.
struct publisher_fixture {
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
  decltype(sirius::test::operator_utils::initialize_memory_manager(1)) memory_manager =
    sirius::test::operator_utils::initialize_memory_manager(1);
  std::vector<sirius::op::dynamic_filter_replica_space> replica_spaces =
    get_replica_spaces(*memory_manager);
  rmm::cuda_stream_view stream = replica_spaces.front().get_gpu_space().acquire_stream();

  std::vector<std::unique_ptr<cudf::column>> columns;

  // Append an INT64 sequence column.
  void add_key_column(std::size_t rows, std::int64_t first = 0)
  {
    auto& source_space = replica_spaces.front().get_gpu_space();
    columns.push_back(cudf::sequence(static_cast<cudf::size_type>(rows),
                                     cudf::numeric_scalar<std::int64_t>(first, true, stream),
                                     cudf::numeric_scalar<std::int64_t>(1, true, stream),
                                     stream,
                                     source_space.get_default_allocator()));
  }

  [[nodiscard]] cudf::table_view build_view() const
  {
    std::vector<cudf::column_view> views;
    views.reserve(columns.size());
    for (auto const& column : columns) {
      views.push_back(column->view());
    }
    return cudf::table_view{views};
  }
};

// Copy INT64 values into a device column.
std::unique_ptr<cudf::column> make_int64_values(publisher_fixture const& fixture,
                                                std::vector<std::int64_t> const& values)
{
  auto column    = cudf::make_numeric_column(kInt64,
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          fixture.stream,
                                          cudf::get_current_device_resource_ref());
  auto const err = cudaMemcpyAsync(column->mutable_view().data<std::int64_t>(),
                                   values.data(),
                                   values.size() * sizeof(std::int64_t),
                                   cudaMemcpyHostToDevice,
                                   fixture.stream.value());
  REQUIRE(err == cudaSuccess);
  fixture.stream.synchronize();
  return column;
}

// Apply a membership filter and copy its keep mask to the host.
std::vector<std::uint8_t> membership_mask(sirius::op::sirius_dynamic_filter const& filter,
                                          cudf::column_view const& probe,
                                          publisher_fixture const& fixture)
{
  auto const* applicable = dynamic_cast<sirius::op::sirius_mask_applicable const*>(&filter);
  REQUIRE(applicable != nullptr);
  auto const mask = applicable->compute_mask(
    probe, kDeviceId, fixture.stream, cudf::get_current_device_resource_ref());
  REQUIRE(mask != nullptr);
  REQUIRE(mask->view().type().id() == cudf::type_id::BOOL8);
  std::vector<std::uint8_t> host(static_cast<std::size_t>(mask->view().size()));
  auto const err = cudaMemcpyAsync(host.data(),
                                   mask->view().data<bool>(),
                                   host.size() * sizeof(bool),
                                   cudaMemcpyDeviceToHost,
                                   fixture.stream.value());
  REQUIRE(err == cudaSuccess);
  fixture.stream.synchronize();
  return host;
}

template <typename Filter>
std::size_t count_filters_of_kind(
  std::vector<std::shared_ptr<sirius::op::sirius_dynamic_filter const>> const& snapshot)
{
  return static_cast<std::size_t>(
    std::count_if(snapshot.begin(), snapshot.end(), [](auto const& filter) {
      return dynamic_cast<Filter const*>(filter.get()) != nullptr;
    }));
}

template <typename ExpectedFilter>
void require_published_membership(std::size_t rows)
{
  publisher_fixture fixture;
  fixture.add_key_column(rows);

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  dynamic_filter_publish_plan plan{
    {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces)};

  auto const& keys = *fixture.columns.front();
  if constexpr (std::is_same_v<ExpectedFilter, sirius::op::sirius_dynamic_small_in_list_filter>) {
    REQUIRE(sirius::op::sirius_dynamic_small_in_list_filter::supports(keys.view()));
  } else {
    REQUIRE_FALSE(sirius::op::sirius_dynamic_small_in_list_filter::supports(keys.view()));
    int l2_bytes = 0;
    REQUIRE(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, kDeviceId) == cudaSuccess);
    REQUIRE(l2_bytes > 0);
    REQUIRE(sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(rows, kInt64) <=
            static_cast<std::size_t>(l2_bytes));
  }

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);

  auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(snapshot.size() == 1);
  auto const* selected = dynamic_cast<ExpectedFilter const*>(snapshot.front().get());
  REQUIRE(selected != nullptr);
  REQUIRE(selected->is_available_on_device(kDeviceId));
  REQUIRE(selected->size() == rows);
  REQUIRE(selected->replica_count() == 1);
  if constexpr (std::is_same_v<ExpectedFilter, sirius::op::sirius_dynamic_in_list_filter>) {
    REQUIRE(selected->has_persistent_set());
  }
}

// Require a publication attempt with no considered keys, built filters, or active targets.
void require_nothing_published(sirius::op::dynamic_filter_publication_outcome const& outcome)
{
  REQUIRE(outcome.keys_considered == 0);
  REQUIRE(outcome.keys_skipped_domain_gate == 0);
  REQUIRE(outcome.keys_skipped_type_mismatch == 0);
  REQUIRE(outcome.membership_filters_built == 0);
  REQUIRE(outcome.zone_map_filters_built == 0);
  REQUIRE(outcome.active_targets == 0);
  REQUIRE(outcome.filters_pushed == 0);
}

// Return a mutable host memory space for invalid-tier plan tests.
cucascade::memory::memory_space& host_memory_space(publisher_fixture& fixture)
{
  auto const host_spaces =
    fixture.memory_manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE_FALSE(host_spaces.empty());
  auto* space = fixture.memory_manager->get_memory_space(cucascade::memory::Tier::HOST,
                                                         host_spaces.front()->get_device_id());
  REQUIRE(space != nullptr);
  return *space;
}

// Verify that coverage gating uses the selected key's domain, build ordinal, and push ordinal.
void require_domain_gate_skips_only(std::size_t gated_key_index)
{
  constexpr std::size_t kBuildRows = 3;
  // Coverage 3/3 = 1.0 trips the plan's default threshold; 3/1000 = 0.003 stays far below it.
  constexpr std::size_t kCoveredDomain   = kBuildRows;
  constexpr std::size_t kWideDomain      = 1000;
  constexpr std::size_t kKey0PushOrdinal = 3;
  constexpr std::size_t kKey1PushOrdinal = 5;

  publisher_fixture fixture;
  // Disjoint values and reversed build ordinals expose coordinate-space mixups.
  fixture.add_key_column(kBuildRows, 100);
  fixture.add_key_column(kBuildRows, 0);

  auto const domain_of = [gated_key_index](std::size_t key_index) {
    return key_index == gated_key_index ? kCoveredDomain : kWideDomain;
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kKey0PushOrdinal,
                                                   .probe_storage_type   = kInt64},
                                                  {.admitted_key_index   = 1,
                                                   .channel_push_ordinal = kKey1PushOrdinal,
                                                   .probe_storage_type   = kInt64}}});
  dynamic_filter_publish_plan plan{
    {make_int64_key(0, 1, domain_of(0)), make_int64_key(1, 0, domain_of(1))},
    std::move(targets),
    std::move(fixture.replica_spaces)};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);

  REQUIRE(outcome.keys_considered == 2);
  REQUIRE(outcome.keys_skipped_domain_gate == 1);
  REQUIRE(outcome.keys_skipped_type_mismatch == 0);
  REQUIRE(outcome.membership_filters_built == 1);
  REQUIRE(outcome.zone_map_filters_built == 0);
  REQUIRE(outcome.active_targets == 1);
  REQUIRE(outcome.filters_pushed == 1);

  auto const gated_ordinal     = gated_key_index == 0 ? kKey0PushOrdinal : kKey1PushOrdinal;
  auto const surviving_ordinal = gated_key_index == 0 ? kKey1PushOrdinal : kKey0PushOrdinal;
  REQUIRE(channel->filters_for_column(gated_ordinal).empty());
  auto const surviving = channel->filters_for_column(surviving_ordinal);
  REQUIRE(surviving.size() == 1);

  // Applying the filter distinguishes the two build columns, not just their push ordinals.
  auto const probe = make_int64_values(fixture, {0, 100});
  auto const expected =
    gated_key_index == 0 ? std::vector<std::uint8_t>{0, 1} : std::vector<std::uint8_t>{1, 0};
  REQUIRE(membership_mask(*surviving.front(), probe->view(), fixture) == expected);
}

dynamic_filter_publish_plan make_accumulator_plan(
  publisher_fixture& fixture, std::shared_ptr<sirius::op::sirius_dynamic_filter_set> const& channel)
{
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  return dynamic_filter_publish_plan{
    {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces)};
}

cudf::table_view one_column_view(cudf::column const& column)
{
  return cudf::table_view{std::vector<cudf::column_view>{column.view()}};
}

}  // namespace

TEST_CASE("dynamic-filter publisher selects the raw small IN-list", "[dynamic_filter][publisher]")
{
  require_published_membership<sirius::op::sirius_dynamic_small_in_list_filter>(3);
}

TEST_CASE("dynamic-filter publisher falls through to the hash IN-list above the small-list gate",
          "[dynamic_filter][publisher]")
{
  require_published_membership<sirius::op::sirius_dynamic_in_list_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1);
}

TEST_CASE("dynamic-filter publisher fans out sparsely: each target receives only its bound keys",
          "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  // Reverse key-to-column ordinals and use disjoint values to expose coordinate-space mixups.
  fixture.add_key_column(3, 100);
  fixture.add_key_column(3, 0);

  auto channel_a = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto channel_b = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back(
    {.filter_set               = channel_a,
     .route_class              = dynamic_filter_route_class::scan,
     .accepts_zone_map_filters = true,
     .key_bindings             = {
       {.admitted_key_index = 0, .channel_push_ordinal = 3, .probe_storage_type = kInt64}}});
  targets.push_back(
    {.filter_set               = channel_b,
     .route_class              = dynamic_filter_route_class::scan,
     .accepts_zone_map_filters = true,
     .key_bindings             = {
       {.admitted_key_index = 1, .channel_push_ordinal = 5, .probe_storage_type = kInt64}}});
  dynamic_filter_publish_plan plan{{make_int64_key(0, 1), make_int64_key(1, 0)},
                                   std::move(targets),
                                   std::move(fixture.replica_spaces)};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);

  REQUIRE(outcome.keys_considered == 2);
  REQUIRE(outcome.membership_filters_built == 2);
  REQUIRE(outcome.active_targets == 2);
  REQUIRE(outcome.filters_pushed == 2);
  REQUIRE(outcome.keys_skipped_domain_gate == 0);
  REQUIRE(outcome.keys_skipped_type_mismatch == 0);

  REQUIRE(channel_a->filters_for_column(3).size() == 1);
  REQUIRE(channel_a->filters_for_column(5).empty());
  REQUIRE(channel_b->filters_for_column(5).size() == 1);
  REQUIRE(channel_b->filters_for_column(3).empty());

  // Verify key identity, not just placement: apply each published membership filter to a probe
  // column holding one value from each build key's domain. Channel A must represent {0,1,2}
  // (build column 1) and channel B {100,101,102} (build column 0).
  auto const probe = make_int64_values(fixture, {0, 100});
  REQUIRE(membership_mask(*channel_a->filters_for_column(3).front(), probe->view(), fixture) ==
          std::vector<std::uint8_t>{1, 0});
  REQUIRE(membership_mask(*channel_b->filters_for_column(5).front(), probe->view(), fixture) ==
          std::vector<std::uint8_t>{0, 1});
}

TEST_CASE("dynamic-filter publisher applies the domain-coverage gate to each key's own domain",
          "[dynamic_filter][publisher]")
{
  SECTION("the first admitted key covers its domain") { require_domain_gate_skips_only(0); }
  SECTION("the second admitted key covers its domain") { require_domain_gate_skips_only(1); }
}

TEST_CASE("dynamic-filter publisher fails loudly on a plan/runtime key-mapping inconsistency",
          "[dynamic_filter][publisher]")
{
  // Admission normally keeps these fields consistent; exercise the publisher's runtime guard.
  publisher_fixture fixture;
  fixture.add_key_column(3);

  auto make_plan = [&fixture](dynamic_filter_publish_plan::admitted_key key) {
    auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = channel,
                       .route_class              = dynamic_filter_route_class::scan,
                       .accepts_zone_map_filters = true,
                       .key_bindings             = {{.admitted_key_index   = 0,
                                                     .channel_push_ordinal = kProbeColumnIndex,
                                                     .probe_storage_type   = kInt64}}});
    auto replica_spaces = fixture.replica_spaces;  // copy; each section builds its own plan
    return dynamic_filter_publish_plan{{key}, std::move(targets), std::move(replica_spaces)};
  };

  SECTION("build ordinal outside the runtime build table")
  {
    auto const plan = make_plan(make_int64_key(0, 5));
    REQUIRE_THROWS_AS(
      (void)sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream),
      std::logic_error);
  }
  SECTION("recorded storage type disagreeing with the runtime build column skips the key")
  {
    // Advisory data: the join stays authoritative, so a type-derivation disagreement skips the
    // key and is counted rather than failing the query.
    auto key         = make_int64_key(0, 0);
    key.storage_type = cudf::data_type{cudf::type_id::INT32};
    auto const plan  = make_plan(key);
    auto const outcome =
      sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
    REQUIRE(outcome.keys_skipped_type_mismatch == 1);
    REQUIRE(outcome.filters_pushed == 0);
    REQUIRE(outcome.membership_filters_built == 0);
  }
}

TEST_CASE("dynamic-filter publisher suppresses zone maps per binding on probe-type mismatch",
          "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3);

  auto matching_channel   = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto mismatched_channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = matching_channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  targets.push_back(
    {.filter_set               = mismatched_channel,
     .route_class              = dynamic_filter_route_class::scan,
     .accepts_zone_map_filters = true,
     .key_bindings             = {{.admitted_key_index   = 0,
                                   .channel_push_ordinal = kProbeColumnIndex,
                                   .probe_storage_type   = cudf::data_type{cudf::type_id::INT32}}}});
  dynamic_filter_publish_plan plan{{make_int64_key(0, 0)},
                                   std::move(targets),
                                   std::move(fixture.replica_spaces),
                                   {.emit_zone_map_filters = true}};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);

  auto const matching_snapshot = matching_channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(matching_snapshot) ==
          1);
  REQUIRE(
    count_filters_of_kind<sirius::op::sirius_dynamic_small_in_list_filter>(matching_snapshot) == 1);

  // The membership filter still arrives; only the zone map is suppressed for this binding.
  auto const mismatched_snapshot = mismatched_channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(mismatched_snapshot) ==
          0);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_small_in_list_filter>(
            mismatched_snapshot) == 1);
}

TEST_CASE("dynamic-filter publisher keeps zone maps out of membership-only targets",
          "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3);

  auto scan_channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto edge_channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = scan_channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  targets.push_back(
    {.filter_set               = edge_channel,
     .route_class              = dynamic_filter_route_class::direct,
     .accepts_zone_map_filters = false,
     .key_bindings             = {
       {.admitted_key_index = 0, .channel_push_ordinal = 2, .probe_storage_type = kInt64}}});
  dynamic_filter_publish_plan plan{{make_int64_key(0, 0)},
                                   std::move(targets),
                                   std::move(fixture.replica_spaces),
                                   {.emit_zone_map_filters = true}};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);

  auto const scan_snapshot = scan_channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(scan_snapshot) == 1);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_small_in_list_filter>(scan_snapshot) ==
          1);

  auto const edge_snapshot = edge_channel->filters_for_column(2);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(edge_snapshot) == 0);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_small_in_list_filter>(edge_snapshot) ==
          1);
}

TEST_CASE("dynamic-filter publisher completes on a plan with targets but no admitted keys",
          "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3);

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {}});
  dynamic_filter_publish_plan plan{{}, std::move(targets), std::move(fixture.replica_spaces)};
  REQUIRE(plan.enabled());

  // A producer whose keys were all inadmissible still claims publication and publishes nothing.
  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
  REQUIRE_FALSE(channel->has_filters());
}

TEST_CASE("dynamic-filter publisher publishes nothing from an empty build",
          "[dynamic_filter][publisher]")
{
  // An empty build carries no key values to construct from, and its join emits no rows whatever the
  // probe side keeps, so nothing is built, replicated, or pushed. Zone maps are enabled here so the
  // reduction path is skipped too.
  publisher_fixture fixture;
  fixture.add_key_column(0);

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  dynamic_filter_publish_plan plan{{make_int64_key(0, 0)},
                                   std::move(targets),
                                   std::move(fixture.replica_spaces),
                                   {.emit_zone_map_filters = true}};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
  require_nothing_published(outcome);
  REQUIRE(channel->empty());
  REQUIRE_FALSE(channel->has_filters());
}

TEST_CASE("dynamic-filter publisher publishes nothing once every target has drained",
          "[dynamic_filter][publisher]")
{
  // No consumer remains to observe a filter, so construction is skipped.
  publisher_fixture fixture;
  fixture.add_key_column(3);

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  channel->close_for_new_filters();
  REQUIRE_FALSE(channel->accepting_filters());

  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  dynamic_filter_publish_plan plan{
    {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces)};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
  require_nothing_published(outcome);
  REQUIRE(channel->empty());
  REQUIRE_FALSE(channel->has_filters());
}

TEST_CASE("dynamic-filter publisher serves a live target beside a drained one",
          "[dynamic_filter][publisher]")
{
  // The drained target comes first, so a fan-out that stopped at the first unusable target -- or
  // that counted it as active -- fails here.
  publisher_fixture fixture;
  fixture.add_key_column(3);

  auto drained_channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  drained_channel->close_for_new_filters();
  auto live_channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();

  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back(
    {.filter_set               = drained_channel,
     .route_class              = dynamic_filter_route_class::scan,
     .accepts_zone_map_filters = true,
     .key_bindings             = {
       {.admitted_key_index = 0, .channel_push_ordinal = 3, .probe_storage_type = kInt64}}});
  targets.push_back(
    {.filter_set               = live_channel,
     .route_class              = dynamic_filter_route_class::scan,
     .accepts_zone_map_filters = true,
     .key_bindings             = {
       {.admitted_key_index = 0, .channel_push_ordinal = 5, .probe_storage_type = kInt64}}});
  dynamic_filter_publish_plan plan{
    {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces)};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);

  REQUIRE(outcome.keys_considered == 1);
  REQUIRE(outcome.membership_filters_built == 1);
  REQUIRE(outcome.active_targets == 1);
  REQUIRE(outcome.filters_pushed == 1);

  REQUIRE(live_channel->filters_for_column(5).size() == 1);
  REQUIRE(drained_channel->empty());
}

TEST_CASE("dynamic-filter publish plan rejects invalid targets and bindings",
          "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();

  SECTION("null endpoint channel")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = nullptr,
                       .route_class              = dynamic_filter_route_class::scan,
                       .accepts_zone_map_filters = true,
                       .key_bindings             = {}});
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan(
        {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces)),
      std::invalid_argument);
  }

  SECTION("membership-only (direct) target accepting zone maps")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = channel,
                       .route_class              = dynamic_filter_route_class::direct,
                       .accepts_zone_map_filters = true,
                       .key_bindings             = {}});
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan(
        {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces)),
      std::invalid_argument);
  }

  SECTION("binding referencing a nonexistent admitted key")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::scan,
       .accepts_zone_map_filters = true,
       .key_bindings             = {
         {.admitted_key_index = 1, .channel_push_ordinal = 0, .probe_storage_type = kInt64}}});
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan(
        {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces)),
      std::invalid_argument);
  }

  SECTION("one admitted key bound twice on one target")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::scan,
       .accepts_zone_map_filters = true,
       .key_bindings             = {
         {.admitted_key_index = 0, .channel_push_ordinal = 0, .probe_storage_type = kInt64},
         {.admitted_key_index = 0, .channel_push_ordinal = 1, .probe_storage_type = kInt64}}});
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan(
        {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces)),
      std::invalid_argument);
  }

  SECTION("two admitted keys binding one probe column stays legal")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::scan,
       .accepts_zone_map_filters = true,
       .key_bindings             = {
         {.admitted_key_index = 0, .channel_push_ordinal = 4, .probe_storage_type = kInt64},
         {.admitted_key_index = 1, .channel_push_ordinal = 4, .probe_storage_type = kInt64}}});
    REQUIRE_NOTHROW(dynamic_filter_publish_plan({make_int64_key(0, 0), make_int64_key(1, 1)},
                                                std::move(targets),
                                                std::move(fixture.replica_spaces)));
  }
}

TEST_CASE("dynamic-filter publish plan validates a mixed scan-plus-direct target set",
          "[dynamic_filter][publisher]")
{
  // The constructor validates direct-target invariants in the combined route set.
  publisher_fixture fixture;
  auto scan_channel   = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto direct_channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();

  auto make_targets = [&scan_channel, &direct_channel](bool direct_accepts_zone_maps,
                                                       std::size_t direct_admitted_key_index) {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = scan_channel,
       .route_class              = dynamic_filter_route_class::scan,
       .accepts_zone_map_filters = true,
       .key_bindings             = {
         {.admitted_key_index = 0, .channel_push_ordinal = 3, .probe_storage_type = kInt64}}});
    targets.push_back({.filter_set               = direct_channel,
                       .route_class              = dynamic_filter_route_class::direct,
                       .accepts_zone_map_filters = direct_accepts_zone_maps,
                       .key_bindings = {{.admitted_key_index   = direct_admitted_key_index,
                                         .channel_push_ordinal = 9,
                                         .probe_storage_type   = kInt64}}});
    return targets;
  };

  SECTION("both routes together are valid, each keeping its own class and position")
  {
    auto replica_spaces = fixture.replica_spaces;  // copy; each section builds its own plan
    dynamic_filter_publish_plan const plan{{make_int64_key(0, 0), make_int64_key(1, 1)},
                                           make_targets(/*direct_accepts_zone_maps=*/false, 1),
                                           std::move(replica_spaces)};

    REQUIRE(plan.enabled());
    REQUIRE(plan.probe_targets().size() == 2);
    REQUIRE(plan.probe_targets()[0].filter_set == scan_channel);
    REQUIRE(plan.probe_targets()[0].route_class == dynamic_filter_route_class::scan);
    REQUIRE(plan.probe_targets()[1].filter_set == direct_channel);
    REQUIRE(plan.probe_targets()[1].route_class == dynamic_filter_route_class::direct);
  }

  SECTION("a direct target that accepts zone maps is rejected")
  {
    auto replica_spaces = fixture.replica_spaces;
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan({make_int64_key(0, 0), make_int64_key(1, 1)},
                                  make_targets(/*direct_accepts_zone_maps=*/true, 1),
                                  std::move(replica_spaces)),
      std::invalid_argument);
  }

  SECTION("a direct binding naming a nonexistent admitted key is rejected")
  {
    auto replica_spaces = fixture.replica_spaces;
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan({make_int64_key(0, 0)},
                                  make_targets(/*direct_accepts_zone_maps=*/false, 1),
                                  std::move(replica_spaces)),
      std::invalid_argument);
  }
}

TEST_CASE("dynamic-filter publisher builds filters only for bound keys",
          "[dynamic_filter][publisher]")
{
  // Admission records legality beyond consumption, so a plan can carry an admitted key no target
  // binds. Such a key must cost no filter construction and must not be counted as walked.
  constexpr std::size_t kBuildRows        = 3;
  constexpr std::size_t kBoundPushOrdinal = 5;

  publisher_fixture fixture;
  // Reversed ordinals and disjoint values distinguish the bound key from the unbound key.
  fixture.add_key_column(kBuildRows, 100);
  fixture.add_key_column(kBuildRows, 0);

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 1,
                                                   .channel_push_ordinal = kBoundPushOrdinal,
                                                   .probe_storage_type   = kInt64}}});
  dynamic_filter_publish_plan plan{{make_int64_key(0, 1), make_int64_key(1, 0)},
                                   std::move(targets),
                                   std::move(fixture.replica_spaces)};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);

  REQUIRE(outcome.keys_considered == 1);
  REQUIRE(outcome.membership_filters_built == 1);
  REQUIRE(outcome.filters_pushed == 1);
  REQUIRE(outcome.keys_skipped_domain_gate == 0);
  REQUIRE(outcome.keys_skipped_type_mismatch == 0);

  // Verify both the bound push ordinal and the selected build column.
  auto const published = channel->filters_for_column(kBoundPushOrdinal);
  REQUIRE(published.size() == 1);
  auto const probe = make_int64_values(fixture, {0, 100});
  REQUIRE(membership_mask(*published.front(), probe->view(), fixture) ==
          std::vector<std::uint8_t>{0, 1});
}

TEST_CASE("dynamic-filter publish plan rejects unusable replica placements",
          "[dynamic_filter][publisher]")
{
  // A placement the publisher cannot allocate a replica in, or stage a transfer through, would
  // otherwise surface as an allocation failure mid-publication rather than at plan construction.
  publisher_fixture fixture;
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();

  auto make_targets = [&channel] {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = channel,
                       .route_class              = dynamic_filter_route_class::scan,
                       .accepts_zone_map_filters = true,
                       .key_bindings             = {{.admitted_key_index   = 0,
                                                     .channel_push_ordinal = kProbeColumnIndex,
                                                     .probe_storage_type   = kInt64}}});
    return targets;
  };

  SECTION("probe targets with no replica placement at all")
  {
    REQUIRE_THROWS_AS(dynamic_filter_publish_plan({make_int64_key(0, 0)}, make_targets(), {}),
                      std::invalid_argument);
  }

  SECTION("a placement whose GPU slot holds a host space")
  {
    auto& host_space = host_memory_space(fixture);
    std::vector<sirius::op::dynamic_filter_replica_space> spaces{{host_space, host_space}};
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan({make_int64_key(0, 0)}, make_targets(), std::move(spaces)),
      std::invalid_argument);
  }

  SECTION("a placement whose staging slot holds a GPU space")
  {
    auto& gpu_space = fixture.replica_spaces.front().get_gpu_space();
    std::vector<sirius::op::dynamic_filter_replica_space> spaces{{gpu_space, gpu_space}};
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan({make_int64_key(0, 0)}, make_targets(), std::move(spaces)),
      std::invalid_argument);
  }
}

TEST_CASE("multi-partition accumulator publishes only after every exact build ID",
          "[dynamic_filter][publisher][accumulator]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(plan, 6, {101, 202});

  auto const first =
    accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
  REQUIRE(channel->empty());

  auto const duplicate =
    accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(duplicate.state == sirius::op::dynamic_filter_accumulation_result::status::duplicate);
  REQUIRE(channel->empty());

  auto const last =
    accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);
  REQUIRE(last.state == sirius::op::dynamic_filter_accumulation_result::status::published);
  REQUIRE(accumulator.complete());
  REQUIRE_FALSE(accumulator.aborted());
  REQUIRE_FALSE(accumulator.abort_if_incomplete());

  auto const published = channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(published.size() == 1);
  REQUIRE(dynamic_cast<sirius::op::sirius_dynamic_bloom_filter const*>(published.front().get()) !=
          nullptr);
  auto const probe = make_int64_values(fixture, {0, 1, 2, 3, 4, 5});
  REQUIRE(membership_mask(*published.front(), probe->view(), fixture) ==
          std::vector<std::uint8_t>{1, 1, 1, 1, 1, 1});

  auto const unknown =
    accumulator.contribute(999, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(unknown.state == sirius::op::dynamic_filter_accumulation_result::status::aborted);
}

TEST_CASE("multi-partition accumulator close prevents publication with a missing ID",
          "[dynamic_filter][publisher][accumulator]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(plan, 6, {11, 22});

  auto const first =
    accumulator.contribute(11, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
  REQUIRE(accumulator.abort_if_incomplete());
  REQUIRE(accumulator.aborted());

  auto const late =
    accumulator.contribute(22, one_column_view(*fixture.columns[1]), fixture.stream);
  REQUIRE(late.state == sirius::op::dynamic_filter_accumulation_result::status::aborted);
  REQUIRE(channel->empty());
}

TEST_CASE("multi-partition accumulator fails closed on a contribution type mismatch",
          "[dynamic_filter][publisher][accumulator]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  auto wrong_type = cudf::sequence(3,
                                   cudf::numeric_scalar<std::int32_t>(3, true, fixture.stream),
                                   cudf::numeric_scalar<std::int32_t>(1, true, fixture.stream),
                                   fixture.stream,
                                   cudf::get_current_device_resource_ref());
  fixture.stream.synchronize();

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(plan, 6, {31, 32});
  REQUIRE(accumulator.contribute(31, one_column_view(*fixture.columns[0]), fixture.stream).state ==
          sirius::op::dynamic_filter_accumulation_result::status::pending);

  auto const mismatch = accumulator.contribute(32, one_column_view(*wrong_type), fixture.stream);
  REQUIRE(mismatch.state == sirius::op::dynamic_filter_accumulation_result::status::aborted);
  REQUIRE(mismatch.publication.keys_skipped_type_mismatch == 1);
  REQUIRE(accumulator.aborted());
  REQUIRE(channel->empty());
}

TEST_CASE("accumulated Bloom storage outlives a task-owned contribution stream",
          "[dynamic_filter][publisher][accumulator][lifetime]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  {
    auto plan = make_accumulator_plan(fixture, channel);
    sirius::op::dynamic_filter_accumulator accumulator(plan, 6, {101, 202});
    {
      rmm::cuda_stream task_stream{rmm::cuda_stream::flags::non_blocking};
      auto const first =
        accumulator.contribute(101, one_column_view(*fixture.columns[0]), task_stream.view());
      REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
      auto const last =
        accumulator.contribute(202, one_column_view(*fixture.columns[1]), task_stream.view());
      REQUIRE(last.state == sirius::op::dynamic_filter_accumulation_result::status::published);
    }
    REQUIRE(channel->filter_count() == 1);
  }

  // Destroy the CUCO owner after task_stream to exercise its captured memory-space stream.
  channel.reset();
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

TEST_CASE("an in-flight duplicate cannot insert or advance accumulator completion",
          "[dynamic_filter][publisher][accumulator][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  std::latch id_claimed{1};
  std::latch release_claim{1};
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.after_id_claim = [&](std::uint64_t batch_id) {
    if (batch_id != 101) { return; }
    id_claimed.count_down();
    release_claim.wait();
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(plan, 6, {101, 202}, std::move(hooks));
  rmm::cuda_stream task_stream{rmm::cuda_stream::flags::non_blocking};

  auto first_future = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    return accumulator.contribute(101, one_column_view(*fixture.columns[0]), task_stream.view());
  });
  id_claimed.wait();
  auto const duplicate =
    accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream);
  release_claim.count_down();
  auto const first = first_future.get();

  REQUIRE(duplicate.state == sirius::op::dynamic_filter_accumulation_result::status::duplicate);
  REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
  REQUIRE(channel->empty());

  auto const last =
    accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);
  REQUIRE(last.state == sirius::op::dynamic_filter_accumulation_result::status::published);
  REQUIRE(channel->filter_count() == 1);
}

TEST_CASE("different final contributions race to exactly one publication",
          "[dynamic_filter][publisher][accumulator][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  std::barrier insertions_complete{2};
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.after_insert_sync = [&](std::uint64_t) { insertions_complete.arrive_and_wait(); };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(plan, 6, {101, 202}, std::move(hooks));
  rmm::cuda_stream first_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream second_stream{rmm::cuda_stream::flags::non_blocking};

  auto contribute =
    [&](std::uint64_t batch_id, cudf::column const& column, rmm::cuda_stream_view stream) {
      rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
      return accumulator.contribute(batch_id, one_column_view(column), stream);
    };
  auto first_future = std::async(
    std::launch::async, contribute, 101, std::cref(*fixture.columns[0]), first_stream.view());
  auto second_future = std::async(
    std::launch::async, contribute, 202, std::cref(*fixture.columns[1]), second_stream.view());
  auto const first  = first_future.get();
  auto const second = second_future.get();

  auto const published =
    static_cast<int>(first.state ==
                     sirius::op::dynamic_filter_accumulation_result::status::published) +
    static_cast<int>(second.state ==
                     sirius::op::dynamic_filter_accumulation_result::status::published);
  auto const pending =
    static_cast<int>(first.state ==
                     sirius::op::dynamic_filter_accumulation_result::status::pending) +
    static_cast<int>(second.state ==
                     sirius::op::dynamic_filter_accumulation_result::status::pending);
  REQUIRE(published == 1);
  REQUIRE(pending == 1);
  REQUIRE(accumulator.complete());
  REQUIRE(channel->filter_count() == 1);
}

TEST_CASE("strict replica failure aborts before any accumulator fan-out",
          "[dynamic_filter][publisher][accumulator][replication_failure]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);

  std::size_t replication_calls = 0;
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.strict_replicate = [&](auto&, auto) {
    ++replication_calls;
    throw std::runtime_error("injected required-replica failure");
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(plan, 6, {101, 202}, std::move(hooks));

  REQUIRE(accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream).state ==
          sirius::op::dynamic_filter_accumulation_result::status::pending);
  auto const failed =
    accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);

  REQUIRE(failed.state == sirius::op::dynamic_filter_accumulation_result::status::aborted);
  REQUIRE(replication_calls == 1);
  REQUIRE(failed.publication.membership_filters_built == 0);
  REQUIRE(failed.publication.filters_pushed == 0);
  REQUIRE(accumulator.aborted());
  REQUIRE(channel->empty());
}

TEST_CASE("equal-geometry Bloom partials OR into the root without false negatives",
          "[dynamic_filter][publisher][bloom_reduction]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto const& root_space = fixture.replica_spaces.front();

  sirius::op::sirius_dynamic_bloom_filter root(
    kInt64, 6, fixture.stream, root_space.get_gpu_space().get_default_allocator());
  sirius::op::sirius_dynamic_bloom_filter partial(
    kInt64, 6, fixture.stream, root_space.get_gpu_space().get_default_allocator());
  root.add(fixture.columns[0]->view(), fixture.stream);
  partial.add(fixture.columns[1]->view(), fixture.stream);
  fixture.stream.synchronize();

  root.merge_from(partial, root_space, root_space, fixture.stream);
  fixture.stream.synchronize();
  root.release_reduction_scratch();

  auto const probe = make_int64_values(fixture, {0, 1, 2, 3, 4, 5});
  REQUIRE(membership_mask(root, probe->view(), fixture) ==
          std::vector<std::uint8_t>{1, 1, 1, 1, 1, 1});
}
