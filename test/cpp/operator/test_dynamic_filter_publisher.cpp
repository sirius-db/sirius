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

/**
 * @file test_dynamic_filter_publisher.cpp
 * @brief Pins publish_dynamic_filters' per-key filter construction, fan-out, and plan validation
 *
 * Sections, in file order:
 *  - Tier precedence: the raw small IN-list wins at <= k_max_keys build rows, and one row above
 *    that gate falls through to the hash IN-list under the default L2 fraction.
 *  - inlist_max_l2_fraction plumbing: a vanishing fraction demotes the hash IN-list to the Bloom
 *    filter, 1.0 keeps every L2-fitting set, and the inclusive boundary is bracketed from the
 *    same estimated_set_bytes / cudaDevAttrL2CacheSize inputs the publisher reads.
 *  - Type-coverage canary: Bloom support covers every hash-IN-list key type, so the policy's
 *    "keep a fitting IN-list when the type lacks Bloom support" clause is unreachable through the
 *    real publisher today.
 *  - Sparse fan-out and gating: each target receives only its bound keys, the domain-coverage
 *    gate consults each key's own domain, and unbound keys cost no construction.
 *  - Floating-point suppression: a FLOAT64 key whose build holds a NaN receives no zone map (the
 *    lowered IEEE bounds would drop NaN probe rows the authoritative join matches) while its
 *    INT64 sibling keeps both of its filters.
 *  - Runtime guards and degenerate publications: plan/runtime key-mapping inconsistencies,
 *    per-binding zone-map suppression, membership-only targets, keyless plans, empty builds, and
 *    drained targets.
 *  - Plan validation: invalid targets, bindings, and replica placements are rejected at
 *    construction.
 *  - Replica restriction: restrict_replicas_to disables a plan whose admitted GPU set is
 *    disjoint from its replica devices, leaves member/empty restrictions intact, and
 *    has_replica_on_device reports per-device replica presence.
 *
 * The fraction semantics themselves (0 always demotes to the Bloom, small-list precedence, the
 * exact inclusive boundary, and the no-Bloom exception) are pinned GPU-free in
 * `test_dynamic_filter_source_policy.cpp`.
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

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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

constexpr auto kInt64   = cudf::data_type{cudf::type_id::INT64};
constexpr auto kFloat64 = cudf::data_type{cudf::type_id::FLOAT64};

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

std::unique_ptr<cudf::column> make_float64_values(publisher_fixture const& fixture,
                                                  std::vector<double> const& values)
{
  auto column    = cudf::make_numeric_column(kFloat64,
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          fixture.stream,
                                          cudf::get_current_device_resource_ref());
  auto const err = cudaMemcpyAsync(column->mutable_view().data<double>(),
                                   values.data(),
                                   values.size() * sizeof(double),
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
void require_published_membership(
  std::size_t rows,
  double inlist_max_l2_fraction =
    sirius::op::dynamic_filter_publication_policy::k_default_inlist_max_l2_fraction)
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
  dynamic_filter_publish_plan plan{{make_int64_key(0, 0)},
                                   std::move(targets),
                                   std::move(fixture.replica_spaces),
                                   {.inlist_max_l2_fraction = inlist_max_l2_fraction}};

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
  if constexpr (requires(ExpectedFilter const& f) { f.size(); }) {
    REQUIRE(selected->size() == rows);
  }
  REQUIRE(selected->replica_count() == 1);
  if constexpr (std::is_same_v<ExpectedFilter, sirius::op::sirius_dynamic_in_list_filter>) {
    REQUIRE(selected->has_persistent_set());
  }
}

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

// One scan target bound to one admitted key, replicated on the fixture's single GPU-0 space.
dynamic_filter_publish_plan make_single_replica_plan(publisher_fixture const& fixture)
{
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  return dynamic_filter_publish_plan{
    {make_int64_key(0, 0)}, std::move(targets), fixture.replica_spaces};
}

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

TEST_CASE("dynamic-filter publisher demotes the hash IN-list to Bloom above the L2 fraction",
          "[dynamic_filter][publisher]")
{
  // A vanishing fraction makes the residency threshold (fraction x L2) smaller than any real
  // hash set, so the smallest hash-tier build must demote to the Bloom; this pins the
  // plan-to-publisher plumbing of inlist_max_l2_fraction end to end.
  require_published_membership<sirius::op::sirius_dynamic_bloom_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1, 1e-12);
}

TEST_CASE("dynamic-filter publisher fraction 1.0 reproduces the legacy L2-fit rule",
          "[dynamic_filter][publisher]")
{
  // fraction = 1.0 -> threshold = the full L2, so every L2-fitting set keeps the exact IN-list.
  require_published_membership<sirius::op::sirius_dynamic_in_list_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1, 1.0);
}

TEST_CASE("dynamic-filter publisher L2-fraction boundary is inclusive",
          "[dynamic_filter][publisher]")
{
  // The exact set_bytes <= fraction x l2_bytes boundary arithmetic is pinned GPU-free in
  // test_dynamic_filter_source_policy.cpp; this bracket pins that the publisher feeds the choice
  // the same estimated_set_bytes and min-cudaDevAttrL2CacheSize values this test computes. The
  // +/-1e-9 margin absorbs double rounding while staying under one byte of threshold on any
  // real L2.
  auto const rows      = sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1;
  auto const set_bytes = sirius::op::sirius_dynamic_in_list_filter::estimated_set_bytes(
    rows, cudf::data_type{cudf::type_id::INT64});
  int l2_bytes = 0;
  REQUIRE(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, kDeviceId) == cudaSuccess);
  REQUIRE(l2_bytes > 0);
  double const ratio = static_cast<double>(set_bytes) / static_cast<double>(l2_bytes);
  require_published_membership<sirius::op::sirius_dynamic_in_list_filter>(rows, ratio * (1 + 1e-9));
  require_published_membership<sirius::op::sirius_dynamic_bloom_filter>(rows, ratio * (1 - 1e-9));
}

TEST_CASE("dynamic-filter Bloom support covers every hash-IN-list key type",
          "[dynamic_filter][publisher]")
{
  // choose_membership_filter keeps an L2-fitting hash IN-list at any fraction when the key type
  // has no Bloom fallback; that clause is pinned GPU-free in test_dynamic_filter_source_policy.cpp.
  // This canary documents that the clause remains unreachable through the real publisher because
  // the hash-IN-list and Bloom supported-type sets are identical (INT32/INT64). If either REQUIRE
  // ever fails (the type sets diverge), add a publish-path test asserting the divergent type keeps
  // the fitting IN-list at fraction 0.
  REQUIRE(sirius::op::sirius_dynamic_bloom_filter::supports(cudf::data_type{cudf::type_id::INT32}));
  REQUIRE(sirius::op::sirius_dynamic_bloom_filter::supports(cudf::data_type{cudf::type_id::INT64}));
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

  // Verify key identity, not just placement: channel A must hold build column 1's domain
  // ({0,1,2}) and channel B build column 0's ({100,101,102}).
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

TEST_CASE("dynamic-filter publisher suppresses zone maps for floating-point keys",
          "[dynamic_filter][publisher]")
{
  // The lowered zone-map AST compares with IEEE semantics under which NaN fails both bounds,
  // while the authoritative join matches NaN keys to each other; a FLOAT64 key must therefore
  // build no zone map. The NaN in the build column reproduces the observed row-dropping bug;
  // suppression is type-based, so no reduction runs on the column at all.
  constexpr std::size_t kInt64PushOrdinal   = 3;
  constexpr std::size_t kFloat64PushOrdinal = 5;

  publisher_fixture fixture;
  fixture.add_key_column(3);
  fixture.columns.push_back(
    make_float64_values(fixture, {1.0, std::numeric_limits<double>::quiet_NaN(), 2.0}));

  auto float64_key         = make_int64_key(1, 1);
  float64_key.storage_type = kFloat64;

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kInt64PushOrdinal,
                                                   .probe_storage_type   = kInt64},
                                                  {.admitted_key_index   = 1,
                                                   .channel_push_ordinal = kFloat64PushOrdinal,
                                                   .probe_storage_type   = kFloat64}}});
  dynamic_filter_publish_plan plan{{make_int64_key(0, 0), float64_key},
                                   std::move(targets),
                                   std::move(fixture.replica_spaces),
                                   {.emit_zone_map_filters = true}};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);

  // Suppression is a capability outcome, not the type-mismatch skip path.
  REQUIRE(outcome.keys_considered == 2);
  REQUIRE(outcome.keys_skipped_type_mismatch == 0);
  REQUIRE(outcome.zone_map_filters_built == 1);
  REQUIRE(outcome.membership_filters_built == 1);
  REQUIRE(outcome.filters_pushed == 2);

  // With no filter published, every probe row (NaN included) reaches the authoritative join.
  REQUIRE(channel->filters_for_column(kFloat64PushOrdinal).empty());

  auto const int64_snapshot = channel->filters_for_column(kInt64PushOrdinal);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(int64_snapshot) == 1);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_small_in_list_filter>(int64_snapshot) ==
          1);
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
    // The join stays authoritative, so a type-derivation disagreement is a counted skip, not
    // a failure.
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
  // Zone maps are enabled so the empty build skips the range-reduction path too.
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

  SECTION("direct binding with a non-INT32/INT64 probe storage type")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::direct,
       .accepts_zone_map_filters = false,
       .key_bindings             = {{.admitted_key_index   = 0,
                                     .channel_push_ordinal = 0,
                                     .probe_storage_type   = cudf::data_type{cudf::type_id::INT8}}}});
    // The admitted key is INT8 as well, so probe/build equality holds and only the INT32/INT64
    // whitelist arm rejects.
    auto key         = make_int64_key(0, 0);
    key.storage_type = cudf::data_type{cudf::type_id::INT8};
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan({key}, std::move(targets), std::move(fixture.replica_spaces)),
      std::invalid_argument);
  }

  SECTION("direct binding whose probe storage type differs from the admitted key's build type")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::direct,
       .accepts_zone_map_filters = false,
       .key_bindings             = {{.admitted_key_index   = 0,
                                     .channel_push_ordinal = 0,
                                     .probe_storage_type   = cudf::data_type{cudf::type_id::INT32}}}});
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
  // A plan may carry an admitted key no target binds; such a key must cost no filter
  // construction and must not be counted.
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
  // Unusable placements must fail at plan construction, not mid-publication as allocation errors.
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

TEST_CASE("restricting replicas to a disjoint GPU set disables the plan",
          "[dynamic_filter][publish_plan]")
{
  publisher_fixture fixture;
  auto plan = make_single_replica_plan(fixture);
  REQUIRE(plan.enabled());

  plan.restrict_replicas_to({kDeviceId + 1});

  CHECK_FALSE(plan.enabled());
  CHECK(plan.probe_targets().empty());
  CHECK(plan.replica_spaces().empty());

  // Restricting an already-disabled plan is a no-op, not an error.
  CHECK_NOTHROW(plan.restrict_replicas_to({kDeviceId + 1}));
  CHECK_FALSE(plan.enabled());
}

TEST_CASE("replica restriction to a member GPU or the empty set leaves the plan enabled",
          "[dynamic_filter][publish_plan]")
{
  publisher_fixture fixture;
  auto plan = make_single_replica_plan(fixture);

  // An empty list means "no subset".
  plan.restrict_replicas_to({});
  CHECK(plan.enabled());
  CHECK(plan.replica_spaces().size() == 1);
  CHECK(plan.probe_targets().size() == 1);

  // A restriction that keeps the plan's replica GPU erases nothing.
  plan.restrict_replicas_to({kDeviceId});
  CHECK(plan.enabled());
  CHECK(plan.replica_spaces().size() == 1);
  CHECK(plan.probe_targets().size() == 1);
}

TEST_CASE("the plan reports replica presence per device", "[dynamic_filter][publish_plan]")
{
  publisher_fixture fixture;
  auto plan = make_single_replica_plan(fixture);

  CHECK(plan.has_replica_on_device(kDeviceId));
  CHECK_FALSE(plan.has_replica_on_device(kDeviceId + 1));

  plan.restrict_replicas_to({kDeviceId + 1});
  CHECK_FALSE(plan.has_replica_on_device(kDeviceId));
}
