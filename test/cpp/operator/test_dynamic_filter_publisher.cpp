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
 *  - Narrowed build carriers: a build column arriving at a carrier restorable to the recorded
 *    storage type publishes (filters built at the carrier); an unrelated type is still skipped.
 *  - DATE keys: a TIMESTAMP_DAYS key publishes a membership filter and a zone map, is probed by
 *    the native column and by an INT16 carrier alike, and a build column arriving at an INT16
 *    carrier is restored to TIMESTAMP_DAYS so it stays in the DATE family.
 *  - Decimal keys: a DECIMAL64 key arriving as a same-scale DECIMAL32 carrier publishes a 32-bit
 *    set; a DECIMAL128 key publishes membership only when its unscaled build values fit int64
 *    (otherwise only the zone map), and the join-edge validator requires an identical scale.
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

#include "op/dynamic_filter/dynamic_filter_key_domain.hpp"
#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

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
constexpr auto kDays    = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
auto const kDecimal32   = cudf::data_type{cudf::type_id::DECIMAL32, -2};
auto const kDecimal64   = cudf::data_type{cudf::type_id::DECIMAL64, -2};
auto const kDecimal128  = cudf::data_type{cudf::type_id::DECIMAL128, -2};
constexpr auto kString  = cudf::data_type{cudf::type_id::STRING};

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

  // Append the same sequence stored at a narrower carrier, as a pinned-narrow build would.
  void add_key_column_as(std::size_t rows, cudf::data_type carrier, std::int64_t first = 0)
  {
    add_key_column(rows, first);
    auto& source_space = replica_spaces.front().get_gpu_space();
    columns.back() =
      cudf::cast(columns.back()->view(), carrier, stream, source_space.get_default_allocator());
  }

  // Null out rows [begin, end) of the last key column, as a nullable fact-table FK build would.
  void null_last_key_rows(cudf::size_type begin, cudf::size_type end)
  {
    auto& column       = *columns.back();
    auto& source_space = replica_spaces.front().get_gpu_space();
    auto mask          = cudf::create_null_mask(
      column.size(), cudf::mask_state::ALL_VALID, stream, source_space.get_default_allocator());
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), begin, end, false, stream);
    column.set_null_mask(std::move(mask), end - begin);
    stream.synchronize();
  }

  // Append a DATE (TIMESTAMP_DAYS) column of consecutive epoch days starting at `first_day`.
  void add_date_key_column(std::size_t rows, std::int32_t first_day)
  {
    std::vector<std::int32_t> days(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      days[i] = first_day + static_cast<std::int32_t>(i);
    }
    auto& source_space = replica_spaces.front().get_gpu_space();
    auto column        = cudf::make_timestamp_column(kDays,
                                              static_cast<cudf::size_type>(rows),
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              source_space.get_default_allocator());
    REQUIRE(cudaMemcpyAsync(column->mutable_view().head<std::int32_t>(),
                            days.data(),
                            days.size() * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
    columns.push_back(std::move(column));
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

// Upload a null-free STRING column.
std::unique_ptr<cudf::column> make_string_values(publisher_fixture const& fixture,
                                                 std::vector<std::string> const& values)
{
  auto const mr = cudf::get_current_device_resource_ref();
  auto const n  = static_cast<cudf::size_type>(values.size());
  std::vector<cudf::size_type> offsets(static_cast<std::size_t>(n) + 1, 0);
  std::string chars;
  for (std::size_t i = 0; i < values.size(); ++i) {
    chars += values[i];
    offsets[i + 1] = static_cast<cudf::size_type>(chars.size());
  }
  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               static_cast<cudf::size_type>(offsets.size()),
                                               cudf::mask_state::UNALLOCATED,
                                               fixture.stream,
                                               mr);
  REQUIRE(cudaMemcpyAsync(offsets_col->mutable_view().data<cudf::size_type>(),
                          offsets.data(),
                          offsets.size() * sizeof(cudf::size_type),
                          cudaMemcpyHostToDevice,
                          fixture.stream.value()) == cudaSuccess);
  rmm::device_buffer chars_buf{chars.data(), chars.size(), fixture.stream, mr};
  fixture.stream.synchronize();
  return cudf::make_strings_column(
    n, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{});
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

std::unique_ptr<cudf::column> make_day_values(publisher_fixture const& fixture,
                                              std::vector<std::int32_t> const& days)
{
  auto column    = cudf::make_timestamp_column(kDays,
                                            static_cast<cudf::size_type>(days.size()),
                                            cudf::mask_state::UNALLOCATED,
                                            fixture.stream,
                                            cudf::get_current_device_resource_ref());
  auto const err = cudaMemcpyAsync(column->mutable_view().head<std::int32_t>(),
                                   days.data(),
                                   days.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice,
                                   fixture.stream.value());
  REQUIRE(err == cudaSuccess);
  fixture.stream.synchronize();
  return column;
}

std::unique_ptr<cudf::column> make_int16_values(publisher_fixture const& fixture,
                                                std::vector<std::int16_t> const& values)
{
  auto column    = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT16},
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          fixture.stream,
                                          cudf::get_current_device_resource_ref());
  auto const err = cudaMemcpyAsync(column->mutable_view().data<std::int16_t>(),
                                   values.data(),
                                   values.size() * sizeof(std::int16_t),
                                   cudaMemcpyHostToDevice,
                                   fixture.stream.value());
  REQUIRE(err == cudaSuccess);
  fixture.stream.synchronize();
  return column;
}

// Fixed-point column at @p type whose unscaled storage holds @p values narrowed to Rep.
template <typename Rep>
std::unique_ptr<cudf::column> make_decimal_values(publisher_fixture const& fixture,
                                                  cudf::data_type type,
                                                  std::vector<__int128_t> const& values)
{
  std::vector<Rep> typed;
  typed.reserve(values.size());
  for (auto const v : values) {
    typed.push_back(static_cast<Rep>(v));
  }
  auto column    = cudf::make_fixed_point_column(type,
                                              static_cast<cudf::size_type>(typed.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              fixture.stream,
                                              cudf::get_current_device_resource_ref());
  auto const err = cudaMemcpyAsync(column->mutable_view().data<Rep>(),
                                   typed.data(),
                                   typed.size() * sizeof(Rep),
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

// A pinned chunk may store a join key NARROWED -- pin-time compressed materialization casts each
// column to the narrowest carrier its values fit -- while the filter was published at the key's
// native carrier. Probing must RESTORE that carrier rather than decline: a decline costs the
// chunk its whole decode-side compaction (measured: every membership probe declined, filtered
// decode fell back to a plain one), and the restored probe must answer exactly as the native
// column would.
TEST_CASE("membership probes restore a narrowed probe carrier", "[dynamic_filter]")
{
  publisher_fixture fixture;
  auto const mr = cudf::get_current_device_resource_ref();

  auto const keys = make_int64_values(fixture, {10, 20, 30});
  sirius::op::sirius_dynamic_in_list_filter filter(keys->view(), fixture.stream, mr);

  auto const native = make_int64_values(fixture, {10, 15, 20, 999, 30});
  std::vector<std::uint8_t> const expected{1, 0, 1, 0, 1};
  REQUIRE(membership_mask(filter, native->view(), fixture) == expected);

  // The same logical values as a narrowed pin serves them.
  auto const narrowed =
    cudf::cast(native->view(), cudf::data_type{cudf::type_id::INT32}, fixture.stream, mr);
  REQUIRE(narrowed->view().type().id() == cudf::type_id::INT32);
  REQUIRE(membership_mask(filter, narrowed->view(), fixture) == expected);

  // Restoration is exact, never a reinterpretation: an unrelated carrier still declines.
  auto const wrong       = make_float64_values(fixture, {10.0, 15.0, 20.0, 999.0, 30.0});
  auto const* applicable = dynamic_cast<sirius::op::sirius_mask_applicable const*>(&filter);
  REQUIRE(applicable != nullptr);
  REQUIRE(applicable->compute_mask(wrong->view(), kDeviceId, fixture.stream, mr) == nullptr);
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
  // the hash-IN-list and Bloom supported-type sets are identical: both are exactly
  // membership_key_supported. If this ever fails (the type sets diverge), add a publish-path test
  // asserting the divergent type keeps the fitting IN-list at fraction 0.
  using id = cudf::type_id;
  for (auto const t : {id::INT8,
                       id::INT16,
                       id::INT32,
                       id::INT64,
                       id::UINT8,
                       id::UINT16,
                       id::UINT32,
                       id::UINT64,
                       id::BOOL8,
                       id::FLOAT32,
                       id::FLOAT64,
                       id::TIMESTAMP_DAYS,
                       id::TIMESTAMP_MICROSECONDS,
                       id::DECIMAL32,
                       id::DECIMAL64,
                       id::DECIMAL128,
                       id::STRING,
                       id::LIST,
                       id::STRUCT,
                       id::EMPTY}) {
    auto const type = cudf::data_type{t};
    REQUIRE(sirius::op::sirius_dynamic_bloom_filter::supports(type) ==
            sirius::op::membership_key_supported(type));
    // The hash IN-list's gate takes a column; an empty column isolates the type gate. Nulls no
    // longer enter the gate (null keys are compacted out), so a nullable column agrees too.
    auto const empty = cudf::column_view{type, 0, nullptr, nullptr, 0};
    REQUIRE(sirius::op::sirius_dynamic_in_list_filter::supports(empty) ==
            sirius::op::membership_key_supported(type));
    // STRING is supported but not fixed-width; its nullable-build acceptance is covered by the
    // string probe tests instead.
    if (sirius::op::membership_key_supported(type) && cudf::is_fixed_width(type)) {
      auto const stream   = cudf::get_default_stream();
      auto const nullable = cudf::make_fixed_width_column(
        type, 1, cudf::mask_state::ALL_NULL, stream, cudf::get_current_device_resource_ref());
      REQUIRE(sirius::op::sirius_dynamic_in_list_filter::supports(nullable->view()));
    }
  }
}

// A fact-table foreign key on the build side carries nulls. Null keys match nothing under the
// join's null_equality::UNEQUAL, so the publisher must publish an *exact* IN-list over the valid
// keys (sized and tiered on the valid row count) rather than fall through to Bloom, and the
// values sitting under the null build slots must not survive a probe.
TEST_CASE("dynamic-filter publisher builds exact IN-lists from a nullable build column",
          "[dynamic_filter][publisher][nulls]")
{
  publisher_fixture fixture;

  auto const publish = [&]() {
    auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = channel,
                       .route_class              = dynamic_filter_route_class::scan,
                       .accepts_zone_map_filters = false,
                       .key_bindings             = {{.admitted_key_index   = 0,
                                                     .channel_push_ordinal = kProbeColumnIndex,
                                                     .probe_storage_type   = kInt64}}});
    auto replica_spaces = fixture.replica_spaces;
    dynamic_filter_publish_plan plan{
      {make_int64_key(0, 0)}, std::move(targets), std::move(replica_spaces)};
    auto const outcome =
      sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
    REQUIRE(outcome.keys_considered == 1);
    REQUIRE(outcome.keys_skipped_type_mismatch == 0);
    REQUIRE(outcome.membership_filters_built == 1);
    REQUIRE(outcome.filters_pushed == 1);
    auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
    REQUIRE(snapshot.size() == 1);
    return snapshot.front();
  };

  SECTION("small IN-list tier is gated on the valid rows, not the column length")
  {
    // 15 slots, 5 of them null: 10 valid keys {0..9}, under k_max_keys although 15 is over it.
    auto const rows = sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 3;
    fixture.add_key_column(rows);
    fixture.null_last_key_rows(10, static_cast<cudf::size_type>(rows));
    REQUIRE(fixture.columns.front()->null_count() == 5);

    auto const filter = publish();
    auto const* small =
      dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(filter.get());
    REQUIRE(small != nullptr);
    CHECK(small->size() == 10);

    // 10..14 sit under the null build slots and must not survive; a null probe row is dropped.
    auto probe = make_int64_values(fixture, {0, 9, 10, 14, 5, 100});
    REQUIRE(membership_mask(*filter, probe->view(), fixture) ==
            std::vector<std::uint8_t>{1, 1, 0, 0, 1, 0});
    auto mask = cudf::create_null_mask(probe->size(),
                                       cudf::mask_state::ALL_VALID,
                                       fixture.stream,
                                       cudf::get_current_device_resource_ref());
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 0, 1, false, fixture.stream);
    probe->set_null_mask(std::move(mask), 1);
    REQUIRE(membership_mask(*filter, probe->view(), fixture) ==
            std::vector<std::uint8_t>{0, 1, 0, 0, 1, 0});
  }

  SECTION("hash IN-list tier stores exactly the valid keys")
  {
    constexpr std::size_t rows = 40;
    fixture.add_key_column(rows);
    fixture.null_last_key_rows(0, 8);  // keys 0..7 are null; 32 valid keys remain
    auto const filter = publish();
    auto const* hashed =
      dynamic_cast<sirius::op::sirius_dynamic_in_list_filter const*>(filter.get());
    REQUIRE(hashed != nullptr);
    REQUIRE(hashed->has_persistent_set());
    CHECK(hashed->size() == 32);
    auto const probe = make_int64_values(fixture, {0, 7, 8, 39, 40});
    REQUIRE(membership_mask(*filter, probe->view(), fixture) ==
            std::vector<std::uint8_t>{0, 0, 1, 1, 0});
  }
}

// Compressed materialization may hand the publisher a build key column at a narrower carrier
// than the plan recorded (nation/region keys fit INT8, supplier keys INT16 at small scale). That
// carrier restores losslessly to the recorded type, so the key must publish rather than count as a
// type mismatch; the filters are built at the carrier and probes range-check into it.
TEST_CASE(
  "dynamic-filter publisher accepts a build column at a narrowed carrier of the recorded "
  "type",
  "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  auto const mr = cudf::get_current_device_resource_ref();

  auto const publish = [&]() {
    auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = channel,
                       .route_class              = dynamic_filter_route_class::scan,
                       .accepts_zone_map_filters = true,
                       .key_bindings             = {{.admitted_key_index   = 0,
                                                     .channel_push_ordinal = kProbeColumnIndex,
                                                     .probe_storage_type   = kInt64}}});
    auto replica_spaces = fixture.replica_spaces;
    dynamic_filter_publish_plan plan{
      {make_int64_key(0, 0)}, std::move(targets), std::move(replica_spaces)};
    auto const outcome =
      sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
    REQUIRE(outcome.keys_considered == 1);
    REQUIRE(outcome.keys_skipped_type_mismatch == 0);
    REQUIRE(outcome.membership_filters_built == 1);
    REQUIRE(outcome.filters_pushed == 1);
    auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
    REQUIRE(snapshot.size() == 1);
    return snapshot.front();
  };

  SECTION("INT16 carrier, small IN-list tier")
  {
    fixture.add_key_column_as(3, cudf::data_type{cudf::type_id::INT16}, /*first=*/10);
    REQUIRE(fixture.columns.front()->type().id() == cudf::type_id::INT16);
    auto const filter = publish();
    auto const* small =
      dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(filter.get());
    REQUIRE(small != nullptr);
    CHECK(small->domain().rep == sirius::op::membership_key_rep::i32);
    CHECK(small->domain().native.id() == cudf::type_id::INT16);

    // The native INT64 probe (post-decode cascade) and the INT16 carrier (fused decode) agree.
    auto const native = make_int64_values(fixture, {10, 11, 12, 13, 5'000'000'000LL});
    std::vector<std::uint8_t> const expected{1, 1, 1, 0, 0};
    REQUIRE(membership_mask(*filter, native->view(), fixture) == expected);
    auto const narrowed = cudf::cast(make_int64_values(fixture, {10, 11, 12, 13})->view(),
                                     cudf::data_type{cudf::type_id::INT16},
                                     fixture.stream,
                                     mr);
    REQUIRE(membership_mask(*filter, narrowed->view(), fixture) ==
            std::vector<std::uint8_t>{1, 1, 1, 0});
  }

  SECTION("INT8 carrier, hash IN-list tier")
  {
    auto const rows = sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1;
    fixture.add_key_column_as(rows, cudf::data_type{cudf::type_id::INT8});
    auto const filter = publish();
    auto const* hashed =
      dynamic_cast<sirius::op::sirius_dynamic_in_list_filter const*>(filter.get());
    REQUIRE(hashed != nullptr);
    REQUIRE(hashed->has_persistent_set());
    CHECK(hashed->domain().rep == sirius::op::membership_key_rep::i32);
    CHECK(hashed->domain().native.id() == cudf::type_id::INT8);
    auto const native = make_int64_values(
      fixture, {0, static_cast<std::int64_t>(rows) - 1, static_cast<std::int64_t>(rows), -1});
    REQUIRE(membership_mask(*filter, native->view(), fixture) ==
            std::vector<std::uint8_t>{1, 1, 0, 0});
  }
}

// TPC-DS d_date semi-joins (q58/q83, q38/q87) publish a DATE key. The build column is
// TIMESTAMP_DAYS, so both a membership filter and a zone map are built and pushed to a binding
// probing at the native type; the membership filter answers the native probe and the INT16
// carrier a pinned chunk would serve identically, and declines INT64 (not a DATE carrier).
TEST_CASE("dynamic-filter publisher publishes membership and zone-map filters for a DATE key",
          "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  fixture.add_date_key_column(3, /*first_day=*/10957);  // 2000-01-01 .. 2000-01-03

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kDays}}});
  auto key         = make_int64_key(0, 0);
  key.storage_type = kDays;
  dynamic_filter_publish_plan plan{
    {key}, std::move(targets), std::move(fixture.replica_spaces), {.emit_zone_map_filters = true}};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
  REQUIRE(outcome.keys_considered == 1);
  REQUIRE(outcome.keys_skipped_type_mismatch == 0);
  REQUIRE(outcome.membership_filters_built == 1);
  REQUIRE(outcome.zone_map_filters_built == 1);
  REQUIRE(outcome.filters_pushed == 2);

  auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(snapshot) == 1);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_small_in_list_filter>(snapshot) == 1);
  auto const membership = std::find_if(snapshot.begin(), snapshot.end(), [](auto const& filter) {
    return dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(filter.get()) !=
           nullptr;
  });
  REQUIRE(membership != snapshot.end());
  auto const* small =
    dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(membership->get());
  CHECK(small->domain().family == sirius::op::membership_key_family::date_days);
  CHECK(small->domain().rep == sirius::op::membership_key_rep::i32);
  CHECK(small->domain().native == kDays);

  // Native probe (post-decode cascade) and INT16 carrier (fused decode) agree on the same days.
  std::vector<std::uint8_t> const expected{1, 0, 1, 1, 0};
  auto const native = make_day_values(fixture, {10957, 10956, 10958, 10959, 20000});
  REQUIRE(membership_mask(**membership, native->view(), fixture) == expected);
  auto const narrowed = make_int16_values(fixture, {10957, 10956, 10958, 10959, 20000});
  REQUIRE(membership_mask(**membership, narrowed->view(), fixture) == expected);

  // INT64 is not a DATE carrier: decline, as the host mirror says.
  auto const wide        = make_int64_values(fixture, {10957, 10956});
  auto const* applicable = dynamic_cast<sirius::op::sirius_mask_applicable const*>(small);
  REQUIRE(applicable != nullptr);
  REQUIRE(applicable->compute_mask(
            wide->view(), kDeviceId, fixture.stream, cudf::get_current_device_resource_ref()) ==
          nullptr);
}

// A DATE build column that compressed materialization stored as INT16 arrives at that carrier
// while the plan recorded TIMESTAMP_DAYS. The publisher must not build a signed-integer set (it
// would decline the native TIMESTAMP_DAYS probe): it restores the column, so the key publishes in
// the DATE family with its zone map, and the native probe and the carrier both work.
TEST_CASE("dynamic-filter publisher restores a DATE build column arriving at an INT16 carrier",
          "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  fixture.add_key_column_as(3, cudf::data_type{cudf::type_id::INT16}, /*first=*/10957);
  REQUIRE(fixture.columns.front()->type().id() == cudf::type_id::INT16);

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kDays}}});
  auto key         = make_int64_key(0, 0);
  key.storage_type = kDays;
  dynamic_filter_publish_plan plan{
    {key}, std::move(targets), std::move(fixture.replica_spaces), {.emit_zone_map_filters = true}};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
  REQUIRE(outcome.keys_considered == 1);
  REQUIRE(outcome.keys_skipped_type_mismatch == 0);
  REQUIRE(outcome.membership_filters_built == 1);
  REQUIRE(outcome.zone_map_filters_built == 1);
  // The restored build type equals the native probe type, so the zone map is pushed as well.
  REQUIRE(outcome.filters_pushed == 2);

  auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(snapshot) == 1);
  auto const membership = std::find_if(snapshot.begin(), snapshot.end(), [](auto const& filter) {
    return dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(filter.get()) !=
           nullptr;
  });
  REQUIRE(membership != snapshot.end());
  auto const* small =
    dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(membership->get());
  CHECK(small->domain().family == sirius::op::membership_key_family::date_days);
  CHECK(small->domain().native == kDays);

  std::vector<std::uint8_t> const expected{1, 0, 1, 1, 0};
  auto const native = make_day_values(fixture, {10957, 10956, 10958, 10959, 20000});
  REQUIRE(membership_mask(**membership, native->view(), fixture) == expected);
  auto const narrowed = make_int16_values(fixture, {10957, 10956, 10958, 10959, 20000});
  REQUIRE(membership_mask(**membership, narrowed->view(), fixture) == expected);
}

// A DECIMAL(15,2) key (TPC-H q2's ps_supplycost) is DECIMAL64 in the plan, but a pinned build
// column may arrive as DECIMAL32 at the same scale. That carrier restores losslessly, so the key
// publishes a 32-bit set at the carrier and both the native DECIMAL64 probe and the DECIMAL32
// carrier probe answer identically.
TEST_CASE("dynamic-filter publisher accepts a DECIMAL32-carrier build column for a DECIMAL64 key",
          "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  auto const mr = cudf::get_current_device_resource_ref();

  auto const native_keys =
    make_decimal_values<std::int64_t>(fixture, kDecimal64, {1000, 2000, 3000});
  fixture.columns.push_back(cudf::cast(native_keys->view(), kDecimal32, fixture.stream, mr));
  REQUIRE(fixture.columns.front()->type() == kDecimal32);

  auto key         = make_int64_key(0, 0);
  key.storage_type = kDecimal64;
  auto channel     = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = true,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kDecimal64}}});
  dynamic_filter_publish_plan plan{{key}, std::move(targets), std::move(fixture.replica_spaces)};
  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
  REQUIRE(outcome.keys_considered == 1);
  REQUIRE(outcome.keys_skipped_type_mismatch == 0);
  REQUIRE(outcome.membership_filters_built == 1);
  REQUIRE(outcome.filters_pushed == 1);

  auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(snapshot.size() == 1);
  auto const* small =
    dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(snapshot.front().get());
  REQUIRE(small != nullptr);
  CHECK(small->domain().rep == sirius::op::membership_key_rep::i32);
  CHECK(small->domain().family == sirius::op::membership_key_family::decimal);
  CHECK(small->domain().native == kDecimal32);
  CHECK(small->domain().scale == -2);

  auto const native = make_decimal_values<std::int64_t>(
    fixture, kDecimal64, {1000, 1500, 2000, 3000, 6'000'000'000LL});
  REQUIRE(membership_mask(*small, native->view(), fixture) ==
          std::vector<std::uint8_t>{1, 0, 1, 1, 0});
  auto const carrier =
    make_decimal_values<std::int32_t>(fixture, kDecimal32, {1000, 1500, 2000, 3000});
  REQUIRE(membership_mask(*small, carrier->view(), fixture) ==
          std::vector<std::uint8_t>{1, 0, 1, 1});
  // Another scale never reaches a filter (the planner casts), and is declined if it did.
  auto const rescaled = make_decimal_values<std::int64_t>(
    fixture, cudf::data_type{cudf::type_id::DECIMAL64, -3}, {10000, 20000});
  REQUIRE(small->compute_mask(rescaled->view(), kDeviceId, fixture.stream, mr) == nullptr);
}

// A DECIMAL128 key (TPC-H q15's total_revenue, join-edge route) sits on the int64 rep. The
// publisher decides per build whether the unscaled values fit: when they do, membership publishes;
// when they do not, membership is declined for the key -- not counted as a type mismatch -- while
// the zone map, exact at DECIMAL128, still publishes.
TEST_CASE("dynamic-filter publisher publishes DECIMAL128 membership only when the build fits int64",
          "[dynamic_filter][publisher]")
{
  publisher_fixture fixture;
  auto const mr = cudf::get_current_device_resource_ref();

  auto const publish = [&](std::vector<__int128_t> const& build_values) {
    fixture.columns.push_back(make_decimal_values<__int128_t>(fixture, kDecimal128, build_values));
    auto key         = make_int64_key(0, 0);
    key.storage_type = kDecimal128;
    auto channel     = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = channel,
                       .route_class              = dynamic_filter_route_class::scan,
                       .accepts_zone_map_filters = true,
                       .key_bindings             = {{.admitted_key_index   = 0,
                                                     .channel_push_ordinal = kProbeColumnIndex,
                                                     .probe_storage_type   = kDecimal128}}});
    auto replica_spaces = fixture.replica_spaces;
    dynamic_filter_publish_plan plan{
      {key}, std::move(targets), std::move(replica_spaces), {.emit_zone_map_filters = true}};
    auto const outcome =
      sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
    return std::pair{outcome, channel->filters_for_column(kProbeColumnIndex)};
  };

  SECTION("values within int64: membership and zone map both publish")
  {
    auto const [outcome, snapshot] = publish({100, std::numeric_limits<std::int64_t>::max(), 300});
    REQUIRE(outcome.keys_considered == 1);
    REQUIRE(outcome.keys_skipped_type_mismatch == 0);
    REQUIRE(outcome.zone_map_filters_built == 1);
    REQUIRE(outcome.membership_filters_built == 1);
    REQUIRE(outcome.filters_pushed == 2);
    REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(snapshot) == 1);
    REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_small_in_list_filter>(snapshot) == 1);
    auto const membership = std::find_if(snapshot.begin(), snapshot.end(), [](auto const& f) {
      return dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(f.get()) !=
             nullptr;
    });
    auto const* small =
      static_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(membership->get());
    CHECK(small->domain().rep == sirius::op::membership_key_rep::i64);
    CHECK(small->domain().native == kDecimal128);

    auto const wide = make_decimal_values<__int128_t>(
      fixture,
      kDecimal128,
      {100, 200, 300, static_cast<__int128_t>(std::numeric_limits<std::int64_t>::max()) + 1});
    REQUIRE(membership_mask(*small, wide->view(), fixture) ==
            std::vector<std::uint8_t>{1, 0, 1, 0});
    auto const narrow = make_decimal_values<std::int64_t>(fixture, kDecimal64, {300, 301});
    REQUIRE(membership_mask(*small, narrow->view(), fixture) == std::vector<std::uint8_t>{1, 0});
  }

  SECTION("a value beyond int64: membership declined, zone map kept")
  {
    auto const [outcome, snapshot] = publish({100, static_cast<__int128_t>(1) << 70, 300});
    REQUIRE(outcome.keys_considered == 1);
    REQUIRE(outcome.keys_skipped_type_mismatch == 0);
    REQUIRE(outcome.zone_map_filters_built == 1);
    REQUIRE(outcome.membership_filters_built == 0);
    REQUIRE(outcome.filters_pushed == 1);
    REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(snapshot) == 1);
  }
}

// A VARCHAR join key (TPC-DS i_item_id, ca_county, ...) publishes a fingerprint membership filter
// through the same plan the planner records for it: STRING build storage type, STRING probe
// storage type. The mask answers exact string membership for these keys (fingerprints are
// pairwise distinct) and a zone map over the strings is emitted when zone maps are enabled.
TEST_CASE("dynamic-filter publisher publishes fingerprint membership for STRING keys",
          "[dynamic_filter][publisher][string]")
{
  publisher_fixture fixture;

  auto const publish = [&](bool emit_zone_maps) {
    auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = channel,
                       .route_class              = dynamic_filter_route_class::scan,
                       .accepts_zone_map_filters = true,
                       .key_bindings             = {{.admitted_key_index   = 0,
                                                     .channel_push_ordinal = kProbeColumnIndex,
                                                     .probe_storage_type   = kString}}});
    auto key            = make_int64_key(0, 0);
    key.storage_type    = kString;
    auto replica_spaces = fixture.replica_spaces;
    sirius::op::dynamic_filter_publication_policy policy{};
    policy.emit_zone_map_filters = emit_zone_maps;
    dynamic_filter_publish_plan plan{{key}, std::move(targets), std::move(replica_spaces), policy};
    auto const outcome =
      sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);
    REQUIRE(outcome.keys_considered == 1);
    REQUIRE(outcome.keys_skipped_type_mismatch == 0);
    REQUIRE(outcome.membership_filters_built == 1);
    REQUIRE(outcome.zone_map_filters_built == (emit_zone_maps ? 1 : 0));
    REQUIRE(outcome.filters_pushed == (emit_zone_maps ? 2 : 1));
    return channel->filters_for_column(kProbeColumnIndex);
  };

  SECTION("small IN-list tier")
  {
    fixture.columns.push_back(
      make_string_values(fixture, {"AAAAAAAAAAAAAAAA", "", "county of long names and spaces"}));
    auto const snapshot = publish(/*emit_zone_maps=*/false);
    REQUIRE(snapshot.size() == 1);
    auto const* small =
      dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(snapshot.front().get());
    REQUIRE(small != nullptr);
    CHECK(small->domain().rep == sirius::op::membership_key_rep::u64);
    CHECK(small->domain().family == sirius::op::membership_key_family::string_hash);
    CHECK(small->domain().native == kString);

    auto const probe = make_string_values(
      fixture,
      {"AAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAB", "", " ", "county of long names and spaces"});
    REQUIRE(membership_mask(*snapshot.front(), probe->view(), fixture) ==
            std::vector<std::uint8_t>{1, 0, 1, 0, 1});
  }

  SECTION("hash IN-list tier, with a zone map beside it")
  {
    std::vector<std::string> keys;
    for (std::size_t i = 0; i < sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 5;
         ++i) {
      keys.push_back("item_" + std::to_string(i * 2));
    }
    fixture.columns.push_back(make_string_values(fixture, keys));
    auto const snapshot = publish(/*emit_zone_maps=*/true);
    REQUIRE(snapshot.size() == 2);
    REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_zone_map_filter>(snapshot) == 1);
    REQUIRE(count_filters_of_kind<sirius::op::sirius_dynamic_in_list_filter>(snapshot) == 1);
    auto const it      = std::find_if(snapshot.begin(), snapshot.end(), [](auto const& f) {
      return dynamic_cast<sirius::op::sirius_dynamic_in_list_filter const*>(f.get()) != nullptr;
    });
    auto const* hashed = dynamic_cast<sirius::op::sirius_dynamic_in_list_filter const*>(it->get());
    REQUIRE(hashed->has_persistent_set());
    CHECK(hashed->domain().rep == sirius::op::membership_key_rep::u64);
    CHECK(hashed->domain().family == sirius::op::membership_key_family::string_hash);

    auto const probe = make_string_values(fixture, {"item_0", "item_1", "item_2", "item_", ""});
    REQUIRE(membership_mask(**it, probe->view(), fixture) ==
            std::vector<std::uint8_t>{1, 0, 1, 0, 0});
  }
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

  SECTION("direct binding with a membership-unsupported probe storage type")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::direct,
       .accepts_zone_map_filters = false,
       .key_bindings             = {
         {.admitted_key_index = 0, .channel_push_ordinal = 0, .probe_storage_type = kFloat64}}});
    // The admitted key is FLOAT64 as well, so probe/build equality holds and only the
    // membership_key_supported arm rejects.
    auto key         = make_int64_key(0, 0);
    key.storage_type = kFloat64;
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan({key}, std::move(targets), std::move(fixture.replica_spaces)),
      std::invalid_argument);
  }

  SECTION("direct binding over a DATE key is accepted")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::direct,
       .accepts_zone_map_filters = false,
       .key_bindings             = {
         {.admitted_key_index = 0, .channel_push_ordinal = 0, .probe_storage_type = kDays}}});
    auto key         = make_int64_key(0, 0);
    key.storage_type = kDays;
    REQUIRE_NOTHROW(
      dynamic_filter_publish_plan({key}, std::move(targets), std::move(fixture.replica_spaces)));
  }

  SECTION("direct binding mixing timestamp units is rejected")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = channel,
                       .route_class              = dynamic_filter_route_class::direct,
                       .accepts_zone_map_filters = false,
                       .key_bindings             = {{.admitted_key_index   = 0,
                                                     .channel_push_ordinal = 0,
                                                     .probe_storage_type   = cudf::data_type{
                                           cudf::type_id::TIMESTAMP_MICROSECONDS}}}});
    auto key         = make_int64_key(0, 0);
    key.storage_type = cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS};
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan({key}, std::move(targets), std::move(fixture.replica_spaces)),
      std::invalid_argument);
  }

  SECTION("direct binding over a supported narrow integer type is accepted")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::direct,
       .accepts_zone_map_filters = false,
       .key_bindings             = {{.admitted_key_index   = 0,
                                     .channel_push_ordinal = 0,
                                     .probe_storage_type   = cudf::data_type{cudf::type_id::INT8}}}});
    auto key         = make_int64_key(0, 0);
    key.storage_type = cudf::data_type{cudf::type_id::INT8};
    REQUIRE_NOTHROW(
      dynamic_filter_publish_plan({key}, std::move(targets), std::move(fixture.replica_spaces)));
  }

  SECTION("direct binding over a same-scale decimal key is accepted")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::direct,
       .accepts_zone_map_filters = false,
       .key_bindings             = {
         {.admitted_key_index = 0, .channel_push_ordinal = 0, .probe_storage_type = kDecimal128}}});
    auto key         = make_int64_key(0, 0);
    key.storage_type = kDecimal128;
    REQUIRE_NOTHROW(
      dynamic_filter_publish_plan({key}, std::move(targets), std::move(fixture.replica_spaces)));
  }

  SECTION("direct binding whose decimal scale differs from the admitted key's is rejected")
  {
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::direct,
       .accepts_zone_map_filters = false,
       .key_bindings             = {{.admitted_key_index   = 0,
                                     .channel_push_ordinal = 0,
                                     .probe_storage_type   = cudf::data_type{cudf::type_id::DECIMAL64, -3}}}});
    auto key         = make_int64_key(0, 0);
    key.storage_type = kDecimal64;
    REQUIRE_THROWS_AS(
      dynamic_filter_publish_plan({key}, std::move(targets), std::move(fixture.replica_spaces)),
      std::invalid_argument);
  }

  SECTION("direct binding over STRING storage on both sides is accepted")
  {
    // Join-edge probes of CTE outputs (TPC-DS q4/q11/q74 customer_id) are STRING operator
    // outputs; the string family admits them through the same equal-type gate as integers.
    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back(
      {.filter_set               = channel,
       .route_class              = dynamic_filter_route_class::direct,
       .accepts_zone_map_filters = false,
       .key_bindings             = {
         {.admitted_key_index = 0, .channel_push_ordinal = 0, .probe_storage_type = kString}}});
    auto key         = make_int64_key(0, 0);
    key.storage_type = kString;
    REQUIRE_NOTHROW(
      dynamic_filter_publish_plan({key}, std::move(targets), std::move(fixture.replica_spaces)));
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
