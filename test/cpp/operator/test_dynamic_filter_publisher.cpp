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
 *  - Multi-partition accumulation: complete_build_snapshot validation, the aggregate Bloom
 *    budget, dynamic_filter_accumulator exact-ID accumulation and root OR-reduction, and
 *    dynamic_filter_publication_session one-shot/accumulated arbitration under concurrency,
 *    logging failure, and strict-replication failure.
 *
 * The fraction semantics themselves (0 always demotes to the Bloom, small-list precedence, the
 * exact inclusive boundary, and the no-Bloom exception) are pinned GPU-free in
 * `test_dynamic_filter_source_policy.cpp`.
 */

#include "log/sink.hpp"
#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <semaphore>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr int kDeviceId                 = 0;
constexpr std::size_t kProbeColumnIndex = 7;
constexpr std::uint64_t kJoinOperatorId = 41;
constexpr auto kConcurrencyTimeout      = std::chrono::seconds{30};

using sirius::op::dynamic_filter_publish_plan;
using sirius::op::dynamic_filter_route_class;

struct cuda_callback_gate {
  std::binary_semaphore proceed{0};
  std::atomic<bool> timed_out{false};
};

void CUDART_CB wait_for_cuda_test_gate(void* opaque) noexcept
{
  auto& gate = *static_cast<cuda_callback_gate*>(opaque);
  if (!gate.proceed.try_acquire_for(kConcurrencyTimeout)) {
    gate.timed_out.store(true, std::memory_order_relaxed);
  }
}

class throwing_log_sink final : public sirius::log::sink {
 public:
  void set_level(sirius::log::level) override {}

  [[nodiscard]] bool should_log(sirius::log::level) const override { return true; }

  void log(sirius::log::level, std::source_location const&, std::string_view) override
  {
    throw std::runtime_error("injected logging failure");
  }

  bool flush() override { return true; }
};

class scoped_throwing_log_sink final {
 public:
  scoped_throwing_log_sink()
    : _previous(sirius::log::get_sink()), _throwing(std::make_shared<throwing_log_sink>())
  {
    sirius::log::set_sink(_throwing);
  }

  ~scoped_throwing_log_sink() { sirius::log::set_sink(std::move(_previous)); }

  scoped_throwing_log_sink(scoped_throwing_log_sink const&)            = delete;
  scoped_throwing_log_sink& operator=(scoped_throwing_log_sink const&) = delete;

 private:
  std::shared_ptr<sirius::log::sink> _previous;
  std::shared_ptr<throwing_log_sink> _throwing;
};

[[nodiscard]] sirius::op::complete_build_snapshot make_complete_build_snapshot(
  std::uint64_t rows, std::vector<std::uint64_t> batch_ids, std::size_t partition_count = 2)
{
  auto snapshot =
    sirius::op::complete_build_snapshot::try_create(rows, std::move(batch_ids), partition_count);
  if (!snapshot) { throw std::logic_error("invalid complete build snapshot in test"); }
  return std::move(*snapshot);
}

constexpr auto kInt64   = cudf::data_type{cudf::type_id::INT64};
constexpr auto kFloat64 = cudf::data_type{cudf::type_id::FLOAT64};

template <typename MemoryManager>
std::vector<sirius::op::dynamic_filter_replica_space> get_replica_spaces(
  MemoryManager& memory_manager, std::size_t expected_device_count = 1)
{
  auto const gpu_spaces  = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto const host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(gpu_spaces.size() == expected_device_count);
  REQUIRE_FALSE(host_spaces.empty());

  std::vector<sirius::op::dynamic_filter_replica_space> result;
  result.reserve(gpu_spaces.size());
  for (auto const* gpu_space_view : gpu_spaces) {
    auto* gpu_space = memory_manager.get_memory_space(cucascade::memory::Tier::GPU,
                                                      gpu_space_view->get_device_id());
    REQUIRE(gpu_space != nullptr);
    auto const local_host =
      std::find_if(host_spaces.begin(), host_spaces.end(), [gpu_space](auto const* host_space) {
        return host_space->get_device_id() == gpu_space->get_device_id();
      });
    auto const* host_space = local_host == host_spaces.end() ? host_spaces.front() : *local_host;
    result.emplace_back(*gpu_space, *host_space);
  }
  std::sort(result.begin(), result.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.get_gpu_space().get_device_id() < rhs.get_gpu_space().get_device_id();
  });
  return result;
}

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_per_thread_memory_manager(
  std::size_t n_gpus)
{
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(n_gpus)
    .set_gpu_usage_limit(256ull << 20)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_numa_region_capacity(512ull << 20)
    .use_gpu_id_as_host_id()
    .track_reservation_per_stream(false)
    .set_reservation_fraction_per_numa_region(0.75);
  return std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
}

bool require_two_gpus()
{
  int count      = 0;
  auto const err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count < 2) {
    WARN("dynamic-filter accumulator reservation test requires two visible GPUs; skipping");
    return false;
  }
  return true;
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
void require_published_membership(std::size_t rows,
                                  sirius::op::dynamic_filter_publication_policy policy = {})
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
    {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces), policy};

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

dynamic_filter_publish_plan make_accumulator_plan(
  publisher_fixture& fixture,
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> const& channel,
  sirius::op::dynamic_filter_publication_policy policy = {})
{
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  return dynamic_filter_publish_plan{
    {make_int64_key(0, 0)}, std::move(targets), std::move(fixture.replica_spaces), policy};
}

dynamic_filter_publish_plan make_accumulator_plan_from_spaces(
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> const& channel,
  std::vector<sirius::op::dynamic_filter_replica_space> replica_spaces,
  sirius::op::dynamic_filter_publication_policy policy = {})
{
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  return dynamic_filter_publish_plan{
    {make_int64_key(0, 0)}, std::move(targets), std::move(replica_spaces), policy};
}

cudf::table_view one_column_view(cudf::column const& column)
{
  return cudf::table_view{std::vector<cudf::column_view>{column.view()}};
}

}  // namespace

TEST_CASE("complete build snapshots reject invalid identity and partition geometry",
          "[dynamic_filter][publisher][snapshot][partition_snapshot]")
{
  using sirius::op::complete_build_snapshot;

  SECTION("batch IDs must be present")
  {
    REQUIRE_FALSE(complete_build_snapshot::try_create(0, {}, 2));
  }

  SECTION("batch IDs must be unique")
  {
    REQUIRE_FALSE(complete_build_snapshot::try_create(2, {17, 17}, 2));
  }

  SECTION("partition count must exceed one")
  {
    REQUIRE_FALSE(complete_build_snapshot::try_create(1, {17}, 0));
    REQUIRE_FALSE(complete_build_snapshot::try_create(1, {17}, 1));
  }

  SECTION("row count must fit size_t")
  {
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
      auto const first_unrepresentable =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) + 1;
      REQUIRE_FALSE(complete_build_snapshot::try_create(first_unrepresentable, {17}, 2));
    } else {
      STATIC_REQUIRE(sizeof(std::size_t) == sizeof(std::uint64_t));
    }
  }

  auto snapshot = complete_build_snapshot::try_create(5, {11, 29}, 3);
  REQUIRE(snapshot);
  REQUIRE(snapshot->total_rows() == 5);
  REQUIRE(snapshot->partition_count() == 3);
  REQUIRE(std::vector<std::uint64_t>(snapshot->batch_ids().begin(), snapshot->batch_ids().end()) ==
          std::vector<std::uint64_t>{11, 29});
}

TEST_CASE("PARTITION build summaries preserve exact global geometry",
          "[dynamic_filter][publisher][snapshot][partition_snapshot]")
{
  using sirius::op::detail::complete_build_batch_summary;
  using sirius::op::detail::try_summarize_complete_build;

  SECTION("rows and original IDs are frozen together")
  {
    std::vector<complete_build_batch_summary> const batches{{.batch_id = 41, .rows = 3},
                                                            {.batch_id = 73, .rows = 5}};
    auto snapshot = try_summarize_complete_build(batches, 4);
    REQUIRE(snapshot);
    REQUIRE(snapshot->total_rows() == 8);
    REQUIRE(snapshot->partition_count() == 4);
    REQUIRE(
      std::vector<std::uint64_t>(snapshot->batch_ids().begin(), snapshot->batch_ids().end()) ==
      std::vector<std::uint64_t>{41, 73});
  }

  SECTION("row summation rejects uint64 overflow")
  {
    std::vector<complete_build_batch_summary> const batches{
      {.batch_id = 41, .rows = std::numeric_limits<std::uint64_t>::max()},
      {.batch_id = 73, .rows = 1}};
    REQUIRE_FALSE(try_summarize_complete_build(batches, 2));
  }
}

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

TEST_CASE("one-shot publication ignores the per-GPU Bloom cap",
          "[dynamic_filter][publisher][bloom_budget]")
{
  // Dev parity: max_dynamic_filter_bloom_bytes_per_gpu budgets only the multi-partition
  // accumulated Bloom, so a Bloom-selected one-shot key builds and pushes even under a zero cap.
  publisher_fixture fixture;
  fixture.add_key_column(sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1);

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64}}});
  dynamic_filter_publish_plan plan{{make_int64_key(0, 0)},
                                   std::move(targets),
                                   std::move(fixture.replica_spaces),
                                   {.inlist_max_l2_fraction = 1e-12, .max_bloom_bytes_per_gpu = 0}};

  auto const outcome =
    sirius::op::publish_dynamic_filters(plan, fixture.build_view(), fixture.stream);

  REQUIRE(outcome.keys_considered == 1);
  REQUIRE(outcome.keys_skipped_bloom_size_gate == 0);
  REQUIRE(outcome.membership_filters_built == 1);
  REQUIRE(outcome.filters_pushed == 1);
  auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(snapshot.size() == 1);
  REQUIRE(dynamic_cast<sirius::op::sirius_dynamic_bloom_filter const*>(snapshot.front().get()) !=
          nullptr);
}

TEST_CASE("Bloom construction rejects a footprint whose allocator alignment would overflow",
          "[dynamic_filter][publisher][bloom_budget]")
{
  publisher_fixture fixture;
  auto& source_space  = fixture.replica_spaces.front().get_gpu_space();
  auto const size_max = std::numeric_limits<std::size_t>::max();
  // Produces SIZE_MAX/32 blocks: raw bytes fit at SIZE_MAX-31, but CUDA alignment overflows.
  auto const expected_num_keys = size_max / 2 - 15;

  REQUIRE(sirius::op::sirius_dynamic_bloom_filter::estimated_bytes(expected_num_keys) == size_max);
  REQUIRE_THROWS_AS(
    sirius::op::sirius_dynamic_bloom_filter(
      kInt64, expected_num_keys, fixture.stream, source_space.get_default_allocator()),
    std::length_error);
}

TEST_CASE("dynamic-filter publisher demotes the hash IN-list to Bloom above the L2 fraction",
          "[dynamic_filter][publisher]")
{
  // A vanishing fraction makes the residency threshold (fraction x L2) smaller than any real
  // hash set, so the smallest hash-tier build must demote to the Bloom; this pins the
  // plan-to-publisher plumbing of inlist_max_l2_fraction end to end.
  require_published_membership<sirius::op::sirius_dynamic_bloom_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1,
    {.inlist_max_l2_fraction = 1e-12});
}

TEST_CASE("dynamic-filter publisher fraction 1.0 reproduces the legacy L2-fit rule",
          "[dynamic_filter][publisher]")
{
  // fraction = 1.0 -> threshold = the full L2, so every L2-fitting set keeps the exact IN-list.
  require_published_membership<sirius::op::sirius_dynamic_in_list_filter>(
    sirius::op::sirius_dynamic_small_in_list_filter::k_max_keys + 1,
    {.inlist_max_l2_fraction = 1.0});
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
  require_published_membership<sirius::op::sirius_dynamic_in_list_filter>(
    rows, {.inlist_max_l2_fraction = ratio * (1 + 1e-9)});
  require_published_membership<sirius::op::sirius_dynamic_bloom_filter>(
    rows, {.inlist_max_l2_fraction = ratio * (1 - 1e-9)});
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

TEST_CASE("one-shot failure drains queued source reads before rethrowing the original error",
          "[dynamic_filter][publisher][publication_session][failure_drain]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3);
  fixture.stream.synchronize();

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64},
                                                  {.admitted_key_index   = 1,
                                                   .channel_push_ordinal = kProbeColumnIndex + 1,
                                                   .probe_storage_type   = kInt64}}});
  dynamic_filter_publish_plan plan{{make_int64_key(0, 0), make_int64_key(1, 1)},
                                   std::move(targets),
                                   std::move(fixture.replica_spaces)};

  std::binary_semaphore invalid_key_reached{0};
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  hooks.before_one_shot_key = [&](std::size_t key_index) {
    if (key_index == 1) { invalid_key_reached.release(); }
  };

  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true, std::move(hooks));
  rmm::cuda_stream task_stream{rmm::cuda_stream::flags::non_blocking};
  cuda_callback_gate gate;
  REQUIRE(cudaLaunchHostFunc(task_stream.value(), wait_for_cuda_test_gate, &gate) == cudaSuccess);

  REQUIRE(session.try_claim_one_shot());
  auto publication = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    session.publish_one_shot(fixture.build_view(), task_stream.view());
  });

  auto const reached = invalid_key_reached.try_acquire_for(kConcurrencyTimeout);
  auto const blocked = publication.wait_for(std::chrono::milliseconds{100});
  gate.proceed.release();
  auto const completed = publication.wait_for(kConcurrencyTimeout);

  bool caught_original = false;
  std::string message;
  if (completed == std::future_status::ready) {
    try {
      publication.get();
    } catch (std::logic_error const& error) {
      caught_original = true;
      message         = error.what();
    }
  }

  REQUIRE(reached);
  REQUIRE(blocked == std::future_status::timeout);
  REQUIRE(completed == std::future_status::ready);
  REQUIRE_FALSE(gate.timed_out.load(std::memory_order_relaxed));
  REQUIRE(caught_original);
  REQUIRE(message.find("build ordinal") != std::string::npos);
  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(stats.publications_failed.load() == 1);
  REQUIRE(channel->empty());
}

TEST_CASE("concurrent one-shot callers claim publication exactly once",
          "[dynamic_filter][publisher][publication_session][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3);
  fixture.stream.synchronize();

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  std::binary_semaphore first_key_reached{0};
  std::binary_semaphore release_first_caller{0};
  std::atomic<bool> hook_timed_out{false};
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  hooks.before_one_shot_key = [&](std::size_t key_index) {
    if (key_index != 0) { return; }
    first_key_reached.release();
    if (!release_first_caller.try_acquire_for(kConcurrencyTimeout)) {
      hook_timed_out.store(true, std::memory_order_relaxed);
      throw std::runtime_error("timed out waiting to release the first one-shot caller");
    }
  };

  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true, std::move(hooks));
  auto first_publication = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    if (session.try_claim_one_shot()) {
      session.publish_one_shot(fixture.build_view(), fixture.stream);
    }
  });

  auto const first_claimed = first_key_reached.try_acquire_for(kConcurrencyTimeout);
  if (!first_claimed) { release_first_caller.release(); }
  REQUIRE(first_claimed);
  REQUIRE(stats.publication_attempts.load() == 1);

  if (session.try_claim_one_shot()) {
    session.publish_one_shot(fixture.build_view(), fixture.stream);
  }
  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(channel->empty());

  release_first_caller.release();
  REQUIRE(first_publication.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  first_publication.get();
  REQUIRE_FALSE(hook_timed_out.load(std::memory_order_relaxed));
  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.membership_filters_built.load() == 1);
  REQUIRE(stats.filters_pushed.load() == 1);
  REQUIRE(channel->filter_count() == 1);
}

TEST_CASE("finalization leaves a claimed one-shot window for its owner",
          "[dynamic_filter][publisher][publication_session]")
{
  // The regression for the routing/finalize race: the hook claims before routing, routing can
  // finish the build pipeline and run finalization, and finalization must never close a window
  // this delivery already owns. The explicit claim makes the race expressible sequentially.
  publisher_fixture fixture;
  fixture.add_key_column(3);
  fixture.stream.synchronize();
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  REQUIRE(session.try_claim_one_shot());
  session.finalize_or_abort();
  session.publish_one_shot(fixture.build_view(), fixture.stream);

  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.filters_pushed.load() == 1);
  REQUIRE(channel->filter_count() == 1);
}

TEST_CASE("a released claim reopens the window", "[dynamic_filter][publisher][publication_session]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3);
  fixture.stream.synchronize();
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  REQUIRE(session.try_claim_one_shot());
  session.reopen_from_claim();
  REQUIRE(session.is_open());
  REQUIRE(session.try_claim_one_shot());
  REQUIRE(stats.publication_attempts.load() == 2);
}

TEST_CASE("publish_one_shot without a claim is inert",
          "[dynamic_filter][publisher][publication_session]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3);
  fixture.stream.synchronize();
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  session.publish_one_shot(fixture.build_view(), fixture.stream);

  REQUIRE(stats.publication_attempts.load() == 0);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(channel->empty());
  REQUIRE(session.is_open());
}

TEST_CASE("an exception between claim and publication fails the claim",
          "[dynamic_filter][publisher][publication_session]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3);
  fixture.stream.synchronize();
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  REQUIRE(session.try_claim_one_shot());
  session.fail_claim();

  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.publications_failed.load() == 1);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(channel->empty());

  // FAILED is terminal: a later claim must not reopen the window.
  REQUIRE_FALSE(session.is_open());
  REQUIRE_FALSE(session.try_claim_one_shot());
  REQUIRE(stats.publication_attempts.load() == 1);
}

TEST_CASE("finalization during an in-flight one-shot publication is inert",
          "[dynamic_filter][publisher][publication_session][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3);
  fixture.stream.synchronize();
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);

  std::atomic<bool> hook_timed_out{false};
  sirius::op::dynamic_filter_stats stats;
  std::optional<sirius::op::dynamic_filter_publication_session> session;
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  std::future<void> finalization;
  hooks.before_one_shot_key = [&](std::size_t) {
    // Run finalization on a sibling thread while this publication is in flight and wait for its
    // return before letting the publication proceed.
    finalization = std::async(std::launch::async, [&] { session->finalize_or_abort(); });
    if (finalization.wait_for(kConcurrencyTimeout) != std::future_status::ready) {
      hook_timed_out.store(true, std::memory_order_relaxed);
    }
  };
  session.emplace(plan, &stats, true, std::move(hooks));

  REQUIRE(session->try_claim_one_shot());
  session->publish_one_shot(fixture.build_view(), fixture.stream);

  REQUIRE_FALSE(hook_timed_out.load(std::memory_order_relaxed));
  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.filters_pushed.load() == 1);
  REQUIRE(channel->filter_count() == 1);
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

TEST_CASE("multi-partition accumulator publishes only after every exact build ID",
          "[dynamic_filter][publisher][accumulator]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(
    fixture,
    channel,
    {.max_bloom_bytes_per_gpu = sirius::op::sirius_dynamic_bloom_filter::estimated_bytes(6)});
  sirius::op::dynamic_filter_accumulator accumulator(plan,
                                                     make_complete_build_snapshot(6, {101, 202}));

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
  REQUIRE(last.exact_contribution_count == 2);
  REQUIRE(last.global_build_rows == 6);
  REQUIRE(last.root_device_id == kDeviceId);
  REQUIRE(last.publication.keys_skipped_bloom_size_gate == 0);
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
  REQUIRE(unknown.state == sirius::op::dynamic_filter_accumulation_result::status::duplicate);
}

TEST_CASE("an unknown ID aborts accumulation before completion",
          "[dynamic_filter][publisher][accumulator]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(plan,
                                                     make_complete_build_snapshot(6, {101, 202}));

  auto const unknown =
    accumulator.contribute(999, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(unknown.state == sirius::op::dynamic_filter_accumulation_result::status::aborted);
  REQUIRE(unknown.publication.keys_considered == 1);
  REQUIRE(accumulator.aborted());
  REQUIRE_FALSE(accumulator.complete());

  auto const expected =
    accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(expected.state == sirius::op::dynamic_filter_accumulation_result::status::aborted);
  REQUIRE(channel->empty());
}

TEST_CASE("zero-row accumulated snapshots complete without constructing a Bloom",
          "[dynamic_filter][publisher][accumulator][snapshot]")
{
  publisher_fixture fixture;
  fixture.add_key_column(0);
  fixture.add_key_column(0);
  std::size_t insert_syncs = 0;
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.after_insert_sync = [&](std::uint64_t) { ++insert_syncs; };
  auto channel            = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan               = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(
    plan, make_complete_build_snapshot(0, {101, 202}), std::move(hooks));

  auto const first =
    accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
  auto const last =
    accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);
  REQUIRE(last.state == sirius::op::dynamic_filter_accumulation_result::status::published);
  REQUIRE(last.exact_contribution_count == 2);
  REQUIRE(last.global_build_rows == 0);
  REQUIRE(last.root_device_id == kDeviceId);
  REQUIRE(last.publication.membership_filters_built == 0);
  REQUIRE(last.publication.filters_pushed == 0);
  // A zero-row snapshot leaves every key inactive, so no contribution synchronizes its stream.
  REQUIRE(insert_syncs == 0);
  REQUIRE(accumulator.complete());
  REQUIRE_FALSE(accumulator.aborted());
  REQUIRE(channel->empty());
}

TEST_CASE("publication session excludes one-shot and accumulated paths",
          "[dynamic_filter][publisher][publication_session]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  SECTION("accumulator claim excludes one-shot publication")
  {
    REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
    REQUIRE_FALSE(session.try_claim_one_shot());
    session.publish_one_shot(fixture.build_view(), fixture.stream);
    REQUIRE(channel->empty());

    session.contribute(kJoinOperatorId, 101, one_column_view(*fixture.columns[0]), fixture.stream);
    session.contribute(kJoinOperatorId, 202, one_column_view(*fixture.columns[1]), fixture.stream);
    REQUIRE(channel->filter_count() == 1);
    auto const events = stats.event_snapshot();
    REQUIRE(events.last_event_id == 1);
    REQUIRE(events.records.size() == 1);
  }

  SECTION("one-shot claim excludes accumulator arming")
  {
    REQUIRE(session.try_claim_one_shot());
    session.publish_one_shot(fixture.build_view(), fixture.stream);
    REQUIRE(channel->filter_count() == 1);
    REQUIRE_FALSE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
    session.contribute(kJoinOperatorId, 101, one_column_view(*fixture.columns[0]), fixture.stream);
    REQUIRE(channel->filter_count() == 1);
    REQUIRE(stats.event_snapshot().records.empty());
  }

  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
}

TEST_CASE("moved-from build snapshots cannot claim a publication session",
          "[dynamic_filter][publisher][publication_session][snapshot]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  auto source      = make_complete_build_snapshot(6, {101, 202});
  auto destination = std::move(source);
  REQUIRE_FALSE(source.valid());
  REQUIRE(destination.valid());

  auto assignment_source      = make_complete_build_snapshot(6, {303, 404});
  auto assignment_destination = make_complete_build_snapshot(6, {505, 606});
  assignment_destination      = std::move(assignment_source);
  REQUIRE_FALSE(assignment_source.valid());
  REQUIRE(assignment_destination.valid());

  REQUIRE_THROWS_AS(sirius::op::dynamic_filter_accumulator(plan, std::move(assignment_source)),
                    std::invalid_argument);

  REQUIRE_FALSE(session.try_arm(std::move(source)));
  REQUIRE(session.is_open());
  REQUIRE(stats.publication_attempts.load() == 0);
  REQUIRE(stats.publications_failed.load() == 0);

  REQUIRE(session.try_arm(std::move(destination)));
  REQUIRE(stats.publication_attempts.load() == 1);
  session.finalize_or_abort();
  REQUIRE(stats.publications_failed.load() == 1);
}

TEST_CASE("publication failure cleanup survives a throwing log sink",
          "[dynamic_filter][publisher][publication_session][logging_failure]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  {
    scoped_throwing_log_sink throwing_sink;
    REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
    session.contribute(kJoinOperatorId, 999, one_column_view(*fixture.columns[0]), fixture.stream);
  }

  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.publications_failed.load() == 1);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(stats.event_snapshot().records.empty());
  REQUIRE(channel->empty());
}

TEST_CASE("incomplete-session cleanup survives a throwing log sink",
          "[dynamic_filter][publisher][publication_session][logging_failure]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  {
    scoped_throwing_log_sink throwing_sink;
    REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
    session.finalize_or_abort();
  }

  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.publications_failed.load() == 1);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(stats.event_snapshot().records.empty());
  REQUIRE(channel->empty());
}

TEST_CASE("successful accumulated session folds statistics exactly once",
          "[dynamic_filter][publisher][publication_session]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
  session.contribute(kJoinOperatorId, 101, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(stats.keys_considered.load() == 0);
  session.contribute(kJoinOperatorId, 202, one_column_view(*fixture.columns[1]), fixture.stream);

  REQUIRE(channel->filter_count() == 1);
  REQUIRE(stats.producers_enabled.load() == 1);
  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.keys_considered.load() == 1);
  REQUIRE(stats.membership_filters_built.load() == 1);
  REQUIRE(stats.filters_pushed.load() == 1);
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  auto const first_events = stats.event_snapshot();
  REQUIRE(first_events.last_event_id == 1);
  REQUIRE(first_events.records.size() == 1);
  CHECK(first_events.records[0].join_operator_id == kJoinOperatorId);
  CHECK(first_events.records[0].exact_contribution_count == 2);
  CHECK(first_events.records[0].device_id == kDeviceId);
  CHECK(first_events.records[0].build_rows == 6);
  CHECK(first_events.records[0].filters_built == 1);
  CHECK(first_events.records[0].active_targets == 1);
  CHECK(first_events.records[0].filters_pushed == 1);

  session.contribute(kJoinOperatorId, 202, one_column_view(*fixture.columns[1]), fixture.stream);
  session.finalize_or_abort();
  session.finalize_or_abort();
  REQUIRE(channel->filter_count() == 1);
  REQUIRE(stats.keys_considered.load() == 1);
  REQUIRE(stats.membership_filters_built.load() == 1);
  REQUIRE(stats.filters_pushed.load() == 1);
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  auto const final_events = stats.event_snapshot();
  REQUIRE(final_events.last_event_id == 1);
  REQUIRE(final_events.records.size() == 1);
}

TEST_CASE("an accumulator counts keys Bloom cannot represent",
          "[dynamic_filter][publisher][accumulator]")
{
  // A FLOAT64 build key has no Bloom representation, so an all-unsupported key set arms an inert
  // accumulator: the session still resolves through the unified completion path, and the new
  // counter reports why nothing was built.
  publisher_fixture fixture;
  auto first_batch  = make_float64_values(fixture, {0.5, 1.5, 2.5});
  auto second_batch = make_float64_values(fixture, {3.5, 4.5, 5.5});

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kFloat64}}});
  dynamic_filter_publish_plan::admitted_key float_key{.planner_condition_index = 0,
                                                      .build_key_ordinal       = 0,
                                                      .storage_type            = kFloat64,
                                                      .key_shape               = {}};
  dynamic_filter_publish_plan plan{
    {float_key}, std::move(targets), std::move(fixture.replica_spaces)};

  std::size_t insert_syncs = 0;
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  hooks.accumulator.after_insert_sync = [&](std::uint64_t) { ++insert_syncs; };
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true, std::move(hooks));

  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
  session.contribute(kJoinOperatorId, 101, one_column_view(*first_batch), fixture.stream);
  session.contribute(kJoinOperatorId, 202, one_column_view(*second_batch), fixture.stream);

  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.keys_skipped_bloom_unsupported.load() == 1);
  REQUIRE(stats.membership_filters_built.load() == 0);
  REQUIRE(stats.filters_pushed.load() == 0);
  REQUIRE(channel->empty());
  // The inert accumulator never runs an insertion block, so no contribution synchronized.
  REQUIRE(insert_syncs == 0);
  auto const events = stats.event_snapshot();
  REQUIRE(events.records.size() == 1);
  CHECK(events.records[0].filters_built == 0);
  CHECK(events.records[0].filters_pushed == 0);
}

TEST_CASE("mixed supported and unsupported keys accumulate only the supported one",
          "[dynamic_filter][publisher][accumulator]")
{
  constexpr std::size_t kIntPushOrdinal   = 3;
  constexpr std::size_t kFloatPushOrdinal = 5;

  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto first_floats  = make_float64_values(fixture, {0.5, 1.5, 2.5});
  auto second_floats = make_float64_values(fixture, {3.5, 4.5, 5.5});

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kIntPushOrdinal,
                                                   .probe_storage_type   = kInt64},
                                                  {.admitted_key_index   = 1,
                                                   .channel_push_ordinal = kFloatPushOrdinal,
                                                   .probe_storage_type   = kFloat64}}});
  dynamic_filter_publish_plan::admitted_key float_key{.planner_condition_index = 1,
                                                      .build_key_ordinal       = 1,
                                                      .storage_type            = kFloat64,
                                                      .key_shape               = {}};
  dynamic_filter_publish_plan plan{
    {make_int64_key(0, 0), float_key}, std::move(targets), std::move(fixture.replica_spaces)};

  auto two_column_view = [](cudf::column const& ints, cudf::column const& floats) {
    return cudf::table_view{std::vector<cudf::column_view>{ints.view(), floats.view()}};
  };

  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
  session.contribute(
    kJoinOperatorId, 101, two_column_view(*fixture.columns[0], *first_floats), fixture.stream);
  session.contribute(
    kJoinOperatorId, 202, two_column_view(*fixture.columns[1], *second_floats), fixture.stream);

  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.keys_skipped_bloom_unsupported.load() == 1);
  REQUIRE(stats.membership_filters_built.load() == 1);
  REQUIRE(stats.filters_pushed.load() == 1);
  REQUIRE(channel->filters_for_column(kFloatPushOrdinal).empty());
  auto const surviving = channel->filters_for_column(kIntPushOrdinal);
  REQUIRE(surviving.size() == 1);
  REQUIRE(dynamic_cast<sirius::op::sirius_dynamic_bloom_filter const*>(surviving.front().get()) !=
          nullptr);
}

TEST_CASE("drained targets complete an accumulated session without fan-out",
          "[dynamic_filter][publisher][publication_session]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  channel->close_for_new_filters();
  auto plan = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
  // The channel drained before arming, so the first contribution seals the accumulator and the
  // second is a terminal no-op.
  session.contribute(kJoinOperatorId, 101, one_column_view(*fixture.columns[0]), fixture.stream);
  session.contribute(kJoinOperatorId, 202, one_column_view(*fixture.columns[1]), fixture.stream);

  REQUIRE(channel->empty());
  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.keys_considered.load() == 1);
  REQUIRE(stats.publications_skipped_targets_drained.load() == 1);
  REQUIRE(stats.membership_filters_built.load() == 0);
  REQUIRE(stats.filters_pushed.load() == 0);
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  auto const events = stats.event_snapshot();
  REQUIRE(events.last_event_id == 1);
  REQUIRE(events.records.size() == 1);
  CHECK(events.records[0].join_operator_id == kJoinOperatorId);
  CHECK(events.records[0].exact_contribution_count == 0);
  CHECK(events.records[0].device_id == kDeviceId);
  CHECK(events.records[0].build_rows == 6);
  CHECK(events.records[0].filters_built == 0);
  CHECK(events.records[0].active_targets == 0);
  CHECK(events.records[0].filters_pushed == 0);
}

TEST_CASE("a session sealed by a mid-accumulation drain reports finished with a drained skip",
          "[dynamic_filter][publisher][publication_session]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
  session.contribute(kJoinOperatorId, 101, one_column_view(*fixture.columns[0]), fixture.stream);
  channel->close_for_new_filters();
  session.contribute(kJoinOperatorId, 202, one_column_view(*fixture.columns[1]), fixture.stream);

  REQUIRE(channel->empty());
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.publications_skipped_targets_drained.load() == 1);
  auto const events = stats.event_snapshot();
  REQUIRE(events.last_event_id == 1);
  REQUIRE(events.records.size() == 1);
  CHECK(events.records[0].exact_contribution_count == 1);
  CHECK(events.records[0].filters_built == 0);
  CHECK(events.records[0].active_targets == 0);
}

TEST_CASE("multi-partition Bloom admission accounts for every key on one GPU",
          "[dynamic_filter][publisher][accumulator][bloom_budget]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 100);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  targets.push_back({.filter_set               = channel,
                     .route_class              = dynamic_filter_route_class::scan,
                     .accepts_zone_map_filters = false,
                     .key_bindings             = {{.admitted_key_index   = 0,
                                                   .channel_push_ordinal = kProbeColumnIndex,
                                                   .probe_storage_type   = kInt64},
                                                  {.admitted_key_index   = 1,
                                                   .channel_push_ordinal = kProbeColumnIndex + 1,
                                                   .probe_storage_type   = kInt64}}});
  dynamic_filter_publish_plan plan{
    {make_int64_key(0, 0), make_int64_key(1, 1)},
    std::move(targets),
    std::move(fixture.replica_spaces),
    {.max_bloom_bytes_per_gpu = sirius::op::sirius_dynamic_bloom_filter::estimated_bytes(3)}};
  sirius::op::dynamic_filter_accumulator accumulator(plan, make_complete_build_snapshot(3, {101}));

  auto const result = accumulator.contribute(101, fixture.build_view(), fixture.stream);
  REQUIRE(result.state == sirius::op::dynamic_filter_accumulation_result::status::published);
  REQUIRE(result.publication.keys_considered == 2);
  REQUIRE(result.publication.keys_skipped_bloom_size_gate == 2);
  REQUIRE(result.publication.membership_filters_built == 0);
  REQUIRE(result.publication.filters_pushed == 0);
  REQUIRE(accumulator.complete());
  REQUIRE_FALSE(accumulator.aborted());
  REQUIRE(channel->empty());
}

TEST_CASE("targets that drain mid-accumulation seal the accumulator without further building",
          "[dynamic_filter][publisher][accumulator]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  std::size_t insert_syncs = 0;
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.after_insert_sync = [&](std::uint64_t) { ++insert_syncs; };
  auto channel            = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan               = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(
    plan, make_complete_build_snapshot(6, {101, 202}), std::move(hooks));

  auto const first =
    accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
  REQUIRE(insert_syncs == 1);

  channel->close_for_new_filters();

  auto const sealed =
    accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);
  REQUIRE(sealed.state == sirius::op::dynamic_filter_accumulation_result::status::published);
  REQUIRE(sealed.exact_contribution_count == 1);
  REQUIRE(sealed.publication.skipped_targets_drained == 1);
  REQUIRE(sealed.publication.membership_filters_built == 0);
  REQUIRE(sealed.publication.filters_pushed == 0);
  // The sealing contribution performed no per-batch device work: no insert, no stream sync.
  REQUIRE(insert_syncs == 1);
  REQUIRE(accumulator.complete());
  REQUIRE(channel->empty());
  REQUIRE_FALSE(accumulator.abort_if_incomplete());

  auto const late =
    accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);
  REQUIRE(late.state == sirius::op::dynamic_filter_accumulation_result::status::duplicate);
}

TEST_CASE("a drain seal during an in-flight contribution publishes exactly once",
          "[dynamic_filter][publisher][accumulator][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  std::binary_semaphore id_claimed{0};
  std::binary_semaphore release_claim{0};
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.after_id_claim = [&](std::uint64_t batch_id) {
    if (batch_id != 101) { return; }
    id_claimed.release();
    if (!release_claim.try_acquire_for(kConcurrencyTimeout)) {
      throw std::runtime_error("timed out waiting to release the claimed contribution");
    }
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(
    plan, make_complete_build_snapshot(6, {101, 202}), std::move(hooks));
  rmm::cuda_stream task_stream{rmm::cuda_stream::flags::non_blocking};

  auto first_future         = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    return accumulator.contribute(101, one_column_view(*fixture.columns[0]), task_stream.view());
  });
  auto const claim_observed = id_claimed.try_acquire_for(kConcurrencyTimeout);
  if (!claim_observed) { release_claim.release(); }
  REQUIRE(claim_observed);

  channel->close_for_new_filters();
  auto const sealed =
    accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);
  REQUIRE(sealed.state == sirius::op::dynamic_filter_accumulation_result::status::published);
  REQUIRE(sealed.exact_contribution_count == 0);

  release_claim.release();
  REQUIRE(first_future.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  auto const first = first_future.get();

  // The released contribution observes the seal through the final locked block's terminal
  // check: the documented post-completion `duplicate`, not a stale `pending`, and the sealed
  // result's exact_contribution_count stays frozen at 0.
  REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::duplicate);
  REQUIRE(accumulator.complete());
  REQUIRE(channel->empty());
}

TEST_CASE("a contribution failing after a drain seal reports the sealed terminal",
          "[dynamic_filter][publisher][publication_session][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  auto wrong_type = cudf::sequence(3,
                                   cudf::numeric_scalar<std::int32_t>(3, true, fixture.stream),
                                   cudf::numeric_scalar<std::int32_t>(1, true, fixture.stream),
                                   fixture.stream,
                                   cudf::get_current_device_resource_ref());
  fixture.stream.synchronize();

  std::binary_semaphore id_claimed{0};
  std::binary_semaphore release_claim{0};
  // The semaphore handshake strictly orders the two contribute calls, so the plain vector is
  // written race-free: the sealing status lands before the mismatched contribution is released.
  std::vector<sirius::op::dynamic_filter_accumulation_result::status> observed_statuses;
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  hooks.accumulator.after_id_claim = [&](std::uint64_t batch_id) {
    if (batch_id != 101) { return; }
    id_claimed.release();
    if (!release_claim.try_acquire_for(kConcurrencyTimeout)) {
      throw std::runtime_error("timed out waiting to release the claimed contribution");
    }
  };
  hooks.after_accumulation_result = [&](auto status) { observed_statuses.push_back(status); };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true, std::move(hooks));
  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));

  rmm::cuda_stream task_stream{rmm::cuda_stream::flags::non_blocking};
  auto mismatch_future      = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    session.contribute(kJoinOperatorId, 101, one_column_view(*wrong_type), task_stream.view());
  });
  auto const claim_observed = id_claimed.try_acquire_for(kConcurrencyTimeout);
  if (!claim_observed) { release_claim.release(); }
  REQUIRE(claim_observed);

  channel->close_for_new_filters();
  session.contribute(kJoinOperatorId, 202, one_column_view(*fixture.columns[0]), fixture.stream);

  release_claim.release();
  REQUIRE(mismatch_future.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  mismatch_future.get();

  // The type-mismatched contribution failed after the seal, so it must report the sealed
  // terminal (duplicate), leave the session finished, and record no failure or mismatch.
  using status = sirius::op::dynamic_filter_accumulation_result::status;
  REQUIRE(observed_statuses == std::vector<status>{status::published, status::duplicate});
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.publications_skipped_targets_drained.load() == 1);
  REQUIRE(stats.keys_skipped_type_mismatch.load() == 0);
  REQUIRE(stats.event_snapshot().records.size() == 1);
  REQUIRE(channel->empty());
}

TEST_CASE("an inert budget-gated accumulator contributes without stream synchronization",
          "[dynamic_filter][publisher][accumulator][bloom_budget]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  std::size_t insert_syncs = 0;
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.after_insert_sync = [&](std::uint64_t) { ++insert_syncs; };
  auto channel            = std::make_shared<sirius::op::sirius_dynamic_filter_set>();

  SECTION("budget gate zeroes every key")
  {
    auto plan = make_accumulator_plan(fixture, channel, {.max_bloom_bytes_per_gpu = 1});
    sirius::op::dynamic_filter_accumulator accumulator(
      plan, make_complete_build_snapshot(6, {101, 202}), std::move(hooks));

    auto const first =
      accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream);
    REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
    auto const last =
      accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);
    REQUIRE(last.state == sirius::op::dynamic_filter_accumulation_result::status::published);
    REQUIRE(insert_syncs == 0);
    REQUIRE(last.publication.keys_skipped_bloom_size_gate == 1);
    REQUIRE(channel->empty());
  }

  SECTION("active keys still synchronize per accepted contribution")
  {
    auto plan = make_accumulator_plan(fixture, channel);
    sirius::op::dynamic_filter_accumulator accumulator(
      plan, make_complete_build_snapshot(6, {101, 202}), std::move(hooks));

    auto const first =
      accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream);
    REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
    auto const last =
      accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);
    REQUIRE(last.state == sirius::op::dynamic_filter_accumulation_result::status::published);
    REQUIRE(insert_syncs == 2);
    REQUIRE(channel->filter_count() == 1);
  }
}

TEST_CASE("multi-partition accumulator close prevents publication with a missing ID",
          "[dynamic_filter][publisher][accumulator]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(plan,
                                                     make_complete_build_snapshot(6, {11, 22}));

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
  sirius::op::dynamic_filter_accumulator accumulator(plan,
                                                     make_complete_build_snapshot(6, {31, 32}));
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
    sirius::op::dynamic_filter_accumulator accumulator(plan,
                                                       make_complete_build_snapshot(6, {101, 202}));
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

  std::binary_semaphore id_claimed{0};
  std::binary_semaphore release_claim{0};
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.after_id_claim = [&](std::uint64_t batch_id) {
    if (batch_id != 101) { return; }
    id_claimed.release();
    if (!release_claim.try_acquire_for(kConcurrencyTimeout)) {
      throw std::runtime_error("timed out waiting to release the claimed contribution");
    }
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(
    plan, make_complete_build_snapshot(6, {101, 202}), std::move(hooks));
  rmm::cuda_stream task_stream{rmm::cuda_stream::flags::non_blocking};

  auto first_future         = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    return accumulator.contribute(101, one_column_view(*fixture.columns[0]), task_stream.view());
  });
  auto const claim_observed = id_claimed.try_acquire_for(kConcurrencyTimeout);
  if (!claim_observed) { release_claim.release(); }
  REQUIRE(claim_observed);
  auto const duplicate =
    accumulator.contribute(101, one_column_view(*fixture.columns[0]), fixture.stream);
  release_claim.release();
  REQUIRE(first_future.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  auto const first = first_future.get();

  REQUIRE(duplicate.state == sirius::op::dynamic_filter_accumulation_result::status::duplicate);
  REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
  REQUIRE(channel->empty());

  auto const last =
    accumulator.contribute(202, one_column_view(*fixture.columns[1]), fixture.stream);
  REQUIRE(last.state == sirius::op::dynamic_filter_accumulation_result::status::published);
  REQUIRE(channel->filter_count() == 1);
}

TEST_CASE("same-device contributions are not serialized behind a sibling's stream drain",
          "[dynamic_filter][publisher][accumulator][concurrency]")
{
  // Insertion submission holds the per-device lock; the task-stream drain runs outside it. A
  // gate enqueued on the first contribution's stream keeps its synchronize() genuinely blocked
  // on the device, so the sibling's completion below can only happen when the drain runs outside
  // the lock. With the drain moved back inside the lock, the first contribution blocks in its
  // gated sync while holding the lock (never reaching the submission signal when the hooks stay
  // outside the scope), and one of the two timed REQUIREs below fails after kConcurrencyTimeout;
  // the gate is always released before asserting, so the failure is an assertion, never a hang.
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  std::binary_semaphore first_submitted{0};
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.before_insert_sync = [&](std::uint64_t batch_id) {
    if (batch_id == 101) { first_submitted.release(); }
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(
    plan, make_complete_build_snapshot(6, {101, 202}), std::move(hooks));
  rmm::cuda_stream stream_a{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream stream_b{rmm::cuda_stream::flags::non_blocking};

  cuda_callback_gate gate;
  REQUIRE(cudaLaunchHostFunc(stream_a.value(), wait_for_cuda_test_gate, &gate) == cudaSuccess);

  auto first           = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    return accumulator.contribute(101, one_column_view(*fixture.columns[0]), stream_a.view());
  });
  auto const submitted = first_submitted.try_acquire_for(kConcurrencyTimeout);
  if (!submitted) { gate.proceed.release(); }
  REQUIRE(submitted);

  // The first contribution has released the lock and its stream drain is held open by the gate;
  // the sibling on the same device must submit and complete meanwhile.
  auto second                 = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    return accumulator.contribute(202, one_column_view(*fixture.columns[1]), stream_b.view());
  });
  auto const second_completed = second.wait_for(kConcurrencyTimeout);
  gate.proceed.release();
  REQUIRE(second_completed == std::future_status::ready);
  auto const second_result = second.get();
  REQUIRE(second_result.state == sirius::op::dynamic_filter_accumulation_result::status::pending);

  REQUIRE(first.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  auto const first_result = first.get();
  REQUIRE_FALSE(gate.timed_out.load(std::memory_order_relaxed));
  REQUIRE(first_result.state == sirius::op::dynamic_filter_accumulation_result::status::published);
  REQUIRE(first_result.exact_contribution_count == 2);

  // No lost inserts: the published Bloom must keep every key from both batches.
  auto const snapshot = channel->filters_for_column(kProbeColumnIndex);
  REQUIRE(snapshot.size() == 1);
  auto const probe = make_int64_values(fixture, {0, 1, 2, 3, 4, 5});
  REQUIRE(membership_mask(*snapshot.front(), probe->view(), fixture) ==
          std::vector<std::uint8_t>(6, 1));
}

TEST_CASE("different final contributions race to exactly one publication",
          "[dynamic_filter][publisher][accumulator][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  std::counting_semaphore<2> insertions_complete{0};
  std::counting_semaphore<2> release_insertions{0};
  sirius::op::detail::dynamic_filter_accumulator_test_hooks hooks;
  hooks.after_insert_sync = [&](std::uint64_t) {
    insertions_complete.release();
    if (!release_insertions.try_acquire_for(kConcurrencyTimeout)) {
      throw std::runtime_error("timed out waiting to release completed insertions");
    }
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_accumulator accumulator(
    plan, make_complete_build_snapshot(6, {101, 202}), std::move(hooks));
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
  auto const first_inserted  = insertions_complete.try_acquire_for(kConcurrencyTimeout);
  auto const second_inserted = insertions_complete.try_acquire_for(kConcurrencyTimeout);
  if (!first_inserted || !second_inserted) { release_insertions.release(2); }
  REQUIRE(first_inserted);
  REQUIRE(second_inserted);
  release_insertions.release(2);
  REQUIRE(first_future.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  REQUIRE(second_future.wait_for(kConcurrencyTimeout) == std::future_status::ready);
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

TEST_CASE("finalize wins against an in-flight final contribution",
          "[dynamic_filter][publisher][publication_session][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  std::binary_semaphore final_insertion_complete{0};
  std::binary_semaphore release_final_contribution{0};
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  hooks.accumulator.after_insert_sync = [&](std::uint64_t batch_id) {
    if (batch_id != 202) { return; }
    final_insertion_complete.release();
    if (!release_final_contribution.try_acquire_for(kConcurrencyTimeout)) {
      throw std::runtime_error("timed out waiting to release the final contribution");
    }
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true, std::move(hooks));
  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
  session.contribute(kJoinOperatorId, 101, one_column_view(*fixture.columns[0]), fixture.stream);

  rmm::cuda_stream final_stream{rmm::cuda_stream::flags::non_blocking};
  auto final_future             = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    session.contribute(
      kJoinOperatorId, 202, one_column_view(*fixture.columns[1]), final_stream.view());
  });
  auto const insertion_observed = final_insertion_complete.try_acquire_for(kConcurrencyTimeout);
  if (!insertion_observed) { release_final_contribution.release(); }
  REQUIRE(insertion_observed);

  session.finalize_or_abort();
  REQUIRE(channel->empty());
  REQUIRE(stats.publications_failed.load() == 1);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(stats.membership_filters_built.load() == 0);
  REQUIRE(stats.filters_pushed.load() == 0);

  release_final_contribution.release();
  REQUIRE(final_future.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  final_future.get();
  session.finalize_or_abort();
  REQUIRE(channel->empty());
  REQUIRE(stats.publications_failed.load() == 1);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(stats.event_snapshot().records.empty());
}

TEST_CASE("final contribution wins against finalization after publication",
          "[dynamic_filter][publisher][publication_session][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  std::binary_semaphore publication_complete{0};
  std::binary_semaphore release_terminal_commit{0};
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  hooks.after_accumulation_result = [&](auto status) {
    if (status != sirius::op::dynamic_filter_accumulation_result::status::published) { return; }
    publication_complete.release();
    if (!release_terminal_commit.try_acquire_for(kConcurrencyTimeout)) {
      throw std::runtime_error("timed out waiting to release the terminal commit");
    }
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true, std::move(hooks));
  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
  session.contribute(kJoinOperatorId, 101, one_column_view(*fixture.columns[0]), fixture.stream);

  rmm::cuda_stream final_stream{rmm::cuda_stream::flags::non_blocking};
  auto final_future               = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    session.contribute(
      kJoinOperatorId, 202, one_column_view(*fixture.columns[1]), final_stream.view());
  });
  auto const publication_observed = publication_complete.try_acquire_for(kConcurrencyTimeout);
  if (!publication_observed) { release_terminal_commit.release(); }
  REQUIRE(publication_observed);

  REQUIRE(channel->filter_count() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.publications_finished.load() == 0);
  session.finalize_or_abort();
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.publications_finished.load() == 0);

  release_terminal_commit.release();
  REQUIRE(final_future.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  final_future.get();
  session.finalize_or_abort();
  REQUIRE(channel->filter_count() == 1);
  REQUIRE(stats.membership_filters_built.load() == 1);
  REQUIRE(stats.filters_pushed.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.publications_finished.load() == 1);
  auto const events = stats.event_snapshot();
  REQUIRE(events.last_event_id == 1);
  REQUIRE(events.records.size() == 1);
  CHECK(events.records[0].join_operator_id == kJoinOperatorId);
}

TEST_CASE("a late invalid contribution cannot overturn completed publication",
          "[dynamic_filter][publisher][publication_session][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  std::binary_semaphore publication_complete{0};
  std::binary_semaphore release_terminal_commit{0};
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  hooks.after_accumulation_result = [&](auto status) {
    if (status != sirius::op::dynamic_filter_accumulation_result::status::published) { return; }
    publication_complete.release();
    if (!release_terminal_commit.try_acquire_for(kConcurrencyTimeout)) {
      throw std::runtime_error("timed out waiting to release the terminal commit");
    }
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true, std::move(hooks));
  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));
  session.contribute(kJoinOperatorId, 101, one_column_view(*fixture.columns[0]), fixture.stream);

  rmm::cuda_stream final_stream{rmm::cuda_stream::flags::non_blocking};
  auto final_future               = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    session.contribute(
      kJoinOperatorId, 202, one_column_view(*fixture.columns[1]), final_stream.view());
  });
  auto const publication_observed = publication_complete.try_acquire_for(kConcurrencyTimeout);
  if (!publication_observed) { release_terminal_commit.release(); }
  REQUIRE(publication_observed);

  session.contribute(kJoinOperatorId, 999, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(channel->filter_count() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.publications_finished.load() == 0);

  release_terminal_commit.release();
  REQUIRE(final_future.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  final_future.get();
  REQUIRE(channel->filter_count() == 1);
  REQUIRE(stats.membership_filters_built.load() == 1);
  REQUIRE(stats.filters_pushed.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.event_snapshot().records.size() == 1);
}

TEST_CASE("a throwing pending-result hook cannot overturn concurrent completed publication",
          "[dynamic_filter][publisher][publication_session][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  fixture.add_key_column(3, 3);
  fixture.stream.synchronize();

  std::binary_semaphore pending_result_reached{0};
  std::binary_semaphore published_result_reached{0};
  std::binary_semaphore release_pending_result{0};
  std::binary_semaphore release_published_result{0};
  std::atomic<bool> hook_timed_out{false};
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  hooks.after_accumulation_result = [&](auto status) {
    if (status == sirius::op::dynamic_filter_accumulation_result::status::pending) {
      pending_result_reached.release();
      if (!release_pending_result.try_acquire_for(kConcurrencyTimeout)) {
        hook_timed_out.store(true, std::memory_order_relaxed);
        throw std::runtime_error("timed out waiting to release the pending result");
      }
      throw std::runtime_error("injected failure after a pending result");
    }
    if (status == sirius::op::dynamic_filter_accumulation_result::status::published) {
      published_result_reached.release();
      if (!release_published_result.try_acquire_for(kConcurrencyTimeout)) {
        hook_timed_out.store(true, std::memory_order_relaxed);
        throw std::runtime_error("timed out waiting to release the published result");
      }
    }
  };

  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true, std::move(hooks));
  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));

  rmm::cuda_stream first_stream{rmm::cuda_stream::flags::non_blocking};
  auto first_contribution     = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    session.contribute(
      kJoinOperatorId, 101, one_column_view(*fixture.columns[0]), first_stream.view());
  });
  auto const pending_observed = pending_result_reached.try_acquire_for(kConcurrencyTimeout);
  if (!pending_observed) {
    release_pending_result.release();
    release_published_result.release();
  }
  REQUIRE(pending_observed);

  rmm::cuda_stream final_stream{rmm::cuda_stream::flags::non_blocking};
  auto final_contribution       = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
    session.contribute(
      kJoinOperatorId, 202, one_column_view(*fixture.columns[1]), final_stream.view());
  });
  auto const published_observed = published_result_reached.try_acquire_for(kConcurrencyTimeout);
  if (!published_observed) {
    release_pending_result.release();
    release_published_result.release();
  }
  REQUIRE(published_observed);
  REQUIRE(channel->filter_count() == 1);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(stats.publications_failed.load() == 0);

  release_pending_result.release();
  auto const first_ready                  = first_contribution.wait_for(kConcurrencyTimeout);
  auto const finished_after_pending_throw = stats.publications_finished.load();
  auto const failed_after_pending_throw   = stats.publications_failed.load();
  release_published_result.release();
  REQUIRE(first_ready == std::future_status::ready);
  first_contribution.get();
  REQUIRE(finished_after_pending_throw == 1);
  REQUIRE(failed_after_pending_throw == 0);

  REQUIRE(final_contribution.wait_for(kConcurrencyTimeout) == std::future_status::ready);
  final_contribution.get();
  REQUIRE_FALSE(hook_timed_out.load(std::memory_order_relaxed));
  REQUIRE(channel->filter_count() == 1);
  REQUIRE(stats.membership_filters_built.load() == 1);
  REQUIRE(stats.filters_pushed.load() == 1);
  REQUIRE(stats.publications_finished.load() == 1);
  REQUIRE(stats.publications_failed.load() == 0);
  auto const events = stats.event_snapshot();
  REQUIRE(events.last_event_id == 1);
  REQUIRE(events.records.size() == 1);
  CHECK(events.records[0].exact_contribution_count == 2);
  CHECK(events.records[0].build_rows == 6);
  CHECK(events.records[0].device_id == kDeviceId);
}

TEST_CASE("a throwing aborted-result hook preserves the accumulator outcome",
          "[dynamic_filter][publisher][publication_session][concurrency]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::detail::dynamic_filter_publication_session_test_hooks hooks;
  hooks.after_accumulation_result = [](auto status) {
    if (status == sirius::op::dynamic_filter_accumulation_result::status::aborted) {
      throw std::runtime_error("injected failure after an aborted result");
    }
  };
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true, std::move(hooks));
  REQUIRE(session.try_arm(make_complete_build_snapshot(6, {101, 202})));

  session.contribute(kJoinOperatorId, 999, one_column_view(*fixture.columns[0]), fixture.stream);
  REQUIRE(stats.publication_attempts.load() == 1);
  REQUIRE(stats.keys_considered.load() == 1);
  REQUIRE(stats.publications_finished.load() == 0);
  REQUIRE(stats.publications_failed.load() == 1);
  REQUIRE(stats.event_snapshot().records.empty());
  REQUIRE(channel->empty());
}

TEST_CASE("skip diagnostics do not move after the publication session closes",
          "[dynamic_filter][publisher][publication_session][statistics]")
{
  publisher_fixture fixture;
  fixture.add_key_column(3, 0);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan(fixture, channel);
  sirius::op::dynamic_filter_stats stats;
  sirius::op::dynamic_filter_publication_session session(plan, &stats, true);

  session.finalize_or_abort();
  session.record_source_not_resident();
  session.record_build_not_whole();
  REQUIRE(stats.publication_attempts.load() == 0);
  REQUIRE(stats.publications_skipped_source_not_resident.load() == 0);
  REQUIRE(stats.publications_skipped_build_not_whole.load() == 0);
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
  sirius::op::dynamic_filter_accumulator accumulator(
    plan, make_complete_build_snapshot(6, {101, 202}), std::move(hooks));

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

TEST_CASE("the final contribution GPU roots strict replication under per-thread reservations",
          "[dynamic_filter][publisher][accumulator][mgpu][reservation]")
{
  if (!require_two_gpus()) { return; }

  constexpr int first_device                  = 0;
  constexpr int final_device                  = 1;
  constexpr std::size_t rows_per_contribution = 3;
  auto memory_manager                         = make_per_thread_memory_manager(2);
  auto replica_spaces                         = get_replica_spaces(*memory_manager, 2);
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto plan    = make_accumulator_plan_from_spaces(channel, replica_spaces);
  sirius::op::dynamic_filter_accumulator accumulator(
    plan, make_complete_build_snapshot(2 * rows_per_contribution, {101, 202}));

  std::unique_ptr<cudf::column> first_column;
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{first_device}};
    auto& space       = replica_spaces[0].get_gpu_space();
    auto const stream = space.acquire_stream();
    first_column      = cudf::sequence(rows_per_contribution,
                                  cudf::numeric_scalar<std::int64_t>(0, true, stream),
                                  cudf::numeric_scalar<std::int64_t>(1, true, stream),
                                  stream,
                                  space.get_default_allocator());
    stream.synchronize();
    auto const first = accumulator.contribute(101, one_column_view(*first_column), stream);
    REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);
  }

  std::unique_ptr<cudf::column> final_column;
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{final_device}};
    auto& space       = replica_spaces[1].get_gpu_space();
    auto const stream = space.acquire_stream();
    final_column =
      cudf::sequence(rows_per_contribution,
                     cudf::numeric_scalar<std::int64_t>(rows_per_contribution, true, stream),
                     cudf::numeric_scalar<std::int64_t>(1, true, stream),
                     stream,
                     space.get_default_allocator());
    stream.synchronize();

    // Match gpu_pipeline_task::execute: PER_THREAD tracking occupies GPU 1's adaptor for the
    // whole contribution, regardless of which stream strict replication later acquires.
    auto task_reservation = space.make_reservation_or_null(16ull << 20);
    REQUIRE(task_reservation != nullptr);
    auto* allocator =
      task_reservation
        ->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(allocator != nullptr);
    REQUIRE(allocator->attach_reservation_to_tracker(stream, std::move(task_reservation)));
    struct tracker_reset {
      cucascade::memory::reservation_aware_resource_adaptor* allocator;
      rmm::cuda_stream_view stream;
      ~tracker_reset() { allocator->reset_stream_reservation(stream); }
    } reset{allocator, stream};

    auto const last = accumulator.contribute(202, one_column_view(*final_column), stream);
    REQUIRE(last.state == sirius::op::dynamic_filter_accumulation_result::status::published);
    REQUIRE(last.exact_contribution_count == 2);
    REQUIRE(last.global_build_rows == 2 * rows_per_contribution);
    REQUIRE(last.root_device_id == final_device);
    REQUIRE(accumulator.complete());

    auto const filters = channel->filters_for_column(kProbeColumnIndex);
    REQUIRE(filters.size() == 1);
    auto const* bloom =
      dynamic_cast<sirius::op::sirius_dynamic_bloom_filter const*>(filters.front().get());
    REQUIRE(bloom != nullptr);
    REQUIRE(bloom->is_available_on_device(first_device));
    REQUIRE(bloom->is_available_on_device(final_device));
  }

  // Destroy every device allocation under its owning CUDA context.
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{final_device}};
    final_column.reset();
  }
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{first_device}};
    first_column.reset();
  }
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

TEST_CASE("null build keys are excluded from Bloom filters on both insertion paths",
          "[dynamic_filter][publisher][accumulator]")
{
  publisher_fixture fixture;

  // The data payload under a null slot is garbage by contract; give the odd (null) slots
  // recognizable poison payloads so an insertion path that stops compacting nulls has real
  // bits to leak. Valid rows carry 0, 1, 2.
  auto nullable_keys    = make_int64_values(fixture, {0, 900, 1, 901, 2, 902});
  auto const rows       = nullable_keys->view().size();
  auto const nulls      = rows / 2;
  auto const mr         = cudf::get_current_device_resource_ref();
  auto const mask_bytes = cudf::bitmask_allocation_size_bytes(rows);
  std::vector<cudf::bitmask_type> mask_words(mask_bytes / sizeof(cudf::bitmask_type), 0);
  for (cudf::size_type i = 0; i < rows; i += 2) {
    mask_words[i / 32] |= cudf::bitmask_type{1} << (i % 32);
  }
  rmm::device_buffer null_mask(mask_words.data(), mask_bytes, fixture.stream, mr);
  nullable_keys->set_null_mask(std::move(null_mask), nulls);
  fixture.stream.synchronize();

  SECTION("the keys constructor compacts nulls before insertion")
  {
    auto& source_space = fixture.replica_spaces.front().get_gpu_space();
    sirius::op::sirius_dynamic_bloom_filter filter(
      nullable_keys->view(), fixture.stream, source_space.get_default_allocator());
    fixture.stream.synchronize();

    // No false negatives: every valid build value probes as present.
    auto const valid_probe = make_int64_values(fixture, {0, 1, 2});
    REQUIRE(membership_mask(filter, valid_probe->view(), fixture) ==
            std::vector<std::uint8_t>{1, 1, 1});

    // Null probe rows are accounted as nulls, never as key values.
    auto const mask = filter.compute_mask(nullable_keys->view(), kDeviceId, fixture.stream, mr);
    REQUIRE(mask != nullptr);
    REQUIRE(mask->null_count() == nulls);
  }

  SECTION("an accumulator contribution compacts nulls before insertion")
  {
    fixture.add_key_column(3, 3);
    auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    auto plan    = make_accumulator_plan(fixture, channel);
    // Global geometry counts GPU-representation rows, nulls included, as production does.
    sirius::op::dynamic_filter_accumulator accumulator(
      plan, make_complete_build_snapshot(rows + 3, {101, 202}));

    auto const first = accumulator.contribute(101, one_column_view(*nullable_keys), fixture.stream);
    REQUIRE(first.state == sirius::op::dynamic_filter_accumulation_result::status::pending);

    auto const last =
      accumulator.contribute(202, one_column_view(*fixture.columns[0]), fixture.stream);
    REQUIRE(last.state == sirius::op::dynamic_filter_accumulation_result::status::published);
    REQUIRE(last.global_build_rows == static_cast<std::size_t>(rows) + 3);
    REQUIRE(last.publication.membership_filters_built == 1);
    REQUIRE(last.publication.filters_pushed == 1);

    auto const published = channel->filters_for_column(kProbeColumnIndex);
    REQUIRE(published.size() == 1);

    // No false negatives across both batches: the nullable batch's valid values and the
    // clean batch's values all probe as present.
    auto const probe = make_int64_values(fixture, {0, 1, 2, 3, 4, 5});
    REQUIRE(membership_mask(*published.front(), probe->view(), fixture) ==
            std::vector<std::uint8_t>{1, 1, 1, 1, 1, 1});
  }
}
