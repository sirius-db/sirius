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
 * Tests the build-port claim condition in `sirius_physical_hash_join::push_data_batch_partitioned`:
 * the one-shot publisher may claim only a build batch carrying the whole build side, as reported
 * through `set_build_arrives_whole`; the join mode is not part of the condition. The tests drive
 * the build port directly, pinning the claim condition -- plus the reopening of a claimed window
 * without a usable GPU source (not GPU-resident, or resident on a GPU outside the plan's replica
 * set) and the fail-open containment of device memory exhaustion -- rather than any partitioning
 * decision.
 */

#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "operator_test_utils.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/sirius_converter_registry.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

using sirius::op::dynamic_filter_publish_plan;
using sirius::op::dynamic_filter_route_class;
using sirius::op::dynamic_filter_stats;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_physical_hash_join;

constexpr int kDeviceId                 = 0;
constexpr std::size_t kProbeColumnIndex = 3;
constexpr std::size_t kBuildRows        = 64;

constexpr auto kInt64 = cudf::data_type{cudf::type_id::INT64};

/// A hash join wired to one scan-route channel, plus the GPU resources its build batches live in.
/// The logical join is declared before the physical one because the physical join holds `op.types`
/// by reference.
struct claim_fixture {
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{kDeviceId}};
  decltype(sirius::test::operator_utils::initialize_memory_manager(1)) memory_manager;
  std::shared_ptr<sirius_dynamic_filter_set> channel =
    std::make_shared<sirius_dynamic_filter_set>();
  dynamic_filter_stats stats;
  cucascade::memory::memory_space* gpu_space        = nullptr;
  cucascade::memory::memory_space const* host_space = nullptr;
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> hash_join;

  /// The plan's replica space stays GPU 0 only regardless of @p num_gpus; extra GPUs exist so
  /// tests can build batches resident outside the replica set.
  explicit claim_fixture(duckdb::JoinType join_type = duckdb::JoinType::INNER,
                         std::size_t num_gpus       = 1)
    : memory_manager(sirius::test::operator_utils::initialize_memory_manager(num_gpus))
  {
    channel->register_producer({kProbeColumnIndex});

    gpu_space = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, kDeviceId);
    REQUIRE(gpu_space != nullptr);
    auto const host_spaces =
      memory_manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    REQUIRE_FALSE(host_spaces.empty());
    auto const local_host =
      std::find_if(host_spaces.begin(), host_spaces.end(), [this](auto const* candidate) {
        return candidate->get_device_id() == gpu_space->get_device_id();
      });
    host_space = local_host == host_spaces.end() ? host_spaces.front() : *local_host;

    std::vector<dynamic_filter_publish_plan::probe_target> targets;
    targets.push_back({.filter_set               = channel,
                       .route_class              = dynamic_filter_route_class::scan,
                       .accepts_zone_map_filters = false,
                       .key_bindings             = {{.admitted_key_index   = 0,
                                                     .channel_push_ordinal = kProbeColumnIndex,
                                                     .probe_storage_type   = kInt64}}});
    dynamic_filter_publish_plan::admitted_key key{.planner_condition_index      = 0,
                                                  .build_key_ordinal            = 0,
                                                  .storage_type                 = kInt64,
                                                  .key_shape                    = {},
                                                  .build_key_domain_cardinality = 0,
                                                  .build_key_proven_unique      = false};
    std::vector<sirius::op::dynamic_filter_replica_space> replica_spaces{{*gpu_space, *host_space}};
    dynamic_filter_publish_plan plan{{key}, std::move(targets), std::move(replica_spaces)};

    duckdb::vector<duckdb::LogicalType> const output_types{duckdb::LogicalType::BIGINT};
    logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
    logical_join->types = output_types;

    auto left_child = duckdb::make_uniq<sirius::op::sirius_physical_operator>(
      sirius::op::SiriusPhysicalOperatorType::PROJECTION, sirius::from_duckdb_vec(output_types), 0);
    auto right_child = duckdb::make_uniq<sirius::op::sirius_physical_operator>(
      sirius::op::SiriusPhysicalOperatorType::PROJECTION, sirius::from_duckdb_vec(output_types), 0);

    duckdb::vector<duckdb::JoinCondition> conditions;
    duckdb::JoinCondition condition;
    condition.left =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0);
    condition.right =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0);
    condition.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
    conditions.push_back(std::move(condition));

    hash_join = duckdb::make_uniq<sirius_physical_hash_join>(
      *logical_join,
      std::move(left_child),
      std::move(right_child),
      sirius::wrap_join_conditions(std::move(conditions)),
      join_type,
      duckdb::vector<duckdb::idx_t>{},
      duckdb::vector<duckdb::idx_t>{},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
      /*estimated_cardinality=*/1000,
      sirius::config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES,
      std::move(plan),
      sirius::config::DEFAULT_HASH_PARTITION_BYTES,
      sirius::config::DEFAULT_MAX_BROADCAST_JOIN_SIZE,
      &stats);

    // This is a bare operator tree with no pipelines, so the converter's assign_operator_ids never
    // runs over it; operator code rejects the unassigned sentinel.
    std::size_t next_id    = 0;
    hash_join->operator_id = next_id++;
    for (auto& child : hash_join->children) {
      if (child) { child->operator_id = next_id++; }
    }

    // Ports the converter would normally materialize. Repositories stay null ("nowhere to
    // route"); publication claims and reads only the batch, independent of routing.
    for (auto const port_id : {std::string_view{"build"}, std::string_view{"default"}}) {
      auto port  = std::make_unique<sirius::op::sirius_physical_operator::port>();
      port->type = sirius::op::MemoryBarrierType::FULL;
      port->repo = nullptr;
      hash_join->add_port(port_id, std::move(port));
    }

    REQUIRE(hash_join->publishes_dynamic_filters());
    REQUIRE(stats.producers_enabled.load() == 1);
  }

  /// A GPU-resident single-column build batch of @p rows distinct INT64 keys, created in
  /// @p space on that space's device.
  [[nodiscard]] static std::shared_ptr<cucascade::data_batch> make_build_batch_on(
    cucascade::memory::memory_space& space, std::size_t rows = kBuildRows)
  {
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{space.get_device_id()}};
    std::vector<std::int64_t> keys(rows);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      keys[i] = static_cast<std::int64_t>(i);
    }
    return sirius::test::operator_utils::make_numeric_batch<std::int64_t>(
      space, keys, cudf::type_id::INT64);
  }

  /// The default whole-build batch: kBuildRows keys in the fixture's GPU-0 space.
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> make_build_batch()
  {
    return make_build_batch_on(*gpu_space);
  }

  void push_build_batch()
  {
    hash_join->push_data_batch_partitioned("build", make_build_batch(), /*partition_idx=*/0);
  }

  /// The same whole-build batch, converted in place to the host tier before delivery (the shape of
  /// a batch downgraded ahead of the publish hook).
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> make_host_build_batch()
  {
    auto batch     = make_build_batch();
    auto& registry = sirius::converter_registry::get();
    // The converter's batched copy path (cudaMemcpyBatchAsync) rejects the default stream.
    auto const stream = gpu_space->acquire_stream();
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::host_data_representation>(registry, host_space, stream);
    }
    return batch;
  }
};

}  // namespace

TEST_CASE("hash join claims a whole build for publication in any join mode",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  claim_fixture fixture;
  REQUIRE_FALSE(fixture.hash_join->is_build_probe_mode());

  fixture.hash_join->set_build_arrives_whole(true);
  fixture.push_build_batch();

  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 1);
  CHECK(fixture.stats.publications_skipped_build_not_whole.load() == 0);
  CHECK_FALSE(fixture.channel->filters_for_column(kProbeColumnIndex).empty());
}

TEST_CASE("hash join in BUILD_PROBE mode still publishes from a whole build",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  claim_fixture fixture;

  // Sizing from a small foldable build side selects BUILD_PROBE.
  auto const strategy =
    fixture.hash_join->get_partition_strategy(sirius::op::partition_sizing_input{
      .total_bytes = 1024, .is_build_side = true, .build_foldable = true});
  REQUIRE(strategy.build_probe);
  REQUIRE(fixture.hash_join->is_build_probe_mode());

  fixture.hash_join->set_build_arrives_whole(true);
  fixture.push_build_batch();

  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 1);
  CHECK_FALSE(fixture.channel->filters_for_column(kProbeColumnIndex).empty());
}

TEST_CASE("a wired join whose build is not whole reports the skip exactly once",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  // `set_build_arrives_whole` is never called, so the window can never claim. The condition is
  // fixed before the first delivery, so every later build batch would report the same fact.
  claim_fixture fixture;

  fixture.push_build_batch();
  CHECK(fixture.stats.publications_skipped_build_not_whole.load() == 1);

  fixture.push_build_batch();
  CHECK(fixture.stats.publications_skipped_build_not_whole.load() == 1);

  CHECK(fixture.stats.publication_attempts.load() == 0);
  CHECK(fixture.channel->filters_for_column(kProbeColumnIndex).empty());
}

TEST_CASE("a claimed but not GPU-resident whole build reopens the window for a sibling delivery",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  // On a multi-GPU broadcast build every GPU delivers the whole build side, so a first delivery
  // that was already downgraded to the host tier must not end the window terminally: a sibling
  // delivery with a GPU-resident replica can still claim and publish.
  claim_fixture fixture;
  fixture.hash_join->set_build_arrives_whole(true);

  fixture.hash_join->push_data_batch_partitioned(
    "build", fixture.make_host_build_batch(), /*partition_idx=*/0);

  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_skipped_source_not_resident.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 0);
  CHECK(fixture.stats.publications_failed.load() == 0);
  CHECK(fixture.channel->filters_for_column(kProbeColumnIndex).empty());

  fixture.push_build_batch();

  // The full accounting identity after the rescue: attempts == finished + failed + skipped.
  CHECK(fixture.stats.publication_attempts.load() == 2);
  CHECK(fixture.stats.publications_finished.load() == 1);
  CHECK(fixture.stats.publications_failed.load() == 0);
  CHECK(fixture.stats.publications_skipped_source_not_resident.load() == 1);
  CHECK_FALSE(fixture.channel->filters_for_column(kProbeColumnIndex).empty());
}

TEST_CASE("a join whose replica restriction removed every GPU never claims",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  claim_fixture fixture;
  // The pipeline converter's per-query GPU restriction admitted none of the plan's replica
  // devices, disabling publication for this join before execution.
  fixture.hash_join->restrict_dynamic_filter_replicas({kDeviceId + 1});
  CHECK_FALSE(fixture.hash_join->publishes_dynamic_filters());

  fixture.hash_join->set_build_arrives_whole(true);
  CHECK_NOTHROW(fixture.push_build_batch());

  CHECK(fixture.stats.publication_attempts.load() == 0);
  CHECK(fixture.stats.publications_skipped_build_not_whole.load() == 0);
  CHECK(fixture.channel->filters_for_column(kProbeColumnIndex).empty());
}

TEST_CASE("device memory exhaustion during a claimed publication fails open",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  claim_fixture fixture;
  fixture.hash_join->set_build_arrives_whole(true);

  // Past the small-list gate, so filter construction must allocate device memory.
  constexpr std::size_t kLargeBuildRows = 100000;
  auto batch = claim_fixture::make_build_batch_on(*fixture.gpu_space, kLargeBuildRows);

  {
    // Leave under 64 KiB of the space's accounting capacity so the filter's first device
    // allocation deterministically throws cucascade_out_of_memory (an rmm::out_of_memory).
    constexpr std::size_t kHeadroomBytes = 64ull << 10;
    auto const stream                    = fixture.gpu_space->acquire_stream();
    auto const available                 = fixture.gpu_space->get_available_memory();
    REQUIRE(available > kHeadroomBytes);
    rmm::device_buffer const ballast{
      available - kHeadroomBytes, stream, fixture.gpu_space->get_default_allocator()};

    CHECK_NOTHROW(
      fixture.hash_join->push_data_batch_partitioned("build", batch, /*partition_idx=*/0));
  }

  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_failed.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 0);
  CHECK(fixture.stats.membership_filters_built.load() == 0);
  CHECK(fixture.channel->filters_for_column(kProbeColumnIndex).empty());

  // FAILED is terminal: with the ballast freed, a second whole-build delivery must not reattempt.
  fixture.push_build_batch();
  CHECK(fixture.stats.publication_attempts.load() == 1);
}

TEST_CASE("a whole build resident on a non-plan GPU reopens the window for a sibling delivery",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN(
      "non-plan-GPU source skip requires >=2 GPUs; single-GPU host -- skipping "
      "(per Catch2 v2 WARN+return convention)");
    return;
  }

  // Two GPU spaces exist, but the plan holds a replica space on GPU 0 only.
  claim_fixture fixture{duckdb::JoinType::INNER, /*num_gpus=*/2};
  fixture.hash_join->set_build_arrives_whole(true);

  auto* other_gpu_space =
    fixture.memory_manager->get_memory_space(cucascade::memory::Tier::GPU, kDeviceId + 1);
  REQUIRE(other_gpu_space != nullptr);

  fixture.hash_join->push_data_batch_partitioned(
    "build", claim_fixture::make_build_batch_on(*other_gpu_space), /*partition_idx=*/0);

  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_skipped_source_not_resident.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 0);
  CHECK(fixture.stats.publications_failed.load() == 0);
  CHECK(fixture.channel->filters_for_column(kProbeColumnIndex).empty());

  fixture.push_build_batch();

  CHECK(fixture.stats.publication_attempts.load() == 2);
  CHECK(fixture.stats.publications_finished.load() == 1);
  CHECK_FALSE(fixture.channel->filters_for_column(kProbeColumnIndex).empty());
}
