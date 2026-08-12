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
 * Unit tests for the build-port claim condition in
 * `sirius_physical_hash_join::push_data_batch_partitioned`.
 *
 * The one-shot publisher may only claim a build batch that carries the whole build side. The
 * upstream PARTITION reports that at sizing time through `set_build_arrives_whole`, and the join
 * mode is not part of the condition. These tests drive the build port directly, so they pin the
 * claim condition itself rather than any partitioning decision that leads to it.
 */

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "operator_test_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
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

[[nodiscard]] sirius::op::complete_build_snapshot make_complete_build_snapshot(
  std::uint64_t rows, std::vector<std::uint64_t> batch_ids, std::size_t partition_count = 2)
{
  auto snapshot =
    sirius::op::complete_build_snapshot::try_create(rows, std::move(batch_ids), partition_count);
  if (!snapshot) { throw std::logic_error("invalid complete build snapshot in test"); }
  return std::move(*snapshot);
}

constexpr auto kInt64 = cudf::data_type{cudf::type_id::INT64};

class controllable_source_pipeline final : public sirius::pipeline::sirius_pipeline {
 public:
  explicit controllable_source_pipeline(bool finished)
    : sirius_pipeline(sirius::pipeline::pipeline_build_context{nullptr, true}), _finished(finished)
  {
  }

  void set_finished(bool finished) noexcept { _finished = finished; }

  [[nodiscard]] bool is_pipeline_finished() const override { return _finished; }

 private:
  bool _finished;
};

class missing_lookup_repository final : public cucascade::shared_data_repository {
 public:
  [[nodiscard]] std::vector<std::uint64_t> get_batch_ids(
    std::size_t partition_idx = 0) const override
  {
    auto ids = cucascade::shared_data_repository::get_batch_ids(partition_idx);
    ids.push_back(std::numeric_limits<std::uint64_t>::max());
    return ids;
  }
};

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

  explicit claim_fixture(duckdb::JoinType join_type   = duckdb::JoinType::INNER,
                         bool enable_multi_partition  = false,
                         std::size_t build_key_domain = 0,
                         std::size_t gpu_count        = 1)
    : memory_manager(sirius::test::operator_utils::initialize_memory_manager(gpu_count))
  {
    channel->register_producer();

    gpu_space = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, kDeviceId);
    REQUIRE(gpu_space != nullptr);
    auto const host_spaces =
      memory_manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    REQUIRE_FALSE(host_spaces.empty());
    auto const local_host =
      std::find_if(host_spaces.begin(), host_spaces.end(), [this](auto const* host_space) {
        return host_space->get_device_id() == gpu_space->get_device_id();
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
                                                  .build_key_domain_cardinality = build_key_domain,
                                                  .build_key_proven_unique = build_key_domain > 0};
    std::vector<sirius::op::dynamic_filter_replica_space> replica_spaces;
    replica_spaces.emplace_back(*gpu_space, *host_space);
    auto const gpu_spaces =
      memory_manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    for (auto const* candidate : gpu_spaces) {
      if (candidate->get_device_id() == gpu_space->get_device_id()) { continue; }
      auto* candidate_gpu =
        memory_manager->get_memory_space(cucascade::memory::Tier::GPU, candidate->get_device_id());
      REQUIRE(candidate_gpu != nullptr);
      auto const candidate_host =
        std::find_if(host_spaces.begin(), host_spaces.end(), [candidate](auto const* host) {
          return host->get_device_id() == candidate->get_device_id();
        });
      auto const* staging =
        candidate_host == host_spaces.end() ? host_spaces.front() : *candidate_host;
      replica_spaces.emplace_back(*candidate_gpu, *staging);
    }
    REQUIRE(replica_spaces.size() == gpu_count);
    dynamic_filter_publish_plan plan{
      {key}, std::move(targets), std::move(replica_spaces), {.domain_coverage_threshold = 1.0}};

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
      &stats,
      enable_multi_partition);

    // This is a bare operator tree with no pipelines, so the converter's assign_operator_ids never
    // runs over it; operator code rejects the unassigned sentinel.
    std::size_t next_id    = 0;
    hash_join->operator_id = next_id++;
    for (auto& child : hash_join->children) {
      if (child) { child->operator_id = next_id++; }
    }

    // The ports the converter would normally materialize. Their repositories stay null: the base
    // push treats a null repo as "nowhere to route", which is all these tests need -- the
    // publication hook runs before routing and reads only the batch.
    for (auto const port_id : {std::string_view{"build"}, std::string_view{"default"}}) {
      auto port  = std::make_unique<sirius::op::sirius_physical_operator::port>();
      port->type = sirius::op::MemoryBarrierType::FULL;
      port->repo = nullptr;
      hash_join->add_port(port_id, std::move(port));
    }

    REQUIRE(hash_join->publishes_dynamic_filters());
    REQUIRE(stats.producers_enabled.load() == 1);
  }

  template <typename T>
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> make_typed_build_batch(cudf::type_id type_id)
  {
    std::vector<T> keys(kBuildRows);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      keys[i] = static_cast<T>(i);
    }
    return sirius::test::operator_utils::make_numeric_batch<T>(*gpu_space, keys, type_id);
  }

  [[nodiscard]] std::shared_ptr<cucascade::data_batch> make_build_batch()
  {
    return make_typed_build_batch<std::int64_t>(cudf::type_id::INT64);
  }

  [[nodiscard]] std::shared_ptr<cucascade::data_batch> make_int32_build_batch()
  {
    return make_typed_build_batch<std::int32_t>(cudf::type_id::INT32);
  }

  void push_build_batch()
  {
    hash_join->push_data_batch_partitioned("build", make_build_batch(), /*partition_idx=*/0);
  }
};

[[nodiscard]] std::unique_ptr<sirius::op::sirius_physical_partition> make_build_partition(
  claim_fixture& fixture,
  cucascade::shared_data_repository& repository,
  duckdb::shared_ptr<controllable_source_pipeline> const& source_pipeline,
  sirius::op::MemoryBarrierType barrier = sirius::op::MemoryBarrierType::FULL)
{
  auto partition = std::make_unique<sirius::op::sirius_physical_partition>(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT}),
    /*estimated_cardinality=*/1000,
    fixture.hash_join.get(),
    /*is_build=*/true);
  partition->operator_id = 100;
  partition->set_num_partitions(2);

  auto port          = std::make_unique<sirius::op::sirius_physical_operator::port>();
  port->type         = barrier;
  port->repo         = &repository;
  port->src_pipeline = source_pipeline;
  partition->add_port("default", std::move(port));
  return partition;
}

}  // namespace

TEST_CASE("hash join claims a whole build for publication in any join mode",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  // The claim condition reads `_build_arrives_whole`, not the join mode: a single-partition
  // STANDARD build publishes on the same terms as BUILD_PROBE.
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

  // Sizing from a small foldable build side selects BUILD_PROBE, the shape that published before
  // the claim condition became mode-agnostic.
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

TEST_CASE("hash join arms a multi-partition snapshot exactly once and closes it incomplete",
          "[dynamic_filter][publication_claim][accumulator]")
{
  claim_fixture fixture(duckdb::JoinType::INNER, true);
  REQUIRE(fixture.hash_join->wants_multi_partition_dynamic_filters());

  REQUIRE(fixture.hash_join->arm_multi_partition_dynamic_filters(
    make_complete_build_snapshot(2 * kBuildRows, {101, 202})));
  REQUIRE_FALSE(fixture.hash_join->arm_multi_partition_dynamic_filters(
    make_complete_build_snapshot(2 * kBuildRows, {101, 202})));
  REQUIRE(fixture.stats.publication_attempts.load() == 1);

  fixture.hash_join->finalize_operator();
  REQUIRE(fixture.stats.publications_failed.load() == 1);
  REQUIRE(fixture.stats.publications_finished.load() == 0);
  REQUIRE(fixture.channel->empty());
}

TEST_CASE("an aborted accumulator folds policy counters into hash-join stats exactly once",
          "[dynamic_filter][publication_claim][accumulator]")
{
  claim_fixture fixture(duckdb::JoinType::INNER, true);
  auto first_batch    = fixture.make_build_batch();
  auto mismatch_batch = fixture.make_int32_build_batch();
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  auto const first_id    = first_batch->get_batch_id();
  auto const mismatch_id = mismatch_batch->get_batch_id();
  REQUIRE(fixture.hash_join->arm_multi_partition_dynamic_filters(
    make_complete_build_snapshot(2 * kBuildRows, {first_id, mismatch_id})));

  auto const stream = fixture.gpu_space->acquire_stream();
  auto first_ro     = first_batch->to_read_only();
  fixture.hash_join->contribute_dynamic_filter_build_batch(
    first_id, sirius::get_cudf_table_view(first_ro), stream);
  REQUIRE(fixture.stats.keys_considered.load() == 0);
  REQUIRE(fixture.stats.keys_skipped_type_mismatch.load() == 0);

  auto mismatch_ro = mismatch_batch->to_read_only();
  fixture.hash_join->contribute_dynamic_filter_build_batch(
    mismatch_id, sirius::get_cudf_table_view(mismatch_ro), stream);

  REQUIRE(fixture.stats.keys_considered.load() == 1);
  REQUIRE(fixture.stats.keys_skipped_type_mismatch.load() == 1);
  REQUIRE(fixture.stats.membership_filters_built.load() == 0);
  REQUIRE(fixture.stats.publications_finished.load() == 0);
  REQUIRE(fixture.stats.publications_failed.load() == 1);
  REQUIRE(fixture.stats.filters_pushed.load() == 0);
  REQUIRE(fixture.channel->empty());

  fixture.hash_join->contribute_dynamic_filter_build_batch(
    mismatch_id, sirius::get_cudf_table_view(mismatch_ro), stream);
  REQUIRE(fixture.stats.keys_skipped_type_mismatch.load() == 1);
  REQUIRE(fixture.stats.publications_failed.load() == 1);
}

TEST_CASE("a source batch can become resident before claiming one-shot publication",
          "[dynamic_filter][publication_claim][source_residency][gpu_execution]")
{
  claim_fixture fixture;
  fixture.hash_join->set_build_arrives_whole(true);

  auto batch = fixture.make_build_batch();
  rmm::cuda_stream conversion_stream{rmm::cuda_stream::flags::non_blocking};
  auto& registry = sirius::converter_registry::get();
  {
    auto mutable_batch = batch->to_mutable();
    mutable_batch.convert_to<cucascade::host_data_representation>(
      registry, fixture.host_space, conversion_stream);
  }
  conversion_stream.synchronize();

  fixture.hash_join->push_data_batch_partitioned("build", batch, /*partition_idx=*/0);
  REQUIRE(fixture.stats.publications_skipped_source_not_resident.load() == 1);
  REQUIRE(fixture.stats.publication_attempts.load() == 0);
  REQUIRE(fixture.stats.publications_finished.load() == 0);
  REQUIRE(fixture.stats.publications_failed.load() == 0);
  REQUIRE(fixture.stats.publications_skipped_build_not_whole.load() == 0);
  REQUIRE(fixture.stats.filters_pushed.load() == 0);
  REQUIRE(fixture.channel->empty());

  {
    auto mutable_batch = batch->to_mutable();
    mutable_batch.convert_to<cucascade::gpu_table_representation>(
      registry, fixture.gpu_space, conversion_stream);
  }
  conversion_stream.synchronize();

  fixture.hash_join->push_data_batch_partitioned("build", batch, /*partition_idx=*/0);
  REQUIRE(fixture.stats.publications_skipped_source_not_resident.load() == 1);
  REQUIRE(fixture.stats.publication_attempts.load() == 1);
  REQUIRE(fixture.stats.publications_finished.load() == 1);
  REQUIRE(fixture.stats.publications_failed.load() == 0);
  REQUIRE(fixture.stats.publications_skipped_build_not_whole.load() == 0);
  REQUIRE(fixture.stats.filters_pushed.load() == 1);
  REQUIRE(fixture.channel->filter_count() == 1);

  fixture.hash_join->push_data_batch_partitioned("build", batch, /*partition_idx=*/0);
  REQUIRE(fixture.stats.publications_skipped_source_not_resident.load() == 1);
  REQUIRE(fixture.stats.publication_attempts.load() == 1);
  REQUIRE(fixture.stats.publications_finished.load() == 1);
  REQUIRE(fixture.stats.publications_failed.load() == 0);
  REQUIRE(fixture.stats.filters_pushed.load() == 1);
}

TEST_CASE("build PARTITION freezes exact batch identities and rows before its first pop",
          "[dynamic_filter][publication_claim][physical_partition][partition_snapshot]")
{
  claim_fixture fixture(duckdb::JoinType::INNER, true, /*build_key_domain=*/3);
  cucascade::shared_data_repository repository;
  auto source_pipeline = duckdb::make_shared_ptr<controllable_source_pipeline>(false);
  auto partition       = make_build_partition(fixture, repository, source_pipeline);

  auto first_batch = sirius::test::operator_utils::make_numeric_batch<std::int64_t>(
    *fixture.gpu_space, {0, 1, 2}, cudf::type_id::INT64);
  auto zero_row_batch = sirius::test::operator_utils::make_numeric_batch<std::int64_t>(
    *fixture.gpu_space, {}, cudf::type_id::INT64);
  auto const first_id    = first_batch->get_batch_id();
  auto const zero_row_id = zero_row_batch->get_batch_id();
  repository.add_data_batch(first_batch, 0);
  repository.add_data_batch(zero_row_batch, 0);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  REQUIRE(partition->get_next_task_input_data() == nullptr);
  REQUIRE(repository.get_batch_ids(0) == std::vector<std::uint64_t>{first_id, zero_row_id});
  REQUIRE(fixture.stats.publication_attempts.load() == 0);

  source_pipeline->set_finished(true);
  auto first_input = partition->get_next_task_input_data();
  REQUIRE(first_input != nullptr);
  REQUIRE(fixture.stats.publication_attempts.load() == 1);
  REQUIRE(repository.get_batch_ids(0) == std::vector<std::uint64_t>{zero_row_id});
  auto const first_task_id =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*first_input).task_input_batch_id();
  REQUIRE(first_task_id == first_id);

  auto const stream = fixture.gpu_space->acquire_stream();
  auto first_output = partition->execute(*first_input, stream);
  REQUIRE(first_output != nullptr);
  REQUIRE(fixture.stats.publications_finished.load() == 0);
  REQUIRE(fixture.stats.publications_failed.load() == 0);

  auto zero_row_input = partition->get_next_task_input_data();
  REQUIRE(zero_row_input != nullptr);
  REQUIRE(repository.get_batch_ids(0).empty());
  auto const zero_row_task_id =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*zero_row_input)
      .task_input_batch_id();
  REQUIRE(zero_row_task_id == zero_row_id);

  auto zero_row_output = partition->execute(*zero_row_input, stream);
  REQUIRE(zero_row_output != nullptr);
  REQUIRE(fixture.stats.publication_attempts.load() == 1);
  REQUIRE(fixture.stats.publications_finished.load() == 1);
  REQUIRE(fixture.stats.publications_failed.load() == 0);
  REQUIRE(fixture.stats.keys_considered.load() == 1);
  REQUIRE(fixture.stats.keys_with_known_domain.load() == 1);
  REQUIRE(fixture.stats.keys_skipped_domain_gate.load() == 1);
  REQUIRE(fixture.stats.keys_build_exceeded_domain.load() == 0);
  REQUIRE(fixture.stats.membership_filters_built.load() == 0);
  REQUIRE(fixture.stats.filters_pushed.load() == 0);
  REQUIRE(fixture.channel->empty());
}

TEST_CASE("build PARTITION contributes frozen identity after cross-GPU clone and retry",
          "[dynamic_filter][publication_claim][physical_partition][partition_snapshot][mgpu]")
{
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count < 2) {
    WARN("skipping: cross-GPU PARTITION identity regression requires at least two GPUs");
    return;
  }

  claim_fixture fixture(duckdb::JoinType::INNER, true, /*build_key_domain=*/3, /*gpu_count=*/2);
  cucascade::shared_data_repository repository;
  auto source_pipeline = duckdb::make_shared_ptr<controllable_source_pipeline>(true);
  auto partition       = make_build_partition(fixture, repository, source_pipeline);
  auto batch           = sirius::test::operator_utils::make_numeric_batch<std::int64_t>(
    *fixture.gpu_space, {0, 1, 2}, cudf::type_id::INT64);
  auto const frozen_id = batch->get_batch_id();
  repository.add_data_batch(batch, 0);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  auto input = partition->get_next_task_input_data();
  REQUIRE(input != nullptr);
  REQUIRE(fixture.stats.publication_attempts.load() == 1);
  auto& task_data = dynamic_cast<sirius::op::pipelineable_operator_data&>(*input);
  REQUIRE(task_data.task_input_batch_id() == frozen_id);

  auto* gpu1 = fixture.memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 1);
  REQUIRE(gpu1 != nullptr);
  rmm::cuda_set_device_raii gpu1_guard{rmm::cuda_device_id{1}};
  rmm::cuda_stream gpu1_stream{rmm::cuda_stream::flags::non_blocking};
  task_data.prepare_for_processing(gpu1, gpu1_stream.view());
  {
    auto prepared = task_data.get_read_only_batches();
    REQUIRE(prepared.size() == 1);
    REQUIRE(prepared.front().get_batch_id() != frozen_id);
    REQUIRE(prepared.front().get_memory_space()->get_device_id() == 1);
  }
  REQUIRE(task_data.task_input_batch_id() == frozen_id);

  auto output = partition->execute(*input, gpu1_stream.view());
  REQUIRE(output != nullptr);
  REQUIRE(fixture.stats.publications_finished.load() == 1);
  REQUIRE(fixture.stats.publications_failed.load() == 0);

  sirius::pipeline::oom_reschedule_exception retry_request(
    std::move(input), 0, "injected post-contribution OOM");
  auto retry_data       = retry_request.release_intermediate_data();
  auto& retry_task_data = dynamic_cast<sirius::op::pipelineable_operator_data&>(*retry_data);
  retry_task_data.remove_read_only_lock();

  sirius::pipeline::gpu_pipeline_task_local_state retry_state(std::move(retry_data), 0);
  auto& requeued_task_data =
    dynamic_cast<sirius::op::pipelineable_operator_data&>(*retry_state._input_data);
  REQUIRE(requeued_task_data.task_input_batch_id() == frozen_id);
  requeued_task_data.prepare_for_processing(gpu1, gpu1_stream.view());
  REQUIRE(requeued_task_data.task_input_batch_id() == frozen_id);
  auto retry_output = partition->execute(*retry_state._input_data, gpu1_stream.view());
  REQUIRE(retry_output != nullptr);
  REQUIRE(fixture.stats.publication_attempts.load() == 1);
  REQUIRE(fixture.stats.publications_finished.load() == 1);
  REQUIRE(fixture.stats.publications_failed.load() == 0);
  REQUIRE(fixture.channel->empty());
}

TEST_CASE("build PARTITION fails open once when an exact snapshot cannot be frozen",
          "[dynamic_filter][publication_claim][physical_partition][partition_snapshot]")
{
  SECTION("a non-GPU batch disables accumulation without blocking the partition task")
  {
    claim_fixture fixture(duckdb::JoinType::INNER, true);
    cucascade::shared_data_repository repository;
    auto source_pipeline = duckdb::make_shared_ptr<controllable_source_pipeline>(true);
    auto partition       = make_build_partition(fixture, repository, source_pipeline);

    auto host_batch = fixture.make_build_batch();
    rmm::cuda_stream conversion_stream{rmm::cuda_stream::flags::non_blocking};
    auto& registry = sirius::converter_registry::get();
    {
      auto mutable_batch = host_batch->to_mutable();
      mutable_batch.convert_to<cucascade::host_data_representation>(
        registry, fixture.host_space, conversion_stream);
    }
    conversion_stream.synchronize();
    repository.add_data_batch(host_batch, 0);

    auto input = partition->get_next_task_input_data();
    REQUIRE(input != nullptr);
    REQUIRE(repository.get_batch_ids(0).empty());
    REQUIRE(fixture.stats.publication_attempts.load() == 0);

    repository.add_data_batch(fixture.make_build_batch(), 0);
    REQUIRE(partition->get_next_task_input_data() != nullptr);
    REQUIRE(fixture.stats.publication_attempts.load() == 0);
  }

  SECTION("an enumerated batch missing at lookup disables accumulation without retry")
  {
    claim_fixture fixture(duckdb::JoinType::INNER, true);
    missing_lookup_repository repository;
    auto source_pipeline = duckdb::make_shared_ptr<controllable_source_pipeline>(true);
    auto partition       = make_build_partition(fixture, repository, source_pipeline);
    repository.add_data_batch(fixture.make_build_batch(), 0);

    auto input = partition->get_next_task_input_data();
    REQUIRE(input != nullptr);
    REQUIRE(repository.get_batch_ids(0).size() == 1);
    REQUIRE(fixture.stats.publication_attempts.load() == 0);

    repository.add_data_batch(fixture.make_build_batch(), 0);
    REQUIRE(partition->get_next_task_input_data() != nullptr);
    REQUIRE(fixture.stats.publication_attempts.load() == 0);
  }

  SECTION("a non-FULL barrier cannot certify a complete build snapshot")
  {
    claim_fixture fixture(duckdb::JoinType::INNER, true);
    cucascade::shared_data_repository repository;
    auto source_pipeline = duckdb::make_shared_ptr<controllable_source_pipeline>(true);
    auto partition       = make_build_partition(
      fixture, repository, source_pipeline, sirius::op::MemoryBarrierType::PARTIAL);
    repository.add_data_batch(fixture.make_build_batch(), 0);

    auto input = partition->get_next_task_input_data();
    REQUIRE(input != nullptr);
    REQUIRE(repository.get_batch_ids(0).empty());
    REQUIRE(fixture.stats.publication_attempts.load() == 0);

    repository.add_data_batch(fixture.make_build_batch(), 0);
    REQUIRE(partition->get_next_task_input_data() != nullptr);
    REQUIRE(fixture.stats.publication_attempts.load() == 0);
  }
}
