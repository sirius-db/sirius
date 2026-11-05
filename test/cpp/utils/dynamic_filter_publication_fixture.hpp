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

#pragma once

#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/sirius_converter_registry.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::test::publication {

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
  /// tests can build batches resident outside the replica set. @p probe_type is the channel
  /// binding's recorded probe storage type.
  explicit claim_fixture(duckdb::JoinType join_type        = duckdb::JoinType::INNER,
                         std::size_t num_gpus              = 1,
                         cudf::size_type build_key_ordinal = 0,
                         cudf::data_type probe_type        = kInt64)
    : memory_manager(sirius::test::operator_utils::initialize_memory_manager(num_gpus))
  {
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
                                                     .probe_storage_type   = probe_type}}});
    dynamic_filter_publish_plan::admitted_key key{.planner_condition_index      = 0,
                                                  .build_key_ordinal            = build_key_ordinal,
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
    duckdb::JoinCondition condition(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0),
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0),
      duckdb::ExpressionType::COMPARE_EQUAL);
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
      auto mut                = batch->to_mutable();
      auto const writer_event = mut.get_data()->get_writer_event();
      REQUIRE(writer_event != nullptr);
      RMM_CUDA_TRY(cudaStreamWaitEvent(stream.get(), writer_event, 0));
      mut.rebind_stream(stream);
      mut.convert_to<cucascade::host_data_representation>(registry, host_space, stream);
    }
    return batch;
  }
};

}  // namespace sirius::test::publication
