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

// A closed zero-batch probe reclaims unused builds without scheduling GPU work.

#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator_test_utils.hpp"
#include "utils/data_utils.hpp"
#include "utils/scoped_temp_directory.hpp"

#include <catch.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <helper/type_conversions.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <pipeline/repository_wiring.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

using namespace sirius::op;
using namespace cucascade::memory;
using sirius::pipeline::materialize_repository_wiring;
using sirius::pipeline::repository_wiring;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;
using sirius::test::operator_utils::make_numeric_batch;

namespace {

// Model measured INTEGER build metadata; the execution fixtures keep their actual batches tiny.
partition_sizing_input measured_build_sizing(uint64_t bytes)
{
  auto const rows = bytes / sizeof(int32_t);
  return {.total_bytes          = bytes,
          .is_build_side        = true,
          .build_foldable       = true,
          .combined_total_bytes = bytes,
          .total_rows           = rows,
          .measured_rows        = rows,
          .resident_bytes       = bytes};
}

memory_space* get_shared_mem_space()
{
  static auto manager = sirius::test::operator_utils::initialize_memory_manager();
  return manager->get_memory_space(Tier::GPU, 0);
}

/// Pipeline stub whose finished-ness the test flips explicitly, standing in for a real upstream
/// pipeline that has (or has not) delivered its last batch.
class finished_stub_pipeline : public sirius_pipeline {
 public:
  using sirius_pipeline::sirius_pipeline;
  bool is_pipeline_finished() const override { return finished; }

  bool finished = false;
};

/**
 * @brief A hash join with fully wired build/probe ports on stub source pipelines.
 *
 * Left/probe child has types {INTEGER, INTEGER} (key + payload), right/build child has {INTEGER}
 * (key only); the single condition is left col[0] = right col[0]. `probe_op` is the probe source
 * pipeline's first operator — the producer a waiting hint must name.
 */
struct reclaim_join_harness {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> hash_join;

  sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  cucascade::shared_data_repository_manager repo_manager;
  std::vector<std::shared_ptr<sirius_pipeline>> pipelines;
  std::shared_ptr<finished_stub_pipeline> probe_src_pipeline;
  std::shared_ptr<finished_stub_pipeline> build_src_pipeline;
  std::shared_ptr<sirius_pipeline> dest_pipeline;
  sirius_physical_operator probe_sink, probe_op, build_sink, build_op, dest_sink;
};

std::unique_ptr<reclaim_join_harness> make_reclaim_join_harness(duckdb::JoinType join_type)
{
  auto h = std::make_unique<reclaim_join_harness>();

  h->logical_join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  if (join_type == duckdb::JoinType::MARK) {
    h->logical_join->types = {
      duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER, duckdb::LogicalType::BOOLEAN};
  } else if (join_type == duckdb::JoinType::SEMI || join_type == duckdb::JoinType::ANTI) {
    h->logical_join->types = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER};
  } else {
    h->logical_join->types = {
      duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER};
  }

  auto left_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER,
                                                                duckdb::LogicalType::INTEGER}),
    0);
  auto right_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0);

  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.emplace_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
    duckdb::ExpressionType::COMPARE_EQUAL);

  h->hash_join = duckdb::make_uniq<sirius_physical_hash_join>(
    *h->logical_join,
    std::move(left_child),
    std::move(right_child),
    sirius::wrap_join_conditions(std::move(conditions)),
    join_type,
    duckdb::vector<duckdb::idx_t>{},  // left_projection_map (empty = all)
    duckdb::vector<duckdb::idx_t>{},  // right_projection_map
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),  // delim_types
    1000);
  h->hash_join->set_active_gpu_ids({0});

  // Source pipelines with one operator each (the producer a waiting hint names), and a
  // destination pipeline whose first operator is the join, so materialize_repository_wiring
  // attaches the "default"/"build" ports (repository + src_pipeline) onto it.
  auto build_stub = [&](sirius_physical_operator* sink,
                        const std::vector<sirius_physical_operator*>& operators,
                        std::size_t pipeline_id) {
    auto pipeline = std::make_shared<finished_stub_pipeline>(h->build_ctx);
    sirius_pipeline_build_state build_state;
    build_state.set_pipeline_sink(*pipeline, sink);
    for (auto* op : operators) {
      build_state.add_pipeline_operator(*pipeline, *op);
    }
    pipeline->set_pipeline_id(pipeline_id);
    h->pipelines.push_back(pipeline);
    return pipeline;
  };
  h->probe_src_pipeline = build_stub(&h->probe_sink, {&h->probe_op}, 0);
  h->build_src_pipeline = build_stub(&h->build_sink, {&h->build_op}, 1);
  {
    auto pipeline = std::make_shared<sirius_pipeline>(h->build_ctx);
    sirius_pipeline_build_state build_state;
    build_state.set_pipeline_sink(*pipeline, &h->dest_sink);
    build_state.add_pipeline_operator(*pipeline, *h->hash_join);
    pipeline->set_pipeline_id(2);
    h->pipelines.push_back(pipeline);
    h->dest_pipeline = pipeline;
  }

  std::vector<repository_wiring> wirings = {
    {std::string_view{"default"},
     MemoryBarrierType::PARTIAL,
     &h->probe_sink,
     h->probe_src_pipeline,
     h->dest_pipeline},
    {std::string_view{"build"},
     MemoryBarrierType::FULL,
     &h->build_sink,
     h->build_src_pipeline,
     h->dest_pipeline},
  };
  sirius::pipeline::assign_operator_ids(h->pipelines);
  materialize_repository_wiring(wirings, h->repo_manager);

  return h;
}

void expect_reclaim_flow(
  reclaim_join_harness& h,
  std::optional<std::weak_ptr<cucascade::data_batch>> released_build = std::nullopt)
{
  CHECK_FALSE(h.hash_join->get_next_task_hint().has_value());
  CHECK(h.hash_join->get_next_task_input_data() == nullptr);
  CHECK(h.hash_join->all_ports_empty());
  if (released_build) { CHECK(released_build->expired()); }
  h.dest_pipeline->update_pipeline_status(false);
  CHECK(h.dest_pipeline->is_pipeline_finished());
  CHECK_FALSE(h.hash_join->get_next_task_hint().has_value());
}

}  // namespace

TEST_CASE("hash_join BUILD_PROBE reclaim - broadcast join with a zero-batch probe side",
          "[hash_join][build_probe_reclaim]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto h = make_reclaim_join_harness(duckdb::JoinType::INNER);

  // A small build on one GPU sizes as a single-partition broadcast BUILD_PROBE join.
  auto const strategy = h->hash_join->get_partition_strategy(measured_build_sizing(1024));
  REQUIRE(strategy.build_probe);
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  REQUIRE(strategy.broadcast);
  REQUIRE(strategy.num_partitions == 1);

  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  expect_reclaim_flow(*h);
}

TEST_CASE("hash_join BUILD_PROBE reclaim - non-broadcast join with a zero-batch probe side",
          "[hash_join][build_probe_reclaim]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto h = make_reclaim_join_harness(duckdb::JoinType::INNER);

  // A build over the small-table threshold (16 MB/GPU) but under the hash-table budget sizes as
  // a single-partition non-broadcast BUILD_PROBE join.
  auto const strategy =
    h->hash_join->get_partition_strategy(measured_build_sizing(32ull * 1024 * 1024));
  REQUIRE(strategy.build_probe);
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  REQUIRE_FALSE(strategy.broadcast);
  REQUIRE(strategy.num_partitions == 1);

  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  expect_reclaim_flow(*h);
}

TEST_CASE("hash_join BUILD_PROBE reclaim - all never-probed slots are discarded",
          "[hash_join][build_probe_reclaim]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto h = make_reclaim_join_harness(duckdb::JoinType::INNER);

  // Two GPUs and a small build: broadcast BUILD_PROBE with one replicated slot per GPU. With the
  // probe side finished and empty, both replicated build slots are discarded.
  h->hash_join->set_active_gpu_ids({0, 7});
  auto const strategy = h->hash_join->get_partition_strategy(measured_build_sizing(1024));
  REQUIRE(strategy.build_probe);
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  REQUIRE(strategy.broadcast);
  REQUIRE(strategy.num_partitions == 2);

  for (std::size_t p = 0; p < 2; ++p) {
    h->hash_join->push_data_batch_partitioned(
      "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), p);
  }
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  expect_reclaim_flow(*h);
  // Both replicated build batches were freed by the reclaim sweep.
  CHECK(h->hash_join->get_port("build")->repo->size(1) == 0);
}

TEST_CASE("hash_join BUILD_PROBE reclaim - MARK join with a zero-batch probe side",
          "[hash_join][build_probe_reclaim]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto h = make_reclaim_join_harness(duckdb::JoinType::MARK);

  // A small foldable MARK build uses one BUILD_PROBE partition on a single GPU.
  auto const strategy = h->hash_join->get_partition_strategy(measured_build_sizing(1024));
  REQUIRE(strategy.build_probe);
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  REQUIRE(strategy.num_partitions == 1);

  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  expect_reclaim_flow(*h);
}

TEST_CASE("hash_join BUILD_PROBE reclaim safely reclaims HOST and DISK builds without upgrading",
          "[hash_join][build_probe_reclaim]")
{
  auto const tier = GENERATE(Tier::HOST, Tier::DISK);
  sirius::test::scoped_temp_directory temporary;
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(std::size_t{256} << 20)
    .set_per_numa_region_capacity(std::size_t{64} << 20)
    .use_gpu_id_as_host_id()
    .set_disk_mounting_point(0, std::size_t{64} << 20, temporary.directory);
  sirius::memory::sirius_memory_reservation_manager manager(builder.build());
  sirius::test::operator_utils::ensure_converter_registry();
  auto* gpu  = manager.get_memory_space(Tier::GPU, 0);
  auto* host = manager.get_memory_space(Tier::HOST, 0);
  REQUIRE(gpu != nullptr);
  REQUIRE(host != nullptr);
  auto batch        = make_numeric_batch<int32_t>(*gpu, {1, 2, 3}, cudf::type_id::INT32);
  auto& registry    = sirius::converter_registry::get();
  auto const stream = gpu->acquire_stream();
  sirius::test::operator_utils::default_stream().sync();
  {
    auto mutable_batch = batch->to_mutable();
    mutable_batch.convert_to<cucascade::host_data_representation>(registry, host, stream);
    if (tier == Tier::DISK) {
      auto* disk = manager.get_memory_space(Tier::DISK, 0);
      REQUIRE(disk != nullptr);
      mutable_batch.convert_to<cucascade::disk_data_representation>(registry, disk, stream);
    }
  }
  REQUIRE(batch->to_read_only().get_current_tier() == tier);
  std::weak_ptr<cucascade::data_batch> lifetime = batch;
  auto h              = make_reclaim_join_harness(duckdb::JoinType::INNER);
  auto const strategy = h->hash_join->get_partition_strategy(measured_build_sizing(32ULL << 20));
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  h->hash_join->push_data_batch_partitioned("build", std::move(batch), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;
  expect_reclaim_flow(*h, lifetime);
  CHECK(lifetime.expired());
}
