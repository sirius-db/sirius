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

// Deterministic zero-batch probe coverage: reclaim unused builds, prepare one metadata-only task
// on its placed GPU, and emit an empty result that lets downstream COUNT return zero.

#include "data/sirius_converter_registry.hpp"
#include "expression/ast/node.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "operator_test_utils.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "utils/data_utils.hpp"
#include "utils/scoped_temp_directory.hpp"
#include "utils/telemetry_utils.hpp"

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

#include <chrono>
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
struct orphan_join_harness {
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

std::unique_ptr<orphan_join_harness> make_orphan_join_harness(duckdb::JoinType join_type)
{
  auto h = std::make_unique<orphan_join_harness>();

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
  duckdb::JoinCondition cond;
  cond.left  = duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.right = duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  conditions.push_back(std::move(cond));

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
    build_state.set_pipeline_sink(*pipeline, sink, /*pipeline_idx=*/0);
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
    build_state.set_pipeline_sink(*pipeline, &h->dest_sink, /*pipeline_idx=*/0);
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

std::unique_ptr<operator_data> expect_orphan_flow(
  orphan_join_harness& h,
  cudf::size_type expected_output_columns,
  memory_space* target                                               = get_shared_mem_space(),
  std::optional<std::weak_ptr<cucascade::data_batch>> released_build = std::nullopt)
{
  auto hint = h.hash_join->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == h.hash_join.get());

  auto input = h.hash_join->get_next_task_input_data();
  REQUIRE(input != nullptr);
  CHECK(dynamic_cast<pipelineable_operator_data*>(input.get()) == nullptr);
  CHECK(input->get_preferred_device_id() == 0);
  CHECK(input->get_estimated_size_in_bytes() >= (std::size_t{1} << 20));
  REQUIRE(h.hash_join->get_next_task_input_data() == nullptr);
  CHECK(h.hash_join->all_ports_empty());
  if (released_build) { CHECK(released_build->expired()); }

  // A failed preparation cannot retire the slot or schedule a second orphan.
  input->prepare_for_processing(nullptr, sirius::test::operator_utils::default_stream());
  REQUIRE_THROWS_AS(h.hash_join->execute(*input, sirius::test::operator_utils::default_stream()),
                    std::runtime_error);
  auto waiting = h.hash_join->get_next_task_hint();
  REQUIRE(waiting.has_value());
  CHECK(waiting->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  input->prepare_for_processing(target, sirius::test::operator_utils::default_stream());
  auto output = h.hash_join->execute(*input, sirius::test::operator_utils::default_stream());
  REQUIRE(output != nullptr);
  auto const& output_data = dynamic_cast<const pipelineable_operator_data&>(*output);
  REQUIRE(output_data.get_data_batches().size() == 1);
  auto out_view = sirius::get_cudf_table_view(*output_data.get_data_batches()[0]);
  CHECK(out_view.num_rows() == 0);
  CHECK(out_view.num_columns() == expected_output_columns);
  CHECK_FALSE(h.hash_join->get_next_task_hint().has_value());
  return output;
}

}  // namespace

TEST_CASE("hash_join BUILD_PROBE orphan - broadcast join with a zero-batch probe side",
          "[hash_join][build_probe_orphan]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto h = make_orphan_join_harness(duckdb::JoinType::INNER);

  // A small build on one GPU sizes as a single-partition broadcast BUILD_PROBE join.
  auto const strategy = h->hash_join->get_partition_strategy(
    {/*total_bytes=*/1024, /*is_build_side=*/true, /*build_foldable=*/true});
  REQUIRE(strategy.build_probe);
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  REQUIRE(strategy.broadcast);
  REQUIRE(strategy.num_partitions == 1);

  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  (void)expect_orphan_flow(*h, /*expected_output_columns=*/3);  // left key + payload + right key
}

TEST_CASE("hash_join BUILD_PROBE orphan - non-broadcast join with a zero-batch probe side",
          "[hash_join][build_probe_orphan]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto h = make_orphan_join_harness(duckdb::JoinType::INNER);

  // A build over the small-table threshold (16 MB/GPU) but under the hash-table budget sizes as
  // a single-partition non-broadcast BUILD_PROBE join.
  auto const strategy = h->hash_join->get_partition_strategy(
    {/*total_bytes=*/32ull * 1024 * 1024, /*is_build_side=*/true, /*build_foldable=*/true});
  REQUIRE(strategy.build_probe);
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  REQUIRE_FALSE(strategy.broadcast);
  REQUIRE(strategy.num_partitions == 1);

  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  (void)expect_orphan_flow(*h, /*expected_output_columns=*/3);
}

TEST_CASE("hash_join BUILD_PROBE orphan - one slot is kept, the other never-probed slot discarded",
          "[hash_join][build_probe_orphan]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto h = make_orphan_join_harness(duckdb::JoinType::INNER);

  // Two GPUs and a small build: broadcast BUILD_PROBE with one replicated slot per GPU. With the
  // probe side finished and empty, the reclaim keeps slot 0 as the orphan and discards slot 1's
  // replica outright.
  h->hash_join->set_active_gpu_ids({0, 7});
  auto const strategy = h->hash_join->get_partition_strategy(
    {/*total_bytes=*/1024, /*is_build_side=*/true, /*build_foldable=*/true});
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

  (void)expect_orphan_flow(*h, /*expected_output_columns=*/3);
  // Slot 1's replicated build batch was freed by the reclaim sweep.
  CHECK(h->hash_join->get_port("build")->repo->size(1) == 0);
}

TEST_CASE("hash_join BUILD_PROBE orphan - MARK join with a zero-batch probe side",
          "[hash_join][build_probe_orphan]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);
  auto h = make_orphan_join_harness(duckdb::JoinType::MARK);

  // MARK is forced into BUILD_PROBE at sizing; single GPU clamps it to one partition.
  auto const strategy = h->hash_join->get_partition_strategy(
    {/*total_bytes=*/1024, /*is_build_side=*/true, /*build_foldable=*/true});
  REQUIRE(strategy.build_probe);
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  REQUIRE(strategy.num_partitions == 1);

  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  (void)expect_orphan_flow(*h, /*expected_output_columns=*/3);  // left key + payload + BOOL8 mark
}

TEST_CASE("hash_join BUILD_PROBE orphan emits COUNT zero with physical carriers",
          "[hash_join][build_probe_orphan]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);
  auto const kind =
    GENERATE(duckdb::JoinType::LEFT, duckdb::JoinType::SEMI, duckdb::JoinType::ANTI);
  auto h              = make_orphan_join_harness(kind);
  auto const strategy = h->hash_join->get_partition_strategy({1024, true, true});
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  auto const columns = static_cast<cudf::size_type>(h->hash_join->get_types().size());
  h->hash_join->set_physical_types(std::vector<cudf::data_type>(
    static_cast<std::size_t>(columns), cudf::data_type{cudf::type_id::INT8}));
  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  auto output       = expect_orphan_flow(*h, columns);
  auto const& batch = dynamic_cast<pipelineable_operator_data&>(*output).get_data_batches().front();
  for (auto const& column : sirius::get_cudf_table_view(*batch)) {
    CHECK(column.type() == cudf::data_type{cudf::type_id::INT8});
  }
  auto const count_type = sirius::logical_type::make(sirius::type_id::BIGINT);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;
  aggregates.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::aggregate{sirius::aggregate_id::count_star, {}, count_type, false}));
  sirius_physical_ungrouped_aggregate count(
    {count_type}, std::move(aggregates), 0, duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
  auto counted = count.execute(*output, sirius::test::operator_utils::default_stream());
  auto const& counted_batch =
    dynamic_cast<pipelineable_operator_data&>(*counted).get_data_batches().front();
  auto const view = sirius::get_cudf_table_view(*counted_batch);
  REQUIRE(view.num_rows() == 1);
  CHECK(sirius::test::operator_utils::copy_column_to_host<int64_t>(view.column(0)) ==
        std::vector<int64_t>{0});
}

TEST_CASE("hash_join BUILD_PROBE orphan safely reclaims HOST and DISK builds without upgrading",
          "[hash_join][build_probe_orphan]")
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
  auto h                                        = make_orphan_join_harness(duckdb::JoinType::INNER);
  auto const strategy = h->hash_join->get_partition_strategy({32ULL << 20, true, true});
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  h->hash_join->push_data_batch_partitioned("build", std::move(batch), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;
  (void)expect_orphan_flow(*h, 3, gpu, lifetime);
  CHECK(lifetime.expired());
}

TEST_CASE("hash_join BUILD_PROBE orphan task retries OOM and publishes COUNT zero before closure",
          "[hash_join][build_probe_orphan][gpu_pipeline_task]")
{
  auto const retry = GENERATE(false, true);
  auto manager     = sirius::test::operator_utils::initialize_memory_manager();
  auto* space      = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto h              = make_orphan_join_harness(duckdb::JoinType::INNER);
  auto const strategy = h->hash_join->get_partition_strategy({32ULL << 20, true, true});
  h->hash_join->set_placement(std::make_shared<const partition_placement>(strategy.placement));
  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  auto const count_type = sirius::logical_type::make(sirius::type_id::BIGINT);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;
  aggregates.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::aggregate{sirius::aggregate_id::count_star, {}, count_type, false}));
  sirius_physical_ungrouped_aggregate count(
    {count_type}, std::move(aggregates), 0, duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
  class capture_count final : public sirius_physical_operator {
   public:
    capture_count()
      : sirius_physical_operator(SiriusPhysicalOperatorType::RESULT_COLLECTOR,
                                 {sirius::logical_type::make(sirius::type_id::BIGINT)},
                                 1)
    {
    }
    std::unique_ptr<operator_data> execute(operator_data const& data, ::cuda::stream_ref) override
    {
      return std::make_unique<pipelineable_operator_data>(
        dynamic_cast<pipelineable_operator_data const&>(data).get_data_batches());
    }
    void sink(operator_data const& data, ::cuda::stream_ref) override
    {
      batches = dynamic_cast<pipelineable_operator_data const&>(data).get_data_batches();
    }
    std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  } collector;
  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*h->dest_pipeline, *h->hash_join);
  build_state.set_pipeline_operators(*h->dest_pipeline, {*h->hash_join, count, collector});
  build_state.set_pipeline_sink(*h->dest_pipeline, &collector, 0);
  sirius::pipeline::assign_operator_ids(h->pipelines);
  auto completion = std::make_shared<sirius::pipeline::completion_handler>();
  if (retry) {
    auto injection          = std::make_shared<sirius::op::scan::test_injections>();
    injection->gpu_task_oom = 1;
    completion->injections  = std::move(injection);
  }
  auto completed = completion->get_awaitable();
  h->dest_pipeline->set_completion_handler(completion);
  auto global = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
    h->dest_pipeline, sirius::test::make_test_telemetry_context());
  global->set_completion_handler(completion);
  auto const make_task = [&](uint64_t id, std::unique_ptr<operator_data> input) {
    auto local =
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(input));
    local->set_preferred_device_id(0);
    auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      id, std::vector<cucascade::shared_data_repository*>{}, std::move(local), global);
    auto const estimate = task->get_estimated_reservation_size_info(space);
    CHECK(estimate.bytes_to_materialize_input == 0);
    auto reservation = manager->request_reservation(
      cucascade::memory::any_memory_space_in_tier{Tier::GPU}, estimate.reservation_size);
    REQUIRE(reservation != nullptr);
    dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(task->local_state())
      ->set_reservation(std::move(reservation), estimate);
    return task;
  };
  auto hint = h->hash_join->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  auto task = make_task(1, h->hash_join->get_next_task_input_data());
  CHECK(h->hash_join->all_ports_empty());
  CHECK_FALSE(h->dest_pipeline->is_pipeline_finished());
  auto const stream = space->acquire_stream();
  if (retry) {
    std::unique_ptr<operator_data> retry_input;
    try {
      task->execute(stream);
      FAIL("the first task must encounter the injected OOM");
    } catch (sirius::pipeline::oom_reschedule_exception& error) {
      CHECK(error.get_resume_operator_index() == 0);
      retry_input = error.release_intermediate_data();
    }
    REQUIRE(retry_input != nullptr);
    CHECK(dynamic_cast<pipelineable_operator_data*>(retry_input.get()) == nullptr);
    CHECK(retry_input->get_preferred_device_id() == 0);
    CHECK_FALSE(h->dest_pipeline->is_pipeline_finished());
    CHECK(collector.batches.empty());
    auto retry_task = make_task(2, std::move(retry_input));
    task.reset();  // The retry is counted before the failed task completes.
    retry_task->execute(stream);
    retry_task.reset();
  } else {
    task->execute(stream);
    task.reset();
  }
  REQUIRE(collector.batches.size() == 1);
  auto const view = sirius::get_cudf_table_view(*collector.batches.front());
  REQUIRE(view.num_rows() == 1);
  CHECK(sirius::test::operator_utils::copy_column_to_host<int64_t>(view.column(0)) ==
        std::vector<int64_t>{0});
  CHECK(h->dest_pipeline->is_pipeline_finished());
  CHECK(completed.wait_for(std::chrono::seconds{0}) == std::future_status::ready);
  CHECK_NOTHROW(completed.get());
  CHECK_FALSE(h->hash_join->get_next_task_hint().has_value());
}
