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

// Deterministic, in-process coverage of the BUILD_PROBE orphan flow of
// sirius_physical_hash_join: a join whose probe side finishes with ZERO
// batches (no batch at all, as a zero-row-group parquet scan delivers) must
// reclaim its never-probed build slots, claim exactly one of them for an
// orphan build task with a synthesized 0-row probe batch, and emit one 0-row
// output batch so the query-terminal pipeline can complete.
//
// The harness wires real ports (repositories + src_pipeline pointers) through
// materialize_repository_wiring onto stub pipelines whose finished-ness the
// test controls, then drives the operator's public scheduling surface
// directly: get_partition_strategy (BUILD_PROBE sizing) -> deposit the folded
// build batch -> get_next_task_hint (reclaim + orphan claim) ->
// get_next_task_input_data (empty-probe synthesis) -> execute (0-row output).
// This covers, without any scan involvement, the same surface the SQL-level
// zero-batch shapes exercise.

#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <helper/type_conversions.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <pipeline/repository_wiring.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <memory>
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
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> pipelines;
  duckdb::shared_ptr<finished_stub_pipeline> probe_src_pipeline;
  duckdb::shared_ptr<finished_stub_pipeline> build_src_pipeline;
  duckdb::shared_ptr<sirius_pipeline> dest_pipeline;
  sirius_physical_operator probe_sink, probe_op, build_sink, build_op, dest_sink;
};

std::unique_ptr<orphan_join_harness> make_orphan_join_harness(duckdb::JoinType join_type)
{
  auto h = std::make_unique<orphan_join_harness>();

  h->logical_join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  if (join_type == duckdb::JoinType::MARK) {
    h->logical_join->types = {
      duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER, duckdb::LogicalType::BOOLEAN};
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
    1000,
    nullptr);

  // Source pipelines with one operator each (the producer a waiting hint names), and a
  // destination pipeline whose first operator is the join, so materialize_repository_wiring
  // attaches the "default"/"build" ports (repository + src_pipeline) onto it.
  auto build_stub = [&](sirius_physical_operator* sink,
                        const std::vector<sirius_physical_operator*>& operators,
                        std::size_t pipeline_id) {
    auto pipeline = duckdb::make_shared_ptr<finished_stub_pipeline>(h->build_ctx);
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
    auto pipeline = duckdb::make_shared_ptr<sirius_pipeline>(h->build_ctx);
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

/// Drive the orphan flow end-to-end on a sized harness whose build batches are deposited and whose
/// probe side is finished with zero batches, and assert every observable step: the hint claims the
/// orphan and reports READY; the single task carries [synthesized 0-row probe, deposited build];
/// execute emits one 0-row batch with `expected_output_columns`; afterwards both ports are drained
/// and the operator idles waiting on the finished probe producer.
void expect_orphan_flow(orphan_join_harness& h, cudf::size_type expected_output_columns)
{
  auto hint = h.hash_join->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == h.hash_join.get());

  auto input = h.hash_join->get_next_task_input_data();
  REQUIRE(input != nullptr);
  auto* partitioned = dynamic_cast<partitioned_operator_data*>(input.get());
  REQUIRE(partitioned != nullptr);
  auto const& input_batches = partitioned->get_data_batches();
  REQUIRE(input_batches.size() == 2);
  auto probe_view = sirius::get_cudf_table_view(*input_batches[0]);
  CHECK(probe_view.num_rows() == 0);
  CHECK(probe_view.num_columns() == 2);  // probe child schema: key + payload
  CHECK(sirius::get_cudf_table_view(*input_batches[1]).num_rows() == 3);

  // Exactly one orphan task is issued.
  REQUIRE(h.hash_join->get_next_task_input_data() == nullptr);

  auto output = h.hash_join->execute(*input, cudf::get_default_stream());
  REQUIRE(output != nullptr);
  auto const& output_data = dynamic_cast<const pipelineable_operator_data&>(*output);
  REQUIRE(output_data.get_data_batches().size() == 1);
  auto out_view = sirius::get_cudf_table_view(*output_data.get_data_batches()[0]);
  CHECK(out_view.num_rows() == 0);
  CHECK(out_view.num_columns() == expected_output_columns);

  // End state: both input repositories are drained (the pipeline can complete via
  // all_ports_empty) and the operator idles waiting on the finished probe producer — the normal
  // BUILD_PROBE end state.
  CHECK(h.hash_join->all_ports_empty());
  auto final_hint = h.hash_join->get_next_task_hint();
  REQUIRE(final_hint.has_value());
  CHECK(final_hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  CHECK(final_hint->producer == &h.probe_op);
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
  REQUIRE(strategy.broadcast);
  REQUIRE(strategy.num_partitions == 1);

  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  expect_orphan_flow(*h, /*expected_output_columns=*/3);  // left key + payload + right key
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
  REQUIRE_FALSE(strategy.broadcast);
  REQUIRE(strategy.num_partitions == 1);

  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  expect_orphan_flow(*h, /*expected_output_columns=*/3);
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
  h->hash_join->set_num_gpus(2);
  auto const strategy = h->hash_join->get_partition_strategy(
    {/*total_bytes=*/1024, /*is_build_side=*/true, /*build_foldable=*/true});
  REQUIRE(strategy.build_probe);
  REQUIRE(strategy.broadcast);
  REQUIRE(strategy.num_partitions == 2);

  for (std::size_t p = 0; p < 2; ++p) {
    h->hash_join->push_data_batch_partitioned(
      "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), p);
  }
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  expect_orphan_flow(*h, /*expected_output_columns=*/3);
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
  REQUIRE(strategy.num_partitions == 1);

  h->hash_join->push_data_batch_partitioned(
    "build", make_numeric_batch<int32_t>(*space, {1, 2, 3}, cudf::type_id::INT32), 0);
  h->probe_src_pipeline->finished = true;
  h->build_src_pipeline->finished = true;

  expect_orphan_flow(*h, /*expected_output_columns=*/3);  // left key + payload + BOOL8 mark
}
