/*
 * Copyright 2025, Sirius Contributors.
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

#include "late_mat/column_origin.hpp"
#include "late_mat/defer_directive.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "operator/aggregate/aggregate_test_utils.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"
#include "planner/late_mat_plan_pass.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/data_utils.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/sirius_physical_concat.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_partition.hpp>
#include <parallel/after_task_work.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <array>
#include <atomic>
#include <future>
#include <numeric>
#include <stdexcept>

using namespace duckdb;
using namespace sirius::op;
using sirius::op::operator_data;
using sirius::op::pipelineable_operator_data;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

struct partition_barrier_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> join;
  sirius_physical_partition* probe_partition = nullptr;
  sirius_physical_partition* build_partition = nullptr;
};

partition_barrier_fixture make_partition_barrier_fixture(
  duckdb::JoinType join_type,
  dynamic_filter_publish_plan filter_plan = {},
  dynamic_filter_stats* stats             = nullptr)
{
  partition_barrier_fixture fixture;
  fixture.logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  fixture.logical_join->types = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER};

  auto make_child = [] {
    return duckdb::make_uniq<sirius_physical_operator>(
      SiriusPhysicalOperatorType::PROJECTION,
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1);
  };

  duckdb::JoinCondition condition;
  condition.left =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  condition.right =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  condition.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.push_back(std::move(condition));

  fixture.join = duckdb::make_uniq<sirius_physical_hash_join>(
    *fixture.logical_join,
    make_child(),
    make_child(),
    sirius::wrap_join_conditions(std::move(conditions)),
    join_type,
    duckdb::vector<std::size_t>{},
    duckdb::vector<std::size_t>{},
    duckdb::vector<sirius::logical_type>{},
    1,
    sirius::config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES,
    std::move(filter_plan),
    sirius::config::DEFAULT_HASH_PARTITION_BYTES,
    sirius::config::DEFAULT_MAX_BROADCAST_JOIN_SIZE,
    stats);

  auto wrap_side = [&](std::size_t child_idx, bool is_build) {
    auto child       = std::move(fixture.join->children[child_idx]);
    auto child_types = child->types;
    auto partition =
      duckdb::make_uniq<sirius_physical_partition>(child_types, 1, fixture.join.get(), is_build);
    auto* partition_ptr = partition.get();
    partition->children.push_back(std::move(child));

    auto concat =
      duckdb::make_uniq<sirius_physical_concat>(child_types, 1, fixture.join.get(), is_build);
    concat->children.push_back(std::move(partition));
    fixture.join->children[child_idx] = std::move(concat);
    return partition_ptr;
  };

  fixture.probe_partition = wrap_side(0, false);
  fixture.build_partition = wrap_side(1, true);
  sirius::planner::sirius_physical_plan_generator::set_parent_ops(*fixture.join, nullptr);
  return fixture;
}
}  // namespace

TEST_CASE("sirius_physical_partition barrier hook preserves producer precedence",
          "[physical_partition][partition_barrier]")
{
  auto fixture = make_partition_barrier_fixture(duckdb::JoinType::INNER);
  REQUIRE(fixture.probe_partition->get_parent_op()->type == SiriusPhysicalOperatorType::CONCAT);
  REQUIRE(fixture.probe_partition->get_parent_op()->get_parent_op() == fixture.join.get());

  sirius_physical_operator order_by{
    SiriusPhysicalOperatorType::ORDER_BY, duckdb::vector<sirius::logical_type>{}, 0};
  CHECK(fixture.probe_partition->input_barrier_for(order_by) == MemoryBarrierType::PIPELINE);

  for (auto producer_type : std::array{SiriusPhysicalOperatorType::PARTITION,
                                       SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE,
                                       SiriusPhysicalOperatorType::TOP_N,
                                       SiriusPhysicalOperatorType::SORT_PARTITION}) {
    CAPTURE(producer_type);
    sirius_physical_operator producer{producer_type, duckdb::vector<sirius::logical_type>{}, 0};
    CHECK(fixture.probe_partition->input_barrier_for(producer) == MemoryBarrierType::FULL);
  }

  sirius_physical_operator projection{
    SiriusPhysicalOperatorType::PROJECTION, duckdb::vector<sirius::logical_type>{}, 0};
  CHECK(fixture.probe_partition->input_barrier_for(projection) == MemoryBarrierType::PARTIAL);
}

TEST_CASE("sirius_physical_partition barrier hook preserves build and right-family barriers",
          "[physical_partition][partition_barrier]")
{
  sirius_physical_operator projection{
    SiriusPhysicalOperatorType::PROJECTION, duckdb::vector<sirius::logical_type>{}, 0};

  auto inner = make_partition_barrier_fixture(duckdb::JoinType::INNER);
  CHECK(inner.build_partition->input_barrier_for(projection) == MemoryBarrierType::FULL);

  auto right = make_partition_barrier_fixture(duckdb::JoinType::RIGHT);
  CHECK(right.probe_partition->input_barrier_for(projection) == MemoryBarrierType::FULL);
}

TEMPLATE_TEST_CASE("sirius_physical_partition partitions data_batch with single partition key",
                   "[physical_partition]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   bool,
                   decimal64_tag,
                   string_tag,
                   timestamp_us_tag,
                   date32_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  std::size_t num_values = 10000;

  std::size_t partition_size = 10000000;

  std::vector<typename Traits::type> values(num_values);
  if constexpr (Traits::is_string) {
    std::vector<std::string> string_values = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"};
    for (int i = 0; i < num_values; ++i) {
      values[i] = string_values[i % string_values.size()];
    }
  } else if constexpr (Traits::is_decimal) {
    for (int i = 0; i < num_values; ++i) {
      values[i] = static_cast<typename Traits::type>(i * 100);
    }
  } else if constexpr (Traits::is_ts) {
    for (int i = 0; i < num_values; ++i) {
      values[i] = static_cast<typename Traits::type>(i * 100'000);
    }
  } else if constexpr (std::is_same_v<typename Traits::type, int32_t> ||
                       std::is_same_v<typename Traits::type, int64_t> ||
                       std::is_same_v<typename Traits::type, int16_t>) {
    std::iota(values.begin(), values.end(), static_cast<typename Traits::type>(0));
  } else if constexpr (std::is_same_v<typename Traits::type, float> ||
                       std::is_same_v<typename Traits::type, double>) {
    for (int i = 0; i < num_values; ++i) {
      values[i] = static_cast<typename Traits::type>(i);
    }
  } else if constexpr (std::is_same_v<typename Traits::type, bool>) {
    for (int i = 0; i < num_values; ++i) {
      values[i] = (i % 2 == 0);
    }
  }
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  // Column 0: aggregation key
  auto col0 = sirius::test::vector_to_cudf_column<Traits>(values, stream, mr);
  // Column 1: aggregation value (all ones)
  auto col1 = sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(
    std::vector<int32_t>(num_values, 1), stream, mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col0));
  columns.push_back(std::move(col1));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::move(table), *space, cudf::get_default_stream());
  auto input_batch = data_batch::make(::sirius::get_next_batch_id(), std::move(gpu_repr));

  // this cardinality is not real, we are setting here this large in order to force more partitions
  // to be made
  std::size_t estimated_cardinality = 100000000;  // 100 million rows = PARTITION_SIZE x 10

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<gpu_type_traits<int32_t>>(
    {0},      // group_indexes: GROUP BY column 0
    {"sum"},  // aggregations: SUM
    {1}       // agg_indexes: SUM(column 1)
  );

  // Create partitioner types (copy of agg_output_types before moving)
  duckdb::vector<sirius::logical_type> partitioner_types = agg_result.output_types;

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregator(std::move(agg_result.output_types),
                                                             std::move(agg_result.aggregates),
                                                             std::move(agg_result.groups),
                                                             estimated_cardinality);

  sirius_physical_partition partitioner(
    partitioner_types, estimated_cardinality, &grouped_aggregator, false);

  // Compute num_partitions from estimated bytes: cardinality * bytes_per_row / partition_size
  // col0 is Traits::type, col1 is int32_t
  std::size_t bytes_per_row               = sizeof(typename Traits::type) + sizeof(int32_t);
  std::size_t estimated_cardinality_bytes = estimated_cardinality * bytes_per_row;
  int num_partitions =
    static_cast<int>(std::max(std::size_t(1), estimated_cardinality_bytes / partition_size));
  partitioner.set_num_partitions(num_partitions);

  auto outputs = partitioner.execute(pipelineable_operator_data({input_batch}), default_stream());

  std::size_t expected_num_partitions = static_cast<std::size_t>(num_partitions);

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() ==
          expected_num_partitions);

  // count the number of rows in each output and make sure it's the same and the initial inputs
  std::size_t total_num_rows = 0;
  for (auto& output :
       dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()) {
    total_num_rows += sirius::get_cudf_table_view(*output).num_rows();
  }
  REQUIRE(total_num_rows == num_values);
}

TEMPLATE_TEST_CASE("sirius_physical_partition partitions data_batch with two partition keys",
                   "[physical_partition]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   bool,
                   decimal64_tag,
                   string_tag,
                   timestamp_us_tag,
                   date32_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  std::size_t num_values0      = 40;
  std::size_t num_values1      = 10;
  std::size_t prime_repeater   = 17;  // repeating all values this number of times
  std::size_t total_num_values = num_values0 * num_values1 * prime_repeater;

  std::vector<typename Traits::type> values0(total_num_values);
  std::vector<int32_t> values1(total_num_values);
  std::size_t vidx0 = 0, vidx1 = 0;

  for (int i_prime = 0; i_prime < prime_repeater; ++i_prime) {
    if constexpr (Traits::is_string) {
      for (int i = 0; i < num_values0; ++i) {
        for (int32_t j = 0; j < num_values1; ++j) {
          values0[vidx0++] = std::to_string(i);
          values1[vidx1++] = j;
        }
      }
    } else if constexpr (std::is_same_v<typename Traits::type, int32_t> ||
                         std::is_same_v<typename Traits::type, int64_t> ||
                         std::is_same_v<typename Traits::type, int16_t> ||
                         std::is_same_v<typename Traits::type, float> ||
                         std::is_same_v<typename Traits::type, double>) {
      for (int i = 0; i < num_values0; ++i) {
        for (int32_t j = 0; j < num_values1; ++j) {
          values0[vidx0++] = static_cast<typename Traits::type>(i);
          values1[vidx1++] = j;
        }
      }
    } else if constexpr (std::is_same_v<typename Traits::type, bool>) {
      for (int i = 0; i < num_values0; ++i) {
        for (int32_t j = 0; j < num_values1; ++j) {
          values0[vidx0++] = (i % 2 == 0);
          values1[vidx1++] = j;
        }
      }
    }
  }
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  // Column 0: aggregation key0
  auto col0 = sirius::test::vector_to_cudf_column<Traits>(values0, stream, mr);
  // Column 1: aggregation key1
  auto col1 = sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(values1, stream, mr);
  // Column 2: aggregation value (same as column 1; values won't matter)
  auto col2 = sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(values1, stream, mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col0));
  columns.push_back(std::move(col1));
  columns.push_back(std::move(col2));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::move(table), *space, cudf::get_default_stream());
  auto input_batch = data_batch::make(::sirius::get_next_batch_id(), std::move(gpu_repr));

  std::size_t partition_size = 10000000;
  // this cardinality is not real, we are setting here this large in order to force more partitions
  // to be made
  std::size_t estimated_cardinality = 100000000;  // 100 million rows = PARTITION_SIZE x 10

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<gpu_type_traits<int32_t>>(
    {0, 1},   // group_indexes: GROUP BY column 0 and 1
    {"min"},  // aggregations: MIN
    {2}       // agg_indexes: MIN(column 2)
  );

  // Create partitioner types (copy of agg_output_types before moving)
  duckdb::vector<sirius::logical_type> partitioner_types = agg_result.output_types;

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregator(std::move(agg_result.output_types),
                                                             std::move(agg_result.aggregates),
                                                             std::move(agg_result.groups),
                                                             estimated_cardinality);

  sirius_physical_partition partitioner(
    partitioner_types, estimated_cardinality, &grouped_aggregator, false);

  // Compute num_partitions from estimated bytes: cardinality * bytes_per_row / partition_size
  // col0 is Traits::type, col1 and col2 are int32_t
  std::size_t bytes_per_row               = sizeof(typename Traits::type) + sizeof(int32_t) * 2;
  std::size_t estimated_cardinality_bytes = estimated_cardinality * bytes_per_row;
  int num_partitions =
    static_cast<int>(std::max(std::size_t(1), estimated_cardinality_bytes / partition_size));
  partitioner.set_num_partitions(num_partitions);

  auto outputs = partitioner.execute(pipelineable_operator_data({input_batch}), default_stream());

  std::size_t expected_num_partitions = static_cast<std::size_t>(num_partitions);

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() ==
          expected_num_partitions);

  // count the number of rows in each output and make sure it's the same and the initial inputs
  std::size_t total_num_rows = 0;
  for (auto& output :
       dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()) {
    std::size_t num_rows_out = sirius::get_cudf_table_view(*output).num_rows();
    REQUIRE(num_rows_out % prime_repeater ==
            0);  // each group was created to have prime_repeater rows, so each partition should
                 // have a multiple of that
    total_num_rows += num_rows_out;
  }
  REQUIRE(total_num_rows == total_num_values);
}

TEST_CASE(
  "sirius_physical_partition partitions data_batch with single partition key and 1 partition",
  "[physical_partition]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  std::size_t num_values = 10000;

  std::vector<int32_t> values(num_values);
  std::iota(values.begin(), values.end(), 0);

  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  // Column 0: partition key
  auto col0 = sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(values, stream, mr);
  // Column 1: aggregation value (all ones)
  auto col1 = sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(
    std::vector<int32_t>(num_values, 1), stream, mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col0));
  columns.push_back(std::move(col1));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::move(table), *space, cudf::get_default_stream());
  auto input_batch = data_batch::make(::sirius::get_next_batch_id(), std::move(gpu_repr));

  std::size_t estimated_cardinality = num_values;

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<gpu_type_traits<int32_t>>(
    {0},      // group_indexes: GROUP BY column 0
    {"sum"},  // aggregations: SUM
    {1}       // agg_indexes: SUM(column 1)
  );

  // Create partitioner types (copy of agg_output_types before moving)
  duckdb::vector<sirius::logical_type> partitioner_types = agg_result.output_types;

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregator(std::move(agg_result.output_types),
                                                             std::move(agg_result.aggregates),
                                                             std::move(agg_result.groups),
                                                             estimated_cardinality);

  sirius_physical_partition partitioner(
    partitioner_types, estimated_cardinality, &grouped_aggregator, false);

  // Compute num_partitions from estimated bytes: cardinality * bytes_per_row / partition_size
  // col0 and col1 are both int32_t; uses default partition size (512 MB)
  std::size_t bytes_per_row               = sizeof(int32_t) * 2;
  std::size_t estimated_cardinality_bytes = estimated_cardinality * bytes_per_row;
  int num_partitions                      = static_cast<int>(std::max(
    std::size_t(1), estimated_cardinality_bytes / sirius::config::DEFAULT_HASH_PARTITION_BYTES));
  partitioner.set_num_partitions(num_partitions);

  auto outputs = partitioner.execute(pipelineable_operator_data({input_batch}), default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  REQUIRE(sirius::get_cudf_table_view(
            *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0])
            .num_rows() == num_values);
}

namespace {

/// A source pipeline whose completion the test controls.
class controlled_source_pipeline final : public sirius::pipeline::sirius_pipeline {
 public:
  controlled_source_pipeline() : sirius_pipeline{sirius::pipeline::pipeline_build_context{nullptr}}
  {
  }
  bool is_pipeline_finished() const override { return finished.load(); }
  std::atomic<bool> finished{true};
};

/// Counts batch lookups and records whether any batch left before accumulation was decided.
class observed_build_repository final : public cucascade::shared_data_repository {
 public:
  explicit observed_build_repository(dynamic_filter_stats const& counters) : stats{counters} {}

  std::shared_ptr<data_batch> get_data_batch_by_id(std::uint64_t id,
                                                   std::size_t partition = 0) const override
  {
    ++lookups;
    return data_repository::get_data_batch_by_id(id, partition);
  }

  std::shared_ptr<data_batch> pop_next_data_batch(std::size_t partition = 0) override
  {
    auto const counters = stats.snapshot();
    if (counters.accumulations_started == 0 && counters.accumulations_skipped_inventory == 0) {
      popped_before_decision.store(true);
    }
    return data_repository::pop_next_data_batch(partition);
  }

  dynamic_filter_stats const& stats;
  mutable std::atomic<std::size_t> lookups{0};
  std::atomic<bool> popped_before_decision{false};
};

/// A build PARTITION with two partitions, fed through a FULL `default` port, whose join accumulates
/// an INT32 key.
struct partition_accumulation_fixture {
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  decltype(sirius::test::operator_utils::initialize_memory_manager()) manager =
    sirius::test::operator_utils::initialize_memory_manager();
  memory_space* gpu        = manager->get_memory_space(Tier::GPU, 0);
  memory_space const* host = manager->get_memory_spaces_for_tier(Tier::HOST).front();
  dynamic_filter_stats stats;
  observed_build_repository repository{stats};
  std::shared_ptr<sirius_dynamic_filter_set> channel =
    std::make_shared<sirius_dynamic_filter_set>();
  std::shared_ptr<controlled_source_pipeline> producer =
    std::make_shared<controlled_source_pipeline>();
  partition_barrier_fixture tree;

  explicit partition_accumulation_fixture(MemoryBarrierType barrier = MemoryBarrierType::FULL,
                                          std::uint64_t cap         = 64ULL << 20)
  {
    auto const type = cudf::data_type{cudf::type_id::INT32};
    dynamic_filter_publish_plan plan{
      {{.build_key_ordinal = 0, .storage_type = type}},
      {{channel, dynamic_filter_route_class::scan, false, {{0, 0, type}}}},
      {{*gpu, *host}},
      {.enable_multi_partition = true, .max_bloom_bytes_per_gpu = cap}};
    tree = make_partition_barrier_fixture(duckdb::JoinType::INNER, std::move(plan), &stats);
    tree.join->operator_id            = 1;
    tree.build_partition->operator_id = 2;
    tree.probe_partition->operator_id = 3;
    tree.build_partition->set_num_partitions(2);
    tree.probe_partition->set_num_partitions(2);
    auto port          = std::make_unique<sirius_physical_operator::port>();
    port->type         = barrier;
    port->repo         = &repository;
    port->src_pipeline = producer;
    tree.build_partition->add_port("default", std::move(port));
  }

  std::shared_ptr<data_batch> push(std::vector<std::int32_t> const& values)
  {
    auto batch = make_numeric_batch<std::int32_t>(*gpu, values, cudf::type_id::INT32);
    tree.build_partition->push_data_batch("default", batch);
    return batch;
  }

  /// Runs the build PARTITION's task-input hook as a task would, then its after-task work.
  void contribute(operator_data const& input)
  {
    auto const stream = gpu->acquire_stream();
    auto work         = tree.build_partition->observe_task_input(input, stream);
    stream.sync();
    if (work) { std::move(work)(stream); }
  }
};

}  // namespace

TEST_CASE("build PARTITION certifies its pushed input once before any batch leaves",
          "[physical_partition][dynamic_filter][multi_partition]")
{
  partition_accumulation_fixture fixture;
  auto const first  = fixture.push({1, 3});
  auto const second = fixture.push({7001, 9001, 11003});
  // Two creator threads race for the first pull.
  auto pull_a = std::async(
    std::launch::async, [&] { return fixture.tree.build_partition->get_next_task_input_data(); });
  auto pull_b = std::async(
    std::launch::async, [&] { return fixture.tree.build_partition->get_next_task_input_data(); });
  auto input_a = pull_a.get();
  auto input_b = pull_b.get();
  REQUIRE(input_a);
  REQUIRE(input_b);
  REQUIRE_FALSE(fixture.repository.popped_before_decision.load());
  REQUIRE(fixture.repository.lookups.load() == 0);
  auto counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulations_started == 1);
  REQUIRE(counters.accumulation_expected_contributions == 2);

  // The probe side never accumulates.
  REQUIRE_FALSE(
    fixture.tree.probe_partition->observe_task_input(*input_a, fixture.gpu->acquire_stream()));

  fixture.contribute(*input_a);
  fixture.contribute(*input_b);
  counters = fixture.stats.snapshot();
  REQUIRE(counters.accumulation_completed_contributions == 2);
  REQUIRE(counters.accumulation_publications_finished == 1);
  REQUIRE(counters.accumulations_skipped_error == 0);
  REQUIRE(fixture.channel->filter_count() == 1);
}

TEST_CASE("a push after certification fails, and is harmless once the ledger is abandoned",
          "[physical_partition][dynamic_filter][multi_partition]")
{
  SECTION("certified: the late batch is rejected before it becomes poppable")
  {
    partition_accumulation_fixture fixture;
    fixture.push({1, 3});
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    REQUIRE(fixture.stats.snapshot().accumulations_started == 1);
    auto late = make_numeric_batch<std::int32_t>(*fixture.gpu, {5}, cudf::type_id::INT32);
    REQUIRE_THROWS_AS(fixture.tree.build_partition->push_data_batch("default", late),
                      std::logic_error);
    REQUIRE(fixture.repository.total_size() == 0);
    fixture.tree.join->cancel_dynamic_filter_publication();
  }
  SECTION("abandoned after a declined start: the late batch is queued normally")
  {
    partition_accumulation_fixture fixture{MemoryBarrierType::FULL, 0};
    fixture.push({1, 3});
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    REQUIRE(fixture.stats.snapshot().accumulations_started == 0);
    auto late = make_numeric_batch<std::int32_t>(*fixture.gpu, {5}, cudf::type_id::INT32);
    REQUIRE_NOTHROW(fixture.tree.build_partition->push_data_batch("default", late));
    REQUIRE(fixture.repository.total_size() == 1);
  }
}

TEST_CASE("build PARTITION declines inputs it cannot account for",
          "[physical_partition][dynamic_filter][multi_partition]")
{
  partition_accumulation_fixture fixture;
  auto const first  = fixture.push({1, 3});
  auto const second = fixture.push({7, 9});
  SECTION("a multi-batch task input")
  {
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    fixture.contribute(pipelineable_operator_data({first, second}));
    REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
  }
  SECTION("a null-bearing task input")
  {
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    fixture.contribute(pipelineable_operator_data({first, nullptr}));
    REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
  }
  SECTION("a task input whose deferred columns are not yet materialized")
  {
    // A late-materialization directive on the PARTITION's port: the task input still carries
    // placeholders for deferred columns, so its keys cannot be accounted for.
    sirius_physical_operator scan{
      SiriusPhysicalOperatorType::ORDER_BY, duckdb::vector<sirius::logical_type>{}, 0};
    auto const pin    = std::make_shared<sirius::late_mat::pin_entry_handle>("accumulation_pin", 1);
    auto const origin = [&pin](std::uint32_t position) {
      return sirius::late_mat::column_origin{
        .handle = pin, .column_pos = position, .generation = pin->generation()};
    };
    std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::INT32},
                                              cudf::data_type{cudf::type_id::INT32}};
    auto pair =
      sirius::late_mat::make_defer_pair(schema, {0, 1}, schema, {0, 1}, {origin(0), origin(1)});
    REQUIRE(pair.valid());
    REQUIRE(
      sirius::planner::install_deferral(scan, *fixture.tree.build_partition, std::move(pair)));
    REQUIRE_FALSE(fixture.tree.build_partition->port_directive().empty());
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    REQUIRE(fixture.stats.snapshot().accumulations_started == 1);
    fixture.contribute(pipelineable_operator_data({first}));
    REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
    REQUIRE(fixture.stats.snapshot().accumulation_completed_contributions == 0);
  }
  SECTION("an extra repository partition")
  {
    fixture.repository.set_num_partitions(2);
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
    REQUIRE(fixture.stats.snapshot().accumulations_started == 0);
  }
  SECTION("a second data-bearing port")
  {
    cucascade::shared_data_repository other_repository;
    other_repository.add_data_batch(
      make_numeric_batch<std::int32_t>(*fixture.gpu, {11}, cudf::type_id::INT32));
    auto other          = std::make_unique<sirius_physical_operator::port>();
    other->type         = MemoryBarrierType::FULL;
    other->repo         = &other_repository;
    other->src_pipeline = fixture.producer;
    fixture.tree.build_partition->add_port("other", std::move(other));
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
    REQUIRE(fixture.stats.snapshot().accumulations_started == 0);
  }
  SECTION("a dependency-only port does not enlarge the input")
  {
    auto dependency          = std::make_unique<sirius_physical_operator::port>();
    dependency->type         = MemoryBarrierType::FULL;
    dependency->src_pipeline = fixture.producer;
    fixture.tree.build_partition->add_port("dependency", std::move(dependency));
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    REQUIRE(fixture.stats.snapshot().accumulations_started == 1);
    REQUIRE(fixture.stats.snapshot().accumulation_expected_contributions == 2);
    fixture.tree.join->cancel_dynamic_filter_publication();
  }
  REQUIRE(fixture.channel->empty());
}

TEST_CASE("a non-FULL build input never accumulates and never blocks",
          "[physical_partition][dynamic_filter][multi_partition]")
{
  sirius_physical_operator order_by{
    SiriusPhysicalOperatorType::ORDER_BY, duckdb::vector<sirius::logical_type>{}, 0};
  partition_accumulation_fixture fixture{MemoryBarrierType::PIPELINE};
  REQUIRE(fixture.tree.build_partition->input_barrier_for(order_by) == MemoryBarrierType::PIPELINE);
  fixture.producer->finished.store(false);
  fixture.push({1, 3});
  auto input = fixture.tree.build_partition->get_next_task_input_data();
  REQUIRE(input);
  REQUIRE(fixture.stats.snapshot().accumulations_skipped_inventory == 1);
  REQUIRE(fixture.stats.snapshot().accumulations_started == 0);
  REQUIRE_FALSE(
    fixture.tree.build_partition->observe_task_input(*input, fixture.gpu->acquire_stream()));
  // Later batches are queued and pulled as usual.
  REQUIRE_NOTHROW(fixture.push({5, 7}));
  REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
  fixture.tree.join->cancel_dynamic_filter_publication();
}

TEST_CASE("a FULL build input is certified only after its source pipeline finished",
          "[physical_partition][dynamic_filter][multi_partition]")
{
  SECTION("an unfinished source keeps every batch queued")
  {
    partition_accumulation_fixture fixture;
    fixture.push({1, 3});
    fixture.producer->finished.store(false);
    REQUIRE_FALSE(fixture.tree.build_partition->get_next_task_input_data());
    REQUIRE(fixture.repository.total_size() == 1);
    REQUIRE(fixture.stats.snapshot().publication_attempts == 0);
    fixture.producer->finished.store(true);
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    REQUIRE(fixture.stats.snapshot().accumulations_started == 1);
    fixture.tree.join->cancel_dynamic_filter_publication();
  }
  SECTION("one partition attempts nothing")
  {
    partition_accumulation_fixture fixture;
    fixture.tree.build_partition->set_num_partitions(1);
    fixture.push({1, 3});
    REQUIRE(fixture.tree.build_partition->get_next_task_input_data());
    auto const counters = fixture.stats.snapshot();
    REQUIRE(counters.publication_attempts == 0);
    REQUIRE(counters.accumulations_skipped_inventory == 0);
  }
}
