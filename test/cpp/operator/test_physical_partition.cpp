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

#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "operator/aggregate/aggregate_test_utils.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/data_utils.hpp"

#include <catch.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/sirius_physical_concat.hpp>
#include <op/sirius_physical_delim_join.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_partition.hpp>

#include <array>
#include <numeric>

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

partition_barrier_fixture make_partition_barrier_fixture(duckdb::JoinType join_type)
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
    1);

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

struct sizing_source : sirius_physical_operator {
  sizing_source() : sirius_physical_operator(SiriusPhysicalOperatorType::PROJECTION, {}, 0) {}

  std::optional<std::size_t> total_source_output_bytes() const override { return projected_bytes; }
  std::optional<std::size_t> projected_bytes;
};

struct sizing_pipeline : sirius::pipeline::sirius_pipeline {
  explicit sizing_pipeline(const sirius::pipeline::pipeline_build_context& ctx)
    : sirius_pipeline(ctx)
  {
  }
  bool is_pipeline_finished() const override { return finished; }
  bool finished = false;
};

// Observe the sizing input at the consumer boundary, without context-level test counters.
// NESTED_LOOP_JOIN supplies a key-free partition; these tests exercise scheduling, not hashing.
struct sizing_consumer : sirius_physical_partition_consumer_operator {
  sizing_consumer()
    : sirius_physical_partition_consumer_operator(
        SiriusPhysicalOperatorType::NESTED_LOOP_JOIN, {}, 0)
  {
  }

  partition_strategy get_partition_strategy(const partition_sizing_input& in) override
  {
    inputs.push_back(in.total_bytes);
    // Collected eagerly here so the tests can inspect the snapshot the partition would provide.
    if (in.bypass_metadata_source) {
      bypass_metadata = in.bypass_metadata_source();
    } else {
      saw_null_bypass_metadata = true;
    }
    count = natural_num_partitions(in.total_bytes, target_bytes, 1);
    return {count, false, false};
  }

  uint64_t target_bytes = 1;
  int count             = 0;
  std::vector<uint64_t> inputs;
  std::optional<group_by_bypass_metadata> bypass_metadata;
  bool saw_null_bypass_metadata = false;
};

struct partition_sizing_fixture {
  explicit partition_sizing_fixture(bool enabled = true)
    : partition({}, 0, &consumer, false, nullptr, enabled)
  {
    // This fixture bypasses plan conversion, which normally assigns IDs before port wiring.
    source.operator_id    = 0;
    partition.operator_id = 1;
    consumer.operator_id  = 2;

    auto* space = memory_manager->get_memory_space(Tier::GPU, 0);
    REQUIRE(space != nullptr);
    auto stream = default_stream();
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(
      std::vector<int32_t>(100, 1), stream, get_resource_ref(*space)));
    auto representation = std::make_unique<gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(columns)), *space, stream);
    received_bytes = representation->get_size_in_bytes();
    REQUIRE(received_bytes > 0);
    consumer.target_bytes = received_bytes;
    repo.add_data_batch(data_batch::make(sirius::get_next_batch_id(), std::move(representation)));

    source.set_pipeline(producer);
    sirius::pipeline::sirius_pipeline_build_state build_state;
    build_state.set_pipeline_source(*producer, source);
    build_state.set_pipeline_operators(*producer, {source});
    build_state.set_pipeline_sink(
      *producer, sirius::optional_ptr<sirius_physical_operator>(&source), 0);
    auto port          = std::make_unique<sirius_physical_operator::port>();
    port->type         = enabled ? MemoryBarrierType::PARTIAL : MemoryBarrierType::FULL;
    port->repo         = &repo;
    port->src_pipeline = producer;
    partition.add_port("default", std::move(port));
  }

  void finish(std::size_t bytes)
  {
    producer->get_memory_history().record(
      sirius::pipeline::task_memory_record{bytes, bytes, bytes});
    producer->finished = true;
  }

  decltype(sirius::test::operator_utils::initialize_memory_manager()) memory_manager =
    sirius::test::operator_utils::initialize_memory_manager();
  sirius::pipeline::pipeline_build_context context{nullptr, true};
  std::shared_ptr<sizing_pipeline> producer = std::make_shared<sizing_pipeline>(context);
  sizing_source source;
  shared_data_repository repo;
  sizing_consumer consumer;
  sirius_physical_partition partition;
  std::size_t received_bytes = 0;
};

struct bypass_metadata_fixture : partition_sizing_fixture {
  explicit bypass_metadata_fixture(bool enabled = true, bool estimate = false)
    : partition_sizing_fixture(estimate)
  {
    auto params                                 = std::make_shared<sirius::operator_params>();
    params->enable_group_by_memory_aware_bypass = enabled;
    sirius::pipeline::pipeline_build_context ctx{nullptr, true, 1, params};
    partition.set_pipeline(std::make_shared<sirius::pipeline::sirius_pipeline>(ctx));
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(partition, &merge_parent);
  }

  sirius_physical_operator merge_parent{SiriusPhysicalOperatorType::MERGE_GROUP_BY, {}, 0};
};

}  // namespace

TEST_CASE("partition sizing uses a projection and keeps its first decision",
          "[physical_partition][size_estimation]")
{
  partition_sizing_fixture f;
  f.source.projected_bytes = 4 * f.received_bytes;
  auto hint                = f.partition.get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  // Scheduling and sizing must agree even if the source projection changes between them.
  f.source.projected_bytes = 6 * f.received_bytes;
  REQUIRE(f.partition.get_next_task_input_data());
  REQUIRE(f.consumer.inputs == std::vector<uint64_t>{4 * f.received_bytes});
  CHECK(f.consumer.count == 4);

  // A later sizing attempt must keep the count, even when the final total differs.
  f.finish(8 * f.received_bytes);
  CHECK_FALSE(f.partition.get_next_task_input_data());
  CHECK(f.consumer.inputs.size() == 1);
}

TEST_CASE("partition sizing waits for an estimate even with a queued batch",
          "[physical_partition][size_estimation]")
{
  partition_sizing_fixture f;
  auto hint = f.partition.get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  CHECK(hint->producer == &f.source);
  CHECK(f.consumer.inputs.empty());

  // Completion supplies an exact total even when no projection was available.
  f.finish(f.received_bytes);
  hint = f.partition.get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(f.partition.get_next_task_input_data());
  REQUIRE(f.consumer.inputs == std::vector<uint64_t>{f.received_bytes});
  CHECK(f.consumer.count == 1);
  CHECK(f.partition.no_history_peak_memory_estimate({1, f.received_bytes}) == 0);
}

TEST_CASE("partition sizing uses completed output instead of a stale source projection",
          "[physical_partition][size_estimation]")
{
  partition_sizing_fixture f;
  f.source.projected_bytes = 8 * f.received_bytes;
  f.finish(f.received_bytes);
  REQUIRE(f.partition.get_next_task_input_data());
  REQUIRE(f.consumer.inputs == std::vector<uint64_t>{f.received_bytes});
  CHECK(f.consumer.count == 1);
}

TEST_CASE("partition sizing floors a projection at the bytes already received",
          "[physical_partition][size_estimation]")
{
  partition_sizing_fixture f;
  f.source.projected_bytes = f.received_bytes / 2;
  REQUIRE(f.partition.get_next_task_input_data());
  REQUIRE(f.consumer.inputs == std::vector<uint64_t>{f.received_bytes});
  CHECK(f.consumer.count == 1);
}

TEST_CASE("disabled partition estimation waits for completion and uses measured input",
          "[physical_partition][size_estimation]")
{
  partition_sizing_fixture f(false);
  f.source.projected_bytes = 4 * f.received_bytes;
  auto hint                = f.partition.get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);

  f.finish(f.received_bytes);
  hint = f.partition.get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(f.partition.get_next_task_input_data());
  REQUIRE(f.consumer.inputs == std::vector<uint64_t>{f.received_bytes});
  CHECK(f.consumer.count == 1);
}

TEST_CASE("partition sizing preserves integer bytes above double precision",
          "[physical_partition][size_estimation]")
{
  partition_sizing_fixture f;
  constexpr uint64_t bytes = (uint64_t{1} << 53) + 1;
  f.consumer.target_bytes  = bytes;
  SECTION("projected") { f.source.projected_bytes = bytes; }
  SECTION("upstream complete") { f.finish(bytes); }

  REQUIRE(f.partition.get_next_task_input_data());
  REQUIRE(f.consumer.inputs == std::vector<uint64_t>{bytes});
  CHECK(f.consumer.count == 1);
}

TEST_CASE("bypass metadata is only collected when bypass is enabled",
          "[physical_partition][group_by_bypass]")
{
  bypass_metadata_fixture f(false);
  REQUIRE_FALSE(f.partition.is_memory_aware_bypass_enabled());
  f.finish(f.received_bytes);
  REQUIRE(f.partition.get_next_task_input_data());
  // With the setting off the PARTITION must not walk its batches at all, so the consumer sees no
  // metadata and the existing sizing path is unchanged.
  CHECK(f.consumer.saw_null_bypass_metadata);
  CHECK_FALSE(f.consumer.bypass_metadata.has_value());
}

TEST_CASE("bypass metadata reports the real rows, schema and target device",
          "[physical_partition][group_by_bypass]")
{
  bypass_metadata_fixture f;
  REQUIRE(f.partition.is_memory_aware_bypass_enabled());
  f.finish(f.received_bytes);
  // Keep the returned task input alive while checking the budget. It owns the batch popped from
  // the repository; destroying it here would release the batch's aligned GPU allocation after
  // collect_bypass_metadata() took its snapshot and make the later charged-byte sample 512 bytes
  // smaller than the state represented by the metadata.
  auto task_input = f.partition.get_next_task_input_data();
  REQUIRE(task_input);

  REQUIRE(f.consumer.bypass_metadata.has_value());
  auto const& meta = *f.consumer.bypass_metadata;
  CHECK(meta.upstream_complete);
  CHECK(meta.single_gpu_resident);
  REQUIRE(meta.total_rows.has_value());
  CHECK(*meta.total_rows == 100);  // the fixture deposits one 100-row INT32 column

  REQUIRE(meta.columns.has_value());
  REQUIRE(meta.columns->size() == 1);
  CHECK((*meta.columns)[0].type_id == static_cast<int>(cudf::type_id::INT32));
  CHECK((*meta.columns)[0].fixed_width_bytes == sizeof(int32_t));
  CHECK_FALSE((*meta.columns)[0].nullable);

  // The budget and device come from the space the input actually lives in, never from a
  // hardcoded device 0 lookup.
  auto* space = f.memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  CHECK(meta.target_device_id == space->get_device_id());
  REQUIRE(meta.admissible_additional_budget.has_value());

  // The budget must be what the executor could actually *reserve*, which is the space's
  // reservation limit minus everything already charged against it — not device memory that merely
  // happens to be unallocated. get_available_memory() measures the latter, against the larger
  // allocation capacity, and can exceed get_max_memory() outright when the reservation limit is
  // below capacity; a budget taken from it would over-admit by the bytes already in use.
  auto const limit   = space->get_max_memory();
  auto const charged = space->get_memory_resource_of<Tier::GPU>()->get_total_allocated_bytes();
  CHECK(*meta.admissible_additional_budget == (limit > charged ? limit - charged : 0));
}

TEST_CASE("bypass metadata marks a still-running upstream as incomplete",
          "[physical_partition][group_by_bypass]")
{
  // Production scheduling waits at the FULL barrier. Call the input hook directly to verify
  // that the metadata still identifies incomplete input defensively.
  bypass_metadata_fixture f;
  REQUIRE(f.partition.get_next_task_input_data());

  REQUIRE(f.consumer.bypass_metadata.has_value());
  CHECK_FALSE(f.consumer.bypass_metadata->upstream_complete);
  CHECK_FALSE(f.consumer.bypass_metadata->columns.has_value());
  CHECK(f.consumer.inputs == std::vector<uint64_t>{f.received_bytes});
}

TEST_CASE("bypass preserves the full-input barrier", "[physical_partition][group_by_bypass]")
{
  bypass_metadata_fixture f;
  auto hint = f.partition.get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  CHECK_FALSE(f.consumer.bypass_metadata.has_value());
  f.finish(f.received_bytes);
  hint = f.partition.get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->hint == TaskCreationHint::READY);
  REQUIRE(f.partition.get_next_task_input_data());
  REQUIRE(f.consumer.bypass_metadata.has_value());
  CHECK(f.consumer.bypass_metadata->upstream_complete);
}

TEST_CASE("bypass does not turn projected input into complete metadata",
          "[physical_partition][group_by_bypass][size_estimation]")
{
  bypass_metadata_fixture f(true, true);
  f.source.projected_bytes = 4 * f.received_bytes;
  auto hint                = f.partition.get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(f.partition.get_next_task_input_data());
  REQUIRE(f.consumer.bypass_metadata.has_value());
  CHECK_FALSE(f.consumer.bypass_metadata->upstream_complete);
  CHECK_FALSE(f.consumer.bypass_metadata->columns.has_value());
  CHECK(f.consumer.inputs == std::vector<uint64_t>{4 * f.received_bytes});
  // The first sizing decision remains fixed after the producer completes.
  f.finish(4 * f.received_bytes);
  CHECK_FALSE(f.partition.get_next_task_input_data());
  CHECK(f.consumer.inputs.size() == 1);
}

TEST_CASE("bypass query option only enables eligible group-by partitions",
          "[physical_partition][group_by_bypass]")
{
  bypass_metadata_fixture f;
  REQUIRE(f.partition.is_memory_aware_bypass_enabled());
  sirius_physical_delim_join delim(
    SiriusPhysicalOperatorType::LEFT_DELIM_JOIN, {}, nullptr, {}, 0, {});
  SECTION("unrelated consumer")
  {
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(f.partition, &f.consumer);
  }
  SECTION("delimiter-owned merge") { f.merge_parent.set_owning_delim_join(&delim); }
  SECTION("missing query context") { f.partition.set_pipeline(nullptr); }
  CHECK_FALSE(f.partition.is_memory_aware_bypass_enabled());
  f.finish(f.received_bytes);
  REQUIRE(f.partition.get_next_task_input_data());
  CHECK(f.consumer.saw_null_bypass_metadata);
}
