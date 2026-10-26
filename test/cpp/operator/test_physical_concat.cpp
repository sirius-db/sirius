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

#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "operator/partition_test_utils.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"
#include "utils/sirius_test_env.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <cucascade/cudf/host_table_packed.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/sirius_physical_concat.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_partition.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <atomic>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using sirius::op::operator_data;
using sirius::op::pipelineable_operator_data;

namespace {

using namespace sirius::test::operator_utils;

/// Partition `partition_idx` of an unpinned exchange, as an operator built without an engine sees
/// it (no GPU list, so no placement pins a device).
partitioned_operator_data unpinned_partition(
  std::vector<std::shared_ptr<cucascade::data_batch>> batches, std::size_t partition_idx)
{
  return partitioned_operator_data(
    std::move(batches), partition_idx, partition_placement::unpinned(partition_idx + 1));
}

//===----------------------------------------------------------------------===//
// Hash join fixture for partition negotiation tests
//===----------------------------------------------------------------------===//

/**
 * @brief Holds the LogicalComparisonJoin and hash join objects needed for
 * partition negotiation. The logical_join must outlive the
 * hash_join because the hash_join stores op.types by reference.
 */
struct hash_join_test_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> hash_join;
};

//! Depth-first, root-first numbering of a bare operator tree, standing in for
//! pipeline::assign_operator_ids in fixtures that never build pipelines.
void number_operator_tree(sirius_physical_operator& op, size_t& next_id)
{
  op.operator_id = next_id++;
  for (auto& child : op.children) {
    if (child) { number_operator_tree(*child, next_id); }
  }
}

auto& get_shared_memory_manager()
{
  static auto manager = sirius::test::operator_utils::initialize_memory_manager();
  return *manager;
}

/**
 * @brief Create a minimal sirius_physical_hash_join for testing concat.
 *
 * @param join_type The join type (INNER, LEFT, RIGHT, etc.)
 * @param output_types The logical types for the join output columns
 * @return hash_join_test_fixture owning both the logical and physical join
 */
hash_join_test_fixture create_test_hash_join(
  duckdb::JoinType join_type,
  duckdb::vector<duckdb::LogicalType> output_types,
  uint64_t hash_partition_bytes = sirius::config::DEFAULT_HASH_PARTITION_BYTES,
  bool mixed_filter             = false)
{
  hash_join_test_fixture fixture;

  // Create a LogicalComparisonJoin with the desired join type
  fixture.logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  fixture.logical_join->types = output_types;

  // Create minimal child operators (need at least one type each for the hash join constructor)
  auto left_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0);
  auto right_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0);

  // Create a single equality join condition (column 0 = column 0)
  duckdb::vector<duckdb::JoinCondition> conditions;
  duckdb::JoinCondition cond(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
    duckdb::ExpressionType::COMPARE_EQUAL);
  conditions.push_back(std::move(cond));

  dynamic_filter_publish_plan filter_plan;
  if (mixed_filter) {
    // A mixed INNER join cannot use BUILD_PROBE; its small single-partition build is
    // nevertheless eligible for whole-build dynamic-filter publication.
    conditions.emplace_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
      duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto& manager    = get_shared_memory_manager();
    auto const hosts = manager.get_memory_spaces_for_tier(Tier::HOST);
    REQUIRE_FALSE(hosts.empty());
    auto* gpu = manager.get_memory_space(Tier::GPU, 0);
    filter_plan =
      dynamic_filter_publish_plan({},
                                  {{.filter_set = std::make_shared<sirius_dynamic_filter_set>()}},
                                  {{*gpu, *hosts.front()}});
  }

  // Build the hash join
  fixture.hash_join = duckdb::make_uniq<sirius_physical_hash_join>(
    *fixture.logical_join,
    std::move(left_child),
    std::move(right_child),
    sirius::wrap_join_conditions(std::move(conditions)),
    join_type,
    duckdb::vector<duckdb::idx_t>{},  // left_projection_map (empty = all)
    duckdb::vector<duckdb::idx_t>{},  // right_projection_map (empty = all)
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),  // delim_types
    1000,                                                            // estimated_cardinality
    sirius::config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES,
    std::move(filter_plan),
    hash_partition_bytes);

  // These fixtures build a bare operator tree with no pipelines, so the converter's
  // assign_operator_ids never runs over it. Number it here — operator code reads
  // get_operator_id(), which rejects the unassigned sentinel.
  size_t next_id = 0;
  number_operator_tree(*fixture.hash_join, next_id);

  return fixture;
}

/**
 * @brief Get a shared memory space that persists across all test cases.
 */
memory_space* get_shared_mem_space()
{
  return get_shared_memory_manager().get_memory_space(Tier::GPU, 0);
}

//===----------------------------------------------------------------------===//
// Source-pipeline fixture for get_next_task_hint / get_next_task_input_data
//===----------------------------------------------------------------------===//

/**
 * @brief A sirius_pipeline whose finished state the test controls, standing in for the source
 * pipeline feeding the concat operator's input port.
 */
class mock_gpu_pipeline : public sirius::pipeline::sirius_pipeline {
 public:
  explicit mock_gpu_pipeline(const sirius::pipeline::pipeline_build_context& ctx)
    : sirius_pipeline(ctx)
  {
  }

  void set_finished(bool finished) { _finished = finished; }

  bool is_pipeline_finished() const override { return _finished; }

 private:
  bool _finished = false;
};

/**
 * @brief Bundles a controllable source pipeline with the upstream producer operator that
 * get_next_task_hint reports in WAITING_FOR_INPUT_DATA hints. The producer must outlive the
 * pipeline, which stores it by reference.
 */
struct source_pipeline_fixture {
  duckdb::unique_ptr<sirius_physical_operator> upstream_producer;
  std::shared_ptr<mock_gpu_pipeline> pipeline;
};

source_pipeline_fixture create_unfinished_source_pipeline()
{
  source_pipeline_fixture fixture;
  const sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
  fixture.pipeline          = std::make_shared<mock_gpu_pipeline>(build_ctx);
  fixture.upstream_producer = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000);
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.add_pipeline_operator(*fixture.pipeline, *fixture.upstream_producer);
  return fixture;
}

/**
 * @brief Attach a single input port backed by @p repo to @p concat_op, optionally wired to a
 * source pipeline.
 */
void attach_concat_port(sirius_physical_concat& concat_op,
                        cucascade::shared_data_repository& repo,
                        std::shared_ptr<mock_gpu_pipeline> src_pipeline = nullptr)
{
  auto port           = std::make_unique<sirius_physical_operator::port>();
  port->type          = MemoryBarrierType::FULL;
  port->repo          = &repo;
  port->src_pipeline  = std::move(src_pipeline);
  port->dest_pipeline = nullptr;
  concat_op.add_port("input", std::move(port));
}

std::unique_ptr<operator_data> claim_concat_input(sirius_physical_concat& concat,
                                                  cucascade::shared_data_repository& repo,
                                                  std::vector<std::shared_ptr<data_batch>> batches,
                                                  std::size_t partition)
{
  attach_concat_port(concat, repo);
  for (auto& batch : batches) {
    repo.add_data_batch(std::move(batch), partition);
  }
  auto input = concat.get_next_task_input_data();
  REQUIRE(input != nullptr);
  return input;
}

/**
 * @brief Create an int32 batch of @p num_rows rows (4 bytes per row).
 */
std::shared_ptr<data_batch> make_int32_batch(memory_space& space, std::size_t num_rows)
{
  std::vector<int32_t> values(num_rows);
  std::iota(values.begin(), values.end(), 0);
  return make_numeric_batch<int32_t>(space, values, cudf::type_id::INT32);
}

// Metadata-only host input for planning tests. It must never be uploaded or executed.
std::shared_ptr<data_batch> make_host_metadata_batch(cudf::size_type rows)
{
  auto const hosts = get_shared_memory_manager().get_memory_spaces_for_tier(Tier::HOST);
  REQUIRE_FALSE(hosts.empty());
  auto* host = const_cast<memory_space*>(hosts.front());
  cucascade::memory::column_metadata column{};
  column.type_id  = static_cast<int32_t>(cudf::type_id::INT32);
  column.num_rows = rows;
  auto table      = cucascade::memory::host_table_allocation::create({}, {column}, sizeof(int32_t));
  auto repr       = std::make_unique<cucascade::host_data_representation>(std::move(table), host);
  return data_batch::make(sirius::get_next_batch_id(), std::move(repr));
}

}  // namespace

//===----------------------------------------------------------------------===//
// 1. Execute tests
//===----------------------------------------------------------------------===//

TEMPLATE_TEST_CASE("sirius_physical_concat concatenates multiple data_batches",
                   "[physical_concat]",
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

  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  // Create 5 batches of varying sizes
  std::vector<std::size_t> batch_sizes = {100, 200, 300, 400, 500};
  std::size_t total_rows               = 0;
  for (auto s : batch_sizes) {
    total_rows += s;
  }

  // Build input values for each batch
  std::vector<std::shared_ptr<data_batch>> input_batches;
  std::vector<typename Traits::type> all_values;  // expected concatenated values

  for (auto num_rows : batch_sizes) {
    std::vector<typename Traits::type> values(num_rows);
    if constexpr (Traits::is_string) {
      std::vector<std::string> pool = {"alpha", "beta", "gamma", "delta", "epsilon"};
      for (std::size_t i = 0; i < num_rows; ++i) {
        values[i] = pool[i % pool.size()];
      }
    } else if constexpr (Traits::is_decimal) {
      for (std::size_t i = 0; i < num_rows; ++i) {
        values[i] = static_cast<typename Traits::type>(i * 100);
      }
    } else if constexpr (Traits::is_ts) {
      for (std::size_t i = 0; i < num_rows; ++i) {
        values[i] = static_cast<typename Traits::type>(i * 1'000'000);
      }
    } else if constexpr (std::is_same_v<typename Traits::type, bool>) {
      for (std::size_t i = 0; i < num_rows; ++i) {
        values[i] = (i % 2 == 0);
      }
    } else {
      for (std::size_t i = 0; i < num_rows; ++i) {
        values[i] = static_cast<typename Traits::type>(i);
      }
    }

    all_values.insert(all_values.end(), values.begin(), values.end());

    std::shared_ptr<data_batch> batch;
    if constexpr (Traits::is_string) {
      batch = make_string_batch(*space, values);
    } else if constexpr (Traits::is_decimal) {
      batch = make_decimal64_batch(*space, values, Traits::scale);
    } else if constexpr (Traits::is_ts) {
      batch = make_timestamp_batch(*space, values, Traits::cudf_type);
    } else {
      batch = make_numeric_batch<typename Traits::type>(*space, values, Traits::cudf_type);
    }
    input_batches.push_back(std::move(batch));
  }

  // Create a generic CONCAT operator
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{Traits::logical_type()}),
    1000,
    false);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

  // Execute
  cucascade::shared_data_repository repo;
  auto input = claim_concat_input(concat_op, repo, input_batches, 0);
  input->prepare_for_processing(space, default_stream());
  auto outputs = concat_op.execute(*input, default_stream());

  // Verify: single output batch with correct total rows
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  auto out_table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  REQUIRE(static_cast<std::size_t>(out_table.num_rows()) == total_rows);
  REQUIRE(out_table.num_columns() == 1);

  // Verify data content
  auto host_data = copy_column_to_host<typename Traits::type>(out_table.column(0));
  REQUIRE(host_data.size() == all_values.size());
  for (std::size_t i = 0; i < all_values.size(); ++i) {
    REQUIRE(host_data[i] == all_values[i]);
  }
}

TEST_CASE("sirius_physical_concat returns single batch as-is", "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  std::size_t num_rows = 500;
  std::vector<int32_t> values(num_rows);
  std::iota(values.begin(), values.end(), 0);
  auto input_batch = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);

  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

  cucascade::shared_data_repository repo;
  auto input = claim_concat_input(concat_op, repo, {input_batch}, 0);
  input->prepare_for_processing(space, default_stream());
  auto outputs = concat_op.execute(*input, default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  // Single batch should be the same pointer (passthrough)
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0].get() ==
          input_batch.get());
}

TEST_CASE("sirius_physical_concat handles empty input", "[physical_concat]")
{
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

  cucascade::shared_data_repository repo;
  attach_concat_port(concat_op, repo);
  CHECK(concat_op.get_next_task_input_data() == nullptr);
  CHECK_FALSE(concat_op.get_next_task_hint().has_value());
}

//===----------------------------------------------------------------------===//
// 2. Sink tests
//===----------------------------------------------------------------------===//

TEST_CASE(
  "sirius_physical_concat sink forwards batches to downstream operator with partition index",
  "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  // Create two batches with known values
  std::vector<int32_t> values1 = {10, 20, 30};
  std::vector<int32_t> values2 = {40, 50, 60};
  auto batch1                  = make_numeric_batch<int32_t>(*space, values1, cudf::type_id::INT32);
  auto batch2                  = make_numeric_batch<int32_t>(*space, values2, cudf::type_id::INT32);
  auto batch1_id               = batch1->get_batch_id();
  auto batch2_id               = batch2->get_batch_id();

  // Create the concat operator
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false);

  // Create a downstream partition consumer operator to receive the sink output
  sirius_physical_concat downstream_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false);

  // Set up a data repository on the downstream operator's port
  auto downstream_repo           = std::make_unique<cucascade::shared_data_repository>();
  auto downstream_port           = std::make_unique<sirius_physical_operator::port>();
  downstream_port->type          = MemoryBarrierType::FULL;
  downstream_port->repo          = downstream_repo.get();
  downstream_port->src_pipeline  = nullptr;
  downstream_port->dest_pipeline = nullptr;
  downstream_op.add_port("input", std::move(downstream_port));

  // Register the downstream operator as the next sink target
  concat_op.add_next_port_after_sink({&downstream_op, "input"});

  // Sink partitioned data with partition_idx = 3
  constexpr std::size_t partition_idx = 3;
  auto const sink_data                = unpinned_partition({batch1, batch2}, partition_idx);
  concat_op.sink(sink_data, default_stream());

  // Verify: downstream repo should have both batches in partition 3
  auto batch_ids = downstream_repo->get_batch_ids(partition_idx);
  REQUIRE(batch_ids.size() == 2);

  // Verify the batch IDs match
  std::set<uint64_t> expected_ids = {batch1_id, batch2_id};
  std::set<uint64_t> actual_ids(batch_ids.begin(), batch_ids.end());
  REQUIRE(actual_ids == expected_ids);
}

TEST_CASE("sirius_physical_concat sink forwards to multiple downstream operators",
          "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  std::vector<int32_t> values = {1, 2, 3};
  auto batch                  = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
  auto batch_id               = batch->get_batch_id();

  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false);

  // Create two downstream operators
  sirius_physical_concat downstream1(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false);
  sirius_physical_concat downstream2(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false);

  auto repo1           = std::make_unique<cucascade::shared_data_repository>();
  auto port1           = std::make_unique<sirius_physical_operator::port>();
  port1->type          = MemoryBarrierType::FULL;
  port1->repo          = repo1.get();
  port1->src_pipeline  = nullptr;
  port1->dest_pipeline = nullptr;
  downstream1.add_port("input", std::move(port1));

  auto repo2           = std::make_unique<cucascade::shared_data_repository>();
  auto port2           = std::make_unique<sirius_physical_operator::port>();
  port2->type          = MemoryBarrierType::FULL;
  port2->repo          = repo2.get();
  port2->src_pipeline  = nullptr;
  port2->dest_pipeline = nullptr;
  downstream2.add_port("input", std::move(port2));

  concat_op.add_next_port_after_sink({&downstream1, "input"});
  concat_op.add_next_port_after_sink({&downstream2, "input"});

  constexpr std::size_t partition_idx = 1;
  auto const sink_data                = unpinned_partition({batch}, partition_idx);
  concat_op.sink(sink_data, default_stream());

  // Both downstream repos should have the batch in partition 1
  auto ids1 = repo1->get_batch_ids(partition_idx);
  REQUIRE(ids1.size() == 1);
  REQUIRE(ids1[0] == batch_id);

  auto ids2 = repo2->get_batch_ids(partition_idx);
  REQUIRE(ids2.size() == 1);
  REQUIRE(ids2[0] == batch_id);
}

//===----------------------------------------------------------------------===//
// 3. get_next_task_input_batch threshold tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_concat stops concatenating at concat_batch_bytes threshold",
          "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  // Use a small threshold so our test batches exceed it
  constexpr uint64_t threshold = 1024;  // 1 KB

  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false,
    threshold);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

  // Set up a port with a data repository
  auto repo = std::make_unique<cucascade::shared_data_repository>();

  // Create batches that are each bigger than 1 KB (1000 int32 values = 4000 bytes > 1 KB)
  constexpr int num_batches            = 5;
  constexpr std::size_t rows_per_batch = 1000;
  for (int b = 0; b < num_batches; ++b) {
    std::vector<int32_t> values(rows_per_batch);
    std::iota(values.begin(), values.end(), static_cast<int32_t>(b * rows_per_batch));
    auto batch = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
    repo->add_data_batch(std::move(batch), 0);
  }

  // Add the port to the concat operator
  auto port           = std::make_unique<sirius_physical_operator::port>();
  port->type          = MemoryBarrierType::FULL;
  port->repo          = repo.get();
  port->src_pipeline  = nullptr;
  port->dest_pipeline = nullptr;
  concat_op.add_port("input", std::move(port));

  // First call: should return some batches but not all (threshold exceeded)
  auto result1 = concat_op.get_next_task_input_data();
  REQUIRE(result1 != nullptr);
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*result1).get_data_batches().size() <
          static_cast<std::size_t>(num_batches));
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*result1).get_data_batches().size() >= 1);

  // Collect total batches returned across multiple calls
  std::size_t total_batches_returned =
    dynamic_cast<const pipelineable_operator_data&>(*result1).get_data_batches().size();
  while (true) {
    auto result = concat_op.get_next_task_input_data();
    if (!result) { break; }
    total_batches_returned +=
      dynamic_cast<const pipelineable_operator_data&>(*result).get_data_batches().size();
  }

  // All batches should eventually be consumed
  REQUIRE(total_batches_returned == static_cast<std::size_t>(num_batches));
}

TEST_CASE("sirius_physical_concat with concat_all=true ignores threshold", "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  // Use a small threshold (ignored when concat_all=true)
  constexpr uint64_t threshold = 1024;  // 1 KB

  // Complete-partition mode ignores the byte threshold.
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    true,
    threshold);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

  // Set up a port with a data repository
  auto repo = std::make_unique<cucascade::shared_data_repository>();

  // Create batches that are each bigger than 1 KB
  constexpr int num_batches            = 5;
  constexpr std::size_t rows_per_batch = 1000;
  for (int b = 0; b < num_batches; ++b) {
    std::vector<int32_t> values(rows_per_batch);
    std::iota(values.begin(), values.end(), static_cast<int32_t>(b * rows_per_batch));
    auto batch = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
    repo->add_data_batch(std::move(batch), 0);
  }

  auto port           = std::make_unique<sirius_physical_operator::port>();
  port->type          = MemoryBarrierType::FULL;
  port->repo          = repo.get();
  port->src_pipeline  = nullptr;
  port->dest_pipeline = nullptr;
  concat_op.add_port("input", std::move(port));

  // With concat_all=true, all batches in the partition should be returned in one call
  auto result = concat_op.get_next_task_input_data();
  REQUIRE(result != nullptr);
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*result).get_data_batches().size() ==
          static_cast<std::size_t>(num_batches));

  // No more batches remaining
  auto result2 = concat_op.get_next_task_input_data();
  REQUIRE(result2 == nullptr);
}

//===----------------------------------------------------------------------===//
// Unfinished-source-pipeline gating tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_concat defers under-threshold groups while the source pipeline runs",
          "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold = 1024;  // 1 KB

  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false,
    threshold);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

  auto repo   = std::make_unique<cucascade::shared_data_repository>();
  auto source = create_unfinished_source_pipeline();
  attach_concat_port(concat_op, *repo, source.pipeline);

  // Two 400-byte batches accumulate below the threshold: no group forms yet.
  auto batch_a    = make_int32_batch(*space, 100);
  auto batch_b    = make_int32_batch(*space, 100);
  const auto id_a = batch_a->get_batch_id();
  const auto id_b = batch_b->get_batch_id();
  repo->add_data_batch(std::move(batch_a), 0);
  repo->add_data_batch(std::move(batch_b), 0);

  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
  REQUIRE(repo->get_batch_ids(0).size() == 2);

  auto waiting_hint = concat_op.get_next_task_hint();
  REQUIRE(waiting_hint.has_value());
  REQUIRE(waiting_hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(waiting_hint->producer == source.upstream_producer.get());

  // A third 400-byte batch crosses the threshold: the first two form a complete group and the
  // overflowing batch stays behind to seed the next group.
  auto batch_c    = make_int32_batch(*space, 100);
  const auto id_c = batch_c->get_batch_id();
  repo->add_data_batch(std::move(batch_c), 0);

  auto ready_hint = concat_op.get_next_task_hint();
  REQUIRE(ready_hint.has_value());
  REQUIRE(ready_hint->hint == TaskCreationHint::READY);
  REQUIRE(ready_hint->producer == &concat_op);

  auto group = concat_op.get_next_task_input_data();
  REQUIRE(group != nullptr);
  const auto& group_batches =
    dynamic_cast<const pipelineable_operator_data&>(*group).get_data_batches();
  REQUIRE(group_batches.size() == 2);
  REQUIRE(group_batches[0]->get_batch_id() == id_a);
  REQUIRE(group_batches[1]->get_batch_id() == id_b);

  // The leftover batch is under the threshold, so it keeps accumulating.
  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
  REQUIRE(repo->get_batch_ids(0) == std::vector<uint64_t>{id_c});

  // Pipeline finish flushes the under-threshold tail.
  source.pipeline->set_finished(true);
  auto flush_hint = concat_op.get_next_task_hint();
  REQUIRE(flush_hint.has_value());
  REQUIRE(flush_hint->hint == TaskCreationHint::READY);

  auto tail = concat_op.get_next_task_input_data();
  REQUIRE(tail != nullptr);
  const auto& tail_batches =
    dynamic_cast<const pipelineable_operator_data&>(*tail).get_data_batches();
  REQUIRE(tail_batches.size() == 1);
  REQUIRE(tail_batches[0]->get_batch_id() == id_c);

  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
  REQUIRE_FALSE(concat_op.get_next_task_hint().has_value());
}

TEST_CASE("sirius_physical_concat holds a lone oversized batch until more data or pipeline finish",
          "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold = 1024;  // 1 KB

  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false,
    threshold);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

  auto repo   = std::make_unique<cucascade::shared_data_repository>();
  auto source = create_unfinished_source_pipeline();
  attach_concat_port(concat_op, *repo, source.pipeline);

  // A single 1200-byte batch exceeds the threshold on its own, but more data may still arrive
  // behind it, so nothing is released yet.
  auto oversized          = make_int32_batch(*space, 300);
  const auto oversized_id = oversized->get_batch_id();
  repo->add_data_batch(std::move(oversized), 0);

  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
  REQUIRE(repo->get_batch_ids(0) == std::vector<uint64_t>{oversized_id});

  auto waiting_hint = concat_op.get_next_task_hint();
  REQUIRE(waiting_hint.has_value());
  REQUIRE(waiting_hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(waiting_hint->producer == source.upstream_producer.get());

  // A second batch behind it releases the oversized batch as a single-batch group.
  auto small          = make_int32_batch(*space, 100);
  const auto small_id = small->get_batch_id();
  repo->add_data_batch(std::move(small), 0);

  auto ready_hint = concat_op.get_next_task_hint();
  REQUIRE(ready_hint.has_value());
  REQUIRE(ready_hint->hint == TaskCreationHint::READY);

  auto group = concat_op.get_next_task_input_data();
  REQUIRE(group != nullptr);
  const auto& group_batches =
    dynamic_cast<const pipelineable_operator_data&>(*group).get_data_batches();
  REQUIRE(group_batches.size() == 1);
  REQUIRE(group_batches[0]->get_batch_id() == oversized_id);

  // The remaining under-threshold batch keeps accumulating until the source is done.
  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
  REQUIRE(repo->get_batch_ids(0) == std::vector<uint64_t>{small_id});

  source.pipeline->set_finished(true);
  auto tail = concat_op.get_next_task_input_data();
  REQUIRE(tail != nullptr);
  const auto& tail_batches =
    dynamic_cast<const pipelineable_operator_data&>(*tail).get_data_batches();
  REQUIRE(tail_batches.size() == 1);
  REQUIRE(tail_batches[0]->get_batch_id() == small_id);

  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
}

TEST_CASE("sirius_physical_concat with concat_all defers all batches until pipeline finish",
          "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold = 1024;  // 1 KB

  // Complete-partition mode ignores the byte threshold.
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    true,
    threshold);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

  auto repo   = std::make_unique<cucascade::shared_data_repository>();
  auto source = create_unfinished_source_pipeline();
  attach_concat_port(concat_op, *repo, source.pipeline);

  // Mix under- and over-threshold batches: the threshold plays no role with concat_all.
  std::vector<uint64_t> expected_ids;
  for (std::size_t num_rows : {100UL, 300UL, 100UL}) {
    auto batch = make_int32_batch(*space, num_rows);
    expected_ids.push_back(batch->get_batch_id());
    repo->add_data_batch(std::move(batch), 0);
  }

  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
  REQUIRE(repo->get_batch_ids(0) == expected_ids);

  auto waiting_hint = concat_op.get_next_task_hint();
  REQUIRE(waiting_hint.has_value());
  REQUIRE(waiting_hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(waiting_hint->producer == source.upstream_producer.get());

  // Pipeline finish releases the whole partition as one group.
  source.pipeline->set_finished(true);
  auto ready_hint = concat_op.get_next_task_hint();
  REQUIRE(ready_hint.has_value());
  REQUIRE(ready_hint->hint == TaskCreationHint::READY);

  auto all = concat_op.get_next_task_input_data();
  REQUIRE(all != nullptr);
  const auto& all_batches =
    dynamic_cast<const pipelineable_operator_data&>(*all).get_data_batches();
  REQUIRE(all_batches.size() == expected_ids.size());
  for (std::size_t i = 0; i < expected_ids.size(); ++i) {
    REQUIRE(all_batches[i]->get_batch_id() == expected_ids[i]);
  }

  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
}

TEST_CASE("sirius_physical_concat pulls from a later partition when earlier ones are not ready",
          "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold = 1024;  // 1 KB

  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false,
    threshold);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(2)));

  auto repo   = std::make_unique<cucascade::shared_data_repository>();
  auto source = create_unfinished_source_pipeline();
  attach_concat_port(concat_op, *repo, source.pipeline);

  // Partition 0 holds a lone under-threshold batch: no group is ready there.
  auto p0_batch    = make_int32_batch(*space, 100);
  const auto p0_id = p0_batch->get_batch_id();
  repo->add_data_batch(std::move(p0_batch), 0);

  // Partition 1 holds three 400-byte batches: the third crosses the threshold, so the first two
  // form a ready group.
  std::vector<uint64_t> p1_ids;
  for (int i = 0; i < 3; ++i) {
    auto batch = make_int32_batch(*space, 100);
    p1_ids.push_back(batch->get_batch_id());
    repo->add_data_batch(std::move(batch), 1);
  }

  auto ready_hint = concat_op.get_next_task_hint();
  REQUIRE(ready_hint.has_value());
  REQUIRE(ready_hint->hint == TaskCreationHint::READY);
  REQUIRE(ready_hint->producer == &concat_op);

  // The pull must skip the not-ready partition 0 and release partition 1's group.
  auto group = concat_op.get_next_task_input_data();
  REQUIRE(group != nullptr);
  const auto& group_data = dynamic_cast<const partitioned_operator_data&>(*group);
  REQUIRE(group_data.get_partition_idx() == std::optional<std::size_t>{1});
  const auto& group_batches = group_data.get_data_batches();
  REQUIRE(group_batches.size() == 2);
  REQUIRE(group_batches[0]->get_batch_id() == p1_ids[0]);
  REQUIRE(group_batches[1]->get_batch_id() == p1_ids[1]);

  // Both partitions now hold under-threshold tails: nothing further until the source finishes.
  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
  REQUIRE(repo->get_batch_ids(0) == std::vector<uint64_t>{p0_id});
  REQUIRE(repo->get_batch_ids(1) == std::vector<uint64_t>{p1_ids[2]});

  // Pipeline finish flushes the tails in partition order.
  source.pipeline->set_finished(true);
  auto tail0 = concat_op.get_next_task_input_data();
  REQUIRE(tail0 != nullptr);
  REQUIRE(dynamic_cast<const partitioned_operator_data&>(*tail0).get_partition_idx() ==
          std::optional<std::size_t>{0});
  auto tail1 = concat_op.get_next_task_input_data();
  REQUIRE(tail1 != nullptr);
  REQUIRE(dynamic_cast<const partitioned_operator_data&>(*tail1).get_partition_idx() ==
          std::optional<std::size_t>{1});
  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
}

TEST_CASE("sirius_physical_concat keeps a batch exactly at the threshold in its group",
          "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  // Size the threshold to exactly match one batch: the strict '>' comparison keeps that batch in
  // the accumulating group instead of overflowing on its own.
  auto exact_batch         = make_int32_batch(*space, 256);
  const auto exact_id      = exact_batch->get_batch_id();
  const uint64_t threshold = exact_batch->to_read_only().get_data()->get_size_in_bytes();

  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false,
    threshold);
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

  auto repo   = std::make_unique<cucascade::shared_data_repository>();
  auto source = create_unfinished_source_pipeline();
  attach_concat_port(concat_op, *repo, source.pipeline);
  repo->add_data_batch(std::move(exact_batch), 0);

  // Exactly at the threshold is not over it: the batch keeps accumulating.
  REQUIRE(concat_op.get_next_task_input_data() == nullptr);
  auto waiting_hint = concat_op.get_next_task_hint();
  REQUIRE(waiting_hint.has_value());
  REQUIRE(waiting_hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);

  // The next batch overflows the group, releasing the exact-threshold batch alone.
  auto follower          = make_int32_batch(*space, 100);
  const auto follower_id = follower->get_batch_id();
  repo->add_data_batch(std::move(follower), 0);

  auto ready_hint = concat_op.get_next_task_hint();
  REQUIRE(ready_hint.has_value());
  REQUIRE(ready_hint->hint == TaskCreationHint::READY);

  auto group = concat_op.get_next_task_input_data();
  REQUIRE(group != nullptr);
  const auto& group_batches =
    dynamic_cast<const pipelineable_operator_data&>(*group).get_data_batches();
  REQUIRE(group_batches.size() == 1);
  REQUIRE(group_batches[0]->get_batch_id() == exact_id);
  REQUIRE(repo->get_batch_ids(0) == std::vector<uint64_t>{follower_id});
}

//===----------------------------------------------------------------------===//
// 3. Constructor tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_hash_join identifies right-family joins", "[physical_concat]")
{
  for (auto join_type :
       {duckdb::JoinType::RIGHT, duckdb::JoinType::RIGHT_ANTI, duckdb::JoinType::RIGHT_SEMI}) {
    INFO("join_type=" << duckdb::JoinTypeToString(join_type));
    auto fixture = create_test_hash_join(join_type, {duckdb::LogicalType::INTEGER});
    REQUIRE(fixture.hash_join->is_right_family());
  }

  for (auto join_type : {duckdb::JoinType::INNER,
                         duckdb::JoinType::LEFT,
                         duckdb::JoinType::SEMI,
                         duckdb::JoinType::ANTI,
                         duckdb::JoinType::MARK,
                         duckdb::JoinType::OUTER}) {
    INFO("join_type=" << duckdb::JoinTypeToString(join_type));
    auto fixture = create_test_hash_join(join_type, {duckdb::LogicalType::INTEGER});
    REQUIRE_FALSE(fixture.hash_join->is_right_family());
  }
}

TEST_CASE("right-family sibling partitions round up from the probe input", "[physical_partition]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  auto build_batch =
    make_numeric_batch<int32_t>(*space, std::vector<int32_t>(4, 1), cudf::type_id::INT32);
  auto probe_batch =
    make_numeric_batch<int32_t>(*space, std::vector<int32_t>(9, 1), cudf::type_id::INT32);
  auto const build_bytes = build_batch->to_read_only().get_data()->get_size_in_bytes();
  auto const probe_bytes = probe_batch->to_read_only().get_data()->get_size_in_bytes();
  REQUIRE(probe_bytes > build_bytes);
  auto const partition_size = probe_bytes - 1;

  // The join owns hash_partition_bytes (the natural-count divisor) now, so it must be constructed
  // with partition_size for the probe side to size to two partitions.
  auto fixture =
    create_test_hash_join(duckdb::JoinType::RIGHT, {duckdb::LogicalType::INTEGER}, partition_size);
  auto make_types = [] {
    return sirius::from_duckdb_vec(
      duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  };
  sirius_physical_partition build_partition(
    make_types(), 4, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, true));
  sirius_physical_partition probe_partition(
    make_types(), 9, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, false));
  // These partitions hang off the join by pointer, not as children, so create_test_hash_join's
  // numbering does not reach them. Number them past the join tree's ids.
  size_t partition_next_id = 100;
  number_operator_tree(build_partition, partition_next_id);
  number_operator_tree(probe_partition, partition_next_id);

  auto build_repo = std::make_unique<cucascade::shared_data_repository>();
  auto probe_repo = std::make_unique<cucascade::shared_data_repository>();
  build_repo->add_data_batch(std::move(build_batch), 0);
  probe_repo->add_data_batch(std::move(probe_batch), 0);

  // The join's per-partition input repos that the sizing decision pre-sizes inside
  // sirius_physical_hash_join::get_partition_strategy (created during pipeline construction in
  // production): the build side targets the join's "build" port, the probe side its "default" port.
  auto join_build_repo   = std::make_unique<cucascade::shared_data_repository>();
  auto join_default_repo = std::make_unique<cucascade::shared_data_repository>();

  auto attach_port = [](sirius_physical_operator& op,
                        std::string_view port_id,
                        cucascade::shared_data_repository& repo) {
    auto port           = std::make_unique<sirius_physical_operator::port>();
    port->type          = MemoryBarrierType::FULL;
    port->repo          = &repo;
    port->src_pipeline  = nullptr;
    port->dest_pipeline = nullptr;
    op.add_port(port_id, std::move(port));
  };
  attach_port(build_partition, "default", *build_repo);
  attach_port(probe_partition, "default", *probe_repo);
  attach_port(*fixture.hash_join, "build", *join_build_repo);
  attach_port(*fixture.hash_join, "default", *join_default_repo);

  // Enter through the non-driving build side first. It must still size both siblings from probe.
  bind_partition_inputs(*fixture.hash_join, build_partition, probe_partition, nullptr, true);
  auto build_input = build_partition.get_next_task_input_data();
  REQUIRE(build_input != nullptr);
  auto build_output = build_partition.execute(*build_input, default_stream());
  REQUIRE(
    dynamic_cast<const pipelineable_operator_data&>(*build_output).get_data_batches().size() == 2);

  auto probe_input = probe_partition.get_next_task_input_data();
  REQUIRE(probe_input != nullptr);
  auto probe_output = probe_partition.execute(*probe_input, default_stream());
  REQUIRE(
    dynamic_cast<const pipelineable_operator_data&>(*probe_output).get_data_batches().size() == 2);
}

TEST_CASE("sibling partitions install one placement on the join and both CONCATs",
          "[physical_partition]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  auto build_batch =
    make_numeric_batch<int32_t>(*space, std::vector<int32_t>(4, 1), cudf::type_id::INT32);
  auto probe_batch =
    make_numeric_batch<int32_t>(*space, std::vector<int32_t>(9, 1), cudf::type_id::INT32);
  auto const probe_bytes = probe_batch->to_read_only().get_data()->get_size_in_bytes();

  // A right join sized by its probe side into two partitions (as in the test above).
  auto fixture = create_test_hash_join(
    duckdb::JoinType::RIGHT, {duckdb::LogicalType::INTEGER}, /*partition_size=*/probe_bytes - 1);
  fixture.hash_join->set_active_gpu_ids({space->get_device_id()});
  auto make_types = [] {
    return sirius::from_duckdb_vec(
      duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  };
  sirius_physical_partition build_partition(
    make_types(), 4, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, true));
  sirius_physical_partition probe_partition(
    make_types(), 9, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, false));
  sirius_physical_concat build_concat(make_types(), 4, false);
  sirius_physical_concat probe_concat(make_types(), 9, true);
  size_t next_id = 100;
  number_operator_tree(build_partition, next_id);
  number_operator_tree(probe_partition, next_id);
  number_operator_tree(build_concat, next_id);
  number_operator_tree(probe_concat, next_id);
  build_partition.add_next_port_after_sink({&build_concat, "input"});
  probe_partition.add_next_port_after_sink({&probe_concat, "input"});

  cucascade::shared_data_repository build_repo;
  cucascade::shared_data_repository probe_repo;
  cucascade::shared_data_repository join_build_repo;
  cucascade::shared_data_repository join_default_repo;
  build_repo.add_data_batch(std::move(build_batch), 0);
  probe_repo.add_data_batch(std::move(probe_batch), 0);
  auto attach_port = [](sirius_physical_operator& op,
                        std::string_view port_id,
                        cucascade::shared_data_repository& repo) {
    auto port          = std::make_unique<sirius_physical_operator::port>();
    port->type         = MemoryBarrierType::FULL;
    port->repo         = &repo;
    port->src_pipeline = nullptr;
    op.add_port(port_id, std::move(port));
  };
  attach_port(build_partition, "default", build_repo);
  attach_port(probe_partition, "default", probe_repo);
  attach_port(*fixture.hash_join, "build", join_build_repo);
  attach_port(*fixture.hash_join, "default", join_default_repo);

  REQUIRE_THROWS_AS(fixture.hash_join->placement(), sirius::internal_exception);
  bind_partition_inputs(*fixture.hash_join, build_partition, probe_partition, &build_concat, true);
  REQUIRE(build_partition.get_next_task_input_data() != nullptr);

  auto const placement = fixture.hash_join->placement();
  REQUIRE(placement != nullptr);
  CHECK(*placement == partition_placement::round_robin(2, {space->get_device_id()}));
  // Every emitter of this exchange shares the one object the join chose.
  CHECK(build_concat.placement() == placement);
  CHECK(probe_concat.placement() == placement);

  // The CONCAT stamps the partition's device on what it emits.
  cucascade::shared_data_repository concat_repo;
  concat_repo.set_num_partitions(2);
  concat_repo.add_data_batch(make_int32_batch(*space, 4), 1);
  build_concat.set_concat_all(true);
  attach_concat_port(build_concat, concat_repo);
  auto concat_input = build_concat.get_next_task_input_data();
  REQUIRE(concat_input != nullptr);
  auto const& partitioned = dynamic_cast<const partitioned_operator_data&>(*concat_input);
  CHECK(partitioned.get_partition_idx() == std::optional<std::size_t>{1});
  CHECK(partitioned.get_preferred_device_id() == std::optional<int>{space->get_device_id()});
}

TEST_CASE("an empty sizing side installs a single-partition placement", "[physical_partition]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  fixture.hash_join->set_active_gpu_ids({space->get_device_id()});
  auto make_types = [] {
    return sirius::from_duckdb_vec(
      duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  };
  // The build side drives sizing; its producer finishes without delivering anything.
  sirius_physical_partition build_partition(
    make_types(), 4, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, true));
  sirius_physical_partition probe_partition(
    make_types(), 9, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, false));
  sirius_physical_concat probe_concat(make_types(), 9, false);
  size_t next_id = 100;
  number_operator_tree(build_partition, next_id);
  number_operator_tree(probe_partition, next_id);
  number_operator_tree(probe_concat, next_id);
  probe_partition.add_next_port_after_sink({&probe_concat, "input"});

  auto finished_build_source = create_unfinished_source_pipeline();
  finished_build_source.pipeline->set_finished(true);
  cucascade::shared_data_repository build_repo;
  cucascade::shared_data_repository probe_repo;
  auto build_port          = std::make_unique<sirius_physical_operator::port>();
  build_port->type         = MemoryBarrierType::FULL;
  build_port->repo         = &build_repo;
  build_port->src_pipeline = finished_build_source.pipeline;
  build_partition.add_port("default", std::move(build_port));
  auto probe_port          = std::make_unique<sirius_physical_operator::port>();
  probe_port->type         = MemoryBarrierType::FULL;
  probe_port->repo         = &probe_repo;
  probe_port->src_pipeline = nullptr;
  probe_partition.add_port("default", std::move(probe_port));

  // The waiting probe side elects one partition without asking the join (a zero-byte build would
  // otherwise pick BUILD_PROBE, which cannot run without a build batch).
  bind_partition_inputs(*fixture.hash_join, build_partition, probe_partition, nullptr);
  CHECK_FALSE(probe_partition.get_next_task_hint().has_value());
  auto const placement = fixture.hash_join->placement();
  REQUIRE(placement != nullptr);
  CHECK(*placement == partition_placement::unpinned(1));
  CHECK(probe_concat.placement() == placement);
  CHECK_FALSE(fixture.hash_join->is_build_probe_mode());

  // The surviving side's CONCAT retains its partition index without forcing a GPU.
  cucascade::shared_data_repository concat_repo;
  concat_repo.add_data_batch(make_int32_batch(*space, 4), 0);
  probe_concat.set_concat_all(true);
  attach_concat_port(probe_concat, concat_repo);
  auto concat_input = probe_concat.get_next_task_input_data();
  REQUIRE(concat_input != nullptr);
  auto const& partitioned = dynamic_cast<const partitioned_operator_data&>(*concat_input);
  CHECK(partitioned.get_partition_idx() == std::optional<std::size_t>{0});
  CHECK_FALSE(partitioned.get_preferred_device_id().has_value());
}

//===----------------------------------------------------------------------===//
// 4. Multithreading tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_concat get_next_task_input_batch is thread-safe", "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  // Use a small threshold to force multiple get_next_task_input_batch calls
  constexpr uint64_t threshold = 1024;  // 1 KB

  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    false,
    threshold);

  // Set up a port with a data repository containing many batches across partitions
  auto repo = std::make_unique<cucascade::shared_data_repository>();

  constexpr int num_batches_per_partition = 20;
  constexpr int num_partitions            = 5;
  concat_op.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(num_partitions)));
  constexpr std::size_t rows_per_batch = 500;
  int total_batches                    = num_batches_per_partition * num_partitions;

  std::set<uint64_t> expected_batch_ids;
  for (int p = 0; p < num_partitions; ++p) {
    for (int b = 0; b < num_batches_per_partition; ++b) {
      std::vector<int32_t> values(rows_per_batch);
      std::iota(values.begin(),
                values.end(),
                static_cast<int32_t>((p * num_batches_per_partition + b) * rows_per_batch));
      auto batch = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
      expected_batch_ids.insert(batch->get_batch_id());
      repo->add_data_batch(std::move(batch), static_cast<size_t>(p));
    }
  }

  auto port           = std::make_unique<sirius_physical_operator::port>();
  port->type          = MemoryBarrierType::FULL;
  port->repo          = repo.get();
  port->src_pipeline  = nullptr;
  port->dest_pipeline = nullptr;
  concat_op.add_port("input", std::move(port));

  // Launch multiple threads each pulling batches
  constexpr int num_threads = 8;
  std::mutex collected_mutex;
  std::vector<uint64_t> collected_batch_ids;
  std::atomic<int> total_calls{0};

  auto worker = [&]() {
    while (true) {
      auto result = concat_op.get_next_task_input_data();
      if (!result) { break; }
      total_calls.fetch_add(1, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lg(collected_mutex);
      for (auto& batch :
           dynamic_cast<const pipelineable_operator_data&>(*result).get_data_batches()) {
        if (batch) { collected_batch_ids.push_back(batch->get_batch_id()); }
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker);
  }
  for (auto& t : threads) {
    t.join();
  }

  // Verify: all batches consumed exactly once
  REQUIRE(collected_batch_ids.size() == static_cast<std::size_t>(total_batches));

  // Check no duplicates
  std::set<uint64_t> collected_set(collected_batch_ids.begin(), collected_batch_ids.end());
  REQUIRE(collected_set.size() == collected_batch_ids.size());

  // Check all expected IDs are present
  REQUIRE(collected_set == expected_batch_ids);
}

TEST_CASE("sirius_physical_concat execute is thread-safe with independent streams",
          "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});

  constexpr int num_threads            = 4;
  constexpr std::size_t rows_per_batch = 200;
  constexpr int batches_per_thread     = 3;

  // Streams must outlive both input and output buffers, which retain their deallocation stream.
  std::vector<std::unique_ptr<rmm::cuda_stream>> streams;
  for (int t = 0; t < num_threads; ++t) {
    streams.push_back(std::make_unique<rmm::cuda_stream>());
  }

  // Pre-create input batches for each thread
  std::vector<std::vector<std::shared_ptr<data_batch>>> thread_inputs(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    for (int b = 0; b < batches_per_thread; ++b) {
      std::vector<int32_t> values(rows_per_batch);
      std::iota(values.begin(),
                values.end(),
                static_cast<int32_t>((t * batches_per_thread + b) * rows_per_batch));
      auto batch = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
      thread_inputs[t].push_back(std::move(batch));
    }
  }

  // Each thread gets its own concat operator and CUDA stream
  std::vector<std::vector<std::shared_ptr<data_batch>>> thread_outputs(num_threads);
  std::mutex error_mutex;
  std::string error_msg;

  auto worker = [&](int thread_id) {
    try {
      sirius_physical_concat concat_op(
        sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
        1000,
        false);
      concat_op.set_placement(
        std::make_shared<const partition_placement>(partition_placement::unpinned(1)));

      ::cuda::stream_ref stream(streams[thread_id]->value());

      cucascade::shared_data_repository repo;
      auto input = claim_concat_input(concat_op, repo, thread_inputs[thread_id], 0);
      input->prepare_for_processing(space, stream);
      auto outputs = concat_op.execute(*input, stream);

      // Synchronize the stream before accessing results
      streams[thread_id]->synchronize();

      thread_outputs[thread_id] =
        dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches();

    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lg(error_mutex);
      error_msg = e.what();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& t : threads) {
    t.join();
  }

  // Check no errors occurred
  REQUIRE(error_msg.empty());

  // Verify each thread's output
  for (int t = 0; t < num_threads; ++t) {
    REQUIRE(thread_outputs[t].size() == 1);
    auto out_table = sirius::get_cudf_table_view(*thread_outputs[t][0]);
    REQUIRE(static_cast<std::size_t>(out_table.num_rows()) == rows_per_batch * batches_per_thread);
  }
}

TEST_CASE(
  "complete CONCAT publishes one ordered batch per nonempty partition under concurrent claims",
  "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat(types, 0, true, 1);
  sirius_physical_concat receiver(types, 0, true);
  concat.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(6)));
  cucascade::shared_data_repository input_repo;
  cucascade::shared_data_repository output_repo;
  input_repo.set_num_partitions(6);
  output_repo.set_num_partitions(6);
  sirius_physical_operator producer(SiriusPhysicalOperatorType::PROJECTION, types, 0);
  auto source = std::make_shared<sirius::pipeline::sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{nullptr});
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*source, producer);
  build_state.set_pipeline_sink(*source, &producer, 0);
  build_state.add_pipeline_operator(*source, producer);
  source->mark_task_created();
  source->mark_task_created();
  auto port          = std::make_unique<sirius_physical_operator::port>();
  port->type         = MemoryBarrierType::FULL;
  port->repo         = &input_repo;
  port->src_pipeline = source;
  concat.add_port("input", std::move(port));
  attach_concat_port(receiver, output_repo);
  concat.add_next_port_after_sink({&receiver, "input"});
  auto singleton = make_numeric_batch<int32_t>(*space, {91, 92}, cudf::type_id::INT32);
  input_repo.add_data_batch(singleton, 4);
  input_repo.add_data_batch(make_numeric_batch<int32_t>(*space, {3, 1}, cudf::type_id::INT32), 1);
  source->mark_task_completed();
  source->update_pipeline_status(false);
  CHECK_FALSE(source->is_pipeline_finished());
  // Even an oversized ready-looking group cannot escape while publication is outstanding.
  CHECK(concat.get_next_task_input_data() == nullptr);
  input_repo.add_data_batch(make_numeric_batch<int32_t>(*space, {7, 7, 2}, cudf::type_id::INT32),
                            1);
  CHECK(concat.get_next_task_input_data() == nullptr);
  source->mark_task_completed();
  source->update_pipeline_status(false);
  REQUIRE(source->is_pipeline_finished());
  REQUIRE(concat.get_next_task_hint()->hint == TaskCreationHint::READY);

  std::mutex results_mutex;
  std::vector<std::unique_ptr<operator_data>> claims;
  std::vector<std::thread> claimers;
  for (int i = 0; i < 4; ++i) {
    claimers.emplace_back([&] {
      while (auto data = concat.get_next_task_input_data()) {
        std::lock_guard guard(results_mutex);
        claims.push_back(std::move(data));
      }
    });
  }
  for (auto& thread : claimers) {
    thread.join();
  }
  REQUIRE(claims.size() == 2);
  CHECK(input_repo.total_size() == 0);
  for (auto const& claim : claims) {
    claim->prepare_for_processing(space, default_stream());
    auto output = concat.execute(*claim, default_stream());
    concat.sink(*output, default_stream());
  }
  CHECK(output_repo.total_size() == 2);
  for (std::size_t idx : {0UL, 2UL, 3UL, 5UL}) {
    CHECK(output_repo.size(idx) == 0);
  }
  REQUIRE(output_repo.size(1) == 1);
  REQUIRE(output_repo.size(4) == 1);
  auto const merged = output_repo.get_data_batch_by_id(output_repo.get_batch_ids(1).front(), 1);
  auto const ro     = merged->to_read_only();
  CHECK(copy_column_to_host<int32_t>(sirius::get_cudf_table_view(ro).column(0)) ==
        std::vector<int32_t>{3, 1, 7, 7, 2});
  CHECK(output_repo.get_data_batch_by_id(output_repo.get_batch_ids(4).front(), 4) == singleton);
  CHECK_FALSE(concat.get_next_task_hint().has_value());
}

TEST_CASE("CONCAT rejects changing grouping after any input claim", "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat(types, 0, false, 1);
  concat.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));
  cucascade::shared_data_repository repo;
  attach_concat_port(concat, repo);
  // An unsuccessful claim leaves configuration open.
  CHECK(concat.get_next_task_input_data() == nullptr);
  REQUIRE_NOTHROW(concat.set_concat_all(true));
  REQUIRE_NOTHROW(concat.set_concat_all(false));
  repo.add_data_batch(make_int32_batch(*space, 4), 0);
  repo.add_data_batch(make_int32_batch(*space, 4), 0);
  REQUIRE(concat.get_next_task_input_data() != nullptr);
  CHECK(repo.size(0) == 1);
  CHECK_THROWS_AS(concat.set_concat_all(true), std::logic_error);
  CHECK_NOTHROW(concat.set_concat_all(false));
}

TEST_CASE("either partition sibling fixes runtime CONCAT requirements before publishing",
          "[physical_partition][physical_concat]")
{
  bool const probe_initiates = GENERATE(false, true);
  bool const mixed_filter    = GENERATE(false, true);
  auto* space                = get_shared_mem_space();
  auto fixture               = create_test_hash_join(duckdb::JoinType::INNER,
                                                     {duckdb::LogicalType::INTEGER},
                                       sirius::config::DEFAULT_HASH_PARTITION_BYTES,
                                       mixed_filter);
  fixture.hash_join->set_active_gpu_ids({space->get_device_id()});
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_partition build(
    types, 4, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, true));
  sirius_physical_partition probe(
    types, 9, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, false));
  sirius_physical_concat build_concat(types, 4, false, 1);
  sirius_physical_concat probe_concat(types, 9, false, 1);
  build.add_next_port_after_sink({&build_concat, "input"});
  probe.add_next_port_after_sink({&probe_concat, "input"});
  cucascade::shared_data_repository build_repo;
  cucascade::shared_data_repository probe_repo;
  auto attach = [](sirius_physical_operator& op, cucascade::shared_data_repository& repo) {
    auto port  = std::make_unique<sirius_physical_operator::port>();
    port->type = MemoryBarrierType::FULL;
    port->repo = &repo;
    op.add_port("default", std::move(port));
  };
  attach(build, build_repo);
  attach(probe, probe_repo);
  build_repo.add_data_batch(make_int32_batch(*space, 4), 0);
  probe_repo.add_data_batch(make_int32_batch(*space, 9), 0);
  size_t id = 100;
  number_operator_tree(build, id);
  number_operator_tree(probe, id);
  bind_partition_inputs(*fixture.hash_join, build, probe, &build_concat);
  auto& first = probe_initiates ? probe : build;
  REQUIRE(first.get_next_task_input_data() != nullptr);
  CHECK(fixture.hash_join->is_build_probe_mode() == !mixed_filter);

  // Model fragmented partition publication after negotiation, before producer completion.
  cucascade::shared_data_repository concat_repo;
  auto producer = create_unfinished_source_pipeline();
  attach_concat_port(build_concat, concat_repo, producer.pipeline);
  concat_repo.add_data_batch(make_int32_batch(*space, 2), 0);
  concat_repo.add_data_batch(make_int32_batch(*space, 2), 0);
  CHECK(build_concat.get_next_task_input_data() == nullptr);
  producer.pipeline->set_finished(true);
  auto claim = build_concat.get_next_task_input_data();
  REQUIRE(claim != nullptr);
  CHECK(dynamic_cast<pipelineable_operator_data&>(*claim).get_data_batches().size() == 2);
  CHECK(concat_repo.total_size() == 0);
}

TEST_CASE("complete CONCAT uses normal cross-device input preparation",
          "[physical_concat][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) {
    SUCCEED("requires two GPUs");
    return;
  }
  auto manager = sirius::test::operator_utils::initialize_memory_manager(2);
  auto* source = manager->get_memory_space(Tier::GPU, 0);
  auto* target = manager->get_memory_space(Tier::GPU, 1);
  std::vector<std::shared_ptr<data_batch>> batches;
  {
    rmm::cuda_set_device_raii device(rmm::cuda_device_id{0});
    batches.push_back(make_numeric_batch<int32_t>(*source, {4, 2}, cudf::type_id::INT32));
  }
  rmm::cuda_set_device_raii device(rmm::cuda_device_id{1});
  batches.push_back(make_numeric_batch<int32_t>(*target, {8, 1}, cudf::type_id::INT32));
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat(types, 0, true);
  concat.set_placement(
    std::make_shared<const partition_placement>(partition_placement::round_robin(3, {1})));
  cucascade::shared_data_repository repo;
  auto input = claim_concat_input(concat, repo, std::move(batches), 2);
  input->prepare_for_processing(target, default_stream());
  auto output         = concat.execute(*input, default_stream());
  auto const& indexed = dynamic_cast<partitioned_operator_data const&>(*output);
  CHECK(indexed.get_partition_idx() == 2);
  REQUIRE(indexed.get_data_batches().size() == 1);
  auto const ro = indexed.get_data_batches()[0]->to_read_only();
  CHECK(ro.get_memory_space() == target);
  CHECK(copy_column_to_host<int32_t>(sirius::get_cudf_table_view(ro).column(0)) ==
        std::vector<int32_t>{4, 2, 8, 1});
}

TEST_CASE("complete CONCAT checks known row counts before claiming or transferring input",
          "[physical_concat]")
{
  bool const overflow = GENERATE(false, true);
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat(types, 0, true);
  concat.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(3)));
  cucascade::shared_data_repository repo;
  std::vector<std::shared_ptr<data_batch>> batches;
  // Neither case may execute against these metadata-only payloads.
  auto const first_rows = std::numeric_limits<cudf::size_type>::max() - (overflow ? 0 : 1);
  for (auto const rows : {first_rows, cudf::size_type{1}}) {
    batches.push_back(make_host_metadata_batch(rows));
  }
  attach_concat_port(concat, repo);
  for (auto const& batch : batches) {
    repo.add_data_batch(batch, 2);
  }
  if (overflow) {
    CHECK_THROWS_AS(concat.get_next_task_input_data(), std::overflow_error);
    CHECK(repo.total_size() == batches.size());
    CHECK_NOTHROW(concat.set_concat_all(false));
  } else {
    auto input = concat.get_next_task_input_data();
    REQUIRE(input != nullptr);
    CHECK(repo.total_size() == 0);
    CHECK(dynamic_cast<pipelineable_operator_data&>(*input).get_data_batches() == batches);
    CHECK_THROWS_AS(concat.set_concat_all(false), std::logic_error);
  }
  for (auto const& batch : batches) {
    CHECK(batch->to_read_only().get_memory_space()->get_tier() == Tier::HOST);
  }
}

TEST_CASE("bounded CONCAT hints and claims split groups at the row-count limit",
          "[physical_concat]")
{
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat(types, 0, false, std::numeric_limits<uint64_t>::max());
  concat.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));
  cucascade::shared_data_repository repo;
  auto producer = create_unfinished_source_pipeline();
  attach_concat_port(concat, repo, producer.pipeline);
  auto first  = make_host_metadata_batch(std::numeric_limits<cudf::size_type>::max() - 1);
  auto second = make_host_metadata_batch(1);
  auto tail   = make_host_metadata_batch(1);
  repo.add_data_batch(first, 0);
  repo.add_data_batch(second, 0);
  CHECK(concat.get_next_task_input_data() == nullptr);
  repo.add_data_batch(tail, 0);
  auto hint = concat.get_next_task_hint();
  REQUIRE(hint.has_value());
  CHECK(hint->hint == TaskCreationHint::READY);
  auto input = concat.get_next_task_input_data();
  REQUIRE(input != nullptr);
  CHECK(dynamic_cast<pipelineable_operator_data&>(*input).get_data_batches() ==
        std::vector<std::shared_ptr<data_batch>>{first, second});
  CHECK(repo.total_size() == 1);
  CHECK(concat.get_next_task_input_data() == nullptr);
  producer.pipeline->set_finished(true);
  auto final_input = concat.get_next_task_input_data();
  REQUIRE(final_input != nullptr);
  CHECK(dynamic_cast<pipelineable_operator_data&>(*final_input).get_data_batches() ==
        std::vector<std::shared_ptr<data_batch>>{tail});
}

TEST_CASE("CONCAT execute validates row counts on plain partitioned input", "[physical_concat]")
{
  bool const overflow = GENERATE(false, true);
  auto* space         = get_shared_mem_space();
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat(types, 0, true);
  concat.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));
  auto owner         = make_int32_batch(*space, 1);
  auto const ro      = owner->to_read_only();
  auto const pointer = sirius::get_cudf_table_view(ro).column(0).data<int32_t>();
  cudf::column_view column(cudf::data_type{cudf::type_id::INT32},
                           std::numeric_limits<cudf::size_type>::max(),
                           pointer,
                           nullptr,
                           0);
  auto repr = std::make_unique<cucascade::gpu_table_representation>(
    cudf::table_view{{column}}, owner, sizeof(int32_t), *space, default_stream());
  auto large = data_batch::make(sirius::get_next_batch_id(), std::move(repr));
  std::vector<std::shared_ptr<data_batch>> batches{large};
  if (overflow) { batches.push_back(make_int32_batch(*space, 1)); }
  auto input = unpinned_partition(std::move(batches), 0);
  // Synthetic large views must never be read: only singleton forwarding or early rejection
  // may run. These calls intentionally bypass the CONCAT claim and preparation hooks.
  if (overflow) {
    CHECK_THROWS_AS(concat.execute(input, default_stream()), std::overflow_error);
  } else {
    auto output = concat.execute(input, default_stream());
    CHECK(dynamic_cast<partitioned_operator_data&>(*output).get_data_batches() ==
          std::vector<std::shared_ptr<data_batch>>{large});
  }
}

TEST_CASE("CONCAT execute rejects inputs from different memory spaces", "[physical_concat]")
{
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat(types, 0, true);
  concat.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));
  auto input = unpinned_partition(
    {make_int32_batch(*get_shared_mem_space(), 1), make_host_metadata_batch(1)}, 0);
  CHECK_THROWS_WITH(concat.execute(input, default_stream()),
                    "CONCAT input was not prepared on one memory space");
}

TEST_CASE("CONCAT can claim input without row-count metadata", "[physical_concat]")
{
  bool const concat_all = GENERATE(false, true);
  auto known            = make_host_metadata_batch(std::numeric_limits<cudf::size_type>::max());
  auto* host            = known->to_read_only().get_memory_space();
  // Packed representations expose bytes but no row count. Metadata-only input must not be
  // uploaded: selection may claim it, leaving exact validation to execute after decoding.
  auto table = std::make_unique<cucascade::memory::host_table_packed_allocation>(
    nullptr, std::make_unique<std::vector<uint8_t>>(), sizeof(int32_t));
  auto repr = std::make_unique<cucascade::host_data_packed_representation>(std::move(table), host);
  auto unknown = data_batch::make(sirius::get_next_batch_id(), std::move(repr));
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat(types, 0, concat_all);
  concat.set_placement(
    std::make_shared<const partition_placement>(partition_placement::unpinned(1)));
  cucascade::shared_data_repository repo;
  auto input = claim_concat_input(concat, repo, {known, unknown}, 0);
  CHECK(dynamic_cast<partitioned_operator_data&>(*input).get_data_batches() ==
        std::vector<std::shared_ptr<data_batch>>{known, unknown});
  CHECK(unknown->to_read_only().get_memory_space() == host);
}

TEST_CASE("partitioned input replacement preserves partition and device metadata",
          "[physical_concat][operator_data]")
{
  bool const indexed = GENERATE(false, true);
  bool const pinned  = GENERATE(false, true);
  auto original      = make_host_metadata_batch(3);
  auto replacement   = make_host_metadata_batch(3);
  std::unique_ptr<partitioned_operator_data> input;
  if (indexed) {
    auto layout =
      pinned ? partition_placement::round_robin(4, {2, 5}) : partition_placement::unpinned(4);
    input = std::make_unique<partitioned_operator_data>(
      std::vector<std::shared_ptr<data_batch>>{original}, 3, layout);
  } else {
    input = std::make_unique<partitioned_operator_data>(
      std::vector<std::shared_ptr<data_batch>>{original});
    if (pinned) { input->set_preferred_device_id(5); }
  }
  auto rebuilt      = input->with_batches_preserving_rows({replacement});
  auto* partitioned = dynamic_cast<partitioned_operator_data*>(rebuilt.get());
  REQUIRE(partitioned != nullptr);
  CHECK(partitioned->get_partition_idx() == input->get_partition_idx());
  CHECK(partitioned->get_preferred_device_id() == input->get_preferred_device_id());
  CHECK(partitioned->get_data_batches() == std::vector<std::shared_ptr<data_batch>>{replacement});
  CHECK(input->get_data_batches() == std::vector<std::shared_ptr<data_batch>>{original});
}

TEST_CASE("BUILD_PROBE rejects a missing explicit build concatenation path",
          "[physical_partition][physical_concat]")
{
  bool const probe_initiates = GENERATE(false, true);
  // MARK requires BUILD_PROBE even when the foldable capability is absent.
  auto fixture = create_test_hash_join(duckdb::JoinType::MARK, {duckdb::LogicalType::BOOLEAN});
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  sirius_physical_partition build(
    types, 0, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, true));
  sirius_physical_partition probe(
    types, 0, sirius::test::operator_utils::hash_partition_spec(*fixture.hash_join, false));
  cucascade::shared_data_repository build_repo;
  cucascade::shared_data_repository probe_repo;
  auto attach = [](sirius_physical_operator& op, cucascade::shared_data_repository& repo) {
    auto port  = std::make_unique<sirius_physical_operator::port>();
    port->type = MemoryBarrierType::FULL;
    port->repo = &repo;
    op.add_port("default", std::move(port));
  };
  attach(build, build_repo);
  attach(probe, probe_repo);
  auto* space = get_shared_mem_space();
  build_repo.add_data_batch(make_int32_batch(*space, 1));
  size_t id = 100;
  number_operator_tree(build, id);
  number_operator_tree(probe, id);
  bind_partition_inputs(*fixture.hash_join, build, probe, nullptr);
  auto& first = probe_initiates ? probe : build;
  CHECK_THROWS_WITH(
    first.get_next_task_input_data(),
    Catch::Matchers::ContainsSubstring("BUILD_PROBE requires a bound build CONCAT"));
}
