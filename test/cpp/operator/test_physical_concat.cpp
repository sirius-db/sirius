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
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/sirius_physical_concat.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_partition.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
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

//===----------------------------------------------------------------------===//
// Hash join fixture for constructing sirius_physical_concat
//===----------------------------------------------------------------------===//

/**
 * @brief Holds the LogicalComparisonJoin and hash join objects needed for
 * sirius_physical_concat construction. The logical_join must outlive the
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
  uint64_t hash_partition_bytes = sirius::config::DEFAULT_HASH_PARTITION_BYTES)
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
  duckdb::JoinCondition cond;
  cond.left  = duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.right = duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  conditions.push_back(std::move(cond));

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
    nullptr,                                                         // pushdown_info
    sirius::config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES,
    sirius::op::dynamic_filter_publish_plan{},  // dynamic_filter_plan
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
  static auto manager = sirius::test::operator_utils::initialize_memory_manager();
  return manager->get_memory_space(Tier::GPU, 0);
}

//===----------------------------------------------------------------------===//
// Hint / forward fixtures
//===----------------------------------------------------------------------===//

/**
 * @brief Pipeline stub with a settable is_pipeline_finished(), standing in for the concat's
 * source pipeline in hint and forward tests.
 */
class mock_gpu_pipeline : public sirius::pipeline::sirius_pipeline {
 public:
  explicit mock_gpu_pipeline(const sirius::pipeline::pipeline_build_context& ctx)
    : sirius_pipeline(ctx), _finished(false)
  {
  }

  void set_finished(bool finished) { _finished = finished; }

  bool is_pipeline_finished() const override { return _finished; }

 private:
  bool _finished;
};

/**
 * @brief Attach an "input" port backed by @p repo, optionally sourced from @p src_pipeline.
 */
void attach_input_port(sirius_physical_operator& op,
                       cucascade::shared_data_repository& repo,
                       duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> src_pipeline = nullptr)
{
  auto port           = std::make_unique<sirius_physical_operator::port>();
  port->type          = MemoryBarrierType::FULL;
  port->repo          = &repo;
  port->src_pipeline  = std::move(src_pipeline);
  port->dest_pipeline = nullptr;
  op.add_port("input", std::move(port));
}

//! Whether a rig's concat has a source pipeline and, if so, whether it reports finished.
enum class source_state { none, running, finished };

/**
 * @brief Everything a hint/forward test needs: the concat under test with its "input" port and
 * repository attached, plus (unless source_state::none) a mock source pipeline whose first
 * operator is the producer the WAITING hint must name. `repo` is declared before `op` so it
 * outlives the operator whose port holds a raw pointer to it.
 */
struct concat_test_rig {
  hash_join_test_fixture join;
  std::unique_ptr<cucascade::shared_data_repository> repo;
  std::unique_ptr<sirius_physical_concat> op;
  duckdb::shared_ptr<mock_gpu_pipeline> src_pipeline;
  duckdb::unique_ptr<sirius_physical_operator> upstream_producer;
};

concat_test_rig make_concat_rig(uint64_t threshold,
                                source_state source        = source_state::running,
                                duckdb::JoinType join_type = duckdb::JoinType::INNER,
                                bool is_build              = false)
{
  concat_test_rig rig;
  rig.join = create_test_hash_join(join_type, {duckdb::LogicalType::INTEGER});
  rig.op   = std::make_unique<sirius_physical_concat>(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    rig.join.hash_join.get(),
    is_build,
    threshold);
  rig.repo = std::make_unique<cucascade::shared_data_repository>();
  if (source != source_state::none) {
    const sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
    rig.src_pipeline = duckdb::make_shared_ptr<mock_gpu_pipeline>(build_ctx);
    rig.src_pipeline->set_finished(source == source_state::finished);
    rig.upstream_producer = duckdb::make_uniq<sirius_physical_operator>(
      SiriusPhysicalOperatorType::PROJECTION,
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1000);
    sirius::pipeline::sirius_pipeline_build_state build_state;
    build_state.add_pipeline_operator(*rig.src_pipeline, *rig.upstream_producer);
  }
  attach_input_port(*rig.op, *rig.repo, rig.src_pipeline);
  return rig;
}

/**
 * @brief A downstream partition consumer wired as a forward target, owning its "input" port repo.
 * `repo` is declared before `op` so it outlives the operator whose port points at it.
 */
struct downstream_rig {
  hash_join_test_fixture join;
  std::unique_ptr<cucascade::shared_data_repository> repo;
  std::unique_ptr<sirius_physical_concat> op;
};

downstream_rig make_downstream_rig()
{
  downstream_rig rig;
  rig.join = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  rig.op   = std::make_unique<sirius_physical_concat>(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    rig.join.hash_join.get(),
    false);
  rig.repo = std::make_unique<cucascade::shared_data_repository>();
  attach_input_port(*rig.op, *rig.repo);
  return rig;
}

std::shared_ptr<data_batch> make_int_batch(std::size_t num_rows)
{
  std::vector<int32_t> values(num_rows);
  std::iota(values.begin(), values.end(), 0);
  return make_numeric_batch<int32_t>(*get_shared_mem_space(), values, cudf::type_id::INT32);
}

uint64_t batch_bytes(const std::shared_ptr<data_batch>& batch)
{
  return batch->to_read_only().get_data()->get_size_in_bytes();
}

//! Batch ids of @p data in group order.
std::vector<uint64_t> group_batch_ids(const operator_data& data)
{
  std::vector<uint64_t> ids;
  for (auto& batch : dynamic_cast<const pipelineable_operator_data&>(data).get_data_batches()) {
    ids.push_back(batch->get_batch_id());
  }
  return ids;
}

//! Deref-safe hint accessor: fails the test instead of dereferencing a nullopt hint.
TaskCreationHint require_hint(sirius_physical_concat& op)
{
  auto hint = op.get_next_task_hint();
  REQUIRE(hint.has_value());
  return hint->hint;
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

  // Create hash join fixture and concat operator
  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {Traits::logical_type()});
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{Traits::logical_type()}),
    1000,
    fixture.hash_join.get(),
    false);

  // Execute
  auto outputs = concat_op.execute(partitioned_operator_data(input_batches, 0), default_stream());

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

  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
    false);

  auto outputs = concat_op.execute(partitioned_operator_data({input_batch}, 0), default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  // Single batch should be the same pointer (passthrough)
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0].get() ==
          input_batch.get());
}

TEST_CASE("sirius_physical_concat handles empty input", "[physical_concat]")
{
  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
    false);

  auto outputs = concat_op.execute(
    partitioned_operator_data(std::vector<std::shared_ptr<cucascade::data_batch>>{}, 0),
    default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().empty());
}

TEST_CASE("sirius_physical_concat filters null batches", "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  std::vector<int32_t> values1 = {1, 2, 3};
  std::vector<int32_t> values2 = {4, 5, 6};
  auto batch1                  = make_numeric_batch<int32_t>(*space, values1, cudf::type_id::INT32);
  auto batch2                  = make_numeric_batch<int32_t>(*space, values2, cudf::type_id::INT32);

  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
    false);

  // Mix valid and null batches
  std::vector<std::shared_ptr<data_batch>> input = {batch1, nullptr, batch2, nullptr};
  auto outputs = concat_op.execute(partitioned_operator_data(input, 0), default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  auto out_table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  REQUIRE(out_table.num_rows() == 6);

  auto host_data                = copy_column_to_host<int32_t>(out_table.column(0));
  std::vector<int32_t> expected = {1, 2, 3, 4, 5, 6};
  REQUIRE(host_data == expected);
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
  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
    false);

  // Create a downstream partition consumer operator to receive the sink output
  sirius_physical_concat downstream_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
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
  partitioned_operator_data sink_data({batch1, batch2}, partition_idx);
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

  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
    false);

  // Create two downstream operators
  sirius_physical_concat downstream1(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
    false);
  sirius_physical_concat downstream2(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
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
  partitioned_operator_data sink_data({batch}, partition_idx);
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

  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
    false,
    threshold);

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

  // LEFT join + is_build=true -> _concat_all = true
  auto fixture = create_test_hash_join(duckdb::JoinType::LEFT, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000,
    fixture.hash_join.get(),
    true,
    threshold);

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
// 3. Constructor tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_concat constructor sets concat_all for different join types",
          "[physical_concat]")
{
  SECTION("INNER join -> is_build_concat reflects is_build flag")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
    sirius_physical_concat concat_build(
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1000,
      fixture.hash_join.get(),
      true);
    REQUIRE(concat_build.is_build_concat() == true);

    sirius_physical_concat concat_probe(
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1000,
      fixture.hash_join.get(),
      false);
    REQUIRE(concat_probe.is_build_concat() == false);
  }

  SECTION("LEFT join + is_build=true -> is_build_concat returns true")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::LEFT, {duckdb::LogicalType::INTEGER});
    sirius_physical_concat concat_op(
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1000,
      fixture.hash_join.get(),
      true);
    REQUIRE(concat_op.is_build_concat() == true);
  }

  SECTION("LEFT join + is_build=false -> is_build_concat returns false")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::LEFT, {duckdb::LogicalType::INTEGER});
    sirius_physical_concat concat_op(
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1000,
      fixture.hash_join.get(),
      false);
    REQUIRE(concat_op.is_build_concat() == false);
  }

  SECTION("RIGHT join constructs successfully")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::RIGHT, {duckdb::LogicalType::INTEGER});
    REQUIRE_NOTHROW(sirius_physical_concat(
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1000,
      fixture.hash_join.get(),
      true));
  }

  SECTION("SEMI join constructs successfully")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::SEMI, {duckdb::LogicalType::INTEGER});
    REQUIRE_NOTHROW(sirius_physical_concat(
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1000,
      fixture.hash_join.get(),
      false));
  }

  SECTION("OUTER join throws unsupported join type")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::OUTER, {duckdb::LogicalType::INTEGER});
    REQUIRE_NOTHROW(sirius_physical_concat(
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1000,
      fixture.hash_join.get(),
      false));
  }

  SECTION("Non-hash-join parent throws")
  {
    sirius_physical_operator non_join_op(
      SiriusPhysicalOperatorType::PROJECTION,
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      1000);
    REQUIRE_THROWS_AS(
      sirius_physical_concat(
        sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
        1000,
        &non_join_op,
        false),
      std::runtime_error);
  }
}

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
  sirius_physical_partition build_partition(make_types(), 4, fixture.hash_join.get(), true);
  sirius_physical_partition probe_partition(make_types(), 9, fixture.hash_join.get(), false);
  // These partitions hang off the join by pointer, not as children, so create_test_hash_join's
  // numbering does not reach them. Number them past the join tree's ids.
  size_t partition_next_id = 100;
  number_operator_tree(build_partition, partition_next_id);
  number_operator_tree(probe_partition, partition_next_id);
  build_partition.set_sibling_partition_op(&probe_partition);
  probe_partition.set_sibling_partition_op(&build_partition);
  build_partition.set_drives_partition_count(false);
  probe_partition.set_drives_partition_count(true);

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
    fixture.hash_join.get(),
    false,
    threshold);

  // Set up a port with a data repository containing many batches across partitions
  auto repo = std::make_unique<cucascade::shared_data_repository>();

  constexpr int num_batches_per_partition = 20;
  constexpr int num_partitions            = 5;
  constexpr std::size_t rows_per_batch    = 500;
  int total_batches                       = num_batches_per_partition * num_partitions;

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
        fixture.hash_join.get(),
        false);

      // Create a dedicated CUDA stream for this thread
      cudaStream_t raw_stream;
      cudaStreamCreate(&raw_stream);
      rmm::cuda_stream_view stream(raw_stream);

      auto outputs =
        concat_op.execute(partitioned_operator_data(thread_inputs[thread_id], 0), default_stream());

      // Synchronize the stream before accessing results
      cudaStreamSynchronize(raw_stream);

      thread_outputs[thread_id] =
        dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches();

      cudaStreamDestroy(raw_stream);
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

//===----------------------------------------------------------------------===//
// 5. get_next_task_hint totals tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_concat hint fires exactly on the totals predicate", "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto first                 = make_int_batch(256);
  auto second                = make_int_batch(256);
  const uint64_t small_bytes = batch_bytes(first);
  REQUIRE(small_bytes > 1);
  // Exact-boundary thresholds below assume identically sized batches.
  REQUIRE(batch_bytes(second) == small_bytes);

  SECTION("lone oversized batch waits for a groupmate")
  {
    auto rig = make_concat_rig(small_bytes - 1);
    rig.op->push_data_batch_partitioned("input", first, 0);
    auto hint = rig.op->get_next_task_hint();
    REQUIRE(hint.has_value());
    REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
    REQUIRE(hint->producer == rig.upstream_producer.get());
  }

  SECTION("oversized batch plus a second batch fires")
  {
    auto rig = make_concat_rig(small_bytes - 1);
    rig.op->push_data_batch_partitioned("input", first, 0);
    rig.op->push_data_batch_partitioned("input", second, 0);
    auto hint = rig.op->get_next_task_hint();
    REQUIRE(hint.has_value());
    REQUIRE(hint->hint == TaskCreationHint::READY);
    REQUIRE(hint->producer == rig.op.get());
  }

  SECTION("two batches under the threshold wait")
  {
    auto rig = make_concat_rig(2 * small_bytes + 1);
    rig.op->push_data_batch_partitioned("input", first, 0);
    rig.op->push_data_batch_partitioned("input", second, 0);
    auto hint = rig.op->get_next_task_hint();
    REQUIRE(hint.has_value());
    REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
    REQUIRE(hint->producer == rig.upstream_producer.get());
  }

  SECTION("two batches exactly at the threshold wait (strict >)")
  {
    auto rig = make_concat_rig(2 * small_bytes);
    rig.op->push_data_batch_partitioned("input", first, 0);
    rig.op->push_data_batch_partitioned("input", second, 0);
    auto hint = rig.op->get_next_task_hint();
    REQUIRE(hint.has_value());
    REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  }

  SECTION("two batches over the threshold fire")
  {
    auto rig = make_concat_rig(2 * small_bytes - 1);
    rig.op->push_data_batch_partitioned("input", first, 0);
    rig.op->push_data_batch_partitioned("input", second, 0);
    auto hint = rig.op->get_next_task_hint();
    REQUIRE(hint.has_value());
    REQUIRE(hint->hint == TaskCreationHint::READY);
    REQUIRE(hint->producer == rig.op.get());
  }

  SECTION("concat_all waits regardless of buffered size")
  {
    auto rig = make_concat_rig(small_bytes - 1);
    rig.op->set_concat_all(true);
    rig.op->push_data_batch_partitioned("input", first, 0);
    rig.op->push_data_batch_partitioned("input", second, 0);
    auto hint = rig.op->get_next_task_hint();
    REQUIRE(hint.has_value());
    REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
    REQUIRE(hint->producer == rig.upstream_producer.get());
  }
}

TEST_CASE("sirius_physical_concat totals stay consistent across push, pull, and flush",
          "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto big                   = make_int_batch(1024);
  auto small_0               = make_int_batch(256);
  auto small_1               = make_int_batch(256);
  auto small_2               = make_int_batch(256);
  const uint64_t small_bytes = batch_bytes(small_0);
  REQUIRE(batch_bytes(small_1) == small_bytes);
  REQUIRE(batch_bytes(small_2) == small_bytes);
  // small < threshold < 2 * small < big
  const uint64_t threshold = small_bytes + small_bytes / 2;
  REQUIRE(batch_bytes(big) > threshold);

  auto rig = make_concat_rig(threshold);

  // Lone oversized batch: over the threshold but count == 1 -> WAITING.
  rig.op->push_data_batch_partitioned("input", big, 0);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::WAITING_FOR_INPUT_DATA);

  // A groupmate arrives -> partition 0 fires.
  rig.op->push_data_batch_partitioned("input", small_0, 0);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::READY);

  // A batch on another partition does not disturb partition 0's READY.
  rig.op->push_data_batch_partitioned("input", small_1, 1);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::READY);

  // Pull 1: the oversized head is popped alone; small_0 stays behind.
  auto pull_1 = rig.op->get_next_task_input_data();
  REQUIRE(pull_1 != nullptr);
  REQUIRE(group_batch_ids(*pull_1) == std::vector<uint64_t>{big->get_batch_id()});
  REQUIRE(dynamic_cast<const partitioned_operator_data&>(*pull_1).get_partition_idx() == 0);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::WAITING_FOR_INPUT_DATA);

  // Partition 1 crosses the threshold with two batches -> READY.
  rig.op->push_data_batch_partitioned("input", small_2, 1);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::READY);

  // Pull 2: partition 0's residual runt goes first; partition 1 still fires.
  auto pull_2 = rig.op->get_next_task_input_data();
  REQUIRE(pull_2 != nullptr);
  REQUIRE(group_batch_ids(*pull_2) == std::vector<uint64_t>{small_0->get_batch_id()});
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::READY);

  // Pull 3: partition 1's greedy prefix stops before the batch that crosses the threshold.
  auto pull_3 = rig.op->get_next_task_input_data();
  REQUIRE(pull_3 != nullptr);
  REQUIRE(group_batch_ids(*pull_3) == std::vector<uint64_t>{small_1->get_batch_id()});
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::WAITING_FOR_INPUT_DATA);

  // Residual flush: READY on a nonempty repo, nullopt once drained.
  rig.src_pipeline->set_finished(true);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::READY);
  auto pull_4 = rig.op->get_next_task_input_data();
  REQUIRE(pull_4 != nullptr);
  REQUIRE(group_batch_ids(*pull_4) == std::vector<uint64_t>{small_2->get_batch_id()});
  REQUIRE_FALSE(rig.op->get_next_task_hint().has_value());
}

TEST_CASE("sirius_physical_concat hint treats a port without a source pipeline as finished",
          "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto rig   = make_concat_rig(1024, source_state::none);
  auto batch = make_int_batch(64);
  rig.op->push_data_batch_partitioned("input", batch, 0);

  auto hint = rig.op->get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == rig.op.get());

  REQUIRE(rig.op->get_next_task_input_data() != nullptr);
  REQUIRE_FALSE(rig.op->get_next_task_hint().has_value());
}

TEST_CASE("sirius_physical_concat hint returns while a buffered batch is exclusively locked",
          "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto big_0               = make_int_batch(1024);
  auto big_1               = make_int_batch(1024);
  const uint64_t big_bytes = batch_bytes(big_0);

  auto rig = make_concat_rig(big_bytes - 1);
  rig.op->push_data_batch_partitioned("input", big_0, 0);
  rig.op->push_data_batch_partitioned("input", big_1, 0);

  // A helper thread holds big_0's exclusive lock, standing in for the downgrade executor
  // converting an idle buffered batch in place.
  std::mutex sync_mutex;
  std::condition_variable sync_cv;
  bool lock_held         = false;
  bool release_requested = false;
  std::thread lock_holder([&] {
    auto exclusive = big_0->to_mutable();
    {
      std::lock_guard<std::mutex> lg(sync_mutex);
      lock_held = true;
    }
    sync_cv.notify_all();
    {
      std::unique_lock<std::mutex> ul(sync_mutex);
      sync_cv.wait(ul, [&] { return release_requested; });
    }
    auto idle = cucascade::data_batch::to_idle(std::move(exclusive));
  });
  {
    std::unique_lock<std::mutex> ul(sync_mutex);
    sync_cv.wait(ul, [&] { return lock_held; });
  }

  // Watchdog: run the hint on its own thread and bound the wait, so a hint that blocks on the
  // batch lock fails the test instead of hanging the suite.
  std::optional<sirius::op::task_creation_hint> hint;
  bool hint_done = false;
  std::thread hint_runner([&] {
    auto result = rig.op->get_next_task_hint();
    {
      std::lock_guard<std::mutex> lg(sync_mutex);
      hint      = result;
      hint_done = true;
    }
    sync_cv.notify_all();
  });

  bool completed_in_time = false;
  {
    std::unique_lock<std::mutex> ul(sync_mutex);
    completed_in_time = sync_cv.wait_for(ul, std::chrono::seconds(10), [&] { return hint_done; });
  }

  // Release the exclusive lock before any assertion so a blocked hint unblocks and both threads
  // join even when the test fails.
  {
    std::lock_guard<std::mutex> lg(sync_mutex);
    release_requested = true;
  }
  sync_cv.notify_all();
  lock_holder.join();
  hint_runner.join();

  REQUIRE(completed_in_time);
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == rig.op.get());
}

TEST_CASE("sirius_physical_concat pull returns while a buffered batch is exclusively locked",
          "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto small_0 = make_int_batch(64);
  auto small_1 = make_int_batch(64);
  auto big     = make_int_batch(1024);

  // Threshold admits exactly the two smalls; the walk must then consult big's size — from the
  // ledger, not the batch — and leave it buffered.
  auto rig = make_concat_rig(batch_bytes(small_0) + batch_bytes(small_1));
  rig.op->push_data_batch_partitioned("input", small_0, 0);
  rig.op->push_data_batch_partitioned("input", small_1, 0);
  rig.op->push_data_batch_partitioned("input", big, 0);

  // A helper thread holds big's exclusive lock, standing in for the downgrade executor converting
  // an idle buffered batch in place.
  std::mutex sync_mutex;
  std::condition_variable sync_cv;
  bool lock_held         = false;
  bool release_requested = false;
  std::thread lock_holder([&] {
    auto exclusive = big->to_mutable();
    {
      std::lock_guard<std::mutex> lg(sync_mutex);
      lock_held = true;
    }
    sync_cv.notify_all();
    {
      std::unique_lock<std::mutex> ul(sync_mutex);
      sync_cv.wait(ul, [&] { return release_requested; });
    }
    auto idle = cucascade::data_batch::to_idle(std::move(exclusive));
  });
  {
    std::unique_lock<std::mutex> ul(sync_mutex);
    sync_cv.wait(ul, [&] { return lock_held; });
  }

  // Watchdog: run the pull on its own thread and bound the wait, so a walk that blocks on the
  // batch lock fails the test instead of hanging the suite.
  std::unique_ptr<operator_data> pulled;
  bool pull_done = false;
  std::thread pull_runner([&] {
    auto result = rig.op->get_next_task_input_data();
    {
      std::lock_guard<std::mutex> lg(sync_mutex);
      pulled    = std::move(result);
      pull_done = true;
    }
    sync_cv.notify_all();
  });

  bool completed_in_time = false;
  {
    std::unique_lock<std::mutex> ul(sync_mutex);
    completed_in_time = sync_cv.wait_for(ul, std::chrono::seconds(10), [&] { return pull_done; });
  }

  // Release the exclusive lock before any assertion so a blocked pull unblocks and both threads
  // join even when the test fails.
  {
    std::lock_guard<std::mutex> lg(sync_mutex);
    release_requested = true;
  }
  sync_cv.notify_all();
  lock_holder.join();
  pull_runner.join();

  REQUIRE(completed_in_time);
  REQUIRE(pulled != nullptr);
  REQUIRE(group_batch_ids(*pulled) ==
          std::vector<uint64_t>{small_0->get_batch_id(), small_1->get_batch_id()});
  REQUIRE(dynamic_cast<const partitioned_operator_data&>(*pulled).get_partition_idx() == 0);
  // The locked batch stayed behind, untouched.
  REQUIRE(rig.repo->get_batch_ids(0) == std::vector<uint64_t>{big->get_batch_id()});
}

//===----------------------------------------------------------------------===//
// 6. Single-batch forward tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_concat forwards a single-batch group downstream without a task",
          "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto rig        = make_concat_rig(1024, source_state::finished);
  auto downstream = make_downstream_rig();
  rig.op->add_next_port_after_sink({downstream.op.get(), "input"});

  auto batch = make_int_batch(64);
  rig.op->push_data_batch_partitioned("input", batch, 2);

  REQUIRE(rig.op->get_next_task_input_data() == nullptr);
  REQUIRE(rig.repo->total_size() == 0);

  // The forward preserves the partition index and moves the very same batch object.
  REQUIRE(downstream.repo->get_batch_ids(2) == std::vector<uint64_t>{batch->get_batch_id()});
  auto forwarded = downstream.repo->pop_data_batch_by_id(batch->get_batch_id(), 2);
  REQUIRE(forwarded.get() == batch.get());
}

TEST_CASE("sirius_physical_concat returns multi-batch groups as task input with wiring present",
          "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto batch_0               = make_int_batch(256);
  auto batch_1               = make_int_batch(256);
  auto batch_2               = make_int_batch(256);
  const uint64_t small_bytes = batch_bytes(batch_0);
  REQUIRE(batch_bytes(batch_1) == small_bytes);
  REQUIRE(batch_bytes(batch_2) == small_bytes);

  // Two batches fit exactly; the third crosses the threshold and stays behind.
  auto rig        = make_concat_rig(2 * small_bytes);
  auto downstream = make_downstream_rig();
  rig.op->add_next_port_after_sink({downstream.op.get(), "input"});

  rig.op->push_data_batch_partitioned("input", batch_0, 0);
  rig.op->push_data_batch_partitioned("input", batch_1, 0);
  rig.op->push_data_batch_partitioned("input", batch_2, 0);

  auto result = rig.op->get_next_task_input_data();
  REQUIRE(result != nullptr);
  REQUIRE(group_batch_ids(*result) ==
          std::vector<uint64_t>{batch_0->get_batch_id(), batch_1->get_batch_id()});
  REQUIRE(dynamic_cast<const partitioned_operator_data&>(*result).get_partition_idx() == 0);

  // The multi-batch group takes the task path: nothing was forwarded, the leftover remains.
  REQUIRE(downstream.repo->total_size() == 0);
  REQUIRE(rig.repo->get_batch_ids(0) == std::vector<uint64_t>{batch_2->get_batch_id()});
}

TEST_CASE("sirius_physical_concat drains singles by forwarding and returns the first multi group",
          "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto lone                  = make_int_batch(256);
  auto pair_0                = make_int_batch(256);
  auto pair_1                = make_int_batch(256);
  const uint64_t small_bytes = batch_bytes(lone);

  auto rig        = make_concat_rig(3 * small_bytes, source_state::finished);
  auto downstream = make_downstream_rig();
  rig.op->add_next_port_after_sink({downstream.op.get(), "input"});

  rig.op->push_data_batch_partitioned("input", lone, 0);
  rig.op->push_data_batch_partitioned("input", pair_0, 1);
  rig.op->push_data_batch_partitioned("input", pair_1, 1);

  // One call: partition 0's single is forwarded, partition 1's pair comes back as task input.
  auto result = rig.op->get_next_task_input_data();
  REQUIRE(result != nullptr);
  REQUIRE(group_batch_ids(*result) ==
          std::vector<uint64_t>{pair_0->get_batch_id(), pair_1->get_batch_id()});
  REQUIRE(dynamic_cast<const partitioned_operator_data&>(*result).get_partition_idx() == 1);
  REQUIRE(downstream.repo->get_batch_ids(0) == std::vector<uint64_t>{lone->get_batch_id()});

  // Drained: the hint reports nothing left and further pulls forward nothing new.
  REQUIRE_FALSE(rig.op->get_next_task_hint().has_value());
  REQUIRE(rig.op->get_next_task_input_data() == nullptr);
  REQUIRE(downstream.repo->total_size() == 1);
}

TEST_CASE("sirius_physical_concat returns a single-batch group as task input without sink wiring",
          "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto rig   = make_concat_rig(1024, source_state::finished);
  auto batch = make_int_batch(64);
  rig.op->push_data_batch_partitioned("input", batch, 0);

  auto result = rig.op->get_next_task_input_data();
  REQUIRE(result != nullptr);
  auto& group = dynamic_cast<const partitioned_operator_data&>(*result);
  REQUIRE(group.get_data_batches().size() == 1);
  REQUIRE(group.get_data_batches()[0].get() == batch.get());
  REQUIRE(group.get_partition_idx() == 0);
}

TEST_CASE("sirius_physical_concat with concat_all forwards a lone batch at pipeline finish",
          "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  // LEFT join + is_build=true -> _concat_all = true.
  auto rig =
    make_concat_rig(1024, source_state::finished, duckdb::JoinType::LEFT, /*is_build=*/true);
  auto downstream = make_downstream_rig();
  rig.op->add_next_port_after_sink({downstream.op.get(), "input"});

  auto batch = make_int_batch(64);
  rig.op->push_data_batch_partitioned("input", batch, 0);

  REQUIRE(rig.op->get_next_task_input_data() == nullptr);
  auto forwarded = downstream.repo->pop_data_batch_by_id(batch->get_batch_id(), 0);
  REQUIRE(forwarded.get() == batch.get());
}

TEST_CASE("sirius_physical_concat totals stay balanced after forwards", "[physical_concat]")
{
  REQUIRE(get_shared_mem_space() != nullptr);

  auto batch_0               = make_int_batch(256);
  auto batch_1               = make_int_batch(256);
  auto batch_2               = make_int_batch(256);
  auto batch_3               = make_int_batch(256);
  auto batch_4               = make_int_batch(256);
  const uint64_t small_bytes = batch_bytes(batch_0);
  REQUIRE(small_bytes > 1);
  for (const auto& batch : {batch_1, batch_2, batch_3, batch_4}) {
    REQUIRE(batch_bytes(batch) == small_bytes);
  }

  auto rig        = make_concat_rig(2 * small_bytes - 1);
  auto downstream = make_downstream_rig();
  rig.op->add_next_port_after_sink({downstream.op.get(), "input"});

  // A runt single is forwarded rather than returned.
  rig.op->push_data_batch_partitioned("input", batch_0, 0);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(rig.op->get_next_task_input_data() == nullptr);
  REQUIRE(downstream.repo->total_size() == 1);

  // The predicate fires exactly as before the forward.
  rig.op->push_data_batch_partitioned("input", batch_1, 0);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  rig.op->push_data_batch_partitioned("input", batch_2, 0);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::READY);

  // The greedy walk splits the pair into two singles; both forward, none returns.
  REQUIRE(rig.op->get_next_task_input_data() == nullptr);
  REQUIRE(downstream.repo->total_size() == 3);
  REQUIRE(rig.repo->total_size() == 0);

  // No drift: a fresh pair fires the predicate exactly at the same boundary.
  rig.op->push_data_batch_partitioned("input", batch_3, 0);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  rig.op->push_data_batch_partitioned("input", batch_4, 0);
  REQUIRE(require_hint(*rig.op) == TaskCreationHint::READY);
}
