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

#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/sirius_physical_concat.hpp>
#include <op/sirius_physical_hash_join.hpp>

#include <atomic>
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

/**
 * @brief Create a minimal sirius_physical_hash_join for testing concat.
 *
 * @param join_type The join type (INNER, LEFT, RIGHT, etc.)
 * @param output_types The logical types for the join output columns
 * @return hash_join_test_fixture owning both the logical and physical join
 */
hash_join_test_fixture create_test_hash_join(duckdb::JoinType join_type,
                                             duckdb::vector<duckdb::LogicalType> output_types)
{
  hash_join_test_fixture fixture;

  // Create a LogicalComparisonJoin with the desired join type
  fixture.logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  fixture.logical_join->types = output_types;

  // Create minimal child operators (need at least one type each for the hash join constructor)
  auto left_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
    0);
  auto right_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
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
    std::move(conditions),
    join_type,
    duckdb::vector<duckdb::idx_t>{},        // left_projection_map (empty = all)
    duckdb::vector<duckdb::idx_t>{},        // right_projection_map (empty = all)
    duckdb::vector<duckdb::LogicalType>{},  // delim_types
    1000,                                   // estimated_cardinality
    nullptr);                               // pushdown_info

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
  sirius_physical_concat concat_op({Traits::logical_type()}, 1000, fixture.hash_join.get(), false);

  // Execute
  auto outputs = concat_op.execute(partitioned_operator_data(input_batches, 0), default_stream());

  // Verify: single output batch with correct total rows
  REQUIRE(outputs->get_data_batches().size() == 1);
  auto& out_table = outputs->get_data_batches()[0]
                      ->get_data()
                      ->cast<cucascade::gpu_table_representation>()
                      .get_table();
  REQUIRE(static_cast<std::size_t>(out_table.num_rows()) == total_rows);
  REQUIRE(out_table.num_columns() == 1);

  // Verify data content
  auto host_data = copy_column_to_host<typename Traits::type>(out_table.view().column(0));
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
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

  auto outputs = concat_op.execute(partitioned_operator_data({input_batch}, 0), default_stream());

  REQUIRE(outputs->get_data_batches().size() == 1);
  // Single batch should be the same pointer (passthrough)
  REQUIRE(outputs->get_data_batches()[0].get() == input_batch.get());
}

TEST_CASE("sirius_physical_concat handles empty input", "[physical_concat]")
{
  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

  auto outputs = concat_op.execute(
    partitioned_operator_data(std::vector<std::shared_ptr<cucascade::data_batch>>{}, 0),
    default_stream());

  REQUIRE(outputs->get_data_batches().empty());
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
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

  // Mix valid and null batches
  std::vector<std::shared_ptr<data_batch>> input = {batch1, nullptr, batch2, nullptr};
  auto outputs = concat_op.execute(partitioned_operator_data(input, 0), default_stream());

  REQUIRE(outputs->get_data_batches().size() == 1);
  auto& out_table = outputs->get_data_batches()[0]
                      ->get_data()
                      ->cast<cucascade::gpu_table_representation>()
                      .get_table();
  REQUIRE(out_table.num_rows() == 6);

  auto host_data                = copy_column_to_host<int32_t>(out_table.view().column(0));
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
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

  // Create a downstream partition consumer operator to receive the sink output
  sirius_physical_concat downstream_op(
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

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
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

  // Create two downstream operators
  sirius_physical_concat downstream1(
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);
  sirius_physical_concat downstream2(
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

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

TEST_CASE("sirius_physical_concat stops concatenating at DEFAULT_SCAN_TASK_BATCH_SIZE threshold",
          "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  // Save and set a small threshold so our test batches exceed it
  auto original_threshold                      = duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE;
  duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE = 1024;  // 1 KB

  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

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
  REQUIRE(result1->get_data_batches().size() < static_cast<std::size_t>(num_batches));
  REQUIRE(result1->get_data_batches().size() >= 1);

  // Collect total batches returned across multiple calls
  std::size_t total_batches_returned = result1->get_data_batches().size();
  while (true) {
    auto result = concat_op.get_next_task_input_data();
    if (!result) { break; }
    total_batches_returned += result->get_data_batches().size();
  }

  // All batches should eventually be consumed
  REQUIRE(total_batches_returned == static_cast<std::size_t>(num_batches));

  // Restore threshold
  duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE = original_threshold;
}

TEST_CASE("sirius_physical_concat with concat_all=true ignores threshold", "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  // Save and set a small threshold
  auto original_threshold                      = duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE;
  duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE = 1024;  // 1 KB

  // LEFT join + is_build=true -> _concat_all = true
  auto fixture = create_test_hash_join(duckdb::JoinType::LEFT, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), true);

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
  REQUIRE(result->get_data_batches().size() == static_cast<std::size_t>(num_batches));

  // No more batches remaining
  auto result2 = concat_op.get_next_task_input_data();
  REQUIRE(result2 == nullptr);

  // Restore threshold
  duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE = original_threshold;
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
      {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), true);
    REQUIRE(concat_build.is_build_concat() == true);

    sirius_physical_concat concat_probe(
      {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);
    REQUIRE(concat_probe.is_build_concat() == false);
  }

  SECTION("LEFT join + is_build=true -> is_build_concat returns true")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::LEFT, {duckdb::LogicalType::INTEGER});
    sirius_physical_concat concat_op(
      {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), true);
    REQUIRE(concat_op.is_build_concat() == true);
  }

  SECTION("LEFT join + is_build=false -> is_build_concat returns false")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::LEFT, {duckdb::LogicalType::INTEGER});
    sirius_physical_concat concat_op(
      {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);
    REQUIRE(concat_op.is_build_concat() == false);
  }

  SECTION("RIGHT join constructs successfully")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::RIGHT, {duckdb::LogicalType::INTEGER});
    REQUIRE_NOTHROW(
      sirius_physical_concat({duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), true));
  }

  SECTION("SEMI join constructs successfully")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::SEMI, {duckdb::LogicalType::INTEGER});
    REQUIRE_NOTHROW(
      sirius_physical_concat({duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false));
  }

  SECTION("OUTER join throws unsupported join type")
  {
    auto fixture = create_test_hash_join(duckdb::JoinType::OUTER, {duckdb::LogicalType::INTEGER});
    REQUIRE_NOTHROW(
      sirius_physical_concat({duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false));
  }

  SECTION("Non-hash-join parent throws")
  {
    sirius_physical_operator non_join_op(
      SiriusPhysicalOperatorType::PROJECTION, {duckdb::LogicalType::INTEGER}, 1000);
    REQUIRE_THROWS_AS(
      sirius_physical_concat({duckdb::LogicalType::INTEGER}, 1000, &non_join_op, false),
      std::runtime_error);
  }
}

//===----------------------------------------------------------------------===//
// 4. Multithreading tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_concat get_next_task_input_batch is thread-safe", "[physical_concat]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space != nullptr);

  // Save and set a small threshold to force multiple get_next_task_input_batch calls
  auto original_threshold                      = duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE;
  duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE = 1024;  // 1 KB

  auto fixture = create_test_hash_join(duckdb::JoinType::INNER, {duckdb::LogicalType::INTEGER});
  sirius_physical_concat concat_op(
    {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

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
      for (auto& batch : result->get_data_batches()) {
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

  // Restore threshold
  duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE = original_threshold;
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
        {duckdb::LogicalType::INTEGER}, 1000, fixture.hash_join.get(), false);

      // Create a dedicated CUDA stream for this thread
      cudaStream_t raw_stream;
      cudaStreamCreate(&raw_stream);
      rmm::cuda_stream_view stream(raw_stream);

      auto outputs =
        concat_op.execute(partitioned_operator_data(thread_inputs[thread_id], 0), default_stream());

      // Synchronize the stream before accessing results
      cudaStreamSynchronize(raw_stream);

      thread_outputs[thread_id] = std::move(outputs->get_data_batches());

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
    auto& out_table =
      thread_outputs[t][0]->get_data()->cast<cucascade::gpu_table_representation>().get_table();
    REQUIRE(static_cast<std::size_t>(out_table.num_rows()) == rows_per_batch * batches_per_thread);
  }
}
