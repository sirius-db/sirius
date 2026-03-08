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

// Tests for COUNT(DISTINCT col) via sirius_physical_grouped_aggregate +
// sirius_physical_grouped_aggregate_merge.  Scenarios exercised:
//
//   1. Single batch: verifies the local COLLECT_SET → merge count_elements pipeline.
//   2. Multiple batches (randomly striped): verifies correctness when a group's rows
//      are spread across several input batches within a partition.
//   3. Cross-batch duplicates within a partition: the critical test that the same
//      (key, value) pair appearing in multiple local aggregate outputs is counted
//      only once after MERGE_SETS. Partitions are mutually exclusive by group key,
//      but a partition receives multiple batches each processed independently.
//   4. Mixed count(distinct) + other aggregations.
//   5. Multiple partitions: verifies that separate execute() calls per partition
//      (as done by the real pipeline) each produce correct results.

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "utils/data_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/table/table.hpp>

#include <catch.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <memory>
#include <numeric>
#include <vector>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
using sirius::test::vector_to_cudf_column;

// ---------------------------------------------------------------------------
// Helper: build input table [key_col (int32), val_col (T)]
// ---------------------------------------------------------------------------
template <typename ValTraits>
std::unique_ptr<cudf::table> make_count_distinct_input(
  const std::vector<int32_t>& keys,
  const std::vector<typename ValTraits::type>& values,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>(keys, stream, mr));
  cols.push_back(vector_to_cudf_column<ValTraits>(values, stream, mr));
  return std::make_unique<cudf::table>(std::move(cols));
}

// ---------------------------------------------------------------------------
// Helper: build expected output table [key (int32), count_distinct (int64)]
// ---------------------------------------------------------------------------
std::unique_ptr<cudf::table> make_count_distinct_expected(const std::vector<int32_t>& exp_keys,
                                                          const std::vector<int64_t>& exp_counts,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>(exp_keys, stream, mr));
  cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>(exp_counts, stream, mr));
  return std::make_unique<cudf::table>(std::move(cols));
}

// ---------------------------------------------------------------------------
// Helper: run local aggregate on one batch and return the output data_batch
// ---------------------------------------------------------------------------
std::shared_ptr<data_batch> run_local(sirius_physical_grouped_aggregate& local_op,
                                      std::shared_ptr<data_batch> input)
{
  auto out = local_op.execute(operator_data(std::vector<std::shared_ptr<data_batch>>{input}),
                              default_stream());
  REQUIRE(out->get_data_batches().size() == 1);
  return out->get_data_batches()[0];
}

}  // namespace

// ===========================================================================
// TEST 1: Single batch - basic correctness
//
// Data layout: col0=key (int32), col1=value (int32)
//   key=0 → values [10, 20, 10, 30, 20]  → distinct: {10,20,30} → count=3
//   key=1 → values [40, 50, 40]           → distinct: {40,50}    → count=2
//   key=2 → values [60, 60, 60]           → distinct: {60}       → count=1
// ===========================================================================
TEST_CASE("count distinct: single batch, basic correctness",
          "[physical_grouped_aggregate_count_distinct]")
{
  using KeyTraits = gpu_type_traits<int32_t>;
  using ValTraits = gpu_type_traits<int32_t>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto mr     = get_resource_ref(*space);
  auto stream = default_stream();

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  std::vector<int32_t> keys   = {0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2};
  std::vector<int32_t> values = {10, 20, 10, 30, 20, 40, 50, 40, 60, 60, 60};

  auto input_batch =
    sirius::make_data_batch(make_count_distinct_input<ValTraits>(keys, values, stream, mr), *space);

  auto expected_table = make_count_distinct_expected({0, 1, 2}, {3, 2, 1}, stream, mr);

  // Local operator
  auto agg1 = sirius::test::create_count_distinct_expressions<KeyTraits, ValTraits>({0}, 1);
  sirius_physical_grouped_aggregate local_op(context,
                                             std::move(agg1.output_types),
                                             std::move(agg1.aggregates),
                                             std::move(agg1.groups),
                                             3 /*estimated_cardinality*/);

  // Merge operator (copies metadata from local_op)
  sirius_physical_grouped_aggregate_merge merge_op(&local_op);

  auto local_out = run_local(local_op, input_batch);

  // Merge post-processes LIST → INT64 count, even for a single batch
  auto final_out = merge_op.execute(
    operator_data(std::vector<std::shared_ptr<data_batch>>{local_out}), default_stream());
  REQUIRE(final_out->get_data_batches().size() == 1);

  bool match = sirius::test::expect_data_batch_equivalent_to_table(
    final_out->get_data_batches()[0], expected_table->view(), true /*sort*/);
  REQUIRE(match);
}

// ===========================================================================
// TEST 2: Multiple batches - randomly striped split
//
// The full dataset is randomly shuffled and split into 5 batches, ensuring
// that most groups have rows spread across multiple partitions (and therefore
// the same value may appear in several batches). MERGE_SETS must deduplicate.
//
// Data design: 30 groups, each group g contains values
//   [g*100, g*100, g*100+1, g*100+1, g*100+2, g*100+2]
// so exactly 3 distinct values per group.
// ===========================================================================
TEMPLATE_TEST_CASE("count distinct: multiple batches, randomly striped",
                   "[physical_grouped_aggregate_count_distinct]",
                   int32_t,
                   int64_t)
{
  using KeyTraits = gpu_type_traits<int32_t>;
  using ValTraits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto mr     = get_resource_ref(*space);
  auto stream = default_stream();

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  constexpr int num_groups = 30;

  // Build input: each group has 6 rows, 3 distinct values (each repeated twice)
  std::vector<int32_t> keys;
  std::vector<typename ValTraits::type> values;
  keys.reserve(num_groups * 6);
  values.reserve(num_groups * 6);

  for (int g = 0; g < num_groups; ++g) {
    auto base = static_cast<typename ValTraits::type>(g * 100);
    for (int rep = 0; rep < 2; ++rep) {
      keys.push_back(g);
      values.push_back(base);
      keys.push_back(g);
      values.push_back(base + static_cast<typename ValTraits::type>(1));
      keys.push_back(g);
      values.push_back(base + static_cast<typename ValTraits::type>(2));
    }
  }

  auto full_table = make_count_distinct_input<ValTraits>(keys, values, stream, mr);

  // Split into 5 random, non-contiguous partitions
  auto splits = sirius::test::make_random_striped_split(std::move(full_table), 5, stream, mr);

  // Build expected: all groups have count_distinct == 3
  std::vector<int32_t> exp_keys(num_groups);
  std::iota(exp_keys.begin(), exp_keys.end(), 0);
  std::vector<int64_t> exp_counts(num_groups, 3);
  auto expected_table = make_count_distinct_expected(exp_keys, exp_counts, stream, mr);

  // Local operator
  auto agg1 = sirius::test::create_count_distinct_expressions<KeyTraits, ValTraits>({0}, 1);
  sirius_physical_grouped_aggregate local_op(context,
                                             std::move(agg1.output_types),
                                             std::move(agg1.aggregates),
                                             std::move(agg1.groups),
                                             num_groups);

  // Merge operator
  sirius_physical_grouped_aggregate_merge merge_op(&local_op);

  // Run local aggregate on each split
  std::vector<std::shared_ptr<data_batch>> local_results;
  for (auto& split : splits) {
    auto input_batch = sirius::make_data_batch(std::move(split), *space);
    local_results.push_back(run_local(local_op, input_batch));
  }

  // Merge all local results
  auto final_out = merge_op.execute(operator_data(local_results), default_stream());
  REQUIRE(final_out->get_data_batches().size() == 1);

  bool match = sirius::test::expect_data_batch_equivalent_to_table(
    final_out->get_data_batches()[0], expected_table->view(), true /*sort*/);
  REQUIRE(match);
}

// ===========================================================================
// TEST 3: Cross-batch duplicate deduplication within a partition
//
// Partitions are mutually exclusive by group key hash, but a single partition
// receives multiple input batches processed independently by the local aggregate.
// Each batch produces its own COLLECT_SET, so the same (key, value) pair can
// appear in multiple local outputs for the same partition. MERGE_SETS must
// union and deduplicate them.
//
// Batch 1: key=[0,0,1,2], val=[10,20,40,60]
// Batch 2: key=[0,1,1,2], val=[10,50,40,60]  ← (0,10),(1,40),(2,60) are cross-batch dups
// Batch 3: key=[0,1,2],   val=[30,40,60]     ← (1,40),(2,60) are cross-batch dups
//
// Expected:
//   key=0 → distinct {10,20,30} → 3
//   key=1 → distinct {40,50}    → 2
//   key=2 → distinct {60}       → 1
// ===========================================================================
TEST_CASE("count distinct: cross-batch duplicate deduplication within a partition",
          "[physical_grouped_aggregate_count_distinct]")
{
  using KeyTraits = gpu_type_traits<int32_t>;
  using ValTraits = gpu_type_traits<int32_t>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto mr     = get_resource_ref(*space);
  auto stream = default_stream();

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Batch 1
  auto batch1 = sirius::make_data_batch(
    make_count_distinct_input<ValTraits>({0, 0, 1, 2}, {10, 20, 40, 60}, stream, mr), *space);

  // Batch 2 — intentional cross-batch dups: (0,10), (1,40), (2,60)
  auto batch2 = sirius::make_data_batch(
    make_count_distinct_input<ValTraits>({0, 1, 1, 2}, {10, 50, 40, 60}, stream, mr), *space);

  // Batch 3 — further cross-batch dups: (1,40), (2,60)
  auto batch3 = sirius::make_data_batch(
    make_count_distinct_input<ValTraits>({0, 1, 2}, {30, 40, 60}, stream, mr), *space);

  auto expected_table = make_count_distinct_expected({0, 1, 2}, {3, 2, 1}, stream, mr);

  // Local operator (shared across all batches)
  auto agg1 = sirius::test::create_count_distinct_expressions<KeyTraits, ValTraits>({0}, 1);
  sirius_physical_grouped_aggregate local_op(context,
                                             std::move(agg1.output_types),
                                             std::move(agg1.aggregates),
                                             std::move(agg1.groups),
                                             3 /*estimated_cardinality*/);

  sirius_physical_grouped_aggregate_merge merge_op(&local_op);

  std::vector<std::shared_ptr<data_batch>> local_results;
  local_results.push_back(run_local(local_op, batch1));
  local_results.push_back(run_local(local_op, batch2));
  local_results.push_back(run_local(local_op, batch3));

  auto final_out = merge_op.execute(operator_data(local_results), default_stream());
  REQUIRE(final_out->get_data_batches().size() == 1);

  bool match = sirius::test::expect_data_batch_equivalent_to_table(
    final_out->get_data_batches()[0], expected_table->view(), true /*sort*/);
  REQUIRE(match);
}

// ===========================================================================
// TEST 4: Count distinct mixed with regular aggregations
//
// Verifies that count(distinct val) coexists correctly with count(val) and
// min(val) in the same grouped aggregate.
//
// Input (col0=key, col1=val):
//   key=0: val=[10,10,20,20,30] → count_distinct=3, min=10, count=5
//   key=1: val=[40,40,50]       → count_distinct=2, min=40, count=3
//
// The expression order is: [count(distinct val), min(val), count(val)]
// so the output columns are: [key, count_distinct(int64), min(int32), count(int64)]
// ===========================================================================
TEST_CASE("count distinct: mixed with regular aggregations, multiple batches",
          "[physical_grouped_aggregate_count_distinct]")
{
  using KeyTraits = gpu_type_traits<int32_t>;
  using ValTraits = gpu_type_traits<int32_t>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto mr     = get_resource_ref(*space);
  auto stream = default_stream();

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Batch 1 and Batch 2 together form the full dataset:
  // key=0: [10,10,20] ++ [20,30]  → {10,20,30} → count_distinct=3, min=10, count=5
  // key=1: [40,40]    ++ [50]     → {40,50}    → count_distinct=2, min=40, count=3
  auto batch1 = sirius::make_data_batch(
    make_count_distinct_input<ValTraits>({0, 0, 0, 1, 1}, {10, 10, 20, 40, 40}, stream, mr),
    *space);

  auto batch2 = sirius::make_data_batch(
    make_count_distinct_input<ValTraits>({0, 0, 1}, {20, 30, 50}, stream, mr), *space);

  // Build expected: [key(int32) | count_distinct(int64) | min(int32) | count(int64)]
  std::vector<std::unique_ptr<cudf::column>> exp_cols;
  exp_cols.push_back(vector_to_cudf_column<KeyTraits>({0, 1}, stream, mr));
  exp_cols.push_back(
    vector_to_cudf_column<gpu_type_traits<int64_t>>({3, 2}, stream, mr));      // count distinct
  exp_cols.push_back(vector_to_cudf_column<ValTraits>({10, 40}, stream, mr));  // min
  exp_cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>({5, 3}, stream, mr));  // count
  auto expected_table = std::make_unique<cudf::table>(std::move(exp_cols));

  // Build expressions: [count(distinct col1), min(col1), count(col1)]
  duckdb::vector<duckdb::LogicalType> output_types;
  output_types.push_back(KeyTraits::logical_type());    // group key
  output_types.push_back(duckdb::LogicalType::BIGINT);  // count distinct
  output_types.push_back(ValTraits::logical_type());    // min
  output_types.push_back(duckdb::LogicalType::BIGINT);  // count

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups;
  groups.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(KeyTraits::logical_type(), 0));

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;

  // count(distinct col1)
  {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> ch;
    ch.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(ValTraits::logical_type(), 1));
    auto fn = sirius::test::MakeDummyAggregate(
      "count", {ValTraits::logical_type()}, duckdb::LogicalType::BIGINT);
    aggregates.push_back(duckdb::make_uniq<duckdb::BoundAggregateExpression>(
      fn, std::move(ch), nullptr, nullptr, duckdb::AggregateType::DISTINCT));
  }
  // min(col1)
  {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> ch;
    ch.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(ValTraits::logical_type(), 1));
    auto fn = sirius::test::MakeDummyAggregate(
      "min", {ValTraits::logical_type()}, ValTraits::logical_type());
    aggregates.push_back(duckdb::make_uniq<duckdb::BoundAggregateExpression>(
      fn, std::move(ch), nullptr, nullptr, duckdb::AggregateType::NON_DISTINCT));
  }
  // count(col1)
  {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> ch;
    ch.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(ValTraits::logical_type(), 1));
    auto fn = sirius::test::MakeDummyAggregate(
      "count", {ValTraits::logical_type()}, ValTraits::logical_type());
    aggregates.push_back(duckdb::make_uniq<duckdb::BoundAggregateExpression>(
      fn, std::move(ch), nullptr, nullptr, duckdb::AggregateType::NON_DISTINCT));
  }

  // Clone expressions for merge operator (it takes the same spec)
  duckdb::vector<duckdb::LogicalType> output_types2 = output_types;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups2;
  groups2.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(KeyTraits::logical_type(), 0));
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates2;
  for (auto& agg : aggregates) {
    aggregates2.push_back(agg->Copy());
  }

  sirius_physical_grouped_aggregate local_op(context,
                                             std::move(output_types),
                                             std::move(aggregates),
                                             std::move(groups),
                                             2 /*estimated_cardinality*/);

  sirius_physical_grouped_aggregate_merge merge_op(context,
                                                   std::move(output_types2),
                                                   std::move(aggregates2),
                                                   std::move(groups2),
                                                   2 /*estimated_cardinality*/);

  auto local1 = run_local(local_op, batch1);
  auto local2 = run_local(local_op, batch2);

  auto final_out = merge_op.execute(
    operator_data(std::vector<std::shared_ptr<data_batch>>{local1, local2}), default_stream());
  REQUIRE(final_out->get_data_batches().size() == 1);

  bool match = sirius::test::expect_data_batch_equivalent_to_table(
    final_out->get_data_batches()[0], expected_table->view(), true /*sort*/);
  REQUIRE(match);
}

// ===========================================================================
// TEST 5: Multiple partitions - separate execute() calls per partition
//
// In the real pipeline, sirius_physical_partition distributes rows by group-key
// hash so each group belongs to exactly ONE partition.  The merge operator's
// execute() is then called once per partition with ALL of that partition's local
// aggregate outputs.  This test replicates that structure:
//
//   Partition 0 owns keys {0, 1, 2}  – 3 batches each
//   Partition 1 owns keys {3, 4}     – 3 batches each
//
// Each batch is a random stripe of its partition's rows, so the same value can
// appear in multiple batches (cross-batch dedup must work within a partition).
//
// Expected per group: exactly 4 distinct values (val = key*10 + {0,1,2,3}).
// The merge operator is called independently for each partition and the per-
// partition result is verified separately — exactly as the real engine does it.
// ===========================================================================
TEST_CASE("count distinct: multiple partitions with multiple batches per partition",
          "[physical_grouped_aggregate_count_distinct]")
{
  using KeyTraits = gpu_type_traits<int32_t>;
  using ValTraits = gpu_type_traits<int32_t>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto mr     = get_resource_ref(*space);
  auto stream = default_stream();

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Helper: build [key_col, val_col] where each group g gets values
  //   {g*10+0, g*10+1, g*10+2, g*10+3} each repeated twice (8 rows per group)
  auto make_partition_table = [&](const std::vector<int32_t>& group_keys) {
    std::vector<int32_t> keys, values;
    for (int g : group_keys) {
      for (int v = 0; v < 4; ++v) {
        // Repeat each distinct value twice so dedup is exercised within a batch
        keys.push_back(g);
        values.push_back(g * 10 + v);
        keys.push_back(g);
        values.push_back(g * 10 + v);
      }
    }
    return make_count_distinct_input<ValTraits>(keys, values, stream, mr);
  };

  // Partition 0: keys 0, 1, 2
  auto table0  = make_partition_table({0, 1, 2});
  auto splits0 = sirius::test::make_random_striped_split(std::move(table0), 3, stream, mr);

  // Partition 1: keys 3, 4
  auto table1  = make_partition_table({3, 4});
  auto splits1 = sirius::test::make_random_striped_split(std::move(table1), 3, stream, mr);

  // Shared local + merge operators
  auto agg_spec = sirius::test::create_count_distinct_expressions<KeyTraits, ValTraits>({0}, 1);
  sirius_physical_grouped_aggregate local_op(context,
                                             std::move(agg_spec.output_types),
                                             std::move(agg_spec.aggregates),
                                             std::move(agg_spec.groups),
                                             5 /*estimated_cardinality*/);
  sirius_physical_grouped_aggregate_merge merge_op(&local_op);

  // --- Process partition 0 (separate execute() call, as the real pipeline does) ---
  std::vector<std::shared_ptr<data_batch>> local_p0;
  for (auto& split : splits0) {
    local_p0.push_back(run_local(local_op, sirius::make_data_batch(std::move(split), *space)));
  }
  auto result_p0 = merge_op.execute(operator_data(local_p0), default_stream());
  REQUIRE(result_p0->get_data_batches().size() == 1);

  auto expected_p0 = make_count_distinct_expected({0, 1, 2}, {4, 4, 4}, stream, mr);
  REQUIRE(sirius::test::expect_data_batch_equivalent_to_table(
    result_p0->get_data_batches()[0], expected_p0->view(), true /*sort*/));

  // --- Process partition 1 (separate execute() call) ---
  std::vector<std::shared_ptr<data_batch>> local_p1;
  for (auto& split : splits1) {
    local_p1.push_back(run_local(local_op, sirius::make_data_batch(std::move(split), *space)));
  }
  auto result_p1 = merge_op.execute(operator_data(local_p1), default_stream());
  REQUIRE(result_p1->get_data_batches().size() == 1);

  auto expected_p1 = make_count_distinct_expected({3, 4}, {4, 4}, stream, mr);
  REQUIRE(sirius::test::expect_data_batch_equivalent_to_table(
    result_p1->get_data_batches()[0], expected_p1->view(), true /*sort*/));
}

// ===========================================================================
// TEST 6: Multi-column COUNT(DISTINCT (col_a, col_b))
//
// Distinct counts are over COMBINATIONS of (col_a, col_b), not individual cols.
//
// Input (col0=key, col1=val_a, col2=val_b):
//   key=0: (10,1),(10,2),(10,1),(20,1)  → {(10,1),(10,2),(20,1)} → 3
//   key=1: (30,3),(30,3),(40,3)          → {(30,3),(40,3)}         → 2
//
// Exercises:
//   - (10,1) appearing twice counts once (intra-batch dedup).
//   - (10,1) and (10,2) are different combos and both counted.
//   - Multiple batches per partition: same combo in different batches counts once.
// ===========================================================================
TEST_CASE("count distinct: multi-column struct expression",
          "[physical_grouped_aggregate_count_distinct]")
{
  using KeyTraits = gpu_type_traits<int32_t>;
  using ValTraits = gpu_type_traits<int32_t>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto mr     = get_resource_ref(*space);
  auto stream = default_stream();

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Input table: 3 columns — key (col0), val_a (col1), val_b (col2)
  auto make_3col_input = [&](const std::vector<int32_t>& keys,
                             const std::vector<int32_t>& val_a,
                             const std::vector<int32_t>& val_b) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(vector_to_cudf_column<KeyTraits>(keys, stream, mr));
    cols.push_back(vector_to_cudf_column<ValTraits>(val_a, stream, mr));
    cols.push_back(vector_to_cudf_column<ValTraits>(val_b, stream, mr));
    return std::make_unique<cudf::table>(std::move(cols));
  };

  // Batch 1
  auto batch1 =
    sirius::make_data_batch(make_3col_input({0, 0, 1, 1}, {10, 10, 30, 30}, {1, 1, 3, 3}), *space);

  // Batch 2 — introduces new combos and cross-batch dups
  auto batch2 =
    sirius::make_data_batch(make_3col_input({0, 0, 0, 1}, {10, 10, 20, 40}, {2, 1, 1, 3}), *space);

  // Expected: key=0 → 3 (combos: (10,1),(10,2),(20,1)), key=1 → 2 (combos: (30,3),(40,3))
  auto expected_table = make_count_distinct_expected({0, 1}, {3, 2}, stream, mr);

  // Build COUNT(DISTINCT (col1, col2)) grouped by col0
  auto agg_spec = sirius::test::create_count_distinct_struct_col_expressions(
    {{duckdb::LogicalType::INTEGER, 0}},  // GROUP BY col0
    {{duckdb::LogicalType::INTEGER, 1},   // struct_pack(col1, col2)
     {duckdb::LogicalType::INTEGER, 2}});

  sirius_physical_grouped_aggregate local_op(context,
                                             std::move(agg_spec.output_types),
                                             std::move(agg_spec.aggregates),
                                             std::move(agg_spec.groups),
                                             2 /*estimated_cardinality*/);
  sirius_physical_grouped_aggregate_merge merge_op(&local_op);

  auto local1 = run_local(local_op, batch1);
  auto local2 = run_local(local_op, batch2);

  auto final_out = merge_op.execute(
    operator_data(std::vector<std::shared_ptr<data_batch>>{local1, local2}), default_stream());
  REQUIRE(final_out->get_data_batches().size() == 1);

  REQUIRE(sirius::test::expect_data_batch_equivalent_to_table(
    final_out->get_data_batches()[0], expected_table->view(), true /*sort*/));
}
