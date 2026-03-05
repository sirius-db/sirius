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

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "utils/data_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <catch.hpp>

#include <algorithm>
#include <utility>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
using sirius::test::vector_to_cudf_column;
}  // namespace

// single batch expects to return the exact same data as the input batch, since it assumes it was
// already aggregated
TEST_CASE("sirius_physical_grouped_aggregate_merge grouped aggregates single data_batch",
          "[physical_grouped_aggregate_merge]")
{
  using Traits = gpu_type_traits<int32_t>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 10;

  // Create test data with single group key column
  // For the merge test, we need the expected (aggregated) table as input
  auto [raw_input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate<Traits>(num_groups, 1, stream, mr);

  // Create DuckDB context for aggregate function binding
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
    {0},                      // group_indexes: GROUP BY column 0
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {1, 1, 1}                 // agg_indexes: MIN(column 1), MAX(column 1), COUNT(column 1)
  );

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregate_merger(
    context,
    std::move(agg_result.output_types),
    std::move(agg_result.aggregates),
    std::move(agg_result.groups),
    num_groups);

  // For merge test, the input is already aggregated data (the expected table)
  // The merge operator should return it unchanged for a single batch
  auto input_table = std::make_unique<cudf::table>(expected_table->view());
  auto input_batch = sirius::make_data_batch(std::move(input_table), *space);

  auto outputs =
    grouped_aggregate_merger.execute(operator_data({std::move(input_batch)}), default_stream());

  // Verify we got one output batch
  REQUIRE(outputs->get_data_batches().size() == 1);

  // Compare output with expected using the validation utility
  // Dont Sort both tables before comparison since they should be identical
  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    outputs->get_data_batches()[0], expected_table->view(), false);
  REQUIRE(tables_match);
}

TEMPLATE_TEST_CASE(
  "sirius_physical_grouped_aggregate_merge grouped aggregates data_batch with multiple partition "
  "key, multiple aggregations",
  "[physical_grouped_aggregate_merge]",
  int32_t,
  int64_t,
  float,
  double,
  int16_t,
  decimal64_tag,
  string_tag,
  timestamp_us_tag,
  date32_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 1000;

  // Create test data with two group key columns
  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate<Traits>(num_groups, 2, stream, mr);

  auto input_tables =
    sirius::test::make_random_striped_split(std::move(input_table), 5, stream, mr);

  // Create DuckDB context for aggregate function binding
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Create aggregate expressions for grouped_aggregator
  auto agg_result1 = sirius::test::create_aggregate_expressions<Traits>(
    {0, 1},                   // group_indexes: GROUP BY column 0 and 1
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {2, 2, 2}                 // agg_indexes: MIN(column 2), MAX(column 2), COUNT(column 2)
  );

  // Create aggregate expressions for grouped_aggregate_merger
  auto agg_result2 = sirius::test::create_aggregate_expressions<Traits>(
    {0, 1},                   // group_indexes: GROUP BY column 0 and 1
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {2, 2, 2}                 // agg_indexes: MIN(column 2), MAX(column 2), COUNT(column 2)
  );

  // Create the grouped aggregate operator
  sirius_physical_grouped_aggregate grouped_aggregator(context,
                                                       std::move(agg_result1.output_types),
                                                       std::move(agg_result1.aggregates),
                                                       std::move(agg_result1.groups),
                                                       num_groups);

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregate_merger(
    context,
    std::move(agg_result2.output_types),
    std::move(agg_result2.aggregates),
    std::move(agg_result2.groups),
    num_groups);

  std::vector<std::shared_ptr<data_batch>> agg_outputs;

  for (auto& input_table : input_tables) {
    std::shared_ptr<data_batch> input_batch =
      sirius::make_data_batch(std::move(input_table), *space);

    auto outputs = grouped_aggregator.execute(operator_data({input_batch}), default_stream());

    // Verify we got one output batch
    REQUIRE(outputs->get_data_batches().size() == 1);
    agg_outputs.push_back(outputs->get_data_batches()[0]);
  }

  auto outputs = grouped_aggregate_merger.execute(operator_data(agg_outputs), default_stream());
  REQUIRE(outputs->get_data_batches().size() == 1);

  // need to cast the expected table column 4 (which is the count column) to int64_t since at the
  // merge stage, we end up doing a sum of int32_t which becomes an int64_t
  auto expected_columns = expected_table->release();

  // Validate that column 4 is of type int32
  REQUIRE(expected_columns[4]->type().id() == cudf::type_id::INT32);

  // Cast column 4 from int32 to int64
  auto casted_column =
    cudf::cast(expected_columns[4]->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);

  // Replace column 4 with the casted version
  expected_columns[4] = std::move(casted_column);

  // Recreate the expected_table with the updated columns
  expected_table = std::make_unique<cudf::table>(std::move(expected_columns));

  // Compare output with expected using the validation utility
  // Sort both tables before comparison since aggregation order is not guaranteed
  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    outputs->get_data_batches()[0], expected_table->view(), true);
  REQUIRE(tables_match);
}

TEMPLATE_TEST_CASE("sirius_physical_grouped_aggregate_merge end-to-end with AVG",
                   "[physical_grouped_aggregate_merge]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t)
//  decimal64_tag)  TODO: the unit test with decimal64_tag is failing with a cuda memory alignment
//  error due to https://github.com/rapidsai/cudf/issues/21512

{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 100;

  // Create test data with AVG expected values
  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate_with_avg<Traits>(num_groups, 1, stream, mr);

  // Split input into 5 batches for distributed aggregation
  auto input_tables =
    sirius::test::make_random_striped_split(std::move(input_table), 5, stream, mr);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Create aggregate expressions for local operator
  auto agg_result1 = sirius::test::create_aggregate_expressions<Traits>(
    {0},                             // GROUP BY column 0
    {"min", "max", "count", "avg"},  // aggregations including AVG
    {1, 1, 1, 1}                     // all on column 1
  );

  // Create aggregate expressions for merge operator
  auto agg_result2 = sirius::test::create_aggregate_expressions<Traits>(
    {0}, {"min", "max", "count", "avg"}, {1, 1, 1, 1});

  // Create local and merge operators
  sirius_physical_grouped_aggregate grouped_aggregator(context,
                                                       std::move(agg_result1.output_types),
                                                       std::move(agg_result1.aggregates),
                                                       std::move(agg_result1.groups),
                                                       num_groups);

  sirius_physical_grouped_aggregate_merge grouped_aggregate_merger(&grouped_aggregator);

  // Run local aggregation on each split
  std::vector<std::shared_ptr<data_batch>> agg_outputs;
  for (auto& split_table : input_tables) {
    auto input_batch = sirius::make_data_batch(std::move(split_table), *space);
    auto outputs     = grouped_aggregator.execute(operator_data({input_batch}), default_stream());
    REQUIRE(outputs->get_data_batches().size() == 1);
    agg_outputs.push_back(outputs->get_data_batches()[0]);
  }

  // Run merge with AVG projection
  auto outputs = grouped_aggregate_merger.execute(operator_data(agg_outputs), default_stream());
  REQUIRE(outputs->get_data_batches().size() == 1);

  // Cast expected count column from int32 to int64 (merge sums int32 counts -> int64)
  auto expected_columns = expected_table->release();
  auto count_col_idx    = 3;  // column 0=group, 1=min, 2=max, 3=count, 4=avg
  REQUIRE(expected_columns[count_col_idx]->type().id() == cudf::type_id::INT32);
  expected_columns[count_col_idx] = cudf::cast(
    expected_columns[count_col_idx]->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
  expected_table = std::make_unique<cudf::table>(std::move(expected_columns));

  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    outputs->get_data_batches()[0], expected_table->view(), true);
  REQUIRE(tables_match);
}
