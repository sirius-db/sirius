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
#include "data/data_batch_utils.hpp"
#include "utils/data_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/table/table.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/parser/query_error_context.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>

#include <algorithm>
#include <numeric>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

}  // namespace

TEMPLATE_TEST_CASE(
  "sirius_physical_grouped_aggregate grouped aggregates data_batch with single partition key, "
  "multiple aggregations on numerics",
  "[physical_grouped_aggregate]",
  int32_t,
  int64_t,
  float,
  double,
  int16_t,
  decimal64_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 1000;

  // Create test data with single group key column
  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate<Traits>(num_groups, 1, stream, mr);

  std::shared_ptr<data_batch> input_batch = sirius::make_data_batch(std::move(input_table), *space);

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
  sirius_physical_grouped_aggregate grouped_aggregator(context,
                                                       std::move(agg_result.output_types),
                                                       std::move(agg_result.aggregates),
                                                       std::move(agg_result.groups),
                                                       num_groups);

  auto outputs = grouped_aggregator.execute({input_batch});

  // Verify we got one output batch
  REQUIRE(outputs.size() == 1);

  // Compare output with expected using the validation utility
  // Sort both tables before comparison since aggregation order is not guaranteed
  bool tables_match =
    sirius::test::expect_data_batch_equivalent_to_table(outputs[0], expected_table->view(), true);
  REQUIRE(tables_match);
}

TEMPLATE_TEST_CASE(
  "sirius_physical_grouped_aggregate grouped aggregates data_batch with multiple partition key, "
  "multiple aggregations",
  "[physical_grouped_aggregate]",
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

  std::shared_ptr<data_batch> input_batch = sirius::make_data_batch(std::move(input_table), *space);

  // Create DuckDB context for aggregate function binding
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
    {0, 1},                   // group_indexes: GROUP BY column 0 and 1
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {2, 2, 2}                 // agg_indexes: MIN(column 2), MAX(column 2), COUNT(column 2)
  );

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate grouped_aggregator(context,
                                                       std::move(agg_result.output_types),
                                                       std::move(agg_result.aggregates),
                                                       std::move(agg_result.groups),
                                                       num_groups);

  auto outputs = grouped_aggregator.execute({input_batch});

  // Verify we got one output batch
  REQUIRE(outputs.size() == 1);

  // Compare output with expected using the validation utility
  // Sort both tables before comparison since aggregation order is not guaranteed
  bool tables_match =
    sirius::test::expect_data_batch_equivalent_to_table(outputs[0], expected_table->view(), true);
  REQUIRE(tables_match);
}
