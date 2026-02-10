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

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/parser/query_error_context.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_partition.hpp>

#include <numeric>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
}  // namespace

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
      values[i] = static_cast<typename Traits::type>(i * 1'000'000);
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
  std::shared_ptr<data_batch> input_batch0;  // batch for the aggregation key
  if constexpr (Traits::is_string) {
    input_batch0 = make_string_batch(*space, values);
  } else if constexpr (Traits::is_decimal) {
    input_batch0 = make_decimal64_batch(*space, values, Traits::scale);
  } else if constexpr (Traits::is_ts) {
    input_batch0 = make_timestamp_batch(*space, values, Traits::cudf_type);
  } else {
    input_batch0 = make_numeric_batch<typename Traits::type>(*space, values, Traits::cudf_type);
  }

  // create a batch for the aggregation value to just be a int32_t
  std::shared_ptr<data_batch> input_batch1;
  input_batch1 =
    make_numeric_batch<int32_t>(*space, std::vector<int32_t>(num_values, 1), cudf::type_id::INT32);

  // Concatenate the two batches horizontally (side by side columns)
  auto input_batch = concatenate_batches_horizontal({input_batch0, input_batch1}, *space);

  // this cardinality is not real, we are setting here this large in order to force more partitions
  // to be made
  std::size_t estimated_cardinality = 100000000;  // 100 million rows = PARTITION_SIZE x 10

  // Create DuckDB context for aggregate function binding
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<gpu_type_traits<int32_t>>(
    {0},      // group_indexes: GROUP BY column 0
    {"sum"},  // aggregations: SUM
    {1}       // agg_indexes: SUM(column 1)
  );

  // Create partitioner types (copy of agg_output_types before moving)
  duckdb::vector<duckdb::LogicalType> partitioner_types = agg_result.output_types;

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregator(context,
                                                             std::move(agg_result.output_types),
                                                             std::move(agg_result.aggregates),
                                                             std::move(agg_result.groups),
                                                             estimated_cardinality);

  sirius_physical_partition partitioner(
    std::move(partitioner_types), estimated_cardinality, &grouped_aggregator, false);

  auto outputs = partitioner.execute({input_batch}, default_stream());

  std::size_t partition_size = 10000000;  // from sirius_physical_partition.hpp

  std::size_t expected_num_partitions =
    (estimated_cardinality + partition_size - 1) / partition_size;

  REQUIRE(outputs.size() == expected_num_partitions);

  // count the number of rows in each output and make sure it's the same and the initial inputs
  std::size_t total_num_rows = 0;
  for (auto& output : outputs) {
    total_num_rows += output->get_data()->cast<gpu_table_representation>().get_table().num_rows();
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
  std::shared_ptr<data_batch> input_batch0;  // batch for the aggregation key0 and 1
  if constexpr (Traits::is_string) {
    input_batch0 = make_string_batch(*space, values0);
  } else if constexpr (Traits::is_decimal) {
    input_batch0 = make_decimal64_batch(*space, values0, Traits::scale);
  } else if constexpr (Traits::is_ts) {
    input_batch0 = make_timestamp_batch(*space, values0, Traits::cudf_type);
  } else {
    input_batch0 = make_numeric_batch<typename Traits::type>(*space, values0, Traits::cudf_type);
  }

  std::shared_ptr<data_batch> input_batch1 =
    make_numeric_batch<int32_t>(*space, values1, cudf::type_id::INT32);

  // we can make the third column which is for aggregation the same as the second column because the
  // values won't matter
  std::shared_ptr<data_batch> input_batch2 =
    make_numeric_batch<int32_t>(*space, values1, cudf::type_id::INT32);
  auto input_batch =
    concatenate_batches_horizontal({input_batch0, input_batch1, input_batch2}, *space);

  // this cardinality is not real, we are setting here this large in order to force more partitions
  // to be made
  std::size_t estimated_cardinality = 100000000;  // 100 million rows = PARTITION_SIZE x 10

  // Create DuckDB context for aggregate function binding
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<gpu_type_traits<int32_t>>(
    {0, 1},   // group_indexes: GROUP BY column 0 and 1
    {"min"},  // aggregations: MIN
    {2}       // agg_indexes: MIN(column 2)
  );

  // Create partitioner types (copy of agg_output_types before moving)
  duckdb::vector<duckdb::LogicalType> partitioner_types = agg_result.output_types;

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregator(context,
                                                             std::move(agg_result.output_types),
                                                             std::move(agg_result.aggregates),
                                                             std::move(agg_result.groups),
                                                             estimated_cardinality);

  sirius_physical_partition partitioner(
    std::move(partitioner_types), estimated_cardinality, &grouped_aggregator, false);

  auto outputs = partitioner.execute({input_batch}, default_stream());

  std::size_t partition_size = 10000000;  // from sirius_physical_partition.hpp

  std::size_t expected_num_partitions =
    (estimated_cardinality + partition_size - 1) / partition_size;

  REQUIRE(outputs.size() == expected_num_partitions);

  // count the number of rows in each output and make sure it's the same and the initial inputs
  std::size_t total_num_rows = 0;
  for (auto& output : outputs) {
    std::size_t num_rows_out =
      output->get_data()->cast<gpu_table_representation>().get_table().num_rows();
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
  std::shared_ptr<data_batch> input_batch0 =
    make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);

  // create a batch for the aggregation value to just be a int32_t
  std::shared_ptr<data_batch> input_batch1 =
    make_numeric_batch<int32_t>(*space, std::vector<int32_t>(num_values, 1), cudf::type_id::INT32);

  // Concatenate the two batches horizontally (side by side columns)
  auto input_batch = concatenate_batches_horizontal({input_batch0, input_batch1}, *space);

  std::size_t estimated_cardinality = num_values;

  // Create DuckDB context for aggregate function binding
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& context = *con.context;

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<gpu_type_traits<int32_t>>(
    {0},      // group_indexes: GROUP BY column 0
    {"sum"},  // aggregations: SUM
    {1}       // agg_indexes: SUM(column 1)
  );

  // Create partitioner types (copy of agg_output_types before moving)
  duckdb::vector<duckdb::LogicalType> partitioner_types = agg_result.output_types;

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate_merge grouped_aggregator(context,
                                                             std::move(agg_result.output_types),
                                                             std::move(agg_result.aggregates),
                                                             std::move(agg_result.groups),
                                                             estimated_cardinality);

  sirius_physical_partition partitioner(
    std::move(partitioner_types), estimated_cardinality, &grouped_aggregator, false);

  auto outputs = partitioner.execute({input_batch}, default_stream());
  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0]->get_data()->cast<gpu_table_representation>().get_table().num_rows() ==
          num_values);
}
