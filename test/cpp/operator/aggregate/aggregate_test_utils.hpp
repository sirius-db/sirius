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

#pragma once

#include "operator/operator_type_traits.hpp"
#include "utils/data_utils.hpp"

#include <cudf/table/table.hpp>

#include <duckdb.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace sirius {
namespace test {

/**
 * @brief Result structure for create_aggregate_expressions
 */
struct AggregateExpressionResult {
  duckdb::vector<duckdb::LogicalType> output_types;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
};

/**
 * @brief Helper to create a dummy AggregateFunction for testing
 *
 * Creates a minimal AggregateFunction with just name and types,
 * suitable for GPU operator testing where full aggregate logic isn't needed.
 */
inline duckdb::AggregateFunction MakeDummyAggregate(const std::string& name,
                                                    const duckdb::vector<duckdb::LogicalType>& args,
                                                    const duckdb::LogicalType& ret_type)
{
  return duckdb::AggregateFunction(
    name, args, ret_type, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

/**
 * @brief Create DuckDB aggregate expressions for testing
 *
 * This utility function creates the necessary DuckDB expressions for grouped aggregations,
 * which are commonly used in physical operator tests.
 *
 * @tparam Traits Type traits class providing logical_type() method
 * @param group_indexes Column indices for GROUP BY expressions
 * @param aggregations Names of aggregation functions (e.g., "sum", "count", "avg")
 * @param agg_indexes Column indices for aggregation input expressions
 * @return AggregateExpressionResult containing output types, group expressions, and aggregate
 * expressions
 */
template <typename Traits>
AggregateExpressionResult create_aggregate_expressions(
  const std::vector<std::size_t>& group_indexes,
  const std::vector<std::string>& aggregations,
  const std::vector<std::size_t>& agg_indexes)
{
  AggregateExpressionResult result;

  // Create output types: first the group by column types, then the aggregate result types
  for (std::size_t group_idx : group_indexes) {
    result.output_types.push_back(Traits::logical_type());
  }
  for (std::size_t i = 0; i < aggregations.size(); ++i) {
    if (aggregations[i] == "avg") {
      // DECIMAL AVG preserves the DECIMAL type; non-DECIMAL AVG returns DOUBLE
      if constexpr (Traits::is_decimal) {
        result.output_types.push_back(Traits::logical_type());
      } else {
        result.output_types.push_back(duckdb::LogicalType::DOUBLE);
      }
    } else {
      result.output_types.push_back(Traits::logical_type());
    }
  }

  // Create group by expressions
  for (std::size_t group_idx : group_indexes) {
    result.groups.push_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(Traits::logical_type(), group_idx));
  }

  // Create aggregate expressions
  for (std::size_t i = 0; i < aggregations.size(); ++i) {
    const std::string& agg_name = aggregations[i];
    std::size_t agg_idx         = agg_indexes[i];

    // AVG on DECIMAL preserves the DECIMAL type; on other types returns DOUBLE
    duckdb::LogicalType ret_type;
    if (agg_name == "avg") {
      if constexpr (Traits::is_decimal) {
        ret_type = Traits::logical_type();
      } else {
        ret_type = duckdb::LogicalType::DOUBLE;
      }
    } else {
      ret_type = Traits::logical_type();
    }

    // Create children for the aggregate (the column to aggregate)
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> agg_children;
    agg_children.push_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(Traits::logical_type(), agg_idx));

    // Create the dummy aggregate function
    duckdb::AggregateFunction agg_function =
      MakeDummyAggregate(agg_name, {Traits::logical_type()}, ret_type);

    // Create the BoundAggregateExpression
    auto agg_expr =
      duckdb::make_uniq<duckdb::BoundAggregateExpression>(agg_function,
                                                          std::move(agg_children),
                                                          nullptr,  // filter
                                                          nullptr,  // bind_info
                                                          duckdb::AggregateType::NON_DISTINCT);

    result.aggregates.push_back(std::move(agg_expr));
  }

  return result;
}

/**
 * @brief Create test data for grouped aggregate operations
 *
 * This utility function creates input and expected output tables for testing
 * grouped aggregate operators with configurable group key columns.
 *
 * @tparam Traits Type traits class providing type information and logical_type() method
 * @param num_groups Number of groups to generate
 * @param num_group_key_columns Number of group key columns (1 or 2)
 * @param stream CUDA stream for cuDF operations
 * @param mr Memory resource for cuDF allocations
 * @return std::pair<input_table, expected_table> where:
 *         - input_table contains group keys and values for aggregation
 *         - expected_table contains group keys followed by min, max, and count aggregations
 *
 * @note When num_group_key_columns == 2, the second group key column is always a string type
 * @note The expected table includes min, max, and count aggregations
 */
template <typename Traits>
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
make_test_data_for_grouped_aggregate(std::size_t num_groups,
                                     std::size_t num_group_key_columns,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  std::vector<int32_t> group_sizes(num_groups);
  std::iota(group_sizes.begin(), group_sizes.end(), 1);
  std::size_t total_num_values = std::accumulate(group_sizes.begin(), group_sizes.end(), 0);

  using namespace operator_utils;

  // Prepare input vectors
  std::vector<typename Traits::type> key_values0(total_num_values);
  std::vector<std::string> key_values1(total_num_values);  // Used when num_group_key_columns == 2
  std::vector<typename Traits::type> value_values(total_num_values);

  // Prepare expected vectors
  std::vector<typename Traits::type> expected_group_by0(num_groups);
  std::vector<std::string> expected_group_by1(num_groups);  // Used when num_group_key_columns == 2
  std::vector<typename Traits::type> expected_min(num_groups);
  std::vector<typename Traits::type> expected_max(num_groups);
  std::vector<int32_t> expected_count(num_groups);

  // Populate data for each group
  std::size_t offset = 0;
  for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
    std::size_t num_values = group_sizes[group_idx];

    // Set group keys
    if constexpr (Traits::is_string) {
      std::fill(key_values0.begin() + offset,
                key_values0.begin() + offset + num_values,
                std::to_string(group_idx));
    } else {
      std::fill(key_values0.begin() + offset,
                key_values0.begin() + offset + num_values,
                static_cast<typename Traits::type>(group_idx));
    }

    if (num_group_key_columns == 2) {
      std::fill(key_values1.begin() + offset,
                key_values1.begin() + offset + num_values,
                std::to_string(group_idx));
    }

    // Set values for aggregation
    if constexpr (Traits::is_string) {
      std::fill(value_values.begin() + offset,
                value_values.begin() + offset + num_values,
                std::to_string(group_idx));
    } else {
      std::iota(value_values.begin() + offset,
                value_values.begin() + offset + num_values,
                static_cast<typename Traits::type>(-group_idx) / 2);
    }

    // Set expected values
    if constexpr (Traits::is_string) {
      expected_group_by0[group_idx] = std::to_string(group_idx);
    } else {
      expected_group_by0[group_idx] = group_idx;
    }

    if (num_group_key_columns == 2) { expected_group_by1[group_idx] = std::to_string(group_idx); }

    expected_min[group_idx] =
      *std::min_element(value_values.begin() + offset, value_values.begin() + offset + num_values);
    expected_max[group_idx] =
      *std::max_element(value_values.begin() + offset, value_values.begin() + offset + num_values);
    expected_count[group_idx] = num_values;

    offset += num_values;
  }

  // Create input table
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  input_columns.push_back(vector_to_cudf_column<Traits>(key_values0, stream, mr));
  if (num_group_key_columns == 2) {
    input_columns.push_back(
      vector_to_cudf_column<operator_utils::gpu_type_traits<operator_utils::string_tag>>(
        key_values1, stream, mr));
  }
  input_columns.push_back(vector_to_cudf_column<Traits>(value_values, stream, mr));
  auto input_table = std::make_unique<cudf::table>(std::move(input_columns));

  // Create expected table
  std::vector<std::unique_ptr<cudf::column>> expected_cols;
  expected_cols.push_back(vector_to_cudf_column<Traits>(expected_group_by0, stream, mr));
  if (num_group_key_columns == 2) {
    expected_cols.push_back(
      vector_to_cudf_column<operator_utils::gpu_type_traits<operator_utils::string_tag>>(
        expected_group_by1, stream, mr));
  }
  expected_cols.push_back(vector_to_cudf_column<Traits>(expected_min, stream, mr));
  expected_cols.push_back(vector_to_cudf_column<Traits>(expected_max, stream, mr));
  expected_cols.push_back(
    vector_to_cudf_column<operator_utils::gpu_type_traits<int32_t>>(expected_count, stream, mr));
  auto expected_table = std::make_unique<cudf::table>(std::move(expected_cols));

  return {std::move(input_table), std::move(expected_table)};
}

/**
 * @brief Create test data for grouped aggregate operations that include AVG.
 *
 * The expected table contains group keys followed by min, max, count, and avg columns.
 * AVG is computed as FLOAT64 regardless of input type.
 *
 * @tparam Traits Type traits class providing type information
 * @param num_groups Number of groups to generate
 * @param num_group_key_columns Number of group key columns (1 or 2)
 * @param stream CUDA stream for cuDF operations
 * @param mr Memory resource for cuDF allocations
 * @return std::pair<input_table, expected_table>
 */
template <typename Traits>
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
make_test_data_for_grouped_aggregate_with_avg(std::size_t num_groups,
                                              std::size_t num_group_key_columns,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  std::vector<int32_t> group_sizes(num_groups);
  std::iota(group_sizes.begin(), group_sizes.end(), 1);
  std::size_t total_num_values = std::accumulate(group_sizes.begin(), group_sizes.end(), 0);

  using namespace operator_utils;

  // Prepare input vectors
  std::vector<typename Traits::type> key_values0(total_num_values);
  std::vector<std::string> key_values1(total_num_values);
  std::vector<typename Traits::type> value_values(total_num_values);

  // Prepare expected vectors
  std::vector<typename Traits::type> expected_group_by0(num_groups);
  std::vector<std::string> expected_group_by1(num_groups);
  std::vector<typename Traits::type> expected_min(num_groups);
  std::vector<typename Traits::type> expected_max(num_groups);
  std::vector<int32_t> expected_count(num_groups);

  // For DECIMAL, AVG preserves the DECIMAL type; otherwise FLOAT64
  std::vector<double> expected_avg_f64(num_groups);
  std::vector<typename Traits::type> expected_avg_dec(num_groups);

  std::size_t offset = 0;
  for (std::size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
    std::size_t num_values = group_sizes[group_idx];

    if constexpr (Traits::is_string) {
      std::fill(key_values0.begin() + offset,
                key_values0.begin() + offset + num_values,
                std::to_string(group_idx));
    } else {
      std::fill(key_values0.begin() + offset,
                key_values0.begin() + offset + num_values,
                static_cast<typename Traits::type>(group_idx));
    }

    if (num_group_key_columns == 2) {
      std::fill(key_values1.begin() + offset,
                key_values1.begin() + offset + num_values,
                std::to_string(group_idx));
    }

    if constexpr (Traits::is_string) {
      std::fill(value_values.begin() + offset,
                value_values.begin() + offset + num_values,
                std::to_string(group_idx));
    } else {
      std::iota(value_values.begin() + offset,
                value_values.begin() + offset + num_values,
                static_cast<typename Traits::type>(-static_cast<int>(group_idx)) / 2);
    }

    if constexpr (Traits::is_string) {
      expected_group_by0[group_idx] = std::to_string(group_idx);
    } else {
      expected_group_by0[group_idx] = static_cast<typename Traits::type>(group_idx);
    }

    if (num_group_key_columns == 2) { expected_group_by1[group_idx] = std::to_string(group_idx); }

    expected_min[group_idx] =
      *std::min_element(value_values.begin() + offset, value_values.begin() + offset + num_values);
    expected_max[group_idx] =
      *std::max_element(value_values.begin() + offset, value_values.begin() + offset + num_values);
    expected_count[group_idx] = static_cast<int32_t>(num_values);

    // Compute AVG
    double sum = 0.0;
    for (std::size_t j = 0; j < num_values; ++j) {
      sum += static_cast<double>(value_values[offset + j]);
    }
    double avg = sum / static_cast<double>(num_values);

    if constexpr (Traits::is_decimal) {
      // For DECIMAL: the GPU does fixed-point integer division (sum_underlying / count),
      // which truncates toward zero, preserving full decimal precision.
      auto sum_underlying = static_cast<int64_t>(sum);
      expected_avg_dec[group_idx] =
        static_cast<typename Traits::type>(sum_underlying / static_cast<int64_t>(num_values));
    } else {
      expected_avg_f64[group_idx] = avg;
    }

    offset += num_values;
  }

  // Create input table
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  input_columns.push_back(vector_to_cudf_column<Traits>(key_values0, stream, mr));
  if (num_group_key_columns == 2) {
    input_columns.push_back(
      vector_to_cudf_column<operator_utils::gpu_type_traits<operator_utils::string_tag>>(
        key_values1, stream, mr));
  }
  input_columns.push_back(vector_to_cudf_column<Traits>(value_values, stream, mr));
  auto input_table = std::make_unique<cudf::table>(std::move(input_columns));

  // Create expected table: group keys, min, max, count, avg
  std::vector<std::unique_ptr<cudf::column>> expected_cols;
  expected_cols.push_back(vector_to_cudf_column<Traits>(expected_group_by0, stream, mr));
  if (num_group_key_columns == 2) {
    expected_cols.push_back(
      vector_to_cudf_column<operator_utils::gpu_type_traits<operator_utils::string_tag>>(
        expected_group_by1, stream, mr));
  }
  expected_cols.push_back(vector_to_cudf_column<Traits>(expected_min, stream, mr));
  expected_cols.push_back(vector_to_cudf_column<Traits>(expected_max, stream, mr));
  expected_cols.push_back(
    vector_to_cudf_column<operator_utils::gpu_type_traits<int32_t>>(expected_count, stream, mr));
  if constexpr (Traits::is_decimal) {
    expected_cols.push_back(vector_to_cudf_column<Traits>(expected_avg_dec, stream, mr));
  } else {
    expected_cols.push_back(
      vector_to_cudf_column<operator_utils::gpu_type_traits<double>>(expected_avg_f64, stream, mr));
  }
  auto expected_table = std::make_unique<cudf::table>(std::move(expected_cols));

  return {std::move(input_table), std::move(expected_table)};
}

}  // namespace test
}  // namespace sirius
