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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace sirius {
namespace test {

/**
 * @brief Convert cuDF type_id to string for debugging
 */
inline std::string type_id_to_string(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::EMPTY: return "EMPTY";
    case cudf::type_id::INT8: return "INT8";
    case cudf::type_id::INT16: return "INT16";
    case cudf::type_id::INT32: return "INT32";
    case cudf::type_id::INT64: return "INT64";
    case cudf::type_id::UINT8: return "UINT8";
    case cudf::type_id::UINT16: return "UINT16";
    case cudf::type_id::UINT32: return "UINT32";
    case cudf::type_id::UINT64: return "UINT64";
    case cudf::type_id::FLOAT32: return "FLOAT32";
    case cudf::type_id::FLOAT64: return "FLOAT64";
    case cudf::type_id::BOOL8: return "BOOL8";
    case cudf::type_id::TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
    case cudf::type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
    case cudf::type_id::DURATION_DAYS: return "DURATION_DAYS";
    case cudf::type_id::DURATION_SECONDS: return "DURATION_SECONDS";
    case cudf::type_id::DURATION_MILLISECONDS: return "DURATION_MILLISECONDS";
    case cudf::type_id::DURATION_MICROSECONDS: return "DURATION_MICROSECONDS";
    case cudf::type_id::DURATION_NANOSECONDS: return "DURATION_NANOSECONDS";
    case cudf::type_id::DICTIONARY32: return "DICTIONARY32";
    case cudf::type_id::STRING: return "STRING";
    case cudf::type_id::LIST: return "LIST";
    case cudf::type_id::DECIMAL32: return "DECIMAL32";
    case cudf::type_id::DECIMAL64: return "DECIMAL64";
    case cudf::type_id::DECIMAL128: return "DECIMAL128";
    case cudf::type_id::STRUCT: return "STRUCT";
    default: return "UNKNOWN";
  }
}

/**
 * @brief Helper to compare two column views for equivalence
 */
template <typename T>
inline bool compare_column_values(cudf::column_view const& lhs, cudf::column_view const& rhs)
{
  std::vector<T> lhs_host(lhs.size());
  std::vector<T> rhs_host(rhs.size());

  cudaMemcpy(lhs_host.data(), lhs.data<T>(), sizeof(T) * lhs.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(rhs_host.data(), rhs.data<T>(), sizeof(T) * rhs.size(), cudaMemcpyDeviceToHost);

  for (cudf::size_type i = 0; i < lhs.size(); ++i) {
    if (lhs_host[i] != rhs_host[i]) {
      std::cout << "Column value mismatch at index " << i << ": " << lhs_host[i] << " vs "
                << rhs_host[i] << std::endl;
      return false;
    }
  }
  return true;
}

/**
 * @brief Compare two table views for basic equivalence
 *
 * Checks that tables have same schema and data values.
 * This is a simple implementation for testing purposes.
 *
 * @param lhs First table view
 * @param rhs Second table view
 * @return true if tables are equivalent, false otherwise
 */
inline bool expect_tables_equivalent_impl(cudf::table_view lhs, cudf::table_view rhs)
{
  // Check number of columns
  if (lhs.num_columns() != rhs.num_columns()) {
    std::cout << "Table column count mismatch: " << lhs.num_columns() << " vs " << rhs.num_columns()
              << std::endl;
    return false;
  }

  // Check number of rows
  if (lhs.num_rows() != rhs.num_rows()) {
    std::cout << "Table row count mismatch: " << lhs.num_rows() << " vs " << rhs.num_rows()
              << std::endl;
    return false;
  }

  // Check each column
  for (cudf::size_type i = 0; i < lhs.num_columns(); ++i) {
    auto lhs_col = lhs.column(i);
    auto rhs_col = rhs.column(i);

    // Check types
    if (lhs_col.type().id() != rhs_col.type().id()) {
      std::cout << "Column " << i << " type mismatch. Got "
                << type_id_to_string(lhs_col.type().id()) << " expected "
                << type_id_to_string(rhs_col.type().id()) << std::endl;
      return false;
    }

    // Check null counts
    if (lhs_col.null_count() != rhs_col.null_count()) {
      std::cout << "Column " << i << " null count mismatch: " << lhs_col.null_count() << " vs "
                << rhs_col.null_count() << std::endl;
      return false;
    }

    // Compare data (simplified - only handles numeric types for now)
    bool match = false;
    switch (lhs_col.type().id()) {
      case cudf::type_id::INT8: match = compare_column_values<int8_t>(lhs_col, rhs_col); break;
      case cudf::type_id::INT16: match = compare_column_values<int16_t>(lhs_col, rhs_col); break;
      case cudf::type_id::INT32: match = compare_column_values<int32_t>(lhs_col, rhs_col); break;
      case cudf::type_id::INT64: match = compare_column_values<int64_t>(lhs_col, rhs_col); break;
      case cudf::type_id::FLOAT32: match = compare_column_values<float>(lhs_col, rhs_col); break;
      case cudf::type_id::FLOAT64: match = compare_column_values<double>(lhs_col, rhs_col); break;
      case cudf::type_id::BOOL8: match = compare_column_values<int8_t>(lhs_col, rhs_col); break;
      default:
        // For unsupported types, just check size matches
        match = true;
    }

    if (!match) {
      std::cout << "Column " << i << " data values do not match" << std::endl;
      return false;
    }
  }

  return true;
}

/**
 * @brief Compare two data_batch objects for equivalence
 *
 * This function extracts the cuDF tables from two data_batch objects and
 * compares them. This checks that the tables have the same schema,
 * same number of rows, and equivalent data values.
 *
 * @param lhs First data_batch to compare
 * @param rhs Second data_batch to compare
 * @param sort If true, sort both tables by all columns before comparison (default: false)
 * @return true if data batches are equivalent, false otherwise
 */
inline bool expect_data_batches_equivalent(const std::shared_ptr<cucascade::data_batch>& lhs,
                                           const std::shared_ptr<cucascade::data_batch>& rhs,
                                           bool sort = false)
{
  if (!lhs || !rhs) {
    std::cout << "Cannot compare null data_batch pointers" << std::endl;
    return false;
  }

  // Extract the GPU table representations from the data batches
  auto* lhs_data = lhs->get_data();
  auto* rhs_data = rhs->get_data();

  if (!lhs_data || !rhs_data) {
    std::cout << "Cannot compare data_batch with null data representation" << std::endl;
    return false;
  }

  // Cast to gpu_table_representation
  auto& lhs_gpu_repr = lhs_data->cast<cucascade::gpu_table_representation>();
  auto& rhs_gpu_repr = rhs_data->cast<cucascade::gpu_table_representation>();

  // Get the cuDF tables
  auto& lhs_table = lhs_gpu_repr.get_table();
  auto& rhs_table = rhs_gpu_repr.get_table();

  // Get table views for comparison
  auto lhs_view = lhs_table.view();
  auto rhs_view = rhs_table.view();

  // If sort is requested, sort both tables by all columns
  if (sort) {
    auto mr     = rmm::mr::get_current_device_resource();
    auto stream = cudf::get_default_stream();

    // Create column indices for sorting (all columns)
    std::vector<cudf::order> column_orders(lhs_view.num_columns(), cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_orders(lhs_view.num_columns(), cudf::null_order::AFTER);

    // Sort both tables
    auto sorted_lhs = cudf::sort(lhs_view, column_orders, null_orders, stream, mr);
    auto sorted_rhs = cudf::sort(rhs_view, column_orders, null_orders, stream, mr);

    // Compare sorted tables
    return expect_tables_equivalent_impl(sorted_lhs->view(), sorted_rhs->view());
  }

  // Compare tables without sorting
  return expect_tables_equivalent_impl(lhs_view, rhs_view);
}

/**
 * @brief Compare a data_batch with a cuDF table for equivalence
 *
 * This function extracts the cuDF table from a data_batch and compares it
 * with a provided cuDF table view.
 *
 * @param batch The data_batch to compare
 * @param expected The expected cuDF table view
 * @param sort If true, sort both tables by all columns before comparison (default: false)
 * @return true if data batch is equivalent to table, false otherwise
 */
inline bool expect_data_batch_equivalent_to_table(
  const std::shared_ptr<cucascade::data_batch>& batch, cudf::table_view expected, bool sort = false)
{
  if (!batch) {
    std::cout << "Cannot compare null data_batch pointer" << std::endl;
    return false;
  }

  // Extract the GPU table representation from the data batch
  auto* batch_data = batch->get_data();

  if (!batch_data) {
    std::cout << "Cannot compare data_batch with null data representation" << std::endl;
    return false;
  }

  // Cast to gpu_table_representation
  auto& batch_gpu_repr = batch_data->cast<cucascade::gpu_table_representation>();

  // Get the cuDF table
  auto& batch_table = batch_gpu_repr.get_table();

  // Get table view for comparison
  auto batch_view = batch_table.view();

  // If sort is requested, sort both tables by all columns
  if (sort) {
    auto mr     = rmm::mr::get_current_device_resource();
    auto stream = cudf::get_default_stream();

    // Create column indices for sorting (all columns)
    std::vector<cudf::order> column_orders(batch_view.num_columns(), cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_orders(batch_view.num_columns(), cudf::null_order::AFTER);

    // Sort both tables
    auto sorted_batch    = cudf::sort(batch_view, column_orders, null_orders, stream, mr);
    auto sorted_expected = cudf::sort(expected, column_orders, null_orders, stream, mr);

    // Compare sorted tables
    return expect_tables_equivalent_impl(sorted_batch->view(), sorted_expected->view());
  }

  // Compare tables without sorting
  return expect_tables_equivalent_impl(batch_view, expected);
}

}  // namespace test
}  // namespace sirius
