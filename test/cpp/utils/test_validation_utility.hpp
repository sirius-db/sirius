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
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <data/data_batch_utils.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
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

namespace detail {

template <typename T = char>
inline std::vector<T> copy_device_values(void const* source,
                                         std::size_t count,
                                         ::cuda::stream_ref stream)
{
  std::vector<T> host(count);
  if (count != 0) {
    auto const status =
      cudaMemcpyAsync(host.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost, stream.get());
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
    stream.sync();
  }
  return host;
}

/// Row-wise validity of @p column; every row of a column without a null mask is valid.
inline std::vector<char> column_row_validity(cudf::column_view const& column,
                                             ::cuda::stream_ref stream)
{
  std::vector<char> validity(static_cast<std::size_t>(column.size()), 1);
  if (!column.nullable() || validity.empty()) { return validity; }
  auto const mask  = cudf::copy_bitmask(column, stream, cudf::get_current_device_resource_ref());
  auto const words = copy_device_values(mask.data(), mask.size(), stream);
  for (std::size_t row = 0; row < validity.size(); ++row) {
    cudf::bitmask_type word = 0;
    std::memcpy(
      &word, words.data() + (row / (sizeof(cudf::bitmask_type) * 8)) * sizeof(word), sizeof(word));
    validity[row] = static_cast<char>((word >> (row % (sizeof(cudf::bitmask_type) * 8))) & 1U);
  }
  return validity;
}

inline bool supported_column_types_equal(cudf::column_view const& lhs, cudf::column_view const& rhs)
{
  if (lhs.type() != rhs.type()) { return false; }
  auto const id = lhs.type().id();
  // STRING offsets are storage, so INT32 and INT64 offsets represent the same logical type.
  if (id == cudf::type_id::STRING) { return true; }
  if (lhs.num_children() != rhs.num_children()) { return false; }
  if (!cudf::is_fixed_width(lhs.type()) && id != cudf::type_id::LIST &&
      id != cudf::type_id::STRUCT) {
    return false;
  }
  for (cudf::size_type child = 0; child < lhs.num_children(); ++child) {
    if (!supported_column_types_equal(lhs.child(child), rhs.child(child))) { return false; }
  }
  return true;
}

}  // namespace detail

/**
 * @brief Compares two columns under cuDF value semantics: type, row count, row validity and the
 * values of the valid rows. Fixed-width values must match bit-for-bit, including floating point.
 *
 * Allocation layout is ignored: null-mask padding is unspecified, the payload bytes behind a null
 * slot are unspecified, a column holding no nulls may or may not carry a mask at all, and a list
 * child may retain elements no surviving row references. Only fixed-width, STRING, LIST and STRUCT
 * columns are supported; any other type compares unequal so an unsupported payload cannot pass
 * silently.
 */
inline bool columns_logically_equal(cudf::column_view const& lhs,
                                    cudf::column_view const& rhs,
                                    ::cuda::stream_ref stream = cudf::get_default_stream())
{
  if (!detail::supported_column_types_equal(lhs, rhs) || lhs.size() != rhs.size()) { return false; }
  auto const validity = detail::column_row_validity(lhs, stream);
  if (validity != detail::column_row_validity(rhs, stream)) { return false; }
  if (validity.empty()) { return true; }

  if (lhs.type().id() == cudf::type_id::STRING) {
    auto const lhs_strings  = cudf::strings_column_view{lhs};
    auto const rhs_strings  = cudf::strings_column_view{rhs};
    auto const read_offsets = [stream](cudf::strings_column_view const& strings) {
      auto const offsets = strings.offsets();
      auto const count   = static_cast<std::size_t>(strings.size()) + 1;
      if (offsets.type().id() == cudf::type_id::INT64) {
        return detail::copy_device_values<int64_t>(
          offsets.data<int64_t>() + strings.offset(), count, stream);
      }
      auto const values = detail::copy_device_values<int32_t>(
        offsets.data<int32_t>() + strings.offset(), count, stream);
      return std::vector<int64_t>(values.begin(), values.end());
    };
    auto const lhs_offsets = read_offsets(lhs_strings);
    auto const rhs_offsets = read_offsets(rhs_strings);
    auto const lhs_chars =
      detail::copy_device_values(lhs_strings.chars_begin(stream) + lhs_offsets.front(),
                                 static_cast<std::size_t>(lhs_offsets.back() - lhs_offsets.front()),
                                 stream);
    auto const rhs_chars =
      detail::copy_device_values(rhs_strings.chars_begin(stream) + rhs_offsets.front(),
                                 static_cast<std::size_t>(rhs_offsets.back() - rhs_offsets.front()),
                                 stream);
    for (std::size_t row = 0; row < validity.size(); ++row) {
      if (validity[row] == 0) { continue; }
      auto const length = lhs_offsets[row + 1] - lhs_offsets[row];
      if (length != rhs_offsets[row + 1] - rhs_offsets[row]) { return false; }
      if (length > 0 && std::memcmp(lhs_chars.data() + lhs_offsets[row] - lhs_offsets.front(),
                                    rhs_chars.data() + rhs_offsets[row] - rhs_offsets.front(),
                                    static_cast<std::size_t>(length)) != 0) {
        return false;
      }
    }
    return true;
  }

  if (lhs.type().id() == cudf::type_id::LIST) {
    auto const lhs_lists   = cudf::lists_column_view{lhs};
    auto const rhs_lists   = cudf::lists_column_view{rhs};
    auto const lhs_offsets = detail::copy_device_values<cudf::size_type>(
      lhs_lists.offsets_begin(), validity.size() + 1, stream);
    auto const rhs_offsets = detail::copy_device_values<cudf::size_type>(
      rhs_lists.offsets_begin(), validity.size() + 1, stream);
    for (std::size_t row = 0; row < validity.size(); ++row) {
      if (validity[row] == 0) { continue; }
      std::vector<cudf::size_type> const lhs_range{lhs_offsets[row], lhs_offsets[row + 1]};
      std::vector<cudf::size_type> const rhs_range{rhs_offsets[row], rhs_offsets[row + 1]};
      if (lhs_range[1] - lhs_range[0] != rhs_range[1] - rhs_range[0]) { return false; }
      if (!columns_logically_equal(cudf::slice(lhs_lists.child(), lhs_range, stream).front(),
                                   cudf::slice(rhs_lists.child(), rhs_range, stream).front(),
                                   stream)) {
        return false;
      }
    }
    return true;
  }

  if (lhs.type().id() == cudf::type_id::STRUCT) {
    auto const lhs_structs = cudf::structs_column_view{lhs};
    auto const rhs_structs = cudf::structs_column_view{rhs};
    for (cudf::size_type child = 0; child < lhs.num_children(); ++child) {
      if (!columns_logically_equal(lhs_structs.get_sliced_child(child, stream),
                                   rhs_structs.get_sliced_child(child, stream),
                                   stream)) {
        return false;
      }
    }
    return true;
  }

  auto const width = cudf::size_of(lhs.type());
  auto const lhs_bytes =
    detail::copy_device_values(lhs.head<char>() + static_cast<std::size_t>(lhs.offset()) * width,
                               validity.size() * width,
                               stream);
  auto const rhs_bytes =
    detail::copy_device_values(rhs.head<char>() + static_cast<std::size_t>(rhs.offset()) * width,
                               validity.size() * width,
                               stream);
  for (std::size_t row = 0; row < validity.size(); ++row) {
    if (validity[row] == 0) { continue; }
    if (std::memcmp(lhs_bytes.data() + row * width, rhs_bytes.data() + row * width, width) != 0) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Compare table schemas and row-ordered values, ignoring null payloads and unused storage.
 *
 * Supports fixed-width, STRING, LIST and STRUCT columns. Fixed-width values are compared
 * bit-for-bit; unsupported types compare unequal. CUDA reads are ordered on @p stream.
 *
 * @param lhs First table view
 * @param rhs Second table view
 * @param stream CUDA stream on which the inputs are ready
 * @return true if tables are equivalent, false otherwise
 */
inline bool expect_tables_equivalent_impl(cudf::table_view lhs,
                                          cudf::table_view rhs,
                                          ::cuda::stream_ref stream = cudf::get_default_stream())
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

    bool const match = columns_logically_equal(lhs_col, rhs_col, stream);

    if (!match) {
      std::cout << "Column " << i << " schema, validity or values do not match" << std::endl;
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

  // Extract GPU table views from data batches via RAII read-only accessors
  auto lhs_view = sirius::get_cudf_table_view(*lhs);
  auto rhs_view = sirius::get_cudf_table_view(*rhs);

  // If sort is requested, sort both tables by all columns
  if (sort) {
    auto mr     = cudf::get_current_device_resource_ref();
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

  // Extract GPU table view from the data batch via RAII read-only accessor
  auto batch_view = sirius::get_cudf_table_view(*batch);

  // If sort is requested, sort both tables by all columns
  if (sort) {
    auto mr     = cudf::get_current_device_resource_ref();
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
