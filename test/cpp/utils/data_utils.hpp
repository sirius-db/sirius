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
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace sirius {
namespace test {

/**
 * @brief Create a cuDF column from a vector of values
 *
 * This function creates a cuDF column from a host vector, handling all types
 * supported by gpu_type_traits including numeric types, strings, decimals,
 * timestamps, and dates.
 *
 * @tparam Traits Type traits class (e.g., gpu_type_traits<int32_t>)
 * @param values Vector of values to convert to column
 * @param stream CUDA stream for async operations
 * @param mr Memory resource for GPU allocations
 * @return std::unique_ptr<cudf::column> The created cuDF column
 */
template <typename Traits>
inline std::unique_ptr<cudf::column> vector_to_cudf_column(
  const std::vector<typename Traits::type>& values,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  auto size = static_cast<cudf::size_type>(values.size());

  // Handle strings specially
  if constexpr (Traits::is_string) {
    // Build offsets and chars on host
    std::vector<cudf::size_type> offsets(size + 1, 0);
    cudf::size_type total_chars = 0;
    for (cudf::size_type i = 0; i < size; ++i) {
      offsets[i + 1] = offsets[i] + static_cast<cudf::size_type>(values[i].size());
      total_chars    = offsets[i + 1];
    }

    std::vector<char> chars(static_cast<std::size_t>(total_chars));
    cudf::size_type cursor = 0;
    for (cudf::size_type i = 0; i < size; ++i) {
      auto const& s = values[i];
      std::memcpy(chars.data() + cursor, s.data(), s.size());
      cursor += static_cast<cudf::size_type>(s.size());
    }

    // Offsets column
    auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                 static_cast<cudf::size_type>(offsets.size()),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
    cudaMemcpyAsync(offsets_col->mutable_view().data<cudf::size_type>(),
                    offsets.data(),
                    offsets.size() * sizeof(cudf::size_type),
                    cudaMemcpyHostToDevice,
                    stream.value());

    // Chars buffer
    rmm::device_buffer chars_buf(total_chars, stream, mr);
    if (total_chars > 0) {
      cudaMemcpyAsync(chars_buf.data(),
                      chars.data(),
                      chars.size() * sizeof(char),
                      cudaMemcpyHostToDevice,
                      stream.value());
    }

    return cudf::make_strings_column(
      size, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{0, stream, mr});
  }
  // Handle decimal types
  else if constexpr (Traits::is_decimal) {
    auto col = cudf::make_fixed_point_column(cudf::data_type{Traits::cudf_type, Traits::scale},
                                             size,
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr);
    cudaMemcpy(col->mutable_view().data<int64_t>(),
               values.data(),
               sizeof(int64_t) * values.size(),
               cudaMemcpyHostToDevice);
    return col;
  }
  // Handle timestamp/date types
  else if constexpr (Traits::is_ts) {
    auto col = cudf::make_timestamp_column(
      cudf::data_type{Traits::cudf_type}, size, cudf::mask_state::UNALLOCATED, stream, mr);

    // TIMESTAMP_DAYS uses 32-bit underlying storage; others use 64-bit
    if constexpr (Traits::cudf_type == cudf::type_id::TIMESTAMP_DAYS) {
      cudaMemcpy(col->mutable_view().data<int32_t>(),
                 values.data(),
                 sizeof(int32_t) * values.size(),
                 cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(col->mutable_view().data<int64_t>(),
                 values.data(),
                 sizeof(int64_t) * values.size(),
                 cudaMemcpyHostToDevice);
    }
    return col;
  }
  // Handle numeric types (including bool)
  else {
    using T  = typename Traits::type;
    auto col = cudf::make_numeric_column(
      cudf::data_type{Traits::cudf_type}, size, cudf::mask_state::UNALLOCATED, stream, mr);

    // Special handling for bool (stored as int8_t in cuDF)
    if constexpr (std::is_same_v<T, bool>) {
      std::vector<int8_t> tmp(values.size());
      for (size_t i = 0; i < values.size(); ++i) {
        tmp[i] = static_cast<int8_t>(values[i]);
      }
      cudaMemcpy(col->mutable_view().data<int8_t>(),
                 tmp.data(),
                 sizeof(int8_t) * tmp.size(),
                 cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(col->mutable_view().data<T>(),
                 values.data(),
                 sizeof(T) * values.size(),
                 cudaMemcpyHostToDevice);
    }
    return col;
  }
}

/**
 * @brief Split a table into random non-contiguous splits
 *
 * Takes an input table and splits it into multiple tables by randomly
 * distributing rows across the splits. This creates a "striped" pattern
 * where each split contains non-contiguous random rows from the input.
 *
 * @param input Input table to split (will be moved from)
 * @param num_splits Number of splits to create
 * @param stream CUDA stream for async operations
 * @param mr Memory resource for GPU allocations
 * @return std::vector<std::unique_ptr<cudf::table>> Vector of split tables
 *
 * @note The sum of rows across all splits equals the input table row count
 * @note Uses a random shuffle to distribute rows, so results are non-deterministic
 */
inline std::vector<std::unique_ptr<cudf::table>> make_random_striped_split(
  std::unique_ptr<cudf::table> input,
  std::size_t num_splits,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  if (num_splits == 0) { return {}; }

  auto num_rows = input->num_rows();
  if (num_rows == 0) { return {}; }

  // Create a vector of all row indices
  std::vector<cudf::size_type> all_indices(num_rows);
  std::iota(all_indices.begin(), all_indices.end(), 0);

  // Shuffle the indices randomly
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(all_indices.begin(), all_indices.end(), gen);

  // Calculate rows per split (distribute evenly with remainder going to last split)
  auto base_rows_per_split = num_rows / num_splits;
  auto remainder           = num_rows % num_splits;

  std::vector<std::unique_ptr<cudf::table>> result;
  result.reserve(num_splits);

  cudf::size_type offset = 0;
  for (std::size_t i = 0; i < num_splits; ++i) {
    // Calculate how many rows this split gets
    auto rows_in_split = base_rows_per_split + (i < remainder ? 1 : 0);

    if (rows_in_split == 0) { continue; }

    // Create gather map for this split
    std::vector<cudf::size_type> split_indices(all_indices.begin() + offset,
                                               all_indices.begin() + offset + rows_in_split);

    // Create cuDF column from indices for gather map
    auto gather_map = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                static_cast<cudf::size_type>(split_indices.size()),
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);

    cudaMemcpyAsync(gather_map->mutable_view().data<cudf::size_type>(),
                    split_indices.data(),
                    split_indices.size() * sizeof(cudf::size_type),
                    cudaMemcpyHostToDevice,
                    stream.value());

    // Use cuDF gather to create the split table
    auto split_table = cudf::gather(
      input->view(), gather_map->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

    result.push_back(std::move(split_table));
    offset += rows_in_split;
  }

  return result;
}

}  // namespace test
}  // namespace sirius
