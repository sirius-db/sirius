/*
 * Copyright 2026, Sirius Contributors.
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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/**
 * @brief Device work behind `sirius_physical_replicate`
 *
 * Repeats each row of a table by a count column. `plan_slices` cuts the whole expansion into slices
 * of consecutive output rows before any row is copied, and `materialize` copies one slice. Output
 * row `j` is a copy of input row `i` when `S[i-1] <= j < S[i]`, where `S` is the inclusive `INT64`
 * prefix sum of the counts. Each `cudf::repeat` call covers one slice, so its unchecked 32-bit
 * count sum never overflows.
 */
namespace sirius::op::gpu_replicate_impl {

/**
 * @brief Caps on one output table
 *
 * Both caps are positive. A `max_bytes` past `INT64_MAX` acts as `INT64_MAX`.
 */
struct limits {
  cudf::size_type max_rows;  ///< Rows per output table
  std::size_t max_bytes;     ///< Bytes per output table, exceeded by at most one row
};

/**
 * @brief Output rows `[output_begin, output_end)` of the whole expansion, and the input rows
 * `[input_begin, input_end)` they copy
 */
struct slice {
  std::int64_t output_begin;    ///< First output row
  std::int64_t output_end;      ///< One past the last output row
  cudf::size_type input_begin;  ///< First input row with a copy in the slice
  cudf::size_type input_end;    ///< One past the last input row with a copy in the slice
};

/**
 * @brief The slices one input table expands into
 */
struct plan {
  std::unique_ptr<cudf::column> row_prefix;  ///< `S`, `INT64`, one row per input row
  std::vector<slice> slices;  ///< Non-empty, in order, tiling `[0, S[last])`; empty if no copies
};

/**
 * @brief Cuts the expansion of @p data by @p counts into slices within @p caps
 *
 * A row's size is its `cudf::row_bit_count` in whole bytes, at least one. The sum over all rows of
 * `count * size` must fit in `INT64`; nothing checks it.
 *
 * @throws sirius::internal_exception if @p counts is not an integer column, differs from @p data in
 * length, or has a null or a negative value
 * @throws sirius::internal_exception if a cap in @p caps is not positive
 *
 * @param data Rows to repeat
 * @param counts Copies of each row of @p data
 * @param caps Caps on each slice
 * @param stream CUDA stream for device work and copies
 * @param mr Memory resource for the returned row prefix and temporary device memory
 * @return The row prefix and the slices, in output order
 */
plan plan_slices(cudf::table_view const& data,
                 cudf::column_view const& counts,
                 limits const& caps,
                 ::cuda::stream_ref stream,
                 rmm::device_async_resource_ref mr);

/**
 * @brief Copies one slice of an expansion
 *
 * @param data The table @p expansion was planned for
 * @param expansion What `plan_slices` returned for @p data
 * @param part One of `expansion.slices`
 * @param stream CUDA stream for device work
 * @param mr Memory resource for the returned table and temporary device memory
 * @return Output rows `[part.output_begin, part.output_end)` of the expansion
 */
std::unique_ptr<cudf::table> materialize(cudf::table_view const& data,
                                         plan const& expansion,
                                         slice const& part,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace sirius::op::gpu_replicate_impl
