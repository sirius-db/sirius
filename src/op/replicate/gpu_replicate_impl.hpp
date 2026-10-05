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

//! Device work behind `sirius_physical_replicate`: repeat each row of a table by a count column.
//! `plan_slices` cuts the whole expansion into slices of consecutive output rows before any row is
//! copied, and `materialize` copies one slice. Output row `j` is a copy of input row `i` when
//! `S[i-1] <= j < S[i]`, where `S` is the inclusive `INT64` prefix sum of the counts. Each
//! `cudf::repeat` call covers one slice, so its unchecked 32-bit count sum never overflows.
namespace sirius::op::gpu_replicate_impl {

//! Caps on one output table.
struct limits {
  cudf::size_type max_rows;  //!< Rows per output table; positive.
  std::size_t max_bytes;     //!< Bytes per output table, exceeded by at most one row; positive.
};

//! Output rows `[lo, hi)` of the whole expansion, and the input rows `[first_row, end_row)` they
//! copy.
struct slice {
  std::int64_t lo;            //!< First output row.
  std::int64_t hi;            //!< One past the last output row.
  cudf::size_type first_row;  //!< First input row with a copy in the slice.
  cudf::size_type end_row;    //!< One past the last input row with a copy in the slice.
};

//! The slices one input table expands into.
struct plan {
  std::unique_ptr<cudf::column> row_prefix;  //!< `S`, `INT64`, one row per input row.
  std::vector<slice> slices;  //!< Non-empty, in order, tiling `[0, S[last])`; empty if no copies.
};

//! Cuts the expansion of @p data by @p counts into slices of at most `caps.max_rows` rows and
//! `caps.max_bytes` bytes plus one row, a row's size being its `cudf::row_bit_count` in whole
//! bytes. Throws `sirius::internal_exception` if @p counts is not an integer column, has a null
//! or a negative value, or differs from @p data in length.
plan plan_slices(cudf::table_view const& data,
                 cudf::column_view const& counts,
                 limits const& caps,
                 ::cuda::stream_ref stream,
                 rmm::device_async_resource_ref mr);

//! Output rows `[part.lo, part.hi)` of @p expansion, which `plan_slices` returned for @p data.
std::unique_ptr<cudf::table> materialize(cudf::table_view const& data,
                                         plan const& expansion,
                                         slice const& part,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace sirius::op::gpu_replicate_impl
