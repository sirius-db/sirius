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

#include <cudf/column/column_view.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cstdint>

namespace sirius::vss {

/// [n_rows, dim], non-owning, row-major view over a contiguous block of FLOAT32 vectors
using dataset_matrix_view = raft::device_matrix_view<const float, int64_t, raft::row_major>;

/**
 * @brief Convert cuDF LIST column to a raft device matrix view.
 *
 * Sirius stores FLOAT[dim] as a cudf LIST column whose values child is a
 * contiguous @c [n_rows*dim] FLOAT32 buffer. cuVS brute-force requires a
 * device_matrix_view. For a fixed-size, gap-free list that is exactly the
 * values buffer reinterpreted, so this is a zero-copy reinterpretation.
 *
 * @warning NON-OWNING. The returned view borrows @p list_col's device memory.
 *          The caller must keep @p list_col (and its backing GPU allocation)
 *          alive AND resident on-device for the entire lifetime of the view,
 *          since cuVS reads it as a raw blob, so a free or a tiering spill (D2H)
 *          underneath it is a use-after-free that the C++ compiler will not catch.
 *
 * Preconditions (throw @c cudf::logic_error otherwise):
 *   - @p list_col is a LIST column,
 *   - its values child is FLOAT32,
 *   - @p list_col is unsliced (@c offset()==0) and has no nulls,
 *   - the values child holds exactly @c n_rows*dim elements (uniform lists).
 *
 * @param list_col Sirius ARRAY<FLOAT> column (cudf LIST).
 * @param dim      Fixed vector dimensionality (from the ARRAY logical type).
 * @return Non-owning @c [n_rows,dim] row-major device matrix view.
 */
dataset_matrix_view list_column_as_dataset_view(cudf::column_view const& list_col, int64_t dim);

}  // namespace sirius::vss
