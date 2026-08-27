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

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>

namespace sirius::late_mat {

/// Gather `count` elements by GLOBAL pin-order id, across a table's batches, in
/// one pass — the alternative to sorting a selection just to split it.
///
/// For each id, the owning batch is found by binary search over `row_start_dev`
/// (B entries, the batches' first global rows) and `elem_size` bytes are copied
/// from `bases_dev[b]` at the batch-local offset. Ids may repeat and need not be
/// ordered: these are ordinary gather semantics, and the output is in the
/// caller's order.
///
/// `bases_dev` and `row_start_dev` are DEVICE arrays of B entries. `elem_size`
/// must be 1, 2, 4, 8 or 16 — a variable-width column has no element width to
/// copy and belongs on the canonical path. Ids must be valid pin-order
/// positions; they are not bounds-checked, matching cudf::gather's DONT_CHECK.
///
/// `masks_dev` is an optional DEVICE array of B per-batch null-mask pointers
/// (cudf::bitmask_type); a null entry means that batch has no nulls (the usual
/// cudf convention). Pass `masks_dev == nullptr` to skip validity work
/// entirely; otherwise `out_mask` must be sized for `count` bits.
///
/// Asynchronous on `stream`.
void multi_source_gather_fixed(void const* const* bases_dev,
                               std::int64_t const* row_start_dev,
                               int num_batches,
                               std::size_t elem_size,
                               std::uint64_t const* ids,
                               std::int64_t count,
                               void* out,
                               std::uint32_t const* const* masks_dev,
                               std::uint32_t* out_mask,
                               rmm::cuda_stream_view stream);

}  // namespace sirius::late_mat
