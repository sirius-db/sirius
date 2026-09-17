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

// Producing one deferred column, for a selection already prepared (env gate:
// SIRIUS_EXP_LATE_MAT).
//
// prepare_selection resolved the rows once, for every column of the origin
// table. This is the per-column half: for each batch the selection touches,
// read that batch's surviving rows out of the pinned storage, and assemble the
// results into one column in the order the caller asked for.
//
// The output's row order is the contract. Batches are visited in pin order and
// their rows come back ascending within a batch, so the assembled column is in
// pinned-table order — which is the caller's order exactly when the caller's
// ids were already sorted and unique. Otherwise the prepared selection carries
// the restore ranks, and one final gather puts the rows back in the order the
// caller asked, repeats included. That gather is over the narrow materialized
// column, never over the pinned data, which is what makes deduplicating worth
// it in the first place.

#include "late_mat/prepared_selection.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace sirius {
class compressed_device_representation;
}

namespace sirius::late_mat {

/// One pinned batch of one origin column. Exactly one form is populated, which
/// mirrors device_pin_chunk: compression is decided per chunk, so a single pin
/// may interleave the two.
struct batch_source {
  sirius::compressed_device_representation const* compressed = nullptr;
  std::size_t column_index                                   = 0;  ///< within the compressed chunk
  cudf::column_view uncompressed{};  ///< valid iff compressed == nullptr
  std::int64_t num_rows = 0;

  [[nodiscard]] bool is_compressed() const noexcept { return compressed != nullptr; }
};

/// One origin column across the pinned table's batches, positionally consistent
/// with the layout the selection was prepared against.
struct pinned_column_view {
  std::vector<batch_source> batches;
  cudf::data_type dtype{cudf::type_id::EMPTY};
  pin_generation_t pin_generation = 0;
};

/// Materialize `column` for `selection`.
///
/// The result has selection.original_count rows — the caller's own count, not
/// the deduplicated one — and is in the caller's order.
///
/// Throws if the column's batches disagree with the layout the selection was
/// prepared against, since a positional mismatch would read the right number of
/// rows from the wrong batches.
///
/// A COMPRESSED origin IS served here, by materialize_compressed: a dense batch
/// takes an ordinary full decode, and a selective one tries the sparse walk
/// (simpatico::decompress_column_rows over the CSR each batch already carries),
/// falling back to the mask route for the shapes with no random access
/// (dictionary, str_split, render rejections) and to a full decode as the last
/// resort. None of these routes writes an output validity buffer, so a decoded
/// column that turns out to contain nulls is rejected rather than returned
/// half-formed (require_non_null).
///
/// What this materializer can serve and what the INSTALLER admits are two
/// different questions. Today no compressed origin reaches here: the install
/// gate refuses one outright, because pinned_column_null_count cannot read a
/// compressed chunk's null count without decoding it and so reports "unknown",
/// which the nullability check treats as unsafe. These routes are therefore
/// exercised by their own tests rather than by a query, and they are what a
/// future relaxation of that gate would rest on.
std::unique_ptr<cudf::column> materialize(pinned_column_view const& column,
                                          prepared_selection const& selection,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

}  // namespace sirius::late_mat
