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
/// COMPRESSED ORIGINS ARE NOT YET SERVED and are refused rather than silently
/// decoded full width. The sparse decode they need exists and is tested
/// (codegen/selection/chunk_row_set.hpp, and the CSR each batch already
/// carries), but reaching it needs a decompression entry point that takes a
/// selection for a column subset; the shipped surface only takes one inside
/// decompress_scan_filter, which builds its own.
std::unique_ptr<cudf::column> materialize(pinned_column_view const& column,
                                          prepared_selection const& selection,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

}  // namespace sirius::late_mat
