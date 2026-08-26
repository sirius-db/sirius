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

/// One column's buffers inside one HOST-tier pinned chunk, as a gather has to
/// address them.
///
/// Host-tier storage is a list of equally sized pinned blocks that are NOT
/// contiguous with one another, so a column's buffer has no single base pointer
/// and cannot be described by a cudf::column_view at all. What it does have is a
/// byte offset into the logical concatenation of those blocks, which the gather
/// translates to (block, offset) per element.
///
/// The block pointers are the host addresses. Under unified virtual addressing a
/// registered pinned host pointer is also a valid device pointer, which is what
/// lets the gather read the rows it wants where they lie instead of staging the
/// whole column back to the device; the resolver refuses the pin outright when
/// the device does not offer that.
struct host_blocked_buffers {
  std::vector<void*> blocks;  ///< block bases in allocation order, device-addressable
  std::size_t block_size       = 0;
  std::size_t data_offset      = 0;  ///< first data byte, logical offset over `blocks`
  std::size_t null_mask_offset = 0;  ///< first mask byte, same coordinates; ignored unless
                                     ///< `has_null_mask`
  bool has_null_mask = false;
};

/// One pinned batch of one origin column. Exactly one form is populated, which
/// mirrors device_pin_chunk: compression is decided per chunk, so a single pin
/// may interleave the two. A host-tier pin populates `host` instead, for every
/// batch or none — a pinned entry lives in one tier.
struct batch_source {
  sirius::compressed_device_representation const* compressed = nullptr;
  std::size_t column_index                                   = 0;  ///< within the compressed chunk
  cudf::column_view uncompressed{};  ///< valid iff compressed == nullptr and host == nullptr
  std::shared_ptr<host_blocked_buffers const> host;  ///< set iff the pin is host-tier
  std::int64_t num_rows = 0;

  [[nodiscard]] bool is_compressed() const noexcept { return compressed != nullptr; }
  [[nodiscard]] bool is_host() const noexcept { return host != nullptr; }
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
/// A HOST-tier origin is gathered where it lies, out of the pinned blocks
/// described by batch_source::host, with no staging pass: fixed-width columns
/// only, because a variable-width one has no element width to translate and its
/// offsets would have to be rebuilt against buffers that are not contiguous.
/// Validity is carried through, one mask word at a time, in the same
/// coordinates.
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
