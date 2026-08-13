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

#include "late_mat/materialize.hpp"

#include "late_mat/multi_source_gather.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::late_mat {

namespace {

std::unique_ptr<cudf::column> gather_one(cudf::column_view const& source,
                                         cudf::column_view const& map,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  // DONT_CHECK: ids are pin-order positions the caller is responsible for, the
  // same contract cudf::gather itself takes. Checking would cost a pass to
  // re-establish what the addressing already guarantees.
  auto gathered = cudf::gather(
    cudf::table_view{{source}}, map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return std::move(gathered->release().front());
}

cudf::column_view int32_map(rmm::device_buffer const& buf, std::int64_t count)
{
  return cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(count),
                           buf.data(),
                           nullptr,
                           0};
}

/// Can this column be produced straight from the raw ids, with no sort?
///
/// Two things have to hold. Nothing may be compressed, since a compressed row
/// is decoded rather than copied and producing it once per reference is the
/// waste deduplication exists to avoid. And a multi-batch column has to be
/// fixed width, because the one-pass gather copies a fixed element and a
/// variable-width column would need its offsets rebuilt.
bool can_gather_raw(pinned_column_view const& column)
{
  for (auto const& b : column.batches) {
    if (b.is_compressed()) { return false; }
  }
  if (column.batches.size() == 1) { return true; }
  return cudf::is_fixed_width(column.dtype);
}

/// The raw path: gather by global id, no canonical form, no restoring pass.
std::unique_ptr<cudf::column> materialize_raw(pinned_column_view const& column,
                                              prepared_selection const& selection,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto const& ids = selection.ids();

  // One batch starts at global row 0, so the ids ARE the batch-local map and
  // cudf gathers straight off the borrowed list. Any dtype, since cudf does the
  // element handling.
  if (column.batches.size() == 1) {
    auto const map = cudf::column_view{cudf::data_type{cudf::type_id::UINT64},
                                       static_cast<cudf::size_type>(ids.count),
                                       ids.ids,
                                       nullptr,
                                       0};
    return gather_one(column.batches.front().uncompressed, map, stream, mr);
  }

  auto const elem_size = cudf::size_of(column.dtype);
  auto out             = cudf::make_fixed_width_column(column.dtype,
                                           static_cast<cudf::size_type>(ids.count),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);

  std::vector<void const*> host_bases;
  std::vector<std::int64_t> host_starts;
  host_bases.reserve(column.batches.size());
  host_starts.reserve(column.batches.size());
  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    host_bases.push_back(column.batches[b].uncompressed.head<void>());
    host_starts.push_back(selection.layout().batch_row_start[b]);
  }

  rmm::device_buffer bases(host_bases.size() * sizeof(void const*), stream, mr);
  rmm::device_buffer starts(host_starts.size() * sizeof(std::int64_t), stream, mr);
  cudaMemcpyAsync(bases.data(),
                  host_bases.data(),
                  host_bases.size() * sizeof(void const*),
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaMemcpyAsync(starts.data(),
                  host_starts.data(),
                  host_starts.size() * sizeof(std::int64_t),
                  cudaMemcpyHostToDevice,
                  stream.value());

  multi_source_gather_fixed(static_cast<void const* const*>(bases.data()),
                            static_cast<std::int64_t const*>(starts.data()),
                            static_cast<int>(column.batches.size()),
                            elem_size,
                            ids.ids,
                            ids.count,
                            out->mutable_view().head<void>(),
                            stream);
  // The base and start arrays are freed as this returns; they are stream-
  // ordered on the same stream the gather was enqueued on, so the deallocation
  // is already ordered after it.
  return out;
}

/// A batch's surviving rows, as a column of its own.
///
/// Dense batches are copied rather than gathered: the gather map would be the
/// identity, and building one to apply it is more work than the copy.
std::unique_ptr<cudf::column> materialize_batch(batch_source const& source,
                                                batch_selection const& selection,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  if (source.is_compressed()) {
    throw std::runtime_error(
      "late_mat::materialize: a compressed origin needs a decompression entry point that takes a "
      "selection; refusing rather than decoding the batch full width");
  }
  if (selection.dense) {
    auto copied =
      std::make_unique<cudf::table>(cudf::table_view{{source.uncompressed}}, stream, mr);
    return std::move(copied->release().front());
  }
  return gather_one(
    source.uncompressed, int32_map(selection.local_indices, selection.survivors), stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> materialize(pinned_column_view const& column,
                                          prepared_selection const& selection,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const& layout = selection.layout();
  if (column.batches.size() != layout.num_batches()) {
    throw std::runtime_error(
      "late_mat::materialize: the column's batches do not match the "
      "layout the selection was prepared against");
  }
  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    if (column.batches[b].num_rows != layout.batch_rows[b]) {
      throw std::runtime_error("late_mat::materialize: batch " + std::to_string(b) +
                               " has a different row count than the prepared layout");
    }
  }

  if (selection.original_count() == 0) { return cudf::make_empty_column(column.dtype); }

  // Nothing here needs the rows sorted or deduplicated, so do not build a
  // canonical form that a later compressed column may never ask for either.
  if (can_gather_raw(column)) { return materialize_raw(column, selection, stream, mr); }

  auto const& canonical = selection.canonical(stream, mr);
  if (canonical.total_survivors == 0) { return cudf::make_empty_column(column.dtype); }

  // In pin order, so the assembled column is in pinned-table order.
  std::vector<std::unique_ptr<cudf::column>> pieces;
  pieces.reserve(column.batches.size());
  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    if (canonical.batches[b].survivors == 0) { continue; }
    pieces.push_back(materialize_batch(column.batches[b], canonical.batches[b], stream, mr));
  }

  std::unique_ptr<cudf::column> assembled;
  if (pieces.size() == 1) {
    assembled = std::move(pieces.front());
  } else {
    std::vector<cudf::column_view> views;
    views.reserve(pieces.size());
    for (auto const& p : pieces) {
      views.push_back(p->view());
    }
    assembled = cudf::concatenate(views, stream, mr);
  }

  if (!canonical.needs_restore()) { return assembled; }

  // Back into the caller's order, repeats included. This gather is over the
  // materialized column — narrow, and already reduced to the surviving rows —
  // which is why deduplicating was worth a sort for the consumers that need it.
  return gather_one(
    assembled->view(), int32_map(canonical.restore_rank, selection.original_count()), stream, mr);
}

}  // namespace sirius::late_mat
