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

#include "compression/compressed_representation.hpp"
#include "late_mat/multi_source_gather.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <api/simpatico_codegen.hpp>
#include <codegen/selection/chunk_row_set.hpp>
#include <codegen/selection/selection.hpp>

#include <algorithm>
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
  // A host batch has no cudf::column_view for the canonical path to gather from
  // — its buffers are not contiguous — so the blocked gather is its only route,
  // whatever the batch count.
  if (column.batches.front().is_host()) { return cudf::is_fixed_width(column.dtype); }
  if (column.batches.size() == 1) { return true; }
  return cudf::is_fixed_width(column.dtype);
}

/// Copy a host array of trivially-copyable descriptors into a stream-ordered
/// device buffer.
template <typename T>
rmm::device_buffer to_device(std::vector<T> const& values,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  rmm::device_buffer buf(values.size() * sizeof(T), stream, mr);
  cudaMemcpyAsync(
    buf.data(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream.value());
  return buf;
}

/// The host-tier path: gather by global id straight out of the pinned blocks.
///
/// Each batch contributes its blocks to one flattened array plus a byte offset
/// into their logical concatenation, which is all the kernel needs to turn a row
/// into an address. Nothing is staged to the device but the output itself.
std::unique_ptr<cudf::column> materialize_host_raw(pinned_column_view const& column,
                                                   prepared_selection const& selection,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto const& ids        = selection.ids();
  auto const elem_size   = cudf::size_of(column.dtype);
  std::size_t block_size = 0;

  std::vector<void const*> blocks;
  std::vector<std::int64_t> block_base;
  std::vector<std::int64_t> data_off;
  std::vector<std::int64_t> mask_off;
  std::vector<std::int64_t> row_start;
  block_base.reserve(column.batches.size());
  data_off.reserve(column.batches.size());
  mask_off.reserve(column.batches.size());
  row_start.reserve(column.batches.size());
  bool any_nullable = false;

  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    auto const* host = column.batches[b].host.get();
    if (host == nullptr) {
      throw std::runtime_error("late_mat::materialize: a host-tier column mixes storage forms");
    }
    if (block_size == 0) {
      block_size = host->block_size;
    } else if (block_size != host->block_size) {
      throw std::runtime_error("late_mat::materialize: host chunks disagree on their block size");
    }
    block_base.push_back(static_cast<std::int64_t>(blocks.size()));
    blocks.insert(blocks.end(), host->blocks.begin(), host->blocks.end());
    data_off.push_back(static_cast<std::int64_t>(host->data_offset));
    mask_off.push_back(host->has_null_mask ? static_cast<std::int64_t>(host->null_mask_offset)
                                           : std::int64_t{-1});
    row_start.push_back(selection.layout().batch_row_start[b]);
    any_nullable = any_nullable || host->has_null_mask;
  }

  auto out = cudf::make_fixed_width_column(
    column.dtype,
    static_cast<cudf::size_type>(ids.count),
    any_nullable ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED,
    stream,
    mr);

  auto const blocks_dev     = to_device(blocks, stream, mr);
  auto const block_base_dev = to_device(block_base, stream, mr);
  auto const data_off_dev   = to_device(data_off, stream, mr);
  auto const mask_off_dev   = to_device(mask_off, stream, mr);
  auto const row_start_dev  = to_device(row_start, stream, mr);

  multi_source_gather_fixed_host(
    static_cast<void const* const*>(blocks_dev.data()),
    static_cast<std::int64_t const*>(block_base_dev.data()),
    static_cast<std::int64_t const*>(data_off_dev.data()),
    static_cast<std::int64_t const*>(mask_off_dev.data()),
    static_cast<std::int64_t const*>(row_start_dev.data()),
    static_cast<int>(column.batches.size()),
    block_size,
    elem_size,
    ids.ids,
    ids.count,
    out->mutable_view().head<void>(),
    any_nullable ? reinterpret_cast<std::uint32_t*>(out->mutable_view().null_mask()) : nullptr,
    stream);
  // The descriptor buffers are freed as this returns; they are stream-ordered on
  // the same stream the gather was enqueued on, so the deallocation is already
  // ordered after it.
  if (any_nullable) {
    out->set_null_count(cudf::null_count(
      out->view().null_mask(), 0, static_cast<cudf::size_type>(ids.count), stream));
  }
  return out;
}

/// The raw path: gather by global id, no canonical form, no restoring pass.
std::unique_ptr<cudf::column> materialize_raw(pinned_column_view const& column,
                                              prepared_selection const& selection,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto const& ids = selection.ids();

  if (column.batches.front().is_host()) {
    return materialize_host_raw(column, selection, stream, mr);
  }

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
  bool const any_nullable =
    std::any_of(column.batches.begin(), column.batches.end(), [](batch_source const& b) {
      return b.uncompressed.nullable();
    });
  auto out = cudf::make_fixed_width_column(
    column.dtype,
    static_cast<cudf::size_type>(ids.count),
    any_nullable ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED,
    stream,
    mr);

  std::vector<void const*> host_bases;
  std::vector<std::int64_t> host_starts;
  std::vector<cudf::bitmask_type const*> host_masks;
  host_bases.reserve(column.batches.size());
  host_starts.reserve(column.batches.size());
  if (any_nullable) { host_masks.reserve(column.batches.size()); }
  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    host_bases.push_back(column.batches[b].uncompressed.head<void>());
    host_starts.push_back(selection.layout().batch_row_start[b]);
    if (any_nullable) { host_masks.push_back(column.batches[b].uncompressed.null_mask()); }
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

  rmm::device_buffer masks;
  if (any_nullable) {
    masks = rmm::device_buffer(host_masks.size() * sizeof(cudf::bitmask_type const*), stream, mr);
    cudaMemcpyAsync(masks.data(),
                    host_masks.data(),
                    host_masks.size() * sizeof(cudf::bitmask_type const*),
                    cudaMemcpyHostToDevice,
                    stream.value());
  }

  multi_source_gather_fixed(
    static_cast<void const* const*>(bases.data()),
    static_cast<std::int64_t const*>(starts.data()),
    static_cast<int>(column.batches.size()),
    elem_size,
    ids.ids,
    ids.count,
    out->mutable_view().head<void>(),
    any_nullable ? static_cast<std::uint32_t const* const*>(masks.data()) : nullptr,
    any_nullable ? reinterpret_cast<std::uint32_t*>(out->mutable_view().null_mask()) : nullptr,
    stream);
  // The base, start and mask arrays are freed as this returns; they are stream-
  // ordered on the same stream the gather was enqueued on, so the deallocation
  // is already ordered after it.
  if (any_nullable) {
    out->set_null_count(cudf::null_count(
      out->view().null_mask(), 0, static_cast<cudf::size_type>(ids.count), stream));
  }
  return out;
}

/// Below this the mask route still beats decoding everything and gathering
/// once; above it, it does not. Measured on the original campaign's crossover
/// microbench.
constexpr double kMaskRouteMaxDensity = 0.35;

std::unique_ptr<cudf::column> require_non_null(std::unique_ptr<cudf::column> col)
{
  // None of these routes gathers a validity mask alongside the values, so a
  // nullable column would come back with someone else's nulls.
  if (col && col->null_count() != 0) {
    throw std::runtime_error("late_mat::materialize: nullable origin columns are not supported");
  }
  return col;
}

/// One compressed batch, by the cheapest route its plan can take.
///
/// A route returning nothing means this plan cannot take it, and the next one
/// is tried; the full decode ends every cascade. That is what lets the deferral
/// decision stay ignorant of decode internals — any column may be deferred, and
/// the worst case is the decode that would have happened anyway.
std::unique_ptr<cudf::column> materialize_compressed(batch_source const& source,
                                                     batch_selection const& selection,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto const& table = source.compressed->table();
  auto const idx    = source.column_index;

  // Whole batch live: no selection to express, so this is an ordinary decode.
  if (selection.dense) {
    std::string err;
    auto full = simpatico::decompress_column_full(table, idx, stream, mr, &err);
    if (!full) { throw std::runtime_error("late_mat::materialize: full decode failed: " + err); }
    return require_non_null(std::move(full));
  }

  auto const rows = selection.rows.view();

  // (a) Sparse walk: one block per touched chunk. Tried at every density,
  // because the mask route never won a density-based choice when it was
  // measured — it is the capability fallback below, not the faster one.
  {
    std::string err;
    auto sparse = simpatico::decompress_column_rows(table, idx, rows, stream, mr, &err);
    if (sparse) { return require_non_null(std::move(sparse)); }
  }

  // (b) Mask route: the shipped kernels, for the shapes with no random access
  // (dictionary, str_split, render rejections). The mask is derived from the
  // row set rather than rebuilt from the ids.
  if (selection.density < kMaskRouteMaxDensity) {
    auto const num_rows = source.num_rows;
    auto const chunks   = sirius::codegen::selection_mask::ChunksFor(num_rows);
    rmm::device_buffer mask_words(
      static_cast<std::size_t>(sirius::codegen::selection_mask::WordsFor(num_rows)) *
        sizeof(std::uint32_t),
      stream,
      mr);
    rmm::device_buffer chunk_offsets(
      (static_cast<std::size_t>(chunks) + 1) * sizeof(std::uint32_t), stream, mr);
    sirius::codegen::row_set_to_mask(rows,
                                     static_cast<std::uint32_t*>(mask_words.data()),
                                     static_cast<std::uint32_t*>(chunk_offsets.data()),
                                     stream,
                                     mr);
    sirius::codegen::selection_mask mask{static_cast<std::uint32_t*>(mask_words.data()),
                                         num_rows,
                                         rows.num_survivors,
                                         static_cast<std::uint32_t*>(chunk_offsets.data())};
    std::string err;
    auto compacted = simpatico::decompress_column_compacted(table, idx, mask, stream, mr, &err);
    if (compacted) { return require_non_null(std::move(compacted)); }
  }

  // (c) The route that always works: decode everything, gather once.
  std::string err;
  auto full = simpatico::decompress_column_full(table, idx, stream, mr, &err);
  if (!full) { throw std::runtime_error("late_mat::materialize: full decode failed: " + err); }
  auto checked = require_non_null(std::move(full));
  return gather_one(
    checked->view(), int32_map(selection.local_indices, selection.survivors), stream, mr);
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
  if (source.is_compressed()) { return materialize_compressed(source, selection, stream, mr); }
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
  nvtx3::scoped_range nvtx_range{"sirius::late_mat::materialize"};
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

  // A host-tier column reaches exactly one route, and every batch of it has to
  // take that route: the canonical path below reads batch_source::uncompressed,
  // which a host batch does not have. The resolver refuses anything else, so a
  // column arriving here half host or variable-width is a disagreement between
  // the two, not a shape to handle.
  bool const any_host = std::any_of(column.batches.begin(),
                                    column.batches.end(),
                                    [](batch_source const& b) { return b.is_host(); });
  if (any_host) {
    bool const all_host = std::all_of(column.batches.begin(),
                                      column.batches.end(),
                                      [](batch_source const& b) { return b.is_host(); });
    if (!all_host || !cudf::is_fixed_width(column.dtype)) {
      throw std::runtime_error(
        "late_mat::materialize: a host-tier origin must be fixed width in every batch");
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
