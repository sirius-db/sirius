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

#include "scan_manager/late_mat_resolver.hpp"

#include "compression/compressed_representation.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <cudf/column/column.hpp>

#include <api/simpatico_codegen.hpp>

namespace sirius::scan_manager {

namespace {

/// A GPU-tier entry stores its chunks one of two ways, and only one is
/// populated: device_pin_chunks (compression-enabled, possibly interleaved) or
/// a per-column-name map (the plain pin). Reading the wrong one silently gives
/// an entry with no chunks, so the choice is made once, here.
bool uses_device_chunks(pinned_entry const& entry) { return !entry.device_chunks.empty(); }

/// Rows in one device_pin_chunk, whichever form it took.
std::int64_t chunk_rows(sirius::device_pin_chunk const& chunk)
{
  if (chunk.compressed) { return chunk.compressed->num_rows(); }
  if (chunk.columns.empty() || !chunk.columns.front()) { return 0; }
  return chunk.columns.front()->size();
}

}  // namespace

std::optional<late_mat::pinned_table_layout> resolve_pinned_layout(
  late_mat::column_origin const& origin)
{
  auto const* entry = origin.resolve();
  if (entry == nullptr) { return std::nullopt; }
  if (entry->tier != cucascade::memory::Tier::GPU) { return std::nullopt; }

  std::vector<std::int64_t> rows;
  if (uses_device_chunks(*entry)) {
    rows.reserve(entry->device_chunks.size());
    for (auto const& chunk : entry->device_chunks) {
      rows.push_back(chunk_rows(chunk));
    }
  } else {
    // The plain pin holds one chunk vector per column NAME; every column has
    // the same chunking, so any present column gives the layout.
    auto const& names = entry->cache_info.column_names();
    if (names.empty()) { return std::nullopt; }
    auto const it = entry->data_batches_by_column.find(names.front());
    if (it == entry->data_batches_by_column.end()) { return std::nullopt; }
    rows.reserve(it->second.size());
    for (auto const& col : it->second) {
      if (!col) { return std::nullopt; }
      rows.push_back(col->size());
    }
  }
  if (rows.empty()) { return std::nullopt; }
  return late_mat::pinned_table_layout::from_batch_rows(std::move(rows), origin.generation);
}

std::optional<late_mat::pinned_column_view> resolve_pinned_column(
  late_mat::column_origin const& origin)
{
  auto const* entry = origin.resolve();
  if (entry == nullptr) { return std::nullopt; }
  if (entry->tier != cucascade::memory::Tier::GPU) { return std::nullopt; }

  auto const& names = entry->cache_info.column_names();
  if (origin.column_pos >= names.size()) { return std::nullopt; }

  late_mat::pinned_column_view view;
  view.pin_generation = origin.generation;

  if (uses_device_chunks(*entry)) {
    // Every gather shape propagates validity: the uncompressed ones (single-batch, multi-batch
    // fixed-width, multi-batch variable-width) natively, and a compressed one through the decode
    // routes (materialize.cpp's attach_selected_validity). A chunk with no blob is refused below,
    // since its validity sidecar is unreadable along with everything else about it.
    for (auto const& chunk : entry->device_chunks) {
      late_mat::batch_source source;
      source.num_rows = chunk_rows(chunk);
      if (chunk.compressed) {
        if (!chunk.compressed->has_table()) { return std::nullopt; }
        source.compressed   = chunk.compressed.get();
        source.column_index = origin.column_pos;
        // The compressed table carries its own carrier dtype per column, and it
        // is the one the decode re-tags to. That carrier is chosen PER CHUNK —
        // narrow_pin_chunk() computes each chunk's exact range independently, so
        // one logical BIGINT can be stored INT8 in one chunk and INT16 in the
        // next. The view carries a single dtype, which the gather reads every
        // batch at, so a pin whose chunks disagree has no single answer here and
        // is refused rather than read at the first chunk's width.
        auto const& table = chunk.compressed->table();
        if (origin.column_pos >= table.columns.size()) { return std::nullopt; }
        auto const carrier = table.columns[origin.column_pos].dtype;
        if (view.dtype.id() == cudf::type_id::EMPTY) {
          view.dtype = carrier;
        } else if (view.dtype != carrier) {
          return std::nullopt;
        }
      } else {
        if (origin.column_pos >= chunk.columns.size() || !chunk.columns[origin.column_pos]) {
          return std::nullopt;
        }
        // Same per-chunk carrier caveat as the compressed branch above: with
        // compressed materialization on, narrow_pin_chunk() narrows every chunk
        // it materializes, including ones the sink then stores uncompressed.
        auto const& col     = *chunk.columns[origin.column_pos];
        source.uncompressed = col.view();
        if (view.dtype.id() == cudf::type_id::EMPTY) {
          view.dtype = col.type();
        } else if (view.dtype != col.type()) {
          return std::nullopt;
        }
      }
      view.batches.push_back(source);
    }
  } else {
    auto const it = entry->data_batches_by_column.find(names[origin.column_pos]);
    if (it == entry->data_batches_by_column.end()) { return std::nullopt; }
    for (auto const& col : it->second) {
      if (!col) { return std::nullopt; }
      late_mat::batch_source source;
      source.uncompressed = col->view();
      source.num_rows     = col->size();
      view.batches.push_back(source);
      if (view.dtype.id() == cudf::type_id::EMPTY) {
        view.dtype = col->type();
      } else if (view.dtype != col->type()) {
        return std::nullopt;
      }
    }
  }

  if (view.batches.empty()) { return std::nullopt; }
  return view;
}

}  // namespace sirius::scan_manager
