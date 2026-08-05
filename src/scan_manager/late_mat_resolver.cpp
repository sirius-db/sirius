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

#include <api/simpatico_codegen.hpp>

#include <cudf/column/column.hpp>

namespace sirius::scan_manager {

std::optional<late_mat::pinned_table_layout> resolve_pinned_layout(
  late_mat::column_origin const& origin)
{
  auto const* entry = origin.resolve();
  if (entry == nullptr) { return std::nullopt; }
  // v1: device-resident storage only. HOST-tier entries would need a staging
  // hop the materializer does not model yet.
  if (entry->tier != cucascade::memory::Tier::GPU) { return std::nullopt; }
  return late_mat::pinned_table_layout::from_batch_rows(pinned_chunk_row_counts(*entry),
                                                        origin.generation);
}

std::optional<late_mat::pinned_column_view> resolve_pinned_column(
  late_mat::column_origin const& origin)
{
  auto const* entry = origin.resolve();
  if (entry == nullptr) { return std::nullopt; }
  if (entry->tier != cucascade::memory::Tier::GPU) { return std::nullopt; }
  auto const col = static_cast<std::size_t>(origin.column_pos);
  if (col >= entry->cache_info.column_ids.size()) { return std::nullopt; }

  late_mat::pinned_column_view view;
  view.pin_generation = origin.generation;

  if (!entry->device_chunks.empty()) {
    // Compression-enabled GPU pin: per chunk either a compressed table (all
    // pinned columns, cache_info order) or plain device columns.
    view.batches.reserve(entry->device_chunks.size());
    for (auto const& chunk : entry->device_chunks) {
      late_mat::batch_source src;
      if (chunk.compressed) {
        auto const& ct = chunk.compressed->table();
        if (col >= ct.num_columns()) { return std::nullopt; }
        src.compressed   = &ct;
        src.column_index = col;
        src.num_rows     = ct.columns[col].num_rows;
        if (view.dtype.id() == cudf::type_id::EMPTY) { view.dtype = ct.columns[col].dtype; }
      } else {
        if (col >= chunk.columns.size() || !chunk.columns[col]) { return std::nullopt; }
        auto cv = chunk.columns[col]->view();
        if (cv.null_count() > 0) { return std::nullopt; }  // v1: non-null only
        src.uncompressed = cv;
        src.num_rows     = cv.size();
        if (view.dtype.id() == cudf::type_id::EMPTY) { view.dtype = cv.type(); }
      }
      view.batches.push_back(std::move(src));
    }
    return view;
  }

  // Plain (non-compression) GPU pin: one chunk vector per column name.
  auto const& names = entry->cache_info.column_names();
  auto const it     = entry->data_batches_by_column.find(names[col]);
  if (it == entry->data_batches_by_column.end()) { return std::nullopt; }
  view.batches.reserve(it->second.size());
  for (auto const& chunk : it->second) {
    if (!chunk) { return std::nullopt; }
    auto cv = chunk->view();
    if (cv.null_count() > 0) { return std::nullopt; }  // v1: non-null only
    late_mat::batch_source src;
    src.uncompressed = cv;
    src.num_rows     = cv.size();
    if (view.dtype.id() == cudf::type_id::EMPTY) { view.dtype = cv.type(); }
    view.batches.push_back(std::move(src));
  }
  return view;
}

}  // namespace sirius::scan_manager
