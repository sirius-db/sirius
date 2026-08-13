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

#include "late_mat/prepared_selection.hpp"

#include <codegen/selection/row_id_space.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::late_mat {

pinned_table_layout pinned_table_layout::from_batch_rows(std::vector<std::int64_t> rows,
                                                         pin_generation_t generation)
{
  pinned_table_layout layout;
  layout.batch_rows     = std::move(rows);
  layout.pin_generation = generation;
  layout.batch_row_start.reserve(layout.batch_rows.size() + 1);
  layout.batch_row_start.push_back(0);
  for (auto const r : layout.batch_rows) {
    if (r < 0) { throw std::runtime_error("prepared_selection: a batch with negative rows"); }
    layout.batch_row_start.push_back(layout.batch_row_start.back() + r);
  }
  return layout;
}

prepared_selection prepare_selection(pinned_table_layout const& layout,
                                     row_id_list const& ids,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  if (layout.batch_rows.size() + 1 != layout.batch_row_start.size()) {
    throw std::runtime_error("prepared_selection: layout row starts do not match its batches");
  }
  if (layout.num_batches() == 0) {
    throw std::runtime_error("prepared_selection: a layout with no batches");
  }

  prepared_selection prepared;
  prepared.layout         = layout;
  prepared.original_count = ids.count;
  prepared.batches.resize(layout.num_batches());
  prepared.out_base.assign(layout.num_batches() + 1, 0);
  if (ids.count == 0) { return prepared; }
  if (ids.count < 0 || ids.ids == nullptr) {
    throw std::runtime_error("prepared_selection: prepare from an unbound id list");
  }

  // Order and dedup, unless the caller can promise both. The restore ranks are
  // what let a deduplicated, table-ordered output answer a caller that asked
  // in its own order with repeats.
  std::uint64_t const* sorted   = ids.ids;
  std::int32_t const* count_dev = nullptr;
  codegen::sorted_unique_ids canonical;
  if (!ids.sorted_unique) {
    canonical = codegen::sort_unique_global_ids(ids.ids, ids.count, stream, mr);
    sorted    = static_cast<std::uint64_t const*>(canonical.ids.data());
    count_dev = static_cast<std::int32_t const*>(canonical.count_dev.data());
  }

  // One sync, for the batch boundaries — slicing and per-batch sizing are host
  // decisions. The deduplicated count comes back in the same copy.
  std::int64_t live = 0;
  auto const starts = codegen::split_sorted_ids_by_batch(
    sorted, ids.count, count_dev, layout.batch_row_start, &live, stream, mr);

  // Ids outside the pinned table would silently vanish here — the split would
  // simply not assign them to any batch — so the disagreement is caught rather
  // than materialized as missing rows.
  if (starts.front() != 0 || starts.back() != live) {
    throw std::runtime_error("prepared_selection: row ids fall outside the pinned table");
  }

  prepared.total_survivors = live;
  if (!ids.sorted_unique) { prepared.restore_rank = std::move(canonical.restore_rank); }

  for (std::size_t b = 0; b < layout.num_batches(); ++b) {
    auto const count         = starts[b + 1] - starts[b];
    auto& out                = prepared.batches[b];
    out.survivors            = count;
    prepared.out_base[b + 1] = prepared.out_base[b] + count;
    if (count == 0) { continue; }

    auto const batch_rows = layout.batch_rows[b];
    out.density           = static_cast<double>(count) / static_cast<double>(batch_rows);
    if (count == batch_rows) {
      // Every row lives: no selection to express, and no host sync to pay for
      // expressing it.
      out.dense = true;
      continue;
    }

    out.local_indices =
      rmm::device_buffer(static_cast<std::size_t>(count) * sizeof(std::int32_t), stream, mr);
    codegen::global_slice_to_local(sorted + starts[b],
                                   count,
                                   layout.batch_row_start[b],
                                   static_cast<std::int32_t*>(out.local_indices.data()),
                                   stream);
    out.rows = codegen::build_chunk_row_set(
      static_cast<std::int32_t const*>(out.local_indices.data()), count, batch_rows, stream, mr);
  }

  return prepared;
}

}  // namespace sirius::late_mat
