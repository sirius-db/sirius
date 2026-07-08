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

#include "op/scan/pinned_chunk_source.hpp"

#include "data/data_batch_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// cached_scan_info
//===----------------------------------------------------------------------===//

cached_scan_info::cached_scan_info(std::shared_ptr<cucascade::data_batch> batch,
                                   std::size_t chunk_index)
  : _batch(std::move(batch)), _chunk_index(chunk_index)
{
}

cached_scan_info::~cached_scan_info() = default;

std::shared_ptr<cucascade::data_batch> cached_scan_info::take_batch() noexcept
{
  return std::move(_batch);
}

//===----------------------------------------------------------------------===//
// cached_batch_coalescer
//===----------------------------------------------------------------------===//

std::vector<std::unique_ptr<scan_info>> cached_batch_coalescer::push(
  std::unique_ptr<scan_info> split)
{
  std::vector<std::unique_ptr<scan_info>> out;
  if (dynamic_cast<cached_scan_info*>(split.get()) != nullptr) { out.push_back(std::move(split)); }
  // Foreign split types are dropped, mirroring the disk-format coalescers.
  return out;
}

std::vector<std::unique_ptr<scan_info>> cached_batch_coalescer::flush()
{
  // No template-split fallback: a zero-chunk pin closes with zero splits.
  return {};
}

//===----------------------------------------------------------------------===//
// pinned_chunk_source
//===----------------------------------------------------------------------===//

pinned_chunk_source::pinned_chunk_source(std::vector<gpu_chunk> chunks,
                                         telemetry::batch_telemetry_info telemetry_info)
  : _gpu_chunks(std::move(chunks)), _telemetry_info(telemetry_info), _n_chunks(_gpu_chunks.size())
{
  for (auto const& chunk : _gpu_chunks) {
    if (chunk.memory_space == nullptr) {
      throw std::runtime_error("pinned_chunk_source: GPU chunk without a memory_space");
    }
  }
}

pinned_chunk_source::pinned_chunk_source(std::vector<host_chunk> chunks,
                                         std::vector<std::size_t> column_indices,
                                         telemetry::batch_telemetry_info telemetry_info)
  : _host_chunks(std::move(chunks)),
    _host_column_indices(std::move(column_indices)),
    _telemetry_info(telemetry_info),
    _n_chunks(_host_chunks.size())
{
  for (auto const& chunk : _host_chunks) {
    if (!chunk.data) { throw std::runtime_error("pinned_chunk_source: null pinned host chunk"); }
  }
}

std::function<std::unique_ptr<scan_info>()> pinned_chunk_source::next_work_item()
{
  auto const chunk = _next_chunk.fetch_add(1, std::memory_order_relaxed);
  if (chunk >= _n_chunks) { return nullptr; }  // lost the race to the last chunk
  return [this, chunk]() -> std::unique_ptr<scan_info> {
    return std::make_unique<cached_scan_info>(make_batch(chunk), chunk);
  };
}

std::shared_ptr<cucascade::data_batch> pinned_chunk_source::make_batch(std::size_t index) const
{
  auto const batch_id = ::sirius::get_next_batch_id();
  if (!_gpu_chunks.empty()) {
    auto const& chunk                                  = _gpu_chunks.at(index);
    std::vector<std::shared_ptr<cudf::column>> columns = chunk.columns;
    std::vector<cudf::column_view> column_views;
    column_views.reserve(columns.size());
    std::size_t alloc_size = 0;
    for (auto const& column : columns) {
      column_views.emplace_back(column->view());
      alloc_size += column->alloc_size();
    }
    cudf::table_view view(column_views);
    auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
      view, std::move(columns), alloc_size, *chunk.memory_space, rmm::cuda_stream_view{});
    return cucascade::data_batch::make(
      batch_id,
      std::move(gpu_repr),
      telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));
  }
  auto const& chunk = _host_chunks.at(index);
  auto data_rep     = chunk.data->slice(_host_column_indices);
  return cucascade::data_batch::make(
    batch_id,
    std::move(data_rep),
    telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));
}

}  // namespace sirius::op::scan
