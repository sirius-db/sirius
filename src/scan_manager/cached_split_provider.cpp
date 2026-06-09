/*
 * Copyright 2025, Sirius Contributors.
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

#include "scan_manager/cached_split_provider.hpp"

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda_runtime.h>

#include <data/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <data/gpu_table_representation.hpp>
#include <cucascade/data/representation_converter.hpp>

#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::scan_manager {

cached_split_provider::cached_split_provider(
  std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
  std::shared_ptr<duckdb::Expression> filter_expression,
  std::shared_ptr<op::scan::scan_plan const> plan)
  : _chunk_memory_spaces(std::move(chunk_memory_spaces)),
    _filter_expression(std::move(filter_expression)),
    _plan(std::move(plan))
{
  // Transpose [column_idx][chunk_idx] -> [chunk_idx][column_idx]. The
  // per-chunk-column layout matches how produce_split() consumes it (one
  // table_view's worth of columns per chunk), and lets us hold every chunk in a
  // uniform variant alongside the host-mode shape.
  std::size_t const num_batches =
    columns_per_request.empty() ? 0 : columns_per_request.front().size();
  for (auto const& col_chunks : columns_per_request) {
    if (col_chunks.size() != num_batches) {
      throw std::runtime_error(
        "[cached_split_provider] mismatched chunk count across requested columns");
    }
  }

  _batches.reserve(num_batches);
  for (std::size_t b = 0; b < num_batches; ++b) {
    gpu_chunk chunk_cols;
    chunk_cols.reserve(columns_per_request.size());
    for (auto& col_chunks : columns_per_request) {
      chunk_cols.push_back(std::move(col_chunks[b]));
    }
    _batches.emplace_back(std::move(chunk_cols));
  }
}

cached_split_provider::cached_split_provider(
  std::vector<std::shared_ptr<sirius::host_data_representation>> host_chunks,
  std::vector<std::size_t> column_indices,
  cucascade::memory::memory_space& memory_space,
  std::unordered_map<int, cucascade::memory::memory_space*> gpu_memory_spaces,
  std::shared_ptr<duckdb::Expression> filter_expression,
  std::shared_ptr<op::scan::scan_plan const> plan)
  : _column_indices(std::move(column_indices)),
    _memory_space(&memory_space),
    _gpu_memory_spaces(std::move(gpu_memory_spaces)),
    _filter_expression(std::move(filter_expression)),
    _plan(std::move(plan))
{
  if (_gpu_memory_spaces.empty()) {
    throw std::runtime_error(
      "[cached_split_provider] HOST-tier constructor requires non-empty gpu_memory_spaces "
      "so produce_split can convert host chunks to the executing GPU's space");
  }
  _batches.reserve(host_chunks.size());
  for (auto& chunk : host_chunks) {
    if (!chunk) {
      throw std::runtime_error(
        "[cached_split_provider] null host_data_representation in pinned chunks");
    }
    _batches.emplace_back(std::move(chunk));
  }
}

bool cached_split_provider::has_more_splits() const
{
  return _next_batch_idx.load(std::memory_order_relaxed) < _batches.size();
}

std::function<std::vector<std::unique_ptr<op::operator_data>>()>
cached_split_provider::next_split_provider()
{
  // Atomic claim happens here so the produce_split() work captured below can
  // run in parallel on a worker pool — distinct callables operate on distinct
  // chunk indices.
  auto const batch_idx = _next_batch_idx.fetch_add(1, std::memory_order_relaxed);
  if (batch_idx >= _batches.size()) { return nullptr; }
  return [this, batch_idx]() { return produce_split(batch_idx); };
}

std::vector<std::unique_ptr<op::operator_data>> cached_split_provider::produce_split(
  std::size_t batch_idx)
{
  auto batch = std::visit(
    [this, batch_idx](auto& chunk) -> std::shared_ptr<cucascade::data_batch> {
      using T = std::decay_t<decltype(chunk)>;
      if constexpr (std::is_same_v<T, gpu_chunk>) {
        std::vector<cudf::column_view> col_views;
        col_views.reserve(chunk.size());
        std::size_t alloc_size = 0;
        for (auto const& col_ptr : chunk) {
          col_views.emplace_back(col_ptr->view());
          alloc_size += col_ptr->alloc_size();
        }
        cudf::table_view view(col_views);
        auto* chunk_space =
          !_chunk_memory_spaces.empty() ? _chunk_memory_spaces.at(batch_idx) : _memory_space;
        auto gpu_repr = std::make_unique<sirius::gpu_table_representation>(
          view, std::move(chunk), alloc_size, *chunk_space, rmm::cuda_stream_view{});
        return std::make_shared<cucascade::data_batch>(::sirius::get_next_batch_id(),
                                                       std::move(gpu_repr));
      } else if constexpr (std::is_same_v<T, host_chunk>) {
        // HOST tier: each chunk holds the full pinned table; slice down to the
        // columns the scan actually wants. Conversion back to GPU happens later
        // in scan_cached_operator_data::prepare_for_processing.
        auto sliced = chunk->slice(std::span<const std::size_t>(_column_indices));

        return std::make_shared<cucascade::data_batch>(::sirius::get_next_batch_id(),
                                                       std::move(sliced));
      }
    },
    _batches[batch_idx]);

  std::vector<std::unique_ptr<op::operator_data>> out;
  out.push_back(std::make_unique<op::scan::scan_cached_operator_data>(
    std::move(batch), _filter_expression, _plan));
  return out;
}

}  // namespace sirius::scan_manager
