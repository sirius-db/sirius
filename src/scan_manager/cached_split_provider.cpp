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
#include "op/scan/parquet_scan_operator_data.hpp"
#include "op/sirius_physical_operator.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

cached_split_provider::cached_split_provider(
  std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
  std::shared_ptr<duckdb::Expression> filter_expression,
  std::shared_ptr<op::scan::scan_plan const> plan)
  : _columns_per_request(std::move(columns_per_request)),
    _chunk_memory_spaces(std::move(chunk_memory_spaces)),
    _filter_expression(std::move(filter_expression)),
    _plan(std::move(plan))
{
}

std::future<void> cached_split_provider::start(exec::thread_pool& /*pool*/,
                                               split_connector& connector)
{
  std::promise<void> promise;
  auto future = promise.get_future();

  std::size_t const num_batches =
    _columns_per_request.empty() ? 0 : _columns_per_request.front().size();

  // Sanity-check: every column must contribute the same number of chunks.
  for (auto const& col_chunks : _columns_per_request) {
    if (col_chunks.size() != num_batches) {
      throw std::runtime_error(
        "[cached_split_provider] mismatched chunk count across requested columns");
    }
  }

  // Phase 22 D-04: per-chunk memory_space vector must align with the chunk
  // count derived from columns_per_request. Reject any caller that passes a
  // misaligned vector loudly rather than silently dispatching cached batches
  // to the wrong GPU.
  if (_chunk_memory_spaces.size() != num_batches) {
    throw std::runtime_error(
      "[cached_split_provider] chunk_memory_spaces.size() (" +
      std::to_string(_chunk_memory_spaces.size()) + ") does not match num_batches (" +
      std::to_string(num_batches) + ")");
  }

  for (std::size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<cudf::column_view> col_views;
    col_views.reserve(_columns_per_request.size());
    // Owner keeps the shared_ptr<column> chunks alive for the lifetime of the
    // emitted data_batch, so the table_view's pointers stay valid even though
    // the gpu_table_representation does not own the underlying memory.
    std::vector<std::shared_ptr<cudf::column>> owner;
    owner.reserve(_columns_per_request.size());
    std::size_t alloc_size = 0;
    for (auto const& col_chunks : _columns_per_request) {
      auto const& col_ptr = col_chunks[batch_idx];
      col_views.emplace_back(col_ptr->view());
      alloc_size += col_ptr->alloc_size();
      owner.push_back(col_ptr);
    }

    cudf::table_view view(col_views);
    // Phase 18 / DB-04: cucascade #117 makes writer_stream REQUIRED on all
    // gpu_table_representation constructors (Phase 13-04 Path-2 stream-lineage
    // contract). The cached path wraps already-pinned data: the underlying
    // GPU memory was written long ago by whichever pipeline originally
    // populated the pinned cache, on a stream that no longer exists at this
    // call site. Passing a default-constructed cuda_stream_view records no
    // writer event — documented as the "legacy, no-stream" pattern in
    // cucascade/include/cucascade/data/gpu_data_representation.hpp:60-66:
    // "passing a default-constructed cuda_stream_view records no event
    // (legacy, only acceptable for paths whose data was never produced on
    // any stream)". The cached pinned data is exactly such a path — any
    // downstream cross-device reader that needs ordering must obtain it
    // via record_writer_event() at the actual writing site. This is NOT
    // the legacy default-stream wrapper (which would violate HYG-02);
    // cuda_stream_view{} is a null stream view.
    rmm::cuda_stream_view const no_writer_stream{};
    // Phase 22 D-04: per-chunk memory_space lookup. Replaces entry-level
    // _memory_space (now gone post-PIN-MGPU-01); each chunk carries the
    // memory_space its data lives on so SCHED-01 routing fans tasks correctly.
    auto* chunk_space = _chunk_memory_spaces.at(batch_idx);
    if (chunk_space == nullptr) {
      throw std::runtime_error(
        "[cached_split_provider] chunk_memory_spaces[" + std::to_string(batch_idx) +
        "] is null");
    }
    auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
      view, std::move(owner), alloc_size, *chunk_space, no_writer_stream);
    auto batch =
      std::make_shared<cucascade::data_batch>(::sirius::get_next_batch_id(), std::move(gpu_repr));

    connector.push_split(std::make_unique<op::scan::scan_cached_operator_data>(
      std::move(batch), _filter_expression, _plan));
  }

  connector.close();
  promise.set_value();
  return future;
}

}  // namespace sirius::scan_manager
