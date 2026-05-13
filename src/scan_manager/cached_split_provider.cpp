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

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

cached_split_provider::cached_split_provider(
  std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<duckdb::Expression> filter_expression,
  std::shared_ptr<op::scan::scan_plan const> plan)
  : _columns_per_request(std::move(columns_per_request)),
    _memory_space(&memory_space),
    _filter_expression(std::move(filter_expression)),
    _plan(std::move(plan))
{
}

cached_split_provider::cached_split_provider(
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
  std::vector<std::size_t> column_indices,
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<duckdb::Expression> filter_expression,
  std::shared_ptr<op::scan::scan_plan const> plan)
  : _host_chunks(std::move(host_chunks)),
    _column_indices(std::move(column_indices)),
    _memory_space(&memory_space),
    _filter_expression(std::move(filter_expression)),
    _plan(std::move(plan))
{
}

std::future<void> cached_split_provider::start(exec::thread_pool& /*pool*/,
                                               split_connector& connector)
{
  std::promise<void> promise;
  auto future = promise.get_future();

  if (!_host_chunks.empty()) {
    // HOST tier: slice each pinned chunk down to the requested columns and emit a
    // data_batch wrapping the resulting host_data_representation. The batch is
    // converted to GPU later by scan_cached_operator_data::prepare_for_processing,
    // so downstream execute() still sees a gpu_table_representation.
    for (auto const& chunk : _host_chunks) {
      if (!chunk) {
        throw std::runtime_error(
          "[cached_split_provider] null host_data_representation in pinned chunks");
      }
      auto sliced = chunk->slice(_column_indices);
      auto batch =
        cucascade::data_batch::make(::sirius::get_next_batch_id(), std::move(sliced));
      connector.push_split(std::make_unique<op::scan::scan_cached_operator_data>(
        std::move(batch), _filter_expression, _plan));
    }

    connector.close();
    promise.set_value();
    return future;
  }

  std::size_t const num_batches =
    _columns_per_request.empty() ? 0 : _columns_per_request.front().size();

  // Sanity-check: every column must contribute the same number of chunks.
  for (auto const& col_chunks : _columns_per_request) {
    if (col_chunks.size() != num_batches) {
      throw std::runtime_error(
        "[cached_split_provider] mismatched chunk count across requested columns");
    }
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
    auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
      view, std::move(owner), alloc_size, *_memory_space);
    auto batch =
      cucascade::data_batch::make(::sirius::get_next_batch_id(), std::move(gpu_repr));

    connector.push_split(std::make_unique<op::scan::scan_cached_operator_data>(
      std::move(batch), _filter_expression, _plan));
  }

  connector.close();
  promise.set_value();
  return future;
}

}  // namespace sirius::scan_manager
