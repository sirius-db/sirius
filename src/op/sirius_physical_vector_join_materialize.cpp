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

#include "vss/sirius_physical_vector_join_materialize.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "vss/pinned_column.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op {

sirius_physical_vector_join_materialize::sirius_physical_vector_join_materialize(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  sirius::vss::vector_join_request request,
  sirius::scan_manager::sirius_scan_manager* scan_manager)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN_MATERIALIZE, std::move(types), estimated_cardinality),
    _request(std::move(request)),
    _scan_manager(scan_manager)
{
}

void sirius_physical_vector_join_materialize::ensure_initialized()
{
  std::lock_guard<std::mutex> lg(_init_mutex);
  if (_initialized) { return; }
  if (_scan_manager == nullptr) {
    throw std::runtime_error("[sirius_physical_vector_join_materialize] no scan manager set");
  }

  auto const& right = _request.right;
  const auto* right_pin =
    _scan_manager->find_pinned_entry_for_duckdb_table(right.catalog, right.schema, right.table);
  if (right_pin == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_materialize] right table is no longer pinned");
  }
  auto& right_space = vss::pinned_entry_gpu_space(*right_pin);

  // Per-batch views of each right output column so a row can be gathered from whichever
  // batch owns it, plus each batch's global row offset (prefix sum). This keeps the right
  // table addressable without a second copy.
  _right_output_views.resize(right.output_columns.size());
  for (std::size_t c = 0; c < right.output_columns.size(); ++c) {
    _right_output_views[c] =
      vss::pinned_column_chunk_views(*right_pin, right.output_columns[c], right_space);
  }

  auto const n_batches = _right_output_views.empty() ? 0 : _right_output_views.front().size();
  _right_offsets.resize(n_batches);
  std::int64_t acc = 0;
  for (std::size_t b = 0; b < n_batches; ++b) {
    _right_offsets[b] = acc;
    acc += static_cast<std::int64_t>(_right_output_views.front()[b].size());
  }

  _initialized = true;
}

std::unique_ptr<cudf::table> sirius_physical_vector_join_materialize::gather_right_by_batch(
  cudf::column_view const& col0_partitioned,
  std::vector<cudf::size_type> const& part_offsets,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  auto const n_batches = _right_output_views.empty() ? 0 : _right_output_views.front().size();

  std::vector<std::unique_ptr<cudf::table>> pieces;
  std::vector<cudf::table_view> piece_views;
  pieces.reserve(n_batches);
  piece_views.reserve(n_batches);

  for (std::size_t b = 0; b < n_batches; ++b) {
    auto const start = part_offsets[b];
    auto const end   = part_offsets[b + 1];
    if (start == end) { continue; }  // no rows routed to this batch

    // This batch's global row numbers
    auto const slice = cudf::slice(col0_partitioned, {start, end}).front();
    cudf::numeric_scalar<std::int64_t> const off(_right_offsets[b], true, stream);
    auto local = cudf::binary_operation(
      slice, off, cudf::binary_operator::SUB, cudf::data_type{cudf::type_id::INT64}, stream, mr);

    std::vector<cudf::column_view> batch_cols;
    batch_cols.reserve(_right_output_views.size());
    for (auto const& per_batch : _right_output_views) {
      batch_cols.push_back(per_batch[b]);
    }

    auto gathered = cudf::gather(cudf::table_view(batch_cols),
                                 local->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
    piece_views.push_back(gathered->view());
    pieces.push_back(std::move(gathered));
  }

  CUDF_EXPECTS(!piece_views.empty(), "VSS materialize: no right rows to gather");
  // Pieces are in ascending batch order, matching the partitioned row order.
  if (piece_views.size() == 1) { return std::move(pieces.front()); }
  return cudf::concatenate(piece_views, stream, mr);
}

std::unique_ptr<operator_data> sirius_physical_vector_join_materialize::get_next_task_input_data()
{
  // One task per partition (one left batch): drain its merge outputs.
  std::lock_guard<std::mutex> lg(_drain_mutex);

  auto* repo = ports.begin()->second->repo;
  if (_current_partition_index >= repo->num_partitions()) { return nullptr; }

  std::vector<std::shared_ptr<cucascade::data_batch>> all_batches;
  while (true) {
    auto batch = repo->pop_next_data_batch(_current_partition_index);
    if (!batch) { break; }
    all_batches.push_back(std::move(batch));
  }
  auto const partition_idx = _current_partition_index++;
  if (all_batches.empty()) { return nullptr; }
  return std::make_unique<partitioned_operator_data>(std::move(all_batches), partition_idx);
}

std::unique_ptr<operator_data> sirius_physical_vector_join_materialize::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_join_materialize::execute"};

  auto const& input         = dynamic_cast<const partitioned_operator_data&>(input_data);
  auto const& input_batches = input.get_read_only_batches();

  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (space == nullptr) { space = batch.get_memory_space(); }
  }
  if (input_batches.empty() || space == nullptr) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }
  auto const mr = space->get_default_allocator();

  // Input layout from reduce_local: reduce_local emits one batch per partition
  std::vector<cudf::table_view> input_views;
  input_views.reserve(input_batches.size());
  for (auto const& ro : input_batches) {
    input_views.push_back(sirius::get_cudf_table_view(ro));
  }
  std::unique_ptr<cudf::table> concatenated;
  cudf::table_view in_tv = input_views.front();
  if (input_views.size() > 1) {
    concatenated = cudf::concatenate(input_views, stream, mr);
    in_tv        = concatenated->view();
  }

  auto const n_left_cols = _request.left.output_columns.size();
  CUDF_EXPECTS(static_cast<std::size_t>(in_tv.num_columns()) == n_left_cols + 2,
               "VSS materialize: input is not [left cols…, col0, distance]");
  auto const col0_idx = static_cast<cudf::size_type>(n_left_cols);

  // Score: distance, or cosine similarity = max(0, 1 - distance).
  auto make_score = [&](cudf::column_view const& distance_view) -> std::unique_ptr<cudf::column> {
    if (_request.metric == "cosine") {
      cudf::numeric_scalar<float> const lo(0.0F, true, stream);
      cudf::numeric_scalar<float> const hi(2.0F, true, stream);
      auto distance = cudf::clamp(distance_view, lo, hi, stream, mr);
      if (_request.output_type == sirius::vss::vector_join_output_type::similarity) {
        cudf::numeric_scalar<float> const one(1.0F, true, stream);
        return cudf::binary_operation(one,
                                      distance->view(),
                                      cudf::binary_operator::SUB,
                                      cudf::data_type{cudf::type_id::FLOAT32},
                                      stream,
                                      mr);
      }
      return distance;
    }
    return std::make_unique<cudf::column>(distance_view, stream, mr);
  };
  auto copy_col = [&](cudf::column_view const& v) {
    return std::make_unique<cudf::column>(v, stream, mr);
  };

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(n_left_cols + _request.right.output_columns.size() + 1);

  if (_request.right.is_fast_path) {
    for (std::size_t c = 0; c < n_left_cols; ++c) {
      out_cols.push_back(copy_col(in_tv.column(static_cast<cudf::size_type>(c))));
    }
    auto const& col0  = in_tv.column(col0_idx);
    auto const target = sirius::get_cudf_type(get_types()[col0_idx]);
    out_cols.push_back(col0.type().id() == target.id() ? copy_col(col0)
                                                       : cudf::cast(col0, target, stream, mr));
    out_cols.push_back(make_score(in_tv.column(col0_idx + 1)));
  } else {
    ensure_initialized();
    auto const n_batches = _right_offsets.size();

    // batch_id per row = upper_bound(batch_starts, col0) - 1 (which batch owns col0).
    auto batch_starts = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                  static_cast<cudf::size_type>(n_batches),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(batch_starts->mutable_view().data<std::int64_t>(),
                                  _right_offsets.data(),
                                  n_batches * sizeof(std::int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    auto counts = cudf::upper_bound(cudf::table_view{{batch_starts->view()}},
                                    cudf::table_view{{in_tv.column(col0_idx)}},
                                    {cudf::order::ASCENDING},
                                    {cudf::null_order::BEFORE},
                                    stream,
                                    mr);
    cudf::numeric_scalar<std::int32_t> const one_i(1, true, stream);
    auto batch_id = cudf::binary_operation(counts->view(),
                                           one_i,
                                           cudf::binary_operator::SUB,
                                           cudf::data_type{cudf::type_id::INT32},
                                           stream,
                                           mr);

    auto [parted, part_offsets] =
      cudf::partition(in_tv, batch_id->view(), static_cast<cudf::size_type>(n_batches), stream, mr);
    auto const parted_tv = parted->view();

    for (std::size_t c = 0; c < n_left_cols; ++c) {
      out_cols.push_back(copy_col(parted_tv.column(static_cast<cudf::size_type>(c))));
    }
    auto right_cols = gather_right_by_batch(parted_tv.column(col0_idx), part_offsets, stream, mr);
    for (auto& c : right_cols->release()) {
      out_cols.push_back(std::move(c));
    }
    out_cols.push_back(make_score(parted_tv.column(col0_idx + 1)));
  }

  auto out_table = std::make_unique<cudf::table>(std::move(out_cols));
  auto batch     = sirius::make_data_batch(std::move(out_table), *space, stream, batch_telemetry());
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_physical_vector_join_materialize::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  return std::max<std::size_t>(stats.bytes, std::size_t{1} << 20);
}

std::string sirius_physical_vector_join_materialize::params_to_string() const
{
  return _request.left.table + " x " + _request.right.table + " k=" + std::to_string(_request.k);
}

}  // namespace sirius::op
