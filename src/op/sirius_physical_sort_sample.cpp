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

#include "op/sirius_physical_sort_sample.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "exchange/all_gather.hpp"
#include "exchange/exchange_service.hpp"
#include "log/logging.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/search.hpp>

#include <algorithm>
#include <functional>
#include <numeric>

namespace sirius {
namespace op {

namespace {

uint64_t get_batch_bytes(const std::shared_ptr<::cucascade::data_batch>& batch)
{
  if (!batch) { return 0; }
  auto ro = batch->to_read_only();
  if (!ro.get_data()) { return 0; }
  return ro.get_data()->get_size_in_bytes();
}

bool repo_has_enough_sample_bytes(::cucascade::shared_data_repository* repo,
                                  uint64_t sort_sample_bytes,
                                  bool upstream_finished)
{
  if (!repo) { return false; }
  // Upstream done with no rows still needs a READY signal so the pipeline can drain.
  if (upstream_finished) { return true; }
  if (repo->total_size() == 0) { return false; }

  uint64_t accumulated = 0;
  for (size_t part = 0; part < repo->num_partitions(); ++part) {
    for (auto batch_id : repo->get_batch_ids(part)) {
      auto batch = repo->get_data_batch_by_id(batch_id, part);
      if (!batch) { continue; }
      accumulated += get_batch_bytes(batch);
      if (accumulated >= sort_sample_bytes) { return true; }
    }
  }
  return false;
}

}  // namespace

sirius_physical_sort_sample::sirius_physical_sort_sample(sirius_physical_order* order_by,
                                                         uint64_t sort_sample_bytes,
                                                         uint64_t max_partition_bytes,
                                                         double max_partition_memory_fraction)
  : sirius_physical_sort_sample(order_by->types,
                                copy_orders(order_by->orders),
                                order_by->estimated_cardinality,
                                sort_sample_bytes,
                                max_partition_bytes,
                                max_partition_memory_fraction)
{
}

sirius_physical_sort_sample::sirius_physical_sort_sample(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  std::size_t estimated_cardinality,
  uint64_t sort_sample_bytes,
  uint64_t max_partition_bytes,
  double max_partition_memory_fraction)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::SORT_SAMPLE, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    _sort_sample_bytes(sort_sample_bytes),
    _max_partition_bytes_override(max_partition_bytes),
    _max_partition_memory_fraction(max_partition_memory_fraction)
{
}

std::optional<task_creation_hint> sirius_physical_sort_sample::get_next_task_hint()
{
  const auto state = _boundary_state.load(std::memory_order_acquire);

  // Boundaries already computed — process batches one at a time.
  if (state == BoundaryState::DONE) { return sirius_physical_operator::get_next_task_hint(); }

  // Boundary task already scheduled; wait for it to finish before creating more tasks.
  if (state == BoundaryState::SCHEDULED) { return std::nullopt; }

  // NOT_DONE: wait for enough sample bytes before scheduling the boundary task.
  auto port_ids = get_port_ids();
  if (port_ids.empty()) { return std::nullopt; }

  auto* p = get_port(port_ids[0]);
  if (!p || !p->repo) { return std::nullopt; }

  bool upstream_finished = p->src_pipeline && p->src_pipeline->is_pipeline_finished();
  if (repo_has_enough_sample_bytes(p->repo, _sort_sample_bytes, upstream_finished)) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }

  if (p->src_pipeline && !upstream_finished) {
    auto* producer = &(p->src_pipeline->get_operators()[0].get());
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
  }

  return std::nullopt;
}

bool sirius_physical_sort_sample::all_ports_empty()
{
  if (_distributed && _boundary_state.load(std::memory_order_acquire) == BoundaryState::NOT_DONE) {
    return false;
  }
  return sirius_physical_operator::all_ports_empty();
}

std::unique_ptr<operator_data> sirius_physical_sort_sample::get_next_task_input_data()
{
  const auto state = _boundary_state.load(std::memory_order_acquire);

  // After boundaries are computed, process one batch at a time (passthrough mode).
  if (state == BoundaryState::DONE) { return sirius_physical_operator::get_next_task_input_data(); }

  // Boundary input already claimed by the in-flight boundary task.
  if (state == BoundaryState::SCHEDULED) { return nullptr; }

  // NOT_DONE: accumulate batches up to the sample byte threshold so execute() can merge
  // pre-sorted runs and derive partition boundaries from a representative sample.
  auto port_ids = get_port_ids();
  if (port_ids.empty()) { return nullptr; }

  auto* p = get_port(port_ids[0]);
  if (!p || !p->repo) { return nullptr; }

  std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
  uint64_t accumulated_bytes = 0;
  while (true) {
    auto batch = p->repo->pop_next_data_batch();
    if (!batch) { break; }
    accumulated_bytes += get_batch_bytes(batch);
    input_batch.push_back(std::move(batch));
    if (accumulated_bytes >= _sort_sample_bytes) { break; }
  }

  if (input_batch.empty()) {
    // Distributed: a rank with no input still joins the boundary exchange the others wait in.
    bool const upstream_finished = p->src_pipeline && p->src_pipeline->is_pipeline_finished();
    if (!_distributed || !upstream_finished) { return nullptr; }
  }

  _boundary_state.store(BoundaryState::SCHEDULED, std::memory_order_release);
  return std::make_unique<pipelineable_operator_data>(std::move(input_batch));
}

std::unique_ptr<operator_data> sirius_physical_sort_sample::execute(const operator_data& input_data,
                                                                    ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_sort_sample::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  // Fast path: boundaries already computed — just pass through.
  if (_boundary_state.load(std::memory_order_acquire) == BoundaryState::DONE) {
    SIRIUS_LOG_DEBUG("Sort sample: passthrough ({} batches)", input_batches.size());
    return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
  }

  // do boundary computation
  SIRIUS_LOG_DEBUG("Sort sample: computing partition boundaries from {} batches ({} bytes target)",
                   input_batches.size(),
                   _sort_sample_bytes);
  auto start = std::chrono::high_resolution_clock::now();

  // 1. Build cudf order vectors from BoundOrderByNode
  std::vector<int> order_key_idx;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  order_key_idx.reserve(orders.size());
  column_order.reserve(orders.size());
  null_precedence.reserve(orders.size());

  for (auto const& ord : orders) {
    if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("Sort sample only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index());
    order_key_idx.push_back(idx);
    column_order.push_back(to_cudf_order(ord.type));
    null_precedence.push_back(to_cudf_null_order(ord.type, ord.null_order));
  }

  // 2. Collect valid batches and find memory space
  std::vector<::cucascade::read_only_data_batch> valid_batches;
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (!space) { space = batch.get_memory_space(); }
    valid_batches.push_back(batch);
  }

  if (valid_batches.empty() || !space) {
    if (_distributed) {
      compute_distributed_boundaries({}, 1, column_order, null_precedence, stream);
    }
    _boundary_state.store(BoundaryState::DONE, std::memory_order_release);
    return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
  }

  // Wrap GPU work in try/catch: if any allocation throws (e.g. rmm::out_of_memory), leave the state
  // at SCHEDULED and rethrow. The rescheduled task carries the same input and re-enters here to
  // retry, and keeping SCHEDULED prevents task_creator from spawning a duplicate boundary task.
  try {
    size_t total_sample_bytes = 0;
    for (auto const& batch : valid_batches) {
      total_sample_bytes += batch.get_data()->get_size_in_bytes();
    }

    // 3. Merge pre-sorted sample batches (ORDER_BY sorts each batch locally).
    cudf::table_view merged_sample_view;
    std::shared_ptr<::cucascade::data_batch> merged_sample_batch;
    if (valid_batches.size() == 1) {
      merged_sample_view = get_cudf_table_view(valid_batches[0]);
    } else {
      merged_sample_batch = gpu_merge_impl::merge_order_by(valid_batches,
                                                           order_key_idx,
                                                           column_order,
                                                           null_precedence,
                                                           stream,
                                                           *space,
                                                           batch_telemetry());
      merged_sample_view  = get_cudf_table_view(merged_sample_batch->to_read_only());
    }

    // 4. Compute number of partitions
    bool complete_input = false;
    auto port_ids       = get_port_ids();
    if (!port_ids.empty()) {
      auto* port = get_port(port_ids[0]);
      // Check completion before repository emptiness: once completion is visible,
      // the upstream pipeline cannot publish another batch.
      complete_input = port && port->src_pipeline && port->src_pipeline->is_pipeline_finished() &&
                       port->repo && port->repo->all_empty();
    }

    size_t total_rows      = static_cast<size_t>(merged_sample_view.num_rows());
    size_t avg_batch_bytes = valid_batches.empty() ? 0 : total_sample_bytes / valid_batches.size();
    size_t avg_rows_per_batch = valid_batches.empty() ? 0 : total_rows / valid_batches.size();
    size_t num_parts          = 1;
    bool const can_size = complete_input || (estimated_cardinality > 0 && avg_rows_per_batch > 0);
    if (!can_size) {
      SIRIUS_LOG_WARN(
        "Sort sample: estimated_cardinality={} or avg_rows_per_batch={} is zero, "
        "defaulting to 1 partition",
        estimated_cardinality,
        avg_rows_per_batch);
    } else {
      size_t total_batch_count = valid_batches.size();
      size_t bytes_for_sizing  = total_sample_bytes;
      if (!complete_input) {
        total_batch_count = (estimated_cardinality + avg_rows_per_batch - 1) / avg_rows_per_batch;
        bytes_for_sizing  = avg_batch_bytes * total_batch_count;
      }
      size_t available_memory    = space->get_available_memory(stream);
      size_t max_partition_bytes = _max_partition_bytes_override > 0
                                     ? _max_partition_bytes_override
                                     : static_cast<size_t>(static_cast<double>(available_memory) *
                                                           _max_partition_memory_fraction);

      if (max_partition_bytes > 0 && bytes_for_sizing > max_partition_bytes) {
        num_parts = (bytes_for_sizing + max_partition_bytes - 1) / max_partition_bytes;
      }
      num_parts = std::min(num_parts, std::max<size_t>(1, total_rows));

      SIRIUS_LOG_DEBUG(
        "Sort sample: complete_input={}, estimated_cardinality={}, total_rows={}, "
        "avg_rows_per_batch={}, "
        "avg_batch_bytes={}, total_batch_count={}, "
        "bytes_for_sizing={}, available_memory={}, max_partition_bytes={}, num_partitions={}",
        complete_input,
        estimated_cardinality,
        total_rows,
        avg_rows_per_batch,
        avg_batch_bytes,
        total_batch_count,
        bytes_for_sizing,
        available_memory,
        max_partition_bytes,
        num_parts);
    }

    // 5. Pick P-1 evenly-spaced boundary rows from the merged sample (sort key columns only)
    if (_distributed) {
      std::vector<cudf::column_view> key_cols;
      for (int idx : order_key_idx) {
        key_cols.push_back(merged_sample_view.column(idx));
      }
      compute_distributed_boundaries(
        cudf::table_view{key_cols}, num_parts, column_order, null_precedence, stream);
    } else if (num_parts <= 1 || total_rows == 0) {
      // Single partition — no boundaries needed
      _num_partitions = 1;
      _partition_boundaries.reset();
    } else {
      // Compute boundary row indices: [total_rows/P, 2*total_rows/P, ..., (P-1)*total_rows/P]
      size_t num_boundaries = num_parts - 1;
      std::vector<int32_t> boundary_indices_host;
      boundary_indices_host.reserve(num_boundaries);
      for (size_t i = 1; i <= num_boundaries; i++) {
        auto idx = static_cast<int32_t>((i * total_rows) / num_parts);
        if (idx >= static_cast<int32_t>(total_rows)) { idx = static_cast<int32_t>(total_rows) - 1; }
        boundary_indices_host.push_back(idx);
      }

      // Create a device column with the boundary indices
      auto indices_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                   static_cast<cudf::size_type>(num_boundaries),
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   space->get_default_allocator());
      CUDF_CUDA_TRY(cudaMemcpyAsync(indices_col->mutable_view().data<int32_t>(),
                                    boundary_indices_host.data(),
                                    num_boundaries * sizeof(int32_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));

      // Extract only the sort key columns from merged sample for the boundaries
      std::vector<cudf::column_view> sort_key_cols;
      for (int idx : order_key_idx) {
        sort_key_cols.push_back(merged_sample_view.column(idx));
      }
      cudf::table_view sort_keys_view(sort_key_cols);

      // Gather boundary rows
      _partition_boundaries = cudf::gather(sort_keys_view,
                                           indices_col->view(),
                                           cudf::out_of_bounds_policy::DONT_CHECK,
                                           stream,
                                           space->get_default_allocator());
      _num_partitions       = num_parts;
    }

    // sort_partition runs in the same gpu_pipeline_task immediately after this
    // execute() returns, so _partition_boundaries is visible before partition runs.
    _boundary_state.store(BoundaryState::DONE, std::memory_order_release);

  } catch (...) {
    // Leave the state at SCHEDULED so the rescheduled task retries boundary computation without
    // task_creator treating the operator as unscheduled and creating its own duplicate task.
    throw;
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Sort sample: computed {} partitions with {} boundaries in {:.2f} ms",
                   _num_partitions,
                   _partition_boundaries ? _partition_boundaries->num_rows() : 0,
                   duration.count() / 1000.0);

  return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
}

namespace {

/// A device INT32 column holding @p indices.
std::unique_ptr<cudf::column> index_column(std::vector<int32_t> const& indices,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(indices.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<int32_t>(),
                                indices.data(),
                                indices.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  return column;
}

/// Rows i * rows / parts of @p sorted for i in [1, parts), offset by @p first: parts - 1
/// evenly spaced boundaries.
std::unique_ptr<cudf::table> evenly_spaced(cudf::table_view sorted,
                                           cudf::size_type first,
                                           cudf::size_type rows,
                                           std::size_t parts,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  std::vector<int32_t> indices;
  for (std::size_t i = 1; i < parts; ++i) {
    auto const at = static_cast<std::size_t>(rows) * i / parts;
    indices.push_back(first + static_cast<int32_t>(std::min<std::size_t>(at, rows - 1)));
  }
  auto const column = index_column(indices, stream, mr);
  return cudf::gather(sorted, column->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
}

}  // namespace

void sirius_physical_sort_sample::compute_distributed_boundaries(
  cudf::table_view sample_keys,
  std::size_t num_parts,
  std::vector<cudf::order> const& column_order,
  std::vector<cudf::null_order> const& null_precedence,
  ::cuda::stream_ref stream)
{
  auto& d          = *_distributed;
  auto& gpu        = *d.gpu;
  auto const mr    = gpu.get_default_allocator();
  auto const world = d.service->world_size();
  auto const rank  = d.service->rank();

  // 1. This rank's summary: up to distributed_summary_rows evenly spaced rows of its sample.
  std::shared_ptr<::cucascade::data_batch> summary;
  if (auto const rows = sample_keys.num_rows(); rows > 0) {
    auto const take = std::min<std::size_t>(rows, distributed_summary_rows);
    std::vector<int32_t> indices(take);
    for (std::size_t i = 0; i < take; ++i) {
      indices[i] = static_cast<int32_t>(i * static_cast<std::size_t>(rows) / take);
    }
    auto const column = index_column(indices, stream, mr);
    summary           = make_data_batch(
      cudf::gather(sample_keys, column->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr),
      gpu,
      stream,
      batch_telemetry());
  }

  // 2. Every rank's summary, merged in rank order: the same global sample on every rank.
  auto gathered = exchange::all_gather(*d.service, d.stream, summary, gpu, stream.get());
  std::vector<::cucascade::read_only_data_batch> parts;
  for (auto const& g : gathered) {
    if (g) { parts.push_back(g->to_read_only()); }
  }
  _num_partitions = 1;
  _partition_boundaries.reset();
  if (parts.empty()) { return; }
  std::vector<int> keys(column_order.size());
  std::iota(keys.begin(), keys.end(), 0);
  std::shared_ptr<::cucascade::data_batch> merged;
  if (parts.size() > 1) {
    merged = gpu_merge_impl::merge_order_by(
      parts, keys, column_order, null_precedence, stream, gpu, batch_telemetry());
  }
  auto const global =
    merged ? get_cudf_table_view(merged->to_read_only()) : get_cudf_table_view(parts.front());
  auto const total = global.num_rows();

  // 3. The range exchange's world - 1 boundaries, evenly spaced over the global sample.
  cudf::size_type lo = 0;
  cudf::size_type hi = total;
  if (world > 1) {
    std::shared_ptr<cudf::table const> exchange_boundaries =
      evenly_spaced(global, 0, total, world, stream, mr);
    // This rank's range of the global sample: from the first row not below its lower boundary
    // to the first not below its upper one, as the exchange routes rows.
    auto const starts = cudf::lower_bound(
      global, exchange_boundaries->view(), column_order, null_precedence, stream, mr);
    std::vector<cudf::size_type> at(world - 1);
    CUDF_CUDA_TRY(cudaMemcpyAsync(at.data(),
                                  starts->view().data<cudf::size_type>(),
                                  at.size() * sizeof(cudf::size_type),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    if (rank > 0) { lo = at[rank - 1]; }
    if (rank + 1 < world) { hi = at[rank]; }
    d.set_exchange_boundaries(std::move(exchange_boundaries));
  }

  // 4. This rank's partition boundaries, from the part of the global sample inside its range.
  auto const parts_here = std::min<std::size_t>(num_parts, std::max<cudf::size_type>(1, hi - lo));
  if (parts_here <= 1) { return; }
  _partition_boundaries = evenly_spaced(global, lo, hi - lo, parts_here, stream, mr);
  _num_partitions       = parts_here;
}

}  // namespace op
}  // namespace sirius
