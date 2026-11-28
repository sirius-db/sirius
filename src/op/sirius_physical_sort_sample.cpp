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

#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "helper/cuda_launch_error.hpp"
#include "log/logging.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/sort_range_count.hpp"
#include "pipeline/data_size_estimator.hpp"
#include "pipeline/sirius_pipeline.hpp"  // IWYU pragma: keep
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/merge.hpp>
#include <cudf/reduction/unique_count.hpp>

#include <rmm/error.hpp>

#include <thrust/system/system_error.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace sirius {
namespace op {

namespace {

std::optional<std::size_t> count_repeated_boundaries(cudf::table const* boundaries,
                                                     ::cuda::stream_ref stream)
{
  if (!boundaries || boundaries->num_rows() == 0) { return 0; }
  auto const view = boundaries->view();
  // Diagnostic only. Retrying boundary computation would re-enter the completed collective's
  // hook, which rejects the retry and fails the query. So an error the task would retry on
  // leaves the count unavailable; any other error propagates.
  try {
    return view.num_rows() - cudf::unique_count(view, cudf::null_equality::EQUAL, stream);
  } catch (rmm::out_of_memory const&) {
    return std::nullopt;
  } catch (thrust::system_error const& error) {
    if (!is_retryable_launch_error(static_cast<cudaError_t>(error.code().value()))) { throw; }
    return std::nullopt;
  }
}

}  // namespace

std::unique_ptr<cudf::column> sirius_physical_sort_sample::index_column(
  std::vector<int32_t> const& indices, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
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

std::unique_ptr<cudf::table> sirius_physical_sort_sample::evenly_spaced(
  cudf::table_view sorted,
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

sort_boundaries sirius_physical_sort_sample::local_boundaries(
  cudf::table_view sample_keys,
  std::size_t num_parts,
  cucascade::memory::memory_space* space,
  ::cuda::stream_ref stream)
{
  auto const rows = sample_keys.num_rows();
  // More partitions than key rows would take a key row twice and leave partitions empty.
  num_parts = std::min(num_parts, static_cast<std::size_t>(rows));
  // Single partition — no boundaries needed
  if (num_parts <= 1) { return {}; }
  return {num_parts,
          evenly_spaced(sample_keys, 0, rows, num_parts, stream, space->get_default_allocator())};
}

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

bool sirius_physical_sort_sample::all_ports_empty()
{
  if (_cross_rank_boundaries && boundary_state() == BoundaryState::NOT_DONE) { return false; }
  return sirius_physical_operator::all_ports_empty();
}

std::optional<task_creation_hint> sirius_physical_sort_sample::get_next_task_hint()
{
  const auto state = _boundary_state.load(std::memory_order_acquire);

  // Boundaries already computed — process batches one at a time.
  if (state == BoundaryState::DONE) { return sirius_physical_operator::get_next_task_hint(); }

  // Boundary task already scheduled; wait for it to finish before creating more tasks.
  if (state == BoundaryState::SCHEDULED) { return std::nullopt; }

  // NOT_DONE: wait until enough input was handed over before scheduling the boundary task.
  auto port_ids = get_port_ids();
  if (port_ids.empty()) { return std::nullopt; }

  auto* p = get_port(port_ids[0]);
  if (!p || !p->repo) { return std::nullopt; }

  bool upstream_finished = p->src_pipeline && p->src_pipeline->is_pipeline_finished();
  if (boundary_input_ready(*p)) { return task_creation_hint{TaskCreationHint::READY, this}; }

  if (p->src_pipeline && !upstream_finished) {
    auto* producer = &(p->src_pipeline->get_operators()[0].get());
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
  }

  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_sort_sample::get_next_task_input_data()
{
  const auto state = _boundary_state.load(std::memory_order_acquire);

  // After boundaries are computed, process one batch at a time (passthrough mode).
  if (state == BoundaryState::DONE) {
    auto input = sirius_physical_operator::get_next_task_input_data();
    if (input) {
      auto const& batches =
        dynamic_cast<pipelineable_operator_data const&>(*input).get_data_batches();
      _input_batches.fetch_add(batches.size(), std::memory_order_relaxed);
    }
    return input;
  }

  // Boundary input already claimed by the in-flight boundary task.
  if (state == BoundaryState::SCHEDULED) { return nullptr; }

  // NOT_DONE: the boundary task takes one batch. Its sample is the key rows handed over.
  auto port_ids = get_port_ids();
  if (port_ids.empty()) { return nullptr; }

  auto* p = get_port(port_ids[0]);
  if (!p || !p->repo) { return nullptr; }
  if (!boundary_input_ready(*p)) { return nullptr; }

  std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
  if (auto batch = p->repo->pop_next_data_batch()) {
    _input_batches.fetch_add(1, std::memory_order_relaxed);
    input_batch.push_back(std::move(batch));
  }

  if (input_batch.empty()) {
    // An operator that owes its boundary task without input takes it once its input has ended.
    bool const upstream_finished = p->src_pipeline && p->src_pipeline->is_pipeline_finished();
    if (!runs_without_input() || !upstream_finished) { return nullptr; }
  }

  _boundary_state.store(BoundaryState::SCHEDULED, std::memory_order_release);
  return std::make_unique<pipelineable_operator_data>(std::move(input_batch));
}

sirius_physical_sort_sample::key_order sirius_physical_sort_sample::sort_key_order() const
{
  key_order keys;
  keys.columns.reserve(orders.size());
  keys.column_order.reserve(orders.size());
  keys.null_precedence.reserve(orders.size());
  for (auto const& ord : orders) {
    if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("Sort sample only supports bound reference expressions");
    }
    keys.columns.push_back(
      static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index()));
    keys.column_order.push_back(to_cudf_order(ord.type));
    keys.null_precedence.push_back(to_cudf_null_order(ord.type, ord.null_order));
  }
  return keys;
}

bool sirius_physical_sort_sample::boundary_input_ready(port const& input) const
{
  // Upstream done with no rows still needs a READY signal so the pipeline can drain.
  if (input.src_pipeline && input.src_pipeline->is_pipeline_finished()) { return true; }
  if (input.repo->total_size() == 0) { return false; }
  // 0 waits for the input to end; a port with no source pipeline has no end to wait for.
  if (_sort_sample_bytes == 0) { return !input.src_pipeline; }
  return _key_sample->seen().bytes >= _sort_sample_bytes;
}

sirius_physical_sort_sample::range_sizing sirius_physical_sort_sample::size_ranges(
  sample_summary const& summary,
  cucascade::memory::memory_space& space,
  ::cuda::stream_ref stream) const
{
  // Size ranges from a finished input, runtime projection, or planner fallback.
  range_sizing sizing;
  sizing.estimated_total_bytes = summary.bytes;
  std::size_t rows_for_sizing  = summary.rows;
  if (summary.upstream_finished) {
    sizing.est_source = "measured";
  } else {
    auto const rows_for_bytes = [&](std::size_t bytes) {
      if (summary.bytes == 0 || summary.rows == 0) { return std::size_t{0}; }
      auto const projected =
        std::ceil(static_cast<long double>(bytes) * summary.rows / summary.bytes);
      auto const limit = static_cast<long double>(std::numeric_limits<std::size_t>::max());
      return projected >= limit ? std::numeric_limits<std::size_t>::max()
                                : static_cast<std::size_t>(projected);
    };
    if (summary.projected_bytes) {
      sizing.est_source            = "projected";
      sizing.estimated_total_bytes = std::max(*summary.projected_bytes, summary.bytes);
    } else if (summary.rows > 0 && estimated_cardinality > 0) {
      sizing.est_source = "planner";
      auto const scaled =
        std::ceil(static_cast<long double>(summary.bytes) * estimated_cardinality / summary.rows);
      auto const limit = static_cast<long double>(std::numeric_limits<std::size_t>::max());
      sizing.estimated_total_bytes =
        std::max(scaled >= limit ? std::numeric_limits<std::size_t>::max()
                                 : static_cast<std::size_t>(scaled),
                 summary.bytes);
    }
    rows_for_sizing = std::max(summary.rows, rows_for_bytes(sizing.estimated_total_bytes));
  }

  auto const available_memory = space.get_available_memory(stream);
  sizing.target_range_bytes   = _max_partition_bytes_override > 0
                                  ? _max_partition_bytes_override
                                  : static_cast<std::size_t>(static_cast<double>(available_memory) *
                                                           _max_partition_memory_fraction);
  sizing.num_parts            = choose_sort_range_count(
    sizing.estimated_total_bytes, rows_for_sizing, sizing.target_range_bytes, summary.rows);
  SIRIUS_LOG_DEBUG(
    "Sort sample: est_source={}, complete_input={}, sample_rows={}, sample_bytes={}, "
    "bytes_for_sizing={}, rows_for_sizing={}, available_memory={}, max_partition_bytes={}, "
    "num_partitions={}",
    sizing.est_source,
    summary.complete_input,
    summary.rows,
    summary.bytes,
    sizing.estimated_total_bytes,
    rows_for_sizing,
    available_memory,
    sizing.target_range_bytes,
    sizing.num_parts);
  return sizing;
}

void sirius_physical_sort_sample::set_boundaries_without_sample(::cuda::stream_ref stream)
{
  auto made                      = _cross_rank_boundaries ? _cross_rank_boundaries->make(
                                         {{}, 0, _sort_sample_bytes, 1, batch_telemetry()}, stream)
                                                          : sort_boundaries{};
  _num_partitions                = made.num_partitions;
  _partition_boundaries          = std::move(made.table);
  auto const repeated_boundaries = count_repeated_boundaries(_partition_boundaries.get(), stream);
  _range_stats = {.num_ranges = _num_partitions, .repeated_boundaries = repeated_boundaries};
  _boundary_state.store(BoundaryState::DONE, std::memory_order_release);
}

void sirius_physical_sort_sample::set_boundaries(sample_summary const& summary,
                                                 range_sizing const& sizing,
                                                 key_order const& keys,
                                                 std::vector<cudf::table_view> const& key_tables,
                                                 std::size_t stride,
                                                 cucascade::memory::memory_space& space,
                                                 ::cuda::stream_ref stream)
{
  if (key_tables.empty()) {
    set_boundaries_without_sample(stream);
    return;
  }
  std::unique_ptr<cudf::table> merged_keys;
  if (key_tables.size() > 1) {
    std::vector<cudf::size_type> key_indices(keys.columns.size());
    std::iota(key_indices.begin(), key_indices.end(), 0);
    merged_keys = cudf::merge(key_tables,
                              key_indices,
                              keys.column_order,
                              keys.null_precedence,
                              stream,
                              space.get_default_allocator());
  }
  cudf::table_view const sort_keys_view = merged_keys ? merged_keys->view() : key_tables.front();
  auto const sample_key_rows            = static_cast<std::size_t>(sort_keys_view.num_rows());
  auto made =
    _cross_rank_boundaries
      ? _cross_rank_boundaries->make(
          {sort_keys_view, summary.bytes, _sort_sample_bytes, sizing.num_parts, batch_telemetry()},
          stream)
      : local_boundaries(sort_keys_view, sizing.num_parts, &space, stream);
  _num_partitions                = made.num_partitions;
  _partition_boundaries          = std::move(made.table);
  auto const repeated_boundaries = count_repeated_boundaries(_partition_boundaries.get(), stream);
  _range_stats                   = {.complete_input        = summary.complete_input,
                                    .sample_batches        = summary.batches,
                                    .sample_rows           = summary.rows,
                                    .sample_key_rows       = sample_key_rows,
                                    .sample_stride         = stride,
                                    .sample_bytes          = summary.bytes,
                                    .estimated_total_bytes = sizing.estimated_total_bytes,
                                    .est_source            = sizing.est_source,
                                    .target_range_bytes    = sizing.target_range_bytes,
                                    .num_ranges            = _num_partitions,
                                    .repeated_boundaries   = repeated_boundaries};

  // sort_partition runs in the same gpu_pipeline_task immediately after execute() returns, so
  // _partition_boundaries is visible before partition runs.
  _boundary_state.store(BoundaryState::DONE, std::memory_order_release);
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
  SIRIUS_LOG_DEBUG("Sort sample: computing partition boundaries ({} bytes target)",
                   _sort_sample_bytes);
  auto start = std::chrono::high_resolution_clock::now();

  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (!space) { space = batch.get_memory_space(); }
  }

  if (!space) {
    set_boundaries_without_sample(stream);
    _key_sample->close();
    _key_sample->clear();
    return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
  }

  // If the GPU work below throws (e.g. rmm::out_of_memory) the state stays SCHEDULED and the
  // handed-over key rows stay. The rescheduled task carries the same input and retries here, and
  // task_creator spawns no duplicate boundary task.
  auto const keys = sort_key_order();
  sample_summary summary;
  // Read before the key rows: once upstream has finished, every batch has been handed over.
  auto port_ids = get_port_ids();
  if (!port_ids.empty()) {
    auto* port                = get_port(port_ids[0]);
    summary.upstream_finished = port->src_pipeline && port->src_pipeline->is_pipeline_finished();
  }
  summary.complete_input = summary.upstream_finished;
  auto const sample      = _key_sample->on_device(*space, stream);
  summary.batches        = sample.input.batches;
  summary.rows           = sample.input.rows;
  summary.bytes          = sample.input.bytes;
  if (!summary.upstream_finished && !port_ids.empty()) {
    if (auto projected = pipeline::estimate_port_total_input_bytes(*this, port_ids[0])) {
      summary.projected_bytes = projected->bytes;
    }
  }
  auto const sizing = size_ranges(summary, *space, stream);
  set_boundaries(summary, sizing, keys, sample.tables, sample.stride, *space, stream);
  _key_sample->close();
  _key_sample->clear();

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Sort sample: computed {} partitions with {} boundaries in {:.2f} ms",
                   _num_partitions,
                   _partition_boundaries ? _partition_boundaries->num_rows() : 0,
                   duration.count() / 1000.0);

  return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
}

}  // namespace op
}  // namespace sirius
