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
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace sirius {
namespace op {

namespace {

constexpr std::size_t min_sample_key_rows          = 4096;
constexpr std::size_t key_rows_per_range           = 64;
constexpr std::size_t key_rows_per_batch_and_range = 10;
constexpr std::size_t max_sample_key_rows          = 1048576;

std::size_t capped_product(std::size_t a, std::size_t b)
{
  return b != 0 && a > max_sample_key_rows / b ? max_sample_key_rows
                                               : std::min(a * b, max_sample_key_rows);
}

std::size_t count_repeated_boundaries(cudf::table const* boundaries, ::cuda::stream_ref stream)
{
  if (!boundaries || boundaries->num_rows() == 0) { return 0; }
  auto const view = boundaries->view();
  return view.num_rows() - cudf::unique_count(view, cudf::null_equality::EQUAL, stream);
}

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
  // Single partition — no boundaries needed
  if (num_parts <= 1 || rows == 0) { return {}; }
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
    _input_batches.fetch_add(1, std::memory_order_relaxed);
    accumulated_bytes += get_batch_bytes(batch);
    input_batch.push_back(std::move(batch));
    if (accumulated_bytes >= _sort_sample_bytes) { break; }
  }

  if (input_batch.empty()) {
    // An operator that owes its boundary task without input takes it once its input has ended.
    bool const upstream_finished = p->src_pipeline && p->src_pipeline->is_pipeline_finished();
    if (!runs_without_input() || !upstream_finished) { return nullptr; }
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

  // 1. Collect valid batches and find memory space
  std::vector<::cucascade::read_only_data_batch> valid_batches;
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (!space) { space = batch.get_memory_space(); }
    valid_batches.push_back(batch);
  }

  if (valid_batches.empty() || !space) {
    auto made =
      _cross_rank_boundaries
        ? _cross_rank_boundaries->make({{}, 0, _sort_sample_bytes, 1, batch_telemetry()}, stream)
        : sort_boundaries{};
    _num_partitions                = made.num_partitions;
    _partition_boundaries          = std::move(made.table);
    auto const repeated_boundaries = count_repeated_boundaries(_partition_boundaries.get(), stream);
    _range_stats = {.num_ranges = _num_partitions, .repeated_boundaries = repeated_boundaries};
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

    // 2. Build cudf order vectors from BoundOrderByNode
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

    // 3. Count rows without merging the full-width sample.
    std::size_t total_rows = 0;
    for (auto const& batch : valid_batches) {
      total_rows += static_cast<std::size_t>(get_cudf_table_view(batch).num_rows());
    }

    // 4. Size ranges from a finished input, runtime projection, or planner fallback.
    auto const rows_for_bytes = [&](std::size_t bytes) {
      if (total_sample_bytes == 0 || total_rows == 0) { return std::size_t{0}; }
      auto const projected =
        std::ceil(static_cast<long double>(bytes) * total_rows / total_sample_bytes);
      auto const limit = static_cast<long double>(std::numeric_limits<std::size_t>::max());
      return projected >= limit ? std::numeric_limits<std::size_t>::max()
                                : static_cast<std::size_t>(projected);
    };
    bool complete_input           = false;
    bool upstream_finished        = false;
    std::size_t waiting_bytes     = 0;
    std::size_t waiting_rows      = 0;
    std::size_t unknown_row_bytes = 0;
    auto port_ids                 = get_port_ids();
    if (!port_ids.empty()) {
      auto* port = get_port(port_ids[0]);
      if (port && port->repo) {
        // Once upstream completion is visible, it publishes no more batches to this repository.
        upstream_finished = port->src_pipeline && port->src_pipeline->is_pipeline_finished();
        complete_input    = upstream_finished && port->repo->all_empty();
        for (std::size_t part = 0; part < port->repo->num_partitions(); ++part) {
          for (auto batch_id : port->repo->get_batch_ids(part)) {
            auto batch = port->repo->get_data_batch_by_id(batch_id, part);
            if (!batch) { continue; }
            auto const size = get_batch_rows_and_bytes(*batch);
            waiting_bytes   = memory::saturating_add(waiting_bytes, size.bytes);
            if (size.rows) {
              waiting_rows = memory::saturating_add(waiting_rows, *size.rows);
            } else {
              unknown_row_bytes = memory::saturating_add(unknown_row_bytes, size.bytes);
            }
          }
        }
      }
    }
    auto const observed_bytes    = memory::saturating_add(total_sample_bytes, waiting_bytes);
    std::size_t bytes_for_sizing = observed_bytes;
    std::size_t rows_for_sizing  = total_rows;
    std::string_view est_source  = "observed";
    if (upstream_finished) {
      est_source      = "measured";
      rows_for_sizing = memory::saturating_add(rows_for_sizing, waiting_rows);
      rows_for_sizing = memory::saturating_add(rows_for_sizing, rows_for_bytes(unknown_row_bytes));
    } else {
      auto projected = port_ids.empty()
                         ? std::optional<pipeline::data_size_estimate>{}
                         : pipeline::estimate_port_total_input_bytes(*this, port_ids[0]);
      if (projected) {
        est_source       = "projected";
        bytes_for_sizing = std::max(projected->bytes, observed_bytes);
      } else if (total_rows > 0 && estimated_cardinality > 0) {
        est_source        = "planner";
        auto const scaled = std::ceil(static_cast<long double>(total_sample_bytes) *
                                      estimated_cardinality / total_rows);
        auto const limit  = static_cast<long double>(std::numeric_limits<std::size_t>::max());
        bytes_for_sizing  = std::max(scaled >= limit ? std::numeric_limits<std::size_t>::max()
                                                    : static_cast<std::size_t>(scaled),
                                    observed_bytes);
      }
      rows_for_sizing = std::max(total_rows, rows_for_bytes(bytes_for_sizing));
    }

    auto const available_memory = space->get_available_memory(stream);
    auto const max_partition_bytes =
      _max_partition_bytes_override > 0
        ? _max_partition_bytes_override
        : static_cast<std::size_t>(static_cast<double>(available_memory) *
                                   _max_partition_memory_fraction);
    auto const num_parts =
      choose_sort_range_count(bytes_for_sizing, rows_for_sizing, max_partition_bytes, total_rows);
    SIRIUS_LOG_DEBUG(
      "Sort sample: est_source={}, complete_input={}, sample_rows={}, waiting_rows={}, "
      "waiting_bytes={}, bytes_for_sizing={}, rows_for_sizing={}, available_memory={}, "
      "max_partition_bytes={}, num_partitions={}",
      est_source,
      complete_input,
      total_rows,
      waiting_rows,
      waiting_bytes,
      bytes_for_sizing,
      rows_for_sizing,
      available_memory,
      max_partition_bytes,
      num_parts);

    // 5. Keep regularly spaced key rows from every sorted sample batch.
    auto const desired_key_rows =
      std::min(max_sample_key_rows,
               std::max({min_sample_key_rows,
                         capped_product(key_rows_per_range, num_parts),
                         capped_product(key_rows_per_batch_and_range,
                                        capped_product(valid_batches.size(), num_parts))}));
    auto const sample_stride = std::max<std::size_t>(
      1, total_rows / desired_key_rows + (total_rows % desired_key_rows != 0));
    std::vector<std::unique_ptr<cudf::table>> sampled_tables;
    std::vector<cudf::table_view> sampled_views;
    sampled_tables.reserve(valid_batches.size());
    sampled_views.reserve(valid_batches.size());
    for (auto const& batch : valid_batches) {
      auto const table = get_cudf_table_view(batch);
      std::vector<cudf::column_view> key_columns;
      key_columns.reserve(order_key_idx.size());
      for (int idx : order_key_idx) {
        key_columns.push_back(table.column(idx));
      }
      auto const keys = cudf::table_view(key_columns);
      std::vector<int32_t> indices;
      for (std::size_t row = sample_stride / 2; row < static_cast<std::size_t>(table.num_rows());
           row += sample_stride) {
        indices.push_back(static_cast<int32_t>(row));
      }
      if (indices.empty()) { continue; }
      auto index = index_column(indices, stream, space->get_default_allocator());
      sampled_tables.push_back(cudf::gather(keys,
                                            index->view(),
                                            cudf::out_of_bounds_policy::DONT_CHECK,
                                            stream,
                                            space->get_default_allocator()));
      sampled_views.push_back(sampled_tables.back()->view());
    }
    if (sampled_views.empty()) {
      auto const table = get_cudf_table_view(valid_batches.front());
      std::vector<cudf::column_view> key_columns;
      for (int idx : order_key_idx) {
        key_columns.push_back(table.column(idx));
      }
      sampled_views.emplace_back(key_columns);
    }
    std::unique_ptr<cudf::table> merged_keys;
    if (sampled_views.size() > 1) {
      std::vector<cudf::size_type> key_indices(order_key_idx.size());
      std::iota(key_indices.begin(), key_indices.end(), 0);
      merged_keys = cudf::merge(sampled_views,
                                key_indices,
                                column_order,
                                null_precedence,
                                stream,
                                space->get_default_allocator());
    }
    cudf::table_view const sort_keys_view =
      merged_keys ? merged_keys->view() : sampled_views.front();
    auto const sample_key_rows = static_cast<std::size_t>(sort_keys_view.num_rows());
    auto made =
      _cross_rank_boundaries
        ? _cross_rank_boundaries->make(
            {sort_keys_view, total_sample_bytes, _sort_sample_bytes, num_parts, batch_telemetry()},
            stream)
        : local_boundaries(sort_keys_view, num_parts, space, stream);
    _num_partitions                = made.num_partitions;
    _partition_boundaries          = std::move(made.table);
    auto const repeated_boundaries = count_repeated_boundaries(_partition_boundaries.get(), stream);
    _range_stats                   = {.complete_input        = complete_input,
                                      .sample_batches        = valid_batches.size(),
                                      .sample_rows           = total_rows,
                                      .sample_key_rows       = sample_key_rows,
                                      .sample_stride         = sample_stride,
                                      .sample_bytes          = total_sample_bytes,
                                      .estimated_total_bytes = bytes_for_sizing,
                                      .est_source            = est_source,
                                      .target_range_bytes    = max_partition_bytes,
                                      .num_ranges            = _num_partitions,
                                      .repeated_boundaries   = repeated_boundaries};

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

}  // namespace op
}  // namespace sirius
