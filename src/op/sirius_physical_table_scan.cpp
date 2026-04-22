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

#include "op/sirius_physical_table_scan.hpp"

#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"
#include "op/scan/scan_utils.hpp"
#include "sirius_config.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <format>

namespace sirius {
namespace op {

uint64_t get_chunk_data_byte_size(sirius::logical_type type, std::size_t cardinality)
{
  return cardinality * type.fixed_width_byte_size();
}

sirius_physical_table_scan::sirius_physical_table_scan(
  duckdb::vector<sirius::logical_type> types,
  duckdb::TableFunction function_p,
  duckdb::unique_ptr<duckdb::FunctionData> bind_data_p,
  duckdb::vector<sirius::logical_type> returned_types_p,
  duckdb::vector<duckdb::ColumnIndex> column_ids_p,
  duckdb::vector<std::size_t> projection_ids_p,
  duckdb::vector<std::string> names_p,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters_p,
  std::size_t estimated_cardinality,
  duckdb::ExtraOperatorInfo extra_info,
  duckdb::vector<duckdb::Value> parameters_p,
  duckdb::virtual_column_map_t virtual_columns_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::TABLE_SCAN, std::move(types), estimated_cardinality),
    function(std::move(function_p)),
    bind_data(std::move(bind_data_p)),
    returned_types(std::move(returned_types_p)),
    column_ids(std::move(column_ids_p)),
    projection_ids(std::move(projection_ids_p)),
    names(std::move(names_p)),
    table_filters(std::move(table_filters_p)),
    extra_info(std::move(extra_info)),
    parameters(std::move(parameters_p)),
    virtual_columns(std::move(virtual_columns_p))
{
}

std::unique_ptr<operator_data> sirius_physical_table_scan::get_next_task_input_data()
{
  // Coalesce multiple small scan batches into a single task to reduce per-task
  // overhead and improve GPU utilization. The batches are concatenated into one
  // table in execute().
  D_ASSERT(ports.size() == 1);
  auto& [port_name, port_ptr] = *ports.begin();

  std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
  uint64_t accumulated_bytes = 0;
  size_t batch_count         = 0;
  // Cap per-task batch count to avoid grabbing too many compressed batches
  // whose representation bytes understate their actual GPU processing cost.
  constexpr size_t max_batches_per_task = 32;
  while (true) {
    auto batch = port_ptr->repo->pop_data_batch(::cucascade::batch_state::task_created);
    if (!batch) { break; }
    uint64_t batch_bytes = 0;
    if (batch->get_data()) { batch_bytes = batch->get_data()->get_size_in_bytes(); }
    accumulated_bytes += batch_bytes;
    input_batch.push_back(std::move(batch));
    ++batch_count;
    if (accumulated_bytes >= config::DEFAULT_SCAN_TASK_BATCH_SIZE ||
        batch_count >= max_batches_per_task) {
      break;
    }
  }
  if (input_batch.empty()) { return nullptr; }
  return std::make_unique<pipelineable_operator_data>(input_batch);
}

std::unique_ptr<operator_data> sirius_physical_table_scan::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_table_scan::execute"};
  auto& input                   = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& raw_input_batches = input.get_data_batches();

  // For parquet scan pipelines, filter and projection are already applied in
  // parquet_scan_task and the host_parquet_representation converters.
  // Also, only parquet file tails are small due to the partitioning logic, so batch concatenation
  // is not needed.
  if (passthrough) { return std::make_unique<pipelineable_operator_data>(raw_input_batches); }

  // Build the column_ids index → batch position mapping once.
  // Both filter expression construction and post-filter projection use this.
  auto batch_column_map = build_batch_column_map(projection_ids, column_ids.size());

  // When multiple small batches were coalesced by get_next_task_input_data(),
  // concatenate their GPU tables into one to issue fewer, larger kernel launches.
  std::shared_ptr<cucascade::data_batch> single_batch;
  if (raw_input_batches.size() > 1) {
    std::vector<cudf::table_view> table_views;
    table_views.reserve(raw_input_batches.size());
    cucascade::memory::memory_space* space = nullptr;
    for (const auto& batch : raw_input_batches) {
      if (batch && batch->get_data()) {
        auto& gpu_rep = batch->get_data()->cast<cucascade::gpu_table_representation>();
        table_views.push_back(gpu_rep.get_table().view());
        if (!space) { space = batch->get_memory_space(); }
      }
    }
    if (table_views.size() > 1 && space) {
      auto concatenated = cudf::concatenate(table_views, stream, space->get_default_allocator());
      auto concat_rep =
        std::make_unique<cucascade::gpu_table_representation>(std::move(concatenated), *space);
      single_batch = std::make_shared<cucascade::data_batch>(0, std::move(concat_rep));
    }
  }

  // After concatenation (or if only one batch), work with a single batch.
  const auto& batch_ref =
    single_batch ? single_batch : (!raw_input_batches.empty() ? raw_input_batches[0] : nullptr);
  if (!batch_ref || !batch_ref->get_data()) {
    return std::make_unique<pipelineable_operator_data>();
  }

  // Apply table filters as a GPU expression if present.
  std::shared_ptr<cucascade::data_batch> output_batch;
  duckdb::unique_ptr<duckdb::Expression> filter_expr;
  if (table_filters) {
    filter_expr = convert_table_filters_to_expression(
      *table_filters, column_ids, returned_types, batch_column_map);
  }

  if (filter_expr != nullptr) {
    duckdb::sirius::GpuExpressionExecutor gpu_expression_executor(*filter_expr);
    output_batch = gpu_expression_executor.select(batch_ref, stream);
    if (!output_batch) { return std::make_unique<pipelineable_operator_data>(); }
  } else {
    output_batch = batch_ref;
  }

  // After filtering, normalize the batch to the logical scan output shape.
  //
  // Normal DuckDB scan tasks already emit columns in projected scan order, so
  // this is often a no-op. GPU-resident cached tables, however, arrive in raw
  // table order; when that differs from projection_ids we must reorder even if
  // the column count already matches the operator output.
  std::size_t expected_output_columns = types.size();
  auto& gpu_rep   = output_batch->get_data()->cast<cucascade::gpu_table_representation>();
  auto& out_table = gpu_rep.get_table();

  if (expected_output_columns == 0) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(output_batch)});
  }

  bool needs_projection =
    static_cast<std::size_t>(out_table.num_columns()) != expected_output_columns;
  std::vector<std::size_t> selected_indices;
  selected_indices.reserve(expected_output_columns);

  if (!projection_ids.empty()) {
    SIRIUS_LOG_DEBUG(
      "TABLE_SCAN projection: expected_output_columns={}, projection_ids.size()={}, "
      "column_ids.size()={}",
      expected_output_columns,
      projection_ids.size(),
      column_ids.size());

    if (expected_output_columns > projection_ids.size()) {
      throw std::runtime_error(
        std::format("TABLE_SCAN projection error: expected_output_columns ({}) > "
                    "projection_ids.size() ({})",
                    expected_output_columns,
                    projection_ids.size()));
    }
  } else if (needs_projection) {
    throw std::runtime_error(
      std::format("TABLE_SCAN projection error: expected {} columns but batch has {} and no "
                  "projection_ids are available",
                  expected_output_columns,
                  out_table.num_columns()));
  }

  for (std::size_t i = 0; i < expected_output_columns; i++) {
    std::size_t batch_idx = i;
    if (!projection_ids.empty()) {
      batch_idx = batch_column_map[projection_ids[i]];
    }
    if (batch_idx == static_cast<std::size_t>(-1) ||
        batch_idx >= static_cast<std::size_t>(out_table.num_columns())) {
      throw std::runtime_error(
        std::format("TABLE_SCAN projection OOB: output_idx={} batch_idx={} >= "
                    "view.num_columns()={}",
                    i,
                    batch_idx,
                    out_table.num_columns()));
    }
    selected_indices.push_back(batch_idx);
    if (batch_idx != i) {
      needs_projection = true;
    }
  }

  if (needs_projection) {
    auto* space = output_batch->get_memory_space();
    if (!space) {
      throw std::runtime_error("TABLE_SCAN projection error: output batch has no memory space");
    }

    // Deep-copy the projected columns into a new table. This avoids moving
    // ownership out of the scan-produced table, which has been unstable for
    // some reader outputs during downstream materialization/destruction.
    std::vector<cudf::column_view> selected;
    selected.reserve(expected_output_columns);
    auto view = out_table.view();
    for (auto batch_idx : selected_indices) {
      selected.push_back(view.column(batch_idx));
    }

    auto projected_table = std::make_unique<cudf::table>(
      cudf::table_view(selected), stream, space->get_default_allocator());
    auto projected_rep =
      std::make_unique<cucascade::gpu_table_representation>(std::move(projected_table), *space);
    output_batch = std::make_shared<cucascade::data_batch>(output_batch->get_batch_id(),
                                                           std::move(projected_rep));
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.push_back(std::move(output_batch));
  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius
