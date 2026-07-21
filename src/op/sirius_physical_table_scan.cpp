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

#include "data/data_batch_utils.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "log/logging.hpp"
#include "op/scan/scan_utils.hpp"
#include "sirius_config.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/table/table.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

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
    auto batch = port_ptr->repo->pop_next_data_batch();
    if (!batch) { break; }
    uint64_t batch_bytes = 0;
    {
      auto ro = batch->to_read_only();
      if (ro.get_data()) { batch_bytes = ro.get_data()->get_size_in_bytes(); }
    }
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
  auto& input                  = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& ro_input_batches = input.get_read_only_batches();

  // Passthrough inputs arrive with filter and projection already applied upstream, in
  // batches small enough that concatenation is not needed.
  if (passthrough) {
    return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
  }

  if (ro_input_batches.empty()) { return std::make_unique<pipelineable_operator_data>(); }

  // Build the column_ids index → batch position mapping once.
  // Both filter expression construction and post-filter projection use this.
  auto batch_column_map = build_batch_column_map(projection_ids, column_ids.size());

  // When multiple small batches were coalesced by get_next_task_input_data(),
  // concatenate their GPU tables into one to issue fewer, larger kernel launches.
  std::shared_ptr<cucascade::data_batch> single_batch;
  if (ro_input_batches.size() > 1) {
    std::vector<cudf::table_view> table_views;
    table_views.reserve(ro_input_batches.size());
    cucascade::memory::memory_space* space = nullptr;
    for (const auto& batch : ro_input_batches) {
      if (batch.get_data()) {
        auto& gpu_rep = batch.get_data()->cast<cucascade::gpu_table_representation>();
        table_views.push_back(gpu_rep.get_table_view());
        if (!space) { space = batch.get_memory_space(); }
      }
    }
    if (table_views.size() > 1 && space) {
      auto concatenated = cudf::concatenate(table_views, stream, space->get_default_allocator());
      auto concat_rep   = std::make_unique<cucascade::gpu_table_representation>(
        std::move(concatenated), *space, stream);
      size_t const batch_id = get_next_batch_id();
      single_batch          = cucascade::data_batch::make(
        batch_id,
        std::move(concat_rep),
        telemetry::quent_data_batch_probe::create(batch_telemetry(), batch_id));
    }
  }

  // After concatenation (or if only one batch), work with a single batch.
  // For a concatenated batch (new idle), acquire read lock. For a single input batch, use directly.
  ::cucascade::read_only_data_batch batch_ref =
    single_batch ? single_batch->to_read_only() : ro_input_batches[0];
  if (!batch_ref.get_data()) { return std::make_unique<pipelineable_operator_data>(); }

  // Apply table filters as a GPU expression if present.
  std::shared_ptr<cucascade::data_batch> output_batch;
  std::unique_ptr<sirius::ast::node> local_filter_expr;
  if (table_filters) {
    auto duckdb_filter = convert_table_filters_to_expression(
      *table_filters, column_ids, returned_types, batch_column_map);
    if (duckdb_filter) { local_filter_expr = sirius::ast::from_duckdb(*duckdb_filter); }
  }

  if (local_filter_expr != nullptr) {
    sirius::expression_evaluator evaluator(
      *local_filter_expr, cudf::get_current_device_resource_ref(), stream);
    auto filtered_table = evaluator.select(
      batch_ref.get_data()->cast<cucascade::gpu_table_representation>().get_table_view());
    output_batch = sirius::make_data_batch(
      std::move(filtered_table), *batch_ref.get_memory_space(), stream, batch_telemetry());
  } else {
    output_batch = ::cucascade::data_batch::to_idle(std::move(batch_ref));
  }

  // After filtering, project away filter-only columns if the batch has more
  // columns than the operator's output type list expects.
  std::size_t expected_output_columns = types.size();

  if (expected_output_columns == 0) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(output_batch)});
  }

  // Read batch column count under read-only lock, then release lock before mutation
  std::size_t num_batch_cols = 0;
  {
    auto output_ro = output_batch->to_read_only();
    auto& gpu_rep  = output_ro.get_data()->cast<cucascade::gpu_table_representation>();
    num_batch_cols = static_cast<std::size_t>(gpu_rep.get_table_view().num_columns());
  }  // read lock released here

  if (num_batch_cols > expected_output_columns) {
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
    auto output_ro              = output_batch->to_read_only();
    auto* space                 = output_ro.get_memory_space();
    auto const& gpu_rep         = output_ro.get_data()->cast<cucascade::gpu_table_representation>();
    cudf::table_view input_view = gpu_rep.get_table_view();

    // Select output columns using the batch column map.
    // projection_ids[0..expected_output_columns) are the output columns
    // in the order the downstream operator expects.
    std::vector<cudf::column_view> selected;
    std::vector<cudf::size_type> referenced_indices;
    selected.reserve(expected_output_columns);
    referenced_indices.reserve(expected_output_columns);
    for (std::size_t i = 0; i < expected_output_columns; i++) {
      auto const& batch_idx_opt = batch_column_map[projection_ids[i]];
      if (!batch_idx_opt.has_value() || *batch_idx_opt >= input_view.num_columns()) {
        throw std::runtime_error(
          std::format("TABLE_SCAN projection OOB: projection_ids[{}]={} → batch_idx={} >= "
                      "input_view.num_columns()={}",
                      i,
                      projection_ids[i],
                      batch_idx_opt.has_value() ? std::to_string(*batch_idx_opt) : "(nullopt)",
                      input_view.num_columns()));
      }
      auto const batch_idx = static_cast<cudf::size_type>(*batch_idx_opt);
      selected.push_back(input_view.column(batch_idx));
      referenced_indices.push_back(batch_idx);
    }

    cudf::table_view projected_view(selected);
    std::size_t const referenced_bytes = sirius::estimate_referenced_column_bytes(
      input_view, referenced_indices, output_ro.get_data()->get_size_in_bytes());
    output_batch = sirius::make_data_batch_from_view(
      projected_view, std::move(output_ro), referenced_bytes, *space, stream, batch_telemetry());
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.push_back(std::move(output_batch));
  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius
