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
    return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
  }

  if (ro_input_batches.empty()) { return std::make_unique<pipelineable_operator_data>(); }

  // Build the column_ids index → batch position mapping once.
  // Both filter expression construction and post-filter projection use this.
  auto batch_column_map = build_batch_column_map(projection_ids, column_ids.size());

  std::shared_ptr<cucascade::data_batch> output_batch = nullptr;
  {
    std::shared_ptr<cucascade::data_batch> single_batch = nullptr;
    if (ro_input_batches.size() > 1) {
      // When multiple small batches were coalesced by get_next_task_input_data(),
      // concatenate their GPU tables into one to issue fewer, larger kernel launches.
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
        single_batch = make_data_batch(std::move(concatenated), *space, stream, batch_telemetry());
      }
    }

    if (single_batch == nullptr) {
      // If a new concatenated batch is not set, deep-clone the first batch
      // to allow independent handling (column dropping in filtering/projection).
      const uint64_t cloned_batch_id = get_next_batch_id();
      single_batch                   = ro_input_batches[0].clone(
        cloned_batch_id,
        stream,
        telemetry::quent_data_batch_probe::create(batch_telemetry(), cloned_batch_id));
    }

    // Apply table filters as a GPU expression if present.
    std::unique_ptr<sirius::ast::node> local_filter_expr;
    if (table_filters) {
      auto duckdb_filter = convert_table_filters_to_expression(
        *table_filters, column_ids, returned_types, batch_column_map);
      if (duckdb_filter) {
        local_filter_expr = sirius::ast::from_duckdb(*duckdb_filter);
        if (local_filter_expr == nullptr) {
          throw duckdb::InvalidInputException(
            "TABLE_SCAN filter: cannot evaluate pushed-down predicate on GPU: %s",
            duckdb_filter->ToString());
        }
      }
    }

    if (local_filter_expr != nullptr) {
      expression_evaluator evaluator(
        *local_filter_expr, cudf::get_current_device_resource_ref(), stream);
      {
        cucascade::read_only_data_batch single_batch_ro = single_batch->to_read_only();
        auto filtered_table                             = evaluator.select(
          single_batch_ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view());
        output_batch = make_data_batch(std::move(filtered_table),
                                       *single_batch_ro.get_memory_space(),
                                       stream,
                                       batch_telemetry());
      }
    } else {
      output_batch = std::move(single_batch);
    }
  }

  // After filtering, project away filter-only columns if the batch has more
  // columns than the operator's output type list expects.
  const std::size_t expected_output_columns = types.size();

  if (expected_output_columns == 0) {
    return std::make_unique<pipelineable_operator_data>(std::vector{std::move(output_batch)});
  }

  // Read batch column count under read-only lock, then release lock before mutation
  std::size_t num_batch_cols = 0;
  {
    auto output_ro = output_batch->to_read_only();
    auto& gpu_rep  = output_ro.get_data()->cast<cucascade::gpu_table_representation>();
    num_batch_cols = static_cast<std::size_t>(gpu_rep.get_table_view().num_columns());
  }  // read lock released here

  if (num_batch_cols > expected_output_columns) {
    // Apply the projection on output_batch

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

    cucascade::memory::memory_space* space = nullptr;
    std::vector<std::unique_ptr<cudf::column>> input_columns{};
    {
      auto mut      = output_batch->to_mutable();
      space         = mut.get_memory_space();
      input_columns = mut.get_data()
                        ->cast<cucascade::gpu_table_representation>()
                        .release_table(stream)
                        ->release();
    }
    output_batch.reset();  // the underlying table was released, the batch is now invalid.

    // Select output columns using the batch column map.
    // projection_ids[0..expected_output_columns) are the output columns
    // in the order the downstream operator expects.
    std::vector<std::unique_ptr<cudf::column>> output_columns;
    output_columns.reserve(expected_output_columns);
    for (std::size_t idx = 0; idx < expected_output_columns; idx++) {
      const std::optional<std::size_t> maybe_batch_idx = batch_column_map[projection_ids[idx]];
      if (not maybe_batch_idx.has_value() or *maybe_batch_idx >= num_batch_cols) {
        throw std::runtime_error(
          std::format("TABLE_SCAN projection OOB: projection_ids[{}]={} → batch_idx={} >= "
                      "num_batch_cols={}",
                      idx,
                      projection_ids[idx],
                      maybe_batch_idx.has_value() ? std::to_string(*maybe_batch_idx) : "(nullopt)",
                      num_batch_cols));
      }
      auto const batch_idx = static_cast<cudf::size_type>(*maybe_batch_idx);

      assert(input_columns[batch_idx] && "scan projection_ids are distinct");
      // distinct projection_ids allow directly moving the columns
      output_columns.push_back(std::move(input_columns[batch_idx]));
    }

    // reassign the output batch with the projected cols.
    output_batch = make_data_batch({std::move(output_columns)}, *space, stream, batch_telemetry());
  }

  return std::make_unique<pipelineable_operator_data>(std::vector{output_batch});
}

}  // namespace op
}  // namespace sirius
