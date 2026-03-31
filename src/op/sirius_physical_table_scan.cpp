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
#include "sirius_config.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <algorithm>
#include <format>

namespace sirius {
namespace op {

uint64_t get_chunk_data_byte_size(duckdb::LogicalType type, duckdb::idx_t cardinality)
{
  auto physical_size = duckdb::GetTypeIdSize(type.InternalType());
  return cardinality * physical_size;
}

sirius_physical_table_scan::sirius_physical_table_scan(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::TableFunction function_p,
  duckdb::unique_ptr<duckdb::FunctionData> bind_data_p,
  duckdb::vector<duckdb::LogicalType> returned_types_p,
  duckdb::vector<duckdb::ColumnIndex> column_ids_p,
  duckdb::vector<duckdb::idx_t> projection_ids_p,
  duckdb::vector<std::string> names_p,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters_p,
  duckdb::idx_t estimated_cardinality,
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

/// Build a mapping from column_ids index to batch column position.
///
/// The parquet scan (make_selected_column_indices) produces batch columns in
/// column_ids order, but only for indices present in projection_ids.
/// For example, if column_ids has 5 entries and projection_ids = {1, 3}:
///   batch position 0 → column_ids[1]
///   batch position 1 → column_ids[3]
///
/// Returns a vector of size column_ids_count where:
///   result[i] = batch position of column_ids[i], or idx_t(-1) if not projected.
///
/// When projection_ids is empty, every column_ids entry maps to its own index.
static std::vector<duckdb::idx_t> build_batch_column_map(
  const duckdb::vector<duckdb::idx_t>& projection_ids, duckdb::idx_t column_ids_count)
{
  constexpr auto NOT_PROJECTED = static_cast<duckdb::idx_t>(-1);
  std::vector<duckdb::idx_t> map(column_ids_count, NOT_PROJECTED);

  if (projection_ids.empty()) {
    for (duckdb::idx_t i = 0; i < column_ids_count; i++) {
      map[i] = i;
    }
    return map;
  }

  // Sort projected indices — this matches the iteration order in
  // make_selected_column_indices which walks column_ids[0..N) and
  // includes only indices present in the projected set.
  std::vector<duckdb::idx_t> sorted(projection_ids.begin(), projection_ids.end());
  std::sort(sorted.begin(), sorted.end());

  for (duckdb::idx_t batch_pos = 0; batch_pos < sorted.size(); batch_pos++) {
    if (sorted[batch_pos] < column_ids_count) { map[sorted[batch_pos]] = batch_pos; }
  }
  return map;
}

duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<duckdb::LogicalType>& returned_types,
  const std::vector<duckdb::idx_t>& batch_column_map)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> filter_expressions;

  for (auto& [column_index, filter] : filters.filters) {
    // Skip optional and IS_NOT_NULL filters
    if (filter->filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
        filter->filter_type == duckdb::TableFilterType::IS_NOT_NULL) {
      continue;
    }

    if (column_index >= column_ids.size()) {
      throw std::runtime_error(
        std::format("TABLE_SCAN filter: column_index ({}) >= column_ids.size() ({})",
                    column_index,
                    column_ids.size()));
    }
    auto primary_idx = column_ids[column_index].GetPrimaryIndex();
    if (primary_idx >= returned_types.size()) {
      throw std::runtime_error(
        std::format("TABLE_SCAN filter: primary_idx ({}) >= returned_types.size() ({})",
                    primary_idx,
                    returned_types.size()));
    }
    auto col_type = returned_types[primary_idx];

    SIRIUS_LOG_DEBUG("TABLE_SCAN filter: column_index={}, primary_idx={}, type={}, filter_type={}",
                     column_index,
                     primary_idx,
                     col_type.ToString(),
                     static_cast<int>(filter->filter_type));

    auto batch_column_index = batch_column_map[column_index];
    if (batch_column_index == static_cast<duckdb::idx_t>(-1)) {
      throw std::runtime_error(
        std::format("TABLE_SCAN filter: column_index ({}) not in projected batch", column_index));
    }

    SIRIUS_LOG_DEBUG("TABLE_SCAN filter: batch_column_index={}", batch_column_index);

    auto column_ref =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(col_type, batch_column_index);
    auto expr = filter->ToExpression(*column_ref);
    filter_expressions.push_back(std::move(expr));
  }

  if (filter_expressions.empty()) { return nullptr; }
  if (filter_expressions.size() == 1) { return std::move(filter_expressions[0]); }

  auto conjunction =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
  for (auto& expr : filter_expressions) {
    conjunction->children.push_back(std::move(expr));
  }
  return conjunction;
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
  return std::make_unique<operator_data>(input_batch);
}

std::unique_ptr<operator_data> sirius_physical_table_scan::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_table_scan::execute"};
  const auto& raw_input_batches = input_data.get_data_batches();

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
  if (!batch_ref || !batch_ref->get_data()) { return std::make_unique<operator_data>(); }

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
    if (!output_batch) { return std::make_unique<operator_data>(); }
  } else {
    output_batch = batch_ref;
  }

  // After filtering, project away filter-only columns if the batch has more
  // columns than the operator's output type list expects.
  duckdb::idx_t expected_output_columns = types.size();
  auto& gpu_rep   = output_batch->get_data()->cast<cucascade::gpu_table_representation>();
  auto& out_table = gpu_rep.get_table();

  if (static_cast<duckdb::idx_t>(out_table.num_columns()) > expected_output_columns) {
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

    auto table   = gpu_rep.release_table();
    auto columns = table->release();

    // Select output columns using the batch column map.
    // projection_ids[0..expected_output_columns) are the output columns
    // in the order the downstream operator expects.
    std::vector<std::unique_ptr<cudf::column>> selected;
    selected.reserve(expected_output_columns);
    for (duckdb::idx_t i = 0; i < expected_output_columns; i++) {
      auto batch_idx = batch_column_map[projection_ids[i]];
      if (batch_idx == static_cast<duckdb::idx_t>(-1) || batch_idx >= columns.size()) {
        throw std::runtime_error(
          std::format("TABLE_SCAN projection OOB: projection_ids[{}]={} → batch_idx={} >= "
                      "columns.size()={}",
                      i,
                      projection_ids[i],
                      batch_idx,
                      columns.size()));
      }
      selected.push_back(std::move(columns[batch_idx]));
    }

    auto projected_table = std::make_unique<cudf::table>(std::move(selected));
    auto* space          = output_batch->get_memory_space();
    auto projected_rep =
      std::make_unique<cucascade::gpu_table_representation>(std::move(projected_table), *space);
    output_batch = std::make_shared<cucascade::data_batch>(output_batch->get_batch_id(),
                                                           std::move(projected_rep));
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.push_back(std::move(output_batch));
  return std::make_unique<operator_data>(output_batches);
}

std::string sirius_physical_table_scan::params_to_string() const
{
  std::string result;
  // Show table name from the function
  if (!function.name.empty()) { result += " " + function.name; }
  // Show scanned column names
  if (!names.empty()) {
    result += " [";
    for (size_t i = 0; i < names.size(); i++) {
      if (i > 0) { result += ", "; }
      result += names[i];
    }
    result += "]";
  }
  return result;
}

}  // namespace op
}  // namespace sirius
