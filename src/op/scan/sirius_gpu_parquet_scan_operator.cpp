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

// sirius
#include <data/data_batch_utils.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <log/logging.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>

// cudf
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

// standard library
#include <mutex>
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//
sirius_gpu_parquet_scan_operator::sirius_gpu_parquet_scan_operator(
  duckdb::vector<sirius::logical_type> types, duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_PARQUET_SCAN, std::move(types), estimated_cardinality)
{
}

//===----------------------------------------------------------------------===//
// Metadata handoff (invoked from metadata_scan pipeline)
//===----------------------------------------------------------------------===//
void sirius_gpu_parquet_scan_operator::accumulate_metadata(
  const partitioned_parquet_metadata& metadata)
{
  auto metadata_ptr = std::make_shared<partitioned_parquet_metadata>(metadata);
  std::lock_guard<std::mutex> lock(_metadata_mutex);
  for (std::size_t i = 0; i < metadata.row_group_partitions.size(); ++i) {
    _partition_index.emplace_back(metadata_ptr, i);
  }
}

//===----------------------------------------------------------------------===//
// Scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_gpu_parquet_scan_operator::get_next_task_hint()
{
  std::lock_guard<std::mutex> lock(_metadata_mutex);

  // 1. Work available? Dispatch immediately, even if metadata pipeline
  //    is still producing.
  if (_next_partition_idx < _partition_index.size()) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }

  // 2. Metadata pipeline still running? Wait on it.
  auto it = ports.find("handoff");
  if (it != ports.end() && it->second && it->second->src_pipeline &&
      !it->second->src_pipeline->is_pipeline_finished()) {
    if (auto upstream = it->second->src_pipeline->get_source()) {
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, upstream.get()};
    }
  }

  // 3. No work, pipeline done — finished.
  return std::nullopt;
}

bool sirius_gpu_parquet_scan_operator::all_ports_empty()
{
  std::lock_guard<std::mutex> lock(_metadata_mutex);
  return _next_partition_idx >= _partition_index.size();
}

std::unique_ptr<operator_data> sirius_gpu_parquet_scan_operator::get_next_task_input_data()
{
  std::size_t idx;
  partition_entry entry;
  {
    std::lock_guard<std::mutex> lock(_metadata_mutex);
    if (_next_partition_idx >= _partition_index.size()) { return nullptr; }
    idx   = _next_partition_idx++;
    entry = _partition_index[idx];
  }

  auto meta            = entry.metadata;
  auto const& rg_range = meta->row_group_partitions[entry.partition_idx];

  SIRIUS_LOG_DEBUG(
    "[sirius_gpu_parquet_scan_operator] Creating parquet_scan_data for partition {} "
    "(file_idx={}, {} row groups)",
    idx,
    rg_range.file_idx,
    rg_range.row_group_indices.size());

  return std::make_unique<parquet_scan_data>(meta->file_paths[rg_range.file_idx],
                                             rg_range,
                                             meta->reader_options,
                                             meta->filter_expression,
                                             meta->post_filter_projection_ids,
                                             meta->datasources[rg_range.file_idx],
                                             meta->retranslation_filter,
                                             meta->filter_name_resolver);
}

//===----------------------------------------------------------------------===//
// execute()
//===----------------------------------------------------------------------===//
std::unique_ptr<operator_data> sirius_gpu_parquet_scan_operator::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  auto const* scan_data = dynamic_cast<const parquet_scan_data*>(&input_data);
  if (!scan_data) {
    throw std::runtime_error(
      "[sirius_gpu_parquet_scan_operator] execute() called with unexpected operator_data type; "
      "expected parquet_scan_data.");
  }
  if (!scan_data->gpu_memory_space) {
    throw std::runtime_error(
      "[sirius_gpu_parquet_scan_operator] execute() called with null gpu_memory_space in "
      "input_data.");
  }
  auto datasource = scan_data->datasource;
  auto& mem_space = *scan_data->gpu_memory_space;

  // Build reader options for this partition's row groups.
  auto opts = *scan_data->reader_options;
  opts.set_source(cudf::io::source_info{datasource.get()});
  opts.set_row_groups({scan_data->rg_range.row_group_indices});

  // Multi-GPU correctness: the AST filter currently set on opts (if any) was
  // built by the metadata-scan task on its own CURRENT device. cudf::ast
  // scalars are device-resident, so evaluating that AST on a different GPU
  // silently prunes every row → 0 rows. Re-translate the original duckdb
  // expression here on this scan task's current device + stream so the AST
  // scalars live where the read happens. The freshly-translated AST is held
  // in a local shared_ptr; cudf borrows it by reference until read_parquet
  // returns, which is why this MUST stay in scope through the call.
  std::shared_ptr<gpu_expression_translator::translated_expression> local_ast_filter;
  if (scan_data->retranslation_filter && scan_data->filter_name_resolver) {
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    auto optional_filter = translator.translate_expression_with_names(
      *scan_data->retranslation_filter, scan_data->filter_name_resolver);
    if (optional_filter) {
      stream.synchronize();
      local_ast_filter = std::make_shared<gpu_expression_translator::translated_expression>(
        std::move(*optional_filter));
      opts.set_filter(local_ast_filter->back());
    }
    // If translation now fails (it shouldn't, since metadata-scan succeeded), fall through
    // to the post-read duckdb path below — opts retains the metadata-device AST, but
    // we'll skip pushdown semantics by relying on the post-filter step.
  }

  // Read the parquet data onto the GPU.
  auto [table, metadata] = cudf::io::read_parquet(opts, stream);
  SIRIUS_LOG_DEBUG("[sirius_gpu_parquet_scan_operator] Read {} — {} rows, {} columns",
                   scan_data->file_path,
                   table->num_rows(),
                   table->num_columns());

  // Apply the filter if it was not pushed down into the parquet scan.
  if (std::holds_alternative<std::shared_ptr<duckdb::Expression>>(scan_data->filter_expression)) {
    auto& duckdb_expr = std::get<std::shared_ptr<duckdb::Expression>>(scan_data->filter_expression);
    if (duckdb_expr) {
      sirius::gpu_expression_executor gpu_expression_executor(
        duckdb_expr.get(), cudf::get_current_device_resource_ref(), stream);
      auto input_batch  = sirius::make_data_batch(std::move(table), mem_space, stream);
      auto output_batch = gpu_expression_executor.select(input_batch);
      if (!output_batch) {
        return std::make_unique<pipelineable_operator_data>(
          std::vector<std::shared_ptr<cucascade::data_batch>>());
      }
      table = output_batch->get_data()->cast<cucascade::gpu_table_representation>().release_table();
      SIRIUS_LOG_DEBUG(
        "[sirius_gpu_parquet_scan_operator] Applied duckdb filter expression post parquet scan.");
    }
  }

  // Prune pure filter columns if necessary.
  auto const& post_filter_projection_ids = scan_data->post_filter_projection_ids;
  if (!post_filter_projection_ids.empty()) {
    auto columns = table->release();
    std::vector<std::unique_ptr<cudf::column>> projected_columns;
    projected_columns.reserve(post_filter_projection_ids.size());
    for (auto const col_idx : post_filter_projection_ids) {
      projected_columns.push_back(std::move(columns[col_idx]));
    }
    table = std::make_unique<cudf::table>(std::move(projected_columns));
    SIRIUS_LOG_DEBUG(
      "[sirius_gpu_parquet_scan_operator] Pruned pure filter columns; post-filter projection has "
      "{} columns",
      table->num_columns());
  }

  // Inject hive partition columns, if necessary
  if (_hive_partition_inject_fn) {
    table = _hive_partition_inject_fn(std::move(table), scan_data->file_path, stream);
    SIRIUS_LOG_DEBUG("[sirius_gpu_parquet_scan_operator] Injected hive partition columns.");
  }

  // Wrap the GPU table in operator_data for the downstream pipeline.
  auto batch = sirius::make_data_batch(std::move(table), mem_space, stream);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

}  // namespace sirius::op::scan
