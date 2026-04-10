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
#include <log/logging.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>

// cudf
#include <cudf/io/parquet.hpp>

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
  duckdb::vector<duckdb::LogicalType> types, duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_PARQUET_SCAN, std::move(types), estimated_cardinality)
{
}

//===----------------------------------------------------------------------===//
// Sink interface (pipeline 1)
//===----------------------------------------------------------------------===//
void sirius_gpu_parquet_scan_operator::sink(const operator_data& input_data,
                                            rmm::cuda_stream_view /*stream*/)
{
  auto const* metadata = dynamic_cast<const partitioned_parquet_metadata*>(&input_data);
  if (!metadata) {
    SIRIUS_LOG_WARN(
      "[sirius_gpu_parquet_scan_operator] sink() received unexpected operator_data type "
      "(typeid: {}); ignoring.",
      typeid(input_data).name());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(_metadata_mutex);
    _accumulated_metadata.push_back(*metadata);  // Copy unavoidable but relatively cheap
  }

  SIRIUS_LOG_DEBUG(
    "[sirius_gpu_parquet_scan_operator] Received partitioned_parquet_metadata with {} partitions",
    metadata->row_group_partitions.size());
}

//===----------------------------------------------------------------------===//
// Pipeline 1 → Pipeline 2 transition
//===----------------------------------------------------------------------===//
void sirius_gpu_parquet_scan_operator::finalize_metadata()
{
  std::lock_guard<std::mutex> lock(_metadata_mutex);

  _partition_index.clear();
  for (auto const& meta : _accumulated_metadata) {
    for (std::size_t i = 0; i < meta.row_group_partitions.size(); ++i) {
      _partition_index.push_back(partition_entry{&meta, i});
    }
  }

  // Release store: all writes to _partition_index happen-before this store.
  _finalized.store(true, std::memory_order_release);

  SIRIUS_LOG_DEBUG(
    "[sirius_gpu_parquet_scan_operator] Finalized {} partitions from {} metadata objects",
    _partition_index.size(),
    _accumulated_metadata.size());
}

//===----------------------------------------------------------------------===//
// Source interface (pipeline 2)
//===----------------------------------------------------------------------===//
std::size_t sirius_gpu_parquet_scan_operator::get_total_partitions() const
{
  if (_finalized.load(std::memory_order_acquire)) { return _partition_index.size(); }
  std::lock_guard<std::mutex> lock(_metadata_mutex);
  std::size_t total = 0;
  for (auto const& meta : _accumulated_metadata) {
    total += meta.row_group_partitions.size();
  }
  return total;
}

std::optional<task_creation_hint> sirius_gpu_parquet_scan_operator::get_next_task_hint()
{
  if (!_finalized.load(std::memory_order_acquire)) { return std::nullopt; }
  if (_next_partition_idx.load(std::memory_order_relaxed) < _partition_index.size()) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  return std::nullopt;
}

bool sirius_gpu_parquet_scan_operator::all_ports_empty()
{
  if (!_finalized.load(std::memory_order_acquire)) { return false; }
  return _next_partition_idx.load(std::memory_order_relaxed) >= _partition_index.size();
}

std::unique_ptr<operator_data> sirius_gpu_parquet_scan_operator::get_next_task_input_data()
{
  if (!_finalized.load(std::memory_order_acquire)) { return nullptr; }

  auto const idx = _next_partition_idx.fetch_add(1, std::memory_order_relaxed);
  if (idx >= _partition_index.size()) { return nullptr; }

  auto const& entry    = _partition_index[idx];
  auto const* meta     = entry.metadata;
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
                                             meta->datasources[rg_range.file_idx]);
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
      duckdb::sirius::GpuExpressionExecutor gpu_expression_executor(*duckdb_expr);
      auto input_batch  = sirius::make_data_batch(std::move(table), mem_space);
      auto output_batch = gpu_expression_executor.select(input_batch, stream);
      if (!output_batch) { return std::make_unique<operator_data>(); }
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

  // Wrap the GPU table in operator_data for the downstream pipeline.
  auto batch = sirius::make_data_batch(std::move(table), mem_space);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

}  // namespace sirius::op::scan
