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
#include <op/scan/parquet_scan_info.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <scan_manager/split_connector.hpp>

// cudf
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

// standard library
#include <stdexcept>
#include <utility>
#include <variant>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//
sirius_gpu_parquet_scan_operator::sirius_gpu_parquet_scan_operator(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  std::unique_ptr<parquet_scan_info> scan_info)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_PARQUET_SCAN, std::move(types), estimated_cardinality),
    _split_connector(std::make_unique<scan_manager::split_connector>()),
    _scan_info(std::move(scan_info))
{
  _split_connector->close();
}

sirius_gpu_parquet_scan_operator::~sirius_gpu_parquet_scan_operator() = default;

//===----------------------------------------------------------------------===//
// Friend access — wired by sirius_scan_manager during prepare_for_query.
//===----------------------------------------------------------------------===//
std::unique_ptr<parquet_scan_info> sirius_gpu_parquet_scan_operator::take_scan_info()
{
  return std::move(_scan_info);
}

void sirius_gpu_parquet_scan_operator::set_split_connector(
  std::unique_ptr<scan_manager::split_connector> connector)
{
  _split_connector = std::move(connector);
}

//===----------------------------------------------------------------------===//
// Scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_gpu_parquet_scan_operator::get_next_task_hint()
{
  if (_split_connector->is_closed()) { return std::nullopt; }
  // Returns READY even when the split queue is empty but not yet closed. The dispatched worker
  // then parks inside split_connector::get_next_split's cv until the scan_manager pushes a split
  // or closes the connector. This trades one parked worker per active scan for not needing a
  // scheduler-visible wake-up signal from the scan_manager — the task_creator schedules the scan
  // op once and the worker self-perpetuates inside task_creator's "loop until all_ports_empty"
  // branch (see task_creator::manager_loop default else-branch).
  //
  // Lifecycle requirement: any error/drain path that joins task_creator's worker pool MUST first
  // close the operator's split_connector (e.g. via sirius_scan_manager::close_all_connectors());
  // otherwise the parked worker hangs forever and bounded_pool::wait_all deadlocks. Normal
  // completion is fine — the split_provider closes the connector when it has no more splits.
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_gpu_parquet_scan_operator::all_ports_empty()
{
  return _split_connector->is_closed() && !_split_connector->has_more_splits();
}

std::unique_ptr<operator_data> sirius_gpu_parquet_scan_operator::get_next_task_input_data()
{
  auto next = _split_connector->get_next_split();
  if (!next.has_value()) { return nullptr; }
  return std::move(*next);
}

//===----------------------------------------------------------------------===//
// read_table_from_metadata()
//===----------------------------------------------------------------------===//
std::unique_ptr<cudf::table> sirius_gpu_parquet_scan_operator::read_table_from_metadata(
  const parquet_scan_data& scan_data, rmm::cuda_stream_view stream)
{
  auto datasource        = scan_data.datasource;
  auto filter_expression = scan_data.filter_expression;

  // Build reader options for this partition's row groups.
  auto opts = *scan_data.reader_options;
  opts.set_source(cudf::io::source_info{datasource.get()});
  opts.set_row_groups({scan_data.rg_range.row_group_indices});

  // Try to translate the filter to a *stream-local* cudf AST for filter pushdown. If not
  // successful, fall back to post-read DuckDB-expression evaluation.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (filter_expression) {
    auto name_resolver = [plan = scan_data.plan](duckdb::idx_t ref_index) -> std::string {
      return plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression = translator.translate_expression_with_names(*filter_expression, name_resolver);
    if (ast_expression) {
      opts.set_filter(ast_expression->back());
      SIRIUS_LOG_DEBUG(
        "[sirius_gpu_parquet_scan_operator] Translated filter expression for parquet reader filter "
        "pushdown.");
    } else {
      SIRIUS_LOG_DEBUG(
        "[sirius_gpu_parquet_scan_operator] AST translation failed for parquet reader filter "
        "pushdown.");
    }
  }

  auto [table, metadata] = cudf::io::read_parquet(opts, stream);

  SIRIUS_LOG_DEBUG("[sirius_gpu_parquet_scan_operator] Read {} — {} rows, {} columns",
                   scan_data.file_path,
                   table->num_rows(),
                   table->num_columns());

  // Apply the filter if it was not pushed down into the parquet scan.
  if (filter_expression && !ast_expression) {
    sirius::gpu_expression_executor gpu_expression_executor(
      filter_expression.get(), cudf::get_current_device_resource_ref(), stream);
    // Hold the source table in a separate variable so its destruction (and the
    // stream-ordered dealloc of its buffers) is sequenced after select()'s kernels
    // have been queued. Assigning select()'s result directly back to `table` would
    // destruct the source synchronously before the queued kernels read from it.
    auto input_table = std::move(table);
    table            = gpu_expression_executor.select(input_table->view());
    SIRIUS_LOG_DEBUG(
      "[sirius_gpu_parquet_scan_operator] Applied duckdb filter expression post parquet scan.");
  }

  // Reshape the reader's output to the scan_plan's D-order layout: reorder data
  // columns, drop pure-filter columns, inject hive-partition columns. No-op when
  // the scan is a trivial identity (no partitions, 1:1 data layout).
  if (_partition_inject_fn) {
    table = _partition_inject_fn(std::move(table), scan_data.file_path, stream);
    SIRIUS_LOG_DEBUG("[sirius_gpu_parquet_scan_operator] Applied partition_inject_fn.");
  }

  return std::move(table);
}

//===----------------------------------------------------------------------===//
// execute()
//===----------------------------------------------------------------------===//
std::unique_ptr<operator_data> sirius_gpu_parquet_scan_operator::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  std::unique_ptr<cudf::table> table;
  cucascade::memory::memory_space* mem_space = nullptr;

  if (auto const* scan_data = dynamic_cast<const parquet_scan_data*>(&input_data)) {
    if (!scan_data->gpu_memory_space) {
      throw std::runtime_error(
        "[sirius_gpu_parquet_scan_operator] execute() called with null gpu_memory_space in "
        "input_data.");
    }
    table     = read_table_from_metadata(*scan_data, stream);
    mem_space = scan_data->gpu_memory_space;
  } else if (auto const* cached = dynamic_cast<const scan_cached_operator_data*>(&input_data)) {
    auto& gpu_rep    = cached->batch->get_data()->cast<cucascade::gpu_table_representation>();
    auto cached_view = gpu_rep.get_table_view();

    // Apply the filter (only the duckdb::Expression variant is currently produced by
    // sirius_scan_manager; the by-name translated_expression form is reserved for parquet
    // reader pushdown and not applicable to an already-materialized cached table).
    auto const* duckdb_filter =
      std::get_if<std::shared_ptr<duckdb::Expression>>(&cached->filter_expression);
    bool const has_filter     = (duckdb_filter != nullptr) && static_cast<bool>(*duckdb_filter);
    bool const has_projection = !cached->post_filter_projection_ids.empty();

    if (!has_filter && !has_projection) {
      // Fast path: forward the cached batch unchanged. Avoids any allocation or copy
      // since the cached batch's gpu_table_representation already views the pinned
      // columns and co-owns them via the shared_ptr<column> chunks.
      std::vector<std::shared_ptr<cucascade::data_batch>> batches;
      batches.push_back(cached->batch);
      return std::make_unique<pipelineable_operator_data>(std::move(batches));
    }

    if (has_filter) {
      sirius::gpu_expression_executor gpu_expression_executor(
        duckdb_filter->get(), cudf::get_current_device_resource_ref(), stream);
      table = gpu_expression_executor.select(cached_view);
      SIRIUS_LOG_DEBUG(
        "[sirius_gpu_parquet_scan_operator] Applied duckdb filter expression on cached batch.");
    }

    if (has_projection) {
      auto columns = has_filter ? table->release() : std::vector<std::unique_ptr<cudf::column>>{};
      std::vector<std::unique_ptr<cudf::column>> selected;
      selected.reserve(cached->post_filter_projection_ids.size());
      if (has_filter) {
        for (auto const idx : cached->post_filter_projection_ids) {
          if (idx >= columns.size()) {
            throw std::runtime_error(
              "[sirius_gpu_parquet_scan_operator] post_filter_projection_ids out of range");
          }
          selected.push_back(std::move(columns[idx]));
        }
      } else {
        // No filter: deep-copy the projected columns out of the cached view so the
        // emitted batch owns its memory independently of the pinned cache entry.
        for (auto const idx : cached->post_filter_projection_ids) {
          if (idx >= static_cast<std::size_t>(cached_view.num_columns())) {
            throw std::runtime_error(
              "[sirius_gpu_parquet_scan_operator] post_filter_projection_ids out of range");
          }
          selected.push_back(std::make_unique<cudf::column>(
            cached_view.column(idx), stream, cudf::get_current_device_resource_ref()));
        }
      }
      table = std::make_unique<cudf::table>(std::move(selected));
    }

    mem_space = cached->batch->get_memory_space();
  } else {
    throw std::runtime_error(
      "[sirius_gpu_parquet_scan_operator] execute() called with unexpected operator_data type; "
      "expected parquet_scan_data or scan_cached_operator_data.");
  }

  // Wrap the GPU table in operator_data for the downstream pipeline.
  auto batch = sirius::make_data_batch(std::move(table), *mem_space);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

}  // namespace sirius::op::scan
