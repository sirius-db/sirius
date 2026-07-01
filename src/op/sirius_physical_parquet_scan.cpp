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

#include "op/sirius_physical_parquet_scan.hpp"

#include "expression/ast/from_duckdb.hpp"
#include "expression_evaluator/gpu_expression_translator_internal.hpp"
#include "log/logging.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/sirius_physical_table_scan.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <duckdb/common/multi_file/multi_file_states.hpp>

#include <unordered_set>

namespace sirius {
namespace op {

// Helper function to deep copy ExtraOperatorInfo
duckdb::ExtraOperatorInfo copy_extra_info_parquet_scan(const duckdb::ExtraOperatorInfo& src)
{
  duckdb::ExtraOperatorInfo copy;
  copy.file_filters = src.file_filters;
  if (src.total_files.IsValid()) { copy.total_files = src.total_files.GetIndex(); }
  if (src.filtered_files.IsValid()) { copy.filtered_files = src.filtered_files.GetIndex(); }
  if (src.sample_options) { copy.sample_options = src.sample_options->Copy(); }
  return copy;
}

sirius_physical_parquet_scan::sirius_physical_parquet_scan(sirius_physical_table_scan* table_scan,
                                                           std::vector<int> gpu_device_ids)
  : sirius_physical_parquet_scan(
      table_scan->types,
      table_scan->function,
      table_scan->bind_data ? table_scan->bind_data->Copy() : nullptr,
      table_scan->returned_types,
      table_scan->column_ids,
      table_scan->projection_ids,
      table_scan->names,
      table_scan->table_filters ? table_scan->table_filters->Copy() : nullptr,
      table_scan->estimated_cardinality,
      copy_extra_info_parquet_scan(table_scan->extra_info),
      table_scan->parameters,
      table_scan->virtual_columns,
      table_scan,  // Pass pointer to the table scan for filter pushdown
      std::move(gpu_device_ids))
{
}

sirius_physical_parquet_scan::sirius_physical_parquet_scan(
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
  duckdb::virtual_column_map_t virtual_columns_p,
  sirius_physical_table_scan* table_scan,
  std::vector<int> gpu_device_ids)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PARQUET_SCAN, std::move(types), estimated_cardinality),
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
  if (table_filters && !table_filters->filters.empty()) {
    auto name_resolver = [&](duckdb::idx_t ref_index) -> std::string {
      auto const primary_idx = column_ids[ref_index].GetPrimaryIndex();
      return names[primary_idx];
    };
    auto batch_column_map = build_batch_column_map(projection_ids, column_ids.size());
    // Drop filters on hive-partition columns — they aren't in the parquet file, so pushing
    // them into the reader crashes libcudf. DuckDB already prunes partitions at the file-list
    // level when hive_partitioning is enabled. Only native read_parquet / parquet_scan binds
    // produce MultiFileBindData; sirius_read_parquet binds a SiriusReadParquetBindData (single
    // S3 URI, no hive partitioning), so dynamic_cast lets that path fall through with an
    // empty partition index set instead of reinterpret-casting through the wrong type.
    std::unordered_set<std::size_t> hive_partition_primary_indices;
    if (auto const* multi_file_bind =
          dynamic_cast<duckdb::MultiFileBindData const*>(bind_data.get())) {
      for (auto const& hpi : multi_file_bind->reader_bind.hive_partitioning_indexes) {
        hive_partition_primary_indices.insert(hpi.index);
      }
    }
    auto duckdb_expression = convert_table_filters_to_expression(
      *table_filters, column_ids, returned_types, batch_column_map, hive_partition_primary_indices);
    if (duckdb_expression) {
      // Build one translated filter tree PER GPU. Each tree's cudf::scalar
      // device buffers live on the tree's own device, so a task dispatched to
      // device N evaluates the filter's AST entirely against device-N memory.
      // Without this, a single translation at planner-time would bind scalars
      // to the planner's current device (typically GPU 0), and tasks on other
      // GPUs would hit cudaErrorInvalidValue / cudaErrorIllegalAddress when
      // cudf::io::read_parquet evaluates the AST against per-rowgroup stats.
      std::vector<int> device_ids_for_translation = gpu_device_ids;
      if (device_ids_for_translation.empty()) {
        // Fallback: translate for the current device only when no explicit
        // device list is provided (e.g. operator constructed outside the
        // multi-GPU engine path). Preserves single-GPU semantics.
        int current_device = 0;
        (void)::cudaGetDevice(&current_device);
        device_ids_for_translation.push_back(current_device);
      }

      bool all_translations_ok = true;
      for (int device_id : device_ids_for_translation) {
        rmm::cuda_set_device_raii device_raii{rmm::cuda_device_id{device_id}};
        // Translate on an explicit per-device stream, then synchronize it
        // before handing the translated AST off. The translator materializes
        // cudf::string_scalar / numeric_scalar device buffers via
        // cudaMallocAsync on whatever stream it's given; if it uses the
        // default stream, later kernels running on *other* streams (e.g.
        // filter_row_groups_with_stats using a throwaway `planning_stream` in
        // parquet_scan_task.cpp) can launch before the alloc event has fired,
        // yielding a use-before-alloc stream-ordered race (surfaces as
        // cudaErrorIllegalAddress inside cudf::detail::compute_column_kernel).
        // Confirmed by compute-sanitizer --track-stream-ordered-races=all.
        rmm::cuda_stream translation_stream;
        gpu_expression_translator translator(
          translation_stream.view(),
          rmm::mr::get_per_device_resource_ref(rmm::cuda_device_id{device_id}));
        auto translated = sirius::op::translate_duckdb_expression_with_names(
          translator, *duckdb_expression, name_resolver);
        // Synchronize BEFORE storing into translated_filter_by_device so any
        // future reader — on any stream, on any device after peer-access — is
        // guaranteed to observe the scalar allocations as already completed.
        translation_stream.synchronize();
        if (!translated) {
          all_translations_ok = false;
          break;
        }
        // Transfer ownership of the stream into the translated_expression so
        // it lives exactly as long as the cudf::scalar device buffers that
        // were allocated on it.  Without this, the stream is destroyed here
        // (end of for-loop body) while the scalars still reference its handle
        // for future cudaFreeAsync deallocation — causing a use-after-destroy
        // SIGSEGV when the translated_expression is eventually torn down.
        translated->owned_stream = std::move(translation_stream);
        translated_filter_by_device.emplace(device_id, std::move(*translated));
      }

      if (!all_translations_ok) {
        translated_filter_by_device.clear();
        SIRIUS_LOG_INFO(
          "[sirius_physical_parquet_scan] Failed to translate filter expression for pushdown. "
          "Filter will be applied in the table scan operator.");
      } else {
        // Instruct the sirius_physical_table_scan operator to bypass execute()
        if (table_scan) { table_scan->passthrough = true; }
      }
      // Translate the duckdb_expression into the table scan's Sirius AST filter
      if (table_scan) { table_scan->filter_expr = sirius::ast::from_duckdb(*duckdb_expression); }
    }
  }
}

std::size_t sirius_physical_parquet_scan::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  return stats.bytes * 8;
}

}  // namespace op
}  // namespace sirius
