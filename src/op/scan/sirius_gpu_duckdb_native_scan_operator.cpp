/*
 * Copyright 2026, Sirius Contributors.
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

#include <cudf/utilities/default_stream.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <expression/expression.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <log/logging.hpp>
#include <op/scan/duckdb_native_scan_task.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_duckdb_native_scan_operator.hpp>
#include <scan_manager/duckdb_native_split_provider.hpp>
#include <scan_manager/split_connector.hpp>

#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

sirius_gpu_duckdb_native_scan_operator::sirius_gpu_duckdb_native_scan_operator(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  std::unique_ptr<duckdb_native_scan_info> scan_info)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_DUCKDB_NATIVE_SCAN, std::move(types), estimated_cardinality),
    _split_connector(std::make_unique<scan_manager::split_connector>()),
    _scan_info(std::move(scan_info))
{
  _split_connector->close();
}

sirius_gpu_duckdb_native_scan_operator::~sirius_gpu_duckdb_native_scan_operator() = default;

std::unique_ptr<duckdb_native_scan_info> sirius_gpu_duckdb_native_scan_operator::take_scan_info()
{
  return std::move(_scan_info);
}

void sirius_gpu_duckdb_native_scan_operator::set_split_connector(
  std::unique_ptr<scan_manager::split_connector> connector)
{
  _split_connector = std::move(connector);
}

std::optional<task_creation_hint> sirius_gpu_duckdb_native_scan_operator::get_next_task_hint()
{
  if (_split_connector->is_closed()) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_gpu_duckdb_native_scan_operator::all_ports_empty()
{
  return _split_connector->is_closed();
}

std::unique_ptr<operator_data> sirius_gpu_duckdb_native_scan_operator::get_next_task_input_data()
{
  auto next = _split_connector->get_next_split();
  if (!next.has_value()) { return nullptr; }
  return std::move(*next);
}

std::unique_ptr<operator_data> sirius_gpu_duckdb_native_scan_operator::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  auto const* split =
    dynamic_cast<const scan_manager::duckdb_native_split_provider::split_payload*>(&input_data);
  if (split == nullptr) {
    throw std::runtime_error(
      "[sirius_gpu_duckdb_native_scan_operator] execute() called with unexpected "
      "operator_data type; expected duckdb_native_split_provider::split_payload.");
  }
  if (!split->scan_info) {
    throw std::runtime_error(
      "[sirius_gpu_duckdb_native_scan_operator] split payload missing scan_info");
  }

  auto* mem_space = pick_gpu_memory_space_for_duckdb_native_scan(*split->scan_info);
  if (mem_space == nullptr) {
    throw std::runtime_error(
      "[sirius_gpu_duckdb_native_scan_operator] no GPU memory space available");
  }

  auto table = decode_duckdb_native_split(*split, *mem_space, stream);

  SIRIUS_LOG_DEBUG(
    "[sirius_gpu_duckdb_native_scan_operator] decoded split: row_groups={} rows={} cols={}",
    split->row_groups.size(),
    table->num_rows(),
    table->num_columns());

  // Mirror sirius_physical_table_scan::execute(): apply table_filters via gpu_expression_executor,
  // then project filter-only columns away. The decode emits columns in `source_ids` order
  // (= projection_ids unsorted), so we build a custom column_index → batch_position map
  // rather than using build_batch_column_map (which sorts by column index).
  auto const& info = *split->scan_info;
  auto mr_ref      = mem_space->get_default_allocator();

  auto const& source_ids = !info.projection_ids.empty() ? info.projection_ids : [&] {
    duckdb::vector<duckdb::idx_t> all;
    all.reserve(info.column_ids.size());
    for (duckdb::idx_t i = 0; i < info.column_ids.size(); ++i) {
      all.push_back(i);
    }
    return all;
  }();

  std::vector<std::optional<std::size_t>> emission_order_map(info.column_ids.size());
  for (std::size_t k = 0; k < source_ids.size(); ++k) {
    emission_order_map[source_ids[k]] = k;
  }

  if (info.table_filters && !info.table_filters->filters.empty()) {
    auto filter_expr_duckdb = convert_table_filters_to_expression(
      *info.table_filters, info.column_ids, info.returned_types, emission_order_map);
    if (filter_expr_duckdb) {
      auto filter_expr = sirius::wrap(std::move(filter_expr_duckdb));
      sirius::gpu_expression_executor exec(filter_expr, mr_ref, stream);
      table = exec.select(table->view());
    }
  }

  // Project down to output_types if the scan emitted trailing filter-only columns.
  // After filtering, scan_output emission order matches source_ids, so the first
  // output_types.size() positions are the original projection columns.
  std::size_t const expected_out = info.output_types.size();
  if (expected_out > 0 && static_cast<std::size_t>(table->num_columns()) > expected_out) {
    auto cols = table->release();
    std::vector<std::unique_ptr<cudf::column>> selected;
    selected.reserve(expected_out);
    for (std::size_t i = 0; i < expected_out; ++i) {
      selected.push_back(std::move(cols[i]));
    }
    table = std::make_unique<cudf::table>(std::move(selected));
  }

  auto batch = sirius::make_data_batch(std::move(table), *mem_space, stream);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_gpu_duckdb_native_scan_operator::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  return stats.bytes * 4;
}

}  // namespace sirius::op::scan
