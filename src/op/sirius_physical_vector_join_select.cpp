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

#include "vss/sirius_physical_vector_join_select.hpp"

#include "data/data_batch_utils.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "vss/brute_force_search.hpp"
#include "vss/cudf_raft_interop.hpp"
#include "vss/distance_metric.hpp"
#include "vss/pinned_column.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <raft/core/device_resources.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace sirius::op {

sirius_physical_vector_join_select::sirius_physical_vector_join_select(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  sirius::vss::vector_join_request request,
  sirius::scan_manager::sirius_scan_manager* scan_manager,
  bool is_fast_path)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN_SELECT, std::move(types), estimated_cardinality),
    _request(std::move(request)),
    _scan_manager(scan_manager),
    _is_fast_path(is_fast_path)
{
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//
void sirius_physical_vector_join_select::ensure_initialized_locked()
{
  if (_initialized) { return; }
  if (_scan_manager == nullptr) {
    throw std::runtime_error("[sirius_physical_vector_join_select] no scan manager set");
  }

  auto const& left  = _request.left;
  auto const& right = _request.right;

  const auto* left_pin =
    _scan_manager->find_pinned_entry_for_duckdb_table(left.catalog, left.schema, left.table);
  const auto* right_pin =
    _scan_manager->find_pinned_entry_for_duckdb_table(right.catalog, right.schema, right.table);
  if (left_pin == nullptr || right_pin == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_select] left or right table is no longer pinned");
  }

  // Zero-copy views over each pinned batch's vector column, in row order.
  _left_views =
    vss::pinned_column_chunk_views(*left_pin, left.column, vss::pinned_entry_gpu_space(*left_pin));
  _right_views = vss::pinned_column_chunk_views(
    *right_pin, right.column, vss::pinned_entry_gpu_space(*right_pin));

  // Each right batch's neighbor ids are local (0..batch_rows); record the global
  // row offset so execute() can shift them into the whole-right-table id space.
  // (Only the payload path uses these.)
  _right_offsets.resize(_right_views.size());
  std::int64_t acc = 0;
  for (std::size_t j = 0; j < _right_views.size(); ++j) {
    _right_offsets[j] = acc;
    acc += static_cast<std::int64_t>(_right_views[j].size());
  }

  // Fast path: snapshot the right id column's per-batch views so execute() can
  // gather each pair's id values from local neighbor positions and carry them
  // through the merge.
  if (_is_fast_path) {
    if (right.output_columns.size() != 1) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_select] is_fast_path requires exactly one right output "
        "column (the id)");
    }
    _right_id_views = vss::pinned_column_chunk_views(
      *right_pin, right.output_columns.front(), vss::pinned_entry_gpu_space(*right_pin));
  }

  _num_pairs   = _left_views.size() * _right_views.size();
  _initialized = true;
}

//===----------------------------------------------------------------------===//
// Source / scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_physical_vector_join_select::get_next_task_hint()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  ensure_initialized_locked();
  // One READY is enough for the task creator to drain every pair via
  // get_next_task_input_data() until all_ports_empty() reports done.
  if (_num_pairs == 0 || _next_pair >= _num_pairs || _hint_returned) { return std::nullopt; }
  _hint_returned = true;
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_physical_vector_join_select::all_ports_empty()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  ensure_initialized_locked();
  return _next_pair >= _num_pairs;
}

std::unique_ptr<operator_data> sirius_physical_vector_join_select::get_next_task_input_data()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  ensure_initialized_locked();
  if (_next_pair >= _num_pairs) { return nullptr; }

  auto const n_right    = _right_views.size();
  auto const pair_index = _next_pair++;
  auto const left_idx   = pair_index / n_right;
  auto const right_idx  = pair_index % n_right;

  return std::make_unique<vector_join_input>(
    left_idx, right_idx, per_pair_estimate(left_idx, right_idx));
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//
std::unique_ptr<operator_data> sirius_physical_vector_join_select::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_join_select::execute"};

  auto const* join_input = dynamic_cast<const vector_join_input*>(&input_data);
  if (join_input == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_select::execute] expected vector_join_input; got " +
      std::string(typeid(input_data).name()));
  }
  auto* mem_space = join_input->get_gpu_memory_space();
  if (mem_space == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_select::execute] no memory space set; prepare_for_processing "
      "was not called");
  }

  auto const left_idx  = join_input->left_idx();
  auto const right_idx = join_input->right_idx();
  auto const dim       = _request.dim;

  // Zero-copy matrix views over this pair's pinned batches.
  auto const queries = vss::list_column_as_dataset_view(_left_views[left_idx], dim);
  auto const dataset = vss::list_column_as_dataset_view(_right_views[right_idx], dim);

  auto const n_right = dataset.extent(0);
  auto const k_eff   = std::min<std::int64_t>(_request.k, n_right);

  // Per-pair brute-force top-k. brute_force_knn tiles the pairwise distances
  // internally, so the dense [n_left, n_right] block is never materialized.
  // Allocations flow through the task's reservation-aware current resource.
  auto const mr = mem_space->get_default_allocator();
  raft::device_resources res{stream};
  auto const exact_unexpanded = _request.search_mode == vss::vector_join_search_mode::exact;
  auto const metric =
    vss::join_selection_distance_type_from_metric(_request.metric, exact_unexpanded);
  auto knn = vss::brute_force_knn(res, dataset, queries, k_eff, metric, mr);

  // On the fast path, the neighbor stores the actual right table's id values. It is widened
  // to INT64 for downstream ops (reduce and materialize).
  std::unique_ptr<cudf::column> neighbors;
  if (_is_fast_path) {
    auto const id_table = cudf::table_view{{_right_id_views[right_idx]}};
    auto gathered       = cudf::gather(
      id_table, knn.neighbors->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    auto gathered_cols = gathered->release();
    neighbors          = std::move(gathered_cols.front());
    if (neighbors->type().id() != cudf::type_id::INT64) {
      neighbors = cudf::cast(neighbors->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
    }
  }
  // On the payload path, the neighbor stores right table's row number. Materialize op uses the
  // row number later to gather the requested output columns.
  else {
    neighbors         = std::move(knn.neighbors);
    auto const offset = _right_offsets[right_idx];
    if (offset != 0) {
      cudf::numeric_scalar<std::int64_t> const off_scalar(offset, true, stream);
      neighbors = cudf::binary_operation(neighbors->view(),
                                         off_scalar,
                                         cudf::binary_operator::ADD,
                                         cudf::data_type{cudf::type_id::INT64},
                                         stream,
                                         mr);
    }
  }

  // Tag with the left batch index so all of a left batch's per-right-batch partials
  // land in one partition for reduce.
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(2);
  out_cols.push_back(std::move(neighbors));
  out_cols.push_back(std::move(knn.distances));
  auto out_table = std::make_unique<cudf::table>(std::move(out_cols));

  auto batch = sirius::make_data_batch(std::move(out_table), *mem_space, stream, batch_telemetry());
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<partitioned_operator_data>(std::move(batches), left_idx);
}

//===----------------------------------------------------------------------===//
// Sink: route each partial to the merge partition for its left batch
//===----------------------------------------------------------------------===//
void sirius_physical_vector_join_select::sink(const operator_data& output_data,
                                              rmm::cuda_stream_view /*stream*/)
{
  auto const& part         = dynamic_cast<const partitioned_operator_data&>(output_data);
  auto const partition_idx = part.get_partition_idx();
  for (auto& batch : part.get_data_batches()) {
    for (auto& next_port_info : next_port_after_sink) {
      auto* consumer =
        dynamic_cast<sirius_physical_partition_consumer_operator*>(next_port_info.next_operator);
      if (consumer == nullptr) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_select::sink] next operator is not a partition consumer");
      }
      consumer->push_data_batch_partitioned(
        next_port_info.next_operator_port_name, batch, partition_idx);
    }
  }
}

//===----------------------------------------------------------------------===//
// Memory estimation
//===----------------------------------------------------------------------===//
std::size_t sirius_physical_vector_join_select::per_pair_estimate(std::size_t left_idx,
                                                                  std::size_t right_idx) const
{
  // Output is n_left * k (of INT64 id and FLOAT32 distance), so n_left * k * 12 bytes.
  auto const n_left  = static_cast<std::int64_t>(_left_views[left_idx].size());
  auto const n_right = static_cast<std::int64_t>(_right_views[right_idx].size());
  auto const k       = std::max<std::int64_t>(_request.k, 1);
  auto const output  = static_cast<std::size_t>(n_left) * static_cast<std::size_t>(k) *
                      (sizeof(std::int64_t) + sizeof(float));

  // brute_force_knn's scratch dominates and scales with the pairwise-distance tile (n_right, dim)
  auto const exact_unexpanded = _request.search_mode == vss::vector_join_search_mode::exact;
  auto const metric =
    vss::join_selection_distance_type_from_metric(_request.metric, exact_unexpanded);
  auto const scratch =
    vss::brute_force_peak_scratch_bytes(n_left, n_right, _request.dim, k, metric);

  return output + scratch;
}

std::size_t sirius_physical_vector_join_select::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // The per-pair input already carries a sized estimate (output + cuVS search
  // scratch, from per_pair_estimate/brute_force_peak_scratch_bytes); floor it so
  // the reservation request is always well-formed.
  return std::max<std::size_t>(stats.bytes, std::size_t{1} << 20);
}

std::string sirius_physical_vector_join_select::params_to_string() const
{
  return _request.left.table + "(" + _request.left.column + ") x " + _request.right.table + "(" +
         _request.right.column + ") metric=" + _request.metric +
         " k=" + std::to_string(_request.k) +
         " mode=" + std::to_string(static_cast<int>(_request.mode));
}

}  // namespace sirius::op
