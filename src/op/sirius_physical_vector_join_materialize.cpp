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

#include "vss/sirius_physical_vector_join_materialize.hpp"

#include "data/data_batch_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "vss/pinned_column.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op {

sirius_physical_vector_join_materialize::sirius_physical_vector_join_materialize(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  sirius::vss::vector_join_request request,
  sirius::scan_manager::sirius_scan_manager* scan_manager)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN_MATERIALIZE, std::move(types), estimated_cardinality),
    _request(std::move(request)),
    _scan_manager(scan_manager)
{
}

void sirius_physical_vector_join_materialize::ensure_initialized(
  rmm::cuda_stream_view stream, ::cucascade::memory::memory_space& space)
{
  std::lock_guard<std::mutex> lg(_init_mutex);
  if (_initialized) { return; }
  if (_scan_manager == nullptr) {
    throw std::runtime_error("[sirius_physical_vector_join_materialize] no scan manager set");
  }

  auto const& left  = _request.left;
  auto const& right = _request.right;

  const auto* left_pin =
    _scan_manager->find_pinned_entry_for_duckdb_table(left.catalog, left.schema, left.table);
  const auto* right_pin =
    _scan_manager->find_pinned_entry_for_duckdb_table(right.catalog, right.schema, right.table);
  if (left_pin == nullptr || right_pin == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_materialize] left/right table is no longer pinned");
  }
  auto& left_space  = vss::pinned_entry_gpu_space(*left_pin);
  auto& right_space = vss::pinned_entry_gpu_space(*right_pin);

  // Left output columns as zero-copy per-batch views: _left_output_cols[col][batch].
  _left_output_cols.resize(left.output_columns.size());
  for (std::size_t c = 0; c < left.output_columns.size(); ++c) {
    _left_output_cols[c] =
      vss::pinned_column_chunk_views(*left_pin, left.output_columns[c], left_space);
  }

  // Right output columns concatenated once across batches, so a global right id
  // gathers straight into row i. Small columns (not the vectors), so cheap.
  auto const mr = space.get_default_allocator();
  std::vector<std::unique_ptr<cudf::column>> right_cols;
  right_cols.reserve(right.output_columns.size());
  for (auto const& name : right.output_columns) {
    auto views = vss::pinned_column_chunk_views(*right_pin, name, right_space);
    right_cols.push_back(cudf::concatenate(views, stream, mr));
  }
  _right_output_concat = std::make_unique<cudf::table>(std::move(right_cols));

  _initialized = true;
}

std::unique_ptr<operator_data> sirius_physical_vector_join_materialize::get_next_task_input_data()
{
  // One task per partition (= one left batch): drain its merge outputs.
  std::lock_guard<std::mutex> lg(_drain_mutex);

  auto* repo = ports.begin()->second->repo;
  if (_current_partition_index >= repo->num_partitions()) { return nullptr; }

  std::vector<std::shared_ptr<cucascade::data_batch>> all_batches;
  while (true) {
    auto batch = repo->pop_next_data_batch(_current_partition_index);
    if (!batch) { break; }
    all_batches.push_back(std::move(batch));
  }
  auto const partition_idx = _current_partition_index++;
  if (all_batches.empty()) { return nullptr; }
  return std::make_unique<partitioned_operator_data>(std::move(all_batches), partition_idx);
}

std::unique_ptr<operator_data> sirius_physical_vector_join_materialize::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_join_materialize::execute"};

  auto const& input         = dynamic_cast<const partitioned_operator_data&>(input_data);
  auto const partition_idx  = input.get_partition_idx();  // = left batch index
  auto const& input_batches = input.get_read_only_batches();

  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (space == nullptr) { space = batch.get_memory_space(); }
  }
  if (input_batches.empty() || space == nullptr) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  ensure_initialized(stream, *space);
  auto const mr = space->get_default_allocator();

  // Merge emits one result batch per partition; concatenate defensively if more.
  std::vector<cudf::column_view> neighbor_views;
  std::vector<cudf::column_view> distance_views;
  for (auto const& ro : input_batches) {
    auto const tv = sirius::get_cudf_table_view(ro);
    neighbor_views.push_back(tv.column(0));  // INT64 global right id
    distance_views.push_back(tv.column(1));  // FLOAT32 distance
  }
  std::unique_ptr<cudf::column> neighbor_owned;
  std::unique_ptr<cudf::column> distance_owned;
  cudf::column_view neighbor_view = neighbor_views.front();
  cudf::column_view distance_view = distance_views.front();
  if (input_batches.size() > 1) {
    neighbor_owned = cudf::concatenate(neighbor_views, stream, mr);
    distance_owned = cudf::concatenate(distance_views, stream, mr);
    neighbor_view  = neighbor_owned->view();
    distance_view  = distance_owned->view();
  }

  // Left columns of batch `partition_idx`, each row repeated for its k neighbors.
  std::vector<cudf::column_view> left_batch_cols;
  left_batch_cols.reserve(_left_output_cols.size());
  for (auto const& per_batch : _left_output_cols) {
    left_batch_cols.push_back(per_batch[partition_idx]);
  }
  auto const left_table = cudf::table_view(left_batch_cols);
  auto const total_rows = static_cast<int64_t>(neighbor_view.size());
  auto const left_rows  = static_cast<int64_t>(left_table.num_rows());
  CUDF_EXPECTS(left_rows > 0 && total_rows % left_rows == 0,
               "VSS materialize: result rows are not a whole multiple of the left batch rows");
  auto const k_eff = static_cast<cudf::size_type>(total_rows / left_rows);
  auto left_repeated = cudf::repeat(left_table, k_eff, stream, mr);

  // Right columns gathered by the global neighbor id.
  auto right_gathered = cudf::gather(
    _right_output_concat->view(), neighbor_view, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

  // Score: distance, or cosine similarity (1 - distance) when similarity is asked.
  std::unique_ptr<cudf::column> score;
  if (_request.output_type == sirius::vss::vector_join_output_type::similarity &&
      _request.metric == "cosine") {
    cudf::numeric_scalar<float> const one(1.0F, true, stream);
    score = cudf::binary_operation(one,
                                   distance_view,
                                   cudf::binary_operator::SUB,
                                   cudf::data_type{cudf::type_id::FLOAT32},
                                   stream,
                                   mr);
  } else {
    score = std::make_unique<cudf::column>(distance_view, stream, mr);
  }

  // Assemble [left cols..., right cols..., score] — the TVF schema.
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  auto left_cols  = left_repeated->release();
  auto right_cols = right_gathered->release();
  out_cols.reserve(left_cols.size() + right_cols.size() + 1);
  for (auto& c : left_cols) { out_cols.push_back(std::move(c)); }
  for (auto& c : right_cols) { out_cols.push_back(std::move(c)); }
  out_cols.push_back(std::move(score));
  auto out_table = std::make_unique<cudf::table>(std::move(out_cols));

  auto batch = sirius::make_data_batch(std::move(out_table), *space, stream, batch_telemetry());
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_physical_vector_join_materialize::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  return std::max<std::size_t>(stats.bytes, std::size_t{1} << 20);
}

std::string sirius_physical_vector_join_materialize::params_to_string() const
{
  return _request.left.table + " x " + _request.right.table + " k=" + std::to_string(_request.k);
}

}  // namespace sirius::op
