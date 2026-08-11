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

#include "vss/sirius_physical_vector_join_local_merge.hpp"

#include "data/data_batch_utils.hpp"
#include "vss/knn_merge.hpp"

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <raft/core/device_resources.hpp>

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

sirius_physical_vector_join_local_merge::sirius_physical_vector_join_local_merge(
  duckdb::vector<sirius::logical_type> types, duckdb::idx_t estimated_cardinality, std::int64_t k)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN_LOCAL_MERGE, std::move(types), estimated_cardinality),
    _k(k)
{
}

std::unique_ptr<operator_data> sirius_physical_vector_join_local_merge::get_next_task_input_data()
{
  // One merge task per partition (= one left batch): drain all its partials.
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

std::unique_ptr<operator_data> sirius_physical_vector_join_local_merge::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_join_local_merge::execute"};

  auto const& input          = dynamic_cast<const partitioned_operator_data&>(input_data);
  auto const partition_idx   = input.get_partition_idx();
  auto const& input_batches  = input.get_read_only_batches();

  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (space == nullptr) { space = batch.get_memory_space(); }
  }
  if (input_batches.empty() || space == nullptr) {
    return std::make_unique<partitioned_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{}, partition_idx);
  }

  // Each partial is [neighbor_id INT64 (col 0), distance FLOAT32 (col 1)],
  // flattened row-major [n_samples * k]. Collect the two columns from every
  // partial; concatenating in order yields the part-major layout knn_merge_parts
  // expects (part p's [n_samples, k] block at row p*n_samples).
  std::vector<cudf::column_view> neighbor_views;
  std::vector<cudf::column_view> distance_views;
  neighbor_views.reserve(input_batches.size());
  distance_views.reserve(input_batches.size());
  for (auto const& ro : input_batches) {
    auto const tv = sirius::get_cudf_table_view(ro);
    neighbor_views.push_back(tv.column(0));
    distance_views.push_back(tv.column(1));
  }

  auto const n_parts   = static_cast<std::int64_t>(input_batches.size());
  auto const part_rows = static_cast<std::int64_t>(sirius::get_cudf_table_view(input_batches[0]).num_rows());
  auto const n_samples = part_rows / _k;  // uniform k assumed; leaf re-checks the total size

  auto const mr = space->get_default_allocator();

  auto const stacked_neighbors = cudf::concatenate(neighbor_views, stream, mr);
  auto const stacked_distances = cudf::concatenate(distance_views, stream, mr);

  raft::device_resources res{stream};
  auto merged = vss::knn_merge_parts_topk(res,
                                          stacked_distances->view(),
                                          stacked_neighbors->view(),
                                          n_samples,
                                          n_parts,
                                          _k,
                                          stream,
                                          mr);

  // Preserve the [neighbor_id, distance] layout and the partition (left batch)
  // so the materialize stage can gather each left row's output columns.
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(2);
  out_cols.push_back(std::move(merged.neighbors));
  out_cols.push_back(std::move(merged.distances));
  auto out_table = std::make_unique<cudf::table>(std::move(out_cols));

  auto batch = sirius::make_data_batch(std::move(out_table), *space, stream, batch_telemetry());
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<partitioned_operator_data>(std::move(batches), partition_idx);
}

void sirius_physical_vector_join_local_merge::sink(const operator_data& output_data,
                                        rmm::cuda_stream_view /*stream*/)
{
  auto const& part         = dynamic_cast<const partitioned_operator_data&>(output_data);
  auto const partition_idx = part.get_partition_idx();
  for (auto& batch : part.get_data_batches()) {
    for (auto& next_port_info : next_port_after_sink) {
      auto* consumer = dynamic_cast<sirius_physical_partition_consumer_operator*>(
        next_port_info.next_operator);
      if (consumer == nullptr) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_local_merge::sink] next operator is not a partition consumer");
      }
      consumer->push_data_batch_partitioned(
        next_port_info.next_operator_port_name, batch, partition_idx);
    }
  }
}

std::size_t sirius_physical_vector_join_local_merge::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // Peak ~ the stacked partials (input) + the merged output; floor at 1 MiB.
  return std::max<std::size_t>(stats.bytes, std::size_t{1} << 20);
}

std::string sirius_physical_vector_join_local_merge::params_to_string() const
{
  return "k=" + std::to_string(_k);
}

}  // namespace sirius::op
