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

#include "vss/sirius_physical_vector_join.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace sirius::op {

namespace {

std::unique_ptr<cudf::column> make_empty_column_for(const sirius::logical_type& t)
{
  if (t.is_array()) {
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(2);
    // LIST layout is [offsets(INT32), child values]
    children.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
    children.push_back(make_empty_column_for(t.array_child()));
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                          0,
                                          rmm::device_buffer{},
                                          rmm::device_buffer{},
                                          0,
                                          std::move(children));
  }
  return cudf::make_empty_column(sirius::get_cudf_type(t));
}

std::unique_ptr<cudf::table> make_empty_table(const duckdb::vector<sirius::logical_type>& types)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(types.size());
  for (auto const& t : types) {
    columns.push_back(make_empty_column_for(t));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace

sirius_physical_vector_join::sirius_physical_vector_join(duckdb::vector<sirius::logical_type> types,
                                                         duckdb::idx_t estimated_cardinality,
                                                         sirius::vss::vector_join_request request)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN, std::move(types), estimated_cardinality),
    _request(std::move(request))
{
}

//===----------------------------------------------------------------------===//
// Source / scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_physical_vector_join::get_next_task_hint()
{
  bool expected = false;
  if (!_task_scheduled.compare_exchange_strong(expected, true)) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_physical_vector_join::all_ports_empty()
{
  return _input_handed_out.load(std::memory_order_acquire);
}

std::unique_ptr<operator_data> sirius_physical_vector_join::get_next_task_input_data()
{
  if (_input_handed_out.exchange(true, std::memory_order_acq_rel)) { return nullptr; }
  return std::make_unique<vector_join_input>(estimated_source_bytes());
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//
std::unique_ptr<operator_data> sirius_physical_vector_join::execute(const operator_data& input_data,
                                                                    rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_join::execute"};

  auto const* join_in = dynamic_cast<const vector_join_input*>(&input_data);
  if (!join_in) {
    throw std::runtime_error(
      "[sirius_physical_vector_join::execute] expected input of type vector_join_input; got " +
      std::string(typeid(input_data).name()));
  }
  auto* mem_space = join_in->get_gpu_memory_space();
  if (!mem_space) {
    throw std::runtime_error(
      "[sirius_physical_vector_join::execute] no memory space set on task input; "
      "prepare_for_processing was not called");
  }

  auto output_table = make_empty_table(types);

  auto batch =
    sirius::make_data_batch(std::move(output_table), *mem_space, stream, batch_telemetry());
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches{std::move(batch)};
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

//===----------------------------------------------------------------------===//
// Memory estimation
//===----------------------------------------------------------------------===//
std::size_t sirius_physical_vector_join::estimated_source_bytes() const
{
  // Nominal basis for the skeleton's single task. Refined once execute() reads
  // the pinned left/right and sizes the k-means / adjacency working set from
  // the actual row counts and _request.dim.
  return std::size_t{1} << 20;
}

std::size_t sirius_physical_vector_join::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // Skeleton emits an empty table, so peak is nominal. Floor at 1 MiB so the
  // reservation request is well-formed. Revisit when the [m x n] per-cluster
  // adjacency intermediate becomes the memory-bound term.
  return std::max<std::size_t>(stats.bytes, std::size_t{1} << 20);
}

std::string sirius_physical_vector_join::params_to_string() const
{
  return _request.left.table + "(" + _request.left.column + ") x " + _request.right.table + "(" +
         _request.right.column + ") metric=" + _request.metric +
         " k=" + std::to_string(_request.k) + " n_clusters=" + std::to_string(_request.n_clusters) +
         " n_probes=" + std::to_string(_request.n_probes) + " eps=" + std::to_string(_request.eps);
}

}  // namespace sirius::op
