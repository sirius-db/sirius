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

#include "op/sirius_physical_vss.hpp"

#include "data/data_batch_utils.hpp"
#include "op/sirius_physical_vss_merge.hpp"
#include "op/vss_top_k.hpp"
#include "sirius/exception.hpp"
#include "vss/brute_force_search.hpp"
#include "vss/cudf_raft_interop.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace sirius {
namespace op {

namespace {

// Returns an empty output table with the correct VSS output schema for
// the no-work cases (i.e., empty input or limit = 0).
std::unique_ptr<cudf::table> make_empty_vss_output(cudf::table_view input,
                                                   sirius::vss::vss_top_k_pattern const& pattern)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(pattern.output_columns.size());
  for (auto const& oc : pattern.output_columns) {
    if (oc.which == sirius::vss::vss_output_column::kind::distance) {
      cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT32}));
    } else {
      cols.push_back(cudf::empty_like(input.column(oc.input_index)));
    }
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

std::unique_ptr<cudf::table> compute_vss_top_k(
  cudf::table_view input,
  sirius::vss::vss_top_k_pattern const& pattern,
  std::size_t limit,
  std::size_t offset,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref memory_resource,
  std::optional<sirius::vss::dataset_matrix_view> query)
{
  if (limit == 0 || input.num_rows() == 0) { return make_empty_vss_output(input, pattern); }

  auto const keep = std::min<int64_t>(input.num_rows(), static_cast<int64_t>(offset + limit));
  if (keep == 0) { return make_empty_vss_output(input, pattern); }

  auto const col_idx = pattern.vector_column_index;
  if (col_idx < 0 || col_idx >= input.num_columns()) {
    throw internal_exception("VSS vector column index out of range");
  }

  // Zero-copy reinterpretation from cudf LIST column into a matrix view
  auto dataset_view = sirius::vss::list_column_as_dataset_view(input.column(col_idx), pattern.dim);

  // The query vector is constant (a [1, dim] device matrix)
  rmm::device_buffer local_query;
  if (!query.has_value()) {
    local_query = rmm::device_buffer(pattern.query.size() * sizeof(float), stream, memory_resource);
    CUDF_CUDA_TRY(cudaMemcpyAsync(local_query.data(),
                                  pattern.query.data(),
                                  local_query.size(),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    stream.synchronize();
  }
  auto query_view = query.has_value()
                      ? *query
                      : raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
                          static_cast<float const*>(local_query.data()), int64_t{1}, pattern.dim);

  auto knn = sirius::vss::brute_force_knn(dataset_view, query_view, keep, pattern.metric);

  auto gathered = cudf::gather(
    input, knn.neighbors->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, memory_resource);

  // Release so passthroughs are moved into the output instead of deep-copied
  auto gathered_cols = gathered->release();
  std::vector<int> remaining_refs(gathered_cols.size(), 0);
  for (auto const& oc : pattern.output_columns) {
    if (oc.which == sirius::vss::vss_output_column::kind::gather_input) {
      // Per-column reference count
      ++remaining_refs[oc.input_index];
    }
  }

  // Assemble the output columns (local per-batch top-k handed to VSS_MERGE)
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(pattern.output_columns.size());
  for (auto const& oc : pattern.output_columns) {
    if (oc.which == sirius::vss::vss_output_column::kind::distance) {
      // Distance is computed from cuVS
      out_cols.push_back(std::move(knn.distances));
    } else if (--remaining_refs[oc.input_index] == 0) {
      // Last (or only) use of this passthrough column (steal it)
      out_cols.push_back(std::move(gathered_cols[oc.input_index]));
    } else {
      // Columns with more than 1 reference count need to be deep-copied
      out_cols.push_back(std::make_unique<cudf::column>(
        gathered_cols[oc.input_index]->view(), stream, memory_resource));
    }
  }
  return std::make_unique<cudf::table>(std::move(out_cols));
}

// Merge per-batch VSS candidates into the global nearest rows. Each input row
// already carries its cuVS distance in column `distance_index`, so the merge is
// a plain top-k sort on that column.
std::unique_ptr<cudf::table> merge_vss_top_k(cudf::table_view input,
                                             cudf::size_type distance_index,
                                             std::size_t limit,
                                             std::size_t offset,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref memory_resource)
{
  if (limit == 0 || input.num_rows() == 0) { return duckdb::make_empty_like(input); }

  auto const keep =
    std::min<cudf::size_type>(input.num_rows(), static_cast<cudf::size_type>(offset + limit));
  if (keep == 0) { return duckdb::make_empty_like(input); }
  if (distance_index < 0 || distance_index >= input.num_columns()) {
    throw internal_exception("VSS merge distance index out of range");
  }

  auto indices = cudf::top_k_order(
    input.column(distance_index), keep, cudf::order::ASCENDING, stream, memory_resource);
  auto gathered = cudf::gather(
    input, indices->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, memory_resource);
  // top_k_order does not guarantee sorted output, it sorts the kept rows by distance.
  return cudf::sort_by_key(gathered->view(),
                           cudf::table_view({gathered->view().column(distance_index)}),
                           {cudf::order::ASCENDING},
                           {cudf::null_order::AFTER},
                           stream,
                           memory_resource);
}

sirius_physical_vss::sirius_physical_vss(duckdb::vector<sirius::logical_type> types_p,
                                         sirius::vss::vss_top_k_pattern pattern_p,
                                         std::size_t limit,
                                         std::size_t offset,
                                         std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::VSS, std::move(types_p), estimated_cardinality),
    pattern(std::move(pattern_p)),
    limit(limit),
    offset(offset)
{
}

sirius_physical_vss::~sirius_physical_vss() {}

std::unique_ptr<operator_data> sirius_physical_vss::execute(const operator_data& input_data,
                                                            rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vss::execute"};
  auto& input = dynamic_cast<const pipelineable_operator_data&>(input_data);
  // LIFETIME/RESIDENCY: get_read_only_batches() returns shared-locked
  // read accessors by value; binding to `const auto&` lifetime-extends them for
  // this whole scope. brute_force_knn borrows the dataset column as a non-owning
  // mdspan, so it must stay alive and on-device for the search. The shared lock
  // guarantees both: prepare_for_processing already made the batch GPU-resident
  // before execute, and a spill (D2H) would need the batch's exclusive lock,
  // which cannot be taken while these shared locks are held. The search is
  // synchronous, so the locks (and residency) outlive every cuVS read.
  const auto& input_batches = input.get_read_only_batches();
  if (limit == 0) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  if (input_batches.empty()) {
    return std::make_unique<pipelineable_operator_data>();
  } else if (input_batches.size() > 1) {
    throw internal_exception("VSS expects a single input batch per execution");
  }

  // Copy holds its own shared lock; alive through the call
  auto input_batch = input_batches[0];
  auto* space      = input_batch.get_memory_space();
  if (space == nullptr) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }
  // brute_force_knn reads the dataset directly on the GPU, so the batch
  // must be GPU-resident. prepare_for_processing should guarantee this
  if (space->get_tier() != cucascade::memory::Tier::GPU) {
    throw internal_exception("VSS input batch is not GPU-resident");
  }

  auto input_table_view =
    input_batch.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();

  // Upload query vector once and reuse across batches
  std::call_once(query_uploaded, [&] {
    query_buf = rmm::device_buffer(
      pattern.query.size() * sizeof(float), stream, space->get_default_allocator());
    CUDF_CUDA_TRY(cudaMemcpyAsync(query_buf.data(),
                                  pattern.query.data(),
                                  query_buf.size(),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    stream.synchronize();
  });
  auto query_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    static_cast<float const*>(query_buf.data()), int64_t{1}, pattern.dim);

  auto output_table = compute_vss_top_k(
    input_table_view, pattern, limit, offset, stream, space->get_default_allocator(), query_view);

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  // STREAM-LINEAGE: compute_vss_top_k writes the output table on stream; the
  // constructor records the writer event so cross-device readers honor ordering
  auto output_repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(output_table), *space, stream);
  std::unique_ptr<cucascade::idata_representation> output_data = std::move(output_repr);
  outputs.push_back(
    std::make_shared<cucascade::data_batch>(::sirius::get_next_batch_id(), std::move(output_data)));
  return std::make_unique<pipelineable_operator_data>(outputs);
}

sirius_physical_vss_merge::sirius_physical_vss_merge(sirius_physical_vss* vss)
  : sirius_physical_vss_merge(vss->types,    // copied by value
                              vss->pattern,  // deep copy
                              vss->limit,
                              vss->offset,
                              vss->estimated_cardinality)
{
  child_op = vss;
}

sirius_physical_vss_merge::sirius_physical_vss_merge(duckdb::vector<sirius::logical_type> types_p,
                                                     sirius::vss::vss_top_k_pattern pattern_p,
                                                     std::size_t limit,
                                                     std::size_t offset,
                                                     std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_VSS, std::move(types_p), estimated_cardinality),
    pattern(std::move(pattern_p)),
    limit(limit),
    offset(offset),
    child_op(nullptr)
{
}

std::unique_ptr<operator_data> sirius_physical_vss_merge::execute(const operator_data& input_data,
                                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vss_merge::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (limit == 0) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    space = batch.get_memory_space();
    break;
  }
  if (space == nullptr) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  std::vector<cudf::table_view> concat_views;
  for (auto const& batch : input_batches) {
    concat_views.push_back(
      batch.get_data()->cast<cucascade::gpu_table_representation>().get_table_view());
  }

  if (concat_views.empty()) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  std::unique_ptr<cudf::table> combined;
  if (concat_views.size() == 1) {
    combined =
      std::make_unique<cudf::table>(concat_views.front(), stream, space->get_default_allocator());
  } else {
    combined = cudf::concatenate(concat_views, stream, space->get_default_allocator());
  }

  auto output_table = merge_vss_top_k(combined->view(),
                                      pattern.distance_output_index,
                                      limit,
                                      offset,
                                      stream,
                                      space->get_default_allocator());
  if (output_table->num_rows() <= static_cast<cudf::size_type>(offset)) {
    output_table = duckdb::make_empty_like(output_table->view());
  } else if (offset > 0) {
    auto out_start = static_cast<cudf::size_type>(offset);
    auto out_slices =
      cudf::slice(output_table->view(), {out_start, output_table->num_rows()}, stream);
    output_table =
      std::make_unique<cudf::table>(out_slices.front(), stream, space->get_default_allocator());
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  // STREAM-LINEAGE: merge_vss_top_k + slice write on stream; the constructor
  // records the writer event for downstream cross-device readers.
  auto output_repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(output_table), *space, stream);
  std::unique_ptr<cucascade::idata_representation> output_data = std::move(output_repr);
  outputs.push_back(
    std::make_shared<cucascade::data_batch>(::sirius::get_next_batch_id(), std::move(output_data)));
  return std::make_unique<pipelineable_operator_data>(outputs);
}

std::unique_ptr<operator_data> sirius_physical_vss_merge::get_next_task_input_data()
{
  // Lock, drain all batches from the single partition, return them.
  std::lock_guard<std::mutex> lg(lock);
  std::vector<::std::shared_ptr<::cucascade::data_batch>> input_batch;
  bool found_batch = true;
  while (found_batch) {
    auto batch = ports.begin()->second->repo->pop_next_data_batch();
    if (batch) {
      input_batch.push_back(std::move(batch));
    } else {
      found_batch = false;
    }
  }
  if (input_batch.empty()) { return nullptr; }
  return std::make_unique<pipelineable_operator_data>(input_batch);
}

}  // namespace op
}  // namespace sirius
