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

#include "op/sirius_physical_vss_ann_ivf_flat.hpp"

#include "data/data_batch_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "vss/cuvs_index_cache.hpp"
#include "vss/ivf_flat_index.hpp"
#include "vss/pinned_column.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <utility>

namespace sirius {
namespace op {

namespace {

// Build an empty output table with the projection's VSS output schema, for the
// no-work cases (limit == 0 or an empty pinned table). gather_input columns take
// their type from the pinned table column; the distance column is FLOAT32.
std::unique_ptr<cudf::table> make_empty_ann_output(
  const scan_manager::pinned_entry& pin,
  const sirius::vss::vss_top_k_pattern& pattern,
  const std::vector<std::string>& output_column_names)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(pattern.output_columns.size());
  for (std::size_t i = 0; i < pattern.output_columns.size(); ++i) {
    auto const& oc = pattern.output_columns[i];
    if (oc.which == sirius::vss::vss_output_column::kind::distance) {
      cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT32}));
      continue;
    }
    auto it = pin.data_batches_by_column.find(output_column_names[i]);
    if (it == pin.data_batches_by_column.end() || it->second.empty()) {
      throw internal_exception("ANN (IVF-Flat): pinned table missing output column '" +
                               output_column_names[i] + "'");
    }
    cols.push_back(cudf::empty_like(it->second.front()->view()));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

sirius_physical_vss_ann_ivf_flat::sirius_physical_vss_ann_ivf_flat(
  duckdb::vector<sirius::logical_type> types_p,
  sirius::vss::vss_top_k_pattern pattern_p,
  std::size_t limit,
  std::size_t offset,
  std::size_t estimated_cardinality,
  duckdb::SiriusContext* sirius_context,
  std::string table_name,
  std::string vector_column_name,
  std::vector<std::string> output_column_names)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::ANN_IVF_FLAT, std::move(types_p), estimated_cardinality),
    pattern(std::move(pattern_p)),
    limit(limit),
    offset(offset),
    sirius_context_(sirius_context),
    table_name_(std::move(table_name)),
    vector_column_name_(std::move(vector_column_name)),
    output_column_names_(std::move(output_column_names))
{
}

sirius_physical_vss_ann_ivf_flat::~sirius_physical_vss_ann_ivf_flat() {}

std::optional<task_creation_hint> sirius_physical_vss_ann_ivf_flat::get_next_task_hint()
{
  if (dispatched_.load()) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};
}

std::unique_ptr<operator_data> sirius_physical_vss_ann_ivf_flat::get_next_task_input_data()
{
  bool expected = false;
  if (!dispatched_.compare_exchange_strong(expected, true)) { return nullptr; }

  // One-shot source: the single task carries no upstream data. Route it to the GPU
  // where the pinned index + pinned table live (gpu_spaces[0]) so execute runs there.
  auto data = std::make_unique<operator_data>();
  auto gpu_spaces =
    sirius_context_->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (!gpu_spaces.empty()) { data->set_preferred_device_id(gpu_spaces.front()->get_device_id()); }
  return data;
}

bool sirius_physical_vss_ann_ivf_flat::all_ports_empty() { return dispatched_.load(); }

std::unique_ptr<operator_data> sirius_physical_vss_ann_ivf_flat::execute(
  const operator_data& /*input*/, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vss_ann_ivf_flat::execute"};

  auto gpu_spaces =
    sirius_context_->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw internal_exception("ANN (IVF-Flat): no GPU memory space available");
  }
  auto* space          = const_cast<cucascade::memory::memory_space*>(gpu_spaces.front());
  int const target_gpu = space->get_device_id();
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_gpu}};
  auto mr = space->get_default_allocator();

  // Pinned IVF-Flat index (built on this GPU) and the GPU-resident pinned table
  // (rows in the same build order). The recognizer guarantees both exist.
  const auto* index_entry = sirius_context_->get_cuvs_index_cache().find_by_column(
    table_name_, vector_column_name_, pattern.metric);
  if (index_entry == nullptr || !index_entry->index) {
    throw internal_exception("ANN (IVF-Flat): pinned index for '" + table_name_ + "." +
                             vector_column_name_ + "' is no longer available");
  }
  const auto* pin = sirius_context_->get_scan_manager().find_pinned_entry(table_name_);
  if (pin == nullptr) {
    throw internal_exception("ANN (IVF-Flat): table '" + table_name_ + "' is no longer pinned");
  }
  if (pin->tier != cucascade::memory::Tier::GPU) {
    throw internal_exception("ANN (IVF-Flat): table '" + table_name_ +
                             "' must be pinned on the GPU tier for ANN search");
  }

  auto const num_rows = static_cast<int64_t>(pin->num_rows);

  auto emit = [&](std::unique_ptr<cudf::table> table) -> std::unique_ptr<operator_data> {
    auto output_repr =
      std::make_unique<cucascade::gpu_table_representation>(std::move(table), *space, stream);
    std::unique_ptr<cucascade::idata_representation> output_data = std::move(output_repr);
    std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
    outputs.push_back(
      cucascade::data_batch::make(::sirius::get_next_batch_id(), std::move(output_data)));
    return std::make_unique<pipelineable_operator_data>(std::move(outputs));
  };

  if (limit == 0 || num_rows == 0) {
    return emit(make_empty_ann_output(*pin, pattern, output_column_names_));
  }

  auto const k = std::min<int64_t>(num_rows, static_cast<int64_t>(offset + limit));

  // Upload the (constant) query vector once: a [1, dim] device matrix.
  rmm::device_buffer query_buf(pattern.query.size() * sizeof(float), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(query_buf.data(),
                                pattern.query.data(),
                                query_buf.size(),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  stream.synchronize();

  // Probe a bounded number of inverted lists (accuracy/speed knob).
  auto const n_lists =
    index_entry->meta.n_lists > 0 ? static_cast<std::uint32_t>(index_entry->meta.n_lists) : 1u;
  auto const n_probes = std::min<std::uint32_t>(n_lists, 32u);

  auto search = sirius::vss::search_ivf_flat_index(*index_entry->index,
                                                   static_cast<const float*>(query_buf.data()),
                                                   pattern.dim,
                                                   k,
                                                   n_probes,
                                                   stream,
                                                   mr);

  // Gather the requested output columns from the pinned table by the returned
  // global row indices. Concatenate each output column's chunks (build order)
  // into one contiguous column, then gather all at once.
  std::vector<std::unique_ptr<cudf::column>> full_cols;
  for (std::size_t i = 0; i < pattern.output_columns.size(); ++i) {
    if (pattern.output_columns[i].which == sirius::vss::vss_output_column::kind::distance) {
      continue;
    }
    full_cols.push_back(
      sirius::vss::concat_pinned_column(*pin, output_column_names_[i], *space, stream));
  }

  std::vector<cudf::column_view> full_views;
  full_views.reserve(full_cols.size());
  for (auto const& col : full_cols) {
    full_views.push_back(col->view());
  }

  auto gathered      = cudf::gather(cudf::table_view(full_views),
                               search.neighbors->view(),
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               stream,
                               mr);
  auto gathered_cols = gathered->release();

  // Assemble the projection's output: distance column from the search, the rest
  // from the gathered passthroughs, in pattern.output_columns order.
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(pattern.output_columns.size());
  std::size_t gathered_idx = 0;
  for (auto const& oc : pattern.output_columns) {
    if (oc.which == sirius::vss::vss_output_column::kind::distance) {
      out_cols.push_back(std::move(search.distances));
    } else {
      out_cols.push_back(std::move(gathered_cols[gathered_idx++]));
    }
  }
  auto output_table = std::make_unique<cudf::table>(std::move(out_cols));

  // Sort the k candidates by distance ascending (nearest-first) so the offset
  // slice is well-defined regardless of the search's output ordering.
  output_table = cudf::sort_by_key(
    output_table->view(),
    cudf::table_view({output_table->view().column(pattern.distance_output_index)}),
    {cudf::order::ASCENDING},
    {cudf::null_order::AFTER},
    stream,
    mr);

  if (output_table->num_rows() <= static_cast<cudf::size_type>(offset)) {
    output_table = make_empty_ann_output(*pin, pattern, output_column_names_);
  } else if (offset > 0) {
    auto const out_start = static_cast<cudf::size_type>(offset);
    auto slices  = cudf::slice(output_table->view(), {out_start, output_table->num_rows()}, stream);
    output_table = std::make_unique<cudf::table>(slices.front(), stream, mr);
  }

  return emit(std::move(output_table));
}

}  // namespace op
}  // namespace sirius
