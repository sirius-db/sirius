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

#include "op/sirius_physical_top_n.hpp"

#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/sorting.hpp>

#include <rmm/resource_ref.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>

#include <algorithm>
#include <limits>
#include <memory>

namespace sirius {
namespace op {

namespace {

std::size_t candidate_row_cap(std::size_t offset, std::size_t limit) noexcept
{
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  return offset > max - limit ? max : offset + limit;
}

//! Selects the first `keep_rows` rows of `input` under the given lexicographic ordering.
//!
//! Sorting the keys into a permutation and gathering only the surviving rows is much cheaper than
//! `cudf::sort_by_key` for a top-N: the sort materializes an int32 index column rather than a fully
//! sorted copy of every payload column, and the gather touches `keep_rows` rows instead of all of
//! them. Unlike `cudf::top_k_order` this honors SQL NULLS FIRST/LAST, so it is the fallback for
//! nullable and multi-key orderings.
std::unique_ptr<cudf::table> sorted_order_top_k(cudf::table_view input,
                                                cudf::table_view keys,
                                                std::vector<cudf::order> const& key_orders,
                                                std::vector<cudf::null_order> const& null_orders,
                                                cudf::size_type keep_rows,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref memory_resource)
{
  auto indices = cudf::sorted_order(keys, key_orders, null_orders, stream, memory_resource);

  auto indices_view = indices->view();
  if (keep_rows < indices_view.size()) {
    // Views into `indices`, which outlives the gather below.
    indices_view = cudf::slice(indices_view, {0, keep_rows}, stream).front();
  }

  return cudf::gather(
    input, indices_view, cudf::out_of_bounds_policy::DONT_CHECK, stream, memory_resource);
}

std::unique_ptr<cudf::table> compute_top_n_table(
  cudf::table_view input,
  duckdb::vector<duckdb::BoundOrderByNode> const& orders,
  std::size_t limit,
  std::size_t offset,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref memory_resource)
{
  if (limit == 0 || input.num_rows() == 0) { return duckdb::make_empty_like(input); }
  if (orders.empty()) { throw internal_exception("TopN requires at least one ordering key"); }

  auto const keep_rows = static_cast<cudf::size_type>(std::min<std::size_t>(
    static_cast<std::size_t>(input.num_rows()), candidate_row_cap(offset, limit)));
  if (keep_rows == 0) { return duckdb::make_empty_like(input); }

  std::unique_ptr<cudf::table> kept;
  if (orders.size() == 1) {
    auto const& ord = orders[0];
    if (ord.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("TopN only supports bound reference expressions");
    }
    auto const idx =
      static_cast<cudf::size_type>(ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
    if (idx < 0 || idx >= input.num_columns()) {
      throw internal_exception("TopN order index out of range");
    }

    auto order    = to_cudf_order(ord.type);
    auto null_ord = to_cudf_null_order(ord.type, ord.null_order);
    if (input.column(idx).has_nulls()) {
      // cudf::top_k_order takes no null_order, so it cannot honor SQL NULLS FIRST/LAST when
      // selecting which rows are in the top k (it treats NULLs as the largest value). For a
      // nullable key, fall back to a sort that honors null placement, then take the top k.
      kept = sorted_order_top_k(input,
                                cudf::table_view({input.column(idx)}),
                                {order},
                                {null_ord},
                                keep_rows,
                                stream,
                                memory_resource);
    } else {
      auto indices =
        cudf::top_k_order(input.column(idx), keep_rows, order, stream, memory_resource);
      auto gathered = cudf::gather(
        input, indices->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, memory_resource);
      // top_k_order does not guarantee sorted output — sort the gathered rows
      kept = cudf::sort_by_key(gathered->view(),
                               cudf::table_view({gathered->view().column(idx)}),
                               {order},
                               {null_ord},
                               stream,
                               memory_resource);
    }
  } else {
    // Multi-key: cudf::top_k_order is single-column only, so sort the key tuple and take the top k.
    std::vector<cudf::column_view> key_views;
    key_views.reserve(orders.size());
    std::vector<cudf::order> key_orders;
    key_orders.reserve(orders.size());
    std::vector<cudf::null_order> null_orders;
    null_orders.reserve(orders.size());

    for (auto const& ord : orders) {
      if (ord.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
        throw not_implemented_exception("TopN only supports bound reference expressions");
      }
      auto const idx = static_cast<cudf::size_type>(
        ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
      if (idx < 0 || idx >= input.num_columns()) {
        throw internal_exception("TopN order index out of range");
      }
      key_views.push_back(input.column(idx));
      key_orders.push_back(to_cudf_order(ord.type));
      null_orders.push_back(to_cudf_null_order(ord.type, ord.null_order));
    }

    kept = sorted_order_top_k(input,
                              cudf::table_view(key_views),
                              key_orders,
                              null_orders,
                              keep_rows,
                              stream,
                              memory_resource);
  }

  return kept;
}

}  // namespace

sirius_physical_top_n::sirius_physical_top_n(
  duckdb::vector<sirius::logical_type> types_p,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  std::size_t limit,
  std::size_t offset,
  duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter_p,
  std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::TOP_N, std::move(types_p), estimated_cardinality),
    orders(std::move(orders)),
    limit(limit),
    offset(offset),
    dynamic_filter(std::move(dynamic_filter_p))
{
}

sirius_physical_top_n::~sirius_physical_top_n() {}

std::unique_ptr<operator_data> sirius_physical_top_n::execute(const operator_data& input_data,
                                                              rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_top_n::execute"};
  auto& input        = dynamic_cast<const pipelineable_operator_data&>(input_data);
  auto input_batches = input.get_read_only_batches();
  if (limit == 0 || input_batches.empty()) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  // Keeping each batch's top `offset + limit` candidates is sufficient; merge applies `offset`
  // once.
  auto const candidate_cap = candidate_row_cap(offset, limit);
  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  outputs.reserve(input_batches.size());
  for (auto& batch : input_batches) {
    auto const* data = batch.get_data();
    auto* space      = batch.get_memory_space();
    if (data == nullptr || space == nullptr) { continue; }

    auto const input_table_view =
      data->cast<cucascade::gpu_table_representation>().get_table_view();
    if (input_table_view.num_rows() == 0) { continue; }

    if (static_cast<std::size_t>(input_table_view.num_rows()) <= candidate_cap) {
      outputs.push_back(cucascade::data_batch::to_idle(std::move(batch)));
      continue;
    }

    auto candidates = compute_top_n_table(
      input_table_view, orders, limit, offset, stream, space->get_default_allocator());
    // STREAM-LINEAGE: candidates were last written on `stream`.
    outputs.push_back(
      sirius::make_data_batch(std::move(candidates), *space, stream, batch_telemetry()));
  }
  return std::make_unique<pipelineable_operator_data>(std::move(outputs));
}

void sirius_physical_top_n_merge::build_pipelines(pipeline::sirius_pipeline& current,
                                                  pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // The child sink still creates the upstream pipeline boundary.
  if (fuse_into_parent()) {
    D_ASSERT(children.size() == 1);
    meta_pipeline.get_state().add_pipeline_operator(current, *this);
    children[0]->build_pipelines(current, meta_pipeline);
    return;
  }
  sirius_physical_operator::build_pipelines(current, meta_pipeline);
}

sirius_physical_top_n_merge::sirius_physical_top_n_merge(sirius_physical_top_n* top_n)
  : sirius_physical_top_n_merge(
      top_n->types,                // copied by value
      copy_orders(top_n->orders),  // deep copy
      top_n->limit,                // primitive
      top_n->offset,               // primitive
      top_n->dynamic_filter,       // shared_ptr - shares ownership (reference count increases)
      top_n->estimated_cardinality)
{
  child_op = top_n;
}

sirius_physical_top_n_merge::sirius_physical_top_n_merge(
  duckdb::vector<sirius::logical_type> types_p,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  std::size_t limit,
  std::size_t offset,
  duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter_p,
  std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_TOP_N, std::move(types_p), estimated_cardinality),
    orders(std::move(orders)),
    limit(limit),
    offset(offset),
    dynamic_filter(std::move(dynamic_filter_p))
{
}

std::unique_ptr<operator_data> sirius_physical_top_n_merge::execute(const operator_data& input_data,
                                                                    rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_top_n_merge::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (limit == 0) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  // INVARIANT: all input batches arrive on target_space via
  // gpu_pipeline_task::execute_pipeline_task_round ->
  // pipelineable_operator_data::prepare_for_processing -> lock_or_prepare_batch.
  // batches[0]->get_memory_space() == target_space here.
  if (input_batches.empty()) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }
  auto* space = input_batches.front().get_memory_space();

  // The read-only accessors in `input_batches` keep the table-view storage alive through selection.
  std::vector<cudf::table_view> concat_views;
  concat_views.reserve(input_batches.size());
  auto const first_input_view =
    input_batches.front().get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
  for (auto const& batch : input_batches) {
    auto const view =
      batch.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
    if (view.num_rows() != 0) { concat_views.push_back(view); }
  }

  std::unique_ptr<cudf::table> combined;
  // Use the first input's schema when every batch is empty.
  auto combined_view = first_input_view;
  if (concat_views.size() == 1) {
    combined_view = concat_views.front();
  } else if (concat_views.size() > 1) {
    combined      = cudf::concatenate(concat_views, stream, space->get_default_allocator());
    combined_view = combined->view();
  }

  auto output_table = compute_top_n_table(
    combined_view, orders, limit, offset, stream, space->get_default_allocator());
  if (static_cast<std::size_t>(output_table->num_rows()) <= offset) {
    output_table = duckdb::make_empty_like(output_table->view());
  } else if (offset > 0) {
    auto out_start = static_cast<cudf::size_type>(offset);
    auto out_slices =
      cudf::slice(output_table->view(), {out_start, output_table->num_rows()}, stream);
    output_table =
      std::make_unique<cudf::table>(out_slices.front(), stream, space->get_default_allocator());
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  // STREAM-LINEAGE: compute_top_n_table + slice write on `stream`; the
  // constructor records the writer event for downstream cross-device readers.
  auto output_repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(output_table), *space, stream);
  std::unique_ptr<cucascade::idata_representation> output_data = std::move(output_repr);
  auto const batch_id                                          = ::sirius::get_next_batch_id();
  outputs.push_back(cucascade::data_batch::make(
    batch_id,
    std::move(output_data),
    telemetry::quent_data_batch_probe::create(batch_telemetry(), batch_id)));
  return std::make_unique<pipelineable_operator_data>(outputs);
}

std::unique_ptr<operator_data> sirius_physical_top_n_merge::get_next_task_input_data()
{
  // we need to lock, then pull all the batches from one partition and return them, and increment
  // the partition index
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
