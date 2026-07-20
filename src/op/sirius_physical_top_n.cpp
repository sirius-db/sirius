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
#include "sirius/exception.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/sorting.hpp>

#include <rmm/resource_ref.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>

#include <algorithm>
#include <memory>

namespace sirius {
namespace op {

namespace {

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

  auto const keep_rows =
    std::min<cudf::size_type>(input.num_rows(), static_cast<cudf::size_type>(offset + limit));
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
      // nullable key, fall back to a full sort that honors null placement, then slice the top k.
      auto sorted = cudf::sort_by_key(
        input, cudf::table_view({input.column(idx)}), {order}, {null_ord}, stream, memory_resource);
      if (keep_rows == sorted->num_rows()) {
        kept = std::move(sorted);
      } else {
        auto slices = cudf::slice(sorted->view(), {0, keep_rows}, stream);
        kept        = std::make_unique<cudf::table>(slices.front(), stream, memory_resource);
      }
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
    // Multi-key: fall back to full sort_by_key
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

    auto keys_table = cudf::table_view(key_views);
    auto sorted =
      cudf::sort_by_key(input, keys_table, key_orders, null_orders, stream, memory_resource);

    if (keep_rows == sorted->num_rows()) {
      kept = std::move(sorted);
    } else {
      auto slices = cudf::slice(sorted->view(), {0, keep_rows}, stream);
      kept        = std::make_unique<cudf::table>(slices.front(), stream, memory_resource);
    }
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
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (limit == 0) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  if (input_batches.empty()) {
    return std::make_unique<pipelineable_operator_data>();
  } else if (input_batches.size() > 1) {
    throw internal_exception("TopN expects a single input batch per execution");
  }

  auto input_batch = input_batches[0];
  auto* space      = input_batch.get_memory_space();
  if (space == nullptr) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  auto input_table_view =
    input_batch.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
  auto output_table = compute_top_n_table(
    input_table_view, orders, limit, offset, stream, space->get_default_allocator());
  // ro released at end of function

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  // STREAM-LINEAGE: compute_top_n_table writes the output table on `stream`;
  // the constructor records the writer event so cross-device readers honor
  // the producer-consumer ordering.
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
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    space = batch.get_memory_space();
    break;
  }
  if (space == nullptr) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  // R1 — read-only accessors held in a vector for the duration of cudf::concatenate
  // so the underlying table_views remain valid. The old code declared ro_views
  // and reserved capacity but never populated it — the loop only pushed to
  // concat_views. The table_views stayed valid only because input_batches (a
  // const&) kept the owning batches alive; if ownership ever changes (e.g.
  // batches released before concatenate completes), the views dangle.
  // Populate ro_views alongside concat_views so the read-only accessors pin
  // the data for concatenate's lifetime.
  std::vector<cucascade::read_only_data_batch> ro_views;
  ro_views.reserve(input_batches.size());
  std::vector<cudf::table_view> concat_views;
  for (auto const& batch : input_batches) {
    auto ro = batch.to_read_only();
    concat_views.push_back(
      ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view());
    ro_views.push_back(std::move(ro));
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

  auto output_table = compute_top_n_table(
    combined->view(), orders, limit, offset, stream, space->get_default_allocator());
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
  // Drain every batch currently available in this operator's single input repo.
  // top_n_merge has one input port, so all buffered batches belong to the same
  // logical stream; no per-partition indexing is required.
  std::lock_guard<std::mutex> lg(lock);
  std::vector<::std::shared_ptr<::cucascade::data_batch>> input_batch;
  bool found_batch = true;
  if (ports.empty() || !ports.begin()->second->repo) { return nullptr; }
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
