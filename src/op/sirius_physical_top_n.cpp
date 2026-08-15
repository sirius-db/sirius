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
#include "op/dynamic_filter/top_n_boundary_filter.hpp"
#include "op/dynamic_filter/top_n_threshold_coordinator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/resource_ref.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

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

//===----------------------------------------------------------------------===//
// Sink self-consumption seam (Stage-1 Top-N threshold refinement)
//===----------------------------------------------------------------------===//
// These helpers run only when the planner attached a threshold coordinator, which guarantees
// every ORDER BY key is a bound reference and checked K fits cudf::size_type.

/// The ORDER BY keys' input-column indices, in key order. The prefilter consumes the indices
/// before compute_top_n_table's own checks run, so range-validate here with the same diagnostic.
std::vector<cudf::size_type> key_column_indices(
  duckdb::vector<duckdb::BoundOrderByNode> const& orders, cudf::size_type num_columns)
{
  std::vector<cudf::size_type> indices;
  indices.reserve(orders.size());
  for (auto const& ord : orders) {
    auto const idx =
      static_cast<cudf::size_type>(ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
    if (idx < 0 || idx >= num_columns) {
      throw internal_exception("TopN order index out of range");
    }
    indices.push_back(idx);
  }
  return indices;
}

/// Marshal the shared boundary into by-value kernel launch parameters through the shared detail
/// marshaller: the full tuple as the strict form when every key is admitted, only component 0 as
/// the inclusive first-key form otherwise (the marshaller itself degrades an over-wide boundary
/// to the inclusive prefix form).
detail::boundary_filter_params make_boundary_filter_params(
  exact_host_key_tuple const& boundary,
  std::span<top_n_key_semantics const> keys,
  bool lex_admitted)
{
  return detail::make_boundary_filter_params(
    boundary, keys, lex_admitted ? keys.size() : std::size_t{1}, lex_admitted);
}

/// Read row @p row of @p col to an exact host value. The typed `is_valid`/`value` reads
/// synchronize @p stream, so the returned value is the completed host copy the witness-handoff
/// contract requires. A null element, a column whose type is not exactly @p storage_type, or a
/// storage type outside the allowlist yields a disengaged component.
std::optional<exact_host_scalar> extract_boundary_component(cudf::column_view const& col,
                                                            cudf::size_type row,
                                                            cudf::data_type storage_type,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  // Full `data_type` equality, not just the id: the rep this reads is labelled with
  // @p storage_type everywhere downstream -- the published fixed-point literal, the kernel's
  // component width, and the sink prefilter -- so reading it from a column of a different type
  // would attach the wrong label. Before fixed-point keys that could only happen across differing
  // ids, where the `static_cast` below is already undefined; now the ids can agree while the
  // scales differ, which is well-defined and silently wrong. Refusing yields a disengaged
  // component, which only ever loosens the boundary.
  if (col.type() != storage_type) { return std::nullopt; }

  auto const read_numeric = [&]<typename T>() -> std::optional<exact_host_scalar> {
    auto const element = cudf::get_element(col, row, stream, mr);
    auto const& typed  = static_cast<cudf::numeric_scalar<T> const&>(*element);
    if (!typed.is_valid(stream)) { return std::nullopt; }
    return exact_host_scalar{typed.value(stream), storage_type};
  };
  switch (storage_type.id()) {
    case cudf::type_id::INT8: return read_numeric.template operator()<std::int8_t>();
    case cudf::type_id::INT16: return read_numeric.template operator()<std::int16_t>();
    case cudf::type_id::INT32: return read_numeric.template operator()<std::int32_t>();
    case cudf::type_id::INT64: return read_numeric.template operator()<std::int64_t>();
    case cudf::type_id::TIMESTAMP_DAYS: {
      auto const element = cudf::get_element(col, row, stream, mr);
      auto const& typed  = static_cast<cudf::timestamp_scalar<cudf::timestamp_D> const&>(*element);
      if (!typed.is_valid(stream)) { return std::nullopt; }
      return exact_host_scalar{
        static_cast<std::int32_t>(typed.value(stream).time_since_epoch().count()), storage_type};
    }
    // A fixed-point key is recorded as its scaled integer; `storage_type` carries the scale, so
    // the pair is exactly the value and nothing is rescaled.
    case cudf::type_id::DECIMAL32: {
      auto const element = cudf::get_element(col, row, stream, mr);
      auto const& typed =
        static_cast<cudf::fixed_point_scalar<numeric::decimal32> const&>(*element);
      if (!typed.is_valid(stream)) { return std::nullopt; }
      return exact_host_scalar{typed.value(stream), storage_type};
    }
    case cudf::type_id::DECIMAL64: {
      auto const element = cudf::get_element(col, row, stream, mr);
      auto const& typed =
        static_cast<cudf::fixed_point_scalar<numeric::decimal64> const&>(*element);
      if (!typed.is_valid(stream)) { return std::nullopt; }
      return exact_host_scalar{typed.value(stream), storage_type};
    }
    case cudf::type_id::DECIMAL128: {
      auto const element = cudf::get_element(col, row, stream, mr);
      auto const& typed =
        static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(*element);
      if (!typed.is_valid(stream)) { return std::nullopt; }
      return exact_host_scalar{typed.value(stream), storage_type};
    }
    default: return std::nullopt;
  }
}

/// Exact host key tuple of row `K - 1` of a K-row local result, one component per ORDER BY key.
exact_host_key_tuple extract_boundary_tuple(cudf::table_view const& output,
                                            std::span<top_n_key_semantics const> keys,
                                            std::span<cudf::size_type const> key_columns,
                                            cudf::size_type boundary_row,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  std::vector<std::optional<exact_host_scalar>> components;
  components.reserve(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    components.push_back(extract_boundary_component(
      output.column(key_columns[i]), boundary_row, keys[i].storage_type, stream, mr));
  }
  return exact_host_key_tuple{std::move(components)};
}

}  // namespace

sirius_physical_top_n::sirius_physical_top_n(
  duckdb::vector<sirius::logical_type> types_p,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  std::size_t limit,
  std::size_t offset,
  duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter_p,
  std::size_t estimated_cardinality,
  double gate_keep_threshold)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::TOP_N, std::move(types_p), estimated_cardinality),
    orders(std::move(orders)),
    limit(limit),
    offset(offset),
    dynamic_filter(std::move(dynamic_filter_p)),
    prefilter_gate(gate_keep_threshold)
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
  auto const memory_resource = space->get_default_allocator();

  std::vector<cudf::size_type> key_columns;
  if (threshold_coordinator) {
    key_columns = key_column_indices(orders, input_table_view.num_columns());
  }

  // Sink self-consumption: drop rows the shared boundary already excludes before the local sort,
  // in one fused kernel pass. A stale boundary prunes less, never more; the keep-ratio gate
  // stops unselective prefilters and one boundary tightening re-arms one measurement.
  std::unique_ptr<cudf::table> prefiltered;  // backs input_table_view when rows were dropped
  if (threshold_coordinator && input_table_view.num_rows() > 0) {
    auto const observed_updates = threshold_coordinator->boundary_update_count();
    if (prefilter_gate.applicable(observed_updates)) {
      if (auto const boundary = threshold_coordinator->tightest_boundary()) {
        auto const params = make_boundary_filter_params(
          *boundary, threshold_coordinator->keys(), threshold_coordinator->lex_admitted());
        auto result = detail::apply_boundary_filter(
          input_table_view, key_columns, params, stream, memory_resource);
        auto const rows_in  = static_cast<std::size_t>(input_table_view.num_rows());
        auto const rows_out = static_cast<std::size_t>(result.rows_kept);
        threshold_coordinator->record_prefilter(rows_in, rows_out);
        prefilter_gate.record_keep_ratio(rows_in, rows_out, observed_updates);
        if (!prefilter_gate.applicable(observed_updates)) {
          threshold_coordinator->record_prefilter_disabled();
        }
        // A null result is the all-pass fast path: keep the original view, copy-free.
        if (result.filtered) {
          prefiltered      = std::move(result.filtered);
          input_table_view = prefiltered->view();
        }
      }
    }
  }

  auto output_table =
    compute_top_n_table(input_table_view, orders, limit, offset, stream, memory_resource);
  // ro released at end of function

  // Witness extraction: a local result owning exactly K rows proves row K-1's key tuple is a
  // legal boundary; sub-K results publish nothing.
  std::optional<exact_host_key_tuple> boundary_tuple;
  if (threshold_coordinator &&
      static_cast<std::size_t>(output_table->num_rows()) == threshold_coordinator->k()) {
    boundary_tuple = extract_boundary_tuple(output_table->view(),
                                            threshold_coordinator->keys(),
                                            key_columns,
                                            output_table->num_rows() - 1,
                                            stream,
                                            memory_resource);
  }

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

  if (boundary_tuple) {
    // The host reads completed inside the extraction, and the witness co-owns the K-row result
    // batch. The offer's own result needs no handling here: publication, coalescing, and
    // rejection are all the coordinator's business and are recorded in its counters.
    (void)threshold_coordinator->offer({std::move(*boundary_tuple), outputs.back()});
  }
  return std::make_unique<pipelineable_operator_data>(outputs);
}

void sirius_physical_top_n_merge::on_finalize_operator()
{
  // Producer-side drain after the merge's pipeline completes -- after the last merge execute in
  // both the fused and the standalone pipeline shapes. Idempotent. It blocks this finalization
  // task until the publisher is quiescent, never a scan consumer.
  if (threshold_coordinator) { threshold_coordinator->finish(); }
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
  // shared_ptr - shares the execution coordinator so on_finalize_operator can drain it. The
  // prefilter gate stays per-operator: the merge never prefilters.
  threshold_coordinator = top_n->threshold_coordinator;
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
  // so the underlying table_views remain valid.
  std::vector<cucascade::read_only_data_batch> ro_views;
  ro_views.reserve(input_batches.size());
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
