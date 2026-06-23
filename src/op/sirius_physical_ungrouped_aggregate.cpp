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

#include "op/sirius_physical_ungrouped_aggregate.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "sirius/exception.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <algorithm>
#include <limits>
#include <optional>

namespace sirius {
namespace op {

sirius_physical_ungrouped_aggregate::sirius_physical_ungrouped_aggregate(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  std::size_t estimated_cardinality,
  duckdb::TupleDataValidityType /*distinct_validity*/)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE, std::move(types), estimated_cardinality),
    aggregates(std::move(expressions))
{
  // Sirius's GPU aggregate path does not support DISTINCT aggregates — see the throw in
  // build_aggregate_layout. DistinctAggregateCollectionInfo / DistinctAggregateData are not
  // wired into any subsequent code path here, so we skip populating them.
}

namespace {

// Map LogicalType to cudf::data_type using existing utility
cudf::data_type ToCudfType(const duckdb::LogicalType& t) { return duckdb::GetCudfType(t); }

template <typename ScalarType>
ScalarType const& scalar_cast(const cudf::scalar& s)
{
  return static_cast<ScalarType const&>(s);
}

template <typename ScalarType>
ScalarType& scalar_cast(cudf::scalar& s)
{
  return static_cast<ScalarType&>(s);
}

template <typename T>
std::unique_ptr<cudf::scalar> make_numeric_scalar_with_value(cudf::data_type type,
                                                             T value,
                                                             rmm::cuda_stream_view stream)
{
  auto out = cudf::make_numeric_scalar(type, stream);
  scalar_cast<cudf::numeric_scalar<T>>(*out).set_value(value, stream);
  return out;
}

enum class aggregate_kind { SUM, MIN, MAX, COUNT, COUNT_STAR, AVG, FIRST };

struct aggregate_spec {
  aggregate_kind kind;
  int input_idx;
  duckdb::LogicalType return_type;
  size_t local_sum_idx;
  size_t local_count_idx;
};

struct aggregate_layout {
  std::vector<aggregate_spec> aggregates;
  std::vector<duckdb::LogicalType> local_types;
  std::vector<cudf::aggregation::Kind> merge_kinds;
  std::vector<std::optional<cudf::size_type>>
    merge_nth_index;  // when merge_kinds[i] == NTH_ELEMENT
  bool has_avg = false;
};

aggregate_layout build_aggregate_layout(
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& aggregates)
{
  aggregate_layout layout;
  size_t local_idx = 0;
  layout.aggregates.reserve(aggregates.size());

  for (size_t i = 0; i < aggregates.size(); ++i) {
    auto const& agg = sirius::ast::require_aggregate(aggregates[i].get(), "ungrouped aggregate");
    if (agg.distinct()) {
      throw not_implemented_exception("Distinct aggregates not supported in GPU path yet");
    }
    auto const& children = agg.arguments();
    if (children.size() > 1) {
      throw not_implemented_exception("Aggregates with multiple children not supported yet");
    }

    auto agg_return_type = sirius::to_duckdb(agg.return_type());
    auto child_ref_index = [&]() { return children[0]->as_reference().column_index; };

    aggregate_spec spec;
    spec.input_idx       = -1;
    spec.return_type     = agg_return_type;
    spec.local_sum_idx   = std::numeric_limits<size_t>::max();
    spec.local_count_idx = std::numeric_limits<size_t>::max();

    switch (agg.function()) {
      case sirius::aggregate_id::count_star:
        spec.kind          = aggregate_kind::COUNT_STAR;
        spec.return_type   = duckdb::LogicalType::BIGINT;
        spec.local_sum_idx = local_idx++;
        layout.local_types.push_back(duckdb::LogicalType::BIGINT);
        layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
        layout.merge_nth_index.push_back(std::nullopt);
        break;
      case sirius::aggregate_id::count:
        if (children.empty()) {
          throw not_implemented_exception("count() without arguments not supported");
        }
        spec.kind          = aggregate_kind::COUNT;
        spec.return_type   = duckdb::LogicalType::BIGINT;
        spec.input_idx     = child_ref_index();
        spec.local_sum_idx = local_idx++;
        layout.local_types.push_back(duckdb::LogicalType::BIGINT);
        layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
        layout.merge_nth_index.push_back(std::nullopt);
        break;
      case sirius::aggregate_id::sum:
      case sirius::aggregate_id::sum_no_overflow:
        if (children.empty()) {
          throw not_implemented_exception("sum() without arguments not supported");
        }
        spec.kind          = aggregate_kind::SUM;
        spec.input_idx     = child_ref_index();
        spec.local_sum_idx = local_idx++;
        layout.local_types.push_back(agg_return_type);
        layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
        layout.merge_nth_index.push_back(std::nullopt);
        break;
      case sirius::aggregate_id::min:
        if (children.empty()) {
          throw not_implemented_exception("min() without arguments not supported");
        }
        spec.kind          = aggregate_kind::MIN;
        spec.input_idx     = child_ref_index();
        spec.local_sum_idx = local_idx++;
        layout.local_types.push_back(agg_return_type);
        layout.merge_kinds.push_back(cudf::aggregation::Kind::MIN);
        layout.merge_nth_index.push_back(std::nullopt);
        break;
      case sirius::aggregate_id::max:
        if (children.empty()) {
          throw not_implemented_exception("max() without arguments not supported");
        }
        spec.kind          = aggregate_kind::MAX;
        spec.input_idx     = child_ref_index();
        spec.local_sum_idx = local_idx++;
        layout.local_types.push_back(agg_return_type);
        layout.merge_kinds.push_back(cudf::aggregation::Kind::MAX);
        layout.merge_nth_index.push_back(std::nullopt);
        break;
      case sirius::aggregate_id::avg:
        if (children.empty()) {
          throw not_implemented_exception("avg() without arguments not supported");
        }
        spec.kind          = aggregate_kind::AVG;
        spec.input_idx     = child_ref_index();
        spec.local_sum_idx = local_idx++;
        layout.local_types.push_back(agg_return_type);
        layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
        layout.merge_nth_index.push_back(std::nullopt);
        spec.local_count_idx = local_idx++;
        layout.local_types.push_back(duckdb::LogicalType::BIGINT);
        layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
        layout.merge_nth_index.push_back(std::nullopt);
        layout.has_avg = true;
        break;
      case sirius::aggregate_id::first:
        spec.kind          = aggregate_kind::FIRST;
        spec.input_idx     = child_ref_index();
        spec.local_sum_idx = local_idx++;
        layout.local_types.push_back(agg_return_type);
        layout.merge_kinds.push_back(cudf::aggregation::Kind::NTH_ELEMENT);
        layout.merge_nth_index.push_back(0);  // first element
        break;
      default:
        throw not_implemented_exception(
          "Aggregate not supported: {}",
          std::string{sirius::to_duckdb_aggregate_name(agg.function())});
    }

    layout.aggregates.push_back(std::move(spec));
  }

  return layout;
}

std::unique_ptr<cudf::column> make_avg_column(const cudf::column_view& sum_view,
                                              const cudf::column_view& count_view,
                                              const duckdb::LogicalType& return_type,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref memory_resource)
{
  // The merged sum/count columns are already single-row, so divide them on-device
  // rather than copying scalars to the host and converting through long double.
  // This mirrors sirius_physical_grouped_aggregate_merge and avoids the precision
  // loss of the host-side decimal->long double->decimal round trips.
  auto out_type = ToCudfType(return_type);

  if (sirius::IsCudfTypeDecimal(out_type)) {
    // DECIMAL output: divide directly in fixed-point to preserve precision.
    return cudf::binary_operation(
      sum_view, count_view, cudf::binary_operator::DIV, out_type, stream, memory_resource);
  }

  // Non-DECIMAL output (typically DOUBLE): cast both operands to FLOAT64 and divide.
  auto sum_f64 =
    cudf::cast(sum_view, cudf::data_type{cudf::type_id::FLOAT64}, stream, memory_resource);
  auto count_f64 =
    cudf::cast(count_view, cudf::data_type{cudf::type_id::FLOAT64}, stream, memory_resource);
  auto avg_f64 = cudf::binary_operation(sum_f64->view(),
                                        count_f64->view(),
                                        cudf::binary_operator::DIV,
                                        cudf::data_type{cudf::type_id::FLOAT64},
                                        stream,
                                        memory_resource);
  if (out_type.id() == cudf::type_id::FLOAT64) { return avg_f64; }
  // Defensive: cast to the requested numeric return type if it isn't already FLOAT64.
  return cudf::cast(avg_f64->view(), out_type, stream, memory_resource);
}

}  // namespace

std::unique_ptr<operator_data> sirius_physical_ungrouped_aggregate::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_ungrouped_aggregate::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (aggregates.empty()) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  auto layout = build_aggregate_layout(aggregates);
  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  outputs.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    auto* space = batch.get_memory_space();
    if (!space) { continue; }

    auto view = batch.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(layout.local_types.size());

    for (auto const& spec : layout.aggregates) {
      switch (spec.kind) {
        case aggregate_kind::COUNT_STAR: {
          auto scalar = make_numeric_scalar_with_value<int64_t>(
            cudf::data_type{cudf::type_id::INT64}, static_cast<int64_t>(view.num_rows()), stream);
          cols.push_back(cudf::make_column_from_scalar(*scalar, 1, stream));
          break;
        }
        case aggregate_kind::COUNT: {
          auto col    = view.column(static_cast<cudf::size_type>(spec.input_idx));
          auto agg_op = cudf::make_count_aggregation<cudf::reduce_aggregation>();
          auto scalar =
            cudf::reduce(col, *agg_op, cudf::data_type(cudf::type_id::INT64), std::nullopt, stream);
          cols.push_back(cudf::make_column_from_scalar(*scalar, 1, stream));
          break;
        }
        case aggregate_kind::FIRST: {
          auto col = view.column(static_cast<cudf::size_type>(spec.input_idx));
          std::unique_ptr<cudf::scalar> first_scalar;
          if (col.size() == 0) {
            first_scalar = cudf::make_fixed_width_scalar(
              col.type(), stream, cudf::get_current_device_resource_ref());
            first_scalar->set_valid_async(false, stream);
          } else {
            first_scalar =
              cudf::get_element(col, 0, stream, cudf::get_current_device_resource_ref());
          }
          cols.push_back(cudf::make_column_from_scalar(*first_scalar, 1, stream));
          break;
        }
        case aggregate_kind::SUM:
        case aggregate_kind::MIN:
        case aggregate_kind::MAX:
        case aggregate_kind::AVG: {
          auto col      = view.column(static_cast<cudf::size_type>(spec.input_idx));
          auto out_type = ToCudfType(spec.return_type);
          std::unique_ptr<cudf::reduce_aggregation> agg_op;
          if (spec.kind == aggregate_kind::MIN) {
            agg_op = cudf::make_min_aggregation<cudf::reduce_aggregation>();
          } else if (spec.kind == aggregate_kind::MAX) {
            agg_op = cudf::make_max_aggregation<cudf::reduce_aggregation>();
          } else {
            agg_op = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
          }
          // cuDF requires output type == input type for fixed-point (decimal) reductions.
          // For AVG we use input type and apply return type in the merge step (SUM/COUNT).
          // For SUM we widen (expected by duckdb) before the aggregation to avoid overflow.
          bool is_decimal = sirius::IsCudfTypeDecimal(col.type());

          std::unique_ptr<cudf::column> casted_col;
          if (spec.kind == aggregate_kind::SUM) {
            if (col.type().id() == cudf::type_id::DECIMAL32) {
              casted_col = cudf::cast(
                col, cudf::data_type(cudf::type_id::DECIMAL64, col.type().scale()), stream);
              col = casted_col->view();
            }
            if (col.type().id() == cudf::type_id::DECIMAL64) {
              casted_col = cudf::cast(
                col, cudf::data_type(cudf::type_id::DECIMAL128, col.type().scale()), stream);
              col = casted_col->view();
            }
          }
          if (is_decimal) {
            // cuDF requires output type == input type for fixed-point reductions.
            out_type = col.type();
          } else if (spec.kind == aggregate_kind::AVG) {
            // Widen small integer types to INT64 so the partial sum is stored as INT64.
            // merge_ungrouped_aggregate sums INT64 partial sums without cross-type reduction,
            // which avoids cuDF cross-type reduce issues that produce wrong results.
            if (col.type().id() == cudf::type_id::INT8 || col.type().id() == cudf::type_id::INT16 ||
                col.type().id() == cudf::type_id::INT32) {
              casted_col = cudf::cast(col, cudf::data_type(cudf::type_id::INT64), stream);
              col        = casted_col->view();
            }
            out_type = col.type();
          }
          auto scalar = cudf::reduce(col, *agg_op, out_type, std::nullopt, stream);
          cols.push_back(cudf::make_column_from_scalar(*scalar, 1, stream));
          if (spec.kind == aggregate_kind::AVG) {
            auto count_scalar = make_numeric_scalar_with_value<int64_t>(
              cudf::data_type{cudf::type_id::INT64}, static_cast<int64_t>(view.num_rows()), stream);
            cols.push_back(cudf::make_column_from_scalar(*count_scalar, 1, stream));
          }
          break;
        }
      }
    }

    auto out_table = std::make_unique<cudf::table>(std::move(cols), stream);
    // STREAM-LINEAGE: cudf::table ctor + cudf::make_column_from_scalar wrote
    // on `stream`; the constructor records the writer event for downstream
    // cross-device readers.
    auto out_repr =
      std::make_unique<cucascade::gpu_table_representation>(std::move(out_table), *space, stream);
    std::unique_ptr<cucascade::idata_representation> output_data = std::move(out_repr);
    auto const batch_id                                          = ::sirius::get_next_batch_id();

    // Record output_batch_id -> source (scan) batch_id so the merge operator can restore
    // scan order when selecting NTH_ELEMENT(0) for first() aggregates.
    {
      std::lock_guard<std::mutex> lg(source_id_mutex_);
      output_to_source_id_[batch_id] = batch.get_batch_id();
    }

    outputs.push_back(std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data)));
  }

  return std::make_unique<pipelineable_operator_data>(outputs);
}

uint64_t sirius_physical_ungrouped_aggregate::get_source_batch_id(uint64_t output_batch_id) const
{
  std::lock_guard<std::mutex> lg(source_id_mutex_);
  auto it = output_to_source_id_.find(output_batch_id);
  return (it != output_to_source_id_.end()) ? it->second : output_batch_id;
}

// Helper to deep copy the aggregate AST expressions (used by the merge overload below).
static duckdb::vector<std::unique_ptr<sirius::ast::node>> copy_expressions(
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& src)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> result;
  result.reserve(src.size());
  for (const auto& expr : src) {
    // node is move-only and aggregate nodes cannot round-trip through to_duckdb,
    // so deep-clone the AST node directly.
    if (expr == nullptr) {
      throw not_implemented_exception("copy_expressions: cannot clone a null aggregate expression");
    }
    result.push_back(sirius::ast::clone(*expr));
  }
  return result;
}

sirius_physical_ungrouped_aggregate_merge::sirius_physical_ungrouped_aggregate_merge(
  sirius_physical_ungrouped_aggregate* ungrouped_aggregate)
  : sirius_physical_ungrouped_aggregate_merge(
      ungrouped_aggregate->types,                         // copied by value
      copy_expressions(ungrouped_aggregate->aggregates),  // deep copy
      ungrouped_aggregate->estimated_cardinality,
      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES)  // default - not stored in source
{
  child_op = ungrouped_aggregate;
}

sirius_physical_ungrouped_aggregate_merge::sirius_physical_ungrouped_aggregate_merge(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  std::size_t estimated_cardinality,
  duckdb::TupleDataValidityType /*distinct_validity*/)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_AGGREGATE, std::move(types), estimated_cardinality),
    aggregates(std::move(expressions))
{
}

std::unique_ptr<operator_data> sirius_physical_ungrouped_aggregate_merge::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_ungrouped_aggregate_merge::execute"};
  auto& input        = dynamic_cast<const pipelineable_operator_data&>(input_data);
  auto input_batches = input.get_read_only_batches();
  if (aggregates.empty()) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  if (input_batches.empty()) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  cucascade::memory::memory_space* space = input_batches[0].get_memory_space();
  if (space == nullptr) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  auto layout = build_aggregate_layout(aggregates);

  // For NTH_ELEMENT (first()) aggregates, sort partial results by their source scan
  // batch ID so that NTH_ELEMENT(0) always selects the value from the earliest-scanned
  // batch, matching DuckDB's first() semantics. Without this, concurrent pipeline tasks
  // push results in non-deterministic completion order and first() may return a value
  // from any batch.
  if (input_batches.size() > 1) {
    bool has_nth_element = std::any_of(
      layout.merge_kinds.begin(), layout.merge_kinds.end(), [](cudf::aggregation::Kind k) {
        return k == cudf::aggregation::Kind::NTH_ELEMENT;
      });
    if (has_nth_element) {
      auto* local_agg = dynamic_cast<sirius_physical_ungrouped_aggregate*>(child_op);
      if (local_agg) {
        std::stable_sort(input_batches.begin(),
                         input_batches.end(),
                         [&local_agg](const cucascade::read_only_data_batch& a,
                                      const cucascade::read_only_data_batch& b) {
                           return local_agg->get_source_batch_id(a.get_batch_id()) <
                                  local_agg->get_source_batch_id(b.get_batch_id());
                         });
      }
    }
  }

  std::shared_ptr<cucascade::data_batch> merged_batch;
  if (input_batches.size() == 1) {
    merged_batch = cucascade::data_batch::to_idle(std::move(input_batches[0]));
  } else {
    merged_batch = gpu_merge_impl::merge_ungrouped_aggregate(
      input_batches, layout.merge_kinds, layout.merge_nth_index, stream, *space);
  }

  if (!layout.has_avg) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(merged_batch)});
  }

  // Acquire read access to merged batch to extract table
  auto merged_ro = merged_batch->to_read_only();
  auto merged_view =
    merged_ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();

  std::vector<std::unique_ptr<cudf::column>> output_cols;
  output_cols.reserve(layout.aggregates.size());
  for (auto const& spec : layout.aggregates) {
    if (spec.kind == aggregate_kind::AVG) {
      auto sum_view   = merged_view.column(static_cast<cudf::size_type>(spec.local_sum_idx));
      auto count_view = merged_view.column(static_cast<cudf::size_type>(spec.local_count_idx));
      output_cols.push_back(make_avg_column(
        sum_view, count_view, spec.return_type, stream, cudf::get_current_device_resource_ref()));
    } else {
      auto col_view = merged_view.column(static_cast<cudf::size_type>(spec.local_sum_idx));
      output_cols.push_back(std::make_unique<cudf::column>(col_view, stream));
    }
  }

  auto out_table = std::make_unique<cudf::table>(
    std::move(output_cols), stream, cudf::get_current_device_resource_ref());
  // STREAM-LINEAGE: cudf::table ctor + make_avg_column write on `stream`;
  // the constructor records the writer event for downstream cross-device
  // readers.
  auto out_repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(out_table), *space, stream);
  std::unique_ptr<cucascade::idata_representation> output_data = std::move(out_repr);
  auto const batch_id                                          = ::sirius::get_next_batch_id();
  auto output_batch = std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data));

  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(output_batch)});
}

std::unique_ptr<operator_data> sirius_physical_ungrouped_aggregate_merge::get_next_task_input_data()
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
