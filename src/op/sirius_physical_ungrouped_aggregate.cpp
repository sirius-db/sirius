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
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace sirius {
namespace op {

sirius_physical_ungrouped_aggregate::sirius_physical_ungrouped_aggregate(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::idx_t estimated_cardinality,
  duckdb::TupleDataValidityType distinct_validity)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE, std::move(types), estimated_cardinality),
    aggregates(std::move(expressions))
{
  distinct_collection_info = duckdb::DistinctAggregateCollectionInfo::Create(aggregates);
  // aggregation_result       = duckdb::make_shared_ptr<GPUIntermediateRelation>(aggregates.size());
  if (!distinct_collection_info) { return; }
  distinct_data =
    duckdb::make_uniq<duckdb::DistinctAggregateData>(*distinct_collection_info, distinct_validity);
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
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& aggregates)
{
  aggregate_layout layout;
  size_t local_idx = 0;
  layout.aggregates.reserve(aggregates.size());

  for (size_t i = 0; i < aggregates.size(); ++i) {
    auto& agg = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
    if (agg.IsDistinct()) {
      throw duckdb::NotImplementedException("Distinct aggregates not supported in GPU path yet");
    }
    if (agg.children.size() > 1) {
      throw duckdb::NotImplementedException("Aggregates with multiple children not supported yet");
    }

    aggregate_spec spec;
    spec.input_idx       = -1;
    spec.return_type     = agg.return_type;
    spec.local_sum_idx   = std::numeric_limits<size_t>::max();
    spec.local_count_idx = std::numeric_limits<size_t>::max();

    const auto& fname = agg.function.name;
    if (fname == "count_star") {
      spec.kind          = aggregate_kind::COUNT_STAR;
      spec.return_type   = duckdb::LogicalType::BIGINT;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(duckdb::LogicalType::BIGINT);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
      layout.merge_nth_index.push_back(std::nullopt);
    } else if (fname == "count") {
      if (agg.children.empty()) {
        throw duckdb::NotImplementedException("count() without arguments not supported");
      }
      spec.kind          = aggregate_kind::COUNT;
      spec.return_type   = duckdb::LogicalType::BIGINT;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(duckdb::LogicalType::BIGINT);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
      layout.merge_nth_index.push_back(std::nullopt);
    } else if (fname == "sum" || fname == "sum_no_overflow") {
      if (agg.children.empty()) {
        throw duckdb::NotImplementedException("sum() without arguments not supported");
      }
      spec.kind          = aggregate_kind::SUM;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(agg.return_type);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
      layout.merge_nth_index.push_back(std::nullopt);
    } else if (fname == "min") {
      if (agg.children.empty()) {
        throw duckdb::NotImplementedException("min() without arguments not supported");
      }
      spec.kind          = aggregate_kind::MIN;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(agg.return_type);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::MIN);
      layout.merge_nth_index.push_back(std::nullopt);
    } else if (fname == "max") {
      if (agg.children.empty()) {
        throw duckdb::NotImplementedException("max() without arguments not supported");
      }
      spec.kind          = aggregate_kind::MAX;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(agg.return_type);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::MAX);
      layout.merge_nth_index.push_back(std::nullopt);
    } else if (fname == "avg") {
      if (agg.children.empty()) {
        throw duckdb::NotImplementedException("avg() without arguments not supported");
      }
      spec.kind          = aggregate_kind::AVG;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(agg.return_type);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
      layout.merge_nth_index.push_back(std::nullopt);
      spec.local_count_idx = local_idx++;
      layout.local_types.push_back(duckdb::LogicalType::BIGINT);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
      layout.merge_nth_index.push_back(std::nullopt);
      layout.has_avg = true;
    } else if (fname == "first") {
      spec.kind          = aggregate_kind::FIRST;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(agg.return_type);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::NTH_ELEMENT);
      layout.merge_nth_index.push_back(0);  // first element
    } else {
      throw duckdb::NotImplementedException("Aggregate not supported: " + fname);
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
  auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  // Reduce the sum column using its own type (cudf requires output type == input type
  // for fixed-point reductions). The final AVG return type is applied after division.
  auto sum_type    = sum_view.type();
  auto sum_value   = cudf::reduce(sum_view, *sum_agg, sum_type, stream, memory_resource);
  auto count_value = cudf::reduce(
    count_view, *sum_agg, cudf::data_type(cudf::type_id::INT64), stream, memory_resource);

  auto const count_host = scalar_cast<cudf::numeric_scalar<int64_t>>(*count_value).value();

  // Step 1: Extract the sum value as long double, regardless of the source type.
  // The sum column type may differ from the return type (e.g., sum is DECIMAL64 but
  // DuckDB's avg return type is DOUBLE).
  long double sum_host = 0.0L;
  bool sum_is_decimal =
    (sum_type.id() == cudf::type_id::DECIMAL32 || sum_type.id() == cudf::type_id::DECIMAL64 ||
     sum_type.id() == cudf::type_id::DECIMAL128);
  if (sum_is_decimal) {
    auto denom = std::pow(10.0L, static_cast<long double>(-sum_type.scale()));
    switch (sum_type.id()) {
      case cudf::type_id::DECIMAL32: {
        auto& s  = static_cast<cudf::fixed_point_scalar<numeric::decimal32>&>(*sum_value);
        sum_host = static_cast<long double>(s.value(stream)) / denom;
        break;
      }
      case cudf::type_id::DECIMAL64: {
        auto& s  = static_cast<cudf::fixed_point_scalar<numeric::decimal64>&>(*sum_value);
        sum_host = static_cast<long double>(s.value(stream)) / denom;
        break;
      }
      case cudf::type_id::DECIMAL128: {
        auto& s  = static_cast<cudf::fixed_point_scalar<numeric::decimal128>&>(*sum_value);
        sum_host = static_cast<long double>(s.value(stream)) / denom;
        break;
      }
      default: break;
    }
  } else {
    switch (sum_type.id()) {
      case cudf::type_id::FLOAT32:
        sum_host = scalar_cast<cudf::numeric_scalar<float>>(*sum_value).value();
        break;
      case cudf::type_id::FLOAT64:
        sum_host = scalar_cast<cudf::numeric_scalar<double>>(*sum_value).value();
        break;
      case cudf::type_id::INT32:
        sum_host = scalar_cast<cudf::numeric_scalar<int32_t>>(*sum_value).value();
        break;
      case cudf::type_id::INT64:
        sum_host = scalar_cast<cudf::numeric_scalar<int64_t>>(*sum_value).value();
        break;
      default: throw duckdb::NotImplementedException("AVG: unsupported sum column type");
    }
  }

  // Step 2: Compute avg and produce the output scalar in the DuckDB return type.
  long double avg_host = (count_host == 0) ? 0.0L : (sum_host / count_host);
  std::unique_ptr<cudf::scalar> out_scalar;

  if (return_type.id() == duckdb::LogicalTypeId::DECIMAL) {
    auto scale         = duckdb::DecimalType::GetScale(return_type);
    auto width         = duckdb::DecimalType::GetWidth(return_type);
    auto scale_type    = numeric::scale_type{-scale};
    auto denom         = std::pow(10.0L, static_cast<long double>(scale));
    long double scaled = avg_host * denom;
    if (width <= duckdb::Decimal::MAX_WIDTH_INT32) {
      out_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        static_cast<int32_t>(std::llround(scaled)), scale_type, true, stream, memory_resource);
    } else if (width <= duckdb::Decimal::MAX_WIDTH_INT64) {
      out_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        static_cast<int64_t>(std::llround(scaled)), scale_type, true, stream, memory_resource);
    } else {
      out_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
        static_cast<__int128_t>(std::llround(scaled)), scale_type, true, stream, memory_resource);
    }
  } else {
    // Non-decimal return type (typically DOUBLE)
    auto out_cudf_type = ToCudfType(return_type);
    switch (out_cudf_type.id()) {
      case cudf::type_id::FLOAT64:
        out_scalar = make_numeric_scalar_with_value<double>(
          out_cudf_type, static_cast<double>(avg_host), stream);
        break;
      case cudf::type_id::FLOAT32:
        out_scalar = make_numeric_scalar_with_value<float>(
          out_cudf_type, static_cast<float>(avg_host), stream);
        break;
      case cudf::type_id::INT64:
        out_scalar = make_numeric_scalar_with_value<int64_t>(
          out_cudf_type, static_cast<int64_t>(avg_host), stream);
        break;
      case cudf::type_id::INT32:
        out_scalar = make_numeric_scalar_with_value<int32_t>(
          out_cudf_type, static_cast<int32_t>(avg_host), stream);
        break;
      default: throw duckdb::NotImplementedException("AVG: unsupported return type");
    }
  }

  return cudf::make_column_from_scalar(*out_scalar, 1, stream, memory_resource);
}

}  // namespace

std::unique_ptr<operator_data> sirius_physical_ungrouped_aggregate::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_ungrouped_aggregate::execute"};
  const auto& input_batches = input_data.get_data_batches();
  if (aggregates.empty()) {
    return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  auto layout = build_aggregate_layout(aggregates);
  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  outputs.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    if (!batch) { continue; }
    auto* space = batch->get_memory_space();
    if (!space) { continue; }

    auto table = batch->get_data()->cast<cucascade::gpu_table_representation>().get_table();
    auto view  = table.view();

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
          bool is_decimal = (col.type().id() == cudf::type_id::DECIMAL32 ||
                             col.type().id() == cudf::type_id::DECIMAL64 ||
                             col.type().id() == cudf::type_id::DECIMAL128);

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
    std::unique_ptr<cucascade::idata_representation> output_data =
      std::make_unique<cucascade::gpu_table_representation>(std::move(out_table), *space);
    auto const batch_id = ::sirius::get_next_batch_id();
    outputs.push_back(std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data)));
  }

  return std::make_unique<operator_data>(outputs);
}

// Helper to deep copy Expression vector (same as in grouped_aggregate)
static duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> copy_expressions(
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& src)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> result;
  result.reserve(src.size());
  for (const auto& expr : src) {
    result.push_back(expr->Copy());
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
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::idx_t estimated_cardinality,
  duckdb::TupleDataValidityType distinct_validity)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_AGGREGATE, std::move(types), estimated_cardinality),
    aggregates(std::move(expressions))
{
  distinct_collection_info = duckdb::DistinctAggregateCollectionInfo::Create(aggregates);
  // aggregation_result       = duckdb::make_shared_ptr<GPUIntermediateRelation>(aggregates.size());
  if (!distinct_collection_info) { return; }
  distinct_data =
    duckdb::make_uniq<duckdb::DistinctAggregateData>(*distinct_collection_info, distinct_validity);
}

std::unique_ptr<operator_data> sirius_physical_ungrouped_aggregate_merge::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_ungrouped_aggregate_merge::execute"};
  const auto& input_batches = input_data.get_data_batches();
  if (aggregates.empty()) {
    return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> valid_batches;
  valid_batches.reserve(input_batches.size());
  for (auto const& batch : input_batches) {
    if (batch) { valid_batches.push_back(batch); }
  }
  if (valid_batches.empty()) {
    return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  cucascade::memory::memory_space* space = valid_batches[0]->get_memory_space();
  if (space == nullptr) {
    return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  auto layout = build_aggregate_layout(aggregates);
  std::shared_ptr<cucascade::data_batch> merged_batch;
  if (valid_batches.size() == 1) {
    merged_batch = valid_batches[0];
  } else {
    merged_batch = gpu_merge_impl::merge_ungrouped_aggregate(
      valid_batches, layout.merge_kinds, layout.merge_nth_index, stream, *space);
  }

  if (!layout.has_avg) {
    return std::make_unique<operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(merged_batch)});
  }

  auto merged_table =
    merged_batch->get_data()->cast<cucascade::gpu_table_representation>().get_table();
  auto merged_view = merged_table.view();

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
  std::unique_ptr<cucascade::idata_representation> output_data =
    std::make_unique<cucascade::gpu_table_representation>(std::move(out_table), *space);
  auto const batch_id = ::sirius::get_next_batch_id();
  auto output_batch   = std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data));

  return std::make_unique<operator_data>(
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
    auto batch =
      ports.begin()->second->repo->pop_data_batch(::cucascade::batch_state::task_created);
    if (batch) {
      input_batch.push_back(std::move(batch));
    } else {
      found_batch = false;
    }
  }
  return std::make_unique<operator_data>(input_batch);
}

}  // namespace op
}  // namespace sirius
