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
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/resource_ref.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

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
std::unique_ptr<cudf::scalar> make_numeric_scalar_with_value(cudf::data_type type, T value)
{
  auto out = cudf::make_numeric_scalar(type);
  scalar_cast<cudf::numeric_scalar<T>>(*out).set_value(value, cudf::get_default_stream());
  return out;
}

enum class aggregate_kind { SUM, MIN, MAX, COUNT, COUNT_STAR, AVG };

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
    } else if (fname == "sum" || fname == "sum_no_overflow") {
      if (agg.children.empty()) {
        throw duckdb::NotImplementedException("sum() without arguments not supported");
      }
      spec.kind          = aggregate_kind::SUM;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(agg.return_type);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
    } else if (fname == "min") {
      if (agg.children.empty()) {
        throw duckdb::NotImplementedException("min() without arguments not supported");
      }
      spec.kind          = aggregate_kind::MIN;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(agg.return_type);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::MIN);
    } else if (fname == "max") {
      if (agg.children.empty()) {
        throw duckdb::NotImplementedException("max() without arguments not supported");
      }
      spec.kind          = aggregate_kind::MAX;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(agg.return_type);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::MAX);
    } else if (fname == "avg") {
      if (agg.children.empty()) {
        throw duckdb::NotImplementedException("avg() without arguments not supported");
      }
      spec.kind          = aggregate_kind::AVG;
      spec.input_idx     = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
      spec.local_sum_idx = local_idx++;
      layout.local_types.push_back(agg.return_type);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
      spec.local_count_idx = local_idx++;
      layout.local_types.push_back(duckdb::LogicalType::BIGINT);
      layout.merge_kinds.push_back(cudf::aggregation::Kind::SUM);
      layout.has_avg = true;
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
  auto sum_agg     = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto sum_type    = ToCudfType(return_type);
  auto sum_value   = cudf::reduce(sum_view, *sum_agg, sum_type, stream, memory_resource);
  auto count_value = cudf::reduce(
    count_view, *sum_agg, cudf::data_type(cudf::type_id::INT64), stream, memory_resource);

  auto const count_host = scalar_cast<cudf::numeric_scalar<int64_t>>(*count_value).value();
  std::unique_ptr<cudf::scalar> out_scalar;
  if (return_type.id() == duckdb::LogicalTypeId::DECIMAL) {
    auto scale           = duckdb::DecimalType::GetScale(return_type);
    auto width           = duckdb::DecimalType::GetWidth(return_type);
    auto scale_type      = numeric::scale_type{-scale};
    auto denom           = std::pow(10.0L, static_cast<long double>(scale));
    long double sum_host = 0.0L;
    switch (sum_type.id()) {
      case cudf::type_id::DECIMAL32: {
        auto& sum_scalar = static_cast<cudf::fixed_point_scalar<numeric::decimal32>&>(*sum_value);
        sum_host         = static_cast<long double>(sum_scalar.value(stream)) / denom;
        break;
      }
      case cudf::type_id::DECIMAL64: {
        auto& sum_scalar = static_cast<cudf::fixed_point_scalar<numeric::decimal64>&>(*sum_value);
        sum_host         = static_cast<long double>(sum_scalar.value(stream)) / denom;
        break;
      }
      case cudf::type_id::DECIMAL128: {
        auto& sum_scalar = static_cast<cudf::fixed_point_scalar<numeric::decimal128>&>(*sum_value);
        sum_host         = static_cast<long double>(sum_scalar.value(stream)) / denom;
        break;
      }
      default: throw duckdb::NotImplementedException("AVG decimal sum type not supported");
    }
    long double avg_host = (count_host == 0) ? 0.0L : (sum_host / count_host);
    long double scaled   = avg_host * denom;
    if (width <= duckdb::Decimal::MAX_WIDTH_INT32) {
      auto rep   = static_cast<int32_t>(std::llround(scaled));
      out_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        rep, scale_type, true, stream, memory_resource);
    } else if (width <= duckdb::Decimal::MAX_WIDTH_INT64) {
      auto rep   = static_cast<int64_t>(std::llround(scaled));
      out_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        rep, scale_type, true, stream, memory_resource);
    } else {
      auto rep   = static_cast<__int128_t>(std::llround(scaled));
      out_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
        rep, scale_type, true, stream, memory_resource);
    }
  } else {
    if (count_host == 0) {
      switch (sum_type.id()) {
        case cudf::type_id::FLOAT32:
          out_scalar = make_numeric_scalar_with_value<float>(sum_type, 0.0f);
          break;
        case cudf::type_id::FLOAT64:
          out_scalar = make_numeric_scalar_with_value<double>(sum_type, 0.0);
          break;
        case cudf::type_id::INT32:
          out_scalar = make_numeric_scalar_with_value<int32_t>(sum_type, 0);
          break;
        case cudf::type_id::INT64:
          out_scalar = make_numeric_scalar_with_value<int64_t>(sum_type, 0);
          break;
        default: throw duckdb::NotImplementedException("AVG output type not supported");
      }
    } else {
      switch (sum_type.id()) {
        case cudf::type_id::FLOAT32: {
          auto sum_host = scalar_cast<cudf::numeric_scalar<float>>(*sum_value).value();
          out_scalar    = make_numeric_scalar_with_value<float>(sum_type, sum_host / count_host);
          break;
        }
        case cudf::type_id::FLOAT64: {
          auto sum_host = scalar_cast<cudf::numeric_scalar<double>>(*sum_value).value();
          out_scalar    = make_numeric_scalar_with_value<double>(sum_type, sum_host / count_host);
          break;
        }
        case cudf::type_id::INT32: {
          auto sum_host = scalar_cast<cudf::numeric_scalar<int32_t>>(*sum_value).value();
          out_scalar    = make_numeric_scalar_with_value<int32_t>(
            sum_type, static_cast<int32_t>(sum_host / count_host));
          break;
        }
        case cudf::type_id::INT64: {
          auto sum_host = scalar_cast<cudf::numeric_scalar<int64_t>>(*sum_value).value();
          out_scalar    = make_numeric_scalar_with_value<int64_t>(
            sum_type, static_cast<int64_t>(sum_host / count_host));
          break;
        }
        default: throw duckdb::NotImplementedException("AVG output type not supported");
      }
    }
  }

  return cudf::make_column_from_scalar(*out_scalar, 1, stream, memory_resource);
}

}  // namespace

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_ungrouped_aggregate::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  if (aggregates.empty()) { return {}; }

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
            cudf::data_type{cudf::type_id::INT64}, static_cast<int64_t>(view.num_rows()));
          cols.push_back(
            cudf::make_column_from_scalar(*scalar, 1, stream, space->get_default_allocator()));
          break;
        }
        case aggregate_kind::COUNT: {
          auto col    = view.column(static_cast<cudf::size_type>(spec.input_idx));
          auto agg_op = cudf::make_count_aggregation<cudf::reduce_aggregation>();
          auto scalar = cudf::reduce(col,
                                     *agg_op,
                                     cudf::data_type(cudf::type_id::INT64),
                                     std::nullopt,
                                     stream,
                                     space->get_default_allocator());
          cols.push_back(
            cudf::make_column_from_scalar(*scalar, 1, stream, space->get_default_allocator()));
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
          auto scalar = cudf::reduce(
            col, *agg_op, out_type, std::nullopt, stream, space->get_default_allocator());
          cols.push_back(
            cudf::make_column_from_scalar(*scalar, 1, stream, space->get_default_allocator()));
          if (spec.kind == aggregate_kind::AVG) {
            auto count_scalar = make_numeric_scalar_with_value<int64_t>(
              cudf::data_type{cudf::type_id::INT64}, static_cast<int64_t>(view.num_rows()));
            cols.push_back(cudf::make_column_from_scalar(
              *count_scalar, 1, stream, space->get_default_allocator()));
          }
          break;
        }
      }
    }

    auto out_table = std::make_unique<cudf::table>(std::move(cols));
    std::unique_ptr<cucascade::idata_representation> output_data =
      std::make_unique<cucascade::gpu_table_representation>(std::move(out_table), *space);
    auto const batch_id = ::sirius::get_next_batch_id();
    outputs.push_back(std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data)));
  }

  return outputs;
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

std::vector<std::shared_ptr<cucascade::data_batch>>
sirius_physical_ungrouped_aggregate_merge::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  if (aggregates.empty()) { return {}; }

  std::vector<std::shared_ptr<cucascade::data_batch>> valid_batches;
  valid_batches.reserve(input_batches.size());
  for (auto const& batch : input_batches) {
    if (batch) { valid_batches.push_back(batch); }
  }
  if (valid_batches.empty()) { return {}; }

  cucascade::memory::memory_space* space = valid_batches[0]->get_memory_space();
  if (space == nullptr) { return {}; }

  auto layout = build_aggregate_layout(aggregates);
  std::shared_ptr<cucascade::data_batch> merged_batch;
  if (valid_batches.size() == 1) {
    merged_batch = valid_batches[0];
  } else {
    merged_batch =
      gpu_merge_impl::merge_ungrouped_aggregate(valid_batches, layout.merge_kinds, stream, *space);
  }

  if (!layout.has_avg) { return {std::move(merged_batch)}; }

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
        sum_view, count_view, spec.return_type, stream, space->get_default_allocator()));
    } else {
      auto col_view = merged_view.column(static_cast<cudf::size_type>(spec.local_sum_idx));
      output_cols.push_back(
        std::make_unique<cudf::column>(col_view, stream, space->get_default_allocator()));
    }
  }

  auto out_table =
    std::make_unique<cudf::table>(std::move(output_cols), stream, space->get_default_allocator());
  std::unique_ptr<cucascade::idata_representation> output_data =
    std::make_unique<cucascade::gpu_table_representation>(std::move(out_table), *space);
  auto const batch_id = ::sirius::get_next_batch_id();
  auto output_batch   = std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data));

  return {std::move(output_batch)};
}

}  // namespace op
}  // namespace sirius
