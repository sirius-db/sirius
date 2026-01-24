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

#include "data/data_batch_utils.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "log/logging.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <algorithm>
#include <mutex>

namespace sirius {
namespace op {

sirius_physical_ungrouped_aggregate::sirius_physical_ungrouped_aggregate(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
  duckdb::idx_t estimated_cardinality,
  duckdb::TupleDataValidityType distinct_validity)
  : sirius_physical_operator(duckdb::PhysicalOperatorType::UNGROUPED_AGGREGATE,
                             std::move(types),
                             estimated_cardinality),
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
cudf::data_type ToCudfType(const duckdb::LogicalType& t)
{
  return duckdb::sirius::GpuExpressionState::GetCudfType(t);
}

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
void set_fixed_point_value(cudf::fixed_point_scalar<T>& s, typename T::rep value)
{
  cudaMemcpyAsync(
    s.data(), &value, sizeof(value), cudaMemcpyHostToDevice, cudf::get_default_stream().value());
}

template <typename T>
std::unique_ptr<cudf::scalar> make_numeric_scalar_with_value(cudf::data_type type, T value)
{
  auto out = cudf::make_numeric_scalar(type);
  scalar_cast<cudf::numeric_scalar<T>>(*out).set_value(value, cudf::get_default_stream());
  return out;
}

template <typename Rep>
std::unique_ptr<cudf::scalar> make_fixed_point_scalar_with_value(cudf::data_type type, Rep value)
{
  auto out = cudf::make_fixed_point_scalar(type, value);
  return out;
}

// Sum two scalars of same type_id, returning updated accumulator (in-place)
void accumulate_sum(cudf::scalar& acc, const cudf::scalar& incoming)
{
  auto id = acc.type().id();
  switch (id) {
    case cudf::type_id::INT8: {
      auto v = scalar_cast<cudf::numeric_scalar<int8_t>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<int8_t>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<int8_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::INT16: {
      auto v = scalar_cast<cudf::numeric_scalar<int16_t>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<int16_t>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<int16_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::INT32: {
      auto v = scalar_cast<cudf::numeric_scalar<int32_t>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<int32_t>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<int32_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::INT64: {
      auto v = scalar_cast<cudf::numeric_scalar<int64_t>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<int64_t>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<int64_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::UINT8: {
      auto v = scalar_cast<cudf::numeric_scalar<uint8_t>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<uint8_t>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<uint8_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::UINT16: {
      auto v = scalar_cast<cudf::numeric_scalar<uint16_t>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<uint16_t>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<uint16_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::UINT32: {
      auto v = scalar_cast<cudf::numeric_scalar<uint32_t>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<uint32_t>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<uint32_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::UINT64: {
      auto v = scalar_cast<cudf::numeric_scalar<uint64_t>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<uint64_t>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<uint64_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::FLOAT32: {
      auto v = scalar_cast<cudf::numeric_scalar<float>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<float>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<float>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::FLOAT64: {
      auto v = scalar_cast<cudf::numeric_scalar<double>>(acc).value() +
               scalar_cast<cudf::numeric_scalar<double>>(incoming).value();
      scalar_cast<cudf::numeric_scalar<double>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::DECIMAL32: {
      using dec_t = numeric::decimal32;
      auto v      = scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc).value() +
               scalar_cast<cudf::fixed_point_scalar<dec_t>>(incoming).value();
      set_fixed_point_value(scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc), v);
      break;
    }
    case cudf::type_id::DECIMAL64: {
      using dec_t = numeric::decimal64;
      auto v      = scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc).value() +
               scalar_cast<cudf::fixed_point_scalar<dec_t>>(incoming).value();
      set_fixed_point_value(scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc), v);
      break;
    }
    case cudf::type_id::DECIMAL128: {
      using dec_t = numeric::decimal128;
      auto v      = scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc).value() +
               scalar_cast<cudf::fixed_point_scalar<dec_t>>(incoming).value();
      set_fixed_point_value(scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc), v);
      break;
    }
    default:
      throw duckdb::NotImplementedException("Unsupported type for sum in GPU ungrouped aggregate");
  }
}

// update min/max
enum class minmax_op { MIN, MAX };
void accumulate_minmax(cudf::scalar& acc, const cudf::scalar& incoming, minmax_op op)
{
  auto id         = acc.type().id();
  auto choose_min = op == minmax_op::MIN;
  switch (id) {
    case cudf::type_id::INT8: {
      auto a = scalar_cast<cudf::numeric_scalar<int8_t>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<int8_t>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<int8_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::INT16: {
      auto a = scalar_cast<cudf::numeric_scalar<int16_t>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<int16_t>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<int16_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::INT32: {
      auto a = scalar_cast<cudf::numeric_scalar<int32_t>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<int32_t>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<int32_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::INT64: {
      auto a = scalar_cast<cudf::numeric_scalar<int64_t>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<int64_t>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<int64_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::UINT8: {
      auto a = scalar_cast<cudf::numeric_scalar<uint8_t>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<uint8_t>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<uint8_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::UINT16: {
      auto a = scalar_cast<cudf::numeric_scalar<uint16_t>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<uint16_t>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<uint16_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::UINT32: {
      auto a = scalar_cast<cudf::numeric_scalar<uint32_t>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<uint32_t>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<uint32_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::UINT64: {
      auto a = scalar_cast<cudf::numeric_scalar<uint64_t>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<uint64_t>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<uint64_t>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::FLOAT32: {
      auto a = scalar_cast<cudf::numeric_scalar<float>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<float>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<float>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::FLOAT64: {
      auto a = scalar_cast<cudf::numeric_scalar<double>>(acc).value();
      auto b = scalar_cast<cudf::numeric_scalar<double>>(incoming).value();
      auto v = choose_min ? std::min(a, b) : std::max(a, b);
      scalar_cast<cudf::numeric_scalar<double>>(acc).set_value(v, cudf::get_default_stream());
      break;
    }
    case cudf::type_id::DECIMAL32: {
      using dec_t = numeric::decimal32;
      auto a      = scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc).value();
      auto b      = scalar_cast<cudf::fixed_point_scalar<dec_t>>(incoming).value();
      auto v      = choose_min ? std::min(a, b) : std::max(a, b);
      set_fixed_point_value(scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc), v);
      break;
    }
    case cudf::type_id::DECIMAL64: {
      using dec_t = numeric::decimal64;
      auto a      = scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc).value();
      auto b      = scalar_cast<cudf::fixed_point_scalar<dec_t>>(incoming).value();
      auto v      = choose_min ? std::min(a, b) : std::max(a, b);
      set_fixed_point_value(scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc), v);
      break;
    }
    case cudf::type_id::DECIMAL128: {
      using dec_t = numeric::decimal128;
      auto a      = scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc).value();
      auto b      = scalar_cast<cudf::fixed_point_scalar<dec_t>>(incoming).value();
      auto v      = choose_min ? std::min(a, b) : std::max(a, b);
      set_fixed_point_value(scalar_cast<cudf::fixed_point_scalar<dec_t>>(acc), v);
      break;
    }
    default:
      throw duckdb::NotImplementedException(
        "Unsupported type for min/max in GPU ungrouped aggregate");
  }
}

}  // namespace

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_ungrouped_aggregate::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches)
{
  if (aggregates.empty()) { return {}; }

  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (batch) {
      space = batch->get_memory_space();
      break;
    }
  }
  if (space == nullptr) { return {}; }

  std::unique_lock<std::mutex> lk(_state->_mutex);
  if (!_state->_initialized) {
    _state->_running_values.resize(aggregates.size());
    _state->_running_counts.resize(aggregates.size(), 0);
    _state->_initialized = true;
  }

  for (auto const& batch : input_batches) {
    if (!batch) { continue; }
    auto table = batch->get_data()->cast<cucascade::gpu_table_representation>().get_table();
    auto view  = table.view();

    for (idx_t i = 0; i < aggregates.size(); ++i) {
      auto& agg = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
      if (agg.IsDistinct()) {
        throw duckdb::NotImplementedException("Distinct aggregates not supported in GPU path yet");
      }
      auto fname = agg.function.name;

      if (fname == "count_star") {
        _state->_running_counts[i] += static_cast<int64_t>(view.num_rows());
      } else if (fname == "count") {
        D_ASSERT(agg.children.size() == 1);
        auto idx    = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
        auto col    = view.column(static_cast<cudf::size_type>(idx));
        auto agg_op = cudf::make_count_aggregation<cudf::reduce_aggregation>();
        auto s      = cudf::reduce(col,
                              *agg_op,
                              cudf::data_type(cudf::type_id::INT64),
                              std::nullopt,
                              cudf::get_default_stream(),
                              rmm::mr::get_current_device_resource());
        _state->_running_counts[i] += static_cast<const cudf::numeric_scalar<int64_t>&>(*s).value();
      } else {
        D_ASSERT(agg.children.size() == 1);
        auto idx      = agg.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
        auto col      = view.column(static_cast<cudf::size_type>(idx));
        auto out_type = ToCudfType(agg.return_type);
        std::unique_ptr<cudf::scalar> s;
        if (fname == "sum" || fname == "sum_no_overflow") {
          auto agg_op = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
          s           = cudf::reduce(col,
                           *agg_op,
                           out_type,
                           std::nullopt,
                           cudf::get_default_stream(),
                           rmm::mr::get_current_device_resource());
          if (!_state->_running_values[i]) {
            _state->_running_values[i] = std::move(s);
          } else {
            accumulate_sum(*_state->_running_values[i], *s);
          }
        } else if (fname == "min") {
          auto agg_op = cudf::make_min_aggregation<cudf::reduce_aggregation>();
          s           = cudf::reduce(col,
                           *agg_op,
                           out_type,
                           std::nullopt,
                           cudf::get_default_stream(),
                           rmm::mr::get_current_device_resource());
          if (!_state->_running_values[i]) {
            _state->_running_values[i] = std::move(s);
          } else {
            accumulate_minmax(*_state->_running_values[i], *s, minmax_op::MIN);
          }
        } else if (fname == "max") {
          auto agg_op = cudf::make_max_aggregation<cudf::reduce_aggregation>();
          s           = cudf::reduce(col,
                           *agg_op,
                           out_type,
                           std::nullopt,
                           cudf::get_default_stream(),
                           rmm::mr::get_current_device_resource());
          if (!_state->_running_values[i]) {
            _state->_running_values[i] = std::move(s);
          } else {
            accumulate_minmax(*_state->_running_values[i], *s, minmax_op::MAX);
          }
        } else if (fname == "avg") {
          auto agg_op = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
          s           = cudf::reduce(col,
                           *agg_op,
                           out_type,
                           std::nullopt,
                           cudf::get_default_stream(),
                           rmm::mr::get_current_device_resource());
          if (!_state->_running_values[i]) {
            _state->_running_values[i] = std::move(s);
          } else {
            accumulate_sum(*_state->_running_values[i], *s);
          }
          _state->_running_counts[i] += static_cast<int64_t>(view.num_rows());
        } else {
          throw duckdb::NotImplementedException("Aggregate not supported: " + fname);
        }
      }
    }
  }

  // Build output row
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(aggregates.size());
  auto stream = cudf::get_default_stream();

  for (idx_t i = 0; i < aggregates.size(); ++i) {
    auto& agg = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
    auto tid  = ToCudfType(agg.return_type);

    std::unique_ptr<cudf::scalar> tmp_scalar;
    const cudf::scalar* out_scalar = nullptr;
    if (agg.function.name == "avg") {
      auto const cnt = _state->_running_counts[i];
      if (cnt == 0) {
        // produce zero of target type
        switch (tid.id()) {
          case cudf::type_id::FLOAT32: {
            tmp_scalar = make_numeric_scalar_with_value<float>(tid, 0.0f);
            out_scalar = tmp_scalar.get();
            break;
          }
          case cudf::type_id::FLOAT64: {
            tmp_scalar = make_numeric_scalar_with_value<double>(tid, 0.0);
            out_scalar = tmp_scalar.get();
            break;
          }
          case cudf::type_id::INT32: {
            tmp_scalar = make_numeric_scalar_with_value<int32_t>(tid, 0);
            out_scalar = tmp_scalar.get();
            break;
          }
          case cudf::type_id::INT64: {
            tmp_scalar = make_numeric_scalar_with_value<int64_t>(tid, 0);
            out_scalar = tmp_scalar.get();
            break;
          }
          default: throw duckdb::NotImplementedException("AVG output type not supported");
        }
      } else {
        // compute avg in double then cast to target type
        double sum_host = 0.0;
        switch (tid.id()) {
          case cudf::type_id::FLOAT32:
            sum_host =
              scalar_cast<cudf::numeric_scalar<float>>(*_state->_running_values[i]).value();
            _state->_running_values[i] =
              make_numeric_scalar_with_value<float>(tid, static_cast<float>(sum_host / cnt));
            break;
          case cudf::type_id::FLOAT64:
            sum_host =
              scalar_cast<cudf::numeric_scalar<double>>(*_state->_running_values[i]).value();
            _state->_running_values[i] =
              make_numeric_scalar_with_value<double>(tid, sum_host / cnt);
            break;
          case cudf::type_id::INT32:
            sum_host =
              scalar_cast<cudf::numeric_scalar<int32_t>>(*_state->_running_values[i]).value();
            _state->_running_values[i] =
              make_numeric_scalar_with_value<int32_t>(tid, static_cast<int32_t>(sum_host / cnt));
            break;
          case cudf::type_id::INT64:
            sum_host =
              scalar_cast<cudf::numeric_scalar<int64_t>>(*_state->_running_values[i]).value();
            _state->_running_values[i] =
              make_numeric_scalar_with_value<int64_t>(tid, static_cast<int64_t>(sum_host / cnt));
            break;
          default: throw duckdb::NotImplementedException("AVG output type not supported");
        }
        out_scalar = _state->_running_values[i].get();
      }
    } else if (agg.function.name == "count" || agg.function.name == "count_star") {
      tmp_scalar = make_numeric_scalar_with_value<int64_t>(cudf::data_type{cudf::type_id::INT64},
                                                           _state->_running_counts[i]);
      out_scalar = tmp_scalar.get();
    } else {
      // sum/min/max already accumulated in _running_values
      out_scalar = _state->_running_values[i].get();  // non-owning
    }

    cols.push_back(cudf::make_column_from_scalar(
      *out_scalar, 1, stream, rmm::mr::get_current_device_resource()));
  }

  auto out_table = std::make_unique<cudf::table>(std::move(cols));
  std::unique_ptr<cucascade::idata_representation> output_data =
    std::make_unique<cucascade::gpu_table_representation>(*out_table, *space);
  auto const batch_id = ::sirius::get_next_batch_id();
  auto output_batch   = std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data));

  return {std::move(output_batch)};
}

}  // namespace op
}  // namespace sirius
