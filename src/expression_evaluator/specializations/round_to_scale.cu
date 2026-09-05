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

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <expression_evaluator/round_to_scale.hpp>
#include <sirius/exception.hpp>

#include <cmath>

namespace sirius {
namespace {

/// `round(x * p) / p` on the whole value, `round` half away from zero: DuckDB's
/// `CAST(double AS DECIMAL)` arithmetic (duckdb/src/common/operator/cast_operators.cpp,
/// DoubleToDecimalCast), so the two engines round the same doubles the same way.
template <typename T>
struct round_scaled {
  T scale;
  __device__ T operator()(T value) const
  {
    if (!isfinite(value)) { return value; }
    return round(value * scale) / scale;
  }
};

struct dispatch_round {
  template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           std::int32_t decimal_places,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    auto output = cudf::make_numeric_column(input.type(),
                                            input.size(),
                                            cudf::copy_bitmask(input, stream, mr),
                                            input.null_count(),
                                            stream,
                                            mr);
    // Powers of ten are exact in double up to 10^22, far beyond any decimal scale.
    auto const scale = static_cast<T>(std::pow(10.0, static_cast<double>(decimal_places)));
    thrust::transform(rmm::exec_policy(stream),
                      input.begin<T>(),
                      input.end<T>(),
                      output->mutable_view().begin<T>(),
                      round_scaled<T>{scale});
    return output;
  }

  template <typename T, std::enable_if_t<!std::is_floating_point_v<T>>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const&,
                                           std::int32_t,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref) const
  {
    throw sirius::invalid_input_exception(
      "round_to_scale_like_duckdb: only FLOAT32/FLOAT64 columns can be rounded to a scale");
  }
};

}  // namespace

std::unique_ptr<cudf::column> round_to_scale_like_duckdb(cudf::column_view const& input,
                                                         std::int32_t decimal_places,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (!cudf::is_floating_point(input.type())) {
    throw sirius::invalid_input_exception(
      "round_to_scale_like_duckdb: only FLOAT32/FLOAT64 columns can be rounded to a scale");
  }
  return cudf::type_dispatcher(input.type(), dispatch_round{}, input, decimal_places, stream, mr);
}

}  // namespace sirius
