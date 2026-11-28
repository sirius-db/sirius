/*
 * Copyright 2026, Sirius Contributors.
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

// sirius
#include <expression_evaluator/round_floating_point.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cmath>
#include <cstdint>

namespace sirius {
namespace {

constexpr int threads_per_block = 256;

// Mirrors DuckDB's RoundOperatorPrecision, including its handling of non-finite results. For a
// precision of 1 to 22 the product x * modifier can round onto a midpoint that x itself is not on:
// 2.675 * 100 becomes 267.5 while the stored 2.675 is below it. The sign of the product's rounding
// error, which fma yields exactly, then decides the direction instead of the tie rule.
template <typename T, bool NegativePrecision>
__global__ void round_floating_point_kernel(
  T const* input, T* output, cudf::size_type num_rows, double modifier, int32_t precision)
{
  auto const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_rows;
       i += stride) {
    auto const x = static_cast<double>(input[i]);
    double rounded;
    if constexpr (NegativePrecision) {
      if (::isinf(modifier)) {
        rounded = ::isfinite(input[i]) ? 0 : x;
      } else {
        rounded = ::round(x / modifier) * modifier;
      }
      if (!::isfinite(rounded)) { rounded = x; }
    } else {
      // __dmul_rn is never contracted into a fused multiply-add, so the midpoint test sees the
      // rounded product.
      auto const scaled = __dmul_rn(x, modifier);
      double error      = 0;
      if (precision > 0 && precision <= 22 && ::fabs(scaled - ::trunc(scaled)) == 0.5) {
        error = ::fma(x, modifier, -scaled);
      }
      double nearest;
      if (error != 0) {
        nearest = error < 0 ? ::floor(scaled) : ::ceil(scaled);
      } else {
        nearest = ::round(scaled);
      }
      rounded = nearest / modifier;
      if (!::isfinite(rounded)) { rounded = x; }
    }
    output[i] = static_cast<T>(rounded);
  }
}

template <typename T>
void launch_round(cudf::column_view const& input,
                  cudf::mutable_column_view output,
                  int32_t precision,
                  ::cuda::stream_ref stream)
{
  auto const negative_precision = precision < 0;
  // The same host std::pow call as DuckDB, so the modifier is bit-identical.
  auto const modifier =
    std::pow(10, static_cast<T>(negative_precision ? -static_cast<int64_t>(precision) : precision));
  auto const num_blocks = static_cast<uint32_t>(
    (static_cast<int64_t>(input.size()) + threads_per_block - 1) / threads_per_block);
  if (negative_precision) {
    round_floating_point_kernel<T, true><<<num_blocks, threads_per_block, 0, stream.get()>>>(
      input.data<T>(), output.data<T>(), input.size(), modifier, precision);
  } else {
    round_floating_point_kernel<T, false><<<num_blocks, threads_per_block, 0, stream.get()>>>(
      input.data<T>(), output.data<T>(), input.size(), modifier, precision);
  }
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace

std::unique_ptr<cudf::column> round_floating_point(cudf::column_view const& input,
                                                   int32_t precision,
                                                   ::cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto const type_id = input.type().id();
  if (type_id != cudf::type_id::FLOAT32 && type_id != cudf::type_id::FLOAT64) {
    throw invalid_input_exception("[round_floating_point] unsupported input type {}",
                                  static_cast<int>(type_id));
  }
  auto result = cudf::make_fixed_width_column(input.type(),
                                              input.size(),
                                              cudf::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);
  if (input.is_empty()) { return result; }
  if (type_id == cudf::type_id::FLOAT32) {
    launch_round<float>(input, result->mutable_view(), precision, stream);
  } else {
    launch_round<double>(input, result->mutable_view(), precision, stream);
  }
  return result;
}

}  // namespace sirius
