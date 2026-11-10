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

#pragma once

#include "cudf/ast_compat.hpp"

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/version_config.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <memory>
#include <vector>

namespace sirius::cudf_compat {

namespace detail {

// Nightly cuDF accepts an LTO fragment list before the stream and allocator. Detect that
// signature instead of a release number, since stable and nightly builds can share a version.
template <typename Expression>
std::unique_ptr<cudf::column> compute_column_jit_unchecked(cudf::table_view const& input,
                                                           Expression const& expression,
                                                           ::cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  if constexpr (requires { cudf::compute_column_jit(input, expression, {}, stream, mr); }) {
    return cudf::compute_column_jit(input, expression, {}, stream, mr);
  } else {
    return cudf::compute_column_jit(input, expression, stream, mr);
  }
}

// Upper bound on the occupancy-sized grid of the JIT transform kernel: one resident thread per
// hardware slot. Batches at or below it are processed in a single grid-stride iteration.
inline cudf::size_type max_resident_threads()
{
  int device = 0;
  int sms    = 0;
  int per_sm = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
  return static_cast<cudf::size_type>(sms) * per_sm;
}

}  // namespace detail

/**
 * @brief Evaluates `expression` with cuDF's JIT backend.
 *
 * Before 26.12 (rapidsai/cudf#23981), the null-aware JIT transform kernel lets every divergent
 * subgroup of a warp store the whole validity word, so on batches spanning several grid strides
 * the warp holding row `size % stride` can mark up to 31 valid rows NULL. Grid strides and warp
 * bases are multiples of 32, so a 32-aligned row count never diverges; the remaining tail fits in
 * one grid-stride iteration.
 */
template <typename Expression>
std::unique_ptr<cudf::column> compute_column_jit(cudf::table_view const& input,
                                                 Expression const& expression,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
#if CUDF_VERSION_LT(26, 12, 0)
  constexpr cudf::size_type warp_size = 32;
  auto const rows                     = input.num_rows();
  auto const aligned_rows             = rows - rows % warp_size;
  if (aligned_rows != rows && rows > detail::max_resident_threads()) {
    auto const parts = cudf::split(input, std::vector<cudf::size_type>{aligned_rows}, stream);
    std::array<std::unique_ptr<cudf::column>, 2> results{
      detail::compute_column_jit_unchecked(parts[0], expression, stream, mr),
      detail::compute_column_jit_unchecked(parts[1], expression, stream, mr)};
    std::vector<cudf::column_view> views{results[0]->view(), results[1]->view()};
    return cudf::concatenate(views, stream, mr);
  }
#endif
  return detail::compute_column_jit_unchecked(input, expression, stream, mr);
}

}  // namespace sirius::cudf_compat
