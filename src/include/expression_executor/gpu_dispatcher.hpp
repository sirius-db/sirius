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

#pragma once

#include "gpu_columns.hpp"
#include <cudf/column/column.hpp>
#include <memory>
#include <rmm/resource_ref.hpp>
#include <tuple>

namespace duckdb
{
namespace sirius
{

enum class StringMatchingType : uint8_t
{
  LIKE,
  NOT_LIKE,
  CONTAINS,
  PREFIX,
  SUFFIX
};

//----------Gpu Dispatcher----------//
struct GpuDispatcher
{
  //----------Materialize----------//
  static std::unique_ptr<cudf::column>
  DispatchMaterialize(const GPUColumn* input,
                      rmm::device_async_resource_ref mr,
                      rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  //----------Substring----------//
  static std::unique_ptr<cudf::column>
  DispatchSubstring(const cudf::column_view& input,
                    uint64_t start_idx,
                    uint64_t len,
                    rmm::device_async_resource_ref mr,
                    rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  //----------String Matching----------//
  template <StringMatchingType MatchType>
  static std::unique_ptr<cudf::column>
  DispatchStringMatching(const cudf::column_view& input,
                         const std::string& match_str,
                         rmm::device_async_resource_ref mr,
                         rmm::cuda_stream_view = rmm::cuda_stream_default);

  //----------Selection----------//
  static std::tuple<uint64_t*, uint64_t>
  DispatchSelect(const cudf::column_view& bitmap,
                 rmm::device_async_resource_ref mr,
                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);
};

} // namespace sirius
} // namespace duckdb