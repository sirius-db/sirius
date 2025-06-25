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

#include "expression_executor/gpu_dispatcher.hpp"
#include "gpu_physical_strings_matching.hpp"
#include "gpu_physical_substring.hpp"
#include <cub/cub.cuh>
#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <memory>
#include <thrust/iterator/counting_iterator.h>

namespace duckdb
{
namespace sirius
{

#define SPLIT_DELIMITER "%"

//----------Substring----------//
std::unique_ptr<cudf::column> GpuDispatcher::DispatchSubstring(const cudf::column_view& input,
                                                               uint64_t start_idx,
                                                               uint64_t len,
                                                               rmm::device_async_resource_ref mr,
                                                               rmm::cuda_stream_view stream)
{
  return DoSubstring(input.data<char>(),
                     input.size(),
                     input.child(0).data<int64_t>(),
                     static_cast<int64_t>(start_idx),
                     static_cast<int64_t>(len),
                     mr,
                     stream);
};

//----------String Matching----------//
template <StringMatchingType MatchType>
std::unique_ptr<cudf::column>
GpuDispatcher::DispatchStringMatching(const cudf::column_view& input,
                                      const std::string& match_str,
                                      rmm::device_async_resource_ref mr,
                                      rmm::cuda_stream_view stream)
{
  cudf::strings_column_view input_view(input);
  const auto byte_count = input_view.chars_size(stream);

  if constexpr (MatchType == StringMatchingType::LIKE || MatchType == StringMatchingType::NOT_LIKE)
  {
    std::vector<std::string> match_terms = string_split(match_str, SPLIT_DELIMITER);
    auto result                          = DoMultiStringMatching(input.data<char>(),
                                        input.size(),
                                        input.child(0).data<int64_t>(),
                                        byte_count,
                                        match_terms,
                                        mr,
                                        stream);
    if constexpr (MatchType == StringMatchingType::LIKE)
    {
      return std::move(result);
    }

    // Otherwise, we need to invert the result
    return cudf::unary_operation(result->view(), cudf::unary_operator::NOT, stream, mr);
  }
  else if constexpr (MatchType == StringMatchingType::CONTAINS)
  {
    return DoStringMatching(input.data<char>(),
                            input.size(),
                            input.child(0).data<int64_t>(),
                            byte_count,
                            match_str,
                            mr,
                            stream);
  }
  else if constexpr (MatchType == StringMatchingType::PREFIX)
  {
    return DoPrefixMatching(input.data<char>(),
                            input.size(),
                            input.child(0).data<int64_t>(),
                            byte_count,
                            match_str,
                            mr,
                            stream);
  }
  else
  {
    throw NotImplementedException("Unsupported StringMatchingType when not using cudf: %d",
                                  static_cast<int>(MatchType));
  }
}

//----------Instantiations----------//
#define INSTANTIATE_STR_MATCHING(T)                                                                \
  template std::unique_ptr<cudf::column>                                                           \
  GpuDispatcher::DispatchStringMatching<StringMatchingType::T>(const cudf::column_view& input,     \
                                                               const std::string& match_str,       \
                                                               rmm::device_async_resource_ref mr,  \
                                                               rmm::cuda_stream_view stream);

INSTANTIATE_STR_MATCHING(CONTAINS)
INSTANTIATE_STR_MATCHING(LIKE)
INSTANTIATE_STR_MATCHING(NOT_LIKE)
INSTANTIATE_STR_MATCHING(PREFIX)
INSTANTIATE_STR_MATCHING(SUFFIX)

#undef SPLIT_DELIMITER
#undef INSTANTIATE_STR_MATCHING

} // namespace sirius
} // namespace duckdb