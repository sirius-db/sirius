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