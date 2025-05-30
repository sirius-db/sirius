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
  PREFIX
};

//----------Gpu Dispatcher----------//
struct GpuDispatcher
{
    //----------Materialize----------//
  static std::unique_ptr<cudf::column> DispatchMaterialize(const GPUColumn* input,
                                                           rmm::device_async_resource_ref mr);

  //----------Substring----------//
  static std::unique_ptr<cudf::column> DispatchSubstring(const cudf::column_view& input,
                                                         uint64_t start_idx,
                                                         uint64_t len,
                                                         rmm::device_async_resource_ref mr);

  //----------String Matching----------//
  template<StringMatchingType MatchType>
  static std::unique_ptr<cudf::column> DispatchStringMatching(const cudf::column_view& input,
                                                              const std::string& match_str,
                                                              rmm::device_async_resource_ref mr);

  //----------Selection----------//
  static std::tuple<uint64_t*, uint64_t> DispatchSelect(const cudf::column_view& bitmap,
                                                        rmm::device_async_resource_ref mr);

  //----------Utilities----------//
  // For substring
  static std::unique_ptr<cudf::column> MakeColumnFromPtrs(char* data,
                                                          uint64_t* offsets,
                                                          uint64_t count,
                                                          rmm::device_async_resource_ref mr);
  // For matching
  static std::unique_ptr<cudf::column> MakeColumnFromRowIds(uint64_t* row_ids,
                                                            uint64_t row_id_count,
                                                            uint64_t input_count,
                                                            rmm::device_async_resource_ref mr);

  // For extracting data from cudf::column types for sirius string APIs
  static std::tuple<uint64_t, char*, uint64_t*, uint64_t>
  PrepareStringData(const cudf::column_view& input);
};

} // namespace sirius
} // namespace duckdb