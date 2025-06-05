#include "expression_executor/gpu_dispatcher.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_physical_strings_matching.hpp"
#include "gpu_physical_substring.hpp"
#include <cub/cub.cuh>
#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <memory>
#include <thrust/iterator/counting_iterator.h>
#include <tuple>

namespace duckdb
{
namespace sirius
{

#define SPLIT_DELIMITER "%"

//----------Substring----------//
std::unique_ptr<cudf::column> GpuDispatcher::DispatchSubstring(const cudf::column_view& input,
                                                               uint64_t start_idx,
                                                               uint64_t len,
                                                               rmm::device_async_resource_ref mr)
{
  return DoSubstring(input.data<char>(),
                     input.size(),
                     input.child(0).data<cudf::size_type>(),
                     static_cast<cudf::size_type>(start_idx),
                     static_cast<cudf::size_type>(len),
                     mr);
};

//----------String Matching----------//
template <StringMatchingType MatchType>
std::unique_ptr<cudf::column>
GpuDispatcher::DispatchStringMatching(const cudf::column_view& input,
                                      const std::string& match_str,
                                      rmm::device_async_resource_ref mr)
{
  auto stream = cudf::get_default_stream();
  cudf::strings_column_view input_view(input);
  const auto byte_count = static_cast<cudf::size_type>(input_view.chars_size(stream));

  if constexpr (MatchType == StringMatchingType::LIKE || MatchType == StringMatchingType::NOT_LIKE)
  {
    std::vector<std::string> match_terms = string_split(match_str, SPLIT_DELIMITER);
    auto result = DoMultiStringMatching(input.data<char>(),
                                        input.size(),
                                        input.child(0).data<cudf::size_type>(),
                                        byte_count,
                                        match_terms,
                                        mr);
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
                            input.child(0).data<cudf::size_type>(),
                            byte_count,
                            match_str,
                            mr);
  }
  else if constexpr (MatchType == StringMatchingType::PREFIX)
  {
    return DoPrefixMatching(input.data<char>(),
                            input.size(),
                            input.child(0).data<cudf::size_type>(),
                            byte_count,
                            match_str,
                            mr);
  }
}

std::tuple<uint64_t*, uint64_t> GpuDispatcher::DispatchSelect(const cudf::column_view& bitmap,
                                                              rmm::device_async_resource_ref mr)
{
  auto stream              = cudf::get_default_stream();
  auto* gpu_buffer_manager = &GPUBufferManager::GetInstance();

  // The row ids are owned by the query executor and so must be managed by the buffer manager
  uint64_t* row_ids = gpu_buffer_manager->customCudaMalloc<uint64_t>(bitmap.size(), 0, false);
  rmm::device_scalar<uint64_t> d_num_selected(0, stream, mr);

  size_t temp_storage_bytes = 0;
  uint64_t num_selected     = 0;
  cub::DeviceSelect::Flagged(nullptr,
                             temp_storage_bytes,
                             thrust::make_counting_iterator<uint64_t>(0),
                             bitmap.data<bool>(),
                             row_ids,
                             d_num_selected.data(),
                             bitmap.size(),
                             cudf::get_default_stream());
  rmm::device_buffer temp_storage(temp_storage_bytes, stream, mr);
  cub::DeviceSelect::Flagged(temp_storage.data(),
                             temp_storage_bytes,
                             thrust::make_counting_iterator<uint64_t>(0),
                             bitmap.data<bool>(),
                             row_ids,
                             d_num_selected.data(),
                             bitmap.size(),
                             cudf::get_default_stream());
  num_selected = d_num_selected.value(stream);
  return std::make_tuple(row_ids, num_selected);
}

//----------Instantiations----------//
#define INSTANTIATE_STR_MATCHING(T)                                                                \
  template std::unique_ptr<cudf::column>                                                           \
  GpuDispatcher::DispatchStringMatching<StringMatchingType::T>(const cudf::column_view& input,     \
                                                               const std::string& match_str,       \
                                                               rmm::device_async_resource_ref mr);

INSTANTIATE_STR_MATCHING(CONTAINS)
INSTANTIATE_STR_MATCHING(LIKE)
INSTANTIATE_STR_MATCHING(NOT_LIKE)
INSTANTIATE_STR_MATCHING(PREFIX)

#undef SPLIT_DELIMITER
#undef INSTANTIATE_STR_MATCHING

} // namespace sirius
} // namespace duckdb