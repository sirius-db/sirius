#include "duckdb/common/assert.hpp"
#include "duckdb/common/exception.hpp"
#include "expression_executor/gpu_dispatcher.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_physical_strings_matching.hpp"
#include "gpu_physical_substring.hpp"
#include <cub/cub.cuh>
#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <memory>
#include <rmm/device_uvector.hpp>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
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
  D_ASSERT(start_idx > 0);

  // Convert from cudf to sirius types
  std::cout << "PREPARING STRING DATA...\n";
  auto [input_count, input_data, input_offsets, input_num_bytes] = PrepareStringData(input);

  // Execute the substring operation
  std::cout << "PERFORMING SUBSTRING...\n";
  auto [result_data, result_offsets, result_num_bytes] =
    PerformSubstring(input_data, input_offsets, input_num_bytes, input.size(), start_idx, len);

  std::cout << "Result num bytes: " << result_num_bytes << "\n";

  // Construct a CuDF column (performing deep copies) and return
  std::cout << "MAKING COLUMN...\n";
  return MakeColumnFromPtrs(result_data, result_offsets, input_count, mr);
};

//----------String Matching----------//
template <StringMatchingType MatchType>
std::unique_ptr<cudf::column>
GpuDispatcher::DispatchStringMatching(const cudf::column_view& input,
                                      const std::string& match_str,
                                      rmm::device_async_resource_ref mr)
{

  // Convert from cudf to sirius types
  auto [input_count, input_data, input_offsets, input_num_bytes] = PrepareStringData(input);

  // Execute the string matching
  std::vector<std::string> match_terms = string_split(match_str, SPLIT_DELIMITER);
  uint64_t* result_row_ids             = nullptr;
  uint64_t* result_count_ptr           = nullptr;
  if constexpr (MatchType == StringMatchingType::LIKE)
  {
    MultiStringMatching(input_data,
                        input_offsets,
                        match_terms,
                        result_row_ids,
                        result_count_ptr,
                        input_num_bytes,
                        input_count,
                        false);
  }
  else if constexpr (MatchType == StringMatchingType::NOT_LIKE)
  {
    MultiStringMatching(input_data,
                        input_offsets,
                        match_terms,
                        result_row_ids,
                        result_count_ptr,
                        input_num_bytes,
                        input_count,
                        true);
  }
  else if constexpr (MatchType == StringMatchingType::CONTAINS)
  {
    StringMatching(input_data,
                   input_offsets,
                   match_str,
                   result_row_ids,
                   result_count_ptr,
                   input_num_bytes,
                   input_count,
                   false);
  }
  else if constexpr (MatchType == StringMatchingType::PREFIX)
  {
    PrefixMatching(input_data,
                   input_offsets,
                   match_str,
                   result_row_ids,
                   result_count_ptr,
                   input_num_bytes,
                   input_count,
                   false);
  }

  // Get the number of matches
  uint64_t result_count = *result_count_ptr;
  if (result_count == 0)
  {
    throw InternalException("Dispatch[Like]: Zero matches found!");
  }

  // Make boolean column result and return
  return MakeColumnFromRowIds(result_row_ids, result_count, input_count, mr);
}

// std::tuple<rmm::device_uvector<uint64_t>, uint64_t>
std::tuple<uint64_t*, uint64_t> GpuDispatcher::DispatchSelect(const cudf::column_view& bitmap,
                                                              rmm::device_async_resource_ref mr)
{
  auto stream = cudf::get_default_stream();
  // The row ids
  uint64_t* row_ids =
    GPUBufferManager::GetInstance().customCudaMalloc<uint64_t>(bitmap.size(), 0, false);
  rmm::device_scalar<uint64_t> d_num_selected(0, stream, mr);

  size_t temp_storage_bytes = 0;
  uint64_t num_selected     = 0;
  cub::DeviceSelect::Flagged(nullptr,
                             temp_storage_bytes,
                             thrust::make_counting_iterator<uint64_t>(0),
                             bitmap.data<bool>(),
                             // row_ids.data(),
                             row_ids,
                             d_num_selected.data(),
                             bitmap.size(),
                             cudf::get_default_stream());
  rmm::device_buffer temp_storage(temp_storage_bytes, stream, mr);
  cub::DeviceSelect::Flagged(temp_storage.data(),
                             temp_storage_bytes,
                             thrust::make_counting_iterator<uint64_t>(0),
                             bitmap.data<bool>(),
                             // row_ids.data(),
                             row_ids,
                             d_num_selected.data(),
                             bitmap.size(),
                             cudf::get_default_stream());
  CubDebugExit(cudaDeviceSynchronize());
  num_selected = d_num_selected.value(stream);
  return std::make_tuple(row_ids, num_selected);
}

//----------Utilities----------//

/*
 * This is admittedly hack-ey for strings, since, for each string function, we must convert the
 * underlying offsets from cudf::size_type to uint64_t and then back in order to use the sirius
 * string APIs. [Put GitHub Issue link here when it exists]
 * There are 4 extra I/Os for the conversion and 4 extra I/Os to copy the raw pointers to
 * cudf::columns, resulting 8 superfluous I/Os...
 */

std::unique_ptr<cudf::column> GpuDispatcher::MakeColumnFromPtrs(char* data,
                                                                uint64_t* offsets,
                                                                uint64_t count,
                                                                rmm::device_async_resource_ref mr)
{
  uint64_t offsets_count = count + 1;

  // First, convert offsets back to cudf::size_type
  int32_t* offsets_cudf = convertUInt64ToInt32(offsets, offsets_count);

  // Next, construct views from which to manufacture the column
  cudf::column_view offsets_view(cudf::data_type(cudf::type_id::INT32),
                                 static_cast<cudf::size_type>(offsets_count),
                                 reinterpret_cast<void*>(offsets_cudf),
                                 nullptr,
                                 0);
  std::vector<cudf::column_view> children = {offsets_view};
  cudf::column_view string_view(cudf::data_type(cudf::type_id::STRING),
                                static_cast<cudf::size_type>(count),
                                reinterpret_cast<void*>(data),
                                nullptr,
                                0,
                                0,
                                children);

  // Wrap in strings_column_view for printing
  cudf::strings_column_view str_view(string_view);

  // Manufacture the column (deep copies) and return
  return std::make_unique<cudf::column>(string_view, cudf::get_default_stream(), mr);
}

std::unique_ptr<cudf::column> GpuDispatcher::MakeColumnFromRowIds(uint64_t* row_ids,
                                                                  uint64_t row_id_count,
                                                                  uint64_t input_count,
                                                                  rmm::device_async_resource_ref mr)
{
  auto stream = cudf::get_default_stream();

  // Allocate output buffer and initialize to false
  rmm::device_uvector<bool> output_buffer(input_count, stream, mr);
  thrust::fill(thrust::cuda::par.on(stream), output_buffer.begin(), output_buffer.end(), false);

  // Scatter true values to locations specified by the row ids
  auto true_iterator  = thrust::make_constant_iterator(true);
  auto rowid_iterator = thrust::device_ptr<uint64_t>(row_ids);
  thrust::scatter(thrust::cuda::par.on(stream),
                  true_iterator,
                  true_iterator + static_cast<int64_t>(row_id_count),
                  rowid_iterator,
                  output_buffer.begin());

  // Make the (boolean) cudf column (zero-copy)
  return std::make_unique<cudf::column>(std::move(output_buffer),
                                        rmm::device_buffer{0, stream, mr},
                                        0);
}

std::tuple<uint64_t, char*, uint64_t*, uint64_t>
GpuDispatcher::PrepareStringData(const cudf::column_view& input)
{
  // We must convert offsets to uint64_t and cast away constness...
  uint64_t input_count   = input.size();
  char* input_data       = const_cast<char*>(input.data<char>()); /// !!!CONST CAST!!!
  uint64_t offsets_count = input_count + 1;
  uint64_t* input_offsets =
    convertInt32ToUInt64(const_cast<int32_t*>(input.child(0).data<int32_t>()),
                         offsets_count); /// !!!CONST CAST!!!
  uint64_t input_num_bytes = 0;
  CUDF_CUDA_TRY(cudaMemcpy(&input_num_bytes,
                           input_offsets + input_count,
                           sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));
  std::cout << "Num bytes: " << input_num_bytes << "\n";
  return std::make_tuple(input_count, input_data, input_offsets, input_num_bytes);
}

//----------Instantiations----------//
template std::unique_ptr<cudf::column>
GpuDispatcher::DispatchStringMatching<StringMatchingType::CONTAINS>(
  const cudf::column_view& input,
  const std::string& match_str,
  rmm::device_async_resource_ref mr);
template std::unique_ptr<cudf::column>
GpuDispatcher::DispatchStringMatching<StringMatchingType::LIKE>(const cudf::column_view& input,
                                                                const std::string& match_str,
                                                                rmm::device_async_resource_ref mr);
template std::unique_ptr<cudf::column>
GpuDispatcher::DispatchStringMatching<StringMatchingType::NOT_LIKE>(
  const cudf::column_view& input,
  const std::string& match_str,
  rmm::device_async_resource_ref mr);
template std::unique_ptr<cudf::column>
GpuDispatcher::DispatchStringMatching<StringMatchingType::PREFIX>(
  const cudf::column_view& input,
  const std::string& match_str,
  rmm::device_async_resource_ref mr);

#undef SPLIT_DELIMITER

} // namespace sirius
} // namespace duckdb