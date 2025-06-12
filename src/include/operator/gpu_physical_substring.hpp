#pragma once

#include "gpu_columns.hpp"
#include <tuple>

namespace duckdb {

std::tuple<char*, uint64_t*, uint64_t> PerformSubstring(char* char_data, uint64_t* str_indices, uint64_t num_chars, uint64_t num_strings, uint64_t start_idx, uint64_t length);
shared_ptr<GPUColumn> HandleSubString(shared_ptr<GPUColumn> string_column, uint64_t start_idx, uint64_t length);

// For CuDF compatibility
std::unique_ptr<cudf::column> DoSubstring(const char* input_data,
                                          cudf::size_type input_count,
                                          const int64_t* input_offsets,
                                          int64_t start_idx,
                                          int64_t length,
                                          rmm::device_async_resource_ref mr,
                                          rmm::cuda_stream_view stream = rmm::cuda_stream_default);

} // namespace duckdb