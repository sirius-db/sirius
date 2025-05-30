#pragma once

#include "gpu_columns.hpp"
#include <tuple>

namespace duckdb {

std::tuple<char*, uint64_t*, uint64_t> PerformSubstring(char* char_data, uint64_t* str_indices, uint64_t num_chars, uint64_t num_strings, uint64_t start_idx, uint64_t length);
shared_ptr<GPUColumn> HandleSubString(shared_ptr<GPUColumn> string_column, uint64_t start_idx, uint64_t length);

// For CuDF compatibility
std::unique_ptr<cudf::column> DoSubstring(const char* input_data,
                                          cudf::size_type input_count,
                                          const cudf::size_type* input_offsets,
                                          cudf::size_type start_idx,
                                          cudf::size_type length,
                                          rmm::device_async_resource_ref mr);

} // namespace duckdb