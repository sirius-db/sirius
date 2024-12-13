#pragma once

#include "gpu_columns.hpp"
#include "gpu_buffer_manager.hpp"
#include <tuple>

namespace duckdb {

std::tuple<char*, uint64_t*, uint64_t> PerformSubstring(char* char_data, uint64_t* str_indices, uint64_t num_chars, uint64_t num_strings, uint64_t start_idx, uint64_t length);
GPUColumn* HandleSubString(GPUColumn* string_column, uint64_t start_idx, uint64_t length);

} // namespace duckdb