#include "operator/gpu_physical_substring.hpp"

namespace duckdb {

GPUColumn* HandleSubString(GPUColumn* string_column, uint64_t start_idx, uint64_t length) {
    // Get the current column data
    DataWrapper str_data_wrapper = string_column->data_wrapper;
    uint64_t num_chars = str_data_wrapper.num_bytes;
    char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
    uint64_t num_strings = string_column->column_length;
    uint64_t* d_str_indices = str_data_wrapper.offset;
    if (start_idx < 1) throw InvalidInputException("Start index should be greater than 0");
    uint64_t actual_start_idx = start_idx - 1;

    // Run the actual kernel
    std::tuple<char*, uint64_t*, uint64_t> result = PerformSubstring(d_char_data, d_str_indices, num_chars, num_strings, actual_start_idx, length);

    uint8_t* result_data = reinterpret_cast<uint8_t*>(std::get<0>(result));
    uint64_t* result_offset = std::get<1>(result);
    uint64_t result_bytes = std::get<2>(result);

    return new GPUColumn(string_column->column_length, ColumnType::VARCHAR, result_data, result_offset, result_bytes, true);
}

}