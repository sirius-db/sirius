#include "operator/gpu_physical_substring.hpp"

namespace duckdb {

void HandleSubString(GPUColumn* string_column, uint64_t start_idx, uint64_t length) {
    // Get the current column data
    DataWrapper str_data_wrapper = string_column->data_wrapper;
    uint64_t num_chars = str_data_wrapper.num_bytes;
    char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
    uint64_t num_strings = string_column->column_length;
    uint64_t* d_str_indices = str_data_wrapper.offset;

    // Run the actual kernel
    std::tuple<char*, uint64_t*, uint64_t> result = PerformSubstring(d_char_data, d_str_indices, num_chars, num_strings, start_idx, length);

    // Update the data wrapper
    string_column->data_wrapper.data = reinterpret_cast<uint8_t*>(std::get<0>(result));
    string_column->data_wrapper.offset = std::get<1>(result);
    string_column->data_wrapper.num_bytes = std::get<2>(result);
}

}