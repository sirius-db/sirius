#pragma once

#include "gpu_columns.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

void StringMatching(char* char_data, uint64_t* str_indices, std::string match_string, uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings);
void MultiStringMatching(char* char_data, uint64_t* str_indices, std::vector<std::string> all_terms,
       uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings);

void HandleStringMatching(GPUColumn* string_column, std::string match_string, uint64_t* &row_id, uint64_t* &num_match_rows);
void HandleMultiStringMatching(GPUColumn* string_column, std::string match_string, uint64_t* &row_id, uint64_t* &num_match_rows);

} // namespace duckdb