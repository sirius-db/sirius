#pragma once

#include "gpu_columns.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

void StringMatching(char* char_data, uint64_t* str_indices, std::string match_string, uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings, int not_equal);
void MultiStringMatching(char* char_data, uint64_t* str_indices, std::vector<std::string> all_terms,
       uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_string, int not_equal);
void PrefixMatching(char* char_data, uint64_t* str_indices, std::string match_prefix, uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings, int not_equal);

void HandleStringMatching(shared_ptr<GPUColumn> string_column, std::string match_string, uint64_t* &row_id, uint64_t* &count, int not_equal);
void HandleMultiStringMatching(shared_ptr<GPUColumn> string_column, std::string match_string, uint64_t* &row_id, uint64_t* &count, int not_equal);
void HandlePrefixMatching(shared_ptr<GPUColumn> string_column, std::string match_prefix, uint64_t* &row_id, uint64_t* &count, int not_equal);

} // namespace duckdb