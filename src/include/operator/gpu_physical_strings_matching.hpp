#pragma once

#include "gpu_columns.hpp"

namespace duckdb {

std::vector<std::string> string_split(std::string s, std::string delimiter);
void StringMatching(char* char_data, uint64_t* str_indices, std::string match_string, uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings, int not_equal);
void MultiStringMatching(char* char_data, uint64_t* str_indices, std::vector<std::string> all_terms,
       uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_string, int not_equal);
void PrefixMatching(char* char_data, uint64_t* str_indices, std::string match_prefix, uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings, int not_equal);

void HandleStringMatching(shared_ptr<GPUColumn> string_column, std::string match_string, uint64_t* &row_id, uint64_t* &count, int not_equal);
void HandleMultiStringMatching(shared_ptr<GPUColumn> string_column, std::string match_string, uint64_t* &row_id, uint64_t* &count, int not_equal);
void HandlePrefixMatching(shared_ptr<GPUColumn> string_column, std::string match_prefix, uint64_t* &row_id, uint64_t* &count, int not_equal);

// For CuDF compatibility
std::unique_ptr<cudf::column>
DoStringMatching(const char* input_data,
                 cudf::size_type input_count,
                 const int64_t* input_offsets,
                 int64_t byte_count,
                 const std::string& match_string,
                 rmm::device_async_resource_ref mr,
                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);
std::unique_ptr<cudf::column>
DoMultiStringMatching(const char* input_data,
                      cudf::size_type input_count,
                      const int64_t* input_offsets,
                      int64_t byte_count,
                      const std::vector<std::string>& match_strings,
                      rmm::device_async_resource_ref mr,
                      rmm::cuda_stream_view stream = rmm::cuda_stream_default);
std::unique_ptr<cudf::column>
DoPrefixMatching(const char* input_data,
                 cudf::size_type input_count,
                 const int64_t* input_offsets,
                 int64_t byte_count,
                 const std::string& match_prefix,
                 rmm::device_async_resource_ref mr,
                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);

} // namespace duckdb