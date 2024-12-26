#include "operator/gpu_physical_strings_matching.hpp"

namespace duckdb {

// Code from https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
std::vector<std::string> string_split(std::string s, std::string delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
      token = s.substr(0, pos);
      if(token.length() > 0) {
        tokens.push_back(token);
      }
      s.erase(0, pos + delimiter.length());
    }

    if(s.length() > 0) {
      tokens.push_back(s);
    }

    return tokens;
}

void HandleStringMatching(GPUColumn* string_column, std::string match_string, uint64_t* &row_id, uint64_t* &count, int not_equal) {

    DataWrapper str_data_wrapper = string_column->data_wrapper;
    uint64_t num_chars = str_data_wrapper.num_bytes;
    char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
    uint64_t num_strings = string_column->column_length;
    uint64_t* d_str_indices = str_data_wrapper.offset;

    StringMatching(d_char_data, d_str_indices, match_string, row_id, count, num_chars, num_strings, not_equal);

}

void HandleMultiStringMatching(GPUColumn* string_column, std::string match_string, uint64_t* &row_id, uint64_t* &count, int not_equal) {

    std::vector<std::string> match_terms = string_split(match_string, "%");
    DataWrapper str_data_wrapper = string_column->data_wrapper;
    uint64_t num_chars = str_data_wrapper.num_bytes;
    char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
    uint64_t num_strings = string_column->column_length;
    uint64_t* d_str_indices = reinterpret_cast<uint64_t*>(str_data_wrapper.offset);

    MultiStringMatching(d_char_data, d_str_indices, match_terms, row_id, count, num_chars, num_strings, not_equal);

}

void HandlePrefixMatching(GPUColumn* string_column, std::string match_prefix, uint64_t* &row_id, uint64_t* &count, int not_equal) {
  
    DataWrapper str_data_wrapper = string_column->data_wrapper;
    uint64_t num_chars = str_data_wrapper.num_bytes;
    char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
    uint64_t num_strings = string_column->column_length;
    uint64_t* d_str_indices = str_data_wrapper.offset;
    PrefixMatching(d_char_data, d_str_indices, match_prefix, row_id, count, num_chars, num_strings, not_equal);
}

} // namespace duckdb