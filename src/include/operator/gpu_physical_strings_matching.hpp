#pragma once

#include "gpu_columns.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

uint64_t* StringMatching(GPUColumn* string_column, std::string match_string, uint64_t* num_match_rows); 
uint64_t* MultiStringMatching(GPUColumn* string_column, std::vector<std::string> all_terms, uint64_t* num_match_rows); 

}