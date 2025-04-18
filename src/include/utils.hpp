#pragma once

#include "duckdb.hpp"

namespace duckdb {

// Required information about exchange table in distributed execution
struct GPUExchangeTableInfo {
  string table_name;
  std::vector<string> column_names;
};

void warmup_gpu();

} // namespace duckdb