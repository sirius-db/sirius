#ifndef SIMPATICO_CUDF_HANDLE_HPP
#define SIMPATICO_CUDF_HANDLE_HPP

#include <cudf/table/table.hpp>

#include <memory>
#include <string>
#include <vector>

/// Opaque handle used by C API; shared between cudf_shim.cpp and rle_compressor.cu.
struct cudf_table_handle {
  std::unique_ptr<cudf::table> table;
  std::vector<std::string> column_names;
};

struct cudf_table_builder {
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<std::string> names;
};

#endif
