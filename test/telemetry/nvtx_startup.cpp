// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "duckdb.hpp"
#include "sirius/ffi.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

// This helper intentionally has no test fixtures: the first extension load must
// exercise process startup before NVTX or Quent have cached any state.
int main(int argc, char** argv)
{
  try {
    if (argc == 2 && std::string_view{argv[1]} == "ffi") {
      auto const* path = std::getenv("SIRIUS_CONFIG_FILE");
      if (path == nullptr) { throw std::runtime_error("missing FFI config path"); }
      sirius::ffi::Context context{path};
      return 0;
    }
    duckdb::DuckDB db(nullptr);
    duckdb::Connection connection(db);
    auto settings = connection.Query("SET enable_duckdb_fallback = false");
    if (settings->HasError()) { throw std::runtime_error(settings->GetError()); }
    auto result = connection.Query(
      "SELECT k, sum(v)::BIGINT FROM (VALUES (1, 10), (1, 20), (2, 5)) t(k, v) "
      "GROUP BY k ORDER BY k");
    if (result->HasError()) { throw std::runtime_error(result->GetError()); }
    if (result->RowCount() != 2 || result->GetValue(0, 0).GetValue<int32_t>() != 1 ||
        result->GetValue(1, 0).GetValue<int64_t>() != 30 ||
        result->GetValue(0, 1).GetValue<int32_t>() != 2 ||
        result->GetValue(1, 1).GetValue<int64_t>() != 5) {
      throw std::runtime_error("unexpected GPU result");
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
