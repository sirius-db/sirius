// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

inline void require_compatible_duckdb(std::string_view sdk_revision,
                                      std::string_view runtime_revision)
{
  if (sdk_revision.empty()) {
    throw std::runtime_error(
      "Sirius UDF SDK has no DuckDB revision identity; ask the provider to rebuild "
      "the SDK with SIRIUS_UDF_DUCKDB_REVISION set to its pinned DuckDB commit");
  }
  auto const is_hash = [](std::string_view value) {
    return value.size() >= 7 && std::all_of(value.begin(), value.end(), [](char digit) {
             return (digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f');
           });
  };
  auto const length = std::min(sdk_revision.size(), runtime_revision.size());
  if (!is_hash(sdk_revision) || !is_hash(runtime_revision) ||
      sdk_revision.substr(0, length) != runtime_revision.substr(0, length)) {
    throw std::runtime_error("Incompatible DuckDB library: SDK revision " +
                             std::string(sdk_revision) + ", runtime revision " +
                             std::string(runtime_revision) +
                             "; use the provider's matching libduckdb.so");
  }
}
