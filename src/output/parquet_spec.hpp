// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "helper/logical_type.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace sirius::output {
// Frontend-independent names; never persist cuDF's or DuckDB's enum ordinals.
enum class parquet_compression { uncompressed, snappy, gzip, brotli, zstd, lz4 };
parquet_compression parse_compression(std::string name);
std::string compression_name(parquet_compression codec);

// Recursive logical metadata supplements the physical cuDF schema (notably decimals
// and nested names). The adapter must reject types whose semantics it cannot preserve.
struct parquet_field {
  std::string name;
  sirius::logical_type type;
  std::uint8_t decimal_precision = 0;
  bool is_map                    = false;
  std::vector<parquet_field> children;
};
struct parquet_export_spec {
  std::filesystem::path directory;
  parquet_compression compression = parquet_compression::snappy;
  std::vector<parquet_field> fields;
};
struct parquet_export_result {
  std::uint64_t rows  = 0;
  std::uint64_t files = 0;
  std::uint64_t bytes = 0;
};
}  // namespace sirius::output
