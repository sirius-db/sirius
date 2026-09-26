/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "op/scan/table_scan/scan_contract.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_schema.hpp>

#include <span>

namespace sirius::op::scan {
struct parquet_encryption_evidence {
  bool footer_encrypted  = false;
  bool columns_encrypted = false;
  bool complete          = false;
};
// Compact-Thrift evidence is independent of cuDF's schema, which drops crypto fields.
parquet_encryption_evidence inspect_parquet_encryption(std::span<uint8_t const> footer);
std::unique_ptr<cudf::io::datasource::buffer> fetch_plaintext_parquet_footer(
  cudf::io::datasource& source, scan_contract_id contract, std::string const& identity);

struct physical_profile_result {
  bool approved         = false;
  verdict_reason reason = verdict_reason::none;
  std::string text;
  validation_set validation;
  profile_id profile       = 0;
  uint64_t type_mismatches = 0;
};
std::optional<type_conversion> qualified_type_conversion(parquet_physical_type,
                                                         parquet_logical_type,
                                                         duckdb::LogicalType const&);
physical_profile_result check_parquet_split_profile(
  cudf::io::parquet::FileMetaData const&,
  parquet_encryption_evidence const&,
  bound_table_scan const&,
  effective_reader_projection const&,
  std::span<std::size_t const> retained_row_groups,
  leaf_set const& semantic_columns);

struct iceberg_table_schema {
  struct field {
    std::string name;
    int32_t id;
    std::string type;
  };
  std::vector<field> fields;  // Preorder, containers included with empty type.
};
physical_profile_result check_iceberg_file_schema(cudf::io::parquet::FileMetaData const&,
                                                  iceberg_table_schema const&,
                                                  std::string_view probe_path);
}  // namespace sirius::op::scan
