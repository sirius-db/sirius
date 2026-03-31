/*
 * Copyright 2025, Sirius Contributors.
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

#include <duckdb/main/client_context.hpp>

#include <string>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Discovered Iceberg delete-file metadata for one table.
 */
struct IcebergDeleteFiles {
  std::vector<std::string> positional_delete_files;
  std::vector<std::string> equality_delete_files;
};

/**
 * @brief Read Iceberg delete-file metadata for the given table path.
 *
 * Uses a single discovery path:
 *   1. iceberg_snapshots() → manifest_list path
 *   2. Avro manifest-list reader → list of manifests with content types
 *   3. Avro manifest reader → delete file paths per content type
 *
 * This avoids DuckDB's iceberg_metadata() which fails on tables with
 * equality-delete manifests in DuckDB 1.4.4.
 *
 * Falls back gracefully on errors (treats as V1 with no deletes).
 *
 * @param context    DuckDB client context for running iceberg_snapshots().
 * @param table_path The Iceberg table path passed to iceberg_scan().
 * @return Delete file metadata (both vectors empty for V1 tables).
 */
IcebergDeleteFiles read_iceberg_delete_metadata(duckdb::ClientContext& context,
                                                std::string const& table_path);

}  // namespace sirius::op::scan
