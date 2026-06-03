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

#include <cudf/io/parquet_schema.hpp>

#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

#include <string>

namespace sirius::io::parquet_helpers {

/// Column types + names inferred from a parquet file footer, in file-schema
/// order. Produced at bind time by @c extract_schema and copied into the
/// DuckDB table-function bind out-parameters.
struct schema_info {
  duckdb::vector<duckdb::LogicalType> types;
  duckdb::vector<std::string> names;
};

/// Walk the top-level columns of a parquet @c FileMetaData and map each to a
/// DuckDB @c LogicalType.
///
/// Leaf columns map by physical/logical type: BOOLEAN, INT32 / INT64 (and their
/// signed/unsigned width variants), INT96, FLOAT, DOUBLE, BYTE_ARRAY /
/// FIXED_LEN_BYTE_ARRAY (UTF8 → VARCHAR, otherwise BLOB), DECIMAL, DATE, TIME
/// and TIMESTAMP.
///
/// Nested columns recurse over the preorder schema subtree: a plain group →
/// @c LogicalType::STRUCT(children); a group annotated LIST (the standard
/// 3-level @c LIST → repeated group → element encoding) → @c LogicalType::LIST;
/// a group annotated MAP (@c MAP → repeated key_value → {key, value}) →
/// @c LogicalType::MAP(key, value). The mapping matches DuckDB's own
/// @c read_parquet bind shape for the same file.
///
/// @throws std::runtime_error on a malformed / truncated nested subtree or an
///         unsupported physical type.
schema_info extract_schema(cudf::io::parquet::FileMetaData const& meta);

}  // namespace sirius::io::parquet_helpers
