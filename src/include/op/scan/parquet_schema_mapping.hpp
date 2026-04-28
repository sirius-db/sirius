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

// cudf
#include <cudf/io/parquet_schema.hpp>

// standard library
#include <cstddef>
#include <string>
#include <vector>

namespace sirius::op::scan::detail {

/**
 * @brief Return the parquet column-chunk indices whose top-level schema name
 *        (path_in_schema[0]) matches @p column_name.
 *
 * The lookup is performed against the column chunks of the first row group,
 * which is authoritative for column-chunk ordering in a single file — parquet
 * guarantees that every row group in a file has column chunks in the same
 * schema-leaf order.
 *
 * For flat schemas, this returns at most one index (0 if the column is absent).
 * For nested schemas (STRUCT, LIST), it returns one index per leaf chunk under
 * the named top-level column — e.g. a STRUCT with three fields yields three
 * leaf indices, all sharing the same path_in_schema[0].
 *
 * Matching is case-sensitive by design. @p column_name is expected to come
 * from DuckDB's parquet bind, which already extracted the name verbatim from
 * the file's schema — so we are comparing a file-schema name to itself and
 * exact equality is sufficient. DuckDB's SQL-layer case-insensitivity is
 * resolved upstream of this call and is not relevant here. Parquet and cuDF
 * both treat column names as case-sensitive, so broadening the match would
 * silently collapse the pathological (but legal) case of two file-schema
 * columns differing only in case.
 *
 * Returns an empty vector if @p metadata has no row groups (the file is empty
 * and has no chunk-ordering information to report).
 *
 * @param metadata     Parsed parquet file metadata.
 * @param column_name  Top-level column name to look up.
 * @return             Parquet column-chunk indices (into
 *                     row_groups[i].columns) that belong to @p column_name.
 */
std::vector<std::size_t> leaf_indices_for_column(cudf::io::parquet::FileMetaData const& metadata,
                                                 std::string const& column_name);

}  // namespace sirius::op::scan::detail
