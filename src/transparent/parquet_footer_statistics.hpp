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

#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>

namespace sirius::transparent {

/// \brief Give every multi-file Parquet scan in \p plan the statistics of all its files' footers.
///
/// DuckDB reads column statistics for a scan of several Parquet files only with `union_by_name`;
/// otherwise its optimizer has none and estimates every scan filter at a fixed selectivity, even an
/// `IS NOT NULL` on a column without NULLs. For each such scan this merges the footer statistics of
/// all its files and hands them to DuckDB through the scan's `column_statistics` option, which the
/// optimizer then uses as it would a single file's statistics. Per-file statistics are cached for
/// the process and reused while the file keeps its size, inode and a modification time old enough
/// to trust. Remote files are left alone, since checking them costs a request per file.
///
/// Must run before DuckDB's optimizer, from the pre-optimizer hook.
///
/// \return The number of scans given statistics.
std::size_t add_parquet_footer_statistics(duckdb::ClientContext& context,
                                          duckdb::LogicalOperator& plan);

}  // namespace sirius::transparent
