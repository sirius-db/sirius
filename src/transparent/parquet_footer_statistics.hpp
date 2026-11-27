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

#include <duckdb/function/function.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace duckdb {
class LogicalGet;
}  // namespace duckdb

namespace duckdb_parquet {
class FileMetaData;
}  // namespace duckdb_parquet

namespace sirius::transparent {

/// \brief Give every multi-file Parquet scan in \p plan the statistics of all its files' footers.
///
/// DuckDB reads column statistics for a scan of several Parquet files only with `union_by_name`;
/// otherwise its optimizer has none and estimates every scan filter at a fixed selectivity, even an
/// `IS NOT NULL` on a column without NULLs. For each such scan this merges the footer statistics of
/// all its files and hands them to DuckDB through the scan's `column_statistics` option, which the
/// optimizer then uses as it would a single file's statistics. Per-file statistics are cached for
/// the process and reused while the file keeps its size, inode and version tag (an object store's
/// ETag) or a modification time old enough to trust. Checking a remote file costs one request, so
/// the files are checked and read concurrently.
///
/// A scan is covered whole or not at all, since the statistics of some of its files are not
/// statistics of the scan. The footers a scan would have read, those not cached for the file's
/// current version and the scan's reader options, are counted first, remote and local files apart.
/// If either count exceeds its limit the scan is left as DuckDB bound it: no footer is read and its
/// cached files are not used. The limits are the connection's
/// `parquet_footer_statistics_max_remote_reads` and `parquet_footer_statistics_max_local_reads`,
/// or where those are NULL the engine's `operator_params` of the same names: -1 for no limit, 0 for
/// cached statistics only.
///
/// Must run before DuckDB's optimizer, from the pre-optimizer hook.
///
/// \return The number of scans given statistics.
std::size_t add_parquet_footer_statistics(duckdb::ClientContext& context,
                                          duckdb::LogicalOperator& plan);

/// The most footers add_parquet_footer_statistics may read for one scan; a negative limit is none.
struct footer_read_limits {
  std::int64_t remote = -1;
  std::int64_t local  = -1;
};

/// \brief The limits in force for a planning on \p context: the connection's settings, and the
/// engine configuration where the connection has set none.
[[nodiscard]] footer_read_limits footer_read_limits_of(duckdb::ClientContext& context);

/// \brief The number of rows in all files of \p get, a Parquet scan add_parquet_footer_statistics
/// covered (a one-file scan too), or nullopt. Exact and unfiltered: the sum of the footers' row
/// counts. DuckDB's own cardinality for a multi-file scan extrapolates from one file.
///
/// The count belongs to the plan whose planning covered the scan: it is found through the file
/// list of the scan's bind data. Another plan over the same paths does not have it, and neither
/// does a scan whose file list DuckDB replaced while optimizing. A copy of the plan has it only
/// through copy_parquet_scan_row_count.
///
/// A count is kept, without a capacity limit, while the scan's file list, or the list of a copy
/// that received it through copy_parquet_scan_row_count, is alive; entries of lists that are gone
/// are dropped lazily.
[[nodiscard]] std::optional<std::size_t> parquet_scan_row_count(duckdb::LogicalGet const& get);

/// \brief The same count for a scan that is no longer a LogicalGet: \p function_name and
/// \p bind_data are the function name and the bind data of the LogicalGet the scan was planned
/// from. The bind data must be that LogicalGet's own object, not a copy of it: the count is found
/// through its file list.
[[nodiscard]] std::optional<std::size_t> parquet_scan_row_count(
  std::string_view function_name, duckdb::FunctionData const* bind_data);

/// \brief How many values lie from the smallest to the largest that column \p column holds across
/// all files of \p get (max - min + 1), or nullopt when the footers give it no integer range.
/// \p get is a Parquet scan add_parquet_footer_statistics covered, a one-file scan too, and
/// \p column counts the scan's columns as its bind names them. The column cannot hold more distinct
/// values than that. Integers, DATEs and DECIMALs of up to 18 digits have such a range.
///
/// The spans are kept with the scan's row count: a plan and its copies have them exactly when
/// parquet_scan_row_count answers, so a scan over a read limit has neither.
[[nodiscard]] std::optional<std::uint64_t> parquet_scan_value_span(duckdb::LogicalGet const& get,
                                                                   std::size_t column);

/// \brief What Parquet footers state about the decoded size of one string column.
struct string_size {
  /// Decoded bytes of the column's strings, offsets excluded, in the rows counted below
  /// (`unencoded_byte_array_data_bytes`; encoded sizes do not describe decoded strings).
  std::uint64_t bytes = 0;
  /// Rows of the row groups that state the column's decoded size; 0 when none does.
  std::uint64_t rows = 0;

  [[nodiscard]] bool operator==(string_size const&) const = default;
};

/// \brief The decoded string sizes the footers of a Parquet scan state, indexed by the scan's
/// bound column (`MultiFileBindData::names`) and summed over all its files. A file or row group
/// that states no size for a column adds nothing to it, so `rows` can be less than the scan's row
/// count; GPU admission prices the remaining rows at the configured average. Only top-level
/// string columns have an entry with rows. Empty when no column has one, the scan has no row
/// count, or GPU admission was not sizing queries when the scan was planned
/// (planner::admission_estimate_needed). The arguments are those of parquet_scan_row_count, and
/// the sizes are kept, copied and forgotten with the row count.
[[nodiscard]] std::vector<string_size> parquet_scan_string_sizes(
  std::string_view function_name, duckdb::FunctionData const* bind_data);

/// \brief Make the row count of every Parquet scan in \p plan unknown, and its value spans and
/// string sizes.
void forget_parquet_scan_row_counts(duckdb::LogicalOperator const& plan);

/// \brief Give \p to, the copy of the scan \p from in a copy of its plan, the row count of \p from
/// and its value spans and string sizes.
void copy_parquet_scan_row_count(duckdb::LogicalGet const& from, duckdb::LogicalGet const& to);

/// \brief Read the footer statistics of \p paths into the cache add_parquet_footer_statistics uses,
/// so the first query over them does not pay for reading every footer.
///
/// The footers are read with DuckDB's default Parquet options, so only scans with those options
/// reuse them; a scan with other options reads the footers again for its own statistics. The read
/// limits of add_parquet_footer_statistics do not apply. All files are processed; only records
/// that fit remain cached.
///
/// \return The number of files whose statistics were read or found cached.
std::size_t read_parquet_footer_statistics(duckdb::ClientContext& context,
                                           std::vector<std::string> const& paths);

/// \brief The rows in all of \p paths, summed from their footers through the same cache as
/// read_parquet_footer_statistics. Throws when a footer cannot be read.
[[nodiscard]] std::size_t parquet_files_row_count(duckdb::ClientContext& context,
                                                  std::vector<std::string> const& paths);

namespace detail {
/// What the caches keep once no planning is running.
struct cache_capacities {
  std::size_t statistics;  ///< Files whose statistics are kept, each reader-options set apart.
  std::size_t merges;      ///< Merged statistics kept.
  std::size_t merge_references;  ///< Files all kept merged statistics were merged from, together.
};
/// No capacity limit.
inline constexpr std::size_t unbounded = std::numeric_limits<std::size_t>::max();
/// No limit on per-file statistics or on the files merged statistics refer to. An evicted
/// per-file record is a miss at the next planning: its footer is read again, or it counts against
/// a read limit. Merged statistics evicted for their references are merged again.
inline constexpr cache_capacities default_cache_capacities{unbounded, 128, unbounded};
/// The most sets of scan paths whose merged statistics are kept.
inline constexpr std::size_t merged_statistics_cache_capacity = default_cache_capacities.merges;
/// The number of scans whose merged statistics are cached, for tests.
std::size_t merged_statistics_cache_size();
/// The number of scans of live plans that have a row count, a plan's copies counted once; for
/// tests. Drops what is kept for file lists that are gone.
std::size_t row_count_cache_size();
/// The row counts kept, those of file lists that are gone and not yet dropped included; for tests.
std::size_t row_count_association_count();
/// The number of files whose statistics are cached, for tests.
std::size_t statistics_cache_size();
/// The number of files the cached merged statistics were merged from, for tests.
std::size_t merged_reference_count();
/// The number of per-file statistics alive in the process, cached or in use by a planning.
std::size_t live_statistics_records();
/// The number of footers read for statistics since the process started.
std::uint64_t footer_read_count();
/// The file entries inspected by the footer admission and by the unlimited cache walks that
/// completed normally in this process, for tests. Includes the entry that exceeds a limit.
std::uint64_t inspected_file_count();
/// For tests: replace the capacities, drop what exceeds them and return the previous ones.
cache_capacities set_cache_capacities_for_testing(cache_capacities replacement);
/// For tests: drop every cached statistic, merged statistic and row count.
void clear_caches_for_testing();
/// The decoded string size one file's footer states for each of its columns, by leaf. A column
/// has rows only when it is a top-level byte array (the leaf of a LIST<VARCHAR> or of a STRUCT
/// field is not), and only for the row groups that state its decoded size. Empty when the file
/// has no row group or not one leaf per bound column (\p bound_columns), as with a STRUCT of
/// several fields: the leaves then cannot be matched to columns by position.
std::vector<string_size> parquet_string_sizes(duckdb_parquet::FileMetaData const& metadata,
                                              std::size_t bound_columns);
/// One column's size over several files: the sum of what each states. Nothing (zero rows) when
/// the sum overflows.
string_size combined_string_size(std::vector<string_size> const& files);
}  // namespace detail

}  // namespace sirius::transparent
