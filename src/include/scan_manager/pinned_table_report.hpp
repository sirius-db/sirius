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

#include <duckdb/common/typedefs.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace duckdb {
class ClientContext;
}  // namespace duckdb

namespace sirius::scan_manager {

struct pinned_entry;
class sirius_scan_manager;

/// One sirius_pinned_tables() row: a pinned entry's identity, shape, and — for
/// duckdb pins — its live MVCC delta state. Optional fields render as NULL:
/// mvcc fields are absent for parquet pins, live-table fields are absent when
/// the pinned table no longer resolves in the catalog.
struct pinned_table_report {
  std::string name;    ///< pin key (the CALL pin_table name)
  std::string format;  ///< 'duckdb' | 'parquet'
  std::string database_name;
  std::string schema_name;
  std::string table_name;
  std::string tier;  ///< 'gpu' | 'host'
  std::size_t column_count{0};
  std::size_t chunk_count{0};
  /// Rows served from the cache: the mvcc base prefix n_cache() for duckdb
  /// pins (grows with delta promotion), plain num_rows for parquet pins.
  std::size_t base_rows{0};
  bool is_valid{false};  ///< validate_pinned_entry_for_serving verdict

  // duckdb (mvcc) pins only.
  std::optional<duckdb::transaction_t> v_base;
  std::optional<std::size_t> promoted_chunks;
  std::optional<std::size_t> promoted_rows;
  /// Sticky disabled_reason if promotion is off for this pin, else the most
  /// recent skip reason, else absent.
  std::optional<std::string> promotion_status;

  // Live-table state (duckdb pins whose table still resolves in the catalog).
  std::optional<std::size_t> delta_insert_rows;  ///< rows above the cached prefix
  std::optional<std::size_t> delta_delete_rows;  ///< invisible rows inside the prefix
  std::optional<std::size_t> dirty_chunks;       ///< chunks needing a mask this snapshot
  /// True when the pinned DataTable no longer matches the live one (a structural
  /// ALTER / DROP): the cache is over a stale schema and scans decline to CPU.
  /// Absent for parquet pins and when the table no longer resolves.
  std::optional<bool> stale;
};

/// Build one report row for @p entry. Live-table fields resolve through
/// @p context's catalog and transaction; every live probe is best-effort
/// (failure -> the field stays absent, never a throw). SERIAL — call under the
/// query-lifecycle discipline (a table function's execute qualifies).
pinned_table_report make_pinned_table_report(std::string_view name,
                                             pinned_entry const& entry,
                                             duckdb::ClientContext& context);

/// One report per pinned entry, in visit order.
std::vector<pinned_table_report> collect_pinned_table_reports(sirius_scan_manager const& manager,
                                                              duckdb::ClientContext& context);

}  // namespace sirius::scan_manager
