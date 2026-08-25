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
#include <duckdb/main/connection.hpp>
#include <sirius_context.hpp>

#include <string>

namespace sirius::op::scan {

/**
 * @brief The one place Sirius opens a DuckDB connection to read Iceberg metadata while planning.
 *
 * ## Why a connection at all
 *
 * Iceberg metadata lives behind `duckdb-iceberg`'s table functions. Sirius does not link against
 * that extension, and the facts it needs are not on the bound plan: `MultiFileBindData` exposes no
 * snapshot, `OpenFileInfo::extended_info` carries only file_size/etag/last_modified/first_row_id/
 * sequence_number, and duckdb-iceberg overrides no `GetBindInfo`. So the delete files, the
 * manifest entries and the data-file footers are reachable only by asking DuckDB for them.
 *
 * ## Why a SECOND connection cannot see a different table than the bind did
 *
 * This was the reviewers' hazard, and it is closed by construction rather than by agreement: an
 * `iceberg_scan` with no `snapshot_from_id` is declined at plan time, so every query issued
 * through this connection names the SAME snapshot id the outer scan was bound to. A commit landing
 * mid-plan moves `current`; it does not move a pinned id. Callers that read Iceberg metadata must
 * therefore pass `snapshot_from_id` through, and there is no path here that resolves "current".
 *
 * ## What it deliberately does NOT do
 *
 * It does not force settings on. A fresh connection inherits GLOBAL settings from the shared
 * `DatabaseInstance` but not session-scoped ones, so a session-level flag is MIRRORED from the
 * outer context. Forcing `unsafe_enable_version_guessing = true` here would mean Sirius reading a
 * table the user's own session refuses to read, and deciding on its behalf that a table with no
 * version hint may be guessed at. When the outer session has not enabled it, the query fails, the
 * caller's "any failure declines" rule fires, and the table is read by DuckDB instead.
 */
class iceberg_metadata_connection {
 public:
  explicit iceberg_metadata_connection(duckdb::ClientContext& context)
    : _conn(*context.db), _conn_guard(*_conn.context)
  {
    // Session-scoped settings that change which tables are LEGIBLE. Every site that reads Iceberg
    // metadata must agree on this set: if the delete gate and the delete discovery disagree about
    // which tables they can read, the gate's verdict describes a different table than the scan.
    static constexpr auto kMirroredSettings = {"unsafe_enable_version_guessing"};
    for (auto const* setting : kMirroredSettings) {
      duckdb::Value value;
      if (context.TryGetCurrentSetting(setting, value) && !value.IsNull() &&
          value.DefaultCastAs(duckdb::LogicalType::BOOLEAN).GetValue<bool>()) {
        _conn.Query(std::string("SET ") + setting + " = true");
      }
    }
  }

  duckdb::Connection& get() { return _conn; }

  duckdb::unique_ptr<duckdb::MaterializedQueryResult> Query(std::string const& sql)
  {
    return _conn.Query(sql);
  }

 private:
  duckdb::Connection _conn;
  // Per-connection bracket. Opening a connection to the same database re-registers the SAME
  // SiriusContext, whose query-lifecycle callbacks would otherwise fire QueryBegin/QueryEnd
  // underneath the query being planned; and unbracketed these contend for the plan slot the outer
  // query already holds. Declared after _conn so it is constructed from a live connection.
  duckdb::SiriusContext::InternalQueryGuard _conn_guard;
};

}  // namespace sirius::op::scan
