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

#include <duckdb/common/identifier.hpp>
#include <duckdb/common/shared_ptr.hpp>
#include <duckdb/common/typedefs.hpp>

namespace duckdb {
class ClientContext;
class AttachedDatabase;
}  // namespace duckdb

namespace sirius::helper {

/// Returns the MVCC-domain start_time of the active query's DuckTransaction on
/// the default-database catalog. Use this when you need the value that
/// RowVersionManager::GetSelVector(ScanOptions{start_time}, ...) compares
/// against per-row insert_id / delete_id for visibility.
///
/// Not the MetaTransaction value: MetaTransaction::start_timestamp is a
/// wall-clock timestamp_t and MetaTransaction::global_transaction_id is the
/// transaction's global identifier; neither lives in the MVCC start_time
/// domain. Only DuckTransaction::start_time (per AttachedDatabase) does.
///
/// Multi-AttachedDatabase note: each DuckTransaction inside a MetaTransaction
/// has its own start_time counter. This helper returns the default DB's value.
/// Callers targeting a specific AttachedDatabase should call
/// DuckTransaction::Get(context, that_catalog).start_time directly.
duckdb::transaction_t get_query_start_time(duckdb::ClientContext& context);

duckdb::shared_ptr<duckdb::AttachedDatabase> find_global_database(duckdb::ClientContext& context,
                                                                  duckdb::Identifier const& name);

}  // namespace sirius::helper
