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

//! @file
//! Connection classification and per-attempt GPU exclusion.

#pragma once

#include <cstdint>

namespace duckdb {
class ClientContext;
class LogicalOperator;
class SiriusConnectionState;
class SiriusContext;
struct StatementProperties;
}  // namespace duckdb

namespace sirius::transparent {

/// Connection classification used by transparent execution.
enum class connection_provenance : uint8_t {
  unclassified,
  user,               ///< no provider marker or fingerprint; rechecked on the next attempt
  provider_internal,  ///< provider marker or fingerprint found; latched for the connection
};

/// Reason for excluding a planning attempt from GPU execution.
enum class decline_reason : uint8_t {
  none,
  provider_internal,
  hidden_catalog,         ///< a registered catalog or scanned table is HIDDEN
  classification_failed,  ///< inspection failed or a registered catalog identity did not match
};

struct classification_result {
  connection_provenance provenance;  ///< unchanged on classification failure
  decline_reason reason;             ///< none, provider_internal or classification_failed
};

/// Latch provider_internal when the connection carries the DuckLake internal-connection
/// marker in its registered state, or, for DuckLake builds without the marker, when the
/// default catalog is HIDDEN and this connection's effective catalog_error_max_schemas is 0.
/// A latched connection is not reclassified. Exceptions return classification_failed
/// without changing the stored provenance.
classification_result classify_connection_provenance(duckdb::ClientContext& context,
                                                     duckdb::SiriusConnectionState& state) noexcept;

/// Inspect LogicalGet tables for HIDDEN catalogs. GetTable() can run extension code;
/// exceptions yield classification_failed unless a hidden table is also found.
decline_reason inspect_plan_for_hidden_catalog(duckdb::LogicalOperator const& plan) noexcept;

/// Inspect binder-registered read and modified catalogs, independent of optimizer hooks.
/// Resolve transaction names first and require the registered oid to match. Missing or
/// mismatched catalogs yield classification_failed; a hidden catalog takes precedence.
decline_reason inspect_statement_for_hidden_catalog(
  duckdb::ClientContext& context, duckdb::StatementProperties const& properties) noexcept;

/// The one decision the pre-optimizer hook, the optimizer hook and OnFinalizePrepare make
/// before their GPU gates: a non-none reason means this planning attempt stays on DuckDB.
/// Checks in order: the connection (once per attempt, latched per connection), the binder's
/// statement properties when given, the plan when given. The first decline is stamped on the
/// connection state and counted once; later calls in the same attempt return the stored reason.
/// Returns none when the connection carries no Sirius state.
decline_reason should_use_duckdb(duckdb::ClientContext& context,
                                 duckdb::LogicalOperator const* plan,
                                 duckdb::StatementProperties const* properties) noexcept;

}  // namespace sirius::transparent
