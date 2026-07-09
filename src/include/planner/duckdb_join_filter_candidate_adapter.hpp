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

//===-----------------------------------------------------------------------------------------===//
// The "pinned DuckDB" contract
//
// The dynamic-filter metadata read here — `JoinFilterPushdownInfo`, `JoinFilterPushdownFilter`,
// `DynamicTableFilterSet` — is DuckDB-internal, with no stability guarantee across versions. This
// adapter (header + .cpp) is therefore the ONE module allowed to depend on that layout: everything
// downstream consumes the Sirius value types below and never re-reads DuckDB data structures.
//
// On a submodule pin bump, re-audit this module and its contract tests before anything else.
//
// Boundary status: C1a-1 lands the adapter and routes the preservation reads through it. The
// remaining direct reads (sirius_plan_comparison_join.cpp, sirius_plan_get.cpp, the hash join and
// dynamic_filter_publisher) are known, enumerated debt that migrates here in C1a-2 — until then
// the rule is: no NEW read may be added outside this module.
//===-----------------------------------------------------------------------------------------===//

#include "planner/preserved_counts.hpp"

#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace duckdb {
class LogicalComparisonJoin;
class LogicalGet;
}  // namespace duckdb

namespace sirius::planner {

/// @brief The classification of the `JoinFilterPushdownInfo` on one comparison join for
/// dynamic-filter routing.
///
/// @c admitted candidates carry a dynamic-filter route.
/// @c malformed is reserved for locally-provable structural corruption (see the pin note above — a
///              fence for future pins, unreachable today).
enum class duckdb_candidate_kind : std::uint8_t {
  absent,
  statistics_only,
  admitted,
  malformed,
};

/// @brief The dynamic-filter probe target (one downstream scan) named by a producing join, reduced
/// to Sirius-owned values plus the shared channel identity.
struct duckdb_probe_target_candidate {
  struct probe_column {
    std::size_t column_index = 0;  ///< `JoinFilterPushdownColumn::probe_column_index.column_index`.
    duckdb::LogicalType storage_type{};  ///< `JoinFilterPushdownColumn::storage_type`
  };
  duckdb::shared_ptr<duckdb::DynamicTableFilterSet> channel_identity;
  std::vector<probe_column> columns;
};

/// @brief The Sirius-owned snapshot of one comparison join's dynamic-filter metadata.
struct duckdb_join_filter_candidate {
  duckdb_candidate_kind kind = duckdb_candidate_kind::absent;
  bool build_subtree_has_filter_hint =
    false;  ///< `JoinFilterPushdownInfo::build_side_has_filter`. Gate for dynamic filter routing.
  std::vector<std::size_t>
    condition_indexes;  ///< `JoinFilterPushdownInfo::join_condition` ordinals
  std::vector<duckdb::ExpressionType>
    condition_comparisons;  ///< The comparison of each recorded ordinal, aligned with @ref
                            ///< condition_indexes.
  std::vector<duckdb_probe_target_candidate>
    targets;  ///< One entry per `probe_info` target with a non-null channel. Empty for non-@c
              ///< admitted kinds.
};

/// @brief The single owner of every read of DuckDB dynamic-filter metadata. The adapter is the only
/// module allowed to depend on the DuckDB layout; everything downstream consumes Sirius-owned
/// values and never re-reads DuckDB data structures.
namespace duckdb_join_filter_candidate_adapter {

namespace detail {

/// @brief Clone helper for @ref preserve_dynamic_filter_metadata, exposed only for unit tests.
[[nodiscard]] duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> clone_filter_pushdown_info(
  duckdb::JoinFilterPushdownInfo const& src);

}  // namespace detail

/// @brief Re-attach dynamic-filter metadata from @p original onto @p copy while a logical plan is
/// deep-copied for Sirius's transparent execution path.
///
/// `LogicalOperator::Copy` round-trips through serialize/deserialize, and neither
/// `LogicalComparisonJoin::filter_pushdown` nor `LogicalGet::dynamic_filters` is in DuckDB's
/// serialization schema, so a plain `Copy` strips them. The copy is expected to have the same tree
/// shape; debug builds assert this, and release builds leave mismatched copies unchanged.
///
/// @param original  Read-only source plan. Its `filter_pushdown` / `dynamic_filters` stay put so
///                  DuckDB's CPU fallback still sees them; only shared_ptrs are copied out.
/// @param copy      The freshly-`Copy`-produced plan that receives the metadata.
/// @param counts    In-out accumulator threaded through the recursion; the caller zero-initializes
///                  it and reads the join/get totals after the walk returns.
void preserve_dynamic_filter_metadata(duckdb::LogicalOperator const& original,
                                      duckdb::LogicalOperator& copy,
                                      preserved_counts& counts);

/// @brief Classify and snapshot one comparison join's dynamic-filter metadata into Sirius values.
///
/// Fails closed (@c malformed) on structural corruption it can prove locally
///  - an out-of-range or duplicate condition index,
///  - a target whose column count disagrees with the recorded ordinal count,
///  - recorded targets with an empty ordinal list, or
///  - a non-empty `probe_info` whose every channel identity is null (not @c statistics_only:
///    that kind is reserved for DuckDB's deliberate zero-target state, whose telemetry count
///    sizes the Track E opportunity).
/// A single null channel among live siblings drops only that target.
/// Range comparisons are @b not malformed; their ordinals are kept and narrowed later.
[[nodiscard]] duckdb_join_filter_candidate extract(duckdb::LogicalComparisonJoin const& op);

/// @brief The channel identity for consumer end of the join-scan channel pairing.
[[nodiscard]] duckdb::shared_ptr<duckdb::DynamicTableFilterSet> scan_channel_identity(
  duckdb::LogicalGet const& get);

}  // namespace duckdb_join_filter_candidate_adapter

}  // namespace sirius::planner
