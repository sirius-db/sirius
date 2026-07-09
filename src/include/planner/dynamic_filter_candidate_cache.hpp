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

#include "planner/duckdb_join_filter_candidate_adapter.hpp"

#include <memory>
#include <unordered_map>

namespace duckdb {
class ClientContext;
class LogicalComparisonJoin;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/// @brief sirius_physical_plan_generator-local immutable candidate cache: each comparison join's
/// dynamic-filter metadata is extracted through the duckdb_join_filter_candidate_adapter once,
/// before recursive planning drains `op.conditions`/`op.filter_pushdown`. Every later consumer —
/// `plan_comparison_join` now, C3 route discovery later — calls @ref find and shares the same const
/// entry; nobody re-extracts.
///
/// Lifecycle inside `sirius_physical_plan_generator::create_plan(unique_ptr)`, in order:
///   1. @ref capture_pre_resolver on the UNRESOLVED tree (node identity + the pre/post-resolver
///      evidence; C1b attaches its domain snapshot here);
///   2. the generator's `ResolveOperatorTypes` + sole `ColumnBindingResolver` pass;
///   3. @ref extract_post_resolver (one adapter extraction per join);
///   4. recursive physical planning, consuming @ref find.
///
/// Single-use, single-threaded, generator-owned; entries are keyed by logical node address,
/// which is stable through `create_plan` (nodes outlive planning; only their contents drain).
/// @ref find is meaningful only while the source logical plan is alive — i.e. during
/// `create_plan(unique_ptr)`, which owns and destroys the tree on return; afterwards the keys
/// dangle, and the single-use flags prevent the generator from ever planning a second tree.
class dynamic_filter_candidate_cache {
 public:
  /// @brief Record pre-resolver evidence: every comparison/delim join carrying pushdown metadata,
  /// and whether its join conditions are still in pre-resolver form (`BOUND_COLUMN_REF`).
  ///
  /// A plan whose pushdown-bearing joins already contain positional references (`BOUND_REF`) was
  /// resolved before the generator — the snapshot this pass exists to take is impossible there.
  /// Debug builds reject such a plan; release builds record @ref saw_resolved_plan and proceed
  /// (extraction still works positionally; C1b's domain evidence is simply unavailable, i.e.
  /// nullopt).
  ///
  /// @param context  Reserved for C1b's preservation-time domain snapshot; unused in C1a-2.
  void capture_pre_resolver(duckdb::LogicalOperator const& root, duckdb::ClientContext& context);

  /// @brief Run the adapter's extraction once per captured join and freeze the entries.
  void extract_post_resolver(duckdb::LogicalOperator const& root);

  /// @brief The immutable candidate for @p join, or null when @p join was not in the extracted
  /// tree (or extraction has not run). Never triggers re-extraction.
  [[nodiscard]] std::shared_ptr<duckdb_join_filter_candidate const> find(
    duckdb::LogicalComparisonJoin const& join) const;

  [[nodiscard]] bool capture_completed() const noexcept { return _captured; }
  [[nodiscard]] bool extraction_completed() const noexcept { return _extracted; }

  /// @brief True when capture saw a pushdown-bearing join already in post-resolver form (see
  /// @ref capture_pre_resolver). Exposed so release-build tests can assert the fence signal.
  [[nodiscard]] bool saw_resolved_plan() const noexcept { return _saw_resolved_plan; }

 private:
  /// Keyed by logical node address; covers LOGICAL_COMPARISON_JOIN and LOGICAL_DELIM_JOIN (both
  /// plan through `plan_comparison_join`).
  std::unordered_map<duckdb::LogicalOperator const*,
                     std::shared_ptr<duckdb_join_filter_candidate const>>
    _entries;
  bool _captured          = false;
  bool _extracted         = false;
  bool _saw_resolved_plan = false;
};

}  // namespace sirius::planner
