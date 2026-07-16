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

// sirius
#include <planner/duckdb_join_filter_candidate_adapter.hpp>

// standard library
#include <unordered_map>
#include <unordered_set>

namespace duckdb {
class LogicalComparisonJoin;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @brief `sirius_physical_plan_generator`-local, immutable snapshots of DuckDB dynamic-filter
 * candidates (`duckdb_join_filter_candidate`s: see duckdb_join_filter_candidate_adapter.hpp).
 *
 * Why are there separate "pre-resolver" and "post-resolver" methods?
 *
 * DuckDB's ColumnBindingResolver changes the meaning of join expressions in place:
 *
 *   before resolver                          after resolver
 *   ----------------                         --------------
 *   BoundColumnRef                           BoundRef
 *   logical table + column identity   --->   position in a physical input
 *
 * Some evidence planned for C1b can only be computed from the logical table/column identity on
 * the left. Recursive physical planning, however, expects the resolved expressions on the right
 * and may move data out of the logical operators. The cache brackets those mutations:
 *
 *   sirius_physical_plan_generator::create_plan() owns one logical tree
 *          |
 *          v
 *   capture_pre_resolver(root)
 *     CURRENT: remember the exact comparison/delim node set
 *     C1b:     also capture domain evidence that needs BoundColumnRef on entry paths
 *              that expose it
 *          |
 *          v
 *   ResolveOperatorTypes + ColumnBindingResolver
 *          |
 *          v
 *   extract_post_resolver(root)
 *     CURRENT: call the adapter once per comparison/delim join and store a const snapshot
 *          |
 *          v
 *   recursive physical planning --find(join)--> same immutable snapshot
 *
 * The pre pass records only node identity in C1a-2a; all in-tree entrypoints now provide the
 * unresolved tree so C1b can add logical source evidence at this seam without another entrypoint
 * migration.
 *
 * The cache is single-use, single-threaded, and owned by one `sirius_physical_plan_generator`.
 * Entries are keyed by logical-node address because nodes remain alive at stable addresses
 * throughout `create_plan`; their contents may be moved, but the nodes themselves are not.
 * @ref candidate_for is meaningful only during that call. Once the owned logical tree is
 * destroyed, the map keys dangle and the cache must also be discarded.
 *
 * Extraction is transactional: it first builds a complete temporary map through the
 * `duckdb_join_filter_candidate_adapter`, verifies that the post-resolver node set exactly equals
 * the captured set, and only then publishes the immutable snapshots. Failure leaves the cache
 * captured but unextracted, so callers may retry extraction with the correct tree.
 */
class dynamic_filter_candidate_cache {
 public:
  dynamic_filter_candidate_cache()                                                 = default;
  ~dynamic_filter_candidate_cache()                                                = default;
  dynamic_filter_candidate_cache(dynamic_filter_candidate_cache const&)            = delete;
  dynamic_filter_candidate_cache& operator=(dynamic_filter_candidate_cache const&) = delete;
  dynamic_filter_candidate_cache(dynamic_filter_candidate_cache&&)                 = delete;
  dynamic_filter_candidate_cache& operator=(dynamic_filter_candidate_cache&&)      = delete;

  /**
   * @brief Record every comparison/delim join's node identity before binding resolution.
   *
   * Joins without pushdown metadata are still captured so the post pass caches an explicit @c
   * absent candidate for them.
   */
  void capture_pre_resolver(duckdb::LogicalOperator const& root);

  /** @brief Run adapter extraction once per comparison/delim join and freeze the entries. */
  void extract_post_resolver(duckdb::LogicalOperator const& root);

  /**
   * @brief Return the immutable candidate for one captured join after exact extraction.
   *
   * Calling before successful extraction, or with a node outside the captured tree, is an internal
   * correlation error. DuckDB metadata absence is represented by a returned candidate whose `kind`
   * is @c duckdb_candidate_kind::absent, never by a nullable result.
   */
  [[nodiscard]] duckdb_join_filter_candidate const& candidate_for(
    duckdb::LogicalComparisonJoin const& join) const;

 private:
  using logical_join_set = std::unordered_set<duckdb::LogicalOperator const*>;
  using candidate_map =
    std::unordered_map<duckdb::LogicalOperator const*, duckdb_join_filter_candidate>;

  /// Covers LOGICAL_COMPARISON_JOIN and LOGICAL_DELIM_JOIN because both plan through
  /// `plan_comparison_join`, whether or not they carry pushdown metadata.
  logical_join_set _captured_joins;
  candidate_map _entries;
  bool _captured  = false;
  bool _extracted = false;
};

}  // namespace sirius::planner
