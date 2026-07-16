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

/**
 * @file
 * @brief The encapsulation boundary between DuckDB's internal dynamic-filter representation and
 * Sirius.
 *
 * The normal data flow is:
 *
 *   DuckDB logical join                         Sirius-owned planning data
 *   +---------------------------+               +-----------------------------+
 *   | filter_pushdown           |   extract()   | kind                        |
 *   |   join_condition[]        | ------------> | condition indexes/types     |
 *   |   probe_info[]            |               | probe columns               |
 *   |   DynamicTableFilterSet * |               | opaque shared channel key   |
 *   +---------------------------+               +-----------------------------+
 *                                                          |
 *                                                          v
 *                                              downstream Sirius planning
 *
 * DuckDB's `JoinFilterPushdownInfo` (`join_filter_pushdown.hpp`) is the DuckDB planner's internal
 * representation of dynamic-filter metadata. Two of its member vectors are:
 *
 * - `join_condition[]`: indexes of the join conditions for which DuckDB will build filters;
 * - `probe_info[]`: probe targets (one per `LogicalGet`) and the columns in each target that
 *   correspond to the `join_condition[]` entries.
 *
 * A DuckDB filter ordinal is the index position into the aligned `join_condition[]` and
 * `probe_info[t].columns[]` vectors for each `JoinFilterPushdownFilter` target `t`.
 *
 * For a join:
 *
 * ```text
 *   ON f.a = d.a AND f.ts < d.ts AND f.b = d.b     -- conditions[0], [1], [2]
 * where DuckDB recorded join_condition = [2, 0]:
 *   DuckDB filter ordinal j     0                     1
 *   join_condition[j]           2  (f.b = d.b)        0  (f.a = d.a)
 *   target t's columns          columns[0] -> b       columns[1] -> a
 *                               (each an index into this scan's column_ids vector)
 * ```
 */

// duckdb
#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/shared_ptr.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/planner/logical_operator.hpp>

// standard library
#include <cstddef>
#include <vector>

namespace duckdb {
class LogicalComparisonJoin;
class LogicalGet;
}  // namespace duckdb

namespace sirius::planner {

namespace duckdb_join_filter_candidate_adapter::detail {
class candidate_builder;
}  // namespace duckdb_join_filter_candidate_adapter::detail

using duckdb_dynamic_filter_channel = duckdb::shared_ptr<duckdb::DynamicTableFilterSet const>;

/**
 * @brief Structural classification of one comparison join's DuckDB dynamic-filter metadata.
 */
enum class duckdb_candidate_kind {
  absent,           ///< No JoinFilterPushdownInfo was attached to the join.
  statistics_only,  ///< DuckDB deliberately recorded no probe target; the hint is observational.
  admitted,         ///< The recorded indexes, target arity, and channel identities are usable.
  malformed,        ///< The adapter can prove that the recorded structure is inconsistent.
};

/**
 * @brief One probe target named by the producing join, copied into Sirius-owned values.
 *
 * A target is one base-table scan (`LogicalGet`) on the join's probe subtree that DuckDB's
 * pushdown walk selected to receive the build-side filters — one entry per `probe_info` element.
 * A join can name several scans (e.g., a probe side built from a UNION reaches one per branch),
 * and every target carries the full key arity: `columns[j]` names where filter ordinal `j`'s
 * key lives in THIS scan's output.
 *
 * `channel_identity` keeps the exact shared DuckDB object alive and exposes it as a read-only,
 * opaque map key during planning. DuckDB owns mutation of the underlying runtime filter set.
 * Targets have no disengaged state, so they are copyable but deliberately not movable.
 */
class duckdb_probe_target_candidate final {
 public:
  struct probe_column {
    /// Index INTO the target scan's `column_ids` vector (its projection position, i.e.,
    /// probe_info[t].columns[j]), NOT the base-table column index stored at that position
    /// (`column_ids[i].GetPrimaryIndex()`).
    std::size_t column_index = 0;
    duckdb::LogicalType storage_type{};
  };

  ~duckdb_probe_target_candidate()                                               = default;
  duckdb_probe_target_candidate(duckdb_probe_target_candidate const&)            = default;
  duckdb_probe_target_candidate& operator=(duckdb_probe_target_candidate const&) = default;
  duckdb_probe_target_candidate(duckdb_probe_target_candidate&&)                 = delete;
  duckdb_probe_target_candidate& operator=(duckdb_probe_target_candidate&&)      = delete;

  [[nodiscard]] duckdb_dynamic_filter_channel const& channel_identity() const& noexcept;
  [[nodiscard]] duckdb_dynamic_filter_channel const& channel_identity() const&& = delete;
  [[nodiscard]] std::vector<probe_column> const& columns() const& noexcept;
  [[nodiscard]] std::vector<probe_column> const& columns() const&& = delete;

 private:
  friend class duckdb_join_filter_candidate_adapter::detail::candidate_builder;

  duckdb_probe_target_candidate(duckdb_dynamic_filter_channel channel_identity,
                                std::vector<probe_column> columns);

  duckdb_dynamic_filter_channel _channel_identity;
  std::vector<probe_column> _columns;
};

/**
 * @brief Immutable Sirius-owned snapshot of one comparison join's dynamic-filter metadata.
 *
 * The filter-key set is decided once per producing join and is SHARED by every target; a target
 * records only where each shared key lands in its own scan. The vectors form a keys-by-targets
 * grid indexed by DuckDB filter ordinal `j` (the position in the aligned vectors — a different
 * ordinal space from the join-condition index stored at that position):
 *
 *   shared across targets    condition_indexes[j]       index into the join's condition vector
 *                            condition_comparisons[j]   comparison at that condition index
 *   per target t             targets[t].columns[j]      where key j lands in scan t's output
 *
 * (Runtime uses the same correlation: every filter generated for ordinal `j` is fanned out to
 * each target.)
 *
 * Later planning wraps these raw adapter values in strong ordinal types and may compact admitted
 * equality keys into a third space, the Sirius key ordinal.
 */
class duckdb_join_filter_candidate final {
 public:
  ~duckdb_join_filter_candidate()                                   = default;
  duckdb_join_filter_candidate(duckdb_join_filter_candidate const&) = default;
  duckdb_join_filter_candidate& operator=(duckdb_join_filter_candidate const& other);
  duckdb_join_filter_candidate(duckdb_join_filter_candidate&& other) noexcept;
  duckdb_join_filter_candidate& operator=(duckdb_join_filter_candidate&& other) noexcept;

  [[nodiscard]] duckdb_candidate_kind kind() const noexcept;
  [[nodiscard]] bool build_subtree_has_filter_hint() const noexcept;
  [[nodiscard]] std::vector<std::size_t> const& condition_indexes() const& noexcept;
  [[nodiscard]] std::vector<std::size_t> const& condition_indexes() const&& = delete;
  [[nodiscard]] std::vector<duckdb::ExpressionType> const& condition_comparisons() const& noexcept;
  [[nodiscard]] std::vector<duckdb::ExpressionType> const& condition_comparisons() const&& = delete;
  [[nodiscard]] std::vector<duckdb_probe_target_candidate> const& targets() const& noexcept;
  [[nodiscard]] std::vector<duckdb_probe_target_candidate> const& targets() const&& = delete;

 private:
  friend class duckdb_join_filter_candidate_adapter::detail::candidate_builder;

  [[nodiscard]] static duckdb_join_filter_candidate absent();
  [[nodiscard]] static duckdb_join_filter_candidate statistics_only(
    bool build_subtree_has_filter_hint);
  [[nodiscard]] static duckdb_join_filter_candidate malformed();
  [[nodiscard]] static duckdb_join_filter_candidate admitted(
    bool build_subtree_has_filter_hint,
    std::vector<std::size_t> condition_indexes,
    std::vector<duckdb::ExpressionType> condition_comparisons,
    std::vector<duckdb_probe_target_candidate> targets);

  duckdb_join_filter_candidate(duckdb_candidate_kind kind,
                               bool build_subtree_has_filter_hint,
                               std::vector<std::size_t> condition_indexes,
                               std::vector<duckdb::ExpressionType> condition_comparisons,
                               std::vector<duckdb_probe_target_candidate> targets);

  void reset_to_absent() noexcept;

  duckdb_candidate_kind _kind;
  bool _build_subtree_has_filter_hint;
  std::vector<std::size_t> _condition_indexes;
  std::vector<duckdb::ExpressionType> _condition_comparisons;
  std::vector<duckdb_probe_target_candidate> _targets;
};

/**
 * @brief The semantic owner of reads from DuckDB's dynamic-filter metadata layout.
 *
 * Consumers receive Sirius-owned structural values and do not retain or dereference DuckDB's
 * `JoinFilterPushdownInfo`. The opaque channel handle is the deliberate exception: it keeps the
 * shared route identity alive without exposing mutation. The remaining direct planner reads of
 * `filter_pushdown` / `dynamic_filters` belong to the intentionally preserved legacy runtime path,
 * which remains production authority through C1a-2a and is removed at the C1a-2b freeze cutover.
 */
namespace duckdb_join_filter_candidate_adapter {

namespace detail {

/**
 * @brief Clone the subset of metadata consumed by Sirius during logical-plan preservation.
 *
 * Shares each `DynamicTableFilterSet` to preserve route identity and intentionally omits
 * `min_max_aggregates`. The result is only for Sirius planning and must not be handed to DuckDB's
 * physical join execution; the untouched original retains the complete CPU-fallback metadata.
 * The detail function is declared here so preservation behavior can also be tested directly.
 */
[[nodiscard]] duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> clone_sirius_filter_pushdown_info(
  duckdb::JoinFilterPushdownInfo const& src);

}  // namespace detail

/**
 * @brief Re-attach dynamic-filter metadata from @p original onto @p copy while a logical plan is
 * deep-copied for Sirius's transparent execution path.
 *
 * `LogicalOperator::Copy` round-trips through serialize/deserialize, and neither
 * `LogicalComparisonJoin::filter_pushdown` nor `LogicalGet::dynamic_filters` is in DuckDB's
 * serialization schema, so a plain `Copy` strips them. The copy is expected to have the same tree
 * shape; a whole-tree preflight leaves a mismatched copy unchanged in every build.
 *
 * @param original  Read-only source plan. Its `filter_pushdown` / `dynamic_filters` stay put so
 *                  DuckDB's CPU fallback still sees them; only shared_ptrs are copied out.
 * @param copy      The freshly-`Copy`-produced plan that receives the metadata.
 */
void preserve_dynamic_filter_metadata(duckdb::LogicalOperator const& original,
                                      duckdb::LogicalOperator& copy);

/**
 * @brief Classify and snapshot one comparison join's dynamic-filter metadata into Sirius values.
 *
 * Fails closed (@c malformed) on producer-level structural corruption it can prove locally:
 *
 * - an out-of-range or duplicate condition index;
 * - recorded targets with an empty ordinal list; or
 * - a non-empty `probe_info` from which no arity-correct target with a non-null channel survives.
 *
 * A target whose column count disagrees with the recorded ordinal count is dropped as one
 * corrupt route. Other structurally valid targets remain eligible; if none survive, the
 * candidate is malformed. Metadata whose `probe_info` is empty is @c statistics_only regardless
 * of the build hint. The hint is retained as an observation and is not canonical-sidecar route
 * admission; the intentionally preserved legacy runtime path still applies its existing hint gate.
 *
 * A malformed result carries only `kind == malformed`; callers must not rely on partially copied
 * fields from the rejected metadata.
 */
[[nodiscard]] duckdb_join_filter_candidate extract(duckdb::LogicalComparisonJoin const& op);

/**
 * @brief Return the opaque channel identity at the scan end of the join-scan pairing.
 *
 * The PRODUCER obtains the same shared object through @ref extract. Pointer equality pairs the two
 * producer/consumer objects during planning. The returned handle keeps the channel alive but
 * exposes only a const pointee; DuckDB owns runtime mutation. Exposed for the CONSUMER.
 */
[[nodiscard]] duckdb_dynamic_filter_channel scan_channel_identity(duckdb::LogicalGet const& get);

}  // namespace duckdb_join_filter_candidate_adapter

}  // namespace sirius::planner
