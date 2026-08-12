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
 * @brief Test-only parity oracle over DuckDB's join-filter pushdown metadata
 *
 * Parity tests snapshot DuckDB's `JoinFilterPushdownInfo` and compare it with Sirius-owned target
 * discovery. Production code does not consume these values, and the adapter translation unit is
 * linked only into the test target.
 */

#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/shared_ptr.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/planner/logical_operator.hpp>

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
  statistics_only,  ///< DuckDB deliberately recorded no probe target; the build hint is false.
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
 * Targets have no disengaged state, so they are copyable but not movable.
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
 * The filter ordinal exists only to align these vectors; the stored condition index and
 * per-target scan-column position remain distinct coordinate spaces.
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
  [[nodiscard]] static duckdb_join_filter_candidate statistics_only();
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
 * @brief Reads DuckDB's dynamic-filter metadata layout into Sirius-owned values.
 *
 * Consumers do not retain or dereference DuckDB's `JoinFilterPushdownInfo`; only the opaque
 * channel handle keeps a shared DuckDB object alive, without exposing mutation.
 */
namespace duckdb_join_filter_candidate_adapter {

/**
 * @brief Classify and snapshot one comparison join's dynamic-filter metadata into Sirius values.
 *
 * Fails closed (@c malformed) on structural corruption it can prove locally:
 *  - an empty recorded condition-index list,
 *  - an out-of-range or duplicate condition index,
 *  - a target whose column count disagrees with the recorded ordinal count,
 *  - targetless metadata carrying a build-side filter hint, or
 *  - a non-empty `probe_info` whose every channel identity is null.
 *
 * A malformed result carries only `kind == malformed`; callers must not rely on partially copied
 * fields from the rejected metadata.
 */
[[nodiscard]] duckdb_join_filter_candidate extract(duckdb::LogicalComparisonJoin const& op);

/**
 * @brief Return the opaque channel identity at the scan end of the join-scan pairing.
 *
 * Pointer equality with the producer-side handle from @ref extract pairs the two ends during
 * planning. The returned handle keeps the channel alive but exposes only a const pointee; DuckDB
 * owns runtime mutation.
 */
[[nodiscard]] duckdb_dynamic_filter_channel scan_channel_identity(duckdb::LogicalGet const& get);

}  // namespace duckdb_join_filter_candidate_adapter

}  // namespace sirius::planner
