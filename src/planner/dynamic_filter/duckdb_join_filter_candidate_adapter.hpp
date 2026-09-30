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
 * @brief Test-only adapter for comparing Sirius discovery with DuckDB pushdown metadata
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

enum class duckdb_candidate_kind { absent, statistics_only, admitted, malformed };

/**
 * @brief Snapshot of one DuckDB probe target
 *
 * `columns[j]` locates filter ordinal `j` in the scan's `column_ids`; `channel_identity` is
 * used only to pair producer and scan snapshots.
 */
class duckdb_probe_target_candidate final {
 public:
  struct probe_column {
    // Position in column_ids, not a base-table column ID.
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
 * @brief Immutable snapshot keyed by DuckDB filter ordinal
 *
 * `condition_indexes[j]` and `condition_comparisons[j]` are shared; each
 * `targets[t].columns[j]` locates that key in target `t`.
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

namespace duckdb_join_filter_candidate_adapter {

/**
 * @brief Snapshots metadata, returning malformed for inconsistent indices, arity, or identities
 *
 * Malformed results expose only `kind`.
 */
[[nodiscard]] duckdb_join_filter_candidate extract(duckdb::LogicalComparisonJoin const& op);

/**
 * @brief Returns the scan-side identity used to pair with @ref extract
 */
[[nodiscard]] duckdb_dynamic_filter_channel scan_channel_identity(duckdb::LogicalGet const& get);

}  // namespace duckdb_join_filter_candidate_adapter

}  // namespace sirius::planner
