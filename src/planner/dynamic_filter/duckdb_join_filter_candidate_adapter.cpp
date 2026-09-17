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

#include "planner/dynamic_filter/duckdb_join_filter_candidate_adapter.hpp"

#include <duckdb/common/helper.hpp>
#include <duckdb/common/typedefs.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::planner {

duckdb_probe_target_candidate::duckdb_probe_target_candidate(
  duckdb_dynamic_filter_channel channel_identity, std::vector<probe_column> columns)
  : _channel_identity(std::move(channel_identity)), _columns(std::move(columns))
{
  if (!_channel_identity) {
    throw std::invalid_argument(
      "[duckdb_probe_target_candidate] A DuckDB dynamic-filter target requires a channel identity");
  }
  if (_columns.empty()) {
    throw std::invalid_argument(
      "[duckdb_probe_target_candidate] A DuckDB dynamic-filter target requires at least one "
      "column");
  }
}

duckdb_dynamic_filter_channel const& duckdb_probe_target_candidate::channel_identity()
  const& noexcept
{
  return _channel_identity;
}

std::vector<duckdb_probe_target_candidate::probe_column> const&
duckdb_probe_target_candidate::columns() const& noexcept
{
  return _columns;
}

duckdb_join_filter_candidate::duckdb_join_filter_candidate(
  duckdb_candidate_kind kind,
  bool build_subtree_has_filter_hint,
  std::vector<std::size_t> condition_indexes,
  std::vector<duckdb::ExpressionType> condition_comparisons,
  std::vector<duckdb_probe_target_candidate> targets)
  : _kind(kind),
    _build_subtree_has_filter_hint(build_subtree_has_filter_hint),
    _condition_indexes(std::move(condition_indexes)),
    _condition_comparisons(std::move(condition_comparisons)),
    _targets(std::move(targets))
{
}

duckdb_join_filter_candidate::duckdb_join_filter_candidate(
  duckdb_join_filter_candidate&& other) noexcept
  : _kind(other._kind),
    _build_subtree_has_filter_hint(other._build_subtree_has_filter_hint),
    _condition_indexes(std::move(other._condition_indexes)),
    _condition_comparisons(std::move(other._condition_comparisons)),
    _targets(std::move(other._targets))
{
  other.reset_to_absent();
}

duckdb_join_filter_candidate& duckdb_join_filter_candidate::operator=(
  duckdb_join_filter_candidate const& other)
{
  if (this == &other) { return *this; }
  auto copy    = other;
  return *this = std::move(copy);
}

duckdb_join_filter_candidate& duckdb_join_filter_candidate::operator=(
  duckdb_join_filter_candidate&& other) noexcept
{
  if (this == &other) { return *this; }
  _kind                          = other._kind;
  _build_subtree_has_filter_hint = other._build_subtree_has_filter_hint;
  _condition_indexes             = std::move(other._condition_indexes);
  _condition_comparisons         = std::move(other._condition_comparisons);
  _targets                       = std::move(other._targets);
  other.reset_to_absent();
  return *this;
}

duckdb_join_filter_candidate duckdb_join_filter_candidate::absent()
{
  return duckdb_join_filter_candidate{duckdb_candidate_kind::absent, false, {}, {}, {}};
}

duckdb_join_filter_candidate duckdb_join_filter_candidate::statistics_only()
{
  return duckdb_join_filter_candidate{duckdb_candidate_kind::statistics_only, false, {}, {}, {}};
}

duckdb_join_filter_candidate duckdb_join_filter_candidate::malformed()
{
  return duckdb_join_filter_candidate{duckdb_candidate_kind::malformed, false, {}, {}, {}};
}

duckdb_join_filter_candidate duckdb_join_filter_candidate::admitted(
  bool build_subtree_has_filter_hint,
  std::vector<std::size_t> condition_indexes,
  std::vector<duckdb::ExpressionType> condition_comparisons,
  std::vector<duckdb_probe_target_candidate> targets)
{
  if (condition_indexes.empty()) {
    throw std::invalid_argument(
      "[duckdb_join_filter_candidate::admitted] An admitted DuckDB dynamic-filter candidate "
      "requires a key");
  }
  if (condition_indexes.size() != condition_comparisons.size()) {
    throw std::invalid_argument(
      "[duckdb_join_filter_candidate::admitted] DuckDB dynamic-filter condition vectors must have "
      "equal arity");
  }
  for (auto it = condition_indexes.begin(); it != condition_indexes.end(); ++it) {
    if (std::find(condition_indexes.begin(), it, *it) != it) {
      throw std::invalid_argument(
        "[duckdb_join_filter_candidate::admitted] DuckDB dynamic-filter condition indexes must be "
        "unique");
    }
  }
  if (targets.empty()) {
    throw std::invalid_argument(
      "[duckdb_join_filter_candidate::admitted] An admitted DuckDB dynamic-filter candidate "
      "requires a target");
  }
  for (auto const& target : targets) {
    if (target.columns().size() != condition_indexes.size()) {
      throw std::invalid_argument(
        "[duckdb_join_filter_candidate::admitted] DuckDB dynamic-filter target and key arity must "
        "match");
    }
  }
  return duckdb_join_filter_candidate{duckdb_candidate_kind::admitted,
                                      build_subtree_has_filter_hint,
                                      std::move(condition_indexes),
                                      std::move(condition_comparisons),
                                      std::move(targets)};
}

duckdb_candidate_kind duckdb_join_filter_candidate::kind() const noexcept { return _kind; }

bool duckdb_join_filter_candidate::build_subtree_has_filter_hint() const noexcept
{
  return _build_subtree_has_filter_hint;
}

std::vector<std::size_t> const& duckdb_join_filter_candidate::condition_indexes() const& noexcept
{
  return _condition_indexes;
}

std::vector<duckdb::ExpressionType> const& duckdb_join_filter_candidate::condition_comparisons()
  const& noexcept
{
  return _condition_comparisons;
}

std::vector<duckdb_probe_target_candidate> const& duckdb_join_filter_candidate::targets()
  const& noexcept
{
  return _targets;
}

void duckdb_join_filter_candidate::reset_to_absent() noexcept
{
  _kind                          = duckdb_candidate_kind::absent;
  _build_subtree_has_filter_hint = false;
  _condition_indexes.clear();
  _condition_comparisons.clear();
  _targets.clear();
}

namespace duckdb_join_filter_candidate_adapter {

namespace detail {

class candidate_builder final {
 public:
  static void append_probe_target(std::vector<duckdb_probe_target_candidate>& targets,
                                  duckdb_dynamic_filter_channel channel_identity,
                                  std::vector<duckdb_probe_target_candidate::probe_column> columns)
  {
    duckdb_probe_target_candidate target{std::move(channel_identity), std::move(columns)};
    targets.push_back(target);
  }

  [[nodiscard]] static duckdb_join_filter_candidate absent()
  {
    return duckdb_join_filter_candidate::absent();
  }

  [[nodiscard]] static duckdb_join_filter_candidate statistics_only()
  {
    return duckdb_join_filter_candidate::statistics_only();
  }

  [[nodiscard]] static duckdb_join_filter_candidate malformed()
  {
    return duckdb_join_filter_candidate::malformed();
  }

  [[nodiscard]] static duckdb_join_filter_candidate admitted(
    bool build_subtree_has_filter_hint,
    std::vector<std::size_t> condition_indexes,
    std::vector<duckdb::ExpressionType> condition_comparisons,
    std::vector<duckdb_probe_target_candidate> targets)
  {
    return duckdb_join_filter_candidate::admitted(build_subtree_has_filter_hint,
                                                  std::move(condition_indexes),
                                                  std::move(condition_comparisons),
                                                  std::move(targets));
  }
};

}  // namespace detail

static_assert(std::numeric_limits<std::size_t>::max() >= std::numeric_limits<duckdb::idx_t>::max(),
              "DuckDB idx_t ordinals must fit in std::size_t without narrowing");

duckdb_join_filter_candidate extract(duckdb::LogicalComparisonJoin const& op)
{
  if (!op.filter_pushdown) { return detail::candidate_builder::absent(); }
  auto const& info = *op.filter_pushdown;

  auto const malformed = []() { return detail::candidate_builder::malformed(); };

  if (info.join_condition.empty()) { return malformed(); }

  std::vector<std::size_t> condition_indexes;
  std::vector<duckdb::ExpressionType> condition_comparisons;
  condition_indexes.reserve(info.join_condition.size());
  condition_comparisons.reserve(info.join_condition.size());
  for (auto const cond_idx : info.join_condition) {
    if (cond_idx >= op.conditions.size()) { return malformed(); }
    auto const condition_index = static_cast<std::size_t>(cond_idx);
    if (std::find(condition_indexes.begin(), condition_indexes.end(), condition_index) !=
        condition_indexes.end()) {
      return malformed();
    }
    condition_indexes.push_back(condition_index);
    condition_comparisons.push_back(op.conditions[condition_index].comparison);
  }

  if (info.probe_info.empty()) {
    if (info.build_side_has_filter) { return malformed(); }
    return detail::candidate_builder::statistics_only();
  }

  std::vector<duckdb_probe_target_candidate> targets;
  targets.reserve(info.probe_info.size());
  for (auto const& pi : info.probe_info) {
    if (pi.columns.size() != info.join_condition.size()) { return malformed(); }
    if (!pi.dynamic_filters) { continue; }

    std::vector<duckdb_probe_target_candidate::probe_column> columns;
    columns.reserve(pi.columns.size());
    for (auto const& col : pi.columns) {
      columns.push_back(duckdb_probe_target_candidate::probe_column{
        static_cast<std::size_t>(col.probe_column_index.column_index), col.storage_type});
    }
    detail::candidate_builder::append_probe_target(targets, pi.dynamic_filters, std::move(columns));
  }

  if (targets.empty()) { return malformed(); }

  return detail::candidate_builder::admitted(info.build_side_has_filter,
                                             std::move(condition_indexes),
                                             std::move(condition_comparisons),
                                             std::move(targets));
}

duckdb_dynamic_filter_channel scan_channel_identity(duckdb::LogicalGet const& get)
{
  return get.dynamic_filters;
}

}  // namespace duckdb_join_filter_candidate_adapter

}  // namespace sirius::planner
