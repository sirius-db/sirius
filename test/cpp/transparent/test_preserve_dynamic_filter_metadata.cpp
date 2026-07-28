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

/**
 * @file test_preserve_dynamic_filter_metadata.cpp
 * @brief Tests for the version-pinned adapter's post-Copy fixup helpers
 *        (sirius::planner::duckdb_join_filter_candidate_adapter::preserve_dynamic_filter_metadata
 *         and its detail::clone_sirius_filter_pushdown_info helper), which the transparent
 *         execution path's copy_logical_plan drives.
 *
 * LogicalOperator::Copy round-trips through serialize/deserialize. LogicalComparisonJoin's
 * filter_pushdown and LogicalGet's dynamic_filters are not in the serialization schemas, so the
 * copy has them null by default. The helpers under test walk both trees in parallel and re-attach
 * those fields, sharing the DynamicTableFilterSet shared_ptrs to preserve the route-key pointer
 * identity that downstream wiring (Phase 1.1 of dynamic filter pushdown) relies on.
 */

#include "planner/dynamic_filter/duckdb_join_filter_candidate_adapter.hpp"
#include "transparent/sirius_optimizer_extension.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

using sirius::planner::duckdb_join_filter_candidate_adapter::preserve_dynamic_filter_metadata;
using sirius::planner::duckdb_join_filter_candidate_adapter::detail::
  clone_sirius_filter_pushdown_info;

namespace {

class transaction_rollback_guard final {
 public:
  explicit transaction_rollback_guard(duckdb::Connection& connection) : _connection(connection)
  {
    _connection.BeginTransaction();
  }

  ~transaction_rollback_guard() noexcept
  {
    try {
      _connection.Rollback();
    } catch (...) {
      // Cleanup must not mask the original Catch failure; Connection destruction is the fallback.
    }
  }

  transaction_rollback_guard(transaction_rollback_guard const&)            = delete;
  transaction_rollback_guard& operator=(transaction_rollback_guard const&) = delete;
  transaction_rollback_guard(transaction_rollback_guard&&)                 = delete;
  transaction_rollback_guard& operator=(transaction_rollback_guard&&)      = delete;

 private:
  duckdb::Connection& _connection;
};

/// Construct a representative JoinFilterPushdownInfo: two join-condition indices, one probe
/// target with two columns, and a caller-provided DynamicTableFilterSet (so the test can assert
/// pointer identity).
duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> make_pushdown_info(
  duckdb::shared_ptr<duckdb::DynamicTableFilterSet> dyn_filters)
{
  auto info                   = duckdb::make_uniq<duckdb::JoinFilterPushdownInfo>();
  info->join_condition        = {0, 1};
  info->build_side_has_filter = true;

  duckdb::JoinFilterPushdownFilter pi;
  pi.dynamic_filters = std::move(dyn_filters);

  duckdb::JoinFilterPushdownColumn col_a;
  col_a.probe_column_index = duckdb::ColumnBinding{1, 2};
  col_a.storage_type       = duckdb::LogicalType::INTEGER;
  duckdb::JoinFilterPushdownColumn col_b;
  col_b.probe_column_index = duckdb::ColumnBinding{1, 3};
  col_b.storage_type       = duckdb::LogicalType::BIGINT;
  pi.columns               = {col_a, col_b};

  info->probe_info.push_back(std::move(pi));
  return info;
}

void require_query_ok(duckdb::Connection& connection, std::string const& sql)
{
  auto result = connection.Query(sql);
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

duckdb::LogicalComparisonJoin const* find_dynamic_filter_join(duckdb::LogicalOperator const& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    auto const& join = op.Cast<duckdb::LogicalComparisonJoin>();
    if (join.filter_pushdown && !join.filter_pushdown->probe_info.empty()) { return &join; }
  }
  for (auto const& child : op.children) {
    if (auto const* join = find_dynamic_filter_join(*child)) { return join; }
  }
  return nullptr;
}

duckdb::LogicalGet const* find_get_with_channel(duckdb::LogicalOperator const& op,
                                                duckdb::DynamicTableFilterSet const* channel)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto const& get = op.Cast<duckdb::LogicalGet>();
    if (get.dynamic_filters.get() == channel) { return &get; }
  }
  for (auto const& child : op.children) {
    if (auto const* get = find_get_with_channel(*child, channel)) { return get; }
  }
  return nullptr;
}
}  // namespace

TEST_CASE("clone_sirius_filter_pushdown_info copies scalar fields", "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto src = make_pushdown_info(dyn);
  auto dst = clone_sirius_filter_pushdown_info(*src);

  REQUIRE(dst->join_condition == src->join_condition);
  REQUIRE(dst->build_side_has_filter == src->build_side_has_filter);
  REQUIRE(dst->probe_info.size() == src->probe_info.size());
}

TEST_CASE("clone_sirius_filter_pushdown_info intentionally omits DuckDB min-max aggregates",
          "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto src = make_pushdown_info(dyn);
  // The expression is an opaque non-null sentinel; this test covers the container-level policy.
  src->min_max_aggregates.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0ULL));

  auto dst = clone_sirius_filter_pushdown_info(*src);

  REQUIRE(src->min_max_aggregates.size() == 1);
  REQUIRE(dst->min_max_aggregates.empty());
}

TEST_CASE("clone_sirius_filter_pushdown_info shares DynamicTableFilterSet pointer identity",
          "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto src = make_pushdown_info(dyn);
  auto dst = clone_sirius_filter_pushdown_info(*src);

  // Critical: the DynamicTableFilterSet pointer is the route key paired between the join's
  // filter_pushdown and the consumer scan's dynamic_filters. Deep-copying would break that
  // pairing.
  REQUIRE(dst->probe_info[0].dynamic_filters.get() == dyn.get());
  REQUIRE(dst->probe_info[0].dynamic_filters.get() == src->probe_info[0].dynamic_filters.get());
}

TEST_CASE("clone_sirius_filter_pushdown_info copies column bindings and storage types",
          "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto src = make_pushdown_info(dyn);
  auto dst = clone_sirius_filter_pushdown_info(*src);

  auto const& src_cols = src->probe_info[0].columns;
  auto const& dst_cols = dst->probe_info[0].columns;
  REQUIRE(dst_cols.size() == src_cols.size());
  for (std::size_t i = 0; i < src_cols.size(); ++i) {
    REQUIRE(dst_cols[i].probe_column_index == src_cols[i].probe_column_index);
    REQUIRE(dst_cols[i].storage_type == src_cols[i].storage_type);
  }
}

TEST_CASE("preserve_dynamic_filter_metadata is a no-op when original has no metadata",
          "[transparent][preserve]")
{
  duckdb::LogicalComparisonJoin original(duckdb::JoinType::INNER);
  duckdb::LogicalComparisonJoin copy(duckdb::JoinType::INNER);

  preserve_dynamic_filter_metadata(original, copy);

  REQUIRE_FALSE(copy.filter_pushdown);
}

TEST_CASE("preserve_dynamic_filter_metadata copies filter_pushdown onto the copy",
          "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  duckdb::LogicalComparisonJoin original(duckdb::JoinType::INNER);
  original.filter_pushdown = make_pushdown_info(dyn);

  duckdb::LogicalComparisonJoin copy(duckdb::JoinType::INNER);

  preserve_dynamic_filter_metadata(original, copy);

  REQUIRE(copy.filter_pushdown);
  REQUIRE(copy.filter_pushdown->probe_info[0].dynamic_filters.get() == dyn.get());
  // Original is untouched — DuckDB's CPU fallback path still consumes it.
  REQUIRE(original.filter_pushdown);
}

TEST_CASE("preserve_dynamic_filter_metadata recurses into children", "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();

  // Tree: outer_join → [child_join_with_pushdown, leaf_join]
  duckdb::LogicalComparisonJoin original_outer(duckdb::JoinType::INNER);
  auto original_child = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  original_child->filter_pushdown = make_pushdown_info(dyn);
  auto original_leaf = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  original_outer.children.push_back(std::move(original_child));
  original_outer.children.push_back(std::move(original_leaf));

  // Parallel copy tree (same structure, all filter_pushdown null).
  duckdb::LogicalComparisonJoin copy_outer(duckdb::JoinType::INNER);
  copy_outer.children.push_back(
    duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER));
  copy_outer.children.push_back(
    duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER));

  preserve_dynamic_filter_metadata(original_outer, copy_outer);

  REQUIRE(copy_outer.children[0]->Cast<duckdb::LogicalComparisonJoin>().filter_pushdown);
  REQUIRE_FALSE(copy_outer.children[1]->Cast<duckdb::LogicalComparisonJoin>().filter_pushdown);
  REQUIRE_FALSE(copy_outer.filter_pushdown);
}

TEST_CASE("preserve_dynamic_filter_metadata bails on top-level structural mismatch",
          "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  duckdb::LogicalComparisonJoin original(duckdb::JoinType::INNER);
  original.filter_pushdown = make_pushdown_info(dyn);
  original.children.push_back(
    duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER));

  duckdb::LogicalComparisonJoin copy(duckdb::JoinType::INNER);
  // No children — child-count mismatch fails the whole-tree preflight, so nothing is preserved.

  preserve_dynamic_filter_metadata(original, copy);

  REQUIRE_FALSE(copy.filter_pushdown);
}

TEST_CASE("copy_logical_plan preserves real DuckDB join-to-GET channel identity",
          "[transparent][preserve]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection connection(db);

  // LogicalOperator::Copy deserializes table-scan bindings through the catalog, so use real
  // catalog-backed tables and keep their in-memory database/context alive through the copy.
  require_query_ok(connection,
                   "CREATE TABLE probe AS "
                   "SELECT i::INTEGER AS k, i::BIGINT AS payload FROM range(32) AS r(i)");
  require_query_ok(connection,
                   "CREATE TABLE build AS SELECT i::INTEGER AS k FROM range(8) AS r(i)");

  // Deserialization resolves catalog entries through ActiveTransaction(), matching the optimizer
  // hook's production prepare-time context. The guard rolls back even if a REQUIRE aborts the test.
  transaction_rollback_guard transaction{connection};

  auto original =
    connection.ExtractPlan("SELECT p.payload FROM probe AS p JOIN build AS b ON p.k = b.k");
  REQUIRE(original);

  auto const* original_join = find_dynamic_filter_join(*original);
  REQUIRE(original_join != nullptr);
  REQUIRE(original_join->filter_pushdown);
  REQUIRE(original_join->filter_pushdown->probe_info.size() == 1);

  auto const original_channel = original_join->filter_pushdown->probe_info.front().dynamic_filters;
  REQUIRE(original_channel);
  auto const* original_get = find_get_with_channel(*original, original_channel.get());
  REQUIRE(original_get != nullptr);

  auto copy = sirius::transparent::copy_logical_plan(*original, *connection.context);
  REQUIRE(copy);
  REQUIRE(copy.get() != original.get());

  auto const* copied_join = find_dynamic_filter_join(*copy);
  REQUIRE(copied_join != nullptr);
  REQUIRE(copied_join != original_join);
  REQUIRE(copied_join->filter_pushdown.get() != original_join->filter_pushdown.get());
  REQUIRE(copied_join->filter_pushdown->probe_info.size() == 1);

  auto const copied_channel = copied_join->filter_pushdown->probe_info.front().dynamic_filters;
  REQUIRE(copied_channel.get() == original_channel.get());
  auto const* copied_get = find_get_with_channel(*copy, copied_channel.get());
  REQUIRE(copied_get != nullptr);
  REQUIRE(copied_get != original_get);
  REQUIRE(copied_get->dynamic_filters.get() == copied_channel.get());

  // The source remains intact for DuckDB's CPU fallback path.
  REQUIRE(original_join->filter_pushdown);
  REQUIRE(original_get->dynamic_filters.get() == original_channel.get());
}
