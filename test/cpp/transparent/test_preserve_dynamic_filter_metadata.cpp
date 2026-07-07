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
 * @brief Tests for the transparent-optimizer post-Copy fixup helpers
 *        (sirius::transparent::detail::clone_filter_pushdown_info,
 *         sirius::transparent::detail::preserve_dynamic_filter_metadata).
 *
 * LogicalOperator::Copy round-trips through serialize/deserialize. LogicalComparisonJoin's
 * filter_pushdown and LogicalGet's dynamic_filters are not in the serialization schemas, so the
 * copy has them null by default. The helpers under test walk both trees in parallel and re-attach
 * those fields, sharing the DynamicTableFilterSet shared_ptrs to preserve the route-key pointer
 * identity that downstream wiring (Phase 1.1 of dynamic filter pushdown) relies on.
 */

#include "transparent/sirius_optimizer_extension.hpp"

#include <catch.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <utility>

using sirius::transparent::detail::clone_filter_pushdown_info;
using sirius::transparent::detail::preserve_dynamic_filter_metadata;
using sirius::transparent::detail::preserved_counts;

namespace {

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

}  // namespace

TEST_CASE("clone_filter_pushdown_info copies scalar fields", "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto src = make_pushdown_info(dyn);
  auto dst = clone_filter_pushdown_info(*src);

  REQUIRE(dst->join_condition == src->join_condition);
  REQUIRE(dst->build_side_has_filter == src->build_side_has_filter);
  REQUIRE(dst->probe_info.size() == src->probe_info.size());
}

TEST_CASE("clone_filter_pushdown_info shares DynamicTableFilterSet pointer identity",
          "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto src = make_pushdown_info(dyn);
  auto dst = clone_filter_pushdown_info(*src);

  // Critical: the DynamicTableFilterSet pointer is the route key paired between the join's
  // filter_pushdown and the consumer scan's dynamic_filters. Deep-copying would break that
  // pairing.
  REQUIRE(dst->probe_info[0].dynamic_filters.get() == dyn.get());
  REQUIRE(dst->probe_info[0].dynamic_filters.get() == src->probe_info[0].dynamic_filters.get());
}

TEST_CASE("clone_filter_pushdown_info copies column bindings and storage types",
          "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto src = make_pushdown_info(dyn);
  auto dst = clone_filter_pushdown_info(*src);

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

  preserved_counts counts;
  preserve_dynamic_filter_metadata(original, copy, counts);

  REQUIRE(counts.joins == 0);
  REQUIRE(counts.gets == 0);
  REQUIRE_FALSE(copy.filter_pushdown);
}

TEST_CASE("preserve_dynamic_filter_metadata copies filter_pushdown onto the copy",
          "[transparent][preserve]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  duckdb::LogicalComparisonJoin original(duckdb::JoinType::INNER);
  original.filter_pushdown = make_pushdown_info(dyn);

  duckdb::LogicalComparisonJoin copy(duckdb::JoinType::INNER);

  preserved_counts counts;
  preserve_dynamic_filter_metadata(original, copy, counts);

  REQUIRE(counts.joins == 1);
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

  preserved_counts counts;
  preserve_dynamic_filter_metadata(original_outer, copy_outer, counts);

  REQUIRE(counts.joins == 1);
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
  // No children — child-count mismatch triggers the defensive bail.

  preserved_counts counts;
  preserve_dynamic_filter_metadata(original, copy, counts);

  REQUIRE(counts.joins == 0);
  REQUIRE_FALSE(copy.filter_pushdown);
}
