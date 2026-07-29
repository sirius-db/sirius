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
 * @file test_duckdb_join_filter_candidate_adapter.cpp
 * @brief Contract tests for the version-pinned adapter's extraction/classification surface
 *        (sirius::planner::duckdb_join_filter_candidate_adapter::extract / scan_channel_identity)
 *        and the join-to-GET channel-identity topology the preservation walk must keep intact.
 *
 * These are the pin-bump sentinels. Every `malformed` case below is unreachable under the pinned
 * DuckDB: it exists to fail loudly if a future submodule bump breaks a structural invariant that
 * the positional key-to-column pairing depends on.
 */

#include "planner/duckdb_join_filter_candidate_adapter.hpp"

#include <catch.hpp>
#include <duckdb/common/typedefs.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

using sirius::planner::duckdb_candidate_kind;
using sirius::planner::duckdb_join_filter_candidate;
using sirius::planner::duckdb_join_filter_candidate_adapter::extract;
using sirius::planner::duckdb_join_filter_candidate_adapter::preserve_dynamic_filter_metadata;
using sirius::planner::duckdb_join_filter_candidate_adapter::scan_channel_identity;

namespace {

duckdb::JoinCondition make_condition(duckdb::ExpressionType comparison)
{
  return duckdb::JoinCondition(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0ULL),
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0ULL),
    comparison);
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_join(
  std::initializer_list<duckdb::ExpressionType> comparisons)
{
  auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  for (auto comparison : comparisons) {
    join->conditions.push_back(make_condition(comparison));
  }
  return join;
}

/// One probe target: @p column_count columns, INTEGER storage, channel @p dyn (may be null).
duckdb::JoinFilterPushdownFilter make_target(duckdb::shared_ptr<duckdb::DynamicTableFilterSet> dyn,
                                             duckdb::idx_t column_count)
{
  duckdb::JoinFilterPushdownFilter pi;
  pi.dynamic_filters = std::move(dyn);
  for (duckdb::idx_t i = 0; i < column_count; ++i) {
    duckdb::JoinFilterPushdownColumn col;
    col.probe_column_index =
      duckdb::ColumnBinding{duckdb::TableIndex(0), duckdb::ProjectionIndex(i)};
    col.storage_type = duckdb::LogicalType::INTEGER;
    pi.columns.push_back(col);
  }
  return pi;
}

duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> make_pushdown_info(
  duckdb::vector<duckdb::idx_t> join_condition, bool build_side_has_filter = true)
{
  auto info                   = duckdb::make_uniq<duckdb::JoinFilterPushdownInfo>();
  info->join_condition        = std::move(join_condition);
  info->build_side_has_filter = build_side_has_filter;
  return info;
}

/// A minimal constructible LogicalGet (the default ctor is private; TableFunction() is enough for
/// a structural test — the walk and the adapter never invoke the function).
duckdb::unique_ptr<duckdb::LogicalGet> make_get()
{
  return duckdb::make_uniq<duckdb::LogicalGet>(
    duckdb::TableIndex(0),
    duckdb::TableFunction(),
    /*bind_data=*/nullptr,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
    duckdb::vector<duckdb::Identifier>{"a"});
}

/// The malformed contract: only `kind` is meaningful; every other field is default/empty.
void require_malformed_carries_only_kind(duckdb_join_filter_candidate const& c)
{
  REQUIRE(c.kind() == duckdb_candidate_kind::malformed);
  REQUIRE_FALSE(c.build_subtree_has_filter_hint());
  REQUIRE(c.condition_indexes().empty());
  REQUIRE(c.condition_comparisons().empty());
  REQUIRE(c.targets().empty());
}

}  // namespace

//===-----------------------------------------------------------------------------------------===//
// Classification
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("extract classifies absent when the join has no pushdown metadata",
          "[dynamic_filter][adapter]")
{
  auto join      = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
  auto candidate = extract(*join);

  REQUIRE(candidate.kind() == duckdb_candidate_kind::absent);
  REQUIRE_FALSE(candidate.build_subtree_has_filter_hint());
  REQUIRE(candidate.condition_indexes().empty());
  REQUIRE(candidate.condition_comparisons().empty());
  REQUIRE(candidate.targets().empty());
}

TEST_CASE("extract classifies statistics_only when probe_info is empty",
          "[dynamic_filter][adapter]")
{
  auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0}, false);

  auto candidate = extract(*join);

  REQUIRE(candidate.kind() == duckdb_candidate_kind::statistics_only);
  REQUIRE_FALSE(candidate.build_subtree_has_filter_hint());
  REQUIRE(candidate.condition_indexes().empty());
  REQUIRE(candidate.targets().empty());
  REQUIRE(candidate.condition_comparisons().empty());
}

TEST_CASE("extract rejects anomalous targetless metadata", "[dynamic_filter][adapter]")
{
  SECTION("empty condition ordinals")
  {
    auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
    join->filter_pushdown = make_pushdown_info({}, false);

    require_malformed_carries_only_kind(extract(*join));
  }

  SECTION("out-of-range condition ordinal")
  {
    auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
    join->filter_pushdown = make_pushdown_info({1}, false);

    require_malformed_carries_only_kind(extract(*join));
  }

  SECTION("duplicate condition ordinals")
  {
    auto join =
      make_join({duckdb::ExpressionType::COMPARE_EQUAL, duckdb::ExpressionType::COMPARE_EQUAL});
    join->filter_pushdown = make_pushdown_info({0, 0}, false);

    require_malformed_carries_only_kind(extract(*join));
  }

  SECTION("build-side filter hint without a probe target")
  {
    auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
    join->filter_pushdown = make_pushdown_info({0}, true);

    require_malformed_carries_only_kind(extract(*join));
  }
}

TEST_CASE("extract admits a candidate and snapshots ordinals, comparisons, and target columns",
          "[dynamic_filter][adapter]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join =
    make_join({duckdb::ExpressionType::COMPARE_EQUAL, duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0, 1});
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 2));

  auto candidate = extract(*join);

  REQUIRE(candidate.kind() == duckdb_candidate_kind::admitted);
  REQUIRE(candidate.build_subtree_has_filter_hint());
  REQUIRE(candidate.condition_indexes() == std::vector<std::size_t>{0, 1});
  REQUIRE(candidate.condition_comparisons() ==
          std::vector<duckdb::ExpressionType>{duckdb::ExpressionType::COMPARE_EQUAL,
                                              duckdb::ExpressionType::COMPARE_EQUAL});
  REQUIRE(candidate.targets().size() == 1);
  REQUIRE(candidate.targets()[0].columns().size() == 2);
  REQUIRE(candidate.targets()[0].columns()[0].column_index == 0);
  REQUIRE(candidate.targets()[0].columns()[1].column_index == 1);
  REQUIRE(candidate.targets()[0].columns()[0].storage_type == duckdb::LogicalType::INTEGER);
}

TEST_CASE("extract preserves the filter-ordinal to condition-index mapping",
          "[dynamic_filter][adapter]")
{
  auto dyn  = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join = make_join({duckdb::ExpressionType::COMPARE_EQUAL,
                         duckdb::ExpressionType::COMPARE_GREATERTHAN,
                         duckdb::ExpressionType::COMPARE_EQUAL});
  // DuckDB selected conditions 2 and 0, in that order: filter ordinal j is the POSITION here, and
  // the value at that position is the condition index it names. The two must never be swapped.
  join->filter_pushdown = make_pushdown_info({2, 0});
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 2));

  auto candidate = extract(*join);

  REQUIRE(candidate.kind() == duckdb_candidate_kind::admitted);
  REQUIRE(candidate.condition_indexes() == std::vector<std::size_t>{2, 0});
  // Comparisons are looked up THROUGH the condition index, not by ordinal position.
  REQUIRE(candidate.condition_comparisons() ==
          std::vector<duckdb::ExpressionType>{duckdb::ExpressionType::COMPARE_EQUAL,
                                              duckdb::ExpressionType::COMPARE_EQUAL});
  // Target columns stay in filter-ordinal space, parallel to condition_indexes.
  REQUIRE(candidate.targets()[0].columns().size() == 2);
  REQUIRE(candidate.targets()[0].columns()[0].column_index == 0);
  REQUIRE(candidate.targets()[0].columns()[1].column_index == 1);
}

TEST_CASE("extract keeps range comparisons at full arity", "[dynamic_filter][adapter]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join =
    make_join({duckdb::ExpressionType::COMPARE_EQUAL, duckdb::ExpressionType::COMPARE_GREATERTHAN});
  join->filter_pushdown = make_pushdown_info({0, 1});
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 2));

  auto candidate = extract(*join);

  // Range ordinals are valid candidates (narrowed per key later, C1a-2), never malformed.
  REQUIRE(candidate.kind() == duckdb_candidate_kind::admitted);
  REQUIRE(candidate.condition_indexes() == std::vector<std::size_t>{0, 1});
  REQUIRE(candidate.condition_comparisons()[1] == duckdb::ExpressionType::COMPARE_GREATERTHAN);
}

TEST_CASE("extract shares one key set across every target", "[dynamic_filter][adapter]")
{
  auto dyn_a = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto dyn_b = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join =
    make_join({duckdb::ExpressionType::COMPARE_EQUAL, duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0, 1});
  // Two scans (e.g. a UNION-ed probe side): the key set is a property of the JOIN, and each target
  // records only where those same keys land in its own scan.
  join->filter_pushdown->probe_info.push_back(make_target(dyn_a, 2));
  join->filter_pushdown->probe_info.push_back(make_target(dyn_b, 2));

  auto candidate = extract(*join);

  REQUIRE(candidate.kind() == duckdb_candidate_kind::admitted);
  REQUIRE(candidate.condition_indexes().size() == 2);  // one shared key set, not one per target
  REQUIRE(candidate.targets().size() == 2);
  REQUIRE(candidate.targets()[0].channel_identity().get() == dyn_a.get());
  REQUIRE(candidate.targets()[1].channel_identity().get() == dyn_b.get());
  // Every target carries the full key arity.
  REQUIRE(candidate.targets()[0].columns().size() == candidate.condition_indexes().size());
  REQUIRE(candidate.targets()[1].columns().size() == candidate.condition_indexes().size());
}

//===-----------------------------------------------------------------------------------------===//
// Fail-closed fences (malformed cases are unreachable at the pin) and null-channel handling
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("extract fails closed on an out-of-range condition index", "[dynamic_filter][adapter]")
{
  auto dyn              = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({5});  // join has 1 condition
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 1));

  require_malformed_carries_only_kind(extract(*join));
}

TEST_CASE("extract fails closed on a duplicate condition index", "[dynamic_filter][adapter]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join =
    make_join({duckdb::ExpressionType::COMPARE_EQUAL, duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0, 0});
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 2));

  require_malformed_carries_only_kind(extract(*join));
}

TEST_CASE("extract fails closed on recorded targets with no condition ordinals",
          "[dynamic_filter][adapter]")
{
  auto dyn              = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({});                    // empty join_condition…
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 0));  // …with a recorded target

  require_malformed_carries_only_kind(extract(*join));
}

TEST_CASE("extract fails closed on target column-arity mismatch", "[dynamic_filter][adapter]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join =
    make_join({duckdb::ExpressionType::COMPARE_EQUAL, duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0, 1});
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 1));  // 1 column, 2 ordinals

  require_malformed_carries_only_kind(extract(*join));
}

TEST_CASE("extract drops a null-channel target and keeps its live sibling",
          "[dynamic_filter][adapter]")
{
  auto dyn              = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0});
  join->filter_pushdown->probe_info.push_back(make_target(nullptr, 1));  // null channel: dropped
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 1));      // live sibling: kept

  auto candidate = extract(*join);

  REQUIRE(candidate.kind() == duckdb_candidate_kind::admitted);
  REQUIRE(candidate.targets().size() == 1);
  REQUIRE(candidate.targets()[0].channel_identity().get() == dyn.get());
}

TEST_CASE("extract fails closed when every recorded target has a null channel",
          "[dynamic_filter][adapter]")
{
  auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0});
  join->filter_pushdown->probe_info.push_back(make_target(nullptr, 1));
  join->filter_pushdown->probe_info.push_back(make_target(nullptr, 1));

  // NOT statistics_only: that kind is reserved for DuckDB's deliberate zero-target state (its
  // telemetry count sizes the Track E opportunity); an anomaly must not inflate it.
  require_malformed_carries_only_kind(extract(*join));
}

TEST_CASE("extract checks target arity before the null-channel drop", "[dynamic_filter][adapter]")
{
  auto dyn              = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0});
  join->filter_pushdown->probe_info.push_back(make_target(nullptr, 2));  // null AND corrupt arity
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 1));      // live sibling

  // Arity corruption impeaches the candidate even on a null-channel target: the whole candidate
  // fails closed rather than the corrupt entry being silently dropped.
  require_malformed_carries_only_kind(extract(*join));
}

//===-----------------------------------------------------------------------------------------===//
// Channel identity: ownership and the join-to-GET pairing
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("extracted candidate co-owns the channel identity", "[dynamic_filter][adapter]")
{
  auto dyn              = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0});
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 1));

  auto* raw      = dyn.get();
  auto candidate = extract(*join);
  join.reset();  // destroy the plan the candidate was extracted from

  REQUIRE(candidate.targets()[0].channel_identity().get() == raw);
  // Exactly two owners remain: the candidate and `dyn`.
  REQUIRE(candidate.targets()[0].channel_identity().use_count() == 2);

  // The candidate's shared ownership keeps the identity alive on its own: release the last other
  // holder and the candidate remains the sole owner of a live object.
  dyn.reset();
  REQUIRE(candidate.targets()[0].channel_identity().use_count() == 1);
}

TEST_CASE("adapter values preserve their invariants after assignment and candidate moves",
          "[dynamic_filter][adapter]")
{
  auto dyn              = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join             = make_join({duckdb::ExpressionType::COMPARE_EQUAL});
  join->filter_pushdown = make_pushdown_info({0});
  join->filter_pushdown->probe_info.push_back(make_target(dyn, 1));

  auto candidate       = extract(*join);
  auto assigned_target = candidate.targets().front();
  assigned_target      = candidate.targets().front();
  REQUIRE(assigned_target.channel_identity().get() == dyn.get());
  REQUIRE_FALSE(assigned_target.columns().empty());

  auto copy_assigned_candidate = extract(*join);
  copy_assigned_candidate      = candidate;
  REQUIRE(copy_assigned_candidate.kind() == duckdb_candidate_kind::admitted);
  REQUIRE_FALSE(copy_assigned_candidate.targets().empty());

  auto moved_candidate = std::move(candidate);
  REQUIRE(moved_candidate.kind() == duckdb_candidate_kind::admitted);
  REQUIRE_FALSE(moved_candidate.targets().empty());
  REQUIRE(candidate.kind() == duckdb_candidate_kind::absent);
  REQUIRE_FALSE(candidate.build_subtree_has_filter_hint());
  REQUIRE(candidate.condition_indexes().empty());
  REQUIRE(candidate.condition_comparisons().empty());
  REQUIRE(candidate.targets().empty());

  auto move_assigned_candidate = extract(*join);
  move_assigned_candidate      = std::move(moved_candidate);
  REQUIRE(move_assigned_candidate.kind() == duckdb_candidate_kind::admitted);
  REQUIRE_FALSE(move_assigned_candidate.targets().empty());
  REQUIRE(moved_candidate.kind() == duckdb_candidate_kind::absent);
  REQUIRE_FALSE(moved_candidate.build_subtree_has_filter_hint());
  REQUIRE(moved_candidate.condition_indexes().empty());
  REQUIRE(moved_candidate.condition_comparisons().empty());
  REQUIRE(moved_candidate.targets().empty());
}

TEST_CASE("scan_channel_identity returns the gets channel or null", "[dynamic_filter][adapter]")
{
  auto get = make_get();
  REQUIRE(scan_channel_identity(*get) == nullptr);

  auto dyn             = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  get->dynamic_filters = dyn;
  REQUIRE(scan_channel_identity(*get).get() == dyn.get());
}

TEST_CASE("preservation keeps a producing join and its GET paired to one channel",
          "[dynamic_filter][adapter]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();

  // Original: join → [get(with channel), get(plain)]; the join's target names the same channel.
  duckdb::LogicalComparisonJoin original(duckdb::JoinType::INNER);
  {
    auto probe_get             = make_get();
    probe_get->dynamic_filters = dyn;
    original.children.push_back(std::move(probe_get));
    original.children.push_back(make_get());
    original.conditions.push_back(make_condition(duckdb::ExpressionType::COMPARE_EQUAL));
    original.filter_pushdown = make_pushdown_info({0});
    original.filter_pushdown->probe_info.push_back(make_target(dyn, 1));
  }

  // Structurally aligned copy, metadata stripped (the post-Copy state).
  duckdb::LogicalComparisonJoin copy(duckdb::JoinType::INNER);
  copy.children.push_back(make_get());
  copy.children.push_back(make_get());
  copy.conditions.push_back(make_condition(duckdb::ExpressionType::COMPARE_EQUAL));

  preserve_dynamic_filter_metadata(original, copy);

  // The pairing invariant: the copied join's target and the copied GET reference the SAME set —
  // and extraction on the copy surfaces that same identity.
  auto& copy_get = copy.children[0]->Cast<duckdb::LogicalGet>();
  REQUIRE(copy.filter_pushdown);
  REQUIRE(copy.filter_pushdown->probe_info[0].dynamic_filters.get() == dyn.get());
  REQUIRE(copy_get.dynamic_filters.get() == dyn.get());
  auto reextracted = extract(copy);
  REQUIRE(reextracted.kind() == duckdb_candidate_kind::admitted);
  REQUIRE(reextracted.targets()[0].channel_identity().get() ==
          scan_channel_identity(copy_get).get());
}

TEST_CASE("preservation is all-or-nothing on a descendant structural mismatch",
          "[dynamic_filter][adapter]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();

  // Original: join(pushdown) → [get(with channel), get].
  duckdb::LogicalComparisonJoin original(duckdb::JoinType::INNER);
  {
    auto probe_get             = make_get();
    probe_get->dynamic_filters = dyn;
    original.children.push_back(std::move(probe_get));
    original.children.push_back(make_get());
    original.conditions.push_back(make_condition(duckdb::ExpressionType::COMPARE_EQUAL));
    original.filter_pushdown = make_pushdown_info({0});
    original.filter_pushdown->probe_info.push_back(make_target(dyn, 1));
  }

  // Copy: aligned AT the join, but child 0 is a join instead of a get — descendant mismatch.
  duckdb::LogicalComparisonJoin copy(duckdb::JoinType::INNER);
  copy.children.push_back(
    duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER));
  copy.children.push_back(make_get());
  copy.conditions.push_back(make_condition(duckdb::ExpressionType::COMPARE_EQUAL));

  preserve_dynamic_filter_metadata(original, copy);

  // The locally-aligned ancestor must receive NOTHING: attaching its metadata while the scan
  // below cannot be restored would half-build the join-to-GET pairing.
  REQUIRE_FALSE(copy.filter_pushdown);
}

TEST_CASE("preservation keeps two producing joins sharing one GET channel",
          "[dynamic_filter][adapter]")
{
  auto dyn = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();

  // Original: P2 → [P1 → [get(with channel), get], get]; both joins target the same channel.
  duckdb::LogicalComparisonJoin original_p2(duckdb::JoinType::INNER);
  {
    auto p1        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
    auto probe_get = make_get();
    probe_get->dynamic_filters = dyn;
    p1->children.push_back(std::move(probe_get));
    p1->children.push_back(make_get());
    p1->conditions.push_back(make_condition(duckdb::ExpressionType::COMPARE_EQUAL));
    p1->filter_pushdown = make_pushdown_info({0});
    p1->filter_pushdown->probe_info.push_back(make_target(dyn, 1));

    original_p2.children.push_back(std::move(p1));
    original_p2.children.push_back(make_get());
    original_p2.conditions.push_back(make_condition(duckdb::ExpressionType::COMPARE_EQUAL));
    original_p2.filter_pushdown = make_pushdown_info({0});
    original_p2.filter_pushdown->probe_info.push_back(make_target(dyn, 1));
  }

  // Structurally aligned copy.
  duckdb::LogicalComparisonJoin copy_p2(duckdb::JoinType::INNER);
  {
    auto p1 = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
    p1->children.push_back(make_get());
    p1->children.push_back(make_get());
    copy_p2.children.push_back(std::move(p1));
    copy_p2.children.push_back(make_get());
  }

  preserve_dynamic_filter_metadata(original_p2, copy_p2);

  // N-producer/one-consumer: both copied joins and the copied GET share ONE set.
  auto& copy_p1  = copy_p2.children[0]->Cast<duckdb::LogicalComparisonJoin>();
  auto& copy_get = copy_p1.children[0]->Cast<duckdb::LogicalGet>();
  REQUIRE(copy_p2.filter_pushdown);
  REQUIRE(copy_p1.filter_pushdown);
  REQUIRE(copy_p2.filter_pushdown->probe_info[0].dynamic_filters.get() == dyn.get());
  REQUIRE(copy_p1.filter_pushdown->probe_info[0].dynamic_filters.get() == dyn.get());
  REQUIRE(copy_get.dynamic_filters.get() == dyn.get());
}
