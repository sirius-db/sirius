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
 * @file test_surrogate_groupby.cpp
 * @brief Functional tests for the surrogate-key group-by rewrite (late string
 *        materialization): the deferral store's addressing/retention contract, the planner
 *        pass's activation shape and decline gates over hand-built operator trees, and the
 *        MERGE_GROUP_BY finalization paths (fast path, conservative full-tuple re-group,
 *        floating-point NaN gate, multi-source base-offset addressing, store release).
 */

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "op/groupby_surrogate_deferral.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/sirius_plan_surrogate_groupby.hpp"
#include "utils/data_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/table/table.hpp>

#include <catch.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;
using sirius::test::vector_to_cudf_column;

namespace {

sirius::logical_type bigint_type() { return sirius::logical_type::make(sirius::type_id::BIGINT); }
sirius::logical_type tinyint_type() { return sirius::logical_type::make(sirius::type_id::TINYINT); }
sirius::logical_type varchar_type() { return sirius::logical_type::make(sirius::type_id::VARCHAR); }
sirius::logical_type double_type() { return sirius::logical_type::make(sirius::type_id::DOUBLE); }

std::unique_ptr<sirius::ast::node> make_reference(uint32_t column_index, sirius::logical_type type)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::reference{column_index, std::move(type)});
}

/// A TABLE_SCAN leaf with the given logical output types (identity projection).
duckdb::unique_ptr<sirius_physical_table_scan> make_scan(duckdb::vector<sirius::logical_type> types,
                                                         std::size_t estimated_cardinality)
{
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<std::size_t> projection_ids;
  duckdb::vector<std::string> names;
  for (std::size_t i = 0; i < types.size(); i++) {
    column_ids.emplace_back(i);
    projection_ids.push_back(i);
    names.push_back("c" + std::to_string(i));
  }
  duckdb::TableFunction function("test_scan", {}, nullptr, nullptr);
  return duckdb::make_uniq<sirius_physical_table_scan>(types,
                                                       std::move(function),
                                                       /*bind_data=*/nullptr,
                                                       types,
                                                       std::move(column_ids),
                                                       std::move(projection_ids),
                                                       std::move(names),
                                                       duckdb::make_uniq<duckdb::TableFilterSet>(),
                                                       estimated_cardinality,
                                                       duckdb::ExtraOperatorInfo(),
                                                       duckdb::vector<duckdb::Value>{},
                                                       duckdb::virtual_column_map_t{});
}

/// An INNER (or other) hash join on column 0 of both inputs; output = all left + all right cols.
duckdb::unique_ptr<sirius_physical_hash_join> make_join(
  duckdb::JoinType join_type,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> left,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> right,
  std::size_t estimated_cardinality)
{
  duckdb::vector<duckdb::LogicalType> output_types;
  for (auto const& t : left->types) {
    output_types.push_back(sirius::to_duckdb(t));
  }
  for (auto const& t : right->types) {
    output_types.push_back(sirius::to_duckdb(t));
  }
  duckdb::LogicalDummyScan stub(0);
  stub.types                 = std::move(output_types);
  stub.estimated_cardinality = estimated_cardinality;
  duckdb::vector<sirius::join_condition> conditions;
  sirius::join_condition condition;
  condition.left  = make_reference(0, bigint_type());
  condition.right = make_reference(0, bigint_type());
  conditions.push_back(std::move(condition));
  auto join = duckdb::make_uniq<sirius_physical_hash_join>(stub,
                                                           std::move(left),
                                                           std::move(right),
                                                           std::move(conditions),
                                                           join_type,
                                                           estimated_cardinality);
  return join;
}

/// A projection of pure references (input index, type) pairs.
duckdb::unique_ptr<sirius_physical_projection> make_ref_projection(
  std::vector<std::pair<uint32_t, sirius::logical_type>> refs,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
  std::size_t estimated_cardinality)
{
  duckdb::vector<sirius::logical_type> types;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
  for (auto& [idx, type] : refs) {
    types.push_back(type);
    select_list.push_back(make_reference(idx, type));
  }
  auto projection = duckdb::make_uniq<sirius_physical_projection>(
    std::move(types), std::move(select_list), estimated_cardinality);
  projection->children.push_back(std::move(child));
  return projection;
}

/// A HASH_GROUP_BY over the given (index, type) group keys plus SUM(agg_index BIGINT).
duckdb::unique_ptr<sirius_physical_grouped_aggregate> make_group_by(
  std::vector<std::pair<uint32_t, sirius::logical_type>> group_refs,
  uint32_t agg_index,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
  std::size_t estimated_cardinality)
{
  duckdb::vector<sirius::logical_type> types;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups;
  for (auto& [idx, type] : group_refs) {
    types.push_back(type);
    auto ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(sirius::to_duckdb(type), idx);
    groups.push_back(sirius::ast::from_duckdb(*ref));
  }
  types.push_back(bigint_type());

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> agg_children;
  agg_children.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, agg_index));
  auto agg_function = sirius::test::MakeDummyAggregate(
    "sum", {duckdb::LogicalType::BIGINT}, duckdb::LogicalType::BIGINT);
  auto agg_expr = duckdb::make_uniq<duckdb::BoundAggregateExpression>(
    agg_function, std::move(agg_children), nullptr, nullptr, duckdb::AggregateType::NON_DISTINCT);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;
  aggregates.push_back(sirius::ast::from_duckdb(*agg_expr));

  auto agg = duckdb::make_uniq<sirius_physical_grouped_aggregate>(
    std::move(types), std::move(aggregates), std::move(groups), estimated_cardinality);
  agg->children.push_back(std::move(child));
  return agg;
}

/// The canonical activation tree:
///   GROUP_BY{k, s1, s2; SUM(v)} <- PROJ[0,1,2,5] <- JOIN(k=k) <- (scan[k,s1,s2,x], scan[k,v])
duckdb::unique_ptr<sirius::op::sirius_physical_operator> make_activation_tree(
  duckdb::JoinType join_type = duckdb::JoinType::INNER, bool two_string_keys = true)
{
  duckdb::vector<sirius::logical_type> left_types{
    bigint_type(), varchar_type(), varchar_type(), bigint_type()};
  duckdb::vector<sirius::logical_type> right_types{bigint_type(), bigint_type()};
  auto join = make_join(
    join_type, make_scan(std::move(left_types), 100), make_scan(std::move(right_types), 100), 100);
  std::vector<std::pair<uint32_t, sirius::logical_type>> proj_refs{
    {0, bigint_type()}, {1, varchar_type()}, {2, varchar_type()}, {5, bigint_type()}};
  auto proj = make_ref_projection(std::move(proj_refs), std::move(join), 100);
  std::vector<std::pair<uint32_t, sirius::logical_type>> group_refs{{0, bigint_type()},
                                                                    {1, varchar_type()}};
  if (two_string_keys) { group_refs.push_back({2, varchar_type()}); }
  return make_group_by(std::move(group_refs), /*agg_index=*/3, std::move(proj), 100);
}

sirius::operator_params test_params()
{
  sirius::operator_params params;
  params.groupby_surrogate_keys            = true;
  params.groupby_surrogate_unique_fastpath = true;
  params.groupby_surrogate_min_string_keys = 2;
  params.groupby_surrogate_min_rows        = 1;
  return params;
}

sirius_physical_grouped_aggregate& find_group_by(sirius::op::sirius_physical_operator& root)
{
  sirius::op::sirius_physical_operator* cur = &root;
  while (cur->type != SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    REQUIRE(!cur->children.empty());
    cur = cur->children[0].get();
  }
  return cur->Cast<sirius_physical_grouped_aggregate>();
}

sirius_physical_hash_join& find_join(sirius::op::sirius_physical_operator& root)
{
  sirius::op::sirius_physical_operator* cur = &root;
  while (cur->type != SiriusPhysicalOperatorType::HASH_JOIN) {
    REQUIRE(!cur->children.empty());
    cur = cur->children[0].get();
  }
  return cur->Cast<sirius_physical_hash_join>();
}

/// A merge operator in the shape produced by the rewrite for keys [k, rowid] + SUM: constructed
/// against the rewritten carriers, then given the spec and the original (restored) schema —
/// mirroring the clone-from-aggregate constructor.
std::unique_ptr<sirius_physical_grouped_aggregate_merge> make_surrogate_merge(
  sirius::logical_type key_type, std::shared_ptr<surrogate_groupby_spec> spec)
{
  duckdb::vector<sirius::logical_type> carrier_types{key_type, bigint_type(), bigint_type()};
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups;
  {
    auto ref0 =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(sirius::to_duckdb(key_type), 0ULL);
    auto ref1 =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 1ULL);
    groups.push_back(sirius::ast::from_duckdb(*ref0));
    groups.push_back(sirius::ast::from_duckdb(*ref1));
  }
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> agg_children;
  agg_children.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 2ULL));
  auto agg_function = sirius::test::MakeDummyAggregate(
    "sum", {duckdb::LogicalType::BIGINT}, duckdb::LogicalType::BIGINT);
  auto agg_expr = duckdb::make_uniq<duckdb::BoundAggregateExpression>(
    agg_function, std::move(agg_children), nullptr, nullptr, duckdb::AggregateType::NON_DISTINCT);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;
  aggregates.push_back(sirius::ast::from_duckdb(*agg_expr));

  auto merge = std::make_unique<sirius_physical_grouped_aggregate_merge>(
    std::move(carrier_types), std::move(aggregates), std::move(groups), /*estimated=*/10);
  merge->types          = spec->original_output_types;
  merge->surrogate_spec = std::move(spec);
  return merge;
}

/// One committed string source batch; returns its base.
int64_t commit_string_source(surrogate_deferral_store& store,
                             std::vector<std::string> const& values,
                             cucascade::memory::memory_space& space)
{
  auto stream = default_stream();
  auto col =
    vector_to_cudf_column<gpu_type_traits<string_tag>>(values, stream, get_resource_ref(space));
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto batch      = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                       space,
                                       stream,
                                       sirius::telemetry::batch_telemetry_info{});
  auto const base = store.reserve(
    /*is_left=*/true, batch->get_batch_id(), static_cast<cudf::size_type>(values.size()));
  store.commit(/*is_left=*/true, batch->get_batch_id(), batch->to_read_only());
  return base;
}

/// A merged-shape input batch [key, rowid BIGINT, sum BIGINT].
template <typename KeyTraits>
std::shared_ptr<cucascade::data_batch> make_merged_batch(
  std::vector<typename KeyTraits::type> const& keys,
  std::vector<int64_t> const& rowids,
  std::vector<int64_t> const& sums,
  cucascade::memory::memory_space& space)
{
  auto stream = default_stream();
  auto mr     = get_resource_ref(space);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(vector_to_cudf_column<KeyTraits>(keys, stream, mr));
  cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>(rowids, stream, mr));
  cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>(sums, stream, mr));
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                 space,
                                 stream,
                                 sirius::telemetry::batch_telemetry_info{});
}

std::shared_ptr<surrogate_groupby_spec> make_test_spec(
  std::shared_ptr<surrogate_deferral_store> store,
  sirius::logical_type key_type,
  bool unique_fastpath = true)
{
  auto spec   = std::make_shared<surrogate_groupby_spec>();
  spec->store = std::move(store);
  surrogate_groupby_spec::restore_group group;
  group.from_left         = true;
  group.rowid_key_slot    = 1;
  group.restore_key_slots = {1};
  group.source_input_cols = {0};
  group.restored_types    = {varchar_type()};
  spec->groups.push_back(std::move(group));
  spec->real_key_slots        = {0};
  spec->original_output_types = {std::move(key_type), varchar_type(), bigint_type()};
  spec->unique_fastpath       = unique_fastpath;
  return spec;
}

}  // namespace

//===--------------------------------------------------------------------------------------===//
// Deferral store: addressing and retention contract
//===--------------------------------------------------------------------------------------===//

TEST_CASE("surrogate store assigns contiguous ranges and dedupes by batch id",
          "[surrogate_groupby]")
{
  surrogate_deferral_store store;
  REQUIRE(store.reserve(true, /*batch_id=*/11, 10) == 0);
  REQUIRE(store.reserve(true, /*batch_id=*/22, 5) == 10);
  // Same batch id (a task retry, or a BUILD_PROBE build table shared by many probe tasks)
  // returns the existing range instead of burning new address space.
  REQUIRE(store.reserve(true, /*batch_id=*/11, 10) == 0);
  REQUIRE(store.reserve(true, /*batch_id=*/33, 1) == 15);
  // Sides are independent address spaces.
  REQUIRE(store.reserve(false, /*batch_id=*/11, 7) == 0);
  // Re-reserving with a different row count is a contract violation.
  REQUIRE_THROWS(store.reserve(true, /*batch_id=*/11, 9));
}

TEST_CASE("surrogate store overflow guard rejects before mutating", "[surrogate_groupby]")
{
  constexpr auto max_rows = std::numeric_limits<cudf::size_type>::max();
  surrogate_deferral_store store;
  REQUIRE(store.reserve(true, 1, max_rows - 5) == 0);
  // Would exceed int32 addressing: throws...
  REQUIRE_THROWS(store.reserve(true, 2, 10));
  // ...without having consumed any address space (check-before-mutate).
  REQUIRE(store.reserve(true, 3, 5) == max_rows - 5);
  // The deduped range is still resolvable.
  REQUIRE(store.reserve(true, 1, max_rows - 5) == 0);
}

TEST_CASE("surrogate store snapshot requires committed sources and release drops them",
          "[surrogate_groupby]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = default_stream();

  surrogate_deferral_store store;
  std::vector<int64_t> values{1, 2, 3};
  auto col =
    vector_to_cudf_column<gpu_type_traits<int64_t>>(values, stream, get_resource_ref(*space));
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto batch = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                       *space,
                                       stream,
                                       sirius::telemetry::batch_telemetry_info{});

  REQUIRE(store.reserve(true, batch->get_batch_id(), 3) == 0);
  // Reserved but not committed: the producing task has not succeeded yet.
  REQUIRE_THROWS(store.snapshot(true));
  store.commit(true, batch->get_batch_id(), batch->to_read_only());
  // Idempotent commit.
  store.commit(true, batch->get_batch_id(), batch->to_read_only());
  auto sources = store.snapshot(true);
  REQUIRE(sources.size() == 1);
  REQUIRE(sources[0].base == 0);
  REQUIRE(sources[0].rows == 3);

  auto const [count, bytes] = store.release();
  REQUIRE(count == 1);
  REQUIRE(bytes > 0);
  auto const [count2, bytes2] = store.release();
  REQUIRE(count2 == 0);
  // Committing to an unknown (never reserved) batch id is a contract violation.
  REQUIRE_THROWS(store.commit(true, /*batch_id=*/999, batch->to_read_only()));
}

//===--------------------------------------------------------------------------------------===//
// Planner pass: activation shape and decline gates
//===--------------------------------------------------------------------------------------===//

TEST_CASE("surrogate planner pass rewrites the canonical INNER-join shape", "[surrogate_groupby]")
{
  auto plan = make_activation_tree();
  sirius::planner::apply_groupby_surrogate_keys(plan, test_params());

  auto& agg = find_group_by(*plan);
  REQUIRE(agg.surrogate_spec != nullptr);
  auto const& spec = *agg.surrogate_spec;
  REQUIRE(spec.real_key_slots == std::vector<int>{0});
  REQUIRE(spec.groups.size() == 1);
  REQUIRE(spec.groups[0].from_left);
  REQUIRE(spec.groups[0].rowid_key_slot == 1);
  REQUIRE(spec.groups[0].restore_key_slots == std::vector<int>{1, 2});
  REQUIRE(spec.groups[0].source_input_cols == std::vector<cudf::size_type>{1, 2});
  // Carrier types swapped in place: rowid BIGINT at the first deferred slot, TINYINT dummy at
  // the second; the original schema is preserved on the spec.
  REQUIRE(sirius::get_cudf_type(agg.types[1]) == cudf::data_type{cudf::type_id::INT64});
  REQUIRE(sirius::get_cudf_type(agg.types[2]) == cudf::data_type{cudf::type_id::INT8});
  REQUIRE(spec.original_output_types[1].is_varchar());
  REQUIRE(spec.original_output_types[2].is_varchar());
  // Partition hashing collapses to the real key slots.
  REQUIRE(agg.get_output_grouping_indices() == std::vector<int>{0});

  auto& join = find_join(*plan);
  REQUIRE(join.surrogate_emit.has_value());
  REQUIRE(join.surrogate_emit->left.has_value());
  REQUIRE_FALSE(join.surrogate_emit->right.has_value());
  REQUIRE(join.surrogate_emit->left->rowid_out_pos == 1);
  REQUIRE(join.surrogate_emit->left->dummy_out_pos == std::vector<cudf::size_type>{2});
  REQUIRE(sirius::get_cudf_type(join.types[1]) == cudf::data_type{cudf::type_id::INT64});
  REQUIRE(sirius::get_cudf_type(join.types[2]) == cudf::data_type{cudf::type_id::INT8});
  REQUIRE(join.surrogate_emit->store == spec.store);
}

TEST_CASE("surrogate planner pass declines non-INNER joins and leaves the plan untouched",
          "[surrogate_groupby]")
{
  auto plan = make_activation_tree(duckdb::JoinType::LEFT);
  sirius::planner::apply_groupby_surrogate_keys(plan, test_params());
  auto& agg = find_group_by(*plan);
  REQUIRE(agg.surrogate_spec == nullptr);
  REQUIRE(agg.types[1].is_varchar());
  REQUIRE_FALSE(find_join(*plan).surrogate_emit.has_value());
}

TEST_CASE("surrogate planner pass declines when a second consumer reads a deferred column",
          "[surrogate_groupby]")
{
  auto plan = make_activation_tree();
  // Add a computed projection output that consumes the deferred string column (input col 1).
  auto& proj = [&]() -> sirius_physical_projection& {
    sirius::op::sirius_physical_operator* cur = plan.get();
    while (cur->type != SiriusPhysicalOperatorType::PROJECTION) {
      cur = cur->children[0].get();
    }
    return cur->Cast<sirius_physical_projection>();
  }();
  proj.select_list.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::cast{make_reference(1, varchar_type()), varchar_type(), /*try_cast=*/false}));
  proj.types.push_back(varchar_type());

  sirius::planner::apply_groupby_surrogate_keys(plan, test_params());
  REQUIRE(find_group_by(*plan).surrogate_spec == nullptr);
  REQUIRE_FALSE(find_join(*plan).surrogate_emit.has_value());
}

TEST_CASE("surrogate planner pass respects the knob and the min-string-keys gate",
          "[surrogate_groupby]")
{
  {
    auto plan                     = make_activation_tree();
    auto params                   = test_params();
    params.groupby_surrogate_keys = false;
    sirius::planner::apply_groupby_surrogate_keys(plan, params);
    REQUIRE(find_group_by(*plan).surrogate_spec == nullptr);
    REQUIRE(find_group_by(*plan).types[1].is_varchar());
  }
  {
    auto plan = make_activation_tree(duckdb::JoinType::INNER, /*two_string_keys=*/false);
    sirius::planner::apply_groupby_surrogate_keys(plan, test_params());  // min_string_keys = 2
    REQUIRE(find_group_by(*plan).surrogate_spec == nullptr);
  }
}

//===--------------------------------------------------------------------------------------===//
// MERGE_GROUP_BY finalization
//===--------------------------------------------------------------------------------------===//

TEST_CASE("surrogate merge finalize fast path restores strings without a re-group",
          "[surrogate_groupby]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  auto store = std::make_shared<surrogate_deferral_store>();
  commit_string_source(*store, {"alpha", "beta", "gamma", "delta"}, *space);
  auto merge = make_surrogate_merge(bigint_type(), make_test_spec(store, bigint_type()));

  auto b1 = make_merged_batch<gpu_type_traits<int64_t>>({1, 2}, {0, 1}, {10, 20}, *space);
  auto b2 = make_merged_batch<gpu_type_traits<int64_t>>({1, 3}, {0, 2}, {5, 7}, *space);
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs;
  inputs.push_back(std::move(b1));
  inputs.push_back(std::move(b2));
  auto outputs = merge->execute(pipelineable_operator_data(std::move(inputs)), stream);

  std::vector<std::unique_ptr<cudf::column>> expected_cols;
  expected_cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>({1, 2, 3}, stream, mr));
  expected_cols.push_back(
    vector_to_cudf_column<gpu_type_traits<string_tag>>({"alpha", "beta", "gamma"}, stream, mr));
  expected_cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>({15, 20, 7}, stream, mr));
  cudf::table expected(std::move(expected_cols));

  auto const& batches =
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches();
  REQUIRE(batches.size() == 1);
  REQUIRE(sirius::test::expect_data_batch_equivalent_to_table(batches[0],
                                                              expected.view(),
                                                              /*sort=*/true));

  // The finalize hook releases the retained sources exactly once.
  merge->on_finalize_operator();
  auto const [count, bytes] = store->release();
  REQUIRE(count == 0);
}

TEST_CASE("surrogate merge finalize re-groups duplicate full tuples (conservative path)",
          "[surrogate_groupby]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  auto store = std::make_shared<surrogate_deferral_store>();
  // Two DISTINCT source rows carrying an IDENTICAL string: the wrong-results class the
  // conservative path exists for.
  commit_string_source(*store, {"x", "x"}, *space);
  auto merge = make_surrogate_merge(bigint_type(), make_test_spec(store, bigint_type()));

  auto b1 = make_merged_batch<gpu_type_traits<int64_t>>({1}, {0}, {10}, *space);
  auto b2 = make_merged_batch<gpu_type_traits<int64_t>>({1}, {1}, {5}, *space);
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs;
  inputs.push_back(std::move(b1));
  inputs.push_back(std::move(b2));
  auto outputs = merge->execute(pipelineable_operator_data(std::move(inputs)), stream);

  std::vector<std::unique_ptr<cudf::column>> expected_cols;
  expected_cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>({1}, stream, mr));
  expected_cols.push_back(vector_to_cudf_column<gpu_type_traits<string_tag>>({"x"}, stream, mr));
  expected_cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>({15}, stream, mr));
  cudf::table expected(std::move(expected_cols));

  auto const& batches =
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches();
  REQUIRE(batches.size() == 1);
  REQUIRE(sirius::test::expect_data_batch_equivalent_to_table(batches[0],
                                                              expected.view(),
                                                              /*sort=*/true));
}

TEST_CASE("surrogate merge finalize declines the fast path on floating-point real keys",
          "[surrogate_groupby]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();

  auto store = std::make_shared<surrogate_deferral_store>();
  commit_string_source(*store, {"x", "x"}, *space);
  auto merge = make_surrogate_merge(double_type(), make_test_spec(store, double_type()));

  // Two NaN-keyed rows with distinct rowids but identical strings: SQL grouping semantics
  // (all NaNs are one group) require ONE output row, which only the conservative re-group can
  // produce — the distinct-count proof must not be consulted for floating-point keys.
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  auto b1        = make_merged_batch<gpu_type_traits<double>>({nan}, {0}, {10}, *space);
  auto b2        = make_merged_batch<gpu_type_traits<double>>({nan}, {1}, {5}, *space);
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs;
  inputs.push_back(std::move(b1));
  inputs.push_back(std::move(b2));
  auto outputs = merge->execute(pipelineable_operator_data(std::move(inputs)), stream);

  auto const& batches =
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto const view = sirius::get_cudf_table_view(*batches[0]);
  REQUIRE(view.num_rows() == 1);
  REQUIRE(view.num_columns() == 3);
}

TEST_CASE("surrogate merge finalize gathers across multiple source base ranges",
          "[surrogate_groupby]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  auto store = std::make_shared<surrogate_deferral_store>();
  REQUIRE(commit_string_source(*store, {"a", "b"}, *space) == 0);
  REQUIRE(commit_string_source(*store, {"c", "d"}, *space) == 2);
  auto merge = make_surrogate_merge(bigint_type(), make_test_spec(store, bigint_type()));

  // Single-batch merge input (exercises the clone -> finalize path); rowids straddle the two
  // source ranges: 1 -> "b" (first source), 2 -> "c" (second source).
  auto b1 = make_merged_batch<gpu_type_traits<int64_t>>({1, 2}, {1, 2}, {10, 20}, *space);
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs;
  inputs.push_back(std::move(b1));
  auto outputs = merge->execute(pipelineable_operator_data(std::move(inputs)), stream);

  std::vector<std::unique_ptr<cudf::column>> expected_cols;
  expected_cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>({1, 2}, stream, mr));
  expected_cols.push_back(
    vector_to_cudf_column<gpu_type_traits<string_tag>>({"b", "c"}, stream, mr));
  expected_cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>({10, 20}, stream, mr));
  cudf::table expected(std::move(expected_cols));

  auto const& batches =
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches();
  REQUIRE(batches.size() == 1);
  REQUIRE(sirius::test::expect_data_batch_equivalent_to_table(batches[0],
                                                              expected.view(),
                                                              /*sort=*/true));
}
