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
 * @file test_surrogate_groupby_pass.cpp
 * @brief Contract tests for the surrogate-key group-by planner pass
 *        (`sirius::planner::apply_groupby_surrogate_keys`) over hand-built physical operator
 *        trees: the canonical activation shape (installed emit/restore plans, patched carrier
 *        types, real-key partition indices) and the decline gates for non-INNER joins, second
 *        consumers of a deferred column, the enable knob, and the minimum-string-keys gate.
 */

#include "cudf/cudf_utils.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "op/groupby_surrogate_deferral.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "operator/aggregate/aggregate_test_utils.hpp"
#include "planner/sirius_plan_surrogate_groupby.hpp"

#include <catch.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::op;

namespace {

sirius::logical_type bigint_type() { return sirius::logical_type::make(sirius::type_id::BIGINT); }
sirius::logical_type varchar_type() { return sirius::logical_type::make(sirius::type_id::VARCHAR); }

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

}  // namespace

TEST_CASE("surrogate planner pass rewrites the canonical INNER-join shape", "[surrogate_groupby]")
{
  auto plan = make_activation_tree();
  sirius::planner::apply_groupby_surrogate_keys(plan, test_params());

  auto& agg = find_group_by(*plan);
  REQUIRE(agg.surrogate_restore() != nullptr);
  auto const& restore = *agg.surrogate_restore();
  REQUIRE(restore.real_key_slots() == std::vector<int>{0});
  REQUIRE(restore.groups().size() == 1);
  auto const& group = restore.groups()[0];
  REQUIRE(group.side() == join_side::left);
  REQUIRE(group.rowid_key_slot() == 1);
  REQUIRE(group.keys().size() == 2);
  REQUIRE(group.keys()[0].key_slot == 1);
  REQUIRE(group.keys()[1].key_slot == 2);
  REQUIRE(group.keys()[0].source_col == 1);
  REQUIRE(group.keys()[1].source_col == 2);
  // Carrier types swapped in place: rowid BIGINT at the first deferred slot, TINYINT dummy at
  // the second; the original schema is preserved on the restore plan.
  REQUIRE(sirius::get_cudf_type(agg.types[1]) == cudf::data_type{cudf::type_id::INT64});
  REQUIRE(sirius::get_cudf_type(agg.types[2]) == cudf::data_type{cudf::type_id::INT8});
  REQUIRE(restore.original_output_types()[1].is_varchar());
  REQUIRE(restore.original_output_types()[2].is_varchar());
  // Partition hashing collapses to the real key slots.
  REQUIRE(agg.get_output_grouping_indices() == std::vector<int>{0});

  auto& join = find_join(*plan);
  REQUIRE(join.surrogate_emit().has_value());
  auto const& emit = *join.surrogate_emit();
  REQUIRE(emit.side(join_side::left).has_value());
  REQUIRE_FALSE(emit.side(join_side::right).has_value());
  REQUIRE(emit.side(join_side::left)->rowid_out_pos() == 1);
  REQUIRE(emit.side(join_side::left)->dummy_out_pos() == std::vector<cudf::size_type>{2});
  REQUIRE(sirius::get_cudf_type(join.types[1]) == cudf::data_type{cudf::type_id::INT64});
  REQUIRE(sirius::get_cudf_type(join.types[2]) == cudf::data_type{cudf::type_id::INT8});
  REQUIRE(&emit.store() == restore.store().get());
}

TEST_CASE("surrogate planner pass declines non-INNER joins and leaves the plan untouched",
          "[surrogate_groupby]")
{
  auto plan = make_activation_tree(duckdb::JoinType::LEFT);
  sirius::planner::apply_groupby_surrogate_keys(plan, test_params());
  auto& agg = find_group_by(*plan);
  REQUIRE(agg.surrogate_restore() == nullptr);
  REQUIRE(agg.types[1].is_varchar());
  REQUIRE_FALSE(find_join(*plan).surrogate_emit().has_value());
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
  REQUIRE(find_group_by(*plan).surrogate_restore() == nullptr);
  REQUIRE_FALSE(find_join(*plan).surrogate_emit().has_value());
}

TEST_CASE("surrogate planner pass respects the knob and the min-string-keys gate",
          "[surrogate_groupby]")
{
  {
    auto plan                     = make_activation_tree();
    auto params                   = test_params();
    params.groupby_surrogate_keys = false;
    sirius::planner::apply_groupby_surrogate_keys(plan, params);
    REQUIRE(find_group_by(*plan).surrogate_restore() == nullptr);
    REQUIRE(find_group_by(*plan).types[1].is_varchar());
  }
  {
    auto plan = make_activation_tree(duckdb::JoinType::INNER, /*two_string_keys=*/false);
    sirius::planner::apply_groupby_surrogate_keys(plan, test_params());  // min_string_keys = 2
    REQUIRE(find_group_by(*plan).surrogate_restore() == nullptr);
  }
}
