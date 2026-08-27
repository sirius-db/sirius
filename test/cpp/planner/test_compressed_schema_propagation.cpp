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
 * @file test_compressed_schema_propagation.cpp
 * @brief Contract tests for the compressed-materialization planner passes
 *        (`propagate_compressed_schema`, `restore_native_schema`,
 *        `prune_immediate_scan_restores`, `apply_compressed_schema_passes`) over hand-built
 *        physical operator trees: hash-join key restoration and per-join-type output-layout
 *        mapping, native boundaries for other operators, dynamic-filter scans, root restores,
 *        DELIM_JOIN sub-tree restoration, zero-benefit restore pruning (including the
 *        aliased-duplicate guard), and the unmappable-type full clear.
 */

#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/join_condition.hpp"
#include "expression/value.hpp"
#include "helper/logical_type.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_dense_count_join.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/sirius_plan_compressed_schema.hpp"

#include <cudf/cudf_utils.hpp>
#include <cudf/types.hpp>

#include <catch.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>
#include <utils/dense_count_join_test_builder.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

namespace {
using sirius::test::make_dense_count_join;

constexpr cudf::data_type k_int8{cudf::type_id::INT8};
constexpr cudf::data_type k_int16{cudf::type_id::INT16};
constexpr cudf::data_type k_int32{cudf::type_id::INT32};
constexpr cudf::data_type k_int64{cudf::type_id::INT64};
constexpr cudf::data_type k_bool8{cudf::type_id::BOOL8};

sirius::logical_type integer_type() { return sirius::logical_type::make(sirius::type_id::INTEGER); }

duckdb::vector<sirius::logical_type> integer_types(std::size_t count)
{
  duckdb::vector<sirius::logical_type> types;
  types.reserve(count);
  for (std::size_t i = 0; i < count; i++) {
    types.push_back(integer_type());
  }
  return types;
}

std::unique_ptr<sirius::ast::node> make_reference(uint32_t column_index)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::reference{column_index, integer_type()});
}

// A restore cast as the passes emit it: an untyped reference under a cast to the native
// logical type, tagged with the carrier_restore provenance.
std::unique_ptr<sirius::ast::node> make_restore_cast(uint32_t column_index)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::cast{std::make_unique<sirius::ast::node>(sirius::ast::reference{column_index}),
                      integer_type(),
                      /*try_cast=*/false,
                      sirius::ast::cast_kind::carrier_restore});
}

// A user-written cast of the same shape as a restore cast, carrying the default semantic
// provenance.
std::unique_ptr<sirius::ast::node> make_semantic_cast(uint32_t column_index)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::cast{std::make_unique<sirius::ast::node>(sirius::ast::reference{column_index}),
                      integer_type(),
                      /*try_cast=*/false});
}

// A TABLE_SCAN leaf over INTEGER columns with @p physical installed as its sidecar (one entry
// per column, or empty for native). @p projection_ids overrides the identity output mapping:
// output i then reads column_ids position projection_ids[i], and column_ids covers enough
// positions for the largest one.
duckdb::unique_ptr<sirius::op::sirius_physical_table_scan> make_scan(
  std::size_t column_count,
  std::vector<cudf::data_type> physical      = {},
  duckdb::vector<std::size_t> projection_ids = {})
{
  if (projection_ids.empty()) {
    for (std::size_t i = 0; i < column_count; i++) {
      projection_ids.push_back(i);
    }
  }
  REQUIRE(projection_ids.size() == column_count);
  std::size_t ids_count = 0;
  for (auto const ids_position : projection_ids) {
    ids_count = std::max(ids_count, ids_position + 1);
  }
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<std::string> names;
  for (std::size_t i = 0; i < ids_count; i++) {
    column_ids.emplace_back(i);
    names.push_back("c" + std::to_string(i));
  }
  duckdb::TableFunction function("test_scan", {}, nullptr, nullptr);
  auto scan = duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(
    integer_types(column_count),
    std::move(function),
    /*bind_data=*/nullptr,
    integer_types(ids_count),
    std::move(column_ids),
    std::move(projection_ids),
    std::move(names),
    duckdb::make_uniq<duckdb::TableFilterSet>(),
    /*estimated_cardinality=*/1,
    duckdb::ExtraOperatorInfo(),
    duckdb::vector<duckdb::Value>{},
    duckdb::virtual_column_map_t{});
  scan->set_physical_types(std::move(physical));
  return scan;
}

// A pure-reference projection forwarding @p input_indices of @p child, in output order.
duckdb::unique_ptr<sirius::op::sirius_physical_projection> make_pure_reference_projection(
  std::vector<uint32_t> input_indices, duckdb::unique_ptr<sirius_physical_operator> child)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
  for (auto const input_idx : input_indices) {
    select_list.push_back(make_reference(input_idx));
  }
  auto projection = duckdb::make_uniq<sirius::op::sirius_physical_projection>(
    integer_types(input_indices.size()), std::move(select_list), child->estimated_cardinality);
  projection->children.push_back(std::move(child));
  return projection;
}

// A hash join on column 0 of both inputs, with identity output maps over each side (the
// convenience constructor derives them from the children's types). @p output_types is the
// join's logical output schema for the tested join type.
duckdb::unique_ptr<sirius::op::sirius_physical_hash_join> make_hash_join(
  duckdb::JoinType join_type,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  duckdb::vector<duckdb::LogicalType> output_types)
{
  duckdb::LogicalDummyScan stub(0);
  stub.types = std::move(output_types);
  duckdb::vector<sirius::join_condition> conditions;
  sirius::join_condition condition;
  condition.left  = make_reference(0);
  condition.right = make_reference(0);
  conditions.push_back(std::move(condition));
  return duckdb::make_uniq<sirius::op::sirius_physical_hash_join>(
    stub,
    std::move(left),
    std::move(right),
    std::move(conditions),
    join_type,
    /*left_projection_map=*/duckdb::vector<std::size_t>{},
    /*right_projection_map=*/duckdb::vector<std::size_t>{},
    /*delim_types=*/duckdb::vector<sirius::logical_type>{},
    /*estimated_cardinality=*/1);
}

duckdb::vector<duckdb::LogicalType> duckdb_integer_types(std::size_t count)
{
  duckdb::vector<duckdb::LogicalType> types;
  for (std::size_t i = 0; i < count; i++) {
    types.push_back(duckdb::LogicalType::INTEGER);
  }
  return types;
}

// An aggregate expression over child column @p input_idx with a BIGINT return type.
std::unique_ptr<sirius::ast::node> make_aggregate(uint32_t input_idx, sirius::aggregate_id function)
{
  std::vector<std::unique_ptr<sirius::ast::node>> arguments;
  arguments.push_back(make_reference(input_idx));
  return std::make_unique<sirius::ast::node>(
    sirius::ast::aggregate{function,
                           std::move(arguments),
                           sirius::logical_type::make(sirius::type_id::BIGINT),
                           /*distinct=*/false});
}

// A HASH_GROUP_BY over @p child grouping on @p group_columns with one @p function per entry of
// @p aggregate_input_columns; output layout is [keys..., aggregates...].
duckdb::unique_ptr<sirius::op::sirius_physical_grouped_aggregate> make_grouped_aggregate(
  std::vector<uint32_t> group_columns,
  std::vector<uint32_t> aggregate_input_columns,
  duckdb::unique_ptr<sirius_physical_operator> child,
  duckdb::vector<duckdb::GroupingSet> grouping_sets = {},
  sirius::aggregate_id function                     = sirius::aggregate_id::sum)
{
  duckdb::vector<sirius::logical_type> output_types;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups;
  for (auto const group_idx : group_columns) {
    output_types.push_back(integer_type());
    groups.push_back(make_reference(group_idx));
  }
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions;
  for (auto const input_idx : aggregate_input_columns) {
    output_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    expressions.push_back(make_aggregate(input_idx, function));
  }
  auto aggregate = duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate>(
    std::move(output_types),
    std::move(expressions),
    std::move(groups),
    std::move(grouping_sets),
    duckdb::vector<duckdb::unsafe_vector<std::size_t>>{},
    /*estimated_cardinality=*/1,
    duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
    duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES);
  aggregate->children.push_back(std::move(child));
  return aggregate;
}

void require_restore_projection_at(sirius_physical_operator const& op,
                                   std::size_t restored_idx,
                                   std::vector<cudf::data_type> const& expected)
{
  REQUIRE(op.type == SiriusPhysicalOperatorType::PROJECTION);
  auto const& projection = op.Cast<sirius::op::sirius_physical_projection>();
  REQUIRE(projection.select_list.size() == expected.size());
  for (std::size_t output_idx = 0; output_idx < projection.select_list.size(); ++output_idx) {
    if (output_idx == restored_idx) {
      REQUIRE(projection.select_list[output_idx]->holds<sirius::ast::cast>());
    } else {
      REQUIRE(projection.select_list[output_idx]->holds<sirius::ast::reference>());
    }
  }
  REQUIRE(op.get_physical_types() == expected);
}

void require_key_restore_projection(sirius_physical_operator const& op,
                                    std::vector<cudf::data_type> const& expected)
{
  require_restore_projection_at(op, 0, expected);
}

}  // namespace

TEST_CASE("compressed_schema_propagation - hash join restores keys and maps payload sidecars",
          "[compressed_schema_propagation]")
{
  // Both scans join on column 0 (narrowed) and carry a narrowed payload in column 1, with
  // side-distinct payload carriers (INT8 left, INT16 right) so the assertions prove which
  // side each join-output sidecar entry came from.
  auto make_join = [](duckdb::JoinType join_type, duckdb::vector<duckdb::LogicalType> out_types) {
    return make_hash_join(join_type,
                          make_scan(2, {k_int8, k_int8}),
                          make_scan(2, {k_int8, k_int16}),
                          std::move(out_types));
  };
  std::vector<cudf::data_type> const restored_left{k_int32, k_int8};
  std::vector<cudf::data_type> const restored_right{k_int32, k_int16};

  SECTION("INNER emits lhs then rhs payload carriers")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_join(duckdb::JoinType::INNER, duckdb_integer_types(4));
    sirius::planner::propagate_compressed_schema(plan);
    require_key_restore_projection(*plan->children[0], restored_left);
    require_key_restore_projection(*plan->children[1], restored_right);
    REQUIRE(plan->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int8, k_int32, k_int16});
  }

  SECTION("LEFT emits lhs then rhs payload carriers")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_join(duckdb::JoinType::LEFT, duckdb_integer_types(4));
    sirius::planner::propagate_compressed_schema(plan);
    require_key_restore_projection(*plan->children[0], restored_left);
    require_key_restore_projection(*plan->children[1], restored_right);
    REQUIRE(plan->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int8, k_int32, k_int16});
  }

  SECTION("SEMI emits only lhs carriers but restores keys on both sides")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_join(duckdb::JoinType::SEMI, duckdb_integer_types(2));
    sirius::planner::propagate_compressed_schema(plan);
    require_key_restore_projection(*plan->children[0], restored_left);
    require_key_restore_projection(*plan->children[1], restored_right);
    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int32, k_int8});
  }

  SECTION("ANTI emits only lhs carriers but restores keys on both sides")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_join(duckdb::JoinType::ANTI, duckdb_integer_types(2));
    sirius::planner::propagate_compressed_schema(plan);
    require_key_restore_projection(*plan->children[0], restored_left);
    require_key_restore_projection(*plan->children[1], restored_right);
    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int32, k_int8});
  }

  SECTION("MARK emits lhs carriers plus a native boolean mark column")
  {
    auto out_types = duckdb_integer_types(2);
    out_types.push_back(duckdb::LogicalType::BOOLEAN);
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_join(duckdb::JoinType::MARK, std::move(out_types));
    sirius::planner::propagate_compressed_schema(plan);
    require_key_restore_projection(*plan->children[0], restored_left);
    require_key_restore_projection(*plan->children[1], restored_right);
    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int32, k_int8, k_bool8});
  }

  SECTION("RIGHT_SEMI emits only rhs carriers")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_join(duckdb::JoinType::RIGHT_SEMI, duckdb_integer_types(2));
    sirius::planner::propagate_compressed_schema(plan);
    require_key_restore_projection(*plan->children[0], restored_left);
    require_key_restore_projection(*plan->children[1], restored_right);
    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int32, k_int16});
  }

  SECTION("a dynamic-filter-target key needs no restore projection")
  {
    // The dynamic-filter guard already forced the probe key native at the scan, so the join's key
    // restore finds nothing to cast there and inserts no projection; the probe payload maps
    // through narrow.
    auto probe                    = make_scan(2, {k_int8, k_int8});
    probe->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    probe->sirius_dynamic_filters->register_producer({0});
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_hash_join(duckdb::JoinType::INNER,
                     std::move(probe),
                     make_scan(2, {k_int8, k_int16}),
                     duckdb_integer_types(4));

    sirius::planner::propagate_compressed_schema(plan);

    REQUIRE(plan->children[0]->type == SiriusPhysicalOperatorType::TABLE_SCAN);
    REQUIRE(plan->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int8});
    require_key_restore_projection(*plan->children[1], restored_right);
    REQUIRE(plan->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int8, k_int32, k_int16});
  }
}

TEST_CASE("compressed_schema_propagation - dense count restores only keys and emits native",
          "[compressed_schema_propagation]")
{
  duckdb::unique_ptr<sirius_physical_operator> plan = make_dense_count_join(
    /*preserved_key_idx=*/1,
    /*counted_key_idx=*/0,
    /*counted_value_idx=*/1,
    make_scan(3, {k_int8, k_int16, k_int8}),
    make_scan(3, {k_int8, k_int16, k_int8}));

  sirius::planner::propagate_compressed_schema(plan);

  REQUIRE(!plan->has_physical_overrides());
  require_restore_projection_at(*plan->children[0], 1, {k_int8, k_int32, k_int8});
  require_restore_projection_at(*plan->children[1], 0, {k_int32, k_int16, k_int8});
}

TEST_CASE("compressed_schema_propagation - native boundaries restore children fully",
          "[compressed_schema_propagation]")
{
  // HASH_GROUP_BY is absent here: it has a dedicated propagation case (narrow group keys) whose
  // fallback-to-native shapes are covered by the grouped-aggregation test below with the real
  // operator class.
  auto const boundary_type = GENERATE(SiriusPhysicalOperatorType::NESTED_LOOP_JOIN,
                                      SiriusPhysicalOperatorType::ORDER_BY,
                                      SiriusPhysicalOperatorType::TOP_N);

  auto boundary = duckdb::make_uniq<sirius_physical_operator>(
    boundary_type, integer_types(2), /*estimated_cardinality=*/1);
  boundary->children.push_back(make_scan(2, {k_int8, k_int8}));
  duckdb::unique_ptr<sirius_physical_operator> plan = std::move(boundary);

  sirius::planner::propagate_compressed_schema(plan);

  REQUIRE(!plan->has_physical_overrides());
  auto const& restored = *plan->children[0];
  REQUIRE(restored.type == SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(!restored.has_physical_overrides());
  auto const& projection = restored.Cast<sirius::op::sirius_physical_projection>();
  REQUIRE(projection.select_list[0]->holds<sirius::ast::cast>());
  REQUIRE(projection.select_list[1]->holds<sirius::ast::cast>());
  REQUIRE(restored.children[0]->type == SiriusPhysicalOperatorType::TABLE_SCAN);
}

TEST_CASE("compressed_schema_propagation - grouped aggregation keeps narrow group keys",
          "[compressed_schema_propagation]")
{
  SECTION("group keys stay narrow, aggregate inputs are restored")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_grouped_aggregate({0}, {1}, make_scan(2, {k_int8, k_int8}));

    sirius::planner::propagate_compressed_schema(plan);

    // The child restore casts only the SUM input (column 1); the key keeps a bare reference.
    auto const& restored = *plan->children[0];
    REQUIRE(restored.type == SiriusPhysicalOperatorType::PROJECTION);
    auto const& projection = restored.Cast<sirius::op::sirius_physical_projection>();
    REQUIRE(projection.select_list[0]->holds<sirius::ast::reference>());
    REQUIRE(projection.select_list[1]->holds<sirius::ast::cast>());
    REQUIRE(restored.get_physical_types() == std::vector<cudf::data_type>{k_int8, k_int32});
    // The aggregate output keeps the narrow key carrier; the SUM output is native.
    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int8, k_int64});
  }

  SECTION("unused payload columns remain narrow")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_grouped_aggregate({0}, {1}, make_scan(3, {k_int8, k_int8, k_int8}));

    sirius::planner::propagate_compressed_schema(plan);

    auto const& restored = *plan->children[0];
    REQUIRE(restored.type == SiriusPhysicalOperatorType::PROJECTION);
    auto const& projection = restored.Cast<sirius::op::sirius_physical_projection>();
    REQUIRE(projection.select_list[0]->holds<sirius::ast::reference>());
    REQUIRE(projection.select_list[1]->holds<sirius::ast::cast>());
    REQUIRE(projection.select_list[2]->holds<sirius::ast::reference>());
    REQUIRE(restored.get_physical_types() == std::vector<cudf::data_type>{k_int8, k_int32, k_int8});
    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int8, k_int64});
  }

  SECTION("COUNT_VALID leaves a counted group key narrow")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_grouped_aggregate({0}, {0}, make_scan(1, {k_int8}), {}, sirius::aggregate_id::count);

    sirius::planner::propagate_compressed_schema(plan);

    REQUIRE(plan->children[0]->type == SiriusPhysicalOperatorType::TABLE_SCAN);
    REQUIRE(plan->children[0]->get_physical_types() == std::vector<cudf::data_type>{k_int8});
    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int8, k_int64});
  }

  SECTION("a column used by a value-sensitive aggregate goes native")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_grouped_aggregate({0}, {0}, make_scan(1, {k_int8}));

    sirius::planner::propagate_compressed_schema(plan);

    auto const& restored = *plan->children[0];
    REQUIRE(restored.type == SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(restored.Cast<sirius::op::sirius_physical_projection>()
              .select_list[0]
              ->holds<sirius::ast::cast>());
    REQUIRE(!restored.has_physical_overrides());
    REQUIRE(!plan->has_physical_overrides());
  }

  SECTION("grouping sets or grouping functions fall back to the native boundary")
  {
    duckdb::vector<duckdb::GroupingSet> grouping_sets;
    grouping_sets.push_back(duckdb::GroupingSet{0});
    grouping_sets.push_back(duckdb::GroupingSet{});
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_grouped_aggregate({0}, {1}, make_scan(2, {k_int8, k_int8}), std::move(grouping_sets));

    sirius::planner::propagate_compressed_schema(plan);

    auto const& restored = *plan->children[0];
    REQUIRE(restored.type == SiriusPhysicalOperatorType::PROJECTION);
    auto const& projection = restored.Cast<sirius::op::sirius_physical_projection>();
    REQUIRE(projection.select_list[0]->holds<sirius::ast::cast>());
    REQUIRE(projection.select_list[1]->holds<sirius::ast::cast>());
    REQUIRE(!restored.has_physical_overrides());
    REQUIRE(!plan->has_physical_overrides());
  }

  SECTION("native group keys leave no sidecar")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_grouped_aggregate({0}, {1}, make_scan(2));

    sirius::planner::propagate_compressed_schema(plan);

    REQUIRE(plan->children[0]->type == SiriusPhysicalOperatorType::TABLE_SCAN);
    REQUIRE(!plan->children[0]->has_physical_overrides());
    REQUIRE(!plan->has_physical_overrides());
  }
}

TEST_CASE("compressed_schema_propagation - dynamic-filter targets clear only their columns",
          "[compressed_schema_propagation]")
{
  SECTION("a scoped producer marks only target columns native")
  {
    auto scan                    = make_scan(2, {k_int8, k_int8});
    scan->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    scan->sirius_dynamic_filters->register_producer({0});
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(scan);

    sirius::planner::propagate_compressed_schema(plan);

    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int32, k_int8});
  }

  SECTION("targets covering every narrow column drop the sidecar")
  {
    auto scan                    = make_scan(2, {k_int8, k_int8});
    scan->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    scan->sirius_dynamic_filters->register_producer({0, 1});
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(scan);

    sirius::planner::propagate_compressed_schema(plan);

    REQUIRE(!plan->has_physical_overrides());
  }

  SECTION("an unscoped producer clears the whole sidecar")
  {
    auto scan                    = make_scan(2, {k_int8, k_int8});
    scan->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    scan->sirius_dynamic_filters->register_producer({});
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(scan);

    sirius::planner::propagate_compressed_schema(plan);

    REQUIRE(!plan->has_physical_overrides());
  }

  SECTION("a producer-less filter set keeps the sidecar")
  {
    auto scan                    = make_scan(2, {k_int8, k_int8});
    scan->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(scan);

    sirius::planner::propagate_compressed_schema(plan);

    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int8, k_int8});
  }

  SECTION("targets are output positions even when projection_ids indirect the scan")
  {
    // Planned targets arrive in the scan's output space -- the channel's push, store, and lookup
    // coordinate -- so the pass must not translate through projection_ids: targeting output 0
    // flips it native even though that output reads column_ids position 2.
    auto scan = make_scan(2, {k_int8, k_int8}, duckdb::vector<std::size_t>{2, 0});
    scan->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    scan->sirius_dynamic_filters->register_producer({0});
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(scan);

    sirius::planner::propagate_compressed_schema(plan);

    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int32, k_int8});
  }
}

TEST_CASE("compressed_schema_propagation - root restore inserts a projection only when narrowed",
          "[compressed_schema_propagation]")
{
  SECTION("a narrowed root gains a restore projection")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan = make_scan(2, {k_int8, k_int32});
    sirius::planner::propagate_compressed_schema(plan);
    sirius::planner::restore_native_schema(plan);

    REQUIRE(plan->type == SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(!plan->has_physical_overrides());
    auto const& projection = plan->Cast<sirius::op::sirius_physical_projection>();
    REQUIRE(projection.select_list[0]->holds<sirius::ast::cast>());
    REQUIRE(projection.select_list[1]->holds<sirius::ast::reference>());
    REQUIRE(plan->children[0]->type == SiriusPhysicalOperatorType::TABLE_SCAN);
  }

  SECTION("an all-native root is left untouched")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan = make_scan(2);
    sirius::planner::propagate_compressed_schema(plan);
    sirius::planner::restore_native_schema(plan);

    REQUIRE(plan->type == SiriusPhysicalOperatorType::TABLE_SCAN);
  }
}

TEST_CASE("compressed_schema_propagation - DELIM_JOIN sub-trees are restored native",
          "[compressed_schema_propagation]")
{
  auto join_sub = make_pure_reference_projection({0}, make_scan(1, {k_int8}));
  auto delim    = duckdb::make_uniq<sirius::op::sirius_physical_delim_join>(
    SiriusPhysicalOperatorType::LEFT_DELIM_JOIN,
    integer_types(1),
    std::move(join_sub),
    duckdb::vector<duckdb::const_reference<sirius_physical_operator>>{},
    /*estimated_cardinality=*/1,
    duckdb::optional_idx());
  delim->distinct_root = make_scan(1, {k_int8});
  delim->children.push_back(make_scan(1, {k_int8}));
  duckdb::unique_ptr<sirius_physical_operator> plan = std::move(delim);

  sirius::planner::propagate_compressed_schema(plan);

  auto const& delim_ref = plan->Cast<sirius::op::sirius_physical_delim_join>();
  // The join sub-root advertises native output, with a restore projection inserted over its
  // still-narrowed scan child.
  REQUIRE(!delim_ref.join->has_physical_overrides());
  auto const& sub_child = *delim_ref.join->children[0];
  REQUIRE(sub_child.type == SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(!sub_child.has_physical_overrides());
  REQUIRE(sub_child.Cast<sirius::op::sirius_physical_projection>()
            .select_list[0]
            ->holds<sirius::ast::cast>());
  // A scan-leaf sub-root is forced native by clearing its sidecar.
  REQUIRE(delim_ref.distinct_root->type == SiriusPhysicalOperatorType::TABLE_SCAN);
  REQUIRE(!delim_ref.distinct_root->has_physical_overrides());
  // The DELIM_JOIN itself is a conservative boundary: children restored, own sidecar empty.
  REQUIRE(!plan->has_physical_overrides());
  REQUIRE(plan->children[0]->type == SiriusPhysicalOperatorType::PROJECTION);
}

TEST_CASE("compressed_schema_propagation - sidecar fast path inspects DELIM_JOIN sub-trees",
          "[compressed_schema_propagation]")
{
  auto delim = duckdb::make_uniq<sirius::op::sirius_physical_delim_join>(
    SiriusPhysicalOperatorType::LEFT_DELIM_JOIN,
    integer_types(1),
    make_scan(1, {k_int8}),
    duckdb::vector<duckdb::const_reference<sirius_physical_operator>>{},
    /*estimated_cardinality=*/1,
    duckdb::optional_idx());
  delim->distinct_root = make_scan(1);
  delim->children.push_back(make_scan(1));
  duckdb::unique_ptr<sirius_physical_operator> plan = std::move(delim);

  sirius::planner::apply_compressed_schema_passes(plan);

  auto const& delim_ref = plan->Cast<sirius::op::sirius_physical_delim_join>();
  REQUIRE(delim_ref.join->type == SiriusPhysicalOperatorType::TABLE_SCAN);
  REQUIRE(!delim_ref.join->has_physical_overrides());
}

TEST_CASE("compressed_schema_propagation - pruning removes only zero-benefit restores",
          "[compressed_schema_propagation]")
{
  SECTION("a restore directly above the scan is pruned and the identity projection removed")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan = make_scan(2, {k_int8, k_int32});
    sirius::planner::restore_native_schema(plan);
    sirius::planner::prune_immediate_scan_restores(plan);

    REQUIRE(plan->type == SiriusPhysicalOperatorType::TABLE_SCAN);
    REQUIRE(!plan->has_physical_overrides());
  }

  SECTION("a restore across a pure-reference chain is pruned with chain sidecars re-derived")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_pure_reference_projection({1, 0}, make_scan(2, {k_int8, k_int8}));
    sirius::planner::propagate_compressed_schema(plan);
    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int8, k_int8});
    sirius::planner::restore_native_schema(plan);
    sirius::planner::prune_immediate_scan_restores(plan);

    // The identity restore was removed; the surviving root is the chain projection, now native.
    REQUIRE(plan->type == SiriusPhysicalOperatorType::PROJECTION);
    auto const& chain = plan->Cast<sirius::op::sirius_physical_projection>();
    REQUIRE(chain.select_list[0]->get<sirius::ast::reference>().column_index == 1);
    REQUIRE(!plan->has_physical_overrides());
    REQUIRE(plan->children[0]->type == SiriusPhysicalOperatorType::TABLE_SCAN);
    REQUIRE(!plan->children[0]->has_physical_overrides());
  }

  SECTION("a restore separated from the scan by a filter keeps the narrowing")
  {
    auto filter = duckdb::make_uniq<sirius::op::sirius_physical_filter>(
      integer_types(1),
      std::make_unique<sirius::ast::node>(sirius::ast::constant{
        sirius::value{true}, sirius::logical_type::make(sirius::type_id::BOOLEAN)}),
      /*estimated_cardinality=*/1);
    filter->children.push_back(make_scan(1, {k_int8}));
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(filter);
    sirius::planner::propagate_compressed_schema(plan);
    sirius::planner::restore_native_schema(plan);
    sirius::planner::prune_immediate_scan_restores(plan);

    REQUIRE(plan->type == SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(plan->Cast<sirius::op::sirius_physical_projection>()
              .select_list[0]
              ->holds<sirius::ast::cast>());
    auto const& filter_op = *plan->children[0];
    REQUIRE(filter_op.type == SiriusPhysicalOperatorType::FILTER);
    REQUIRE(filter_op.get_physical_types() == std::vector<cudf::data_type>{k_int8});
    REQUIRE(filter_op.children[0]->get_physical_types() == std::vector<cudf::data_type>{k_int8});
  }

  SECTION("a scan column forwarded by a bare-reference duplicate is not pruned")
  {
    // The chain duplicates the narrowed scan column; the restore projection casts one
    // duplicate and forwards the other. Pruning the cast would flip the forwarded output
    // native and invalidate every ancestor sidecar derived from this projection's schema.
    auto chain = make_pure_reference_projection({0, 0}, make_scan(1, {k_int8}));
    chain->set_physical_types({k_int8, k_int8});
    duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
    select_list.push_back(make_restore_cast(0));
    select_list.push_back(make_reference(1));
    auto restore = duckdb::make_uniq<sirius::op::sirius_physical_projection>(
      integer_types(2), std::move(select_list), /*estimated_cardinality=*/1);
    restore->children.push_back(std::move(chain));
    restore->set_physical_types({k_int32, k_int8});
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(restore);

    sirius::planner::prune_immediate_scan_restores(plan);

    auto const& restore_op = plan->Cast<sirius::op::sirius_physical_projection>();
    REQUIRE(restore_op.select_list[0]->holds<sirius::ast::cast>());
    REQUIRE(plan->get_physical_types() == std::vector<cudf::data_type>{k_int32, k_int8});
    auto const& chain_op = *plan->children[0];
    REQUIRE(chain_op.get_physical_types() == std::vector<cudf::data_type>{k_int8, k_int8});
    REQUIRE(chain_op.children[0]->get_physical_types() == std::vector<cudf::data_type>{k_int8});
  }
}

TEST_CASE("compressed_schema_propagation - pruning requires the carrier_restore provenance",
          "[compressed_schema_propagation][compressed_materialization]")
{
  // One projection over a narrowed scan, holding a single cast-over-reference expression whose
  // shape is identical in both sections; only the provenance tag differs.
  auto make_projection_over_narrow_scan = [](std::unique_ptr<sirius::ast::node> expression) {
    auto scan = make_scan(1, {k_int8});
    duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
    select_list.push_back(std::move(expression));
    auto projection = duckdb::make_uniq<sirius::op::sirius_physical_projection>(
      integer_types(1), std::move(select_list), /*estimated_cardinality=*/1);
    projection->children.push_back(std::move(scan));
    return projection;
  };

  SECTION("a tagged restore cast is pruned to a native scan")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_projection_over_narrow_scan(make_restore_cast(0));
    sirius::planner::prune_immediate_scan_restores(plan);

    REQUIRE(plan->type == SiriusPhysicalOperatorType::TABLE_SCAN);
    REQUIRE(!plan->has_physical_overrides());
  }

  SECTION("an untagged semantic cast of identical shape is never misidentified as a restore")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_projection_over_narrow_scan(make_semantic_cast(0));
    sirius::planner::prune_immediate_scan_restores(plan);

    REQUIRE(plan->type == SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(plan->Cast<sirius::op::sirius_physical_projection>()
              .select_list[0]
              ->holds<sirius::ast::cast>());
    REQUIRE(plan->children[0]->get_physical_types() == std::vector<cudf::data_type>{k_int8});
  }
}

TEST_CASE("compressed_schema_propagation - an unmappable type clears every sidecar",
          "[compressed_schema_propagation]")
{
  duckdb::vector<sirius::logical_type> unmappable_types;
  unmappable_types.push_back(sirius::logical_type::make(sirius::type_id::SQLNULL));
  auto unmappable =
    duckdb::make_uniq<sirius_physical_operator>(SiriusPhysicalOperatorType::COLUMN_DATA_SCAN,
                                                std::move(unmappable_types),
                                                /*estimated_cardinality=*/1);

  auto delim = duckdb::make_uniq<sirius::op::sirius_physical_delim_join>(
    SiriusPhysicalOperatorType::LEFT_DELIM_JOIN,
    integer_types(1),
    std::move(unmappable),
    duckdb::vector<duckdb::const_reference<sirius_physical_operator>>{},
    /*estimated_cardinality=*/1,
    duckdb::optional_idx());
  delim->children.push_back(make_scan(1, {k_int8}));
  duckdb::unique_ptr<sirius_physical_operator> plan = std::move(delim);

  sirius::planner::apply_compressed_schema_passes(plan);

  REQUIRE(!plan->has_physical_overrides());
  REQUIRE(plan->children[0]->type == SiriusPhysicalOperatorType::TABLE_SCAN);
  REQUIRE(!plan->children[0]->has_physical_overrides());
}
