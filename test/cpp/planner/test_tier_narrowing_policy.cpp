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
 * @file test_tier_narrowing_policy.cpp
 * @brief Contract tests for `apply_tier_narrowing_policy` over hand-built physical operator
 *        trees: the per-column verdict rule (transport keeps, narrow-domain comparisons rescue,
 *        boundary restores veto, evaluator restores retract), the exact eligibility mirrors of
 *        the propagation cases (hash-join keys and payload maps, the grouped-aggregate ladder,
 *        DELIM_JOIN sub-trees, unmodeled boundaries), host-tier invisibility, multi-scan
 *        independence, and the pass-pipeline composition over a Q1-shaped tree.
 */

#include "expression/aggregate_id.hpp"
#include "expression/ast/constant_range.hpp"
#include "expression/ast/node.hpp"
#include "expression/function_id.hpp"
#include "expression/join_condition.hpp"
#include "expression/value.hpp"
#include "helper/logical_type.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/sirius_plan_compressed_schema.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

constexpr cudf::data_type k_int8{cudf::type_id::INT8};
constexpr cudf::data_type k_int16{cudf::type_id::INT16};
constexpr cudf::data_type k_int32{cudf::type_id::INT32};

sirius::logical_type integer_type() { return sirius::logical_type::make(sirius::type_id::INTEGER); }

sirius::logical_type date_type() { return sirius::logical_type::make(sirius::type_id::DATE); }

duckdb::vector<sirius::logical_type> integer_types(std::size_t count)
{
  duckdb::vector<sirius::logical_type> types;
  types.reserve(count);
  for (std::size_t i = 0; i < count; i++) {
    types.push_back(integer_type());
  }
  return types;
}

duckdb::vector<sirius::logical_type> type_pair(sirius::logical_type lhs, sirius::logical_type rhs)
{
  duckdb::vector<sirius::logical_type> types;
  types.push_back(std::move(lhs));
  types.push_back(std::move(rhs));
  return types;
}

duckdb::vector<duckdb::LogicalType> duckdb_integer_types(std::size_t count)
{
  duckdb::vector<duckdb::LogicalType> types;
  for (std::size_t i = 0; i < count; i++) {
    types.push_back(duckdb::LogicalType::INTEGER);
  }
  return types;
}

std::unique_ptr<sirius::ast::node> make_reference(uint32_t column_index,
                                                  sirius::logical_type type = integer_type())
{
  return std::make_unique<sirius::ast::node>(sirius::ast::reference{column_index, std::move(type)});
}

std::unique_ptr<sirius::ast::node> make_integer_constant(int32_t value)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::constant{sirius::value{value}, integer_type()});
}

// A DECIMAL(18, @p scale) literal with unscaled value @p unscaled.
std::unique_ptr<sirius::ast::node> make_decimal_constant(int64_t unscaled, uint8_t scale)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::constant{sirius::value{sirius::decimal64{unscaled, scale}},
                          sirius::logical_type::make_decimal(18, scale)});
}

std::unique_ptr<sirius::ast::node> make_comparison(std::unique_ptr<sirius::ast::node> left,
                                                   std::unique_ptr<sirius::ast::node> right)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::lt, std::move(left), std::move(right)});
}

std::unique_ptr<sirius::ast::node> make_between(std::unique_ptr<sirius::ast::node> input,
                                                std::unique_ptr<sirius::ast::node> lower,
                                                std::unique_ptr<sirius::ast::node> upper)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::between{std::move(input),
                                                                  std::move(lower),
                                                                  std::move(upper),
                                                                  /*lower_inclusive=*/true,
                                                                  /*upper_inclusive=*/true});
}

std::unique_ptr<sirius::ast::node> make_add(std::unique_ptr<sirius::ast::node> left,
                                            std::unique_ptr<sirius::ast::node> right)
{
  std::vector<std::unique_ptr<sirius::ast::node>> arguments;
  arguments.push_back(std::move(left));
  arguments.push_back(std::move(right));
  return std::make_unique<sirius::ast::node>(
    sirius::ast::function_call{sirius::function_id::add, std::move(arguments), integer_type()});
}

// A TABLE_SCAN leaf over @p types with @p physical installed as its sidecar and the GPU-tier
// backing flag set per @p gpu_tier_pin (an identity output mapping throughout).
duckdb::unique_ptr<sirius::op::sirius_physical_table_scan> make_scan(
  duckdb::vector<sirius::logical_type> types,
  std::vector<cudf::data_type> physical,
  bool gpu_tier_pin = true)
{
  auto const column_count = types.size();
  duckdb::vector<std::size_t> projection_ids;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<std::string> names;
  for (std::size_t i = 0; i < column_count; i++) {
    projection_ids.push_back(i);
    column_ids.emplace_back(i);
    names.push_back("c" + std::to_string(i));
  }
  duckdb::TableFunction function("test_scan", {}, nullptr, nullptr);
  auto returned_types = types;
  auto scan           = duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(
    std::move(types),
    std::move(function),
    /*bind_data=*/nullptr,
    std::move(returned_types),
    std::move(column_ids),
    std::move(projection_ids),
    std::move(names),
    duckdb::make_uniq<duckdb::TableFilterSet>(),
    /*estimated_cardinality=*/1,
    duckdb::ExtraOperatorInfo(),
    duckdb::vector<duckdb::Value>{},
    duckdb::virtual_column_map_t{});
  scan->set_physical_types(std::move(physical));
  scan->sidecar_from_gpu_tier_pin = gpu_tier_pin;
  return scan;
}

duckdb::unique_ptr<sirius::op::sirius_physical_table_scan> make_integer_scan(
  std::size_t column_count, std::vector<cudf::data_type> physical, bool gpu_tier_pin = true)
{
  return make_scan(integer_types(column_count), std::move(physical), gpu_tier_pin);
}

// A passthrough filter over @p child evaluating @p predicate.
duckdb::unique_ptr<sirius::op::sirius_physical_filter> make_filter(
  std::unique_ptr<sirius::ast::node> predicate, duckdb::unique_ptr<sirius_physical_operator> child)
{
  auto filter = duckdb::make_uniq<sirius::op::sirius_physical_filter>(
    child->types, std::move(predicate), child->estimated_cardinality);
  filter->children.push_back(std::move(child));
  return filter;
}

duckdb::unique_ptr<sirius::op::sirius_physical_projection> make_projection(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
  duckdb::unique_ptr<sirius_physical_operator> child)
{
  auto projection = duckdb::make_uniq<sirius::op::sirius_physical_projection>(
    std::move(types), std::move(select_list), child->estimated_cardinality);
  projection->children.push_back(std::move(child));
  return projection;
}

// A hash join on column 0 of both inputs with identity output maps over each side.
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

std::unique_ptr<sirius::ast::node> make_aggregate_expression(uint32_t input_idx,
                                                             sirius::aggregate_id function,
                                                             bool distinct = false)
{
  std::vector<std::unique_ptr<sirius::ast::node>> arguments;
  arguments.push_back(make_reference(input_idx));
  return std::make_unique<sirius::ast::node>(sirius::ast::aggregate{
    function, std::move(arguments), sirius::logical_type::make(sirius::type_id::BIGINT), distinct});
}

// A HASH_GROUP_BY over @p child grouping on @p group_columns with one @p function per entry of
// @p aggregate_input_columns; output layout is [keys..., aggregates...].
duckdb::unique_ptr<sirius::op::sirius_physical_grouped_aggregate> make_grouped_aggregate(
  std::vector<uint32_t> group_columns,
  std::vector<uint32_t> aggregate_input_columns,
  duckdb::unique_ptr<sirius_physical_operator> child,
  duckdb::vector<duckdb::GroupingSet> grouping_sets = {},
  sirius::aggregate_id function                     = sirius::aggregate_id::sum,
  bool distinct                                     = false)
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
    expressions.push_back(make_aggregate_expression(input_idx, function, distinct));
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

// A two-column scan of `types` carried as `physical`, filtered by the column-versus-column
// comparison `c0 < c1` and funnelled into a constant-only projection, so each column's verdict is
// decided purely by how the filter classifies the reference pair.
duckdb::unique_ptr<sirius_physical_operator> make_reference_pair_plan(
  duckdb::vector<sirius::logical_type> types, std::vector<cudf::data_type> physical)
{
  auto predicate = make_comparison(make_reference(0, types[0]), make_reference(1, types[1]));
  auto filter = make_filter(std::move(predicate), make_scan(std::move(types), std::move(physical)));
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
  select_list.push_back(make_integer_constant(1));
  return make_projection(integer_types(1), std::move(select_list), std::move(filter));
}

sirius_physical_operator const& pair_plan_scan(sirius_physical_operator const& plan)
{
  auto const& scan = *plan.children[0]->children[0];
  REQUIRE(scan.type == SiriusPhysicalOperatorType::TABLE_SCAN);
  return scan;
}

// The Q1 shape: scan(c0..c2 narrow, c3 native) -> filter(constant comparison on the
// non-narrowed c3) -> projection [c0 + c1, bare c2, bare c3] -> AVG-bearing aggregate grouped on
// the c3 forward. c0/c1 are evaluator restores, c2 dies at the ineligible aggregate's boundary.
duckdb::unique_ptr<sirius_physical_operator> build_q1_plan(bool gpu_tier_pin)
{
  auto scan = make_integer_scan(4, {k_int8, k_int8, k_int8, k_int32}, gpu_tier_pin);
  auto filter =
    make_filter(make_comparison(make_reference(3), make_integer_constant(5)), std::move(scan));
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
  select_list.push_back(make_add(make_reference(0), make_reference(1)));
  select_list.push_back(make_reference(2));
  select_list.push_back(make_reference(3));
  auto projection = make_projection(integer_types(3), std::move(select_list), std::move(filter));
  return make_grouped_aggregate({2}, {1}, std::move(projection), {}, sirius::aggregate_id::avg);
}

sirius_physical_operator& q1_scan(sirius_physical_operator& plan)
{
  auto& scan = *plan.children[0]->children[0]->children[0];
  REQUIRE(scan.type == SiriusPhysicalOperatorType::TABLE_SCAN);
  return scan;
}

// Assert @p subject and @p reference are structurally identical: same operator types, logical
// schemas, child arity, sidecar presence, and (for projections) the same select-list node kinds.
void require_same_structure(sirius_physical_operator const& subject,
                            sirius_physical_operator const& reference)
{
  REQUIRE(subject.type == reference.type);
  REQUIRE(subject.types == reference.types);
  REQUIRE(subject.has_physical_overrides() == reference.has_physical_overrides());
  REQUIRE(subject.children.size() == reference.children.size());
  if (subject.type == SiriusPhysicalOperatorType::PROJECTION) {
    auto const& subject_list = subject.Cast<sirius::op::sirius_physical_projection>().select_list;
    auto const& reference_list =
      reference.Cast<sirius::op::sirius_physical_projection>().select_list;
    REQUIRE(subject_list.size() == reference_list.size());
    for (std::size_t i = 0; i < subject_list.size(); ++i) {
      REQUIRE(subject_list[i]->v.index() == reference_list[i]->v.index());
    }
  }
  for (std::size_t i = 0; i < subject.children.size(); ++i) {
    require_same_structure(*subject.children[i], *reference.children[i]);
  }
}

}  // namespace

TEST_CASE("tier_narrowing_policy - Q1 shape retracts every restore-only column",
          "[tier_narrowing_policy]")
{
  auto plan = build_q1_plan(/*gpu_tier_pin=*/true);

  sirius::planner::apply_tier_narrowing_policy(*plan);

  // c0/c1 (arithmetic) and c2 (boundary at the AVG-ineligible aggregate) all retract; with no
  // narrow target left the sidecar drops entirely.
  REQUIRE(!q1_scan(*plan).has_physical_overrides());
}

TEST_CASE("tier_narrowing_policy - a host-tier-backed sidecar is untouched",
          "[tier_narrowing_policy]")
{
  auto plan = build_q1_plan(/*gpu_tier_pin=*/false);

  sirius::planner::apply_tier_narrowing_policy(*plan);

  REQUIRE(q1_scan(*plan).get_physical_types() ==
          std::vector<cudf::data_type>{k_int8, k_int8, k_int8, k_int32});
}

TEST_CASE("tier_narrowing_policy - join payloads keep narrow, join keys retract",
          "[tier_narrowing_policy]")
{
  SECTION("INNER join forwards payloads on both sides")
  {
    auto plan = make_hash_join(duckdb::JoinType::INNER,
                               make_integer_scan(2, {k_int8, k_int8}),
                               make_integer_scan(2, {k_int8, k_int16}),
                               duckdb_integer_types(4));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    // The payloads survive into the join output (transport, which also outweighs the plan-root
    // boundary above); the key columns die at the key restore below the join.
    REQUIRE(plan->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int8});
    REQUIRE(plan->children[1]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int16});
  }

  SECTION("SEMI join: an uncollected build payload dies free and stays narrow")
  {
    auto plan = make_hash_join(duckdb::JoinType::SEMI,
                               make_integer_scan(2, {k_int8, k_int8}),
                               make_integer_scan(2, {k_int8, k_int16}),
                               duckdb_integer_types(2));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(plan->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int8});
    REQUIRE(plan->children[1]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int16});
  }

  SECTION("dynamic-filter targets are not modeled; the propagation guard commutes")
  {
    // The policy keeps the probe payload narrow (transport at the join output) even though it is
    // a registered dynamic-filter target; the propagation TABLE_SCAN case then forces the target
    // native. Both passes only flip narrow to native, so the plan converges either way — the
    // retraction count records the policy's verdict, not the final sidecar.
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_hash_join(duckdb::JoinType::INNER,
                     make_integer_scan(2, {k_int8, k_int8}),
                     make_integer_scan(2, {k_int8, k_int16}),
                     duckdb_integer_types(4));
    auto& probe = plan->children[0]->Cast<sirius::op::sirius_physical_table_scan>();
    probe.sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    probe.sirius_dynamic_filters->register_producer({1});

    auto const retracted = sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(retracted == 2);  // both join keys; the targeted payload is kept
    REQUIRE(probe.get_physical_types() == std::vector<cudf::data_type>{k_int32, k_int8});

    sirius::planner::propagate_compressed_schema(plan);

    REQUIRE(!probe.has_physical_overrides());
  }
}

TEST_CASE("tier_narrowing_policy - grouped-aggregate keys keep narrow only when eligible",
          "[tier_narrowing_policy]")
{
  SECTION("eligible aggregate: group key kept, SUM input retracted")
  {
    auto plan = make_grouped_aggregate({0}, {1}, make_integer_scan(2, {k_int8, k_int8}));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(plan->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int8, k_int32});
  }

  SECTION("COUNT_VALID does not constrain a counted group key")
  {
    auto plan = make_grouped_aggregate(
      {0}, {0}, make_integer_scan(1, {k_int8}), {}, sirius::aggregate_id::count);

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(plan->children[0]->get_physical_types() == std::vector<cudf::data_type>{k_int8});
  }

  SECTION("AVG makes the aggregate ineligible: no transport, all columns retract")
  {
    auto plan = make_grouped_aggregate(
      {0}, {1}, make_integer_scan(2, {k_int8, k_int8}), {}, sirius::aggregate_id::avg);

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(!plan->children[0]->has_physical_overrides());
  }

  SECTION("COUNT(DISTINCT) makes the aggregate ineligible")
  {
    auto plan = make_grouped_aggregate({0},
                                       {1},
                                       make_integer_scan(2, {k_int8, k_int8}),
                                       {},
                                       sirius::aggregate_id::count,
                                       /*distinct=*/true);

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(!plan->children[0]->has_physical_overrides());
  }

  SECTION("multiple grouping sets make the aggregate ineligible")
  {
    duckdb::vector<duckdb::GroupingSet> grouping_sets;
    grouping_sets.push_back(duckdb::GroupingSet{0});
    grouping_sets.push_back(duckdb::GroupingSet{});
    auto plan = make_grouped_aggregate(
      {0}, {1}, make_integer_scan(2, {k_int8, k_int8}), std::move(grouping_sets));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(!plan->children[0]->has_physical_overrides());
  }
}

TEST_CASE("tier_narrowing_policy - transport outweighs evaluator restores above the join",
          "[tier_narrowing_policy]")
{
  // Q9 shape: the payloads are join-payload transport AND arithmetic inputs above the join. The
  // exchange benefit keeps them narrow; only the keys retract.
  auto join = make_hash_join(duckdb::JoinType::INNER,
                             make_integer_scan(2, {k_int8, k_int8}),
                             make_integer_scan(2, {k_int8, k_int16}),
                             duckdb_integer_types(4));
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
  select_list.push_back(make_add(make_reference(1), make_reference(3)));
  auto plan = make_projection(integer_types(1), std::move(select_list), std::move(join));

  sirius::planner::apply_tier_narrowing_policy(*plan);

  auto const& join_op = *plan->children[0];
  REQUIRE(join_op.children[0]->get_physical_types() ==
          std::vector<cudf::data_type>{k_int32, k_int8});
  REQUIRE(join_op.children[1]->get_physical_types() ==
          std::vector<cudf::data_type>{k_int32, k_int16});
}

TEST_CASE("tier_narrowing_policy - narrow-domain comparisons rescue, ineligible shapes do not",
          "[tier_narrowing_policy]")
{
  // Every section funnels the carried column into a projection output it does not survive, so
  // the verdict is decided purely by the filter-predicate classification plus any boundary flag.
  auto make_dropping_projection = [](duckdb::unique_ptr<sirius_physical_operator> child) {
    duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
    select_list.push_back(make_add(make_reference(0), make_reference(0)));
    return make_projection(integer_types(1), std::move(select_list), std::move(child));
  };

  SECTION("a representable constant comparison keeps the column narrow")
  {
    auto plan = make_dropping_projection(
      make_filter(make_comparison(make_reference(0), make_integer_constant(5)),
                  make_integer_scan(1, {k_int8})));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(plan->children[0]->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int8});
  }

  SECTION("a representable BETWEEN keeps the column narrow")
  {
    auto plan = make_dropping_projection(make_filter(
      make_between(make_reference(0), make_integer_constant(1), make_integer_constant(10)),
      make_integer_scan(1, {k_int8})));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(plan->children[0]->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int8});
  }

  SECTION("a constant outside the carrier's range does not rescue")
  {
    auto plan = make_dropping_projection(
      make_filter(make_comparison(make_reference(0), make_integer_constant(1000)),
                  make_integer_scan(1, {k_int8})));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(!plan->children[0]->children[0]->has_physical_overrides());
  }

  SECTION("a decimal constant rescues only at the carrier's scale")
  {
    auto const decimal_carrier = cudf::data_type{cudf::type_id::DECIMAL32, -2};
    auto make_decimal_plan     = [&](uint8_t constant_scale) {
      duckdb::vector<sirius::logical_type> types;
      types.push_back(sirius::logical_type::make_decimal(18, 2));
      auto filter =
        make_filter(make_comparison(make_reference(0, sirius::logical_type::make_decimal(18, 2)),
                                    make_decimal_constant(1234, constant_scale)),
                    make_scan(std::move(types), {decimal_carrier}));
      duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
      select_list.push_back(make_integer_constant(1));
      return make_projection(integer_types(1), std::move(select_list), std::move(filter));
    };

    auto matched = make_decimal_plan(/*constant_scale=*/2);
    sirius::planner::apply_tier_narrowing_policy(*matched);
    REQUIRE(matched->children[0]->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{decimal_carrier});

    auto mismatched = make_decimal_plan(/*constant_scale=*/3);
    sirius::planner::apply_tier_narrowing_policy(*mismatched);
    REQUIRE(!mismatched->children[0]->children[0]->has_physical_overrides());
  }

  SECTION("a reference-versus-reference comparison at different carriers does not rescue")
  {
    // Only a shared carrier lets a pair evaluate narrow; at different widths each reference is an
    // ordinary evaluator restore. The shared-carrier case is covered below.
    auto plan = make_reference_pair_plan(integer_types(2), {k_int8, k_int16});

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(!pair_plan_scan(*plan).has_physical_overrides());
  }

  SECTION("an IN list restores in the evaluator and does not rescue")
  {
    std::vector<std::unique_ptr<sirius::ast::node>> in_values;
    in_values.push_back(make_integer_constant(1));
    in_values.push_back(make_integer_constant(2));
    auto predicate = std::make_unique<sirius::ast::node>(
      sirius::ast::in_list{make_reference(0), std::move(in_values), /*negated=*/false});
    auto plan =
      make_dropping_projection(make_filter(std::move(predicate), make_integer_scan(1, {k_int8})));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(!plan->children[0]->children[0]->has_physical_overrides());
  }

  SECTION("a boundary restore vetoes the comparison rescue")
  {
    // c0 engages a narrow comparison but is also a SUM input: resurrecting its narrow target
    // would resurrect the whole restore projection below the aggregate. c1 keeps its key
    // transport.
    auto filter = make_filter(make_comparison(make_reference(0), make_integer_constant(5)),
                              make_integer_scan(2, {k_int8, k_int8}));
    auto plan   = make_grouped_aggregate({1}, {0}, std::move(filter));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(plan->children[0]->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int8});
  }
}

TEST_CASE("tier_narrowing_policy - a same-carrier reference pair rescues both columns",
          "[tier_narrowing_policy]")
{
  SECTION("both integer columns keep narrow")
  {
    auto plan = make_reference_pair_plan(integer_types(2), {k_int8, k_int8});

    auto const retracted = sirius::planner::apply_tier_narrowing_policy(*plan);

    // Both sides are marked, not just the one the probe happened to reach first.
    REQUIRE(retracted == 0);
    REQUIRE(pair_plan_scan(*plan).get_physical_types() ==
            std::vector<cudf::data_type>{k_int8, k_int8});
  }

  SECTION("both DATE columns keep narrow")
  {
    auto plan = make_reference_pair_plan(type_pair(date_type(), date_type()), {k_int16, k_int16});

    auto const retracted = sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(retracted == 0);
    REQUIRE(pair_plan_scan(*plan).get_physical_types() ==
            std::vector<cudf::data_type>{k_int16, k_int16});
  }

  SECTION("a column the sidecar leaves native is no pair candidate")
  {
    // c1 is native in the sidecar, so no pair forms and c0 falls through to the ordinary
    // evaluator-restore classification and retracts alone.
    auto plan = make_reference_pair_plan(integer_types(2), {k_int8, k_int32});

    auto const retracted = sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(retracted == 1);
    REQUIRE(!pair_plan_scan(*plan).has_physical_overrides());
  }

  SECTION("comparing one column with itself marks its single origin")
  {
    auto filter = make_filter(make_comparison(make_reference(0), make_reference(0)),
                              make_integer_scan(1, {k_int8}));
    duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
    select_list.push_back(make_integer_constant(1));
    auto plan = make_projection(integer_types(1), std::move(select_list), std::move(filter));

    auto const retracted = sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(retracted == 0);
    REQUIRE(plan->children[0]->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int8});
  }

  SECTION("a boundary restore still vetoes the pair rescue")
  {
    // Both columns engage the pair, but c0 is also a SUM input: the restore projection the
    // aggregate forces below itself outweighs the rescue, while c1 keeps its group-key transport.
    auto filter = make_filter(make_comparison(make_reference(0), make_reference(1)),
                              make_integer_scan(2, {k_int8, k_int8}));
    auto plan   = make_grouped_aggregate({1}, {0}, std::move(filter));

    auto const retracted = sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(retracted == 1);
    REQUIRE(plan->children[0]->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int8});
  }
}

TEST_CASE("tier_narrowing_policy - pair verdicts track the shared eligibility predicate",
          "[tier_narrowing_policy]")
{
  // The policy's plan-time pair probe must accept exactly what the evaluator's
  // `narrow_domain_reference_pair_eligible` accepts at runtime: a plan that keeps a pair narrow on
  // a shape the evaluator then restores pays the restore the verdict assumed it had avoided. Each
  // row states the predicate's answer for one (logical, carrier) quadruple and asserts the policy
  // agrees -- kept at the planned carriers when eligible, retracted to native when not.
  struct pair_case {
    char const* description;
    sirius::logical_type lhs;
    cudf::data_type lhs_carrier;
    sirius::logical_type rhs;
    cudf::data_type rhs_carrier;
    bool eligible;
  };

  auto const decimal_2 = sirius::logical_type::make_decimal(18, 2);
  auto const decimal_3 = sirius::logical_type::make_decimal(18, 3);
  cudf::data_type const decimal32_2{cudf::type_id::DECIMAL32, -2};
  cudf::data_type const decimal32_3{cudf::type_id::DECIMAL32, -3};
  auto const smallint = sirius::logical_type::make(sirius::type_id::SMALLINT);
  auto const bigint   = sirius::logical_type::make(sirius::type_id::BIGINT);
  auto const uinteger = sirius::logical_type::make(sirius::type_id::UINTEGER);
  cudf::data_type const k_uint16{cudf::type_id::UINT16};

  std::vector<pair_case> const cases{
    {"integers at one carrier", integer_type(), k_int8, integer_type(), k_int8, true},
    {"different logical widths at one carrier", bigint, k_int16, integer_type(), k_int16, true},
    {"dates at one carrier", date_type(), k_int16, date_type(), k_int16, true},
    {"decimals at one carrier and scale", decimal_2, decimal32_2, decimal_2, decimal32_2, true},
    {"integers at different carriers", integer_type(), k_int8, integer_type(), k_int16, false},
    {"epoch days against plain integers", date_type(), k_int16, integer_type(), k_int16, false},
    {"plain integers against epoch days", integer_type(), k_int16, date_type(), k_int16, false},
    {"an already-minimal type is not narrowed", smallint, k_int16, integer_type(), k_int16, false},
    {"decimals at different scales", decimal_2, decimal32_2, decimal_3, decimal32_3, false},
    {"unsigned against signed", uinteger, k_uint16, integer_type(), k_int16, false},
  };

  for (auto const& test_case : cases) {
    DYNAMIC_SECTION(test_case.description)
    {
      REQUIRE(sirius::ast::narrow_domain_reference_pair_eligible(
                test_case.lhs, test_case.lhs_carrier, test_case.rhs, test_case.rhs_carrier) ==
              test_case.eligible);

      std::vector<cudf::data_type> const physical{test_case.lhs_carrier, test_case.rhs_carrier};
      auto plan = make_reference_pair_plan(type_pair(test_case.lhs, test_case.rhs), physical);

      sirius::planner::apply_tier_narrowing_policy(*plan);

      auto const& scan = pair_plan_scan(*plan);
      if (test_case.eligible) {
        REQUIRE(scan.get_physical_types() == physical);
      } else {
        // Every ineligible row retracts each carried column to native, which empties the sidecar.
        REQUIRE(!scan.has_physical_overrides());
      }
    }
  }
}

TEST_CASE("tier_narrowing_policy - unmodeled operators and DELIM_JOIN sub-trees are boundaries",
          "[tier_narrowing_policy]")
{
  SECTION("an unmodeled operator above the carrier retracts it")
  {
    auto boundary = duckdb::make_uniq<sirius_physical_operator>(
      SiriusPhysicalOperatorType::TOP_N, integer_types(1), /*estimated_cardinality=*/1);
    boundary->children.push_back(make_integer_scan(1, {k_int8}));
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(boundary);

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(!plan->children[0]->has_physical_overrides());
  }

  SECTION("a hash join as a DELIM_JOIN sub-root awards no payload transport")
  {
    // Propagation restores every child of the sub-root in place, so these payloads never cross
    // the sub-root join narrow; the would-be transport of its output map must not rescue them.
    auto sub_join = make_hash_join(duckdb::JoinType::INNER,
                                   make_integer_scan(2, {k_int8, k_int8}),
                                   make_integer_scan(2, {k_int8, k_int16}),
                                   duckdb_integer_types(4));
    auto delim    = duckdb::make_uniq<sirius::op::sirius_physical_delim_join>(
      SiriusPhysicalOperatorType::LEFT_DELIM_JOIN,
      integer_types(4),
      std::move(sub_join),
      duckdb::vector<duckdb::const_reference<sirius_physical_operator>>{},
      /*estimated_cardinality=*/1,
      duckdb::optional_idx());
    delim->children.push_back(make_integer_scan(1, {k_int8}));
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(delim);

    sirius::planner::apply_tier_narrowing_policy(*plan);

    auto const& delim_ref = plan->Cast<sirius::op::sirius_physical_delim_join>();
    REQUIRE(!delim_ref.join->children[0]->has_physical_overrides());
    REQUIRE(!delim_ref.join->children[1]->has_physical_overrides());
    REQUIRE(!plan->children[0]->has_physical_overrides());
  }

  SECTION("DELIM_JOIN sub-tree survivors retract")
  {
    auto delim = duckdb::make_uniq<sirius::op::sirius_physical_delim_join>(
      SiriusPhysicalOperatorType::LEFT_DELIM_JOIN,
      integer_types(1),
      make_integer_scan(1, {k_int8}),
      duckdb::vector<duckdb::const_reference<sirius_physical_operator>>{},
      /*estimated_cardinality=*/1,
      duckdb::optional_idx());
    delim->distinct_root = make_integer_scan(1, {k_int8});
    delim->children.push_back(make_integer_scan(1, {k_int8}));
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(delim);

    sirius::planner::apply_tier_narrowing_policy(*plan);

    auto const& delim_ref = plan->Cast<sirius::op::sirius_physical_delim_join>();
    REQUIRE(!delim_ref.join->has_physical_overrides());
    REQUIRE(!delim_ref.distinct_root->has_physical_overrides());
    REQUIRE(!plan->children[0]->has_physical_overrides());
  }
}

TEST_CASE("tier_narrowing_policy - verdicts are per scan and per origin", "[tier_narrowing_policy]")
{
  SECTION("two scans do not cross-contaminate")
  {
    // The left payload earns transport; the right scan's only narrow column is the join key, so
    // its whole sidecar retracts while the left sidecar keeps its payload.
    auto plan = make_hash_join(duckdb::JoinType::INNER,
                               make_integer_scan(2, {k_int8, k_int8}),
                               make_integer_scan(1, {k_int8}),
                               duckdb_integer_types(3));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(plan->children[0]->get_physical_types() ==
            std::vector<cudf::data_type>{k_int32, k_int8});
    REQUIRE(!plan->children[1]->has_physical_overrides());
  }

  SECTION("duplicated forwards accumulate on one origin")
  {
    // Both projection outputs forward the same scan column, so its root boundary restore reaches
    // the single origin and vetoes the comparison rescue.
    auto filter = make_filter(make_comparison(make_reference(0), make_integer_constant(5)),
                              make_integer_scan(1, {k_int8}));
    duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
    select_list.push_back(make_reference(0));
    select_list.push_back(make_reference(0));
    auto plan = make_projection(integer_types(2), std::move(select_list), std::move(filter));

    sirius::planner::apply_tier_narrowing_policy(*plan);

    REQUIRE(!plan->children[0]->children[0]->has_physical_overrides());
  }
}

TEST_CASE("tier_narrowing_policy - pass pipeline composes to the feature-off plan for Q1 shapes",
          "[tier_narrowing_policy]")
{
  auto subject   = build_q1_plan(/*gpu_tier_pin=*/true);
  auto reference = build_q1_plan(/*gpu_tier_pin=*/false);
  q1_scan(*reference).set_physical_types({});

  sirius::planner::apply_compressed_schema_passes(subject);

  // Every narrow target retracts before propagation, so no restore projection and no sidecar
  // exists anywhere: the tree is structurally identical to the same plan built with the feature
  // off.
  require_same_structure(*subject, *reference);
}
