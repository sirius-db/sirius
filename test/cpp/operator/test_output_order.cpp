/*
 * Copyright 2025, Sirius Contributors.
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

#include "helper/type_conversions.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "op/sirius_physical_passthrough_sink.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_streaming_sink.hpp"
#include "op/sirius_physical_union.hpp"
#include "sirius/exception.hpp"

#include <catch.hpp>

#include <type_traits>

namespace {
using sirius::OrderPreservationType;
using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

class order_node : public sirius_physical_operator {
 public:
  explicit order_node(OrderPreservationType preservation,
                      SiriusPhysicalOperatorType type = SiriusPhysicalOperatorType::FILTER)
    : sirius_physical_operator(type, {}, 0), preservation(preservation)
  {
  }
  OrderPreservationType output_order_preservation() const override { return preservation; }
  // Analysis must walk through sources that preserve their child's order.
  bool is_source() const override { return true; }
  OrderPreservationType preservation;
};

static_assert(std::is_abstract_v<sirius_physical_operator>);
static_assert(std::is_abstract_v<sirius::op::sirius_physical_partition_consumer_operator>);
static_assert(std::is_abstract_v<sirius::op::sirius_physical_delim_join>);
static_assert(std::is_abstract_v<sirius::op::sirius_physical_result_collector>);
}  // namespace

TEST_CASE("Output order follows preservation through source boundaries", "[output_order]")
{
  order_node projection(OrderPreservationType::INSERTION_ORDER);
  auto filter = duckdb::make_uniq<order_node>(OrderPreservationType::INSERTION_ORDER);
  filter->children.push_back(duckdb::make_uniq<order_node>(OrderPreservationType::FIXED_ORDER));
  projection.children.push_back(std::move(filter));
  REQUIRE(projection.output_order_preservation() == OrderPreservationType::INSERTION_ORDER);
  REQUIRE(projection.output_order() == OrderPreservationType::FIXED_ORDER);
  projection.children[0]->children[0] =
    duckdb::make_uniq<order_node>(OrderPreservationType::NO_ORDER);
  REQUIRE(projection.output_order() == OrderPreservationType::NO_ORDER);
}

TEST_CASE("Output order is replaced by a defining or destroying operator", "[output_order]")
{
  order_node root(OrderPreservationType::NO_ORDER);
  root.children.push_back(duckdb::make_uniq<order_node>(OrderPreservationType::FIXED_ORDER));
  root.children.push_back(duckdb::make_uniq<order_node>(OrderPreservationType::INSERTION_ORDER));
  REQUIRE(root.output_order() == OrderPreservationType::NO_ORDER);
  root.preservation = OrderPreservationType::FIXED_ORDER;
  REQUIRE(root.output_order() == OrderPreservationType::FIXED_ORDER);
}

TEST_CASE("CTE output order follows consumer rather than materialization", "[output_order]")
{
  order_node cte(OrderPreservationType::INSERTION_ORDER, SiriusPhysicalOperatorType::CTE);
  REQUIRE_THROWS_AS(cte.output_order(), sirius::internal_exception);
  cte.children.push_back(duckdb::make_uniq<order_node>(OrderPreservationType::FIXED_ORDER));
  cte.children.push_back(duckdb::make_uniq<order_node>(OrderPreservationType::NO_ORDER));
  REQUIRE(cte.output_order() == OrderPreservationType::NO_ORDER);
  cte.children[0] = duckdb::make_uniq<order_node>(OrderPreservationType::NO_ORDER);
  cte.children[1] = duckdb::make_uniq<order_node>(OrderPreservationType::FIXED_ORDER);
  REQUIRE(cte.output_order() == OrderPreservationType::FIXED_ORDER);
}

TEST_CASE("Output order preserves external provenance and rejects ambiguous fan-in",
          "[output_order]")
{
  order_node leaf(OrderPreservationType::INSERTION_ORDER);
  REQUIRE(leaf.output_order() == OrderPreservationType::INSERTION_ORDER);
  leaf.children.push_back(duckdb::make_uniq<order_node>(OrderPreservationType::FIXED_ORDER));
  leaf.children.push_back(duckdb::make_uniq<order_node>(OrderPreservationType::NO_ORDER));
  REQUIRE_THROWS_AS(leaf.output_order(), sirius::internal_exception);
}

TEST_CASE("Output order traverses virtual child views", "[output_order]")
{
  class referenced_output : public order_node {
   public:
    referenced_output() : order_node(OrderPreservationType::INSERTION_ORDER) {}
    duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_children() const override
    {
      return {input};
    }
    order_node input{OrderPreservationType::FIXED_ORDER};
  } collector;
  REQUIRE(collector.children.empty());
  REQUIRE(collector.output_order() == OrderPreservationType::FIXED_ORDER);
}

TEST_CASE("Physical order analysis distinguishes local sorting from global merge", "[output_order]")
{
  using namespace sirius::op;
  sirius_physical_order local({}, {}, {}, 0);
  REQUIRE(local.output_order() == OrderPreservationType::NO_ORDER);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions;
  expressions.push_back(std::make_unique<sirius::ast::node>(sirius::ast::reference{0}));
  sirius_physical_projection projection({}, std::move(expressions), 0);
  auto boundary = duckdb::make_uniq<sirius_physical_passthrough_sink>(
    duckdb::vector<sirius::logical_type>{}, 0, "default");
  boundary->children.push_back(
    duckdb::make_uniq<sirius_physical_merge_sort>(duckdb::vector<sirius::logical_type>{},
                                                  duckdb::vector<duckdb::BoundOrderByNode>{},
                                                  duckdb::vector<std::size_t>{},
                                                  0));
  projection.children.push_back(std::move(boundary));
  REQUIRE(projection.output_order() == OrderPreservationType::FIXED_ORDER);
  auto union_op =
    duckdb::make_uniq<sirius_physical_union>(duckdb::vector<sirius::logical_type>{}, 0);
  union_op->children = std::move(projection.children);
  projection.children.push_back(std::move(union_op));
  REQUIRE(projection.output_order() == OrderPreservationType::NO_ORDER);
}

TEST_CASE("Streaming sink order follows routing configuration", "[output_order]")
{
  using namespace sirius::op;
  auto first  = std::make_shared<cucascade::shared_data_repository>();
  auto second = std::make_shared<cucascade::shared_data_repository>();
  sirius_physical_streaming_sink single({}, 0, first);
  REQUIRE(single.output_order_preservation() == OrderPreservationType::INSERTION_ORDER);
  partition_spec broadcast;
  broadcast.mode = partition_mode::broadcast;
  sirius_physical_streaming_sink replicated({}, 0, {first, second}, broadcast);
  REQUIRE(replicated.output_order_preservation() == OrderPreservationType::INSERTION_ORDER);
  sirius_physical_streaming_sink hashed(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    {first, second},
    partition_spec{{0}});
  REQUIRE(hashed.output_order_preservation() == OrderPreservationType::NO_ORDER);
}
