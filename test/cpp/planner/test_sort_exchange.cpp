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

#include "catch.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "helper/type_conversions.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_sort_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "planner/sirius_plan_sort_exchange.hpp"
#include "sirius_config.hpp"
#include "utils/test_physical_operator.hpp"

#include <stdexcept>

using sirius::op::SiriusPhysicalOperatorType;

namespace {

duckdb::unique_ptr<sirius::op::sirius_physical_order> make_two_key_order(
  duckdb::vector<std::size_t> projections)
{
  auto types = sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{
    duckdb::LogicalType::INTEGER, duckdb::LogicalType::BIGINT, duckdb::LogicalType::DOUBLE});
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.emplace_back(duckdb::OrderType::ASCENDING,
                      duckdb::OrderByNullType::NULLS_FIRST,
                      duckdb::make_uniq<duckdb::BoundReferenceExpression>(
                        duckdb::LogicalType::INTEGER, duckdb::idx_t{0}));
  orders.emplace_back(duckdb::OrderType::DESCENDING,
                      duckdb::OrderByNullType::NULLS_LAST,
                      duckdb::make_uniq<duckdb::BoundReferenceExpression>(
                        duckdb::LogicalType::BIGINT, duckdb::idx_t{1}));
  duckdb::vector<sirius::logical_type> output_types;
  for (auto idx : projections) {
    output_types.push_back(types[idx]);
  }
  auto order = duckdb::make_uniq<sirius::op::sirius_physical_order>(
    std::move(output_types), std::move(orders), std::move(projections), 100);
  order->children.push_back(duckdb::make_uniq<sirius::op::test_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION, types, 100));
  return order;
}

}  // namespace

TEST_CASE("to_duckdb_order inverts the cuDF order mapping", "[planner][sort_exchange]")
{
  for (auto type : {duckdb::OrderType::ASCENDING, duckdb::OrderType::DESCENDING}) {
    for (auto nulls : {duckdb::OrderByNullType::NULLS_FIRST, duckdb::OrderByNullType::NULLS_LAST}) {
      auto const back = sirius::op::to_duckdb_order(sirius::op::to_cudf_order(type),
                                                    sirius::op::to_cudf_null_order(type, nulls));
      CHECK(back.type == type);
      CHECK(back.null_order == nulls);
    }
  }
}

TEST_CASE("build_sort_exchange builds the distributed sort chain", "[planner][sort_exchange]")
{
  auto options = sirius::planner::sort_exchange_options{};
  SECTION("default options use every key and keep an identity projection")
  {
    auto root = sirius::planner::build_sort_exchange(make_two_key_order({0, 1, 2}), options);
    REQUIRE(root->type == SiriusPhysicalOperatorType::MERGE_SORT);
    CHECK(root->types.size() == 3);
    auto& partition = root->children[0]->Cast<sirius::op::sirius_physical_sort_partition>();
    auto& sample    = partition.children[0]->Cast<sirius::op::sirius_physical_sort_sample>();
    CHECK(partition.get_sample_op() == &sample);
    CHECK(sample.boundary_key_count() == 2);
    REQUIRE(sample.children[0]->type == SiriusPhysicalOperatorType::ORDER_BY);
    CHECK(sample.children[0]->children[0]->type == SiriusPhysicalOperatorType::PROJECTION);
  }
  SECTION("a boundary prefix reaches the sample and a projection moves to MERGE_SORT")
  {
    options.boundary_key_count = 1;
    auto root = sirius::planner::build_sort_exchange(make_two_key_order({2, 0}), options);
    REQUIRE(root->type == SiriusPhysicalOperatorType::MERGE_SORT);
    REQUIRE(root->types.size() == 2);
    CHECK(root->types[0] == sirius::from_duckdb(duckdb::LogicalType::DOUBLE));
    auto& sample = root->children[0]->children[0]->Cast<sirius::op::sirius_physical_sort_sample>();
    CHECK(sample.boundary_key_count() == 1);
    auto& order = sample.children[0]->Cast<sirius::op::sirius_physical_order>();
    CHECK(order.projections == duckdb::vector<std::size_t>{0, 1, 2});
    CHECK(order.types.size() == 3);
  }
  SECTION("invalid input throws")
  {
    options.boundary_key_count = 3;
    CHECK_THROWS_AS(sirius::planner::build_sort_exchange(make_two_key_order({0, 1, 2}), options),
                    std::invalid_argument);
    auto childless = make_two_key_order({0, 1, 2});
    childless->children.clear();
    CHECK_THROWS_AS(sirius::planner::build_sort_exchange(std::move(childless),
                                                         sirius::planner::sort_exchange_options{}),
                    std::invalid_argument);
  }
}
