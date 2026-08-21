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

/*
 * Unit tests for `planner/dynamic_filter/build_filter_evidence.hpp`. The parity suite compares
 * the DuckDB-mirroring rules against DuckDB's own implementation on optimized plans.
 */

#include "planner/dynamic_filter/build_filter_evidence.hpp"

#include <catch.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_any_join.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_cteref.hpp>
#include <duckdb/planner/operator/logical_delim_get.hpp>
#include <duckdb/planner/operator/logical_distinct.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_limit.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_set_operation.hpp>
#include <duckdb/planner/operator/logical_top_n.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <utility>

namespace {

using sirius::planner::build_relation_is_opaque;
using sirius::planner::build_subtree_is_filtering;

// Minimal constructible scan; the evidence walk never invokes its table function.
duckdb::unique_ptr<duckdb::LogicalGet> make_get()
{
  return duckdb::make_uniq<duckdb::LogicalGet>(
    /*table_index=*/0,
    duckdb::TableFunction(),
    /*bind_data=*/nullptr,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
    duckdb::vector<duckdb::string>{"a"});
}

duckdb::unique_ptr<duckdb::LogicalGet> make_filtered_get()
{
  auto get = make_get();
  get->table_filters.PushFilter(
    duckdb::ColumnIndex{0},
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                              duckdb::Value::INTEGER(5)));
  return get;
}

duckdb::unique_ptr<duckdb::LogicalProjection> make_projection_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> child)
{
  auto projection = duckdb::make_uniq<duckdb::LogicalProjection>(
    /*table_index=*/7, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
  projection->children.push_back(std::move(child));
  return projection;
}

duckdb::unique_ptr<duckdb::LogicalDelimGet> make_delim_get()
{
  return duckdb::make_uniq<duckdb::LogicalDelimGet>(
    /*table_index=*/2, duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
}

duckdb::unique_ptr<duckdb::LogicalCTERef> make_cte_ref()
{
  return duckdb::make_uniq<duckdb::LogicalCTERef>(
    /*table_index=*/3,
    /*cte_index=*/0,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
    duckdb::vector<duckdb::string>{"a"});
}

duckdb::unique_ptr<duckdb::LogicalAggregate> make_aggregate_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> child)
{
  auto aggregate = duckdb::make_uniq<duckdb::LogicalAggregate>(
    /*group_index=*/4,
    /*aggregate_index=*/5,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
  aggregate->children.push_back(std::move(child));
  return aggregate;
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_join_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> left,
  duckdb::unique_ptr<duckdb::LogicalOperator> right)
{
  auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  join->children.push_back(std::move(left));
  join->children.push_back(std::move(right));
  return join;
}

duckdb::unique_ptr<duckdb::LogicalLimit> make_limit_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> child)
{
  auto limit = duckdb::make_uniq<duckdb::LogicalLimit>(duckdb::BoundLimitNode::ConstantValue(10),
                                                       duckdb::BoundLimitNode());
  limit->children.push_back(std::move(child));
  return limit;
}

duckdb::unique_ptr<duckdb::LogicalOrder> make_order_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> child)
{
  auto order = duckdb::make_uniq<duckdb::LogicalOrder>(duckdb::vector<duckdb::BoundOrderByNode>{});
  order->children.push_back(std::move(child));
  return order;
}

duckdb::unique_ptr<duckdb::LogicalSetOperation> make_set_operation(duckdb::LogicalOperatorType type)
{
  return duckdb::make_uniq<duckdb::LogicalSetOperation>(
    /*table_index=*/8,
    /*column_count=*/1,
    make_get(),
    make_get(),
    type,
    /*setop_all=*/false);
}

}  // namespace

TEST_CASE("a GET without table filters carries no evidence", "[dynamic_filter][evidence]")
{
  auto const get = make_get();
  REQUIRE_FALSE(build_subtree_is_filtering(*get));
}

TEST_CASE("a GET with a table filter carries evidence", "[dynamic_filter][evidence]")
{
  auto const get = make_filtered_get();
  REQUIRE(build_subtree_is_filtering(*get));
}

TEST_CASE("a bare FILTER carries evidence", "[dynamic_filter][evidence]")
{
  duckdb::LogicalFilter filter;
  REQUIRE(build_subtree_is_filtering(filter));
}

TEST_CASE("a TOP_N carries evidence", "[dynamic_filter][evidence]")
{
  duckdb::LogicalTopN top_n({}, /*limit=*/10, /*offset=*/0);
  REQUIRE(build_subtree_is_filtering(top_n));
}

TEST_CASE("evidence propagates up through non-filtering operators", "[dynamic_filter][evidence]")
{
  SECTION("a projection over a filtered GET fires")
  {
    auto const projection = make_projection_over(make_filtered_get());
    REQUIRE(build_subtree_is_filtering(*projection));
  }

  SECTION("a projection over an unfiltered GET stays quiet")
  {
    auto const projection = make_projection_over(make_get());
    REQUIRE_FALSE(build_subtree_is_filtering(*projection));
  }

  SECTION("a join fires when ANY child subtree fires")
  {
    auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
    join->children.push_back(make_get());
    join->children.push_back(make_projection_over(make_filtered_get()));
    REQUIRE(build_subtree_is_filtering(*join));
  }

  SECTION("a join over only unfiltered children stays quiet")
  {
    auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
    join->children.push_back(make_get());
    join->children.push_back(make_get());
    REQUIRE_FALSE(build_subtree_is_filtering(*join));
  }
}

TEST_CASE("a childless non-filtering operator carries no evidence", "[dynamic_filter][evidence]")
{
  auto const projection = duckdb::make_uniq<duckdb::LogicalProjection>(
    /*table_index=*/1, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
  REQUIRE_FALSE(build_subtree_is_filtering(*projection));
}

TEST_CASE("opaque build leaves carry fallback evidence", "[dynamic_filter][evidence]")
{
  SECTION("a bare DELIM_GET") { REQUIRE(build_relation_is_opaque(*make_delim_get())); }

  SECTION("a projected DELIM_GET")
  {
    REQUIRE(build_relation_is_opaque(*make_projection_over(make_delim_get())));
  }

  SECTION("a DELIM_GET below stacked projections")
  {
    REQUIRE(
      build_relation_is_opaque(*make_projection_over(make_projection_over(make_delim_get()))));
  }

  SECTION("a bare CTE_REF") { REQUIRE(build_relation_is_opaque(*make_cte_ref())); }

  SECTION("a projected CTE_REF")
  {
    REQUIRE(build_relation_is_opaque(*make_projection_over(make_cte_ref())));
  }

  SECTION("a CTE_REF below stacked projections")
  {
    REQUIRE(build_relation_is_opaque(*make_projection_over(make_projection_over(make_cte_ref()))));
  }
}

TEST_CASE("base-table builds carry no opaque fallback evidence", "[dynamic_filter][evidence]")
{
  REQUIRE_FALSE(build_relation_is_opaque(*make_get()));
  REQUIRE_FALSE(build_relation_is_opaque(*make_filtered_get()));
  REQUIRE_FALSE(build_relation_is_opaque(*make_projection_over(make_get())));
}

TEST_CASE("malformed projections carry no opaque fallback evidence", "[dynamic_filter][evidence]")
{
  SECTION("a childless projection")
  {
    auto projection = duckdb::make_uniq<duckdb::LogicalProjection>(
      /*table_index=*/6, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
    REQUIRE_FALSE(build_relation_is_opaque(*projection));
  }

  SECTION("a projection with a null child")
  {
    auto projection = duckdb::make_uniq<duckdb::LogicalProjection>(
      /*table_index=*/6, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
    projection->children.push_back(nullptr);
    REQUIRE_FALSE(build_relation_is_opaque(*projection));
  }

  SECTION("a projection with multiple children")
  {
    auto projection = make_projection_over(make_cte_ref());
    projection->children.push_back(make_get());
    REQUIRE_FALSE(build_relation_is_opaque(*projection));
  }
}

TEST_CASE("reducing operators are not opaque build roots", "[dynamic_filter][evidence]")
{
  SECTION("a grouped aggregate")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_aggregate_over(make_get())));
  }

  SECTION("a DISTINCT")
  {
    auto distinct = duckdb::make_uniq<duckdb::LogicalDistinct>(
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{}, duckdb::DistinctType::DISTINCT);
    distinct->children.push_back(make_get());
    REQUIRE_FALSE(build_relation_is_opaque(*distinct));
  }

  SECTION("a comparison join")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_join_over(make_get(), make_get())));
  }

  SECTION("a DELIM_JOIN")
  {
    auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(
      duckdb::JoinType::INNER, duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN);
    join->children.push_back(make_get());
    join->children.push_back(make_get());
    REQUIRE_FALSE(build_relation_is_opaque(*join));
  }

  SECTION("an ANY_JOIN")
  {
    auto join = duckdb::make_uniq<duckdb::LogicalAnyJoin>(duckdb::JoinType::INNER);
    join->children.push_back(make_get());
    join->children.push_back(make_get());
    REQUIRE_FALSE(build_relation_is_opaque(*join));
  }

  SECTION("an INTERSECT")
  {
    REQUIRE_FALSE(build_relation_is_opaque(
      *make_set_operation(duckdb::LogicalOperatorType::LOGICAL_INTERSECT)));
  }

  SECTION("an EXCEPT")
  {
    REQUIRE_FALSE(
      build_relation_is_opaque(*make_set_operation(duckdb::LogicalOperatorType::LOGICAL_EXCEPT)));
  }
}

TEST_CASE("non-projection wrappers hide opaque leaves from the fallback classifier",
          "[dynamic_filter][evidence]")
{
  SECTION("an aggregate over a CTE_REF")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_aggregate_over(make_cte_ref())));
  }

  SECTION("a join over a CTE_REF")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_join_over(make_cte_ref(), make_get())));
  }

  SECTION("a LIMIT over a CTE_REF")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_limit_over(make_cte_ref())));
  }

  SECTION("an ORDER_BY over a CTE_REF")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_order_over(make_cte_ref())));
  }
}
