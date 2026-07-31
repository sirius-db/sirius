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
 * Unit tests for `planner/dynamic_filter/build_filter_evidence.hpp`.
 *
 * `build_subtree_is_filtering` mirrors DuckDB's join-filter evidence rules for filtered scans,
 * filters, top-N operators, and containing subtrees; the parity suite compares both implementations
 * on optimized plans. `build_relation_is_derived` recognizes only delim-scan and CTE-reference
 * roots with optional projection wrappers; the tests pin that root-down boundary.
 */

#include "planner/dynamic_filter/build_filter_evidence.hpp"

#include <catch.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_cteref.hpp>
#include <duckdb/planner/operator/logical_delim_get.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_top_n.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <utility>

namespace {

using sirius::planner::build_relation_is_derived;
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

TEST_CASE("a derived-leaf root is a derived relation", "[dynamic_filter][evidence]")
{
  SECTION("a DELIM_GET root") { REQUIRE(build_relation_is_derived(*make_delim_get())); }

  SECTION("a CTE_REF root") { REQUIRE(build_relation_is_derived(*make_cte_ref())); }
}

TEST_CASE("projection wrappers are transparent to derivation", "[dynamic_filter][evidence]")
{
  // Projection expressions do not affect whether the input is a derived leaf.
  SECTION("a projection over a DELIM_GET")
  {
    auto const projection = make_projection_over(make_delim_get());
    REQUIRE(build_relation_is_derived(*projection));
  }

  SECTION("stacked projections over a CTE_REF")
  {
    auto const projection = make_projection_over(make_projection_over(make_cte_ref()));
    REQUIRE(build_relation_is_derived(*projection));
  }
}

TEST_CASE("derivation is orthogonal to filtering", "[dynamic_filter][evidence]")
{
  // The two predicates disagree on purpose in both directions.
  SECTION("a base-table GET is never derived, filtered or not")
  {
    REQUIRE_FALSE(build_relation_is_derived(*make_get()));
    REQUIRE_FALSE(build_relation_is_derived(*make_filtered_get()));
  }

  SECTION("the mirror carries no evidence at a derived leaf")
  {
    // The opacity this predicate exists for: the mirror bottoms out at the childless reference.
    REQUIRE_FALSE(build_subtree_is_filtering(*make_delim_get()));
    REQUIRE_FALSE(build_subtree_is_filtering(*make_cte_ref()));
  }
}

TEST_CASE("operators in the mirror's jurisdiction are not derived", "[dynamic_filter][evidence]")
{
  SECTION("a FILTER over a GET")
  {
    duckdb::LogicalFilter filter;
    filter.children.push_back(make_get());
    REQUIRE_FALSE(build_relation_is_derived(filter));
  }

  SECTION("a TOP_N")
  {
    duckdb::LogicalTopN top_n({}, /*limit=*/10, /*offset=*/0);
    REQUIRE_FALSE(build_relation_is_derived(top_n));
  }
}

TEST_CASE("derivation is root-down, not any-descendant", "[dynamic_filter][evidence]")
{
  SECTION("a group-less aggregate over a projected CTE_REF is not derived")
  {
    // The aggregate presents a new relation, even though its child is derived.
    auto aggregate = duckdb::make_uniq<duckdb::LogicalAggregate>(
      /*group_index=*/4,
      /*aggregate_index=*/5,
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
    aggregate->children.push_back(make_projection_over(make_cte_ref()));
    REQUIRE_FALSE(build_relation_is_derived(*aggregate));
  }

  SECTION("a comparison join with a CTE_REF child is not derived")
  {
    // The join presents a new relation rather than forwarding either child.
    auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
    join->children.push_back(make_cte_ref());
    join->children.push_back(make_get());
    REQUIRE_FALSE(build_relation_is_derived(*join));
  }
}

TEST_CASE("a childless projection is not derived", "[dynamic_filter][evidence]")
{
  auto const projection = duckdb::make_uniq<duckdb::LogicalProjection>(
    /*table_index=*/6, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
  REQUIRE_FALSE(build_relation_is_derived(*projection));
}
