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

#include "planner/join_implied_filters.hpp"

#include <catch.hpp>
#include <duckdb/common/multi_file/multi_file_function.hpp>
#include <duckdb/planner/bound_result_modifier.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_limit.hpp>

#include <cstdint>
#include <optional>
#include <string>

using sirius::planner::drop_join_implied_filters;

namespace {

// A one-column Parquet scan of key `k` whose footer statistics give [min, max], optionally
// filtered with `k <= filter_max`.
duckdb::unique_ptr<duckdb::LogicalGet> make_scan(duckdb::idx_t table_index,
                                                 std::int64_t min,
                                                 std::int64_t max,
                                                 std::optional<std::int64_t> filter_max = {},
                                                 bool with_statistics                   = true)
{
  auto const text = duckdb::LogicalType::VARCHAR;
  auto bind       = duckdb::make_uniq<duckdb::MultiFileBindData>();
  if (with_statistics) {
    auto bounds =
      duckdb::Value::MAP(text,
                         text,
                         {duckdb::Value("min"), duckdb::Value("max")},
                         {duckdb::Value(std::to_string(min)), duckdb::Value(std::to_string(max))});
    bind->file_options.column_statistics = duckdb::Value::MAP(
      text, duckdb::LogicalType::MAP(text, text), {duckdb::Value("k")}, {std::move(bounds)});
  }
  auto get = duckdb::make_uniq<duckdb::LogicalGet>(
    duckdb::TableIndex(table_index),
    duckdb::BoundTableFunction(duckdb::TableFunction()),
    std::move(bind),
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT},
    duckdb::vector<duckdb::Identifier>{"k"});
  get->AddColumnId(0);
  if (filter_max) {
    get->table_filters.PushFilter(
      duckdb::ProjectionIndex(0),
      duckdb::ExpressionFilter::CreateComparisonFilter(
        duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO, duckdb::Value::BIGINT(*filter_max)));
  }
  return get;
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_join(
  duckdb::JoinType join_type,
  duckdb::unique_ptr<duckdb::LogicalOperator> left,
  duckdb::unique_ptr<duckdb::LogicalOperator> right)
{
  auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  join->children.push_back(std::move(left));
  join->children.push_back(std::move(right));
  join->conditions.emplace_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0),
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0),
    duckdb::ExpressionType::COMPARE_EQUAL);
  join->ResolveOperatorTypes();
  return join;
}

bool has_filter(duckdb::LogicalGet const& get)
{
  return get.table_filters.HasFilter(duckdb::ProjectionIndex(0));
}

constexpr std::int64_t k_customers = 150'000'000;

}  // namespace

TEST_CASE("join implied filters - the join's own range filter on the other side goes",
          "[planner][join_implied_filters]")
{
  // TPC-H q18: orders' customer keys stop one short of the customer table's. The statistics give
  // no null count, so the keys may be NULL, which never match.
  for (auto const type : {duckdb::JoinType::INNER, duckdb::JoinType::SEMI}) {
    auto customer   = make_scan(1, 1, k_customers, k_customers - 1);
    auto const* raw = customer.get();
    auto join       = make_join(type, make_scan(0, 1, k_customers - 1), std::move(customer));
    drop_join_implied_filters(*join);
    CHECK_FALSE(has_filter(*raw));
  }
}

TEST_CASE("join implied filters - filters that can remove matching rows stay",
          "[planner][join_implied_filters]")
{
  auto customer   = make_scan(1, 1, k_customers, 1000);
  auto const* raw = customer.get();
  auto join       = make_join(duckdb::JoinType::INNER, make_scan(0, 1, 2000), std::move(customer));
  drop_join_implied_filters(*join);
  CHECK(has_filter(*raw));
}

TEST_CASE("join implied filters - implied filters that remove many rows stay",
          "[planner][join_implied_filters]")
{
  // Implied, but it removes 99% of the scan before the join would.
  auto customer   = make_scan(1, 1, k_customers, k_customers / 100);
  auto const* raw = customer.get();
  auto join =
    make_join(duckdb::JoinType::INNER, make_scan(0, 1, k_customers / 100), std::move(customer));
  drop_join_implied_filters(*join);
  CHECK(has_filter(*raw));
}

TEST_CASE("join implied filters - implied filters on ranges that barely overlap stay",
          "[planner][join_implied_filters]")
{
  // Equally wide ranges that share ten keys: the filter keeps 1% of the scan.
  auto other      = make_scan(1, 991, 1990, 1000);
  auto const* raw = other.get();
  auto join       = make_join(duckdb::JoinType::INNER, make_scan(0, 1, 1000), std::move(other));
  drop_join_implied_filters(*join);
  CHECK(has_filter(*raw));
}

TEST_CASE("join implied filters - a scan under a SINGLE join keeps its filter",
          "[planner][join_implied_filters]")
{
  // Rows the filter removes could match two scalar-subquery rows, which throws.
  auto customer   = make_scan(1, 1, k_customers, k_customers - 1);
  auto const* raw = customer.get();
  auto single     = make_join(duckdb::JoinType::SINGLE, std::move(customer), make_scan(2, 1, 10));
  auto join =
    make_join(duckdb::JoinType::INNER, make_scan(0, 1, k_customers - 1), std::move(single));
  drop_join_implied_filters(*join);
  CHECK(has_filter(*raw));
}

TEST_CASE("join implied filters - only joins that drop unmatched rows qualify",
          "[planner][join_implied_filters]")
{
  for (auto const type : {duckdb::JoinType::LEFT, duckdb::JoinType::ANTI, duckdb::JoinType::MARK}) {
    auto customer   = make_scan(1, 1, k_customers, k_customers - 1);
    auto const* raw = customer.get();
    auto join       = make_join(type, make_scan(0, 1, k_customers - 1), std::move(customer));
    drop_join_implied_filters(*join);
    CHECK(has_filter(*raw));
  }
}

TEST_CASE("join implied filters - a scan under a LIMIT keeps its filter",
          "[planner][join_implied_filters]")
{
  // Without the filter the LIMIT could pick other rows.
  auto customer   = make_scan(1, 1, k_customers, k_customers - 1);
  auto const* raw = customer.get();
  auto limit = duckdb::make_uniq<duckdb::LogicalLimit>(duckdb::BoundLimitNode::ConstantValue(10),
                                                       duckdb::BoundLimitNode());
  limit->children.push_back(std::move(customer));
  auto join =
    make_join(duckdb::JoinType::INNER, make_scan(0, 1, k_customers - 1), std::move(limit));
  drop_join_implied_filters(*join);
  CHECK(has_filter(*raw));
}

TEST_CASE("join implied filters - scans without footer statistics keep their filters",
          "[planner][join_implied_filters]")
{
  SECTION("the other side has none")
  {
    auto customer   = make_scan(1, 1, k_customers, k_customers - 1);
    auto const* raw = customer.get();
    auto join       = make_join(duckdb::JoinType::INNER,
                          make_scan(0, 1, k_customers - 1, {}, /*with_statistics=*/false),
                          std::move(customer));
    drop_join_implied_filters(*join);
    CHECK(has_filter(*raw));
  }
  SECTION("the filtered side has none")
  {
    auto customer   = make_scan(1, 1, k_customers, k_customers - 1, /*with_statistics=*/false);
    auto const* raw = customer.get();
    auto join =
      make_join(duckdb::JoinType::INNER, make_scan(0, 1, k_customers - 1), std::move(customer));
    drop_join_implied_filters(*join);
    CHECK(has_filter(*raw));
  }
}
