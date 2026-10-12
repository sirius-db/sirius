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

#include "transparent/sirius_optimizer_extension.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace {

// A selective filter on a large table, a join with a small one and a cross product, so the
// optimizer sets estimates on filtered scans, joins and cross products.
constexpr char const* estimate_query =
  "SELECT big.k, small.k, tiny.x FROM big JOIN small ON big.k = small.k, tiny WHERE big.v < 5";

struct plan_fixture {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  plan_fixture()
  {
    for (auto const* sql : {
           "CREATE TABLE big AS SELECT i AS k, i % 100 AS v FROM range(100000) t(i)",
           "CREATE TABLE small AS SELECT i AS k FROM range(1000) t(i)",
           "CREATE TABLE tiny AS SELECT i AS x FROM range(3) t(i)",
         }) {
      auto result = con.Query(sql);
      REQUIRE_FALSE(result->HasError());
    }
  }

  duckdb::unique_ptr<duckdb::LogicalOperator> copy(duckdb::LogicalOperator const& plan)
  {
    duckdb::unique_ptr<duckdb::LogicalOperator> result;
    con.context->RunFunctionInTransaction(
      [&] { result = sirius::transparent::copy_logical_plan(plan, *con.context); });
    return result;
  }

  duckdb::unique_ptr<duckdb::LogicalOperator> raw_copy(duckdb::LogicalOperator const& plan)
  {
    duckdb::unique_ptr<duckdb::LogicalOperator> result;
    con.context->RunFunctionInTransaction([&] { result = plan.Copy(*con.context); });
    return result;
  }
};

void preorder(duckdb::LogicalOperator& op, std::vector<duckdb::LogicalOperator*>& out)
{
  out.push_back(&op);
  for (auto& child : op.children) {
    preorder(*child, out);
  }
}

std::vector<duckdb::LogicalOperator*> preorder(duckdb::LogicalOperator& op)
{
  std::vector<duckdb::LogicalOperator*> out;
  preorder(op, out);
  return out;
}

bool contains(std::vector<duckdb::LogicalOperator*> const& nodes, duckdb::LogicalOperatorType type)
{
  return std::any_of(
    nodes.begin(), nodes.end(), [type](auto const* op) { return op->type == type; });
}

}  // namespace

TEST_CASE_METHOD(plan_fixture,
                 "copy_logical_plan preserves DuckDB's cardinality estimates",
                 "[planner][transparent][cardinality]")
{
  auto plan = con.ExtractPlan(estimate_query);
  REQUIRE(plan->has_estimated_cardinality);
  // The optimizer estimates every node of this plan. Unset the root so the copy also covers a
  // node that computes its estimate lazily.
  plan->has_estimated_cardinality = false;
  plan->estimated_cardinality     = 7;
  auto clone                      = copy(*plan);
  REQUIRE(clone);

  auto const original = preorder(*plan);
  auto const copied   = preorder(*clone);
  REQUIRE(original.size() == copied.size());
  REQUIRE(contains(original, duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN));
  REQUIRE(contains(original, duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT));

  for (std::size_t i = 0; i < original.size(); ++i) {
    INFO("node " << i << ": " << duckdb::LogicalOperatorToString(original[i]->type));
    CHECK(copied[i]->type == original[i]->type);
    CHECK(copied[i]->has_estimated_cardinality == original[i]->has_estimated_cardinality);
    CHECK(copied[i]->estimated_cardinality == original[i]->estimated_cardinality);
  }
}

TEST_CASE_METHOD(plan_fixture,
                 "LogicalOperator::Copy drops cardinality estimates",
                 "[planner][transparent][cardinality]")
{
  // copy_cardinality_estimates exists because of this. If a DuckDB upgrade makes Copy() keep the
  // estimates, this test fails and the helper can be removed.
  auto plan  = con.ExtractPlan(estimate_query);
  auto clone = raw_copy(*plan);
  REQUIRE(clone);

  auto const copied = preorder(*clone);
  CHECK(std::none_of(
    copied.begin(), copied.end(), [](auto const* op) { return op->has_estimated_cardinality; }));
}

TEST_CASE_METHOD(plan_fixture,
                 "copy_cardinality_estimates skips subtrees whose shape differs",
                 "[planner][transparent][cardinality]")
{
  auto plan  = con.ExtractPlan(estimate_query);
  auto clone = raw_copy(*plan);
  REQUIRE(clone);

  auto const copied = preorder(*clone);
  auto cross        = std::find_if(copied.begin(), copied.end(), [](auto const* op) {
    return op->type == duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
  });
  REQUIRE(cross != copied.end());
  auto& mismatched = **cross;
  mismatched.children.pop_back();
  mismatched.estimated_cardinality     = 12345;
  mismatched.has_estimated_cardinality = true;

  CHECK_FALSE(sirius::transparent::copy_cardinality_estimates(*plan, *clone));

  // Nodes above the mismatch are still copied, the mismatched node keeps its own values.
  CHECK(clone->has_estimated_cardinality == plan->has_estimated_cardinality);
  CHECK(clone->estimated_cardinality == plan->estimated_cardinality);
  CHECK(mismatched.has_estimated_cardinality);
  CHECK(mismatched.estimated_cardinality == 12345);

  SECTION("unrelated plans")
  {
    auto other = con.ExtractPlan("SELECT 42");
    CHECK_FALSE(sirius::transparent::copy_cardinality_estimates(*plan, *other));
    CHECK_FALSE(sirius::transparent::copy_cardinality_estimates(*other, *plan));
  }
}
