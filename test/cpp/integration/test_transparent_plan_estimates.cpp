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

// The optimizer hook's plan capture carries the cardinality estimates of the plan DuckDB keeps.

#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <vector>

namespace {

class PlanEstimatesFixture : public sirius::test::GpuExecutionFixture {
 public:
  PlanEstimatesFixture()
  {
    run_ok("CREATE TABLE big AS SELECT i AS k, i % 100 AS v FROM range(100000) t(i);");
    run_ok("CREATE TABLE small AS SELECT i AS k FROM range(1000) t(i);");
    run_ok("CREATE TABLE tiny AS SELECT i AS x FROM range(3) t(i);");
    run_ok("CHECKPOINT;");
  }
};

void preorder(duckdb::LogicalOperator& op, std::vector<duckdb::LogicalOperator*>& out)
{
  out.push_back(&op);
  for (auto& child : op.children) {
    preorder(*child, out);
  }
}

}  // namespace

TEST_CASE_METHOD(PlanEstimatesFixture,
                 "transparent plan capture keeps DuckDB's cardinality estimates",
                 "[integration][transparent][cardinality]")
{
  auto const sql = GENERATE(
    "SELECT big.k, small.k FROM big JOIN small ON big.k = small.k WHERE big.v < 5",
    "SELECT big.k, small.k, tiny.x FROM big JOIN small ON big.k = small.k, tiny WHERE big.v < 5");
  INFO(sql);

  auto original = con->ExtractPlan(sql);
  auto state    = duckdb::get_sirius_connection_state(*con->context);
  REQUIRE(state);
  auto captured = state->take_captured_plan_if_current();
  REQUIRE(captured);

  std::vector<duckdb::LogicalOperator*> expected;
  std::vector<duckdb::LogicalOperator*> actual;
  preorder(*original, expected);
  preorder(*captured, actual);
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    INFO("node " << i << ": " << duckdb::LogicalOperatorToString(expected[i]->type));
    CHECK(actual[i]->type == expected[i]->type);
    CHECK(actual[i]->has_estimated_cardinality == expected[i]->has_estimated_cardinality);
    CHECK(actual[i]->estimated_cardinality == expected[i]->estimated_cardinality);
  }
}

TEST_CASE_METHOD(PlanEstimatesFixture,
                 "transparent join planned from DuckDB's estimates matches CPU",
                 "[integration][transparent][cardinality]")
{
  compare_gpu_vs_cpu(
    "SELECT big.k, small.k FROM big JOIN small ON big.k = small.k WHERE big.v < 5");
}
