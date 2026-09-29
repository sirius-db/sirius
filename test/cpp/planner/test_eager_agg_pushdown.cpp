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
 * @file test_eager_agg_pushdown.cpp
 * @brief Plan-shape tests for the eager-aggregation-pushdown pass
 *        (src/planner/eager_agg_pushdown_plan_pass.cpp).
 *
 * The pass runs at the head of sirius_physical_plan_generator::create_plan on
 * the (unresolved) optimized logical plan, so these tests hand create_plan the
 * optimizer output directly — exactly what the transparent execution path
 * captures. A fired rewrite shows up as one extra HASH_GROUP_BY that lives
 * BELOW the HASH_JOIN; every refusal case must keep the plan's aggregate count
 * unchanged. GPU-vs-CPU result equality is covered separately by
 * test/cpp/integration/test_gpu_execution_eager_agg.cpp.
 */

#include "op/sirius_physical_operator.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/plan_shape_test_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

using sirius::test::count_ops;
using sirius::test::find_first;
using sirius::test::find_first_logical;
using sirius::test::generate_optimized_logical_plan;
using sirius::test::generate_sirius_plan;
using sirius::test::scoped_setting;
using sirius::test::scoped_temp_db_path;
using sirius::test::tree_to_string;

namespace {

/// Locate the optimizer's upper aggregate — the node the pass matches on. The
/// default plan_generation_options are exactly this suite's: plain optimizer
/// output with the bindings left UNRESOLVED, the same shape the transparent
/// capture path hands create_plan.
LogicalAggregate& require_upper_aggregate(LogicalOperator& logical)
{
  auto* found = find_first_logical(&logical, LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);
  REQUIRE(found != nullptr);
  return found->Cast<LogicalAggregate>();
}

struct eager_agg_pushdown_fixture {
  eager_agg_pushdown_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(_db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    // cust is the preserved / non-pushed side (bare, unfiltered scan); ord is
    // the pushed side with duplicate keys so the pre-aggregation actually
    // reduces rows. o_cid spans cust's full c_id domain [0, 19] on purpose: a
    // narrower domain would let DuckDB's statistics propagation derive a c_id
    // range filter on the cust scan, and the benefit gate (correctly) refuses
    // non-bare preserved sides — which would mask the organic fire shapes
    // this file asserts on.
    con->Query("CREATE TABLE cust (c_id INTEGER, c_grp INTEGER)");
    con->Query("INSERT INTO cust SELECT range, range % 2 FROM range(20)");
    con->Query("CREATE TABLE ord (o_cid INTEGER, o_grp INTEGER, o_key INTEGER, o_val INTEGER)");
    con->Query("INSERT INTO ord SELECT range % 20, range % 2, range, range * 3 FROM range(200)");
    // Same shape, but the join key is a PRIMARY KEY: pre-aggregating it cannot
    // reduce anything, so the benefit gate must decline.
    con->Query("CREATE TABLE ordpk (p_id INTEGER PRIMARY KEY, p_val INTEGER)");
    con->Query("INSERT INTO ordpk SELECT range, range * 3 FROM range(20)");
  }

  ~eager_agg_pushdown_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  /// A fired rewrite adds exactly one grouped aggregate, and it must sit below
  /// the join.
  void require_fired(const std::string& query, std::size_t baseline_aggregates = 1)
  {
    auto plan = generate_sirius_plan(*con, query);
    INFO(tree_to_string(plan.get()));
    CHECK(count_ops(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY) ==
          baseline_aggregates + 1);
    auto* join = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(join != nullptr);
    std::size_t below_join = 0;
    for (auto& child : join->children) {
      below_join += count_ops(child.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY);
    }
    CHECK(below_join == 1);
  }

  void require_refused(const std::string& query, std::size_t baseline_aggregates = 1)
  {
    auto plan = generate_sirius_plan(*con, query);
    INFO(tree_to_string(plan.get()));
    CHECK(count_ops(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY) == baseline_aggregates);
  }

  // Declared before db/con so the backing file outlives the database.
  scoped_temp_db_path _db_path{"sirius_eager_agg_pushdown"};
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

constexpr const char* kQ13Inner =
  "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY c_id";

}  // namespace

//===----------------------------------------------------------------------===//
// Fired shapes
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires on the q13 shape (LEFT join + grouped COUNT)",
                 "[eager_agg_pushdown][isolated_context]")
{
  require_fired(kQ13Inner);
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires with a RIGHT join (DuckDB's flipped q13 plan)",
                 "[eager_agg_pushdown][isolated_context]")
{
  require_fired("SELECT c_id, count(o_key) FROM ord RIGHT JOIN cust ON o_cid = c_id GROUP BY c_id");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires on INNER joins and SUM/MIN/MAX",
                 "[eager_agg_pushdown][isolated_context]")
{
  require_fired(
    "SELECT c_id, sum(o_val), min(o_val), max(o_val) FROM cust JOIN ord ON c_id = o_cid "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires through a pass-through projection above the join",
                 "[eager_agg_pushdown][isolated_context]")
{
  // The pass's projection tracing (trace_to_join_output + the kept_slots /
  // slot_remap / partial_base rewrite) only runs when a pure column-ref
  // PROJECTION actually sits between the aggregate and the join. The direct
  // `cust JOIN ord ... GROUP BY c_id` shapes above do NOT get one on this DuckDB
  // version — the explicit sub-select below does — so assert that input shape
  // here: if DuckDB stops inserting the projection, this fails loudly instead of
  // quietly turning the tracing into dead code.
  const std::string query =
    "SELECT c_id, sum(o_val) FROM ("
    "  SELECT c_id, o_val FROM cust JOIN ord ON c_id = o_cid) GROUP BY c_id";

  auto logical    = generate_optimized_logical_plan(*con, query);
  auto& aggregate = require_upper_aggregate(*logical);
  INFO(logical->ToString());
  REQUIRE(aggregate.children.size() == 1);
  auto& pass_through = *aggregate.children[0];
  REQUIRE(pass_through.type == LogicalOperatorType::LOGICAL_PROJECTION);
  for (auto& slot : pass_through.expressions) {
    CHECK(slot->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF);
  }
  REQUIRE(pass_through.children.size() == 1);
  REQUIRE(pass_through.children[0]->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN);

  require_fired(query);
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires on multi-key equi joins",
                 "[eager_agg_pushdown][isolated_context]")
{
  require_fired(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid AND c_grp = o_grp "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires on the full q13 (nested second GROUP BY)",
                 "[eager_agg_pushdown][isolated_context]")
{
  // Two aggregates in the baseline plan (inner per-customer count + outer
  // histogram); the rewrite adds a third below the join.
  require_fired(
    "SELECT c_count, count(*) FROM ("
    "  SELECT c_id, count(o_key) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cid "
    "  GROUP BY c_id) GROUP BY c_count",
    /*baseline_aggregates=*/2);
}

//===----------------------------------------------------------------------===//
// Refused shapes (correctness gates)
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - refuses non-decomposable or decorated aggregates",
                 "[eager_agg_pushdown][isolated_context]")
{
  SECTION("count(*) counts join rows, not pushed-side rows")
  {
    require_refused("SELECT c_id, count(*) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("avg is not in the decomposable set")
  {
    require_refused(
      "SELECT c_id, avg(o_val) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("DISTINCT aggregates")
  {
    require_refused(
      "SELECT c_id, count(DISTINCT o_key) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("FILTER clauses")
  {
    require_refused(
      "SELECT c_id, count(o_key) FILTER (WHERE o_val > 30) FROM cust LEFT JOIN ord "
      "ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("aggregate over an expression, not a plain column")
  {
    // o_val + o_grp (two columns) so DuckDB's SumRewriter cannot reduce it to
    // sum(col) + C*count(col), which WOULD be a legitimately pushable shape.
    require_refused(
      "SELECT c_id, sum(o_val + o_grp) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - refuses when references would escape the pushed side",
                 "[eager_agg_pushdown][isolated_context]")
{
  SECTION("group key on the pushed side")
  {
    require_refused("SELECT o_grp, count(o_key) FROM cust JOIN ord ON c_id = o_cid GROUP BY o_grp");
  }
  SECTION("aggregates over both sides")
  {
    require_refused(
      "SELECT c_id, count(o_key), sum(c_grp) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("pass-through projection slot is not a plain column ref")
  {
    // Same sub-select shape as the pass-through fire test, but one slot is a
    // computed expression: the matcher refuses the whole projection rather than
    // trace through it.
    const std::string query =
      "SELECT c_id, sum(v) FROM ("
      "  SELECT c_id, o_val * 2 AS v FROM cust JOIN ord ON c_id = o_cid) GROUP BY c_id";

    auto logical    = generate_optimized_logical_plan(*con, query);
    auto& aggregate = require_upper_aggregate(*logical);
    INFO(logical->ToString());
    REQUIRE(aggregate.children.size() == 1);
    auto& pass_through = *aggregate.children[0];
    REQUIRE(pass_through.type == LogicalOperatorType::LOGICAL_PROJECTION);
    bool has_computed_slot = false;
    for (auto& slot : pass_through.expressions) {
      has_computed_slot =
        has_computed_slot || slot->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF;
    }
    CHECK(has_computed_slot);

    require_refused(query);
  }
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - refuses to pre-aggregate an outer join's preserved side",
                 "[eager_agg_pushdown][isolated_context]")
{
  // The aggregate reads the PRESERVED side (cust), so cust would be the pushed
  // side: pre-aggregating it would collapse preserved rows that the outer join
  // must emit one by one. DuckDB flips this into `ord RIGHT JOIN cust`, which
  // the RIGHT mirror of the gate refuses; every other gate passes (the group key
  // sits on the non-pushed side, the non-pushed side is a bare scan, and c_id is
  // not a primary key).
  require_refused(
    "SELECT o_grp, sum(c_grp) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY o_grp");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - refuses unsupported join shapes",
                 "[eager_agg_pushdown][isolated_context]")
{
  SECTION("FULL OUTER join")
  {
    require_refused(
      "SELECT c_id, count(o_key) FROM cust FULL JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("mixed equality + inequality conditions")
  {
    require_refused(
      "SELECT c_id, count(o_key) FROM cust JOIN ord ON c_id = o_cid AND c_grp < o_grp "
      "GROUP BY c_id");
  }
  SECTION("pushed-side join key is an expression")
  {
    require_refused(
      "SELECT c_id, count(o_key) FROM cust JOIN ord ON c_id = o_cid + 1 GROUP BY c_id");
  }
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - refuses ungrouped aggregates",
                 "[eager_agg_pushdown][isolated_context]")
{
  auto plan =
    generate_sirius_plan(*con, "SELECT count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid");
  INFO(tree_to_string(plan.get()));
  CHECK(count_ops(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY) == 0);
}

//===----------------------------------------------------------------------===//
// Benefit gate + kill switch
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - benefit gate refuses a filtered non-pushed side",
                 "[eager_agg_pushdown][isolated_context]")
{
  // The preserved side carries a table filter, so the join is expected to throw
  // most pre-aggregated groups away: the heuristic declines.
  require_refused(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid WHERE c_grp = 1 "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - force setting bypasses the benefit gate",
                 "[eager_agg_pushdown][isolated_context]")
{
  scoped_setting force(*con, "eager_agg_pushdown_force", "true");
  require_fired(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid WHERE c_grp = 1 "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - benefit gate refuses a provably unique pushed key",
                 "[eager_agg_pushdown][isolated_context]")
{
  // ordpk's join key is its PRIMARY KEY, so the lower aggregate would emit one
  // row per input row — a full partition + merge that reduces nothing.
  require_refused("SELECT c_id, sum(p_val) FROM cust JOIN ordpk ON c_id = p_id GROUP BY c_id");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - kill switch disables the pass",
                 "[eager_agg_pushdown][isolated_context]")
{
  scoped_setting off(*con, "enable_eager_agg_pushdown", "false");
  require_refused(kQ13Inner);
}
