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

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

/// RAII environment variable override.
struct scoped_env {
  scoped_env(const char* name, const char* value) : _name(name) { setenv(name, value, 1); }
  ~scoped_env() { unsetenv(_name); }
  scoped_env(const scoped_env&)            = delete;
  scoped_env& operator=(const scoped_env&) = delete;

 private:
  const char* _name;
};

class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_eager_agg_pushdown_XXXXXX";
    int fd      = ::mkstemp(tmpl);
    REQUIRE(fd >= 0);
    ::close(fd);
    ::unlink(tmpl);
    _path = tmpl;
  }

  ~scoped_temp_db_path()
  {
    if (!_path.empty()) {
      std::remove(_path.c_str());
      std::remove((_path + ".wal").c_str());
    }
  }

  scoped_temp_db_path(const scoped_temp_db_path&)            = delete;
  scoped_temp_db_path& operator=(const scoped_temp_db_path&) = delete;

  const std::string& path() const { return _path; }

 private:
  std::string _path;
};

/// Parse + bind + optimize, then hand the UNRESOLVED optimized plan to
/// create_plan — the same shape the transparent capture path provides (the
/// eager-agg pass matches bound column refs, which ColumnBindingResolver would
/// have rewritten away; create_plan resolves them itself).
duckdb::unique_ptr<sirius_physical_operator> generate_sirius_plan(Connection& con,
                                                                  const std::string& query)
{
  auto& context = *con.context;

  con.Query("BEGIN TRANSACTION");
  duckdb::unique_ptr<sirius_physical_operator> result;
  try {
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(parser.statements.size() == 1);

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    Optimizer optimizer(*planner.binder, context);
    auto plan = optimizer.Optimize(std::move(planner.plan));

    sirius::planner::sirius_physical_plan_generator gen(context);
    result = gen.create_plan(std::move(plan));
  } catch (...) {
    con.Query("ROLLBACK");
    throw;
  }
  con.Query("COMMIT");
  return result;
}

template <typename Fn>
void for_each_operator(sirius_physical_operator* root, const Fn& fn)
{
  if (!root) { return; }
  fn(root);
  for (auto& child : root->children) {
    for_each_operator(child.get(), fn);
  }
}

std::size_t count_ops(sirius_physical_operator* root, SiriusPhysicalOperatorType type)
{
  std::size_t count = 0;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op->type == type) { count++; }
  });
  return count;
}

sirius_physical_operator* find_first(sirius_physical_operator* root,
                                     SiriusPhysicalOperatorType type)
{
  sirius_physical_operator* found = nullptr;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (found == nullptr && op->type == type) { found = op; }
  });
  return found;
}

void tree_to_string(sirius_physical_operator* root, int depth, std::ostringstream& out)
{
  if (!root) { return; }
  out << std::string(static_cast<size_t>(depth) * 2, ' ')
      << sirius::op::SiriusPhysicalOperatorToString(root->type) << "\n";
  for (auto& child : root->children) {
    tree_to_string(child.get(), depth + 1, out);
  }
}

std::string tree_to_string(sirius_physical_operator* root)
{
  std::ostringstream out;
  tree_to_string(root, 0, out);
  return out.str();
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
  scoped_temp_db_path _db_path;
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
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
  require_fired(kQ13Inner);
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires with a RIGHT join (DuckDB's flipped q13 plan)",
                 "[eager_agg_pushdown][isolated_context]")
{
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
  require_fired("SELECT c_id, count(o_key) FROM ord RIGHT JOIN cust ON o_cid = c_id GROUP BY c_id");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires on INNER joins and SUM/MIN/MAX",
                 "[eager_agg_pushdown][isolated_context]")
{
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
  require_fired(
    "SELECT c_id, sum(o_val), min(o_val), max(o_val) FROM cust JOIN ord ON c_id = o_cid "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires on multi-key equi joins",
                 "[eager_agg_pushdown][isolated_context]")
{
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
  require_fired(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid AND c_grp = o_grp "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - fires on the full q13 (nested second GROUP BY)",
                 "[eager_agg_pushdown][isolated_context]")
{
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
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
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
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
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
  SECTION("group key on the pushed side")
  {
    require_refused("SELECT o_grp, count(o_key) FROM cust JOIN ord ON c_id = o_cid GROUP BY o_grp");
  }
  SECTION("aggregates over both sides")
  {
    require_refused(
      "SELECT c_id, count(o_key), sum(c_grp) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - refuses unsupported join shapes",
                 "[eager_agg_pushdown][isolated_context]")
{
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
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
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
  auto plan =
    generate_sirius_plan(*con, "SELECT count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid");
  INFO(tree_to_string(plan.get()));
  CHECK(count_ops(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY) == 0);
}

//===----------------------------------------------------------------------===//
// Benefit gate + kill switch
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - estimate ratio gate is honored",
                 "[eager_agg_pushdown][isolated_context]")
{
  // Plans generated directly (not via the transparent capture copy) carry
  // optimizer cardinality estimates, so the ratio branch decides. An
  // unattainable threshold must refuse the otherwise-provable q13 shape.
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "1000000000");
  require_refused(kQ13Inner);
}

TEST_CASE_METHOD(eager_agg_pushdown_fixture,
                 "eager agg pushdown - kill switch disables the pass",
                 "[eager_agg_pushdown][isolated_context]")
{
  scoped_env ratio("SIRIUS_EAGER_AGG_MIN_RATIO", "0");
  scoped_env off("SIRIUS_EAGER_AGG_PUSHDOWN", "0");
  require_refused(kQ13Inner);
}
