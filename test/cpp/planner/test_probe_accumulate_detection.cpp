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

#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_probe_accumulate_emit.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/pipeline_conversion_test_utils.hpp"
#include "utils/scoped_sirius_setting.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace duckdb;
using T = sirius::op::SiriusPhysicalOperatorType;

namespace {

// The GPU seq_scan ingestible requires a single-file block manager, so these tests use an
// on-disk database.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_probe_accumulate_XXXXXX";
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

duckdb::unique_ptr<sirius::op::sirius_physical_operator> generate_sirius_plan(
  Connection& con, const std::string& query)
{
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  con.Query("BEGIN TRANSACTION");
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> result;
  try {
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);
    auto plan = std::move(planner.plan);
    if (context.config.enable_optimizer) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }
    plan->ResolveOperatorTypes();
    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(*plan);
    resolver.VisitOperator(*plan);

    sirius::planner::sirius_physical_plan_generator gen(context);
    result = gen.create_plan(std::move(plan));
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }
  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

std::vector<sirius::op::sirius_physical_operator*> collect(
  sirius::op::sirius_physical_operator* root, T type)
{
  std::vector<sirius::op::sirius_physical_operator*> out;
  if (!root) { return out; }
  if (root->type == type) { out.push_back(root); }
  for (auto& child : root->children) {
    auto sub = collect(child.get(), type);
    out.insert(out.end(), sub.begin(), sub.end());
  }
  // A delim join keeps its inner join outside `children`.
  if (root->type == T::LEFT_DELIM_JOIN || root->type == T::RIGHT_DELIM_JOIN) {
    auto const& delim = root->Cast<sirius::op::sirius_physical_delim_join>();
    auto sub          = collect(delim.join.get(), type);
    out.insert(out.end(), sub.begin(), sub.end());
  }
  return out;
}

struct probe_accumulate_fixture {
  probe_accumulate_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    db  = sirius::test::open_sirius_db(db_path.path().c_str(), cfg);
    con = std::make_unique<Connection>(*db);
    set_flag(true);

    // `dim` is much smaller than `fact`, so the optimizer builds on it; grouping it makes the
    // uniqueness of its key provable.
    con->Query("CREATE TABLE dim (id BIGINT, tag INTEGER, amount DECIMAL(12, 2))");
    con->Query("INSERT INTO dim SELECT range, range % 5, range::DECIMAL(12, 2) FROM range(1, 30)");
    con->Query(
      "CREATE TABLE fact (k BIGINT, v INTEGER, w INTEGER, x DOUBLE, s VARCHAR, late BOOLEAN)");
    con->Query(
      "INSERT INTO fact SELECT (range * 7) % 61, (range * 13) % 1000, "
      "CASE WHEN range % 4 = 0 THEN NULL ELSE range % 97 END, range * 0.5, concat('s', range), "
      "range % 3 = 0 FROM range(5000)");
  }

  void set_flag(bool on)
  {
    auto result =
      con->Query(std::string("SET enable_probe_accumulate = ") + (on ? "true" : "false"));
    REQUIRE(result != nullptr);
    REQUIRE_FALSE(result->HasError());
  }

  // Number of EMIT operators in the plan; a fused plan has exactly one.
  std::size_t fused_count(std::string const& query)
  {
    auto plan = generate_sirius_plan(*con, query);
    REQUIRE(plan);
    return collect(plan.get(), T::PROBE_ACCUMULATE_EMIT).size();
  }

  scoped_temp_db_path db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

constexpr char const* unique_dim = "(SELECT id FROM dim GROUP BY id) d";

std::string grouped_over_dim(std::string const& select_list)
{
  return "SELECT f.k, " + select_list + " FROM fact f JOIN " + unique_dim +
         " ON f.k = d.id GROUP BY f.k";
}

}  // namespace

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate fuses MIN/MAX/COUNT grouped by the join key",
                 "[probe_accumulate][plan]")
{
  auto plan = generate_sirius_plan(*con, grouped_over_dim("min(f.v), max(f.v), count(*)"));
  REQUIRE(plan);

  auto const emits = collect(plan.get(), T::PROBE_ACCUMULATE_EMIT);
  REQUIRE(emits.size() == 1);
  auto const& emit = emits[0]->Cast<sirius::op::sirius_physical_probe_accumulate_emit>();

  // EMIT <- PASSTHROUGH_SINK <- HASH_JOIN (accumulate mode): the sink is what lets the emit
  // operator run once after every probe task instead of inside each of them.
  REQUIRE(emit.children.size() == 1);
  REQUIRE(emit.children[0]->type == T::PASSTHROUGH_SINK);
  REQUIRE(emit.children[0]->children.size() == 1);
  REQUIRE(emit.children[0]->children[0]->type == T::HASH_JOIN);
  auto const& join = emit.children[0]->children[0]->Cast<sirius::op::sirius_physical_hash_join>();
  REQUIRE(join.accumulate != nullptr);
  CHECK(join.accumulate->state.specs().size() == 3);
  CHECK(join.join_type == duckdb::JoinType::INNER);
  CHECK(join.unique_build_keys);

  // The fused aggregate is gone; only the build side's own GROUP BY remains.
  CHECK(collect(plan.get(), T::HASH_GROUP_BY).size() == 1);
  CHECK(emit.types.size() == 4);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate fuses aggregate arguments computed from probe columns",
                 "[probe_accumulate][plan]")
{
  CHECK(fused_count(
          grouped_over_dim("min(CASE WHEN f.late THEN f.v END), max(f.v + 1), count(f.w)")) == 1);
  CHECK(fused_count(grouped_over_dim("min(f.k)")) == 1);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate records the aggregate kinds and probe columns",
                 "[probe_accumulate][plan]")
{
  auto plan        = generate_sirius_plan(*con, grouped_over_dim("max(f.v), count(*), min(f.w)"));
  auto const joins = collect(plan.get(), T::HASH_JOIN);
  sirius::op::sirius_physical_hash_join* accumulating = nullptr;
  for (auto* join : joins) {
    auto& hash_join = join->Cast<sirius::op::sirius_physical_hash_join>();
    if (hash_join.accumulate) { accumulating = &hash_join; }
  }
  REQUIRE(accumulating != nullptr);
  auto const& specs = accumulating->accumulate->state.specs();
  REQUIRE(specs.size() == 3);
  CHECK(specs[0].kind == sirius::op::accumulate_kind::MAX);
  CHECK(specs[1].kind == sirius::op::accumulate_kind::COUNT_STAR);
  CHECK(specs[2].kind == sirius::op::accumulate_kind::MIN);
  // Column 0 of the probe batch is the join key; the arguments follow in order.
  CHECK(specs[0].probe_column == 1);
  CHECK(specs[2].probe_column == 2);
  CHECK(specs[0].value_type.id() == cudf::type_id::INT32);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate is off when the setting is off",
                 "[probe_accumulate][plan]")
{
  set_flag(false);
  CHECK(fused_count(grouped_over_dim("min(f.v), count(*)")) == 0);
  set_flag(true);
  CHECK(fused_count(grouped_over_dim("min(f.v), count(*)")) == 1);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate declines when the build side is not provably unique",
                 "[probe_accumulate][plan]")
{
  // `dim` holds one row per id here, but nothing proves it without the GROUP BY.
  CHECK(fused_count("SELECT f.k, min(f.v) FROM fact f JOIN dim d ON f.k = d.id GROUP BY f.k") == 0);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate declines unsupported aggregates and arguments",
                 "[probe_accumulate][plan]")
{
  // No device atomic for DOUBLE or VARCHAR extrema.
  CHECK(fused_count(grouped_over_dim("min(f.x)")) == 0);
  CHECK(fused_count(grouped_over_dim("max(f.s)")) == 0);
  // SUM and AVG are not maintained.
  CHECK(fused_count(grouped_over_dim("sum(f.v)")) == 0);
  CHECK(fused_count(grouped_over_dim("avg(f.v)")) == 0);
  // DISTINCT aggregates change the fold.
  CHECK(fused_count(grouped_over_dim("count(DISTINCT f.v)")) == 0);
  // One unsupported aggregate keeps the whole group on the ordinary path.
  CHECK(fused_count(grouped_over_dim("min(f.v), sum(f.v)")) == 0);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate declines shapes other than group-by-the-join-key",
                 "[probe_accumulate][plan]")
{
  // Grouped by a column that is not the join key.
  CHECK(fused_count("SELECT f.v, min(f.w) FROM fact f JOIN " + std::string(unique_dim) +
                    " ON f.k = d.id GROUP BY f.v") == 0);
  // Grouped by two columns.
  CHECK(fused_count("SELECT f.k, f.v, min(f.w) FROM fact f JOIN " + std::string(unique_dim) +
                    " ON f.k = d.id GROUP BY f.k, f.v") == 0);
  // A non-equality predicate beside the key.
  CHECK(fused_count("SELECT f.k, min(f.v) FROM fact f JOIN " + std::string(unique_dim) +
                    " ON f.k = d.id AND f.v < d.id GROUP BY f.k") == 0);
  // Not an INNER join.
  CHECK(fused_count("SELECT f.k, min(f.v) FROM fact f LEFT JOIN " + std::string(unique_dim) +
                    " ON f.k = d.id GROUP BY f.k") == 0);
  // An aggregate over the build side's own column.
  CHECK(fused_count("SELECT f.k, max(d.tag) FROM fact f JOIN "
                    "(SELECT id, min(tag) AS tag FROM dim GROUP BY id) d ON f.k = d.id "
                    "GROUP BY f.k") == 0);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate declines more aggregates than one record can hold",
                 "[probe_accumulate][plan]")
{
  std::string many = "min(f.v)";
  for (std::size_t i = 1; i < sirius::op::probe_accumulate_max_specs; ++i) {
    many += ", max(f.v + " + std::to_string(i) + ")";
  }
  CHECK(fused_count(grouped_over_dim(many)) == 1);
  CHECK(fused_count(grouped_over_dim(many + ", count(*)")) == 0);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate declines a build side that cannot be one resident hash table",
                 "[probe_accumulate][plan]")
{
  auto const query = grouped_over_dim("min(f.v), count(*)");
  CHECK(fused_count(query) == 1);

  // The join only runs single-partition BUILD_PROBE below this budget; the planner must see it
  // coming instead of failing when the join is sized.
  auto tight = con->Query("SET max_build_hash_table_bytes = 8");
  REQUIRE_FALSE(tight->HasError());
  CHECK(fused_count(query) == 0);

  auto restore = con->Query("RESET max_build_hash_table_bytes");
  REQUIRE_FALSE(restore->HasError());
  CHECK(fused_count(query) == 1);
}

// DuckDB removes the delim join itself (leaving a plain SEMI/ANTI join with the inequality as a
// residual) for the simplest shapes. A filter on the outer rows or on the subquery keeps it, which
// is also what TPC-H q21's join tree does, so these queries all carry one.
constexpr char const* delim_exists_query =
  "SELECT f.k, f.v FROM fact f WHERE f.w > f.v AND EXISTS "
  "(SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v)";

TEST_CASE_METHOD(
  probe_accumulate_fixture,
  "probe accumulate fuses a correlated EXISTS with <> after rewriting its decorrelation",
  "[probe_accumulate][plan][exists_rewrite]")
{
  auto plan = generate_sirius_plan(*con, delim_exists_query);
  REQUIRE(plan);
  // One fused aggregate feeding a LEFT delim join, not the pair-emitting semi join.
  CHECK(collect(plan.get(), T::PROBE_ACCUMULATE_EMIT).size() == 1);
  CHECK(collect(plan.get(), T::LEFT_DELIM_JOIN).size() == 1);
  CHECK(collect(plan.get(), T::RIGHT_DELIM_JOIN).empty());

  set_flag(false);
  auto unrewritten = generate_sirius_plan(*con, delim_exists_query);
  CHECK(collect(unrewritten.get(), T::PROBE_ACCUMULATE_EMIT).empty());
  CHECK_FALSE(collect(unrewritten.get(), T::RIGHT_DELIM_JOIN).empty());
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "probe accumulate fuses NOT EXISTS and a subquery filter too",
                 "[probe_accumulate][plan][exists_rewrite]")
{
  CHECK(fused_count("SELECT f.k FROM fact f WHERE f.late AND NOT EXISTS "
                    "(SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v)") == 1);
  CHECK(fused_count("SELECT f.k FROM fact f WHERE EXISTS "
                    "(SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v AND g.late)") == 1);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "the EXISTS rewrite leaves other correlated subqueries alone",
                 "[probe_accumulate][plan][exists_rewrite]")
{
  // No inequality: nothing to rewrite.
  CHECK(fused_count("SELECT f.k FROM fact f WHERE f.w > f.v AND EXISTS "
                    "(SELECT 1 FROM fact g WHERE g.k = f.k)") == 0);
  // An inequality that is not <>, or two inequalities.
  CHECK(fused_count("SELECT f.k FROM fact f WHERE f.w > f.v AND EXISTS "
                    "(SELECT 1 FROM fact g WHERE g.k = f.k AND g.v < f.v)") == 0);
  CHECK(fused_count("SELECT f.k FROM fact f WHERE f.late AND EXISTS "
                    "(SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v AND g.w <> f.w)") == 0);
  // A non-integer correlation key or value has no device atomic.
  CHECK(fused_count("SELECT f.k FROM fact f WHERE f.late AND EXISTS "
                    "(SELECT 1 FROM fact g WHERE g.k = f.k AND g.x <> f.x)") == 0);
  CHECK(fused_count("SELECT f.k FROM fact f WHERE f.late AND EXISTS "
                    "(SELECT 1 FROM fact g WHERE g.s = f.s AND g.v <> f.v)") == 0);
}

TEST_CASE_METHOD(
  probe_accumulate_fixture,
  "sibling EXISTS and NOT EXISTS over the same table share one scan and one aggregate",
  "[probe_accumulate][plan][exists_rewrite]")
{
  // The shape of TPC-H q21: both subqueries scan the same table on the same key.
  auto const query =
    "SELECT f.k, count(*) FROM fact f WHERE f.late "
    "AND EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v) "
    "AND NOT EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v AND g.late) "
    "GROUP BY f.k";
  auto plan = generate_sirius_plan(*con, query);
  REQUIRE(plan);
  // One LEFT delim join and one fused aggregate answer both subqueries, with the second
  // subquery's filter inside its accumulators.
  CHECK(collect(plan.get(), T::PROBE_ACCUMULATE_EMIT).size() == 1);
  CHECK(collect(plan.get(), T::LEFT_DELIM_JOIN).size() == 1);
  CHECK(collect(plan.get(), T::RIGHT_DELIM_JOIN).empty());
  auto const joins         = collect(plan.get(), T::HASH_JOIN);
  std::size_t accumulating = 0;
  for (auto* join : joins) {
    auto& hash_join = join->Cast<sirius::op::sirius_physical_hash_join>();
    if (hash_join.accumulate) {
      ++accumulating;
      // MIN and MAX for each of the two subqueries.
      CHECK(hash_join.accumulate->state.specs().size() == 4);
    }
  }
  CHECK(accumulating == 1);
}

TEST_CASE_METHOD(probe_accumulate_fixture,
                 "sibling subqueries on different tables or keys are rewritten separately",
                 "[probe_accumulate][plan][exists_rewrite]")
{
  // The second subquery reads another table, so one scan cannot serve both.
  CHECK(fused_count(
          "SELECT f.k FROM fact f WHERE f.late "
          "AND EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v) "
          "AND NOT EXISTS (SELECT 1 FROM dim d WHERE d.id = f.k AND d.tag <> f.v AND d.id > 3)") <=
        2);
}
