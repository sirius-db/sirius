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
 * @file test_delim_direct_lowering.cpp
 * @brief The delim-direct lowering pass (sirius_plan_delim_direct): eligible pure-equality
 *        EXISTS / NOT EXISTS DELIM joins collapse to a single direct semi/anti hash join,
 *        ineligible shapes are refused with their pinned typed reason (scalar-aggregate
 *        correlations like TPC-H q2/q17/q20, non-equality correlations like q21), and the
 *        `enable_delim_direct_lowering` knob restores the regular delim lowering.
 */

#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_delim_direct.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_delim_get.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::planner::classify_delim_direct_lowering;

namespace {

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so these tests need an on-disk database rather than :memory:.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_delim_direct_XXXXXX";
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

/// Restores the disabled-optimizer set and closes the planning transaction.
struct scoped_planning_session {
  explicit scoped_planning_session(Connection& con_p) : con(con_p)
  {
    original_disabled = DBConfig::GetConfig(*con.context).options.disabled_optimizers;
    auto& disabled    = DBConfig::GetConfig(*con.context).options.disabled_optimizers;
    // Match the shape-sensitive planner suites: statistics propagation folds tiny test tables
    // into constants and lets the deliminator drop the DELIM_JOINs asserted below.
    disabled.insert(OptimizerType::IN_CLAUSE);
    disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
    disabled.insert(OptimizerType::STATISTICS_PROPAGATION);
    con.Query("BEGIN TRANSACTION");
  }

  ~scoped_planning_session()
  {
    con.Query(committed ? "COMMIT" : "ROLLBACK");
    DBConfig::GetConfig(*con.context).options.disabled_optimizers = original_disabled;
  }

  Connection& con;
  duckdb::set<OptimizerType> original_disabled;
  bool committed = true;
};

/// Parse, plan, and optimize @p query into a resolved logical plan (types + column bindings),
/// ready for the Sirius physical plan generator or for direct classification.
duckdb::unique_ptr<LogicalOperator> build_logical_plan(Connection& con, const std::string& query)
{
  auto& context = *con.context;

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
  return plan;
}

LogicalComparisonJoin* find_delim_join(LogicalOperator* op)
{
  if (!op) { return nullptr; }
  if (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
    return &op->Cast<LogicalComparisonJoin>();
  }
  for (auto& child : op->children) {
    if (auto* found = find_delim_join(child.get())) { return found; }
  }
  return nullptr;
}

/// Walk the physical tree including the DELIM JOIN internals that live outside `children[]`.
template <typename Fn>
void for_each_operator(sirius_physical_operator* root, const Fn& fn)
{
  if (!root) { return; }
  fn(root);
  for (auto& child : root->children) {
    for_each_operator(child.get(), fn);
  }
  if (root->type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      root->type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = root->Cast<sirius::op::sirius_physical_delim_join>();
    for_each_operator(delim.join.get(), fn);
    for_each_operator(delim.distinct_root.get(), fn);
  }
}

std::vector<sirius_physical_operator*> collect(sirius_physical_operator* root,
                                               SiriusPhysicalOperatorType type)
{
  std::vector<sirius_physical_operator*> found;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op->type == type) { found.push_back(op); }
  });
  return found;
}

bool contains_delim_machinery(sirius_physical_operator* root)
{
  return !collect(root, SiriusPhysicalOperatorType::LEFT_DELIM_JOIN).empty() ||
         !collect(root, SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN).empty() ||
         !collect(root, SiriusPhysicalOperatorType::DELIM_SCAN).empty();
}

struct delim_direct_fixture_base {
  /// @param with_sirius_context true builds a SiriusContext (GPU pools — needed for the full
  ///        physical create_plan); false keeps SIRIUS_DISABLE=1 so classification-only tests
  ///        run without touching the GPU.
  explicit delim_direct_fixture_base(bool with_sirius_context)
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    if (with_sirius_context) {
      unsetenv("SIRIUS_DISABLE");
    } else {
      setenv("SIRIUS_DISABLE", "1", 1);
    }
    db = std::make_unique<DuckDB>(_db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    // outer_t is larger so the optimizer keeps inner_t-derived sides as builds where it can.
    con->Query("CREATE TABLE outer_t (id INTEGER, val INTEGER)");
    con->Query(
      "INSERT INTO outer_t VALUES (0,0),(1,3),(2,6),(3,9),(4,12),(5,15),(6,18),(7,21),(8,24),"
      "(9,27),(10,30),(11,33),(12,36),(13,39),(14,42),(15,45),(16,48),(17,51),(18,54),(19,57)");
    con->Query("CREATE TABLE inner_t (rid INTEGER, qty INTEGER)");
    con->Query("INSERT INTO inner_t VALUES (0, 0), (1, 1), (2, 4), (3, 9)");
  }

  ~delim_direct_fixture_base() { unsetenv("SIRIUS_CONFIG_FILE"); }

  /// Generate the full Sirius physical plan (through create_plan) for @p query.
  duckdb::unique_ptr<sirius_physical_operator> generate_sirius_plan(const std::string& query)
  {
    scoped_planning_session session{*con};
    try {
      auto plan = build_logical_plan(*con, query);
      sirius::planner::sirius_physical_plan_generator gen(*con->context);
      return gen.create_plan(std::move(plan));
    } catch (...) {
      session.committed = false;
      throw;
    }
  }

  /// Classify the first DELIM join planned for @p query, with an optional mutation applied to
  /// the logical delim join first (for shapes SQL cannot produce directly). Returns the typed
  /// refusal's log-stable name so a failing CHECK prints the reason, not an enum number.
  template <typename MutateFn>
  std::string classify_query(const std::string& query, const MutateFn& mutate)
  {
    scoped_planning_session session{*con};
    auto plan   = build_logical_plan(*con, query);
    auto* delim = find_delim_join(plan.get());
    REQUIRE(delim != nullptr);
    mutate(*delim);
    return sirius::planner::to_string(classify_delim_direct_lowering(*delim).refusal);
  }

  std::string classify_query(const std::string& query)
  {
    return classify_query(query, [](LogicalComparisonJoin&) {});
  }

  sirius::operator_params& operator_params()
  {
    auto ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(ctx != nullptr);
    return ctx->get_config().get_operator_params();
  }

  // Declared before db/con so the backing file outlives the database.
  scoped_temp_db_path _db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

/// Full fixture: SiriusContext present, physical create_plan available.
struct delim_direct_fixture : delim_direct_fixture_base {
  delim_direct_fixture() : delim_direct_fixture_base(/*with_sirius_context=*/true) {}
};

/// Classification-only fixture: no SiriusContext, no GPU touched. DuckDB's own deliminator is
/// disabled so the delim shapes under classification survive deterministically at toy scale
/// (whether the deliminator dissolves a given tiny delim can vary with table statistics);
/// production-faithful physical coverage lives in the delim_direct_fixture tests.
struct delim_classify_fixture : delim_direct_fixture_base {
  delim_classify_fixture() : delim_direct_fixture_base(/*with_sirius_context=*/false)
  {
    DBConfig::GetConfig(*con->context)
      .options.disabled_optimizers.insert(OptimizerType::DELIMINATOR);
  }
};

constexpr const char* exists_query =
  "SELECT val, count(*) FROM outer_t WHERE EXISTS "
  "(SELECT 1 FROM inner_t WHERE rid = outer_t.id AND qty < 100) GROUP BY val";

constexpr const char* not_exists_query =
  "SELECT val, count(*) FROM outer_t WHERE NOT EXISTS "
  "(SELECT 1 FROM inner_t WHERE rid = outer_t.id AND qty < 100) GROUP BY val";

}  // namespace

TEST_CASE_METHOD(delim_direct_fixture,
                 "delim direct - pure-equality EXISTS lowers to a direct semi hash join",
                 "[delim_direct][isolated_context]")
{
  auto plan = generate_sirius_plan(exists_query);
  REQUIRE(plan);
  CHECK_FALSE(contains_delim_machinery(plan.get()));

  auto joins = collect(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
  REQUIRE(joins.size() == 1);
  auto& hj = joins[0]->Cast<sirius::op::sirius_physical_hash_join>();
  // Always emitted right-family: probe = inner relation, build = outer relation.
  CHECK(hj.join_type == JoinType::RIGHT_SEMI);
  CHECK(hj.conditions.size() == 1);
}

TEST_CASE_METHOD(delim_direct_fixture,
                 "delim direct - pure-equality NOT EXISTS lowers to a direct anti hash join",
                 "[delim_direct][isolated_context]")
{
  auto plan = generate_sirius_plan(not_exists_query);
  REQUIRE(plan);
  CHECK_FALSE(contains_delim_machinery(plan.get()));

  auto joins = collect(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
  REQUIRE(joins.size() == 1);
  auto& hj = joins[0]->Cast<sirius::op::sirius_physical_hash_join>();
  // Always emitted right-family: probe = inner relation, build = outer relation.
  CHECK(hj.join_type == JoinType::RIGHT_ANTI);
}

TEST_CASE_METHOD(delim_direct_fixture,
                 "delim direct - the enable_delim_direct_lowering knob restores the delim plan",
                 "[delim_direct][isolated_context]")
{
  auto& params                        = operator_params();
  params.enable_delim_direct_lowering = false;
  auto restore                        = [&params]() { params.enable_delim_direct_lowering = true; };
  try {
    auto plan = generate_sirius_plan(exists_query);
    REQUIRE(plan);
    CHECK(contains_delim_machinery(plan.get()));
  } catch (...) {
    restore();
    throw;
  }
  restore();
}

TEST_CASE_METHOD(delim_classify_fixture,
                 "delim direct - scalar-aggregate correlations are refused as unsupported join "
                 "types",
                 "[delim_direct][isolated_context]")
{
  // TPC-H q17-shaped: correlated scalar AVG compared with `<`.
  const std::string q17_shape =
    "SELECT t1.id FROM outer_t t1 WHERE t1.val < "
    "(SELECT 0.5 * avg(t2.val) FROM outer_t t2 WHERE t2.id = t1.id)";
  CHECK(classify_query(q17_shape) == "unsupported_join_type");

  // TPC-H q2/q20-shaped: correlated scalar MIN/SUM compared with `=` / `>`.
  const std::string q2_shape =
    "SELECT t1.id FROM outer_t t1 WHERE t1.val = "
    "(SELECT min(t2.val) FROM outer_t t2 WHERE t2.id = t1.id)";
  CHECK(classify_query(q2_shape) == "unsupported_join_type");

  const std::string q20_shape =
    "SELECT t1.id FROM outer_t t1 WHERE t1.id IN (SELECT t2.id FROM outer_t t2 WHERE t2.val > "
    "(SELECT 0.5 * sum(i.qty) FROM inner_t i WHERE i.rid = t2.id))";
  CHECK(classify_query(q20_shape) == "unsupported_join_type");
}

TEST_CASE_METHOD(delim_classify_fixture,
                 "delim direct - non-equality correlations are refused",
                 "[delim_direct][isolated_context]")
{
  // TPC-H q21-shaped: EXISTS with an extra `<>` correlation.
  const std::string q21_shape =
    "SELECT t1.id FROM outer_t t1 WHERE EXISTS "
    "(SELECT 1 FROM outer_t t2 WHERE t2.id = t1.id AND t2.val <> t1.val)";
  CHECK(classify_query(q21_shape) == "non_equality_correlation");
}

TEST_CASE_METHOD(delim_direct_fixture,
                 "delim direct - refused shapes plan identically with the pass on and off",
                 "[delim_direct][isolated_context]")
{
  // Refusal means no plan change: the physical plan must be operator-for-operator identical
  // to the knob-off plan. (At toy scale the pre-existing lowering may itself elide the delim
  // machinery, so "delim ops present" is not the invariant — plan invariance is.)
  auto shape_of = [&](const std::string& sql) {
    auto plan = generate_sirius_plan(sql);
    REQUIRE(plan);
    std::vector<SiriusPhysicalOperatorType> shape;
    for_each_operator(plan.get(),
                      [&shape](sirius_physical_operator* op) { shape.push_back(op->type); });
    return shape;
  };
  const std::string agg_shape =
    "SELECT t1.id FROM outer_t t1 WHERE t1.val < "
    "(SELECT 0.5 * avg(t2.val) FROM outer_t t2 WHERE t2.id = t1.id)";
  const std::string neq_shape =
    "SELECT t1.id FROM outer_t t1 WHERE EXISTS "
    "(SELECT 1 FROM outer_t t2 WHERE t2.id = t1.id AND t2.val <> t1.val)";

  auto& params = operator_params();
  auto restore = [&params]() { params.enable_delim_direct_lowering = true; };
  try {
    auto agg_on                         = shape_of(agg_shape);
    auto neq_on                         = shape_of(neq_shape);
    params.enable_delim_direct_lowering = false;
    CHECK(shape_of(agg_shape) == agg_on);
    CHECK(shape_of(neq_shape) == neq_on);
  } catch (...) {
    restore();
    throw;
  }
  restore();
}

TEST_CASE_METHOD(delim_classify_fixture,
                 "delim direct - eligible shapes classify as eligible",
                 "[delim_direct][isolated_context]")
{
  CHECK(classify_query(exists_query) == "none");
  CHECK(classify_query(not_exists_query) == "none");
}

TEST_CASE_METHOD(delim_classify_fixture,
                 "delim direct - defensive refusals: orientation flip and plain-equal join-back "
                 "over a null-safe correlation",
                 "[delim_direct][isolated_context]")
{
  // Flipping the recorded delim orientation must refuse (the dedup keys would no longer come
  // from the membership-output side).
  CHECK(classify_query(exists_query, [](LogicalComparisonJoin& delim) {
          delim.delim_flipped = !delim.delim_flipped;
        }) == "orientation_mismatch");

  // A null-safe correlation under a null-safe join-back is exact and stays eligible.
  const std::string null_safe_correlation =
    "SELECT t1.id FROM outer_t t1 WHERE EXISTS "
    "(SELECT 1 FROM inner_t i WHERE i.rid IS NOT DISTINCT FROM t1.id)";
  CHECK(classify_query(null_safe_correlation) == "none");

  // The same correlation under a (hand-mutated) plain `=` join-back is not provably
  // NULL-preserving: the original plan drops NULL-keyed outer rows at the join-back, the
  // direct null-safe join would match them.
  CHECK(classify_query(null_safe_correlation, [](LogicalComparisonJoin& delim) {
          for (auto& condition : delim.conditions) {
            if (condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
              condition.comparison = ExpressionType::COMPARE_EQUAL;
            }
          }
        }) == "null_safety");
}

TEST_CASE_METHOD(delim_direct_fixture,
                 "delim direct - nested correlation keeps its delim machinery",
                 "[delim_direct][isolated_context]")
{
  // The outer correlation is consumed inside the inner relation too — collapsing the outer
  // delim would orphan that consumer.
  const std::string nested =
    "SELECT t1.id FROM outer_t t1 WHERE EXISTS "
    "(SELECT 1 FROM inner_t i WHERE i.rid = t1.id AND EXISTS "
    "(SELECT 1 FROM outer_t t3 WHERE t3.id = t1.val))";
  auto plan = generate_sirius_plan(nested);
  REQUIRE(plan);
  CHECK(contains_delim_machinery(plan.get()));
}

namespace {

/// Locate the correlated INNER join under a delim join's sandwich side (through any stacked
/// reference-only projections).
LogicalComparisonJoin& sandwich_inner_join(LogicalComparisonJoin& delim)
{
  auto* node = delim.delim_flipped ? delim.children[0].get() : delim.children[1].get();
  while (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
    node = node->children[0].get();
  }
  return node->Cast<LogicalComparisonJoin>();
}

/// Hand-built two-dedup-key delim (SEMI, unflipped, no projections): dedup keys (#0, #1) of a
/// two-column outer; the correlated join constrains key #0 only; both keys joined back
/// null-safely. Shapes SQL cannot reach (DuckDB's decorrelator constrains every dedup key), for
/// pinning the prove-stage obligations on unconstrained key columns.
duckdb::unique_ptr<LogicalComparisonJoin> make_two_key_delim_with_unconstrained_key()
{
  auto outer   = make_uniq<LogicalDummyScan>(1U);
  outer->types = {LogicalType::INTEGER, LogicalType::INTEGER};

  duckdb::vector<LogicalType> key_types{LogicalType::INTEGER, LogicalType::INTEGER};
  auto delim_get   = make_uniq<LogicalDelimGet>(2U, key_types);
  delim_get->types = key_types;

  auto inner   = make_uniq<LogicalDummyScan>(3U);
  inner->types = {LogicalType::INTEGER};

  auto join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
  join->children.push_back(std::move(delim_get));  // dedup keys on the left (d = 0)
  join->children.push_back(std::move(inner));
  JoinCondition correlated;
  correlated.comparison = ExpressionType::COMPARE_EQUAL;
  correlated.left       = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0U);
  correlated.right      = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0U);
  join->conditions.push_back(std::move(correlated));
  join->types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER};

  auto delim =
    make_uniq<LogicalComparisonJoin>(JoinType::SEMI, LogicalOperatorType::LOGICAL_DELIM_JOIN);
  delim->delim_flipped = false;  // dedup source = children[0], sandwich = children[1]
  delim->children.push_back(std::move(outer));
  delim->children.push_back(std::move(join));
  delim->duplicate_eliminated_columns.push_back(
    make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0U));
  delim->duplicate_eliminated_columns.push_back(
    make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 1U));
  for (duckdb::idx_t key = 0; key < 2; key++) {
    JoinCondition join_back;
    join_back.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
    join_back.left       = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, key);
    join_back.right      = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, key);
    delim->conditions.push_back(std::move(join_back));
  }
  delim->types = {LogicalType::INTEGER, LogicalType::INTEGER};
  return delim;
}

std::string classify_name(LogicalComparisonJoin& delim)
{
  return sirius::planner::to_string(classify_delim_direct_lowering(delim).refusal);
}

}  // namespace

TEST_CASE("delim direct - build-driven sizing exception excludes plain RIGHT joins",
          "[delim_direct]")
{
  // The converter's sizing exception must hold exactly for joins this pass produces
  // (RIGHT_SEMI / RIGHT_ANTI with a published filter) and never for stock shapes: plain RIGHT
  // joins DO receive join-filter pushdown from DuckDB and must stay probe-driven.
  using sirius::op::sirius_physical_hash_join;
  STATIC_REQUIRE(
    sirius_physical_hash_join::right_family_join_sizes_build_driven(JoinType::RIGHT_SEMI, true));
  STATIC_REQUIRE(
    sirius_physical_hash_join::right_family_join_sizes_build_driven(JoinType::RIGHT_ANTI, true));
  STATIC_REQUIRE_FALSE(
    sirius_physical_hash_join::right_family_join_sizes_build_driven(JoinType::RIGHT, true));
  STATIC_REQUIRE_FALSE(
    sirius_physical_hash_join::right_family_join_sizes_build_driven(JoinType::RIGHT_SEMI, false));
  STATIC_REQUIRE_FALSE(
    sirius_physical_hash_join::right_family_join_sizes_build_driven(JoinType::RIGHT_ANTI, false));
  STATIC_REQUIRE_FALSE(
    sirius_physical_hash_join::right_family_join_sizes_build_driven(JoinType::SEMI, true));
}

TEST_CASE_METHOD(delim_classify_fixture,
                 "delim direct - structural refusals are pinned per mutated shape",
                 "[delim_direct][isolated_context]")
{
  // sandwich_shape: the correlated join is not INNER.
  CHECK(classify_query(exists_query, [](LogicalComparisonJoin& delim) {
          sandwich_inner_join(delim).join_type = JoinType::LEFT;
        }) == "sandwich_shape");

  // residual_predicate: the delim join carries an extra ON-clause predicate.
  CHECK(classify_query(exists_query, [](LogicalComparisonJoin& delim) {
          delim.predicate = make_uniq<BoundConstantExpression>(duckdb::Value::BOOLEAN(true));
        }) == "residual_predicate");

  // inner_condition_shape: the correlated condition's dedup-key side is not a valid key column.
  CHECK(classify_query(exists_query, [](LogicalComparisonJoin& delim) {
          auto& join            = sandwich_inner_join(delim);
          const bool delim_left = join.children[0]->type == LogicalOperatorType::LOGICAL_DELIM_GET;
          auto& key_side        = delim_left ? join.conditions[0].left : join.conditions[0].right;
          key_side->Cast<BoundReferenceExpression>().index = 7;
        }) == "inner_condition_shape");

  // join_back_shape: the join-back pairs a dedup key with the WRONG outer column.
  CHECK(classify_query(exists_query, [](LogicalComparisonJoin& delim) {
          auto& outer_side =
            delim.delim_flipped ? delim.conditions[0].right : delim.conditions[0].left;
          outer_side->Cast<BoundReferenceExpression>().index += 1;
        }) == "join_back_shape");

  // residual_delim_consumer: the inner relation still replays delim data.
  CHECK(classify_query(exists_query, [](LogicalComparisonJoin& delim) {
          auto& join = sandwich_inner_join(delim);
          const std::size_t delim_side =
            join.children[0]->type == LogicalOperatorType::LOGICAL_DELIM_GET ? 0 : 1;
          duckdb::vector<LogicalType> key_types{LogicalType::INTEGER};
          auto delim_get   = make_uniq<LogicalDelimGet>(4242U, key_types);
          delim_get->types = key_types;
          duckdb::vector<duckdb::unique_ptr<Expression>> exprs;
          exprs.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0U));
          auto projection = make_uniq<LogicalProjection>(4243U, std::move(exprs));
          projection->children.push_back(std::move(delim_get));
          projection->types             = {LogicalType::INTEGER};
          join.children[1 - delim_side] = std::move(projection);
        }) == "residual_delim_consumer");
}

TEST_CASE(
  "delim direct - unconstrained dedup keys: null-safe join-back is exact, plain '=' "
  "join-back is refused, missing join-back is refused",
  "[delim_direct]")
{
  // Baseline: dedup key #1 has no correlated condition but a null-safe join-back — that
  // join-back is vacuous (a row's own key group always matches itself), so the collapse is
  // exact and the shape is eligible.
  {
    auto delim = make_two_key_delim_with_unconstrained_key();
    CHECK(classify_name(*delim) == "none");
  }

  // The same key under a plain '=' join-back drops NULL-keyed outer rows in the delim plan;
  // the direct join deletes that join-back and cannot reproduce the drop. Must refuse.
  {
    auto delim                      = make_two_key_delim_with_unconstrained_key();
    delim->conditions[1].comparison = ExpressionType::COMPARE_EQUAL;
    CHECK(classify_name(*delim) == "null_safety");
  }

  // A dedup key with NO join-back at all leaves outer rows unpinned. Must refuse.
  {
    auto delim = make_two_key_delim_with_unconstrained_key();
    delim->conditions.pop_back();
    CHECK(classify_name(*delim) == "delim_column_mismatch");
  }
}
