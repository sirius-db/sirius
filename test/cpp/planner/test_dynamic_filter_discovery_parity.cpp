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
 * @file test_dynamic_filter_discovery_parity.cpp
 * @brief Compares Sirius scan-target discovery with DuckDB's join-filter pushdown walk.
 *
 * The DuckDB oracle uses a pre-binding-resolver plan copy; Sirius converts the original
 * optimized plan. Opaque-build fallback is covered by the evidence, plan-shape, and
 * integration suites rather than this parity suite.
 */

// Reaching a join operator through these headers instantiates `vector<join_condition>`'s
// destructor, which needs the AST node definition.
#include "expression/ast/node.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "planner/dynamic_filter/build_filter_evidence.hpp"
#include "planner/dynamic_filter/duckdb_join_filter_candidate_adapter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/join_filter_pushdown_optimizer.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

// GPU-native sequential scans require the single-file block manager used by an on-disk database.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_discovery_parity_XXXXXX";
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

class dynamic_filter_on_guard {
 public:
  explicit dynamic_filter_on_guard(Connection& con)
    : _state(con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state"))
  {
    REQUIRE(_state != nullptr);
    auto& params                 = _state->get_config().get_operator_params();
    _original                    = params.enable_dynamic_filter;
    params.enable_dynamic_filter = true;
  }

  ~dynamic_filter_on_guard()
  {
    _state->get_config().get_operator_params().enable_dynamic_filter = _original;
  }

  dynamic_filter_on_guard(const dynamic_filter_on_guard&)            = delete;
  dynamic_filter_on_guard& operator=(const dynamic_filter_on_guard&) = delete;

 private:
  duckdb::shared_ptr<duckdb::SiriusContext> _state;
  bool _original = true;
};

// Logical oracle inputs and the physical Sirius plan for one query.
struct parity_case {
  duckdb::unique_ptr<duckdb::LogicalOperator> oracle_plan;
  std::vector<bool> original_join_had_pushdown;
  duckdb::unique_ptr<sirius_physical_operator> physical;
};

void collect_comparison_joins(duckdb::LogicalOperator& op,
                              std::vector<duckdb::LogicalComparisonJoin*>& out)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    out.push_back(&op.Cast<duckdb::LogicalComparisonJoin>());
  }
  for (auto& child : op.children) {
    collect_comparison_joins(*child, out);
  }
}

std::vector<duckdb::LogicalComparisonJoin*> comparison_joins_of(duckdb::LogicalOperator& plan)
{
  std::vector<duckdb::LogicalComparisonJoin*> joins;
  collect_comparison_joins(plan, joins);
  return joins;
}

// Plan and optimize while preserving the query's join order, build sides, and test shape.
duckdb::unique_ptr<duckdb::LogicalOperator> optimize_query(
  Connection& con, const std::string& query, const std::vector<OptimizerType>& also_disabled = {})
{
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  disabled.insert(OptimizerType::STATISTICS_PROPAGATION);
  disabled.insert(OptimizerType::JOIN_ORDER);
  disabled.insert(OptimizerType::BUILD_SIDE_PROBE_SIDE);
  for (auto const optimizer : also_disabled) {
    disabled.insert(optimizer);
  }

  con.Query("BEGIN TRANSACTION");
  duckdb::unique_ptr<duckdb::LogicalOperator> plan;
  try {
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    plan = std::move(planner.plan);
    if (context.config.enable_optimizer) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }
    plan->ResolveOperatorTypes();
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }
  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return plan;
}

parity_case plan_parity_case(Connection& con,
                             const std::string& query,
                             const std::vector<OptimizerType>& also_disabled = {})
{
  auto& context = *con.context;
  auto original = optimize_query(con, query, also_disabled);

  parity_case result;
  con.Query("BEGIN TRANSACTION");
  try {
    // Copy before binding resolution to retain binding-space expressions. Serialization strips
    // pushdown metadata so the oracle derives targets from the plan structure.
    // Deserialization re-binds each scan's catalog entry, so the copy needs an active transaction.
    result.oracle_plan = original->Copy(context);
    for (auto const* join : comparison_joins_of(*original)) {
      result.original_join_had_pushdown.push_back(join->filter_pushdown != nullptr);
    }

    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(*original);
    resolver.VisitOperator(*original);

    sirius::planner::sirius_physical_plan_generator gen(context);
    result.physical = gen.create_plan(std::move(original));
  } catch (...) {
    con.Query("ROLLBACK");
    throw;
  }
  con.Query("COMMIT");
  return result;
}

// Return the binding DuckDB tracks for a plain reference or supported integral cast chain.
std::optional<duckdb::ColumnBinding> oracle_probe_binding(duckdb::Expression const& expr)
{
  if (expr.return_type.IsNested()) { return std::nullopt; }
  if (expr.return_type.id() == duckdb::LogicalTypeId::INTERVAL) { return std::nullopt; }
  switch (expr.GetExpressionClass()) {
    case duckdb::ExpressionClass::BOUND_COLUMN_REF:
      return expr.Cast<duckdb::BoundColumnRefExpression>().binding;
    case duckdb::ExpressionClass::BOUND_CAST: {
      auto const& cast = expr.Cast<duckdb::BoundCastExpression>();
      auto const& src  = cast.child->return_type;
      auto const& tgt  = cast.return_type;
      if (!src.IsIntegral() || !tgt.IsIntegral()) { return std::nullopt; }
      if (GetTypeIdSize(src.InternalType()) > GetTypeIdSize(duckdb::PhysicalType::INT64) ||
          GetTypeIdSize(tgt.InternalType()) > GetTypeIdSize(duckdb::PhysicalType::INT64)) {
        return std::nullopt;
      }
      return oracle_probe_binding(*cast.child);
    }
    default: return std::nullopt;
  }
}

// Target found by DuckDB's walk for one join condition.
struct oracle_key_target {
  std::size_t condition_index = 0;
  std::string table_name;
  std::size_t output_ordinal = 0;
};

// Run DuckDB's walk independently for each condition and resolve the target scan output ordinal.
std::vector<oracle_key_target> oracle_targets_for_join(duckdb::LogicalComparisonJoin& join_copy)
{
  std::vector<oracle_key_target> found;
  for (std::size_t condition_index = 0; condition_index < join_copy.conditions.size();
       ++condition_index) {
    auto const& condition = join_copy.conditions[condition_index];
    if (condition.comparison != duckdb::ExpressionType::COMPARE_EQUAL) { continue; }
    auto const binding = oracle_probe_binding(*condition.left);
    if (!binding.has_value()) { continue; }

    duckdb::JoinFilterPushdownColumn column;
    column.probe_column_index = *binding;
    duckdb::vector<duckdb::PushdownFilterTarget> targets;
    duckdb::JoinFilterPushdownOptimizer::GetPushdownFilterTargets(
      *join_copy.children[0], {column}, targets);

    for (auto& target : targets) {
      auto& get           = target.get;
      auto const bindings = get.GetColumnBindings();
      REQUIRE(target.columns.size() == 1);
      std::optional<std::size_t> output_ordinal;
      for (std::size_t i = 0; i < bindings.size(); ++i) {
        if (bindings[i] == target.columns[0].probe_column_index) {
          output_ordinal = i;
          break;
        }
      }
      REQUIRE(output_ordinal.has_value());
      auto const table = get.GetTable();
      found.push_back(oracle_key_target{.condition_index = condition_index,
                                        .table_name      = table ? table->name : std::string{},
                                        .output_ordinal  = *output_ordinal});
    }
  }
  return found;
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

std::vector<sirius::op::sirius_physical_hash_join*> hash_joins_of(sirius_physical_operator* root)
{
  std::vector<sirius::op::sirius_physical_hash_join*> joins;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
      joins.push_back(&op->Cast<sirius::op::sirius_physical_hash_join>());
    }
  });
  return joins;
}

// Sirius scan binding in the same coordinate spaces as oracle_key_target.
struct sirius_scan_binding {
  std::size_t condition_index = 0;
  std::string table_name;
  std::size_t push_ordinal = 0;

  [[nodiscard]] bool operator==(sirius_scan_binding const&) const = default;
};

// Resolve the unique GPU scan that owns filter_set and return its base table name.
std::string owning_scan_table_of(
  sirius_physical_operator* root,
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> const& filter_set)
{
  std::vector<sirius::op::scan::duckdb_native_ingestible_table_info const*> owners;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op->type != SiriusPhysicalOperatorType::GPU_SCAN) { return; }
    auto const& scan = op->Cast<sirius::op::scan::sirius_gpu_scan_operator>();
    auto const* info = dynamic_cast<sirius::op::scan::duckdb_native_ingestible_table_info const*>(
      &scan.get_ingestible().table_info());
    REQUIRE(info != nullptr);
    if (info->sirius_dynamic_filters.get() == filter_set.get()) { owners.push_back(info); }
  });
  REQUIRE(owners.size() == 1);
  return owners[0]->table_name;
}

std::vector<sirius_scan_binding> scan_bindings_of(sirius::op::sirius_physical_hash_join const& hj,
                                                  sirius_physical_operator* root)
{
  std::vector<sirius_scan_binding> bindings;
  auto const& plan = hj.dynamic_filter_plan();
  for (auto const& target : plan.probe_targets()) {
    if (target.route_class != sirius::op::dynamic_filter_route_class::scan) { continue; }
    auto table_name = owning_scan_table_of(root, target.filter_set);
    for (auto const& binding : target.key_bindings) {
      bindings.push_back(sirius_scan_binding{
        .condition_index =
          plan.admitted_keys().at(binding.admitted_key_index).planner_condition_index,
        .table_name   = table_name,
        .push_ordinal = binding.channel_push_ordinal});
    }
  }
  return bindings;
}

struct discovery_parity_fixture {
  discovery_parity_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(_db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    // big_left is larger so the syntactic build sides the held-back optimizers preserve are also
    // the plausible ones.
    con->Query("CREATE TABLE big_left (id INTEGER, val INTEGER)");
    con->Query(
      "INSERT INTO big_left VALUES (0,0),(1,3),(2,6),(3,9),(4,12),(5,15),(6,18),(7,21),(8,24),"
      "(9,27),(10,30),(11,33),(12,36),(13,39),(14,42),(15,45),(16,48),(17,51),(18,54),(19,57)");
    con->Query("CREATE TABLE small_right (rid INTEGER, other INTEGER)");
    con->Query("INSERT INTO small_right VALUES (0, 0), (1, 1)");
    con->Query("CREATE TABLE small_c (ckey INTEGER, cother INTEGER)");
    con->Query("INSERT INTO small_c VALUES (0, 0), (1, 1)");
  }

  ~discovery_parity_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  // Declared before db/con so the backing file outlives the database.
  scoped_temp_db_path _db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - probe-scan bind matches DuckDB's per-key walk",
                 "[dynamic_filter][parity][isolated_context]")
{
  dynamic_filter_on_guard filter_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid "
                            "WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);
  REQUIRE(oracle[0].condition_index == 0);
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  auto const bindings = scan_bindings_of(*physical_joins[0], c.physical.get());
  REQUIRE(bindings == std::vector<sirius_scan_binding>{{.condition_index = 0,
                                                        .table_name      = oracle[0].table_name,
                                                        .push_ordinal = oracle[0].output_ordinal}});
  REQUIRE(physical_joins[0]->dynamic_filter_plan().enabled());
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - an unfiltered build wires nothing on either side",
                 "[dynamic_filter][parity][isolated_context]")
{
  dynamic_filter_on_guard filter_on(*con);
  auto c = plan_parity_case(*con, "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  REQUIRE_FALSE(sirius::planner::build_subtree_is_filtering(*joins[0]->children[1]));
  REQUIRE_FALSE(duckdb::JoinFilterPushdownOptimizer::IsFiltering(joins[0]->children[1]));

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  REQUIRE_FALSE(physical_joins[0]->dynamic_filter_plan().enabled());
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - filter evidence via bare FILTER and via TOP_N",
                 "[dynamic_filter][parity][isolated_context]")
{
  dynamic_filter_on_guard filter_on(*con);

  SECTION("a residual build-side FILTER arms the gate on both sides")
  {
    // `other % 2 = 0` is not pushable into table_filters, so it survives as a LOGICAL_FILTER.
    auto c     = plan_parity_case(*con,
                              "SELECT * FROM big_left l JOIN "
                                  "(SELECT * FROM small_right WHERE other % 2 = 0) r "
                                  "ON l.id = r.rid");
    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    REQUIRE(sirius::planner::build_subtree_is_filtering(*joins[0]->children[1]));
    REQUIRE(duckdb::JoinFilterPushdownOptimizer::IsFiltering(joins[0]->children[1]));

    auto const oracle = oracle_targets_for_join(*joins[0]);
    REQUIRE(oracle.size() == 1);

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    auto const bindings = scan_bindings_of(*physical_joins[0], c.physical.get());
    REQUIRE(bindings.size() == 1);
    REQUIRE(bindings[0].table_name == oracle[0].table_name);
  }

  SECTION("a build-side TOP_N arms the gate on both sides")
  {
    auto c     = plan_parity_case(*con,
                              "SELECT * FROM big_left l JOIN "
                                  "(SELECT * FROM small_right ORDER BY other LIMIT 1) r "
                                  "ON l.id = r.rid");
    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    REQUIRE(sirius::planner::build_subtree_is_filtering(*joins[0]->children[1]));
    REQUIRE(duckdb::JoinFilterPushdownOptimizer::IsFiltering(joins[0]->children[1]));

    auto const oracle = oracle_targets_for_join(*joins[0]);
    REQUIRE(oracle.size() == 1);

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    auto const bindings = scan_bindings_of(*physical_joins[0], c.physical.get());
    REQUIRE(bindings.size() == 1);
    REQUIRE(bindings[0].table_name == oracle[0].table_name);
  }
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - LIMIT on the probe spine: DuckDB binds, Sirius refuses",
                 "[dynamic_filter][parity][isolated_context]")
{
  // Filtering below a position selection can change which rows LIMIT selects, so Sirius stops
  // where DuckDB continues.
  dynamic_filter_on_guard filter_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT * FROM big_left LIMIT 15) l "
                            "JOIN small_right r ON l.id = r.rid WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get()).empty());
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - TOP_N on the probe spine: DuckDB binds, Sirius refuses",
                 "[dynamic_filter][parity][isolated_context]")
{
  dynamic_filter_on_guard filter_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT * FROM big_left ORDER BY val LIMIT 15) l "
                            "JOIN small_right r ON l.id = r.rid WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get()).empty());
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - an intervening RIGHT join: both bind through it",
                 "[dynamic_filter][parity][isolated_context]")
{
  // Removing a nested match can synthesize a NULL probe key, but the producing equality rejects it.
  dynamic_filter_on_guard filter_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT l.id AS id FROM big_left l "
                            "RIGHT JOIN small_c c ON l.val = c.ckey) x "
                            "JOIN small_right r ON x.id = r.rid WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 2);
  auto* producing   = joins[0];
  auto* intervening = joins[1];
  REQUIRE(comparison_joins_of(*producing->children[0]) ==
          std::vector<duckdb::LogicalComparisonJoin*>{intervening});
  REQUIRE(intervening->join_type == duckdb::JoinType::RIGHT);

  auto const oracle = oracle_targets_for_join(*producing);
  REQUIRE(oracle.size() == 1);
  REQUIRE(oracle[0].condition_index == 0);
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 2);
  REQUIRE(hash_joins_of(physical_joins[0]->children[0].get()) ==
          std::vector<sirius::op::sirius_physical_hash_join*>{physical_joins[1]});
  REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get()) ==
          std::vector<sirius_scan_binding>{{.condition_index = 0,
                                            .table_name      = oracle[0].table_name,
                                            .push_ordinal    = oracle[0].output_ordinal}});
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - intervening OUTER and ANTI joins preserve probe descent",
                 "[dynamic_filter][parity][isolated_context]")
{
  struct case_spec {
    duckdb::JoinType join_type;
    std::string query;
  };
  std::vector<case_spec> const cases{{duckdb::JoinType::OUTER,
                                      "SELECT * FROM (SELECT l.id AS id FROM big_left l "
                                      "FULL OUTER JOIN small_c c ON l.val = c.ckey) x "
                                      "JOIN small_right r ON x.id = r.rid WHERE r.other > 0"},
                                     {duckdb::JoinType::ANTI,
                                      "SELECT * FROM (SELECT l.id AS id FROM big_left l "
                                      "ANTI JOIN small_c c ON l.val = c.ckey) x "
                                      "JOIN small_right r ON x.id = r.rid WHERE r.other > 0"}};

  dynamic_filter_on_guard filter_on(*con);
  for (auto const& spec : cases) {
    CAPTURE(spec.join_type);
    auto c     = plan_parity_case(*con, spec.query);
    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 2);
    auto* producing   = joins[0];
    auto* intervening = joins[1];
    REQUIRE(comparison_joins_of(*producing->children[0]) ==
            std::vector<duckdb::LogicalComparisonJoin*>{intervening});
    REQUIRE(intervening->join_type == spec.join_type);

    auto const oracle = oracle_targets_for_join(*producing);
    REQUIRE(oracle.size() == 1);
    REQUIRE(oracle[0].condition_index == 0);
    REQUIRE(oracle[0].table_name == "big_left");

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 2);
    REQUIRE(hash_joins_of(physical_joins[0]->children[0].get()) ==
            std::vector<sirius::op::sirius_physical_hash_join*>{physical_joins[1]});
    REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get()) ==
            std::vector<sirius_scan_binding>{{.condition_index = 0,
                                              .table_name      = oracle[0].table_name,
                                              .push_ordinal    = oracle[0].output_ordinal}});
  }
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - a cast projection: DuckDB crosses it, Sirius refuses",
                 "[dynamic_filter][parity][isolated_context]")
{
  // DuckDB follows supported integral casts; Sirius accepts only plain projection references.
  dynamic_filter_on_guard filter_on(*con);
  auto c =
    plan_parity_case(*con,
                     "SELECT * FROM (SELECT CAST(id AS BIGINT) AS bid, val FROM big_left) l "
                     "JOIN small_right r ON l.bid = CAST(r.rid AS BIGINT) WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get()).empty());
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - a computed projection refuses on both sides",
                 "[dynamic_filter][parity][isolated_context]")
{
  dynamic_filter_on_guard filter_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT id + 1 AS cid, val FROM big_left) l "
                            "JOIN small_right r ON l.cid = r.rid WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  REQUIRE(oracle_targets_for_join(*joins[0]).empty());

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get()).empty());
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - aggregate grouping keys pass, aggregate results refuse",
                 "[dynamic_filter][parity][isolated_context]")
{
  dynamic_filter_on_guard filter_on(*con);

  SECTION("a grouping key binds on both sides")
  {
    auto c     = plan_parity_case(*con,
                              "SELECT * FROM (SELECT rid FROM small_right GROUP BY rid) g "
                                  "JOIN small_c c ON g.rid = c.ckey WHERE c.cother > 0");
    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    auto const oracle = oracle_targets_for_join(*joins[0]);
    REQUIRE(oracle.size() == 1);
    REQUIRE(oracle[0].table_name == "small_right");

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    auto const bindings = scan_bindings_of(*physical_joins[0], c.physical.get());
    REQUIRE(bindings ==
            std::vector<sirius_scan_binding>{{.condition_index = 0,
                                              .table_name      = oracle[0].table_name,
                                              .push_ordinal    = oracle[0].output_ordinal}});
  }

  SECTION("an aggregate result refuses on both sides")
  {
    auto c     = plan_parity_case(*con,
                              "SELECT * FROM "
                                  "(SELECT rid, min(other) AS m FROM small_right GROUP BY rid) g "
                                  "JOIN small_c c ON g.m = c.ckey WHERE c.cother > 0");
    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    REQUIRE(oracle_targets_for_join(*joins[0]).empty());  // the binding is aggregate_index

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get()).empty());
  }
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - a residual FILTER on the probe spine binds on both sides",
                 "[dynamic_filter][parity][isolated_context]")
{
  // Q19-class shape: a predicate table filters cannot express keeps a FILTER between the join and
  // the probe scan.
  dynamic_filter_on_guard filter_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid "
                            "WHERE l.val % 2 = 0 AND r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  auto const bindings = scan_bindings_of(*physical_joins[0], c.physical.get());
  REQUIRE(bindings == std::vector<sirius_scan_binding>{{.condition_index = 0,
                                                        .table_name      = oracle[0].table_name,
                                                        .push_ordinal = oracle[0].output_ordinal}});
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - per-key discovery binds the clean key DuckDB's joint walk "
                 "abandons",
                 "[dynamic_filter][parity][isolated_context]")
{
  // DuckDB walks all hinted columns jointly and abandons the whole branch when any column fails;
  // Sirius walks per key. The extra binding is still one DuckDB itself makes for that key in
  // isolation.
  dynamic_filter_on_guard filter_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT id, val + 1 AS w FROM big_left) l "
                            "JOIN small_right r ON l.id = r.rid AND l.w = r.other "
                            "WHERE r.other > 0");

  // DuckDB's own joint walk found nothing: no filter_pushdown was attached to the original join.
  REQUIRE(c.original_join_had_pushdown == std::vector<bool>{false});

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);
  REQUIRE(oracle[0].condition_index == 0);
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  auto const bindings = scan_bindings_of(*physical_joins[0], c.physical.get());
  REQUIRE(bindings == std::vector<sirius_scan_binding>{{.condition_index = 0,
                                                        .table_name      = oracle[0].table_name,
                                                        .push_ordinal = oracle[0].output_ordinal}});
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - producer join-type gate",
                 "[dynamic_filter][parity][isolated_context]")
{
  dynamic_filter_on_guard filter_on(*con);

  SECTION("a RIGHT producer binds on both sides")
  {
    // RIGHT is in DuckDB's admitted set: a probe-side filter removes only probe rows the join
    // drops; the NULL-padded rows a RIGHT join fabricates come from unmatched BUILD rows.
    auto c     = plan_parity_case(*con,
                              "SELECT * FROM big_left l RIGHT JOIN "
                                  "(SELECT * FROM small_right WHERE other > 0) r ON l.id = r.rid");
    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    REQUIRE(joins[0]->join_type == duckdb::JoinType::RIGHT);
    auto const oracle = oracle_targets_for_join(*joins[0]);
    REQUIRE(oracle.size() == 1);
    REQUIRE(oracle[0].table_name == "big_left");

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    auto const bindings = scan_bindings_of(*physical_joins[0], c.physical.get());
    REQUIRE(bindings ==
            std::vector<sirius_scan_binding>{{.condition_index = 0,
                                              .table_name      = oracle[0].table_name,
                                              .push_ordinal    = oracle[0].output_ordinal}});
  }

  SECTION("a LEFT producer refuses on both sides")
  {
    // LEFT keeps every probe row, so a probe-side membership filter would change results.
    auto c = plan_parity_case(*con,
                              "SELECT * FROM big_left l LEFT JOIN "
                              "(SELECT * FROM small_right WHERE other > 0) r ON l.id = r.rid");
    REQUIRE(c.original_join_had_pushdown == std::vector<bool>{false});

    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    REQUIRE(joins[0]->join_type == duckdb::JoinType::LEFT);

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    REQUIRE_FALSE(physical_joins[0]->dynamic_filter_plan().enabled());
  }

  SECTION("a SEMI producer binds on both sides")
  {
    // Uncorrelated IN produces the bare SEMI comparison join needed by this case.
    auto c     = plan_parity_case(*con,
                              "SELECT * FROM big_left l "
                                  "WHERE id IN (SELECT rid FROM small_right WHERE other > 0)");
    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    REQUIRE(joins[0]->join_type == duckdb::JoinType::SEMI);
    auto const oracle = oracle_targets_for_join(*joins[0]);
    REQUIRE(oracle.size() == 1);
    REQUIRE(oracle[0].table_name == "big_left");

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    auto const bindings = scan_bindings_of(*physical_joins[0], c.physical.get());
    REQUIRE(bindings ==
            std::vector<sirius_scan_binding>{{.condition_index = 0,
                                              .table_name      = oracle[0].table_name,
                                              .push_ordinal    = oracle[0].output_ordinal}});
  }

  SECTION("a MARK producer refuses on both sides")
  {
    // A projected IN keeps the membership result as a column, which plans as a MARK join.
    auto c = plan_parity_case(
      *con, "SELECT id, id IN (SELECT rid FROM small_right WHERE other > 0) AS flag FROM big_left");
    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    REQUIRE(joins[0]->join_type == duckdb::JoinType::MARK);
    REQUIRE(c.original_join_had_pushdown == std::vector<bool>{false});

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    REQUIRE_FALSE(physical_joins[0]->dynamic_filter_plan().enabled());
  }

  SECTION("an ANTI producer refuses on both sides")
  {
    // ANTI keeps exactly the probe rows without a build match -- the rows a probe-side membership
    // filter would remove. Explicit ANTI JOIN produces the bare comparison join needed by this
    // case.
    auto c = plan_parity_case(*con,
                              "SELECT * FROM big_left l ANTI JOIN "
                              "(SELECT * FROM small_right WHERE other > 0) r ON l.id = r.rid");
    REQUIRE(c.original_join_had_pushdown == std::vector<bool>{false});

    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    REQUIRE(joins[0]->join_type == duckdb::JoinType::ANTI);

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    REQUIRE_FALSE(physical_joins[0]->dynamic_filter_plan().enabled());
  }

  SECTION("an ANTI producer with an opaque build still refuses on both sides")
  {
    // A materialized CTE build arms discovery with opaque evidence, so this pins the refusal to
    // the join type rather than to missing evidence.
    auto c = plan_parity_case(*con,
                              "WITH c AS MATERIALIZED "
                              "(SELECT rid FROM small_right WHERE other > 0) "
                              "SELECT * FROM big_left l ANTI JOIN c ON l.id = c.rid");
    REQUIRE(c.original_join_had_pushdown == std::vector<bool>{false});

    auto joins = comparison_joins_of(*c.oracle_plan);
    REQUIRE(joins.size() == 1);
    REQUIRE(joins[0]->join_type == duckdb::JoinType::ANTI);
    REQUIRE(sirius::planner::build_relation_is_opaque(*joins[0]->children[1]));
    REQUIRE_FALSE(sirius::planner::build_subtree_is_filtering(*joins[0]->children[1]));

    auto physical_joins = hash_joins_of(c.physical.get());
    REQUIRE(physical_joins.size() == 1);
    REQUIRE_FALSE(physical_joins[0]->dynamic_filter_plan().enabled());
  }
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - wiring is independent of DuckDB pushdown metadata",
                 "[dynamic_filter][parity][isolated_context]")
{
  dynamic_filter_on_guard filter_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid "
                            "WHERE r.other > 0");

  // Premise: the original plan carried DuckDB's pushdown metadata and the serialization
  // round-trip stripped it from the copy.
  REQUIRE(c.original_join_had_pushdown == std::vector<bool>{true});
  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  REQUIRE(joins[0]->filter_pushdown == nullptr);

  // Build a second Sirius physical plan from the stripped copy, mirroring plan_parity_case's
  // transaction discipline.
  duckdb::unique_ptr<sirius_physical_operator> from_stripped;
  con->Query("BEGIN TRANSACTION");
  try {
    c.oracle_plan->ResolveOperatorTypes();
    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(*c.oracle_plan);
    resolver.VisitOperator(*c.oracle_plan);

    sirius::planner::sirius_physical_plan_generator gen(*con->context);
    from_stripped = gen.create_plan(std::move(c.oracle_plan));
  } catch (...) {
    con->Query("ROLLBACK");
    throw;
  }
  con->Query("COMMIT");

  // The stripped plan wires the same scan route as the original.
  auto original_joins = hash_joins_of(c.physical.get());
  auto stripped_joins = hash_joins_of(from_stripped.get());
  REQUIRE(original_joins.size() == 1);
  REQUIRE(stripped_joins.size() == 1);
  REQUIRE(stripped_joins[0]->dynamic_filter_plan().enabled());
  auto const stripped_bindings = scan_bindings_of(*stripped_joins[0], from_stripped.get());
  REQUIRE_FALSE(stripped_bindings.empty());
  REQUIRE(stripped_bindings == scan_bindings_of(*original_joins[0], c.physical.get()));
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - the adapter oracle agrees with the per-key walk",
                 "[dynamic_filter][parity][isolated_context]")
{
  dynamic_filter_on_guard filter_on(*con);
  auto original = optimize_query(*con,
                                 "SELECT * FROM big_left l "
                                 "JOIN (SELECT * FROM small_right WHERE other > 0) r "
                                 "ON l.id = r.rid "
                                 "JOIN (SELECT * FROM small_c WHERE cother >= 0) c "
                                 "ON l.val = c.ckey");

  auto joins = comparison_joins_of(*original);
  REQUIRE(joins.size() == 2);
  for (auto* join : joins) {
    auto const candidate = sirius::planner::duckdb_join_filter_candidate_adapter::extract(*join);
    REQUIRE(candidate.kind() == sirius::planner::duckdb_candidate_kind::admitted);
    REQUIRE(candidate.condition_indexes() == std::vector<std::size_t>{0});
    REQUIRE(candidate.targets().size() == 1);

    auto const oracle = oracle_targets_for_join(*join);
    REQUIRE(oracle.size() == 1);
    REQUIRE(oracle[0].condition_index == 0);
    REQUIRE(oracle[0].table_name == "big_left");
  }
}
