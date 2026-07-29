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
 * @brief The parity ORACLE: Sirius-owned scan-target discovery against DuckDB's own pushdown walk
 *
 * Sirius discovers dynamic-filter scan targets by walking its built physical probe subtree
 * (`dynamic_filter_target_discovery.hpp`); DuckDB's `JoinFilterPushdownOptimizer` walks the
 * logical probe subtree. This suite pins the per-key parity contract between the two: for each
 * Sirius-admissible equality key of a producing join, Sirius binds a scan target exactly when
 * DuckDB's public `GetPushdownFilterTargets`, given that key alone, reaches the same base scan --
 * modulo the documented conservative divergences, each of which has a section here asserting BOTH
 * sides so the divergence is pinned rather than hidden.
 *
 * Method per case: plan and optimize the query; deep-copy the optimized logical plan BEFORE the
 * ColumnBindingResolver runs (the copy keeps binding-space expressions the DuckDB walk consumes,
 * and Copy's serialize round-trip strips DuckDB's own pushdown metadata, so the oracle re-derives
 * targets from structure alone); convert the original through `create_plan`; compare each hash
 * join's `dynamic_filter_plan()` bindings against per-key oracle expectations whose ordinals are
 * DuckDB-computed output positions -- never hardcoded.
 *
 * Divergences without an oracle section: a bare ORDER BY between a join and a scan cannot be
 * planned from SQL (DuckDB elides subquery ORDER BY without LIMIT/OFFSET), so that refusal row is
 * pinned by the shared refusal list in the discovery unit tests and by the LIMIT/TOP_N sections
 * here, which exercise the same walk path. Join-type syntax note: the ANTI producer section uses
 * the explicit `ANTI JOIN` clause, which binds a bare ANTI comparison join directly at bind time;
 * the SQL derivations do not produce one under the pinned optimizer set -- a correlated NOT EXISTS
 * plans through a DELIM_JOIN wrapper, and NOT IN plans as a MARK join (the MARK-to-ANTI conversion
 * lives in the held-back STATISTICS_PROPAGATION pass).
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

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so these tests need an on-disk database rather than :memory:.
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

/// RAII flip of the dynamic-filter switches on the connection's registered SiriusContext: master
/// on, SIP off, so discovery runs and every wired target is scan class.
class master_only_switch_guard {
 public:
  explicit master_only_switch_guard(Connection& con)
    : _state(con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state"))
  {
    REQUIRE(_state != nullptr);
    auto& params                          = _state->get_config().get_operator_params();
    _original_pushdown                    = params.enable_dynamic_filter_pushdown;
    _original_sip                         = params.enable_dynamic_filter_sip;
    params.enable_dynamic_filter_pushdown = true;
    params.enable_dynamic_filter_sip      = false;
  }

  ~master_only_switch_guard()
  {
    auto& params                          = _state->get_config().get_operator_params();
    params.enable_dynamic_filter_pushdown = _original_pushdown;
    params.enable_dynamic_filter_sip      = _original_sip;
  }

  master_only_switch_guard(const master_only_switch_guard&)            = delete;
  master_only_switch_guard& operator=(const master_only_switch_guard&) = delete;

 private:
  duckdb::shared_ptr<duckdb::SiriusContext> _state;
  bool _original_pushdown = true;
  bool _original_sip      = false;
};

/// One planned query, carrying both comparison sides. `oracle_plan` is the pre-resolver deep copy
/// the DuckDB walk consumes; `original_join_had_pushdown[i]` records, per pre-order comparison
/// join of the ORIGINAL, whether DuckDB's own optimizer attached `filter_pushdown` (read before
/// `create_plan` consumes the original); `physical` is the Sirius plan built from the original.
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

/// Plan, optimize, and (optionally) convert one query. JOIN_ORDER and BUILD_SIDE_PROBE_SIDE are
/// always held back so the physical tree keeps the syntactic join order and syntactic build
/// sides; STATISTICS_PROPAGATION so constant folding does not erase the shapes under test.
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
    // The oracle copy is taken BEFORE the resolver so it keeps binding-space expressions; the
    // serialize round-trip strips filter_pushdown / dynamic_filters, which the oracle re-derives.
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

/// Mirror of DuckDB's private `PushdownJoinFilterExpression` entry shapes
/// (join_filter_pushdown_optimizer.cpp:23-55): nested and INTERVAL keys are refused; otherwise a
/// plain column reference, or an integral cast chain no wider than 64 bits over one. Returns the
/// binding the DuckDB walk would track.
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

/// One per-key oracle finding: DuckDB's walk, given only this condition's probe column, reached
/// base table `table_name` and expects the key at OUTPUT position `output_ordinal` of that scan.
struct oracle_key_target {
  std::size_t condition_index = 0;
  std::string table_name;
  std::size_t output_ordinal = 0;
};

/// Run DuckDB's `GetPushdownFilterTargets` once per condition of @p join_copy (per key -- the
/// public walk is static, so a single-column vector isolates each key from its siblings) and
/// convert each found target's binding to the target scan's OUTPUT position.
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

/// One Sirius scan binding, read back through the publish plan: which planner condition bound, into
/// which base table's scan, at which channel push ordinal (== the bound scan's output ordinal).
/// Field order mirrors `oracle_key_target` so the two sides compare member for member.
struct sirius_scan_binding {
  std::size_t condition_index = 0;
  std::string table_name;
  std::size_t push_ordinal = 0;

  [[nodiscard]] bool operator==(sirius_scan_binding const&) const = default;
};

/// Resolve the base table read by the one GPU scan leaf whose dynamic-filter channel is
/// pointer-identical to @p filter_set. The producing join attaches the channel to the probe scan it
/// reaches, and plan construction hands that same shared channel to the replacing
/// `sirius_gpu_scan_operator` through its ingestible table info -- the finished tree carries no
/// TABLE_SCAN nodes. Channel identity IS the production producer/consumer pairing, so a scan-class
/// target must resolve to exactly one leaf. The fixture's tables are duckdb-native, so the info is
/// `duckdb_native_ingestible_table_info`, which carries the channel and the resolved base-table
/// name together.
std::string owning_scan_table_of(
  sirius_physical_operator* root,
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> const& filter_set)
{
  std::vector<sirius::op::scan::duckdb_native_ingestible_table_info const*> owners;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op->type != SiriusPhysicalOperatorType::GPU_SCAN) { return; }
    auto const& scan = op->Cast<sirius::op::scan::sirius_gpu_scan_operator>();
    auto const* info = dynamic_cast<sirius::op::scan::duckdb_native_ingestible_table_info const*>(
      &scan.peek_table_info());
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
  master_only_switch_guard master_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid "
                            "WHERE r.other > 0");

  // Oracle side: the single key reaches big_left; the expected push ordinal is the DuckDB-computed
  // output position of l.id in that scan.
  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);
  REQUIRE(oracle[0].condition_index == 0);
  REQUIRE(oracle[0].table_name == "big_left");

  // Sirius side: exactly that binding, at exactly that ordinal.
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
  master_only_switch_guard master_on(*con);
  auto c = plan_parity_case(*con, "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid");

  // Evidence parity: the Sirius mirror and DuckDB's IsFiltering agree the build child is
  // unfiltered.
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
  master_only_switch_guard master_on(*con);

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
  // Filtering below a position selection changes which rows are selected, so the Sirius walk
  // refuses LIMIT as a correctness rule where DuckDB pushes through it. Costs speed, never
  // correctness -- and this section pins BOTH sides of that divergence.
  master_only_switch_guard master_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT * FROM big_left LIMIT 15) l "
                            "JOIN small_right r ON l.id = r.rid WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);  // DuckDB descends through LOGICAL_LIMIT
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  REQUIRE(
    scan_bindings_of(*physical_joins[0], c.physical.get()).empty());  // Sirius refuses at the LIMIT
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - TOP_N on the probe spine: DuckDB binds, Sirius refuses",
                 "[dynamic_filter][parity][isolated_context]")
{
  master_only_switch_guard master_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT * FROM big_left ORDER BY val LIMIT 15) l "
                            "JOIN small_right r ON l.id = r.rid WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);  // DuckDB descends through LOGICAL_TOP_N
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  REQUIRE(
    scan_bindings_of(*physical_joins[0], c.physical.get()).empty());  // Sirius refuses at the TOP_N
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - an intervening RIGHT join: DuckDB binds through it, Sirius "
                 "refuses",
                 "[dynamic_filter][parity][isolated_context]")
{
  // DuckDB's walk recurses children[0] of ANY comparison join, RIGHT included, so it binds l.id
  // through the intervening join; the Sirius walk refuses every probe block that NULL-pads its
  // traced column (probe_block_is_value_preserving). Conservative: costs speed, never correctness
  // -- and this section pins BOTH sides of the divergence.
  master_only_switch_guard master_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT l.id AS id FROM big_left l "
                            "RIGHT JOIN small_c c ON l.val = c.ckey) x "
                            "JOIN small_right r ON x.id = r.rid WHERE r.other > 0");

  // Containment pins which join is which, and the join-type guard makes an optimizer flip of the
  // intervening RIGHT join fail loudly instead of degrading this section into testing nothing.
  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 2);
  auto* producing   = joins[0];
  auto* intervening = joins[1];
  REQUIRE(comparison_joins_of(*producing->children[0]) ==
          std::vector<duckdb::LogicalComparisonJoin*>{intervening});
  REQUIRE(intervening->join_type == duckdb::JoinType::RIGHT);

  auto const oracle = oracle_targets_for_join(*producing);
  REQUIRE(oracle.size() == 1);  // DuckDB descends the RIGHT join's children[0]
  REQUIRE(oracle[0].condition_index == 0);
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 2);
  REQUIRE(hash_joins_of(physical_joins[0]->children[0].get()) ==
          std::vector<sirius::op::sirius_physical_hash_join*>{physical_joins[1]});
  REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get())
            .empty());  // Sirius refuses at the RIGHT join
  REQUIRE_FALSE(physical_joins[0]->dynamic_filter_plan().enabled());
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - a cast projection: DuckDB crosses it, Sirius refuses",
                 "[dynamic_filter][parity][isolated_context]")
{
  // DuckDB pushes through integral casts no wider than 64 bits; crossing a cast is a deferred
  // follow-up for Sirius, whose projection rule accepts plain references only. Conservative.
  master_only_switch_guard master_on(*con);
  auto c =
    plan_parity_case(*con,
                     "SELECT * FROM (SELECT CAST(id AS BIGINT) AS bid, val FROM big_left) l "
                     "JOIN small_right r ON l.bid = CAST(r.rid AS BIGINT) WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);  // DuckDB reaches big_left through the cast
  REQUIRE(oracle[0].table_name == "big_left");

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get())
            .empty());  // Sirius refuses the cast crossing
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - a computed projection refuses on both sides",
                 "[dynamic_filter][parity][isolated_context]")
{
  master_only_switch_guard master_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT id + 1 AS cid, val FROM big_left) l "
                            "JOIN small_right r ON l.cid = r.rid WHERE r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  REQUIRE(oracle_targets_for_join(*joins[0]).empty());  // DuckDB bails at the computed expression

  auto physical_joins = hash_joins_of(c.physical.get());
  REQUIRE(physical_joins.size() == 1);
  REQUIRE(scan_bindings_of(*physical_joins[0], c.physical.get()).empty());
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - aggregate grouping keys pass, aggregate results refuse",
                 "[dynamic_filter][parity][isolated_context]")
{
  master_only_switch_guard master_on(*con);

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
  // the probe scan. The FILTER descent row exists precisely so this shape keeps its filter.
  master_only_switch_guard master_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid "
                            "WHERE l.val % 2 = 0 AND r.other > 0");

  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);  // DuckDB descends LOGICAL_FILTER
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
  // DuckDB walks all hinted columns jointly and abandons the whole branch when ANY column fails;
  // Sirius walks per key. The per-key never-more contract still holds: the extra binding is one
  // DuckDB itself makes for that key in isolation, which the per-key oracle verifies.
  master_only_switch_guard master_on(*con);
  auto c = plan_parity_case(*con,
                            "SELECT * FROM (SELECT id, val + 1 AS w FROM big_left) l "
                            "JOIN small_right r ON l.id = r.rid AND l.w = r.other "
                            "WHERE r.other > 0");

  // DuckDB's own joint walk found nothing: no filter_pushdown was attached to the original join.
  REQUIRE(c.original_join_had_pushdown == std::vector<bool>{false});

  // The per-key oracle binds the clean key (condition 0, id) and refuses the computed one.
  auto joins = comparison_joins_of(*c.oracle_plan);
  REQUIRE(joins.size() == 1);
  auto const oracle = oracle_targets_for_join(*joins[0]);
  REQUIRE(oracle.size() == 1);
  REQUIRE(oracle[0].condition_index == 0);
  REQUIRE(oracle[0].table_name == "big_left");

  // Sirius binds exactly the clean key at the oracle's ordinal.
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
  master_only_switch_guard master_on(*con);

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
    // LEFT keeps every probe row, so a probe-side membership filter would change results. DuckDB
    // returns early from GenerateJoinFilters; Sirius's join-type gate refuses the bind.
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
    // Uncorrelated IN: a correlated EXISTS plans through a DELIM_JOIN wrapper rather than a bare
    // SEMI comparison join (see the join-type syntax note in the file docblock).
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
    // filter would remove. DuckDB returns early from GenerateJoinFilters; Sirius's join-type gate
    // refuses the bind. The explicit ANTI JOIN clause binds this bare shape directly (see the file
    // docblock's syntax note).
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
}

TEST_CASE_METHOD(discovery_parity_fixture,
                 "discovery parity - the adapter oracle agrees with the per-key walk",
                 "[dynamic_filter][parity][isolated_context]")
{
  // The adapter (test-only since unified discovery) extracts what DuckDB's optimizer itself
  // recorded on the ORIGINAL, before Copy strips it. Cross-checking extract() against the per-key
  // walk on a plain two-join shape proves the oracle's two data paths agree with each other, so a
  // future DuckDB bump that breaks one is caught by the other.
  master_only_switch_guard master_on(*con);
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
