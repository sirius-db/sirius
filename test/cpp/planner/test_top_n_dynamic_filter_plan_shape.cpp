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
 * @file test_top_n_dynamic_filter_plan_shape.cpp
 * @brief Where the Stage-4 Top-N producer binds its targets, asserted on the built physical plan.
 *
 * Siting is otherwise observable only through counters, which say a target was bound but not
 * where. Each case here asserts the R1 triple: the node's position in the tree, its
 * `provenance()`, and the ordinals the producer registered on that node's channel. Ordinals are
 * asserted by arity and distinctness rather than by hard-coded index, so a projection layout
 * change cannot silently turn a real assertion into a vacuous one.
 *
 * These also exercise `trace_top_n_all_keys` on real plans: the all-ordinals-survive rule shows
 * up as a two-ordinal slot (scenario 9), UNION fan-out as one slot per branch (scenario 6), and
 * hop refusal as no slot at all (scenario 5).
 *
 * **Why the scan-binding cases read Parquet.** The siting rule sites a target only where it saves
 * work nothing else saves: the consumer skips reads, or material work lies between the site and
 * the sink. A DuckDB-native scan reached over pass-through hops alone satisfies neither, so the
 * planner correctly binds nothing there and a native table cannot pin *where* a trace bottoms out.
 * Every case whose subject is trace mechanics therefore scans Parquet, whose reader prunes row
 * groups by statistics. Rewriting one of them back to a native table would leave it asserting the
 * siting rule instead, and the trace it was written to pin would go untested. The native shapes
 * are covered deliberately and separately, by the refusal cases and by the material-work cases
 * that must still bind.
 */

#include "expression/ast/node.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/dynamic_filter/top_n_group_key_producer.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "planner/dynamic_filter/dynamic_filter_target_discovery.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::op::scan::dynamic_filter_endpoint_provenance;
using sirius::op::scan::sirius_physical_dynamic_filter;

namespace {

class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_top_n_plan_shape_XXXXXX";
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
  scoped_temp_db_path(scoped_temp_db_path const&)            = delete;
  scoped_temp_db_path& operator=(scoped_temp_db_path const&) = delete;

  [[nodiscard]] std::string const& path() const { return _path; }

 private:
  std::string _path;
};

duckdb::unique_ptr<sirius_physical_operator> generate_sirius_plan(Connection& con,
                                                                  std::string const& query)
{
  auto& context          = *con.context;
  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  disabled.insert(OptimizerType::STATISTICS_PROPAGATION);

  con.Query("BEGIN TRANSACTION");
  duckdb::unique_ptr<sirius_physical_operator> result;
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

template <typename Fn>
void for_each_operator(sirius_physical_operator* root, Fn const& fn)
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
  std::vector<sirius_physical_operator*> out;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op->type == type) { out.push_back(op); }
  });
  return out;
}

sirius_physical_operator* find_first(sirius_physical_operator* root,
                                     SiriusPhysicalOperatorType type)
{
  auto all = collect(root, type);
  return all.empty() ? nullptr : all.front();
}

/// Whether @p needle is @p root or one of its descendants.
bool contains_operator(sirius_physical_operator* root, sirius_physical_operator const* needle)
{
  bool found = false;
  for_each_operator(root, [&](sirius_physical_operator* node) { found = found || node == needle; });
  return found;
}

/// Every DYNAMIC_FILTER operator that sits directly above a GPU_SCAN -- the scan-route wrapper
/// the planner installs when a scan's channel has a registered producer.
std::vector<sirius_physical_dynamic_filter*> scan_route_filters(sirius_physical_operator* root)
{
  std::vector<sirius_physical_dynamic_filter*> out;
  for (auto* node : collect(root, SiriusPhysicalOperatorType::DYNAMIC_FILTER)) {
    auto* filter = static_cast<sirius_physical_dynamic_filter*>(node);
    if (node->children.size() == 1 &&
        node->children[0]->type == SiriusPhysicalOperatorType::GPU_SCAN) {
      out.push_back(filter);
    }
  }
  return out;
}

/// Refinement slots registered on every scan-route channel in the plan.
std::vector<sirius::op::sirius_dynamic_filter_set::refinement_slot_view> registered_slots(
  sirius_physical_operator* root)
{
  std::vector<sirius::op::sirius_dynamic_filter_set::refinement_slot_view> slots;
  for (auto* filter : scan_route_filters(root)) {
    // Position + provenance: this is the scan-route wrapper, not a sited Top-N endpoint.
    CHECK(filter->provenance() == dynamic_filter_endpoint_provenance::scan_route);
    auto const channel = filter->filters();
    if (!channel) { continue; }
    for (auto& slot : channel->refinement_slots()) {
      slots.push_back(std::move(slot));
    }
  }
  return slots;
}

/// Slots registered on the one scan-route channel a single-table plan has.
std::vector<sirius::op::sirius_dynamic_filter_set::refinement_slot_view> single_scan_slots(
  sirius_physical_operator* root)
{
  auto const filters = scan_route_filters(root);
  REQUIRE(filters.size() == 1);
  auto const channel = filters.front()->filters();
  REQUIRE(channel != nullptr);
  return channel->refinement_slots();
}

/// Set `enable_top_n_dynamic_filter` for one scope and restore it.
class top_n_filter_switch_guard {
 public:
  top_n_filter_switch_guard(Connection& con, bool enabled)
    : _state(con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state"))
  {
    REQUIRE(_state != nullptr);
    auto& params                       = _state->get_config().get_operator_params();
    _original                          = params.enable_top_n_dynamic_filter;
    params.enable_top_n_dynamic_filter = enabled;
  }
  ~top_n_filter_switch_guard()
  {
    _state->get_config().get_operator_params().enable_top_n_dynamic_filter = _original;
  }
  top_n_filter_switch_guard(top_n_filter_switch_guard const&)            = delete;
  top_n_filter_switch_guard& operator=(top_n_filter_switch_guard const&) = delete;

 private:
  duckdb::shared_ptr<duckdb::SiriusContext> _state;
  bool _original = false;
};

struct top_n_plan_shape_fixture {
  top_n_plan_shape_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(_db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    con->Query("CREATE TABLE facts (id INTEGER, v INTEGER, w INTEGER, pay VARCHAR)");
    con->Query(
      "INSERT INTO facts SELECT range, (range * 7) % 101, range % 13, concat('p', range % 5) "
      "FROM range(200)");
    con->Query("CREATE TABLE other (oid INTEGER, ov INTEGER)");
    con->Query("INSERT INTO other SELECT range, range % 17 FROM range(50)");
    con->Query("CREATE TABLE dim (dk INTEGER, dv INTEGER)");
    con->Query("INSERT INTO dim SELECT range, range FROM range(200)");
    facts_parquet = _db_path.path() + ".facts.parquet";
    con->Query("COPY facts TO '" + facts_parquet + "' (FORMAT PARQUET)");
    // One column per decimal admission band: DECIMAL32 (p<=9), DECIMAL64 (p<=18), and DECIMAL128
    // (p>=19), which the boundary components cannot hold. The decimal cases scan the parquet
    // copy (see the file comment). For `huge_dec` parquet is also the only reachable path: the
    // duckdb-native scan refuses a DECIMAL128 column outright, while parquet has no precision
    // gate and is where the p>=19 refusal is actually live.
    con->Query(
      "CREATE TABLE money (id INTEGER, small_dec DECIMAL(9,2), big_dec DECIMAL(18,4), "
      "huge_dec DECIMAL(38,4))");
    con->Query(
      "INSERT INTO money SELECT range, range * 1.5, range * 2.25, range * 3.125 FROM range(200)");
    money_parquet = _db_path.path() + ".money.parquet";
    con->Query("COPY money TO '" + money_parquet + "' (FORMAT PARQUET)");
  }
  ~top_n_plan_shape_fixture()
  {
    unsetenv("SIRIUS_CONFIG_FILE");
    if (!money_parquet.empty()) { std::remove(money_parquet.c_str()); }
    if (!facts_parquet.empty()) { std::remove(facts_parquet.c_str()); }
  }

  /// `facts`, read through the Parquet scan -- a consumer that can skip reads, which is what makes
  /// a trace's stop point observable at all (see the file comment).
  [[nodiscard]] std::string facts_scan() const { return "read_parquet('" + facts_parquet + "')"; }

  /// `money`, read through the Parquet scan, for the same reason as `facts_scan()`.
  [[nodiscard]] std::string money_scan() const { return "read_parquet('" + money_parquet + "')"; }

  std::string facts_parquet;
  std::string money_parquet;
  scoped_temp_db_path _db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - flag off registers no refinement slot",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, false);
  auto plan =
    generate_sirius_plan(*con, "SELECT id, v FROM " + facts_scan() + " ORDER BY v LIMIT 10");
  // Feature dark: no channel is minted for Top-N, so the scan carries no producer and the
  // scan-route wrapper is elided entirely.
  CHECK(scan_route_filters(plan.get()).empty());
  CHECK(registered_slots(plan.get()).empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - single admitted key binds its scan",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto plan =
    generate_sirius_plan(*con, "SELECT id, v FROM " + facts_scan() + " ORDER BY v LIMIT 10");
  auto slots = single_scan_slots(plan.get());

  // One producer, one slot; a lone key references no further ordinal.
  REQUIRE(slots.size() == 1);
  CHECK(slots.front().referenced_ordinals.empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - a scan that cannot skip reads is not sited",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();

  // Scenario 2, and the siting rule's whole point. This is scenario 1's query over the *native*
  // table: the trace reaches the scan exactly as it does on Parquet, but the DuckDB-native reader
  // has no read-time filter path, so a predicate published there prunes nothing before decode --
  // it buys one O(rows) compaction pass at the scan to save the sink prefilter an identical pass
  // over the same rows. Measured, that shape regresses (+6.8% adversary, +11.5% at 0.52% rows
  // kept), which is why "is it a scan?" is no longer accepted as a proxy for "can it skip reads?".
  auto plan        = generate_sirius_plan(*con, "SELECT id, v FROM facts ORDER BY v LIMIT 10");
  auto const after = state->get_dynamic_filter_stats_snapshot();

  // The producer still exists and still self-consumes; only the external target is refused.
  CHECK(after.top_n_producers_eligible > before.top_n_producers_eligible);
  CHECK(after.top_n_sites_skipped_no_work_saved > before.top_n_sites_skipped_no_work_saved);
  CHECK(after.top_n_first_key_scan_targets == before.top_n_first_key_scan_targets);
  CHECK(after.top_n_lex_scan_targets == before.top_n_lex_scan_targets);
  CHECK(after.top_n_endpoint_sites_placed == before.top_n_endpoint_sites_placed);
  CHECK(registered_slots(plan.get()).empty());
  CHECK(scan_route_filters(plan.get()).empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - a scan that cannot skip reads is sited when it shields "
                 "material work",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();

  // The asymmetry the rule must preserve, on the same native table the case above refuses. The
  // predicate is over a computed expression, so DuckDB cannot lower it into the scan's table
  // filters and a FILTER operator survives between the scan and the Top-N. That filter evaluates
  // per row, so a boundary applied at the scan shields real work on every rejected row and the
  // target pays for itself -- the rule refuses sites that duplicate the sink's own pass, never a
  // backend. Collapse the rule to "refuse native scans" and this case stops binding.
  auto plan =
    generate_sirius_plan(*con, "SELECT v FROM facts WHERE v + w <> 12345 ORDER BY v LIMIT 10");
  auto const after = state->get_dynamic_filter_stats_snapshot();

  // The shape is genuinely the intended one: a FILTER, not a pushed-down table filter.
  REQUIRE(find_first(plan.get(), SiriusPhysicalOperatorType::FILTER) != nullptr);
  CHECK(after.top_n_first_key_scan_targets > before.top_n_first_key_scan_targets);
  CHECK(after.top_n_sites_skipped_no_work_saved == before.top_n_sites_skipped_no_work_saved);
  auto slots = single_scan_slots(plan.get());
  REQUIRE(slots.size() == 1);
  CHECK(slots.front().referenced_ordinals.empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - all keys reaching one scan bind as a single LEX slot",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto plan =
    generate_sirius_plan(*con, "SELECT id, v, w FROM " + facts_scan() + " ORDER BY v, w LIMIT 10");
  auto slots = single_scan_slots(plan.get());

  // Both ordinals survived every hop together, so the site carries the full tuple: one slot with
  // one referenced ordinal, distinct from the primary. The first-key bound is subsumed here, so
  // there is no second slot on this channel.
  REQUIRE(slots.size() == 1);
  REQUIRE(slots.front().referenced_ordinals.size() == 1);
  CHECK(slots.front().referenced_ordinals.front() != slots.front().primary_ordinal);
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - an unadmitted tail type degrades to the first-key layer",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  // A VARCHAR tail is outside the allowlist, so no LEX layer exists and only key zero is bound.
  auto plan =
    generate_sirius_plan(*con, "SELECT v, pay FROM " + facts_scan() + " ORDER BY v, pay LIMIT 10");
  auto slots = single_scan_slots(plan.get());

  REQUIRE(slots.size() == 1);
  CHECK(slots.front().referenced_ordinals.empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - pass-through hops still reach the scan",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  // A FILTER and a plain projection are accepted hops, so the trace bottoms out at the scan and
  // binds it rather than siting an endpoint above the filter.
  auto plan = generate_sirius_plan(
    *con, "SELECT v FROM " + facts_scan() + " WHERE id <> 3 ORDER BY v LIMIT 10");
  auto slots = single_scan_slots(plan.get());

  REQUIRE(slots.size() == 1);
  // Reaching a scan means no Top-N endpoint was spliced anywhere.
  for (auto* node : collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER)) {
    CHECK(static_cast<sirius_physical_dynamic_filter*>(node)->provenance() !=
          dynamic_filter_endpoint_provenance::top_n_endpoint);
  }
}

// Scenario 6 (UNION fan-out to both branches) has no plan-shape test: Sirius rejects set
// operations during planning ("Set operation not supported"), so no physical plan containing a
// UNION can be built today. `descent_steps` already fans the trace across a physical UNION's
// children, and that path becomes testable when set operations are supported.

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - the row producer's trace still refuses aggregates",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  // The Stage-5 anti-shortcut test. Ordering by the *grouping key* is the shape a widened row
  // trace would descend through: `group_by_key_input` maps a grouping-key output back to its input
  // column, so without the `top_n_self_trace` guard the row trace would walk through
  // HASH_GROUP_BY and bind the scan underneath -- publishing a *strict, row-level* boundary
  // against grouped input, which drops rows of groups that are in the answer and silently lowers
  // their aggregate values. Stage 5 admits this shape through a separate path rooted below the
  // aggregate with inclusive-only predicates; the row trace's refusal must survive that, and
  // anyone "fixing" a later stage by widening the hop set fails here.
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();
  auto plan         = generate_sirius_plan(
    *con, "SELECT w, min(v) AS m FROM " + facts_scan() + " GROUP BY w ORDER BY w LIMIT 5");
  auto const after = state->get_dynamic_filter_stats_snapshot();

  auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY);
  REQUIRE(aggregate != nullptr);
  REQUIRE(aggregate->children.size() == 1);

  // The hop itself, at the source: refused under the Top-N self-trace and accepted under the join
  // policy, so the refusal is the policy bit rather than an accident of this plan's layout.
  sirius::planner::descent_policy const self_trace{.descend_build_blocks = false,
                                                   .top_n_self_trace     = true};
  CHECK(sirius::planner::descent_steps(*aggregate, 0, self_trace).empty());
  CHECK_FALSE(
    sirius::planner::descent_steps(*aggregate, 0, sirius::planner::descent_policy{}).empty());

  // And in the built plan: the scan below the aggregate carries exactly *one* refinement slot,
  // bound by exactly one producer. A row trace that descended the aggregate would bind the same
  // scan a second time, so a second slot here is the visible symptom of the widening.
  // (Re-tracing the finished tree would not show it: the aggregate is wrapped into its merge
  // chain after this plan was traced, so a post-hoc trace stops higher for unrelated reasons.)
  auto slots = single_scan_slots(plan.get());
  CHECK(slots.size() == 1);
  CHECK(after.top_n_first_key_scan_targets - before.top_n_first_key_scan_targets == 1);
  CHECK(after.top_n_lex_scan_targets == before.top_n_lex_scan_targets);
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - a grouping-key order binds the scan below the aggregate",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();

  // Scenario 15. Every ORDER BY key is a grouping key, nothing filters between the aggregate and
  // the Top-N, so the group-key producer is admitted and traces from the aggregate's *input* into
  // the scan feeding it.
  auto plan = generate_sirius_plan(
    *con, "SELECT w, min(v) AS m FROM " + facts_scan() + " GROUP BY w ORDER BY w LIMIT 5");
  auto const after = state->get_dynamic_filter_stats_snapshot();

  CHECK(after.top_n_group_producers_eligible > before.top_n_group_producers_eligible);
  CHECK(after.top_n_group_producers_rejected == before.top_n_group_producers_rejected);

  auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY);
  REQUIRE(aggregate != nullptr);
  REQUIRE(aggregate->children.size() == 1);

  // One slot, on the scan feeding the aggregate -- a single key references no further ordinal.
  auto const filters = scan_route_filters(plan.get());
  REQUIRE(filters.size() == 1);
  CHECK(contains_operator(aggregate->children[0].get(), filters.front()));
  auto slots = single_scan_slots(plan.get());
  REQUIRE(slots.size() == 1);
  CHECK(slots.front().referenced_ordinals.empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - the group-key producer binds the aggregate-input column, not "
                 "the top-n ordinal",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  // The ORDER BY ordinal is remapped twice before it becomes the column the prefilter compares:
  // through the projection above the aggregate, then through the aggregate's `group_idx`. Get it
  // wrong and the predicate stops being group-uniform -- it drops rows *within* a surviving group,
  // which is the corruption this whole stage is built to avoid.
  //
  // Selecting the aggregate before the grouping key makes the first remap non-trivial: the Top-N
  // sees `w` at ordinal 1 while the aggregate emits it at ordinal 0. Every other group-key
  // scenario in the repo has an identity remap, so without this shape a producer that used the
  // pre-remap ordinal would bind the same column anyway and no test would notice.
  auto plan = generate_sirius_plan(
    *con, "SELECT min(v) AS m, w FROM " + facts_scan() + " GROUP BY w ORDER BY w LIMIT 5");

  auto* top_n = find_first(plan.get(), SiriusPhysicalOperatorType::TOP_N);
  REQUIRE(top_n != nullptr);
  auto const& order = top_n->Cast<sirius::op::sirius_physical_top_n>().orders.front();
  auto const top_n_ordinal =
    static_cast<std::size_t>(order.expression->Cast<BoundReferenceExpression>().index);
  // The shape is genuinely non-identity, so the assertion below can distinguish the two ordinals.
  REQUIRE(top_n_ordinal == 1);

  auto slots = single_scan_slots(plan.get());
  REQUIRE(slots.size() == 1);
  CHECK(slots.front().referenced_ordinals.empty());
  // Bound on the grouping key's own column, which is column 0 of the scan feeding the aggregate.
  // Carrying the Top-N ordinal down instead would bind column 1 -- `min(v)`'s input -- and prune
  // by the wrong column.
  CHECK(slots.front().primary_ordinal != top_n_ordinal);
  CHECK(slots.front().primary_ordinal == 0);
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - two grouping keys bind one LEX slot below the aggregate",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();

  // The multi-key group path: both ordinals survive the hops together, so the site carries the
  // full tuple and the producer publishes an inclusive LEX predicate there. A single-key shape
  // never plans a LEX target at all, so this is the only place the multi-key remap, the LEX target
  // creation, and the two-ordinal slot are exercised on a real plan.
  auto plan = generate_sirius_plan(
    *con,
    "SELECT v, w, count(*) AS c FROM " + facts_scan() + " GROUP BY v, w ORDER BY v, w LIMIT 5");
  auto const after = state->get_dynamic_filter_stats_snapshot();

  CHECK(after.top_n_group_producers_eligible > before.top_n_group_producers_eligible);
  CHECK(after.top_n_lex_scan_targets > before.top_n_lex_scan_targets);

  auto* aggregate = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY);
  REQUIRE(aggregate != nullptr);
  REQUIRE(aggregate->children.size() == 1);
  auto const filters = scan_route_filters(plan.get());
  REQUIRE(filters.size() == 1);
  CHECK(contains_operator(aggregate->children[0].get(), filters.front()));

  // One slot carrying both ordinals, in ORDER BY key order. The values matter, not just their
  // distinctness: a target's `component_ordinals` are zipped against the coordinator's keys in key
  // order, so a permuted remap would compare key zero's boundary component against key one's
  // column. The scan feeding this aggregate projects (v, w), so key zero (`v`) is the primary
  // ordinal 0 and key one (`w`) is the referenced ordinal 1.
  auto slots = single_scan_slots(plan.get());
  REQUIRE(slots.size() == 1);
  REQUIRE(slots.front().referenced_ordinals.size() == 1);
  CHECK(slots.front().primary_ordinal == 0);
  CHECK(slots.front().referenced_ordinals.front() == 1);
  CHECK(after.top_n_first_key_scan_targets == before.top_n_first_key_scan_targets);
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - a K beyond the group-key cap admits no producer",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();

  // Above the cap the seam costs more per batch than the looser boundary can ever repay, so the
  // shape is refused rather than run. The row producer over the same limit stays eligible: it
  // extracts one row's key tuple per batch whatever K is.
  auto const over_cap = sirius::op::top_n_group_key_producer::k_max_admitted_k + 1;
  auto plan           = generate_sirius_plan(
    *con,
    "SELECT w, min(v) AS m FROM facts GROUP BY w ORDER BY w LIMIT " + std::to_string(over_cap));
  auto const after = state->get_dynamic_filter_stats_snapshot();

  CHECK(after.top_n_group_producers_rejected > before.top_n_group_producers_rejected);
  CHECK(after.top_n_group_producers_eligible == before.top_n_group_producers_eligible);
  CHECK(after.top_n_producers_eligible > before.top_n_producers_eligible);
  CHECK(registered_slots(plan.get()).empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - decimal keys are admitted by precision band",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);

  SECTION("DECIMAL32 (p<=9) binds its scan")
  {
    auto const before = state->get_dynamic_filter_stats_snapshot();
    auto plan         = generate_sirius_plan(
      *con, "SELECT small_dec FROM " + money_scan() + " ORDER BY small_dec LIMIT 5");
    auto const after = state->get_dynamic_filter_stats_snapshot();
    CHECK(after.top_n_producers_eligible > before.top_n_producers_eligible);
    CHECK(after.top_n_producers_rejected == before.top_n_producers_rejected);
    auto slots = single_scan_slots(plan.get());
    REQUIRE(slots.size() == 1);
  }

  SECTION("DECIMAL64 (p<=18) binds its scan")
  {
    auto const before = state->get_dynamic_filter_stats_snapshot();
    auto plan         = generate_sirius_plan(
      *con, "SELECT big_dec FROM " + money_scan() + " ORDER BY big_dec LIMIT 5");
    auto const after = state->get_dynamic_filter_stats_snapshot();
    CHECK(after.top_n_producers_eligible > before.top_n_producers_eligible);
    CHECK(after.top_n_producers_rejected == before.top_n_producers_rejected);
    auto slots = single_scan_slots(plan.get());
    REQUIRE(slots.size() == 1);
  }

  SECTION("DECIMAL128 (p>=19) forms no producer on the path where it is reachable")
  {
    // The duckdb-native scan refuses a DECIMAL128 column before planning gets this far, but the
    // parquet path has no precision gate, so this is where the refusal is live in production.
    // The components widen to int64 and the kernel reads by width 1/2/4/8: a 16-byte scaled
    // integer has no case there and would be read as zero rather than rejected.
    auto const before = state->get_dynamic_filter_stats_snapshot();
    auto plan         = generate_sirius_plan(
      *con, "SELECT huge_dec FROM " + money_scan() + " ORDER BY huge_dec LIMIT 5");
    auto const after = state->get_dynamic_filter_stats_snapshot();
    CHECK(after.top_n_producers_rejected > before.top_n_producers_rejected);
    CHECK(after.top_n_producers_eligible == before.top_n_producers_eligible);
    CHECK(registered_slots(plan.get()).empty());
  }

  SECTION("a decimal tail beside an integer head enables the LEX layer")
  {
    auto plan = generate_sirius_plan(
      *con, "SELECT id, small_dec FROM " + money_scan() + " ORDER BY id, small_dec LIMIT 5");
    auto slots = single_scan_slots(plan.get());
    REQUIRE(slots.size() == 1);
    // Both keys admitted, so the site carries the full tuple rather than degrading to first-key.
    REQUIRE(slots.front().referenced_ordinals.size() == 1);
    CHECK(slots.front().referenced_ordinals.front() != slots.front().primary_ordinal);
  }
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - an aggregate-output order key admits no group-key producer",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();

  // Scenario 16 (the Q3 shape). `s` is computed by the aggregate, so no input row's value
  // determines it and no input row can be ruled out by a bound on it.
  auto plan =
    generate_sirius_plan(*con, "SELECT w, sum(v) AS s FROM facts GROUP BY w ORDER BY s LIMIT 5");
  auto const after = state->get_dynamic_filter_stats_snapshot();

  CHECK(after.top_n_group_producers_rejected > before.top_n_group_producers_rejected);
  CHECK(after.top_n_group_producers_eligible == before.top_n_group_producers_eligible);
  CHECK(registered_slots(plan.get()).empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - a filter between aggregate and top-n admits no group-key "
                 "producer",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();

  // Scenario 17. DuckDB lowers HAVING to a FILTER above the aggregate, and a predicate there can
  // eliminate whole groups -- so a witnessed distinct key no longer proves a surviving group and
  // the K-distinct proof fails. The refusal is structural: the predicate is never classified.
  auto plan = generate_sirius_plan(
    *con, "SELECT w, min(v) AS m FROM facts GROUP BY w HAVING min(v) > 0 ORDER BY w LIMIT 5");
  auto const after = state->get_dynamic_filter_stats_snapshot();

  REQUIRE(find_first(plan.get(), SiriusPhysicalOperatorType::FILTER) != nullptr);
  CHECK(after.top_n_group_producers_rejected > before.top_n_group_producers_rejected);
  CHECK(after.top_n_group_producers_eligible == before.top_n_group_producers_eligible);
  CHECK(registered_slots(plan.get()).empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - flag off admits no group-key producer either",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, false);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();

  auto plan =
    generate_sirius_plan(*con, "SELECT w, min(v) AS m FROM facts GROUP BY w ORDER BY w LIMIT 5");
  auto const after = state->get_dynamic_filter_stats_snapshot();

  CHECK(after.top_n_group_producers_eligible == before.top_n_group_producers_eligible);
  CHECK(after.top_n_group_producers_rejected == before.top_n_group_producers_rejected);
  CHECK(scan_route_filters(plan.get()).empty());
  CHECK(registered_slots(plan.get()).empty());
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - a join disruptor stops the trace and sites nothing",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  // The self-trace refuses join hops, so the trace stops at the join output. Only pass-through
  // hops separate that stop point from the sink, so the site is immaterial and is skipped -- the
  // sink prefilter already applies the same predicate over the same rows.
  //
  // Facts reads Parquet so the refusal itself stays observable: a wrongly widened join hop would
  // continue into the scan and mint a slot, failing the emptiness check below. On a native table
  // that regression would be absorbed by the siting rule's skip and the test would pass vacuously.
  auto plan = generate_sirius_plan(
    *con, "SELECT f.v FROM " + facts_scan() + " f JOIN dim d ON f.id = d.dk ORDER BY f.v LIMIT 10");

  CHECK(registered_slots(plan.get()).empty());
  for (auto* node : collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER)) {
    CHECK(static_cast<sirius_physical_dynamic_filter*>(node)->provenance() !=
          dynamic_filter_endpoint_provenance::top_n_endpoint);
  }
}

TEST_CASE_METHOD(top_n_plan_shape_fixture,
                 "top-n plan shape - split keys across a join defer the LEX site and endpoint the "
                 "first-key bound",
                 "[plan_tree_shape][isolated_context][top_n]")
{
  top_n_filter_switch_guard flag(*con, true);
  auto* state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  REQUIRE(state != nullptr);
  auto const before = state->get_dynamic_filter_stats_snapshot();

  // Scenario 10: the keys come from opposite sides of a join, so the all-keys trace stops at the
  // join output while key zero continues into its own scan. A residual FILTER above the join
  // makes that stop point material, so the LEX site is worth having -- and is deferred, not
  // cost-gate skipped, because splicing it needs multi-ordinal placement (Stage 7).
  auto plan        = generate_sirius_plan(*con,
                                   "SELECT f.v, d.dv FROM facts f JOIN dim d ON f.id = d.dk "
                                          "WHERE f.v + d.dv <> 12345 ORDER BY f.v, d.dv LIMIT 10");
  auto const after = state->get_dynamic_filter_stats_snapshot();

  // The staged gap is reported as itself, never folded into the cost-gate skip counter.
  CHECK(after.top_n_lex_endpoint_sites_deferred > before.top_n_lex_endpoint_sites_deferred);
  CHECK(after.top_n_lex_scan_targets == before.top_n_lex_scan_targets);

  // The minimal hop set refuses join hops for *both* traces, so key zero also stops at the join
  // rather than continuing into its own scan (that widening is Stage 7). Its stop point is
  // material, and a single-ordinal endpoint is exactly what `place_endpoint` can splice -- so the
  // inclusive first-key bound is applied at a sited endpoint above the join.
  auto const endpoints = collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER);
  auto const top_n_endpoints =
    std::count_if(endpoints.begin(), endpoints.end(), [](sirius_physical_operator* node) {
      return static_cast<sirius_physical_dynamic_filter*>(node)->provenance() ==
             dynamic_filter_endpoint_provenance::top_n_endpoint;
    });
  CHECK(top_n_endpoints == 1);
  CHECK(after.top_n_endpoint_sites_placed > before.top_n_endpoint_sites_placed);

  // That endpoint carries the first-key layer alone: one slot, no referenced ordinal.
  for (auto* node : endpoints) {
    auto* filter = static_cast<sirius_physical_dynamic_filter*>(node);
    if (filter->provenance() != dynamic_filter_endpoint_provenance::top_n_endpoint) { continue; }
    auto const channel = filter->filters();
    REQUIRE(channel != nullptr);
    auto const slots = channel->refinement_slots();
    REQUIRE(slots.size() == 1);
    CHECK(slots.front().referenced_ordinals.empty());
  }

  // No scan was bound: neither trace got past the join.
  CHECK(registered_slots(plan.get()).empty());
  CHECK(after.top_n_first_key_scan_targets == before.top_n_first_key_scan_targets);
}
