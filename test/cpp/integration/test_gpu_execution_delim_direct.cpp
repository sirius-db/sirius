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

// End-to-end correctness of the delim-direct lowering (sirius_plan_delim_direct): equality
// EXISTS / NOT EXISTS run on the GPU through the direct semi/anti hash join and must match
// DuckDB CPU exactly, with emphasis on the NULL semantics the rewrite has to preserve:
//   - EXISTS excludes NULL-keyed outer rows (NULL = x is never true);
//   - NOT EXISTS keeps NULL-keyed outer rows (no match is possible);
//   - NULL-keyed inner rows never match anything;
// plus the degenerate cardinalities (runtime-empty inner side, all-match, no-match) and the
// enable_delim_direct_lowering knob A/B.
//
// The lowering only runs when DuckDB plans a DELIM join in the first place; at this toy scale
// DuckDB's subquery unnesting plans a bare semi/anti join instead unless the subquery carries an
// inner-side predicate. The predicates below (qty > 0, qty > -1000, qty % 1000 > -1000, ...)
// exist exactly for that — the all-matching ones keep every inner row in play so they change
// only the plan, never the membership outcome. A predicate comparing a plain-`=`-correlated
// column with a constant is not enough: DuckDB's filter equivalence rewrites it onto the dedup
// keys, un-baring the DELIM_GET so the classifier refuses (sandwich_shape) — hence the opaque
// modulo form wherever the filtered column is itself a correlation key. Every case asserts the
// full precondition via require_lowerable_delim_plan (DELIM planned on this connection AND
// classified eligible), so neither a DuckDB planning change nor a classifier change can silently
// turn a case into a stock-path or delim-machinery test.
//
// Every query goes through the shared file-backed GpuExecutionFixture, which runs it once on
// the GPU (asserting a real GPU execution with no fallback) and once on DuckDB CPU, then
// compares the results (order-insensitive).

#include "planner/sirius_plan_delim_direct.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/planner.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

duckdb::LogicalComparisonJoin* find_delim_join(duckdb::LogicalOperator* op)
{
  if (!op) { return nullptr; }
  if (op->type == duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN) {
    return &op->Cast<duckdb::LogicalComparisonJoin>();
  }
  for (auto& child : op->children) {
    if (auto* found = find_delim_join(child.get())) { return found; }
  }
  return nullptr;
}

// Outer rows with NULL keys (ids 6, 7) and duplicate keys (10 twice); inner rows with a NULL
// key, duplicate matching keys (10 twice), and a key with no outer counterpart (99).
class DelimDirectFixture : public sirius::test::GpuExecutionFixture {
 public:
  DelimDirectFixture()
  {
    run_ok("CREATE TABLE outer_t (id INTEGER, k INTEGER, tag VARCHAR);");
    run_ok(
      "INSERT INTO outer_t VALUES (1, 10, 'a'), (2, 20, 'b'), (3, 30, 'a'), (4, 40, 'b'), "
      "(5, 50, 'a'), (6, NULL, 'b'), (7, NULL, 'a'), (8, 10, 'b');");
    run_ok("CREATE TABLE inner_t (k INTEGER, qty INTEGER);");
    run_ok("INSERT INTO inner_t VALUES (10, 1), (10, 2), (20, 3), (NULL, 4), (99, 5), (30, -1);");
    run_ok("CHECKPOINT;");
  }

  /// Require that on THIS connection (same tables, statistics, and optimizer settings as the
  /// GPU run) DuckDB's optimized logical plan for @p sql contains a DELIM join AND that the
  /// delim-direct classifier accepts it — so the GPU execution that follows provably runs
  /// through the lowered direct join (see the file comment). Planned manually on the CPU path;
  /// a refusal failure prints the typed reason.
  void require_lowerable_delim_plan(const std::string& sql)
  {
    run_ok("SET gpu_execution = false;");
    run_ok("BEGIN TRANSACTION;");
    try {
      auto& context = *con->context;
      duckdb::Parser parser(context.GetParserOptions());
      parser.ParseQuery(sql);
      REQUIRE(parser.statements.size() == 1);
      duckdb::Planner planner(context);
      planner.CreatePlan(std::move(parser.statements[0]));
      REQUIRE(planner.plan);
      duckdb::Optimizer optimizer(*planner.binder, context);
      auto plan = optimizer.Optimize(std::move(planner.plan));
      plan->ResolveOperatorTypes();
      duckdb::ColumnBindingResolver resolver;
      resolver.VisitOperator(*plan);

      auto* delim = find_delim_join(plan.get());
      REQUIRE(delim != nullptr);
      const std::string refusal =
        sirius::planner::to_string(sirius::planner::classify_delim_direct_lowering(*delim).refusal);
      CHECK(refusal == "none");
    } catch (...) {
      con->Query("ROLLBACK;");
      con->Query("SET gpu_execution = true;");
      throw;
    }
    run_ok("ROLLBACK;");
    run_ok("SET gpu_execution = true;");
  }

  /// require_lowerable_delim_plan + the GPU-vs-CPU comparison: the standard shape of every
  /// case here.
  void compare_lowered_gpu_vs_cpu(const std::string& sql)
  {
    require_lowerable_delim_plan(sql);
    compare_gpu_vs_cpu(sql);
  }
};

constexpr const char* exists_sql =
  "SELECT id, tag FROM outer_t WHERE EXISTS "
  "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty > 0)";

constexpr const char* not_exists_sql =
  "SELECT id, tag FROM outer_t WHERE NOT EXISTS "
  "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty > 0)";

}  // namespace

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct EXISTS matches CPU incl. NULL correlation keys",
                 "[integration][gpu_execution][delim_direct][nulls]")
{
  // NULL-keyed outer rows (6, 7) are excluded; the NULL-keyed inner row matches nothing;
  // duplicate outer key 10 keeps both its rows; duplicate inner matches do not multiply rows.
  compare_lowered_gpu_vs_cpu(exists_sql);
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct NOT EXISTS matches CPU incl. NULL correlation keys",
                 "[integration][gpu_execution][delim_direct][nulls]")
{
  // NULL-keyed outer rows (6, 7) are KEPT: no match is possible, so NOT EXISTS is true.
  compare_lowered_gpu_vs_cpu(not_exists_sql);
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct EXISTS/NOT EXISTS with an aggregate on top",
                 "[integration][gpu_execution][delim_direct]")
{
  // The TPC-H q4 / q22 shape: membership test feeding a GROUP BY. The all-matching qty
  // predicate keeps every inner row in play (unlike the qty > 0 of exists_sql).
  compare_lowered_gpu_vs_cpu(
    "SELECT tag, count(*) FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty > -1000) GROUP BY tag");
  compare_lowered_gpu_vs_cpu(
    "SELECT tag, count(*) FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty > -1000) GROUP BY tag");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct handles a runtime-empty inner side",
                 "[integration][gpu_execution][delim_direct]")
{
  // The predicate keeps no inner rows at runtime (opaque to the optimizer's stats): EXISTS
  // yields nothing, NOT EXISTS yields every outer row (including the NULL-keyed ones).
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty % 2 = 7)");
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND qty % 2 = 7)");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct handles all-match and no-match outer sides",
                 "[integration][gpu_execution][delim_direct]")
{
  // All non-NULL outer keys match (subquery over the union of outer keys themselves; the
  // all-matching id predicate only keeps the DELIM join planned).
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM outer_t o2 WHERE o2.k = outer_t.k AND o2.id > -1000)");
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM outer_t o2 WHERE o2.k = outer_t.k AND o2.id > -1000)");
  // No outer key matches (inner keys shifted out of range at runtime).
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k + 1000 = outer_t.k)");
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k + 1000 = outer_t.k)");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct matches the knob-off delim lowering",
                 "[integration][gpu_execution][delim_direct]")
{
  // A/B the same queries through the regular delim lowering; both must match CPU. The delim
  // precondition holds for both legs (the knob only selects how Sirius lowers the delim).
  require_lowerable_delim_plan(exists_sql);
  require_lowerable_delim_plan(not_exists_sql);
  run_ok("SET enable_delim_direct_lowering = false;");
  try {
    compare_gpu_vs_cpu(exists_sql);
    compare_gpu_vs_cpu(not_exists_sql);
  } catch (...) {
    con->Query("RESET enable_delim_direct_lowering;");
    throw;
  }
  run_ok("RESET enable_delim_direct_lowering;");
  compare_gpu_vs_cpu(exists_sql);
  compare_gpu_vs_cpu(not_exists_sql);
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct two-key EXISTS/NOT EXISTS matches CPU",
                 "[integration][gpu_execution][delim_direct][nulls]")
{
  // Compound correlation (two dedup keys, both constrained), so the lowered join carries two
  // conditions and apply's per-condition key substitution executes beyond n = 1. NULL-keyed
  // outer rows are excluded by EXISTS and kept by NOT EXISTS, per key vector. (The modulo form
  // of the all-matching filter is load-bearing: qty is a correlation key — see the file
  // comment.)
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND inner_t.qty = outer_t.id "
    "AND qty % 1000 > -1000)");
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k = outer_t.k AND inner_t.qty = outer_t.id "
    "AND qty % 1000 > -1000)");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct null-safe correlation matches CPU",
                 "[integration][gpu_execution][delim_direct][nulls]")
{
  // IS NOT DISTINCT FROM correlation: NULL outer keys DO match the NULL inner key here, for
  // both the EXISTS and NOT EXISTS forms — the null-safe/null-safe pairing the classifier
  // accepts, executed end-to-end through the direct join.
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k IS NOT DISTINCT FROM outer_t.k AND qty > -1000)");
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k IS NOT DISTINCT FROM outer_t.k AND qty > -1000)");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct mixed null-safe and plain-equality correlation "
                 "matches CPU",
                 "[integration][gpu_execution][delim_direct][nulls]")
{
  // Two correlated conditions of different comparison kinds on one lowered join: the null-safe
  // key (k) lets a NULL outer key match the NULL inner key, while the plain `=` key (id) never
  // matches NULL — the pairing the null_safety proof admits, executed end-to-end. (The modulo
  // form of the all-matching filter is load-bearing: qty is a correlation key — see the file
  // comment.)
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k IS NOT DISTINCT FROM outer_t.k "
    "AND inner_t.qty = outer_t.id AND qty % 1000 > -1000)");
  compare_lowered_gpu_vs_cpu(
    "SELECT id FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_t WHERE inner_t.k IS NOT DISTINCT FROM outer_t.k "
    "AND inner_t.qty = outer_t.id AND qty % 1000 > -1000)");
}

TEST_CASE_METHOD(DelimDirectFixture,
                 "gpu_execution delim-direct executes over a row-bounded multi-partition probe",
                 "[integration][gpu_execution][delim_direct][fold_limit]")
{
  // The lowered join is right-family, so a CONCAT folds its whole probe partition into one cuDF
  // table and the partition count is what bounds that fold (INV-FOLD, op/fold_limits.hpp).
  // max_concat_fold_rows brings that bound down to a volume CI can reach.
  //
  // The case is self-proving: 300k probe rows arrive in three scan batches (one per DuckDB row
  // group, kept separate by the minimum scan_task_batch_size), and a 200k-row limit means a
  // single-partition plan would fold 300k rows and be refused. The query can only succeed if the
  // row-aware floor spread the probe across partitions first. Dynamic filters are off so the
  // probe -- the side that folds -- is also the side that drives the count, which is the regime
  // the floor governs.
  //
  // The 5000-value key domain is load-bearing: three partitions of a 300k-row probe average 100k
  // rows against a 200k limit, so the test must not depend on the hash landing those keys evenly.
  // A handful of keys could put most of the probe in one bucket and fail on the hash function
  // rather than on the behaviour under test.
  run_ok(
    "CREATE TABLE inner_many AS SELECT ((i % 5000) * 10)::INTEGER AS k, i::INTEGER AS qty "
    "FROM range(300000) t(i);");
  run_ok("CHECKPOINT;");

  const std::string exists_many =
    "SELECT id, tag FROM outer_t WHERE EXISTS "
    "(SELECT 1 FROM inner_many WHERE inner_many.k = outer_t.k AND qty > 0)";
  const std::string not_exists_many =
    "SELECT id, tag FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM inner_many WHERE inner_many.k = outer_t.k AND qty > 0)";

  run_ok("SET enable_dynamic_filter = false;");
  try {
    sirius::test::scoped_setting scan_batches(*con, "scan_task_batch_size", 1);
    sirius::test::scoped_setting fold_rows(*con, "max_concat_fold_rows", 200000);
    compare_lowered_gpu_vs_cpu(exists_many);
    compare_lowered_gpu_vs_cpu(not_exists_many);
  } catch (...) {
    con->Query("RESET enable_dynamic_filter;");
    throw;
  }
  run_ok("RESET enable_dynamic_filter;");
}
