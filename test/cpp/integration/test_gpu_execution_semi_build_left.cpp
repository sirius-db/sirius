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

// GPU-vs-CPU correctness for RIGHT_SEMI / RIGHT_ANTI joins built on the small kept side while the
// large side streams through (semi_join_kept_build_ratio). DuckDB flips SEMI / ANTI to RIGHT_SEMI /
// RIGHT_ANTI when the kept side is the smaller one, which is the shape these cases produce.

#include "op/sirius_physical_hash_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/settings.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <string>
#include <vector>

namespace {

/// Plans @p sql with the Sirius plan generator on @p con 's context, without running it.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> sirius_plan(duckdb::Connection& con,
                                                                     std::string const& sql)
{
  auto& context = *con.context;
  con.Query("BEGIN TRANSACTION");
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> result;
  try {
    duckdb::Parser parser(context);
    parser.ParseQuery(sql);
    REQUIRE(!parser.statements.empty());
    duckdb::Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);
    auto plan = std::move(planner.plan);
    if (duckdb::Settings::Get<duckdb::EnableOptimizerSetting>(context)) {
      duckdb::Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }
    plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);
    sirius::planner::sirius_physical_plan_generator generator(context);
    result = generator.create_plan(std::move(plan));
  } catch (...) {
    con.Query("ROLLBACK");
    throw;
  }
  con.Query("COMMIT");
  return result;
}

void collect_hash_joins(sirius::op::sirius_physical_operator* root,
                        std::vector<sirius::op::sirius_physical_hash_join*>& out)
{
  if (root == nullptr) { return; }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::HASH_JOIN) {
    out.push_back(&root->Cast<sirius::op::sirius_physical_hash_join>());
  }
  for (auto& child : root->children) {
    collect_hash_joins(child.get(), out);
  }
}

/// Number of RIGHT_SEMI / RIGHT_ANTI hash joins in the plan of @p sql that build on the kept side.
int kept_build_joins(duckdb::Connection& con, std::string const& sql)
{
  auto plan = sirius_plan(con, sql);
  REQUIRE(plan);
  std::vector<sirius::op::sirius_physical_hash_join*> joins;
  collect_hash_joins(plan.get(), joins);
  int kept = 0;
  for (auto* j : joins) {
    if (j->kept_build_streaming()) { ++kept; }
  }
  return kept;
}

// `big` (60000 rows, the streamed side) repeats keys 0..2999 twenty times, plus NULL keys. For keys
// below 500 every `big` row carries s = k % 7, which equals the kept row's s, so a residual
// `small.s <> big.s` rejects all of their equality matches. `small` (2000 rows, the kept side)
// holds even keys 0..3998, so keys at or above 3000 never match, plus NULL keys.
class SemiBuildLeftFixture : public sirius::test::GpuExecutionFixture {
 public:
  SemiBuildLeftFixture()
  {
    run_ok(
      "COPY (SELECT CASE WHEN i % 1009 = 0 THEN NULL ELSE (i % 3000)::INTEGER END AS k, "
      "(CASE WHEN i % 3000 < 500 THEN (i % 3000) % 7 ELSE i % 7 END)::INTEGER AS s, "
      "i::BIGINT AS v FROM range(60000) t(i)) TO " +
      dir.file_literal("big.parquet") + " (FORMAT PARQUET, ROW_GROUP_SIZE 4096);");
    run_ok(
      "COPY (SELECT CASE WHEN i % 97 = 0 THEN NULL ELSE (2 * i)::INTEGER END AS k, "
      "((2 * i) % 7)::INTEGER AS s, i::BIGINT AS id FROM range(2000) t(i)) TO " +
      dir.file_literal("small.parquet") + " (FORMAT PARQUET, ROW_GROUP_SIZE 512);");
    run_ok("CREATE VIEW big AS SELECT * FROM read_parquet(" + dir.file_literal("big.parquet") +
           ");");
    run_ok("CREATE VIEW small AS SELECT * FROM read_parquet(" + dir.file_literal("small.parquet") +
           ");");
  }

  sirius::test::scratch_dir const dir{"semi_build_left"};
};

std::vector<std::string> const kept_queries = {
  // Equality only.
  "SELECT count(*), sum(id) FROM small SEMI JOIN big ON small.k = big.k",
  "SELECT count(*), sum(id) FROM small ANTI JOIN big ON small.k = big.k",
  // Equality plus a residual on another column.
  "SELECT count(*), sum(id) FROM small SEMI JOIN big ON small.k = big.k AND small.s <> big.s",
  "SELECT count(*), sum(id) FROM small ANTI JOIN big ON small.k = big.k AND small.s <> big.s",
  // Kept rows themselves, not just aggregates.
  "SELECT id, k, s FROM small SEMI JOIN big ON small.k = big.k AND small.s <> big.s",
  "SELECT id, k, s FROM small ANTI JOIN big ON small.k = big.k AND small.s <> big.s",
};

}  // namespace

TEST_CASE_METHOD(SemiBuildLeftFixture,
                 "semi_build_left - planner builds RIGHT_SEMI / RIGHT_ANTI on the kept side",
                 "[integration][gpu_execution][join][semi_build_left]")
{
  for (auto const& q : kept_queries) {
    INFO(q);
    REQUIRE(kept_build_joins(*con, q) == 1);
  }
  sirius::test::scoped_setting const off{*con, "semi_join_kept_build_ratio", 0};
  for (auto const& q : kept_queries) {
    INFO(q);
    REQUIRE(kept_build_joins(*con, q) == 0);
  }
}

TEST_CASE_METHOD(SemiBuildLeftFixture,
                 "semi_build_left - semi / anti with and without residual match the CPU",
                 "[integration][gpu_execution][join][semi_build_left]")
{
  auto const ratio = GENERATE(0, 1);
  INFO("semi_join_kept_build_ratio = " << ratio);
  sirius::test::scoped_setting const kept{*con, "semi_join_kept_build_ratio", std::uint64_t(ratio)};
  for (auto const& q : kept_queries) {
    INFO(q);
    compare_gpu_vs_cpu(q);
  }
}

TEST_CASE_METHOD(SemiBuildLeftFixture,
                 "semi_build_left - many probe batches and partitions match the CPU",
                 "[integration][gpu_execution][join][semi_build_left]")
{
  // 16 KiB scan and CONCAT targets split `big` into many probe batches; tiny partition and
  // hash-table budgets spread the kept build over several BUILD_PROBE slots.
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 16384};
  sirius::test::scoped_setting const concat_batch{*con, "concat_batch_bytes", 16384};
  auto const partition_bytes = GENERATE(std::uint64_t{1} << 30, std::uint64_t{4096});
  INFO("hash_partition_bytes = " << partition_bytes);
  sirius::test::scoped_setting const partition{*con, "hash_partition_bytes", partition_bytes};
  sirius::test::scoped_setting const table_budget{*con, "max_build_hash_table_bytes", 1024};
  sirius::test::scoped_setting const resident_budget{*con, "max_build_probe_resident_bytes", 1024};
  for (auto const& q : kept_queries) {
    INFO(q);
    compare_gpu_vs_cpu(q);
  }
}

TEST_CASE_METHOD(SemiBuildLeftFixture,
                 "semi_build_left - empty kept or streamed side matches the CPU",
                 "[integration][gpu_execution][join][semi_build_left]")
{
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 16384};
  for (auto const* kind : {"SEMI", "ANTI"}) {
    std::string const k = kind;
    INFO(k);
    // Empty streamed side: SEMI keeps nothing, ANTI keeps every kept row.
    compare_gpu_vs_cpu("SELECT count(*), sum(id) FROM small " + k +
                       " JOIN (SELECT * FROM big WHERE v < 0) b ON small.k = b.k");
    compare_gpu_vs_cpu("SELECT count(*), sum(id) FROM small " + k +
                       " JOIN (SELECT * FROM big WHERE v < 0) b ON small.k = b.k AND small.s <> "
                       "b.s");
    // Empty kept side.
    compare_gpu_vs_cpu("SELECT count(*), sum(id) FROM (SELECT * FROM small WHERE id < 0) s " + k +
                       " JOIN big ON s.k = big.k");
    // Streamed side with only NULL keys.
    compare_gpu_vs_cpu("SELECT count(*), sum(id) FROM small " + k +
                       " JOIN (SELECT * FROM big WHERE k IS NULL) b ON small.k = b.k");
  }
}

TEST_CASE_METHOD(SemiBuildLeftFixture,
                 "semi_build_left - correlated EXISTS / NOT EXISTS with a residual match the CPU",
                 "[integration][gpu_execution][join][semi_build_left]")
{
  // The TPC-H Q21 shape: a correlated (NOT) EXISTS on an equal key and a different second column.
  auto const cte = GENERATE("true", "false");
  INFO("delim_join_as_cte = " << cte);
  run_ok(std::string("SET delim_join_as_cte = ") + cte + ";");
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 16384};
  compare_gpu_vs_cpu(
    "SELECT count(*), sum(id) FROM small WHERE EXISTS (SELECT 1 FROM big WHERE big.k = small.k AND "
    "big.s <> small.s)");
  compare_gpu_vs_cpu(
    "SELECT count(*), sum(id) FROM small WHERE NOT EXISTS (SELECT 1 FROM big WHERE big.k = "
    "small.k AND big.s <> small.s)");
  compare_gpu_vs_cpu(
    "SELECT count(*), sum(id) FROM small WHERE EXISTS (SELECT 1 FROM big WHERE big.k = small.k AND "
    "big.s <> small.s) AND NOT EXISTS (SELECT 1 FROM big WHERE big.k = small.k AND big.s <> "
    "small.s AND big.v > 50000)");
  run_ok("RESET delim_join_as_cte;");
}
