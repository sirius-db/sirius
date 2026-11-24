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

// Answers of the IN-subquery SUM reuse rewrite (`rewrite_repeated_scans`) on the GPU, with the
// rewrite on and off, against DuckDB on the CPU. Plan-level cases, including shapes the GPU does
// not run, are in test/cpp/planner/test_scan_reuse_rewrites.cpp.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/scoped_sirius_setting.hpp>
#include <utils/tpch_queries.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace {

/// Whether a query's rewrite must fire, must not fire (it would change the answer, or the rule does
/// not cover it), or may do either (the answer is the same both ways).
enum class reuse { fires, declines, either };

/// TPC-H Q18 without its ORDER BY and LIMIT (ties would make the kept rows engine-dependent), with
/// its tables, aggregates and predicates as parameters. An empty `groups` gives an ungrouped outer
/// aggregate.
struct q18_query {
  std::string outer_aggregates  = "sum(l_quantity)";
  std::string outer_lineitem    = "syn.lineitem";
  std::string subquery_lineitem = "syn.lineitem";
  std::string subquery_where    = "";
  std::string subquery_groups   = "l_orderkey";
  std::string having            = "having sum(l_quantity) > 120";
  std::string membership        = "o_orderkey in";
  std::string outer_where       = "";
  std::string groups            = "c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice";

  [[nodiscard]] std::string sql() const
  {
    return "select " + (groups.empty() ? "" : groups + ", ") + outer_aggregates +
           " from syn.customer, syn.orders, " + outer_lineitem + " where " + membership +
           " (select l_orderkey from " + subquery_lineitem + " " + subquery_where + " group by " +
           subquery_groups + " " + having +
           ") and c_custkey = o_custkey and o_orderkey = l_orderkey" + outer_where +
           (groups.empty() ? "" : " group by " + groups);
  }
};

// TPC-H's customer, orders and lineitem as Parquet views, as TPC-H runs read them, and a small
// schema `syn` of the same tables with NULL keys on both sides of the join, orders that share a
// key, and an order whose quantities are all NULL. `syn` also reads the same lineitem rows
// hive-partitioned, through views that filter or rename them, and with other reader options, plus
// tables with keys and sums of other types.
class ScanReuseFixture : public sirius::test::GpuExecutionFixture {
 public:
  ScanReuseFixture()
  {
    enable_guard.emplace(*con, "enable_scan_reuse_rewrites", true);
    auto const tpch = std::filesystem::path(__FILE__).parent_path() / "data/parquet";
    for (char const* table : {"customer", "orders", "lineitem"}) {
      run_ok("CREATE VIEW " + std::string(table) + " AS SELECT * FROM read_parquet(" +
             sirius::test::sql_literal((tpch / (std::string(table) + ".parquet")).string()) + ");");
    }

    run_ok("CREATE SCHEMA syn;");
    auto const lineitem = std::string(
      "SELECT CASE WHEN i % 31 = 0 THEN NULL ELSE (i % 400)::BIGINT END AS l_orderkey, "
      "(i % 7)::INTEGER AS l_linenumber, CASE WHEN i % 400 = 9 THEN NULL ELSE "
      "((i * 7) % 50 + 1)::DECIMAL(15, 2) END AS l_quantity, ((i * 13) % 9000 / "
      "100)::DECIMAL(15, 2) AS l_extendedprice, DATE '1994-01-01' + (i % 700)::INTEGER "
      "AS l_shipdate, (i % 3)::INTEGER AS l_part FROM range(4000) t(i)");
    parquet_view("lineitem", lineitem);
    parquet_view("lineitem2", lineitem);
    auto const file = data.file_literal("lineitem.parquet");
    run_ok("CREATE VIEW syn.lineitem_filtered AS SELECT * FROM read_parquet(" + file +
           ") WHERE l_linenumber <> 3;");
    run_ok("CREATE VIEW syn.lineitem_binary_as_string AS SELECT * FROM read_parquet(" + file +
           ", binary_as_string = true);");
    auto const hive = data.path() / "lineitem_hive";
    run_ok("COPY (" + lineitem + ") TO " + sirius::test::sql_literal(hive.string()) +
           " (FORMAT parquet, PARTITION_BY (l_part));");
    auto const hive_files = sirius::test::sql_literal((hive / "**" / "*.parquet").string());
    run_ok("CREATE VIEW syn.lineitem_hive AS SELECT * FROM read_parquet(" + hive_files +
           ", hive_partitioning = true);");
    run_ok("CREATE VIEW syn.lineitem_filename AS SELECT * FROM read_parquet(" + hive_files +
           ", hive_partitioning = true, filename = true);");
    // Keys 0..449: 400..449 have no lineitem; every 50th key appears twice; some keys are NULL.
    parquet_view("orders",
                 "SELECT CASE WHEN i % 37 = 0 THEN NULL ELSE i % 450 END::BIGINT AS o_orderkey, (i "
                 "% 60)::BIGINT AS o_custkey, DATE '1995-01-01' + (i % 300)::INTEGER AS "
                 "o_orderdate, ((i * 37) % 20000 / 4)::DECIMAL(15, 2) AS o_totalprice FROM "
                 "range(450) t(i) UNION ALL SELECT i::BIGINT, 1, DATE '1996-01-01', 1.00 FROM "
                 "range(0, 450, 50) t(i)");
    parquet_view("customer",
                 "SELECT i::BIGINT AS c_custkey, 'Customer#' || i AS c_name FROM range(60) t(i)");
    // DECIMALs stored in 32 and 64 bits, the latter near the 64-bit limit.
    parquet_view(
      "lineitem_decimal",
      "SELECT (i % 400)::BIGINT AS l_orderkey, ((i * 7) % 99999 / 100)::DECIMAL(9, 2) AS "
      "d9, (900000000000000000 - i * 1000003)::DECIMAL(18, 0) AS d18, "
      "(-900000000000000000 + i * 7)::DECIMAL(18, 0) AS d18_negative FROM range(4000) "
      "t(i)");
    // VARCHAR keys with NULLs, and DOUBLE keys with -0.0, 0.0 and NaN.
    parquet_view("lineitem_varchar",
                 "SELECT CASE WHEN i % 31 = 0 THEN NULL ELSE 'k' || (i % 400) END AS l_orderkey, "
                 "((i * 7) % 50 + 1)::DECIMAL(15, 2) AS l_quantity FROM range(4000) t(i)");
    parquet_view(
      "orders_varchar",
      "SELECT 'k' || i AS o_orderkey, (i % 60)::BIGINT AS o_custkey FROM range(450) t(i)");
    parquet_view("lineitem_double",
                 "SELECT CASE i % 5 WHEN 0 THEN -0.0::DOUBLE WHEN 1 THEN 0.0::DOUBLE WHEN 2 THEN "
                 "'nan'::DOUBLE ELSE (i % 40)::DOUBLE END AS l_orderkey, ((i * 7) % 50 + "
                 "1)::DECIMAL(15, 2) AS l_quantity FROM range(400) t(i)");
    parquet_view("orders_double",
                 "SELECT CASE i WHEN 0 THEN 0.0::DOUBLE WHEN 1 THEN -0.0::DOUBLE WHEN 2 THEN "
                 "'nan'::DOUBLE ELSE i::DOUBLE END AS o_orderkey, (i % 60)::BIGINT AS o_custkey "
                 "FROM range(45) t(i)");
  }

  std::uint64_t rewrites()
  {
    return sirius::test::get_transparent_execution_stats(*con).scan_reuse_rewrites;
  }

  /// The GPU plans with and without the rewrite against the CPU. The rewrite must fire, or not, as
  /// @p expected says when enabled, and never when disabled.
  void compare_reused_and_not(std::string const& query, reuse expected)
  {
    INFO(query);
    auto const before = rewrites();
    compare_gpu_vs_cpu(query);
    auto const after = rewrites();
    if (expected == reuse::fires) { CHECK(after > before); }
    if (expected == reuse::declines) { CHECK(after == before); }
    sirius::test::scoped_sirius_setting off{*con, "enable_scan_reuse_rewrites", false};
    compare_gpu_vs_cpu(query);
    CHECK(rewrites() == after);
  }

  std::optional<sirius::test::scoped_sirius_setting> enable_guard;

 private:
  void parquet_view(std::string const& name, std::string const& select)
  {
    auto const file = data.file_literal(name + ".parquet");
    run_ok("COPY (" + select + ") TO " + file + " (FORMAT parquet);");
    run_ok("CREATE VIEW syn." + name + " AS SELECT * FROM read_parquet(" + file + ");");
  }

  sirius::test::scratch_dir data{"scan_reuse"};
};

/// Q18 grouped by order over the `syn` tables @p orders and @p lineitem, summing @p column.
std::string per_order(std::string const& orders,
                      std::string const& lineitem,
                      std::string const& column,
                      std::string const& having)
{
  return "select o_custkey, sum(" + column + ") from " + orders + ", " + lineitem +
         " where o_orderkey in (select l_orderkey from " + lineitem +
         " group by l_orderkey having sum(" + column + ") " + having +
         ") and o_orderkey = l_orderkey group by o_custkey";
}

}  // namespace

TEST_CASE_METHOD(ScanReuseFixture,
                 "gpu_execution TPC-H Q18 reuses its IN-subquery sums",
                 "[integration][gpu_execution][scan_reuse]")
{
  compare_reused_and_not(sirius::test::kTpchQ18, reuse::fires);
  // A lower threshold keeps more orders.
  auto lower = std::string(sirius::test::kTpchQ18);
  lower.replace(lower.find("> 300"), 5, "> 250");
  compare_reused_and_not(lower, reuse::fires);
}

TEST_CASE_METHOD(ScanReuseFixture,
                 "gpu_execution Q18 shapes the scan reuse rewrites",
                 "[integration][gpu_execution][scan_reuse]")
{
  compare_reused_and_not(q18_query{}.sql(), reuse::fires);
  compare_reused_and_not(q18_query{.outer_aggregates = "sum(l_quantity), sum(l_extendedprice)",
                                   .having           = "having sum(l_quantity) > 120 and "
                                                       "sum(l_extendedprice) > 100"}
                           .sql(),
                         reuse::fires);
  compare_reused_and_not(
    q18_query{.outer_aggregates = "sum(l_quantity), sum(l_quantity) * 2"}.sql(), reuse::fires);
  // So few orders that DuckDB builds the semi join on them (RIGHT_SEMI).
  compare_reused_and_not(
    q18_query{.outer_where =
                " and o_totalprice > 4900 and o_custkey < 3 and o_orderdate > DATE '1995-06-01'"}
      .sql(),
    reuse::fires);
  // HAVING that keeps nothing, that keeps the order whose sum is NULL, and that reads another
  // aggregate.
  compare_reused_and_not(q18_query{.having = "having sum(l_quantity) > 1000000"}.sql(),
                         reuse::fires);
  compare_reused_and_not(
    q18_query{.having = "having sum(l_quantity) is null or count(*) > 5"}.sql(), reuse::fires);
  compare_reused_and_not(
    q18_query{.having = "having sum(l_quantity) > 100 and max(l_quantity) < 50"}.sql(),
    reuse::fires);
  // Ungrouped, with and without rows, and grouped by an expression over the orders key.
  compare_reused_and_not(q18_query{.groups = ""}.sql(), reuse::fires);
  compare_reused_and_not(q18_query{.having = "having sum(l_quantity) > 1e9", .groups = ""}.sql(),
                         reuse::fires);
  compare_reused_and_not(
    "select o_orderkey % 7 as g, sum(l_quantity) from syn.orders, "
    "syn.lineitem where o_orderkey in (select l_orderkey from syn.lineitem "
    "group by l_orderkey having sum(l_quantity) > 120) and o_orderkey = "
    "l_orderkey group by o_orderkey % 7",
    reuse::fires);
  // The same filter on both scans, on another column or on the key.
  compare_reused_and_not(q18_query{.subquery_where = "where l_shipdate > DATE '1994-06-01'",
                                   .outer_where    = " and l_shipdate > DATE '1994-06-01'"}
                           .sql(),
                         reuse::fires);
  compare_reused_and_not(
    q18_query{.subquery_where = "where l_orderkey < 300", .outer_where = " and l_orderkey < 300"}
      .sql(),
    reuse::fires);
  compare_reused_and_not(
    q18_query{.outer_lineitem = "syn.lineitem_hive", .subquery_lineitem = "syn.lineitem_hive"}
      .sql(),
    reuse::fires);
}

TEST_CASE_METHOD(ScanReuseFixture,
                 "gpu_execution scan reuse over DECIMAL sums and VARCHAR and DOUBLE keys",
                 "[integration][gpu_execution][scan_reuse]")
{
  for (char const* column : {"d9", "d18", "d18_negative"}) {
    compare_reused_and_not(per_order("syn.orders", "syn.lineitem_decimal", column, "<> 0"),
                           reuse::fires);
  }
  compare_reused_and_not(
    per_order("syn.orders_varchar", "syn.lineitem_varchar", "l_quantity", "> 100"), reuse::fires);
  compare_reused_and_not(per_order("syn.orders_double", "syn.lineitem_double", "l_quantity", "> 0"),
                         reuse::fires);
}

TEST_CASE_METHOD(ScanReuseFixture,
                 "gpu_execution scan reuse in larger plans",
                 "[integration][gpu_execution][scan_reuse]")
{
  auto const base = q18_query{}.sql();
  compare_reused_and_not(
    base + " union all " + q18_query{.having = "having sum(l_quantity) > 150"}.sql(), reuse::fires);
  compare_reused_and_not(
    "with q as materialized (" + base +
      ") select a.o_orderkey, a.\"sum(l_quantity)\" + b.\"sum(l_quantity)\" from q a join q b "
      "using (o_orderkey)",
    reuse::fires);
  compare_reused_and_not("select sum(s), max(s), count(*) from (" +
                           q18_query{.outer_aggregates = "sum(l_quantity) as s"}.sql() + ")",
                         reuse::fires);
  // Another join and a residual predicate between the two joins.
  compare_reused_and_not(
    "select c.c_name, c2.c_name, sum(l_quantity) from syn.customer c, syn.customer c2, "
    "syn.orders, syn.lineitem where o_orderkey in (select l_orderkey from syn.lineitem group by "
    "l_orderkey having sum(l_quantity) > 120) and c.c_custkey = o_custkey and c2.c_custkey = "
    "(o_custkey + 1) % 60 and o_orderkey = l_orderkey and o_totalprice > c2.c_custkey "
    "group by c.c_name, c2.c_name",
    reuse::fires);
  // The subquery read from a CTE, and a second IN subquery over another file.
  compare_reused_and_not(
    "with s as materialized (select l_orderkey from syn.lineitem group by l_orderkey having "
    "sum(l_quantity) > 120) select o_orderkey, sum(l_quantity) from syn.orders, syn.lineitem "
    "where o_orderkey in (select l_orderkey from s) and o_orderkey = l_orderkey group by "
    "o_orderkey",
    reuse::either);
  compare_reused_and_not(
    q18_query{.outer_where = " and o_orderkey in (select l_orderkey from syn.lineitem2 group by "
                             "l_orderkey having sum(l_quantity) > 50)"}
      .sql(),
    reuse::either);
}

TEST_CASE_METHOD(ScanReuseFixture,
                 "gpu_execution Q18 shapes the scan reuse leaves alone",
                 "[integration][gpu_execution][scan_reuse]")
{
  // No SUM to reuse: every order with a lineitem.
  compare_reused_and_not(q18_query{.having = ""}.sql(), reuse::declines);
  compare_reused_and_not(q18_query{.subquery_groups = "l_orderkey, l_linenumber"}.sql(),
                         reuse::declines);
  compare_reused_and_not(q18_query{.membership = "o_custkey in"}.sql(), reuse::declines);
  // Aggregates the rule does not reuse.
  for (char const* aggregates : {"avg(l_quantity)",
                                 "sum(l_quantity), count(*)",
                                 "sum(l_quantity), sum(l_extendedprice)",
                                 "sum(l_quantity) filter (where o_totalprice > 100)"}) {
    compare_reused_and_not(q18_query{.outer_aggregates = aggregates}.sql(), reuse::declines);
  }
  compare_reused_and_not(
    q18_query{.having = "having sum(l_quantity) filter (where l_linenumber = 1) > 10"}.sql(),
    reuse::declines);
  compare_reused_and_not(
    q18_query{.groups = "c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, l_linenumber"}
      .sql(),
    reuse::declines);
  // A predicate on the outer lineitem the subquery does not have.
  compare_reused_and_not(q18_query{.outer_where = " and l_shipdate > DATE '1995-01-01'"}.sql(),
                         reuse::declines);
  compare_reused_and_not(q18_query{.subquery_where = "where l_shipdate > DATE '1994-06-01'",
                                   .outer_where    = " and l_shipdate > DATE '1994-07-01'"}
                           .sql(),
                         reuse::declines);
  compare_reused_and_not(q18_query{.outer_where = " and l_quantity > o_totalprice / 1000"}.sql(),
                         reuse::declines);
}

TEST_CASE_METHOD(ScanReuseFixture,
                 "gpu_execution scan reuse leaves alone scans of different rows",
                 "[integration][gpu_execution][scan_reuse]")
{
  compare_reused_and_not(q18_query{.subquery_lineitem = "syn.lineitem2"}.sql(), reuse::declines);
  compare_reused_and_not(q18_query{.outer_lineitem = "syn.lineitem_filtered"}.sql(),
                         reuse::declines);
  compare_reused_and_not(q18_query{.subquery_lineitem = "syn.lineitem_filtered"}.sql(),
                         reuse::declines);
  // Partition and file name filters select files, not rows.
  auto const hive =
    q18_query{.outer_lineitem = "syn.lineitem_hive", .subquery_lineitem = "syn.lineitem_hive"};
  auto outer_only        = hive;
  outer_only.outer_where = " and l_part = 1";
  compare_reused_and_not(outer_only.sql(), reuse::declines);
  auto subquery_only           = hive;
  subquery_only.subquery_where = "where l_part = 1";
  compare_reused_and_not(subquery_only.sql(), reuse::declines);
  auto differ           = subquery_only;
  differ.subquery_where = "where l_part = 2";
  differ.outer_where    = " and l_part = 1";
  compare_reused_and_not(differ.sql(), reuse::declines);
  compare_reused_and_not(q18_query{.outer_lineitem    = "syn.lineitem_filename",
                                   .subquery_lineitem = "syn.lineitem_filename",
                                   .outer_where       = " and filename like '%l_part=1%'"}
                           .sql(),
                         reuse::declines);
  // Same rows either way, whether or not the rule proves it.
  compare_reused_and_not(q18_query{.outer_lineitem = "syn.lineitem_binary_as_string"}.sql(),
                         reuse::either);
  compare_reused_and_not(q18_query{.subquery_where = "where l_orderkey < 300"}.sql(),
                         reuse::either);
  compare_reused_and_not(q18_query{.outer_where = " and l_orderkey < 300"}.sql(), reuse::either);
  compare_reused_and_not(q18_query{.outer_where = " and o_orderkey < 200"}.sql(), reuse::either);
  compare_reused_and_not(q18_query{.subquery_where = "where l_quantity is not null"}.sql(),
                         reuse::either);
}
