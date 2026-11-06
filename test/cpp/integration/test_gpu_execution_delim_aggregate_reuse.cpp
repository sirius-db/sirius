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

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <optional>
#include <string>

namespace {

// Correlated MIN/MAX/AVG subqueries that Sirius computes from the delim join's outer rows. Parts 5
// and 7 appear twice, so the outer rows repeat their lineitems; some lineitems have a NULL part,
// and some parts have no lineitems.
class DelimAggregateReuseFixture : public sirius::test::GpuExecutionFixture {
 public:
  DelimAggregateReuseFixture()
  {
    enable_guard.emplace(*con, "enable_delim_aggregate_reuse", true);
    run_ok(
      "CREATE TABLE part (p_partkey BIGINT, p_brand VARCHAR, p_size INTEGER, p_mfgr VARCHAR);");
    run_ok(
      "INSERT INTO part SELECT range, 'Brand#' || (range % 3), range % 4, 'M' || (range % 2) "
      "FROM range(1, 81);");
    run_ok("INSERT INTO part VALUES (5, 'Brand#2', 1, 'M1'), (7, 'Brand#1', 3, 'M1');");
    run_ok(
      "CREATE TABLE lineitem (l_partkey BIGINT, l_quantity DECIMAL(15, 2), "
      "l_extendedprice DECIMAL(15, 2));");
    run_ok(
      "INSERT INTO lineitem SELECT CASE WHEN range % 97 = 0 THEN NULL ELSE (range * 7) % 70 + 1 "
      "END, ((range * 13) % 5000) / 100, (range % 1000) * 1.25 FROM range(4000);");
    run_ok(
      "CREATE TABLE partsupp (ps_partkey BIGINT, ps_suppkey BIGINT, "
      "ps_supplycost DECIMAL(15, 2));");
    run_ok(
      "INSERT INTO partsupp SELECT p, (p + s * 10) % 40 + 1, ((p * 31 + s * 17) % 97) / 10 "
      "FROM range(1, 81) t(p), range(4) u(s);");
    run_ok(
      "CREATE TABLE supplier (s_suppkey BIGINT, s_name VARCHAR, s_nationkey INTEGER, "
      "s_acctbal DECIMAL(15, 2));");
    run_ok(
      "INSERT INTO supplier SELECT range, 'S' || range, range % 25, range * 3.25 "
      "FROM range(1, 41);");
    run_ok("CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR, n_regionkey INTEGER);");
    run_ok("INSERT INTO nation SELECT range, 'N' || range, range % 5 FROM range(25);");
    run_ok("CREATE TABLE region (r_regionkey INTEGER, r_name VARCHAR);");
    run_ok(
      "INSERT INTO region VALUES (0, 'AFRICA'), (1, 'AMERICA'), (2, 'ASIA'), (3, 'EUROPE'), "
      "(4, 'MIDDLE EAST');");
    run_ok("CHECKPOINT;");
  }

  /// Rewritten and unrewritten GPU plans against the CPU, under both of DuckDB's
  /// representations of the delim join. @p approx_column, when set, is a DOUBLE compared with a
  /// tolerance.
  void compare_rewritten_and_not(std::string const& query,
                                 std::optional<std::size_t> approx_column = std::nullopt)
  {
    for (bool const as_cte : {false, true}) {
      CAPTURE(as_cte);
      sirius::test::scoped_sirius_setting representation{*con, "delim_join_as_cte", as_cte};
      for (bool const reuse : {true, false}) {
        CAPTURE(reuse);
        sirius::test::scoped_sirius_setting setting{*con, "enable_delim_aggregate_reuse", reuse};
        if (approx_column) {
          compare_gpu_vs_cpu_approx(query, {*approx_column});
        } else {
          compare_gpu_vs_cpu(query);
        }
      }
    }
  }

  std::optional<sirius::test::scoped_sirius_setting> enable_guard;
};

}  // namespace

TEST_CASE_METHOD(DelimAggregateReuseFixture,
                 "gpu_execution correlated AVG in the shape of TPC-H q17",
                 "[integration][gpu_execution][delim_reuse]")
{
  compare_rewritten_and_not(
    "SELECT sum(l_extendedprice) AS total FROM lineitem, part "
    "WHERE p_partkey = l_partkey AND p_brand IN ('Brand#1', 'Brand#2') "
    "AND l_quantity < (SELECT 0.8 * avg(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)");
  compare_rewritten_and_not(
    "SELECT p_partkey, l_quantity, l_extendedprice FROM lineitem, part "
    "WHERE p_partkey = l_partkey AND p_size < 3 "
    "AND l_quantity >= (SELECT avg(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)");
}

TEST_CASE_METHOD(DelimAggregateReuseFixture,
                 "gpu_execution correlated aggregate values per outer row",
                 "[integration][gpu_execution][delim_reuse]")
{
  compare_rewritten_and_not(
    "SELECT p_partkey, l_quantity, (SELECT max(l_quantity) - min(l_quantity) FROM lineitem "
    "WHERE l_partkey = p_partkey) AS spread FROM lineitem, part WHERE p_partkey = l_partkey "
    "AND p_brand = 'Brand#1'");
  // Keyed by the lineitems' own column, NULL for some, with every lineitem repeated per region.
  compare_rewritten_and_not(
    "SELECT l_partkey, r_name, l_quantity, (SELECT max(l2.l_quantity) FROM lineitem l2 "
    "WHERE l2.l_partkey = l1.l_partkey) AS top FROM lineitem l1, region WHERE r_regionkey < 2");
  compare_rewritten_and_not(
    "SELECT l_partkey, r_name, l_quantity, (SELECT avg(l2.l_quantity) FROM lineitem l2 "
    "WHERE l2.l_partkey = l1.l_partkey) AS mean FROM lineitem l1, region WHERE r_regionkey < 2",
    3);
  // The outer table alone, with NULL keys whose subquery is empty.
  compare_rewritten_and_not(
    "SELECT l_partkey, l_quantity, (SELECT max(l_quantity) FROM lineitem l2 "
    "WHERE l2.l_partkey = l1.l_partkey) AS top FROM lineitem l1");
  compare_rewritten_and_not(
    "SELECT l_partkey, l_quantity, (SELECT avg(l_quantity) FROM lineitem l2 "
    "WHERE l2.l_partkey = l1.l_partkey) AS mean FROM lineitem l1",
    2);
}

TEST_CASE_METHOD(DelimAggregateReuseFixture,
                 "gpu_execution correlated MIN in the shape of TPC-H q2",
                 "[integration][gpu_execution][delim_reuse]")
{
  compare_rewritten_and_not(
    "SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr "
    "FROM part, supplier, partsupp, nation, region "
    "WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 1 "
    "AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE' "
    "AND ps_supplycost = (SELECT min(ps_supplycost) FROM partsupp, supplier, nation, region "
    "WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND s_nationkey = n_nationkey "
    "AND n_regionkey = r_regionkey AND r_name = 'EUROPE')");
}
