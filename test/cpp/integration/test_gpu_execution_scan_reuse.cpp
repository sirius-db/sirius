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

// Answers of the plans planner/scan_reuse_rewrites.hpp rewrites, on the GPU against DuckDB on the
// CPU (which runs DuckDB's own plan), on data with what the rewrites must survive: duplicate keys
// on the side they do not read twice, NULL keys and NULL values.

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/gpu_execution_fixture.hpp>

#include <atomic>
#include <filesystem>
#include <string>
#include <system_error>

namespace {

class ScanReuseFixture : public sirius::test::GpuExecutionFixture {
 public:
  /// Parquet files read through views, as TPC-H runs: over native tables DuckDB's COMMON_SUBPLAN
  /// optimizer already shares Q2's and Q11's joins, and the rewrites would not be exercised.
  std::filesystem::path dir;

  void parquet_table(std::string const& name, std::string const& select)
  {
    auto const file = (dir / (name + ".parquet")).string();
    run_ok("COPY (" + select + ") TO '" + file + "' (FORMAT parquet);");
    run_ok("CREATE VIEW " + name + " AS FROM read_parquet('" + file + "');");
  }

  ScanReuseFixture()
  {
    static std::atomic<unsigned> counter{0};
    dir = std::filesystem::temp_directory_path() /
          ("sirius_gpu_scan_reuse_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
    std::filesystem::create_directories(dir);
    // lineitem: 600 rows over orders 0..99, a NULL orderkey every 97 rows, a NULL quantity every
    // 13 rows. orders: keys 0..119 (some without lineitems), keys 0..9 twice.
    parquet_table("lineitem",
                  "SELECT CASE WHEN i % 97 = 0 THEN NULL ELSE (i % 100)::BIGINT END AS l_orderkey, "
                  "(i % 37)::BIGINT AS l_partkey, (i % 11)::BIGINT AS l_suppkey, (i % 7)::INTEGER "
                  "AS l_linenumber, CASE WHEN i % 13 = 0 THEN NULL ELSE ((i * 7) % 50 + "
                  "1)::DECIMAL(15, 2) END AS l_quantity, (i * 1.5)::DECIMAL(15, 2) AS "
                  "l_extendedprice FROM range(600) t(i)");
    parquet_table("orders",
                  "SELECT (i % 120)::BIGINT AS o_orderkey, (i % 30)::BIGINT AS o_custkey, DATE "
                  "'1995-01-01' + (i % 300)::INTEGER AS o_orderdate, (i * 3.25)::DECIMAL(15, 2) "
                  "AS o_totalprice FROM range(130) t(i)");
    parquet_table("customer",
                  "SELECT i::BIGINT AS c_custkey, 'Customer#' || i AS c_name FROM range(30) t(i)");
    // part: keys 0..36 with 0..5 twice (two part rows per key repeat their lineitems in a join);
    // partsupp: several suppliers per part, a NULL cost every 17 rows.
    parquet_table("part",
                  "SELECT (i % 37)::BIGINT AS p_partkey, 'Brand#' || (i % 2) AS p_brand, CASE "
                  "WHEN i % 3 = 0 THEN 'JUMBO CAN' ELSE 'SM BOX' END AS p_container, (i % "
                  "4)::INTEGER AS p_size, CASE WHEN i % 2 = 0 THEN 'STANDARD NICKEL' ELSE 'SMALL "
                  "BRASS' END AS p_type, 'Manufacturer#' || (i % 3) AS p_mfgr FROM range(43) t(i)");
    parquet_table("supplier",
                  "SELECT i::BIGINT AS s_suppkey, 'Supplier#' || i AS s_name, (i % 5)::BIGINT AS "
                  "s_nationkey, (i * 10.5)::DECIMAL(15, 2) AS s_acctbal, 'addr' || i AS "
                  "s_address, 'phone' || i AS s_phone, 'comment' || i AS s_comment FROM "
                  "range(11) t(i)");
    parquet_table("partsupp",
                  "SELECT (i % 37)::BIGINT AS ps_partkey, (i % 11)::BIGINT AS ps_suppkey, CASE "
                  "WHEN i % 17 = 0 THEN NULL ELSE ((i * 13) % 100 + 1)::DECIMAL(15, 2) END AS "
                  "ps_supplycost, (i % 9)::INTEGER AS ps_availqty FROM range(400) t(i)");
    parquet_table("nation",
                  "SELECT i::BIGINT AS n_nationkey, CASE WHEN i = 1 THEN 'GERMANY' ELSE 'N' || i "
                  "END AS n_name, (i % 2)::BIGINT AS n_regionkey FROM range(5) t(i)");
    parquet_table("region",
                  "SELECT i::BIGINT AS r_regionkey, CASE WHEN i = 0 THEN 'EUROPE' ELSE 'ASIA' END "
                  "AS r_name FROM range(2) t(i)");
  }

  ~ScanReuseFixture()
  {
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
  }
};

}  // namespace

TEST_CASE_METHOD(ScanReuseFixture,
                 "gpu_execution Q18 shape: the IN-subquery aggregate reused for the outer SUM",
                 "[integration][gpu_execution][scan_reuse]")
{
  // TPC-H Q18 without its LIMIT, so ties cannot pick different rows.
  compare_gpu_vs_cpu(
    "select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) from "
    "customer, orders, lineitem where o_orderkey in (select l_orderkey from lineitem group by "
    "l_orderkey having sum(l_quantity) > 120) and c_custkey = o_custkey and o_orderkey = "
    "l_orderkey group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice");
  // Two SUMs of the same column, and a HAVING that keeps every key with a non-NULL sum.
  compare_gpu_vs_cpu(
    "select o_orderkey, o_totalprice, sum(l_quantity), sum(l_quantity) from orders, lineitem "
    "where o_orderkey in (select l_orderkey from lineitem group by l_orderkey having "
    "sum(l_quantity) > 0) and o_orderkey = l_orderkey group by o_orderkey, o_totalprice");
  // Not rewritten (the outer lineitem is filtered): the same answer either way.
  compare_gpu_vs_cpu(
    "select o_orderkey, sum(l_quantity) from orders, lineitem where o_orderkey in (select "
    "l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 120) and o_orderkey = "
    "l_orderkey and l_linenumber > 2 group by o_orderkey");
}
