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
 * @file test_gpu_execution_agg_table_tpch.cpp
 * @brief TPC-H on the C++ tests' SF 0.1 database with the table path of GROUP BY off and on.
 *
 * Two groups of checks, each on one GPU and on two (the second variant is tagged [multi_gpu]
 * and returns early on a single-GPU host, like the TPC-H cases of test_gpu_execution_tpch.cpp):
 *
 *  - all 22 queries, as test_gpu_execution_tpch.cpp runs them;
 *  - the single-key aggregations inside Q13, Q15, Q17 and Q18 on their own, which are the
 *    aggregations the table path is built for: an integer key, DECIMAL sums, COUNT and AVG. They
 *    run with the default scan batch (one batch) and with one batch per row group.
 *
 * Every check is the three-way comparison of utils/agg_table_test_utils.hpp: CPU, GPU with the
 * table path off, GPU with it on.
 */

#include <catch.hpp>
#include <utils/agg_table_test_utils.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/tpch_queries.hpp>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;

using sirius::test::scoped_sirius_setting;
using sirius::test::agg_table::compare_on_off;
using sirius::test::agg_table::compare_options;
using sirius::test::agg_table::kScanBatch;

namespace {

/// A connection to the integration environment the Catch2 listener activated for this case's
/// tags ([integration] is the one-GPU environment, [integration][multi_gpu] the two-GPU one),
/// with the TPC-H database attached read-only. Null when @p num_gpus GPUs are not visible.
std::unique_ptr<duckdb::Connection> connect_tpch(int num_gpus)
{
  if (!sirius::test::has_gpus(num_gpus)) { return nullptr; }
  auto* env = sirius::test::acquire_integration_env_for(num_gpus);
  REQUIRE(env != nullptr);
  REQUIRE(env->is_active());
  auto con = std::make_unique<duckdb::Connection>(env->make_connection());

  char const* override_path = std::getenv("SIRIUS_INTEGRATION_TEST_DB_PATH");
  auto const db_path        = override_path != nullptr
                                ? fs::path(override_path)
                                : fs::path(__FILE__).parent_path() / "data/duckdb/integration.duckdb";
  REQUIRE(fs::exists(db_path));
  auto attach = con->Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
  REQUIRE(attach);
  REQUIRE_FALSE(attach->HasError());
  auto use_db = con->Query("USE tpch;");
  REQUIRE(use_db);
  REQUIRE_FALSE(use_db->HasError());
  return con;
}

/// FLOAT and DOUBLE columns (AVG, and ratios of sums) within a relative 1e-9; everything else,
/// including every DECIMAL sum, exact.
compare_options const kTpchOptions{.approx_float_types = true, .rel_tol = 1e-9};

void run_all_queries(int num_gpus)
{
  for (auto const& spec : sirius::test::kTpchQueries) {
    DYNAMIC_SECTION("TPC-H Q" << spec.number)
    {
      auto con = connect_tpch(num_gpus);
      if (!con) { return; }
      std::string const query{spec.sql};
      if (!spec.retry_once) {
        compare_on_off(*con, query, kTpchOptions);
      } else {
        // The policy of test_gpu_execution_tpch.cpp for this query: retry once on an exception.
        try {
          compare_on_off(*con, query, kTpchOptions);
        } catch (std::exception const& first_error) {
          WARN("first attempt failed (known intermittent flake); retrying once: "
               << first_error.what());
          compare_on_off(*con, query, kTpchOptions);
        }
      }
    }
  }
}

// The aggregation inside Q13: one row per customer, COUNT of a nullable column over an outer join.
constexpr char kQ13Inner[] =
  "select c.c_custkey, count(o.o_orderkey) as c_count "
  "from customer c left outer join orders o "
  "  on c.c_custkey = o.o_custkey and o.o_comment not like '%special%requests%' "
  "group by c.c_custkey";

// The view of Q15: a DECIMAL product summed per supplier under a date filter.
constexpr char kQ15Inner[] =
  "select l_suppkey as supplier_no, sum(l_extendedprice * (1 - l_discount)) as total_revenue "
  "from lineitem "
  "where l_shipdate >= date '1993-05-01' and l_shipdate < date '1993-08-01' "
  "group by l_suppkey";

// The correlated subquery of Q17, decorrelated by hand: AVG of a DECIMAL per part.
constexpr char kQ17Inner[] =
  "select l_partkey, 0.2 * avg(l_quantity) as threshold, count(*) as lines "
  "from lineitem group by l_partkey";

// The IN list of Q18: a DECIMAL sum per order under a HAVING. 300 selects almost nothing at
// SF 0.1, so a second form lowers it, and a third reduces all 150000 groups to one row.
constexpr char kQ18Inner[] =
  "select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300";
constexpr char kQ18InnerLow[] =
  "select l_orderkey, sum(l_quantity) as quantity from lineitem "
  "group by l_orderkey having sum(l_quantity) > 250";
constexpr char kQ18InnerAll[] =
  "select count(*) as orders, sum(quantity) as quantity, min(quantity) as least, "
  "max(quantity) as most, sum(lines) as lines "
  "from (select l_orderkey, sum(l_quantity) as quantity, count(*) as lines "
  "      from lineitem group by l_orderkey) t";

// The same keys with every aggregate the table holds, and with the extended price as a second
// DECIMAL input.
constexpr char kOrderStats[] =
  "select count(*) as orders, sum(q) as q, sum(p) as p, sum(mx - mn) as spread, sum(c) as c "
  "from (select l_orderkey, sum(l_quantity) as q, sum(l_extendedprice) as p, "
  "             min(l_extendedprice) as mn, max(l_extendedprice) as mx, count(*) as c "
  "      from lineitem group by l_orderkey) t";

void run_inner_aggregations(int num_gpus, bool batch_per_row_group)
{
  auto con = connect_tpch(num_gpus);
  if (!con) { return; }
  // lineitem at SF 0.1 is 600572 rows in five row groups: one scan batch by default, five with
  // one batch per row group.
  std::unique_ptr<scoped_sirius_setting> scan_batches;
  if (batch_per_row_group) {
    scan_batches = std::make_unique<scoped_sirius_setting>(*con, kScanBatch, std::uint64_t{1});
  }
  compare_on_off(*con, kQ13Inner);
  compare_on_off(*con, kQ15Inner);
  compare_on_off(*con, kQ17Inner, {.approx_cols = {1}, .rel_tol = 1e-9});
  compare_on_off(*con, kQ18Inner);
  compare_on_off(*con, kQ18InnerLow);
  compare_on_off(*con, kQ18InnerAll);
  compare_on_off(*con, kOrderStats);
  // The four queries themselves, on the same connection and batch size.
  compare_on_off(*con, std::string{sirius::test::kTpchQ13}, kTpchOptions);
  compare_on_off(*con, std::string{sirius::test::kTpchQ15}, kTpchOptions);
  compare_on_off(*con, std::string{sirius::test::kTpchQ17}, kTpchOptions);
  compare_on_off(*con, std::string{sirius::test::kTpchQ18}, kTpchOptions);
}

}  // namespace

#define AGG_TABLE_TPCH_TAGS "[integration][gpu_execution][aggregate][agg_table][TPC-H]"

TEST_CASE("agg table - TPC-H, all 22 queries", AGG_TABLE_TPCH_TAGS) { run_all_queries(1); }

TEST_CASE("agg table - TPC-H, all 22 queries on 2 GPUs", AGG_TABLE_TPCH_TAGS "[multi_gpu]")
{
  run_all_queries(2);
}

TEST_CASE("agg table - TPC-H Q13, Q15, Q17 and Q18 inner aggregations", AGG_TABLE_TPCH_TAGS)
{
  SECTION("one scan batch") { run_inner_aggregations(1, false); }
  SECTION("one scan batch per row group") { run_inner_aggregations(1, true); }
}

TEST_CASE("agg table - TPC-H Q13, Q15, Q17 and Q18 inner aggregations on 2 GPUs",
          AGG_TABLE_TPCH_TAGS "[multi_gpu]")
{
  SECTION("one scan batch") { run_inner_aggregations(2, false); }
  SECTION("one scan batch per row group") { run_inner_aggregations(2, true); }
}
