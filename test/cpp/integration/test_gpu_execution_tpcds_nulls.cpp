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

// Differential (GPU-vs-CPU) NULL-correctness coverage over a real TPC-DS dataset
// (issue #1095, sub-issue: broad NULL regression net). These are hand-written,
// NULL-focused queries over the TPC-DS *tables* (not the canonical q1-q99), which
// naturally carry NULLs in measures, dimension foreign keys, dates and strings.
// They exercise NULL handling in realistic multi-column / multi-table shapes:
// three-valued predicate logic, NULL-skipping aggregates, NULL group keys,
// LEFT-join NULL-padding, star joins, string (concat vs ||) semantics, and NULL
// propagation through arithmetic / CAST / COALESCE / CASE / date functions.
//
// Every query runs through the shared GpuExecutionFixture: on the GPU with no
// fallback, then on DuckDB CPU, and the results are compared.
//
// The dataset is a pre-generated TPC-DS (sf=0.01) DuckDB file committed under
// data/duckdb/tpcds.duckdb (see generate_tpcds_duckdb.sh), attached read-only -- so
// the suite needs no network or `tpcds` extension at runtime and runs in CI.
// Override the path with SIRIUS_TPCDS_TEST_DB_PATH.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

// Path to the pre-generated TPC-DS (sf=0.01) database committed under data/duckdb
// (see generate_tpcds_duckdb.sh). Attached read-only at test time, so the suite
// needs no network or `tpcds` extension at runtime. Override with
// SIRIUS_TPCDS_TEST_DB_PATH.
std::filesystem::path get_tpcds_db_path()
{
  const char* env = std::getenv("SIRIUS_TPCDS_TEST_DB_PATH");
  auto db_path    = env ? std::filesystem::path(env)
                        : std::filesystem::path(__FILE__).parent_path() / "data/duckdb/tpcds.duckdb";
  REQUIRE(std::filesystem::exists(db_path));
  return db_path;
}

class TpcdsNullFixture : public sirius::test::GpuExecutionFixture {
 public:
  TpcdsNullFixture()
  {
    // Attach the committed TPC-DS DB read-only and switch to it. IF NOT EXISTS so
    // fixtures sharing one DuckDB instance reuse the single attachment; run_ok
    // REQUIREs success, so a missing/incompatible file fails the test loudly.
    run_ok("ATTACH IF NOT EXISTS '" + get_tpcds_db_path().string() + "' AS tpcds (READ_ONLY);");
    run_ok("USE tpcds;");
  }
};

}  // namespace

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds dataset precondition: relevant NULLs exist",
                 "[integration][gpu_execution][tpcds][nulls]")
{
  // Guard against a silently non-null dataset: if a future regeneration stops
  // emitting NULLs in these columns (or unmatched join rows), the differential
  // cases below would still pass while testing nothing about NULLs. Run on CPU so
  // this is a pure data assertion, independent of the GPU path.
  con->Query("SET gpu_execution = false;");
  auto positive = [&](const std::string& sql) {
    auto result = con->Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("precondition query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    auto const n = result->GetValue(0, 0).GetValue<int64_t>();
    UNSCOPED_INFO("expected NULLs from: " << sql << " (got " << n << ")");
    REQUIRE(n > 0);
  };

  positive("SELECT count(*) FROM store_sales WHERE ss_addr_sk IS NULL");
  positive("SELECT count(*) FROM store_sales WHERE ss_store_sk IS NULL");
  positive("SELECT count(*) FROM store_sales WHERE ss_customer_sk IS NULL");
  positive("SELECT count(*) FROM store_sales WHERE ss_net_profit IS NULL");
  positive("SELECT count(*) FROM store_sales WHERE ss_sales_price IS NULL");
  positive("SELECT count(*) FROM customer WHERE c_last_name IS NULL");
  positive("SELECT count(*) FROM customer WHERE c_email_address IS NULL");
  positive("SELECT count(*) FROM web_sales WHERE ws_ship_date_sk IS NULL");
  positive("SELECT count(*) FROM store_returns WHERE sr_customer_sk IS NULL");
  positive("SELECT count(*) FROM store_returns WHERE sr_return_amt IS NULL");
  // The LEFT-join cases rely on unmatched (NULL-padded) rows actually existing.
  positive(
    "SELECT count(*) FROM store_sales ss "
    "LEFT JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk "
    "WHERE c.c_customer_sk IS NULL");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds NULL filters and three-valued logic",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // store_sales foreign keys (ss_addr_sk, ss_cdemo_sk, ss_hdemo_sk, …) are
  // nullable by design.
  compare_gpu_vs_cpu("SELECT count(*) FROM store_sales WHERE ss_addr_sk IS NULL");
  compare_gpu_vs_cpu("SELECT count(*) FROM store_sales WHERE ss_addr_sk IS NOT NULL");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM store_sales WHERE ss_cdemo_sk IS NULL OR ss_hdemo_sk IS NULL");
  // Three-valued OR: a TRUE branch survives a NULL branch (TRUE OR NULL = TRUE).
  compare_gpu_vs_cpu("SELECT count(*) FROM store_sales WHERE ss_addr_sk = 5 OR ss_promo_sk = 1");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM store_sales WHERE ss_quantity IS NULL AND ss_sales_price IS NULL");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM store_sales WHERE ss_addr_sk IS NOT DISTINCT FROM ss_cdemo_sk");
  compare_gpu_vs_cpu("SELECT count(*) FROM store_sales WHERE ss_quantity BETWEEN 1 AND 20");
  compare_gpu_vs_cpu("SELECT count(*) FROM store_sales WHERE ss_store_sk IN (1, 2, 4)");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds aggregates over nullable columns",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // COUNT(*) counts rows; COUNT(col) skips NULLs on nullable foreign keys.
  compare_gpu_vs_cpu(
    "SELECT count(*), count(ss_addr_sk), count(ss_customer_sk), count(ss_promo_sk) "
    "FROM store_sales");
  // Exact comparison uses only order-independent aggregates (integer SUM/AVG,
  // MIN/MAX); decimal SUM/AVG go through the approx case below. NULL-skipping over
  // decimal measures is still covered here via COUNT / MIN / MAX.
  compare_gpu_vs_cpu(
    "SELECT sum(ss_quantity), avg(ss_quantity), min(ss_sales_price), max(ss_sales_price) "
    "FROM store_sales");
  compare_gpu_vs_cpu(
    "SELECT count(ss_net_profit), min(ss_net_profit), max(ss_net_profit) FROM store_sales");
  // GROUP BY a nullable foreign key: NULL forms its own group.
  compare_gpu_vs_cpu(
    "SELECT ss_store_sk, count(*), sum(ss_quantity), min(ss_sales_price), max(ss_sales_price) "
    "FROM store_sales GROUP BY ss_store_sk");
  // Multi-key GROUP BY over two nullable keys.
  compare_gpu_vs_cpu(
    "SELECT ss_store_sk, ss_promo_sk, count(*) FROM store_sales GROUP BY ss_store_sk, ss_promo_sk");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds float aggregates over nullable measures (approx)",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // Decimal SUM/AVG compared with a relative tolerance (reduction-order low-bit
  // differences); approx_cols name the 0-based measure columns, keys/counts stay
  // exact. Non-negative columns only; cancellation-prone signed measures (e.g.
  // ss_net_profit) stay in the exact COUNT/MIN/MAX case above.
  compare_gpu_vs_cpu_approx("SELECT sum(ss_sales_price), avg(ss_sales_price) FROM store_sales",
                            {0, 1});
  compare_gpu_vs_cpu_approx("SELECT sum(ws_sales_price), avg(ws_sales_price) FROM web_sales",
                            {0, 1});
  // Grouped by a unique key (col 0, exact) with an approximate measure (col 1).
  compare_gpu_vs_cpu_approx(
    "SELECT ss_store_sk, avg(ss_sales_price) FROM store_sales GROUP BY ss_store_sk", {1});
  compare_gpu_vs_cpu_approx(
    "SELECT i.i_category, avg(ss.ss_sales_price) "
    "FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk "
    "GROUP BY i.i_category",
    {1});
  compare_gpu_vs_cpu_approx("SELECT sum(sr_return_amt), avg(sr_return_amt) FROM store_returns",
                            {0, 1});
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds nullable columns across web_sales / catalog_sales",
                 "[integration][gpu_execution][tpcds][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT count(*), count(ws_ship_date_sk), count(ws_ship_addr_sk) FROM web_sales");
  compare_gpu_vs_cpu("SELECT count(*) FROM web_sales WHERE ws_ship_date_sk IS NULL");
  compare_gpu_vs_cpu(
    "SELECT sum(ws_quantity), min(ws_sales_price), max(ws_sales_price) FROM web_sales");
  compare_gpu_vs_cpu("SELECT count(*), count(cs_ship_date_sk) FROM catalog_sales");
  compare_gpu_vs_cpu(
    "SELECT cs_warehouse_sk, count(*), sum(cs_quantity) FROM catalog_sales GROUP BY "
    "cs_warehouse_sk");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds LEFT JOIN NULL-pads unmatched rows",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // A NULL / unmatched ss_customer_sk leaves the customer side NULL-padded.
  compare_gpu_vs_cpu(
    "SELECT count(*), count(c.c_customer_sk) "
    "FROM store_sales ss LEFT JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk");
  compare_gpu_vs_cpu(
    "SELECT count(*) "
    "FROM store_sales ss LEFT JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk "
    "WHERE c.c_customer_sk IS NULL");
  compare_gpu_vs_cpu(
    "SELECT c.c_current_addr_sk, count(*) "
    "FROM store_sales ss LEFT JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk "
    "GROUP BY c.c_current_addr_sk");
  // Nullable join key on the probe side (ws_ship_date_sk) -> unmatched -> NULL.
  compare_gpu_vs_cpu(
    "SELECT count(*), count(d.d_date_sk) "
    "FROM web_sales ws LEFT JOIN date_dim d ON ws.ws_ship_date_sk = d.d_date_sk");
  // Contrast with an INNER join on the same nullable key: NULL never equals NULL,
  // so rows with a NULL ss_customer_sk are dropped entirely (fewer than the LEFT
  // join above keeps).
  compare_gpu_vs_cpu(
    "SELECT count(*) "
    "FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds star joins with nullable filters/aggregates",
                 "[integration][gpu_execution][tpcds][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT count(*), sum(ss.ss_quantity) "
    "FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk "
    "WHERE i.i_current_price IS NOT NULL");
  compare_gpu_vs_cpu(
    "SELECT i.i_category, count(*), min(ss.ss_sales_price), max(ss.ss_sales_price) "
    "FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk "
    "GROUP BY i.i_category");
  compare_gpu_vs_cpu(
    "SELECT d.d_year, count(*), sum(ss.ss_quantity) "
    "FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk "
    "GROUP BY d.d_year");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds string NULL semantics (concat vs ||)",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // Customer name/email columns are nullable. concat() ignores NULLs; ||
  // propagates them; length/substring propagate NULL.
  compare_gpu_vs_cpu(
    "SELECT c_customer_sk, concat(c_first_name, ' ', c_last_name) AS n FROM customer");
  compare_gpu_vs_cpu("SELECT c_customer_sk, c_first_name || ' ' || c_last_name AS n FROM customer");
  compare_gpu_vs_cpu("SELECT c_customer_sk, length(c_email_address) AS l FROM customer");
  compare_gpu_vs_cpu("SELECT count(*) FROM customer WHERE c_email_address IS NULL");
  compare_gpu_vs_cpu("SELECT i_item_sk, coalesce(i_size, 'unknown') AS sz FROM item");
  // LIKE / NOT LIKE on a nullable column is three-valued: NULL rows are excluded.
  compare_gpu_vs_cpu("SELECT count(*) FROM customer WHERE c_last_name LIKE 'A%'");
  compare_gpu_vs_cpu("SELECT count(*) FROM customer WHERE c_email_address NOT LIKE '%.com'");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds NULL propagation through expressions",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // Aggregate the expression results rather than materializing + sorting all of
  // store_sales twice: COUNT(expr) vs COUNT(*) verifies NULL propagation, and the
  // integer SUMs verify the produced values deterministically (order-independent).
  // Arithmetic propagates NULL when either operand is NULL.
  compare_gpu_vs_cpu(
    "SELECT count(*), count(ss_ext_sales_price - ss_ext_discount_amt) FROM store_sales");
  // COALESCE replaces NULL keys with -1.
  compare_gpu_vs_cpu("SELECT sum(coalesce(ss_addr_sk, -1)) FROM store_sales");
  // NULLIF yields NULL where ss_promo_sk = 1 (and where it is already NULL).
  compare_gpu_vs_cpu("SELECT count(*), count(nullif(ss_promo_sk, 1)) FROM store_sales");
  // CAST preserves NULL; SUM verifies the cast values.
  compare_gpu_vs_cpu(
    "SELECT count(*), count(CAST(ss_quantity AS BIGINT)), sum(CAST(ss_quantity AS BIGINT)) "
    "FROM store_sales");
  compare_gpu_vs_cpu(
    "SELECT sum(CASE WHEN ss_addr_sk IS NULL THEN 1 ELSE 0 END) AS null_addrs FROM store_sales");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds ORDER BY NULLS FIRST/LAST and top-N",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // Group first so the ordering key (ss_store_sk, including its NULL group) is
  // unique per row -- a deterministic total order -- so NULLS FIRST/LAST placement
  // is actually verified. Ordered comparison keeps emitted order instead of
  // sorting it away.
  compare_gpu_vs_cpu_ordered(
    "SELECT ss_store_sk, count(*) FROM store_sales GROUP BY ss_store_sk "
    "ORDER BY ss_store_sk NULLS FIRST");
  compare_gpu_vs_cpu_ordered(
    "SELECT ss_store_sk, count(*) FROM store_sales GROUP BY ss_store_sk "
    "ORDER BY ss_store_sk NULLS LAST");
  compare_gpu_vs_cpu_ordered(
    "SELECT ss_store_sk, count(*) FROM store_sales GROUP BY ss_store_sk "
    "ORDER BY ss_store_sk DESC NULLS FIRST");
  // Top-N: the NULL group must sort to the correct end before the LIMIT cut.
  compare_gpu_vs_cpu_ordered(
    "SELECT ss_store_sk, count(*) FROM store_sales GROUP BY ss_store_sk "
    "ORDER BY ss_store_sk NULLS LAST LIMIT 5");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds three-valued NOT IN with NULL",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // NOT IN over a set containing NULL is three-valued: every non-matching row is
  // UNKNOWN, so the result is empty.
  compare_gpu_vs_cpu("SELECT count(*) FROM store_sales WHERE ss_store_sk NOT IN (1, 2, NULL)");
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds three-valued NOT IN (subquery with NULL)",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // Like the literal-list case but via anti/MARK-join semantics: NOT IN a subquery
  // whose result contains NULL (ss_addr_sk is nullable) is three-valued, so the
  // predicate is UNKNOWN for every non-matching row and the result is empty.
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM store_sales "
    "WHERE ss_store_sk NOT IN (SELECT ss_addr_sk FROM store_sales)");
}

// KNOWN GPU DIVERGENCE (quarantined; tracked in sirius-db/sirius#1291): a join
// keyed on IS NOT DISTINCT FROM must be null-safe -- NULL matches NULL -- but the
// GPU lowers it to a plain `=` and drops the NULL-to-NULL matches, so it silently
// undercounts (observed GPU=148018 vs CPU=280213 at sf=0.01). It runs on the GPU
// (asserted below: exactly one execution, no fallback) and returns a wrong result.
// We can't use Catch2's [!shouldfail] tag here: it conflicts with the WARN+skip
// path when the tpcds extension is unavailable (a skip registers as an unexpected
// pass). Instead assert the divergence directly -- this stays green while the bug
// exists, skips cleanly, and flips to a failure (prompting un-quarantine) once the
// fix (sirius-db/sirius#1291) lands and the counts agree.
TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds null-safe join (IS NOT DISTINCT FROM) [known divergence]",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // Both ss_addr_sk and sr_addr_sk are nullable, so a null-safe join matches their
  // NULL rows to each other; a plain `=` drops them.
  const std::string query =
    "SELECT count(*) FROM store_sales ss JOIN store_returns sr "
    "ON ss.ss_addr_sk IS NOT DISTINCT FROM sr.sr_addr_sk";

  // Assert the query actually ran on the GPU with no fallback, so this stays a
  // "silent GPU wrong answer" and not an unnoticed CPU fallback.
  con->Query("SET gpu_execution = true;");
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto gpu    = con->Query(query);
  auto after  = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(gpu);
  REQUIRE_FALSE(gpu->HasError());
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);

  con->Query("SET gpu_execution = false;");
  auto cpu = con->Query(query);
  con->Query("SET gpu_execution = true;");
  REQUIRE(cpu);
  REQUIRE_FALSE(cpu->HasError());

  auto const gpu_n = gpu->GetValue(0, 0).GetValue<int64_t>();
  auto const cpu_n = cpu->GetValue(0, 0).GetValue<int64_t>();

  // The bug makes the GPU treat IS NOT DISTINCT FROM as a plain `=` (NULL never
  // matches NULL), so its count equals the `=` join and is strictly below the
  // correct null-safe CPU count. Assert that specific shape -- a bare
  // gpu_n != cpu_n would accept any wrong GPU value.
  con->Query("SET gpu_execution = false;");
  auto eq_result = con->Query(
    "SELECT count(*) FROM store_sales ss JOIN store_returns sr "
    "ON ss.ss_addr_sk = sr.sr_addr_sk");
  con->Query("SET gpu_execution = true;");
  REQUIRE(eq_result);
  REQUIRE_FALSE(eq_result->HasError());
  auto const eq_n = eq_result->GetValue(0, 0).GetValue<int64_t>();

  UNSCOPED_INFO("null-safe join known divergence (sirius#1291): GPU="
                << gpu_n << " CPU=" << cpu_n << " ('=' join=" << eq_n << ")");
  REQUIRE(cpu_n > eq_n);   // NULL=NULL matches genuinely exist at this scale
  REQUIRE(gpu_n == eq_n);  // GPU computes the plain '=' join; flips red when the fix lands
}

TEST_CASE_METHOD(TpcdsNullFixture,
                 "gpu_execution tpcds date functions and returns tables with NULLs",
                 "[integration][gpu_execution][tpcds][nulls]")
{  // year() over a NULL date (unmatched LEFT join) must propagate NULL.
  compare_gpu_vs_cpu(
    "SELECT year(d.d_date) AS y, count(*) "
    "FROM web_sales ws LEFT JOIN date_dim d ON ws.ws_ship_date_sk = d.d_date_sk "
    "GROUP BY year(d.d_date)");
  compare_gpu_vs_cpu("SELECT count(*), count(sr_return_amt) FROM store_returns");
  compare_gpu_vs_cpu(
    "SELECT sr_store_sk, count(sr_return_amt), min(sr_return_amt), max(sr_return_amt) "
    "FROM store_returns GROUP BY sr_store_sk");
  compare_gpu_vs_cpu("SELECT count(*) FROM store_returns WHERE sr_customer_sk IS NULL");
}
