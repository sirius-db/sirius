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
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path get_tpch_db_path()
{
  const char* env = std::getenv("SIRIUS_INTEGRATION_TEST_DB_PATH");
  auto db_path =
    env ? fs::path(env) : fs::path(__FILE__).parent_path() / "data/duckdb/integration.duckdb";
  REQUIRE(fs::exists(db_path));
  return db_path;
}

//! RAII toggle for the opt-in zone-map kind (default off). The SET mutates the shared
//! SiriusContext, so restoring the default is mandatory for later tests.
struct zone_map_switch_guard {
  explicit zone_map_switch_guard(duckdb::Connection& c) : con(c)
  {
    con.Query("SET enable_dynamic_zone_map_filter = true;");
  }
  ~zone_map_switch_guard() { con.Query("SET enable_dynamic_zone_map_filter = false;"); }

  zone_map_switch_guard(const zone_map_switch_guard&)            = delete;
  zone_map_switch_guard& operator=(const zone_map_switch_guard&) = delete;

  duckdb::Connection& con;
};

std::vector<std::vector<std::string>> collect_rows(duckdb::MaterializedQueryResult& result)
{
  std::vector<std::vector<std::string>> rows;
  for (duckdb::idx_t r = 0; r < result.RowCount(); r++) {
    std::vector<std::string> row;
    row.reserve(result.ColumnCount());
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); c++) {
      row.push_back(result.GetValue(c, r).ToString());
    }
    rows.push_back(std::move(row));
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

//! Transparent GPU run (asserted to actually execute on GPU) vs CPU run, exact row-set
//! equality. All queries below aggregate to integer/decimal values, so no float tolerance.
void compare_gpu_vs_cpu(duckdb::Connection& con, const std::string& query)
{
  con.Query("SET gpu_execution = true;");
  auto before_gpu_stats = sirius::test::get_transparent_execution_stats(con);

  auto gpu_result = con.Query(query);
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) {
    UNSCOPED_INFO("transparent GPU execution error: " << gpu_result->GetError());
  }
  REQUIRE_FALSE(gpu_result->HasError());
  auto after_gpu_stats = sirius::test::get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(before_gpu_stats, after_gpu_stats, 1, 0, 1);

  con.Query("SET gpu_execution = false;");
  auto cpu_result = con.Query(query);
  con.Query("SET gpu_execution = true;");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());

  REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
  REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

  auto gpu_rows = collect_rows(gpu_result->Cast<duckdb::MaterializedQueryResult>());
  auto cpu_rows = collect_rows(cpu_result->Cast<duckdb::MaterializedQueryResult>());
  REQUIRE(gpu_rows == cpu_rows);
}

}  // namespace

//! End-to-end coverage for dynamic filters over duckdb-native (seq_scan) tables: a selective
//! hash-join build publishes membership (and, opted in, zone-map) filters into the probe-side
//! native scan's post-decode DYNAMIC_FILTER operator. The join stays authoritative, so every
//! result must be bit-identical to the CPU run whether or not a filter applied.
TEST_CASE("gpu_execution - dynamic filters over duckdb-native tables",
          "[integration][gpu_execution][dynamic_filter]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  auto db_path = get_tpch_db_path();
  auto r       = con.Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  r = con.Query("USE tpch;");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  SECTION("selective membership filter on the probe scan")
  {
    // ~1/1000-selective part build: the publisher emits an IN-list over p_partkey and the
    // lineitem native scan's post-decode operator drops non-matching rows before the probe.
    compare_gpu_vs_cpu(con,
                       "SELECT count(*), min(l_orderkey), max(l_orderkey) "
                       "FROM lineitem JOIN part ON l_partkey = p_partkey "
                       "WHERE p_size = 15 AND p_container = 'SM BOX'");
  }

  SECTION("membership filter composed with a probe-side static filter")
  {
    // l_quantity is a pure-filter probe column (decoded, filtered, then dropped from the
    // output): the dynamic filter on l_partkey must key by output position regardless.
    compare_gpu_vs_cpu(con,
                       "SELECT count(*) "
                       "FROM lineitem JOIN part ON l_partkey = p_partkey "
                       "WHERE p_size = 15 AND p_container = 'SM BOX' AND l_quantity < 10");
  }

  SECTION("opt-in zone map rides the post-decode AST row-mask path")
  {
    // The native scan has no reader-side set_filter, so an opted-in zone map is applied
    // row-wise post-decode (include_ast_row_masks). The build's key range is runtime-derived
    // (integer division defeats static transitive pushdown) and narrow, so publication passes
    // the domain-coverage gate.
    zone_map_switch_guard zone_maps_on(con);
    compare_gpu_vs_cpu(con,
                       "SELECT count(*), sum(l.l_orderkey) "
                       "FROM lineitem l "
                       "JOIN (SELECT o_orderkey FROM orders WHERE o_orderkey / 100 = 50) o "
                       "ON l.l_orderkey = o.o_orderkey");
  }
}
