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

#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_set>

namespace fs = std::filesystem;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

fs::path integration_data_dir()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data";
#else
  return fs::path(__FILE__).parent_path().parent_path() / "integration/data";
#endif
}

duckdb::Connection tpch_connection()
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  auto db_path = integration_data_dir() / "duckdb/integration.duckdb";
  REQUIRE(fs::exists(db_path));
  auto r = con.Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  r = con.Query("USE tpch;");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  return con;
}

// Converts `query` and requires exactly one pipeline without an outgoing edge: the root.
// `start_query` seeds only the first source and the active task creator reaches every other
// pipeline by following edges downstream, so a second edgeless pipeline strands the rest.
// Returns the number of pipelines whose sink is a CTE.
std::size_t require_single_terminal(duckdb::Connection& con, const std::string& query)
{
  std::size_t cte_sinks = 0;
  sirius::test::with_conversion_result(
    con, query, [&](sirius::pipeline::pipeline_conversion_result& result) {
      std::unordered_set<const sirius::pipeline::sirius_pipeline*> producers;
      for (auto const& wiring : result.repository_wirings) {
        producers.insert(wiring.source_pipeline.get());
      }
      std::size_t terminals = 0;
      for (auto const& pipeline : result.scheduled_pipelines) {
        if (!producers.contains(pipeline.get())) { ++terminals; }
        auto sink = pipeline->get_sink();
        if (sink && sink->type == SiriusPhysicalOperatorType::CTE) { ++cte_sinks; }
      }
      INFO(sirius::pipeline::dump_pipeline_schedule_raw(result));
      REQUIRE(terminals == 1);
    });
  return cte_sinks;
}

}  // namespace

// Statistics propagation proves `c` empty and replaces its only CTE_SCAN with an empty result,
// after unused-CTE removal has run. The materialization is then unread and must not be planned.
TEST_CASE("materialized CTE with every reference folded away plans no CTE pipeline",
          "[integration][pipeline][cte]")
{
  auto con = tpch_connection();

  SECTION("single CTE")
  {
    REQUIRE(require_single_terminal(
              con,
              "WITH c AS MATERIALIZED (SELECT n_regionkey AS c0, max(n_nationkey) AS c1 "
              "FROM nation WHERE (673 IS NULL) GROUP BY n_regionkey HAVING max(n_nationkey) < 188) "
              "SELECT c1, count(*) FROM c WHERE c1 IS NULL GROUP BY c1") == 0);
  }

  SECTION("outer CTE read only by an unread inner CTE")
  {
    REQUIRE(require_single_terminal(con,
                                    "WITH o AS MATERIALIZED (SELECT n_nationkey AS k FROM nation), "
                                    "c AS MATERIALIZED (SELECT k AS c0, max(k) AS c1 FROM o "
                                    "WHERE (673 IS NULL) GROUP BY k HAVING max(k) < 188) "
                                    "SELECT c1, count(*) FROM c WHERE c1 IS NULL GROUP BY c1") ==
            0);
  }

  SECTION("control: a read CTE keeps its pipeline")
  {
    REQUIRE(require_single_terminal(
              con,
              "WITH c AS MATERIALIZED (SELECT n_regionkey AS c0, max(n_nationkey) AS c1 "
              "FROM nation GROUP BY n_regionkey) "
              "SELECT c1, count(*) FROM c GROUP BY c1") == 1);
  }

  SECTION("many nested CTE definitions stay cheap to inspect")
  {
    constexpr int cte_count = 20;
    std::string query       = "WITH ";
    for (int i = 0; i < cte_count; ++i) {
      if (i > 0) { query += ", "; }
      query += "c" + std::to_string(i) + " AS MATERIALIZED (SELECT n_nationkey AS k FROM nation)";
    }
    query += " SELECT k FROM c0";
    for (int i = 1; i < cte_count; ++i) {
      query += " UNION ALL SELECT k FROM c" + std::to_string(i);
    }
    sirius::test::with_conversion_result(
      con, query, [&](sirius::pipeline::pipeline_conversion_result& result) {
        std::size_t cte_sinks = 0;
        for (auto const& pipeline : result.scheduled_pipelines) {
          auto sink = pipeline->get_sink();
          if (sink && sink->type == SiriusPhysicalOperatorType::CTE) { ++cte_sinks; }
        }
        REQUIRE(cte_sinks == cte_count);
      });
  }
}
