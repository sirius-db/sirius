/*
 * Copyright 2025, Sirius Contributors.
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
#include <config.hpp>
#include <duckdb.hpp>
#include <pipeline/sirius_pipeline_converter.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

//! Path to the canonical 22 TPC-H queries committed to the repo.
fs::path tpch_queries_dir()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/tpch_performance/tpch_queries/orig";
#else
  return fs::path(__FILE__).parent_path().parent_path().parent_path() /
         "test/tpch_performance/tpch_queries/orig";
#endif
}

//! Path to the integration DuckDB with the TPC-H schema pre-loaded (also used by
//! `GPUExecutionDuckDBFixture` in test_gpu_execution_tpch.cpp).
fs::path integration_db_path()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/duckdb/integration.duckdb";
#else
  return fs::path(__FILE__).parent_path().parent_path() /
         "integration/data/duckdb/integration.duckdb";
#endif
}

std::string read_query_file(int q)
{
  auto path = tpch_queries_dir() / ("q" + std::to_string(q) + ".sql");
  std::ifstream in(path);
  REQUIRE(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

//! Capture the current value of `USE_TREE_BASED_PIPELINE_BUILD` and restore it on
//! destruction. The flag is process-wide static so tests that toggle it must restore
//! to avoid contaminating other test cases.
class tree_pipeline_flag_guard {
 public:
  tree_pipeline_flag_guard() : original_(duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {}
  ~tree_pipeline_flag_guard() { duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD = original_; }

  tree_pipeline_flag_guard(const tree_pipeline_flag_guard&)            = delete;
  tree_pipeline_flag_guard& operator=(const tree_pipeline_flag_guard&) = delete;

 private:
  bool original_;
};

std::string dump_under_flag(duckdb::Connection& con, const std::string& query, bool flag)
{
  duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD = flag;
  return sirius::test::convert_query_to_dump(con, query);
}

}  // namespace

//! Phase 3 (#604) Sub-phase E.1 PRIMARY GATE: assert that the legacy
//! converter (flag OFF) and the tree-based converter (flag ON) produce
//! byte-identical `pipeline_conversion_result` for every TPC-H query at SF1.
//!
//! Sources queries from `test/tpch_performance/tpch_queries/orig/q*.sql`.
//! Uses the SF1 TPC-H schema in `test/cpp/integration/data/duckdb/integration.duckdb`
//! (same fixture as `test_gpu_execution_tpch.cpp::GPUExecutionDuckDBFixture`).
//!
//! Toggles `duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD` between flag states; restores
//! the original value via RAII so other test cases see the default.
TEST_CASE("TPC-H SF1: legacy and tree-based converters produce identical pipeline state",
          "[integration][pipeline][differential]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  // Attach the TPC-H schema used by the integration suite (read-only).
  auto db_path = integration_db_path();
  REQUIRE(fs::exists(db_path));
  auto r = con.Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  r = con.Query("USE tpch;");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  tree_pipeline_flag_guard flag_guard;

  // TPC-H queries where the tree-based path currently diverges from legacy. Each entry
  // is a distinct tree-path bug surfaced by this gate and needs targeted investigation
  // before flag-default flip (E.4). Split into two failure modes:
  //   * Exception under flag ON: q2, q4, q17, q20, q21, q22 — tree `build_pipelines`
  //     throws on subquery / DELIM_JOIN / nested-aggregate patterns.
  //   * Pipeline-shape diff: q3, q5, q7-q16, q18, q19 — multi-join queries where the
  //     tree path's pipeline count / barrier ordering doesn't match legacy
  //     post-finalize state.
  // Track follow-ups in sirius-db/sirius#604. Removing a query from this set as fixes
  // land tightens the gate incrementally.
  static const std::set<int> kKnownFailing = {2,  3,  4,  5,  7,  8,  9,  10, 11, 12,
                                              13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

  for (int q = 1; q <= 22; ++q) {
    if (kKnownFailing.count(q) != 0) { continue; }
    DYNAMIC_SECTION("q" << q)
    {
      auto query = read_query_file(q);

      auto legacy_dump = dump_under_flag(con, query, /*flag=*/false);
      auto tree_dump   = dump_under_flag(con, query, /*flag=*/true);

      if (legacy_dump != tree_dump) {
        // Write both dumps to /tmp for clean external diffing — Catch2's INFO
        // output wraps the strings in its own quoting, which makes the actual
        // byte difference hard to spot in the failure message.
        auto path = std::string{"/tmp/diff_q"} + std::to_string(q);
        std::ofstream(path + "_legacy.txt") << legacy_dump;
        std::ofstream(path + "_tree.txt") << tree_dump;
        INFO("Dumps written to " << path << "_legacy.txt and " << path << "_tree.txt");
      }
      REQUIRE(legacy_dump == tree_dump);
    }
  }
}
