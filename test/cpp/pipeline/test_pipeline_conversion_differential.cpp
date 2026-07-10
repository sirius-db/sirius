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
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <filesystem>
#include <fstream>
#include <set>
#include <string>

namespace fs = std::filesystem;

namespace {

//! Path to the integration DuckDB with the SF1 TPC-H schema pre-loaded.
fs::path integration_db_path()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/duckdb/integration.duckdb";
#else
  return fs::path(__FILE__).parent_path().parent_path() /
         "integration/data/duckdb/integration.duckdb";
#endif
}

}  // namespace

//! Primary differential gate for the tree-based pipeline build: the legacy converter
//! (flag OFF) and the tree-based converter (flag ON) must produce byte-identical
//! `pipeline_conversion_result` dumps for every TPC-H query at SF1.
TEST_CASE("TPC-H SF1: legacy and tree-based converters produce identical pipeline state",
          "[integration][pipeline][differential]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  auto db_path = integration_db_path();
  REQUIRE(fs::exists(db_path));
  auto r = con.Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  r = con.Query("USE tpch;");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  sirius::test::tree_pipeline_flag_guard flag_guard;

  // Queries excluded to keep the gate green until fixes land; flipping the flag on by
  // default requires this list to be empty.
  static const std::set<int> kKnownFailing = {};

  for (int q = 1; q <= 22; ++q) {
    if (kKnownFailing.count(q) != 0) { continue; }
    DYNAMIC_SECTION("q" << q)
    {
      auto query = sirius::test::read_tpch_query_file(q);

      auto legacy_dump = sirius::test::dump_under_flag(con, query, /*flag=*/false);
      auto tree_dump   = sirius::test::dump_under_flag(con, query, /*flag=*/true);

      if (legacy_dump != tree_dump) {
        // Dump to /tmp for external diffing; Catch2's INFO quoting obscures byte diffs.
        auto path = std::string{"/tmp/diff_q"} + std::to_string(q);
        std::ofstream(path + "_legacy.txt") << legacy_dump;
        std::ofstream(path + "_tree.txt") << tree_dump;
        INFO("Dumps written to " << path << "_legacy.txt and " << path << "_tree.txt");
      }
      REQUIRE(legacy_dump == tree_dump);
    }
  }
}
