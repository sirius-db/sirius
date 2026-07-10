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

#include <array>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <utility>

namespace fs = std::filesystem;

namespace {

fs::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT);
#else
  return fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
#endif
}

//! Create the eight TPC-H tables as views over `test/cpp/integration/data/parquet/*.parquet`.
//! No `USE` — the views live in the default schema.
void create_tpch_parquet_views(duckdb::Connection& con)
{
  auto parquet_dir = (project_root() / "test/cpp/integration/data/parquet").string();
  static constexpr std::array<const char*, 8> kTables = {
    "nation", "region", "customer", "orders", "part", "partsupp", "supplier", "lineitem"};
  for (auto* t : kTables) {
    auto sql = std::string{"CREATE VIEW IF NOT EXISTS "} + t + " AS SELECT * FROM read_parquet('" +
               parquet_dir + "/" + t + ".parquet');";
    auto r = con.Query(sql);
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
}

std::string hive_partition_root()
{
  return (project_root() / "test/cpp/integration/data/hive_partitioned/**/*.parquet").string();
}

}  // namespace

//! Differential gate for the tree-based pipeline build over parquet reads: the same 22-query
//! sweep as the DuckDB-attached gate, but with the TPC-H schema as views over
//! `read_parquet(...)`. The attached-DuckDB gate never exercises the parquet scan path.
TEST_CASE("TPC-H SF1 parquet: legacy and tree-based converters produce identical pipeline state",
          "[integration][pipeline][differential][parquet]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  create_tpch_parquet_views(con);

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
        auto path = std::string{"/tmp/diff_parquet_q"} + std::to_string(q);
        std::ofstream(path + "_legacy.txt") << legacy_dump;
        std::ofstream(path + "_tree.txt") << tree_dump;
        INFO("Dumps written to " << path << "_legacy.txt and " << path << "_tree.txt");
      }
      REQUIRE(legacy_dump == tree_dump);
    }
  }
}

//! Differential gate for hive-partitioned parquet reads — `hive_partitioning=true` exercises
//! op_params plumbing that the plain parquet gate does not reach.
TEST_CASE("hive_partition: legacy and tree-based converters produce identical pipeline state",
          "[integration][pipeline][differential][hive_partition]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  sirius::test::tree_pipeline_flag_guard flag_guard;

  auto const hive                                                  = hive_partition_root();
  std::array<std::pair<const char*, std::string>, 6> const queries = {{
    {"basic_scan_with_partition_columns",
     "SELECT * FROM read_parquet('" + hive + "', hive_partitioning=true) ORDER BY id"},
    {"filter_on_data_column",
     "SELECT * FROM read_parquet('" + hive +
       "', hive_partitioning=true) WHERE id >= 2 ORDER BY id"},
    {"filter_on_partition_column",
     "SELECT id, name, year FROM read_parquet('" + hive +
       "', hive_partitioning=true) WHERE year = 2024 ORDER BY id"},
    {"group_by_partition_column",
     "SELECT year, SUM(amount) as total FROM read_parquet('" + hive +
       "', hive_partitioning=true) GROUP BY year ORDER BY year"},
    {"reversed_column_order",
     "SELECT year, month, amount, name, id FROM read_parquet('" + hive +
       "', hive_partitioning=true) ORDER BY id"},
    {"aggregation_on_data_column",
     "SELECT SUM(amount) as total FROM read_parquet('" + hive + "', hive_partitioning=true)"},
  }};

  // Variants excluded by name to keep the gate green; must be empty before the flag flips on.
  static const std::set<std::string> kKnownFailing = {};

  for (auto const& [name, sql] : queries) {
    if (kKnownFailing.count(name) != 0) { continue; }
    DYNAMIC_SECTION(name)
    {
      auto legacy_dump = sirius::test::dump_under_flag(con, sql, /*flag=*/false);
      auto tree_dump   = sirius::test::dump_under_flag(con, sql, /*flag=*/true);

      if (legacy_dump != tree_dump) {
        auto path = std::string{"/tmp/diff_hive_"} + name;
        std::ofstream(path + "_legacy.txt") << legacy_dump;
        std::ofstream(path + "_tree.txt") << tree_dump;
        INFO("Dumps written to " << path << "_legacy.txt and " << path << "_tree.txt");
      }
      REQUIRE(legacy_dump == tree_dump);
    }
  }
}
