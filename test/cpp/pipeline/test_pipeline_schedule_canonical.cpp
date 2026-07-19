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

#include "op/sirius_physical_concat.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "pipeline/sirius_plan_printer.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using sirius::pipeline::pipeline_conversion_result;
using sirius::pipeline::sirius_pipeline;

namespace {

//! Root of the committed integration fixtures (duckdb / parquet / hive-partitioned).
fs::path integration_data_dir()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data";
#else
  return fs::path(__FILE__).parent_path().parent_path() / "integration/data";
#endif
}

//! Operator chain with the `(id=N)` tokens stripped, for readable failure messages.
std::string operator_chain(const sirius_pipeline& pipeline)
{
  static const std::regex kIdToken{" \\(id=\\d+\\)"};
  return std::regex_replace(
    sirius::pipeline::sirius_plan_printer::build_operator_chain(pipeline), kIdToken, "");
}

//! Every pipeline must appear after all of its `dependencies` (producers).
void require_strictly_topological(
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& scheduled)
{
  std::unordered_map<const sirius_pipeline*, size_t> position;
  for (size_t i = 0; i < scheduled.size(); i++) {
    position[scheduled[i].get()] = i;
  }
  for (size_t i = 0; i < scheduled.size(); i++) {
    for (const auto& producer : scheduled[i]->dependencies) {
      INFO("pipeline #" << i << " [" << operator_chain(*scheduled[i]) << "] scheduled before its "
                        << "producer #" << position.at(producer.get()) << " ["
                        << operator_chain(*producer) << "]");
      REQUIRE(position.at(producer.get()) < i);
    }
  }
}

//! The per-query battery: the schedule must be strictly topological (raw meta-sweep
//! emission is not), pipeline ids must equal vector positions with `dependencies` sorted
//! by them, join dependencies must stay build-side-first (dynamic filters publish before
//! the probe-side scans they prune), reordering an already-canonical schedule must be a
//! no-op, and converting twice must reproduce the raw schedule byte-identically — the
//! canonical dump is signature-sorted and would miss a schedule-order flip.
void require_canonical_schedule(duckdb::Connection& con, const std::string& query)
{
  sirius::test::with_conversion_result(con, query, [&](pipeline_conversion_result& result) {
    auto& scheduled = result.scheduled_pipelines;
    require_strictly_topological(scheduled);

    // pipeline_id equals the vector position; `dependencies` sorted by it (printer order).
    for (size_t i = 0; i < scheduled.size(); i++) {
      REQUIRE(scheduled[i]->get_pipeline_id() == i);
      REQUIRE(std::is_sorted(scheduled[i]->dependencies.begin(),
                             scheduled[i]->dependencies.end(),
                             [](const duckdb::shared_ptr<sirius_pipeline>& a,
                                const duckdb::shared_ptr<sirius_pipeline>& b) {
                               return a->get_pipeline_id() < b->get_pipeline_id();
                             }));
    }

    // Join dependencies are build-side-first: dependencies[0] is the build CONCAT
    // (finalize_pipeline_structure front-inserts it; link_join_partition_siblings and
    // dynamic-filter scheduling read the slots positionally).
    for (const auto& pipeline : scheduled) {
      auto source = pipeline->get_source();
      if (!source || (source->type != sirius::op::SiriusPhysicalOperatorType::HASH_JOIN &&
                      source->type != sirius::op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN)) {
        continue;
      }
      INFO("join pipeline [" << operator_chain(*pipeline) << "]");
      REQUIRE(!pipeline->dependencies.empty());
      auto build_sink = pipeline->dependencies[0]->get_sink();
      REQUIRE(build_sink);
      REQUIRE(build_sink->type == sirius::op::SiriusPhysicalOperatorType::CONCAT);
      CHECK(build_sink->Cast<sirius::op::sirius_physical_concat>().is_build_concat());
    }

    // Reordering an already-canonical schedule is a no-op.
    std::vector<const sirius_pipeline*> before;
    before.reserve(scheduled.size());
    for (const auto& pipeline : scheduled) {
      before.push_back(pipeline.get());
    }
    sirius::pipeline::reorder_pipelines_topologically(scheduled);
    REQUIRE(scheduled.size() == before.size());
    for (size_t i = 0; i < scheduled.size(); i++) {
      REQUIRE(scheduled[i].get() == before[i]);
    }
  });

  // Convert-twice raw-schedule determinism (see the battery doc above).
  auto first_raw  = sirius::test::convert_query_to_raw_schedule(con, query);
  auto second_raw = sirius::test::convert_query_to_raw_schedule(con, query);
  REQUIRE(first_raw == second_raw);
}

//! The 8 TPC-H tables as views over the committed parquet fixtures.
void create_tpch_parquet_views(duckdb::Connection& con)
{
  auto parquet_dir = (integration_data_dir() / "parquet").string();
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

}  // namespace

//! Gate for `reorder_pipelines_topologically` and conversion determinism over the attached
//! DuckDB-native TPC-H schema; `require_canonical_schedule` documents the invariants.
TEST_CASE("pipeline schedule is strictly topological and deterministic",
          "[integration][pipeline][schedule_canonical]")
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

  for (int q = 1; q <= 22; ++q) {
    DYNAMIC_SECTION("q" << q)
    {
      require_canonical_schedule(con, sirius::test::read_tpch_query_file(q));
    }
  }
}

//! The same gate through the parquet ingestible.
TEST_CASE("parquet pipeline schedule is strictly topological and deterministic",
          "[integration][pipeline][schedule_canonical][parquet]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  create_tpch_parquet_views(con);

  for (int q = 1; q <= 22; ++q) {
    DYNAMIC_SECTION("q" << q)
    {
      require_canonical_schedule(con, sirius::test::read_tpch_query_file(q));
    }
  }
}

TEST_CASE("hive-partitioned parquet pipeline schedule is strictly topological and deterministic",
          "[integration][pipeline][schedule_canonical][hive_partition]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  // `reversed_column_order` is safe here: its known binder failure only reproduces
  // through DuckDB's default ExtractPlan order (used by the hidden end-to-end hive
  // tests), not the sirius-order extraction `with_conversion_result` uses.
  auto const hive = (integration_data_dir() / "hive_partitioned").string() + "/**/*.parquet";
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

  for (auto const& [name, sql] : queries) {
    DYNAMIC_SECTION(name) { require_canonical_schedule(con, sql); }
  }
}
