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

#include "config.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "pipeline/sirius_plan_printer.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <filesystem>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

using sirius::pipeline::pipeline_conversion_result;
using sirius::pipeline::sirius_pipeline;

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

//! Operator chain with the `(id=N)` tokens stripped — operator IDs legitimately differ
//! between the two flag paths.
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

}  // namespace

//! Gate for `reorder_pipelines_topologically`: the tree-based schedule must be strictly
//! topological (the raw meta-sweep emission is not), deterministic, and structurally
//! equivalent to legacy. Order is NOT compared against legacy — its interleaving comes
//! from meta-sweep state absent from the final graph; the differential dumps prove equality.
TEST_CASE("tree-based schedule is strictly topological and deterministic",
          "[integration][pipeline][schedule_canonical]")
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

  for (int q = 1; q <= 22; ++q) {
    DYNAMIC_SECTION("q" << q)
    {
      auto query = sirius::test::read_tpch_query_file(q);

      // Legacy reference property: its native emission is already strictly topological.
      std::multiset<std::string> legacy_chains;
      duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD = false;
      sirius::test::with_conversion_result(con, query, [&](pipeline_conversion_result& result) {
        require_strictly_topological(result.scheduled_pipelines);
        for (const auto& pipeline : result.scheduled_pipelines) {
          legacy_chains.insert(operator_chain(*pipeline));
        }
      });

      duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD = true;
      sirius::test::with_conversion_result(con, query, [&](pipeline_conversion_result& result) {
        auto& scheduled = result.scheduled_pipelines;
        require_strictly_topological(scheduled);

        // pipeline_id equals the vector position; `dependencies` sorted by it (printer order).
        std::multiset<std::string> tree_chains;
        for (size_t i = 0; i < scheduled.size(); i++) {
          REQUIRE(scheduled[i]->get_pipeline_id() == i);
          REQUIRE(std::is_sorted(scheduled[i]->dependencies.begin(),
                                 scheduled[i]->dependencies.end(),
                                 [](const duckdb::shared_ptr<sirius_pipeline>& a,
                                    const duckdb::shared_ptr<sirius_pipeline>& b) {
                                   return a->get_pipeline_id() < b->get_pipeline_id();
                                 }));
          tree_chains.insert(operator_chain(*scheduled[i]));
        }

        // Same pipelines as legacy, order aside.
        REQUIRE(tree_chains == legacy_chains);

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
    }
  }
}
