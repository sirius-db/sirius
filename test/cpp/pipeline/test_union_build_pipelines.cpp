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

#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <filesystem>
#include <string>
#include <vector>

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

}  // namespace

// UNION drains its arms in child order and the executor runs pipelines in schedule order, so the
// two must agree: arm i's pipeline is scheduled i-th among the arms, and arm 0's is pipeline #0 --
// the scan `start_query` seeds and the one that runs first. The converter schedules child metas
// last-created-first, which is why `build_pipelines` creates them in reverse; this pins the
// result, not the loop.
TEST_CASE("physical_union - arm pipelines are scheduled in arm order, arm 0 first",
          "[integration][pipeline][union_all]")
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

  // Three arms of different widths so a reversed order cannot pass by symmetry. Wrapped in an
  // aggregate: this harness adds no RESULT_COLLECTOR, and a root UNION gets no pipeline of its own.
  const std::string query =
    "SELECT count(*) FROM (SELECT n_nationkey AS k FROM nation "
    "UNION ALL SELECT r_regionkey FROM region "
    "UNION ALL SELECT s_suppkey FROM supplier) t";

  sirius::test::with_conversion_result(
    con, query, [&](sirius::pipeline::pipeline_conversion_result& result) {
      auto& scheduled = result.scheduled_pipelines;

      sirius::op::sirius_physical_operator* union_op = nullptr;
      for (auto& pipeline : scheduled) {
        auto src = pipeline->get_source();
        if (src && src->type == SiriusPhysicalOperatorType::UNION) {
          union_op = src.get();
          break;
        }
      }
      REQUIRE(union_op != nullptr);
      REQUIRE(union_op->children.size() == 3);

      // Schedule position of each arm: the pipeline whose sink is that arm's child.
      std::vector<size_t> position(union_op->children.size(), scheduled.size());
      for (size_t i = 0; i < scheduled.size(); i++) {
        auto sink = scheduled[i]->get_sink();
        if (!sink) { continue; }
        for (size_t arm = 0; arm < union_op->children.size(); arm++) {
          if (sink.get() == union_op->children[arm].get()) { position[arm] = i; }
        }
      }
      for (size_t arm = 0; arm < position.size(); arm++) {
        INFO("arm " << arm << " scheduled at #" << position[arm]);
        REQUIRE(position[arm] < scheduled.size());
      }

      // Arm order is schedule order, and arm 0 is pipeline #0. `query::build_indices` collects
      // scan operators in this order, so this is also what makes arm 0's scan `scans.front()`.
      REQUIRE(position[0] == 0);
      REQUIRE(position[0] < position[1]);
      REQUIRE(position[1] < position[2]);
    });
}
