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

#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_streaming_sink.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius_engine.hpp"

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <filesystem>
#include <memory>
#include <vector>

namespace fs = std::filesystem;

using sirius::op::sirius_physical_streaming_sink;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::sirius_pipeline;

namespace {

fs::path integration_db_path()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/duckdb/integration.duckdb";
#else
  return fs::path(__FILE__).parent_path().parent_path() /
         "integration/data/duckdb/integration.duckdb";
#endif
}

struct streaming_sink_root_fixture {
  streaming_sink_root_fixture()
  {
    REQUIRE(sirius::test::g_integration_env != nullptr);
    if (!sirius::test::g_integration_env->is_active()) {
      sirius::test::g_integration_env->resume();
    }
    con = std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());

    auto db_path = integration_db_path();
    REQUIRE(fs::exists(db_path));
    auto result =
      con->Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    result = con->Query("USE tpch;");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }

  std::unique_ptr<duckdb::Connection> con;
};

std::vector<std::shared_ptr<cucascade::shared_data_repository>> make_repos(std::size_t n)
{
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> repos;
  repos.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    repos.push_back(std::make_shared<cucascade::shared_data_repository>());
  }
  return repos;
}

//! The pipeline whose `operators` contain `op`, or nullptr.
const sirius_pipeline* pipeline_with_operator(
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines,
  const sirius::op::sirius_physical_operator& target)
{
  for (const auto& pipeline : pipelines) {
    for (const auto& op : pipeline->get_operators()) {
      if (&op.get() == &target) { return pipeline.get(); }
    }
  }
  return nullptr;
}

}  // namespace

// ============================================================================
// SINKROOT-1: a fragment plan rooted in a STREAMING_SINK initializes
// ============================================================================

TEST_CASE_METHOD(streaming_sink_root_fixture,
                 "SINKROOT-1: a streaming-sink-rooted plan initializes",
                 "[integration][pipeline][streaming_sink_root]")
{
  auto repos = make_repos(1);

  sirius::test::with_initialized_streaming_fragment(
    *con,
    "SELECT n_regionkey FROM nation",
    repos,
    [&](sirius::sirius_engine& engine, sirius_physical_streaming_sink& sink) {
      // The sink is the plan root and its subtree hangs off children[], so the plan generator
      // needs none of the out-of-tree descent the RESULT_COLLECTOR requires.
      REQUIRE(sink.type == SiriusPhysicalOperatorType::STREAMING_SINK);
      REQUIRE(sink.children.size() == 1);
      REQUIRE(sink.get_parent_op() == nullptr);

      // A streaming fragment never produces a QueryResult.
      REQUIRE_FALSE(engine.has_result_collector());

      REQUIRE(sink.num_output_streams() == 1);
    });
}

// ============================================================================
// SINKROOT-2: the sink lands in a pipeline's operators, so EOS can fire
// ============================================================================

TEST_CASE_METHOD(streaming_sink_root_fixture,
                 "SINKROOT-2: the streaming sink reaches a pipeline's operators",
                 "[integration][pipeline][streaming_sink_root]")
{
  auto repos = make_repos(1);

  sirius::test::with_initialized_streaming_fragment(
    *con,
    "SELECT n_regionkey FROM nation",
    repos,
    [&](sirius::sirius_engine& engine, sirius_physical_streaming_sink& sink) {
      // This is the load-bearing invariant. on_finalize_operator() -- the sink's only route to
      // end-of-stream -- is driven by update_pipeline_status() iterating get_operators(), which
      // returns the `operators` vector and EXCLUDES the source/sink members. If the sink is not
      // in that vector, EOS never fires and every consumer blocked in wait() hangs forever, with
      // no error anywhere.
      const auto* owning = pipeline_with_operator(engine.sirius_pipelines, sink);
      REQUIRE(owning != nullptr);

      // And it is the pipeline's terminal sink, which is what tells the executor the query is
      // complete once that pipeline finishes.
      REQUIRE(owning->get_sink().get() == &sink);
      REQUIRE(owning->is_query_terminal());
    });
}
