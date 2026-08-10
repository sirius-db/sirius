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

/**
 * @file test_partition_build_pipelines.cpp
 * @brief PARTITION's custom `build_pipelines` override and the wiring it produces: every
 *        PARTITION is its own single-operator pipeline, its tree child terminates the deeper
 *        meta-pipeline as sink, its input edge uses consumer-owned probe/build barrier semantics,
 *        and its output routes to the downstream CONCAT/MERGE pipeline.
 */

#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_partition.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "pipeline/sirius_plan_printer.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::pipeline_conversion_result;
using sirius::pipeline::repository_wiring;
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

//! All scheduled pipelines whose sink is a PARTITION.
std::vector<duckdb::shared_ptr<sirius_pipeline>> partition_pipelines(
  pipeline_conversion_result& result)
{
  std::vector<duckdb::shared_ptr<sirius_pipeline>> out;
  for (const auto& pipeline : result.scheduled_pipelines) {
    if (pipeline->get_sink() &&
        pipeline->get_sink()->type == SiriusPhysicalOperatorType::PARTITION) {
      out.push_back(pipeline);
    }
  }
  return out;
}

std::vector<const repository_wiring*> wirings_into(pipeline_conversion_result& result,
                                                   const sirius_pipeline* pipeline)
{
  std::vector<const repository_wiring*> out;
  for (const auto& wiring : result.repository_wirings) {
    if (wiring.dest_pipeline.get() == pipeline) { out.push_back(&wiring); }
  }
  return out;
}

std::vector<const repository_wiring*> wirings_out_of(pipeline_conversion_result& result,
                                                     const sirius_pipeline* pipeline)
{
  std::vector<const repository_wiring*> out;
  for (const auto& wiring : result.repository_wirings) {
    if (wiring.source_pipeline.get() == pipeline) { out.push_back(&wiring); }
  }
  return out;
}

//! Render every scheduled pipeline's operator chain for failure diagnostics.
std::string chains_to_string(pipeline_conversion_result& result)
{
  std::string out;
  for (const auto& pipeline : result.scheduled_pipelines) {
    out += sirius::pipeline::sirius_plan_printer::build_operator_chain(*pipeline);
    out += "\n";
  }
  return out;
}

//! Shared PARTITION invariants: single-operator pipeline whose tree child terminated its own
//! deeper pipeline as sink (the promotion PARTITION's build_pipelines performs).
void require_partition_pipeline_shape(pipeline_conversion_result& result,
                                      const duckdb::shared_ptr<sirius_pipeline>& pipeline)
{
  auto ops = pipeline->get_operators();
  REQUIRE(ops.size() == 1);
  auto* partition = pipeline->get_sink().get();
  CHECK(&ops[0].get() == partition);
  CHECK(pipeline->get_source().get() == partition);

  REQUIRE(partition->children.size() == 1);
  auto inputs = wirings_into(result, pipeline.get());
  REQUIRE(inputs.size() == 1);
  REQUIRE(inputs[0]->source_pipeline);
  CHECK(inputs[0]->source_pipeline->get_sink().get() == partition->children[0].get());
}

struct partition_build_fixture {
  partition_build_fixture()
  {
    REQUIRE(sirius::test::g_integration_env != nullptr);
    if (!sirius::test::g_integration_env->is_active()) {
      sirius::test::g_integration_env->resume();
    }
    con = std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());

    auto db_path = integration_db_path();
    REQUIRE(fs::exists(db_path));
    auto r = con->Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    r = con->Query("USE tpch;");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }

  std::unique_ptr<duckdb::Connection> con;
};

}  // namespace

TEST_CASE_METHOD(partition_build_fixture,
                 "PARTITION build_pipelines - join partitions promote their child and route "
                 "to CONCAT",
                 "[integration][pipeline][partition_build]")
{
  // ORDER BY keeps the join out of the root pipeline: root pipelines are not part of
  // `scheduled_pipelines`, and the join-dependency assertions below need the join's
  // pipeline in the schedule.
  const std::string query =
    "SELECT n.n_name, r.r_name FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey "
    "ORDER BY n.n_name";

  sirius::test::with_conversion_result(*con, query, [&](pipeline_conversion_result& result) {
    INFO(chains_to_string(result));
    auto partitions = partition_pipelines(result);
    // One probe-side and one build-side PARTITION from the join-child wraps.
    REQUIRE(partitions.size() == 2);

    for (const auto& pipeline : partitions) {
      auto& partition = pipeline->get_sink()->Cast<sirius::op::sirius_physical_partition>();
      INFO((partition.is_build_partition() ? "build" : "probe") << "-side PARTITION pipeline");

      require_partition_pipeline_shape(result, pipeline);

      // Probe-side partitions stream batches (PARTIAL); the build side must accumulate
      // every partition before the join can run (FULL).
      auto inputs                 = wirings_into(result, pipeline.get());
      const auto expected_barrier = partition.is_build_partition()
                                      ? sirius::op::MemoryBarrierType::FULL
                                      : sirius::op::MemoryBarrierType::PARTIAL;
      CHECK(inputs[0]->barrier_type == expected_barrier);

      // PARTITION output routes to its parent CONCAT: a single-operator [CONCAT] pipeline
      // draining incrementally on the default port.
      auto outputs = wirings_out_of(result, pipeline.get());
      REQUIRE(outputs.size() == 1);
      REQUIRE(outputs[0]->dest_pipeline);
      auto* concat = outputs[0]->dest_pipeline->get_sink().get();
      REQUIRE(concat != nullptr);
      REQUIRE(concat->type == SiriusPhysicalOperatorType::CONCAT);
      CHECK(concat == partition.get_parent_op());
      CHECK(outputs[0]->dest_pipeline->get_operators().size() == 1);
      CHECK(outputs[0]->barrier_type == sirius::op::MemoryBarrierType::PARTIAL);
      CHECK(outputs[0]->port_id == "default");
    }

    // The join pipeline consumes the two CONCATs build-side-first (dynamic filters must
    // publish before the probe subtree is scheduled).
    const sirius_pipeline* join_pipeline = nullptr;
    for (const auto& pipeline : result.scheduled_pipelines) {
      if (pipeline->get_source() &&
          pipeline->get_source()->type == SiriusPhysicalOperatorType::HASH_JOIN) {
        join_pipeline = pipeline.get();
        break;
      }
    }
    REQUIRE(join_pipeline != nullptr);
    REQUIRE(join_pipeline->dependencies.size() >= 2);
    auto* build_sink = join_pipeline->dependencies[0]->get_sink().get();
    REQUIRE(build_sink != nullptr);
    REQUIRE(build_sink->type == SiriusPhysicalOperatorType::CONCAT);
    CHECK(build_sink->Cast<sirius::op::sirius_physical_concat>().is_build_concat());
    bool has_probe_concat_dep = false;
    for (const auto& dep : join_pipeline->dependencies) {
      auto* sink = dep->get_sink().get();
      if (sink != nullptr && sink->type == SiriusPhysicalOperatorType::CONCAT &&
          !sink->Cast<sirius::op::sirius_physical_concat>().is_build_concat()) {
        has_probe_concat_dep = true;
      }
    }
    CHECK(has_probe_concat_dep);

    // Scan companions: each base table's GPU_SCAN terminates its own single-op pipeline.
    size_t gpu_scan_pipelines = 0;
    for (const auto& pipeline : result.scheduled_pipelines) {
      if (pipeline->get_sink() &&
          pipeline->get_sink()->type == SiriusPhysicalOperatorType::GPU_SCAN) {
        gpu_scan_pipelines++;
        CHECK(pipeline->get_operators().size() == 1);
      }
    }
    CHECK(gpu_scan_pipelines == 2);
  });
}

TEST_CASE_METHOD(partition_build_fixture,
                 "PARTITION build_pipelines - non-leaf build subtree still terminates at the "
                 "promoted child",
                 "[integration][pipeline][partition_build]")
{
  // A grouped subquery keeps the join's build subtree non-leaf (plain filters get pushed
  // into the scan as expression table-filters): the build PARTITION's promoted child is
  // the aggregation chain's root, and its producer pipeline carries multiple operators.
  const std::string query =
    "SELECT n.n_name, agg.c FROM nation n JOIN (SELECT r_regionkey, count(*) AS c "
    "FROM region GROUP BY r_regionkey) agg ON n.n_regionkey = agg.r_regionkey "
    "ORDER BY n.n_name";

  sirius::test::with_conversion_result(*con, query, [&](pipeline_conversion_result& result) {
    INFO(chains_to_string(result));
    // Two join-child partitions plus the aggregate-fanout partition of the subquery.
    auto partitions = partition_pipelines(result);
    REQUIRE(partitions.size() == 3);
    for (const auto& pipeline : partitions) {
      require_partition_pipeline_shape(result, pipeline);
    }

    // At least one PARTITION's promoted child heads a multi-operator producer pipeline
    // (the filtered side); the wiring still lands on that child as the producer's sink.
    bool saw_multi_op_producer = false;
    for (const auto& pipeline : partitions) {
      auto inputs = wirings_into(result, pipeline.get());
      REQUIRE(inputs.size() == 1);
      if (inputs[0]->source_pipeline->get_operators().size() > 1) { saw_multi_op_producer = true; }
    }
    CHECK(saw_multi_op_producer);
  });
}

TEST_CASE_METHOD(
  partition_build_fixture,
  "PARTITION build_pipelines - aggregate fanout ingress follows runtime size estimation",
  "[integration][pipeline][partition_build]")
{
  const std::string query = "SELECT n_regionkey, count(*) FROM nation GROUP BY n_regionkey";

  // The ingress barrier of an aggregate-fanout PARTITION is the one thing
  // enable_runtime_size_estimation changes at plan-conversion time, so assert it under both
  // settings rather than only the one the feature ships in. Everything else about the shape —
  // and the FULL egress into the merge — must be identical either way.
  auto check_ingress = [&](bool estimation_on, sirius::op::MemoryBarrierType expected) {
    auto set = con->Query(std::string("SET enable_runtime_size_estimation = ") +
                          (estimation_on ? "true;" : "false;"));
    REQUIRE(set);
    REQUIRE_FALSE(set->HasError());

    sirius::test::with_conversion_result(*con, query, [&](pipeline_conversion_result& result) {
      INFO("enable_runtime_size_estimation = " << estimation_on);
      auto partitions = partition_pipelines(result);
      // The MERGE_GROUP_BY fanout partition from wrap_hash_group_by.
      REQUIRE(partitions.size() == 1);
      const auto& pipeline = partitions[0];

      require_partition_pipeline_shape(result, pipeline);

      auto inputs = wirings_into(result, pipeline.get());
      REQUIRE(inputs.size() == 1);
      CHECK(inputs[0]->barrier_type == expected);
      CHECK(inputs[0]->source_pipeline->get_sink()->type ==
            SiriusPhysicalOperatorType::HASH_GROUP_BY);

      // Downstream, the merge consumes the partition output — a FULL barrier under either
      // setting, because the merge needs every per-thread bucket before it can combine a
      // partition.
      auto outputs = wirings_out_of(result, pipeline.get());
      REQUIRE(outputs.size() == 1);
      CHECK(outputs[0]->barrier_type == sirius::op::MemoryBarrierType::FULL);
      auto* partition = pipeline->get_sink().get();
      CHECK(outputs[0]->dest_pipeline->get_source().get() == partition->get_parent_op());
      CHECK(partition->get_parent_op()->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    });
  };

  // Off (the shipped default): the edge is never relaxed, so this is the pre-feature plan
  // exactly — not an emulation of it maintained by the operator's task hint.
  check_ingress(/*estimation_on=*/false, sirius::op::MemoryBarrierType::FULL);

  // On: the per-thread HASH_GROUP_BY output may stream into the partition, because a projected
  // total lets it fix its count from the first batches instead of the whole input.
  // sirius_physical_partition::get_next_task_hint still withholds tasks until a count is
  // pinned, so this is safe even when the estimate never materializes.
  check_ingress(/*estimation_on=*/true, sirius::op::MemoryBarrierType::PARTIAL);

  // Leave the setting as found; the connection's operator_params outlive this scope.
  con->Query("SET enable_runtime_size_estimation = false;");
}
