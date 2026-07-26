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
 * @file test_small_query_bypass_converter.cpp
 * @brief Plan-time structure tests for the small-query bypass (issue #990).
 *
 * Runs real SQL through the DuckDB planner and the Sirius physical plan
 * generator (which applies the bypass when `small_query_bytes_threshold` is
 * set), then converts the plan to pipelines and asserts on structure:
 *   - bypass ON:  no PARTITION / SORT_SAMPLE / SORT_PARTITION anywhere;
 *                 the terminal CONCAT / MERGE_* operators are still present;
 *                 the ORDER_BY -> MERGE_SORT wiring uses a FULL barrier;
 *                 the build-side CONCAT still wires a "build" port.
 *   - bypass OFF: an explicit zero threshold keeps the partition stages.
 *
 * The converter is exercised directly (no GPU execution), mirroring
 * sirius_engine::initialize_internal after plan generation.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <pipeline/pipeline_build_context.hpp>
#include <pipeline/repository_wiring.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <pipeline/sirius_pipeline_converter.hpp>
#include <planner/sirius_physical_plan_generator.hpp>
#include <sirius_config.hpp>
#include <sirius_interface.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::pipeline_conversion_result;
using sirius::pipeline::sirius_pipeline;

namespace {

/// Fixture: file-backed DuckDB with the Sirius extension loaded (shared
/// integration env when available), plus a converter driver.
class SmallQueryBypassFixture : public sirius::test::GpuExecutionFixture {
 public:
  void create_test_tables()
  {
    run_ok("CREATE TABLE dim (id INTEGER, name VARCHAR);");
    run_ok("INSERT INTO dim VALUES (1, 'a'), (2, 'b'), (3, 'c');");
    run_ok("CREATE TABLE fact (dim_id INTEGER, qty INTEGER);");
    run_ok("INSERT INTO fact VALUES (1, 10), (1, 20), (2, 30), (4, 40);");
    run_ok("CHECKPOINT;");
  }

  /// Plan `query` with the real DuckDB planner + Sirius plan generator and run
  /// the pipeline converter. Bypass is controlled via `small_query_bytes_threshold`
  /// before planning (wraps happen inside create_plan).
  /// @param threshold_bytes If set, uses that threshold; otherwise true → 256 MiB,
  ///        false → 0 (disabled).
  pipeline_conversion_result convert_query(const std::string& query,
                                           bool small_query_bypass,
                                           std::optional<uint64_t> threshold_bytes = std::nullopt)
  {
    auto& context = *con->context;

    const uint64_t threshold =
      threshold_bytes.value_or(small_query_bypass ? uint64_t{268435456} : uint64_t{0});
    run_ok("SET small_query_bytes_threshold = " + std::to_string(threshold) + ";");

    context.config.enable_optimizer      = true;
    context.config.use_replacement_scans = false;

    run_ok("BEGIN TRANSACTION;");

    duckdb::Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(parser.statements.size() == 1);

    duckdb::Planner planner(context);
    auto statement_type = parser.statements[0]->type;
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    auto prepared       = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(statement_type);
    prepared->names     = planner.names;
    prepared->types     = planner.types;
    prepared->value_map = std::move(planner.value_map);

    duckdb::Optimizer optimizer(*planner.binder, context);
    auto logical_plan = optimizer.Optimize(std::move(planner.plan));
    logical_plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    duckdb::ColumnBindingResolver::Verify(*logical_plan);
    resolver.VisitOperator(*logical_plan);

    sirius::planner::sirius_physical_plan_generator physical_planner(context);
    auto sirius_plan = physical_planner.create_plan(std::move(logical_plan));
    REQUIRE(sirius_plan);

    // Wrap in a result collector, like the transparent execution path does.
    // Keep the prepared-statement data alive for the collector's lifetime.
    prepared_data_ = duckdb::make_shared_ptr<sirius::sirius_prepared_statement_data>(
      prepared, std::move(sirius_plan));
    auto collector = duckdb::make_uniq_base<sirius::op::sirius_physical_result_collector,
                                            sirius::op::sirius_physical_materialized_collector>(
      *prepared_data_, context);

    // RESULT_COLLECTOR wrap lands after create_plan's set_parent_ops — re-walk.
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(*collector,
                                                                    /*parent=*/nullptr);

    run_ok("COMMIT TRANSACTION;");

    // Mirror sirius_engine::initialize_internal: meta-pipeline build + convert.
    sirius::pipeline::pipeline_build_context build_ctx{nullptr, true, 1};

    sirius::pipeline::sirius_pipeline_build_state state;
    auto root_pipeline =
      duckdb::make_shared_ptr<sirius::pipeline::sirius_meta_pipeline>(build_ctx, state, nullptr);
    root_pipeline->build(*collector);
    root_pipeline->ready();

    sirius::operator_params op_params{};
    sirius::pipeline::sirius_pipeline_converter converter(build_ctx, op_params);
    auto result = converter.convert(*root_pipeline);

    // The collector (and the plan inside prepared_data_) must outlive the
    // conversion result — pipelines hold raw pointers into the plan.
    collectors_.push_back(std::move(collector));
    return result;
  }

  duckdb::shared_ptr<sirius::sirius_prepared_statement_data> prepared_data_;
  std::vector<duckdb::unique_ptr<sirius::op::sirius_physical_result_collector>> collectors_;
};

/// Count operators of `type` across pipeline sources, operators, and sinks.
size_t count_ops(const pipeline_conversion_result& result, SiriusPhysicalOperatorType type)
{
  size_t count = 0;
  for (const auto& pipeline : result.scheduled_pipelines) {
    // After finalize_pipeline_structure, operators[] spans source..sink.
    for (const auto& op : pipeline->get_operators()) {
      if (op.get().type == type) { count++; }
    }
  }
  return count;
}

bool has_partition_stage(const pipeline_conversion_result& result)
{
  return count_ops(result, SiriusPhysicalOperatorType::PARTITION) > 0 ||
         count_ops(result, SiriusPhysicalOperatorType::SORT_SAMPLE) > 0 ||
         count_ops(result, SiriusPhysicalOperatorType::SORT_PARTITION) > 0;
}

/// GPU count the bypass gate sees (same source the multi-GPU join guard reads). Joins decline the
/// bypass when this is > 1, so join-shape assertions must branch on it.
int active_num_gpus(duckdb::Connection& con)
{
  return static_cast<int>(
    sirius::test::get_registered_sirius_context(con)->get_hw_topology().gpus.size());
}

}  // namespace

TEST_CASE("small-query bypass defaults to 256 MiB", "[integration][small_query_bypass][converter]")
{
  CHECK(sirius::operator_params{}.small_query_bytes_threshold == uint64_t{256} * 1024 * 1024);
}

TEST_CASE_METHOD(SmallQueryBypassFixture,
                 "small-query bypass - GROUP_BY skips PARTITION, keeps MERGE_GROUP_BY",
                 "[integration][small_query_bypass][converter]")
{
  create_test_tables();
  const std::string query = "SELECT dim_id, AVG(qty) FROM fact GROUP BY dim_id;";

  auto bypass = convert_query(query, /*small_query_bypass=*/true);
  CHECK_FALSE(has_partition_stage(bypass));
  CHECK(count_ops(bypass, SiriusPhysicalOperatorType::MERGE_GROUP_BY) > 0);

  // The GROUP_BY pipeline must feed the MERGE_GROUP_BY pipeline directly.
  bool group_by_feeds_merge = false;
  for (const auto& wiring : bypass.repository_wirings) {
    if (wiring.source_pipeline->get_sink()->type == SiriusPhysicalOperatorType::HASH_GROUP_BY &&
        wiring.dest_pipeline->get_sink()->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY) {
      group_by_feeds_merge = true;
    }
  }
  CHECK(group_by_feeds_merge);

  auto normal = convert_query(query, /*small_query_bypass=*/false);
  CHECK(count_ops(normal, SiriusPhysicalOperatorType::PARTITION) > 0);
  CHECK(count_ops(normal, SiriusPhysicalOperatorType::MERGE_GROUP_BY) > 0);
}

TEST_CASE_METHOD(SmallQueryBypassFixture,
                 "small-query bypass - HASH_JOIN skips PARTITION, keeps build CONCAT",
                 "[integration][small_query_bypass][converter]")
{
  create_test_tables();
  const std::string query = "SELECT f.qty FROM fact f JOIN dim d ON f.dim_id = d.id;";

  auto bypass = convert_query(query, /*small_query_bypass=*/true);
  if (active_num_gpus(*con) > 1) {
    // Multi-GPU join guard (issue #990): join-bearing plans decline the bypass and keep the
    // partitioned path even under the small-query threshold.
    CHECK(count_ops(bypass, SiriusPhysicalOperatorType::PARTITION) > 0);
  } else {
    CHECK_FALSE(has_partition_stage(bypass));
    CHECK(count_ops(bypass, SiriusPhysicalOperatorType::CONCAT) > 0);

    // A "build" wiring must target the pipeline that contains the HASH_JOIN.
    bool build_port_targets_join = false;
    for (const auto& wiring : bypass.repository_wirings) {
      if (wiring.port_id != "build") { continue; }
      const auto& dest = wiring.dest_pipeline;
      for (const auto& op : dest->get_operators()) {
        if (op.get().type == SiriusPhysicalOperatorType::HASH_JOIN) {
          build_port_targets_join = true;
        }
      }
    }
    CHECK(build_port_targets_join);
  }

  auto normal = convert_query(query, /*small_query_bypass=*/false);
  CHECK(count_ops(normal, SiriusPhysicalOperatorType::PARTITION) > 0);
}

TEST_CASE_METHOD(SmallQueryBypassFixture,
                 "small-query bypass - ORDER_BY skips SORT_SAMPLE/SORT_PARTITION, FULL barrier "
                 "into MERGE_SORT",
                 "[integration][small_query_bypass][converter]")
{
  create_test_tables();
  // Projection drops the sort key, so MERGE_SORT must carry the final projection.
  const std::string query = "SELECT qty FROM fact ORDER BY dim_id;";

  auto bypass = convert_query(query, /*small_query_bypass=*/true);
  CHECK_FALSE(has_partition_stage(bypass));
  CHECK(count_ops(bypass, SiriusPhysicalOperatorType::MERGE_SORT) > 0);

  // ORDER_BY -> MERGE_SORT wiring must be FULL (merge waits for all local sorts).
  bool found_order_to_merge = false;
  for (const auto& wiring : bypass.repository_wirings) {
    if (wiring.source_pipeline->get_sink()->type == SiriusPhysicalOperatorType::ORDER_BY &&
        wiring.dest_pipeline->get_sink()->type == SiriusPhysicalOperatorType::MERGE_SORT) {
      found_order_to_merge = true;
      CHECK(wiring.barrier_type == sirius::op::MemoryBarrierType::FULL);
    }
  }
  CHECK(found_order_to_merge);

  auto normal = convert_query(query, /*small_query_bypass=*/false);
  CHECK(count_ops(normal, SiriusPhysicalOperatorType::SORT_SAMPLE) > 0);
  CHECK(count_ops(normal, SiriusPhysicalOperatorType::SORT_PARTITION) > 0);
}

TEST_CASE_METHOD(SmallQueryBypassFixture,
                 "small-query bypass - multi-join query has no partitions in bypass mode",
                 "[integration][small_query_bypass][converter]")
{
  create_test_tables();
  run_ok("CREATE TABLE dim2 (id INTEGER, tag VARCHAR);");
  run_ok("INSERT INTO dim2 VALUES (1, 'x'), (2, 'y');");
  run_ok("CHECKPOINT;");

  // Two joins so both build and probe sides exercise wrap_join_child.
  const std::string query =
    "SELECT f.qty, d.name, d2.tag FROM fact f "
    "JOIN dim d ON f.dim_id = d.id JOIN dim2 d2 ON f.dim_id = d2.id;";

  auto bypass = convert_query(query, /*small_query_bypass=*/true);
  if (active_num_gpus(*con) > 1) {
    // Multi-GPU join guard (issue #990): join-bearing plans keep the partitioned path.
    CHECK(count_ops(bypass, SiriusPhysicalOperatorType::PARTITION) >= 2);
  } else {
    CHECK_FALSE(has_partition_stage(bypass));
    // Both joins keep their CONCATs (one build-side each, plus probe-side concats).
    CHECK(count_ops(bypass, SiriusPhysicalOperatorType::CONCAT) >= 2);
  }

  auto normal = convert_query(query, /*small_query_bypass=*/false);
  CHECK(count_ops(normal, SiriusPhysicalOperatorType::PARTITION) >= 2);
}

TEST_CASE_METHOD(SmallQueryBypassFixture,
                 "small-query bypass - threshold below scan estimate keeps partitions",
                 "[integration][small_query_bypass][converter]")
{
  create_test_tables();
  // fact is 4 rows × 2 INT cols ≈ 32 bytes; a 1-byte threshold must not activate bypass.
  const std::string query = "SELECT dim_id, AVG(qty) FROM fact GROUP BY dim_id;";
  auto result = convert_query(query, /*small_query_bypass=*/true, /*threshold_bytes=*/uint64_t{1});
  CHECK(count_ops(result, SiriusPhysicalOperatorType::PARTITION) > 0);
  CHECK(count_ops(result, SiriusPhysicalOperatorType::MERGE_GROUP_BY) > 0);
}

TEST_CASE_METHOD(SmallQueryBypassFixture,
                 "small-query bypass - materialized CTE keeps partitions",
                 "[integration][small_query_bypass][converter]")
{
  create_test_tables();
  // Materialized CTEs make the whole query ineligible even with a high threshold.
  const std::string query =
    "WITH c AS MATERIALIZED (SELECT * FROM fact) "
    "SELECT dim_id, count(*) FROM c GROUP BY dim_id ORDER BY dim_id;";

  auto result = convert_query(query, /*small_query_bypass=*/true);
  CHECK(has_partition_stage(result));
}

// Regression for issue #990: a VALUES list is planned as a COLUMN_DATA_SCAN whose materialized
// bytes must be counted in the gate. If it were scored as 0 (the old `default:`-case bug), the
// estimate would fall below any threshold and wrongly bypass — dropping the PARTITION that the
// retained _concat_all build-side CONCAT relies on for a potentially large source.
TEST_CASE_METHOD(SmallQueryBypassFixture,
                 "small-query bypass - COLUMN_DATA_SCAN (VALUES) bytes are counted, not zero",
                 "[integration][small_query_bypass][converter]")
{
  // The only base scan is the VALUES-backed COLUMN_DATA_SCAN, so the gate estimate is entirely
  // that scan's size. GROUP BY inserts a PARTITION in non-bypass mode.
  std::string values = "(0)";
  for (int i = 1; i < 100; ++i) {
    values += ",(" + std::to_string(i) + ")";
  }
  const std::string query =
    "SELECT v.id, COUNT(*) FROM (VALUES " + values + ") AS v(id) GROUP BY v.id;";

  // threshold = 1 byte: below any non-zero estimate, so bypass must NOT activate. Pre-fix the
  // COLUMN_DATA_SCAN scored 0 (0 < 1) and wrongly bypassed; counted, it is >0 (>= 1) and keeps
  // the PARTITION. This cleanly separates "0" from "counted" with no size calibration.
  auto counted = convert_query(query, /*small_query_bypass=*/true, /*threshold_bytes=*/uint64_t{1});
  CHECK(count_ops(counted, SiriusPhysicalOperatorType::PARTITION) > 0);
  CHECK(count_ops(counted, SiriusPhysicalOperatorType::MERGE_GROUP_BY) > 0);

  // At the default 256 MiB threshold the small VALUES stays comfortably eligible: measuring the
  // collection must not make ordinary VALUES queries ineligible for the bypass.
  auto eligible = convert_query(query, /*small_query_bypass=*/true);
  CHECK_FALSE(has_partition_stage(eligible));
  CHECK(count_ops(eligible, SiriusPhysicalOperatorType::MERGE_GROUP_BY) > 0);
}
