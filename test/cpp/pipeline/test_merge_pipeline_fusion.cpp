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
 * @file test_merge_pipeline_fusion.cpp
 * @brief Pipeline-shape tests for merge fusion.
 */

#include "config.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius_engine.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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

class fusion_flag_guard {
 public:
  explicit fusion_flag_guard(bool enabled) : original_(duckdb::Config::FUSE_MERGE_PIPELINES)
  {
    duckdb::Config::FUSE_MERGE_PIPELINES = enabled;
  }

  ~fusion_flag_guard() { duckdb::Config::FUSE_MERGE_PIPELINES = original_; }

  fusion_flag_guard(const fusion_flag_guard&)            = delete;
  fusion_flag_guard& operator=(const fusion_flag_guard&) = delete;

 private:
  bool original_;
};

const sirius_pipeline* pipeline_containing(
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines,
  SiriusPhysicalOperatorType type)
{
  for (const auto& pipeline : pipelines) {
    for (const auto& op : pipeline->get_operators()) {
      if (op.get().type == type) { return pipeline.get(); }
    }
  }
  return nullptr;
}

struct merge_fusion_fixture {
  merge_fusion_fixture()
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

void require_fused_terminal_shape(sirius::sirius_engine& engine,
                                  SiriusPhysicalOperatorType merge_type)
{
  const auto* merge_pipeline = pipeline_containing(engine.new_scheduled, merge_type);
  const auto* collector_pipeline =
    pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::RESULT_COLLECTOR);

  REQUIRE(merge_pipeline != nullptr);
  REQUIRE(collector_pipeline != nullptr);
  CHECK(merge_pipeline == collector_pipeline);
  REQUIRE(merge_pipeline->get_source());
  REQUIRE(merge_pipeline->get_sink());
  CHECK(merge_pipeline->get_source()->type == merge_type);
  CHECK(merge_pipeline->get_sink()->type == SiriusPhysicalOperatorType::RESULT_COLLECTOR);
}

void require_unfused_terminal_shape(sirius::sirius_engine& engine,
                                    SiriusPhysicalOperatorType merge_type)
{
  const auto* merge_pipeline = pipeline_containing(engine.new_scheduled, merge_type);
  const auto* collector_pipeline =
    pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::RESULT_COLLECTOR);

  REQUIRE(merge_pipeline != nullptr);
  REQUIRE(collector_pipeline != nullptr);
  CHECK(merge_pipeline != collector_pipeline);
  REQUIRE(merge_pipeline->get_source());
  REQUIRE(merge_pipeline->get_sink());
  CHECK(merge_pipeline->get_source()->type == merge_type);
  CHECK(merge_pipeline->get_sink()->type == merge_type);
}

//! True when some scheduled pipeline contains both operator types.
bool pipeline_contains_both(const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines,
                            SiriusPhysicalOperatorType a,
                            SiriusPhysicalOperatorType b)
{
  for (const auto& pipeline : pipelines) {
    bool has_a = false;
    bool has_b = false;
    for (const auto& op : pipeline->get_operators()) {
      if (op.get().type == a) { has_a = true; }
      if (op.get().type == b) { has_b = true; }
    }
    if (has_a && has_b) { return true; }
  }
  return false;
}

//! Dump scheduled pipeline shapes for diagnostic failures.
std::string dump_pipeline_shapes(
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines)
{
  std::string out;
  for (std::size_t i = 0; i < pipelines.size(); i++) {
    out += "P" + std::to_string(i) + ":[";
    const auto ops = pipelines[i]->get_operators();
    for (std::size_t j = 0; j < ops.size(); j++) {
      if (j > 0) { out += " -> "; }
      out += SiriusPhysicalOperatorToString(ops[j].get().type);
    }
    out += "]\n";
  }
  return out;
}

//! Collect all rows as sorted vectors of stringified values for order-independent comparison.
std::vector<std::vector<std::string>> collect_rows(duckdb::MaterializedQueryResult& result)
{
  std::vector<std::vector<std::string>> rows;
  for (duckdb::idx_t r = 0; r < result.RowCount(); r++) {
    std::vector<std::string> row;
    row.reserve(result.ColumnCount());
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); c++) {
      row.push_back(result.GetValue(c, r).ToString());
    }
    rows.push_back(std::move(row));
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

//! Run `query` on the GPU with fusion enabled and disabled, asserting both ran transparently on
//! the GPU and returned identical results. Toggling via `SET fuse_merge_pipelines` also exercises
//! the extension-option wiring end to end.
void require_fusion_results_match(duckdb::Connection& con, const std::string& query)
{
  con.Query("SET gpu_execution = true;");

  con.Query("SET fuse_merge_pipelines = true;");
  auto before_on = sirius::test::get_transparent_execution_stats(con);
  auto fused     = con.Query(query);
  REQUIRE(fused);
  if (fused->HasError()) { UNSCOPED_INFO("fused execution error: " << fused->GetError()); }
  REQUIRE_FALSE(fused->HasError());
  auto after_on = sirius::test::get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(before_on, after_on, 1, 0, 1);

  con.Query("SET fuse_merge_pipelines = false;");
  auto before_off = sirius::test::get_transparent_execution_stats(con);
  auto unfused    = con.Query(query);
  REQUIRE(unfused);
  if (unfused->HasError()) { UNSCOPED_INFO("unfused execution error: " << unfused->GetError()); }
  REQUIRE_FALSE(unfused->HasError());
  auto after_off = sirius::test::get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(before_off, after_off, 1, 0, 1);

  con.Query("SET fuse_merge_pipelines = true;");  // restore default

  REQUIRE(fused->ColumnCount() == unfused->ColumnCount());
  REQUIRE(fused->RowCount() == unfused->RowCount());

  auto& fused_mat   = fused->Cast<duckdb::MaterializedQueryResult>();
  auto& unfused_mat = unfused->Cast<duckdb::MaterializedQueryResult>();
  auto fused_rows   = collect_rows(fused_mat);
  auto unfused_rows = collect_rows(unfused_mat);

  for (std::size_t r = 0; r < fused_rows.size(); r++) {
    for (std::size_t c = 0; c < fused_rows[r].size(); c++) {
      if (fused_rows[r][c] != unfused_rows[r][c]) {
        UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: fused=[" << fused_rows[r][c]
                             << "] unfused=[" << unfused_rows[r][c] << "]");
      }
      REQUIRE(fused_rows[r][c] == unfused_rows[r][c]);
    }
  }
}

}  // namespace

TEST_CASE_METHOD(merge_fusion_fixture,
                 "merge pipeline fusion folds terminal GROUP BY and TOP_N into the collector",
                 "[integration][pipeline][merge_fusion]")
{
  fusion_flag_guard guard(/*enabled=*/true);

  SECTION("GROUP BY")
  {
    sirius::test::with_initialized_engine(
      *con,
      "SELECT n_regionkey, count(*) FROM nation GROUP BY n_regionkey",
      [&](sirius::sirius_engine& engine) {
        require_fused_terminal_shape(engine, SiriusPhysicalOperatorType::MERGE_GROUP_BY);

        const auto* partition =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::PARTITION);
        const auto* merge =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::MERGE_GROUP_BY);
        REQUIRE(partition != nullptr);
        CHECK(partition != merge);
      });
  }

  SECTION("TOP_N")
  {
    sirius::test::with_initialized_engine(
      *con,
      "SELECT n_nationkey FROM nation ORDER BY n_nationkey DESC LIMIT 3",
      [&](sirius::sirius_engine& engine) {
        require_fused_terminal_shape(engine, SiriusPhysicalOperatorType::MERGE_TOP_N);

        const auto* local_top_n =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::TOP_N);
        const auto* merge =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::MERGE_TOP_N);
        REQUIRE(local_top_n != nullptr);
        CHECK(local_top_n != merge);
      });
  }
}

TEST_CASE_METHOD(merge_fusion_fixture,
                 "merge pipeline fusion flag restores the merge boundary",
                 "[integration][pipeline][merge_fusion]")
{
  fusion_flag_guard guard(/*enabled=*/false);

  SECTION("GROUP BY")
  {
    sirius::test::with_initialized_engine(
      *con,
      "SELECT n_regionkey, count(*) FROM nation GROUP BY n_regionkey",
      [&](sirius::sirius_engine& engine) {
        require_unfused_terminal_shape(engine, SiriusPhysicalOperatorType::MERGE_GROUP_BY);
      });
  }

  SECTION("TOP_N")
  {
    sirius::test::with_initialized_engine(
      *con,
      "SELECT n_nationkey FROM nation ORDER BY n_nationkey DESC LIMIT 3",
      [&](sirius::sirius_engine& engine) {
        require_unfused_terminal_shape(engine, SiriusPhysicalOperatorType::MERGE_TOP_N);
      });
  }
}

TEST_CASE_METHOD(merge_fusion_fixture,
                 "merge pipeline fusion excludes ungrouped aggregate and join terminals",
                 "[integration][pipeline][merge_fusion]")
{
  fusion_flag_guard guard(/*enabled=*/true);

  SECTION("UNGROUPED_AGGREGATE")
  {
    sirius::test::with_initialized_engine(
      *con, "SELECT avg(n_nationkey) FROM nation", [&](sirius::sirius_engine& engine) {
        const auto* merge =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::MERGE_AGGREGATE);
        const auto* collector =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::RESULT_COLLECTOR);
        REQUIRE(merge != nullptr);
        REQUIRE(collector != nullptr);
        CHECK(merge != collector);
      });
  }

  SECTION("HASH_JOIN")
  {
    sirius::test::with_initialized_engine(
      *con,
      "SELECT grouped.n_regionkey, grouped.c, r.r_name "
      "FROM (SELECT n_regionkey, count(*) AS c FROM nation GROUP BY n_regionkey) grouped "
      "JOIN region r ON grouped.n_regionkey = r.r_regionkey",
      [&](sirius::sirius_engine& engine) {
        const auto* merge =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::MERGE_GROUP_BY);
        const auto* join =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::HASH_JOIN);
        REQUIRE(merge != nullptr);
        REQUIRE(join != nullptr);
        CHECK(merge != join);
      });
  }
}

TEST_CASE_METHOD(merge_fusion_fixture,
                 "merge pipeline fusion into structural sinks (ORDER_BY / TOP_N / nested GROUP BY)",
                 "[integration][pipeline][merge_fusion][structural_sinks]")
{
  fusion_flag_guard guard(/*enabled=*/true);

  // A merge whose downstream chain is a unary streaming path ending in a total-input sink
  // (ORDER_BY, TOP_N, an outer GROUP BY) fuses into that sink's pipeline. These sinks already
  // buffer their full input, so folding the merge in removes a task launch and a repository
  // round-trip without changing when the sink observes complete data.
  SECTION("ORDER BY after GROUP BY fuses MERGE_GROUP_BY into ORDER_BY")
  {
    sirius::test::with_initialized_engine(
      *con,
      "SELECT n_regionkey, count(*) AS c FROM nation GROUP BY n_regionkey ORDER BY c DESC",
      [&](sirius::sirius_engine& engine) {
        INFO(dump_pipeline_shapes(engine.new_scheduled));

        const auto* merge =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::MERGE_GROUP_BY);
        const auto* order =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::ORDER_BY);
        const auto* merge_sort =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::MERGE_SORT);

        REQUIRE(merge != nullptr);
        REQUIRE(order != nullptr);
        REQUIRE(merge_sort != nullptr);

        // MERGE_GROUP_BY fuses into the ORDER_BY sink that consumes it.
        CHECK(pipeline_contains_both(engine.new_scheduled,
                                     SiriusPhysicalOperatorType::MERGE_GROUP_BY,
                                     SiriusPhysicalOperatorType::ORDER_BY));
        CHECK(merge == order);
        REQUIRE(merge->get_sink());
        CHECK(merge->get_sink()->type == SiriusPhysicalOperatorType::ORDER_BY);

        // Global sort chain stays above the local ORDER_BY pipeline.
        CHECK(merge != merge_sort);
      });
  }

  SECTION("TOP_N after GROUP BY fuses MERGE_GROUP_BY into TOP_N")
  {
    sirius::test::with_initialized_engine(
      *con,
      "SELECT n_regionkey, count(*) AS c FROM nation GROUP BY n_regionkey "
      "ORDER BY c DESC LIMIT 3",
      [&](sirius::sirius_engine& engine) {
        INFO(dump_pipeline_shapes(engine.new_scheduled));

        const auto* group_merge =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::MERGE_GROUP_BY);
        const auto* top_n =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::TOP_N);
        const auto* top_n_merge =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::MERGE_TOP_N);
        const auto* collector =
          pipeline_containing(engine.new_scheduled, SiriusPhysicalOperatorType::RESULT_COLLECTOR);

        REQUIRE(group_merge != nullptr);
        REQUIRE(top_n != nullptr);
        REQUIRE(top_n_merge != nullptr);
        REQUIRE(collector != nullptr);

        // Inner MERGE_GROUP_BY fuses into the local TOP_N sink pipeline.
        CHECK(pipeline_contains_both(engine.new_scheduled,
                                     SiriusPhysicalOperatorType::MERGE_GROUP_BY,
                                     SiriusPhysicalOperatorType::TOP_N));
        CHECK(group_merge == top_n);
        REQUIRE(group_merge->get_sink());
        CHECK(group_merge->get_sink()->type == SiriusPhysicalOperatorType::TOP_N);

        // Outer MERGE_TOP_N still folds into the collector (streaming dead-end).
        CHECK(top_n_merge == collector);
        CHECK(group_merge != top_n_merge);
      });
  }

  SECTION("nested HASH_GROUP_BY fuses inner MERGE_GROUP_BY into outer HASH_GROUP_BY")
  {
    sirius::test::with_initialized_engine(
      *con,
      "SELECT regionkey, count(*) FROM ("
      "  SELECT n_regionkey AS regionkey, n_nationkey "
      "  FROM nation GROUP BY n_regionkey, n_nationkey"
      ") t GROUP BY regionkey",
      [&](sirius::sirius_engine& engine) {
        INFO(dump_pipeline_shapes(engine.new_scheduled));

        // Outer merge should still fuse into the collector.
        std::size_t merge_count                = 0;
        std::size_t merges_with_hgb_sink       = 0;
        std::size_t merges_with_collector_sink = 0;
        for (const auto& pipeline : engine.new_scheduled) {
          bool has_merge = false;
          for (const auto& op : pipeline->get_operators()) {
            if (op.get().type == SiriusPhysicalOperatorType::MERGE_GROUP_BY) {
              has_merge = true;
              merge_count++;
            }
          }
          if (!has_merge) { continue; }
          REQUIRE(pipeline->get_sink());
          if (pipeline->get_sink()->type == SiriusPhysicalOperatorType::HASH_GROUP_BY) {
            merges_with_hgb_sink++;
          }
          if (pipeline->get_sink()->type == SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
            merges_with_collector_sink++;
          }
        }

        REQUIRE(merge_count >= 2);
        // Inner MERGE_GROUP_BY fuses into the outer HASH_GROUP_BY sink.
        CHECK(pipeline_contains_both(engine.new_scheduled,
                                     SiriusPhysicalOperatorType::MERGE_GROUP_BY,
                                     SiriusPhysicalOperatorType::HASH_GROUP_BY));
        CHECK(merges_with_hgb_sink >= 1);
        CHECK(merges_with_collector_sink >= 1);
      });
  }
}

TEST_CASE_METHOD(merge_fusion_fixture,
                 "merge pipeline fusion preserves query results",
                 "[integration][pipeline][merge_fusion][correctness]")
{
  // Guard restores the global default even if a query below fails mid-comparison.
  fusion_flag_guard guard(/*enabled=*/true);

  SECTION("terminal GROUP BY")
  {
    require_fusion_results_match(*con,
                                 "SELECT n_regionkey, count(*) FROM nation GROUP BY n_regionkey");
  }

  SECTION("terminal TOP_N")
  {
    require_fusion_results_match(
      *con, "SELECT n_nationkey FROM nation ORDER BY n_nationkey DESC LIMIT 3");
  }

  SECTION("GROUP BY into ORDER BY")
  {
    require_fusion_results_match(
      *con, "SELECT n_regionkey, count(*) AS c FROM nation GROUP BY n_regionkey ORDER BY c DESC");
  }

  SECTION("GROUP BY into TOP_N")
  {
    require_fusion_results_match(*con,
                                 "SELECT n_regionkey, count(*) AS c FROM nation "
                                 "GROUP BY n_regionkey ORDER BY c DESC LIMIT 3");
  }

  SECTION("nested GROUP BY")
  {
    require_fusion_results_match(*con,
                                 "SELECT regionkey, count(*) FROM ("
                                 "  SELECT n_regionkey AS regionkey, n_nationkey "
                                 "  FROM nation GROUP BY n_regionkey, n_nationkey"
                                 ") t GROUP BY regionkey");
  }
}
