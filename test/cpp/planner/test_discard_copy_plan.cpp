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
 * @file test_discard_copy_plan.cpp
 * @brief Planning of `COPY (query) TO … (FORMAT sirius_discard)`: the query's plan ends in
 *        DISCARD_SINK → ungrouped SUM of the per-batch counts → the result collector, the sink's
 *        input port streams (PIPELINE barrier), `check_order` carries the ORDER BY keys into the
 *        sink or is turned off with a WARN, the planned MERGE_SORT gets its device-budget provider,
 *        and every other COPY format still declines to DuckDB.
 */

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_discard_sink.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"
#include "sirius_engine.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/pipeline_conversion_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/types.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using sirius::op::MemoryBarrierType;
using sirius::op::sirius_physical_discard_sink;
using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

/// Two BIGINT columns, k1 and k2.
constexpr char const* kSource =
  "sirius_generate(num_batches := 4, rows_per_batch := 1000, seed := 42, columns := ["
  "{name: 'k1', type: 'BIGINT', dist: 'uniform', p1: 0, p2: 1000, cardinality: 0, "
  "null_fraction: 0.0}, "
  "{name: 'k2', type: 'BIGINT', dist: 'uniform', p1: 0, p2: 1000, cardinality: 0, "
  "null_fraction: 0.0}])";

/// COPY of @p query to sirius_discard; @p options is appended after the FORMAT name, so it starts
/// with a comma (", check_order true"). The path is never opened.
std::string discard_copy(std::string const& query, std::string const& options = "")
{
  return "COPY (" + query + ") TO '/nonexistent/sirius_discard_plan' (FORMAT sirius_discard" +
         options + ")";
}

// Plans without executing: gpu_execution is off, so the connection only binds and optimizes.
struct discard_copy_plan_fixture {
  duckdb::Connection con;

  discard_copy_plan_fixture() : con(shared_connection())
  {
    REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  }

  /// The final physical plan of @p sql, GPU pipeline operators included.
  duckdb::unique_ptr<sirius_physical_operator> plan(std::string const& sql)
  {
    con.BeginTransaction();
    try {
      auto extracted = sirius::test::extract_logical_plan_sirius_order(*con.context, sql);
      sirius::planner::sirius_physical_plan_generator generator(*con.context);
      auto physical = generator.create_plan(std::move(extracted.logical_plan));
      con.Rollback();
      return physical;
    } catch (...) {
      con.Rollback();
      throw;
    }
  }

  /// A GPU batch of two BIGINT columns, the schema of `SELECT k2, k1` (or `k1, k2`).
  std::shared_ptr<cucascade::data_batch> batch(std::vector<std::int64_t> const& first,
                                               std::vector<std::int64_t> const& second)
  {
    using namespace sirius::test::operator_utils;
    auto& space = gpu_space();
    return concatenate_batches_horizontal(
      {make_numeric_batch<std::int64_t>(space, first, cudf::type_id::INT64),
       make_numeric_batch<std::int64_t>(space, second, cudf::type_id::INT64)},
      space);
  }

 private:
  static duckdb::Connection shared_connection()
  {
    REQUIRE(sirius::test::g_shared_env);
    return sirius::test::g_shared_env->make_connection();
  }

  cucascade::memory::memory_space& gpu_space()
  {
    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);
    auto spaces =
      sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    REQUIRE_FALSE(spaces.empty());
    return *const_cast<cucascade::memory::memory_space*>(spaces.front());
  }
};

/// The operator types from @p root down its single-child chain.
std::vector<SiriusPhysicalOperatorType> chain_of(sirius_physical_operator& root)
{
  std::vector<SiriusPhysicalOperatorType> types;
  for (auto* node = &root;; node = node->children.front().get()) {
    types.push_back(node->type);
    if (node->children.empty()) { break; }
    REQUIRE(node->children.size() == 1);
  }
  return types;
}

sirius_physical_discard_sink& sink_of(sirius_physical_operator& root)
{
  for (auto* node = &root;; node = node->children.front().get()) {
    if (node->type == SiriusPhysicalOperatorType::DISCARD_SINK) {
      return node->Cast<sirius_physical_discard_sink>();
    }
    REQUIRE(node->children.size() == 1);
  }
}

/// Order violations the sink counts within one task holding @p batch at sort position @p position;
/// seams between tasks are checked only at finalize.
std::uint64_t violations_in(sirius_physical_discard_sink& sink,
                            std::shared_ptr<cucascade::data_batch> batch,
                            std::size_t position)
{
  auto const before = sink.order_violations();
  sirius::op::pipelineable_operator_data input({std::move(batch)}, position);
  sink.execute(input, sirius::test::operator_utils::default_stream());
  return sink.order_violations() - before;
}

/// Whether the planner logged, at WARN, that a requested order check is not done.
bool warned_order_unchecked(sirius::test::scoped_recording_log_sink const& log)
{
  auto const records = log.records();
  return std::any_of(records.begin(), records.end(), [](auto const& record) {
    return record.level == sirius::log::level::warn &&
           record.message.rfind("sirius_discard: check_order disabled", 0) == 0;
  });
}

const sirius::pipeline::sirius_pipeline* pipeline_starting_at(
  std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>> const& pipelines,
  SiriusPhysicalOperatorType type)
{
  for (auto const& pipeline : pipelines) {
    auto const operators = pipeline->get_operators();
    if (!operators.empty() && operators.front().get().type == type) { return pipeline.get(); }
  }
  return nullptr;
}

/// The barriers of the ports that feed a DISCARD_SINK in the pipelines of @p sql.
std::vector<MemoryBarrierType> discard_input_barriers(duckdb::Connection& con,
                                                      std::string const& sql,
                                                      SiriusPhysicalOperatorType producer)
{
  std::vector<MemoryBarrierType> barriers;
  sirius::test::with_conversion_result(con, sql, [&](auto const& result) {
    auto const* discard =
      pipeline_starting_at(result.scheduled_pipelines, SiriusPhysicalOperatorType::DISCARD_SINK);
    REQUIRE(discard != nullptr);
    for (auto const& wiring : result.repository_wirings) {
      if (wiring.dest_pipeline.get() != discard) { continue; }
      REQUIRE(wiring.source_op != nullptr);
      CHECK(wiring.source_op->type == producer);
      // The producer runs in its own pipeline, so the two can run at the same time.
      CHECK(wiring.source_pipeline.get() != discard);
      barriers.push_back(wiring.barrier_type);
    }
  });
  return barriers;
}

}  // namespace

TEST_CASE("sirius_discard COPY of an ORDER BY plans MERGE_SORT into a streaming discard sink",
          "[planner][discard_copy_plan][shared_context]")
{
  using T = SiriusPhysicalOperatorType;
  discard_copy_plan_fixture fixture;
  auto const sql = discard_copy(std::string("SELECT k1, k2 FROM ") + kSource + " ORDER BY k1");

  SECTION("tree: MERGE_SORT -> DISCARD_SINK -> SUM -> one BIGINT")
  {
    auto physical = fixture.plan(sql);
    REQUIRE(physical);
    auto const chain = chain_of(*physical);
    REQUIRE(chain.size() > 5);
    CHECK(
      std::vector<T>(chain.begin(), chain.begin() + 5) ==
      std::vector<T>{
        T::PROJECTION, T::MERGE_AGGREGATE, T::UNGROUPED_AGGREGATE, T::DISCARD_SINK, T::MERGE_SORT});
    // COPY's result: the row count.
    CHECK(physical->types == sirius::from_duckdb_vec(
                               duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT}));
    // The sink sees the query's columns and declares what it emits: one count.
    auto const& sink = sink_of(*physical);
    CHECK(sink.children.front()->types.size() == 2);
    CHECK(sink.input_types().size() == 2);
    CHECK(sink.get_types() == physical->types);
    // Without the order check, batches that spilled while they waited are counted where they are.
    CHECK_FALSE(sink.input_residency_policy("default").materialize_inputs_on_gpu);
    CHECK_FALSE(sink.is_sink());
    CHECK_FALSE(sink.is_source());
  }

  SECTION("the MERGE_SORT sizes its windows from the context's device budget")
  {
    // Without the provider the operator falls back to a space hint and the default executor
    // width, so the configured num_threads and the real GPU budget would never reach the sort.
    auto physical = fixture.plan(sql);
    REQUIRE(physical);
    auto* node = physical.get();
    while (node->type != T::MERGE_SORT) {
      REQUIRE(node->children.size() == 1);
      node = node->children.front().get();
    }
    CHECK(node->Cast<sirius::op::sirius_physical_merge_sort>().has_device_budget_provider());
  }

  SECTION("the MERGE_SORT -> DISCARD_SINK port is a PIPELINE barrier")
  {
    CHECK(discard_input_barriers(fixture.con, sql, T::MERGE_SORT) ==
          std::vector<MemoryBarrierType>{MemoryBarrierType::PIPELINE});
  }

  SECTION("the count reaches the result collector")
  {
    sirius::test::with_initialized_engine(fixture.con, sql, [](sirius::sirius_engine& engine) {
      auto const* discard = pipeline_starting_at(engine.new_scheduled, T::DISCARD_SINK);
      REQUIRE(discard != nullptr);
      // The sink and the partial SUM share one pipeline, fed by MERGE_SORT's.
      auto const operators = discard->get_operators();
      REQUIRE(operators.size() == 2);
      CHECK(operators.back().get().type == T::UNGROUPED_AGGREGATE);
      REQUIRE(pipeline_starting_at(engine.new_scheduled, T::MERGE_SORT) != nullptr);

      auto const collector =
        std::find_if(engine.new_scheduled.begin(), engine.new_scheduled.end(), [](auto const& p) {
          return p->get_sink() && p->get_sink()->type == T::RESULT_COLLECTOR;
        });
      REQUIRE(collector != engine.new_scheduled.end());
      CHECK((*collector)->get_source()->type == T::PROJECTION);
    });
  }
}

TEST_CASE("sirius_discard check_order carries the ORDER BY keys into the sink",
          "[planner][discard_copy_plan][shared_context]")
{
  discard_copy_plan_fixture fixture;
  // The sink's input is (k2, k1): the keys are column 1 descending, then column 0 ascending.
  auto const query = std::string("SELECT k2, k1 FROM ") + kSource + " ORDER BY k1 DESC, k2";

  SECTION("check_order true")
  {
    sirius::test::scoped_recording_log_sink log("warn");
    auto physical = fixture.plan(discard_copy(query, ", check_order true"));
    CHECK_FALSE(warned_order_unchecked(log));
    auto& sink = sink_of(*physical);
    CHECK(sink.options().check_order);
    CHECK(sink.order_checked());
    CHECK(sink.options().key_columns == std::vector<cudf::size_type>{1, 0});
    CHECK(sink.options().column_order ==
          std::vector<cudf::order>{cudf::order::DESCENDING, cudf::order::ASCENDING});
    // The order check reads the keys on the GPU.
    CHECK(sink.input_residency_policy("default").materialize_inputs_on_gpu);
    CHECK(violations_in(sink, fixture.batch({1, 2, 0}, {9, 9, 5}), 0) == 0);
    // k1 ascending: the first key's direction.
    CHECK(violations_in(sink, fixture.batch({0, 1, 2}, {5, 9, 9}), 1) == 1);
    // k1 ties with k2 descending: the second key.
    CHECK(violations_in(sink, fixture.batch({2, 1, 0}, {9, 9, 5}), 2) == 1);
    CHECK(sink.ranges_checked() == 3);
  }

  SECTION("check_order false")
  {
    sirius::test::scoped_recording_log_sink log("warn");
    auto physical = fixture.plan(discard_copy(query, ", check_order false"));
    // Not asked for, so nothing to warn about.
    CHECK_FALSE(warned_order_unchecked(log));
    CHECK_FALSE(sink_of(*physical).options().check_order);
    CHECK_FALSE(sink_of(*physical).order_checked());
    CHECK(violations_in(sink_of(*physical), fixture.batch({0, 1, 2}, {5, 9, 9}), 0) == 0);
  }

  SECTION("default")
  {
    auto physical = fixture.plan(discard_copy(query));
    CHECK(violations_in(sink_of(*physical), fixture.batch({0, 1, 2}, {5, 9, 9}), 0) == 0);
  }
}

TEST_CASE(
  "sirius_discard check_order is off, with a WARN, when the ORDER BY keys are not output "
  "columns",
  "[planner][discard_copy_plan][shared_context]")
{
  discard_copy_plan_fixture fixture;
  auto const from = std::string(" FROM ") + kSource;
  // Descending in both columns: a violation for any ascending key the sink might check.
  auto const descending = [&] { return fixture.batch({2, 1, 0}, {9, 5, 3}); };
  // The user asked for the check: a plan that cannot do it says so, in the log and on the sink.
  sirius::test::scoped_recording_log_sink log("warn");

  SECTION("control: a plain column key is checked")
  {
    auto physical =
      fixture.plan(discard_copy("SELECT k1, k2" + from + " ORDER BY k1", ", check_order true"));
    CHECK(violations_in(sink_of(*physical), descending(), 0) == 1);
    CHECK(sink_of(*physical).order_checked());
    CHECK_FALSE(warned_order_unchecked(log));
  }

  SECTION("expression key")
  {
    auto physical = fixture.plan(
      discard_copy("SELECT k1, k2" + from + " ORDER BY k1 + k2", ", check_order true"));
    CHECK(violations_in(sink_of(*physical), descending(), 0) == 0);
    CHECK_FALSE(sink_of(*physical).order_checked());
    CHECK(sink_of(*physical).ranges_checked() == 0);
    CHECK(warned_order_unchecked(log));
  }

  SECTION("key dropped from the output")
  {
    // Two output columns, neither of them the key k1.
    auto physical = fixture.plan(
      discard_copy("SELECT k2, k2 + 1 AS k3" + from + " ORDER BY k1", ", check_order true"));
    REQUIRE(sink_of(*physical).children.front()->types.size() == 2);
    CHECK(violations_in(sink_of(*physical), descending(), 0) == 0);
    CHECK_FALSE(sink_of(*physical).order_checked());
    CHECK(warned_order_unchecked(log));
  }

  SECTION("ORDER BY with LIMIT: the root is not an ORDER BY")
  {
    auto physical = fixture.plan(
      discard_copy("SELECT k1, k2" + from + " ORDER BY k1 LIMIT 10", ", check_order true"));
    CHECK(violations_in(sink_of(*physical), descending(), 0) == 0);
    CHECK_FALSE(sink_of(*physical).order_checked());
    CHECK(warned_order_unchecked(log));
  }
}

TEST_CASE("sirius_discard COPY of a query without ORDER BY plans with the order check off",
          "[planner][discard_copy_plan][shared_context]")
{
  using T = SiriusPhysicalOperatorType;
  discard_copy_plan_fixture fixture;
  auto const sql = discard_copy(std::string("SELECT k1, k2 FROM ") + kSource, ", check_order true");

  sirius::test::scoped_recording_log_sink log("warn");
  auto physical = fixture.plan(sql);
  REQUIRE(physical);
  CHECK(warned_order_unchecked(log));
  CHECK_FALSE(sink_of(*physical).order_checked());
  auto const chain = chain_of(*physical);
  REQUIRE(chain.size() >= 5);
  CHECK(std::vector<T>(chain.begin(), chain.begin() + 4) ==
        std::vector<T>{T::PROJECTION, T::MERGE_AGGREGATE, T::UNGROUPED_AGGREGATE, T::DISCARD_SINK});
  CHECK(std::find(chain.begin(), chain.end(), T::MERGE_SORT) == chain.end());
  // Nothing to check against: unsorted input is not a violation.
  CHECK(violations_in(sink_of(*physical), fixture.batch({3, 1, 2}, {5, 9, 1}), 0) == 0);

  // With no MERGE_SORT to end a pipeline below it, the sink is fused into its source's pipeline:
  // no pipeline starts at it and no repository feeds it, so its input cannot wait behind a barrier.
  sirius::test::with_conversion_result(fixture.con, sql, [](auto const& result) {
    CHECK(pipeline_starting_at(result.scheduled_pipelines, T::DISCARD_SINK) == nullptr);
    std::vector<std::vector<T>> with_sink;
    for (auto const& pipeline : result.scheduled_pipelines) {
      std::vector<T> types;
      for (auto const& op : pipeline->get_operators()) {
        types.push_back(op.get().type);
      }
      if (std::find(types.begin(), types.end(), T::DISCARD_SINK) != types.end()) {
        with_sink.push_back(std::move(types));
      }
    }
    REQUIRE(with_sink.size() == 1);
    CHECK(with_sink.front() ==
          std::vector<T>{T::GENERATOR, T::DISCARD_SINK, T::UNGROUPED_AGGREGATE});
  });
}

// The plan is run on the engine directly, without the transparent path: the SUM, its cast and
// the streaming port have to work together. ([copy_discard] covers the statement typed as SQL.)
TEST_CASE("sirius_discard COPY plan returns the query's row count",
          "[planner][discard_copy_plan][shared_context]")
{
  discard_copy_plan_fixture fixture;
  auto const rows_copied = [&](std::string const& sql) {
    std::int64_t rows = -1;
    auto sirius_ctx =
      fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
    duckdb::SiriusContext::StandaloneQueryScope window(
      *sirius_ctx, *fixture.con.context, "discard_copy_plan");
    sirius::test::with_initialized_engine(
      fixture.con,
      sql,
      [&](sirius::sirius_engine& engine) {
        engine.execute();
        auto result = engine.get_result();
        REQUIRE(result != nullptr);
        INFO(result->ToString());
        REQUIRE_FALSE(result->HasError());
        REQUIRE(result->RowCount() == 1);
        REQUIRE(result->ColumnCount() == 1);
        REQUIRE(result->GetTypes()[0] == duckdb::LogicalType::BIGINT);
        rows = result->Collection().GetRows().GetValue(0, 0).GetValue<std::int64_t>();
      },
      window.query_id());
    window.finish();
    return rows;
  };
  auto const source = std::string("SELECT k2, k1 FROM ") + kSource;

  SECTION("ORDER BY with the order check")
  {
    CHECK(rows_copied(discard_copy(source + " ORDER BY k1 DESC, k2", ", check_order true")) ==
          4000);
  }
  SECTION("no ORDER BY") { CHECK(rows_copied(discard_copy(source)) == 4000); }
  SECTION("no rows")
  {
    CHECK(rows_copied(discard_copy(source + " WHERE k1 < 0 ORDER BY k1")) == 0);
  }
}
