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

#include "exec/streaming_fragment.hpp"
#include "helper/type_conversions.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "sirius_engine.hpp"

#include <catch.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <data/data_batch_utils.hpp>
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <filesystem>
#include <memory>
#include <vector>

namespace fs = std::filesystem;

using namespace sirius::exec;

namespace {

//! A leaf source that produces real batches without depending on duckdb-native table ingestion:
//! the GPU_VALUES path is self-contained, so the test isolates the streaming seam rather than
//! the scan setup.
constexpr const char* kLeafQuery = "SELECT a FROM (VALUES (1), (2), (3), (4), (5)) t(a)";
constexpr std::size_t kLeafRows  = 5;

fs::path integration_db_path()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/duckdb/integration.duckdb";
#else
  return fs::path(__FILE__).parent_path().parent_path() /
         "integration/data/duckdb/integration.duckdb";
#endif
}

struct fragment_fixture {
  fragment_fixture()
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

    // sirius_stream_source's bind resolves its schema here; the transparent path does not
    // register a catalog, so the fragment supplies one for this connection.
    catalog = duckdb::make_shared_ptr<stream_bind_catalog>();
    con->context->registered_state->Insert(stream_bind_catalog::kStateKey, catalog);
  }

  std::unique_ptr<duckdb::Connection> con;
  duckdb::shared_ptr<stream_bind_catalog> catalog;
};

//! Total rows sitting in an output stream, draining it.
std::size_t drain_row_count(streaming_fragment& fragment, stream_id_t id)
{
  std::size_t rows = 0;
  while (auto batch = fragment.session().pull(id)) {
    rows += static_cast<std::size_t>(sirius::get_cudf_table_view(**batch).num_rows());
  }
  return rows;
}

}  // namespace

// ============================================================================
// FRAG-1: a leaf fragment runs to completion and parks its output
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-1: a leaf fragment runs and its output survives QueryEnd",
                 "[integration][streaming_fragment]")
{
  fragment_spec spec;
  spec.plan_source = sirius::test::sql_plan_source(kLeafQuery);
  spec.outputs     = {0};

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  con->BeginTransaction();
  try {
    streaming_fragment fragment(*con->context, std::move(spec));

    // One lifecycle spanning build + run: QueryBeginStandalone resets the task creator, so
    // beginning it after build() would discard the plan-time registrations.
    sirius_ctx->QueryBeginStandalone(*con->context, "frag_1");
    fragment.build();
    // The whole point of change 2: without the completion gate this call never returns.
    fragment.run();
    sirius_ctx->QueryEnd();

    // Diagnostic: separate "the source never produced a task" from "tasks ran but the sink got
    // nothing". Without this the empty-output failure has two very different causes.
    std::size_t created = 0, completed = 0;
    for (const auto& p : fragment.engine().sirius_pipelines) {
      created += p->get_tasks_created();
      completed += p->get_tasks_completed();
    }
    INFO("pipelines=" << fragment.engine().sirius_pipelines.size()
                      << " scheduled=" << fragment.engine().new_scheduled.size()
                      << " tasks_created=" << created << " tasks_completed=" << completed);
    REQUIRE(created > 0);

    // Invariant 2 -- the output repository is session-owned, outside data_repository_manager_,
    // so QueryEnd()'s clear_all_repositories() cannot touch it. The batches are still here.
    REQUIRE(fragment.output_repository(0)->total_size() > 0);
    REQUIRE(drain_row_count(fragment, 0) == kLeafRows);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-2: two fragments chained by stream id produce the single-fragment answer
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-2: a two-fragment chain matches the equivalent single query",
                 "[integration][streaming_fragment]")
{
  // The answer the chain must reproduce.
  auto expected = con->Query(std::string("SELECT count(*) FROM (") + kLeafQuery + ") t");
  REQUIRE_FALSE(expected->HasError());
  auto const expected_rows = expected->GetValue(0, 0).GetValue<std::int64_t>();

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  con->BeginTransaction();
  try {
    // Sender: reads the table, writes to its output stream.
    fragment_spec sender_spec;
    sender_spec.plan_source = sirius::test::sql_plan_source(kLeafQuery);
    sender_spec.outputs     = {0};
    streaming_fragment sender(*con->context, std::move(sender_spec));

    // Receiver: reads that stream instead of a table. No file, no parquet round-trip.
    fragment_spec receiver_spec;
    receiver_spec.plan_source =
      sirius::test::sql_plan_source("SELECT a FROM sirius_stream_source(0)");
    receiver_spec.inputs[0] = stream_input_spec{
      {"a"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      {0}};
    receiver_spec.outputs = {1};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    // Each fragment gets its own query lifecycle, spanning its build and run. The sender's
    // output repository is session-owned, so it survives the sender's QueryEnd and is still
    // there for the relay.
    sirius_ctx->QueryBeginStandalone(*con->context, "frag_sender");
    sender.build();
    sender.run();
    sirius_ctx->QueryEnd();

    sirius_ctx->QueryBeginStandalone(*con->context, "frag_receiver");
    receiver.build();

    // The relay the compute node will perform: pull from the sender's output stream and push
    // into the receiver's input stream, as native batches. No Arrow, no disk.
    std::size_t relayed_batches = 0;
    while (auto batch = sender.session().pull(0)) {
      REQUIRE(receiver.session().push(0, *batch));
      ++relayed_batches;
    }
    REQUIRE(relayed_batches > 0);
    receiver.session().close_input(0, 0);

    receiver.run();
    sirius_ctx->QueryEnd();

    REQUIRE(drain_row_count(receiver, 1) == static_cast<std::size_t>(expected_rows));

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-3: malformed specs are rejected at construction
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-3: a malformed fragment spec is rejected",
                 "[integration][streaming_fragment]")
{
  auto source = sirius::test::sql_plan_source(kLeafQuery);

  SECTION("no output stream")
  {
    fragment_spec spec;
    spec.plan_source = source;
    REQUIRE_THROWS_AS(streaming_fragment(*con->context, std::move(spec)),
                      sirius::invalid_input_exception);
  }

  SECTION("fan-out without a partition spec")
  {
    // Two destinations and no partitioning would leave the sink unable to decide where a row
    // goes, so it is refused rather than silently broadcasting.
    fragment_spec spec;
    spec.plan_source = source;
    spec.outputs     = {0, 1};
    REQUIRE_THROWS_AS(streaming_fragment(*con->context, std::move(spec)),
                      sirius::invalid_input_exception);
  }

  SECTION("duplicate output id")
  {
    fragment_spec spec;
    spec.plan_source  = source;
    spec.outputs      = {0, 0};
    spec.partitioning = sirius::op::partition_spec{{0}, {}};
    REQUIRE_THROWS_AS(streaming_fragment(*con->context, std::move(spec)),
                      sirius::invalid_input_exception);
  }

  SECTION("a declared input the plan never reads")
  {
    fragment_spec spec;
    spec.plan_source = source;  // reads nation, not the stream
    spec.inputs[7]   = stream_input_spec{
        {"a"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
        {0}};
    spec.outputs = {0};

    auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    con->BeginTransaction();
    streaming_fragment fragment(*con->context, std::move(spec));
    sirius_ctx->QueryBeginStandalone(*con->context, "frag_3");
    REQUIRE_THROWS_AS(fragment.build(), sirius::invalid_input_exception);
    sirius_ctx->QueryEnd();
    con->Rollback();
  }
}

// ============================================================================
// FRAG-CONTROL: does a RESULT_COLLECTOR-rooted plan execute when the engine is
// driven directly? Isolates "streaming sink is broken" from "this harness is".
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-CONTROL: which queries actually materialize rows on the direct path",
                 "[integration][streaming_fragment_control]")
{
  // Assert the ROW COUNT, not merely that execute() did not error. A plan that runs cleanly and
  // produces nothing looks identical to a working one unless the rows are counted, and the whole
  // streaming investigation hinges on knowing whether the source emits anything here at all.
  auto row_count_of = [&](const std::string& query) -> std::size_t {
    std::size_t rows = 0;
    sirius::test::with_initialized_engine(*con, query, [&](sirius::sirius_engine& engine) {
      REQUIRE(engine.has_result_collector());
      engine.execute();
      auto result = engine.get_result();
      REQUIRE(result != nullptr);
      REQUIRE_FALSE(result->HasError());
      auto materialized =
        duckdb::unique_ptr_cast<duckdb::QueryResult, duckdb::MaterializedQueryResult>(
          std::move(result));
      rows = materialized->RowCount();
    });
    return rows;
  };

  SECTION("VALUES leaf")
  {
    INFO("kLeafQuery = " << kLeafQuery);
    REQUIRE(row_count_of(kLeafQuery) == kLeafRows);
  }

  SECTION("table scan") { REQUIRE(row_count_of("SELECT n_regionkey FROM nation") == 25); }

  SECTION("filtered table scan")
  {
    REQUIRE(row_count_of("SELECT n_nationkey FROM nation WHERE n_regionkey = 1") == 5);
  }
}
