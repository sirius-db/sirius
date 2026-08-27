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

#include "../operator/operator_test_utils.hpp"
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

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace fs = std::filesystem;

using namespace sirius::exec;

namespace {

//! A leaf source that produces real batches without depending on duckdb-native table ingestion:
//! the GPU_VALUES path is self-contained, so the test isolates the streaming seam rather than
//! the scan setup.
constexpr const char* kLeafQuery = "SELECT a FROM (VALUES (1), (2), (3), (4), (5)) t(a)";
constexpr std::size_t kLeafRows  = 5;

fs::path lineitem_parquet_path()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/parquet/lineitem.parquet";
#else
  return fs::path(__FILE__).parent_path().parent_path() /
         "integration/data/parquet/lineitem.parquet";
#endif
}

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

//! The execution window a fragment's build/run must sit inside. RAII matters here: a `REQUIRE`
//! that fails inside a hand-bracketed window would leave the slot held and self-deadlock in the
//! test's `Rollback`, so the scope's destructor backstop is what lets a failing assertion fail.
//! Tests that assert on post-cleanup state call `finish()` explicitly.
using query_window = duckdb::SiriusContext::StandaloneQueryScope;

//! Every INTEGER value sitting in an output stream, draining it. Row counts alone would not
//! catch a hop that corrupted, dropped or duplicated values.
std::vector<std::int32_t> drain_values(streaming_fragment& fragment, stream_id_t id)
{
  std::vector<std::int32_t> values;
  while (auto batch = fragment.session().pull(id)) {
    auto view = sirius::get_cudf_table_view(**batch);
    auto col  = sirius::test::operator_utils::copy_column_to_host<std::int32_t>(view.column(0));
    values.insert(values.end(), col.begin(), col.end());
  }
  std::sort(values.begin(), values.end());
  return values;
}

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
                 "FRAG-1: a leaf fragment runs and its output survives the window cleanup",
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

    // One window spanning build + run (shared query window).
    query_window window(*sirius_ctx, *con->context, "frag_1");
    fragment.build(window.query_id());
    // Pipeline completion gate: without it run() blocks forever.
    fragment.run();
    window.finish();

    // Separate "source never produced" from "tasks ran but sink empty".
    std::size_t created = 0, completed = 0;
    for (const auto& p : fragment.engine().sirius_pipelines) {
      created += p->get_tasks_created();
      completed += p->get_tasks_completed();
    }
    INFO("pipelines=" << fragment.engine().sirius_pipelines.size()
                      << " scheduled=" << fragment.engine().new_scheduled.size()
                      << " tasks_created=" << created << " tasks_completed=" << completed);
    REQUIRE(created > 0);

    // Repositories escape data_repository_manager_ cleanup — batches survive the window.
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
    // Sender: reads a VALUES list, writes to its output stream.
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

    // Each fragment gets its own query window, spanning its build and run. The sender's
    // output repository is session-owned, so it survives the sender's window cleanup and is
    // still there for the relay.
    {
      query_window sender_window(*sirius_ctx, *con->context, "frag_sender");
      sender.build(sender_window.query_id());
      sender.run();
      sender_window.finish();
    }

    query_window receiver_window(*sirius_ctx, *con->context, "frag_receiver");
    receiver.build(receiver_window.query_id());

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
    receiver_window.finish();

    // Values, not just a count: the chain must deliver exactly what the sender produced.
    auto const received = drain_values(receiver, 1);
    REQUIRE(received.size() == static_cast<std::size_t>(expected_rows));
    REQUIRE(received == std::vector<std::int32_t>{1, 2, 3, 4, 5});

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
    // Two destinations without partitioning would silently broadcast; refuse instead.
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
    spec.partitioning = sirius::op::partition_spec{{0}};
    REQUIRE_THROWS_AS(streaming_fragment(*con->context, std::move(spec)),
                      sirius::invalid_input_exception);
  }

  SECTION("a declared input the plan never reads")
  {
    fragment_spec spec;
    spec.plan_source = source;  // reads a VALUES list, not the stream
    spec.inputs[7]   = stream_input_spec{
        {"a"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
        {0}};
    spec.outputs = {0};

    auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    con->BeginTransaction();
    streaming_fragment fragment(*con->context, std::move(spec));
    {
      query_window window(*sirius_ctx, *con->context, "frag_3");
      REQUIRE_THROWS_AS(fragment.build(window.query_id()), sirius::invalid_input_exception);
    }
    con->Rollback();
  }
}

// ============================================================================
// FRAG-CONTROL: RESULT_COLLECTOR-rooted plan on the direct engine path.
// Isolates harness failures from sink failures (pair with SINKROOT-4).
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-CONTROL: which queries actually materialize rows on the direct path",
                 "[integration][streaming_fragment_control]")
{
  // Assert row count, not merely that execute() succeeded.
  auto row_count_of = [&](const std::string& query) -> std::size_t {
    std::size_t rows = 0;
    // with_initialized_engine synthesizes its own query id, but execute() still needs a real
    // window open: only a window begin points the task creator at this connection.
    auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
    query_window window(*sirius_ctx, *con->context, "frag_control");
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
    window.finish();
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

// ============================================================================
// FRAG-4: parquet GPU scan across a fragment boundary (real batch counts).
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-4: a parquet scan crosses a fragment boundary",
                 "[integration][streaming_fragment]")
{
  auto const parquet = lineitem_parquet_path();
  REQUIRE(fs::exists(parquet));

  // Filter on l_quantity so row-group pruning does not collapse the scan. Still one batch
  // per file; FRAG-5 covers multi-batch streams.
  auto const leaf =
    "SELECT l_orderkey FROM read_parquet('" + parquet.string() + "') WHERE l_quantity < 2";

  auto expected = con->Query("SELECT count(*) FROM (" + leaf + ") t");
  REQUIRE_FALSE(expected->HasError());
  auto const expected_rows =
    static_cast<std::size_t>(expected->GetValue(0, 0).GetValue<std::int64_t>());
  REQUIRE(expected_rows > 0);

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  con->BeginTransaction();
  try {
    fragment_spec sender_spec;
    sender_spec.plan_source = sirius::test::sql_plan_source(leaf);
    sender_spec.outputs     = {0};
    streaming_fragment sender(*con->context, std::move(sender_spec));

    fragment_spec receiver_spec;
    receiver_spec.plan_source =
      sirius::test::sql_plan_source("SELECT l_orderkey FROM sirius_stream_source(0)");
    receiver_spec.inputs[0] = stream_input_spec{
      {"l_orderkey"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT}),
      {0}};
    receiver_spec.outputs = {1};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    {
      query_window sender_window(*sirius_ctx, *con->context, "frag4_sender");
      sender.build(sender_window.query_id());
      sender.run();
      sender_window.finish();
    }

    query_window receiver_window(*sirius_ctx, *con->context, "frag4_receiver");
    receiver.build(receiver_window.query_id());

    std::size_t relayed_batches = 0;
    std::size_t relayed_rows    = 0;
    while (auto batch = sender.session().pull(0)) {
      relayed_rows += static_cast<std::size_t>(sirius::get_cudf_table_view(**batch).num_rows());
      REQUIRE(receiver.session().push(0, *batch));
      ++relayed_batches;
    }
    REQUIRE(relayed_batches > 0);
    // Everything the sender produced crosses the hop; nothing is dropped in transit.
    REQUIRE(relayed_rows == expected_rows);
    receiver.session().close_input(0, 0);

    receiver.run();
    receiver_window.finish();

    REQUIRE(drain_row_count(receiver, 1) == expected_rows);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-5: multi-batch drain. FRAG-2/4 hop one batch; two senders fill the queue here.
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-5: a multi-batch stream drains completely",
                 "[integration][streaming_fragment]")
{
  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  // Disjoint halves so the output identifies which batches arrived.
  constexpr const char* kFirstHalf  = "SELECT a FROM (VALUES (1), (2), (3)) t(a)";
  constexpr const char* kSecondHalf = "SELECT a FROM (VALUES (4), (5), (6)) t(a)";

  con->BeginTransaction();
  try {
    auto make_sender = [&](const char* query) {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(query);
      spec.outputs     = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };

    auto first  = make_sender(kFirstHalf);
    auto second = make_sender(kSecondHalf);

    fragment_spec receiver_spec;
    receiver_spec.plan_source =
      sirius::test::sql_plan_source("SELECT a FROM sirius_stream_source(0)");
    receiver_spec.inputs[0] = stream_input_spec{
      {"a"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      {0}};
    receiver_spec.outputs = {1};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    for (auto* sender : {first.get(), second.get()}) {
      query_window sender_window(*sirius_ctx, *con->context, "frag5_sender");
      sender->build(sender_window.query_id());
      sender->run();
      sender_window.finish();
    }

    query_window receiver_window(*sirius_ctx, *con->context, "frag5_receiver");
    receiver.build(receiver_window.query_id());

    std::size_t relayed_batches = 0;
    for (auto* sender : {first.get(), second.get()}) {
      while (auto batch = sender->session().pull(0)) {
        REQUIRE(receiver.session().push(0, *batch));
        ++relayed_batches;
      }
    }
    // Multi-batch premise: if only one batch arrives this degrades to FRAG-2.
    REQUIRE(relayed_batches > 1);
    receiver.session().close_input(0, 0);

    receiver.run();
    receiver_window.finish();

    // INFO only: one batch per task today; coalescing would change the count.
    std::size_t created = 0, completed = 0;
    for (const auto& p : receiver.engine().sirius_pipelines) {
      created += p->get_tasks_created();
      completed += p->get_tasks_completed();
    }
    INFO("relayed_batches=" << relayed_batches << " receiver tasks_created=" << created
                            << " tasks_completed=" << completed);

    REQUIRE(drain_values(receiver, 1) == std::vector<std::int32_t>{1, 2, 3, 4, 5, 6});

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-6: a join fragment where one input stream closes empty before run().
// The CN flow delivers and closes ALL exchange input before run(); a stream
// whose rows all hashed to the other CN closes with zero batches. The join
// must still run to completion with an empty result — not wedge waiting on
// the side that will never produce (TPC-H q07 hang shape: one empty input
// among several live ones).
// ============================================================================

namespace {

//! Turns a wedged run() into a loud failure: the engine-side scheduling watchdog
//! (SIRIUS_QUERY_WATCHDOG_SECS) fails a query with no scheduling progress, and
//! streaming_fragment::run() rethrows it — so a hang fails in seconds instead of
//! blocking the suite forever.
struct watchdog_guard {
  explicit watchdog_guard(const char* secs) { setenv("SIRIUS_QUERY_WATCHDOG_SECS", secs, 1); }
  ~watchdog_guard() { unsetenv("SIRIUS_QUERY_WATCHDOG_SECS"); }
  watchdog_guard(const watchdog_guard&)            = delete;
  watchdog_guard& operator=(const watchdog_guard&) = delete;
};

}  // namespace

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-6: a join whose input stream closes empty before run() completes",
                 "[integration][streaming_fragment][empty_input_join]")
{
  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  constexpr const char* kJoinQuery =
    "SELECT l.a, r.b FROM sirius_stream_source(0) l JOIN sirius_stream_source(1) r ON l.a = r.b";

  // Which input closes empty. Both directions are covered because the Sirius planner may swap
  // the join's build/probe sides (both stream sources estimate cardinality 1), and nothing
  // guarantees which side of the exchange starves first in a cluster.
  stream_id_t empty_stream = 1;
  SECTION("build side closes empty") { empty_stream = 1; }
  SECTION("probe side closes empty") { empty_stream = 0; }
  stream_id_t const live_stream = empty_stream == 1 ? 0 : 1;

  watchdog_guard watchdog("20");

  con->BeginTransaction();
  try {
    // Sender: real batches for the live side.
    fragment_spec sender_spec;
    sender_spec.plan_source = sirius::test::sql_plan_source(kLeafQuery);
    sender_spec.outputs     = {0};
    streaming_fragment sender(*con->context, std::move(sender_spec));

    fragment_spec receiver_spec;
    receiver_spec.plan_source = sirius::test::sql_plan_source(kJoinQuery);
    receiver_spec.inputs[0]   = stream_input_spec{
        {"a"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
        {0}};
    receiver_spec.inputs[1] = stream_input_spec{
      {"b"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      {0}};
    receiver_spec.outputs = {2};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    {
      query_window sender_window(*sirius_ctx, *con->context, "frag6_sender");
      sender.build(sender_window.query_id());
      sender.run();
      sender_window.finish();
    }

    query_window receiver_window(*sirius_ctx, *con->context, "frag6_receiver");
    receiver.build(receiver_window.query_id());

    // The CN arrival order: every batch lands and every input closes before run().
    std::size_t relayed_batches = 0;
    while (auto batch = sender.session().pull(0)) {
      REQUIRE(receiver.session().push(live_stream, *batch));
      ++relayed_batches;
    }
    REQUIRE(relayed_batches > 0);
    receiver.session().close_input(live_stream, 0);
    // The empty side: closed without ever carrying a batch.
    receiver.session().close_input(empty_stream, 0);

    receiver.run();
    receiver_window.finish();

    // An inner join against an empty side yields no rows — but it must yield.
    REQUIRE(drain_row_count(receiver, 2) == 0);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-7: the q07 hang shape — a join CASCADE where the empty stream feeds the
// build side of an upper join whose probe side is another join's output, not a
// stream source. All input arrives and closes pre-run; only one stream is
// empty. FRAG-6 covers the empty stream feeding a join directly; here the
// probe rows only materialize while the query is running, after the empty
// build side already finished.
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-7: a join cascade completes when an upper build stream closes empty",
                 "[integration][streaming_fragment][empty_input_join]")
{
  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  // Optimizer disabled, so the join tree stays left-deep as written: ((a JOIN b) JOIN c) with
  // c — the empty stream — as the build side of the upper join.
  constexpr const char* kCascadeQuery =
    "SELECT a.x, c.z FROM sirius_stream_source(0) a "
    "JOIN sirius_stream_source(1) b ON a.x = b.y "
    "JOIN sirius_stream_source(2) c ON a.x = c.z";

  watchdog_guard watchdog("20");

  con->BeginTransaction();
  try {
    auto make_sender = [&]() {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(kLeafQuery);
      spec.outputs     = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };

    auto int_types = [] {
      return sirius::from_duckdb_vec(
        duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
    };

    fragment_spec receiver_spec;
    receiver_spec.plan_source = sirius::test::sql_plan_source(kCascadeQuery);
    receiver_spec.inputs[0]   = stream_input_spec{{"x"}, int_types(), {0}};
    receiver_spec.inputs[1]   = stream_input_spec{{"y"}, int_types(), {0}};
    receiver_spec.inputs[2]   = stream_input_spec{{"z"}, int_types(), {0}};
    receiver_spec.outputs     = {3};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    // Senders run first, each in its own window; their output repositories survive the
    // window cleanup, so the batches are still parked for the relay below.
    auto first  = make_sender();
    auto second = make_sender();
    for (auto* sender : {first.get(), second.get()}) {
      query_window sender_window(*sirius_ctx, *con->context, "frag7_sender");
      sender->build(sender_window.query_id());
      sender->run();
      sender_window.finish();
    }

    query_window receiver_window(*sirius_ctx, *con->context, "frag7_receiver");
    receiver.build(receiver_window.query_id());

    // Live streams 0 and 1 get real batches; stream 2 closes without ever carrying one.
    for (auto [sender, live] :
         {std::pair{first.get(), stream_id_t{0}}, std::pair{second.get(), stream_id_t{1}}}) {
      std::size_t relayed = 0;
      while (auto batch = sender->session().pull(0)) {
        REQUIRE(receiver.session().push(live, *batch));
        ++relayed;
      }
      REQUIRE(relayed > 0);
      receiver.session().close_input(live, 0);
    }
    receiver.session().close_input(2, 0);

    receiver.run();
    receiver_window.finish();

    // The empty build side annihilates the cascade — zero rows, but delivered.
    REQUIRE(drain_row_count(receiver, 3) == 0);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-8: the full q07-cn1 fragment shape. Lower join: probe stream carries
// real fan-in batches, build stream closes EMPTY. Upper join: build stream
// carries one real batch, probe side is the (empty) lower join result. A
// grouped aggregate sits on top and the sink hash-partitions to two
// destinations. All input arrives and closes before run(), exactly the CN
// arrival order. Every piece passed alone (FRAG-6/7); q07 hangs with them
// composed.
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-8: the q07 fragment shape completes when the lower build stream is empty",
                 "[integration][streaming_fragment][empty_input_join]")
{
  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  // Mirrors the hung fragment on cn1 (streams 14/16/20 -> 24):
  //   Aggregate[group, sum]
  //     Join2[Inner f.k2 = n2.b]        <- build (n2) has ONE batch
  //       Join1[Inner f.k1 = n1.a]      <- build (n1) closed EMPTY
  //         f  (probe: fan-in, 2 senders x multiple batches)
  //         n1
  //       n2
  constexpr const char* kQ07Shape =
    "SELECT n1.a AS g1, n2.b AS g2, sum(f.v) AS s "
    "FROM sirius_stream_source(0) f "
    "JOIN sirius_stream_source(1) n1 ON f.k1 = n1.a "
    "JOIN sirius_stream_source(2) n2 ON f.k2 = n2.b "
    "GROUP BY n1.a, n2.b";

  constexpr const char* kProbeQuery =
    "SELECT a AS v, a AS k1, a AS k2 FROM (VALUES (1), (2), (3), (4), (5)) t(a)";

  watchdog_guard watchdog("20");

  con->BeginTransaction();
  try {
    auto make_sender = [&](const char* query) {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(query);
      spec.outputs     = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };

    auto int_types = [](std::size_t n) {
      duckdb::vector<duckdb::LogicalType> t(n, duckdb::LogicalType::INTEGER);
      return sirius::from_duckdb_vec(t);
    };

    // Senders first: their windows may not nest inside the receiver's.
    auto probe_a = make_sender(kProbeQuery);  // fan-in sender 0 of stream 0
    auto probe_b = make_sender(kProbeQuery);  // fan-in sender 1 of stream 0
    auto n2      = make_sender(kLeafQuery);   // the one real batch for stream 2
    for (auto* sender : {probe_a.get(), probe_b.get(), n2.get()}) {
      query_window sender_window(*sirius_ctx, *con->context, "frag8_sender");
      sender->build(sender_window.query_id());
      sender->run();
      sender_window.finish();
    }

    fragment_spec receiver_spec;
    receiver_spec.plan_source  = sirius::test::sql_plan_source(kQ07Shape);
    receiver_spec.inputs[0]    = stream_input_spec{{"v", "k1", "k2"}, int_types(3), {0, 1}};
    receiver_spec.inputs[1]    = stream_input_spec{{"a"}, int_types(1), {0}};
    receiver_spec.inputs[2]    = stream_input_spec{{"b"}, int_types(1), {0}};
    receiver_spec.outputs      = {3, 4};
    receiver_spec.partitioning = sirius::op::partition_spec{{0, 1}};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    query_window receiver_window(*sirius_ctx, *con->context, "frag8_receiver");
    receiver.build(receiver_window.query_id());

    // CN arrival order: all pushes, then all closes, then run().
    for (auto [sender, sender_id] :
         {std::pair{probe_a.get(), sender_id_t{0}}, std::pair{probe_b.get(), sender_id_t{1}}}) {
      std::size_t relayed = 0;
      while (auto batch = sender->session().pull(0)) {
        REQUIRE(receiver.session().push(0, *batch));
        ++relayed;
      }
      REQUIRE(relayed > 0);
      receiver.session().close_input(0, sender_id);
    }
    receiver.session().close_input(1, 0);  // n1: closed without ever carrying a batch
    {
      std::size_t relayed = 0;
      while (auto batch = n2->session().pull(0)) {
        REQUIRE(receiver.session().push(2, *batch));
        ++relayed;
      }
      REQUIRE(relayed == 1);
      receiver.session().close_input(2, 0);
    }

    receiver.run();
    receiver_window.finish();

    // The empty n1 build annihilates everything above it: both partitions deliver zero rows.
    REQUIRE(drain_row_count(receiver, 3) + drain_row_count(receiver, 4) == 0);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-9: FRAG-8 with full q07 fidelity — real types (fp64 / DATE / VARCHAR),
// the OR residual on the upper join, year() in the projection, a 3-key
// grouped sum, and a real-volume parquet probe. The volume is what matters:
// the planner sizes the folded ~33 MB lineitem side as the lower join's BUILD
// (too big for the small-table broadcast threshold, so BUILD_PROBE runs
// non-broadcast) and the EMPTY nation stream becomes its PROBE. Before the
// build-only-slot discard was extended past broadcast mode, that slot held
// its build batch forever, select_build_probe_action answered wait_for_probe
// against the finished-empty probe producer, and the fragment hung — the
// TPC-H q07 2-CN wedge this test reproduces (~33 s hang, 3/3 deterministic).
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-9: the faithful q07 fragment completes when a probe stream closes empty",
                 "[integration][streaming_fragment][empty_input_join]")
{
  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  auto const parquet = lineitem_parquet_path();
  REQUIRE(fs::exists(parquet));

  // The exact hung plan on cn1 (streams 14/16/20 -> 24), keys synthesized from lineitem:
  //   Aggregate[n1name, n2name, year => sum(price * (1 - discount))]
  //     Join2[Inner c_nationkey = n2key AND (FRANCE/GERMANY OR GERMANY/FRANCE)]
  //       Join1[Inner s_nationkey = n1key]   <- n1 stream EMPTY; the planner sizes the folded
  //                                             lineitem side as this join's build, so the empty
  //                                             stream is its PROBE
  constexpr const char* kQ07 =
    "SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, year(f.l_shipdate) AS l_year, "
    "sum(f.l_extendedprice * (1 - f.l_discount)) AS revenue "
    "FROM sirius_stream_source(0) f "
    "JOIN sirius_stream_source(1) n1 ON f.s_nationkey = n1.n_nationkey "
    "JOIN sirius_stream_source(2) n2 ON f.c_nationkey = n2.n_nationkey AND "
    "((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY') OR "
    " (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE')) "
    "GROUP BY n1.n_name, n2.n_name, year(f.l_shipdate)";

  auto const probe_query =
    "SELECT CAST(l_extendedprice AS DOUBLE) AS l_extendedprice, "
    "CAST(l_discount AS DOUBLE) AS l_discount, l_shipdate, "
    "CAST(l_suppkey % 25 AS INTEGER) AS s_nationkey, "
    "CAST(l_orderkey % 25 AS INTEGER) AS c_nationkey "
    "FROM read_parquet('" +
    parquet.string() + "')";

  constexpr const char* kNationQuery =
    "SELECT * FROM (VALUES (2, 'GERMANY'), (7, 'FRANCE')) t(n_nationkey, n_name)";

  watchdog_guard watchdog("30");

  con->BeginTransaction();
  try {
    auto make_sender = [&](const std::string& query) {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(query);
      spec.outputs     = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };

    auto probe_a = make_sender(probe_query);
    auto probe_b = make_sender(probe_query);
    auto n2      = make_sender(kNationQuery);
    for (auto* sender : {probe_a.get(), probe_b.get(), n2.get()}) {
      query_window sender_window(*sirius_ctx, *con->context, "frag9_sender");
      sender->build(sender_window.query_id());
      sender->run();
      sender_window.finish();
    }

    auto probe_types =
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::DOUBLE,
                                                                  duckdb::LogicalType::DOUBLE,
                                                                  duckdb::LogicalType::DATE,
                                                                  duckdb::LogicalType::INTEGER,
                                                                  duckdb::LogicalType::INTEGER});
    auto nation_types = sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{
      duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR});

    fragment_spec receiver_spec;
    receiver_spec.plan_source = sirius::test::sql_plan_source(kQ07);
    receiver_spec.inputs[0]   = stream_input_spec{
        {"l_extendedprice", "l_discount", "l_shipdate", "s_nationkey", "c_nationkey"},
      probe_types,
        {0, 1}};
    receiver_spec.inputs[1]    = stream_input_spec{{"n_nationkey", "n_name"}, nation_types, {0}};
    receiver_spec.inputs[2]    = stream_input_spec{{"n_nationkey", "n_name"}, nation_types, {0}};
    receiver_spec.outputs      = {3, 4};
    receiver_spec.partitioning = sirius::op::partition_spec{{0, 1, 2}};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    query_window receiver_window(*sirius_ctx, *con->context, "frag9_receiver");
    receiver.build(receiver_window.query_id());

    // CN arrival order: all pushes, then all closes, then run().
    for (auto [sender, sender_id] :
         {std::pair{probe_a.get(), sender_id_t{0}}, std::pair{probe_b.get(), sender_id_t{1}}}) {
      std::size_t relayed = 0;
      while (auto batch = sender->session().pull(0)) {
        REQUIRE(receiver.session().push(0, *batch));
        ++relayed;
      }
      REQUIRE(relayed > 0);
      receiver.session().close_input(0, sender_id);
    }
    receiver.session().close_input(1, 0);  // n1: closed without ever carrying a batch
    {
      std::size_t relayed = 0;
      while (auto batch = n2->session().pull(0)) {
        REQUIRE(receiver.session().push(2, *batch));
        ++relayed;
      }
      REQUIRE(relayed > 0);
      receiver.session().close_input(2, 0);
    }

    try {
      receiver.run();
    } catch (...) {
      // Stall diagnostics: dump every pipeline's scheduling state before rethrowing.
      std::cerr << "==== FRAG-9 stall pipeline dump ====\n";
      for (const auto& pl : receiver.engine().sirius_pipelines) {
        if (!pl) { continue; }
        std::cerr << "pipeline " << pl->get_pipeline_id()
                  << " finished=" << pl->is_pipeline_finished()
                  << " created=" << pl->get_tasks_created()
                  << " completed=" << pl->get_tasks_completed();
        if (auto src = pl->get_source()) {
          std::cerr << " source=" << src->get_name()
                    << " src_ports_empty=" << src->all_ports_empty()
                    << " src_pipeline_finished=" << src->is_source_pipeline_finished();
        }
        std::cerr << " ops=[";
        for (auto& op_ref : pl->get_operators()) {
          std::cerr << op_ref.get().get_name() << " ";
        }
        std::cerr << "]";
        if (auto sink = pl->get_sink()) { std::cerr << " sink=" << sink->get_name(); }
        std::cerr << "\n";
      }
      std::cerr.flush();
      throw;
    }
    receiver_window.finish();

    REQUIRE(drain_row_count(receiver, 3) + drain_row_count(receiver, 4) == 0);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-10: declared stream cardinalities steer the hash-join build side.
// A stream source binds with no backing rows, so without a declared count
// DuckDB's optimizer sees cardinality 1 on every stream and picks build sides
// blind — on the 2-CN q07 it built on a multi-GB lineitem-derived stream while
// the 2-row nation stream probed (14.8s -> 164s at SF500). The receiver's CN
// already holds every input before build(), so it can declare exact counts;
// this pins that a declared tiny stream becomes the build side (children[1],
// DuckDB's convention) in BOTH directions, and that the undeclared path still
// plans and runs (backward compatibility).
// ============================================================================

namespace {

sirius::op::sirius_physical_operator* find_first_hash_join(
  sirius::op::sirius_physical_operator& node)
{
  if (node.type == sirius::op::SiriusPhysicalOperatorType::HASH_JOIN) { return &node; }
  for (auto& child : node.children) {
    if (auto* join = find_first_hash_join(*child)) { return join; }
  }
  return nullptr;
}

bool subtree_contains(const sirius::op::sirius_physical_operator& node,
                      const sirius::op::sirius_physical_operator* target)
{
  if (&node == target) { return true; }
  for (const auto& child : node.children) {
    if (subtree_contains(*child, target)) { return true; }
  }
  return false;
}

}  // namespace

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-10: declared stream cardinalities pick the hash-join build side",
                 "[integration][streaming_fragment][stream_cardinality]")
{
  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  constexpr const char* kJoinQuery =
    "SELECT l.a, r.b FROM sirius_stream_source(0) l JOIN sirius_stream_source(1) r ON l.a = r.b";

  // The declared counts alone drive the optimizer; the actual pushed volume (5 rows per side)
  // never reaches the planner. Both directions are covered so the assertion cannot pass by the
  // optimizer's accidental default order.
  std::optional<stream_id_t> small_stream;
  SECTION("stream 1 declared tiny -> stream 1 builds") { small_stream = 1; }
  SECTION("stream 0 declared tiny -> stream 0 builds") { small_stream = 0; }
  SECTION("no declared cardinality still plans and runs") { small_stream = std::nullopt; }

  watchdog_guard watchdog("20");

  con->BeginTransaction();
  try {
    auto make_sender = [&]() {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(kLeafQuery);
      spec.outputs     = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };

    auto int_types = [] {
      return sirius::from_duckdb_vec(
        duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
    };

    auto left  = make_sender();
    auto right = make_sender();
    for (auto* sender : {left.get(), right.get()}) {
      query_window sender_window(*sirius_ctx, *con->context, "frag10_sender");
      sender->build(sender_window.query_id());
      sender->run();
      sender_window.finish();
    }

    fragment_spec receiver_spec;
    receiver_spec.plan_source = sirius::test::sql_plan_source(kJoinQuery);
    receiver_spec.inputs[0]   = stream_input_spec{{"a"}, int_types(), {0}, std::nullopt};
    receiver_spec.inputs[1]   = stream_input_spec{{"b"}, int_types(), {0}, std::nullopt};
    if (small_stream.has_value()) {
      stream_id_t const big_stream                       = *small_stream == 1 ? 0 : 1;
      receiver_spec.inputs[*small_stream].estimated_rows = 2;
      receiver_spec.inputs[big_stream].estimated_rows    = 100'000;
    }
    receiver_spec.outputs = {2};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    query_window receiver_window(*sirius_ctx, *con->context, "frag10_receiver");
    receiver.build(receiver_window.query_id());

    // Plan-shape assertion, straight off the built physical tree: DuckDB's build side is the
    // join's second child, and it must be the subtree reading the stream declared tiny.
    auto* root = receiver.engine().sirius_physical_plan.get();
    REQUIRE(root != nullptr);
    auto* join = find_first_hash_join(*root);
    REQUIRE(join != nullptr);
    REQUIRE(join->children.size() == 2);
    if (small_stream.has_value()) {
      stream_id_t const big_stream = *small_stream == 1 ? 0 : 1;
      // catalog_for, not the fixture's member: the transparent path already registered a catalog
      // when the connection opened (RegisteredStateManager::Insert never overwrites), so the
      // fixture's own object is only a fallback and the fragment binds through the live one.
      auto live_catalog  = catalog_for(*con->context);
      auto* small_source = live_catalog->get(*small_stream).built;
      auto* big_source   = live_catalog->get(big_stream).built;
      REQUIRE(small_source != nullptr);
      REQUIRE(big_source != nullptr);
      // The declared counts must have reached the lowered sources verbatim...
      REQUIRE(small_source->estimated_cardinality == 2);
      REQUIRE(big_source->estimated_cardinality == 100'000);
      // ...and flipped the build side onto the tiny stream.
      REQUIRE(subtree_contains(*join->children[1], small_source));
      REQUIRE(subtree_contains(*join->children[0], big_source));
    } else {
      // The pre-fix blindness this feature exists for, pinned as documentation: with nothing
      // declared, every stream source estimates cardinality 1 and the optimizer cannot tell a
      // 2-row nation stream from a multi-GB lineitem stream.
      auto live_catalog = catalog_for(*con->context);
      REQUIRE(live_catalog->get(0).built->estimated_cardinality <= 1);
      REQUIRE(live_catalog->get(1).built->estimated_cardinality <= 1);
    }

    // The plan is not just well-shaped — it still runs and joins correctly. CN arrival order:
    // all pushes, then all closes, then run().
    for (auto [sender, stream] :
         {std::pair{left.get(), stream_id_t{0}}, std::pair{right.get(), stream_id_t{1}}}) {
      std::size_t relayed = 0;
      while (auto batch = sender->session().pull(0)) {
        REQUIRE(receiver.session().push(stream, *batch));
        ++relayed;
      }
      REQUIRE(relayed > 0);
      receiver.session().close_input(stream, 0);
    }

    receiver.run();
    receiver_window.finish();

    // Both sides carry 1..5, so the equi-join yields exactly the 5 matches whichever side built.
    REQUIRE(drain_row_count(receiver, 2) == kLeafRows);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}
