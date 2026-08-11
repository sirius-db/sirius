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
#include <map>
#include <memory>
#include <string>
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
    // The whole point of change 2: without the completion gate this call never returns.
    fragment.run();
    window.finish();

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
    // so the window cleanup's clear_all_repositories() cannot touch it. The batches are still here.
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

    // Each fragment gets its own query window, spanning its build and run. The sender's
    // output repository is session-owned, so it survives the sender's window cleanup and is
    // still there for the relay.
    {
      query_window sender_window(*sirius_ctx, *con->context, "frag_sender");
      sender.build(sender_window.query_id());
      sender.run();
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
    {
      query_window window(*sirius_ctx, *con->context, "frag_3");
      REQUIRE_THROWS_AS(fragment.build(window.query_id()), sirius::invalid_input_exception);
    }
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

// ============================================================================
// FRAG-4: the demo's real shape -- a parquet GPU scan across a fragment
// boundary. VALUES is self-contained; a parquet scan exercises the actual
// path the StarRocks compute node will drive, at real batch counts.
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-4: a parquet scan crosses a fragment boundary",
                 "[integration][streaming_fragment]")
{
  auto const parquet = lineitem_parquet_path();
  REQUIRE(fs::exists(parquet));

  // A filtered projection, the shape TPC-H Q6 has: scan, filter, then exchange. The filter is on
  // l_quantity deliberately: the file's five row groups are ordered by l_orderkey, so an
  // l_orderkey range predicate prunes to a single row group and the scan would read a fraction
  // of the file. l_quantity's statistics span every row group, so all five are read.
  //
  // This is still a ONE-batch hop -- the GPU scan emits a single batch per file regardless of
  // row-group count. FRAG-5 is what covers a multi-batch stream.
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
// FRAG-5: a MULTI-BATCH stream drains completely.
//
// FRAG-2 and FRAG-4 both hop a single batch, which one task consumes in one
// go. Whether a queue holding several batches drains is a separate question:
// it is the task creator's per-batch loop and the pipeline's completion gate
// that have to agree, and neither is exercised by a one-batch stream. Two
// sender fragments fill the queue, because a GPU scan emits one batch per
// file and a VALUES leaf one batch per fragment -- the batch count comes from
// the number of senders, not from the leaf.
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-5: a multi-batch stream drains completely",
                 "[integration][streaming_fragment]")
{
  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  // Disjoint halves, so the receiver's output identifies which batches arrived, not merely how
  // many rows did.
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
    // The premise of this test. If a sender ever starts emitting one batch per fragment run
    // *and* the other stops emitting at all, the test would silently degrade to FRAG-2.
    REQUIRE(relayed_batches > 1);
    receiver.session().close_input(0, 0);

    receiver.run();
    receiver_window.finish();

    // Diagnostic, not a contract: the source hands out one batch per task, so a healthy drain of
    // N batches runs N tasks. Reported rather than asserted because coalescing batches into one
    // task is a live design option -- if that lands, this number changes and the value assertion
    // below is still the thing that must hold.
    std::size_t created = 0, completed = 0;
    for (const auto& p : receiver.engine().sirius_pipelines) {
      created += p->get_tasks_created();
      completed += p->get_tasks_completed();
    }
    INFO("relayed_batches=" << relayed_batches << " receiver tasks_created=" << created
                            << " tasks_completed=" << completed);

    // Every value from every batch: a receiver that ran one task and stopped would return the
    // first batch's rows and look like a plausible partial success.
    REQUIRE(drain_values(receiver, 1) == std::vector<std::int32_t>{1, 2, 3, 4, 5, 6});

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

namespace {

//! Every row in an output stream with all columns widened to int64, draining it. Two-phase
//! results mix BIGINT (sum/count) and INTEGER (min/max/keys) columns, so a single-type drain
//! cannot read them.
std::vector<std::vector<std::int64_t>> drain_rows_as_i64(streaming_fragment& fragment,
                                                         stream_id_t id)
{
  std::vector<std::vector<std::int64_t>> rows;
  while (auto batch = fragment.session().pull(id)) {
    auto view = sirius::get_cudf_table_view(**batch);
    std::vector<std::vector<std::int64_t>> cols;
    for (int c = 0; c < view.num_columns(); ++c) {
      const auto& col = view.column(c);
      std::vector<std::int64_t> host;
      switch (col.type().id()) {
        case cudf::type_id::INT32: {
          auto v = sirius::test::operator_utils::copy_column_to_host<std::int32_t>(col);
          host.assign(v.begin(), v.end());
          break;
        }
        case cudf::type_id::INT64: {
          host = sirius::test::operator_utils::copy_column_to_host<std::int64_t>(col);
          break;
        }
        // The scaled integer representation; assertions compare scaled values.
        case cudf::type_id::DECIMAL64: {
          host = sirius::test::operator_utils::copy_column_to_host<std::int64_t>(col);
          break;
        }
        default:
          FAIL("drain_rows_as_i64: unexpected column type id "
               << static_cast<int>(col.type().id()));
      }
      cols.push_back(std::move(host));
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(view.num_rows()); ++r) {
      std::vector<std::int64_t> row;
      row.reserve(cols.size());
      for (const auto& col : cols) {
        row.push_back(col[r]);
      }
      rows.push_back(std::move(row));
    }
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

}  // namespace

// ============================================================================
// FRAG-6: partial aggregates merge to the one-shot answer
// ============================================================================
//
// The two-phase aggregation design rests on two engine behaviours nothing else
// tests: an ungrouped aggregate under a STREAMING_SINK emits its single-row
// partial state, and a plain aggregate with substituted merge functions over
// those rows -- sum(s), sum(c), min(mn), max(mx) -- reproduces the one-shot
// answer. The substitution table is the one the engine applies internally
// between a local aggregate and its merge wrap (gpu_merge_impl.cpp), applied
// here across a fragment boundary instead.

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-6: partial aggregates merge to the one-shot answer",
                 "[integration][streaming_fragment]")
{
  constexpr const char* kPartials = "sum(a) AS s, count(*) AS c, min(a) AS mn, max(a) AS mx";
  const std::string first_leaf    = "(VALUES (1), (2), (3), (4), (5)) t(a)";
  std::string second_leaf         = "(VALUES (6), (7), (8), (9), (10)) t(a)";
  std::string oracle_leaf = "(VALUES (1), (2), (3), (4), (5), (6), (7), (8), (9), (10)) t(a)";

  // A compute node whose scan got no rows still participates in the merge; its partial state
  // (sum=NULL, count=0) must not corrupt the answer.
  SECTION("both senders contribute") {}
  SECTION("one sender has an empty input")
  {
    second_leaf = "(VALUES (6), (7), (8), (9), (10)) t(a) WHERE a > 100";
    oracle_leaf = "(VALUES (1), (2), (3), (4), (5)) t(a)";
  }

  auto expected = con->Query("SELECT sum(a), count(*), min(a), max(a) FROM " + oracle_leaf);
  REQUIRE_FALSE(expected->HasError());
  const std::vector<std::int64_t> oracle{expected->GetValue(0, 0).GetValue<std::int64_t>(),
                                         expected->GetValue(1, 0).GetValue<std::int64_t>(),
                                         expected->GetValue(2, 0).GetValue<std::int64_t>(),
                                         expected->GetValue(3, 0).GetValue<std::int64_t>()};

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  con->BeginTransaction();
  try {
    auto make_sender = [&](const std::string& leaf) {
      fragment_spec spec;
      spec.plan_source =
        sirius::test::sql_plan_source(std::string("SELECT ") + kPartials + " FROM " + leaf);
      spec.outputs = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };
    auto first  = make_sender(first_leaf);
    auto second = make_sender(second_leaf);

    for (auto* sender : {first.get(), second.get()}) {
      query_window sender_window(*sirius_ctx, *con->context, "frag6_sender");
      sender->build(sender_window.query_id());
      sender->run();
    }

    // Pins the partial-state wire types the translator's schema model has to predict:
    // sum(INTEGER) and count(*) are BIGINT (the plan generator downcasts DuckDB's HUGEINT sum),
    // min/max keep their input type. A change here is a change to that model.
    {
      const auto& types = first->sink_types();
      REQUIRE(types.size() == 4);
      INFO("sink types: " << types[0].to_string() << ", " << types[1].to_string() << ", "
                          << types[2].to_string() << ", " << types[3].to_string());
      CHECK(types[0].to_string() == "BIGINT");
      CHECK(types[1].to_string() == "BIGINT");
      CHECK(types[2].to_string() == "INTEGER");
      CHECK(types[3].to_string() == "INTEGER");
    }

    // The merge side: a plain aggregate with the substituted merge functions. count merges by
    // SUMMING partial counts -- merging it with count() would count rows and be silently wrong.
    fragment_spec receiver_spec;
    receiver_spec.plan_source = sirius::test::sql_plan_source(
      "SELECT sum(s), sum(c), min(mn), max(mx) FROM sirius_stream_source(0)");
    receiver_spec.inputs[0] = stream_input_spec{
      {"s", "c", "mn", "mx"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT,
                                                                  duckdb::LogicalType::BIGINT,
                                                                  duckdb::LogicalType::INTEGER,
                                                                  duckdb::LogicalType::INTEGER}),
      {0, 1}};
    receiver_spec.outputs = {1};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    query_window receiver_window(*sirius_ctx, *con->context, "frag6_receiver");
    receiver.build(receiver_window.query_id());

    std::uint32_t sender_id = 0;
    for (auto* sender : {first.get(), second.get()}) {
      std::size_t relayed = 0;
      while (auto batch = sender->session().pull(0)) {
        REQUIRE(receiver.session().push(0, *batch));
        ++relayed;
      }
      INFO("sender " << sender_id << " relayed " << relayed << " partial batches");
      receiver.session().close_input(0, sender_id);
      ++sender_id;
    }

    receiver.run();
    receiver_window.finish();

    auto rows = drain_rows_as_i64(receiver, 1);
    REQUIRE(rows.size() == 1);
    REQUIRE(rows[0] == oracle);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-7: grouped partial aggregates merge to the one-shot answer
// ============================================================================
//
// The distributed grouped path stays blocked until partitioned streaming
// output lands, but the merge *semantics* -- group keys carried in the partial
// rows, per-key substitution -- are pinned here so unblocking it later is a
// translator change, not an engine question.

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-7: grouped partial aggregates merge to the one-shot answer",
                 "[integration][streaming_fragment]")
{
  constexpr const char* kFirstLeaf  = "(VALUES (1), (2), (3), (4), (5)) t(a)";
  constexpr const char* kSecondLeaf = "(VALUES (6), (7), (8), (9), (10)) t(a)";
  constexpr const char* kOracleLeaf =
    "(VALUES (1), (2), (3), (4), (5), (6), (7), (8), (9), (10)) t(a)";

  auto expected = con->Query(std::string("SELECT a % 2 AS k, sum(a), count(*), min(a), max(a) "
                                         "FROM ") +
                             kOracleLeaf + " GROUP BY k ORDER BY k");
  REQUIRE_FALSE(expected->HasError());
  std::vector<std::vector<std::int64_t>> oracle;
  for (duckdb::idx_t r = 0; r < expected->RowCount(); ++r) {
    oracle.push_back({expected->GetValue(0, r).GetValue<std::int64_t>(),
                      expected->GetValue(1, r).GetValue<std::int64_t>(),
                      expected->GetValue(2, r).GetValue<std::int64_t>(),
                      expected->GetValue(3, r).GetValue<std::int64_t>(),
                      expected->GetValue(4, r).GetValue<std::int64_t>()});
  }
  std::sort(oracle.begin(), oracle.end());

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  con->BeginTransaction();
  try {
    auto make_sender = [&](const char* leaf) {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(
        std::string("SELECT a % 2 AS k, sum(a) AS s, count(*) AS c, min(a) AS mn, max(a) AS mx "
                    "FROM ") +
        leaf + " GROUP BY k");
      spec.outputs = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };
    auto first  = make_sender(kFirstLeaf);
    auto second = make_sender(kSecondLeaf);

    for (auto* sender : {first.get(), second.get()}) {
      query_window sender_window(*sirius_ctx, *con->context, "frag7_sender");
      sender->build(sender_window.query_id());
      sender->run();
    }

    {
      const auto& types = first->sink_types();
      REQUIRE(types.size() == 5);
      INFO("sink types: " << types[0].to_string() << ", " << types[1].to_string() << ", "
                          << types[2].to_string() << ", " << types[3].to_string() << ", "
                          << types[4].to_string());
      CHECK(types[0].to_string() == "INTEGER");
      CHECK(types[1].to_string() == "BIGINT");
      CHECK(types[2].to_string() == "BIGINT");
      CHECK(types[3].to_string() == "INTEGER");
      CHECK(types[4].to_string() == "INTEGER");
    }

    fragment_spec receiver_spec;
    receiver_spec.plan_source = sirius::test::sql_plan_source(
      "SELECT k, sum(s), sum(c), min(mn), max(mx) FROM sirius_stream_source(0) GROUP BY k");
    receiver_spec.inputs[0] = stream_input_spec{
      {"k", "s", "c", "mn", "mx"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER,
                                                                  duckdb::LogicalType::BIGINT,
                                                                  duckdb::LogicalType::BIGINT,
                                                                  duckdb::LogicalType::INTEGER,
                                                                  duckdb::LogicalType::INTEGER}),
      {0, 1}};
    receiver_spec.outputs = {1};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    query_window receiver_window(*sirius_ctx, *con->context, "frag7_receiver");
    receiver.build(receiver_window.query_id());

    std::uint32_t sender_id = 0;
    for (auto* sender : {first.get(), second.get()}) {
      while (auto batch = sender->session().pull(0)) {
        REQUIRE(receiver.session().push(0, *batch));
      }
      receiver.session().close_input(0, sender_id);
      ++sender_id;
    }

    receiver.run();
    receiver_window.finish();

    REQUIRE(drain_rows_as_i64(receiver, 1) == oracle);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-8: decimal min/max partial states cross the hop unchanged
// ============================================================================
//
// min/max keep their input type, so their partial-state wire type is the
// identity mapping -- but only if a DECIMAL column survives the GPU aggregate
// and the hop without being rewritten. sum over decimals is lowered to FP64
// upstream and never reaches this path.

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-8: decimal min/max partials survive the hop unchanged",
                 "[integration][streaming_fragment]")
{
  constexpr const char* kFirstLeaf =
    "(VALUES (CAST('1.10' AS DECIMAL(15,2))), (CAST('7.25' AS DECIMAL(15,2))), "
    "(CAST('3.50' AS DECIMAL(15,2)))) t(d)";
  constexpr const char* kSecondLeaf =
    "(VALUES (CAST('0.75' AS DECIMAL(15,2))), (CAST('9.99' AS DECIMAL(15,2)))) t(d)";

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  con->BeginTransaction();
  try {
    auto make_sender = [&](const char* leaf) {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(
        std::string("SELECT min(d) AS mn, max(d) AS mx FROM ") + leaf);
      spec.outputs = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };
    auto first  = make_sender(kFirstLeaf);
    auto second = make_sender(kSecondLeaf);

    for (auto* sender : {first.get(), second.get()}) {
      query_window sender_window(*sirius_ctx, *con->context, "frag8_sender");
      sender->build(sender_window.query_id());
      sender->run();
    }

    // The identity mapping the wire-type model relies on: the partial state is still
    // DECIMAL(15,2), not a widened or lowered stand-in.
    {
      const auto& types = first->sink_types();
      REQUIRE(types.size() == 2);
      INFO("sink types: " << types[0].to_string() << ", " << types[1].to_string());
      CHECK(types[0].to_string() == "DECIMAL(15,2)");
      CHECK(types[1].to_string() == "DECIMAL(15,2)");
    }

    fragment_spec receiver_spec;
    receiver_spec.plan_source =
      sirius::test::sql_plan_source("SELECT min(mn), max(mx) FROM sirius_stream_source(0)");
    receiver_spec.inputs[0] = stream_input_spec{
      {"mn", "mx"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{
        duckdb::LogicalType::DECIMAL(15, 2), duckdb::LogicalType::DECIMAL(15, 2)}),
      {0, 1}};
    receiver_spec.outputs = {1};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    query_window receiver_window(*sirius_ctx, *con->context, "frag8_receiver");
    receiver.build(receiver_window.query_id());

    std::uint32_t sender_id = 0;
    for (auto* sender : {first.get(), second.get()}) {
      while (auto batch = sender->session().pull(0)) {
        REQUIRE(receiver.session().push(0, *batch));
      }
      receiver.session().close_input(0, sender_id);
      ++sender_id;
    }

    receiver.run();
    receiver_window.finish();

    // Scaled-integer representation of DECIMAL(15,2): 0.75 -> 75, 9.99 -> 999.
    auto rows = drain_rows_as_i64(receiver, 1);
    REQUIRE(rows.size() == 1);
    REQUIRE(rows[0] == std::vector<std::int64_t>{75, 999});

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-9: a DECIMAL hash key routes equal values together, deterministically
// ============================================================================
//
// build() derives a FLOAT64 hash cast for DECIMAL keys. The two invariants that
// make that safe are pinned here: equal decimal values land on the SAME output
// stream — within one fragment and across two independently built ones, the
// cross-sender parity a distributed shuffle rests on — and the streams union to
// the whole input (a lossy cast may skew the split, never drop or misroute).

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-9: a DECIMAL hash key routes equal values together, deterministically",
                 "[integration][streaming_fragment]")
{
  // Every key value appears twice, so co-location of equal decimals is asserted directly
  // rather than inferred from key uniqueness.
  const std::vector<std::string> keys{
    "1.10", "7.25", "3.50", "0.75", "9.99", "12.00", "845.31", "2.00", "5.55", "100.10"};
  std::string leaf = "(VALUES ";
  bool first_value = true;
  for (const auto& key : keys) {
    for (int repeat = 0; repeat < 2; ++repeat) {
      if (!first_value) { leaf += ", "; }
      first_value = false;
      leaf += "(CAST('" + key + "' AS DECIMAL(15,2)))";
    }
  }
  leaf += ") t(d)";
  const std::size_t expected_rows = keys.size() * 2;

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  con->BeginTransaction();
  try {
    auto route_keys = [&](const std::string& label) {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source("SELECT d FROM " + leaf);
      spec.outputs     = {0, 1};
      // key_cast_types left empty: build() must derive the FLOAT64 cast for the DECIMAL key.
      spec.partitioning = sirius::op::partition_spec{{0}, {}};
      streaming_fragment fragment(*con->context, std::move(spec));
      {
        query_window window(*sirius_ctx, *con->context, label);
        fragment.build(window.query_id());
        fragment.run();
      }

      std::map<std::int64_t, stream_id_t> destination_of;
      std::size_t rows = 0;
      for (auto id : {stream_id_t{0}, stream_id_t{1}}) {
        while (auto batch = fragment.session().pull(id)) {
          auto view = sirius::get_cudf_table_view(**batch);
          // The partition output keeps the original schema; the FLOAT64 cast is transient.
          REQUIRE(view.column(0).type().id() == cudf::type_id::DECIMAL64);
          auto host =
            sirius::test::operator_utils::copy_column_to_host<std::int64_t>(view.column(0));
          rows += host.size();
          for (auto scaled : host) {
            auto [it, inserted] = destination_of.emplace(scaled, id);
            // One key on two streams would hand a downstream merge a partial group.
            REQUIRE(it->second == id);
          }
        }
      }
      // Nothing dropped, nothing duplicated: the streams union to the whole input.
      REQUIRE(rows == expected_rows);
      REQUIRE(destination_of.size() == keys.size());
      return destination_of;
    };

    auto first  = route_keys("frag9_first");
    auto second = route_keys("frag9_second");
    // Independently built fragments must agree on every key's destination.
    REQUIRE(first == second);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}
