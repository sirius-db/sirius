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

#include <cudf/unary.hpp>

#include <catch.hpp>
#include <cucascade/data/data_batch.hpp>
#include <data/data_batch_utils.hpp>
#include <duckdb.hpp>
#include <duckdb/main/materialized_query_result.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

using Catch::Matchers::ContainsSubstring;

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

    // OnConnectionOpened already registered this connection's catalog, and registered_state's
    // Insert never overwrites a key, so a test must read that one rather than insert its own.
    catalog = catalog_for(*con->context);
  }

  std::unique_ptr<duckdb::Connection> con;
  duckdb::shared_ptr<stream_bind_catalog> catalog;
};

//! The execution window a FRAG-CONTROL engine must sit inside. RAII matters here: a `REQUIRE`
//! that fails inside a hand-bracketed window would leave the slot held and self-deadlock in the
//! test's `Rollback`, so the scope's destructor backstop is what lets a failing assertion fail.
//! streaming_fragment tests must not open one of these around build() or run(): run() opens
//! its own window, and build() takes the slot to generate the plan.
using query_window = duckdb::SiriusContext::StandaloneQueryScope;

//! Every INTEGER value sitting in an output stream, draining it. Row counts alone would not
//! catch a hop that corrupted, dropped or duplicated values.
std::vector<std::int32_t> drain_values(streaming_fragment& fragment, stream_id_t id)
{
  std::vector<std::int32_t> values;
  while (auto batch = fragment.pull(id)) {
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
  while (auto batch = fragment.pull(id)) {
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

  con->BeginTransaction();
  try {
    streaming_fragment fragment(*con->context, std::move(spec));
    fragment.build();
    fragment.run();

    REQUIRE(fragment.output_batch_count(0) > 0);
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
  auto expected = con->Query(std::string("SELECT count(*) FROM (") + kLeafQuery + ") t");
  REQUIRE_FALSE(expected->HasError());
  auto const expected_rows = expected->GetValue(0, 0).GetValue<std::int64_t>();

  con->BeginTransaction();
  try {
    fragment_spec sender_spec;
    sender_spec.plan_source = sirius::test::sql_plan_source(kLeafQuery);
    sender_spec.outputs     = {0};
    streaming_fragment sender(*con->context, std::move(sender_spec));

    fragment_spec receiver_spec;
    receiver_spec.plan_source =
      sirius::test::sql_plan_source("SELECT a FROM sirius_stream_source(0)");
    receiver_spec.inputs[0] = stream_input_spec{
      {"a"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      {0}};
    receiver_spec.outputs = {1};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    sender.build();
    sender.run();

    receiver.build();
    // The binding lives only for build(); a later fragment may declare the same id.
    REQUIRE_FALSE(catalog->contains(0));
    auto const relayed_batches = receiver.relay_from(sender, 0, 0, 0);
    REQUIRE(relayed_batches > 0);

    receiver.run();

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

  SECTION("partitioning on a result fragment")
  {
    fragment_spec spec;
    spec.plan_source  = source;
    spec.partitioning = sirius::op::partition_spec{{0}};
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

    con->BeginTransaction();
    streaming_fragment fragment(*con->context, std::move(spec));
    REQUIRE_THROWS_AS(fragment.build(), sirius::invalid_input_exception);
    REQUIRE_FALSE(catalog->contains(7));
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
  auto row_count_of = [&](const std::string& query) -> std::size_t {
    std::size_t rows = 0;
    auto sirius_ctx  = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
    query_window window(*sirius_ctx, *con->context, "frag_control");
    // execute() routes through task_creator::prepare_for_query, which requires
    // set_client_context to have already run for the engine's exact query id — that only
    // happens for the window's own id (begin_execution_window calls it), so the engine must be
    // built on window.query_id() rather than with_initialized_engine's default synthesized one.
    sirius::test::with_initialized_engine(
      *con,
      query,
      [&](sirius::sirius_engine& engine) {
        REQUIRE(engine.has_result_collector());
        engine.execute();
        auto result = engine.get_result();
        REQUIRE(result != nullptr);
        REQUIRE_FALSE(result->HasError());
        auto materialized =
          duckdb::unique_ptr_cast<duckdb::QueryResult, duckdb::MaterializedQueryResult>(
            std::move(result));
        rows = materialized->RowCount();
      },
      window.query_id());
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

    sender.build();
    sender.run();

    receiver.build();
    auto const relayed_batches = receiver.relay_from(sender, 0, 0, 0);
    REQUIRE(relayed_batches > 0);

    receiver.run();
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
      {0, 1}};
    receiver_spec.outputs = {1};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    for (auto* sender : {first.get(), second.get()}) {
      sender->build();
      sender->run();
    }

    receiver.build();
    std::size_t relayed_batches = 0;
    relayed_batches += receiver.relay_from(*first, 0, 0, 0);
    relayed_batches += receiver.relay_from(*second, 0, 0, 1);
    // Multi-batch premise: if only one batch arrives this degrades to FRAG-2.
    REQUIRE(relayed_batches > 1);

    receiver.run();
    REQUIRE(drain_values(receiver, 1) == std::vector<std::int32_t>{1, 2, 3, 4, 5, 6});

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-6: empty outputs are a RESULT_COLLECTOR terminal on the same streaming_fragment.
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-6: a result fragment materializes rows through take_result()",
                 "[integration][streaming_fragment]")
{
  fragment_spec spec;
  spec.plan_source = sirius::test::sql_plan_source(kLeafQuery);

  con->BeginTransaction();
  try {
    streaming_fragment fragment(*con->context, std::move(spec));
    fragment.build();
    fragment.run();

    auto result = fragment.take_result();
    REQUIRE(result != nullptr);
    REQUIRE_FALSE(result->HasError());
    auto materialized =
      duckdb::unique_ptr_cast<duckdb::QueryResult, duckdb::MaterializedQueryResult>(
        std::move(result));
    REQUIRE(materialized->RowCount() == kLeafRows);
    REQUIRE(materialized->names == duckdb::vector<std::string>{"col_0"});
    REQUIRE(materialized->types ==
            duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
    std::vector<std::int32_t> values;
    for (duckdb::idx_t row = 0; row < materialized->RowCount(); ++row) {
      values.push_back(materialized->GetValue(0, row).GetValue<std::int32_t>());
    }
    std::sort(values.begin(), values.end());
    REQUIRE(values == std::vector<std::int32_t>{1, 2, 3, 4, 5});

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-7: relay_from rejects a bad relay before any batch moves
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-7: relay_from checks its preconditions before moving data",
                 "[integration][streaming_fragment]")
{
  auto const integer_type =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});

  auto make_fragment = [&](duckdb::ClientContext& context,
                           const std::string& query,
                           std::vector<stream_id_t> outputs) {
    fragment_spec spec;
    spec.plan_source = sirius::test::sql_plan_source(query);
    spec.outputs     = std::move(outputs);
    return std::make_unique<streaming_fragment>(context, std::move(spec));
  };
  auto make_receiver = [&](stream_input_spec input) {
    fragment_spec spec;
    // SELECT *: a stream read cannot project a subset of its declared columns.
    spec.plan_source = sirius::test::sql_plan_source("SELECT * FROM sirius_stream_source(0)");
    spec.inputs[0]   = std::move(input);
    spec.outputs     = {1};
    return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
  };

  con->BeginTransaction();
  try {
    auto sender = make_fragment(*con->context, kLeafQuery, {0});
    sender->build();
    sender->run();

    // Each section throws and must leave the receiver's input open: the valid relay at the end
    // still delivers every row.
    auto require_receiver_still_open = [&](streaming_fragment& receiver) {
      REQUIRE(receiver.relay_from(*sender, 0, 0, 0) > 0);
      receiver.run();
      REQUIRE(drain_values(receiver, 1) == std::vector<std::int32_t>{1, 2, 3, 4, 5});
    };

    SECTION("source was never built")
    {
      auto unbuilt  = make_fragment(*con->context, kLeafQuery, {0});
      auto receiver = make_receiver({{"a"}, integer_type, {0}});
      receiver->build();
      REQUIRE_THROWS_AS(receiver->relay_from(*unbuilt, 0, 0, 0), sirius::invalid_input_exception);
      require_receiver_still_open(*receiver);
    }

    SECTION("source's run() failed")
    {
      // A pin change between build() and run() fails the run without depending on I/O timing.
      auto failed = make_fragment(*con->context, kLeafQuery, {0});
      failed->build();
      con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state")
        ->get_scan_manager()
        .bump_pin_registry_epoch_for_testing();
      REQUIRE_THROWS(failed->run());

      auto receiver = make_receiver({{"a"}, integer_type, {0}});
      receiver->build();
      REQUIRE_THROWS_WITH(receiver->relay_from(*failed, 0, 0, 0),
                          ContainsSubstring("source fragment's run() failed"));
      require_receiver_still_open(*receiver);
    }

    SECTION("source is a result fragment")
    {
      auto result = make_fragment(*con->context, kLeafQuery, {});
      result->build();
      result->run();
      auto receiver = make_receiver({{"a"}, integer_type, {0}});
      receiver->build();
      REQUIRE_THROWS_AS(receiver->relay_from(*result, 0, 0, 0), sirius::invalid_input_exception);
      require_receiver_still_open(*receiver);
    }

    SECTION("source is on another ClientContext")
    {
      auto other_con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
      other_con->BeginTransaction();
      auto foreign = make_fragment(*other_con->context, kLeafQuery, {0});
      foreign->build();
      foreign->run();
      other_con->Rollback();

      auto receiver = make_receiver({{"a"}, integer_type, {0}});
      receiver->build();
      REQUIRE_THROWS_AS(receiver->relay_from(*foreign, 0, 0, 0), sirius::invalid_input_exception);
      require_receiver_still_open(*receiver);
    }

    SECTION("input stream was never declared")
    {
      auto receiver = make_receiver({{"a"}, integer_type, {0}});
      receiver->build();
      REQUIRE_THROWS_AS(receiver->relay_from(*sender, 0, 9, 0), sirius::invalid_input_exception);
      require_receiver_still_open(*receiver);
    }

    SECTION("sender is not in the expected set")
    {
      auto receiver = make_receiver({{"a"}, integer_type, {0}});
      receiver->build();
      REQUIRE_THROWS_AS(receiver->relay_from(*sender, 0, 0, 5), sirius::invalid_input_exception);
      require_receiver_still_open(*receiver);
    }

    SECTION("column count differs from the declared input")
    {
      auto receiver = make_receiver({{"a", "b"},
                                     sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{
                                       duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER}),
                                     {0}});
      receiver->build();
      REQUIRE_THROWS_AS(receiver->relay_from(*sender, 0, 0, 0), sirius::invalid_input_exception);
    }

    SECTION("column type differs from the declared input")
    {
      auto receiver = make_receiver(
        {{"a"},
         sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT}),
         {0}});
      receiver->build();
      REQUIRE_THROWS_AS(receiver->relay_from(*sender, 0, 0, 0), sirius::invalid_input_exception);
    }

    SECTION("target has already run")
    {
      auto receiver = make_receiver({{"a"}, integer_type, {0}});
      receiver->build();
      receiver->close_input(0, 0);
      receiver->run();
      REQUIRE_THROWS_AS(receiver->relay_from(*sender, 0, 0, 0), sirius::invalid_input_exception);
    }

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-8: failure paths and single-use calls
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-8: failed and out-of-order calls throw and release the window",
                 "[integration][streaming_fragment]")
{
  auto make_fragment = [&](const std::string& query, std::vector<stream_id_t> outputs) {
    fragment_spec spec;
    spec.plan_source = sirius::test::sql_plan_source(query);
    spec.outputs     = std::move(outputs);
    return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
  };
  // A later fragment on the connection builds and runs, so the window was released.
  auto require_window_free = [&] {
    auto next = make_fragment(kLeafQuery, {0});
    next->build();
    next->run();
    REQUIRE(drain_values(*next, 0) == std::vector<std::int32_t>{1, 2, 3, 4, 5});
  };

  con->BeginTransaction();
  try {
    SECTION("run() twice")
    {
      auto fragment = make_fragment(kLeafQuery, {0});
      fragment->build();
      fragment->run();
      REQUIRE_THROWS_AS(fragment->run(), sirius::invalid_input_exception);
    }

    SECTION("pull() before run()")
    {
      auto fragment = make_fragment(kLeafQuery, {0});
      fragment->build();
      REQUIRE_THROWS_AS(fragment->pull(0), sirius::invalid_input_exception);
      fragment->run();
    }

    SECTION("take_result() on a streaming fragment")
    {
      auto fragment = make_fragment(kLeafQuery, {0});
      fragment->build();
      fragment->run();
      REQUIRE_THROWS_AS(fragment->take_result(), sirius::invalid_input_exception);
    }

    SECTION("take_result() twice, and output_batch_count() on a result fragment")
    {
      auto fragment = make_fragment(kLeafQuery, {});
      fragment->build();
      fragment->run();
      REQUIRE_THROWS_AS(fragment->output_batch_count(0), sirius::invalid_input_exception);
      REQUIRE(fragment->take_result() != nullptr);
      REQUIRE_THROWS_AS(fragment->take_result(), sirius::invalid_input_exception);
    }

    SECTION("prepared types that do not match the plan")
    {
      fragment_spec spec;
      spec.plan_source =
        [leaf = sirius::test::sql_plan_source(kLeafQuery)](duckdb::ClientContext& context) {
          auto bound     = leaf(context);
          bound.prepared = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
            duckdb::StatementType::SELECT_STATEMENT);
          bound.prepared->names = {"a", "b"};
          bound.prepared->types = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR};
          return bound;
        };
      streaming_fragment fragment(*con->context, std::move(spec));
      REQUIRE_THROWS_AS(fragment.build(), sirius::invalid_input_exception);
      // A failed build() is single-shot, rather than failing later on a half-registered session.
      REQUIRE_THROWS_WITH(fragment.build(), ContainsSubstring("cannot be retried"));
      require_window_free();
    }

    SECTION("a HUGEINT prepared column over a BIGINT plan column, as DuckDB types SUM(BIGINT)")
    {
      fragment_spec spec;
      spec.plan_source =
        [leaf = sirius::test::sql_plan_source("SELECT a::BIGINT FROM (VALUES (-7), (2)) t(a)")](
          duckdb::ClientContext& context) {
          auto bound     = leaf(context);
          bound.prepared = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
            duckdb::StatementType::SELECT_STATEMENT);
          bound.prepared->names = {"a"};
          bound.prepared->types = {duckdb::LogicalType::HUGEINT};
          return bound;
        };
      streaming_fragment fragment(*con->context, std::move(spec));
      fragment.build();
      fragment.run();
      auto result = duckdb::unique_ptr_cast<duckdb::QueryResult, duckdb::MaterializedQueryResult>(
        fragment.take_result());
      REQUIRE(result->types == duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::HUGEINT});
      REQUIRE(result->RowCount() == 2);
      // A sign-dropping widen would turn -7 into a value no int64 holds.
      REQUIRE(result->GetValue(0, 0).GetValue<std::int64_t>() +
                result->GetValue(0, 1).GetValue<std::int64_t>() ==
              -5);
    }

    SECTION("a failed run() poisons outputs, refuses a retry, and frees the window")
    {
      // A pin change between build() and run() fails the run without depending on I/O timing:
      // the scan cache may already hold a small file, so deleting it is not a reliable failure.
      auto fragment = make_fragment(kLeafQuery, {0});
      fragment->build();
      con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state")
        ->get_scan_manager()
        .bump_pin_registry_epoch_for_testing();

      std::string cause;
      try {
        fragment->run();
      } catch (std::exception const& e) {
        cause = e.what();
      }
      REQUIRE_FALSE(cause.empty());
      // The poisoned output surfaces the run's cause, not an empty or finished stream.
      REQUIRE_THROWS_WITH(fragment->pull(0), cause);
      REQUIRE_FALSE(fragment->drained(0));
      REQUIRE_THROWS_WITH(fragment->run(), ContainsSubstring("previous run() failed"));
      require_window_free();
    }

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-9: a hash-partitioned sink routes each key to one destination from every sender
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-9: hash-partitioned senders agree on each key's destination",
                 "[integration][streaming_fragment]")
{
  // Both senders emit the same INTEGER keys, so a key split across destinations means the
  // senders hashed it differently.
  std::string values = "SELECT a FROM (VALUES ";
  for (int k = 1; k <= 20; ++k) {
    values += (k > 1 ? ", (" : "(") + std::to_string(k) + ")";
  }
  values += ") t(a)";

  con->BeginTransaction();
  try {
    auto make_sender = [&] {
      fragment_spec spec;
      spec.plan_source  = sirius::test::sql_plan_source(values);
      spec.outputs      = {0, 1};
      spec.partitioning = sirius::op::partition_spec{{0}};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };
    auto first  = make_sender();
    auto second = make_sender();
    for (auto* sender : {first.get(), second.get()}) {
      sender->build();
      sender->run();
    }

    std::vector<std::vector<std::int32_t>> per_destination;
    for (stream_id_t destination : {0, 1}) {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source("SELECT * FROM sirius_stream_source(0)");
      spec.inputs[0]   = stream_input_spec{
          {"a"},
        sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
          {0, 1}};
      spec.outputs = {1};
      streaming_fragment receiver(*con->context, std::move(spec));
      receiver.build();
      receiver.relay_from(*first, destination, 0, 0);
      receiver.relay_from(*second, destination, 0, 1);
      receiver.run();
      per_destination.push_back(drain_values(receiver, 1));
    }

    std::vector<std::int32_t> all;
    for (const auto& received : per_destination) {
      REQUIRE_FALSE(received.empty());
      // Sorted, so each key's two copies are adjacent.
      REQUIRE(received.size() % 2 == 0);
      for (std::size_t i = 0; i < received.size(); i += 2) {
        REQUIRE(received[i] == received[i + 1]);
      }
      all.insert(all.end(), received.begin(), received.end());
    }
    std::sort(all.begin(), all.end());
    std::vector<std::int32_t> expected;
    for (int k = 1; k <= 20; ++k) {
      expected.push_back(k);
      expected.push_back(k);
    }
    REQUIRE(all == expected);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-9b: senders planned with different integer widths agree on each key's destination
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-9b: an INTEGER and a BIGINT sender route each key to the same destination",
                 "[integration][streaming_fragment]")
{
  // cuDF hashes raw bytes, so without the INT32 -> INT64 key cast the two senders would split
  // matching keys across destinations.
  std::string keys;
  for (int k = 1; k <= 20; ++k) {
    keys += (k > 1 ? ", (" : "(") + std::to_string(k) + ")";
  }

  con->BeginTransaction();
  try {
    auto make_sender = [&](const std::string& query) {
      fragment_spec spec;
      spec.plan_source  = sirius::test::sql_plan_source(query);
      spec.outputs      = {0, 1};
      spec.partitioning = sirius::op::partition_spec{{0}};
      auto sender       = std::make_unique<streaming_fragment>(*con->context, std::move(spec));
      sender->build();
      sender->run();
      return sender;
    };
    auto narrow = make_sender("SELECT a FROM (VALUES " + keys + ") t(a)");
    auto wide   = make_sender("SELECT a::BIGINT AS a FROM (VALUES " + keys + ") t(a)");
    REQUIRE(narrow->sink_types()[0].id() == sirius::type_id::INTEGER);
    REQUIRE(wide->sink_types()[0].id() == sirius::type_id::BIGINT);

    auto drain_keys = [](streaming_fragment& sender, stream_id_t destination) {
      std::vector<std::int64_t> values;
      while (auto batch = sender.pull(destination)) {
        auto wide = cudf::cast(sirius::get_cudf_table_view(**batch).column(0),
                               cudf::data_type{cudf::type_id::INT64});
        auto host = sirius::test::operator_utils::copy_column_to_host<std::int64_t>(*wide);
        values.insert(values.end(), host.begin(), host.end());
      }
      std::sort(values.begin(), values.end());
      return values;
    };

    std::size_t routed = 0;
    for (stream_id_t destination : {0, 1}) {
      auto const from_narrow = drain_keys(*narrow, destination);
      auto const from_wide   = drain_keys(*wide, destination);
      // Both destinations get keys, so matching sets are not just "everything went to one".
      REQUIRE_FALSE(from_narrow.empty());
      REQUIRE(from_narrow == from_wide);
      routed += from_narrow.size();
    }
    REQUIRE(routed == 20);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-10: build() holds no query window, so fragments build up front and run in any order
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-10: fragments built up front run in any order",
                 "[integration][streaming_fragment]")
{
  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  con->BeginTransaction();
  try {
    auto make_sender = [&](const char* query) {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(query);
      spec.outputs     = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };
    auto first  = make_sender("SELECT a FROM (VALUES (1), (2), (3)) t(a)");
    auto second = make_sender("SELECT a FROM (VALUES (4), (5), (6)) t(a)");

    fragment_spec receiver_spec;
    receiver_spec.plan_source =
      sirius::test::sql_plan_source("SELECT a FROM sirius_stream_source(0)");
    receiver_spec.inputs[0] = stream_input_spec{
      {"a"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      {0, 1}};
    receiver_spec.outputs = {1};
    streaming_fragment receiver(*con->context, std::move(receiver_spec));

    receiver.build();
    first->build();
    second->build();
    REQUIRE_FALSE(sirius_ctx->is_query_lifecycle_active());

    second->run();
    first->run();
    receiver.relay_from(*first, 0, 0, 0);
    receiver.relay_from(*second, 0, 0, 1);
    receiver.run();
    REQUIRE(drain_values(receiver, 1) == std::vector<std::int32_t>{1, 2, 3, 4, 5, 6});

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-11: other queries between a fragment's build() and run() leave its plan intact
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-11: another fragment and a transparent query between build and run",
                 "[integration][streaming_fragment]")
{
  auto const parquet = lineitem_parquet_path();
  REQUIRE(fs::exists(parquet));
  auto const scan =
    "SELECT l_orderkey FROM read_parquet('" + parquet.string() + "') WHERE l_quantity < 2";
  auto expected = con->Query("SELECT count(*) FROM (" + scan + ") t");
  REQUIRE_FALSE(expected->HasError());
  auto const expected_rows =
    static_cast<std::size_t>(expected->GetValue(0, 0).GetValue<std::int64_t>());
  REQUIRE(expected_rows > 0);

  con->BeginTransaction();
  try {
    auto make_scan = [&] {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(scan);
      spec.outputs     = {0};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };
    auto later = make_scan();
    later->build();

    // A full window in between: its cleanup resets the scan manager and task creator state.
    auto between = make_scan();
    between->build();
    between->run();
    REQUIRE(drain_row_count(*between, 0) == expected_rows);
    auto transparent = con->Query("SELECT count(*) FROM nation");
    REQUIRE_FALSE(transparent->HasError());

    later->run();
    REQUIRE(drain_row_count(*later, 0) == expected_rows);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-12: run() guards: a stale plan, an open input, and a window already held on this thread
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-12: run() rejects a stale plan and open inputs, build() a held window",
                 "[integration][streaming_fragment]")
{
  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);
  auto make_leaf = [&] {
    fragment_spec spec;
    spec.plan_source = sirius::test::sql_plan_source(kLeafQuery);
    spec.outputs     = {0};
    return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
  };

  con->BeginTransaction();
  try {
    SECTION("a pin change between build() and run() fails the run")
    {
      auto stale = make_leaf();
      stale->build();
      sirius_ctx->get_scan_manager().bump_pin_registry_epoch_for_testing();
      REQUIRE_THROWS_WITH(stale->run(), ContainsSubstring("pinned or unpinned"));
      REQUIRE_THROWS_WITH(stale->pull(0), ContainsSubstring("pinned or unpinned"));
      REQUIRE_THROWS_WITH(stale->run(), ContainsSubstring("previous run() failed"));

      auto next = make_leaf();
      next->build();
      next->run();
      REQUIRE(drain_values(*next, 0) == std::vector<std::int32_t>{1, 2, 3, 4, 5});
    }

    SECTION("run() with an input still open throws and leaves the fragment runnable")
    {
      auto sender = make_leaf();
      sender->build();
      sender->run();

      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source("SELECT a FROM sirius_stream_source(0)");
      spec.inputs[0]   = stream_input_spec{
          {"a"},
        sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
          {0, 1}};
      spec.outputs = {1};
      streaming_fragment receiver(*con->context, std::move(spec));
      receiver.build();
      receiver.relay_from(*sender, 0, 0, 0);

      REQUIRE_THROWS_WITH(receiver.run(), ContainsSubstring("still open"));
      receiver.close_input(0, 1);
      receiver.run();
      REQUIRE(drain_values(receiver, 1) == std::vector<std::int32_t>{1, 2, 3, 4, 5});
    }

    SECTION("build() inside a window already held on this thread throws")
    {
      auto fragment = make_leaf();
      {
        query_window window(*sirius_ctx, *con->context, "frag12_outer");
        REQUIRE_THROWS_WITH(fragment->build(),
                            ContainsSubstring("Nested Sirius execution windows"));
      }
      REQUIRE_THROWS_WITH(fragment->build(), ContainsSubstring("cannot be retried"));
    }

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}

// ============================================================================
// FRAG-13: runs on two connections from two threads serialize on the slot
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-13: fragments on two connections run from two threads",
                 "[integration][streaming_fragment]")
{
  auto other_con =
    std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());

  con->BeginTransaction();
  other_con->BeginTransaction();
  try {
    auto make_leaf = [&](duckdb::Connection& connection, const char* query) {
      fragment_spec spec;
      spec.plan_source = sirius::test::sql_plan_source(query);
      spec.outputs     = {0};
      return std::make_unique<streaming_fragment>(*connection.context, std::move(spec));
    };
    auto first  = make_leaf(*con, "SELECT a FROM (VALUES (1), (2), (3)) t(a)");
    auto second = make_leaf(*other_con, "SELECT a FROM (VALUES (4), (5), (6)) t(a)");
    first->build();
    second->build();

    std::exception_ptr first_error;
    std::exception_ptr second_error;
    std::thread first_run([&] {
      try {
        first->run();
      } catch (...) {
        first_error = std::current_exception();
      }
    });
    std::thread second_run([&] {
      try {
        second->run();
      } catch (...) {
        second_error = std::current_exception();
      }
    });
    first_run.join();
    second_run.join();
    if (first_error) { std::rethrow_exception(first_error); }
    if (second_error) { std::rethrow_exception(second_error); }

    REQUIRE(drain_values(*first, 0) == std::vector<std::int32_t>{1, 2, 3});
    REQUIRE(drain_values(*second, 0) == std::vector<std::int32_t>{4, 5, 6});

    other_con->Rollback();
    con->Rollback();
  } catch (...) {
    other_con->Rollback();
    con->Rollback();
    throw;
  }
}
