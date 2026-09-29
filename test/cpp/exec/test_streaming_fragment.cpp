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
#include <data/data_batch_utils.hpp>
#include <duckdb.hpp>
#include <duckdb/main/materialized_query_result.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <cstdint>
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

//! The execution window a FRAG-CONTROL engine must sit inside. RAII matters here: a `REQUIRE`
//! that fails inside a hand-bracketed window would leave the slot held and self-deadlock in the
//! test's `Rollback`, so the scope's destructor backstop is what lets a failing assertion fail.
//! streaming_fragment tests must not open one of these. streaming_fragment::build() owns the
//! window.
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
    // Only one fragment may sit between build() and run(), so every source runs before the
    // receiver is built.
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
      other_con->context->registered_state->Insert(stream_bind_catalog::kStateKey,
                                                   duckdb::make_shared_ptr<stream_bind_catalog>());
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
      spec.plan_source = sirius::test::sql_plan_source(kLeafQuery);
      spec.prepared    = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
        duckdb::StatementType::SELECT_STATEMENT);
      spec.prepared->names = {"a", "b"};
      spec.prepared->types = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR};
      streaming_fragment fragment(*con->context, std::move(spec));
      REQUIRE_THROWS_AS(fragment.build(), sirius::invalid_input_exception);
      // A failed build() is single-shot, rather than failing later on a half-registered session.
      REQUIRE_THROWS_WITH(fragment.build(), Catch::Contains("cannot be retried"));
      require_window_free();
    }

    SECTION("a failed run() poisons outputs, refuses a retry, and frees the window")
    {
      // The scan reads the file during run(), so deleting it after build() fails execution.
      auto const copy = fs::temp_directory_path() / "sirius_frag8_lineitem.parquet";
      fs::copy_file(lineitem_parquet_path(), copy, fs::copy_options::overwrite_existing);
      auto fragment =
        make_fragment("SELECT l_orderkey FROM read_parquet('" + copy.string() + "')", {0});
      fragment->build();
      fs::remove(copy);

      REQUIRE_THROWS(fragment->run());
      REQUIRE_FALSE(fragment->drained(0));
      REQUIRE_THROWS_WITH(fragment->run(), Catch::Contains("query window is closed"));
      require_window_free();
    }

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}
