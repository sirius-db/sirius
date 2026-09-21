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
#include "helper/logical_type.hpp"
#include "helper/type_conversions.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "sirius_engine.hpp"

#include <catch.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <data/data_batch_utils.hpp>
#include <duckdb.hpp>
#include <utils/parquet_fixture_utils.hpp>
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

//! region → SUM(amount) from a gather stream of VARCHAR + BIGINT.
std::map<std::string, std::int64_t> drain_groups(streaming_fragment& fragment, stream_id_t id)
{
  std::map<std::string, std::int64_t> groups;
  while (auto batch = fragment.session().pull(id)) {
    auto view = sirius::get_cudf_table_view(**batch);
    REQUIRE(view.num_columns() == 2);
    auto regions = sirius::test::operator_utils::copy_column_to_host<std::string>(view.column(0));
    auto amounts = sirius::test::operator_utils::copy_column_to_host<std::int64_t>(view.column(1));
    REQUIRE(regions.size() == amounts.size());
    for (std::size_t i = 0; i < regions.size(); ++i) {
      groups[regions[i]] += amounts[i];
    }
  }
  return groups;
}

void merge_disjoint_groups(std::map<std::string, std::int64_t>& dst,
                           std::map<std::string, std::int64_t> src)
{
  for (auto& [region, amount] : src) {
    // Hash partitioning sends each key to one destination; a split would mean the
    // sink hashed the same region onto both roots.
    INFO("region=" << region);
    REQUIRE(dst.count(region) == 0);
    dst[region] = amount;
  }
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
    // execute() routes through task_creator::prepare_for_query, which requires
    // set_client_context to have already run for the engine's exact query id — that only
    // happens for the window's own id (begin_execution_window calls it), so the engine must be
    // built on window.query_id() rather than with_initialized_engine's default synthesized one.
    auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
    query_window window(*sirius_ctx, *con->context, "frag_control");
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
// FRAG-6: two-shard GROUP BY (partial SUM → hash N=2 → merge SUM). Same process,
// native pull/push, sender-set EOS. No FFI, no Arrow — the GPU seam the CN will
// later hop with Arrow.
// ============================================================================

TEST_CASE_METHOD(fragment_fixture,
                 "FRAG-6: two-shard GROUP BY via hash sink and sender-set EOS",
                 "[integration][streaming_fragment]")
{
  sirius::test::scratch_dir sales("frag6_sales");
  auto const sales_0 = sales.file("sales_0.parquet");
  auto const sales_1 = sales.file("sales_1.parquet");

  {
    auto off = con->Query("SET gpu_execution = false");
    REQUIRE(off);
    REQUIRE_FALSE(off->HasError());

    auto write = [&](const std::string& path, const char* values_sql) {
      auto copied = con->Query(std::string("COPY (SELECT * FROM (VALUES ") + values_sql +
                               ") t(region, amount)) TO " + sirius::test::sql_literal(path) +
                               " (FORMAT PARQUET)");
      REQUIRE(copied);
      REQUIRE_FALSE(copied->HasError());
    };
    // Distinct keys so a 2-way hash is likely to populate both destinations.
    write(sales_0,
          "('east'::VARCHAR, 10::BIGINT), ('west'::VARCHAR, 20::BIGINT), "
          "('east'::VARCHAR, 5::BIGINT), ('north'::VARCHAR, 3::BIGINT), "
          "('south'::VARCHAR, 8::BIGINT)");
    write(sales_1,
          "('west'::VARCHAR, 7::BIGINT), ('east'::VARCHAR, 3::BIGINT), "
          "('north'::VARCHAR, 11::BIGINT), ('south'::VARCHAR, 1::BIGINT), "
          "('midwest'::VARCHAR, 4::BIGINT)");
  }

  auto expected_result = con->Query(
    "SELECT region, CAST(SUM(amount) AS BIGINT) FROM ("
    "SELECT * FROM read_parquet(" +
    sirius::test::sql_literal(sales_0) + ") UNION ALL SELECT * FROM read_parquet(" +
    sirius::test::sql_literal(sales_1) + ")) t GROUP BY region");
  REQUIRE(expected_result);
  REQUIRE_FALSE(expected_result->HasError());
  std::map<std::string, std::int64_t> expected;
  for (std::size_t i = 0; i < expected_result->RowCount(); ++i) {
    expected[expected_result->GetValue(0, i).ToString()] =
      expected_result->GetValue(1, i).GetValue<std::int64_t>();
  }
  REQUIRE(expected == (std::map<std::string, std::int64_t>{
                        {"east", 18}, {"west", 27}, {"north", 14}, {"south", 9}, {"midwest", 4}}));

  auto sirius_ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  con->BeginTransaction();
  try {
    auto make_leaf = [&](const std::string& path) {
      fragment_spec spec;
      spec.plan_source =
        sirius::test::sql_plan_source("SELECT region, SUM(amount) FROM read_parquet(" +
                                      sirius::test::sql_literal(path) + ") GROUP BY region");
      spec.outputs      = {0, 1};
      spec.partitioning = sirius::op::partition_spec{{0}};
      return std::make_unique<streaming_fragment>(*con->context, std::move(spec));
    };

    auto leaf0 = make_leaf(sales_0);
    auto leaf1 = make_leaf(sales_1);

    for (auto* leaf : {leaf0.get(), leaf1.get()}) {
      query_window window(*sirius_ctx, *con->context, "frag6_leaf");
      leaf->build(window.query_id());
      leaf->run();
      window.finish();
    }

    REQUIRE(leaf0->sink_types().size() == 2);
    REQUIRE(leaf0->sink_types()[0].id() == sirius::type_id::VARCHAR);
    REQUIRE(leaf0->sink_types()[1].id() == sirius::type_id::BIGINT);
    REQUIRE(leaf1->sink_types() == leaf0->sink_types());

    // Stream names match the root SQL (`SUM(amount)`); types come from the leaf sink
    // so the hop cannot silently widen/narrow.
    duckdb::vector<sirius::logical_type> hop_types = leaf0->sink_types();

    std::map<std::string, std::int64_t> got;
    for (stream_id_t dest : {stream_id_t{0}, stream_id_t{1}}) {
      fragment_spec root_spec;
      root_spec.plan_source = sirius::test::sql_plan_source(
        "SELECT region, SUM(amount) FROM sirius_stream_source(0) GROUP BY region");
      root_spec.inputs[0] = stream_input_spec{{"region", "amount"}, hop_types, {0, 1}};
      root_spec.outputs   = {1};
      streaming_fragment root(*con->context, std::move(root_spec));

      query_window window(*sirius_ctx, *con->context, "frag6_root");
      root.build(window.query_id());

      sender_id_t sender = 0;
      for (auto* leaf : {leaf0.get(), leaf1.get()}) {
        while (auto batch = leaf->session().pull(dest)) {
          REQUIRE(root.session().push(0, *batch));
        }
        root.session().close_input(0, sender);
        ++sender;
      }

      root.run();
      window.finish();

      auto part = drain_groups(root, 1);
      INFO("dest=" << dest << " groups=" << part.size());
      merge_disjoint_groups(got, std::move(part));
    }

    REQUIRE(got == expected);

    con->Rollback();
  } catch (...) {
    con->Rollback();
    throw;
  }
}
