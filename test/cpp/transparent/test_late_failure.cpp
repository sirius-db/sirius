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

#include "catch.hpp"
#include "io/io_errors.hpp"
#include "pipeline/completion_handler.hpp"
#include "sirius_context.hpp"
#include "transparent/replay_admission.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <duckdb.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/transaction/meta_transaction.hpp>

#include <barrier>
#include <future>
#include <thread>

using namespace sirius::transparent;

TEST_CASE("late failure keeps the terminal winner across racing reporters",
          "[transparent][late_failure]")
{
  for (int attempt = 0; attempt < 32; ++attempt) {
    sirius::pipeline::completion_handler completion;
    auto future = completion.get_awaitable();
    std::barrier start(3);
    auto report = [&](late_failure_cause cause, const char* message) {
      start.arrive_and_wait();
      completion.report_error(std::make_exception_ptr(classified_execution_error(cause, message)));
    };
    std::jthread first(report, late_failure_cause::physical_input, "physical");
    std::jthread second(report, late_failure_cause::oom_exhausted, "oom");
    start.arrive_and_wait();
    try {
      future.get();
      FAIL("expected terminal error");
    } catch (classified_execution_error const& error) {
      auto failure = completion.failure();
      CHECK(failure.cause == error.cause);
      CHECK(failure.detail == error.what());
    }
    first.join();
    second.join();
    auto winner = completion.failure();
    completion.report_error("later reporter");
    CHECK(completion.failure().cause == winner.cause);
    CHECK(completion.failure().detail == winner.detail);
  }
}

TEST_CASE("replay admission checks transactions cancellation and nonrollbackable state",
          "[transparent][late_failure]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& ctx = *con.context;
  CHECK(admit_cpu_replay(ctx, true, 1, false, false).refused_by ==
        late_failure_condition::transaction_invalid);
  REQUIRE_FALSE(con.Query("BEGIN TRANSACTION")->HasError());
  auto transaction = ctx.transaction.ActiveTransaction().global_transaction_id;
  CHECK(admit_cpu_replay(ctx, true, transaction, false, false).admitted);
  CHECK(admit_cpu_replay(ctx, false, transaction, false, false).admitted);
  CHECK(admit_cpu_replay(ctx, false, transaction, false, true).refused_by ==
        late_failure_condition::not_read_only);
  CHECK(admit_cpu_replay(ctx, true, transaction ^ 1, false, false).refused_by ==
        late_failure_condition::transaction_invalid);
  ctx.interrupted = true;
  CHECK(admit_cpu_replay(ctx, true, transaction, false, false).refused_by ==
        late_failure_condition::cancelled);
  ctx.interrupted = false;
  CHECK(admit_cpu_replay(ctx, true, transaction, true, false).refused_by ==
        late_failure_condition::non_rollbackable);
  ctx.transaction.ActiveTransaction().transaction_validity.Invalidate(
    "test transaction invalidation");
  CHECK(admit_cpu_replay(ctx, true, transaction, false, false).refused_by ==
        late_failure_condition::transaction_invalid);
  CHECK(admit_cpu_replay(ctx, true, std::nullopt, false, false).admitted);
  con.Query("ROLLBACK");
  REQUIRE_FALSE(con.Query("CREATE SEQUENCE late_seq")->HasError());
  auto bound = con.Prepare("SELECT nextval('late_seq')");
  REQUIRE_FALSE(bound->HasError());
  REQUIRE(bound->data);
  CHECK_FALSE(bound->data->properties.IsReadOnly());
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "late retry exhaustion uses common replay admission on both entrypoints",
                 "[integration][transparent][late_failure]")
{
  sirius::test::scratch_dir directory("late_retry");
  run_ok("SET gpu_execution=false");
  run_ok("COPY (SELECT i::INTEGER x FROM range(128) t(i)) TO " +
         directory.file_literal("data.parquet") + " (FORMAT PARQUET)");
  auto sql = "SELECT sum(x) FROM read_parquet(" + directory.file_literal("data.parquet") + ")";
  run_ok("SET gpu_execution=true");
  run_ok("SET sirius_test_gpu_task_retry_limit=1");
  run_ok("SET sirius_test_gpu_task_retry_backoff_ms=0");
  for (bool explicit_entry : {false, true}) {
    for (bool oom : {true, false}) {
      for (bool read_only : {true, false}) {
        INFO("explicit=" << explicit_entry << " oom=" << oom << " read_only=" << read_only);
        run_ok(std::string("SET sirius_test_override_read_only=") + (read_only ? "true" : "false"));
        run_ok(std::string("SET sirius_test_inject_gpu_task_oom=") + (oom ? "1000" : "0"));
        run_ok(std::string("SET sirius_test_inject_gpu_task_launch_error=") + (oom ? "0" : "1000"));
        auto before  = sirius::test::get_transparent_execution_stats(*con);
        auto started = std::chrono::steady_clock::now();
        auto result  = con->Query(
          explicit_entry ? "CALL gpu_execution(" + sirius::test::sql_literal(sql) + ")" : sql);
        if (result->HasError()) INFO(result->GetError());
        REQUIRE_FALSE(result->HasError());
        CHECK(result->GetValue(0, 0).ToString() == "8128");
        CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(1));
        auto after = sirius::test::get_transparent_execution_stats(*con);
        auto cause = static_cast<size_t>(oom ? late_failure_cause::oom_exhausted
                                             : late_failure_cause::retry_exhausted);
        CHECK(after.late_failures[cause] == before.late_failures[cause] + 1);
        CHECK(after.late_replays[cause] == before.late_replays[cause] + 1);
        CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
        CHECK(after.late_replay_not_read_only == before.late_replay_not_read_only + !read_only);
        CHECK(after.late_failure_no_replay == before.late_failure_no_replay);
        CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
      }
    }
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "late failure safety refusals and explicit fresh transaction exception",
                 "[integration][transparent][late_failure]")
{
  sirius::test::scratch_dir directory("late_refusal");
  run_ok("SET gpu_execution=false");
  run_ok("COPY (SELECT i::INTEGER x FROM range(128) t(i)) TO " +
         directory.file_literal("data.parquet") + " (FORMAT PARQUET)");
  auto sql = "SELECT sum(x) FROM read_parquet(" + directory.file_literal("data.parquet") + ")";
  run_ok("SET gpu_execution=true");
  run_ok("SET sirius_test_gpu_task_retry_limit=1");
  run_ok("SET sirius_test_gpu_task_retry_backoff_ms=0");
  run_ok("SET sirius_test_inject_gpu_task_oom=1000");
  for (bool explicit_entry : {false, true}) {
    for (auto [option, condition] : std::vector<std::pair<std::string, late_failure_condition>>{
           {"inject_transaction_mismatch", late_failure_condition::transaction_invalid},
           {"interrupt_before_replay", late_failure_condition::cancelled},
           {"inject_non_rollbackable_state", late_failure_condition::non_rollbackable}}) {
      INFO(option << " explicit=" << explicit_entry);
      run_ok("SET sirius_test_" + option + "=true");
      auto before = sirius::test::get_transparent_execution_stats(*con);
      auto result = con->Query(
        explicit_entry ? "CALL gpu_execution(" + sirius::test::sql_literal(sql) + ")" : sql);
      bool fresh_transaction =
        explicit_entry && condition == late_failure_condition::transaction_invalid;
      CHECK(result->HasError() != fresh_transaction);
      auto after = sirius::test::get_transparent_execution_stats(*con);
      CHECK(after.late_failures[static_cast<size_t>(late_failure_cause::oom_exhausted)] ==
            before.late_failures[static_cast<size_t>(late_failure_cause::oom_exhausted)] + 1);
      CHECK(after.late_failure_no_replay[static_cast<size_t>(condition)] ==
            before.late_failure_no_replay[static_cast<size_t>(condition)] + !fresh_transaction);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks + fresh_transaction);
      run_ok("SET sirius_test_" + option + "=false");
    }
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "late physical failure follows publication and discards only unconsumed batches",
                 "[integration][transparent][late_failure]")
{
  sirius::test::scratch_dir directory("late_physical");
  run_ok("SET gpu_execution=false");
  run_ok("COPY (SELECT i::INTEGER x FROM range(4096) t(i)) TO " +
         directory.file_literal("a.parquet") + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  run_ok("COPY (SELECT x::DOUBLE x FROM (VALUES (1.2),(2.0)) t(x)) TO " +
         directory.file_literal("b.parquet") + " (FORMAT PARQUET)");
  auto sql = "SELECT sum(x) FROM read_parquet(" + directory.file_literal("*.parquet") + ")";
  auto cpu = con->Query(sql);
  REQUIRE_FALSE(cpu->HasError());
  auto expected = cpu->GetValue(0, 0).ToString();
  run_ok("SET gpu_execution=true");
  run_ok("SET scan_task_batch_size=1");
  run_ok("SET sirius_test_hold_footer_index=2");
  auto state = sirius::test::get_registered_sirius_context(*con);
  for (bool hold_batch : {false, true}) {
    for (bool fallback : {true, false}) {
      run_ok(std::string("SET sirius_test_hold_published_batch=") +
             (hold_batch ? "true" : "false"));
      run_ok(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
      auto before = state->get_transparent_execution_stats();
      std::promise<void> footer_started;
      auto started = footer_started.get_future();
      std::atomic<bool> notified{false};
      auto counters                       = state->physical_counters();
      counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
        if (footer && !notified.exchange(true)) footer_started.set_value();
      };
      auto result    = std::async(std::launch::async, [&] { return con->Query(sql); });
      bool ready     = started.wait_for(std::chrono::seconds(20)) == std::future_status::ready;
      bool published = ready && state->get_scan_manager().wait_for_publication_for_testing(
                                  std::chrono::seconds(20));
      state->get_scan_manager().release_footer_hold_for_testing(2);
      auto rows                           = result.get();
      counters->parquet_phase_for_testing = {};
      REQUIRE(ready);
      REQUIRE(published);
      if (rows->HasError()) INFO(rows->GetError());
      CHECK(rows->HasError() == !fallback);
      if (fallback) CHECK(rows->GetValue(0, 0).ToString() == expected);
      auto after = state->get_transparent_execution_stats();
      auto cause = static_cast<size_t>(late_failure_cause::physical_input);
      CHECK(after.late_failures[cause] == before.late_failures[cause] + 1);
      CHECK(after.late_replays[cause] == before.late_replays[cause] + fallback);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks + fallback);
      if (hold_batch) CHECK(after.discarded_speculative_work > before.discarded_speculative_work);
      CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
    }
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "late prepare checkpoint failure retains its typed cause through ErrorData",
                 "[integration][transparent][late_failure]")
{
  run_ok("CREATE TABLE late_pin AS SELECT i::INTEGER x FROM range(128) t(i)");
  run_ok("CHECKPOINT");
  run_ok("CALL pin_table(format='duckdb', name='late_pin', tier='host')");
  struct unpin_guard {
    duckdb::Connection& con;
    ~unpin_guard() { con.Query("CALL unpin_table('late_pin')"); }
  } unpin{*con};
  run_ok("INSERT INTO late_pin VALUES (1000)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  for (bool fallback : {true, false}) {
    run_ok(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query("SELECT sum(x) FROM late_pin");
    if (result->HasError()) INFO(result->GetError());
    REQUIRE(result->HasError() == !fallback);
    if (fallback)
      CHECK(result->GetValue(0, 0).ToString() == "9128");
    else
      CHECK(result->GetError().find("checkpointed after pin_table") != std::string::npos);
    auto after = sirius::test::get_transparent_execution_stats(*con);
    auto cause = static_cast<size_t>(late_failure_cause::checkpoint_revalidation);
    CHECK(after.late_failures[cause] == before.late_failures[cause] + 1);
    CHECK(after.late_replays[cause] == before.late_replays[cause] + fallback);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks + fallback);
    CHECK(after.window_tasks_started == before.window_tasks_started);
    CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "explicit late replay opens a fresh snapshot after a concurrent commit",
                 "[integration][transparent][late_failure]")
{
  run_ok("CREATE TABLE late_view AS SELECT i::INTEGER x FROM range(128) t(i)");
  run_ok("CHECKPOINT");
  run_ok("CALL pin_table(format='duckdb', name='late_view', tier='host')");
  struct cleanup_pin {
    duckdb::Connection& con;
    ~cleanup_pin()
    {
      con.Query("ROLLBACK");
      con.Query("CALL unpin_table('late_view')");
    }
  } cleanup{*con};
  run_ok("SET gpu_execution=false");
  run_ok("BEGIN TRANSACTION");
  run_ok("SELECT sum(x) FROM late_view");  // Establish the caller's old snapshot.
  duckdb::Connection writer(*con->context->db);
  REQUIRE_FALSE(writer.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(
    writer.Query("INSERT INTO " + attach_alias + ".late_view VALUES (1000)")->HasError());
  run_ok("SET gpu_execution=true");
  run_ok("SET sirius_test_inject_gpu_task_oom=1000");
  run_ok("SET sirius_test_gpu_task_retry_limit=1");
  run_ok("SET sirius_test_gpu_task_retry_backoff_ms=0");
  auto query       = "SELECT sum(x) FROM " + attach_alias + ".late_view";
  auto before      = sirius::test::get_transparent_execution_stats(*con);
  auto transparent = con->Query(query);
  REQUIRE_FALSE(transparent->HasError());
  CHECK(transparent->GetValue(0, 0).ToString() == "8128");
  auto explicit_result = con->Query("CALL gpu_execution(" + sirius::test::sql_literal(query) + ")");
  if (explicit_result->HasError()) INFO(explicit_result->GetError());
  REQUIRE_FALSE(explicit_result->HasError());
  CHECK(explicit_result->GetValue(0, 0).ToString() == "9128");
  auto after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 2);
  CHECK(after.late_replays[static_cast<size_t>(late_failure_cause::oom_exhausted)] ==
        before.late_replays[static_cast<size_t>(late_failure_cause::oom_exhausted)] + 2);
  run_ok("ROLLBACK");
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "replay retry and rendezvous settings are latched once for each window",
                 "[integration][transparent][late_failure]")
{
  run_ok("SET sirius_test_gpu_task_retry_limit=3");
  run_ok("SET sirius_test_gpu_task_retry_backoff_ms=7");
  run_ok("SET sirius_test_hold_footer_index=2");
  run_ok("SET sirius_test_hold_published_batch=true");
  auto state  = sirius::test::get_registered_sirius_context(*con);
  auto before = state->get_transparent_execution_stats();
  {
    duckdb::SiriusContext::StandaloneQueryScope window(*state, *con->context, "latch_test");
    auto injections = window.injections();
    REQUIRE(injections);
    auto& config = duckdb::DBConfig::GetConfig(*con->context);
    duckdb::optional_ptr<const duckdb::ConfigurationOption> option;
    for (auto name : {"sirius_test_gpu_task_retry_limit",
                      "sirius_test_gpu_task_retry_backoff_ms",
                      "sirius_test_hold_footer_index"}) {
      auto index = config.TryGetSettingIndex(name, option);
      REQUIRE(index.IsValid());
      con->context->config.user_settings.SetUserSetting(index.GetIndex(),
                                                        duckdb::Value::UBIGINT(9));
    }
    CHECK(injections->gpu_task_retry_limit == 3);
    CHECK(injections->gpu_task_retry_backoff_ms == 7);
    CHECK(injections->hold_footer_index == 2);
    CHECK(injections->hold_published_batch);
    CHECK(window.injections() == injections);
    window.finish();
  }
  CHECK(state->get_transparent_execution_stats().setting_lookups_per_attempt ==
        before.setting_lookups_per_attempt + 10);
  {
    duckdb::SiriusContext::StandaloneQueryScope window(*state, *con->context, "next_latch_test");
    CHECK(window.injections()->gpu_task_retry_limit == 9);
    CHECK(window.injections()->gpu_task_retry_backoff_ms == 9);
    CHECK(window.injections()->hold_footer_index == 9);
    window.finish();
  }
  CHECK(state->get_transparent_execution_stats().setting_lookups_per_attempt ==
        before.setting_lookups_per_attempt + 20);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "late replay accounting includes the retained SQL text S3 veto",
                 "[integration][transparent][late_failure]")
{
  sirius::test::scratch_dir directory("late_text_veto");
  run_ok("SET gpu_execution=false");
  run_ok("COPY (SELECT i::INTEGER x FROM range(128) t(i)) TO " +
         directory.file_literal("data.parquet") + " (FORMAT PARQUET)");
  // DuckDB never binds this unused CTE. The bound policy sees only the local
  // scan; R1's retained SQL-text veto still conservatively forbids replay.
  auto sql =
    "WITH unused AS (SELECT * FROM read_parquet('s3://r2a-unread/file.parquet')) "
    "SELECT sum(x) FROM read_parquet(" +
    directory.file_literal("data.parquet") + ")";
  run_ok("SET gpu_execution=true");
  run_ok("SET sirius_test_inject_gpu_task_oom=1000");
  run_ok("SET sirius_test_gpu_task_retry_limit=1");
  run_ok("SET sirius_test_gpu_task_retry_backoff_ms=0");
  for (bool explicit_entry : {false, true}) {
    auto before = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::scoped_recording_log_sink logs;
    auto result = con->Query(
      explicit_entry ? "CALL gpu_execution(" + sirius::test::sql_literal(sql) + ")" : sql);
    REQUIRE(result->HasError());
    CHECK(result->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
    auto after = sirius::test::get_transparent_execution_stats(*con);
    auto cause = static_cast<size_t>(late_failure_cause::oom_exhausted);
    CHECK(after.late_failures[cause] == before.late_failures[cause] + 1);
    CHECK(after.late_replays == before.late_replays);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    size_t traces = 0;
    for (auto const& record : logs.records()) {
      if (record.message.starts_with("late failure cause=")) {
        ++traces;
        CHECK(record.message.find("replay=false") != std::string::npos);
        CHECK(record.message.size() < 128);
      }
    }
    CHECK(traces == 1);
  }
}

TEST_CASE("late failure classification uses exception types rather than message text",
          "[transparent][late_failure]")
{
  auto io = std::make_exception_ptr(duckdb::IOException("reader failed"));
  CHECK(classify_failure(io, late_failure_cause::gpu_error).cause == late_failure_cause::reader_io);
  auto credentials = std::make_exception_ptr(sirius::io::credential_error("signing failed"));
  CHECK(classify_failure(credentials, late_failure_cause::gpu_error).cause ==
        late_failure_cause::reader_io);
  auto text_only = std::make_exception_ptr(
    std::runtime_error("OOM physical input checkpoint certificate reader failed"));
  CHECK(classify_failure(text_only).cause == late_failure_cause::other);
  CHECK(classify_failure(text_only, late_failure_cause::gpu_error).cause ==
        late_failure_cause::gpu_error);
}
