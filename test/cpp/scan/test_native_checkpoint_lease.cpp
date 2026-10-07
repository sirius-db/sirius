/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/catalog/catalog_entry/schema_catalog_entry.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/storage/single_file_block_manager.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <fcntl.h>
#include <op/scan/duckdb_native_metadata_cache.hpp>
#include <op/scan/iceberg_metadata_connection.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/metadata_walk_parallel.hpp>
#include <signal.h>
#include <sirius_context.hpp>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utils/child_process_environment.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/isolated_checkpoint_test.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

struct NativeLeaseFixture : sirius::test::GpuExecutionFixture {
  ~NativeLeaseFixture()
  {
    auto state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    if (state) {
      CHECK(state->get_transparent_execution_stats().checkpoint_revalidation_failures == 0);
    }
  }
};
using namespace std::chrono_literals;

namespace {

constexpr char const* kUnavailableChildCase  = "native checkpoint never-entered child runner";
constexpr char const* kUnavailableVariantEnv = "SIRIUS_NATIVE_LEASE_UNAVAILABLE_VARIANT";

void query_ok(duckdb::Connection& connection, std::string const& sql)
{
  auto result = connection.Query(sql);
  REQUIRE(result);
  INFO("query failed: " << sql << "\n" << (result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

std::string scalar_value(duckdb::QueryResult& result)
{
  auto chunk = result.Fetch();
  REQUIRE(chunk);
  REQUIRE(chunk->size() == 1);
  return chunk->GetValue(0, 0).ToString();
}

std::string scalar_or_error(duckdb::unique_ptr<duckdb::MaterializedQueryResult> result)
{
  if (!result) { return "null query result"; }
  if (result->HasError()) { return result->GetError(); }
  if (result->RowCount() != 1) {
    return "query returned " + std::to_string(result->RowCount()) + " rows instead of one";
  }
  return result->GetValue(0, 0).ToString();
}

std::unique_ptr<duckdb::Connection> sibling_connection(NativeLeaseFixture& fixture)
{
  auto result = std::make_unique<duckdb::Connection>(*fixture.con->context->db);
  query_ok(*result, "USE " + fixture.attach_alias);
  return result;
}

void prepare_native_table(NativeLeaseFixture& fixture, bool sync_observer = true)
{
  query_ok(*fixture.con, "RESET disabled_optimizers");
  query_ok(*fixture.con, "SET gpu_execution = true");
  query_ok(*fixture.con, "SET enable_duckdb_fallback = true");
  query_ok(*fixture.con, "SET sirius_test_inject_read_view_mismatch = 'off'");
  query_ok(*fixture.con, "SET sirius_test_inject_pin_registry_change = false");
  query_ok(*fixture.con, "SET sirius_test_mark_runtime_unavailable_before_window = false");
  query_ok(*fixture.con, "SET sirius_test_pause_native_after_prepare_ms = 0");
  query_ok(*fixture.con, "SET sirius_test_pause_native_decode_ms = 0");
  if (sync_observer) { query_ok(*fixture.con, "SET sirius_test_sync_native_checkpoint = true"); }
  query_ok(*fixture.con, "SET sirius_test_sync_cpu_replay = false");
  query_ok(*fixture.con, "SET sirius_test_inject_checkpoint_cleanup_failure = false");
  query_ok(*fixture.con, "SET sirius_test_inject_native_walk_failure = ''");
  query_ok(*fixture.con, "SET sirius_test_inject_native_decode_failure = ''");
  query_ok(
    *fixture.con,
    "CREATE OR REPLACE TABLE native_lease_t AS SELECT range::BIGINT AS i FROM range(300000)");
  query_ok(*fixture.con, "CHECKPOINT");
  // SiriusContext is registered lazily by the first intercepted statement in a standalone
  // filtered test run; establish it before the concurrency assertions inspect the scan manager.
  query_ok(*fixture.con, "SELECT count(*) FROM native_lease_t");
}

bool wait_for_key(duckdb::SiriusContext& context, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (context.get_scan_manager().holds_any_checkpoint_key()) { return true; }
    std::this_thread::sleep_for(5ms);
  }
  return false;
}

bool wait_for_key_count(duckdb::SiriusContext& context,
                        std::size_t expected,
                        std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (context.get_scan_manager().checkpoint_key_count() == expected) { return true; }
    std::this_thread::sleep_for(5ms);
  }
  return false;
}

bool wait_for_no_key(duckdb::SiriusContext& context, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (!context.get_scan_manager().holds_any_checkpoint_key()) { return true; }
    std::this_thread::sleep_for(5ms);
  }
  return false;
}

struct phase_gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool arrived       = false;
  bool released      = false;
  uint64_t iteration = 0;
  void stop(uint64_t value = 0)
  {
    std::unique_lock lock(mutex);
    iteration = value;
    arrived   = true;
    cv.notify_all();
    if (!cv.wait_for(lock, 20s, [&] { return released; }))
      throw std::runtime_error("native checkpoint phase rendezvous timed out");
  }
  bool wait()
  {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 10s, [&] { return arrived; });
  }
  void release()
  {
    std::lock_guard lock(mutex);
    released = true;
    cv.notify_all();
  }
  ~phase_gate() { release(); }
};

struct scoped_native_observer {
  duckdb::SiriusContext& context;
  explicit scoped_native_observer(
    duckdb::SiriusContext& value,
    std::function<void(duckdb::ClientContext&, std::string_view, uint64_t)> hook)
    : context(value)
  {
    context.native_checkpoint_hook_for_testing = std::move(hook);
  }
  ~scoped_native_observer() { context.native_checkpoint_hook_for_testing = {}; }
};

struct walk_threads_guard {
  std::optional<std::string> previous;
  walk_threads_guard()
  {
    if (auto const* value = std::getenv("SIRIUS_METADATA_WALK_THREADS")) { previous = value; }
    setenv("SIRIUS_METADATA_WALK_THREADS", "2", 1);
  }
  ~walk_threads_guard()
  {
    if (previous) {
      setenv("SIRIUS_METADATA_WALK_THREADS", previous->c_str(), 1);
    } else {
      unsetenv("SIRIUS_METADATA_WALK_THREADS");
    }
  }
};

void check_metadata_walk_lease(bool cache_hit)
{
  NativeLeaseFixture fixture;
  prepare_native_table(fixture);
  auto& con    = *fixture.con;
  auto sibling = sibling_connection(fixture);
  auto context = sirius::test::get_registered_sirius_context(con);
  auto& cache  = sirius::op::scan::duckdb_native_metadata_cache::instance();
  walk_threads_guard threads;
  cache.clear();
  auto const sql = "SELECT sum(i) FROM native_lease_t WHERE i >= 100000";
  if (cache_hit) { query_ok(con, sql); }

  query_ok(con, "BEGIN TRANSACTION READ ONLY");
  auto& catalog = duckdb::Catalog::GetCatalog(*con.context, fixture.attach_alias);
  auto& table = catalog.GetEntry<duckdb::TableCatalogEntry>(*con.context, "main", "native_lease_t")
                  .Cast<duckdb::DuckTableEntry>();
  auto& blocks = dynamic_cast<duckdb::SingleFileBlockManager&>(
    table.GetStorage().GetAttached().GetStorageManager().GetBlockManager());
  auto const iteration = blocks.GetCheckpointIteration();
  query_ok(con, "COMMIT");

  auto const hits           = cache.hits();
  auto const products       = cache.product_hits();
  auto const rebuilds       = cache.rebuilds();
  auto const parallel_walks = sirius::op::scan::parallel_metadata_walks_for_testing.load();
  auto const before         = context->get_transparent_execution_stats();
  phase_gate prepared;
  scoped_native_observer observer(*context, [&](auto&, auto phase, uint64_t value) {
    if (phase == "native_prepared") { prepared.stop(value); }
  });
  auto query = std::async(std::launch::async, [&] { return scalar_or_error(con.Query(sql)); });
  REQUIRE(prepared.wait());
  CHECK(context->get_scan_manager().checkpoint_key_count() == 1);
  CHECK(prepared.iteration == iteration);
  CHECK(blocks.GetCheckpointIteration() == iteration);
  if (cache_hit) {
    CHECK(cache.hits() == hits + 1);
    CHECK(cache.product_hits() == products + 1);
    CHECK(cache.rebuilds() == rebuilds);
    CHECK(sirius::op::scan::parallel_metadata_walks_for_testing.load() == parallel_walks);
  } else {
    CHECK(cache.hits() == hits);
    CHECK(cache.product_hits() == products);
    CHECK(cache.rebuilds() == rebuilds + 1);
    CHECK(sirius::op::scan::parallel_metadata_walks_for_testing.load() > parallel_walks);
  }
  auto checkpoint = sibling->Query("CHECKPOINT");
  REQUIRE(checkpoint);
  REQUIRE(checkpoint->HasError());
  CHECK(checkpoint->GetErrorType() == duckdb::ExceptionType::TRANSACTION);
  auto forced = std::async(std::launch::async, [&] {
    auto result = sibling->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(forced.wait_for(150ms) == std::future_status::timeout);
  CHECK(blocks.GetCheckpointIteration() == iteration);
  prepared.release();
  REQUIRE(query.wait_for(10s) == std::future_status::ready);
  CHECK(query.get() == "39999900000");
  REQUIRE(forced.wait_for(10s) == std::future_status::ready);
  CHECK(forced.get().empty());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());

  query_ok(*sibling, "INSERT INTO native_lease_t VALUES (300000)");
  query_ok(*sibling, "CHECKPOINT");
  auto const next_iteration = blocks.GetCheckpointIteration();
  CHECK(next_iteration > iteration);
  CHECK(scalar_or_error(con.Query(sql)) == "40000200000");
  CHECK(prepared.iteration == next_iteration);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  auto const after = context->get_transparent_execution_stats();
  CHECK(after.successful_rebinds == before.successful_rebinds + 2);
  CHECK(after.executions == before.executions + 2);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(after.checkpoint_revalidation_failures == before.checkpoint_revalidation_failures);
}

}  // namespace

TEST_CASE("native checkpoint observer requires its session test option",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  prepare_native_table(fixture, false);
  auto context = sirius::test::get_registered_sirius_context(*fixture.con);
  std::vector<std::string> phases;
  scoped_native_observer observer(*context,
                                  [&](auto&, auto phase, uint64_t) { phases.emplace_back(phase); });
  for (auto const* phase :
       {"native_leaf", "before_window", "native_prepared", "native_walk_failed", "pin_prepared"}) {
    context->observe_native_checkpoint_for_testing(*fixture.con->context, phase, 42);
  }
  REQUIRE(phases.empty());

  query_ok(*fixture.con, "SET sirius_test_sync_native_checkpoint = true");
  context->observe_native_checkpoint_for_testing(*fixture.con->context, "native_walk_failed");
  CHECK(phases == std::vector<std::string>{"native_walk_failed"});
  phases.clear();
  query_ok(*fixture.con, "SELECT count(*) FROM native_lease_t");
  CHECK(std::find(phases.begin(), phases.end(), "native_leaf") != phases.end());
  CHECK(std::find(phases.begin(), phases.end(), "before_window") != phases.end());
  CHECK(std::find(phases.begin(), phases.end(), "native_prepared") != phases.end());

  phases.clear();
  auto sibling = sibling_connection(fixture);
  context->observe_native_checkpoint_for_testing(*sibling->context, "native_prepared");
  CHECK(phases.empty());
  query_ok(*fixture.con, "SET sirius_test_sync_native_checkpoint = false");
  query_ok(*fixture.con, "SELECT count(*) FROM native_lease_t");
  CHECK(phases.empty());
}

TEST_CASE(kUnavailableChildCase, "[.][scan][native][checkpoint][integration]")
{
  auto const* variant = std::getenv(kUnavailableVariantEnv);
  if (variant == nullptr) { return; }

  NativeLeaseFixture fixture;
  prepare_native_table(fixture);
  auto context      = sirius::test::get_registered_sirius_context(*fixture.con);
  auto const before = context->get_transparent_execution_stats();
  if (std::string_view{variant}.starts_with("cleanup_")) {
    query_ok(*fixture.con, "SET sirius_test_inject_checkpoint_cleanup_failure = true");
    if (std::string_view{variant} == "cleanup_state") {
      query_ok(*fixture.con, "BEGIN TRANSACTION READ ONLY");
      auto& catalog = duckdb::Catalog::GetCatalog(*fixture.con->context, fixture.attach_alias);
      auto& table =
        catalog.GetEntry<duckdb::TableCatalogEntry>(*fixture.con->context, "main", "native_lease_t")
          .Cast<duckdb::DuckTableEntry>();
      {
        duckdb::SiriusContext::StandaloneQueryScope window(
          *context, *fixture.con->context, "cleanup_failure");
        context->get_scan_manager().acquire_checkpoint_key(window.query_id(),
                                                           table.GetStorage().GetAttached());
        CHECK_THROWS_WITH(
          window.finish(),
          Catch::Matchers::ContainsSubstring("injected checkpoint cleanup failure"));
        CHECK(window.lease_release().state ==
              duckdb::SiriusContext::StandaloneQueryScope::lease_release_state::cleanup_failed);
        CHECK(window.lease_release().keys_released == 0);
      }
      CHECK(context->get_scan_manager().checkpoint_key_count() == 1);
    } else {
      auto sql = "SELECT count(*) FROM " + fixture.attach_alias + ".main.native_lease_t";
      if (std::string_view{variant} == "cleanup_explicit")
        sql = "SELECT * FROM gpu_execution('" + sql + "')";
      auto result = fixture.con->Query(sql);
      REQUIRE(result);
      REQUIRE(result->HasError());
      CHECK(result->GetError().find("injected checkpoint cleanup failure") != std::string::npos);
      CHECK(context->get_scan_manager().checkpoint_key_count() > 0);
    }
    CHECK(context->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
    CHECK(context->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
    CHECK(context->get_transparent_execution_stats().lease_held_at_replay ==
          before.lease_held_at_replay);
    // This injection is after all worker drains. Explicit teardown is safe in this isolated
    // child; production keeps the keys when cleanup fails and never claims they were released.
    context->get_scan_manager().reset_all();
    if (std::string_view{variant} == "cleanup_state") query_ok(*fixture.con, "ROLLBACK");
    return;
  }
  query_ok(*fixture.con, "SET sirius_test_mark_runtime_unavailable_before_window = true");

  std::unique_ptr<duckdb::MaterializedQueryResult> result;
  if (std::string_view{variant} == "transparent") {
    result = fixture.con->Query("SELECT count(*) FROM native_lease_t");
  } else if (std::string_view{variant} == "explicit") {
    auto const inner = "SELECT count(*) FROM " + fixture.attach_alias + ".main.native_lease_t";
    result           = fixture.con->Query("SELECT * FROM gpu_execution('" + inner + "')");
  } else {
    FAIL("unknown unavailable child variant: " << variant);
  }
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  CHECK(result->GetValue(0, 0).ToString() == "300000");
  auto const after = context->get_transparent_execution_stats();
  CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  if (std::string_view{variant} == "transparent") {
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.successful_rebinds == before.successful_rebinds + 1);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  }
}

TEST_CASE("never-entered execution windows preserve local CPU fallback",
          "[scan][native][checkpoint]")
{
  for (auto const* variant : {"transparent", "explicit"}) {
    auto const result = sirius::test::run_test_child(kUnavailableChildCase, variant);
    INFO("variant: " << variant << "\nchild output:\n" << result.output);
    CHECK_FALSE(result.timed_out);
    CHECK(result.signal == -1);
    CHECK(result.exit_code == 0);
  }
}

TEST_CASE("native checkpoint lease starts at execution preparation and releases on success",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling = sibling_connection(fixture);
  auto context = sirius::test::get_registered_sirius_context(*con);

  query_ok(*con, "SET sirius_test_pause_native_decode_ms = 1200");
  auto query = std::async(std::launch::async, [&] {
    auto result       = con->Query("SELECT sum(i) FROM native_lease_t");
    auto const failed = result->HasError();
    return std::make_pair(failed, scalar_or_error(std::move(result)));
  });

  REQUIRE(wait_for_key(*context, 5s));
  auto checkpoint = sibling->Query("CHECKPOINT");
  REQUIRE(checkpoint);
  REQUIRE(checkpoint->HasError());
  CHECK(checkpoint->GetErrorType() == duckdb::ExceptionType::TRANSACTION);
  auto forced_checkpoint = std::async(std::launch::async, [&] {
    auto result = sibling->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(forced_checkpoint.wait_for(150ms) == std::future_status::timeout);
  CHECK(query.wait_for(5s) == std::future_status::ready);
  auto [failed, value] = query.get();
  INFO(value);
  REQUIRE_FALSE(failed);
  CHECK(value == "44999850000");
  CHECK(forced_checkpoint.wait_for(5s) == std::future_status::ready);
  CHECK(forced_checkpoint.get().empty());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*sibling, "CHECKPOINT");
}

TEST_CASE("native metadata cache hits retain checkpoint protection",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  check_metadata_walk_lease(true);
}

TEST_CASE("fresh parallel native metadata walks retain checkpoint protection",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  check_metadata_walk_lease(false);
}

TEST_CASE("planning and retained prepared native plans hold no checkpoint lease",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling = sibling_connection(fixture);
  auto context = sirius::test::get_registered_sirius_context(*con);

  query_ok(*con, "SET sirius_test_inject_read_view_mismatch = 'finalize'");
  auto declined = con->Prepare("SELECT sum(i) FROM native_lease_t");
  REQUIRE(declined);
  REQUIRE_FALSE(declined->HasError());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*sibling, "CHECKPOINT");

  query_ok(*con, "SET sirius_test_inject_read_view_mismatch = 'off'");
  auto prepared = con->Prepare("SELECT sum(i) FROM native_lease_t");
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*sibling, "CHECKPOINT");

  query_ok(*con, "SET sirius_test_pause_native_after_prepare_ms = 1000");
  auto query = std::async(std::launch::async, [&] {
    auto result = con->Query("SELECT count(*) FROM native_lease_t");
    return scalar_or_error(std::move(result));
  });
  REQUIRE(query.wait_for(150ms) == std::future_status::timeout);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*sibling, "CHECKPOINT");
  CHECK(query.wait_for(5s) == std::future_status::ready);
  CHECK(query.get() == "300000");
}

TEST_CASE("native walk refusal is an execution failure and releases before replay",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto context      = sirius::test::get_registered_sirius_context(*con);
  auto const before = context->get_transparent_execution_stats();
  query_ok(*con, "SET sirius_test_inject_native_walk_failure = 'native_lease_t'");

  auto result = con->Query("SELECT count(*) FROM native_lease_t");
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  CHECK(result->GetValue(0, 0).ToString() == "300000");
  auto const after = context->get_transparent_execution_stats();
  CHECK(after.successful_rebinds == before.successful_rebinds + 1);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());

  auto prepared = con->Prepare("SELECT count(*) FROM native_lease_t");
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());
  auto prepared_before = context->get_transparent_execution_stats();
  for (int execution = 0; execution < 2; ++execution) {
    auto replayed = prepared->Execute();
    REQUIRE(replayed);
    REQUIRE_FALSE(replayed->HasError());
    CHECK(scalar_value(*replayed) == "300000");
    CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  }
  auto prepared_after = context->get_transparent_execution_stats();
  CHECK(prepared_after.successful_rebinds == prepared_before.successful_rebinds + 2);
  CHECK(prepared_after.runtime_fallbacks == prepared_before.runtime_fallbacks + 2);
  CHECK(prepared_after.fallbacks == prepared_before.fallbacks);

  query_ok(*con, "SET enable_duckdb_fallback = false");
  auto failed = con->Query("SELECT count(*) FROM native_lease_t");
  REQUIRE(failed);
  REQUIRE(failed->HasError());
  CHECK(failed->GetError().find("Sirius GPU execution failed:") != std::string::npos);
  CHECK(failed->GetError().find("injected native metadata walk failure") != std::string::npos);
  CHECK(failed->GetError().find("GPU plan generation failed:") == std::string::npos);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
}

TEST_CASE("explicit native walk refusal releases before checkpoint and CPU replay",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling      = sibling_connection(fixture);
  auto context      = sirius::test::get_registered_sirius_context(*con);
  auto const before = context->get_transparent_execution_stats();
  query_ok(*con, "SET sirius_test_inject_native_walk_failure = 'native_lease_t'");
  query_ok(*con, "SET sirius_test_pause_native_decode_ms = 1200");
  auto const inner = "SELECT count(*) FROM " + attach_alias + ".main.native_lease_t";

  auto explicit_query = std::async(std::launch::async, [&] {
    auto result = con->Query("SELECT * FROM gpu_execution('" + inner + "')");
    return scalar_or_error(std::move(result));
  });
  REQUIRE(wait_for_key(*context, 5s));
  auto forced_checkpoint = std::async(std::launch::async, [&] {
    auto result = sibling->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(forced_checkpoint.wait_for(150ms) == std::future_status::timeout);
  REQUIRE(explicit_query.wait_for(10s) == std::future_status::ready);
  CHECK(explicit_query.get() == "300000");
  REQUIRE(forced_checkpoint.wait_for(5s) == std::future_status::ready);
  CHECK(forced_checkpoint.get().empty());
  auto const after = context->get_transparent_execution_stats();
  CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*con, "SET sirius_test_pause_native_decode_ms = 0");
  query_ok(*con, "SET gpu_execution = false");
  query_ok(*con, "SET enable_duckdb_fallback = false");
  auto failed = con->Query("SELECT * FROM gpu_execution('" + inner + "')");
  REQUIRE(failed);
  REQUIRE(failed->HasError());
  INFO(failed->GetError());
  CHECK(failed->GetError().find("SiriusExecuteQuery error:") != std::string::npos);
  CHECK(failed->GetError().find("injected native metadata walk failure") != std::string::npos);
  CHECK(failed->GetError().find("GPU plan generation failed:") == std::string::npos);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
}

TEST_CASE("native checkpoint lease releases on cancellation with a forced checkpoint",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling = sibling_connection(fixture);
  auto context = sirius::test::get_registered_sirius_context(*con);
  query_ok(*con, "SET sirius_test_pause_native_decode_ms = 1500");

  auto cancelled = std::async(std::launch::async, [&] {
    auto result = con->Query("SELECT sum(i) FROM native_lease_t");
    return result->HasError();
  });
  REQUIRE(wait_for_key(*context, 5s));
  auto forced_checkpoint = std::async(std::launch::async, [&] {
    auto result = sibling->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(forced_checkpoint.wait_for(150ms) == std::future_status::timeout);
  con->Interrupt();
  CHECK(cancelled.wait_for(5s) == std::future_status::ready);
  CHECK(cancelled.get());
  CHECK(forced_checkpoint.wait_for(5s) == std::future_status::ready);
  CHECK(forced_checkpoint.get().empty());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*sibling, "CHECKPOINT");
}

TEST_CASE("native checkpoint lease releases on a decode error-result",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling = sibling_connection(fixture);
  auto context = sirius::test::get_registered_sirius_context(*con);
  query_ok(*con, "SET sirius_test_inject_native_decode_failure = 'native_lease_t'");
  auto const before = context->get_transparent_execution_stats();
  auto replayed     = con->Query("SELECT count(*) FROM native_lease_t");
  REQUIRE(replayed);
  INFO((replayed->HasError() ? replayed->GetError() : ""));
  REQUIRE_FALSE(replayed->HasError());
  CHECK(replayed->GetValue(0, 0).ToString() == "300000");
  auto const after = context->get_transparent_execution_stats();
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*sibling, "CHECKPOINT");
}

TEST_CASE("framework internal connections remain read-only while a native lease is held",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling        = sibling_connection(fixture);
  auto internal_outer = sibling_connection(fixture);
  auto context        = sirius::test::get_registered_sirius_context(*con);
  query_ok(*con, "SET sirius_test_pause_native_decode_ms = 1500");

  auto query = std::async(std::launch::async, [&] {
    auto result = con->Query("SELECT sum(i) FROM native_lease_t");
    return scalar_or_error(std::move(result));
  });
  REQUIRE(wait_for_key(*context, 5s));

  auto forced_checkpoint = std::async(std::launch::async, [&] {
    auto result = sibling->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(forced_checkpoint.wait_for(150ms) == std::future_status::timeout);

  auto internal        = duckdb::SiriusContext::open_internal_connection(*internal_outer->context);
  auto const qualified = attach_alias + ".main.native_lease_t";
  auto selected        = internal.Query("SELECT count(*) FROM " + qualified);
  REQUIRE(selected);
  REQUIRE_FALSE(selected->HasError());
  CHECK(selected->GetValue(0, 0).ToString() == "300000");

  CHECK_THROWS_AS(internal.Query("UPDATE " + qualified + " SET i = i WHERE false"),
                  duckdb::InvalidInputException);

  CHECK(query.wait_for(5s) == std::future_status::ready);
  CHECK(query.get() == "44999850000");
  CHECK(forced_checkpoint.wait_for(5s) == std::future_status::ready);
  CHECK(forced_checkpoint.get().empty());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
}

TEST_CASE("a multi-native plan releases every checkpoint key",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling = sibling_connection(fixture);
  auto context = sirius::test::get_registered_sirius_context(*con);
  query_ok(*con, "SET sirius_test_pause_native_decode_ms = 700");

  auto query = std::async(std::launch::async, [&] {
    auto result = con->Query(
      "SELECT count(*) FROM native_lease_t lhs JOIN native_lease_t rhs ON lhs.i = rhs.i");
    return scalar_or_error(std::move(result));
  });
  REQUIRE(wait_for_key_count(*context, 2, 5s));
  auto checkpoint = sibling->Query("CHECKPOINT");
  REQUIRE(checkpoint);
  REQUIRE(checkpoint->HasError());
  CHECK(checkpoint->GetErrorType() == duckdb::ExceptionType::TRANSACTION);
  CHECK(query.wait_for(10s) == std::future_status::ready);
  CHECK(query.get() == "300000");
  CHECK(context->get_scan_manager().checkpoint_key_count() == 0);
  query_ok(*sibling, "CHECKPOINT");
}

TEST_CASE("a waiting forced checkpoint stalls writes but not read-only transactions",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto checkpoint_connection = sibling_connection(fixture);
  auto writer                = sibling_connection(fixture);
  auto reader                = sibling_connection(fixture);
  auto context               = sirius::test::get_registered_sirius_context(*con);
  query_ok(*con, "SET sirius_test_pause_native_decode_ms = 1800");
  query_ok(*writer, "SET gpu_execution = false");
  query_ok(*reader, "SET gpu_execution = false");

  auto query = std::async(std::launch::async, [&] {
    auto result = con->Query("SELECT sum(i) FROM native_lease_t");
    return scalar_or_error(std::move(result));
  });
  REQUIRE(wait_for_key(*context, 5s));
  auto forced_checkpoint = std::async(std::launch::async, [&] {
    auto result = checkpoint_connection->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(forced_checkpoint.wait_for(150ms) == std::future_status::timeout);

  auto write = std::async(std::launch::async, [&] {
    auto result = writer->Query("INSERT INTO native_lease_t VALUES (300000)");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(write.wait_for(150ms) == std::future_status::timeout);
  query_ok(*reader, "BEGIN TRANSACTION READ ONLY");
  auto read = reader->Query("SELECT count(*) FROM native_lease_t");
  REQUIRE(read);
  REQUIRE_FALSE(read->HasError());
  CHECK(read->GetValue(0, 0).ToString() == "300000");
  query_ok(*reader, "ROLLBACK");

  CHECK(query.wait_for(5s) == std::future_status::ready);
  CHECK(query.get() == "44999850000");
  CHECK(forced_checkpoint.wait_for(5s) == std::future_status::ready);
  CHECK(forced_checkpoint.get().empty());
  CHECK(write.wait_for(5s) == std::future_status::ready);
  CHECK(write.get().empty());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*checkpoint_connection, "CHECKPOINT");
}

TEST_CASE("native-first planning cannot deadlock a cold Iceberg metadata connection",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto checkpoint_connection = sibling_connection(fixture);
  auto context               = sirius::test::get_registered_sirius_context(*con);
  query_ok(*con, "LOAD iceberg");
  query_ok(*con, "SET unsafe_enable_version_guessing = true");
  query_ok(*con, "SET disabled_optimizers = 'join_order,build_side_probe_side'");
  phase_gate native_leaf;
  scoped_native_observer observer(*context, [&](auto&, auto phase, auto) {
    if (phase == "native_leaf") { native_leaf.stop(); }
  });

  auto const table_path =
    (std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/iceberg_v2_delete")
      .string();
  auto const iceberg = "iceberg_scan('" + table_path + "', snapshot_from_id = 2000000000000000001)";
  auto const before  = sirius::op::scan::iceberg_delete_data_uncached_read_count();
  auto query         = std::async(std::launch::async, [&] {
    auto result = con->Query("SELECT count(*) FROM native_lease_t native JOIN " + iceberg +
                             " ice ON native.i = ice.count");
    return scalar_or_error(std::move(result));
  });

  // Native lowering is paused before the metadata walk. Planning holds no checkpoint key, so
  // FORCE CHECKPOINT can finish before the query enters Iceberg's cold metadata lookup.
  REQUIRE(native_leaf.wait());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  auto checkpoint = std::async(std::launch::async, [&] {
    auto result = checkpoint_connection->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  REQUIRE(checkpoint.wait_for(1s) == std::future_status::ready);
  CHECK(checkpoint.get().empty());
  native_leaf.release();
  REQUIRE(query.wait_for(20s) == std::future_status::ready);
  CHECK(query.get() == "3");
  CHECK(sirius::op::scan::iceberg_delete_data_uncached_read_count() == before + 1);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
}

TEST_CASE("prepared native re-execution reacquires a fresh checkpoint lease",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling  = sibling_connection(fixture);
  auto context  = sirius::test::get_registered_sirius_context(*con);
  auto prepared = con->Prepare("SELECT count(*) FROM native_lease_t");
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());

  std::vector<uint64_t> iterations;
  std::vector<bool> held;
  scoped_native_observer observer(*context, [&](auto&, auto phase, uint64_t iteration) {
    if (phase == "native_prepared") {
      iterations.push_back(iteration);
      held.push_back(context->get_scan_manager().holds_any_checkpoint_key());
    }
  });
  auto const before = context->get_transparent_execution_stats();
  auto first        = prepared->Execute();
  REQUIRE(first);
  REQUIRE_FALSE(first->HasError());
  CHECK(scalar_value(*first) == "300000");
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*sibling, "INSERT INTO native_lease_t VALUES (300000)");
  query_ok(*sibling, "CHECKPOINT");
  auto second = prepared->Execute();
  REQUIRE(second);
  REQUIRE_FALSE(second->HasError());
  CHECK(scalar_value(*second) == "300001");
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  REQUIRE(iterations.size() == 2);
  CHECK(iterations[1] > iterations[0]);
  CHECK(held == std::vector<bool>{true, true});
  auto const after = context->get_transparent_execution_stats();
  CHECK(after.successful_rebinds == before.successful_rebinds + 2);
  CHECK(after.executions == before.executions + 2);
  CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
  CHECK(after.checkpoint_revalidation_failures == before.checkpoint_revalidation_failures);
}

TEST_CASE("pin_table holds the native checkpoint lease through materialization",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling = sibling_connection(fixture);
  auto context = sirius::test::get_registered_sirius_context(*con);
  phase_gate pin_ready;
  scoped_native_observer observer(*context, [&](auto&, auto phase, uint64_t iteration) {
    if (phase == "pin_prepared") { pin_ready.stop(iteration); }
  });

  auto pin = std::async(std::launch::async, [&] {
    auto result = con->Query("CALL pin_table(format='duckdb', name='native_lease_t', tier='gpu')");
    return result->HasError() ? result->GetError() : std::string{};
  });
  REQUIRE(pin_ready.wait());
  CHECK(context->get_scan_manager().holds_any_checkpoint_key());
  auto checkpoint = sibling->Query("CHECKPOINT");
  REQUIRE(checkpoint);
  REQUIRE(checkpoint->HasError());
  CHECK(checkpoint->GetErrorType() == duckdb::ExceptionType::TRANSACTION);
  pin_ready.release();
  CHECK(pin.wait_for(10s) == std::future_status::ready);
  CHECK(pin.get().empty());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*con, "BEGIN TRANSACTION READ ONLY");
  auto& catalog = duckdb::Catalog::GetCatalog(*con->context, attach_alias);
  auto& table = catalog.GetEntry<duckdb::TableCatalogEntry>(*con->context, "main", "native_lease_t")
                  .Cast<duckdb::DuckTableEntry>();
  auto const entry = context->get_scan_manager().find_pinned_entry_for_duckdb_table(
    attach_alias,
    "main",
    "native_lease_t",
    {table.oid, table.GetStorage().GetRowGroupCollection()});
  query_ok(*con, "COMMIT");
  REQUIRE(entry);
  REQUIRE(entry->mvcc);
  CHECK(entry->mvcc->checkpoint_iteration == pin_ready.iteration);
  query_ok(*sibling, "CHECKPOINT");
  query_ok(*con, "CALL unpin_table('native_lease_t')");
}

TEST_CASE("epoch-invalidated execution rebuild takes the native checkpoint lease",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling      = sibling_connection(fixture);
  auto context      = sirius::test::get_registered_sirius_context(*con);
  auto const before = context->get_transparent_execution_stats();
  query_ok(*con, "SET sirius_test_inject_pin_registry_change = true");
  phase_gate before_window;
  phase_gate prepared;
  scoped_native_observer observer(*context, [&](auto&, auto phase, uint64_t iteration) {
    if (phase == "before_window") { before_window.stop(); }
    if (phase == "native_prepared") { prepared.stop(iteration); }
  });
  auto query = std::async(std::launch::async, [&] {
    return scalar_or_error(con->Query("SELECT count(*) FROM native_lease_t"));
  });
  REQUIRE(before_window.wait());
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  query_ok(*sibling, "CHECKPOINT");
  before_window.release();
  REQUIRE(prepared.wait());
  CHECK(context->get_scan_manager().holds_any_checkpoint_key());
  auto checkpoint = sibling->Query("CHECKPOINT");
  REQUIRE(checkpoint);
  REQUIRE(checkpoint->HasError());
  CHECK(checkpoint->GetErrorType() == duckdb::ExceptionType::TRANSACTION);
  auto forced = std::async(std::launch::async, [&] {
    auto result = sibling->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(forced.wait_for(150ms) == std::future_status::timeout);
  prepared.release();
  REQUIRE(query.wait_for(10s) == std::future_status::ready);
  CHECK(query.get() == "300000");
  REQUIRE(forced.wait_for(10s) == std::future_status::ready);
  CHECK(forced.get().empty());
  auto const after = context->get_transparent_execution_stats();
  CHECK(after.execution_rebuilds == before.execution_rebuilds + 1);
  CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
}

TEST_CASE("replay admission uses its completed window rather than a later live lease",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling = sibling_connection(fixture);
  auto context = sirius::test::get_registered_sirius_context(*con);
  query_ok(*sibling, "SET gpu_execution = true");
  query_ok(*sibling, "SET sirius_test_sync_native_checkpoint = true");
  phase_gate later_prepared;
  scoped_native_observer observer(*context, [&](auto& client, auto phase, uint64_t iteration) {
    if (&client == sibling->context.get() && phase == "native_prepared") {
      later_prepared.stop(iteration);
    }
  });
  query_ok(*con, "SET sirius_test_inject_native_decode_failure = 'native_lease_t'");
  query_ok(*con, "SET sirius_test_sync_cpu_replay = true");

  std::string sql = "SELECT count(*) FROM native_lease_t";
  SECTION("transparent") {}
  SECTION("explicit")
  {
    sql = "SELECT * FROM gpu_execution('SELECT count(*) FROM " + attach_alias +
          ".main.native_lease_t')";
  }
  std::promise<void> at_replay;
  std::promise<void> resume_replay;
  auto arrived                         = at_replay.get_future();
  auto resume                          = resume_replay.get_future().share();
  context->cpu_replay_hook_for_testing = [&] {
    at_replay.set_value();
    if (resume.wait_for(10s) != std::future_status::ready)
      throw std::runtime_error("CPU replay rendezvous timed out");
  };
  struct hook_reset {
    duckdb::SiriusContext& context;
    ~hook_reset() { context.cpu_replay_hook_for_testing = {}; }
  } reset{*context};
  auto const before = context->get_transparent_execution_stats();
  auto first = std::async(std::launch::async, [&] { return scalar_or_error(con->Query(sql)); });
  // Declared after the future: release A before a failed assertion joins its worker.
  struct resume_guard {
    std::promise<void>& promise;
    bool sent = false;
    void release()
    {
      if (!sent) {
        sent = true;
        promise.set_value();
      }
    }
    ~resume_guard() { release(); }
  } release{resume_replay};
  REQUIRE(arrived.wait_for(10s) == std::future_status::ready);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
  auto later = std::async(std::launch::async, [&] {
    return scalar_or_error(sibling->Query("SELECT count(*) FROM native_lease_t"));
  });
  REQUIRE(later_prepared.wait());
  CHECK(context->get_scan_manager().holds_any_checkpoint_key());
  release.release();
  REQUIRE(first.wait_for(10s) == std::future_status::ready);
  CHECK(first.get() == "300000");
  CHECK(context->get_scan_manager().holds_any_checkpoint_key());
  CHECK(context->get_transparent_execution_stats().lease_held_at_replay ==
        before.lease_held_at_replay);
  later_prepared.release();
  REQUIRE(later.wait_for(10s) == std::future_status::ready);
  CHECK(later.get() == "300000");
  context->cpu_replay_hook_for_testing = {};
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
}

TEST_CASE("unwinding an unfinished window releases its native checkpoint lease",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling = sibling_connection(fixture);
  auto context = sirius::test::get_registered_sirius_context(*con);
  query_ok(*con, "BEGIN TRANSACTION READ ONLY");
  auto& catalog = duckdb::Catalog::GetCatalog(*con->context, attach_alias);
  auto& table = catalog.GetEntry<duckdb::TableCatalogEntry>(*con->context, "main", "native_lease_t")
                  .Cast<duckdb::DuckTableEntry>();
  std::future<std::string> checkpoint;
  bool unwound = false;
  try {
    duckdb::SiriusContext::StandaloneQueryScope window(*context, *con->context, "lease_unwind");
    context->get_scan_manager().acquire_checkpoint_key(window.query_id(),
                                                       table.GetStorage().GetAttached());
    REQUIRE(context->get_scan_manager().checkpoint_key_count() == 1);
    checkpoint = std::async(std::launch::async, [&] {
      auto result = sibling->Query("FORCE CHECKPOINT");
      return result->HasError() ? result->GetError() : std::string{};
    });
    CHECK(checkpoint.wait_for(150ms) == std::future_status::timeout);
    throw 42;
  } catch (int value) {
    unwound = value == 42;
  }
  CHECK(unwound);
  CHECK(context->get_scan_manager().checkpoint_key_count() == 0);
  REQUIRE(checkpoint.wait_for(5s) == std::future_status::ready);
  CHECK(checkpoint.get().empty());
  query_ok(*con, "ROLLBACK");
}

TEST_CASE("Iceberg metadata settings do not change the database default",
          "[scan][iceberg][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con = fixture.con;
  query_ok(*con, "LOAD iceberg");
  auto setting_value = [](auto& connection) {
    auto result = connection.Query("SELECT current_setting('unsafe_enable_version_guessing')");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    return result->GetValue(0, 0).template GetValue<bool>();
  };
  for (bool global_value : {false, true}) {
    CAPTURE(global_value);
    query_ok(*con,
             std::string("SET GLOBAL unsafe_enable_version_guessing = ") +
               (global_value ? "true" : "false"));
    query_ok(*con,
             std::string("SET SESSION unsafe_enable_version_guessing = ") +
               (global_value ? "false" : "true"));
    duckdb::Connection sibling(*con->context->db);
    sirius::op::scan::iceberg_metadata_connection internal(*con->context);
    CHECK(setting_value(internal) == !global_value);
    CHECK(setting_value(*con) == !global_value);
    CHECK(setting_value(sibling) == global_value);
    duckdb::Connection fresh(*con->context->db);
    CHECK(setting_value(fresh) == global_value);
  }
}

TEST_CASE("internal metadata queries cannot escape their read-only transaction",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto context = sirius::test::get_registered_sirius_context(*con);
  query_ok(*con, "BEGIN TRANSACTION READ ONLY");
  auto& catalog = duckdb::Catalog::GetCatalog(*con->context, attach_alias);
  auto& table = catalog.GetEntry<duckdb::TableCatalogEntry>(*con->context, "main", "native_lease_t")
                  .Cast<duckdb::DuckTableEntry>();
  {
    duckdb::SiriusContext::StandaloneQueryScope window(
      *context, *con->context, "internal_read_only");
    context->get_scan_manager().acquire_checkpoint_key(window.query_id(),
                                                       table.GetStorage().GetAttached());
    auto internal = duckdb::SiriusContext::open_internal_connection(*con->context);
    for (auto const* sql : {"COMMIT",
                            "ROLLBACK",
                            "BEGIN",
                            "COMMIT; BEGIN; SELECT 1",
                            "SELECT 1; COMMIT",
                            "PREPARE escape AS SELECT 1"}) {
      INFO(sql);
      CHECK_THROWS_AS(internal.Query(sql), duckdb::InvalidInputException);
    }
    auto selected = internal.Query("SELECT count(*) FROM " + attach_alias + ".main.native_lease_t");
    REQUIRE(selected);
    REQUIRE_FALSE(selected->HasError());
    CHECK(selected->GetValue(0, 0).ToString() == "300000");
    window.finish();
  }
  query_ok(*con, "ROLLBACK");
}

TEST_CASE("explicit error-result releases its lease before CPU replay",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  auto& con                = fixture.con;
  auto const& attach_alias = fixture.attach_alias;
  prepare_native_table(fixture);
  auto sibling      = sibling_connection(fixture);
  auto context      = sirius::test::get_registered_sirius_context(*con);
  auto const before = context->get_transparent_execution_stats();
  query_ok(*con, "SET sirius_test_pause_native_decode_ms = 1200");
  query_ok(*con, "SET sirius_test_inject_native_decode_failure = 'native_lease_t'");
  auto const inner = "SELECT count(*) FROM " + attach_alias + ".main.native_lease_t";

  auto explicit_query = std::async(std::launch::async, [&] {
    auto result = con->Query("SELECT * FROM gpu_execution('" + inner + "')");
    return scalar_or_error(std::move(result));
  });
  REQUIRE(wait_for_key(*context, 5s));
  auto forced_checkpoint = std::async(std::launch::async, [&] {
    auto result = sibling->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(forced_checkpoint.wait_for(150ms) == std::future_status::timeout);
  CHECK(explicit_query.wait_for(10s) == std::future_status::ready);
  CHECK(explicit_query.get() == "300000");
  CHECK(forced_checkpoint.wait_for(5s) == std::future_status::ready);
  CHECK(forced_checkpoint.get().empty());
  auto const after = context->get_transparent_execution_stats();
  CHECK(after.lease_held_at_replay == before.lease_held_at_replay);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
}

TEST_CASE("failed checkpoint cleanup retains keys and prevents replay",
          "[scan][native][checkpoint]")
{
  for (auto const* variant : {"cleanup_state", "cleanup_transparent", "cleanup_explicit"}) {
    auto result = sirius::test::run_test_child(kUnavailableChildCase, variant);
    INFO("variant: " << variant << "\nchild output:\n" << result.output);
    CHECK_FALSE(result.timed_out);
    CHECK(result.signal == -1);
    CHECK(result.exit_code == 0);
  }
}

TEST_CASE("checkpoint leases are released only for their owning query",
          "[scan][native][checkpoint][integration]")
{
  if (sirius::test::run_isolated()) { return; }
  NativeLeaseFixture fixture;
  prepare_native_table(fixture);
  auto context  = sirius::test::get_registered_sirius_context(*fixture.con);
  auto& manager = context->get_scan_manager();
  auto sibling  = sibling_connection(fixture);
  query_ok(*fixture.con, "BEGIN TRANSACTION READ ONLY");
  auto& catalog = duckdb::Catalog::GetCatalog(*fixture.con->context, fixture.attach_alias);
  auto& table =
    catalog.GetEntry<duckdb::TableCatalogEntry>(*fixture.con->context, "main", "native_lease_t")
      .Cast<duckdb::DuckTableEntry>();
  constexpr auto first  = sirius::make_query_id(100001);
  constexpr auto second = sirius::make_query_id(100002);
  manager.acquire_checkpoint_key(first, table.GetStorage().GetAttached());
  manager.acquire_checkpoint_key(second, table.GetStorage().GetAttached());
  CHECK(manager.checkpoint_key_count(first) == 1);
  CHECK(manager.checkpoint_key_count(second) == 1);
  CHECK(manager.checkpoint_key_count() == 2);
  auto checkpoint = std::async(std::launch::async, [&] {
    auto result = sibling->Query("FORCE CHECKPOINT");
    return result->HasError() ? result->GetError() : std::string{};
  });
  CHECK(checkpoint.wait_for(150ms) == std::future_status::timeout);
  manager.reset(first);
  CHECK(manager.checkpoint_key_count(first) == 0);
  CHECK(manager.checkpoint_key_count(second) == 1);
  CHECK(manager.checkpoint_key_count() == 1);
  CHECK(checkpoint.wait_for(150ms) == std::future_status::timeout);
  manager.reset(first);
  CHECK(manager.checkpoint_key_count(second) == 1);
  manager.reset(second);
  CHECK_FALSE(manager.holds_any_checkpoint_key());
  REQUIRE(checkpoint.wait_for(5s) == std::future_status::ready);
  CHECK(checkpoint.get().empty());
  query_ok(*fixture.con, "ROLLBACK");
}
