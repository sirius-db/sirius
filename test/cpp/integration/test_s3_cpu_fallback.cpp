#include "io/object_store_config.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "transparent/plan_source_policy.hpp"
#include "util/env_guard.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/isolated_checkpoint_test.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/s3_backend.hpp"
#include "utils/s3_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {
using sirius::test::s3::env_or;
using sirius::test::s3::sql_quote;

bool enter_case(bool kvikio = false)
{
  if (sirius::test::s3::skip_or_fail_unless(sirius::test::ensure_s3_test_env(),
                                            "SeaweedFS test environment is not available")) {
    return false;
  }
  auto const name   = Catch::getResultCapture().getCurrentTestName();
  auto const* child = std::getenv("SIRIUS_NATIVE_LEASE_CHILD_CASE");
  if (child && name == child) { return true; }
  sirius::test::scratch_dir dir("c1_config");
  std::ifstream input(sirius::test::integration_config_path());
  REQUIRE(input.good());
  std::string config{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  auto const position = config.find("  executor:\n");
  REQUIRE(position != std::string::npos);
  config.insert(
    position + std::string("  executor:\n").size(),
    std::string("    scan_manager:\n      backend: ") + (kvikio ? "kvikio" : "sirius") +
      "\n      cache:\n        mode: none\n      rest:\n        request_timeout_s: 30\n");
  auto const path = dir.file("config.yaml");
  {
    std::ofstream output(path);
    output << config;
    REQUIRE(output.good());
  }
  sirius::util::env_guard child_config("SIRIUS_TEST_SHARED_CONFIG_OVERRIDE", path);
  auto result = sirius::test::run_test_child(name);
  std::cout << result.output;
  INFO(result.output);
  REQUIRE_FALSE(result.timed_out);
  REQUIRE(result.signal == -1);
  REQUIRE(result.exit_code == 0);
  return false;
}

auto query_ok(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  return result;
}

void query_error(duckdb::Connection& con, std::string const& sql, std::string const& text)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO((result->HasError() ? result->GetError() : ""));
  CHECK(result->GetError().find(text) != std::string::npos);
}

struct fixture {
  fixture() : con(sirius::test::g_integration_env->make_connection()), reference(reference_db)
  {
    context = sirius::test::get_registered_sirius_context(con);
    sirius::io::object_store_config config;
    config.endpoint      = sirius::test::s3::require_env("SIRIUS_TEST_S3_ENDPOINT");
    config.region        = env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
    config.access_key    = sirius::test::s3::require_env("SIRIUS_TEST_S3_ACCESS_KEY");
    config.secret_key    = sirius::test::s3::require_env("SIRIUS_TEST_S3_SECRET_KEY");
    config.session_token = env_or("SIRIUS_TEST_S3_SESSION_TOKEN");
    config.tls_verify    = false;
    bucket               = sirius::test::s3::require_env("SIRIUS_TEST_S3_BUCKET");
    context->get_config().set_object_store_config(config);
    context->get_scan_manager().install_s3_config("s3://" + bucket, config);
    local = std::filesystem::path(sirius::test::s3::require_env("SIRIUS_TEST_S3_LOCAL_DIR")) /
            "parquet" / "nation.parquet";
    REQUIRE(std::filesystem::exists(local));
    query_ok(con, "SET enable_external_file_cache=false");
    query_ok(con, "SET gpu_execution=true");
    query_ok(con, "SET enable_duckdb_fallback=true");
    query_ok(reference, "SET gpu_execution=false");
  }

  ~fixture()
  {
    context->cpu_replay_hook_for_testing        = {};
    context->native_checkpoint_hook_for_testing = {};
  }

  void enable() { query_ok(con, "SET sirius_s3_cpu_fallback=true"); }
  std::string scan(bool remote) const
  {
    return "read_parquet(" +
           sql_quote(remote ? "s3://" + bucket + "/parquet/nation.parquet" : local.string()) + ")";
  }
  std::string rows(bool remote) const
  {
    return "SELECT n_nationkey, n_name FROM " + scan(remote) + " ORDER BY n_nationkey";
  }
  std::string window(bool remote) const
  {
    return "SELECT n_nationkey, row_number() OVER (ORDER BY n_nationkey) AS rn FROM " +
           scan(remote) + " ORDER BY n_nationkey";
  }
  void equal(std::string const& actual, std::string const& expected)
  {
    auto wanted = query_ok(reference, expected);
    auto got    = query_ok(con, actual);
    CHECK(sirius::test::collect_rows(*got, false) == sirius::test::collect_rows(*wanted, false));
  }
  void explicit_equal(std::string const& sql, std::string const& expected)
  {
    unsigned replays                     = 0;
    context->cpu_replay_hook_for_testing = [&] { ++replays; };
    query_ok(con, "SET sirius_test_sync_cpu_replay=true");
    equal("SELECT * FROM gpu_execution(" + sql_quote(sql) + ")", expected);
    CHECK(replays == 1);
    context->cpu_replay_hook_for_testing = {};
  }

  duckdb::Connection con;
  duckdb::DuckDB reference_db{nullptr};
  duckdb::Connection reference;
  duckdb::shared_ptr<duckdb::SiriusContext> context;
  std::string bucket;
  std::filesystem::path local;
};

std::string folded_count_query(fixture& f)
{
  auto const sql = "SELECT count(*) AS n, row_number() OVER () AS rn FROM " + f.scan(true);
  auto explain   = query_ok(f.con, "EXPLAIN " + sql);
  std::string plan;
  for (duckdb::idx_t row = 0; row < explain->RowCount(); ++row) {
    for (duckdb::idx_t column = 0; column < explain->ColumnCount(); ++column) {
      plan += explain->GetValue(column, row).ToString() + "\n";
    }
  }
  INFO(plan);
  REQUIRE(plan.find("WINDOW") != std::string::npos);
  REQUIRE(plan.find("READ_PARQUET") == std::string::npos);
  REQUIRE(plan.find("PARQUET_SCAN") == std::string::npos);
  return sql;
}

void check_folded_admission(bool revoke)
{
  fixture f;
  duckdb::Connection control(sirius::test::g_integration_env->database());
  query_ok(f.con, "SET threads=1");
  query_ok(control, "SET GLOBAL sirius_s3_cpu_fallback=true");
  REQUIRE(duckdb::s3_cpu_fallback_enabled(*f.con.context));
  auto const sql = folded_count_query(f);
  auto expected =
    query_ok(f.reference, "SELECT count(*) AS n, 1::BIGINT AS rn FROM " + f.scan(false));
  auto const before     = f.context->get_transparent_execution_stats();
  auto const cpu_before = f.context->cpu_only_executions();
  auto pending = f.con.PendingQuery(sql, duckdb::QueryResultOutputType::FORCE_MATERIALIZED);
  REQUIRE(pending);
  INFO((pending->HasError() ? pending->GetError() : ""));
  REQUIRE_FALSE(pending->HasError());
  REQUIRE(f.context->cpu_only_executions() == cpu_before);
  REQUIRE(f.context->get_transparent_execution_stats().fallbacks == before.fallbacks + 1);
  if (revoke) { query_ok(control, "SET GLOBAL sirius_s3_cpu_fallback=false"); }
  REQUIRE(duckdb::s3_cpu_fallback_enabled(*f.con.context) == !revoke);
  auto result = pending->Execute();
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  if (revoke) {
    REQUIRE(result->HasError());
    CHECK(result->GetErrorType() == duckdb::ExceptionType::EXECUTOR);
    CHECK(result->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
    CHECK(f.context->cpu_only_executions() == cpu_before);
  } else {
    REQUIRE_FALSE(result->HasError());
    CHECK(sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>(), false) ==
          sirius::test::collect_rows(*expected, false));
    CHECK(f.context->cpu_only_executions() == cpu_before + 1);
  }
  auto const after = f.context->get_transparent_execution_stats();
  CHECK(after.successful_rebinds == before.successful_rebinds);
  CHECK(after.executions == before.executions);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
}

void check_unavailable_view(bool warm)
{
  fixture f;
  if (warm) {
    REQUIRE(std::filesystem::file_size(f.local) <= 16384);
    query_ok(f.con, "SET enable_external_file_cache=true");
    query_ok(f.con, "SET GLOBAL validate_external_file_cache='NO_VALIDATION'");
    auto prepared = f.con.Prepare(f.rows(true));
    REQUIRE(prepared);
    INFO((prepared->HasError() ? prepared->GetError() : ""));
    REQUIRE_FALSE(prepared->HasError());
    REQUIRE_FALSE(duckdb::s3_cpu_fallback_enabled(*f.con.context));
    query_ok(f.con, "SET gpu_execution=false");
    f.equal(f.rows(true), f.rows(false));
    query_ok(f.con, "SET gpu_execution=true");
  }
  f.enable();
  query_ok(f.con, "CREATE VIEW c1_unavailable_view AS SELECT * FROM " + f.scan(true));
  unsigned replays                       = 0;
  f.context->cpu_replay_hook_for_testing = [&] { ++replays; };
  query_ok(f.con, "SET sirius_test_sync_cpu_replay=true");
  query_ok(f.con, "SET sirius_test_mark_runtime_unavailable_before_window=true");
  REQUIRE(f.context->get_runtime_health() == duckdb::SiriusContext::runtime_health::OK);
  auto const before = f.context->get_transparent_execution_stats();
  auto result       = f.con.Query(
    "SELECT * FROM gpu_execution('SELECT n_nationkey FROM c1_unavailable_view ORDER BY "
          "n_nationkey')");
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(result->GetErrorType() == duckdb::ExceptionType::EXECUTOR);
  CHECK(result->GetError().find("Sirius GPU runtime is unavailable") != std::string::npos);
  CHECK(result->GetError().find("S3 CPU fallback is not supported") == std::string::npos);
  CHECK(replays == 0);
  CHECK(f.context->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  CHECK(f.context->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
}

void check_publication_failure(bool remote, bool allow_s3)
{
  sirius::test::scratch_dir dir("c1_publish_local");
  fixture f;
  if (allow_s3) { f.enable(); }
  std::string sql       = f.rows(true);
  std::string reference = f.rows(false);
  if (!remote) {
    query_ok(f.con, "ATTACH " + dir.file_literal("native.duckdb") + " AS c1_publish_local");
    query_ok(f.con,
             "CREATE TABLE c1_publish_local.main.t AS SELECT range::BIGINT AS i FROM range(10)");
    query_ok(f.con, "CHECKPOINT c1_publish_local");
    sql       = "SELECT i FROM c1_publish_local.main.t ORDER BY i";
    reference = "SELECT range::BIGINT AS i FROM range(10) ORDER BY i";
  }
  auto const control_before = f.context->get_transparent_execution_stats();
  f.equal(sql, reference);
  auto const before = f.context->get_transparent_execution_stats();
  REQUIRE(before.successful_rebinds == control_before.successful_rebinds + 1);
  REQUIRE(before.executions == control_before.executions + 1);
  REQUIRE(before.fallbacks == control_before.fallbacks);
  REQUIRE(before.runtime_fallbacks == control_before.runtime_fallbacks);
  auto const cpu_before = f.context->cpu_only_executions();
  query_ok(f.con, "SET sirius_test_inject_finalize_publish_error=true");
  if (remote && !allow_s3) {
    auto result = f.con.Query(sql);
    REQUIRE(result);
    REQUIRE(result->HasError());
    INFO(result->GetError());
    CHECK(result->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
    CHECK(result->GetError().find("injected transparent plan publication failure") !=
          std::string::npos);
  } else {
    f.equal(sql, reference);
  }
  auto const after = f.context->get_transparent_execution_stats();
  CHECK(after.successful_rebinds == before.successful_rebinds);
  CHECK(after.executions == before.executions);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(after.fallbacks == before.fallbacks + (remote && !allow_s3 ? 0 : 1));
  CHECK(f.context->cpu_only_executions() == cpu_before + (remote && allow_s3 ? 1 : 0));
  if (!remote) { query_ok(f.con, "DETACH c1_publish_local"); }
}
}  // namespace

TEST_CASE("C1 default is false", "[s3][integration][cpu_fallback][c1_default]")
{
  if (!enter_case()) { return; }
  fixture f;
  auto result = query_ok(f.con, "SELECT current_setting('sirius_s3_cpu_fallback')");
  REQUIRE(result->RowCount() == 1);
  CHECK_FALSE(result->GetValue(0, 0).GetValue<bool>());
}

TEST_CASE("C1 A1 runtime replay matches local CPU", "[s3][integration][cpu_fallback][c1_a1]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "SET sirius_test_inject_transparent_gpu_error='c1 runtime probe'");
  auto before = f.context->get_transparent_execution_stats();
  f.equal(f.rows(true), f.rows(false));
  CHECK(f.context->get_transparent_execution_stats().runtime_fallbacks ==
        before.runtime_fallbacks + 1);
}

TEST_CASE("C1 A2 planning decline matches local CPU", "[s3][integration][cpu_fallback][c1_a2]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  f.equal(f.window(true), f.window(false));
}

TEST_CASE("C1 A2 kvikio decline is refused", "[s3][integration][cpu_fallback][c1_backend]")
{
  if (!enter_case(true)) { return; }
  fixture f;
  f.enable();
  query_error(f.con, f.window(true), "S3 CPU fallback requires the REST backend");
}

TEST_CASE("C1 A2 prepared execution repeats with rebind",
          "[s3][integration][cpu_fallback][c1_prepare]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  auto expected = query_ok(f.reference, f.window(false));
  auto prepared = f.con.Prepare(f.window(true));
  REQUIRE(prepared);
  INFO((prepared->HasError() ? prepared->GetError() : ""));
  REQUIRE_FALSE(prepared->HasError());
  duckdb::vector<duckdb::Value> values;
  for (int run = 0; run < 2; ++run) {
    CAPTURE(run);
    auto result = prepared->Execute(values, false);
    REQUIRE(result);
    INFO((result->HasError() ? result->GetError() : ""));
    REQUIRE_FALSE(result->HasError());
    CHECK(sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>(), false) ==
          sirius::test::collect_rows(*expected, false));
  }
}

TEST_CASE("C1 eligibility drift rebinds a GPU plan into A2",
          "[s3][integration][cpu_fallback][c1_rebind]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  sirius::test::scratch_dir dir("c1_drift");
  query_ok(f.con, "ATTACH " + dir.file_literal("drift.duckdb") + " AS c1_drift");
  query_ok(f.con, "CREATE TABLE c1_drift.main.t AS SELECT 0::BIGINT id, 'short'::VARCHAR AS v");
  query_ok(f.con, "CHECKPOINT c1_drift");
  auto before   = f.context->get_transparent_execution_stats();
  auto prepared = f.con.Prepare("SELECT s.n_nationkey, t.v FROM " + f.scan(true) +
                                " s JOIN c1_drift.main.t t ON s.n_nationkey=t.id");
  REQUIRE(prepared);
  INFO((prepared->HasError() ? prepared->GetError() : ""));
  REQUIRE_FALSE(prepared->HasError());
  CHECK(f.context->get_transparent_execution_stats().successful_rebinds >
        before.successful_rebinds);
  query_ok(f.con, "UPDATE c1_drift.main.t SET v=repeat('x', 5000)");
  query_ok(f.con, "CHECKPOINT c1_drift");
  auto expected = query_ok(
    f.reference,
    "SELECT n_nationkey, repeat('x', 5000) AS v FROM " + f.scan(false) + " WHERE n_nationkey=0");
  duckdb::vector<duckdb::Value> values;
  auto result = prepared->Execute(values, false);
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  CHECK(sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>()) ==
        sirius::test::collect_rows(*expected));
  prepared.reset();
  query_ok(f.con, "DETACH c1_drift");
}

TEST_CASE("C1 A3 replay carries the outer decision", "[s3][integration][cpu_fallback][c1_a3]")
{
  if (!enter_case()) { return; }
  fixture f;
  duckdb::Connection other(sirius::test::g_integration_env->database());
  f.enable();
  auto default_value = query_ok(other, "SELECT current_setting('sirius_s3_cpu_fallback')");
  CHECK_FALSE(default_value->GetValue(0, 0).GetValue<bool>());
  f.explicit_equal(f.window(true), f.window(false));
}

TEST_CASE("C1 A3 default off refuses replay", "[s3][integration][cpu_fallback][c1_control]")
{
  if (!enter_case()) { return; }
  fixture f;
  query_error(f.con,
              "SELECT * FROM gpu_execution(" + sql_quote(f.window(true)) + ")",
              "S3 CPU fallback is not supported");
}

TEST_CASE("C1 plain CPU read is enabled", "[s3][integration][cpu_fallback][c1_plain]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "SET gpu_execution=false");
  f.equal(f.rows(true), f.rows(false));
}

TEST_CASE("C1 plain CPU default off refuses opens", "[s3][integration][cpu_fallback][c1_control]")
{
  if (!enter_case()) { return; }
  fixture f;
  query_ok(f.con, "SET gpu_execution=false");
  query_error(f.con, f.rows(true), "requires GPU execution");
}

TEST_CASE("C1 plain CPU glob is enabled", "[s3][integration][cpu_fallback][c1_glob]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "SET gpu_execution=false");
  auto sql = "SELECT n_nationkey, n_name FROM read_parquet(" +
             sql_quote("s3://" + f.bucket + "/parquet/nation*.parquet") + ") ORDER BY n_nationkey";
  f.equal(sql, f.rows(false));
}

TEST_CASE("C1 view hidden S3 planning decline is enabled",
          "[s3][integration][cpu_fallback][c1_view]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "CREATE VIEW c1_remote AS SELECT * FROM " + f.scan(true));
  f.equal(
    "SELECT n_nationkey, row_number() OVER (ORDER BY n_nationkey) AS rn FROM c1_remote ORDER BY "
    "n_nationkey",
    f.window(false));
}

TEST_CASE("C1 fallback disabled keeps the GPU error",
          "[s3][integration][cpu_fallback][c1_precedence]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "SET enable_duckdb_fallback=false");
  query_ok(f.con, "SET sirius_test_inject_transparent_gpu_error='c1 disabled probe'");
  auto before = f.context->get_transparent_execution_stats();
  auto result = f.con.Query(f.rows(true));
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(result->GetError().find("c1 disabled probe") != std::string::npos);
  CHECK(result->GetError().find("S3 CPU fallback is not supported") == std::string::npos);
  CHECK(f.context->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE("C1 unavailable runtime keeps the typed error",
          "[s3][integration][cpu_fallback][c1_precedence]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "SET sirius_test_mark_runtime_unavailable_before_window=true");
  auto result = f.con.Query(f.rows(true));
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO((result->HasError() ? result->GetError() : ""));
  CHECK(result->GetErrorType() == duckdb::ExceptionType::EXECUTOR);
  CHECK(result->GetError().find("Sirius GPU runtime is unavailable") != std::string::npos);
  CHECK(result->GetError().find("S3 CPU fallback is not supported") == std::string::npos);
}

TEST_CASE("C1 interrupt is not replayed", "[s3][integration][cpu_fallback][c1_precedence]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "SET sirius_test_sync_native_checkpoint=true");
  unsigned interrupts = 0;
  f.context->native_checkpoint_hook_for_testing =
    [&](duckdb::ClientContext&, std::string_view phase, uint64_t) {
      if (phase == "before_window") {
        ++interrupts;
        throw duckdb::InterruptException();
      }
    };
  auto before = f.context->get_transparent_execution_stats();
  auto result = f.con.Query(f.rows(true));
  REQUIRE(result);
  REQUIRE(result->HasError());
  CHECK(result->GetErrorType() == duckdb::ExceptionType::INTERRUPT);
  CHECK(interrupts == 1);
  CHECK(f.context->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE("C1 A2 permission is checked again by Execute",
          "[s3][integration][cpu_fallback][c1_prepare]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  auto prepared = f.con.Prepare(f.window(true));
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());
  query_ok(f.con, "SET sirius_s3_cpu_fallback=false");
  auto result = prepared->Execute();
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO((result->HasError() ? result->GetError() : ""));
  CHECK(result->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
}

TEST_CASE("C1 preparing A2 does not activate CPU admission",
          "[s3][integration][cpu_fallback][c1_prepare]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  auto prepared = f.con.Prepare(f.window(true));
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());
  auto before = f.context->get_transparent_execution_stats();
  f.equal(f.rows(true), f.rows(false));
  auto after = f.context->get_transparent_execution_stats();
  CHECK(after.executions == before.executions + 1);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE("C1 plain CPU kvikio is refused before object open",
          "[s3][integration][cpu_fallback][c1_backend]")
{
  if (!enter_case(true)) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "SET gpu_execution=false");
  query_error(f.con,
              "SELECT * FROM read_parquet(" +
                sql_quote("s3://" + f.bucket + "/missing-c1-object.parquet") + ")",
              "S3 CPU fallback requires the REST backend");
}

TEST_CASE("C1 rewritten CPU callback remains unsupported off",
          "[s3][integration][cpu_fallback][c1_control]")
{
  if (!enter_case()) { return; }
  fixture f;
  query_ok(f.con, "SET gpu_execution=false");
  query_error(f.con,
              "SELECT * FROM sirius_read_parquet(" +
                sql_quote("s3://" + f.bucket + "/parquet/nation.parquet") + ")",
              "internal rewrite target");
}

TEST_CASE("C1 rewritten CPU callback remains unsupported on",
          "[s3][integration][cpu_fallback][c1_rewritten]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "SET gpu_execution=false");
  auto result = f.con.Query("SELECT * FROM sirius_read_parquet(" +
                            sql_quote("s3://" + f.bucket + "/parquet/nation.parquet") + ")");
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO((result->HasError() ? result->GetError() : ""));
  CHECK(result->GetError().find("internal rewrite target") != std::string::npos);
}

TEST_CASE("C1 setting respects connection and global scope",
          "[s3][integration][cpu_fallback][c1_scope]")
{
  if (!enter_case()) { return; }
  fixture f;
  duckdb::Connection other(sirius::test::g_integration_env->database());
  f.enable();
  auto local     = query_ok(f.con, "SELECT current_setting('sirius_s3_cpu_fallback')");
  auto unchanged = query_ok(other, "SELECT current_setting('sirius_s3_cpu_fallback')");
  CHECK(local->GetValue(0, 0).GetValue<bool>());
  CHECK_FALSE(unchanged->GetValue(0, 0).GetValue<bool>());
  query_ok(f.con, "SET GLOBAL sirius_s3_cpu_fallback=true");
  duckdb::Connection fresh(sirius::test::g_integration_env->database());
  auto inherited = query_ok(fresh, "SELECT current_setting('sirius_s3_cpu_fallback')");
  CHECK(inherited->GetValue(0, 0).GetValue<bool>());
}

TEST_CASE("C1 warm EFC with default off is characterized",
          "[s3][integration][cpu_fallback][c1_characterization]")
{
  if (!enter_case()) { return; }
  fixture f;
  query_ok(f.con, "SET enable_external_file_cache=true");
  query_ok(f.con, "SET validate_external_file_cache='NO_VALIDATION'");
  auto prepared = f.con.Prepare(f.rows(true));
  REQUIRE(prepared);
  INFO((prepared->HasError() ? prepared->GetError() : ""));
  REQUIRE_FALSE(prepared->HasError());
  query_ok(f.con, "SET gpu_execution=false");
  auto result = f.con.Query(f.rows(true));
  REQUIRE(result);
  std::cout << "C1_EFC_OBSERVATION outcome=" << (result->HasError() ? "error" : "success");
  if (result->HasError()) {
    std::cout << " error=" << result->GetError();
  } else {
    std::cout << " rows=" << result->RowCount();
    auto expected = query_ok(f.reference, f.rows(false));
    std::cout << " matches_local="
              << (sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>(),
                                             false) ==
                  sirius::test::collect_rows(*expected, false));
  }
  std::cout << std::endl;
}

TEST_CASE("C1 explicit CPU ignores the fallback enabled option",
          "[s3][integration][cpu_fallback][c1_plain]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  query_ok(f.con, "SET gpu_execution=false");
  query_ok(f.con, "SET enable_duckdb_fallback=false");
  f.equal(f.rows(true), f.rows(false));
}

TEST_CASE("C1 decision preserves stream and incomplete discovery vetoes",
          "[cpu_fallback][policy][c1_api]")
{
  using namespace sirius::transparent;
  plan_source_policy policy;
  policy.scans.push_back(
    {"remote", byte_source_class::sirius_owned_s3, false, "S3 CPU fallback is not supported"});
  CHECK_FALSE(policy.cpu_replay_permitted());
  CHECK_FALSE(policy.cpu_replay_permitted(cpu_replay_decision{false}));
  CHECK(policy.cpu_replay_permitted(cpu_replay_decision{true}));
  REQUIRE_NOTHROW(require_cpu_replay(policy, cpu_replay_decision{true}, "", "probe"));
  REQUIRE_THROWS(require_cpu_replay(policy, "", "probe"));
  policy.scans.push_back(
    {"sirius_stream_source", byte_source_class::stream, false, "stream source has no CPU body"});
  CHECK_FALSE(policy.cpu_replay_permitted(cpu_replay_decision{true}));
  REQUIRE_THROWS_WITH(require_cpu_replay(policy, cpu_replay_decision{true}, "", "probe"),
                      "CPU fallback is not supported: sirius_stream_source: stream source has no "
                      "CPU body. Underlying GPU error: probe");
  policy.scans.clear();
  policy.discovery_complete = false;
  CHECK_FALSE(policy.cpu_replay_permitted(cpu_replay_decision{true}));
  REQUIRE_THROWS_WITH(
    require_non_s3_cpu_replay(policy, cpu_replay_decision{true}, "probe"),
    "CPU fallback is not supported: source discovery incomplete. Underlying GPU error: probe");
}

TEST_CASE("C1 A2 observer counts execution rather than preparation",
          "[s3][integration][cpu_fallback][c1_api]")
{
  if (!enter_case()) { return; }
  fixture f;
  f.enable();
  auto const before = f.context->cpu_only_executions();
  f.equal(f.window(true), f.window(false));
  CHECK(f.context->cpu_only_executions() == before + 1);
  auto prepared = f.con.Prepare(f.window(true));
  REQUIRE(prepared);
  INFO((prepared->HasError() ? prepared->GetError() : ""));
  REQUIRE_FALSE(prepared->HasError());
  CHECK(f.context->cpu_only_executions() == before + 1);
  auto expected = query_ok(f.reference, f.window(false));
  duckdb::vector<duckdb::Value> values;
  for (std::uint64_t run = 0; run < 2; ++run) {
    auto result = prepared->Execute(values, false);
    REQUIRE(result);
    INFO((result->HasError() ? result->GetError() : ""));
    REQUIRE_FALSE(result->HasError());
    CHECK(sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>(), false) ==
          sirius::test::collect_rows(*expected, false));
    CHECK(f.context->cpu_only_executions() == before + run + 2);
  }
}

TEST_CASE("C1 folded metadata A2 refuses a revoked pending admission",
          "[s3][integration][cpu_fallback][c1_review]")
{
  if (!enter_case()) { return; }
  check_folded_admission(true);
}

TEST_CASE("C1 folded metadata A2 pending control matches local CPU",
          "[s3][integration][cpu_fallback][c1_review]")
{
  if (!enter_case()) { return; }
  check_folded_admission(false);
}

TEST_CASE("C1 unavailable explicit S3 view refuses replay with cold EFC",
          "[s3][integration][cpu_fallback][c1_review]")
{
  if (!enter_case()) { return; }
  check_unavailable_view(false);
}

TEST_CASE("C1 unavailable explicit S3 view refuses replay with warm EFC",
          "[s3][integration][cpu_fallback][c1_review]")
{
  if (!enter_case()) { return; }
  check_unavailable_view(true);
}

TEST_CASE("C1 publication failure restores the admitted S3 CPU plan",
          "[s3][integration][cpu_fallback][c1_review]")
{
  if (!enter_case()) { return; }
  check_publication_failure(true, true);
}

TEST_CASE("C1 publication failure preserves the S3 off refusal",
          "[s3][integration][cpu_fallback][c1_review]")
{
  if (!enter_case()) { return; }
  check_publication_failure(true, false);
}

TEST_CASE("C1 publication failure restores the local CPU plan",
          "[s3][integration][cpu_fallback][c1_review]")
{
  if (!enter_case()) { return; }
  check_publication_failure(false, false);
}
