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

// Tests for execution-time (runtime) fallback from GPU to DuckDB CPU on the
// transparent interception path. GPU failures are forced deterministically via the
// test-only `sirius_test_inject_transparent_gpu_error` setting, which makes
// PhysicalSiriusExecution fail *after* plan generation succeeds — i.e. a runtime
// failure, distinct from a plan-time (create_plan) fallback.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/client_context.hpp>
#include <unistd.h>  // getpid
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

/// Guard that sets SIRIUS_CONFIG_FILE for the duration of the test.
struct config_env_guard {
  explicit config_env_guard(const std::string& path)
  {
    if (const char* current = std::getenv("SIRIUS_CONFIG_FILE")) {
      had_original_value = true;
      original_value     = current;
    }
    setenv("SIRIUS_CONFIG_FILE", path.c_str(), 1);
  }

  ~config_env_guard()
  {
    if (had_original_value) {
      setenv("SIRIUS_CONFIG_FILE", original_value.c_str(), 1);
    } else {
      unsetenv("SIRIUS_CONFIG_FILE");
    }
  }

  std::string original_value;
  bool had_original_value = false;
};

/// Fixture: a DuckDB connection with Sirius transparent execution enabled. Reuses
/// the shared integration environment when active, otherwise spins up its own
/// DuckDB against integration.yaml (same pattern as the transparent-execution
/// integration suite).
class RuntimeFallbackFixture {
 public:
  RuntimeFallbackFixture()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<config_env_guard>(cfg_path.string());
      db           = std::make_unique<duckdb::DuckDB>(nullptr);
      con          = std::make_unique<duckdb::Connection>(*db);
    }
    con->Query("SET gpu_execution = true;");
    con->Query("SET enable_duckdb_fallback = true;");
    clear_injection();

    // Sirius's duckdb-native GPU scan requires a single-file block manager, which
    // the in-memory catalog does not have. Create tables in a file-backed attached
    // database (unique per fixture) so queries actually reach GPU execution — a
    // prerequisite for exercising the *runtime* fallback path.
    static std::atomic<unsigned> seq{0};
    alias_   = "rfdb_" + std::to_string(seq.fetch_add(1));
    db_file_ = fs::temp_directory_path() / (alias_ + "_" + std::to_string(::getpid()) + ".db");
    std::error_code ec;
    fs::remove(db_file_, ec);
    fs::remove(fs::path(db_file_.string() + ".wal"), ec);
    auto attach = con->Query("ATTACH '" + db_file_.string() + "' AS " + alias_ + ";");
    REQUIRE_FALSE(attach->HasError());
    con->Query("USE " + alias_ + ";");
  }

  ~RuntimeFallbackFixture()
  {
    // Reset session state and detach so later tests are unaffected, then remove the
    // temp database files.
    if (con) {
      clear_injection();
      con->Query("ROLLBACK;");  // no-op if no transaction is open
      con->Query("USE memory;");
      con->Query("DETACH " + alias_ + ";");
    }
    std::error_code ec;
    fs::remove(db_file_, ec);
    fs::remove(fs::path(db_file_.string() + ".wal"), ec);
  }

  std::unique_ptr<duckdb::Connection> make_connection()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      return std::make_unique<duckdb::Connection>(
        sirius::test::g_integration_env->make_connection());
    }
    REQUIRE(db);
    return std::make_unique<duckdb::Connection>(*db);
  }

  // Create a table and CHECKPOINT it so its data lands in on-disk blocks. Sirius's
  // duckdb-native GPU scan reads committed blocks from the single-file block
  // manager; freshly created (WAL-resident) rows decode as empty segments on GPU.
  void create_table(const std::string& create_sql)
  {
    auto r = con->Query(create_sql);
    REQUIRE_FALSE(r->HasError());
    con->Query("CHECKPOINT;");
  }

  void inject(duckdb::Connection& c, const std::string& msg = "injected boom")
  {
    auto r = c.Query("SET sirius_test_inject_transparent_gpu_error = '" + msg + "';");
    REQUIRE_FALSE(r->HasError());
  }
  void inject(const std::string& msg = "injected boom") { inject(*con, msg); }

  void clear_injection(duckdb::Connection& c)
  {
    c.Query("SET sirius_test_inject_transparent_gpu_error = '';");
  }
  void clear_injection() { clear_injection(*con); }

  static std::string scalar(duckdb::Connection& c, const std::string& sql)
  {
    auto r = c.Query(sql);
    REQUIRE(r);
    if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->RowCount() == 1);
    return r->GetValue(0, 0).ToString();
  }

 protected:
  std::unique_ptr<config_env_guard> config_guard;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::string alias_;
  fs::path db_file_;
};

}  // namespace

// A GPU failure at runtime completes the query on CPU and is counted as a runtime
// fallback. Without injection the same query runs on the GPU.
TEST_CASE_METHOD(RuntimeFallbackFixture,
                 "runtime fallback: basic",
                 "[transparent][fallback][integration]")
{
  create_table("CREATE TABLE rf_basic AS SELECT i AS id, i * 2 AS val FROM range(1000) t(i);");
  const std::string q = "SELECT count(*) AS n, sum(val) AS s FROM rf_basic WHERE val > 100;";

  // With injection: GPU is attempted (rebind + execution), fails, falls back to CPU.
  inject();
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto gpu    = con->Query(q);
  REQUIRE(gpu);
  REQUIRE_FALSE(gpu->HasError());
  auto after = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 1);

  // Same query with no injection runs fully on the GPU (no runtime fallback).
  clear_injection();
  before    = sirius::test::get_transparent_execution_stats(*con);
  auto gpu2 = con->Query(q);
  REQUIRE(gpu2);
  REQUIRE_FALSE(gpu2->HasError());
  after = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 0);

  // The fallback result matches the GPU result.
  REQUIRE(gpu->Cast<duckdb::MaterializedQueryResult>().GetValue(0, 0).ToString() ==
          gpu2->Cast<duckdb::MaterializedQueryResult>().GetValue(0, 0).ToString());
}

// The CPU fallback runs in the SAME transaction as the failed GPU attempt, so it
// sees this transaction's own uncommitted writes. A fresh-connection replay could
// not (it would start its own transaction) — this is the core MVCC requirement.
TEST_CASE_METHOD(RuntimeFallbackFixture,
                 "runtime fallback: sees own uncommitted writes",
                 "[transparent][fallback][mvcc][integration]")
{
  create_table("CREATE TABLE rf_mvcc AS SELECT i AS id FROM range(100) t(i);");
  inject();

  con->Query("BEGIN TRANSACTION;");
  con->Query("INSERT INTO rf_mvcc VALUES (100);");  // uncommitted, this transaction
  // The SELECT falls back to CPU, but under the same transaction — must see 101.
  REQUIRE(scalar(*con, "SELECT count(*) FROM rf_mvcc;") == "101");
  con->Query("ROLLBACK;");

  // After rollback the uncommitted row is gone.
  REQUIRE(scalar(*con, "SELECT count(*) FROM rf_mvcc;") == "100");
}

// The CPU fallback runs under the failed attempt's snapshot: a concurrent commit
// made after this transaction pinned its snapshot must stay invisible.
TEST_CASE_METHOD(RuntimeFallbackFixture,
                 "runtime fallback: snapshot is stable across a concurrent commit",
                 "[transparent][fallback][mvcc][integration]")
{
  create_table("CREATE TABLE rf_snap AS SELECT i AS id FROM range(100) t(i);");
  auto other = make_connection();

  inject(*con);
  con->Query("BEGIN TRANSACTION;");
  // First read pins this transaction's snapshot at 100 rows (falls back to CPU).
  REQUIRE(scalar(*con, "SELECT count(*) FROM rf_snap;") == "100");

  // A different connection inserts and commits (qualified: it has not USE'd alias_).
  auto ins = other->Query("INSERT INTO " + alias_ + ".rf_snap VALUES (1000);");
  REQUIRE_FALSE(ins->HasError());

  // The in-transaction read still sees the pinned snapshot, not the new commit.
  REQUIRE(scalar(*con, "SELECT count(*) FROM rf_snap;") == "100");
  con->Query("COMMIT;");

  // A fresh statement sees the committed row.
  REQUIRE(scalar(*con, "SELECT count(*) FROM rf_snap;") == "101");
}

// With enable_duckdb_fallback = false, a runtime GPU failure surfaces as a query
// error instead of falling back — and, critically, does NOT invalidate the
// database/session (a bare InternalException would have).
TEST_CASE_METHOD(RuntimeFallbackFixture,
                 "runtime fallback: disabled surfaces error without invalidating session",
                 "[transparent][fallback][integration]")
{
  create_table("CREATE TABLE rf_off AS SELECT i AS id FROM range(10) t(i);");
  con->Query("SET enable_duckdb_fallback = false;");
  inject("boom-off");

  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto err    = con->Query("SELECT count(*) FROM rf_off;");
  REQUIRE(err);
  REQUIRE(err->HasError());
  REQUIRE(err->GetError().find("boom-off") != std::string::npos);
  auto after = sirius::test::get_transparent_execution_stats(*con);
  // GPU was attempted (rebind + execution) but did not fall back.
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 0);

  // The session is still usable: a subsequent query succeeds.
  clear_injection();
  con->Query("SET enable_duckdb_fallback = true;");
  REQUIRE(scalar(*con, "SELECT count(*) FROM rf_off;") == "10");
}

// enable_duckdb_fallback also gates plan-time fallback: an unsupported operator
// (window function) errors when fallback is off, and silently runs on CPU when on.
TEST_CASE_METHOD(RuntimeFallbackFixture,
                 "runtime fallback: plan-time fallback is gated by the setting",
                 "[transparent][fallback][integration]")
{
  create_table("CREATE TABLE rf_plan AS SELECT i AS id, i % 5 AS grp FROM range(100) t(i);");
  const std::string window =
    "SELECT id, grp, ROW_NUMBER() OVER (PARTITION BY grp ORDER BY id) AS rn FROM rf_plan;";

  // Off: unsupported operator surfaces an error instead of silently using CPU.
  con->Query("SET enable_duckdb_fallback = false;");
  auto err = con->Query(window);
  REQUIRE(err);
  REQUIRE(err->HasError());
  // Session stays usable.
  REQUIRE(scalar(*con, "SELECT count(*) FROM rf_plan;") == "100");

  // On: the same query silently falls back at plan time (plan-time fallback + 1).
  con->Query("SET enable_duckdb_fallback = true;");
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto ok     = con->Query(window);
  REQUIRE(ok);
  REQUIRE_FALSE(ok->HasError());
  REQUIRE(ok->RowCount() == 100);
  auto after = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0, 0);
}

// Note: SQL-level `PREPARE ... AS SELECT` / `EXECUTE` is not intercepted by Sirius
// (transparent interception gates on statement_type == SELECT_STATEMENT, and those
// carry PREPARE/EXECUTE statement types), so it runs on DuckDB CPU and the runtime
// fallback path does not apply. Extending interception to prepared statements is a
// separate concern, out of scope for runtime fallback.

// The enable_duckdb_fallback setting defaults to true and is overridable per session.
TEST_CASE_METHOD(RuntimeFallbackFixture,
                 "runtime fallback: setting defaults true and is session-overridable",
                 "[transparent][fallback][integration]")
{
  duckdb::Value v;
  auto lr = con->context->TryGetCurrentSetting("enable_duckdb_fallback", v);
  REQUIRE(lr.GetScope() != duckdb::SettingScope::INVALID);
  REQUIRE_FALSE(v.IsNull());
  REQUIRE(v.GetValue<bool>() == true);

  con->Query("SET enable_duckdb_fallback = false;");
  con->context->TryGetCurrentSetting("enable_duckdb_fallback", v);
  REQUIRE(v.GetValue<bool>() == false);

  con->Query("SET enable_duckdb_fallback = true;");
  con->context->TryGetCurrentSetting("enable_duckdb_fallback", v);
  REQUIRE(v.GetValue<bool>() == true);
}
