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
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>  // getpid
#include <util/duckdb_error_message.hpp>
#include <utils/child_runner.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

extern char** environ;

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

namespace {

constexpr char kS3MixChildCase[]        = "S3 mix fallback child runner";
constexpr char kS3MixScenarioEnv[]      = "SIRIUS_S3MIX_CHILD_SCENARIO";
constexpr char kRuntimeFallbackBanner[] = "Error in Sirius GPU execution, fallback to DuckDB";

std::string sql_literal(std::string const& value)
{
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('\'');
  for (auto const c : value) {
    if (c == '\'') { out.push_back('\''); }
    out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

class S3MixFixture : public sirius::test::GpuExecutionFixture {
 public:
  S3MixFixture()
  {
    static std::atomic<std::uint64_t> next_id{0};
    work_dir = fs::temp_directory_path() / ("sirius_s3mix_" + std::to_string(::getpid()) + "_" +
                                            std::to_string(next_id.fetch_add(1)));
    fs::create_directories(work_dir);
    parquet_path = work_dir / "nested.parquet";
    empty_path   = work_dir / "empty.parquet";

    std::string const source =
      "SELECT CAST(i - 100 AS BIGINT) AS id, "
      "CAST(i - 100 AS BIGINT) AS a, CAST(i AS BIGINT) AS b, "
      "CASE WHEN i % 7 = 0 THEN NULL ELSE CAST(i - 100 AS BIGINT) END AS x, "
      "struct_pack(a := i, b := i * 10) AS st, "
      "[i, i + 1] AS li, MAP([1, 2], [i, i + 1]) AS mp "
      "FROM range(300) t(i)";

    run_ok("SET gpu_execution = false;");
    run_ok("COPY (" + source + ") TO " + sql_literal(parquet_path.string()) + " (FORMAT PARQUET);");
    run_ok("COPY (SELECT * FROM (" + source + ") empty_source WHERE false) TO " +
           sql_literal(empty_path.string()) + " (FORMAT PARQUET);");
    run_ok("CREATE TABLE mix_native AS " + source + ";");
    run_ok("CHECKPOINT;");

    auto const partition_dir = work_dir / "hive" / "part=1";
    fs::create_directories(partition_dir);
    fs::copy_file(
      parquet_path, partition_dir / "data.parquet", fs::copy_options::overwrite_existing);
    run_ok("SET gpu_execution = true;");
  }

  ~S3MixFixture() { fs::remove_all(work_dir); }

  std::string parquet_scan() const
  {
    return "read_parquet(" + sql_literal(parquet_path.string()) + ")";
  }

  std::string empty_scan() const
  {
    return "read_parquet(" + sql_literal(empty_path.string()) + ")";
  }

  std::string hive_scan() const
  {
    return "read_parquet(" + sql_literal((work_dir / "hive" / "*" / "*.parquet").string()) +
           ", hive_partitioning=true)";
  }

  std::vector<std::vector<std::string>> require_query_matches_cpu(
    std::string const& query,
    std::uint64_t expected_rebinds,
    std::uint64_t expected_fallbacks,
    std::uint64_t expected_executions,
    std::optional<std::string> expected_first_value = std::nullopt,
    std::uint64_t expected_runtime_fallbacks        = 0)
  {
    UNSCOPED_INFO("query: " << query);
    run_ok("SET gpu_execution = false;");
    auto cpu = con->Query(query);
    REQUIRE(cpu);
    if (cpu->HasError()) { UNSCOPED_INFO("CPU query error: " << cpu->GetError()); }
    REQUIRE_FALSE(cpu->HasError());

    run_ok("SET gpu_execution = true;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu          = con->Query(query);
    REQUIRE(gpu);
    if (gpu->HasError()) { UNSCOPED_INFO("GPU/fallback query error: " << gpu->GetError()); }
    REQUIRE_FALSE(gpu->HasError());
    auto const after = sirius::test::get_transparent_execution_stats(*con);
    INFO("query: " << query);
    REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks + expected_runtime_fallbacks);
    sirius::test::require_transparent_execution_delta(before,
                                                      after,
                                                      expected_rebinds,
                                                      expected_fallbacks,
                                                      expected_executions,
                                                      expected_runtime_fallbacks);

    REQUIRE(gpu->ColumnCount() == cpu->ColumnCount());
    REQUIRE(gpu->RowCount() == cpu->RowCount());
    if (expected_first_value.has_value()) {
      REQUIRE(gpu->RowCount() > 0);
      CHECK(gpu->GetValue(0, 0).ToString() == *expected_first_value);
    }

    auto gpu_rows =
      sirius::test::GpuExecutionFixture::collect_rows(gpu->Cast<duckdb::MaterializedQueryResult>());
    auto cpu_rows =
      sirius::test::GpuExecutionFixture::collect_rows(cpu->Cast<duckdb::MaterializedQueryResult>());
    CHECK(gpu_rows == cpu_rows);
    return gpu_rows;
  }

  void require_plan_fallback(std::string const& query,
                             std::optional<std::string> expected_first_value = std::nullopt)
  {
    (void)require_query_matches_cpu(query, 0, 1, 0, std::move(expected_first_value));
  }

  void require_gpu(std::string const& query,
                   std::optional<std::string> expected_first_value = std::nullopt)
  {
    (void)require_query_matches_cpu(query, 1, 0, 1, std::move(expected_first_value));
  }

 private:
  fs::path work_dir;
  fs::path parquet_path;
  fs::path empty_path;
};

void run_s3mix_scenario(std::string const& scenario)
{
  S3MixFixture fixture;
  auto const parquet = fixture.parquet_scan();

  if (scenario == "projection_round") {
    fixture.require_plan_fallback("SELECT round(id) FROM " + parquet + " LIMIT 3", "-100");
  } else if (scenario == "projection_struct_extract") {
    fixture.require_plan_fallback("SELECT st.a FROM " + parquet + " LIMIT 3", "0");
  } else if (scenario == "projection_list_extract") {
    fixture.require_plan_fallback("SELECT li[1] FROM " + parquet + " LIMIT 3", "0");
  } else if (scenario == "order_round") {
    fixture.require_plan_fallback("SELECT id FROM " + parquet + " ORDER BY round(id) LIMIT 3",
                                  "-100");
  } else if (scenario == "topn_round") {
    fixture.require_plan_fallback("SELECT id FROM " + parquet + " ORDER BY round(id) DESC LIMIT 3",
                                  "199");
  } else if (scenario == "aggregate_child_projection") {
    fixture.require_plan_fallback(
      "SELECT r, count(*) FROM "
      "(SELECT round(a) AS r, b FROM " +
      parquet + ") q GROUP BY r");
  } else if (scenario == "join_round") {
    fixture.require_plan_fallback(
      "SELECT count(*) FROM " + parquet + " a JOIN " + parquet + " b ON round(a.id) = round(b.id)",
      "300");
  } else if (scenario == "regression_bundle") {
    fixture.require_gpu("SELECT st FROM " + parquet);
    fixture.require_gpu("SELECT li FROM " + parquet);

    fixture.require_plan_fallback("SELECT count(*) FROM " + parquet + " WHERE round(id) > 1",
                                  "198");
    fixture.require_plan_fallback("SELECT count(*) FROM mix_native WHERE round(id) > 1", "198");
    fixture.require_gpu("SELECT count(id) FROM " + fixture.hive_scan() + " WHERE part = 1", "300");

    fixture.require_gpu("SELECT count(st) FROM " + parquet, "300");
    fixture.require_gpu("SELECT count(li) FROM " + parquet, "300");
    fixture.require_plan_fallback("SELECT count(round(x)) FROM " + parquet, "257");
  } else if (scenario == "parquet_is_not_null") {
    fixture.require_gpu("SELECT count(*) FROM " + parquet + " WHERE x IS NOT NULL", "257");
  } else if (scenario == "native_is_not_null") {
    fixture.require_gpu("SELECT count(*) FROM mix_native WHERE x IS NOT NULL", "257");
  } else if (scenario == "nested_sort_regression") {
    for (auto const* key : {"st", "li", "mp"}) {
      fixture.require_plan_fallback("SELECT id FROM " + parquet + " ORDER BY " + key);
      fixture.require_plan_fallback("SELECT id FROM " + parquet + " ORDER BY " + key + " LIMIT 5");
    }
    fixture.require_gpu("SELECT id FROM " + parquet + " ORDER BY id LIMIT 5", "-100");
  } else if (scenario == "zero_limit") {
    fixture.require_gpu("SELECT round(id) FROM " + parquet + " LIMIT 0");
  } else if (scenario == "zero_empty") {
    fixture.require_plan_fallback("SELECT round(id) FROM " + fixture.empty_scan());
  } else if (scenario == "zero_stats_pruned") {
    fixture.require_gpu("SELECT round(id) FROM " + parquet + " WHERE id > 10000");
  } else {
    FAIL("unknown S3 mix child scenario: " << scenario);
  }
}

struct child_result {
  bool timed_out{false};
  bool exited{false};
  int exit_code{-1};
  int signal{-1};
  std::string output;
};

child_result run_s3mix_child(std::string const& scenario, std::chrono::seconds timeout)
{
  if (sirius::test::g_integration_env != nullptr && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }

  static std::atomic<std::uint64_t> next_id{0};
  auto const output_path =
    fs::temp_directory_path() / ("sirius_s3mix_child_" + std::to_string(::getpid()) + "_" +
                                 std::to_string(next_id.fetch_add(1)) + ".log");
  int const fd = ::open(output_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
  REQUIRE(fd >= 0);

  std::vector<std::string> child_args{"sirius_unittest", kS3MixChildCase, "--reporter", "compact"};
  std::vector<char*> child_argv;
  child_argv.reserve(child_args.size() + 1);
  for (auto& arg : child_args) {
    child_argv.push_back(arg.data());
  }
  child_argv.push_back(nullptr);

  auto const scenario_prefix     = std::string{kS3MixScenarioEnv} + "=";
  auto const child_runner_prefix = std::string{sirius::test::child_runner_env} + "=";
  std::vector<std::string> child_environment;
  for (auto entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
    std::string value{*entry};
    if (value.rfind(scenario_prefix, 0) != 0 && value.rfind(child_runner_prefix, 0) != 0) {
      child_environment.push_back(std::move(value));
    }
  }
  child_environment.push_back(scenario_prefix + scenario);
  child_environment.push_back(child_runner_prefix + sirius::test::child_runner_env_value);

  std::vector<char*> child_envp;
  child_envp.reserve(child_environment.size() + 1);
  for (auto& entry : child_environment) {
    child_envp.push_back(entry.data());
  }
  child_envp.push_back(nullptr);

  char* const* child_argv_ptr = child_argv.data();
  char* const* child_envp_ptr = child_envp.data();
  auto const pid              = ::fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    (void)::dup2(fd, STDOUT_FILENO);
    (void)::dup2(fd, STDERR_FILENO);
    (void)::close(fd);
    ::execve("/proc/self/exe", child_argv_ptr, child_envp_ptr);
    ::_exit(127);
  }
  (void)::close(fd);

  int status       = 0;
  auto const stop  = std::chrono::steady_clock::now() + timeout;
  bool timed_out   = true;
  bool wait_failed = false;
  while (std::chrono::steady_clock::now() < stop) {
    auto const waited = ::waitpid(pid, &status, WNOHANG);
    if (waited == pid) {
      timed_out = false;
      break;
    }
    if (waited < 0 && errno != EINTR) {
      timed_out   = false;
      wait_failed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
  }
  if (timed_out) {
    (void)::kill(pid, SIGKILL);
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  }

  std::ifstream input(output_path);
  std::ostringstream text;
  text << input.rdbuf();
  std::error_code ec;
  fs::remove(output_path, ec);

  child_result result;
  result.timed_out = timed_out;
  result.exited    = !timed_out && !wait_failed && WIFEXITED(status);
  result.exit_code = result.exited ? WEXITSTATUS(status) : -1;
  result.signal    = !timed_out && !wait_failed && WIFSIGNALED(status) ? WTERMSIG(status) : -1;
  result.output    = text.str();
  return result;
}

void require_s3mix_child_survives(std::string const& scenario,
                                  bool forbid_runtime_banner         = false,
                                  std::chrono::seconds const timeout = std::chrono::seconds{90})
{
  auto result = run_s3mix_child(scenario, timeout);
  INFO("scenario=" << scenario);
  INFO("child output:\n" << result.output);
  CHECK_FALSE(result.timed_out);
  CHECK(result.signal == -1);
  REQUIRE(result.exited);
  CHECK(result.exit_code == 0);
  if (forbid_runtime_banner) {
    CHECK(result.output.find(kRuntimeFallbackBanner) == std::string::npos);
  }
}

}  // namespace

TEST_CASE("S3 mix fallback child runner", "[.][transparent][integration][s3mix_child]")
{
  REQUIRE(sirius::test::g_shared_env == nullptr);
  REQUIRE(sirius::test::g_integration_env == nullptr);
  REQUIRE(sirius::test::g_integration_env_2gpu == nullptr);

  auto const* scenario = std::getenv(kS3MixScenarioEnv);
  if (scenario == nullptr) { return; }
  run_s3mix_scenario(scenario);
}

TEST_CASE("unsupported parquet round projection falls back without killing the process",
          "[transparent][fallback][integration][s3mix]")
{
  require_s3mix_child_survives("projection_round");
}

TEST_CASE("unsupported parquet struct extraction falls back without killing the process",
          "[transparent][fallback][integration][s3mix]")
{
  require_s3mix_child_survives("projection_struct_extract");
}

TEST_CASE("unsupported parquet list extraction falls back without killing the process",
          "[transparent][fallback][integration][s3mix]")
{
  require_s3mix_child_survives("projection_list_extract");
}

TEST_CASE("unsupported ORDER BY expression falls back without killing the process",
          "[transparent][fallback][integration][s3mix]")
{
  require_s3mix_child_survives("order_round");
}

TEST_CASE("unsupported TOP-N expression falls back without killing the process",
          "[transparent][fallback][integration][s3mix]")
{
  require_s3mix_child_survives("topn_round");
}

TEST_CASE("unsupported aggregate child projection falls back without killing the process",
          "[transparent][fallback][integration][s3mix]")
{
  require_s3mix_child_survives("aggregate_child_projection");
}

TEST_CASE("unsupported join expressions fall back without killing the process",
          "[transparent][fallback][integration][s3mix]")
{
  require_s3mix_child_survives("join_round");
}

TEST_CASE("S3 mix fallback guards preserve supported nested and aggregate behavior",
          "[transparent][fallback][integration][s3mix]")
{
  require_s3mix_child_survives("regression_bundle");
}

TEST_CASE("an unsupported filter above a join falls back during planning",
          "[transparent][fallback][integration][s3mix][filter]")
{
  S3MixFixture fixture;
  auto const parquet = fixture.parquet_scan();

  fixture.require_plan_fallback("SELECT count(*) FROM " + parquet + " a JOIN " + parquet +
                                  " b ON a.id = b.id WHERE round(a.a + b.b) > 1",
                                "249");
  fixture.require_gpu("SELECT count(*) FROM " + parquet + " a JOIN " + parquet +
                        " b ON a.id = b.id WHERE a.a + b.b > 1",
                      "249");
}

TEST_CASE("supported IS NOT NULL filter stays on GPU for parquet",
          "[transparent][fallback][integration][s3mix][filter]")
{
  require_s3mix_child_survives("parquet_is_not_null");
}

TEST_CASE("supported IS NOT NULL filter stays on GPU for native storage",
          "[transparent][fallback][integration][s3mix][filter]")
{
  require_s3mix_child_survives("native_is_not_null");
}

TEST_CASE("nested sort keys fall back during planning rather than execution",
          "[transparent][fallback][integration][s3mix]")
{
  require_s3mix_child_survives("nested_sort_regression", true);
}

TEST_CASE("LIMIT zero is optimized to an empty result that stays on GPU",
          "[transparent][fallback][integration][s3mix][zero-input]")
{
  require_s3mix_child_survives("zero_limit", false, std::chrono::seconds{30});
}

TEST_CASE("unsupported projection over an empty parquet falls back during planning",
          "[transparent][fallback][integration][s3mix][zero-input]")
{
  require_s3mix_child_survives("zero_empty", false, std::chrono::seconds{30});
}

TEST_CASE("a stats-pruned scan is optimized to an empty result that stays on GPU",
          "[transparent][fallback][integration][s3mix][zero-input]")
{
  require_s3mix_child_survives("zero_stats_pruned", false, std::chrono::seconds{30});
}

TEST_CASE_METHOD(RuntimeFallbackFixture,
                 "unsupported table-function diagnostics include the function name",
                 "[transparent][fallback][integration][s3mix][diagnostics]")
{
  auto const csv_path =
    fs::temp_directory_path() / ("sirius_s3mix_" + std::to_string(::getpid()) + ".csv");
  {
    std::ofstream csv(csv_path);
    csv << "id\n1\n";
  }

  auto const inner = "SELECT * FROM read_csv_auto(" + sql_literal(csv_path.string()) + ")";
  std::string escaped;
  escaped.reserve(inner.size());
  for (auto const c : inner) {
    if (c == '\'') { escaped.push_back('\''); }
    escaped.push_back(c);
  }
  auto disable_fallback = con->Query("SET enable_duckdb_fallback = false;");
  REQUIRE(disable_fallback);
  REQUIRE_FALSE(disable_fallback->HasError());
  auto disable_transparent = con->Query("SET gpu_execution = false;");
  REQUIRE(disable_transparent);
  REQUIRE_FALSE(disable_transparent->HasError());
  auto result = con->Query("SELECT * FROM gpu_execution('" + escaped + "')");
  std::error_code ec;
  fs::remove(csv_path, ec);

  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(result->GetError().find("Table function 'read_csv_auto' is not supported in Sirius") !=
        std::string::npos);
}

TEST_CASE("DuckDB diagnostic sanitizer preserves invalid JSON-like messages",
          "[transparent][fallback][s3mix][diagnostics]")
{
  std::runtime_error error("{not valid JSON");
  CHECK(sirius::sanitized_message(error) == error.what());
}

TEST_CASE("DuckDB diagnostic sanitizer preserves valid non-envelope JSON messages",
          "[transparent][fallback][s3mix][diagnostics]")
{
  std::runtime_error error(R"({"error":"missing"})");
  CHECK(sirius::sanitized_message(error) == error.what());
}
