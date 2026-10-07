#include "log/logging.hpp"
#include "util/env_guard.hpp"
#include "utils/isolated_checkpoint_test.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

namespace {
constexpr auto key_sentinel    = "AKIAREDACTIONSENTINEL";
constexpr auto secret_sentinel = "sentinel-secret-value";
constexpr auto token_sentinel  = "sentinel-session-token";

using sentinel_set = std::array<char const*, 3>;
constexpr sentinel_set default_sentinels{key_sentinel, secret_sentinel, token_sentinel};
constexpr sentinel_set probe_sentinels{
  "AKIAREDACTIONPROBE", "pw-redaction-probe-value", "tok-redaction-probe"};

std::string credential_sql(std::string const& prefix     = "CREATE SECRET redaction_probe",
                           sentinel_set const& sentinels = default_sentinels)
{
  return prefix + " (TYPE SIRIUS_S3, KEY_ID '" + sentinels[0] + "', SECRET '" + sentinels[1] +
         "', SESSION_TOKEN '" + sentinels[2] + "', REGION 'us-east-1')";
}

void check_query_log(std::string const& sql,
                     bool redact                   = true,
                     bool must_succeed             = true,
                     sentinel_set const& sentinels = default_sentinels,
                     bool expect_original          = false)
{
  auto const name   = Catch::getResultCapture().getCurrentTestName();
  auto const* child = std::getenv("SIRIUS_NATIVE_LEASE_CHILD_CASE");
  if (child && name == child) {
    REQUIRE(sirius::test::g_integration_env);
    auto con = sirius::test::g_integration_env->make_connection();
    REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
    auto const secret_dir = std::filesystem::path(std::getenv("SIRIUS_LOG_DIR")) / "secrets";
    REQUIRE_FALSE(
      con.Query("SET secret_directory=" + sirius::test::sql_literal(secret_dir.string()))
        ->HasError());
    auto result = con.Query(sql);
    REQUIRE(result);
    if (must_succeed) { REQUIRE_FALSE(result->HasError()); }
    REQUIRE_FALSE(con.Query("SELECT 42")->HasError());
    REQUIRE(sirius::log::get_sink()->flush());
    return;
  }

  sirius::test::scratch_dir logs("query_redaction");
  sirius::util::env_guard log_dir("SIRIUS_LOG_DIR", logs.path().string());
  sirius::util::env_guard test_log_dir("SIRIUS_TEST_LOG_DIR", logs.path().string());
  sirius::util::env_guard backend("SIRIUS_LOG_BACKEND", "spdlog");
  sirius::util::env_guard level("SIRIUS_LOG_LEVEL", "info");
  auto result = sirius::test::run_test_child(name);
  INFO(result.output);
  REQUIRE_FALSE(result.timed_out);
  REQUIRE(result.signal == -1);
  REQUIRE(result.exit_code == 0);

  std::regex const begin(R"(QueryBegin: instance=\S+ connection=\d+ query=\d+ SQL: (.*)$)");
  bool placeholder    = false;
  bool select_logged  = false;
  bool visible_logged = false;
  std::size_t files   = 0;
  for (auto const& entry : std::filesystem::directory_iterator(logs.path())) {
    if (!entry.is_regular_file()) { continue; }
    ++files;
    std::ifstream input(entry.path());
    REQUIRE(input.good());
    std::string line;
    while (std::getline(input, line)) {
      INFO("captured log: " << line);
      for (auto const* sentinel : sentinels) {
        CHECK(line.find(sentinel) == std::string::npos);
      }
      std::smatch match;
      if (std::regex_search(line, match, begin)) {
        placeholder |= match[1].str() == "<credential statement redacted>";
        select_logged |= match[1].str() == "SELECT 42";
        visible_logged |= match[1].str() == sql;
      }
    }
  }
  REQUIRE(files > 0);
  CHECK(select_logged);
  if (redact) { CHECK(placeholder); }
  if (expect_original || sql == "SELECT 'visible-sentinel'") { CHECK(visible_logged); }
}
}  // namespace

TEST_CASE("QueryBegin redacts CREATE SECRET statements", "[integration][logging][redaction]")
{
  for (auto const* sentinel : probe_sentinels) {
    REQUIRE_FALSE(std::regex_search(sentinel, std::regex("secret", std::regex::icase)));
  }
  check_query_log(
    credential_sql("CREATE SECRET redaction_probe", probe_sentinels), true, true, probe_sentinels);
}

TEST_CASE("QueryBegin redacts a CREATE SECRET hidden behind a line comment",
          "[integration][logging][redaction]")
{
  check_query_log("-- note\n" + credential_sql(), true);
}

TEST_CASE("QueryBegin redacts replacement temporary secrets", "[integration][logging][redaction]")
{
  check_query_log(credential_sql("CREATE OR REPLACE TEMPORARY SECRET redaction_probe"), true);
}

TEST_CASE("QueryBegin redacts persistent secrets", "[integration][logging][redaction]")
{
  check_query_log(credential_sql("CREATE PERSISTENT SECRET redaction_probe"), true);
}

TEST_CASE("QueryBegin redacts default named secrets", "[integration][logging][redaction]")
{
  check_query_log(credential_sql("CREATE SECRET"), true);
}

TEST_CASE("QueryBegin redacts credential statements in a batch",
          "[integration][logging][redaction]")
{
  check_query_log("SELECT 7; " + credential_sql() + "; SELECT 9", true);
}

TEST_CASE("QueryBegin redacts credential SQL inside gpu_execution",
          "[integration][logging][redaction]")
{
  check_query_log(
    "SELECT * FROM gpu_execution(" + sirius::test::sql_literal(credential_sql()) + ")",
    true,
    false);
}

TEST_CASE("QueryBegin logs non-secret statements unchanged", "[integration][logging][redaction]")
{
  check_query_log("SELECT 'visible-sentinel'", false);
}

TEST_CASE("Malformed credential SQL is not echoed to the file log",
          "[integration][logging][redaction]")
{
  check_query_log(credential_sql() + " SELECT", false, false);
}

TEST_CASE("QueryBegin redacts credential SQL assembled by concatenation inside gpu_execution",
          "[integration][logging][redaction]")
{
  std::string const sql =
    "SELECT * FROM gpu_execution('CREATE SE' || 'CRET probe (TYPE SIRIUS_S3, KEY_ID ''" +
    std::string(probe_sentinels[0]) + "'', SE' || 'CRET ''" + probe_sentinels[1] +
    "'', SESSION_TOKEN ''" + probe_sentinels[2] + "'', REGION ''us-east-1'')')";
  REQUIRE_FALSE(std::regex_search(sql, std::regex("secret", std::regex::icase)));
  check_query_log(sql, true, false, probe_sentinels);
}

TEST_CASE("QueryBegin redacts gpu_execution with a non-literal argument",
          "[integration][logging][redaction]")
{
  std::string const sql = "SELECT * FROM gpu_execution(concat('SEL', 'ECT 42'))";
  REQUIRE_FALSE(std::regex_search(sql, std::regex("secret", std::regex::icase)));
  check_query_log(sql, true, true, probe_sentinels);
}

TEST_CASE("QueryBegin logs gpu_execution with a plain literal argument unchanged",
          "[integration][logging][redaction]")
{
  check_query_log("SELECT * FROM gpu_execution('SELECT 42')", false, true, probe_sentinels, true);
}
