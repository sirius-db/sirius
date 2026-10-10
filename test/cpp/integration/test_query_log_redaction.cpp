#include "log/logging.hpp"
#include "sirius_context.hpp"
#include "util/env_guard.hpp"
#include "utils/isolated_checkpoint_test.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

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

class query_begin_fault_sink final : public sirius::log::sink {
 public:
  struct fault_observation {
    std::uint64_t ordinal     = 0;
    unsigned probes           = 0;
    unsigned emissions        = 0;
    unsigned throws           = 0;
    bool internal             = false;
    bool query_begin_location = false;
  };

  query_begin_fault_sink(std::shared_ptr<sirius::log::sink> downstream,
                         duckdb::ClientContext& client,
                         duckdb::shared_ptr<duckdb::SiriusConnectionState> state,
                         bool throw_from_log)
    : downstream_(std::move(downstream)),
      client_(client),
      state_(std::move(state)),
      owner_(std::this_thread::get_id()),
      throw_from_log_(throw_from_log)
  {
  }

  void set_level(sirius::log::level level) override { downstream_->set_level(level); }

  bool should_log(sirius::log::level level) const override
  {
    if (matches(level)) {
      ++observed.probes;
      if (!throw_from_log_) { inject(); }
    }
    return downstream_->should_log(level);
  }

  void log(sirius::log::level level,
           std::source_location const& location,
           std::string_view message) override
  {
    if (message.starts_with("QueryBegin:") && matches(level)) {
      ++observed.emissions;
      observed.query_begin_location =
        std::string_view(location.file_name()).ends_with("sirius_context.cpp") &&
        std::string_view(location.function_name()).find("QueryBegin") != std::string_view::npos;
      if (throw_from_log_) { inject(); }
    }
    downstream_->log(level, location, message);
  }

  bool flush() override { return downstream_->flush(); }

  void arm(std::string sql)
  {
    expected_sql_     = std::move(sql);
    expected_ordinal_ = state_->current_query_ordinal() + 1;
    observed          = {};
    armed_.store(true, std::memory_order_release);
  }

  bool armed() const { return armed_.load(std::memory_order_acquire); }

  mutable fault_observation observed;

 private:
  bool matches(sirius::log::level level) const
  {
    return level == sirius::log::level::info && std::this_thread::get_id() == owner_ && armed() &&
           state_->current_query_ordinal() == expected_ordinal_ &&
           !state_->is_internal_query_active() && client_.GetCurrentQuery() == expected_sql_;
  }

  void inject() const
  {
    if (!armed_.exchange(false, std::memory_order_acq_rel)) { return; }
    observed.ordinal  = state_->current_query_ordinal();
    observed.internal = state_->is_internal_query_active();
    ++observed.throws;
    throw std::runtime_error("injected QueryBegin observation failure");
  }

  std::shared_ptr<sirius::log::sink> downstream_;
  duckdb::ClientContext& client_;
  duckdb::shared_ptr<duckdb::SiriusConnectionState> state_;
  std::thread::id const owner_;
  bool const throw_from_log_;
  std::string expected_sql_;
  std::uint64_t expected_ordinal_ = 0;
  mutable std::atomic<bool> armed_{false};
};

struct sink_restorer {
  std::shared_ptr<sirius::log::sink> previous = sirius::log::get_sink();
  ~sink_restorer()
  {
    try {
      sirius::log::set_sink(previous);
    } catch (...) {
    }
  }
};

void check_query_begin_fault(bool throw_from_log)
{
  auto const name   = Catch::getResultCapture().getCurrentTestName();
  auto const* child = std::getenv("SIRIUS_NATIVE_LEASE_CHILD_CASE");
  if (child && name == child) {
    REQUIRE(sirius::test::g_integration_env);
    auto con   = sirius::test::g_integration_env->make_connection();
    auto state = duckdb::get_sirius_connection_state(*con.context);
    REQUIRE(state);
    auto run = [&](std::string const& sql) {
      auto result = con.Query(sql);
      REQUIRE(result);
      INFO((result->HasError() ? result->GetError() : ""));
      REQUIRE_FALSE(result->HasError());
      return result;
    };
    run("SELECT 41 AS probe_before");
    auto const before = state->current_query_ordinal();
    auto previous     = sirius::log::get_sink();
    REQUIRE(previous->should_log(sirius::log::level::info));
    {
      sink_restorer restore;
      auto probe =
        std::make_shared<query_begin_fault_sink>(previous, *con.context, state, throw_from_log);
      sirius::log::set_sink(probe);
      std::array<std::string, 2> const statements{
        "SELECT 1", credential_sql("CREATE SECRET observation_probe", probe_sentinels)};
      for (std::size_t i = 0; i < statements.size(); ++i) {
        probe->arm(statements[i]);
        CHECK(probe->should_log(sirius::log::level::info));
        REQUIRE(probe->armed());
        bool forwarded = false;
        std::exception_ptr other_error;
        std::thread other([&] {
          try {
            forwarded = probe->should_log(sirius::log::level::info);
          } catch (...) {
            other_error = std::current_exception();
          }
        });
        other.join();
        REQUIRE_FALSE(other_error);
        CHECK(forwarded);
        REQUIRE(probe->armed());
        auto result = run(statements[i]);
        if (i == 0) {
          REQUIRE(result->RowCount() == 1);
          CHECK(result->GetValue(0, 0).ToString() == "1");
        }
        REQUIRE_FALSE(probe->armed());
        CHECK(probe->observed.throws == 1);
        CHECK(probe->observed.ordinal == before + i + 1);
        CHECK_FALSE(probe->observed.internal);
        CHECK(state->current_query_ordinal() == before + i + 1);
        CHECK(probe->observed.emissions == (throw_from_log ? 1 : 0));
        if (throw_from_log) {
          CHECK(probe->observed.query_begin_location);
        } else {
          CHECK(probe->observed.probes == 1);
        }
      }
    }
    CHECK(sirius::log::get_sink() == previous);
    auto result = run("SELECT 42 AS probe_after");
    REQUIRE(result->RowCount() == 1);
    CHECK(result->GetValue(0, 0).ToString() == "42");
    CHECK(state->current_query_ordinal() == before + 3);
    REQUIRE(previous->flush());
    return;
  }

  sirius::test::scratch_dir logs("query_begin_fault");
  sirius::util::env_guard log_dir("SIRIUS_LOG_DIR", logs.path().string());
  sirius::util::env_guard test_log_dir("SIRIUS_TEST_LOG_DIR", logs.path().string());
  sirius::util::env_guard backend("SIRIUS_LOG_BACKEND", "spdlog");
  sirius::util::env_guard level("SIRIUS_LOG_LEVEL", "info");
  auto result = sirius::test::run_test_child(name);
  INFO(result.output);
  REQUIRE_FALSE(result.timed_out);
  REQUIRE(result.signal == -1);
  REQUIRE(result.exit_code == 0);

  using query_key = std::tuple<std::string, std::uint64_t, std::uint64_t>;
  std::vector<query_key> before_keys, after_keys, all_keys;
  std::regex const begin(R"(QueryBegin: instance=(\S+) connection=(\d+) query=(\d+) SQL: (.*)$)");
  std::size_t files = 0;
  for (auto const& entry : std::filesystem::directory_iterator(logs.path())) {
    if (!entry.is_regular_file()) { continue; }
    ++files;
    std::ifstream input(entry.path());
    REQUIRE(input.good());
    std::string line;
    while (std::getline(input, line)) {
      INFO("captured log: " << line);
      for (auto const* sentinel : probe_sentinels) {
        CHECK(line.find(sentinel) == std::string::npos);
      }
      std::smatch match;
      if (!std::regex_search(line, match, begin)) { continue; }
      query_key key{match[1].str(), std::stoull(match[2].str()), std::stoull(match[3].str())};
      all_keys.push_back(key);
      if (match[4].str() == "SELECT 41 AS probe_before") { before_keys.push_back(key); }
      if (match[4].str() == "SELECT 42 AS probe_after") { after_keys.push_back(key); }
    }
  }
  REQUIRE(files > 0);
  REQUIRE(before_keys.size() == 1);
  REQUIRE(after_keys.size() == 1);
  auto const& [instance, connection, ordinal] = before_keys.front();
  CHECK(std::get<0>(after_keys.front()) == instance);
  CHECK(std::get<1>(after_keys.front()) == connection);
  CHECK(std::get<2>(after_keys.front()) == ordinal + 3);
  for (auto const& [logged_instance, logged_connection, logged_ordinal] : all_keys) {
    if (logged_instance != instance || logged_connection != connection) { continue; }
    CHECK(logged_ordinal != ordinal + 1);
    CHECK(logged_ordinal != ordinal + 2);
  }
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

TEST_CASE("QueryBegin threshold faults preserve query success and ordinal progression",
          "[integration][logging][redaction][query_begin_fault]")
{
  check_query_begin_fault(false);
}

TEST_CASE("QueryBegin emission faults preserve query success and ordinal progression",
          "[integration][logging][redaction][query_begin_fault]")
{
  check_query_begin_fault(true);
}
