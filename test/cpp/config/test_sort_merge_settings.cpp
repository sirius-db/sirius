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

#include "catch.hpp"
#include "op/merge/merge_config.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "util/env_guard.hpp"
#include "utils/sirius_test_env.hpp"

#include <duckdb.hpp>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

using namespace sirius;

namespace {

constexpr uint64_t mib = 1024ULL * 1024;

/// A config file in the temp directory, removed on destruction.
class temp_config_file {
 public:
  explicit temp_config_file(std::string const& contents)
  {
    static std::atomic<int> counter{0};
    _path = std::filesystem::temp_directory_path() /
            ("sirius_sort_merge_settings_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter++) + ".yaml");
    std::ofstream out(_path);
    out << contents;
  }
  ~temp_config_file()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }
  temp_config_file(temp_config_file const&)            = delete;
  temp_config_file& operator=(temp_config_file const&) = delete;

  [[nodiscard]] std::filesystem::path const& path() const noexcept { return _path; }

 private:
  std::filesystem::path _path;
};

std::string operator_params_yaml(std::string const& body)
{
  return "sirius:\n  operator_params:\n" + body;
}

void require_ok(duckdb::Connection& con, std::string const& sql)
{
  INFO("sql=" << sql);
  auto result = con.Query(sql);
  REQUIRE(result != nullptr);
  REQUIRE_FALSE(result->HasError());
}

void require_rejected(duckdb::Connection& con, std::string const& sql, std::string const& message)
{
  INFO("sql=" << sql);
  auto result = con.Query(sql);
  REQUIRE(result != nullptr);
  REQUIRE(result->HasError());
  REQUIRE_THAT(result->GetError(), Catch::Matchers::ContainsSubstring(message));
}

duckdb::Value current_setting(duckdb::Connection& con, std::string const& name)
{
  INFO("setting=" << name);
  auto result = con.Query("SELECT current_setting('" + name + "')");
  REQUIRE(result != nullptr);
  REQUIRE_FALSE(result->HasError());
  return result->Collection().GetValue(0, 0);
}

}  // namespace

TEST_CASE("sort_merge settings default, SET and RESET", "[sort_merge_settings][isolated_context]")
{
  sirius::util::env_guard const disable_sirius{"SIRIUS_DISABLE", "1"};

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto const& params = duckdb::session_operator_params(*con.context);

  SECTION("defaults")
  {
    CHECK(current_setting(con, "sort_merge_mode").ToString() == "auto");
    CHECK(current_setting(con, "sort_merge_window_rows").GetValue<uint64_t>() == 0);
    CHECK(current_setting(con, "sort_merge_output_bytes").GetValue<uint64_t>() == 0);
    CHECK(current_setting(con, "sort_merge_one_shot_max_fraction").GetValue<double>() == 0.0);
    CHECK(current_setting(con, "sort_merge_max_fan_in").GetValue<uint64_t>() ==
          config::DEFAULT_SORT_MERGE_MAX_FAN_IN);
    CHECK(config::DEFAULT_SORT_MERGE_MAX_FAN_IN == 8);

    CHECK(params.sort_merge_mode_setting == sort_merge_mode::automatic);
    CHECK(params.sort_merge_window_rows == 0);
    CHECK(params.sort_merge_output_bytes == 0);
    CHECK(params.sort_merge_one_shot_max_fraction == 0.0);
    CHECK(params.sort_merge_max_fan_in == config::DEFAULT_SORT_MERGE_MAX_FAN_IN);
  }

  SECTION("SET reaches current_setting and the session operator_params")
  {
    require_ok(con, "SET sort_merge_mode = 'windowed'");
    require_ok(con, "SET sort_merge_window_rows = 1048576");
    require_ok(con, "SET sort_merge_output_bytes = 67108864");
    require_ok(con, "SET sort_merge_one_shot_max_fraction = 0.5");
    require_ok(con, "SET sort_merge_max_fan_in = 12");

    CHECK(current_setting(con, "sort_merge_mode").ToString() == "windowed");
    CHECK(current_setting(con, "sort_merge_window_rows").GetValue<uint64_t>() == 1048576);
    CHECK(current_setting(con, "sort_merge_output_bytes").GetValue<uint64_t>() == 64 * mib);
    CHECK(current_setting(con, "sort_merge_one_shot_max_fraction").GetValue<double>() ==
          Approx(0.5));
    CHECK(current_setting(con, "sort_merge_max_fan_in").GetValue<uint64_t>() == 12);

    CHECK(params.sort_merge_mode_setting == sort_merge_mode::windowed);
    CHECK(params.sort_merge_window_rows == 1048576);
    CHECK(params.sort_merge_output_bytes == 64 * mib);
    CHECK(params.sort_merge_one_shot_max_fraction == Approx(0.5));
    CHECK(params.sort_merge_max_fan_in == 12);

    // The merge operator reads the settings through this snapshot.
    auto const settings = op::merge::merge_settings::from(params);
    CHECK(settings.mode == sort_merge_mode::windowed);
    CHECK(settings.window_rows == 1048576);
    CHECK(settings.output_bytes == 64 * mib);
    CHECK(settings.one_shot_max_fraction == Approx(0.5));
    CHECK(settings.max_fan_in == 12);

    require_ok(con, "RESET sort_merge_mode");
    require_ok(con, "RESET sort_merge_window_rows");
    require_ok(con, "RESET sort_merge_output_bytes");
    require_ok(con, "RESET sort_merge_one_shot_max_fraction");
    require_ok(con, "RESET sort_merge_max_fan_in");

    CHECK(current_setting(con, "sort_merge_mode").ToString() == "auto");
    CHECK(params.sort_merge_mode_setting == sort_merge_mode::automatic);
    CHECK(params.sort_merge_window_rows == 0);
    CHECK(params.sort_merge_output_bytes == 0);
    CHECK(params.sort_merge_one_shot_max_fraction == 0.0);
    CHECK(params.sort_merge_max_fan_in == config::DEFAULT_SORT_MERGE_MAX_FAN_IN);
  }

  SECTION("boundary values are accepted")
  {
    require_ok(con, "SET sort_merge_window_rows = 1024");
    CHECK(params.sort_merge_window_rows == op::merge::kMinExplicitWindowRows);
    require_ok(con, "SET sort_merge_window_rows = 0");
    CHECK(params.sort_merge_window_rows == 0);
    require_ok(con, "SET sort_merge_output_bytes = 1048576");
    CHECK(params.sort_merge_output_bytes == 1 * mib);
    require_ok(con, "SET sort_merge_output_bytes = 0");
    CHECK(params.sort_merge_output_bytes == 0);
    require_ok(con, "SET sort_merge_one_shot_max_fraction = 1.0");
    CHECK(params.sort_merge_one_shot_max_fraction == 1.0);
    require_ok(con, "SET sort_merge_one_shot_max_fraction = 0");
    CHECK(params.sort_merge_one_shot_max_fraction == 0.0);
    require_ok(con, "SET sort_merge_max_fan_in = 2");
    CHECK(params.sort_merge_max_fan_in == 2);
  }

  SECTION("the mode is case-insensitive")
  {
    auto const [spelling, mode] = GENERATE(std::pair{"ONE_SHOT", sort_merge_mode::one_shot},
                                           std::pair{"Windowed", sort_merge_mode::windowed},
                                           std::pair{"AUTO", sort_merge_mode::automatic},
                                           std::pair{"one_shot", sort_merge_mode::one_shot});
    // Start from a mode other than the one being set.
    require_ok(con,
               mode == sort_merge_mode::windowed ? "SET sort_merge_mode = 'one_shot'"
                                                 : "SET sort_merge_mode = 'windowed'");
    require_ok(con, "SET sort_merge_mode = '" + std::string(spelling) + "'");
    CHECK(params.sort_merge_mode_setting == mode);
  }
}

TEST_CASE("sort_merge settings reject invalid SET values and keep the previous value",
          "[sort_merge_settings][isolated_context]")
{
  sirius::util::env_guard const disable_sirius{"SIRIUS_DISABLE", "1"};

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto const& params = duckdb::session_operator_params(*con.context);

  require_ok(con, "SET sort_merge_mode = 'windowed'");
  require_ok(con, "SET sort_merge_window_rows = 4096");
  require_ok(con, "SET sort_merge_output_bytes = 2097152");
  require_ok(con, "SET sort_merge_one_shot_max_fraction = 0.5");
  require_ok(con, "SET sort_merge_max_fan_in = 12");

  for (auto const* value : {"'oneshot'", "'window'", "''", "'auto '"}) {
    require_rejected(
      con, "SET sort_merge_mode = " + std::string(value), "Valid values: auto, windowed, one_shot");
  }
  for (auto const* value : {"1", "1023"}) {
    require_rejected(con,
                     "SET sort_merge_window_rows = " + std::string(value),
                     "sort_merge_window_rows must be 0 (derive) or at least 1024");
  }
  for (auto const* value : {"1", "1048575"}) {
    require_rejected(con,
                     "SET sort_merge_output_bytes = " + std::string(value),
                     "sort_merge_output_bytes must be 0 (derive) or at least 1 MiB");
  }
  for (auto const* value : {"-0.1", "1.01", "'NaN'", "'Infinity'"}) {
    require_rejected(con,
                     "SET sort_merge_one_shot_max_fraction = " + std::string(value),
                     "sort_merge_one_shot_max_fraction must be 0 (derive) or in (0, 1]");
  }
  for (auto const* value : {"0", "1"}) {
    require_rejected(con,
                     "SET sort_merge_max_fan_in = " + std::string(value),
                     "sort_merge_max_fan_in must be at least 2");
  }
  require_rejected(con, "SET GLOBAL sort_merge_mode = 'windowed'", "session settings");

  CHECK(params.sort_merge_mode_setting == sort_merge_mode::windowed);
  CHECK(params.sort_merge_window_rows == 4096);
  CHECK(params.sort_merge_output_bytes == 2 * mib);
  CHECK(params.sort_merge_one_shot_max_fraction == Approx(0.5));
  CHECK(params.sort_merge_max_fan_in == 12);
  CHECK(current_setting(con, "sort_merge_mode").ToString() == "windowed");
  CHECK(current_setting(con, "sort_merge_window_rows").GetValue<uint64_t>() == 4096);
  CHECK(current_setting(con, "sort_merge_max_fan_in").GetValue<uint64_t>() == 12);
}

TEST_CASE("sort_merge config-file keys parse", "[sort_merge_settings][config_opt]")
{
  SECTION("omitted keys keep the defaults")
  {
    temp_config_file const file(operator_params_yaml("    max_sort_partition_bytes: 0\n"));
    sirius_config cfg;
    cfg.load_from_file(file.path());
    auto const& params = cfg.get_operator_params();
    CHECK(params.sort_merge_mode_setting == sort_merge_mode::automatic);
    CHECK(params.sort_merge_window_rows == 0);
    CHECK(params.sort_merge_output_bytes == 0);
    CHECK(params.sort_merge_one_shot_max_fraction == 0.0);
    CHECK(params.sort_merge_max_fan_in == config::DEFAULT_SORT_MERGE_MAX_FAN_IN);
  }

  SECTION("every key is read")
  {
    temp_config_file const file(
      operator_params_yaml("    sort_merge_mode: One_Shot\n"
                           "    sort_merge_window_rows: 2048\n"
                           "    sort_merge_output_bytes: 64MiB\n"
                           "    sort_merge_one_shot_max_fraction: 0.25\n"
                           "    sort_merge_max_fan_in: 16\n"));
    sirius_config cfg;
    cfg.load_from_file(file.path());
    auto const& params = cfg.get_operator_params();
    CHECK(params.sort_merge_mode_setting == sort_merge_mode::one_shot);
    CHECK(params.sort_merge_window_rows == 2048);
    CHECK(params.sort_merge_output_bytes == 64 * mib);
    CHECK(params.sort_merge_one_shot_max_fraction == Approx(0.25));
    CHECK(params.sort_merge_max_fan_in == 16);
  }

  SECTION("boundary values are accepted")
  {
    temp_config_file const file(
      operator_params_yaml("    sort_merge_mode: windowed\n"
                           "    sort_merge_window_rows: 1024\n"
                           "    sort_merge_output_bytes: 1048576\n"
                           "    sort_merge_one_shot_max_fraction: 1.0\n"
                           "    sort_merge_max_fan_in: 2\n"));
    sirius_config cfg;
    cfg.load_from_file(file.path());
    auto const& params = cfg.get_operator_params();
    CHECK(params.sort_merge_mode_setting == sort_merge_mode::windowed);
    CHECK(params.sort_merge_window_rows == 1024);
    CHECK(params.sort_merge_output_bytes == 1 * mib);
    CHECK(params.sort_merge_one_shot_max_fraction == 1.0);
    CHECK(params.sort_merge_max_fan_in == 2);
  }
}

TEST_CASE("sort_merge config-file keys reject invalid values", "[sort_merge_settings][config_opt]")
{
  auto const [line, message] =
    GENERATE(std::pair{"sort_merge_mode: oneshot", "valid values: auto, windowed, one_shot"},
             std::pair{"sort_merge_window_rows: 1023", "must be 0 (derive) or at least 1024"},
             std::pair{"sort_merge_window_rows: -1", "must be non-negative"},
             std::pair{"sort_merge_output_bytes: -1", "must be non-negative"},
             std::pair{"sort_merge_max_fan_in: -1", "must be non-negative"},
             std::pair{"sort_merge_output_bytes: 1023KiB", "must be 0 (derive) or at least 1 MiB"},
             std::pair{"sort_merge_one_shot_max_fraction: 1.5", "value out of range"},
             std::pair{"sort_merge_one_shot_max_fraction: -0.1", "value out of range"},
             std::pair{"sort_merge_one_shot_max_fraction: .nan", "value out of range"},
             std::pair{"sort_merge_max_fan_in: 1", "must be at least 2"},
             std::pair{"sort_merge_max_fan_in: 0", "must be at least 2"});
  CAPTURE(line);

  temp_config_file const file(operator_params_yaml("    " + std::string(line) + "\n"));
  sirius_config cfg;
  CHECK_THROWS_WITH(cfg.load_from_file(file.path()), Catch::Matchers::ContainsSubstring(message));
  // The message names the key.
  auto const key = std::string(line).substr(0, std::string(line).find(':'));
  CHECK_THROWS_WITH(cfg.load_from_file(file.path()), Catch::Matchers::ContainsSubstring(key));
}

TEST_CASE("sort_merge config-file values are the DuckDB setting defaults",
          "[sort_merge_settings][isolated_context]")
{
  temp_config_file const file(
    "sirius:\n"
    "  topology:\n"
    "    num_gpus: 1\n"
    "  memory:\n"
    "    gpu:\n"
    "      usage_limit_fraction: 0.05\n"
    "      reservation_limit_fraction: 0.04\n"
    "      downgrade_trigger_fraction: 0.04\n"
    "      downgrade_stop_fraction: 0.02\n"
    "    host:\n"
    "      capacity_bytes: 160000000\n"
    "  executor:\n"
    "    downgrade:\n"
    "      num_threads: 1\n"
    "    scan_manager:\n"
    "      uring_n_reactors: 1\n"
    "  operator_params:\n"
    "    sort_merge_mode: windowed\n"
    "    sort_merge_window_rows: 8192\n"
    "    sort_merge_output_bytes: 4MiB\n"
    "    sort_merge_one_shot_max_fraction: 0.2\n"
    "    sort_merge_max_fan_in: 4\n");

  auto db = sirius::test::open_sirius_db(nullptr, file.path());
  duckdb::Connection con(*db);
  auto const& params = duckdb::session_operator_params(*con.context);

  CHECK(current_setting(con, "sort_merge_mode").ToString() == "windowed");
  CHECK(current_setting(con, "sort_merge_window_rows").GetValue<uint64_t>() == 8192);
  CHECK(current_setting(con, "sort_merge_output_bytes").GetValue<uint64_t>() == 4 * mib);
  CHECK(current_setting(con, "sort_merge_one_shot_max_fraction").GetValue<double>() == Approx(0.2));
  CHECK(current_setting(con, "sort_merge_max_fan_in").GetValue<uint64_t>() == 4);
  CHECK(params.sort_merge_mode_setting == sort_merge_mode::windowed);
  CHECK(params.sort_merge_window_rows == 8192);
  CHECK(params.sort_merge_max_fan_in == 4);

  // A session override is undone by RESET, which restores the config-file value.
  require_ok(con, "SET sort_merge_mode = 'one_shot'");
  require_ok(con, "SET sort_merge_max_fan_in = 32");
  CHECK(params.sort_merge_mode_setting == sort_merge_mode::one_shot);
  CHECK(params.sort_merge_max_fan_in == 32);
  require_ok(con, "RESET sort_merge_mode");
  require_ok(con, "RESET sort_merge_max_fan_in");
  CHECK(params.sort_merge_mode_setting == sort_merge_mode::windowed);
  CHECK(params.sort_merge_max_fan_in == 4);
  CHECK(params.sort_merge_output_bytes == 4 * mib);
  CHECK(params.sort_merge_one_shot_max_fraction == Approx(0.2));
}

TEST_CASE(
  "sort_merge_disabled_rules defaults to the one-shot estimate off, and takes only known bits",
  "[sort_merge_settings][isolated_context]")
{
  sirius::util::env_guard const disable_sirius{"SIRIUS_DISABLE", "1"};

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto const& params = duckdb::session_operator_params(*con.context);
  constexpr auto all = config::sort_merge_rule::ALL;

  constexpr auto by_default = config::DEFAULT_SORT_MERGE_DISABLED_RULES;
  CHECK(by_default == config::sort_merge_rule::ONE_SHOT_ESTIMATE);
  CHECK(current_setting(con, "sort_merge_disabled_rules").GetValue<uint64_t>() == by_default);
  CHECK(params.sort_merge_disabled_rules == by_default);

  // 0 puts every rule in force.
  require_ok(con, "SET sort_merge_disabled_rules = 0");
  CHECK(params.sort_merge_disabled_rules == 0);

  require_ok(con, "SET sort_merge_disabled_rules = 17");
  CHECK(params.sort_merge_disabled_rules ==
        (config::sort_merge_rule::HUFFMAN_PLAN | config::sort_merge_rule::ROUND_ROWS));
  CHECK(current_setting(con, "sort_merge_disabled_rules").GetValue<uint64_t>() == 17);

  require_ok(con, "SET sort_merge_disabled_rules = " + std::to_string(all));
  CHECK(params.sort_merge_disabled_rules == all);

  // A bit that names no rule is rejected, and the previous value stays.
  require_rejected(con,
                   "SET sort_merge_disabled_rules = " + std::to_string(all + 1),
                   "sort_merge_disabled_rules has an unknown rule bit");
  require_rejected(con,
                   "SET sort_merge_disabled_rules = 4096",
                   "sort_merge_disabled_rules has an unknown rule bit");
  CHECK(params.sort_merge_disabled_rules == all);

  require_ok(con, "RESET sort_merge_disabled_rules");
  CHECK(params.sort_merge_disabled_rules == by_default);
  CHECK(current_setting(con, "sort_merge_disabled_rules").GetValue<uint64_t>() == by_default);
}

TEST_CASE("sort_merge_disabled_rules is a config-file key", "[sort_merge_settings][config_opt]")
{
  SECTION("read")
  {
    temp_config_file const file(operator_params_yaml("    sort_merge_disabled_rules: 12\n"));
    sirius_config cfg;
    cfg.load_from_file(file.path());
    CHECK(
      cfg.get_operator_params().sort_merge_disabled_rules ==
      (config::sort_merge_rule::PARTITION_BUDGET | config::sort_merge_rule::IN_PLACE_RESERVATION));
  }

  SECTION("omitted: the default")
  {
    temp_config_file const file(operator_params_yaml("    max_sort_partition_bytes: 0\n"));
    sirius_config cfg;
    cfg.load_from_file(file.path());
    CHECK(cfg.get_operator_params().sort_merge_disabled_rules ==
          config::DEFAULT_SORT_MERGE_DISABLED_RULES);
  }

  SECTION("0 puts every rule in force")
  {
    temp_config_file const file(operator_params_yaml("    sort_merge_disabled_rules: 0\n"));
    sirius_config cfg;
    cfg.load_from_file(file.path());
    CHECK(cfg.get_operator_params().sort_merge_disabled_rules == 0);
  }

  SECTION("rejected")
  {
    auto const [line, message] =
      GENERATE(std::pair{"sort_merge_disabled_rules: 512", "unknown rule bit"},
               std::pair{"sort_merge_disabled_rules: -1", "must be non-negative"});
    CAPTURE(line);
    temp_config_file const file(operator_params_yaml("    " + std::string(line) + "\n"));
    sirius_config cfg;
    CHECK_THROWS_WITH(cfg.load_from_file(file.path()), Catch::Matchers::ContainsSubstring(message));
    CHECK_THROWS_WITH(cfg.load_from_file(file.path()),
                      Catch::Matchers::ContainsSubstring("sort_merge_disabled_rules"));
  }
}
