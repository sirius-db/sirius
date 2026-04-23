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

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/s3_live_test.hpp>

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace {

static fs::path get_project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT);
#else
  return fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
#endif
}

struct sirius_config_env_guard {
  explicit sirius_config_env_guard(std::string const& config_path)
  {
    setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
  }

  ~sirius_config_env_guard() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

struct env_cfg {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  fs::path local_dir;

  bool present() const
  {
    return !endpoint.empty() && !access_key.empty() && !secret_key.empty() &&
           !bucket.empty() && !local_dir.empty();
  }

  [[nodiscard]] fs::path nation_parquet_path() const
  {
    return local_dir / "parquet" / "nation.parquet";
  }

  [[nodiscard]] std::string nation_parquet_uri() const
  {
    return "s3://" + bucket + "/parquet/nation.parquet";
  }

  [[nodiscard]] std::string object_uri(std::string const& key) const
  {
    return "s3://" + bucket + "/" + key;
  }
};

env_cfg read_env()
{
  env_cfg cfg;
  cfg.endpoint   = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_ENDPOINT");
  cfg.region     = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  cfg.access_key = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_ACCESS_KEY");
  cfg.secret_key = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_SECRET_KEY");
  cfg.bucket     = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_BUCKET");
  cfg.local_dir  = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_LOCAL_DIR");
  return cfg;
}

bool skip_if_env_missing(env_cfg const& cfg)
{
  if (!cfg.present()) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_* not set (see test/cpp/integration/s3/README.md)");
    return true;
  }
  if (!fs::is_directory(cfg.local_dir)) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_LOCAL_DIR not present - run `make s3-up` first");
    return true;
  }
  if (!fs::exists(cfg.nation_parquet_path())) {
    SUCCEED("Skipping: parquet/nation.parquet fixture missing - run `make s3-up` first");
    return true;
  }
  return false;
}

std::string sql_quote(std::string value)
{
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value) {
    out.push_back(c);
    if (c == '\'') out.push_back('\'');
  }
  return out;
}

constexpr std::size_t NATION_ROWS = 25;
constexpr std::array<std::int32_t, NATION_ROWS> EXPECTED_REGION_KEYS{
  0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2, 4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1};
constexpr std::array<char const*, NATION_ROWS> EXPECTED_NATION_NAMES{
  "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA", "FRANCE",
  "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA",
  "MOROCCO", "MOZAMBIQUE", "PERU", "CHINA", "ROMANIA", "SAUDI ARABIA", "VIETNAM",
  "RUSSIA", "UNITED KINGDOM", "UNITED STATES"};

std::int64_t expected_sum_regionkeys()
{
  std::int64_t sum = 0;
  for (auto region_key : EXPECTED_REGION_KEYS) {
    sum += region_key;
  }
  return sum;
}

std::string lowercase(std::string value)
{
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool contains_any_token(std::string const& message,
                        std::initializer_list<char const*> expected_tokens)
{
  auto const normalized = lowercase(message);
  for (auto const* token : expected_tokens) {
    if (normalized.find(lowercase(token)) != std::string::npos) return true;
  }
  return false;
}

class s3_gpu_execution_fixture {
 public:
  s3_gpu_execution_fixture()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path =
        get_project_root() / "test" / "cpp" / "integration" / "s3" / "sirius.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<sirius_config_env_guard>(cfg_path.string());
      db           = std::make_unique<duckdb::DuckDB>(nullptr);
      con          = std::make_unique<duckdb::Connection>(*db);
    }
  }

  void configure_s3(env_cfg const& cfg)
  {
    require_ok("SET autoload_known_extensions = false;");
    require_ok("SET autoinstall_known_extensions = false;");
    require_ok("SET enable_duckdb_fallback = false;");
    require_ok("SET s3_transport = 'http';");
    require_ok("SET s3_endpoint = '" + sql_quote(cfg.endpoint) + "';");
    require_ok("SET s3_region = '" + sql_quote(cfg.region) + "';");
    require_ok("SET s3_access_key = '" + sql_quote(cfg.access_key) + "';");
    require_ok("SET s3_secret_key = '" + sql_quote(cfg.secret_key) + "';");
  }

  std::unique_ptr<duckdb::MaterializedQueryResult> run_gpu(std::string const& inner_sql)
  {
    auto result = run_gpu_raw(inner_sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("gpu_execution error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    return result;
  }

  std::unique_ptr<duckdb::MaterializedQueryResult> run_gpu_expect_error(
    std::string const& inner_sql, std::initializer_list<char const*> expected_tokens)
  {
    auto result = run_gpu_raw(inner_sql);
    REQUIRE(result);
    if (!result->HasError()) {
      UNSCOPED_INFO("gpu_execution unexpectedly succeeded");
    }
    REQUIRE(result->HasError());
    auto const error = result->GetError();
    UNSCOPED_INFO("gpu_execution error: " << error);
    REQUIRE(contains_any_token(error, expected_tokens));
    return result;
  }

 private:
  std::unique_ptr<duckdb::MaterializedQueryResult> run_gpu_raw(std::string const& inner_sql)
  {
    // Intentionally route through gpu_execution only. The companion guard test
    // below proves the plain CPU s3:// path is unavailable in the same
    // connection, so success here must be coming from Sirius's own S3
    // datasource path rather than DuckDB's native remote I/O.
    return con->Query("SELECT * FROM gpu_execution(\"" + inner_sql + "\")");
  }

  void require_ok(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("SET failed: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
  }

 public:
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::unique_ptr<sirius_config_env_guard> config_guard;
};

}  // namespace

TEST_CASE_METHOD(s3_gpu_execution_fixture,
                 "gpu_execution s3 - missing object surfaces query error",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  configure_s3(cfg);

  auto const missing_key =
    "parquet/definitely-does-not-exist-" + std::to_string(std::rand()) + ".parquet";
  run_gpu_expect_error(
    "SELECT COUNT(*)::BIGINT FROM read_parquet('" + cfg.object_uri(missing_key) + "')",
    {"404", "missing", "not found", "no such"});
}

TEST_CASE_METHOD(s3_gpu_execution_fixture,
                 "gpu_execution s3 - bad credentials surface query error",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  cfg.secret_key = "not-the-right-secret-key";
  configure_s3(cfg);

  run_gpu_expect_error("SELECT COUNT(*)::BIGINT FROM read_parquet('" + cfg.nation_parquet_uri() +
                         "')",
                       {"403", "signature", "forbidden", "access denied"});
}

TEST_CASE_METHOD(s3_gpu_execution_fixture,
                 "gpu_execution s3 - cpu s3 read stays unavailable without httpfs",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  configure_s3(cfg);

  auto cpu =
    con->Query("SELECT COUNT(*)::BIGINT FROM read_parquet('" + cfg.nation_parquet_uri() + "')");
  REQUIRE(cpu);
  if (!cpu->HasError()) {
    UNSCOPED_INFO("Plain CPU s3:// read unexpectedly succeeded without httpfs guard");
  }
  REQUIRE(cpu->HasError());

  auto gpu = run_gpu("SELECT COUNT(*)::BIGINT FROM read_parquet('" + cfg.nation_parquet_uri() +
                     "')");
  REQUIRE(gpu->ColumnCount() == 1);
  REQUIRE(gpu->RowCount() == 1);
  CHECK(gpu->GetValue(0, 0).GetValue<std::int64_t>() ==
        static_cast<std::int64_t>(NATION_ROWS));
}

TEST_CASE_METHOD(s3_gpu_execution_fixture,
                 "gpu_execution s3 - basic scan end-to-end",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  configure_s3(cfg);

  auto result = run_gpu("SELECT n_nationkey::INTEGER, n_regionkey::INTEGER, n_name "
                        "FROM read_parquet('" +
                        cfg.nation_parquet_uri() + "') ORDER BY n_nationkey");

  REQUIRE(result->ColumnCount() == 3);
  REQUIRE(result->RowCount() == NATION_ROWS);
  for (std::size_t i = 0; i < NATION_ROWS; ++i) {
    auto const nation_key = result->GetValue(0, i).GetValue<std::int32_t>();
    auto const region_key = result->GetValue(1, i).GetValue<std::int32_t>();
    auto const name       = result->GetValue(2, i).GetValue<std::string>();

    REQUIRE(nation_key == static_cast<std::int32_t>(i));
    REQUIRE(region_key == EXPECTED_REGION_KEYS[i]);
    REQUIRE(name == EXPECTED_NATION_NAMES[i]);
  }
}

TEST_CASE_METHOD(s3_gpu_execution_fixture,
                 "gpu_execution s3 - projection filter end-to-end",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  configure_s3(cfg);

  auto result = run_gpu("SELECT n_nationkey::INTEGER, n_regionkey::INTEGER, n_name "
                        "FROM read_parquet('" +
                        cfg.nation_parquet_uri() +
                        "') WHERE n_regionkey = 1 ORDER BY n_nationkey");

  REQUIRE(result->ColumnCount() == 3);
  constexpr std::array<std::int32_t, 5> expected_keys{1, 2, 3, 17, 24};
  REQUIRE(result->RowCount() == expected_keys.size());
  for (std::size_t i = 0; i < result->RowCount(); ++i) {
    auto const nation_key = result->GetValue(0, i).GetValue<std::int32_t>();
    auto const region_key = result->GetValue(1, i).GetValue<std::int32_t>();
    auto const name       = result->GetValue(2, i).GetValue<std::string>();

    REQUIRE(nation_key == expected_keys[i]);
    REQUIRE(region_key == 1);
    REQUIRE(name == EXPECTED_NATION_NAMES[static_cast<std::size_t>(nation_key)]);
  }
}

TEST_CASE_METHOD(s3_gpu_execution_fixture,
                 "gpu_execution s3 - aggregate end-to-end",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  configure_s3(cfg);

  auto result = run_gpu("SELECT COUNT(*)::BIGINT, MIN(n_nationkey)::INTEGER, "
                        "MAX(n_nationkey)::INTEGER, SUM(n_regionkey)::BIGINT "
                        "FROM read_parquet('" +
                        cfg.nation_parquet_uri() + "')");

  REQUIRE(result->ColumnCount() == 4);
  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).GetValue<std::int64_t>() ==
        static_cast<std::int64_t>(NATION_ROWS));
  CHECK(result->GetValue(1, 0).GetValue<std::int32_t>() == 0);
  CHECK(result->GetValue(2, 0).GetValue<std::int32_t>() ==
        static_cast<std::int32_t>(NATION_ROWS - 1));
  CHECK(result->GetValue(3, 0).GetValue<std::int64_t>() == expected_sum_regionkeys());
}
