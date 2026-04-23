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

#include <cctype>
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
    // Direct read_parquet('s3://...') still binds through DuckDB's native
    // table function, which requires httpfs before Sirius can take over.
    // Keep this helper for the guard test below so the current limitation
    // stays explicit instead of failing as a surprise later.
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
                 "gpu_execution s3 - direct read_parquet currently requires DuckDB httpfs",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  configure_s3(cfg);

  auto cpu = con->Query(
    "SELECT COUNT(*)::BIGINT FROM read_parquet('" + cfg.nation_parquet_uri() + "')");
  REQUIRE(cpu);
  REQUIRE(cpu->HasError());
  auto const cpu_error = cpu->GetError();
  UNSCOPED_INFO("cpu read_parquet error: " << cpu_error);
  REQUIRE(contains_any_token(cpu_error, {"httpfs", "missing extension", "requires the extension"}));

  run_gpu_expect_error(
    "SELECT COUNT(*)::BIGINT FROM read_parquet('" + cfg.nation_parquet_uri() + "')",
    {"httpfs", "missing extension", "requires the extension"});
}
