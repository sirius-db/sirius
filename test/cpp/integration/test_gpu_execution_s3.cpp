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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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

  [[nodiscard]] fs::path small_parquet_path() const { return local_dir / "small.parquet"; }

  [[nodiscard]] std::string small_parquet_uri() const
  {
    return "s3://" + bucket + "/small.parquet";
  }
};

std::string getenv_or(char const* key, char const* dflt = "")
{
  auto const* value = std::getenv(key);
  return (value && *value) ? value : dflt;
}

env_cfg read_env()
{
  env_cfg cfg;
  cfg.endpoint   = getenv_or("SIRIUS_TEST_S3_ENDPOINT");
  cfg.region     = getenv_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  cfg.access_key = getenv_or("SIRIUS_TEST_S3_ACCESS_KEY");
  cfg.secret_key = getenv_or("SIRIUS_TEST_S3_SECRET_KEY");
  cfg.bucket     = getenv_or("SIRIUS_TEST_S3_BUCKET");
  cfg.local_dir  = getenv_or("SIRIUS_TEST_S3_LOCAL_DIR");
  return cfg;
}

bool skip_if_env_missing(env_cfg const& cfg)
{
  if (!cfg.present()) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_* not set (see test/integration/s3/README.md)");
    return true;
  }
  if (!fs::is_directory(cfg.local_dir)) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_LOCAL_DIR not present - run `make s3-up` first");
    return true;
  }
  if (!fs::exists(cfg.small_parquet_path())) {
    SUCCEED("Skipping: small.parquet not generated (install pyarrow and rerun `make s3-up`)");
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

constexpr std::size_t PARQUET_ROWS  = 256;
constexpr std::int64_t PARQUET_KNUTH = 2654435761LL;

std::int64_t expected_v(std::int32_t id) { return static_cast<std::int64_t>(id) * PARQUET_KNUTH; }

std::string expected_s(std::int32_t id)
{
  char buf[16];
  std::snprintf(buf, sizeof(buf), "row-%04d", id);
  return std::string(buf);
}

std::int64_t expected_sum_v()
{
  // Sum_{id=0}^{255} id * K = K * (255 * 256 / 2)
  return PARQUET_KNUTH * static_cast<std::int64_t>((PARQUET_ROWS - 1) * PARQUET_ROWS / 2);
}

class s3_gpu_execution_fixture {
 public:
  s3_gpu_execution_fixture()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = get_project_root() / "test" / "integration" / "s3" / "sirius.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<sirius_config_env_guard>(cfg_path.string());
      db           = std::make_unique<duckdb::DuckDB>(nullptr);
      con          = std::make_unique<duckdb::Connection>(*db);
    }
  }

  void configure_s3(env_cfg const& cfg)
  {
    require_ok("SET enable_duckdb_fallback = false;");
    require_ok("SET s3_transport = 'http';");
    require_ok("SET s3_endpoint = '" + sql_quote(cfg.endpoint) + "';");
    require_ok("SET s3_region = '" + sql_quote(cfg.region) + "';");
    require_ok("SET s3_access_key = '" + sql_quote(cfg.access_key) + "';");
    require_ok("SET s3_secret_key = '" + sql_quote(cfg.secret_key) + "';");
  }

  std::unique_ptr<duckdb::MaterializedQueryResult> run_gpu(std::string const& inner_sql)
  {
    // Intentionally route through gpu_execution only; do not install/use httpfs
    // or run a CPU-side s3:// read. If this query succeeds, it is exercising
    // Sirius's own S3 datasource path rather than DuckDB's native remote I/O.
    auto result = con->Query("SELECT * FROM gpu_execution(\"" + inner_sql + "\")");
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("gpu_execution error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    return result;
  }

 private:
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
                 "gpu_execution s3 - basic scan end-to-end",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  configure_s3(cfg);

  auto result = run_gpu("SELECT id::INTEGER, v::BIGINT, s FROM read_parquet('" +
                        cfg.small_parquet_uri() + "') ORDER BY id");

  REQUIRE(result->ColumnCount() == 3);
  REQUIRE(result->RowCount() == PARQUET_ROWS);
  for (std::size_t i = 0; i < PARQUET_ROWS; ++i) {
    auto const id = result->GetValue(0, i).GetValue<std::int32_t>();
    auto const v  = result->GetValue(1, i).GetValue<std::int64_t>();
    auto const s  = result->GetValue(2, i).GetValue<std::string>();

    REQUIRE(id == static_cast<std::int32_t>(i));
    REQUIRE(v == expected_v(id));
    REQUIRE(s == expected_s(id));
  }
}

TEST_CASE_METHOD(s3_gpu_execution_fixture,
                 "gpu_execution s3 - projection filter end-to-end",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  configure_s3(cfg);

  auto result = run_gpu("SELECT id::INTEGER, v::BIGINT, s FROM read_parquet('" +
                        cfg.small_parquet_uri() + "') WHERE id % 17 = 0 ORDER BY id");

  REQUIRE(result->ColumnCount() == 3);
  REQUIRE(result->RowCount() == 16);
  for (std::size_t i = 0; i < result->RowCount(); ++i) {
    auto const expected_id = static_cast<std::int32_t>(i * 17);
    auto const id          = result->GetValue(0, i).GetValue<std::int32_t>();
    auto const v           = result->GetValue(1, i).GetValue<std::int64_t>();
    auto const s           = result->GetValue(2, i).GetValue<std::string>();

    REQUIRE(id == expected_id);
    REQUIRE(v == expected_v(id));
    REQUIRE(s == expected_s(id));
  }
}

TEST_CASE_METHOD(s3_gpu_execution_fixture,
                 "gpu_execution s3 - aggregate end-to-end",
                 "[integration][gpu_execution][s3][parquet]")
{
  auto cfg = read_env();
  if (skip_if_env_missing(cfg)) return;

  configure_s3(cfg);

  auto result = run_gpu("SELECT COUNT(*)::BIGINT, MIN(id)::INTEGER, MAX(id)::INTEGER, "
                        "SUM(v)::BIGINT FROM read_parquet('" + cfg.small_parquet_uri() + "')");

  REQUIRE(result->ColumnCount() == 4);
  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).GetValue<std::int64_t>() ==
        static_cast<std::int64_t>(PARQUET_ROWS));
  CHECK(result->GetValue(1, 0).GetValue<std::int32_t>() == 0);
  CHECK(result->GetValue(2, 0).GetValue<std::int32_t>() ==
        static_cast<std::int32_t>(PARQUET_ROWS - 1));
  CHECK(result->GetValue(3, 0).GetValue<std::int64_t>() == expected_sum_v());
}
