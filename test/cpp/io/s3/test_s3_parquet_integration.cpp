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

// Semantic end-to-end test for the Sirius S3 read path:
//
//   standard parquet fixture -> mc cp -> MinIO -> datasource_factory
//     -> sirius s3_ioctx bytes -> DuckDB parquet reader -> row assertions
//
// This complements test_s3_integration.cpp which only proves byte equality.
// Here we parse the bytes as parquet to confirm the read path delivers a
// correct, complete, readable object (not e.g. a truncated fetch that still
// hashes fine against a truncated local copy).
//
// The parquet fixture is copied from test/cpp/integration/data/parquet, so the
// semantic assertions use the same fixed TPCH content as the regular C++
// integration tests without needing DuckDB httpfs.
//
// Skip conditions (all SUCCEED with a reason):
//   - SIRIUS_TEST_S3_* env vars unset (same pattern as test_s3_integration)
//   - parquet/nation.parquet missing locally - `make s3-up` did not populate
//     fixtures successfully
//   - factory::create throws - skipped in best-effort mode, failed in
//     SIRIUS_TEST_S3_STRICT=1 mode

// IMPORTANT: include order mirrors test_parquet_scan_via_factory.cpp.
// liburing.h (pulled transitively by sirius io headers) defines BLOCK_SIZE as
// a macro that collides with a duckdb concurrentqueue identifier of the same
// name. All duckdb headers must precede sirius io/uring headers.
#include "catch.hpp"
#include "io/datasource_factory.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "sirius_config.hpp"
#include "utils/s3_live_test.hpp"

#include <duckdb.hpp>
#include <duckdb/main/connection.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using sirius::sirius_config;
using sirius::io::datasource_factory;
using sirius::io::datasource_registry;
using sirius::io::io_datasource;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;

namespace {

struct env_cfg {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::filesystem::path local_dir;

  bool present() const
  {
    return !endpoint.empty() && !access_key.empty() && !secret_key.empty() && !bucket.empty() &&
           !local_dir.empty();
  }
};

env_cfg read_env()
{
  env_cfg c;
  c.endpoint   = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_ENDPOINT");
  c.region     = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  c.access_key = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_ACCESS_KEY");
  c.secret_key = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_SECRET_KEY");
  c.bucket     = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_BUCKET");
  c.local_dir  = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_LOCAL_DIR");
  return c;
}

std::shared_ptr<s3_ioctx> make_ctx(env_cfg const& e)
{
  s3_ioctx_config cfg;
  cfg.endpoint   = e.endpoint;
  cfg.region     = e.region;
  cfg.access_key = e.access_key;
  cfg.secret_key = e.secret_key;
  return std::make_shared<s3_ioctx>(std::move(cfg));
}

constexpr std::size_t NATION_ROWS = 25;
constexpr std::array<std::int32_t, NATION_ROWS> EXPECTED_REGION_KEYS{
  0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2, 4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1};
constexpr std::array<char const*, NATION_ROWS> EXPECTED_NATION_NAMES{
  "ALGERIA", "ARGENTINA", "BRAZIL",         "CANADA",       "EGYPT", "ETHIOPIA", "FRANCE",
  "GERMANY", "INDIA",     "INDONESIA",      "IRAN",         "IRAQ",  "JAPAN",    "JORDAN",
  "KENYA",   "MOROCCO",   "MOZAMBIQUE",     "PERU",         "CHINA", "ROMANIA",  "SAUDI ARABIA",
  "VIETNAM", "RUSSIA",    "UNITED KINGDOM", "UNITED STATES"};

std::int64_t expected_sum_regionkeys()
{
  std::int64_t sum = 0;
  for (auto region_key : EXPECTED_REGION_KEYS) {
    sum += region_key;
  }
  return sum;
}

}  // namespace

TEST_CASE("s3_parquet_integration: read_parquet end-to-end through sirius s3 pipeline",
          "[s3][parquet][integration]")
{
  auto e = read_env();
  if (!e.present()) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_* not set (see test/cpp/integration/s3/README.md)");
    return;
  }
  if (!std::filesystem::is_directory(e.local_dir)) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_LOCAL_DIR not present - run `make s3-up` first");
    return;
  }
  auto const local_path = e.local_dir / "parquet" / "nation.parquet";
  if (!std::filesystem::exists(local_path)) {
    SUCCEED("Skipping: parquet/nation.parquet fixture missing; run `make s3-up` first");
    return;
  }

  // ---- stage 1: read remote bytes through the same factory the prod scan
  // task uses (parquet_scan_task.cpp calls datasource_factory::create for
  // every file path).
  datasource_registry reg;
  reg.register_ioctx("s3", make_ctx(e));
  sirius_config cfg;

  std::unique_ptr<io_datasource> ds;
  try {
    ds = datasource_factory::create("s3://" + e.bucket + "/parquet/nation.parquet", reg, cfg);
  } catch (std::exception const& ex) {
    sirius::test::s3::handle_live_runtime_failure(
      "factory::create failed",
      ex,
      "Skipping: MinIO unreachable or parquet/nation.parquet missing in bucket");
    return;
  }
  REQUIRE(ds != nullptr);

  auto const n_bytes = ds->size();
  REQUIRE(n_bytes > 0);
  auto remote = ds->host_read(0, n_bytes);
  REQUIRE(remote != nullptr);
  REQUIRE(remote->size() == n_bytes);

  // ---- stage 2: sanity-check bytes against the local fixture. Divergence
  // here indicates a transport-level bug (range offsets, host header, SigV4)
  // and must be surfaced before we try to parse the object.
  std::ifstream f(local_path, std::ios::binary);
  REQUIRE(f.good());
  f.seekg(0, std::ios::end);
  auto const local_size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> local(local_size);
  f.read(reinterpret_cast<char*>(local.data()), static_cast<std::streamsize>(local_size));
  REQUIRE(f.gcount() == static_cast<std::streamsize>(local_size));
  REQUIRE(local_size == n_bytes);
  REQUIRE(std::memcmp(remote->data(), local.data(), n_bytes) == 0);

  // ---- stage 3: materialize the remote bytes to a temp file and hand them
  // to DuckDB's parquet reader. This validates the semantic end-to-end path:
  // if the reader observes different rows than the checked-in TPCH fixture,
  // the s3 pipeline is returning wrong or truncated data.
  auto const tmp_path =
    std::filesystem::temp_directory_path() /
    ("sirius_s3_parquet_integration_" + std::to_string(std::rand()) + ".parquet");
  {
    std::ofstream out(tmp_path, std::ios::binary);
    REQUIRE(out.good());
    out.write(reinterpret_cast<char const*>(remote->data()), static_cast<std::streamsize>(n_bytes));
  }
  struct scoped_file {
    std::filesystem::path p;
    ~scoped_file()
    {
      std::error_code ec;
      std::filesystem::remove(p, ec);
    }
  } cleanup{tmp_path};

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto const parquet_ref = "read_parquet('" + tmp_path.string() + "')";

  // Row count + key range + region distribution.
  auto agg = con.Query(
    "SELECT COUNT(*)::BIGINT, MIN(n_nationkey)::INTEGER, "
    "MAX(n_nationkey)::INTEGER, SUM(n_regionkey)::BIGINT FROM " +
    parquet_ref);
  REQUIRE(agg);
  REQUIRE(!agg->HasError());
  REQUIRE(agg->RowCount() == 1);
  CHECK(agg->GetValue(0, 0).GetValue<std::int64_t>() == static_cast<std::int64_t>(NATION_ROWS));
  CHECK(agg->GetValue(1, 0).GetValue<std::int32_t>() == 0);
  CHECK(agg->GetValue(2, 0).GetValue<std::int32_t>() == static_cast<std::int32_t>(NATION_ROWS - 1));
  CHECK(agg->GetValue(3, 0).GetValue<std::int64_t>() == expected_sum_regionkeys());

  // Spot-check boundary rows. Full-row scan below
  // then verifies every row - the spot check isolates failures when the full
  // scan reports a mismatch.
  auto first = con.Query("SELECT n_nationkey::INTEGER, n_regionkey::INTEGER, n_name FROM " +
                         parquet_ref + " ORDER BY n_nationkey LIMIT 1");
  REQUIRE(first);
  REQUIRE(!first->HasError());
  REQUIRE(first->RowCount() == 1);
  CHECK(first->GetValue(0, 0).GetValue<std::int32_t>() == 0);
  CHECK(first->GetValue(1, 0).GetValue<std::int32_t>() == EXPECTED_REGION_KEYS[0]);
  CHECK(first->GetValue(2, 0).GetValue<std::string>() == EXPECTED_NATION_NAMES[0]);

  auto last = con.Query("SELECT n_nationkey::INTEGER, n_regionkey::INTEGER, n_name FROM " +
                        parquet_ref + " ORDER BY n_nationkey DESC LIMIT 1");
  REQUIRE(last);
  REQUIRE(!last->HasError());
  REQUIRE(last->RowCount() == 1);
  auto const last_id = static_cast<std::int32_t>(NATION_ROWS - 1);
  CHECK(last->GetValue(0, 0).GetValue<std::int32_t>() == last_id);
  CHECK(last->GetValue(1, 0).GetValue<std::int32_t>() ==
        EXPECTED_REGION_KEYS[static_cast<std::size_t>(last_id)]);
  CHECK(last->GetValue(2, 0).GetValue<std::string>() ==
        EXPECTED_NATION_NAMES[static_cast<std::size_t>(last_id)]);

  // Full-row verification. Stream every row in key order and compare against
  // the fixed TPCH nation fixture; a single mismatch fails the test.
  auto all = con.Query("SELECT n_nationkey::INTEGER, n_regionkey::INTEGER, n_name FROM " +
                       parquet_ref + " ORDER BY n_nationkey");
  REQUIRE(all);
  REQUIRE(!all->HasError());
  REQUIRE(all->RowCount() == NATION_ROWS);
  for (std::size_t i = 0; i < NATION_ROWS; ++i) {
    auto const nation_key = all->GetValue(0, i).GetValue<std::int32_t>();
    auto const region_key = all->GetValue(1, i).GetValue<std::int32_t>();
    auto const name       = all->GetValue(2, i).GetValue<std::string>();
    REQUIRE(nation_key == static_cast<std::int32_t>(i));
    REQUIRE(region_key == EXPECTED_REGION_KEYS[i]);
    REQUIRE(name == EXPECTED_NATION_NAMES[i]);
  }
}
