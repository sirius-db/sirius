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
#include "io/object_store_config.hpp"
#include "io/rest/config.hpp"
#include "sirius_config.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

using sirius::io::enum_to_string;
using sirius::io::object_store_config;
using sirius::io::string_to_enum;

namespace {

void write_yaml(std::filesystem::path const& path, std::string const& text)
{
  std::ofstream out(path);
  out << text;
  REQUIRE(out);
}

std::string load_config_error(std::filesystem::path const& path, std::string const& text)
{
  write_yaml(path, text);
  std::string error;
  try {
    sirius::sirius_config cfg;
    cfg.load_from_file(path);
  } catch (std::exception const& e) {
    error = e.what();
  }
  std::error_code ec;
  std::filesystem::remove(path, ec);
  return error;
}

}  // namespace

TEST_CASE("object_store_config defaults are inert", "[object_store_config]")
{
  object_store_config cfg;

  CHECK(cfg.endpoint.empty());
  CHECK(cfg.region.empty());
  CHECK(cfg.access_key.empty());
  CHECK(cfg.secret_key.empty());
  CHECK(cfg.session_token.empty());
  CHECK(cfg.s3_transport == object_store_config::transport::AUTO);
  CHECK(cfg.s3_signing_mode == object_store_config::signing_mode::presigned);
}

TEST_CASE("object_store_config string_to_enum accepts known transports", "[object_store_config]")
{
  object_store_config::transport t = object_store_config::transport::RDMA;

  REQUIRE(string_to_enum("auto", t));
  CHECK(t == object_store_config::transport::AUTO);

  REQUIRE(string_to_enum("http", t));
  CHECK(t == object_store_config::transport::HTTP);

  REQUIRE(string_to_enum("https", t));
  CHECK(t == object_store_config::transport::HTTP);

  REQUIRE(string_to_enum("rdma", t));
  CHECK(t == object_store_config::transport::RDMA);
}

TEST_CASE("object_store_config string_to_enum rejects unknown transports", "[object_store_config]")
{
  auto t = object_store_config::transport::AUTO;

  CHECK_FALSE(string_to_enum("", t));
  CHECK(t == object_store_config::transport::AUTO);

  CHECK_FALSE(string_to_enum("smb", t));
  CHECK(t == object_store_config::transport::AUTO);

  CHECK_FALSE(string_to_enum("HTTP", t));
  CHECK(t == object_store_config::transport::AUTO);
}

TEST_CASE("object_store_config enum_to_string returns canonical names", "[object_store_config]")
{
  std::string out;

  REQUIRE(enum_to_string(object_store_config::transport::AUTO, out));
  CHECK(out == "auto");

  REQUIRE(enum_to_string(object_store_config::transport::HTTP, out));
  CHECK(out == "http");

  REQUIRE(enum_to_string(object_store_config::transport::RDMA, out));
  CHECK(out == "rdma");
}

TEST_CASE("object_store_config signing_mode string helpers round-trip",
          "[object_store_config][s3][config]")
{
  object_store_config::signing_mode mode = object_store_config::signing_mode::header;

  REQUIRE(string_to_enum("presigned", mode));
  CHECK(mode == object_store_config::signing_mode::presigned);

  REQUIRE(string_to_enum("header", mode));
  CHECK(mode == object_store_config::signing_mode::header);

  CHECK_FALSE(string_to_enum("", mode));
  CHECK(mode == object_store_config::signing_mode::header);

  CHECK_FALSE(string_to_enum("HEADER", mode));
  CHECK(mode == object_store_config::signing_mode::header);

  std::string out;
  REQUIRE(enum_to_string(object_store_config::signing_mode::presigned, out));
  CHECK(out == "presigned");
  REQUIRE(enum_to_string(object_store_config::signing_mode::header, out));
  CHECK(out == "header");
}

TEST_CASE("sirius_config loads object_store_config from YAML", "[object_store_config][s3][config]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_object_store_config.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  executor:\n"
           "    scan_manager:\n"
           "      object_store:\n"
           "        endpoint: http://127.0.0.1:9000\n"
           "        region: us-east-1\n"
           "        access_key: minioadmin\n"
           "        secret_key: minioadmin-secret\n"
           "        session_token: TESTSESSIONTOKEN\n"
           "        signing_mode: header\n"
           "        s3_transport: rdma\n";
    REQUIRE(out);
  }

  sirius::sirius_config cfg;
  cfg.load_from_file(path);

  auto const& os = cfg.get_scan_manager_config().object_store;
  CHECK(os.endpoint == "http://127.0.0.1:9000");
  CHECK(os.region == "us-east-1");
  CHECK(os.access_key == "minioadmin");
  CHECK(os.secret_key == "minioadmin-secret");
  CHECK(os.session_token == "TESTSESSIONTOKEN");
  CHECK(os.s3_signing_mode == object_store_config::signing_mode::header);
  CHECK(os.s3_transport == object_store_config::transport::RDMA);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config loads presigned object_store_config signing mode from YAML",
          "[object_store_config][s3][config]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_presigned_signing_mode.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  executor:\n"
           "    scan_manager:\n"
           "      object_store:\n"
           "        endpoint: http://127.0.0.1:9000\n"
           "        region: us-east-1\n"
           "        access_key: minioadmin\n"
           "        secret_key: minioadmin-secret\n"
           "        signing_mode: presigned\n";
    REQUIRE(out);
  }

  sirius::sirius_config cfg;
  cfg.load_from_file(path);

  CHECK(cfg.get_scan_manager_config().object_store.s3_signing_mode ==
        object_store_config::signing_mode::presigned);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config rejects unknown object_store_config signing modes",
          "[object_store_config][s3][config]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_bad_s3_signing_mode.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  executor:\n"
           "    scan_manager:\n"
           "      object_store:\n"
           "        endpoint: http://127.0.0.1:9000\n"
           "        region: us-east-1\n"
           "        access_key: minioadmin\n"
           "        secret_key: minioadmin-secret\n"
           "        signing_mode: query-string\n";
    REQUIRE(out);
  }

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(path));

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config rejects removed s3_use_async_backend object_store key",
          "[object_store_config][s3][config]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_removed_s3_async_key.yaml";
  write_yaml(path,
             "sirius:\n"
             "  executor:\n"
             "    scan_manager:\n"
             "      object_store:\n"
             "        endpoint: http://127.0.0.1:9000\n"
             "        region: us-east-1\n"
             "        access_key: minioadmin\n"
             "        secret_key: minioadmin-secret\n"
             "        s3_use_async_backend: false\n");

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(path));

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config defaults chunk prewarm to enabled when YAML omits the key",
          "[scan_manager][config][prefetching_cache]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_chunk_prewarm_default.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  executor:\n"
           "    scan_manager:\n"
           "      use_sirius_datasource: true\n";
    REQUIRE(out);
  }

  sirius::sirius_config cfg;
  cfg.load_from_file(path);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config parses rest perf instrumentation flag",
          "[scan_manager][config][s3][rest][perf]")
{
  CHECK_FALSE(sirius::io::rest::config{}.perf_instrumentation);
  CHECK(sirius::io::rest::config{}.footer_probe_bytes == 512UL * 1024);

  auto const path =
    std::filesystem::temp_directory_path() / "sirius_rest_perf_instrumentation.yaml";
  write_yaml(path,
             "sirius:\n"
             "  executor:\n"
             "    scan_manager:\n"
             "      rest:\n"
             "        perf_instrumentation: true\n"
             "        footer_probe_bytes: 256KiB\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(path));
  CHECK(cfg.get_scan_manager_config().rest.perf_instrumentation);
  CHECK(cfg.get_scan_manager_config().rest.footer_probe_bytes == 256UL * 1024);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config preserves an explicit REST bounce block size",
          "[scan_manager][config][s3][rest][bounce_grain]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_rest_bounce_block_size.yaml";
  write_yaml(path,
             "sirius:\n"
             "  executor:\n"
             "    scan_manager:\n"
             "      rest:\n"
             "        bounce_block_size: '8 MiB'\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(path));
  CHECK(cfg.get_scan_manager_config().rest.bounce_block_size == 8UL * 1024 * 1024);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config preflights the resolved REST bounce-span footprint",
          "[scan_manager][config][s3][rest][bounce_accounting]")
{
  auto const path =
    std::filesystem::temp_directory_path() / "sirius_rest_bounce_span_preflight.yaml";

  auto const expect_failure = [&](std::string const& yaml) {
    auto const error = load_config_error(path, yaml);
    INFO(error);
    REQUIRE_FALSE(error.empty());
    CHECK(error.find("needed") != std::string::npos);
    CHECK(error.find("limit") != std::string::npos);
    CHECK(error.find("shortfall") != std::string::npos);
  };

  SECTION("the exact 2 GiB pool boundary is legal")
  {
    auto const error = load_config_error(path,
                                         "sirius:\n"
                                         "  memory:\n"
                                         "    host:\n"
                                         "      capacity_bytes: 3GiB\n"
                                         "      reservation_limit_fraction: 0.5\n"
                                         "  executor:\n"
                                         "    scan_manager:\n"
                                         "      rest_n_reactors: 8\n"
                                         "      rest:\n"
                                         "        max_connections: 256\n"
                                         "        bounce_block_size: 1MiB\n");
    CHECK(error.empty());
  }

  SECTION("a pool larger than 2 GiB is rejected before runtime allocation")
  {
    expect_failure(
      "sirius:\n"
      "  memory:\n"
      "    host:\n"
      "      capacity_bytes: 16GiB\n"
      "  executor:\n"
      "    scan_manager:\n"
      "      rest_n_reactors: 8\n"
      "      rest:\n"
      "        max_connections: 256\n"
      "        bounce_block_size: 4MiB\n");
  }

  SECTION("reactor-count multiplication overflow is rejected")
  {
    auto const max_size = std::to_string(std::numeric_limits<std::int64_t>::max());
    expect_failure(
      "sirius:\n"
      "  memory:\n"
      "    host:\n"
      "      capacity_bytes: 16GiB\n"
      "  executor:\n"
      "    scan_manager:\n"
      "      rest_n_reactors: " +
      max_size +
      "\n"
      "      rest:\n"
      "        max_connections: 2\n"
      "        bounce_block_size: 1MiB\n");
  }

  SECTION("zero REST connections are rejected by the YAML validator")
  {
    auto const error = load_config_error(path,
                                         "sirius:\n"
                                         "  memory:\n"
                                         "    host:\n"
                                         "      capacity_bytes: 16MiB\n"
                                         "  executor:\n"
                                         "    scan_manager:\n"
                                         "      rest:\n"
                                         "        max_connections: 0\n"
                                         "        bounce_block_size: 1MiB\n");
    INFO(error);
    REQUIRE_FALSE(error.empty());
    CHECK(error.find("max_connections") != std::string::npos);
  }

  SECTION("an explicit space configuration without a HOST space is rejected")
  {
    expect_failure(
      "sirius:\n"
      "  space:\n"
      "    gpu:\n"
      "      - device_id: 0\n"
      "        memory_capacity: 4GiB\n"
      "  executor:\n"
      "    scan_manager:\n"
      "      rest:\n"
      "        max_connections: 2\n"
      "        bounce_block_size: 4MiB\n");
  }

  SECTION("the first resolved explicit HOST space, not a later larger one, controls admission")
  {
    expect_failure(
      "sirius:\n"
      "  space:\n"
      "    host:\n"
      "      - numa_id: 0\n"
      "        memory_capacity: 4MiB\n"
      "        reservation_limit_fraction: 1.0\n"
      "        initial_number_pools: 0\n"
      "      - numa_id: 1\n"
      "        memory_capacity: 64MiB\n"
      "        reservation_limit_fraction: 1.0\n"
      "        initial_number_pools: 0\n"
      "  executor:\n"
      "    scan_manager:\n"
      "      rest_n_reactors: 2\n"
      "      rest:\n"
      "        max_connections: 2\n"
      "        bounce_block_size: 4MiB\n");
  }

  SECTION("a resolved custom host block size still validates the configured grain")
  {
    expect_failure(
      "sirius:\n"
      "  space:\n"
      "    host:\n"
      "      - numa_id: 0\n"
      "        memory_capacity: 64MiB\n"
      "        reservation_limit_fraction: 1.0\n"
      "        block_size: 2MiB\n"
      "        initial_number_pools: 0\n"
      "  executor:\n"
      "    scan_manager:\n"
      "      rest:\n"
      "        max_connections: 1\n"
      "        bounce_block_size: 3MiB\n");
  }
}

TEST_CASE("sirius_config rejects unknown rest config keys", "[scan_manager][config][rest]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_rest_unknown_key.yaml";
  write_yaml(path,
             "sirius:\n"
             "  executor:\n"
             "    scan_manager:\n"
             "      rest:\n"
             "        perf_instrumentation_typo: true\n");

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(path));

  std::error_code ec;
  std::filesystem::remove(path, ec);
}
