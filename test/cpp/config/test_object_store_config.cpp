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

#include <filesystem>
#include <fstream>
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
           "        s3_transport: rdma\n"
           "        ca_bundle_path: /tmp/test-ca.pem\n"
           "        tls_verify: false\n";
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
  CHECK(os.ca_bundle_path == "/tmp/test-ca.pem");
  CHECK_FALSE(os.tls_verify);

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
           "      backend: sirius\n";
    REQUIRE(out);
  }

  sirius::sirius_config cfg;
  cfg.load_from_file(path);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config rejects unknown rest config keys", "[scan_manager][config][rest]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_rest_unknown_key.yaml";
  write_yaml(path,
             "sirius:\n"
             "  executor:\n"
             "    scan_manager:\n"
             "      rest:\n"
             "        unknown_rest_option: true\n");

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(path));

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config rejects shadowed REST TLS YAML keys",
          "[scan_manager][config][s3][rest][tls]")
{
  auto check_rejected = [](std::string const& key, std::string const& value) {
    auto const path =
      std::filesystem::temp_directory_path() / ("sirius_shadowed_rest_" + key + ".yaml");
    write_yaml(path,
               "sirius:\n"
               "  executor:\n"
               "    scan_manager:\n"
               "      rest:\n"
               "        " +
                 key + ": " + value + "\n");

    sirius::sirius_config cfg;
    REQUIRE_THROWS_WITH(
      cfg.load_from_file(path),
      Catch::Contains("'sirius.executor.scan_manager.rest." + key + "': removed; configure '") &&
        Catch::Contains("sirius.executor.scan_manager.object_store." + key + "' instead"));

    std::error_code ec;
    std::filesystem::remove(path, ec);
  };

  SECTION("CA bundle") { check_rejected("ca_bundle_path", "/tmp/shadowed-ca.pem"); }
  SECTION("TLS verification") { check_rejected("tls_verify", "false"); }
}

TEST_CASE("sirius_config still loads unrelated REST YAML fields",
          "[scan_manager][config][s3][rest]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_rest_unrelated_fields.yaml";
  write_yaml(path,
             "sirius:\n"
             "  executor:\n"
             "    scan_manager:\n"
             "      rest:\n"
             "        request_timeout_s: 11\n"
             "        merge_max_gap: 1MiB\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(path));
  CHECK(cfg.get_scan_manager_config().rest.request_timeout_s == 11);
  CHECK(cfg.get_scan_manager_config().rest.merge_max_gap == 1UL << 20);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("sirius_config keeps REST connection count internal", "[scan_manager][config][rest]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_rest_max_connections.yaml";
  write_yaml(path,
             "sirius:\n"
             "  executor:\n"
             "    scan_manager:\n"
             "      rest:\n"
             "        max_connections: 7\n");

  sirius::sirius_config cfg;
  REQUIRE_THROWS_WITH(cfg.load_from_file(path),
                      Catch::Contains("unknown config key: 'max_connections' in rest"));
  CHECK(cfg.get_scan_manager_config().rest.max_connections == 64);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}
