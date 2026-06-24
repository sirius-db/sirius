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
#include "sirius_config.hpp"

#include <filesystem>
#include <fstream>
#include <string>

using sirius::io::enum_to_string;
using sirius::io::object_store_config;
using sirius::io::string_to_enum;

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
           "  object_store_config:\n"
           "    endpoint: http://127.0.0.1:9000\n"
           "    region: us-east-1\n"
           "    access_key: minioadmin\n"
           "    secret_key: minioadmin-secret\n"
           "    session_token: TESTSESSIONTOKEN\n"
           "    signing_mode: header\n"
           "    s3_transport: rdma\n";
    REQUIRE(out);
  }

  sirius::sirius_config cfg;
  cfg.load_from_file(path);

  CHECK(cfg.object_store_config.endpoint == "http://127.0.0.1:9000");
  CHECK(cfg.object_store_config.region == "us-east-1");
  CHECK(cfg.object_store_config.access_key == "minioadmin");
  CHECK(cfg.object_store_config.secret_key == "minioadmin-secret");
  CHECK(cfg.object_store_config.session_token == "TESTSESSIONTOKEN");
  CHECK(cfg.object_store_config.s3_signing_mode == object_store_config::signing_mode::header);
  CHECK(cfg.object_store_config.s3_transport == object_store_config::transport::RDMA);

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
           "  object_store_config:\n"
           "    endpoint: http://127.0.0.1:9000\n"
           "    region: us-east-1\n"
           "    access_key: minioadmin\n"
           "    secret_key: minioadmin-secret\n"
           "    signing_mode: presigned\n";
    REQUIRE(out);
  }

  sirius::sirius_config cfg;
  cfg.load_from_file(path);

  CHECK(cfg.object_store_config.s3_signing_mode == object_store_config::signing_mode::presigned);

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
           "  object_store_config:\n"
           "    endpoint: http://127.0.0.1:9000\n"
           "    region: us-east-1\n"
           "    access_key: minioadmin\n"
           "    secret_key: minioadmin-secret\n"
           "    signing_mode: query-string\n";
    REQUIRE(out);
  }

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
