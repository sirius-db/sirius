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

#include <string>
#include <string_view>

using sirius::io::enum_to_string;
using sirius::io::object_store_config;
using sirius::io::string_to_enum;

// The DuckDB SET callback path can't be exercised without a live
// DatabaseInstance + ClientContext, so the round-trip of "SET s3_transport =
// 'rdma'" is covered by test/sql/datasource/set_s3_transport.test. Here we
// cover only the struct semantics and the enum<->string helpers those
// callbacks delegate to.

TEST_CASE("object_store_config defaults", "[config][object_store_config]")
{
  object_store_config cfg;
  CHECK(cfg.endpoint.empty());
  CHECK(cfg.region.empty());
  CHECK(cfg.access_key.empty());
  CHECK(cfg.secret_key.empty());
  CHECK(cfg.s3_transport == object_store_config::transport::AUTO);
}

TEST_CASE("sirius_config exposes mutable object_store_config", "[config][object_store_config]")
{
  sirius::sirius_config sc;
  // mutation through the non-const accessor must persist.
  sc.get_object_store_config().endpoint   = "https://s3.example.com";
  sc.get_object_store_config().region     = "us-west-2";
  sc.get_object_store_config().access_key = "AKIA_TEST";
  sc.get_object_store_config().secret_key = "shh";
  sc.get_object_store_config().s3_transport = object_store_config::transport::RDMA;

  auto const& view = sc.get_object_store_config();
  CHECK(view.endpoint == "https://s3.example.com");
  CHECK(view.region == "us-west-2");
  CHECK(view.access_key == "AKIA_TEST");
  CHECK(view.secret_key == "shh");
  CHECK(view.s3_transport == object_store_config::transport::RDMA);
}

TEST_CASE("set_s3_transport_rdma_updates_config", "[config][object_store_config]")
{
  sirius::sirius_config sc;
  REQUIRE(sc.get_object_store_config().s3_transport == object_store_config::transport::AUTO);

  object_store_config::transport t{};
  REQUIRE(string_to_enum(std::string_view{"rdma"}, t));
  sc.get_object_store_config().s3_transport = t;
  CHECK(sc.get_object_store_config().s3_transport == object_store_config::transport::RDMA);

  REQUIRE(string_to_enum(std::string_view{"http"}, t));
  sc.get_object_store_config().s3_transport = t;
  CHECK(sc.get_object_store_config().s3_transport == object_store_config::transport::HTTP);

  // 'https' is accepted as an alias for HTTP — the transport distinction is
  // RDMA vs. non-RDMA; TLS is a separate knob.
  REQUIRE(string_to_enum(std::string_view{"https"}, t));
  CHECK(t == object_store_config::transport::HTTP);

  REQUIRE(string_to_enum(std::string_view{"auto"}, t));
  sc.get_object_store_config().s3_transport = t;
  CHECK(sc.get_object_store_config().s3_transport == object_store_config::transport::AUTO);
}

TEST_CASE("set_s3_endpoint_updates_config", "[config][object_store_config]")
{
  sirius::sirius_config sc;
  REQUIRE(sc.get_object_store_config().endpoint.empty());

  sc.get_object_store_config().endpoint = "https://minio.internal:9000";
  CHECK(sc.get_object_store_config().endpoint == "https://minio.internal:9000");

  // Setting to empty is valid (means "fall back to AWS default").
  sc.get_object_store_config().endpoint.clear();
  CHECK(sc.get_object_store_config().endpoint.empty());
}

TEST_CASE("unknown_s3_transport_value_rejected", "[config][object_store_config]")
{
  object_store_config::transport t = object_store_config::transport::AUTO;

  // Unknown tokens must not mutate the out-param nor return true. The
  // DuckDB SET callback translates the false return into an
  // InvalidInputException, but we verify the helper itself here.
  CHECK_FALSE(string_to_enum(std::string_view{"bogus"}, t));
  CHECK(t == object_store_config::transport::AUTO);

  CHECK_FALSE(string_to_enum(std::string_view{""}, t));
  CHECK(t == object_store_config::transport::AUTO);

  // Case-sensitive, consistent with cache_level's string_to_enum. Users are
  // expected to write the lowercase form ('rdma' / 'http' / 'auto').
  CHECK_FALSE(string_to_enum(std::string_view{"RDMA"}, t));
  CHECK(t == object_store_config::transport::AUTO);
}

TEST_CASE("enum_to_string round-trips all transports", "[config][object_store_config]")
{
  std::string s;
  REQUIRE(enum_to_string(object_store_config::transport::AUTO, s));
  CHECK(s == "auto");
  REQUIRE(enum_to_string(object_store_config::transport::HTTP, s));
  CHECK(s == "http");
  REQUIRE(enum_to_string(object_store_config::transport::RDMA, s));
  CHECK(s == "rdma");

  // The string produced by enum_to_string must round-trip through string_to_enum.
  for (auto t : {object_store_config::transport::AUTO,
                 object_store_config::transport::HTTP,
                 object_store_config::transport::RDMA}) {
    REQUIRE(enum_to_string(t, s));
    object_store_config::transport back{};
    REQUIRE(string_to_enum(std::string_view{s}, back));
    CHECK(back == t);
  }
}
