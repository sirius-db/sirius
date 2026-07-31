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
#include "io/uri_parser.hpp"

#include <stdexcept>

using sirius::io::parse;

TEST_CASE("uri_parser parses bare absolute paths as file URIs", "[uri_parser]")
{
  auto parsed = parse("/tmp/sirius%20data.parquet?version=1&flag#ignored");

  CHECK(parsed.scheme == "file");
  CHECK(parsed.host.empty());
  CHECK(parsed.path == "/tmp/sirius data.parquet");
  REQUIRE(parsed.query.size() == 2);
  CHECK(parsed.query.at("version") == "1");
  CHECK(parsed.query.at("flag").empty());
}

TEST_CASE("uri_parser parses file scheme with absolute path", "[uri_parser]")
{
  auto parsed = parse("file:///var/data/table%20one.parquet?version=1#ignored");

  CHECK(parsed.scheme == "file");
  CHECK(parsed.host.empty());
  CHECK(parsed.path == "/var/data/table one.parquet");
  REQUIRE(parsed.query.size() == 1);
  CHECK(parsed.query.at("version") == "1");
}

TEST_CASE("uri_parser treats S3 object keys as literal bytes", "[uri_parser][s3]")
{
  auto parsed = parse("s3://bkt/a%20b");

  CHECK(parsed.scheme == "s3");
  CHECK(parsed.host == "bkt");
  CHECK(parsed.path == "a%20b");
  CHECK(parsed.query.empty());
}

TEST_CASE("uri_parser keeps query and fragment delimiters inside S3 keys", "[uri_parser][s3]")
{
  auto query_key = parse("s3://bkt/k?region=x");
  CHECK(query_key.path == "k?region=x");
  CHECK(query_key.query.empty());

  auto fragment_key = parse("s3://bkt/k#frag");
  CHECK(fragment_key.path == "k#frag");
  CHECK(fragment_key.query.empty());

  auto empty_query_key = parse("s3://bkt/key?=value");
  CHECK(empty_query_key.path == "key?=value");
  CHECK(empty_query_key.query.empty());
}

TEST_CASE("uri_parser accepts malformed percent sequences as literal S3 key bytes",
          "[uri_parser][s3]")
{
  CHECK(parse("s3://bkt/key%ZZ").path == "key%ZZ");
  CHECK(parse("s3://bkt/key%A").path == "key%A");
}

TEST_CASE("uri_parser applies the literal S3 path to uppercase schemes", "[uri_parser][s3]")
{
  auto parsed = parse("S3://bkt/a%20b");
  CHECK(parsed.scheme == "s3");
  CHECK(parsed.host == "bkt");
  CHECK(parsed.path == "a%20b");
  CHECK(parsed.query.empty());
}

TEST_CASE("uri_parser preserves S3 leading slashes in object key", "[uri_parser]")
{
  CHECK(parse("s3://bucket/key").path == "key");
  CHECK(parse("s3://bucket//key").path == "/key");
  CHECK(parse("s3://bucket///key").path == "//key");
  CHECK(parse("s3://bucket/a//b").path == "a//b");

  CHECK_THROWS_AS(parse("s3://bucket"), std::invalid_argument);
  CHECK_THROWS_AS(parse("s3://bucket/"), std::invalid_argument);
}

TEST_CASE("uri_parser accepts project-internal schemes", "[uri_parser]")
{
  auto parsed = parse("rdma_s3://bucket/a%20b?x=1#ignored");

  CHECK(parsed.scheme == "rdma_s3");
  CHECK(parsed.host == "bucket");
  CHECK(parsed.path == "a b");
  REQUIRE(parsed.query.size() == 1);
  CHECK(parsed.query.at("x") == "1");
}

TEST_CASE("uri_parser leaves non-S3 object-store URI semantics unchanged", "[uri_parser]")
{
  for (auto const scheme : {"gs", "azure", "http", "https"}) {
    DYNAMIC_SECTION("scheme=" << scheme)
    {
      auto parsed = parse(std::string{scheme} + "://bkt/a%20b?x=1#ignored");
      CHECK(parsed.scheme == scheme);
      CHECK(parsed.host == "bkt");
      CHECK(parsed.path == "a b");
      REQUIRE(parsed.query.size() == 1);
      CHECK(parsed.query.at("x") == "1");
    }
  }
}

TEST_CASE("uri_parser query parser decodes values and keeps last duplicate", "[uri_parser]")
{
  auto parsed = parse("gs://bucket/key?k=old&encoded=a%2Fb&k=new");

  REQUIRE(parsed.query.size() == 2);
  CHECK(parsed.query.at("k") == "new");
  CHECK(parsed.query.at("encoded") == "a/b");
}

TEST_CASE("uri_parser rejects malformed input", "[uri_parser]")
{
  CHECK_THROWS_AS(parse(""), std::invalid_argument);
  CHECK_THROWS_AS(parse("relative/file.parquet"), std::invalid_argument);
  CHECK_THROWS_AS(parse("./file.parquet"), std::invalid_argument);
  CHECK_THROWS_AS(parse("://bucket/key"), std::invalid_argument);
  CHECK_THROWS_AS(parse("file://relative/path"), std::invalid_argument);
  CHECK_THROWS_AS(parse("s3://bucket"), std::invalid_argument);
  CHECK_THROWS_AS(parse("s3://bucket/"), std::invalid_argument);
}
