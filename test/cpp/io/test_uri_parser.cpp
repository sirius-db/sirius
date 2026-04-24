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

#include <random>
#include <stdexcept>
#include <string>

using sirius::io::parse;
using sirius::io::parsed_uri;

TEST_CASE("parse: s3 scheme with bucket and key", "[uri]")
{
  auto p = parse("s3://bucket/key");
  CHECK(p.scheme == "s3");
  CHECK(p.host == "bucket");
  CHECK(p.path == "key");
  CHECK(p.query.empty());
}

TEST_CASE("parse: file:// URI keeps leading slash on path", "[uri]")
{
  auto p = parse("file:///abs/path.parquet");
  CHECK(p.scheme == "file");
  CHECK(p.host.empty());
  CHECK(p.path == "/abs/path.parquet");
  CHECK(p.query.empty());
}

TEST_CASE("parse: bare absolute path -> file scheme", "[uri]")
{
  auto p = parse("/abs/path.parquet");
  CHECK(p.scheme == "file");
  CHECK(p.host.empty());
  CHECK(p.path == "/abs/path.parquet");
  CHECK(p.query.empty());
}

TEST_CASE("parse: gs scheme", "[uri]")
{
  auto p = parse("gs://bucket/obj");
  CHECK(p.scheme == "gs");
  CHECK(p.host == "bucket");
  CHECK(p.path == "obj");
}

TEST_CASE("parse: azure scheme", "[uri]")
{
  auto p = parse("azure://container/blob");
  CHECK(p.scheme == "azure");
  CHECK(p.host == "container");
  CHECK(p.path == "blob");
}

TEST_CASE("parse: single query parameter extracted", "[uri]")
{
  auto p = parse("s3://bucket/k?region=us-west-2");
  CHECK(p.host == "bucket");
  CHECK(p.path == "k");
  REQUIRE(p.query.count("region") == 1);
  CHECK(p.query.at("region") == "us-west-2");
}

TEST_CASE("parse: multiple query parameters extracted", "[uri]")
{
  auto p = parse("s3://bucket/k?region=us-west-2&sse=AES256");
  REQUIRE(p.query.size() == 2);
  CHECK(p.query.at("region") == "us-west-2");
  CHECK(p.query.at("sse") == "AES256");
}

TEST_CASE("parse: percent-decoded path", "[uri]")
{
  auto p = parse("s3://bucket/my%20key");
  CHECK(p.path == "my key");
}

TEST_CASE("parse: percent-decoded query value", "[uri]")
{
  auto p = parse("s3://bucket/k?tag=a%20b");
  CHECK(p.query.at("tag") == "a b");
}

TEST_CASE("parse: s3 with trailing slash and no key rejected", "[uri]")
{
  CHECK_THROWS_AS(parse("s3://bucket/"), std::invalid_argument);
}

TEST_CASE("parse: s3 with no key rejected", "[uri]")
{
  CHECK_THROWS_AS(parse("s3://bucket"), std::invalid_argument);
}

TEST_CASE("parse: relative path rejected", "[uri]")
{
  CHECK_THROWS_AS(parse("relative/path"), std::invalid_argument);
  CHECK_THROWS_AS(parse("./x"), std::invalid_argument);
  CHECK_THROWS_AS(parse("file.parquet"), std::invalid_argument);
}

TEST_CASE("parse: empty URI rejected", "[uri]")
{
  CHECK_THROWS_AS(parse(""), std::invalid_argument);
}

TEST_CASE("parse: double-slash after authority collapses", "[uri]")
{
  auto p = parse("s3://bucket//key");
  CHECK(p.host == "bucket");
  CHECK(p.path == "key");
}

TEST_CASE("parse: query with no equals sign yields empty value", "[uri]")
{
  auto p = parse("s3://bucket/k?flag");
  REQUIRE(p.query.count("flag") == 1);
  CHECK(p.query.at("flag").empty());
}

TEST_CASE("parse: query with empty value", "[uri]")
{
  auto p = parse("s3://bucket/k?empty=");
  REQUIRE(p.query.count("empty") == 1);
  CHECK(p.query.at("empty").empty());
}

TEST_CASE("parse: duplicate query keys — last wins", "[uri]")
{
  auto p = parse("s3://bucket/k?a=1&a=2");
  REQUIRE(p.query.size() == 1);
  CHECK(p.query.at("a") == "2");
}

TEST_CASE("parse: empty query key rejected", "[uri]")
{
  CHECK_THROWS_AS(parse("s3://bucket/k?=nope"), std::invalid_argument);
}

TEST_CASE("parse: malformed percent-encoding rejected", "[uri]")
{
  CHECK_THROWS_AS(parse("s3://bucket/k%ZZ"), std::invalid_argument);
  CHECK_THROWS_AS(parse("s3://bucket/k%A"), std::invalid_argument);
  CHECK_THROWS_AS(parse("s3://bucket/k%"), std::invalid_argument);
}

TEST_CASE("parse: uppercase scheme normalized; fragment stripped", "[uri]")
{
  auto p = parse("S3://bucket/key#frag");
  CHECK(p.scheme == "s3");
  CHECK(p.host == "bucket");
  CHECK(p.path == "key");
  CHECK(p.query.empty());
}

TEST_CASE("parse: host keeps port verbatim", "[uri]")
{
  auto p = parse("s3://bucket:9000/key");
  CHECK(p.host == "bucket:9000");
  CHECK(p.path == "key");
}

TEST_CASE("parse: empty scheme rejected", "[uri]")
{
  CHECK_THROWS_AS(parse("://nopath"), std::invalid_argument);
}

TEST_CASE("parse: empty host on object-store scheme rejected", "[uri]")
{
  CHECK_THROWS_AS(parse("s3:///key"), std::invalid_argument);
}

TEST_CASE("parse: fuzzy 10k random inputs never crash and never throw unexpected types", "[uri]")
{
  // Fixed seed for reproducibility. Alphabet includes the structural delimiters
  // the parser must navigate plus some benign content.
  static constexpr char kAlphabet[]          = "ABCabc012:/?&=%#-._~";
  static constexpr std::size_t kAlphabetSize = sizeof(kAlphabet) - 1;

  std::mt19937_64 rng{0xC0FFEEULL};
  std::uniform_int_distribution<std::size_t> len_dist{0, 128};
  std::uniform_int_distribution<std::size_t> ch_dist{0, kAlphabetSize - 1};

  std::string buf;
  for (int i = 0; i < 10'000; ++i) {
    buf.clear();
    auto len = len_dist(rng);
    buf.reserve(len);
    for (std::size_t j = 0; j < len; ++j)
      buf.push_back(kAlphabet[ch_dist(rng)]);

    try {
      (void)parse(buf);
    } catch (std::invalid_argument const&) {
      // Expected for structurally-invalid inputs.
    } catch (std::exception const& e) {
      FAIL("parse(\"" << buf << "\") threw unexpected exception type: " << e.what());
    } catch (...) {
      FAIL("parse(\"" << buf << "\") threw unexpected non-std exception");
    }
  }
}
