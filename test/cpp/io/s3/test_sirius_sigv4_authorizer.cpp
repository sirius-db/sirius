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
#include "io/io_errors.hpp"
#include "io/object_store_config.hpp"
#include "io/rest/mock_authorizer.hpp"
#include "io/rest/s3/list_parser.hpp"
#include "io/rest/s3/sigv4_authorizer.hpp"
#include "io/rest/s3/static_credentials.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using sirius::io::credential_error;
using sirius::io::object_store_config;
using sirius::io::rest::authorized_request;
using sirius::io::rest::mock_authorizer;
using sirius::io::rest::object_ref;
using sirius::io::rest::request_method;
using sirius::io::rest::s3::sigv4_header_authorizer;
using sirius::io::rest::s3::sigv4_presigned_authorizer;
using sirius::io::rest::s3::static_credentials;
using sirius::io::rest::s3::static_credentials_from;

namespace {

constexpr auto k_presign_timeout = std::chrono::seconds{300};

static_credentials example_static_credentials()
{
  static_credentials creds;
  creds.access_key_id     = "AKIAIOSFODNN7EXAMPLE";
  creds.secret_access_key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
  return creds;
}

std::string query_string(std::string_view url)
{
  auto pos = url.find('?');
  REQUIRE(pos != std::string_view::npos);
  return std::string{url.substr(pos + 1)};
}

std::string query_value(std::string_view url, std::string_view key)
{
  auto query  = query_string(url);
  auto needle = std::string{key} + "=";
  auto begin  = query.find(needle);
  if (begin == std::string::npos) { return {}; }
  begin += needle.size();
  auto end = query.find('&', begin);
  if (end == std::string::npos) { end = query.size(); }
  return query.substr(begin, end - begin);
}

bool contains(std::string_view haystack, std::string_view needle)
{
  return haystack.find(needle) != std::string_view::npos;
}

bool starts_with(std::string_view s, std::string_view prefix)
{
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

bool ascii_iequals(std::string_view lhs, std::string_view rhs)
{
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](unsigned char a, unsigned char b) {
           return std::tolower(a) == std::tolower(b);
         });
}

std::string header_value(std::vector<std::pair<std::string, std::string>> const& headers,
                         std::string_view name)
{
  for (auto const& [key, value] : headers) {
    if (ascii_iequals(key, name)) { return value; }
  }
  return {};
}

bool is_lower_hex_64(std::string_view value)
{
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return std::isdigit(c) || (c >= 'a' && c <= 'f');
         });
}

std::vector<std::string> query_keys(std::string_view url)
{
  auto query = query_string(url);
  std::vector<std::string> keys;
  std::size_t pos = 0;
  while (pos < query.size()) {
    auto amp = query.find('&', pos);
    if (amp == std::string::npos) { amp = query.size(); }
    auto eq = query.find('=', pos);
    REQUIRE(eq != std::string::npos);
    REQUIRE(eq <= amp);
    keys.push_back(query.substr(pos, eq - pos));
    pos = amp + 1;
  }
  return keys;
}

class object_only_authorizer final : public sirius::io::rest::request_authorizer {
 public:
  authorized_request authorize(object_ref const& /*obj*/,
                               request_method /*method*/,
                               std::chrono::seconds /*timeout*/) override
  {
    return {"https://example.invalid/object", {}};
  }
};

}  // namespace

TEST_CASE("ListObjectsV2 parser extracts ordered keys, sizes, and pagination", "[s3][list_parser]")
{
  using sirius::io::rest::s3::parse_list_objects_v2;

  auto page = parse_list_objects_v2(
    R"(<?xml version="1.0" encoding="UTF-8"?>)"
    R"(<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">)"
    R"(<Name>bucket</Name><Prefix>lake/</Prefix>)"
    R"(<Contents><Key>lake/a&amp;b.parquet</Key><Size>12</Size></Contents>)"
    R"(<Contents><Key>lake/year=2024/file.parquet</Key><Size>0</Size></Contents>)"
    R"(<IsTruncated>true</IsTruncated>)"
    R"(<NextContinuationToken>token/with+chars=</NextContinuationToken>)"
    R"(</ListBucketResult>)");

  REQUIRE(page.entries.size() == 2);
  CHECK(page.entries[0].key == "lake/a&b.parquet");
  CHECK(page.entries[0].size == 12);
  CHECK(page.entries[1].key == "lake/year=2024/file.parquet");
  CHECK(page.entries[1].size == 0);
  CHECK(page.is_truncated);
  CHECK(page.next_continuation_token == "token/with+chars=");
}

TEST_CASE("ListObjectsV2 parser ignores non-object keys and preserves flat-key order",
          "[s3][list_parser]")
{
  using sirius::io::rest::s3::parse_list_objects_v2;

  auto page = parse_list_objects_v2(
    R"(<ListBucketResult>)"
    R"(<Key>not-an-object.parquet</Key>)"
    R"(<CommonPrefixes><Prefix>lake/year=2024/</Prefix><Key>also-not-object</Key></CommonPrefixes>)"
    R"(<Contents><Key>lake/part-000.parquet</Key><Size>7</Size></Contents>)"
    R"(<Contents><Key>lake/nested/year=2025/part-001.parquet</Key><Size>8</Size></Contents>)"
    R"(<IsTruncated>false</IsTruncated>)"
    R"(</ListBucketResult>)");

  REQUIRE(page.entries.size() == 2);
  CHECK(page.entries[0].key == "lake/part-000.parquet");
  CHECK(page.entries[0].size == 7);
  CHECK(page.entries[1].key == "lake/nested/year=2025/part-001.parquet");
  CHECK(page.entries[1].size == 8);
  CHECK_FALSE(page.is_truncated);
  CHECK(page.next_continuation_token.empty());
}

TEST_CASE("ListObjectsV2 parser rejects non-list bodies and malformed sizes", "[s3][list_parser]")
{
  using sirius::io::rest::s3::parse_list_objects_v2;

  auto empty = parse_list_objects_v2(
    R"(<ListBucketResult><IsTruncated>false</IsTruncated></ListBucketResult>)");
  CHECK(empty.entries.empty());

  CHECK_THROWS_AS(parse_list_objects_v2(
                    R"(<Error><Code>NoSuchBucket</Code><Message>no bucket</Message></Error>)"),
                  std::runtime_error);
  CHECK_THROWS_AS(parse_list_objects_v2(
                    R"(<ListBucketResult><Contents><Key>a</Key></Contents></ListBucketResult>)"),
                  std::runtime_error);
  CHECK_THROWS_AS(
    parse_list_objects_v2(
      R"(<ListBucketResult><Contents><Key>a</Key><Size>not-a-size</Size></Contents></ListBucketResult>)"),
    std::runtime_error);
  CHECK_THROWS_AS(
    parse_list_objects_v2(R"(<ListBucketResult><Contents><Key>a</Key><Size>1</Size></Contents>)"),
    std::runtime_error);
  CHECK_THROWS_AS(
    parse_list_objects_v2(
      R"(<ListBucketResult><Contents><Key>a</Key><Size>1</Size></Contents><Contents><Key>b</Key>)"),
    std::runtime_error);
  CHECK_THROWS_AS(
    parse_list_objects_v2(
      R"(<ListBucketResult><Contents><Key>a</Key><Size>99999999999999999999999999</Size></Contents></ListBucketResult>)"),
    std::runtime_error);

  auto max_size = parse_list_objects_v2(
    R"(<ListBucketResult><Contents><Key>a</Key><Size>18446744073709551615</Size></Contents><IsTruncated>false</IsTruncated></ListBucketResult>)");
  REQUIRE(max_size.entries.size() == 1);
  CHECK(max_size.entries[0].size == std::numeric_limits<std::uint64_t>::max());
}

TEST_CASE("ListObjectsV2 parser rejects Contents without a Key", "[s3][list_parser]")
{
  CHECK_THROWS_WITH(
    sirius::io::rest::s3::parse_list_objects_v2(
      R"(<ListBucketResult><Contents><Size>5</Size></Contents><IsTruncated>false</IsTruncated></ListBucketResult>)"),
    Catch::Contains("<Contents> without <Key>"));
}

TEST_CASE("ListObjectsV2 parser rejects an unclosed Key", "[s3][list_parser]")
{
  CHECK_THROWS_WITH(
    sirius::io::rest::s3::parse_list_objects_v2(
      R"(<ListBucketResult><Contents><Key>a<Size>5</Size></Contents><IsTruncated>false</IsTruncated></ListBucketResult>)"),
    Catch::Contains("<Contents> without <Key>"));
}

TEST_CASE("ListObjectsV2 parser rejects an empty Key", "[s3][list_parser]")
{
  CHECK_THROWS_WITH(
    sirius::io::rest::s3::parse_list_objects_v2(
      R"(<ListBucketResult><Contents><Key></Key><Size>5</Size></Contents><IsTruncated>false</IsTruncated></ListBucketResult>)"),
    Catch::Contains("empty <Key>"));
}

TEST_CASE("ListObjectsV2 parser requires IsTruncated", "[s3][list_parser]")
{
  CHECK_THROWS_WITH(
    sirius::io::rest::s3::parse_list_objects_v2(
      R"(<ListBucketResult><Contents><Key>a</Key><Size>5</Size></Contents></ListBucketResult>)"),
    Catch::Contains("missing <IsTruncated>"));
}

TEST_CASE("ListObjectsV2 parser rejects invalid IsTruncated values", "[s3][list_parser]")
{
  for (auto const value : {"TRUE", "1", "garbage"}) {
    DYNAMIC_SECTION("value=" << value)
    {
      auto const xml =
        std::string{
          "<ListBucketResult><Contents><Key>a</Key><Size>5</Size>"
          "</Contents><IsTruncated>"} +
        value + "</IsTruncated></ListBucketResult>";
      CHECK_THROWS_WITH(sirius::io::rest::s3::parse_list_objects_v2(xml),
                        Catch::Contains("invalid <IsTruncated>"));
    }
  }

  SECTION("unclosed element")
  {
    CHECK_THROWS_WITH(
      sirius::io::rest::s3::parse_list_objects_v2(
        R"(<ListBucketResult><Contents><Key>a</Key><Size>5</Size></Contents><IsTruncated>true</ListBucketResult>)"),
      Catch::Contains("missing <IsTruncated>"));
  }
}

TEST_CASE("ListObjectsV2 parser requires a token for a truncated page", "[s3][list_parser]")
{
  CHECK_THROWS_WITH(
    sirius::io::rest::s3::parse_list_objects_v2(
      R"(<ListBucketResult><Contents><Key>a</Key><Size>5</Size></Contents><IsTruncated>true</IsTruncated></ListBucketResult>)"),
    Catch::Contains("without") && Catch::Contains("ContinuationToken"));
}

TEST_CASE("ListObjectsV2 parser trims a valid IsTruncated value", "[s3][list_parser]")
{
  auto const page = sirius::io::rest::s3::parse_list_objects_v2(
    R"(<ListBucketResult><Contents><Key>a</Key><Size>5</Size></Contents><IsTruncated> true </IsTruncated><NextContinuationToken>next</NextContinuationToken></ListBucketResult>)");

  CHECK(page.is_truncated);
  CHECK(page.next_continuation_token == "next");
}

TEST_CASE("ListObjectsV2 parser rejects object entries after the root element", "[s3][list_parser]")
{
  CHECK_THROWS_WITH(
    sirius::io::rest::s3::parse_list_objects_v2(
      R"(<ListBucketResult><IsTruncated>false</IsTruncated></ListBucketResult><Contents><Key>outside</Key><Size>1</Size></Contents>)"),
    Catch::Contains("after </ListBucketResult>"));
}

TEST_CASE("ListObjectsV2 parser does not read IsTruncated outside the root", "[s3][list_parser]")
{
  CHECK_THROWS_WITH(sirius::io::rest::s3::parse_list_objects_v2(
                      R"(<ListBucketResult></ListBucketResult><IsTruncated>false</IsTruncated>)"),
                    Catch::Contains("missing <IsTruncated>"));
}

TEST_CASE("ListObjectsV2 parser does not read a continuation token outside the root",
          "[s3][list_parser]")
{
  CHECK_THROWS_WITH(
    sirius::io::rest::s3::parse_list_objects_v2(
      R"(<ListBucketResult><Contents><Key>a</Key><Size>1</Size></Contents><IsTruncated>true</IsTruncated></ListBucketResult><NextContinuationToken>outside</NextContinuationToken>)"),
    Catch::Contains("without") && Catch::Contains("ContinuationToken"));
}

TEST_CASE("ListObjectsV2 parser rejects a root close before the root open", "[s3][list_parser]")
{
  CHECK_THROWS_AS(sirius::io::rest::s3::parse_list_objects_v2(
                    R"(</ListBucketResult><ListBucketResult><IsTruncated>false</IsTruncated>)"),
                  std::runtime_error);
}

TEST_CASE("ListObjectsV2 parser accepts a prologue, root namespace, and trailing whitespace",
          "[s3][list_parser]")
{
  auto const page = sirius::io::rest::s3::parse_list_objects_v2(
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
    "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
    "<Contents><Key>a</Key><Size>1</Size></Contents>"
    "<IsTruncated>false</IsTruncated>"
    "</ListBucketResult> \n\t");

  REQUIRE(page.entries.size() == 1);
  CHECK(page.entries[0].key == "a");
  CHECK(page.entries[0].size == 1);
  CHECK_FALSE(page.is_truncated);
}

TEST_CASE("ListObjectsV2 parser rejects a root-name prefix collision", "[s3][list_parser]")
{
  CHECK_THROWS_WITH(
    sirius::io::rest::s3::parse_list_objects_v2(
      R"(<ListBucketResultBogus><IsTruncated>false</IsTruncated></ListBucketResult>)"),
    Catch::Contains("not a ListObjectsV2 response"));
}

TEST_CASE("ListObjectsV2 parser rejects content before the root element", "[s3][list_parser]")
{
  CHECK_THROWS_WITH(
    sirius::io::rest::s3::parse_list_objects_v2(
      R"(<Foo/><ListBucketResult><IsTruncated>false</IsTruncated></ListBucketResult>)"),
    Catch::Contains("before <ListBucketResult>"));
}

TEST_CASE("ListObjectsV2 parser accepts a prologue and newline before the root",
          "[s3][list_parser]")
{
  auto const page = sirius::io::rest::s3::parse_list_objects_v2(
    "<?xml version=\"1.0\"?>\n "
    "<ListBucketResult><IsTruncated>false</IsTruncated></ListBucketResult>");

  CHECK(page.entries.empty());
  CHECK_FALSE(page.is_truncated);
}

TEST_CASE("request_authorizer base rejects LIST until implementations opt in", "[s3][authorizer]")
{
  object_only_authorizer provider;

  CHECK_THROWS_AS(
    provider.authorize_list("bucket", "list-type=2&max-keys=1000&prefix=p%2F", k_presign_timeout),
    credential_error);
}

TEST_CASE("sigv4_presigned_authorizer signs sorted ListObjectsV2 query params", "[s3][authorizer]")
{
  auto creds          = example_static_credentials();
  creds.session_token = "temporary/session+token=";
  sigv4_presigned_authorizer provider(creds, "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto request =
    provider.authorize_list("bucket", "list-type=2&max-keys=1000&prefix=p%2F", k_presign_timeout);

  REQUIRE(request.headers.empty());
  CHECK(starts_with(request.url, "https://s3.us-east-1.amazonaws.com/bucket?"));
  CHECK(contains(request.url, "list-type=2"));
  CHECK(contains(request.url, "max-keys=1000"));
  CHECK(contains(request.url, "prefix=p%2F"));
  CHECK(contains(request.url, "X-Amz-Security-Token=temporary%2Fsession%2Btoken%3D"));
  CHECK(is_lower_hex_64(query_value(request.url, "X-Amz-Signature")));

  auto keys = query_keys(request.url);
  CHECK(std::is_sorted(keys.begin(), keys.end()));
}

TEST_CASE("sigv4_header_authorizer signs ListObjectsV2 canonical queries", "[s3][authorizer]")
{
  sigv4_header_authorizer provider(
    example_static_credentials(), "us-east-1", "http://minio.local:9000");

  auto request =
    provider.authorize_list("bucket",
                            "continuation-token=page%2F1%2B%3D&list-type=2&max-keys=1&prefix=p%2F",
                            k_presign_timeout);

  CHECK(request.url ==
        "http://minio.local:9000/bucket?continuation-token=page%2F1%2B%3D&list-type=2&max-keys=1&"
        "prefix=p%2F");
  CHECK_FALSE(contains(request.url, "X-Amz-Signature"));
  CHECK(starts_with(header_value(request.headers, "Authorization"), "AWS4-HMAC-SHA256 "));
  CHECK_FALSE(header_value(request.headers, "x-amz-date").empty());
  CHECK_FALSE(header_value(request.headers, "x-amz-content-sha256").empty());
}

TEST_CASE("SigV4 LIST rejects X-Amz query smuggling", "[s3][authorizer]")
{
  sigv4_presigned_authorizer presigned(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");
  sigv4_header_authorizer header(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  CHECK_THROWS(presigned.authorize_list(
    "bucket", "list-type=2&X-Amz-Signature=evil&prefix=p%2F", k_presign_timeout));
  CHECK_THROWS(header.authorize_list(
    "bucket", "list-type=2&x-amz-credential=evil&prefix=p%2F", k_presign_timeout));
}

TEST_CASE("sigv4_presigned_authorizer normalizes HTTPS endpoint", "[s3][authorizer]")
{
  sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-west-2", "HTTPS://S3.US-WEST-2.AMAZONAWS.COM");

  auto request =
    provider.authorize({"examplebucket", "test.txt"}, request_method::GET, k_presign_timeout);
  CHECK(request.headers.empty());
  auto const& url = request.url;

  CHECK(starts_with(url, "https://s3.us-west-2.amazonaws.com/examplebucket/test.txt?"));
  CHECK(query_value(url, "X-Amz-Credential").find("%2Fus-west-2%2Fs3%2Faws4_request") !=
        std::string::npos);
  CHECK(query_value(url, "X-Amz-SignedHeaders") == "host");
}

TEST_CASE("sigv4_presigned_authorizer preserves HTTP endpoint ports", "[s3][authorizer]")
{
  sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-east-1", "http://minio.local:9000");

  auto request =
    provider.authorize({"bucket", "object.parquet"}, request_method::GET, k_presign_timeout);
  CHECK(request.headers.empty());
  auto const& url = request.url;

  CHECK(starts_with(url, "http://minio.local:9000/bucket/object.parquet?"));
  CHECK(is_lower_hex_64(query_value(url, "X-Amz-Signature")));
}

TEST_CASE("sigv4_presigned_authorizer rejects malformed construction inputs", "[s3][authorizer]")
{
  auto creds = example_static_credentials();

  CHECK_THROWS_AS(sigv4_presigned_authorizer(creds, "us-east-1", ""), credential_error);
  CHECK_THROWS_AS(sigv4_presigned_authorizer(creds, "us-east-1", "s3.us-east-1.amazonaws.com"),
                  credential_error);
  CHECK_THROWS_AS(sigv4_presigned_authorizer(creds, "us-east-1", "ftp://example.com"),
                  credential_error);
  CHECK_THROWS_AS(sigv4_presigned_authorizer(creds, "us-east-1", "https://example.com/prefix"),
                  credential_error);
  CHECK_THROWS_AS(sigv4_presigned_authorizer(creds, "us-east-1", "https://example.com?x=1"),
                  credential_error);
  CHECK_THROWS_AS(sigv4_presigned_authorizer(creds, "us-east-1", "https://example.com#fragment"),
                  credential_error);

  auto no_access_key = creds;
  no_access_key.access_key_id.clear();
  CHECK_THROWS_AS(sigv4_presigned_authorizer(no_access_key, "us-east-1", "https://example.com"),
                  credential_error);

  auto no_secret_key = creds;
  no_secret_key.secret_access_key.clear();
  CHECK_THROWS_AS(sigv4_presigned_authorizer(no_secret_key, "us-east-1", "https://example.com"),
                  credential_error);

  CHECK_THROWS_AS(sigv4_presigned_authorizer(creds, "", "https://example.com"), credential_error);
  CHECK_THROWS_AS(
    sigv4_presigned_authorizer(creds, "us-east-1", "https://example.com", std::chrono::seconds{0}),
    credential_error);
}

TEST_CASE("sigv4_header_authorizer signs with headers and plain path-style URLs",
          "[s3][authorizer]")
{
  auto creds          = example_static_credentials();
  creds.session_token = "temporary/session+token=";
  sigv4_header_authorizer provider(creds, "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto get_request =
    provider.authorize({"examplebucket", "test.txt"}, request_method::GET, k_presign_timeout);
  auto head_request =
    provider.authorize({"examplebucket", "test.txt"}, request_method::HEAD, k_presign_timeout);

  CHECK(get_request.url == "https://s3.us-east-1.amazonaws.com/examplebucket/test.txt");
  CHECK_FALSE(contains(get_request.url, "X-Amz-Signature"));
  CHECK_FALSE(contains(get_request.url, "?"));

  auto get_auth = header_value(get_request.headers, "Authorization");
  REQUIRE(starts_with(get_auth, "AWS4-HMAC-SHA256 "));
  CHECK_FALSE(header_value(get_request.headers, "x-amz-date").empty());
  CHECK_FALSE(header_value(get_request.headers, "x-amz-content-sha256").empty());
  CHECK(header_value(get_request.headers, "x-amz-security-token") == creds.session_token);

  auto head_auth = header_value(head_request.headers, "Authorization");
  REQUIRE(starts_with(head_auth, "AWS4-HMAC-SHA256 "));
  CHECK(get_auth != head_auth);
}

TEST_CASE("sigv4_header_authorizer omits session-token header for long-lived keys",
          "[s3][authorizer]")
{
  sigv4_header_authorizer provider(
    example_static_credentials(), "us-east-1", "http://minio.local:9000");

  auto request = provider.authorize(
    {"bucket", "nested/object.parquet"}, request_method::GET, std::chrono::seconds{10});

  CHECK(request.url == "http://minio.local:9000/bucket/nested/object.parquet");
  CHECK_FALSE(contains(request.url, "X-Amz-"));
  CHECK(starts_with(header_value(request.headers, "Authorization"), "AWS4-HMAC-SHA256 "));
  CHECK(header_value(request.headers, "x-amz-security-token").empty());
}

TEST_CASE("sigv4_presigned_authorizer generates distinct GET and HEAD URLs", "[s3][authorizer]")
{
  sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto get_request =
    provider.authorize({"examplebucket", "test.txt"}, request_method::GET, k_presign_timeout);
  auto head_request =
    provider.authorize({"examplebucket", "test.txt"}, request_method::HEAD, k_presign_timeout);
  CHECK(get_request.headers.empty());
  CHECK(head_request.headers.empty());
  auto const& get_url  = get_request.url;
  auto const& head_url = head_request.url;

  CHECK(query_value(get_url, "X-Amz-SignedHeaders") == "host");
  CHECK(query_value(head_url, "X-Amz-SignedHeaders") == "host");
  CHECK(query_value(get_url, "X-Amz-Signature") != query_value(head_url, "X-Amz-Signature"));
}

TEST_CASE("sigv4_presigned_authorizer encodes bucket and key path components", "[s3][authorizer]")
{
  sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto spaced =
    provider
      .authorize({"bucket", "path with space.parquet"}, request_method::GET, k_presign_timeout)
      .url;
  CHECK(
    starts_with(spaced, "https://s3.us-east-1.amazonaws.com/bucket/path%20with%20space.parquet?"));

  auto nested =
    provider.authorize({"bucket", "a/b/c.parquet"}, request_method::GET, k_presign_timeout).url;
  CHECK(starts_with(nested, "https://s3.us-east-1.amazonaws.com/bucket/a/b/c.parquet?"));
  CHECK_FALSE(contains(nested, "a%2Fb%2Fc.parquet"));

  auto leading = provider.authorize({"bucket", "/foo"}, request_method::GET, k_presign_timeout).url;
  CHECK(starts_with(leading, "https://s3.us-east-1.amazonaws.com/bucket//foo?"));

  auto unicode_key =
    provider
      .authorize(
        {"bucket", "\xE4\xB8\xAD\xE6\x96\x87.parquet"}, request_method::GET, k_presign_timeout)
      .url;
  CHECK(starts_with(unicode_key,
                    "https://s3.us-east-1.amazonaws.com/bucket/%E4%B8%AD%E6%96%87.parquet?"));
}

TEST_CASE("sigv4_presigned_authorizer propagates session tokens", "[s3][authorizer]")
{
  auto creds          = example_static_credentials();
  creds.session_token = "temporary/session+token=";
  sigv4_presigned_authorizer provider(creds, "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto request =
    provider.authorize({"examplebucket", "test.txt"}, request_method::GET, k_presign_timeout);
  CHECK(request.headers.empty());
  auto const& url = request.url;

  CHECK(contains(url, "X-Amz-Security-Token=temporary%2Fsession%2Btoken%3D"));
}

TEST_CASE("static_credentials_from maps object_store_config session tokens into SigV4 URLs",
          "[s3][authorizer]")
{
  object_store_config cfg;
  cfg.endpoint      = "https://s3.us-east-1.amazonaws.com";
  cfg.region        = "us-east-1";
  cfg.access_key    = "AKIAIOSFODNN7EXAMPLE";
  cfg.secret_key    = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
  cfg.session_token = "temporary/session+token=";

  auto creds = static_credentials_from(cfg);
  CHECK(creds.access_key_id == cfg.access_key);
  CHECK(creds.secret_access_key == cfg.secret_key);
  CHECK(creds.session_token == cfg.session_token);

  sigv4_presigned_authorizer token_provider(creds, cfg.region, cfg.endpoint);
  auto token_url =
    token_provider.authorize({"examplebucket", "test.txt"}, request_method::GET, k_presign_timeout)
      .url;
  CHECK(contains(token_url, "X-Amz-Security-Token=temporary%2Fsession%2Btoken%3D"));

  cfg.session_token.clear();
  auto no_token_creds = static_credentials_from(cfg);
  CHECK(no_token_creds.session_token.empty());

  sigv4_presigned_authorizer no_token_provider(no_token_creds, cfg.region, cfg.endpoint);
  auto no_token_url =
    no_token_provider
      .authorize({"examplebucket", "test.txt"}, request_method::GET, k_presign_timeout)
      .url;
  CHECK_FALSE(contains(no_token_url, "X-Amz-Security-Token="));
}

TEST_CASE("sigv4_presigned_authorizer honors per-call timeout", "[s3][authorizer]")
{
  auto creds          = example_static_credentials();
  creds.session_token = "temporary/session+token=";
  sigv4_presigned_authorizer provider(
    creds, "us-east-1", "https://s3.us-east-1.amazonaws.com", std::chrono::minutes{30});

  auto short_request = provider.authorize(
    {"examplebucket", "test.txt"}, request_method::GET, std::chrono::seconds{37});
  auto long_request = provider.authorize(
    {"examplebucket", "test.txt"}, request_method::GET, std::chrono::seconds{1800});
  auto head_request = provider.authorize(
    {"examplebucket", "test.txt"}, request_method::HEAD, std::chrono::seconds{37});
  CHECK(short_request.headers.empty());
  CHECK(long_request.headers.empty());
  CHECK(head_request.headers.empty());
  auto const& short_url = short_request.url;
  auto const& long_url  = long_request.url;
  auto const& head_url  = head_request.url;

  CHECK(query_value(short_url, "X-Amz-Expires") == "37");
  CHECK(query_value(long_url, "X-Amz-Expires") == "1800");
  CHECK(starts_with(short_url, "https://s3.us-east-1.amazonaws.com/examplebucket/test.txt?"));
  CHECK(query_value(short_url, "X-Amz-SignedHeaders") == "host");
  CHECK(is_lower_hex_64(query_value(short_url, "X-Amz-Signature")));
  CHECK(query_value(short_url, "X-Amz-Signature") != query_value(head_url, "X-Amz-Signature"));
  CHECK(contains(short_url, "X-Amz-Security-Token=temporary%2Fsession%2Btoken%3D"));
}

TEST_CASE("sigv4_presigned_authorizer rejects empty object references", "[s3][authorizer]")
{
  sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  CHECK_THROWS_AS(provider.authorize({"", "test.txt"}, request_method::GET, k_presign_timeout),
                  credential_error);
  CHECK_THROWS_AS(provider.authorize({"bucket", ""}, request_method::GET, k_presign_timeout),
                  credential_error);
}

TEST_CASE("sigv4_presigned_authorizer is safe under concurrent presigning", "[s3][authorizer]")
{
  sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  constexpr int n_threads = 8;
  constexpr int n_iters   = 25;
  std::atomic<int> malformed{0};
  std::vector<std::thread> threads;
  threads.reserve(n_threads);

  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&provider, &malformed, t] {
      for (int i = 0; i < n_iters; ++i) {
        auto url = provider
                     .authorize({"bucket", "key-" + std::to_string(t) + ".parquet"},
                                request_method::GET,
                                k_presign_timeout)
                     .url;
        if (!starts_with(url, "https://s3.us-east-1.amazonaws.com/bucket/key-") ||
            query_value(url, "X-Amz-SignedHeaders") != "host" ||
            !is_lower_hex_64(query_value(url, "X-Amz-Signature"))) {
          ++malformed;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  CHECK(malformed.load() == 0);
}

TEST_CASE("mock_authorizer returns canned URLs and records calls", "[s3][authorizer]")
{
  mock_authorizer provider(
    authorized_request{"https://signed.example/object", {{"x-test-header", "one"}}});

  auto get_request = provider.authorize({"bucket", "key"}, request_method::GET, k_presign_timeout);
  CHECK(get_request.url == "https://signed.example/object");
  CHECK(get_request.headers ==
        std::vector<std::pair<std::string, std::string>>{{"x-test-header", "one"}});
  auto head_request =
    provider.authorize({"bucket", "head-key"}, request_method::HEAD, k_presign_timeout);
  CHECK(head_request.url == "https://signed.example/object");
  CHECK(head_request.headers ==
        std::vector<std::pair<std::string, std::string>>{{"x-test-header", "one"}});

  CHECK(provider.call_count() == 2);
  CHECK(provider.get_count() == 1);
  CHECK(provider.head_count() == 1);
  CHECK(provider.last_bucket() == "bucket");
  CHECK(provider.last_key() == "head-key");
  CHECK(provider.last_timeout() == k_presign_timeout);
}

TEST_CASE("mock_authorizer can force credential errors", "[s3][authorizer]")
{
  mock_authorizer provider(authorized_request{"https://signed.example/object", {}});
  provider.set_throw("boom");

  CHECK_THROWS_AS(provider.authorize({"bucket", "key"}, request_method::GET, k_presign_timeout),
                  credential_error);

  provider.clear_throw();
  auto request = provider.authorize({"bucket", "key"}, request_method::GET, k_presign_timeout);
  CHECK(request.url == "https://signed.example/object");
  CHECK(request.headers.empty());
}
