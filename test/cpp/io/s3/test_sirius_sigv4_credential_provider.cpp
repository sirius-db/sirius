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
#include "io/s3/mock_credential_provider.hpp"
#include "io/s3/sirius_sigv4_credential_provider.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using sirius::io::credential_error;
using sirius::io::s3::mock_credential_provider;
using sirius::io::s3::presign_method;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::sirius_sigv4_credential_provider;
using sirius::io::s3::static_credentials;

namespace {

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

bool is_lower_hex_64(std::string_view value)
{
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return std::isdigit(c) || (c >= 'a' && c <= 'f');
         });
}

}  // namespace

TEST_CASE("sirius_sigv4_credential_provider normalizes HTTPS endpoint", "[s3][credential_provider]")
{
  sirius_sigv4_credential_provider provider(
    example_static_credentials(), "us-west-2", "HTTPS://S3.US-WEST-2.AMAZONAWS.COM");

  auto url = provider.get_presigned_url({"examplebucket", "test.txt"}, presign_method::GET);

  CHECK(starts_with(url, "https://s3.us-west-2.amazonaws.com/examplebucket/test.txt?"));
  CHECK(query_value(url, "X-Amz-Credential").find("%2Fus-west-2%2Fs3%2Faws4_request") !=
        std::string::npos);
  CHECK(query_value(url, "X-Amz-SignedHeaders") == "host");
}

TEST_CASE("sirius_sigv4_credential_provider preserves HTTP endpoint ports",
          "[s3][credential_provider]")
{
  sirius_sigv4_credential_provider provider(
    example_static_credentials(), "us-east-1", "http://minio.local:9000");

  auto url = provider.get_presigned_url({"bucket", "object.parquet"}, presign_method::GET);

  CHECK(starts_with(url, "http://minio.local:9000/bucket/object.parquet?"));
  CHECK(is_lower_hex_64(query_value(url, "X-Amz-Signature")));
}

TEST_CASE("sirius_sigv4_credential_provider rejects malformed construction inputs",
          "[s3][credential_provider]")
{
  auto creds = example_static_credentials();

  CHECK_THROWS_AS(sirius_sigv4_credential_provider(creds, "us-east-1", ""), credential_error);
  CHECK_THROWS_AS(
    sirius_sigv4_credential_provider(creds, "us-east-1", "s3.us-east-1.amazonaws.com"),
    credential_error);
  CHECK_THROWS_AS(sirius_sigv4_credential_provider(creds, "us-east-1", "ftp://example.com"),
                  credential_error);
  CHECK_THROWS_AS(
    sirius_sigv4_credential_provider(creds, "us-east-1", "https://example.com/prefix"),
    credential_error);
  CHECK_THROWS_AS(sirius_sigv4_credential_provider(creds, "us-east-1", "https://example.com?x=1"),
                  credential_error);
  CHECK_THROWS_AS(
    sirius_sigv4_credential_provider(creds, "us-east-1", "https://example.com#fragment"),
    credential_error);

  auto no_access_key = creds;
  no_access_key.access_key_id.clear();
  CHECK_THROWS_AS(
    sirius_sigv4_credential_provider(no_access_key, "us-east-1", "https://example.com"),
    credential_error);

  auto no_secret_key = creds;
  no_secret_key.secret_access_key.clear();
  CHECK_THROWS_AS(
    sirius_sigv4_credential_provider(no_secret_key, "us-east-1", "https://example.com"),
    credential_error);

  CHECK_THROWS_AS(sirius_sigv4_credential_provider(creds, "", "https://example.com"),
                  credential_error);
  CHECK_THROWS_AS(sirius_sigv4_credential_provider(
                    creds, "us-east-1", "https://example.com", std::chrono::seconds{0}),
                  credential_error);
}

TEST_CASE("sirius_sigv4_credential_provider generates distinct GET and HEAD URLs",
          "[s3][credential_provider]")
{
  sirius_sigv4_credential_provider provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto get_url  = provider.get_presigned_url({"examplebucket", "test.txt"}, presign_method::GET);
  auto head_url = provider.get_presigned_url({"examplebucket", "test.txt"}, presign_method::HEAD);

  CHECK(query_value(get_url, "X-Amz-SignedHeaders") == "host");
  CHECK(query_value(head_url, "X-Amz-SignedHeaders") == "host");
  CHECK(query_value(get_url, "X-Amz-Signature") != query_value(head_url, "X-Amz-Signature"));
}

TEST_CASE("sirius_sigv4_credential_provider encodes bucket and key path components",
          "[s3][credential_provider]")
{
  sirius_sigv4_credential_provider provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto spaced =
    provider.get_presigned_url({"bucket", "path with space.parquet"}, presign_method::GET);
  CHECK(
    starts_with(spaced, "https://s3.us-east-1.amazonaws.com/bucket/path%20with%20space.parquet?"));

  auto nested = provider.get_presigned_url({"bucket", "a/b/c.parquet"}, presign_method::GET);
  CHECK(starts_with(nested, "https://s3.us-east-1.amazonaws.com/bucket/a/b/c.parquet?"));
  CHECK_FALSE(contains(nested, "a%2Fb%2Fc.parquet"));

  auto leading = provider.get_presigned_url({"bucket", "/foo"}, presign_method::GET);
  CHECK(starts_with(leading, "https://s3.us-east-1.amazonaws.com/bucket//foo?"));

  auto unicode_key =
    provider.get_presigned_url({"bucket", "\xE4\xB8\xAD\xE6\x96\x87.parquet"}, presign_method::GET);
  CHECK(starts_with(unicode_key,
                    "https://s3.us-east-1.amazonaws.com/bucket/%E4%B8%AD%E6%96%87.parquet?"));
}

TEST_CASE("sirius_sigv4_credential_provider propagates session tokens", "[s3][credential_provider]")
{
  auto creds          = example_static_credentials();
  creds.session_token = "temporary/session+token=";
  sirius_sigv4_credential_provider provider(
    creds, "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto url = provider.get_presigned_url({"examplebucket", "test.txt"}, presign_method::GET);

  CHECK(contains(url, "X-Amz-Security-Token=temporary%2Fsession%2Btoken%3D"));
}

TEST_CASE("sirius_sigv4_credential_provider uses default and explicit ttl",
          "[s3][credential_provider]")
{
  sirius_sigv4_credential_provider default_provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");
  sirius_sigv4_credential_provider explicit_provider(example_static_credentials(),
                                                     "us-east-1",
                                                     "https://s3.us-east-1.amazonaws.com",
                                                     std::chrono::minutes{30});

  auto default_url =
    default_provider.get_presigned_url({"examplebucket", "test.txt"}, presign_method::GET);
  auto explicit_url =
    explicit_provider.get_presigned_url({"examplebucket", "test.txt"}, presign_method::GET);

  CHECK(query_value(default_url, "X-Amz-Expires") == "300");
  CHECK(query_value(explicit_url, "X-Amz-Expires") == "1800");
}

TEST_CASE("sirius_sigv4_credential_provider rejects empty object references",
          "[s3][credential_provider]")
{
  sirius_sigv4_credential_provider provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  CHECK_THROWS_AS(provider.get_presigned_url({"", "test.txt"}, presign_method::GET),
                  credential_error);
  CHECK_THROWS_AS(provider.get_presigned_url({"bucket", ""}, presign_method::GET),
                  credential_error);
}

TEST_CASE("sirius_sigv4_credential_provider is safe under concurrent presigning",
          "[s3][credential_provider]")
{
  sirius_sigv4_credential_provider provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  constexpr int n_threads = 8;
  constexpr int n_iters   = 25;
  std::atomic<int> malformed{0};
  std::vector<std::thread> threads;
  threads.reserve(n_threads);

  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&provider, &malformed, t] {
      for (int i = 0; i < n_iters; ++i) {
        auto url = provider.get_presigned_url({"bucket", "key-" + std::to_string(t) + ".parquet"},
                                              presign_method::GET);
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

TEST_CASE("mock_credential_provider returns canned URLs and records calls",
          "[s3][credential_provider]")
{
  mock_credential_provider provider("https://signed.example/object");

  CHECK(provider.get_presigned_url({"bucket", "key"}, presign_method::GET) ==
        "https://signed.example/object");
  CHECK(provider.get_presigned_url({"bucket", "head-key"}, presign_method::HEAD) ==
        "https://signed.example/object");

  CHECK(provider.call_count() == 2);
  CHECK(provider.get_count() == 1);
  CHECK(provider.head_count() == 1);
  CHECK(provider.last_bucket() == "bucket");
  CHECK(provider.last_key() == "head-key");
}

TEST_CASE("mock_credential_provider can force credential errors", "[s3][credential_provider]")
{
  mock_credential_provider provider("https://signed.example/object");
  provider.set_throw("boom");

  CHECK_THROWS_AS(provider.get_presigned_url({"bucket", "key"}, presign_method::GET),
                  credential_error);

  provider.clear_throw();
  CHECK(provider.get_presigned_url({"bucket", "key"}, presign_method::GET) ==
        "https://signed.example/object");
}
