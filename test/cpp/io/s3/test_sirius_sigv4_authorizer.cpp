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
#include "io/s3/mock_request_authorizer.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"
#include "io/s3/static_credentials.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using sirius::io::credential_error;
using sirius::io::object_store_config;
using sirius::io::s3::mock_request_authorizer;
using sirius::io::s3::s3_authorized_request;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::s3_request_method;
using sirius::io::s3::sirius_sigv4_header_authorizer;
using sirius::io::s3::sirius_sigv4_presigned_authorizer;
using sirius::io::s3::static_credentials;
using sirius::io::s3::static_credentials_from;

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

}  // namespace

TEST_CASE("sirius_sigv4_presigned_authorizer normalizes HTTPS endpoint", "[s3][authorizer]")
{
  sirius_sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-west-2", "HTTPS://S3.US-WEST-2.AMAZONAWS.COM");

  auto request =
    provider.authorize({"examplebucket", "test.txt"}, s3_request_method::GET, k_presign_timeout);
  CHECK(request.headers.empty());
  auto const& url = request.url;

  CHECK(starts_with(url, "https://s3.us-west-2.amazonaws.com/examplebucket/test.txt?"));
  CHECK(query_value(url, "X-Amz-Credential").find("%2Fus-west-2%2Fs3%2Faws4_request") !=
        std::string::npos);
  CHECK(query_value(url, "X-Amz-SignedHeaders") == "host");
}

TEST_CASE("sirius_sigv4_presigned_authorizer preserves HTTP endpoint ports", "[s3][authorizer]")
{
  sirius_sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-east-1", "http://s3.local:9000");

  auto request =
    provider.authorize({"bucket", "object.parquet"}, s3_request_method::GET, k_presign_timeout);
  CHECK(request.headers.empty());
  auto const& url = request.url;

  CHECK(starts_with(url, "http://s3.local:9000/bucket/object.parquet?"));
  CHECK(is_lower_hex_64(query_value(url, "X-Amz-Signature")));
}

TEST_CASE("sirius_sigv4_presigned_authorizer rejects malformed construction inputs",
          "[s3][authorizer]")
{
  auto creds = example_static_credentials();

  CHECK_THROWS_AS(sirius_sigv4_presigned_authorizer(creds, "us-east-1", ""), credential_error);
  CHECK_THROWS_AS(
    sirius_sigv4_presigned_authorizer(creds, "us-east-1", "s3.us-east-1.amazonaws.com"),
    credential_error);
  CHECK_THROWS_AS(sirius_sigv4_presigned_authorizer(creds, "us-east-1", "ftp://example.com"),
                  credential_error);
  CHECK_THROWS_AS(
    sirius_sigv4_presigned_authorizer(creds, "us-east-1", "https://example.com/prefix"),
    credential_error);
  CHECK_THROWS_AS(sirius_sigv4_presigned_authorizer(creds, "us-east-1", "https://example.com?x=1"),
                  credential_error);
  CHECK_THROWS_AS(
    sirius_sigv4_presigned_authorizer(creds, "us-east-1", "https://example.com#fragment"),
    credential_error);

  auto no_access_key = creds;
  no_access_key.access_key_id.clear();
  CHECK_THROWS_AS(
    sirius_sigv4_presigned_authorizer(no_access_key, "us-east-1", "https://example.com"),
    credential_error);

  auto no_secret_key = creds;
  no_secret_key.secret_access_key.clear();
  CHECK_THROWS_AS(
    sirius_sigv4_presigned_authorizer(no_secret_key, "us-east-1", "https://example.com"),
    credential_error);

  CHECK_THROWS_AS(sirius_sigv4_presigned_authorizer(creds, "", "https://example.com"),
                  credential_error);
  CHECK_THROWS_AS(sirius_sigv4_presigned_authorizer(
                    creds, "us-east-1", "https://example.com", std::chrono::seconds{0}),
                  credential_error);
}

TEST_CASE("sirius_sigv4_header_authorizer signs with headers and plain path-style URLs",
          "[s3][authorizer]")
{
  auto creds          = example_static_credentials();
  creds.session_token = "temporary/session+token=";
  sirius_sigv4_header_authorizer provider(creds, "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto get_request =
    provider.authorize({"examplebucket", "test.txt"}, s3_request_method::GET, k_presign_timeout);
  auto head_request =
    provider.authorize({"examplebucket", "test.txt"}, s3_request_method::HEAD, k_presign_timeout);

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

TEST_CASE("sirius_sigv4_header_authorizer omits session-token header for long-lived keys",
          "[s3][authorizer]")
{
  sirius_sigv4_header_authorizer provider(
    example_static_credentials(), "us-east-1", "http://s3.local:9000");

  auto request = provider.authorize(
    {"bucket", "nested/object.parquet"}, s3_request_method::GET, std::chrono::seconds{10});

  CHECK(request.url == "http://s3.local:9000/bucket/nested/object.parquet");
  CHECK_FALSE(contains(request.url, "X-Amz-"));
  CHECK(starts_with(header_value(request.headers, "Authorization"), "AWS4-HMAC-SHA256 "));
  CHECK(header_value(request.headers, "x-amz-security-token").empty());
}

TEST_CASE("sirius_sigv4_presigned_authorizer generates distinct GET and HEAD URLs",
          "[s3][authorizer]")
{
  sirius_sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto get_request =
    provider.authorize({"examplebucket", "test.txt"}, s3_request_method::GET, k_presign_timeout);
  auto head_request =
    provider.authorize({"examplebucket", "test.txt"}, s3_request_method::HEAD, k_presign_timeout);
  CHECK(get_request.headers.empty());
  CHECK(head_request.headers.empty());
  auto const& get_url  = get_request.url;
  auto const& head_url = head_request.url;

  CHECK(query_value(get_url, "X-Amz-SignedHeaders") == "host");
  CHECK(query_value(head_url, "X-Amz-SignedHeaders") == "host");
  CHECK(query_value(get_url, "X-Amz-Signature") != query_value(head_url, "X-Amz-Signature"));
}

TEST_CASE("sirius_sigv4_presigned_authorizer encodes bucket and key path components",
          "[s3][authorizer]")
{
  sirius_sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto spaced =
    provider
      .authorize({"bucket", "path with space.parquet"}, s3_request_method::GET, k_presign_timeout)
      .url;
  CHECK(
    starts_with(spaced, "https://s3.us-east-1.amazonaws.com/bucket/path%20with%20space.parquet?"));

  auto nested =
    provider.authorize({"bucket", "a/b/c.parquet"}, s3_request_method::GET, k_presign_timeout).url;
  CHECK(starts_with(nested, "https://s3.us-east-1.amazonaws.com/bucket/a/b/c.parquet?"));
  CHECK_FALSE(contains(nested, "a%2Fb%2Fc.parquet"));

  auto leading =
    provider.authorize({"bucket", "/foo"}, s3_request_method::GET, k_presign_timeout).url;
  CHECK(starts_with(leading, "https://s3.us-east-1.amazonaws.com/bucket//foo?"));

  auto unicode_key =
    provider
      .authorize(
        {"bucket", "\xE4\xB8\xAD\xE6\x96\x87.parquet"}, s3_request_method::GET, k_presign_timeout)
      .url;
  CHECK(starts_with(unicode_key,
                    "https://s3.us-east-1.amazonaws.com/bucket/%E4%B8%AD%E6%96%87.parquet?"));
}

TEST_CASE("sirius_sigv4_presigned_authorizer propagates session tokens", "[s3][authorizer]")
{
  auto creds          = example_static_credentials();
  creds.session_token = "temporary/session+token=";
  sirius_sigv4_presigned_authorizer provider(
    creds, "us-east-1", "https://s3.us-east-1.amazonaws.com");

  auto request =
    provider.authorize({"examplebucket", "test.txt"}, s3_request_method::GET, k_presign_timeout);
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

  sirius_sigv4_presigned_authorizer token_provider(creds, cfg.region, cfg.endpoint);
  auto token_url =
    token_provider
      .authorize({"examplebucket", "test.txt"}, s3_request_method::GET, k_presign_timeout)
      .url;
  CHECK(contains(token_url, "X-Amz-Security-Token=temporary%2Fsession%2Btoken%3D"));

  cfg.session_token.clear();
  auto no_token_creds = static_credentials_from(cfg);
  CHECK(no_token_creds.session_token.empty());

  sirius_sigv4_presigned_authorizer no_token_provider(no_token_creds, cfg.region, cfg.endpoint);
  auto no_token_url =
    no_token_provider
      .authorize({"examplebucket", "test.txt"}, s3_request_method::GET, k_presign_timeout)
      .url;
  CHECK_FALSE(contains(no_token_url, "X-Amz-Security-Token="));
}

TEST_CASE("sirius_sigv4_presigned_authorizer honors per-call timeout", "[s3][authorizer]")
{
  auto creds          = example_static_credentials();
  creds.session_token = "temporary/session+token=";
  sirius_sigv4_presigned_authorizer provider(
    creds, "us-east-1", "https://s3.us-east-1.amazonaws.com", std::chrono::minutes{30});

  auto short_request = provider.authorize(
    {"examplebucket", "test.txt"}, s3_request_method::GET, std::chrono::seconds{37});
  auto long_request = provider.authorize(
    {"examplebucket", "test.txt"}, s3_request_method::GET, std::chrono::seconds{1800});
  auto head_request = provider.authorize(
    {"examplebucket", "test.txt"}, s3_request_method::HEAD, std::chrono::seconds{37});
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

TEST_CASE("sirius_sigv4_presigned_authorizer rejects empty object references", "[s3][authorizer]")
{
  sirius_sigv4_presigned_authorizer provider(
    example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  CHECK_THROWS_AS(provider.authorize({"", "test.txt"}, s3_request_method::GET, k_presign_timeout),
                  credential_error);
  CHECK_THROWS_AS(provider.authorize({"bucket", ""}, s3_request_method::GET, k_presign_timeout),
                  credential_error);
}

TEST_CASE("sirius_sigv4_presigned_authorizer is safe under concurrent presigning",
          "[s3][authorizer]")
{
  sirius_sigv4_presigned_authorizer provider(
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
                                s3_request_method::GET,
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

TEST_CASE("mock_request_authorizer returns canned URLs and records calls", "[s3][authorizer]")
{
  mock_request_authorizer provider(
    s3_authorized_request{"https://signed.example/object", {{"x-test-header", "one"}}});

  auto get_request =
    provider.authorize({"bucket", "key"}, s3_request_method::GET, k_presign_timeout);
  CHECK(get_request.url == "https://signed.example/object");
  CHECK(get_request.headers ==
        std::vector<std::pair<std::string, std::string>>{{"x-test-header", "one"}});
  auto head_request =
    provider.authorize({"bucket", "head-key"}, s3_request_method::HEAD, k_presign_timeout);
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

TEST_CASE("mock_request_authorizer can force credential errors", "[s3][authorizer]")
{
  mock_request_authorizer provider(s3_authorized_request{"https://signed.example/object", {}});
  provider.set_throw("boom");

  CHECK_THROWS_AS(provider.authorize({"bucket", "key"}, s3_request_method::GET, k_presign_timeout),
                  credential_error);

  provider.clear_throw();
  auto request = provider.authorize({"bucket", "key"}, s3_request_method::GET, k_presign_timeout);
  CHECK(request.url == "https://signed.example/object");
  CHECK(request.headers.empty());
}
