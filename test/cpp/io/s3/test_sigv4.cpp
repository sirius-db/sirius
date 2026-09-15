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
#include "io/rest/s3/sigv4.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>

using sirius::io::rest::s3::presign_url;
using sirius::io::rest::s3::sha256_hex;
using sirius::io::rest::s3::sign_request;
using sirius::io::rest::s3::sigv4_signer_config;
using sirius::io::rest::s3::uri_encode;

namespace {

sigv4_signer_config aws_example_creds()
{
  sigv4_signer_config creds;
  creds.access_key = "AKIAIOSFODNN7EXAMPLE";
  creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
  creds.region     = "us-east-1";
  creds.service    = "s3";
  return creds;
}

std::string authorization_header(sirius::io::rest::s3::sigv4_signed_request const& req)
{
  for (auto const& [key, value] : req.headers) {
    if (key == "Authorization") { return value; }
  }
  return {};
}

std::string header_value(sirius::io::rest::s3::sigv4_signed_request const& req,
                         std::string_view wanted)
{
  for (auto const& [key, value] : req.headers) {
    if (key == wanted) { return value; }
  }
  return {};
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

bool is_lower_hex_64(std::string_view value)
{
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return std::isdigit(c) || (c >= 'a' && c <= 'f');
         });
}

}  // namespace

TEST_CASE("sha256_hex returns standard digests", "[s3][sigv4]")
{
  CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_CASE("uri_encode follows RFC3986 rules needed by SigV4", "[s3][sigv4]")
{
  CHECK(uri_encode("Abc-_.~", false) == "Abc-_.~");
  CHECK(uri_encode("a/b/c", false) == "a/b/c");
  CHECK(uri_encode("a/b/c", true) == "a%2Fb%2Fc");
  CHECK(uri_encode("a b", true) == "a%20b");
  CHECK(uri_encode("~!@#$", true) == "~%21%40%23%24");
}

TEST_CASE("uri_encode canonicalizes literal S3 keys exactly once", "[s3][sigv4]")
{
  CHECK(uri_encode("path with space.parquet", false) == "path%20with%20space.parquet");
  CHECK(uri_encode("100%.parquet", false) == "100%25.parquet");
  CHECK(uri_encode("a%2Fb.parquet", false) == "a%252Fb.parquet");
}

TEST_CASE("sign_request rejects incomplete signer config", "[s3][sigv4]")
{
  auto creds = aws_example_creds();

  auto bad = creds;
  bad.access_key.clear();
  CHECK_THROWS_AS(sign_request("GET",
                               "examplebucket.s3.amazonaws.com",
                               "/test.txt",
                               "",
                               sha256_hex(""),
                               {},
                               bad,
                               1369353600),
                  std::invalid_argument);

  bad = creds;
  bad.secret_key.clear();
  CHECK_THROWS_AS(sign_request("GET",
                               "examplebucket.s3.amazonaws.com",
                               "/test.txt",
                               "",
                               sha256_hex(""),
                               {},
                               bad,
                               1369353600),
                  std::invalid_argument);

  bad = creds;
  bad.region.clear();
  CHECK_THROWS_AS(sign_request("GET",
                               "examplebucket.s3.amazonaws.com",
                               "/test.txt",
                               "",
                               sha256_hex(""),
                               {},
                               bad,
                               1369353600),
                  std::invalid_argument);

  bad = creds;
  bad.service.clear();
  CHECK_THROWS_AS(sign_request("GET",
                               "examplebucket.s3.amazonaws.com",
                               "/test.txt",
                               "",
                               sha256_hex(""),
                               {},
                               bad,
                               1369353600),
                  std::invalid_argument);
}

TEST_CASE("sign_request signs empty GET payload with sha256 empty digest", "[s3][sigv4]")
{
  auto out = sign_request("GET",
                          "examplebucket.s3.amazonaws.com",
                          "/test.txt",
                          "",
                          sha256_hex(""),
                          {},
                          aws_example_creds(),
                          1369353600);

  CHECK(header_value(out, "x-amz-content-sha256") ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(contains(authorization_header(out), "SignedHeaders=host;x-amz-content-sha256;x-amz-date"));
}

TEST_CASE("sign_request injects temporary credential session token header", "[s3][sigv4]")
{
  auto creds          = aws_example_creds();
  creds.session_token = "temporary/session+token=";

  auto out = sign_request("GET",
                          "examplebucket.s3.amazonaws.com",
                          "/test.txt",
                          "",
                          sha256_hex(""),
                          {},
                          creds,
                          1369353600);

  CHECK(header_value(out, "x-amz-security-token") == "temporary/session+token=");
  CHECK(contains(authorization_header(out),
                 "SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token"));
}

TEST_CASE("presign_url matches AWS S3 published query-auth vector", "[s3][sigv4]")
{
  auto url = presign_url("GET",
                         "https",
                         "examplebucket.s3.amazonaws.com",
                         "/test.txt",
                         aws_example_creds(),
                         1369353600,
                         std::chrono::seconds{86400});

  CHECK(url ==
        "https://examplebucket.s3.amazonaws.com/test.txt?"
        "X-Amz-Algorithm=AWS4-HMAC-SHA256&"
        "X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20130524%2Fus-east-1%2Fs3%2Faws4_request&"
        "X-Amz-Date=20130524T000000Z&"
        "X-Amz-Expires=86400&"
        "X-Amz-SignedHeaders=host&"
        "X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404");
}

TEST_CASE("presign_url binds the signature to the HTTP method", "[s3][sigv4]")
{
  auto get_url  = presign_url("GET",
                             "https",
                             "examplebucket.s3.amazonaws.com",
                             "/test.txt",
                             aws_example_creds(),
                             1369353600,
                             std::chrono::seconds{300});
  auto head_url = presign_url("HEAD",
                              "https",
                              "examplebucket.s3.amazonaws.com",
                              "/test.txt",
                              aws_example_creds(),
                              1369353600,
                              std::chrono::seconds{300});

  CHECK(query_value(get_url, "X-Amz-Signature") != query_value(head_url, "X-Amz-Signature"));
}

TEST_CASE("presign_url injects session token only when present", "[s3][sigv4]")
{
  auto without_token = presign_url("GET",
                                   "https",
                                   "examplebucket.s3.amazonaws.com",
                                   "/test.txt",
                                   aws_example_creds(),
                                   1369353600,
                                   std::chrono::seconds{300});
  CHECK_FALSE(contains(without_token, "X-Amz-Security-Token="));

  auto creds          = aws_example_creds();
  creds.session_token = "temporary/session+token=";
  auto with_token     = presign_url("GET",
                                "https",
                                "examplebucket.s3.amazonaws.com",
                                "/test.txt",
                                creds,
                                1369353600,
                                std::chrono::seconds{300});

  CHECK(contains(with_token, "X-Amz-Security-Token=temporary%2Fsession%2Btoken%3D"));
  CHECK(query_value(with_token, "X-Amz-Signature") !=
        query_value(without_token, "X-Amz-Signature"));
}

TEST_CASE("presign_url keeps canonical query ordering deterministic", "[s3][sigv4]")
{
  auto creds          = aws_example_creds();
  creds.session_token = "token";
  auto url            = presign_url("GET",
                         "https",
                         "examplebucket.s3.amazonaws.com",
                         "/test.txt",
                         creds,
                         1369353600,
                         std::chrono::seconds{300});
  auto query          = query_string(url);

  auto algorithm      = query.find("X-Amz-Algorithm=");
  auto credential     = query.find("X-Amz-Credential=");
  auto date           = query.find("X-Amz-Date=");
  auto expires        = query.find("X-Amz-Expires=");
  auto token          = query.find("X-Amz-Security-Token=");
  auto signed_headers = query.find("X-Amz-SignedHeaders=");
  auto signature      = query.find("X-Amz-Signature=");

  REQUIRE(algorithm != std::string::npos);
  REQUIRE(credential != std::string::npos);
  REQUIRE(date != std::string::npos);
  REQUIRE(expires != std::string::npos);
  REQUIRE(token != std::string::npos);
  REQUIRE(signed_headers != std::string::npos);
  REQUIRE(signature != std::string::npos);

  CHECK(algorithm < credential);
  CHECK(credential < date);
  CHECK(date < expires);
  CHECK(expires < token);
  CHECK(token < signed_headers);
  CHECK(signed_headers < signature);
}

TEST_CASE("presign_url signs only the host header", "[s3][sigv4]")
{
  auto url = presign_url("GET",
                         "https",
                         "examplebucket.s3.amazonaws.com",
                         "/test.txt",
                         aws_example_creds(),
                         1369353600,
                         std::chrono::seconds{300});

  CHECK(query_value(url, "X-Amz-SignedHeaders") == "host");
}

TEST_CASE("presign_url propagates ttl into X-Amz-Expires", "[s3][sigv4]")
{
  auto url_300   = presign_url("GET",
                             "https",
                             "examplebucket.s3.amazonaws.com",
                             "/test.txt",
                             aws_example_creds(),
                             1369353600,
                             std::chrono::seconds{300});
  auto url_86400 = presign_url("GET",
                               "https",
                               "examplebucket.s3.amazonaws.com",
                               "/test.txt",
                               aws_example_creds(),
                               1369353600,
                               std::chrono::seconds{86400});

  CHECK(query_value(url_300, "X-Amz-Expires") == "300");
  CHECK(query_value(url_86400, "X-Amz-Expires") == "86400");
}

TEST_CASE("presign_url rejects invalid required inputs", "[s3][sigv4]")
{
  auto creds = aws_example_creds();

  CHECK_THROWS_AS(presign_url("",
                              "https",
                              "examplebucket.s3.amazonaws.com",
                              "/test.txt",
                              creds,
                              1369353600,
                              std::chrono::seconds{300}),
                  std::invalid_argument);
  CHECK_THROWS_AS(presign_url("GET",
                              "",
                              "examplebucket.s3.amazonaws.com",
                              "/test.txt",
                              creds,
                              1369353600,
                              std::chrono::seconds{300}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
    presign_url("GET", "https", "", "/test.txt", creds, 1369353600, std::chrono::seconds{300}),
    std::invalid_argument);
  CHECK_THROWS_AS(presign_url("GET",
                              "https",
                              "examplebucket.s3.amazonaws.com",
                              "/test.txt",
                              creds,
                              1369353600,
                              std::chrono::seconds{0}),
                  std::invalid_argument);
  CHECK_THROWS_AS(presign_url("GET",
                              "https",
                              "examplebucket.s3.amazonaws.com",
                              "/test.txt",
                              creds,
                              1369353600,
                              std::chrono::seconds{-1}),
                  std::invalid_argument);
}

TEST_CASE("presign_url leaves already-encoded canonical URI untouched", "[s3][sigv4]")
{
  auto url = presign_url("GET",
                         "https",
                         "examplebucket.s3.amazonaws.com",
                         "/path/with%20space.parquet",
                         aws_example_creds(),
                         1369353600,
                         std::chrono::seconds{300});

  CHECK(contains(url, "https://examplebucket.s3.amazonaws.com/path/with%20space.parquet?"));
  CHECK_FALSE(contains(url, "%2520"));
}

TEST_CASE("presign_url preserves the caller-selected scheme", "[s3][sigv4]")
{
  auto url = presign_url("GET",
                         "http",
                         "minio.local:9000",
                         "/bucket/test.txt",
                         aws_example_creds(),
                         1369353600,
                         std::chrono::seconds{300});

  CHECK(url.find("http://minio.local:9000/bucket/test.txt?") == 0);
  CHECK(is_lower_hex_64(query_value(url, "X-Amz-Signature")));
}
