// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/validation.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace sirius::test::object_store_test;
using core::reason;

void expect_violation(std::optional<core::violation> const& actual,
                      std::string_view field,
                      reason why,
                      std::string_view text,
                      core::capabilities const& backend = {})
{
  REQUIRE(actual.has_value());
  CHECK(actual->field == field);
  CHECK(actual->why == why);
  auto const message = core::describe(*actual, backend);
  CHECK(message == text);
  for (auto const sentinel :
       {"CORE_ACCESS_SENTINEL", "CORE_SECRET_SENTINEL", "CORE_TOKEN_SENTINEL"}) {
    CHECK(message.find(sentinel) == std::string::npos);
  }
}

}  // namespace

TEST_CASE("Object-store core validates intrinsic field and reason pairs", "[object_store_core]")
{
  struct row {
    std::string core::settings::* member;
    std::string value;
    std::string_view field;
    reason why;
    std::string_view text;
  };
  std::vector<row> const rows{
    {&core::settings::endpoint, "", "endpoint", reason::required, "required"},
    {&core::settings::region, "", "region", reason::required, "required"},
    {&core::settings::access_key, "", "access_key", reason::required, "required"},
    {&core::settings::secret_key, "", "secret_key", reason::required, "required"},
    {&core::settings::endpoint,
     "host:9000",
     "endpoint",
     reason::endpoint_missing_scheme,
     "missing scheme (expected http:// or https://)"},
    {&core::settings::endpoint,
     "ftp://host",
     "endpoint",
     reason::endpoint_bad_scheme,
     "scheme must be http or https"},
    {&core::settings::endpoint,
     "https://ho st",
     "endpoint",
     reason::endpoint_control_bytes,
     "must not contain control bytes or spaces"},
    {&core::settings::endpoint,
     std::string("https://ho\0st", 13),
     "endpoint",
     reason::endpoint_control_bytes,
     "must not contain control bytes or spaces"},
    {&core::settings::endpoint,
     "https://host\x7f",
     "endpoint",
     reason::endpoint_control_bytes,
     "must not contain control bytes or spaces"},
    {&core::settings::endpoint,
     "https://",
     "endpoint",
     reason::endpoint_missing_host,
     "missing host"},
    {&core::settings::endpoint,
     "https://:443",
     "endpoint",
     reason::endpoint_missing_host,
     "missing host"},
    {&core::settings::endpoint,
     "https://host/path",
     "endpoint",
     reason::endpoint_has_path,
     "must be scheme://host[:port] with no path, query or fragment"},
    {&core::settings::endpoint,
     "https://host?q",
     "endpoint",
     reason::endpoint_has_path,
     "must be scheme://host[:port] with no path, query or fragment"},
    {&core::settings::endpoint,
     "https://host#f",
     "endpoint",
     reason::endpoint_has_path,
     "must be scheme://host[:port] with no path, query or fragment"},
    {&core::settings::endpoint,
     "https://CORE_SECRET_SENTINEL@host",
     "endpoint",
     reason::endpoint_has_user_info,
     "must not contain user-info"},
    {&core::settings::endpoint,
     "https://[::1",
     "endpoint",
     reason::endpoint_unterminated_ipv6,
     "unterminated IPv6 literal"},
    {&core::settings::endpoint,
     "https://[::1]tail",
     "endpoint",
     reason::endpoint_malformed,
     "malformed"},
    {&core::settings::endpoint,
     "https://host:",
     "endpoint",
     reason::endpoint_bad_port,
     "port must be a decimal number in 1..65535"},
    {&core::settings::endpoint,
     "https://host:0",
     "endpoint",
     reason::endpoint_bad_port,
     "port must be a decimal number in 1..65535"},
    {&core::settings::endpoint,
     "https://host:65536",
     "endpoint",
     reason::endpoint_bad_port,
     "port must be a decimal number in 1..65535"},
    {&core::settings::endpoint,
     "https://host:000001",
     "endpoint",
     reason::endpoint_bad_port,
     "port must be a decimal number in 1..65535"},
    {&core::settings::endpoint,
     "https://host:+1",
     "endpoint",
     reason::endpoint_bad_port,
     "port must be a decimal number in 1..65535"},
    {&core::settings::endpoint,
     "https://host:x",
     "endpoint",
     reason::endpoint_bad_port,
     "port must be a decimal number in 1..65535"},
    {&core::settings::uri_prefix,
     "http://bucket/",
     "uri_prefix",
     reason::prefix_shape,
     "must be s3://bucket/ or s3://bucket/prefix/"},
    {&core::settings::uri_prefix,
     "s3://bucket/path",
     "uri_prefix",
     reason::prefix_no_trailing_slash,
     "must end with '/'"},
    {&core::settings::uri_prefix,
     "s3://bucket/path?query/",
     "uri_prefix",
     reason::prefix_query_or_fragment,
     "must not contain a query or fragment"},
    {&core::settings::uri_prefix,
     "s3://bucket/path/../other/",
     "uri_prefix",
     reason::prefix_unmatchable_segment,
     "contains a segment no key can match byte-for-byte (empty, '.', '..' or percent-encoded)"},
  };
  for (auto const& test : rows) {
    INFO(test.field << " reason " << static_cast<unsigned>(test.why));
    auto value         = valid_settings();
    value.*test.member = test.value;
    expect_violation(core::check_settings(value, {}), test.field, test.why, test.text);
  }
}

TEST_CASE("Object-store core preserves endpoint and field precedence", "[object_store_core]")
{
  struct endpoint_row {
    char const* endpoint;
    reason why;
    char const* text;
  };
  std::array const endpoints{
    endpoint_row{
      "host @/?", reason::endpoint_missing_scheme, "missing scheme (expected http:// or https://)"},
    endpoint_row{"ftp:// host/path", reason::endpoint_bad_scheme, "scheme must be http or https"},
    endpoint_row{
      "https:// :0/@x", reason::endpoint_control_bytes, "must not contain control bytes or spaces"},
    endpoint_row{"https://:0/path@x", reason::endpoint_missing_host, "missing host"},
    endpoint_row{"https://user@[invalid/path",
                 reason::endpoint_has_path,
                 "must be scheme://host[:port] with no path, query or fragment"},
    endpoint_row{
      "https://user@[invalid", reason::endpoint_has_user_info, "must not contain user-info"},
    endpoint_row{"https://[::1:0", reason::endpoint_unterminated_ipv6, "unterminated IPv6 literal"},
    endpoint_row{"https://[::1]bad:0", reason::endpoint_malformed, "malformed"},
  };
  for (auto const& test : endpoints) {
    auto value     = valid_settings();
    value.endpoint = test.endpoint;
    value.region.clear();
    expect_violation(core::check_settings(value, {}), "endpoint", test.why, test.text);
  }

  temporary_files files;
  auto value = valid_settings();
  value.endpoint.clear();
  value.region.clear();
  value.access_key.clear();
  value.secret_key.clear();
  value.transport      = 255;
  value.signing_mode   = 255;
  value.ca_bundle_path = (files.directory / "missing.pem").string();
  value.uri_prefix     = "http://bad";
  core::capabilities backend{false, false, false, "precedence"};
  auto check = [&](std::string_view field, reason why, std::string_view text) {
    expect_violation(core::check_settings(value, backend), field, why, text, backend);
  };
  check("endpoint", reason::required, "required");
  value.endpoint = "https://host";
  check("region", reason::required, "required");
  value.region = "region";
  check("access_key", reason::required, "required");
  value.access_key = "CORE_ACCESS_SENTINEL";
  check("secret_key", reason::required, "required");
  value.secret_key = "CORE_SECRET_SENTINEL";
  check("transport", reason::transport_unknown, "must be one of Transport::kAuto, kHttp, kRdma");
  value.transport = core::transport_rdma;
  check("signing_mode",
        reason::signing_mode_unknown,
        "must be one of SigningMode::kPresigned, kHeader");
  value.signing_mode = core::signing_header;
  check("ca_bundle_path", reason::ca_bundle_not_regular_file, "is not a regular file");
  value.ca_bundle_path = files.readable.string();
  check("uri_prefix", reason::prefix_shape, "must be s3://bucket/ or s3://bucket/prefix/");
  value.uri_prefix = "s3://bucket/prefix/";
  check("transport",
        reason::transport_unavailable,
        "no S3-over-RDMA backend is registered in this build");
  backend.rdma_transport = true;
  check("signing_mode",
        reason::signing_mode_unsupported,
        "the configured precedence backend does not honor a signing mode");
  backend.signing_mode_choice = true;
  check("ca_bundle_path",
        reason::ca_bundle_unsupported,
        "the configured precedence backend does not honor a CA bundle");
  backend.custom_ca_bundle = true;
  CHECK_FALSE(core::check_settings(value, backend));
}

TEST_CASE("Object-store core checks CA type and readability before backend support",
          "[object_store_core]")
{
  temporary_files files;
  auto value = valid_settings();
  core::capabilities const backend{false, false, false, "file-check"};
  for (auto const& path : {files.directory, files.directory / "missing.pem"}) {
    value.ca_bundle_path = path.string();
    expect_violation(core::check_settings(value, backend),
                     "ca_bundle_path",
                     reason::ca_bundle_not_regular_file,
                     "is not a regular file",
                     backend);
  }
  if (::geteuid() == 0) { SKIP("The unreadable-file witness requires a non-root user"); }
  REQUIRE_FALSE(std::ifstream(files.unreadable).good());
  value.ca_bundle_path = files.unreadable.string();
  value.uri_prefix     = "malformed";
  expect_violation(core::check_settings(value, backend),
                   "ca_bundle_path",
                   reason::ca_bundle_unreadable,
                   "file is not readable",
                   backend);
  value.ca_bundle_path = files.readable.string();
  value.uri_prefix.clear();
  expect_violation(core::check_settings(value, backend),
                   "ca_bundle_path",
                   reason::ca_bundle_unsupported,
                   "the configured file-check backend does not honor a CA bundle",
                   backend);
  CHECK_FALSE(core::check_settings(value, {}));
}

TEST_CASE("Object-store core retains raw enum values and accepts valid empty fields",
          "[object_store_core]")
{
  CHECK(core::transport_auto == 0);
  CHECK(core::transport_http == 1);
  CHECK(core::transport_rdma == 2);
  CHECK(core::signing_presigned == 0);
  CHECK(core::signing_header == 1);
  auto value = valid_settings("");
  value.session_token.clear();
  value.ca_bundle_path.clear();
  CHECK_FALSE(core::check_settings(value, {}));
  for (auto const endpoint : {"HTTP://host",
                              "https://host:1",
                              "https://host:65535",
                              "http://host:00001",
                              "https://[::1]",
                              "https://[::1]:443"}) {
    value.endpoint = endpoint;
    CHECK_FALSE(core::check_settings(value, {}));
  }
  // Only endpoint has a control-byte policy; credential ranges are not C strings.
  value.region     = std::string("r\0x", 3);
  value.access_key = std::string("a\0x", 3);
  value.secret_key = std::string("s\0x", 3);
  CHECK_FALSE(core::check_settings(value, {}));
  core::capabilities const all{true, true, true, "all"};
  for (std::uint32_t transport : {0U, 1U, 2U}) {
    for (std::uint32_t signing : {0U, 1U}) {
      value.transport    = transport;
      value.signing_mode = signing;
      CHECK_FALSE(core::check_settings(value, all));
    }
  }
  for (std::uint32_t unknown : {3U, 255U, 256U, 0xFFFFFFFFU}) {
    INFO("raw enum " << unknown);
    value.transport    = unknown;
    value.signing_mode = 0;
    expect_violation(core::check_settings(value, all),
                     "transport",
                     reason::transport_unknown,
                     "must be one of Transport::kAuto, kHttp, kRdma",
                     all);
    value.transport    = 0;
    value.signing_mode = unknown;
    expect_violation(core::check_settings(value, all),
                     "signing_mode",
                     reason::signing_mode_unknown,
                     "must be one of SigningMode::kPresigned, kHeader",
                     all);
  }
  value.signing_mode = 2;
  expect_violation(core::check_settings(value, all),
                   "signing_mode",
                   reason::signing_mode_unknown,
                   "must be one of SigningMode::kPresigned, kHeader",
                   all);
}

TEST_CASE("Object-store core capability flags are independent", "[object_store_core]")
{
  temporary_files files;
  for (unsigned mask = 0; mask < 8; ++mask) {
    INFO("capabilities " << mask);
    core::capabilities const backend{
      (mask & 1U) != 0, (mask & 2U) != 0, (mask & 4U) != 0, "independent"};
    auto value      = valid_settings();
    value.transport = core::transport_rdma;
    if (backend.rdma_transport) {
      CHECK_FALSE(core::check_settings(value, backend));
    } else {
      expect_violation(core::check_settings(value, backend),
                       "transport",
                       reason::transport_unavailable,
                       "no S3-over-RDMA backend is registered in this build",
                       backend);
    }
    value.transport    = core::transport_auto;
    value.signing_mode = core::signing_header;
    if (backend.signing_mode_choice) {
      CHECK_FALSE(core::check_settings(value, backend));
    } else {
      expect_violation(core::check_settings(value, backend),
                       "signing_mode",
                       reason::signing_mode_unsupported,
                       "the configured independent backend does not honor a signing mode",
                       backend);
    }
    value.signing_mode   = core::signing_presigned;
    value.ca_bundle_path = files.readable.string();
    if (backend.custom_ca_bundle) {
      CHECK_FALSE(core::check_settings(value, backend));
    } else {
      expect_violation(core::check_settings(value, backend),
                       "ca_bundle_path",
                       reason::ca_bundle_unsupported,
                       "the configured independent backend does not honor a CA bundle",
                       backend);
    }
  }
}

TEST_CASE("Object-store core checks prefix shape in publication order", "[object_store_core]")
{
  for (auto const prefix : {"", "s3://bucket/", "s3://Bucket/Mixed/", "s3://bucket/a/b/"}) {
    CHECK_FALSE(core::check_uri_prefix(prefix));
  }
  struct row {
    char const* prefix;
    reason why;
    char const* text;
  };
  std::array const rows{
    row{"S3://bucket/p", reason::prefix_shape, "must be s3://bucket/ or s3://bucket/prefix/"},
    row{"s3://", reason::prefix_shape, "must be s3://bucket/ or s3://bucket/prefix/"},
    row{"s3:///a?x", reason::prefix_shape, "must be s3://bucket/ or s3://bucket/prefix/"},
    row{"s3://bucket", reason::prefix_no_trailing_slash, "must end with '/'"},
    row{"s3://bucket/a?x", reason::prefix_no_trailing_slash, "must end with '/'"},
    row{"s3://bucket/a//b?x/",
        reason::prefix_query_or_fragment,
        "must not contain a query or fragment"},
    row{
      "s3://bucket/a#x/", reason::prefix_query_or_fragment, "must not contain a query or fragment"},
    row{"s3://bucket/a//b/",
        reason::prefix_unmatchable_segment,
        "contains a segment no key can match byte-for-byte (empty, '.', '..' or percent-encoded)"},
    row{"s3://bucket/a/./b/",
        reason::prefix_unmatchable_segment,
        "contains a segment no key can match byte-for-byte (empty, '.', '..' or percent-encoded)"},
    row{"s3://bucket/a/../b/",
        reason::prefix_unmatchable_segment,
        "contains a segment no key can match byte-for-byte (empty, '.', '..' or percent-encoded)"},
    row{"s3://bucket/a%2Fb/",
        reason::prefix_unmatchable_segment,
        "contains a segment no key can match byte-for-byte (empty, '.', '..' or percent-encoded)"},
    row{"s3://bucket//",
        reason::prefix_unmatchable_segment,
        "contains a segment no key can match byte-for-byte (empty, '.', '..' or percent-encoded)"},
  };
  for (auto const& test : rows) {
    INFO(test.prefix);
    expect_violation(core::check_uri_prefix(test.prefix), "uri_prefix", test.why, test.text);
    auto value = valid_settings(test.prefix);
    expect_violation(core::check_settings(value, {}), "uri_prefix", test.why, test.text);
  }
}
