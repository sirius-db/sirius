// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/validation.hpp"

#include "io/rest/s3/endpoint.hpp"  // sirius::io::rest::s3::parse_endpoint
#include "io/s3/scope_policy.hpp"

#include <cucascade/io/io_errors.hpp>  // cucascade::io::credential_error

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace sirius::object_store_scope {

namespace {

std::optional<violation> refuse(std::string_view field, reason why)
{
  return violation{field, why};
}

// Shape of an S3 service endpoint: value-free checks first, then the SigV4 authorizer's own
// parser, so nothing admitted here is later rejected by the parser (one-way agreement).
std::optional<violation> check_endpoint(const std::string& endpoint)
{
  constexpr std::string_view field = "endpoint";
  if (endpoint.empty()) { return refuse(field, reason::required); }
  auto const sep = endpoint.find("://");
  if (sep == std::string::npos) { return refuse(field, reason::endpoint_missing_scheme); }
  std::string scheme = endpoint.substr(0, sep);
  for (auto& c : scheme) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (scheme != "http" && scheme != "https") { return refuse(field, reason::endpoint_bad_scheme); }
  for (unsigned char const c : endpoint) {
    if (c <= 0x20 || c == 0x7F) { return refuse(field, reason::endpoint_control_bytes); }
  }
  auto const remainder = std::string_view{endpoint}.substr(sep + 3);
  // A port-only authority ("https://:443") has no host either.
  if (remainder.empty() || remainder.front() == ':') {
    return refuse(field, reason::endpoint_missing_host);
  }
  if (remainder.find_first_of("/?#") != std::string_view::npos) {
    return refuse(field, reason::endpoint_has_path);
  }
  if (remainder.find('@') != std::string_view::npos) {
    return refuse(field, reason::endpoint_has_user_info);
  }
  // host[:port]: an optional explicit port is decimal 1..65535. An IPv6 literal keeps its colons
  // inside the brackets; the port, if any, follows "]:".
  auto port_sep = std::string_view::npos;
  if (remainder.front() == '[') {
    auto const close = remainder.find(']');
    if (close == std::string_view::npos) {
      return refuse(field, reason::endpoint_unterminated_ipv6);
    }
    if (close + 1 < remainder.size()) {
      if (remainder[close + 1] != ':') { return refuse(field, reason::endpoint_malformed); }
      port_sep = close + 1;
    }
  } else {
    port_sep = remainder.find(':');
  }
  if (port_sep != std::string_view::npos) {
    auto const port = remainder.substr(port_sep + 1);
    bool const digits =
      !port.empty() && port.size() <= 5 &&
      std::all_of(port.begin(), port.end(), [](char c) { return c >= '0' && c <= '9'; });
    if (!digits || std::stoi(std::string{port}) < 1 || std::stoi(std::string{port}) > 65535) {
      return refuse(field, reason::endpoint_bad_port);
    }
  }
  try {
    (void)sirius::io::rest::s3::parse_endpoint(endpoint);
  } catch (cucascade::io::credential_error const&) {
    return refuse(field, reason::endpoint_malformed);
  }
  return std::nullopt;
}

// The checks every entry point starts with: endpoint, required fields, selector membership.
std::optional<violation> check_leading_fields(const settings& value)
{
  if (auto refused = check_endpoint(value.endpoint)) { return refused; }
  if (value.region.empty()) { return refuse("region", reason::required); }
  if (value.access_key.empty()) { return refuse("access_key", reason::required); }
  if (value.secret_key.empty()) { return refuse("secret_key", reason::required); }
  // A value outside the declared set (a C++ cast or a raw bridge representation) is refused
  // here, before backend support is consulted.
  if (!is_declared_transport(value.transport)) {
    return refuse("transport", reason::transport_unknown);
  }
  if (!is_declared_signing_mode(value.signing_mode)) {
    return refuse("signing_mode", reason::signing_mode_unknown);
  }
  return std::nullopt;
}

}  // namespace

bool is_declared_transport(std::uint32_t value) noexcept
{
  return value == transport_auto || value == transport_http || value == transport_rdma;
}

bool is_declared_signing_mode(std::uint32_t value) noexcept
{
  return value == signing_presigned || value == signing_header;
}

// "s3://bucket/" or "s3://bucket/prefix/": scheme s3, a bucket, a trailing slash, no query. The
// same rule admits a scope key to publication (put/set) and to removal (drop). A prefix no key
// can match byte-for-byte (an empty, "." or ".." segment, or a raw `%` byte) is refused
// with the same rule that refuses such keys at resolution, so a published scope is never dead.
std::optional<violation> check_uri_prefix(std::string_view uri_prefix)
{
  constexpr std::string_view field   = "uri_prefix";
  constexpr std::string_view kScheme = "s3://";
  if (uri_prefix.empty()) { return std::nullopt; }
  if (uri_prefix.substr(0, kScheme.size()) != kScheme || uri_prefix.size() == kScheme.size() ||
      uri_prefix[kScheme.size()] == '/') {
    return refuse(field, reason::prefix_shape);
  }
  if (uri_prefix.back() != '/') { return refuse(field, reason::prefix_no_trailing_slash); }
  if (uri_prefix.find_first_of("?#") != std::string_view::npos) {
    return refuse(field, reason::prefix_query_or_fragment);
  }
  // Re-check the key part as a key: drop the mandatory trailing slash and let the shared rule
  // classify the remaining segments ("s3://bucket/" has no key and is always fine).
  auto const bucket_end = uri_prefix.find('/', kScheme.size());
  if (bucket_end != std::string_view::npos && bucket_end + 1 < uri_prefix.size()) {
    std::string const as_key{uri_prefix.substr(0, uri_prefix.size() - 1)};
    if (sirius::io::s3::key_defeats_prefix_match(as_key)) {
      return refuse(field, reason::prefix_unmatchable_segment);
    }
  }
  return std::nullopt;
}

std::optional<violation> check_settings(const settings& value, const capabilities& backend)
{
  if (auto refused = check_leading_fields(value)) { return refused; }
  if (!value.ca_bundle_path.empty()) {
    // ifstream opens a directory on Linux; require a regular file before the readability probe.
    std::error_code ec;
    if (!std::filesystem::is_regular_file(value.ca_bundle_path, ec)) {
      return refuse("ca_bundle_path", reason::ca_bundle_not_regular_file);
    }
    std::ifstream bundle(value.ca_bundle_path);
    if (!bundle.good()) { return refuse("ca_bundle_path", reason::ca_bundle_unreadable); }
  }
  if (auto refused = check_uri_prefix(value.uri_prefix)) { return refused; }
  if (value.transport == transport_rdma && !backend.rdma_transport) {
    return refuse("transport", reason::transport_unavailable);
  }
  if (value.signing_mode != signing_presigned && !backend.signing_mode_choice) {
    return refuse("signing_mode", reason::signing_mode_unsupported);
  }
  if (!value.ca_bundle_path.empty() && !backend.custom_ca_bundle) {
    return refuse("ca_bundle_path", reason::ca_bundle_unsupported);
  }
  return std::nullopt;
}

std::optional<violation> check_settings_intrinsic(const settings& value)
{
  if (auto refused = check_leading_fields(value)) { return refused; }
  return check_uri_prefix(value.uri_prefix);
}

std::string describe(const violation& refused, const capabilities& backend)
{
  switch (refused.why) {
    case reason::required: return "required";
    case reason::endpoint_missing_scheme: return "missing scheme (expected http:// or https://)";
    case reason::endpoint_bad_scheme: return "scheme must be http or https";
    case reason::endpoint_control_bytes: return "must not contain control bytes or spaces";
    case reason::endpoint_missing_host: return "missing host";
    case reason::endpoint_has_path:
      return "must be scheme://host[:port] with no path, query or fragment";
    case reason::endpoint_has_user_info: return "must not contain user-info";
    case reason::endpoint_unterminated_ipv6: return "unterminated IPv6 literal";
    case reason::endpoint_malformed: return "malformed";
    case reason::endpoint_bad_port: return "port must be a decimal number in 1..65535";
    case reason::transport_unknown: return "must be one of Transport::kAuto, kHttp, kRdma";
    case reason::signing_mode_unknown: return "must be one of SigningMode::kPresigned, kHeader";
    case reason::ca_bundle_not_regular_file: return "is not a regular file";
    case reason::ca_bundle_unreadable: return "file is not readable";
    case reason::prefix_shape: return "must be s3://bucket/ or s3://bucket/prefix/";
    case reason::prefix_no_trailing_slash: return "must end with '/'";
    case reason::prefix_query_or_fragment: return "must not contain a query or fragment";
    case reason::prefix_unmatchable_segment:
      return "contains a segment no key can match byte-for-byte (empty, '.', '..' or "
             "percent-encoded)";
    case reason::transport_unavailable:
      return "no S3-over-RDMA backend is registered in this build";
    case reason::signing_mode_unsupported:
      return "the configured " + std::string{backend.backend_name} +
             " backend does not honor a signing mode";
    case reason::ca_bundle_unsupported:
      return "the configured " + std::string{backend.backend_name} +
             " backend does not honor a CA bundle";
  }
  return {};
}

}  // namespace sirius::object_store_scope
