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

#include "io/s3/sirius_sigv4_credential_provider.hpp"

#include "io/io_errors.hpp"
#include "io/s3/sigv4.hpp"

#include <cctype>
#include <ctime>
#include <stdexcept>
#include <utility>

namespace sirius::io::s3 {

namespace {

/// Lowercase ASCII. RFC 3986 says scheme is case-insensitive and DNS host is
/// case-insensitive; the SigV4 canonical form requires the lowercased version.
std::string to_lower(std::string_view s)
{
  std::string out(s);
  for (auto& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

/// Split @p endpoint into (scheme, host_with_optional_port). Trailing path /
/// query / fragment is rejected — the endpoint identifies the S3 service, not
/// a specific bucket or object.
std::pair<std::string, std::string> parse_endpoint(std::string_view endpoint)
{
  if (endpoint.empty()) {
    throw credential_error("sirius_sigv4_credential_provider: empty endpoint");
  }
  auto sep = endpoint.find("://");
  if (sep == std::string_view::npos) {
    throw credential_error(
      "sirius_sigv4_credential_provider: endpoint missing scheme (expected http:// or https://)");
  }
  std::string scheme = to_lower(endpoint.substr(0, sep));
  if (scheme != "http" && scheme != "https") {
    throw credential_error(
      "sirius_sigv4_credential_provider: endpoint scheme must be http or https (got '" + scheme +
      "')");
  }
  std::string remainder{endpoint.substr(sep + 3)};
  // Anything past host[:port] (path, query, fragment) is rejected — endpoint
  // is service-level, not URL.
  auto bad = remainder.find_first_of("/?#");
  if (bad != std::string::npos) {
    throw credential_error(
      "sirius_sigv4_credential_provider: endpoint must be scheme://host[:port] only (got "
      "trailing '" +
      remainder.substr(bad) + "')");
  }
  if (remainder.empty()) {
    throw credential_error("sirius_sigv4_credential_provider: endpoint missing host");
  }
  return {std::move(scheme), to_lower(remainder)};
}

}  // namespace

sirius_sigv4_credential_provider::sirius_sigv4_credential_provider(static_credentials creds,
                                                                   std::string region,
                                                                   std::string endpoint,
                                                                   std::chrono::seconds default_ttl)
  : _creds(std::move(creds)), _region(std::move(region)), _ttl(default_ttl)
{
  if (_creds.access_key_id.empty() || _creds.secret_access_key.empty()) {
    throw credential_error(
      "sirius_sigv4_credential_provider: access_key_id and secret_access_key must be non-empty");
  }
  if (_region.empty()) {
    throw credential_error("sirius_sigv4_credential_provider: region must be non-empty");
  }
  if (_ttl.count() <= 0) {
    throw credential_error("sirius_sigv4_credential_provider: default_ttl must be positive");
  }

  auto [scheme, host] = parse_endpoint(endpoint);
  _scheme             = std::move(scheme);
  _host               = std::move(host);
}

std::string sirius_sigv4_credential_provider::get_presigned_url(s3_object_ref const& obj,
                                                                presign_method method)
{
  if (obj.bucket.empty()) {
    throw credential_error("sirius_sigv4_credential_provider: empty bucket");
  }
  if (obj.key.empty()) { throw credential_error("sirius_sigv4_credential_provider: empty key"); }

  // Path-style URL: /<bucket>/<key>. Bucket is encoded with encode_slash=true
  // (no slash valid in bucket names anyway); key is encoded with
  // encode_slash=false so '/' separators in nested keys pass through.
  std::string canonical_uri = "/";
  canonical_uri += uri_encode(obj.bucket, /*encode_slash=*/true);
  canonical_uri += '/';
  canonical_uri += uri_encode(obj.key, /*encode_slash=*/false);

  // Translate static_credentials → sigv4_signer_config. Region/service are
  // signing-scope, separate from the credentials snapshot; session_token
  // (when present) participates in the signed query.
  sigv4_signer_config signer{};
  signer.access_key    = _creds.access_key_id;
  signer.secret_key    = _creds.secret_access_key;
  signer.region        = _region;
  signer.service       = "s3";
  signer.session_token = _creds.session_token;

  std::string_view method_str;
  switch (method) {
    case presign_method::GET: method_str = "GET"; break;
    case presign_method::HEAD: method_str = "HEAD"; break;
  }

  try {
    return presign_url(method_str, _scheme, _host, canonical_uri, signer, std::time(nullptr), _ttl);
  } catch (credential_error const&) {
    throw;
  } catch (std::exception const& e) {
    throw credential_error(std::string("sirius_sigv4_credential_provider: ") + e.what());
  }
}

}  // namespace sirius::io::s3
