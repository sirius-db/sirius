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

#include "io/s3/sirius_sigv4_authorizer.hpp"

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
    throw credential_error("sirius_sigv4_presigned_authorizer: empty endpoint");
  }
  auto sep = endpoint.find("://");
  if (sep == std::string_view::npos) {
    throw credential_error(
      "sirius_sigv4_presigned_authorizer: endpoint missing scheme (expected http:// or https://)");
  }
  std::string scheme = to_lower(endpoint.substr(0, sep));
  if (scheme != "http" && scheme != "https") {
    throw credential_error(
      "sirius_sigv4_presigned_authorizer: endpoint scheme must be http or https (got '" + scheme +
      "')");
  }
  std::string remainder{endpoint.substr(sep + 3)};
  // Anything past host[:port] (path, query, fragment) is rejected — endpoint
  // is service-level, not URL.
  auto bad = remainder.find_first_of("/?#");
  if (bad != std::string::npos) {
    throw credential_error(
      "sirius_sigv4_presigned_authorizer: endpoint must be scheme://host[:port] only (got "
      "trailing '" +
      remainder.substr(bad) + "')");
  }
  if (remainder.empty()) {
    throw credential_error("sirius_sigv4_presigned_authorizer: endpoint missing host");
  }
  return {std::move(scheme), to_lower(remainder)};
}

/// Path-style canonical URI "/<bucket>/<key>", RFC3986-encoded. Bucket encodes
/// '/' (none valid in bucket names); key leaves '/' so nested keys pass through.
std::string make_canonical_uri(s3_object_ref const& obj)
{
  if (obj.bucket.empty()) { throw credential_error("sirius_sigv4_authorizer: empty bucket"); }
  if (obj.key.empty()) { throw credential_error("sirius_sigv4_authorizer: empty key"); }
  std::string uri = "/";
  uri += uri_encode(obj.bucket, /*encode_slash=*/true);
  uri += '/';
  uri += uri_encode(obj.key, /*encode_slash=*/false);
  return uri;
}

/// static_credentials -> sigv4_signer_config. Region/service are signing-scope;
/// session_token (when present) participates in the signature.
sigv4_signer_config make_signer(static_credentials const& creds, std::string const& region)
{
  sigv4_signer_config signer{};
  signer.access_key    = creds.access_key_id;
  signer.secret_key    = creds.secret_access_key;
  signer.region        = region;
  signer.service       = "s3";
  signer.session_token = creds.session_token;
  return signer;
}

std::string_view method_to_str(s3_request_method method)
{
  switch (method) {
    case s3_request_method::GET: return "GET";
    case s3_request_method::HEAD: return "HEAD";
  }
  return "GET";  // unreachable; all enumerators handled
}

}  // namespace

sirius_sigv4_authorizer_base::sirius_sigv4_authorizer_base(static_credentials creds,
                                                           std::string region,
                                                           std::string endpoint)
  : _creds(std::move(creds)), _region(std::move(region))
{
  if (_creds.access_key_id.empty() || _creds.secret_access_key.empty()) {
    throw credential_error(
      "sirius_sigv4_authorizer: access_key_id and secret_access_key must be non-empty");
  }
  if (_region.empty()) {
    throw credential_error("sirius_sigv4_authorizer: region must be non-empty");
  }

  auto [scheme, host] = parse_endpoint(endpoint);
  _scheme             = std::move(scheme);
  _host               = std::move(host);
}

sirius_sigv4_presigned_authorizer::sirius_sigv4_presigned_authorizer(
  static_credentials creds,
  std::string region,
  std::string endpoint,
  std::chrono::seconds default_ttl)
  : sirius_sigv4_authorizer_base(std::move(creds), std::move(region), std::move(endpoint)),
    _ttl(default_ttl)
{
  if (_ttl.count() <= 0) {
    throw credential_error("sirius_sigv4_presigned_authorizer: default_ttl must be positive");
  }
}

s3_authorized_request sirius_sigv4_presigned_authorizer::authorize(s3_object_ref const& obj,
                                                                   s3_request_method method,
                                                                   std::chrono::seconds timeout)
{
  auto const canonical_uri = make_canonical_uri(obj);
  auto const signer        = make_signer(_creds, _region);

  // Per-call timeout drives X-Amz-Expires; fall back to the construction-time
  // default TTL when the caller passes a non-positive value.
  auto const effective_ttl = timeout.count() > 0 ? timeout : _ttl;

  try {
    // Auth lives entirely in the URL query, so the URL is returned with EMPTY
    // headers.
    return s3_authorized_request{presign_url(method_to_str(method),
                                             _scheme,
                                             _host,
                                             canonical_uri,
                                             signer,
                                             std::time(nullptr),
                                             effective_ttl),
                                 {}};
  } catch (credential_error const&) {
    throw;
  } catch (std::exception const& e) {
    throw credential_error(std::string("sirius_sigv4_presigned_authorizer: ") + e.what());
  }
}

sirius_sigv4_header_authorizer::sirius_sigv4_header_authorizer(static_credentials creds,
                                                               std::string region,
                                                               std::string endpoint)
  : sirius_sigv4_authorizer_base(std::move(creds), std::move(region), std::move(endpoint))
{
}

s3_authorized_request sirius_sigv4_header_authorizer::authorize(s3_object_ref const& obj,
                                                                s3_request_method method,
                                                                std::chrono::seconds /*timeout*/)
{
  auto const canonical_uri = make_canonical_uri(obj);
  auto const signer        = make_signer(_creds, _region);

  try {
    // Auth in the Authorization header. GET/HEAD have no body, and any Range is
    // added unsigned by the caller, so the canonical query and extra signed
    // headers are empty and the payload hash is sha256("").
    auto signed_req = sign_request(method_to_str(method),
                                   _host,
                                   canonical_uri,
                                   /*canonical_query=*/"",
                                   sha256_hex(""),
                                   /*extra_headers=*/{},
                                   signer,
                                   std::time(nullptr));
    std::string url = _scheme + "://" + _host + canonical_uri;
    return s3_authorized_request{std::move(url), std::move(signed_req.headers)};
  } catch (credential_error const&) {
    throw;
  } catch (std::exception const& e) {
    throw credential_error(std::string("sirius_sigv4_header_authorizer: ") + e.what());
  }
}

}  // namespace sirius::io::s3
