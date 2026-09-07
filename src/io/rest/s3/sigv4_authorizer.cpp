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

#include "io/rest/s3/sigv4_authorizer.hpp"

#include "io/io_errors.hpp"
#include "io/rest/s3/sigv4.hpp"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <stdexcept>
#include <utility>

namespace sirius::io::rest::s3 {

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
  if (endpoint.empty()) { throw credential_error("sigv4_presigned_authorizer: empty endpoint"); }
  auto sep = endpoint.find("://");
  if (sep == std::string_view::npos) {
    throw credential_error(
      "sigv4_presigned_authorizer: endpoint missing scheme (expected http:// or https://)");
  }
  std::string scheme = to_lower(endpoint.substr(0, sep));
  if (scheme != "http" && scheme != "https") {
    throw credential_error(
      "sigv4_presigned_authorizer: endpoint scheme must be http or https (got '" + scheme + "')");
  }
  std::string remainder{endpoint.substr(sep + 3)};
  // Anything past host[:port] (path, query, fragment) is rejected — endpoint
  // is service-level, not URL.
  auto bad = remainder.find_first_of("/?#");
  if (bad != std::string::npos) {
    throw credential_error(
      "sigv4_presigned_authorizer: endpoint must be scheme://host[:port] only (got "
      "trailing '" +
      remainder.substr(bad) + "')");
  }
  if (remainder.empty()) {
    throw credential_error("sigv4_presigned_authorizer: endpoint missing host");
  }
  return {std::move(scheme), to_lower(remainder)};
}

/// Path-style canonical URI "/<bucket>/<key>", RFC3986-encoded. Bucket encodes
/// '/' (none valid in bucket names); key leaves '/' so nested keys pass through.
std::string make_canonical_uri(object_ref const& obj)
{
  if (obj.bucket.empty()) { throw credential_error("sigv4_authorizer: empty bucket"); }
  if (obj.key.empty()) { throw credential_error("sigv4_authorizer: empty key"); }
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

std::string_view method_to_str(request_method method)
{
  switch (method) {
    case request_method::GET: return "GET";
    case request_method::HEAD: return "HEAD";
  }
  return "GET";  // unreachable; all enumerators handled
}

// Reject X-Amz-* parameters case-insensitively to prevent signing-parameter
// injection: a caller-supplied X-Amz-* key would otherwise be merged into the
// signed set (presigned mode) or signed verbatim (header mode), duplicating or
// overriding the authorizer's own signing parameters.
void reject_amz_query_params(std::string_view canonical_query)
{
  constexpr std::string_view k_amz = "x-amz-";
  for (std::size_t b = 0; b < canonical_query.size();) {
    auto const amp     = canonical_query.find('&', b);
    auto const len     = (amp == std::string::npos ? canonical_query.size() : amp) - b;
    auto const key_end = std::min(std::string_view{canonical_query}.substr(b, len).find('='), len);
    auto const key     = std::string_view{canonical_query}.substr(b, key_end);
    if (key.size() >= k_amz.size()) {
      bool amz = true;
      for (std::size_t i = 0; i < k_amz.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(key[i])) != k_amz[i]) {
          amz = false;
          break;
        }
      }
      if (amz) {
        throw credential_error(
          "sigv4_authorizer: X-Amz-* keys are not allowed in a LIST canonical query (got '" +
          std::string{key} + "')");
      }
    }
    if (amp == std::string::npos) { break; }
    b = amp + 1;
  }
}

}  // namespace

sigv4_authorizer_base::sigv4_authorizer_base(static_credentials creds,
                                             std::string region,
                                             std::string endpoint)
  : _creds(std::move(creds)), _region(std::move(region))
{
  if (_creds.access_key_id.empty() || _creds.secret_access_key.empty()) {
    throw credential_error(
      "sigv4_authorizer: access_key_id and secret_access_key must be non-empty");
  }
  if (_region.empty()) { throw credential_error("sigv4_authorizer: region must be non-empty"); }

  auto [scheme, host] = parse_endpoint(endpoint);
  _scheme             = std::move(scheme);
  _host               = std::move(host);
}

sigv4_presigned_authorizer::sigv4_presigned_authorizer(static_credentials creds,
                                                       std::string region,
                                                       std::string endpoint,
                                                       std::chrono::seconds default_ttl)
  : sigv4_authorizer_base(std::move(creds), std::move(region), std::move(endpoint)),
    _ttl(default_ttl)
{
  if (_ttl.count() <= 0) {
    throw credential_error("sigv4_presigned_authorizer: default_ttl must be positive");
  }
}

authorized_request sigv4_presigned_authorizer::authorize(object_ref const& obj,
                                                         request_method method,
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
    return authorized_request{presign_url(method_to_str(method),
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
    throw credential_error(std::string("sigv4_presigned_authorizer: ") + e.what());
  }
}

authorized_request sigv4_presigned_authorizer::authorize_list(std::string_view bucket,
                                                              std::string_view canonical_query,
                                                              std::chrono::seconds timeout)
{
  if (bucket.empty()) { throw credential_error("sigv4_authorizer: empty bucket"); }
  reject_amz_query_params(canonical_query);
  std::string const canonical_uri = "/" + uri_encode(bucket, /*encode_slash=*/true);
  auto const signer               = make_signer(_creds, _region);
  auto const effective_ttl        = timeout.count() > 0 ? timeout : _ttl;

  try {
    // The list params are merged into the signed canonical query by presign_url,
    // so the URL carries both them and the X-Amz-* auth params; headers empty.
    return authorized_request{presign_url(/*method=*/"GET",
                                          _scheme,
                                          _host,
                                          canonical_uri,
                                          signer,
                                          std::time(nullptr),
                                          effective_ttl,
                                          canonical_query),
                              {}};
  } catch (credential_error const&) {
    throw;
  } catch (std::exception const& e) {
    throw credential_error(std::string("sigv4_presigned_authorizer: ") + e.what());
  }
}

sigv4_header_authorizer::sigv4_header_authorizer(static_credentials creds,
                                                 std::string region,
                                                 std::string endpoint)
  : sigv4_authorizer_base(std::move(creds), std::move(region), std::move(endpoint))
{
}

authorized_request sigv4_header_authorizer::authorize(object_ref const& obj,
                                                      request_method method,
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
    return authorized_request{std::move(url), std::move(signed_req.headers)};
  } catch (credential_error const&) {
    throw;
  } catch (std::exception const& e) {
    throw credential_error(std::string("sigv4_header_authorizer: ") + e.what());
  }
}

authorized_request sigv4_header_authorizer::authorize_list(std::string_view bucket,
                                                           std::string_view canonical_query,
                                                           std::chrono::seconds /*timeout*/)
{
  if (bucket.empty()) { throw credential_error("sigv4_authorizer: empty bucket"); }
  reject_amz_query_params(canonical_query);
  std::string const canonical_uri = "/" + uri_encode(bucket, /*encode_slash=*/true);
  auto const signer               = make_signer(_creds, _region);

  try {
    // sign_request already signs an arbitrary pre-sorted canonical query; the
    // URL carries the same query verbatim, auth rides in the returned headers.
    auto signed_req = sign_request(/*method=*/"GET",
                                   _host,
                                   canonical_uri,
                                   canonical_query,
                                   sha256_hex(""),
                                   /*extra_headers=*/{},
                                   signer,
                                   std::time(nullptr));
    std::string url = _scheme + "://" + _host + canonical_uri;
    if (!canonical_query.empty()) {
      url += '?';
      url += canonical_query;
    }
    return authorized_request{std::move(url), std::move(signed_req.headers)};
  } catch (credential_error const&) {
    throw;
  } catch (std::exception const& e) {
    throw credential_error(std::string("sigv4_header_authorizer: ") + e.what());
  }
}

}  // namespace sirius::io::rest::s3
