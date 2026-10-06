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

#pragma once

#include <chrono>
#include <ctime>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::io::rest::s3 {

/// Static credentials + region/service for SigV4 signing.
///
/// @c session_token is non-empty for temporary credentials (STS / IMDS / SSO);
/// when non-empty, @c sign_request injects it as the @c x-amz-security-token
/// header and @c presign_url injects it as the @c X-Amz-Security-Token query
/// parameter. The token participates in the signature in both cases.
struct sigv4_signer_config {
  std::string access_key;
  std::string secret_key;
  std::string region;
  std::string service = "s3";
  std::string session_token;
};

/// Headers to attach to an HTTP request, in the order caller should emit them.
/// Caller adds every entry verbatim (Authorization, x-amz-date, etc.).
struct sigv4_signed_request {
  std::vector<std::pair<std::string, std::string>> headers;
};

/**
 * @brief Sign an HTTP request using AWS Signature Version 4.
 *
 * Computes the canonical request / string-to-sign / signing key chain per
 * https://docs.aws.amazon.com/general/latest/gr/sigv4_signing.html and returns
 * the full set of headers the caller must attach.
 *
 * @param method             HTTP method (`"GET"`, `"HEAD"`).
 * @param host               Host header value (e.g. `"s3.us-west-2.amazonaws.com"`
 *                           or `"minio.local:9000"`).
 * @param canonical_uri      URI path, already RFC3986-encoded (e.g. `"/bucket/key"`).
 * @param canonical_query    Canonical query string, already encoded & sorted; empty if none.
 * @param payload_sha256_hex Hex SHA256 of the request body. For GET/HEAD with
 *                           no body, pass @c sha256_hex("").
 * @param extra_headers      Additional headers to sign (e.g. `{"range", "bytes=0-99"}`).
 *                           Keys lowercase; do NOT include `host`, `x-amz-date`,
 *                           or `x-amz-content-sha256` here — they are added
 *                           automatically.
 * @param creds              Access key, secret key, region, service.
 * @param timestamp_utc      Seconds since epoch; sign produces `x-amz-date`
 *                           and the scope's date from this. Passed explicitly
 *                           so tests are deterministic.
 *
 * @throw std::invalid_argument on empty credentials.
 */
sigv4_signed_request sign_request(
  std::string_view method,
  std::string_view host,
  std::string_view canonical_uri,
  std::string_view canonical_query,
  std::string_view payload_sha256_hex,
  std::vector<std::pair<std::string, std::string>> const& extra_headers,
  sigv4_signer_config const& creds,
  std::time_t timestamp_utc);

/**
 * @brief Build a SigV4 presigned URL.
 *
 * Authentication parameters go in the query string (X-Amz-Algorithm,
 * X-Amz-Credential, X-Amz-Date, X-Amz-Expires, X-Amz-SignedHeaders, optional
 * X-Amz-Security-Token, X-Amz-Signature). Returns the fully-qualified URL.
 *
 * Unlike @c sign_request, this path uses @c "UNSIGNED-PAYLOAD" as the canonical
 * payload hash (the body of the actual request is not part of the presigned
 * URL's signature; only @c host is signed) so callers may freely add @c Range
 * or other unsigned headers to the resulting GET / HEAD without breaking the
 * signature.
 *
 * @param method        HTTP method (@c "GET" / @c "HEAD" — the only methods
 *                      Sirius uses for read-only S3).
 * @param scheme        URL scheme (@c "http" / @c "https"), already lowercase.
 * @param host          Host[:port], already lowercase.
 * @param canonical_uri URI path, already RFC3986-encoded
 *                      (e.g. @c "/bucket/key" or @c "/bucket/path%20space.parquet").
 * @param creds         Access key, secret key, region, service, optional
 *                      @c session_token.
 * @param timestamp_utc Seconds since epoch; @c X-Amz-Date is derived from this.
 *                      Passed explicitly so tests are deterministic.
 * @param ttl           Validity window. Becomes the @c X-Amz-Expires query
 *                      parameter (in seconds).
 * @param extra_canonical_query  Additional request query parameters, already
 *                      RFC3986-encoded, `&`-joined, and in SigV4 canonical
 *                      (byte-sorted) order (e.g.
 *                      @c "list-type=2&max-keys=1000&prefix=p%2F" for a
 *                      ListObjectsV2 request). Empty for plain object GET/HEAD.
 *                      These are merged with the @c X-Amz-* parameters and the
 *                      whole set is re-sorted before signing, so they
 *                      participate in the signature (required for S3 to accept
 *                      a presigned LIST). Each element is taken verbatim — pass
 *                      it pre-encoded, not raw.
 *
 * @throw std::invalid_argument on empty credentials / region / service /
 *                              method / scheme / host, or non-positive @p ttl.
 */
std::string presign_url(std::string_view method,
                        std::string_view scheme,
                        std::string_view host,
                        std::string_view canonical_uri,
                        sigv4_signer_config const& creds,
                        std::time_t timestamp_utc,
                        std::chrono::seconds ttl,
                        std::string_view extra_canonical_query = {});

/// Hex-encoded SHA256 digest of @p data. Thin wrapper around OpenSSL SHA256.
std::string sha256_hex(std::string_view data);

/// RFC 3986 URI-encode @p s. If @p encode_slash is false, `/` passes through
/// (used for path segments); if true, `/` becomes `%2F` (used for query components).
std::string uri_encode(std::string_view s, bool encode_slash);

}  // namespace sirius::io::rest::s3
