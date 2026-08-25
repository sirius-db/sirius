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

#include "io/rest/authorizer.hpp"
#include "io/rest/s3/static_credentials.hpp"

#include <chrono>
#include <string>

namespace sirius::io::rest::s3 {

/**
 * @brief Common base for the built-in SigV4 authorizers: hand-rolled SigV4 over
 *        static credentials, no @c aws-sdk-cpp.
 *
 * Holds the credentials + signing scope and parses/validates the endpoint into
 * scheme + host. Concrete subclasses choose the signing form (presigned URL vs
 * Authorization header). Suitable for long-lived access keys and
 * externally-rotated temporary credentials (reconstruct the authorizer on
 * rotation; these impls do not refresh). Downstream projects that want
 * refresh-aware credentials (IMDS / STS chain / SSO) ship their own
 * @c request_authorizer; the public surface is a single @c authorize() call
 * so they can do so without exposing raw keys to Sirius.
 */
class sigv4_authorizer_base : public request_authorizer {
 protected:
  /// @param region    Signing region (e.g. @c "us-east-1").
  /// @param endpoint  scheme://host[:port], no path / query / fragment.
  /// @throw sirius::io::credential_error on empty creds, empty region, or
  ///        malformed endpoint.
  sigv4_authorizer_base(static_credentials creds, std::string region, std::string endpoint);

  static_credentials _creds;
  std::string _region;
  std::string _scheme;  // "https" / "http"
  std::string _host;    // host[:port]
};

/**
 * @brief Default authorizer: SigV4 presigned URL for path-style S3 access
 *        (@c "{scheme}://{host}/{bucket}/{key}?X-Amz-...").
 *
 * @c authorize() returns the presigned URL with EMPTY headers (auth lives in the
 * URL query). @p default_ttl sets @c X-Amz-Expires; a positive per-call timeout
 * overrides it. Inline-at-request-time presigning keeps TTLs short, limiting the
 * blast radius of an accidentally-leaked URL.
 */
class sigv4_presigned_authorizer final : public sigv4_authorizer_base {
 public:
  /// @throw sirius::io::credential_error on empty creds/region, malformed
  ///        endpoint, or non-positive @p default_ttl.
  sigv4_presigned_authorizer(static_credentials creds,
                             std::string region,
                             std::string endpoint,
                             std::chrono::seconds default_ttl = std::chrono::minutes{5});

  /// Thread-safe: @c sigv4::presign_url is pure and members are immutable after
  /// construction.
  /// @throw sirius::io::credential_error on empty bucket / key or SigV4 failure.
  authorized_request authorize(object_ref const& obj,
                               request_method method,
                               std::chrono::seconds timeout) override;

  /// Presigned bucket-level ListObjectsV2: the request params are merged into
  /// the signed query, so the returned URL carries both the list params and the
  /// X-Amz-* auth params; headers are empty.
  /// @throw sirius::io::credential_error on empty bucket, an X-Amz-* key inside
  ///        @p canonical_query (signing-param smuggling), or SigV4 failure.
  authorized_request authorize_list(std::string_view bucket,
                                    std::string_view canonical_query,
                                    std::chrono::seconds timeout) override;

 private:
  std::chrono::seconds _ttl;
};

/**
 * @brief Header-signing authorizer: SigV4 in the @c Authorization header
 *        (@c sigv4::sign_request).
 *
 * @c authorize() returns a plain path-style URL (@c "{scheme}://{host}/{bucket}/
 * {key}", no query auth) plus @c Authorization, @c x-amz-date,
 * @c x-amz-content-sha256, and @c x-amz-security-token (only for temporary
 * credentials). For on-prem / S3-compatible stores whose gateways prefer header
 * auth over presigned query strings. @c timeout is unused (header auth carries
 * no explicit expiry).
 */
class sigv4_header_authorizer final : public sigv4_authorizer_base {
 public:
  /// @throw sirius::io::credential_error on empty creds/region or malformed
  ///        endpoint.
  sigv4_header_authorizer(static_credentials creds, std::string region, std::string endpoint);

  /// Thread-safe: @c sigv4::sign_request is pure and members are immutable after
  /// construction.
  /// @throw sirius::io::credential_error on empty bucket / key or SigV4 failure.
  authorized_request authorize(object_ref const& obj,
                               request_method method,
                               std::chrono::seconds timeout) override;

  /// Header-signed bucket-level ListObjectsV2: returns a plain
  /// @c "{scheme}://{host}/{bucket}?{canonical_query}" URL plus the signed
  /// Authorization / x-amz-* headers. @c timeout is unused (header auth carries
  /// no explicit expiry).
  /// @throw sirius::io::credential_error on empty bucket, an X-Amz-* key inside
  ///        @p canonical_query (signing-param smuggling), or SigV4 failure.
  authorized_request authorize_list(std::string_view bucket,
                                    std::string_view canonical_query,
                                    std::chrono::seconds timeout) override;
};

}  // namespace sirius::io::rest::s3
