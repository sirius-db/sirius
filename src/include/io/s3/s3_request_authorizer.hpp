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

#include "io/s3/s3_object_ref.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sirius::io::s3 {

/// Method-specific S3 request. AWS request authorization is bound to the HTTP
/// method (a presigned-GET URL != a presigned-HEAD URL; a SigV4 header
/// signature also covers the method) — passing the wrong method to the
/// underlying HTTP client results in a SignatureDoesNotMatch error from S3.
/// Sirius needs only read-only S3 operations; PUT / DELETE etc. are
/// intentionally absent.
enum class s3_request_method : std::uint8_t { GET, HEAD };

/// Result of authorizing one S3 request: the URL to fetch plus headers to
/// attach verbatim. Presigned authorizers put auth in the URL query and
/// return empty @c headers; header-signing authorizers return a plain URL
/// plus Authorization / x-amz-* headers.
struct s3_authorized_request {
  std::string url;
  std::vector<std::pair<std::string, std::string>> headers;
};

/**
 * @brief Pluggable S3 request authorizer (credential / signer seam).
 *
 * Lets downstream projects plug in their own credential / signer
 * implementation (AWS SDK presigner, internal auth broker, IMDS-backed STS
 * chain, SSO, ...) without forcing Sirius to depend on @c aws-sdk-cpp.
 * Sirius ships @c sirius_sigv4_presigned_authorizer as the default impl using
 * the existing hand-rolled SigV4 over @c static_credentials.
 *
 * The public surface is intentionally a single @c authorize() call — there is
 * no @c get_credentials() method. Implementations that lack raw key material
 * (signed-URL services, broker-issued URLs) compose cleanly. @c authorize
 * returns the URL to fetch plus the headers to attach: presigned authorizers
 * return a query-signed URL with empty headers, while header-signing
 * authorizers return a plain URL plus the signed Authorization / x-amz-*
 * headers.
 *
 * @par Lifetime
 *   Implementations should be safe to share across threads via @c shared_ptr.
 *   Backends call @c authorize() once per request, inline at the call site
 *   that issues the underlying HTTP request. Never call at scan-task creation
 *   time — presigned URLs carry an expiration and may become invalid before
 *   the deferred task runs.
 *
 * @par Errors
 *   Implementations throw @c sirius::io::credential_error on credential /
 *   signing failure. Backends translate into the broader IO error path.
 */
class s3_request_authorizer {
 public:
  virtual ~s3_request_authorizer() = default;

  s3_request_authorizer()                                        = default;
  s3_request_authorizer(s3_request_authorizer const&)            = delete;
  s3_request_authorizer& operator=(s3_request_authorizer const&) = delete;

  /**
   * @brief Authorize an S3 request for the given object + HTTP method.
   *
   * Returns the URL to fetch plus the headers to attach verbatim. A presigned
   * authorizer returns a fully-qualified, query-signed URL
   * (@c "scheme://host/canonical_uri?...") and empty headers — the caller may
   * append a @c Range header on the actual HTTP request without invalidating
   * the signature (the presigned URL signs only the @c host header). A
   * header-signing authorizer returns a plain URL plus the signed
   * Authorization / x-amz-* headers that must be attached to the request.
   *
   * @param timeout  Per-call URL lifetime (@c X-Amz-Expires). The IO layer sizes
   *                  it to cover a single request attempt (not the whole
   *                  scan/task) — URLs are minted inline per request, so a short
   *                  TTL is safe. Implementations may treat a non-positive value
   *                  as "use an implementation default".
   * @throw sirius::io::credential_error on credential / signing failure.
   */
  [[nodiscard]] virtual s3_authorized_request authorize(s3_object_ref const& obj,
                                                        s3_request_method method,
                                                        std::chrono::seconds timeout) = 0;
};

}  // namespace sirius::io::s3
