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

#include <cstdint>
#include <string>

namespace sirius::io::s3 {

/// Method-specific presigned URL request. AWS presigned URLs are bound to
/// the HTTP method specified at signing time (presigned-GET != presigned-HEAD)
/// — passing the wrong method to the underlying HTTP client results in a
/// SignatureDoesNotMatch error from S3. Sirius needs only read-only S3
/// operations; PUT / DELETE etc. are intentionally absent.
enum class presign_method : std::uint8_t { GET, HEAD };

/**
 * @brief Object reference passed to @c credential_provider::get_presigned_url.
 *
 * @c bucket carries the object-store bucket name (no scheme, no trailing
 * slashes). @c key carries the object key, RFC3986-decoded — the provider
 * re-encodes for canonical URI construction.
 */
struct s3_object_ref {
  std::string bucket;
  std::string key;
};

/**
 * @brief Pluggable source of presigned object URLs (credential / signer seam).
 *
 * Lets downstream projects plug in their own credential / signer
 * implementation (AWS SDK presigner, internal auth broker, IMDS-backed STS
 * chain, SSO, ...) without forcing Sirius to depend on @c aws-sdk-cpp.
 * Sirius ships @c sirius_sigv4_credential_provider as the default impl using
 * the existing hand-rolled SigV4 over @c static_credentials.
 *
 * The public surface is intentionally presign-only — there is no
 * @c get_credentials() method. Implementations that lack raw key material
 * (signed-URL services, broker-issued URLs) compose cleanly.
 *
 * @par Lifetime
 *   Implementations should be safe to share across threads via @c shared_ptr.
 *   Backends call @c get_presigned_url() once per request, inline at the
 *   call site that issues the underlying HTTP request. Never call at
 *   scan-task creation time — presigned URLs carry an expiration and may
 *   become invalid before the deferred task runs.
 *
 * @par Errors
 *   Implementations throw @c sirius::io::credential_error on credential /
 *   signing failure. Backends translate into the broader IO error path.
 */
class credential_provider {
 public:
  virtual ~credential_provider() = default;

  credential_provider()                                      = default;
  credential_provider(credential_provider const&)            = delete;
  credential_provider& operator=(credential_provider const&) = delete;

  /**
   * @brief Generate a presigned URL for the given object + HTTP method.
   *
   * The returned URL is fully qualified (@c "scheme://host/canonical_uri?...").
   * Caller appends a @c Range header on the actual HTTP request when needed
   * — the presigned URL signs only the @c host header so range / accept /
   * etc. headers may be added unsigned without invalidating the signature.
   *
   * @throw sirius::io::credential_error on credential / signing failure.
   */
  [[nodiscard]] virtual std::string get_presigned_url(s3_object_ref const& obj,
                                                      presign_method method) = 0;
};

}  // namespace sirius::io::s3
