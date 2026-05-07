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

#include "io/s3/credential_provider.hpp"
#include "io/s3/static_credentials.hpp"

#include <chrono>
#include <string>

namespace sirius::io::s3 {

/**
 * @brief Default @c credential_provider implementation: hand-rolled SigV4
 *        over static credentials.
 *
 * Wraps the @c sirius::io::s3::sigv4 module to produce presigned URLs without
 * pulling in @c aws-sdk-cpp. Suitable for:
 *   - production with long-lived access keys (typical SET s3_access_key /
 *     s3_secret_key flow),
 *   - production with externally-rotated temporary credentials (caller
 *     reconstructs the provider on rotation; this impl does not refresh).
 *
 * Downstream projects that want refresh-aware credentials (IMDS / STS chain /
 * SSO) should ship their own @c credential_provider implementation; the
 * public surface is presign-only so they can do so without exposing raw keys
 * to Sirius.
 */
class sirius_sigv4_credential_provider final : public credential_provider {
 public:
  /**
   * @brief Construct with static credentials, signing scope, endpoint, and
   *        default TTL.
   *
   * @param creds        Access key, secret key, optional session token,
   *                     optional @c expires_at (informational only — this
   *                     impl does not refresh).
   * @param region       Signing region (e.g. @c "us-east-1").
   * @param endpoint     Fully-qualified service URL:
   *                     @c "https://s3.us-east-1.amazonaws.com",
   *                     @c "http://minio.local:9000", etc. Must be
   *                     scheme://host[:port] with no path / query / fragment.
   * @param default_ttl  @c X-Amz-Expires for every presigned URL produced.
   *                     Default 5 minutes — inline-at-request-time presigning
   *                     means URLs are issued just before the libcurl call,
   *                     so a short TTL is safe and limits the blast radius
   *                     of an accidentally-leaked URL.
   *
   * @throw sirius::io::credential_error on empty creds, empty region,
   *                                     non-positive @p default_ttl, or
   *                                     malformed endpoint.
   */
  sirius_sigv4_credential_provider(static_credentials creds,
                                   std::string region,
                                   std::string endpoint,
                                   std::chrono::seconds default_ttl = std::chrono::minutes{5});

  /**
   * @brief Generate a SigV4 presigned URL for path-style S3 access:
   *        @c "{scheme}://{host}/{bucket}/{key}?X-Amz-...".
   *
   * Thread-safe: the underlying @c sigv4::presign_url is pure (no shared
   * mutable state) and this provider's members are immutable after
   * construction.
   *
   * @throw sirius::io::credential_error on empty bucket / key, or any failure
   *                                     surfaced from the SigV4 layer.
   */
  std::string get_presigned_url(s3_object_ref const& obj, presign_method method) override;

 private:
  static_credentials _creds;
  std::string _region;
  std::string _scheme;  // "https" / "http"
  std::string _host;    // host[:port]
  std::chrono::seconds _ttl;
};

}  // namespace sirius::io::s3
