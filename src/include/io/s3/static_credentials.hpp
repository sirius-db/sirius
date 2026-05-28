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
#include <optional>
#include <string>

namespace sirius::io::s3 {

/**
 * @brief Credentials snapshot consumed by @c sirius_sigv4_credential_provider's
 *        constructor (and other static-creds-based providers).
 *
 * Long-lived creds (typical SET s3_access_key / s3_secret_key flow): leave
 * @c session_token empty and @c expires_at @c nullopt.
 *
 * Temporary creds (STS AssumeRole / IMDS / SSO): populate @c session_token;
 * set @c expires_at when the source can report it. The static provider treats
 * @c expires_at as informational; future refresh-aware providers (downstream)
 * may use it to schedule re-acquisition.
 *
 * This is a pure POD by design: integration code constructs it from
 * @c object_store_config string fields, hands it to the provider's
 * constructor, and the provider holds an internal copy. There is no public
 * accessor on @c credential_provider to read these back — the seam is
 * presigned-URL-only (see @c credential_provider.hpp).
 */
struct static_credentials {
  std::string access_key_id;
  std::string secret_access_key;
  std::string session_token;
  std::optional<std::chrono::system_clock::time_point> expires_at;
};

}  // namespace sirius::io::s3
