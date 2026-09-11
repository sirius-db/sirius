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

#include <stdexcept>

namespace sirius::io {

/**
 * @brief Raised by @c request_authorizer implementations when credential
 *        acquisition or signing fails.
 *
 * Surfaces from:
 *   - missing or malformed static credentials (caught at provider construction
 *     or first call),
 *   - misconfigured endpoint / region,
 *   - underlying signing-library failure (HMAC / SHA256),
 *   - upstream credential broker errors (in future refresh-aware impls).
 *
 * Backends translate this into the broader IO error path of the caller.
 * Future IO-layer error types share this header.
 */
class credential_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace sirius::io
