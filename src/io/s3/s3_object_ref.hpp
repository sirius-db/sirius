/*
 * Copyright 2026, Sirius Contributors.
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

#include <string>

namespace sirius::io::s3 {

/**
 * @brief Object reference passed to the credential / authorizer seam.
 *
 * @c bucket carries the object-store bucket name (no scheme, no trailing
 * slashes). @c key carries the RAW, literal object key bytes (a `%`, `?` or
 * `#` in the key is a literal byte, not an escape) — the provider / authorizer
 * RFC3986-encodes it once for canonical URI construction, so a literal `%`
 * becomes `%25` on the wire.
 *
 * Used by @c s3_request_authorizer (the presigned/header signing seam) as the
 * object-identity type.
 */
struct s3_object_ref {
  std::string bucket;
  std::string key;
};

}  // namespace sirius::io::s3
